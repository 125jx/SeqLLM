# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/trainer_seq2seq.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
from types import MethodType
from typing import TYPE_CHECKING, Any, Optional, Union, List

import numpy as np
import torch
from transformers import Seq2SeqTrainer
from typing_extensions import override

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from ...extras.packages import is_transformers_version_greater_than
from ..callbacks import SaveProcessorCallback
from ..fp8_utils import configure_fp8_environment, verify_fp8_status
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler


if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from transformers import PreTrainedTokenizer, ProcessorMixin
    from transformers.trainer import PredictionOutput

    from ...hparams import FinetuningArguments, ModelArguments


logger = logging.get_logger(__name__)


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    r"""Inherits Seq2SeqTrainer to compute generative metrics such as BLEU and ROUGE."""

    def __init__(
        self,
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        model_args: Optional["ModelArguments"] = None,
        gen_kwargs: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        # Configure FP8 environment if enabled
        if model_args is not None and model_args.fp8:
            configure_fp8_environment(model_args)
        if is_transformers_version_greater_than("4.46"):
            kwargs["processing_class"] = kwargs.pop("tokenizer")
        else:
            self.processing_class: PreTrainedTokenizer = kwargs.get("tokenizer")

        super().__init__(**kwargs)
        if processor is not None:
            # avoid wrong loss under gradient accumulation
            # https://github.com/huggingface/transformers/pull/36044#issuecomment-2746657112
            self.model_accepts_loss_kwargs = False

        self.finetuning_args = finetuning_args
        if gen_kwargs is not None:
            # https://github.com/huggingface/transformers/blob/v4.45.0/src/transformers/trainer_seq2seq.py#L287
            self._gen_kwargs = gen_kwargs

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

        if finetuning_args.use_dft_loss:
            from ..trainer_utils import dft_loss_func

            self.compute_loss_func = dft_loss_func

        # Initialize SID legal-candidate constraint settings (independent of
        # behavior-only loss below; this is the masked-CE-over-legal-SID-sets
        # scheme). Tables are built offline by scripts/build_sid_constraints.py.
        self._sid_constraint_enabled = bool(getattr(finetuning_args, "sid_constraint", False))
        self._sid_constraint_path = getattr(finetuning_args, "sid_constraint_path", None)
        self._sid_constraint_s3_mode = str(getattr(finetuning_args, "sid_constraint_s3_mode", "valid_s3"))
        self._sid_masker = None  # lazy-built on first compute_loss (needs device)
        if self._sid_constraint_enabled:
            if not self._sid_constraint_path:
                raise ValueError(
                    "sid_constraint=True requires sid_constraint_path to point at the .npz "
                    "built by scripts/build_sid_constraints.py."
                )
            logger.info_rank0(
                f"SID constraint ENABLED: path={self._sid_constraint_path!r}, "
                f"s3_mode={self._sid_constraint_s3_mode!r}. Loss uses masked CE over legal "
                f"SID candidate sets (s1->valid_s1, s2->a2b[s1], s3->ab2c/valid_s3)."
            )

        # Initialize behavior-only loss settings
        self._beh_only_loss_enabled = getattr(finetuning_args, "beh_only_loss", False)
        self._beh_only_loss_start_id = getattr(finetuning_args, "beh_only_loss_start_id", None)
        self._beh_only_loss_end_id = getattr(finetuning_args, "beh_only_loss_end_id", None)
        if self._beh_only_loss_enabled:
            logger.info_rank0(
                f"Behavior-only loss enabled: will only compute loss on token positions where label is in "
                f"[{self._beh_only_loss_start_id}, {self._beh_only_loss_end_id})"
            )

        # Ensure left-padding is used during generation-based evaluation.
        self._eval_padding_side_fixed = False

        # Verify FP8 status after trainer initialization (accelerator should be available)
        if model_args is not None and model_args.fp8 and hasattr(self, "accelerator"):
            verify_fp8_status(self.accelerator, model_args)

    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    @override
    def _get_train_sampler(self, *args, **kwargs) -> Optional["torch.utils.data.Sampler"]:
        if self.finetuning_args.disable_shuffling:
            return torch.utils.data.SequentialSampler(self.train_dataset)

        return super()._get_train_sampler(*args, **kwargs)

    @override
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """LF-DIAG-wrapped compute_loss; delegates to the original logic and
        emits per-micro-batch raw-loss / label-validity diagnostics on the way out.
        """
        out = self._compute_loss_inner(model, inputs, return_outputs=return_outputs, **kwargs)
        self._diag_log_micro_batch_loss(out, inputs)
        return out

    def _diag_log_micro_batch_loss(self, out, inputs) -> None:
        # === LF-DIAG: per-micro-batch raw loss + label-validity probe =========
        # Default off; INFO logs gated by env LF_DIAG_FORWARD_LOSS=1.
        # ALWAYS warns on the most decisive bug signal: raw loss == 0 while
        # there ARE valid labels in the batch (rules out packing-mask cause,
        # points at model/loss-path bug). No-op on exceptions.
        try:
            diag_on = int(os.environ.get("LF_DIAG_FORWARD_LOSS", "0") or "0") > 0
            loss_t = out[0] if isinstance(out, tuple) else out
            if not torch.is_tensor(loss_t):
                return
            loss_val = float(loss_t.detach().item())
            finite = bool(torch.isfinite(loss_t.detach()).item())
            labels = inputs.get("labels", None)
            if labels is not None and torch.is_tensor(labels):
                total = int(labels.numel())
                valid = int((labels != IGNORE_INDEX).sum().item())
            else:
                total, valid = -1, -1

            self._diag_mb_idx = getattr(self, "_diag_mb_idx", 0) + 1

            if diag_on:
                logger.info_rank0(
                    f"[LF-DIAG][forward] mb={self._diag_mb_idx} "
                    f"raw_loss={loss_val:.6f} finite={finite} "
                    f"labels={valid}/{total}"
                )

            if finite and loss_val == 0.0 and valid > 0:
                logger.warning_rank0(
                    f"[LF-DIAG][forward] !! RAW LOSS == 0 BUT valid_labels={valid} !! "
                    f"mb={self._diag_mb_idx} total={total} -- forward returned exact 0 "
                    f"despite non-IGNORE labels. Indicates a model/loss-path bug "
                    f"(NOT a packing-mask cause)."
                )
            if not finite:
                logger.warning_rank0(
                    f"[LF-DIAG][forward] !! NON-FINITE RAW LOSS !! "
                    f"mb={self._diag_mb_idx} loss={loss_val} valid={valid}/{total} -- "
                    f"forward produced NaN/Inf; downstream grad_norm will be unreliable."
                )
        except Exception as _e:  # noqa: BLE001
            logger.warning_rank0(f"[LF-DIAG][forward] diag failed: {_e}")
        # === end LF-DIAG ======================================================

    def _get_sid_masker(self, device: "torch.device"):
        """Lazy-build the SID constraint masker on the given device.

        Returns ``None`` when SID constraints are disabled.
        """
        if not self._sid_constraint_enabled:
            return None
        if self._sid_masker is None:
            from ..sid_constraint import SidConstraintMasker

            self._sid_masker = SidConstraintMasker(
                path=self._sid_constraint_path,
                s3_mode=self._sid_constraint_s3_mode,
                ignore_index=IGNORE_INDEX,
            )
            logger.info_rank0(
                f"SID constraint tables loaded from {self._sid_constraint_path!r} "
                f"(s3_mode={self._sid_constraint_s3_mode}, slot_size={self._sid_masker.slot_size}, "
                f"a_base={self._sid_masker.a_base}, b_base={self._sid_masker.b_base}, "
                f"c_base={self._sid_masker.c_base})."
            )
        self._sid_masker.to(device)
        return self._sid_masker

    def _compute_loss_sid_constraint(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Masked cross-entropy over legal SID candidate sets.

        Standard HF next-token shift: ``shift_logits[p]`` predicts
        ``shift_labels[p] = labels[p+1]``. The token actually fed at the same
        position is ``input_ids[p]`` and the one before it is ``input_ids[p-1]``,
        which is exactly what the masker needs to recover the (s1, s2) prefix of
        an s2 / s3 slot. The model forward is unchanged; only the loss differs.

        Loss-scaling parity with HF ``ForCausalLMLoss`` (CRITICAL for correct
        gradients under gradient accumulation): when transformers passes
        ``num_items_in_batch`` (the global #valid-label tokens across the whole
        grad-accum window on this rank) and ``model_accepts_loss_kwargs=True``,
        HF does NOT divide the returned loss by ``gradient_accumulation_steps``;
        it expects a SUM loss normalized by ``num_items_in_batch`` so the
        per-micro-batch losses add up to the correct global token-mean. We
        therefore return ``per_token.sum() / num_items_in_batch`` here (NOT a
        per-micro-batch mean, which would inflate the effective gradient by
        ~grad_accum). Falls back to the per-token mean when
        ``num_items_in_batch`` is unavailable (older HF / no loss kwargs path,
        where HF divides by grad_accum itself).
        """
        labels = inputs["labels"]
        input_ids = inputs["input_ids"]
        outputs = model(**{k: v for k, v in inputs.items() if k != "labels"}, return_dict=True)
        logits = outputs["logits"]

        masker = self._get_sid_masker(logits.device)

        shift_logits = logits[:, :-1, :].contiguous()  # [B, T-1, V]
        shift_labels = labels[:, 1:].contiguous()  # [B, T-1]
        cur = input_ids[:, :-1].contiguous()  # input_ids[p], p = 0..T-2
        # prev[:, p] = input_ids[:, p-1]; prepend a dummy 0 for p == 0 (never an s3 slot).
        prev = torch.cat([input_ids[:, :1].new_zeros((input_ids.shape[0], 1)), input_ids[:, :-2]], dim=1)

        V = shift_logits.size(-1)
        avg, per_token = masker.masked_token_loss(
            shift_logits.reshape(-1, V),
            shift_labels.reshape(-1),
            cur.reshape(-1),
            prev.reshape(-1),
        )

        total_valid = (shift_labels.reshape(-1) != IGNORE_INDEX).sum()
        if total_valid.item() == 0:
            # No valid tokens this shard: `avg` is a grad-connected 0 (keeps
            # DDP/DeepSpeed backward collective-consistent).
            loss = avg
        elif num_items_in_batch is not None:
            # HF-parity sum loss normalized by the global valid-token count.
            loss = per_token.sum() / num_items_in_batch
        else:
            # Per-token mean (HF will divide by grad_accum itself in this path).
            loss = per_token.sum() / total_valid

        return (loss, outputs) if return_outputs else loss

    def _compute_loss_user_balanced(self, model, inputs, return_outputs=False):
        """Per-user balanced cross-entropy.

        Each sample carries a scalar ``loss_weight`` w_b (built by the data pipeline as
        ``scale / n_u`` for the targeted datasets, ``1.0`` otherwise). We broadcast w_b over that
        sample's valid (non-IGNORE) label positions and return the weighted token-mean

            loss = sum_{b,t valid} (w_b * ce_{b,t}) / sum_{b,t valid} w_b

        Properties:
          * a sample-uniform weight (all w_b == 1, e.g. a pure non-targeted batch) reduces EXACTLY
            to the standard token-mean loss;
          * within a targeted dataset, user u contributes total weight n_u * (scale / n_u) = scale,
            i.e. every user is weighted equally regardless of how many samples / tokens they have.

        We force ``model_accepts_loss_kwargs = False`` (idempotent) so that
        ``Trainer.training_step`` normalizes this micro-batch mean by
        ``gradient_accumulation_steps`` (transformers 4.57 behavior); we do NOT use
        ``num_items_in_batch`` here.
        """
        if not getattr(self, "_ub_loss_kwargs_off", False):
            self.model_accepts_loss_kwargs = False
            self._ub_loss_kwargs_off = True
            logger.info_rank0(
                "[user_balanced_loss] enabled: per-token loss re-weighted by per-sample "
                "`loss_weight`; model_accepts_loss_kwargs set to False so HF normalizes by "
                "gradient_accumulation_steps."
            )

        loss_weight = inputs.pop("loss_weight")  # [B], float
        labels = inputs["labels"]
        model_inputs = {k: v for k, v in inputs.items() if k != "labels"}
        outputs = model(**model_inputs, return_dict=True)
        logits = outputs["logits"]

        # Standard next-token shift: logits[:, :-1] predicts labels[:, 1:].
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        bsz, seq_len_m1, vocab = shift_logits.shape

        per_token = torch.nn.functional.cross_entropy(
            shift_logits.reshape(-1, vocab).float(),
            shift_labels.reshape(-1),
            ignore_index=IGNORE_INDEX,
            reduction="none",
        ).view(bsz, seq_len_m1)  # 0.0 on IGNORE positions

        valid = (shift_labels != IGNORE_INDEX).to(per_token.dtype)  # [B, T-1]
        w = loss_weight.to(per_token.dtype).unsqueeze(1) * valid  # [B, T-1]
        denom = w.sum().clamp_min(1e-8)
        loss = (w * per_token).sum() / denom

        return (loss, outputs) if return_outputs else loss

    def _compute_loss_inner(self, model, inputs, return_outputs=False, **kwargs):
        """
        Compute loss with optional behavior-only loss:
        - If beh_only_loss is enabled, only compute loss on positions where the label
          is in [beh_only_loss_start_id, beh_only_loss_end_id).
        - Softmax is also restricted to this range for more efficient and focused training.
        """
        # Per-user balanced loss takes precedence when the data pipeline attached a per-sample
        # `loss_weight` (only when `data_args.user_balanced_loss=True`). Re-weights the per-token
        # CE so that listed datasets count each unique user equally instead of each token.
        if "loss_weight" in inputs and inputs.get("labels") is not None:
            return self._compute_loss_user_balanced(model, inputs, return_outputs=return_outputs)

        # SID legal-candidate constraint takes precedence and is fully
        # independent of the behavior-only-loss path below.
        if self._sid_constraint_enabled and inputs.get("labels") is not None:
            return self._compute_loss_sid_constraint(
                model,
                inputs,
                return_outputs=return_outputs,
                num_items_in_batch=kwargs.get("num_items_in_batch"),
            )

        if not self._beh_only_loss_enabled:
            return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)

        # Behavior-only loss computation
        labels = inputs.get("labels")
        if labels is None:
            return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)

        # Forward pass
        outputs = model(**{k: v for k, v in inputs.items() if k != "labels"}, return_dict=True)
        logits = outputs.get("logits")
        if logits is None:
            return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)

        # Get behavior token range
        start_id = self._beh_only_loss_start_id
        end_id = self._beh_only_loss_end_id
        if start_id is None:
            return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)
        if end_id is None:
            end_id = logits.size(-1)  # default to vocab size

        # Shift for next-token prediction: logits[:-1] predicts labels[1:]
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Mask out labels that are not in behavior token range
        # Keep only positions where label is in [start_id, end_id)
        beh_mask = (shift_labels >= start_id) & (shift_labels < end_id)
        
        # Set non-behavior positions to IGNORE_INDEX (-100)
        masked_labels = shift_labels.clone()
        masked_labels[~beh_mask] = IGNORE_INDEX

        # Slice logits to only include behavior token range for softmax
        # This makes softmax more focused and efficient
        beh_logits = shift_logits[..., start_id:end_id]
        
        # Adjust labels to be relative to the sliced range
        adjusted_labels = masked_labels.clone()
        valid_mask = masked_labels != IGNORE_INDEX
        adjusted_labels[valid_mask] = adjusted_labels[valid_mask] - start_id

        # Flatten for cross entropy
        beh_logits_flat = beh_logits.view(-1, beh_logits.size(-1))
        adjusted_labels_flat = adjusted_labels.view(-1)

        # Compute cross entropy loss
        loss = torch.nn.functional.cross_entropy(
            beh_logits_flat.float(),
            adjusted_labels_flat,
            ignore_index=IGNORE_INDEX,
            reduction="mean"
        )

        if return_outputs:
            return loss, outputs
        return loss

    @override
    def prediction_step(
        self,
        model: "torch.nn.Module",
        inputs: dict[str, Union["torch.Tensor", Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
        **gen_kwargs,
    ) -> tuple[Optional[float], Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        r"""Remove the prompt part in the generated tokens.

        Subclass and override to inject custom behavior.
        """
        if self.args.predict_with_generate and not self._eval_padding_side_fixed:
            if getattr(self.processing_class, "padding_side", None) != "left":
                self.processing_class.padding_side = "left"
                if hasattr(self.processing_class, "init_kwargs"):
                    self.processing_class.init_kwargs["padding_side"] = "left"
            self._eval_padding_side_fixed = True

        if self.args.predict_with_generate:  # do not pass labels to model when generate
            labels = inputs.pop("labels", None)
        else:
            labels = inputs.get("labels")

        loss, generated_tokens, _ = super().prediction_step(
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys, **gen_kwargs
        )
        if generated_tokens is not None and self.args.predict_with_generate:
            generated_tokens[:, : inputs["input_ids"].size(-1)] = self.processing_class.pad_token_id
            generated_tokens = generated_tokens.contiguous()

        return loss, generated_tokens, labels

    @override
    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        if logs is not None:
            logs = dict(logs)
            # Make the log line self-contained: include current global step (optimizer step).
            logs.setdefault("step", int(getattr(self.state, "global_step", 0)))

        return super().log(logs, start_time=start_time)

    def save_predictions(
        self, dataset: "Dataset", predict_results: "PredictionOutput", skip_special_tokens: bool = True
    ) -> None:
        r"""Save model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info_rank0(f"Saving prediction results to {output_prediction_file}")

        labels = np.where(
            predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.processing_class.pad_token_id
        )
        preds = np.where(
            predict_results.predictions != IGNORE_INDEX,
            predict_results.predictions,
            self.processing_class.pad_token_id,
        )

        for i in range(len(preds)):
            pad_len = np.nonzero(preds[i] != self.processing_class.pad_token_id)[0]
            if len(pad_len):  # move pad token to last
                preds[i] = np.concatenate((preds[i][pad_len[0] :], preds[i][: pad_len[0]]), axis=-1)

        decoded_inputs = self.processing_class.batch_decode(dataset["input_ids"], skip_special_tokens=False)
        decoded_preds = self.processing_class.batch_decode(preds, skip_special_tokens=skip_special_tokens)
        decoded_labels = self.processing_class.batch_decode(labels, skip_special_tokens=skip_special_tokens)

        with open(output_prediction_file, "w", encoding="utf-8") as f:
            for text, pred, label in zip(decoded_inputs, decoded_preds, decoded_labels):
                f.write(json.dumps({"prompt": text, "predict": pred, "label": label}, ensure_ascii=False) + "\n")
