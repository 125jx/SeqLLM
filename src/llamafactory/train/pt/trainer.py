# Copyright 2025 the LlamaFactory team.
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

from types import MethodType
from typing import TYPE_CHECKING, Optional

import re
import torch
import torch.nn.functional as F
from transformers import Trainer
from typing_extensions import override

from ...extras.constants import IGNORE_INDEX
from ...extras.packages import is_transformers_version_greater_than
from ..callbacks import SaveProcessorCallback
from ..fp8_utils import configure_fp8_environment, verify_fp8_status
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler


if TYPE_CHECKING:
    from transformers import ProcessorMixin

    from ...hparams import FinetuningArguments, ModelArguments


class CustomTrainer(Trainer):
    r"""Inherit Trainer for custom optimizer."""

    def __init__(
        self,
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        model_args: Optional["ModelArguments"] = None,
        **kwargs,
    ) -> None:
        # Configure FP8 environment if enabled
        if model_args is not None and model_args.fp8:
            configure_fp8_environment(model_args)
        if is_transformers_version_greater_than("4.46"):
            kwargs["processing_class"] = kwargs.pop("tokenizer")

        super().__init__(**kwargs)
        if processor is not None:
            # avoid wrong loss under gradient accumulation
            # https://github.com/huggingface/transformers/pull/36044#issuecomment-2746657112
            self.model_accepts_loss_kwargs = False

        self.finetuning_args = finetuning_args
        self._train_hr_metrics: dict[str, float] = {}
        self._stage1_loss_metrics: dict[str, float] = {}
        self._hr_beh_token_ids: Optional["torch.Tensor"] = None
        self._hr_beh_token_ids_orig_size: Optional[int] = None
        self._hr_beh_token_ids_vocab_size: Optional[int] = None
        self._stage1_beh_desc_by_token_id: Optional[dict[int, str]] = None
        self._stage1_beh_token_ids_all: Optional[list[int]] = None
        self._stage1_tok_cache: dict[str, list[int]] = {}
        self._stage1_base_norm_target: Optional[float] = None
        self._stage1_last_gen_preview: list[tuple[str, str, str]] = []  # (beh_token, prompt, target_desc)

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

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
    def compute_loss(self, model, inputs, return_outputs: bool = False, **kwargs):
        """
        Keep normal training behavior, but (optionally) compute a lightweight train-batch HR@K (acc@k)
        for next-token prediction at the last valid position (next-item style).

        Metric keys are MLflow-safe: train_HR_at_1 / train_HR_at_5.
        """
        labels = inputs.get("labels", None)
        enable_train_hr = bool(getattr(self.finetuning_args, "compute_hit_rate", False)) and labels is not None

        # Always reuse logits from the same forward pass when possible.
        # We may also add Stage1 extra loss (behavior->text generation) on top of the standard LM loss.
        base_loss, outputs = super().compute_loss(model, inputs, return_outputs=True, **kwargs)
        loss = base_loss

        # Stage1 extra losses (optional)
        gen_w = float(getattr(self.finetuning_args, "stage1_gen_loss_weight", 0.0) or 0.0)
        if gen_w > 0.0:
            self._stage1_loss_metrics = {}
            gen_loss = self._compute_stage1_gen_loss(model, weight=gen_w)
            loss = loss + gen_loss

        if not enable_train_hr:
            if return_outputs:
                return loss, outputs
            return loss

        logits = getattr(outputs, "logits", None)
        if logits is not None:
            k_list = list(getattr(self.finetuning_args, "hit_rate_k_list", [1, 5]))
            if not k_list:
                k_list = [1, 5]
            self._train_hr_metrics = self._compute_train_hr_last_position(logits, labels, k_list)

        if return_outputs:
            return loss, outputs
        return loss

    def _get_tokenizer(self):
        pc = getattr(self, "processing_class", None)
        if pc is not None:
            return pc
        return getattr(self, "tokenizer", None)

    def _stage1_init_cache(self, model) -> None:
        if self._stage1_beh_desc_by_token_id is not None and self._stage1_beh_token_ids_all is not None:
            return
        csv_path = getattr(self.finetuning_args, "stage1_beh_text_csv_path", None)
        if not csv_path:
            raise ValueError("stage1_beh_text_csv_path is required when enabling Stage1 extra losses.")
        id_col = str(getattr(self.finetuning_args, "stage1_beh_text_id_col", "feature_lookup_id"))
        text_col = str(getattr(self.finetuning_args, "stage1_beh_text_col", "summary_tags"))
        tok = self._get_tokenizer()
        if tok is None:
            raise RuntimeError("Tokenizer is not available in trainer (processing_class/tokenizer missing).")

        import csv
        import os

        def infer_iceberg_vocab_clip() -> tuple[int, Optional[int], Optional[int]]:
            """
            Mirror iceberg beh-id mapping in `IcebergPartitionDataset`:
            - shift_id_by (default 0)
            - max_vocab_size: keep ids in [0, max_vocab_size), map others to oov_behavior_id (or max_vocab_size-1)
            """
            shift_id_by = 0
            max_vocab_size = None
            oov_behavior_id = None

            # Prefer train config (since Stage1 is used in training)
            cfg_path = os.getenv("ICEBERG_BEH_TEXT_TRAIN_CONFIG", None) or os.getenv("ICEBERG_BEH_TEXT_EVAL_CONFIG", None)
            if cfg_path not in [None, ""]:
                try:
                    import yaml

                    with open(str(cfg_path), "r", encoding="utf-8") as f:
                        cfg = yaml.safe_load(f) or {}
                    shift_id_by = int(cfg.get("shift_id_by", 0) or 0)
                    mv = cfg.get("max_vocab_size", None)
                    if mv is None:
                        env_mv = os.getenv("ICEBERG_BEH_TEXT_MAX_VOCAB_SIZE", None)
                        if env_mv not in [None, ""]:
                            mv = env_mv
                    if mv is not None:
                        max_vocab_size = int(mv)
                    ov = cfg.get("oov_behavior_id", None)
                    if ov is not None:
                        oov_behavior_id = int(ov)
                except Exception:
                    # Best-effort only; fall back to no clipping.
                    pass
            return shift_id_by, max_vocab_size, oov_behavior_id

        shift_id_by, max_vocab_size, oov_behavior_id = infer_iceberg_vocab_clip()

        def strip_feature_code(text: str) -> str:
            # 清理“特征码/feature_code”字段，避免公共模式干扰释义文本
            s = str(text)
            # kv_tags: ...;特征码=xxx
            s = re.sub(r"(;?\s*特征码\s*=\s*[^;｜|]+)", "", s)
            s = re.sub(r"(;?\s*feature_code\s*=\s*[^;｜|]+)", "", s, flags=re.IGNORECASE)
            # 去掉多余分隔符
            s = s.strip().strip(";").strip("｜").strip("|").strip()
            return s

        mapping: dict[int, str] = {}
        token_ids: list[int] = []
        with open(str(csv_path), "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError(f"Empty CSV header: {csv_path}")
            if id_col not in reader.fieldnames or text_col not in reader.fieldnames:
                raise ValueError(
                    f"CSV missing columns for Stage1: need {id_col}/{text_col}, got {reader.fieldnames}"
                )
            for row in reader:
                beh_id = str(row.get(id_col, "")).strip()
                if not beh_id:
                    continue
                try:
                    bid = int(float(beh_id))
                except Exception:
                    continue

                # Keep mapping consistent with dataset preprocessing (shift + clip to oov)
                bid = int(bid) + int(shift_id_by)
                if max_vocab_size is not None and max_vocab_size > 0 and bid >= int(max_vocab_size):
                    bid = int(oov_behavior_id) if oov_behavior_id is not None else int(max_vocab_size) - 1

                text = strip_feature_code(str(row.get(text_col, "")).strip())
                # OOV bucket text should be stable (many ids may collapse into same beh token)
                if max_vocab_size is not None and oov_behavior_id is not None and bid == int(oov_behavior_id):
                    text = text or "低频合并行为"
                if not text:
                    continue
                beh_token = f"<beh_{bid}>"
                tid = tok.convert_tokens_to_ids(beh_token)
                if tid is None or int(tid) < 0:
                    continue
                # Guard: unknown strings may map to unk_token_id; ensure roundtrip matches.
                if tok.convert_ids_to_tokens(int(tid)) != beh_token:
                    continue
                tid_i = int(tid)
                # Keep first occurrence for stability (especially important for OOV bucket).
                if tid_i not in mapping:
                    mapping[tid_i] = text
                    token_ids.append(tid_i)

        # de-dup while keeping order
        seen = set()
        token_ids_uniq = []
        for t in token_ids:
            if t in seen:
                continue
            seen.add(t)
            token_ids_uniq.append(t)

        self._stage1_beh_desc_by_token_id = mapping
        self._stage1_beh_token_ids_all = token_ids_uniq

    def _tokenize_cached(self, text: str) -> list[int]:
        if text in self._stage1_tok_cache:
            return self._stage1_tok_cache[text]
        tok = self._get_tokenizer()
        if tok is None:
            return []
        out = tok(text, add_special_tokens=False)
        ids = list(out.get("input_ids", []))
        self._stage1_tok_cache[text] = ids
        return ids

    def _sample_beh_token_ids(
        self, inputs: dict, k: int, prefer_from_batch: bool, orig_vocab_size: Optional[int]
    ) -> list[int]:
        self._stage1_init_cache(model=self.model)
        assert self._stage1_beh_desc_by_token_id is not None
        assert self._stage1_beh_token_ids_all is not None

        # candidate pool
        cand: list[int] = []
        if prefer_from_batch and isinstance(inputs, dict) and "input_ids" in inputs and orig_vocab_size is not None:
            x = inputs["input_ids"]
            if isinstance(x, torch.Tensor):
                with torch.no_grad():
                    ids = x.detach().view(-1)
                    ids = ids[ids >= int(orig_vocab_size)]
                    if ids.numel() > 0:
                        uniq = torch.unique(ids).tolist()
                        for t in uniq:
                            ti = int(t)
                            if ti in self._stage1_beh_desc_by_token_id:
                                cand.append(ti)

        # if not (or not enough), fall back to global pool
        if len(cand) < k:
            if not cand:
                cand = list(self._stage1_beh_token_ids_all)
            else:
                cand_set = set(cand)
                cand.extend([int(t) for t in self._stage1_beh_token_ids_all if int(t) not in cand_set])
        if not cand:
            return []

        # random sample without replacement (torch for determinism on device)
        if k > 0 and len(cand) > k:
            g = torch.Generator(device="cpu")
            g.manual_seed(int(getattr(self.args, "seed", 42)) + int(getattr(self.state, "global_step", 0)))
            perm = torch.randperm(len(cand), generator=g).tolist()
            cand = [cand[i] for i in perm[:k]]
        elif k > 0 and len(cand) < k:
            # not enough unique candidates, just return all we have
            pass
        return cand

    def _compute_stage1_gen_loss(self, model, weight: float) -> torch.Tensor:
        if weight <= 0.0:
            return torch.zeros((), device=next(model.parameters()).device)

        self._stage1_init_cache(model)
        assert self._stage1_beh_desc_by_token_id is not None
        assert self._stage1_beh_token_ids_all is not None

        tok = self._get_tokenizer()
        assert tok is not None

        device = next(model.parameters()).device
        B = int(getattr(self.finetuning_args, "stage1_gen_batch_size", 32))
        tmpl = str(getattr(self.finetuning_args, "stage1_gen_prompt_template", "行为含义: {beh_token} -> "))
        if "{beh_token}" not in tmpl:
            tmpl = "行为含义: {beh_token} -> "

        # sample from global (generation does not need to depend on current AR batch)
        token_ids = self._sample_beh_token_ids({}, B, prefer_from_batch=False, orig_vocab_size=None)
        if not token_ids:
            self._stage1_loss_metrics["train_gen_loss"] = 0.0
            self._stage1_loss_metrics["train_gen_B"] = 0.0
            return torch.zeros((), device=device)

        prompts_ids: list[list[int]] = []
        targets_ids: list[list[int]] = []
        for tid in token_ids:
            beh_token = tok.convert_ids_to_tokens(int(tid))
            desc = self._stage1_beh_desc_by_token_id.get(int(tid), "")
            if not desc:
                continue
            prompt = tmpl.format(beh_token=beh_token)
            p_ids = self._tokenize_cached(prompt)
            d_ids = self._tokenize_cached(desc)
            if not p_ids or not d_ids:
                continue
            prompts_ids.append(p_ids)
            targets_ids.append(d_ids)

        if not prompts_ids:
            self._stage1_loss_metrics["train_gen_loss"] = 0.0
            self._stage1_loss_metrics["train_gen_B"] = 0.0
            return torch.zeros((), device=device)

        seqs = [p + d for p, d in zip(prompts_ids, targets_ids)]
        max_len = max(len(s) for s in seqs)
        pad_id = tok.pad_token_id
        if pad_id is None:
            pad_id = tok.eos_token_id if tok.eos_token_id is not None else 0

        input_ids = torch.full((len(seqs), max_len), int(pad_id), device=device, dtype=torch.long)
        labels = torch.full((len(seqs), max_len), int(IGNORE_INDEX), device=device, dtype=torch.long)
        attn = torch.zeros((len(seqs), max_len), device=device, dtype=torch.long)
        for i, (p, d, s) in enumerate(zip(prompts_ids, targets_ids, seqs)):
            l = len(s)
            input_ids[i, :l] = torch.tensor(s, device=device, dtype=torch.long)
            attn[i, :l] = 1
            # only target part contributes to loss
            labels[i, len(p) : l] = torch.tensor(d, device=device, dtype=torch.long)

        out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
        gen_loss = out.loss

        # token-level accuracy on target positions (proxy for “重构越来越准”)
        with torch.no_grad():
            logits = getattr(out, "logits", None)
            if logits is not None:
                pred = logits.argmax(dim=-1)
                m = labels.ne(int(IGNORE_INDEX))
                if m.any():
                    acc = (pred[m] == labels[m]).float().mean()
                    self._stage1_loss_metrics["train_gen_tok_acc"] = float(acc.item())
                else:
                    self._stage1_loss_metrics["train_gen_tok_acc"] = 0.0

        self._stage1_loss_metrics["train_gen_loss"] = float(gen_loss.detach().item())
        self._stage1_loss_metrics["train_gen_B"] = float(len(seqs))

        return gen_loss * float(weight)

    @torch.no_grad()
    def _stage1_project_beh_embeddings_(self, model) -> None:
        """
        每个 optimizer step 后，对 beh embedding table 做一次轻量“投影/归一化”稳定化处理：
        - 去均值（保持 zero-mean）
        - 全局 norm match（保持尺度）
        只作用于 token_id >= original_vocab_size 的行。
        """
        orig_size = getattr(self.finetuning_args, "original_vocab_size", None)
        if orig_size is None:
            return
        orig_size = int(orig_size)
        emb_w = model.get_input_embeddings().weight
        if emb_w.size(0) <= orig_size:
            return
        beh = emb_w[orig_size:]  # view
        if beh.numel() == 0:
            return

        # 1) zero-mean
        mu = beh.mean(dim=0, keepdim=True)
        beh.sub_(mu)

        # 2) global norm match to base vocab mean norm (estimated once)
        if self._stage1_base_norm_target is None:
            base = emb_w[:orig_size]
            n = int(min(50000, base.size(0)))
            g = torch.Generator(device="cpu")
            g.manual_seed(int(getattr(self.args, "seed", 42)))
            idx = torch.randperm(base.size(0), generator=g)[:n].to(device=base.device)
            self._stage1_base_norm_target = float(base.index_select(0, idx).float().norm(dim=1).mean().item())

        target = float(self._stage1_base_norm_target)
        cur = float(beh.float().norm(dim=1).mean().item())
        scale = target / (cur + 1e-12)
        beh.mul_(scale)

        # expose metrics for logging
        self._stage1_loss_metrics.setdefault("train_proj_scale", float(scale))
        self._stage1_loss_metrics.setdefault("train_proj_mu_norm", float(mu.float().norm().item()))

    @override
    def optimizer_step(self, *args, **kwargs):
        out = super().optimizer_step(*args, **kwargs)
        # Only project when Stage1 extra losses are enabled (avoid touching other stages).
        gen_w = float(getattr(self.finetuning_args, "stage1_gen_loss_weight", 0.0) or 0.0)
        if gen_w > 0.0:
            try:
                self._stage1_project_beh_embeddings_(self.model)
            except Exception:
                # Do not crash training for projection; keep a one-line signal in logs.
                self._stage1_loss_metrics["train_proj_failed"] = 1.0
        return out

    @torch.no_grad()
    def _compute_train_hr_last_position(
        self, logits: "torch.Tensor", labels: "torch.Tensor", k_list: list[int]
    ) -> dict[str, float]:
        """Compute HR@K for the last non-IGNORE target token of each sample (skip EOS if possible)."""
        device = logits.device
        max_k = int(max(k_list))
        vocab_size = int(logits.size(-1))

        # If `original_vocab_size` is provided, we treat token ids >= original_vocab_size as behavior tokens
        # and compute HR@K only within those candidates.
        orig_size = getattr(self.finetuning_args, "original_vocab_size", None)
        if orig_size is not None:
            orig_size = int(orig_size)
            if not (0 <= orig_size < vocab_size):
                orig_size = None

        beh_ids: Optional["torch.Tensor"] = None
        if orig_size is not None:
            cache_ok = (
                self._hr_beh_token_ids is not None
                and self._hr_beh_token_ids.device == device
                and self._hr_beh_token_ids_orig_size == orig_size
                and self._hr_beh_token_ids_vocab_size == vocab_size
            )
            if cache_ok:
                beh_ids = self._hr_beh_token_ids
            else:
                beh_ids = torch.arange(orig_size, vocab_size, device=device, dtype=torch.long)
                self._hr_beh_token_ids = beh_ids
                self._hr_beh_token_ids_orig_size = orig_size
                self._hr_beh_token_ids_vocab_size = vocab_size

        # Skip EOS token if we can infer eos_token_id (avoid only evaluating "predict EOS").
        eos_id = None
        pc = getattr(self, "processing_class", None)
        if pc is not None:
            eos_id = getattr(pc, "eos_token_id", None)
        # NOTE: Do not access `Trainer.tokenizer` here (deprecated in transformers>=4.46),
        # otherwise it will spam warnings on every training step.

        hits = torch.zeros(len(k_list), dtype=torch.long, device=device)
        total = torch.zeros(1, dtype=torch.long, device=device)

        bsz = int(labels.size(0))
        seq_len = int(labels.size(1))
        for b in range(bsz):
            last_idx = -1
            for i in range(seq_len - 1, -1, -1):
                y = int(labels[b, i].item())
                if y == IGNORE_INDEX:
                    continue
                if eos_id is not None and y == int(eos_id):
                    continue
                if orig_size is not None and y < orig_size:
                    continue
                last_idx = i
                break
            if last_idx <= 0:
                continue  # need a previous position to predict this token

            gt = labels[b, last_idx]
            pred_logits = logits[b, last_idx - 1]  # predict current token from previous position
            if beh_ids is None:
                k_eff = min(max_k, int(pred_logits.size(-1)))
                topk = torch.topk(pred_logits, k=k_eff, dim=-1).indices  # [k_eff]
            else:
                beh_logits = pred_logits.index_select(0, beh_ids)  # [num_beh_tokens]
                k_eff = min(max_k, int(beh_logits.size(-1)))
                if k_eff <= 0:
                    continue
                topk_local = torch.topk(beh_logits, k=k_eff, dim=-1).indices  # [k_eff]
                topk = beh_ids[topk_local]  # [k_eff], global token ids
            total += 1
            for j, k in enumerate(k_list):
                if (topk[: int(k)] == gt).any():
                    hits[j] += 1

        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(hits, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(total, op=torch.distributed.ReduceOp.SUM)

        denom = float(total.item())
        out: dict[str, float] = {}
        for k, h in zip(k_list, hits):
            out[f"train_HR_at_{int(k)}"] = float(h.item() / denom) if denom > 0 else 0.0
        out["train_total_predictions"] = float(denom)
        return out

    @override
    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        # Attach train-batch HR@K only to training logs (loss lines).
        if logs is not None:
            logs = dict(logs)

            # Make the log line self-contained: include current global step (optimizer step).
            # This is especially useful when tqdm is disabled and the logger only prints the metrics dict.
            logs.setdefault("step", int(getattr(self.state, "global_step", 0)))

            if "loss" in logs:
                if self._train_hr_metrics:
                    logs.update(self._train_hr_metrics)
                if self._stage1_loss_metrics:
                    logs.update(self._stage1_loss_metrics)
                # NOTE: Stage1 Gen Preview (model.generate) removed to avoid hanging under distributed/ZeRO settings.

        return super().log(logs, start_time=start_time)
