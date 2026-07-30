# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/examples/pytorch/language-modeling/run_clm.py
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

import math
from typing import TYPE_CHECKING, Optional

from transformers import DataCollatorForLanguageModeling
import torch

from ...data import get_dataset, get_template_and_fix_tokenizer
from ...extras.ploting import plot_loss
from ...model import load_model, load_tokenizer
from ..trainer_utils import create_modelcard_and_push
from ...extras.misc import is_env_enabled
from .trainer import CustomTrainer
from .metric import (
    ComputeDualHitRate,
    ComputeHitRate,
    dual_hit_rate_logit_processor,
    hit_rate_logit_processor,
    load_beh_token_ids,
    set_beh_token_ids,
)


if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments, ModelArguments


class _PTDataCollatorPreserveLabels(DataCollatorForLanguageModeling):
    """
    PT collator that preserves `labels` if the dataset already provides them.
    - If any feature contains `labels`, we strip them before `tokenizer.pad` (which would
      otherwise fail at `BatchEncoding.convert_to_tensors` because it does NOT pad unknown
      keys), then manually pad them with IGNORE_INDEX (-100) and add them back as a tensor.
    - Extra integer feature streams (e.g. triplet `category_ids` / `rating_ids`) are padded
      with 0, and extra label streams (e.g. `labels_cat`) are padded with IGNORE_INDEX.
    - Otherwise, fall back to the default CLM behavior (labels = input_ids, pad masked).
    """

    # Extra streams produced by the triplet-fusion processor. id streams pad with 0,
    # label streams pad with IGNORE_INDEX (-100).
    _EXTRA_ID_FIELDS = ("category_ids", "rating_ids")
    _EXTRA_LABEL_FIELDS = ("labels_cat",)

    @staticmethod
    def _pad_field(seqs, max_len, pad_value):
        out = []
        for s in seqs:
            s = list(s)
            if len(s) < max_len:
                s = s + [pad_value] * (max_len - len(s))
            else:
                s = s[:max_len]
            out.append(s)
        return torch.tensor(out, dtype=torch.long)

    def __call__(self, features):
        has_any_labels = any(("labels" in f and f["labels"] is not None) for f in features)
        if not has_any_labels:
            return super().__call__(features)

        # Pop labels + extra streams out of features so tokenizer.pad sees only input_ids / attention_mask.
        saved_labels: list[list[int]] = []
        saved_extra_ids: dict[str, list] = {k: [] for k in self._EXTRA_ID_FIELDS}
        saved_extra_labels: dict[str, list] = {k: [] for k in self._EXTRA_LABEL_FIELDS}
        for f in features:
            if "labels" in f and f["labels"] is not None:
                saved_labels.append(list(f["labels"]))
                f.pop("labels", None)
            else:
                saved_labels.append(list(f["input_ids"]))
            for k in self._EXTRA_ID_FIELDS:
                if k in f and f[k] is not None:
                    saved_extra_ids[k].append(f[k])
                    f.pop(k, None)
            for k in self._EXTRA_LABEL_FIELDS:
                if k in f and f[k] is not None:
                    saved_extra_labels[k].append(f[k])
                    f.pop(k, None)

        # Pad input_ids / attention_mask only.
        batch = self.tokenizer.pad(
            features,
            padding=True,
            return_tensors="pt",
        )

        max_len = int(batch["input_ids"].shape[1])
        batch["labels"] = self._pad_field(saved_labels, max_len, -100)
        for k in self._EXTRA_ID_FIELDS:
            if len(saved_extra_ids[k]) == len(saved_labels):
                batch[k] = self._pad_field(saved_extra_ids[k], max_len, 0)
        for k in self._EXTRA_LABEL_FIELDS:
            if len(saved_extra_labels[k]) == len(saved_labels):
                batch[k] = self._pad_field(saved_extra_labels[k], max_len, -100)
        return batch


def run_pt(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    callbacks: Optional[list["TrainerCallback"]] = None,
):
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    dataset_module = get_dataset(template, model_args, data_args, training_args, stage="pt", **tokenizer_module)
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)
    data_collator = _PTDataCollatorPreserveLabels(tokenizer=tokenizer, mlm=False)

    triplet_fusion = is_env_enabled("LLAMAFACTORY_TRIPLET_FUSION") or getattr(
        model_args, "numeric_triplet_fusion", False
    )

    # Setup HR@K metric if enabled
    compute_metrics = None
    preprocess_logits_for_metrics = None
    if finetuning_args.compute_hit_rate:
        max_k = max(finetuning_args.hit_rate_k_list)
        eos_token_id = getattr(tokenizer, "eos_token_id", None)

        if triplet_fusion:
            # Dual-head: top-k over the full item vocab and the full category vocab.
            set_beh_token_ids(None, max_k, beh_range_start=None, eos_token_id=eos_token_id)
            compute_metrics = ComputeDualHitRate(
                k_list=finetuning_args.hit_rate_k_list,
                last_position_only=finetuning_args.hr_last_position_only,
                item_eos_id=eos_token_id,
            )
            preprocess_logits_for_metrics = dual_hit_rate_logit_processor
            # Both label streams must be gathered for metric computation.
            training_args.label_names = ["labels", "labels_cat"]
            # Accumulate HR incrementally per batch to bound eval memory (dual top-k tensors).
            training_args.batch_eval_metrics = True
        else:
            # Load beh token IDs if provided
            if finetuning_args.beh_tokens_file:
                beh_token_ids = load_beh_token_ids(tokenizer, finetuning_args.beh_tokens_file)
                set_beh_token_ids(beh_token_ids, max_k, beh_range_start=None, eos_token_id=eos_token_id)
            else:
                # If no explicit beh token list is provided, fall back to "range" definition:
                # treat token ids >= original_vocab_size as behavior tokens (aligned with pt/trainer.py train HR).
                beh_start = getattr(finetuning_args, "original_vocab_size", None)
                set_beh_token_ids(None, max_k, beh_range_start=beh_start, eos_token_id=eos_token_id)

            compute_metrics = ComputeHitRate(
                k_list=finetuning_args.hit_rate_k_list,
                last_position_only=finetuning_args.hr_last_position_only
            )
            preprocess_logits_for_metrics = hit_rate_logit_processor

    # Initialize our Trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        finetuning_args=finetuning_args,
        data_collator=data_collator,
        callbacks=callbacks,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        **dataset_module,
        **tokenizer_module,
    )

    # Training
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        if trainer.is_world_process_zero() and finetuning_args.plot_loss:
            keys = ["loss"]
            if isinstance(dataset_module.get("eval_dataset"), dict):
                keys += [f"eval_{key}_loss" for key in dataset_module["eval_dataset"].keys()]
            else:
                keys += ["eval_loss"]

            plot_loss(training_args.output_dir, keys=keys)

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval")

        if isinstance(dataset_module.get("eval_dataset"), dict):
            for key in dataset_module["eval_dataset"].keys():
                try:
                    perplexity = math.exp(metrics[f"eval_{key}_loss"])
                except OverflowError:
                    perplexity = float("inf")

                metrics[f"eval_{key}_perplexity"] = perplexity
        else:
            try:
                perplexity = math.exp(metrics["eval_loss"])
            except OverflowError:
                perplexity = float("inf")

            metrics["eval_perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Create model card
    create_modelcard_and_push(trainer, model_args, data_args, training_args, finetuning_args)
