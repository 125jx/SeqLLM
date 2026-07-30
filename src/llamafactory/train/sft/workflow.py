# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/examples/pytorch/summarization/run_summarization.py
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

from typing import TYPE_CHECKING, Optional

import torch

from ...data import SFTDataCollatorWith4DAttentionMask, get_dataset, get_template_and_fix_tokenizer
from ...extras.constants import IGNORE_INDEX
from ...extras.logging import get_logger
from ...extras.misc import calculate_tps
from ...extras.ploting import plot_loss
from ...model import load_model, load_tokenizer
from ..callbacks import PredictionLogCallback
from ..trainer_utils import create_modelcard_and_push
from .metric import (
    ComputeAccuracy,
    ComputeRecallAtK,
    ComputeSimilarity,
    eval_logit_processor,
    recall_at_k_logit_processor,
    set_recall_settings,
)
from .trainer import CustomSeq2SeqTrainer


if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments


logger = get_logger(__name__)


def _collect_first_response_token_ids(eval_dataset) -> "torch.Tensor":
    """Scan the (already tokenized) eval dataset and return the unique set of
    first-response token ids as a 1-D long tensor.

    The candidate vocab for Recall@K is taken to be exactly the set of
    first-response tokens that appear in the eval set: since each model is
    trained for one evaluation dataset (with its own added-token range), this
    automatically yields the correct candidate vocabulary without requiring
    any manual range configuration.
    """
    cand: set[int] = set()
    for sample in eval_dataset:
        labels = sample.get("labels")
        if labels is None:
            continue
        for tok in labels:
            tok = int(tok)
            if tok != IGNORE_INDEX:
                cand.add(tok)
                break
    if not cand:
        return torch.empty(0, dtype=torch.long)
    return torch.tensor(sorted(cand), dtype=torch.long)


def run_sft(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    generating_args: "GeneratingArguments",
    callbacks: Optional[list["TrainerCallback"]] = None,
):
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    dataset_module = get_dataset(template, model_args, data_args, training_args, stage="sft", **tokenizer_module)
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)

    eval_dataset_names = [str(name).lower() for name in (data_args.eval_dataset or [])]
    primary_eval_dataset = eval_dataset_names[0] if len(eval_dataset_names) == 1 else None
    is_review_gen_eval = primary_eval_dataset is not None and "review_gen" in primary_eval_dataset
    is_fav_eval = primary_eval_dataset is not None and (
        "fav_category" in primary_eval_dataset or "fav_genre" in primary_eval_dataset
    )

    if is_review_gen_eval and (training_args.do_eval or training_args.do_predict) and not training_args.predict_with_generate:
        logger.warning_rank0(
            "Detected review_gen eval dataset `%s`: auto-enabling predict_with_generate to compute ROUGE/BLEU metrics.",
            primary_eval_dataset,
        )
        training_args.predict_with_generate = True

    if getattr(model, "is_quantized", False) and not training_args.do_train:
        setattr(model, "_hf_peft_config_loaded", True)  # hack here: make model compatible with prediction

    data_collator = SFTDataCollatorWith4DAttentionMask(
        template=template,
        model=model if not training_args.predict_with_generate else None,
        pad_to_multiple_of=8 if training_args.do_train else None,  # for shift short attention
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
        block_diag_attn=model_args.block_diag_attn,
        attn_implementation=getattr(model.config, "_attn_implementation", None),
        compute_dtype=model_args.compute_dtype,
        **tokenizer_module,
    )

    # Metric utils
    metric_module = {}
    if training_args.predict_with_generate:
        metric_module["compute_metrics"] = ComputeSimilarity(tokenizer=tokenizer)
    elif finetuning_args.compute_accuracy:
        metric_module["compute_metrics"] = ComputeAccuracy()
        metric_module["preprocess_logits_for_metrics"] = eval_logit_processor
    elif training_args.do_eval or training_args.do_predict:
        # Recall@K for next-item prediction (first response token).
        # Priority:
        #   1) if explicit [eval_candidate_start_id, eval_candidate_end_id) is set and valid,
        #      use this range;
        #   2) else if `original_vocab_size` is set and valid, evaluate on
        #      [original_vocab_size, vocab_size);
        #   3) otherwise, fall back to auto-inferred candidates from eval dataset
        #      first-response token ids;
        #   4) if inference fails, fall back to full vocab.
        eval_ds = dataset_module.get("eval_dataset")
        candidate_ids: Optional["torch.Tensor"] = None

        model_vocab_size = int(getattr(model.config, "vocab_size", len(tokenizer)))
        source = "full vocab"

        # 1) Optional explicit candidate range for behavior-token-only evaluation.
        # For fav_category/fav_genre eval datasets, eval_fav_* has highest priority.
        cand_start = getattr(finetuning_args, "eval_candidate_start_id", None)
        cand_end = getattr(finetuning_args, "eval_candidate_end_id", None)
        cand_label_start = "eval_candidate_start_id"
        cand_label_end = "eval_candidate_end_id"

        if is_fav_eval and getattr(finetuning_args, "eval_fav_start_id", None) is not None:
            cand_start = getattr(finetuning_args, "eval_fav_start_id", None)
            cand_end = getattr(finetuning_args, "eval_fav_end_id", None)
            cand_label_start = "eval_fav_start_id"
            cand_label_end = "eval_fav_end_id"

        if cand_start is not None:
            try:
                cand_start = int(cand_start)
            except (TypeError, ValueError):
                cand_start = None

            if cand_end is None:
                cand_end = model_vocab_size
            else:
                try:
                    cand_end = int(cand_end)
                except (TypeError, ValueError):
                    cand_end = None

            if cand_start is not None and cand_end is not None and 0 <= cand_start < cand_end <= model_vocab_size:
                candidate_ids = torch.arange(cand_start, cand_end, dtype=torch.long)
                source = f"explicit range [{cand_label_start}, {cand_label_end})"
            else:
                logger.warning_rank0(
                    "Recall@K: invalid explicit candidate range %s=%s %s=%s for model vocab_size=%s; "
                    "falling back to original_vocab_size/eval-dataset inference.",
                    cand_label_start,
                    str(getattr(finetuning_args, cand_label_start, None)),
                    cand_label_end,
                    str(getattr(finetuning_args, cand_label_end, None)),
                    str(model_vocab_size),
                )
        elif is_fav_eval:
            logger.warning_rank0(
                "Recall@K: detected fav eval dataset `%s` but eval_fav_start_id is not set; "
                "falling back to eval_candidate_*/original_vocab_size.",
                primary_eval_dataset,
            )

        # 2) original_vocab_size fallback.
        orig_size = getattr(finetuning_args, "original_vocab_size", None)
        if orig_size is not None:
            try:
                orig_size = int(orig_size)
            except (TypeError, ValueError):
                orig_size = None

        if candidate_ids is None and orig_size is not None and 0 <= orig_size < model_vocab_size:
            candidate_ids = torch.arange(orig_size, model_vocab_size, dtype=torch.long)
            source = "full added-token range [original_vocab_size, vocab_size)"
        elif candidate_ids is None:
            if orig_size is not None:
                logger.warning_rank0(
                    "Recall@K: invalid original_vocab_size=%s for model vocab_size=%s; "
                    "falling back to eval-dataset inferred candidates.",
                    str(orig_size),
                    str(model_vocab_size),
                )
            # 3) eval dataset inferred fallback.
            if eval_ds is not None and not isinstance(eval_ds, dict):
                candidate_ids = _collect_first_response_token_ids(eval_ds)
                if candidate_ids.numel() == 0:
                    logger.warning_rank0(
                        "Recall@K: failed to infer any candidate token id from the eval dataset; "
                        "falling back to full vocab."
                    )
                    candidate_ids = None
                else:
                    source = "auto-inferred from eval dataset's first-response tokens"

        max_k = max(finetuning_args.recall_k_list) if finetuning_args.recall_k_list else 10
        set_recall_settings(candidate_ids=candidate_ids, max_k=max_k)
        metric_module["compute_metrics"] = ComputeRecallAtK(
            k_list=finetuning_args.recall_k_list,
            candidate_ids=candidate_ids,
        )
        metric_module["preprocess_logits_for_metrics"] = recall_at_k_logit_processor
        n_cand = int(candidate_ids.numel()) if candidate_ids is not None else 0
        if n_cand > 0:
            cand_min = int(candidate_ids.min().item())
            cand_max = int(candidate_ids.max().item())
            logger.info_rank0(
                f"Recall@K enabled: k_list={finetuning_args.recall_k_list}, "
                f"|candidates|={n_cand}, id range observed=[{cand_min}, {cand_max}] "
                f"(source: {source})."
            )
        else:
            logger.info_rank0(
                f"Recall@K enabled: k_list={finetuning_args.recall_k_list}, "
                "no candidate restriction (using full vocab)."
            )

    # Keyword arguments for `model.generate`
    gen_kwargs = generating_args.to_dict(obey_generation_config=True)
    gen_kwargs["eos_token_id"] = [tokenizer.eos_token_id] + tokenizer.additional_special_tokens_ids
    gen_kwargs["pad_token_id"] = tokenizer.pad_token_id

    # 添加预测日志 Callback（在训练过程中打印模型预测）
    if training_args.do_train and finetuning_args.log_predictions_steps > 0:
        train_dataset = dataset_module.get("train_dataset")
        if train_dataset is not None and len(train_dataset) > 0:
            # 获取第一个样本作为预测示例
            sample = train_dataset[0]
            sample_input_ids = sample["input_ids"]
            # 获取标签（过滤掉 IGNORE_INDEX）
            sample_labels = [lid for lid in sample["labels"] if lid != IGNORE_INDEX]
            sample_label = tokenizer.decode(sample_labels, skip_special_tokens=True) if sample_labels else ""
            
            if callbacks is None:
                callbacks = []
            callbacks.append(PredictionLogCallback(
                tokenizer=tokenizer,
                sample_input_ids=sample_input_ids,
                sample_label=sample_label,
                log_steps=finetuning_args.log_predictions_steps,
            ))
            logger.info_rank0(f"启用预测日志，每 {finetuning_args.log_predictions_steps} 步打印一次模型预测")
    
    # Initialize our Trainer
    trainer = CustomSeq2SeqTrainer(
        model=model,
        args=training_args,
        finetuning_args=finetuning_args,
        data_collator=data_collator,
        callbacks=callbacks,
        gen_kwargs=gen_kwargs,
        **dataset_module,
        **tokenizer_module,
        **metric_module,
    )

    # Training
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
        if finetuning_args.include_effective_tokens_per_second:
            train_result.metrics["effective_tokens_per_sec"] = calculate_tps(
                dataset_module["train_dataset"], train_result.metrics, stage="sft"
            )

        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        if trainer.is_world_process_zero() and finetuning_args.plot_loss:
            keys = ["loss"]
            if isinstance(dataset_module.get("eval_dataset"), dict):
                keys += sum(
                    [[f"eval_{key}_loss", f"eval_{key}_accuracy"] for key in dataset_module["eval_dataset"].keys()], []
                )
            else:
                keys += ["eval_loss", "eval_accuracy"]

            plot_loss(training_args.output_dir, keys=keys)

    if training_args.predict_with_generate:
        tokenizer.padding_side = "left"  # use left-padding in generation

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval", **gen_kwargs)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Predict
    if training_args.do_predict:
        logger.warning_rank0_once("Batch generation can be very slow. Consider using `scripts/vllm_infer.py` instead.")
        predict_results = trainer.predict(dataset_module["eval_dataset"], metric_key_prefix="predict", **gen_kwargs)
        trainer.log_metrics("predict", predict_results.metrics)
        trainer.save_metrics("predict", predict_results.metrics)
        trainer.save_predictions(dataset_module["eval_dataset"], predict_results, generating_args.skip_special_tokens)

    # Create model card
    create_modelcard_and_push(trainer, model_args, data_args, training_args, finetuning_args)
