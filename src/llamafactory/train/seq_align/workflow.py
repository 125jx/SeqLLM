"""
Sequence-Alignment workflow: train a BehaviorProjector on SFT-formatted data.

Phase 1 of the LLaVA-style two-phase approach:
  LLM frozen, only Projector learns to map behavior-token embeddings.
"""

from typing import TYPE_CHECKING, Optional

from ...data import SFTDataCollatorWith4DAttentionMask, get_dataset, get_template_and_fix_tokenizer
from ...extras.constants import IGNORE_INDEX
from ...extras.logging import get_logger
from ...extras.misc import calculate_tps
from ...extras.ploting import plot_loss
from ...model import load_model, load_tokenizer
from ..trainer_utils import create_modelcard_and_push
from .projector import BehaviorProjector
from .trainer import SeqAlignTrainer

if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments

logger = get_logger(__name__)


def run_seq_align(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    generating_args: "GeneratingArguments",
    callbacks: Optional[list["TrainerCallback"]] = None,
) -> None:
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    setattr(data_args, "seq_align_compress_recent_first", True)
    template = get_template_and_fix_tokenizer(tokenizer, data_args)

    dataset_module = get_dataset(template, model_args, data_args, training_args, stage="sft", **tokenizer_module)

    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)

    # ---- Projector ----
    hidden_size = model.config.hidden_size
    behavior_token_start_id = finetuning_args.seq_align_behavior_token_start_id

    checkpoint = finetuning_args.seq_align_projector_checkpoint
    if checkpoint is not None:
        logger.info_rank0(f"Loading projector from {checkpoint}")
        projector = BehaviorProjector.from_pretrained(checkpoint)
    else:
        logger.info_rank0("Creating new BehaviorProjector")
        projector = BehaviorProjector(
            dim=hidden_size,
            num_layers=finetuning_args.seq_align_projector_layers,
            dropout=finetuning_args.seq_align_projector_dropout,
        )

    projector = projector.to(device=model.device, dtype=model.dtype)

    # ---- Data collator ----
    data_collator = SFTDataCollatorWith4DAttentionMask(
        template=template,
        model=model if not training_args.predict_with_generate else None,
        pad_to_multiple_of=8 if training_args.do_train else None,
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
        block_diag_attn=model_args.block_diag_attn,
        attn_implementation=getattr(model.config, "_attn_implementation", None),
        compute_dtype=model_args.compute_dtype,
        **tokenizer_module,
    )

    # ---- Trainer ----
    trainer_tok = {k: v for k, v in tokenizer_module.items() if k != "processor"}

    freeze_llm = finetuning_args.seq_align_freeze_llm
    logger.info_rank0(f"freeze_llm={freeze_llm} ({'Phase 1: projector only' if freeze_llm else 'Phase 2: projector + LLM'})")

    trainer = SeqAlignTrainer(
        model=model,
        args=training_args,
        projector=projector,
        behavior_token_start_id=behavior_token_start_id,
        freeze_llm=freeze_llm,
        data_collator=data_collator,
        callbacks=callbacks,
        **dataset_module,
        **trainer_tok,
    )

    # ---- Train ----
    if training_args.do_train:
        result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
        if finetuning_args.include_effective_tokens_per_second:
            result.metrics["effective_tokens_per_sec"] = calculate_tps(
                dataset_module["train_dataset"], result.metrics, stage="sft"
            )
        trainer.log_metrics("train", result.metrics)
        trainer.save_metrics("train", result.metrics)
        trainer.save_state()
        if trainer.is_world_process_zero() and finetuning_args.plot_loss:
            plot_loss(training_args.output_dir, keys=["loss"])

    # ---- Eval ----
    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    create_modelcard_and_push(trainer, model_args, data_args, training_args, finetuning_args)
