"""
SeqAlignTrainer -- trains BehaviorProjector (+ optionally LLM) with inputs_embeds.

Phase 1: freeze_llm=True  -> only projector trainable
Phase 2: freeze_llm=False -> projector + full LLM trainable
"""

import json
import os
from typing import TYPE_CHECKING, Any, Dict, Optional

import torch
import torch.nn as nn
from transformers import Seq2SeqTrainer
from typing_extensions import override

from ...extras.logging import get_logger
from .projector import BehaviorProjector

if TYPE_CHECKING:
    from transformers import PreTrainedModel

logger = get_logger(__name__)

BEHAVIOR_TOKEN_START_ID_DEFAULT = 151675


class SeqAlignTrainer(Seq2SeqTrainer):
    def __init__(
        self,
        projector: BehaviorProjector,
        behavior_token_start_id: int = BEHAVIOR_TOKEN_START_ID_DEFAULT,
        freeze_llm: bool = True,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.behavior_token_start_id = behavior_token_start_id
        self.freeze_llm = freeze_llm

        self.model.behavior_projector = projector

        if self.freeze_llm:
            self._freeze_llm()
        self._log_params()

    def _freeze_llm(self) -> None:
        for name, p in self.model.named_parameters():
            if "behavior_projector" not in name:
                p.requires_grad = False
        logger.info_rank0("LLM parameters frozen (only behavior_projector is trainable)")

    def _log_params(self) -> None:
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info_rank0(f"Total params: {total:,}  Trainable: {trainable:,} ({100*trainable/total:.4f}%)")

    @staticmethod
    def _get_embed_tokens(model: "PreTrainedModel") -> nn.Embedding:
        m = model.module if hasattr(model, "module") else model
        if hasattr(m, "model") and hasattr(m.model, "embed_tokens"):
            return m.model.embed_tokens
        return m.get_input_embeddings()

    @staticmethod
    def _get_projector(model: "PreTrainedModel") -> BehaviorProjector:
        m = model.module if hasattr(model, "module") else model
        return m.behavior_projector

    def compute_loss(
        self,
        model: "PreTrainedModel",
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ):
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")
        labels = inputs.get("labels")

        embed_tokens = self._get_embed_tokens(model)
        projector = self._get_projector(model)

        all_embeds = embed_tokens(input_ids)

        behavior_mask = input_ids >= self.behavior_token_start_id
        idx = behavior_mask.nonzero(as_tuple=False)
        bi, si = idx[:, 0], idx[:, 1]
        projected = projector(all_embeds[bi, si])
        inputs_embeds = all_embeds.clone()
        inputs_embeds[bi, si] = projected

        outputs = model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )
        return (outputs.loss, outputs) if return_outputs else outputs.loss

    @override
    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        if logs is not None:
            logs = dict(logs)
            logs.setdefault("step", int(getattr(self.state, "global_step", 0)))

        return super().log(logs, start_time=start_time)

    def _detach_projector(self) -> BehaviorProjector:
        """Remove projector from model's module tree so DeepSpeed doesn't touch it during save."""
        m = self.model.module if hasattr(self.model, "module") else self.model
        proj = m._modules.pop("behavior_projector")
        if hasattr(self.model, "_modules") and "behavior_projector" in self.model._modules:
            self.model._modules.pop("behavior_projector")
        return proj

    def _attach_projector(self, projector: BehaviorProjector) -> None:
        m = self.model.module if hasattr(self.model, "module") else self.model
        m.behavior_projector = projector

    def _save_projector(self, projector: BehaviorProjector, output_dir: str) -> None:
        proj_dir = os.path.join(output_dir, "projector")
        if self.is_deepspeed_enabled:
            import deepspeed
            # GatheredParameters is a collective: ALL ranks must enter, only rank 0 saves
            with deepspeed.zero.GatheredParameters(list(projector.parameters()), modifier_rank=0):
                if self.args.should_save:
                    os.makedirs(proj_dir, exist_ok=True)
                    sd = {k: v.detach().cpu().clone() for k, v in projector.state_dict().items()}
                    torch.save(sd, os.path.join(proj_dir, "projector.pt"))
        else:
            if not self.args.should_save:
                return
            os.makedirs(proj_dir, exist_ok=True)
            sd = {k: v.detach().cpu().clone() for k, v in projector.state_dict().items()}
            torch.save(sd, os.path.join(proj_dir, "projector.pt"))

        if self.args.should_save:
            cfg = {"dim": projector.dim, "hidden_dim": projector.hidden_dim, "num_layers": projector.num_layers}
            with open(os.path.join(proj_dir, "projector_config.json"), "w") as f:
                json.dump(cfg, f, indent=2)

    @override
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False) -> None:
        if output_dir is None:
            output_dir = self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        projector = self._detach_projector()
        try:
            self._save_projector(projector, output_dir)

            if not self.freeze_llm:
                super().save_model(output_dir, _internal_call=_internal_call)
                if self.args.should_save:
                    logger.info_rank0(f"SeqAlign Phase2 checkpoint (LLM + Projector) saved to {output_dir}")
            else:
                if self.args.should_save:
                    info = {
                        "behavior_token_start_id": self.behavior_token_start_id,
                        "freeze_llm": self.freeze_llm,
                        "base_model": getattr(self.model.config, "_name_or_path", "unknown"),
                    }
                    with open(os.path.join(output_dir, "seq_align_info.json"), "w") as f:
                        json.dump(info, f, indent=2)
                    if self.tokenizer is not None:
                        self.tokenizer.save_pretrained(output_dir)
                    torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
                    logger.info_rank0(f"SeqAlign Phase1 checkpoint (Projector only) saved to {output_dir}")
        finally:
            self._attach_projector(projector)
