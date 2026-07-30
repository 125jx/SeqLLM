# Copyright 2025 HuggingFace Inc., THUDM, and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library and the THUDM's ChatGLM implementation.
# https://github.com/huggingface/transformers/blob/v4.40.0/examples/pytorch/summarization/run_summarization.py
# https://github.com/THUDM/ChatGLM-6B/blob/main/ptuning/main.py
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

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch
from transformers.utils import is_nltk_available

from ...extras.constants import IGNORE_INDEX
from ...extras.misc import numpify
from ...extras.packages import is_jieba_available, is_rouge_available


if TYPE_CHECKING:
    from transformers import EvalPrediction, PreTrainedTokenizer


if is_jieba_available():
    import jieba  # type: ignore


if is_nltk_available():
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu  # type: ignore


if is_rouge_available():
    from rouge_chinese import Rouge  # type: ignore


def eval_logit_processor(logits: "torch.Tensor", labels: "torch.Tensor") -> "torch.Tensor":
    r"""Compute the token with the largest likelihood to reduce memory footprint."""
    if isinstance(logits, (list, tuple)):
        if logits[0].dim() == 3:  # (batch_size, seq_len, vocab_size)
            logits = logits[0]
        else:  # moe models have aux loss
            logits = logits[1]

    if logits.dim() != 3:
        raise ValueError("Cannot process the logits.")

    return torch.argmax(logits, dim=-1)


@dataclass
class ComputeAccuracy:
    r"""Compute accuracy and support `batch_eval_metrics`."""

    def _dump(self) -> Optional[dict[str, float]]:
        result = None
        if hasattr(self, "score_dict"):
            result = {k: float(np.mean(v)) for k, v in self.score_dict.items()}

        self.score_dict = {"accuracy": []}
        return result

    def __post_init__(self):
        self._dump()

    def __call__(self, eval_preds: "EvalPrediction", compute_result: bool = True) -> Optional[dict[str, float]]:
        preds, labels = numpify(eval_preds.predictions), numpify(eval_preds.label_ids)
        for i in range(len(preds)):
            pred, label = preds[i, :-1], labels[i, 1:]
            label_mask = label != IGNORE_INDEX
            self.score_dict["accuracy"].append(np.mean(pred[label_mask] == label[label_mask]))

        if compute_result:
            return self._dump()


@dataclass
class ComputeSimilarity:
    r"""Compute text similarity scores and support `batch_eval_metrics`.

    Wraps the tokenizer into metric functions, used in CustomSeq2SeqTrainer.
    """

    tokenizer: "PreTrainedTokenizer"

    def _dump(self) -> Optional[dict[str, float]]:
        result = None
        if hasattr(self, "score_dict"):
            result = {k: float(np.mean(v)) for k, v in self.score_dict.items()}

        self.score_dict = {"rouge-1": [], "rouge-2": [], "rouge-l": [], "bleu-4": []}
        return result

    def __post_init__(self):
        self._dump()

    def __call__(self, eval_preds: "EvalPrediction", compute_result: bool = True) -> Optional[dict[str, float]]:
        preds, labels = numpify(eval_preds.predictions), numpify(eval_preds.label_ids)

        preds = np.where(preds != IGNORE_INDEX, preds, self.tokenizer.pad_token_id)
        labels = np.where(labels != IGNORE_INDEX, labels, self.tokenizer.pad_token_id)

        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        for pred, label in zip(decoded_preds, decoded_labels):
            hypothesis = list(jieba.cut(pred))
            reference = list(jieba.cut(label))

            if len(" ".join(hypothesis).split()) == 0 or len(" ".join(reference).split()) == 0:
                result = {"rouge-1": {"f": 0.0}, "rouge-2": {"f": 0.0}, "rouge-l": {"f": 0.0}}
            else:
                rouge = Rouge()
                scores = rouge.get_scores(" ".join(hypothesis), " ".join(reference))
                result = scores[0]

            for k, v in result.items():
                self.score_dict[k].append(round(v["f"] * 100, 4))

        if compute_result:
            return self._dump()


# ---------------------------------------------------------------------------
# Recall@K for SFT next-item prediction
# ---------------------------------------------------------------------------
# We evaluate the model's prediction for the *first response token* of every
# sample. The softmax is restricted to a finite set of candidate token ids
# (``_RECALL_CAND_IDS``), which is auto-inferred from the eval dataset (the
# union of all first-response token ids).  Because each model is trained with
# its own added-token range corresponding to one evaluation dataset, this
# automatically yields the correct candidate vocabulary without requiring the
# user to specify start/end ranges manually.
#
# The logit processor reduces per-step output to ``[batch, seq_len, max_k]``
# token ids (memory friendly), and the metric class then aggregates Recall@K
# only on the first non-IGNORE_INDEX label of each sequence.

_RECALL_CAND_IDS: Optional[torch.Tensor] = None  # 1-D long tensor on CPU
_RECALL_MAX_K: int = 10


def set_recall_settings(
    candidate_ids: Optional["torch.Tensor"],
    max_k: int = 10,
) -> None:
    """Configure global settings for Recall@K logit processor / metric.

    Args:
        candidate_ids: 1-D long tensor of candidate token ids (the vocab over
            which softmax/top-K is restricted). If ``None``, fall back to the
            full vocab (no restriction).
        max_k: maximum K used for the top-K reduction in the logit processor.
    """
    global _RECALL_CAND_IDS, _RECALL_MAX_K
    if candidate_ids is None:
        _RECALL_CAND_IDS = None
    else:
        ids = torch.as_tensor(candidate_ids, dtype=torch.long).flatten().unique()
        _RECALL_CAND_IDS = ids
    _RECALL_MAX_K = int(max_k)


def get_recall_candidate_ids() -> Optional["torch.Tensor"]:
    """Return the configured candidate id tensor (CPU, 1-D long), or None."""
    return _RECALL_CAND_IDS


def recall_at_k_logit_processor(logits: "torch.Tensor", labels: "torch.Tensor") -> "torch.Tensor":
    """Return top-K candidate token ids restricted to the configured candidate set.

    Output shape: ``[batch, seq_len, max_k]``.
    """
    global _RECALL_CAND_IDS, _RECALL_MAX_K

    if isinstance(logits, (list, tuple)):
        if logits[0].dim() == 3:
            logits = logits[0]
        else:  # MoE models may have aux loss as the first element.
            logits = logits[1]

    if logits.dim() != 3:
        raise ValueError("Cannot process the logits.")

    _, _, vocab_size = logits.shape
    device = logits.device

    if _RECALL_CAND_IDS is not None and _RECALL_CAND_IDS.numel() > 0:
        cand_ids = _RECALL_CAND_IDS.to(device=device, dtype=torch.long)
        # Filter ids that are actually within the model's vocab range (defensive).
        cand_ids = cand_ids[(cand_ids >= 0) & (cand_ids < vocab_size)]
        cand_logits = logits[:, :, cand_ids]
        k = min(_RECALL_MAX_K, cand_logits.size(-1))
        _, topk_local = torch.topk(cand_logits, k, dim=-1)
        topk_token_ids = cand_ids[topk_local]
    else:
        k = min(_RECALL_MAX_K, vocab_size)
        _, topk_token_ids = torch.topk(logits, k, dim=-1)

    return topk_token_ids  # [batch, seq_len, max_k]


@dataclass
class ComputeRecallAtK:
    """Recall@K on the first *target* token within the configured candidate set.

    With a candidate set configured (recommended), for each sample we locate the
    first position ``i`` such that:
      - ``labels[i] != IGNORE_INDEX`` and
      - ``labels[i]`` is in candidate ids.

    This makes the metric align with "behavior-token prediction" semantics
    (e.g. when candidates are all added tokens ``[original_vocab_size, vocab)``):
    we evaluate the first behavior token instead of blindly using the first
    response token.

    The model prediction for label position ``i`` is produced by ``logits[i - 1]``
    (next-token shift), so we look up ``preds[i - 1]``.

    If a sample has response tokens but no token inside the candidate set, it is
    counted as ``oov_predictions`` and skipped from recall denominator.
    """

    k_list: list[int] = field(default_factory=lambda: [1, 5, 10])
    candidate_ids: Optional["torch.Tensor"] = None  # 1-D long tensor (CPU)

    def _dump(self) -> Optional[dict[str, float]]:
        result = None
        if hasattr(self, "hr_counts") and hasattr(self, "total_count"):
            # Keep this local-only, consistent with other metric classes in this
            # file and HF batch_eval_metrics behavior. The trainer already gathers
            # eval predictions/labels for metric computation; doing all_reduce here
            # would over-count by world size in DDP.
            total_count = float(self.total_count)
            oov_count = float(self.oov_count)
            hr_counts = {k: float(v) for k, v in self.hr_counts.items()}

            if total_count > 0 or oov_count > 0:
                if total_count > 0:
                    # MLflow metric names cannot contain '@'.
                    result = {f"recall_at_{k}": hr_counts[k] / total_count for k in self.k_list}
                    result["total_predictions"] = float(total_count)
                    result["oov_predictions"] = float(oov_count)
                else:
                    result = {f"recall_at_{k}": 0.0 for k in self.k_list}
                    result["total_predictions"] = 0.0
                    result["oov_predictions"] = float(oov_count)

        self.hr_counts = {k: 0 for k in self.k_list}
        self.total_count = 0
        self.oov_count = 0
        return result

    def __post_init__(self):
        # Pre-compute candidate id set for fast membership tests.
        if self.candidate_ids is not None:
            ids = torch.as_tensor(self.candidate_ids, dtype=torch.long).flatten().unique()
            self._cand_set: Optional[set] = set(int(x) for x in ids.tolist())
        else:
            self._cand_set = None
        self._dump()

    def __call__(self, eval_preds: "EvalPrediction", compute_result: bool = True) -> Optional[dict[str, float]]:
        preds = numpify(eval_preds.predictions)  # [batch, seq_len, max_k]
        labels = numpify(eval_preds.label_ids)   # [batch, seq_len]

        if preds.ndim != 3:
            raise ValueError(
                f"ComputeRecallAtK expects predictions of shape [B, T, K], got shape {preds.shape}. "
                "Did you forget to set preprocess_logits_for_metrics=recall_at_k_logit_processor?"
            )

        cand_set = self._cand_set

        for topk_seq, label_seq in zip(preds, labels):
            # topk_seq: [seq_len, max_k]; label_seq: [seq_len]
            first_resp_idx = -1
            first_target_idx = -1

            for i in range(len(label_seq)):
                label_i = int(label_seq[i])
                if label_i == IGNORE_INDEX:
                    continue
                if first_resp_idx < 0:
                    first_resp_idx = i
                if cand_set is None or label_i in cand_set:
                    first_target_idx = i
                    break

            if first_target_idx < 0:
                # Has response but no token in candidate set -> OOV sample.
                if first_resp_idx >= 0:
                    self.oov_count += 1
                continue

            if first_target_idx <= 0:
                # Target starts at position 0 (no preceding token to predict from).
                continue

            label = int(label_seq[first_target_idx])
            topk_ids = topk_seq[first_target_idx - 1]  # predict-from-previous shift
            self.total_count += 1
            for k in self.k_list:
                if label in topk_ids[:k]:
                    self.hr_counts[k] += 1

        if compute_result:
            return self._dump()
        return None

