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

"""
Metrics for pretraining evaluation, including HR@K for behavior prediction.
"""

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch

from ...extras.constants import IGNORE_INDEX


if TYPE_CHECKING:
    from transformers import EvalPrediction, PreTrainedTokenizer


# Global variables for HR@K computation.
# - `_BEH_TOKEN_IDS`: explicit list of candidate token ids (global vocab ids).
# - `_BEH_RANGE_START`: if set, treat token ids in [start, vocab_size) as behavior tokens.
#   This is useful when we don't want to ship a giant beh_tokens.json; we can infer candidates
#   from `original_vocab_size` (base vocab size) and the current logits' vocab_size.
# - `_EOS_TOKEN_ID`: if set, skip evaluating positions whose ground-truth label is EOS.
_BEH_TOKEN_IDS: Optional[torch.Tensor] = None
_BEH_RANGE_START: Optional[int] = None
_EOS_TOKEN_ID: Optional[int] = None
_MAX_K: int = 50


def set_beh_token_ids(
    beh_token_ids: Optional[torch.Tensor],
    max_k: int = 50,
    beh_range_start: Optional[int] = None,
    eos_token_id: Optional[int] = None,
):
    """Set global settings for HR@K logit processor and metric."""
    global _BEH_TOKEN_IDS, _BEH_RANGE_START, _EOS_TOKEN_ID, _MAX_K
    _BEH_TOKEN_IDS = beh_token_ids
    _BEH_RANGE_START = int(beh_range_start) if beh_range_start is not None else None
    _EOS_TOKEN_ID = int(eos_token_id) if eos_token_id is not None else None
    _MAX_K = int(max_k)


def hit_rate_logit_processor(logits: "torch.Tensor", labels: "torch.Tensor") -> "torch.Tensor":
    """
    Process logits to extract top-k token IDs within beh tokens only.
    Returns: [batch, seq_len, max_k] - top-k token IDs at each position
    
    This is memory efficient: instead of saving [batch, seq_len, vocab_size] logits,
    we only save [batch, seq_len, max_k] token IDs.
    """
    global _BEH_TOKEN_IDS, _MAX_K
    
    if isinstance(logits, (list, tuple)):
        if logits[0].dim() == 3:
            logits = logits[0]
        else:
            logits = logits[1]
    
    if logits.dim() != 3:
        raise ValueError("Cannot process the logits.")
    
    batch_size, seq_len, vocab_size = logits.shape
    device = logits.device
    
    if _BEH_TOKEN_IDS is not None:
        # Only consider beh tokens (explicit id list)
        beh_ids = _BEH_TOKEN_IDS.to(device)
        # Extract logits for beh tokens: [batch, seq_len, num_beh_tokens]
        beh_logits = logits[:, :, beh_ids]
        # Get top-k within beh tokens
        _, topk_local_indices = torch.topk(beh_logits, min(_MAX_K, beh_logits.size(-1)), dim=-1)
        # Convert local indices back to global token IDs
        topk_token_ids = beh_ids[topk_local_indices]
    elif _BEH_RANGE_START is not None and 0 <= int(_BEH_RANGE_START) < int(vocab_size):
        # Only consider beh tokens in [range_start, vocab_size)
        beh_start = int(_BEH_RANGE_START)
        beh_ids = torch.arange(beh_start, int(vocab_size), device=device, dtype=torch.long)
        beh_logits = logits[:, :, beh_ids]
        _, topk_local_indices = torch.topk(beh_logits, min(_MAX_K, beh_logits.size(-1)), dim=-1)
        topk_token_ids = beh_ids[topk_local_indices]
    else:
        # Consider all tokens
        _, topk_token_ids = torch.topk(logits, _MAX_K, dim=-1)
    
    return topk_token_ids  # [batch, seq_len, max_k]


def numpify(inputs):
    if isinstance(inputs, torch.Tensor):
        inputs = inputs.cpu().numpy()
    return inputs


@dataclass
class ComputeHitRate:
    """
    Compute HR@K for behavior prediction.
    Only compute softmax on beh tokens (not full vocabulary).
    
    Expects predictions from hit_rate_logit_processor: [batch, seq_len, max_k]
    
    Args:
        k_list: List of K values for HR@K
        last_position_only: If True, only evaluate at the last position (like SASRec/pretraining project).
                           If False, evaluate at all positions (LLM-style autoregressive evaluation).
    """
    
    k_list: list[int] = field(default_factory=lambda: [1, 5, 10, 50])
    last_position_only: bool = False  # 默认全序列评估，设为 True 则只评估最后一个位置
    
    def _dump(self) -> Optional[dict[str, float]]:
        result = None
        if hasattr(self, "hr_counts") and hasattr(self, "total_count"):
            if self.total_count > 0:
                # Distributed aggregation
                if torch.distributed.is_initialized():
                    # Aggregate counts across all processes
                    total_tensor = torch.tensor([self.total_count], dtype=torch.float64, device="cuda")
                    hr_tensors = {k: torch.tensor([self.hr_counts[k]], dtype=torch.float64, device="cuda") for k in self.k_list}
                    
                    torch.distributed.all_reduce(total_tensor, op=torch.distributed.ReduceOp.SUM)
                    for k in self.k_list:
                        torch.distributed.all_reduce(hr_tensors[k], op=torch.distributed.ReduceOp.SUM)
                    
                    total_count = total_tensor.item()
                    hr_counts = {k: hr_tensors[k].item() for k in self.k_list}
                else:
                    total_count = self.total_count
                    hr_counts = self.hr_counts
                
                if total_count > 0:
                    # MLflow metric names cannot contain '@', so we use 'HR_at_K' format.
                    result = {f"HR_at_{k}": hr_counts[k] / total_count for k in self.k_list}
                    result["total_predictions"] = float(total_count)
        
        # Reset counters
        self.hr_counts = {k: 0 for k in self.k_list}
        self.total_count = 0
        return result
    
    def __post_init__(self):
        self._dump()
    
    def __call__(self, eval_preds: "EvalPrediction", compute_result: bool = True) -> Optional[dict[str, float]]:
        """
        Compute HR@K from predictions.
        
        Args:
            eval_preds: 
                predictions: [batch, seq_len, max_k] - top-k token IDs from hit_rate_logit_processor
                label_ids: [batch, seq_len] - ground truth token IDs
            compute_result: Whether to return the final result
        """
        preds = numpify(eval_preds.predictions)  # [batch, seq_len, max_k]
        labels = numpify(eval_preds.label_ids)   # [batch, seq_len]
        
        # predictions shape: [batch, seq_len, max_k]
        # labels shape: [batch, seq_len]
        
        global _BEH_RANGE_START, _EOS_TOKEN_ID
        beh_start = int(_BEH_RANGE_START) if _BEH_RANGE_START is not None else None
        eos_id = int(_EOS_TOKEN_ID) if _EOS_TOKEN_ID is not None else None

        for topk_seq, label_seq in zip(preds, labels):
            # topk_seq: [seq_len, max_k]
            # label_seq: [seq_len]
            
            if self.last_position_only:
                # 只评估最后一个位置（类似 SASRec/pretraining 项目）
                # 找到最后一个非 IGNORE_INDEX 的位置
                last_valid_idx = -1
                for i in range(len(label_seq) - 1, -1, -1):
                    y = int(label_seq[i])
                    if y == IGNORE_INDEX:
                        continue
                    if eos_id is not None and y == eos_id:
                        continue
                    if beh_start is not None and y < beh_start:
                        continue
                    last_valid_idx = i
                    break
                
                if last_valid_idx > 0:  # 需要至少有一个前置位置来预测
                    label = int(label_seq[last_valid_idx])
                    topk_ids = topk_seq[last_valid_idx - 1]  # 用前一个位置预测当前
                    self.total_count += 1
                    
                    for k in self.k_list:
                        if label in topk_ids[:k]:
                            self.hr_counts[k] += 1
            else:
                # 全序列评估（LLM 风格）
                # Shift: predict next token
                # topk_seq[:-1] predicts label_seq[1:]
                for i in range(len(label_seq) - 1):
                    label = int(label_seq[i + 1])
                    if label == IGNORE_INDEX:
                        continue
                    if eos_id is not None and label == eos_id:
                        continue
                    if beh_start is not None and label < beh_start:
                        continue
                    
                    topk_ids = topk_seq[i]  # [max_k]
                    self.total_count += 1
                    
                    for k in self.k_list:
                        if label in topk_ids[:k]:
                            self.hr_counts[k] += 1
        
        if compute_result:
            return self._dump()
        return None


def dual_hit_rate_logit_processor(logits: "torch.Tensor", labels: "torch.Tensor") -> tuple:
    """Top-k extractor for the triplet dual-head model.

    `logits` is a tuple (item_logits, cat_logits), each [batch, seq, vocab].
    Returns (item_topk, cat_topk), each [batch, seq, max_k] of token ids.
    """
    global _MAX_K
    if not isinstance(logits, (list, tuple)) or len(logits) < 2:
        raise ValueError("dual_hit_rate_logit_processor expects (item_logits, cat_logits).")
    item_logits, cat_logits = logits[0], logits[1]
    _, item_topk = torch.topk(item_logits, min(_MAX_K, item_logits.size(-1)), dim=-1)
    _, cat_topk = torch.topk(cat_logits, min(_MAX_K, cat_logits.size(-1)), dim=-1)
    return item_topk, cat_topk


def _hr_counts_for_stream(preds, labels, k_list, last_position_only, eos_id):
    """Compute hit/total counts for one stream. Returns (hr_counts dict, total)."""
    hr_counts = {k: 0 for k in k_list}
    total = 0
    for topk_seq, label_seq in zip(preds, labels):
        if last_position_only:
            last_valid_idx = -1
            for i in range(len(label_seq) - 1, -1, -1):
                y = int(label_seq[i])
                if y == IGNORE_INDEX:
                    continue
                if eos_id is not None and y == eos_id:
                    continue
                last_valid_idx = i
                break
            if last_valid_idx > 0:
                label = int(label_seq[last_valid_idx])
                topk_ids = topk_seq[last_valid_idx - 1]
                total += 1
                for k in k_list:
                    if label in topk_ids[:k]:
                        hr_counts[k] += 1
        else:
            for i in range(len(label_seq) - 1):
                label = int(label_seq[i + 1])
                if label == IGNORE_INDEX:
                    continue
                if eos_id is not None and label == eos_id:
                    continue
                topk_ids = topk_seq[i]
                total += 1
                for k in k_list:
                    if label in topk_ids[:k]:
                        hr_counts[k] += 1
    return hr_counts, total


@dataclass
class ComputeDualHitRate:
    """Compute HR@K for BOTH item-id and category-id heads of the triplet model.

    predictions: tuple (item_topk, cat_topk), each [batch, seq, max_k]
    label_ids  : tuple (item_labels, cat_labels), each [batch, seq]
    Reports HR_item_at_k and HR_cat_at_k.
    """

    k_list: list[int] = field(default_factory=lambda: [1, 5, 10, 50])
    last_position_only: bool = False
    item_eos_id: Optional[int] = None

    def _reset(self):
        self.item_hr = {k: 0 for k in self.k_list}
        self.cat_hr = {k: 0 for k in self.k_list}
        self.item_total = 0
        self.cat_total = 0

    def __post_init__(self):
        self._reset()

    def _dump(self) -> Optional[dict[str, float]]:
        result = None
        if self.item_total > 0 or self.cat_total > 0:
            if torch.distributed.is_initialized():
                def _ar(val):
                    t = torch.tensor([val], dtype=torch.float64, device="cuda")
                    torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
                    return t.item()

                item_total = _ar(self.item_total)
                cat_total = _ar(self.cat_total)
                item_hr = {k: _ar(self.item_hr[k]) for k in self.k_list}
                cat_hr = {k: _ar(self.cat_hr[k]) for k in self.k_list}
            else:
                item_total, cat_total = self.item_total, self.cat_total
                item_hr, cat_hr = self.item_hr, self.cat_hr

            result = {}
            if item_total > 0:
                for k in self.k_list:
                    result[f"HR_item_at_{k}"] = item_hr[k] / item_total
                result["item_total_predictions"] = float(item_total)
            if cat_total > 0:
                for k in self.k_list:
                    result[f"HR_cat_at_{k}"] = cat_hr[k] / cat_total
                result["cat_total_predictions"] = float(cat_total)

        self._reset()
        return result

    def __call__(self, eval_preds: "EvalPrediction", compute_result: bool = True) -> Optional[dict[str, float]]:
        preds = eval_preds.predictions
        labels = eval_preds.label_ids
        item_topk, cat_topk = numpify(preds[0]), numpify(preds[1])
        item_labels, cat_labels = numpify(labels[0]), numpify(labels[1])

        item_hr, item_total = _hr_counts_for_stream(
            item_topk, item_labels, self.k_list, self.last_position_only, self.item_eos_id
        )
        cat_hr, cat_total = _hr_counts_for_stream(
            cat_topk, cat_labels, self.k_list, self.last_position_only, None
        )
        for k in self.k_list:
            self.item_hr[k] += item_hr[k]
            self.cat_hr[k] += cat_hr[k]
        self.item_total += item_total
        self.cat_total += cat_total

        if compute_result:
            return self._dump()
        return None


def load_beh_token_ids(tokenizer: "PreTrainedTokenizer", beh_tokens_file: str) -> torch.Tensor:
    """Load beh token IDs from file."""
    with open(beh_tokens_file, 'r') as f:
        data = json.load(f)
    
    beh_tokens = data['beh_tokens']
    beh_token_ids = []
    for token in beh_tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id != tokenizer.unk_token_id:
            beh_token_ids.append(token_id)
    
    print(f"Loaded {len(beh_token_ids)} beh token IDs for HR@K evaluation")
    return torch.tensor(beh_token_ids)

