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

"""Triplet-fusion GPT-2 for the pure-transformer pretraining (B 方案).

Input side : each position is one interaction, embedded as the SUM of three tables
    emb_t = wte[item_t] + E_cat[category_t] + E_rating[rating_t]

Output side: two prediction heads computed from the same hidden states
    - item head     : the standard (tied) `lm_head` over the item vocab  -> next-item
    - category head : a new `category_head` over the category vocab       -> next-category

Both heads are trained jointly: loss = loss_item + cat_loss_weight * loss_cat.
This lets eval report HR@K for BOTH item-id and category-id.

The two extra config fields (`cat_vocab_size`, `rating_vocab_size`) are persisted into
config.json so `from_pretrained` can rebuild the heads/embeddings at eval time.
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel
from transformers.utils import ModelOutput


@dataclass
class TripletFusionOutput(ModelOutput):
    """Output carrying both heads' logits so eval can compute two HR@K metrics."""

    loss: Optional[torch.FloatTensor] = None
    item_logits: Optional[torch.FloatTensor] = None
    cat_logits: Optional[torch.FloatTensor] = None


class TripletFusionGPT2LMHeadModel(GPT2LMHeadModel):
    """GPT-2 with (item, category, rating) input fusion and dual (item + category) heads."""

    def __init__(self, config):
        super().__init__(config)
        cat_vocab_size = int(getattr(config, "cat_vocab_size"))
        rating_vocab_size = int(getattr(config, "rating_vocab_size"))
        self.cat_vocab_size = cat_vocab_size
        self.rating_vocab_size = rating_vocab_size
        self.cat_loss_weight = float(getattr(config, "triplet_cat_loss_weight", 1.0))
        self.item_loss_weight = float(getattr(config, "triplet_item_loss_weight", 1.0))

        n_embd = config.n_embd
        # padding_idx=0 keeps the pad row at zero so padded/eos positions add nothing.
        self.cat_embeddings = nn.Embedding(cat_vocab_size, n_embd, padding_idx=0)
        self.rating_embeddings = nn.Embedding(rating_vocab_size, n_embd, padding_idx=0)
        self.category_head = nn.Linear(n_embd, cat_vocab_size, bias=False)

        self._init_triplet_weights(config.initializer_range)

    def _init_triplet_weights(self, std: float):
        for module in (self.cat_embeddings, self.rating_embeddings):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        self.category_head.weight.data.normal_(mean=0.0, std=std)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        category_ids: Optional[torch.LongTensor] = None,
        rating_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        labels_cat: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        num_items_in_batch: Optional[int] = None,
        **kwargs,
    ) -> TripletFusionOutput:
        # Build fused input embeddings. category/rating default to pad(0) if not provided.
        item_embeds = self.transformer.wte(input_ids)
        inputs_embeds = item_embeds
        if category_ids is not None:
            inputs_embeds = inputs_embeds + self.cat_embeddings(category_ids)
        if rating_ids is not None:
            inputs_embeds = inputs_embeds + self.rating_embeddings(rating_ids)

        transformer_outputs = self.transformer(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            return_dict=True,
        )
        hidden_states = transformer_outputs.last_hidden_state

        item_logits = self.lm_head(hidden_states)
        cat_logits = self.category_head(hidden_states)

        # IMPORTANT: match HF's ForCausalLMLoss convention. When the Trainer passes
        # `num_items_in_batch` (it does, because this forward accepts **kwargs), we MUST
        # use sum-reduction divided by num_items_in_batch. Otherwise a mean loss gets
        # scaled by num_processes under DDP (loss inflates ~world_size×, grad always clipped).
        reduction = "sum" if num_items_in_batch is not None else "mean"
        loss_fct = nn.CrossEntropyLoss(reduction=reduction)

        def _ce(logits, lab):
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = lab[..., 1:].contiguous()
            l = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            if num_items_in_batch is not None:
                l = l / num_items_in_batch
            return l

        # Per-head weights let you do single-task training:
        #   item-only     : triplet_item_loss_weight=1, triplet_cat_loss_weight=0
        #   category-only : triplet_item_loss_weight=0, triplet_cat_loss_weight=1
        loss = None
        if labels is not None and self.item_loss_weight != 0.0:
            loss = self.item_loss_weight * _ce(item_logits, labels)
        if labels_cat is not None and self.cat_loss_weight != 0.0:
            loss_cat = self.cat_loss_weight * _ce(cat_logits, labels_cat)
            loss = loss_cat if loss is None else loss + loss_cat

        return TripletFusionOutput(loss=loss, item_logits=item_logits, cat_logits=cat_logits)
