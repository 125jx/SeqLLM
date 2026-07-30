# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""SID (semantic-id) constraint masker (shared by the SFT / OneRec trainers).

Implements the "masked cross-entropy over legal candidate sets" scheme: the
model forward is UNCHANGED (one pass over all positions producing full-vocab
logits / hidden_states); only the LOSS is altered. For every SID slot we look
up -- from the *current sample's real prefix* -- the legal candidate token set
and renormalize the softmax to that set only (set illegal logits to a large
negative value before CE). This prevents the model from spending probability
mass on item combinations that never exist.

Slot recovery (no extra metadata needed -- read straight from input_ids/labels)
-------------------------------------------------------------------------------
The pre-shifted OneRec labels satisfy ``labels[t] == input_ids[t+1]`` on loss
positions, and a SID is laid out contiguously as
``<|sid_begin|><s_a><s_b><s_c><|sid_end|>``. Therefore, for a loss position t:

  * ``labels[t]`` in the s_a range  -> predicting **s1**.
        legal set = ``valid_s1``.
  * ``labels[t]`` in the s_b range  -> predicting **s2**.
        the true s1 is ``input_ids[t]`` (the s_a token just fed).
        legal set = ``a2b[s1]``.
  * ``labels[t]`` in the s_c range  -> predicting **s3**.
        the true s2 is ``input_ids[t]`` (s_b), the true s1 is
        ``input_ids[t-1]`` (s_a).
        legal set = ``ab2c[(s1, s2)]``  (``s3_mode = "csr"``)
                 or ``valid_s3``         (``s3_mode = "valid_s3"``).
  * anything else (text tokens, ``<|sid_begin|>``/``<|sid_end|>``) -> standard
    full-vocab CE (unconstrained).

Two third-position modes (parameter switch ``s3_mode``)
-------------------------------------------------------
  * ``"csr"``      : tight per-prefix constraint via the ``ab2c`` CSR table.
                     55% of prefixes have a single candidate, so this often
                     yields ~0 loss on s3 (the design-doc risk). Use when you
                     want the strict constraint.
  * ``"valid_s3"`` : loose constraint -- s3 only has to fall inside the global
                     set of third-position codes that ever appeared
                     (``valid_s3``). Keeps a real discriminative signal on s3
                     while still forbidding never-seen codes. This is the
                     recommended default per the design ("重点约束前两位 SID,
                     第三位不做过细限制").

Efficiency / multi-GPU
----------------------
All tables are constant lookup tensors replicated per rank (loaded from the
``.npz`` built by ``scripts/build_sid_constraints.py`` and moved to the rank's
device on first use). The masked CE is a pure, differentiable, per-rank
function of the logits -- it plugs straight into the existing
``ChunkedLossComputer`` / FSDP gradient machinery with NO extra collectives,
so DDP / FSDP / single-GPU all work unchanged. Candidate masks are built only
over the 8192-wide slot sub-range (never the full 176k vocab) and only for the
SID loss positions, so memory stays small.
"""

from __future__ import annotations

import json
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


# Large finite negative bias for illegal candidates (avoid -inf so an all-masked
# row -- which never happens because the target is always legal -- can't nan).
NEG_BIAS = -1.0e9


class SidConstraintMasker:
    """Loads SID constraint tables and computes masked cross-entropy.

    Args:
        path: path to the ``.npz`` produced by ``build_sid_constraints.py``.
        s3_mode: ``"csr"`` (tight ab2c) or ``"valid_s3"`` (loose global set).
        ignore_index: label id to ignore (default -100).
        a_base/b_base/c_base/slot_size: vocab layout. If None, read from the
            ``meta`` field stored in the npz.
    """

    def __init__(
        self,
        path: str,
        s3_mode: str = "valid_s3",
        ignore_index: int = -100,
        a_base: Optional[int] = None,
        b_base: Optional[int] = None,
        c_base: Optional[int] = None,
        slot_size: Optional[int] = None,
    ) -> None:
        s3_mode = str(s3_mode).lower()
        if s3_mode == "ab2c":  # alias: ab2c == csr (tight per-prefix ab2c table)
            s3_mode = "csr"
        if s3_mode not in ("csr", "valid_s3"):
            raise ValueError(
                f"[SidConstraint] s3_mode must be 'csr'/'ab2c' or 'valid_s3', got {s3_mode!r}"
            )
        self.s3_mode = s3_mode
        self.ignore_index = int(ignore_index)
        self.path = path

        data = np.load(path, allow_pickle=False)
        meta = {}
        if "meta" in data:
            try:
                meta = json.loads(str(data["meta"]))
            except Exception:  # noqa: BLE001
                meta = {}

        self.slot_size = int(slot_size if slot_size is not None else meta.get("slot_size", 8192))
        self.a_base = int(a_base if a_base is not None else meta.get("a_base", 151669))
        self.b_base = int(b_base if b_base is not None else meta.get("b_base", 159861))
        self.c_base = int(c_base if c_base is not None else meta.get("c_base", 168053))
        self.a_hi = self.a_base + self.slot_size
        self.b_hi = self.b_base + self.slot_size
        self.c_hi = self.c_base + self.slot_size

        # ---- valid_s1 / valid_s3 as boolean [slot_size] masks (additive bias) ----
        valid_s1 = torch.from_numpy(np.asarray(data["valid_s1"], dtype=np.int64))
        valid_s3 = torch.from_numpy(np.asarray(data["valid_s3"], dtype=np.int64))
        s1_bias = torch.full((self.slot_size,), NEG_BIAS, dtype=torch.float32)
        s1_bias[valid_s1] = 0.0
        s3_bias = torch.full((self.slot_size,), NEG_BIAS, dtype=torch.float32)
        s3_bias[valid_s3] = 0.0
        self._s1_bias_cpu = s1_bias
        self._s3_bias_cpu = s3_bias

        # ---- a2b CSR ----
        self._a2b_indptr_cpu = torch.from_numpy(np.asarray(data["a2b_indptr"], dtype=np.int64))
        self._a2b_indices_cpu = torch.from_numpy(np.asarray(data["a2b_indices"], dtype=np.int64))

        # ---- ab2c CSR (sparse keys) ----
        self._ab2c_keys_cpu = torch.from_numpy(np.asarray(data["ab2c_keys"], dtype=np.int64))
        self._ab2c_indptr_cpu = torch.from_numpy(np.asarray(data["ab2c_indptr"], dtype=np.int64))
        self._ab2c_indices_cpu = torch.from_numpy(np.asarray(data["ab2c_indices"], dtype=np.int64))

        self._device: Optional[torch.device] = None  # set on first .to()

    # ----------------------- device management -----------------------
    def to(self, device: torch.device) -> "SidConstraintMasker":
        if self._device is not None and self._device == device:
            return self
        self.s1_bias = self._s1_bias_cpu.to(device)
        self.s3_bias = self._s3_bias_cpu.to(device)
        self.a2b_indptr = self._a2b_indptr_cpu.to(device)
        self.a2b_indices = self._a2b_indices_cpu.to(device)
        self.ab2c_keys = self._ab2c_keys_cpu.to(device)
        self.ab2c_indptr = self._ab2c_indptr_cpu.to(device)
        self.ab2c_indices = self._ab2c_indices_cpu.to(device)
        self._device = device
        return self

    # ----------------------- CSR helpers -----------------------
    @staticmethod
    def _csr_ranges_to_pairs(
        starts: torch.Tensor, ends: torch.Tensor, indices: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Expand variable-length CSR rows into flat (row_id, value) pairs.

        Given per-row ``starts`` / ``ends`` (into ``indices``), returns
        ``(row_ids, values)`` where ``row_ids[k]`` is the local row index in
        ``[0, len(starts))`` and ``values[k]`` is the corresponding entry from
        ``indices``. Fully vectorized (no python loop).
        """
        counts = ends - starts  # [m]
        total = int(counts.sum().item())
        if total == 0:
            empty = torch.empty(0, dtype=torch.long, device=starts.device)
            return empty, empty
        m = starts.shape[0]
        row_ids = torch.repeat_interleave(
            torch.arange(m, device=starts.device), counts
        )
        # exclusive prefix sum of counts = flat offset where each row begins
        excl = torch.cumsum(counts, 0) - counts  # [m]
        within = torch.arange(total, device=starts.device) - torch.repeat_interleave(excl, counts)
        flat_idx = torch.repeat_interleave(starts, counts) + within
        values = indices[flat_idx]
        return row_ids, values

    def _build_dense_csr_mask(
        self,
        keys: torch.Tensor,
        indptr: torch.Tensor,
        indices: torch.Tensor,
        valid_row: torch.Tensor,
    ) -> torch.Tensor:
        """Build an additive ``[m, slot_size]`` bias from CSR rows.

        ``keys[i]`` is the CSR row index for position i (already resolved).
        ``valid_row[i]`` says whether position i has a real CSR row; rows with
        ``valid_row=False`` are left UNCONSTRAINED (bias 0 everywhere -> the
        whole slot sub-range is allowed) as a safe fallback.
        """
        m = keys.shape[0]
        mask = torch.full(
            (m, self.slot_size), NEG_BIAS, dtype=torch.float32, device=keys.device
        )
        # Unconstrained fallback rows: allow everything in the slot.
        if (~valid_row).any():
            mask[~valid_row] = 0.0
        vpos = valid_row.nonzero(as_tuple=False).squeeze(1)
        if vpos.numel() > 0:
            rows = keys[vpos]
            starts = indptr[rows]
            ends = indptr[rows + 1]
            local_row, cols = self._csr_ranges_to_pairs(starts, ends, indices)
            if local_row.numel() > 0:
                mask[vpos[local_row], cols] = 0.0
        return mask

    # ----------------------- main entry -----------------------
    def masked_token_loss(
        self,
        logits_flat: torch.Tensor,
        labels_flat: torch.Tensor,
        cur_ids_flat: torch.Tensor,
        prev_ids_flat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute masked per-token CE.

        Args:
            logits_flat: ``[n, vocab]`` logits (will be cast to float32).
            labels_flat: ``[n]`` pre-shifted OneRec labels (target token id or
                ``ignore_index``).
            cur_ids_flat: ``[n]`` input_ids at the SAME position as the logits
                (i.e. ``input_ids[t]``). Used to recover s1 (for s2 slots) and
                s2 (for s3 slots).
            prev_ids_flat: ``[n]`` input_ids at the PREVIOUS position
                (``input_ids[t-1]``). Used to recover s1 for s3 slots.

        Returns:
            ``(avg_loss, per_token_loss)`` matching the
            :class:`CrossEntropyLoss` contract used by ChunkedLossComputer:
            ``avg_loss`` is the mean over valid tokens (differentiable),
            ``per_token_loss`` is ``[n]`` (0 on ignored positions).
        """
        device = logits_flat.device
        if self._device is None or self._device != device:
            self.to(device)

        # NOTE: we do NOT cast the whole [n, vocab] logits to float here (that
        # would double the memory of the full-vocab logits tensor). Instead each
        # branch casts only its sliced sub-tensor to float32 before CE.
        n = labels_flat.shape[0]
        per_token = torch.zeros(n, dtype=torch.float32, device=device)  # tracks grad via index_put

        valid = labels_flat != self.ignore_index
        in_a = (labels_flat >= self.a_base) & (labels_flat < self.a_hi)
        in_b = (labels_flat >= self.b_base) & (labels_flat < self.b_hi)
        in_c = (labels_flat >= self.c_base) & (labels_flat < self.c_hi)

        is_s1 = valid & in_a
        is_s2 = valid & in_b
        is_s3 = valid & in_c
        is_text = valid & (~in_a) & (~in_b) & (~in_c)

        # ---- text positions: standard full-vocab CE ----
        if is_text.any():
            idx = is_text.nonzero(as_tuple=False).squeeze(1)
            ce = F.cross_entropy(
                logits_flat[idx].float(), labels_flat[idx], reduction="none"
            )
            per_token = per_token.index_put((idx,), ce)

        # ---- s1 positions: restrict to valid_s1 ----
        if is_s1.any():
            idx = is_s1.nonzero(as_tuple=False).squeeze(1)
            slot_logits = logits_flat[idx, self.a_base : self.a_hi].float()
            masked = slot_logits + self.s1_bias  # broadcast [slot_size]
            tgt = labels_flat[idx] - self.a_base
            ce = F.cross_entropy(masked, tgt, reduction="none")
            per_token = per_token.index_put((idx,), ce)

        # ---- s2 positions: restrict to a2b[s1] ----
        if is_s2.any():
            idx = is_s2.nonzero(as_tuple=False).squeeze(1)
            slot_logits = logits_flat[idx, self.b_base : self.b_hi].float()
            s1code = cur_ids_flat[idx] - self.a_base
            valid_prefix = (s1code >= 0) & (s1code < self.slot_size)
            s1code = s1code.clamp(0, self.slot_size - 1)
            bias = self._build_dense_csr_mask(
                s1code, self.a2b_indptr, self.a2b_indices, valid_prefix
            )
            masked = slot_logits + bias
            tgt = labels_flat[idx] - self.b_base
            ce = F.cross_entropy(masked, tgt, reduction="none")
            per_token = per_token.index_put((idx,), ce)

        # ---- s3 positions: csr (ab2c) or loose valid_s3 ----
        if is_s3.any():
            idx = is_s3.nonzero(as_tuple=False).squeeze(1)
            slot_logits = logits_flat[idx, self.c_base : self.c_hi].float()
            tgt = labels_flat[idx] - self.c_base
            if self.s3_mode == "valid_s3":
                masked = slot_logits + self.s3_bias  # broadcast [slot_size]
            else:  # csr
                s2code = cur_ids_flat[idx] - self.b_base
                s1code = prev_ids_flat[idx] - self.a_base
                valid_prefix = (
                    (s1code >= 0)
                    & (s1code < self.slot_size)
                    & (s2code >= 0)
                    & (s2code < self.slot_size)
                )
                key = s1code.clamp(0, self.slot_size - 1) * self.slot_size + s2code.clamp(
                    0, self.slot_size - 1
                )
                # locate row via searchsorted on the sorted ab2c_keys
                row = torch.searchsorted(self.ab2c_keys, key)
                row = row.clamp(0, self.ab2c_keys.shape[0] - 1)
                found = self.ab2c_keys[row] == key
                valid_row = valid_prefix & found
                bias = self._build_dense_csr_mask(
                    row, self.ab2c_indptr, self.ab2c_indices, valid_row
                )
                masked = slot_logits + bias
            ce = F.cross_entropy(masked, tgt, reduction="none")
            per_token = per_token.index_put((idx,), ce)

        total = valid.sum()
        if total > 0:
            avg = per_token.sum() / total
        else:
            # No valid tokens in this micro-batch shard (e.g. an all-prompt /
            # all-padding shard under data parallelism). Return a GRAD-CONNECTED
            # zero (tied to logits) -- NOT a bare constant -- so loss.backward()
            # still builds a graph and produces (zero) grads on EVERY rank. This
            # keeps the DDP / DeepSpeed gradient all-reduce collective-consistent
            # across ranks and avoids "element 0 does not require grad" errors.
            avg = logits_flat.float().sum() * 0.0
        return avg, per_token

    def masked_loss_for_chunk(
        self,
        logits_flat: torch.Tensor,
        labels_flat: torch.Tensor,
        input_ids: torch.Tensor,
        start: int,
        end: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convenience wrapper for the chunked path (batch size == 1).

        Slices ``cur`` = ``input_ids[0, start:end]`` and ``prev`` =
        ``input_ids[0, start-1:end-1]`` (with a dummy 0 prepended when
        ``start == 0``) so they line up row-for-row with ``logits_flat`` /
        ``labels_flat``.
        """
        ids = input_ids[0]
        cur = ids[start:end]
        if start == 0:
            prev = torch.cat([ids.new_zeros(1), ids[0 : max(end - 1, 0)]])
        else:
            prev = ids[start - 1 : end - 1]
        # Defensive length alignment (chunk at sequence tail).
        nlog = labels_flat.shape[0]
        if cur.shape[0] != nlog:
            cur = cur[:nlog]
        if prev.shape[0] != nlog:
            prev = prev[:nlog]
        return self.masked_token_loss(logits_flat, labels_flat, cur, prev)
