"""PyTorch datasets and collate functions for TokenSelection / LaneAssignment.

Both datasets are map-style with a per-line byte-offset index over the labeling
JSONL output. Random access per record allows
true shuffling with DataLoader(shuffle=True) while keeping memory bounded.

Pool features are looked up at collate time from a shared PoolRegistry
(see training/utils.py).
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .utils import (
    LANE_POOL_PREFIX_COLS,
    LANE_POOL_SPECTRAL_COLS,
    NYQUIST_HZ,
    PHASE_ENCODE,
    PoolRegistry,
    build_offset_index,
)


class _JsonlIndexed(Dataset):
    """Map-style Dataset over a JSONL file with a cached byte-offset index.

    Optionally filters records by `package_id` membership in `allowed_packages`.
    Because filtering requires parsing each record to check package_id, we
    perform a pre-scan pass once per split and cache the filtered index.
    """

    def __init__(
        self,
        jsonl_path: str,
        offset_cache_path: str,
        allowed_packages: Optional[Sequence[str]] = None,
        split_cache_path: Optional[str] = None,
    ) -> None:
        self.jsonl_path = jsonl_path
        self.offsets = build_offset_index(jsonl_path, offset_cache_path)

        if allowed_packages is not None:
            if split_cache_path and os.path.exists(split_cache_path):
                self.offsets = np.load(split_cache_path)
            else:
                allowed = set(allowed_packages)
                kept: List[int] = []
                with open(jsonl_path, "rb") as f:
                    for off in self.offsets:
                        f.seek(int(off))
                        line = f.readline()
                        if not line:
                            continue
                        rec = json.loads(line)
                        if rec.get("package_id") in allowed:
                            kept.append(int(off))
                self.offsets = np.asarray(kept, dtype=np.int64)
                if split_cache_path:
                    os.makedirs(os.path.dirname(split_cache_path) or ".", exist_ok=True)
                    np.save(split_cache_path, self.offsets)

        self._file = None  # opened lazily per worker

    def __len__(self) -> int:
        return int(self.offsets.shape[0])

    def _ensure_open(self) -> None:
        if self._file is None:
            self._file = open(self.jsonl_path, "rb")

    def __getitem__(self, idx: int) -> dict:
        self._ensure_open()
        off = int(self.offsets[idx])
        self._file.seek(off)
        line = self._file.readline()
        return json.loads(line)


def worker_init_fn(worker_id: int) -> None:
    """DataLoader worker initializer — each worker opens its own file handle."""
    info = torch.utils.data.get_worker_info()
    if info is not None:
        info.dataset._file = None  # force re-open in worker process


# ---------------------------------------------------------------------------
# TokenSelection dataset
# ---------------------------------------------------------------------------


class TokenSelectionDataset(_JsonlIndexed):
    """Dataset of TokenSelection records."""

    def __init__(
        self,
        jsonl_path: str,
        offset_cache_path: str,
        allowed_packages: Optional[Sequence[str]] = None,
        split_cache_path: Optional[str] = None,
    ) -> None:
        super().__init__(jsonl_path, offset_cache_path, allowed_packages, split_cache_path)


def token_collate(records: List[dict], pools: PoolRegistry) -> Dict[str, torch.Tensor]:
    """Concatenate multiple measures into a single flat batch.

    Output tensors (all shape `(sum_P,)` unless noted):
      measure   (sum_P, 4)
      pool      (sum_P, 14)
      context   (sum_P, 12)
      targets   (sum_P,)        — 0/1 per labeled token, 0 otherwise
      label_mask (sum_P,)       — 1 at labeled indices, 0 otherwise
      measure_id (sum_P,)       — int64, 0..len(records)-1
      notes_in_measure (M,)     — int64, per-record ground truth for top-K metric
    """
    M = len(records)
    measure_chunks = []
    pool_chunks = []
    context_chunks = []
    target_chunks = []
    mask_chunks = []
    mid_chunks = []
    notes_per_rec = np.zeros(M, dtype=np.int64)

    for m_idx, rec in enumerate(records):
        pkg_id = rec["package_id"]
        pool_14 = pools.get_features_14(pkg_id)  # (P, 14)
        storage = pools.get_storage(pkg_id)       # (P, 13); col 1 = attack_rms
        P = pool_14.shape[0]

        phase_enc = PHASE_ENCODE[rec["phase"]]
        measure_vec = np.array(
            [
                float(rec["measure"]),
                float(rec["density_rank"]),
                float(phase_enc),
                float(rec["notes_in_measure"]),
            ],
            dtype=np.float32,
        )
        measure_chunks.append(np.broadcast_to(measure_vec, (P, 4)).copy())
        pool_chunks.append(pool_14)

        # Build context tensor (4, 3) then flatten to (12,)
        ctx = np.zeros((4, 3), dtype=np.float32)
        for i, c in enumerate(rec.get("context", [])[:4]):
            if c.get("measure") is None:
                continue
            ctx[i, 0] = float(c.get("tkey_delta", 0) or 0)
            ctx[i, 1] = float(c.get("placed_count", 0) or 0)
            placed_idx = c.get("placed_pool_indices") or []
            if placed_idx:
                # col 1 of storage = attack_rms
                ctx[i, 2] = float(storage[np.asarray(placed_idx, dtype=np.int64), 1].mean())
        ctx_flat = ctx.reshape(12)
        context_chunks.append(np.broadcast_to(ctx_flat, (P, 12)).copy())

        # Dense label / mask aligned to pool_index
        t = np.zeros(P, dtype=np.float32)
        lm = np.zeros(P, dtype=np.float32)
        for pair in rec.get("labels", []):
            if len(pair) != 2:
                continue
            pi = int(pair[0])
            lbl = int(pair[1])
            if 0 <= pi < P:
                t[pi] = float(lbl)
                lm[pi] = 1.0
        target_chunks.append(t)
        mask_chunks.append(lm)
        mid_chunks.append(np.full(P, m_idx, dtype=np.int64))
        notes_per_rec[m_idx] = int(rec.get("notes_in_measure", 0))

    return {
        "measure": torch.from_numpy(np.concatenate(measure_chunks)),
        "pool": torch.from_numpy(np.concatenate(pool_chunks)),
        "context": torch.from_numpy(np.concatenate(context_chunks)),
        "targets": torch.from_numpy(np.concatenate(target_chunks)),
        "label_mask": torch.from_numpy(np.concatenate(mask_chunks)),
        "measure_id": torch.from_numpy(np.concatenate(mid_chunks)),
        "notes_in_measure": torch.from_numpy(notes_per_rec),
    }


# ---------------------------------------------------------------------------
# LaneAssignment dataset
# ---------------------------------------------------------------------------


class LaneAssignmentDataset(_JsonlIndexed):
    """Dataset of LaneAssignment records."""

    def __init__(
        self,
        jsonl_path: str,
        offset_cache_path: str,
        allowed_packages: Optional[Sequence[str]] = None,
        split_cache_path: Optional[str] = None,
    ) -> None:
        super().__init__(jsonl_path, offset_cache_path, allowed_packages, split_cache_path)


def lane_collate(records: List[dict], pools: PoolRegistry) -> Dict[str, torch.Tensor]:
    """Stack events into fixed-size batch tensors.

    Output tensors:
      event   (B, 16)
      context (B, 8, 5)
      mask    (B, 7)
      label   (B,)   int64, 0..6
    """
    B = len(records)
    event = np.zeros((B, 16), dtype=np.float32)
    context = np.zeros((B, 8, 5), dtype=np.float32)
    mask = np.zeros((B, 7), dtype=np.float32)
    label = np.zeros(B, dtype=np.int64)

    for i, rec in enumerate(records):
        pkg_id = rec["package_id"]
        storage = pools.get_storage(pkg_id)  # (P, 13)
        pi = int(rec["pool_index"])

        # Event feature row
        # cols 0..5 from pool (duration, attack_rms, attack_peak, intensity_origin, key_occ, scratch_occ)
        for dst, src in enumerate(LANE_POOL_PREFIX_COLS):
            event[i, dst] = storage[pi, src]
        event[i, 6] = float(rec["idx192"])
        event[i, 7] = float(rec["density_rank"])
        event[i, 8] = float(PHASE_ENCODE[rec["phase"]])
        event[i, 9] = float(rec.get("total_placed_key_notes", 0))
        # cols 10..15: 6 spectral from pool (storage cols 7..12); normalize centroid mean/std
        for dst_off, src in enumerate(LANE_POOL_SPECTRAL_COLS):
            val = float(storage[pi, src])
            if src in (7, 8):  # spectral_centroid_mean / std
                val /= NYQUIST_HZ
            event[i, 10 + dst_off] = val

        # Context
        ctx_list = rec.get("context", [])[:8]
        for k, c in enumerate(ctx_list):
            if c.get("tkey_delta") is None:
                context[i, k, 4] = 1.0  # is_padded
                continue
            context[i, k, 0] = float(c["tkey_delta"])
            context[i, k, 1] = float(c.get("lane", 0) or 0)
            cpi = c.get("pool_index")
            if cpi is not None:
                context[i, k, 2] = float(storage[int(cpi), 1])  # attack_rms col
            context[i, k, 3] = float(c.get("idx192", 0) or 0)
            context[i, k, 4] = 0.0

        # Mask
        al = rec.get("available_lanes", [])
        for k in range(min(7, len(al))):
            mask[i, k] = float(al[k])

        # Label (stored as 1..7, convert to 0..6)
        label[i] = int(rec["label"]) - 1

    return {
        "event": torch.from_numpy(event),
        "context": torch.from_numpy(context),
        "mask": torch.from_numpy(mask),
        "label": torch.from_numpy(label),
    }
