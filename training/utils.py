"""Shared utilities for training TokenSelection and LaneAssignment models.

Covers pool table loading, phase encoding, and the column-order translation
between the on-disk pool table and the inference
tensor layout.
"""

from __future__ import annotations

import json
import os
import random
from typing import Dict, List, Tuple

import numpy as np
import torch


PHASE_ENCODE = {"rush": 2, "normal": 1, "rest": 0}

# Pool table on-disk column order
POOL_STORAGE_COLUMNS = [
    "duration_ms",            # 0
    "attack_rms",             # 1
    "attack_peak",            # 2
    "intensity_origin",       # 3
    "key_occurrence",         # 4
    "scratch_occurrence",     # 5
    "bgm_occurrence",         # 6
    "spectral_centroid_mean", # 7
    "spectral_centroid_std",  # 8
    "spectral_flatness_mean", # 9
    "low_freq_energy_ratio",  # 10
    "zero_crossing_rate_mean",# 11
    "zero_crossing_rate_std", # 12
]

# Inference pool tensor column order, 14 cols including whitelist_pass
#   cols 0..6 identical to storage
#   col  7     = whitelist_pass (separate array in storage)
#   cols 8..13 = storage cols 7..12 (spectral)
POOL_TO_INFERENCE_14 = [0, 1, 2, 3, 4, 5, 6, -1, 7, 8, 9, 10, 11, 12]
# -1 is a sentinel meaning "use whitelist_mask instead"

# Lane event feature order, 16 cols
# cols 0..5 from pool (duration_ms, attack_rms, attack_peak, intensity_origin, key_occ, scratch_occ)
# col  6     from record idx192
# col  7     from record density_rank
# col  8     from record phase_encoded
# col  9     from record total_placed_key_notes
# cols 10..15 from pool spectral (storage cols 7..12)
LANE_POOL_PREFIX_COLS = [0, 1, 2, 3, 4, 5]            # 6 cols from pool
LANE_POOL_SPECTRAL_COLS = [7, 8, 9, 10, 11, 12]       # 6 cols from pool

SAMPLE_RATE = 44100
NYQUIST_HZ = SAMPLE_RATE / 2.0


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class PoolRegistry:
    """In-memory cache of per-package pool tables keyed by package_id.

    Stores, per package:
      features_14  — numpy (P, 14) float32, in the inference tensor order
      whitelist    — numpy (P,)  float32 (0/1)
    The 14-column features array normalizes spectral_centroid_mean / std by
    NYQUIST_HZ, matching the inference-time normalization in placement_engine.
    """

    def __init__(self, pools_path: str) -> None:
        with open(pools_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        self.feature_columns = raw.get("feature_columns", POOL_STORAGE_COLUMNS)
        self._packages: Dict[str, Dict[str, np.ndarray]] = {}
        packages = raw.get("packages", {})
        for pkg_id, entry in packages.items():
            storage = np.asarray(entry["features"], dtype=np.float32)  # (P, 13)
            whitelist = np.asarray(entry["whitelist_mask"], dtype=np.float32)  # (P,)
            if storage.ndim != 2 or storage.shape[1] != 13:
                raise ValueError(
                    f"{pkg_id}: pool features shape {storage.shape}, expected (P, 13)"
                )
            if whitelist.shape[0] != storage.shape[0]:
                raise ValueError(
                    f"{pkg_id}: whitelist length {whitelist.shape[0]} != pool size {storage.shape[0]}"
                )
            features_14 = self._build_inference_tensor(storage, whitelist)
            self._packages[pkg_id] = {
                "features_14": features_14,
                "whitelist": whitelist,
                "storage": storage,
            }

    @staticmethod
    def _build_inference_tensor(
        storage: np.ndarray, whitelist: np.ndarray
    ) -> np.ndarray:
        """Translate (P, 13) storage layout into the (P, 14) inference layout."""
        P = storage.shape[0]
        out = np.zeros((P, 14), dtype=np.float32)
        for dst, src in enumerate(POOL_TO_INFERENCE_14):
            if src == -1:
                out[:, dst] = whitelist
            else:
                out[:, dst] = storage[:, src]
        # Spectral centroid normalization — inference cols 8 and 9
        out[:, 8] /= NYQUIST_HZ
        out[:, 9] /= NYQUIST_HZ
        return out

    def get_features_14(self, package_id: str) -> np.ndarray:
        return self._packages[package_id]["features_14"]

    def get_storage(self, package_id: str) -> np.ndarray:
        return self._packages[package_id]["storage"]

    def get_whitelist(self, package_id: str) -> np.ndarray:
        return self._packages[package_id]["whitelist"]

    def has(self, package_id: str) -> bool:
        return package_id in self._packages

    def __len__(self) -> int:
        return len(self._packages)


def load_split(run_log_path: str) -> Tuple[List[str], List[str]]:
    """Extract train / validation package lists from labeling_run_log.json.

    The labeling run log can be very large (~1GB) because it stores per-package
    chart metadata; we parse it once and keep only the two split lists.
    """
    with open(run_log_path, "r", encoding="utf-8") as f:
        log = json.load(f)
    summary = log.get("summary", {})
    train = list(summary.get("train_packages", []))
    val = list(summary.get("validation_packages", []))
    if not train or not val:
        raise ValueError(
            f"train/val split empty in {run_log_path}: "
            f"train={len(train)}, val={len(val)}"
        )
    return train, val


def build_offset_index(jsonl_path: str, cache_path: str) -> np.ndarray:
    """Return a numpy array of byte offsets for each JSONL record line.

    The first line (meta header) is skipped. Caches the index to `cache_path`
    as a .npy file so successive runs do not rescan the JSONL.
    """
    if os.path.exists(cache_path):
        idx = np.load(cache_path)
        return idx

    offsets: List[int] = []
    with open(jsonl_path, "rb") as f:
        # Skip meta header line
        header_line = f.readline()
        if not header_line:
            raise ValueError(f"{jsonl_path}: empty file")
        while True:
            pos = f.tell()
            line = f.readline()
            if not line:
                break
            offsets.append(pos)
    arr = np.asarray(offsets, dtype=np.int64)
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    np.save(cache_path, arr)
    return arr
