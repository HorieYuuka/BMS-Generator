"""
data_labeling.py — BMS chart labeling for TokenSelectionModel and LaneAssignmentModel
Builds TokenSelection and LaneAssignment training datasets.

Reads each source package's token_analysis.json (produced by mix_generation.py),
walks the human-authored BMS charts in the package, and emits two JSONL datasets:
  - token_selection_dataset.jsonl
  - lane_assignment_dataset.jsonl
plus labeling_run_log.json. Caches per-package results in dataset_root/.labeling_cache/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent))
from bms_parser import parse_bms

# ---------------------------------------------------------------------------
# Constants — must match the placement policy and labeling layout
# ---------------------------------------------------------------------------

KEY_CHANNELS = frozenset({"11", "12", "13", "14", "15", "18", "19"})
SCRATCH_CHANNELS = frozenset({"16"})
BGM_CHANNELS = frozenset({"01"})

CH_TO_KEY_LANE = {
    "11": "P1_KEY1", "12": "P1_KEY2", "13": "P1_KEY3",
    "14": "P1_KEY4", "15": "P1_KEY5", "18": "P1_KEY6", "19": "P1_KEY7",
}
KEY_LANES = ["P1_KEY1", "P1_KEY2", "P1_KEY3", "P1_KEY4",
             "P1_KEY5", "P1_KEY6", "P1_KEY7"]
LANE_INDEX = {lane: i + 1 for i, lane in enumerate(KEY_LANES)}
LEFT_KEY_LANES = frozenset({"P1_KEY1", "P1_KEY2", "P1_KEY3"})
RIGHT_KEY_LANES = frozenset({"P1_KEY4", "P1_KEY5", "P1_KEY6", "P1_KEY7"})

JACK_DELTA_TICKS = 12
HAND_BALANCE_MIN_NOTES = 10
HAND_BALANCE_LOW = 0.30
HAND_BALANCE_HIGH = 0.70

REQUIRED_CONFIG_KEYS = [
    "TOKEN_SELECTION_CONTEXT_WINDOW",
    "LANE_ASSIGNMENT_CONTEXT_WINDOW",
    "MIN_PLAYABLE_NOTE_COUNT",
    "MIN_MEASURE_COUNT",
    "MAX_DECODE_FAIL_RATIO",
    "TRAIN_SPLIT_RATIO",
    "TRAIN_SPLIT_SEED",
    "LABELING_MIN_WAV_COVERAGE",
    "LABELING_PHASE_MERGE_RATIO_MAX",
    "LABELING_WHITELIST_DURATION_MAX",
    "LABELING_WHITELIST_MIN_OCCURRENCE",
    "LABELING_WHITELIST_MIN_ATTACK_PERCENTILE",
    "LABELING_FX_DURATION_THRESHOLD",
    "LABELING_FX_ATTACK_THRESHOLD",
    "LABELING_FX_ORIGIN_FILTER_ENABLED",
]

SCHEMA_VERSION = "3.0"
CACHE_INDEX_SCHEMA_VERSION = "3.0"

POOL_FEATURE_COLUMNS = [
    "duration_ms",
    "attack_rms",
    "attack_peak",
    "intensity_origin",
    "key_occurrence",
    "scratch_occurrence",
    "bgm_occurrence",
    "spectral_centroid_mean",
    "spectral_centroid_std",
    "spectral_flatness_mean",
    "low_freq_energy_ratio",
    "zero_crossing_rate_mean",
    "zero_crossing_rate_std",
]


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------

def compute_file_hash(file_path: str) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return f"sha256:{h.hexdigest()}"


def compute_config_hash(config: dict) -> str:
    canonical = json.dumps(config, sort_keys=True, ensure_ascii=True)
    return f"sha256:{hashlib.sha256(canonical.encode()).hexdigest()}"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        config = json.load(f)
    missing = [k for k in REQUIRED_CONFIG_KEYS if k not in config]
    if missing:
        sys.exit(f"ERROR: config missing required keys: {missing}")
    return config


# ---------------------------------------------------------------------------
# Package discovery
# ---------------------------------------------------------------------------

def discover_packages(dataset_root: str) -> List[str]:
    if not os.path.isdir(dataset_root):
        sys.exit(f"ERROR: dataset_root does not exist: {dataset_root}")
    out = []
    for name in sorted(os.listdir(dataset_root)):
        full = os.path.join(dataset_root, name)
        if os.path.isdir(full) and not name.startswith("."):
            out.append(full)
    return out


def find_bms_files(package_dir: str) -> List[str]:
    out = []
    for root, _dirs, files in os.walk(package_dir):
        for name in files:
            if name.lower().endswith((".bms", ".bme", ".bml")):
                out.append(os.path.join(root, name))
    return sorted(out)


# ---------------------------------------------------------------------------
# token_analysis.json loading
# ---------------------------------------------------------------------------

def load_token_analysis(path: str) -> Dict[str, dict]:
    with open(path, "r", encoding="utf-8") as f:
        entries = json.load(f)
    return {e["token"]: e for e in entries if "token" in e}


def feature_or_zero(entry: Optional[dict], key: str) -> float:
    if not entry or not entry.get("decode_ok"):
        return 0.0
    val = entry.get(key)
    return float(val) if val is not None else 0.0


# ---------------------------------------------------------------------------
# Pool construction (package scope)
# ---------------------------------------------------------------------------

def build_package_pool(
    charts: List[dict],
) -> Tuple[set, Dict[str, int], Dict[str, int], Dict[str, int]]:
    """
    Returns (pool_tokens, key_occ, scratch_occ, bgm_occ) at package scope.
    LN end tokens are not counted as occurrences.
    """
    pool: set = set()
    key_occ: Dict[str, int] = defaultdict(int)
    scratch_occ: Dict[str, int] = defaultdict(int)
    bgm_occ: Dict[str, int] = defaultdict(int)

    for chart in charts:
        for ev in chart["events"]:
            etype = ev.get("type")
            if etype == "Tap":
                token = ev["token"]
                ch = ev["rawChannel"]
            elif etype == "Long":
                token = ev["tokenStart"]
                ch = ev["rawChannelStart"]
            elif etype == "BGM":
                token = ev["token"]
                ch = ev["rawChannel"]
            else:
                continue
            if not token or token == "00":
                continue
            pool.add(token)
            if ch in KEY_CHANNELS:
                key_occ[token] += 1
            elif ch in SCRATCH_CHANNELS:
                scratch_occ[token] += 1
            elif ch in BGM_CHANNELS:
                bgm_occ[token] += 1
    return pool, dict(key_occ), dict(scratch_occ), dict(bgm_occ)


def compute_attack_percentile(
    pool_tokens: set, ta_map: Dict[str, dict]
) -> Dict[str, float]:
    valid: List[Tuple[float, str]] = []
    # F.1: sorted() removes PYTHONHASHSEED-dependent set ordering of pool_tokens.
    for t in sorted(pool_tokens):
        info = ta_map.get(t)
        if info and info.get("decode_ok"):
            valid.append((float(info.get("attack_rms", 0.0)), t))
    n = len(valid)
    if n == 0:
        return {}
    if n == 1:
        return {valid[0][1]: 50.0}
    # token tie-break stabilizes percentile assignment when attack_rms collides.
    valid.sort(key=lambda x: (x[0], x[1]))
    pct: Dict[str, float] = {}
    i = 0
    while i < n:
        j = i
        while j < n and valid[j][0] == valid[i][0]:
            j += 1
        avg_rank = (i + j - 1) / 2.0
        p = (avg_rank / (n - 1)) * 100.0
        for k in range(i, j):
            pct[valid[k][1]] = p
        i = j
    return pct


def compute_intensity_origin(
    pool_tokens: set, key_occ: Dict[str, int], scratch_occ: Dict[str, int]
) -> Dict[str, int]:
    return {
        t: (1 if key_occ.get(t, 0) > 0 or scratch_occ.get(t, 0) > 0 else 0)
        for t in pool_tokens
    }


def classify_fx(
    pool_tokens: set,
    ta_map: Dict[str, dict],
    pct_map: Dict[str, float],
    intensity_origin: Dict[str, int],
    config: dict,
) -> Dict[str, bool]:
    """Returns {token: is_background_fx}. decode_ok=false → False (handled separately)."""
    out: Dict[str, bool] = {}
    fx_dur = config["LABELING_FX_DURATION_THRESHOLD"]
    fx_atk = config["LABELING_FX_ATTACK_THRESHOLD"]
    fx_origin = config["LABELING_FX_ORIGIN_FILTER_ENABLED"]
    for token in pool_tokens:
        info = ta_map.get(token)
        if not info or not info.get("decode_ok"):
            out[token] = False
            continue
        is_fx = (
            info.get("duration_ms", 0.0) > fx_dur
            or pct_map.get(token, 0.0) <= fx_atk
            or (intensity_origin.get(token, 0) == 0 and fx_origin)
        )
        out[token] = bool(is_fx)
    return out


def compute_whitelist(
    pool_tokens: set,
    ta_map: Dict[str, dict],
    key_occ: Dict[str, int],
    scratch_occ: Dict[str, int],
    bgm_occ: Dict[str, int],
    pct_map: Dict[str, float],
    fx_map: Dict[str, bool],
    config: dict,
) -> Tuple[set, set]:
    """Returns (whitelist_tokens, whitelist_excluded_tokens)."""
    wl_dur = config["LABELING_WHITELIST_DURATION_MAX"]
    wl_occ = config["LABELING_WHITELIST_MIN_OCCURRENCE"]
    wl_atk = config["LABELING_WHITELIST_MIN_ATTACK_PERCENTILE"]
    whitelist: set = set()
    excluded: set = set()
    # F.1: sorted() removes PYTHONHASHSEED-dependent iteration ordering — though
    # whitelist/excluded are sets (not order-sensitive consumers), upstream
    # add() order can leak into downstream iteration if any caller uses them
    # as iterables; sorting here is cheap insurance.
    for token in sorted(pool_tokens):
        info = ta_map.get(token)
        if not info or not info.get("decode_ok"):
            excluded.add(token)
            continue
        dur = info.get("duration_ms", 0.0)
        total_occ = key_occ.get(token, 0) + scratch_occ.get(token, 0) + bgm_occ.get(token, 0)
        pct = pct_map.get(token, 0.0)
        if (
            dur > wl_dur
            or total_occ <= wl_occ
            or pct <= wl_atk
            or fx_map.get(token, False)
        ):
            excluded.add(token)
        else:
            whitelist.add(token)
    return whitelist, excluded


def detect_wav_conflicts(
    charts: List[dict], ta_map: Dict[str, dict]
) -> List[dict]:
    """
    Compares each chart's #WAVxx to the wav_file recorded in token_analysis.json
    (representative declaration). Returns list of conflict records.
    """
    expected: Dict[str, str] = {tok: info.get("wav_file", "") for tok, info in ta_map.items()}
    conflicts: List[dict] = []
    for chart in charts:
        chart_wavs: Dict[str, str] = {}
        for k, v in chart["headers"].items():
            if k.startswith("WAV") and len(k) > 3:
                chart_wavs[k[3:]] = v.strip()
        for tok, fname in chart_wavs.items():
            if tok in expected and expected[tok] and fname != expected[tok]:
                conflicts.append({
                    "token": tok,
                    "chart_file": chart["file"],
                    "expected": expected[tok],
                    "found": fname,
                })
    return conflicts


# ---------------------------------------------------------------------------
# Chart-level helpers
# ---------------------------------------------------------------------------

def chart_playable_events(events: List[dict]) -> List[dict]:
    """
    Returns sorted list of (lane, measure, idx192, token) dicts for key events
    in ch11~15, ch18~19. LN start treated as Tap, LN end discarded.
    """
    out: List[dict] = []
    for ev in events:
        etype = ev.get("type")
        if etype == "Tap":
            ch = ev["rawChannel"]
            if ch not in KEY_CHANNELS:
                continue
            out.append({
                "lane": CH_TO_KEY_LANE[ch],
                "measure": ev["measure"],
                "idx192": ev["idx192"],
                "token": ev["token"],
                "tkey": ev["measure"] * 192 + ev["idx192"],
            })
        elif etype == "Long":
            ch = ev["rawChannelStart"]
            if ch not in KEY_CHANNELS:
                continue
            out.append({
                "lane": CH_TO_KEY_LANE[ch],
                "measure": ev["measureStart"],
                "idx192": ev["idx192Start"],
                "token": ev["tokenStart"],
                "tkey": ev["measureStart"] * 192 + ev["idx192Start"],
            })
    out.sort(key=lambda e: (e["tkey"], LANE_INDEX[e["lane"]]))
    return out


def compute_measure_density(
    playables: List[dict],
) -> Tuple[Dict[int, int], int]:
    """Returns (measure_density, measure_count)."""
    md: Dict[int, int] = defaultdict(int)
    last_measure = -1
    for ev in playables:
        md[ev["measure"]] += 1
        if ev["measure"] > last_measure:
            last_measure = ev["measure"]
    measure_count = last_measure + 1 if last_measure >= 0 else 0
    return dict(md), measure_count


def compute_eligible_measures(
    measure_density: Dict[int, int], measure_count: int
) -> set:
    if measure_count <= 0:
        return set()
    total = sum(measure_density.get(m, 0) for m in range(measure_count))
    mean = total / measure_count
    return {m for m in range(measure_count) if measure_density.get(m, 0) >= mean}


# ---------------------------------------------------------------------------
# Phase segmentation (replicates the placement policy's phase logic
# with LABELING_PHASE_MERGE_RATIO_MAX substituted)
# ---------------------------------------------------------------------------

def _lerp_percentile(sorted_vals: List[float], pct: float) -> float:
    n = len(sorted_vals)
    if n == 1:
        return float(sorted_vals[0])
    idx = pct / 100.0 * (n - 1)
    lo = int(idx)
    hi = min(lo + 1, n - 1)
    return sorted_vals[lo] + (idx - lo) * (sorted_vals[hi] - sorted_vals[lo])


def segment_phases(
    pool_events: List[Tuple[int, int, str]], measure_max: int, merge_ratio: float
) -> List[dict]:
    """
    pool_events: list of (measure, idx192, token) for ALL channel events in chart
    (BGM + key + scratch + LN), used for triple chord detection.
    """
    num_measures = max(measure_max + 1, 1)
    block_starts = list(range(0, num_measures, 4)) or [0]
    blocks: List[dict] = []
    for bs in block_starts:
        evs = [(m, i, t) for (m, i, t) in pool_events if bs <= m < bs + 4]
        pos_tokens: Dict[Tuple[int, int], set] = defaultdict(set)
        for (m, i, t) in evs:
            pos_tokens[(m, i)].add(t)
        triple = sum(1 for toks in pos_tokens.values() if len(toks) >= 3)
        blocks.append({
            "start": bs,
            "end": min(bs + 4, num_measures),
            "phase_score": len(evs) + 1.5 * triple,
        })
    n = len(blocks)
    if n == 0:
        return []
    for i in range(n):
        w = blocks[max(0, i - 3): i + 1]
        blocks[i]["smoothed_score"] = sum(b["phase_score"] for b in w) / len(w)
    gm = sum(b["smoothed_score"] for b in blocks) / n
    ss = sorted(b["smoothed_score"] for b in blocks)
    rush_thr = _lerp_percentile(ss, 85.0)
    rest_thr = _lerp_percentile(ss, 20.0)
    for b in blocks:
        s = b["smoothed_score"]
        if s >= rush_thr:
            b["phase"] = "rush"
        elif s <= rest_thr and s <= gm:
            b["phase"] = "rest"
        else:
            b["phase"] = "normal"
    return _merge_phase_blocks(blocks, merge_ratio)


def _merge_phase_blocks(blocks: List[dict], merge_ratio: float) -> List[dict]:
    changed = True
    while changed:
        changed = False
        new_blocks: List[dict] = []
        i = 0
        while i < len(blocks):
            if i + 1 < len(blocks):
                a, b = blocks[i], blocks[i + 1]
                sa_sz = a["end"] - a["start"]
                sb_sz = b["end"] - b["start"]
                if a["phase"] == b["phase"] and sa_sz + sb_sz <= 8:
                    sa, sb = a["smoothed_score"], b["smoothed_score"]
                    mx = max(sa, sb)
                    if mx == 0 or abs(sa - sb) / mx <= merge_ratio:
                        new_blocks.append({
                            "start": a["start"],
                            "end": b["end"],
                            "phase": a["phase"],
                            "smoothed_score": (sa * sa_sz + sb * sb_sz) / (sa_sz + sb_sz),
                        })
                        i += 2
                        changed = True
                        continue
            new_blocks.append(blocks[i])
            i += 1
        blocks = new_blocks
    return blocks


def measure_phase(measure: int, phase_blocks: List[dict]) -> str:
    for blk in phase_blocks:
        if blk["start"] <= measure < blk["end"]:
            return blk["phase"]
    return "normal"


def chart_pool_events(events: List[dict]) -> Tuple[List[Tuple[int, int, str]], int]:
    """All channel events as (measure, idx192, token) for phase computation."""
    out: List[Tuple[int, int, str]] = []
    measure_max = 0
    for ev in events:
        etype = ev.get("type")
        if etype == "Tap":
            t = ev["token"]
            if t and t != "00":
                out.append((ev["measure"], ev["idx192"], t))
                if ev["measure"] > measure_max:
                    measure_max = ev["measure"]
        elif etype == "Long":
            t = ev["tokenStart"]
            if t and t != "00":
                out.append((ev["measureStart"], ev["idx192Start"], t))
                if ev["measureStart"] > measure_max:
                    measure_max = ev["measureStart"]
        elif etype == "BGM":
            t = ev["token"]
            if t and t != "00":
                out.append((ev["measure"], ev["idx192"], t))
                if ev["measure"] > measure_max:
                    measure_max = ev["measure"]
    return out, measure_max


# ---------------------------------------------------------------------------
# available_lanes constraint walk
# ---------------------------------------------------------------------------

def compute_available_lanes(
    placed_at_pos: Dict[Tuple[int, int], set],
    last_tkey_per_lane: Dict[str, int],
    left_count: int,
    right_count: int,
    measure: int,
    idx192: int,
    current_tkey: int,
) -> List[str]:
    used = placed_at_pos.get((measure, idx192), set())
    avail = [l for l in KEY_LANES if l not in used]
    avail = [l for l in avail
             if abs(current_tkey - last_tkey_per_lane.get(l, -10**9)) > JACK_DELTA_TICKS]
    total = left_count + right_count
    if total >= HAND_BALANCE_MIN_NOTES:
        new_total = total + 1
        if (left_count + 1) / new_total > HAND_BALANCE_HIGH:
            avail = [l for l in avail if l not in LEFT_KEY_LANES]
        if (right_count + 1) / new_total > HAND_BALANCE_HIGH:
            avail = [l for l in avail if l not in RIGHT_KEY_LANES]
    return avail


# ---------------------------------------------------------------------------
# Chart-level filtering (§12.1)
# ---------------------------------------------------------------------------

def chart_filter_reason(
    chart: dict,
    playable_count: int,
    measure_count: int,
    pool_decode_fail_ratio: float,
    used_wav_coverage: float,
    config: dict,
) -> Optional[str]:
    if chart["headers"].get("PLAYER", "").strip() != "1":
        return "player_not_1"
    if playable_count < config["MIN_PLAYABLE_NOTE_COUNT"]:
        return "playable_too_low"
    if measure_count < config["MIN_MEASURE_COUNT"]:
        return "measure_too_short"
    if pool_decode_fail_ratio > config["MAX_DECODE_FAIL_RATIO"]:
        return "decode_fail_too_high"
    if used_wav_coverage < config["LABELING_MIN_WAV_COVERAGE"]:
        return "wav_coverage_below_threshold"
    return None


def used_wav_coverage(chart: dict) -> float:
    declared = {k[3:] for k in chart["headers"] if k.startswith("WAV") and len(k) > 3}
    if not declared:
        return 0.0
    used: set = set()
    for ev in chart["events"]:
        etype = ev.get("type")
        if etype in ("Tap", "BGM"):
            t = ev["token"]
        elif etype == "Long":
            t = ev["tokenStart"]
        else:
            continue
        if t and t != "00":
            used.add(t)
    return len(used & declared) / len(declared)


# ---------------------------------------------------------------------------
# Per-chart record generation
# ---------------------------------------------------------------------------

SPECTRAL_FIELDS = [
    "spectral_centroid_mean",
    "spectral_centroid_std",
    "spectral_flatness_mean",
    "low_freq_energy_ratio",
    "zero_crossing_rate_mean",
    "zero_crossing_rate_std",
]


def build_pool_table(
    pool_tokens: set,
    ta_map: Dict[str, dict],
    intensity_origin: Dict[str, int],
    key_occ: Dict[str, int],
    scratch_occ: Dict[str, int],
    bgm_occ: Dict[str, int],
    whitelist_excluded: set,
) -> dict:
    """
    Build the per-package pool table.

    Returns a dict with three parallel entries:
      tokens         — sorted token id list (defines pool_index)
      features       — list of 13-element rows in POOL_FEATURE_COLUMNS order
      whitelist_mask — list of 1/0, parallel to tokens
    """
    sorted_tokens = sorted(pool_tokens)
    features: List[List[float]] = []
    whitelist_mask: List[int] = []
    for token in sorted_tokens:
        info = ta_map.get(token, {}) or {}
        decode_ok = bool(info.get("decode_ok"))
        if decode_ok:
            row = [
                float(info.get("duration_ms", 0.0)),
                float(info.get("attack_rms", 0.0)),
                float(info.get("attack_peak", 0.0)),
                int(intensity_origin.get(token, 0)),
                int(key_occ.get(token, 0)),
                int(scratch_occ.get(token, 0)),
                int(bgm_occ.get(token, 0)),
                feature_or_zero(info, "spectral_centroid_mean"),
                feature_or_zero(info, "spectral_centroid_std"),
                feature_or_zero(info, "spectral_flatness_mean"),
                feature_or_zero(info, "low_freq_energy_ratio"),
                feature_or_zero(info, "zero_crossing_rate_mean"),
                feature_or_zero(info, "zero_crossing_rate_std"),
            ]
        else:
            row = [
                0.0, 0.0, 0.0,
                int(intensity_origin.get(token, 0)),
                int(key_occ.get(token, 0)),
                int(scratch_occ.get(token, 0)),
                int(bgm_occ.get(token, 0)),
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ]
        features.append(row)
        whitelist_mask.append(0 if token in whitelist_excluded else 1)
    return {
        "tokens": sorted_tokens,
        "features": features,
        "whitelist_mask": whitelist_mask,
    }


def build_chart_records(
    package_id: str,
    chart_file: str,
    chart: dict,
    pool_index_map: Dict[str, int],
    whitelist_excluded: set,
    scratch_lane_tokens: set,
    density_rank: float,
    config: dict,
) -> Tuple[List[dict], List[dict], int]:
    """
    Returns (token_selection_records, lane_assignment_records, skipped_lane_records).
    Records use pool_index references into the per-package pool table.
    """
    ts_window = config["TOKEN_SELECTION_CONTEXT_WINDOW"]
    la_window = config["LANE_ASSIGNMENT_CONTEXT_WINDOW"]
    merge_ratio = config["LABELING_PHASE_MERGE_RATIO_MAX"]

    playables = chart_playable_events(chart["events"])
    measure_density, measure_count = compute_measure_density(playables)
    eligible = compute_eligible_measures(measure_density, measure_count)

    # Phase computed over full chart pool
    pool_evs, m_max = chart_pool_events(chart["events"])
    phase_blocks = segment_phases(pool_evs, m_max, merge_ratio)

    ts_records: List[dict] = []
    la_records: List[dict] = []
    skipped = 0

    # ── TokenSelectionModel records ────────────────────────────────────────
    placed_per_measure: Dict[int, List[str]] = defaultdict(list)
    for ev in playables:
        placed_per_measure[ev["measure"]].append(ev["token"])

    eligible_sorted = sorted(eligible)

    # Token order for label emission (excluding whitelist_excluded and scratch-only tokens)
    labelable_tokens = [
        t for t in sorted(pool_index_map.keys())
        if t not in whitelist_excluded and t not in scratch_lane_tokens
    ]
    labelable_indices = [(pool_index_map[t], t) for t in labelable_tokens]

    for measure in eligible_sorted:
        ph = measure_phase(measure, phase_blocks)
        notes_in_measure = measure_density.get(measure, 0)
        idx_in_eligible = eligible_sorted.index(measure)
        prev = eligible_sorted[max(0, idx_in_eligible - ts_window):idx_in_eligible]
        context_entries: List[dict] = []
        for cm in reversed(prev):
            placed_indices = sorted({
                pool_index_map[t] for t in placed_per_measure.get(cm, [])
                if t in pool_index_map
            })
            context_entries.append({
                "measure": cm,
                "tkey_delta": (cm - measure) * 192,
                "placed_count": measure_density.get(cm, 0),
                "placed_pool_indices": placed_indices,
            })
        while len(context_entries) < ts_window:
            context_entries.append({
                "measure": None,
                "tkey_delta": None,
                "placed_count": 0,
                "placed_pool_indices": [],
            })

        placed_tokens = set(placed_per_measure.get(measure, []))
        labels = [
            [pi, 1 if t in placed_tokens else 0]
            for pi, t in labelable_indices
        ]
        ts_records.append({
            "package_id": package_id,
            "chart_file": chart_file,
            "measure": measure,
            "density_rank": density_rank,
            "phase": ph,
            "notes_in_measure": notes_in_measure,
            "context": context_entries,
            "labels": labels,
        })

    # ── LaneAssignmentModel records ────────────────────────────────────────
    placed_at_pos: Dict[Tuple[int, int], set] = defaultdict(set)
    last_tkey_per_lane: Dict[str, int] = {}
    left_count = 0
    right_count = 0
    context_history: List[dict] = []  # most recent appended at end
    total_placed_key_notes = 0

    for ev in playables:
        measure = ev["measure"]
        idx192 = ev["idx192"]
        tkey = ev["tkey"]
        lane = ev["lane"]
        token = ev["token"]

        if measure in eligible:
            available_list = compute_available_lanes(
                placed_at_pos, last_tkey_per_lane,
                left_count, right_count, measure, idx192, tkey,
            )
            if lane not in available_list:
                skipped += 1
            else:
                ph = measure_phase(measure, phase_blocks)
                avail_mask = [1 if l in available_list else 0 for l in KEY_LANES]
                # context: preceding N events, most recent first
                ctx_entries: List[dict] = []
                for prior in reversed(context_history[-la_window:]):
                    ctx_entries.append({
                        "tkey_delta": prior["tkey"] - tkey,
                        "lane": LANE_INDEX[prior["lane"]],
                        "pool_index": pool_index_map.get(prior["token"]),
                        "idx192": prior["idx192"],
                    })
                while len(ctx_entries) < la_window:
                    ctx_entries.append({
                        "tkey_delta": None,
                        "lane": 0,
                        "pool_index": None,
                        "idx192": None,
                    })
                la_records.append({
                    "package_id": package_id,
                    "chart_file": chart_file,
                    "measure": measure,
                    "idx192": idx192,
                    "tkey": tkey,
                    "density_rank": density_rank,
                    "phase": ph,
                    "pool_index": pool_index_map.get(token),
                    "total_placed_key_notes": total_placed_key_notes,
                    "available_lanes": avail_mask,
                    "context": ctx_entries,
                    "label": LANE_INDEX[lane],
                })

        # advance constraint state regardless of eligibility
        placed_at_pos[(measure, idx192)].add(lane)
        last_tkey_per_lane[lane] = tkey
        if lane in LEFT_KEY_LANES:
            left_count += 1
        else:
            right_count += 1
        total_placed_key_notes += 1
        context_history.append({
            "tkey": tkey, "lane": lane, "token": token, "idx192": idx192,
        })

    return ts_records, la_records, skipped


# ---------------------------------------------------------------------------
# Per-package processing
# ---------------------------------------------------------------------------

def process_package(package_dir: str, config: dict, run_warnings: List[str]) -> dict:
    package_id = os.path.basename(package_dir.rstrip(os.sep))
    ta_path = os.path.join(package_dir, "token_analysis.json")
    if not os.path.exists(ta_path):
        return {
            "status": "error",
            "error": "token_analysis.json not found",
            "package_id": package_id,
        }

    bms_files = find_bms_files(package_dir)
    if not bms_files:
        return {
            "status": "error",
            "error": "no BMS files found",
            "package_id": package_id,
        }

    ta_map = load_token_analysis(ta_path)

    # Parse all charts in package
    charts: List[dict] = []
    for path in bms_files:
        try:
            with open(path, "rb") as f:
                pr = parse_bms(f.read())
            pr["file"] = os.path.basename(path)
            pr["path"] = path
            charts.append(pr)
        except Exception as exc:
            run_warnings.append(f"{package_id}: failed to parse {os.path.basename(path)}: {exc}")

    if not charts:
        return {
            "status": "error",
            "error": "all BMS parsing failed",
            "package_id": package_id,
        }

    # Pool construction
    pool_tokens, key_occ, scratch_occ, bgm_occ = build_package_pool(charts)

    # Pool decode fail ratio (package-shared)
    fail_count = sum(
        1 for t in pool_tokens
        if not (ta_map.get(t) and ta_map[t].get("decode_ok"))
    )
    pool_decode_fail_ratio = fail_count / len(pool_tokens) if pool_tokens else 1.0

    # Spectral fields warning (pre-addendum cache)
    spectral_missing = [
        t for t in pool_tokens
        if (ta_map.get(t) and ta_map[t].get("decode_ok")
            and "spectral_centroid_mean" not in ta_map[t])
    ]
    if spectral_missing:
        run_warnings.append(
            f"{package_id}: {len(spectral_missing)} tokens missing spectral fields "
            f"(treated as 0.0); re-run mix_generation to regenerate"
        )

    pct_map = compute_attack_percentile(pool_tokens, ta_map)
    intensity_origin = compute_intensity_origin(pool_tokens, key_occ, scratch_occ)
    fx_map = classify_fx(pool_tokens, ta_map, pct_map, intensity_origin, config)
    whitelist, whitelist_excluded = compute_whitelist(
        pool_tokens, ta_map, key_occ, scratch_occ, bgm_occ, pct_map, fx_map, config
    )
    scratch_lane_tokens = {
        t for t in pool_tokens
        if scratch_occ.get(t, 0) > 0 and key_occ.get(t, 0) == 0 and bgm_occ.get(t, 0) == 0
    }

    pool_table = build_pool_table(
        pool_tokens, ta_map, intensity_origin,
        key_occ, scratch_occ, bgm_occ, whitelist_excluded,
    )
    pool_index_map = {t: i for i, t in enumerate(pool_table["tokens"])}

    wav_conflicts = detect_wav_conflicts(charts, ta_map)
    for wc in wav_conflicts:
        run_warnings.append(
            f"WAV conflict: token {wc['token']}, chart {wc['chart_file']}, "
            f"expected {wc['expected']}, found {wc['found']}"
        )

    # Density rank: per chart, normalized within package
    chart_densities: List[Tuple[float, dict]] = []
    chart_meta: List[dict] = []  # parallel: per chart precomputed data
    for chart in charts:
        playables = chart_playable_events(chart["events"])
        md, mc = compute_measure_density(playables)
        density = (len(playables) / mc) if mc > 0 else 0.0
        chart_meta.append({
            "chart": chart,
            "playables": playables,
            "measure_density": md,
            "measure_count": mc,
            "playable_count": len(playables),
            "density": density,
            "wav_coverage": used_wav_coverage(chart),
        })
        chart_densities.append((density, chart))

    n = len(chart_meta)
    if n == 1:
        chart_meta[0]["density_rank"] = 0.5
    else:
        sorted_by_density = sorted(range(n), key=lambda i: chart_meta[i]["density"])
        # Average rank for ties
        i = 0
        while i < n:
            j = i
            while j < n and chart_meta[sorted_by_density[j]]["density"] == chart_meta[sorted_by_density[i]]["density"]:
                j += 1
            avg_rank = (i + j - 1) / 2.0
            rank_norm = avg_rank / (n - 1)
            for k in range(i, j):
                chart_meta[sorted_by_density[k]]["density_rank"] = rank_norm
            i = j

    # Per-chart record generation
    ts_records: List[dict] = []
    la_records: List[dict] = []
    chart_results: List[dict] = []
    skipped_lane_total = 0
    any_processed = False

    for cm in chart_meta:
        chart = cm["chart"]
        chart_file = chart["file"]
        reason = chart_filter_reason(
            chart,
            cm["playable_count"],
            cm["measure_count"],
            pool_decode_fail_ratio,
            cm["wav_coverage"],
            config,
        )
        if reason:
            chart_results.append({
                "chart_file": chart_file,
                "filtered": True,
                "filter_reason": reason,
            })
            continue
        any_processed = True
        ts, la, skipped = build_chart_records(
            package_id, chart_file, chart,
            pool_index_map,
            whitelist_excluded, scratch_lane_tokens,
            cm["density_rank"], config,
        )
        skipped_lane_total += skipped
        ts_records.extend(ts)
        la_records.extend(la)
        chart_results.append({
            "chart_file": chart_file,
            "filtered": False,
            "token_selection_records": len(ts),
            "lane_assignment_records": len(la),
            "skipped_lane_records": skipped,
        })

    if not any_processed:
        return {
            "status": "filtered",
            "package_id": package_id,
            "filter_reason": "all_charts_filtered",
            "filtered_out": True,
            "charts": chart_results,
            "wav_conflicts": wav_conflicts,
            "token_selection_records": 0,
            "lane_assignment_records": 0,
            "ts_records": [],
            "la_records": [],
            "pool": None,
        }

    return {
        "status": "processed",
        "package_id": package_id,
        "filtered_out": False,
        "filter_reason": None,
        "charts": chart_results,
        "wav_conflicts": wav_conflicts,
        "skipped_lane_records": skipped_lane_total,
        "token_selection_records": len(ts_records),
        "lane_assignment_records": len(la_records),
        "ts_records": ts_records,
        "la_records": la_records,
        "pool": pool_table,
        "pool_token_count": len(pool_table["tokens"]),
    }


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

def load_cache_index(dataset_root: str) -> dict:
    path = os.path.join(dataset_root, "labeling_cache_index.json")
    if not os.path.exists(path):
        return {"schema_version": CACHE_INDEX_SCHEMA_VERSION, "entries": {}}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"schema_version": CACHE_INDEX_SCHEMA_VERSION, "entries": {}}


def save_cache_index(dataset_root: str, index: dict) -> None:
    path = os.path.join(dataset_root, "labeling_cache_index.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)


def cache_hit(
    cache_entry: Optional[dict],
    ta_path: str,
    bms_files: List[str],
    config_hash: str,
    dataset_root: str,
) -> bool:
    if not cache_entry:
        return False
    if cache_entry.get("config_hash") != config_hash:
        return False
    try:
        if cache_entry.get("token_analysis_hash") != compute_file_hash(ta_path):
            return False
    except Exception:
        return False
    cached_bms = cache_entry.get("bms_hashes", {})
    if set(cached_bms.keys()) != {os.path.basename(b) for b in bms_files}:
        return False
    for path in bms_files:
        try:
            if cached_bms.get(os.path.basename(path)) != compute_file_hash(path):
                return False
        except Exception:
            return False
    cache_file_rel = cache_entry.get("cache_file")
    if not cache_file_rel:
        return False
    cache_file_abs = os.path.join(dataset_root, cache_file_rel)
    if not os.path.exists(cache_file_abs):
        return False
    # Schema version gate: cache files written by v2 (no schema_version field
    # or value != "3.0") are treated as a miss so they get reprocessed.
    try:
        with open(cache_file_abs, "r", encoding="utf-8") as f:
            head = json.load(f)
        if head.get("schema_version") != SCHEMA_VERSION:
            return False
    except Exception:
        return False
    return True


def write_cache_file(
    dataset_root: str,
    package_id: str,
    ta_hash: str,
    pool: dict,
    ts_records: List[dict],
    la_records: List[dict],
) -> str:
    cache_dir = os.path.join(dataset_root, ".labeling_cache")
    os.makedirs(cache_dir, exist_ok=True)
    short = ta_hash.split(":", 1)[1][:8] if ":" in ta_hash else ta_hash[:8]
    fname = f"{package_id}_{short}.json"
    abs_path = os.path.join(cache_dir, fname)
    rel_path = os.path.relpath(abs_path, dataset_root).replace(os.sep, "/")
    with open(abs_path, "w", encoding="utf-8") as f:
        json.dump({
            "package_id": package_id,
            "schema_version": SCHEMA_VERSION,
            "pool": pool,
            "token_selection_records": ts_records,
            "lane_assignment_records": la_records,
        }, f, ensure_ascii=False)
    return rel_path


def read_cache_file(
    dataset_root: str, rel_path: str
) -> Tuple[dict, List[dict], List[dict]]:
    abs_path = os.path.join(dataset_root, rel_path)
    with open(abs_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return (
        data.get("pool") or {"tokens": [], "features": [], "whitelist_mask": []},
        data.get("token_selection_records", []),
        data.get("lane_assignment_records", []),
    )


# ---------------------------------------------------------------------------
# Main run
# ---------------------------------------------------------------------------

def run(
    dataset_root: str,
    output_dir: str,
    config_path: str,
    force: bool = False,
    seed_override: Optional[int] = None,
    limit: Optional[int] = None,
) -> dict:
    config = load_config(config_path)
    seed = seed_override if seed_override is not None else config["TRAIN_SPLIT_SEED"]
    config_hash = compute_config_hash(config)

    os.makedirs(output_dir, exist_ok=True)
    dataset_root = os.path.abspath(dataset_root)
    output_dir = os.path.abspath(output_dir)

    packages = discover_packages(dataset_root)
    print(f"Found {len(packages)} packages in {dataset_root}")
    if limit is not None and limit > 0:
        packages = packages[:limit]
        print(f"Limit active: processing first {len(packages)} packages")

    cache_index = load_cache_index(dataset_root)
    run_warnings: List[str] = []
    package_results: Dict[str, dict] = {}
    package_pools: Dict[str, dict] = {}

    cached_count = 0
    processed_count = 0
    filtered_count = 0
    error_count = 0
    total_ts_count = 0
    total_la_count = 0

    created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    ts_path = os.path.join(output_dir, "token_selection_dataset.jsonl")
    la_path = os.path.join(output_dir, "lane_assignment_dataset.jsonl")
    ts_file = open(ts_path, "w", encoding="utf-8")
    la_file = open(la_path, "w", encoding="utf-8")
    ts_file.write(json.dumps({"_meta": {
        "schema_version": SCHEMA_VERSION,
        "model": "TokenSelectionModel",
        "context_window": config["TOKEN_SELECTION_CONTEXT_WINDOW"],
        "created_at": created_at,
    }}, ensure_ascii=False) + "\n")
    la_file.write(json.dumps({"_meta": {
        "schema_version": SCHEMA_VERSION,
        "model": "LaneAssignmentModel",
        "context_window": config["LANE_ASSIGNMENT_CONTEXT_WINDOW"],
        "created_at": created_at,
    }}, ensure_ascii=False) + "\n")

    def _stream_records(file, records):
        for rec in records:
            file.write(json.dumps(rec, ensure_ascii=False) + "\n")

    for pkg_dir in packages:
        package_id = os.path.basename(pkg_dir.rstrip(os.sep))
        print(f"Processing {package_id} ...")
        ta_path = os.path.join(pkg_dir, "token_analysis.json")
        bms_files = find_bms_files(pkg_dir)

        cache_entry = cache_index.get("entries", {}).get(package_id)
        is_hit = (not force) and os.path.exists(ta_path) and cache_hit(
            cache_entry, ta_path, bms_files, config_hash, dataset_root
        )

        if is_hit:
            pool, ts, la = read_cache_file(dataset_root, cache_entry["cache_file"])
            ts_len = len(ts)
            la_len = len(la)
            _stream_records(ts_file, ts)
            _stream_records(la_file, la)
            ts_file.flush()
            la_file.flush()
            total_ts_count += ts_len
            total_la_count += la_len
            package_pools[package_id] = pool
            del ts, la
            package_results[package_id] = {
                "status": "cached",
                "cache_file": cache_entry["cache_file"],
                "token_selection_records": ts_len,
                "lane_assignment_records": la_len,
                "pool_token_count": len(pool.get("tokens", [])),
            }
            cached_count += 1
            print(f"  cached: ts={ts_len} la={la_len}")
            continue

        result = process_package(pkg_dir, config, run_warnings)
        status = result["status"]

        if status == "error":
            package_results[package_id] = {
                "status": "error",
                "error": result.get("error"),
            }
            error_count += 1
            print(f"  error: {result.get('error')}")
            continue

        if status == "filtered":
            package_results[package_id] = {
                "status": "filtered",
                "filtered_out": True,
                "filter_reason": result.get("filter_reason"),
                "charts": result.get("charts", []),
                "wav_conflicts": result.get("wav_conflicts", []),
                "token_selection_records": 0,
                "lane_assignment_records": 0,
            }
            filtered_count += 1
            print(f"  filtered: {result.get('filter_reason')}")
            continue

        # processed
        ts = result["ts_records"]
        la = result["la_records"]
        pool = result["pool"]
        ts_len = len(ts)
        la_len = len(la)

        ta_hash = compute_file_hash(ta_path)
        bms_hashes = {os.path.basename(b): compute_file_hash(b) for b in bms_files}
        cache_rel = write_cache_file(dataset_root, package_id, ta_hash, pool, ts, la)

        _stream_records(ts_file, ts)
        _stream_records(la_file, la)
        ts_file.flush()
        la_file.flush()
        total_ts_count += ts_len
        total_la_count += la_len
        package_pools[package_id] = pool
        del ts, la
        result["ts_records"] = None
        result["la_records"] = None
        cache_index.setdefault("schema_version", CACHE_INDEX_SCHEMA_VERSION)
        cache_index.setdefault("entries", {})[package_id] = {
            "package_path": os.path.relpath(pkg_dir, dataset_root).replace(os.sep, "/") + "/",
            "token_analysis_hash": ta_hash,
            "bms_hashes": bms_hashes,
            "cache_file": cache_rel,
            "cached_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
            "config_hash": config_hash,
        }

        package_results[package_id] = {
            "status": "processed",
            "filtered_out": False,
            "filter_reason": None,
            "wav_conflicts": result.get("wav_conflicts", []),
            "skipped_lane_records": result.get("skipped_lane_records", 0),
            "token_selection_records": ts_len,
            "lane_assignment_records": la_len,
            "charts": result.get("charts", []),
            "pool_token_count": result.get("pool_token_count", 0),
        }
        processed_count += 1
        print(f"  processed: ts={ts_len} la={la_len} skipped={result.get('skipped_lane_records', 0)}")

    ts_file.close()
    la_file.close()
    save_cache_index(dataset_root, cache_index)

    # ── package_pools.json (per-package pool tables) ────────────────────────
    pools_path = os.path.join(output_dir, "package_pools.json")
    with open(pools_path, "w", encoding="utf-8") as f:
        json.dump({
            "schema_version": SCHEMA_VERSION,
            "created_at": created_at,
            "feature_columns": POOL_FEATURE_COLUMNS,
            "packages": package_pools,
        }, f, ensure_ascii=False)

    # ── train/val split (package-level) ─────────────────────────────────────
    eligible_packages = [
        pid for pid, r in package_results.items()
        if r["status"] in ("cached", "processed")
    ]
    rng = random.Random(seed)
    shuffled = list(eligible_packages)
    rng.shuffle(shuffled)
    split_idx = int(len(shuffled) * config["TRAIN_SPLIT_RATIO"])
    train_packages = sorted(shuffled[:split_idx])
    val_packages = sorted(shuffled[split_idx:])

    # ── run log ────────────────────────────────────────────────────────────
    run_log = {
        "created_at": created_at,
        "dataset_root": dataset_root,
        "output_dir": output_dir,
        "config": config,
        "seed": seed,
        "force": force,
        "warnings": run_warnings,
        "packages": package_results,
        "summary": {
            "total_packages": len(packages),
            "cached": cached_count,
            "processed": processed_count,
            "filtered": filtered_count,
            "errors": error_count,
            "total_token_selection_records": total_ts_count,
            "total_lane_assignment_records": total_la_count,
            "train_packages": train_packages,
            "validation_packages": val_packages,
        },
    }
    log_path = os.path.join(output_dir, "labeling_run_log.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(run_log, f, ensure_ascii=False, indent=2)

    print()
    print(f"token_selection_dataset.jsonl → {ts_path} ({total_ts_count} records)")
    print(f"lane_assignment_dataset.jsonl → {la_path} ({total_la_count} records)")
    print(f"package_pools.json            → {pools_path} ({len(package_pools)} packages)")
    print(f"labeling_run_log.json → {log_path}")
    print(f"summary: cached={cached_count} processed={processed_count} "
          f"filtered={filtered_count} errors={error_count}")
    print(f"train_packages={len(train_packages)} validation_packages={len(val_packages)}")

    return run_log


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="BMS data labeling")
    ap.add_argument("--dataset", required=True, help="dataset_root path")
    ap.add_argument("--output", required=True, help="output directory")
    ap.add_argument("--config", required=True, help="config JSON path")
    ap.add_argument("--seed", type=int, default=None,
                    help="train/val split seed override (default: from config)")
    ap.add_argument("--force", action="store_true", help="ignore cache")
    ap.add_argument("--limit", type=int, default=None,
                    help="process only first N packages (dry-run)")
    args = ap.parse_args()

    run(
        dataset_root=args.dataset,
        output_dir=args.output,
        config_path=args.config,
        force=args.force,
        seed_override=args.seed,
        limit=args.limit,
    )
