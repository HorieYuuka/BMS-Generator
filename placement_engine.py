#!/usr/bin/env python3
"""
placement_engine.py
Implements the note-placement policy.

3-A: §4 Pool universe / §5 Whitelist / §6 Intensity / §7 Phase segmentation
3-B: §9 Per-measure loop / §10 Primitive grammar
3-C: §11 Placement constraints / §12 Scratch policy
3-D: Output schema / Diagnostics / Conformance checks
     + --intensity / --scratch / --ln support
"""

import json
import math
import os
import random
import sys
import zipfile
from collections import defaultdict, deque

import bms_parser as bmsparser
import resume_state as _resume_state  # v12 §23 carry-over state serialization

# ── Hyperparameters (§5, §7) — defaults at intensity=5 ───────────────────────
FX_DURATION_THRESHOLD    = 1000
FX_ATTACK_THRESHOLD      = 20
FX_ORIGIN_FILTER_ENABLED = True

WHITELIST_DURATION_MAX          = 1000

PHASE_MERGE_RATIO_MAX = 0.30

# ── Hyperparameters (§9, §10) — defaults at intensity=5 ──────────────────────
PLACEMENT_RANDOM_SEED  = 42
STREAM_MAX_CHORD_RATIO = 0.30
STREAM_MAX_SAME_HAND   = 2

# ── Hyperparameters (§12 scratch) — defaults at scratch=5 ────────────────────
SCRATCH_MIN_INTERVAL       = 12
SCRATCH_MAX_PER_MEASURE    = 4
SCRATCH_RUSH_WINDOW        = 3
SCRATCH_RUSH_THRESHOLD     = 3
SCRATCH_RUSH_REST_ENABLED  = True
SCRATCH_RUSH_REST_MEASURES = 4
# §12.2 fallback seed / §12.9 tier-2b functional gate (scratch-fallback,
# not level-scaled). Previously inlined as magic numbers in
# _determine_scratch_seeds; lifted to module constants so the tier-2b
# supplement (§12.9) shares the exact same gate.
SCRATCH_FALLBACK_DURATION_MAX          = 300   # ms
SCRATCH_FALLBACK_MIN_ATTACK_PERCENTILE = 40    # percentile
SCRATCH_FALLBACK_MIN_OCCURRENCE        = 15    # total occurrence

# ── Hyperparameters (LN) ─────────────────────────────────────────────────────
LN_MIN_DURATION_MS = 800
LN_MAX_RATIO       = 0.02
LN_MAX_HOLD_TICKS  = 96     # 2 beats; visible-bar cap regardless of audio dur_ms

DENSITY_REBALANCE_MAX_DELTA = 0.20


# ── Parameter scaling ─────────────────────────────────────────────────────────

def _lerp(v_lo, v_mid, level, v_hi=None):
    """Piecewise linear interpolation.

    Level 1~10  maps v_lo → v_mid  (backward compatible with old 1~10 range).
    Level 11~20 maps v_mid → v_hi  (extended range).
    If v_hi is None, levels 11~20 clamp to v_mid.
    """
    if level <= 10:
        return v_lo + (v_mid - v_lo) * (level - 1) / 9
    if v_hi is None:
        return v_mid
    return v_mid + (v_hi - v_mid) * (level - 10) / 10


def compute_intensity_params(level: int) -> dict:
    # v10: WHITELIST_MIN_OCCURRENCE / ATTACK_PERCENTILE removed (band quota replaces threshold)
    #                                     lv1    lv10   lv20
    return {
        "WHITELIST_DURATION_MAX":          _lerp(700,  1500, level, 3000),
        "STREAM_CHORD_RATIO_MAX":          round(_lerp(0.20, 0.45, level, 0.70), 4),
        "STREAM_MAX_SAME_HAND":            round(_lerp(1,    3,    level, 5)),
        "MEASURE_NOTE_CAP":                round(_lerp(25,   40,   level, 60)),
        "PHASE_MERGE_RATIO_MAX":           round(_lerp(0.40, 0.15, level, 0.05), 4),
        "DENSITY_REBALANCE_MAX_DELTA":     round(_lerp(0.15, 0.28, level, 0.60), 4),
        # 2026-05-02: floor at 13 ticks so gap=12 (16th-note) same-lane is
        # blocked regardless of intensity. Was 12 (which permits 16th-jack
        # at the boundary). Lv15 was producing ~40% same-lane 16th pairs.
        "MIN_JACK_DELTA_TICKS":            round(_lerp(16,   13,   level, 13)),
        "MIN_JACK_DELTA_MS":               round(_lerp(120,  80,   level, 60)),
        # 2026-05-02: cap MAX_JACK_STREAK at 3. Higher streak (4-5) compounds
        # with lowered tick floor to produce repetitive same-lane patterns.
        "MAX_JACK_STREAK":                 round(_lerp(2,    3,    level, 3)),
        # 2026-05-02: cap simultaneous chord size. Excess chord-mates beyond
        # this count become residuals (auto-played as BGM at original timing).
        # Direct cap — does NOT shift timing, unlike the earlier overflow
        # redistribution approach which caused audio interference.
        # Keep tight at high intensity (5-key max) to suppress chord walls.
        "MAX_CHORD_SIZE":                  round(_lerp(3,    4,    level, 5)),
        # DP per-side simultaneous-key cap (one hand). Separate from
        # SP — a hand has 5 fingers, so 4 is the stable max and 5 is reachable at
        # extreme intensity (lv5=2 typical [corpus side≤2=94%], lv10=3, lv15=4,
        # lv20=5). Combined (both-hand) cap = 2× this in _place_measure_dp.
        "DP_MAX_CHORD_SIZE_PER_SIDE":      round(_lerp(2,    3,    level, 5)),
    }


def compute_scratch_params(level: int) -> dict:
    #                                     lv1    lv10   lv20
    return {
        "SCRATCH_MIN_INTERVAL":       round(_lerp(24, 6,  level, 2)),
        "SCRATCH_MAX_PER_MEASURE":    round(_lerp(2,  6,  level, 10)),
        "SCRATCH_RUSH_THRESHOLD":     round(_lerp(2,  5,  level, 8)),
        "SCRATCH_RUSH_REST_MEASURES": round(_lerp(6,  2,  level, 1)),
    }


def _default_params() -> dict:
    # Defaults correspond to intensity=5 / scratch=5 (computed via _lerp).
    # Kept in sync with compute_intensity_params/compute_scratch_params at lv5.
    return {
        "WHITELIST_DURATION_MAX": 1055,
        "STREAM_CHORD_RATIO_MAX": 0.311,
        "STREAM_MAX_SAME_HAND": 2,
        "PHASE_MERGE_RATIO_MAX": 0.289,
        "DENSITY_REBALANCE_MAX_DELTA": 0.208,
        "MEASURE_NOTE_CAP": 32,
        "MIN_JACK_DELTA_TICKS": 15,
        "MIN_JACK_DELTA_MS": 102,
        "MAX_JACK_STREAK": 2,
        "MAX_CHORD_SIZE": 3,
        "DP_MAX_CHORD_SIZE_PER_SIDE": 2,
        "SCRATCH_MIN_INTERVAL": 16,
        "SCRATCH_MAX_PER_MEASURE": 4,
        "SCRATCH_RUSH_THRESHOLD": 3,
        "SCRATCH_RUSH_REST_MEASURES": 4,
    }


# ── ML integration ─────────────────

ML_SAMPLE_RATE = 44100
TOKEN_SELECTION_CONTEXT_WINDOW = 4
LANE_ASSIGNMENT_CONTEXT_WINDOW = 8
PHASE_ENCODE = {"rush": 2, "normal": 1, "rest": 0}
LANE_DIVERSITY_WINDOW = 16
LANE_DIVERSITY_WEIGHT = 1.5
LANE_GLOBAL_BALANCE_WEIGHT = 4.0
SPREAD_BONUS_WEIGHT = 0.6


def _load_ml_model(path, label, warnings_list):
    if not path:
        return None
    try:
        import torch
        model = torch.jit.load(path, map_location="cpu")
        model.eval()
        return model
    except Exception as e:
        warnings_list.append(f"ML model load failed [{label}]: {e}. Falling back to rule-based.")
        return None


def _infer_safe(model, inputs, fallback_fn, label, warnings_list):
    try:
        import torch
        import numpy as _np
        with torch.no_grad():
            tensors = [torch.tensor(x, dtype=torch.float32) for x in inputs]
            output = model(*tensors)
            result = output.detach().cpu().numpy()
            result = _np.where(_np.isneginf(result), -1e9, result)
            if not _np.isfinite(result).all():
                raise ValueError("non-finite output")
            return result
    except Exception as e:
        warnings_list.append(f"ML inference failed [{label}]: {e}. Using fallback.")
        return fallback_fn()


def _make_ml_state(enable, token_path, lane_path, warnings_list):
    state = {
        "enable_token": False,
        "enable_lane": False,
        "token_model": None,
        "lane_model": None,
        "token_path": token_path,
        "lane_path": lane_path,
        "token_load_ok": False,
        "lane_load_ok": False,
        "token_fallback_count": 0,
        "lane_fallback_count": 0,
        "warnings": warnings_list,
    }
    if not enable:
        return state
    state["token_model"] = _load_ml_model(token_path, "TokenSelection", warnings_list)
    state["lane_model"] = _load_ml_model(lane_path, "LaneAssignment", warnings_list)
    state["token_load_ok"] = state["token_model"] is not None
    state["lane_load_ok"] = state["lane_model"] is not None
    state["enable_token"] = state["token_load_ok"]
    state["enable_lane"] = state["lane_load_ok"]
    return state


def _build_ml_context(ml_state, pool_tokens, ta_map, whitelist,
                      key_occ, scratch_occ, bgm_occ, intensity_origin,
                      pct_map, intensity_level):
    """
    Precomputes per-pool-token feature rows used by both models.
    pool_tensor (pool_size, 14) is fixed for the whole run; built once.
    """
    import numpy as _np
    pool_list = sorted(pool_tokens)
    pool_index = {t: i for i, t in enumerate(pool_list)}
    nyq = ML_SAMPLE_RATE / 2.0

    pool_tensor = _np.zeros((len(pool_list), 14), dtype=_np.float32)
    pct_scores = _np.zeros(len(pool_list), dtype=_np.float32)
    for i, t in enumerate(pool_list):
        info = ta_map.get(t) or {}
        decode_ok = bool(info.get("decode_ok"))
        pool_tensor[i, 0] = float(info.get("duration_ms", 0.0)) if decode_ok else 0.0
        pool_tensor[i, 1] = float(info.get("attack_rms", 0.0)) if decode_ok else 0.0
        pool_tensor[i, 2] = float(info.get("attack_peak", 0.0)) if decode_ok else 0.0
        pool_tensor[i, 3] = float(intensity_origin.get(t, 0))
        pool_tensor[i, 4] = float(key_occ.get(t, 0))
        pool_tensor[i, 5] = float(scratch_occ.get(t, 0))
        pool_tensor[i, 6] = float(bgm_occ.get(t, 0))
        pool_tensor[i, 7] = 1.0 if t in whitelist else 0.0
        if decode_ok:
            pool_tensor[i, 8] = float(info.get("spectral_centroid_mean", 0.0)) / nyq
            pool_tensor[i, 9] = float(info.get("spectral_centroid_std", 0.0)) / nyq
            pool_tensor[i, 10] = float(info.get("spectral_flatness_mean", 0.0))
            pool_tensor[i, 11] = float(info.get("low_freq_energy_ratio", 0.0))
            pool_tensor[i, 12] = float(info.get("zero_crossing_rate_mean", 0.0))
            pool_tensor[i, 13] = float(info.get("zero_crossing_rate_std", 0.0))
        pct_scores[i] = float(pct_map.get(t, 0.0))

    return {
        "state": ml_state,
        "pool_list": pool_list,
        "pool_index": pool_index,
        "pool_tensor": pool_tensor,
        "pct_scores": pct_scores,
        "ta_map": ta_map,
        "key_occ": key_occ,
        "scratch_occ": scratch_occ,
        "intensity_origin": intensity_origin,
        "density_rank": (intensity_level - 1) / 9.0,
        "nyq": nyq,
        "token_context": deque(maxlen=TOKEN_SELECTION_CONTEXT_WINDOW),
        "lane_context": deque(maxlen=max(LANE_ASSIGNMENT_CONTEXT_WINDOW, LANE_DIVERSITY_WINDOW)),
        "global_lane_counts": _np.zeros(7, dtype=_np.float32),
    }


def _ml_score_tokens(ml_ctx, measure, phase_label, candidate_count):
    import numpy as _np
    state = ml_ctx["state"]
    if not state["enable_token"]:
        return None
    pool_size = len(ml_ctx["pool_list"])
    measure_tensor = _np.array(
        [float(measure), float(ml_ctx["density_rank"]),
         float(PHASE_ENCODE.get(phase_label, 1)), float(candidate_count)],
        dtype=_np.float32,
    )
    context_tensor = _np.zeros((TOKEN_SELECTION_CONTEXT_WINDOW, 3), dtype=_np.float32)
    history = list(ml_ctx["token_context"])  # most-recent at end
    for i, entry in enumerate(reversed(history)):
        if i >= TOKEN_SELECTION_CONTEXT_WINDOW:
            break
        context_tensor[i, 0] = float(entry["tkey_delta_base"] - measure * 192)
        context_tensor[i, 1] = float(entry["placed_count"])
        context_tensor[i, 2] = float(entry["mean_attack_rms"])

    def fallback():
        state["token_fallback_count"] += 1
        return ml_ctx["pct_scores"].copy()

    scores = _infer_safe(
        state["token_model"],
        [measure_tensor, ml_ctx["pool_tensor"], context_tensor],
        fallback, "TokenSelection", state["warnings"],
    )
    if scores is None or scores.shape != (pool_size,):
        state["token_fallback_count"] += 1
        return ml_ctx["pct_scores"].copy()
    return scores


def _ml_select_lane(ml_ctx, token, idx192, measure, density_rank, phase_label,
                    total_placed, available, rng):
    """
    Returns chosen lane via ML scoring, or None to signal the caller to use
    the rule-based centroid path (core spec §9.7).
    Inference-side failures (exception, shape mismatch) increment
    lane_fallback_count; on exception we synthesize a fisher_yates-like
    score vector locally and continue (Layer A in addon §21.5.4). On shape
    mismatch and on upstream bypass (lane model disabled, empty `available`)
    we return None for caller-side fallback (Layer B).
    `available` is the constraint-passing lane list.
    """
    import numpy as _np
    state = ml_ctx["state"]
    if not state["enable_lane"] or not available:
        return None  # signal: caller uses rule-based path
    info = ml_ctx["ta_map"].get(token) or {}
    decode_ok = bool(info.get("decode_ok"))
    nyq = ml_ctx["nyq"]

    def feat(key):
        return float(info.get(key, 0.0)) if decode_ok else 0.0

    event_tensor = _np.array([
        feat("duration_ms"), feat("attack_rms"), feat("attack_peak"),
        float(ml_ctx["intensity_origin"].get(token, 0)),
        float(ml_ctx["key_occ"].get(token, 0)),
        float(ml_ctx["scratch_occ"].get(token, 0)),
        float(idx192), float(density_rank),
        float(PHASE_ENCODE.get(phase_label, 1)),
        float(total_placed),
        feat("spectral_centroid_mean") / nyq,
        feat("spectral_centroid_std") / nyq,
        feat("spectral_flatness_mean"),
        feat("low_freq_energy_ratio"),
        feat("zero_crossing_rate_mean"),
        feat("zero_crossing_rate_std"),
    ], dtype=_np.float32)

    context_tensor = _np.zeros((LANE_ASSIGNMENT_CONTEXT_WINDOW, 5), dtype=_np.float32)
    history = list(ml_ctx["lane_context"])
    current_tkey = measure * 192 + idx192
    for i, entry in enumerate(reversed(history)):
        if i >= LANE_ASSIGNMENT_CONTEXT_WINDOW:
            break
        context_tensor[i, 0] = float(entry["tkey"] - current_tkey)
        context_tensor[i, 1] = float(LANE_INDEX[entry["lane"]])
        context_tensor[i, 2] = float(entry["attack_rms"])
        context_tensor[i, 3] = float(entry["idx192"])
        context_tensor[i, 4] = 0.0
    # padded entries already zero-init; mark is_padded=1.0
    real_count = min(len(history), LANE_ASSIGNMENT_CONTEXT_WINDOW)
    for i in range(real_count, LANE_ASSIGNMENT_CONTEXT_WINDOW):
        context_tensor[i, 4] = 1.0

    mask_tensor = _np.array(
        [1.0 if l in available else 0.0 for l in KEY_LANES], dtype=_np.float32
    )

    def fallback():
        state["lane_fallback_count"] += 1
        # produce a scrambled scoring that mimics fisher_yates first-pick
        order = fisher_yates_shuffle(KEY_LANES, rng)
        scrambled = _np.zeros(7, dtype=_np.float32)
        for rank, lane in enumerate(order):
            scrambled[KEY_LANES.index(lane)] = float(7 - rank)
        return scrambled

    scores = _infer_safe(
        state["lane_model"],
        [event_tensor, context_tensor, mask_tensor],
        fallback, "LaneAssignment", state["warnings"],
    )
    if scores is None or scores.shape != (7,):
        state["lane_fallback_count"] += 1
        return None
    adjusted = scores.copy()

    # ── Lane diversity penalty: penalize recently overused lanes ────────
    full_history = list(ml_ctx["lane_context"])
    recent = full_history[-LANE_DIVERSITY_WINDOW:]
    if recent:
        lane_counts = _np.zeros(7, dtype=_np.float32)
        for entry in recent:
            li = LANE_INDEX[entry["lane"]] - 1
            lane_counts[li] += 1.0
        total_recent = float(len(recent))
        adjusted -= LANE_DIVERSITY_WEIGHT * (lane_counts / total_recent)

    # ── Same-tkey spread bonus: prefer lanes far from already-placed at same tkey ──
    current_tkey = measure * 192 + idx192
    placed_at_tkey = [
        LANE_INDEX[e["lane"]] - 1 for e in full_history if e["tkey"] == current_tkey
    ]
    if placed_at_tkey:
        for li in range(7):
            min_dist = min(abs(li - pi) for pi in placed_at_tkey)
            adjusted[li] += SPREAD_BONUS_WEIGHT * float(min_dist)

    # ── Global lane balance: penalize lanes that exceed 1/7 proportion ────
    glc = ml_ctx["global_lane_counts"]
    total_global = float(glc.sum())
    if total_global >= 14:  # enough history to be meaningful
        ideal = total_global / 7.0
        excess = (glc - ideal) / ideal  # positive = overused
        adjusted -= LANE_GLOBAL_BALANCE_WEIGHT * _np.clip(excess, 0.0, None)

    masked = adjusted.copy()
    masked[mask_tensor == 0.0] = -_np.inf
    if not (masked > -_np.inf).any():
        return None
    best_idx = int(_np.argmax(masked))
    ml_ctx["global_lane_counts"][best_idx] += 1.0
    return KEY_LANES[best_idx]


# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT_DIR       = os.path.dirname(os.path.abspath(__file__))
ZIP_PATH       = os.path.join(ROOT_DIR, "[- 4 5] A D D i c T i O N 4 5 0 0 0 0 0.zip")
TARGET_BMS     = "Addiction_INFERNO24.bms"
TOKEN_ANALYSIS = os.path.join(ROOT_DIR, "token_analysis.json")
RESULT_PATH    = os.path.join(ROOT_DIR, "placement_result.json")

# ── Channel sets ───────────────────────────────────────────────────────────────
KEY_CHANNELS     = {"11", "12", "13", "14", "15", "18", "19"}
SCRATCH_CHANNELS = {"16"}
BGM_CHANNELS     = {"01"}

# ── Lane constants ──────────────────────────────────────────────────────────────
KEY_LANES = ["P1_KEY1", "P1_KEY2", "P1_KEY3", "P1_KEY4",
             "P1_KEY5", "P1_KEY6", "P1_KEY7"]
LANE_INDEX = {
    "P1_SCR": 0, "P1_KEY1": 1, "P1_KEY2": 2, "P1_KEY3": 3,
    "P1_KEY4": 4, "P1_KEY5": 5, "P1_KEY6": 6, "P1_KEY7": 7,
}
IDX_TO_KEY_LANE = {1: "P1_KEY1", 2: "P1_KEY2", 3: "P1_KEY3", 4: "P1_KEY4",
                   5: "P1_KEY5", 6: "P1_KEY6", 7: "P1_KEY7"}
HAND_MAP = {
    "P1_KEY1": "left",  "P1_KEY2": "left",  "P1_KEY3": "left",
    "P1_KEY4": "right", "P1_KEY5": "right", "P1_KEY6": "right", "P1_KEY7": "right",
}
LEFT_KEY_LANES_SET  = frozenset({"P1_KEY1", "P1_KEY2", "P1_KEY3"})
RIGHT_KEY_LANES_SET = frozenset({"P1_KEY4", "P1_KEY5", "P1_KEY6", "P1_KEY7"})
EIGHTH_NOTE_POSITIONS = frozenset(i * 24 for i in range(8))
CH_TO_KEY_LANE = {
    "11": "P1_KEY1", "12": "P1_KEY2", "13": "P1_KEY3",
    "14": "P1_KEY4", "15": "P1_KEY5", "18": "P1_KEY6", "19": "P1_KEY7",
}

# ── DP (double-play) lanes ─────────────────────────────────────
# The P2 side mirrors P1. Side-local placement (addon §2) runs the SP machine on
# P1_* lanes for BOTH hands; the right-side output is remapped P1_*→P2_* before
# emission. Active only when dp=True — the SP path never references these.
P2_KEY_LANES = ["P2_KEY1", "P2_KEY2", "P2_KEY3", "P2_KEY4",
                "P2_KEY5", "P2_KEY6", "P2_KEY7"]
P1_TO_P2_LANE = {
    "P1_SCR": "P2_SCR", "P1_KEY1": "P2_KEY1", "P1_KEY2": "P2_KEY2",
    "P1_KEY3": "P2_KEY3", "P1_KEY4": "P2_KEY4", "P1_KEY5": "P2_KEY5",
    "P1_KEY6": "P2_KEY6", "P1_KEY7": "P2_KEY7",
}


# ── §4  Pool universe ──────────────────────────────────────────────────────────

def load_bms_bytes() -> bytes:
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        for name in zf.namelist():
            if os.path.basename(name) == TARGET_BMS:
                return zf.read(name)
    sys.exit(f"ERROR: {TARGET_BMS} not found in {ZIP_PATH}")


def build_pool_universe(events):
    key_occ, scratch_occ, bgm_occ = defaultdict(int), defaultdict(int), defaultdict(int)
    pool_tokens, pool_events = set(), []
    for ev in events:
        etype = ev.get("type")
        if etype == "Tap":
            token, ch, measure, idx192 = ev["token"], ev["rawChannel"], ev["measure"], ev["idx192"]
        elif etype == "Long":
            token, ch, measure, idx192 = (ev["tokenStart"], ev["rawChannelStart"],
                                          ev["measureStart"], ev["idx192Start"])
        elif etype == "BGM":
            token, ch, measure, idx192 = ev["token"], ev["rawChannel"], ev["measure"], ev["idx192"]
        else:
            continue
        pool_tokens.add(token); pool_events.append((measure, idx192, token))
        if ch in KEY_CHANNELS:     key_occ[token] += 1
        elif ch in SCRATCH_CHANNELS: scratch_occ[token] += 1
        elif ch in BGM_CHANNELS:     bgm_occ[token] += 1
    measure_max = max((m for m, _, _ in pool_events), default=0)
    return pool_tokens, key_occ, scratch_occ, bgm_occ, pool_events, measure_max


def compute_attack_percentile(pool_tokens, ta):
    valid = [(ta[t]["attack_rms"], ta[t]["attack_peak"], t)
             for t in pool_tokens if ta.get(t, {}).get("decode_ok")]
    N = len(valid)
    pct_map = {}
    if N == 0: return pct_map
    if N == 1: pct_map[valid[0][2]] = 50.0; return pct_map
    # token (x[2]) tie-break removes PYTHONHASHSEED-dependent ordering when
    # both attack_rms and attack_peak coincide.
    valid.sort(key=lambda x: (x[0], x[1], x[2]))
    i = 0
    while i < N:
        j = i
        while j < N and valid[j][0] == valid[i][0] and valid[j][1] == valid[i][1]:
            j += 1
        avg_rank = (i + j - 1) / 2.0
        pct = (avg_rank / (N - 1)) * 100.0
        for k in range(i, j): pct_map[valid[k][2]] = pct
        i = j
    return pct_map


def compute_intensity_origin(pool_tokens, key_occ, scratch_occ):
    return {t: (1 if key_occ.get(t, 0) > 0 or scratch_occ.get(t, 0) > 0 else 0) for t in pool_tokens}


def classify_fx(pool_tokens, ta, pct_map, intensity_origin):
    result = {}
    for token in pool_tokens:
        info = ta.get(token)
        if not info or not info.get("decode_ok"):
            result[token] = {"is_background_fx": False, "status": "unknown"}; continue
        is_fx = (info["duration_ms"] > FX_DURATION_THRESHOLD or
                 pct_map.get(token, 0.0) <= FX_ATTACK_THRESHOLD or
                 (intensity_origin.get(token, 0) == 0 and FX_ORIGIN_FILTER_ENABLED))
        result[token] = {"is_background_fx": is_fx, "status": "classified"}
    return result


def build_whitelist(pool_tokens, ta, key_occ, scratch_occ, bgm_occ, pct_map, fx_info, params=None):
    p = params or _default_params()
    wl_dur = p.get("WHITELIST_DURATION_MAX", WHITELIST_DURATION_MAX)

    # ── Step 1: hard filter (fx / unknown only) ──────────────────────────
    hard_excluded = {}
    eligible = []
    # sorted() removes PYTHONHASHSEED-dependent set iteration ordering — eligible
    # list order propagates through bands → quota selection → whitelist.
    for token in sorted(pool_tokens):
        info = ta.get(token)
        decode_ok = bool(info and info.get("decode_ok"))
        fx = fx_info.get(token, {})
        if fx.get("is_background_fx"):
            hard_excluded[token] = "fx"
        elif fx.get("status") == "unknown":
            hard_excluded[token] = "unknown"
        elif not decode_ok:
            hard_excluded[token] = "unknown"
        else:
            eligible.append(token)

    # ── Step 2: classify into 3 spectral bands ───────────────────────────
    centroids = {}
    for tok in eligible:
        info = ta.get(tok)
        if info and info.get("spectral_centroid_mean", 0) > 0:
            centroids[tok] = info["spectral_centroid_mean"]

    # Fallback: if no centroid data, use legacy behavior
    if len(centroids) < 6:
        whitelist = set(eligible)
        excluded = dict(hard_excluded)
        exc_counts = defaultdict(int)
        for r in excluded.values():
            exc_counts[r] += 1
        return whitelist, excluded, dict(exc_counts)

    vals = sorted(centroids.values())
    lo_thr = vals[len(vals) // 3]
    hi_thr = vals[2 * len(vals) // 3]

    def band(tok):
        c = centroids.get(tok, 0)
        if c < lo_thr:
            return "lo"
        if c < hi_thr:
            return "mid"
        return "hi"

    bands = {"lo": [], "mid": [], "hi": []}
    no_centroid = []
    for tok in eligible:
        if tok in centroids:
            bands[band(tok)].append(tok)
        else:
            no_centroid.append(tok)

    # ── Step 3: rank within each band, select quota ──────────────────────
    def rank_score(tok):
        """Higher = more suitable for whitelist. Occurrence primary, duration penalty."""
        total_occ = key_occ.get(tok, 0) + scratch_occ.get(tok, 0) + bgm_occ.get(tok, 0)
        info = ta.get(tok) or {}
        dur = info.get("duration_ms", 0)
        dur_penalty = max(0.0, (dur - wl_dur) / wl_dur) if dur > wl_dur else 0.0
        return total_occ - dur_penalty * 5.0

    total_eligible = sum(len(b) for b in bands.values())
    # Each band gets proportional quota based on pool composition
    whitelist = set()
    soft_excluded = {}

    for b_name in ("lo", "mid", "hi"):
        pool = bands[b_name]
        if not pool:
            continue
        # Quota: proportional to band size, minimum 1/3 of total target
        band_ratio = len(pool) / total_eligible if total_eligible > 0 else 1.0 / 3.0
        # Target whitelist size per band: at least 33% of pool, scaled by intensity
        # 2026-05-03: raised from 15% to 25% — listening tests showed rare tokens
        # (occurrence ≤2) often dropped to band_quota even when they were the
        # melodic highlight (e.g., Lepontinia m16 8X/8Y/9B). 25% lets more
        # melodic-but-rare tokens reach placement.
        # 2026-05-03 (later): trimmed to 20% — at lv10 ML/RB diverge less when
        # whitelist is too generous. Rare rescue (below) still recovers the
        # melodic-highlight tokens 25% was meant to protect.
        quota = max(3, round(len(pool) * 0.20))
        # token tie-break stabilizes order when rank_score collides.
        pool_ranked = sorted(pool, key=lambda t: (-rank_score(t), t))
        for i, tok in enumerate(pool_ranked):
            if i < quota:
                whitelist.add(tok)
            else:
                soft_excluded[tok] = "band_quota"

    # 2026-05-03: rescue rare-but-not-fx tokens. FX was hard-filtered in step 1,
    # so anything with reason="band_quota" is already non-FX. Tokens with very
    # low occurrence (≤3) likely encode melodic highlights authored sparsely;
    # the band-rank cutoff would otherwise drop them whenever the band has many
    # popular tokens. Restore these to whitelist so they reach placement.
    RARE_OCCURRENCE_THRESHOLD = 3
    rare_rescued = []
    # sorted() pins iteration order; soft_excluded was built from PYTHONHASHSEED-
    # dependent eligible/pool order originally.
    for tok in sorted(soft_excluded.keys()):
        if soft_excluded[tok] != "band_quota":
            continue
        total_occ = key_occ.get(tok, 0) + scratch_occ.get(tok, 0) + bgm_occ.get(tok, 0)
        if 0 < total_occ <= RARE_OCCURRENCE_THRESHOLD:
            whitelist.add(tok)
            del soft_excluded[tok]
            rare_rescued.append(tok)

    # Tokens without centroid data: apply occurrence-only filter
    for tok in no_centroid:
        total_occ = key_occ.get(tok, 0) + scratch_occ.get(tok, 0) + bgm_occ.get(tok, 0)
        if total_occ > 5:
            whitelist.add(tok)
        else:
            soft_excluded[tok] = "occurrence"

    excluded = {**hard_excluded, **soft_excluded}
    exc_counts = defaultdict(int)
    for r in excluded.values():
        exc_counts[r] += 1
    return whitelist, excluded, dict(exc_counts)




# ── §7  Phase segmentation ─────────────────────────────────────────────────────

def _lerp_percentile(sorted_vals, pct):
    N = len(sorted_vals)
    if N == 1: return float(sorted_vals[0])
    idx = pct / 100.0 * (N - 1)
    lo = int(idx); hi = min(lo + 1, N - 1)
    return sorted_vals[lo] + (idx - lo) * (sorted_vals[hi] - sorted_vals[lo])


def segment_phases(pool_events, measure_max, params=None):
    num_measures = measure_max + 1
    block_starts = list(range(0, num_measures, 4)) or [0]
    blocks = []
    for bs in block_starts:
        evs = [(m, i, t) for (m, i, t) in pool_events if bs <= m < bs + 4]
        pos_tokens = defaultdict(set)
        for (m, i, t) in evs: pos_tokens[(m, i)].add(t)
        triple = sum(1 for toks in pos_tokens.values() if len(toks) >= 3)
        blocks.append({"start": bs, "end": min(bs + 4, num_measures),
                       "event_count": len(evs), "triple_chord_count": triple,
                       "phase_score": len(evs) + 1.5 * triple})
    n = len(blocks)
    if n == 0: return []
    for i in range(n):
        w = blocks[max(0, i - 3): i + 1]
        blocks[i]["smoothed_score"] = sum(b["phase_score"] for b in w) / len(w)
    gm = sum(b["smoothed_score"] for b in blocks) / n
    ss = sorted(b["smoothed_score"] for b in blocks)
    rush_thr, rest_thr = _lerp_percentile(ss, 85.0), _lerp_percentile(ss, 20.0)
    for b in blocks:
        s = b["smoothed_score"]
        if s >= rush_thr:                        b["phase"] = "rush"
        elif s <= rest_thr and s <= gm:          b["phase"] = "rest"
        else:                                    b["phase"] = "normal"
    return _merge_blocks(blocks, params)


def _merge_blocks(blocks, params=None):
    p = params or _default_params()
    merge_ratio = p.get("PHASE_MERGE_RATIO_MAX", PHASE_MERGE_RATIO_MAX)
    changed = True
    while changed:
        changed = False; i = 0; new_blocks = []
        while i < len(blocks):
            if i + 1 < len(blocks):
                a, b = blocks[i], blocks[i + 1]
                sa_sz, sb_sz = a["end"] - a["start"], b["end"] - b["start"]
                if a["phase"] == b["phase"] and sa_sz + sb_sz <= 8:
                    sa, sb = a["smoothed_score"], b["smoothed_score"]
                    mx = max(sa, sb)
                    if mx == 0 or abs(sa - sb) / mx <= merge_ratio:
                        new_blocks.append({"start": a["start"], "end": b["end"],
                                           "phase": a["phase"],
                                           "smoothed_score": (sa * sa_sz + sb * sb_sz) / (sa_sz + sb_sz)})
                        i += 2; changed = True; continue
            new_blocks.append(blocks[i]); i += 1
        blocks = new_blocks
    return blocks


# ── Helpers ────────────────────────────────────────────────────────────────────

def fisher_yates_shuffle(items, rng):
    items = list(items)
    for i in range(len(items) - 1, 0, -1):
        j = rng.randint(0, i); items[i], items[j] = items[j], items[i]
    return items

def _derive_chord_hand(lanes):
    """Derive a chord's effective hand label (left / right / balanced) from
    its placed lanes. Used by §23.7 boundary lookahead to predict whether N+1
    would extend N's hand streak."""
    if not lanes:
        return "balanced"
    hands = {HAND_MAP[l] for l in lanes if l in HAND_MAP}
    if len(hands) == 1:
        return hands.pop()
    return "balanced"


def normalize_lookahead(la_data):
    """§23.7 lookahead JSON → engine-internal dict {tkey, lanes, hand, tokens}.

    Accepts three shapes:
    - {"tkey": <int>, "lanes": [...], "tokens": [...]}      already normalized
    - {"measure": <int>, "idx192": <int>, "lanes": [...]}   pre-keyed object
    - [event, event, ...]                                   raw events list
      → smallest (measure, idx192) tkey 의 chord 추출.
    """
    if not la_data:
        raise ValueError("next_chord_lookahead JSON is empty")
    if isinstance(la_data, dict):
        if "tkey" in la_data:
            tkey = int(la_data["tkey"])
        elif "measure" in la_data and "idx192" in la_data:
            tkey = int(la_data["measure"]) * 192 + int(la_data["idx192"])
        else:
            raise ValueError(
                "next_chord_lookahead object must contain either 'tkey' or "
                "both 'measure' and 'idx192'")
        lanes = set(la_data.get("lanes") or [])
        tokens = list(la_data.get("tokens") or [])
    elif isinstance(la_data, list):
        def _ev_tkey(ev):
            m = ev.get("measure", ev.get("measure_start"))
            i = ev.get("idx192", ev.get("idx192_start"))
            if m is None or i is None:
                raise ValueError("lookahead event missing measure/idx192")
            return m * 192 + i
        min_tkey = min(_ev_tkey(ev) for ev in la_data)
        chord_evs = [ev for ev in la_data if _ev_tkey(ev) == min_tkey]
        tkey = min_tkey
        lanes = {ev["lane"] for ev in chord_evs if ev.get("lane")}
        tokens = [ev.get("token") or ev.get("tokenStart") for ev in chord_evs]
    else:
        raise ValueError(
            "next_chord_lookahead JSON must be a list of events or an object")
    return {
        "tkey": tkey,
        "lanes": lanes,
        "tokens": tokens,
        "hand": _derive_chord_hand(lanes),
    }


def _update_hand(last_hand, streak, lane):
    hand = HAND_MAP[lane]
    if last_hand == "balanced": return hand, 1
    return (hand, streak + 1) if hand == last_hand else (hand, 1)

def _get_phase(measure, phase_blocks):
    for block in phase_blocks:
        if block["start"] <= measure < block["end"]: return block["phase"]
    return "normal"


# ── ChordBurst feasibility ────────────────────────────────────────────────────

def _chordburst_feasible(curr_cands, next_cands_or_none):
    curr_by_pos = defaultdict(set)
    for (idx192, token, ia, io) in curr_cands: curr_by_pos[idx192].add(token)
    if any(len(t) >= 3 for t in curr_by_pos.values()): return True
    if next_cands_or_none is None: return False
    if not all(p in curr_by_pos for p in EIGHTH_NOTE_POSITIONS): return False
    next_by_pos = defaultdict(set)
    for (idx192, token, ia, io) in next_cands_or_none: next_by_pos[idx192].add(token)
    return all(p in next_by_pos for p in EIGHTH_NOTE_POSITIONS)


# ── Scratch seed selection ───────────────────────────────────────────────────

def _determine_scratch_seeds(pool_tokens, key_occ, scratch_occ, bgm_occ,
                              whitelist, ta_map, pct_map, fx_info):
    primary = sorted([t for t in pool_tokens if scratch_occ.get(t, 0) > 0],
                     key=lambda t: -scratch_occ.get(t, 0))
    if primary: return primary, "primary"
    fallback = []
    for token in whitelist:
        info = ta_map.get(token)
        if not info or not info.get("decode_ok"): continue
        if info["duration_ms"] > SCRATCH_FALLBACK_DURATION_MAX: continue
        if pct_map.get(token, 0.0) < SCRATCH_FALLBACK_MIN_ATTACK_PERCENTILE: continue
        total_occ = key_occ.get(token, 0) + scratch_occ.get(token, 0) + bgm_occ.get(token, 0)
        if total_occ < SCRATCH_FALLBACK_MIN_OCCURRENCE: continue
        if fx_info.get(token, {}).get("is_background_fx"): continue
        fallback.append(token)
    if fallback:
        fallback.sort(key=lambda t: (-(key_occ.get(t, 0) + scratch_occ.get(t, 0) + bgm_occ.get(t, 0)), t))
        return fallback, "fallback"
    return [], "disabled"


CENTROID_EPSILON_RANDOM = 0.30  # probability of picking random lane instead of centroid-preferred

# Token→lane affinity (v12 §9.8, NEW 2026-06-11): "same sound = same finger".
# Human charts keep a token's placements concentrated on one lane within a
# section (6-song corpus: HUMAN modal-lane share 0.542 vs RB 0.348); the
# relative centroid walk + ε-greedy alone lets the same token wander. When a
# token was placed within the last AFFINITY_WINDOW measures and its
# remembered lane survives all hard constraints, reuse that lane with
# probability AFFINITY_PROB. Soft preference: never overrides §11 gates.
TOKEN_LANE_AFFINITY_PROB   = 0.90
TOKEN_LANE_AFFINITY_WINDOW = 32   # measures; affinity memory expires beyond this

def _centroid_lane_select(token, avail, ta, prev_lane_idx, prev_centroid, rng, step_unit=300.0):
    """Pick lane based on relative centroid change from previous note.
    Higher centroid → move right, lower → move left.
    step_unit: Hz per 1 lane step (auto-calibrated per song).
    ε-greedy: CENTROID_EPSILON_RANDOM probability of pure random (diversification).
    Falls back to random if centroid unavailable."""
    if not ta:
        return fisher_yates_shuffle(avail, rng)[0], prev_lane_idx, prev_centroid
    # ε-greedy exploration: occasional random pick breaks centroid drift
    if rng.random() < CENTROID_EPSILON_RANDOM:
        lane = fisher_yates_shuffle(avail, rng)[0]
        info = ta.get(token)
        centroid_val = info.get("spectral_centroid_mean", 0) if info and info.get("decode_ok") else 0
        return lane, LANE_INDEX[lane] - 1, centroid_val if centroid_val > 0 else prev_centroid
    info = ta.get(token)
    if not info or not info.get("decode_ok"):
        lane = fisher_yates_shuffle(avail, rng)[0]
        return lane, LANE_INDEX[lane] - 1, prev_centroid
    centroid = info.get("spectral_centroid_mean", 0)
    if centroid <= 0 or prev_centroid is None:
        lane = fisher_yates_shuffle(avail, rng)[0]
        return lane, LANE_INDEX[lane] - 1, centroid if centroid > 0 else prev_centroid

    # Relative movement: saturating curve — sensitive to small changes, caps at large
    import math
    delta = centroid - prev_centroid
    abs_delta = abs(delta)
    magnitude = 4.0 * (1.0 - math.exp(-abs_delta / step_unit))  # 0~4, saturates
    step = magnitude if delta >= 0 else -magnitude

    preferred_idx = prev_lane_idx + step
    preferred_idx = max(0.0, min(6.0, preferred_idx))

    # Pick available lane closest to preferred position
    avail_with_dist = []
    for lane in avail:
        lane_idx = LANE_INDEX[lane] - 1
        dist = abs(lane_idx - preferred_idx)
        avail_with_dist.append((dist, rng.random(), lane))
    avail_with_dist.sort()
    chosen = avail_with_dist[0][2]
    return chosen, LANE_INDEX[chosen] - 1, centroid


# ── Scale-aware time axis (v12 §11.5 revision, 2026-06-11) ──────────────────
# Raw tkey (= measure*192 + idx192) assumes every measure spans 4 beats. A
# #xxx02 measure scale breaks that: in a scale-0.5 measure the same tick gap
# is half the real time, so any ms<->tick conversion done on raw ticks is
# wrong. These helpers map raw tkeys onto a "beat-true" tick axis where
# 48 ticks = 1 beat ALWAYS holds (the same fix NoteAttributes_v2 §2 applied
# on the analysis side as abs_tick). On charts with no non-1.0 scale the
# mapping is the identity (ints in, ints out) so all arithmetic stays
# byte-identical to the pre-scale-aware era.

def _normalize_measure_scale(measure_scale):
    """Coerce a #xxx02 scale map (int or str keys, parser or JSON origin) to
    {int: float}, keeping only valid non-trivial (>0, != 1.0) entries."""
    out = {}
    for k, v in (measure_scale or {}).items():
        try:
            m, s = int(k), float(v)
        except (TypeError, ValueError):
            continue
        if s > 0 and s != 1.0:
            out[m] = s
    return out


def build_scaled_tick_fn(measure_scale, measure_max=0):
    """Return f(raw_tkey) -> beat-true tick position.

    Identity (returns the int unchanged) when the chart has no non-trivial
    scale. Negative tkeys (jack_state sentinel -999) pass through unchanged
    so sentinel deltas stay huge.
    """
    sc = _normalize_measure_scale(measure_scale)
    if not sc:
        return lambda tkey: tkey
    last_m = max(measure_max, max(sc)) + 1
    base = [0.0] * (last_m + 2)
    for m in range(last_m + 1):
        base[m + 1] = base[m] + 192.0 * sc.get(m, 1.0)

    def scaled(tkey):
        if tkey < 0:
            return float(tkey)
        m, idx = divmod(tkey, 192)
        if m > last_m:
            return base[last_m + 1] + (m - last_m - 1) * 192.0 + idx
        return base[m] + idx * sc.get(m, 1.0)
    return scaled


def advance_scaled_ticks(start_tkey, scaled_ticks, measure_scale=None):
    """Raw tkey lying `scaled_ticks` beat-true ticks after `start_tkey`.

    Inverse companion of build_scaled_tick_fn, used for caps expressed in
    beats (e.g. LN_MAX_HOLD_TICKS = 96 = 2 beats). Trivial-scale charts get
    exact legacy behavior (start + int ticks)."""
    sc = _normalize_measure_scale(measure_scale)
    if not sc:
        return start_tkey + int(round(scaled_ticks))
    m, idx = divmod(start_tkey, 192)
    remaining = float(scaled_ticks)
    max_scaled_m = max(sc)
    while True:
        s = sc.get(m, 1.0)
        left_scaled = (192 - idx) * s
        if remaining < left_scaled or m > max_scaled_m:
            return m * 192 + idx + int(round(remaining / s))
        remaining -= left_scaled
        m += 1
        idx = 0


# ── Constrained placement ────────────────────────────────────────────────────

def _place_measure_constrained(curr_cands, rng, hand_state, jack_state,
                                measure, is_chordburst, params=None,
                                ml_ctx=None, phase_label="normal",
                                ta=None, centroid_state=None,
                                bpm_lookup=None, jack_streak=None,
                                next_chord_lookahead=None,
                                scaled_tick=None,
                                token_lane_memory=None,
                                external_tkey_count=None,
                                combined_chord_cap=None):
    """
    external_tkey_count / combined_chord_cap: DP combined (both-hand)
        chord cap. When placing the second side, external_tkey_count maps
        tkey→notes the first side already placed there; a candidate is dropped
        if own+external would reach combined_chord_cap. Keeps both-side chord
        size bounded while max_chord_size (= MAX_CHORD_SIZE_PER_SIDE) bounds one
        hand. None on the SP path / first side → no combined check.
    next_chord_lookahead: optional dict {tkey: int, lanes: set[str], hand: "left"/"right"/"balanced"}
        — N+1 의 첫 chord 정보 (§23.7 boundary lookahead). 마지막 measure 의 마지막 chord
        처리 시만 적용되어 jack/hand 양방향 hard constraint inject. 다른 chord 에는 영향 없음.
    """
    p = params or _default_params()
    chord_ratio_max = p.get("STREAM_CHORD_RATIO_MAX", STREAM_MAX_CHORD_RATIO)
    max_same_hand   = p.get("STREAM_MAX_SAME_HAND", STREAM_MAX_SAME_HAND)
    measure_note_cap = p.get("MEASURE_NOTE_CAP", 999)
    min_jack_ticks = p.get("MIN_JACK_DELTA_TICKS", 12)
    min_jack_ms    = p.get("MIN_JACK_DELTA_MS", 100)
    max_jack_streak = p.get("MAX_JACK_STREAK", 2)
    max_chord_size = p.get("MAX_CHORD_SIZE", 7)
    # jack_streak: dict lane -> current streak count (mutated, shared across measures)
    if jack_streak is None:
        jack_streak = defaultdict(int)
    # scaled_tick: beat-true tick mapping (scale-aware jack deltas). Identity
    # when the caller has no #xxx02 scale information.
    if scaled_tick is None:
        scaled_tick = lambda t: t

    # §23.7 boundary lookahead: precompute the last tkey in this measure so we
    # can recognize when we're placing the last chord (where N+1 lookahead applies).
    last_tkey_in_measure = None
    if next_chord_lookahead is not None and curr_cands:
        last_tkey_in_measure = max(measure * 192 + idx for (idx, _, _, _) in curr_cands)

    placed, residuals = [], {}
    positions_with_notes, chord_positions = set(), set()
    used_at_pos, tokens_at_pos = defaultdict(set), defaultdict(set)
    left_count = right_count = hand_bal_ct = same_hand_ct = affinity_ct = 0
    last_hand, streak = hand_state

    # v10 §11.5: defer streak update to tkey boundaries (chord mates shouldn't reset streak)
    _chord_tkey = -1
    _chord_lanes = set()

    def _commit_chord_streak():
        """Apply streak updates for the just-completed chord/tkey."""
        if _chord_tkey < 0:
            return
        for _l in KEY_LANES:
            if _l in _chord_lanes:
                jack_streak[_l] = jack_streak.get(_l, 0) + 1
            else:
                jack_streak[_l] = 0

    for (idx192, token, ia, io) in curr_cands:
        tkey = measure * 192 + idx192
        # Chord boundary: commit previous chord's streak update
        if tkey != _chord_tkey:
            _commit_chord_streak()
            _chord_tkey = tkey
            _chord_lanes = set()
        if len(placed) >= measure_note_cap:
            residuals[(measure, idx192, token)] = "measure_cap"; continue
        if is_chordburst and idx192 % 24 != 0: continue
        if token in tokens_at_pos[idx192]:
            residuals[(measure, idx192, token)] = "collision"; continue
        is_new = idx192 not in positions_with_notes
        is_second = not is_new and idx192 not in chord_positions
        if not is_chordburst and is_second:
            if len(chord_positions) + 1 > chord_ratio_max * len(positions_with_notes):
                residuals[(measure, idx192, token)] = "no_lane_available"; continue
        # 2026-05-02: hard chord size cap — drops to BGM (preserves timing)
        if not is_new and len(used_at_pos[idx192]) >= max_chord_size:
            residuals[(measure, idx192, token)] = "chord_size_cap"; continue
        # DP combined (both-hand) cap — this side's count plus the
        # other side's already-placed count at this tkey must stay under the cap.
        if combined_chord_cap is not None and external_tkey_count is not None:
            if len(used_at_pos[idx192]) + external_tkey_count.get(tkey, 0) >= combined_chord_cap:
                residuals[(measure, idx192, token)] = "chord_size_cap"; continue
        avail = [l for l in KEY_LANES if l not in used_at_pos[idx192]]
        if not avail:
            residuals[(measure, idx192, token)] = "collision"; continue
        # v10 §11.5: BPM-aware jack floor.
        # 2026-06-11: deltas are measured on the beat-true (scale-aware) tick
        # axis. On that axis 1 tick = 1250/bpm ms regardless of #xxx02 scale,
        # so both the grid floor (MIN_JACK_DELTA_TICKS) and the ms floor keep
        # their intended meaning inside scaled measures. Raw-tick deltas
        # understate real time in scale<1 measures (and overstate in scale>1).
        cur_bpm = bpm_lookup(tkey) if bpm_lookup else 130.0
        import math as _m_jk
        bpm_floor = _m_jk.ceil(min_jack_ms * cur_bpm / 1250.0)
        effective_min_ticks = max(min_jack_ticks, bpm_floor)
        _st = scaled_tick(tkey)
        avail = [l for l in avail
                 if round(_st - scaled_tick(jack_state.get(l, -999)), 6) >= effective_min_ticks]
        # Per-lane streak cap
        avail = [l for l in avail if jack_streak.get(l, 0) < max_jack_streak]
        # §23.7 boundary lookahead (E-β): forward jack constraint for the last
        # chord in the last measure of a reroll. Reject any lane that would
        # collide with N+1's first chord within effective_min_ticks.
        if (next_chord_lookahead is not None
                and last_tkey_in_measure is not None
                and tkey == last_tkey_in_measure):
            la_tkey = next_chord_lookahead.get("tkey")
            la_lanes = next_chord_lookahead.get("lanes") or set()
            if la_tkey is not None and la_lanes:
                gap = round(scaled_tick(la_tkey) - _st, 6)
                if gap < effective_min_ticks:
                    avail = [l for l in avail if l not in la_lanes]
        if not avail:
            residuals[(measure, idx192, token)] = "jack_violation"; continue
        T = left_count + right_count
        if T >= 10:
            ok_l = 0.30 <= (left_count + 1) / (T + 1) <= 0.70
            ok_r = 0.30 <= left_count / (T + 1) <= 0.70
            if not (ok_l and ok_r): hand_bal_ct += 1
            if ok_l and not ok_r:       avail = [l for l in avail if l in LEFT_KEY_LANES_SET]
            elif ok_r and not ok_l:     avail = [l for l in avail if l in RIGHT_KEY_LANES_SET]
            elif not ok_l and not ok_r: avail = []
            if not avail:
                residuals[(measure, idx192, token)] = "hand_balance"; continue
        if is_new and last_hand != "balanced" and streak >= max_same_hand:
            same_hand_ct += 1
            avail = [l for l in avail if HAND_MAP[l] == ("right" if last_hand == "left" else "left")]
            if not avail:
                residuals[(measure, idx192, token)] = "no_lane_available"; continue
        # §23.7 boundary lookahead (E-β): forward hand-streak constraint for
        # the last chord in the last measure. If placing this chord with
        # last_hand would let the lookahead extend our streak past max_same_hand,
        # prefer the opposite hand (soft — fall back to original avail if no
        # opposite-hand lane is available).
        if (is_new
                and next_chord_lookahead is not None
                and last_tkey_in_measure is not None
                and tkey == last_tkey_in_measure):
            la_hand = next_chord_lookahead.get("hand")
            if (la_hand and la_hand != "balanced"
                    and last_hand == la_hand
                    and streak + 1 >= max_same_hand):
                opposite = "right" if la_hand == "left" else "left"
                avail_op = [l for l in avail if HAND_MAP[l] == opposite]
                if avail_op:
                    avail = avail_op
        # 2026-05-03: chord-mate spread preference — prefer lanes ≥2 apart
        # from already-placed lanes at this tkey. Soft: falls back to full
        # avail if no spread-2 option available (e.g., 5-key chord forced
        # to pack 5 lanes within 7-key span).
        if not is_new and used_at_pos[idx192]:
            existing_idx = [LANE_INDEX[ex] - 1 for ex in used_at_pos[idx192]]
            avail_spread = [
                l for l in avail
                if all(abs((LANE_INDEX[l] - 1) - ei) >= 2 for ei in existing_idx)
            ]
            if avail_spread:
                avail = avail_spread
        lane = None
        if ml_ctx is not None:
            total_now = left_count + right_count
            lane = _ml_select_lane(
                ml_ctx, token, idx192, measure,
                ml_ctx["density_rank"], phase_label, total_now, avail, rng,
            )
        # v12 §9.8 token→lane affinity (RB lane path only): reuse the lane this
        # token was recently placed on. Runs after every hard gate, so the
        # remembered lane must already be in `avail`; otherwise fall through to
        # the centroid walk.
        if lane is None and token_lane_memory is not None:
            _mem = token_lane_memory.get(token)
            if _mem is not None:
                _mem_lane, _mem_measure = _mem
                if (measure - _mem_measure <= TOKEN_LANE_AFFINITY_WINDOW
                        and _mem_lane in avail
                        and rng.random() < TOKEN_LANE_AFFINITY_PROB):
                    lane = _mem_lane
                    affinity_ct += 1
        if lane is None:
            if centroid_state is not None and ta:
                lane, centroid_state["prev_lane_idx"], centroid_state["prev_centroid"] = \
                    _centroid_lane_select(token, avail, ta,
                                         centroid_state["prev_lane_idx"],
                                         centroid_state["prev_centroid"], rng,
                                         step_unit=centroid_state["step_unit"])
            else:
                lane = fisher_yates_shuffle(avail, rng)[0]
        if ml_ctx is not None and ml_ctx["state"]["enable_lane"]:
            atk = ml_ctx["ta_map"].get(token, {}) or {}
            ml_ctx["lane_context"].append({
                "tkey": tkey, "lane": lane, "idx192": idx192,
                "attack_rms": float(atk.get("attack_rms", 0.0)) if atk.get("decode_ok") else 0.0,
            })
        placed.append((idx192, token, lane))
        if token_lane_memory is not None:
            token_lane_memory[token] = (lane, measure)  # §9.8 affinity memory
        used_at_pos[idx192].add(lane); tokens_at_pos[idx192].add(token)
        if is_new: positions_with_notes.add(idx192)
        elif is_second: chord_positions.add(idx192)
        if is_new: last_hand, streak = _update_hand(last_hand, streak, lane)
        if lane in LEFT_KEY_LANES_SET: left_count += 1
        else: right_count += 1
        jack_state[lane] = tkey
        _chord_lanes.add(lane)

    # Commit the final chord's streak update
    _commit_chord_streak()

    eligible = sum(1 for (i, _, _, _) in curr_cands if i % 24 == 0) if is_chordburst else len(curr_cands)
    failed = len(placed) == 0 and eligible > 0
    return placed, (last_hand, streak), residuals, failed, {"hand_balance_applied": hand_bal_ct, "same_hand_applied": same_hand_ct, "lane_affinity_applied": affinity_ct}


# ── Scratch insertion ────────────────────────────────────────────────────────

def _insert_scratch_in_measure(scratch_candidates, placed_notes, measure,
                                jack_scr_tkey, pct_map, params=None,
                                max_per_m_override=None,
                                min_interval_override=None):
    """
    max_per_m_override: per-measure budget derived from source's scratch density.
        When non-None, used in place of SCRATCH_MAX_PER_MEASURE for this measure.
        Source-aware path (primary scratch mode) passes round(source_scratch_count[m] * scale).
    min_interval_override: when non-None, replaces SCRATCH_MIN_INTERVAL. Primary mode
        uses source's own minimum interval (with safety floor).

    2026-05-03: scratch no longer counts toward MAX_CHORD_SIZE — scratch is a
    wrist motion on a separate lane, doesn't compete with finger keys, and
    scratch:r authors regularly write 5+ event measures with chord+scratch.
    """
    p = params or _default_params()
    max_per_m    = max_per_m_override if max_per_m_override is not None \
                   else p.get("SCRATCH_MAX_PER_MEASURE", SCRATCH_MAX_PER_MEASURE)
    min_interval = min_interval_override if min_interval_override is not None \
                   else p.get("SCRATCH_MIN_INTERVAL", SCRATCH_MIN_INTERVAL)

    key_ia = defaultdict(float)
    for (idx192, token, lane) in placed_notes:
        key_ia[idx192] = max(key_ia[idx192], pct_map.get(token, 50.0))
    pos_best = {}
    for (idx192, token) in scratch_candidates:
        rep_ia = key_ia[idx192] if idx192 in key_ia else pct_map.get(token, 50.0)
        if idx192 not in pos_best or rep_ia > pos_best[idx192][1]:
            pos_best[idx192] = (token, rep_ia)
    # 2026-05-03: Primary path (source-aware override) walks positions in
    # source order (idx ascending) so the interval check is forward-only and
    # cannot reject earlier positions just because intensity put them last.
    # Fallback path keeps intensity-first greedy order.
    if max_per_m_override is not None:
        sorted_positions = sorted(pos_best.items(), key=lambda x: x[0])
    else:
        sorted_positions = sorted(pos_best.items(), key=lambda x: (-x[1][1], x[0]))

    scratch_placed = []
    scr_residuals = {"scratch_interval": 0, "scratch_density_cap": 0}
    last_tkey, scratch_count = jack_scr_tkey, 0
    for i, (idx192, (token, _)) in enumerate(sorted_positions):
        if scratch_count >= max_per_m:
            scr_residuals["scratch_density_cap"] += len(sorted_positions) - i; break
        tkey = measure * 192 + idx192
        if tkey - last_tkey < min_interval:
            scr_residuals["scratch_interval"] += 1; continue
        scratch_placed.append((idx192, token))
        last_tkey = tkey; scratch_count += 1
    return scratch_placed, scr_residuals, last_tkey


def _insert_scratch_supplement(tier2_cands, supplement_m, measure,
                                existing_scr_tkeys, placed_key_set,
                                tier1_placed, min_interval):
    """§12 tier-2 supplement insertion (REVISED 2026-06-13).

    tier2_cands: [(rank_tuple, idx192, token), ...] pre-sorted — 2a (source
        wheel timbres at new onsets) before 2b (functional-gate tokens by
        weak audio prior).
    existing_scr_tkeys: absolute tkeys of scratches already placed (this
        measure's tier 1 + the previous measure's last via jack_scr_tkey).
    Unlike tier 1's forward-only walk, tier-2 candidates arrive in preference
    order, so the interval check is two-sided (gap to both neighbors).
    """
    import bisect as _bi
    tks = sorted(t for t in existing_scr_tkeys if t >= 0)
    placed2 = []
    tier1_set = set(tier1_placed)
    for (_, idx192, token) in tier2_cands:
        if len(placed2) >= supplement_m:
            break
        if (idx192, token) in placed_key_set or (idx192, token) in tier1_set:
            continue
        tkey = measure * 192 + idx192
        pos = _bi.bisect_left(tks, tkey)
        if pos > 0 and tkey - tks[pos - 1] < min_interval:
            continue
        if pos < len(tks) and tks[pos] - tkey < min_interval:
            continue
        _bi.insort(tks, tkey)
        placed2.append((idx192, token))
    return placed2


# ── LN post-processing ───────────────────────────────────────────────────────

def compute_end_tkey(start_tkey, duration_ms, bpm_events, base_bpm,
                     measure_scale=None):
    current_bpm = base_bpm
    for (tk, bpm) in bpm_events:
        if tk <= start_tkey: current_bpm = bpm
        else: break

    remaining_ms = duration_ms
    pos = start_tkey
    future = [(tk, bpm) for (tk, bpm) in bpm_events if tk > start_tkey]

    sc = _normalize_measure_scale(measure_scale)
    if not sc:
        # Legacy fast path — byte-identical to the pre-scale-aware walk.
        for (next_tk, next_bpm) in future:
            ticks_seg = next_tk - pos
            ms_seg = ticks_seg * 1250.0 / current_bpm
            if ms_seg >= remaining_ms: break
            remaining_ms -= ms_seg
            pos = next_tk; current_bpm = next_bpm
        return pos + round(remaining_ms * current_bpm / 1250.0)

    # 2026-06-11 scale-aware walk: in a measure with #xxx02 scale s, one raw
    # tick spans 1250*s/bpm ms. Segment boundaries are BPM changes plus every
    # measure start where the scale value changes, so the scale is constant
    # within each walked segment.
    boundaries = list(future)  # (tkey, bpm)
    m0 = start_tkey // 192
    for m in range(m0 + 1, max(sc) + 2):
        if sc.get(m, 1.0) != sc.get(m - 1, 1.0):
            boundaries.append((m * 192, None))  # scale boundary, no BPM change
    boundaries.sort(key=lambda x: (x[0], x[1] is None))

    def _scale_at(tkey):
        return sc.get(tkey // 192, 1.0)

    for (next_tk, maybe_bpm) in boundaries:
        if next_tk <= pos:
            if maybe_bpm is not None:
                current_bpm = maybe_bpm
            continue
        ticks_seg = next_tk - pos
        ms_seg = ticks_seg * 1250.0 * _scale_at(pos) / current_bpm
        if ms_seg >= remaining_ms: break
        remaining_ms -= ms_seg
        pos = next_tk
        if maybe_bpm is not None:
            current_bpm = maybe_bpm
    return pos + round(remaining_ms * current_bpm / (1250.0 * _scale_at(pos)))


def select_lnobj_token(headers, declared_wav_tokens):
    existing = headers.get("LNOBJ", "").strip()
    if existing: return existing.upper()
    chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for c1 in chars:
        for c2 in chars:
            tok = c1 + c2
            if tok != "00" and tok not in declared_wav_tokens:
                return tok
    return None


def compute_placement_distribution_metrics(placed_events):
    """
    Post-placement distribution metrics for stability/dispersion analysis.
    Key-lane only (P1_SCR excluded). Used to compare placement smoothness
    between RB and ML or between source variants.

    Returns dict with:
      - lane_counts:                 [c1..c7] per-lane usage over the chart
      - per_measure_lr:              [(m, left, right, total), ...] sorted by m
      - right_share_mean / std / n:  L/R balance summary across measures with ≥4 notes
      - hand_jump_distribution:      Counter of |Δlane| between consecutive non-simultaneous notes
      - hand_jump_mean
      - same_hand_streak_distribution: Counter of consecutive same-hand run lengths
      - same_hand_streak_mean
      - lane_transition_matrix:      7x7 P(next | current), counts (not normalized)
    """
    LANE_TO_IDX = {
        "P1_KEY1": 1, "P1_KEY2": 2, "P1_KEY3": 3, "P1_KEY4": 4,
        "P1_KEY5": 5, "P1_KEY6": 6, "P1_KEY7": 7,
    }
    LEFT_IDX = {1, 2, 3}

    key_evs = []
    for ev in placed_events:
        if ev.get("type") == "LN":
            m, i = ev["measure_start"], ev["idx192_start"]
        else:
            m, i = ev.get("measure"), ev.get("idx192")
        lane_idx = LANE_TO_IDX.get(ev.get("lane"))
        if lane_idx is None:
            continue
        key_evs.append((m, i, lane_idx))
    key_evs.sort(key=lambda x: (x[0] * 192 + x[1], x[2]))

    # Per-measure L/R counts
    per_measure = defaultdict(lambda: [0, 0])  # [left, right]
    for (m, _, lane_idx) in key_evs:
        if lane_idx in LEFT_IDX:
            per_measure[m][0] += 1
        else:
            per_measure[m][1] += 1
    per_measure_lr = [(m, l, r, l + r) for m, (l, r) in sorted(per_measure.items())]

    # Right-share variance across measures with ≥4 notes
    qual = [(l, r) for (_, l, r, t) in per_measure_lr if t >= 4]
    right_shares = [r / (l + r) for (l, r) in qual]
    if right_shares:
        n = len(right_shares)
        m_share = sum(right_shares) / n
        var_share = sum((s - m_share) ** 2 for s in right_shares) / n
        std_share = var_share ** 0.5
    else:
        m_share = std_share = 0.0
        n = 0

    # Hand-jump magnitude (measured between consecutive distinct positions,
    # comparing first-placed lane of each chord — limited interpretive value
    # for chord-heavy charts; preferred companion: per-chord hand composition).
    jumps = []
    last_lane = None
    last_pos = None
    for (m, i, lane_idx) in key_evs:
        pos = m * 192 + i
        if last_pos is not None and pos != last_pos:
            jumps.append(abs(lane_idx - last_lane))
        last_lane = lane_idx
        last_pos = pos

    # Streak length: collapse each chord position to its hand composition,
    # then compute three flavors so chord-heavy charts don't artifact:
    #   - hand_only_streak:   consecutive chord positions where ALL lanes are
    #                         the same hand (L-only or R-only). MIXED breaks
    #                         the streak.
    #   - hand_majority_streak: streak by majority hand (>=50% lanes); MIXED
    #                         50/50 breaks the streak.
    #   - L_fraction_per_chord: histogram of (#L / #total) per chord
    #                         position, exposes lopsided-chord patterns
    #                         without forcing a single-hand label.
    # Chord positions are produced by grouping key_evs by (m, idx).
    chord_lanes = {}
    for (m, i, lane_idx) in key_evs:
        chord_lanes.setdefault((m, i), set()).add(lane_idx)

    def _chord_hand_label(lanes):
        l_n = sum(1 for x in lanes if x in LEFT_IDX)
        r_n = len(lanes) - l_n
        if r_n == 0: return "L"
        if l_n == 0: return "R"
        if l_n > r_n: return "Lmaj"
        if r_n > l_n: return "Rmaj"
        return "EQ"

    sorted_chords = sorted(chord_lanes.items())
    only_streaks = []
    maj_streaks = []
    cur_only = None; cur_only_len = 0
    cur_maj = None;  cur_maj_len = 0
    L_fractions = []
    chord_hand_counts = {"L": 0, "R": 0, "Lmaj": 0, "Rmaj": 0, "EQ": 0}
    for (_, lanes) in sorted_chords:
        l_n = sum(1 for x in lanes if x in LEFT_IDX)
        total_n = len(lanes)
        L_fractions.append(round(l_n / total_n, 2))
        label = _chord_hand_label(lanes)
        chord_hand_counts[label] += 1

        # only-streak: track strict L-only / R-only runs; MIXED labels break.
        only_label = label if label in ("L", "R") else None
        if only_label and only_label == cur_only:
            cur_only_len += 1
        else:
            if cur_only is not None and cur_only_len > 0:
                only_streaks.append(cur_only_len)
            cur_only = only_label
            cur_only_len = 1 if only_label else 0

        # majority-streak: any majority hand counts; EQ breaks the streak.
        maj_label = ("L" if label in ("L", "Lmaj")
                     else "R" if label in ("R", "Rmaj")
                     else None)
        if maj_label and maj_label == cur_maj:
            cur_maj_len += 1
        else:
            if cur_maj is not None and cur_maj_len > 0:
                maj_streaks.append(cur_maj_len)
            cur_maj = maj_label
            cur_maj_len = 1 if maj_label else 0
    if cur_only is not None and cur_only_len > 0:
        only_streaks.append(cur_only_len)
    if cur_maj is not None and cur_maj_len > 0:
        maj_streaks.append(cur_maj_len)

    # Legacy "streaks" computation (kept for backward log compatibility) —
    # collapses chord to first-placed lane's hand. Known artifact on
    # chord-heavy charts; prefer hand_only / hand_majority.
    streaks = []
    cur_streak_hand = None
    cur_streak_len = 0
    last_pos = None
    for (m, i, lane_idx) in key_evs:
        pos = m * 192 + i
        cur_hand = "L" if lane_idx in LEFT_IDX else "R"
        if last_pos is not None and pos != last_pos:
            if cur_hand == cur_streak_hand:
                cur_streak_len += 1
            else:
                if cur_streak_hand is not None:
                    streaks.append(cur_streak_len)
                cur_streak_hand = cur_hand
                cur_streak_len = 1
        elif last_pos is None:
            cur_streak_hand = cur_hand
            cur_streak_len = 1
        last_pos = pos
    if cur_streak_hand is not None and cur_streak_len > 0:
        streaks.append(cur_streak_len)

    from collections import Counter as _C
    jump_dist = _C(jumps)
    streak_dist = _C(streaks)
    only_streak_dist = _C(only_streaks)
    maj_streak_dist = _C(maj_streaks)
    L_frac_dist = _C(L_fractions)

    # Lane transition matrix 7x7
    trans = [[0] * 7 for _ in range(7)]
    last_lane = None
    last_pos = None
    for (m, i, lane_idx) in key_evs:
        pos = m * 192 + i
        if last_pos is not None and pos != last_pos and last_lane is not None:
            trans[last_lane - 1][lane_idx - 1] += 1
        last_lane = lane_idx
        last_pos = pos

    # Per-lane counts
    lane_counts = [0] * 7
    for (_, _, lane_idx) in key_evs:
        lane_counts[lane_idx - 1] += 1

    return {
        "lane_counts": lane_counts,
        "per_measure_lr": [{"m": m, "left": l, "right": r, "total": t}
                           for (m, l, r, t) in per_measure_lr],
        "right_share_mean": round(m_share, 4),
        "right_share_std": round(std_share, 4),
        "right_share_n_measures": n,
        "hand_jump_distribution": {str(k): v for k, v in sorted(jump_dist.items())},
        "hand_jump_mean": round(sum(jumps) / max(1, len(jumps)), 3) if jumps else 0.0,
        "same_hand_streak_distribution": {str(k): v for k, v in sorted(streak_dist.items())},
        "same_hand_streak_mean": round(sum(streaks) / max(1, len(streaks)), 3) if streaks else 0.0,
        "hand_only_streak_distribution": {str(k): v for k, v in sorted(only_streak_dist.items())},
        "hand_only_streak_mean": round(sum(only_streaks) / max(1, len(only_streaks)), 3) if only_streaks else 0.0,
        "hand_majority_streak_distribution": {str(k): v for k, v in sorted(maj_streak_dist.items())},
        "hand_majority_streak_mean": round(sum(maj_streaks) / max(1, len(maj_streaks)), 3) if maj_streaks else 0.0,
        "L_fraction_per_chord_distribution": {str(k): v for k, v in sorted(L_frac_dist.items())},
        "chord_hand_counts": chord_hand_counts,
        "lane_transition_matrix": trans,
    }


def run_ln_postprocess(placed_events, ta, bpm_events, measure_scale,
                       base_bpm, ln_min_duration_ms, ln_max_ratio,
                       ln_max_hold_ticks=LN_MAX_HOLD_TICKS):
    total_placed = len(placed_events)
    ln_candidates = [
        ev for ev in placed_events
        if ev.get("type", "Tap") == "Tap"
        and ev.get("lane") != "P1_SCR"
        and ta.get(ev["token"], {}).get("duration_ms", 0) >= ln_min_duration_ms
        and ta.get(ev["token"], {}).get("decode_ok", False)
    ]
    ln_candidates.sort(key=lambda ev: -ta[ev["token"]]["duration_ms"])

    # lane_blocked: all tkeys with a token on each lane (Tap + LN end markers)
    lane_blocked = defaultdict(set)
    for ev in placed_events:
        if ev.get("type", "Tap") == "Tap":
            lane_blocked[ev["lane"]].add(ev["measure"] * 192 + ev["idx192"])

    lane_intervals = defaultdict(list)
    promoted = set()
    ln_promoted_count = 0
    skip_interior = skip_end = skip_ratio = 0

    for ev in ln_candidates:
        if ln_promoted_count > 0 and (ln_promoted_count + 1) / total_placed > ln_max_ratio:
            skip_ratio += len(ln_candidates) - (ln_promoted_count + skip_interior + skip_end + skip_ratio)
            break

        start_tkey = ev["measure"] * 192 + ev["idx192"]
        dur_ms = ta[ev["token"]]["duration_ms"]
        end_tkey = compute_end_tkey(start_tkey, dur_ms, bpm_events, base_bpm,
                                    measure_scale)
        # §13.0 hold cap is 2 beats; expressed in beat-true ticks so a #xxx02
        # scaled measure keeps the "2 beats visible" semantics.
        end_tkey = min(end_tkey,
                       advance_scaled_ticks(start_tkey, ln_max_hold_ticks,
                                            measure_scale))
        lane = ev["lane"]

        if end_tkey <= start_tkey:
            continue

        # Collision: check if end_tkey is blocked (Tap or existing LN end marker)
        if end_tkey in lane_blocked[lane]:
            skip_end += 1; continue

        # Interior collision: any blocked tkey strictly between start and end
        collision = False
        for tk in lane_blocked[lane]:
            if tk == start_tkey: continue
            if start_tkey < tk < end_tkey:
                collision = True; break
        # Also check existing LN interval overlaps
        if not collision:
            for (ls, le) in lane_intervals[lane]:
                if ls < end_tkey and start_tkey < le:
                    collision = True; break
        if collision:
            skip_interior += 1; continue

        # Ratio check
        if (ln_promoted_count + 1) / total_placed > ln_max_ratio:
            skip_ratio += 1; continue

        # Promote: mark end position as blocked for future candidates
        promoted.add(id(ev))
        lane_blocked[lane].add(end_tkey)
        lane_intervals[lane].append((start_tkey, end_tkey))
        ln_promoted_count += 1

    # Build new placed_events
    new_events = []
    for ev in placed_events:
        if id(ev) in promoted:
            start_tkey = ev["measure"] * 192 + ev["idx192"]
            dur_ms = ta[ev["token"]]["duration_ms"]
            end_tkey = compute_end_tkey(start_tkey, dur_ms, bpm_events, base_bpm,
                                        measure_scale)
            # Apply same cap as the collision check above so the stored end
            # matches the position that was actually validated as collision-free.
            end_tkey = min(end_tkey,
                           advance_scaled_ticks(start_tkey, ln_max_hold_ticks,
                                                measure_scale))
            new_events.append({
                "type": "LN",
                "token": ev["token"],
                "lane": ev["lane"],
                "measure_start": ev["measure"],
                "idx192_start": ev["idx192"],
                "measure_end": end_tkey // 192,
                "idx192_end": end_tkey % 192,
                "primitive": ev["primitive"],
                "phase": ev["phase"],
            })
        else:
            new_events.append(ev)

    return {
        "placed_events": new_events,
        "ln_meta": {
            "enabled": True,
            "ln_candidates_found": len(ln_candidates),
            "ln_promoted_count": ln_promoted_count,
            "ln_skipped_interior_collision": skip_interior,
            "ln_skipped_end_collision": skip_end,
            "ln_skipped_ratio_cap": skip_ratio,
            "ln_max_hold_ticks": ln_max_hold_ticks,
            "lnobj_token": None,
        }
    }


# ── Per-measure decision loop ─────────────────────────────────────────────────

def _compute_source_scratch_per_measure(events):
    """Source per-measure scratch counts (channel 16 only)."""
    out = defaultdict(int)
    for ev in events:
        et = ev.get("type")
        if et == "Tap" and ev.get("rawChannel") == "16":
            out[ev["measure"]] += 1
        elif et == "Long" and ev.get("rawChannelStart") == "16":
            out[ev["measureStart"]] += 1
    return dict(out)


def _compute_source_min_scratch_interval(events, floor=4):
    """Smallest non-zero gap between scratches in source. Returns None when <2 events."""
    tkeys = []
    for ev in events:
        et = ev.get("type")
        if et == "Tap" and ev.get("rawChannel") == "16":
            tkeys.append(ev["measure"] * 192 + ev["idx192"])
        elif et == "Long" and ev.get("rawChannelStart") == "16":
            tkeys.append(ev["measureStart"] * 192 + ev["idx192Start"])
    if len(tkeys) < 2: return None
    tkeys.sort()
    gaps = [tkeys[i + 1] - tkeys[i] for i in range(len(tkeys) - 1)]
    gaps = [g for g in gaps if g > 0]
    if not gaps: return None
    return max(floor, min(gaps))


# ── DP template layer: separation axis (SPLIT) ─────────────────
DP_SPLIT_FEATURE = "low_freq_energy_ratio"  # S1, η² p50 0.913 (DR-DP4)
DP_MIN_SIDE_SHARE = 0.25   # each hand gets ≥ this fraction of a block's note-weight
                            # (loosened from 0.40 so the spectral split dominates →
                            #  sharper bass/melodic separation; swap-on-repeat balances)
DP_REPEAT_SIMILARITY = 0.55  # token-set Jaccard ≥ this → phase blocks are "the same phrase"
DP_SCRATCH_KEY_GUARD_TICKS = 24  # scratch-hand keys within this of a scratch are gated (8th note)
# DP played-content rescue: the SP whitelist (band quota + fx-duration)
# drops much of what the source actually PLAYED on key channels — invisible in SP
# (→ BGM) but it strips the DP chart of its lead/bass (e.g. hardtek long-synth
# leads: 79% of key onsets cut by fx-duration). In DP, restore the SOURCE KEY
# onsets of those excluded tokens — at THOSE positions only, NOT the token's BGM
# occurrences (Codex 2026-06-14: token-level whitelist would flood BGM; position-
# targeting avoids it). Gate only on attack > fx-soft line so genuine soft pads
# stay excluded; the source having played it as a key is the keep-signal. DP-only,
# SP path untouched.
DP_RESCUE_MIN_ATTACK = FX_ATTACK_THRESHOLD   # rescue non-soft (>20) played onsets; pads stay cut
# DP multi-strategy split router. timbre (low_freq) suits STREAM songs;
# balance (split chords / alternate bursts) suits CHORD/PEAK songs. Strategy is
# user-selected (`--dp-split`); the dominant NoteAttribute per song is read with
# tools/dp_source_character.py. NoteAttributes corpus: stream 4 / peak 4 / chord 2.


def _dp_block_token_sides(block_weight, ta, swap=False):
    """Assign one phrase block's candidate tokens to two hands by timbral role.

    One hand is the **bass hand** (mid-low notes), the other the **melodic hand**
    — addon §3 separation axis = low_freq_energy_ratio. Tokens are ordered on that
    axis (event-weighted), split at the point that maximises spectral separation
    *while keeping both hands within the balance band* (DP_MIN_SIDE_SHARE) so
    neither hand goes idle on a mono-timbral passage; if no split is in band,
    fall back to the most weight-balanced cut. The higher-low_freq (bassy)
    cluster goes to the bass hand, the trebly cluster to the melodic hand.

    swap: when True, the two clusters bind to the opposite physical hands. The
    router sets this on repeated phrases (mirror-on-repeat) so a looped phrase
    doesn't always land on the same hand — variety + reduced fatigue. Returns
    {token: "L"|"R"}.
    """
    bass_side = "R" if swap else "L"
    mel_side  = "L" if swap else "R"
    feats = []
    for t in block_weight:
        info = ta.get(t) if ta else None
        if info and info.get("decode_ok"):
            v = info.get(DP_SPLIT_FEATURE)
            if v is not None:
                feats.append((float(v), t))
    sides = {}
    if len(feats) < 2:
        for t in block_weight:
            sides[t] = bass_side
        return sides
    feats.sort(key=lambda x: (x[0], x[1]))
    vals = [v for (v, _) in feats]
    wts = [block_weight[t] for (_, t) in feats]
    n = len(vals)
    total_w = sum(wts)
    pre_v = [0.0]
    for v in vals:
        pre_v.append(pre_v[-1] + v)
    pre_w = [0.0]
    for w in wts:
        pre_w.append(pre_w[-1] + w)
    mean = pre_v[n] / n
    band_lo = DP_MIN_SIDE_SHARE * total_w
    best_i, best_between = None, -1.0
    bal_i, bal_gap = 1, None  # most weight-balanced fallback
    for i in range(1, n):
        left_w, right_w = pre_w[i], total_w - pre_w[i]
        gap = abs(left_w - right_w)
        if bal_gap is None or gap < bal_gap:
            bal_gap, bal_i = gap, i
        if left_w < band_lo or right_w < band_lo:
            continue  # outside balance band — not a candidate split
        ml = pre_v[i] / i
        mr = (pre_v[n] - pre_v[i]) / (n - i)
        between = i * (ml - mean) ** 2 + (n - i) * (mr - mean) ** 2
        if between > best_between:
            best_between, best_i = between, i
    split_i = best_i if best_i is not None else bal_i
    melodic = [t for (_, t) in feats[:split_i]]   # lower low_freq → melodic hand
    bass    = [t for (_, t) in feats[split_i:]]   # higher low_freq → bass hand
    load = {"L": 0.0, "R": 0.0}
    for t in bass:
        sides[t] = bass_side; load[bass_side] += block_weight[t]
    for t in melodic:
        sides[t] = mel_side;  load[mel_side]  += block_weight[t]
    for t in block_weight:           # tokens with no decodable feature → lighter hand
        if t not in sides:
            light = "L" if load["L"] <= load["R"] else "R"
            sides[t] = light; load[light] += block_weight[t]
    return sides


def _dp_build_side_maps(cands_by_m, phase_blocks, ta):
    """Template router: per phase block (variable-length granularity, #2), derive
    a bass-hand/melodic-hand token→side map and project it onto every measure.

    Mirror-on-repeat: phase blocks are grouped into phrase classes by token-set
    similarity (Jaccard ≥ DP_REPEAT_SIMILARITY — fuzzy, because real loops vary
    notes/positions slightly between iterations and exact-match almost never
    fires). Within a class the hands toggle across occurrences (1st A, 2nd B, 3rd
    A, …) so a recurring phrase doesn't always sit on the same hand — variety +
    less fatigue, and the bass/melodic role still reads. Returns (dp_side_by_m
    {measure: {token: "L"|"R"}}, n_swapped).
    """
    dp_side_by_m = {}
    classes = []  # [token_set, occurrence_count]
    n_swapped = 0
    max_m = max(cands_by_m.keys(), default=-1)
    blocks = phase_blocks if phase_blocks else [{"start": 0, "end": max_m + 1}]
    for b in blocks:
        block_weight = defaultdict(int)
        for m in range(b["start"], b["end"]):
            for (idx, tok, pct, io) in cands_by_m.get(m, []):
                block_weight[tok] += 1
        if not block_weight:
            continue
        fp = set(block_weight)
        matched = None
        for cls in classes:
            inter = len(fp & cls[0])
            union = len(fp | cls[0])
            if union and inter / union >= DP_REPEAT_SIMILARITY:
                matched = cls
                break
        if matched is None:
            classes.append([fp, 1])
            swap = False
        else:
            swap = (matched[1] % 2 == 1)   # prior count odd → mirror this occurrence
            matched[1] += 1
        if swap:
            n_swapped += 1
        sides = _dp_block_token_sides(block_weight, ta, swap=swap)
        for m in range(b["start"], b["end"]):
            mm = {}
            for (idx, tok, pct, io) in cands_by_m.get(m, []):
                mm[tok] = sides.get(tok, "L")
            if mm:
                dp_side_by_m[m] = mm
    return dp_side_by_m, n_swapped


def _dp_measure_balanced_split(curr_cands, ta):
    """Per-measure fallback split (user 2026-06-13): when the block token→side
    map leaves a measure one-sided (one hand empty — common in mono-timbral
    sections like a bass-only breakdown, where the block lumps similar onsets on
    one hand), re-split THIS measure's tokens 50/50 by note-weight along the
    low_freq axis so neither hand is empty. Lower low_freq → R (melodic side),
    higher → L (bass side), consistent with the main split. Returns {token: side}.
    """
    w = defaultdict(int)
    for (idx, tok, ia, io) in curr_cands:
        w[tok] += 1
    def _lf(t):
        v = (ta.get(t) or {}).get("low_freq_energy_ratio") if ta else None
        return float(v) if v is not None else 0.0
    toks = sorted(w, key=lambda t: (_lf(t), t))
    half = sum(w.values()) / 2.0
    sides, cum = {}, 0
    for t in toks:
        sides[t] = "R" if cum < half else "L"
        cum += w[t]
    return sides


def _dp_balance_sides(curr_cands, ta):
    """Load-balance per-onset hand assignment (addon §9, CHORD/PEAK character).

    Instead of grouping by timbre (which piles a same-timbre burst onto one hand
    → jack drops), distribute physical load: chord-mates at a tkey are split
    across hands (by centroid, alternating), and sequential single notes go to
    the lighter hand (→ burst alternation). Returns a side list parallel to
    curr_cands.
    """
    def cen(t):
        info = ta.get(t) if ta else None
        return float((info or {}).get("spectral_centroid_mean", 0.0) or 0.0)
    sides = [None] * len(curr_cands)
    lcnt = rcnt = 0
    i = 0
    while i < len(curr_cands):
        j = i
        while j < len(curr_cands) and curr_cands[j][0] == curr_cands[i][0]:
            j += 1
        if j - i == 1:
            s = "L" if lcnt <= rcnt else "R"
            sides[i] = s
            if s == "L": lcnt += 1
            else: rcnt += 1
        else:
            order = sorted(range(i, j), key=lambda k: (cen(curr_cands[k][1]), k))
            cur = "L" if lcnt <= rcnt else "R"
            for k in order:
                sides[k] = cur
                if cur == "L": lcnt += 1
                else: rcnt += 1
                cur = "R" if cur == "L" else "L"
        i = j
    return sides


def _place_measure_dp(curr_cands, cand_sides, rng, state_L, state_R,
                      measure, params, ta, bpm_lookup, scaled_tick):
    """Side-local DP placement (addon §2). cand_sides is a per-onset "L"/"R" list
    parallel to curr_cands (built by the active split strategy — timbre or
    balance). Run the SP placement machine independently per hand (own jack/hand/
    centroid/streak/affinity state), remap the right side's lanes P1_*→P2_*. One
    hand is capped at DP_MAX_CHORD_SIZE_PER_SIDE; both summed at 2× that
    (combined) via external_tkey_count.

    Returns (placed, residuals, diag).
    """
    def _state(side):
        return (state_L, False) if side == "L" else (state_R, True)

    per_side_cap = params.get("DP_MAX_CHORD_SIZE_PER_SIDE", 2)
    # combined (both-hand) cap = DP's two-hand capacity = 2 × per-side (NOT the SP
    # MAX_CHORD_SIZE). Attribute-independent — only binds on dense chords.
    combined_cap = 2 * per_side_cap
    p_side = dict(params); p_side["MAX_CHORD_SIZE"] = per_side_cap
    placed_all, res_all = [], {}
    diag_all = {"hand_balance_applied": 0, "same_hand_applied": 0,
                "lane_affinity_applied": 0}

    def _run(cands, st, remap, ext):
        placed, hand_state, m_res, _f, m_diag = _place_measure_constrained(
            cands, rng, st["hand"], st["jack"], measure, False, p_side,
            ml_ctx=None, phase_label="normal", ta=ta,
            centroid_state=st["centroid"], bpm_lookup=bpm_lookup,
            jack_streak=st["jack_streak"], next_chord_lookahead=None,
            scaled_tick=scaled_tick, token_lane_memory=st["lane_mem"],
            external_tkey_count=ext, combined_chord_cap=combined_cap)
        st["hand"] = hand_state
        out = [(i, t, P1_TO_P2_LANE[l] if remap else l) for (i, t, l) in placed]
        for k in diag_all:
            diag_all[k] += m_diag.get(k, 0)
        return out, m_res

    cands_L = [c for c, s in zip(curr_cands, cand_sides) if s == "L"]
    cands_R = [c for c, s in zip(curr_cands, cand_sides) if s == "R"]
    order = (("L", cands_L), ("R", cands_R))
    if len(cands_R) < len(cands_L):
        order = (("R", cands_R), ("L", cands_L))
    ext = defaultdict(int)
    for side, cands in order:
        if not cands:
            continue
        st, remap = _state(side)
        out, m_res = _run(cands, st, remap, dict(ext))
        for (i, _t, _l) in out:
            ext[measure * 192 + i] += 1
        placed_all.extend(out)
        res_all.update(m_res)
    return placed_all, res_all, diag_all


def _dp_scratch_measure(measure, placed, scratch_candidates, budget_m, min_int,
                        scratch_hand):
    """DP scratch placement (addon §4, measure-unit revision 2026-06-13).

    User feedback: per-onset L/R scratch alternation made the scratch hand flip
    rapidly within a measure → fatiguing. Instead the **whole measure's scratch
    goes to one hand** (scratch_hand, chosen by the caller and swapped measure to
    measure). The scratch hand is stable within the bar; the other hand owns the
    keys.

    Anti-jump gate (user 2026-06-13): the scratch→key→scratch jumping on the
    turntable hand is uncomfortable only when a key sits CLOSE to a scratch on
    that hand. So scratch-hand keys within DP_SCRATCH_KEY_GUARD_TICKS of any
    scratch onset are gated out (→ residual, audio survives as BGM, timing kept);
    scratch-hand keys far from scratches stay. A single scratch only clears a
    small neighbourhood, not the whole measure; a dense scratch run clears most
    of the scratch hand naturally (overlapping windows). Opposite-hand keys are
    untouched.

    placed: list[(idx192, token, lane)], MUTATED in place to drop gated keys.
    Returns (scr_placed [(idx192, token, scratch_hand)],
             gated [(idx192, token, lane)]).
    """
    import bisect as _bi

    def _side_of(lane):
        return "R" if lane.startswith("P2") else "L"

    scr_placed, gated = [], []
    n_budget = budget_m if budget_m is not None else len(scratch_candidates)
    last_tkey = -999
    scr_idxs = []
    for (idx, token) in scratch_candidates:
        if len(scr_placed) >= n_budget:
            break
        tkey = measure * 192 + idx
        if min_int is not None and tkey - last_tkey < min_int:
            continue  # respect source spacing on the (single) scratch hand
        scr_placed.append((idx, token, scratch_hand))
        last_tkey = tkey
        scr_idxs.append(idx)
    # anti-jump gate: drop scratch-hand keys within the guard window of a scratch.
    if scr_idxs:
        scr_idxs.sort()
        kept = []
        for (i, t, l) in placed:
            if _side_of(l) == scratch_hand:
                pos = _bi.bisect_left(scr_idxs, i)
                near = min((abs(i - scr_idxs[j]) for j in (pos - 1, pos)
                            if 0 <= j < len(scr_idxs)), default=10 ** 9)
                if near <= DP_SCRATCH_KEY_GUARD_TICKS:
                    gated.append((i, t, l))
                    continue
            kept.append((i, t, l))
        placed[:] = kept
    return scr_placed, gated


def run_per_measure_loop(events, whitelist, pct_map, intensity_origin_map,
                         phase_blocks, measure_max, scratch_seeds, params=None,
                         ml_ctx=None, excluded=None, ta=None,
                         key_occ=None, scratch_occ=None, bgm_occ=None,
                         seed=PLACEMENT_RANDOM_SEED,
                         bpm_events=None, base_bpm=130.0,
                         scratch_mode="primary",
                         source_scratch_per_measure=None,
                         scratch_scale=1.0,
                         source_min_interval=None,
                         resume_state=None,
                         start_measure=None,
                         end_measure=None,
                         next_chord_lookahead=None,
                         measure_scale=None,
                         dp=False, dp_split="auto"):
    """
    dp: when True, synthesize a DP (14-key + 2 scratch) chart from the SP pool
        via the DP template layer — per phase block SPLIT on the
        separation axis, then side-local placement on P1/P2. RB-only, full-chart
        only (resume/ML rejected). Scratch is disabled in this step (addon §4
        DP scratch gate is a later step). SP path (dp=False) is byte-identical.
    scratch_mode: "primary" (source has scratch tokens, mirror per-measure),
                  "fallback" (synthesized from key tokens, level-based caps + RUSH),
                  "disabled" (no scratch insertion).
    source_scratch_per_measure: dict[int, int] — source's scratch event count per measure.
                  Only used in primary mode for per-measure budget derivation.
    scratch_scale: multiplier on source per-measure count (level/5 by default).
                  At scale=1.0 the output mirrors source 1:1.
    """
    p = params or _default_params()
    rush_threshold   = p.get("SCRATCH_RUSH_THRESHOLD", SCRATCH_RUSH_THRESHOLD)
    rush_rest_measures = p.get("SCRATCH_RUSH_REST_MEASURES", SCRATCH_RUSH_REST_MEASURES)

    # v12 §23.4 (DR-23-5): per-measure RNG (β-1). RNG instance is created fresh
    # inside the measure loop; here we only normalize the chart-level seed.
    # G.2: explicit type check — silent fallback to 42 on str/float caused
    # debugging confusion. Caller must supply a proper int (CLI 'random' is
    # resolved to an int upstream in main()).
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError(
            f"placement seed must be int (got {type(seed).__name__}: {seed!r}); "
            f"if invoking main() directly, resolve 'random' to int.from_bytes(os.urandom(4), 'big') first.")
    seed_int = seed

    # DP: synthesis is RB-only and full-chart in this step.
    if dp:
        if resume_state is not None or start_measure is not None or end_measure is not None:
            raise NotImplementedError(
                "DP synthesis does not support resume mode yet (side-paired "
                "carry-over state → resume-v3 scope, addon §5).")
        if ml_ctx is not None and (
                ml_ctx["state"].get("enable_token") or ml_ctx["state"].get("enable_lane")):
            raise NotImplementedError("DP synthesis is RB-only (addon §4); --ml not supported.")
        if next_chord_lookahead is not None:
            raise NotImplementedError("DP synthesis does not support boundary lookahead.")

    seen, cands_by_m = set(), defaultdict(list)
    excluded_by_m = defaultdict(list)
    for ev in events:
        etype = ev.get("type")
        if etype == "Tap":
            ch = ev.get("rawChannel", "")
            token, measure, idx192 = ev["token"], ev["measure"], ev["idx192"]
            # 2026-05-03: ch16 (scratch) events are reserved for scratch lane and
            # must not enter cands_by_m, otherwise key placement consumes them
            # before _insert_scratch_in_measure can claim the natural scratch slot.
            if ch == "16": continue
        elif etype == "Long":
            ch = ev.get("rawChannelStart", "")
            token, measure, idx192 = ev["tokenStart"], ev["measureStart"], ev["idx192Start"]
            if ch == "16": continue
        elif etype == "BGM":   token, measure, idx192 = ev["token"], ev["measure"], ev["idx192"]
        else: continue
        key = (measure, idx192, token)
        if key in seen: continue
        seen.add(key)
        if token in whitelist:
            cands_by_m[measure].append((idx192, token, pct_map.get(token, 50.0),
                                        intensity_origin_map.get(token, 0)))
        elif excluded and token in excluded:
            # v10: "unknown" = decode-failed → cannot play, never rescue
            if excluded[token] == "unknown":
                continue
            excluded_by_m[measure].append((idx192, token, pct_map.get(token, 50.0),
                                           intensity_origin_map.get(token, 0)))
    # ── Windowed whitelist supplement ────────────────────────────────────
    # Window defines the rescue candidate pool; per-measure pass_rate triggers rescue.
    WINDOW_SIZE = 8
    WINDOW_RESCUE_THRESHOLD = 0.40  # per-measure pass_rate below this triggers rescue
    rescued_tokens_global = set()   # token-level (used for residual filter)
    rescued_pairs_global = set()    # (measure, token) pairs for measure-local check
    if excluded and ta:
        max_m = max(max(cands_by_m.keys(), default=0), max(excluded_by_m.keys(), default=0))
        for w_start in range(0, max_m + 1, WINDOW_SIZE):
            w_end = w_start + WINDOW_SIZE

            # Compute window-adaptive duration threshold from all tokens in window
            window_durations = []
            for m in range(w_start, w_end):
                for (idx, tok, pct, iorigin) in list(cands_by_m.get(m, [])) + list(excluded_by_m.get(m, [])):
                    info = ta.get(tok) if ta else None
                    if info and info.get("decode_ok"):
                        window_durations.append(info.get("duration_ms", 0))
            if window_durations:
                window_durations.sort()
                p75 = window_durations[min(len(window_durations) - 1,
                                           len(window_durations) * 3 // 4)]  # P75
                window_dur_threshold = max(10.0, p75)
            else:
                window_dur_threshold = FX_DURATION_THRESHOLD

            # Build window-level rescue candidate pool (ranked by occurrence)
            # Include fx tokens if their duration is within window P75
            window_tokens = {}
            for m in range(w_start, w_end):
                for (idx, tok, pct, iorigin) in excluded_by_m.get(m, []):
                    if tok not in window_tokens:
                        # Check if fx-excluded token is rescuable by window duration
                        reason = excluded.get(tok, "")
                        if reason == "fx":
                            info = ta.get(tok) if ta else None
                            dur = (info or {}).get("duration_ms", 99999)
                            if dur > window_dur_threshold * 1.5:
                                continue  # truly extreme duration, skip
                        total_occ = (key_occ or {}).get(tok, 0) + (scratch_occ or {}).get(tok, 0) + (bgm_occ or {}).get(tok, 0)
                        window_tokens[tok] = (total_occ, pct)
            if not window_tokens:
                continue
            # Rank once for the window
            ranked_tokens = sorted(window_tokens.keys(),
                                   key=lambda t: (-window_tokens[t][0], -window_tokens[t][1]))

            # Per-measure rescue within this window
            for m in range(w_start, w_end):
                m_wl = len(cands_by_m.get(m, []))
                m_ex = len(excluded_by_m.get(m, []))
                m_total = m_wl + m_ex
                if m_total == 0:
                    continue
                m_pass_rate = m_wl / m_total
                if m_pass_rate >= WINDOW_RESCUE_THRESHOLD:
                    continue

                # Rescue from window pool for this measure
                target = max(1, round(m_total * WINDOW_RESCUE_THRESHOLD))
                to_rescue = target - m_wl
                if to_rescue <= 0:
                    continue

                # Find excluded entries in this measure whose token is in ranked pool
                m_excluded_tokens = {entry[1] for entry in excluded_by_m.get(m, [])}
                rescue_set = set()
                for tok in ranked_tokens:
                    if len(rescue_set) >= to_rescue:
                        break
                    if tok in m_excluded_tokens:
                        rescue_set.add(tok)

                # Move rescued entries
                new_excluded = []
                for entry in excluded_by_m.get(m, []):
                    idx, tok, pct, iorigin = entry
                    if tok in rescue_set:
                        cands_by_m[m].append(entry)
                        rescued_tokens_global.add(tok)
                        rescued_pairs_global.add((m, tok))
                    else:
                        new_excluded.append(entry)
                excluded_by_m[m] = new_excluded

    # ── DP played-content rescue ───────────────────────────────────────
    # Restore the SOURCE KEY onsets (ch 11-19 only — NOT the token's BGM
    # occurrences, which a token-level whitelist change would flood; Codex
    # 2026-06-14) of whitelist-excluded tokens, so the DP chart plays the lead /
    # bass the SP whitelist stripped. Gate on attack > fx-soft line: a soft pad
    # (the only fx reason worth respecting at a played position) stays cut; a
    # non-soft onset the source played as a key is restored. DP-only.
    dp_rescued = 0
    if dp and excluded and ta:
        existing = {(m, c[0], c[1]) for m in cands_by_m for c in cands_by_m[m]}
        rescued_here = set()
        for ev in events:
            et = ev.get("type")
            if et == "Tap":
                ch, tok, m, idx = (ev.get("rawChannel", ""), ev["token"],
                                   ev["measure"], ev["idx192"])
            elif et == "Long":
                ch, tok, m, idx = (ev.get("rawChannelStart", ""), ev["tokenStart"],
                                   ev["measureStart"], ev["idx192Start"])
            else:
                continue
            if ch not in CH_TO_KEY_LANE:
                continue  # source key channels only (no BGM occurrences)
            if tok not in excluded or excluded[tok] == "unknown":
                continue
            key = (m, idx, tok)
            if key in existing or key in rescued_here:
                continue
            infox = ta.get(tok) or {}
            if not infox.get("decode_ok"):
                continue
            if pct_map.get(tok, 0.0) <= DP_RESCUE_MIN_ATTACK:
                continue  # soft pad — leave excluded (→ BGM)
            rescued_here.add(key)
            cands_by_m[m].append((idx, tok, pct_map.get(tok, 50.0),
                                  intensity_origin_map.get(tok, 0)))
            rescued_tokens_global.add(tok)
            rescued_pairs_global.add((m, tok))
            dp_rescued += 1

    # Centroid-based lane state: auto-calibrate step sensitivity per song
    centroid_step_unit = 300.0  # fallback
    if ta:
        pool_centroids = []
        for tok in whitelist:
            info = ta.get(tok)
            if info and info.get("decode_ok") and info.get("spectral_centroid_mean", 0) > 0:
                pool_centroids.append(info["spectral_centroid_mean"])
        if len(pool_centroids) >= 10:
            pool_centroids.sort()
            diffs = [abs(pool_centroids[i] - pool_centroids[i-1])
                     for i in range(1, len(pool_centroids)) if pool_centroids[i] != pool_centroids[i-1]]
            if diffs:
                diffs.sort()
                # P50 of non-zero diffs = 1 lane step
                # Floor at 300Hz to desensitize against tiny inter-token variation
                centroid_step_unit = max(300.0, diffs[len(diffs) // 2])
    centroid_state = {"prev_lane_idx": 3, "prev_centroid": None,
                      "step_unit": centroid_step_unit}  # start at center (KEY4)

    # Rule-based default ordering preserved; ML may override per measure inside the loop.
    for m in cands_by_m:
        cands_by_m[m].sort(key=lambda x: (x[0], -x[2], -x[3]))

    # ── DP: template router + side-paired carry-over state ─────────────
    # Build the per-measure token→side map once (phase-block SPLIT, #2), then
    # initialise two independent SP-state sets — one per hand. The SP placement
    # machine is reused side-local (addon §2); each side keeps its own jack /
    # hand / centroid / streak / lane-affinity memory.
    dp_side_by_m = {}
    dp_state_L = dp_state_R = None
    dp_scr_state = None
    dp_split_mode = "timbre"
    dp_placed_left = dp_placed_right = 0
    dp_scratch_left = dp_scratch_right = dp_scratch_gated = 0
    dp_blocks_swapped = dp_measure_rebalanced = 0
    if dp:
        dp_side_by_m, dp_blocks_swapped = _dp_build_side_maps(cands_by_m, phase_blocks, ta)
        def _new_dp_state():
            return {"hand": ("balanced", 0), "jack": {}, "jack_streak": defaultdict(int),
                    "centroid": {"prev_lane_idx": 3, "prev_centroid": None,
                                 "step_unit": centroid_step_unit},
                    "lane_mem": {}}
        dp_state_L, dp_state_R = _new_dp_state(), _new_dp_state()
        dp_scr_state = {"scr_measures": 0}  # counts scratch measures → hand swap parity
        # split-strategy router (addon §9): timbre (low_freq, STREAM songs) vs
        # balance (split chords / alternate bursts, CHORD/PEAK songs). A naive
        # candidate/source proxy can't tell jumpstream (stream-dominant, e.g.
        # signal) from chord-dominant (e.g. wanwan) — both pack simultaneous
        # notes; only the calibrated NoteAttributes chord/stream formula
        # discriminates (tools/dp_source_character.py). So `auto` defaults to the
        # validated-safe timbre split; pick `balance` explicitly for chord/peak
        # songs (use dp_source_character.py to read the dominant attribute).
        dp_split_mode = dp_split if dp_split in ("timbre", "balance") else "timbre"

    key_lane_idx_by_m = defaultdict(list)
    for ev in events:
        etype = ev.get("type")
        if etype == "Tap":    token, ch, m = ev["token"], ev["rawChannel"], ev["measure"]
        elif etype == "Long": token, ch, m = ev["tokenStart"], ev["rawChannelStart"], ev["measureStart"]
        else: continue
        if ch in CH_TO_KEY_LANE and token in whitelist:
            key_lane_idx_by_m[m].append(LANE_INDEX[CH_TO_KEY_LANE[ch]])

    scratch_seed_set = set(scratch_seeds)
    scratch_active = bool(scratch_seed_set)
    scratch_pool_by_m = defaultdict(set)
    for ev in events:
        etype = ev.get("type")
        if etype == "Tap" and ev["rawChannel"] == "16":
            scratch_pool_by_m[ev["measure"]].add((ev["idx192"], ev["token"]))
        elif etype == "Long" and ev["rawChannelStart"] == "16":
            scratch_pool_by_m[ev["measureStart"]].add((ev["idx192Start"], ev["tokenStart"]))

    jack_scr_tkey = -999
    scratch_history = deque(maxlen=SCRATCH_RUSH_WINDOW)
    scr_rest_remain = 0

    # ── §12 REVISED 2026-06-13: tier-2 scratch supplement ────────────────────
    # Source ch16 positions are tier 1 (exhausted first, source order — keeps
    # scale <= 1.0 output byte-identical). When the requested budget exceeds
    # the source supply (scratch level > 5), additional wheel events are drawn
    # from the package pool at non-ch16 onsets:
    #   tier 2a — tokens this source itself used on the wheel (scratch_occ>0):
    #             the author's own wheel timbres, extended to new onsets.
    #   tier 2b — §12.2 functional-gate tokens (duration/attack/occurrence/
    #             origin), ordered by a weak audio prior. The prior is a
    #             RANKING signal only, deliberately not a gate: corpus
    #             measurement (DR-F7) found no audio-intrinsic "scratch-ness"
    #             (per-feature AUC 0.47-0.62 over 6,150 packages) — wheel
    #             timbre is author-subjective, so package-local evidence (2a)
    #             outranks any universal audio rule.
    scratch_supplement_by_m = {}
    tier2_by_m = {}
    if (scratch_active and scratch_mode == "primary"
            and source_scratch_per_measure is not None
            and scratch_scale > 1.0):
        _supplement_total = round(sum(source_scratch_per_measure.values())
                                  * (scratch_scale - 1.0))
        if _supplement_total > 0 and ta:
            _t2_rank = {}  # token -> sort rank (lower = preferred)
            _2a = sorted((t for t in (scratch_occ or {}) if scratch_occ.get(t, 0) > 0),
                         key=lambda t: (-scratch_occ.get(t, 0), t))
            for _i, _t in enumerate(_2a):
                _t2_rank[_t] = (0, _i)
            _2b = []
            for _t in sorted(ta):
                if _t in _t2_rank:
                    continue
                _info = ta.get(_t) or {}
                if not _info.get("decode_ok"):
                    continue
                if _info.get("duration_ms", 1e9) > SCRATCH_FALLBACK_DURATION_MAX:
                    continue
                if pct_map.get(_t, 0.0) < SCRATCH_FALLBACK_MIN_ATTACK_PERCENTILE:
                    continue
                _tot = ((key_occ or {}).get(_t, 0) + (scratch_occ or {}).get(_t, 0)
                        + (bgm_occ or {}).get(_t, 0))
                if _tot < SCRATCH_FALLBACK_MIN_OCCURRENCE:
                    continue
                if intensity_origin_map.get(_t, 0) != 1:
                    continue  # bgm-only origin ≈ background FX (§12.2 fx gate)
                _prior = (2.0 * float(_info.get("spectral_flatness_mean", 0.0) or 0.0)
                          + 2.0 * float(_info.get("zero_crossing_rate_mean", 0.0) or 0.0)
                          + pct_map.get(_t, 0.0) / 100.0)
                _2b.append((-_prior, _t))
            for _i, (_, _t) in enumerate(sorted(_2b)):
                _t2_rank[_t] = (1, _i)

            # candidate onsets: every pool event of a tier-2 token EXCEPT the
            # source ch16 positions (those are tier 1)
            _t2_seen = set()
            for ev in events:
                etype = ev.get("type")
                if etype == "Tap":
                    if ev.get("rawChannel") == "16":
                        continue
                    tok, m, idx = ev["token"], ev["measure"], ev["idx192"]
                elif etype == "Long":
                    if ev.get("rawChannelStart") == "16":
                        continue
                    tok, m, idx = ev["tokenStart"], ev["measureStart"], ev["idx192Start"]
                elif etype == "BGM":
                    tok, m, idx = ev["token"], ev["measure"], ev["idx192"]
                else:
                    continue
                if tok not in _t2_rank or (m, idx, tok) in _t2_seen:
                    continue
                _t2_seen.add((m, idx, tok))
                tier2_by_m.setdefault(m, []).append((_t2_rank[tok], idx, tok))
            for m in tier2_by_m:
                tier2_by_m[m].sort()

            # distribute the supplement over measures, weighted by candidate
            # availability, ceilinged per measure by min(level-lerped cap −
            # tier-1 budget, candidate count). Largest-remainder apportionment
            # (not plain floor) so the supplement isn't lost to per-measure
            # rounding when candidates are spread thin across many measures
            # (plain int() floor zeroed lv10 entirely — DR-F6 follow-up).
            _cap = p.get("SCRATCH_MAX_PER_MEASURE", SCRATCH_MAX_PER_MEASURE)
            _w_total = sum(len(v) for v in tier2_by_m.values())
            if _w_total > 0:
                _ms = sorted(tier2_by_m)
                _headroom = {}
                for m in _ms:
                    _budget_m = max(0, round(source_scratch_per_measure.get(m, 0)
                                             * scratch_scale))
                    _headroom[m] = min(max(0, _cap - _budget_m), len(tier2_by_m[m]))
                _assignable = min(_supplement_total, sum(_headroom.values()))
                # ideal real-valued share, capped by headroom; distribute the
                # integer floor first, then hand out the remaining units to the
                # measures with the largest fractional remainder (ties: lower m).
                _ideal = {m: min(_headroom[m],
                                 _supplement_total * len(tier2_by_m[m]) / _w_total)
                          for m in _ms}
                for m in _ms:
                    _s = int(_ideal[m])
                    if _s > 0:
                        scratch_supplement_by_m[m] = _s
                _remaining = _assignable - sum(scratch_supplement_by_m.values())
                # Multi-pass top-up: a single pass adds at most +1 per measure,
                # which under-distributes when a candidate-rich measure's share
                # was capped by headroom and the leftover exceeds the count of
                # measures with spare room (Codex audit 2026-06-13 D). Loop the
                # fractional-ordered measures until the remaining units are all
                # placed or every measure is at its headroom ceiling.
                _rema = sorted(
                    _ms,
                    key=lambda m: (-(_ideal[m] - int(_ideal[m])), m))
                while _remaining > 0:
                    _progressed = False
                    for m in _rema:
                        if _remaining <= 0:
                            break
                        if scratch_supplement_by_m.get(m, 0) < _headroom[m]:
                            scratch_supplement_by_m[m] = scratch_supplement_by_m.get(m, 0) + 1
                            _remaining -= 1
                            _progressed = True
                    if not _progressed:
                        break  # every measure at headroom ceiling
    scratch_tier2_inserted = 0
    prev_candidate_count = 0
    hand_state = ("balanced", 0); center_lane = "P1_KEY4"; jack_state = {}
    jack_streak = defaultdict(int)  # v10 §11.5 per-lane jack streak

    # 2026-06-11: beat-true tick mapping for scale-aware jack deltas (§11.5).
    # Identity on charts without #xxx02 scaling.
    scaled_tick = build_scaled_tick_fn(measure_scale, measure_max)

    # BPM lookup for v10 §11.5 jack floor (reuses the BPM-walk pattern from §13.4)
    _bpm_timeline = []
    if bpm_events:
        _bpm_timeline = sorted(bpm_events, key=lambda x: x[0])
    def bpm_lookup(tkey):
        if not _bpm_timeline:
            return base_bpm
        cur = base_bpm
        for (ev_tkey, ev_bpm) in _bpm_timeline:
            if ev_tkey <= tkey:
                cur = ev_bpm
            else:
                break
        return cur

    chb_count = stream_count = downgrade_count = total_placed = total_scratch = 0
    res = defaultdict(int)
    all_scratch_placed_keys = set()
    placed_events, residual_events = [], []
    residual_by_measure = defaultdict(int)
    hand_balance_total = same_hand_total = lane_affinity_total = 0
    density_ramp_warnings, lane_continuity_resets = [], []
    scratch_rush_rest_activations = []

    # 2026-05-03: running token usage for under-used boost.
    # Within same-idx candidates, after the highest-attack first pick, prefer
    # tokens that (a) maximize spectral distance from already-picked AND
    # (b) have low usage count so far in chart. This counteracts the natural
    # head-heavy distribution where ~60% of tokens get used ≤2× and the top
    # 10% accounts for 40-70% of placements.
    token_usage = defaultdict(int)

    # v12 §9.8 (2026-06-11): token→lane affinity memory — token: (lane, measure)
    # of the most recent key-lane placement. Cross-measure carry-over state
    # (serialized in resume-v2).
    token_lane_memory = {}

    # ── v12 §23 Resume API: carry-over state override + measure range ─────────
    if resume_state is not None:
        # v1 = RB-only (§23.8). ML carry-over state has separate schema (future).
        if ml_ctx is not None and (
                ml_ctx["state"].get("enable_token") or ml_ctx["state"].get("enable_lane")):
            raise NotImplementedError(
                "Resume API v1 is RB-only (§23.8). ML resume requires a separate "
                "schema (token_context / lane_context / global_lane_counts) — "
                "not supported in v1.")
        loaded = _resume_state.load_resume_state(
            resume_state, scratch_history_maxlen=SCRATCH_RUSH_WINDOW)
        (after_measure_loaded, seed_int_loaded,
         jack_state, jack_streak, centroid_state_loaded,
         hand_state, token_usage,
         jack_scr_tkey, scratch_history, scr_rest_remain,
         token_lane_memory) = loaded
        # centroid step_unit is chart-input deterministic; load_resume_state
        # leaves it unset, so we inject the freshly computed value.
        centroid_state_loaded["step_unit"] = centroid_step_unit
        centroid_state = centroid_state_loaded
        seed_int = seed_int_loaded
        expected_start = after_measure_loaded + 1
        if start_measure is None:
            start_measure = expected_start
        elif start_measure != expected_start:
            raise ValueError(
                f"start_measure ({start_measure}) does not match resume_state "
                f"after_measure+1 ({expected_start}). resume input mismatch.")

    loop_start = start_measure if start_measure is not None else 0
    loop_end = end_measure if end_measure is not None else measure_max
    if loop_start < 0 or loop_end > measure_max or loop_start > loop_end:
        raise ValueError(
            f"Invalid measure range [start={loop_start}, end={loop_end}] "
            f"for chart with measure_max={measure_max}.")

    def _reorder_within_idx(group, ta_map, usage):
        if len(group) <= 1:
            return group
        def centroid(tok):
            info = ta_map.get(tok) if ta_map else None
            if info and info.get("decode_ok"):
                return float(info.get("spectral_centroid_mean", 0.0) or 0.0)
            return 0.0
        # 2026-05-03: also apply usage penalty to FIRST pick — otherwise the
        # same popular high-pct token gets selected as first-mate at every
        # chord position in the chart, which keeps the head-heavy long-tail
        # distribution intact. Score = pct - usage_penalty * usage_count.
        # Scale: pct is 0-100, so usage_penalty=10 means each prior use costs
        # 10 pct points (~1/10 of the percentile range).
        usage_penalty_first = 10.0
        # Reference scale for spread reorder: 1000 Hz centroid distance ≈
        # usage_count of 1 for the secondary picks.
        usage_weight_spread = 1000.0
        # First pick: combined attack pct + usage penalty
        first_score = lambda item: item[2] - usage_penalty_first * usage.get(item[1], 0)
        first_idx = max(range(len(group)), key=lambda i: first_score(group[i]))
        first = group[first_idx]
        remaining = [g for i, g in enumerate(group) if i != first_idx]
        ordered = [first]
        chosen_c = [centroid(first[1])]
        while remaining:
            def score(item):
                tok = item[1]
                c = centroid(tok)
                spectral_min_d = min(abs(c - cc) for cc in chosen_c)
                return spectral_min_d - usage_weight_spread * usage.get(tok, 0)
            best = max(range(len(remaining)), key=lambda i: score(remaining[i]))
            picked = remaining.pop(best)
            ordered.append(picked)
            chosen_c.append(centroid(picked[1]))
        return ordered

    for measure in range(loop_start, loop_end + 1):
        # v12 §23.4 (DR-23-5): fresh per-measure RNG (β-1) — measure-level isolation
        # for resume API. simple arithmetic mapping (Python 3.13 random.Random rejects
        # tuple seeds); 1M offset is collision-free for any realistic chart length.
        rng = random.Random(seed_int * 1_000_000 + measure)
        curr_cands = cands_by_m.get(measure, [])
        next_cands = cands_by_m.get(measure + 1, []) if measure + 1 <= measure_max else None

        # 2026-05-03: per-measure within-idx reorder for spectral spread + low-usage boost.
        if curr_cands:
            grouped = []
            i, lst = 0, curr_cands
            while i < len(lst):
                j = i
                while j < len(lst) and lst[j][0] == lst[i][0]:
                    j += 1
                grouped.extend(_reorder_within_idx(lst[i:j], ta, token_usage))
                i = j
            curr_cands = grouped

        kidx = key_lane_idx_by_m.get(measure, [])
        computed_center = IDX_TO_KEY_LANE.get(sorted(kidx)[len(kidx) // 2], "P1_KEY4") if kidx else "P1_KEY4"
        candidate_delta = len(curr_cands) - prev_candidate_count
        if abs(candidate_delta) > 8:
            center_lane = computed_center; lane_continuity_resets.append(measure)
        if candidate_delta > 8: density_ramp_warnings.append(measure)

        phase = _get_phase(measure, phase_blocks)
        primitive = "Stream"

        # ── DP: self-contained measure handling ─────────────────────
        if dp:
            # Decide this measure's scratch hand (measure-unit, swapped each
            # scratch measure). Keys are placed by the normal bass/melodic split;
            # the anti-jump gate then clears scratch-hand keys near scratches.
            scr_cands, budget_m, scratch_hand = [], 0, None
            if scratch_active and scratch_mode == "primary":
                src_n = (source_scratch_per_measure or {}).get(measure, 0)
                budget_m = max(0, round(src_n * scratch_scale))
                scr_cands = [(i, t) for (i, t) in sorted(scratch_pool_by_m.get(measure, set()))
                             if t in scratch_seed_set]
                if scr_cands and budget_m > 0:
                    scratch_hand = "L" if dp_scr_state["scr_measures"] % 2 == 0 else "R"
                    dp_scr_state["scr_measures"] += 1
            # build per-onset side assignment by the active strategy.
            if dp_split_mode == "balance":
                cand_sides = _dp_balance_sides(curr_cands, ta)
            else:
                # timbre: block bass/melodic map + one-sided rebalance fallback.
                side_map = dp_side_by_m.get(measure, {})
                if curr_cands and len({c[1] for c in curr_cands}) >= 2:
                    _lc = sum(1 for c in curr_cands if side_map.get(c[1], "L") == "L")
                    if _lc == 0 or _lc == len(curr_cands):
                        side_map = _dp_measure_balanced_split(curr_cands, ta)
                        dp_measure_rebalanced += 1
                cand_sides = [side_map.get(c[1], "L") for c in curr_cands]
            placed, m_res, m_diag = _place_measure_dp(
                curr_cands, cand_sides, rng,
                dp_state_L, dp_state_R, measure, p, ta, bpm_lookup, scaled_tick)
            # anti-jump gate — remove scratch-hand keys near scratch onsets
            # (their audio survives as BGM; timing untouched).
            scr_list = []
            if scratch_hand is not None:
                scr_list, gated = _dp_scratch_measure(
                    measure, placed, scr_cands, budget_m,
                    source_min_interval, scratch_hand)
                for (gi, gt, gl) in gated:
                    res["dp_scratch_gate"] += 1
                    residual_events.append({"token": gt, "measure": measure,
                                            "idx192": gi, "reason": "dp_scratch_gate"})
                    residual_by_measure[measure] += 1
                    dp_scratch_gated += 1
            stream_count += 1
            for (idx192, token, lane) in placed:
                placed_events.append({"token": token, "lane": lane, "measure": measure,
                                      "idx192": idx192, "primitive": primitive, "phase": phase})
                token_usage[token] += 1
                if lane.startswith("P2"): dp_placed_right += 1
                else: dp_placed_left += 1
            total_placed += len(placed)
            for (idx192, token, side) in scr_list:
                lane = "P2_SCR" if side == "R" else "P1_SCR"
                placed_events.append({"token": token, "lane": lane, "measure": measure,
                                      "idx192": idx192, "primitive": primitive, "phase": phase})
                all_scratch_placed_keys.add((measure, idx192, token))
                if side == "R": dp_scratch_right += 1
                else: dp_scratch_left += 1
            total_scratch += len(scr_list)
            for (m_i_t, reason) in m_res.items():
                res[reason] += 1
                residual_events.append({"token": m_i_t[2], "measure": m_i_t[0],
                                        "idx192": m_i_t[1], "reason": reason})
                residual_by_measure[m_i_t[0]] += 1
            hand_balance_total += m_diag["hand_balance_applied"]
            same_hand_total += m_diag["same_hand_applied"]
            lane_affinity_total += m_diag.get("lane_affinity_applied", 0)
            prev_candidate_count = len(curr_cands)
            continue

        # ML soft re-ranking: replace pct-based ordering with model scores when active.
        if ml_ctx is not None and ml_ctx["state"]["enable_token"] and curr_cands:
            ml_scores = _ml_score_tokens(ml_ctx, measure, phase, len(curr_cands))
            if ml_scores is not None:
                pi = ml_ctx["pool_index"]
                curr_cands = sorted(
                    curr_cands,
                    key=lambda x: (x[0], -float(ml_scores[pi.get(x[1], 0)]), -x[3]),
                )

        # §23.7 boundary lookahead: only forward N+1 first-chord constraints
        # when processing the final measure of the loop range.
        lookahead_for_this_measure = (
            next_chord_lookahead if (next_chord_lookahead is not None
                                      and measure == loop_end) else None)
        if primitive == "ChordBurst":
            placed, hand_state, m_res, failed, m_diag = _place_measure_constrained(
                curr_cands, rng, hand_state, jack_state, measure, True, p,
                ml_ctx=ml_ctx, phase_label=phase, ta=ta, centroid_state=centroid_state,
                bpm_lookup=bpm_lookup, jack_streak=jack_streak,
                next_chord_lookahead=lookahead_for_this_measure,
                scaled_tick=scaled_tick, token_lane_memory=token_lane_memory)
            if failed:
                placed, hand_state, m_res, _, m_diag = _place_measure_constrained(
                    curr_cands, rng, hand_state, jack_state, measure, False, p,
                    ml_ctx=ml_ctx, phase_label=phase, ta=ta, centroid_state=centroid_state,
                    bpm_lookup=bpm_lookup, jack_streak=jack_streak,
                    next_chord_lookahead=lookahead_for_this_measure,
                    scaled_tick=scaled_tick, token_lane_memory=token_lane_memory)
                downgrade_count += 1; primitive = "Stream"; stream_count += 1
            else: chb_count += 1
        else:
            placed, hand_state, m_res, stream_failed, m_diag = _place_measure_constrained(
                curr_cands, rng, hand_state, jack_state, measure, False, p,
                ml_ctx=ml_ctx, phase_label=phase, ta=ta, centroid_state=centroid_state,
                bpm_lookup=bpm_lookup, jack_streak=jack_streak,
                next_chord_lookahead=lookahead_for_this_measure,
                scaled_tick=scaled_tick, token_lane_memory=token_lane_memory)
            stream_count += 1
            if stream_failed:
                res["primitive_failed"] += len(curr_cands)
                for (idx192, token, _, _) in curr_cands:
                    residual_events.append({"token": token, "measure": measure, "idx192": idx192, "reason": "primitive_failed"})
                residual_by_measure[measure] += len(curr_cands)
                m_res = {}; placed = []

        hand_balance_total += m_diag["hand_balance_applied"]
        same_hand_total += m_diag["same_hand_applied"]
        lane_affinity_total += m_diag.get("lane_affinity_applied", 0)

        for (idx192, token, lane) in placed:
            placed_events.append({"token": token, "lane": lane, "measure": measure,
                                  "idx192": idx192, "primitive": primitive, "phase": phase})
            token_usage[token] += 1  # for next-measure under-used boost
        total_placed += len(placed)

        scratch_placed_keys = set()
        if scratch_active:
            placed_key_set = set((i, t) for (i, t, _) in placed)
            # 2026-05-05: sorted() to remove PYTHONHASHSEED-dependent set iteration
            # ordering — _insert_scratch_in_measure consumes candidates in input order,
            # so set hash randomization made placement nondeterministic across runs.
            scratch_candidates = [(i, t) for (i, t) in sorted(scratch_pool_by_m.get(measure, set()))
                                  if t in scratch_seed_set and (i, t) not in placed_key_set]
            supplement_m = scratch_supplement_by_m.get(measure, 0)
            if scratch_candidates or supplement_m > 0:
                if scr_rest_remain > 0:
                    res["scratch_suppressed"] += len(scratch_candidates)
                    for (idx192, token) in scratch_candidates:
                        residual_events.append({"token": token, "measure": measure, "idx192": idx192, "reason": "scratch_suppressed"})
                    residual_by_measure[measure] += len(scratch_candidates)
                    scr_rest_remain -= 1; scratch_history.append(0)
                else:
                    # 2026-05-03: per-measure budget + min-interval from source in primary mode.
                    if scratch_mode == "primary" and source_scratch_per_measure is not None:
                        src_n = source_scratch_per_measure.get(measure, 0)
                        budget_m = max(0, round(src_n * scratch_scale))
                        # Use source's own min interval as floor (with safety) instead
                        # of the level-based 16 ticks. Source author already shaped
                        # the spacing.
                        min_int_m = source_min_interval
                    else:
                        budget_m = None  # use legacy SCRATCH_MAX_PER_MEASURE from params
                        min_int_m = None
                    scr_placed = []
                    if scratch_candidates:
                        scr_placed, scr_res, jack_scr_tkey = _insert_scratch_in_measure(
                            scratch_candidates, placed, measure, jack_scr_tkey, pct_map, p,
                            max_per_m_override=budget_m,
                            min_interval_override=min_int_m)
                        res["scratch_interval"] += scr_res["scratch_interval"]
                        res["scratch_density_cap"] += scr_res["scratch_density_cap"]
                    # §12 REVISED 2026-06-13: tier-2 supplement — only reachable
                    # when the requested budget exceeds the source supply
                    # (scratch level > 5; scratch_supplement_by_m empty otherwise,
                    # keeping scale <= 1.0 byte-identical to the tier-1-only era).
                    if supplement_m > 0:
                        _eff_min_int = (min_int_m if min_int_m is not None
                                        else p.get("SCRATCH_MIN_INTERVAL", SCRATCH_MIN_INTERVAL))
                        t2_placed = _insert_scratch_supplement(
                            tier2_by_m.get(measure, []), supplement_m, measure,
                            [measure * 192 + i for (i, _) in scr_placed] + [jack_scr_tkey],
                            placed_key_set, scr_placed, _eff_min_int)
                        if t2_placed:
                            scratch_tier2_inserted += len(t2_placed)
                            scr_placed = scr_placed + t2_placed
                            jack_scr_tkey = max(
                                jack_scr_tkey,
                                max(measure * 192 + i for (i, _) in t2_placed))
                    total_scratch += len(scr_placed)
                    scratch_history.append(len(scr_placed))
                    for (idx192, tok) in scr_placed:
                        placed_events.append({"token": tok, "lane": "P1_SCR", "measure": measure,
                                              "idx192": idx192, "primitive": primitive, "phase": phase})
                        scratch_placed_keys.add((measure, idx192, tok))
                    # 2026-05-03: RUSH only in fallback mode. Primary mode trusts
                    # source pacing — author already shaped the rhythm.
                    if (scratch_mode == "fallback"
                            and len(scratch_history) == SCRATCH_RUSH_WINDOW
                            and all(c >= rush_threshold for c in scratch_history)
                            and SCRATCH_RUSH_REST_ENABLED and scr_rest_remain == 0):
                        scr_rest_remain = rush_rest_measures
                        scratch_rush_rest_activations.append({
                            "trigger_measure": measure,
                            "suppressed_range": [measure + 1, measure + rush_rest_measures]})
            else: scratch_history.append(0)
        else: scratch_history.append(0)

        for (m_i_t, reason) in m_res.items():
            if m_i_t in scratch_placed_keys: continue
            res[reason] += 1
            residual_events.append({"token": m_i_t[2], "measure": m_i_t[0], "idx192": m_i_t[1], "reason": reason})
            residual_by_measure[m_i_t[0]] += 1

        all_scratch_placed_keys.update(scratch_placed_keys)
        prev_candidate_count = len(curr_cands)

        if ml_ctx is not None and ml_ctx["state"]["enable_token"] and len(placed) > 0:
            rms_vals = []
            for (_, tok, _) in placed:
                info = ml_ctx["ta_map"].get(tok, {}) or {}
                if info.get("decode_ok"):
                    rms_vals.append(float(info.get("attack_rms", 0.0)))
            mean_rms = sum(rms_vals) / len(rms_vals) if rms_vals else 0.0
            ml_ctx["token_context"].append({
                "tkey_delta_base": measure * 192,
                "placed_count": len(placed),
                "mean_attack_rms": mean_rms,
            })

    result = {
        "chb_count": chb_count, "stream_count": stream_count,
        "downgrade_count": downgrade_count,
        "total_placed": total_placed, "scratch_inserted": total_scratch,
        "scratch_tier2_inserted": scratch_tier2_inserted,
        "scratch_supplement_by_measure": scratch_supplement_by_m,
        "residual_counts": dict(res),
        "placed_events": placed_events, "residual_events": residual_events,
        "scratch_placed_keys": all_scratch_placed_keys,
        "hand_balance_avoided_count": hand_balance_total,
        "same_hand_alternation_count": same_hand_total,
        "lane_affinity_applied_count": lane_affinity_total,
        "density_ramp_warnings": density_ramp_warnings,
        "lane_continuity_resets": lane_continuity_resets,
        "scratch_rush_rest_activations": scratch_rush_rest_activations,
        "residual_by_measure": dict(residual_by_measure),
        "rescued_tokens": rescued_tokens_global,
        "rescued_pairs": rescued_pairs_global,
        "centroid_step_unit": round(centroid_step_unit, 2),
        "dp_enabled": dp,
        "dp_placed_left": dp_placed_left,
        "dp_placed_right": dp_placed_right,
        "dp_scratch_left": dp_scratch_left,
        "dp_scratch_right": dp_scratch_right,
        "dp_scratch_gated": dp_scratch_gated,
        "dp_scratch_measures": (dp_scr_state["scr_measures"] if dp_scr_state else 0),
        "dp_blocks_swapped": dp_blocks_swapped,
        "dp_rescued": dp_rescued if dp else 0,
        "dp_measure_rebalanced": dp_measure_rebalanced,
        "dp_split_mode": dp_split_mode,
    }
    # v12 §23.5: resume mode emits partial result with end_state for cascading.
    if resume_state is not None:
        result["mode"] = "resume"
        result["start_measure"] = loop_start
        result["end_measure"] = loop_end
        result["end_state"] = _resume_state.extract_resume_state(
            seed_int=seed_int, after_measure=loop_end,
            jack_state=jack_state, jack_streak=jack_streak,
            centroid_state=centroid_state,
            hand_state=hand_state, token_usage=token_usage,
            jack_scr_tkey=jack_scr_tkey,
            scratch_history=scratch_history, scr_rest_remain=scr_rest_remain,
            token_lane_memory=token_lane_memory)
    return result


# ── §13.9  Scratch adjustment (post-LN) ──────────────────────────────────────

def run_scratch_adjustment(placed_events, residual_events, ln_meta, pct_map,
                           scratch_max_per_measure):
    if not ln_meta.get("enabled") or ln_meta.get("ln_promoted_count", 0) == 0:
        return placed_events, residual_events, {
            "ln_occupied_measures": 0, "scratch_suppressed_by_ln": 0}

    ln_occupied = set()
    for ev in placed_events:
        if ev.get("type") == "LN":
            for m in range(ev["measure_start"], ev["measure_end"] + 1):
                ln_occupied.add(m)

    effective_cap = max(1, scratch_max_per_measure // 2)

    scr_by_measure = defaultdict(list)
    other_events = []
    for ev in placed_events:
        if (ev.get("type", "Tap") == "Tap" and ev.get("lane") == "P1_SCR"
                and ev.get("measure", ev.get("measure_start", -1)) in ln_occupied):
            scr_by_measure[ev["measure"]].append(ev)
        else:
            other_events.append(ev)

    suppressed = 0
    kept_scratch = []
    for measure in sorted(scr_by_measure.keys()):
        evs = scr_by_measure[measure]
        if len(evs) <= effective_cap:
            kept_scratch.extend(evs)
            continue
        evs.sort(key=lambda e: pct_map.get(e["token"], 0.0))
        to_keep = evs[len(evs) - effective_cap:]
        to_remove = evs[:len(evs) - effective_cap]
        kept_scratch.extend(to_keep)
        for ev in to_remove:
            residual_events.append({
                "token": ev["token"], "measure": ev["measure"],
                "idx192": ev["idx192"], "reason": "ln_scratch_suppressed"})
            suppressed += 1

    new_placed = other_events + kept_scratch
    return new_placed, residual_events, {
        "ln_occupied_measures": len(ln_occupied),
        "scratch_suppressed_by_ln": suppressed,
    }


# ── §13.10  Density rebalancing ───────────────────────────────────────────────

def run_density_rebalance(placed_events, residual_events, measure_max,
                          pct_map, density_rebalance_max_delta,
                          phase_blocks, measure_note_cap=999,
                          max_chord_size=7, ml_ctx=None,
                          max_jack_streak=3, rng=None,
                          measure_scale=None):
    import math as _math
    # Use placed events' actual measure range (not pool measure_max which
    # includes BGM-only trailing measures that dilute density calculation)
    placed_measures = [ev.get("measure", ev.get("measure_start", -1))
                       for ev in placed_events
                       if ev.get("type", "Tap") == "Tap"
                       and ev.get("lane", "") != "P1_SCR"]
    _ml_fillback_used_skip = (
        ml_ctx is not None and ml_ctx["state"]["enable_token"]
    )
    if not placed_measures:
        return placed_events, residual_events, {
            "rebalancing_skipped": True, "delta_ratio_before": 0.0,
            "segment_densities": [], "notes_removed_per_segment": [],
            "notes_added_per_segment": [],
            "ml_fillback_used": _ml_fillback_used_skip,
            "ml_fillback_score_calls": 0,
            "ml_fillback_fallback_count": 0}
    actual_max = max(placed_measures)
    total_measures = actual_max + 1
    seg_size = _math.ceil(total_measures / 4)
    segments = []
    for i in range(4):
        start = i * seg_size
        end = min(start + seg_size, total_measures)
        if start >= total_measures:
            break
        segments.append((start, end))

    def _seg_density(start, end):
        count = sum(1 for ev in placed_events
                    if ev.get("type", "Tap") == "Tap"
                    and ev.get("lane", "") != "P1_SCR"
                    and start <= ev.get("measure", ev.get("measure_start", -1)) < end)
        return count / (end - start) if end > start else 0.0

    densities = [_seg_density(s, e) for s, e in segments]
    max_d = max(densities) if densities else 0.0
    min_d = min(densities) if densities else 0.0
    delta_ratio = (max_d - min_d) / max_d if max_d > 0 else 0.0

    no_change = {
        "rebalancing_skipped": True,
        "delta_ratio_before": round(delta_ratio, 4),
        "segment_densities": [round(d, 2) for d in densities],
        "notes_removed_per_segment": [0] * len(segments),
        "notes_added_per_segment": [0] * len(segments),
        "ml_fillback_used": _ml_fillback_used_skip,
        "ml_fillback_score_calls": 0,
        "ml_fillback_fallback_count": 0,
    }

    if delta_ratio <= density_rebalance_max_delta:
        return placed_events, residual_events, no_change

    import math as _m
    SOFT_KNEE_K = 5.0
    overshoot = delta_ratio - density_rebalance_max_delta
    correction = 1.0 - _m.exp(-SOFT_KNEE_K * overshoot)
    hard_target = max_d * (1 - density_rebalance_max_delta)

    notes_removed = [0] * len(segments)
    notes_added   = [0] * len(segments)
    ml_fillback_score_calls    = 0   # measures for which we asked the ML token model
    ml_fillback_fallback_count = 0   # fill-back-local model failures (§21.4.5)
    ml_fillback_used           = (
        ml_ctx is not None and ml_ctx["state"]["enable_token"]
    )

    # ── Shrink high-density segments (soft-knee) ─────────────────────────
    for i, (start, end) in enumerate(segments):
        if densities[i] <= hard_target:
            continue
        current_count = sum(1 for ev in placed_events
                            if ev.get("type", "Tap") == "Tap"
                            and ev.get("lane", "") != "P1_SCR"
                            and start <= ev.get("measure", -1) < end)
        hard_count = round(hard_target * (end - start))
        to_remove = round((current_count - hard_count) * correction)
        if to_remove <= 0:
            continue

        candidates = [ev for ev in placed_events
                      if ev.get("type", "Tap") == "Tap"
                      and ev.get("lane", "") != "P1_SCR"
                      and start <= ev.get("measure", -1) < end]
        candidates.sort(key=lambda ev: pct_map.get(ev["token"], 0.0))

        for ev in candidates[:to_remove]:
            placed_events.remove(ev)
            residual_events.append({
                "token": ev["token"], "measure": ev["measure"],
                "idx192": ev["idx192"], "reason": "density_rebalance"})
            notes_removed[i] += 1

    # ── Build jack state from current placed events ───────────────────────
    jack_state = defaultdict(int)
    for ev in placed_events:
        if ev.get("type", "Tap") == "Tap":
            tkey = ev["measure"] * 192 + ev["idx192"]
            lane = ev["lane"]
            if tkey > jack_state[lane]:
                jack_state[lane] = tkey
        elif ev.get("type") == "LN":
            tkey = ev["measure_start"] * 192 + ev["idx192_start"]
            lane = ev["lane"]
            if tkey > jack_state[lane]:
                jack_state[lane] = tkey

    # Build full occupancy for collision checks
    occupied = set()
    token_at_pos = defaultdict(set)
    for ev in placed_events:
        if ev.get("type", "Tap") == "Tap":
            occupied.add((ev["measure"], ev["idx192"], ev["lane"]))
            token_at_pos[(ev["measure"], ev["idx192"])].add(ev["token"])
        elif ev.get("type") == "LN":
            occupied.add((ev["measure_start"], ev["idx192_start"], ev["lane"]))
            token_at_pos[(ev["measure_start"], ev["idx192_start"])].add(ev["token"])
            # 2026-05-03: LN end position (where LNOBJ marker is written) is
            # also a slot in the BMS channel grid; fill-back must not insert
            # a Tap there (causes A_placed_completeness FAIL — the writer can
            # only emit one token per (m, ch, idx)).
            occupied.add((ev["measure_end"], ev["idx192_end"], ev["lane"]))

    # Per-lane sorted tkey list for jack check
    lane_tkeys = defaultdict(list)
    for ev in placed_events:
        if ev.get("type", "Tap") == "Tap":
            lane_tkeys[ev["lane"]].append(ev["measure"] * 192 + ev["idx192"])
        elif ev.get("type") == "LN":
            lane_tkeys[ev["lane"]].append(ev["measure_start"] * 192 + ev["idx192_start"])
    for lane in lane_tkeys:
        lane_tkeys[lane].sort()

    # 2026-06-11: jack gaps measured on the beat-true axis (scale-aware).
    _scaled_fb = build_scaled_tick_fn(measure_scale, measure_max)

    def _jack_ok(lane, tkey):
        import bisect
        tks = lane_tkeys[lane]
        pos = bisect.bisect_left(tks, tkey)
        st = _scaled_fb(tkey)
        if pos > 0 and round(st - _scaled_fb(tks[pos - 1]), 6) <= 12:
            return False
        if pos < len(tks) and round(_scaled_fb(tks[pos]) - st, 6) <= 12:
            return False
        return True

    # 2026-05-03: precompute global chord-boundary tkeys (sorted unique tkeys
    # across all key lanes) for jack_streak check. Recomputed lazily after
    # inserts since fill-back doesn't process residuals in tkey order.
    def _all_chord_boundaries():
        seen = set()
        for L in KEY_LANES:
            seen.update(lane_tkeys[L])
        return sorted(seen)

    _boundaries_cache = {"valid": False, "list": []}

    def _streak_ok(lane, tkey):
        """Reject lane if it would create a > max_jack_streak run of consecutive
        chord-boundary uses around `tkey`. Mirrors _place_measure_constrained."""
        if max_jack_streak <= 0:
            return True
        if not _boundaries_cache["valid"]:
            _boundaries_cache["list"] = _all_chord_boundaries()
            _boundaries_cache["valid"] = True
        boundaries = _boundaries_cache["list"]
        import bisect
        pos = bisect.bisect_left(boundaries, tkey)
        # Recent K boundaries strictly before tkey
        prev_K = boundaries[max(0, pos - max_jack_streak):pos]
        if len(prev_K) >= max_jack_streak:
            lane_tks = set(lane_tkeys[lane])
            if all(b in lane_tks for b in prev_K):
                return False  # would extend streak past cap
        # Also check forward (we are inserting between events): if K boundaries
        # immediately after tkey all use this lane, inserting here makes K+1.
        next_K = boundaries[pos:pos + max_jack_streak]
        if len(next_K) >= max_jack_streak:
            lane_tks = set(lane_tkeys[lane])
            if all(b in lane_tks for b in next_K):
                return False
        return True

    def _invalidate_boundaries_cache():
        _boundaries_cache["valid"] = False

    # Build per-measure placed count for cap enforcement (v10 §11.3)
    from collections import Counter as _Counter
    measure_placed = _Counter()
    for ev in placed_events:
        if ev.get("type", "Tap") == "Tap" and ev.get("lane", "") != "P1_SCR":
            measure_placed[ev["measure"]] += 1
    measure_cap = measure_note_cap

    # ── Fill low-density segments ─────────────────────────────────────────
    for i, (start, end) in enumerate(segments):
        if densities[i] >= hard_target:
            continue
        current_count = sum(1 for ev in placed_events
                            if ev.get("type", "Tap") == "Tap"
                            and ev.get("lane", "") != "P1_SCR"
                            and start <= ev.get("measure", -1) < end)
        target_count = round(hard_target * (end - start))
        to_add = target_count - current_count
        if to_add <= 0:
            continue

        seg_residuals = [
            ev for ev in residual_events
            if start <= ev["measure"] < end
            and ev.get("reason") in ("density_rebalance", "no_lane_available",
                                     "jack_violation", "hand_balance",
                                     "collision", "primitive_failed")
        ]
        if ml_fillback_used:
            # 2026-05-03: ML token-model ranking for fill-back. RB selects WHICH
            # measures are low-density; ML ranks WHICH residual token to pull
            # back into placement first. Score is fetched per measure (cached
            # across the segment) and falls back to pct ordering when the model
            # cannot return a usable vector. Fill-back model failures are
            # tracked in `ml_fillback_fallback_count` and intentionally do NOT
            # contribute to `state["token_fallback_count"]` (the main per-measure
            # counter); we save-restore to keep the two responsibilities split.
            score_cache = {}
            pi = ml_ctx["pool_index"]
            state = ml_ctx["state"]
            def _ml_fillback_key(ev):
                m = ev["measure"]
                if m not in score_cache:
                    nonlocal ml_fillback_score_calls, ml_fillback_fallback_count
                    phase_label = _get_phase(m, phase_blocks)
                    n_for_m = sum(1 for r in seg_residuals if r["measure"] == m)
                    saved_fb = state["token_fallback_count"]
                    score_cache[m] = _ml_score_tokens(
                        ml_ctx, m, phase_label, max(1, n_for_m))
                    if state["token_fallback_count"] > saved_fb:
                        # Inference failure — restore main-loop counter and
                        # bump the fill-back-local counter.
                        ml_fillback_fallback_count += 1
                        state["token_fallback_count"] = saved_fb
                    ml_fillback_score_calls += 1
                scores = score_cache[m]
                if scores is None:
                    return -pct_map.get(ev["token"], 0.0)
                idx = pi.get(ev["token"], 0)
                return -float(scores[idx])
            seg_residuals.sort(key=_ml_fillback_key)
        else:
            seg_residuals.sort(key=lambda ev: -pct_map.get(ev["token"], 0.0))

        added = 0
        for ev in seg_residuals:
            if added >= to_add:
                break
            m, idx, tok = ev["measure"], ev["idx192"], ev["token"]
            tkey = m * 192 + idx

            # Respect per-measure note cap (v10 §11.3)
            if measure_placed[m] >= measure_cap:
                continue

            if tok in token_at_pos[(m, idx)]:
                continue

            # 2026-05-03: respect MAX_CHORD_SIZE cap during fill-back too.
            # Without this, residuals (e.g. no_lane_available) would get
            # re-placed onto positions whose chord already hit the cap from
            # _place_measure_constrained, growing chords past the limit.
            cur_chord_size = sum(1 for lane in KEY_LANES
                                 if (m, idx, lane) in occupied)
            if cur_chord_size >= max_chord_size:
                continue

            # 2026-05-03: also apply chord-mate spread preference here.
            # Without this, fill-back creates adjacent-lane chords (e.g.
            # {1,2,3}) that the constrained placer rejected for spread.
            existing_at_tkey_idx = [LANE_INDEX[lane] - 1
                                    for lane in KEY_LANES
                                    if (m, idx, lane) in occupied]
            assigned_lane = None
            # 2026-05-03: shuffle iteration order so K1 isn't always tried first
            # (the previous fixed-order loop concentrated fill-back additions
            # on K1 and produced 47-streak runs on dense charts like mightyA).
            shuffled_lanes = (fisher_yates_shuffle(KEY_LANES, rng)
                              if rng is not None else list(KEY_LANES))
            # Pass 1: prefer lanes ≥2 apart from existing chord-mates AND
            # respect per-lane jack_streak cap (mirrors _place_measure_constrained).
            for lane in shuffled_lanes:
                if (m, idx, lane) in occupied:
                    continue
                if not _jack_ok(lane, tkey):
                    continue
                if not _streak_ok(lane, tkey):
                    continue
                lane_idx = LANE_INDEX[lane] - 1
                if existing_at_tkey_idx and any(
                    abs(lane_idx - ei) < 2 for ei in existing_at_tkey_idx
                ):
                    continue
                assigned_lane = lane
                break
            # Pass 2: drop spread-2 preference, keep streak/jack guards.
            if assigned_lane is None:
                for lane in shuffled_lanes:
                    if (m, idx, lane) in occupied:
                        continue
                    if not _jack_ok(lane, tkey):
                        continue
                    if not _streak_ok(lane, tkey):
                        continue
                    assigned_lane = lane
                    break
            # Pass 3: last resort — drop streak guard. Better to violate the
            # soft constraint than leave the residual unplaced (matches main
            # placement loop's soft-fallback semantics).
            if assigned_lane is None:
                for lane in shuffled_lanes:
                    if (m, idx, lane) in occupied:
                        continue
                    if not _jack_ok(lane, tkey):
                        continue
                    assigned_lane = lane
                    break

            if assigned_lane is None:
                continue

            new_ev = {
                "type": "Tap", "token": tok, "lane": assigned_lane,
                "measure": m, "idx192": idx,
                "primitive": "Stream", "phase": _get_phase(m, phase_blocks),
            }
            placed_events.append(new_ev)
            residual_events.remove(ev)
            occupied.add((m, idx, assigned_lane))
            token_at_pos[(m, idx)].add(tok)

            import bisect
            bisect.insort(lane_tkeys[assigned_lane], tkey)
            _invalidate_boundaries_cache()
            notes_added[i] += 1
            added += 1
            measure_placed[m] += 1

    diag = {
        "rebalancing_skipped": False,
        "delta_ratio_before": round(delta_ratio, 4),
        "segment_densities": [round(d, 2) for d in densities],
        "notes_removed_per_segment": notes_removed,
        "notes_added_per_segment": notes_added,
        "ml_fillback_used": ml_fillback_used,
        "ml_fillback_score_calls": ml_fillback_score_calls,
        "ml_fillback_fallback_count": ml_fillback_fallback_count,
    }
    return placed_events, residual_events, diag


# ── Conformance checks ───────────────────────────────────────────────────────

def _run_conformance(result_data, pool_events, excluded, ta, pct_map, fx_info,
                     whitelist, key_occ, scratch_occ, bgm_occ, phase_blocks,
                     scratch_seeds, events, intensity_origin_map, measure_max, params=None,
                     ml_active=False, scratch_mode="primary",
                     source_scratch_per_measure=None, scratch_scale=1.0,
                     source_min_interval=None, measure_scale=None,
                     is_finalize=False):
    p = params or _default_params()
    placed = result_data["placed"]
    checks = {}

    # Check A (v10): placed event's token must be in whitelist OR rescued for THIS measure.
    # Measure-local rescue check ensures a token rescued only in measure M cannot "leak"
    # into an unrescued measure N.
    rescued_pairs = result_data.get("_rescued_pairs", set())
    violations_a = []
    for ev in placed:
        if ev.get("lane") == "P1_SCR": continue
        tok = ev["token"]
        if tok in whitelist: continue
        m = ev.get("measure_start", ev.get("measure"))
        if (m, tok) in rescued_pairs: continue  # §5.6 rescue for this measure
        violations_a.append(tok)
    checks["A_whitelist_hard_filters"] = "PASS" if not violations_a else f"FAIL ({len(violations_a)} violations)"

    # Check B
    pool_positions = set((m, i) for (m, i, t) in pool_events)
    bad_b = []
    for ev in placed:
        if ev.get("type", "Tap") == "LN":
            if (ev["measure_start"], ev["idx192_start"]) not in pool_positions: bad_b.append(ev)
        else:
            if (ev["measure"], ev["idx192"]) not in pool_positions: bad_b.append(ev)
    checks["B_timing_preservation"] = "PASS" if not bad_b else f"FAIL ({len(bad_b)})"

    # Check C — jack constraint applies to KEY lanes only. Scratch lane has its
    # own interval check (Check F) and its physical motion (wrist) is independent
    # from finger jacks. Including P1_SCR here would flag any scratch denser than
    # 12 ticks, which is normal for scratch:r sources at 150bpm+.
    by_lane = defaultdict(list)
    for ev in placed:
        if ev.get("lane") == "P1_SCR":
            continue
        if ev.get("type", "Tap") == "LN":
            by_lane[ev["lane"]].append(ev["measure_start"] * 192 + ev["idx192_start"])
        else:
            by_lane[ev["lane"]].append(ev["measure"] * 192 + ev["idx192"])
    # 2026-06-11: Check C measures gaps on the beat-true axis, matching the
    # scale-aware placement-time jack constraint.
    _scaled_c = build_scaled_tick_fn(measure_scale, measure_max)
    jack_violations = 0
    for lane, tkeys in by_lane.items():
        tkeys.sort()
        for i in range(1, len(tkeys)):
            if round(_scaled_c(tkeys[i]) - _scaled_c(tkeys[i - 1]), 6) <= 12: jack_violations += 1
    checks["C_jack_prohibition"] = "PASS" if jack_violations == 0 else f"FAIL ({jack_violations})"

    # Check D
    bad_d = [ev for ev in placed if ev["primitive"] == "ChordBurst" and ev["phase"] in ("normal", "rest")]
    scratch_mode = result_data["diagnostics"]["scratch_mode"]
    d_ok = len(bad_d) == 0 and (scratch_seeds or scratch_mode == "disabled")
    if not scratch_seeds and scratch_mode != "disabled": d_ok = False
    checks["D_fallback_behavior"] = "PASS" if d_ok else f"FAIL"

    # Check E
    tok_pos_lanes = defaultdict(set)
    for ev in placed:
        if ev.get("type", "Tap") == "LN":
            tok_pos_lanes[(ev["token"], ev["measure_start"], ev["idx192_start"])].add(ev["lane"])
        else:
            tok_pos_lanes[(ev["token"], ev["measure"], ev["idx192"])].add(ev["lane"])
    bad_e = sum(1 for lanes in tok_pos_lanes.values() if len(lanes) > 1)
    checks["E_candidate_collision"] = "PASS" if bad_e == 0 else f"FAIL ({bad_e})"

    # Check F
    scr_max = p.get("SCRATCH_MAX_PER_MEASURE", SCRATCH_MAX_PER_MEASURE)
    scr_min_interval = p.get("SCRATCH_MIN_INTERVAL", SCRATCH_MIN_INTERVAL)
    scr_notes = [ev for ev in placed if ev.get("lane") == "P1_SCR"]
    scr_by_measure = defaultdict(int)
    scr_tkeys = []
    for ev in scr_notes:
        m = ev.get("measure_start", ev.get("measure"))
        idx = ev.get("idx192_start", ev.get("idx192"))
        scr_by_measure[m] += 1; scr_tkeys.append(m * 192 + idx)
    scr_tkeys.sort()
    # In primary mode the active min_interval is source-derived (with floor).
    effective_min_interval = (source_min_interval
                              if scratch_mode == "primary" and source_min_interval is not None
                              else scr_min_interval)
    f_interval = sum(1 for i in range(1, len(scr_tkeys)) if scr_tkeys[i] - scr_tkeys[i - 1] < effective_min_interval)
    # Density check: in primary mode, budget is per-measure source-derived;
    # in fallback mode, single scr_max applies to all.
    f_density = 0
    # §12 REVISED 2026-06-13: per-measure allowance = tier-1 mirror budget +
    # tier-2 supplement (emitted in diagnostics, str-keyed after JSON round-trip).
    # The supplement is bounded so budget+supplement never exceeds the
    # level-lerped cap (§12.9 headroom). In finalize mode the engine did not
    # run placement, so the supplement map is unavailable (externally spliced
    # events); fall back to the cap as the allowance — every legitimately
    # placed scratch is <= cap by construction (Codex audit 2026-06-13).
    _suppl = result_data["diagnostics"].get("scratch_supplement_by_measure", {})
    if scratch_mode == "primary" and source_scratch_per_measure is not None:
        for m, cnt in scr_by_measure.items():
            budget = max(0, round(source_scratch_per_measure.get(m, 0) * scratch_scale))
            if is_finalize:
                # Supplement map is unavailable (engine didn't run placement).
                # Reconstruct the per-measure ceiling from the §12.9 rule:
                # tier-1 fills up to `budget` (source mirror, cap-exempt — a
                # source-scratch-heavy measure legitimately exceeds the cap),
                # and tier-2 can add up to the cap when budget < cap. So the
                # standing allowance is max(budget, cap).
                allowance = max(budget, scr_max)
            else:
                allowance = budget + _suppl.get(str(m), _suppl.get(m, 0))
            if cnt > allowance:
                f_density += 1
    else:
        f_density = sum(1 for c in scr_by_measure.values() if c > scr_max)
    activations = result_data["diagnostics"]["scratch_rush_rest_activations"]
    f_rush_ok = True
    for act in activations:
        for m in range(act["suppressed_range"][0], act["suppressed_range"][1] + 1):
            if scr_by_measure.get(m, 0) > 0: f_rush_ok = False
    f_ok = f_interval == 0 and f_density == 0 and f_rush_ok
    checks["F_scratch_constraints"] = "PASS" if f_ok else f"FAIL (interval:{f_interval}, density:{f_density}, rush:{'OK' if f_rush_ok else 'BAD'})"

    # Check G — compare pre-LN placed events for reproducibility
    seed_used = result_data.get("_seed", PLACEMENT_RANDOM_SEED)
    if ml_active:
        checks["G_seeded_reproducibility"] = "SKIPPED (ml active)"
    elif seed_used != PLACEMENT_RANDOM_SEED:
        checks["G_seeded_reproducibility"] = f"SKIPPED (custom seed={seed_used})"
    else:
        placed_for_g = result_data.get("placed_pre_ln", placed)
        _bpm_events_g = sorted(
            [(ev["measure"] * 192 + ev["idx192"], ev["bpm"])
             for ev in events if ev["type"] == "BPMChange"],
            key=lambda x: x[0])
        _base_bpm_g = result_data.get("_base_bpm", 130.0)
        result2 = run_per_measure_loop(events, whitelist, pct_map, intensity_origin_map,
                                       phase_blocks, measure_max, scratch_seeds, params,
                                       excluded=excluded, ta=ta, key_occ=key_occ,
                                       scratch_occ=scratch_occ, bgm_occ=bgm_occ,
                                       seed=seed_used,
                                       bpm_events=_bpm_events_g, base_bpm=_base_bpm_g,
                                       scratch_mode=scratch_mode,
                                       source_scratch_per_measure=source_scratch_per_measure,
                                       scratch_scale=scratch_scale,
                                       source_min_interval=source_min_interval,
                                       measure_scale=measure_scale)
        placed2 = result2["placed_events"]
        g_ok = len(placed_for_g) == len(placed2) and all(a == b for a, b in zip(placed_for_g, placed2))
        checks["G_seeded_reproducibility"] = "PASS" if g_ok else f"FAIL (len1={len(placed_for_g)}, len2={len(placed2)})"

    # Check K (v10): MEASURE_NOTE_CAP enforcement
    # Residual reason "measure_cap" is valid even after post-processing (density_rebalance
    # can remove placed notes, dropping measure below cap). Only verify that no measure
    # exceeds cap in the final placed set.
    cap = p.get("MEASURE_NOTE_CAP", 999)
    from collections import Counter as _Counter
    placed_per_m = _Counter()
    for ev in placed:
        if ev.get("lane") == "P1_SCR": continue
        if ev.get("type", "Tap") == "LN":
            placed_per_m[ev["measure_start"]] += 1
        else:
            placed_per_m[ev["measure"]] += 1
    over = [(m, c) for m, c in placed_per_m.items() if c > cap]
    checks["K_measure_cap"] = "PASS" if not over else f"FAIL ({len(over)} measures over cap={cap})"

    return checks


# ── Main ───────────────────────────────────────────────────────────────────────

def main(intensity_level=5, scratch_level=5, enable_ln=False,
         enable_ml=False, model_token_path=None, model_lane_path=None,
         seed=PLACEMENT_RANDOM_SEED,
         resume_state=None, start_measure=None, end_measure=None,
         finalize_input_events=None,
         next_chord_lookahead=None, dp=False, dp_split="auto"):
    if dp and (resume_state is not None or finalize_input_events is not None):
        raise NotImplementedError(
            "DP synthesis (--dp) does not support resume/finalize mode yet (addon §5).")
    effective_params = {**compute_intensity_params(intensity_level),
                        **compute_scratch_params(scratch_level)}
    # Resolve "random" seed to an actual integer for reproducibility logging
    import os as _os
    if seed == "random" or seed is None:
        seed = int.from_bytes(_os.urandom(4), "big")
    ml_warnings: list = []
    ml_state = _make_ml_state(enable_ml, model_token_path, model_lane_path, ml_warnings)
    for w in ml_warnings:
        print(f"WARNING: {w}")

    print(f"Effective params (intensity={intensity_level}, scratch={scratch_level}, ln={enable_ln}):")
    for k, v in sorted(effective_params.items()):
        print(f"  {k} = {v}")

    bms_bytes = load_bms_bytes()
    parsed = bmsparser.parse_bms(bms_bytes)
    events = parsed["events"]
    measure_scale = parsed.get("measure_scale", {})
    _scaled_measures = _normalize_measure_scale(measure_scale)
    if _scaled_measures:
        print(f"  Measure scale: {len(_scaled_measures)} non-1.0 measure(s) "
              f"(scale-aware time axis active)")

    pool_tokens, key_occ, scratch_occ, bgm_occ, pool_events, measure_max = build_pool_universe(events)

    with open(TOKEN_ANALYSIS, "r", encoding="utf-8") as f:
        ta = {e["token"]: e for e in json.load(f)}

    pct_map = compute_attack_percentile(pool_tokens, ta)
    intensity_origin = compute_intensity_origin(pool_tokens, key_occ, scratch_occ)
    fx_info = classify_fx(pool_tokens, ta, pct_map, intensity_origin)
    whitelist, excluded, exc_counts = build_whitelist(
        pool_tokens, ta, key_occ, scratch_occ, bgm_occ, pct_map, fx_info, effective_params)
    phase_blocks = segment_phases(pool_events, measure_max, effective_params)

    scratch_seeds, scratch_mode = _determine_scratch_seeds(
        pool_tokens, key_occ, scratch_occ, bgm_occ, whitelist, ta, pct_map, fx_info)

    # 2026-05-03: source-aware scratch density
    source_scratch_per_measure = _compute_source_scratch_per_measure(events)
    source_scratch_min_interval = _compute_source_min_scratch_interval(events)
    scratch_scale = scratch_level / 5.0  # lv5 = 1:1 source mirror
    src_scr_total = sum(source_scratch_per_measure.values())
    src_scr_pct = (src_scr_total / max(1, sum(key_occ.values()) + src_scr_total)) * 100.0
    print(f"  Scratch: mode={scratch_mode}, scale={scratch_scale:.2f}x "
          f"(level={scratch_level}/5), src_total={src_scr_total} ({src_scr_pct:.1f}%), "
          f"src_min_interval={source_scratch_min_interval}")

    ml_ctx = None
    if ml_state["enable_token"] or ml_state["enable_lane"]:
        ml_ctx = _build_ml_context(
            ml_state, pool_tokens, ta, whitelist, key_occ, scratch_occ, bgm_occ,
            intensity_origin, pct_map, intensity_level)

    # v10 §11.5: BPM timeline for jack floor
    bpm_events_timeline = sorted(
        [(ev["measure"] * 192 + ev["idx192"], ev["bpm"])
         for ev in events if ev["type"] == "BPMChange"],
        key=lambda x: x[0])
    base_bpm = parsed.get("base_bpm", 130.0)

    if finalize_input_events is None:
        # ── Normal path (also covers resume mode via run_per_measure_loop) ──
        # v12 §23.7 / DR-23-8: lookahead only meaningful in resume mode and
        # only applied to the final measure of the loop range.
        if next_chord_lookahead is not None and resume_state is None:
            raise ValueError(
                "next_chord_lookahead requires resume mode (resume_state must be supplied)")
        if next_chord_lookahead is not None and (
                ml_ctx is not None
                and (ml_ctx["state"].get("enable_token")
                     or ml_ctx["state"].get("enable_lane"))):
            raise NotImplementedError(
                "next_chord_lookahead + --ml is not supported (§23.7, RB-only v1).")
        result = run_per_measure_loop(
            events, whitelist, pct_map, intensity_origin, phase_blocks, measure_max,
            scratch_seeds, effective_params, ml_ctx=ml_ctx,
            excluded=excluded, ta=ta, key_occ=key_occ, scratch_occ=scratch_occ,
            bgm_occ=bgm_occ, seed=seed,
            bpm_events=bpm_events_timeline, base_bpm=base_bpm,
            scratch_mode=scratch_mode,
            source_scratch_per_measure=source_scratch_per_measure,
            scratch_scale=scratch_scale,
            source_min_interval=source_scratch_min_interval,
            resume_state=resume_state,
            start_measure=start_measure,
            end_measure=end_measure,
            next_chord_lookahead=next_chord_lookahead,
            measure_scale=measure_scale,
            dp=dp, dp_split=dp_split)
        # v12 §23.5: resume mode short-circuits with partial schema, no
        # post-processing — caller (BMS.Compare) splices and invokes finalize.
        if resume_state is not None:
            partial_output = {
                "schema_version": "placement-result-v1",
                "mode": "resume",
                "start_measure": result["start_measure"],
                "end_measure": result["end_measure"],
                "events": result["placed_events"],
                "residuals": result["residual_events"],
                "end_state": result["end_state"],
                "diagnostics": {
                    "centroid_step_unit": result.get("centroid_step_unit"),
                    "downgrade_count": result["downgrade_count"],
                    "residual_by_measure": dict(result.get("residual_by_measure", {})),
                },
            }
            with open(RESULT_PATH, "w", encoding="utf-8") as f:
                json.dump(partial_output, f, ensure_ascii=False, indent=2)
            print(f"placement_result.json (resume mode) written: "
                  f"measures {partial_output['start_measure']}..{partial_output['end_measure']}, "
                  f"{len(partial_output['events'])} events")
            return
        # DP step 1: SP-specific post-processing (LN / scratch adjustment /
        # density rebalance / conformance) assumes P1-only lanes. DP variants are
        # later steps, so DP short-circuits with the raw side-local placement.
        if dp:
            _l, _r = result["dp_placed_left"], result["dp_placed_right"]
            _sl, _sr = result["dp_scratch_left"], result["dp_scratch_right"]
            # audio preservation (timing invariant): whitelist-excluded source
            # tokens must still play as BGM (#01), exactly as the SP path builds
            # wl_residuals. Without this they're neither placed nor residual →
            # silent, breaking the source's content (DP early-return omitted it).
            _scr_keys = result.get("scratch_placed_keys", set())
            _placed_keys = {(e["measure"], e["idx192"], e["token"])
                            for e in result["placed_events"]}
            _wl_resid, _wl_seen = [], set()
            for (_m, _i, _tok) in pool_events:
                if _tok in excluded:
                    _k = (_m, _i, _tok)
                    if _k in _scr_keys or _k in _placed_keys or _k in _wl_seen:
                        continue
                    _wl_seen.add(_k)
                    _wl_resid.append({"token": _tok, "measure": _m, "idx192": _i,
                                      "reason": excluded[_tok]})
            dp_residual = _wl_resid + result["residual_events"]
            dp_output = {
                "placed": result["placed_events"],
                "residual": dp_residual,
                "diagnostics": {
                    "dp_enabled": True,
                    "dp_placed_left": _l,
                    "dp_placed_right": _r,
                    "dp_left_share": round(_l / max(1, _l + _r), 4),
                    "dp_scratch_left": _sl,
                    "dp_scratch_right": _sr,
                    "dp_scratch_total": _sl + _sr,
                    "dp_scratch_gated": result["dp_scratch_gated"],
                    "dp_scratch_measures": result["dp_scratch_measures"],
                    "dp_blocks_swapped": result["dp_blocks_swapped"],
                    "dp_rescued": result["dp_rescued"],
                    "dp_measure_rebalanced": result["dp_measure_rebalanced"],
                    "dp_split_mode": result["dp_split_mode"],
                    "total_placed": result["total_placed"],
                    "residual_total": len(dp_residual),
                    "wl_residual_count": len(_wl_resid),
                    "residual_reasons": dict(result.get("residual_counts", {})),
                    "centroid_step_unit": result.get("centroid_step_unit"),
                    "seed": seed,
                },
                "ln_meta": {"enabled": False},
            }
            with open(RESULT_PATH, "w", encoding="utf-8") as f:
                json.dump(dp_output, f, ensure_ascii=False, indent=2)
            print(f"placement_result.json (DP synthesis, split={result['dp_split_mode']}) written: "
                  f"{len(result['placed_events'])} placed "
                  f"(key L={_l}/R={_r} L-share={dp_output['diagnostics']['dp_left_share']}, "
                  f"swapped_blocks={result['dp_blocks_swapped']}, "
                  f"rescued={result['dp_rescued']}, "
                  f"rebalanced={result['dp_measure_rebalanced']}, "
                  f"scr L={_sl}/R={_sr} over {result['dp_scratch_measures']} scr-measures "
                  f"gated={result['dp_scratch_gated']}), "
                  f"{len(dp_residual)} residual ({len(_wl_resid)} wl→BGM)")
            return
    else:
        # ── v12 §23.6 finalize mode: skip placement loop, run post-processing
        # only over caller-supplied events (BMS.Compare splices reroll outputs).
        # v1 = RB-only (§23.8); ML + finalize is rejected to keep contracts clean.
        if ml_state["enable_token"] or ml_state["enable_lane"]:
            raise NotImplementedError(
                "Resume API v1 does not support --finalize with --ml (§23.8). "
                "ML carry-over context is out of scope.")
        scratch_placed_keys = set()
        for ev in finalize_input_events:
            if ev.get("type", "Tap") == "Tap" and ev.get("lane") == "P1_SCR":
                scratch_placed_keys.add((ev["measure"], ev["idx192"], ev["token"]))
        result = {
            "placed_events": list(finalize_input_events),
            "residual_events": [],
            "scratch_placed_keys": scratch_placed_keys,
            "rescued_tokens": set(),
            "rescued_pairs": set(),
            "centroid_step_unit": None,
            "chb_count": 0, "stream_count": 0, "downgrade_count": 0,
            "total_placed": len(finalize_input_events),
            "scratch_inserted": len(scratch_placed_keys),
            "residual_counts": {},
            "residual_by_measure": {},
            "hand_balance_avoided_count": 0,
            "same_hand_alternation_count": 0,
            "lane_affinity_applied_count": 0,
            "density_ramp_warnings": [],
            "lane_continuity_resets": [],
            "scratch_rush_rest_activations": [],
        }

    # Save pre-LN events for conformance Check G
    placed_pre_ln = list(result["placed_events"])

    # LN post-processing
    ln_meta = {"enabled": False}
    if enable_ln:
        bpm_events = sorted(
            [(ev["measure"] * 192 + ev["idx192"], ev["bpm"])
             for ev in events if ev["type"] == "BPMChange"],
            key=lambda x: x[0])
        headers = parsed["headers"]
        declared_wav = {k[3:].upper() for k in headers if k.startswith("WAV") and len(k) > 3}
        lnobj_token = select_lnobj_token(headers, declared_wav)

        ln_result = run_ln_postprocess(
            result["placed_events"], ta, bpm_events, parsed.get("measure_scale", {}),
            parsed.get("base_bpm", 130.0), LN_MIN_DURATION_MS, LN_MAX_RATIO,
            LN_MAX_HOLD_TICKS)
        result["placed_events"] = ln_result["placed_events"]
        ln_meta = ln_result["ln_meta"]
        ln_meta["lnobj_token"] = lnobj_token

    # Build whitelist-excluded residual events (event-level, not token-level)
    # An excluded token's (m,i,token) event is a residual UNLESS actually placed at that position.
    scratch_placed_keys = result.get("scratch_placed_keys", set())
    placed_keys = set()
    for e in result["placed_events"]:
        if e.get("type", "Tap") == "LN":
            placed_keys.add((e["measure_start"], e["idx192_start"], e["token"]))
        else:
            placed_keys.add((e["measure"], e["idx192"], e["token"]))
    wl_residuals, wl_seen = [], set()
    for (m, i, tok) in pool_events:
        if tok in excluded:
            key = (m, i, tok)
            if key in scratch_placed_keys: continue
            if key in placed_keys: continue  # rescued & actually placed → not residual
            if key not in wl_seen:
                wl_seen.add(key)
                reason = excluded[tok]  # fx | unknown | band_quota | occurrence
                wl_residuals.append({"token": tok, "measure": m, "idx192": i, "reason": reason})
    all_residuals = wl_residuals + result["residual_events"]

    # §13.9 Scratch adjustment (post-LN)
    scr_max = effective_params.get("SCRATCH_MAX_PER_MEASURE", SCRATCH_MAX_PER_MEASURE)
    result["placed_events"], all_residuals, scratch_adj_diag = run_scratch_adjustment(
        result["placed_events"], all_residuals, ln_meta, pct_map, scr_max)

    # §13.10 Density rebalancing
    result["placed_events"], all_residuals, rebal_diag = run_density_rebalance(
        result["placed_events"], all_residuals, measure_max,
        pct_map, effective_params.get("DENSITY_REBALANCE_MAX_DELTA", DENSITY_REBALANCE_MAX_DELTA),
        phase_blocks,
        measure_note_cap=effective_params.get("MEASURE_NOTE_CAP", 999),
        max_chord_size=effective_params.get("MAX_CHORD_SIZE", 7),
        ml_ctx=ml_ctx,
        max_jack_streak=effective_params.get("MAX_JACK_STREAK", 3),
        rng=random.Random(seed),  # main() resolves 'random' to int before reaching here
        measure_scale=measure_scale)

    # 2026-05-03: post-placement distribution metrics for stability/dispersion analysis
    placement_distribution = compute_placement_distribution_metrics(result["placed_events"])

    # FX classification counts
    fx_dur = fx_atk = fx_ori = 0
    for token in pool_tokens:
        info = ta.get(token)
        if not info or not info.get("decode_ok"): continue
        if info["duration_ms"] > FX_DURATION_THRESHOLD: fx_dur += 1
        if pct_map.get(token, 0.0) <= FX_ATTACK_THRESHOLD: fx_atk += 1
        if intensity_origin.get(token, 0) == 0 and FX_ORIGIN_FILTER_ENABLED: fx_ori += 1

    # Coverage by phase
    pool_by_phase = defaultdict(int)
    for (m, i, t) in pool_events: pool_by_phase[_get_phase(m, phase_blocks)] += 1
    placed_by_phase = defaultdict(int)
    for ev in result["placed_events"]: placed_by_phase[ev["phase"]] += 1
    coverage_by_phase = {}
    for ph in ("rush", "normal", "rest"):
        total = pool_by_phase.get(ph, 0)
        coverage_by_phase[ph] = {"target": 0.7, "achieved": round(placed_by_phase.get(ph, 0) / total, 4) if total else 0.0}

    # ── v10 extended diagnostics ──────────────────────────────────────────
    # Band distribution (eligible pool vs whitelist) per §5.3
    # "pool" here means eligible tokens (hard filter fx/unknown excluded),
    # matching the band classification input used in build_whitelist.
    def _compute_band_stats():
        centroids_all = {}
        for tok in pool_tokens:
            if excluded.get(tok) in ("fx", "unknown"):
                continue
            info = ta.get(tok)
            if info and info.get("decode_ok") and info.get("spectral_centroid_mean", 0) > 0:
                centroids_all[tok] = info["spectral_centroid_mean"]
        if len(centroids_all) < 6:
            # Not enough data for meaningful banding; emit zeroed dict to keep schema stable
            return {"lo_thr_hz": None, "hi_thr_hz": None,
                    "pool": {"lo": 0, "mid": 0, "hi": 0},
                    "whitelist": {"lo": 0, "mid": 0, "hi": 0},
                    "skipped": "insufficient_pool"}
        vals = sorted(centroids_all.values())
        lo_thr = vals[len(vals) // 3]
        hi_thr = vals[2 * len(vals) // 3]
        def _b(c):
            if c < lo_thr: return "lo"
            if c < hi_thr: return "mid"
            return "hi"
        pool_b = {"lo": 0, "mid": 0, "hi": 0}
        wl_b = {"lo": 0, "mid": 0, "hi": 0}
        for tok, c in centroids_all.items():
            pool_b[_b(c)] += 1
            if tok in whitelist:
                wl_b[_b(c)] += 1
        return {"lo_thr_hz": round(lo_thr, 1), "hi_thr_hz": round(hi_thr, 1),
                "pool": pool_b, "whitelist": wl_b}
    band_stats = _compute_band_stats()

    # Rescue / measure_cap stats
    rescued_count = len(result.get("rescued_tokens", set()))
    measure_cap_count = sum(1 for r in all_residuals if r.get("reason") == "measure_cap")

    # Diagnostics
    diagnostics = {
        "effective_params": effective_params,
        "whitelist_total": len(pool_tokens),
        "whitelist_passed": len(whitelist),
        "whitelist_excluded_by_reason": exc_counts,
        "spectral_bands": band_stats,
        "windowed_rescue": {
            "rescued_token_count": rescued_count,
            "window_size": 8,
            "threshold": 0.40,
        },
        "measure_cap_activations": measure_cap_count,
        "seed": seed,
        "centroid_step_unit": result.get("centroid_step_unit"),
        "fx_classification_counts": {"duration": fx_dur, "attack": fx_atk, "origin": fx_ori},
        "phase_blocks": [[b["start"], b["end"], b["phase"], round(b["smoothed_score"], 2)] for b in phase_blocks],
        "coverage_by_phase": coverage_by_phase,
        "chordburst_downgrade_count": result["downgrade_count"],
        "residual_by_measure": result["residual_by_measure"],
        "residual_reasons": dict(result.get("residual_counts", {})),
        "hand_balance_avoided_count": result["hand_balance_avoided_count"],
        "same_hand_alternation_count": result["same_hand_alternation_count"],
        "lane_affinity_applied_count": result.get("lane_affinity_applied_count", 0),
        "dp_enabled": result.get("dp_enabled", False),
        "dp_placed_left": result.get("dp_placed_left", 0),
        "dp_placed_right": result.get("dp_placed_right", 0),
        "placement_distribution": placement_distribution,
        "scratch_mode": scratch_mode,
        "scratch_scale": scratch_scale,
        "scratch_tier2_inserted": result.get("scratch_tier2_inserted", 0),
        "scratch_supplement_by_measure": {
            str(k): v for k, v in result.get("scratch_supplement_by_measure", {}).items()},
        "scaled_measure_count": len(_scaled_measures),
        "source_scratch_total": sum(source_scratch_per_measure.values()),
        "source_scratch_pct": src_scr_pct,
        "scratch_rush_rest_activations": result["scratch_rush_rest_activations"],
        "residual_total": len(all_residuals),
        "density_ramp_warnings": result["density_ramp_warnings"],
        "lane_continuity_resets": result["lane_continuity_resets"],
        "scratch_adjustment": scratch_adj_diag,
        "density_rebalance": rebal_diag,
        "ml": (
            {
                "token_model_enabled": ml_state["enable_token"],
                "token_model_path": ml_state["token_path"],
                "token_model_load_ok": ml_state["token_load_ok"],
                "token_fallback_count": ml_state["token_fallback_count"],
                "lane_model_enabled": ml_state["enable_lane"],
                "lane_model_path": ml_state["lane_path"],
                "lane_model_load_ok": ml_state["lane_load_ok"],
                "lane_fallback_count": ml_state["lane_fallback_count"],
            }
            if enable_ml
            else {"token_model_enabled": False, "lane_model_enabled": False}
        ),
    }

    output = {
        "placed": result["placed_events"],
        "residual": all_residuals,
        "diagnostics": diagnostics,
        "ln_meta": ln_meta,
        "placed_pre_ln": placed_pre_ln,
    }
    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"placement_result.json written ({len(result['placed_events'])} placed, {len(all_residuals)} residual)")

    # Conformance checks
    output["_rescued_tokens"] = result.get("rescued_tokens", set())
    output["_rescued_pairs"] = result.get("rescued_pairs", set())
    output["_seed"] = seed
    output["_base_bpm"] = base_bpm
    checks = _run_conformance(output, pool_events, excluded, ta, pct_map, fx_info,
                              whitelist, key_occ, scratch_occ, bgm_occ, phase_blocks,
                              scratch_seeds, events, intensity_origin, measure_max, effective_params,
                              ml_active=(ml_ctx is not None),
                              scratch_mode=scratch_mode,
                              source_scratch_per_measure=source_scratch_per_measure,
                              scratch_scale=scratch_scale,
                              source_min_interval=source_scratch_min_interval,
                              measure_scale=measure_scale,
                              is_finalize=(finalize_input_events is not None))
    print()
    for name, status in checks.items():
        print(f"  {name}: {status}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="BMS Placement Engine")
    ap.add_argument("--intensity", type=int, default=5, help="Note placement aggressiveness 1~20 (default: 5)")
    ap.add_argument("--scratch", type=int, default=5, help="Scratch frequency 1~20 (default: 5)")
    ap.add_argument("--ln", action="store_true", default=False, help="Enable LN post-processing")
    ap.add_argument("--ml", action="store_true", default=False, help="Enable ML soft-ranking integration")
    ap.add_argument("--model-token", default=None, help="TokenSelectionModel TorchScript path")
    ap.add_argument("--model-lane", default=None, help="LaneAssignmentModel TorchScript path")
    ap.add_argument("--seed", default=None,
                    help=f"Placement seed: integer or 'random' (default: {PLACEMENT_RANDOM_SEED}; "
                         "in resume mode the state's seed takes precedence — passing --seed there errors)")
    # v12 §23 Resume API
    ap.add_argument("--resume-state", default=None,
                    help="Resume API: input state JSON path (§23.3 schema)")
    ap.add_argument("--start-measure", type=int, default=None,
                    help="Resume API: start measure M (0-based)")
    ap.add_argument("--end-measure", type=int, default=None,
                    help="Resume API: end measure N (M ≤ N ≤ chart_max_measure)")
    ap.add_argument("--finalize", default=None,
                    help="Finalize API: events JSON path; runs post-processing only (§23.6)")
    ap.add_argument("--next-chord-lookahead", default=None,
                    help="Boundary lookahead (§23.7): N+1 first-chord JSON path. "
                         "Object with {measure,idx192,lanes,tokens} OR raw events list — "
                         "engine extracts the smallest-tkey chord. Requires --resume-state.")
    ap.add_argument("--dp", action="store_true", default=False,
                    help="DP synthesis: produce a 14-key+2-scratch chart "
                         "from the SP pool via side-local placement. RB-only, full-chart.")
    ap.add_argument("--dp-split", default="auto", choices=["auto", "timbre", "balance"],
                    help="DP hand-split strategy: timbre (low_freq, stream songs), "
                         "balance (split chords / alternate bursts, chord/peak songs), "
                         "or auto (route by local chord/burst proxies). Default auto.")
    args = ap.parse_args()
    if args.dp and (args.resume_state or args.finalize or args.ml):
        ap.error("--dp cannot be combined with --resume-state/--finalize/--ml (addon §5)")
    if not (1 <= args.intensity <= 20): ap.error("--intensity must be between 1 and 20")
    if not (1 <= args.scratch <= 20):   ap.error("--scratch must be between 1 and 20")
    if args.ml and (not args.model_token or not args.model_lane):
        ap.error("--ml requires both --model-token and --model-lane")
    seed_explicit = args.seed is not None
    if not seed_explicit:
        seed_val = PLACEMENT_RANDOM_SEED
    elif args.seed == "random":
        seed_val = "random"
    else:
        try:
            seed_val = int(args.seed)
        except ValueError:
            ap.error("--seed must be an integer or 'random'")
    # §23.2 branch rules: resume and finalize are mutually exclusive
    if args.resume_state and args.finalize:
        ap.error("--resume-state and --finalize cannot be used together (§23.2)")
    resume_state_data = None
    finalize_events_data = None
    if args.resume_state:
        if args.start_measure is None or args.end_measure is None:
            ap.error("--resume-state requires --start-measure and --end-measure")
        with open(args.resume_state, "r", encoding="utf-8") as f:
            resume_state_data = json.load(f)
        # v12 §23.4: state seed is authoritative in resume mode. Reject explicit
        # CLI --seed to prevent silent override (D.4).
        if seed_explicit:
            ap.error("--seed cannot be combined with --resume-state — state seed "
                     "(rng.seed) takes precedence (§23.4 β-1)")
        state_seed = (resume_state_data.get("rng") or {}).get("seed")
        if isinstance(state_seed, int):
            seed_val = state_seed
    next_chord_lookahead_arg = None
    if args.next_chord_lookahead:
        if not args.resume_state:
            ap.error("--next-chord-lookahead requires --resume-state (§23.7)")
        with open(args.next_chord_lookahead, "r", encoding="utf-8") as f:
            la_data = json.load(f)
        try:
            next_chord_lookahead_arg = normalize_lookahead(la_data)
        except ValueError as e:
            ap.error(f"--next-chord-lookahead invalid: {e}")
    if args.finalize:
        with open(args.finalize, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Accept either {"placed": [...]} (BMS.Compare splice export) or a raw list.
        if isinstance(data, list):
            finalize_events_data = data
        elif isinstance(data, dict):
            if "placed" in data:
                finalize_events_data = data["placed"]
            elif "events" in data:
                finalize_events_data = data["events"]
            else:
                ap.error("--finalize JSON object must contain 'placed' or 'events' key")
            if not isinstance(finalize_events_data, list):
                ap.error("--finalize JSON 'placed'/'events' value must be a list")
        else:
            ap.error("--finalize JSON must be a list of events or an object with 'placed'/'events'")
    main(intensity_level=args.intensity, scratch_level=args.scratch, enable_ln=args.ln,
         enable_ml=args.ml, model_token_path=args.model_token, model_lane_path=args.model_lane,
         seed=seed_val,
         resume_state=resume_state_data,
         start_measure=args.start_measure,
         end_measure=args.end_measure,
         finalize_input_events=finalize_events_data,
         next_chord_lookahead=next_chord_lookahead_arg,
         dp=args.dp, dp_split=args.dp_split)
