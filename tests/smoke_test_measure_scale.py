"""Smoke tests for the scale-aware time axis (v12 §11.5 revision, 2026-06-11).

Raw tkey (= measure*192 + idx192) assumes every measure spans 4 beats; a
#xxx02 measure scale breaks that. These cases verify the beat-true tick
mapping and its consumers (jack floor, LN end/hold-cap, fill-back jack).

Cases:
  1. scaled_tick_identity   — trivial scale map → identity (ints unchanged).
  2. scaled_tick_values     — scale 0.5 / 2.0 measures map to beat-true ticks.
  3. advance_identity       — trivial scale → start + N exactly (legacy cap).
  4. advance_scaled         — 96 beat-true ticks inside a 0.5 measure = 192 raw.
  5. end_tkey_legacy        — no scale → byte-identical legacy result.
  6. end_tkey_scaled        — 0.5-scale measure doubles the raw-tick span.
  7. jack_scale_aware       — raw gap 16 passes at scale 1.0 but is rejected
                              inside a 0.5-scale measure (real time halved).

Each case prints PASS/FAIL. Exit code = number of failures.
"""

import os
import random
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import placement_engine as pe

failures = 0


def check(name, cond, detail=""):
    global failures
    if cond:
        print(f"  PASS {name}")
    else:
        failures += 1
        print(f"  FAIL {name} {detail}")


# ── 1. identity on trivial scale ─────────────────────────────────────────────
fn = pe.build_scaled_tick_fn({}, measure_max=10)
check("scaled_tick_identity_empty", fn(1234) == 1234 and isinstance(fn(1234), int))
fn = pe.build_scaled_tick_fn({0: 1.0, 5: 1.0}, measure_max=10)
check("scaled_tick_identity_all_one", fn(777) == 777 and isinstance(fn(777), int))

# ── 2. beat-true values under non-trivial scale ──────────────────────────────
# measure 0: scale 0.5 (96 beat-true ticks), measure 1: scale 1.0, measure 2: 2.0
fn = pe.build_scaled_tick_fn({0: 0.5, 2: 2.0}, measure_max=4)
check("scaled_m0_mid", fn(96) == 48.0, f"got {fn(96)}")              # half into m0
check("scaled_m1_start", fn(192) == 96.0, f"got {fn(192)}")          # m1 starts after 96
check("scaled_m2_start", fn(384) == 288.0, f"got {fn(384)}")         # 96 + 192
check("scaled_m3_start", fn(576) == 672.0, f"got {fn(576)}")         # 96 + 192 + 384
check("scaled_beyond_table", fn(576 + 192 * 3) == 672.0 + 192 * 3,
      f"got {fn(576 + 192 * 3)}")
check("scaled_sentinel", fn(-999) == -999.0, f"got {fn(-999)}")

# ── 3/4. advance_scaled_ticks (LN hold cap) ──────────────────────────────────
check("advance_identity", pe.advance_scaled_ticks(100, 96, {}) == 196)
# 96 beat-true ticks starting at m0 idx0 with scale 0.5 → whole m0 (192 raw)
check("advance_scaled_half", pe.advance_scaled_ticks(0, 96, {0: 0.5}) == 192,
      f"got {pe.advance_scaled_ticks(0, 96, {0: 0.5})}")
# crossing: 48 beat-true from m0 idx96 (48 left in m0) → exactly m1 start
check("advance_scaled_boundary",
      pe.advance_scaled_ticks(96, 48, {0: 0.5}) == 192,
      f"got {pe.advance_scaled_ticks(96, 48, {0: 0.5})}")

# ── 5. compute_end_tkey legacy path unchanged ────────────────────────────────
# 800 ms @ 150 BPM = 800 * 150 / 1250 = 96 ticks
legacy = pe.compute_end_tkey(0, 800.0, [], 150.0)
with_trivial = pe.compute_end_tkey(0, 800.0, [], 150.0, {0: 1.0})
check("end_tkey_legacy", legacy == 96 and with_trivial == 96,
      f"got {legacy} / {with_trivial}")

# ── 6. compute_end_tkey scale-aware ──────────────────────────────────────────
# m0 at scale 0.5: one raw tick = 1250*0.5/150 ms, so 800 ms = 192 raw ticks
scaled_end = pe.compute_end_tkey(0, 800.0, [], 150.0, {0: 0.5})
check("end_tkey_scaled_half", scaled_end == 192, f"got {scaled_end}")
# spans m0 (0.5, 800 ms total) into m1 (1.0): remaining 400 ms in m1 = 48 raw
scaled_cross = pe.compute_end_tkey(0, 1200.0, [], 150.0, {0: 0.5, 1: 1.0})
check("end_tkey_scaled_cross", scaled_cross == 240, f"got {scaled_cross}")

# ── 7. jack constraint end-to-end via _place_measure_constrained ─────────────
# One candidate at m1 idx16; every lane's jack_state pre-seeded to tkey 192
# (raw gap 16). lv5 params: MIN_JACK_DELTA_TICKS=15, MIN_JACK_DELTA_MS=102.
# At 130 BPM the bpm floor is ceil(102*130/1250) = 11, so effective_min = 15.
#   scale 1.0 → beat-true gap 16 ≥ 15 → placed
#   scale 0.5 → beat-true gap  8 < 15 → all lanes rejected → jack_violation
params = pe.compute_intensity_params(5)


def _place_with(scale_map):
    cands = [(16, "AA", 80.0, 1)]
    jack_state = {l: 192 for l in pe.KEY_LANES}
    scaled = pe.build_scaled_tick_fn(scale_map, measure_max=2)
    placed, _, residuals, _, _ = pe._place_measure_constrained(
        cands, random.Random(1), ("balanced", 0), jack_state,
        1, False, params, scaled_tick=scaled)
    return placed, residuals


placed_flat, res_flat = _place_with({})
check("jack_flat_placed", len(placed_flat) == 1 and not res_flat,
      f"placed={placed_flat} residuals={res_flat}")
placed_half, res_half = _place_with({1: 0.5})
check("jack_half_rejected",
      not placed_half and list(res_half.values()) == ["jack_violation"],
      f"placed={placed_half} residuals={res_half}")

print()
print(f"{'ALL PASS' if failures == 0 else f'{failures} FAILURE(S)'}")
sys.exit(failures)
