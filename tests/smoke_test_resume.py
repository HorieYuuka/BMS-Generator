"""Smoke + edge-case tests for v12 §23 Resume API.

Cases:
  1. base_split        — full chart vs prefix(0..29) + resume(30..max), byte-identical.
  2. M0_single         — single-measure reroll at M=0 from initial state.
  3. last_measure      — single-measure reroll at measure_max with prefix end_state.
  4. cascading         — three-stage cascade (0..A-1) + (A..B-1) + (B..max).
  5. ml_resume_blocked — --ml + --resume-state exits non-zero (NotImplementedError).
  6. schema_mismatch   — load_resume_state raises ValueError on wrong schema_version.
  7. rng_strategy      — load_resume_state raises ValueError on wrong rng strategy.

Each case prints PASS/FAIL. Exit code = number of failures.
"""

import json
import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PKG = os.path.join(ROOT, "source_packages",
                   "[Neun_jack] Bumblebee(Hardtek_Refix)")
SEED = 42
TMP_PREFIX = os.path.join(ROOT, "_smoke_state_")
RESULT_PATH = os.path.join(ROOT, "placement_result.json")

sys.path.insert(0, ROOT)  # tests/ lives below repo root; import engine modules from there
import resume_state as rs


# ── helpers ───────────────────────────────────────────────────────────────────

def _measure_of(ev):
    return ev.get("measure") if "measure" in ev else ev.get("measure_start")


def _normalize(ev):
    return (
        ev.get("type", "Tap"),
        _measure_of(ev),
        ev.get("idx192", ev.get("idx192_start")),
        ev.get("token", ev.get("tokenStart")),
        ev.get("lane"),
    )


def _run(args, expect_failure=False):
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    proc = subprocess.run(
        [sys.executable, os.path.join(ROOT, "run_pipeline.py")] + args,
        cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8",
    )
    if expect_failure:
        return proc
    if proc.returncode != 0:
        print("STDOUT:", proc.stdout[-1500:])
        print("STDERR:", proc.stderr[-1500:])
        raise RuntimeError(f"pipeline failed (exit {proc.returncode})")
    return proc


def _load_result():
    with open(RESULT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _dump_state(state, name):
    p = TMP_PREFIX + name + ".json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump(state, f)
    return p


def _cleanup():
    for f in os.listdir(ROOT):
        if f.startswith("_smoke_state_"):
            try:
                os.remove(os.path.join(ROOT, f))
            except OSError:
                pass


# ── test cases ────────────────────────────────────────────────────────────────

def test_base_split(split=30):
    """Full chart vs prefix+resume at split."""
    print(f"\n[1] base_split (split={split})")
    _run(["--folder", PKG, "--intensity", "5", "--seed", str(SEED)])
    full = _load_result()
    pre_ln = full.get("placed_pre_ln") or full["placed"]
    measure_max = max(_measure_of(e) for e in pre_ln)
    ref = sorted(map(_normalize, [e for e in pre_ln if _measure_of(e) >= split]))

    init = rs.make_initial_resume_state(seed=SEED)
    p_init = _dump_state(init, "init")
    _run(["--folder", PKG, "--intensity", "5",
          "--resume-state", p_init,
          "--start-measure", "0",
          "--end-measure", str(split - 1)])
    prefix = _load_result()
    p_end = _dump_state(prefix["end_state"], "prefix_end")
    _run(["--folder", PKG, "--intensity", "5",
          "--resume-state", p_end,
          "--start-measure", str(split),
          "--end-measure", str(measure_max)])
    resume = _load_result()
    res = sorted(map(_normalize, resume["events"]))
    ok = ref == res
    print(f"  full[M≥{split}]={len(ref)}, resume={len(res)}: {'PASS' if ok else 'FAIL'}")
    return ok, measure_max


def test_M0_single(measure_max):
    """Single-measure reroll at M=0 from initial state."""
    print(f"\n[2] M0_single")
    _run(["--folder", PKG, "--intensity", "5", "--seed", str(SEED)])
    full = _load_result()
    pre_ln = full.get("placed_pre_ln") or full["placed"]
    ref = sorted(map(_normalize, [e for e in pre_ln if _measure_of(e) == 0]))

    init = rs.make_initial_resume_state(seed=SEED)
    p_init = _dump_state(init, "M0init")
    _run(["--folder", PKG, "--intensity", "5",
          "--resume-state", p_init,
          "--start-measure", "0",
          "--end-measure", "0"])
    res_data = _load_result()
    res = sorted(map(_normalize, res_data["events"]))
    ok = ref == res
    print(f"  full[M=0]={len(ref)}, resume={len(res)}: {'PASS' if ok else 'FAIL'}")
    return ok


def test_last_measure(measure_max):
    """Single-measure reroll at measure_max using prefix end_state."""
    print(f"\n[3] last_measure (M={measure_max})")
    _run(["--folder", PKG, "--intensity", "5", "--seed", str(SEED)])
    full = _load_result()
    pre_ln = full.get("placed_pre_ln") or full["placed"]
    ref = sorted(map(_normalize, [e for e in pre_ln if _measure_of(e) == measure_max]))

    init = rs.make_initial_resume_state(seed=SEED)
    p_init = _dump_state(init, "lastinit")
    _run(["--folder", PKG, "--intensity", "5",
          "--resume-state", p_init,
          "--start-measure", "0",
          "--end-measure", str(measure_max - 1)])
    prefix = _load_result()
    p_end = _dump_state(prefix["end_state"], "last_prefix_end")
    _run(["--folder", PKG, "--intensity", "5",
          "--resume-state", p_end,
          "--start-measure", str(measure_max),
          "--end-measure", str(measure_max)])
    res_data = _load_result()
    res = sorted(map(_normalize, res_data["events"]))
    ok = ref == res
    print(f"  full[M={measure_max}]={len(ref)}, resume={len(res)}: "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def test_cascading(measure_max, A=20, B=55):
    """Three-stage cascade: prefix(0..A-1) + resume(A..B-1) + resume(B..max)."""
    print(f"\n[4] cascading (A={A}, B={B}, end={measure_max})")
    _run(["--folder", PKG, "--intensity", "5", "--seed", str(SEED)])
    full = _load_result()
    pre_ln = full.get("placed_pre_ln") or full["placed"]

    # stage 1
    init = rs.make_initial_resume_state(seed=SEED)
    p1 = _dump_state(init, "casc_init")
    _run(["--folder", PKG, "--intensity", "5",
          "--resume-state", p1,
          "--start-measure", "0",
          "--end-measure", str(A - 1)])
    s1 = _load_result()
    p2 = _dump_state(s1["end_state"], "casc_s1")
    # stage 2
    _run(["--folder", PKG, "--intensity", "5",
          "--resume-state", p2,
          "--start-measure", str(A),
          "--end-measure", str(B - 1)])
    s2 = _load_result()
    p3 = _dump_state(s2["end_state"], "casc_s2")
    # stage 3
    _run(["--folder", PKG, "--intensity", "5",
          "--resume-state", p3,
          "--start-measure", str(B),
          "--end-measure", str(measure_max)])
    s3 = _load_result()

    cascade_events = s1["events"] + s2["events"] + s3["events"]
    ref = sorted(map(_normalize, pre_ln))
    casc = sorted(map(_normalize, cascade_events))
    ok = ref == casc
    print(f"  full={len(ref)}, cascade=({len(s1['events'])}+"
          f"{len(s2['events'])}+{len(s3['events'])})={len(casc)}: "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def test_ml_resume_blocked():
    """--ml + --resume-state should exit non-zero (RB-only v1)."""
    print(f"\n[5] ml_resume_blocked")
    init = rs.make_initial_resume_state(seed=SEED)
    p_init = _dump_state(init, "ml_init")
    proc = _run([
        "--folder", PKG, "--intensity", "5",
        "--ml",
        "--model-token", "training/checkpoints/token_selection_model.pt",
        "--model-lane",  "training/checkpoints/lane_assignment_model.pt",
        "--resume-state", p_init,
        "--start-measure", "0",
        "--end-measure", "5",
    ], expect_failure=True)
    blocked = (proc.returncode != 0
               and ("Resume API v1" in proc.stdout
                    or "Resume API v1" in proc.stderr
                    or "ML" in proc.stdout
                    or "ML" in proc.stderr))
    print(f"  exit={proc.returncode}, blocked={blocked}: "
          f"{'PASS' if blocked else 'FAIL'}")
    return blocked


def test_schema_mismatch():
    """load_resume_state raises ValueError on wrong schema_version."""
    print(f"\n[6] schema_mismatch")
    bad = rs.make_initial_resume_state(seed=SEED)
    bad["schema_version"] = "resume-v999"
    try:
        rs.load_resume_state(bad, scratch_history_maxlen=8)
        print("  no exception raised: FAIL")
        return False
    except ValueError as e:
        ok = "schema_version" in str(e)
        print(f"  ValueError: {str(e)[:80]}: {'PASS' if ok else 'FAIL'}")
        return ok


def test_rng_strategy_mismatch():
    """load_resume_state raises ValueError on wrong rng strategy."""
    print(f"\n[7] rng_strategy_mismatch")
    bad = rs.make_initial_resume_state(seed=SEED)
    bad["rng"]["strategy"] = "serialized-mt-state"
    try:
        rs.load_resume_state(bad, scratch_history_maxlen=8)
        print("  no exception raised: FAIL")
        return False
    except ValueError as e:
        ok = "RNG strategy" in str(e) or "strategy" in str(e)
        print(f"  ValueError: {str(e)[:80]}: {'PASS' if ok else 'FAIL'}")
        return ok


def test_lookahead_requires_resume():
    """--next-chord-lookahead without --resume-state exits non-zero."""
    print(f"\n[8] lookahead_requires_resume")
    la = {"measure": 1, "idx192": 0, "lanes": ["P1_KEY1"], "tokens": ["AB"]}
    p = _dump_state(la, "la_solo")
    proc = _run(["--folder", PKG, "--intensity", "5",
                 "--next-chord-lookahead", p],
                expect_failure=True)
    txt = (proc.stdout + proc.stderr).lower()
    ok = proc.returncode != 0 and ("resume-state" in txt or "resume_state" in txt)
    print(f"  exit={proc.returncode}, mentions resume-state: "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def test_lookahead_basic(measure_max):
    """Single-measure reroll with lookahead built from full chart's M+1 first chord.

    Verifies (a) the run succeeds with --next-chord-lookahead wiring, (b) when
    the gap between N's last placed tkey and N+1's first chord is shorter than
    the BPM-aware jack floor, no lane overlap exists between N's last chord
    and the lookahead chord.
    """
    print(f"\n[9] lookahead_basic")
    _run(["--folder", PKG, "--intensity", "5", "--seed", str(SEED)])
    full = _load_result()
    pre_ln = full.get("placed_pre_ln") or full["placed"]
    M = 30
    next_evs = [e for e in pre_ln if _measure_of(e) == M + 1]
    if not next_evs:
        print("  SKIP: no events at M+1"); return True

    def _evi(e): return e.get("idx192", e.get("idx192_start", 192))
    min_idx = min(_evi(e) for e in next_evs)
    first_chord = [e for e in next_evs if _evi(e) == min_idx]
    la_lanes = sorted({e.get("lane") for e in first_chord if e.get("lane")})
    la = {
        "measure": M + 1, "idx192": min_idx,
        "lanes": la_lanes,
        "tokens": [e.get("token") or e.get("tokenStart") for e in first_chord],
    }
    p_la = _dump_state(la, "la_chord")

    init = rs.make_initial_resume_state(seed=SEED)
    p_init = _dump_state(init, "la_init")
    _run(["--folder", PKG, "--intensity", "5",
          "--resume-state", p_init,
          "--start-measure", "0",
          "--end-measure", str(M - 1)])
    prefix = _load_result()
    p_end = _dump_state(prefix["end_state"], "la_prefix_end")

    _run(["--folder", PKG, "--intensity", "5",
          "--resume-state", p_end,
          "--start-measure", str(M),
          "--end-measure", str(M),
          "--next-chord-lookahead", p_la])
    result = _load_result()
    if result.get("mode") != "resume":
        print("  FAIL: not resume schema"); return False
    events = result["events"]
    if not events:
        print("  SKIP: empty placement"); return True

    def _ek(e): return _measure_of(e) * 192 + _evi(e)
    max_tk = max(_ek(e) for e in events)
    la_tk = (M + 1) * 192 + min_idx
    last_chord = [e for e in events if _ek(e) == max_tk]
    last_lanes = {e.get("lane") for e in last_chord}
    gap = la_tk - max_tk
    # MIN_JACK_DELTA_TICKS at lv5 ≈ 15 (per Quick Reference / v12 §11.5).
    # If gap < 15, lookahead constraint should have rejected overlapping lanes.
    if gap >= 15:
        print(f"  gap={gap}≥15, constraint inactive — wiring smoke only: PASS")
        return True
    overlap = last_lanes & set(la_lanes)
    ok = not overlap
    print(f"  gap={gap}, last={sorted(last_lanes)}, la={la_lanes}: "
          f"{'PASS (no overlap)' if ok else f'FAIL (overlap={sorted(overlap)})'}")
    return ok


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    if not os.path.isdir(PKG):
        raise SystemExit(f"missing source package: {PKG}")
    results = {}
    try:
        ok1, measure_max = test_base_split(split=30)
        results["base_split"] = ok1
        results["M0_single"] = test_M0_single(measure_max)
        results["last_measure"] = test_last_measure(measure_max)
        results["cascading"] = test_cascading(measure_max)
        results["ml_resume_blocked"] = test_ml_resume_blocked()
        results["schema_mismatch"] = test_schema_mismatch()
        results["rng_strategy_mismatch"] = test_rng_strategy_mismatch()
        results["lookahead_requires_resume"] = test_lookahead_requires_resume()
        results["lookahead_basic"] = test_lookahead_basic(measure_max)
    finally:
        _cleanup()

    print("\n=== summary ===")
    failures = [k for k, v in results.items() if not v]
    for k, v in results.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")
    return len(failures)


if __name__ == "__main__":
    sys.exit(main())
