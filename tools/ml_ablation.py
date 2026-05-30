#!/usr/bin/env python3
"""
B. Ablation: toggle each ML post-processing component to isolate model vs heuristic.

Variants:
  - all_on        : baseline (current production)
  - no_diversity  : LANE_DIVERSITY_WEIGHT = 0
  - no_spread     : SPREAD_BONUS_WEIGHT = 0
  - no_balance    : LANE_GLOBAL_BALANCE_WEIGHT = 0
  - raw_model     : all three = 0 (closest to raw model)

For each variant, runs placement_engine on a target package, saves output JSON,
then runs motif analysis to compare.
"""
import json
import os
import shutil
import subprocess
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


VARIANTS = {
    "all_on":       {"diversity": 1.5, "spread": 0.6, "balance": 4.0},
    "no_diversity": {"diversity": 0.0, "spread": 0.6, "balance": 4.0},
    "no_spread":    {"diversity": 1.5, "spread": 0.0, "balance": 4.0},
    "no_balance":   {"diversity": 1.5, "spread": 0.6, "balance": 0.0},
    "raw_model":    {"diversity": 0.0, "spread": 0.0, "balance": 0.0},
}


def run_variant_pkg(variant_name, weights, folder, bms, pkg_name, intensity=5, seed=42):
    """Like run_variant but with package-specific output path and forces mix_generation per package."""
    out_path = os.path.join(ROOT, f"ablation_{pkg_name}_{variant_name}.json")
    print(f"\n>>> {pkg_name}/{variant_name}  (div={weights['diversity']}, spread={weights['spread']}, balance={weights['balance']})")
    runner = f"""
import os, sys, json
sys.path.insert(0, r"{ROOT}")
import placement_engine as pe
pe.LANE_DIVERSITY_WEIGHT = {weights['diversity']}
pe.SPREAD_BONUS_WEIGHT = {weights['spread']}
pe.LANE_GLOBAL_BALANCE_WEIGHT = {weights['balance']}

import os.path as _osp
_folder = r"{folder}"
_bms = r"{bms}"
_target_path = _osp.join(_folder, _bms)
pe.load_bms_bytes = lambda: open(_target_path, "rb").read()
pe.TARGET_BMS = _bms
pe.TOKEN_ANALYSIS = _osp.join(r"{ROOT}", "token_analysis.json")
pe.RESULT_PATH = _osp.join(r"{ROOT}", "placement_result.json")

pe.main(
    intensity_level={intensity},
    scratch_level=5,
    enable_ln=False,
    enable_ml=True,
    model_token_path=_osp.join(r"{ROOT}", "training/checkpoints/token_selection_model.pt"),
    model_lane_path=_osp.join(r"{ROOT}", "training/checkpoints/lane_assignment_model.pt"),
    seed={seed},
)
"""
    runner_path = os.path.join(ROOT, "tools", "_ablation_runner.py")
    with open(runner_path, "w", encoding="utf-8") as f:
        f.write(runner)

    mix_runner = f"""
import sys, os
sys.path.insert(0, r"{ROOT}")
import mix_generation
mix_generation.run(
    folder=r"{folder}",
    output_dir=r"{ROOT}",
    bms_filename=r"{bms}",
)
"""
    mix_path = os.path.join(ROOT, "tools", "_mix_runner.py")
    with open(mix_path, "w", encoding="utf-8") as f:
        f.write(mix_runner)

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    # Run mix_generation only for the first variant of this package
    if variant_name == "all_on":
        subprocess.run([sys.executable, mix_path], check=True, env=env, cwd=ROOT)

    res = subprocess.run([sys.executable, runner_path], capture_output=True, env=env, cwd=ROOT, encoding="utf-8")
    if res.returncode != 0:
        print(res.stdout)
        print(res.stderr)
        raise RuntimeError(f"variant {variant_name} for {pkg_name} failed")

    src = os.path.join(ROOT, "placement_result.json")
    shutil.copy(src, out_path)
    return out_path


def analyze_paths(paths):
    """Analyze ablation outputs and print per-variant summary."""
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    from motif_analysis import (
        build_single_timeline,
        consecutive_3grams,
        canonical_motif_3,
        classify_family,
    )
    import math

    # human baseline
    human_path = os.path.join(ROOT, "human_motif_baseline.json")
    h_map = None
    h_total = 0
    if os.path.exists(human_path):
        with open(human_path, encoding="utf-8") as f:
            hb = json.load(f)
        h_map = hb["fam_counts_all"]
        h_total = sum(h_map.values())

    rows = []
    for variant_name, path in paths.items():
        with open(path, encoding="utf-8") as f:
            events = json.load(f)["placed"]
        timeline = build_single_timeline(events)
        grams = consecutive_3grams(timeline, max_gap_tkeys=24)
        families = [classify_family(canonical_motif_3(g)) for g in grams]
        fc = Counter(families)
        total = sum(fc.values())
        unique = len(fc)
        max_h = math.log(unique) if unique else 1
        h = -sum((c / total) * math.log(c / total) for c in fc.values() if c > 0)
        h_norm = h / max_h if max_h else 0
        lc = lane_counts(events)
        lc_total = sum(lc.values())

        kl_h = None
        if h_map:
            all_fams = set(h_map) | set(fc)
            kl = 0.0
            for fam in all_fams:
                p = fc.get(fam, 0) / max(total, 1)
                q = h_map.get(fam, 0) / max(h_total, 1)
                if p > 0 and q > 0:
                    kl += p * (math.log(p) - math.log(q))
            kl_h = kl

        rows.append({
            "variant": variant_name,
            "n_events": len(events),
            "n_singles": len(timeline),
            "n_grams": len(grams),
            "n_families": unique,
            "family_entropy": h,
            "family_entropy_norm": h_norm,
            "kl_to_human": kl_h,
            "top10": fc.most_common(10),
            "lane_dist": [lc.get(l, 0) / lc_total * 100 if lc_total else 0 for l in range(1, 8)],
        })

    print(f"\n{'variant':<14s} {'events':>7s} {'singles':>8s} {'grams':>7s} {'fams':>5s} {'H_norm':>7s}  KL_H   K1..K7 distribution")
    for r in rows:
        ld = " ".join(f"{x:5.1f}" for x in r["lane_dist"])
        kl = f"{r['kl_to_human']:.3f}" if r['kl_to_human'] is not None else "  -  "
        print(f"  {r['variant']:<12s} {r['n_events']:>7d} {r['n_singles']:>8d} {r['n_grams']:>7d} {r['n_families']:>5d} {r['family_entropy_norm']:>6.3f} {kl}  {ld}")

    return rows


def run_variant(variant_name, weights, folder, bms, intensity=5, seed=42):
    out_path = os.path.join(ROOT, f"ablation_{variant_name}.json")
    print(f"\n>>> running variant: {variant_name}  (div={weights['diversity']}, spread={weights['spread']}, balance={weights['balance']})")

    # Inject overrides via env vars (placement_engine doesn't read these natively;
    # we'll patch via a small runner script below).
    runner = f"""
import os, sys, json
sys.path.insert(0, r"{ROOT}")
import placement_engine as pe
pe.LANE_DIVERSITY_WEIGHT = {weights['diversity']}
pe.SPREAD_BONUS_WEIGHT = {weights['spread']}
pe.LANE_GLOBAL_BALANCE_WEIGHT = {weights['balance']}

# replicate run_pipeline step 2 for ML mode
import os.path as _osp
_folder = r"{folder}"
_bms = r"{bms}"
_target_path = _osp.join(_folder, _bms)
pe.load_bms_bytes = lambda: open(_target_path, "rb").read()
pe.TARGET_BMS = _bms
pe.TOKEN_ANALYSIS = _osp.join(r"{ROOT}", "token_analysis.json")
pe.RESULT_PATH = _osp.join(r"{ROOT}", "placement_result.json")

pe.main(
    intensity_level={intensity},
    scratch_level=5,
    enable_ln=False,
    enable_ml=True,
    model_token_path=_osp.join(r"{ROOT}", "training/checkpoints/token_selection_model.pt"),
    model_lane_path=_osp.join(r"{ROOT}", "training/checkpoints/lane_assignment_model.pt"),
    seed={seed},
)
"""
    runner_path = os.path.join(ROOT, "tools", "_ablation_runner.py")
    with open(runner_path, "w", encoding="utf-8") as f:
        f.write(runner)

    # Need token_analysis.json to be current for this BMS. Run mix_generation step first.
    mix_runner = f"""
import sys, os
sys.path.insert(0, r"{ROOT}")
import mix_generation
mix_generation.run(
    folder=r"{folder}",
    output_dir=r"{ROOT}",
    bms_filename=r"{bms}",
)
"""
    mix_path = os.path.join(ROOT, "tools", "_mix_runner.py")
    with open(mix_path, "w", encoding="utf-8") as f:
        f.write(mix_runner)

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    if variant_name == list(VARIANTS.keys())[0]:
        # Run mix_generation only once (token_analysis.json caches)
        subprocess.run([sys.executable, mix_path], check=True, env=env, cwd=ROOT)

    res = subprocess.run([sys.executable, runner_path], capture_output=True, env=env, cwd=ROOT, encoding="utf-8")
    if res.returncode != 0:
        print(res.stdout)
        print(res.stderr)
        raise RuntimeError(f"variant {variant_name} failed")

    # copy placement_result.json to ablation file
    src = os.path.join(ROOT, "placement_result.json")
    shutil.copy(src, out_path)
    return out_path


def lane_counts(events):
    c = Counter()
    for ev in events:
        l = ev.get("lane", "")
        if l.startswith("P1_KEY"):
            try:
                c[int(l[6:])] += 1
            except ValueError:
                pass
    return c


def main():
    targets = [
        (os.path.join(ROOT, "[- 4 5] A D D i c T i O N 4 5 0 0 0 0 0"), "Addiction_INFERNO.bms", "addiction"),
        (os.path.join(ROOT, "Insane BMS (2025-12-14)", "Coolish Space (by MC_HUM)"), "Coolish_Space_ANOTHER.bms", "coolish"),
        (os.path.join(ROOT, "Insane BMS (2025-12-14)", "Adansonia (by SWEEZ & Meine Meinung)"), "adansonia_hyper.bms", "adansonia"),
    ]
    if len(sys.argv) > 1 and sys.argv[1] != "all":
        idx = int(sys.argv[1])
        targets = [targets[idx]]

    all_results = {}
    for folder, bms, name in targets:
        print(f"\n\n{'#'*70}")
        print(f"# PACKAGE: {name}")
        print(f"{'#'*70}")
        # Override out paths to include package suffix
        paths = {}
        for variant_name, weights in VARIANTS.items():
            paths[variant_name] = run_variant_pkg(variant_name, weights, folder, bms, name)

        rows = analyze_paths(paths)
        all_results[name] = rows

    # Save
    out_path = os.path.join(ROOT, "ml_ablation_multi.json")
    with open(out_path, "w", encoding="utf-8") as f:
        def coerce(o):
            if isinstance(o, dict):
                return {str(k): coerce(v) for k, v in o.items()}
            if isinstance(o, (list, tuple)):
                return [coerce(x) for x in o]
            return o
        json.dump(coerce(all_results), f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_path}")
    return

    # OLD path:
    folder = os.path.join(ROOT, "[- 4 5] A D D i c T i O N 4 5 0 0 0 0 0")
    bms = "Addiction_INFERNO.bms"

    paths = {}
    for variant_name, weights in VARIANTS.items():
        paths[variant_name] = run_variant(variant_name, weights, folder, bms)

    # Run motif analysis on each variant
    print(f"\n{'='*70}")
    print(f"MOTIF FAMILY ANALYSIS PER VARIANT")
    print(f"{'='*70}")

    sys.path.insert(0, os.path.join(ROOT, "tools"))
    from motif_analysis import (
        build_single_timeline,
        consecutive_3grams,
        canonical_motif_3,
        classify_family,
    )
    import math

    rows = []
    for variant_name, path in paths.items():
        with open(path, encoding="utf-8") as f:
            events = json.load(f)["placed"]
        timeline = build_single_timeline(events)
        grams = consecutive_3grams(timeline, max_gap_tkeys=24)
        families = [classify_family(canonical_motif_3(g)) for g in grams]
        fc = Counter(families)
        total = sum(fc.values())
        unique = len(fc)
        max_h = math.log(unique) if unique else 1
        h = -sum((c / total) * math.log(c / total) for c in fc.values() if c > 0)
        h_norm = h / max_h if max_h else 0
        lc = lane_counts(events)
        lc_total = sum(lc.values())
        rows.append({
            "variant": variant_name,
            "n_events": len(events),
            "n_singles": len(timeline),
            "n_grams": len(grams),
            "n_families": unique,
            "family_entropy": h,
            "family_entropy_norm": h_norm,
            "top10": fc.most_common(10),
            "lane_dist": [lc.get(l, 0) / lc_total * 100 if lc_total else 0 for l in range(1, 8)],
        })

    print(f"\n{'variant':<14s} {'events':>7s} {'singles':>8s} {'grams':>7s} {'fams':>5s} {'H_norm':>7s}  K1..K7 distribution")
    for r in rows:
        ld = " ".join(f"{x:5.1f}" for x in r["lane_dist"])
        print(f"  {r['variant']:<12s} {r['n_events']:>7d} {r['n_singles']:>8d} {r['n_grams']:>7d} {r['n_families']:>5d} {r['family_entropy_norm']:>6.3f}  {ld}")

    print(f"\nTop 5 families per variant:")
    for r in rows:
        top5 = ", ".join(f"{f}({c})" for f, c in r["top10"][:5])
        print(f"  {r['variant']:<12s}: {top5}")

    # Compare to human baseline
    human_path = os.path.join(ROOT, "human_motif_baseline.json")
    if os.path.exists(human_path):
        with open(human_path, encoding="utf-8") as f:
            hb = json.load(f)
        h_map = hb["fam_counts_all"]
        h_total = sum(h_map.values())

        print(f"\n{'='*70}")
        print(f"KL DIVERGENCE TO HUMAN BASELINE")
        print(f"{'='*70}")
        for r in rows:
            ml_total = r["n_grams"]
            top10 = dict(r["top10"])
            # Use full counts
            # We didn't save fam_counts_all here; reconstruct from top10 plus rest
            # For accurate KL, recompute from events
        # Recompute KL using full family counts
        for variant_name, path in paths.items():
            with open(path, encoding="utf-8") as f:
                events = json.load(f)["placed"]
            timeline = build_single_timeline(events)
            grams = consecutive_3grams(timeline, max_gap_tkeys=24)
            families = [classify_family(canonical_motif_3(g)) for g in grams]
            fc = Counter(families)
            total = sum(fc.values())
            all_fams = set(h_map) | set(fc)
            kl = 0.0
            for fam in all_fams:
                p = fc.get(fam, 0) / max(total, 1)
                q = h_map.get(fam, 0) / max(h_total, 1)
                if p > 0 and q > 0:
                    kl += p * (math.log(p) - math.log(q))
            print(f"  KL({variant_name:<12s} || HUMAN) = {kl:.4f}")

    out = {"variants": rows, "package": bms}
    with open(os.path.join(ROOT, "ml_ablation.json"), "w", encoding="utf-8") as f:
        # tuples → lists for JSON
        def coerce(o):
            if isinstance(o, dict):
                return {str(k): coerce(v) for k, v in o.items()}
            if isinstance(o, (list, tuple)):
                return [coerce(x) for x in o]
            return o
        json.dump(coerce(out), f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
