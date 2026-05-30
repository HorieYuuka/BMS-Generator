#!/usr/bin/env python3
"""
B'. Human motif family baseline from training data.

Reconstruct per-chart timelines from lane_assignment_dataset.jsonl labels,
extract single-note timelines (chord_size==1 tkeys), compute motif family
distribution. Compare to ML and RB outputs.
"""
import json
import math
import os
import sys
from collections import Counter, defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET = os.path.join(ROOT, "labeling_out_full", "lane_assignment_dataset.jsonl")
OFFSETS = os.path.join(ROOT, "labeling_out_full", "lane_assignment_dataset.jsonl.offsets.npy")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from motif_analysis import (
    canonical_motif_3,
    classify_family,
    family_repeat_burstiness,
    family_run_length,
)


def safe_log(x):
    return math.log(x) if x > 0 else 0.0


def entropy(probs):
    return -sum(p * safe_log(p) for p in probs)


def sample_chart_records(n_charts=200, seed=42, max_records=None):
    """
    Sample N distinct (package_id, chart_file) combos from dataset.
    For each chart, accumulate ALL records.
    Returns: dict[(package, chart)] -> list of records.
    """
    offsets = np.load(OFFSETS)
    rng = np.random.default_rng(seed)
    # First pass: discover charts by sampling 50k records
    discover_n = 50_000
    discover_idx = rng.choice(len(offsets), size=discover_n, replace=False)
    chart_set = set()
    with open(DATASET, "rb") as f:
        for off in offsets[discover_idx]:
            f.seek(int(off))
            line = f.readline()
            try:
                rec = json.loads(line.decode("utf-8"))
            except Exception:
                continue
            pkg = rec.get("package_id")
            chart = rec.get("chart_file")
            if pkg and chart:
                chart_set.add((pkg, chart))
                if len(chart_set) >= n_charts:
                    break
    print(f"  discovered {len(chart_set)} unique (package, chart) pairs in first pass")

    # Second pass: scan all records and keep ones from selected charts
    chart_records = defaultdict(list)
    selected = list(chart_set)[:n_charts]
    selected_set = set(selected)
    n_total = len(offsets) if max_records is None else min(max_records, len(offsets))
    with open(DATASET, "rb") as f:
        for i, off in enumerate(offsets[:n_total]):
            if i % 500_000 == 0 and i > 0:
                print(f"  scanned {i}/{n_total}", file=sys.stderr)
            f.seek(int(off))
            line = f.readline()
            try:
                rec = json.loads(line.decode("utf-8"))
            except Exception:
                continue
            pkg = rec.get("package_id")
            chart = rec.get("chart_file")
            if (pkg, chart) in selected_set:
                chart_records[(pkg, chart)].append({
                    "measure": rec.get("measure"),
                    "idx192": rec.get("idx192"),
                    "label": rec.get("label"),
                })

    return chart_records


def build_human_timeline_per_chart(chart_records):
    """
    For one chart's records, build single-note timeline.
    Group by (measure, idx192), compute chord_size, keep only chord_size==1.
    """
    by_tkey = defaultdict(list)
    for rec in chart_records:
        m = rec["measure"]
        idx = rec["idx192"]
        label = rec["label"]
        if not isinstance(label, int):
            continue
        by_tkey[(m, idx)].append(label)
    timeline = []
    chord_sizes = Counter()
    for tkey, labels in sorted(by_tkey.items()):
        chord_sizes[len(labels)] += 1
        if len(labels) == 1:
            timeline.append((tkey[0], tkey[1], labels[0]))
    return timeline, chord_sizes


def consecutive_3grams_with_gap(timeline, max_gap_tkeys=24):
    grams = []
    for i in range(len(timeline) - 2):
        e1, e2, e3 = timeline[i], timeline[i+1], timeline[i+2]
        t1 = e1[0] * 192 + e1[1]
        t2 = e2[0] * 192 + e2[1]
        t3 = e3[0] * 192 + e3[1]
        if (t2 - t1) > max_gap_tkeys or (t3 - t2) > max_gap_tkeys:
            continue
        grams.append((e1[2], e2[2], e3[2]))
    return grams


def main():
    n_charts = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    print(f"Sampling {n_charts} charts from training data...")
    chart_records = sample_chart_records(n_charts=n_charts, max_records=8_000_000)
    print(f"  collected records for {len(chart_records)} charts")

    all_grams = []
    all_chord_sizes = Counter()
    n_singles_total = 0
    per_chart_singles = []

    for chart_key, recs in chart_records.items():
        timeline, chord_sizes = build_human_timeline_per_chart(recs)
        all_chord_sizes.update(chord_sizes)
        n_singles_total += len(timeline)
        per_chart_singles.append(len(timeline))
        grams = consecutive_3grams_with_gap(timeline, max_gap_tkeys=24)
        all_grams.extend(grams)

    motifs = [canonical_motif_3(g) for g in all_grams]
    families = [classify_family(m) for m in motifs]
    fam_counts = Counter(families)
    total = sum(fam_counts.values())
    fam_probs = [c / total for c in fam_counts.values()] if total else []
    fam_entropy = entropy(fam_probs) if fam_probs else 0
    fam_max_h = math.log(len(fam_counts)) if fam_counts else 1

    print(f"\n=== HUMAN motif baseline (training data) ===")
    print(f"  charts:                {len(chart_records)}")
    print(f"  total records:         {sum(len(r) for r in chart_records.values())}")
    print(f"  single-note tkeys:     {n_singles_total}")
    print(f"  single per chart median: {sorted(per_chart_singles)[len(per_chart_singles)//2]}")
    print(f"  consecutive 3-grams:   {len(all_grams)}")
    print(f"  unique motif families: {len(fam_counts)}")
    print(f"  family entropy:        {fam_entropy:.4f}  (norm: {fam_entropy/fam_max_h:.4f})")

    print(f"\n  Top 20 families:")
    for fam, cnt in fam_counts.most_common(20):
        pct = cnt / total * 100 if total else 0
        print(f"    {fam:24s} {cnt:6d}  ({pct:5.2f}%)")

    print(f"\n  Chord size distribution: {dict(sorted(all_chord_sizes.items()))}")

    # Compare to ML and RB on addiction (most informative case)
    print(f"\n{'='*70}")
    print(f"COMPARISON: HUMAN vs ML vs RB (using existing motif_analysis.json)")
    print(f"{'='*70}")
    motif_path = os.path.join(ROOT, "motif_analysis.json")
    if os.path.exists(motif_path):
        with open(motif_path, encoding="utf-8") as f:
            ma = json.load(f)
        for pkg in ("marion", "addiction"):
            if pkg not in ma:
                continue
            ml = ma[pkg]["ml"]
            rb = ma[pkg]["rb"]
            ml_total = ml["n_grams"]
            rb_total = rb["n_grams"]
            ml_map = ml["fam_counts_all"]
            rb_map = rb["fam_counts_all"]

            all_fams = set(fam_counts) | set(ml_map) | set(rb_map)
            print(f"\n{pkg}:")
            print(f"  HUMAN(N={total}) | ML(N={ml_total}) | RB(N={rb_total})")
            print(f"\n  {'Family':<22s} {'HUM%':>7s} {'ML%':>7s} {'RB%':>7s} {'ML-HUM':>7s} {'RB-HUM':>7s}")

            rows = []
            for fam in all_fams:
                hp = fam_counts.get(fam, 0) / total * 100 if total else 0
                mp = ml_map.get(fam, 0) / ml_total * 100 if ml_total else 0
                rp = rb_map.get(fam, 0) / rb_total * 100 if rb_total else 0
                rows.append((fam, hp, mp, rp, mp - hp, rp - hp))
            # sort by max abs delta from human
            rows.sort(key=lambda x: -max(abs(x[4]), abs(x[5])))
            for fam, hp, mp, rp, dml, drb in rows[:18]:
                flag = ""
                if abs(dml) > 2 or abs(drb) > 2:
                    flag = "  <--"
                print(f"  {fam:<22s} {hp:6.2f}% {mp:6.2f}% {rp:6.2f}% {dml:+6.2f}% {drb:+6.2f}%{flag}")

            # KL each direction
            all_list = sorted(all_fams)
            p_h = [fam_counts.get(f, 0) / max(total, 1) for f in all_list]
            p_ml = [ml_map.get(f, 0) / max(ml_total, 1) for f in all_list]
            p_rb = [rb_map.get(f, 0) / max(rb_total, 1) for f in all_list]
            kl_ml_h = sum(p * (math.log(p) - math.log(q)) for p, q in zip(p_ml, p_h) if p > 0 and q > 0)
            kl_rb_h = sum(p * (math.log(p) - math.log(q)) for p, q in zip(p_rb, p_h) if p > 0 and q > 0)
            kl_h_ml = sum(p * (math.log(p) - math.log(q)) for p, q in zip(p_h, p_ml) if p > 0 and q > 0)
            kl_h_rb = sum(p * (math.log(p) - math.log(q)) for p, q in zip(p_h, p_rb) if p > 0 and q > 0)
            print(f"\n  KL(ML || HUMAN) = {kl_ml_h:.4f}     KL(HUMAN || ML) = {kl_h_ml:.4f}")
            print(f"  KL(RB || HUMAN) = {kl_rb_h:.4f}     KL(HUMAN || RB) = {kl_h_rb:.4f}")
            print(f"  ==> Lower KL_X_HUMAN means X is closer to human motif distribution.")

    # Save
    out = {
        "n_charts": len(chart_records),
        "n_grams": len(all_grams),
        "n_unique_families": len(fam_counts),
        "family_entropy": fam_entropy,
        "family_entropy_norm": fam_entropy / fam_max_h if fam_max_h else 0,
        "fam_counts_all": dict(fam_counts),
        "top20": fam_counts.most_common(20),
    }
    out_path = os.path.join(ROOT, "human_motif_baseline.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=lambda x: list(x) if isinstance(x, tuple) else x)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
