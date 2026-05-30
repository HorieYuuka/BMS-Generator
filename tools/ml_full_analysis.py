#!/usr/bin/env python3
"""
Full ML vs RB analysis across 2 packages (marion, addiction).
Includes:
  - lane distribution + entropy + KL
  - chord size distribution
  - lane 3-gram analysis (which sequences are ML-unique vs RB-unique)
  - hand alternation
  - lane reuse (jack tendency)
"""
import json
import math
import os
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LANES = [1, 2, 3, 4, 5, 6, 7]


def safe_log(x):
    return math.log(x) if x > 0 else 0.0


def entropy(probs):
    return -sum(p * safe_log(p) for p in probs)


def kl(p, q):
    s = 0.0
    for pi, qi in zip(p, q):
        if pi > 0 and qi > 0:
            s += pi * (math.log(pi) - math.log(qi))
    return s


def lane_str_to_int(s):
    if not s.startswith("P1_KEY"):
        return None
    try:
        return int(s[len("P1_KEY"):])
    except ValueError:
        return None


def normalize(counts):
    total = sum(counts.get(l, 0) for l in LANES)
    if total == 0:
        return [0.0] * 7
    return [counts.get(l, 0) / total for l in LANES]


def build_timeline(events):
    """Return list of lanes in time order. Within a chord, sort ascending."""
    by_tkey = defaultdict(list)
    for ev in events:
        l = lane_str_to_int(ev.get("lane", ""))
        if l is None:
            continue
        by_tkey[(ev["measure"], ev["idx192"])].append(l)
    timeline = []
    chord_sizes = Counter()
    for (m, idx), lanes in sorted(by_tkey.items()):
        chord_sizes[len(lanes)] += 1
        for l in sorted(lanes):
            timeline.append(l)
    return timeline, chord_sizes


def lane_3grams(timeline):
    grams = Counter()
    for i in range(len(timeline) - 2):
        grams[(timeline[i], timeline[i+1], timeline[i+2])] += 1
    return grams


def hand_alt_rate(timeline):
    def hand(l):
        return "L" if l <= 3 else ("C" if l == 4 else "R")
    alt, same = 0, 0
    for a, b in zip(timeline, timeline[1:]):
        ha, hb = hand(a), hand(b)
        if ha == "C" or hb == "C":
            continue
        if ha == hb:
            same += 1
        else:
            alt += 1
    total = alt + same
    return alt / total if total else 0, alt, same


def lane_reuse(timeline):
    same = sum(1 for a, b in zip(timeline, timeline[1:]) if a == b)
    return same, len(timeline) - 1


def load(name):
    with open(os.path.join(ROOT, name), encoding="utf-8") as f:
        return json.load(f)["placed"]


def per_pkg_analysis(pkg, ml_path, rb_path):
    print(f"\n{'='*70}")
    print(f"PACKAGE: {pkg}")
    print(f"{'='*70}")
    ml_ev = load(ml_path)
    rb_ev = load(rb_path)
    ml_tl, ml_chord = build_timeline(ml_ev)
    rb_tl, rb_chord = build_timeline(rb_ev)

    ml_counts = Counter(ml_tl)
    rb_counts = Counter(rb_tl)
    p_ml = normalize(ml_counts)
    p_rb = normalize(rb_counts)

    print(f"\n  events:        ML={len(ml_ev):4d}   RB={len(rb_ev):4d}")
    print(f"  key timeline:  ML={len(ml_tl):4d}   RB={len(rb_tl):4d}")
    print(f"\n  ML K1..K7: " + "  ".join(f"K{i+1}={p_ml[i]*100:5.1f}%" for i in range(7)))
    print(f"  RB K1..K7: " + "  ".join(f"K{i+1}={p_rb[i]*100:5.1f}%" for i in range(7)))
    print(f"\n  ML entropy norm: {entropy(p_ml)/math.log(7):.4f}   RB entropy norm: {entropy(p_rb)/math.log(7):.4f}")
    print(f"  KL(ML||RB) = {kl(p_ml, p_rb):.4f}   KL(RB||ML) = {kl(p_rb, p_ml):.4f}")

    ml_alt, _, _ = hand_alt_rate(ml_tl)
    rb_alt, _, _ = hand_alt_rate(rb_tl)
    print(f"\n  hand alternation:  ML={ml_alt*100:.1f}%   RB={rb_alt*100:.1f}%")

    ml_reuse, ml_total = lane_reuse(ml_tl)
    rb_reuse, rb_total = lane_reuse(rb_tl)
    print(f"  lane reuse (jack): ML={ml_reuse}/{ml_total}={ml_reuse/ml_total*100:.2f}%   RB={rb_reuse}/{rb_total}={rb_reuse/rb_total*100:.2f}%")

    print(f"\n  chord sizes ML: {dict(sorted(ml_chord.items()))}")
    print(f"  chord sizes RB: {dict(sorted(rb_chord.items()))}")

    # 3-gram analysis
    ml_grams = lane_3grams(ml_tl)
    rb_grams = lane_3grams(rb_tl)

    # Top 10 ML, top 10 RB, top 10 disagreement
    print(f"\n  Top-10 3-grams (ML):")
    for gram, cnt in ml_grams.most_common(10):
        rb_cnt = rb_grams.get(gram, 0)
        print(f"    {gram}: ML={cnt:4d}  RB={rb_cnt:4d}")

    print(f"\n  Top-10 3-grams (RB):")
    for gram, cnt in rb_grams.most_common(10):
        ml_cnt = ml_grams.get(gram, 0)
        print(f"    {gram}: RB={cnt:4d}  ML={ml_cnt:4d}")

    # ML-unique (high ML, low RB)
    print(f"\n  Top-10 ML-OVERREPRESENTED 3-grams (ML count - RB count):")
    diffs = []
    all_grams = set(ml_grams) | set(rb_grams)
    for g in all_grams:
        diffs.append((g, ml_grams.get(g, 0) - rb_grams.get(g, 0)))
    diffs.sort(key=lambda x: -x[1])
    for g, d in diffs[:10]:
        print(f"    {g}: ML={ml_grams.get(g, 0):3d}  RB={rb_grams.get(g, 0):3d}  delta=+{d}")

    print(f"\n  Top-10 RB-OVERREPRESENTED 3-grams:")
    diffs.sort(key=lambda x: x[1])
    for g, d in diffs[:10]:
        print(f"    {g}: ML={ml_grams.get(g, 0):3d}  RB={rb_grams.get(g, 0):3d}  delta={d}")

    # Detect "awkward" patterns: pinky-ring-pinky type
    awkward_specs = [
        ("K6-K7-K6 (pinky-ring-pinky right)", (6, 7, 6)),
        ("K7-K6-K7 (right twist)",            (7, 6, 7)),
        ("K1-K2-K1 (pinky-ring-pinky left)",  (1, 2, 1)),
        ("K2-K1-K2 (left twist)",             (2, 1, 2)),
        ("K3-K6-K3 (cross-hand jump)",        (3, 6, 3)),
        ("K1-K7-K1 (long jump back)",         (1, 7, 1)),
        ("K7-K1-K7 (long jump back)",         (7, 1, 7)),
    ]
    print(f"\n  Awkward 3-grams check:")
    for label, gram in awkward_specs:
        print(f"    {label}: ML={ml_grams.get(gram, 0):3d}  RB={rb_grams.get(gram, 0):3d}")

    # Position overlap
    ml_pos = {(ev["measure"], ev["idx192"], ev["token"]) for ev in ml_ev}
    rb_pos = {(ev["measure"], ev["idx192"], ev["token"]) for ev in rb_ev}
    common = ml_pos & rb_pos
    ml_triple = {(ev["measure"], ev["idx192"], ev["token"], ev["lane"]) for ev in ml_ev}
    rb_triple = {(ev["measure"], ev["idx192"], ev["token"], ev["lane"]) for ev in rb_ev}
    common_triple = ml_triple & rb_triple
    print(f"\n  Position overlap: {len(common)/max(len(ml_pos),len(rb_pos))*100:.1f}%")
    print(f"  Lane match within common: {len(common_triple)/max(len(common),1)*100:.1f}%")

    return {
        "package": pkg,
        "ml_dist": [round(x*100, 2) for x in p_ml],
        "rb_dist": [round(x*100, 2) for x in p_rb],
        "ml_entropy_norm": entropy(p_ml)/math.log(7),
        "rb_entropy_norm": entropy(p_rb)/math.log(7),
        "kl_ml_rb": kl(p_ml, p_rb),
        "ml_lane_reuse_pct": ml_reuse/ml_total*100,
        "rb_lane_reuse_pct": rb_reuse/rb_total*100,
        "ml_hand_alt_pct": ml_alt*100,
        "rb_hand_alt_pct": rb_alt*100,
        "lane_match_within_common": len(common_triple)/max(len(common),1)*100,
        "awkward_3grams": {
            label: {"ML": ml_grams.get(g, 0), "RB": rb_grams.get(g, 0)}
            for label, g in awkward_specs
        },
    }


def main():
    results = []
    results.append(per_pkg_analysis(
        "marion (09_Marion_last.bml)",
        "ml_marion_result.json",
        "rb_marion_v2.json",
    ))
    results.append(per_pkg_analysis(
        "addiction (Addiction_INFERNO.bms)",
        "ml_addiction_result.json",
        "rb_addiction_result.json",
    ))

    out_path = os.path.join(ROOT, "ml_full_analysis.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
