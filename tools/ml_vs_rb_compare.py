#!/usr/bin/env python3
"""
ML vs Rule-Based head-to-head comparison on same source.
Both ml_marion_result.json and rb_marion_v2.json were generated from
09_Marion_last.bml with seed=42, intensity=5.
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


def lane_counts(events):
    c = Counter()
    for ev in events:
        l = lane_str_to_int(ev.get("lane", ""))
        if l is not None:
            c[l] += 1
    return c


def adjacent_lane_transitions(events):
    """For each (measure, idx192) tkey group, sort by some order and count transitions."""
    by_tkey = defaultdict(list)
    for ev in events:
        l = lane_str_to_int(ev.get("lane", ""))
        if l is None:
            continue
        m = ev["measure"]
        idx = ev["idx192"]
        by_tkey[(m, idx)].append(l)
    # collect time-ordered single lane events (skip chords) for transition analysis
    timeline = sorted(by_tkey.items())
    transitions = Counter()
    last_lane = None
    same_lane_count = 0
    total_transitions = 0
    for (m, idx), lanes in timeline:
        for l in sorted(lanes):  # order within chord is arbitrary
            if last_lane is not None:
                transitions[(last_lane, l)] += 1
                if last_lane == l:
                    same_lane_count += 1
                total_transitions += 1
            last_lane = l
    return transitions, same_lane_count, total_transitions


def chord_size_distribution(events):
    by_tkey = defaultdict(list)
    for ev in events:
        l = lane_str_to_int(ev.get("lane", ""))
        if l is None:
            continue
        by_tkey[(ev["measure"], ev["idx192"])].append(l)
    sizes = Counter()
    for lanes in by_tkey.values():
        sizes[len(lanes)] += 1
    return sizes


def hand_alternation_rate(events):
    """% of consecutive notes where hand alternates (left=1-3, center=4, right=5-7)."""
    timeline = []
    by_tkey = defaultdict(list)
    for ev in events:
        l = lane_str_to_int(ev.get("lane", ""))
        if l is None:
            continue
        by_tkey[(ev["measure"], ev["idx192"])].append(l)
    for (m, idx), lanes in sorted(by_tkey.items()):
        for l in sorted(lanes):
            timeline.append(l)

    def hand(l):
        if l <= 3:
            return "L"
        elif l == 4:
            return "C"
        else:
            return "R"

    alt_count = 0
    same_count = 0
    for a, b in zip(timeline, timeline[1:]):
        ha, hb = hand(a), hand(b)
        if ha == "C" or hb == "C":
            continue  # skip center
        if ha == hb:
            same_count += 1
        else:
            alt_count += 1
    total = alt_count + same_count
    return (alt_count / total) if total > 0 else 0.0, alt_count, same_count


def load_events(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)["placed"]


def report(name, events):
    counts = lane_counts(events)
    p = normalize(counts)
    h = entropy(p)
    sizes = chord_size_distribution(events)
    alt_rate, alt_n, same_n = hand_alternation_rate(events)
    print(f"\n=== {name} ===")
    print(f"  total events: {len(events)}")
    print(f"  key total:   {sum(counts.get(l, 0) for l in LANES)}")
    print(f"  K1..K7 dist: " + " ".join(f"{x*100:5.1f}%" for x in p))
    print(f"  entropy:     {h:.4f} / {math.log(7):.4f}  (norm: {h/math.log(7):.3f})")
    left = sum(counts.get(l, 0) for l in (1, 2, 3))
    right = sum(counts.get(l, 0) for l in (5, 6, 7))
    total = left + right + counts.get(4, 0)
    print(f"  hand bal:    L={left/total*100:.1f}%  C={counts.get(4,0)/total*100:.1f}%  R={right/total*100:.1f}%")
    print(f"  chord sizes: {dict(sorted(sizes.items()))}")
    print(f"  hand alt:    {alt_rate*100:.1f}%  (alt={alt_n}, same={same_n})")
    return p, counts


def main():
    ml_events = load_events(os.path.join(ROOT, "ml_marion_result.json"))
    rb_events = load_events(os.path.join(ROOT, "rb_marion_v2.json"))

    print("=" * 60)
    print("ML vs Rule-Based: head-to-head on 09_Marion_last.bml seed=42 lv5")
    print("=" * 60)

    p_ml, c_ml = report("ML (lane model)", ml_events)
    p_rb, c_rb = report("RULE-BASED (centroid + ε-greedy)", rb_events)

    # Position overlap analysis
    ml_pos = {(ev["measure"], ev["idx192"], ev["token"]) for ev in ml_events}
    rb_pos = {(ev["measure"], ev["idx192"], ev["token"]) for ev in rb_events}
    common = ml_pos & rb_pos
    print(f"\n=== POSITION OVERLAP ===")
    print(f"  ML unique:   {len(ml_pos)}")
    print(f"  RB unique:   {len(rb_pos)}")
    print(f"  Common:      {len(common)}  ({len(common)/len(ml_pos)*100:.1f}% of ML, {len(common)/len(rb_pos)*100:.1f}% of RB)")

    # Triple overlap (m, idx, token, lane)
    ml_triple = {(ev["measure"], ev["idx192"], ev["token"], ev["lane"]) for ev in ml_events}
    rb_triple = {(ev["measure"], ev["idx192"], ev["token"], ev["lane"]) for ev in rb_events}
    common_triple = ml_triple & rb_triple
    print(f"  Lane-match:  {len(common_triple)}  ({len(common_triple)/len(common)*100:.1f}% of common positions)")
    print(f"  ==> 100*(1-x) = % of common positions where ML/RB chose DIFFERENT lanes")

    # KL divergence between ML and RB
    print(f"\n=== KL DIVERGENCES ===")
    uniform = [1.0 / 7] * 7
    print(f"  KL(ML || uniform)  = {kl(p_ml, uniform):.4f}")
    print(f"  KL(RB || uniform)  = {kl(p_rb, uniform):.4f}")
    print(f"  KL(ML || RB)       = {kl(p_ml, p_rb):.4f}")
    print(f"  KL(RB || ML)       = {kl(p_rb, p_ml):.4f}")

    # Per-key delta
    print(f"\n=== PER-KEY DELTA (ML - RB) ===")
    for l in LANES:
        ml_pct = c_ml.get(l, 0) / sum(c_ml.values()) * 100 if c_ml else 0
        rb_pct = c_rb.get(l, 0) / sum(c_rb.values()) * 100 if c_rb else 0
        delta = ml_pct - rb_pct
        flag = ""
        if abs(delta) > 3:
            flag = "  <-- DEVIATION"
        print(f"  K{l}: ML={ml_pct:5.1f}%  RB={rb_pct:5.1f}%  delta={delta:+5.1f}%{flag}")

    # Local lane reuse rate (jack tendency)
    _, ml_same, ml_trans = adjacent_lane_transitions(ml_events)
    _, rb_same, rb_trans = adjacent_lane_transitions(rb_events)
    print(f"\n=== LANE REUSE (consecutive-event same-lane rate) ===")
    print(f"  ML: {ml_same}/{ml_trans} = {ml_same/ml_trans*100:.2f}%")
    print(f"  RB: {rb_same}/{rb_trans} = {rb_same/rb_trans*100:.2f}%")

    # Save
    out = {
        "ml_distribution_pct": [round(x*100, 2) for x in p_ml],
        "rb_distribution_pct": [round(x*100, 2) for x in p_rb],
        "ml_entropy_normalized": entropy(p_ml) / math.log(7),
        "rb_entropy_normalized": entropy(p_rb) / math.log(7),
        "kl_ml_rb": kl(p_ml, p_rb),
        "position_overlap": len(common) / max(len(ml_pos), len(rb_pos)),
        "triple_overlap_pct": len(common_triple) / max(len(common), 1) * 100,
    }
    with open(os.path.join(ROOT, "ml_vs_rb_compare.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
