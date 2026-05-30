#!/usr/bin/env python3
"""
Chord-aware motif family analysis (CODEX feedback fix).

Bug fixed: previous build_timeline() flattened chord lanes via sorted()
ascending, which polluted the timeline with synthetic ascending walks
(e.g., chord {5,6,7} became 3-gram (5,6,7)).

This version:
  - Builds SINGLE-NOTE timeline (chord_size==1 events only) for sequence analysis
  - Records chord shape distribution separately
  - Canonicalizes motifs by delta tuple (translation invariant) + mirror invariance
  - Reports motif family entropy, top mass, sliding-window repeat burstiness
"""
import json
import math
import os
import sys
from collections import Counter, defaultdict, deque

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LANES = list(range(1, 8))


def safe_log(x):
    return math.log(x) if x > 0 else 0.0


def entropy(probs):
    return -sum(p * safe_log(p) for p in probs)


def lane_str_to_int(s):
    if not s.startswith("P1_KEY"):
        return None
    try:
        return int(s[len("P1_KEY"):])
    except ValueError:
        return None


def group_by_tkey(events):
    """Return {(measure, idx192): [lanes...]} ordered."""
    by_tkey = defaultdict(list)
    for ev in events:
        l = lane_str_to_int(ev.get("lane", ""))
        if l is None:
            continue
        by_tkey[(ev["measure"], ev["idx192"])].append(l)
    return by_tkey


def build_single_timeline(events):
    """
    Single-note timeline: only events where the tkey has exactly one key note.
    This is the proper basis for sequence/motif analysis.
    Returns: list of (measure, idx192, lane) in time order.
    """
    by_tkey = group_by_tkey(events)
    timeline = []
    for tkey, lanes in sorted(by_tkey.items()):
        if len(lanes) == 1:
            timeline.append((tkey[0], tkey[1], lanes[0]))
    return timeline


def chord_shape_dist(events):
    """Distribution of chord shapes (set of lanes per tkey, canonicalized)."""
    by_tkey = group_by_tkey(events)
    sizes = Counter()
    shapes = Counter()
    for lanes in by_tkey.values():
        n = len(lanes)
        sizes[n] += 1
        if n >= 2:
            # canonicalize chord shape: sort + translate to start at 0
            sorted_lanes = tuple(sorted(lanes))
            shifted = tuple(l - sorted_lanes[0] for l in sorted_lanes)
            shapes[shifted] += 1
    return sizes, shapes


def canonical_motif_3(triple):
    """
    Canonical motif from a 3-gram of lanes (a, b, c).
    Uses delta tuple (b-a, c-b), then mirror invariance: (-d1, -d2) is the same family.
    Returns lex-smaller of {(d1,d2), (-d1,-d2)}.
    """
    a, b, c = triple
    d1, d2 = b - a, c - b
    forward = (d1, d2)
    mirror = (-d1, -d2)
    return min(forward, mirror)


def classify_family(motif):
    """Human-readable family name."""
    d1, d2 = motif
    abs1, abs2 = abs(d1), abs(d2)
    # contiguous walk: |d|==1 in same direction
    if abs1 == 1 and abs2 == 1 and (d1 > 0) == (d2 > 0):
        return "walk_step1"
    if abs1 == 2 and abs2 == 2 and (d1 > 0) == (d2 > 0):
        return "walk_step2"
    # trill / mirror: a,b,a → d2 = -d1
    if d1 + d2 == 0 and abs1 >= 1:
        return f"trill_dist{abs1}"
    # zigzag: opposite directions, unequal magnitude
    if (d1 > 0) != (d2 > 0):
        return f"zigzag_{abs1}_{abs2}"
    # same direction, mixed step
    if (d1 > 0) == (d2 > 0):
        return f"walk_{abs1}_{abs2}"
    # repeat: d=0 anywhere
    if d1 == 0 or d2 == 0:
        return f"hold_{abs1}_{abs2}"
    return f"other_{d1}_{d2}"


def consecutive_3grams(timeline, max_gap_tkeys=24):
    """
    From single-note timeline, yield 3-grams of consecutive events.
    max_gap_tkeys: skip 3-gram if any gap between consecutive events exceeds this (in tick96 = idx192).
                   24 idx192 ticks ≈ 16th note.
    """
    grams = []
    for i in range(len(timeline) - 2):
        e1, e2, e3 = timeline[i], timeline[i+1], timeline[i+2]
        # tick distance in absolute idx192 (treat measure boundary as +192)
        t1 = e1[0] * 192 + e1[1]
        t2 = e2[0] * 192 + e2[1]
        t3 = e3[0] * 192 + e3[1]
        if (t2 - t1) > max_gap_tkeys or (t3 - t2) > max_gap_tkeys:
            continue
        grams.append((e1[2], e2[2], e3[2]))
    return grams


def family_repeat_burstiness(family_seq, window=16):
    """
    Sliding window: how often does the same family appear ≥ k times in a window?
    Returns: max same-family count in any window, mean, distribution.
    """
    if len(family_seq) < window:
        return None
    max_same = 0
    same_counts = []
    dq = deque(family_seq[:window])
    fc = Counter(dq)
    same_counts.append(max(fc.values()))
    for i in range(window, len(family_seq)):
        out = dq.popleft()
        fc[out] -= 1
        if fc[out] == 0:
            del fc[out]
        dq.append(family_seq[i])
        fc[family_seq[i]] += 1
        same_counts.append(max(fc.values()))
    return {
        "max": max(same_counts),
        "mean": sum(same_counts) / len(same_counts),
        "p95": sorted(same_counts)[int(len(same_counts) * 0.95)],
        "n_windows": len(same_counts),
    }


def family_run_length(family_seq):
    """Distribution of consecutive-same-family run lengths."""
    if not family_seq:
        return Counter()
    runs = Counter()
    cur = family_seq[0]
    n = 1
    for f in family_seq[1:]:
        if f == cur:
            n += 1
        else:
            runs[n] += 1
            cur = f
            n = 1
    runs[n] += 1
    return runs


def analyze(label, events):
    timeline = build_single_timeline(events)
    sizes, shapes = chord_shape_dist(events)
    grams = consecutive_3grams(timeline, max_gap_tkeys=24)
    motifs = [canonical_motif_3(g) for g in grams]
    families = [classify_family(m) for m in motifs]

    fam_counts = Counter(families)
    total_grams = sum(fam_counts.values())
    fam_probs = [c / total_grams for c in fam_counts.values()] if total_grams else []
    fam_entropy = entropy(fam_probs) if fam_probs else 0
    fam_max_entropy = math.log(len(fam_counts)) if len(fam_counts) > 0 else 1

    burst = family_repeat_burstiness(families, window=16)
    runs = family_run_length(families)

    print(f"\n--- {label} ---")
    print(f"  total events: {len(events)}")
    print(f"  single-note timeline: {len(timeline)}  (chord-internal events excluded)")
    print(f"  consecutive 3-grams (gap≤24 ticks): {len(grams)}")
    print(f"  unique motif families: {len(fam_counts)}")
    print(f"  family entropy: {fam_entropy:.4f}  (max if uniform: {fam_max_entropy:.4f})")
    print(f"  normalized family entropy: {fam_entropy/fam_max_entropy if fam_max_entropy else 0:.4f}")

    print(f"\n  Top 10 families:")
    for fam, cnt in fam_counts.most_common(10):
        pct = cnt / total_grams * 100 if total_grams else 0
        print(f"    {fam:24s} {cnt:5d}  ({pct:5.1f}%)")

    if burst:
        print(f"\n  Sliding-window-16 repeat burstiness (max same-family in window):")
        print(f"    max={burst['max']}  mean={burst['mean']:.2f}  p95={burst['p95']}  windows={burst['n_windows']}")

    print(f"\n  Run-length distribution (consecutive same family):")
    for run_len in sorted(runs.keys())[:10]:
        print(f"    run={run_len}: {runs[run_len]}")

    print(f"\n  Chord size distribution: {dict(sorted(sizes.items()))}")
    print(f"  Top 5 chord shapes (size≥2):")
    for shape, cnt in shapes.most_common(5):
        print(f"    {shape}: {cnt}")

    return {
        "label": label,
        "n_events": len(events),
        "n_singles": len(timeline),
        "n_grams": len(grams),
        "n_families": len(fam_counts),
        "family_entropy": fam_entropy,
        "family_entropy_norm": fam_entropy / fam_max_entropy if fam_max_entropy else 0,
        "fam_counts_all": dict(fam_counts),
        "top10_families": fam_counts.most_common(10),
        "burstiness": burst,
        "runs_top10": [(k, runs[k]) for k in sorted(runs.keys())[:10]],
        "chord_sizes": dict(sizes),
    }


def cross_compare(name, ml_data, rb_data):
    """Print delta in family ranks/counts using full counters."""
    print(f"\n{'='*70}")
    print(f"CROSS COMPARISON: {name}")
    print(f"{'='*70}")
    ml_map = ml_data["fam_counts_all"]
    rb_map = rb_data["fam_counts_all"]
    all_fams = set(ml_map) | set(rb_map)
    rows = []
    for fam in all_fams:
        ml = ml_map.get(fam, 0)
        rb = rb_map.get(fam, 0)
        ml_pct = ml / ml_data["n_grams"] * 100 if ml_data["n_grams"] else 0
        rb_pct = rb / rb_data["n_grams"] * 100 if rb_data["n_grams"] else 0
        rows.append((fam, ml, rb, ml_pct, rb_pct, ml_pct - rb_pct))
    print(f"\n{'Family':<22s} {'ML':>6s} {'RB':>6s} {'ML%':>6s} {'RB%':>6s} {'Δ%':>6s}")
    rows.sort(key=lambda x: -abs(x[5]))
    for fam, ml, rb, mlp, rbp, d in rows[:20]:
        flag = "  <--" if abs(d) > 2 else ""
        print(f"  {fam:<22s} {ml:>6d} {rb:>6d} {mlp:>5.1f}% {rbp:>5.1f}% {d:+5.1f}%{flag}")

    # KL between family distributions
    all_list = sorted(all_fams)
    p_ml = [ml_map.get(f, 0) / max(ml_data["n_grams"], 1) for f in all_list]
    p_rb = [rb_map.get(f, 0) / max(rb_data["n_grams"], 1) for f in all_list]
    kl_mr = sum(p * (math.log(p) - math.log(q)) for p, q in zip(p_ml, p_rb) if p > 0 and q > 0)
    kl_rm = sum(p * (math.log(p) - math.log(q)) for p, q in zip(p_rb, p_ml) if p > 0 and q > 0)
    # Jensen-Shannon symmetric
    p_avg = [(a + b) / 2 for a, b in zip(p_ml, p_rb)]
    js = 0.5 * sum(p * (math.log(p) - math.log(q)) for p, q in zip(p_ml, p_avg) if p > 0 and q > 0) + \
         0.5 * sum(p * (math.log(p) - math.log(q)) for p, q in zip(p_rb, p_avg) if p > 0 and q > 0)
    print(f"\n  KL(ML || RB) = {kl_mr:.4f}")
    print(f"  KL(RB || ML) = {kl_rm:.4f}")
    print(f"  JS divergence = {js:.4f}  (0 = identical, log2 = max)")


def load(path):
    with open(os.path.join(ROOT, path), encoding="utf-8") as f:
        return json.load(f)["placed"]


def main():
    print("=" * 70)
    print("Chord-aware motif family analysis")
    print("=" * 70)

    pkgs = [
        ("marion", "ml_marion_result.json", "rb_marion_v2.json"),
        ("addiction", "ml_addiction_result.json", "rb_addiction_result.json"),
    ]

    out = {}
    for name, ml_path, rb_path in pkgs:
        print(f"\n{'#'*70}")
        print(f"# PACKAGE: {name}")
        print(f"{'#'*70}")
        ml_evs = load(ml_path)
        rb_evs = load(rb_path)
        ml_data = analyze(f"{name} ML", ml_evs)
        rb_data = analyze(f"{name} RB", rb_evs)
        cross_compare(name, ml_data, rb_data)
        out[name] = {"ml": ml_data, "rb": rb_data}

    out_path = os.path.join(ROOT, "motif_analysis.json")
    # tuples → lists for JSON
    def coerce(o):
        if isinstance(o, dict):
            return {str(k): coerce(v) for k, v in o.items()}
        if isinstance(o, list):
            return [coerce(x) for x in o]
        if isinstance(o, tuple):
            return [coerce(x) for x in o]
        if isinstance(o, Counter):
            return dict(o)
        return o
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(coerce(out), f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
