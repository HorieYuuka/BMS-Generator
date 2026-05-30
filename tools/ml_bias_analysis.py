#!/usr/bin/env python3
"""
ML bias analysis: compute lane distributions and entropy metrics
to verify whether LaneAssignmentModel reproduces human bias.
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

LANES = [1, 2, 3, 4, 5, 6, 7]
LANE_NAMES = {1: "K1", 2: "K2", 3: "K3", 4: "K4", 5: "K5", 6: "K6", 7: "K7"}


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


def normalize_counts(counts):
    total = sum(counts.get(l, 0) for l in LANES)
    if total == 0:
        return [0.0] * 7
    return [counts.get(l, 0) / total for l in LANES]


def lane_str_to_int(s):
    if not s.startswith("P1_KEY"):
        return None
    try:
        return int(s[len("P1_KEY"):])
    except ValueError:
        return None


def hand_balance(counts):
    left = sum(counts.get(l, 0) for l in (1, 2, 3))
    right = sum(counts.get(l, 0) for l in (5, 6, 7))
    total = left + right + counts.get(4, 0)
    if total == 0:
        return None
    return {
        "left_3": left / total,
        "center_4": counts.get(4, 0) / total,
        "right_3": right / total,
        "total_lr_keys": total,
    }


def sample_human_prior(n_sample=200_000, seed=42):
    """Sample N records from training dataset, return label distribution + per-key features."""
    offsets = np.load(OFFSETS)
    rng = np.random.default_rng(seed)
    n = min(n_sample, len(offsets))
    idx = rng.choice(len(offsets), size=n, replace=False)
    label_counts = Counter()
    # also track conditional: label | last_lane (for smoking gun)
    cond_counts = defaultdict(Counter)  # last_lane -> Counter(label)
    available_block_counts = Counter()  # how often each lane is masked out
    with open(DATASET, "rb") as f:
        for i, off in enumerate(offsets[idx]):
            f.seek(int(off))
            line = f.readline()
            try:
                rec = json.loads(line.decode("utf-8"))
            except Exception:
                continue
            label = rec.get("label")
            if not isinstance(label, int):
                continue
            label_counts[label] += 1
            avail = rec.get("available_lanes", [])
            for li, v in enumerate(avail):
                if v == 0:
                    available_block_counts[li + 1] += 1
            ctx = rec.get("context", [])
            if ctx:
                last_lane = ctx[0].get("lane")
                if isinstance(last_lane, int):
                    cond_counts[last_lane][label] += 1
            if (i + 1) % 50000 == 0:
                print(f"  ... sampled {i+1}/{n}", file=sys.stderr)
    return {
        "label_counts": dict(label_counts),
        "cond_counts": {k: dict(v) for k, v in cond_counts.items()},
        "available_block_counts": dict(available_block_counts),
        "n_sampled": n,
    }


def lane_dist_from_placed_events(placed_events):
    counts = Counter()
    for ev in placed_events:
        lane_str = ev.get("lane", "")
        l = lane_str_to_int(lane_str)
        if l is not None:
            counts[l] += 1
    return counts


def report_dist(name, counts):
    p = normalize_counts(counts)
    h = entropy(p)
    h_max = math.log(7)
    hb = hand_balance(counts)
    print(f"\n=== {name} ===")
    print(f"  total key notes: {sum(counts.get(l, 0) for l in LANES)}")
    print(f"  distribution (K1..K7): " + " ".join(f"{x*100:5.1f}%" for x in p))
    print(f"  entropy: {h:.4f} / {h_max:.4f}  (normalized: {h/h_max:.3f})")
    if hb:
        print(f"  hand balance: L3={hb['left_3']*100:.1f}%  C={hb['center_4']*100:.1f}%  R3={hb['right_3']*100:.1f}%")
    return p


def main():
    print("=" * 60)
    print("LaneAssignmentModel Bias Analysis")
    print("=" * 60)

    # 1. Human prior from training data
    print("\n[1/4] Sampling human prior from training data...")
    prior = sample_human_prior(n_sample=200_000)
    human_counts = prior["label_counts"]
    print(f"  sampled {prior['n_sampled']} records")
    p_human = report_dist("HUMAN (training labels)", human_counts)

    # 2. Rule-based marion
    print("\n[2/4] Rule-based marion (rb_marion.json)...")
    with open(os.path.join(ROOT, "rb_marion.json"), encoding="utf-8") as f:
        rb_marion = json.load(f)
    rb_marion_counts = lane_dist_from_placed_events(rb_marion["placed"])
    p_rb_marion = report_dist("RULE-BASED (marion)", rb_marion_counts)

    # 3. Current placement_result.json
    print("\n[3/4] Current placement_result.json...")
    with open(os.path.join(ROOT, "placement_result.json"), encoding="utf-8") as f:
        cur = json.load(f)
    cur_counts = lane_dist_from_placed_events(cur["placed"])
    p_cur = report_dist("RULE-BASED (current placement_result)", cur_counts)

    # 4. KL divergences
    print("\n[4/4] KL divergences and comparisons:")
    uniform = [1.0 / 7] * 7
    print(f"\n  KL(HUMAN     || uniform) = {kl(p_human,    uniform):.4f}")
    print(f"  KL(RB_MARION || uniform) = {kl(p_rb_marion, uniform):.4f}")
    print(f"  KL(RB_CURRENT|| uniform) = {kl(p_cur,       uniform):.4f}")
    print()
    print(f"  KL(RB_MARION || HUMAN)   = {kl(p_rb_marion, p_human):.4f}  (rule diverges from human?)")
    print(f"  KL(RB_CURRENT|| HUMAN)   = {kl(p_cur,       p_human):.4f}")
    print()
    print(f"  KL(HUMAN || uniform) > 0.05 means human is biased away from uniform.")
    print(f"  Higher = more biased.")

    # 5. Conditional entropy smoking gun
    print("\n[SMOKING GUN] Conditional entropy H(label | last_lane):")
    cond = prior["cond_counts"]
    h_marginal = entropy(p_human)
    weighted_cond = 0.0
    total_n = sum(sum(c.values()) for c in cond.values())
    print(f"  H(label) marginal = {h_marginal:.4f}")
    print(f"  Per last_lane:")
    for last_lane in sorted(cond.keys()):
        c = cond[last_lane]
        n = sum(c.values())
        p = normalize_counts(c)
        h = entropy(p)
        weighted_cond += (n / total_n) * h
        topk = sorted(c.items(), key=lambda x: -x[1])[:3]
        topk_str = ", ".join(f"K{l}:{cnt}" for l, cnt in topk)
        print(f"    last_lane=K{last_lane} (n={n:6d}): H={h:.4f}  top3=[{topk_str}]")
    print(f"\n  Weighted H(label | last_lane) = {weighted_cond:.4f}")
    print(f"  Mutual info I(label; last_lane) = {h_marginal - weighted_cond:.4f}")
    print(f"  ==> If MI is near 0, last_lane gives ~no info about next label.")
    print(f"      MI > 0.1 means there IS contextual signal (model could exploit).")

    # Save JSON
    out = {
        "n_sampled": prior["n_sampled"],
        "human_distribution": dict(zip(LANE_NAMES.values(), p_human)),
        "human_entropy": entropy(p_human),
        "rb_marion_distribution": dict(zip(LANE_NAMES.values(), p_rb_marion)),
        "rb_current_distribution": dict(zip(LANE_NAMES.values(), p_cur)),
        "kl_human_uniform": kl(p_human, uniform),
        "kl_rb_marion_uniform": kl(p_rb_marion, uniform),
        "kl_rb_current_uniform": kl(p_cur, uniform),
        "kl_rb_marion_human": kl(p_rb_marion, p_human),
        "kl_rb_current_human": kl(p_cur, p_human),
        "h_marginal": h_marginal,
        "h_cond_last_lane": weighted_cond,
        "mutual_info_last_lane": h_marginal - weighted_cond,
    }
    out_path = os.path.join(ROOT, "ml_bias_analysis.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
