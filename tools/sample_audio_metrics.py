#!/usr/bin/env python3
"""Compute audio-aware metrics for ML/RB sample comparison.

For each sample:
  1) avg attack_rms of placed events (audio strength)
  2) token rotation per second (window-based unique-token rate)
  3) top-K reuse rate (token concentration)
"""
import json
import os
import sys
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import mix_generation

SAMPLES = [
    ("bumblebee", "[Neun_jack] Bumblebee(Hardtek_Refix)", "bumblebee(Hardtek_Refix)_bombus.bms"),
    ("egosa", "[ねこみりん feat.みゆ] 先天性エゴサ依存症候群", "egs_00_Twitch.bme"),
    ("tsuramic", "[RearDawn] T'Suramic", "tsuramic_spnanami01.bms"),
    ("wanwan", "[ねこみりん feat. みゆ] ☆わんわんぷらねっつ☆ ～ちきゅうせーふくだいさくせん～", "wanwan_nanami01.bme"),
    ("signal", "[Kuwagata] シグナルほっぴんぐ", "00_kuwagata_signal_[del].bms"),
    ("lepontinia", "[pr.s] Lepontinia", "lepontinia_7k_tlp.bms"),
]


def load_token_attack(folder, bms):
    """Run mix_generation (cache hot, fast) and return token → attack_rms map."""
    mix_generation.run(folder=folder, output_dir=ROOT, bms_filename=bms)
    with open(os.path.join(ROOT, "token_analysis.json"), encoding="utf-8") as f:
        d = json.load(f)
    # token_analysis.json is a list of {token, attack_rms, ...} entries
    return {entry["token"]: entry.get("attack_rms", 0.0) for entry in d}


def analyze_sample(label, mode, attack_map, total_seconds):
    path = os.path.join(ROOT, f"sample_{label}_lv5_{mode}.json")
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    placed = d["placed"]
    tokens = [ev["token"] for ev in placed]
    n = len(tokens)
    if n == 0:
        return None

    # 1) Average attack_rms
    attacks = [attack_map.get(t, 0.0) for t in tokens]
    avg_attack = sum(attacks) / len(attacks)
    high_attack_pct = sum(1 for a in attacks if a > 0.05) / len(attacks) * 100  # threshold 0.05

    # 2) Token rotation per second — count distinct tokens within sliding 1-second windows
    # Use measure*192+idx as tick coordinate. tick_per_sec varies with BPM.
    # Approximation: total ticks / total_seconds = ticks_per_sec.
    if total_seconds <= 0:
        rotation_per_sec = 0
    else:
        # Sort placements by tick
        events_sorted = sorted([(ev["measure"]*192 + ev["idx192"], ev["token"]) for ev in placed])
        max_tick = events_sorted[-1][0] if events_sorted else 0
        ticks_per_sec = max_tick / total_seconds if total_seconds > 0 else 1
        # Window size in ticks for 1 second
        window_ticks = ticks_per_sec
        # Slide by 1-second steps; for each window count distinct tokens
        unique_per_window = []
        i = 0
        n_events = len(events_sorted)
        for win_start in range(0, max_tick, max(int(ticks_per_sec / 4), 1)):
            win_end = win_start + window_ticks
            # Get tokens in window
            ws = set()
            for tk, tok in events_sorted:
                if tk < win_start: continue
                if tk >= win_end: break
                ws.add(tok)
            if ws:
                unique_per_window.append(len(ws))
        rotation_per_sec = sum(unique_per_window) / len(unique_per_window) if unique_per_window else 0

    # 3) Top-K reuse rate (top-3 token coverage)
    counter = Counter(tokens)
    total_uses = sum(counter.values())
    top3 = sum(c for _, c in counter.most_common(3))
    top1 = counter.most_common(1)[0][1] if counter else 0
    top3_pct = top3 / total_uses * 100
    top1_pct = top1 / total_uses * 100

    return {
        "n_events": n,
        "avg_attack": avg_attack,
        "high_attack_pct": high_attack_pct,
        "rotation_per_sec": rotation_per_sec,
        "top1_pct": top1_pct,
        "top3_pct": top3_pct,
    }


def main():
    rows = []
    for (label, pkg, bms) in SAMPLES:
        folder = os.path.join(ROOT, pkg)
        print(f"\n>>> loading attack_rms for {label}", flush=True)
        attack_map = load_token_attack(folder, bms)
        # Get duration from mix_generation_log
        with open(os.path.join(ROOT, "mix_generation_log.json"), encoding="utf-8") as f:
            log = json.load(f)
        duration = log.get("chart", {}).get("estimated_duration_seconds", 0)
        for mode in ("ml", "rb"):
            r = analyze_sample(label, mode, attack_map, duration)
            if r:
                r["label"] = label
                r["mode"] = mode
                rows.append(r)

    print()
    print(f"{'sample':<14s} {'mode':<5s} {'events':>7s} {'avg_atk':>8s} {'hi_atk%':>8s} {'rot/sec':>8s} {'top1%':>7s} {'top3%':>7s}")
    print("-" * 80)
    for r in rows:
        print(f"{r['label']:<14s} {r['mode']:<5s} {r['n_events']:>7d} "
              f"{r['avg_attack']:>8.4f} {r['high_attack_pct']:>7.1f}% "
              f"{r['rotation_per_sec']:>7.2f} {r['top1_pct']:>6.1f}% {r['top3_pct']:>6.1f}%")


if __name__ == "__main__":
    main()
