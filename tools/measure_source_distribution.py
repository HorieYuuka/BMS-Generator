"""
Measure placement_distribution metrics directly on a source BMS file.

Used to validate the "ML mimics human pattern" reframing hypothesis: if
human sources show fat-tail same-hand streaks like ML's output, then
ML's deviation from RB on that axis IS the human-mimicry signal we
couldn't quantify before.

Usage:
    python tools/measure_source_distribution.py <chart.bms>
    python tools/measure_source_distribution.py --folder <package>
    python tools/measure_source_distribution.py --batch <pattern>

The tool parses key-channel events (11-15, 18, 19) directly from the BMS,
maps them to P1_KEY1..7 lanes, and calls
placement_engine.compute_placement_distribution_metrics. Channel 16
(scratch) is excluded — same convention as the runtime metric.
"""
import argparse
import glob
import os
import sys

ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, ROOT)

from bms_parser import parse_bms
from placement_engine import compute_placement_distribution_metrics


CH_TO_KEY_LANE = {
    "11": "P1_KEY1", "12": "P1_KEY2", "13": "P1_KEY3",
    "14": "P1_KEY4", "15": "P1_KEY5",
    "18": "P1_KEY6", "19": "P1_KEY7",
}
LN_CH_TO_KEY_LANE = {
    "51": "P1_KEY1", "52": "P1_KEY2", "53": "P1_KEY3",
    "54": "P1_KEY4", "55": "P1_KEY5",
    "58": "P1_KEY6", "59": "P1_KEY7",
}


def source_to_placed_events(bms_path):
    """Build a placed_events-shaped list from source BMS for the metric."""
    with open(bms_path, "rb") as f:
        pr = parse_bms(f.read())
    events = pr["events"]
    placed = []
    for ev in events:
        et = ev.get("type")
        if et == "Tap":
            ch = ev.get("rawChannel", "")
            lane = CH_TO_KEY_LANE.get(ch)
            if lane is None:
                continue
            placed.append({
                "type": "Tap",
                "lane": lane,
                "measure": ev["measure"],
                "idx192": ev["idx192"],
                "token": ev.get("token", ""),
            })
        elif et == "Long":
            ch = ev.get("rawChannelStart", "")
            lane = LN_CH_TO_KEY_LANE.get(ch) or CH_TO_KEY_LANE.get(ch)
            if lane is None:
                continue
            placed.append({
                "type": "LN",
                "lane": lane,
                "measure_start": ev["measureStart"],
                "idx192_start": ev["idx192Start"],
                "measure_end": ev.get("measureEnd", ev["measureStart"]),
                "idx192_end": ev.get("idx192End", ev["idx192Start"]),
                "token": ev.get("tokenStart", ""),
            })
    return placed, pr


def report(label, placed, pr):
    pd = compute_placement_distribution_metrics(placed)
    n_total = len(placed)
    n_ln = sum(1 for e in placed if e.get("type") == "LN")
    bpm = pr.get("base_bpm", 130.0)
    print(f"\n=== {label}  (BPM {bpm}, key-lane events={n_total}, LN={n_ln}) ===")
    print(f"  lane_counts:    {pd['lane_counts']}")
    spread = max(pd['lane_counts']) / max(1, min(c for c in pd['lane_counts'] if c > 0))
    print(f"  lane spread (max/min>0): {spread:.2f}x")
    print(f"  right_share:    mean={pd['right_share_mean']:.3f}  "
          f"std={pd['right_share_std']:.3f}  n_measures={pd['right_share_n_measures']}")
    print(f"  hand_jump:      mean={pd['hand_jump_mean']:.2f}  "
          f"dist={pd['hand_jump_distribution']}")
    print(f"  same_hand_streak: mean={pd['same_hand_streak_mean']:.2f}  "
          f"dist={pd['same_hand_streak_distribution']}")
    streak_max = max((int(k) for k in pd['same_hand_streak_distribution'].keys()), default=0)
    print(f"  streak max:     {streak_max}")
    return pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", help="BMS file or folder")
    ap.add_argument("--folder", help="Folder; tool picks all .bms/.bme/.bml within")
    ap.add_argument("--batch", help="Glob pattern, e.g. '[*]*/*_SP*.bms'")
    args = ap.parse_args()

    targets = []
    if args.batch:
        targets = sorted(glob.glob(args.batch))
    elif args.folder:
        for f in os.listdir(args.folder):
            if f.lower().endswith((".bms", ".bme", ".bml")):
                targets.append(os.path.join(args.folder, f))
        targets.sort()
    elif args.path:
        if os.path.isdir(args.path):
            for f in os.listdir(args.path):
                if f.lower().endswith((".bms", ".bme", ".bml")):
                    targets.append(os.path.join(args.path, f))
            targets.sort()
        else:
            targets = [args.path]
    else:
        ap.error("provide a path, --folder, or --batch")

    summaries = []
    for path in targets:
        try:
            placed, pr = source_to_placed_events(path)
            if len(placed) < 50:
                continue  # skip near-empty charts
            label = os.path.relpath(path, ROOT)
            pd = report(label, placed, pr)
            summaries.append((label, pd))
        except Exception as ex:
            print(f"  {path}: ERROR {ex}")

    # Aggregate summary line
    if len(summaries) > 1:
        print("\n=== SUMMARY (chart count: %d) ===" % len(summaries))
        print(f"{'chart':<70s} {'streak_mean':>11s} {'streak_max':>10s} {'jump_mean':>9s} {'rs_std':>6s}")
        for label, pd in summaries:
            streak_max = max((int(k) for k in pd['same_hand_streak_distribution'].keys()), default=0)
            print(f"{label[-70:]:<70s} {pd['same_hand_streak_mean']:>11.2f} "
                  f"{streak_max:>10d} {pd['hand_jump_mean']:>9.2f} {pd['right_share_std']:>6.3f}")


if __name__ == "__main__":
    main()
