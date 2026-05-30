#!/usr/bin/env python3
"""Run lv5 ML+RB pipeline on 6 sample packages and collect outputs."""
import os
import shutil
import subprocess
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # BMS.Generator root
OUT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# (label, pkg_name, bms_filename, axis_profile)
SAMPLES = [
    ("bumblebee", "[Neun_jack] Bumblebee(Hardtek_Refix)",
     "bumblebee(Hardtek_Refix)_bombus.bms", "stream+stair+peak"),
    ("egosa", "[ねこみりん feat.みゆ] 先天性エゴサ依存症候群",
     "egs_00_Twitch.bme", "chord+distraction"),
    ("tsuramic", "[RearDawn] T'Suramic",
     "tsuramic_spnanami01.bms", "peak"),
    ("wanwan", "[ねこみりん feat. みゆ] ☆わんわんぷらねっつ☆ ～ちきゅうせーふくだいさくせん～",
     "wanwan_nanami01.bme", "pure_chord"),
    ("signal", "[Kuwagata] シグナルほっぴんぐ",
     "00_kuwagata_signal_[del].bms", "stair+ln"),
    ("lepontinia", "[pr.s] Lepontinia",
     "lepontinia_7k_tlp.bms", "balanced"),
]

ML_TOKEN = os.path.join(OUT_DIR, "training/checkpoints/token_selection_model.pt")
ML_LANE = os.path.join(OUT_DIR, "training/checkpoints/lane_assignment_model.pt")


def run_one(label, pkg, bms, mode):
    folder = os.path.join(BASE, pkg)
    print(f"\n>>> {label} / {mode} (bms={bms})")
    cmd = [
        sys.executable, "run_pipeline.py",
        "--folder", folder,
        "--bms", bms,
        "--intensity", "5",
        "--seed", "42",
    ]
    if mode == "ml":
        cmd.extend([
            "--ml",
            "--model-token", ML_TOKEN,
            "--model-lane", ML_LANE,
        ])
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    res = subprocess.run(cmd, env=env, cwd=OUT_DIR, capture_output=True, text=True, encoding="utf-8")
    if res.returncode != 0:
        print(f"  FAIL ({res.returncode})")
        print(res.stdout[-500:])
        print(res.stderr[-500:])
        return False
    # Show last few lines
    last = res.stdout.strip().split("\n")[-3:]
    for line in last:
        print(f"  {line}")
    # Save outputs
    for src, dst_suffix in [
        ("placement_result.bms", f"sample_{label}_lv5_{mode}.bms"),
        ("placement_result.json", f"sample_{label}_lv5_{mode}.json"),
    ]:
        src_path = os.path.join(OUT_DIR, src)
        if os.path.isfile(src_path):
            shutil.copy(src_path, os.path.join(OUT_DIR, dst_suffix))
    return True


def main():
    succeeded = 0
    failed = []
    for (label, pkg, bms, profile) in SAMPLES:
        for mode in ("ml", "rb"):
            ok = run_one(label, pkg, bms, mode)
            if ok:
                succeeded += 1
            else:
                failed.append((label, mode))
    print(f"\n\n=== Done: {succeeded} ok, {len(failed)} failed ===")
    if failed:
        print("Failed:")
        for f in failed:
            print(f"  {f}")


if __name__ == "__main__":
    main()
