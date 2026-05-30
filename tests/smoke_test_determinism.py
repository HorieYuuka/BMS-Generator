"""Multi-song determinism regression for samples/baseline_lv5/.

Re-runs the 6 baseline songs (RB + ML) under the default seed and compares
each output to the saved baseline byte-for-byte. Catches PYTHONHASHSEED-
dependent ordering and other regressions that break determinism, scoped
across the operational pipeline (mix → placement → bms → similarity).

Per-song cost: ~25s RB + ~25s ML. Total ~5 minutes when all songs ship.

Songs that are missing under source_packages/ are skipped (logged, not failed).
"""

import filecmp
import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "source_packages")
BASE = os.path.join(ROOT, "samples", "baseline_lv5")

SONGS = [
    ("bumblebee",   "[Neun_jack] Bumblebee(Hardtek_Refix)"),
    ("egosa",       "[ねこみりん feat.みゆ] 先天性エゴサ依存症候群"),
    ("lepontinia",  "[pr.s] Lepontinia"),
    ("signal",      "[Kuwagata] シグナルほっぴんぐ"),
    ("tsuramic",    "[RearDawn] T'Suramic"),
    ("wanwan",      "[ねこみりん feat. みゆ] ☆わんわんぷらねっつ☆ ～ちきゅうせーふくだいさくせん～"),
]
MODES = [
    ("rb", []),
    ("ml", ["--ml",
            "--model-token", "training/checkpoints/token_selection_model.pt",
            "--model-lane",  "training/checkpoints/lane_assignment_model.pt"]),
]


def _run(args):
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    proc = subprocess.run(
        [sys.executable, os.path.join(ROOT, "run_pipeline.py")] + args,
        cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8",
    )
    if proc.returncode != 0:
        print("STDOUT:", proc.stdout[-1000:])
        print("STDERR:", proc.stderr[-1000:])
        raise RuntimeError(f"pipeline failed (exit {proc.returncode})")


def main():
    fails = []
    skips = []
    for slug, folder in SONGS:
        pkg = os.path.join(SRC, folder)
        if not os.path.isdir(pkg):
            skips.append(slug)
            print(f"[SKIP] {slug}: package missing ({pkg})")
            continue
        for mode_name, extra_args in MODES:
            base_bms = os.path.join(BASE, f"sample_{slug}_lv5_{mode_name}.bms")
            base_json = os.path.join(BASE, f"sample_{slug}_lv5_{mode_name}.json")
            if not (os.path.isfile(base_bms) and os.path.isfile(base_json)):
                skips.append(f"{slug}_{mode_name}")
                print(f"[SKIP] {slug}/{mode_name}: baseline missing")
                continue
            print(f"[RUN ] {slug}/{mode_name}", flush=True)
            _run(["--folder", pkg, "--intensity", "5"] + extra_args)
            out_bms = os.path.join(ROOT, "placement_result.bms")
            out_json = os.path.join(ROOT, "placement_result.json")
            bms_ok = filecmp.cmp(out_bms, base_bms, shallow=False)
            json_ok = filecmp.cmp(out_json, base_json, shallow=False)
            if bms_ok and json_ok:
                print(f"       PASS")
            else:
                fails.append((slug, mode_name, bms_ok, json_ok))
                print(f"       FAIL (bms={bms_ok}, json={json_ok})")

    print("\n=== summary ===")
    print(f"  songs checked: {len(SONGS) - len(skips) // 2}")
    print(f"  skipped: {skips if skips else 'none'}")
    if fails:
        print(f"  FAIL: {fails}")
        return 1
    print(f"  all PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
