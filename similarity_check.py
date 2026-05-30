#!/usr/bin/env python3
"""
similarity_check.py
Compare placement_result.bms against reference charts in a BMS package.
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from bms_parser import parse_bms

# ── Constants ─────────────────────────────────────────────────────────────────

PLAYABLE_CHANNELS = {"11", "12", "13", "14", "15", "16", "18", "19"}

SIMILARITY_WARNING_THRESHOLD = 0.90

OUTPUT_CHART = "placement_result.bms"

ROOT_DIR   = os.path.dirname(os.path.abspath(__file__))
REPORT_PATH = os.path.join(ROOT_DIR, "similarity_report.json")


# ── Fingerprint extraction ────────────────────────────────────────────────────

def extract_fingerprint(bms_bytes: bytes) -> set:
    """
    Extract playable note fingerprint from BMS data.
    Returns set of (measure, idx192, lane) tuples.
    """
    pr = parse_bms(bms_bytes)
    fp = set()
    for ev in pr["events"]:
        etype = ev.get("type")
        if etype == "Tap":
            ch = ev["rawChannel"]
            if ch in PLAYABLE_CHANNELS:
                fp.add((ev["measure"], ev["idx192"], ev["lane"]))
        elif etype == "Long":
            ch = ev["rawChannelStart"]
            if ch in PLAYABLE_CHANNELS:
                fp.add((ev["measureStart"], ev["idx192Start"], ev["lane"]))
    return fp


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="BMS Similarity Check")
    ap.add_argument("--zip", dest="zip_path", help="Path to BMS zip archive")
    ap.add_argument("--folder", help="Path to extracted BMS folder")
    args = ap.parse_args()

    if bool(args.zip_path) == bool(args.folder):
        print("ERROR: --zip or --folder (exactly one)", file=sys.stderr)
        sys.exit(1)

    # Locate working directory
    tmp_dir = None
    if args.zip_path:
        zip_path = os.path.abspath(args.zip_path)
        tmp_dir = tempfile.mkdtemp(prefix="bms_sim_")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmp_dir)
        contents = os.listdir(tmp_dir)
        work_dir = (
            os.path.join(tmp_dir, contents[0])
            if len(contents) == 1 and os.path.isdir(os.path.join(tmp_dir, contents[0]))
            else tmp_dir
        )
    else:
        work_dir = os.path.abspath(args.folder)

    # Find placement_result.bms — check work_dir first, then ROOT_DIR
    output_path = os.path.join(work_dir, OUTPUT_CHART)
    if not os.path.isfile(output_path):
        output_path = os.path.join(ROOT_DIR, OUTPUT_CHART)
    if not os.path.isfile(output_path):
        print(f"ERROR: {OUTPUT_CHART} not found", file=sys.stderr)
        if tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        sys.exit(1)

    # Extract output fingerprint
    parse_warnings = []
    try:
        with open(output_path, "rb") as f:
            output_fp = extract_fingerprint(f.read())
    except Exception as exc:
        print(f"ERROR: Failed to parse {OUTPUT_CHART}: {exc}", file=sys.stderr)
        if tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        sys.exit(1)

    output_note_count = len(output_fp)
    print(f"Output: {OUTPUT_CHART} ({output_note_count} notes)")

    # Collect reference files
    ref_files = []
    for root, _dirs, files in os.walk(work_dir):
        for name in files:
            if name.lower().endswith((".bms", ".bme", ".bml")):
                full = os.path.join(root, name)
                if os.path.basename(full) != OUTPUT_CHART:
                    ref_files.append(full)
    ref_files.sort()

    print(f"References: {len(ref_files)} files")

    # Compare
    results = []
    for ref_path in ref_files:
        ref_name = os.path.basename(ref_path)
        try:
            with open(ref_path, "rb") as f:
                ref_fp = extract_fingerprint(f.read())
        except Exception as exc:
            parse_warnings.append(f"Failed to parse {ref_name}: {exc}")
            continue

        common = len(output_fp & ref_fp)
        sim = round(common / output_note_count, 4) if output_note_count > 0 else 0.0

        results.append({
            "reference_file": ref_name,
            "reference_note_count": len(ref_fp),
            "common_notes": common,
            "similarity": sim,
            "warning": sim >= SIMILARITY_WARNING_THRESHOLD,
        })

    results.sort(key=lambda x: -x["similarity"])

    # Console output
    for r in results:
        tag = " *** WARNING ***" if r["warning"] else ""
        print(f"  {r['reference_file']:45s} sim={r['similarity']:.4f} "
              f"common={r['common_notes']}{tag}")

    warning_count = sum(1 for r in results if r["warning"])
    if warning_count:
        print(f"\n{warning_count} file(s) above similarity threshold "
              f"({SIMILARITY_WARNING_THRESHOLD})")

    # Write report
    report = {
        "output_chart": OUTPUT_CHART,
        "output_note_count": output_note_count,
        "similarity_warning_threshold": SIMILARITY_WARNING_THRESHOLD,
        "results": results,
        "parse_warnings": parse_warnings,
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\nsimilarity_report.json written")

    if tmp_dir:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
