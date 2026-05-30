"""Batch runner for mix_generation.py over many BMS package folders.

Runs mix_generation as a subprocess per package so a single crash does not
kill the whole batch. Writes a JSON summary log with per-package status.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def find_packages(root: str) -> list:
    entries = []
    for name in os.listdir(root):
        path = os.path.join(root, name)
        if os.path.isdir(path):
            entries.append((name, path))
    entries.sort(key=lambda e: e[0])
    return entries


def has_any_bms(folder: str) -> bool:
    for name in os.listdir(folder):
        if name.lower().endswith((".bms", ".bme", ".bml")):
            return True
    return False


def run_one(mix_script: str, pkg_folder: str, timeout_sec: int) -> dict:
    start = time.time()
    cmd = [
        sys.executable,
        mix_script,
        "--folder", pkg_folder,
        "--output-dir", pkg_folder,
    ]
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_sec,
            env=env,
        )
        elapsed = time.time() - start
        token_path = os.path.join(pkg_folder, "token_analysis.json")
        produced = os.path.exists(token_path)
        return {
            "ok": proc.returncode == 0 and produced,
            "returncode": proc.returncode,
            "elapsed_sec": round(elapsed, 2),
            "produced_token_analysis": produced,
            "stderr_tail": proc.stderr[-500:] if proc.stderr else "",
        }
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "returncode": None,
            "elapsed_sec": round(time.time() - start, 2),
            "produced_token_analysis": False,
            "stderr_tail": f"TIMEOUT after {timeout_sec}s",
        }
    except Exception as e:
        return {
            "ok": False,
            "returncode": None,
            "elapsed_sec": round(time.time() - start, 2),
            "produced_token_analysis": False,
            "stderr_tail": f"EXCEPTION: {type(e).__name__}: {e}",
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Folder containing package folders")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--timeout", type=int, default=600, help="Per-package timeout seconds")
    ap.add_argument("--log", default=None, help="Output log JSON path")
    args = ap.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    mix_script = os.path.join(repo_root, "mix_generation.py")
    if not os.path.exists(mix_script):
        sys.exit(f"mix_generation.py not found at {mix_script}")

    log_path = args.log or os.path.join(
        repo_root, "tools", f"batch_mixgen_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )

    all_pkgs = find_packages(args.root)
    selected = all_pkgs[args.offset : args.offset + args.limit]

    print(f"Total packages discovered: {len(all_pkgs)}")
    print(f"Selected window: offset={args.offset}, limit={args.limit}, actual={len(selected)}")
    print(f"Log path: {log_path}")
    print()

    results = []
    skipped_no_bms = 0
    success = 0
    failed = 0
    batch_start = time.time()

    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    def flush_log():
        snapshot = {
            "started_at": datetime.now(timezone.utc).isoformat(),
            "root": args.root,
            "offset": args.offset,
            "limit": args.limit,
            "total_discovered": len(all_pkgs),
            "selected": len(selected),
            "success": success,
            "failed": failed,
            "skipped_no_bms": skipped_no_bms,
            "batch_elapsed_sec": round(time.time() - batch_start, 2),
            "in_progress": True,
            "results": results,
        }
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, ensure_ascii=False, indent=2)

    for i, (name, path) in enumerate(selected, 1):
        if not has_any_bms(path):
            skipped_no_bms += 1
            results.append({
                "index": i,
                "package": name,
                "skipped": "no_bms_files",
            })
            print(f"[{i}/{len(selected)}] SKIP (no .bms): {name}")
            continue

        result = run_one(mix_script, path, args.timeout)
        result["index"] = i
        result["package"] = name
        results.append(result)

        if result["ok"]:
            success += 1
            print(f"[{i}/{len(selected)}] OK   ({result['elapsed_sec']}s): {name}", flush=True)
        else:
            failed += 1
            tail = result.get("stderr_tail", "").replace("\n", " | ")[:200]
            print(f"[{i}/{len(selected)}] FAIL ({result['elapsed_sec']}s): {name} :: {tail}", flush=True)
        flush_log()

    batch_elapsed = time.time() - batch_start

    summary = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "root": args.root,
        "offset": args.offset,
        "limit": args.limit,
        "total_discovered": len(all_pkgs),
        "selected": len(selected),
        "success": success,
        "failed": failed,
        "skipped_no_bms": skipped_no_bms,
        "batch_elapsed_sec": round(batch_elapsed, 2),
        "in_progress": False,
        "results": results,
    }

    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print()
    print("=" * 60)
    print(f"Done in {batch_elapsed:.1f}s")
    print(f"Success: {success} / Failed: {failed} / Skipped(no bms): {skipped_no_bms}")
    print(f"Log written: {log_path}")


if __name__ == "__main__":
    main()
