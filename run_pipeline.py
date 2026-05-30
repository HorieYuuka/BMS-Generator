#!/usr/bin/env python3
"""
run_pipeline.py
Single entry point for the BMS generation pipeline.

Execution order:
  1. mix_generation    -> token_analysis.json, mix_generation_log.json
  2. placement_engine  -> placement_result.json
  3. bms_writer        -> placement_result.bms
  4. similarity_check  -> similarity_report.json
"""

import argparse
import json
import os
import sys
import traceback

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


TOTAL_STEPS = 4


def _run_step(step_num, step_name, fn):
    print(f"[{step_num}/{TOTAL_STEPS}] {step_name} 시작...")
    try:
        result = fn()
    except SystemExit as e:
        print(f"[{step_num}/{TOTAL_STEPS}] {step_name} 실패 (exit code: {e.code})")
        sys.exit(1)
    except Exception:
        print(f"[{step_num}/{TOTAL_STEPS}] {step_name} 실패")
        traceback.print_exc()
        sys.exit(1)
    return result


def main():
    ap = argparse.ArgumentParser(description="BMS Generation Pipeline")
    ap.add_argument("--zip", dest="zip_path", help="Path to BMS zip archive")
    ap.add_argument("--folder", help="Path to extracted BMS folder")
    ap.add_argument("--bms", dest="bms_filename", default=None,
                    help="Explicitly select a BMS file within the package (filename only)")
    ap.add_argument("--intensity", type=int, default=5,
                    help="Note placement aggressiveness 1~20 (default: 5)")
    ap.add_argument("--scratch", type=int, default=5,
                    help="Scratch frequency 1~20 (default: 5)")
    ap.add_argument("--ln", action="store_true", default=False,
                    help="Enable LN post-processing")
    ap.add_argument("--ml", action="store_true", default=False,
                    help="Enable ML soft-ranking integration")
    ap.add_argument("--model-token", default=None,
                    help="TokenSelectionModel TorchScript path (.pt)")
    ap.add_argument("--model-lane", default=None,
                    help="LaneAssignmentModel TorchScript path (.pt)")
    ap.add_argument("--seed", default=None,
                    help="Placement seed: integer or 'random' (default: 42; "
                         "in resume mode the state's seed takes precedence — passing --seed there errors)")
    # v12 §23 Resume API (pass-through to placement_engine; skips Steps 3-4
    # because BMS write / similarity check don't apply to partial output).
    ap.add_argument("--resume-state", default=None,
                    help="Resume API: input state JSON path (§23.3)")
    ap.add_argument("--start-measure", type=int, default=None,
                    help="Resume API: start measure M (0-based)")
    ap.add_argument("--end-measure", type=int, default=None,
                    help="Resume API: end measure N")
    ap.add_argument("--finalize", default=None,
                    help="Finalize API: events JSON path; runs post-processing only (§23.6)")
    ap.add_argument("--next-chord-lookahead", default=None,
                    help="Boundary lookahead (§23.7): N+1 first-chord JSON path. Requires --resume-state.")
    ap.add_argument("--dp", action="store_true", default=False,
                    help="DP synthesis: produce a 14-key+2-scratch chart from the SP pool.")
    ap.add_argument("--dp-split", default="auto", choices=["auto", "timbre", "balance"],
                    help="DP hand-split strategy (auto|timbre|balance). Default auto.")
    args = ap.parse_args()

    if bool(args.zip_path) == bool(args.folder):
        print("ERROR: --zip 또는 --folder 중 하나만 지정하세요.", file=sys.stderr)
        sys.exit(1)

    if args.ml and (not args.model_token or not args.model_lane):
        print("ERROR: --ml 사용 시 --model-token과 --model-lane 모두 필요합니다.", file=sys.stderr)
        sys.exit(1)

    if args.resume_state and args.finalize:
        print("ERROR: --resume-state 와 --finalize 는 동시 사용 불가 (v12 §23.2).", file=sys.stderr)
        sys.exit(1)
    if args.resume_state and (args.start_measure is None or args.end_measure is None):
        print("ERROR: --resume-state 사용 시 --start-measure 와 --end-measure 필요.", file=sys.stderr)
        sys.exit(1)
    resume_mode = bool(args.resume_state)
    finalize_mode = bool(args.finalize)

    seed_explicit = args.seed is not None
    if not seed_explicit:
        seed_val = 42
    elif args.seed == "random":
        seed_val = "random"
    else:
        try:
            seed_val = int(args.seed)
        except ValueError:
            print("ERROR: --seed는 정수 또는 'random' 이어야 합니다.", file=sys.stderr)
            sys.exit(1)

    # Pre-load resume / finalize JSON inputs in the outer scope so step2's
    # closure only reads them — avoids Python local-binding issues when seed_val
    # is reassigned inside step2.
    resume_state_data = None
    finalize_events_data = None
    if resume_mode:
        with open(os.path.abspath(args.resume_state), "r", encoding="utf-8") as f:
            resume_state_data = json.load(f)
        # v12 §23.4 D.4: state seed is authoritative; reject CLI --seed to
        # prevent silent override.
        if seed_explicit:
            print("ERROR: --seed 와 --resume-state 동시 사용 불가 — state seed "
                  "(rng.seed) 가 우선입니다 (§23.4 β-1).", file=sys.stderr)
            sys.exit(1)
        state_seed = (resume_state_data.get("rng") or {}).get("seed")
        if isinstance(state_seed, int):
            seed_val = state_seed
    next_chord_lookahead_data = None
    if args.next_chord_lookahead:
        if not resume_mode:
            print("ERROR: --next-chord-lookahead 는 --resume-state 필요 (§23.7).",
                  file=sys.stderr)
            sys.exit(1)
        with open(os.path.abspath(args.next_chord_lookahead), "r", encoding="utf-8") as f:
            la_raw = json.load(f)
        import placement_engine as _pe
        try:
            next_chord_lookahead_data = _pe.normalize_lookahead(la_raw)
        except ValueError as e:
            print(f"ERROR: --next-chord-lookahead invalid: {e}", file=sys.stderr)
            sys.exit(1)
    if finalize_mode:
        with open(os.path.abspath(args.finalize), "r", encoding="utf-8") as f:
            _data = json.load(f)
        if isinstance(_data, list):
            finalize_events_data = _data
        elif isinstance(_data, dict):
            if "placed" in _data:
                finalize_events_data = _data["placed"]
            elif "events" in _data:
                finalize_events_data = _data["events"]
            else:
                print("ERROR: --finalize JSON object 는 'placed' 또는 'events' 키 필요.",
                      file=sys.stderr)
                sys.exit(1)
            if not isinstance(finalize_events_data, list):
                print("ERROR: --finalize JSON 'placed'/'events' 값은 list 여야 함.",
                      file=sys.stderr)
                sys.exit(1)
        else:
            print("ERROR: --finalize JSON 은 list 또는 'placed'/'events' 객체.",
                  file=sys.stderr)
            sys.exit(1)

    zip_path = os.path.abspath(args.zip_path) if args.zip_path else None
    folder   = os.path.abspath(args.folder) if args.folder else None
    bms_filename = args.bms_filename
    output_dir = ROOT_DIR

    # ── Step 1: MixGeneration ─────────────────────────────────────────────
    def step1():
        import mix_generation
        return mix_generation.run(
            zip_path=zip_path,
            folder=folder,
            output_dir=output_dir,
            bms_filename=bms_filename,
        )

    mix_result = _run_step(1, "MixGeneration", step1)

    selected_path     = mix_result["run_log"]["chart"]["selected_path"]
    selected_basename = os.path.basename(selected_path)

    print(f"[1/{TOTAL_STEPS}] MixGeneration 완료 -> token_analysis.json, mix_generation_log.json")

    # ── Step 2: PlacementEngine ───────────────────────────────────────────
    def step2():
        import placement_engine

        # Patch module-level paths to match pipeline context
        if zip_path:
            placement_engine.ZIP_PATH = zip_path
        else:
            _sel = selected_path
            placement_engine.load_bms_bytes = lambda: open(_sel, "rb").read()

        placement_engine.TARGET_BMS     = selected_basename
        placement_engine.TOKEN_ANALYSIS = os.path.join(output_dir, "token_analysis.json")
        placement_engine.RESULT_PATH    = os.path.join(output_dir, "placement_result.json")

        placement_engine.main(
            intensity_level=args.intensity,
            scratch_level=args.scratch,
            enable_ln=args.ln,
            enable_ml=args.ml,
            model_token_path=args.model_token,
            model_lane_path=args.model_lane,
            seed=seed_val,
            resume_state=resume_state_data,
            start_measure=args.start_measure,
            end_measure=args.end_measure,
            finalize_input_events=finalize_events_data,
            next_chord_lookahead=next_chord_lookahead_data,
            dp=args.dp, dp_split=args.dp_split,
        )

    _run_step(2, "PlacementEngine", step2)
    print(f"[2/{TOTAL_STEPS}] PlacementEngine 완료 -> placement_result.json")

    # v12 §23: resume mode emits a partial schema (no full chart). BMSWriter /
    # SimilarityCheck don't apply — caller (BMS.Compare) splices and finalizes.
    if resume_mode:
        print(f"  Resume mode: skipping Steps 3-4 (partial result, finalize via "
              f"--finalize on splice).")
        return

    # ── Step 3: BMSWriter ─────────────────────────────────────────────────
    def step3():
        import bms_writer

        if zip_path:
            bms_writer.ZIP_PATH = zip_path
        else:
            _sel = selected_path
            bms_writer.load_source_bms = lambda: bms_writer._decode_bms(open(_sel, "rb").read())

        bms_writer.TARGET_BMS  = selected_basename
        bms_writer.RESULT_JSON = os.path.join(output_dir, "placement_result.json")
        bms_writer.OUTPUT_BMS  = os.path.join(output_dir, "placement_result.bms")

        bms_writer.main()

    _run_step(3, "BMSWriter", step3)
    print(f"[3/{TOTAL_STEPS}] BMSWriter 완료 -> placement_result.bms")

    # ── Step 4: SimilarityCheck ───────────────────────────────────────────
    def step4():
        import similarity_check

        similarity_check.REPORT_PATH = os.path.join(output_dir, "similarity_report.json")

        if zip_path:
            similarity_check.main.__wrapped_args__ = None
            # Reuse similarity_check internals directly
            import tempfile as _tf, shutil as _sh
            _tmp = _tf.mkdtemp(prefix="bms_sim_")
            import zipfile as _zf
            with _zf.ZipFile(zip_path, "r") as z:
                z.extractall(_tmp)
            _contents = os.listdir(_tmp)
            _wdir = (os.path.join(_tmp, _contents[0])
                     if len(_contents) == 1 and os.path.isdir(os.path.join(_tmp, _contents[0]))
                     else _tmp)

            # Copy placement_result.bms into the work dir so similarity_check finds it
            bms_out = os.path.join(output_dir, "placement_result.bms")
            if os.path.isfile(bms_out) and _wdir != output_dir:
                import shutil
                shutil.copy2(bms_out, os.path.join(_wdir, "placement_result.bms"))

            sys.argv = ["similarity_check", "--folder", _wdir]
            similarity_check.main()
            _sh.rmtree(_tmp, ignore_errors=True)
        else:
            # folder mode — placement_result.bms should already be in output_dir
            # If output_dir != folder, copy it
            bms_out = os.path.join(output_dir, "placement_result.bms")
            if folder and output_dir != folder and os.path.isfile(bms_out):
                import shutil
                shutil.copy2(bms_out, os.path.join(folder, "placement_result.bms"))
            sys.argv = ["similarity_check", "--folder", folder or output_dir]
            similarity_check.main()

    _run_step(4, "SimilarityCheck", step4)
    print(f"[4/{TOTAL_STEPS}] SimilarityCheck 완료 -> similarity_report.json")

    print("\nPipeline 완료.")


if __name__ == "__main__":
    main()
