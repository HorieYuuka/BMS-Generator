# BMS.Generator

[🇰🇷 한국어 README](README.ko.md) · [🇯🇵 日本語 README](README.ja.md)

A pipeline that automatically generates note placement for a BMS chart from the song's keysound pool.

## Requirements

- Python 3.10+
- ffmpeg (on PATH)
- numpy
- scipy (optional, STFT acceleration — falls back to numpy FFT if absent)
- torch (optional, required only for `--ml` mode — TorchScript model loading)

```bash
pip install -r requirements.txt   # numpy, scipy (core)
# torch is needed only for ML mode (--ml); install separately to match your CUDA build — see the comments in requirements.txt
```

ffmpeg is required for keysound decoding (it is an executable on PATH, not a pip package):

```bash
winget install ffmpeg      # Windows. Or choco install ffmpeg / https://ffmpeg.org/download.html
ffmpeg -version            # verify PATH registration (printing a version means OK)
```

## Input — a BMS package

The input is **one BMS song's folder** (or a zip of it): the `.bms` / `.bme` / `.bml` chart file(s) together with the `.wav` keysounds those charts reference. A BMS song download is usually already this shape.

- Pass either `--folder <dir>` or `--zip <zip>`
- If the folder has multiple charts, a 1P single chart is auto-selected by coverage. Override with `--bms <filename>`

## Quick start

```bash
python run_pipeline.py --folder "package_folder"
```

Or directly from a zip archive:

```bash
python run_pipeline.py --zip "package.zip"
```

Specify exactly one of `--zip` / `--folder`. On success the working directory contains `placement_result.bms` (your finished chart).

> **Windows note**: BMS package and keysound filenames frequently contain Korean / Japanese characters, and the default console encoding (cp949) will crash on them. Switch the encoding to UTF-8 before running:
> ```powershell
> $env:PYTHONIOENCODING="utf-8"      # PowerShell
> ```
> ```cmd
> set PYTHONIOENCODING=utf-8          REM cmd
> ```

### Options

| Option | Default | Description |
|------|--------|------|
| `--zip <path>` | - | BMS zip archive path |
| `--folder <path>` | - | unpacked BMS folder path |
| `--bms <filename>` | auto-select | explicit BMS file inside the package (filename only) |
| `--intensity <1-20>` | 5 | placement aggressiveness (1 = conservative, 20 = aggressive) |
| `--scratch <1-20>` | 5 | scratch multiplier — in primary mode, source per-measure × (level / 5) |
| `--ln` | off | enable LN (long-note) post-processing |
| `--ml` | off | enable ML model integration (see "ML mode" below) |
| `--model-token <path>` | - | TokenSelectionModel TorchScript (.pt) |
| `--model-lane <path>` | - | LaneAssignmentModel TorchScript (.pt) |

```bash
# example: aggressive placement + LN enabled
python run_pipeline.py --zip "package.zip" --intensity 8 --scratch 7 --ln

# example: explicit chart file + conservative placement
python run_pipeline.py --zip "package.zip" --bms "chart_HARD.bms" --intensity 3
```

## Pipeline stages

| Stage | Script | Input | Output |
|------|----------|------|------|
| 1 | mix_generation.py | BMS package (zip/folder) | token_analysis.json, mix_generation_log.json |
| 2 | placement_engine.py | token_analysis.json, source BMS | placement_result.json |
| 3 | bms_writer.py | placement_result.json, source BMS | placement_result.bms |
| 4 | similarity_check.py | placement_result.bms, package BMS files | similarity_report.json |

`bms_parser.py` is imported internally by every stage.

### Stage 1 — MixGeneration

- Auto-selects a 1P single-play chart from the package (`#PLAYER 1`)
- `--bms` skips the pre-filter and uses the specified file directly
- Selection priority: max WAV coverage > max playable event count > filename order
- Pre-filter removes `#PLAYER != 1` and charts ≥ 300 s long
- Decodes every keysound through ffmpeg and computes per-token audio statistics
- Output: `token_analysis.json` (per-token duration, attack_rms, attack_peak)

### Stage 2 — PlacementEngine

- Builds a global playable whitelist from the token pool
- Phase segmentation into 4-measure blocks (rush / normal / rest)
- Per-measure primitive selection: rush → ChordBurst, else → Stream
- Placement constraints: collision, jack prohibition, hand balance, same-hand alternation
- Scratch insertion: placed only at source ch16 event positions (primary seed / fallback / disabled)
- With `--ln`, duration-based LN post-processing promotes Tap to LN (LNOBJ mode)
- Output: `placement_result.json` (placed events, residual events, diagnostics, ln_meta)

### Stage 3 — BMSWriter

- Preserves the source's header, timing channels (02/03/08/09), and BGM (`#01`)
- Strips the source's playable channels (11–19, 21–29) and replaces them with placed events
- With LN: injects a `#LNOBJ` header, emits LN start/end tokens, auto-creates a silent WAV
- Appends residual events to `#01` (deduped against the source's existing `#01`)
- Output: `placement_result.bms`

### Stage 4 — SimilarityCheck

- Compares the generated chart against every reference chart in the package
- Fingerprint over playable channels `(measure, idx192, lane)`
- Warns when similarity ≥ 0.90
- Output: `similarity_report.json`

## Intensity / Scratch scaling

`--intensity` and `--scratch` are integers in 1–20 (default 5).

| Parameter | intensity=1 | intensity=5 | intensity=10 | intensity=20 |
|----------|-------------|-------------|--------------|--------------|
| WHITELIST_MIN_OCCURRENCE | 15 | 10 | 5 | 1 |
| WHITELIST_MIN_ATTACK_PERCENTILE | 30 | 20 | 10 | 5 |
| WHITELIST_DURATION_MAX | 700 ms | 1056 ms | 1500 ms | 2500 ms |
| STREAM_CHORD_RATIO_MAX | 0.20 | 0.31 | 0.45 | 0.65 |
| STREAM_MAX_SAME_HAND | 1 | 2 | 3 | 4 |
| MAX_CHORD_SIZE | 3 | 3 | 4 | 5 |

**Scratch (primary mode, updated 2026-05-03, v12 §12):** `--scratch` is a **multiplier** applied to the source's per-measure scratch density. `scratch=5` (default) → 1:1 source mirror. Minimum interval uses the source's own natural minimum (safety floor 4 ticks).

| Parameter | scratch=1 | scratch=5 | scratch=10 | scratch=20 |
|----------|-----------|-----------|------------|------------|
| budget_per_measure | 0.2× src | 1.0× src | 2.0× src | 4.0× src |
| min_interval | source min (≥ 4 ticks) | (same) | (same) | (same) |

If the source has no scratch tokens, the pipeline falls back to a synthesized mode using the legacy v9 absolute lerp table (v12 §12.7).

Intermediate values are linearly interpolated.

## LN mode

`--ln` runs LN post-processing after placement:

- Collect Tap events whose token duration ≥ 200 ms as LN candidates
- Sort by duration descending and try to promote the longest first
- Promotion conditions: no interior/end collision on the same lane, LN ratio ≤ 30%
- Auto-allocates an LNOBJ token (an unused WAV slot)
- BMSWriter emits a `#LNOBJ` header and start/end tokens

## ML mode

`--ml` swaps two rule-based decisions for ML models (soft re-ranker only — hard constraints still apply):

- **TokenSelectionModel** — per-measure candidate token ordering (re-ranks on top of RB §6 base order + §6.1 within-idx reorder)
- **LaneAssignmentModel** — per-event lane priority (replaces centroid-based RB lane assignment; v9 fisher_yates is no longer the RB default)

```bash
python placement_engine.py --ml \
  --model-token token_selection_model.pt \
  --model-lane  lane_assignment_model.pt
```

- Models are PyTorch TorchScript (`torch.jit.save`), loaded on CPU
- On a model-load or inference failure, the call falls back to the rule path automatically (granularity: per measure / per event)
- Without `--ml`, the pipeline runs even if torch is not installed (lazy import)
- For training-data labeling see `data_labeling.py`; the input tensor schema is documented in the technical report under `docs/`

## Data labeling (for model training)

`data_labeling.py` generates TokenSelectionModel / LaneAssignmentModel training datasets (JSONL) from a package of human BMS charts.

```bash
python data_labeling.py --dataset <package_root> --output <output_dir> --config labeling_config.json
```

- Discovers per-package → each package's `token_analysis.json` must exist beforehand
- Cache: SHA-256 over token_analysis + BMS file + config
- Outputs: `token_selection_dataset.jsonl`, `lane_assignment_dataset.jsonl`, `package_pools.json`, `labeling_run_log.json`
- At scale (thousands of packages): expect ≥ 100 GB disk; streaming output is already used

## Model training (how to make the ML models, for reproduction)

> ML is **frozen** — it shows no measurable advantage over RB and is non-recommended for operation. The flow below is for reproduction / retraining. Full background and training setup are in `docs/bms-generator-pipeline.en.md` §8.

End-to-end flow: **labeling → train → checkpoint → `--ml`**. The training inputs are the outputs of the "Data labeling" section above (`*_dataset.jsonl` + `package_pools.json` + `labeling_run_log.json`).

```bash
# 1. TokenSelectionModel (BCE)
python -m training.train_token \
  --dataset <out>/token_selection_dataset.jsonl \
  --pools   <out>/package_pools.json \
  --run-log <out>/labeling_run_log.json \
  --output  training/checkpoints

# 2. LaneAssignmentModel (class-weighted CE — recommended to correct the lane imbalance)
python -m training.train_lane \
  --dataset <out>/lane_assignment_dataset.jsonl \
  --pools   <out>/package_pools.json \
  --run-log <out>/labeling_run_log.json \
  --output  training/checkpoints \
  --class-weights auto --class-weight-power 2.0

# 3. Use: point --model-token / --model-lane (in "ML mode" above) at the trained .pt files
python run_pipeline.py --folder "package_folder" --ml \
  --model-token training/checkpoints/token_selection_model.pt \
  --model-lane  training/checkpoints/lane_assignment_model.pt
```

- Models are exported as TorchScript via `torch.jit.script` and loaded on CPU at inference (no GPU needed for generation)
- Training benefits from a GPU. CUDA build: Python 3.13 + GTX 1070 → **cu118** (see the comments in `requirements.txt`)
- `train_token` defaults to 50 epochs, `train_lane` to 20; both use early stopping + a package-level train/val split (`seed=42`)

## Running individual stages

You can also run each stage standalone, without the pipeline driver:

```bash
# stage 1 only
python mix_generation.py --zip "package.zip"
python mix_generation.py --zip "package.zip" --bms "chart.bms"

# stage 2 only (token_analysis.json must already exist)
python placement_engine.py
python placement_engine.py --intensity 8 --scratch 3 --ln

# stage 3 only (placement_result.json must already exist)
python bms_writer.py

# similarity check only
python similarity_check.py --zip "package.zip"
```

## Output files

| File | Description |
|------|------|
| token_analysis.json | per-token audio analysis (duration, attack RMS/peak, decode success) |
| mix_generation_log.json | chart-selection log, coverage, token diagnostics |
| placement_result.json | placement result (placed + residual events + diagnostics + ln_meta) |
| placement_result.bms | the final BMS chart file |
| similarity_report.json | similarity report against every reference chart in the package |
| lnobj_silent.wav | silent WAV (44 bytes) auto-generated when LN mode is on |

## Design documents

The full pipeline design — BMS parsing and channel semantics; chart/token
selection and spectral analysis; the note-placement policy (whitelist, phrasing,
primitives, constraints, scratch, LN, the Resume API); ML integration and freeze;
data labeling; model architecture; output/header policy; similarity checking; and
note-attribute definitions — is described in the technical report under `docs/`.

### Technical report (`docs/`)

- `docs/bms-generator-pipeline.en.md` — a full pipeline technical report (design / Resume API / validation / ML training & freeze / limitations). Korean: `docs/bms-generator-pipeline.ko.md`.

### Tests (`tests/`)

- `tests/smoke_test_resume.py` — 9 Resume API cases (base split / M=0 / last-measure / cascade / ML rejection / schema·rng mismatch / lookahead)
- `tests/smoke_test_determinism.py` — regression: 6 songs × {RB, ML} fresh runs are byte-identical to `samples/baseline_lv5/`
