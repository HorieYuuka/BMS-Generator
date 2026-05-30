"""
mix_generation.py — BMS token analysis
Implements mix generation: source-chart selection and per-token analysis.
Responsibilities: BMS file selection (§0.3) + token analysis (§4).
WAV mix rendering has been removed from this pipeline.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import scipy.signal as _scipy_signal
    _HAVE_SCIPY = True
except ImportError:
    _scipy_signal = None
    _HAVE_SCIPY = False

sys.path.insert(0, str(Path(__file__).parent))
from bms_parser import parse_bms

# ---------------------------------------------------------------------------
# Constants / tunables
# ---------------------------------------------------------------------------

SAMPLE_RATE = 44100
DECODE_WORKERS = 8
ATTACK_WINDOW_MS = 50.0

# Spectral feature hyperparameters
SPECTRAL_FFT_SIZE = 2048
SPECTRAL_HOP_SIZE = 512
SPECTRAL_WINDOW = "hann"
LOW_FREQ_THRESHOLD_HZ = 300

_SPECTRAL_FALLBACK_WARNED = False

# §2.1 token discovery channels
_BGM_CH = "01"
_P1_VISIBLE = frozenset({"11", "12", "13", "14", "15", "16", "17", "18", "19"})

AUDIO_EXTENSIONS = [".ogg", ".wav", ".mp3", ".flac", ".aiff"]

CHART_DURATION_MAX_SECONDS = 300  # 5 minutes


# ---------------------------------------------------------------------------
# §0.1  Dependency check
# ---------------------------------------------------------------------------

def check_dependencies() -> None:
    proc = subprocess.run(["ffmpeg", "-version"], capture_output=True)
    if proc.returncode != 0:
        sys.exit("ERROR: ffmpeg not found. Please install ffmpeg.")
    try:
        import numpy  # noqa: F401
    except ImportError:
        sys.exit("ERROR: numpy not found. Please install numpy.")


# ---------------------------------------------------------------------------
# WAV decode cache helpers
# ---------------------------------------------------------------------------

def compute_wav_hash(file_path: str) -> str:
    """SHA-256 of raw file bytes. Returns 'sha256:<hexdigest>'."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return f"sha256:{h.hexdigest()}"


def load_token_cache(cache_path: str) -> dict:
    """Load existing token_analysis.json as {token: entry} dict. Empty on failure."""
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            entries = json.load(f)
        return {e["token"]: e for e in entries if "token" in e}
    except Exception:
        return {}


def is_cache_valid(cached_entry: dict, wav_file: str, file_path: Optional[str]) -> bool:
    """Cache hit if wav_file matches, decode_ok, and SHA-256 unchanged."""
    if not cached_entry:
        return False
    if cached_entry.get("wav_file") != wav_file:
        return False
    if not cached_entry.get("decode_ok", False):
        return False
    if file_path is None:
        return False
    cached_hash = cached_entry.get("wav_hash")
    if not cached_hash:
        return False
    # Spectral addendum: pre-spectral cached entries must be recomputed.
    if "spectral_centroid_mean" not in cached_entry:
        return False
    try:
        current_hash = compute_wav_hash(file_path)
    except Exception:
        return False
    return current_hash == cached_hash


# ---------------------------------------------------------------------------
# §0.3  BMS file selection
# ---------------------------------------------------------------------------

def _find_bms_files(folder: str) -> List[str]:
    result = []
    for root, _dirs, files in os.walk(folder):
        for name in files:
            if name.lower().endswith((".bms", ".bme", ".bml")):
                if name.lower().startswith("placement_result"):
                    continue
                result.append(os.path.join(root, name))
    return sorted(result)


def _selection_metrics(pr: dict) -> Tuple[float, int, int]:
    """
    Returns (used_wav_coverage, playable_count_11_19, scratch_event_count_16).
    used_wav_coverage = |used_tokens ∩ declared_wavs| / |declared_wavs|
    playable_count    = Tap+Long events on channels 11-19 only (§0.3.2 note)
    scratch_events    = Tap+Long on channel 16 (used as tie-breaker preference)
    """
    headers = pr["headers"]
    events = pr["events"]

    declared: set = {k[3:] for k in headers if k.startswith("WAV") and len(k) > 3}

    used: set = set()
    playable = 0
    scratch = 0
    for ev in events:
        t = ev.get("token") or ev.get("tokenStart")
        if t and t != "00":
            used.add(t)
        if ev["type"] in ("Tap", "Long"):
            ch = ev.get("rawChannel") or ev.get("rawChannelStart", "")
            if ch in _P1_VISIBLE:
                playable += 1
            if ch == "16":
                scratch += 1

    coverage = round(len(used & declared) / len(declared), 6) if declared else 0.0
    return coverage, playable, scratch


def _estimate_duration(pr: dict) -> float:
    """
    Estimate chart duration in seconds.
    Each measure: (4 * measure_scale / BPM_at_measure_start) * 60
    """
    events = pr["events"]
    base_bpm = pr.get("base_bpm", 130.0)
    measure_scale = pr.get("measure_scale", {})

    max_measure = 0
    for ev in events:
        m = ev.get("measure", ev.get("measureStart", 0))
        if m > max_measure:
            max_measure = m

    # Collect BPM changes sorted by tkey
    bpm_events = []
    for ev in events:
        if ev["type"] == "BPMChange":
            tkey = ev["measure"] * 192 + ev["idx192"]
            bpm_events.append((tkey, ev["bpm"]))
    bpm_events.sort()

    total = 0.0
    current_bpm = base_bpm
    bpm_idx = 0

    for measure in range(max_measure + 1):
        start_tkey = measure * 192
        while bpm_idx < len(bpm_events) and bpm_events[bpm_idx][0] <= start_tkey:
            current_bpm = bpm_events[bpm_idx][1]
            bpm_idx += 1
        scale = measure_scale.get(measure, 1.0)
        if current_bpm > 0:
            total += (4.0 * scale / current_bpm) * 60.0

    return round(total, 2)


def select_bms_file(
    candidates: List[str], run_warnings: List[str]
) -> Tuple[str, dict, dict]:
    """
    §0.3.1 pre-filter (#PLAYER + duration) then §0.3.2 priority selection.
    Returns (selected_path, parse_result, selection_info_for_log).
    """
    all_info: List[dict] = []
    scored: List[dict] = []

    for path in candidates:
        try:
            with open(path, "rb") as f:
                data = f.read()
            pr = parse_bms(data)
            player = pr["headers"].get("PLAYER", "").strip()
            coverage, playable, scratch = _selection_metrics(pr)
            dur = _estimate_duration(pr)

            entry = {
                "path": path, "coverage": coverage, "playable_count": playable,
                "scratch_events": scratch,
                "player": player, "pr": pr, "estimated_duration": dur,
                "prefilter_passed": True, "prefilter_discard_reason": None,
            }

            # §0.3.1 #PLAYER filter
            if player != "1":
                reason = "#PLAYER absent" if player == "" else f"#PLAYER={player}"
                entry["prefilter_passed"] = False
                entry["prefilter_discard_reason"] = reason
                all_info.append(entry)
                continue

            # Duration filter
            if dur >= CHART_DURATION_MAX_SECONDS:
                reason = f"duration >= {CHART_DURATION_MAX_SECONDS}s"
                entry["prefilter_passed"] = False
                entry["prefilter_discard_reason"] = reason
                all_info.append(entry)
                continue

            all_info.append(entry)
            scored.append(entry)

        except Exception as exc:
            run_warnings.append(f"Failed to parse {path}: {exc}")

    discarded = [e for e in all_info if not e["prefilter_passed"]]
    if discarded:
        run_warnings.append(
            f"Pre-filter removed {len(discarded)} file(s): "
            + "; ".join(
                f"{os.path.basename(d['path'])} ({d['prefilter_discard_reason']})"
                for d in discarded
            )
        )

    if not scored:
        sys.exit(
            "ERROR: No BMS files passed pre-filters. "
            f"Discarded: {[os.path.basename(d['path']) for d in discarded]}"
        )

    # §0.3.2 steps 2-6
    qualifying = [c for c in scored if c["coverage"] >= 0.95]
    threshold_bypassed = False
    if not qualifying:
        run_warnings.append(
            "No file meets 0.95 WAV coverage threshold; using best available"
        )
        threshold_bypassed = True
        qualifying = scored

    qualifying.sort(key=lambda c: (-c["coverage"], -c["scratch_events"],
                                   -c["playable_count"], c["path"]))
    sel = qualifying[0]

    info = {
        "all_candidates": [
            {"path": c["path"], "coverage": c["coverage"],
             "playable_count": c["playable_count"],
             "scratch_events": c["scratch_events"],
             "player": c["player"],
             "estimated_duration_seconds": c["estimated_duration"],
             "prefilter_passed": c["prefilter_passed"],
             "prefilter_discard_reason": c["prefilter_discard_reason"]}
            for c in all_info
        ],
        "prefilter_discarded": [
            {"path": d["path"], "reason": d["prefilter_discard_reason"]}
            for d in discarded
        ],
        "selected_path": sel["path"],
        "selection_reason": "highest Used WAV Coverage; tie-break prefers scratch_events"
        + (" (threshold bypassed)" if threshold_bypassed else ""),
        "threshold_met": not threshold_bypassed,
        "coverage": sel["coverage"],
        "playable_count": sel["playable_count"],
        "scratch_events": sel["scratch_events"],
        "player": sel["player"],
        "estimated_duration_seconds": sel["estimated_duration"],
    }
    return sel["path"], sel["pr"], info


# ---------------------------------------------------------------------------
# §4.2  Audio decode — mono
# ---------------------------------------------------------------------------

def _find_audio_file(folder: str, wav_filename: str) -> Optional[str]:
    """Resolve #WAVxx filename with extension fallback."""
    stem = Path(wav_filename).stem
    for ext in [Path(wav_filename).suffix] + AUDIO_EXTENSIONS:
        candidate = Path(folder) / (stem + ext)
        if candidate.exists():
            return str(candidate)
    return None


def _decode_audio_mono(
    file_path: str, sample_rate: int = SAMPLE_RATE
) -> Optional[np.ndarray]:
    """
    Decode audio to float32, resample, convert to mono = (L+R)/2.
    Returns 1-D float32 ndarray or None on failure.
    """
    cmd = [
        "ffmpeg", "-loglevel", "error",
        "-i", file_path,
        "-f", "f32le", "-acodec", "pcm_f32le",
        "-ar", str(sample_rate), "-ac", "2",
        "-",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=60)
        if proc.returncode != 0 or not proc.stdout:
            return None
        stereo = np.frombuffer(proc.stdout, dtype=np.float32).copy().reshape(-1, 2)
        return (stereo[:, 0] + stereo[:, 1]) / 2.0  # §4.2: average channels
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Spectral feature computation
# ---------------------------------------------------------------------------

def _zero_spectral() -> dict:
    return {
        "spectral_centroid_mean": 0.0,
        "spectral_centroid_std": 0.0,
        "spectral_flatness_mean": 0.0,
        "spectral_flatness_std": 0.0,
        "low_freq_energy_ratio": 0.0,
        "zero_crossing_rate_mean": 0.0,
        "zero_crossing_rate_std": 0.0,
    }


def _frame_signal(signal: np.ndarray, frame_size: int, hop_size: int) -> np.ndarray:
    n = len(signal)
    if n < frame_size:
        padded = np.zeros(frame_size, dtype=signal.dtype)
        padded[:n] = signal
        return padded[None, :]
    num_frames = 1 + (n - frame_size) // hop_size
    idx = np.arange(frame_size)[None, :] + hop_size * np.arange(num_frames)[:, None]
    return signal[idx]


def _stft_magnitude(
    mono: np.ndarray, frames: np.ndarray, fft_size: int, hop_size: int, sample_rate: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (magnitude[n_frames, n_freq], freqs[n_freq]).
    Prefers scipy.signal.stft; falls back to numpy FFT with hann window.
    """
    global _SPECTRAL_FALLBACK_WARNED
    freqs = np.fft.rfftfreq(fft_size, d=1.0 / sample_rate)
    # scipy.signal.stft shortens nperseg for signals shorter than fft_size,
    # which would change the freq bin count. Use the manual numpy path instead
    # so the output is always (n_frames, fft_size//2 + 1).
    if _HAVE_SCIPY and len(mono) >= fft_size:
        try:
            _, _, zxx = _scipy_signal.stft(
                mono,
                fs=sample_rate,
                window=SPECTRAL_WINDOW,
                nperseg=fft_size,
                noverlap=fft_size - hop_size,
                boundary=None,
                padded=False,
                return_onesided=True,
            )
            mag = np.abs(zxx).T  # (n_frames, n_freq)
            if mag.shape[0] > 0 and mag.shape[1] == freqs.shape[0]:
                return mag, freqs
        except Exception:
            pass
    if not _HAVE_SCIPY and not _SPECTRAL_FALLBACK_WARNED:
        print("WARNING: scipy not available — using numpy FFT fallback for spectral metrics")
        _SPECTRAL_FALLBACK_WARNED = True
    window = np.hanning(fft_size).astype(np.float32)
    spec = np.fft.rfft(frames * window, axis=1)
    return np.abs(spec), freqs


def _compute_spectral_metrics(mono: np.ndarray, sample_rate: int) -> dict:
    n = len(mono)
    if n == 0:
        return _zero_spectral()

    fft_size = SPECTRAL_FFT_SIZE
    hop_size = SPECTRAL_HOP_SIZE

    frames = _frame_signal(mono, fft_size, hop_size)

    # Zero crossing rate per frame (raw frames, no window)
    signs = np.signbit(frames).astype(np.int8)
    sign_changes = np.abs(np.diff(signs, axis=1)).sum(axis=1)
    zcr = sign_changes.astype(np.float64) / fft_size

    # STFT magnitude
    mag, freqs = _stft_magnitude(mono, frames, fft_size, hop_size, sample_rate)

    # Spectral centroid per frame
    mag_sum = mag.sum(axis=1)
    centroid = np.zeros(mag.shape[0], dtype=np.float64)
    nz = mag_sum > 0
    if nz.any():
        centroid[nz] = (mag[nz] * freqs[None, :]).sum(axis=1) / mag_sum[nz]

    # Spectral flatness per frame
    arith = mag.mean(axis=1)
    log_mag = np.log(np.maximum(mag, 1e-20))
    geo = np.exp(log_mag.mean(axis=1))
    flatness = np.zeros(mag.shape[0], dtype=np.float64)
    nz_a = arith > 0
    if nz_a.any():
        flatness[nz_a] = geo[nz_a] / arith[nz_a]
    flatness = np.clip(flatness, 0.0, 1.0)

    # Low frequency energy ratio (single FFT of full signal)
    spec_full = np.fft.rfft(mono)
    power = spec_full.real ** 2 + spec_full.imag ** 2
    freqs_full = np.fft.rfftfreq(n, d=1.0 / sample_rate)
    total_power = float(power.sum())
    if total_power > 0:
        low_ratio = float(power[freqs_full < LOW_FREQ_THRESHOLD_HZ].sum() / total_power)
    else:
        low_ratio = 0.0

    return {
        "spectral_centroid_mean": round(float(centroid.mean()), 4),
        "spectral_centroid_std": round(float(centroid.std()), 4),
        "spectral_flatness_mean": round(float(flatness.mean()), 6),
        "spectral_flatness_std": round(float(flatness.std()), 6),
        "low_freq_energy_ratio": round(float(low_ratio), 6),
        "zero_crossing_rate_mean": round(float(zcr.mean()), 6),
        "zero_crossing_rate_std": round(float(zcr.std()), 6),
    }


# ---------------------------------------------------------------------------
# §4.3  Per-token metrics
# ---------------------------------------------------------------------------

def _token_metrics(mono: np.ndarray, sample_rate: int, attack_ms: float) -> dict:
    n = len(mono)
    duration_ms = round((n / sample_rate) * 1000.0, 4)
    aw = min(int(attack_ms * sample_rate / 1000), n)
    if aw == 0:
        return {"duration_ms": duration_ms, "attack_rms": 0.0, "attack_peak": 0.0}
    window = mono[:aw]
    rms = round(float(np.sqrt(np.mean(window ** 2))), 6)
    peak = round(float(np.max(np.abs(window))), 6)
    return {"duration_ms": duration_ms, "attack_rms": rms, "attack_peak": peak}


# ---------------------------------------------------------------------------
# §4.1  Token discovery
# ---------------------------------------------------------------------------

def _collect_used_tokens(events: list) -> set:
    """
    §2.1: collect token ids from BGM (ch01) and P1 visible (ch11-19) events.
    For Long events, include both tokenStart and tokenEndOptional.
    """
    tokens: set = set()
    for ev in events:
        ev_type = ev["type"]
        if ev_type == "BGM":
            t = ev.get("token")
            if t and t != "00":
                tokens.add(t)
        elif ev_type == "Tap":
            ch = ev.get("rawChannel", "")
            if ch in _P1_VISIBLE or ch == _BGM_CH:
                t = ev.get("token")
                if t and t != "00":
                    tokens.add(t)
        elif ev_type == "Long":
            ch = ev.get("rawChannelStart", "")
            if ch in _P1_VISIBLE:
                t = ev.get("tokenStart")
                if t and t != "00":
                    tokens.add(t)
                t2 = ev.get("tokenEndOptional")
                if t2 and t2 != "00":
                    tokens.add(t2)
    return tokens


# ---------------------------------------------------------------------------
# Main run
# ---------------------------------------------------------------------------

def run(
    zip_path: Optional[str] = None,
    folder: Optional[str] = None,
    sample_rate: int = SAMPLE_RATE,
    decode_workers: int = DECODE_WORKERS,
    attack_ms: float = ATTACK_WINDOW_MS,
    output_dir: Optional[str] = None,
    bms_filename: Optional[str] = None,
) -> dict:
    check_dependencies()

    run_warnings: List[str] = []
    run_log: dict = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runtime_params": {
            "sample_rate": sample_rate,
            "decode_workers": decode_workers,
            "attack_window_ms": attack_ms,
        },
    }

    # ── 0. Locate working directory ──────────────────────────────────────
    tmp_dir = None
    if zip_path:
        print(f"Extracting {zip_path} ...")
        tmp_dir = tempfile.mkdtemp(prefix="bms_ta_")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(tmp_dir)
        contents = os.listdir(tmp_dir)
        work_dir = (
            os.path.join(tmp_dir, contents[0])
            if len(contents) == 1 and os.path.isdir(os.path.join(tmp_dir, contents[0]))
            else tmp_dir
        )
    elif folder:
        work_dir = folder
    else:
        raise ValueError("Must provide zip_path or folder")

    if output_dir is None:
        output_dir = (
            os.path.dirname(os.path.abspath(zip_path)) if zip_path else work_dir
        )

    # ── 1. Chart selection ───────────────────────────────────────────────
    print("Scanning BMS files ...")
    candidates = _find_bms_files(work_dir)
    if not candidates:
        sys.exit(f"ERROR: No BMS files found in {work_dir}")
    print(f"  Found {len(candidates)} candidates")

    if bms_filename:
        # User-specified BMS — skip pre-filter
        selected_path = None
        for path in candidates:
            if os.path.basename(path) == bms_filename:
                selected_path = path
                break
        if not selected_path:
            sys.exit(f"ERROR: {bms_filename} not found in package")

        with open(selected_path, "rb") as f:
            pr = parse_bms(f.read())
        player = pr["headers"].get("PLAYER", "").strip()
        coverage, playable, scratch = _selection_metrics(pr)
        dur = _estimate_duration(pr)

        if player != "1":
            run_warnings.append(
                f"WARNING: {bms_filename} has #PLAYER={player or 'absent'} (not 1P single)")
        if dur >= CHART_DURATION_MAX_SECONDS:
            run_warnings.append(
                f"WARNING: {bms_filename} duration {dur:.1f}s >= {CHART_DURATION_MAX_SECONDS}s")

        sel_info = {
            "all_candidates": [
                {"path": selected_path, "coverage": coverage,
                 "playable_count": playable, "scratch_events": scratch,
                 "player": player,
                 "estimated_duration_seconds": dur,
                 "prefilter_passed": True, "prefilter_discard_reason": None}
            ],
            "prefilter_discarded": [],
            "selected_path": selected_path,
            "selection_reason": "user_specified",
            "threshold_met": coverage >= 0.95,
            "coverage": coverage,
            "playable_count": playable,
            "scratch_events": scratch,
            "player": player,
            "estimated_duration_seconds": dur,
        }
        print(f"  User-specified: {bms_filename} "
              f"(coverage={coverage:.4f}, playable={playable}, duration={dur:.1f}s)")
    else:
        # Automatic selection
        selected_path, pr, sel_info = select_bms_file(candidates, run_warnings)
        print(
            f"  Selected: {os.path.basename(selected_path)} "
            f"(coverage={sel_info['coverage']:.4f}, "
            f"playable={sel_info['playable_count']})"
        )

    with open(selected_path, "rb") as f:
        chart_bytes = f.read()
    chart_hash = hashlib.sha256(chart_bytes).hexdigest()

    events = pr["events"]
    headers = pr["headers"]
    unique_measures = len({ev.get("measure", ev.get("measureStart")) for ev in events})

    run_log["chart"] = {
        **{k: v for k, v in sel_info.items() if k != "all_candidates"},
        "all_candidates": sel_info["all_candidates"],
        "hash_sha256": chart_hash,
        "parse_warnings": pr["warnings"],
        "parse_summary": {
            "measure_count": unique_measures,
            "event_count": len(events),
        },
    }

    # ── 2. Token discovery ───────────────────────────────────────────────
    wav_map: Dict[str, str] = {
        k[3:]: v.strip()
        for k, v in headers.items()
        if k.startswith("WAV") and len(k) > 3
    }
    declared_set = set(wav_map.keys())
    used_tokens = _collect_used_tokens(events)
    # §4.1: only tokens that are both declared in #WAVxx AND used in events
    scope_tokens = used_tokens & declared_set
    unused_declared = sorted(declared_set - used_tokens)

    print(
        f"  Declared WAV: {len(declared_set)}, "
        f"Used in events: {len(used_tokens)}, "
        f"In scope: {len(scope_tokens)}"
    )

    run_log["coverage"] = {
        "declared_wav_count": len(declared_set),
        "used_token_count": len(used_tokens),
        "in_scope_count": len(scope_tokens),
        "used_wav_coverage": round(
            len(used_tokens & declared_set) / len(declared_set) if declared_set else 0.0, 4
        ),
    }

    # ── 3. Decode and analyze tokens (with cache) ─────────────────────────
    ta_path = os.path.join(output_dir, "token_analysis.json")
    cache = load_token_cache(ta_path)
    print(f"  Cache loaded: {len(cache)} entries from {ta_path}")
    print(f"Decoding {len(scope_tokens)} tokens ({decode_workers} workers) ...")

    token_analysis: List[dict] = []
    missing_tokens: Dict[str, str] = {}
    decode_failed_tokens: Dict[str, str] = {}
    cache_hit_count = 0
    cache_miss_count = 0

    def _decode_one(tok: str) -> Tuple[str, str, Optional[np.ndarray], str, Optional[str]]:
        fname = wav_map[tok]
        fpath = _find_audio_file(work_dir, fname)

        if is_cache_valid(cache.get(tok), fname, fpath):
            return tok, fname, None, "cache_hit", cache[tok]["wav_hash"]

        if fpath is None:
            return tok, fname, None, "missing", None

        try:
            wav_hash = compute_wav_hash(fpath)
        except Exception:
            wav_hash = None

        mono = _decode_audio_mono(fpath, sample_rate)
        if mono is None:
            return tok, fname, None, "failed", wav_hash
        return tok, fname, mono, "ok", wav_hash

    with concurrent.futures.ThreadPoolExecutor(max_workers=decode_workers) as ex:
        futs = {ex.submit(_decode_one, tok): tok for tok in scope_tokens}
        done = 0
        for fut in concurrent.futures.as_completed(futs):
            tok, fname, mono, status, wav_hash = fut.result()
            done += 1
            if done % 200 == 0:
                print(f"  {done}/{len(scope_tokens)} ...")

            if status == "cache_hit":
                token_analysis.append(cache[tok])
                cache_hit_count += 1
            elif status == "ok":
                metrics = _token_metrics(mono, sample_rate, attack_ms)
                try:
                    spectral = _compute_spectral_metrics(mono, sample_rate)
                except Exception as exc:
                    token_analysis.append(
                        {"token": tok, "wav_file": fname, "wav_hash": wav_hash,
                         "decode_ok": False}
                    )
                    cache_miss_count += 1
                    decode_failed_tokens[tok] = f"{fname} (spectral failure: {exc})"
                    continue
                token_analysis.append(
                    {"token": tok, "wav_file": fname, "wav_hash": wav_hash,
                     "decode_ok": True, **metrics, **spectral}
                )
                cache_miss_count += 1
            else:
                token_analysis.append(
                    {"token": tok, "wav_file": fname, "wav_hash": wav_hash,
                     "decode_ok": False}
                )
                cache_miss_count += 1
                if status == "missing":
                    missing_tokens[tok] = fname
                else:
                    decode_failed_tokens[tok] = fname

    ok_count = sum(1 for e in token_analysis if e["decode_ok"])
    fail_count = len(token_analysis) - ok_count
    print(f"  OK: {ok_count}, Failed/Missing: {fail_count}")
    print(f"  Cache hits: {cache_hit_count}, Decoded: {cache_miss_count}")

    token_analysis.sort(key=lambda x: x["token"])

    # ── 4. Write outputs ─────────────────────────────────────────────────
    log_path = os.path.join(output_dir, "mix_generation_log.json")

    with open(ta_path, "w", encoding="utf-8") as f:
        json.dump(token_analysis, f, ensure_ascii=False, indent=2)

    run_log["token_diagnostics"] = {
        "missing_tokens": missing_tokens,
        "decode_failed_tokens": decode_failed_tokens,
        "unused_declared_tokens": unused_declared,
    }
    run_log["analysis_summary"] = {
        "total_analyzed": len(token_analysis),
        "decode_ok_count": ok_count,
        "decode_failed_count": fail_count,
        "cache_hit_count": cache_hit_count,
        "cache_miss_count": cache_miss_count,
    }
    run_log["warnings"] = run_warnings

    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(run_log, f, ensure_ascii=False, indent=2)

    print(f"token_analysis.json → {ta_path}")
    print(f"mix_generation_log.json → {log_path}")

    if tmp_dir:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return {
        "run_log": run_log,
        "token_analysis": token_analysis,
        "scope_tokens": scope_tokens,
        "declared_set": declared_set,
        "used_tokens": used_tokens,
        "unused_declared": unused_declared,
        "output_dir": output_dir,
    }


# ---------------------------------------------------------------------------
# §6  Conformance checklist
# ---------------------------------------------------------------------------

def run_conformance_checks(result: dict) -> None:
    log = result["run_log"]
    ta = result["token_analysis"]
    scope_tokens = result["scope_tokens"]
    declared_set = result["declared_set"]
    used_tokens = result["used_tokens"]
    unused_declared = result["unused_declared"]
    output_dir = result["output_dir"]
    results: Dict[int, str] = {}

    # ── Check 1: Pre-filter discards non-#PLAYER-1 files ─────────────────
    discarded = log["chart"].get("prefilter_discarded", [])
    # Verify every discarded entry has a non-1 reason
    all_non1 = all(
        "#PLAYER absent" in d["reason"] or "#PLAYER=3" in d["reason"] or "#PLAYER=2" in d["reason"]
        for d in discarded
    )
    # Verify selected file has player=1
    sel_player = log["chart"].get("player", "")
    # Verify DP candidates from the package are in discarded list
    dp_discarded = [d for d in discarded if "_DP" in d["path"]]
    results[1] = (
        "PASS"
        if all_non1 and sel_player == "1" and len(dp_discarded) >= 1
        else f"FAIL (all_non1={all_non1}, sel_player={sel_player!r}, dp_discarded={len(dp_discarded)})"
    )

    # ── Check 2: Selection follows §0.3.2 priority deterministically ──────
    # Order: coverage desc → scratch_events desc → playable_count desc
    candidates_log = log["chart"]["all_candidates"]
    sel_path = log["chart"]["selected_path"]
    sel_cov = log["chart"]["coverage"]
    sel_scr = log["chart"].get("scratch_events", 0)
    sel_play = log["chart"]["playable_count"]
    # Filter to prefilter-passed candidates only (others are discarded upfront)
    pf = [c for c in candidates_log if c.get("prefilter_passed", True)]
    max_cov = max(c["coverage"] for c in pf)
    cov_tier = [c for c in pf if c["coverage"] == max_cov]
    max_scr_among = max(c.get("scratch_events", 0) for c in cov_tier)
    scr_tier = [c for c in cov_tier if c.get("scratch_events", 0) == max_scr_among]
    max_play_among = max(c["playable_count"] for c in scr_tier)
    pass_2 = (sel_cov == max_cov and sel_scr == max_scr_among
              and sel_play == max_play_among)
    results[2] = "PASS" if pass_2 else (
        f"FAIL (sel_cov={sel_cov}/{max_cov}, "
        f"sel_scr={sel_scr}/{max_scr_among}, "
        f"sel_play={sel_play}/{max_play_among})"
    )

    # ── Check 3: All tokens in channel events are analyzed ────────────────
    # Every token in scope_tokens must have an entry in token_analysis
    analyzed_tokens = {e["token"] for e in ta}
    missing_from_analysis = scope_tokens - analyzed_tokens
    results[3] = (
        "PASS"
        if not missing_from_analysis
        else f"FAIL (missing {len(missing_from_analysis)} tokens from analysis)"
    )

    # ── Check 4: Unused declared tokens excluded from token_analysis ──────
    unused_in_analysis = set(unused_declared) & analyzed_tokens
    results[4] = (
        "PASS"
        if not unused_in_analysis
        else f"FAIL ({len(unused_in_analysis)} unused tokens found in analysis)"
    )

    # ── Check 5: Missing/decode-failed tokens logged with decode_ok:false ─
    # Synthetic: decode a nonexistent file → must return None
    mono = _decode_audio_mono("__nonexistent__.wav", SAMPLE_RATE)
    no_crash = mono is None
    # And verify decode_ok:false entries omit numeric fields
    fail_entries = [e for e in ta if not e["decode_ok"]]
    fail_no_numerics = all(
        "duration_ms" not in e and "attack_rms" not in e
        for e in fail_entries
    )
    # Log must have the diagnostic fields
    diag = log.get("token_diagnostics", {})
    has_diag = "missing_tokens" in diag and "decode_failed_tokens" in diag
    results[5] = (
        "PASS"
        if no_crash and fail_no_numerics and has_diag
        else f"FAIL (no_crash={no_crash}, fail_no_numerics={fail_no_numerics}, has_diag={has_diag})"
    )

    # ── Check 6: token_analysis.json schema matches §4.5 ─────────────────
    ta_file = os.path.join(output_dir, "token_analysis.json")
    schema_ok = False
    if os.path.exists(ta_file):
        with open(ta_file, encoding="utf-8") as f:
            ta_loaded = json.load(f)
        ok_entries = [e for e in ta_loaded if e.get("decode_ok") is True]
        fail_entries_f = [e for e in ta_loaded if e.get("decode_ok") is False]
        req_ok = {"token", "wav_file", "duration_ms", "attack_rms", "attack_peak", "decode_ok"}
        req_fail = {"token", "wav_file", "decode_ok"}
        ok_schema = all(req_ok <= e.keys() for e in ok_entries)
        fail_schema = all(
            req_fail <= e.keys() and "duration_ms" not in e
            for e in fail_entries_f
        ) if fail_entries_f else True
        # All token ids are 2-char base36
        tok_valid = all(
            len(e["token"]) == 2 and e["token"].isalnum()
            for e in ta_loaded
        )
        schema_ok = ok_schema and fail_schema and tok_valid
    results[6] = (
        "PASS"
        if schema_ok
        else f"FAIL (file_exists={os.path.exists(ta_file)}, ok_schema={ok_schema if os.path.exists(ta_file) else 'N/A'})"
    )

    # ── Check 7: Log contains full candidate list + selection + diagnostics
    required = [
        ("chart", "all_candidates"),
        ("chart", "prefilter_discarded"),
        ("chart", "selected_path"),
        ("chart", "selection_reason"),
        ("chart", "hash_sha256"),
        ("coverage", "used_wav_coverage"),
        ("token_diagnostics", "missing_tokens"),
        ("token_diagnostics", "decode_failed_tokens"),
        ("token_diagnostics", "unused_declared_tokens"),
        ("analysis_summary", "decode_ok_count"),
    ]
    missing_log = [f"{s}.{k}" for s, k in required if k not in log.get(s, {})]
    results[7] = "PASS" if not missing_log else f"FAIL (missing: {missing_log})"

    print()
    print("=== §6 Conformance Checks ===")
    for i in range(1, 8):
        print(f"Check {i}: {results.get(i, 'NOT RUN')}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="BMS Token Analysis (v4)")
    ap.add_argument("--zip", dest="zip_path")
    ap.add_argument("--folder")
    ap.add_argument("--sample-rate", type=int, default=SAMPLE_RATE)
    ap.add_argument("--workers", type=int, default=DECODE_WORKERS)
    ap.add_argument("--attack-ms", type=float, default=ATTACK_WINDOW_MS)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--bms", dest="bms_filename", default=None,
                    help="Use this BMS file (filename only, skip auto-selection)")
    args = ap.parse_args()

    if not args.zip_path and not args.folder:
        default_zip = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "[- 4 5] A D D i c T i O N 4 5 0 0 0 0 0.zip",
        )
        if os.path.exists(default_zip):
            args.zip_path = default_zip
        else:
            ap.error("Must provide --zip or --folder")

    out_dir = args.output_dir or os.path.dirname(os.path.abspath(__file__))

    result = run(
        zip_path=args.zip_path,
        folder=args.folder,
        sample_rate=args.sample_rate,
        decode_workers=args.workers,
        attack_ms=args.attack_ms,
        output_dir=out_dir,
        bms_filename=args.bms_filename,
    )

    run_conformance_checks(result)
