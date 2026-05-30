"""
Render a BMS chart's keysounds into a single mixed stereo WAV.

Used by BMS.Compare WPF viewer to play A/V-synced audio: viewer drives scroll
position from this WAV's playback position via NAudio.

Usage:
    python tools/render_chart_audio.py <chart.bms> [--folder <package>] [--out <path>]

Defaults:
    --folder = directory containing the chart
    --out    = <chart>.mixed.wav next to the chart

Approach:
    1. parse_bms on the chart
    2. Build tick->seconds map from base_bpm + BPMChange events + measure_scale
    3. Decode every referenced WAV once (ffmpeg -> float32 stereo @ 44.1kHz)
    4. Allocate output buffer, mix in each event's audio at its time offset
    5. Soft-clip and write 16-bit PCM WAV
"""
import argparse
import os
import struct
import subprocess
import sys
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, ROOT)
from bms_parser import parse_bms

SAMPLE_RATE = 44100
TICKS_PER_MEASURE = 192


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------

def decode_audio_stereo(file_path: str, sample_rate: int = SAMPLE_RATE
                        ) -> Optional[np.ndarray]:
    """ffmpeg -> float32 stereo. Returns shape (n, 2) or None."""
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
        return np.frombuffer(proc.stdout, dtype=np.float32).copy().reshape(-1, 2)
    except Exception:
        return None


def resolve_wav_path(folder: str, fname: str) -> Optional[str]:
    """Resolve a WAV filename: try as-given, then try common audio extensions."""
    direct = os.path.join(folder, fname)
    if os.path.isfile(direct):
        return direct
    base, _ = os.path.splitext(fname)
    for cand_base in (base, base.lower(), base.upper()):
        for ext in (".wav", ".ogg", ".mp3", ".flac"):
            p = os.path.join(folder, cand_base + ext)
            if os.path.isfile(p):
                return p
    return None


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

def build_tick_to_seconds(events: List[dict],
                          measure_scale: Dict[str, float],
                          base_bpm: float) -> "callable":
    """
    Returns tick_to_sec(tk: int) -> float.

    Walks BPM segments and respects per-measure scale. BPM changes are
    expected at any tick; measure_scale applies per measure.

    seconds_per_tick(bpm, scale) = 60 * scale / (48 * bpm)
        (1 measure with scale s = 4*s beats = 4*s * 60/bpm sec; 192 ticks)
        => per tick = 240*s / (192*bpm) = 5*s / (4*bpm) = 60*s / (48*bpm)
    """
    bpm_changes = sorted(
        (e["measure"] * 192 + e["idx192"], e["bpm"])
        for e in events if e["type"] == "BPMChange"
    )

    def tick_to_sec(tk: int) -> float:
        cur_bpm = base_bpm
        cur_pos = 0
        sec = 0.0
        seg_iter = list(bpm_changes) + [(tk, None)]
        for evt_tk, evt_bpm in seg_iter:
            target = min(evt_tk, tk)
            if target > cur_pos:
                pos = cur_pos
                while pos < target:
                    m = pos // 192
                    m_end = (m + 1) * 192
                    sub_target = min(target, m_end)
                    s = float(measure_scale.get(str(m), 1.0))
                    seg_ticks = sub_target - pos
                    sec += seg_ticks * 60.0 * s / (48.0 * cur_bpm)
                    pos = sub_target
                cur_pos = target
            if evt_tk == tk:
                break
            cur_bpm = float(evt_bpm)
        return sec

    return tick_to_sec


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------

PLAYABLE_KEY_CHANNELS = {"11", "12", "13", "14", "15", "16", "18", "19"}
PLAYABLE_LONG_CHANNELS = {"51", "52", "53", "54", "55", "56", "58", "59"}


def collect_sound_events(events: List[dict]) -> List[Tuple[int, str]]:
    """Return list of (tick, token) for all sample triggers (BGM + visible)."""
    out = []
    for ev in events:
        et = ev["type"]
        if et == "Tap":
            ch = ev.get("rawChannel", "")
            tok = ev.get("token", "")
            if not tok or tok == "00":
                continue
            if ch in PLAYABLE_KEY_CHANNELS or ch == "01":
                tk = ev["measure"] * 192 + ev["idx192"]
                out.append((tk, tok.upper()))
        elif et == "Long":
            ch = ev.get("rawChannelStart", "")
            tok = ev.get("tokenStart", "")
            if not tok or tok == "00":
                continue
            if ch in PLAYABLE_KEY_CHANNELS or ch in PLAYABLE_LONG_CHANNELS:
                tk = ev["measureStart"] * 192 + ev["idx192Start"]
                out.append((tk, tok.upper()))
        elif et == "BGM":
            tok = ev.get("token", "")
            if not tok or tok == "00":
                continue
            tk = ev["measure"] * 192 + ev["idx192"]
            out.append((tk, tok.upper()))
    return out


def render(bms_path: str, folder: str, out_path: str,
           sample_rate: int = SAMPLE_RATE,
           workers: int = 8) -> None:
    t0 = time.time()
    print(f"[1/5] Parsing {os.path.basename(bms_path)} ...", flush=True)
    with open(bms_path, "rb") as f:
        pr = parse_bms(f.read())
    headers = pr["headers"]
    events = pr["events"]
    base_bpm = pr["base_bpm"]
    measure_scale = pr["measure_scale"]

    print(f"[2/5] Resolving WAV declarations ...", flush=True)
    token_to_file: Dict[str, str] = {}
    missing: List[str] = []
    for k, v in headers.items():
        if not k.startswith("WAV") or len(k) <= 3:
            continue
        tok = k[3:].upper()
        path = resolve_wav_path(folder, v.strip())
        if path:
            token_to_file[tok] = path
        else:
            missing.append(f"{tok}={v}")
    if missing:
        print(f"  Missing {len(missing)} samples (skipped). First 5: {missing[:5]}")

    print(f"[3/5] Collecting sound events ...", flush=True)
    sound_events = collect_sound_events(events)
    used_tokens = sorted({tok for _, tok in sound_events})
    print(f"  {len(sound_events)} events, {len(used_tokens)} unique tokens")

    print(f"[4/5] Decoding {len(used_tokens)} samples (ffmpeg, {workers} workers) ...", flush=True)
    decoded: Dict[str, Optional[np.ndarray]] = {}
    decode_targets = [(tok, token_to_file[tok])
                       for tok in used_tokens if tok in token_to_file]
    decode_fail = 0
    progress_step = max(1, len(decode_targets) // 10)

    def _decode_one(item: Tuple[str, str]) -> Tuple[str, Optional[np.ndarray]]:
        tok, path = item
        return tok, decode_audio_stereo(path, sample_rate)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, (tok, audio) in enumerate(ex.map(_decode_one, decode_targets), 1):
            decoded[tok] = audio
            if audio is None:
                decode_fail += 1
            if i % progress_step == 0:
                print(f"  {i}/{len(decode_targets)}", flush=True)
    print(f"  decoded={len(decode_targets)-decode_fail}, failed={decode_fail}", flush=True)

    print(f"[5/5] Mixing into output buffer ...", flush=True)
    tick_to_sec = build_tick_to_seconds(events, measure_scale, base_bpm)

    # Compute total length: latest event's start_sec + that sample's duration
    max_end_sec = 0.0
    for tk, tok in sound_events:
        audio = decoded.get(tok)
        if audio is None:
            continue
        start_sec = tick_to_sec(tk)
        end_sec = start_sec + len(audio) / sample_rate
        if end_sec > max_end_sec:
            max_end_sec = end_sec
    total_samples = int(max_end_sec * sample_rate) + sample_rate  # +1s tail
    print(f"  total length: {max_end_sec:.1f}s ({total_samples} samples)", flush=True)

    out_buf = np.zeros((total_samples, 2), dtype=np.float32)
    skipped_no_decode = 0
    mixed = 0
    for tk, tok in sound_events:
        audio = decoded.get(tok)
        if audio is None:
            skipped_no_decode += 1
            continue
        offset = int(round(tick_to_sec(tk) * sample_rate))
        end = min(offset + len(audio), total_samples)
        seg = end - offset
        if seg <= 0:
            continue
        out_buf[offset:end] += audio[:seg]
        mixed += 1

    # Soft-clip then convert to int16
    peak = float(np.max(np.abs(out_buf))) if out_buf.size else 1.0
    if peak > 1.0:
        print(f"  peak {peak:.3f} > 1.0; normalizing", flush=True)
        out_buf /= peak
    out_int16 = np.clip(out_buf * 32767.0, -32768, 32767).astype(np.int16)

    with wave.open(out_path, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(out_int16.tobytes())

    elapsed = time.time() - t0
    sz_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"DONE: {os.path.basename(out_path)}  size={sz_mb:.1f}MB  "
          f"mixed={mixed}/{len(sound_events)} events  elapsed={elapsed:.1f}s")
    if skipped_no_decode:
        print(f"  WARNING: {skipped_no_decode} events skipped (no decoded sample)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bms_path", help="Path to .bms / .bme / .bml file")
    ap.add_argument("--folder", default=None,
                    help="Sample folder (default: directory containing the chart)")
    ap.add_argument("--out", default=None,
                    help="Output WAV path (default: <bms>.mixed.wav)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--sample-rate", type=int, default=SAMPLE_RATE)
    args = ap.parse_args()

    bms_path = os.path.abspath(args.bms_path)
    if not os.path.isfile(bms_path):
        sys.exit(f"ERROR: {bms_path} not found")
    folder = args.folder or os.path.dirname(bms_path)
    out_path = args.out or os.path.splitext(bms_path)[0] + ".mixed.wav"

    render(bms_path, folder, out_path,
           sample_rate=args.sample_rate, workers=args.workers)


if __name__ == "__main__":
    main()
