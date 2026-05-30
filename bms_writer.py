#!/usr/bin/env python3
"""
bms_writer.py
Generate placement_result.bms from placement_result.json + source BMS chart.
Supports LN output via #LNOBJ when ln_meta is present.
"""

import json
import math
import os
import re
import struct
import sys
import zipfile
from collections import defaultdict

# ── Constants ─────────────────────────────────────────────────────────────────

LANE_TO_CHANNEL = {
    "P1_SCR":  "16",
    "P1_KEY1": "11",
    "P1_KEY2": "12",
    "P1_KEY3": "13",
    "P1_KEY4": "14",
    "P1_KEY5": "15",
    "P1_KEY6": "18",
    "P1_KEY7": "19",
    # DP: P2 side (double play). Mirrors P1; scratch = 26.
    "P2_SCR":  "26",
    "P2_KEY1": "21",
    "P2_KEY2": "22",
    "P2_KEY3": "23",
    "P2_KEY4": "24",
    "P2_KEY5": "25",
    "P2_KEY6": "28",
    "P2_KEY7": "29",
}

PLAYABLE_CHANNELS = frozenset({
    "11", "12", "13", "14", "15", "16", "17", "18", "19",
    "21", "22", "23", "24", "25", "26", "27", "28", "29",
    "51", "52", "53", "54", "55", "56", "57", "58", "59",  # LNTYPE 1 P1
    "61", "62", "63", "64", "65", "66", "67", "68", "69",  # LNTYPE 1 P2
})

CHANNEL_RE = re.compile(r"^#(\d{3})([0-9A-Za-z]{2}):(.+)$")
WAV_RE = re.compile(r"^#WAV([0-9A-Za-z]{2})\s+(.+)$", re.I)

# Source LN headers to strip (player interprets these and forces LN mode)
STRIP_HEADERS = {"LNTYPE", "LNOBJ"}

ROOT_DIR    = os.path.dirname(os.path.abspath(__file__))
ZIP_PATH    = os.path.join(ROOT_DIR, "[- 4 5] A D D i c T i O N 4 5 0 0 0 0 0.zip")
TARGET_BMS  = "Addiction_INFERNO24.bms"
RESULT_JSON = os.path.join(ROOT_DIR, "placement_result.json")
OUTPUT_BMS  = os.path.join(ROOT_DIR, "placement_result.bms")


# ── Source BMS loading ────────────────────────────────────────────────────────

def _decode_bms(data: bytes) -> str:
    for enc in ("utf-8-sig", "utf-8", "cp932"):
        try:
            return data.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    return data.decode("utf-8", errors="replace")


def load_source_bms() -> str:
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        for name in zf.namelist():
            if os.path.basename(name) == TARGET_BMS:
                return _decode_bms(zf.read(name))
    sys.exit(f"ERROR: {TARGET_BMS} not found in zip")


# ── Source parsing ────────────────────────────────────────────────────────────

def _parse_tokens_from_data(data):
    N = len(data) // 2
    return [data[i * 2: i * 2 + 2] for i in range(N)]


def parse_source(text):
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    kept_lines, source_bgm_events, source_playable_lines = [], set(), set()
    source_bgm_lines = []  # list of (measure, tokens_list_uppercased) per #01 line
    tok_to_wav = {}        # token (upper) → wav path (lower)
    for raw in text.split("\n"):
        line = raw.rstrip()
        m = CHANNEL_RE.match(line)
        if m:
            measure, channel, data = int(m.group(1)), m.group(2).upper(), m.group(3)
            if channel in PLAYABLE_CHANNELS:
                source_playable_lines.add(line); continue
            if channel == "01":
                # Track #01 lines structurally so we can rebuild them after
                # excluding tokens that got placed as playable (avoids audio
                # doubling — same keysound playing via BGM and playable).
                tokens = [t.upper() for t in _parse_tokens_from_data(data)]
                N = len(tokens)
                for i, tok in enumerate(tokens):
                    if tok != "00":
                        source_bgm_events.add((measure, round(i * 192 / N), tok))
                source_bgm_lines.append((measure, tokens))
                continue  # do NOT append raw #01 line; rebuilt below
        # Track #WAV definitions for alias-aware dedup
        wm = WAV_RE.match(line)
        if wm:
            tok_to_wav[wm.group(1).upper()] = wm.group(2).strip().lower()
        # Strip source LN headers (LNTYPE/LNOBJ) — writer injects its own if needed
        if line.startswith("#") and not m:
            key = line[1:].split()[0].upper() if len(line) > 1 else ""
            if key in STRIP_HEADERS:
                continue
        kept_lines.append(line)
    return kept_lines, source_bgm_events, source_playable_lines, source_bgm_lines, tok_to_wav


def build_dedup_bgm_lines(source_bgm_lines, placed_keys, tok_to_wav, placed_wav_keys,
                           seen_wav=None):
    """Rebuild source #01 BGM lines, deduplicating against placed events and aliases.

    Drops a BGM occurrence at (m, idx192, token) when:
      (a) (m, idx192, token) is in placed_keys (exact-token match with placed)
      (b) (m, idx192, wav(token)) is in placed_wav_keys (alias of a placed token)
      (c) the same wav has already been emitted at (m, idx192) earlier in BGM
          processing (within-BGM alias dedup)

    `seen_wav` is the running set of (m, idx, wav) already emitted; if provided,
    it's mutated in place so callers (e.g., residual builder) can extend the
    dedup contract across both source-BGM and residual-BGM.
    """
    out = []
    if seen_wav is None:
        seen_wav = set()  # (m, idx192, wav) already emitted in BGM
    for (measure, tokens) in source_bgm_lines:
        N = len(tokens)
        new_tokens = list(tokens)
        for i, tok in enumerate(tokens):
            if tok == "00":
                continue
            idx192 = round(i * 192 / N)
            wav = tok_to_wav.get(tok, tok)  # fallback to token id if no #WAV
            # (a) exact-token match with placed
            if (measure, idx192, tok) in placed_keys:
                new_tokens[i] = "00"
                continue
            # (b) alias-WAV match with placed
            if (measure, idx192, wav) in placed_wav_keys:
                new_tokens[i] = "00"
                continue
            # (c) within-BGM alias dedup (cross-line and intra-line)
            if (measure, idx192, wav) in seen_wav:
                new_tokens[i] = "00"
                continue
            seen_wav.add((measure, idx192, wav))
        if any(t != "00" for t in new_tokens):
            out.append(f"#{measure:03d}01:{''.join(new_tokens)}")
    return out


def dedup_placed_alias(placed_events, tok_to_wav):
    """Drop placed events whose WAV alias collides with an earlier event at same tkey.

    Two distinct tokens that map to the same WAV file (e.g. snr.wav referenced
    by both 2E and 2F) at the same (measure, idx192) cause the same audio to
    fire twice. Keep the first encountered, drop later ones. Returns
    (filtered_events, dropped_count).
    """
    seen = set()  # (m, idx192, wav)
    keep = []
    dropped = 0
    for ev in placed_events:
        ev_type = ev.get("type", "Tap")
        if ev_type == "LN":
            m_, idx = ev["measure_start"], ev["idx192_start"]
        else:
            m_ = ev.get("measure_start", ev.get("measure"))
            idx = ev.get("idx192_start", ev.get("idx192"))
        tok = ev["token"].upper()
        wav = tok_to_wav.get(tok, tok)
        key = (m_, idx, wav)
        if key in seen:
            dropped += 1
            continue
        seen.add(key)
        keep.append(ev)
    return keep, dropped


# ── Channel sentence builder ──────────────────────────────────────────────────

def build_channel_sentence(measure, channel, events):
    if not events: return ""
    denominators = [1 if idx == 0 else 192 // math.gcd(idx, 192) for (idx, _) in events]
    N = 1
    for d in denominators: N = N * d // math.gcd(N, d)
    slots = ["00"] * N
    for (idx192, token) in events:
        slots[idx192 * N // 192] = token
    return f"#{measure:03d}{channel}:{''.join(slots)}"


# ── Placed events → playable channel sentences ───────────────────────────────

def build_placed_lines(placed_events, ln_meta=None):
    ln_enabled = ln_meta and ln_meta.get("enabled")
    lnobj_token = ln_meta.get("lnobj_token") if ln_enabled else None

    by_mc = defaultdict(list)
    for ev in placed_events:
        ev_type = ev.get("type", "Tap")
        if ev_type == "LN" and lnobj_token:
            ch = LANE_TO_CHANNEL[ev["lane"]]
            by_mc[(ev["measure_start"], ch)].append((ev["idx192_start"], ev["token"]))
            by_mc[(ev["measure_end"], ch)].append((ev["idx192_end"], lnobj_token))
        else:
            ch = LANE_TO_CHANNEL[ev["lane"]]
            m = ev.get("measure_start", ev.get("measure"))
            idx = ev.get("idx192_start", ev.get("idx192"))
            by_mc[(m, ch)].append((idx, ev["token"]))

    lines = []
    for (measure, channel) in sorted(by_mc.keys()):
        sentence = build_channel_sentence(measure, channel, by_mc[(measure, channel)])
        if sentence: lines.append(sentence)
    return lines


# ── Residual events → new #01 lines ──────────────────────────────────────────

def build_residual_lines(residual_events, source_bgm_events,
                          placed_keys=None, placed_wav_keys=None,
                          tok_to_wav=None, seen_wav=None):
    """Build #01 BGM lines from residual events with full dedup.

    Drops a residual at (m, idx, tok) when:
      - already in source BGM (handled separately, exact-match against source)
      - already in placed_keys (would double with placement)
      - alias-WAV match in placed_wav_keys (alias of a placed token)
      - alias-WAV already emitted by source BGM rebuild (`seen_wav`) or earlier residual
    """
    if placed_keys is None: placed_keys = set()
    if placed_wav_keys is None: placed_wav_keys = set()
    if tok_to_wav is None: tok_to_wav = {}
    if seen_wav is None: seen_wav = set()
    new_by_measure = defaultdict(list)
    dedup_count, seen = 0, set()
    for ev in residual_events:
        m, idx = ev["measure"], ev["idx192"]
        tok = ev["token"].upper()
        key = (m, idx, tok)
        if key in source_bgm_events: dedup_count += 1; continue
        if key in seen: dedup_count += 1; continue
        if key in placed_keys: dedup_count += 1; continue
        wav = tok_to_wav.get(tok, tok)
        wav_key = (m, idx, wav)
        if wav_key in placed_wav_keys: dedup_count += 1; continue
        if wav_key in seen_wav: dedup_count += 1; continue
        seen.add(key)
        seen_wav.add(wav_key)
        new_by_measure[m].append((idx, ev["token"]))
    lines = []
    for measure in sorted(new_by_measure.keys()):
        groups = []
        for (idx192, token) in new_by_measure[measure]:
            placed = False
            for group in groups:
                if all(i != idx192 for (i, _) in group):
                    group.append((idx192, token)); placed = True; break
            if not placed: groups.append([(idx192, token)])
        for group in groups:
            sentence = build_channel_sentence(measure, "01", group)
            if sentence: lines.append(sentence)
    return lines, dedup_count


# ── Silent WAV ────────────────────────────────────────────────────────────────

def write_silent_wav(path):
    if os.path.exists(path): return
    header = struct.pack('<4sI4s4sIHHIIHH4sI',
        b'RIFF', 36, b'WAVE',
        b'fmt ', 16, 1, 1, 44100, 88200, 2, 16,
        b'data', 0)
    with open(path, 'wb') as f:
        f.write(header)


# ── Conformance helpers ───────────────────────────────────────────────────────

def _extract_channel_events(bms_text, ch_filter):
    out = []
    for line in bms_text.replace("\r\n", "\n").split("\n"):
        line = line.rstrip()
        m = CHANNEL_RE.match(line)
        if not m: continue
        measure, channel = int(m.group(1)), m.group(2).upper()
        if channel not in ch_filter: continue
        tokens = _parse_tokens_from_data(m.group(3))
        N = len(tokens)
        for i, tok in enumerate(tokens):
            if tok.upper() != "00":
                out.append((measure, channel, round(i * 192 / N), tok.upper()))
    return out


def _collect_timing_lines(bms_text):
    timing = set()
    for line in bms_text.replace("\r\n", "\n").split("\n"):
        line = line.rstrip()
        m = CHANNEL_RE.match(line)
        if m and m.group(2).upper() in {"02", "03", "08", "09"}: timing.add(line)
    return timing


def run_conformance(output_text, placed_json, residual_json,
                    source_bgm_events, source_text, source_playable_lines,
                    generated_placed_lines, ln_meta=None,
                    placed_keys=None, placed_wav_keys=None, tok_to_wav=None):
    checks = {}
    lnobj_tok = (ln_meta.get("lnobj_token", "").upper()
                 if ln_meta and ln_meta.get("enabled") else None)
    placed_keys = placed_keys or set()
    placed_wav_keys = placed_wav_keys or set()
    tok_to_wav = tok_to_wav or {}

    # Check A
    out_play = _extract_channel_events(output_text, PLAYABLE_CHANNELS)
    out_play_ms = defaultdict(int)
    for ev in out_play: out_play_ms[ev] += 1
    expected = defaultdict(int)
    for pev in placed_json:
        ev_type = pev.get("type", "Tap")
        if ev_type == "LN":
            ch = LANE_TO_CHANNEL[pev["lane"]]
            expected[(pev["measure_start"], ch, pev["idx192_start"], pev["token"].upper())] += 1
            if lnobj_tok:
                expected[(pev["measure_end"], ch, pev["idx192_end"], lnobj_tok)] += 1
        else:
            ch = LANE_TO_CHANNEL[pev["lane"]]
            expected[(pev["measure"], ch, pev["idx192"], pev["token"].upper())] += 1
    missing_a = sum(max(0, cnt - out_play_ms.get(k, 0)) for k, cnt in expected.items())
    extra_a = sum(out_play_ms.values()) - sum(expected.values())
    checks["A_placed_completeness"] = "PASS" if missing_a == 0 and extra_a <= 0 else f"FAIL (missing={missing_a}, extra={extra_a})"

    # Check B (residual completeness): a residual is expected in output #01 unless
    # it was deduped — same exact (m,idx,tok) in
    # source BGM, in placed_keys, or its (m,idx,wav) in placed_wav_keys, or
    # already emitted earlier in BGM/residual via WAV alias.
    out_bgm = set((ev[0], ev[2], ev[3]) for ev in _extract_channel_events(output_text, {"01"}))
    out_bgm_wav = set((m, i, tok_to_wav.get(t, t)) for (m, i, t) in out_bgm)
    missing_b, seen_b = 0, set()
    for rev in residual_json:
        m_, i_ = rev["measure"], rev["idx192"]
        tok = rev["token"].upper()
        key = (m_, i_, tok)
        if key in source_bgm_events: continue
        if key in seen_b: continue
        if key in placed_keys: continue
        wav_key = (m_, i_, tok_to_wav.get(tok, tok))
        if wav_key in placed_wav_keys: continue
        # Already emitted via earlier residual or BGM alias?
        if wav_key in out_bgm_wav and key not in out_bgm: continue
        seen_b.add(key)
        if key not in out_bgm: missing_b += 1
    checks["B_residual_completeness"] = "PASS" if missing_b == 0 else f"FAIL (missing={missing_b})"

    # Check C
    checks["C_timing_preservation"] = "PASS" if _collect_timing_lines(source_text) == _collect_timing_lines(output_text) else "FAIL"

    # Check D
    generated_set = set(generated_placed_lines)
    out_lines = set(l.rstrip() for l in output_text.replace("\r\n", "\n").split("\n"))
    leaked = source_playable_lines & out_lines - generated_set
    checks["D_no_original_playable"] = "PASS" if not leaked else f"FAIL ({len(leaked)} leaked)"

    return checks


# ── Main ──────────────────────────────────────────────────────────────────────

def main(ln_meta=None):
    with open(RESULT_JSON, "r", encoding="utf-8") as f:
        result = json.load(f)

    # v12 §23 E.4: reject partial schema produced by resume mode. Resume output
    # has top-level "events"/"end_state" keys instead of "placed"/"residual"; if
    # a user accidentally points BMSWriter at a resume-mode result, fail loud
    # rather than crash on a missing "placed" key or write a corrupted .bms.
    if result.get("mode") == "resume":
        raise SystemExit(
            f"BMSWriter cannot consume resume-mode placement_result.json "
            f"(measures {result.get('start_measure', '?')}..{result.get('end_measure', '?')}). "
            f"Splice the partial events into a full chart and re-run with --finalize first.")

    placed   = result["placed"]
    residual = result["residual"]

    # Load ln_meta from result if not provided
    if ln_meta is None:
        ln_meta = result.get("ln_meta", {"enabled": False})

    source_text = load_source_bms()
    kept_lines, source_bgm_events, source_playable_lines, source_bgm_lines, tok_to_wav = parse_source(source_text)

    # DP: a DP chart must declare #PLAYER 3 (double play) so players route
    # the 21-29/26 channels to side 2. Source is SP (#PLAYER 1); rewrite it (or
    # inject before the first channel sentence if absent).
    dp_mode = bool(result.get("diagnostics", {}).get("dp_enabled"))
    if dp_mode:
        replaced = False
        for i, line in enumerate(kept_lines):
            if line.upper().lstrip().startswith("#PLAYER"):
                kept_lines[i] = "#PLAYER 3"
                replaced = True
                break
        if not replaced:
            insert_idx = len(kept_lines)
            for i, line in enumerate(kept_lines):
                if CHANNEL_RE.match(line):
                    insert_idx = i; break
            kept_lines.insert(insert_idx, "#PLAYER 3")

    # LNOBJ header injection
    lnobj_inject = []
    if ln_meta.get("enabled") and ln_meta.get("lnobj_token"):
        lnobj_tok = ln_meta["lnobj_token"]
        if not any(line.upper().startswith("#LNOBJ") for line in kept_lines):
            lnobj_inject.append(f"#LNOBJ {lnobj_tok}")
        if not any(line.upper().startswith(f"#WAV{lnobj_tok.upper()}") for line in kept_lines):
            lnobj_inject.append(f"#WAV{lnobj_tok} lnobj_silent.wav")

    # Find injection point: just before first channel sentence in kept_lines
    if lnobj_inject:
        insert_idx = len(kept_lines)
        for i, line in enumerate(kept_lines):
            if CHANNEL_RE.match(line):
                insert_idx = i; break
        for j, inj in enumerate(lnobj_inject):
            kept_lines.insert(insert_idx + j, inj)

    # Step 1: dedup alias collisions within placed events (e.g., 2E and 2F both
    # → snr.wav at same tkey would fire snr twice; keep first, drop later).
    placed, alias_dropped = dedup_placed_alias(placed, tok_to_wav)

    placed_lines = build_placed_lines(placed, ln_meta)

    # Build placed_keys / placed_wav_keys for downstream dedup.
    placed_keys = set()
    placed_wav_keys = set()
    for ev in placed:
        ev_type = ev.get("type", "Tap")
        if ev_type == "LN":
            m, idx = ev["measure_start"], ev["idx192_start"]
        else:
            m = ev.get("measure_start", ev.get("measure"))
            idx = ev.get("idx192_start", ev.get("idx192"))
        tok = ev["token"].upper()
        placed_keys.add((m, idx, tok))
        wav = tok_to_wav.get(tok, tok)
        placed_wav_keys.add((m, idx, wav))

    # Step 2: rebuild source #01 BGM with placed/alias dedup, sharing seen_wav
    # so step 3 (residuals) extends the same dedup ledger.
    seen_wav = set()
    bgm_lines = build_dedup_bgm_lines(source_bgm_lines, placed_keys, tok_to_wav,
                                       placed_wav_keys, seen_wav=seen_wav)
    bgm_doubles_removed = sum(1 for (m, ts) in source_bgm_lines for i, t in enumerate(ts)
                              if t != "00" and (m, round(i * 192 / len(ts)), t) in placed_keys)

    # Step 3: build residual #01 lines with full dedup against placed+BGM+aliases
    residual_lines, dedup_count = build_residual_lines(
        residual, source_bgm_events,
        placed_keys=placed_keys, placed_wav_keys=placed_wav_keys,
        tok_to_wav=tok_to_wav, seen_wav=seen_wav,
    )

    output_lines = kept_lines + bgm_lines + placed_lines + residual_lines
    while output_lines and output_lines[-1] == "": output_lines.pop()
    output_text = "\r\n".join(output_lines) + "\r\n"

    with open(OUTPUT_BMS, "w", encoding="utf-8", newline="") as f:
        f.write(output_text)

    # Write silent WAV if needed
    if ln_meta.get("enabled") and ln_meta.get("lnobj_token"):
        wav_dir = os.path.dirname(OUTPUT_BMS)
        write_silent_wav(os.path.join(wav_dir, "lnobj_silent.wav"))

    ln_count = sum(1 for ev in placed if ev.get("type") == "LN")
    print(f"placement_result.bms written "
          f"({len(placed_lines)} placed ch lines, "
          f"{len(bgm_lines)} bgm #01 lines (rebuilt), "
          f"{len(residual_lines)} residual #01 lines, "
          f"{dedup_count} deduped, {bgm_doubles_removed} BGM doubles removed, "
          f"{alias_dropped} placed alias dropped, "
          f"{ln_count} LN events)")

    checks = run_conformance(output_text, placed, residual,
                             source_bgm_events, source_text,
                             source_playable_lines, placed_lines, ln_meta,
                             placed_keys=placed_keys,
                             placed_wav_keys=placed_wav_keys,
                             tok_to_wav=tok_to_wav)
    print()
    for name, status in checks.items():
        print(f"  {name}: {status}")


if __name__ == "__main__":
    main()
