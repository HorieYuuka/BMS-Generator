"""
bms_parser.py — BMS/BME/BML parser
Handles headers, channel semantics, and lane mapping.
"""

from __future__ import annotations

import json
import math
import re
import sys
import warnings
import zipfile
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# §7  Lane mapping
# ---------------------------------------------------------------------------

VISIBLE_CHANNEL_TO_LANE: Dict[str, str] = {
    "11": "P1_KEY1",
    "12": "P1_KEY2",
    "13": "P1_KEY3",
    "14": "P1_KEY4",
    "15": "P1_KEY5",
    "16": "P1_SCR",
    "17": "P1_FREE",
    "18": "P1_KEY6",
    "19": "P1_KEY7",
    "21": "P2_KEY1",
    "22": "P2_KEY2",
    "23": "P2_KEY3",
    "24": "P2_KEY4",
    "25": "P2_KEY5",
    "26": "P2_SCR",
    "27": "P2_FREE",
    "28": "P2_KEY6",
    "29": "P2_KEY7",
}

VISIBLE_CHANNELS = frozenset(VISIBLE_CHANNEL_TO_LANE.keys())

# LNTYPE 1: channels 51-59, 61-69 (hex) -> subtract 0x40 -> 11-19, 21-29
def _ln_channel_to_visible(ch: str) -> Optional[str]:
    try:
        n = int(ch, 16)
    except ValueError:
        return None
    visible_n = n - 0x40
    visible_ch = format(visible_n, "02X")
    return visible_ch if visible_ch in VISIBLE_CHANNELS else None


# ---------------------------------------------------------------------------
# §12  Normalized event types
# ---------------------------------------------------------------------------

@dataclass
class TapEvent:
    type: str
    lane: str
    measure: int
    pos: float
    idx192: int
    token: str
    rawChannel: str


@dataclass
class LongEvent:
    type: str
    lane: str
    measureStart: int
    posStart: float
    idx192Start: int
    measureEnd: int
    posEnd: float
    idx192End: int
    tokenStart: str
    tokenEndOptional: Optional[str]
    rawChannelStart: str
    rawChannelEnd: str


@dataclass
class BgmEvent:
    type: str
    measure: int
    pos: float
    idx192: int
    token: str
    rawChannel: str


@dataclass
class BpmChangeEvent:
    type: str
    measure: int
    pos: float
    idx192: int
    bpm: float
    rawChannel: str


@dataclass
class StopEvent:
    type: str
    measure: int
    pos: float
    idx192: int
    duration: float
    rawChannel: str


# ---------------------------------------------------------------------------
# §3  Input handling
# ---------------------------------------------------------------------------

def _decode_bms_bytes(data: bytes) -> str:
    """§3.1: Try encodings in order, fallback to utf-8 with replacement."""
    for enc in ("utf-8-sig", "utf-8", "cp932"):
        try:
            return data.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    warnings.warn("BMS: falling back to utf-8 with replacement characters")
    return data.decode("utf-8", errors="replace")


def _normalize_newlines(text: str) -> str:
    """§3.2: Normalize CRLF and CR to LF; trim trailing newlines only."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text.rstrip("\n")


# ---------------------------------------------------------------------------
# §5.1  Slot math
# ---------------------------------------------------------------------------

def _slot_to_pos_and_idx192(slot_index: int, slot_count: int) -> Tuple[float, int]:
    pos = slot_index / slot_count
    idx192 = round(pos * 192)
    idx192 = max(0, min(191, idx192))
    return pos, idx192


# ---------------------------------------------------------------------------
# §9.3  LCM merge helpers
# ---------------------------------------------------------------------------

def _lcm(a: int, b: int) -> int:
    return a * b // math.gcd(a, b)


def _lcm_list(vals: List[int]) -> int:
    result = 1
    for v in vals:
        result = _lcm(result, v)
    return result


def _upsample(tokens: List[str], target_n: int) -> List[str]:
    """Upsample token list from len(tokens) to target_n via LCM expansion."""
    n = len(tokens)
    if n == 0 or target_n == 0:
        return ["00"] * target_n
    factor = target_n // n
    result: List[str] = []
    for tok in tokens:
        result.append(tok)
        for _ in range(factor - 1):
            result.append("00")
    return result


def _merge_token_lists(line_list: List[List[str]]) -> List[str]:
    """§9.2/9.3: LCM-merge multiple token lists; later non-zero wins."""
    if len(line_list) == 1:
        return list(line_list[0])
    counts = [len(l) for l in line_list if l]
    if not counts:
        return []
    n = _lcm_list(counts)
    result = ["00"] * n
    for tokens in line_list:
        if not tokens:
            continue
        up = _upsample(tokens, n)
        for i, tok in enumerate(up):
            if tok != "00":
                result[i] = tok
            # later "00" does NOT overwrite an earlier non-zero (§9.2)
    return result


# ---------------------------------------------------------------------------
# §4  Regexes and token validation
# ---------------------------------------------------------------------------

_CHANNEL_SENTENCE_RE = re.compile(r"^#(\d{3})([0-9A-Za-z]{2}):(.*)$")
_HEADER_RE = re.compile(r"^#([^\s:]+)\s(.*)$")
_BASE36_CHARS = frozenset("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")


def _is_valid_token(s: str) -> bool:
    return len(s) == 2 and all(c in _BASE36_CHARS for c in s)


# ---------------------------------------------------------------------------
# §11  Main parse function
# ---------------------------------------------------------------------------

def parse_bms(data: bytes) -> dict:
    """
    Parse BMS/BME/BML from raw bytes.

    Returns:
        {
          "headers": dict,
          "events":  list of serializable event dicts (§12 schema),
          "warnings": list of warning strings,
        }
    """
    text = _decode_bms_bytes(data)
    text = _normalize_newlines(text)
    lines = text.split("\n")

    warn: List[str] = []
    headers: Dict[str, str] = {}

    # Stores for channel-sentence raw data
    # (measure, channel) -> list-of-token-lists   (for mergeable channels)
    # (measure, "01")    -> list-of-token-lists   (BGM, kept independent)
    ch_raw: Dict[Tuple[int, str], List[List[str]]] = {}
    # channel 02 scale: (measure) -> raw data string (last line wins)
    scale_raw: Dict[int, str] = {}

    # BPM / STOP tables
    bpm_table: Dict[str, float] = {}
    stop_table: Dict[str, float] = {}

    # ── Pass 1: lex lines ──────────────────────────────────────────────────

    for line in lines:
        # §3.3: skip non-# lines
        if not line.startswith("#"):
            continue

        # Try channel sentence: #mmmCC:data
        m = _CHANNEL_SENTENCE_RE.match(line)
        if m:
            measure = int(m.group(1))
            raw_ch = m.group(2).upper()
            data_str = m.group(3).strip()

            # Channel 02 is special: the data is a float string, not token pairs
            if raw_ch == "02":
                scale_raw[measure] = data_str
                continue

            # §4.2: validate even length
            if len(data_str) % 2 != 0:
                warn.append(
                    f"Line has odd-length data, truncating: {line!r}"
                )
                data_str = data_str[:-1]

            if not data_str:
                continue

            # Split into 2-char tokens (uppercase)
            tokens: List[str] = []
            for i in range(0, len(data_str), 2):
                tok = data_str[i : i + 2].upper()
                if not _is_valid_token(tok):
                    warn.append(
                        f"Malformed token {tok!r} in m{measure} ch{raw_ch}; treating as 00"
                    )
                    tok = "00"
                tokens.append(tok)

            key = (measure, raw_ch)
            ch_raw.setdefault(key, []).append(tokens)
            continue

        # Try header: #KEY value  (split at first whitespace)
        m = _HEADER_RE.match(line)
        if m:
            raw_key = m.group(1).upper()
            value = m.group(2).strip()

            # §6.1: #BPMxx / #EXBPMxx
            if raw_key.startswith("EXBPM") and len(raw_key) > 5:
                idx = raw_key[5:]
                _store_bpm(bpm_table, idx, value, raw_key, warn)
                continue
            if raw_key.startswith("BPM") and len(raw_key) > 3:
                idx = raw_key[3:]
                # Could be plain #BPM (no index) – handled below if idx is empty
                if idx:
                    _store_bpm(bpm_table, idx, value, raw_key, warn)
                    continue

            # §6.2: #STOPxx
            if raw_key.startswith("STOP") and len(raw_key) > 4:
                idx = raw_key[4:]
                if idx:
                    _store_stop(stop_table, idx, value, raw_key, warn)
                    continue

            # §4.1: generic header; last wins
            headers[raw_key] = value
            continue

        # Lines starting with # but matching neither form: ignore silently
        # (§2 FAIL-SOFT, §3.3 only lines without # are explicitly ignorable,
        #  but a malformed # line is still not a crash)

    # ── Resolve #BPM base ──────────────────────────────────────────────────

    base_bpm = 130.0
    if "BPM" in headers:
        try:
            v = float(headers["BPM"])
            if v > 0:
                base_bpm = v
            else:
                warn.append(f"Non-positive #BPM value: {headers['BPM']!r}")
        except ValueError:
            warn.append(f"Invalid #BPM value: {headers['BPM']!r}")

    # ── Resolve LNOBJ / LNTYPE ────────────────────────────────────────────

    lnobj: Optional[str] = None
    if "LNOBJ" in headers:
        lnobj = headers["LNOBJ"].strip().upper()

    lntype1 = False
    if lnobj is None and headers.get("LNTYPE", "").strip() == "1":
        lntype1 = True

    # ── Resolve measure scales ─────────────────────────────────────────────

    measure_scale: Dict[int, float] = {}
    for measure, raw_val in scale_raw.items():
        try:
            v = float(raw_val)
            if v > 0:
                measure_scale[measure] = v
            else:
                warn.append(
                    f"Non-positive measure scale for m{measure}: {raw_val!r}"
                )
        except ValueError:
            warn.append(
                f"Invalid measure scale for m{measure}: {raw_val!r}"
            )

    # ── Pass 2: merge duplicate channel lines (§9) ─────────────────────────

    merged: Dict[Tuple[int, str], List[List[str]]] = {}

    for (measure, ch), line_list in ch_raw.items():
        if ch == "01":
            # §9.1: BGM – keep all sequences independently
            merged[(measure, ch)] = line_list
        else:
            # §9.2/9.3: later non-zero wins; LCM-merge if different slot counts
            merged[(measure, ch)] = [_merge_token_lists(line_list)]

    # ── Pass 3: build positional event candidates ──────────────────────────

    # Raw positional events grouped by tkey for ordering
    # Each entry: (tkey, event_type_priority, event_object)
    #   BPM change  priority 0  (§6.3)
    #   note/BGM    priority 1
    #   STOP        priority 2

    all_raw: List[Tuple[int, int, object]] = []

    current_bpm = base_bpm

    for (measure, ch), line_list in merged.items():

        # ── BGM channel 01 ──
        if ch == "01":
            for tokens in line_list:
                n = len(tokens)
                for i, tok in enumerate(tokens):
                    if tok == "00":
                        continue
                    pos, idx192 = _slot_to_pos_and_idx192(i, n)
                    tkey = measure * 192 + idx192
                    ev = BgmEvent(
                        type="BGM",
                        measure=measure,
                        pos=round(pos, 8),
                        idx192=idx192,
                        token=tok,
                        rawChannel="01",
                    )
                    all_raw.append((tkey, 1, ev))
            continue

        # ── BPM direct channel 03 ──
        if ch == "03":
            tokens = line_list[0]
            n = len(tokens)
            for i, tok in enumerate(tokens):
                if tok == "00":
                    continue
                try:
                    bpm_val = int(tok, 16)
                    if bpm_val <= 0:
                        warn.append(f"Non-positive direct BPM {tok!r} in m{measure}")
                        continue
                except ValueError:
                    warn.append(f"Invalid direct BPM token {tok!r} in m{measure}")
                    continue
                pos, idx192 = _slot_to_pos_and_idx192(i, n)
                tkey = measure * 192 + idx192
                ev = BpmChangeEvent(
                    type="BPMChange",
                    measure=measure,
                    pos=round(pos, 8),
                    idx192=idx192,
                    bpm=float(bpm_val),
                    rawChannel="03",
                )
                all_raw.append((tkey, 0, ev))
            continue

        # ── Extended BPM channel 08 ──
        if ch == "08":
            tokens = line_list[0]
            n = len(tokens)
            for i, tok in enumerate(tokens):
                if tok == "00":
                    continue
                if tok not in bpm_table:
                    warn.append(
                        f"ch08 references undefined BPM table entry {tok!r} in m{measure}; ignoring"
                    )
                    continue
                bpm_val = bpm_table[tok]
                pos, idx192 = _slot_to_pos_and_idx192(i, n)
                tkey = measure * 192 + idx192
                ev = BpmChangeEvent(
                    type="BPMChange",
                    measure=measure,
                    pos=round(pos, 8),
                    idx192=idx192,
                    bpm=bpm_val,
                    rawChannel="08",
                )
                all_raw.append((tkey, 0, ev))
            continue

        # ── STOP channel 09 ──
        if ch == "09":
            tokens = line_list[0]
            n = len(tokens)
            for i, tok in enumerate(tokens):
                if tok == "00":
                    continue
                if tok not in stop_table:
                    warn.append(
                        f"ch09 references undefined STOP table entry {tok!r} in m{measure}; ignoring"
                    )
                    continue
                stop_val = stop_table[tok]
                # §6.2: stopSeconds = (60 / currentBpm) * (stopValue / 48)
                # We use base_bpm as a placeholder here; actual BPM at time needs
                # the timing trace. We store duration in stop-units and convert later.
                pos, idx192 = _slot_to_pos_and_idx192(i, n)
                tkey = measure * 192 + idx192
                ev = StopEvent(
                    type="Stop",
                    measure=measure,
                    pos=round(pos, 8),
                    idx192=idx192,
                    duration=stop_val,  # raw stop units; resolved after BPM trace
                    rawChannel="09",
                )
                all_raw.append((tkey, 2, ev))
            continue

        # ── Visible note channels ──
        if ch in VISIBLE_CHANNELS:
            tokens = line_list[0]
            n = len(tokens)
            lane = VISIBLE_CHANNEL_TO_LANE[ch]
            for i, tok in enumerate(tokens):
                if tok == "00":
                    continue
                pos, idx192 = _slot_to_pos_and_idx192(i, n)
                tkey = measure * 192 + idx192
                ev = TapEvent(
                    type="Tap",
                    lane=lane,
                    measure=measure,
                    pos=round(pos, 8),
                    idx192=idx192,
                    token=tok,
                    rawChannel=ch,
                )
                all_raw.append((tkey, 1, ev))
            continue

        # ── LNTYPE 1 channels 51-59, 61-69 ──
        if lntype1:
            visible_ch = _ln_channel_to_visible(ch)
            if visible_ch is not None:
                tokens = line_list[0]
                n = len(tokens)
                lane = VISIBLE_CHANNEL_TO_LANE[visible_ch]
                for i, tok in enumerate(tokens):
                    if tok == "00":
                        continue
                    pos, idx192 = _slot_to_pos_and_idx192(i, n)
                    tkey = measure * 192 + idx192
                    # Store as tap for now; LN pairing done in Pass 4
                    ev = TapEvent(
                        type="_LNTap",  # internal marker
                        lane=lane,
                        measure=measure,
                        pos=round(pos, 8),
                        idx192=idx192,
                        token=tok,
                        rawChannel=ch,
                    )
                    all_raw.append((tkey, 1, ev))
                continue

        # ── Unknown channels: ignore safely (§10) ──

    # ── Pass 4: sort by (tkey, priority) and apply LN semantics ───────────

    all_raw.sort(key=lambda x: (x[0], x[1]))

    # ── Compute actual BPM at each STOP event ─────────────────────────────
    # Walk the sorted list once to track current BPM, then fix up stop durations

    running_bpm = base_bpm
    for tkey, pri, ev in all_raw:
        if isinstance(ev, BpmChangeEvent):
            running_bpm = ev.bpm
        elif isinstance(ev, StopEvent):
            ev.duration = (60.0 / running_bpm) * (ev.duration / 48.0)

    # ── Apply LNOBJ semantics (§8.1) ──────────────────────────────────────
    # Scan visible Tap events in lane order, match lnobj token to prior tap

    if lnobj is not None:
        # last_tap_per_lane: lane -> (list_index_in_final, TapEvent)
        # We need to process in tkey order (already sorted).
        # We'll rebuild the event list while doing LN conversion.

        last_tap_per_lane: Dict[str, Tuple[int, TapEvent]] = {}
        final_events: List[object] = []

        for tkey, pri, ev in all_raw:
            if isinstance(ev, TapEvent) and ev.type == "Tap" and ev.rawChannel in VISIBLE_CHANNELS:
                if ev.token == lnobj:
                    # This is an LN end marker
                    if ev.lane not in last_tap_per_lane:
                        warn.append(
                            f"LNOBJ end at m{ev.measure} pos{ev.pos} lane {ev.lane} "
                            "has no prior start; ignoring"
                        )
                        continue
                    idx, start_ev = last_tap_per_lane.pop(ev.lane)
                    # Replace the start tap with a Long event in-place
                    ln = LongEvent(
                        type="Long",
                        lane=start_ev.lane,
                        measureStart=start_ev.measure,
                        posStart=start_ev.pos,
                        idx192Start=start_ev.idx192,
                        measureEnd=ev.measure,
                        posEnd=ev.pos,
                        idx192End=ev.idx192,
                        tokenStart=start_ev.token,
                        tokenEndOptional=ev.token,
                        rawChannelStart=start_ev.rawChannel,
                        rawChannelEnd=ev.rawChannel,
                    )
                    final_events[idx] = ln
                    # Do not append end marker as separate event
                else:
                    # Regular tap — remember as potential LN start
                    last_tap_per_lane[ev.lane] = (len(final_events), ev)
                    final_events.append(ev)
            else:
                final_events.append(ev)

        # §8.4: unterminated LNs
        for lane, (idx, start_ev) in last_tap_per_lane.items():
            warn.append(
                f"Unterminated LNOBJ LN on lane {lane} started at "
                f"m{start_ev.measure} pos{start_ev.pos}; closing at EOF"
            )
            # Emit as LN with no explicit end
            ln = LongEvent(
                type="Long",
                lane=start_ev.lane,
                measureStart=start_ev.measure,
                posStart=start_ev.pos,
                idx192Start=start_ev.idx192,
                measureEnd=start_ev.measure,
                posEnd=start_ev.pos,
                idx192End=start_ev.idx192,
                tokenStart=start_ev.token,
                tokenEndOptional=None,
                rawChannelStart=start_ev.rawChannel,
                rawChannelEnd=start_ev.rawChannel,
            )
            final_events[idx] = ln

    # ── Apply LNTYPE 1 semantics (§8.2) ───────────────────────────────────
    elif lntype1:
        ln_open_per_lane: Dict[str, Tuple[int, TapEvent]] = {}
        final_events = []

        for tkey, pri, ev in all_raw:
            if isinstance(ev, TapEvent) and ev.type == "_LNTap":
                lane = ev.lane
                if lane not in ln_open_per_lane:
                    # First non-zero: opens LN
                    ln_open_per_lane[lane] = (len(final_events), ev)
                    final_events.append(ev)  # placeholder; will be replaced
                else:
                    # Second non-zero: closes LN
                    idx, start_ev = ln_open_per_lane.pop(lane)
                    ln = LongEvent(
                        type="Long",
                        lane=lane,
                        measureStart=start_ev.measure,
                        posStart=start_ev.pos,
                        idx192Start=start_ev.idx192,
                        measureEnd=ev.measure,
                        posEnd=ev.pos,
                        idx192End=ev.idx192,
                        tokenStart=start_ev.token,
                        tokenEndOptional=ev.token,
                        rawChannelStart=start_ev.rawChannel,
                        rawChannelEnd=ev.rawChannel,
                    )
                    final_events[idx] = ln
            else:
                final_events.append(ev)

        # §8.4: unterminated LNTYPE 1 LNs
        for lane, (idx, start_ev) in ln_open_per_lane.items():
            warn.append(
                f"Unterminated LNTYPE1 LN on lane {lane} started at "
                f"m{start_ev.measure} pos{start_ev.pos}; closing at EOF"
            )
            ln = LongEvent(
                type="Long",
                lane=start_ev.lane,
                measureStart=start_ev.measure,
                posStart=start_ev.pos,
                idx192Start=start_ev.idx192,
                measureEnd=start_ev.measure,
                posEnd=start_ev.pos,
                idx192End=start_ev.idx192,
                tokenStart=start_ev.token,
                tokenEndOptional=None,
                rawChannelStart=start_ev.rawChannel,
                rawChannelEnd=start_ev.rawChannel,
            )
            final_events[idx] = ln

    else:
        # No LN mode: just flatten
        final_events = [ev for _, _, ev in all_raw]

    # ── Serialize ─────────────────────────────────────────────────────────

    serialized = []
    for ev in final_events:
        d = asdict(ev)  # type: ignore[arg-type]
        serialized.append(d)

    return {
        "headers": headers,
        "bpm_table": bpm_table,
        "stop_table": stop_table,
        "measure_scale": {str(k): v for k, v in measure_scale.items()},
        "base_bpm": base_bpm,
        "events": serialized,
        "warnings": warn,
    }


# ---------------------------------------------------------------------------
# Helper functions for table parsing
# ---------------------------------------------------------------------------

def _store_bpm(
    table: Dict[str, float], idx: str, value: str, raw_key: str, warn: List[str]
) -> None:
    try:
        v = float(value)
        if v > 0:
            table[idx] = v
        else:
            warn.append(f"Non-positive BPM value for {raw_key}: {value!r}")
    except ValueError:
        warn.append(f"Invalid BPM value for {raw_key}: {value!r}")


def _store_stop(
    table: Dict[str, float], idx: str, value: str, raw_key: str, warn: List[str]
) -> None:
    try:
        v = float(value)
        if v > 0:
            table[idx] = v
        else:
            warn.append(f"Non-positive STOP value for {raw_key}: {value!r}")
    except ValueError:
        warn.append(f"Invalid STOP value for {raw_key}: {value!r}")


# ---------------------------------------------------------------------------
# §13  Conformance test helpers
# ---------------------------------------------------------------------------

def run_conformance_checks() -> None:
    """Run §13 conformance checks A, B, C, D and print pass/fail."""

    results: Dict[str, str] = {}

    # ── Test A: basic placement ──────────────────────────────────────────
    test_a = b"#00111:00112233\n"
    res_a = parse_bms(test_a)
    tap_events = [e for e in res_a["events"] if e["type"] == "Tap"]

    # Expected: 3 taps on P1_KEY1 at pos 1/4, 2/4, 3/4
    # §13 Test A: #00111 → measure 1; spec checks lane + positions only
    expected_a = [
        ("P1_KEY1", 1 / 4),
        ("P1_KEY1", 2 / 4),
        ("P1_KEY1", 3 / 4),
    ]
    got_a = [(e["lane"], e["pos"]) for e in tap_events]
    results["A"] = "PASS" if got_a == expected_a else f"FAIL (got {got_a!r})"

    # ── Test B: LNOBJ ────────────────────────────────────────────────────
    test_b = b"#LNOBJ ZZ\n#00111:00220000\n#00211:000000ZZ\n"
    res_b = parse_bms(test_b)
    long_events = [e for e in res_b["events"] if e["type"] == "Long"]
    tap_events_b = [e for e in res_b["events"] if e["type"] == "Tap"]

    pass_b = (
        len(long_events) == 1
        and long_events[0]["lane"] == "P1_KEY1"
        and long_events[0]["tokenStart"] == "22"
        and long_events[0]["tokenEndOptional"] == "ZZ"
        and not any(
            e["token"] == "22"
            for e in tap_events_b
            if e.get("lane") == "P1_KEY1"
        )
    )
    results["B"] = "PASS" if pass_b else f"FAIL (longs={long_events!r}, taps={tap_events_b!r})"

    # ── Test C: non-merge BGM ────────────────────────────────────────────
    test_c = b"#00001:AABB\n#00001:CCDD\n"
    res_c = parse_bms(test_c)
    bgm_events = [e for e in res_c["events"] if e["type"] == "BGM"]

    # Both sequences must be present independently (4 total tokens: AA, BB, CC, DD)
    bgm_tokens = {e["token"] for e in bgm_events}
    pass_c = {"AA", "BB", "CC", "DD"} == bgm_tokens
    results["C"] = "PASS" if pass_c else f"FAIL (tokens={bgm_tokens!r})"

    # ── Test D: BPM then STOP ordering ──────────────────────────────────
    # BPM and STOP at same slot → BPM applies before STOP duration computed
    # We verify that BPM change event appears before Stop event at the same tkey.
    test_d = b"#BPM 120\n#BPMzz 200\n#STOPyy 96\n#00003:0000ZZ00\n#00008:0000zz00\n#00009:0000yy00\n"
    res_d = parse_bms(test_d)

    timed_events = [
        e for e in res_d["events"]
        if e["type"] in ("BPMChange", "Stop")
        and e["measure"] == 0
        and e["idx192"] == round(2 / 4 * 192)
    ]
    types_in_order = [e["type"] for e in timed_events]
    # BPM changes (priority 0) must precede Stop (priority 2)
    pass_d = (
        len(timed_events) == 2
        and types_in_order.index("BPMChange") < types_in_order.index("Stop")
    )
    results["D"] = "PASS" if pass_d else f"FAIL (events={timed_events!r})"

    for check, result in results.items():
        print(f"Check {check}: {result}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    zip_path = os.path.join(
        os.path.dirname(__file__),
        "[- 4 5] A D D i c T i O N 4 5 0 0 0 0 0.zip",
    )
    bms_name = "[- 4 5] A D D i c T i O N 4 5 0 0 0 0 0/Addiction_HARDEST.bms"

    with zipfile.ZipFile(zip_path) as z:
        bms_data = z.read(bms_name)

    result = parse_bms(bms_data)

    out_path = os.path.join(os.path.dirname(__file__), "normalized_events.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Parsed {len(result['events'])} events.")
    print(f"Warnings: {len(result['warnings'])}")
    print(f"Output written to: {out_path}")
    print()
    print("=== §13 Conformance Checks ===")
    run_conformance_checks()
