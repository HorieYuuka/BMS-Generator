# BMS.Generator — A Source-Faithful Chart Generation Pipeline: Band Whitelist, Centroid Lane Assignment, and a Deterministic Resume API

**Subtitle**: Design and validation of a rule-based pipeline that rebuilds playable BMS charts from a source song's keysound pool, preserving the source's timing while remaining deterministic enough to support partial regeneration.

**Version**: v12 (2026-05-25)

> This is a technical report aimed at the BMS community. It assumes basic familiarity with the BMS chart format and rhythm-game charting, but explains every introduced term, policy, and constant in place.

---

## Abstract

Most automatic-charting work frames the task as *difficulty synthesis* — generate a chart of a target hardness. BMS.Generator takes the opposite stance: it treats charting as *source-faithful reconstruction*. Given a source BMS file, the pipeline reuses the song's own keysound assets (the `#WAV` token pool) and re-derives a playable note layout at a requested intensity, **without re-timing** the source. The result sounds like the original song because it is built from the original sounds at the original onsets.

The pipeline is rule-based (RB). A machine-learning path (token-selection + lane-assignment models) was trained and integrated behind an `--ml` flag, but a statistical evaluation found no measurable advantage over the rule-based path, so ML is operationally frozen (§8). The main contributions of the RB design are:

1. A **band-based quota whitelist** with rare-token rescue — selects which pool tokens are playable using spectral-band-relative occurrence quotas rather than a single global hard filter, protecting melodic-but-sparse tokens.
2. A **centroid-based relative lane assignment** — maps each note to a key lane by following the spectral-centroid trajectory of the music (brighter → move right, darker → move left) on a saturating curve, rather than assigning lanes randomly.
3. A **source-aware scratch policy** — mirrors the source's own per-measure scratch density scaled by a level multiplier, instead of synthesizing scratches from an absolute density table.
4. A **deterministic Resume API** — a per-measure RNG isolation (β-1) that makes any single measure (or measure range) reproducible in isolation, enabling an external editor to re-roll part of a chart without disturbing the rest.

Validation is by construction rather than by corpus statistics: a 17-point conformance check suite (§6.1), a 6-song × {RB, ML} byte-identical regression baseline, and a 9-case Resume API smoke suite. The operational corpus is a 13-package source set spanning chord-dense, LN-heavy, scratch-heavy, and BPM-trick charts.

---

## 1. Introduction

### 1.1 Problem statement

A BMS song ships as a set of audio samples (`#WAV` slots) plus one or more charts that place those samples on a timeline of lanes. Given only the song's samples and *one* reference chart, can we synthesize a **new playable chart** that:

- uses the song's own sounds (so it still sounds like the song),
- lands notes on the song's own onsets (so it stays musically honest),
- is playable at a controllable intensity (note density / scratch frequency / long-note usage), and
- is **reproducible** — the same inputs always yield the same chart, and a single measure can be regenerated in isolation?

The naive approach — "keep every sound event as a note" — produces an unplayable wall. The opposite — "synthesize notes to hit a target difficulty" — discards the song's identity. The design problem is the middle: *select* which sound events become playable notes, *assign* them to lanes that read naturally, and *shape* density to a requested intensity, all while never moving a note off its source onset.

### 1.2 Goal

Produce an **adequate chart** (not a maximally hard or maximally clever one): a chart that a human would accept as a faithful, playable rendering of the song at the chosen intensity. Concretely the pipeline emits, per run:

- `placement_result.bms` — the playable chart in BMS format,
- `placement_result.json` — the placed/residual event sets plus diagnostics,
- a similarity report against the package's existing charts.

The "adequate" framing (v12 §21) is deliberate: the rule-based policy aims to *block clearly definable bad patterns* (same-lane jacks faster than a floor, oversized chords, hand-balance collapse) rather than to *maximize* any aesthetic, because aesthetics are left to the source and to the user.

### 1.3 Contributions

- **C1 — Band quota whitelist + rare rescue.** Token selection evolved from a global hard filter (v9) to spectral-band-relative occurrence quotas (v10), then to a widened quota (0.20) plus an explicit rescue for non-FX tokens with occurrence ≤ 3 (v11). This keeps melodic highlights that a flat occurrence cut would drop. (§4.2)
- **C2 — Centroid relative lane assignment.** Lanes are chosen by following the music's spectral-centroid motion on a per-song-calibrated saturating curve, with ε-greedy diversification, replacing v9's random lane pick. (§4.6)
- **C3 — Source-aware scratch mirror.** In primary mode the output mirrors the source's per-measure scratch count × (level / 5), using the source's own minimum interval as the spacing floor, rather than a level-indexed absolute table. (§4.7)
- **C4 — Deterministic Resume API.** A per-measure RNG (β-1: `Random(seed × 10⁶ + measure)`) removes the chart-wide sequential RNG coupling, so a measure range `[M, N]` can be regenerated from a serialized carry-over state, with optional N+1 boundary lookahead. This is the substrate an external editor uses to re-roll a single measure. (§5)

A fifth, cross-cutting property — **determinism** — was hardened along the way: a class of `PYTHONHASHSEED`-dependent set/dict iteration non-determinism was found and removed (§5.5), and a 6-song regression baseline now guards byte-identical output.

---

## 2. Background

### 2.1 BMS format and the token pool

A BMS chart is a text file of channel lines keyed by measure. Each line `#mmmCC:...` places a sequence of two-character base-36 **tokens** in measure `mmm`, channel `CC`, at evenly divided sub-positions. The pipeline canonicalizes positions to **idx192** — 192 ticks per measure (LCM of common note divisions) — so a global timestamp is `tkey = measure × 192 + idx192`.

Key terms used throughout:

- **token** — a `#WAV` slot id; one audio sample.
- **event** — one token occurrence at a `(measure, idx192)`.
- **pool universe** — every used token event eligible for review.
- **playable whitelist** — the subset of tokens allowed to become playable notes.
- **residual** — a token event left as a BGM/auto-play object (channel `01`) rather than a playable note. Residualizing preserves the sound at its onset without demanding a key-press.
- **spectral centroid** — the frequency-domain center of mass of a token's sample; the brightness proxy that drives lane assignment.

The crucial invariant: **timing is never altered**. A note either becomes playable at its source onset or is residualized at that same onset. The pipeline only decides *playable-or-not* and *which lane*, never *when*.

### 2.2 The four-stage pipeline

```
mix_generation.py  →  placement_engine.py  →  bms_writer.py  →  similarity_check.py
   (analyze)             (place)                (emit)            (verify)
```

| Stage | Input | Output | Role |
|---|---|---|---|
| **MixGeneration** | source package (`.bms` + `.wav`s) | `token_analysis.json`, `mix_generation_log.json` | select the source chart, decode each keysound, compute per-token spectral / attack features (cached) |
| **PlacementEngine** | `token_analysis.json` + source events | `placement_result.json` | the policy core — whitelist, phase, per-measure placement, scratch, LN, density rebalance |
| **BMSWriter** | `placement_result.json` + source `.bms` | `placement_result.bms` | render placed notes + residual BGM back into a valid BMS file, preserving source timing lines |
| **SimilarityCheck** | output `.bms` + package charts | `similarity_report.json` | report overlap against the package's existing charts (diagnostic, non-gating) |

`run_pipeline.py` chains the four. The operational mode is **RB-only**; the `--ml` flag exists but is non-recommended (§8).

### 2.3 Rule-based vs machine-learned — why RB is frozen-on

Two models were trained: a `TokenSelectionModel` (which tokens to play) and a `LaneAssignmentModel` (which lane). On a v9 baseline the lane model reached ~50% top-1 accuracy versus 25% chance — a large apparent gain. But the gain was measured against the *random* lane baseline that v9 used. Once the rule-based path adopted centroid lane assignment (C2), the RB baseline itself captured much of the structure the model had learned, and a 2026-05-03 statistical evaluation found the ML models offered **no measurable advantage** over the RB path (v12 §19.5).

A caveat is recorded honestly: in blind A/B listening the ML output was sometimes felt to be "more stable / more human," yet none of the RB-aligned metrics captured that impression. The verdict is therefore "no *measurable* advantage," not "no advantage" — a metric-blindness possibility (§8.2, §9.2). ML remains frozen behind a flag, not deleted.

### 2.4 Framework stance: source dependence is acknowledged, not hidden

This pipeline is **not** a difficulty engine. It does not invent rhythms, does not move onsets, and does not aim for a target θ. Its identity comes from the source: the playable layout is a *projection* of the source's own sound events onto a playable lane space. This stance has a direct consequence for scope — anything that would require *re-timing* (e.g. synthesizing new rhythmic figures, or "improving" a sparse section) is out of scope by construction (v12 §2 FORBIDDEN). The Resume API (§5) respects the same boundary: re-rolling a measure changes *which lane / which token*, never *when*.

---

## 3. Architecture

### 3.1 Pipeline stages in detail

**MixGeneration** scans the package for candidate `.bms` files, selects one by coverage and playable-note count (or honors an explicit `--bms`), then decodes every declared keysound. For each token it computes attack strength (RMS / peak), duration, and STFT spectral features (centroid, flatness, low-frequency ratio, zero-crossing rate). Results are cached in `token_analysis.json` so re-runs skip decoding.

**PlacementEngine** is the policy core and the subject of §4–§6. It builds the pool universe, constructs the band whitelist, segments the chart into phases, then walks measures left-to-right placing notes under hard constraints, inserting scratches, post-processing long notes, and finally rebalancing density across the chart.

**BMSWriter** renders the result back to BMS. Placed notes go to key/scratch channels; residual tokens go to BGM channel `01`; source timing lines (BPM changes, STOPs, measure-length scales) are preserved verbatim. A set of writer-side conformance checks (A–D, §6.1) verify completeness and that no original playable note leaked through unaltered.

**SimilarityCheck** compares the output against the package's existing charts and reports overlap. It is diagnostic only — it never gates generation.

### 3.2 Data flow

```
source.bms ─┬─► MixGeneration ─► token_analysis.json ─┐
            │                                          ▼
            └────────────────────────────► PlacementEngine ─► placement_result.json ─┐
                                                                                       ▼
                                          source.bms ─────────────► BMSWriter ─► placement_result.bms
                                                                                       │
                                                                                       ▼
                                                                              SimilarityCheck ─► similarity_report.json
```

`placement_result.json` is the contract between policy and renderer. It is additively versioned: the Resume API (§5) adds `mode` / `end_state` fields in resume mode without changing the fields BMSWriter consumes in the default path, so existing consumers are unaffected.

---

## 4. Placement policy design

The PlacementEngine decides, for every pool event, whether it becomes a playable note and on which lane, then shapes the resulting density to the requested intensity. The policy is a left-to-right single pass over measures (the per-measure loop), bracketed by pool/whitelist construction before and density rebalancing after. All constants below are shown at the intensity = 5 / scratch = 5 default; both scale by a lerp curve (§6.2).

### 4.0 Implementation structure

Before the per-mechanism policies, the code-level shape. PlacementEngine is a sequence of mostly-pure functions around one stateful loop:

| Phase | Function | Produces | Policy § |
|---|---|---|---|
| pool build | `build_pool_universe` | pool tokens + occurrence counts + events | 4.1 |
| features | `compute_attack_percentile` / `compute_intensity_origin` / `classify_fx` | pct_map / intensity_origin / FX flags | 4.3 / 4.1 |
| whitelist | `build_whitelist` | whitelist, excluded, band stats | 4.2 |
| phase | `segment_phases` | rush / normal / rest blocks | 4.4 |
| scratch seed | `_determine_scratch_seeds` | scratch tokens + mode | 4.7 |
| **main loop** | `run_per_measure_loop` | placed + residual events (+ end_state) | 4.3–4.7 |
| ↳ per measure | `_place_measure_constrained` | placement under constraints | 4.5 |
| ↳ per note | `_centroid_lane_select` | lane | 4.6 |
| LN | `run_ln_postprocess` | LN-promoted events | 4.8 |
| scratch adj | `run_scratch_adjustment` | LN-aware scratch thinning | 4.7 |
| density | `run_density_rebalance` | balanced events | 4.9 |
| verify | `_run_conformance` | check pass/fail | 6.1 |

**Per-measure loop** (`run_per_measure_loop` — the one stateful pass; carry-over state per §5.2):

```text
for measure in [start .. end]:                 # default 0 .. measure_max
    rng = Random(seed × 10⁶ + measure)          # β-1 (§5.1)
    curr = reorder_within_idx(cands[measure])   # §4.3
    if ml: curr = ml_token_rerank(curr)         # §8.4 (optional)
    placed, hand_state, residual =
        _place_measure_constrained(curr, rng, hand_state, jack_state, …)
    token_usage += placed                       # under-used boost feeds next measure
    if scratch_active: insert_scratch(measure)  # §4.7
    residualize(unplaced)
# then (full-chart / finalize only): LN → scratch_adj → density_rebalance → conformance
```

**Constraint gate order** inside `_place_measure_constrained` — each candidate, in measure-candidate order, passes these gates; failing any one drops it to residual with a reason code:

```text
1.  measure note cap           → "measure_cap"
2.  collision (token @ pos)     → "collision"
3.  chord-ratio cap             → "no_lane_available"
4.  chord-size cap              → "chord_size_cap"
5.  available lanes ≠ ∅         → "collision"
6.  jack floor (BPM-aware)      → "jack_violation"
7.  jack streak                 → "jack_violation"
8.  hand balance (T ≥ 10)       → "hand_balance"
9.  same-hand streak            → "no_lane_available"
10. chord-mate spread (soft)    → narrows avail, never drops
11. boundary lookahead          → last chord only (§5.4)
12. lane select: ML → centroid → fisher-yates
→ place; update used_at_pos / jack_state / jack_streak / hand_state / centroid_state
```

The gate order is load-bearing: a note rejected at gate 6 (jack) never reaches gate 8 (hand balance), so the diagnostics' reason code reflects the *first* violated constraint, not all of them. §4.1–4.9 explain each phase's policy and rationale.

### 4.1 Pool universe and residual policy

The pool universe is every used token event. A token is hard-excluded as **FX** (background, not a key candidate) if its sample duration exceeds `FX_DURATION_THRESHOLD = 1000 ms`, its attack percentile is ≤ `FX_ATTACK_THRESHOLD = 20`, or it never appears on a key/scratch channel. A token whose sample fails to decode is marked **unknown** and can never be rescued.

Everything not placed is **residual**: it is written to BGM channel `01` at its original onset. This is the mechanism that lets the pipeline drop a note for playability *without* silencing the song — the sound still fires, it just no longer demands a key-press.

### 4.2 Band-based quota whitelist (C1)

The whitelist evolved through three forms:

- **v9** — a single global hard filter on duration / attack / occurrence.
- **v10** — a band-based quota: eligible tokens are split into three spectral bands (lo / mid / hi) by centroid tertiles, and each band keeps its top `BAND_QUOTA_RATIO` fraction by an occurrence-weighted rank.
- **v11/v12** — the ratio was raised `0.15 → 0.20` and a **rare-token rescue** added.

Within a band, tokens are ranked by

$$\text{rank}(t) = \text{occ}(t) - 5 \cdot \max\!\left(0, \frac{\text{dur}(t) - \text{WL}_{\text{dur}}}{\text{WL}_{\text{dur}}}\right)$$

where `occ` is total occurrence and the penalty discourages over-long samples (`WL_dur = 1055 ms` at lv5). Each band keeps `max(3, round(0.20 × |band|))` tokens; the rest are soft-excluded with reason `band_quota`.

**Rare rescue.** A flat occurrence cut drops sparsely-authored melodic highlights (e.g. Lepontinia m16's 8X/8Y/9B). So any *non-FX* token with total occurrence ≤ `RARE_OCCURRENCE_THRESHOLD = 3` rejoins the whitelist regardless of band rank. FX was already removed in §4.1, so a `band_quota` reason always means "non-FX but rank-cut," exactly the set worth protecting.

**Windowed rescue.** Over each 8-measure window (`WINDOW_SIZE = 8`), if a measure's whitelist pass-rate falls below `WINDOW_RESCUE_THRESHOLD = 0.40`, excluded tokens (ranked by window-level occurrence) are rescued up to that threshold. This prevents a locally token-poor measure from going empty.

### 4.3 Token intensity and within-idx reorder

A token's **intensity** is its attack percentile (`pct_map`), computed from RMS / peak across the pool. This drives ordering: louder onsets are preferred as the first note at a position.

The **within-idx reorder** (v11, C1-adjacent) refines same-position ordering to counteract the natural head-heavy distribution where a few loud tokens monopolize placement. After the highest-attack first pick, remaining chord-mates are scored by

$$\text{score}(t) = \min_{c \in \text{chosen}} |c_{\text{cent}} - t_{\text{cent}}| - 1000 \cdot \text{usage}(t)$$

— maximize spectral distance from already-picked mates (variety) while penalizing tokens already used a lot chart-wide (`USAGE_WEIGHT_SPREAD = 1000 Hz ≈ 1 prior use`). The first pick itself also carries a usage penalty (`USAGE_PENALTY_FIRST = 10` attack-pct points per prior use).

### 4.4 Phase segmentation

The chart is segmented into `rush` / `normal` / `rest` phases over 4-measure blocks using a smoothed candidate-density score, merging adjacent blocks whose ratio gap is below `PHASE_MERGE_RATIO_MAX = 0.289`. Phases are **diagnostics only** — they label sections for the report and do not gate placement. (Earlier versions used phase-adaptive relaxation; that was replaced by windowed rescue, §4.2.)

### 4.5 Placement constraints

Each measure is placed under hard constraints, in order. A note that fails a constraint is dropped to residual (preserving its onset), never re-timed.

- **Collision** — one token per `(position, lane)`; one lane per token at a position.
- **Jack floor** — the same lane may not repeat faster than `effective_min_ticks = max(MIN_JACK_DELTA_TICKS, ceil(MIN_JACK_DELTA_MS × bpm / 1250))`. The BPM-aware term means a fixed tick floor (15 at lv5) is raised on fast sections so a 16th-note same-lane repeat is blocked at every tempo.
- **Jack streak** — a single lane may carry at most `MAX_JACK_STREAK = 2` consecutive chord-anchored repeats before being forced off.
- **Chord size cap** — at most `MAX_CHORD_SIZE = 3` simultaneous lanes per position at lv5; overflow drops to residual (so timing is preserved, the chord just thins).
- **Hand balance** — once a measure has ≥ 10 notes, the running left/right share is held within `[0.30, 0.70]`; a note that would break balance is steered to the under-used hand or dropped.
- **Same-hand streak** — a new position may not extend the same hand past `STREAM_MAX_SAME_HAND = 2`.
- **Chord-mate spread** (v11) — a soft preference for chord-mate lanes ≥ `CHORD_MATE_SPREAD_MIN_GAP = 2` indices apart, eliminating `{1,2,3}`-style adjacent-lane clusters; falls back to any available lane when a wide chord forces packing.

### 4.6 Centroid-based relative lane assignment (C2)

The lane is chosen by following the music's **spectral-centroid trajectory** rather than at random. From the previous note's lane index and centroid, the next preferred lane is

$$\text{step} = \text{sign}(\Delta) \cdot \text{LANE\_STEP\_MAX} \cdot \left(1 - e^{-|\Delta| / \text{step\_unit}}\right), \qquad \Delta = c_{\text{cur}} - c_{\text{prev}}$$

A brighter token (positive Δ) moves the hand right; a darker one moves left. The saturating exponential makes the curve sensitive to small changes but caps large jumps at `LANE_STEP_MAX = 4` lanes. `step_unit` is auto-calibrated per song as the median of non-zero inter-token centroid differences, floored at `CENTROID_STEP_UNIT_FLOOR = 300 Hz`, so the same absolute brightness change maps to a consistent spatial step regardless of the song's spectral range.

An **ε-greedy** term (`CENTROID_EPSILON_RANDOM = 0.30`) picks a random available lane 30% of the time to break centroid drift and avoid monotone hand-walking. When centroid data is missing the assignment falls back to a Fisher-Yates shuffle.

### 4.7 Scratch policy — source-aware mirror (C3)

Scratch has three modes:

- **primary** — the source already has scratch tokens (channel `16`). The output mirrors the source's per-measure scratch count × `scale`, where `scale = level / 5` (so scratch = 5 is a 1:1 source mirror). The spacing floor is the source's *own* minimum interval, not the level table — the source author already shaped the rhythm.
- **fallback** — the source has no scratch. Scratches are synthesized from key tokens under the level-indexed absolute caps (`SCRATCH_MAX_PER_MEASURE = 4`, `SCRATCH_MIN_INTERVAL = 16` at scratch = 5), with a RUSH-rest rule that inserts a `SCRATCH_RUSH_REST_MEASURES = 4` cooldown after a sustained burst.
- **disabled** — no scratch insertion.

RUSH-rest fires in fallback only; primary mode trusts source pacing and disables it.

### 4.8 Long-note post-processing

After placement, eligible Tap notes are promoted to Long Notes (LN). A Tap is an LN candidate when its token's sample duration ≥ `LN_MIN_DURATION_MS = 800` (a gate tuned to roughly the p75 of human LN duration). The drawn hold length is capped at `LN_MAX_HOLD_TICKS = 96` (a 2-beat visible cap, v11) so a long sample does not paint a screen-filling bar; the *audio* sample still plays in full — only the visible bar is capped. The hold is written using the chart's `#LNOBJ` token when one is declared.

A known tension (§9.3): the 800 ms selection gate blocks ~75% of *naturally short* human LNs, so the pipeline under-produces LNs on LN-heavy songs. Lowering the gate uniformly would over-LN everything; a source-LN-signal infrastructure is the prerequisite for a per-song gate (future work).

### 4.9 Density rebalancing

The final step balances note density across four chart segments. Overloaded segments shed their lowest-intensity tokens to residual; under-dense segments pull tokens back from residual (fill-back). The correction uses a **soft-knee exponential damping** so the adjustment is gentle near the target and firm far from it, bounded by `DENSITY_REBALANCE_MAX_DELTA` (≈ 0.21 at lv5). Fill-back re-applies the same placement constraints (jack, chord size, chord-mate spread) so a rebalanced note is never illegal.

Density rebalance is **chart-wide**, not local — a property that matters for the Resume API's finalize step (§5.3): re-rolling one measure and then finalizing can move notes in *other* segments. This is intentional (chart consistency outranks local edit preservation) but must be surfaced in any editor UI built on the API.

---

## 5. Determinism and the Resume API (C4)

The motivating use case is an external editor (BMS.Compare) that wants to **re-roll** a part of a generated chart — regenerate one measure, or a measure range, with a different seed — while leaving the rest untouched. This requires two things the original engine did not have: (a) the ability to regenerate a measure range in isolation, and (b) true determinism, so "leave the rest untouched" actually holds byte-for-byte. §5.1–5.5 describe how both were achieved.

### 5.1 Per-measure RNG (β-1)

The original engine seeded a single `random.Random(seed)` at the start of the chart and consumed it across all measures. This couples every measure to every prior measure's RNG draws — fatal for partial regeneration, because to reproduce measure M you would have to replay measures 0…M-1 first.

Two strategies were considered:

| | α — serialize RNG state | β — per-measure RNG |
|---|---|---|
| mechanism | dump `Random.getstate()` (the Mersenne-Twister 624-tuple) into the carry-over state | seed a fresh RNG per measure from `(seed, measure)` |
| byte-identical to old output | yes | no (output changes, baseline regenerated) |
| schema risk | high — pins Python's internal RNG layout | none |
| undo/redo | must cascade prior RNG state | trivial — measure_idx alone reproduces |

**β was chosen** (the "β-1" single-stream-per-measure variant): the per-measure loop now seeds

$$\text{rng}_{\text{measure}} = \texttt{random.Random}(\text{seed} \times 10^{6} + \text{measure})$$

A pitfall surfaced during implementation: Python 3.13's `random.Random` rejects a tuple seed (`TypeError: only int/float/str/bytes/bytearray`), so a tuple-hash form was replaced by the arithmetic mapping above; the `10⁶` offset is collision-free for any realistic chart length (charts are hundreds of measures). A sub-decision (β-2: per-call-site keyed RNG) was rejected because there are only five RNG consumer sites and per-call-site partitioning gives no real robustness — measure-level isolation is exactly what the Resume API needs.

The trade-off accepted: chart output is no longer byte-identical to the chart-wide-stream era, so the `samples/baseline_lv5/` regression baseline was regenerated under β-1.

### 5.2 Carry-over state

To resume at measure M, the engine must inherit the cross-measure state that the left-to-right loop would have built up by the end of M-1. That state, serialized as a versioned JSON (`schema_version: "resume-v1"`), is:

| Field | Role |
|---|---|
| `jack_state` (lane → last tkey) | jack-floor delta check |
| `jack_streak` (lane → count) | consecutive same-lane cap |
| `centroid` (`prev_lane_idx`, `prev_centroid`) | centroid-trajectory continuity |
| `hand` (`last_hand`, `streak`) | same-hand-streak cap |
| `token_usage` (token → count) | under-used-token boost + fill-back ordering |
| `scratch` (`jack_scr_tkey`, `scratch_history`, `scr_rest_remain`) | scratch jack / RUSH-rest window |
| `rng` (`strategy`, `seed`) | β-1 base seed (per-measure RNG recomputed from it) |

A Codex audit corrected two earlier mis-identifications: `lane_tkeys` is reconstructed at finalize (not carried), and `hand_balance` is measure-local (the chart-wide accumulator is diagnostics-only). The centroid `step_unit` is *not* carried — it is chart-input deterministic and recomputed by the resume entry. ML carry-over (`token_context`, `lane_context`, `global_lane_counts`) is **out of scope for v1** (RB-only); loading a state whose `rng.strategy` or `schema_version` does not match raises a `ValueError`.

### 5.3 Resume and finalize modes

Three CLI entry shapes share one engine:

```
(default)                  full chart, unchanged behavior
--resume-state S \
  --start-measure M \
  --end-measure N          resume mode: regenerate [M,N] from state S,
                           emit raw events + end_state, SKIP post-processing
--finalize EVENTS          finalize mode: take spliced full-chart events,
                           run post-processing only
```

**Resume mode** narrows the per-measure loop to `[M, N]`, overrides the carry-over state from `S`, and emits a partial `placement_result.json` (`mode: "resume"`, the M…N events, and an `end_state` for cascading). It skips LN / scratch-adjust / density-rebalance / conformance entirely — the caller splices the raw events and finalizes later. In `run_pipeline.py`, resume mode also skips BMSWriter and SimilarityCheck (a partial chart has no meaningful `.bms` or similarity).

**Finalize mode** is the inverse: given the full chart events that the editor has spliced together, run the post-processing chain in its fixed order — LN postprocess → scratch adjustment → density rebalance → conformance — and emit a normal full `placement_result.json`. Because density rebalance is chart-wide (§4.9), finalize can move notes outside the re-rolled region; the policy decision is explicit (v12 §23.6 / DR-23-4): **chart consistency outranks preservation of the user's local edit**, so finalize is invoked only on an explicit user action, and the editor must warn that re-rolled regions are not guaranteed to survive.

### 5.4 Boundary lookahead (E-β)

Resume mode inherits the *left* neighbor perfectly (the carry-over state is the M-1 end state). It does **not** see the *right* neighbor: the engine's `next_cands` lookahead reads only the raw candidate pool of measure N+1, never N+1's already-placed lanes. So the N → N+1 boundary of a re-rolled region can collide — a jack against N+1's first chord, or a hand-streak that N+1 then extends.

For single-measure re-roll (the motivating case, where both neighbors are fixed), this matters. The fix (`--next-chord-lookahead`, v12 §23.7) lets the caller pass N+1's first chord; the engine then applies, *only to the last chord of the last measure*, a forward constraint:

- **jack** (hard) — reject any lane that would collide with an N+1 lane within `effective_min_ticks`;
- **hand-streak** (soft) — if the chosen hand equals N+1's hand and the streak would exceed the cap, prefer the opposite hand.

Centroid two-sided interpolation (E-γ) was deferred — it would require rewriting the saturating-curve lane selector, and unlike jack/hand it is a soft preference. Only the first N+1 chord is used; later chords are buffered by it.

### 5.5 The determinism bug: PYTHONHASHSEED

While verifying that the default path stayed byte-identical after β-1, two consecutive runs of the *same* input produced *different* charts. β-1 itself is deterministic; the culprit was elsewhere: with `PYTHONHASHSEED` unset (Python's default), the hash of strings and tuples-of-strings is randomized per process, so iterating a `set`/`dict` of tokens yields a different order each run — and that order leaked into placement decisions.

Five sites were hardened with `sorted()` or a token tie-break: the attack-percentile sort, the whitelist eligible-iteration and band-rank and rare-rescue iterations, and the per-measure scratch-candidate iteration. A sixth, in the labeling pipeline, was fixed for training determinism. After the fix, the 6-song × {RB, ML} baseline is byte-identical across runs regardless of `PYTHONHASHSEED`. The lesson is filed as a standing review item (v12 §22 DR-23-6): *a `set`/`dict` whose elements are strings is safe to hold, but the moment its iteration order feeds a decision, it must be sorted.*

---

## 6. Conformance and calibration

Because the pipeline has no large labeled corpus to score against, validation is **by construction**: every output is checked against the policy's own invariants, and a regression baseline guards determinism.

### 6.1 Conformance checks

Each run emits pass/fail on a check suite split across the two stages that own the relevant invariant:

- **PlacementEngine** — A whitelist hard-filter, B timing preservation, C jack prohibition (scratch lane excluded), D fallback behavior, E candidate collision, F scratch constraints, G seeded reproducibility, K measure cap. Check J (density-rebalance legality) is documented as manual-inspection-only: making it a hard gate would require re-running the per-measure jack/collision logic over the post-rebalance event set, which is out of scope (v12 §22 DR-J1).
- **BMSWriter** — A placed completeness, B residual completeness, C timing-line preservation (BPM / STOP / scale lines byte-identical to source), D no-original-playable-leak.

Check B (timing preservation) and BMSWriter's Check C together enforce the core invariant of §2.1: the output's timing is the source's timing, unaltered.

### 6.2 Intensity and scratch scaling

`--intensity` (1–20, default 5) and `--scratch` (1–20, default 5) scale the constants of §4 along piecewise-linear lerp curves anchored at lv1 / lv10 / lv20. Intensity moves jack floors, chord-size cap, measure note cap, same-hand streak, and stream ratios; scratch moves the per-measure scratch budget and minimum interval. The default lv5 is the calibration point used throughout this report and for the baseline.

### 6.3 Regression baseline

`samples/baseline_lv5/` holds 6 songs × {RB, ML} × {bms, json} = 24 files at lv5. Two automated suites guard the pipeline:

- `smoke_test_determinism.py` — regenerates all 6 songs (RB + ML) and asserts byte-identical to the saved baseline. Catches `PYTHONHASHSEED`-class regressions and any accidental policy drift.
- `smoke_test_resume.py` — 9 Resume API cases: base split, M=0 single, last-measure single, 3-stage cascade, ML+resume rejection, schema-version mismatch, RNG-strategy mismatch, lookahead-requires-resume, and a lookahead-wiring smoke. All pass.

The baseline was regenerated under β-1 + the nondet fix; the regression is what makes "β-1 changed the output, intentionally" a one-time event rather than silent drift.

---

## 7. Case studies

### 7.1 mightyA — the 47-streak and the chord-collapse artifact

An early diagnostic metric, `same_hand_streak`, showed an alarming fat tail on mightyA (a streak of 47 same-hand notes). It looked like a real hand-balance failure. On inspection it was a **measurement artifact**: the metric counted chord-collapsed positions as separate same-hand events, so a dense chord wall inflated the streak without any actual same-hand run. A chord-aware re-metric (`hand_only_streak`) put source / RB / ML all at max 3–8 — no signal. The lesson (v12 §22 DR-K1): a scary number in a diagnostic is a hypothesis about the *metric* first, the *chart* second.

### 7.2 happiness — the BPM-naive LN cap

`hapiness_lnext` is a human LN-heavy chart (757 LNs). The pipeline produced 0–6. Two causes compounded: the 800 ms selection gate (§4.8) blocks ~75% of naturally-short human LNs, and an earlier draw-length policy was BPM-naive — at 242 BPM a fixed tick hold painted a screen-filling bar. The draw-length cap `LN_MAX_HOLD_TICKS = 96` (visible-only) fixed the painting problem (v12 §22 DR-G1); the selection-gate problem remains open (§9.3) because lowering it uniformly would over-LN every chart.

### 7.3 Single-measure reroll — the Resume API end-to-end

`samples/reroll_demo_2026-05-18/` demonstrates the API: the signal chart's measure 58 is re-rolled in isolation (prefix `[0,57]` → `end_state` → resume `[58,58]` with N+1 lookahead), then spliced back. The output `.bms` differs from the base in **measure 58 only** — every other measure is byte-identical, confirming β-1's measure isolation. (The demo also exposed a limit: m58's last chord landed on the scratch lane, where the KEY-lane jack lookahead does not apply, so it is a wiring demonstration rather than a lane-swap demonstration — see §9.)

### 7.4 Source chord composition divergence

A cross-cutting finding (v12 §22 DR-K2): the pipeline systematically alters the source's chord composition. On mightyA, source 6% two-hand-MIXED chords became 39% in output; on happiness, 67% → 34%. The direction is inconsistent across songs. The suspected cause is that pool dedup discards channel information and centroid lane assignment spreads chord-mates, so a source single-hand chord can land split across both hands. The audibility ("busier") is plausible but hard to objectify — it is an open item, not a fixed bug.

---

## 8. Machine learning — trained, integrated, and frozen

The ML path is not a sketch. Both models were trained end-to-end from a labeled corpus, exported to TorchScript, and wired into the live inference path behind `--ml`. It is documented in full here because "we tried ML and it didn't win" is only useful to the community if the *how* — the data, the architecture, the training setup — is on the record.

| Model | Replaces | Params | Loss | Status |
|---|---|---:|---|---|
| `TokenSelectionModel` | pct-based candidate ordering (§4.3) | ≈ 6.3K | masked BCE | wired (`--ml`), no measurable gain |
| `LaneAssignmentModel` | centroid lane pick (§4.6) | ≈ 24.8K | masked CE | wired (`--ml`), no measurable gain |

### 8.1 Data preparation — the labeling pipeline

Training data is extracted from real human charts by `data_labeling.py`: for every eligible measure of every chart in a package, it emits a record pairing the *situation* (measure + pool + context features) with the *human decision* (which tokens were played, on which lanes).

- **Pool features** — 14 columns per token, stored once per package in a pool table (records reference tokens by integer `pool_index`): `duration_ms`, `attack_rms`, `attack_peak`, `intensity_origin`, key / scratch / bgm occurrence counts, six STFT spectral features (centroid mean/std, flatness, low-frequency ratio, zero-crossing mean/std), and a whitelist-pass flag.
- **Measure features** — `measure_index`, chart-level `density_rank`, `phase`, `notes_in_measure`.
- **Context window** — the preceding C = 4 *eligible* measures (ineligible measures skipped; chart-start zero-padded oldest-first), each carrying `tkey_delta`, `placed_count`, and the placed pool-index / lane history. The inference-time context builder (`placement_engine._build_ml_context`) reproduces this ordering exactly, so training and inference see an identical tensor layout.
- **Labels** — token labels are 0/1 (played or not → BCE); lane labels are the human lane 1..7 (→ CE). A record whose ground-truth lane is unavailable under the constraints is *skipped*, never taught an illegal move.

The pipeline was run at full scale (~6,395 packages). A **v2 → v3 schema redesign** was forced mid-run: v2 accumulated all records in memory before writing, which OOM-crashed and filled ~471 GB of disk on the full corpus. v3 streams records to JSONL with a package-level pool table (lifting the 14-feature rows out of the per-record payload), and — per §5.5 — its token iteration was later sorted for training determinism.

### 8.2 Model architecture

Both are deliberately small MLPs with a fixed inference interface; the tensor column orders are pinned by the model I/O contract so a retrain can never silently shift a feature.

- **TokenSelectionModel** — a siamese scorer over a variable-size pool. Per pool row it concatenates `[4 measure ⊕ 14 pool ⊕ 12 flattened context] = 30` dims, then `LayerNorm(30) → Linear(30,64) → ReLU → Dropout(0.3) → Linear(64,64) → ReLU → Dropout(0.3) → Linear(64,1)`, squeezed to one score per token. A variable pool size `P` is handled by concatenating measures along the row axis rather than padding to a common width. ≈ 6,269 parameters.
- **LaneAssignmentModel** — a 7-way classifier. It concatenates `[16 event ⊕ 40 flattened context] = 56` dims, then `LayerNorm(56) → Linear(56,128) → ReLU → Dropout(0.3) → Linear(128,128) → ReLU → Dropout(0.3) → Linear(128,7)`, masking unavailable lanes to `-inf` before softmax so the caller can argmax directly. ≈ 24,823 parameters.

The small size is intentional (by design): the models are *re-rankers that assist RB*, not the dominant decision-maker, so a capacity ceiling guards against a model overpowering the rule constraints. `LayerNorm` is the first layer in both — input features span very different natural scales (`duration_ms` in the hundreds vs `attack_rms` in [0,1]) and per-sample normalization avoids maintaining running statistics at inference.

### 8.3 Training setup

- **Optimizer** — Adam, `lr = 1e-3`, `weight_decay = 1e-4`; `Dropout(0.3)` after each hidden ReLU. No BatchNorm (it interacts poorly with variable-`P` siamese batches and small batches).
- **Schedule** — up to 20 epochs with early stopping (`patience = 3`), batch size 256.
- **Class weighting (lane)** — the human lane distribution is imbalanced. The lane model is retrained with class-weighted cross-entropy (`--class-weights auto --class-weight-power 2.0`), up-weighting rare lanes by inverse frequency raised to power 2.
- **Split** — a package-level deterministic shuffle (`seed = 42`, `DR-7`): a package's measures never straddle the train/val boundary, so the model cannot memorize a chart it is then validated on.
- **Export** — TorchScript `script` (not `trace`, `DR-4`, so control flow survives), loaded `map_location="cpu"` at inference (`DR-8`) — generation needs no GPU.
- **Environment** — the operational trap worth recording: Python 3.13 + a GTX 1070 requires the CUDA **cu118** PyTorch build (cu121 ships no 3.13 wheel). The class-weighted lane retrain (`training/checkpoints`, `lane_cw2` TensorBoard run) was performed under this setup.

### 8.4 Integration — soft re-ranker with rule-based fallback

Both models integrate as *soft re-rankers*: the RB policy owns every structural decision (which segment, how many notes, which constraints), and the model only reorders within the RB-permitted set. On any inference failure (exception, shape mismatch, lane model disabled, empty availability) the call falls back to the rule path — the token model to pct ordering, the lane model to the centroid / Fisher-Yates pick. By construction the model can improve ordering but can *never* violate a constraint. A fill-back ranking hook (density rebalance uses the token model to order pull-backs) was added later with separate diagnostics counters.

### 8.5 The verdict, and the metric-blindness caveat

On a v9 baseline the lane model's ~50% top-1 accuracy (vs 25% chance) looked decisive. But that baseline was *random* lane assignment. After the RB path adopted centroid lane assignment (§4.6), the RB baseline itself captured most of the learnable structure, and the 2026-05-03 statistical evaluation found **no measurable advantage** for either model over RB. A separate finding from that audit: the lane model had learned a K1/K3/K4 lane prior — it leaned on the most common lanes rather than the context, which is exactly what class weighting (§8.3) was introduced to counter.

The honest caveat: in blind A/B listening the ML output was repeatedly *felt* to be more stable / more human, yet no RB-aligned metric captured it. So the verdict is "no *measurable* advantage," with an explicit metric-blindness possibility — the listening impression is a fact; only its quantification is unsolved (§9.2). The class-weighted retrain improved the lane prior but did not move the verdict; its result was absorbed into the freeze decision.

### 8.6 Lesson — conceptual soundness does not guarantee empirical adoption

Two ideas were conceptually clean — a token model that knows token preference, a lane model that learned human lane habits — and both lost to a cheaper rule once the rule got good enough. The honest reading (recorded by the user) is that the models never learned from *calibrated* charts — the training corpus is "whatever humans charted," not "charts judged good" — so an uncalibrated corpus cannot yield a calibrated model; the gap is in the training setup, not the model capacity. ML is frozen behind a flag rather than deleted, so the contract survives a future re-design, but adding more injection points (fill-back, scratch seed, LN candidate) is gated on measured benefit, not intuition. The same shape recurs in the character-framework's audio-FFT stair detection: *a sound idea is a hypothesis, and a hypothesis must survive an empirical audit before it earns a place in the pipeline.*

---

## 9. Limitations and future work

### 9.1 Source dependence

The output is a projection of the source. A sparse or monotone source yields a sparse or monotone chart; the pipeline cannot invent rhythm because doing so would break the timing invariant (§2.4). This is a deliberate boundary, not a defect — but it means the pipeline is a *renderer*, not a *composer*.

### 9.2 The listening-proxy gap

The strongest open problem: RB-aligned metrics do not capture the ML-vs-RB listening difference (§8.2). Until a metric correlates with the blind-A/B impression, any "ML is worse" claim is metric-bounded. The planned next step is a listening-decomposition protocol (short A/B segments + user "stable" annotations) to narrow the hypothesis before designing a new metric — the first metric attempt (same-hand fat-tail) already failed as a chord-collapse artifact (§7.1).

### 9.3 LN-style blindness

Both RB and ML are blind to the source's LN *style*: `build_pool_universe` and the labeling pipeline both flatten a Long event to a `(start, token)` pair, discarding hold length. So the training data itself has no LN-style dimension, and the RB gate is a fixed 800 ms. A source-LN-signal infrastructure (preserve `(start, end)` hold ticks; derive per-package LN statistics; drive a dynamic gate) is the prerequisite for both a per-song RB gate and any future LN-aware model.

### 9.4 Resume API v1 scope

v1 is RB-only and single-pass: ML resume, partial (region-local) conformance, and the centroid two-sided lookahead (E-γ) are all out of scope. The chord-composition divergence (§7.4) and the scratch-lane lookahead gap (§7.3) are known and tracked.

---

## Appendix

### A. Hyperparameter reference (intensity = 5 / scratch = 5 defaults)

```text
# FX classification
FX_DURATION_THRESHOLD          = 1000   # ms
FX_ATTACK_THRESHOLD            = 20     # percentile
FX_ORIGIN_FILTER_ENABLED       = true

# Band whitelist
BAND_QUOTA_RATIO               = 0.20
RARE_OCCURRENCE_THRESHOLD      = 3
WINDOW_SIZE                    = 8      # measures
WINDOW_RESCUE_THRESHOLD        = 0.40
WHITELIST_DURATION_MAX         = 1055   # ms

# Within-idx reorder
USAGE_PENALTY_FIRST            = 10.0   # attack-pct points / prior use
USAGE_WEIGHT_SPREAD            = 1000.0 # Hz ≈ 1 prior use

# Phase
PHASE_MERGE_RATIO_MAX          = 0.289

# Lane assignment (centroid)
PLACEMENT_RANDOM_SEED          = 42
LANE_STEP_MAX                  = 4.0
CENTROID_EPSILON_RANDOM        = 0.30
CENTROID_STEP_UNIT_FLOOR       = 300    # Hz

# Stream / hand
STREAM_CHORD_RATIO_MAX         = 0.311
STREAM_MAX_SAME_HAND           = 2
MEASURE_NOTE_CAP               = 32

# Jack (BPM-aware)
MIN_JACK_DELTA_TICKS           = 15
MIN_JACK_DELTA_MS              = 102
MAX_JACK_STREAK                = 2

# Chord
MAX_CHORD_SIZE                 = 3
CHORD_MATE_SPREAD_MIN_GAP      = 2      # lanes (soft)

# Scratch
SCRATCH_MIN_INTERVAL           = 16     # ticks
SCRATCH_MAX_PER_MEASURE        = 4
SCRATCH_RUSH_WINDOW            = 3
SCRATCH_RUSH_THRESHOLD         = 3
SCRATCH_RUSH_REST_MEASURES     = 4
SCRATCH_FALLBACK_DURATION_MAX  = 300    # ms

# LN
LN_MIN_DURATION_MS             = 800
LN_MAX_HOLD_TICKS              = 96     # 2-beat visible cap

# Density rebalance
DENSITY_REBALANCE_MAX_DELTA    ≈ 0.21
```

Full table and lerp curves: see `compute_intensity_params` in `placement_engine.py`.

### B. Conformance check table

| Check | Stage | Invariant |
|---|---|---|
| A whitelist hard-filter | Placement | placed tokens pass the FX/unknown hard filter |
| B timing preservation | Placement | every placed note sits on a source onset |
| C jack prohibition | Placement | no same-lane repeat below the jack floor (scratch excluded) |
| D fallback behavior | Placement | primitive-failed measures residualize cleanly |
| E candidate collision | Placement | no `(pos, lane)` or `(pos, token)` double |
| F scratch constraints | Placement | scratch interval / density / RUSH respected |
| G seeded reproducibility | Placement | same seed → identical placement |
| K measure cap | Placement | no measure exceeds `MEASURE_NOTE_CAP` |
| J density rebalance | Placement | manual-inspection only (DR-J1) |
| A placed completeness | Writer | all placed events rendered |
| B residual completeness | Writer | all residuals rendered to BGM |
| C timing-line preservation | Writer | BPM / STOP / scale lines byte-identical to source |
| D no-original-playable leak | Writer | no source playable note passes through unaltered |

### C. CLI reference

```text
python run_pipeline.py --folder <package> [options]

--intensity <1-20>           note aggressiveness (default 5)
--scratch <1-20>             scratch frequency, source-aware mirror (default 5)
--ln                         enable LN post-processing
--ml                         enable ML soft-ranking (non-recommended, §8)
--model-token / --model-lane TorchScript paths (required with --ml)
--bms <filename>             explicit source chart select
--seed <int|random>          placement seed (default 42)

# Resume API (§5)
--resume-state <path>        carry-over state JSON (resume mode)
--start-measure <M>          resume range start
--end-measure <N>            resume range end
--next-chord-lookahead <p>   N+1 first-chord boundary input (requires --resume-state)
--finalize <events.json>     post-processing-only over spliced events
```

### D. Operational corpus (source package set)

13 source packages spanning the characters the policy must survive:

| Package | Stress tested |
|---|---|
| addiction | standard chart — RB/ML comparison origin |
| mightyA | dense chart — fill-back streak, chord-collapse artifact (§7.1) |
| blacksphere | short-LN style, MIXED-chord composition |
| signal | distraction-style short keysounds (reroll demo, §7.3) |
| bumblebee | hardtek, token-rich (baseline + smoke anchor) |
| tsuramic | general stream |
| marion | melodic, multi-chart variant select |
| lepontinia | rare-token rescue (§4.2) |
| happiness | 242 BPM + LN-heavy (§7.2) |
| sacrifice | LN + collision |
| nakama | coverage gap (soft) |
| egosa, wanwan | baseline-set general charts |

Regression baseline: 6 of these (bumblebee / egosa / lepontinia / signal / tsuramic / wanwan) × {RB, ML} × {bms, json} at lv5 (§6.3).

---

*This report describes BMS.Generator's note-placement pipeline as of 2026-05-25. The source code is the authoritative reference for exact behavior; this document is a narrative synthesis for readers.*


