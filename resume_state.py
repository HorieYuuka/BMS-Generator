"""Resume state serialization.

This module defines:
- Initial state factory for M=0 reroll / cascading first input.
- Extraction (engine internal state → JSON-serializable dict).
- Loading (JSON dict → engine internal state).

RB-only. ML carry-over state (token_context / lane_context /
global_lane_counts) is not yet supported.

RNG strategy = β-1 per-measure deterministic (§23.4 / DR-23-5).
The state's `rng.seed` is the chart-level base seed; per-measure RNG is
recomputed inside the measure loop as `Random(seed * 1_000_000 + measure)`.

Schema history:
- resume-v1 — initial Resume API (2026-05-25).
- resume-v2 (2026-06-11) — adds `token_lane_memory` ({token: [lane, measure]})
  for the §9.8 token→lane affinity carry-over. v1 states are rejected with
  ValueError (resume states are ephemeral; regenerate the prefix).
"""

from collections import defaultdict, deque

SCHEMA_VERSION = "resume-v2"
RNG_STRATEGY = "per-measure-deterministic"
SUPPORTED_MODE = "RB-v1"


def make_initial_resume_state(seed: int, mode: str = SUPPORTED_MODE) -> dict:
    """§23.3 initial state factory.

    Used when M=0 is rerolled (no prior measure to inherit from), or as the
    first input in a cascading reroll chain.

    `after_measure: -1` signals "no measure has been processed yet" — the
    next measure to place is M=0.
    """
    if mode != SUPPORTED_MODE:
        raise ValueError(
            f"Unsupported resume mode: {mode!r}. v1 supports {SUPPORTED_MODE!r} only."
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "after_measure": -1,
        "rng": {
            "strategy": RNG_STRATEGY,
            "seed": int(seed),
        },
        "jack_state": {},
        "jack_streak": {},
        "centroid": {"prev_lane_idx": 3, "prev_centroid": None},
        "hand": {"last_hand": "balanced", "streak": 0},
        "token_usage": {},
        "token_lane_memory": {},
        "scratch": {
            "jack_scr_tkey": -999,
            "scratch_history": [],
            "scr_rest_remain": 0,
        },
    }


def extract_resume_state(*, seed_int, after_measure,
                         jack_state, jack_streak, centroid_state,
                         hand_state, token_usage,
                         jack_scr_tkey, scratch_history, scr_rest_remain,
                         token_lane_memory=None) -> dict:
    """Serialize per-measure loop carry-over state at the end of `after_measure`.

    Inputs match the variable shapes in `placement_engine.run_per_measure_loop`:
    - jack_state: dict[lane: str, last_tkey: int]
    - jack_streak: defaultdict(int) keyed by lane
    - centroid_state: {prev_lane_idx: int, prev_centroid: float|None, step_unit: float}
    - hand_state: tuple (last_hand: "L"|"R"|"balanced", streak: int)
    - token_usage: defaultdict(int) keyed by token
    - jack_scr_tkey, scr_rest_remain: int
    - scratch_history: deque
    - token_lane_memory: dict[token: str, (lane: str, measure: int)] — §9.8
      affinity memory (resume-v2)

    Output schema: §23.3.

    Note: centroid `step_unit` is excluded from state — it is chart-input
    deterministic and reconstructed by the resume entry caller, not carried.
    """
    last_hand, streak = hand_state
    return {
        "schema_version": SCHEMA_VERSION,
        "after_measure": int(after_measure),
        "rng": {
            "strategy": RNG_STRATEGY,
            "seed": int(seed_int),
        },
        "jack_state": {str(k): int(v) for k, v in jack_state.items()},
        "jack_streak": {str(k): int(v) for k, v in jack_streak.items()},
        "centroid": {
            "prev_lane_idx": int(centroid_state["prev_lane_idx"]),
            "prev_centroid": (None if centroid_state.get("prev_centroid") is None
                              else float(centroid_state["prev_centroid"])),
        },
        "hand": {
            "last_hand": str(last_hand),
            "streak": int(streak),
        },
        "token_usage": {str(k): int(v) for k, v in token_usage.items()},
        "token_lane_memory": {
            str(k): [str(lane), int(m)]
            for k, (lane, m) in (token_lane_memory or {}).items()
        },
        "scratch": {
            "jack_scr_tkey": int(jack_scr_tkey),
            "scratch_history": [int(x) for x in scratch_history],
            "scr_rest_remain": int(scr_rest_remain),
        },
    }


def load_resume_state(state: dict, *, scratch_history_maxlen: int):
    """Deserialize §23.3 dict → engine internal state.

    Returns a tuple ready to be unpacked into `run_per_measure_loop` carry-over
    variables (see `extract_resume_state` for the inverse direction):

        (after_measure, seed_int,
         jack_state, jack_streak, centroid_state,
         hand_state, token_usage,
         jack_scr_tkey, scratch_history, scr_rest_remain,
         token_lane_memory)

    `centroid_state["step_unit"]` is left unset — caller (resume entry) must
    populate it from chart context before entering the measure loop.

    Raises ValueError on schema_version or RNG strategy mismatch.
    """
    sv = state.get("schema_version")
    if sv != SCHEMA_VERSION:
        raise ValueError(
            f"Resume state schema_version mismatch: got {sv!r}, "
            f"expected {SCHEMA_VERSION!r}. State was produced by a different "
            f"engine version; cannot resume."
        )

    rng_meta = state.get("rng") or {}
    strategy = rng_meta.get("strategy")
    if strategy != RNG_STRATEGY:
        raise ValueError(
            f"Resume state RNG strategy mismatch: got {strategy!r}, "
            f"expected {RNG_STRATEGY!r} (v12 §23.4 β-1). cannot resume."
        )

    seed_int = int(rng_meta["seed"])
    after_measure = int(state["after_measure"])

    jack_state = {str(k): int(v) for k, v in state.get("jack_state", {}).items()}

    jack_streak = defaultdict(int)
    for k, v in state.get("jack_streak", {}).items():
        jack_streak[str(k)] = int(v)

    centroid_in = state.get("centroid") or {}
    centroid_state = {
        "prev_lane_idx": int(centroid_in.get("prev_lane_idx", 3)),
        "prev_centroid": centroid_in.get("prev_centroid"),
        # step_unit: caller injects from chart context.
    }

    hand_in = state.get("hand") or {}
    hand_state = (
        str(hand_in.get("last_hand", "balanced")),
        int(hand_in.get("streak", 0)),
    )

    token_usage = defaultdict(int)
    for k, v in state.get("token_usage", {}).items():
        token_usage[str(k)] = int(v)

    token_lane_memory = {
        str(k): (str(v[0]), int(v[1]))
        for k, v in state.get("token_lane_memory", {}).items()
    }

    scratch_in = state.get("scratch") or {}
    jack_scr_tkey = int(scratch_in.get("jack_scr_tkey", -999))
    scratch_history = deque(
        [int(x) for x in scratch_in.get("scratch_history", [])],
        maxlen=scratch_history_maxlen,
    )
    scr_rest_remain = int(scratch_in.get("scr_rest_remain", 0))

    return (
        after_measure, seed_int,
        jack_state, jack_streak, centroid_state,
        hand_state, token_usage,
        jack_scr_tkey, scratch_history, scr_rest_remain,
        token_lane_memory,
    )
