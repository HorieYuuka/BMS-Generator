# BMS.Generator — 소스 충실 채보 생성 파이프라인: Band 화이트리스트, Centroid 레인 배정, 그리고 결정론적 Resume API

**부제**: 소스 곡의 키음 풀로부터 연주 가능한 BMS 채보를 재구성하되, 소스의 타이밍을 보존하면서 부분 재생성(reroll)까지 지원할 만큼 결정론적인 룰 기반 파이프라인의 설계와 검증.

**버전**: v12 (2026-05-25), draft 더블 플레이 합성 addendum 포함 (§8; 2026-06-14)

> 이 문서는 BMS 커뮤니티를 대상으로 한 기술 보고서다. BMS 채보 포맷과 리듬게임 채보에 대한 기본 이해를 전제하되, 도입하는 모든 용어·정책·상수는 본문에서 설명한다.

---

## 초록 (Abstract)

대부분의 자동 채보 작업은 이 과제를 *난이도 합성* — 목표 난이도의 채보를 생성 — 으로 본다. BMS.Generator 는 반대 입장을 취한다: 채보를 *소스 충실 재구성*으로 다룬다. 소스 BMS 파일이 주어지면, 파이프라인은 그 곡 자신의 키음 자산(`#WAV` 토큰 풀)을 재사용하고, 요청된 강도에서 연주 가능한 노트 배치를 **타이밍 변경 없이** 다시 도출한다. 결과물이 원곡처럼 들리는 이유는 원래의 소리를 원래의 onset 에 그대로 쌓아 만들기 때문이다.

파이프라인은 룰 기반(RB)이다. 머신러닝 경로(토큰 선택 + 레인 배정 모델)도 학습·통합해 `--ml` 플래그 뒤에 두었으나, 통계적 평가에서 룰 기반 대비 측정 가능한 우위가 없어 운영상 동결(freeze)되어 있다(§9). RB 설계의 핵심 기여는:

1. **Band 기반 quota 화이트리스트** + 희소 토큰 rescue — 단일 global hard filter 대신 spectral-band 상대 occurrence quota 로 어떤 풀 토큰이 연주 가능한지 선택하여, 멜로디적이지만 드물게 등장하는 토큰을 보호한다.
2. **Centroid 기반 상대 레인 배정** — 각 노트를 음악의 spectral-centroid 궤적(밝아지면 오른쪽, 어두워지면 왼쪽)을 saturating 곡선으로 따라가며 키 레인에 매핑한다. 무작위 배정이 아니다.
3. **소스 인지(source-aware) 스크래치 정책** — 절대 밀도 테이블에서 스크래치를 합성하는 대신, 소스 자신의 measure 별 스크래치 밀도에 레벨 배율을 곱해 미러링한다.
4. **결정론적 Resume API** — measure 단위 RNG 격리(β-1)로 단일 measure(또는 measure 구간)를 독립적으로 재현 가능하게 만들어, 외부 에디터가 나머지를 건드리지 않고 채보 일부를 reroll 할 수 있게 한다.

draft 확장(§8)은 이 기제를 재사용해 SP 소스로부터 **더블 플레이(14키)** 채보를 합성한다. 같은 타이밍 불변식을 지키고 SP 출력을 byte-identical 로 둔 채, per-note *side* 결정만 더한다.

검증은 corpus 통계가 아니라 구성(by construction)으로 한다: 17개 항목 conformance check 스위트(§6.1), 6곡 × {RB, ML} byte-identical 회귀 baseline, 9-케이스 Resume API smoke 스위트. 운영 corpus 는 코드-밀집·LN-과다·스크래치-과다·BPM-trick 채보를 아우르는 13개 소스 패키지다.

---

## 1. 서론

### 1.1 문제 정의

BMS 곡은 오디오 샘플 묶음(`#WAV` 슬롯)과, 그 샘플을 레인×타임라인에 배치한 하나 이상의 채보로 배포된다. 곡의 샘플과 *하나*의 참조 채보만 주어졌을 때, 다음을 만족하는 **새 연주 가능 채보**를 합성할 수 있는가:

- 곡 자신의 소리를 쓰고(그래서 여전히 그 곡처럼 들리고),
- 곡 자신의 onset 에 노트를 올리며(그래서 음악적으로 정직하고),
- 조절 가능한 강도로 연주 가능하고(노트 밀도 / 스크래치 빈도 / 롱노트 사용량),
- **재현 가능**한가 — 같은 입력은 항상 같은 채보를 내고, 단일 measure 를 독립 재생성할 수 있는가?

순진한 접근 — "모든 소리 이벤트를 노트로" — 은 연주 불가능한 벽을 만든다. 반대 — "목표 난이도에 맞춰 노트를 합성" — 은 곡의 정체성을 버린다. 설계 문제는 그 중간이다: 어떤 소리 이벤트를 연주 노트로 *선택*하고, 자연스럽게 읽히는 레인에 *배정*하며, 요청 강도로 밀도를 *형성*하되, 노트를 소스 onset 에서 절대 옮기지 않는 것.

### 1.2 목표

**적절한 채보(adequate chart)**를 만든다 — 최대로 어렵거나 최대로 영리한 채보가 아니라. 선택한 강도에서 곡을 충실하고 연주 가능하게 렌더링했다고 사람이 받아들일 채보. 구체적으로 매 실행마다 다음을 낸다:

- `placement_result.bms` — BMS 포맷의 연주 채보,
- `placement_result.json` — 배치/잔여(residual) 이벤트 집합 + 진단,
- 패키지의 기존 채보 대비 유사도 리포트.

"적절함" 프레이밍(v12 §21)은 의도적이다: 룰 기반 정책은 어떤 미학을 *최대화*하기보다 *명확히 정의 가능한 나쁜 패턴*(바닥값보다 빠른 same-lane 잭, 과대 코드, hand-balance 붕괴)을 *차단*하는 것을 목표로 한다. 미학은 소스와 사용자에게 맡긴다.

### 1.3 기여

- **C1 — Band quota 화이트리스트 + rare rescue.** 토큰 선택은 global hard filter(v9) → spectral-band 상대 occurrence quota(v10) → quota 확대(0.20) + occurrence ≤ 3 비-FX 토큰의 명시적 rescue(v11)로 진화했다. flat occurrence cut 이 떨어뜨릴 멜로디 하이라이트를 지킨다. (§4.2)
- **C2 — Centroid 상대 레인 배정.** 레인은 음악의 spectral-centroid 움직임을 곡별 자동 보정 saturating 곡선으로 따라가며 선택하고, ε-greedy 다양화를 더한다. v9 의 무작위 레인 선택을 대체. (§4.6)
- **C3 — 소스 인지 스크래치 미러.** primary 모드에서 출력은 소스의 measure 별 스크래치 개수 × (level / 5) 를 미러링하며, 소스 자신의 최소 간격을 spacing 바닥값으로 쓴다. 레벨 인덱스 절대 테이블이 아니다. (§4.7)
- **C4 — 결정론적 Resume API.** measure 단위 RNG(β-1: `Random(seed × 10⁶ + measure)`)가 chart 전체 sequential RNG 결합을 제거하여, measure 구간 `[M, N]` 을 직렬화된 carry-over 상태로부터 재생성할 수 있게 한다. N+1 경계 lookahead 옵션 포함. 외부 에디터의 단일 measure reroll 의 기반. (§5)

- **C5 — 더블 플레이 합성 (draft).** `--dp` 모드가 같은 소스 충실 입장 아래 SP 소스로부터 14키 DP 채보를 합성한다: 각 phrase 를 양손에 부하-분산(기본 balance, §8.2)으로 나누고, 타이밍 불변·jack-aware 레인 후처리기로 결과를 정제한다. onset 을 옮기지 않고 SP 출력을 바꾸지 않는다. (§8)

다섯 번째 횡단 속성 — **결정론(determinism)** — 도 작업 중 단단해졌다: `PYTHONHASHSEED` 의존 set/dict 순회 비결정성 한 부류를 발견·제거했고(§5.5), 6곡 회귀 baseline 이 byte-identical 출력을 지킨다.

---

## 2. 배경

### 2.1 BMS 포맷과 토큰 풀

BMS 채보는 measure 별로 키잉된 채널 라인의 텍스트 파일이다. 각 라인 `#mmmCC:...` 는 measure `mmm`, 채널 `CC` 에 두 글자 base-36 **토큰**을 균등 분할 sub-position 으로 배치한다. 파이프라인은 위치를 **idx192** — measure 당 192 틱(흔한 분할의 LCM) — 로 정규화하므로, 전역 타임스탬프는 `tkey = measure × 192 + idx192` 이다.

본문에서 쓰는 핵심 용어:

- **token** — `#WAV` 슬롯 id; 하나의 오디오 샘플.
- **event** — `(measure, idx192)` 에서의 한 토큰 발생.
- **pool universe** — 검토 대상이 되는 모든 사용된 토큰 이벤트.
- **playable whitelist** — 연주 노트가 될 수 있도록 허용된 토큰 부분집합.
- **residual** — 연주 노트 대신 BGM/자동재생 객체(채널 `01`)로 남는 토큰 이벤트. residual 화는 키 입력을 요구하지 않으면서 소리를 onset 에 보존한다.
- **spectral centroid** — 토큰 샘플의 주파수 영역 무게중심; 레인 배정을 구동하는 밝기 proxy.

핵심 불변식: **타이밍은 절대 바뀌지 않는다**. 노트는 자기 소스 onset 에서 연주 가능해지거나, 같은 onset 에서 residual 화된다. 파이프라인은 *연주 여부*와 *어느 레인*만 결정하고, *언제*는 결정하지 않는다.

### 2.2 4단계 파이프라인

```
mix_generation.py  →  placement_engine.py  →  bms_writer.py  →  similarity_check.py
   (분석)                (배치)                 (출력)            (검증)
```

| 단계 | 입력 | 출력 | 역할 |
|---|---|---|---|
| **MixGeneration** | 소스 패키지(`.bms` + `.wav`들) | `token_analysis.json`, `mix_generation_log.json` | 소스 채보 선택, 각 키음 디코드, 토큰별 spectral / attack feature 계산(캐시) |
| **PlacementEngine** | `token_analysis.json` + 소스 이벤트 | `placement_result.json` | 정책 핵심 — 화이트리스트, phase, measure별 배치, 스크래치, LN, density rebalance |
| **BMSWriter** | `placement_result.json` + 소스 `.bms` | `placement_result.bms` | 배치 노트 + residual BGM 을 유효한 BMS 파일로 렌더, 소스 타이밍 라인 보존 |
| **SimilarityCheck** | 출력 `.bms` + 패키지 채보 | `similarity_report.json` | 패키지 기존 채보 대비 중첩 리포트(진단용, 비-gating) |

`run_pipeline.py` 가 네 단계를 연결한다. 운영 모드는 **RB 단독**이며, `--ml` 플래그는 존재하나 비권장이다(§9).

### 2.3 룰 기반 vs 머신러닝 — 왜 RB 가 동결-on 인가

두 모델을 학습했다: `TokenSelectionModel`(어떤 토큰을 연주)과 `LaneAssignmentModel`(어느 레인). v9 baseline 에서 레인 모델은 chance 25% 대비 ~50% top-1 정확도에 도달 — 큰 외견상 이득. 그러나 그 이득은 v9 가 쓰던 *무작위* 레인 baseline 기준이었다. 룰 기반 경로가 centroid 레인 배정(C2)을 채택하자 RB baseline 자체가 모델이 배운 구조 대부분을 포착했고, 2026-05-03 통계 평가는 ML 모델이 RB 경로 대비 **측정 가능한 우위가 없음**을 발견했다(v12 §19.5).

한 가지 caveat 를 정직하게 기록한다: 블라인드 A/B 청취에서 ML 출력이 때때로 "더 안정적/더 인간적"으로 느껴졌으나, RB-aligned metric 중 그 인상을 잡은 것이 없었다. 따라서 판정은 "측정 가능한 우위 없음"이지 "우위 없음"이 아니다 — metric-blindness 가능성(§9.2, §10.2). ML 은 삭제가 아니라 플래그 뒤에 동결되어 있다.

### 2.4 프레임워크 입장: 소스 의존은 숨기지 않고 인정한다

이 파이프라인은 난이도 엔진이 **아니다**. 리듬을 발명하지 않고, onset 을 옮기지 않으며, 목표 θ 를 겨냥하지 않는다. 그 정체성은 소스에서 온다: 연주 배치는 소스 자신의 소리 이벤트를 연주 가능한 레인 공간에 *투영*한 것이다. 이 입장은 범위에 직접적 결과를 갖는다 — *재타이밍*을 요구하는 모든 것(새 리듬 figure 합성, 희소 구간 "개선")은 구성상 범위 밖이다(v12 §2 FORBIDDEN). Resume API(§5)도 같은 경계를 지킨다: measure 를 reroll 하면 *어느 레인 / 어느 토큰*은 바뀌어도 *언제*는 바뀌지 않는다.

---

## 3. 아키텍처

### 3.1 단계별 상세

**MixGeneration** 은 패키지에서 후보 `.bms` 들을 스캔하고, coverage 와 연주 노트 수로 하나를 선택(또는 명시 `--bms` 존중)한 뒤, 선언된 모든 키음을 디코드한다. 토큰마다 attack 강도(RMS / peak), duration, STFT spectral feature(centroid, flatness, low-frequency ratio, zero-crossing rate)를 계산한다. 결과는 `token_analysis.json` 에 캐시되어 재실행 시 디코딩을 건너뛴다.

**PlacementEngine** 은 정책 핵심이자 §4–§6 의 주제다. pool universe 를 만들고, band 화이트리스트를 구성하고, 채보를 phase 로 분할한 뒤, hard constraint 아래 measure 를 좌→우로 배치하고, 스크래치를 삽입하고, 롱노트를 후처리하고, 마지막으로 chart 전체 density 를 rebalance 한다.

**BMSWriter** 는 결과를 BMS 로 렌더한다. 배치 노트는 key/scratch 채널로, residual 토큰은 BGM 채널 `01` 로 가고, 소스 타이밍 라인(BPM 변경, STOP, measure-length scale)은 그대로 보존된다. writer 측 conformance check(A–D, §6.1)이 완전성과 원본 연주 노트가 변형 없이 새어나가지 않았는지 검증한다.

**SimilarityCheck** 는 출력을 패키지 기존 채보와 비교해 중첩을 리포트한다. 진단용일 뿐 — 생성을 gating 하지 않는다.

### 3.2 데이터 흐름

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

`placement_result.json` 은 정책과 렌더러 사이의 계약이다. additive 하게 버전 관리된다: Resume API(§5)는 resume 모드에서 `mode` / `end_state` 필드를 추가하되 default 경로에서 BMSWriter 가 소비하는 필드는 바꾸지 않으므로, 기존 소비자는 영향받지 않는다.

---

## 4. 배치 정책 설계

PlacementEngine 은 모든 풀 이벤트에 대해 연주 노트가 될지, 어느 레인일지 결정하고, 그 결과 밀도를 요청 강도로 형성한다. 정책은 measure 좌→우 단일 패스(per-measure loop)이며, 앞에 pool/화이트리스트 구성, 뒤에 density rebalance 가 둘러싼다. 아래 모든 상수는 intensity = 5 / scratch = 5 기본값이며, 둘 다 lerp 곡선으로 스케일한다(§6.2).

### 4.0 구현 구조

per-mechanism 정책에 앞서, 코드 레벨 형태. PlacementEngine 은 하나의 상태 보유 loop 를 둘러싼 대체로-순수 함수들의 시퀀스다:

| 단계 | 함수 | 산출 | 정책 § |
|---|---|---|---|
| pool 빌드 | `build_pool_universe` | pool 토큰 + occurrence 카운트 + events | 4.1 |
| feature | `compute_attack_percentile` / `compute_intensity_origin` / `classify_fx` | pct_map / intensity_origin / FX 플래그 | 4.3 / 4.1 |
| 화이트리스트 | `build_whitelist` | whitelist, excluded, band 통계 | 4.2 |
| phase | `segment_phases` | rush / normal / rest 블록 | 4.4 |
| scratch seed | `_determine_scratch_seeds` | scratch 토큰 + 모드 | 4.7 |
| **메인 loop** | `run_per_measure_loop` | placed + residual events (+ end_state) | 4.3–4.7 |
| ↳ measure별 | `_place_measure_constrained` | 제약 아래 배치 | 4.5 |
| ↳ note별 | `_centroid_lane_select` | 레인 | 4.6 |
| LN | `run_ln_postprocess` | LN 승격 events | 4.8 |
| scratch adj | `run_scratch_adjustment` | LN-aware scratch thinning | 4.7 |
| density | `run_density_rebalance` | 균형 events | 4.9 |
| 검증 | `_run_conformance` | check pass/fail | 6.1 |

**Per-measure loop** (`run_per_measure_loop` — 유일한 상태 보유 패스; carry-over 상태는 §5.2):

```text
for measure in [start .. end]:                 # 기본 0 .. measure_max
    rng = Random(seed × 10⁶ + measure)          # β-1 (§5.1)
    curr = reorder_within_idx(cands[measure])   # §4.3
    if ml: curr = ml_token_rerank(curr)         # §9.4 (옵션)
    placed, hand_state, residual =
        _place_measure_constrained(curr, rng, hand_state, jack_state, …)
    token_usage += placed                       # 저사용 부스트가 다음 measure 에 전달
    if scratch_active: insert_scratch(measure)  # §4.7
    residualize(unplaced)
# 이후 (전체 chart / finalize 만): LN → scratch_adj → density_rebalance → conformance
```

**제약 게이트 순서** — `_place_measure_constrained` 안에서 각 후보가 measure-candidate 순서로 다음 게이트를 통과한다; 하나라도 실패하면 사유 코드와 함께 residual 로 떨어진다:

```text
1.  measure note cap           → "measure_cap"
2.  collision (token @ pos)     → "collision"
3.  chord-ratio cap             → "no_lane_available"
4.  chord-size cap              → "chord_size_cap"
5.  가용 레인 ≠ ∅                → "collision"
6.  jack floor (BPM-aware)      → "jack_violation"
7.  jack streak                 → "jack_violation"
8.  hand balance (T ≥ 10)       → "hand_balance"
9.  same-hand streak            → "no_lane_available"
10. chord-mate spread (soft)    → avail 좁힘, 떨어뜨리지 않음
11. boundary lookahead          → 마지막 코드만 (§5.4)
12. lane select: ML → centroid → fisher-yates
→ 배치; used_at_pos / jack_state / jack_streak / hand_state / centroid_state 갱신
```

게이트 순서는 load-bearing 이다: 게이트 6(jack)에서 거부된 노트는 게이트 8(hand balance)에 도달하지 않으므로, 진단의 사유 코드는 *첫* 위반 제약을 반영하지 모든 위반을 반영하지 않는다. §4.1–4.9 가 각 단계의 정책과 근거를 설명한다.

### 4.1 Pool universe 와 residual 정책

pool universe 는 모든 사용된 토큰 이벤트다. 토큰은 샘플 duration 이 `FX_DURATION_THRESHOLD = 1000 ms` 초과, attack percentile 이 `FX_ATTACK_THRESHOLD = 20` 이하, 또는 key/scratch 채널에 한 번도 등장하지 않으면 **FX**(배경, 키 후보 아님)로 hard-exclude 된다. 샘플 디코드에 실패한 토큰은 **unknown** 으로 표시되고 절대 rescue 되지 않는다.

배치되지 않은 모든 것은 **residual** 이다: 자기 원래 onset 에서 BGM 채널 `01` 로 쓰인다. 이것이 곡을 *침묵시키지 않으면서* 연주성을 위해 노트를 떨어뜨릴 수 있게 하는 메커니즘이다 — 소리는 여전히 울리되, 키 입력을 더는 요구하지 않는다.

### 4.2 Band 기반 quota 화이트리스트 (C1)

화이트리스트는 세 형태로 진화했다:

- **v9** — duration / attack / occurrence 에 대한 단일 global hard filter.
- **v10** — band 기반 quota: eligible 토큰을 centroid 3분위로 세 spectral band(lo / mid / hi)로 나누고, 각 band 가 occurrence 가중 rank 상위 `BAND_QUOTA_RATIO` 비율을 유지.
- **v11/v12** — 비율을 `0.15 → 0.20` 으로 올리고 **rare-token rescue** 를 추가.

band 안에서 토큰은 다음으로 랭크된다:

$$\text{rank}(t) = \text{occ}(t) - 5 \cdot \max\!\left(0, \frac{\text{dur}(t) - \text{WL}_{\text{dur}}}{\text{WL}_{\text{dur}}}\right)$$

`occ` 는 총 occurrence, 패널티는 과대 길이 샘플을 억제한다(lv5 에서 `WL_dur = 1055 ms`). 각 band 는 `max(3, round(0.20 × |band|))` 토큰을 유지하고, 나머지는 `band_quota` 사유로 soft-exclude 된다.

**Rare rescue.** flat occurrence cut 은 드물게 작곡된 멜로디 하이라이트(예: Lepontinia m16 의 8X/8Y/9B)를 떨어뜨린다. 그래서 총 occurrence ≤ `RARE_OCCURRENCE_THRESHOLD = 3` 인 *비-FX* 토큰은 band rank 와 무관하게 화이트리스트에 복귀한다. FX 는 §4.1 에서 이미 제거됐으므로 `band_quota` 사유는 항상 "비-FX 인데 rank-cut" 을 뜻하며, 정확히 보호할 가치가 있는 집합이다.

**Windowed rescue.** 8-measure 윈도우(`WINDOW_SIZE = 8`)마다, measure 의 화이트리스트 통과율이 `WINDOW_RESCUE_THRESHOLD = 0.40` 미만이면, 제외된 토큰(윈도우 레벨 occurrence 로 랭크)을 그 임계까지 rescue 한다. 국소적으로 토큰이 빈약한 measure 가 비는 것을 막는다.

### 4.3 토큰 강도와 within-idx reorder

토큰의 **강도(intensity)**는 attack percentile(`pct_map`)이며, 풀 전반의 RMS / peak 에서 계산된다. 이것이 순서를 구동한다: 더 큰 onset 이 한 위치의 첫 노트로 선호된다.

**within-idx reorder**(v11, C1-인접)는 소수의 큰 토큰이 배치를 독점하는 자연스러운 head-heavy 분포를 상쇄하도록 같은 위치 순서를 정제한다. 최고 attack 첫 픽 이후, 남은 코드-메이트는 다음으로 점수화된다:

$$\text{score}(t) = \min_{c \in \text{chosen}} |c_{\text{cent}} - t_{\text{cent}}| - 1000 \cdot \text{usage}(t)$$

— 이미 뽑힌 메이트와의 spectral 거리를 최대화(다양성)하면서 chart 전반에서 이미 많이 쓰인 토큰을 패널티(`USAGE_WEIGHT_SPREAD = 1000 Hz ≈ 1회 사용`). 첫 픽 자체도 사용 패널티를 진다(`USAGE_PENALTY_FIRST = 10` attack-pct point / 사용 1회).

### 4.4 Phase 분할

채보는 smoothed 후보 밀도 점수로 4-measure 블록에 걸쳐 `rush` / `normal` / `rest` phase 로 분할되며, 비율 격차가 `PHASE_MERGE_RATIO_MAX = 0.289` 미만인 인접 블록을 병합한다. phase 는 **진단 전용** — 구간을 리포트용으로 라벨링할 뿐 배치를 gating 하지 않는다. (이전 버전은 phase-adaptive relaxation 을 썼으나 windowed rescue 로 대체됨, §4.2.)

### 4.5 배치 제약

각 measure 는 hard constraint 아래 순서대로 배치된다. 제약을 실패한 노트는 residual 로 떨어지며(onset 보존), 절대 재타이밍되지 않는다.

- **충돌(Collision)** — `(위치, 레인)` 당 토큰 하나; 한 위치에서 토큰당 레인 하나.
- **잭 바닥값(Jack floor)** — 같은 레인은 `effective_min_ticks = max(MIN_JACK_DELTA_TICKS, ceil(MIN_JACK_DELTA_MS × bpm / 1250))` 보다 빠르게 반복될 수 없다. BPM-aware 항은 고정 틱 바닥값(lv5 에서 15)이 빠른 구간에서 올라가, 16분음 same-lane 반복이 모든 템포에서 차단됨을 뜻한다.
- **잭 스트릭(Jack streak)** — 한 레인은 강제 이탈 전 최대 `MAX_JACK_STREAK = 2` 연속 코드-앵커 반복을 가질 수 있다.
- **코드 크기 cap** — lv5 에서 위치당 최대 `MAX_CHORD_SIZE = 3` 동시 레인; 초과분은 residual 로(타이밍 보존, 코드만 얇아짐).
- **Hand balance** — measure 가 ≥ 10 노트가 되면 좌/우 비율을 `[0.30, 0.70]` 안에 유지; balance 를 깨뜨릴 노트는 덜 쓰인 손으로 유도되거나 떨어진다.
- **Same-hand streak** — 새 위치는 같은 손을 `STREAM_MAX_SAME_HAND = 2` 너머로 연장할 수 없다.
- **Chord-mate spread**(v11) — 코드-메이트 레인을 `CHORD_MATE_SPREAD_MIN_GAP = 2` 인덱스 이상 떨어뜨리는 soft 선호로, `{1,2,3}` 류 인접-레인 클러스터를 제거; 넓은 코드가 packing 을 강제하면 임의 레인으로 fallback.

### 4.6 Centroid 기반 상대 레인 배정 (C2)

레인은 무작위가 아니라 음악의 **spectral-centroid 궤적**을 따라 선택된다. 이전 노트의 레인 인덱스와 centroid 로부터, 다음 선호 레인은:

$$\text{step} = \text{sign}(\Delta) \cdot \text{LANE\_STEP\_MAX} \cdot \left(1 - e^{-|\Delta| / \text{step\_unit}}\right), \qquad \Delta = c_{\text{cur}} - c_{\text{prev}}$$

더 밝은 토큰(양의 Δ)은 손을 오른쪽으로, 어두운 토큰은 왼쪽으로 움직인다. saturating 지수는 작은 변화에 민감하되 큰 점프를 `LANE_STEP_MAX = 4` 레인에서 cap 한다. `step_unit` 은 곡별로 비-0 토큰 간 centroid 차이의 중앙값으로 자동 보정되며 `CENTROID_STEP_UNIT_FLOOR = 300 Hz` 로 floor 되어, 같은 절대 밝기 변화가 곡의 spectral 범위와 무관하게 일관된 공간 step 으로 매핑된다.

**ε-greedy** 항(`CENTROID_EPSILON_RANDOM = 0.30`)은 30% 확률로 임의 가용 레인을 골라 centroid drift 를 끊고 단조로운 hand-walking 을 피한다. centroid 데이터가 없으면 Fisher-Yates shuffle 로 fallback.

### 4.7 스크래치 정책 — 소스 인지 미러 (C3)

스크래치는 세 모드:

- **primary** — 소스에 이미 스크래치 토큰(채널 `16`)이 있음. 출력은 소스의 measure 별 스크래치 개수 × `scale` 을 미러링하며, `scale = level / 5`(즉 scratch = 5 는 1:1 소스 미러). spacing 바닥값은 레벨 테이블이 아니라 소스 *자신*의 최소 간격 — 소스 작곡자가 이미 리듬을 빚었다.
- **fallback** — 소스에 스크래치 없음. key 토큰에서 레벨 인덱스 절대 cap(`SCRATCH_MAX_PER_MEASURE = 4`, scratch = 5 에서 `SCRATCH_MIN_INTERVAL = 16`) 아래 스크래치를 합성하며, 지속 burst 후 `SCRATCH_RUSH_REST_MEASURES = 4` 쿨다운을 넣는 RUSH-rest 규칙 적용.
- **disabled** — 스크래치 삽입 없음.

RUSH-rest 는 fallback 에서만 발동; primary 모드는 소스 pacing 을 신뢰하고 비활성화한다.

### 4.8 롱노트 후처리

배치 후, 적격 Tap 노트가 롱노트(LN)로 승격된다. Tap 은 토큰 샘플 duration 이 `LN_MIN_DURATION_MS = 800`(인간 LN duration 의 대략 p75 에 맞춘 게이트) 이상일 때 LN 후보다. 그려지는 hold 길이는 `LN_MAX_HOLD_TICKS = 96`(2-beat 가시 cap, v11)으로 cap 되어 긴 샘플이 화면을 채우는 막대를 그리지 않게 한다; *오디오* 샘플은 끝까지 재생된다 — 가시 막대만 cap. hold 는 선언된 경우 채보의 `#LNOBJ` 토큰으로 쓰인다.

알려진 긴장(§10.3): 800 ms 선정 게이트가 *자연스럽게 짧은* 인간 LN 의 ~75% 를 차단하므로, 파이프라인은 LN-과다 곡에서 LN 을 과소 생산한다. 게이트를 일률적으로 낮추면 모든 것을 over-LN 하므로, source-LN-signal 인프라가 곡별 게이트의 선행 조건이다(future work).

### 4.9 Density rebalance

마지막 단계는 4개 chart segment 에 걸쳐 노트 밀도를 균형 잡는다. 과부하 segment 는 최저 강도 토큰을 residual 로 떨어내고; 저밀도 segment 는 residual 에서 토큰을 끌어온다(fill-back). 보정은 **soft-knee 지수 damping** 을 써서 목표 근처에선 부드럽고 멀리선 단단하며, `DENSITY_REBALANCE_MAX_DELTA`(lv5 에서 ≈ 0.21)로 bound 된다. fill-back 은 같은 배치 제약(잭, 코드 크기, chord-mate spread)을 재적용하므로 rebalance 된 노트는 절대 불법이 아니다.

Density rebalance 는 국소가 아니라 **chart 전체**다 — Resume API 의 finalize 단계(§5.3)에 중요한 속성: 한 measure 를 reroll 한 뒤 finalize 하면 *다른* segment 의 노트가 움직일 수 있다. 이는 의도된 것(chart 일관성이 국소 편집 보존보다 우선)이지만 API 위에 만드는 에디터 UI 에서 반드시 드러내야 한다.

---

## 5. 결정론과 Resume API (C4)

동기가 된 사용 사례는 외부 에디터(BMS.Compare)가 생성된 채보의 일부를 **reroll** — 한 measure, 또는 measure 구간을 다른 seed 로 재생성 — 하면서 나머지는 건드리지 않는 것이다. 이를 위해 원래 엔진에 없던 두 가지가 필요했다: (a) measure 구간을 독립 재생성하는 능력, (b) "나머지를 건드리지 않음"이 byte 단위로 실제 성립하도록 하는 진짜 결정론. §5.1–5.5 가 둘 다 어떻게 달성됐는지 설명한다.

### 5.1 measure 단위 RNG (β-1)

원래 엔진은 chart 시작에 단일 `random.Random(seed)` 를 seed 하고 모든 measure 에 걸쳐 소비했다. 이는 모든 measure 를 이전 모든 measure 의 RNG draw 에 결합시킨다 — 부분 재생성에 치명적인데, measure M 을 재현하려면 먼저 measure 0…M-1 을 재생해야 하기 때문이다.

두 전략을 검토했다:

| | α — RNG 상태 직렬화 | β — measure 단위 RNG |
|---|---|---|
| 메커니즘 | `Random.getstate()`(Mersenne-Twister 624-tuple)를 carry-over 상태에 dump | `(seed, measure)` 로부터 measure 마다 fresh RNG seed |
| 구버전과 byte-identical | 예 | 아니오(출력 변경, baseline 재생성) |
| 스키마 위험 | 높음 — Python 내부 RNG 레이아웃 고정 | 없음 |
| undo/redo | 이전 RNG 상태 cascade 필요 | 간단 — measure_idx 만으로 재현 |

**β 를 선택했다**(single-stream-per-measure 변종인 "β-1"): per-measure loop 이 이제 다음을 seed 한다

$$\text{rng}_{\text{measure}} = \texttt{random.Random}(\text{seed} \times 10^{6} + \text{measure})$$

구현 중 함정 하나가 드러났다: Python 3.13 의 `random.Random` 은 tuple seed 를 거부한다(`TypeError: only int/float/str/bytes/bytearray`). 그래서 tuple-hash 형태를 위 산술 매핑으로 교체했고, `10⁶` offset 은 어떤 현실적 chart 길이에서도 collision-free 다(채보는 수백 measure). sub-decision(β-2: per-call-site keyed RNG)은 거부됐는데, RNG 소비 site 가 다섯 곳뿐이라 per-call-site 분할이 실질적 robustness 를 주지 않기 때문이다 — Resume API 가 필요로 하는 것은 정확히 measure 수준 격리다.

수용한 트레이드오프: chart 출력이 chart-wide-stream 시절과 더는 byte-identical 이 아니므로, `samples/baseline_lv5/` 회귀 baseline 을 β-1 으로 재생성했다.

### 5.2 Carry-over 상태

measure M 에서 resume 하려면, 엔진은 좌→우 loop 이 M-1 끝까지 쌓았을 cross-measure 상태를 상속해야 한다. 그 상태를 버전 명시 JSON(`schema_version: "resume-v1"`)으로 직렬화하면:

| 필드 | 역할 |
|---|---|
| `jack_state` (레인 → last tkey) | 잭 바닥값 delta 검사 |
| `jack_streak` (레인 → count) | 연속 same-lane cap |
| `centroid` (`prev_lane_idx`, `prev_centroid`) | centroid 궤적 연속성 |
| `hand` (`last_hand`, `streak`) | same-hand-streak cap |
| `token_usage` (토큰 → count) | 저사용 토큰 부스트 + fill-back 순서 |
| `scratch` (`jack_scr_tkey`, `scratch_history`, `scr_rest_remain`) | 스크래치 잭 / RUSH-rest 윈도우 |
| `rng` (`strategy`, `seed`) | β-1 base seed (per-measure RNG 을 여기서 재계산) |

Codex 감사가 이전 오식별 두 개를 정정했다: `lane_tkeys` 는 finalize 시 재구성(carry 아님), `hand_balance` 는 measure-local(chart-wide 누적은 진단 전용). centroid `step_unit` 은 carry 하지 *않는다* — chart-input 결정론이라 resume entry 가 재계산. ML carry-over(`token_context`, `lane_context`, `global_lane_counts`)는 **v1 범위 밖**(RB 전용); `rng.strategy` 나 `schema_version` 이 안 맞는 상태를 로드하면 `ValueError`.

### 5.3 Resume 모드와 finalize 모드

세 CLI entry 형태가 한 엔진을 공유한다:

```
(default)                  전체 chart, 동작 불변
--resume-state S \
  --start-measure M \
  --end-measure N          resume 모드: 상태 S 로부터 [M,N] 재생성,
                           raw events + end_state 출력, post-processing SKIP
--finalize EVENTS          finalize 모드: splice 된 전체 chart events 받아
                           post-processing 만 실행
```

**resume 모드**는 per-measure loop 을 `[M, N]` 으로 좁히고, carry-over 상태를 `S` 로 override 하며, 부분 `placement_result.json`(`mode: "resume"`, M…N events, cascading 용 `end_state`)을 낸다. LN / scratch-adjust / density-rebalance / conformance 를 완전히 건너뛴다 — 호출자가 raw events 를 splice 하고 나중에 finalize 한다. `run_pipeline.py` 에서 resume 모드는 BMSWriter 와 SimilarityCheck 도 건너뛴다(부분 chart 는 의미 있는 `.bms` 나 유사도가 없으므로).

**finalize 모드**는 그 역이다: 에디터가 splice 한 전체 chart events 가 주어지면, post-processing 체인을 고정 순서 — LN postprocess → scratch adjustment → density rebalance → conformance — 로 실행하고 정상 전체 `placement_result.json` 을 낸다. density rebalance 가 chart 전체(§4.9)이므로 finalize 는 reroll 한 영역 밖의 노트를 움직일 수 있다; 정책 결정은 명시적이다(v12 §23.6 / DR-23-4): **chart 일관성이 사용자 국소 편집 보존보다 우선**하므로, finalize 는 명시적 사용자 액션에서만 호출되고, 에디터는 reroll 영역이 유지 보장되지 않음을 경고해야 한다.

### 5.4 경계 lookahead (E-β)

resume 모드는 *왼쪽* 이웃을 완벽히 상속한다(carry-over 상태가 M-1 끝 상태). *오른쪽* 이웃은 보지 *못한다*: 엔진의 `next_cands` lookahead 는 measure N+1 의 raw 후보 풀만 읽고, N+1 의 이미 배치된 레인은 읽지 않는다. 그래서 reroll 영역의 N → N+1 경계가 충돌할 수 있다 — N+1 첫 코드와의 잭, 또는 N+1 이 이어받는 hand-streak.

양 이웃이 고정된 단일 measure reroll(동기 사례)에서 이는 중요하다. 수정(`--next-chord-lookahead`, v12 §23.7)은 호출자가 N+1 첫 코드를 넘기게 하고, 엔진은 *마지막 measure 의 마지막 코드에만* forward constraint 를 적용한다:

- **잭**(hard) — N+1 레인과 `effective_min_ticks` 안에서 충돌할 레인을 reject;
- **hand-streak**(soft) — 선택 손이 N+1 손과 같고 streak 이 cap 을 넘기면 반대 손을 선호.

centroid 양방향 보간(E-γ)은 보류됐다 — saturating-curve 레인 선택기를 재작성해야 하고, 잭/hand 과 달리 soft 선호이기 때문. N+1 첫 코드만 쓰며, 이후 코드는 그것으로 buffer 된다.

### 5.5 결정론 버그: PYTHONHASHSEED

β-1 이후 default 경로가 byte-identical 로 유지되는지 검증하던 중, *같은* 입력의 연속 두 실행이 *다른* 채보를 냈다. β-1 자체는 결정론적이다; 범인은 다른 데 있었다: `PYTHONHASHSEED` 가 unset(Python 기본)이면 문자열과 tuple-of-string 의 hash 가 프로세스마다 randomized 되어, 토큰 `set`/`dict` 순회가 매 실행 다른 순서를 내고 — 그 순서가 배치 결정에 새어들었다.

다섯 site 를 `sorted()` 또는 토큰 tie-break 로 단단히 했다: attack-percentile 정렬, 화이트리스트 eligible-순회·band-rank·rare-rescue 순회, measure별 scratch-candidate 순회. 여섯 번째(라벨링 파이프라인)는 학습 결정론을 위해 고쳤다. 수정 후 6곡 × {RB, ML} baseline 은 `PYTHONHASHSEED` 와 무관하게 실행 간 byte-identical 이다. 교훈은 상시 리뷰 항목으로 보관됐다(v12 §22 DR-23-6): *원소가 문자열인 `set`/`dict` 는 보유는 안전하나, 그 순회 순서가 결정에 들어가는 순간 정렬되어야 한다.*

---

## 6. Conformance 와 보정

파이프라인은 점수화할 대규모 라벨 corpus 가 없으므로, 검증은 **구성(by construction)**으로 한다: 모든 출력을 정책 자신의 불변식에 대해 검사하고, 회귀 baseline 이 결정론을 지킨다.

### 6.1 Conformance check

매 실행이 check 스위트의 pass/fail 을 내며, 관련 불변식을 소유한 두 단계에 분산된다:

- **PlacementEngine** — A 화이트리스트 hard-filter, B 타이밍 보존, C 잭 금지(스크래치 레인 제외), D fallback 동작, E candidate 충돌, F 스크래치 제약, G seeded 재현성, K measure cap. Check J(density-rebalance 적법성)는 수동 검사 전용으로 문서화 — hard gate 로 만들려면 post-rebalance 이벤트 집합에 per-measure 잭/충돌 로직을 재실행해야 하므로 범위 밖(v12 §22 DR-J1).
- **BMSWriter** — A placed 완전성, B residual 완전성, C 타이밍 라인 보존(BPM / STOP / scale 라인이 소스와 byte-identical), D no-original-playable-leak.

Check B(타이밍 보존)와 BMSWriter 의 Check C 가 함께 §2.1 의 핵심 불변식을 강제한다: 출력 타이밍은 소스 타이밍, 변경 없음.

### 6.2 강도와 스크래치 스케일

`--intensity`(1–20, 기본 5)와 `--scratch`(1–20, 기본 5)는 §4 의 상수를 lv1 / lv10 / lv20 에 앵커된 piecewise-linear lerp 곡선으로 스케일한다. intensity 는 잭 바닥값, 코드 크기 cap, measure note cap, same-hand streak, stream 비율을 움직이고; scratch 는 measure별 스크래치 예산과 최소 간격을 움직인다. 기본 lv5 가 본 보고서 전반과 baseline 의 보정점이다.

### 6.3 회귀 baseline

`samples/baseline_lv5/` 는 6곡 × {RB, ML} × {bms, json} = 24 파일을 lv5 로 보유한다. 두 자동 스위트가 파이프라인을 지킨다:

- `smoke_test_determinism.py` — 6곡 전부(RB + ML)를 재생성하고 저장 baseline 과 byte-identical 을 단언. `PYTHONHASHSEED` 부류 회귀와 우발적 정책 drift 를 잡는다.
- `smoke_test_resume.py` — 9개 Resume API 케이스: base split, M=0 single, last-measure single, 3단계 cascade, ML+resume 거부, schema-version mismatch, RNG-strategy mismatch, lookahead-requires-resume, lookahead-wiring smoke. 전부 통과.

baseline 은 β-1 + nondet fix 로 재생성됐다; 이 회귀가 "β-1 이 출력을 의도적으로 바꿨다"를 silent drift 가 아니라 일회성 사건으로 만든다.

---

## 7. 사례 연구

### 7.1 mightyA — 47-streak 과 chord-collapse artifact

초기 진단 metric `same_hand_streak` 이 mightyA 에서 경고성 fat tail(47 same-hand 노트 streak)을 보였다. 진짜 hand-balance 실패처럼 보였다. 살펴보니 **측정 artifact** 였다: metric 이 chord-collapse 된 위치를 별개 same-hand 이벤트로 세어, 밀집 코드 벽이 실제 same-hand 런 없이 streak 을 부풀렸다. chord-aware 재metric(`hand_only_streak`)은 source / RB / ML 모두 max 3–8 로 — 시그널 없음. 교훈(v12 §22 DR-K1): 진단의 무서운 숫자는 *채보*보다 *metric* 에 대한 가설이 먼저다.

### 7.2 happiness — BPM-naive LN cap

`hapiness_lnext` 는 인간 LN-과다 채보(757 LN)다. 파이프라인은 0–6 을 냈다. 두 원인이 겹쳤다: 800 ms 선정 게이트(§4.8)가 자연스럽게 짧은 인간 LN 의 ~75% 를 차단, 그리고 이전 draw-length 정책이 BPM-naive — 242 BPM 에서 고정 틱 hold 가 화면을 채우는 막대를 그렸다. draw-length cap `LN_MAX_HOLD_TICKS = 96`(가시 전용)이 막대 문제를 고쳤다(v12 §22 DR-G1); 선정-게이트 문제는 일률 인하 시 모든 채보를 over-LN 하므로 열린 채로 남아있다(§10.3).

### 7.3 단일 measure reroll — Resume API end-to-end

`samples/reroll_demo_2026-05-18/` 가 API 를 시연한다: signal 채보의 measure 58 을 독립 reroll(prefix `[0,57]` → `end_state` → N+1 lookahead 와 함께 resume `[58,58]`)한 뒤 splice 한다. 출력 `.bms` 는 base 와 **measure 58 에서만** 다르다 — 다른 모든 measure 는 byte-identical 로, β-1 의 measure 격리를 확인한다. (이 데모는 한계도 노출했다: m58 의 마지막 코드가 스크래치 레인에 떨어졌는데, 거기엔 KEY-레인 잭 lookahead 가 적용되지 않으므로 lane-swap 시연이 아니라 wiring 시연이다 — §10 참조.)

### 7.4 소스 코드 구성 변형

횡단 발견(v12 §22 DR-K2): 파이프라인이 소스의 코드 구성을 체계적으로 변형한다. mightyA 에서 소스 6% 양손-MIXED 코드가 출력 39% 가 됐고; happiness 에서 67% → 34%. 방향은 곡마다 일관되지 않다. 의심 원인은 pool dedup 이 채널 정보를 버리고 centroid 레인 배정이 코드-메이트를 spread 하여, 소스 단손 코드가 양손에 분산 배치될 수 있다는 것. 가청성("더 분주함")은 그럴듯하나 객관화가 어렵다 — 고정된 버그가 아니라 열린 항목이다.

---

## 8. 더블 플레이 합성 (`--dp`)

싱글 플레이(SP) 채보는 7개 키 레인 + 1개 스크래치를 한 손-쌍으로 연주한다. **더블 플레이(DP)** 채보는 이를 14키 + 2스크래치로 두 배로 늘려, 독립적인 두 side — **P1**(왼손, 키 레인 1–7 + 스크래치)과 **P2**(오른손) — 에 나눈다. 커뮤니티는 DP 채보를 SP 보다 훨씬 적게 만들고, 많은 패키지가 SP 전용으로 배포되므로, 어떤 곡의 DP 렌더링을 원하는 플레이어는 종종 연주할 것이 없다. `--dp` 모드는 **SP 소스로부터** 하나를 합성한다.

DP 는 §2.4 의 소스 충실 입장을 그대로 지킨다. 소스 onset 은 절대 옮기지 않고, SP 의 pool universe·화이트리스트·강도·레인 제약을 변경 없이 재사용한다. DP 가 새로 도입하는 유일한 결정은 **side** — 각 노트를 어느 손이 연주하는가 — 이며, 그 뒤에 기존 side 내 레인 선택(§4.6)이 따른다. 따라서 DP 도 SP 경로와 마찬가지로 난이도 엔진이 아니다: 같은 소스 소리 이벤트를 연주 가능한 레인 공간에 투영한 것이며, 이제 그 공간이 14+2 다.

> **상태.** DP 는 구현되어 있고(RB 전용, 전체 chart) 세 가지 소스 character(stream, peak, chord)에서 청취 검증을 통과했다. 이는 **draft 확장**이지 아직 normative 정책이 아니다(별도 DRAFT addendum 으로 관리). 아래의 구조적 설계 — split router, side-local 배치, 스크래치 정책, 타이밍 / SP-byte-동일성 보장 — 는 안정적이다. §8.5 의 패턴 후처리 상수는 아직 청취 튜닝 중이라 *provisional* 로 표시한다: 값은 움직일 수 있으나, 레이어의 불변식(타이밍 보존, 잭 신설 없음, SP 미접촉)은 고정이다.

### 8.1 인간 DP 채보는 실제로 무엇을 하는가 (corpus 측정)

설계는 **1,852개 인간 DP 채보** 조사(`tools/dp_corpus_survey.py`, 커뮤니티 DP corpus 매칭)에 앵커된다. 세 발견이 정책을 빚었다:

- **양손은 함께 친다; 교대하지 않는다.** 비어있지 않은 beat 의 82.8% 가 *양손(BOTH)* beat 이고, 몇 beat 를 넘는 단일손 런은 사실상 부재다(32+ beat 단일손 런: 0.06%). 한 손을 measure 단위로 쉬게 하는 초기 설계("마디-블록 ALTERNATE")는 이 근거로 폐기됐다 — 인간 DP 는 교대가 아니라 *동시 콘텐츠의 분담*이다. (DR-DP2)
- **같은 소리 ⇒ 같은 손이 1차 규칙성이다.** 토큰의 side 는 그 특정 레인(0.376)보다 훨씬 일관적이다(modal-side share **0.641**). 그래서 DP 는 2단계 affinity 를 진다: 강한 토큰→**side** 기억과, side 안에서 SP 엔진이 재사용하는 약한 토큰→레인 기억.
- **인접 / "무리(impossible)" 스크래치 코드는 극히 드물다.** 스크래치가 같은 side 의 1–3 영역 키와 동시에 울리는 경우(인접)는 1.5%, 4–7 영역(무리)은 1.2% 다. 이 비율은 curator 문화로 깨끗이 갈린다(현대 "stella" 계열 테이블 0.1–1%, 구식 "satellite" 4–5%), 그래서 전면 금지는 임의적 조임이 아니라 현대 관습의 채택이다(DR-DP3).

셋 모두 **난이도 family 에 불변**이었다(family 별 재측정; side-코드, 양손동시, 토큰→side share 가 ★에서 ★★★까지 거의 안 움직임), 그래서 lv5 앵커는 난이도별 보정이 필요 없었다.

### 8.2 Side 배정 — split router

핵심 결정은 *각 노트를 어느 손이 연주하는가* 다. 기본 모드는 **SPLIT** 이다: phrase 블록 안에서 분리 축을 찾아 결과 두 stream 을 두 손에 배정한다.

- **balance**(`--dp-split balance`, **기본**, `auto`→balance)는 음색을 무시하고 **부하**로 분리한다 — 코드를 양손에 분해(centroid 교대)하고 지속 버스트를 가벼운 손으로 교대. **전 corpus 청취로 balance 가 timbre 이상임이 확정됐다(DR-DP13)**: chord·peak 에서 timbre 는 비슷한 음색을 한 손에 쌓아 잭 드롭을 내고, stream 에선 둘이 무차별이다. 그래서 balance 가 보편 기본이다.
- **timbre**(`--dp-split timbre`, opt-in)는 **low-frequency energy ratio** 로 분리한다 — 베이스 / 킥을 한 손에, 밝은 나머지를 다른 손에. 이 축은 경험적으로 실재한다: SP 소스 13곡 × 1,216 measure 에서, 최적 2-way split 의 between-class variance ratio(η²) 중앙값이 0.913 이고 measure 의 99.8% 에서 0.5 를 넘는다(DR-DP4). 다만 "밝은 나머지"를 한 손(R)에 몰아 load 편향을 내므로 stream 곡 한정 opt-in 으로 남긴다. *balance band*(`DP_MIN_SIDE_SHARE = 0.25`)와 measure 단위 50/50 fallback 이 어느 split 이든 한 손 굶김을 막는다.
- **auto** 는 **balance** 로 간다(DR-DP13). balance 가 보편 기본이 되면서 DR-DP7 의 auto-routing — 곡 성격별로 전략을 고르는 문제 — 자체가 소멸했다(character 분류 없이 항상 balance).

phrase **블록**(기존 phase 분할 §4.4, 여기서 첫 실무 용도로 승격)이 입자이고, 토큰→side affinity 가 블록 경계를 넘어 손 배정을 접착하므로 입자가 정밀할 필요는 없다. *mirror-on-repeat* 휴리스틱이 재등장 시 phrase class 의 손을 toggle 해 변주를 준다.

### 8.3 Side-local 배치와 DP 신규 제약

배치는 SP 머신(§4.5)을 **side 별로** 재사용한다: 각 손이 독립적인 잭 / hand / centroid / streak / affinity 상태로 `_place_measure_constrained` 를 돌리고, 오른손 레인은 P1_* → P2_* 로 remap 된다. SP 전용 hand-balance 와 same-hand-alternation 규칙은 비활성화(side 배정이 포섭); 잭·충돌·streak·scale-aware 틱 축은 per-lane 이라 불변이다.

DP 는 두 제약을 더한다:

- **Per-side 코드 cap.** `DP_MAX_CHORD_SIZE_PER_SIDE` 는 DP 전용 lerp 다 — lv5 에서 2, lv20 에서 5 까지 상승(손가락 5개; 4 안정, 5 극단). 합산 양손 cap 은 per-side cap 의 두 배. (인간 side-코드의 94% 가 ≤ 2 라 lv5 = 2 앵커.)
- **스크래치 게이트(hard).** 한 side 에 스크래치가 배치되면, 그 타임스탬프(+ chord-tier 윈도우)의 **same-side 키 레인 7개 전부** 가 가용에서 제거된다 — 인접/무리 구분 없는 전면 차단(DR-DP3). 키는 반대 손으로 가거나 residual 화된다. 스크래치 *side* 는 부하 회피 보정을 가진 교대로 선택된다(DR-DP5): 직전 스크래치의 반대 side 를 선호하되, 교대가 게이트와 충돌하거나 한 손을 과부하시키면 반대 side 로 양보.

### 8.4 콘텐츠 rescue 와 무음 버그

DP 청취가 SP-화이트리스트 상호작용 둘을 드러냈다(Codex 교차검증, DR-DP8):

- **played-content rescue.** SP 화이트리스트의 FX-duration 필터(1000 ms 초과 샘플은 배경 처리, §4.1)가 hardtek 같은 장르의 긴 신스 *리드*를 FX 로 오분류해 키 onset 의 최대 ~79% 를 자른다. duration 필터 제거는 *unsound* 다(그 토큰들의 BGM 발생까지 채보에 쏟아짐). 채택한 수정은 **position-targeted** 다: 제외된 토큰을 *소스가 키 채널(11–19)에서 연주한 위치에 한해* DP 후보로 복원하며, soft pad 를 빼는 attack 게이트를 둔다 — 토큰의 BGM 발생은 건드리지 않으므로 flood 가 없다. bumblebee 소스키 연주율이 62% → 84% 로 올랐다. SP 경로는 미접촉.
- **무음 버그.** DP early-return 경로가 원래 whitelist-*제외* 토큰을 BGM 으로 보내는 SP 단계를 빠뜨려, 그 소리들이 연주도 자동재생도 아닌 — **완전 무음** 이었다(bumblebee 소스키 콘텐츠의 33%). 이것이 들리는 "멜로디 끊김"의 직접 원인이었다. DP 경로에 residual 구성을 추가해 오디오를 복원했고; DP 는 이제 SP 와 같은 방식으로 전체 소스 오디오를 보존한다.

### 8.5 패턴 후처리 — 타이밍 불변 레인 정제

배치 후 레이어(`_dp_postprocess`, per-measure loop 뒤·writer 앞, `if dp:` 가드)가 배치된 노트의 *레인*을 정제한다 — **타이밍은 절대 아니다**. DP 전용이다: SP 경로는 `lane_weights = None` 을 넘기고 레이어의 게이트가 SP 출력을 pre-DP 시절과 **byte-identical** 로 만든다. 매 실행 `dp_pp_report` 진단이 검증하는 레이어의 hard 불변식은: 타이밍 불변, 콘텐츠 보존(`(measure, idx, token)` 멀티셋 보존 — 거부 노트는 BGM 으로, 절대 사라지지 않음), **잭 신설 없음** 이다.

결정적 교훈(DR-DP9)은 독립적인 정제 패스들이 서로 싸우고 잭 보장을 깬다는 것이었다. 첫 버전은 chord / stair / trill 을 레인을 자유 재배정하는 별개 패스로 돌렸고; *잭 113개를 신설*했다(빠른 패시지를 trill 로 오검출해 손이 교대할 수 있는 것보다 빠른 2-레인으로 강제 — "잭보다 빠른 트릴은 트릴이 아니다"). 재설계는 모든 패스를 **공유 side별 lane-tick 타임라인 위 원자적 커밋**으로 돌린다: 패턴은 모든 멤버가 잭 바닥값을 통과할 때만 적용되고, 아니면 런 전체가 원래 jack-safe 레인으로 **폴백**한다. 후처리는 배치-시 hard 제약(잭)을 *재검증하고 양보*할 수 있어도, 절대 위반할 수 없다.

패스 순서:

- **chord** — 코드 레인을 spectral centroid 로 정렬하고, cap 초과 노트를 빈 손의 jack-safe 빈 레인으로 재분배;
- **stair** — 단조 런을 오름/내림 계단으로 정돈;
- **trill** — 교대를 ≥ 2 떨어진 두 레인으로 매핑;
- **금지 모양** — stair 가 그것이 깨야 할 인접쌍을 만들기 때문에 fixpoint loop 와 함께 **맨 뒤**에서 돈다. 어색한 단손 동시치기 모양 `{2,3}` 과 `{5,6,7}`(오른손 미러)을 *contains-match* 로 금지하고, **즈레(zure)**(near-simultaneous 를 뜻하는 IIDX 용어)를 포함한다: `DP_PP_ZURE_TICKS = 4`(48분음 창) 안의 노트는 사실상 한 코드처럼 연주되므로 그렇게 묶는다. 위반 노트는 jack-safe 빈 레인으로 spread 된다.

분포 레벨 제어 둘이 레이어를 마무리하며, 둘 다 *후처리 lever 는 노트 일부만 건드리므로 분포 목표는 배치 단계에서 구동해야 한다*는 교훈을 따른다:

- **밀도 cap** — 합산 양손 per-measure 상한(`DP_PP_COMBINED_MEASURE_CAP = 40`, *provisional*)이 최약-attack 초과분을 BGM 으로 떨어낸다. 이는 *피크* 제한이지 전역 밀도 감소가 아니다(cap 미만 measure 무영향); 전역 밀도는 intensity 의 몫이다(§8.6).
- **rail 레인-가중치** — DP 한 손이 7키 전부를 커버하므로 약지 레인(키 2–3)이 누르기 어렵다. per-lane 출현 가중치(`{2: 0.9, 3: 0.95, 5: 0.9}`, 미러; *provisional*)를 centroid 레인 선택기의 keep-확률로 **배치 단계에 주입**한다 — 후처리 density/stair 패스에서 구동하면 너무 약했다(노이즈만). SP 는 `lane_weights = None` 가드로 미영향.

마지막으로, **cross-measure 스크래치 게이트**(`_dp_pp_scratch_gate`, DR-DP11)가 §8.3 스크래치 금지를 *전역* 틱 축에서 강제한다. per-measure 게이트는 measure 내부 인덱스만 봐서 인접/무리 스크래치가 바 경계를 넘어 새어나갔다(signal 76건 위반 중 64건이 cross-measure); chord 패스 뒤에 돌려 재분배 노트까지 잡는 전역 게이트가 무리 스크래치를 0으로 만든다. 교훈: 경계-spanning 제약은 per-measure 가 아니라 전역 틱 축에서 강제해야 한다.

### 8.6 character 별 intensity

DP 전용 보정 교훈(DR-DP11): DP 에서 **intensity 는 약한 전역-밀도 lever** 다. 소스 onset 을 그대로 깔고 per-side cap 이 binding 이라 lv8 과 lv12 가 거의 같은 밀도를 내고; lv5 라야 median 이 유의미하게 얇아진다. intensity 가 강하게 제어하는 것은 **per-side 코드 capacity**(`DP_MAX_CHORD_SIZE_PER_SIDE`)다. 그래서 따라오는 지침:

- **chord-character 곡 → 높은 intensity**, 곡의 코드가 필요로 하는 양손 동시치기 capacity 를 주기 위해;
- **stream / peak / stair 곡 → 보수적(낮은) intensity** — 이들은 추가 capacity 가 필요 없는 밀집 단음 텍스처이고, 낮은 intensity 가 가독성을 유지한다.

(stream 곡 signal 은 lv5 로 확정.)

### 8.7 소스 인지 스크래치 생성 (DR-S1)

§4.7 의 SP 스크래치 정책은 세 모드(primary 미러 / fallback 합성 / disabled)였고, DP 는 오래 **primary 미러만** 가졌다 — 그래서 `--scratch level > 5` 가 no-op 이었다. 이를 푼 관찰은 둘이다.

첫째, **스크래치는 음향 속성이 아니라 채보(배치) 결정**이다. 같은 `#WAV` 이 곡마다 키로도 휠로도 쓰이므로, 오디오 feature 로 "스크래치 *토큰*"을 분류하려는 시도는 견고하게 실패한다(rank-AUC 0.47–0.62, 우연 수준). 그러나 둘째, 스크래치 **위치는 예측 가능**하다: SP corpus ~600곡(49.6k 스크래치 onset)에서 스크래치는 강박에 집중하며(quarter rate-lift 2.19×, on-quarter rank-AUC 0.69 — 오디오 천장을 넘는다), beat-phase 엔트로피가 키(0.532)보다 낮다(0.314). DP 도 같다. **음색은 작자 주관이어도 자리는 리듬 관습이다.**

따라서 스크래치 생성을 **이동(move)** 으로 푼다 — 발명이 아니라. 생성기는 scratch-*적합* residual(BGM) onset 을 골라 휠로 옮긴다:

- **적합성** = 2a(소스 자신의 휠 토큰, 채널 `16` 출현) ∪ 2b(기능 게이트: 짧은 duration, 충분한 attack, 반복, 키-origin — 배경 FX 아님). 옮겨진 onset 은 *자기 토큰을 그대로* 유지한다(치환 없음); 적합성은 멜로딕 리드를 휠로 끌어오는 것을 막는 느슨한 필터일 뿐이다.
- **밀도**는 소스가 아니라 **인간 DP corpus 에 앵커**된다(active-measure 당 스크래치 p50 = 0.26 / p90 = 1.03; SP 는 0.69 / 1.79 — DP 가 더 희소한 건 양손이 키로 바쁘기 때문). level 이 p50 ↔ p90 을 lerp 한다.
- **배치 관습**: measure 당 단일 스크래치 손(기존 스크래치 measure 는 그 손, 신규는 교대), §8.3 anti-jump 게이트가 새 스크래치에 인접한 same-hand 키를 BGM 으로 강등한다.

이동만 하므로 `(measure, idx, token)` 멀티셋이 보존된다 — §8.5 의 콘텐츠 불변식과 같은 보장, **발명 0**. DP primary supplement 는 이로써 **기본 on**(level > 5 no-op 해소; level ≤ 5 는 미러라 byte-identical). SP 의 위치 재랭킹과 fallback 합성은 opt-in 으로 둔다(SP 기본 출력 불변; fallback 은 corpus 표본 부재로 강제-fallback fixture 로만 e2e 검증).

**정직한 천장.** 위치 선택과 밀도 *목표* 는 소스 독립이나, 후보 *가용성* 과 밀도 *상한* 은 소스에 묶인다(발명 0 의 필연). 적합 residual onset 이 소진되면 생성기는 평탄해진다(mightyA 는 ~62 에서 cap — 소스 천장의 증거; lepontinia 는 공급이 많아 더 간다). 이는 결함이 아니라 §2.4 입장의 직접적 귀결이다: 발명 없이 가능한 최대 위치-독립.

### 8.8 Conformance 와 DP 전용 한계

DP 는 5곡 smoke 스위트(`tools/_dp_smoke.py`)를 돌린다: side-코드 > 2 = 0, 합산 코드 > 3 = 0, 스크래치-게이트 위반 = 0, 충돌 = 0. 매 실행은 `dp_pp_report`(타이밍-불변 플래그, 잭 전/후, 패턴 적용/폴백 카운트, per-lane 분포)도 내고, A/B harness(`tools/dp_pp_report.py`)가 후처리 레이어를 on/off 로 재검증한다. DP 는 resume / finalize / ML 을 거부한다 — 전체 chart, RB 전용이다.

열린 항목:

- **위치 적절성의 무발명 천장 — onset-invention 경계.** §8.7 의 이동 생성은 스크래치를 강박-정렬된 *적합 residual* 위치에 둔다. 그 위치가 곡의 그루브(킥/스네어)와 어긋날 때 — 그루브 자리가 키나 스크래치-부적합 토큰에 점유됐기 때문 — 이동만으로는 닿을 수 없다. 닿으려면 곡의 기존 팔레트 토큰을 *새 시각* 에 두는 **onset-invention**(인간 제작자가 채보 시 하는 것)이 필요하다. 이를 가늠하려 onset-coalition 백본 스코어 — 한 시각의 *모든* 이벤트 증거를 합산(단일 dominant-토큰 검출의 개선판) — 를 측정했다: 합성 점수는 위치 천장(per-chart AUC 0.69)을 넘되(0.77), 위치를 고정한 *조건부* 신호는 얇았고(on-quarter 내 AUC 0.56), 그 신호를 기존 이동 랭킹에 주입해도 배치 품질 측정은 불변이었다 — 이동 후보가 이미 균일하게 강박-정렬이고, 약한-배치 스크래치 대부분이 우리가 옮길 수 없는 *소스 미러*이기 때문. 결론: 무발명 스크래치의 가치는 *더 나은 랭킹* 이 아니라 *도달(reach)* 이며, 그 reach(= onset-invention)는 §2.4 의 소스 충실 입장이 구성상 범위 밖에 두는 *재타이밍* 에 해당한다 — 그래서 의도적으로 frontier 로 남긴다(토큰 SET ⊆ 소스 + 채보 보존 불변식 아래의 미래 작업).
- **6개 character 중 둘이 미커버.** split router 는 stream / peak / chord 를 다룬다; **ln**(롱노트 vs 탭 손-분리)과 **soft** character 는 아직 다루지 않는다.
- **전략 auto-routing 미해결**(DR-DP7); character 는 사용자가 선택해야 한다.
- **normative 승격은** 폭넓은 검증, LN 패스, 더 정밀한 rail-가중치 보정(거부 노트의 spillover 때문에 현재 rail bias 는 정확 비율이 아닌 "느낌")을 기다린다.

---

## 9. 머신러닝 — 학습·통합했으나 동결

ML 경로는 스케치가 아니다. 두 모델 모두 라벨 corpus 에서 end-to-end 학습되고, TorchScript 로 export 되고, `--ml` 뒤 라이브 추론 경로에 wiring 됐다. 여기서 전모를 기록하는 이유는, "ML 을 시도했으나 이기지 못했다"가 커뮤니티에 쓸모 있으려면 *어떻게* — 데이터, 아키텍처, 학습 setup — 가 기록에 남아야 하기 때문이다.

| 모델 | 대체 대상 | 파라미터 | Loss | 상태 |
|---|---|---:|---|---|
| `TokenSelectionModel` | pct 기반 candidate 순서 (§4.3) | ≈ 6.3K | masked BCE | wired(`--ml`), 측정 가능한 이득 없음 |
| `LaneAssignmentModel` | centroid 레인 선택 (§4.6) | ≈ 24.8K | masked CE | wired(`--ml`), 측정 가능한 이득 없음 |

### 9.1 데이터 준비 — 라벨링 파이프라인

학습 데이터는 `data_labeling.py` 가 실제 인간 채보에서 추출한다: 패키지 내 모든 채보의 모든 적격 measure 에 대해, *상황*(measure + pool + context feature)과 *인간의 결정*(어떤 토큰을 어느 레인에 연주했는가)을 짝지은 레코드를 낸다.

- **Pool feature** — 토큰당 14 컬럼, 패키지당 pool table 에 한 번 저장(레코드는 정수 `pool_index` 로 토큰 참조): `duration_ms`, `attack_rms`, `attack_peak`, `intensity_origin`, key / scratch / bgm occurrence 카운트, 6개 STFT spectral feature(centroid 평균/표준편차, flatness, low-freq 비율, zero-crossing 평균/표준편차), whitelist-pass 플래그.
- **Measure feature** — `measure_index`, chart 레벨 `density_rank`, `phase`, `notes_in_measure`.
- **Context window** — 앞선 C = 4 *적격* measure(부적격 measure 는 skip; chart 시작은 oldest-first zero-pad), 각각 `tkey_delta`, `placed_count`, 그리고 배치된 pool-index / 레인 이력. 추론 시 context builder(`placement_engine._build_ml_context`)가 이 순서를 정확히 재현하므로, 학습과 추론이 동일한 텐서 레이아웃을 본다.
- **레이블** — 토큰 레이블은 0/1(연주 여부 → BCE); 레인 레이블은 인간 레인 1..7(→ CE). ground-truth 레인이 제약상 불가용인 레코드는 *skip* 하며, 불법 수를 가르치지 않는다.

파이프라인은 풀스케일(~6,395 패키지)로 실행됐다. 실행 중 **v2 → v3 스키마 재설계**가 강제됐다: v2 는 쓰기 전 모든 레코드를 메모리에 누적했고, 이것이 전체 corpus 에서 OOM 크래시 + ~471 GB 디스크를 채웠다. v3 는 레코드를 JSONL 로 streaming 하며 패키지 레벨 pool table 을 둔다(14-feature 행을 레코드별 payload 에서 들어냄). 그리고 §5.5 대로 토큰 순회는 나중에 학습 결정론을 위해 정렬됐다.

### 9.2 모델 아키텍처

둘 다 고정 추론 인터페이스를 가진 의도적으로 작은 MLP 다; 텐서 컬럼 순서는 모델 I/O 계약에 고정되어 retrain 이 feature 를 silent 하게 옮길 수 없다.

- **TokenSelectionModel** — 가변 크기 pool 에 대한 siamese 스코어러. pool 행마다 `[4 measure ⊕ 14 pool ⊕ 12 flattened context] = 30` 차원을 concat 한 뒤 `LayerNorm(30) → Linear(30,64) → ReLU → Dropout(0.3) → Linear(64,64) → ReLU → Dropout(0.3) → Linear(64,1)`, 토큰당 한 점수로 squeeze. 가변 pool 크기 `P` 는 공통 폭으로 padding 하는 대신 measure 를 행 축으로 concat 하여 처리. ≈ 6,269 파라미터.
- **LaneAssignmentModel** — 7-way 분류기. `[16 event ⊕ 40 flattened context] = 56` 차원을 concat 한 뒤 `LayerNorm(56) → Linear(56,128) → ReLU → Dropout(0.3) → Linear(128,128) → ReLU → Dropout(0.3) → Linear(128,7)`, softmax 전 불가용 레인을 `-inf` 로 마스킹하여 호출자가 바로 argmax. ≈ 24,823 파라미터.

작은 크기는 의도적이다(설계상): 모델은 *RB 를 보조하는 재랭커*이지 지배적 결정자가 아니므로, capacity ceiling 이 모델이 룰 제약을 압도하는 것을 막는다. 둘 다 첫 레이어가 `LayerNorm` 이다 — 입력 feature 가 매우 다른 자연 스케일(`duration_ms` 수백 대 `attack_rms` [0,1])을 가지며, per-sample 정규화가 추론 시 running statistics 유지를 피한다.

### 9.3 학습 setup

- **Optimizer** — Adam, `lr = 1e-3`, `weight_decay = 1e-4`; 각 hidden ReLU 뒤 `Dropout(0.3)`. BatchNorm 없음(가변-`P` siamese 배치 및 작은 배치와 상성 나쁨).
- **스케줄** — 최대 20 epoch + early stopping(`patience = 3`), batch size 256.
- **Class weighting (lane)** — 인간 레인 분포가 불균형하다. 레인 모델은 class-weighted cross-entropy(`--class-weights auto --class-weight-power 2.0`)로 retrain 되어, 드문 레인을 inverse-frequency 의 2제곱으로 up-weight 한다.
- **Split** — 패키지 레벨 결정론적 shuffle(`seed = 42`, `DR-7`): 한 패키지의 measure 가 train/val 경계를 가로지르지 않으므로, 모델이 검증할 채보를 암기할 수 없다.
- **Export** — TorchScript `script`(control flow 보존 위해 `trace` 아님, `DR-4`), 추론 시 `map_location="cpu"` 로 로드(`DR-8`) — 생성에 GPU 불필요.
- **환경** — 기록할 운영 함정: Python 3.13 + GTX 1070 은 CUDA **cu118** PyTorch 빌드 필요(cu121 은 3.13 wheel 없음). class-weighted 레인 retrain(`training/checkpoints`, `lane_cw2` TensorBoard run)이 이 setup 에서 수행됐다.

### 9.4 통합 — 룰 기반 fallback 을 가진 soft 재랭커

두 모델은 *soft 재랭커*로 통합된다: RB 정책이 모든 구조적 결정(어느 segment, 몇 노트, 어느 제약)을 소유하고, 모델은 RB-허용 집합 안에서 순서만 바꾼다. 추론 실패(예외, shape mismatch, 레인 모델 비활성, 가용성 없음) 시 룰 경로로 fallback — 토큰 모델은 pct 순서로, 레인 모델은 centroid / Fisher-Yates 선택으로. 구성상 모델은 순서를 개선할 수 있어도 제약을 *절대* 위반할 수 없다. fill-back ranking hook(density rebalance 가 토큰 모델로 pull-back 순서 결정)이 나중에 별도 진단 카운터와 함께 추가됐다.

### 9.5 판정, 그리고 metric-blindness caveat

v9 baseline 에서 레인 모델의 ~50% top-1 정확도(chance 25% 대비)는 결정적으로 보였다. 그러나 그 baseline 은 *무작위* 레인 배정이었다. RB 경로가 centroid 레인 배정(§4.6)을 채택하자 RB baseline 자체가 학습 가능한 구조 대부분을 포착했고, 2026-05-03 통계 평가는 두 모델 모두 RB 대비 **측정 가능한 우위 없음**을 발견했다. 그 감사의 별도 발견: 레인 모델이 K1/K3/K4 레인 prior 를 학습했다 — context 가 아니라 가장 흔한 레인에 기댔으며, 이것이 정확히 class weighting(§9.3)이 도입되어 상쇄하려던 것이다.

정직한 caveat: 블라인드 A/B 청취에서 ML 출력이 반복적으로 더 안정적/더 인간적으로 *느껴졌으나*, RB-aligned metric 중 그것을 잡은 것이 없었다. 따라서 판정은 "*측정 가능한* 우위 없음" 이며, 명시적 metric-blindness 가능성을 동반한다 — 청취 인상은 fact 이고, 정량화만 미해결이다(§10.2). class-weighted retrain 은 레인 prior 를 개선했으나 판정을 바꾸지 못했고; 그 결과는 동결 결정에 흡수됐다.

### 9.6 교훈 — 개념적 타당성이 경험적 채택을 보장하지 않는다

두 아이디어는 개념적으로 깔끔했다 — 토큰 선호를 아는 토큰 모델, 인간 레인 습관을 배운 레인 모델 — 그리고 룰이 충분히 좋아지자 둘 다 더 싼 룰에 졌다. 정직한 해석(사용자가 기록)은 모델이 *보정된(calibrated)* 채보로부터 배운 적이 없다는 것이다 — 학습 corpus 는 "판정상 좋은 채보"가 아니라 "인간이 만든 무엇이든"이므로, 보정 안 된 corpus 는 보정된 모델을 낼 수 없다; 격차는 모델 capacity 가 아니라 학습 setup 에 있다. ML 은 삭제 대신 플래그 뒤에 동결되어 future re-design 을 위해 계약은 살아있지만, injection point 추가(fill-back, scratch seed, LN candidate)는 직감이 아니라 측정된 이득에 gating 된다. 같은 형태가 character-framework 의 audio-FFT stair 검출에서도 재현된다: *건전한 아이디어는 가설이며, 가설은 파이프라인에 자리를 얻기 전에 경험적 감사를 통과해야 한다.*

---

## 10. 한계와 향후 과제

### 10.1 소스 의존

출력은 소스의 투영이다. 희소하거나 단조로운 소스는 희소하거나 단조로운 채보를 낸다; 파이프라인은 리듬을 발명할 수 없는데 그것이 타이밍 불변식을 깨기 때문이다(§2.4). 이는 결함이 아니라 의도적 경계지만 — 파이프라인이 *작곡가*가 아니라 *렌더러*임을 뜻한다.

### 10.2 청취-proxy 격차

가장 강한 열린 문제: RB-aligned metric 이 ML-vs-RB 청취 차이를 못 잡는다(§9.2). 어떤 metric 이 블라인드 A/B 인상과 상관할 때까지, 모든 "ML 이 더 나쁨" 주장은 metric-한정이다. 계획된 다음 단계는 청취-분해 protocol(짧은 A/B 구간 + 사용자 "안정적" 주석)로 새 metric 설계 전에 가설을 좁히는 것 — 첫 metric 시도(same-hand fat-tail)는 이미 chord-collapse artifact 로 실패했다(§7.1).

### 10.3 LN-스타일 blindness

RB 도 ML 도 소스의 LN *스타일*에 blind 하다: `build_pool_universe` 와 라벨링 파이프라인 둘 다 Long event 를 `(start, token)` 페어로 평탄화하여 hold 길이를 버린다. 그래서 학습 데이터 자체에 LN-스타일 차원이 없고, RB 게이트는 고정 800 ms 다. source-LN-signal 인프라(`(start, end)` hold 틱 보존; 패키지별 LN 통계 산출; 동적 게이트 구동)가 곡별 RB 게이트와 미래 LN-aware 모델 둘의 선행 조건이다.

### 10.4 Resume API v1 범위

v1 은 RB 전용·단일 패스다: ML resume, 부분(영역-국소) conformance, centroid 양방향 lookahead(E-γ)는 모두 범위 밖. 코드-구성 변형(§7.4)과 스크래치-레인 lookahead 격차(§7.3)는 알려졌고 추적 중이다.

---

## 부록 (Appendix)

### A. 하이퍼파라미터 레퍼런스 (intensity = 5 / scratch = 5 기본값)

```text
# FX 분류
FX_DURATION_THRESHOLD          = 1000   # ms
FX_ATTACK_THRESHOLD            = 20     # percentile
FX_ORIGIN_FILTER_ENABLED       = true

# Band 화이트리스트
BAND_QUOTA_RATIO               = 0.20
RARE_OCCURRENCE_THRESHOLD      = 3
WINDOW_SIZE                    = 8      # measures
WINDOW_RESCUE_THRESHOLD        = 0.40
WHITELIST_DURATION_MAX         = 1055   # ms

# Within-idx reorder
USAGE_PENALTY_FIRST            = 10.0   # attack-pct point / 사용 1회
USAGE_WEIGHT_SPREAD            = 1000.0 # Hz ≈ 사용 1회

# Phase
PHASE_MERGE_RATIO_MAX          = 0.289

# 레인 배정 (centroid)
PLACEMENT_RANDOM_SEED          = 42
LANE_STEP_MAX                  = 4.0
CENTROID_EPSILON_RANDOM        = 0.30
CENTROID_STEP_UNIT_FLOOR       = 300    # Hz

# Stream / hand
STREAM_CHORD_RATIO_MAX         = 0.311
STREAM_MAX_SAME_HAND           = 2
MEASURE_NOTE_CAP               = 32

# 잭 (BPM-aware)
MIN_JACK_DELTA_TICKS           = 15
MIN_JACK_DELTA_MS              = 102
MAX_JACK_STREAK                = 2

# 코드
MAX_CHORD_SIZE                 = 3
CHORD_MATE_SPREAD_MIN_GAP      = 2      # 레인 (soft)

# 스크래치
SCRATCH_MIN_INTERVAL           = 16     # ticks
SCRATCH_MAX_PER_MEASURE        = 4
SCRATCH_RUSH_WINDOW            = 3
SCRATCH_RUSH_THRESHOLD         = 3
SCRATCH_RUSH_REST_MEASURES     = 4
SCRATCH_FALLBACK_DURATION_MAX  = 300    # ms

# LN
LN_MIN_DURATION_MS             = 800
LN_MAX_HOLD_TICKS              = 96     # 2-beat 가시 cap

# Density rebalance
DENSITY_REBALANCE_MAX_DELTA    ≈ 0.21
```

전체 표와 lerp 곡선: `placement_engine.py` 의 `compute_intensity_params` 참조.

### B. Conformance check 표

| Check | 단계 | 불변식 |
|---|---|---|
| A 화이트리스트 hard-filter | Placement | 배치 토큰이 FX/unknown hard filter 통과 |
| B 타이밍 보존 | Placement | 모든 배치 노트가 소스 onset 에 위치 |
| C 잭 금지 | Placement | 잭 바닥값 미만 same-lane 반복 없음(스크래치 제외) |
| D fallback 동작 | Placement | primitive-failed measure 가 깨끗이 residual 화 |
| E candidate 충돌 | Placement | `(pos, lane)` / `(pos, token)` 중복 없음 |
| F 스크래치 제약 | Placement | 스크래치 간격 / 밀도 / RUSH 준수 |
| G seeded 재현성 | Placement | 같은 seed → 동일 배치 |
| K measure cap | Placement | `MEASURE_NOTE_CAP` 초과 measure 없음 |
| J density rebalance | Placement | 수동 검사 전용(DR-J1) |
| A placed 완전성 | Writer | 모든 배치 이벤트 렌더 |
| B residual 완전성 | Writer | 모든 residual 이 BGM 으로 렌더 |
| C 타이밍 라인 보존 | Writer | BPM / STOP / scale 라인이 소스와 byte-identical |
| D no-original-playable leak | Writer | 소스 연주 노트가 변형 없이 통과하지 않음 |

### C. CLI 레퍼런스

```text
python run_pipeline.py --folder <패키지> [옵션]

--intensity <1-20>           노트 aggressiveness (기본 5)
--scratch <1-20>             스크래치 빈도, 소스 인지 미러 (기본 5)
--ln                         LN 후처리 활성화
--ml                         ML soft-ranking 활성화 (비권장, §9)
--model-token / --model-lane TorchScript 경로 (--ml 시 필요)
--bms <파일명>               소스 채보 명시 선택
--seed <int|random>          배치 seed (기본 42)

# 더블 플레이 합성 (§8)
--dp                         더블 플레이(14키) 채보 합성
--dp-split <timbre|balance|auto>  side-split 전략 (기본 balance; auto→balance, DR-DP13)

# Resume API (§5)
--resume-state <path>        carry-over 상태 JSON (resume 모드)
--start-measure <M>          resume 구간 시작
--end-measure <N>            resume 구간 끝
--next-chord-lookahead <p>   N+1 첫 코드 경계 입력 (--resume-state 필요)
--finalize <events.json>     splice 된 events 에 post-processing 만
```

### D. 운영 corpus (소스 패키지 세트)

정책이 견뎌야 할 character 를 아우르는 13개 소스 패키지:

| 패키지 | 검증 대상 |
|---|---|
| addiction | 표준 채보 — RB/ML 비교 시발점 |
| mightyA | 밀집 채보 — fill-back streak, chord-collapse artifact (§7.1) |
| blacksphere | short-LN 스타일, MIXED-코드 구성 |
| signal | distraction-style 짧은 키음 (reroll 데모, §7.3) |
| bumblebee | hardtek, 토큰 풍부 (baseline + smoke 앵커) |
| tsuramic | 일반 stream |
| marion | melodic, multi-chart variant 선택 |
| lepontinia | rare-token rescue (§4.2) |
| happiness | 242 BPM + LN-과다 (§7.2) |
| sacrifice | LN + collision |
| nakama | coverage gap (soft) |
| egosa, wanwan | baseline-set 일반 채보 |

회귀 baseline: 이 중 6곡(bumblebee / egosa / lepontinia / signal / tsuramic / wanwan) × {RB, ML} × {bms, json} lv5 (§6.3).

DP 확장(§8)은 별도의 인간 DP 채보 1,852개 조사(§8.1)에 앵커되며, 이 SP 회귀 baseline 이 아니라 5곡 DP smoke 스위트 + 실행별 `dp_pp_report` 진단으로 검증된다.

---

*이 보고서는 2026-05-25 시점의 BMS.Generator 노트 배치 파이프라인을 draft 더블 플레이 합성 addendum(§8)과 함께 기술한다. 정확한 동작의 권위 있는 출처는 소스 코드이며, 이 문서는 독자를 위한 서사적 종합이다.*

