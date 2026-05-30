# BMS.Generator

[🇺🇸 English README](README.md)

BMS 차트의 키음 풀에서 노트 배치를 자동 생성하는 파이프라인.

## 요구 사항

- Python 3.10+
- ffmpeg (PATH에 등록)
- numpy
- scipy (optional, STFT 가속 — 없으면 numpy FFT 폴백)
- torch (optional, `--ml` 모드 활성화 시에만 필요 — TorchScript 모델 로딩)

```bash
pip install -r requirements.txt   # numpy, scipy (core)
# torch 는 ML 모드(--ml)를 쓸 때만, CUDA 빌드에 맞춰 별도 설치 — requirements.txt 주석 참조
```

ffmpeg 은 키음 디코딩에 필수다 (pip 패키지가 아니라 PATH 의 실행 파일):

```bash
winget install ffmpeg      # Windows. 또는 choco install ffmpeg / https://ffmpeg.org/download.html
ffmpeg -version            # PATH 등록 확인 (버전이 찍히면 OK)
```

## 입력 — BMS 패키지

입력은 **BMS 곡 한 개의 폴더**(또는 그 zip)다: `.bms` / `.bme` / `.bml` 차트 파일과, 그 차트가 참조하는 `.wav` 키음들이 함께 든 폴더. BMS 곡을 받으면 보통 이 구조다.

- `--folder <폴더>` 또는 `--zip <zip>` 중 하나로 지정
- 폴더에 차트가 여러 개면 1P 싱글 차트를 coverage 기준으로 자동 선택. `--bms <파일명>` 으로 직접 지정 가능

## 빠른 시작

```bash
python run_pipeline.py --folder "패키지_폴더"
```

또는 zip 아카이브 직접:

```bash
python run_pipeline.py --zip "패키지.zip"
```

`--zip`과 `--folder` 중 하나만 지정. 실행이 끝나면 작업 디렉토리에 `placement_result.bms`가 생성된다 (성공 신호).

> **Windows 주의**: 패키지명/키음 파일명에 한글·일본어가 있으면 (BMS 에서 흔함) 콘솔 기본 인코딩(cp949)이 크래시한다. 실행 전 인코딩을 UTF-8 로:
> ```powershell
> $env:PYTHONIOENCODING="utf-8"      # PowerShell
> ```
> ```cmd
> set PYTHONIOENCODING=utf-8          REM cmd
> ```

### 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--zip <경로>` | - | BMS zip 아카이브 경로 |
| `--folder <경로>` | - | 압축 해제된 BMS 폴더 경로 |
| `--bms <파일명>` | 자동 선택 | 패키지 내 BMS 파일 명시 지정 (파일명만) |
| `--intensity <1~20>` | 5 | 노트 배치 공격성 (1=보수적, 20=공격적) |
| `--scratch <1~20>` | 5 | 스크래치 배율 — primary 모드에서 source per-measure × (level/5) |
| `--ln` | off | LN(롱노트) 후처리 활성화 |
| `--ml` | off | ML 모델 통합 활성화 (아래 ML 모드 참조) |
| `--model-token <경로>` | - | TokenSelectionModel TorchScript (.pt) |
| `--model-lane <경로>` | - | LaneAssignmentModel TorchScript (.pt) |

```bash
# 예: 공격적 배치 + LN 활성화
python run_pipeline.py --zip "패키지.zip" --intensity 8 --scratch 7 --ln

# 예: 특정 BMS 파일 지정 + 보수적 배치
python run_pipeline.py --zip "패키지.zip" --bms "chart_HARD.bms" --intensity 3
```

## 파이프라인 단계

| 단계 | 스크립트 | 입력 | 출력 |
|------|----------|------|------|
| 1 | mix_generation.py | BMS 패키지 (zip/folder) | token_analysis.json, mix_generation_log.json |
| 2 | placement_engine.py | token_analysis.json, 원본 BMS | placement_result.json |
| 3 | bms_writer.py | placement_result.json, 원본 BMS | placement_result.bms |
| 4 | similarity_check.py | placement_result.bms, 패키지 내 BMS | similarity_report.json |

bms_parser.py는 각 스크립트가 내부적으로 import하여 사용한다.

### 1단계: MixGeneration

- BMS 패키지에서 1P 싱글 플레이 차트를 자동 선택 (`#PLAYER 1`)
- `--bms` 지정 시 pre-filter를 건너뛰고 해당 파일을 바로 사용
- 선택 기준: WAV Coverage 최대 > playable 이벤트 수 최대 > 파일명 순
- Pre-filter: `#PLAYER != 1` 제거, duration >= 300초 제거
- 모든 키음 파일을 ffmpeg으로 디코딩하여 토큰별 오디오 통계 산출
- 출력: `token_analysis.json` (토큰별 duration, attack_rms, attack_peak)

### 2단계: PlacementEngine

- 토큰 풀에서 글로벌 playable whitelist 구성
- 4마디 블록 기반 phase segmentation (rush / normal / rest)
- 마디별 primitive 선택: rush -> ChordBurst, 그 외 -> Stream
- 배치 제약: collision, jack prohibition, hand-balance, same-hand alternation
- 스크래치 삽입: 원본 ch16 이벤트 위치에서만 배치 (primary seed / fallback / disabled)
- `--ln` 활성화 시: duration 기반 LN 후처리 (Tap -> LN 승격, LNOBJ 모드)
- 출력: `placement_result.json` (placed events, residual events, diagnostics, ln_meta)

### 3단계: BMSWriter

- 원본 차트에서 header, timing channel (02/03/08/09), BGM (#01) 보존
- 원본 playable channel (11~19, 21~29) 제거 후 placed events로 대체
- LN 활성화 시: `#LNOBJ` 헤더 주입, LN start/end 토큰 출력, 무음 WAV 자동 생성
- residual events를 #01에 추가 (원본 #01과 중복 시 dedup)
- 출력: `placement_result.bms`

### 4단계: SimilarityCheck

- 생성된 차트와 패키지 내 모든 reference 차트의 유사도 비교
- Playable 채널 (measure, idx192, lane) fingerprint 기반
- similarity >= 0.90 시 경고
- 출력: `similarity_report.json`

## Intensity / Scratch 스케일링

`--intensity`와 `--scratch`는 1~20 정수, 기본값 5.

| 파라미터 | intensity=1 | intensity=5 | intensity=10 | intensity=20 |
|----------|-------------|-------------|--------------|--------------|
| WHITELIST_MIN_OCCURRENCE | 15 | 10 | 5 | 1 |
| WHITELIST_MIN_ATTACK_PERCENTILE | 30 | 20 | 10 | 5 |
| WHITELIST_DURATION_MAX | 700 ms | 1056 ms | 1500 ms | 2500 ms |
| STREAM_CHORD_RATIO_MAX | 0.20 | 0.31 | 0.45 | 0.65 |
| STREAM_MAX_SAME_HAND | 1 | 2 | 3 | 4 |
| MAX_CHORD_SIZE | 3 | 3 | 4 | 5 |

**Scratch (primary mode, 2026-05-03 갱신, v12 §12):** `--scratch`는 source의 per-measure 스크래치 밀도에 적용되는 **배율**이다. `scratch=5` (기본) → source 1:1 미러. min interval은 source의 자연 최소값(safety floor 4 ticks)을 사용.

| 파라미터 | scratch=1 | scratch=5 | scratch=10 | scratch=20 |
|----------|-----------|-----------|------------|------------|
| budget_per_measure | 0.2× src | 1.0× src | 2.0× src | 4.0× src |
| min_interval | source min (≥4 ticks) | (좌동) | (좌동) | (좌동) |

소스에 스크래치 토큰이 없으면 fallback 모드로 전환되며, 기존 v9 절대값 lerp 표가 적용된다 (v12 §12.7).

중간값은 선형 보간으로 계산된다.

## LN 모드

`--ln` 플래그를 지정하면 배치 완료 후 LN 후처리가 실행된다.

- duration >= 200ms인 Tap 이벤트를 LN 후보로 수집
- duration 내림차순으로 정렬하여 긴 음부터 승격 시도
- 승격 조건: 같은 레인에 interior/end collision 없음, LN 비율 <= 30%
- LNOBJ 토큰 자동 할당 (미사용 WAV 슬롯)
- BMSWriter가 `#LNOBJ` 헤더 + start/end 토큰 출력

## ML 모드

`--ml`을 지정하면 두 가지 룰 기반 결정이 ML 모델로 대체된다 (soft ranker only — hard constraint는 그대로 적용됨).

- **TokenSelectionModel**: 마디별 후보 토큰 정렬 (RB §6 base order + §6.1 within-idx reorder 위에 ML score 재정렬)
- **LaneAssignmentModel**: 이벤트별 레인 우선순위 (centroid-based RB lane assignment 대체; v9 fisher_yates는 더 이상 RB 기본 아님)

```bash
python placement_engine.py --ml \
  --model-token token_selection_model.pt \
  --model-lane  lane_assignment_model.pt
```

- 모델은 PyTorch TorchScript (`torch.jit.save`) 형식, CPU 로드
- 모델 로딩 실패 또는 inference 실패 시 자동으로 룰 기반 fallback (단위: 마디 / 이벤트)
- `--ml` 없으면 torch가 미설치돼 있어도 동작 (lazy import)
- 학습 데이터 라벨링은 `data_labeling.py` 참조 — 입력 텐서 스키마는 `docs/` 의 기술 보고서에 기술됨

## 데이터 라벨링 (모델 학습용)

`data_labeling.py`는 인간 채보 BMS 패키지에서 TokenSelectionModel/LaneAssignmentModel 학습 데이터셋 (JSONL)을 생성한다.

```bash
python data_labeling.py --dataset <패키지_루트> --output <출력_디렉토리> --config labeling_config.json
```

- 패키지 단위로 발견 → 각 패키지의 `token_analysis.json`이 선행돼 있어야 함
- 캐시: token_analysis + BMS 파일 + config의 SHA-256 해시 기반
- 산출: `token_selection_dataset.jsonl`, `lane_assignment_dataset.jsonl`, `package_pools.json`, `labeling_run_log.json`
- 대규모 (수천 패키지): 디스크 ≥ 100GB 권장, 스트리밍 출력 사용 (이미 적용됨)

## 모델 학습 (ML 모델 만들기, 재현용)

> ML 은 **동결(frozen)** 상태다 — RB 대비 측정 가능한 우위가 없어 운영 비권장. 아래는 재현/재학습용 흐름. 배경과 학습 setup 상세는 `docs/bms-generator-pipeline.ko.md` §8.

전체 흐름: **라벨링 → 학습 → 체크포인트 → `--ml` 사용**. 위 "데이터 라벨링" 산출물(`*_dataset.jsonl` + `package_pools.json` + `labeling_run_log.json`)을 학습 입력으로 쓴다.

```bash
# 1. TokenSelectionModel 학습 (BCE)
python -m training.train_token \
  --dataset <출력>/token_selection_dataset.jsonl \
  --pools   <출력>/package_pools.json \
  --run-log <출력>/labeling_run_log.json \
  --output  training/checkpoints

# 2. LaneAssignmentModel 학습 (class-weighted CE — lane 분포 불균형 보정 권장)
python -m training.train_lane \
  --dataset <출력>/lane_assignment_dataset.jsonl \
  --pools   <출력>/package_pools.json \
  --run-log <출력>/labeling_run_log.json \
  --output  training/checkpoints \
  --class-weights auto --class-weight-power 2.0

# 3. 사용: 위 "ML 모드" 의 --model-token / --model-lane 에 학습된 .pt 지정
python run_pipeline.py --folder "패키지_폴더" --ml \
  --model-token training/checkpoints/token_selection_model.pt \
  --model-lane  training/checkpoints/lane_assignment_model.pt
```

- 모델은 `torch.jit.script` 로 TorchScript export → 추론은 CPU 로드 (생성에 GPU 불필요)
- 학습엔 GPU 권장. torch CUDA 빌드: Python 3.13 + GTX 1070 → **cu118** (`requirements.txt` 주석 참조)
- train_token 기본 50 epoch / train_lane 20 epoch, 둘 다 early stopping + package-level train/val split (seed 42)

## 개별 스크립트 실행

파이프라인 없이 각 단계를 독립 실행할 수도 있다.

```bash
# 1단계만
python mix_generation.py --zip "패키지.zip"
python mix_generation.py --zip "패키지.zip" --bms "chart.bms"

# 2단계만 (token_analysis.json이 이미 있어야 함)
python placement_engine.py
python placement_engine.py --intensity 8 --scratch 3 --ln

# 3단계만 (placement_result.json이 이미 있어야 함)
python bms_writer.py

# 유사도 검사만
python similarity_check.py --zip "패키지.zip"
```

## 출력 파일 설명

| 파일 | 설명 |
|------|------|
| token_analysis.json | 토큰별 오디오 분석 결과 (duration, attack RMS/peak, decode 성공 여부) |
| mix_generation_log.json | 차트 선택 로그, 커버리지, 토큰 진단 정보 |
| placement_result.json | 배치 결과 (placed/residual 이벤트 + diagnostics + ln_meta) |
| placement_result.bms | 최종 BMS 차트 파일 |
| similarity_report.json | 패키지 내 reference 차트 대비 유사도 보고 |
| lnobj_silent.wav | LN 모드 시 자동 생성되는 무음 WAV (44바이트) |

## 설계 문서

파이프라인 전체 설계 — BMS 파싱·채널 시맨틱; 차트/토큰 선택 및 스펙트럼 분석;
배치 정책 (whitelist, phrasing, primitive, constraint, scratch, LN, Resume API);
ML 통합·동결; 데이터 라벨링; 모델 아키텍처; 출력/헤더 정책; 유사도 검사; 노트
속성 정의 — 은 `docs/` 의 기술 보고서에 기술되어 있다.

### 종합 보고서 (`docs/`)

- `docs/bms-generator-pipeline.ko.md` — 파이프라인 전체 기술 보고서 (설계 / Resume API / 검증 / ML 학습·동결 / 한계). 영문: `docs/bms-generator-pipeline.en.md`.

### 테스트 (`tests/`)

- `tests/smoke_test_resume.py` — Resume API 9-case (base split / M=0 / 마지막 measure / cascade / ML 차단 / schema·rng mismatch / lookahead)
- `tests/smoke_test_determinism.py` — 6곡 × {RB, ML} fresh run 이 `samples/baseline_lv5/` 와 byte-identical 인지 회귀
