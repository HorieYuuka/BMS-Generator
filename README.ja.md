# BMS.Generator

[🇬🇧 English README](README.md) · [🇰🇷 한국어 README](README.ko.md)

楽曲のキーサウンドプールから BMS 譜面のノート配置を自動生成するパイプライン。

## 必要環境

- Python 3.10+
- ffmpeg (PATH 上)
- numpy
- scipy (任意、STFT 高速化 — 無ければ numpy FFT にフォールバック)
- torch (任意、`--ml` モードでのみ必要 — TorchScript モデルのロード)

```bash
pip install -r requirements.txt   # numpy, scipy (コア)
# torch は ML モード (--ml) でのみ必要。CUDA ビルドに合わせて別途インストール — requirements.txt のコメント参照
```

ffmpeg はキーサウンドのデコードに必須 (pip パッケージではなく PATH 上の実行ファイル):

```bash
winget install ffmpeg      # Windows。または choco install ffmpeg / https://ffmpeg.org/download.html
ffmpeg -version            # PATH 登録の確認 (バージョンが表示されれば OK)
```

## 入力 — BMS パッケージ

入力は **1 曲分の BMS フォルダ** (またはその zip)。`.bms` / `.bme` / `.bml` 譜面ファイルと、それらが参照する `.wav` キーサウンドのセット。BMS 楽曲のダウンロードは通常すでにこの形になっている。

- `--folder <dir>` または `--zip <zip>` のどちらかを渡す
- フォルダに複数譜面がある場合、coverage で 1P シングル譜面が自動選択される。`--bms <filename>` で上書き可能

## クイックスタート

```bash
python run_pipeline.py --folder "package_folder"
```

または zip アーカイブから直接:

```bash
python run_pipeline.py --zip "package.zip"
```

`--zip` / `--folder` はちょうど一方だけ指定する。成功すると作業ディレクトリに `placement_result.bms` (完成した譜面) が出力される。

> **Windows 注意**: BMS パッケージやキーサウンドのファイル名には韓国語 / 日本語がよく含まれ、デフォルトのコンソールエンコーディング (cp949) ではクラッシュする。実行前に UTF-8 へ切り替えること:
> ```powershell
> $env:PYTHONIOENCODING="utf-8"      # PowerShell
> ```
> ```cmd
> set PYTHONIOENCODING=utf-8          REM cmd
> ```

### オプション

| オプション | デフォルト | 説明 |
|------|--------|------|
| `--zip <path>` | - | BMS zip アーカイブのパス |
| `--folder <path>` | - | 展開済み BMS フォルダのパス |
| `--bms <filename>` | 自動選択 | パッケージ内の BMS ファイルを明示指定 (ファイル名のみ) |
| `--intensity <1-20>` | 5 | 配置の積極度 (1 = 保守的、20 = 積極的) |
| `--scratch <1-20>` | 5 | スクラッチ倍率 — primary モードでは source の measure 単位 × (level / 5) |
| `--ln` | off | LN (ロングノート) 後処理を有効化 |
| `--ml` | off | ML モデル統合を有効化 (後述「ML モード」) |
| `--model-token <path>` | - | TokenSelectionModel TorchScript (.pt) |
| `--model-lane <path>` | - | LaneAssignmentModel TorchScript (.pt) |

```bash
# 例: 積極的な配置 + LN 有効
python run_pipeline.py --zip "package.zip" --intensity 8 --scratch 7 --ln

# 例: 譜面ファイルを明示 + 保守的な配置
python run_pipeline.py --zip "package.zip" --bms "chart_HARD.bms" --intensity 3
```

## パイプラインの各ステージ

| ステージ | スクリプト | 入力 | 出力 |
|------|----------|------|------|
| 1 | mix_generation.py | BMS パッケージ (zip/folder) | token_analysis.json, mix_generation_log.json |
| 2 | placement_engine.py | token_analysis.json, source BMS | placement_result.json |
| 3 | bms_writer.py | placement_result.json, source BMS | placement_result.bms |
| 4 | similarity_check.py | placement_result.bms, パッケージの BMS ファイル | similarity_report.json |

`bms_parser.py` は各ステージから内部的に import される。

### ステージ 1 — MixGeneration

- パッケージから 1P シングルプレイ譜面 (`#PLAYER 1`) を自動選択
- `--bms` はプレフィルタをスキップし、指定ファイルを直接使用
- 選択優先度: WAV coverage 最大 > 演奏可能イベント数最大 > ファイル名順
- プレフィルタは `#PLAYER != 1` と 300 秒以上の譜面を除外
- 全キーサウンドを ffmpeg でデコードし、トークン単位の音響統計を計算
- 出力: `token_analysis.json` (トークン単位の duration, attack_rms, attack_peak)

### ステージ 2 — PlacementEngine

- トークンプールから全体の演奏可能 whitelist を構築
- 4 measure ブロックへの phase 分割 (rush / normal / rest)
- measure 単位の primitive 選択: rush → ChordBurst、それ以外 → Stream
- 配置制約: 衝突、jack (縦連) 禁止、ハンドバランス、同手交互
- スクラッチ挿入: source の ch16 イベント位置にのみ配置 (primary seed / fallback / disabled)
- `--ln` 指定時、duration ベースの LN 後処理が Tap を LN に昇格 (LNOBJ モード)
- 出力: `placement_result.json` (配置イベント、residual イベント、診断、ln_meta)

### ステージ 3 — BMSWriter

- source のヘッダー、タイミングチャンネル (02/03/08/09)、BGM (`#01`) を保存
- source の演奏チャンネル (11–19, 21–29) を除去し、配置イベントで置き換え
- LN 使用時: `#LNOBJ` ヘッダーを注入、LN の開始/終了トークンを出力、無音 WAV を自動生成
- residual イベントを `#01` に追記 (source 既存の `#01` と重複排除)
- 出力: `placement_result.bms`

### ステージ 4 — SimilarityCheck

- 生成譜面をパッケージ内の各リファレンス譜面と比較
- 演奏チャンネル `(measure, idx192, lane)` 上のフィンガープリント
- 類似度 ≥ 0.90 で警告
- 出力: `similarity_report.json`

## Intensity / Scratch スケーリング

`--intensity` と `--scratch` は 1–20 の整数 (デフォルト 5)。

| パラメータ | intensity=1 | intensity=5 | intensity=10 | intensity=20 |
|----------|-------------|-------------|--------------|--------------|
| WHITELIST_MIN_OCCURRENCE | 15 | 10 | 5 | 1 |
| WHITELIST_MIN_ATTACK_PERCENTILE | 30 | 20 | 10 | 5 |
| WHITELIST_DURATION_MAX | 700 ms | 1056 ms | 1500 ms | 2500 ms |
| STREAM_CHORD_RATIO_MAX | 0.20 | 0.31 | 0.45 | 0.65 |
| STREAM_MAX_SAME_HAND | 1 | 2 | 3 | 4 |
| MAX_CHORD_SIZE | 3 | 3 | 4 | 5 |

**スクラッチ (primary モード、2026-05-03 更新、v12 §12):** `--scratch` は source の measure 単位スクラッチ密度に適用される **倍率**。`scratch=5` (デフォルト) → 1:1 の source ミラー。最小間隔は source 自身の自然な最小値を使用 (安全下限 4 ticks)。

| パラメータ | scratch=1 | scratch=5 | scratch=10 | scratch=20 |
|----------|-----------|-----------|------------|------------|
| budget_per_measure | 0.2× src | 1.0× src | 2.0× src | 4.0× src |
| min_interval | source min (≥ 4 ticks) | (同左) | (同左) | (同左) |

source にスクラッチトークンが無い場合、パイプラインは旧 v9 の絶対 lerp テーブルを用いた合成モードにフォールバックする (v12 §12.7)。

中間値は線形補間される。

## LN モード

`--ln` は配置後に LN 後処理を実行する:

- トークン duration ≥ 200 ms の Tap イベントを LN 候補として収集
- duration 降順にソートし、最も長いものから昇格を試みる
- 昇格条件: 同レーン上で内部/終端の衝突が無く、LN 比率 ≤ 30%
- LNOBJ トークン (未使用の WAV スロット) を自動割り当て
- BMSWriter が `#LNOBJ` ヘッダーと開始/終了トークンを出力

## ML モード

`--ml` はルールベースの 2 つの判断を ML モデルに差し替える (ソフトな再ランカーのみ — ハード制約は依然適用):

- **TokenSelectionModel** — measure 単位の候補トークン順序付け (RB §6 のベース順序 + §6.1 の within-idx reorder の上で再ランク)
- **LaneAssignmentModel** — イベント単位のレーン優先度 (centroid ベースの RB レーン割り当てを置き換え。v9 の fisher_yates はもはや RB のデフォルトではない)

```bash
python placement_engine.py --ml \
  --model-token token_selection_model.pt \
  --model-lane  lane_assignment_model.pt
```

- モデルは PyTorch TorchScript (`torch.jit.save`)、CPU 上でロード
- モデルのロード失敗や推論失敗時は、自動的にルール経路へフォールバック (粒度: measure 単位 / イベント単位)
- `--ml` 無しなら torch 未インストールでもパイプラインは動作する (遅延 import)
- 学習データのラベリングは `data_labeling.py` を参照。入力テンソルスキーマは `docs/` の技術レポートに記述されている

## データラベリング (モデル学習用)

`data_labeling.py` は、人間が作成した BMS 譜面のパッケージから TokenSelectionModel / LaneAssignmentModel の学習データセット (JSONL) を生成する。

```bash
python data_labeling.py --dataset <package_root> --output <output_dir> --config labeling_config.json
```

- パッケージ単位で探索 → 各パッケージの `token_analysis.json` が事前に存在している必要がある
- キャッシュ: token_analysis + BMS ファイル + config の SHA-256
- 出力: `token_selection_dataset.jsonl`, `lane_assignment_dataset.jsonl`, `package_pools.json`, `labeling_run_log.json`
- 大規模時 (数千パッケージ): ディスク ≥ 100 GB を見込むこと。出力はすでにストリーミング方式

## モデル学習 (ML モデルの作り方、再現用)

> ML は **凍結** されている — RB に対して測定可能な優位は無く、運用は非推奨。以下のフローは再現 / 再学習用。背景と学習セットアップの詳細は `docs/bms-generator-pipeline.ja.md` セクション 8 を参照。

エンドツーエンドの流れ: **ラベリング → 学習 → checkpoint → `--ml`**。学習入力は上記「データラベリング」セクションの出力 (`*_dataset.jsonl` + `package_pools.json` + `labeling_run_log.json`)。

```bash
# 1. TokenSelectionModel (BCE)
python -m training.train_token \
  --dataset <out>/token_selection_dataset.jsonl \
  --pools   <out>/package_pools.json \
  --run-log <out>/labeling_run_log.json \
  --output  training/checkpoints

# 2. LaneAssignmentModel (class-weighted CE — レーン不均衡の補正に推奨)
python -m training.train_lane \
  --dataset <out>/lane_assignment_dataset.jsonl \
  --pools   <out>/package_pools.json \
  --run-log <out>/labeling_run_log.json \
  --output  training/checkpoints \
  --class-weights auto --class-weight-power 2.0

# 3. 使用: --model-token / --model-lane (上記「ML モード」) を学習済み .pt ファイルに向ける
python run_pipeline.py --folder "package_folder" --ml \
  --model-token training/checkpoints/token_selection_model.pt \
  --model-lane  training/checkpoints/lane_assignment_model.pt
```

- モデルは `torch.jit.script` で TorchScript としてエクスポートされ、推論時は CPU でロード (生成に GPU 不要)
- 学習は GPU があると有利。CUDA ビルド: Python 3.13 + GTX 1070 → **cu118** (`requirements.txt` のコメント参照)
- `train_token` はデフォルト 50 epoch、`train_lane` は 20 epoch。両方とも early stopping + パッケージ単位の train/val split (`seed=42`) を使用

## 各ステージの個別実行

パイプラインドライバを使わず、各ステージを単独で実行することもできる:

```bash
# ステージ 1 のみ
python mix_generation.py --zip "package.zip"
python mix_generation.py --zip "package.zip" --bms "chart.bms"

# ステージ 2 のみ (token_analysis.json が事前に存在している必要がある)
python placement_engine.py
python placement_engine.py --intensity 8 --scratch 3 --ln

# ステージ 3 のみ (placement_result.json が事前に存在している必要がある)
python bms_writer.py

# 類似度チェックのみ
python similarity_check.py --zip "package.zip"
```

## 出力ファイル

| ファイル | 説明 |
|------|------|
| token_analysis.json | トークン単位の音響分析 (duration, attack RMS/peak, デコード成否) |
| mix_generation_log.json | 譜面選択ログ、coverage、トークン診断 |
| placement_result.json | 配置結果 (配置 + residual イベント + 診断 + ln_meta) |
| placement_result.bms | 最終的な BMS 譜面ファイル |
| similarity_report.json | パッケージ内の各リファレンス譜面に対する類似度レポート |
| lnobj_silent.wav | LN モード時に自動生成される無音 WAV (44 バイト) |

## 設計ドキュメント

パイプライン全体の設計 — BMS パース・チャンネルセマンティクス; 譜面/トークン選択
およびスペクトル分析; 配置ポリシー (whitelist, phrasing, primitive, constraint, scratch,
LN, Resume API); ML 統合・凍結; データラベリング; モデルアーキテクチャ; 出力/ヘッダー
ポリシー; 類似度チェック; ノート属性定義 — は `docs/` の技術レポートに記述されている。

### 技術レポート (`docs/`)

- `docs/bms-generator-pipeline.ja.md` — パイプライン全体の技術レポート (設計 / Resume API / 検証 / ML 学習・凍結 / 限界)。英語: `docs/bms-generator-pipeline.en.md`、韓国語: `docs/bms-generator-pipeline.ko.md`。

### テスト (`tests/`)

- `tests/smoke_test_resume.py` — Resume API の 9 ケース (base split / M=0 / last-measure / cascade / ML rejection / schema·rng mismatch / lookahead)
- `tests/smoke_test_determinism.py` — 回帰: 6 曲 × {RB, ML} の新規実行が `samples/baseline_lv5/` とバイト単位で同一
