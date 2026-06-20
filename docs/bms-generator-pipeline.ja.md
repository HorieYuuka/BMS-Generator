# BMS.Generator — ソース忠実な譜面生成パイプライン: band(帯域)ホワイトリスト、スペクトル重心(セントロイド)によるレーン割り当て、そして決定論的な Resume API

**サブタイトル**: ソース楽曲のキー音プールから演奏可能な BMS 譜面を再構築し、ソースのタイミングを保持しつつ、部分的な再生成(reroll)を支えられるだけの決定論性を備えたルールベースパイプラインの設計と検証。

**バージョン**: v12 (2026-05-25)、ダブルプレイ合成のドラフト addendum を含む (§8; 2026-06-14)

> 本稿は BMS コミュニティを対象とした技術レポートである。BMS 譜面フォーマットとリズムゲームの作譜に関する基本的な理解を前提とするが、導入するすべての用語・ポリシー・定数は本文中で説明する。

---

## Abstract（要旨）

自動作譜の研究の大半は、このタスクを *難易度合成* — すなわち目標とする難易度の譜面を生成すること — として捉える。BMS.Generator はこれと正反対の立場を取り、作譜を *ソース忠実な再構築* として扱う。ソース BMS ファイルが与えられると、本パイプラインはその楽曲自身のキー音アセット(`#WAV` トークンプール)を再利用し、要求された intensity(強度) において演奏可能なノート配置を **タイミングを変更することなく** 再導出する。生成物が原曲のように聞こえるのは、元の音を元の onset(発音点) にそのまま積み上げて作っているからである。

本パイプラインはルールベース(RB)である。機械学習(ML)の経路(トークン選択モデル + レーン割り当てモデル)も学習・統合し、`--ml` フラグの背後に置いたが、統計的評価においてルールベースに対する測定可能な優位が認められなかったため、運用上は凍結(freeze)されている(§9)。RB 設計の主たる貢献は以下のとおりである。

1. **band(帯域)ベースの quota ホワイトリスト** + 希少トークンの rescue — 単一の global hard filter ではなく、スペクトル band 相対の occurrence quota によってどのプールトークンが演奏可能かを選択し、旋律的でありながら出現が疎なトークンを保護する。
2. **スペクトル重心(セントロイド)ベースの相対レーン割り当て** — 各ノートを、音楽のスペクトル重心(セントロイド)の軌跡(明るくなれば右へ、暗くなれば左へ)に saturating(飽和)曲線で追従させながらキーレーンへマッピングする。無作為な割り当てではない。
3. **ソースを認識する(source-aware)スクラッチポリシー** — 絶対密度テーブルからスクラッチを合成する代わりに、ソース自身の measure ごとのスクラッチ密度にレベル倍率を掛けてミラーリングする。
4. **決定論的な Resume API** — measure 単位の RNG 隔離(β-1)により、単一の measure(あるいは measure 区間)を独立して再現可能にし、外部エディタが残りの部分に手を触れずに譜面の一部を reroll できるようにする。

ドラフト拡張(§8)は、この仕組みを再利用して SP ソースから **ダブルプレイ(14キー)** 譜面を合成する。同じタイミング不変条件を守り、SP 出力を byte-identical(バイト単位で同一) に保ったまま、per-note の *side(手)* 決定のみを追加する。

検証は corpus 統計ではなく構成(by construction)によって行う。すなわち、17項目の適合性チェック(conformance check)スイート(§6.1)、6曲 × {RB, ML} の byte-identical(バイト単位で同一) な回帰 baseline、9 ケースの Resume API スモークスイートである。運用 corpus は、コード密集・LN 過多・スクラッチ過多・BPM トリックの譜面を網羅した 13 個のソースパッケージから成る。

---

## 1. はじめに

### 1.1 問題設定

BMS 楽曲は、オーディオサンプル群(`#WAV` スロット)と、それらのサンプルをレーン × タイムラインに配置した一つ以上の譜面として配布される。楽曲のサンプルと *一つ* の参照譜面だけが与えられたとき、次を満たす **新たな演奏可能譜面** を合成できるだろうか。

- 楽曲自身の音を使い(だから依然としてその楽曲のように聞こえ)、
- 楽曲自身の onset にノートを乗せ(だから音楽的に誠実であり)、
- 調整可能な intensity(強度) で演奏可能であり(ノート密度 / スクラッチ頻度 / ロングノート使用量)、
- **再現可能** であるか — 同じ入力は常に同じ譜面を生み、単一の measure を独立して再生成できるか。

素朴なアプローチ —「すべての音イベントをノートにする」— は演奏不可能な壁を作り出す。その逆 —「目標難易度に合わせてノートを合成する」— は楽曲のアイデンティティを捨ててしまう。設計上の課題はその中間にある。すなわち、どの音イベントを演奏ノートとして *選択* し、自然に読み取れるレーンへ *割り当て* 、要求された intensity(強度) へ密度を *整形* する。それでいてノートをソース onset から決して動かさない、ということである。

### 1.2 目標

**適切な譜面(adequate chart)** を作る — 最大限に難しい譜面でも、最大限に巧妙な譜面でもなく。選んだ intensity(強度) において、その楽曲を忠実かつ演奏可能にレンダリングしたものだと人間が受け入れる譜面である。具体的には、各実行ごとに次を出力する。

- `placement_result.bms` — BMS フォーマットの演奏譜面、
- `placement_result.json` — 配置済み / residual(未配置の残余=BGM)イベント集合と診断情報、
- パッケージの既存譜面に対する類似度レポート。

「適切さ」というフレーミング(v12 §21)は意図的なものである。ルールベースのポリシーは、何らかの美的価値を *最大化* することよりも、*明確に定義可能な悪いパターン*(床値より速い same-lane の縦連(jack)、過大なコード、hand-balance の崩壊)を *遮断* することを目標とする。美的価値はソースとユーザーに委ねられているからである。

### 1.3 貢献

- **C1 — band quota ホワイトリスト + rare rescue。** トークン選択は、global hard filter(v9)→ スペクトル band 相対の occurrence quota(v10)→ quota 拡大(0.20)+ occurrence ≤ 3 の非 FX トークンに対する明示的な rescue(v11)へと進化した。これにより、一律の occurrence カットでは落ちてしまう旋律的なハイライトを残す。(§4.2)
- **C2 — スペクトル重心(セントロイド)相対レーン割り当て。** レーンは、音楽のスペクトル重心(セントロイド)の動きを、曲ごとに自動較正された saturating(飽和)曲線で追従しながら選択し、ε-greedy による多様化を加える。v9 の無作為なレーン選択を置き換えるものである。(§4.6)
- **C3 — ソースを認識するスクラッチミラー。** primary モードでは、出力はソースの measure ごとのスクラッチ数 × (level / 5) をミラーリングし、ソース自身の最小間隔を spacing の床値として用いる。レベルインデックスの絶対テーブルではない。(§4.7)
- **C4 — 決定論的な Resume API。** measure 単位の RNG(β-1: `Random(seed × 10⁶ + measure)`)が譜面全体の sequential RNG の結合を取り除き、measure 区間 `[M, N]` を直列化された carry-over 状態から再生成できるようにする。N+1 境界の lookahead はオプションである。これは外部エディタが単一の measure を reroll するための基盤(substrate)である。(§5)

- **C5 — ダブルプレイ合成(ドラフト)。** `--dp` モードは、同じソース忠実な立場のもとで SP ソースから 14 キーの DP 譜面を合成する。各 phrase を負荷(balance、デフォルト、§8.2)によって両手へ分割し、タイミング不変かつ縦連(jack)を認識するレーン後処理器で結果を精製する。onset を動かさず、SP 出力を変更することもない。(§8)

第五の、横断的な性質 — **決定論性(determinism)** — も作業を通じて堅牢化された。すなわち、`PYTHONHASHSEED` に依存する set/dict の反復順序に起因する非決定性の一クラスを発見・除去し(§5.5)、6 曲の回帰 baseline が byte-identical(バイト単位で同一) な出力を守るようにした。

---

## 2. 背景

### 2.1 BMS フォーマットとトークンプール

BMS 譜面は、measure ごとにキー付けされたチャンネル行から成るテキストファイルである。各行 `#mmmCC:...` は、measure `mmm`、チャンネル `CC` に、2 文字の base-36 **トークン**を均等分割された sub-position へ配置する。本パイプラインは位置を **idx192** — measure あたり 192 ティック(一般的な音符分割の LCM)— に正規化するため、グローバルなタイムスタンプは `tkey = measure × 192 + idx192` となる。

本文を通して用いる主要な用語は次の通りである。

- **token** — `#WAV` スロット id。1 つのオーディオサンプル。
- **event** — `(measure, idx192)` における 1 つのトークン発生。
- **pool universe** — レビュー対象となる、使用済みの全トークンイベント。
- **playable whitelist** — 演奏ノートになることを許可されたトークンの部分集合。
- **residual**(未配置の残余=BGM)— 演奏ノートではなく BGM/自動再生オブジェクト(チャンネル `01`)として残されるトークンイベント。residual 化は、キー入力を要求せずに音をその onset(発音点)に保存する。
- **spectral centroid** — トークンサンプルの周波数領域における重心。レーン割り当てを駆動する明るさの proxy。

決定的な不変条件は、**タイミングは決して変更されない**ことである。ノートは自身のソース onset で演奏可能になるか、あるいは同じ onset で residual 化されるかのいずれかである。本パイプラインが決めるのは*演奏可能か否か*と*どのレーンか*だけであり、*いつ*を決めることは決してない。

### 2.2 4 段階パイプライン

```
mix_generation.py  →  placement_engine.py  →  bms_writer.py  →  similarity_check.py
   (analyze)             (place)                (emit)            (verify)
```

| 段階 | 入力 | 出力 | 役割 |
|---|---|---|---|
| **MixGeneration** | ソースパッケージ(`.bms` + `.wav` 群) | `token_analysis.json`, `mix_generation_log.json` | ソース譜面の選択、各キー音のデコード、トークンごとの spectral / attack 特徴量の計算(キャッシュ) |
| **PlacementEngine** | `token_analysis.json` + ソースイベント | `placement_result.json` | ポリシーの中核 — ホワイトリスト、phase、measure ごとの配置、スクラッチ、LN、密度の rebalance |
| **BMSWriter** | `placement_result.json` + ソース `.bms` | `placement_result.bms` | 配置ノート + residual BGM を有効な BMS ファイルへレンダリングし、ソースのタイミング行を保存 |
| **SimilarityCheck** | 出力 `.bms` + パッケージ譜面 | `similarity_report.json` | パッケージ既存譜面との重複をレポート(診断用、非 gating) |

`run_pipeline.py` がこの 4 段階を連結する。運用モードは **RB 単独**であり、`--ml` フラグは存在するが非推奨である(§9)。

### 2.3 ルールベース vs 機械学習 — なぜ RB は凍結された上で有効なのか

2 つのモデルを学習させた。`TokenSelectionModel`(どのトークンを演奏するか)と `LaneAssignmentModel`(どのレーンか)である。v9 baseline において、レーンモデルは chance の 25% に対し約 50% の top-1 精度に到達し、見かけ上は大きな利得であった。しかしその利得は、v9 が用いていた*ランダム*レーン baseline に対して測定されたものである。ルールベース経路が centroid レーン割り当て(C2)を採用すると、RB baseline 自体がモデルの学習した構造の大部分を捉えるようになり、2026-05-03 の統計評価では、ML モデルは RB 経路に対して**測定可能な優位を持たない**ことが判明した(v12 §19.5)。

1 つの caveat を正直に記録しておく。ブラインド A/B 試聴では、ML 出力が時に「より安定している/より人間らしい」と感じられたが、RB-aligned な指標のいずれもその印象を捉えられなかった。したがって判定は「優位なし」ではなく「*測定可能な*優位なし」であり、metric-blindness の可能性がある(§9.2、§10.2)。ML は削除されているのではなく、フラグの背後に凍結されている。

### 2.4 フレームワークの立場：ソース依存は隠さず認める

このパイプラインは難易度エンジン**ではない**。リズムを発明せず、onset を動かさず、目標 θ を狙わない。その同一性はソースに由来する。すなわち演奏配置とは、ソース自身の音イベントを演奏可能なレーン空間へ*投影*したものである。この立場はスコープに直接の帰結をもたらす — *再タイミング*を要する一切(新たなリズム figure の合成や、疎なセクションの「改善」)は、構成上スコープ外である(v12 §2 FORBIDDEN)。Resume API(§5)も同じ境界を守る。measure を再ロールしても変わるのは*どのレーン / どのトークン*であって、*いつ*は決して変わらない。

---

## 3. アーキテクチャ

### 3.1 パイプライン各段階の詳細

**MixGeneration** はパッケージから候補となる `.bms` ファイルを走査し、coverage と演奏ノート数によって 1 つを選択(あるいは明示的な `--bms` を尊重)した後、宣言された全キー音をデコードする。トークンごとに attack 強度(RMS / peak)、duration、STFT spectral 特徴量(centroid、flatness、low-frequency ratio、zero-crossing rate)を計算する。結果は `token_analysis.json` にキャッシュされ、再実行時はデコードをスキップする。

**PlacementEngine** はポリシーの中核であり、§4–§6 の主題である。pool universe を構築し、band(帯域)ホワイトリストを組み立て、譜面を phase に分割した後、hard constraint の下で measure を左から右へ配置し、スクラッチを挿入し、ロングノートを後処理し、最後に譜面全体の密度を rebalance する。

**BMSWriter** は結果を BMS へレンダリングする。配置ノートは key/scratch チャンネルへ、residual トークンは BGM チャンネル `01` へ送られ、ソースのタイミング行(BPM 変更、STOP、measure-length scale)はそのまま保存される。writer 側の適合性チェック(A–D、§6.1)が、完全性と、元の演奏ノートが改変されずに漏れ出ていないことを検証する。

**SimilarityCheck** は出力をパッケージ既存譜面と比較し、重複をレポートする。診断専用であり、生成を gating することは決してない。

### 3.2 データフロー

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

`placement_result.json` はポリシーとレンダラの間の契約である。これは加算的(additive)にバージョン管理される。Resume API(§5)は resume モードにおいて `mode` / `end_state` フィールドを追加するが、default 経路で BMSWriter が消費するフィールドは変更しないため、既存の消費者は影響を受けない。

---

## 4. 配置ポリシー設計

PlacementEngine は、すべてのプール イベントについて、それが演奏ノートになるか、どのレーンに置かれるかを決定し、その結果として得られる密度を要求された intensity(強度)へと整形する。ポリシーは measure を左→右へ走査する単一パス(per-measure loop)であり、その前段に pool/ホワイトリスト構築、後段に density rebalance(密度再均衡)が配置される。以下のすべての定数は intensity = 5 / scratch = 5 のデフォルト値であり、いずれも lerp 曲線でスケールする(§6.2)。

### 4.0 実装構造

各メカニズム別ポリシーに入る前に、まずコードレベルの形を示す。PlacementEngine は、一つの状態を保持するループを取り囲む、ほぼ純粋な関数群のシーケンスである。

| 段階 | 関数 | 産出物 | ポリシー § |
|---|---|---|---|
| pool 構築 | `build_pool_universe` | プール トークン + occurrence カウント + events | 4.1 |
| feature | `compute_attack_percentile` / `compute_intensity_origin` / `classify_fx` | pct_map / intensity_origin / FX フラグ | 4.3 / 4.1 |
| ホワイトリスト | `build_whitelist` | whitelist, excluded, band(帯域)統計 | 4.2 |
| phase | `segment_phases` | rush / normal / rest ブロック | 4.4 |
| scratch seed | `_determine_scratch_seeds` | スクラッチ トークン + モード | 4.7 |
| **メイン loop** | `run_per_measure_loop` | placed + residual events (+ end_state) | 4.3–4.7 |
| ↳ measure ごと | `_place_measure_constrained` | 制約下での配置 | 4.5 |
| ↳ note ごと | `_centroid_lane_select` | レーン | 4.6 |
| LN | `run_ln_postprocess` | LN プロモーション(昇格)済み events | 4.8 |
| scratch adj | `run_scratch_adjustment` | LN を考慮したスクラッチ間引き | 4.7 |
| density | `run_density_rebalance` | 均衡化された events | 4.9 |
| 検証 | `_run_conformance` | check pass/fail | 6.1 |

**Per-measure loop**(`run_per_measure_loop` — 唯一の状態保持パス。carry-over 状態は §5.2):

```text
for measure in [start .. end]:                 # default 0 .. measure_max
    rng = Random(seed × 10⁶ + measure)          # β-1 (§5.1)
    curr = reorder_within_idx(cands[measure])   # §4.3
    if ml: curr = ml_token_rerank(curr)         # §9.4 (optional)
    placed, hand_state, residual =
        _place_measure_constrained(curr, rng, hand_state, jack_state, …)
    token_usage += placed                       # under-used boost feeds next measure
    if scratch_active: insert_scratch(measure)  # §4.7
    residualize(unplaced)
# then (full-chart / finalize only): LN → scratch_adj → density_rebalance → conformance
```

**制約ゲートの順序** — `_place_measure_constrained` の内部で、各候補が measure-candidate 順に以下のゲートを通過する。いずれか一つでも失敗すると、理由コードとともに residual へ落とされる。

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

ゲートの順序は load-bearing(本質的な意味を持つ)である。ゲート 6(jack)で拒否されたノートはゲート 8(hand balance)に到達しないため、診断の理由コードは*最初に*違反した制約を反映し、すべての違反を反映するわけではない。§4.1–4.9 で各段階のポリシーと根拠を説明する。

### 4.1 Pool universe と residual ポリシー

pool universe は、使用されたすべてのトークン イベントである。トークンは、サンプル duration が `FX_DURATION_THRESHOLD = 1000 ms` を超える、attack percentile が `FX_ATTACK_THRESHOLD = 20` 以下である、または key/scratch チャンネルに一度も登場しない場合に、**FX**(背景であり、キー候補ではない)として hard-exclude される。サンプルのデコードに失敗したトークンは **unknown** とマークされ、決して rescue されない。

配置されなかったものはすべて **residual** である。それは自身の元の onset(発音点)位置で BGM チャンネル `01` に書き込まれる。これが、曲を*無音にすることなく*演奏性のためにノートを落とせるようにするメカニズムである — 音は依然として鳴り、ただキー入力をもはや要求しないだけになる。

### 4.2 band(帯域)ベースの quota ホワイトリスト (C1)

ホワイトリストは三つの形態を経て進化した。

- **v9** — duration / attack / occurrence に対する単一の global hard filter。
- **v10** — band ベースの quota: eligible なトークンを centroid(スペクトル重心)の三分位で三つのスペクトル band(lo / mid / hi)に分け、各 band が occurrence 加重ランクの上位 `BAND_QUOTA_RATIO` 割合を保持する。
- **v11/v12** — 比率を `0.15 → 0.20` に引き上げ、**rare-token rescue** を追加。

band 内でトークンは次式でランク付けされる。

$$\text{rank}(t) = \text{occ}(t) - 5 \cdot \max\!\left(0, \frac{\text{dur}(t) - \text{WL}_{\text{dur}}}{\text{WL}_{\text{dur}}}\right)$$

`occ` は総 occurrence であり、ペナルティ項は過大な長さのサンプルを抑制する(lv5 で `WL_dur = 1055 ms`)。各 band は `max(3, round(0.20 × |band|))` 個のトークンを保持し、残りは `band_quota` の理由で soft-exclude される。

**Rare rescue。** flat な occurrence カットは、まばらに作譜されたメロディのハイライト(例: Lepontinia m16 の 8X/8Y/9B)を落としてしまう。そこで、総 occurrence ≤ `RARE_OCCURRENCE_THRESHOLD = 3` の*非 FX* トークンは、band ランクに関係なくホワイトリストへ復帰する。FX は §4.1 ですでに除去されているため、`band_quota` の理由は常に「非 FX だがランクでカットされた」を意味し、それはまさに保護する価値のある集合である。

**Windowed rescue。** 8-measure ウィンドウ(`WINDOW_SIZE = 8`)ごとに、ある measure のホワイトリスト通過率が `WINDOW_RESCUE_THRESHOLD = 0.40` を下回る場合、除外されたトークン(ウィンドウ レベルの occurrence でランク付け)をその閾値まで rescue する。これにより、局所的にトークンが乏しい measure が空になるのを防ぐ。

### 4.3 トークン intensity(強度)と within-idx reorder

トークンの **intensity(強度)** は attack percentile(`pct_map`)であり、プール全体の RMS / peak から計算される。これが順序付けを駆動する。すなわち、より大きい onset がある位置の最初のノートとして優先される。

**within-idx reorder**(v11、C1 に隣接)は、少数の大きいトークンが配置を独占する自然な head-heavy 分布を相殺するよう、同一位置の順序付けを精緻化する。最高 attack の最初のピックの後、残りの chord-mate(和音の構成音)は次式でスコア化される。

$$\text{score}(t) = \min_{c \in \text{chosen}} |c_{\text{cent}} - t_{\text{cent}}| - 1000 \cdot \text{usage}(t)$$

— すでに選ばれたメイトとのスペクトル距離を最大化(多様性)しつつ、chart 全体ですでに多用されたトークンにペナルティを課す(`USAGE_WEIGHT_SPREAD = 1000 Hz ≈ 1 回の事前使用`)。最初のピック自体も使用ペナルティを負う(`USAGE_PENALTY_FIRST = 10` attack-pct ポイント / 事前使用 1 回あたり)。

### 4.4 Phase 分割

譜面は、平滑化された候補密度スコアを用いて 4-measure ブロックにわたって `rush` / `normal` / `rest` の phase に分割され、比率の差が `PHASE_MERGE_RATIO_MAX = 0.289` 未満の隣接ブロックを統合する。phase は **診断専用** である — 区間をレポート用にラベル付けするだけで、配置を gating しない。(以前のバージョンは phase-adaptive relaxation を用いていたが、windowed rescue で置き換えられた、§4.2。)

### 4.5 配置制約

各 measure は hard constraint の下で順番に配置される。制約に失敗したノートは residual へ落とされ(onset を保存)、決して再タイミングされない。

- **衝突(Collision)** — `(位置, レーン)` ごとにトークンは一つ。ある位置でトークンごとにレーンは一つ。
- **jack(縦連)床値(Jack floor)** — 同じレーンは `effective_min_ticks = max(MIN_JACK_DELTA_TICKS, ceil(MIN_JACK_DELTA_MS × bpm / 1250))` より速く繰り返してはならない。BPM-aware の項は、固定 tick 床値(lv5 で 15)が速い区間で引き上げられることを意味し、これにより 16 分音符の same-lane 反復があらゆるテンポで遮断される。
- **jack streak** — 一つのレーンは、強制的に外されるまで最大 `MAX_JACK_STREAK = 2` 回の連続した chord-anchored 反復を持てる。
- **chord-size cap** — lv5 で位置あたり最大 `MAX_CHORD_SIZE = 3` の同時レーン。超過分は residual へ落とされる(タイミングは保存され、和音が薄くなるだけ)。
- **hand balance** — measure が ≥ 10 ノートになると、左右の比率を `[0.30, 0.70]` の範囲内に保つ。balance を崩すノートは、使用の少ない手へ誘導されるか、落とされる。
- **same-hand streak** — 新しい位置は、同じ手を `STREAM_MAX_SAME_HAND = 2` を超えて延長してはならない。
- **chord-mate spread**(v11) — chord-mate のレーンを `CHORD_MATE_SPREAD_MIN_GAP = 2` インデックス以上離す soft な選好で、`{1,2,3}` のような隣接レーンのクラスタを排除する。広い和音が詰め込みを強制する場合は、任意の利用可能なレーンへ fallback する。

### 4.6 centroid(スペクトル重心)ベースの相対レーン割り当て (C2)

レーンはランダムにではなく、音楽の **スペクトル重心(centroid)の軌跡** をたどって選択される。前のノートのレーン インデックスと centroid から、次に優先されるレーンは次式である。

$$\text{step} = \text{sign}(\Delta) \cdot \text{LANE\_STEP\_MAX} \cdot \left(1 - e^{-|\Delta| / \text{step\_unit}}\right), \qquad \Delta = c_{\text{cur}} - c_{\text{prev}}$$

より明るいトークン(正の Δ)は手を右へ、より暗いトークンは左へ動かす。saturating(飽和する)指数関数は、小さな変化には敏感である一方、大きなジャンプを `LANE_STEP_MAX = 4` レーンで cap する。`step_unit` は曲ごとに、非ゼロのトークン間 centroid 差の中央値として自動較正され、`CENTROID_STEP_UNIT_FLOOR = 300 Hz` で floor される。これにより、同じ絶対的な明るさの変化が、曲のスペクトル範囲に関わらず一貫した空間的 step へとマッピングされる。

**ε-greedy** 項(`CENTROID_EPSILON_RANDOM = 0.30`)は 30% の確率で任意の利用可能レーンを選び、centroid のドリフトを断ち切って単調な hand-walking を回避する。centroid データが欠落している場合、割り当ては Fisher-Yates シャッフルへ fallback する。

### 4.7 スクラッチ ポリシー — ソース認識ミラー (C3)

スクラッチには三つのモードがある。

- **primary** — ソースにすでにスクラッチ トークン(チャンネル `16`)が存在する。出力はソースの measure ごとのスクラッチ数 × `scale` をミラーリングし、ここで `scale = level / 5`(すなわち scratch = 5 は 1:1 のソース ミラー)である。spacing の床値はレベル テーブルではなく、ソース*自身*の最小間隔である — ソースの作者がすでにリズムを形作っているからだ。
- **fallback** — ソースにスクラッチがない。key トークンから、レベル インデックスの絶対 cap(`SCRATCH_MAX_PER_MEASURE = 4`、scratch = 5 で `SCRATCH_MIN_INTERVAL = 16`)の下でスクラッチを合成し、持続的なバーストの後に `SCRATCH_RUSH_REST_MEASURES = 4` のクールダウンを挿入する RUSH-rest ルールを適用する。
- **disabled** — スクラッチ挿入なし。

RUSH-rest は fallback でのみ発動する。primary モードはソースのペーシングを信頼し、これを無効化する。

### 4.8 ロングノート後処理

配置後、適格な Tap ノートがロングノート(LN)へプロモーション(昇格)される。Tap は、そのトークンのサンプル duration が `LN_MIN_DURATION_MS = 800`(人間の LN duration のおおよそ p75 に合わせたゲート)以上のとき LN 候補となる。描画される hold 長は `LN_MAX_HOLD_TICKS = 96`(2 拍の可視 cap、v11)で cap され、長いサンプルが画面を埋め尽くすバーを描かないようにする。*オーディオ* のサンプルは最後まで再生される — 可視のバーだけが cap される。hold は、宣言されている場合は譜面の `#LNOBJ` トークンを用いて書き込まれる。

既知の緊張(§10.3): 800 ms の選定ゲートが*本来短い*人間の LN の約 75% を遮断するため、パイプラインは LN の多い曲で LN を過少生産する。ゲートを一律に下げるとすべてが over-LN になってしまうため、source-LN-signal インフラが曲ごとのゲートの前提条件となる(future work)。

### 4.9 Density rebalance(密度再均衡)

最終ステップは、4 つの chart セグメントにわたってノート密度を均衡させる。過負荷のセグメントは最低 intensity のトークンを residual へ放出し、低密度のセグメントは residual からトークンを引き戻す(fill-back)。補正は **soft-knee 指数 damping** を用いるため、目標付近では緩やかに、目標から遠いところでは強固に作用し、`DENSITY_REBALANCE_MAX_DELTA`(lv5 で ≈ 0.21)で bound される。fill-back は同じ配置制約(jack、chord size、chord-mate spread)を再適用するため、再均衡されたノートが不正になることは決してない。

Density rebalance は局所的ではなく **chart 全体** に及ぶ — これは Resume API の finalize ステップ(§5.3)にとって重要な性質である。ある measure を reroll した後で finalize すると、*別の* セグメントのノートが動きうる。これは意図的なもの(chart の一貫性が局所編集の保存に優先する)だが、この API の上に構築されるあらゆるエディタ UI で必ず明示されなければならない。

---

## 5. 決定論とResume API (C4)

動機となるユースケースは、外部エディタ(BMS.Compare)が生成された譜面の一部を**再ロール(re-roll)** — ある1つのmeasure、またはmeasure範囲を別のseedで再生成 — しつつ、残りには手を触れないというものである。これには、元のエンジンが持っていなかった2つが必要だった。(a) measure範囲を単独で再生成する能力、(b) 「残りに手を触れない」がbyte-identicalに実際に成り立つだけの真の決定論。§5.1–5.5でその両方をどう達成したかを説明する。

### 5.1 measure単位のRNG (β-1)

元のエンジンは譜面の先頭で単一の`random.Random(seed)`をseedし、それを全measureにわたって消費していた。これは、すべてのmeasureを以前のすべてのmeasureのRNG draw に結合してしまう — 部分再生成にとって致命的であり、というのもmeasure Mを再現するにはまずmeasure 0…M-1を再生しなければならないからだ。

2つの戦略を検討した。

| | α — RNG状態のシリアライズ | β — measure単位のRNG |
|---|---|---|
| メカニズム | `Random.getstate()`(Mersenne-Twisterの624-tuple)をキャリーオーバー状態にダンプ | `(seed, measure)`からmeasureごとに新しいRNGをseed |
| 旧出力とbyte-identical | はい | いいえ(出力が変化、baselineを再生成) |
| スキーマリスク | 高 — Pythonの内部RNGレイアウトを固定 | なし |
| undo/redo | 以前のRNG状態をカスケードする必要あり | 自明 — measure_idxだけで再現 |

**βを選択した**(「β-1」というmeasureごと単一ストリームの変種)。per-measureループは今や次のようにseedする。

$$\text{rng}_{\text{measure}} = \texttt{random.Random}(\text{seed} \times 10^{6} + \text{measure})$$

実装中に1つの落とし穴が浮上した。Python 3.13の`random.Random`はtuple seedを拒否する(`TypeError: only int/float/str/bytes/bytearray`)。そこでtuple-hash形式を上記の算術マッピングに置き換えた。`10⁶`オフセットは、現実的などの譜面長でもcollision-freeである(譜面は数百measure)。あるサブ決定(β-2: per-call-site keyed RNG)は却下された。RNG消費サイトは5箇所しかなく、per-call-site分割では実質的なロバスト性が得られないためである — Resume APIが必要とするのはまさにmeasureレベルの隔離だ。

受け入れたトレードオフ。譜面出力がchart-wide-stream時代とはもはやbyte-identicalでなくなったため、`samples/baseline_lv5/`の回帰baselineをβ-1で再生成した。

### 5.2 キャリーオーバー状態

measure Mでresumeするには、エンジンは左から右のループがM-1の終わりまでに積み上げたであろうcross-measure状態を継承しなければならない。その状態をバージョン明示のJSON(`schema_version: "resume-v1"`)としてシリアライズすると、次のようになる。

| フィールド | 役割 |
|---|---|
| `jack_state` (レーン → last tkey) | jack floor の delta チェック |
| `jack_streak` (レーン → count) | 連続same-laneのcap |
| `centroid` (`prev_lane_idx`, `prev_centroid`) | centroid軌跡の連続性 |
| `hand` (`last_hand`, `streak`) | same-hand-streakのcap |
| `token_usage` (トークン → count) | 低使用トークンのブースト + fill-back順序 |
| `scratch` (`jack_scr_tkey`, `scratch_history`, `scr_rest_remain`) | スクラッチjack / RUSH-restウィンドウ |
| `rng` (`strategy`, `seed`) | β-1のbase seed(per-measure RNGはここから再計算) |

Codex監査が以前の誤同定2つを訂正した。`lane_tkeys`はfinalize時に再構成される(carryされない)、`hand_balance`はmeasure-localである(chart-wideの累積器は診断専用)。centroidの`step_unit`はcarryされ*ない* — chart-input決定論的であり、resumeエントリで再計算される。MLのキャリーオーバー(`token_context`, `lane_context`, `global_lane_counts`)は**v1の範囲外**(RB専用)。`rng.strategy`または`schema_version`が一致しない状態をロードすると`ValueError`を投げる。

### 5.3 resumeモードとfinalizeモード

3つのCLIエントリ形態が1つのエンジンを共有する。

```
(default)                  全体chart、動作不変
--resume-state S \
  --start-measure M \
  --end-measure N          resumeモード: 状態Sから[M,N]を再生成、
                           raw events + end_stateを出力、post-processingをSKIP
--finalize EVENTS          finalizeモード: spliceされた全体chartのeventsを受け取り、
                           post-processingのみ実行
```

**resumeモード**はper-measureループを`[M, N]`に絞り、キャリーオーバー状態を`S`でオーバーライドし、部分的な`placement_result.json`(`mode: "resume"`、M…Nのevents、カスケード用の`end_state`)を出力する。LN / scratch-adjust / density-rebalance / conformanceを完全にスキップする — 呼び出し側がraw eventsをspliceし、後でfinalizeする。`run_pipeline.py`では、resumeモードはBMSWriterとSimilarityCheckもスキップする(部分chartには意味のある`.bms`や類似度がないため)。

**finalizeモード**はその逆である。エディタがspliceした全体chartのeventsが与えられると、post-processingチェーンを固定順序 — LN postprocess → scratch adjustment → density rebalance → conformance — で実行し、通常の全体`placement_result.json`を出力する。density rebalanceはchart-wide(§4.9)であるため、finalizeは再ロールした領域の外のノートを動かしうる。ポリシー上の決定は明示的である(v12 §23.6 / DR-23-4)。**chartの一貫性がユーザーのローカル編集の保存より優先される**ため、finalizeは明示的なユーザーアクションでのみ呼び出され、エディタは再ロールした領域が維持される保証はないと警告しなければならない。

### 5.4 境界lookahead (E-β)

resumeモードは*左*の隣接を完璧に継承する(キャリーオーバー状態がM-1の終端状態である)。一方で*右*の隣接は**見えない**。エンジンの`next_cands` lookaheadはmeasure N+1のrawな候補プールしか読まず、N+1のすでに配置されたレーンは読まない。そのため再ロール領域のN → N+1境界が衝突しうる — N+1の最初のchordに対するjack、またはN+1が引き継ぐhand-streakである。

両隣が固定された単一measureの再ロール(動機となるケース)では、これは重要だ。修正(`--next-chord-lookahead`、v12 §23.7)は呼び出し側にN+1の最初のchordを渡させ、エンジンは*最後のmeasureの最後のchordにのみ*、forward constraintを適用する。

- **jack**(hard) — N+1のレーンと`effective_min_ticks`以内で衝突するレーンをrejectする。
- **hand-streak**(soft) — 選んだ手がN+1の手と等しく、streakがcapを超える場合は反対の手を優先する。

centroidの両側補間(E-γ)は保留された — saturating-curveのレーンセレクタを書き直す必要があり、jack/handとは違ってsoftな選好だからだ。N+1の最初のchordのみを使い、以降のchordはそれによってバッファされる。

### 5.5 決定論バグ: PYTHONHASHSEED

β-1の後にdefaultパスがbyte-identicalのままかを検証していたところ、*同じ*入力の連続2回の実行が*異なる*譜面を生成した。β-1自体は決定論的である。原因は別のところにあった。`PYTHONHASHSEED`がunset(Pythonのデフォルト)だと、文字列および文字列のtupleのハッシュがプロセスごとにrandomizeされ、トークンの`set`/`dict`を反復すると実行ごとに異なる順序が得られる — そしてその順序が配置の決定に漏れ込んでいた。

5つのサイトを`sorted()`またはトークンのタイブレークで堅牢化した。attack-percentileソート、ホワイトリストのeligible反復・band-rank・rare-rescue反復、そしてmeasureごとのscratch-candidate反復である。6番目(ラベリングパイプライン内)は学習決定論のために修正した。修正後、6曲 × {RB, ML} のbaselineは`PYTHONHASHSEED`によらず実行間でbyte-identicalである。この教訓は常設のレビュー項目として保管されている(v12 §22 DR-23-6)。*要素が文字列の`set`/`dict`は保持するだけなら安全だが、その反復順序が決定に入った瞬間、ソートされなければならない。*

---

## 6. Conformanceと校正

パイプラインにはスコアリングできる大規模なラベル付きコーパスがないため、検証は**構成による(by construction)**。すべての出力をポリシー自身の不変条件に対してチェックし、回帰baselineが決定論を守る。

### 6.1 適合性チェック

各実行は、関連する不変条件を保有する2つのステージに分散されたチェックスイートのpass/failを出力する。

- **PlacementEngine** — A ホワイトリストhard-filter、B タイミング保存、C jack禁止(スクラッチレーンを除外)、D fallback動作、E candidate衝突、F スクラッチ制約、G seeded再現性、K measure cap。Check J(density-rebalanceの適法性)は手動検査のみと文書化されている — hard gateにするにはpost-rebalanceのイベント集合に対してper-measureのjack/衝突ロジックを再実行する必要があり、範囲外だからである(v12 §22 DR-J1)。
- **BMSWriter** — A placed完全性、B residual完全性、C タイミングライン保存(BPM / STOP / scaleラインがソースとbyte-identical)、D no-original-playable-leak。

Check B(タイミング保存)とBMSWriterのCheck Cが合わさって、§2.1の核心となる不変条件を強制する。出力のタイミングはソースのタイミングであり、改変されない。

### 6.2 intensityとscratchのスケーリング

`--intensity`(1–20、デフォルト5)と`--scratch`(1–20、デフォルト5)は、§4の定数を lv1 / lv10 / lv20 にアンカーされたpiecewise-linearなlerp曲線に沿ってスケールする。intensityはjack floor、chord-size cap、measure note cap、same-hand streak、stream比率を動かし、scratchはmeasureごとのスクラッチ予算と最小間隔を動かす。デフォルトのlv5が、本レポート全体およびbaselineで用いられる校正点である。

### 6.3 回帰baseline

`samples/baseline_lv5/`は 6曲 × {RB, ML} × {bms, json} = 24ファイルをlv5で保持する。2つの自動スイートがパイプラインを守る。

- `smoke_test_determinism.py` — 6曲すべて(RB + ML)を再生成し、保存されたbaselineとbyte-identicalであることをアサートする。`PYTHONHASHSEED`系の回帰と、偶発的なポリシーのdriftを捕捉する。
- `smoke_test_resume.py` — 9つのResume APIケース: base split、M=0 single、last-measure single、3段カスケード、ML+resume拒否、schema-version mismatch、RNG-strategy mismatch、lookahead-requires-resume、そしてlookahead-wiring smoke。すべて通過。

baselineはβ-1 + 非決定性修正で再生成された。この回帰こそが「β-1が出力を意図的に変えた」をsilent driftではなく一度きりの事象にするものである。

---

## 7. ケーススタディ

### 7.1 mightyA — 47-streakとchord-collapseアーティファクト

初期の診断metric `same_hand_streak`が、mightyAで警告的なfat tail(47 same-handノートのstreak)を示した。本物のhand-balance失敗のように見えた。調べてみると**測定アーティファクト**だった。metricがchord-collapseした位置を別個のsame-handイベントとして数え、密集したchordの壁が実際のsame-handランなしにstreakを膨らませていたのである。chord-awareな再metric(`hand_only_streak`)は source / RB / ML すべてを max 3–8 に収めた — シグナルなし。教訓(v12 §22 DR-K1): 診断における恐ろしい数字は、*譜面*よりもまず*metric*に関する仮説である。

### 7.2 happiness — BPM無頓着なLN cap

`hapiness_lnext`は人手によるLN過多の譜面(757 LN)である。パイプラインは0–6を生成した。2つの原因が重なった。800 msの選定ゲート(§4.8)が自然に短い人間のLNの約75%を遮断したこと、そして以前のdraw-length ポリシーがBPM無頓着だったこと — 242 BPMでは固定tickのholdが画面を埋め尽くすバーを描いた。draw-length cap `LN_MAX_HOLD_TICKS = 96`(可視のみ)がバーを描く問題を修正した(v12 §22 DR-G1)。選定ゲートの問題は、一律に引き下げるとあらゆる譜面をover-LNしてしまうため、未解決のまま残っている(§10.3)。

### 7.3 単一measure再ロール — Resume APIのend-to-end

`samples/reroll_demo_2026-05-18/`がAPIを実演する。signal譜面のmeasure 58を単独で再ロールし(prefix `[0,57]` → `end_state` → N+1 lookaheadつきでresume `[58,58]`)、その後spliceする。出力`.bms`はbaseと**measure 58においてのみ**異なる — 他のすべてのmeasureはbyte-identicalであり、β-1のmeasure隔離を確認する。(このデモは限界も露呈した。m58の最後のchordがスクラッチレーンに落ち、そこにはKEYレーンのjack lookaheadが適用されないため、lane-swapの実演ではなくwiringの実演になっている — §10を参照。)

### 7.4 ソースchord構成の乖離

横断的な発見(v12 §22 DR-K2): パイプラインはソースのchord構成を体系的に改変する。mightyAではソースの6%の両手MIXED chordが出力では39%になり、happinessでは67% → 34%となった。方向は曲によって一貫しない。疑われる原因は、pool dedupがチャンネル情報を捨て、centroidレーン割り当てがchord-mateをspreadするため、ソースの片手chordが両手に分割して着地しうるというものだ。可聴性(「より忙しい」)はもっともらしいが客観化が難しい — 確定したバグではなく、未解決の項目である。

---

## 8. ダブルプレイ合成 (`--dp`)

シングルプレイ(SP)譜面は7本のキーレーンと1本のスクラッチを、一対の手で演奏する。**ダブルプレイ(DP)**譜面はこれを14キー + 2スクラッチへと倍化し、独立した2つのside(手)— **P1**(左手、キーレーン1〜7 + スクラッチ)と **P2**(右手)— に分割する。コミュニティはDP譜面をSPよりはるかに少なくしか制作せず、多くのパッケージがSP専用で配布されるため、ある曲のDPレンダリングを望むプレイヤーはしばしば演奏するものがない。`--dp` モードは **SPソースから** これを1つ合成する。

DPは §2.4 のソース忠実な姿勢をそのまま守る。ソースのonset(発音点)は決して動かさず、SPのプール宇宙・ホワイトリスト・intensity(強度)・レーン制約を変更なく再利用する。DPが新たに導入する*唯一の*決定は **side(手)** — 各ノートをどちらの手が演奏するか — であり、その後に既存のside内レーン選択(§4.6)が続く。したがってDPもSP経路と同様、難易度エンジンではない。同じソースの音イベントを演奏可能なレーン空間へ投影したものであり、その空間がいまや14+2になっただけである。

> **状態.** DPは実装済み(RB-only、フルチャート)で、3種類のソースcharacter(stream、peak、chord)で聴取評価を通過している。これは **draft拡張** であって、まだnormativeな(規範的)ポリシーではない(別途DRAFT addendumとして管理)。以下の構造的設計 — スプリットルーター、side-localな配置、スクラッチポリシー、タイミング / SP-byte同一性の保証 — は安定している。§8.5 のパターン後処理の定数はまだ聴取チューニング中であり *provisional*(暫定)として明示する。値は動きうるが、このレイヤーの不変条件(タイミング保存、新規jack(jack)なし、SP無改変)は固定である。

### 8.1 人間のDP譜面が実際に行うこと (corpus測定)

設計は **1,852譜面の人間DP譜面** の調査(`tools/dp_corpus_survey.py`、コミュニティDP corpusと照合)にアンカーされている。3つの発見がポリシーを形作った。

- **両手は同時に演奏する。交互ではない。** 空でないbeatの82.8%が *両手(BOTH)* beatであり、数beatを超える片手runは事実上存在しない(32beat以上の片手run: 0.06%)。片手をmeasure単位で休ませる初期設計(「measureブロックALTERNATE」)はこの証拠により破棄された — 人間のDPはターン制ではなく *同時コンテンツの分担* である。(DR-DP2)
- **同じ音 ⇒ 同じ手 が第一の規則性である。** トークンのsideは、その特定のレーン(0.376)よりもはるかに一貫している(modal-side share **0.641**)。そのためDPは2段階のaffinityを持つ。強いトークン→**side**の記憶と、side内ではSPエンジンから再利用される弱いトークン→レーンの記憶である。
- **隣接皿 / 「無理皿(impossible scratch)」の同時押しはきわめて稀である。** スクラッチが同じsideの1〜3領域のキーと同時に鳴る(隣接)のは1.5%、4〜7領域(無理皿、「unreasonable」)は1.2%である。これらの比率はcurator文化できれいに分かれ(現代「stella」系のテーブルは0.1〜1%、旧来の「satellite」は4〜5%)、したがって一律禁止は恣意的な厳格化ではなく、現代的な慣習の採用である(DR-DP3)。

3つすべてが **難易度family間で不変** であった(family別に再測定。side-chord、両手同時、トークン→side shareは★から★★★までほとんど動かない)ため、lv5のアンカーには難易度別の補正が不要だった。

### 8.2 Side割り当て — スプリットルーター

中核となる決定は *各ノートをどちらの手が演奏するか* である。デフォルトモードは **SPLIT** で、phraseブロック内で分離軸を見つけ、結果として得られる2つのstreamを2つの手に割り当てる。

- **balance**(`--dp-split balance`、**デフォルト**、`auto`→balance)は音色(timbre)を無視し、**負荷(load)** で分割する — chordを両手に分解(centroid交代)し、持続するburstを軽い手へ交代させる。**全corpus聴取により balance ≥ timbre が確立された(DR-DP13)**。chord/peak曲ではtimbreが似た音色のノートを片手に積み上げてjackドロップを引き起こし、stream曲では両者が区別不能である — そのためbalanceが普遍的なデフォルトとなる。
- **timbre**(`--dp-split timbre`、opt-in)は **低域エネルギー比** で分割する — ベース / キックを一方の手に、より明るい残りをもう一方の手に。この軸は経験的に実在する。13のSPソース × 1,216 measureにわたり、最良の2分割のクラス間分散比(η²)の中央値は0.913で、99.8%のmeasureで0.5を上回る(DR-DP4)。ただしこれは「より明るい残り」を一方の手(R)に積み上げて負荷の偏りを生むため、stream曲限定のopt-inとして残している。*balance band*(`DP_MIN_SIDE_SHARE = 0.25`)とmeasure単位の50/50フォールバックが、いずれの分割でも片手を枯渇させないようにする。
- **auto** は **balance** に行く(DR-DP13)。balanceが普遍的なデフォルトになったことで、DR-DP7のauto-routing問題 — 曲のcharacterごとに戦略を選ぶ問題 — はまるごと消滅する(常にbalance、character分類なし)。

phrase **ブロック**(既存のphase分割 §4.4 を、ここで初の実運用へと昇格させたもの)が粒度であり、トークン→side affinityがブロック境界を越えて手の割り当てを接着するため、粒度が精密である必要はない。*mirror-on-repeat* ヒューリスティックは、phraseクラスが再出現したときにその手をトグルして変化を与える。

### 8.3 Side-localな配置とDP固有の制約

配置はSPの機構(§4.5)を **side(手)ごとに** 再利用する。各手が独立したjack / hand / centroid / streak / affinityの状態で `_place_measure_constrained` を実行し、右手のレーンは P1_* → P2_* へリマップされる。SP固有のhand-balanceとsame-hand-alternationのルールは無効化される(side割り当てがそれを包含する)。jack、collision、streak、scale-awareなtick軸は per-lane であるため変更されない。

DPは2つの制約を追加する。

- **side単位のchord cap.** `DP_MAX_CHORD_SIZE_PER_SIDE` はDP固有のlerpである — lv5で2、lv20で5まで上昇する(指は5本。4は安定、5は極端)。合算した両手capはside単位capの2倍である。(人間のside-chordの94%が ≤ 2 であるため、lv5 = 2 のアンカー。)
- **スクラッチゲート(hard).** あるsideにスクラッチが置かれると、そのタイムスタンプ(+ chord-tierウィンドウ)における **同一sideの7本のキーレーンすべて** が利用可能性から除去される — 隣接 / 無理皿の区別なしの一律ブロック、DR-DP3に基づく。キーはもう一方の手に移るか、residual化される。スクラッチの *side* は負荷回避補正を伴う交代で選ばれる(DR-DP5)。直前のスクラッチと反対のsideを優先するが、交代がゲートと衝突するか一方の手を過負荷にする場合は、もう一方のsideに譲る。

### 8.4 コンテンツのレスキューと無音バグ

DPの聴取により、SPホワイトリストとの2つの相互作用が露呈した(Codexと相互チェック、DR-DP8)。

- **演奏コンテンツのレスキュー.** SPホワイトリストのFX-durationフィルタ(1000 msを超えるサンプルは背景として扱う、§4.1)が、hardtekのようなジャンルの長いシンセ *リード* をFXと誤分類し、それらのキーonsetの最大 ~79% を切り落とす。durationフィルタの除去は *unsound*(不健全)である(そのトークンのBGM出現まで譜面に氾濫させてしまう)。採用した修正は **位置ターゲット** 型である。除外されたトークンを *ソースがキーチャンネル(11〜19)で演奏した位置に限り* DP候補として復元し、soft padを除外するためのattackゲートを設ける — トークンのBGM出現には手を付けないため、氾濫は起きない。bumblebeeのソースキー演奏率は62% → 84%へ上昇した。SP経路は無改変である。
- **無音バグ.** DPのearly-return経路は元々、ホワイトリストから *除外* されたトークンをBGMへ回すSPステップを欠いていた。そのためそれらの音は演奏も自動再生もされず — **完全に無音** だった(bumblebeeのソースキーコンテンツの33%)。これが聴こえる「メロディのドロップアウト」の直接の原因だった。DP経路にresidual構築を追加して音声を復元し、DPはいまやSPと同じ方法で全ソース音声を保存する。

### 8.5 パターン後処理 — タイミング不変なレーン精製

配置後のレイヤー(`_dp_postprocess`、per-measureループの後・writerの前、`if dp:` でゲート)が、配置済みノートの *レーン* を精製する — **そのタイミングは決して変えない**。これはDP専用である。SP経路は `lane_weights = None` を渡し、このレイヤーのゲートによってSP出力はpre-DP時代と **byte-identical(バイト単位で同一)** になる。毎回の実行で `dp_pp_report` 診断により検証されるこのレイヤーのhard不変条件は、タイミング不変、コンテンツ保存(`(measure, idx, token)` の多重集合を保存する — 拒否されたノートはBGMへ回り、決して消えない)、そして **新規jackを導入しない** ことである。

決定的な教訓(DR-DP9)は、独立した精製パスどうしが互いに干渉し、jack保証を破るということだった。最初のバージョンはchord / stair / trillを、レーンを自由に再割り当てする別個のパスとして実行し — *jackを113個導入してしまった*(高速なパッセージをtrillと誤検出し、手が交代できる速度を超えた2レーンに強制した — 「jackより速いtrillはtrillではない」)。再設計では、すべてのパスを **共有されたside単位のlane-tickタイムラインの上での原子的コミット** として実行する。パターンはすべてのメンバーがjack floorをクリアした場合にのみ適用され、そうでなければrun全体が元のjack-safeなレーンへ **フォールバック** する。後処理は配置時のhard制約(jack)を *再検証して譲る* ことはできるが、それを違反することは決してできない。

パスは順に以下のとおり。

- **chord** — chordのレーンをスペクトル重心(セントロイド)で並べ替え、cap超過のノートを空いた手のjack-safeなフリーレーンへ再分配する。
- **stair** — 単調なrunを昇順 / 降順の階段に整える。
- **trill** — 交代を ≥ 2 離れた2つのレーンへマッピングする。
- **禁止形状** — **最後** に、fixpointループとともに実行される。stairが、これが崩すべき隣接ペアを生み出すからである。不自然な片手同時押し形状 `{2,3}` と `{5,6,7}`(右手ではミラー)を *contains-match* で禁止し、**zure**(near-simultaneousなノートを指すIIDX用語)を含む。`DP_PP_ZURE_TICKS = 4`(48分音符のウィンドウ)以内のノートは、1つのchordのように演奏されるため実効的なchordとして扱われる。違反したノートはjack-safeなフリーレーンに分散される。

2つの分布レベルの制御がこのレイヤーを締めくくる。どちらも *事後のleverは一部のノートにしか触れないため、分布の目標は配置時に駆動しなければならない* という教訓に従う。

- **density cap** — 合算した両手のper-measure上限(`DP_PP_COMBINED_MEASURE_CAP = 40`、*provisional*)が、最も弱いattackの超過分をBGMへ落とす。これは *ピーク* リミッターであって、全体的な密度の削減ではない(capを下回るmeasureは無改変)。全体の密度はintensityの仕事である(§8.6)。
- **rail lane-weights** — DPの片手が7キーすべてをカバーするため、薬指のレーン(キー2〜3)は押しにくい。per-laneの出現重み(`{2: 0.9, 3: 0.95, 5: 0.9}`、ミラー。*provisional*)を、centroidレーン選択器のkeep-確率として **配置時に** 注入する — 後処理のdensity / stairパスから駆動するのは弱すぎることが判明した(ノイズのみ)。SPは `lane_weights = None` ガードにより無影響である。

最後に、**cross-measureスクラッチゲート**(`_dp_pp_scratch_gate`、DR-DP11)が §8.3 のスクラッチ禁止を *グローバルな* tick軸上で強制する。per-measureゲートはmeasure内のインデックスしか見なかったため、隣接皿 / 無理皿のスクラッチが小節境界を越えて漏れていた(signalの76件の違反のうち64件がcross-measure)。chordパスの後に実行して再分配されたノートまで捕捉するグローバルゲートが、無理皿スクラッチをゼロに追い込む。教訓は、境界をまたぐ制約はper-measureではなくグローバルなtick軸で強制しなければならない、ということである。

### 8.6 character別のintensity

DP固有のキャリブレーションの教訓(DR-DP11)。DPでは **intensityは弱い全体密度のlever** である。ソースのonsetがそのまま敷かれ、side単位のcapが拘束的であるため、lv8とlv12はほぼ同じ密度を生む。lv5だけが中央値を有意に薄くする。intensityが強く制御するのは **side単位のchord容量**(`DP_MAX_CHORD_SIZE_PER_SIDE`)である。そこから導かれる指針は以下のとおり。

- **chord-characterの曲 → 高いintensity**。曲のchordが必要とする同時押しの容量を両手に与えるため。
- **stream / peak / stairの曲 → 控えめな(低い)intensity** — これらは追加の容量を必要としない密な単音テクスチャであり、低いintensityのほうが可読性を保つ。

(stream曲のsignalはlv5で確定した。)

### 8.7 ソース認識スクラッチ生成 (DR-S1)

§4.7 のSPスクラッチポリシーは3つのモード(primaryミラー / fallback合成 / disabled)を持っていたが、DPは長らく **primaryミラーのみ** を持っていた — そのため `--scratch level > 5` がno-opだった。2つの観察がこれを解決した。

第一に、**スクラッチは音響的な属性ではなく、譜面(配置)の決定である**。同じ `#WAV` がある譜面ではキーで、別の譜面では皿で鳴らされるため、音声特徴から「スクラッチ *トークン*」を分類しようとする試みは頑健に失敗する(rank-AUC 0.47〜0.62、偶然レベル)。しかし第二に、スクラッチの **位置は予測可能** である。約600譜面のSP corpus(49.6kのスクラッチonset)にわたり、スクラッチは強拍に集中し(quarter rate-lift 2.19×、on-quarter rank-AUC 0.69 — 音声の天井を上回る)、beat-phaseエントロピーがキー(0.532)より低い(0.314)。DPも同様である。**音色は作者の主観的な判断だが、位置はリズムの慣習である。**

そこでスクラッチ生成を **move(移動)** として解く — 新規生成ではなく。生成器はスクラッチに *適格* なresidual(BGM)のonsetを選び、それをスクラッチ盤(皿)へ移す。

- **適格性** = 2a(ソース自身の皿トークン、チャンネル `16` に出現)∪ 2b(機能的なゲート: 短いduration、十分なattack、反復的、キー由来 — 背景FXではない)。移されたonsetは *自身のトークンをそのまま* 保つ(置換なし)。適格性は、メロディックなリードを皿に引きずり込むのを防ぐ緩いフィルタにすぎない。
- **density(密度)** はソースではなく **人間のDP corpus にアンカー** される(active measure当たりのスクラッチ p50 = 0.26 / p90 = 1.03。SPは0.69 / 1.79 — DPは両手がキーで忙しいため、より疎である)。levelがp50とp90の間をlerpする。
- **配置の慣習**: measureあたり単一のスクラッチ手(既存のスクラッチmeasureはその手を保ち、新規のものは交代する)。§8.3 のanti-jumpゲートが、新しいスクラッチに隣接するsame-handのキーをBGMへ降格させる。

移動するだけであるため、`(measure, idx, token)` の多重集合は保存される — §8.5 のコンテンツ不変条件と同じ保証であり、**新規生成ゼロ** である。したがってDPのprimary補完は **デフォルトでON** になる(`level > 5` のno-opを解決。`level ≤ 5` はミラーであり、byte-identicalのまま)。SPの位置再ランキングとfallback合成はopt-inのまま残す(SPのデフォルト出力は不変。fallbackはcorpusのサンプルがないため、強制fallbackのフィクスチャ経由でe2e検証のみ)。

**正直な天井.** 位置の選択と密度の *目標* はソース非依存だが、候補の *availability(可用性)* と密度の *天井* はソースに束縛される(新規生成ゼロの必然)。適格なresidual onsetが枯渇すると生成器は頭打ちになる(mightyAは ~62 でcap — ソース天井の証拠。供給の豊富なlepontiniaはさらに先へ進む)。これは欠陥ではなく、§2.4 の姿勢の直接的な帰結である。すなわち、新規生成なしに到達可能な最大の位置非依存性である。

### 8.8 Conformanceと DP固有の制限

DPは5曲のsmokeスイート(`tools/_dp_smoke.py`)を実行する。side-chord > 2 がゼロ、合算chord > 3 がゼロ、スクラッチゲート違反がゼロ、collisionがゼロ。毎回の実行は `dp_pp_report`(タイミング不変フラグ、jackのbefore / after、パターンの適用 / フォールバック数、per-lane分布)も出力し、A/Bハーネス(`tools/dp_pp_report.py`)が後処理レイヤーをon / offで再検証する。DPはresume / finalize / MLを拒否する — フルチャート、RB-onlyである。

未解決の項目:

- **配置の妥当性における新規生成なしの天井 — onset-inventionのフロンティア.** §8.7 のmoveベースの生成は、スクラッチを強拍に整列した *適格residual* の位置に置く。それらの位置が曲のグルーヴ(キック / スネア)から逸れるとき — グルーヴのスロットがキーやスクラッチ不適格なトークンに占有されているため — moveだけではそこに到達できない。到達するには **onset-invention(発音点の新規生成)**(曲の既存パレットのトークンを *新しい時刻* に置くこと。これは人間の譜面制作者が行うこと)が必要である。これを測るために、onsetコアリションのバックボーンスコア — ある tickの *すべての* イベントからの証拠を合算する(単一の支配トークン検出を精緻化したもの)— を測定した。合成スコアは位置の天井(per-chart AUC 0.69 → 0.77)を上回るが、位置を固定したうえでの *条件付き* 信号は薄く(on-quarter onset内でAUC 0.56)、その薄い信号を既存のmoveランキングに注入しても配置品質の指標は変化しなかった — moveの候補がすでに一様に強拍整列であり、弱く配置されたスクラッチの大半が、我々が動かせない *ソースミラー* だからである。結論: 新規生成なしのスクラッチの価値は *より良いランキング* ではなく *reach(到達)* であり、そのreach(= onset-invention)は §2.4 のソース忠実な姿勢が構成上スコープ外と保持する *リタイミング* に相当する — そのため、これは意図的にフロンティアとして残される(トークンSET ⊆ ソース、および譜面保存の不変条件のもとでの将来の作業)。
- **6つのcharacterのうち2つが未カバー.** スプリットルーターはstream / peak / chordを扱う。**ln**(ロングノート対タップの手分割)と **soft** のcharacterはまだ扱われていない。
- **戦略のauto-routingは未解決**(DR-DP7)。characterはユーザーが選ばなければならない。
- **normativeへの昇格は** より広範な検証、LNパス、そしてより厳密なrail-weightのキャリブレーション(拒否されたノートからのスピルオーバーにより、現在のrailバイアスは正確な比率ではなく「feel(感覚)」である)を待っている。

## 9. 機械学習 — 学習・統合し、凍結

ML経路はスケッチではない。両モデルはともにラベルcorpusからend-to-endで学習され、TorchScriptへexportされ、`--ml`の背後でライブ推論経路にwiringされている。ここで全貌を記録する理由は、「MLを試したが勝てなかった」がコミュニティにとって有用であるためには、*どのように* — データ、アーキテクチャ、学習setup — が記録に残らねばならないからだ。

| モデル | 置き換え対象 | パラメータ | Loss | 状態 |
|---|---|---:|---|---|
| `TokenSelectionModel` | pctベースのcandidate順序付け (§4.3) | ≈ 6.3K | masked BCE | wired(`--ml`)、測定可能な利得なし |
| `LaneAssignmentModel` | centroidレーン選択 (§4.6) | ≈ 24.8K | masked CE | wired(`--ml`)、測定可能な利得なし |

### 9.1 データ準備 — ラベリングパイプライン

学習データは`data_labeling.py`が実際の人間の譜面から抽出する。パッケージ内のすべての譜面のすべての適格measureについて、*状況*(measure + プール + contextの特徴)と*人間の決定*(どのトークンをどのレーンで演奏したか)を対応づけたレコードを出力する。

- **プール特徴** — トークンあたり14列、パッケージごとにプールテーブルに一度だけ格納(レコードは整数`pool_index`でトークンを参照): `duration_ms`、`attack_rms`、`attack_peak`、`intensity_origin`、key / scratch / bgm のoccurrenceカウント、6個のSTFTスペクトル特徴(centroidの平均/標準偏差、flatness、low-frequency比、zero-crossingの平均/標準偏差)、whitelist-passフラグ。
- **measure特徴** — `measure_index`、chartレベルの`density_rank`、`phase`、`notes_in_measure`。
- **contextウィンドウ** — 先行するC = 4個の*適格*measure(不適格measureはskip、chart開始はoldest-firstでゼロパディング)、それぞれが`tkey_delta`、`placed_count`、および配置されたpool-index / レーンの履歴を持つ。推論時のcontext builder(`placement_engine._build_ml_context`)がこの順序を正確に再現するため、学習と推論は同一のテンソルレイアウトを見る。
- **ラベル** — トークンラベルは0/1(演奏したか否か → BCE)、レーンラベルは人間のレーン1..7(→ CE)。ground-truthのレーンが制約上利用不可なレコードは*skip*され、不正な手を教えることはない。

パイプラインはフルスケール(~6,395パッケージ)で実行された。実行中に**v2 → v3のスキーマ再設計**が強制された。v2は書き込み前にすべてのレコードをメモリに蓄積しており、これが全corpusでOOMクラッシュを起こし~471 GBのディスクを埋めた。v3はレコードをJSONLへstreamingし、パッケージレベルのプールテーブルを持つ(14特徴の行をレコードごとのpayloadから持ち上げる)。そして §5.5 に従い、トークン順序付けは後に学習の決定論性のためソートされた。

### 9.2 モデルアーキテクチャ

どちらも固定の推論インターフェースを持つ、意図的に小さなMLPである。テンソルの列順序はモデルI/O契約に固定されており、retrainが特徴をsilentにずらすことはできない。

- **TokenSelectionModel** — 可変サイズのプールに対するsiameseスコアラー。プール行ごとに`[4 measure ⊕ 14 pool ⊕ 12 flattened context] = 30`次元をconcatし、次いで`LayerNorm(30) → Linear(30,64) → ReLU → Dropout(0.3) → Linear(64,64) → ReLU → Dropout(0.3) → Linear(64,1)`で、トークンあたり1スコアにsqueezeする。可変プールサイズ`P`は、共通幅へpaddingする代わりにmeasureを行軸に沿ってconcatすることで処理する。≈ 6,269パラメータ。
- **LaneAssignmentModel** — 7-way分類器。`[16 event ⊕ 40 flattened context] = 56`次元をconcatし、次いで`LayerNorm(56) → Linear(56,128) → ReLU → Dropout(0.3) → Linear(128,128) → ReLU → Dropout(0.3) → Linear(128,7)`、softmax前に利用不可レーンを`-inf`にマスクするため、呼び出し側は直接argmaxできる。≈ 24,823パラメータ。

小ささは意図的である(設計上)。モデルは*RBを補助する再ランカー*であって支配的な決定者ではないため、capacityの上限がモデルがルール制約を圧倒することを防ぐ。どちらも最初のレイヤーが`LayerNorm`である — 入力特徴が非常に異なる自然スケール(`duration_ms`が数百に対し`attack_rms`は[0,1])にまたがっており、per-sample正規化により推論時のrunning statistics維持を回避する。

### 9.3 学習setup

- **Optimizer** — Adam、`lr = 1e-3`、`weight_decay = 1e-4`、各hidden ReLUの後に`Dropout(0.3)`。BatchNormなし(可変-`P`のsiameseバッチおよび小バッチと相性が悪い)。
- **スケジュール** — 最大20 epoch + early stopping(`patience = 3`)、batch size 256。
- **Class weighting (lane)** — 人間のレーン分布は不均衡である。レーンモデルはclass-weighted cross-entropy(`--class-weights auto --class-weight-power 2.0`)でretrainされ、稀なレーンをinverse-frequencyの2乗でup-weightする。
- **Split** — パッケージレベルの決定論的shuffle(`seed = 42`、`DR-7`): あるパッケージのmeasureがtrain/val境界をまたがないため、モデルは後で検証される譜面を記憶できない。
- **Export** — TorchScript `script`(control flowを残すため`trace`ではない、`DR-4`)、推論時に`map_location="cpu"`でロード(`DR-8`) — 生成にGPUは不要。
- **環境** — 記録すべき運用上の落とし穴: Python 3.13 + GTX 1070 はCUDA **cu118** のPyTorchビルドを要する(cu121は3.13のwheelを提供しない)。class-weightedなレーンのretrain(`training/checkpoints`、`lane_cw2` TensorBoard run)はこのsetupで実施された。

### 9.4 統合 — ルールベースfallbackを持つsoftな再ランカー

両モデルは*softな再ランカー*として統合される。RB方針があらゆる構造的決定(どのsegment、何ノート、どの制約)を所有し、モデルはRBが許容した集合の中で順序を入れ替えるだけである。推論失敗時(例外、shape mismatch、レーンモデル無効、利用可能性なし)は呼び出しがルール経路へfallbackする — トークンモデルはpct順序付けへ、レーンモデルはcentroid / Fisher-Yates選択へ。構成上、モデルは順序付けを改善できても制約を*決して*違反できない。fill-backのrankingフック(density rebalanceがトークンモデルでpull-back順序を決める)は後に別個の診断カウンタとともに追加された。

### 9.5 判定、そしてmetric-blindnessのcaveat

v9 baselineではレーンモデルの~50% top-1精度(chance 25%に対し)は決定的に見えた。しかしそのbaselineは*ランダム*なレーン割り当てだった。RB経路がcentroidレーン割り当て(§4.6)を採用すると、RB baseline自体が学習可能な構造の大部分を捉え、2026-05-03の統計評価は両モデルともRBに対して**測定可能な優位なし**を発見した。その監査からの別の発見: レーンモデルはK1/K3/K4のレーンpriorを学習していた — contextではなく最も一般的なレーンに依存しており、これはまさにclass weighting(§9.3)が導入されて相殺しようとしたものだ。

正直なcaveat: ブラインドA/B聴取でML出力は繰り返し、より安定的/より人間的に*感じられた*が、RB-alignedなmetricでそれを捉えたものはなかった。したがって判定は「*測定可能な*優位なし」であり、明示的なmetric-blindnessの可能性を伴う — 聴取の印象はfactであり、その定量化のみが未解決である(§10.2)。class-weightedなretrainはレーンpriorを改善したが判定を動かさず、その結果は凍結の決定に吸収された。

### 9.6 教訓 — 概念的な健全性は経験的な採用を保証しない

二つのアイデアは概念的に綺麗だった — トークンの選好を知るトークンモデル、人間のレーン習慣を学んだレーンモデル — そしてルールが十分に良くなると、両者ともより安価なルールに敗れた。正直な読み(ユーザーが記録)は、モデルが*校正された(calibrated)*譜面から学んだことがないというものだ — 学習corpusは「判定上良い譜面」ではなく「人間が作った何であれ」であり、校正されていないcorpusは校正されたモデルを生み出せない。ギャップはモデルのcapacityではなく学習setupにある(v12 §22 DR-H1)。MLは削除ではなくフラグの背後に凍結され、future re-designのために契約は生きているが、injection point の追加(fill-back、scratch seed、LN candidate)は直感ではなく測定された便益にgatingされる。同じ形がcharacter-frameworkのaudio-FFT階段検出でも再現する: *健全なアイデアは仮説であり、仮説はパイプラインに席を得る前に経験的な監査を通過せねばならない。*

---

## 10. 限界と今後の課題

### 10.1 ソース依存

出力はソースの投影である。希薄または単調なソースは希薄または単調な譜面を生む。パイプラインはリズムを発明できない。なぜならそれがタイミング不変条件を破るからだ(§2.4)。これは欠陥ではなく意図的な境界だが — パイプラインが*作曲家*ではなく*レンダラー*であることを意味する。

### 10.2 聴取-proxyのギャップ

最も強い未解決問題: RB-alignedなmetricがML-vs-RBの聴取差を捉えない(§9.2)。あるmetricがブラインドA/Bの印象と相関するまで、いかなる「MLの方が悪い」という主張もmetric限定である。計画された次のステップは、聴取-分解protocol(短いA/B区間 + ユーザーの「安定的」アノテーション)により、新しいmetricを設計する前に仮説を絞ることだ — 最初のmetricの試み(same-hand fat-tail)はすでにchord-collapseのartifactとして失敗している(§7.1)。

### 10.3 LN-スタイルのblindness

RBもMLもソースのLN*スタイル*にblindである。`build_pool_universe`とラベリングパイプラインはどちらもLong eventを`(start, token)`のペアに平坦化し、hold長を捨てる。そのため学習データ自体にLN-スタイルの次元がなく、RBゲートは固定の800 msである。source-LN-signalのインフラ(`(start, end)`のholdティックを保存、パッケージごとのLN統計を導出、動的ゲートを駆動)が、per-songのRBゲートと将来のLN-awareモデルの両方の前提条件である。

### 10.4 Resume API v1のスコープ

v1はRB専用・単一パスである。MLのresume、部分的(領域局所)な適合性チェック、centroid両側lookahead(E-γ)はすべてスコープ外。コード構成の乖離(§7.4)とスクラッチレーンのlookaheadギャップ(§7.3)は既知であり追跡中である。

## 付録 (Appendix)

### A. ハイパーパラメータリファレンス (intensity = 5 / scratch = 5 デフォルト)

```text
# FX分類
FX_DURATION_THRESHOLD          = 1000   # ms
FX_ATTACK_THRESHOLD            = 20     # percentile
FX_ORIGIN_FILTER_ENABLED       = true

# Band ホワイトリスト
BAND_QUOTA_RATIO               = 0.20
RARE_OCCURRENCE_THRESHOLD      = 3
WINDOW_SIZE                    = 8      # measures
WINDOW_RESCUE_THRESHOLD        = 0.40
WHITELIST_DURATION_MAX         = 1055   # ms

# Within-idx reorder
USAGE_PENALTY_FIRST            = 10.0   # attack-pct point / 使用1回
USAGE_WEIGHT_SPREAD            = 1000.0 # Hz ≈ 使用1回

# Phase
PHASE_MERGE_RATIO_MAX          = 0.289

# レーン割り当て (centroid)
PLACEMENT_RANDOM_SEED          = 42
LANE_STEP_MAX                  = 4.0
CENTROID_EPSILON_RANDOM        = 0.30
CENTROID_STEP_UNIT_FLOOR       = 300    # Hz

# Stream / hand
STREAM_CHORD_RATIO_MAX         = 0.311
STREAM_MAX_SAME_HAND           = 2
MEASURE_NOTE_CAP               = 32

# 縦連(jack) (BPM-aware)
MIN_JACK_DELTA_TICKS           = 15
MIN_JACK_DELTA_MS              = 102
MAX_JACK_STREAK                = 2

# コード
MAX_CHORD_SIZE                 = 3
CHORD_MATE_SPREAD_MIN_GAP      = 2      # レーン (soft)

# スクラッチ
SCRATCH_MIN_INTERVAL           = 16     # ticks
SCRATCH_MAX_PER_MEASURE        = 4
SCRATCH_RUSH_WINDOW            = 3
SCRATCH_RUSH_THRESHOLD         = 3
SCRATCH_RUSH_REST_MEASURES     = 4
SCRATCH_FALLBACK_DURATION_MAX  = 300    # ms

# LN
LN_MIN_DURATION_MS             = 800
LN_MAX_HOLD_TICKS              = 96     # 2-beat 可視 cap

# Density rebalance
DENSITY_REBALANCE_MAX_DELTA    ≈ 0.21
```

完全な表と lerp 曲線: `placement_engine.py` の `compute_intensity_params` を参照。

### B. Conformance check 表

| Check | ステージ | 不変条件 |
|---|---|---|
| A whitelist hard-filter | Placement | 配置トークンが FX/unknown hard filter を通過 |
| B タイミング保存 | Placement | すべての配置ノートがソース onset 上に位置 |
| C 縦連(jack)禁止 | Placement | jack 下限未満の same-lane 反復なし(スクラッチ除外) |
| D fallback 動作 | Placement | primitive-failed measure がクリーンに residual 化 |
| E candidate 衝突 | Placement | `(pos, lane)` / `(pos, token)` の重複なし |
| F スクラッチ制約 | Placement | スクラッチ間隔 / 密度 / RUSH を遵守 |
| G seeded 再現性 | Placement | 同一 seed → 同一配置 |
| K measure cap | Placement | `MEASURE_NOTE_CAP` を超える measure なし |
| J density rebalance | Placement | 手動検査のみ(DR-J1) |
| A placed 完全性 | Writer | すべての配置イベントをレンダリング |
| B residual 完全性 | Writer | すべての residual を BGM へレンダリング |
| C タイミングライン保存 | Writer | BPM / STOP / scale 行がソースと byte-identical |
| D no-original-playable leak | Writer | ソースの演奏ノートが改変されずに通過しない |

### C. CLI リファレンス

```text
python run_pipeline.py --folder <package> [options]

--intensity <1-20>           note aggressiveness (default 5)
--scratch <1-20>             scratch frequency, source-aware mirror (default 5)
--ln                         enable LN post-processing
--ml                         enable ML soft-ranking (non-recommended, §9)
--model-token / --model-lane TorchScript paths (required with --ml)
--bms <filename>             explicit source chart select
--seed <int|random>          placement seed (default 42)

# Double-play synthesis (§8)
--dp                         synthesize a double-play (14-key) chart
--dp-split <timbre|balance|auto>  side-split strategy (default balance; auto→balance, DR-DP13)

# Resume API (§5)
--resume-state <path>        carry-over state JSON (resume mode)
--start-measure <M>          resume range start
--end-measure <N>            resume range end
--next-chord-lookahead <p>   N+1 first-chord boundary input (requires --resume-state)
--finalize <events.json>     post-processing-only over spliced events
```

### D. 運用 corpus (ソースパッケージセット)

ポリシーが乗り越えるべき character を網羅する 13 個のソースパッケージ:

| Package | ストレステスト対象 |
|---|---|
| addiction | 標準譜面 — RB/ML 比較の起点 |
| mightyA | 高密度譜面 — fill-back streak, chord-collapse artifact (§7.1) |
| blacksphere | short-LN スタイル, MIXED-chord 構成 |
| signal | distraction-style の短いキー音 (reroll デモ, §7.3) |
| bumblebee | hardtek, トークン豊富 (baseline + smoke アンカー) |
| tsuramic | 一般的な stream |
| marion | melodic, multi-chart variant 選択 |
| lepontinia | rare-token rescue (§4.2) |
| happiness | 242 BPM + LN 過多 (§7.2) |
| sacrifice | LN + collision |
| nakama | coverage gap (soft) |
| egosa, wanwan | baseline-set の一般譜面 |

回帰 baseline: このうち 6 曲(bumblebee / egosa / lepontinia / signal / tsuramic / wanwan) × {RB, ML} × {bms, json} を lv5 で (§6.3)。

DP 拡張(§8)は別途、人間の DP 譜面 1,852 個の調査(§8.1)にアンカーされており、この SP 回帰 baseline ではなく 5 曲の DP smoke スイート + 実行ごとの `dp_pp_report` 診断によって検証される。

---

*本レポートは 2026-05-25 時点の BMS.Generator のノート配置パイプラインを、draft のダブルプレイ(DP)合成 addendum(§8)とともに記述する。正確な動作の権威ある出典はソースコードであり、本ドキュメントは読者向けの叙述的な統合である。*
