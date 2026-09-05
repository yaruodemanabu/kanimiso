# 仕様書とテストのカバー状況

この文書はアルゴリズムを追加しない。`d581ba8`（2026-09-05）の
クローンを数えた棚卸しである。推測値はない。

対象はワークスペース 9 crate と、同じリポジトリに置かれているが
`Cargo.toml` の `members` に入っていない `Isuzu` / `amatsuki` / `wormhole`
（`coronel` / `jelly-wave` を含む）。

正本の役割分担:

| 文書 / 台帳 | 役割 | この文書との関係 |
|---|---|---|
| [`docs/validation.md`](validation.md) | Verified の証拠契約（層、許容差、CI 再生） | 方針。件数や欠落はここに書かない |
| `kanimiso::coverage::inventory()` | README が参照する機械可読台帳 | 主張の正本。ここに無い公開型は未主張 |
| 各 crate の `docs/validation.md` | その crate だけの証拠と除外範囲 | 下表の「検証契約」 |
| Isuzu `docs/` | 文献付きのアルゴリズム仕様と偏差台帳 | ワークスペースの Verified 契約ではない |
| この文書 | 仕様の有無とテストの層をcrate横断で照合する | 監査用。完了条件ではない |

行数は `find … -name '*.rs'` の `wc -l`、テスト数は `#[test]` 属性の個数。
1 本の巨大 smoke が数十モデルを踏む場合もあるので、テスト数 ≠ 検証された概念数。

## 文書の種類

リポジトリにある Markdown は一様に「仕様書」ではない。

| 種類 | あるもの | ない / 薄いもの |
|---|---|---|
| 決定書（何を作らないか） | ルート `AGENTS.md` | — |
| 検証契約（何をどの証拠で Verified にするか） | `docs/validation.md`、`number-ruler` / `oldwood` / `mayoi-no-mori` / `denshi` / `riko` の `docs/validation.md` | `tsutsumi`、`signlred`、`ojizou-san`、`kanimiso` 本体 |
| 所有境界 | `mayoi-no-mori/docs/architecture.md`、`oldwood/docs/forest-integration.md` | 他 crate |
| 文献付きアルゴリズム仕様 | Isuzu vault（`docs/` 15 本 + README、合計 1,601 行） | ワークスペース crate のほぼ全部。式は rustdoc とコードに散在 |
| 機能対応表（検証状態ではない） | `wormhole/docs/pot-coverage.md` | — |
| 移行・削除名 | `docs/dropped_v0_1.md`、`generated-v0.1-archive/` | — |

`tsutsumi` の README は核の範囲とオラクルを書くが、検証契約としては
`docs/validation.md` の特殊関数行に依存している。`signlred` / `ojizou-san`
は品質契約の rustdoc が仕様で、数値オラクルは持たない。

## ワークスペース全体

| crate | `*.rs` 行 | `#[test]` | 検証契約 | golden / 外部 fixture | 台帳 |
|---|---:|---:|---|---|---|
| `signlred` | 2,551 | 8 | なし（README + rustdoc） | なし | 非掲載 |
| `ojizou-san` | 857 | 2 | なし（README + rustdoc） | なし | 非掲載 |
| `tsutsumi` | 3,829 | 38 | README のみ | `special_functions.json`（1,099 件、scipy） | `special.*` / `optimize.NelderMead` を kanimiso 台帳が参照 |
| `number-ruler` | 2,983 | 14 | あり | `golden/regression.json`（statsmodels / SciPy） | LM / GLM / SHAP が Verified、混合・加法は Experimental |
| `oldwood` | 3,500 | 36 | あり | `golden/sklearn_cart.csv`（sklearn 1.7.2） | CART 2 型が Verified |
| `mayoi-no-mori` | 6,249 | 70 | あり | なし（CART 核は oldwood に依存） | ensemble 13 型が Experimental |
| `denshi` | 380 | 4 | あり | なし（閉形式） | **非掲載** |
| `riko` | 365 | 4 | あり | なし（閉形式） | **非掲載** |
| `kanimiso` | 78,036 | 339 | なし（ルート検証方針 + `coverage.rs`） | ルート `golden/*.json` | 71 項目（Verified 46 / Experimental 25） |

ワークスペース合計の `#[test]` は上表の和で 515。このうち kanimiso が約 66%。
line coverage の ratchet は Linux / Windows 実測 76.30%、下限 76.0%。
これは実行行の割合であり、オラクル一致の割合ではない。

`scripts/lint_redundancy.sh` の `SRC` は
`kanimiso signlred ojizou-san tsutsumi number-ruler oldwood mayoi-no-mori Isuzu/amatsuki`
で、**`denshi` と `riko` を含まない**。

## クレートごと

### `signlred`

仕様は README と rustdoc の品質契約（`Failure` / `Qualified<T>` / `Policy` /
`NumericalCompromise`）。`Policy` には `log_prob_floor`、`underflow_guard`、
`max_difference_order`、`cf_tol` など AGENTS.md D8 の欄がある。

テスト 8 本は定数ターゲット abort、guard、severity の契約確認。数値核の
オラクルはない。これ自体は正しい（品質型に scipy は要らない）。欠落は
「どの IssueCode がどの推定器から必ず出るか」の対応表がないこと。

### `ojizou-san`

仕様は README の ledger 契約。テスト 2 本は session が compromise /
meaningless を記録すること。`IncrementalExplain` の必須フィールドを推定器
横断で検査するテストはない。

### `tsutsumi`

実装範囲は行列契約、faer 分解 / 最小二乗、特殊関数、Nelder–Mead、
Gauss–Hermite。専用の `docs/validation.md` はない。

| 核 | テストの層 | 台帳上の状態 |
|---|---|---|
| 特殊関数 | Tier 0（`golden/special_functions.json` 再生）+ 恒等式 + §0.3 回帰 | ほぼ Verified。`chi2_pvalue` だけ Experimental（上側 p が fixture 未収録、`docs/validation.md` 記載どおり） |
| Nelder–Mead | 反射 / 拡大 / 内外収縮 / shrink / 既知二次・Rosenbrock / 置換・平行移動 / 非有限 / 停止理由（17 本） | Experimental。argmin への依存は無い。外部 solver との数値 fixture も無い |
| 求積 | 正規モーメント、対称性、次数（2 本） | 台帳に項目なし（`number-ruler` GLMM が利用） |
| 線形代数 | 特異、極端値、SPD 境界、非有限、再構成（5 本） | 台帳に項目なし（OLS / 状態空間が利用） |

`norm_cdf_half` と `chi2_df2_mean` は許容差 `1e-6` / `0.02` で、R9 の
「実測 × 3〜4」コメントが付いていない。golden 再生側は関数ごとの閾値を使う。

### `number-ruler`

検証契約が一番厚い。LM / canonical GLM / 線形 interventional SHAP が
Verified。ランダム切片混合と区分線形加法は、外部照合があっても
Experimental（実装範囲が狭く、罰則後 Wald などを出さないため。契約に明記）。

テスト 14 本のうち統合テスト `tests/analysis.rs` が fixture 再生の本体。
許容差は 2026-09-05 Windows 実測の約 4 倍。GLMM の Poisson 固定 128 点積分は
実測絶対誤差 5.83e-7、閾値 2.34e-6 で、契約が「尖った count 尤度は Hermite
が足りない」と先に書いている。

未実装で契約が明示的に切っているもの: offset、試行数、負の二項、一般リンク、
ランダム傾き、三次スプライン、GCV/REML 平滑化選択、ロバスト / cluster SE、
予測区間。

### `oldwood`

決定的 binary CART の仕様は README + `docs/validation.md` +
`docs/forest-integration.md`。検証は sklearn 1.7.2 の 3 ケース（絶対許容差
0.0）、解析不純度、独立 brute-force、置換 / 重み尺度、arena 構造、失敗経路。

意図的に見ないもの: threshold の格納、child index、`apply` の leaf ID、
欠損分岐、カテゴリ部分集合、剪定、multi-output、forest。

### `mayoi-no-mori`

所有境界の仕様はある。検証契約は「ensemble 出力の外部 fixture が無いので
全部 Experimental」。70 本は閉形式（定数 SSE、SAMME 重み、Isolation の
2 点 path）、oldwood brute-force 核、確率和、seed 再現、OOB 拒否、
histogram 単調、失敗経路。LightGBM / CatBoost 名は subset であり、
upstream パリティは契約上ない。

### `denshi` / `riko`

どちらも検証契約 + 閉形式 4 本。Kodansha のオンライン学習 / バンディット
本を設計参照にし、SMPyBandits / VW は数値オラクルにしないと明記している。

| crate | 実装 | 契約上の除外 |
|---|---|---|
| `denshi` | Hedge、射影 OGD | 文脈付き、遅延フィードバック、確率的バンディット（`riko` 側） |
| `riko` | UCB、adversarial 指数重み | 文脈付き、組合せ、連続腕、非定常、事後サンプリング |

`kanimiso` はどちらも依存も再公開もしない。`kanimiso::bandit` は別実装。
台帳にも載らない。CI の `quality-small` clippy / test パッケージ列にも
入っていない（MSRV の `cargo test --workspace` と `workspace-policy` では走る）。

### `kanimiso`

統合 API と品質アダプター。crate 専用の仕様書は無い。主張は
`coverage.rs` の 71 項目だけ。

モジュール構成は AGENTS.md §5 の目標より広い。`hmm/` は 3,022 行で
Gaussian / categorical / Poisson と log-space forward–backward / Viterbi /
Baum–Welch がある。§5 が挙げた `diagnostics.rs`、`cosine_power.rs`、
`two_sided_power.rs`、`transformed.rs` は無い。Baum–Welch は
`model.rs` に折り込まれている。

## 台帳 71 項目

### Verified（46）

| 領域 | 項目 | 主な証拠 |
|---|---|---|
| 回帰 | `number_ruler.{LinearModel,GeneralizedLinearModel,linear_shap}`、`linear_model.LinearRegression` | statsmodels / Decimal OLS / 全 coalition SHAP |
| CART | `oldwood.DecisionTree{Classifier,Regressor}` | sklearn 3 ケース + brute-force |
| オンライン | RLS と 10 個のモーメント / 閾値 | Decimal 一括式、閉形式 EW、失敗時非更新 |
| 特殊関数 | erf, ln_gamma, digamma, gamma_p, betainc_reg, norm/χ² CDF, t/F CDF と p、正規上側 | scipy JSON 1,099 + 裾の直接計算 |
| 状態空間 | `LinearGaussianStateSpace` | Decimal 結合正規の条件付け |
| フィルタ | BK / CF / lfilter 系 7 | statsmodels ソース + 閉形式再帰 |
| ARMA | acf / acovf / impulse / arma2ar / arma2ma | Decimal Lyapunov |
| ボラティリティ | GARCH(1,1)、FIGARCH(1,d,1)、FIEGARCH Eq.11 | Decimal QML |
| 過程 | `stats.process_mle` lite | Decimal dense GLS、5 点 range |

### Experimental（25）

| 項目 | テストはあるか | Experimental の理由（既存文書） |
|---|---|---|
| `number_ruler.MixedModel` / `AdditiveModel` | 外部照合あり | ランダム切片と指定スプラインだけ。ストレス不足 |
| `mayoi_no_mori.*`（13） | 性質・閉形式あり | ensemble 単位の外部 fixture なし |
| `special.chi2_pvalue` | 一部恒等式 | golden に上側 p が未収録 |
| `hmm.*`（4） | Decimal 全経路 + ストレス | 出自の異なる hmmlearn fixture なし |
| `anomaly.KnnDistanceAnomaly` | 閉形式 Minkowski | 外部 k-NN 異常度 fixture なし |
| `optimize.NelderMead` | 分岐 trace は厚い | 外部 solver 数値 fixture なし |
| `tsa.EwmaVol` / `Egarch` | Decimal golden を CI が再生 | 外部 package の fit 照合なし（GARCH 族と同じ但し書き）。ただし GARCH/FIGARCH/FIEGARCH は Verified |
| `tsa.arma_generate_sample` | 長さ・決定性・失敗の原子性 | 分布オラクルなし |

EWMA / EGARCH は golden と CI 再生があるのに Experimental、同系統の
GARCH は Verified、という非対称が台帳に残っている。

## 台帳に無い kanimiso 実装

`kanimiso/src` のうち、hmm / online / tsa / state_space / filters / stats と
薄い `tsutsumi` 再公開を除くと **約 58,000 行**。公開モジュールは実装と
`#[test]` を持つが、`inventory()` に名前が無い。README はこれらを
Verified とも Experimental とも言わない。

残りのテストはおおよそ 180 本で、ほぼ smoke（小さい合成データで精度や
有限性を見る）。golden 再生は無い。大きい例:

| モジュール | 行 | `#[test]` | 中身 |
|---|---:|---:|---|
| `glm` | 6,185 | 6 | Probit, NB2, GEE, ordered logit, ZIP, AFT, Heckman ほかを少数の巨大テストで踏む |
| `metrics` | 4,933 | 10 | 分類 / 回帰 / 距離。閉形式に近いものあり |
| `model_selection` | 4,292 | 8 | split / CV / grid。一部は漏洩フラグの性質 |
| `linear_model`（OLS 以外） | 4,288 の大半 | OLS 以外は smoke | WLS, ridge, lasso, elastic-net, LARS, logistic, kernel ridge |
| `decompose` | 3,918 | 4 | PCA, NMF, ICA, FA, CCA |
| `cluster` | 3,547 | 8 | k-means, DBSCAN, GMM, spectral, AP, HDBSCAN |
| `feature` / `preprocess` | 2,805 / 2,607 | 4 / 4 | 選択、RFF、scaler。preprocess は平均 0・分散 1 など |
| `svm` | 2,219 | 5 | Pegasos / SMO |
| `bandit` | 1,720 | 2 | ε-greedy, UCB1, Thompson。`riko` とは無関係 |
| `neighbors` | 1,548 | 8 | k-NN, LOF, KDE。`KnnDistanceAnomaly` 以外は未掲載 |
| `mixed` | 825 | 5 | ランダム切片 / 傾き。`number-ruler::MixedModel` とは別経路 |
| `neural` / `gp` / `topic` / `vecm` ほか | 各 250–900 | 1–3 | smoke |

アダプター（実装の複製ではない）: `tree`、`histgb`、`data`、`special`、
`optimize`、`context`、`traits`、`validate`。`classification` は再公開と
局所 perceptron / calibration の混在。

R5 超過ファイル 6 本はすべてこの未掲載側
（`glm` / `metrics` / `linear_model` / `model_selection` / `decompose` /
`cluster`）。冗長性 lint の行数予算は「消すまで残す」ratchet で、
検証契約ではない。

## ワークスペース外

### `Isuzu`（22,879 行、`#[test]` 163）+ `amatsuki`（1,500 行、21）

リポジトリ内で唯一、文献・式・「原文から進んだ箇所」が揃っている。
`docs/` は Obsidian vault で、シミュレーション、モデル、推定、点過程、
フィルタ、金融、HFT / 制御 / Malliavin、NPBayes をカバーする。
`deviations.md`（130 行）が意図的差分の正本。

統合テスト:

| ファイル | 層 |
|---|---|
| `tests/audit_oracles.rs` | 閉形式 / KF / 既知ベクトル（粒子尤度、Van Loan、stick-breaking ほか） |
| `tests/shreve_oracles.rs` | BS / 木 / PDE / LSM / Merton の完了条件 |
| `tests/yuima_surface.rs` | YUIMA 風 API の表面 |
| `tests/toy_recovery.rs` | トイ SDE の回復 |
| `tests/npbayes.rs` | DP / IBP 系 |

`Isuzu/.github/workflows/ci.yml` はルート CI と独立で、clippy は
`-D clippy::correctness` まで、coverage ratchet も redundancy lint もない。
ルート `members` に入っていないので、`cargo test --workspace` も
`llvm-cov --workspace` も Isuzu を見ない。例外は
`mayoi-no-mori` が使う `amatsuki` の core-only テスト
（`tree-ensemble` ジョブ）。

`Isuzu/src/finance/special.rs` に `erf` / `norm_cdf` がある。
`tsutsumi::special` とは別実装。

### `wormhole`（12,226 行、`#[test]` 97）

`docs/pot-coverage.md` は POT 0.9.7.post1 との**機能**対応
（implemented / partial / planned）。数値の Verified 台帳ではない。
`tests/pot_oracles.rs`（25 本）と `kanimiso_integration.rs` がある。
ルート workspace 外。kanimiso 側から距離 / OT を呼ぶ核統合は、
AGENTS.md PR 9 のまま未着手に見える。

## 同じ概念の複数実装

AGENTS.md R2 / R6 に照らすと、文書上まだ一本化されていない対:

| 概念 | 実装 A | 実装 B | 備考 |
|---|---|---|---|
| 線形ガウス KF / RTS | `kanimiso::state_space`（Joseph、Verified） | `Isuzu` フィルタ（KB は古典形、離散は Joseph） | ワークスペース境界をまたぐ |
| 特殊関数 | `tsutsumi::special` | `Isuzu::finance::special` | erf / norm_cdf が重複 |
| Nelder–Mead | `tsutsumi::optimize` | `Isuzu` の箱制約クリップ版 | 係数は通例同じ、停止と制約が違う |
| バンディット | `riko`（閉形式、契約あり） | `kanimiso::bandit`（smoke、台帳外） | 依存関係なし |
| 混合モデル | `number-ruler::MixedModel` | `kanimiso::mixed` | 後者は台帳外 |
| RNG | `amatsuki::ChaCha8Rng`（KAT あり、forest が使用） | `kanimiso::rng` xorshift64*（性質テストあり、台帳外） | 用途が違うが「ワークスペース内 1 核」ではない |
| オンライン学習 | `denshi` Hedge / OGD | kanimiso の SGD / PA（`glm` など） | プロトコルも品質契約も別 |

## CI が再生するもの

ルート `.github/workflows/ci.yml` が明示的に踏む数値核:

- `tsutsumi` 特殊関数 golden
- `tsutsumi` + `number-ruler` 全テスト
- oldwood + mayoi-no-mori
- HMM（Python Decimal check + Rust `hmm::`）
- OLS / 状態空間 / online RLS / process MLE / ARMA / lfilter / GARCH 族
- `amatsuki --no-default-features`
- workspace 全体の test / doc / llvm-cov 76.0%

再生しない / 弱いもの:

- `denshi` / `riko` の専用ジョブなし。clippy `-D warnings` の support 列からも外れている
- 台帳外 kanimiso モジュールの oracle ジョブなし（workspace test で smoke だけ走る）
- Isuzu / wormhole のルート CI なし
- hmmlearn / arch / LightGBM / CatBoost の外部 fit fixture なし

## 疑問点

実装や台帳変更の提案ではない。判断が無いと棚卸しが曖昧な点。

1. **台帳外の約 58,000 行をどう扱うか。** Experimental として名前を載せる、
   `generated-v0.1-archive` と同様にモジュールから外す、未主張のまま残す、
   のどれが v0.2 の方針か。公開 API があるのに `inventory()` が黙っている
   状態は、README の「台帳にないものを検証済みと扱わない」とは両立するが、
   「使ってよい Experimental」とも書いていない。
2. **`denshi` / `riko` を `coverage.rs` に載せるか。** 閉形式と検証契約は
   Verified の定義に近い。一方で kanimiso が再公開しておらず、
   support crate の clippy 列と redundancy `SRC` からも外れている。
   「独立 crate だから台帳外」なのか「追加漏れ」なのか。
3. **EWMA / EGARCH と GARCH の非対称。** どちらも Decimal golden と CI 再生が
   ある。Verified に上げない残件は「外部 arch の fit 照合」だけか。
   それなら FIGARCH / FIEGARCH も同じ但し書きを抱えたまま Verified になっている。
4. **HMM を Experimental に留める条件。** Decimal 全経路がある。足りないのは
   hmmlearn だけか。AGENTS.md §5 の cosine / TSP / `Transformed` が未着手なのは
   「残す分布族の三条件」で後回し、でよいか。
5. **Nelder–Mead の昇格条件。** 分岐テストは厚い。外部 argmin 数値 fixture が
   必須か、既知関数 + 分岐で足りるか。D11 は「argmin に依存しない」と
   「分岐を個別検査」までで、Verified とは書いていない。
6. **`chi2_pvalue`。** 正規・t・F の上側は Verified。χ² 上側だけ fixture 未収録。
   追加は golden 再生成だけで足りるか。
7. **Isuzu / wormhole の位置。** 同じ git リポジトリにあるが、ルートの
   Verified、line coverage、redundancy 予算の外。参照実装・将来核・別製品の
   どれとして監査するか。R6「数値核は workspace 内で 1 実装」を
   Isuzu の KF / 特殊関数 / NM に適用するか。
8. **`kanimiso::bandit` と `riko`、`kanimiso::mixed` と `number-ruler`。**
   残すならどちらが核か。消すなら `docs/dropped_v0_1.md` 行きか。
9. **line coverage 76%。** 未掲載モジュールの smoke が行を稼いでいる。
   核だけを測る ratchet に分割するか、現状の workspace 合計を続けるか。
10. **「仕様書」の定義。** この棚卸しでは検証契約と文献付き式を分けた。
    ワークスペース crate に Isuzu 型の文献ノートを増やすのか、
    rustdoc + `docs/validation.md` で足りるとするのか。
11. **この文書の寿命。** 人手更新のスナップショットにするか、
    `coverage.rs` と `#[test]` から機械生成する正本にするか。
    後者にするなら、台帳外モジュールの扱い（疑問 1）を先に決める必要がある。
12. **`linear_model.rs` の OLS 以外。** `LinearRegression` だけ Verified で、
    同じファイルに ridge / lasso / logistic が同居する。R5 予算 4,288 行の
    対象でもある。分割して未検証側を落とすのか、ファイルごと残すのか。

新しい Verified を増やす手続きは従来どおり `docs/validation.md` 末尾。
この文書は手続きを置き換えない。
