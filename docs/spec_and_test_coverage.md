# 仕様書とテストのカバー状況（2026-09-05 実測）

この文書は実装を増やさない監査メモである。README の機能表でも
`coverage::inventory()` の台帳でもなく、**仕様書がどこまであり、テストが
どこまでそれを覆っているか**を、ワークスペース内クレートとリポジトリに
同梱されているサブクレートまで含めて書いた。

数値はクローン上で数えた。推測値は「未確認」と書いた。
`#[test]` の個数は属性の出現回数であり、1 本のテストが数百ケースを回す
こともある。この作業では `cargo test` も `cargo llvm-cov` も再実行していない。
line coverage の 76.30% / 下限 76.0% は [`docs/validation.md`](validation.md) と
CI コメントの記録値である。

関連する正本:

- 検証の定義と主要オラクル: [`docs/validation.md`](validation.md)
- 検証状態の台帳: `kanimiso::coverage::inventory()` / `verified()`
- 何を作らないか: [`AGENTS.md`](../AGENTS.md)
- 削除済み v0.1 面: [`docs/dropped_v0_1.md`](dropped_v0_1.md)
- 残モジュールのサブクレート化と `ogi` の切り出し計画: [`docs/ogi_subcrate_plan.md`](ogi_subcrate_plan.md)

---

## 1. いま何が「仕様」か

リポジトリには少なくとも 4 種類の文書があり、互いに同じ言葉を使っていない。

| 種類 | 置き場 | 何を約束するか | テストとの結び |
|---|---|---|---|
| 検証契約 | `docs/validation.md` と各 crate の `docs/validation.md` | Verified の証拠層、オラクル、許容差、既知の限界 | 台帳と CI ジョブの説明。実装一覧ではない |
| 台帳 | `kanimiso/src/coverage.rs` | 名前付き API の `Verified` / `Experimental` | 名前の一意性、allowlist、シンボル存在。証拠そのものは別ファイル |
| アルゴリズム仕様 | `oldwood/docs/*`、`mayoi-no-mori/docs/architecture.md`、Isuzu vault、`wormhole/docs/pot-coverage.md` | 分岐・所有権・文献・逸脱・POT 対応 | 対応テストがあるものと、文書だけのものがある |
| 方針 | `AGENTS.md`、各 README | 何を残すか、何を主張しないか | CI の ratchet と PR テンプレート |

`Generated` / `Stub` は enum として残っているが、現行台帳の 71 件は
`Verified` 46 + `Experimental` 25 だけである。台帳に無い公開モジュールは
「未登録」であり、Experimental ですらない。

---

## 2. リポジトリ全体の実測

Rust ファイル（`target` / `.git` 除外）: **267 ファイル、133,855 行、`#[test]` 775**。

AGENTS.md §0（2026-09-01、`main` = `99c46d0`）の 96 万行 / テスト 186 は、
生成型アーカイブ後の現状と一致しない。アーカイブ本文は
`generated-v0.1-archive/` に残るが、コンパイル対象ではない。

### 2.1 ワークスペースメンバー（`Cargo.toml`）

| crate | Rust 行 | `#[test]` | 専用仕様書 | 台帳 | ルート CI での扱い |
|---|---:|---:|---|---|---|
| `kanimiso` | 78,036 | 339 | ルート `docs/validation.md` | 正本 | 領域別 oracle ジョブ + workspace 一括 |
| `tsutsumi` | 3,829 | 38 | README のみ。`docs/validation.md` なし | 間接（`special.*` / `optimize.NelderMead`） | `special-oracle` / `regression-shared-kernels` / quality-small |
| `number-ruler` | 2,983 | 14 | [`number-ruler/docs/validation.md`](../number-ruler/docs/validation.md) | 5 件（Verified 3 / Experimental 2） | `regression-shared-kernels` / quality-small |
| `oldwood` | 3,500 | 36 | [`oldwood/docs/validation.md`](../oldwood/docs/validation.md)、[`forest-integration.md`](../oldwood/docs/forest-integration.md) | CART 2 件 Verified | `tree-ensemble` / quality-small |
| `mayoi-no-mori` | 6,249 | 70 | [`validation.md`](../mayoi-no-mori/docs/validation.md)、[`architecture.md`](../mayoi-no-mori/docs/architecture.md) | ensemble 13 件すべて Experimental | `tree-ensemble` / quality-small |
| `signlred` | 2,551 | 8 | README のみ | なし | quality-small（clippy `-D warnings`） |
| `ojizou-san` | 857 | 2 | README のみ | なし | quality-small |
| `denshi` | 380 | 4 | [`denshi/docs/validation.md`](../denshi/docs/validation.md) | **なし** | 名前付きジョブなし。MSRV / workspace-policy の `--workspace` に含まれる |
| `riko` | 365 | 4 | [`riko/docs/validation.md`](../riko/docs/validation.md) | **なし** | denshi と同じ |

`quality-small` の clippy / rustdoc `-D warnings` 対象は
`signlred` `ojizou-san` `tsutsumi` `number-ruler` `oldwood` `mayoi-no-mori` まで。
`denshi` と `riko` はそこから外れている。

### 2.2 ワークスペース外（同梱サブクレート）

| crate | Rust 行 | `#[test]` | 仕様書 | ルート workspace / llvm-cov |
|---|---:|---:|---|---|
| `Isuzu` | 22,879 | 163 | Obsidian vault（`Isuzu/docs/*`）+ 独自 CI | **メンバー外**。ルート CI は `amatsuki --no-default-features` のみ |
| `Isuzu/amatsuki` | 1,500 | 21 | [`Isuzu/docs/rng.md`](../Isuzu/docs/rng.md) | `exclude = ["Isuzu/amatsuki"]`。`tree-ensemble` が core-only を実行 |
| `wormhole`（本体 + coronel + jelly-wave + tests） | 12,226 | 97 | [`wormhole/docs/pot-coverage.md`](../wormhole/docs/pot-coverage.md) | **メンバー外**。ルート CI にジョブなし |

`wormhole` 内訳: `src` 10,011 行 / 58 テスト、`coronel` 502 行 / 6、
`jelly-wave` 643 行 / 7、統合・POT テスト 1,070 行 / 26。

---

## 3. 台帳が覆っている面

`coverage.rs` の 71 件を、証拠の種類で分けた。

### 3.1 Verified（46）

| 領域 | 台帳名 | 仕様の置き場 | 主な証拠 | コメント |
|---|---|---|---|---|
| 回帰 | `number_ruler.LinearModel` / `GeneralizedLinearModel` / `linear_shap` | number-ruler validation | `golden/regression.json`（statsmodels / 全 coalition） | 家族は Gaussian / Bernoulli / Poisson に限定と明記 |
| CART | `oldwood.DecisionTreeClassifier` / `Regressor` | oldwood validation | sklearn 1.7.2 CSV + 解析解 + 独立 brute-force | レイアウト parity は対象外。許容差 0.0 |
| OLS アダプタ | `linear_model.LinearRegression` | ルート validation | `golden/ols.json`（80 桁 Decimal） | `linear_model.rs` 自体は 4,288 行で未登録 API を多く含む |
| オンライン | `online.*` 11 件 | ルート validation | `golden/online_rls.json` + 閉形式モーメント | 失敗時の状態不変も CI で再生 |
| 特殊関数 | `special.*` 12 件（`chi2_pvalue` 以外） | tsutsumi README + ルート validation | `golden/special_functions.json` 1,099 ケース | 実装は `tsutsumi`。kanimiso は再公開 |
| 状態空間 | `state_space.LinearGaussianStateSpace` | ルート validation | `golden/state_space.json`（joint Gaussian） | 外部 package 由来ではない |
| フィルタ | `filters.*` 7 件 | ルート validation | statsmodels ソース oracle + 閉形式 + 状態空間再利用 | `LocalLinearTrend` は Kalman 核の薄いラッパ |
| ARMA | `tsa.arma_*` 5 件 | ルート validation | `golden/arma_acov.json` + 比多項式恒等式 | `arma_generate_sample` は Experimental |
| ボラティリティ | `tsa.Garch11` / `Figarch` / `Fiegarch` | ルート validation | 各 `*_qml.json` Decimal | 外部 arch package の fit 照合は未追加と明記 |
| 過程 | `stats.process_mle` | ルート validation | `golden/process_mle.json`（5 range の dense GLS） | 連続 range 最大化ではない |

### 3.2 Experimental（25）— 実装とテストはあるが昇格条件を満たさない

| 台帳名 | 既にある証拠 | 足りないと言っているもの |
|---|---|---|
| `number_ruler.MixedModel` / `AdditiveModel` | MixedLM / 独立積分 / 閉形式縮約。許容差まで記録 | ランダム切片と指定スプライン以外の設計、ストレス不足 |
| `mayoi_no_mori.*` 13 件 | 閉形式・性質・CART 核の外部 fixture | ensemble 全体の外部 fixture。LightGBM / CatBoost は subset 明記済み |
| `hmm.*` 4 件 | `golden/hmm.json` Decimal 全経路、log-space、ストレス | 出自の異なる hmmlearn fixture。仕様解釈の独立性 |
| `special.chi2_pvalue` | 実装と単位区間テスト | special golden に上側 p が未収録 |
| `anomaly.KnnDistanceAnomaly` | モジュール内テスト 11 | 独立距離核オラクルの明文化が弱い |
| `optimize.NelderMead` | tsutsumi で分岐 trace・Rosenbrock・失敗 18 本 | 台帳上は Experimental。argmin パッケージ照合はしない方針 |
| `tsa.EwmaVol` / `Egarch` | `ewma_qml.json` / `egarch_qml.json` を CI で再生 | Verified にしていない。GARCH 族と同じ「外部 fit 未照合」か、別理由かは台帳から読めない |
| `tsa.arma_generate_sample` | 決定性と失敗の原子性テストあり | 分布オラクルなし |

### 3.3 台帳に無いが、仕様書とテストがある

| 面 | 仕様 | テスト | 問題 |
|---|---|---|---|
| `denshi::{Hedge, OnlineGradientDescent}` | 閉形式 + Kodansha / VW は設計参照 | 4 本。許容差までコメント | 台帳も README 機能表も未登録 |
| `riko::{Ucb, ExpWeights}` | 同上 | 4 本 | 同上。`kanimiso::bandit` とも別実装 |
| `tsutsumi::{linalg, quadrature, logsumexp}` | README の「単独検証」 | 線形代数 4、求積 2、logsumexp 7 | 台帳名がない。特殊関数と NM だけが間接登録 |
| `signlred::Policy` ほか | README の品質契約、D8 フィールド | 8 本（slug、severity、D8、意味のなさ） | 数値核ではないが、契約の正本なのに検証文書がない |
| `ojizou-san::Session` | README | 2 本 | 同上 |
| `amatsuki::ChaCha8Rng` | Isuzu `docs/rng.md` | 21（crate 全体）。ルート CI は `--no-default-features` | `distributions` feature のテストはルート CI に乗っていない |

### 3.4 台帳に無く、検証契約にも載っていない公開面

`kanimiso` の `pub mod` 50 のうち、台帳 prefix にもアダプタ（`tree` / `histgb`）
にもインフラ再公開にも入らないもの:

`bandit` `bayes` `classification` `cluster` `compose` `covariance` `decompose`
`discriminant` `ensemble` `feature` `glm` `gp` `kernel_pca` `manifold` `metrics`
`mixed` `model_selection` `multiclass` `multinomial` `multioutput` `naive_bayes`
`neighbors` `neural` `preprocess` `random_projection` `reducer` `rng` `robust`
`semi` `svm` `text` `topic` `vecm`

実測: **49,879 行、130 テスト、公開項目 1,035**。
台帳側（アダプタ込み）は **27,562 行、206 テスト、公開項目 256**。

テスト密度は未登録側が低い（行あたり）。R5 超過 6 ファイルはすべてこの未登録側
（`glm.rs` 6,185、`metrics.rs` 4,933、`model_selection.rs` 4,292、
`linear_model.rs` 4,288、`decompose.rs` 3,918、`cluster.rs` 3,547）。
`linear_model.rs` だけ Verified OLS と未登録推定器が同居している。

`rng` は台帳外だが、`time-series-oracles` ジョブが `rng::tests` を実行する。
`tree` / `histgb` は品質アダプタで、核の検証は `oldwood` / `mayoi-no-mori` 側。

AGENTS.md §5 の `cosine_power` / `two_sided_power` / `Transformed<E>` は
`kanimiso/src/hmm/emission/` に無い（Gaussian / Poisson / Categorical のみ）。
PR 7 は未着手に見える。

---

## 4. クレート別のカバー状況

読み方: 「仕様」は文書の有無、「テスト」は層（§6）が揃っているか、
「穴」はこの監査で見えた欠け。

### 4.1 `signlred`

- 仕様: README。`Policy` の数値フィールドは D8 とテスト
  `numerical_fields_match_agents_d8` で固定。`docs/validation.md` なし。
- テスト: 契約の骨格（意味のなさで中止、条件数警告、NaN 走査、κ=∞）。
  各 `IssueCode` の発火条件を網羅してはいない。
- 穴: 品質契約の「仕様」が README とコードに分散。どのコードがどの
  `Failure` を返すかの表がない。

### 4.2 `ojizou-san`

- 仕様: README。ledger / compromise / IncrementalExplain の意味は短文のみ。
- テスト: 終端イベントの kind、compromise 記録。2 本。
- 穴: オンライン更新の説明フィールド、journal 永続化、ネストした Session の
  契約テストがほぼ無い。上位 crate のテストに埋没している。

### 4.3 `tsutsumi`

- 仕様: README の検証箇条書き。専用 validation 文書なし。特殊関数の詳細は
  ルート `docs/validation.md`。
- テスト: scipy 1,099 件 replay、分布恒等式、p 値 ∈ [0,1]、Nelder–Mead の
  反射 / 拡大 / 内外収縮 / shrink / 非有限 / 停止、求積モーメント、lstsq。
- 穴: `optimize.NelderMead` が台帳 Experimental のまま。linalg / quadrature は
  Verified 集合に名前が無い。`special_functions.json` はルートと
  `tsutsumi/` に二重配置（CI が `cmp`）。

### 4.4 `number-ruler`

- 仕様: README の範囲表 + validation（家族、許容差、Experimental 理由）。
- テスト: 外部 LM/GLM、Mixed ML/REML、GLMM 積分、加法縮約、SHAP 全 coalition、
  正規裾、失敗境界。14 本で fixture を回す。
- 穴: Mixed / Additive を Experimental に留めている基準（「ストレス不足」）が
  定量でない。`kanimiso::mixed` / `kanimiso::glm` は別実装で台帳外。

### 4.5 `oldwood`

- 仕様: 検証層、分岐契約、数値選択、sklearn 非対象が具体的。
- テスト: 36 本。外部 3 ケース + brute-force + 置換 / 重み / 隣接 float。
- 穴: 小さい。意図的に無いもの（欠損、カテゴリ、剪定）は文書と一致。

### 4.6 `mayoi-no-mori`

- 仕様: 推定器ごとの実装範囲と非互換、所有権、LightGBM/CatBoost 命名ポリシー。
- テスト: 70 本。閉形式・seed 再現・確率和・OOB・bin 単調・ordered statistic
  の leakage 不在。外部 ensemble fixture なし、と validation が明言。
- 穴: 文書とテストは揃っている。足りないのは外部 fixture であり、実装ではない。

### 4.7 `denshi` / `riko`

- 仕様: 短い validation。除外範囲（文脈付き、遅延、連続腕など）が明確。
- テスト: 各 4 本。閉形式 + 性質 + 失敗で状態が進まない。許容差コメントあり。
- 穴: ワークスペースメンバーなのに台帳・README 機能表・quality-small clippy から
  外れている。`kanimiso::bandit`（1,720 行、テスト 2）と概念が重なる。

### 4.8 `kanimiso`（統合面）

台帳内の核（online / tsa / state_space / hmm / filters / anomaly）は、
ルート validation の golden と CI ジョブが対応している。HMM は
`hmm-core` で Decimal check + `hmm::` 一括。ボラティリティは QML replay。

台帳外モジュールは公開 API と smoke / 性質テストを持つが、

- 独立オラクル fixture が無い
- `coverage::inventory()` に `Experimental` すら無い
- README は「台帳にないものを検証済みと扱わない」と書く一方、
  モジュールは `pub` のまま

という意味で、**仕様書のカバーもテストのカバーも「未登録の実装」に対して空**である。
`#[test]` があることと、検証契約の対象であることは別である。

### 4.9 `Isuzu`（メンバー外）

- 仕様: 文献付きアルゴリズムノート + [`deviations.md`](../Isuzu/docs/deviations.md)
  が「原文からどこを変えたか」の台帳。kanimiso の Tier 0–3 とは別体系。
  `Verified` という言葉を使わない。ベンチマークは速度・精度の実測。
- テスト: ユニットがファイルあたり 1–7 本のことが多く、統合は
  `audit_oracles` 12、`shreve_oracles` 9、`toy_recovery` 12、
  `yuima_surface` 12、`npbayes` 5。閉形式・既知ベクトルとの照合がある。
- 独自 CI: `Isuzu/.github/workflows/ci.yml`（test / clippy correctness / MSRV 1.85）。
  ルートの redundancy lint や llvm-cov には入らない。
- 穴:
  - カタログ（拡散・点過程・フィルタ・金融）に対して、文書の式ごとに
    対応テストがあるかの表が無い
  - `nonlinear.rs`（813 行）と `ssm.rs`（614 行）はユニットテスト 0
  - 独自 `optimize.rs` の Nelder–Mead、独自 Kalman は `tsutsumi` /
    `kanimiso::state_space` と並立（R6 の「1 核」は workspace 限定なら矛盾しない）
  - `rustfft` / `thiserror` は D4 の「ndarray と faer のみ」の外。
    メンバー外なので lint 対象外
  - QMLE の p 値は Abramowitz–Stegun の erfc 近似（deviations 記載）。
    kanimiso 側の修正済み `special` とは別経路

### 4.10 `amatsuki`

- 仕様: ChaCha8、分布、既知の逸脱（8 ラウンド、正弦捨て、など）。
- テスト: 21。ルート CI は core-only（`--no-default-features`）。
  分布サンプラーは default feature 側。
- 穴: ensemble が使うのは ChaCha8 ストリームなので、ルート CI の範囲は
  利用側と一致する。Isuzu 本体が使う分布 API の再生は Isuzu CI 依存。

### 4.11 `wormhole` / `coronel` / `jelly-wave`

- 仕様: POT 0.9.7.post1 との機能対応表。状態語は
  `implemented` / `partial` / `planned` / `other crate` / `not applicable`。
  kanimiso の Verified とは定義が違う。「公開 API と直接テストがある」が
  implemented である。
- テスト: `pot_oracles.rs` が POT 0.9.7 由来の回帰値を多数再生。
  `kanimiso_integration.rs` は固定リビジョンの `Matrix::inner()` 受け渡し。
  coronel / jelly-wave は各ファイル内テスト。
- 穴:
  - ルート workspace に含まれず、ルート CI が走らない
  - `planned` が表の大部分を占める（完了ゲートは planned/partial の解消）
  - 統合テストがピン留めする kanimiso revision の鮮度は未確認
  - 距離・カーネルの式が `kanimiso::metrics` / `neighbors` に残っていないかは
    この監査では grep しきっていない（疑問点）

---

## 5. Golden / オラクル資産

| fixture | ケース数（JSON 上） | 再生先 | ルート CI |
|---|---:|---|---|
| `golden/special_functions.json` | 1,099 | `tsutsumi::special` | `special-oracle` + `cmp` |
| `golden/ols.json` | 3 | `linear_model` | `ols-oracle` |
| `golden/state_space.json` | 4 | `state_space` | `state-space-oracle` |
| `golden/online_rls.json` | 3 | `online` | `online-rls-oracle` |
| `golden/arma_acov.json` | 6 | `tsa::arma` | `time-series-oracles` |
| `golden/process_mle.json` | 3 | `stats::process` | 同上 |
| `golden/garch_qml.json` | 6 | `tsa::garch` | `garch-oracle` |
| `golden/ewma_qml.json` | 5 | `tsa::ewma`（Experimental） | 同上 |
| `golden/egarch_qml.json` | 7 | `tsa::egarch`（Experimental） | 同上 |
| `golden/figarch_qml.json` | fixed 9 + qml 4 + invalid 10 | `tsa::figarch` | 同上 |
| `golden/fiegarch_qml.json` | 7 | `tsa::fiegarch` | 同上 |
| `golden/hmm.json` | 4 | `hmm`（Experimental） | `hmm-core` |
| `number-ruler/golden/regression.json` | cases 7 + normal_tails 11 | number-ruler | `regression-shared-kernels` |
| `oldwood/golden/sklearn_cart.csv` | 3 | oldwood | `tree-ensemble` |

Python 生成スクリプトは `scripts/*_oracle.py` と各 crate の `scripts/`。
実行時依存にはしない、という契約は守られている。

無いもの（文書が欲しいと言っている、または §6 で要求している）:

- hmmlearn / arch の fit 結果
- ensemble の sklearn / LightGBM / CatBoost fixture
- 未登録 kanimiso モジュール用の golden
- Isuzu / wormhole をルート CI で再生する経路
- `chi2_pvalue` の golden 行

---

## 6. テスト層（AGENTS.md §6）との対応

| 層 | よく覆っている面 | 薄い / 無い面 |
|---|---|---|
| Tier 0 外部オラクル | special、number-ruler LM/GLM/LMM、CART sklearn、フィルタの statsmodels ソース | HMM、ensemble、未登録 sklearn 風 API、Isuzu カタログ全体、wormhole をルートから |
| Tier 1 閉形式 / 全列挙 | HMM 小問題、ARMA、lfilter、Hedge/UCB、SHAP coalition、NM 既知解、Isuzu の一部 audit | `glm.rs` 等の巨大未登録面、Isuzu `nonlinear` / 多数のモデル |
| Tier 2 性質 | online 分割不変、CART 置換、ensemble 確率和、special 単調、bandit 正規化 | signlred のコード網羅、ojizou の説明契約 |
| Tier 3 ストレス | 状態空間の欠測、online 失敗の原子性、oldwood の極端 float、HMM log-space | 未登録モジュールは「落ちない」程度のことが多い（未確認: 全件は読んでいない） |

line coverage 76% は **未登録モジュールのテストも含む**。
Verified 核だけの被覆率はこの監査では出していない。

---

## 7. 仕様書側の穴（テスト以前）

1. `signlred` / `ojizou-san` / `tsutsumi` に `docs/validation.md` が無い。
   品質契約と共通核の証拠がルート文書と README に分散している。
2. `denshi` / `riko` は検証文書があるのに、台帳と利用者向け README に現れない。
3. 未登録 `kanimiso` モジュールに「残す / 落とす / Stub と書く」決定がない。
   `docs/dropped_v0_1.md` は生成 monolith 向けで、これら 3 万行超は対象外。
4. AGENTS.md §0 の現状表が古い。§5 の HMM ファイルツリー（cosine / TSP /
   Transformed）は未実装のまま残っている。
5. Isuzu は文献仕様が厚いが、kanimiso の Verified 定義に載せていない。
   二つの「正しさ」が並立している。
6. wormhole の `implemented` は POT 面の実装有無であり、許容差付きオラクル
   再生を含意しない（ただし `pot_oracles.rs` は存在する）。
7. `kanimiso::mixed` と `number_ruler::MixedModel`、`kanimiso::glm` と
   `number_ruler::GeneralizedLinearModel`、`kanimiso::bandit` と `riko` のように、
   近い概念の公開面が二重にある。R2 の確認欄が要る状態。

---

## 8. 疑問点

判断がなければ実装に進むな、という類の問い。

1. **未登録モジュール（約 5 万行）はアーカイブ対象か、Experimental 登録対象か、
   個別に検証して残す対象か。** D1 は「検証計画のない新規追加を止める」だが、
   既に `pub` でテストもある。中途半端な削除は AGENTS.md §10.9 が禁じている。
2. **`denshi` / `riko` を `coverage::inventory()` と README に載せるか。**
   閉じた核としては Verified 相当の証拠がある。載せないなら「台帳は
   kanimiso 再公開面だけ」と書いた方がよい。
3. **`quality-small` の clippy 対象に `denshi` / `riko` を入れるか。**
   意図的な段階適用か、漏れか。
4. **`optimize.NelderMead` を Verified にする障壁は何か。**
   分岐テストは揃っている。台帳と文書の更新忘れか、外部 solver 照合が
   まだ必要だと考えているのか。
5. **`tsa.EwmaVol` / `Egarch` は golden を CI 再生しているのに Experimental のまま。**
   Garch/Figarch/Fiegarch との差は何か。
6. **HMM を Experimental に留める条件は「hmmlearn golden が来るまで」で確定か。**
   Decimal 全経路は Tier 1 として強い。
7. **`chi2_pvalue` を golden に足して Verified にする作業は、仕様追加か
   実装か。** 関数は既にある。
8. **Isuzu / wormhole をルート workspace の検証契約に入れるか。**
   入れると D4（依存）と R6（単一核）に抵触しうる。入れないなら
   「同梱参照実装」と README に書いた方がよい。
9. **Isuzu の Nelder–Mead / Kalman / 特殊関数近似を `tsutsumi` に寄せるか。**
   寄せると Isuzu の deviations とベンチが変わる。寄せないと核が二つ。
10. **`kanimiso::metrics` の距離・カーネルは wormhole / coronel と重複するか。**
    重複なら PR 9 の対象。この監査では式の一致まで見ていない。
11. **line coverage ratchet は未登録コードを残す誘因になっていないか。**
    削除すると 76% を割る可能性がある（未確認）。
12. **`amatsuki` の `distributions` テストをルート CI に載せるか。**
    mayoi は core-only しか使わない。
13. **`filters.LocalLinearTrend` の Verified は状態空間 fixture の再利用だけで
    足りるか。** 専用ケースは少ない。
14. **`linear_model.rs` の未登録推定器（大量）を OLS とファイル分割するか。**
    R5 予算はファイル単位なので、Verified 核が見えにくい。
15. **wormhole の `kanimiso_integration` がピン留めする revision は
    いつ更新する契約か。**
16. **AGENTS.md §0 と §8 の PR 表を、現状の行数・テスト数・CI 有無に
    追記し直すか。** この文書は監査であり、方針文書の更新はしていない。
17. **利用者向け README の「CART を使いたい」と、未登録の `svm` / `neural` /
    `cluster` が同じ crate から見えることの許容範囲は何か。**
    検証契約は台帳準拠だが、cargo doc は全 `pub` を出す。

---

## 9. この文書が測っていないこと

- 各 `#[test]` の実行成否、実測誤差の再測定
- `cargo llvm-cov` の領域別内訳
- 未登録モジュール 130 テストの、性質テスト対 smoke の分類（全件は未読）
- wormhole の `implemented` 各行と `pot_oracles.rs` の 1:1 対応
- Isuzu カタログ全モデルと `docs/*.md` 見出しの 1:1 対応
- `kanimiso::metrics` と wormhole / coronel の式の同一性
- 公開項目 1,347（R4 予算）の、台帳 71 件以外の内訳の完全列挙

測ったのは「文書があるか」「台帳にあるか」「テスト属性と fixture があるか」
「CI ジョブが名前で掴んでいるか」までである。
