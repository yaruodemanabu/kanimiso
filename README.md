# kanimiso

Pure Rust の機械学習・統計クレート。外部の科学技術計算クレートは
`ndarray = 0.17.2` と `faer = 0.24.4` のみ。`unsafe` なし。

責務は分ける。N 次元配列・軸・view は `ndarray`、行列分解・固有値・線形方程式・
最小二乗は `faer` が担当する。`ndarray-linalg`、`ndarray-stats`、BLAS/LAPACK の
native backend は採用しない。

v0.2 は小さい検証済み核から再構成中である（`AGENTS.md`）。README の機能主張は
`kanimiso::coverage::verified()` に連動する。`0.2.0-alpha.1` までは
scikit-learn / statsmodels / sktime / tslearn / hmmlearn / river 相当という表記はしない。

## ワークスペース

| クレート | 役割 |
|---|---|
| **kanimiso** | 推定・変換・予測・予測子。計算そのもの。 |
| **signlred** | 機械学習・線形計算の**結果と推測の品質**に責任を持つエラー処理。特異・ランク落ち・分離・定数ターゲット・識別不能・意味のないフィットをコード化して abort / 警告する。 |
| **ojizou-san** | 同上の品質に責任を持つログ。数値計算上の妥協（意図した計算 vs 実際に走った計算）、意味のないフィット、追加学習の説明（何が動いたか・なぜ・識別は残っているか）を台帳に残す。 |

普通の `thiserror` / `log` ではない。計算結果を捨ててよい数字として返すことを拒否する。

## 品質契約

1. `fit` / `predict` / `partial_fit` は必ず `signlred::Qualified<T>`（値 + `Report`）または `Failure` を返す。
2. 数値的妥協（擬似逆、リッジフォールバック、ジッタ、打ち切り SVD）は `NumericalCompromise` として記録される。
3. 統計的に空の計算（定数ターゲット、ランク 0、単一クラス分類器、補間だけの \(R^2=1\)）は `Meaninglessness` を付け、Vacuous / False なら abort する。
4. 追加学習は `ojizou_san::IncrementalExplain` なしでは完了しない。`n_eff`、`‖Δθ‖`、情報利得、識別の可否、warmup、文章による説明が必須。

## 検証済み表面（`verified()`）

現在 `CoverageStatus::Verified` なのは `linear_model::LinearRegression`、`online::LinearRegression` / `OnlineWeightedMean` / `OnlineEwMean` / `OnlineEwVar` / `OnlineMean` / `OnlineVar` / `OnlineCovariance` / `OnlineSum` / `OnlineCount` / `OnlineAutoCorr` / `OnlineVarianceThreshold`、`special.rs` の特殊関数、`state_space::LinearGaussianStateSpace`、`filters::LocalLinearTrend` / `lfilter` / `convolution_filter` / `recursive_filter` / `miso_lfilter` / `bk_filter` / `cf_filter`、`tsa::arma_acf` / `arma_acovf` / `arma2ma` / `arma2ar` / `arma_impulse_response`、`tsa::Garch11`、固定次数 `tsa::Figarch` / `tsa::Fiegarch`、`stats::process_mle` である。OLS は標準ライブラリのみの80桁 Decimal オラクル（`golden/ols.json`）で、切片あり・切片なし・2説明変数の高レバレッジケースを検査する。係数だけでなく、数値ランクに基づく自由度、SVD 共分散、中心化／非中心化 (R^2)、MLE 対数尤度、AIC/BIC、t/F 検定、hat 対角、((1-h)^2) を用いる Cook 距離まで再生し、120/180桁照合と射影恒等式を CI で固定する。特殊関数は scipy 1.18.1 ゴールデン（`golden/special_functions.json`、1,099 ケース）を `cargo test -p kanimiso --lib special::` でリプレイする。GARCH(1,1) は独立した80桁 Decimal オラクル（`golden/garch_qml.json`、内部解・α=0・β=0・高持続・識別不能・extended-real overflow の6ケース）に加え、120桁での再生成照合、解析勾配、スケール同変性、再帰式、境界KKTを検査する。FIGARCH(1,d,1) は Baillie--Bollerslev--Mikkelsen の係数再帰と有限 `K` 打切りを、標準ライブラリのみの80桁 Decimal オラクル（`golden/figarch_qml.json`、9固定・10不正・4 QMLE ケース、全16境界面）で検査し、120桁再生成、独立した係数経路、予測、解析勾配、反射・尺度・打切り不変量も照合する。FIEGARCH は Bollerslev--Mikkelsen (1996) Eq. 11 の one-AR/no-news-MA 特殊化として、有限 `K` を明示した標準ライブラリのみの80桁 Decimal オラクル（`golden/fiegarch_qml.json`、7ケース）を固定パラメータとQMLEの両方で再生する。120桁再生成、非停留点での全5勾配、厳密な `d=0` 面、負の `beta`、`d` の 1/2 近傍、extended-real overflow、反射・尺度・打切り不変量も検査対象である。

状態空間核は `golden/state_space.json` の80桁 Decimal・同時正規分布のブロック条件付けを独立オラクルとし、スカラー閉形式、局所線形トレンド、相関した多変量観測、部分／全欠測、非ゼロ offset を検査する。`t=0` prior、Joseph 共分散更新、正確な Gaussian innovation 尤度、RTS 平滑化を120/180桁で再照合し、Rust replay の実測最大絶対誤差 `3.553e-15` に対して許容差 `1.5e-14` を用いる。末尾欠測と交差共分散ゼロでは不要な RTS 逆行列を作らず、共分散の対称化は `f64::MAX` と最小サブノーマルでも中間 overflow/underflow を起こさない。

`stats::process_mle` は指数共分散を持つ Gaussian process の GLS profile likelihood を、5個の決定論的な median-gap 倍率で探索する。標準ライブラリのみの80桁 Decimal dense-GLS オラクル（`golden/process_mle.json`）を120/180桁で再生成し、パラメータ、profile likelihood、並べ替え不変性、応答の affine 変換を照合する。120/180桁差は `4.63e-117`、Rust replay の最大パラメータ誤差は `1.78e-15`、最大 profile 誤差は `1.43e-14` である。非収束を成功に偽装せず、`DidNotConverge` と数値的妥協を report に残す。

オンラインRLSは `golden/online_rls.json` の80桁 Decimal による幾何重み付き一括正規方程式を、逐次Kalman-gain更新とは独立なオラクルとする。切片なし、2変数と切片、`λ` が1に極めて近い場合を120/180桁で再照合し、係数、逆Gram、予測、有効標本数を検査する。Rust replay の実測最大絶対誤差 `6.218e-15` と最大相対誤差 `2.616e-14` に対し、それぞれ約4倍の許容差を用いる。無効入力、特徴空間変更、途中overflowは推定器を一切変更しない。

`OnlineWeightedMean` は正の有限重みを必須とし、最大重みで正規化した逐次式で巨大重みの総和overflowを避ける。手計算した加重平均と Kish 有効標本数、重み尺度不変性、bitwise な分割不変性、失敗時の状態不変性を検査する。`OnlineEwMean` / `OnlineEwVar` は有限系列の正規化指数重みと Kish 有効標本数をそのまま逐次更新し、閉形式との最大絶対誤差 `1.78e-15`、bitwise な分割不変性、微小 `alpha` の underflow、`alpha=1` の極値置換、全失敗経路の状態不変性を検査する。ARMA ACF/ACOVF は固定長 MA(∞) 打切りを使わず、有限 Yule–Walker 系と後続漸化式で計算する。別実装の80桁 Decimal 状態空間 Lyapunov オラクル（`golden/arma_acov.json`、6ケース）を120/180桁で再生成し、近単位根を含む全点で照合する。

`OnlineMean` / `OnlineVar` / `OnlineCovariance` / `OnlineSum` / `OnlineCount` は共通 preflight と候補状態の一括 commit を使い、空・非有限・shape 不整合・カウンタまたは算術の overflow/underflow で状態を変えない。平均は両符号の `f64::MAX` に耐える凸結合、和・M2・交差モーメントは共通の補償加算で更新する。整数スケールの厳密有理数オラクルに対する最大誤差は平均 `4.44e-16`、標本分散 `3.55e-15`、和と標本共分散は 0 で、bitwise なバッチ分割不変性も固定している。

`OnlineAutoCorr` は同じ transactional preflight と補償付き Welford pair moment を使い、バッチ境界の `(previous_last, current_first)` を一度だけ数える。3観測未満または定数系列は `NaN` とし、floor/clamp や非有限値の黙示スキップは行わない。整数の厳密有理数オラクルに対する誤差は `6.94e-18`、極端スケール同変性の誤差は 0 で、内部状態の bitwise なバッチ分割不変性も固定している。

`OnlineVarianceThreshold` は列ごとの標本分散を同じ補償付き Welford 核で更新し、`variance > threshold` の列だけを保持する。空バッチ、非有限入力、列数変更、無効な閾値、カウンタや算術の overflow/underflow は全列の候補状態を commit する前に失敗する。3列の厳密有理数オラクル、bitwise なバッチ分割不変性、定数列、`f64::MAX` を含む極端スケールで実測誤差 0 を固定している。

`filters::lfilter` は定義どおりの SISO 差分方程式を評価し、ゼロの先頭分母係数や非有限入力、係数正規化・積・累積の underflow/overflow を値へ置換せず失敗として返す。一次 IIR の閉形式インパルス応答、全時点の再帰残差、極小値を含む係数の共通尺度不変性で検証する。FIR の `convolution_filter` と IIR の `recursive_filter` は独自ループを持たず、この核へ委譲する。`miso_lfilter` も各入力チャネルを同じ核で処理し、チャネル和の overflow を失敗として報告する。

`filters::bk_filter` は statsmodels と同じゼロ和の対称係数を共通 `lfilter` で valid 畳み込みし、`K=0`、`2K+1` の整数 overflow、短系列を値へ置換せず失敗にする。`filters::cf_filter` は endpoint-to-endpoint drift と Christiano–Fitzgerald の非対称 A/B endpoint 係数を実装し、以前の OLS fallback を除去した。statsmodels 0.15.1 source oracle に対する最大絶対誤差は BK `8.88e-16`、CF `1.78e-15` である。

内部乱数核は Pure Rust の xorshift64* を用い、非ゼロ seed を潰さない。整数範囲抽選は周期 `2^64-1` に合わせた棄却法、実数区間抽選は幅 `hi-lo` が overflow する有限端点でも凸結合で半開区間を保ち、Poisson は小率の反転法と大率の Hörmann PTRS を使う。剰余クラスの全数列挙、隣接 `f64` 端点と `[-MAX, MAX)`、Poisson の平均・分散・第3中心モーメントを CI で検査する。既存の infallible API では入力エラーを返せないため、無効な区間、および負・非有限・`2^32` 超の Poisson 率は乱数状態を消費せず panic する。

`tsa::arma2ma` / `arma2ar` は単一の補償和による比多項式展開を共有し、既存の Schur 判定で AR 因果性または MA 可逆性を検査する。閉形式との最大誤差 `6.94e-18` と、順変換・逆変換の畳み込みが単位系列になる性質（最大誤差 `1.39e-17`）を固定している。生成は同じ有限性・因果性検査を使うが、正規乱数分布そのものの独立オラクルが未整備なので `arma_generate_sample` は Experimental のままである。

生成型（Cosine/Tsp 冪族など）は削除対象であり、active な検証台帳には `Generated` / `Stub` を1件も残さない。旧台帳5,389件のうち、HMM・tslearn・online の archive 化で実装から消えた4,241名を hash 付きの [`generated-v0.1-archive/coverage.rs.txt`](generated-v0.1-archive/coverage.rs.txt) に保存し、active 台帳は実在する51項目（Verified 42、Experimental 9）だけに縮めた。テスト4件しかなく Verified 項目も workspace 内利用もなかった旧 `tslearn.rs`（60,208行・公開宣言1,581件）は、hash 付きの [`generated-v0.1-archive/tslearn.rs.txt`](generated-v0.1-archive/tslearn.rs.txt) へ退避し、コンパイル対象から外した。旧 `online.rs`（78,273行・直接 `pub` 1,241件）も [`generated-v0.1-archive/online.rs.txt`](generated-v0.1-archive/online.rs.txt) に byte-identical で保存し、公開構造体11個の小さい実装（テストを除き3,183行）へ切り替えた。旧 `LogMinkowski*Anomaly` 群と、固定距離別の `KnnAnomaly` / `ManhattanAnomaly` / `MinkowskiAnomaly` / `LinfAnomaly` / `Log*Anomaly` alias は、距離指数を実行時パラメータに持つ単一の experimental `anomaly::KnnDistanceAnomaly` に統合済みであり、verified ではない。移行時は `KnnDistanceAnomaly::new(k, p, log_transform, window)` を使う。`log_transform = true` は旧実装の `ln(max(|x|, ε))` ではなく、ゼロで有限かつ符号を保つ `sign(x) * ln(1 + |x|)` である。

HMM の公開面は、単一の experimental `hmm::HiddenMarkovModel<E>` と `GaussianEmission` / `CategoricalEmission` / `PoissonEmission` に統合した。旧 569,177 行・直接 `pub` 宣言 12,401 件の生成モノリスはコンパイル対象から外し、SHA-256 を固定した [`generated-v0.1-archive/hmm.rs.txt`](generated-v0.1-archive/hmm.rs.txt) として保存している。新しい `hmm/` は 3,008 行・直接 `pub` 宣言 25 件で、追加スタック環境変数を必要としない。`HiddenMarkovModel::new(initial, transition, emissions, max_iter, left_right, policy)` は初期分布、遷移行列、状態ごとの初期 emission を明示的に受け取り、黙って正規化しない。左–右制約は別型ではなく実行時の `left_right` で指定する。Categorical / Poisson 観測は1列の有限な非負整数だけを受理し、範囲外 support は `-∞` とする。密度・確率への floor は適用しない。

全モデルが共有する forward–backward 核は正規化 log-space の1実装、Viterbi は動的計画法の1実装である。filtering 確率は各 prefix を再計算せず、正規化済み forward 行から1度だけ生成するため `O(T·K²)` である。有限な極端対数放射を確率 0 と誤認せず、K=2・T=3 の全経路列挙、posterior marginal、時刻別シフト不変性、到達不能状態、真のゼロ尤度を直接テストする。さらに標準ライブラリだけの 80/120/180 桁 Decimal 全経路オラクル（`golden/hmm.json`）で Gaussian / Categorical / Poisson の尤度と Viterbi、および Gaussian Baum–Welch 2反復後の全パラメータを再生する。120/180桁差は `6.53e-119`、Rust の最大誤差は Poisson 尤度の `1.60e-14` である。HMM 33テストは通常スタックで通る。ただしこれは hmmlearn 由来ではないため、generic HMM 全体はまだ experimental とし、出自の異なる相互運用ゴールデンを追加するまでは型ごとの旧 smoke test を検証済みの根拠にしない。

微分不要最適化は単一の experimental `optimize::NelderMead` に集約を開始した。argmin 0.11 を参照実装として、反射・拡大・外側収縮・内側収縮・shrink の分岐を決定論的トレースで照合するが、argmin 自体には依存しない。収束には目的値のばらつきと全単体直径の両方を要求し、反復上限・単体崩壊・非有限目的値を品質報告から隠さない。最初の利用箇所である `tsa::EwmaVol` は旧コピー座標探索を廃止し、実行時設定の bounded grid で basin を選んでから共通 solver で精密化する。QML は残差を最大絶対値で正規化してλ推定をスケール不変に保ち、分散 floor は用いない。80桁 Decimal と解析的導関数で `golden/ewma_qml.json` を生成・照合し、定数系列は `UnidentifiedModel`、端点解は `ParameterAtBoundary` として表面化する。

破壊的変更の移行先は次のとおり。

| 削除した公開型 | 移行先 |
|---|---|
| `GaussianHmm` / `GaussianHmmLeftRight` | `HiddenMarkovModel<GaussianEmission>`。左–右制約は `left_right = true` |
| `MultinomialHmm` / `CategoricalHmm` / `MultinomialHmmLeftRight` | `HiddenMarkovModel<CategoricalEmission>` |
| `PoissonHmm` / `PoissonHmmLeftRight` | `HiddenMarkovModel<PoissonEmission>` |
| `HmmAnnotator` | `HiddenMarkovModel<GaussianEmission>::fit` 後に `decode` |

## 使う

```rust
use kanimiso::data::{Matrix, Vector};
use kanimiso::linear_model::LinearRegression;
use kanimiso::traits::Fit;
use kanimiso::log::Session;

let x = Matrix::from_fn(20, 1, |i, _| i as f64);
let y = Vector::from_iter((0..20).map(|i| 1.0 + 2.0 * i as f64));
let session = Session::new("ols", "fit");
let fitted = LinearRegression::new().fit(&x, &y, &session)?;
// fitted.value.coef / intercept / r2 / se / p_values
// fitted.report を読まずに係数を論文に書かないこと
// session.ledger() に妥協と警告が残る
```

オンライン:

```rust
use kanimiso::online::LinearRegression as Rls;
use kanimiso::traits::PartialFit;

let mut rls = Rls::new(1.0); // forgetting λ
let q = rls.partial_fit(&x, Some(&y), &session)?;
// q.value.narrative / trust / quality.effective_sample_size
```

実装一覧は `kanimiso::coverage::inventory()`。検証済みだけを見るなら `verified()`。

## 制約

- **a.** 外部の科学技術計算依存は `ndarray = 0.17.2` と `faer = 0.24.4` のみ（版・feature・native BLAS 不在を `scripts/lint_dependencies.py` で固定）
- **b.** Pure Rust
- **c.** エラーは `signlred`、ログは `ojizou-san`。どちらも計算品質の責任を持つ
- **d.** 追加学習アルゴリズムは説明力を落とさない（`IncrementalExplain` 必須）

## ライセンス

Apache-2.0
