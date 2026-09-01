# Isuzu

Pure Rust だけの確率過程ライブラリです。R の [YUIMA](https://cran.r-project.org/package=yuima) がやるシミュレーション・準最尤・Hayashi–Yoshida・lead-lag に加え、YUIMA にない **HFT ミクロ構造、確率最適制御、Malliavin 重み、CARMA-Hawkes 専用推定、点過程の拡張カタログ、線形 / 非線形カルマン、粒子フィルタ、ベータ・ベルヌーイ過程とノンパラメトリックベイズ** を、scikit-learn 風の `Estimator` / `Dataset` / `Pipeline` で束ねます。

**依存はすべて Pure Rust です。** BLAS / LAPACK / GSL は使いません。`#![forbid(unsafe_code)]`。`src/audit.rs` が `Cargo.lock` を検査します。

乱数は付属クレート **amatsuki**、線形代数は **faer**、分数ブラウンの循環埋め込み FFT は **rustfft**、誤差型は **thiserror** です。`rand` / `rand_chacha` / `rand_distr` / `nalgebra` には依存しません。

この README の数式は **Obsidian** 向けです（`$` インライン、`$$` ディスプレイ）。ノート全体・文献・オリジナルからの進む箇所・ベンチマークは [`docs/`](docs/Home.md) を vault として開いてください。

## カタログ

**拡散 / ジャンプ:** GBM, OU / Vasicek, CIR（非心 $\chi^2$ 厳密）、Hull–White, Black–Karasinski, CKLS / CEV, Jacobi, Bessel, Brownian bridge, Heston（Andersen QE）, Stein–Stein, 3/2, SABR, Bates, Merton, Kou, fractional GBM / OU, CARMA, COGARCH。

**エネルギー / リアルオプション:** Schwartz 1-factor, Schwartz–Smith, Lucia–Schwartz, Cartea–Figueroa, Gibson–Schwartz, レジームスイッチ拡散, スパークスプレッド / Margrabe, Dixit–Pindyck / McDonald–Siegel, CRR スイング / ストレージ。

**金融（`src/finance/`）:** ペイオフ, Black–Scholes, MC（対偶 / 制御変量 / 重要度）, CRR / 3 項, BS PDE / PSOR, LSM（Hermite + 双対）, Vasicek / CIR 債, Hull–White $\theta(t)$, $T$-フォワード, Merton 級数 / PIDE / MC, Kou, バリアのブラウン橋補正。

**Lévy 過程:** 複合ポアソン, VG, NIG, 安定, `levy_path`。

**点過程:** 斉次 / 非斉次ポアソン, 指数 Hawkes, **N 次元指数 Hawkes（解析勾配 + Ogata KS）**, 2 次元 Hawkes, べき乗 Hawkes, 自己補正, Weibull / Gamma 更新, CIR Cox, marked Hawkes, 抑制 Hawkes, **CARMA-Hawkes + 専用 MLE**。

**高頻度 / HFT:** Hayashi–Yoshida, lead-lag, `hyavar`, BNS（四乗パワー比）, Lee–Mykland, 先行平均 HY, realized kernel, two-scale RV, previous-tick / refresh time, Roll spread, ACD(1,1), Almgren–Chriss, Kyle λ, Hawkes LOB, OFI, mid / microprice。

**制御:** Merton / Kelly, 線形二次レギュレータ, 1 次元 HJB（陽 / 陰）と Kushner–Dupuis, TWAP。

**Malliavin:** 第一変分と Fournié 重み, Δ / Γ, 密度, 特性関数, モーメント / 歪度 / 尖度, 小ノイズ展開 $(d_0, d_1)$。

**フィルタ:** 離散カルマン, Kalman–Bucy, RTS / 拡張 RTS, 情報フィルタ, 平方根カルマン, Sage–Husa 適応カルマン, EKF / IEKF / 二次 EKF, UKF, CKF, EnKF, ガウス和, 連続–離散 EKF, SIS / SIR, APF, RPF, UPF, FFBSi, **`ParticleModel` / PMMH / SMC² / 粒子 Gibbs**, KF MLE と Shumway–Stoffer EM。

**ノンパラメトリックベイズ:** Dirichlet / Pitman–Yor（stick-breaking, CRP）、Beta–Bernoulli / IBP、HDP CRF、Neal Alg. 3、線形ガウス IBP Gibbs。

**推定の追加:** L-BFGS-B, `Fit` の vcov / SE, Uchida CIC, 二段階 / 閾値 / Kessler QMLE, QGV / MM Hurst, COGARCH GMM, Sobol + Brownian-bridge QMC。

**sklearn 風:** `QmleEstimator`, `recover`, `Pipeline`, `time_series_folds`, `datasets::make_*`。

## トイデータからの推定

```rust
use isuzu::api::{recover, QmleEstimator};
use isuzu::datasets::make_ou;

let toy = make_ou(1.3, 0.0, 0.4, 10.0, 2000, 7)?;
let mut est = QmleEstimator::new(toy.model.clone(), vec![0.8, 0.0, 0.6]);
let report = recover(&mut est, &toy.path, &toy.truth)?;
```

線形 / 非線形カルマンと粒子フィルタ:

```rust
use isuzu::filter::{
    kalman, particle_filter, unscented_kalman, LinearGaussian, ParticleConfig, UkfParams,
};
use isuzu::prelude::*;

let model = LinearGaussian::new(
    Mat::from_fn(1, 1, |_, _| 0.9),
    Mat::from_fn(1, 1, |_, _| 0.04),
    Mat::from_fn(1, 1, |_, _| 1.0),
    Mat::from_fn(1, 1, |_, _| 0.16),
)?;
let kf = kalman(&model, &obs, &x0, &p0)?;
let ukf = unscented_kalman(&model, &obs, &x0, &p0, UkfParams::default())?;
let mut rng = seed_rng(1);
let pf = particle_filter(&model, &obs, &x0, &p0, ParticleConfig::default(), &mut rng)?;
```

```bash
cargo test
cargo run --example sklearn_ou
cargo run --example carma_hawkes
cargo run --example malliavin_greeks
cargo run --example ou_qmle
cargo run --example async_cov
cargo run --example npbayes
cargo run --release --example benchmark
```

---

## クレート分担（数式処理の入口）

| クレート | 役割 | 数式処理の中身 |
| --- | --- | --- |
| **amatsuki**（workspace メンバ） | 乱数のアルゴリズムとインターフェース | ChaCha8（`set_stream` / `jump_ahead`）、SplitMix64、一様 / 正規（Box–Muller と Ziggurat）/ 指数 / ガンマ / IG / 安定 / ポアソン / Student-t / Beta / Dirichlet / 離散 |
| **faer** 0.24（`std` のみ、BLAS なし） | 密行列 | Cholesky (`llt`)、部分ピボット LU 逆行列、固有値、行列積 |
| **rustfft** 6 | 実 FFT | Davies–Harte / Wood–Chan の循環埋め込み（分数ガウスノイズ）だけ |
| **thiserror** 2 | エラー型 | 数式なし |
| **isuzu** | シミュレーション・推定・フィルタ・金融・エネルギー・HFT・制御・NP ベイズ | 下記の全ロジック。乱数は必ず `amatsuki::Rng` |

`src/audit.rs` は `Cargo.lock` を読み、`nalgebra` / `rand` / `rand_chacha` / `rand_distr` と BLAS / LAPACK / OpenSSL 系クレートが混入していないことをテストで強制します。

---

## amatsuki — 乱数生成の全ロジック

インターフェースは `Rng` / `SeedableRng` / `Distribution` です。`rand` クレートは使いません。

### ビット生成: ChaCha8 + SplitMix64

> [!info] 文献
> Bernstein (2008) ChaCha。シード展開は Steele, Lea & Flood (2014) SplitMix64。

- `seed_rng(seed: u64)` → `ChaCha8Rng`。
- $u64$ シードは **SplitMix64**（定数 $\mathtt{0x9E3779B97F4A7C15}$, $\mathtt{0xBF58476D1CE4E5B9}$, $\mathtt{0x94D049BB133111EB}$）で 256-bit 鍵に展開し、nonce $=0$。
- ストリームは **ChaCha（8 ラウンド = 4 double-round）**。定数 `"expand 32-byte k"`、64-bit カウンタ（ワード 12–13）。
- `next_f64`: 上位 53 bit を $[0,1)$ に写す IEEE-754 慣習 $(u\gg 11)/2^{53}$。

> [!warning] オリジナルからの進む
> ChaCha20 ではなく 8 ラウンド。SplitMix の split は実装していない。

### 分布サンプラー（すべて ChaCha8 の一様ビットから構成）

| 型 / 関数 | 法則 | アルゴリズム | 文献 |
| --- | --- | --- | --- |
| `Open01` | $(0,1)$ | 53-bit 中点 $(k+\tfrac12)/2^{53}$ | 実装規約 |
| `OpenClosed01` | $(0,1]$ | rand 0.8 互換: $(u_{53}+1)/2^{53}$ | 補った |
| `Uniform[low, high)` | 連続一様 | $\mathrm{low}+\mathrm{span}\cdot U[0,1)$ | — |
| `StandardNormal` | $N(0,1)$ | **Box–Muller**（$R=\sqrt{-2\ln U}$, $\Theta=2\pi V$ の余弦だけ。スレッドローカルキャッシュなし） | Box & Muller (1958) |
| `StandardNormalZiggurat` | $N(0,1)$ | Marsaglia–Tsang 128 箱。シード互換は `StandardNormal` のまま | Marsaglia & Tsang (2000) |
| `Bernoulli` / `Binomial` / `Categorical` / `Multinomial` | 離散 | 逆変換。大きい $n$ の二項は BTPE 風 | — |
| `set_stream` / `jump_ahead` | ChaCha ストリーム | nonce を切替、ブロック単位のカウンタ送り | Bernstein (2008) |
| `Normal(μ,σ)` | $N(\mu,\sigma^2)$ | $\mu+\sigma Z$ | — |
| `Exp1` | 標準指数 | $-\ln U$, $U\sim(0,1]$ | — |
| `Exp(λ)` | 指数（レート） | $\mathrm{Exp1}/\lambda$ | — |
| `sample_gamma` | Gamma（尺度） | **Marsaglia–Tsang**。$\mathrm{shape}<1$ は $G_{\alpha+1}\cdot U^{1/\alpha}$ | Marsaglia & Tsang (2000) |
| `sample_inverse_gaussian` | IG / Wald | **Michael–Schucany–Haas** | Michael, Schucany & Haas (1976) |
| `StudentT(ν)` | $t$ | $Z/\sqrt{\chi^2_\nu/\nu}$, $\chi^2_\nu=\mathrm{Gamma}(\nu/2,2)$ | — |
| `Poisson(λ)` | ポアソン | $\lambda<30$: **Knuth 逆変換**。$\lambda\ge 30$: **Hörmann PTRS**（Stirling で $\ln n!$） | Knuth TAOCP 2; Hörmann (1993) |
| `sample_stable_cms` | 標準 $\alpha$-安定 | **Chambers–Mallows–Stuck**（$\alpha=1$ は Cauchy 形） | Chambers, Mallows & Stuck (1976) |
| `Beta(α,β)` | $(0,1)$ | $X/(X+Y)$, $X\sim\mathrm{Gamma}(\alpha,1)$, $Y\sim\mathrm{Gamma}(\beta,1)$ | Devroye (1986) |
| `sample_dirichlet` | 単体 | $X_i\sim\mathrm{Gamma}(\alpha_i,1)$ を正規化 | Ferguson (1973) 有限次元 |

Isuzu 側の乱数はすべてこのストリームです。`seed_rng` は `isuzu::rng` が `amatsuki` を再エクスポートしています。

---

## Isuzu — 線形代数（`src/linalg.rs`、実装は faer）

> [!info] 文献
> Golub & Van Loan（Cholesky / LU）。Higham (2008) scaling-and-squaring（ただし Padé ではなく Taylor 27 項）。

| 操作 | アルゴリズム | クレート |
| --- | --- | --- |
| `cholesky` / `solve_spd` / `logdet_spd` | 下三角 Cholesky。$\log\det=2\sum\ln L_{ii}$ | faer `llt` |
| `try_inverse` | 部分ピボット LU。非有限または $A\hat A\not\approx I$ なら `None` | faer `partial_piv_lu` |
| `expm` | scaling-and-squaring + Taylor（最大 27 項） | 自前（faer の積だけ） |
| `van_loan_discretize` | $\exp\begin{pmatrix}-A & GG^\top\\0 & A^\top\end{pmatrix}\Delta t$ から $F,u,Q$ | Van Loan (1978) |
| `spd_regularize` | フィルタ共分散に $\varepsilon I$（QL では使わない） | 文献に無い保護 |
| `spectral_radius` / `is_hurwitz` | 複素固有値。Hurwitz は $\mathrm{Re}\,\lambda<0$ | faer `eigenvalues` |
| `gram_rowmajor` | $\Sigma=\sigma\sigma^\top$ | faer 積 |

固有値は CARMA / CARMA-Hawkes の因果性（同伴行列が Hurwitz か）に使います。

---

## 最適化（`src/optimize.rs`）

> [!info] 文献
> Nelder & Mead (1965)。1 次元は Kiefer (1953) 黄金分割。

推定の数値最小化はすべてここです。外部ソルバはありません。

- **Nelder–Mead**（箱制約は座標クリップ）。反射 $\alpha=1$、拡大 $\gamma=2$、収縮 $\rho=0.5$、縮小 $\sigma=0.5$。初期単体は各座標に相対 5% ステップ。QMLE / LSE / LASSO / Hawkes MLE / CARMA-Hawkes MLE / ACD MLE がこれを呼びます。
- **黄金分割探索**。1 次元、$\varphi=(1+\sqrt{5})/2$。

> [!warning] オリジナルからの進む
> 箱はクリップ。収束判定は自前の `ftol` / `xtol`。

---

## シミュレーション（`src/simulate.rs`, `src/scheme.rs`, `src/noise.rs`）

モデルは Itô 過程

$$
dX_t = a(t, X_t)\,dt + \sigma(t, X_t)\,dW_t + \gamma(t, X_{t-})\,dJ_t
$$

です（`src/model.rs` の `Sde`）。分数ブラウンは `hurst()`、乗法ジャンプは Merton/Kou 形 $\Delta X=\gamma(e^{\Delta L}-1)$。

### 離散化スキーム

> [!info] 文献
> Maruyama (1955); Milstein (1974); Kloeden & Platen (1992)。

| `Scheme` | 更新 | 備考 |
| --- | --- | --- |
| Euler–Maruyama | $x\leftarrow x+a\Delta t+\sigma\Delta W$ | 既定 |
| Milstein | 可換ノイズの Lévy 面積 $\frac12(\Delta W_j\Delta W_k-\delta_{jk}\Delta t)$。Jacobian が無ければ拡散の前進差分 | 強次数 1 |
| Kloeden–Platen 1.5 | スカラーのみ。空間–時間 Lévy 面積 $\Delta Z=\frac12\Delta t(\Delta W+\sqrt{\Delta t/3}\,Z)$。係数微分は中心 / 前進差分 | |
| Exact | 型付きヘルパだけ | GBM: 対数正規。OU: 下記のガウス遷移 |

OU の厳密ステップ:

$$
X_{t+\Delta t}=\theta+(X_t-\theta)e^{-\kappa\Delta t}+\sqrt{\frac{\sigma^2(1-e^{-2\kappa\Delta t})}{2\kappa}}\,Z.
$$

GBM の厳密ステップ:

$$
X\leftarrow X\exp\bigl((\mu-\tfrac12\sigma^2)\Delta t+\sigma\sqrt{\Delta t}\,Z\bigr).
$$

COGARCH は Euler + Lévy 増分で分散状態を更新します（Klüppelberg et al. 2004 の厳密離散ではない）。

### 駆動ノイズ（乱数はすべて amatsuki）

| 関数 | 法則 | アルゴリズム |
| --- | --- | --- |
| `brownian_increments` | $\Delta W\sim N(0,\Delta t)$ | $\sqrt{\Delta t}\cdot N(0,1)$（Box–Muller） |
| `correlated_brownian_increments` | 相関ブラウン | faer Cholesky $L$ で $LZ$ |
| `fractional_gaussian_noise` | 単位分散 fGn | 自己共分散 $\gamma(k)=\frac12(\|k+1\|^{2H}-2\|k\|^{2H}+\|k-1\|^{2H})$ の **Davies–Harte / Wood–Chan 循環埋め込み**。固有値は **rustfft** の実 FFT。スペクトル上で独立 $N(0,1)$ を乗せて逆 FFT |
| `fractional_brownian_increments` | 正則格子上の fBM 増分 | $\Delta t^H\cdot\mathrm{fGn}$ |
| `LevyMeasure::increment` | 複合ポアソン / ガウス / VG / NIG / 安定 | 複合ポアソン: $N\sim\mathrm{Poisson}(\lambda\Delta t)$ + `JumpLaw`。VG: $G\sim\mathrm{Gamma}(\Delta t/\nu,\nu)$, $\theta G+\sigma\sqrt{G}Z$。NIG: IG 従属 $\mathrm{IG}(\delta\Delta t/\gamma,(\delta\Delta t)^2)$ + $\beta G+\sqrt{G}Z$。安定: CMS を $\sigma\Delta t^{1/\alpha}$ で尺度 |
| `JumpLaw` | 正規 / Kou 二重指数 / 定数 / $t$ / Laplace | Kou: $U<p$ なら $+\mathrm{Exp}/\eta_+$、さもなくば $-\mathrm{Exp}/\eta_-$。Laplace: $\mathrm{scale}(E_1-E_2)$ |
| `poisson_arrivals` | 斉次ポアソン到着 | 指数待ち時間 $t\leftarrow t+E/\lambda$ |
| `poisson_random_sampling` | 間引き観測 | 各点を確率 `rate` で残す（`OpenClosed01`） |

fGn の文献は Davies & Harte (1987)、Wood & Chan (1994)。負の循環固有値はクリップする（原文の非負前提を数値的に補った）。

---

## 推定（`src/infer.rs` と点過程専用 MLE）

乱数を使う推定は **適応ベイズ** だけです（提案は `StandardNormal`、採択は `OpenClosed01`）。点推定は決定論的な準尤度 + Nelder–Mead です。

### 拡散の準最尤・最小二乗

> [!info] 文献
> Yoshida (1992); Kessler (1997)。コントラストは YUIMA `qmle` と同じ Euler ガウス準尤度。最適化は Nelder–Mead であり論文の Newton ではない。

$$
\mathrm{QL}(\theta)=\sum_i -\frac12\Bigl(n\ln 2\pi+\ln\det\Sigma_i+(\Delta X_i-a_i\Delta t)^\top\Sigma_i^{-1}(\Delta X_i-a_i\Delta t)\Bigr),
$$

$$
\Sigma_i=\sigma\sigma^\top(t_i,X_i)\,\Delta t_i
\quad\text{（faer Cholesky で }\log\det\text{ / 二次形式）}.
$$

- `qmle`: $-\mathrm{QL}$ を Nelder–Mead で最小化。
- `lse`: ドリフト対比 $\sum\|\Delta X-a\Delta t\|^2/\Delta t$（YUIMA `lse`）。
- `lasso_qmle`: $-\mathrm{QL}+\lambda\|\theta\|_1$。
- `euler_residuals`: Mahalanobis / ホワイト化残差（$L^{-1}\mathrm{innov}$）。
- 情報量: $\mathrm{AIC}=-2\mathrm{QL}+2k$, $\mathrm{BIC}=-2\mathrm{QL}+k\ln n$, `Fit::qbic` は Eguchi–Masuda $\mathrm{QBIC}=-2\mathrm{QL}+\log\det(-H)$。増分 BIC は `bic_increments`。

変化点:

- `change_point_qv`: 一変量二次変分の二標本ガウス対比 $n_1\ln(\mathrm{QV}_1/n_1)+n_2\ln(\mathrm{QV}_2/n_2)$ を最小化。
- `change_point_qmle`: 左右で凍結した $\mathrm{QL}(\theta_1)+\mathrm{QL}(\theta_2)$ を格子走査（YUIMA `CPoint`）。

適応ベイズ（YUIMA `adaBayes`）:

$$
\pi(\theta\mid\mathrm{data})\propto\exp(\mathrm{QL}(\theta))\,N(\theta;\,\theta_0,\mathrm{diag}(\sigma_{\mathrm{prior}})^2).
$$

ランダムウォーク Metropolis–Hastings。提案 $\theta'=\theta+\sigma_{\mathrm{step}}\odot Z$（$Z$ は amatsuki Box–Muller）。採択 $\min(1,\exp(\ell'-\ell))$ を `OpenClosed01` と比較。事後平均と MAP を返します。YUIMA の適応提案は無い。

Wald 検定: QL の中心差分 Hessian $H$、分散 $(-H)^{-1}$（faer LU）。両側 $p$ 値は Abramowitz–Stegun 7.1.26 の $\mathrm{erfc}$。

### 点過程の尤度と MLE

| モデル | 尤度 / 推定 | 乱数（シミュレーション） | 文献 |
| --- | --- | --- | --- |
| 斉次ポアソン | $n\ln\lambda-\lambda(T-t_0)$ | 指数待ち時間 | — |
| 非斉次ポアソン | 上限 $\bar\lambda$ の **Ogata thinning** | `Exp1`, `Uniform` | Ogata (1981) |
| 指数 Hawkes | 閉形式 $\sum\ln\lambda(t_i)-\mu T-(\alpha/\beta)\sum(1-e^{-\beta(T-t_i)})$。MLE は Nelder–Mead | 再帰状態 $R$ 付き thinning | Hawkes (1971); Ozaki (1979) |
| 2 次元 Hawkes | 相互励起 $\lambda_i=\mu_i+\sum_j\alpha_{ij}e^{-\beta_{ij}(t-t_k^j)}$ | 上と同じ thinning | Hawkes (1971) |
| べき乗 Hawkes | $\lambda=\mu+\sum\alpha(t-t_i+c)^{-p}$ | thinning | — |
| 自己補正 | Isham–Westcott $\lambda=\exp(\mu+\beta t-\alpha N_{t-})$。thinning の上界は区間右端の強度 | thinning | Isham & Westcott (1979)。上界は補った |
| Weibull 更新 | 間隔 $x=\lambda^{-1}(-\ln U)^{1/k}$。対数尤度は間隔項 + 右打ち切り $\log S(T-t_n)=-(\lambda(T-t_n))^k$。`WeibullRenewal::mle` | `Uniform` + 逆変換 | — |
| Gamma 更新 | 間隔 $\mathrm{Gamma}(\mathrm{shape},\mathrm{scale})$（Marsaglia–Tsang） | amatsuki `Gamma` | — |
| CIR Cox | 強度を CIR の Euler で生成し、小区間の中点強度で斉次ポアソンを重ねる | `StandardNormal`, `Exp1` | 正確な Cox 積分は無い |
| marked Hawkes | $\lambda=\mu+\alpha R$、マーク $z\sim\mathrm{Exp}(1/\mathrm{mark\_mean})$、受理時 $R\leftarrow R+z$ | `Exp1`, `Exp`, `Uniform` | — |
| 抑制 Hawkes | 指数 Hawkes と同じ再帰だが $\alpha<0$。強度は正にクリップ | thinning | クリップは補った |
| **CARMA-Hawkes** | 潜在線形状態 $\dot x=Ax$、$\lambda=\mu+b^\top x$。ジャンプ間は $\mathrm{expm}(A\Delta t)$。補償 $\mu\Delta+b^\top\int e^{As}x\,ds$。$p\ge 2$ は局所上界。専用 MLE は Nelder–Mead | thinning + faer `expm` | Mercuri et al.。NM |
| ACD(1,1) | $\psi_i=\omega+\alpha x_{i-1}+\beta\psi_{i-1}$, $\ell=\sum(-\ln\psi-x/\psi)$。MLE は Nelder–Mead | 推定は無乱数 | Engle & Russell (1998) |

---

## フィルタ（`src/filter/`）— カルマン・非線形カルマン・粒子フィルタ

共通モデルは加法ノイズの離散状態空間 `DiscreteSsm` です。

$$
x_{k+1}=f(t_k,\Delta t,x_k)+w_k,\quad w_k\sim N(0,Q),
$$

$$
y_k=h(t_k,x_k)+v_k,\quad v_k\sim N(0,R).
$$

最初の観測時刻は事前分布（$t_0$ では更新しない）。以降は予測→更新。これは既存の YUIMA `KalmanBucy` と同じ規約です。

線形ガウス `LinearGaussian` は $x^+=Fx+u+w$, $y=Hx+v$。SDE からは `SdeSsm`（Euler: $f=x+a\Delta t$, $Q=\sigma\sigma^\top\Delta t$）または `LinearGaussian::from_linear_sde`（Van Loan の $F,u,Q$。不規則 $\Delta t$ で再計算）。

多変量正規の対数密度・サンプリングは faer Cholesky です。粒子・EnKF の $Z\sim N(0,I)$ は amatsuki Box–Muller。

### 線形カルマン

| 関数 | アルゴリズム | 文献 |
| --- | --- | --- |
| `kalman_bucy` | 連続–離散 Kalman–Bucy。Van Loan で $F,u,Q$。更新は古典形 $P\leftarrow(I-KH)P^-$ | Kalman & Bucy (1961); Van Loan (1978) |
| `kalman` | 離散 KF。予測 $x^-=Fx+u$, $P^-=FPF^\top+Q$。更新は **Joseph** $P=(I-KH)P(I-KH)^\top+KRK^\top$。革新尤度 $N(\nu;0,HP^-H^\top+R)$ | Kalman (1960) |
| `square_root_kalman` | 同じ Joseph 更新のあと Cholesky で $P$ を SPD に戻す | QR 平方根形ではない |
| `information_filter` | 予測は共分散形。更新 $Y^+=Y^-+H^\top R^{-1}H$, $y^+=Y^-x^-+H^\top R^{-1}z$ | — |
| `rts_smoother` | **Rauch–Tung–Striebel**。$C=P_f F^\top(P^-)^{-1}$, $x_s=x_f+C(x_s^+-x^-)$ | Rauch, Tung & Striebel (1965) |
| `adaptive_kalman` | **Sage–Husa**。忘却 $b$, $d_k=(1-b)/(1-b^{k+1})$。$R\leftarrow(1-d)R+d(\nu\nu^\top-HP^-H^\top)$, $Q\leftarrow(1-d)Q+d(K\nu\nu^\top K^\top)$ | Sage & Husa (1969) |

### 非線形カルマン

| 関数 | アルゴリズム | 文献 |
| --- | --- | --- |
| `extended_kalman` | EKF。$F=\partial f/\partial x$, $H=\partial h/\partial x$（解析 Jacobian が無ければ中心差分 $\varepsilon=10^{-6}(1+\|x_j\|)$）。Joseph 更新 | — |
| `iterated_ekf` | **Bell–Cathey IEKF**。$x\leftarrow x^-+K(y-h(x)-H(x^--x))$ を再線形化 | Bell & Cathey (1993)。反復は固定上限 |
| `second_order_ekf` | 共分散は 1 次、観測平均に $\hat h_i+\frac12\sum_j P_{jj}\partial^2 h_i/\partial x_j^2$（Jacobian の中心差分） | 対角二階だけ |
| `unscented_kalman` | **Julier–Uhlmann / van der Merwe UKF**。$\lambda=\alpha^2(n+\kappa)-n$、σ 点 $x\pm\mathrm{chol}((n+\lambda)P)$。既定 $\alpha=1,\beta=2,\kappa=0$ | Julier & Uhlmann (1997)。既定は原文の $\kappa=3-n$ でも小さな $\alpha$ でもない |
| `cubature_kalman` | **Arasaratnam–Haykin CKF**。$2n$ 点 $x\pm\sqrt{n}(\mathrm{chol}\,P)_j$、重み $1/(2n)$ | Arasaratnam & Haykin (2009) |
| `ensemble_kalman` | 確率的 **EnKF**（摂動観測）。予測で $w\sim N(0,Q)$、更新 $x\leftarrow x+K(y+v-h(x))$。乗法インフレーション可 | Evensen (2003)。ETKF ではない |
| `gaussian_sum_filter` | 混合 EKF。重み $w_i\propto w_i N(y;\hat h_i,S_i)$ | Alspach & Sorenson (1972) |
| `continuous_discrete_ekf` | モーメント方程式の Euler $\dot x=a$, $\dot P=AP+PA^\top+\sigma\sigma^\top$（$A=\partial a/\partial x$）のあと離散 EKF 更新 | — |
| `extended_rts_smoother` | RTS の $F$ をフィルタ平均での $\partial f/\partial x$ にしたもの | — |

線形ガウスでは KF / EKF / IEKF / UKF / CKF / 情報フィルタの濾波平均は数値誤差の範囲で一致します（単体テストで確認）。

### 粒子フィルタとその拡張

提案分布が遷移のとき $x\sim N(f(x),Q)$、尤度は $N(y;h(x),R)$。正規化対数重みは $-\ln N$（初期化・再標本後）。増分周辺尤度は log-sum-exp。ESS $1/\sum w_i^2$。再標本は ESS が $\texttt{ess\_ratio}\cdot N$ を下回ったとき。

| 関数 | アルゴリズム | 文献 |
| --- | --- | --- |
| `sis_filter` | 逐次重点サンプリング（再標本なし） | Doucet, de Freitas & Gordon (2001) |
| `particle_filter` | **Bootstrap / SIR**。遷移から提案、$p(y\mid x)$ で重み付け | Gordon, Salmond & Smith (1993) |
| `auxiliary_particle_filter` | **Pitt–Shephard APF**。第一段 $\propto w\,p(y\mid\mu)$, $\mu=E[x_k\mid x_{k-1}]$。第二段 $p(y\mid x)/p(y\mid\mu)$ | Pitt & Shephard (1999) |
| `particle_filter_model` | `ParticleModel`（`sample_transition` / `log_obs_density`）。Tobit / $t$ / ポアソン観測 | — |
| `pmmh` / `smc2` / `conditional_smc` | 粒子周辺 MH、外側 $\theta$ 粒子、粒子 Gibbs | Andrieu, Doucet & Holenstein (2010); Chopin, Jacob & Papaspiliopoulos (2013) |
| `ssm_mle` / `shumway_stoffer_em` | KF 革新尤度の最大化、RTS クロス共分散付き EM | Shumway & Stoffer (1982); Durbin & Koopman |
| `regularized_particle_filter` | **Musso–Oudjane–Le Gland RPF**。再標本後に Silverman 帯域 $h=(4/(n+2))^{1/(n+4)}N^{-1/(n+4)}$ のガウス核でジッタ $x+=h\cdot\mathrm{chol}(P)Z$ | Musso, Oudjane & Le Gland (2001) |
| `unscented_particle_filter` | **van der Merwe UPF**。各粒子が 1 ステップ UKF で $q(x_k\mid x_{k-1},y_k)\approx N(m,P)$ を作り、$w\propto p(y\mid x)p(x\mid x^-)/q(x)$ | van der Merwe, Doucet, de Freitas & Wan (2000) |
| `particle_smoother` | **FFBSi**。前向き PF を保存し、$P(i_k\mid i_{k+1})\propto w_k^{(i)}p(x_{k+1}\mid x_k^{(i)})$ で後退パスを引く | Godsill, Doucet & West (2004)。パス本数は引数で制限 |

再標本スキーム（`ResamplingScheme`）:

- **Multinomial** — 独立カテゴリカル。
- **Systematic**（既定）— Kitagawa (1996)。$U\sim[0,1/N)$ + 等間隔格子。
- **Stratified** — 各層 $[i/N,(i+1)/N)$ に 1 点。
- **Residual** — $\lfloor Nw_i\rfloor$ を確定コピーし、残りを多項。

一様乱数は `Rng::next_f64`（ChaCha8、$[0,1)$）。Kitagawa の開区間とは端点が違う。プロセスノイズと核ジッタは Box–Muller。

---

## 高頻度統計（`src/highfreq.rs`, `src/hft.rs`）

推定は二次変分・カーネル・回帰で、乱数は使いません（テスト用パス生成だけ amatsuki）。

| 関数 | 数式 / アルゴリズム | 文献 |
| --- | --- | --- |
| `realized_covariance` | $\sum\Delta X\Delta X^\top$ | — |
| `bipower_variation` | $(\pi/2)\sum\|\Delta X_i\|\|\Delta X_{i-1}\|$ | Barndorff-Nielsen & Shephard (2004) |
| `realized_quarticity` | $(n/3)\sum(\Delta X)^4$ | 同上 |
| `bns_jump_test` | $(\mathrm{QV}-\mathrm{BV})/\sqrt{\theta\,\mathrm{RQ}/n}$, $\theta=\pi^2/4+\pi-5$。$p$ 値は AS `erf` | 同上 |
| `hayashi_yoshida` | $\sum_{i,j}\Delta X_i\Delta Y_j\,1_{\{\text{区間が交わる}\}}$ | Hayashi & Yoshida (2005) |
| `cce` | HY 共分散と相関（YUIMA `cce`） | — |
| `hy_avar` | 対角 $(2/3)\mathrm{RQ}$。非対角は **experimental**（`experimental_hy_avar`） | 未検証 |
| `lead_lag` | $\hat\theta=\arg\max_\theta\|\mathrm{HY}(X,Y_{\cdot+\theta})\|$ | Hoffmann, Rosenbaum & Yoshida (2013) |
| `preaverage` | $g(x)=x\wedge(1-x)$ | Jacod, Li, Mykland, Podolskij & Vetter (2009) |
| `preaveraged_hy` | リフレッシュ同期のあと $\sum(g*\Delta X)(g*\Delta Y)/(\psi k_n)$ | Christensen–Kinnebrock–Podolskij。この $g$ だけ |
| `realized_kernel` | Tukey–Hanning $w(x)=\frac12(1+\cos\pi x)$ の BN–HLS 実現カーネル | Barndorff-Nielsen, Hansen, Lunde & Shephard (2008) |
| `two_scale_rv` | Zhang–Mykland–Aït-Sahalia $\mathrm{RV}^{(K)}-(n/K)n^{-1}\mathrm{RV}^{(1)}$ | Zhang, Mykland & Aït-Sahalia (2005) |
| `previous_tick` / `refresh_times` | 前回値補間、BNHLS リフレッシュ時計。完全一致は `intersection_times` | Barndorff-Nielsen et al. |
| `roll_spread` | $2\sqrt{-\gamma_1}$ | Roll (1984) |
| `kyle_lambda` | $\Delta P$ の符号付き出来高への OLS | Kyle (1985) |

実行アルゴリズム（制御と共有）:

- **Almgren–Chriss** (2001): $\kappa=\sqrt{\lambda\sigma^2/\eta}$, 在庫 $x(t)=x_0\sinh(\kappa(T-t))/\sinh(\kappa T)$。線形一時衝撃の閉形式だけ。
- **TWAP**: 等分割。
- **mid / microprice**: 中値、$\alpha=q_{\mathrm{bid}}/(q_{\mathrm{bid}}+q_{\mathrm{ask}})$ のキュー不均衡加重。

線形代数が要るのは共分散行列の格納（faer `Mat`）だけです。

---

## 確率制御（`src/control.rs`）

| 対象 | 公式 / アルゴリズム | 文献 | 行列 |
| --- | --- | --- | --- |
| Merton CRRA / Kelly | $\pi^*=(\mu-r)/(\gamma\sigma^2)$。Kelly は $\gamma=1$ | Merton (1971); Kelly (1956) | なし |
| 対数成長 | $r+\pi(\mu-r)-\frac12\pi^2\sigma^2$ | — | なし |
| LQR | Euler $x^+=(I+A\Delta t)x+(B\Delta t)u$ の離散 Riccati。$K=(R\Delta t+B^\top PB)^{-1}B^\top PA_d$ | 連続 LQR の厳密解ではない | faer 逆行列 |
| 1D HJB | 陽解法 + 中心差分。CFL $\mathrm{dt}\cdot\sigma^2/\mathrm{dx}^2\le 1/2$ でなければ `Err` | Fleming & Soner | なし |

---

## Malliavin / Itô–Taylor（`src/malliavin.rs`, `src/expansion.rs`）

> [!info] 文献
> Fournié, Lasry, Lebuchoux, Lions & Touzi (1999)。パスは Euler、係数微分は中心差分であり、連続時間の Malliavin 重みそのものではない。

スカラー拡散、正則格子。パスは Euler、$Z$ は amatsuki `StandardNormal`。

- 第一変分 $Y$: $dY=a_x Y\,dt+\sigma_x Y\,dW$（係数は中心差分）。
- Fournié 重み: $u=Y/(\sigma T)$, $\pi_\Delta\leftarrow\pi_\Delta+u\Delta W$。$\Gamma$ 用の二次重みは積の規則の Euler 近似。
- Greeks: $E[f(X)]$, $E[f(X)\pi_\Delta]$, $E[f(X)\pi_\Gamma]$。
- 密度: ガウス核 $K_h(X-y)$ の MC（帯域 $h$）。
- 特性関数 / 生モーメント: Euler パスの MC。歪度・尖度はキュムラント変換。
- 小ノイズ: ODE $\dot x=a$ に沿う $d_0=f(\bar x_T)$ と $d_1=\frac12 f''(\bar x)\int\sigma^2$（$\varepsilon^2$ 項）。
- Itô–Taylor: $E[f(X_T)]\approx f+T\mathcal{L}f+\frac12 T^2\mathcal{L}^2 f$, $\mathcal{L}=a\partial_x+\frac12\sigma^2\partial_{xx}$。係数微分は中心差分。経路汎関数の MC は台形則。

---

## ノンパラメトリックベイズ（`src/npbayes/`）

YUIMA に無いカタログ。詳細は [`docs/npbayes.md`](docs/npbayes.md)。差分の全文は [`docs/deviations.md`](docs/deviations.md)。

### Dirichlet / Pitman–Yor

> [!info] 文献
> Ferguson (1973); Sethuraman (1994); Blackwell & MacQueen (1973); Aldous (1985); Perman, Pitman & Yor (1992); Pitman & Yor (1997); Ishwaran & James (2001)。

$$
V_k\sim\mathrm{Beta}(1,\alpha),\qquad
\pi_k=V_k\prod_{j<k}(1-V_j),\qquad
G=\sum_{k=1}^\infty\pi_k\delta_{\theta_k}.
$$

実装は有限 $K$ 切断で残り質量を最後の原子へ（Sethuraman の無限和そのものではない）。CRP は切断なしで交換可能分割どおり:

$$
P(\text{卓 }k)=\frac{n_k}{i-1+\alpha},\qquad P(\text{新卓})=\frac{\alpha}{i-1+\alpha}.
$$

Pitman–Yor: $V_k\sim\mathrm{Beta}(1-d,\theta+kd)$, $P(\text{卓 }k)=(n_k-d)/(n-1+\theta)$, $P(\text{新})=(\theta+Kd)/(n-1+\theta)$。

### Beta–Bernoulli 過程と IBP

> [!info] 文献
> Hjort (1990); Thibaux & Jordan (2007); Griffiths & Ghahramani (2005, 2011); Teh, Görür & Ghahramani (2007)。

Hjort の Lévy 測度 $\nu(d\pi,d\omega)=c\pi^{-1}(1-\pi)^{c-1}\,d\pi\,B_0(d\omega)$ からの CRM 構成は **実装していない**。実装は次の三つ。

1. 逐次 IBP（$c=1$ のみ、交換可能特徴割当としては原文どおり）: 顧客 $i$ は既存皿を確率 $m_k/i$ で取り、$\mathrm{Poisson}(\alpha/i)$ 個の新品。
2. 有限 Beta 過程 $\pi_k\sim\mathrm{Beta}(c\gamma/K,\,c(1-\gamma/K))$ と Bernoulli 過程 $z_{nk}\mid\pi_k\sim\mathrm{Bern}(\pi_k)$。
3. TGG stick-breaking $v_i\sim\mathrm{Beta}(\alpha,1)$, $\pi_k=\prod_{i=1}^k v_i$（有限 $K$ で尾を切る）。

IBP$(\alpha)$ は $c=1$, $B_0(\Omega)=\alpha$ の Beta–Bernoulli の周辺（Thibaux & Jordan 2007）。

### 推論

- **Neal (2000) Algorithm 3**: $x_i\mid z_i=k\sim N(\mu_k,\sigma^2)$, $\mu_k\sim N(\mu_0,\tau_0^2)$。$\sigma^2$ は固定（Neal の分散事前は無い）。
- **HDP CRF** (Teh, Jordan, Beal & Blei 2006): テーブルを明示。尤度は論文の多項ではなく共役ガウス。
- **線形ガウス IBP Gibbs** (Griffiths & Ghahramani 2011 §4.1): $X=ZA+E$ で $A$ を積分した周辺尤度。新特徴数は $\kappa_{\max}$ までの列挙。

```rust
use isuzu::datasets::{make_dp_gaussians, make_ibp_linear_gaussian};
use isuzu::npbayes::{dp_gaussian_mixture_gibbs, sample_ibp_sequential, IbpParams};
use isuzu::prelude::*;

let mut rng = seed_rng(1);
let z = sample_ibp_sequential(30, IbpParams::new(1.5)?, &mut rng)?;
let (x, truth) = make_dp_gaussians(40, &[-3.0, 3.0], 0.4, 2)?;
let fit = dp_gaussian_mixture_gibbs(&x, 0.5, 0.4, 0.0, 3.0, 40, &mut rng)?;
```

---

## モデルカタログのドリフト / 拡散（実装どおり）

記号はコードのパラメータ名です。連続部分のシミュレーションは上のスキーム、パラメータ推定は `ParametricSde` に載せたうえで QMLE / LSE（点過程は専用 MLE）です。拡散行列が 2 列のモデルは、相関 $\rho$ を **Cholesky** $\begin{pmatrix}1&0\\\rho&\sqrt{1-\rho^2}\end{pmatrix}$ として $\sigma$ に埋め込んでいます。

| モデル | 実装されている方程式 | 文献 |
| --- | --- | --- |
| GBM / Black–Scholes | $dS=\mu S\,dt+\sigma S\,dW$。厳密: $S\leftarrow S\exp((\mu-\frac12\sigma^2)\Delta t+\sigma\sqrt{\Delta t}Z)$ | Black & Scholes (1973) |
| OU / Vasicek | $dX=\kappa(\theta-X)\,dt+\sigma\,dW$。厳密: $X\leftarrow\theta+(X-\theta)e^{-\kappa\Delta t}+\sqrt{\sigma^2(1-e^{-2\kappa\Delta t})/(2\kappa)}\,Z$ | Uhlenbeck & Ornstein (1930); Vasicek (1977) |
| CIR | $dX=\kappa(\theta-X^+)\,dt+\sigma\sqrt{X^+}\,dW$。Feller: $2\kappa\theta\ge\sigma^2$ | Cox, Ingersoll & Ross (1985)。反射 $X^+$。非心 $\chi^2$ は無い |
| Heston | 状態 $[S,v]$。$dS=\mu S\,dt+\sqrt{v}S\,dW^1$, $dv=\kappa(\theta-v)\,dt+\xi\sqrt{v}\,dW^2$。$\sigma=\begin{pmatrix}\sqrt{v}S&0\\\xi\sqrt{v}\rho&\xi\sqrt{v}\sqrt{1-\rho^2}\end{pmatrix}$ | Heston (1993)。Andersen QE は無い |
| Hull–White | $dX=(\theta-\kappa X)\,dt+\sigma\,dW$（定数 $\theta$） | Hull & White (1990)。$\theta(t)$ は無い |
| Black–Karasinski | 状態 $r>0$。Itô: $dr=r[\kappa(\theta-\ln r)+\frac12\sigma^2]\,dt+\sigma r\,dW$ | Black & Karasinski (1991) |
| CKLS | $dX=(\alpha+\beta X)\,dt+\sigma\|X\|^\gamma\,dW$。CEV は $\alpha=0$。Brennan–Schwartz は $\gamma=1$ | Chan, Karolyi, Longstaff & Sanders (1992) |
| Jacobi | $dX=\kappa(\theta-X)\,dt+\sigma\sqrt{X(1-X)}\,dW$（状態は $[0,1]$ にクリップ） | クリップは補った |
| Bessel | $dX=(\delta-1)/(2\|X\|)\,dt+dW$ | $\|X\|$ で 0 を避ける |
| Brownian bridge | $dX=(b-X)/(T-t)\,dt+\sigma\,dW$ | — |
| SABR | 状態 $[F,\alpha]$。$dF=\alpha F^\beta\,dW^1$, $d\alpha=\nu\alpha\,dW^2$。ドリフト 0。$\sigma$ に $\rho$ の Cholesky | Hagan, Kumar, Lesniewski & Woodward (2002) |
| 3/2 | $dv=\kappa v(\theta-v)\,dt+\xi v^{3/2}\,dW$ | — |
| Stein–Stein | 状態 $[S,v]$。$dS=\mu S\,dt+\|v\|S\,dW^1$, $dv=\kappa(\theta-v)\,dt+\xi\,dW^2$ | Stein & Stein (1991) |
| Bates | Heston と同じ拡散 + スポットの乗法 Merton ジャンプ $\Delta S=S(e^Z-1)$, $Z\sim N(\mathrm{jump}_\mu,\mathrm{jump}_\sigma^2)$, 強度 `intensity` | Bates (1996) |
| Merton | $dX=\mu X\,dt+\sigma X\,dW$ + 乗法複合ポアソン、$Z\sim N(\mathrm{jump}_\mu,\mathrm{jump}_\sigma^2)$ | Merton (1976) |
| Kou | 同じ拡散 + 二重指数ジャンプ（`JumpLaw::DoubleExponential`） | Kou (2002) |
| fGBM | $dX=\mu X\,dt+\sigma X\,dB^H$（増分は Davies–Harte fGn $\times\Delta t^H$） | Mandelbrot & Van Ness (1968) |
| fOU | OU 係数 + 同上の fBM 駆動 | — |
| CARMA$(p,q)$ | Brockwell: $Y=b^\top X+c$, $dX=AX\,dt+e\,dL$。同伴行列の最終行は $-(a_p,\ldots,a_1)$。$L$ がガウスなら最後座標の拡散は $\sigma$、それ以外は $e\,dL$ ジャンプ | Brockwell (2001) |
| COGARCH$(p,q)$ | $V=a_0+a^\top Y_{t-}$, $dG=\sqrt{V}\,dL$, $dY=BY\,dt+eV(\Delta L)^2$。(1,1) は $a_0=\beta/\eta$, $a_1=\varphi$, $b_1=\eta$ | Klüppelberg, Lindner & Maller (2004)。Euler |
| 線形状態空間 | $dX=(AX+b)\,dt+\sigma\,dW$、観測 $Y=HX$ | — |
| `FnSde` | 任意の閉包ドリフト / 拡散（YUIMA `setModel` 相当） | — |
| 点過程 | 上表の強度。Hawkes / CARMA-Hawkes / ACD / Weibull は専用対数尤度 + Nelder–Mead | 各節 |

Lévy パス `levy_path` は $X_{i+1}=X_i+L(\Delta t_i)$。`gamma_process` は VG 表現 $\theta=\mathrm{scale}$, $\nu=1/\mathrm{shape\_rate}$, $\sigma=0$ でガンマ過程を出します。

---

## sklearn 風 API と監査

- `QmleEstimator::fit` → `qmle`（Nelder–Mead + Euler QL）。
- `EulerSimulator` → `simulate`（スキーム既定 Euler、乱数は呼び出し側の `amatsuki::Rng`）。
- `recover` / `Pipeline`: 既知パラメータとの絶対誤差。
- `time_series_folds`: 連続ブロックの k-fold（シャッフルなし）。
- `datasets::make_ou` / `make_gbm` / `make_dp_gaussians` / `make_ibp_linear_gaussian` など。
- `audit`: lockfile に禁止クレートが無いこと、直接依存が `amatsuki` / `faer` / `rustfft` / `thiserror` だけであること。

---

## ベンチマーク

`cargo run --release --example benchmark` を Intel Xeon（4 コア、単一スレッド、rustc 1.85、`--release`）で走らせた実測。表の全部と読み方は [`docs/benchmarks.md`](docs/benchmarks.md)。

| 問題 | 速度 | 精度 |
| --- | --- | --- |
| OU Euler QMLE（$n=1500$, $T=8$, 5 本） | 平均 $172\,\mathrm{ms}$ | $\sigma$ の RMSE $0.004$；$\kappa$ は 1 本が箱上限（RMSE $1.31$） |
| 線形ガウス KF / UKF / SIR（$n=200$） | $0.71$ / $1.08$ / $106\,\mathrm{ms}$ | RMSE $0.234$ / $0.234$ / $0.235$ |
| 指数 Hawkes MLE（$T=80$） | $0.4\,\mathrm{ms}$ | $\mu$ RMSE $0.15$；$\beta$ は短い $T$ で不安定（$0.90$） |
| Hayashi–Yoshida（$\rho=0.6$, $T=1$） | $0.003\,\mathrm{ms}$ | 絶対誤差 RMSE $0.092$ |
| GBM 厳密対 Euler（80 パス $\times$ 5 seed） | どちらも $1.4\,\mathrm{ms}$ | 終端平均誤差 $0.032$ / $0.034$ |
| DP 混合 Neal Alg. 3（3 塊 $\times$ 40） | $4.3\,\mathrm{ms}$ | 多数決精度 $1.000$（過剰クラスタあり） |
| 線形ガウス IBP Gibbs | $21.5\,\mathrm{ms}$ | 正規化 Hamming $0.13$ |

再計算:

```bash
cargo run --release --example benchmark
```

---

## Pure Rust ポリシー

直接依存: `faer`（LAPACK なし）, 付属クレート `amatsuki`（乱数のアルゴリズムとインターフェース）, `rustfft`, `thiserror`。
乱数は `amatsuki` の ChaCha8 と分布サンプラーで完結し、`rand` / `nalgebra` には依存しません。
行列分解と固有値は `faer`、分数ガウスの循環埋め込みだけ `rustfft`、準尤度・点過程 MLE・フィルタの数値最適化は自前の Nelder–Mead と faer Cholesky / LU です。

## ライセンス

Apache-2.0
