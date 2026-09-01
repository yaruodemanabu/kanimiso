# シミュレーション

モデルは Itô 過程（`src/model.rs` の `Sde`）

$$
dX_t = a(t,X_t)\,dt + \sigma(t,X_t)\,dW_t + \gamma(t,X_{t-})\,dJ_t.
$$

分数ブラウンは `hurst()`。乗法ジャンプは Merton / Kou 形 $\Delta X=\gamma(e^{\Delta L}-1)$。

## 離散化

> [!info] 文献
> Euler–Maruyama: Maruyama (1955)。Milstein: Milstein (1974)。1.5 次: Kloeden & Platen (1992)。

| `Scheme` | 更新 | 備考 |
| --- | --- | --- |
| Euler–Maruyama | $x\leftarrow x+a\Delta t+\sigma\Delta W$ | 既定 |
| Milstein | 可換ノイズの Lévy 面積。$n>1$ かつ $m>1$ は `Unsupported` | 強次数 1（可換） |
| Kloeden–Platen 1.5 | スカラー。$I_{(1,1,1)}$ を含む。$\Delta Z=\frac12\Delta t(\Delta W+\sqrt{\Delta t/3}\,Z)$ | |
| Exact | `Sde::exact_step` | GBM 対数正規、OU ガウス遷移 |

OU の厳密ステップ:

$$
X_{t+\Delta t}=\theta+(X_t-\theta)e^{-\kappa\Delta t}+\sqrt{\frac{\sigma^2(1-e^{-2\kappa\Delta t})}{2\kappa}}\,Z.
$$

GBM の厳密ステップ:

$$
X\leftarrow X\exp\bigl((\mu-\tfrac12\sigma^2)\Delta t+\sigma\sqrt{\Delta t}\,Z\bigr).
$$

COGARCH は Euler + Lévy 増分で分散状態を更新する（[[deviations]]）。

## 駆動ノイズ

乱数はすべて amatsuki。

| 関数 | 法則 | アルゴリズム | 文献 |
| --- | --- | --- | --- |
| `brownian_increments` | $\Delta W\sim N(0,\Delta t)$ | $\sqrt{\Delta t}\cdot N(0,1)$ | — |
| `correlated_brownian_increments` | 相関ブラウン | faer Cholesky $L$ で $LZ$ | — |
| `fractional_gaussian_noise` | 単位分散 fGn | Davies–Harte $m=2n$（$\gamma(n)$ 込み）。固有値は rustfft | Davies & Harte (1987); Wood & Chan (1994) |
| `fractional_brownian_increments` | 正則格子上の fBM 増分 | $\Delta t^H\cdot\mathrm{fGn}$ | Mandelbrot & Van Ness (1968) |
| `LevyMeasure::increment` | 複合ポアソン / VG / NIG / 安定 | 複合ポアソン: $N\sim\mathrm{Poisson}(\lambda\Delta t)$ + `JumpLaw`。VG: $G\sim\mathrm{Gamma}(\Delta t/\nu,\nu)$, $\theta G+\sigma\sqrt{G}Z$。NIG: IG 従属。安定: CMS を $\sigma\Delta t^{1/\alpha}$ で尺度 | 各法則の標準構成 |
| `JumpLaw` | 正規 / Kou 二重指数 / 定数 / $t$ / Laplace | Kou: $U<p$ なら $+\mathrm{Exp}/\eta_+$, さもなくば $-\mathrm{Exp}/\eta_-$。Laplace: $\mathrm{scale}(E_1-E_2)$ | Kou (2002) |
| `poisson_arrivals` | 斉次ポアソン到着 | $t\leftarrow t+E/\lambda$ | — |
| `poisson_random_sampling` | 間引き観測 | 各点を確率 `rate` で残す | — |
