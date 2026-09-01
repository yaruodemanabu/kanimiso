# 線形代数と最適化

## faer 上の密行列（`src/linalg.rs`）

> [!info] 文献
> Cholesky / 部分ピボット LU は Golub & Van Loan。行列指数の scaling-and-squaring は Higham (2008)。固有値は faer。

| 操作 | 式 / アルゴリズム | 進む |
| --- | --- | --- |
| `cholesky` | $A=LL^\top$ | faer `llt` |
| `logdet_spd` | $\log\det A = 2\sum_i \log L_{ii}$ | — |
| `try_inverse` | 部分ピボット LU | — |
| `expm` | scaling-and-squaring + Taylor（最大 27 項） | Padé ではない |
| `spd_regularize` | $\varepsilon I$ を足して Cholesky が通るまで（最大 12 回、$\varepsilon\leftarrow 10\varepsilon$） | 文献に無い保護 |
| `spectral_radius` | $\rho(A)=\max\|\lambda\|$ | — |
| `is_hurwitz` | すべての $\mathrm{Re}\,\lambda<0$ | 閾値 $10^{-14}$ |
| `gram_rowmajor` | $\Sigma=\sigma\sigma^\top$ | — |

Taylor `expm`（スケール後 $B=2^{-s}A$）:

$$
e^{B}\approx\sum_{k=0}^{27}\frac{B^k}{k!},\qquad e^{A}=(e^{B})^{2^s}.
$$

## Nelder–Mead（`src/optimize.rs`）

> [!info] 文献
> Nelder & Mead (1965)。1 次元は Kiefer (1953) 黄金分割。

反射 $\alpha=1$、拡大 $\gamma=2$、収縮 $\rho=0.5$、縮小 $\sigma=0.5$。

箱制約は座標クリップ。初期単体は各座標に相対 5% ステップ。QMLE / LSE / LASSO / Hawkes MLE / CARMA-Hawkes MLE / ACD MLE がこれを呼ぶ。

黄金分割の比は $\varphi=(1+\sqrt{5})/2$。

## L-BFGS-B

`lbfgs_b` は制限記憶 BFGS + 箱射影。勾配が無ければ前進差分。Cauchy 探索と部分空間制約は実装していない。QR 最小二乗は `qr_least_squares`（列スケール、ridge fallback、rank / 条件数）。
