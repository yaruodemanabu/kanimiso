# 拡散の推定

乱数を使う推定は **適応ベイズ** だけ（提案は `StandardNormal`、採択は `OpenClosed01`）。点推定は準尤度 + Nelder–Mead。

## Euler ガウス準対数尤度

> [!info] 文献
> Yoshida (1992); Kessler (1997) の離散観測。実装のコントラストは YUIMA `qmle` と同じ Euler ガウス準尤度。

$$
\mathrm{QL}(\theta)=\sum_i -\frac12\Bigl(n\ln(2\pi)+\ln\det\Sigma_i+(\Delta X_i-a_i\Delta t)^\top\Sigma_i^{-1}(\Delta X_i-a_i\Delta t)\Bigr),
$$

$$
\Sigma_i=\sigma\sigma^\top(t_i,X_i)\,\Delta t_i
$$

（faer Cholesky で $\log\det$ / 二次形式）。

- `qmle`: $-\mathrm{QL}$ を Nelder–Mead で最小化。
- `lse`: ドリフト対比 $\sum\|\Delta X-a\Delta t\|^2/\Delta t$（YUIMA `lse`）。
- `lasso_qmle`: $-\mathrm{QL}+\lambda\|\theta\|_1$。
- `euler_residuals`: Mahalanobis / ホワイト化残差 $L^{-1}\mathrm{innov}$。
- 情報量: $\mathrm{AIC}=-2\mathrm{QL}+2k$, $\mathrm{BIC}=-2\mathrm{QL}+k\ln n$。`Fit::qbic` は Eguchi–Masuda $\mathrm{QBIC}=-2\mathrm{QL}+\log\det(-H)$。増分 BIC は `bic_increments`。`cic` は $\mathrm{tr}(\hat H^{-1}\hat G)$。
- `two_stage_qmle` / `threshold_qmle` / `kessler_qmle` / `hurst_qgv` / `hurst_mmfrac` / `cogarch_gmm`。
- 箱付き準 Newton は `lbfgs_b`。`Fit::vcov` / `se` は $(-H)^{-1}$。

> [!warning] オリジナルからの進む
> 既定の点推定は Nelder–Mead。L-BFGS-B は有限差分勾配。Kessler はスカラー 2 次まで。

## 変化点

- `change_point_qv`: 一変量二次変分の二標本ガウス対比 $n_1\ln(\mathrm{QV}_1/n_1)+n_2\ln(\mathrm{QV}_2/n_2)$ を最小化。
- `change_point_qmle`: 左右で凍結した $\mathrm{QL}(\theta_1)+\mathrm{QL}(\theta_2)$ を格子走査（YUIMA `CPoint`）。

## 適応ベイズ（YUIMA `adaBayes`）

$$
\pi(\theta\mid\mathrm{data})\propto \exp(\mathrm{QL}(\theta))\,N(\theta;\,\theta_0,\mathrm{diag}(\sigma_{\mathrm{prior}})^2).
$$

ランダムウォーク Metropolis–Hastings。提案 $\theta'=\theta+\sigma_{\mathrm{step}}\odot Z$。採択 $\min(1,\exp(\ell'-\ell))$ を `OpenClosed01` と比較。事後平均と MAP を返す。

Wald 検定: QL の中心差分 Hessian $H$、分散 $(-H)^{-1}$。両側 $p$ 値は Abramowitz–Stegun 7.1.26 の $\mathrm{erfc}$。
