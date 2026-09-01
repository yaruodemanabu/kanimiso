# 点過程

| モデル | 尤度 / 推定 | シミュレーション | 文献 | 進む |
| --- | --- | --- | --- | --- |
| 斉次ポアソン | $n\ln\lambda-\lambda(T-t_0)$ | 指数待ち時間 | — | — |
| 非斉次ポアソン | 上限 $\bar\lambda$ の Ogata thinning | `Exp1`, `Uniform` | Ogata (1981) | 上界はユーザ指定 |
| 指数 Hawkes | $\sum\ln\lambda(t_i)-\mu T-(\alpha/\beta)\sum(1-e^{-\beta(T-t_i)})$。MLE は Nelder–Mead | 再帰状態 $R$ 付き thinning | Hawkes (1971); Ozaki (1979) | 最適化は NM |
| 2 次元 Hawkes | 相互励起 $\lambda_i=\mu_i+\sum_j\alpha_{ij}e^{-\beta_{ij}(t-t_k^j)}$ | 同上 | Hawkes (1971) | — |
| N 次元指数 Hawkes | 同じ核。対数尤度と $\partial/\partial(\mu,\alpha,\beta)$。Ogata 残差の KS | thinning | Ogata (1988) | 核は減少する指数だけ |
| べき乗 Hawkes | $\lambda=\mu+\sum\alpha(t-t_i+c)^{-p}$ | thinning | — | — |
| 自己補正 | $\lambda=\exp(\mu+\beta t-\alpha N_{t-})$ | 上界は区間右端 | Isham & Westcott (1979) | 上界は補った |
| Weibull 更新 | $x=\lambda^{-1}(-\ln U)^{1/k}$。間隔項 + $\log S(T-t_n)$。`mle` | 逆変換 | — | — |
| Gamma 更新 | 間隔 $\mathrm{Gamma}(\mathrm{shape},\mathrm{scale})$ | Marsaglia–Tsang | — | — |
| CIR Cox | 強度を CIR の Euler で生成し、小区間の中点強度で斉次ポアソン | `StandardNormal`, `Exp1` | — | 正確な積分は無い |
| marked Hawkes | $\lambda=\mu+\alpha R$、マーク $z\sim\mathrm{Exp}(1/\mathrm{mark\_mean})$、受理時 $R\leftarrow R+z$ | — | — | — |
| 抑制 Hawkes | 指数 Hawkes と同じ再帰だが $\alpha<0$。強度は正にクリップ | thinning | — | クリップは補った |
| CARMA-Hawkes | $\dot x=Ax$, $\lambda=\mu+b^\top x$。ジャンプ間は `expm(AΔt)`。補償は閉形式。MLE は NM | thinning + `expm` | Mercuri et al. | $p\ge 2$ は局所上界、NM |
| ACD(1,1) | $\psi_i=\omega+\alpha x_{i-1}+\beta\psi_{i-1}$, $\ell=\sum(-\ln\psi-x/\psi)$ | 推定は無乱数 | Engle & Russell (1998) | NM |

指数 Hawkes の再帰:

$$
\lambda(t)=\mu+\alpha R_t,\qquad R_{t_i}=e^{-\beta\Delta}R_{t_{i-1}}+1.
$$
