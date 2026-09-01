# フィルタ

共通モデルは加法ノイズの離散状態空間 `DiscreteSsm`:

$$
x_{k+1}=f(t_k,\Delta t,x_k)+w_k,\quad w_k\sim N(0,Q),
$$

$$
y_k=h(t_k,x_k)+v_k,\quad v_k\sim N(0,R).
$$

最初の観測時刻は事前分布（$t_0$ では更新しない）。以降は予測→更新。YUIMA `KalmanBucy` と同じ規約。

線形ガウス `LinearGaussian` は $x^+=Fx+u+w$, $y=Hx+v$。SDE からは `SdeSsm`（Euler: $f=x+a\Delta t$, $Q=\sigma\sigma^\top\Delta t$）または `LinearGaussian::from_linear_sde`（Van Loan。不規則 $\Delta t$ で再計算）。

## 線形カルマン

| 関数 | アルゴリズム | 文献 | 進む |
| --- | --- | --- | --- |
| `kalman_bucy` | Van Loan の $F,u,Q$。更新 $P\leftarrow(I-KH)P^-$ | Kalman & Bucy (1961); Van Loan (1978) | Joseph ではない |
| `kalman` | $x^-=Fx+u$, $P^-=FPF^\top+Q$。Joseph $P=(I-KH)P(I-KH)^\top+KRK^\top$ | Kalman (1960) | Joseph 形 |
| `square_root_kalman` | Joseph のあと Cholesky で $P$ を SPD に戻す | — | 平方根フィルタの QR 形ではない |
| `information_filter` | $Y^+=Y^-+H^\top R^{-1}H$, $y^+=Y^-x^-+H^\top R^{-1}z$ | — | 予測は共分散形 |
| `rts_smoother` | $C=P_f F^\top(P^-)^{-1}$, $x_s=x_f+C(x_s^+-x^-)$ | Rauch, Tung & Striebel (1965) | — |
| `adaptive_kalman` | Sage–Husa。$d_k=(1-b)/(1-b^{k+1})$。$R\leftarrow(1-d)R+d(\nu\nu^\top-HP^-H^\top)$ | Sage & Husa (1969) | — |

## 非線形カルマン

| 関数 | アルゴリズム | 文献 | 進む |
| --- | --- | --- | --- |
| `extended_kalman` | $F=\partial f/\partial x$, $H=\partial h/\partial x$（無ければ中心差分 $\varepsilon=10^{-6}(1+\|x_j\|)$） | — | — |
| `iterated_ekf` | $x\leftarrow x^-+K(y-h(x)-H(x^--x))$ | Bell & Cathey (1993) | 反復は固定上限 |
| `second_order_ekf` | $\hat h_i+\frac12\sum_j P_{jj}\partial^2 h_i/\partial x_j^2$ | — | 対角だけ |
| `unscented_kalman` | $\lambda=\alpha^2(n+\kappa)-n$, $\sigma$ 点 $x\pm\mathrm{chol}((n+\lambda)P)$。既定 $\alpha=1,\beta=2,\kappa=0$ | Julier & Uhlmann (1997); van der Merwe (2004) | 既定パラメータは原文のどれとも一致しない |
| `cubature_kalman` | $2n$ 点 $x\pm\sqrt{n}(\mathrm{chol}\,P)_j$, 重み $1/(2n)$ | Arasaratnam & Haykin (2009) | — |
| `ensemble_kalman` | 摂動観測。$x\leftarrow x+K(y+v-h(x))$ | Evensen (2003) | ETKF ではない |
| `gaussian_sum_filter` | $w_i\propto w_i N(y;\hat h_i,S_i)$ | Alspach & Sorenson (1972) | — |
| `continuous_discrete_ekf` | 観測間を複数 Euler サブステップでモーメント ODE を積分したあと離散 EKF | — | モーメント ODE は Euler |
| `extended_rts_smoother` | RTS の $F$ を $\partial f/\partial x$ にしたもの | — | — |

線形ガウスでは KF / EKF / IEKF / UKF / CKF / 情報フィルタの濾波平均は数値誤差の範囲で一致する（単体テスト）。

## 粒子フィルタ

提案が遷移のとき $x\sim N(f(x),Q)$、尤度は $N(y;h(x),R)$。正規化対数重みは $-\ln N$。増分は log-sum-exp。ESS $1/\sum w_i^2$。再標本は ESS が `ess_ratio` $N$ を下回ったとき。

| 関数 | アルゴリズム | 文献 |
| --- | --- | --- |
| `sis_filter` | 再標本なし | Doucet et al. (2001) |
| `particle_filter` | Bootstrap / SIR | Gordon, Salmond & Smith (1993) |
| `auxiliary_particle_filter` | 第一段 $\propto w\,p(y\mid\mu)$, $\mu=E[x_k\mid x_{k-1}]$ | Pitt & Shephard (1999) |
| `regularized_particle_filter` | Silverman $h=(4/(n+2))^{1/(n+4)}N^{-1/(n+4)}$ のガウス核 | Musso, Oudjane & Le Gland (2001) |
| `unscented_particle_filter` | 各粒子が 1 ステップ UKF で $q\approx N(m,P)$ | van der Merwe, Doucet, de Freitas & Wan (2000) |
| `particle_smoother` | FFBSi。$P(i_k\mid i_{k+1})\propto w_k^{(i)}p(x_{k+1}\mid x_k^{(i)})$ | Godsill, Doucet & West (2004) |
| `particle_filter_model` | `ParticleModel`。観測は任意の対数密度 | — |
| `pmmh` / `smc2` / `conditional_smc` | 粒子周辺 MH、外側 $\theta$、粒子 Gibbs | Andrieu et al. (2010); Chopin et al. (2013) |
| `ssm_mle` / `shumway_stoffer_em` | KF 革新尤度、RTS ラグ 1 EM | Shumway & Stoffer (1982) |

再標本: Multinomial / Systematic（Kitagawa 1996、既定）/ Stratified / Residual。欠測は `NaN` を Kalman が飛ばす。
