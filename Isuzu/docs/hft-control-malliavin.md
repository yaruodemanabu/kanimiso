# 高頻度・制御・Malliavin

## 高頻度統計

推定は二次変分・カーネル・回帰。乱数はテスト用パスだけ。

| 関数 | 式 | 文献 | 進む |
| --- | --- | --- | --- |
| `realized_covariance` | $\sum\Delta X\Delta X^\top$ | — | — |
| `bipower_variation` | $(\pi/2)\sum\|\Delta X_i\|\|\Delta X_{i-1}\|$ | Barndorff-Nielsen & Shephard (2004) | — |
| `realized_quarticity` | $(n/3)\sum(\Delta X)^4$ | 同上 | — |
| `bns_jump_test` | $(\mathrm{QV}-\mathrm{BV})/\sqrt{\theta\,\mathrm{RQ}/n}$, $\theta=\pi^2/4+\pi-5$ | 同上 | $p$ は自前 `erf` |
| `bns_ratio_test` / `tripower_quarticity` | 四乗パワー分母の比統計 | Barndorff-Nielsen & Shephard | — |
| `lee_mykland` | 局所 $\Delta X/\hat\sigma$ | Lee & Mykland (2008) | — |
| `hayashi_yoshida` | $\sum_{i,j}\Delta X_i\Delta Y_j\,1_{\{\text{区間が交わる}\}}$ | Hayashi & Yoshida (2005) | — |
| `cce` | HY 共分散と相関 | YUIMA `cce` | — |
| `hy_avar` | 対角 $(2/3)\mathrm{RQ}$。非対角は experimental な自作デルタ法 | — | 検証されていない |
| `lead_lag` | $\hat\theta=\arg\max_\theta\|\mathrm{HY}(X,Y_{\cdot+\theta})\|$ | Hoffmann, Rosenbaum & Yoshida (2013) | 格子走査 |
| `preaverage` | $g(x)=x\wedge(1-x)$ | Jacod, Li, Mykland, Podolskij & Vetter (2009) | この $g$ だけ |
| `preaveraged_hy` | リフレッシュ同期のあと $\sum(g*\Delta X)(g*\Delta Y)/(\psi k_n)$ | Christensen–Kinnebrock–Podolskij | $g$ はこの形だけ |
| `realized_kernel` | Tukey–Hanning $w(x)=\frac12(1+\cos\pi x)$ | Barndorff-Nielsen, Hansen, Lunde & Shephard (2008) | この核だけ |
| `two_scale_rv` | $\mathrm{RV}^{(K)}-(n/K)n^{-1}\mathrm{RV}^{(1)}$ | Zhang, Mykland & Aït-Sahalia (2005) | — |
| `previous_tick` / `refresh_times` | 前回値補間、BNHLS リフレッシュ時計 | Barndorff-Nielsen, Hansen, Lunde & Shephard | 完全一致時計は `intersection_times` |
| `roll_spread` | $2\sqrt{-\gamma_1}$ | Roll (1984) | $\gamma_1>0$ なら失敗し得る |
| `kyle_lambda` | $\Delta P$ の符号付き出来高への OLS | Kyle (1985) | 線形回帰だけ |

実行アルゴリズム:

- **Almgren–Chriss** (2001): $\kappa=\sqrt{\lambda\sigma^2/\eta}$, $x(t)=x_0\sinh(\kappa(T-t))/\sinh(\kappa T)$。
- **TWAP**: 等分割。
- **mid / microprice**: 中値、$\alpha=q_{\mathrm{bid}}/(q_{\mathrm{bid}}+q_{\mathrm{ask}})$。

## 確率制御

| 対象 | 公式 | 文献 | 進む |
| --- | --- | --- | --- |
| Merton CRRA / Kelly | $\pi^*=(\mu-r)/(\gamma\sigma^2)$。Kelly は $\gamma=1$ | Merton (1971); Kelly (1956) | 1 資産 |
| 対数成長 | $r+\pi(\mu-r)-\frac12\pi^2\sigma^2$ | — | — |
| LQR | Euler $x^+=(I+A\Delta t)x+(B\Delta t)u$ の離散 Riccati。$K=(R\Delta t+B^\top PB)^{-1}B^\top P A_d$ | — | 連続 LQR の厳密解ではない |
| 1D HJB | 陽解法 + 中心差分。CFL $\mathrm{dt}\cdot\sigma^2/\mathrm{dx}^2\le 1/2$ | Fleming & Soner | `hjb_1d_implicit` / `kushner_dupuis_1d` |

## Malliavin / Itô–Taylor

> [!info] 文献
> Fournié, Lasry, Lebuchoux, Lions & Touzi (1999)。

スカラー拡散、正則格子。パスは Euler。

- 第一変分 $Y$: $dY=a_x Y\,dt+\sigma_x Y\,dW$（係数は中心差分）。
- Fournié 重み: $u=Y/(\sigma T)$, $\pi_\Delta\leftarrow\pi_\Delta+u\Delta W$。
- Greeks: $E[f(X)]$, $E[f(X)\pi_\Delta]$, $E[f(X)\pi_\Gamma]$。
- 密度: IBP `malliavin_density`。KDE は `kernel_density_mc`。
- 小ノイズ: ODE $\dot x=a$ に沿う $d_0=f(\bar x_T)$ と $d_1=\frac12 f''(\bar x)\int\sigma^2$。
- Itô–Taylor: $E[f(X_T)]\approx f+T\mathcal{L}f+\frac12 T^2\mathcal{L}^2 f$, $\mathcal{L}=a\partial_x+\frac12\sigma^2\partial_{xx}$。
