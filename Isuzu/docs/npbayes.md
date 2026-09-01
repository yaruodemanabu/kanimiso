# ノンパラメトリックベイズ

実装は `src/npbayes/`。YUIMA には無い。文献の全文は [[citations]]、差分の全文は [[deviations]]。

## Dirichlet 過程

> [!info] 文献
> Ferguson (1973)。構成は Sethuraman (1994)。交換可能分割は Blackwell & MacQueen (1973)、Aldous (1985) CRP。切断は Ishwaran & James (2001)。

Ferguson: $G\sim\mathrm{DP}(\alpha,H)$ は、任意の可測分割 $A_{1:m}$ に対して

$$
(G(A_1),\ldots,G(A_m))\sim\mathrm{Dirichlet}(\alpha H(A_1),\ldots,\alpha H(A_m)).
$$

Sethuraman:

$$
V_k\sim\mathrm{Beta}(1,\alpha),\qquad
\pi_k=V_k\prod_{j<k}(1-V_j),\qquad
\theta_k\sim H,\qquad
G=\sum_{k=1}^\infty\pi_k\delta_{\theta_k}.
$$

**実装** `sample_stick_breaking`: 有限 $K$ で切り、残り質量を $\pi_K$ に載せる。

CRP（顧客 $i$、1-based）:

$$
P(\text{卓 }k)=\frac{n_k}{i-1+\alpha},\qquad
P(\text{新卓})=\frac{\alpha}{i-1+\alpha}.
$$

期待卓数 $E[K_n]=\alpha\sum_{i=1}^n 1/(\alpha+i-1)$。

## Pitman–Yor

> [!info] 文献
> Perman, Pitman & Yor (1992); Pitman & Yor (1997)。

$$
V_k\sim\mathrm{Beta}(1-d,\,\theta+kd),\qquad
P(\text{卓 }k)=\frac{n_k-d}{n-1+\theta},\qquad
P(\text{新})=\frac{\theta+Kd}{n-1+\theta}.
$$

$d=0$ で DP$(\theta,H)$ に戻る。

## Beta 過程と Bernoulli 過程

> [!info] 文献
> Hjort (1990) CRM。Thibaux & Jordan (2007) が $X_i\mid B\sim\mathrm{BeP}(B)$, $B\sim\mathrm{BP}(c,B_0)$ の周辺が IBP だと示した。

Hjort の Lévy 測度（実装していない構成）:

$$
\nu(d\pi,d\omega)=c\,\pi^{-1}(1-\pi)^{c-1}\,d\pi\,B_0(d\omega),\qquad \pi\in(0,1].
$$

有限近似（実装 `sample_beta_process_finite`）:

$$
\pi_k\stackrel{\mathrm{iid}}{\sim}\mathrm{Beta}\Bigl(\frac{c\gamma}{K},\,c\bigl(1-\tfrac{\gamma}{K}\bigr)\Bigr)\quad(K>\gamma).
$$

Bernoulli 過程（離散 $B$）:

$$
z_{nk}\mid\pi_k\sim\mathrm{Bernoulli}(\pi_k)\quad\text{独立}.
$$

IBP$(\alpha)$ は $c=1$, $\gamma=\alpha$ の Beta–Bernoulli（Thibaux & Jordan 2007, Prop. 3）。

## Indian buffet process

> [!info] 文献
> Griffiths & Ghahramani (2005, 2011)。stick-breaking は Teh, Görür & Ghahramani (2007)。

逐次レストラン（`sample_ibp_sequential`、$c=1$ のみ）: 顧客 $i$（1-based）は既存の皿 $k$ を確率 $m_k/i$ で取り、$\mathrm{Poisson}(\alpha/i)$ 個の新品を出す。

期待皿数 $E[K]=\alpha H_N$。

TGG stick-breaking（`sample_ibp_stick_breaking`）:

$$
v_i\sim\mathrm{Beta}(\alpha,1),\qquad \pi_k=\prod_{i=1}^k v_i,\qquad z_{nk}\mid\pi_k\sim\mathrm{Bernoulli}(\pi_k).
$$

有限 $K$ で尾を切る。

## DP ガウス混合の Gibbs

> [!info] 文献
> Neal (2000) Algorithm 3。混合は Antoniak (1974); Escobar & West (1995)。

$$
z\sim\mathrm{CRP}(\alpha),\qquad
\mu_k\sim N(\mu_0,\tau_0^2),\qquad
x_i\mid z_i=k\sim N(\mu_k,\sigma^2).
$$

$\sigma^2$ は固定（Neal §3 の分散事前は無い）。クラスタ $k$ の $n$ 点、標本和 $s$ に対する完全条件付き:

$$
\tau_n^{-2}=\tau_0^{-2}+n\sigma^{-2},\qquad
\mu_n=\tau_n^2\bigl(\mu_0\tau_0^{-2}+s\sigma^{-2}\bigr).
$$

周辺予測は $N(\mu_n,\sigma^2+\tau_n^2)$。新クラスタは $N(\mu_0,\sigma^2+\tau_0^2)$。

## HDP（Chinese restaurant franchise）

> [!info] 文献
> Teh, Jordan, Beal & Blei (2006) §4–5。

$$
G_0\sim\mathrm{DP}(\gamma,H),\qquad G_j\sim\mathrm{DP}(\alpha,G_0).
$$

実装はテーブルを明示する CRF。尤度は共役ガウス（論文の多項トピックではない）。新テーブルの皿は

$$
\frac{m_k}{m_\cdot+\gamma}f_k(x)+\frac{\gamma}{m_\cdot+\gamma}f_{\mathrm{new}}(x).
$$

## 線形ガウス IBP の collapsed Gibbs

> [!info] 文献
> Griffiths & Ghahramani (2011) §4.1。

$$
X=ZA+E,\qquad A_{kd}\sim N(0,\sigma_A^2),\qquad E_{nd}\sim N(0,\sigma_X^2).
$$

$M=Z^\top Z+(\sigma_X^2/\sigma_A^2)I$ として

$$
p(X\mid Z)=(2\pi)^{-ND/2}\,\sigma_X^{-(N-K)D}\,\sigma_A^{-KD}\,|M|^{-D/2}
\exp\Bigl(-\frac{1}{2\sigma_X^2}\mathrm{tr}\bigl(X^\top(I-ZM^{-1}Z^\top)X\bigr)\Bigr).
$$

既存特徴: $P(z_{nk}=1\mid Z_{-nk})=m_{-n,k}/N$。新特徴数 $\kappa$ は $0,\ldots,\kappa_{\max}$ を

$$
\mathrm{Poisson}(\kappa;\,\alpha/N)\,p(X\mid Z^{+\kappa})
$$

で列挙（非有界 MH ではない）。

有限モデル `finite_beta_bernoulli_gibbs`: $\pi_k\sim\mathrm{Beta}(\alpha/K,1)$ を $K$ 固定で Gibbs。
