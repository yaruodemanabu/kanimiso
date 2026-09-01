# 乱数（amatsuki）

インターフェースは `Rng` / `SeedableRng` / `Distribution`。`rand` クレートは使わない。

## ChaCha8 + SplitMix64

> [!info] 文献
> Bernstein (2008) ChaCha。シード展開は Steele, Lea & Flood (2014) SplitMix64。

`seed_rng(seed: u64)` → `ChaCha8Rng`。

$u64$ シードを SplitMix64 の定数

$$
\gamma = \mathtt{0x9E3779B97F4A7C15},\quad
a = \mathtt{0xBF58476D1CE4E5B9},\quad
b = \mathtt{0x94D049BB133111EB}
$$

で 256-bit 鍵に展開する。nonce $=0$。ストリームは ChaCha の 8 ラウンド（4 double-round）。定数は `"expand 32-byte k"`、64-bit カウンタ（ワード 12–13）。

`next_f64` は上位 53 bit を $[0,1)$ に写す IEEE-754 慣習

$$
U = \frac{u \gg 11}{2^{53}}.
$$

> [!warning] オリジナルからの進む
> 8 ラウンド。split 操作は無い。`set_stream` / `jump_ahead` で並列用に切る。詳細は [[deviations]]。

## 分布

| 型 | 法則 | アルゴリズム | 文献 |
| --- | --- | --- | --- |
| `Open01` | $(0,1)$ | $(k+\tfrac12)/2^{53}$ | —（実装規約） |
| `OpenClosed01` | $(0,1]$ | $(u_{53}+1)/2^{53}$ | rand 0.8 互換（補った） |
| `Uniform[low,high)` | 連続一様 | $\mathrm{low}+\mathrm{span}\cdot U[0,1)$ | — |
| `StandardNormal` | $N(0,1)$ | Box–Muller | Box & Muller (1958) |
| `StandardNormalZiggurat` | $N(0,1)$ | 128 箱 Ziggurat | Marsaglia & Tsang (2000) |
| 離散 | Bernoulli / Binomial / Categorical / Multinomial | 逆変換 | — |
| `Normal(μ,σ)` | $N(\mu,\sigma^2)$ | $\mu+\sigma Z$ | — |
| `Exp1` | 標準指数 | $-\ln U$, $U\sim(0,1]$ | — |
| `Exp(λ)` | 指数（レート） | $\mathrm{Exp1}/\lambda$ | — |
| `Gamma` | 尺度 | Marsaglia–Tsang。$\alpha<1$ は $G_{\alpha+1}U^{1/\alpha}$ | Marsaglia & Tsang (2000) |
| `InverseGaussian` | IG | Michael–Schucany–Haas | Michael, Schucany & Haas (1976) |
| `StudentT(ν)` | $t$ | $Z/\sqrt{\chi^2_\nu/\nu}$, $\chi^2_\nu=\mathrm{Gamma}(\nu/2,2)$ | — |
| `Poisson(λ)` | ポアソン | $\lambda<30$: Knuth 逆変換。$\lambda\ge 30$: Hörmann PTRS | Knuth TAOCP 2; Hörmann (1993) |
| `sample_stable_cms` | 標準 $\alpha$-安定 | Chambers–Mallows–Stuck | Chambers, Mallows & Stuck (1976) |
| `Beta(α,β)` | $(0,1)$ | $X/(X+Y)$, $X\sim\mathrm{Gamma}(\alpha,1)$, $Y\sim\mathrm{Gamma}(\beta,1)$ | Devroye (1986) |
| `sample_dirichlet` | 単体 | $X_i\sim\mathrm{Gamma}(\alpha_i,1)$ を正規化 | Ferguson (1973) の有限次元 |

### Box–Muller

$$
R=\sqrt{-2\ln U},\qquad \Theta=2\pi V,\qquad Z=R\cos\Theta.
$$

正弦サンプルは捨てる（[[deviations]]）。

### Marsaglia–Tsang（$\alpha\ge 1$）

$$
d=\alpha-\tfrac13,\quad c=1/\sqrt{9d}.
$$

$x\sim N(0,1)$, $v=(1+cx)^3$。採択は原文の二段テスト。

### Hörmann PTRS

$\ln n!$ は Stirling

$$
\ln n! \approx \bigl(n+\tfrac12\bigr)\ln n - n + \tfrac12\ln(2\pi) + \frac{1}{12n} - \frac{1}{360n^3}.
$$

### Chambers–Mallows–Stuck

$\alpha=1$ は Cauchy 形。$\alpha\neq 1$ は $\zeta=-\beta\tan(\pi\alpha/2)$ の標準 CMS。
