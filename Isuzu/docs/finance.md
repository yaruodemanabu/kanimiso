# 金融工学（Shreve I / II）

過程の生成は `models` / `simulate`。このノートは `src/finance/` の価格・ヘッジ・停止・金利・ジャンプ・リアルオプション。

> [!info] 文献
> Shreve, *Stochastic Calculus for Finance* I / II。Black & Scholes (1973)。
> Cox, Ross & Rubinstein (1979)。Longstaff & Schwartz (2001)。
> Andersen & Broadie (2004)。Dixit & Pindyck (1994)。McDonald & Siegel (1986)。
> Merton (1976)。Kou (2002)。Vasicek (1977)。Cox, Ingersoll & Ross (1985)。
> Hull & White (1990)。Gibson & Schwartz (1990)。Schwartz (1997)。
> Cont & Voltchkova (2005)。Margrabe (1978)。Jaillet–Ronn–Tompaidis (2004)。

## 解析解

ATM コール $S=K=100$, $r=0.05$, $\sigma=0.2$, $T=1$ は $C\approx 10.45058357$。
プットコールパリティ $C-P=Se^{-qT}-Ke^{-rT}$。$\Phi$ は級数 + 相補誤差関数の連分数。

幾何アジアは Kemna–Vorst の対数正規閉じた式。デジタルは $e^{-rT}\Phi(d_2)$。

## モンテカルロ

割引期待値 + Welford 平均 / 分散。対偶パスは同じ $\Delta W$ の符号反転。
ヨーロピアン制御変量は GBM のとき BS 平均を使う。重要度サンプリングは
Girsanov $Z=\exp(-\theta W_T-\tfrac12\theta^2 T)$。

## ツリー / PDE / LSM

- CRR: $u=e^{\sigma\sqrt{\Delta t}}$, $d=1/u$, $p=(e^{r\Delta t}-d)/(u-d)$。
- PDE: スポット格子の $\theta$ 法。陽解法は CFL $\mathrm{dt}\,\sigma^2 S^2/\mathrm{d}S^2\le 1/2$。
  クランク–ニコルソンは Rannacher で最初の数段を陰的。American は PSOR。
- LSM: ITM だけを多項式 / Laguerre / Hermite に回帰し、列を標準化する。
  双対上界は割引ヨーロピアン・プットをマルチンゲールにした Haugh–Kogan 形
  （入れ子 Andersen–Broadie は要約）。
- スイング: CRR 上の多重停止（日付あたり 1 回、権利数 `Q_{\max}`）。
  ストレージ: 在庫 $q$ で注入 / 引出 / 保有。
- 3 項: Kamrad–Ritchken $\lambda=\sqrt{3}$。

## 金利・ジャンプ

Vasicek / CIR のアフィン債、Jamshidian 形の Vasicek 債券オプション、
Black キャップレット / スワップション、Hull–White $\theta(t)$ はフォワードの数値微分。
Merton 級数、IMEX 対数格子 PIDE（局所項は陰的、ジャンプ畳み込みは陽的）、
ジャンプ MC。Kou は二重指数を正規混合で近似。Esscher は正規ジャンプの傾き。
$T$-フォワード価格はキャッシュ価格 $/\,P(0,T)$。Margrabe 交換は
$\sigma=\sqrt{\sigma_1^2+\sigma_2^2-2\rho\sigma_1\sigma_2}$。

## リアルオプション

Dixit–Pindyck の永久コール $S^*=\beta/(\beta-1)I$、放棄の負根、
Dixit の参入 / 撤退の近似対、有限期限は配当付き BS コール。
