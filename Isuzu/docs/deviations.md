# オリジナルからの進む・補った箇所

「正しい文献」を付けたうえで、論文の式をそのまま写していない箇所をここに全部書く。意図的な省略も、数値上の便宜も、YUIMA 互換のための規約も隠さない。

## 乱数 [[rng]]

- **ChaCha** は Bernstein (2008) の 8 ラウンド（ChaCha8）。論文の主推奨は ChaCha20。nonce は 0、カウンタは 64-bit。
- **SplitMix64** は Steele et al. (2014) の定数を使うが、Java `SplittableRandom` の split 操作は実装していない。`u64` シードを 256-bit 鍵に伸ばすためだけ。
- **Box–Muller** は 1 サンプルあたり余弦だけを使い、正弦は捨てる。Box & Muller (1958) は 2 標本を同時に出す。スレッドローカルキャッシュは無い。`StandardNormalZiggurat` は Marsaglia–Tsang 128 箱（シード列は Box–Muller と互換ではない）。
- **ChaCha `set_stream` / `jump_ahead`** は 64-bit nonce とブロックカウンタ。rand クレートの `ChaCha8Rng` とビット互換ではない。
- **Marsaglia–Tsang** は原文どおり。`shape<1` の $G_{\alpha+1} U^{1/\alpha}$ は彼らが述べている標準拡張。
- **Hörmann PTRS** の $\ln n!$ は Stirling 打ち切り（$1/n$ と $1/n^3$）。Hörmann (1993) はより精密な階乗評価を許す。
- **Beta** は Devroye (1986) の「二つの独立 Gamma の比」。Johnk / Cheng の直接法ではない。
- **一様 $(0,1)$** は 53-bit 中点 $(k+\tfrac12)/2^{53}$。端点クリップ（$2^{-1074}$）ではない。
- **一様 $(0,1]$** は rand 0.8 の `OpenClosed01` とビット互換になるよう補った。Bernstein / Box–Muller の原文には無い。
- **ChaCha8 ゼロ鍵** の先頭ブロックは既知答テストで固定している（ChaCha20 RFC ベクトルではない。8 ラウンド）。

## 線形代数・最適化 [[linalg-optimize]]

- **`expm`** は Higham (2008) の scaling-and-squaring だが、Padé ではなく Taylor 27 項。残差評価もスケーリング定数の Higham 最適化もしていない。
- **固有値** は faer。Hurwitz 判定は $\mathrm{Re}\,\lambda < 0$ の数値閾値 $10^{-14}$。
- **SPD 正則化** はフィルタの共分散に $\varepsilon I$ を最大 12 回足す。**QMLE の準尤度では禁止**：非 SPD は $-\infty$。監査前は QL にもジッタが入っていた。
- **Nelder–Mead** は $\alpha=1,\gamma=2,\rho=0.5,\sigma=0.5$（Nelder & Mead 1965 の通例）だが、箱制約は座標クリップで補った。初期単体は各座標に相対 5%。収束判定は自前の `ftol` / `xtol`。

## シミュレーション [[simulation]]

- **Euler–Maruyama / Milstein / KP1.5** は格子が非一様でも $\Delta t_i$ をそのまま使う。Kloeden–Platen (1992) の本文は一様格子の誤差定理。
- **Milstein** の交差項は可換ノイズの Lévy 面積 $\frac12(\Delta W_j\Delta W_k-\delta_{jk}\Delta t)$。$n>1$ かつ $m>1$ は Lévy 面積が無いので `Unsupported`（Heston の $\rho\neq 0$ など）。Jacobian が無ければ拡散の前進差分。
- **KP1.5** はスカラーのみ。Kloeden–Platen (10.4.1) の $I_{(1,1,1)}=(\Delta W^3-3\Delta t\Delta W)/6$ を含む。$\Delta Z=\frac12\Delta t(\Delta W+\sqrt{\Delta t/3}\,Z)$。係数微分は $\varepsilon=\max(10^{-4}(1+|x|),10^{-8})$ の中心差分。
- **CIR** は Euler では $X^+$ で反射。`Cir::sample_exact` は Glasserman の $X'=c\chi^2_d(\lambda)$（$c=\sigma^2(1-e^{-\kappa\Delta})/(4\kappa)$）。Feller 条件は検査するだけ。
- **Heston / SABR** の相関は拡散行列に Cholesky を埋め込む。`Heston::qe_step` は Andersen (2008) の二次 / 指数スイッチ。SABR に QE は無い。
- **fBM** は Davies–Harte / Wood–Chan の標準埋め込み $m=2n$（$\gamma(n)$ を含む）。2 のべきゼロ埋めはしない。固有値が負なら `Err`（クリップしない）。
- **COGARCH** は Euler + Lévy 増分。Klüppelberg et al. の厳密な離散化ではない。

## 推定 [[inference]]

- **QMLE** は Euler ガウス準尤度（Yoshida 1992 / Kessler 1997 の高頻度漸近ではなく、YUIMA `qmle` と同じ Euler コントラスト）。ジャンプ過程と $H\neq 1/2$ は拒否。最適化は Nelder–Mead であり、論文の Newton / scoring ではない。1 次元はスカラー高速経路。
- **情報量** `Fit::qbic` は Eguchi–Masuda の $\mathrm{QBIC}=-2\mathrm{QL}+\log\det(-H)$（Hessian が負定でなければ増分 BIC に落とす）。古い「増分個数の BIC を qBIC と呼ぶ」実装は `bic_increments`。`cic` は 1 パスのスコア外積 $\mathrm{tr}(\hat H^{-1}\hat G)$（Uchida の漸近 CIC の有限標本形）。
- **L-BFGS-B** は有限差分勾配 + 箱射影。Byrd–Lu–Nocedal–Zhu の Cauchy 探索や部分空間制約は無い。
- **QGV Hurst** は 2 スケール二次変分比。OU のような局所ブラウン近似に使う。fBM の最適推定ではない。
- **adaBayes** は $\exp(\mathrm{QL}(\theta))\,N(\theta;\mu_{\mathrm{prior}},\mathrm{diag}(\sigma)^2)$ に対するランダムウォーク MH。既定の $\mu$ は初期値 `start`。YUIMA の適応提案でも Uchida–Yoshida の適応ベイズでもない。
- **Wald** の Hessian は中心差分。情報行列の理論式ではない。$p$ 値は Abramowitz–Stegun 7.1.26 の $\mathrm{erfc}$ 近似。
- **変化点** は格子走査。YUIMA `CPoint` の連続時間議論を離散格子に落としている。

## 点過程 [[point-processes]]

- **Ogata thinning** は強度の一様上界をユーザが渡す。適応上界の更新は指数 Hawkes の再帰状態だけ。
- **自己補正** の thinning 上界は区間右端の強度。Isham–Westcott (1979) は強度の定義だけ。
- **CIR Cox** は CIR を Euler し、小区間の中点強度で斉次ポアソンを重ねる。正確な Cox 積分は無い。
- **CARMA-Hawkes** の補償項は $A$ が与えられたときの閉形式 $\mu\Delta+b^\top\int_0^\Delta e^{As}x\,ds$（`integrate_expm`）。$p\ge 2$ の thinning は短い horizon 上の $\|b\|_1\|x\|_2 e^{\|A\|_F h}$ 上界。専用 MLE は Nelder–Mead。
- **抑制 Hawkes** は $\alpha<0$ で強度を正にクリップ。Hawkes (1971) の線形励起の外挿。

## フィルタ [[filters]]

- **Kalman–Bucy / `from_linear_sde`** は Van Loan (1978) のブロック指数で $F,u,Q$ を厳密離散化する。YUIMA と同じく最初の観測時刻では更新しない。更新は古典形 $P\leftarrow(I-KH)P^-$（Joseph ではない）。連続生成元を持つ `LinearGaussian` は不規則 $\Delta t$ で $F,Q$ を作り直す。
- **離散 KF** は Joseph。古典形と混在している（KB と離散で違う）。
- **IEKF** は Bell–Cathey の Gauss–Newton 形だが、反復回数は `IekfConfig` の固定上限。
- **二次 EKF** は観測平均に対角二階だけを足す。完全な二次項ではない。
- **UKF** 既定 $\alpha=1,\beta=2,\kappa=0$。Julier–Uhlmann の $\kappa=3-n$ でも van der Merwe の小さな $\alpha$ でもない。
- **EnKF** は摂動観測の確率的 EnKF。決定論的 ETKF / LETKF ではない。
- **RPF** の帯域は Silverman 則。Musso et al. が議論する他の核は無い。
- **UPF** は各粒子に 1 ステップ UKF。van der Merwe et al. (2000) の提案を、Isuzu の UKF 既定パラメータで動かしている。
- **FFBSi** は後退パス本数を引数で制限する。Godsill et al. の完全平滑化ではない場合がある。
- **再標本** の一様乱数は `next_f64`（`[0,1)`）。Kitagawa (1996) の $(0,1)$ 開区間とは端点が違う。
- **粒子周辺尤度** は正規化対数重み $-\ln N$ から log-sum-exp。APF は $+\ell_1+\ell_2-\ln N$。

## 高頻度・制御・Malliavin [[hft-control-malliavin]]

- **`hy_avar` 非対角** は出典のない帯域 $n^{0.45}$ の自作デルタ法であり、**experimental**（`experimental_hy_avar`）。対角は $(2/3)\mathrm{RQ}$。Hayashi–Yoshida 2011 §8.2 の矩形カーネル族そのものではない。
- **`preaveraged_hy`** は BNHLS リフレッシュ時計で同期してから前平均する。同じ添字同士を掛けていた旧実装は `indexed_preaverage_cov`。$g(x)=x\wedge(1-x)$、$\psi=\int g^2=1/12$。
- **realized kernel** は Tukey–Hanning だけ。
- **Almgren–Chriss** は一時衝撃が線形・定数係数の閉形式。原文の一般コスト汎関数ではない。
- **1D HJB** は陽解法 + 中心差分 + 有限行動格子。$\mathrm{dt}\cdot\sigma^2/\mathrm{dx}^2>1/2$ なら `Err`。陰解法は `hjb_1d_implicit`、Markov 連鎖近似は `kushner_dupuis_1d`。
- **Fournié $\Delta$** は Euler パスの第一変分。$\Gamma$ は離散 Skorohod $\delta(u)^2-\|u\|_{L^2}^2$（かつ GBM では $u$ の $x_0$ 依存を補正）。旧実装の $\pi_2$ に付いていた $/\Delta t$ は分散がステップ数で発散する誤りだった。`malliavin_density` は IBP、KDE は `kernel_density_mc`。
- **小ノイズ** は ODE に沿う $(d_0,d_1)$ だけ。より高次の Wiedemann / Freidlin–Wentzell 展開は無い。

## ノンパラメトリックベイズ [[npbayes]]

- **Sethuraman stick-breaking** は有限 $K$ で切断し、残り質量を最後の原子に載せる（Ishwaran & James 2001 の residual-atom）。無限和そのものではない。GEM の正確な有限次元 Dirichlet でもない。
- **CRP / 二パラメータ CRP** は交換可能分割としては原文どおり（切断なし）。
- **Hjort の Beta 過程** の Lévy 測度からの CRM 構成（逆 Lévy / Poisson 点過程）は実装していない。
- **有限 Beta 過程** は $\pi_k\sim\mathrm{Beta}(c\gamma/K,\,c(1-\gamma/K))$。$c(1-\gamma/K)\le 0$ のときは Griffiths–Ghahramani の $\mathrm{Beta}(\gamma/K,1)$ に落とす（戻り値 `used_gg_fallback`）。
- **逐次 IBP** は $c=1$ だけ。Teh–Görür の 3 パラメータ IBP のレストラン過程は無い。
- **TGG stick-breaking IBP** は有限 $K$ で尾を切る。原子の印 $\omega_k\sim H$ は特徴の添え字に退化。
- **Neal (2000) Alg. 3** は観測分散 $\sigma^2$ を固定。$\sigma^2$ の共役事前は無い。ラベルスイッチ補正は無い。
- **HDP CRF** は Teh et al. (2006) の座席過程そのものだが、尤度は論文の多項トピックではなく、共役ガウス。$\sigma^2$ 固定。§5.3 の direct assignment（テーブルを持たない）は実装していない。
- **IBP collapsed Gibbs** は GG (2011) §4.1 の周辺尤度。新しい特徴数 $\kappa$ は $\kappa_{\max}$ までの列挙（既定 4）であり、非有界 Poisson に対する MH ではない。$\sigma_X,\sigma_A,\alpha$ は固定。$A$ は $Z$ のスイープ中は積分し、最後に完全条件付きから 1 回引く。初期値は空行列ではなく逐次 IBP の 1 本。
- **有限 Beta–Bernoulli Gibbs** は $K$ 固定。$K\to\infty$ の IBP 極限は取らない。
- **Hamming / クラスタ精度** のベンチは貪欲列マッチと多数決ラベルであり、Hungarian 最適代入ではない。
- **Pitman–Yor stick-breaking** は $V_k\sim\mathrm{Beta}(1-d,\theta+kd)$（$k=1,2,\ldots$）。$\theta+(k-1)d$ は誤りだった。

## 2026-08-30 監査で直した「逸脱ではなく誤り」

意図的な省略ではなく、数式・定義が間違っていたもの。回帰は `tests/audit_oracles.rs` と各モジュールの単体テスト。

| 箇所 | 誤っていた内容 | 今の定義 |
| --- | --- | --- |
| 粒子フィルタ周辺尤度 | 初期化・再標本後 `log w=0` のまま LSE → $+T\ln N$ | `log w=−\ln N` |
| CMS 安定 | $\sin(u+\xi)$、前係数なし | $\sin(\alpha(u+\xi))$ と $(1+\zeta^2)^{1/(2\alpha)}$（Nolan S0） |
| 線形 SDE の $Q$ | $F=e^{A\Delta t}$ なのに $Q=\sigma\sigma^\top\Delta t$ | Van Loan ブロック指数 |
| KP1.5 | $I_{(1,1,1)}$ 欠落、$\varepsilon=10^{-6}$ | (10.4.1) の三重積分、より大きい差分 |
| Malliavin $\Gamma$ | $\pi_2$ に $/\Delta t$（分散が $n$ で発散） | $\delta(u)^2-\|u\|^2$ |
| Davies–Harte | $2$ のべき埋め + $\gamma(n)$ 欠落 → $H\gtrsim 0.85$ で失敗 | $m=2n$、$\gamma(n)$ 込み |
| `refresh_times` | 時刻の完全一致だけ | BNHLS リフレッシュ時計 |
| `gamma_process` | `VarianceGamma{sigma:0}` → 常に Err | `LevyMeasure::Gamma` |
| `time_series_folds` | テストより未来で訓練 | expanding window |
| `reflect_nonnegative` | ドキュメントは反射、実装は NaN 検査だけ | $x\leftarrow\|x\|$（既定はオフ） |
| `Scheme::Exact` | 常に Unsupported | `Sde::exact_step`（GBM / OU） |
| `Beta` 微小形状 | `U^{1/\alpha}` アンダーフローで panic | 対数空間 |
| `try_inverse` | 特異で $\pm\infty$ を `Some` | 有限性 + $A\hat A\approx I$、さもなくば `None` |
| `hjb_1d` | CFL 無視で $\pm 10^{11}$ | 安定条件違反は `Err` |
| Weibull 尤度 | 右打ち切り生存項なし、`mle` なし | 生存項 + Nelder–Mead |
| 情報フィルタ | $P^-$ を 2 回 LU | 1 回 |
| `square_root_kalman` | 名前が QR SRKF | Joseph + Cholesky 検査と明記 |

## 金融 / エネルギー（2026-08-30 D 節・Shreve 提案）

- **Black–Scholes $\Phi$** は自前の誤差関数（級数 + 連分数）。Cephes / Boost ではない。
- **BS PDE** は一様スポット格子。対数格子・非一様格子は無い。
- **LSM 既定基底** は $1,S/K,(S/K)^2$。Longstaff–Schwartz の重み付き Laguerre は $S/K\in(0,1)$ で早期行使を過小評価するのでオプション。
- **LSM 双対** の既定オラクルは割引ヨーロピアン・プットをマルチンゲールにした Haugh–Kogan 形。Andersen–Broadie の入れ子は 1 段の要約であり、完全なマルチンゲール最大化ではない。
- **Kou** は閉じた変換ではなく正規混合でベンチマークする。
- **Merton PIDE** は Cont–Voltchkova の IMEX（局所拡散は陰的、ジャンプ畳み込みは陽的）。非一様格子ではない。
- **Hull–White $\theta(t)$** は割引債の数値微分。連続な瞬間フォワードの厳密フィットではない。
- **PMMH / SMC²** は定数提案分散の RWMH と 1 段の外側粒子。適応 SMC² のリフレッシュは無い。
- **Shumway–Stoffer EM** は時不変 $F,Q,H,R$。欠測は革新を飛ばす。
- **エネルギー先物** はリスクプレミアムをパラメータに吸収した物理測度の式。OCR 本 `ocrs_iym` は読めなかった。
- **3 項ツリー** は Kamrad–Ritchken $\lambda=\sqrt{3}$。JR / Tian の他パラメータは無い。

まだ残っている意図的ギャップ: Uchida–Yoshida の適応ベイズ、非可換 Lévy 面積 Milstein、QR / Potter 平方根カルマン。`spd_regularize` はフィルタ側に残る。
