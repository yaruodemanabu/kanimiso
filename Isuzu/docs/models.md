# モデルカタログ

記号はコードのパラメータ名。連続部分は [[simulation]]、推定は [[inference]]（点過程は [[point-processes]]）。拡散行列が 2 列のモデルは、相関 $\rho$ を Cholesky

$$
\begin{pmatrix}1&0\\ \rho&\sqrt{1-\rho^2}\end{pmatrix}
$$

として $\sigma$ に埋め込む。

| モデル | 方程式 | 文献 | 進む |
| --- | --- | --- | --- |
| GBM / Black–Scholes | $dS=\mu S\,dt+\sigma S\,dW$。厳密: $S\leftarrow S\exp((\mu-\frac12\sigma^2)\Delta t+\sigma\sqrt{\Delta t}Z)$ | Black & Scholes (1973) | — |
| OU / Vasicek | $dX=\kappa(\theta-X)\,dt+\sigma\,dW$ | Uhlenbeck & Ornstein (1930); Vasicek (1977) | — |
| CIR | $dX=\kappa(\theta-X^+)\,dt+\sigma\sqrt{X^+}\,dW$。Feller: $2\kappa\theta\ge\sigma^2$ | Cox, Ingersoll & Ross (1985) | Euler は $X^+$。`sample_exact` は非心 $\chi^2$ |
| Heston | $dS=\mu S\,dt+\sqrt{v}S\,dW^1$, $dv=\kappa(\theta-v)\,dt+\xi\sqrt{v}\,dW^2$。$\sigma=\begin{pmatrix}\sqrt{v}S&0\\ \xi\sqrt{v}\rho&\xi\sqrt{v}\sqrt{1-\rho^2}\end{pmatrix}$ | Heston (1993) | `qe_step` は Andersen (2008) |
| Hull–White | $dX=(\theta-\kappa X)\,dt+\sigma\,dW$（定数 $\theta$） | Hull & White (1990) | 時間依存 $\theta(t)$ は `hull_white_theta` |
| Black–Karasinski | $dr=r[\kappa(\theta-\ln r)+\frac12\sigma^2]\,dt+\sigma r\,dW$ | Black & Karasinski (1991) | Itô 形に直した |
| CKLS | $dX=(\alpha+\beta X)\,dt+\sigma\|X\|^\gamma\,dW$。CEV は $\alpha=0$ | Chan et al. (1992) | — |
| Jacobi | $dX=\kappa(\theta-X)\,dt+\sigma\sqrt{X(1-X)}\,dW$（状態は $[0,1]$ にクリップ） | 拡散の標準形 | 境界クリップは補った |
| Bessel | $dX=(\delta-1)/(2\|X\|)\,dt+dW$ | — | $\|X\|$ で 0 を避ける |
| Brownian bridge | $dX=(b-X)/(T-t)\,dt+\sigma\,dW$ | — | — |
| SABR | $dF=\alpha F^\beta\,dW^1$, $d\alpha=\nu\alpha\,dW^2$。ドリフト 0 | Hagan et al. (2002) | $\rho$ を Cholesky |
| 3/2 | $dv=\kappa v(\theta-v)\,dt+\xi v^{3/2}\,dW$ | Heston / Platen 系 | — |
| Stein–Stein | $dS=\mu S\,dt+\|v\|S\,dW^1$, $dv=\kappa(\theta-v)\,dt+\xi\,dW^2$ | Stein & Stein (1991) | — |
| Bates | Heston + 乗法 Merton ジャンプ $\Delta S=S(e^Z-1)$, $Z\sim N(\mathrm{jump}_\mu,\mathrm{jump}_\sigma^2)$ | Bates (1996) | — |
| Merton | $dX=\mu X\,dt+\sigma X\,dW$ + 乗法複合ポアソン | Merton (1976) | — |
| Kou | 同じ拡散 + 二重指数ジャンプ | Kou (2002) | — |
| fGBM | $dX=\mu X\,dt+\sigma X\,dB^H$ | Mandelbrot & Van Ness (1968) | 増分は Davies–Harte |
| fOU | OU 係数 + fBM 駆動 | — | 同上 |
| CARMA$(p,q)$ | $Y=b^\top X+c$, $dX=AX\,dt+e\,dL$。同伴行列の最終行は $-(a_p,\ldots,a_1)$ | Brockwell (2001) | ガウスなら最後座標の拡散は $\sigma$ |
| COGARCH$(p,q)$ | $V=a_0+a^\top Y_{t-}$, $dG=\sqrt{V}\,dL$, $dY=BY\,dt+eV(\Delta L)^2$ | Klüppelberg, Lindner & Maller (2004) | Euler 離散 |
| 線形状態空間 | $dX=(AX+b)\,dt+\sigma\,dW$, $Y=HX$ | — | Kalman–Bucy へ |
| `FnSde` | 任意の閉包ドリフト / 拡散 | YUIMA `setModel` 相当 | — |

Lévy パス `levy_path` は $X_{i+1}=X_i+L(\Delta t_i)$。`gamma_process` は VG 表現 $\theta=\mathrm{scale}$, $\nu=1/\mathrm{shape\_rate}$, $\sigma=0$。

エネルギーモデル（[[energy]]）: Schwartz 1-factor、Schwartz–Smith、Lucia–Schwartz、Cartea–Figueroa、Gibson–Schwartz、レジームスイッチ、スパークスプレッド / Margrabe、CRR スイング / ストレージ。
