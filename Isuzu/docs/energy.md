# エネルギー / コモディティ

`ocrs_iym` の energy / real-option OCR は、この環境の GitHub 資格情報では
読めない（private / 404）。実装は公開されている標準モデルに合わせた。

| モデル | 状態 | 文献 |
| --- | --- | --- |
| Schwartz 1-factor | $X=\log S$, $dX=\kappa(\alpha-X)\,dt+\sigma dW$ | Schwartz (1997) |
| Schwartz–Smith | $\log S=\chi+\xi$, $d\chi=-\kappa\chi\,dt+\sigma_\chi dW^\chi$, $d\xi=\mu_\xi dt+\sigma_\xi dW^\xi$ | Schwartz & Smith (2000) |
| Lucia–Schwartz | 短期 OU + フーリエ季節 $f(t)$ | Lucia & Schwartz (2002) |
| Cartea–Figueroa | $dX=-\alpha X\,dt+\sigma dW+J dN$, $S=e^X$ | Cartea & Figueroa (2005) |
| Gibson–Schwartz | $dS=(r-\delta)S\,dt+\sigma_S S dW^1$, $d\delta=\kappa(\alpha-\delta)\,dt+\sigma_\delta dW^2$ | Gibson & Schwartz (1990) |
| レジームスイッチ | 2 状態 GBM または OU + CTMC | Hamilton / energy regime texts |
| スパークスプレッド | $(S_{\mathrm{power}}-h S_{\mathrm{fuel}})^+$ | 発電の標準ペイオフ |

先物の閉じた式:

- Schwartz 1-factor: $F=\exp(e^{-\kappa\tau}X+(1-e^{-\kappa\tau})\alpha+\tfrac12\mathrm{Var}(X_T))$。
- Schwartz–Smith: $\log F=e^{-\kappa\tau}\chi+\xi+\mu_\xi\tau+\tfrac12\mathrm{Var}(\log S_T)$。
- Gibson–Schwartz: $F=S\exp(-H(\tau)\delta+A(\tau))$。

スイング / ストレージは CRR 動的計画（`crr_swing`, `crr_storage`）。
スパークスプレッドの解析ベンチは Margrabe（熱量を $S_2$ に吸収）。

線形状態は `SchwartzSmith::linear_state` → `LinearGaussian::from_linear_sde` →
`ssm_mle` / Kalman。リアルオプションの閾値は [[finance]]。
