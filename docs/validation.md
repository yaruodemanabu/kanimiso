# kanimiso の検証方針

この文書は、README の `Verified` 主張をどの証拠で支えているかをまとめたものです。
利用方法ではなく、数値結果を採用する前に根拠と限界を監査したい人を対象にします。

共通核は [`tsutsumi`](../tsutsumi/README.md) に移動しました。特殊関数・最適化・
線形代数の既存テストも同じ crate で再生し、kanimiso の同名モジュールは再公開です。
新しい注釈付き回帰、混合・加法モデル、線形 SHAP の外部オラクルと実測誤差は
[`number-ruler の検証仕様`](../number-ruler/docs/validation.md) にまとめています。
正規分布の上側 p 値は同 fixture の直接裾 oracle により Verified へ昇格しました。
サブクレートを含む仕様書とテストの対応表は
[カバー状況の監査メモ](spec_and_test_coverage.md) にある。実装状況の正本ではない。

## 状態の意味

| 状態 | 意味 |
|---|---|
| `Verified` | Tier 0 または Tier 1 の参照値に加え、対応する性質テストを持ち、CI で再生している |
| `Experimental` | 実装と内部テストはあるが、出自の異なる参照実装との照合など、必要な証拠が不足している |
| `Generated` | v0.1 互換の生成表面。active v0.2 台帳では使用しない |
| `Stub` | 未完成。active v0.2 台帳では使用しない |

実装一覧の正本は `kanimiso::coverage::inventory()`、Verified だけの正本は
`kanimiso::coverage::verified()` です。README の表はこの台帳の要約であり、公開型の
存在だけを根拠に機能を主張しません。

## 証拠の層

1. **Tier 0 — 外部または独立オラクル**: scipy / statsmodels の参照値、または Rust
   実装と異なる式・高精度演算で生成した決定的 JSON を再生します。
2. **Tier 1 — 閉形式・全列挙**: 小問題の解析解、HMM の全状態経路、厳密有理数などと
   比較します。
3. **Tier 2 — 性質**: 並べ替え・尺度・分割への不変性、確率和、単調性、再帰残差、
   失敗時の状態不変性を検査します。
4. **Tier 3 — ストレス**: 特異行列、非有限値、極端スケール、欠測、underflow / overflow
   の経路で、黙示的な置換や部分更新がないことを検査します。

Tier 0/1 の一点一致だけでは Verified にしません。式の読み違いが実装と自作オラクルに
同時に入る可能性があるため、独立性の種類を明記し、性質テストを併用します。

## 現在の主要オラクル

| 領域 | fixture / 参照 | 検査するもの | 独立性と現状 |
|---|---|---|---|
| 特殊関数 | `golden/special_functions.json` / scipy | beta・gamma、正規・χ² CDF、t・F CDF / p-value | 外部実装由来、1,099 ケース。正規・χ²の上側 p-value は未収録 |
| OLS | `golden/ols.json` / 80桁 Decimal | 係数、rank、共分散、推測、hat、Cook 距離 | 分解経路と異なる高精度方程式 |
| 状態空間 | `golden/state_space.json` / joint Gaussian conditioning | filter、Joseph 更新、尤度、RTS smoother、欠測 | Kalman 再帰と異なるブロック条件付け。ただし外部 package 由来ではない |
| Online RLS | `golden/online_rls.json` / batch normal equations | 係数、逆 Gram、予測、有効標本数 | 逐次 gain 更新と異なる一括式 |
| ARMA 共分散 | `golden/arma_acov.json` / Decimal Lyapunov | ACF / ACOVF、近単位根 | 実装の Yule–Walker 経路と異なる状態空間式 |
| 線形フィルタ | statsmodels source oracle + 閉形式 | BK / CF、IIR / FIR、MISO | 外部実装と再帰恒等式を併用 |
| GARCH | `golden/garch_qml.json` / Decimal | 尤度、勾配、境界 KKT、尺度不変性 | 高精度の別経路。外部 package の fit 結果との照合は今後も追加する |
| FIGARCH | `golden/figarch_qml.json` / Decimal | 係数再帰、有限 K、勾配、全境界面 | 論文式の高精度実装。外部 fit との照合は今後も追加する |
| FIEGARCH | `golden/fiegarch_qml.json` / Decimal | 固定値、QMLE、勾配、境界・尺度 | 論文式の高精度実装。外部 fit との照合は今後も追加する |
| Process MLE lite | `golden/process_mle.json` / dense Decimal GLS | median-gap の 1 / 2 / 4 / 8 / 16 倍という 5 range 候補での profile likelihood、並べ替え・affine 不変性 | 各候補は solver と異なる dense GLS 経路。range の連続最大化ではない |
| HMM | `golden/hmm.json` / Decimal 全経路列挙 | 尤度、Viterbi、Baum–Welch | 小問題では強いが同一仕様解釈のリスクが残るため Experimental |
| CART (`oldwood`) | `oldwood/golden/sklearn_cart.csv` / scikit-learn 1.7.2 + 解析解 + 独立に書いた全候補 brute-force | weighted Gini / entropy / SSE、probe prediction / probability、split、leaf value、tie-break | 外部 fixture は3ケースを `0.0` 絶対許容差で再生。threshold・child index・leaf ID の sklearn parity は対象外。brute-force は prefix/suffix accumulator と異なる候補ごとの再集計 |
| Forest / boosting (`mayoi-no-mori`) | verified CART kernel + ensemble の閉形式・性質 | random forest / ExtraTrees / GBDT / AdaBoost / Isolation Forest の seed 再現性、確率和、OOB、bin 単調性、ordered statistic の target leakage 不在 | 全 estimator が Experimental。外部 ensemble fixture は未収録。LightGBM / CatBoost 名は明記された subset であり upstream parity は未主張 |

各 fixture の隣にある `scripts/*_oracle.py` は、schema / provenance の確認と、より高い
精度での再生成差分を担当します。Python は fixture 生成・監査時だけ必要です。

## 許容差

許容差は一律の `1e-6` ではありません。次の手順で固定します。

1. オラクルごとに複数精度で fixture を再生成し、オラクル側の収束を確認する。
   多くは80桁と120桁、OLS・状態空間・HMMなどは180桁も照合する。
2. Rust replay の最大絶対誤差・相対誤差を Linux と Windows で測る。
3. 実測値の概ね3〜4倍を回帰検出用の閾値にし、テストのコメントへ測定値を残す。

特殊関数は関数ごとに最大誤差を測り、近似精度が異なる関数を一つの緩い閾値へまとめ
ません。代表例として、状態空間 fixture の Rust replay は最大絶対誤差 `3.553e-15` に対して
`1.5e-14`、Online RLS は最大絶対誤差 `6.218e-15` と最大相対誤差 `2.616e-14`
に対してそれぞれ約4倍の許容差を使います。これらは API 全体の一般的な精度保証では
なく、コミット済み fixture に対する回帰境界です。

## 失敗経路も検証対象にする

数値が一致するだけでは十分ではありません。以下も受入条件です。

- 非有限・shape 不整合・算術 overflow の後にオンライン推定器の状態が変わらない
- support 外の対数確率を floor せず `-∞` として扱う
- 不要な逆行列を作らず、到達可能な特異ケースを成功させる
- 定数ターゲット、識別不能、境界解、非収束を成功値として隠さない
- fallback、jitter、ridge、clamp を採用した場合は `NumericalCompromise` を残す

## CI での再生

`.github/workflows/ci.yml` は、特殊関数、OLS、状態空間、Online RLS、時系列、
ボラティリティ、CART / ensemble、tsutsumi / number-ruler を別ジョブに分けています。全ジョブが対応する Rust replay を実行し、
Python オラクルを持つジョブは fixture の provenance または多倍長精度差分も検査します。
特殊関数ジョブは scipy 由来のコミット済み JSON を Rust から再生します。さらに MSRV、
rustfmt、警告を error にした rustdoc、依存境界、cargo-deny、冗長性・Clippy の減少専用
ratchet を検査します。line coverage は Linux と Windows の両方で実測 `76.30%`、
下限は `76.0%` です。レポート生成だけでなく、下限を割る退行は失敗させます。

ローカルで全 Rust テストを再生する最短のコマンドは次です。

```console
cargo test --workspace --all-features --locked
```

個別の Python deep check は CI workflow に記載されたコマンドをそのまま利用できます。

## 既知の限界

- 独自 Decimal オラクルは実装経路を分けていますが、仕様解釈まで第三者から独立して
  いるとは限りません。GARCH 族と状態空間には外部実装由来の相互運用 fixture を追加
  する余地があります。
- HMM は全列挙と高精度再生を持ちますが、hmmlearn など出自の異なる fixture がまだ
  ないため Experimental のままです。
- `mayoi-no-mori` は verified な CART 核を共有しますが、ensemble 全体を外部実装と
  照合する fixture が未収録のため、random forest から LightGBM / CatBoost 系 subset
  まで Experimental のままです。
- Verified は記載した入力領域と性質に対する証拠レベルです。任意のデータで統計的に
  適切なモデル選択が保証される、という意味ではありません。
- v0.2 は alpha 前で、API 安定性を保証しません。

新しい Verified 項目を追加する場合は、fixture または閉形式、性質テスト、実測に基づく
許容差、CI の再生経路を同じ変更に含めます。
