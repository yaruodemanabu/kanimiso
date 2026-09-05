# number-ruler

**分析の最初に使う、注釈付き回帰モデルの Pure Rust crate。**

係数だけでなく、「どの仮定で意味を持つか」「何をまだ言えないか」「次に何を確認するか」
を結果へ同伴させます。`fit(&self, ...)` は推定器を変更せず、`Qualified<T>` と
`annotations` を返します。数値計算は `tsutsumi`、品質基準は `signlred::Policy` に集約しています。

## 最初の分析

```rust
use number_ruler::{LinearModel, Matrix, Vector, Session, linear_shap};

let x = Matrix::from_fn(24, 1, |i, _| i as f64 / 4.0);
let y = Vector::from_iter((0..24).map(|i| {
    1.0 + 0.7 * x.get(i, 0) + ((i * 7) % 11) as f64 / 10.0
}));
let session = Session::new("first_analysis", "fit");
let result = LinearModel::default().fit(&x, &y, &session)?;

for coefficient in &result.value.diagnostics.coefficients {
    println!("{}: {} (SE {:?}, p {:?})", coefficient.name,
        coefficient.estimate, coefficient.standard_error, coefficient.p_value);
}
for note in &result.value.annotations {
    println!("{:?}: {}\n次の確認: {}", note.topic, note.statement, note.action);
}
for issue in result.report.issues() { eprintln!("{issue}"); }

// 説明用背景を明示する。実際の検証では学習・説明・評価データを区別する。
let explanation = linear_shap(&result.value, &x, &x, &session)?;
println!("SHAP baseline: {}", explanation.value.base_value);
# Ok::<(), Box<dyn std::error::Error>>(())
```

未公開・alpha API です。リポジトリで `cargo test -p number-ruler` を実行できます。
組み込みには `number-ruler = { path = "../kanimiso/number-ruler" }`、または採用した
Git commit を固定した依存を使ってください。`kanimiso::number_ruler` からも同じ API を利用できます。

## 対応範囲と制限

| モデル | API | 現在実装している範囲 |
|---|---|---|
| LM | `LinearModel` | OLS、t 検定、モデルベース共分散、leverage、Cook 距離 |
| GLM | `GeneralizedLinearModel { family, .. }` | Gaussian/identity、Bernoulli/logit、Poisson/log、共通 IRLS |
| LMM | `MixedModel::default()` / `LinearMixedModel` | Gaussian ランダム切片、ML / REML、BLUP |
| GLMM | `MixedModel::generalized(family)` / `GeneralizedMixedModel` | Bernoulli / Poisson ランダム切片、周辺 ML、求積の倍次数チェック |
| LAM | `AdditiveModel` / `LinearAdditiveModel` | 指定 knot の連続区分線形スプライン、中心化・尺度化 |
| GAM | `GeneralizedAdditiveModel` | 同じ基底と GLM、任意の非負 L2 penalty、term effect |
| SHAP | `linear_shap` | 線形予測子の interventional SHAP、背景平均を明示 |

Family を切り替える一つの実装を用い、モデル名ごとに IRLS や最適化を複製しません。
別名は互換エイリアスであり、別の学習核ではありません。

GLM の Binomial は **0/1 の単一試行**、Poisson は **非負整数・単位 exposure** です。
offset、試行数、頻度/解析重み、負の二項分布、一般リンク、欠損補完は未対応です。
LMM / GLMM はランダム傾き、多階層・交差効果、分散共分散構造の自動探索には対応しません。
GLMM の局所最適化と求積チェックは大域最適性や積分誤差上界の証明ではありません。

加法モデルの penalty は **中心化・尺度化した基底係数への L2** です。微分に対する
粗さ penalty、三次スプライン、REML/GCV による平滑化選択、自動 knot 選択ではありません。
`terms` を明示し、必要なら別の検証データで penalty を決めてください。

## 結果の読み方

- `report`: 非有限値、rank、条件数、識別不能、数値的妥協、意味のなさを機械可読で保持。
  不正な寸法や応答領域は、abort 閾値を緩めても計算へ進めません。
- `diagnostics`: 係数、標準誤差、p 値、条件数、残差自由度、deviance、Pearson dispersion、
  leverage、収束を確認。p 値は **未調整・モデル条件付き** です。
- `annotations`: 因果識別をしていないこと、選択後推論・多重比較、独立性・等分散性、
  分布・リンク・群構造・外挿の制限と推奨確認を読みます。
- 罰則付き fit、混合モデルの固定効果・分散成分の未検証 Wald 検定は、便宜的な
  p 値を作らず保留します。ロバスト/cluster 標準誤差や予測区間も未実装です。
- 混合モデルは `predict_marginal`（新しい群の効果を積分）と
  `predict_conditional`（既知群の BLUP / 事後平均を代入）を区別します。
  非線形リンクで事後平均を代入した予測は、事後予測平均ではありません。
- SHAP は因果効果でも conditional SHAP でもありません。logit / log モデルでは
  **log-odds / log-mean** 上の加法分解で、確率や count 上の加法分解ではありません。

少標本、強い共線性、完全分離、収束失敗を「とりあえず係数が出た」と扱わない設計です。
成功しても、注釈を省略して p 値だけで結論を出さないでください。

## 検証

`golden/regression.json` と `scripts/regression_oracle.py` に外部由来データ・再生成手順を同梱。
statsmodels の LM / GLM / Gaussian MixedLM と、SciPy の適応積分による GLMM 尤度を
照合します。加法モデルは閉形式縮約・再構成、SHAP は全 coalition 列挙、その他は
行置換・群再ラベル・入力失敗等をテストします。詳しい誤差と未検証範囲は
[検証仕様](docs/validation.md) に分離しています。

公開範囲を網羅する同等性テストではありません。混合・加法モデルは外部照合がある
部分を含みますが、ストレス条件の網羅が十分でないため Experimental と扱います。

参考: [statsmodels MixedLM](https://www.statsmodels.org/stable/mixed_linear.html)、
[SHAP LinearExplainer](https://shap.readthedocs.io/en/latest/generated/shap.LinearExplainer.html)。

Rust 1.85、Apache-2.0、`#![forbid(unsafe_code)]`。Python は検証データ生成時のみ使用し、
実行時の科学計算依存は `tsutsumi` 経由の faer だけです。
