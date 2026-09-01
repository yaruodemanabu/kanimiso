# kanimiso

Pure Rust の機械学習・統計クレート。線形代数は `faer` のみ。`unsafe` なし。

v0.2 は小さい検証済み核から再構成中である（`AGENTS.md`）。README の機能主張は
`kanimiso::coverage::verified()` に連動する。`0.2.0-alpha.1` までは
scikit-learn / statsmodels / sktime / tslearn / hmmlearn / river 相当という表記はしない。

## ワークスペース

| クレート | 役割 |
|---|---|
| **kanimiso** | 推定・変換・予測・予測子。計算そのもの。 |
| **signlred** | 機械学習・線形計算の**結果と推測の品質**に責任を持つエラー処理。特異・ランク落ち・分離・定数ターゲット・識別不能・意味のないフィットをコード化して abort / 警告する。 |
| **ojizou-san** | 同上の品質に責任を持つログ。数値計算上の妥協（意図した計算 vs 実際に走った計算）、意味のないフィット、追加学習の説明（何が動いたか・なぜ・識別は残っているか）を台帳に残す。 |

普通の `thiserror` / `log` ではない。計算結果を捨ててよい数字として返すことを拒否する。

## 品質契約

1. `fit` / `predict` / `partial_fit` は必ず `signlred::Qualified<T>`（値 + `Report`）または `Failure` を返す。
2. 数値的妥協（擬似逆、リッジフォールバック、ジッタ、打ち切り SVD）は `NumericalCompromise` として記録される。
3. 統計的に空の計算（定数ターゲット、ランク 0、単一クラス分類器、補間だけの \(R^2=1\)）は `Meaninglessness` を付け、Vacuous / False なら abort する。
4. 追加学習は `ojizou_san::IncrementalExplain` なしでは完了しない。`n_eff`、`‖Δθ‖`、情報利得、識別の可否、warmup、文章による説明が必須。

## 検証済み表面（`verified()`）

現在 `CoverageStatus::Verified` なのは `special.rs` の特殊関数だけである。scipy 1.18.1 ゴールデン（`golden/special_functions.json`、1,099 ケース）を `cargo test -p kanimiso --lib special::` でリプレイする。

生成型（Cosine/Tsp 冪族、`WindowLag*`, `LogMinkowski*Anomaly` など）は `CoverageStatus::Generated` であり、削除対象である。

## 使う

```rust
use kanimiso::data::{Matrix, Vector};
use kanimiso::linear_model::LinearRegression;
use kanimiso::traits::Fit;
use kanimiso::log::Session;

let x = Matrix::from_fn(20, 1, |i, _| i as f64);
let y = Vector::from_iter((0..20).map(|i| 1.0 + 2.0 * i as f64));
let session = Session::new("ols", "fit");
let fitted = LinearRegression::new().fit(&x, &y, &session)?;
// fitted.value.coef / intercept / r2 / se / p_values
// fitted.report を読まずに係数を論文に書かないこと
// session.ledger() に妥協と警告が残る
```

オンライン:

```rust
use kanimiso::online::LinearRegression as Rls;
use kanimiso::traits::PartialFit;

let mut rls = Rls::new(1.0); // forgetting λ
let q = rls.partial_fit(&x, Some(&y), &session)?;
// q.value.narrative / trust / quality.effective_sample_size
```

実装一覧は `kanimiso::coverage::inventory()`。検証済みだけを見るなら `verified()`。

## 制約

- **a.** 線形代数は `faer`
- **b.** Pure Rust
- **c.** エラーは `signlred`、ログは `ojizou-san`。どちらも計算品質の責任を持つ
- **d.** 追加学習アルゴリズムは説明力を落とさない（`IncrementalExplain` 必須）

## ライセンス

Apache-2.0
