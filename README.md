# kanimiso

[![CI](https://github.com/yaruodemanabu/kanimiso/actions/workflows/ci.yml/badge.svg)](https://github.com/yaruodemanabu/kanimiso/actions/workflows/ci.yml)
[![MSRV](https://img.shields.io/badge/MSRV-1.85-555.svg)](https://github.com/yaruodemanabu/kanimiso/blob/main/Cargo.toml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](https://github.com/yaruodemanabu/kanimiso/blob/main/LICENSE)

**数値品質を結果に同伴させる、Pure Rust の時系列計量・オンライン推定コア。**

`kanimiso` は、OLS、オンライン統計量、線形 Gaussian 状態空間、ARMA、
ボラティリティモデルなどを、一つの品質契約で扱うワークスペースです。各領域には
`fit` / `partial_fit`、`filter` / `smooth`、時系列関数など適した API を残しています。
計算に成功しても、ランク落ち、境界解、数値的な代替処理などがあれば、値だけでなく
機械可読な品質報告を返します。

現在は `0.2.0-alpha.1` に向けた再構成中です。scikit-learn の代替を広く名乗る段階
ではなく、**検証済みの狭い核を必要とする利用者**が対象です。API の破壊的変更もあり得ます。

## どんな用途に向くか

- 線形回帰、時系列フィルタ、状態空間モデル、逐次推定を Pure Rust で組み込みたい
- 推定値だけでなく、警告・数値的妥協・識別不能の理由も保存したい
- native BLAS / LAPACK、C、C++、Fortran に依存しないビルドが必要
- Python 実装や高精度計算との照合根拠を、リポジトリ内で再実行したい

次の用途には、まだ向きません。

- scikit-learn / statsmodels の全面的な drop-in replacement
- API 安定性が必須の本番システム
- GPU 学習、大規模疎行列、深層学習、幅広い分類器を一つの crate に求める用途

## 5分で試す

必要な Rust は 1.85 以上です。まだ crates.io リリース前なので、まずリポジトリから
例を実行してください。

```console
git clone https://github.com/yaruodemanabu/kanimiso.git
cd kanimiso
cargo run -p kanimiso --example ols_quickstart --locked
```

最小の OLS は次の形です。

```rust
use kanimiso::data::{Matrix, Vector};
use kanimiso::linear_model::LinearRegression;
use kanimiso::log::Session;
use kanimiso::traits::Fit;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let noise = [
        0.14, 0.31, -0.07, 0.22, -0.29, -0.11, 0.05, -0.35, 0.28, 0.09,
        -0.18, 0.34, -0.03, -0.24, 0.12, -0.32, 0.26, -0.08, 0.19, -0.15,
    ];
    let x = Matrix::from_fn(noise.len(), 1, |i, _| i as f64);
    let y = Vector::from_iter(
        noise.iter().enumerate().map(|(i, e)| 1.0 + 2.0 * i as f64 + e),
    );

    let session = Session::new("linear_regression", "fit");
    let mut estimator = LinearRegression::new();
    let fitted = estimator.fit(&x, &y, &session)?;

    println!("intercept = {:.6}", fitted.value.intercept);
    println!("slope     = {:.6}", fitted.value.coef[0]);
    println!("R²        = {:.6}", fitted.value.r2);

    // 成功値と切り離さず、必ず検査または保存する。
    for issue in fitted.report.issues() {
        eprintln!("quality: {issue}");
    }

    Ok(())
}
```

Git 依存として組み込む場合は、再現可能性のため実際に採用した commit SHA を
`rev` に固定してください。

```toml
[dependencies]
kanimiso = { git = "https://github.com/yaruodemanabu/kanimiso", rev = "<commit-sha>" }
```

## 現在の機能範囲

検証状態の正本は [`kanimiso::coverage::verified()`](https://github.com/yaruodemanabu/kanimiso/blob/main/kanimiso/src/coverage.rs) です。
公開モジュールが存在しても、この台帳にないものを検証済みとは扱いません。

| 領域 | 主な API | 状態 |
|---|---|---|
| 線形回帰 | `linear_model::LinearRegression` | Verified |
| オンライン推定 | RLS、平均・分散・共分散、指数重み、自己相関、分散選択 | Verified |
| 状態空間 | `state_space::LinearGaussianStateSpace` の filter / RTS smoother | Verified |
| 時系列 | ARMA の ACF・ACOVF・比多項式展開、BK / CF / 線形フィルタ | Verified |
| ボラティリティ | GARCH(1,1)、power-2 FIGARCH(1,d,1)、Eq. 11 の one-AR / news-MA なし FIEGARCH | Verified |
| 確率過程 | median-gap 倍率 5 点を探索する `stats::process_mle` lite | Verified |
| 特殊関数 | beta / gamma、正規・χ² の CDF、t / F の CDF・p-value | Verified |
| HMM | Gaussian / categorical / Poisson emission の汎用 HMM | Experimental |
| その他 | 正規・χ²の上側 p-value、k-NN 異常検知、Nelder–Mead、EWMA / EGARCH、ARMA 生成 | Experimental |

Verified は「正しそう」という意味ではありません。決定的 golden、閉形式または
brute-force、性質テストを CI で再生し、許容差の根拠を記録した項目です。検証方式、
オラクルの独立性と限界、実測誤差は [検証ガイド](https://github.com/yaruodemanabu/kanimiso/blob/main/docs/validation.md) にまとめています。

## 品質契約

通常の estimator と最も違うのは戻り値です。

```text
fit / predict / partial_fit
          │
          ├─ Failure              値を返してはいけない問題
          └─ Qualified<T>
                 ├─ value         計算結果
                 └─ report        警告・妥協・意味のなさ・診断
```

- `Failure` は、非有限入力、識別不能、意味のない fit など、既定ポリシーで中止すべき
  問題を完全な `Report` とともに返します。
- `Qualified<T>` は `#[must_use]` です。値が返っても `report` の検査または永続化が
  契約の一部です。
- 擬似逆、ridge fallback、jitter、確率 clamp など、依頼と異なる計算を採用した場合は
  `NumericalCompromise` に残します。
- `Session` は開始、反復、品質 issue、成功・失敗を同じ ledger に記録します。
- `partial_fit` はパラメータを更新するだけでなく、`IncrementalExplain` として有効標本数、
  パラメータ変化、情報利得、識別状態を返します。

このため「とりあえず一つの数値だけ欲しい」API より少し重くなります。その代わり、
失敗や近似を成功値に見せかけないことを優先しています。

## ワークスペース

| crate | 責務 |
|---|---|
| `kanimiso` | 推定、変換、予測と共有数値核 |
| `signlred` | `Qualified<T>`、`Failure`、`Policy`、数値・統計品質の診断 |
| `ojizou-san` | `Session`、品質 ledger、妥協 journal、オンライン学習の説明 |

直接の科学技術計算依存は、ワークスペース全体で次の二つだけです。

- `ndarray = 0.17.2`: N 次元配列、軸、slice、view
- `faer = 0.24.4`: 行列所有、分解、固有値、線形方程式、最小二乗

両方とも完全固定し、`ndarray-linalg`、`ndarray-stats`、native BLAS / LAPACK は
依存ポリシー lint で拒否します。first-party code は `#![forbid(unsafe_code)]` です。
Python は fixture の生成・監査と CI lint にだけ使い、Rust ライブラリの実行時依存では
ありません。

## 開発と検証

ローカルで主要な CI 相当を実行するには次を使います。Python 3.12、Bash、
`cargo-deny`、`cargo-llvm-cov` が必要です。以下の `RUSTDOCFLAGS` の書式は POSIX shell
向けです。

```console
cargo fmt --all --check
cargo +1.85 test --workspace --all-features --locked
cargo test --workspace --all-features --locked
cargo test --workspace --doc --locked
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --locked
cargo package -p kanimiso --list --locked
python scripts/lint_dependencies.py
python scripts/lint_clippy.py
bash scripts/lint_redundancy.sh
cargo deny check
cargo llvm-cov --workspace --locked --summary-only --fail-under-lines 76.0
```

CI は全ワークスペースを MSRV 1.85 と固定 toolchain 1.98.0 で検査し、stable では
`signlred` / `ojizou-san` の lint・test・docs を先行監視します。オラクル群も領域別
ジョブで再生します。設計判断と冗長性予算は
[AGENTS.md](https://github.com/yaruodemanabu/kanimiso/blob/main/AGENTS.md)、v0.1 の削除 API
と移行先は [移行メモ](https://github.com/yaruodemanabu/kanimiso/blob/main/docs/dropped_v0_1.md)
を参照してください。利用者に影響する変更は
[CHANGELOG](https://github.com/yaruodemanabu/kanimiso/blob/main/CHANGELOG.md) に記録します。

## ライセンス

Apache-2.0
