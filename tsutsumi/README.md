# tsutsumi

Pure Rust の共通計算核です。モデル固有の推定処理を置かず、分解・最小二乗・
特殊関数・最適化・積分と、その数値品質の検証を担当します。

`kanimiso`、`number-ruler` はこの実装を共有します。`oldwood` は既定 feature を
無効にし、依存のない `MatrixView` 契約だけを利用します。逆向きのモデル依存はありません。

## できること

| 領域 | API / 対応範囲 |
|---|---|
| 配列 | faer-backed `Matrix` / `Vector`、所有権を持たない `MatrixView` 契約 |
| 線形代数 | SVD 最小二乗、rank / condition / stationarity 診断、ridge、SPD solve、対称固有値 |
| 最適化 | 実行時係数を持つ一つの `NelderMead`、単体の trace、停止理由、非有限目的値の扱い |
| 特殊関数 | beta / gamma、t / F の直接上側確率、正規分布 p 値の直接裾計算、`logsumexp` |
| 積分 | 標準正規期待値用 Gauss–Hermite / Golub–Welsch 求積、2〜128 点 |
| 品質 | `signlred::Policy` と `ojizou_san::Session` を結ぶ `FitCtx` |

## 試す

未公開のため、このリポジトリで `cargo test -p tsutsumi` を実行するか、
採用した Git commit を固定して依存してください。

```rust
use tsutsumi::{Matrix, Vector};
use tsutsumi::linalg::least_squares_with_diagnostics;
use signlred::{Policy, Report};

let x = Matrix::from_row_major(3, 2, &[1.0, 0.0, 1.0, 1.0, 1.0, 2.0]);
let y = Vector::from_slice(&[1.0, 3.0, 5.0]);
let mut report = Report::new("baseline", "solve");
let result = least_squares_with_diagnostics(&mut report, &x, &y, &Policy::default())
    .expect("identified design");
assert_eq!(result.rank, 2);
// Option の有無だけでなく report を検査・保存する。
for issue in report.issues() { eprintln!("{issue}"); }
```

低水準の `linalg` は `Report` に診断を追記し `Option` を返します。
`Some` だけで品質判定を省略しないでください。上位 API は `FitCtx::finish` または
`Report::finish_with_policy` で `Qualified<T>` / `Failure` にまとめます。
行列コンストラクタの長さ・添字には Rust の通常の範囲契約があります。

## 単独検証

- `special_functions.json`: scipy 由来 1,099 ケースを crate 内に同梱。
- `optimize` のテスト: argmin の分岐規則を独立に固定した trace、反射・拡大・
  内外収縮・shrink、既知の二次関数/Rosenbrock 解、置換・平行移動、停止と失敗。
  argmin 自体への依存や、全 solver の同等性の主張はありません。
- 求積: 正規分布の解析的モーメント、反射対称性、次数増加と外部積分の照合。
- 線形代数: 特異性、極端値、PSD/SPD の境界、非有限入力、再構成と残差。

他クレートの計算ロジックを照合するときは、分岐 trace・外部 fixture・閉形式を
先に固定し、値と停止理由を別々に比較します。ここも参照側も同じ式の実装である
だけでは、仕様解釈の独立性の証明にはなりません。

Rust 1.85、Apache-2.0、`#![forbid(unsafe_code)]`。既定 `linalg` feature の科学計算依存は
`faer = 0.24.4` のみ。`default-features = false` は外部実行時依存ゼロです。

参考: [argmin Nelder–Mead](https://argmin-rs.github.io/argmin/argmin/solver/neldermead/struct.NelderMead.html)。
