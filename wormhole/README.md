# wormhole

`wormhole` は機械学習における距離・最適輸送を扱う Pure Rust
ワークスペースです。線形代数と公開行列型には
[`faer`](https://crates.io/crates/faer) を使います。

このリポジトリは責務ごとに 3 クレートへ分かれます。

- `wormhole`: 距離行列、EMD/Wasserstein、Sinkhorn、unbalanced OT、
  sliced Wasserstein、barycenter などの輸送計算
- `coronel`: Gram 行列、kernel distance、MMD などのカーネル量
- `jelly-wave`: DTW、Soft-DTW、ERP、波形 barycenter などの波形量

公開 API は `faer::Mat<f64>` と `Vec<f64>` / `&[f64]` を境界に使い、
同じ `faer` バージョンを使う
[`kanimiso`](https://github.com/yaruodemanabu/kanimiso) から余分な行列変換なしで
利用できる構成です。全クレートで `unsafe` と非 Rust ネイティブライブラリを
禁止しています。

```rust
use faer::Mat;
use wormhole::{emd, sinkhorn};

let a = [0.5, 0.5];
let b = [0.25, 0.75];
let cost = Mat::<f64>::from_fn(2, 2, |i, j| i.abs_diff(j) as f64);

let exact = emd(&a, &b, &cost)?;
let regularized = sinkhorn(&a, &b, &cost, 0.1)?;

assert!(exact.residual < 1e-9);
assert!(regularized.residual < 1e-8);
# Ok::<(), wormhole::Error>(())
```

Python Optimal Transport (POT) 0.9.7.post1 の機能面との対応状況は
`docs/pot-coverage.md` で追跡します。

## kanimiso integration

`tests/kanimiso_integration.rs` は `kanimiso` の固定リビジョンを使い、
`Matrix::inner()` を `wormhole::solve_samples` へ直接渡し、返された
`faer::Mat<f64>` を `Matrix::from_faer` へコピーなしで移せることを検証します。
Rust 1.84 で統合する下流 workspace も、互換性のない新しい transitive dependency
を避けるため `.cargo/config.toml` の MSRV fallback と 1.84 で生成した lockfile を
使用する必要があります。
