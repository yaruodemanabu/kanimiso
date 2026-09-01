# POT coverage

この表は Python Optimal Transport (POT) `0.9.7.post1`
（Git tag `0.9.7.post1`, commit `9932112dbb0c985f3e57e38977ecc44c02f5ffc0`）を
基準にする。同版の計算コードは `0.9.7` と同一である。POT の配列バックエンドや描画 API をそのまま模倣するのではなく、
距離・輸送として観測可能な計算結果を Pure Rust と `faer` で提供する。

状態:

- **implemented**: 公開 API と直接テストがある
- **partial**: 中核方式はあるが POT の全方式・オプションを満たしていない
- **planned**: この目標を完了するため未実装
- **other crate**: 責務を別クレートへ移した
- **not applicable**: Python 配列バックエンド、描画、時間計測など計算量でないもの

## 責務境界

| 領域 | 所有クレート | 規則 |
|---|---|---|
| 離散・連続 OT、Wasserstein、輸送 plan、barycenter | `wormhole` | `faer::Mat<f64>` を公開行列境界にする |
| カーネル、Gram、kernel distance/alignment、MMD | `coronel` | `wormhole` にカーネル式を重複実装しない |
| DTW、Soft-DTW、波形 alignment/距離/barycenter | `jelly-wave` | `wormhole` に波形 DP を重複実装しない |
| ML estimator の fit/predict と品質 reporting | `kanimiso` | 数値 primitive を上記 3 クレートから呼ぶ |

## Top-level / unified API

| POT API | Rust API | 状態 |
|---|---|---|
| `ot.dist`, `ot.dist_batch` | `wormhole::metrics::{distance,pairwise,pairwise_self,pairwise_batch}` | partial（主要 metric と batch。callable / SciPy 全 metric は未対応） |
| `ot.unif` | solver の `None` weights / 内部 uniform | implemented |
| `ot.emd`, `ot.emd2` | `wormhole::{emd,emd2}` | partial（plan / value。dual、warm start、multi-target は未対応） |
| `ot.emd_1d`, `ot.emd2_1d`, `ot.wasserstein_1d` | `wormhole::{emd_1d,wasserstein_1d}` | partial（dense scalar core） |
| `ot.sinkhorn`, `ot.sinkhorn2` | `wormhole::{sinkhorn,sinkhorn2}` | partial（scaling / log / Greenkhorn 実装、continuation 等は下記） |
| `ot.sinkhorn_unbalanced`, `ot.sinkhorn_unbalanced2` | `wormhole::unbalanced` | partial（KL Sinkhorn） |
| `ot.barycenter` | `wormhole::barycenter::barycenter` | partial（通常 Sinkhorn） |
| `ot.barycenter_unbalanced` | `wormhole::barycenter_unbalanced` | implemented（generalized Sinkhorn） |
| `ot.sliced_wasserstein_distance` | `wormhole::sliced::sliced_wasserstein` | implemented |
| `ot.max_sliced_wasserstein_distance` | `wormhole::sliced::max_sliced_wasserstein` | implemented |
| `ot.min_sliced_transport_plan` | `wormhole::sliced::min_sliced_transport_plan` | implemented（original-space scoring、fixed/random projections） |
| `ot.expected_sliced_plan` | `wormhole::sliced::expected_sliced_plan_with_projections` | implemented（cost / `beta` weighting） |
| `ot.gromov_wasserstein*`, `ot.fused_gromov_wasserstein*` | `wormhole::gromov` | partial（square loss exact/entropic） |
| `ot.gromov_barycenters`, `ot.fgw_barycenters` | `wormhole::gromov::{gromov_barycenter,fused_gromov_barycenter}` | implemented（square loss、fixed support） |
| `ot.lowrank_sinkhorn`, `ot.lowrank_gromov_wasserstein_samples` | `wormhole::lowrank` | partial（cost factorization / low-rank kernel Sinkhorn） |
| `ot.factored_optimal_transport` | `wormhole::factored::factored_optimal_transport` | implemented（exact / entropic subproblems） |
| `ot.weak_optimal_transport` | `wormhole::weak::weak_optimal_transport` | implemented |
| `ot.solve`, `ot.solve_sample` unified dispatch | `wormhole::solvers::{solve,solve_samples}` | partial |
| `ot.solve_gromov`, `ot.solve_bary_sample` | — | planned |
| `ot.compute_bspot_bijection`, `ot.merge_bijections` | — | planned |

## `ot.lp`

| 機能群 | 状態 |
|---|---|
| exact EMD / EMD² quantity | partial（Pure Rust successive-shortest-path residual network。POT の network simplex 自体ではない） |
| lazy EMD² | planned |
| 1-D plan/value/quantile | implemented |
| 1-D dual backprop quantities | planned |
| circle Wasserstein / circular embedding | implemented（`p >= 1`、semidiscrete uniform、linear circular OT） |
| LP histogram barycenter | planned |
| free-support barycenter | implemented |
| generalized free-support barycenter | planned |
| discrete multi-marginal OT | planned |

## `ot.bregman`

| 機能群 | 状態 |
|---|---|
| Sinkhorn-Knopp | implemented |
| log-domain Sinkhorn | implemented |
| stabilized / epsilon-scaling Sinkhorn | implemented（log-dual stabilization / geometric continuation） |
| Greenkhorn | implemented |
| Screenkhorn | planned |
| empirical Sinkhorn / Sinkhorn divergence | implemented |
| lazy / geomloss-compatible empirical solver | planned（外部 native backend は使用しない） |
| Nyström empirical Sinkhorn | planned |
| fixed-support barycenter | implemented |
| stabilized/debiased/free-support entropic barycenter | planned |
| convolutional 2-D barycenter | planned |
| Wasserstein dictionary unmixing | planned |
| JCPOT barycenter | planned |

## `ot.optim` and `ot.smooth`

| 機能群 | 状態 |
|---|---|
| conditional gradient / generalized CG | partial（generic CG + Armijo。generalized CG は planned） |
| semi-relaxed / partial CG | planned |
| Armijo and quadratic line search | planned |
| negative entropy regularizer | partial（Sinkhorn objective） |
| squared-L2 / sparsity constrained smooth OT | partial（squared-L2 CG。sparsity constraint は planned） |
| dual and semi-dual smooth OT | planned |

## `ot.unbalanced`, `ot.partial`, `ot.regpath`

| 機能群 | 状態 |
|---|---|
| KL unbalanced Sinkhorn | implemented |
| stabilized / translation-invariant unbalanced Sinkhorn | planned |
| MM unbalanced (`KL`, `L2`) | planned |
| generic divergence L-BFGS unbalanced | planned |
| unbalanced barycenter | implemented |
| exact 1-D UOT | planned |
| sliced unbalanced OT / unbalanced sliced OT | planned |
| partial Wasserstein / partial Wasserstein² | implemented |
| Lagrangian partial Wasserstein | implemented（dummy-reservoir min-cost flow） |
| entropic partial Wasserstein | implemented（multiplicative Dykstra） |
| partial 1-D Wasserstein | planned |
| partial / entropic partial GW and FGW | implemented（square loss、fixed transported mass） |
| UOT regularization path | planned |

## `ot.sliced`, spherical, semidiscrete

| 機能群 | 状態 |
|---|---|
| random Euclidean projections | implemented（決定的 seed） |
| sliced / max-sliced Wasserstein | implemented |
| all sliced plans | implemented（dense plan collection） |
| min/expected sliced plans | implemented |
| sphere projections / spherical sliced Wasserstein | implemented |
| linear spherical variant | implemented（stereographic variant は stable 0.9.7.post1 の対象外） |
| circle semidiscrete Wasserstein | implemented（uniform target） |
| general semidiscrete map / c-transform / solve | planned |

## `ot.gromov`

| 機能群 | 状態 |
|---|---|
| exact GW / GW² | partial（squared loss conditional gradient） |
| entropic GW | partial（squared loss projected fixed-point） |
| fused GW / FGW² | partial（squared loss） |
| BAPG GW / FGW | implemented（symmetric square loss、marginal-loss relaxation） |
| GW / FGW barycenters | implemented（square loss、fixed support、FGW fixed feature/structure options） |
| partial and entropic partial GW/FGW | implemented（square loss、fixed transported mass） |
| semi-relaxed GW/FGW and barycenters | planned |
| unbalanced and fused-unbalanced GW | planned |
| sampled GW estimators | planned |
| quantized GW/FGW | planned |
| low-rank GW | planned |
| graph dictionary learning | planned |

## Gaussian and GMM transport

| POT module / feature | 状態 |
|---|---|
| Gaussian Bures distance / mapping | partial（dense single-problem core） |
| empirical and high-dimensional Bures variants | partial（empirical dense 実装、low-rank HD は planned） |
| Bures-Wasserstein barycenter | implemented |
| Gaussian GW distance / mapping | implemented（closed form + empirical adapters） |
| Gaussian density / GMM density | implemented |
| GMM component cost, OT plan/loss/map | implemented（deterministic barycentric map） |
| GMM plan density | implemented（finite-tolerance map support） |
| GMM barycenter | implemented（fixed component weights、Euclidean / Bures projection） |

## Large scale, structured, and ML-related solvers

| POT module / feature | 状態 |
|---|---|
| `ot.stochastic` SAG/ASGD/dual/semi-dual | planned |
| `ot.lowrank` low-rank Sinkhorn | partial（squared-cost factors / kernel scaling。non-negative rank solver は planned） |
| `ot.factored` factored OT | implemented |
| `ot.weak` weak OT | implemented（exact-line-search Frank-Wolfe） |
| `ot.coot` balanced/unbalanced COOT | partial（balanced exact / entropic BCD。unbalanced は planned） |
| `ot.sgot` spectral graph OT metric | planned |
| `ot.bsp` BSP-OT bijections | planned |
| `ot.mapping` linear joint mapping | planned |
| kernel joint mapping | planned（kernel primitive は `coronel`、OT 最適化は `wormhole`） |
| nearest Brenier potential and bounds | planned |
| domain-adaptation transport plans/transforms | planned（estimator shell は `kanimiso`） |
| WDA / projection-robust Wasserstein / EWCA | planned（estimator shell は `kanimiso`） |
| GNN transport pooling layers | planned（距離 primitive のみ `wormhole`、model layer は `kanimiso`） |

## Batch and support utilities

| POT module / feature | 状態 |
|---|---|
| batched linear Sinkhorn / proximal Bregman | planned |
| batched sample and GW solve | planned |
| simplex / sparse-simplex projection | implemented |
| cost normalization | implemented |
| SDP projection / Bures exponential | implemented |
| lazy tensors | planned |
| dataset generators | not applicable（例用 fixture は Rust 側に置く） |
| NumPy/PyTorch/JAX/CuPy/TensorFlow backends | not applicable（`faer` が唯一の dense backend） |
| plotting and timers | not applicable |

## 別クレートの対応

### `coronel`

| 量 | 状態 |
|---|---|
| linear/polynomial/RBF/Laplacian/sigmoid/cosine/chi-square kernels | implemented |
| pairwise/Gram/centered Gram | implemented |
| kernel-induced distance | implemented |
| biased/unbiased MMD² | implemented |
| centered kernel alignment | implemented |
| POT `utils.kernel` / kernel mapping の primitive | implemented |

### `jelly-wave`

| 量 | 状態 |
|---|---|
| DTW + Sakoe-Chiba window + alignment path | implemented |
| pairwise DTW | implemented |
| Soft-DTW and debiased divergence | implemented |
| DTW barycenter averaging | implemented |
| ERP / discrete Fréchet | implemented |
| multivariate/variable-length series adapters | implemented（hard / Soft-DTW、pairwise collections） |

## 完了ゲート

目標完了には、すべての **planned** / **partial** 項目を実装済みにするか、
「距離・輸送に関する計算量ではない」ことをコードと POT 0.9.7 API から
根拠付きで分類し直す必要がある。さらに以下をすべて満たす。

1. `cargo test --workspace --all-targets`
2. `cargo clippy --workspace --all-targets -- -D warnings`
3. `cargo fmt --all -- --check`
4. POT で生成して固定した oracle fixture に対する数値回帰
5. first-party crate に `unsafe` がなく、FFI / native BLAS・LAPACK dependency がないことの監査
6. `kanimiso` の `faer::Mat<f64>` を直接渡す統合テスト
7. 3 クレートの責務境界を破る重複実装がないことの監査
