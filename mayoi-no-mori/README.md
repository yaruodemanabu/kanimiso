# mayoi-no-mori

`mayoi-no-mori` is a small, deterministic Pure Rust crate for tree ensembles.
Ordinary CART ensembles share [`oldwood`](https://docs.rs/oldwood)'s split and
traversal kernel. This crate owns resampling, randomized proposals, additive
objectives, histogram binning, categorical encodings, and the distinct Newton
gain tree needed for regularized histogram objectives.

The crate is alpha software, and every estimator below is currently
**Experimental** in the workspace coverage ledger. Its supported surface is
explicit so that names such as “LightGBM” and “CatBoost” do not imply drop-in
compatibility.

## Implemented estimators

| Estimator | Implemented here | Deliberate limits |
|---|---|---|
| `RandomForestClassifier` / `RandomForestRegressor` | bootstrap or subsampling, per-node random feature subsets, sample weights, deterministic seeds, OOB predictions and score, mean impurity-decrease importances | single-process CPU; no warm start or quantile forest |
| `ExtraTreesClassifier` / `ExtraTreesRegressor` | per-node random features and one uniformly random valid threshold per candidate feature | no multi-output target |
| `GradientBoostingClassifier` / `GradientBoostingRegressor` | stochastic first-order softmax or squared-error boosting | no custom objective or early stopping yet |
| `AdaBoostClassifier` / `AdaBoostRegressor` | discrete multiclass SAMME and weighted-bootstrap AdaBoost.R2, including stage diagnostics | SAMME.R is intentionally absent; no alternative loss selector |
| `LightGbmClassifier` / `LightGbmRegressor` | global quantile bins, aggregate regularized Newton split gain and leaf values, prefix/suffix bin sums, L1/L2, minimum leaf Hessian, row/feature sampling, binary and corrected multiclass Hessians, NaN as a separate high bin | depth-wise; no leaf-wise growth, GOSS, EFB, native categorical split, distributed/GPU training, or model format |
| `CatBoostClassifier` / `CatBoostRegressor` | target-free-at-the-current-row ordered category statistics, deterministic permutation averaging, unknown/missing categories, subsequent tree boosting | classifier is binary; no ordered-gradient boosting, category combinations, symmetric trees, text/embedding features, GPU training, or CatBoost model format |
| `IsolationForest` | subsampling, uniformly random partitions, adjusted path lengths, and Liu–Ting–Zhou anomaly scores | no contamination threshold or label conversion; isolation trees correctly do not use CART impurity |

The `LightGbm*` and `CatBoost*` rows are documented subsets inspired by the
respective algorithms. The upstream projects use broader designs: LightGBM normally grows
trees leaf-wise and includes specialized sampling/bundling, while CatBoost's
default symmetric trees and ordered boosting go beyond ordered categorical
statistics alone. The API therefore exposes concrete Rust model types and does
not import or export either upstream model format.

For a general bagged baseline, start with `RandomForest*`; choose `ExtraTrees*`
when randomized thresholds are intentional. Use `GradientBoosting*` for the
ordinary first-order additive model or `AdaBoost*` for discrete SAMME /
AdaBoost.R2. Choose `LightGbm*` only when the documented binned Newton subset is
the desired algorithm, `CatBoost*` only for the documented ordered-category
subset, and `IsolationForest` for unsupervised anomaly scores. Use `oldwood`
directly when one interpretable CART is sufficient.

## Quick start

```rust
use mayoi_no_mori::{
    ClassificationCriterion, DenseMatrix, ForestOptions, RandomForestClassifier,
};

let x = DenseMatrix::from_row_major(
    6,
    2,
    vec![
        0.0, 0.1,
        0.1, 0.0,
        0.2, 0.2,
        0.8, 0.9,
        0.9, 0.8,
        1.0, 1.0,
    ],
)?;
let y = [10, 10, 10, 42, 42, 42];

let estimator = RandomForestClassifier::new(
    ForestOptions {
        trees: 64,
        seed: 7,
        out_of_bag: true,
        ..ForestOptions::default()
    },
    ClassificationCriterion::Gini,
);
let fitted = estimator.fit(&x, &y, None)?;

assert_eq!(fitted.predict(&x)?, y);
assert!(fitted.oob_score().is_some());
# Ok::<(), mayoi_no_mori::Error>(())
```

All supervised fit methods accept `Option<&[f64]>` sample weights. Class labels
are arbitrary `usize` values and are returned unchanged. Probability columns
follow `fitted.classes()`. Weights must be finite and non-negative, and the
aggregates required by `oldwood` and the ensemble objective must remain
representable as `f64`; overflow is a typed error rather than an implicit
rescaling.

Any matrix type can participate without copying by implementing the three
read-only methods in `oldwood::MatrixView`.

## Reproducibility and failure behavior

- Randomness comes from Isuzu's `amatsuki::ChaCha8Rng`, whose bit stream has a
  known-answer test. A seed determines bootstrap rows, node feature subsets,
  random thresholds, stage subsamples, and ordered-statistic permutations.
- Ordinary CART candidate evaluation and traversal live in `oldwood`. Histogram
  Newton trees use `soft_threshold(G,L1)^2/(H+L2)` scores and deterministic ties;
  their `tree.min_impurity_decrease` is a minimum **raw regularized score gain**
  (without a factor of one half), not root-normalized CART impurity.
- AdaBoost exposes discrete SAMME rather than retaining the old Kanimiso
  SAMME.R-shaped update that could not be justified against the specification.
- Invalid dimensions, non-finite numeric features, invalid weights, and invalid
  options return typed errors. Histogram models accept NaN intentionally and
  encode it as a distinct high bin. CatBoost-style models accept NaN only in
  configured categorical columns.
- A requested iteration count is an upper bound for Newton boosting: fitting
  stops when every usable Hessian is zero rather than inventing a probability
  floor.
- Seed replay, closed-form checks, OOB/probability invariants, and CART's
  scikit-learn oracle are covered. Independent external ensemble fixtures are
  not yet present, which is why these estimators remain Experimental.

See [`docs/validation.md`](docs/validation.md) for the current oracle/property
matrix and [`docs/architecture.md`](docs/architecture.md) for ownership rules.

## MSRV, dependencies, and safety

- Rust 1.85 or newer
- `#![forbid(unsafe_code)]`
- runtime dependencies: `oldwood` and the first-party `amatsuki` RNG crate
- no BLAS/LAPACK/native library and no C/C++ binding

For crates.io publication, publish the first-party dependency chain in the
order `signlred`, `ojizou-san`, `tsutsumi`, then `oldwood` and `amatsuki`,
then `mayoi-no-mori`; workspace path dependencies
carry matching version requirements for that release flow.

Licensed under Apache-2.0.
