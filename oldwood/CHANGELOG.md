# Changelog

All notable changes to `oldwood` are documented here.

## 0.1.0

- Replaced the incomplete Python criteria archive with a standalone Pure Rust
  crate and no external dependencies.
- Added deterministic weighted CART classification and regression.
- Added runtime Gini, entropy, and squared-error criteria.
- Added `MatrixView`, `DenseMatrix`, exhaustive fitting, probability
  prediction, leaf application, feature importance, and arena inspection.
- Added the validated `SplitStrategy` extension boundary for reusable forest
  and random-threshold policies without duplicating the CART evaluator.
- Added analytical, brute-force, invariance, weight, strategy, and numerical
  stress tests.
- Added a pinned scikit-learn 1.7.2 Tier-0 fixture, PEP 723 generator, and
  dependency-free Rust replay for weighted classification and regression.
