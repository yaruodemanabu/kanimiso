# Changelog

## 0.1.0 — unreleased

- Added deterministic random-forest and ExtraTrees classification/regression.
- Added OOB predictions, weighted OOB scores, arbitrary integer class labels,
  sample weights, and normalized feature importances.
- Added squared-error and multiclass first-order gradient boosting.
- Added discrete SAMME, AdaBoost.R2, and an independent random-partition
  Isolation Forest with adjusted path-length scores.
- Added a documented LightGBM-style histogram/Newton subset with L1/L2 leaf
  regularization and explicit missing-value binning.
- Added a documented CatBoost-style ordered categorical-statistic subset for
  regression and binary classification.
- Reused `oldwood` for every supervised tree and `amatsuki::ChaCha8Rng` for all
  random streams; no duplicate CART or RNG implementation was introduced.
- Replaced Kanimiso's former tree kernels with compatibility adapters; the
  unsupported legacy SAMME.R selector now fails explicitly instead of claiming
  a probability update the implementation did not satisfy.
