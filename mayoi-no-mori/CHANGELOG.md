# Changelog

## 0.1.0 — unreleased

- Added deterministic random-forest and ExtraTrees classification/regression.
- Added OOB predictions, weighted OOB scores, arbitrary integer class labels,
  sample weights, and normalized feature importances.
- Added squared-error and multiclass first-order gradient boosting.
- Added discrete SAMME, AdaBoost.R2, and an independent random-partition
  Isolation Forest with adjusted path-length scores.
- Added a documented LightGBM-style histogram/Newton subset with aggregate
  L1/L2-regularized gains and leaf values, prefix/suffix bin reductions, and
  explicit missing-value binning.
- Added a documented CatBoost-style ordered categorical-statistic subset for
  regression and binary classification.
- Reused `oldwood` for ordinary CART ensembles and `amatsuki::ChaCha8Rng` for
  all random streams. Newton gain and isolation partitions use separate,
  objective-specific kernels; no duplicate ordinary CART or RNG was introduced.
- Replaced Kanimiso's former tree kernels with compatibility adapters; the
  unsupported legacy SAMME.R selector now fails explicitly instead of claiming
  a probability update the implementation did not satisfy.
