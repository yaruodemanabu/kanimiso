# Dropped v0.1 surface

These names are intentionally absent from the verification-first v0.2 surface.
They remain listed here for migration searches; retaining a name without a
correct, independently verified implementation would overstate the API.

## Generated HMM archive

The generated `kanimiso/src/hmm.rs` monolith is no longer part of the Rust
module tree. Its byte-identical archive is
`generated-v0.1-archive/hmm.rs.txt` (18,790,144 bytes, 569,177 lines,
SHA-256 `169CEF54DD9EC0EDEC4C40B1FFCED69D67E602A51ECF536D7002D1B58F68F102`).
The 4,953 legacy public `*Hmm*` identifiers had no Rust consumers elsewhere
in this workspace. Supported migrations use `hmm::HiddenMarkovModel<E>` and
runtime parameters; the archive is retained only for source-level migration
searches.

The unverified `kanimiso/src/tslearn.rs` surface is likewise outside the
module tree at `generated-v0.1-archive/tslearn.rs.txt` (1,887,283 bytes,
60,208 lines, SHA-256
`1AD48FD80036752EC197BEF04C30C94A71C3BA74553B56908AA70411E973B25C`).
It contained 1,581 direct public declarations but only four tests, no Verified
inventory item, and no Rust consumer elsewhere in this workspace. Individual
algorithms may return only after they have an independent oracle and a compact,
runtime-parameterized API.

## Generated online archive

The `kanimiso/src/online.rs` monolith is no longer part of the Rust module tree.
Its byte-identical archive is `generated-v0.1-archive/online.rs.txt` (2,472,160
bytes, 78,273 lines, SHA-256
`211C726A441D70A9B391F56EDDE3951BE5B5E10D8611A8B5A2CFC43826533EA2`).
The compact replacement retains only the independently verified public surface:
`online::LinearRegression`, `OnlineWeightedMean`, `OnlineEwMean`, `OnlineEwVar`,
`OnlineMean`, `OnlineVar`, `OnlineCovariance`, `OnlineSum`, `OnlineCount`,
`OnlineAutoCorr`, and `OnlineVarianceThreshold`. The old experimental and
generated estimators, including the numbered `WindowLag*` families and
`SgdRegressor` re-export, have no v0.2 replacement. Their names remain searchable
in the archive and may return only with an independent oracle and a compact,
runtime-parameterized design.

## Generated coverage archive

The generated `kanimiso/src/coverage.rs` inventory is no longer compiled.
Its byte-identical archive is
`generated-v0.1-archive/coverage.rs.txt` (681,720 bytes, 23,295 lines,
SHA-256 `9F2E4DDE55EAA12F617B3AA6DBC2D1B7CB67FBC109C1A55AC2CFCB3CF06FFD29`).
The active ledger records only the small Verified/Experimental v0.2 surface;
archived names are not coverage claims.

## TSA monolith archive

The `kanimiso/src/tsa.rs` monolith is no longer part of the Rust module tree.
Its byte-identical archive is `generated-v0.1-archive/tsa.rs.txt` (789,597
bytes, 23,586 lines, SHA-256
`8A49C809C024B9DB41108EFDFA6AAD49CEF434B896CFF224798B1710E68786D7`).
The active `tsa` module retains exact ARMA recurrences, the independently
checked GARCH-family models, EWMA, and the verified filter re-exports.
`tsa::ArmaProcess` and the other 311 dropped direct declarations have no
active replacement; their names remain searchable in the archive.

| v0.1 name | Reason | Replacement / status |
|---|---|---|
| `tsa::ArimaKalman` | The companion recursion skipped transition/process noise on missing rows, omitted MA terms for common orders, and used a non-Gaussian concentrated score. | Reintroduce only through `state_space::LinearGaussianStateSpace` after an independent exact-ARIMA oracle is committed. |
| `tsa::simulation_smoother` | Added independent noise to a filtered level; it was not a disturbance simulation smoother. | Pending a Durbin--Koopman simulation-smoothing implementation and oracle. |
| `tsa::statespace_news` | Returned raw one-step residuals, not a news decomposition. | Pending a revision-aware news decomposition with contribution identities. |

## Stats, IV, and panel archives

The `kanimiso/src/stats.rs` monolith is no longer part of the Rust module tree.
Its byte-identical archive is `generated-v0.1-archive/stats.rs.txt` (1,089,036
bytes, 33,127 lines, SHA-256
`E1F72E0101D003E31117D008EC6CD723F2A1067AF0D45B06B042E191E1985ED8`).
It exposed 509 top-level public declarations but had only eight tests. The
compact replacement retains only `stats::ProcessMleFit` and
`stats::process_mle`, backed by the Decimal dense-GLS oracle, invariance
properties, and explicit invalid-input tests.

The former `iv` and `panel` modules depended on unverified stats routines and
had no other Rust consumers in this workspace. They are absent from the v0.2
module surface and preserved byte-identically as
`generated-v0.1-archive/iv.rs.txt` (99,732 bytes, 2,857 lines, SHA-256
`3238B87C68EED11FF199B7D4A7BDB4A79F140123252BEE87A40380A2A66E22DF`)
and `generated-v0.1-archive/panel.rs.txt` (150,783 bytes, 4,205 lines, SHA-256
`CA2DB281CB4835CA59B3EE00352934620363FCE68A0B99FA935E96EB6AF5C74D`).
Their names may return only with an independent oracle and a compact shared
numerical core.

## Tree adapter migration

`oldwood` now owns the single verified CART split, arena, and traversal kernel;
`mayoi-no-mori` owns random forest, ExtraTrees, boosting, and isolation-tree
logic. The similarly named types under `kanimiso::tree` and
`kanimiso::histgb` are quality-contract adapters, not source-compatible copies
of the v0.1 implementations.

The retained adapters still return `Qualified<T>` through `Fit` / `Predict`,
but these direct helper calls are intentionally breaking:

| v0.1 call | v0.2 migration |
|---|---|
| `FittedTreeClassifier::predict_proba_row(x, row) -> Vec<f64>` | Pass a `Session`; inspect `Result<Qualified<Vec<f64>>>` before using the value. |
| `FittedIsolationForest::{average_path_length, score_samples}(x, ...)` returning bare values | Pass a `Session`; both methods now return `Result<Qualified<_>>` and reject invalid shape/non-finite input. |
| `IsolationForest::new(n_trees)` | Use `IsolationForest::new()` and set the public `n_trees` field, or construct the standalone `mayoi_no_mori::IsolationForest` with `IsolationForestOptions`. |
| `RandomTreesEmbedding` | Removed; there is no active v0.2 replacement. `IsolationForest` remains an anomaly scorer, not an embedding substitute. |
| `AdaBoostAlgorithm::SammeR` | The selector remains only to produce an explicit `InvalidParameter` failure. Use discrete `Samme`; the default changed from SAMME.R/depth 2 to SAMME/depth 1. |

Histogram adapter types keep their legacy names, but their tree construction
now delegates to the documented `mayoi-no-mori::LightGbm*` subset. They accept
NaN as a dedicated high bin, reject infinity, and do not claim parity with
scikit-learn HistGradientBoosting or upstream LightGBM.
