# Validation scope

## Status

The deterministic binary CART core is verified against scikit-learn 1.7.2,
analytical results, an independent exhaustive split evaluator, and
deterministic invariants. The crate does not claim scikit-learn tree-layout
parity beyond the observable outputs covered here.

The original Python archive is not a runtime or test dependency. Its SHA-256
was `9B8FE9329F496DFF135CC4F0C499C6EFE50322C832097A157AD166C143914B3A`;
it contained criteria and Welford primitives but no tree builder.

## Evidence layers

| Layer | Evidence |
|---|---|
| External oracle | scikit-learn 1.7.2 weighted `DecisionTreeClassifier` and `DecisionTreeRegressor` probe predictions and probabilities |
| Analytical | Balanced Gini = 1/2, balanced binary entropy = 1 bit, four-class entropy = 2 bits, weighted leaf probabilities and means |
| Brute force | Independent test code enumerates every feature and midpoint and recomputes classification and regression gain directly |
| Properties | Row permutation, positive weight scaling, and integer-weight/row-duplication equivalence |
| Structural | Arena children, leaf application, class probability sums, normalized feature importance, deterministic tie-breaking |
| Stress/failure | Opposite-sign `f64::MAX` features, adjacent floats, large-offset targets, non-finite data, invalid weights and strategy output, unrepresentable variance |

The initial 2026-09-04 run measured zero ULP difference in every comparison
between the implementation and the included brute-force or invariance paths.
Those assertions therefore use exact bit equality rather than an unmeasured
tolerance. Irrational entropy values are not covered by a loose approximate
oracle; exact power-of-two cases are used instead.

## scikit-learn Tier 0

`golden/sklearn_cart.csv` is generated from scikit-learn 1.7.2 with exact PEP
723 dependency pins in `scripts/sklearn_cart_oracle.py`. The three cases use
`sample_weight`, `max_depth = 2`, and `random_state = 1729`:

| Case | Reference criterion | Compared output |
|---|---|---|
| `weighted_gini_depth2` | classifier `gini` | labels, sorted classes, probabilities |
| `weighted_entropy_depth2` | classifier `entropy` | labels, sorted classes, probabilities |
| `weighted_squared_error_depth2` | regressor `squared_error` | predictions |

The fixture contains training data plus separate probe rows. It excludes
thresholds, child indices, feature traversal order, and `apply` leaf IDs,
which can differ between equivalent layouts. Cases were selected to avoid
equal-gain feature choices. The committed fixture SHA-256 is
`bbb5a48390e7c44db53280d7bdcc6f860098856b09fcdd8e311d8dd4d165b890`.

The initial CPython/scikit-learn replay on 2026-09-04 measured maximum
absolute error `0.0` (zero ULP) for both probabilities and regression
predictions. Rust therefore uses an explicit absolute tolerance of `0.0`.
This is stricter than applying an arbitrary epsilon; a later nonzero tolerance
must record its measured maximum and retain only a 3–4x margin.

Regenerate and exact-check the canonical LF-normalized CSV with:

```text
uv run oldwood/scripts/sklearn_cart_oracle.py emit
uv run oldwood/scripts/sklearn_cart_oracle.py check
```

The ordinary Rust replay uses `include_str!` and the standard library. Python,
NumPy, SciPy, and scikit-learn are not Cargo, test-runtime, or CI dependencies.

## Branch contract

Training checks these conditions before creating a fitted value:

| Condition | Result |
|---|---|
| zero rows or zero columns | `EmptyTrainingData` / `EmptyFeatures` |
| target or weight length mismatch | typed length error |
| NaN/infinite feature or regression target | location-bearing error |
| negative, NaN, or infinite weight | `InvalidWeight` |
| no positive weight | `NoPositiveWeight` |
| invalid option | `InvalidOption` |
| out-of-range strategy feature | `InvalidStrategyFeature` |
| threshold not finite or outside `[minimum, maximum)` | `InvalidStrategyThreshold` |
| non-representable moment, gain, or importance | `NumericalOverflow` |

For each non-terminal node the engine:

1. obtains feature candidates from `SplitStrategy`;
2. validates every feature before stable sorting and deduplication;
3. creates a canonical stable ordering of positive-weight rows;
4. passes sorted unique feature values to the strategy;
5. validates every threshold before stable sorting and deduplication;
6. scores every valid candidate with the configured runtime criterion;
7. selects greater gain, then lower feature, then lower threshold;
8. appends children to one immutable-by-API arena.

`TreeOptions::max_features` limits the sorted strategy output. With the default
`Exhaustive` strategy, `Some(k)` therefore means the leading `k` feature
columns. A randomized caller should return exactly its node-specific subset.

## Numerical choices

Classification Gini is evaluated as `sum(p * (1 - p))`; this avoids the
near-pure subtraction in `1 - sum(p^2)`. Entropy skips zero-weight classes and
uses base-two logarithms. Weight and impurity sums use compensated summation.

Regression uses a weighted central-moment update. Sorted feature groups are
summarized once, then merged into forward prefixes and backward suffixes.
Candidate evaluation does not repeatedly rescan every row and does not derive
variance from subtracting two large raw moments.

A threshold must satisfy `minimum <= threshold < maximum`. Exhaustive midpoint
construction uses same-sign `lower + (upper-lower)/2` and opposite-sign
`lower/2 + upper/2`. When adjacent floats leave no interior representable
value, the lower value is a valid `<=` threshold and separates the pair. Both
the adjacent-value and opposite-extreme branches have classification and
regression coverage.

## Reproduction

```text
cargo +1.85 test -p oldwood --all-targets
cargo +1.85 clippy -p oldwood --all-targets -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo +1.85 doc -p oldwood --no-deps
```

No environment variable, Python installation, native library, network access,
or enlarged thread stack is required.
