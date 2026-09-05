# Validation matrix

The test suite separates exact identities, independent small oracles,
properties, and stress/failure behavior.

| Layer | Current checks |
|---|---|
| Exact/closed form | a constant squared-error target remains exactly its weighted mean; one-stage squared GBDT and Newton updates match hand calculation; a SAMME error of 1/4 gives weight `ln(3)`; two-point Isolation Forest has path length 1 and score 1/2; unique categorical codes encode to the prior because a row cannot see its own target; OOB is rejected without bootstrap |
| Independent kernel oracle | `oldwood` compares root splits with separately written brute-force classification and regression enumerators; ordinary CART ensembles use that exact kernel. Histogram Newton gain has independent aggregate-score and regularized-leaf checks. |
| Properties | probability rows sum to one; original non-consecutive labels survive classification; feature proposals are deterministic per node; bootstrap, AdaBoost.R2, Isolation Forest, and ordered-statistic runs replay exactly for one seed; histogram bins are monotone and reserve a missing bin |
| Stress/failure | zero/negative/non-finite weights, shape mismatches, non-finite targets, invalid sampling fractions, saturated zero-Hessian stages, unseen categories, and numeric NaN outside documented paths return errors or defined behavior |

Tolerance comments record measured error and a fourfold margin. Exact
power-of-two or constant fixtures use equality rather than a broad epsilon.

## Experimental status

Every ensemble estimator remains experimental until its ensemble-level output
is replayed against an independently generated external fixture. The two named
upstream-inspired families additionally need independent LightGBM/CatBoost
coverage for every supported objective and missing/category path. They are
usable as documented algorithms today, but this crate does not claim numerical
parity with upstream training.
