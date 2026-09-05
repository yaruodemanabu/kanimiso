# Regression validation contract

The tested response families are Gaussian/identity, Bernoulli/logit, and
Poisson/log with unit exposure. Neither the fixture nor a `Verified` entry
asserts general statsmodels equivalence.

| Kernel / API | Independent evidence | Additional checks |
|---|---|---|
| LM | statsmodels 0.14.5 OLS coefficients, covariance, SE, p, fitted means, SSE; retained Decimal OLS replay through kanimiso | row permutation, hat trace, input failures, extrapolation notes |
| GLM | statsmodels canonical GLM: the same outputs and deviance for Bernoulli and Poisson | row permutation, complete separation failure, response domains, finite inference |
| Gaussian random intercept | statsmodels MixedLM ML and REML: fixed effects, both variances, log likelihood, BLUPs | zero variance reduces to LM, row and group-label permutations, unknown-group rejection |
| Generalized random intercept | SciPy QUADPACK adaptive integration and separately optimized marginal likelihood | doubled Hermite order, fixed-parameter integral probes, group relabeling, explicit conditional/marginal prediction |
| Additive | no-knot Gaussian fit reduces to independently checked OLS; scalar Gaussian ridge coefficient and effective hat trace have closed forms | centered term means, reconstruction for all three links, per-row prediction reuses training transform, no ordinary penalized Wald tests |
| Linear interventional SHAP | exhaustive coalition/intervention averages, not the attribution formula as its own oracle | efficiency on link scale, background validation, all three families |
| Normal tails | SciPy survival function at z=0 through 37, including probabilities much smaller than machine epsilon | sign symmetry, strict positivity where representable |

`golden/regression.json` has pinned NumPy 2.3.3, SciPy 1.16.2, statsmodels 0.14.5,
and pandas 2.3.3 generation inputs. Rust replays it without Python or network.
The external GLMM integrator deliberately differs from the Rust eigen-based
normal quadrature. Its finite [-12,12] normal integration interval and optimizer
tolerances are documented in the generator; this is not an arbitrary-precision oracle.

```console
cargo test -p number-ruler --all-features --locked -- --nocapture
uv run number-ruler/scripts/regression_oracle.py emit
uv run number-ruler/scripts/regression_oracle.py check
```

## Measured tolerances (2026-09-05, Windows Rust 1.98)

Scaled error means `|actual-reference| / (1+|reference|)`. Optimizer comparisons
are distinct from the configured convergence tolerance; fixture limits are
roughly four times the measured errors.

| Check | Maximum measured error | Replay limit |
|---|---:|---:|
| LM | 1.81e-14 scaled | 7.3e-14 |
| Bernoulli GLM | 2.15e-9 scaled | 8.6e-9 |
| Poisson GLM | 1.59e-14 scaled | 7.3e-14 |
| Mixed ML/REML and GLMM | 1.60e-8 scaled | 6.4e-8 |
| Normal tails | 9.24e-14 relative | 3.7e-13 |
| Fixed 128-point Bernoulli integral | 1.43e-14 absolute log likelihood | 5.8e-14 |
| Fixed 128-point Poisson integral | 5.83e-7 absolute log likelihood | 2.34e-6 |

The larger Poisson integration discrepancy is intentionally exposed: fixed
Hermite quadrature needs more care for sharply peaked count likelihoods.
Fitting checks stabilization again at its selected parameters and fails when
the requested policy tolerance is not met.

Platform variation is checked by MSRV and stable CI, not hidden by updating the
golden values to match Rust. Exact identities allow a small ULP-scale arithmetic
margin. Tests print measured discrepancies for audit.

Mixed and additive models remain **Experimental** despite these checks: only
random intercepts and the specified spline/penalty definitions are implemented;
more designs and stress cases are required before promoting their whole API.
Ordinary Wald inference after penalty selection, variance-boundary tests,
cluster/robust covariance, and interval prediction are deliberately not supplied.
