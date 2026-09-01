# Changelog

## Unreleased

### Added

- **Shreve finance layer** (`src/finance/`): terminal / path payoffs, Black–Scholes
  prices and Greeks, discounted Monte Carlo (antithetic, European control,
  importance sampling), CRR trees, BS PDE (explicit / implicit / CN + Rannacher,
  American PSOR), Longstaff–Schwartz and Andersen–Broadie, Vasicek / CIR bonds,
  Hull–White $\theta(t)$ fit, Black caplet / swaption, Merton series / IMEX
  PIDE / jump MC, Kou mixture, Esscher tilt, Brownian-bridge barrier
  correction, Dixit–Pindyck / McDonald–Siegel real options, Margrabe exchange,
  CRR swing / storage, Kamrad–Ritchken trinomial, Hermite / standardized LSM,
  Haugh–Kogan dual, T-forward prices, and a discrete delta hedge.
- **Energy / commodity SDEs**: Schwartz (1997) one-factor, Schwartz–Smith,
  Lucia–Schwartz seasonality, Cartea–Figueroa spikes, Gibson–Schwartz
  convenience yield, two-regime switching, spark-spread payoff, plus
  closed-form futures for the one-factor, SS, and GS models.
- **`ParticleModel`**: Tobit / Student-t / Poisson observations, PMMH, SMC²,
  conditional SMC (particle Gibbs). Missing observations are `NaN` and skipped
  by the Kalman update.
- **Linear-Gaussian estimation**: KF innovation MLE (`ssm_mle`) and
  Shumway–Stoffer EM with RTS lag-1 covariances; Durbin–Koopman diffuse prior;
  CARMA $\to$ linear SSM wiring.
- **Inference extras**: L-BFGS-B, `Fit::{vcov,se}`, Uchida CIC, two-stage /
  threshold / Kessler QMLE, QGV / MM Hurst, COGARCH GMM.
- **N-d exponential Hawkes** with an analytic likelihood gradient and Ogata
  time-change KS.
- **amatsuki**: Marsaglia–Tsang Ziggurat $N(0,1)$, `set_stream` / `jump_ahead`,
  Bernoulli / Binomial / Categorical / Multinomial, non-central $\chi^2$.
- **MC / control / HFT**: Sobol' + Brownian-bridge QMC, implicit HJB and
  Kushner–Dupuis, CIR ncx2 exact and Andersen QE, BNS ratio / tripower,
  Lee–Mykland, LOB OFI.

The `ocrs_iym` energy / real-option OCR books are not readable with the
available GitHub credentials (private / 404). The energy and real-option
modules follow the standard Schwartz–Smith, Lucia–Schwartz, Cartea–Figueroa,
Gibson–Schwartz, and Dixit–Pindyck / McDonald–Siegel constructions instead.

### Changed

- CI `test` / `msrv` jobs run `cargo test --workspace` so amatsuki is not
  skipped.
- `docs/deviations.md` records the new finance / energy / ParticleModel
  departures. A1–A8 P0 math bugs remain those merged in #6.

### Fixed

- Longstaff–Schwartz discounted OTM cashflows twice per exercise date (the
  collection loop and the leftover loop).
- `qr_least_squares(..., Some(ε))` always took the ridge path and rebuilt
  $A^\top A$ from the destroyed Householder workspace. $\varepsilon$ is now
  only the rank-deficient fallback, using the column-scaled design.
- LSM default basis is $1,S/K,(S/K)^2$ (weighted Laguerre under-exercises the
  Table 1 put). The estimator now sits near Longstaff–Schwartz 4.478 and
  between the European put and the Haugh–Kogan dual.
- `Cir::sample_exact` mixed the two ncx2 conventions and had mean
  $X_0 e^{-\kappa t}+2\theta(1-e^{-\kappa t})$. It now uses Glasserman's
  $X'=c\,\chi^2_d(\lambda)$ so the mean is $\theta$.

### Not in this drop

- Full Uchida–Yoshida adaptive Bayes, non-commutative Lévy-area Milstein, and
  a QR / Potter square-root Kalman. Filter covariances still use
  `spd_regularize`. Kou remains a moment-matched mixture (not the transform).
  LSM nested Andersen–Broadie is still a one-level sketch; the dual used in
  tests is the European-put martingale.
