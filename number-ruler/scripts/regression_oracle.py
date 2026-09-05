# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["numpy==2.3.3", "scipy==1.16.2", "statsmodels==0.14.5", "pandas==2.3.3"]
# ///
"""Independent external regression oracle. Runtime Rust never imports Python.

Regenerate: uv run number-ruler/scripts/regression_oracle.py emit
Verify: uv run number-ruler/scripts/regression_oracle.py check
GLMM uses adaptive QUADPACK integration, not the Rust Hermite/eigen algorithm.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import warnings

import numpy as np
import scipy
from scipy.integrate import quad
from scipy.optimize import minimize
from scipy.special import expit, gammaln
from scipy.stats import norm
import statsmodels
import statsmodels.api as sm

DESTINATION = Path(__file__).resolve().parents[1] / "golden" / "regression.json"


def build():
    cases = []
    x = np.array([[((i * 7) % 23 - 11) / 5, ((i * 11) % 19 - 9) / 7]
                  for i in range(48)], dtype=float)
    design = sm.add_constant(x)
    responses = {
        "Gaussian": 1.2 + x @ np.array([0.7, -0.3]) + np.array([((i * 13) % 17 - 8) / 8 for i in range(48)]),
        "Binomial": np.array([float((i * 17) % 47 / 47 < expit(0.2 + 0.6 * a - 0.35 * b)) for i, (a, b) in enumerate(x)]),
        "Poisson": np.array([((i * 13) % 7) + int(a > 0) for i, (a, _) in enumerate(x)], dtype=float),
    }
    for family, y in responses.items():
        if family == "Gaussian":
            fit = sm.OLS(y, design).fit()
            deviance = float(fit.ssr)
        else:
            distribution = getattr(sm.families, family)()
            fit = sm.GLM(y, design, family=distribution).fit(tol=1e-13, maxiter=500)
            deviance = float(fit.deviance)
        cases.append(dict(kind="regression", family=family, x=x.tolist(), y=y.tolist(),
                          beta=fit.params.tolist(), se=fit.bse.tolist(), p=fit.pvalues.tolist(),
                          fitted=fit.fittedvalues.tolist(), covariance=fit.cov_params().tolist(),
                          deviance=deviance))

    groups = np.repeat(np.arange(10), 6)
    xm = np.array([[(j - 2.5) / 2 + (g % 3) / 4] for g in range(10) for j in range(6)])
    dm = sm.add_constant(xm)
    ym = np.array([1.4 + 0.8 * xm[i, 0] + (int(g) % 5 - 2) * 0.6
                   + ((i * 7) % 13 - 6) / 9 for i, g in enumerate(groups)])
    for reml in (False, True):
        fit = sm.MixedLM(ym, dm, groups).fit(reml=reml, method="bfgs", gtol=1e-11, maxiter=2000)
        cases.append(dict(kind="mixed", family="Gaussian", likelihood="Restricted" if reml else "Maximum",
                          x=xm.tolist(), y=ym.tolist(), groups=groups.tolist(),
                          beta=fit.fe_params.tolist(), residual_variance=float(fit.scale),
                          random_variance=float(fit.cov_re[0, 0]), loglik=float(fit.llf),
                          effects=[float(fit.random_effects[g].iloc[0] if hasattr(fit.random_effects[g], "iloc") else fit.random_effects[g][0]) for g in range(10)]))

    # Fixed-parameter integral probes isolate integration correctness from optimizer agreement.
    for family in ("Binomial", "Poisson"):
        y = np.array([float((i * 17) % 59 / 59 < expit(0.1 + 0.3 * xm[i, 0] + (int(g) % 5 - 2) * 0.5))
                      if family == "Binomial" else float((i * 7) % 4 + (int(g) % 3)) for i, g in enumerate(groups)])

        def likelihood(parameters):
            beta, sd = parameters[:2], np.exp(parameters[2])
            eta = dm @ beta
            loglik = 0.0
            for group in range(10):
                rows = groups == group
                def integrand(z):
                    shifted = eta[rows] + sd * z
                    if family == "Binomial":
                        logdensity = y[rows] * shifted - np.logaddexp(0, shifted)
                    else:
                        with np.errstate(over="ignore", invalid="ignore"):
                            logdensity = y[rows] * shifted - np.exp(shifted) - gammaln(y[rows] + 1)
                    return np.exp(logdensity.sum() - z * z / 2) / np.sqrt(2 * np.pi)
                value, _ = quad(integrand, -12, 12, epsabs=1e-13, epsrel=1e-12)
                if value <= 0:
                    return np.inf
                loglik += np.log(value)
            return -loglik

        initial = sm.GLM(y, dm, family=getattr(sm.families, family)()).fit(tol=1e-13)
        trials = [minimize(likelihood, [*initial.params, np.log(sd)], method="Nelder-Mead",
                           options=dict(maxiter=3000, xatol=1e-9, fatol=1e-11)) for sd in (0.3, 0.8)]
        best = min(trials, key=lambda fit: fit.fun)
        probes = [dict(beta=[0.1, 0.3], sd=sd, loglik=-likelihood(np.array([0.1, 0.3, np.log(sd)]))) for sd in (0.2, 0.5, 0.8)]
        cases.append(dict(kind="glmm", family=family, x=xm.tolist(), y=y.tolist(), groups=groups.tolist(),
                          beta=best.x[:2].tolist(), random_variance=float(np.exp(2 * best.x[2])),
                          loglik=float(-best.fun), probes=probes, converged=bool(best.success)))

    return dict(provenance=dict(numpy=np.__version__, scipy=scipy.__version__, statsmodels=statsmodels.__version__,
                               command="uv run number-ruler/scripts/regression_oracle.py emit",
                               mixed="statsmodels MixedLM ML/REML; GLMM scipy quad + independently optimized likelihood"),
                normal_tails=[dict(z=z, p=float(2 * norm.sf(z))) for z in (0, 0.01, 0.5, 1, 2, 4, 8, 12, 20, 30, 37)],
                cases=cases)


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = build()
    if sys.argv[1:] == ["check"]:
        previous = json.loads(DESTINATION.read_text())
        # Numerical libraries can differ in the last few optimizer digits across platforms.
        def compare(a, b):
            if isinstance(a, dict):
                assert a.keys() == b.keys()
                for key in a:
                    compare(a[key], b[key])
            elif isinstance(a, list):
                assert len(a) == len(b)
                for left, right in zip(a, b):
                    compare(left, right)
            elif isinstance(a, float):
                assert np.isclose(a, b, rtol=1e-7, atol=1e-9), (a, b)
            else:
                assert a == b, (a, b)
        compare(result, previous)
        print("regression external oracle reproduced")
    elif sys.argv[1:] == ["emit"]:
        DESTINATION.parent.mkdir(parents=True, exist_ok=True)
        DESTINATION.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(DESTINATION)
    else:
        raise SystemExit("expected emit or check")
