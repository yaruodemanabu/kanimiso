#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["hmmlearn==0.3.3", "numpy"]
# ///
"""Emit golden/hmm_core.json from hmmlearn (AGENTS.md §6 Tier 0)."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from hmmlearn.hmm import CategoricalHMM, GaussianHMM, PoissonHMM


def brute_loglik(start, trans, log_emit):
    t_len = len(log_emit)
    k = len(start)
    log_start = [math.log(p) if p > 0 else -math.inf for p in start]
    log_a = [
        [math.log(trans[i][j]) if trans[i][j] > 0 else -math.inf for j in range(k)]
        for i in range(k)
    ]
    best = -math.inf
    # enumerate paths
    n_paths = k**t_len
    acc = []
    for code in range(n_paths):
        path = []
        c = code
        for _ in range(t_len):
            path.append(c % k)
            c //= k
        lp = log_start[path[0]] + log_emit[0][path[0]]
        for t in range(1, t_len):
            lp += log_a[path[t - 1]][path[t]] + log_emit[t][path[t]]
        acc.append(lp)
    m = max(acc)
    if not math.isfinite(m):
        return m
    s = sum(math.exp(v - m) for v in acc)
    return m + math.log(s)


def main() -> None:
    cases = []

    g = GaussianHMM(n_components=2, covariance_type="diag", init_params="", params="")
    g.startprob_ = np.array([0.6, 0.4])
    g.transmat_ = np.array([[0.7, 0.3], [0.2, 0.8]])
    g.means_ = np.array([[-1.0], [2.0]])
    g.n_features = 1
    g.covars_ = np.array([[0.25], [0.49]])
    xg = np.array([[-1.1], [-0.8], [1.9], [2.2], [-0.9], [2.1]])
    cases.append(
        {
            "name": "gaussian_diag_t6",
            "kind": "gaussian",
            "start": g.startprob_.tolist(),
            "trans": g.transmat_.tolist(),
            "means": g.means_.tolist(),
            "vars": np.asarray(g.covars_).reshape(g.n_components, -1).tolist(),
            "obs": xg.tolist(),
            "loglik": float(g.score(xg)),
            "viterbi": [int(v) for v in g.predict(xg)],
        }
    )

    xg4 = xg[:4]
    cases.append(
        {
            "name": "gaussian_diag_t4",
            "kind": "gaussian",
            "start": g.startprob_.tolist(),
            "trans": g.transmat_.tolist(),
            "means": g.means_.tolist(),
            "vars": np.asarray(g.covars_).reshape(g.n_components, -1).tolist(),
            "obs": xg4.tolist(),
            "loglik": float(g.score(xg4)),
            "viterbi": [int(v) for v in g.predict(xg4)],
        }
    )

    p = PoissonHMM(n_components=2, init_params="", params="")
    p.startprob_ = np.array([0.55, 0.45])
    p.transmat_ = np.array([[0.8, 0.2], [0.3, 0.7]])
    p.lambdas_ = np.array([[1.0], [4.0]])
    xp = np.array([[0], [1], [4], [5], [1], [3]])
    cases.append(
        {
            "name": "poisson_t6",
            "kind": "poisson",
            "start": p.startprob_.tolist(),
            "trans": p.transmat_.tolist(),
            "rates": [float(p.lambdas_[0, 0]), float(p.lambdas_[1, 0])],
            "obs": [float(v[0]) for v in xp],
            "loglik": float(p.score(xp)),
            "viterbi": [int(v) for v in p.predict(xp)],
        }
    )

    c = CategoricalHMM(n_components=2, init_params="", params="")
    c.startprob_ = np.array([0.5, 0.5])
    c.transmat_ = np.array([[0.9, 0.1], [0.25, 0.75]])
    c.emissionprob_ = np.array([[0.8, 0.15, 0.05], [0.1, 0.2, 0.7]])
    xc = np.array([[0], [0], [2], [2], [1], [2]])
    cases.append(
        {
            "name": "categorical_t6",
            "kind": "categorical",
            "start": c.startprob_.tolist(),
            "trans": c.transmat_.tolist(),
            "emission": c.emissionprob_.tolist(),
            "obs": [int(v[0]) for v in xc],
            "loglik": float(c.score(xc)),
            "viterbi": [int(v) for v in c.predict(xc)],
        }
    )

    xh = np.array([[100.0], [-100.0], [100.0], [-100.0], [100.0], [-100.0]])
    cases.append(
        {
            "name": "gaussian_far_t6",
            "kind": "gaussian",
            "start": g.startprob_.tolist(),
            "trans": g.transmat_.tolist(),
            "means": g.means_.tolist(),
            "vars": np.asarray(g.covars_).reshape(g.n_components, -1).tolist(),
            "obs": xh.tolist(),
            "loglik": float(g.score(xh)),
            "viterbi": [int(v) for v in g.predict(xh)],
        }
    )

    out = {
        "hmmlearn": "0.3.3",
        "note": "score() is sequence log-likelihood; viterbi is predict()",
        "cases": cases,
    }
    dest = Path(__file__).resolve().parents[1] / "golden" / "hmm_core.json"
    dest.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {dest} ({len(cases)} cases)")


if __name__ == "__main__":
    main()
