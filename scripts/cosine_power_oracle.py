#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "scipy", "mpmath"]
# ///
"""Emit golden/cosine_power.json from scipy.betainc + mpmath.quad."""
from __future__ import annotations

import json
import math
from pathlib import Path

import mpmath as mp
import numpy as np
from scipy.special import betainc, gammaln


def log_z(n: float) -> float:
    return math.log(2.0) - 0.5 * math.log(math.pi) + gammaln((n + 1.0) / 2.0) - gammaln(n / 2.0 + 1.0)


def cosine_log_prob(y: float, loc: float, scale: float, n: float) -> float:
    u = (y - loc) / scale
    theta = 0.5 * math.pi * u
    if abs(theta) >= 0.5 * math.pi:
        return float("-inf")
    c = math.cos(theta)
    if c <= 0.0:
        return float("-inf")
    return n * math.log(c) - log_z(n) - math.log(scale)


def cosine_cdf_betainc(y: float, loc: float, scale: float, n: float) -> float:
    u = (y - loc) / scale
    theta = 0.5 * math.pi * u
    if theta <= -0.5 * math.pi:
        return 0.0
    if theta >= 0.5 * math.pi:
        return 1.0
    s2 = math.sin(theta) ** 2
    inc = float(betainc(0.5, (n + 1.0) / 2.0, s2))
    return 0.5 + 0.5 * math.copysign(1.0, theta) * inc


def cosine_log_sf(y: float, loc: float, scale: float, n: float) -> float:
    u = (y - loc) / scale
    theta = 0.5 * math.pi * u
    if theta >= 0.5 * math.pi:
        return float("-inf")
    if theta <= -0.5 * math.pi:
        return 0.0
    if theta >= 0.0:
        c2 = math.cos(theta) ** 2
        inc = float(betainc((n + 1.0) / 2.0, 0.5, c2))
        return math.log(0.5 * inc) if inc > 0 else float("-inf")
    return math.log1p(-cosine_cdf_betainc(y, loc, scale, n))


def main() -> None:
    mp.mp.dps = 40
    loc, scale = 0.0, 1.0
    cases = []
    for n in (0.0, 2.0, 3.0, 5.5):
        for y in (-0.8, -0.3, 0.0, 0.25, 0.8, 0.999):
            lp = cosine_log_prob(y, loc, scale, n)
            cdf = cosine_cdf_betainc(y, loc, scale, n)
            lsf = cosine_log_sf(y, loc, scale, n)
            # mpmath integral of the density on [-1, y]
            n_mp, y_mp = mp.mpf(n), mp.mpf(y)

            def f(t):
                th = mp.pi / 2 * t
                zlog = (
                    mp.log(2)
                    - mp.log(mp.pi) / 2
                    + mp.loggamma((n_mp + 1) / 2)
                    - mp.loggamma(n_mp / 2 + 1)
                )
                return mp.exp(n_mp * mp.log(mp.cos(th)) - zlog)

            integ = float(mp.quad(f, [-1, y_mp]))
            cases.append(
                {
                    "kind": "cosine",
                    "n": n,
                    "y": y,
                    "loc": loc,
                    "scale": scale,
                    "log_prob": lp,
                    "cdf": cdf,
                    "log_sf": lsf,
                    "mpmath_cdf": integ,
                }
            )
    dest = Path(__file__).resolve().parents[1] / "golden" / "cosine_power.json"
    dest.write_text(
        json.dumps({"scipy": "1.18.1", "mpmath": "1.4.1", "cases": cases}, indent=2) + "\n"
    )
    print(f"wrote {dest} ({len(cases)} cases)")


if __name__ == "__main__":
    main()
