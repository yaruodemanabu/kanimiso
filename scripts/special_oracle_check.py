# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy>=2.0", "scipy>=1.14", "typer>=0.12"]
# ///
"""kanimiso ``special.rs`` のオラクル検査とゴールデン生成。

1. ``check``: HEAD 99c46d0 の ``betainc_reg`` を Python に逐語移植し、scipy と比較する。
   補集合分枝が ``B(a,b)`` で 2 回割っている不具合と、修正版の一致を示す。
2. ``emit``: ``special.rs`` の各関数について scipy を正解とする JSON ゴールデンを書き出す。
   Rust 側はこれをリプレイする（Tier 0）。許容差は Rust テスト側で「実測 × 3〜4」で決め、
   実測値をコメントに残す（AGENTS.md R9）。

実行例::

    uv run special_oracle_check.py check
    uv run special_oracle_check.py emit golden/special_functions.json
"""

from __future__ import annotations

import json
import math
import platform
from datetime import date
from pathlib import Path

import numpy as np
import scipy
import scipy.special as sps
import scipy.stats as sst
import typer

app = typer.Typer(add_completion=False, no_args_is_help=True)

# --------------------------------------------------------------------------- #
# kanimiso/src/special.rs (HEAD 99c46d0) の逐語移植
# --------------------------------------------------------------------------- #

_LANCZOS = [
    1.000000000190015,
    76.18009172947146,
    -86.50532032941677,
    24.01409824083091,
    -1.231739572450155,
    0.001208650973866179,
    -0.000005395239384953,
]


def ln_gamma(z: float) -> float:
    """7 項 Lanczos。special.rs と同一。"""
    x = _LANCZOS[0]
    for i in range(1, 7):
        x += _LANCZOS[i] / (z + i)
    t = z + 5.5
    return (z + 0.5) * math.log(t) - t + math.log(2.5066282746310005 * x / z)


def beta_cf(a: float, b: float, x: float) -> float:
    """Numerical Recipes 型の連分数。special.rs と同一（停止条件 1e-12, 200 反復）。"""
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1e-30 if abs(d) < 1e-30 else d
    d = 1.0 / d
    h = d
    for m in range(1, 200):
        m2 = 2.0 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1e-30 if abs(d) < 1e-30 else d
        c = 1.0 + aa / c
        c = 1e-30 if abs(c) < 1e-30 else c
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1e-30 if abs(d) < 1e-30 else d
        c = 1.0 + aa / c
        c = 1e-30 if abs(c) < 1e-30 else c
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-12:
            break
    return h


def betainc_reg_head(a: float, b: float, x: float) -> float:
    """現行実装。補集合分枝で B(a,b) により 2 回割っている（バグ）。"""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    ln_beta = ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b)
    front = math.exp(a * math.log(x) + b * math.log(1.0 - x) - ln_beta) / a
    if x < (a + 1.0) / (a + b + 2.0):
        return front * beta_cf(a, b, x)
    return 1.0 - (1.0 / math.exp(ln_beta)) * ((1.0 - x) ** b * x**a / b) * beta_cf(
        b, a, 1.0 - x
    ) / max(math.exp(ln_beta), 1e-300)


def betainc_reg_fixed(a: float, b: float, x: float) -> float:
    """修正案（AGENTS.md §4.1）。前置因子は exp(a ln x + b ln(1-x) - ln B) / b、B で割るのは 1 回。"""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    ln_beta = ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b)
    log_front = a * math.log(x) + b * math.log1p(-x) - ln_beta
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(log_front) / a * beta_cf(a, b, x)
    return 1.0 - math.exp(log_front) / b * beta_cf(b, a, 1.0 - x)


def t_pvalue_head(t: float, df: float) -> float:
    x = df / (df + t * t)
    ib = betainc_reg_head(0.5 * df, 0.5, x)
    cdf = 1.0 - 0.5 * ib if t >= 0 else 0.5 * ib
    return 2.0 * min(max(1.0 - cdf, 0.0), 1.0)


# --------------------------------------------------------------------------- #
# check
# --------------------------------------------------------------------------- #


@app.command()
def check(n_random: int = 20_000, seed: int = 0) -> None:
    """現行実装と修正案を scipy と比較して表示する。"""
    typer.echo(f"scipy {scipy.__version__} / python {platform.python_version()}\n")
    typer.echo("betainc_reg(a, b, x): HEAD vs fixed vs scipy")
    typer.echo(f"{'a':>6} {'b':>6} {'x':>6} {'HEAD':>14} {'fixed':>14} {'scipy':>14}  branch")
    for a, b, x in [(2, 3, 0.2), (2, 3, 0.8), (0.5, 2.0, 0.9), (5, 0.5, 0.99), (3, 3, 0.5), (1, 1, 0.7)]:
        branch = "cf(a,b,x)" if x < (a + 1) / (a + b + 2) else "complement"
        typer.echo(
            f"{a:>6} {b:>6} {x:>6} {betainc_reg_head(a, b, x):>14.8f} "
            f"{betainc_reg_fixed(a, b, x):>14.8f} {sps.betainc(a, b, x):>14.8f}  {branch}"
        )

    typer.echo("\nstudent_t_pvalue(t, df) two-sided: HEAD vs scipy")
    for t, df in [(0.5, 10), (1.0, 10), (1.5, 10), (2.0, 10), (1.0, 30), (1.0, 200)]:
        typer.echo(f"  t={t:<4} df={df:<4} HEAD={t_pvalue_head(t, df):.6f}  scipy={2 * sst.t.sf(t, df):.6f}")

    rng = np.random.default_rng(seed)
    worst_fixed = 0.0
    worst_head = 0.0
    for _ in range(n_random):
        a = 10 ** rng.uniform(-1, 2)
        b = 10 ** rng.uniform(-1, 2)
        x = rng.uniform(0, 1)
        ref = sps.betainc(a, b, x)
        worst_fixed = max(worst_fixed, abs(betainc_reg_fixed(a, b, x) - ref))
        worst_head = max(worst_head, abs(betainc_reg_head(a, b, x) - ref))
    typer.echo(f"\nmax |HEAD  - scipy| over {n_random} random cases: {worst_head:.2e}")
    typer.echo(f"max |fixed - scipy| over {n_random} random cases: {worst_fixed:.2e}")
    typer.echo("残差は 7 項 Lanczos ln_gamma と 1e-12 の連分数停止条件由来。1e-13 が要るなら両方を締める。")


# --------------------------------------------------------------------------- #
# emit
# --------------------------------------------------------------------------- #


def _cases_grid(*axes: list[float]) -> list[list[float]]:
    out: list[list[float]] = []

    def rec(prefix: list[float], rest: tuple[list[float], ...]) -> None:
        if not rest:
            out.append(prefix)
            return
        for v in rest[0]:
            rec([*prefix, v], rest[1:])

    rec([], axes)
    return out


@app.command()
def emit(path: Path, n_random: int = 200, seed: int = 0) -> None:
    """special.rs 全関数の scipy ゴールデンを JSON で書き出す。"""
    rng = np.random.default_rng(seed)
    z_grid = [0.1, 0.5, 1.0, 1.5, 2.0, 5.0, 10.0, 50.0, 100.0, 170.0]
    x_unit = [1e-6, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 1 - 1e-6]
    ab_grid = [0.1, 0.5, 1.0, 2.0, 3.0, 10.0, 50.0]
    df_grid = [1.0, 2.0, 5.0, 10.0, 30.0, 200.0]
    t_grid = [-5.0, -2.5, -1.5, -1.0, -0.5, 0.0, 0.3, 1.0, 1.5, 2.0, 2.5, 5.0]

    cases: list[dict] = []

    def add(fn: str, args: list[float], expected: float) -> None:
        if math.isfinite(expected):
            cases.append({"fn": fn, "args": args, "expected": float(expected)})

    for z in [-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0, 6.0]:
        add("erf", [z], sps.erf(z))
        add("norm_cdf", [z], sst.norm.cdf(z))
    for z in z_grid:
        add("ln_gamma", [z], sps.gammaln(z))
        add("digamma", [z], sps.digamma(z))
    for s, x in _cases_grid(z_grid[:8], [0.01, 0.5, 1.0, 2.0, 5.0, 20.0, 100.0]):
        add("gamma_p", [s, x], sps.gammainc(s, x))
    for a, b, x in _cases_grid(ab_grid, ab_grid, x_unit):
        add("betainc_reg", [a, b, x], sps.betainc(a, b, x))
    for _ in range(n_random):
        a, b, x = 10 ** rng.uniform(-1, 2), 10 ** rng.uniform(-1, 2), rng.uniform(0, 1)
        add("betainc_reg", [float(a), float(b), float(x)], sps.betainc(a, b, x))
    for x, df in _cases_grid([0.1, 1.0, 3.0, 10.0, 50.0], df_grid):
        add("chi2_cdf", [x, df], sst.chi2.cdf(x, df))
    for t, df in _cases_grid(t_grid, df_grid):
        add("student_t_cdf", [t, df], sst.t.cdf(t, df))
        add("student_t_pvalue", [t, df], 2 * sst.t.sf(abs(t), df))
    for x, d1, d2 in _cases_grid([0.2, 0.5, 1.0, 2.0, 4.0, 10.0], [1.0, 3.0, 5.0, 20.0], [5.0, 20.0, 50.0, 200.0]):
        add("f_cdf", [x, d1, d2], sst.f.cdf(x, d1, d2))
        add("f_pvalue", [x, d1, d2], sst.f.sf(x, d1, d2))

    payload = {
        "generator": "special_oracle_check.py emit",
        "created": date.today().isoformat(),
        "oracle": {"scipy": scipy.__version__, "numpy": np.__version__, "python": platform.python_version()},
        "seed": seed,
        "tolerance_policy": "Rust 側で実測誤差 × 3〜4 を閾値にし、実測値をテストのコメントに残す（AGENTS.md R9）",
        "n_cases": len(cases),
        "cases": cases,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    typer.echo(f"wrote {len(cases)} cases -> {path}")


if __name__ == "__main__":
    app()
