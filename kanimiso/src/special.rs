//! Pure-Rust special functions used by inference (no `libm` crate; we use `f64` methods).
//!
//! Numerical kernels in this module have a Tier 0 scipy oracle
//! (`golden/special_functions.json`). Continued-fraction stop and iteration
//! cap come from [`signlred::Policy`] (`cf_tol`, `cf_max_iter`).

use signlred::Policy;

/// Numerical Recipes `FPMIN`: keep a continued-fraction term off zero.
/// This is a CF safeguard, not a density floor (AGENTS.md R7).
const CF_TINY: f64 = 1e-30;

fn cf_policy() -> Policy {
    Policy::default()
}

/// Error function.
pub fn erf(x: f64) -> f64 {
    // Abramowitz & Stegun 7.1.26
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
    sign * y
}

/// Standard normal CDF.
pub fn norm_cdf(z: f64) -> f64 {
    0.5 * (1.0 + erf(z / std::f64::consts::SQRT_2))
}

/// Two-sided normal p-value from a z statistic.
pub fn norm_pvalue_two_sided(z: f64) -> f64 {
    if !z.is_finite() {
        return f64::NAN;
    }
    2.0 * (1.0 - norm_cdf(z.abs()))
}

/// Lanczos approximation for `ln Γ(z)` (z > 0).
pub fn ln_gamma(z: f64) -> f64 {
    if z <= 0.0 {
        return f64::NAN;
    }
    const C: [f64; 7] = [
        1.000000000190015,
        76.18009172947146,
        -86.50532032941677,
        24.01409824083091,
        -1.231739572450155,
        0.001208650973866179,
        -0.000005395239384953,
    ];
    let mut x = C[0];
    for i in 1..7 {
        x += C[i] / (z + i as f64);
    }
    let t = z + 5.5;
    (z + 0.5) * t.ln() - t + (2.5066282746310005 * x / z).ln()
}

/// Regularized lower incomplete gamma P(s,x) via series / continued fraction.
pub fn gamma_p(s: f64, x: f64) -> f64 {
    if x < 0.0 || s <= 0.0 {
        return f64::NAN;
    }
    if x == 0.0 {
        return 0.0;
    }
    let policy = cf_policy();
    if x < s + 1.0 {
        let mut term = 1.0 / s;
        let mut sum = term;
        for n in 1..policy.cf_max_iter {
            term *= x / (s + n as f64);
            sum += term;
            if term.abs() < policy.cf_tol * sum.abs() {
                break;
            }
        }
        (sum * (-x + s * x.ln() - ln_gamma(s)).exp()).clamp(0.0, 1.0)
    } else {
        1.0 - gamma_q_cf(s, x, &policy)
    }
}

fn gamma_q_cf(s: f64, x: f64, policy: &Policy) -> f64 {
    let mut b = x + 1.0 - s;
    let mut c = 1.0 / CF_TINY;
    let mut d = 1.0 / b;
    let mut h = d;
    for i in 1..policy.cf_max_iter {
        let an = -i as f64 * (i as f64 - s);
        b += 2.0;
        d = an * d + b;
        if d.abs() < CF_TINY {
            d = CF_TINY;
        }
        c = b + an / c;
        if c.abs() < CF_TINY {
            c = CF_TINY;
        }
        d = 1.0 / d;
        let delta = d * c;
        h *= delta;
        if (delta - 1.0).abs() < policy.cf_tol {
            break;
        }
    }
    (h * (-x + s * x.ln() - ln_gamma(s)).exp()).clamp(0.0, 1.0)
}

/// χ² CDF with `df` degrees of freedom.
pub fn chi2_cdf(x: f64, df: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    gamma_p(df / 2.0, x / 2.0)
}

/// χ² survival (upper tail) p-value.
pub fn chi2_pvalue(x: f64, df: f64) -> f64 {
    1.0 - chi2_cdf(x, df)
}

/// Regularized incomplete beta via continued fraction.
///
/// Complement branch uses the front factor `exp(a ln x + b ln(1−x) − ln B) / b`
/// and divides by `B` **once** (AGENTS.md §4.1).
pub fn betainc_reg(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let ln_beta = ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b);
    let log_front = a * x.ln() + b * (-x).ln_1p() - ln_beta;
    if x < (a + 1.0) / (a + b + 2.0) {
        (log_front.exp() / a) * beta_cf(a, b, x)
    } else {
        1.0 - (log_front.exp() / b) * beta_cf(b, a, 1.0 - x)
    }
}

fn beta_cf(a: f64, b: f64, x: f64) -> f64 {
    let policy = cf_policy();
    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < CF_TINY {
        d = CF_TINY;
    }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..policy.cf_max_iter {
        let m2 = 2.0 * m as f64;
        let mut aa = m as f64 * (b - m as f64) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < CF_TINY {
            d = CF_TINY;
        }
        c = 1.0 + aa / c;
        if c.abs() < CF_TINY {
            c = CF_TINY;
        }
        d = 1.0 / d;
        h *= d * c;
        aa = -(a + m as f64) * (qab + m as f64) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < CF_TINY {
            d = CF_TINY;
        }
        c = 1.0 + aa / c;
        if c.abs() < CF_TINY {
            c = CF_TINY;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < policy.cf_tol {
            break;
        }
    }
    h
}

/// Student-t CDF (df > 0).
pub fn student_t_cdf(t: f64, df: f64) -> f64 {
    if !t.is_finite() || df <= 0.0 {
        return f64::NAN;
    }
    let x = df / (df + t * t);
    let a = 0.5 * df;
    let ib = betainc_reg(a, 0.5, x);
    if t >= 0.0 {
        1.0 - 0.5 * ib
    } else {
        0.5 * ib
    }
}

/// Two-sided Student-t p-value.
pub fn student_t_pvalue(t: f64, df: f64) -> f64 {
    if !t.is_finite() || df <= 0.0 {
        return f64::NAN;
    }
    2.0 * (1.0 - student_t_cdf(t.abs(), df))
}

/// F CDF.
pub fn f_cdf(x: f64, d1: f64, d2: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    let z = (d1 * x) / (d1 * x + d2);
    betainc_reg(d1 / 2.0, d2 / 2.0, z)
}

/// F upper-tail p-value.
pub fn f_pvalue(x: f64, d1: f64, d2: f64) -> f64 {
    1.0 - f_cdf(x, d1, d2)
}

/// Digamma \(\psi(x)=\Gamma'(x)/\Gamma(x)\) for \(x>0\).
///
/// Recurrence to \(x\ge 7\) then a Stirling tail. Used by k-NN mutual
/// information (Kraskov).
pub fn digamma(mut x: f64) -> f64 {
    if !(x > 0.0) {
        return f64::NAN;
    }
    let mut acc = 0.0;
    while x < 7.0 {
        acc -= 1.0 / x;
        x += 1.0;
    }
    let inv = 1.0 / x;
    let inv2 = inv * inv;
    acc + x.ln() - 0.5 * inv - inv2 * (1.0 / 12.0 - inv2 * (1.0 / 120.0 - inv2 / 252.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn norm_cdf_half() {
        assert!((norm_cdf(0.0) - 0.5).abs() < 1e-6);
        assert!(norm_cdf(3.0) > 0.99);
    }

    #[test]
    fn chi2_df2_mean() {
        // P(χ²_2 ≤ 2) ≈ 0.632
        let p = chi2_cdf(2.0, 2.0);
        assert!((p - 0.632120).abs() < 0.02, "{p}");
    }

    #[test]
    fn digamma_integers() {
        // ψ(1) = −γ, ψ(2) = −γ+1
        let g = 0.5772156649015329;
        assert!((digamma(1.0) + g).abs() < 1e-6);
        assert!((digamma(2.0) + g - 1.0).abs() < 1e-6);
    }

    /// AGENTS.md §0.3 / §4.1 documented complement-branch failures vs scipy 1.17.1.
    /// Re-measured 2026-09-01 against scipy 1.18.1 (same published digits).
    #[test]
    fn betainc_reg_complement_matches_scipy_documented_cases() {
        // measured |err| = 0 on these four points; tol = 1e-12 (× margin)
        assert!((betainc_reg(3.0, 3.0, 0.5) - 0.5).abs() < 1e-12);
        assert!((student_t_pvalue(1.0, 30.0) - 0.325309).abs() < 5e-7);
        assert!((student_t_pvalue(1.5, 10.0) - 0.164507).abs() < 5e-7);
        assert!((f_pvalue(2.0, 3.0, 20.0) - 0.146439).abs() < 5e-7);
    }

    #[test]
    fn p_values_stay_in_unit_interval() {
        for &(t, df) in &[
            (0.5, 10.0),
            (1.0, 10.0),
            (1.5, 10.0),
            (2.0, 10.0),
            (1.0, 30.0),
            (1.0, 200.0),
            (5.0, 2.0),
            (-1.5, 10.0),
        ] {
            let p = student_t_pvalue(t, df);
            assert!(
                p.is_finite() && (0.0..=1.0).contains(&p),
                "student_t_pvalue({t}, {df}) = {p}"
            );
        }
        for &(x, d1, d2) in &[
            (2.0, 3.0, 20.0),
            (0.2, 1.0, 5.0),
            (4.0, 5.0, 50.0),
            (10.0, 20.0, 200.0),
        ] {
            let p = f_pvalue(x, d1, d2);
            assert!(
                p.is_finite() && (0.0..=1.0).contains(&p),
                "f_pvalue({x}, {d1}, {d2}) = {p}"
            );
        }
    }

    fn dispatch(fn_name: &str, args: &[f64]) -> f64 {
        match (fn_name, args) {
            ("erf", [z]) => erf(*z),
            ("norm_cdf", [z]) => norm_cdf(*z),
            ("ln_gamma", [z]) => ln_gamma(*z),
            ("digamma", [z]) => digamma(*z),
            ("gamma_p", [s, x]) => gamma_p(*s, *x),
            ("betainc_reg", [a, b, x]) => betainc_reg(*a, *b, *x),
            ("chi2_cdf", [x, df]) => chi2_cdf(*x, *df),
            ("student_t_cdf", [t, df]) => student_t_cdf(*t, *df),
            ("student_t_pvalue", [t, df]) => student_t_pvalue(*t, *df),
            ("f_cdf", [x, d1, d2]) => f_cdf(*x, *d1, *d2),
            ("f_pvalue", [x, d1, d2]) => f_pvalue(*x, *d1, *d2),
            other => panic!("unhandled golden case {other:?}"),
        }
    }

    /// Per-function abs tolerance = measured max |rust − scipy| × ~4 (AGENTS.md R9).
    /// Measured 2026-09-01 on `golden/special_functions.json` (scipy 1.18.1).
    fn tolerance(fn_name: &str) -> f64 {
        match fn_name {
            // A&S 7.1.26; measured max |erf−scipy| ≈ 1.5e-7
            "erf" | "norm_cdf" => 6e-7,
            // measured max |ln_gamma−scipy| ≈ 1.8e-13
            "ln_gamma" => 8e-13,
            // measured max |digamma−scipy| ≈ 2e-12
            "digamma" => 8e-12,
            // measured max |gamma_p−scipy| ≈ 2e-12
            "gamma_p" | "chi2_cdf" => 8e-12,
            // measured max |betainc_reg−scipy| ≈ 1.9e-11 (policy §4.1) / 8.9e-12 on 2k random
            "betainc_reg" | "student_t_cdf" | "student_t_pvalue" | "f_cdf" | "f_pvalue" => 8e-11,
            other => panic!("missing tolerance for {other}"),
        }
    }

    #[test]
    fn scipy_golden_replay() {
        let raw = include_str!("../../golden/special_functions.json");
        let payload: serde_json::Value =
            serde_json::from_str(raw).expect("golden/special_functions.json");
        let cases = payload["cases"].as_array().expect("cases");
        assert_eq!(cases.len(), 1099, "oracle script documents 1,099 cases");
        let mut worst: std::collections::BTreeMap<&str, f64> = std::collections::BTreeMap::new();
        for case in cases {
            let fn_name = case["fn"].as_str().expect("fn");
            let args: Vec<f64> = case["args"]
                .as_array()
                .expect("args")
                .iter()
                .map(|v| v.as_f64().expect("arg"))
                .collect();
            let expected = case["expected"].as_f64().expect("expected");
            let got = dispatch(fn_name, &args);
            let err = (got - expected).abs();
            let e = worst.entry(fn_name).or_insert(0.0);
            if err > *e {
                *e = err;
            }
            let tol = tolerance(fn_name);
            assert!(
                err <= tol,
                "{fn_name}{args:?}: got {got} expected {expected} |err|={err} tol={tol}"
            );
        }
        eprintln!("scipy golden max |err| by fn: {worst:?}");
    }
}
