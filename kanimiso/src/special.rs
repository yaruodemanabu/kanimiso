//! Pure-Rust special functions used by inference (no `libm` crate; we use `f64` methods).

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
    2.0 * (1.0 - norm_cdf(z.abs())).clamp(0.0, 1.0)
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
    if x < s + 1.0 {
        // series
        let mut term = 1.0 / s;
        let mut sum = term;
        for n in 1..200 {
            term *= x / (s + n as f64);
            sum += term;
            if term.abs() < 1e-14 * sum.abs() {
                break;
            }
        }
        (sum * (-x + s * x.ln() - ln_gamma(s)).exp()).clamp(0.0, 1.0)
    } else {
        1.0 - gamma_q_cf(s, x)
    }
}

fn gamma_q_cf(s: f64, x: f64) -> f64 {
    let mut b = x + 1.0 - s;
    let mut c = 1e30;
    let mut d = 1.0 / b;
    let mut h = d;
    for i in 1..200 {
        let an = -i as f64 * (i as f64 - s);
        b += 2.0;
        d = an * d + b;
        if d.abs() < 1e-30 {
            d = 1e-30;
        }
        c = b + an / c;
        if c.abs() < 1e-30 {
            c = 1e-30;
        }
        d = 1.0 / d;
        let delta = d * c;
        h *= delta;
        if (delta - 1.0).abs() < 1e-12 {
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
    (1.0 - chi2_cdf(x, df)).clamp(0.0, 1.0)
}

/// Regularized incomplete beta via continued fraction.
pub fn betainc_reg(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let ln_beta = ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b);
    let front = (a * x.ln() + b * (1.0 - x).ln() - ln_beta).exp() / a;
    if x < (a + 1.0) / (a + b + 2.0) {
        front * beta_cf(a, b, x)
    } else {
        1.0 - (ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b)).exp().recip()
            * ((1.0 - x).powf(b) * x.powf(a) / b)
            * beta_cf(b, a, 1.0 - x)
            / (ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b))
                .exp()
                .max(1e-300)
    }
}

fn beta_cf(a: f64, b: f64, x: f64) -> f64 {
    const MAX: usize = 200;
    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < 1e-30 {
        d = 1e-30;
    }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..MAX {
        let m2 = 2.0 * m as f64;
        let mut aa = m as f64 * (b - m as f64) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < 1e-30 {
            d = 1e-30;
        }
        c = 1.0 + aa / c;
        if c.abs() < 1e-30 {
            c = 1e-30;
        }
        d = 1.0 / d;
        h *= d * c;
        aa = -(a + m as f64) * (qab + m as f64) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < 1e-30 {
            d = 1e-30;
        }
        c = 1.0 + aa / c;
        if c.abs() < 1e-30 {
            c = 1e-30;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < 1e-12 {
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
    2.0 * (1.0 - student_t_cdf(t.abs(), df)).clamp(0.0, 1.0)
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
    (1.0 - f_cdf(x, d1, d2)).clamp(0.0, 1.0)
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
}
