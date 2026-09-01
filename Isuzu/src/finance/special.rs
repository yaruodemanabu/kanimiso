//! Scalar special functions used by analytic prices.

/// Error function via a power series for `|x| < 1.5` and a Laplace
/// continued fraction for the complementary tail. The ATM Black–Scholes
/// laboratory needs ~1e-12 here; Abramowitz–Stegun 7.1.26 is only ~1e-7.
pub fn erf(x: f64) -> f64 {
    if !x.is_finite() {
        return if x.is_sign_positive() { 1.0 } else { -1.0 };
    }
    let ax = x.abs();
    let y = if ax < 1.5 {
        erf_series(ax)
    } else {
        (1.0 - erfc_cf(ax)).clamp(0.0, 1.0)
    };
    if x >= 0.0 {
        y
    } else {
        -y
    }
}

fn erf_series(x: f64) -> f64 {
    let x2 = x * x;
    let mut term = x;
    let mut sum = x;
    for n in 1..80 {
        term *= -x2 / n as f64;
        let add = term / (2 * n + 1) as f64;
        sum += add;
        if add.abs() < 1e-18 * (1.0 + sum.abs()) {
            break;
        }
    }
    (sum * std::f64::consts::FRAC_2_SQRT_PI).clamp(-1.0, 1.0)
}

fn erfc_cf(x: f64) -> f64 {
    // erfc(x) = e^{-x²} / (√π (x + ½/(x + 1/(x + 3/2/(x + …))))).
    let mut a = x;
    for n in (1..80).rev() {
        a = x + (0.5 * n as f64) / a;
    }
    ((-x * x).exp() / (a * std::f64::consts::PI.sqrt())).max(0.0)
}

pub fn erfc(x: f64) -> f64 {
    if !x.is_finite() {
        return if x.is_sign_positive() { 0.0 } else { 2.0 };
    }
    if x >= 0.0 {
        if x < 1.5 {
            1.0 - erf_series(x)
        } else {
            erfc_cf(x)
        }
    } else {
        2.0 - erfc(-x)
    }
}

/// Standard normal cdf `Φ`.
pub fn norm_cdf(x: f64) -> f64 {
    if !x.is_finite() {
        return if x.is_sign_positive() { 1.0 } else { 0.0 };
    }
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

/// Standard normal pdf `φ`.
pub fn norm_pdf(x: f64) -> f64 {
    if !x.is_finite() {
        return 0.0;
    }
    (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt()
}

/// Inverse standard normal via Acklam's rational approximation.
pub fn norm_inv(p: f64) -> f64 {
    if !(0.0..=1.0).contains(&p) || !p.is_finite() {
        return f64::NAN;
    }
    if p == 0.0 {
        return f64::NEG_INFINITY;
    }
    if p == 1.0 {
        return f64::INFINITY;
    }
    let a = [
        -3.969_683_028_665_376e1,
        2.209_460_984_245_205e2,
        -2.759_285_104_469_687e2,
        1.383_577_459_574_091e2,
        -3.066_479_806_614_716e1,
        2.506_628_277_459_239e0,
    ];
    let b = [
        -5.447_609_879_822_406e1,
        1.615_858_368_580_409e2,
        -1.556_989_798_598_866e2,
        6.680_131_188_771_972e1,
        -1.328_068_155_288_572e1,
    ];
    let c = [
        -7.784_894_002_430_293e-3,
        -3.223_964_580_411_365e-1,
        -2.400_758_277_161_838e0,
        -2.549_732_539_343_734e0,
        4.374_664_141_464_968e0,
        2.938_163_982_698_783e0,
    ];
    let d = [
        7.784_695_709_041_462e-3,
        3.224_671_290_700_398e-1,
        2.445_134_137_142_996e0,
        3.754_408_661_907_416e0,
    ];
    let plow = 0.02425;
    let phigh = 1.0 - plow;
    let q = if p < plow {
        let q = (-2.0 * p.ln()).sqrt();
        (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    } else if p > phigh {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    } else {
        let q = p - 0.5;
        let r = q * q;
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    };
    q
}

/// KS statistic of `u_i` against `U(0,1)` (two-sided).
pub fn ks_uniform(mut u: Vec<f64>) -> f64 {
    u.retain(|x| x.is_finite() && (0.0..=1.0).contains(x));
    if u.is_empty() {
        return 0.0;
    }
    u.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = u.len() as f64;
    let mut d = 0.0_f64;
    for (i, &x) in u.iter().enumerate() {
        let fn_ = (i + 1) as f64 / n;
        let fm = i as f64 / n;
        d = f64::max(d, f64::max((fn_ - x).abs(), (fm - x).abs()));
    }
    d
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn erf_known_values() {
        assert!(erf(0.0).abs() < 1e-15);
        assert!((erf(1.0) - 0.8427007929497149).abs() < 1e-12);
        assert!((norm_cdf(0.0) - 0.5).abs() < 1e-15);
        assert!((norm_cdf(0.35) - 0.6368306511764219).abs() < 1e-12);
        assert!(erfc(0.0) > 0.99);
        assert!(erf(8.0) > 0.999_999);
    }
}
