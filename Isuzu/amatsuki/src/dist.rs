//! Sampling algorithms for the laws used by Isuzu.

use crate::rng::{Distribution, Rng};

/// Recoverable parameter error for a sampling law.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DistError {
    /// Static reason.
    pub message: &'static str,
}

impl DistError {
    pub(crate) fn new(message: &'static str) -> Self {
        Self { message }
    }
}

impl core::fmt::Display for DistError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.message)
    }
}

impl std::error::Error for DistError {}

/// Uniform sample in `(0, 1)`.
#[derive(Clone, Copy, Debug, Default)]
pub struct Open01;

impl Distribution for Open01 {
    type Value = f64;
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        // 53-bit midpoints: (k + ½) / 2^53 ∈ (0, 1). Never hits 0 or 1.
        let k = rng.next_u64() >> 11;
        (k as f64 + 0.5) * (1.0 / ((1u64 << 53) as f64))
    }
}

/// Uniform sample in `(0, 1]`.
#[derive(Clone, Copy, Debug, Default)]
pub struct OpenClosed01;

impl Distribution for OpenClosed01 {
    type Value = f64;
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        // rand 0.8 OpenClosed01: (u + 1) / 2^53 ∈ (0, 1].
        let precision = 53;
        let scale = 1.0 / ((1u64 << precision) as f64);
        let value = (rng.next_u64() >> (64 - precision)) as f64 * scale;
        value + scale
    }
}

/// Uniform continuous law on `[low, high)`.
#[derive(Clone, Copy, Debug)]
pub struct Uniform {
    low: f64,
    span: f64,
}

impl Uniform {
    /// Inclusive-exclusive interval `[low, high)`.
    ///
    /// Panics if `high ≤ low` or either endpoint is non-finite (same contract
    /// as the previous `rand_distr` constructor used throughout Isuzu).
    pub fn new(low: f64, high: f64) -> Self {
        assert!(
            low < high && low.is_finite() && high.is_finite(),
            "Uniform::new requires finite low < high"
        );
        Self {
            low,
            span: high - low,
        }
    }
}

impl Distribution for Uniform {
    type Value = f64;
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        self.low + self.span * rng.next_f64()
    }
}

/// Standard normal `N(0, 1)` via Box–Muller (one sample per call; no shared cache).
#[derive(Clone, Copy, Debug, Default)]
pub struct StandardNormal;

impl Distribution for StandardNormal {
    type Value = f64;
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        // Box–Muller: R² = −2 ln U, Θ = 2π V. Discard the sine sample so the
        // stream stays attached to a single `&mut R` (no thread-local cache).
        let u = rng.sample(Open01);
        let v = rng.sample(Open01);
        let r = (-2.0 * u.ln()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * v;
        r * theta.cos()
    }
}

/// `N(μ, σ²)` with `σ > 0`.
#[derive(Clone, Copy, Debug)]
pub struct Normal {
    mu: f64,
    sigma: f64,
}

impl Normal {
    /// Mean `mu` and standard deviation `sigma`.
    pub fn new(mu: f64, sigma: f64) -> Result<Self, DistError> {
        if !(sigma >= 0.0 && mu.is_finite() && sigma.is_finite()) {
            return Err(DistError::new("Normal requires finite μ and σ ≥ 0"));
        }
        Ok(Self { mu, sigma })
    }
}

impl Distribution for Normal {
    type Value = f64;
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        self.mu + self.sigma * rng.sample(StandardNormal)
    }
}

/// Standard exponential (rate 1): `−ln U`, `U ∼ (0, 1]`.
#[derive(Clone, Copy, Debug, Default)]
pub struct Exp1;

impl Distribution for Exp1 {
    type Value = f64;
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        -rng.sample(OpenClosed01).ln()
    }
}

/// Exponential with rate `λ > 0` (mean `1/λ`).
#[derive(Clone, Copy, Debug)]
pub struct Exp {
    rate: f64,
}

/// Alias used in docs / re-exports.
pub type Exponential = Exp;

impl Exp {
    /// Rate parameterization (`E[X] = 1/λ`).
    pub fn new(rate: f64) -> Result<Self, DistError> {
        if !(rate > 0.0 && rate.is_finite()) {
            return Err(DistError::new("Exp rate must be positive and finite"));
        }
        Ok(Self { rate })
    }
}

impl Distribution for Exp {
    type Value = f64;
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        rng.sample(Exp1) / self.rate
    }
}

/// Gamma(`shape`, `scale`) — scale parameterization, `E[X] = shape · scale`.
#[derive(Clone, Copy, Debug)]
pub struct Gamma {
    shape: f64,
    scale: f64,
}

impl Gamma {
    /// Shape-scale constructor.
    pub fn new(shape: f64, scale: f64) -> Result<Self, DistError> {
        if !(shape > 0.0 && scale > 0.0 && shape.is_finite() && scale.is_finite()) {
            return Err(DistError::new("Gamma shape and scale must be positive"));
        }
        Ok(Self { shape, scale })
    }
}

impl Distribution for Gamma {
    type Value = f64;
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        sample_gamma(self.shape, self.scale, rng).expect("Gamma::new already validated")
    }
}

/// Marsaglia–Tsang gamma sampler (`shape` may be `< 1`).
pub fn sample_gamma<R: Rng + ?Sized>(
    shape: f64,
    scale: f64,
    rng: &mut R,
) -> Result<f64, DistError> {
    if !(shape > 0.0 && scale > 0.0 && shape.is_finite() && scale.is_finite()) {
        return Err(DistError::new("gamma shape and scale must be positive"));
    }
    if shape < 1.0 {
        // Log-space: G_{α+1} U^{1/α} underflows to 0·∞ for tiny α.
        let u = rng.sample(Open01);
        let g = sample_gamma(shape + 1.0, 1.0, rng)?;
        let log_x = g.ln() + u.ln() / shape + scale.ln();
        let x = log_x.exp();
        if !x.is_finite() {
            return Err(DistError::new("gamma shape<1 overflowed"));
        }
        return Ok(x);
    }
    let d = shape - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    loop {
        let x: f64 = rng.sample(StandardNormal);
        let mut v = 1.0 + c * x;
        if v <= 0.0 {
            continue;
        }
        v = v * v * v;
        let u: f64 = rng.sample(Uniform::new(0.0, 1.0));
        if u < 1.0 - 0.0331 * x.powi(4) {
            return Ok(d * v * scale);
        }
        if u.ln() < 0.5 * x * x + d * (1.0 - v + v.ln()) {
            return Ok(d * v * scale);
        }
    }
}

/// Inverse Gaussian / Wald (`μ`, `λ`).
#[derive(Clone, Copy, Debug)]
pub struct InverseGaussian {
    mu: f64,
    lambda: f64,
}

impl InverseGaussian {
    /// Mean `mu` and shape `lambda`.
    pub fn new(mu: f64, lambda: f64) -> Result<Self, DistError> {
        if !(mu > 0.0 && lambda > 0.0 && mu.is_finite() && lambda.is_finite()) {
            return Err(DistError::new("IG mean and shape must be positive"));
        }
        Ok(Self { mu, lambda })
    }
}

impl Distribution for InverseGaussian {
    type Value = f64;
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        sample_inverse_gaussian(self.mu, self.lambda, rng).expect("validated")
    }
}

/// Michael–Schucany–Haas inverse-Gaussian sampler.
pub fn sample_inverse_gaussian<R: Rng + ?Sized>(
    mu: f64,
    lambda: f64,
    rng: &mut R,
) -> Result<f64, DistError> {
    if !(mu > 0.0 && lambda > 0.0 && mu.is_finite() && lambda.is_finite()) {
        return Err(DistError::new("IG mean and shape must be positive"));
    }
    let n: f64 = rng.sample(StandardNormal);
    let y = n * n;
    let x = mu + (mu * mu * y) / (2.0 * lambda)
        - (mu / (2.0 * lambda)) * (4.0 * mu * lambda * y + mu * mu * y * y).sqrt();
    let u: f64 = rng.sample(Uniform::new(0.0, 1.0));
    if u <= mu / (mu + x) {
        Ok(x)
    } else {
        Ok(mu * mu / x)
    }
}

/// Student-t with `df` degrees of freedom.
#[derive(Clone, Copy, Debug)]
pub struct StudentT {
    df: f64,
}

impl StudentT {
    /// `df > 0`.
    pub fn new(df: f64) -> Result<Self, DistError> {
        if !(df > 0.0 && df.is_finite()) {
            return Err(DistError::new("Student-t df must be positive"));
        }
        Ok(Self { df })
    }
}

impl Distribution for StudentT {
    type Value = f64;
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        let z: f64 = rng.sample(StandardNormal);
        // χ²(ν) = Gamma(ν/2, 2)
        let v = sample_gamma(self.df * 0.5, 2.0, rng).expect("df validated");
        z / (v / self.df).sqrt()
    }
}

/// Poisson(`λ`) returning `f64` (matching the previous `rand_distr` API).
#[derive(Clone, Copy, Debug)]
pub struct Poisson {
    lambda: f64,
}

impl Poisson {
    /// Intensity `λ ≥ 0`.
    pub fn new(lambda: f64) -> Result<Self, DistError> {
        if !(lambda >= 0.0 && lambda.is_finite()) {
            return Err(DistError::new("Poisson λ must be finite and ≥ 0"));
        }
        Ok(Self { lambda })
    }
}

impl Distribution for Poisson {
    type Value = f64;
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        sample_poisson(self.lambda, rng) as f64
    }
}

fn sample_poisson<R: Rng + ?Sized>(lambda: f64, rng: &mut R) -> u64 {
    if lambda == 0.0 {
        return 0;
    }
    if lambda < 30.0 {
        // Knuth inversion.
        let limit = (-lambda).exp();
        let mut k = 0u64;
        let mut p = 1.0;
        loop {
            k += 1;
            p *= rng.next_f64();
            if p <= limit {
                return k - 1;
            }
        }
    }
    // Hormann transformed-rejection (PTRS) for large λ.
    let c = 0.767 - 3.36 / lambda;
    let beta = std::f64::consts::PI / (3.0 * lambda).sqrt();
    let alpha = beta * lambda;
    let k = c.ln() - lambda - beta.ln();
    loop {
        let u = rng.sample(Open01);
        let x = (alpha - ((1.0 - u) / u).ln()) / beta;
        let n = (x + 0.5).floor();
        if n < 0.0 {
            continue;
        }
        let v = rng.sample(Open01);
        let y = alpha - beta * x;
        let lhs = y + (v / (1.0 + y.exp()).powi(2)).ln();
        let rhs = k + n * lambda.ln() - log_factorial(n);
        if lhs <= rhs {
            return n as u64;
        }
    }
}

fn log_factorial(n: f64) -> f64 {
    if n <= 1.0 {
        return 0.0;
    }
    // Stirling series, accurate enough for the Poisson rejection test.
    (n + 0.5) * n.ln() - n + 0.5 * (2.0 * std::f64::consts::PI).ln() + 1.0 / (12.0 * n)
        - 1.0 / (360.0 * n * n * n)
}

/// Beta(`α`, `β`) on `(0, 1)`, constructed as a ratio of independent gammas.
///
/// If $X \sim \mathrm{Gamma}(\alpha, 1)$ and $Y \sim \mathrm{Gamma}(\beta, 1)$
/// (shape-scale), then $X/(X+Y) \sim \mathrm{Beta}(\alpha, \beta)$
/// (Devroye 1986, Ch. IX).
#[derive(Clone, Copy, Debug)]
pub struct Beta {
    alpha: f64,
    beta: f64,
}

impl Beta {
    /// Shape parameters `α > 0`, `β > 0`.
    pub fn new(alpha: f64, beta: f64) -> Result<Self, DistError> {
        if !(alpha > 0.0 && beta > 0.0 && alpha.is_finite() && beta.is_finite()) {
            return Err(DistError::new("Beta α, β must be positive and finite"));
        }
        Ok(Self { alpha, beta })
    }

    /// First shape `α`.
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// Second shape `β`.
    pub fn beta(&self) -> f64 {
        self.beta
    }
}

impl Distribution for Beta {
    type Value = f64;
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        sample_beta(self.alpha, self.beta, rng).expect("Beta::new already validated")
    }
}

/// `ln Gamma(shape, scale)` so ratios stay finite when the gamma itself
/// underflows to 0 (Beta / Dirichlet with tiny shapes).
fn sample_log_gamma<R: Rng + ?Sized>(
    shape: f64,
    scale: f64,
    rng: &mut R,
) -> Result<f64, DistError> {
    if shape < 1.0 {
        let u = rng.sample(Open01);
        let g = sample_gamma(shape + 1.0, 1.0, rng)?;
        let lg = if g > 0.0 && g.is_finite() {
            g.ln()
        } else {
            f64::NEG_INFINITY
        };
        return Ok(lg + u.ln() / shape + scale.ln());
    }
    let x = sample_gamma(shape, scale, rng)?;
    if x > 0.0 && x.is_finite() {
        Ok(x.ln())
    } else if x == 0.0 {
        Ok(f64::NEG_INFINITY)
    } else {
        Err(DistError::new("gamma log overflow"))
    }
}

/// Sample $\mathrm{Beta}(\alpha, \beta)$ via two independent $\mathrm{Gamma}$ draws.
pub fn sample_beta<R: Rng + ?Sized>(alpha: f64, beta: f64, rng: &mut R) -> Result<f64, DistError> {
    if !(alpha > 0.0 && beta > 0.0 && alpha.is_finite() && beta.is_finite()) {
        return Err(DistError::new("Beta α, β must be positive and finite"));
    }
    let lx = sample_log_gamma(alpha, 1.0, rng)?;
    let ly = sample_log_gamma(beta, 1.0, rng)?;
    if lx == f64::NEG_INFINITY && ly == f64::NEG_INFINITY {
        return Ok(0.5);
    }
    if lx == f64::NEG_INFINITY {
        return Ok(0.0);
    }
    if ly == f64::NEG_INFINITY {
        return Ok(1.0);
    }
    let m = lx.max(ly);
    let s = m + ((lx - m).exp() + (ly - m).exp()).ln();
    let p = (lx - s).exp();
    if p.is_finite() {
        Ok(p.clamp(0.0, 1.0))
    } else {
        Ok(if lx >= ly { 1.0 } else { 0.0 })
    }
}

/// Sample a Dirichlet vector: $X_i \sim \mathrm{Gamma}(\alpha_i, 1)$, then normalize.
///
/// Ferguson (1973) uses this finite-dimensional Dirichlet as the finite
/// counterpart of the Dirichlet process.
pub fn sample_dirichlet<R: Rng + ?Sized>(
    alpha: &[f64],
    rng: &mut R,
) -> Result<Vec<f64>, DistError> {
    if alpha.is_empty() {
        return Err(DistError::new("Dirichlet needs at least one coordinate"));
    }
    let mut logs = Vec::with_capacity(alpha.len());
    let mut m = f64::NEG_INFINITY;
    for &a in alpha {
        let lg = sample_log_gamma(a, 1.0, rng)?;
        if lg > m {
            m = lg;
        }
        logs.push(lg);
    }
    if !m.is_finite() {
        let k = alpha.len() as f64;
        return Ok(vec![1.0 / k; alpha.len()]);
    }
    let mut x = Vec::with_capacity(logs.len());
    let mut s = 0.0;
    for lg in &logs {
        let g = (lg - m).exp();
        s += g;
        x.push(g);
    }
    if s <= 0.0 || !s.is_finite() {
        let k = alpha.len() as f64;
        return Ok(vec![1.0 / k; alpha.len()]);
    }
    for xi in &mut x {
        *xi /= s;
    }
    Ok(x)
}

/// Chambers–Mallows–Stuck sampler for a standard α-stable random variable
/// in Nolan's **S0** parameterization (`S(α, β, 1, 0; 0)`).
///
/// For `α ≠ 1` (Weron 1996 / Nolan):
///
/// ```text
/// ζ = −β tan(π α / 2),   ξ = arctan(−ζ) / α,
/// X = (1+ζ²)^{1/(2α)} · sin(α(U+ξ)) / cos(U)^{1/α}
///     · [cos(U − α(U+ξ)) / W]^{(1−α)/α}
/// ```
///
/// with `U ∼ Unif(−π/2, π/2)` and `W ∼ Exp(1)`. Special cases:
/// `α=2` is `N(0, 2)` (any `β`); `α=1, β=0` is standard Cauchy.
pub fn sample_stable_cms<R: Rng + ?Sized>(
    alpha: f64,
    beta: f64,
    rng: &mut R,
) -> Result<f64, DistError> {
    if !(0.0 < alpha && alpha <= 2.0 && beta.abs() <= 1.0 && alpha.is_finite() && beta.is_finite())
    {
        return Err(DistError::new("invalid stable parameters"));
    }
    let half_pi = std::f64::consts::FRAC_PI_2;
    let u = rng.sample(Uniform::new(-half_pi, half_pi));
    let w: f64 = rng.sample(Exp1);
    if (alpha - 1.0).abs() < 1e-12 {
        let two_over_pi = 2.0 / std::f64::consts::PI;
        return Ok(two_over_pi
            * ((half_pi + beta * u) * u.tan()
                - beta * ((half_pi * w * u.cos()) / (half_pi + beta * u)).ln()));
    }
    let zeta = -beta * (half_pi * alpha).tan();
    let xi = (-zeta).atan() / alpha;
    let scale = (1.0 + zeta * zeta).powf(1.0 / (2.0 * alpha));
    let t = scale * (alpha * (u + xi)).sin() / u.cos().powf(1.0 / alpha)
        * ((u - alpha * (u + xi)).cos() / w).powf((1.0 - alpha) / alpha);
    Ok(t)
}

/// χ² with `df` degrees of freedom (`Gamma(df/2, 2)`).
pub fn sample_chi2<R: Rng + ?Sized>(df: f64, rng: &mut R) -> Result<f64, DistError> {
    if !(df > 0.0 && df.is_finite()) {
        return Err(DistError::new("chi-square df must be positive"));
    }
    sample_gamma(df * 0.5, 2.0, rng)
}

/// Non-central χ²(`df`, `λ`) via the Poisson mixture / Gaussian split.
///
/// `df > 1`: `(Z+√λ)² + χ²_{df−1}`. `df ≤ 1`: `χ²_{df+2N}` with `N∼Poisson(λ/2)`.
pub fn sample_noncentral_chi2<R: Rng + ?Sized>(
    df: f64,
    ncp: f64,
    rng: &mut R,
) -> Result<f64, DistError> {
    if !(df > 0.0 && df.is_finite()) {
        return Err(DistError::new("non-central chi-square df must be positive"));
    }
    if !(ncp >= 0.0 && ncp.is_finite()) {
        return Err(DistError::new("non-central chi-square λ must be ≥ 0"));
    }
    if ncp == 0.0 {
        return sample_chi2(df, rng);
    }
    if df > 1.0 {
        let z = rng.sample(StandardNormal) + ncp.sqrt();
        Ok(z * z + sample_chi2(df - 1.0, rng)?)
    } else {
        let n = rng.sample(Poisson::new(ncp * 0.5).map_err(|_| DistError::new("poisson λ"))?);
        sample_chi2(df + 2.0 * n, rng)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chacha::seed_rng;

    fn mean(xs: &[f64]) -> f64 {
        xs.iter().sum::<f64>() / xs.len() as f64
    }

    #[test]
    fn normal_mean_near_zero() {
        let mut rng = seed_rng(3);
        let xs: Vec<f64> = (0..20_000).map(|_| rng.sample(StandardNormal)).collect();
        assert!(mean(&xs).abs() < 0.04);
    }

    #[test]
    fn exp1_mean_near_one() {
        let mut rng = seed_rng(4);
        let xs: Vec<f64> = (0..20_000).map(|_| rng.sample(Exp1)).collect();
        assert!((mean(&xs) - 1.0).abs() < 0.05);
    }

    #[test]
    fn gamma_mean() {
        let mut rng = seed_rng(5);
        let g = Gamma::new(2.0, 3.0).unwrap();
        let xs: Vec<f64> = (0..15_000).map(|_| g.sample(&mut rng)).collect();
        assert!((mean(&xs) - 6.0).abs() < 0.25);
    }

    #[test]
    fn poisson_mean() {
        let mut rng = seed_rng(6);
        let p = Poisson::new(4.0).unwrap();
        let xs: Vec<f64> = (0..12_000).map(|_| p.sample(&mut rng)).collect();
        assert!((mean(&xs) - 4.0).abs() < 0.15);
    }

    #[test]
    fn student_t_finite() {
        let mut rng = seed_rng(7);
        let t = StudentT::new(5.0).unwrap();
        for _ in 0..100 {
            assert!(t.sample(&mut rng).is_finite());
        }
    }

    #[test]
    fn beta_mean() {
        let mut rng = seed_rng(8);
        let b = Beta::new(2.0, 5.0).unwrap();
        let xs: Vec<f64> = (0..20_000).map(|_| b.sample(&mut rng)).collect();
        // E[Beta(2,5)] = 2/7 ≈ 0.2857
        assert!((mean(&xs) - 2.0 / 7.0).abs() < 0.02);
        assert!(xs.iter().all(|&u| (0.0..=1.0).contains(&u)));
    }

    #[test]
    fn dirichlet_sums_to_one() {
        let mut rng = seed_rng(9);
        let p = sample_dirichlet(&[1.0, 2.0, 3.0], &mut rng).unwrap();
        let s: f64 = p.iter().sum();
        assert!((s - 1.0).abs() < 1e-12);
        assert!(p.iter().all(|&u| u > 0.0));
    }

    #[test]
    fn beta_tiny_shapes_do_not_panic() {
        let mut rng = seed_rng(10);
        let b = Beta::new(0.005, 0.005).unwrap();
        for _ in 0..200 {
            let u = b.sample(&mut rng);
            assert!(u.is_finite() && (0.0..=1.0).contains(&u));
        }
    }

    #[test]
    fn stable_alpha2_is_n02() {
        let mut rng = seed_rng(11);
        let xs: Vec<f64> = (0..40_000)
            .map(|_| sample_stable_cms(2.0, 0.0, &mut rng).unwrap())
            .collect();
        let m = mean(&xs);
        let v = xs.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / xs.len() as f64;
        assert!(m.abs() < 0.08, "mean {m}");
        assert!((v - 2.0).abs() < 0.15, "var {v}");
        let tail = xs.iter().filter(|x| x.abs() > 5.0).count() as f64 / xs.len() as f64;
        // N(0,2): P(|X|>5) = 2(1−Φ(5/√2)) ≈ 4.3e-4
        assert!(tail < 0.005, "P(|X|>5)={tail}");
    }

    #[test]
    fn stable_cauchy_median() {
        let mut rng = seed_rng(12);
        let mut xs: Vec<f64> = (0..8_000)
            .map(|_| sample_stable_cms(1.0, 0.0, &mut rng).unwrap())
            .collect();
        xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let med = xs[xs.len() / 2];
        assert!(med.abs() < 0.15, "Cauchy median {med}");
    }

    #[test]
    fn chi2_and_ncx2_means() {
        let mut rng = seed_rng(13);
        let xs: Vec<f64> = (0..8_000)
            .map(|_| sample_chi2(4.0, &mut rng).unwrap())
            .collect();
        assert!((mean(&xs) - 4.0).abs() < 0.2);
        let ys: Vec<f64> = (0..8_000)
            .map(|_| sample_noncentral_chi2(4.0, 2.0, &mut rng).unwrap())
            .collect();
        // E[χ²_{4}(2)] = 4 + 2 = 6
        assert!((mean(&ys) - 6.0).abs() < 0.25);
    }
}
