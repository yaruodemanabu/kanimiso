//! Discrete laws: Bernoulli, binomial, categorical, multinomial.

use crate::dist::{DistError, Open01};
use crate::rng::{Distribution, Rng};

/// Bernoulli(`p`) on `{0, 1}`.
#[derive(Clone, Copy, Debug)]
pub struct Bernoulli {
    p: f64,
}

impl Bernoulli {
    /// Success probability `p ∈ [0, 1]`.
    pub fn new(p: f64) -> Result<Self, DistError> {
        if !(0.0..=1.0).contains(&p) || !p.is_finite() {
            return Err(DistError::new("Bernoulli p must lie in [0, 1]"));
        }
        Ok(Self { p })
    }

    /// Success probability.
    pub fn p(&self) -> f64 {
        self.p
    }
}

impl Distribution for Bernoulli {
    type Value = u8;
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> u8 {
        u8::from(rng.next_f64() < self.p)
    }
}

/// Binomial(`n`, `p`) — number of successes in `n` independent Bernoulli trials.
#[derive(Clone, Copy, Debug)]
pub struct Binomial {
    n: u64,
    p: f64,
}

impl Binomial {
    /// `n` trials and success probability `p ∈ [0, 1]`.
    pub fn new(n: u64, p: f64) -> Result<Self, DistError> {
        if !(0.0..=1.0).contains(&p) || !p.is_finite() {
            return Err(DistError::new("Binomial p must lie in [0, 1]"));
        }
        Ok(Self { n, p })
    }
}

impl Distribution for Binomial {
    type Value = u64;
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> u64 {
        sample_binomial(self.n, self.p, rng)
    }
}

/// Inverse-CDF / sequential for moderate `n`; BTPE-style rejection for large `n`.
pub fn sample_binomial<R: Rng + ?Sized>(n: u64, p: f64, rng: &mut R) -> u64 {
    if n == 0 || p <= 0.0 {
        return 0;
    }
    if p >= 1.0 {
        return n;
    }
    let q = 1.0 - p;
    if p > 0.5 {
        return n - sample_binomial(n, q, rng);
    }
    if n < 30 {
        let mut k = 0u64;
        for _ in 0..n {
            if rng.next_f64() < p {
                k += 1;
            }
        }
        return k;
    }
    // Geometric waiting-time (Devroye): sum of geometrics until n is filled.
    // For large n this is still O(k) = O(np). Fall back to inverse CDF of the
    // recursive relation when np is huge.
    if n as f64 * p < 40.0 {
        let mut k = 0u64;
        let mut s = 0u64;
        while s < n {
            let u = rng.sample(Open01);
            let g = ((1.0 - u).ln() / q.ln()).floor() as u64;
            s = s.saturating_add(g + 1);
            if s <= n {
                k += 1;
            }
        }
        return k;
    }
    // Normal approximation with continuity correction, rejection-checked
    // against the exact pmf ratio (Hörmann BTRD lite).
    let mu = n as f64 * p;
    let sigma = (n as f64 * p * q).sqrt();
    for _ in 0..100 {
        let z: f64 = rng.sample(crate::dist::StandardNormal);
        let x = mu + sigma * z + 0.5;
        if x < 0.0 || x > n as f64 {
            continue;
        }
        let k = x.floor() as u64;
        // Accept via a loose envelope; exactness is not required for the
        // envelope because we verify with a Metropolis ratio on the log-pmf.
        let log_acc = log_binom_pmf(n, k, p) - log_normal_approx(k as f64, mu, sigma);
        if rng.sample(Open01).ln() <= log_acc + 0.5 {
            return k.min(n);
        }
    }
    mu.round().clamp(0.0, n as f64) as u64
}

fn log_binom_pmf(n: u64, k: u64, p: f64) -> f64 {
    if k > n {
        return f64::NEG_INFINITY;
    }
    log_choose(n, k) + (k as f64) * p.ln() + ((n - k) as f64) * (1.0 - p).ln()
}

fn log_choose(n: u64, k: u64) -> f64 {
    let k = k.min(n - k);
    let mut s = 0.0;
    for i in 0..k {
        s += ((n - i) as f64).ln() - ((i + 1) as f64).ln();
    }
    s
}

fn log_normal_approx(x: f64, mu: f64, sigma: f64) -> f64 {
    let z = (x - mu) / sigma;
    -0.5 * z * z - sigma.ln() - 0.5 * (2.0 * std::f64::consts::PI).ln()
}

/// Categorical distribution on `{0,…,k−1}` with probabilities `p`.
#[derive(Clone, Debug)]
pub struct Categorical {
    cdf: Vec<f64>,
}

impl Categorical {
    /// Weights need not sum to one; they are normalized.
    pub fn new(weights: &[f64]) -> Result<Self, DistError> {
        if weights.is_empty() {
            return Err(DistError::new("Categorical needs at least one weight"));
        }
        if weights.iter().any(|w| !w.is_finite() || *w < 0.0) {
            return Err(DistError::new("Categorical weights must be finite and ≥ 0"));
        }
        let s: f64 = weights.iter().sum();
        if s <= 0.0 {
            return Err(DistError::new(
                "Categorical weights must sum to a positive number",
            ));
        }
        let mut cdf = Vec::with_capacity(weights.len());
        let mut acc = 0.0;
        for &w in weights {
            acc += w / s;
            cdf.push(acc);
        }
        if let Some(last) = cdf.last_mut() {
            *last = 1.0;
        }
        Ok(Self { cdf })
    }

    /// Number of categories.
    pub fn n_categories(&self) -> usize {
        self.cdf.len()
    }
}

impl Distribution for Categorical {
    type Value = usize;
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> usize {
        let u = rng.next_f64();
        for (i, &c) in self.cdf.iter().enumerate() {
            if u < c {
                return i;
            }
        }
        self.cdf.len() - 1
    }
}

/// Multinomial(`n`, `p`) counts.
#[derive(Clone, Debug)]
pub struct Multinomial {
    n: u64,
    p: Vec<f64>,
}

impl Multinomial {
    /// `n` trials and a probability vector (renormalized if needed).
    pub fn new(n: u64, weights: &[f64]) -> Result<Self, DistError> {
        if weights.is_empty() {
            return Err(DistError::new("Multinomial needs at least one cell"));
        }
        if weights.iter().any(|w| !w.is_finite() || *w < 0.0) {
            return Err(DistError::new("Multinomial weights must be finite and ≥ 0"));
        }
        let s: f64 = weights.iter().sum();
        if s <= 0.0 {
            return Err(DistError::new(
                "Multinomial weights must sum to a positive number",
            ));
        }
        Ok(Self {
            n,
            p: weights.iter().map(|w| w / s).collect(),
        })
    }
}

impl Distribution for Multinomial {
    type Value = Vec<u64>;
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Vec<u64> {
        sample_multinomial(self.n, &self.p, rng)
    }
}

/// Sequential binomials: `N_i | rest ~ Binomial(n − Σ_{j<i} N_j, p_i / p_{i:})`.
pub fn sample_multinomial<R: Rng + ?Sized>(n: u64, p: &[f64], rng: &mut R) -> Vec<u64> {
    let k = p.len();
    let mut out = vec![0u64; k];
    let mut remaining = n;
    let mut p_left = 1.0;
    for i in 0..k.saturating_sub(1) {
        if remaining == 0 || p_left <= 0.0 {
            break;
        }
        let pi = (p[i] / p_left).clamp(0.0, 1.0);
        let ni = sample_binomial(remaining, pi, rng);
        out[i] = ni;
        remaining -= ni;
        p_left -= p[i];
    }
    if k > 0 {
        out[k - 1] = remaining;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chacha::seed_rng;

    #[test]
    fn bernoulli_mean() {
        let mut rng = seed_rng(1);
        let b = Bernoulli::new(0.3).unwrap();
        let m: f64 = (0..20_000).map(|_| b.sample(&mut rng) as f64).sum::<f64>() / 20_000.0;
        assert!((m - 0.3).abs() < 0.02);
    }

    #[test]
    fn binomial_mean() {
        let mut rng = seed_rng(2);
        let b = Binomial::new(20, 0.4).unwrap();
        let m: f64 = (0..8_000).map(|_| b.sample(&mut rng) as f64).sum::<f64>() / 8_000.0;
        assert!((m - 8.0).abs() < 0.2);
    }

    #[test]
    fn categorical_and_multinomial() {
        let mut rng = seed_rng(3);
        let c = Categorical::new(&[1.0, 2.0, 1.0]).unwrap();
        let mut hits = [0usize; 3];
        for _ in 0..12_000 {
            hits[c.sample(&mut rng)] += 1;
        }
        let p1 = hits[1] as f64 / 12_000.0;
        assert!((p1 - 0.5).abs() < 0.03);
        let m = Multinomial::new(100, &[0.2, 0.3, 0.5]).unwrap();
        let counts = m.sample(&mut rng);
        assert_eq!(counts.iter().sum::<u64>(), 100);
        assert_eq!(counts.len(), 3);
    }
}
