//! Dirichlet process, Chinese restaurant process, stick-breaking, Pitman–Yor.
//!
//! Primary sources:
//! - Ferguson (1973) — Dirichlet process
//! - Blackwell & MacQueen (1973) — Pólya urn / Blackwell–MacQueen
//! - Sethuraman (1994) — stick-breaking
//! - Aldous (1985) — Chinese restaurant process
//! - Perman, Pitman & Yor (1992); Pitman & Yor (1997) — two-parameter PY
//! - Ishwaran & James (2001) — truncated stick-breaking used in the sampler

use amatsuki::{sample_beta, Rng};

use crate::error::{Error, Result};

/// Which stick-breaking law produces the weights.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum StickBreakingKind {
    /// Sethuraman (1994): $V_k \sim \mathrm{Beta}(1, \alpha)$.
    Dirichlet { alpha: f64 },
    /// Pitman–Yor: $V_k \sim \mathrm{Beta}(1-d, \theta + k d)$.
    PitmanYor { discount: f64, strength: f64 },
}

/// Truncated stick-breaking weights $\pi_{1:K}$ and leftover mass.
#[derive(Clone, Debug)]
pub struct StickBreaking {
    pub kind: StickBreakingKind,
    /// $\pi_k = V_k \prod_{j<k}(1-V_j)$ for $k < K$, last atom takes remainder.
    pub weights: Vec<f64>,
    /// $\prod_{j=1}^{K}(1-V_j)$ before the last-atom residual is folded in.
    /// After [`sample_stick_breaking`] this is 0 because the remainder is
    /// assigned to $\pi_K$ (Ishwaran–James residual-atom truncation).
    pub leftover: f64,
}

/// Two-parameter Pitman–Yor $(d, \theta)$ with $0 \le d < 1$, $\theta > -d$.
#[derive(Clone, Copy, Debug)]
pub struct PitmanYorParams {
    pub discount: f64,
    pub strength: f64,
}

impl PitmanYorParams {
    pub fn new(discount: f64, strength: f64) -> Result<Self> {
        if !(discount >= 0.0 && discount < 1.0 && discount.is_finite()) {
            return Err(Error::param("Pitman–Yor discount d must satisfy 0 ≤ d < 1"));
        }
        if !(strength > -discount && strength.is_finite()) {
            return Err(Error::param("Pitman–Yor strength θ must satisfy θ > −d"));
        }
        Ok(Self { discount, strength })
    }
}

/// Occupancy counts of a finite clustering.
#[derive(Clone, Debug)]
pub struct ClusterSizes {
    pub assignments: Vec<usize>,
    pub sizes: Vec<usize>,
}

/// Sample a $K$-atom truncation of Sethuraman / Pitman–Yor stick-breaking.
///
/// **Deviation from Sethuraman (1994):** the original construction is an
/// infinite sum $G = \sum_{k=1}^\infty \pi_k \delta_{\theta_k}$. We cut at
/// finite $K$ and put the leftover mass $\prod_{j=1}^{K-1}(1-V_j)\,(1-V_K)$
/// onto the last atom so that $\sum_k \pi_k = 1$ (Ishwaran & James 2001,
/// residual-atom / "blocked" truncation — not their GEM exact finite DP).
pub fn sample_stick_breaking<R: Rng + ?Sized>(
    kind: StickBreakingKind,
    k: usize,
    rng: &mut R,
) -> Result<StickBreaking> {
    if k == 0 {
        return Err(Error::param("stick-breaking needs K ≥ 1"));
    }
    match kind {
        StickBreakingKind::Dirichlet { alpha } => {
            if !(alpha > 0.0 && alpha.is_finite()) {
                return Err(Error::param("DP concentration α must be positive"));
            }
        }
        StickBreakingKind::PitmanYor { discount, strength } => {
            PitmanYorParams::new(discount, strength)?;
        }
    }
    let mut weights = vec![0.0; k];
    let mut rest = 1.0;
    for i in 0..k {
        let (a, b) = match kind {
            StickBreakingKind::Dirichlet { alpha } => (1.0, alpha),
            StickBreakingKind::PitmanYor { discount, strength } => {
                // V_k ~ Beta(1−d, θ + k d) with k = 1, 2, … so the first
                // stick is Beta(1−d, θ+d). Using θ+(k−1)d made E[π₁] = (1−d)/(1−d+θ).
                (1.0 - discount, strength + (i as f64 + 1.0) * discount)
            }
        };
        let v = if i + 1 == k {
            1.0
        } else {
            sample_beta(a, b, rng).map_err(|e| Error::numeric(e.message))?
        };
        weights[i] = rest * v;
        rest *= 1.0 - v;
    }
    Ok(StickBreaking {
        kind,
        weights,
        leftover: 0.0,
    })
}

/// Sequential Chinese restaurant process (Aldous 1985; Blackwell–MacQueen 1973).
///
/// Customer $i$ (1-based) sits at occupied table $k$ with probability
/// $n_k / (i-1+\alpha)$, or opens a new table with $\alpha / (i-1+\alpha)$.
///
/// This is the **exact** exchangeable partition probability function of the
/// DP (not a truncation). Labels are in order of appearance (0, 1, …).
pub fn sample_crp_assignments<R: Rng + ?Sized>(
    n: usize,
    alpha: f64,
    rng: &mut R,
) -> Result<ClusterSizes> {
    if !(alpha > 0.0 && alpha.is_finite()) {
        return Err(Error::param("CRP concentration α must be positive"));
    }
    if n == 0 {
        return Ok(ClusterSizes {
            assignments: Vec::new(),
            sizes: Vec::new(),
        });
    }
    let mut assignments = Vec::with_capacity(n);
    let mut sizes: Vec<usize> = Vec::new();
    assignments.push(0);
    sizes.push(1);
    for i in 1..n {
        let denom = i as f64 + alpha;
        let u = rng.next_f64() * denom;
        let mut acc = 0.0;
        let mut chosen = None;
        for (k, &nk) in sizes.iter().enumerate() {
            acc += nk as f64;
            if u < acc {
                chosen = Some(k);
                break;
            }
        }
        match chosen {
            Some(k) => {
                sizes[k] += 1;
                assignments.push(k);
            }
            None => {
                assignments.push(sizes.len());
                sizes.push(1);
            }
        }
    }
    Ok(ClusterSizes { assignments, sizes })
}

/// Two-parameter Chinese restaurant (Pitman–Yor).
///
/// $P(\text{join }k) = (n_k - d)/(n-1+\theta)$,
/// $P(\text{new}) = (\theta + K d)/(n-1+\theta)$.
pub fn sample_pitman_yor_crp<R: Rng + ?Sized>(
    n: usize,
    params: PitmanYorParams,
    rng: &mut R,
) -> Result<ClusterSizes> {
    let d = params.discount;
    let theta = params.strength;
    if n == 0 {
        return Ok(ClusterSizes {
            assignments: Vec::new(),
            sizes: Vec::new(),
        });
    }
    let mut assignments = Vec::with_capacity(n);
    let mut sizes: Vec<usize> = Vec::new();
    assignments.push(0);
    sizes.push(1);
    for i in 1..n {
        let denom = i as f64 + theta;
        let k_now = sizes.len() as f64;
        let u = rng.next_f64() * denom;
        let mut acc = 0.0;
        let mut chosen = None;
        for (k, &nk) in sizes.iter().enumerate() {
            acc += nk as f64 - d;
            if u < acc {
                chosen = Some(k);
                break;
            }
        }
        if chosen.is_none() {
            // leftover should be θ + K d
            let _ = k_now;
        }
        match chosen {
            Some(k) => {
                sizes[k] += 1;
                assignments.push(k);
            }
            None => {
                assignments.push(sizes.len());
                sizes.push(1);
            }
        }
    }
    Ok(ClusterSizes { assignments, sizes })
}

/// Expected number of occupied tables in a CRP of size $n$: $\alpha H_n$
/// in the $\alpha \to$ harmonic approximation $E[K_n] = \alpha \sum_{i=1}^n 1/(\alpha+i-1)$.
pub fn expected_crp_tables(n: usize, alpha: f64) -> Result<f64> {
    if !(alpha > 0.0 && alpha.is_finite()) {
        return Err(Error::param("α must be positive"));
    }
    let mut s = 0.0;
    for i in 0..n {
        s += 1.0 / (alpha + i as f64);
    }
    Ok(alpha * s)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::seed_rng;

    #[test]
    fn stick_breaking_sums_to_one() {
        let mut rng = seed_rng(1);
        let sb = sample_stick_breaking(StickBreakingKind::Dirichlet { alpha: 1.5 }, 20, &mut rng)
            .unwrap();
        let s: f64 = sb.weights.iter().sum();
        assert!((s - 1.0).abs() < 1e-12);
        assert!(sb.weights.iter().all(|&w| w >= 0.0));
    }

    #[test]
    fn crp_first_customer_table_zero() {
        let mut rng = seed_rng(2);
        let c = sample_crp_assignments(1, 2.0, &mut rng).unwrap();
        assert_eq!(c.assignments, vec![0]);
        assert_eq!(c.sizes, vec![1]);
    }

    #[test]
    fn crp_table_count_near_expectation() {
        let mut rng = seed_rng(3);
        let n = 80;
        let alpha = 2.0;
        let mut ks = 0.0;
        let reps = 200;
        for _ in 0..reps {
            let c = sample_crp_assignments(n, alpha, &mut rng).unwrap();
            ks += c.sizes.len() as f64;
        }
        let mean_k = ks / reps as f64;
        let ek = expected_crp_tables(n, alpha).unwrap();
        assert!((mean_k - ek).abs() < 1.0, "mean K {mean_k} vs E[K] {ek}");
    }

    #[test]
    fn pitman_yor_more_tables_when_d_positive() {
        let mut rng = seed_rng(4);
        let n = 60;
        let mut k_dp = 0.0;
        let mut k_py = 0.0;
        let reps = 80;
        for _ in 0..reps {
            k_dp += sample_crp_assignments(n, 1.0, &mut rng)
                .unwrap()
                .sizes
                .len() as f64;
            k_py += sample_pitman_yor_crp(n, PitmanYorParams::new(0.4, 1.0).unwrap(), &mut rng)
                .unwrap()
                .sizes
                .len() as f64;
        }
        assert!(
            k_py > k_dp,
            "PY should open more tables: PY {} DP {}",
            k_py / reps as f64,
            k_dp / reps as f64
        );
    }
}
