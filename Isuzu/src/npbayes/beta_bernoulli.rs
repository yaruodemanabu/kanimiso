//! Beta process, Bernoulli process, and the Indian buffet process.
//!
//! Primary sources:
//! - Hjort (1990) — Beta process as a completely random measure
//! - Thibaux & Jordan (2007) — Beta–Bernoulli process $\Leftrightarrow$ IBP
//! - Griffiths & Ghahramani (2005, 2011) — sequential IBP and finite model
//! - Teh, Görür & Ghahramani (2007) — stick-breaking construction for the IBP
//!
//! We do **not** construct Hjort's CRM from its Lévy measure
//! $\nu(d\pi, d\omega) = c\,\pi^{-1}(1-\pi)^{c-1}\,d\pi\,B_0(d\omega)$
//! via an inverse-Lévy / Poisson point process. The implemented generators
//! are the exchangeable IBP, the TGG stick-breaking weights, and two finite
//! approximations. Those deviations are listed next to each function.

use amatsuki::{sample_beta, Poisson, Rng};

use crate::error::{Error, Result};

/// Binary feature allocation $Z \in \{0,1\}^{N \times K}$, row-major.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FeatureMatrix {
    pub n: usize,
    pub k: usize,
    data: Vec<u8>,
}

impl FeatureMatrix {
    pub fn zeros(n: usize, k: usize) -> Self {
        Self {
            n,
            k,
            data: vec![0; n * k],
        }
    }

    pub fn from_rows(rows: &[Vec<u8>]) -> Result<Self> {
        if rows.is_empty() {
            return Ok(Self::zeros(0, 0));
        }
        let n = rows.len();
        let k = rows[0].len();
        let mut data = Vec::with_capacity(n * k);
        for r in rows {
            if r.len() != k {
                return Err(Error::dim("ragged feature matrix"));
            }
            for &z in r {
                if z > 1 {
                    return Err(Error::param("feature entries must be 0 or 1"));
                }
                data.push(z);
            }
        }
        Ok(Self { n, k, data })
    }

    pub fn get(&self, i: usize, k: usize) -> u8 {
        self.data[i * self.k + k]
    }

    pub fn set(&mut self, i: usize, k: usize, v: u8) {
        self.data[i * self.k + k] = v;
    }

    pub fn column_sum(&self, k: usize) -> usize {
        (0..self.n).map(|i| self.get(i, k) as usize).sum()
    }

    pub fn column_sum_except(&self, k: usize, skip: usize) -> usize {
        (0..self.n)
            .filter(|&i| i != skip)
            .map(|i| self.get(i, k) as usize)
            .sum()
    }

    pub fn row(&self, i: usize) -> Vec<u8> {
        self.data[i * self.k..(i + 1) * self.k].to_vec()
    }

    /// Drop columns whose sum is zero.
    pub fn drop_empty_columns(&mut self) {
        let keep: Vec<usize> = (0..self.k).filter(|&k| self.column_sum(k) > 0).collect();
        if keep.len() == self.k {
            return;
        }
        let mut next = FeatureMatrix::zeros(self.n, keep.len());
        for i in 0..self.n {
            for (nk, &k) in keep.iter().enumerate() {
                next.set(i, nk, self.get(i, k));
            }
        }
        *self = next;
    }

    pub fn add_columns(&mut self, extra: usize) {
        if extra == 0 {
            return;
        }
        let mut next = FeatureMatrix::zeros(self.n, self.k + extra);
        for i in 0..self.n {
            for k in 0..self.k {
                next.set(i, k, self.get(i, k));
            }
        }
        *self = next;
    }

    pub fn as_f64_rows(&self) -> Vec<Vec<f64>> {
        (0..self.n)
            .map(|i| (0..self.k).map(|k| self.get(i, k) as f64).collect())
            .collect()
    }

    /// Hamming distance after a greedy column matching (not the Hungarian
    /// optimum). Used only as a diagnostic.
    pub fn hamming_after_greedy_match(&self, other: &FeatureMatrix) -> Result<usize> {
        if self.n != other.n {
            return Err(Error::dim("row counts differ"));
        }
        let mut unused: Vec<usize> = (0..other.k).collect();
        let mut dist = 0usize;
        for k in 0..self.k {
            let mut best = usize::MAX;
            let mut best_j = None;
            for (idx, &j) in unused.iter().enumerate() {
                let mut d = 0usize;
                for i in 0..self.n {
                    if self.get(i, k) != other.get(i, j) {
                        d += 1;
                    }
                }
                if d < best {
                    best = d;
                    best_j = Some(idx);
                }
            }
            if let Some(idx) = best_j {
                dist += best;
                unused.swap_remove(idx);
            } else {
                dist += self.column_sum(k);
            }
        }
        for j in unused {
            dist += other.column_sum(j);
        }
        Ok(dist)
    }
}

/// IBP mass parameter $\alpha > 0$ (Griffiths & Ghahramani). Optional
/// concentration $c$ is the Hjort / Thibaux–Jordan $c$ (default $c=1$
/// recovers the ordinary IBP).
#[derive(Clone, Copy, Debug)]
pub struct IbpParams {
    pub alpha: f64,
    pub concentration: f64,
}

impl IbpParams {
    pub fn new(alpha: f64) -> Result<Self> {
        Self::with_concentration(alpha, 1.0)
    }

    pub fn with_concentration(alpha: f64, concentration: f64) -> Result<Self> {
        if !(alpha > 0.0 && alpha.is_finite()) {
            return Err(Error::param("IBP α must be positive"));
        }
        if !(concentration > 0.0 && concentration.is_finite()) {
            return Err(Error::param("IBP / BP concentration c must be positive"));
        }
        Ok(Self {
            alpha,
            concentration: concentration,
        })
    }
}

/// Finite Beta process parameters: $c$ (concentration) and mass $\gamma$
/// of the base measure $B_0(\Omega)=\gamma$.
#[derive(Clone, Copy, Debug)]
pub struct BetaProcessParams {
    pub concentration: f64,
    pub mass: f64,
}

impl BetaProcessParams {
    pub fn new(concentration: f64, mass: f64) -> Result<Self> {
        if !(concentration > 0.0 && mass > 0.0 && concentration.is_finite() && mass.is_finite()) {
            return Err(Error::param("BP needs c > 0 and mass γ > 0"));
        }
        Ok(Self {
            concentration,
            mass,
        })
    }

    /// The IBP($\alpha$) is BeP $\circ$ BP$(c=1, B_0)$ with $B_0(\Omega)=\alpha$.
    pub fn ibp(alpha: f64) -> Result<Self> {
        Self::new(1.0, alpha)
    }
}

/// Sequential Indian buffet process (Griffiths & Ghahramani 2005, 2011).
///
/// Customer $i$ (1-based) takes already-served dish $k$ independently with
/// probability $m_k / i$, then samples $\mathrm{Poisson}(\alpha / i)$ new dishes.
///
/// This is the **exact** exchangeable feature allocation of the
/// Beta–Bernoulli process with $c=1$ (Thibaux & Jordan 2007, Prop. 3).
///
/// **Deviation:** we only implement the $c=1$ sequential restaurant. The
/// three-parameter IBP of Teh & Görür (with $c \ne 1$) is **not** the
/// sequential generator used here; use [`sample_beta_process_finite`] for
/// $c \ne 1$.
pub fn sample_ibp_sequential<R: Rng + ?Sized>(
    n: usize,
    params: IbpParams,
    rng: &mut R,
) -> Result<FeatureMatrix> {
    if params.concentration != 1.0 {
        return Err(Error::unsupported(
            "sequential IBP is implemented only for c = 1; use the finite Beta–Bernoulli for c ≠ 1",
        ));
    }
    if n == 0 {
        return Ok(FeatureMatrix::zeros(0, 0));
    }
    let alpha = params.alpha;
    let mut rows: Vec<Vec<u8>> = Vec::with_capacity(n);
    let k0 = sample_poisson_u64(alpha, rng)? as usize;
    rows.push(vec![1; k0]);
    for i in 1..n {
        let customer = (i + 1) as f64; // 1-based
        let k = rows[0].len();
        let mut row = vec![0u8; k];
        for kk in 0..k {
            let m = rows.iter().map(|r| r[kk] as usize).sum::<usize>();
            if rng.next_f64() < (m as f64) / customer {
                row[kk] = 1;
            }
        }
        let new_k = sample_poisson_u64(alpha / customer, rng)? as usize;
        row.extend(std::iter::repeat(1).take(new_k));
        for prev in &mut rows {
            prev.extend(std::iter::repeat(0).take(new_k));
        }
        rows.push(row);
    }
    FeatureMatrix::from_rows(&rows)
}

/// Teh–Görür–Ghahramani (2007) stick-breaking weights for the IBP.
///
/// $v_i \sim \mathrm{Beta}(\alpha, 1)$, $\pi_k = \prod_{i=1}^{k} v_i$,
/// then $z_{nk} \mid \pi_k \sim \mathrm{Bernoulli}(\pi_k)$.
///
/// **Deviation from TGG 2007:** we truncate at finite $K$ and do not
/// sample the leftover infinite tail. The atoms $\omega_k$ of the base
/// measure are not drawn; only the feature probabilities $\pi_k$ and the
/// binary matrix are returned (atom locations are irrelevant when $H$ is
/// used only as a feature index).
pub fn sample_ibp_stick_breaking<R: Rng + ?Sized>(
    n: usize,
    k: usize,
    alpha: f64,
    rng: &mut R,
) -> Result<(Vec<f64>, FeatureMatrix)> {
    if !(alpha > 0.0 && alpha.is_finite()) {
        return Err(Error::param("IBP stick-breaking needs α > 0"));
    }
    if k == 0 {
        return Ok((Vec::new(), FeatureMatrix::zeros(n, 0)));
    }
    let mut pi = vec![0.0; k];
    let mut prod = 1.0;
    for i in 0..k {
        let v = sample_beta(alpha, 1.0, rng).map_err(|e| Error::numeric(e.message))?;
        prod *= v;
        pi[i] = prod;
    }
    let z = sample_bernoulli_process(n, &pi, rng)?;
    Ok((pi, z))
}

/// Finite approximation to $B \sim \mathrm{BP}(c, \gamma H)$ on $K$ atoms
/// (Paisley / Teh finite construction):
///
/// $$\pi_k \stackrel{\mathrm{iid}}{\sim} \mathrm{Beta}\Bigl(\tfrac{c\gamma}{K},\, c\bigl(1-\tfrac{\gamma}{K}\bigr)\Bigr)
/// \quad (K > \gamma).$$
///
/// **Deviation from Hjort (1990):** this is a finite-dimensional Dirichlet-
/// like stand-in, not a draw from the CRM Lévy measure. When
/// $c(1-\gamma/K) \le 0$ we fall back to the Griffiths–Ghahramani finite
/// IBP $\mathrm{Beta}(\gamma/K,\, 1)$ and report that in the second
/// return flag `used_gg_fallback`.
pub fn sample_beta_process_finite<R: Rng + ?Sized>(
    params: BetaProcessParams,
    k: usize,
    rng: &mut R,
) -> Result<(Vec<f64>, bool)> {
    if k == 0 {
        return Ok((Vec::new(), false));
    }
    let c = params.concentration;
    let gamma = params.mass;
    let a = c * gamma / k as f64;
    let b = c * (1.0 - gamma / k as f64);
    if a <= 0.0 {
        return Err(Error::param("finite BP needs K large enough that cγ/K > 0"));
    }
    let (shape_b, fallback) = if b > 0.0 { (b, false) } else { (1.0, true) };
    let mut pi = Vec::with_capacity(k);
    for _ in 0..k {
        pi.push(sample_beta(a, shape_b, rng).map_err(|e| Error::numeric(e.message))?);
    }
    Ok((pi, fallback))
}

/// Bernoulli process $X \mid B \sim \mathrm{BeP}(B)$ on a discrete $B$
/// with atoms $\pi_{1:K}$: $z_{nk} \sim \mathrm{Bernoulli}(\pi_k)$ iid.
///
/// This is exactly Thibaux & Jordan (2007) once $B$ is discrete. Combined
/// with [`sample_beta_process_finite`] it is the finite Beta–Bernoulli
/// process.
pub fn sample_bernoulli_process<R: Rng + ?Sized>(
    n: usize,
    pi: &[f64],
    rng: &mut R,
) -> Result<FeatureMatrix> {
    for &p in pi {
        if !(p >= 0.0 && p <= 1.0) {
            return Err(Error::param("Bernoulli process needs π_k ∈ [0, 1]"));
        }
    }
    let k = pi.len();
    let mut z = FeatureMatrix::zeros(n, k);
    for i in 0..n {
        for (kk, &p) in pi.iter().enumerate() {
            if rng.next_f64() < p {
                z.set(i, kk, 1);
            }
        }
    }
    Ok(z)
}

fn sample_poisson_u64<R: Rng + ?Sized>(lambda: f64, rng: &mut R) -> Result<u64> {
    let p = Poisson::new(lambda.max(0.0)).map_err(|_| Error::param("Poisson λ"))?;
    Ok(rng.sample(p) as u64)
}

/// Expected number of dishes served in an IBP($N$, $\alpha$): $\alpha H_N$.
pub fn expected_ibp_features(n: usize, alpha: f64) -> Result<f64> {
    if !(alpha > 0.0 && alpha.is_finite()) {
        return Err(Error::param("α must be positive"));
    }
    let mut h = 0.0;
    for i in 1..=n {
        h += 1.0 / i as f64;
    }
    Ok(alpha * h)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::seed_rng;

    #[test]
    fn ibp_feature_count_near_alpha_harmonic() {
        let mut rng = seed_rng(11);
        let n = 40;
        let alpha = 2.0;
        let mut ks = 0.0;
        let reps = 80;
        for _ in 0..reps {
            let z = sample_ibp_sequential(n, IbpParams::new(alpha).unwrap(), &mut rng).unwrap();
            ks += z.k as f64;
        }
        let mean_k = ks / reps as f64;
        let ek = expected_ibp_features(n, alpha).unwrap();
        assert!((mean_k - ek).abs() < 1.5, "mean K {mean_k} vs α H_N {ek}");
    }

    #[test]
    fn finite_beta_bernoulli_entries_binary() {
        let mut rng = seed_rng(12);
        let (pi, fallback) =
            sample_beta_process_finite(BetaProcessParams::ibp(1.5).unwrap(), 12, &mut rng).unwrap();
        assert!(!fallback);
        let z = sample_bernoulli_process(15, &pi, &mut rng).unwrap();
        assert_eq!(z.n, 15);
        assert_eq!(z.k, 12);
        for i in 0..z.n {
            for k in 0..z.k {
                assert!(z.get(i, k) <= 1);
            }
        }
    }

    #[test]
    fn tgg_stick_breaking_decreasing() {
        let mut rng = seed_rng(13);
        let (pi, z) = sample_ibp_stick_breaking(10, 8, 2.0, &mut rng).unwrap();
        for w in pi.windows(2) {
            assert!(w[1] <= w[0] + 1e-15);
        }
        assert_eq!(z.k, 8);
    }

    #[test]
    fn sequential_rejects_c_neq_1() {
        let mut rng = seed_rng(14);
        let p = IbpParams::with_concentration(1.0, 2.0).unwrap();
        assert!(sample_ibp_sequential(5, p, &mut rng).is_err());
    }
}
