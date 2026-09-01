//! Gibbs samplers for DP mixtures, HDP (CRF), and IBP linear-Gaussian.
//!
//! Primary sources:
//! - Neal (2000) Algorithm 3 — conjugate DP mixture
//! - Escobar & West (1995) — DP Gaussian mixtures
//! - Teh, Jordan, Beal & Blei (2006) — HDP / Chinese restaurant franchise
//! - Griffiths & Ghahramani (2011) — collapsed Gibbs for linear-Gaussian IBP
//!
//! Every departure from those papers is written next to the function.

use amatsuki::{sample_beta, Rng, StandardNormal};

use crate::error::{Error, Result};
use crate::linalg::{col_from_slice, logdet_spd, mat_identity, solve_spd, spd_regularize};
use crate::npbayes::beta_bernoulli::FeatureMatrix;
use faer::{Col, Mat};

const LN_2PI: f64 = 1.8378770664093453;

/// Conjugate univariate DP Gaussian-mixture fit (Neal 2000, Algorithm 3).
#[derive(Clone, Debug)]
pub struct DpGaussianFit {
    pub assignments: Vec<usize>,
    pub means: Vec<f64>,
    pub sizes: Vec<usize>,
    pub n_clusters: usize,
    pub alpha: f64,
    pub log_predictive: f64,
}

/// Hierarchical DP Gaussian-mixture fit via the Chinese restaurant franchise.
#[derive(Clone, Debug)]
pub struct HdpGaussianFit {
    /// `assignments[g][i]` is the **global dish** of observation $i$ in group $g$.
    pub assignments: Vec<Vec<usize>>,
    pub dish_means: Vec<f64>,
    pub n_dishes: usize,
    pub n_tables: usize,
    pub alpha: f64,
    pub gamma: f64,
}

/// Linear-Gaussian IBP posterior sample.
#[derive(Clone, Debug)]
pub struct IbpLinearGaussianFit {
    pub z: FeatureMatrix,
    pub a: Mat<f64>,
    pub alpha: f64,
    pub sigma_x: f64,
    pub sigma_a: f64,
    pub loglik: f64,
}

/// Neal (2000) Algorithm 3 for $x_i \mid z_i=k \sim N(\mu_k, \sigma^2)$,
/// $\mu_k \sim N(\mu_0, \tau_0^2)$, $z \sim \mathrm{CRP}(\alpha)$.
///
/// **Deviations from Neal (2000):**
/// - Observation variance $\sigma^2$ is **fixed** (not given a prior, not
///   Gibbs-updated). Neal's §3 allows a conjugate prior on $\sigma^2$; we
///   do not.
/// - We store one $\mu_k$ per occupied cluster, redrawn from the full
///   conditional after each sweep (standard; Neal samples them implicitly
///   through the marginal).
/// - No label-switching correction.
pub fn dp_gaussian_mixture_gibbs<R: Rng + ?Sized>(
    x: &[f64],
    alpha: f64,
    sigma: f64,
    mu0: f64,
    tau0: f64,
    n_iter: usize,
    rng: &mut R,
) -> Result<DpGaussianFit> {
    if x.is_empty() {
        return Err(Error::param("DP mixture needs at least one observation"));
    }
    if !(alpha > 0.0 && sigma > 0.0 && tau0 > 0.0 && n_iter > 0) {
        return Err(Error::param("DP mixture: α, σ, τ0 > 0 and n_iter ≥ 1"));
    }
    let n = x.len();
    let mut z = vec![0usize; n];
    let mut means = vec![mu0];
    for sweep in 0..n_iter {
        let _ = sweep;
        for i in 0..n {
            // Remove x_i.
            let old = z[i];
            z[i] = usize::MAX;
            let mut sizes = cluster_sizes(&z, means.len());
            if old < means.len() && sizes[old] == 0 {
                // Drop empty cluster (relabel).
                remove_cluster(&mut z, &mut means, old);
                sizes = cluster_sizes(&z, means.len());
            }
            let k = means.len();
            let mut logs = Vec::with_capacity(k + 1);
            for c in 0..k {
                let (m, v) = pred_mean_var(&x, &z, c, sigma, mu0, tau0);
                logs.push((sizes[c] as f64).ln() + log_normal_density(x[i], m, v.sqrt()));
            }
            let (mnew, vnew) = (mu0, sigma * sigma + tau0 * tau0);
            logs.push(alpha.ln() + log_normal_density(x[i], mnew, vnew.sqrt()));
            let choice = categorical_from_logs(&logs, rng)?;
            if choice == k {
                let (post_m, post_sd) = posterior_mu(&[x[i]], sigma, mu0, tau0);
                means.push(post_m + post_sd * rng.sample(StandardNormal));
                z[i] = k;
            } else {
                z[i] = choice;
            }
        }
        // Refresh cluster means from their full conditionals.
        let k = means.len();
        for c in 0..k {
            let pts: Vec<f64> = (0..n).filter(|&i| z[i] == c).map(|i| x[i]).collect();
            if pts.is_empty() {
                continue;
            }
            let (m, sd) = posterior_mu(&pts, sigma, mu0, tau0);
            means[c] = m + sd * rng.sample(StandardNormal);
        }
    }
    compact_labels(&mut z, &mut means);
    let sizes = cluster_sizes(&z, means.len());
    let mut lp = 0.0;
    for i in 0..n {
        let c = z[i];
        lp += log_normal_density(x[i], means[c], sigma);
    }
    Ok(DpGaussianFit {
        n_clusters: means.len(),
        assignments: z,
        means,
        sizes,
        alpha,
        log_predictive: lp,
    })
}

/// Chinese restaurant franchise (Teh et al. 2006, §4–5) for grouped
/// univariate Gaussians with shared dishes.
///
/// Group $j$ is a restaurant. Observation $x_{ji}$ is a customer. Tables
/// choose dishes from a global CRP($\gamma$); customers choose tables from
/// a local CRP($\alpha$). Likelihood is $N(\mu_{\mathrm{dish}}, \sigma^2)$
/// with $\mu \sim N(\mu_0, \tau_0^2)$.
///
/// **Deviations from Teh, Jordan, Beal & Blei (2006):**
/// - Observations are univariate Gaussian, not the paper's multinomial
///   document-topic likelihood. The CRF seating is the same; the
///   predictive $f_k(x)$ is the conjugate Gaussian marginal.
/// - $\sigma^2$ is fixed.
/// - We do **not** implement the direct-assignment (no-table) sampler of
///   §5.3; tables are stored explicitly.
/// - Multiple tables in one restaurant **can** share a dish (true CRF).
pub fn hdp_crf_gaussian_gibbs<R: Rng + ?Sized>(
    groups: &[Vec<f64>],
    alpha: f64,
    gamma: f64,
    sigma: f64,
    mu0: f64,
    tau0: f64,
    n_iter: usize,
    rng: &mut R,
) -> Result<HdpGaussianFit> {
    if groups.is_empty() || groups.iter().any(|g| g.is_empty()) {
        return Err(Error::param("HDP needs non-empty groups"));
    }
    if !(alpha > 0.0 && gamma > 0.0 && sigma > 0.0 && tau0 > 0.0 && n_iter > 0) {
        return Err(Error::param("HDP: α, γ, σ, τ0 > 0 and n_iter ≥ 1"));
    }
    let g = groups.len();
    // table_of[j][i] = table index in restaurant j
    let mut table_of: Vec<Vec<usize>> = groups.iter().map(|gj| vec![0; gj.len()]).collect();
    // tables[j][t] = dish id
    let mut tables: Vec<Vec<usize>> = vec![vec![0]; g];
    let mut dish_means = vec![mu0];

    for _ in 0..n_iter {
        for j in 0..g {
            for i in 0..groups[j].len() {
                reseat_customer(
                    groups,
                    j,
                    i,
                    alpha,
                    gamma,
                    sigma,
                    mu0,
                    tau0,
                    &mut table_of,
                    &mut tables,
                    &mut dish_means,
                    rng,
                )?;
            }
        }
        // Resample each table's dish.
        for j in 0..g {
            let n_tab = tables[j].len();
            for t in 0..n_tab {
                resample_table_dish(
                    groups,
                    j,
                    t,
                    gamma,
                    sigma,
                    mu0,
                    tau0,
                    &table_of,
                    &mut tables,
                    &mut dish_means,
                    rng,
                )?;
            }
        }
        // Refresh dish means.
        let k = dish_means.len();
        for d in 0..k {
            let pts = collect_dish_points(groups, &table_of, &tables, d);
            if pts.is_empty() {
                continue;
            }
            let (m, sd) = posterior_mu(&pts, sigma, mu0, tau0);
            dish_means[d] = m + sd * rng.sample(StandardNormal);
        }
        compact_dishes(&mut table_of, &mut tables, &mut dish_means);
    }
    let assignments = assignments_from_tables(&table_of, &tables);
    let n_tables = tables.iter().map(|t| t.len()).sum();
    Ok(HdpGaussianFit {
        n_dishes: dish_means.len(),
        assignments,
        dish_means,
        n_tables,
        alpha,
        gamma,
    })
}

/// Collapsed Gibbs for the linear-Gaussian IBP
/// $X = ZA + E$, $A_{kd}\sim N(0,\sigma_A^2)$, $E_{nd}\sim N(0,\sigma_X^2)$
/// (Griffiths & Ghahramani 2011, §4.1).
///
/// Existing features: $P(z_{nk}=1\mid Z_{-nk}) = m_{-n,k}/N$.
/// New features: $\kappa \in \{0,\ldots,\kappa_{\max}\}$ is scored by
/// $\mathrm{Poisson}(\kappa; \alpha/N)\,p(X\mid Z^{+k})$.
///
/// **Deviations from Griffiths & Ghahramani (2011):**
/// - $A$ is integrated out during the $Z$ sweep, then drawn once from its
///   Gaussian full conditional at the end (they keep $A$ collapsed).
/// - New-feature count is enumerated up to `kappa_max` (default 4), not
///   proposed with Metropolis–Hastings on an unbounded Poisson.
/// - Empty columns are dropped after every row (bookkeeping; same
///   posterior on the nonempty support).
/// - $\sigma_X, \sigma_A, \alpha$ are fixed (no hyperpriors).
pub fn ibp_linear_gaussian_gibbs<R: Rng + ?Sized>(
    x: &Mat<f64>,
    alpha: f64,
    sigma_x: f64,
    sigma_a: f64,
    n_iter: usize,
    rng: &mut R,
) -> Result<IbpLinearGaussianFit> {
    ibp_linear_gaussian_gibbs_ex(x, alpha, sigma_x, sigma_a, n_iter, 4, rng)
}

/// Same as [`ibp_linear_gaussian_gibbs`] with an explicit $\kappa_{\max}$.
pub fn ibp_linear_gaussian_gibbs_ex<R: Rng + ?Sized>(
    x: &Mat<f64>,
    alpha: f64,
    sigma_x: f64,
    sigma_a: f64,
    n_iter: usize,
    kappa_max: usize,
    rng: &mut R,
) -> Result<IbpLinearGaussianFit> {
    let n = x.nrows();
    let d = x.ncols();
    if n == 0 || d == 0 {
        return Err(Error::param("IBP Gibbs needs a nonempty X"));
    }
    if !(alpha > 0.0 && sigma_x > 0.0 && sigma_a > 0.0 && n_iter > 0) {
        return Err(Error::param("IBP Gibbs: α, σX, σA > 0 and n_iter ≥ 1"));
    }
    // Start from a sequential IBP draw (not in GG 2011, who start at empty
    // or a finite-K matrix). This is an initialization choice only.
    let mut z = crate::npbayes::beta_bernoulli::sample_ibp_sequential(
        n,
        crate::npbayes::beta_bernoulli::IbpParams::new(alpha)?,
        rng,
    )?;
    for _ in 0..n_iter {
        for i in 0..n {
            // Existing columns.
            let mut k = 0;
            while k < z.k {
                let m = z.column_sum_except(k, i);
                if m == 0 {
                    // Singleton belonging only to i: treat as a new-feature
                    // candidate; drop it here.
                    drop_column(&mut z, k);
                    continue;
                }
                let prior1 = m as f64 / n as f64;
                z.set(i, k, 0);
                let ll0 = collapsed_loglik(x, &z, sigma_x, sigma_a)?;
                z.set(i, k, 1);
                let ll1 = collapsed_loglik(x, &z, sigma_x, sigma_a)?;
                let log0 = (1.0 - prior1).max(1e-16).ln() + ll0;
                let log1 = prior1.max(1e-16).ln() + ll1;
                let p1 = softmax2(log1, log0);
                z.set(i, k, if rng.next_f64() < p1 { 1 } else { 0 });
                k += 1;
            }
            // New features: enumerate κ = 0..κ_max.
            let lam = alpha / n as f64;
            let mut logs = Vec::with_capacity(kappa_max + 1);
            let mut candidates: Vec<FeatureMatrix> = Vec::with_capacity(kappa_max + 1);
            for kappa in 0..=kappa_max {
                let mut zc = z.clone();
                if kappa > 0 {
                    let start = zc.k;
                    zc.add_columns(kappa);
                    for t in 0..kappa {
                        zc.set(i, start + t, 1);
                    }
                }
                let ll = collapsed_loglik(x, &zc, sigma_x, sigma_a)?;
                logs.push(log_poisson_pmf(kappa as f64, lam) + ll);
                candidates.push(zc);
            }
            let choice = categorical_from_logs(&logs, rng)?;
            z = candidates[choice].clone();
            z.drop_empty_columns();
        }
    }
    z.drop_empty_columns();
    let a = sample_loadings(x, &z, sigma_x, sigma_a, rng)?;
    let loglik = collapsed_loglik(x, &z, sigma_x, sigma_a)?;
    Ok(IbpLinearGaussianFit {
        z,
        a,
        alpha,
        sigma_x,
        sigma_a,
        loglik,
    })
}

/// Finite Beta–Bernoulli linear-Gaussian Gibbs with fixed $K$.
///
/// $\pi_k \sim \mathrm{Beta}(\alpha/K, 1)$, $z_{nk}\mid\pi_k\sim\mathrm{Bern}(\pi_k)$,
/// $X=ZA+E$ as above. $\pi$ is Gibbs-updated from its Beta full conditional;
/// $Z$ uses the collapsed likelihood times $\pi_k$.
///
/// **Deviation:** Griffiths & Ghahramani's finite model is usually used as
/// an approximation to the IBP ($K\to\infty$). We keep $K$ fixed and never
/// birth new columns. $A$ is drawn at the end only.
pub fn finite_beta_bernoulli_gibbs<R: Rng + ?Sized>(
    x: &Mat<f64>,
    k: usize,
    alpha: f64,
    sigma_x: f64,
    sigma_a: f64,
    n_iter: usize,
    rng: &mut R,
) -> Result<IbpLinearGaussianFit> {
    let n = x.nrows();
    if k == 0 {
        return Err(Error::param("finite Beta–Bernoulli needs K ≥ 1"));
    }
    if !(alpha > 0.0 && sigma_x > 0.0 && sigma_a > 0.0 && n_iter > 0) {
        return Err(Error::param("finite BBP: α, σX, σA > 0 and n_iter ≥ 1"));
    }
    let mut z = FeatureMatrix::zeros(n, k);
    let mut pi = vec![0.5; k];
    for kk in 0..k {
        pi[kk] = sample_beta(alpha / k as f64, 1.0, rng).map_err(|e| Error::numeric(e.message))?;
        for i in 0..n {
            z.set(i, kk, if rng.next_f64() < pi[kk] { 1 } else { 0 });
        }
    }
    for _ in 0..n_iter {
        for i in 0..n {
            for kk in 0..k {
                z.set(i, kk, 0);
                let ll0 = collapsed_loglik(x, &z, sigma_x, sigma_a)?;
                z.set(i, kk, 1);
                let ll1 = collapsed_loglik(x, &z, sigma_x, sigma_a)?;
                let log0 = (1.0 - pi[kk]).max(1e-16).ln() + ll0;
                let log1 = pi[kk].max(1e-16).ln() + ll1;
                let p1 = softmax2(log1, log0);
                z.set(i, kk, if rng.next_f64() < p1 { 1 } else { 0 });
            }
        }
        for kk in 0..k {
            let m = z.column_sum(kk) as f64;
            pi[kk] = sample_beta(alpha / k as f64 + m, 1.0 + n as f64 - m, rng)
                .map_err(|e| Error::numeric(e.message))?;
        }
    }
    let a = sample_loadings(x, &z, sigma_x, sigma_a, rng)?;
    let loglik = collapsed_loglik(x, &z, sigma_x, sigma_a)?;
    Ok(IbpLinearGaussianFit {
        z,
        a,
        alpha,
        sigma_x,
        sigma_a,
        loglik,
    })
}

// ---------------------------------------------------------------------------
// Internals
// ---------------------------------------------------------------------------

fn log_normal_density(x: f64, mu: f64, sd: f64) -> f64 {
    let z = (x - mu) / sd;
    -0.5 * LN_2PI - sd.ln() - 0.5 * z * z
}

fn posterior_mu(xs: &[f64], sigma: f64, mu0: f64, tau0: f64) -> (f64, f64) {
    let n = xs.len() as f64;
    let s2 = sigma * sigma;
    let t2 = tau0 * tau0;
    let prec = 1.0 / t2 + n / s2;
    let mean = if xs.is_empty() {
        mu0
    } else {
        let sum: f64 = xs.iter().sum();
        (mu0 / t2 + sum / s2) / prec
    };
    (mean, (1.0 / prec).sqrt())
}

fn pred_mean_var(
    x: &[f64],
    z: &[usize],
    cluster: usize,
    sigma: f64,
    mu0: f64,
    tau0: f64,
) -> (f64, f64) {
    let pts: Vec<f64> = x
        .iter()
        .enumerate()
        .filter(|(i, _)| z[*i] == cluster)
        .map(|(_, v)| *v)
        .collect();
    let (m, sd) = posterior_mu(&pts, sigma, mu0, tau0);
    (m, sd * sd + sigma * sigma)
}

fn cluster_sizes(z: &[usize], k: usize) -> Vec<usize> {
    let mut s = vec![0; k];
    for &c in z {
        if c < k {
            s[c] += 1;
        }
    }
    s
}

fn remove_cluster(z: &mut [usize], means: &mut Vec<f64>, dead: usize) {
    means.remove(dead);
    for zi in z.iter_mut() {
        if *zi == usize::MAX {
            continue;
        }
        if *zi == dead {
            *zi = usize::MAX;
        } else if *zi > dead {
            *zi -= 1;
        }
    }
}

fn compact_labels(z: &mut [usize], means: &mut Vec<f64>) {
    let k = means.len();
    let sizes = cluster_sizes(z, k);
    let keep: Vec<usize> = (0..k).filter(|&c| sizes[c] > 0).collect();
    if keep.len() == k {
        return;
    }
    let mut new_means = Vec::new();
    let mut map = vec![0usize; k];
    for (nk, &c) in keep.iter().enumerate() {
        map[c] = nk;
        new_means.push(means[c]);
    }
    for zi in z.iter_mut() {
        if *zi < k {
            *zi = map[*zi];
        }
    }
    *means = new_means;
}

fn categorical_from_logs<R: Rng + ?Sized>(logs: &[f64], rng: &mut R) -> Result<usize> {
    if logs.is_empty() {
        return Err(Error::numeric("empty categorical"));
    }
    let m = logs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mut w = Vec::with_capacity(logs.len());
    let mut s = 0.0;
    for &l in logs {
        let e = (l - m).exp();
        w.push(e);
        s += e;
    }
    if !(s > 0.0 && s.is_finite()) {
        return Err(Error::numeric("categorical weights not finite"));
    }
    let u = rng.next_f64() * s;
    let mut acc = 0.0;
    for (i, wi) in w.iter().enumerate() {
        acc += *wi;
        if u < acc {
            return Ok(i);
        }
    }
    Ok(w.len() - 1)
}

fn softmax2(a: f64, b: f64) -> f64 {
    let m = a.max(b);
    let ea = (a - m).exp();
    let eb = (b - m).exp();
    ea / (ea + eb)
}

fn log_poisson_pmf(k: f64, lam: f64) -> f64 {
    if lam <= 0.0 {
        return if k == 0.0 { 0.0 } else { f64::NEG_INFINITY };
    }
    k * lam.ln() - lam - log_factorial(k)
}

fn log_factorial(n: f64) -> f64 {
    if n <= 1.0 {
        return 0.0;
    }
    (n + 0.5) * n.ln() - n + 0.5 * (2.0 * std::f64::consts::PI).ln() + 1.0 / (12.0 * n)
        - 1.0 / (360.0 * n * n * n)
}

/// Griffiths & Ghahramani collapsed likelihood (up to $2\pi$ constants
/// that cancel in Gibbs ratios, but we keep them so the stored `loglik`
/// is the actual marginal $p(X\mid Z)$).
fn collapsed_loglik(x: &Mat<f64>, z: &FeatureMatrix, sigma_x: f64, sigma_a: f64) -> Result<f64> {
    let n = x.nrows();
    let d = x.ncols();
    if z.n != n {
        return Err(Error::dim("Z rows != X rows"));
    }
    let k = z.k;
    let sx2 = sigma_x * sigma_x;
    let sa2 = sigma_a * sigma_a;
    if k == 0 {
        let mut ss = 0.0;
        for i in 0..n {
            for j in 0..d {
                ss += x[(i, j)] * x[(i, j)];
            }
        }
        return Ok(-0.5 * (n * d) as f64 * LN_2PI
            - (n * d) as f64 * sigma_x.ln()
            - ss / (2.0 * sx2));
    }
    let ratio = sx2 / sa2;
    let mut ztz = mat_identity(k);
    for a in 0..k {
        for b in 0..k {
            let mut s = 0.0;
            for i in 0..n {
                s += z.get(i, a) as f64 * z.get(i, b) as f64;
            }
            ztz[(a, b)] = s;
            if a == b {
                ztz[(a, b)] += ratio;
            }
        }
    }
    let m = spd_regularize(ztz, 1e-12)?;
    let ld = logdet_spd(&m)?;
    // Zᵀ X  (K × D) stored row-major
    let mut ztx = vec![0.0; k * d];
    for a in 0..k {
        for j in 0..d {
            let mut s = 0.0;
            for i in 0..n {
                s += z.get(i, a) as f64 * x[(i, j)];
            }
            ztx[a * d + j] = s;
        }
    }
    let mut xf = 0.0;
    for i in 0..n {
        for j in 0..d {
            xf += x[(i, j)] * x[(i, j)];
        }
    }
    let mut quad = 0.0;
    for j in 0..d {
        let col = col_from_slice(&(0..k).map(|a| ztx[a * d + j]).collect::<Vec<_>>());
        let w = solve_spd(&m, &col)?;
        for a in 0..k {
            quad += ztx[a * d + j] * w[a];
        }
    }
    let ll = -0.5 * (n * d) as f64 * LN_2PI
        - (n as f64 - k as f64) * d as f64 * sigma_x.ln()
        - (k * d) as f64 * sigma_a.ln()
        - 0.5 * d as f64 * ld
        - (xf - quad) / (2.0 * sx2);
    Ok(ll)
}

fn sample_loadings<R: Rng + ?Sized>(
    x: &Mat<f64>,
    z: &FeatureMatrix,
    sigma_x: f64,
    sigma_a: f64,
    rng: &mut R,
) -> Result<Mat<f64>> {
    let d = x.ncols();
    let k = z.k;
    if k == 0 {
        return Ok(Mat::zeros(0, d));
    }
    let n = x.nrows();
    let sx2 = sigma_x * sigma_x;
    let sa2 = sigma_a * sigma_a;
    let mut ztz = mat_identity(k);
    for a in 0..k {
        for b in 0..k {
            let mut s = 0.0;
            for i in 0..n {
                s += z.get(i, a) as f64 * z.get(i, b) as f64;
            }
            ztz[(a, b)] = s / sx2;
            if a == b {
                ztz[(a, b)] += 1.0 / sa2;
            }
        }
    }
    let prec = spd_regularize(ztz, 1e-12)?;
    // Mean: Prec^{-1} Zᵀ X / σX²
    let mut a = Mat::zeros(k, d);
    for j in 0..d {
        let mut rhs = Col::zeros(k);
        for t in 0..k {
            let mut s = 0.0;
            for i in 0..n {
                s += z.get(i, t) as f64 * x[(i, j)];
            }
            rhs[t] = s / sx2;
        }
        let mean = solve_spd(&prec, &rhs)?;
        // N(mean, Prec⁻¹): L Lᵀ = Prec, e = L^{-T} g, sample = mean + e.
        let l = crate::linalg::cholesky(&prec)?;
        let mut g = Col::zeros(k);
        for t in 0..k {
            g[t] = rng.sample(StandardNormal);
        }
        let e = solve_upper(&l, &g);
        for t in 0..k {
            a[(t, j)] = mean[t] + e[t];
        }
    }
    Ok(a)
}

fn solve_upper(l: &Mat<f64>, g: &Col<f64>) -> Col<f64> {
    // L is lower; solve Lᵀ e = g
    let n = l.nrows();
    let mut e = Col::zeros(n);
    for i in (0..n).rev() {
        let mut s = g[i];
        for k in (i + 1)..n {
            s -= l[(k, i)] * e[k];
        }
        let d = l[(i, i)];
        e[i] = if d.abs() > 0.0 { s / d } else { s };
    }
    e
}

fn drop_column(z: &mut FeatureMatrix, dead: usize) {
    let mut next = FeatureMatrix::zeros(z.n, z.k.saturating_sub(1));
    for i in 0..z.n {
        let mut nk = 0;
        for k in 0..z.k {
            if k == dead {
                continue;
            }
            next.set(i, nk, z.get(i, k));
            nk += 1;
        }
    }
    *z = next;
}

// ---- HDP CRF helpers -------------------------------------------------------

fn reseat_customer<R: Rng + ?Sized>(
    groups: &[Vec<f64>],
    j: usize,
    i: usize,
    alpha: f64,
    gamma: f64,
    sigma: f64,
    mu0: f64,
    tau0: f64,
    table_of: &mut [Vec<usize>],
    tables: &mut [Vec<usize>],
    dish_means: &mut Vec<f64>,
    rng: &mut R,
) -> Result<()> {
    let old_t = table_of[j][i];
    table_of[j][i] = usize::MAX;
    // Drop empty table.
    if old_t < tables[j].len() && table_occupancy(table_of, j, old_t) == 0 {
        drop_table(table_of, tables, j, old_t);
    }
    compact_dishes(table_of, tables, dish_means);

    let x = groups[j][i];
    let n_tab = tables[j].len();
    let mut logs = Vec::new();
    // Existing tables in this restaurant.
    for t in 0..n_tab {
        let n_t = table_occupancy(table_of, j, t) as f64;
        let d = tables[j][t];
        logs.push(n_t.ln() + log_normal_density(x, dish_means[d], sigma));
    }
    // New table: mixture over existing dishes + new dish.
    let m_dot = tables.iter().map(|tj| tj.len()).sum::<usize>() as f64;
    let k = dish_means.len();
    let mut new_table_parts = Vec::with_capacity(k + 1);
    let mut dish_counts = vec![0.0; k];
    for tj in tables.iter() {
        for &d in tj {
            if d < k {
                dish_counts[d] += 1.0;
            }
        }
    }
    for d in 0..k {
        let w = dish_counts[d] / (m_dot + gamma);
        new_table_parts.push(w.ln() + log_normal_density(x, dish_means[d], sigma));
    }
    let vnew = sigma * sigma + tau0 * tau0;
    new_table_parts.push((gamma / (m_dot + gamma)).ln() + log_normal_density(x, mu0, vnew.sqrt()));
    // log Σ exp
    let m = new_table_parts
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    let mix = m + new_table_parts
        .iter()
        .map(|l| (l - m).exp())
        .sum::<f64>()
        .ln();
    logs.push(alpha.ln() + mix);

    let choice = categorical_from_logs(&logs, rng)?;
    if choice < n_tab {
        table_of[j][i] = choice;
    } else {
        // Open a table; pick a dish from the same mixture.
        let dish_choice = categorical_from_logs(&new_table_parts, rng)?;
        let dish = if dish_choice == k {
            let (pm, psd) = posterior_mu(&[x], sigma, mu0, tau0);
            dish_means.push(pm + psd * rng.sample(StandardNormal));
            dish_means.len() - 1
        } else {
            dish_choice
        };
        tables[j].push(dish);
        table_of[j][i] = tables[j].len() - 1;
    }
    Ok(())
}

fn resample_table_dish<R: Rng + ?Sized>(
    groups: &[Vec<f64>],
    j: usize,
    t: usize,
    gamma: f64,
    sigma: f64,
    mu0: f64,
    tau0: f64,
    table_of: &[Vec<usize>],
    tables: &mut [Vec<usize>],
    dish_means: &mut Vec<f64>,
    rng: &mut R,
) -> Result<()> {
    if t >= tables[j].len() {
        return Ok(());
    }
    let pts = collect_table_points(groups, table_of, j, t);
    if pts.is_empty() {
        return Ok(());
    }
    let old = tables[j][t];
    tables[j][t] = usize::MAX;
    compact_dishes_keep_placeholder(tables);

    let k = dish_means.len();
    let mut dish_counts = vec![0.0; k];
    for tj in tables.iter() {
        for &d in tj {
            if d < k {
                dish_counts[d] += 1.0;
            }
        }
    }
    let mut logs = Vec::with_capacity(k + 1);
    for d in 0..k {
        let mut lp = (dish_counts[d] as f64).ln();
        for &x in &pts {
            lp += log_normal_density(x, dish_means[d], sigma);
        }
        logs.push(lp);
    }
    let mut lp_new = gamma.ln();
    let vnew = (sigma * sigma + tau0 * tau0).sqrt();
    for &x in &pts {
        lp_new += log_normal_density(x, mu0, vnew);
    }
    logs.push(lp_new);
    let choice = categorical_from_logs(&logs, rng)?;
    if choice == k {
        let (pm, psd) = posterior_mu(&pts, sigma, mu0, tau0);
        dish_means.push(pm + psd * rng.sample(StandardNormal));
        tables[j][t] = dish_means.len() - 1;
    } else {
        tables[j][t] = choice;
    }
    let _ = old;
    Ok(())
}

fn table_occupancy(table_of: &[Vec<usize>], j: usize, t: usize) -> usize {
    table_of[j].iter().filter(|&&u| u == t).count()
}

fn drop_table(table_of: &mut [Vec<usize>], tables: &mut [Vec<usize>], j: usize, dead: usize) {
    tables[j].remove(dead);
    for u in table_of[j].iter_mut() {
        if *u == usize::MAX {
            continue;
        }
        if *u == dead {
            *u = usize::MAX;
        } else if *u > dead {
            *u -= 1;
        }
    }
}

fn collect_table_points(
    groups: &[Vec<f64>],
    table_of: &[Vec<usize>],
    j: usize,
    t: usize,
) -> Vec<f64> {
    (0..groups[j].len())
        .filter(|&i| table_of[j][i] == t)
        .map(|i| groups[j][i])
        .collect()
}

fn collect_dish_points(
    groups: &[Vec<f64>],
    table_of: &[Vec<usize>],
    tables: &[Vec<usize>],
    dish: usize,
) -> Vec<f64> {
    let mut pts = Vec::new();
    for j in 0..groups.len() {
        for i in 0..groups[j].len() {
            let t = table_of[j][i];
            if t < tables[j].len() && tables[j][t] == dish {
                pts.push(groups[j][i]);
            }
        }
    }
    pts
}

fn assignments_from_tables(table_of: &[Vec<usize>], tables: &[Vec<usize>]) -> Vec<Vec<usize>> {
    table_of
        .iter()
        .enumerate()
        .map(|(j, tj)| {
            tj.iter()
                .map(|&t| if t < tables[j].len() { tables[j][t] } else { 0 })
                .collect()
        })
        .collect()
}

fn compact_dishes(
    table_of: &mut [Vec<usize>],
    tables: &mut [Vec<usize>],
    dish_means: &mut Vec<f64>,
) {
    let k = dish_means.len();
    let mut used = vec![false; k];
    for tj in tables.iter() {
        for &d in tj {
            if d < k {
                used[d] = true;
            }
        }
    }
    let keep: Vec<usize> = (0..k).filter(|&d| used[d]).collect();
    if keep.len() == k {
        return;
    }
    let mut map = vec![0usize; k];
    let mut new_means = Vec::new();
    for (nk, &d) in keep.iter().enumerate() {
        map[d] = nk;
        new_means.push(dish_means[d]);
    }
    for tj in tables.iter_mut() {
        for d in tj.iter_mut() {
            if *d < k {
                *d = map[*d];
            }
        }
    }
    *dish_means = new_means;
    let _ = table_of;
}

fn compact_dishes_keep_placeholder(tables: &mut [Vec<usize>]) {
    let _ = tables;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::mat_from_row_slice;
    use crate::rng::seed_rng;

    #[test]
    fn dp_mixture_recovers_two_blobs() {
        let mut rng = seed_rng(21);
        let mut x = Vec::new();
        for _ in 0..40 {
            x.push(-3.0 + 0.25 * rng.sample(StandardNormal));
        }
        for _ in 0..40 {
            x.push(3.0 + 0.25 * rng.sample(StandardNormal));
        }
        let fit = dp_gaussian_mixture_gibbs(&x, 0.5, 0.3, 0.0, 3.0, 40, &mut rng).unwrap();
        assert!(
            fit.n_clusters >= 2 && fit.n_clusters <= 6,
            "clusters {}",
            fit.n_clusters
        );
        let mut mins = fit.means.clone();
        mins.sort_by(|a, b| a.partial_cmp(b).unwrap());
        // At least one mean near −3 and one near +3.
        let has_lo = mins.iter().any(|m| (m + 3.0).abs() < 1.2);
        let has_hi = mins.iter().any(|m| (m - 3.0).abs() < 1.2);
        assert!(has_lo && has_hi, "means {:?}", fit.means);
    }

    #[test]
    fn ibp_gibbs_runs_and_finite_loglik() {
        let mut rng = seed_rng(22);
        // X = Z A + E with two obvious features.
        let n = 24;
        let d = 2;
        let mut xdata = vec![0.0; n * d];
        for i in 0..n {
            let f1 = if i < 12 { 1.0 } else { 0.0 };
            let f2 = if i % 2 == 0 { 1.0 } else { 0.0 };
            xdata[i * d] = 2.5 * f1 + 0.15 * rng.sample(StandardNormal);
            xdata[i * d + 1] = 2.5 * f2 + 0.15 * rng.sample(StandardNormal);
        }
        let x = mat_from_row_slice(n, d, &xdata);
        let fit = ibp_linear_gaussian_gibbs_ex(&x, 1.0, 0.2, 1.0, 8, 3, &mut rng).unwrap();
        assert!(fit.loglik.is_finite());
        assert!(fit.z.k >= 1);
        assert_eq!(fit.z.n, n);
    }

    #[test]
    fn finite_bbp_fixed_k() {
        let mut rng = seed_rng(23);
        let n = 16;
        let d = 1;
        let mut xdata = vec![0.0; n];
        for i in 0..n {
            xdata[i] = if i < 8 { 1.5 } else { 0.1 } + 0.1 * rng.sample(StandardNormal);
        }
        let x = mat_from_row_slice(n, d, &xdata);
        let fit = finite_beta_bernoulli_gibbs(&x, 3, 1.0, 0.3, 1.0, 6, &mut rng).unwrap();
        assert_eq!(fit.z.k, 3);
        assert!(fit.loglik.is_finite());
    }

    #[test]
    fn hdp_crf_runs() {
        let mut rng = seed_rng(24);
        let mut g1 = Vec::new();
        let mut g2 = Vec::new();
        for _ in 0..20 {
            g1.push(-2.0 + 0.2 * rng.sample(StandardNormal));
            g2.push(2.0 + 0.2 * rng.sample(StandardNormal));
        }
        let fit =
            hdp_crf_gaussian_gibbs(&[g1, g2], 1.0, 1.0, 0.25, 0.0, 2.0, 15, &mut rng).unwrap();
        assert!(fit.n_dishes >= 1);
        assert_eq!(fit.assignments.len(), 2);
        assert!(fit.n_tables >= 2);
    }
}
