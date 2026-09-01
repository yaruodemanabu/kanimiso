//! High-frequency statistics: realized measures, Hayashi–Yoshida,
//! lead–lag, bipower jump tests (YUIMA `cce`, `llag`, `hyavar`, `bns.test`).

use crate::error::{Error, Result};
use crate::path::{AsyncData, Path, TickSeries};

/// Realized covariance of a **synchronous** path (`Σ ΔX ΔXᵀ`).
pub fn realized_covariance(path: &Path) -> Result<faer::Mat<f64>> {
    let d = path.dim();
    let mut cov = crate::linalg::mat_zeros(d, d);
    for i in 0..path.n_steps() {
        for a in 0..d {
            let da = path.state(i + 1)[a] - path.state(i)[a];
            for b in 0..d {
                let db = path.state(i + 1)[b] - path.state(i)[b];
                cov[(a, b)] += da * db;
            }
        }
    }
    Ok(cov)
}

/// Realized volatility of coordinate `j`.
pub fn realized_variance(path: &Path, j: usize) -> Result<f64> {
    path.quadratic_variation(j)
}

/// Bipower variation `(π/2) Σ |ΔXᵢ| |ΔXᵢ₋₁|` (Barndorff-Nielsen–Shephard).
pub fn bipower_variation(path: &Path, j: usize) -> Result<f64> {
    let dx = path.increments(j)?;
    if dx.len() < 2 {
        return Err(Error::infer("bipower variation needs ≥ 2 increments"));
    }
    let mut s = 0.0;
    for i in 1..dx.len() {
        s += dx[i].abs() * dx[i - 1].abs();
    }
    Ok(std::f64::consts::FRAC_PI_2 * s)
}

/// Realized quarticity ` (n / 3) Σ (ΔX)⁴ ` (for regular grids).
pub fn realized_quarticity(path: &Path, j: usize) -> Result<f64> {
    let dx = path.increments(j)?;
    let n = dx.len() as f64;
    Ok((n / 3.0) * dx.iter().map(|x| x.powi(4)).sum::<f64>())
}

/// BNS jump test: `QV − BV` studentized by quarticity.
#[derive(Clone, Debug)]
pub struct BnsTest {
    pub qv: f64,
    pub bv: f64,
    pub jump_component: f64,
    pub statistic: f64,
    pub pvalue: f64,
}

pub fn bns_jump_test(path: &Path, j: usize) -> Result<BnsTest> {
    let qv = realized_variance(path, j)?;
    let bv = bipower_variation(path, j)?;
    let rq = realized_quarticity(path, j)?;
    let n = path.n_steps() as f64;
    // θ = π²/4 + π − 5
    let theta = std::f64::consts::PI * std::f64::consts::PI / 4.0 + std::f64::consts::PI - 5.0;
    let denom = (theta * rq).max(0.0).sqrt() / n.sqrt();
    let stat = if denom > 0.0 { (qv - bv) / denom } else { 0.0 };
    Ok(BnsTest {
        qv,
        bv,
        jump_component: (qv - bv).max(0.0),
        statistic: stat,
        pvalue: 1.0 - 0.5 * (1.0 + erf(stat / std::f64::consts::SQRT_2)),
    })
}

/// Four-power variation `μ₁⁻⁴ n Σ |Δ_{i} Δ_{i+1} Δ_{i+2} Δ_{i+3}|` (BNS IQ).
pub fn tripower_quarticity(path: &Path, j: usize) -> Result<f64> {
    let dx = path.increments(j)?;
    if dx.len() < 4 {
        return Err(Error::infer("four-power needs ≥ 4 increments"));
    }
    let mu1 = (2.0 / std::f64::consts::PI).sqrt();
    let n = dx.len() as f64;
    let mut s = 0.0;
    for i in 0..dx.len() - 3 {
        s += dx[i].abs() * dx[i + 1].abs() * dx[i + 2].abs() * dx[i + 3].abs();
    }
    Ok(n * s / mu1.powi(4))
}

/// BNS ratio statistic `(BV / RV − 1) / √(θ · IQ / (n RV²))`.
pub fn bns_ratio_test(path: &Path, j: usize) -> Result<BnsTest> {
    let qv = realized_variance(path, j)?;
    let bv = bipower_variation(path, j)?;
    let iq = tripower_quarticity(path, j)?;
    let n = path.n_steps() as f64;
    let theta = std::f64::consts::PI * std::f64::consts::PI / 4.0 + std::f64::consts::PI - 5.0;
    let denom = (theta * iq / (n * qv * qv)).max(0.0).sqrt();
    let stat = if denom > 0.0 {
        (bv / qv - 1.0) / denom
    } else {
        0.0
    };
    Ok(BnsTest {
        qv,
        bv,
        jump_component: (qv - bv).max(0.0),
        statistic: stat,
        pvalue: 1.0 - 0.5 * (1.0 + erf(stat.abs() / std::f64::consts::SQRT_2)),
    })
}

/// Lee–Mykland (2008) return / local-vol statistic.
#[derive(Clone, Debug)]
pub struct LeeMyklandTest {
    pub statistic: Vec<f64>,
    pub threshold: f64,
    pub jumps: Vec<usize>,
}

pub fn lee_mykland(path: &Path, j: usize, window: usize) -> Result<LeeMyklandTest> {
    let dx = path.increments(j)?;
    if window < 4 || dx.len() <= window {
        return Err(Error::infer("Lee–Mykland window too short"));
    }
    let n = dx.len();
    let mut stat = vec![0.0; n];
    let mut jumps = Vec::new();
    // Gumbel threshold: √(2 log n) − (log π + log log n) / (2 √(2 log n))
    let ln = (n as f64).ln();
    let thr = (2.0 * ln).sqrt() - (std::f64::consts::PI.ln() + ln.ln()) / (2.0 * (2.0 * ln).sqrt());
    for i in window..n {
        let mut bp = 0.0;
        for k in (i - window)..(i - 1) {
            bp += dx[k].abs() * dx[k + 1].abs();
        }
        let sig = (std::f64::consts::FRAC_PI_2 * bp / (window as f64 - 1.0)).sqrt();
        stat[i] = if sig > 0.0 { dx[i] / sig } else { 0.0 };
        if stat[i].abs() > thr {
            jumps.push(i);
        }
    }
    Ok(LeeMyklandTest {
        statistic: stat,
        threshold: thr,
        jumps,
    })
}

fn erf(x: f64) -> f64 {
    // 1 - erfc(x) using the same AS approximation as infer.rs
    let z = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * z);
    let a = t
        * (0.254829592
            + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    let y = 1.0 - a * (-z * z).exp();
    if x >= 0.0 {
        y
    } else {
        -y
    }
}

/// Hayashi–Yoshida covariance of two irregular series.
///
/// `HY = Σ_{i,j} ΔXᵢ ΔYⱼ 1_{ (tᵢ₋₁, tᵢ] ∩ (sⱼ₋₁, sⱼ] ≠ ∅ }`
pub fn hayashi_yoshida(x: &TickSeries, y: &TickSeries) -> f64 {
    let mut hy = 0.0;
    let mut j = 0;
    for i in 1..x.n() {
        let a0 = x.times[i - 1];
        let a1 = x.times[i];
        let dx = x.values[i] - x.values[i - 1];
        while j + 1 < y.n() && y.times[j + 1] <= a0 {
            j += 1;
        }
        let mut k = j;
        while k + 1 < y.n() && y.times[k] < a1 {
            let b0 = y.times[k];
            let b1 = y.times[k + 1];
            if intervals_overlap(a0, a1, b0, b1) {
                let dy = y.values[k + 1] - y.values[k];
                hy += dx * dy;
            }
            k += 1;
        }
    }
    hy
}

fn intervals_overlap(a0: f64, a1: f64, b0: f64, b1: f64) -> bool {
    a0 < b1 && b0 < a1
}

/// Hayashi–Yoshida covariance / correlation matrices (`cce`).
#[derive(Clone, Debug)]
pub struct Cce {
    pub cov: faer::Mat<f64>,
    pub corr: faer::Mat<f64>,
}

pub fn cce(data: &AsyncData) -> Result<Cce> {
    let d = data.dim();
    let mut cov = crate::linalg::mat_zeros(d, d);
    for i in 0..d {
        cov[(i, i)] = realized_var_ticks(&data.series[i]);
        for j in (i + 1)..d {
            let h = hayashi_yoshida(&data.series[i], &data.series[j]);
            cov[(i, j)] = h;
            cov[(j, i)] = h;
        }
    }
    let mut corr = crate::linalg::mat_zeros(d, d);
    for i in 0..d {
        corr[(i, i)] = 1.0;
        for j in (i + 1)..d {
            let dnm = (cov[(i, i)] * cov[(j, j)]).sqrt();
            let r = if dnm > 0.0 { cov[(i, j)] / dnm } else { 0.0 };
            let r = r.clamp(-1.0, 1.0);
            corr[(i, j)] = r;
            corr[(j, i)] = r;
        }
    }
    Ok(Cce { cov, corr })
}

fn realized_var_ticks(x: &TickSeries) -> f64 {
    x.increments().iter().map(|d| d * d).sum()
}

/// **Experimental / unverified** kernel asymptotic variance of Hayashi–Yoshida.
///
/// Diagonal: `(2/3) × realized quarticity`.
/// Off-diagonal: a homemade rectangular-kernel / delta-method estimator
/// (bandwidth default `n^{0.45}` has no cited source). Do not treat the
/// off-diagonal as Hayashi–Yoshida (2011) §8.2. Prefer the name
/// [`experimental_hy_avar`].
#[derive(Clone, Debug)]
pub struct HyAvar {
    pub cov: faer::Mat<f64>,
    pub corr: faer::Mat<f64>,
    pub avar_cov: faer::Mat<f64>,
    pub avar_corr: faer::Mat<f64>,
}

pub fn hy_avar(data: &AsyncData, bandwidth: Option<f64>) -> Result<HyAvar> {
    let est = cce(data)?;
    let d = data.dim();
    let mut avar_cov = crate::linalg::mat_zeros(d, d);
    let mut avar_corr = crate::linalg::mat_zeros(d, d);
    for i in 0..d {
        let dx: Vec<f64> = data.series[i].increments();
        let n = dx.len() as f64;
        let rq = (n / 3.0) * dx.iter().map(|x| x.powi(4)).sum::<f64>();
        avar_cov[(i, i)] = (2.0 / 3.0) * rq;
    }
    for i in 0..d {
        for j in (i + 1)..d {
            let bw = bandwidth.unwrap_or_else(|| {
                let ni = data.series[i].n().min(data.series[j].n()) as f64;
                ni.powf(0.45)
            });
            let v = hy_offdiag_avar(&data.series[i], &data.series[j], bw.max(1.0));
            avar_cov[(i, j)] = v;
            avar_cov[(j, i)] = v;
            // Delta-method variance of correlation ρ = c / √(vii vjj)
            let c = est.cov[(i, j)];
            let vii = est.cov[(i, i)].max(1e-18);
            let vjj = est.cov[(j, j)].max(1e-18);
            let denom = (vii * vjj).sqrt();
            let rho = if denom > 0.0 { c / denom } else { 0.0 };
            // Rough delta-method using only the three variances.
            let dc = 1.0 / denom;
            let dvii = -0.5 * c / (vii.powf(1.5) * vjj.sqrt());
            let dvjj = -0.5 * c / (vjj.powf(1.5) * vii.sqrt());
            let var_rho =
                (dc * dc * v + dvii * dvii * avar_cov[(i, i)] + dvjj * dvjj * avar_cov[(j, j)])
                    .max(0.0);
            avar_corr[(i, j)] = var_rho;
            avar_corr[(j, i)] = var_rho;
            let _ = rho;
        }
    }
    Ok(HyAvar {
        cov: est.cov,
        corr: est.corr,
        avar_cov,
        avar_corr,
    })
}

/// Alias of [`hy_avar`] that makes the unverified off-diagonal explicit.
pub fn experimental_hy_avar(data: &AsyncData, bandwidth: Option<f64>) -> Result<HyAvar> {
    hy_avar(data, bandwidth)
}

fn hy_offdiag_avar(x: &TickSeries, y: &TickSeries, bw: f64) -> f64 {
    // Pseudo-aggregate onto the union grid and apply a rectangular kernel
    // to products of overlapping increments (Hayashi–Yoshida 2011, §8.2).
    let mut prod = Vec::new();
    let mut j0 = 0;
    for i in 1..x.n() {
        let a0 = x.times[i - 1];
        let a1 = x.times[i];
        let dx = x.values[i] - x.values[i - 1];
        while j0 + 1 < y.n() && y.times[j0 + 1] <= a0 {
            j0 += 1;
        }
        let mut k = j0;
        while k + 1 < y.n() && y.times[k] < a1 {
            if intervals_overlap(a0, a1, y.times[k], y.times[k + 1]) {
                let dy = y.values[k + 1] - y.values[k];
                prod.push(dx * dy);
            }
            k += 1;
        }
    }
    if prod.is_empty() {
        return 0.0;
    }
    let w = bw.round().max(1.0) as usize;
    let mut v = 0.0;
    for i in 0..prod.len() {
        let mut acc = 0.0;
        let lo = i.saturating_sub(w);
        let hi = (i + w + 1).min(prod.len());
        for p in prod.iter().take(hi).skip(lo) {
            acc += p;
        }
        v += prod[i] * acc;
    }
    v.max(0.0)
}

/// Lead–lag estimator of Hoffmann–Rosenbaum–Yoshida (`llag`).
///
/// `θ̂ = argmax_θ |HY(X, Y_{·+θ})|` on a finite grid.
#[derive(Clone, Debug)]
pub struct LeadLag {
    pub theta: f64,
    pub hy_at_theta: f64,
    pub corr_at_theta: f64,
    pub grid: Vec<f64>,
    pub contrast: Vec<f64>,
}

pub fn lead_lag(x: &TickSeries, y: &TickSeries, grid: &[f64]) -> Result<LeadLag> {
    if grid.is_empty() {
        return Err(Error::infer("lead-lag grid is empty"));
    }
    let vx = realized_var_ticks(x).sqrt();
    let vy = realized_var_ticks(y).sqrt();
    let mut contrast = Vec::with_capacity(grid.len());
    let mut best_i = 0;
    let mut best_abs = -1.0;
    for (i, &th) in grid.iter().enumerate() {
        let ys = y.shift_time(th);
        let hy = hayashi_yoshida(x, &ys);
        contrast.push(hy);
        if hy.abs() > best_abs {
            best_abs = hy.abs();
            best_i = i;
        }
    }
    let hy = contrast[best_i];
    let corr = if vx * vy > 0.0 {
        (hy / (vx * vy)).clamp(-1.0, 1.0)
    } else {
        0.0
    };
    Ok(LeadLag {
        theta: grid[best_i],
        hy_at_theta: hy,
        corr_at_theta: corr,
        grid: grid.to_vec(),
        contrast,
    })
}

/// Default lead-lag grid on `[from, to]` with `division` points.
pub fn lead_lag_grid(from: f64, to: f64, division: usize) -> Result<Vec<f64>> {
    if to <= from || division < 2 {
        return Err(Error::infer("invalid lead-lag grid"));
    }
    let mut g = Vec::with_capacity(division);
    for i in 0..division {
        g.push(from + (to - from) * i as f64 / (division - 1) as f64);
    }
    Ok(g)
}

/// Pairwise lead-lag matrix for `AsyncData` (skew-symmetric `θᵢⱼ`).
pub fn lead_lag_matrix(data: &AsyncData, grid: &[f64]) -> Result<faer::Mat<f64>> {
    let d = data.dim();
    let mut th = crate::linalg::mat_zeros(d, d);
    for i in 0..d {
        for j in (i + 1)..d {
            let ll = lead_lag(&data.series[i], &data.series[j], grid)?;
            th[(i, j)] = ll.theta;
            th[(j, i)] = -ll.theta;
        }
    }
    Ok(th)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::path::TickSeries;

    #[test]
    fn hy_recovers_sync_covariance() {
        // Two identical series: HY = QV
        let t: Vec<f64> = (0..=100).map(|i| i as f64 * 0.01).collect();
        let x: Vec<f64> = t.iter().map(|u| u.sqrt()).collect();
        let a = TickSeries::new(t.clone(), x.clone()).unwrap();
        let b = TickSeries::new(t, x).unwrap();
        let hy = hayashi_yoshida(&a, &b);
        let qv = realized_var_ticks(&a);
        assert!((hy - qv).abs() < 1e-12);
    }

    #[test]
    fn lead_lag_detects_shift() {
        use crate::models::GeometricBrownianMotion;
        use crate::rng::seed_rng;
        use crate::sampling::Sampling;
        use crate::simulate::{simulate, SimConfig};

        // A Brownian-like path has weakly correlated increments, so the
        // Hayashi-Yoshida contrast peaks when the delayed clock is undone.
        let m = GeometricBrownianMotion::new(0.0, 0.3).unwrap();
        let s = Sampling::from_terminal(1.0, 400).unwrap();
        let mut rng = seed_rng(21);
        let path = simulate(&m, &s, &[1.0], &mut rng, &SimConfig::default()).unwrap();
        let a = TickSeries::new(path.times().to_vec(), path.component(0).unwrap()).unwrap();
        let b = a.shift_time(0.04);
        let grid = lead_lag_grid(-0.1, 0.1, 21).unwrap();
        let ll = lead_lag(&a, &b, &grid).unwrap();
        assert!(
            (ll.theta + 0.04).abs() < 0.011,
            "theta = {} contrast={:?}",
            ll.theta,
            ll.contrast
        );
    }
}
