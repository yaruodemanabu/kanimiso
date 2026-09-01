//! High-frequency trading / microstructure tools (beyond YUIMA `cce`).
//!
//! Pre-averaged Hayashi–Yoshida, realized kernels, two-scale RV, Roll
//! implied spread, ACD durations, Almgren–Chriss schedules, Kyle lambda,
//! and a simple two-sided Hawkes limit-order book.

use faer::Mat;

use crate::error::{Error, Result};
use crate::highfreq::{hayashi_yoshida, realized_variance};
use crate::models::point_more::MultivariateHawkes2;
use crate::optimize::{nelder_mead, OptOptions};
use crate::path::{Path, TickSeries};

/// Previous-tick interpolation of an irregular series onto a regular grid.
pub fn previous_tick(series: &TickSeries, grid: &[f64]) -> Result<Vec<f64>> {
    if grid.is_empty() {
        return Err(Error::sampling("empty grid"));
    }
    let mut j = 0;
    let mut out = Vec::with_capacity(grid.len());
    for &t in grid {
        while j + 1 < series.n() && series.times[j + 1] <= t {
            j += 1;
        }
        if series.times[j] > t {
            return Err(Error::sampling("grid starts before the first tick"));
        }
        out.push(series.values[j]);
    }
    Ok(out)
}

/// Intersection of the two clocks (exact timestamp matches only).
///
/// This is **not** the BNHLS refresh-time grid; see [`refresh_times`].
pub fn intersection_times(
    x: &TickSeries,
    y: &TickSeries,
) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    let mut i = 0;
    let mut j = 0;
    let mut t = Vec::new();
    let mut xv = Vec::new();
    let mut yv = Vec::new();
    while i < x.n() && j < y.n() {
        if x.times[i] == y.times[j] {
            t.push(x.times[i]);
            xv.push(x.values[i]);
            yv.push(y.values[j]);
            i += 1;
            j += 1;
        } else if x.times[i] < y.times[j] {
            i += 1;
        } else {
            j += 1;
        }
    }
    if t.len() < 2 {
        return Err(Error::sampling("intersection clock has < 2 points"));
    }
    Ok((t, xv, yv))
}

/// BNHLS refresh-time synchronization.
///
/// `τ₁ = max(tˣ₁, tʸ₁)` and
/// `τ_{k+1} = max(tˣ_{Nˣ(τ_k)+1}, tʸ_{Nʸ(τ_k)+1})`.
/// Values are previous-tick interpolations at each refresh instant.
/// Asynchronous clocks with no common stamps still produce a grid.
pub fn refresh_times(x: &TickSeries, y: &TickSeries) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    if x.n() == 0 || y.n() == 0 {
        return Err(Error::sampling("refresh clock needs nonempty series"));
    }
    let mut ix = 0usize;
    let mut iy = 0usize;
    let mut t = Vec::new();
    let mut xv = Vec::new();
    let mut yv = Vec::new();
    let mut tau = x.times[0].max(y.times[0]);
    loop {
        while ix + 1 < x.n() && x.times[ix + 1] <= tau {
            ix += 1;
        }
        while iy + 1 < y.n() && y.times[iy + 1] <= tau {
            iy += 1;
        }
        if x.times[ix] > tau || y.times[iy] > tau {
            break;
        }
        t.push(tau);
        xv.push(x.values[ix]);
        yv.push(y.values[iy]);
        while ix < x.n() && x.times[ix] <= tau {
            ix += 1;
        }
        while iy < y.n() && y.times[iy] <= tau {
            iy += 1;
        }
        if ix >= x.n() || iy >= y.n() {
            break;
        }
        tau = x.times[ix].max(y.times[iy]);
    }
    if t.len() < 2 {
        return Err(Error::sampling("refresh clock has < 2 points"));
    }
    Ok((t, xv, yv))
}

/// Jacod–Li–Mykland–Podolskij–Vetter pre-averaging of returns.
pub fn preaverage(dx: &[f64], kn: usize) -> Result<Vec<f64>> {
    if kn < 2 || kn >= dx.len() {
        return Err(Error::param("pre-average window kn invalid"));
    }
    let mut out = Vec::with_capacity(dx.len().saturating_sub(kn));
    for i in 0..dx.len().saturating_sub(kn) {
        let mut s = 0.0;
        for j in 1..kn {
            let w = (j as f64 / kn as f64) - 0.5; // g(x) = x ∧ (1−x) equivalent slope
            let g = if (j as f64) <= kn as f64 / 2.0 {
                j as f64 / kn as f64
            } else {
                1.0 - j as f64 / kn as f64
            };
            let _ = w;
            s += g * dx[i + j];
        }
        out.push(s);
    }
    Ok(out)
}

/// Index-aligned pre-average covariance (ignores timestamps).
///
/// Not Christensen–Kinnebrock–Podolskij PHY; see [`preaveraged_hy`].
pub fn indexed_preaverage_cov(x: &TickSeries, y: &TickSeries, kn: usize) -> Result<f64> {
    let dx = x.increments();
    let dy = y.increments();
    let px = preaverage(&dx, kn)?;
    let py = preaverage(&dy, kn)?;
    let n = px.len().min(py.len());
    if n == 0 {
        return Err(Error::infer("pre-average produced no terms"));
    }
    let psi = 1.0 / 12.0;
    let knf = kn as f64;
    let mut s = 0.0;
    for i in 0..n {
        s += px[i] * py[i];
    }
    Ok(s / (psi * knf))
}

/// Pre-averaged Hayashi–Yoshida on the BNHLS refresh clock.
///
/// Synchronize with [`refresh_times`], then pre-average the refresh-grid
/// returns (Christensen–Kinnebrock–Podolskij / Jacod et al.).
pub fn preaveraged_hy(x: &TickSeries, y: &TickSeries, kn: usize) -> Result<f64> {
    let (t, xv, yv) = refresh_times(x, y)?;
    let xs = TickSeries {
        times: t.clone(),
        values: xv,
    };
    let ys = TickSeries {
        times: t,
        values: yv,
    };
    indexed_preaverage_cov(&xs, &ys, kn)
}

/// Realized kernel with Tukey–Hanning weights (Barndorff-Nielsen–Hansen–Lunde–Shephard).
pub fn realized_kernel(path: &Path, j: usize, bandwidth: usize) -> Result<f64> {
    let dx = path.increments(j)?;
    let n = dx.len();
    if bandwidth == 0 || bandwidth >= n {
        return Err(Error::param("kernel bandwidth invalid"));
    }
    let mut gamma0 = 0.0;
    for d in &dx {
        gamma0 += d * d;
    }
    let mut rk = gamma0;
    for h in 1..=bandwidth {
        let mut g = 0.0;
        for i in h..n {
            g += dx[i] * dx[i - h];
        }
        let x = h as f64 / (bandwidth as f64 + 1.0);
        let w = 0.5 * (1.0 + (std::f64::consts::PI * x).cos()); // Tukey-Hanning
        rk += 2.0 * w * g;
    }
    Ok(rk.max(0.0))
}

/// Zhang–Mykland–Aït-Sahalia two-scale realized volatility.
pub fn two_scale_rv(path: &Path, j: usize, k: usize) -> Result<f64> {
    let dx = path.increments(j)?;
    let n = dx.len();
    if k < 2 || k >= n {
        return Err(Error::param("two-scale K invalid"));
    }
    let mut rv1 = 0.0;
    for d in &dx {
        rv1 += d * d;
    }
    let mut rvk = 0.0;
    for start in 0..k {
        let mut s = 0.0;
        let mut i = start;
        while i + k < n {
            let mut inc = 0.0;
            for d in dx.iter().skip(i).take(k) {
                inc += *d;
            }
            s += inc * inc;
            i += k;
        }
        rvk += s;
    }
    rvk /= k as f64;
    let n = n as f64;
    let kf = k as f64;
    Ok((rvk - (n / kf) * rv1 / n).max(0.0))
}

/// Roll (1984) implied spread from first-order return autocovariance: `2 √(−γ₁)`.
pub fn roll_spread(path: &Path, j: usize) -> Result<f64> {
    let dx = path.increments(j)?;
    if dx.len() < 3 {
        return Err(Error::infer("Roll needs ≥ 3 increments"));
    }
    let m = dx.iter().sum::<f64>() / dx.len() as f64;
    let mut g1 = 0.0;
    for i in 1..dx.len() {
        g1 += (dx[i] - m) * (dx[i - 1] - m);
    }
    g1 /= (dx.len() - 1) as f64;
    Ok(2.0 * (-g1).max(0.0).sqrt())
}

/// Engle–Russell ACD(1,1): `ψᵢ = ω + α x_{i−1} + β ψ_{i−1}`, `xᵢ = ψᵢ εᵢ`, `ε~Exp(1)`.
#[derive(Clone, Debug)]
pub struct Acd11 {
    pub omega: f64,
    pub alpha: f64,
    pub beta: f64,
}

impl Acd11 {
    pub fn new(omega: f64, alpha: f64, beta: f64) -> Result<Self> {
        if omega <= 0.0 || alpha < 0.0 || beta < 0.0 || alpha + beta >= 1.0 {
            return Err(Error::param("ACD(1,1) needs ω>0, α,β≥0, α+β<1"));
        }
        Ok(Self { omega, alpha, beta })
    }

    pub fn loglik(&self, durations: &[f64]) -> Result<f64> {
        if durations.is_empty() {
            return Err(Error::infer("no durations"));
        }
        let mut psi = durations.iter().sum::<f64>() / durations.len() as f64;
        let mut ll = 0.0;
        for &x in durations {
            if psi <= 0.0 || x <= 0.0 {
                return Ok(f64::NEG_INFINITY);
            }
            ll += -psi.ln() - x / psi;
            psi = self.omega + self.alpha * x + self.beta * psi;
        }
        Ok(ll)
    }

    pub fn mle(durations: &[f64], start: [f64; 3]) -> Result<(Self, f64)> {
        let obj = |p: &[f64]| match Self::new(p[0], p[1], p[2]).and_then(|m| m.loglik(durations)) {
            Ok(ll) if ll.is_finite() => -ll,
            _ => 1e16,
        };
        let o = nelder_mead(&obj, &start, None, None, OptOptions::default())?;
        let m = Self::new(o.x[0], o.x[1], o.x[2])?;
        let ll = m.loglik(durations)?;
        Ok((m, ll))
    }
}

/// Durations from a strictly increasing tick clock.
pub fn durations(times: &[f64]) -> Result<Vec<f64>> {
    if times.len() < 2 {
        return Err(Error::sampling("need ≥ 2 times"));
    }
    Ok(times.windows(2).map(|w| w[1] - w[0]).collect())
}

/// Almgren–Chriss optimal schedule for selling `X` shares in `N` slices
/// with temporary impact `η` and permanent impact `γ`, risk aversion `λ`,
/// and variance `σ²`.
#[derive(Clone, Debug)]
pub struct AlmgrenChriss {
    pub x0: f64,
    pub n: usize,
    pub tau: f64,
    pub sigma: f64,
    pub eta: f64,
    pub gamma: f64,
    pub lambda: f64,
}

impl AlmgrenChriss {
    pub fn schedule(&self) -> Result<(Vec<f64>, Vec<f64>)> {
        if self.n == 0 || self.tau <= 0.0 {
            return Err(Error::param("invalid AC horizon"));
        }
        let kappa = ((self.lambda * self.sigma * self.sigma) / self.eta)
            .max(0.0)
            .sqrt();
        let t_end = self.n as f64 * self.tau;
        let mut hold = Vec::with_capacity(self.n + 1);
        let mut trade = Vec::with_capacity(self.n);
        for i in 0..=self.n {
            let t = i as f64 * self.tau;
            let rem = if kappa < 1e-12 {
                self.x0 * (1.0 - t / t_end)
            } else {
                self.x0 * ((kappa * (t_end - t)).sinh() / (kappa * t_end).sinh())
            };
            hold.push(rem);
        }
        for i in 0..self.n {
            trade.push(hold[i] - hold[i + 1]);
        }
        Ok((hold, trade))
    }
}

/// TWAP: equal slices.
pub fn twap(x0: f64, n: usize) -> Vec<f64> {
    vec![x0 / n as f64; n]
}

/// Kyle (1985) lambda: OLS of `ΔP` on signed volume.
pub fn kyle_lambda(price: &TickSeries, signed_volume: &[f64]) -> Result<f64> {
    if price.n() != signed_volume.len() + 1 && price.increments().len() != signed_volume.len() {
        return Err(Error::dim(
            "Kyle: volume length must match price increments",
        ));
    }
    let dp = price.increments();
    let n = dp.len().min(signed_volume.len()) as f64;
    if n < 3.0 {
        return Err(Error::infer("Kyle needs more observations"));
    }
    let mut sxx = 0.0;
    let mut sxy = 0.0;
    let mx = signed_volume.iter().take(dp.len()).sum::<f64>() / n;
    let my = dp.iter().sum::<f64>() / n;
    for i in 0..dp.len().min(signed_volume.len()) {
        sxx += (signed_volume[i] - mx) * (signed_volume[i] - mx);
        sxy += (signed_volume[i] - mx) * (dp[i] - my);
    }
    if sxx <= 0.0 {
        return Err(Error::numeric("Kyle regressor has no variance"));
    }
    Ok(sxy / sxx)
}

/// Two-sided Hawkes LOB: intensity of bid/ask market orders.
pub type HawkesLob = MultivariateHawkes2;

/// Mid-price path from bid/ask ticks (use last bid/ask on a union clock).
pub fn mid_price(bid: &TickSeries, ask: &TickSeries) -> Result<TickSeries> {
    let mut times = bid.times.clone();
    times.extend(ask.times.iter().copied());
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    times.dedup_by(|a, b| (*a - *b).abs() < 1e-15);
    let b = previous_tick(bid, &times)?;
    let a = previous_tick(ask, &times)?;
    let mid: Vec<f64> = b.iter().zip(a.iter()).map(|(x, y)| 0.5 * (x + y)).collect();
    TickSeries::new(times, mid)
}

/// Cont–Kukanov–Stoikov order-flow imbalance from L1 snapshots.
///
/// `e = 1_{b↑} q^b − 1_{b↓} q^b_{prev} − 1_{a↓} q^a + 1_{a↑} q^a_{prev}`
/// plus the usual same-price size deltas.
#[derive(Clone, Copy, Debug)]
pub struct LobSnapshot {
    pub bid: f64,
    pub ask: f64,
    pub bid_size: f64,
    pub ask_size: f64,
}

pub fn ofi(prev: LobSnapshot, next: LobSnapshot) -> f64 {
    let mut e = 0.0;
    e += if next.bid > prev.bid {
        next.bid_size
    } else if next.bid == prev.bid {
        next.bid_size - prev.bid_size
    } else {
        -prev.bid_size
    };
    e += if next.ask < prev.ask {
        -next.ask_size
    } else if next.ask == prev.ask {
        prev.ask_size - next.ask_size
    } else {
        prev.ask_size
    };
    e
}

/// Microprice `α ask + (1−α) bid` with `α = qb / (qb+qa)` (queue imbalance).
pub fn microprice(bid: f64, ask: f64, qb: f64, qa: f64) -> f64 {
    let den = (qb + qa).max(1e-12);
    let alpha = qb / den;
    alpha * ask + (1.0 - alpha) * bid
}

/// Synchronous realized variance wrapper (HFT alias).
pub fn tick_rv(path: &Path) -> Result<Mat<f64>> {
    let v = realized_variance(path, 0)?;
    Ok(crate::linalg::mat_from_row_slice(1, 1, &[v]))
}

/// Hayashi–Yoshida on two trade clocks (alias that documents the HFT use).
pub fn hy_trade_clocks(x: &TickSeries, y: &TickSeries) -> f64 {
    hayashi_yoshida(x, y)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn twap_splits() {
        assert_eq!(twap(10.0, 5), vec![2.0; 5]);
    }

    #[test]
    fn refresh_times_async_clocks() {
        let x = TickSeries::new(vec![0.0, 0.3, 0.8, 1.2], vec![0.0, 1.0, 2.0, 3.0]).unwrap();
        let y = TickSeries::new(vec![0.1, 0.5, 0.9, 1.4], vec![0.0, 10.0, 20.0, 30.0]).unwrap();
        let (t, xv, yv) = refresh_times(&x, &y).unwrap();
        assert!(t.len() >= 2);
        assert!((t[0] - 0.1).abs() < 1e-14);
        assert!((xv[0] - 0.0).abs() < 1e-14);
        assert!((yv[0] - 0.0).abs() < 1e-14);
        assert!(intersection_times(&x, &y).is_err());
    }

    #[test]
    fn ac_decreases_inventory() {
        let ac = AlmgrenChriss {
            x0: 1.0,
            n: 10,
            tau: 0.1,
            sigma: 0.2,
            eta: 0.01,
            gamma: 0.001,
            lambda: 1e-6,
        };
        let (h, _) = ac.schedule().unwrap();
        assert!(h[0] > h[h.len() - 1]);
        assert!(h[h.len() - 1].abs() < 1e-8);
    }
}
