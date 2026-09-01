//! Optimal stopping: regression bases, Longstaff–Schwartz, dual bounds, swing.

use amatsuki::Rng;
use faer::{Col, Mat};

use crate::error::{Error, Result};
use crate::finance::black_scholes::{put as bs_put, BlackScholesMarket};
use crate::finance::monte_carlo::{MonteCarloEstimate, OnlineMoments};
use crate::linalg::{col_from_slice, mat_from_row_slice, qr_least_squares};
use crate::models::GeometricBrownianMotion;
use crate::sampling::Sampling;
use crate::simulate::{simulate, SimConfig};

/// Feature map for conditional-expectation regression.
pub trait Basis: Send + Sync {
    fn dimension(&self, state_dimension: usize) -> usize;
    fn evaluate(&self, state: &[f64], output: &mut [f64]) -> Result<()>;
}

/// Conditional expectation `E[Y | X=x]` via a fitted basis.
pub trait ConditionalExpectation {
    fn fit(&mut self, states: &[Vec<f64>], target: &[f64]) -> Result<()>;
    fn predict(&self, state: &[f64]) -> Result<f64>;
}

/// Polynomials `1, x, x², …` on the first coordinate (optionally Laguerre).
#[derive(Clone, Copy, Debug)]
pub struct PolynomialBasis {
    pub degree: usize,
    pub laguerre: bool,
}

impl PolynomialBasis {
    pub fn new(degree: usize) -> Self {
        Self {
            degree,
            laguerre: false,
        }
    }
    pub fn laguerre(degree: usize) -> Self {
        Self {
            degree,
            laguerre: true,
        }
    }
}

impl Basis for PolynomialBasis {
    fn dimension(&self, _state_dimension: usize) -> usize {
        self.degree + 1
    }
    fn evaluate(&self, state: &[f64], output: &mut [f64]) -> Result<()> {
        if state.is_empty() || output.len() != self.degree + 1 {
            return Err(Error::dim("polynomial basis dimension"));
        }
        let x = state[0];
        if self.laguerre {
            // Weighted Laguerre: L0=1, L1=1-x, L_{k+1} = ((2k+1-x)L_k - k L_{k-1})/(k+1)
            output[0] = 1.0;
            if self.degree >= 1 {
                output[1] = 1.0 - x;
            }
            for k in 1..self.degree {
                let kk = k as f64;
                output[k + 1] =
                    ((2.0 * kk + 1.0 - x) * output[k] - kk * output[k - 1]) / (kk + 1.0);
            }
            for v in output.iter_mut() {
                *v *= (-x * 0.5).exp();
            }
        } else {
            let mut p = 1.0;
            for v in output.iter_mut() {
                *v = p;
                p *= x;
            }
        }
        Ok(())
    }
}

/// Probabilists' Hermite polynomials `He_0=1`, `He_1=x`,
/// `He_{n+1} = x He_n − n He_{n−1}` (Longstaff–Schwartz / Shreve I regression).
#[derive(Clone, Copy, Debug)]
pub struct HermiteBasis {
    pub degree: usize,
}

impl HermiteBasis {
    pub fn new(degree: usize) -> Self {
        Self { degree }
    }
}

impl Basis for HermiteBasis {
    fn dimension(&self, _state_dimension: usize) -> usize {
        self.degree + 1
    }
    fn evaluate(&self, state: &[f64], output: &mut [f64]) -> Result<()> {
        if state.is_empty() || output.len() != self.degree + 1 {
            return Err(Error::dim("Hermite basis dimension"));
        }
        let x = state[0];
        output[0] = 1.0;
        if self.degree >= 1 {
            output[1] = x;
        }
        for n in 1..self.degree {
            let nn = n as f64;
            output[n + 1] = x * output[n] - nn * output[n - 1];
        }
        Ok(())
    }
}

/// Linear regression `Y ≈ φ(X)ᵀβ` with column standardization.
#[derive(Clone, Debug)]
pub struct LinearCe<B> {
    pub basis: B,
    pub beta: Vec<f64>,
    pub mean: Vec<f64>,
    pub std: Vec<f64>,
}

impl<B: Basis> LinearCe<B> {
    pub fn new(basis: B) -> Self {
        Self {
            basis,
            beta: Vec::new(),
            mean: Vec::new(),
            std: Vec::new(),
        }
    }
}

impl<B: Basis> ConditionalExpectation for LinearCe<B> {
    fn fit(&mut self, states: &[Vec<f64>], target: &[f64]) -> Result<()> {
        if states.len() != target.len() || states.is_empty() {
            return Err(Error::dim("conditional expectation fit size"));
        }
        let dim = self.basis.dimension(states[0].len());
        let mut rows = Vec::with_capacity(states.len() * dim);
        let mut feat = vec![0.0; dim];
        for s in states {
            self.basis.evaluate(s, &mut feat)?;
            rows.extend_from_slice(&feat);
        }
        let (mean, std) = column_moments(&rows, states.len(), dim);
        standardize_rows(&mut rows, states.len(), dim, &mean, &std);
        let a = mat_from_row_slice(states.len(), dim, &rows);
        let b = col_from_slice(target);
        let fit = qr_least_squares(&a, &b, Some(1e-8))?;
        self.beta = (0..dim).map(|j| fit.beta[j]).collect();
        self.mean = mean;
        self.std = std;
        Ok(())
    }

    fn predict(&self, state: &[f64]) -> Result<f64> {
        if self.beta.is_empty() {
            return Err(Error::numeric("conditional expectation is not fitted"));
        }
        let dim = self.beta.len();
        let mut feat = vec![0.0; dim];
        self.basis.evaluate(state, &mut feat)?;
        let mut y = 0.0;
        for j in 0..dim {
            y += self.beta[j] * apply_scale(feat[j], self.mean[j], self.std[j]);
        }
        Ok(y)
    }
}

fn column_moments(rows: &[f64], n: usize, dim: usize) -> (Vec<f64>, Vec<f64>) {
    let mut mean = vec![0.0; dim];
    for i in 0..n {
        for j in 0..dim {
            mean[j] += rows[i * dim + j];
        }
    }
    let nf = n as f64;
    for m in &mut mean {
        *m /= nf;
    }
    let mut var = vec![0.0; dim];
    for i in 0..n {
        for j in 0..dim {
            let d = rows[i * dim + j] - mean[j];
            var[j] += d * d;
        }
    }
    let std = var.into_iter().map(|v| (v / nf.max(1.0)).sqrt()).collect();
    (mean, std)
}

fn apply_scale(x: f64, mean: f64, std: f64) -> f64 {
    // Leave (near-)constant columns alone so the intercept stays a column of ones.
    if std > 1e-12 {
        (x - mean) / std
    } else {
        x
    }
}

fn standardize_rows(rows: &mut [f64], n: usize, dim: usize, mean: &[f64], std: &[f64]) {
    for i in 0..n {
        for j in 0..dim {
            rows[i * dim + j] = apply_scale(rows[i * dim + j], mean[j], std[j]);
        }
    }
}

/// Fitted continuation policy at one exercise date.
#[derive(Clone, Debug)]
pub struct RegressionPolicy {
    pub time: f64,
    pub beta: Vec<f64>,
    pub mean: Vec<f64>,
    pub std: Vec<f64>,
    pub in_the_money: usize,
}

/// Longstaff–Schwartz output.
#[derive(Clone, Debug)]
pub struct OptimalStoppingResult {
    pub lower_bound: MonteCarloEstimate,
    pub upper_bound: Option<MonteCarloEstimate>,
    pub exercise_probabilities: Vec<f64>,
    pub policies: Vec<RegressionPolicy>,
}

#[derive(Clone, Debug)]
pub struct LongstaffSchwartzConfig<B> {
    pub basis: B,
    pub in_the_money_only: bool,
    pub min_regression_rows: usize,
    pub standardize: bool,
}

impl Default for LongstaffSchwartzConfig<PolynomialBasis> {
    fn default() -> Self {
        Self {
            // Weighted Laguerre on S/K under-exercises the LS Table 1 put.
            // Plain polynomials (degree 2) match CRR / Longstaff–Schwartz.
            basis: PolynomialBasis::new(2),
            in_the_money_only: true,
            min_regression_rows: 8,
            standardize: true,
        }
    }
}

impl LongstaffSchwartzConfig<HermiteBasis> {
    pub fn hermite(degree: usize) -> Self {
        Self {
            basis: HermiteBasis::new(degree),
            in_the_money_only: true,
            min_regression_rows: 8,
            standardize: true,
        }
    }
}

/// American put on GBM via Longstaff–Schwartz.
pub fn lsm_american_put<B, R>(
    spot: f64,
    strike: f64,
    rate: f64,
    vol: f64,
    time: f64,
    n_steps: usize,
    n_train: usize,
    n_eval: usize,
    cfg: &LongstaffSchwartzConfig<B>,
    sim: &SimConfig,
    rng: &mut R,
) -> Result<OptimalStoppingResult>
where
    B: Basis,
    R: Rng + ?Sized,
{
    if n_steps == 0 || n_train == 0 || n_eval == 0 {
        return Err(Error::param("LSM needs positive path / step counts"));
    }
    let model = GeometricBrownianMotion::new(rate, vol)?;
    let samp = Sampling::from_terminal(time, n_steps)?;
    let dt = time / n_steps as f64;
    let df = (-rate * dt).exp();
    let mut train = Vec::with_capacity(n_train);
    for _ in 0..n_train {
        train.push(simulate(&model, &samp, &[spot], rng, sim)?);
    }
    let dim = cfg.basis.dimension(1);
    let mut cash = vec![0.0; n_train];
    for (i, p) in train.iter().enumerate() {
        cash[i] = (strike - p.terminal()[0]).max(0.0);
    }
    let mut policies = Vec::new();
    let mut exercise_n = vec![0usize; n_steps];
    for step in (1..n_steps).rev() {
        let t = samp.times()[step];
        let mut rows = Vec::new();
        let mut rhs = Vec::new();
        let mut idx = Vec::new();
        let mut feat = vec![0.0; dim];
        for (i, p) in train.iter().enumerate() {
            let s = p.state(step)[0];
            let itm = s < strike;
            if cfg.in_the_money_only && !itm {
                continue;
            }
            cfg.basis.evaluate(&[s / strike], &mut feat)?;
            rows.extend_from_slice(&feat);
            rhs.push(cash[i] * df);
            idx.push(i);
        }
        if idx.len() < cfg.min_regression_rows {
            for c in &mut cash {
                *c *= df;
            }
            continue;
        }
        let (mean, std) = if cfg.standardize {
            let m = column_moments(&rows, idx.len(), dim);
            standardize_rows(&mut rows, idx.len(), dim, &m.0, &m.1);
            m
        } else {
            (vec![0.0; dim], vec![1.0; dim])
        };
        let a = mat_from_row_slice(idx.len(), dim, &rows);
        let b = col_from_slice(&rhs);
        let fit = qr_least_squares(&a, &b, Some(1e-8))?;
        let beta = (0..dim).map(|j| fit.beta[j]).collect::<Vec<_>>();
        let mut exercised = vec![false; n_train];
        for &i in &idx {
            let s = train[i].state(step)[0];
            cfg.basis.evaluate(&[s / strike], &mut feat)?;
            let mut cont = 0.0;
            for j in 0..dim {
                cont += beta[j] * apply_scale(feat[j], mean[j], std[j]);
            }
            let ex = (strike - s).max(0.0);
            if ex > cont && ex > 0.0 {
                cash[i] = ex;
                exercised[i] = true;
                exercise_n[step] += 1;
            }
        }
        // One discount for every path that did not exercise. OTM paths must
        // not be discounted in the collection loop (that double-counted).
        for (i, c) in cash.iter_mut().enumerate() {
            if !exercised[i] {
                *c *= df;
            }
        }
        policies.push(RegressionPolicy {
            time: t,
            beta,
            mean,
            std,
            in_the_money: idx.len(),
        });
    }
    let mut acc2 = OnlineMoments::default();
    for _ in 0..n_eval {
        let p = simulate(&model, &samp, &[spot], rng, sim)?;
        acc2.push(lsm_apply_policy(
            &p, strike, rate, &policies, &cfg.basis, dim,
        )?);
    }
    let probs: Vec<f64> = exercise_n
        .iter()
        .map(|&c| c as f64 / n_train as f64)
        .collect();
    let _ = Mat::<f64>::zeros(0, 0);
    let _ = Col::<f64>::zeros(0);
    Ok(OptimalStoppingResult {
        lower_bound: MonteCarloEstimate::from_moments(&acc2, 1.96),
        upper_bound: None,
        exercise_probabilities: probs,
        policies,
    })
}

fn lsm_apply_policy<B: Basis>(
    path: &crate::path::Path,
    strike: f64,
    rate: f64,
    policies: &[RegressionPolicy],
    basis: &B,
    dim: usize,
) -> Result<f64> {
    let times = path.times();
    let mut tau = times.len() - 1;
    let mut payoff = (strike - path.terminal()[0]).max(0.0);
    for step in 1..times.len() - 1 {
        let s = path.state(step)[0];
        let ex = (strike - s).max(0.0);
        if ex <= 0.0 {
            continue;
        }
        if let Some(pol) = policies
            .iter()
            .find(|q| (q.time - times[step]).abs() < 1e-12)
        {
            let mut feat = vec![0.0; dim];
            basis.evaluate(&[s / strike], &mut feat)?;
            let mut cont = 0.0;
            for j in 0..dim {
                cont += pol.beta[j] * apply_scale(feat[j], pol.mean[j], pol.std[j]);
            }
            if ex > cont {
                tau = step;
                payoff = ex;
                break;
            }
        }
    }
    Ok((-rate * times[tau]).exp() * payoff)
}

/// Haugh–Kogan dual upper bound using the discounted European put as the
/// martingale (`M_k = e^{-r t_k} P_{\mathrm{EU}}(S_k, T-t_k) − P_{\mathrm{EU}}(S_0,T)`).
pub fn european_put_dual_upper<R: Rng + ?Sized>(
    spot: f64,
    strike: f64,
    rate: f64,
    vol: f64,
    time: f64,
    n_steps: usize,
    n_paths: usize,
    sim: &SimConfig,
    rng: &mut R,
) -> Result<MonteCarloEstimate> {
    if n_steps == 0 || n_paths == 0 {
        return Err(Error::param("dual upper bound needs paths and steps"));
    }
    let model = GeometricBrownianMotion::new(rate, vol)?;
    let samp = Sampling::from_terminal(time, n_steps)?;
    let p0 = bs_put(
        &BlackScholesMarket::new(spot, rate, 0.0, vol, time)?,
        strike,
    )?
    .price;
    let mut acc = OnlineMoments::default();
    for _ in 0..n_paths {
        let path = simulate(&model, &samp, &[spot], rng, sim)?;
        let mut best = (strike - spot).max(0.0);
        for k in 0..path.n_nodes() {
            let t = path.times()[k];
            let s = path.state(k)[0];
            if !(s > 0.0 && s.is_finite()) {
                continue;
            }
            let tau = (time - t).max(0.0);
            let zk = (-rate * t).exp() * (strike - s).max(0.0);
            let pk = if tau <= 1e-14 {
                zk
            } else {
                (-rate * t).exp()
                    * bs_put(&BlackScholesMarket::new(s, rate, 0.0, vol, tau)?, strike)?.price
            };
            best = best.max(zk - pk + p0);
        }
        acc.push(best);
    }
    Ok(MonteCarloEstimate::from_moments(&acc, 1.96))
}

/// Andersen–Broadie dual upper bound with nested simulations (one-level).
pub fn andersen_broadie_put<R: Rng + ?Sized>(
    spot: f64,
    strike: f64,
    rate: f64,
    vol: f64,
    time: f64,
    n_steps: usize,
    n_outer: usize,
    n_inner: usize,
    sim: &SimConfig,
    rng: &mut R,
) -> Result<MonteCarloEstimate>
where
{
    let model = GeometricBrownianMotion::new(rate, vol)?;
    let samp = Sampling::from_terminal(time, n_steps)?;
    let mut acc = OnlineMoments::default();
    for _ in 0..n_outer {
        let path = simulate(&model, &samp, &[spot], rng, sim)?;
        let mut mart = 0.0;
        let mut maxv = (strike - path.initial()[0]).max(0.0);
        for k in 1..path.n_nodes() {
            let t = path.times()[k];
            let s = path.state(k)[0];
            let ex = (-rate * t).exp() * (strike - s).max(0.0);
            let rem = time - t;
            let mut cont = 0.0;
            if rem > 1e-12 && k + 1 < path.n_nodes() {
                let sub = Sampling::from_terminal(rem, (n_steps - k).max(1))?;
                for _ in 0..n_inner {
                    let q = simulate(&model, &sub, &[s], rng, sim)?;
                    let tau = q.n_nodes() - 1;
                    let pay = (strike - q.terminal()[0]).max(0.0);
                    cont += (-rate * q.times()[tau]).exp() * pay;
                }
                cont /= n_inner as f64;
                cont *= (-rate * t).exp();
            }
            let v = ex.max(cont);
            mart += v - cont;
            maxv = maxv.max(ex - mart);
        }
        acc.push(maxv);
    }
    Ok(MonteCarloEstimate::from_moments(&acc, 1.96))
}

/// Swing call / put on a CRR tree: at most one exercise per date, at most
/// `n_rights` exercises in total (Jaillet–Ronn–Tompaidis multiple stopping).
pub fn crr_swing(
    spot: f64,
    strike: f64,
    rate: f64,
    vol: f64,
    time: f64,
    n_steps: usize,
    n_rights: usize,
    is_call: bool,
) -> Result<f64> {
    if n_steps == 0 || n_rights == 0 || !(spot > 0.0 && strike >= 0.0 && vol >= 0.0 && time > 0.0) {
        return Err(Error::param("swing inputs invalid"));
    }
    let dt = time / n_steps as f64;
    let u = (vol * dt.sqrt()).exp();
    let d = 1.0 / u;
    let growth = (rate * dt).exp();
    if d > growth + 1e-14 || growth > u + 1e-14 {
        return Err(Error::numeric("swing CRR parameters admit arbitrage"));
    }
    let p = (growth - d) / (u - d);
    let qmax = n_rights;
    // val[j][q] at the current time layer (j = number of up moves).
    let mut val = vec![vec![0.0; qmax + 1]; n_steps + 1];
    for j in 0..=n_steps {
        let s = spot * u.powi(j as i32) * d.powi((n_steps - j) as i32);
        let ex = if is_call {
            (s - strike).max(0.0)
        } else {
            (strike - s).max(0.0)
        };
        for q in 1..=qmax {
            val[j][q] = ex;
        }
    }
    for step in (0..n_steps).rev() {
        let mut nxt = vec![vec![0.0; qmax + 1]; step + 1];
        for j in 0..=step {
            let s = spot * u.powi(j as i32) * d.powi((step - j) as i32);
            let ex = if is_call {
                (s - strike).max(0.0)
            } else {
                (strike - s).max(0.0)
            };
            for q in 1..=qmax {
                let hold = (p * val[j + 1][q] + (1.0 - p) * val[j][q]) / growth;
                let after = (p * val[j + 1][q - 1] + (1.0 - p) * val[j][q - 1]) / growth;
                nxt[j][q] = hold.max(ex + after);
            }
        }
        val = nxt;
    }
    Ok(val[0][qmax])
}

/// Commodity storage on a CRR tree. Inventory `q ∈ {0,…,q_max}`; at each
/// node the operator injects one unit (`−S`), withdraws one unit (`+S`), or
/// holds. Residual inventory is sold at expiry.
pub fn crr_storage(
    spot: f64,
    rate: f64,
    vol: f64,
    time: f64,
    n_steps: usize,
    q_max: usize,
) -> Result<f64> {
    if n_steps == 0 || !(spot > 0.0 && vol >= 0.0 && time > 0.0) {
        return Err(Error::param("storage inputs invalid"));
    }
    if q_max == 0 {
        return Ok(0.0);
    }
    let dt = time / n_steps as f64;
    let u = (vol * dt.sqrt()).exp();
    let d = 1.0 / u;
    let growth = (rate * dt).exp();
    if d > growth + 1e-14 || growth > u + 1e-14 {
        return Err(Error::numeric("storage CRR parameters admit arbitrage"));
    }
    let p = (growth - d) / (u - d);
    let mut val = vec![vec![0.0; q_max + 1]; n_steps + 1];
    for j in 0..=n_steps {
        let s = spot * u.powi(j as i32) * d.powi((n_steps - j) as i32);
        for q in 0..=q_max {
            val[j][q] = q as f64 * s;
        }
    }
    for step in (0..n_steps).rev() {
        let mut nxt = vec![vec![0.0; q_max + 1]; step + 1];
        for j in 0..=step {
            let s = spot * u.powi(j as i32) * d.powi((step - j) as i32);
            for q in 0..=q_max {
                let hold = (p * val[j + 1][q] + (1.0 - p) * val[j][q]) / growth;
                let mut best = hold;
                if q < q_max {
                    let inj = -s + (p * val[j + 1][q + 1] + (1.0 - p) * val[j][q + 1]) / growth;
                    best = best.max(inj);
                }
                if q > 0 {
                    let wd = s + (p * val[j + 1][q - 1] + (1.0 - p) * val[j][q - 1]) / growth;
                    best = best.max(wd);
                }
                nxt[j][q] = best;
            }
        }
        val = nxt;
    }
    Ok(val[0][0])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::seed_rng;
    use crate::scheme::Scheme;

    #[test]
    fn hermite_recurrence() {
        let h = HermiteBasis::new(3);
        let mut out = [0.0; 4];
        h.evaluate(&[2.0], &mut out).unwrap();
        assert!((out[0] - 1.0).abs() < 1e-14);
        assert!((out[1] - 2.0).abs() < 1e-14);
        assert!((out[2] - 3.0).abs() < 1e-14); // x² − 1 = 3
        assert!((out[3] - 2.0).abs() < 1e-14); // x³ − 3x = 2
    }

    #[test]
    fn linear_ce_recovers_line() {
        let mut ce = LinearCe::new(PolynomialBasis::new(1));
        let xs: Vec<Vec<f64>> = (0..20).map(|i| vec![i as f64]).collect();
        let ys: Vec<f64> = (0..20).map(|i| 3.0 * i as f64 + 1.0).collect();
        ce.fit(&xs, &ys).unwrap();
        assert!((ce.predict(&[4.0]).unwrap() - 13.0).abs() < 1e-8);
    }

    #[test]
    fn lsm_put_above_european() {
        let mut rng = seed_rng(5);
        let cfg = LongstaffSchwartzConfig::<PolynomialBasis>::default();
        let sim = SimConfig {
            scheme: Scheme::Exact,
            ..SimConfig::default()
        };
        let r = lsm_american_put(
            36.0, 40.0, 0.06, 0.2, 1.0, 50, 3_000, 3_000, &cfg, &sim, &mut rng,
        )
        .unwrap();
        // Longstaff–Schwartz Table 1 ≈ 4.478.
        let euro = crate::finance::black_scholes::put(
            &crate::finance::black_scholes::BlackScholesMarket::new(36.0, 0.06, 0.0, 0.2, 1.0)
                .unwrap(),
            40.0,
        )
        .unwrap()
        .price;
        assert!(
            r.lower_bound.estimate + 0.05 >= euro,
            "LSM {} euro {euro}",
            r.lower_bound.estimate
        );
        let tree = crate::finance::tree::crr_price(36.0, 40.0, 0.06, 0.2, 1.0, 50, false, true)
            .unwrap()
            .price;
        assert!(
            (r.lower_bound.estimate - 4.478).abs() < 0.25,
            "LSM {} tree {tree} euro {euro} vs LS Table 1 4.478",
            r.lower_bound.estimate
        );
        let dual =
            european_put_dual_upper(36.0, 40.0, 0.06, 0.2, 1.0, 25, 1_200, &sim, &mut rng).unwrap();
        assert!(
            r.lower_bound.estimate <= dual.estimate + 3.0 * dual.standard_error + 0.2,
            "LSM {} dual {}",
            r.lower_bound.estimate,
            dual.estimate
        );
    }

    #[test]
    fn swing_monotone_in_rights() {
        let one = crr_swing(36.0, 40.0, 0.06, 0.2, 1.0, 40, 1, false).unwrap();
        let two = crr_swing(36.0, 40.0, 0.06, 0.2, 1.0, 40, 2, false).unwrap();
        let three = crr_swing(36.0, 40.0, 0.06, 0.2, 1.0, 40, 3, false).unwrap();
        assert!(two + 1e-12 >= one, "swing 2 {two} < 1 {one}");
        assert!(three + 1e-12 >= two, "swing 3 {three} < 2 {two}");
        let amer = crate::finance::tree::crr_price(36.0, 40.0, 0.06, 0.2, 1.0, 40, false, true)
            .unwrap()
            .price;
        assert!(
            (one - amer).abs() < 1e-10,
            "1-right swing {one} vs Am {amer}"
        );
    }

    #[test]
    fn storage_monotone_in_qmax() {
        let z = crr_storage(50.0, 0.05, 0.25, 1.0, 24, 0).unwrap();
        assert!(z.abs() < 1e-14);
        let a = crr_storage(50.0, 0.05, 0.25, 1.0, 24, 1).unwrap();
        let b = crr_storage(50.0, 0.05, 0.25, 1.0, 24, 2).unwrap();
        assert!(a >= 0.0 && b + 1e-12 >= a, "storage {a} → {b}");
    }
}
