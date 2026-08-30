//! Linear filters: Baxter–King band-pass and a local-linear-trend Kalman smoother.
//!
//! A series shorter than the BK lead/lag window cannot identify the cycle.
//! A local-linear trend with all state variances at zero is a perfect line
//! through the first two points and is recorded as unidentified.

use crate::context::FitCtx;
use crate::data::{Matrix, Vector};
use crate::linalg::chol_solve;
use crate::traits::FitSeries;
use crate::validate::inspect_xy;
use faer::Mat;
use ojizou_san::Session;
use signlred::{
    Issue, IssueCode, Meaninglessness, NumericalCompromise, Qualified, Result, Severity,
};

fn inspect_univariate(ctx: &mut FitCtx, y: &Vector) {
    let x = Matrix::from_vector(y);
    inspect_xy(&mut ctx.report, &x, Some(y), &ctx.policy);
}

/// Baxter–King band-pass filter (statsmodels `bkfilter`).
///
/// `low` / `high` are periods in samples (e.g. 6 and 32 for quarterly
/// business-cycle bounds). `k` is the lead/lag truncation.
pub fn bk_filter(
    y: &Vector,
    low: f64,
    high: f64,
    k: usize,
    session: &Session,
) -> Result<Qualified<Vector>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_univariate(&mut ctx, y);
    if !(low.is_finite() && high.is_finite()) || low <= 2.0 || high <= low {
        ctx.push(
            Issue::builder(IssueCode::InvalidWeight)
                .message(format!(
                    "BK periods must satisfy 2 < low < high (got {low}, {high})"
                ))
                .build(),
        );
    }
    let n = y.len();
    let kk = k.max(1);
    if n < 2 * kk + 1 {
        ctx.push(
            Issue::builder(IssueCode::WindowTooShort)
                .message(format!(
                    "Baxter–King K={kk} needs n≥{}; series has {n}",
                    2 * kk + 1
                ))
                .meaninglessness(Meaninglessness::vacuous(
                    "band-pass cycle",
                    "the symmetric MA is undefined at the ends when n < 2K+1",
                    "shorten K or lengthen the series",
                ))
                .build(),
        );
        return ctx.finish(Vector::zeros(n));
    }
    let w1 = 2.0 * std::f64::consts::PI / high.max(low + 1e-9);
    let w2 = 2.0 * std::f64::consts::PI / low.max(2.0 + 1e-9);
    let mut a = vec![0.0; kk + 1];
    a[0] = (w2 - w1) / std::f64::consts::PI;
    for h in 1..=kk {
        let hf = h as f64;
        a[h] = (hf * w2).sin() / (hf * std::f64::consts::PI)
            - (hf * w1).sin() / (hf * std::f64::consts::PI);
    }
    // Enforce ∑ a = 0 so a unit-root trend is annihilated.
    let mut s = a[0];
    for h in 1..=kk {
        s += 2.0 * a[h];
    }
    let adj = s / (2 * kk + 1) as f64;
    a[0] -= adj;
    for h in 1..=kk {
        a[h] -= adj;
    }
    let mut cycle = Vector::zeros(n);
    for t in kk..n - kk {
        let mut v = a[0] * y[t];
        for h in 1..=kk {
            v += a[h] * (y[t + h] + y[t - h]);
        }
        cycle[t] = v;
    }
    ctx.push(
        Issue::builder(IssueCode::SpectralLeakage)
            .severity(Severity::Advisory)
            .message(format!(
                "Baxter–King truncates the ideal band-pass at K={kk}; ends are set to 0"
            ))
            .compromise(NumericalCompromise::new(
                "two-sided ideal band-pass",
                "symmetric MA of length 2K+1 with zero-sum weights",
                "finite K leaks power near the cut-off frequencies",
                "do not read the first/last K observations as a cycle",
            ))
            .build(),
    );
    ctx.finish(cycle)
}

/// Christiano–Fitzgerald asymmetric band-pass (statsmodels `cffilter`).
///
/// A linear drift is removed by OLS on \((1,t)\) before the filter. The
/// remaining weights are shifted to sum to zero so a constant is annihilated.
pub fn cf_filter(y: &Vector, low: f64, high: f64, session: &Session) -> Result<Qualified<Vector>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_univariate(&mut ctx, y);
    if !(low.is_finite() && high.is_finite()) || low <= 2.0 || high <= low {
        ctx.push(
            Issue::builder(IssueCode::InvalidWeight)
                .message(format!(
                    "CF periods must satisfy 2 < low < high (got {low}, {high})"
                ))
                .build(),
        );
    }
    let n = y.len();
    if n < 3 {
        ctx.push(
            Issue::builder(IssueCode::WindowTooShort)
                .message(format!("Christiano–Fitzgerald needs n≥3; series has {n}"))
                .meaninglessness(Meaninglessness::vacuous(
                    "asymmetric band-pass cycle",
                    "a series shorter than 3 cannot identify a cycle after drift removal",
                    "lengthen the series",
                ))
                .build(),
        );
        return ctx.finish(Vector::zeros(n));
    }
    let design = Matrix::from_fn(n, 2, |i, j| if j == 0 { 1.0 } else { i as f64 });
    let mut scratch = signlred::Report::new("cf", "drift");
    let cycle_in =
        if let Some(b) = crate::linalg::least_squares(&mut scratch, &design, y, &ctx.policy) {
            ctx.push(
                Issue::builder(IssueCode::SpectralLeakage)
                    .severity(Severity::Advisory)
                    .message("CF removes a linear drift by OLS before the band-pass")
                    .compromise(NumericalCompromise::new(
                        "CF filter on the raw series",
                        "OLS detrend then full-sample asymmetric MA",
                        "the drift is treated as identified",
                        "do not read the cycle as including a stochastic trend",
                    ))
                    .build(),
            );
            y.sub(&design.matvec(&b))
        } else {
            y.clone()
        };
    let w1 = 2.0 * std::f64::consts::PI / high.max(low + 1e-9);
    let w2 = 2.0 * std::f64::consts::PI / low.max(2.0 + 1e-9);
    let bj = |j: i64| -> f64 {
        if j == 0 {
            (w2 - w1) / std::f64::consts::PI
        } else {
            let jf = j as f64;
            ((jf * w2).sin() - (jf * w1).sin()) / (jf * std::f64::consts::PI)
        }
    };
    let mut cycle = Vector::zeros(n);
    for t in 0..n {
        let mut w = vec![0.0; n];
        let mut s = 0.0;
        for u in 0..n {
            let wt = bj(t as i64 - u as i64);
            w[u] = wt;
            s += wt;
        }
        let adj = s / n as f64;
        let mut v = 0.0;
        for u in 0..n {
            v += (w[u] - adj) * cycle_in[u];
        }
        cycle[t] = v;
    }
    ctx.finish(cycle)
}

/// Local linear trend (unobserved-components level + slope).
#[derive(Clone, Debug)]
pub struct LocalLinearTrend {
    /// Observation variance \(\sigma_\varepsilon^2\).
    pub obs_var: f64,
    /// Level innovation \(\sigma_\eta^2\).
    pub level_var: f64,
    /// Slope innovation \(\sigma_\zeta^2\).
    pub slope_var: f64,
}

impl Default for LocalLinearTrend {
    fn default() -> Self {
        Self {
            obs_var: 1.0,
            level_var: 0.1,
            slope_var: 0.01,
        }
    }
}

impl LocalLinearTrend {
    /// Default local linear trend.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Kalman-smoothed local linear trend.
#[derive(Clone, Debug)]
pub struct FittedLocalLinearTrend {
    /// Filtered/smoothed level.
    pub level: Vector,
    /// Filtered/smoothed slope.
    pub slope: Vector,
    /// Irregular \(y - \mathrm{level}\).
    pub irregular: Vector,
}

impl FitSeries for LocalLinearTrend {
    type Fitted = FittedLocalLinearTrend;
    fn fit_series(
        &mut self,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedLocalLinearTrend>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_univariate(&mut ctx, y);
        let n = y.len();
        if n == 0 {
            return ctx.finish(FittedLocalLinearTrend {
                level: Vector::zeros(0),
                slope: Vector::zeros(0),
                irregular: Vector::zeros(0),
            });
        }
        if self.obs_var <= 0.0 && self.level_var <= 0.0 && self.slope_var <= 0.0 {
            ctx.push(
                Issue::builder(IssueCode::UnidentifiedModel)
                    .message("all local-linear-trend variances are 0; the state is a perfect line")
                    .meaninglessness(Meaninglessness::vacuous(
                        "level and slope",
                        "zero process and measurement noise interpolates the first two points",
                        "set at least one positive variance",
                    ))
                    .build(),
            );
        }
        if n < 3 {
            ctx.push(
                Issue::builder(IssueCode::InsufficientSample)
                    .message("local linear trend needs n≥3 to separate level, slope, and irregular")
                    .build(),
            );
        }
        // Diffuse start: level = y0, slope = y1-y0, then Kalman filter + RTS.
        let h = self.obs_var.max(1e-12);
        let qn = self.level_var.max(0.0);
        let qz = self.slope_var.max(0.0);
        let mut mu = Vector::zeros(n);
        let mut beta = Vector::zeros(n);
        let mut p00 = Vector::zeros(n);
        let mut p11 = Vector::zeros(n);
        let mut p01 = Vector::zeros(n);
        mu[0] = y[0];
        beta[0] = if n > 1 { y[1] - y[0] } else { 0.0 };
        p00[0] = 1e2;
        p11[0] = 1e2;
        for t in 1..n {
            let m_pred = mu[t - 1] + beta[t - 1];
            let b_pred = beta[t - 1];
            let f00 = p00[t - 1] + 2.0 * p01[t - 1] + p11[t - 1] + qn;
            let f11 = p11[t - 1] + qz;
            let f01 = p01[t - 1] + p11[t - 1];
            let s = f00 + h;
            if s <= 1e-18 {
                ctx.push(
                    Issue::builder(IssueCode::SingularMatrix)
                        .message("Kalman innovation variance vanished")
                        .build(),
                );
            }
            let k0 = f00 / s.max(1e-18);
            let k1 = f01 / s.max(1e-18);
            let innov = y[t] - m_pred;
            mu[t] = m_pred + k0 * innov;
            beta[t] = b_pred + k1 * innov;
            p00[t] = (1.0 - k0) * f00;
            p11[t] = f11 - k1 * f01;
            p01[t] = (1.0 - k0) * f01;
        }
        // RTS smoother (scalar 2-state, using stored filtered moments).
        let mut sm_mu = mu.clone();
        let mut sm_b = beta.clone();
        for t in (0..n - 1).rev() {
            let f00 = p00[t] + 2.0 * p01[t] + p11[t] + qn;
            let f11 = p11[t] + qz;
            let f01 = p01[t] + p11[t];
            // Predict covariance; use a 2×2 solve for the gain.
            let mut a = Mat::<f64>::zeros(2, 2);
            a[(0, 0)] = f00;
            a[(0, 1)] = f01;
            a[(1, 0)] = f01;
            a[(1, 1)] = f11;
            let rhs0 = Vector::from_slice(&[p00[t] + p01[t], p01[t] + p11[t]]);
            let rhs1 = Vector::from_slice(&[p01[t], p11[t]]);
            let mut scratch = signlred::Report::new("llt", "rts");
            let g0 = chol_solve(&mut scratch, &a, &rhs0, &ctx.policy);
            let g1 = chol_solve(&mut scratch, &a, &rhs1, &ctx.policy);
            if let (Some(g0), Some(g1)) = (g0, g1) {
                let m_pred = mu[t] + beta[t];
                let b_pred = beta[t];
                sm_mu[t] = mu[t] + g0[0] * (sm_mu[t + 1] - m_pred) + g0[1] * (sm_b[t + 1] - b_pred);
                sm_b[t] =
                    beta[t] + g1[0] * (sm_mu[t + 1] - m_pred) + g1[1] * (sm_b[t + 1] - b_pred);
            }
        }
        let irregular = y.sub(&sm_mu);
        ctx.push(
            Issue::builder(IssueCode::PValueUnreliable)
                .severity(Severity::Advisory)
                .message("local linear trend variances are treated as known; this is not QMLE")
                .compromise(NumericalCompromise::new(
                    "unobserved-components MLE of (σε², ση², σζ²)",
                    "Kalman filter + RTS with caller variances",
                    "the hyperparameters are not estimated",
                    "the level is a smoother, not a likelihood maximizer",
                ))
                .build(),
        );
        ctx.finish(FittedLocalLinearTrend {
            level: sm_mu,
            slope: sm_b,
            irregular,
        })
    }
}

/// FIR convolution (statsmodels `tsa.filters.filtertools.convolution_filter`).
///
/// Kernel length is not identification `p`.
pub fn convolution_filter(
    y: &Vector,
    kernel: &Vector,
    session: &Session,
) -> Result<Qualified<Vector>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_univariate(&mut ctx, y);
    if kernel.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message("convolution_filter received an empty kernel")
                .build(),
        );
        return ctx.finish(y.clone());
    }
    if !kernel.as_slice().iter().all(|v| v.is_finite()) {
        ctx.push(
            Issue::builder(IssueCode::NonFiniteInput)
                .message("convolution_filter kernel is non-finite")
                .build(),
        );
    }
    let k = kernel.len();
    let out = Vector::from_iter((0..y.len()).map(|t| {
        let mut s = 0.0;
        for j in 0..k {
            if t >= j {
                s += kernel[j] * y[t - j];
            }
        }
        s
    }));
    ctx.finish(out)
}

/// IIR recursion \(y_t = x_t + \sum_j a_j y_{t-j}\) (statsmodels `recursive_filter`).
///
/// AR coefficient count is not identification `p`.
pub fn recursive_filter(
    x: &Vector,
    ar: &Vector,
    session: &Session,
) -> Result<Qualified<Vector>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_univariate(&mut ctx, x);
    let mut y = Vector::zeros(x.len());
    for t in 0..x.len() {
        let mut s = x[t];
        for j in 0..ar.len() {
            if t > j {
                s += ar[j] * y[t - 1 - j];
            }
        }
        if !s.is_finite() {
            ctx.push(
                Issue::builder(IssueCode::CausalityViolated)
                    .message("recursive_filter overflowed; later outputs set to 0")
                    .build(),
            );
            s = 0.0;
        }
        y[t] = s;
    }
    ctx.finish(y)
}

/// Multi-input single-output FIR (statsmodels `miso_lfilter`).
///
/// Each column of `x` is convolved with `kernel` and the channels are summed.
/// Filter length is not identification `p`.
pub fn miso_lfilter(
    x: &Matrix,
    kernel: &Vector,
    session: &Session,
) -> Result<Qualified<Vector>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, x, None, &ctx.policy);
    if kernel.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message("miso_lfilter received an empty kernel")
                .build(),
        );
        return ctx.finish(Vector::zeros(x.nrows()));
    }
    if !kernel.as_slice().iter().all(|v| v.is_finite()) {
        ctx.push(
            Issue::builder(IssueCode::NonFiniteInput)
                .message("miso_lfilter kernel is non-finite")
                .build(),
        );
    }
    let k = kernel.len();
    let out = Vector::from_iter((0..x.nrows()).map(|t| {
        let mut s = 0.0;
        for j in 0..x.ncols() {
            for u in 0..k {
                if t >= u {
                    s += kernel[u] * x.get(t - u, j);
                }
            }
        }
        s
    }));
    ctx.finish(out)
}

/// Single-input single-output linear filter (SciPy / statsmodels `lfilter`).
///
/// \(a_0 y_t = \sum_i b_i x_{t-i} - \sum_{j\ge 1} a_j y_{t-j}\). Filter length
/// is not identification `p`.
pub fn lfilter(
    b: &Vector,
    a: &Vector,
    x: &Vector,
    session: &Session,
) -> Result<Qualified<Vector>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_univariate(&mut ctx, x);
    if b.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message("lfilter received an empty numerator")
                .build(),
        );
        return ctx.finish(Vector::zeros(x.len()));
    }
    if !b.as_slice().iter().all(|v| v.is_finite()) {
        ctx.push(
            Issue::builder(IssueCode::NonFiniteInput)
                .message("lfilter numerator is non-finite")
                .build(),
        );
    }
    if !a.is_empty() && !a.as_slice().iter().all(|v| v.is_finite()) {
        ctx.push(
            Issue::builder(IssueCode::NonFiniteInput)
                .message("lfilter denominator is non-finite")
                .build(),
        );
    }
    let a0 = a.as_slice().first().copied().unwrap_or(1.0);
    let scale = if !a0.is_finite() || a0.abs() <= 1e-18 {
        ctx.push(
            Issue::builder(IssueCode::ScaleFactorZero)
                .severity(Severity::Warning)
                .message("lfilter a[0] vanished; using 1")
                .build(),
        );
        1.0
    } else {
        a0
    };
    let mut y = Vector::zeros(x.len());
    for t in 0..x.len() {
        let mut s = 0.0_f64;
        for i in 0..b.len() {
            if t >= i {
                s += b[i] * x[t - i];
            }
        }
        for j in 1..a.len() {
            if t >= j {
                s -= a[j] * y[t - j];
            }
        }
        let v = s / scale;
        if !v.is_finite() {
            ctx.push(
                Issue::builder(IssueCode::CausalityViolated)
                    .message("lfilter overflowed; later outputs set to 0")
                    .build(),
            );
            y[t] = 0.0;
        } else {
            y[t] = v;
        }
    }
    ctx.finish(y)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bk_annihilates_a_linear_trend() {
        let y = Vector::from_iter((0..48).map(|i| i as f64));
        let c = bk_filter(&y, 6.0, 32.0, 8, &Session::new("bk", "fit"))
            .expect("bk")
            .value;
        let mid: f64 = (8..40).map(|i| c[i].abs()).sum::<f64>() / 32.0;
        assert!(mid < 0.5, "cycle mean abs={mid}");
        let ker = Vector::from_slice(&[0.25, 0.5, 0.25]);
        let conv = convolution_filter(&y, &ker, &Session::new("conv", "fit"))
            .expect("conv")
            .value;
        assert_eq!(conv.len(), 48);
        assert!(conv.as_slice().iter().all(|v| v.is_finite()));
        let ar = Vector::from_slice(&[0.1]);
        let rec = recursive_filter(&y, &ar, &Session::new("rec", "fit"))
            .expect("rec")
            .value;
        assert!(rec.as_slice().iter().all(|v| v.is_finite()));
        let x2 = Matrix::from_fn(48, 2, |i, j| {
            if j == 0 {
                i as f64
            } else {
                (i as f64).sin()
            }
        });
        let miso = miso_lfilter(&x2, &ker, &Session::new("miso", "fit"))
            .expect("miso")
            .value;
        assert_eq!(miso.len(), 48);
        assert!(miso.as_slice().iter().all(|v| v.is_finite()));
        let one = Vector::from_slice(&[1.0]);
        let lf = lfilter(&ker, &one, &y, &Session::new("lf", "fit"))
            .expect("lfilter")
            .value;
        assert_eq!(lf.len(), 48);
        assert!(lf.as_slice().iter().all(|v| v.is_finite()));
    }

    #[test]
    fn cf_annihilates_a_linear_trend() {
        let y = Vector::from_iter((0..48).map(|i| i as f64));
        let c = cf_filter(&y, 6.0, 32.0, &Session::new("cf", "fit"))
            .expect("cf")
            .value;
        let mid: f64 = (8..40).map(|i| c[i].abs()).sum::<f64>() / 32.0;
        assert!(mid < 0.75, "cf cycle mean abs={mid}");
    }

    #[test]
    fn local_linear_tracks_a_ramp() {
        let y = Vector::from_iter((0..20).map(|i| 3.0 + 0.5 * i as f64));
        let q = LocalLinearTrend {
            obs_var: 0.05,
            level_var: 0.01,
            slope_var: 1e-4,
        }
        .fit_series(&y, &Session::new("llt", "fit"))
        .expect("llt");
        assert!((q.value.level[10] - y[10]).abs() < 0.5);
        assert!(q.value.slope.as_slice().iter().all(|v| v.is_finite()));
    }
}
