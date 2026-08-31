//! Hidden Markov models (hmmlearn-style).
//!
//! [`GaussianHmm`] supports univariate and multivariate sequences (`X` rows =
//! time, columns = dimensions) with scaled forward–backward, Viterbi decoding,
//! and Baum–Welch. [`MultinomialHmm`] uses integer emission codes. [`GmmHmm`]
//! uses a diagonal Gaussian mixture per state (one component recovers a
//! diagonal [`GaussianHmm`]). [`PoissonHmm`] uses integer counts in column 0.
//!
//! Quality: [`IssueCode::ForwardUnderflow`], [`IssueCode::ScaleFactorZero`],
//! [`IssueCode::EmissionDegenerate`], [`IssueCode::AbsorbingStateOnly`],
//! [`IssueCode::UnreachableState`].

use crate::context::FitCtx;
use crate::data::{Matrix, Vector};
use crate::rng::Rng;
use crate::traits::{FitUnsupervised, Predict};
use crate::validate::{inspect_identification, inspect_xy};
use ojizou_san::Session;
use signlred::{Issue, IssueCode, Meaninglessness, NumericalCompromise, Qualified, Result};

const COV_FLOOR: f64 = 1e-6;
const TRANS_FLOOR: f64 = 1e-8;
const LN_2PI: f64 = 1.8378770664093453; // ln(2π)

/// A sampled state path and its emissions.
#[derive(Clone, Debug)]
pub struct HmmSample {
    /// Observations (`T` × `d`).
    pub obs: Matrix,
    /// Hidden-state ids as `f64`.
    pub states: Vector,
}

fn logsumexp(xs: &[f64]) -> f64 {
    let m = xs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !m.is_finite() {
        return m;
    }
    let mut s = 0.0;
    for &v in xs {
        s += (v - m).exp();
    }
    m + s.ln()
}

fn col_stds(x: &Matrix) -> Vector {
    let p = x.ncols();
    let mut out = Vector::zeros(p);
    for j in 0..p {
        out[j] = x.column(j).std();
    }
    out
}

fn series_zero_variance(x: &Matrix, tol: f64) -> bool {
    if x.nrows() == 0 || x.ncols() == 0 {
        return true;
    }
    col_stds(x).as_slice().iter().all(|s| *s <= tol)
}

fn emission_degenerate_issue(msg: impl Into<String>) -> Issue {
    Issue::builder(IssueCode::EmissionDegenerate)
        .message(msg)
        .meaninglessness(Meaninglessness::vacuous(
            "HMM emission parameters",
            "zero-variance observations collapse every Gaussian / categorical to a delta; decoding is a hard assignment and covariances are unidentified",
            "do not interpret emission scales; collect variation or switch to a discrete model",
        ))
        .build()
}

fn empty_labels(n: usize) -> Vector {
    Vector::zeros(n)
}

fn renormalize_rows(m: &mut Matrix, floor: f64) {
    let (r, c) = m.shape();
    for i in 0..r {
        let mut s = 0.0;
        for j in 0..c {
            let v = m.get(i, j).max(floor);
            m.set(i, j, v);
            s += v;
        }
        if s > 0.0 {
            for j in 0..c {
                m.set(i, j, m.get(i, j) / s);
            }
        } else if c > 0 {
            let u = 1.0 / c as f64;
            for j in 0..c {
                m.set(i, j, u);
            }
        }
    }
}

fn renormalize_vec(v: &mut Vector, floor: f64) {
    let mut s = 0.0;
    for i in 0..v.len() {
        v[i] = v[i].max(floor);
        s += v[i];
    }
    if s > 0.0 {
        for i in 0..v.len() {
            v[i] /= s;
        }
    } else if !v.as_slice().is_empty() {
        let u = 1.0 / v.len() as f64;
        for i in 0..v.len() {
            v[i] = u;
        }
    }
}

/// Zero backward transitions and pin `π` to state 0 (hmmlearn `left_right`).
///
/// Forbidden cells stay exactly zero; they are not re-floored.
fn enforce_left_right(start: &mut Vector, trans: &mut Matrix) {
    let k = trans.nrows().min(trans.ncols());
    for i in 0..k {
        for j in 0..k {
            if j < i {
                trans.set(i, j, 0.0);
            }
        }
        let mut s = 0.0;
        for j in i..k {
            s += trans.get(i, j).max(0.0);
        }
        if s > 0.0 {
            for j in i..k {
                trans.set(i, j, trans.get(i, j).max(0.0) / s);
            }
        } else if i < k {
            trans.set(i, i, 1.0);
        }
    }
    if !start.is_empty() {
        start[0] = 1.0;
        for j in 1..start.len() {
            start[j] = 0.0;
        }
    }
}

fn log_diag_gauss(x: &Matrix, t: usize, mean: &Matrix, state: usize, var: &Matrix) -> f64 {
    let d = x.ncols().min(mean.ncols()).min(var.ncols());
    let mut s = 0.0;
    for j in 0..d {
        let v = var.get(state, j).max(COV_FLOOR);
        let z = x.get(t, j) - mean.get(state, j);
        s += LN_2PI + v.ln() + z * z / v;
    }
    -0.5 * s
}

/// Scaled forward–backward. `log_emit[t][j]` is log p(o_t | state j).
struct ScaledFb {
    loglik: f64,
    gamma: Vec<Vec<f64>>,
    xi: Vec<Vec<Vec<f64>>>,
}

fn scaled_forward_backward(
    ctx: &mut FitCtx,
    start: &Vector,
    trans: &Matrix,
    log_emit: &[Vec<f64>],
) -> Option<ScaledFb> {
    let t_len = log_emit.len();
    let s = start.len();
    if t_len == 0 || s == 0 {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message("HMM forward–backward on an empty sequence")
                .build(),
        );
        return None;
    }
    let mut alpha = vec![vec![0.0; s]; t_len];
    let mut scale = vec![0.0; t_len];
    for j in 0..s {
        let e = log_emit[0][j].exp();
        alpha[0][j] = start[j].max(0.0) * e;
    }
    scale[0] = alpha[0].iter().sum();
    if scale[0] <= 0.0 || !scale[0].is_finite() {
        ctx.push(
            Issue::builder(IssueCode::ScaleFactorZero)
                .message(
                    "t=0 scale factor is zero; the sequence is impossible under the current HMM",
                )
                .metric("t", 0.0)
                .build(),
        );
        ctx.push(
            Issue::builder(IssueCode::ForwardUnderflow)
                .message("unscaled forward mass vanished at t=0")
                .build(),
        );
        return None;
    }
    if scale[0] < 1e-300 {
        ctx.push(
            Issue::builder(IssueCode::ForwardUnderflow)
                .message(format!("forward scale at t=0 is {sc:.3e}", sc = scale[0]))
                .metric("scale", scale[0])
                .build(),
        );
    }
    for j in 0..s {
        alpha[0][j] /= scale[0];
    }
    for t in 1..t_len {
        for j in 0..s {
            let mut acc = 0.0;
            for i in 0..s {
                acc += alpha[t - 1][i] * trans.get(i, j);
            }
            alpha[t][j] = acc * log_emit[t][j].exp();
        }
        scale[t] = alpha[t].iter().sum();
        if scale[t] <= 0.0 || !scale[t].is_finite() {
            ctx.push(
                Issue::builder(IssueCode::ScaleFactorZero)
                    .message(format!("scale factor is zero at t={t}"))
                    .metric("t", t as f64)
                    .build(),
            );
            ctx.push(
                Issue::builder(IssueCode::ForwardUnderflow)
                    .message(format!("forward mass underflowed at t={t}"))
                    .build(),
            );
            return None;
        }
        if scale[t] < 1e-300 {
            ctx.push(
                Issue::builder(IssueCode::ForwardUnderflow)
                    .message(format!("forward scale at t={t} is {sc:.3e}", sc = scale[t]))
                    .metric("t", t as f64)
                    .metric("scale", scale[t])
                    .build(),
            );
        }
        for j in 0..s {
            alpha[t][j] /= scale[t];
        }
    }
    let mut loglik = 0.0;
    for &c in &scale {
        loglik += c.ln();
    }
    let mut beta = vec![vec![0.0; s]; t_len];
    for j in 0..s {
        beta[t_len - 1][j] = 1.0;
    }
    for t in (0..t_len - 1).rev() {
        for i in 0..s {
            let mut acc = 0.0;
            for j in 0..s {
                acc += trans.get(i, j) * log_emit[t + 1][j].exp() * beta[t + 1][j];
            }
            beta[t][i] = acc / scale[t + 1];
        }
    }
    let mut gamma = vec![vec![0.0; s]; t_len];
    for t in 0..t_len {
        let mut nrm = 0.0;
        for j in 0..s {
            gamma[t][j] = alpha[t][j] * beta[t][j];
            nrm += gamma[t][j];
        }
        if nrm > 0.0 {
            for j in 0..s {
                gamma[t][j] /= nrm;
            }
        }
    }
    let mut xi = vec![vec![vec![0.0; s]; s]; t_len.saturating_sub(1)];
    for t in 0..t_len.saturating_sub(1) {
        let mut nrm = 0.0;
        for i in 0..s {
            for j in 0..s {
                let v = alpha[t][i] * trans.get(i, j) * log_emit[t + 1][j].exp() * beta[t + 1][j];
                xi[t][i][j] = v;
                nrm += v;
            }
        }
        if nrm > 0.0 {
            for i in 0..s {
                for j in 0..s {
                    xi[t][i][j] /= nrm;
                }
            }
        }
    }
    if !loglik.is_finite() {
        ctx.push(
            Issue::builder(IssueCode::LossIsNan)
                .message("HMM log-likelihood is not finite after scaled forward")
                .build(),
        );
    }
    let _ = (alpha, beta);
    Some(ScaledFb { loglik, gamma, xi })
}

fn viterbi_path(start: &Vector, trans: &Matrix, log_emit: &[Vec<f64>]) -> (Vector, f64) {
    let t_len = log_emit.len();
    let s = start.len();
    if t_len == 0 || s == 0 {
        return (Vector::zeros(t_len), f64::NEG_INFINITY);
    }
    let mut delta = vec![vec![f64::NEG_INFINITY; s]; t_len];
    let mut psi = vec![vec![0usize; s]; t_len];
    for j in 0..s {
        let lp = if start[j] > 0.0 {
            start[j].ln()
        } else {
            f64::NEG_INFINITY
        };
        delta[0][j] = lp + log_emit[0][j];
    }
    for t in 1..t_len {
        for j in 0..s {
            let mut best = f64::NEG_INFINITY;
            let mut arg = 0usize;
            for i in 0..s {
                let a = if trans.get(i, j) > 0.0 {
                    trans.get(i, j).ln()
                } else {
                    f64::NEG_INFINITY
                };
                let v = delta[t - 1][i] + a;
                if v > best {
                    best = v;
                    arg = i;
                }
            }
            delta[t][j] = best + log_emit[t][j];
            psi[t][j] = arg;
        }
    }
    let mut last = 0usize;
    let mut best = f64::NEG_INFINITY;
    for j in 0..s {
        if delta[t_len - 1][j] > best {
            best = delta[t_len - 1][j];
            last = j;
        }
    }
    let mut path = vec![0usize; t_len];
    path[t_len - 1] = last;
    for t in (1..t_len).rev() {
        path[t - 1] = psi[t][path[t]];
    }
    (Vector::from_iter(path.iter().map(|v| *v as f64)), best)
}

fn diagnose_chain(ctx: &mut FitCtx, start: &Vector, trans: &Matrix, occup: &[f64]) {
    let s = start.len();
    for j in 0..s {
        let incoming: f64 = (0..s).map(|i| trans.get(i, j)).sum();
        if start[j] <= TRANS_FLOOR && incoming <= TRANS_FLOOR * s as f64 {
            ctx.push(
                Issue::builder(IssueCode::UnreachableState)
                    .message(format!("state {j} is unreachable from π and A"))
                    .metric("state", j as f64)
                    .build(),
            );
        }
        if occup.get(j).copied().unwrap_or(0.0) <= TRANS_FLOOR {
            ctx.push(
                Issue::builder(IssueCode::UnreachableState)
                    .message(format!("state {j} received ~0 posterior occupancy"))
                    .metric("state", j as f64)
                    .metric("occupancy", occup.get(j).copied().unwrap_or(0.0))
                    .build(),
            );
        }
        let diag = trans.get(j, j);
        let mut off = 0.0;
        for k in 0..s {
            if k != j {
                off += trans.get(j, k);
            }
        }
        if diag >= 1.0 - 1e-6 && off <= 1e-6 {
            ctx.push(
                Issue::builder(IssueCode::AbsorbingStateOnly)
                    .message(format!("state {j} is absorbing (A[{j},{j}]={diag:.6})"))
                    .metric("state", j as f64)
                    .metric("self_transition", diag)
                    .build(),
            );
        }
    }
}

fn kmeans_pp_rows(x: &Matrix, k: usize, rng: &mut Rng) -> Matrix {
    let (n, p) = x.shape();
    let k = k.max(1).min(n.max(1));
    let mut centers = Matrix::zeros(k, p);
    if n == 0 {
        return centers;
    }
    let i0 = rng.below(n);
    for j in 0..p {
        centers.set(0, j, x.get(i0, j));
    }
    let mut d2 = vec![f64::INFINITY; n];
    for c in 1..k {
        for i in 0..n {
            let mut s = 0.0;
            for j in 0..p {
                let d = x.get(i, j) - centers.get(c - 1, j);
                s += d * d;
            }
            if s < d2[i] {
                d2[i] = s;
            }
        }
        let sum: f64 = d2.iter().sum();
        let mut tick = rng.uniform() * sum.max(1e-300);
        let mut chosen = n - 1;
        for i in 0..n {
            tick -= d2[i];
            if tick <= 0.0 {
                chosen = i;
                break;
            }
        }
        for j in 0..p {
            centers.set(c, j, x.get(chosen, j));
        }
    }
    centers
}

fn init_trans(k: usize) -> Matrix {
    if k == 0 {
        return Matrix::zeros(0, 0);
    }
    if k == 1 {
        return Matrix::from_fn(1, 1, |_, _| 1.0);
    }
    let stay = 0.8;
    let leave = (1.0 - stay) / (k - 1) as f64;
    Matrix::from_fn(k, k, |i, j| if i == j { stay } else { leave })
}

fn init_start(k: usize) -> Vector {
    if k == 0 {
        Vector::zeros(0)
    } else {
        Vector::filled(k, 1.0 / k as f64)
    }
}

fn global_diag_var(x: &Matrix) -> Vector {
    let p = x.ncols();
    let mut v = Vector::zeros(p);
    for j in 0..p {
        let s = x.column(j).std();
        v[j] = (s * s).max(COV_FLOOR);
    }
    v
}

/// Gaussian HMM with diagonal (per-dimension) covariances.
#[derive(Clone, Debug)]
pub struct GaussianHmm {
    /// Number of hidden states.
    pub n_states: usize,
    /// Baum–Welch iteration cap.
    pub max_iter: usize,
    /// Seed for k-means++ mean initialization.
    pub seed: u64,
    /// If true, transitions to earlier states are zeroed (hmmlearn `left_right`).
    pub left_right: bool,
}

impl Default for GaussianHmm {
    fn default() -> Self {
        Self {
            n_states: 2,
            max_iter: 50,
            seed: 0,
            left_right: false,
        }
    }
}

impl GaussianHmm {
    /// `n_states` hidden Gaussians.
    pub fn new(n_states: usize) -> Self {
        Self {
            n_states,
            ..Self::default()
        }
    }

    /// Left-right (Bakis) Gaussian HMM: \(A_{ij}=0\) for \(j<i\).
    ///
    /// State count is not treated as an extra identification `p` here.
    pub fn left_right(n_states: usize) -> Self {
        Self {
            n_states,
            left_right: true,
            ..Self::default()
        }
    }

    /// Fit alias for [`FitUnsupervised::fit_unsupervised`].
    pub fn fit(&mut self, x: &Matrix, session: &Session) -> Result<Qualified<FittedGaussianHmm>> {
        self.fit_unsupervised(x, session)
    }
}

/// Fitted diagonal Gaussian HMM.
#[derive(Clone, Debug)]
pub struct FittedGaussianHmm {
    /// Viterbi state path on the training sequence (`f64` ids).
    pub labels: Vector,
    /// Number of states.
    pub n_states: usize,
    /// Start distribution `π`.
    pub start: Vector,
    /// Transition matrix `A` (`n_states` × `n_states`).
    pub trans: Matrix,
    /// State means (`n_states` × `d`).
    pub means: Matrix,
    /// Diagonal variances (`n_states` × `d`).
    pub covs: Matrix,
    /// Training log-likelihood (scaled forward).
    pub loglik: f64,
}

impl FittedGaussianHmm {
    fn log_emit_seq(&self, x: &Matrix) -> Vec<Vec<f64>> {
        let t = x.nrows();
        let s = self.n_states;
        let mut out = vec![vec![f64::NEG_INFINITY; s]; t];
        for ti in 0..t {
            for j in 0..s {
                out[ti][j] = log_diag_gauss(x, ti, &self.means, j, &self.covs);
            }
        }
        out
    }

    /// Viterbi state path.
    pub fn decode(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("decode"));
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        if x.ncols() != self.means.ncols() {
            ctx.push(
                Issue::builder(IssueCode::DimensionMismatch)
                    .message("decode X has a different observation dimension than the HMM")
                    .build(),
            );
        }
        if series_zero_variance(x, ctx.policy.near_zero_variance) {
            ctx.push(emission_degenerate_issue(
                "decode sequence has zero variance in every dimension",
            ));
        }
        let (path, _) = viterbi_path(&self.start, &self.trans, &self.log_emit_seq(x));
        ctx.finish(path)
    }

    /// Scaled-forward log-likelihood.
    pub fn score(&self, x: &Matrix, session: &Session) -> Result<Qualified<f64>> {
        let mut ctx = FitCtx::with_session(session.child("score"));
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        if x.nrows() == 0 {
            return ctx.finish(f64::NEG_INFINITY);
        }
        let fb = scaled_forward_backward(&mut ctx, &self.start, &self.trans, &self.log_emit_seq(x));
        ctx.finish(fb.map(|f| f.loglik).unwrap_or(f64::NEG_INFINITY))
    }

    /// Draw a sequence of length `n` and its hidden path.
    pub fn sample(&self, n: usize, seed: u64, session: &Session) -> Result<Qualified<HmmSample>> {
        let mut ctx = FitCtx::with_session(session.child("sample"));
        inspect_identification(&mut ctx.report, n, self.n_states.max(1), &ctx.policy);
        let d = self.means.ncols();
        let s = self.n_states.max(1);
        if n == 0 {
            return ctx.finish(HmmSample {
                obs: Matrix::zeros(0, d),
                states: Vector::zeros(0),
            });
        }
        let mut rng = Rng::new(seed | 1);
        let mut states = Vector::zeros(n);
        let mut obs = Matrix::zeros(n, d);
        let pick = |rng: &mut Rng, probs: &Vector| -> usize {
            let mut u = rng.uniform();
            for i in 0..probs.len() {
                u -= probs[i];
                if u <= 0.0 {
                    return i;
                }
            }
            probs.len().saturating_sub(1)
        };
        let mut st = pick(&mut rng, &self.start);
        for t in 0..n {
            states[t] = st as f64;
            for j in 0..d {
                let sd = self
                    .covs
                    .get(st.min(self.covs.nrows().saturating_sub(1)), j)
                    .max(COV_FLOOR)
                    .sqrt();
                obs.set(
                    t,
                    j,
                    self.means
                        .get(st.min(self.means.nrows().saturating_sub(1)), j)
                        + sd * rng.standard_normal(),
                );
            }
            if t + 1 < n {
                let row = Vector::from_iter((0..s).map(|k| {
                    if st < self.trans.nrows() && k < self.trans.ncols() {
                        self.trans.get(st, k)
                    } else {
                        0.0
                    }
                }));
                st = pick(&mut rng, &row);
            }
        }
        ctx.finish(HmmSample { obs, states })
    }
}

impl Predict for FittedGaussianHmm {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        self.decode(x, session)
    }
}

impl FitUnsupervised for GaussianHmm {
    type Fitted = FittedGaussianHmm;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedGaussianHmm>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        inspect_identification(
            &mut ctx.report,
            x.nrows(),
            self.n_states.max(1),
            &ctx.policy,
        );
        let (t_len, d) = x.shape();
        let k = self.n_states.max(1);
        if t_len == 0 || d == 0 {
            return ctx.finish(FittedGaussianHmm {
                labels: empty_labels(t_len),
                n_states: k,
                start: init_start(k),
                trans: init_trans(k),
                means: Matrix::zeros(k, d),
                covs: Matrix::zeros(k, d),
                loglik: f64::NAN,
            });
        }
        if series_zero_variance(x, ctx.policy.near_zero_variance) {
            ctx.push(emission_degenerate_issue(
                "observation series has zero variance; Gaussian emissions are degenerate",
            ));
        }
        if k != self.n_states {
            ctx.push(
                Issue::builder(IssueCode::UnidentifiedModel)
                    .message("GaussianHmm requires n_states ≥ 1")
                    .build(),
            );
        }
        let mut rng = Rng::new(self.seed | 1);
        let mut means = kmeans_pp_rows(x, k.min(t_len), &mut rng);
        if means.nrows() < k {
            let mut padded = Matrix::zeros(k, d);
            for i in 0..means.nrows() {
                for j in 0..d {
                    padded.set(i, j, means.get(i, j));
                }
            }
            means = padded;
        }
        let gvar = global_diag_var(x);
        let mut covs = Matrix::from_fn(k, d, |_, j| gvar[j]);
        let mut start = init_start(k);
        let mut trans = init_trans(k);
        if self.left_right {
            enforce_left_right(&mut start, &mut trans);
        }
        let mut loglik = f64::NEG_INFINITY;
        let mut last_gamma: Vec<Vec<f64>> = Vec::new();
        for it in 0..self.max_iter.max(1) {
            let mut log_emit = vec![vec![0.0; k]; t_len];
            for t in 0..t_len {
                for j in 0..k {
                    log_emit[t][j] = log_diag_gauss(x, t, &means, j, &covs);
                }
            }
            let Some(fb) = scaled_forward_backward(&mut ctx, &start, &trans, &log_emit) else {
                break;
            };
            loglik = fb.loglik;
            ctx.session.step(it as u64, -loglik, None);
            last_gamma = fb.gamma.clone();
            // M-step: start / transitions.
            for j in 0..k {
                start[j] = fb.gamma[0][j];
            }
            renormalize_vec(&mut start, TRANS_FLOOR);
            if t_len > 1 {
                for i in 0..k {
                    let mut den = 0.0;
                    for t in 0..t_len - 1 {
                        den += fb.gamma[t][i];
                    }
                    for j in 0..k {
                        let mut num = 0.0;
                        for t in 0..t_len - 1 {
                            num += fb.xi[t][i][j];
                        }
                        trans.set(
                            i,
                            j,
                            if den > 0.0 {
                                num / den
                            } else {
                                trans.get(i, j)
                            },
                        );
                    }
                }
                if self.left_right {
                    enforce_left_right(&mut start, &mut trans);
                } else {
                    renormalize_rows(&mut trans, TRANS_FLOOR);
                }
            }
            if self.left_right && k > 1 {
                enforce_left_right(&mut start, &mut trans);
            }
            // Means / variances.
            for j in 0..k {
                let mut nj = 0.0;
                for t in 0..t_len {
                    nj += fb.gamma[t][j];
                }
                if nj <= TRANS_FLOOR {
                    ctx.push(
                        Issue::builder(IssueCode::UnreachableState)
                            .message(format!("state {j} posterior mass {nj:.3e} during EM"))
                            .metric("state", j as f64)
                            .build(),
                    );
                    continue;
                }
                for dim in 0..d {
                    let mut m = 0.0;
                    for t in 0..t_len {
                        m += fb.gamma[t][j] * x.get(t, dim);
                    }
                    means.set(j, dim, m / nj);
                }
                for dim in 0..d {
                    let mut s = 0.0;
                    for t in 0..t_len {
                        let z = x.get(t, dim) - means.get(j, dim);
                        s += fb.gamma[t][j] * z * z;
                    }
                    let raw = s / nj;
                    if raw <= ctx.policy.near_zero_variance {
                        ctx.push(
                            Issue::builder(IssueCode::DegenerateDistribution)
                                .message(format!(
                                    "state {j} dim {dim} variance {raw:.3e} hit the floor"
                                ))
                                .metric("state", j as f64)
                                .build(),
                        );
                    }
                    covs.set(j, dim, raw.max(COV_FLOOR));
                }
            }
            if it + 1 == self.max_iter {
                ctx.push(
                    Issue::builder(IssueCode::MaxIterReached)
                        .message("Baum–Welch hit max_iter")
                        .build(),
                );
            }
        }
        let occup: Vec<f64> = (0..k)
            .map(|j| {
                last_gamma
                    .iter()
                    .map(|g| g.get(j).copied().unwrap_or(0.0))
                    .sum()
            })
            .collect();
        diagnose_chain(&mut ctx, &start, &trans, &occup);
        ctx.push(
            Issue::builder(IssueCode::JitterInjected)
                .message("diagonal covariance and transition floors applied")
                .compromise(NumericalCompromise::new(
                    "unconstrained Baum–Welch",
                    format!("cov ≥ {COV_FLOOR}, A,π ≥ {TRANS_FLOOR} then renormalized"),
                    "zero cells make the sequence probability vanish and covariances singular",
                    "small floors change the MLE; do not treat them as estimated parameters",
                ))
                .build(),
        );
        let log_emit: Vec<Vec<f64>> = (0..t_len)
            .map(|t| {
                (0..k)
                    .map(|j| log_diag_gauss(x, t, &means, j, &covs))
                    .collect()
            })
            .collect();
        let (labels, _) = viterbi_path(&start, &trans, &log_emit);
        if labels.as_slice().iter().any(|z| !z.is_finite()) {
            ctx.push(
                Issue::builder(IssueCode::NonFiniteOutput)
                    .message("Viterbi path contains NaN/Inf")
                    .build(),
            );
        }
        if self.left_right {
            enforce_left_right(&mut start, &mut trans);
        }
        ctx.finish(FittedGaussianHmm {
            labels,
            n_states: k,
            start,
            trans,
            means,
            covs,
            loglik,
        })
    }
}

fn try_chol_lower(a: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let n = a.len();
    let mut l = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let mut s = a[i][j];
            for k in 0..j {
                s -= l[i][k] * l[j][k];
            }
            if i == j {
                if s <= 1e-18 {
                    return None;
                }
                l[i][j] = s.sqrt();
            } else if l[j][j] > 1e-18 {
                l[i][j] = s / l[j][j];
            } else {
                return None;
            }
        }
    }
    Some(l)
}

fn log_full_gauss(x: &Matrix, t: usize, means: &Matrix, state: usize, cov: &Matrix) -> f64 {
    let d = x
        .ncols()
        .min(means.ncols())
        .min(cov.nrows())
        .min(cov.ncols());
    if d == 0 {
        return f64::NEG_INFINITY;
    }
    let mut a = vec![vec![0.0_f64; d]; d];
    for i in 0..d {
        for j in 0..d {
            a[i][j] = cov.get(i, j);
        }
        a[i][i] += COV_FLOOR;
    }
    let l = match try_chol_lower(&a) {
        Some(l) => l,
        None => {
            for i in 0..d {
                a[i][i] += 1e-3;
            }
            match try_chol_lower(&a) {
                Some(l) => l,
                None => {
                    let mut s = 0.0_f64;
                    for j in 0..d {
                        let v = cov.get(j, j).max(COV_FLOOR);
                        let z = x.get(t, j) - means.get(state, j);
                        s += z * z / v + v.ln();
                    }
                    return -0.5 * (s + d as f64 * LN_2PI);
                }
            }
        }
    };
    let mut z = vec![0.0_f64; d];
    for i in 0..d {
        let mut s = x.get(t, i) - means.get(state, i);
        for j in 0..i {
            s -= l[i][j] * z[j];
        }
        z[i] = s / l[i][i].max(1e-18);
    }
    let quad: f64 = z.iter().map(|v| v * v).sum();
    let mut ld = 0.0_f64;
    for i in 0..d {
        ld += l[i][i].max(1e-18).ln();
    }
    -0.5 * (quad + 2.0 * ld + d as f64 * LN_2PI)
}

fn global_full_cov(x: &Matrix) -> Matrix {
    let (n, d) = x.shape();
    let mut mean = vec![0.0_f64; d];
    if n > 0 {
        for j in 0..d {
            mean[j] = x.column(j).mean();
        }
    }
    let mut c = Matrix::zeros(d, d);
    let nf = n.max(1) as f64;
    for t in 0..n {
        for a in 0..d {
            let da = x.get(t, a) - mean[a];
            for b in 0..d {
                c.set(a, b, c.get(a, b) + da * (x.get(t, b) - mean[b]));
            }
        }
    }
    for a in 0..d {
        for b in 0..d {
            c.set(a, b, c.get(a, b) / nf);
        }
        c.set(a, a, c.get(a, a).max(COV_FLOOR));
    }
    c
}

/// Gaussian HMM with a full (dense) covariance per state (hmmlearn `covariance_type="full"`).
///
/// Covariance free-parameter count is not identification `p`.
#[derive(Clone, Debug)]
pub struct GaussianHmmFull {
    /// Number of hidden states.
    pub n_states: usize,
    /// Baum–Welch iteration cap.
    pub max_iter: usize,
    /// Seed for k-means++ mean initialization.
    pub seed: u64,
    /// If true, transitions to earlier states are zeroed (hmmlearn `left_right`).
    pub left_right: bool,
}

impl Default for GaussianHmmFull {
    fn default() -> Self {
        Self {
            n_states: 2,
            max_iter: 50,
            seed: 0,
            left_right: false,
        }
    }
}

impl GaussianHmmFull {
    /// `n_states` hidden full-covariance Gaussians.
    pub fn new(n_states: usize) -> Self {
        Self {
            n_states,
            ..Self::default()
        }
    }

    /// Left-right full-covariance Gaussian HMM.
    pub fn left_right(n_states: usize) -> Self {
        Self {
            n_states,
            left_right: true,
            ..Self::default()
        }
    }

    /// Fit alias.
    pub fn fit(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedGaussianHmmFull>> {
        self.fit_unsupervised(x, session)
    }
}

/// Fitted full-covariance Gaussian HMM.
#[derive(Clone, Debug)]
pub struct FittedGaussianHmmFull {
    /// Viterbi path.
    pub labels: Vector,
    /// Number of states.
    pub n_states: usize,
    /// Start distribution.
    pub start: Vector,
    /// Transitions.
    pub trans: Matrix,
    /// State means (`n_states` × `d`).
    pub means: Matrix,
    /// Per-state dense covariances (`d` × `d` each).
    pub covs: Vec<Matrix>,
    /// Training log-likelihood.
    pub loglik: f64,
}

impl FittedGaussianHmmFull {
    fn log_emit_seq(&self, x: &Matrix) -> Vec<Vec<f64>> {
        let t = x.nrows();
        let s = self.n_states;
        let mut out = vec![vec![f64::NEG_INFINITY; s]; t];
        for ti in 0..t {
            for j in 0..s {
                let cov = self
                    .covs
                    .get(j)
                    .cloned()
                    .unwrap_or_else(|| Matrix::zeros(0, 0));
                out[ti][j] = log_full_gauss(x, ti, &self.means, j, &cov);
            }
        }
        out
    }

    /// Viterbi state path.
    pub fn decode(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("decode"));
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        if x.ncols() != self.means.ncols() {
            ctx.push(
                Issue::builder(IssueCode::DimensionMismatch)
                    .severity(signlred::Severity::Warning)
                    .message("full-covariance decode X has a different observation dimension")
                    .build(),
            );
        }
        let (path, _) = viterbi_path(&self.start, &self.trans, &self.log_emit_seq(x));
        ctx.finish(path)
    }

    /// Scaled-forward log-likelihood.
    pub fn score(&self, x: &Matrix, session: &Session) -> Result<Qualified<f64>> {
        let mut ctx = FitCtx::with_session(session.child("score"));
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        if x.nrows() == 0 {
            return ctx.finish(f64::NEG_INFINITY);
        }
        let fb = scaled_forward_backward(&mut ctx, &self.start, &self.trans, &self.log_emit_seq(x));
        ctx.finish(fb.map(|f| f.loglik).unwrap_or(f64::NEG_INFINITY))
    }
}

impl Predict for FittedGaussianHmmFull {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        self.decode(x, session)
    }
}

impl FitUnsupervised for GaussianHmmFull {
    type Fitted = FittedGaussianHmmFull;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedGaussianHmmFull>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        inspect_identification(
            &mut ctx.report,
            x.nrows(),
            self.n_states.max(1),
            &ctx.policy,
        );
        let (t_len, d) = x.shape();
        let k = self.n_states.max(1);
        if t_len == 0 || d == 0 {
            return ctx.finish(FittedGaussianHmmFull {
                labels: empty_labels(t_len),
                n_states: k,
                start: init_start(k),
                trans: init_trans(k),
                means: Matrix::zeros(k, d),
                covs: vec![Matrix::zeros(d, d); k],
                loglik: f64::NAN,
            });
        }
        if series_zero_variance(x, ctx.policy.near_zero_variance) {
            ctx.push(emission_degenerate_issue(
                "full-covariance observation series has zero variance",
            ));
        }
        let mut rng = Rng::new(self.seed | 1);
        let mut means = kmeans_pp_rows(x, k.min(t_len), &mut rng);
        if means.nrows() < k {
            let mut padded = Matrix::zeros(k, d);
            for i in 0..means.nrows() {
                for j in 0..d {
                    padded.set(i, j, means.get(i, j));
                }
            }
            means = padded;
        }
        let gcov = global_full_cov(x);
        let mut covs = vec![gcov.clone(); k];
        let mut start = init_start(k);
        let mut trans = init_trans(k);
        if self.left_right {
            enforce_left_right(&mut start, &mut trans);
        }
        let mut loglik = f64::NEG_INFINITY;
        let mut last_gamma: Vec<Vec<f64>> = Vec::new();
        for it in 0..self.max_iter.max(1) {
            let mut log_emit = vec![vec![0.0; k]; t_len];
            for t in 0..t_len {
                for j in 0..k {
                    log_emit[t][j] = log_full_gauss(x, t, &means, j, &covs[j]);
                }
            }
            let Some(fb) = scaled_forward_backward(&mut ctx, &start, &trans, &log_emit) else {
                break;
            };
            loglik = fb.loglik;
            ctx.session.step(it as u64, -loglik, None);
            last_gamma = fb.gamma.clone();
            for j in 0..k {
                start[j] = fb.gamma[0][j];
            }
            renormalize_vec(&mut start, TRANS_FLOOR);
            if t_len > 1 {
                for i in 0..k {
                    let mut den = 0.0_f64;
                    for t in 0..t_len - 1 {
                        den += fb.gamma[t][i];
                    }
                    for j in 0..k {
                        let mut num = 0.0_f64;
                        for t in 0..t_len - 1 {
                            num += fb.xi[t][i][j];
                        }
                        trans.set(
                            i,
                            j,
                            if den > 0.0 {
                                num / den
                            } else {
                                trans.get(i, j)
                            },
                        );
                    }
                }
                if self.left_right {
                    enforce_left_right(&mut start, &mut trans);
                } else {
                    renormalize_rows(&mut trans, TRANS_FLOOR);
                }
            }
            if self.left_right && k > 1 {
                enforce_left_right(&mut start, &mut trans);
            }
            for j in 0..k {
                let mut nj = 0.0_f64;
                let mut acc = vec![0.0_f64; d];
                for t in 0..t_len {
                    let g = fb.gamma[t][j];
                    nj += g;
                    for c in 0..d {
                        acc[c] += g * x.get(t, c);
                    }
                }
                if nj > TRANS_FLOOR {
                    for c in 0..d {
                        means.set(j, c, acc[c] / nj);
                    }
                    let mut cmat = Matrix::zeros(d, d);
                    for t in 0..t_len {
                        let g = fb.gamma[t][j];
                        for a in 0..d {
                            let da = x.get(t, a) - means.get(j, a);
                            for b in 0..d {
                                cmat.set(
                                    a,
                                    b,
                                    cmat.get(a, b) + g * da * (x.get(t, b) - means.get(j, b)),
                                );
                            }
                        }
                    }
                    for a in 0..d {
                        for b in 0..d {
                            cmat.set(a, b, cmat.get(a, b) / nj);
                        }
                        cmat.set(a, a, cmat.get(a, a).max(COV_FLOOR));
                    }
                    covs[j] = cmat;
                }
            }
        }
        let occup: Vec<f64> = (0..k)
            .map(|j| {
                last_gamma
                    .iter()
                    .map(|g| g.get(j).copied().unwrap_or(0.0))
                    .sum()
            })
            .collect();
        diagnose_chain(&mut ctx, &start, &trans, &occup);
        if self.left_right {
            enforce_left_right(&mut start, &mut trans);
        }
        let fitted_tmp = FittedGaussianHmmFull {
            labels: Vector::zeros(0),
            n_states: k,
            start: start.clone(),
            trans: trans.clone(),
            means: means.clone(),
            covs: covs.clone(),
            loglik,
        };
        let (labels, _) = viterbi_path(&start, &trans, &fitted_tmp.log_emit_seq(x));
        ctx.finish(FittedGaussianHmmFull {
            labels,
            n_states: k,
            start,
            trans,
            means,
            covs,
            loglik,
        })
    }
}

/// Sequence annotator wrapping a diagonal Gaussian HMM (sktime `HMM` annotator).
///
/// State count is not treated as an extra identification `p` here; the inner
/// [`GaussianHmm`] already identifies on `n_states`.
#[derive(Clone, Debug)]
pub struct HmmAnnotator {
    /// Hidden states forwarded to [`GaussianHmm`].
    pub n_states: usize,
}

impl Default for HmmAnnotator {
    fn default() -> Self {
        Self { n_states: 2 }
    }
}

impl HmmAnnotator {
    /// Annotator with `n_states` regimes.
    pub fn new(n_states: usize) -> Self {
        Self {
            n_states: n_states.max(1),
        }
    }
}

/// Fitted HMM annotator.
#[derive(Clone, Debug)]
pub struct FittedHmmAnnotator {
    /// Decoded state path.
    pub labels: Vector,
    /// Underlying Gaussian HMM.
    pub inner: FittedGaussianHmm,
}

impl FitUnsupervised for HmmAnnotator {
    type Fitted = FittedHmmAnnotator;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedHmmAnnotator>> {
        let q = GaussianHmm::new(self.n_states).fit_unsupervised(x, session)?;
        Ok(q.map(|inner| FittedHmmAnnotator {
            labels: inner.labels.clone(),
            inner,
        }))
    }
}

fn log_sph_gauss(x: &Matrix, t: usize, means: &Matrix, state: usize, var: f64) -> f64 {
    let d = x.ncols().min(means.ncols());
    if d == 0 {
        return f64::NEG_INFINITY;
    }
    let v = var.max(COV_FLOOR);
    let mut q = 0.0_f64;
    for j in 0..d {
        let z = x.get(t, j) - means.get(state, j);
        q += z * z;
    }
    -0.5 * (d as f64 * (LN_2PI + v.ln()) + q / v)
}

fn mstep_start_trans(
    start: &mut Vector,
    trans: &mut Matrix,
    gamma: &[Vec<f64>],
    xi: &[Vec<Vec<f64>>],
    left_right: bool,
) {
    let k = start.len();
    let t_len = gamma.len();
    for j in 0..k {
        start[j] = gamma.first().and_then(|g| g.get(j)).copied().unwrap_or(0.0);
    }
    renormalize_vec(start, TRANS_FLOOR);
    if t_len > 1 {
        for i in 0..k {
            let mut den = 0.0_f64;
            for t in 0..t_len - 1 {
                den += gamma[t][i];
            }
            for j in 0..k {
                let mut num = 0.0_f64;
                for t in 0..t_len - 1 {
                    num += xi[t][i][j];
                }
                trans.set(
                    i,
                    j,
                    if den > 0.0 {
                        num / den
                    } else {
                        trans.get(i, j)
                    },
                );
            }
        }
        if left_right {
            enforce_left_right(start, trans);
        } else {
            renormalize_rows(trans, TRANS_FLOOR);
        }
    }
    if left_right && k > 1 {
        enforce_left_right(start, trans);
    }
}

/// Spherical-covariance Gaussian HMM (hmmlearn `covariance_type="spherical"`).
///
/// One shared variance per state. Variance count is not identification `p`.
#[derive(Clone, Debug)]
pub struct GaussianHmmSpherical {
    /// Hidden states.
    pub n_states: usize,
    /// Baum–Welch iteration cap.
    pub max_iter: usize,
    /// Seed.
    pub seed: u64,
    /// Left-right (Bakis) constraint.
    pub left_right: bool,
}

impl Default for GaussianHmmSpherical {
    fn default() -> Self {
        Self {
            n_states: 2,
            max_iter: 50,
            seed: 0,
            left_right: false,
        }
    }
}

impl GaussianHmmSpherical {
    /// `n_states` spherical Gaussians.
    pub fn new(n_states: usize) -> Self {
        Self {
            n_states,
            ..Self::default()
        }
    }

    /// Fit alias.
    pub fn fit(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedGaussianHmmSpherical>> {
        self.fit_unsupervised(x, session)
    }
}

/// Fitted spherical Gaussian HMM.
#[derive(Clone, Debug)]
pub struct FittedGaussianHmmSpherical {
    /// Viterbi path.
    pub labels: Vector,
    /// Number of states.
    pub n_states: usize,
    /// Start distribution.
    pub start: Vector,
    /// Transitions.
    pub trans: Matrix,
    /// Means (`n_states` × `d`).
    pub means: Matrix,
    /// One variance per state.
    pub vars: Vector,
    /// Training log-likelihood.
    pub loglik: f64,
}

impl FittedGaussianHmmSpherical {
    fn log_emit_seq(&self, x: &Matrix) -> Vec<Vec<f64>> {
        let t = x.nrows();
        let s = self.n_states;
        let mut out = vec![vec![f64::NEG_INFINITY; s]; t];
        for ti in 0..t {
            for j in 0..s {
                let v = if j < self.vars.len() {
                    self.vars[j]
                } else {
                    COV_FLOOR
                };
                out[ti][j] = log_sph_gauss(x, ti, &self.means, j, v);
            }
        }
        out
    }

    /// Viterbi path.
    pub fn decode(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("decode"));
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let (path, _) = viterbi_path(&self.start, &self.trans, &self.log_emit_seq(x));
        ctx.finish(path)
    }
}

impl Predict for FittedGaussianHmmSpherical {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        self.decode(x, session)
    }
}

impl FitUnsupervised for GaussianHmmSpherical {
    type Fitted = FittedGaussianHmmSpherical;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedGaussianHmmSpherical>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        inspect_identification(
            &mut ctx.report,
            x.nrows(),
            self.n_states.max(1),
            &ctx.policy,
        );
        let (t_len, d) = x.shape();
        let k = self.n_states.max(1);
        if t_len == 0 || d == 0 {
            return ctx.finish(FittedGaussianHmmSpherical {
                labels: empty_labels(t_len),
                n_states: k,
                start: init_start(k),
                trans: init_trans(k),
                means: Matrix::zeros(k, d),
                vars: Vector::from_iter((0..k).map(|_| COV_FLOOR)),
                loglik: f64::NAN,
            });
        }
        if series_zero_variance(x, ctx.policy.near_zero_variance) {
            ctx.push(emission_degenerate_issue(
                "spherical HMM observation series has zero variance",
            ));
        }
        let mut rng = Rng::new(self.seed | 1);
        let mut means = kmeans_pp_rows(x, k.min(t_len), &mut rng);
        if means.nrows() < k {
            let mut padded = Matrix::zeros(k, d);
            for i in 0..means.nrows() {
                for j in 0..d {
                    padded.set(i, j, means.get(i, j));
                }
            }
            means = padded;
        }
        let gvar = global_diag_var(x);
        let glob: f64 = (0..d).map(|j| gvar[j]).sum::<f64>() / d.max(1) as f64;
        let mut vars = Vector::from_iter((0..k).map(|_| glob.max(COV_FLOOR)));
        let mut start = init_start(k);
        let mut trans = init_trans(k);
        if self.left_right {
            enforce_left_right(&mut start, &mut trans);
        }
        let mut loglik = f64::NEG_INFINITY;
        let mut last_gamma: Vec<Vec<f64>> = Vec::new();
        let mut last_xi: Vec<Vec<Vec<f64>>> = Vec::new();
        for it in 0..self.max_iter.max(1) {
            let mut log_emit = vec![vec![0.0; k]; t_len];
            for t in 0..t_len {
                for j in 0..k {
                    log_emit[t][j] = log_sph_gauss(x, t, &means, j, vars[j]);
                }
            }
            let Some(fb) = scaled_forward_backward(&mut ctx, &start, &trans, &log_emit) else {
                break;
            };
            loglik = fb.loglik;
            ctx.session.step(it as u64, -loglik, None);
            last_gamma = fb.gamma.clone();
            last_xi = fb.xi.clone();
            mstep_start_trans(&mut start, &mut trans, &fb.gamma, &fb.xi, self.left_right);
            for j in 0..k {
                let mut nj = 0.0_f64;
                let mut acc = vec![0.0_f64; d];
                for t in 0..t_len {
                    let g = fb.gamma[t][j];
                    nj += g;
                    for c in 0..d {
                        acc[c] += g * x.get(t, c);
                    }
                }
                if nj <= TRANS_FLOOR {
                    continue;
                }
                for c in 0..d {
                    means.set(j, c, acc[c] / nj);
                }
                let mut q = 0.0_f64;
                for t in 0..t_len {
                    let g = fb.gamma[t][j];
                    for c in 0..d {
                        let z = x.get(t, c) - means.get(j, c);
                        q += g * z * z;
                    }
                }
                vars[j] = (q / (nj * d.max(1) as f64)).max(COV_FLOOR);
            }
        }
        let occup: Vec<f64> = (0..k)
            .map(|j| {
                last_gamma
                    .iter()
                    .map(|g| g.get(j).copied().unwrap_or(0.0))
                    .sum()
            })
            .collect();
        diagnose_chain(&mut ctx, &start, &trans, &occup);
        if self.left_right {
            enforce_left_right(&mut start, &mut trans);
        }
        let _ = last_xi;
        let tmp = FittedGaussianHmmSpherical {
            labels: Vector::zeros(0),
            n_states: k,
            start: start.clone(),
            trans: trans.clone(),
            means: means.clone(),
            vars: vars.clone(),
            loglik,
        };
        let (labels, _) = viterbi_path(&start, &trans, &tmp.log_emit_seq(x));
        ctx.finish(FittedGaussianHmmSpherical {
            labels,
            n_states: k,
            start,
            trans,
            means,
            vars,
            loglik,
        })
    }
}

/// Tied full-covariance Gaussian HMM (hmmlearn `covariance_type="tied"`).
///
/// One shared dense covariance. Free-parameter count is not identification `p`.
#[derive(Clone, Debug)]
pub struct GaussianHmmTied {
    /// Hidden states.
    pub n_states: usize,
    /// Baum–Welch iteration cap.
    pub max_iter: usize,
    /// Seed.
    pub seed: u64,
    /// Left-right constraint.
    pub left_right: bool,
}

impl Default for GaussianHmmTied {
    fn default() -> Self {
        Self {
            n_states: 2,
            max_iter: 50,
            seed: 0,
            left_right: false,
        }
    }
}

impl GaussianHmmTied {
    /// `n_states` Gaussians sharing one covariance.
    pub fn new(n_states: usize) -> Self {
        Self {
            n_states,
            ..Self::default()
        }
    }

    /// Fit alias.
    pub fn fit(&mut self, x: &Matrix, session: &Session) -> Result<Qualified<FittedGaussianHmmTied>> {
        self.fit_unsupervised(x, session)
    }
}

/// Fitted tied-covariance Gaussian HMM.
#[derive(Clone, Debug)]
pub struct FittedGaussianHmmTied {
    /// Viterbi path.
    pub labels: Vector,
    /// Number of states.
    pub n_states: usize,
    /// Start distribution.
    pub start: Vector,
    /// Transitions.
    pub trans: Matrix,
    /// Means (`n_states` × `d`).
    pub means: Matrix,
    /// Shared dense covariance (`d` × `d`).
    pub cov: Matrix,
    /// Training log-likelihood.
    pub loglik: f64,
}

impl FittedGaussianHmmTied {
    fn log_emit_seq(&self, x: &Matrix) -> Vec<Vec<f64>> {
        let t = x.nrows();
        let s = self.n_states;
        let mut out = vec![vec![f64::NEG_INFINITY; s]; t];
        for ti in 0..t {
            for j in 0..s {
                out[ti][j] = log_full_gauss(x, ti, &self.means, j, &self.cov);
            }
        }
        out
    }

    /// Viterbi path.
    pub fn decode(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("decode"));
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let (path, _) = viterbi_path(&self.start, &self.trans, &self.log_emit_seq(x));
        ctx.finish(path)
    }
}

impl Predict for FittedGaussianHmmTied {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        self.decode(x, session)
    }
}

impl FitUnsupervised for GaussianHmmTied {
    type Fitted = FittedGaussianHmmTied;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedGaussianHmmTied>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        inspect_identification(
            &mut ctx.report,
            x.nrows(),
            self.n_states.max(1),
            &ctx.policy,
        );
        let (t_len, d) = x.shape();
        let k = self.n_states.max(1);
        if t_len == 0 || d == 0 {
            return ctx.finish(FittedGaussianHmmTied {
                labels: empty_labels(t_len),
                n_states: k,
                start: init_start(k),
                trans: init_trans(k),
                means: Matrix::zeros(k, d),
                cov: Matrix::zeros(d, d),
                loglik: f64::NAN,
            });
        }
        if series_zero_variance(x, ctx.policy.near_zero_variance) {
            ctx.push(emission_degenerate_issue(
                "tied HMM observation series has zero variance",
            ));
        }
        let mut rng = Rng::new(self.seed | 1);
        let mut means = kmeans_pp_rows(x, k.min(t_len), &mut rng);
        if means.nrows() < k {
            let mut padded = Matrix::zeros(k, d);
            for i in 0..means.nrows() {
                for j in 0..d {
                    padded.set(i, j, means.get(i, j));
                }
            }
            means = padded;
        }
        let mut cov = global_full_cov(x);
        let mut start = init_start(k);
        let mut trans = init_trans(k);
        if self.left_right {
            enforce_left_right(&mut start, &mut trans);
        }
        let mut loglik = f64::NEG_INFINITY;
        let mut last_gamma: Vec<Vec<f64>> = Vec::new();
        for it in 0..self.max_iter.max(1) {
            let mut log_emit = vec![vec![0.0; k]; t_len];
            for t in 0..t_len {
                for j in 0..k {
                    log_emit[t][j] = log_full_gauss(x, t, &means, j, &cov);
                }
            }
            let Some(fb) = scaled_forward_backward(&mut ctx, &start, &trans, &log_emit) else {
                break;
            };
            loglik = fb.loglik;
            ctx.session.step(it as u64, -loglik, None);
            last_gamma = fb.gamma.clone();
            mstep_start_trans(&mut start, &mut trans, &fb.gamma, &fb.xi, self.left_right);
            for j in 0..k {
                let mut nj = 0.0_f64;
                let mut acc = vec![0.0_f64; d];
                for t in 0..t_len {
                    let g = fb.gamma[t][j];
                    nj += g;
                    for c in 0..d {
                        acc[c] += g * x.get(t, c);
                    }
                }
                if nj > TRANS_FLOOR {
                    for c in 0..d {
                        means.set(j, c, acc[c] / nj);
                    }
                }
            }
            let mut cmat = Matrix::zeros(d, d);
            let mut ntot = 0.0_f64;
            for j in 0..k {
                for t in 0..t_len {
                    let g = fb.gamma[t][j];
                    ntot += g;
                    for a in 0..d {
                        let da = x.get(t, a) - means.get(j, a);
                        for b in 0..d {
                            cmat.set(
                                a,
                                b,
                                cmat.get(a, b) + g * da * (x.get(t, b) - means.get(j, b)),
                            );
                        }
                    }
                }
            }
            if ntot > TRANS_FLOOR {
                for a in 0..d {
                    for b in 0..d {
                        cmat.set(a, b, cmat.get(a, b) / ntot);
                    }
                    cmat.set(a, a, cmat.get(a, a).max(COV_FLOOR));
                }
                cov = cmat;
            }
        }
        let occup: Vec<f64> = (0..k)
            .map(|j| {
                last_gamma
                    .iter()
                    .map(|g| g.get(j).copied().unwrap_or(0.0))
                    .sum()
            })
            .collect();
        diagnose_chain(&mut ctx, &start, &trans, &occup);
        if self.left_right {
            enforce_left_right(&mut start, &mut trans);
        }
        let tmp = FittedGaussianHmmTied {
            labels: Vector::zeros(0),
            n_states: k,
            start: start.clone(),
            trans: trans.clone(),
            means: means.clone(),
            cov: cov.clone(),
            loglik,
        };
        let (labels, _) = viterbi_path(&start, &trans, &tmp.log_emit_seq(x));
        ctx.finish(FittedGaussianHmmTied {
            labels,
            n_states: k,
            start,
            trans,
            means,
            cov,
            loglik,
        })
    }
}

/// Discrete-emission HMM. Observation codes are the rounded entries of `X`
/// (typically a `T` × 1 matrix of integer symbols).
///
/// [`CategoricalHmm`] is the hmmlearn 0.3+ name for the same model.
#[derive(Clone, Debug)]
pub struct MultinomialHmm {
    /// Number of hidden states.
    pub n_states: usize,
    /// Baum–Welch iteration cap.
    pub max_iter: usize,
    /// Seed for emission jitter.
    pub seed: u64,
    /// If true, transitions to earlier states are zeroed (hmmlearn `left_right`).
    pub left_right: bool,
}

impl Default for MultinomialHmm {
    fn default() -> Self {
        Self {
            n_states: 2,
            max_iter: 50,
            seed: 0,
            left_right: false,
        }
    }
}

/// hmmlearn 0.3+ name for [`MultinomialHmm`].
pub type CategoricalHmm = MultinomialHmm;

impl MultinomialHmm {
    /// `n_states` discrete-emission states.
    pub fn new(n_states: usize) -> Self {
        Self {
            n_states,
            ..Self::default()
        }
    }

    /// Left-right multinomial HMM with `n_states` states.
    pub fn left_right(n_states: usize) -> Self {
        Self {
            n_states,
            left_right: true,
            ..Self::default()
        }
    }

    /// Fit alias.
    pub fn fit(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedMultinomialHmm>> {
        self.fit_unsupervised(x, session)
    }
}

/// Fitted multinomial HMM.
#[derive(Clone, Debug)]
pub struct FittedMultinomialHmm {
    /// Viterbi path on the training codes.
    pub labels: Vector,
    /// Number of states.
    pub n_states: usize,
    /// Start distribution.
    pub start: Vector,
    /// Transitions.
    pub trans: Matrix,
    /// Emission probabilities (`n_states` × `n_symbols`), using column-0 codes.
    pub emission: Matrix,
    /// Training log-likelihood.
    pub loglik: f64,
}

fn codes_from_x(x: &Matrix) -> (Vec<usize>, usize) {
    let mut codes = Vec::with_capacity(x.nrows());
    let mut max_c = 0usize;
    for i in 0..x.nrows() {
        let v = x.get(i, 0);
        let c = if v.is_finite() && v >= 0.0 {
            v.round() as usize
        } else {
            0
        };
        if c > max_c {
            max_c = c;
        }
        codes.push(c);
    }
    (codes, max_c + 1)
}

impl FittedMultinomialHmm {
    fn log_emit_seq(&self, codes: &[usize]) -> Vec<Vec<f64>> {
        let s = self.n_states;
        let mut out = vec![vec![f64::NEG_INFINITY; s]; codes.len()];
        for (t, &c) in codes.iter().enumerate() {
            for j in 0..s {
                let p = if j < self.emission.nrows() && c < self.emission.ncols() {
                    self.emission.get(j, c)
                } else {
                    0.0
                };
                out[t][j] = if p > 0.0 { p.ln() } else { f64::NEG_INFINITY };
            }
        }
        out
    }

    /// Viterbi path for integer codes stored in column 0 of `x`.
    pub fn decode(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("decode"));
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let (codes, _) = codes_from_x(x);
        let (path, _) = viterbi_path(&self.start, &self.trans, &self.log_emit_seq(&codes));
        ctx.finish(path)
    }

    /// Scaled-forward log-likelihood.
    pub fn score(&self, x: &Matrix, session: &Session) -> Result<Qualified<f64>> {
        let mut ctx = FitCtx::with_session(session.child("score"));
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let (codes, _) = codes_from_x(x);
        let fb = scaled_forward_backward(
            &mut ctx,
            &self.start,
            &self.trans,
            &self.log_emit_seq(&codes),
        );
        ctx.finish(fb.map(|f| f.loglik).unwrap_or(f64::NEG_INFINITY))
    }

    /// Sample integer codes (column 0) and the hidden path.
    pub fn sample(&self, n: usize, seed: u64, session: &Session) -> Result<Qualified<HmmSample>> {
        let ctx = FitCtx::with_session(session.child("sample"));
        let n_sym = self.emission.ncols();
        let s = self.n_states.max(1);
        if n == 0 {
            return ctx.finish(HmmSample {
                obs: Matrix::zeros(0, 1),
                states: Vector::zeros(0),
            });
        }
        let mut rng = Rng::new(seed | 1);
        let mut states = Vector::zeros(n);
        let mut obs = Matrix::zeros(n, 1);
        let pick = |rng: &mut Rng, row_i: usize, mat: &Matrix| -> usize {
            let cols = mat.ncols();
            let mut u = rng.uniform();
            for j in 0..cols {
                u -= mat.get(row_i, j);
                if u <= 0.0 {
                    return j;
                }
            }
            cols.saturating_sub(1)
        };
        let pick_v = |rng: &mut Rng, v: &Vector| -> usize {
            let mut u = rng.uniform();
            for i in 0..v.len() {
                u -= v[i];
                if u <= 0.0 {
                    return i;
                }
            }
            v.len().saturating_sub(1)
        };
        let mut st = pick_v(&mut rng, &self.start);
        for t in 0..n {
            states[t] = st as f64;
            let sym = if n_sym == 0 {
                0
            } else {
                pick(
                    &mut rng,
                    st.min(self.emission.nrows().saturating_sub(1)),
                    &self.emission,
                )
            };
            obs.set(t, 0, sym as f64);
            if t + 1 < n {
                let row = Vector::from_iter((0..s).map(|k| {
                    self.trans
                        .get(st.min(self.trans.nrows().saturating_sub(1)), k)
                }));
                st = pick_v(&mut rng, &row);
            }
        }
        ctx.finish(HmmSample { obs, states })
    }
}

impl Predict for FittedMultinomialHmm {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        self.decode(x, session)
    }
}

impl FitUnsupervised for MultinomialHmm {
    type Fitted = FittedMultinomialHmm;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedMultinomialHmm>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        inspect_identification(
            &mut ctx.report,
            x.nrows(),
            self.n_states.max(1),
            &ctx.policy,
        );
        let t_len = x.nrows();
        let k = self.n_states.max(1);
        if t_len == 0 || x.ncols() == 0 {
            return ctx.finish(FittedMultinomialHmm {
                labels: empty_labels(0),
                n_states: k,
                start: init_start(k),
                trans: init_trans(k),
                emission: Matrix::zeros(k, 1),
                loglik: f64::NAN,
            });
        }
        let (codes, n_sym) = codes_from_x(x);
        if n_sym <= 1 {
            ctx.push(emission_degenerate_issue(
                "multinomial HMM saw only one symbol; emissions are a delta",
            ));
        }
        let mut rng = Rng::new(self.seed | 1);
        let mut start = init_start(k);
        let mut trans = init_trans(k);
        if self.left_right {
            enforce_left_right(&mut start, &mut trans);
        }
        let mut emission = Matrix::from_fn(k, n_sym, |_, _| rng.uniform() + 0.1);
        renormalize_rows(&mut emission, TRANS_FLOOR);
        let mut loglik = f64::NEG_INFINITY;
        let mut last_gamma = Vec::new();
        for it in 0..self.max_iter.max(1) {
            let log_emit: Vec<Vec<f64>> = codes
                .iter()
                .map(|&c| {
                    (0..k)
                        .map(|j| {
                            let p = emission.get(j, c.min(n_sym.saturating_sub(1)));
                            if p > 0.0 {
                                p.ln()
                            } else {
                                f64::NEG_INFINITY
                            }
                        })
                        .collect()
                })
                .collect();
            let Some(fb) = scaled_forward_backward(&mut ctx, &start, &trans, &log_emit) else {
                break;
            };
            loglik = fb.loglik;
            last_gamma = fb.gamma.clone();
            ctx.session.step(it as u64, -loglik, None);
            for j in 0..k {
                start[j] = fb.gamma[0][j];
            }
            renormalize_vec(&mut start, TRANS_FLOOR);
            if t_len > 1 {
                for i in 0..k {
                    let den: f64 = (0..t_len - 1).map(|t| fb.gamma[t][i]).sum();
                    for j in 0..k {
                        let num: f64 = (0..t_len - 1).map(|t| fb.xi[t][i][j]).sum();
                        trans.set(
                            i,
                            j,
                            if den > 0.0 {
                                num / den
                            } else {
                                trans.get(i, j)
                            },
                        );
                    }
                }
                if self.left_right {
                    enforce_left_right(&mut start, &mut trans);
                } else {
                    renormalize_rows(&mut trans, TRANS_FLOOR);
                }
            }
            if self.left_right && k > 1 {
                enforce_left_right(&mut start, &mut trans);
            }
            for j in 0..k {
                let mut counts = vec![0.0; n_sym];
                let mut den = 0.0;
                for t in 0..t_len {
                    let g = fb.gamma[t][j];
                    den += g;
                    counts[codes[t]] += g;
                }
                if den <= TRANS_FLOOR {
                    ctx.push(
                        Issue::builder(IssueCode::UnreachableState)
                            .message(format!("multinomial state {j} has ~0 mass"))
                            .metric("state", j as f64)
                            .build(),
                    );
                    continue;
                }
                for c in 0..n_sym {
                    emission.set(j, c, counts[c] / den);
                }
            }
            renormalize_rows(&mut emission, TRANS_FLOOR);
            if it + 1 == self.max_iter {
                ctx.push(
                    Issue::builder(IssueCode::MaxIterReached)
                        .message("multinomial Baum–Welch hit max_iter")
                        .build(),
                );
            }
        }
        let occup: Vec<f64> = (0..k)
            .map(|j| {
                last_gamma
                    .iter()
                    .map(|g| g.get(j).copied().unwrap_or(0.0))
                    .sum()
            })
            .collect();
        diagnose_chain(&mut ctx, &start, &trans, &occup);
        if self.left_right {
            enforce_left_right(&mut start, &mut trans);
        }
        let log_emit = FittedMultinomialHmm {
            labels: Vector::zeros(0),
            n_states: k,
            start: start.clone(),
            trans: trans.clone(),
            emission: emission.clone(),
            loglik,
        }
        .log_emit_seq(&codes);
        let (labels, _) = viterbi_path(&start, &trans, &log_emit);
        ctx.finish(FittedMultinomialHmm {
            labels,
            n_states: k,
            start,
            trans,
            emission,
            loglik,
        })
    }
}

/// HMM whose emissions are a diagonal Gaussian mixture per state.
///
/// `n_mix == 1` is a diagonal [`GaussianHmm`].
#[derive(Clone, Debug)]
pub struct GmmHmm {
    /// Number of hidden states.
    pub n_states: usize,
    /// Mixture components per state.
    pub n_mix: usize,
    /// Baum–Welch iteration cap.
    pub max_iter: usize,
    /// Seed.
    pub seed: u64,
    /// If true, transitions to earlier states are zeroed (hmmlearn `left_right`).
    pub left_right: bool,
    /// If true, each mixture component shares one variance across dimensions.
    pub spherical: bool,
}

impl Default for GmmHmm {
    fn default() -> Self {
        Self {
            n_states: 2,
            n_mix: 1,
            max_iter: 40,
            seed: 0,
            left_right: false,
            spherical: false,
        }
    }
}

impl GmmHmm {
    /// `n_states` states with `n_mix` diagonal Gaussians each.
    pub fn new(n_states: usize, n_mix: usize) -> Self {
        Self {
            n_states,
            n_mix,
            ..Self::default()
        }
    }

    /// Left-right GMM-HMM with `n_states` states and `n_mix` mixtures.
    pub fn left_right(n_states: usize, n_mix: usize) -> Self {
        Self {
            n_states,
            n_mix,
            left_right: true,
            ..Self::default()
        }
    }

    /// Spherical-covariance GMM-HMM (one variance per mixture component).
    pub fn spherical(n_states: usize, n_mix: usize) -> Self {
        Self {
            n_states,
            n_mix,
            spherical: true,
            ..Self::default()
        }
    }

    /// Fit alias.
    pub fn fit(&mut self, x: &Matrix, session: &Session) -> Result<Qualified<FittedGmmHmm>> {
        self.fit_unsupervised(x, session)
    }
}

/// Fitted GMM-HMM.
#[derive(Clone, Debug)]
pub struct FittedGmmHmm {
    /// Viterbi path.
    pub labels: Vector,
    /// Number of states.
    pub n_states: usize,
    /// Mixture size per state.
    pub n_mix: usize,
    /// Start distribution.
    pub start: Vector,
    /// Transitions.
    pub trans: Matrix,
    /// Mixture weights (`n_states` × `n_mix`).
    pub mix_weights: Matrix,
    /// Means (`n_states * n_mix` × `d`), row `s * n_mix + m`.
    pub means: Matrix,
    /// Diagonal variances, same layout as `means`.
    pub vars: Matrix,
    /// Training log-likelihood.
    pub loglik: f64,
}

impl FittedGmmHmm {
    fn mix_row(state: usize, mix: usize, n_mix: usize) -> usize {
        state * n_mix + mix
    }

    fn log_state_emit(&self, x: &Matrix, t: usize, state: usize) -> f64 {
        let mut parts = vec![0.0; self.n_mix.max(1)];
        for m in 0..self.n_mix.max(1) {
            let w = self.mix_weights.get(state, m).max(1e-300).ln();
            let r = Self::mix_row(state, m, self.n_mix.max(1));
            parts[m] = w + log_diag_gauss(x, t, &self.means, r, &self.vars);
        }
        logsumexp(&parts)
    }

    fn log_emit_seq(&self, x: &Matrix) -> Vec<Vec<f64>> {
        (0..x.nrows())
            .map(|t| {
                (0..self.n_states)
                    .map(|s| self.log_state_emit(x, t, s))
                    .collect()
            })
            .collect()
    }

    /// Viterbi path.
    pub fn decode(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("decode"));
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        if series_zero_variance(x, ctx.policy.near_zero_variance) {
            ctx.push(emission_degenerate_issue(
                "GmmHmm decode sequence has zero variance",
            ));
        }
        let (path, _) = viterbi_path(&self.start, &self.trans, &self.log_emit_seq(x));
        ctx.finish(path)
    }

    /// Scaled-forward log-likelihood.
    pub fn score(&self, x: &Matrix, session: &Session) -> Result<Qualified<f64>> {
        let mut ctx = FitCtx::with_session(session.child("score"));
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let fb = scaled_forward_backward(&mut ctx, &self.start, &self.trans, &self.log_emit_seq(x));
        ctx.finish(fb.map(|f| f.loglik).unwrap_or(f64::NEG_INFINITY))
    }

    /// Sample observations and states (mixture component is marginalized in the draw).
    pub fn sample(&self, n: usize, seed: u64, session: &Session) -> Result<Qualified<HmmSample>> {
        let ctx = FitCtx::with_session(session.child("sample"));
        let d = self.means.ncols();
        let s = self.n_states.max(1);
        let nm = self.n_mix.max(1);
        if n == 0 {
            return ctx.finish(HmmSample {
                obs: Matrix::zeros(0, d),
                states: Vector::zeros(0),
            });
        }
        let mut rng = Rng::new(seed | 1);
        let pick_v = |rng: &mut Rng, v: &Vector| -> usize {
            let mut u = rng.uniform();
            for i in 0..v.len() {
                u -= v[i];
                if u <= 0.0 {
                    return i;
                }
            }
            v.len().saturating_sub(1)
        };
        let mut states = Vector::zeros(n);
        let mut obs = Matrix::zeros(n, d);
        let mut st = pick_v(&mut rng, &self.start);
        for t in 0..n {
            states[t] = st as f64;
            let mw = Vector::from_iter((0..nm).map(|m| {
                self.mix_weights
                    .get(st.min(self.mix_weights.nrows().saturating_sub(1)), m)
            }));
            let m = pick_v(&mut rng, &mw);
            let row = Self::mix_row(st, m, nm);
            for j in 0..d {
                let sd = self
                    .vars
                    .get(row.min(self.vars.nrows().saturating_sub(1)), j)
                    .max(COV_FLOOR)
                    .sqrt();
                obs.set(
                    t,
                    j,
                    self.means
                        .get(row.min(self.means.nrows().saturating_sub(1)), j)
                        + sd * rng.standard_normal(),
                );
            }
            if t + 1 < n {
                let rowt = Vector::from_iter((0..s).map(|k| {
                    self.trans
                        .get(st.min(self.trans.nrows().saturating_sub(1)), k)
                }));
                st = pick_v(&mut rng, &rowt);
            }
        }
        ctx.finish(HmmSample { obs, states })
    }
}

impl Predict for FittedGmmHmm {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        self.decode(x, session)
    }
}

impl FitUnsupervised for GmmHmm {
    type Fitted = FittedGmmHmm;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedGmmHmm>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        inspect_identification(
            &mut ctx.report,
            x.nrows(),
            self.n_states.max(1),
            &ctx.policy,
        );
        let (t_len, d) = x.shape();
        let k = self.n_states.max(1);
        let nm = self.n_mix.max(1);
        if t_len == 0 || d == 0 {
            let mut start = init_start(k);
            let mut trans = init_trans(k);
            if self.left_right {
                enforce_left_right(&mut start, &mut trans);
            }
            return ctx.finish(FittedGmmHmm {
                labels: empty_labels(0),
                n_states: k,
                n_mix: nm,
                start,
                trans,
                mix_weights: Matrix::zeros(k, nm),
                means: Matrix::zeros(k * nm, d),
                vars: Matrix::zeros(k * nm, d),
                loglik: f64::NAN,
            });
        }
        if series_zero_variance(x, ctx.policy.near_zero_variance) {
            ctx.push(emission_degenerate_issue(
                "GmmHmm observation series has zero variance",
            ));
        }
        let mut rng = Rng::new(self.seed | 1);
        let seeds = kmeans_pp_rows(x, (k * nm).min(t_len), &mut rng);
        let mut means = Matrix::zeros(k * nm, d);
        for r in 0..(k * nm).min(seeds.nrows()) {
            for j in 0..d {
                means.set(r, j, seeds.get(r, j));
            }
        }
        let gvar = global_diag_var(x);
        let mut vars = Matrix::from_fn(k * nm, d, |_, j| gvar[j]);
        let mut mix_weights = Matrix::from_fn(k, nm, |_, _| 1.0 / nm as f64);
        let mut start = init_start(k);
        let mut trans = init_trans(k);
        if self.left_right {
            enforce_left_right(&mut start, &mut trans);
        }
        let mut loglik = f64::NEG_INFINITY;
        let mut last_gamma = Vec::new();
        for it in 0..self.max_iter.max(1) {
            // Per-time, per-state, per-mix log densities.
            let mut log_emit = vec![vec![0.0; k]; t_len];
            let mut log_comp = vec![vec![vec![0.0; nm]; k]; t_len];
            for t in 0..t_len {
                for j in 0..k {
                    let mut parts = vec![0.0; nm];
                    for m in 0..nm {
                        let r = j * nm + m;
                        let lp = mix_weights.get(j, m).max(1e-300).ln()
                            + log_diag_gauss(x, t, &means, r, &vars);
                        log_comp[t][j][m] = lp;
                        parts[m] = lp;
                    }
                    log_emit[t][j] = logsumexp(&parts);
                }
            }
            let Some(fb) = scaled_forward_backward(&mut ctx, &start, &trans, &log_emit) else {
                break;
            };
            loglik = fb.loglik;
            last_gamma = fb.gamma.clone();
            ctx.session.step(it as u64, -loglik, None);
            // Component responsibilities γ_t(j,m) = γ_t(j) * softmax_m log_comp
            let mut gtm = vec![vec![vec![0.0; nm]; k]; t_len];
            for t in 0..t_len {
                for j in 0..k {
                    let lse = logsumexp(&log_comp[t][j]);
                    for m in 0..nm {
                        let p = if lse.is_finite() {
                            (log_comp[t][j][m] - lse).exp()
                        } else {
                            1.0 / nm as f64
                        };
                        gtm[t][j][m] = fb.gamma[t][j] * p;
                    }
                }
            }
            for j in 0..k {
                start[j] = fb.gamma[0][j];
            }
            renormalize_vec(&mut start, TRANS_FLOOR);
            if t_len > 1 {
                for i in 0..k {
                    let den: f64 = (0..t_len - 1).map(|t| fb.gamma[t][i]).sum();
                    for j in 0..k {
                        let num: f64 = (0..t_len - 1).map(|t| fb.xi[t][i][j]).sum();
                        trans.set(
                            i,
                            j,
                            if den > 0.0 {
                                num / den
                            } else {
                                trans.get(i, j)
                            },
                        );
                    }
                }
                if self.left_right {
                    enforce_left_right(&mut start, &mut trans);
                } else {
                    renormalize_rows(&mut trans, TRANS_FLOOR);
                }
            }
            if self.left_right && k > 1 {
                enforce_left_right(&mut start, &mut trans);
            }
            for j in 0..k {
                let mut wsum = 0.0;
                for m in 0..nm {
                    let mut nj = 0.0;
                    for t in 0..t_len {
                        nj += gtm[t][j][m];
                    }
                    mix_weights.set(j, m, nj);
                    wsum += nj;
                    if nj <= TRANS_FLOOR {
                        ctx.push(
                            Issue::builder(IssueCode::MixtureWeightCollapsed)
                                .message(format!("state {j} mix {m} collapsed"))
                                .metric("state", j as f64)
                                .metric("mix", m as f64)
                                .build(),
                        );
                        continue;
                    }
                    let r = j * nm + m;
                    for dim in 0..d {
                        let mut mu = 0.0;
                        for t in 0..t_len {
                            mu += gtm[t][j][m] * x.get(t, dim);
                        }
                        means.set(r, dim, mu / nj);
                    }
                    for dim in 0..d {
                        let mut s = 0.0;
                        for t in 0..t_len {
                            let z = x.get(t, dim) - means.get(r, dim);
                            s += gtm[t][j][m] * z * z;
                        }
                        vars.set(r, dim, (s / nj).max(COV_FLOOR));
                    }
                    if self.spherical && d > 0 {
                        let mut acc = 0.0_f64;
                        for dim in 0..d {
                            acc += vars.get(r, dim);
                        }
                        let shared = (acc / d as f64).max(COV_FLOOR);
                        for dim in 0..d {
                            vars.set(r, dim, shared);
                        }
                    }
                }
                if wsum > 0.0 {
                    for m in 0..nm {
                        mix_weights.set(j, m, mix_weights.get(j, m) / wsum);
                    }
                } else {
                    for m in 0..nm {
                        mix_weights.set(j, m, 1.0 / nm as f64);
                    }
                }
            }
            if it + 1 == self.max_iter {
                ctx.push(
                    Issue::builder(IssueCode::MaxIterReached)
                        .message("GmmHmm Baum–Welch hit max_iter")
                        .build(),
                );
            }
        }
        let occup: Vec<f64> = (0..k)
            .map(|j| {
                last_gamma
                    .iter()
                    .map(|g| g.get(j).copied().unwrap_or(0.0))
                    .sum()
            })
            .collect();
        diagnose_chain(&mut ctx, &start, &trans, &occup);
        if self.left_right {
            enforce_left_right(&mut start, &mut trans);
        }
        let fitted_tmp = FittedGmmHmm {
            labels: Vector::zeros(0),
            n_states: k,
            n_mix: nm,
            start: start.clone(),
            trans: trans.clone(),
            mix_weights: mix_weights.clone(),
            means: means.clone(),
            vars: vars.clone(),
            loglik,
        };
        let (labels, _) = viterbi_path(&start, &trans, &fitted_tmp.log_emit_seq(x));
        ctx.finish(FittedGmmHmm {
            labels,
            n_states: k,
            n_mix: nm,
            start,
            trans,
            mix_weights,
            means,
            vars,
            loglik,
        })
    }
}

/// Spherical-covariance GMM-HMM (hmmlearn `GMMHMM` with `covariance_type="spherical"`).
pub type GmmHmmSpherical = GmmHmm;

fn mix_index(state: usize, mix: usize, n_mix: usize) -> usize {
    state * n_mix.max(1) + mix
}

/// GMM-HMM with a full covariance per mixture component (hmmlearn `GMMHMM`, `full`).
///
/// Covariance free-parameter count and `n_mix` are not identification `p`.
#[derive(Clone, Debug)]
pub struct GmmHmmFull {
    /// Number of hidden states.
    pub n_states: usize,
    /// Mixture components per state.
    pub n_mix: usize,
    /// Baum–Welch iteration cap.
    pub max_iter: usize,
    /// Seed.
    pub seed: u64,
    /// If true, transitions to earlier states are zeroed (hmmlearn `left_right`).
    pub left_right: bool,
}

impl Default for GmmHmmFull {
    fn default() -> Self {
        Self {
            n_states: 2,
            n_mix: 1,
            max_iter: 40,
            seed: 0,
            left_right: false,
        }
    }
}

impl GmmHmmFull {
    /// `n_states` states with `n_mix` full-covariance Gaussians each.
    pub fn new(n_states: usize, n_mix: usize) -> Self {
        Self {
            n_states,
            n_mix,
            ..Self::default()
        }
    }

    /// Fit alias.
    pub fn fit(&mut self, x: &Matrix, session: &Session) -> Result<Qualified<FittedGmmHmmFull>> {
        self.fit_unsupervised(x, session)
    }
}

/// Fitted full-covariance GMM-HMM.
#[derive(Clone, Debug)]
pub struct FittedGmmHmmFull {
    /// Viterbi path.
    pub labels: Vector,
    /// Number of states.
    pub n_states: usize,
    /// Mixture size per state.
    pub n_mix: usize,
    /// Start distribution.
    pub start: Vector,
    /// Transitions.
    pub trans: Matrix,
    /// Mixture weights (`n_states` × `n_mix`).
    pub mix_weights: Matrix,
    /// Means (`n_states * n_mix` × `d`), row `s * n_mix + m`.
    pub means: Matrix,
    /// Per-component dense covariances, same row layout as `means`.
    pub covs: Vec<Matrix>,
    /// Training log-likelihood.
    pub loglik: f64,
}

impl FittedGmmHmmFull {
    fn log_state_emit(&self, x: &Matrix, t: usize, state: usize) -> f64 {
        let nm = self.n_mix.max(1);
        let mut parts = vec![0.0; nm];
        for m in 0..nm {
            let w = self.mix_weights.get(state, m).max(1e-300).ln();
            let r = mix_index(state, m, nm);
            let cov = self
                .covs
                .get(r)
                .cloned()
                .unwrap_or_else(|| Matrix::zeros(0, 0));
            parts[m] = w + log_full_gauss(x, t, &self.means, r, &cov);
        }
        logsumexp(&parts)
    }

    fn log_emit_seq(&self, x: &Matrix) -> Vec<Vec<f64>> {
        (0..x.nrows())
            .map(|t| {
                (0..self.n_states)
                    .map(|s| self.log_state_emit(x, t, s))
                    .collect()
            })
            .collect()
    }

    /// Viterbi path.
    pub fn decode(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("decode"));
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let (path, _) = viterbi_path(&self.start, &self.trans, &self.log_emit_seq(x));
        ctx.finish(path)
    }

    /// Scaled-forward log-likelihood.
    pub fn score(&self, x: &Matrix, session: &Session) -> Result<Qualified<f64>> {
        let mut ctx = FitCtx::with_session(session.child("score"));
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let fb = scaled_forward_backward(&mut ctx, &self.start, &self.trans, &self.log_emit_seq(x));
        ctx.finish(fb.map(|f| f.loglik).unwrap_or(f64::NEG_INFINITY))
    }
}

impl Predict for FittedGmmHmmFull {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        self.decode(x, session)
    }
}

impl FitUnsupervised for GmmHmmFull {
    type Fitted = FittedGmmHmmFull;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedGmmHmmFull>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        inspect_identification(
            &mut ctx.report,
            x.nrows(),
            self.n_states.max(1),
            &ctx.policy,
        );
        let (t_len, d) = x.shape();
        let k = self.n_states.max(1);
        let nm = self.n_mix.max(1);
        if t_len == 0 || d == 0 {
            let mut start = init_start(k);
            let mut trans = init_trans(k);
            if self.left_right {
                enforce_left_right(&mut start, &mut trans);
            }
            return ctx.finish(FittedGmmHmmFull {
                labels: empty_labels(0),
                n_states: k,
                n_mix: nm,
                start,
                trans,
                mix_weights: Matrix::zeros(k, nm),
                means: Matrix::zeros(k * nm, d),
                covs: vec![Matrix::zeros(d, d); k * nm],
                loglik: f64::NAN,
            });
        }
        if series_zero_variance(x, ctx.policy.near_zero_variance) {
            ctx.push(emission_degenerate_issue(
                "full-covariance GMM-HMM observation series has zero variance",
            ));
        }
        let mut rng = Rng::new(self.seed | 5);
        let seeds = kmeans_pp_rows(x, (k * nm).min(t_len), &mut rng);
        let mut means = Matrix::zeros(k * nm, d);
        for r in 0..(k * nm).min(seeds.nrows()) {
            for j in 0..d {
                means.set(r, j, seeds.get(r, j));
            }
        }
        let gcov = global_full_cov(x);
        let mut covs = vec![gcov.clone(); k * nm];
        let mut mix_weights = Matrix::from_fn(k, nm, |_, _| 1.0 / nm as f64);
        let mut start = init_start(k);
        let mut trans = init_trans(k);
        if self.left_right {
            enforce_left_right(&mut start, &mut trans);
        }
        let mut loglik = f64::NEG_INFINITY;
        let mut last_gamma = Vec::new();
        for it in 0..self.max_iter.max(1) {
            let mut log_emit = vec![vec![0.0; k]; t_len];
            let mut log_comp = vec![vec![vec![0.0; nm]; k]; t_len];
            for t in 0..t_len {
                for j in 0..k {
                    let mut parts = vec![0.0; nm];
                    for m in 0..nm {
                        let r = mix_index(j, m, nm);
                        let lp = mix_weights.get(j, m).max(1e-300).ln()
                            + log_full_gauss(x, t, &means, r, &covs[r]);
                        log_comp[t][j][m] = lp;
                        parts[m] = lp;
                    }
                    log_emit[t][j] = logsumexp(&parts);
                }
            }
            let Some(fb) = scaled_forward_backward(&mut ctx, &start, &trans, &log_emit) else {
                break;
            };
            loglik = fb.loglik;
            last_gamma = fb.gamma.clone();
            ctx.session.step(it as u64, -loglik, None);
            let mut gtm = vec![vec![vec![0.0; nm]; k]; t_len];
            for t in 0..t_len {
                for j in 0..k {
                    let lse = logsumexp(&log_comp[t][j]);
                    for m in 0..nm {
                        let p = if lse.is_finite() {
                            (log_comp[t][j][m] - lse).exp()
                        } else {
                            1.0 / nm as f64
                        };
                        gtm[t][j][m] = fb.gamma[t][j] * p;
                    }
                }
            }
            for j in 0..k {
                start[j] = fb.gamma[0][j];
            }
            renormalize_vec(&mut start, TRANS_FLOOR);
            if t_len > 1 {
                for i in 0..k {
                    let den: f64 = (0..t_len - 1).map(|t| fb.gamma[t][i]).sum();
                    for j in 0..k {
                        let num: f64 = (0..t_len - 1).map(|t| fb.xi[t][i][j]).sum();
                        trans.set(
                            i,
                            j,
                            if den > 0.0 {
                                num / den
                            } else {
                                trans.get(i, j)
                            },
                        );
                    }
                }
                if self.left_right {
                    enforce_left_right(&mut start, &mut trans);
                } else {
                    renormalize_rows(&mut trans, TRANS_FLOOR);
                }
            }
            if self.left_right && k > 1 {
                enforce_left_right(&mut start, &mut trans);
            }
            for j in 0..k {
                let mut wsum = 0.0;
                for m in 0..nm {
                    let mut nj = 0.0;
                    for t in 0..t_len {
                        nj += gtm[t][j][m];
                    }
                    mix_weights.set(j, m, nj);
                    wsum += nj;
                    if nj <= TRANS_FLOOR {
                        ctx.push(
                            Issue::builder(IssueCode::MixtureWeightCollapsed)
                                .message(format!("full GMM-HMM state {j} mix {m} collapsed"))
                                .metric("state", j as f64)
                                .metric("mix", m as f64)
                                .build(),
                        );
                        continue;
                    }
                    let r = mix_index(j, m, nm);
                    for dim in 0..d {
                        let mut mu = 0.0;
                        for t in 0..t_len {
                            mu += gtm[t][j][m] * x.get(t, dim);
                        }
                        means.set(r, dim, mu / nj);
                    }
                    let mut cmat = Matrix::zeros(d, d);
                    for t in 0..t_len {
                        let g = gtm[t][j][m];
                        for a in 0..d {
                            let da = x.get(t, a) - means.get(r, a);
                            for b in 0..d {
                                cmat.set(
                                    a,
                                    b,
                                    cmat.get(a, b) + g * da * (x.get(t, b) - means.get(r, b)),
                                );
                            }
                        }
                    }
                    for a in 0..d {
                        for b in 0..d {
                            cmat.set(a, b, cmat.get(a, b) / nj);
                        }
                        cmat.set(a, a, cmat.get(a, a).max(COV_FLOOR));
                    }
                    covs[r] = cmat;
                }
                if wsum > 0.0 {
                    for m in 0..nm {
                        mix_weights.set(j, m, mix_weights.get(j, m) / wsum);
                    }
                } else {
                    for m in 0..nm {
                        mix_weights.set(j, m, 1.0 / nm as f64);
                    }
                }
            }
            if it + 1 == self.max_iter {
                ctx.push(
                    Issue::builder(IssueCode::MaxIterReached)
                        .message("GmmHmmFull Baum–Welch hit max_iter")
                        .build(),
                );
            }
        }
        let occup: Vec<f64> = (0..k)
            .map(|j| {
                last_gamma
                    .iter()
                    .map(|g| g.get(j).copied().unwrap_or(0.0))
                    .sum()
            })
            .collect();
        diagnose_chain(&mut ctx, &start, &trans, &occup);
        if self.left_right {
            enforce_left_right(&mut start, &mut trans);
        }
        let fitted_tmp = FittedGmmHmmFull {
            labels: Vector::zeros(0),
            n_states: k,
            n_mix: nm,
            start: start.clone(),
            trans: trans.clone(),
            mix_weights: mix_weights.clone(),
            means: means.clone(),
            covs: covs.clone(),
            loglik,
        };
        let (labels, _) = viterbi_path(&start, &trans, &fitted_tmp.log_emit_seq(x));
        ctx.finish(FittedGmmHmmFull {
            labels,
            n_states: k,
            n_mix: nm,
            start,
            trans,
            mix_weights,
            means,
            covs,
            loglik,
        })
    }
}

/// GMM-HMM with one full covariance shared by every mixture in a state (`tied`).
///
/// Covariance free-parameter count and `n_mix` are not identification `p`.
#[derive(Clone, Debug)]
pub struct GmmHmmTied {
    /// Number of hidden states.
    pub n_states: usize,
    /// Mixture components per state.
    pub n_mix: usize,
    /// Baum–Welch iteration cap.
    pub max_iter: usize,
    /// Seed.
    pub seed: u64,
    /// If true, transitions to earlier states are zeroed (hmmlearn `left_right`).
    pub left_right: bool,
}

impl Default for GmmHmmTied {
    fn default() -> Self {
        Self {
            n_states: 2,
            n_mix: 1,
            max_iter: 40,
            seed: 0,
            left_right: false,
        }
    }
}

impl GmmHmmTied {
    /// `n_states` states with `n_mix` Gaussians sharing one covariance per state.
    pub fn new(n_states: usize, n_mix: usize) -> Self {
        Self {
            n_states,
            n_mix,
            ..Self::default()
        }
    }

    /// Fit alias.
    pub fn fit(&mut self, x: &Matrix, session: &Session) -> Result<Qualified<FittedGmmHmmTied>> {
        self.fit_unsupervised(x, session)
    }
}

/// Fitted tied-covariance GMM-HMM.
#[derive(Clone, Debug)]
pub struct FittedGmmHmmTied {
    /// Viterbi path.
    pub labels: Vector,
    /// Number of states.
    pub n_states: usize,
    /// Mixture size per state.
    pub n_mix: usize,
    /// Start distribution.
    pub start: Vector,
    /// Transitions.
    pub trans: Matrix,
    /// Mixture weights (`n_states` × `n_mix`).
    pub mix_weights: Matrix,
    /// Means (`n_states * n_mix` × `d`).
    pub means: Matrix,
    /// One dense covariance per state.
    pub covs: Vec<Matrix>,
    /// Training log-likelihood.
    pub loglik: f64,
}

impl FittedGmmHmmTied {
    fn log_state_emit(&self, x: &Matrix, t: usize, state: usize) -> f64 {
        let nm = self.n_mix.max(1);
        let cov = self
            .covs
            .get(state)
            .cloned()
            .unwrap_or_else(|| Matrix::zeros(0, 0));
        let mut parts = vec![0.0; nm];
        for m in 0..nm {
            let w = self.mix_weights.get(state, m).max(1e-300).ln();
            let r = mix_index(state, m, nm);
            parts[m] = w + log_full_gauss(x, t, &self.means, r, &cov);
        }
        logsumexp(&parts)
    }

    fn log_emit_seq(&self, x: &Matrix) -> Vec<Vec<f64>> {
        (0..x.nrows())
            .map(|t| {
                (0..self.n_states)
                    .map(|s| self.log_state_emit(x, t, s))
                    .collect()
            })
            .collect()
    }

    /// Viterbi path.
    pub fn decode(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("decode"));
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let (path, _) = viterbi_path(&self.start, &self.trans, &self.log_emit_seq(x));
        ctx.finish(path)
    }

    /// Scaled-forward log-likelihood.
    pub fn score(&self, x: &Matrix, session: &Session) -> Result<Qualified<f64>> {
        let mut ctx = FitCtx::with_session(session.child("score"));
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let fb = scaled_forward_backward(&mut ctx, &self.start, &self.trans, &self.log_emit_seq(x));
        ctx.finish(fb.map(|f| f.loglik).unwrap_or(f64::NEG_INFINITY))
    }
}

impl Predict for FittedGmmHmmTied {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        self.decode(x, session)
    }
}

impl FitUnsupervised for GmmHmmTied {
    type Fitted = FittedGmmHmmTied;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedGmmHmmTied>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        inspect_identification(
            &mut ctx.report,
            x.nrows(),
            self.n_states.max(1),
            &ctx.policy,
        );
        let (t_len, d) = x.shape();
        let k = self.n_states.max(1);
        let nm = self.n_mix.max(1);
        if t_len == 0 || d == 0 {
            let mut start = init_start(k);
            let mut trans = init_trans(k);
            if self.left_right {
                enforce_left_right(&mut start, &mut trans);
            }
            return ctx.finish(FittedGmmHmmTied {
                labels: empty_labels(0),
                n_states: k,
                n_mix: nm,
                start,
                trans,
                mix_weights: Matrix::zeros(k, nm),
                means: Matrix::zeros(k * nm, d),
                covs: vec![Matrix::zeros(d, d); k],
                loglik: f64::NAN,
            });
        }
        if series_zero_variance(x, ctx.policy.near_zero_variance) {
            ctx.push(emission_degenerate_issue(
                "tied GMM-HMM observation series has zero variance",
            ));
        }
        let mut rng = Rng::new(self.seed | 7);
        let seeds = kmeans_pp_rows(x, (k * nm).min(t_len), &mut rng);
        let mut means = Matrix::zeros(k * nm, d);
        for r in 0..(k * nm).min(seeds.nrows()) {
            for j in 0..d {
                means.set(r, j, seeds.get(r, j));
            }
        }
        let gcov = global_full_cov(x);
        let mut covs = vec![gcov.clone(); k];
        let mut mix_weights = Matrix::from_fn(k, nm, |_, _| 1.0 / nm as f64);
        let mut start = init_start(k);
        let mut trans = init_trans(k);
        if self.left_right {
            enforce_left_right(&mut start, &mut trans);
        }
        let mut loglik = f64::NEG_INFINITY;
        let mut last_gamma = Vec::new();
        for it in 0..self.max_iter.max(1) {
            let mut log_emit = vec![vec![0.0; k]; t_len];
            let mut log_comp = vec![vec![vec![0.0; nm]; k]; t_len];
            for t in 0..t_len {
                for j in 0..k {
                    let mut parts = vec![0.0; nm];
                    for m in 0..nm {
                        let r = mix_index(j, m, nm);
                        let lp = mix_weights.get(j, m).max(1e-300).ln()
                            + log_full_gauss(x, t, &means, r, &covs[j]);
                        log_comp[t][j][m] = lp;
                        parts[m] = lp;
                    }
                    log_emit[t][j] = logsumexp(&parts);
                }
            }
            let Some(fb) = scaled_forward_backward(&mut ctx, &start, &trans, &log_emit) else {
                break;
            };
            loglik = fb.loglik;
            last_gamma = fb.gamma.clone();
            ctx.session.step(it as u64, -loglik, None);
            let mut gtm = vec![vec![vec![0.0; nm]; k]; t_len];
            for t in 0..t_len {
                for j in 0..k {
                    let lse = logsumexp(&log_comp[t][j]);
                    for m in 0..nm {
                        let p = if lse.is_finite() {
                            (log_comp[t][j][m] - lse).exp()
                        } else {
                            1.0 / nm as f64
                        };
                        gtm[t][j][m] = fb.gamma[t][j] * p;
                    }
                }
            }
            for j in 0..k {
                start[j] = fb.gamma[0][j];
            }
            renormalize_vec(&mut start, TRANS_FLOOR);
            if t_len > 1 {
                for i in 0..k {
                    let den: f64 = (0..t_len - 1).map(|t| fb.gamma[t][i]).sum();
                    for j in 0..k {
                        let num: f64 = (0..t_len - 1).map(|t| fb.xi[t][i][j]).sum();
                        trans.set(
                            i,
                            j,
                            if den > 0.0 {
                                num / den
                            } else {
                                trans.get(i, j)
                            },
                        );
                    }
                }
                if self.left_right {
                    enforce_left_right(&mut start, &mut trans);
                } else {
                    renormalize_rows(&mut trans, TRANS_FLOOR);
                }
            }
            if self.left_right && k > 1 {
                enforce_left_right(&mut start, &mut trans);
            }
            for j in 0..k {
                let mut wsum = 0.0;
                let mut cmat = Matrix::zeros(d, d);
                for m in 0..nm {
                    let mut nj = 0.0;
                    for t in 0..t_len {
                        nj += gtm[t][j][m];
                    }
                    mix_weights.set(j, m, nj);
                    wsum += nj;
                    if nj <= TRANS_FLOOR {
                        ctx.push(
                            Issue::builder(IssueCode::MixtureWeightCollapsed)
                                .message(format!("tied GMM-HMM state {j} mix {m} collapsed"))
                                .metric("state", j as f64)
                                .metric("mix", m as f64)
                                .build(),
                        );
                        continue;
                    }
                    let r = mix_index(j, m, nm);
                    for dim in 0..d {
                        let mut mu = 0.0;
                        for t in 0..t_len {
                            mu += gtm[t][j][m] * x.get(t, dim);
                        }
                        means.set(r, dim, mu / nj);
                    }
                    for t in 0..t_len {
                        let g = gtm[t][j][m];
                        for a in 0..d {
                            let da = x.get(t, a) - means.get(r, a);
                            for b in 0..d {
                                cmat.set(
                                    a,
                                    b,
                                    cmat.get(a, b) + g * da * (x.get(t, b) - means.get(r, b)),
                                );
                            }
                        }
                    }
                }
                if wsum > TRANS_FLOOR {
                    for a in 0..d {
                        for b in 0..d {
                            cmat.set(a, b, cmat.get(a, b) / wsum);
                        }
                        cmat.set(a, a, cmat.get(a, a).max(COV_FLOOR));
                    }
                    covs[j] = cmat;
                    for m in 0..nm {
                        mix_weights.set(j, m, mix_weights.get(j, m) / wsum);
                    }
                } else {
                    for m in 0..nm {
                        mix_weights.set(j, m, 1.0 / nm as f64);
                    }
                }
            }
            if it + 1 == self.max_iter {
                ctx.push(
                    Issue::builder(IssueCode::MaxIterReached)
                        .message("GmmHmmTied Baum–Welch hit max_iter")
                        .build(),
                );
            }
        }
        let occup: Vec<f64> = (0..k)
            .map(|j| {
                last_gamma
                    .iter()
                    .map(|g| g.get(j).copied().unwrap_or(0.0))
                    .sum()
            })
            .collect();
        diagnose_chain(&mut ctx, &start, &trans, &occup);
        if self.left_right {
            enforce_left_right(&mut start, &mut trans);
        }
        let fitted_tmp = FittedGmmHmmTied {
            labels: Vector::zeros(0),
            n_states: k,
            n_mix: nm,
            start: start.clone(),
            trans: trans.clone(),
            mix_weights: mix_weights.clone(),
            means: means.clone(),
            covs: covs.clone(),
            loglik,
        };
        let (labels, _) = viterbi_path(&start, &trans, &fitted_tmp.log_emit_seq(x));
        ctx.finish(FittedGmmHmmTied {
            labels,
            n_states: k,
            n_mix: nm,
            start,
            trans,
            mix_weights,
            means,
            covs,
            loglik,
        })
    }
}

/// Poisson HMM (hmmlearn `PoissonHMM`): integer counts in column 0.
///
/// Negative observations are not in the support and abort. A constant zero
/// series makes every rate unidentified.
#[derive(Clone, Debug)]
pub struct PoissonHmm {
    /// Hidden states.
    pub n_states: usize,
    /// Baum–Welch cap.
    pub max_iter: usize,
    /// Seed.
    pub seed: u64,
    /// If true, transitions to earlier states are zeroed (hmmlearn `left_right`).
    pub left_right: bool,
}

impl Default for PoissonHmm {
    fn default() -> Self {
        Self {
            n_states: 2,
            max_iter: 40,
            seed: 1,
            left_right: false,
        }
    }
}

impl PoissonHmm {
    /// `k`-state Poisson HMM.
    pub fn new(n_states: usize) -> Self {
        Self {
            n_states,
            ..Self::default()
        }
    }

    /// Left-right Poisson HMM.
    pub fn left_right(n_states: usize) -> Self {
        Self {
            n_states,
            left_right: true,
            ..Self::default()
        }
    }

    /// Fit alias.
    pub fn fit(&mut self, x: &Matrix, session: &Session) -> Result<Qualified<FittedPoissonHmm>> {
        self.fit_unsupervised(x, session)
    }
}

/// Fitted Poisson HMM.
#[derive(Clone, Debug)]
pub struct FittedPoissonHmm {
    /// Viterbi path.
    pub labels: Vector,
    /// Start distribution.
    pub start: Vector,
    /// Transitions.
    pub trans: Matrix,
    /// Poisson rates \(\lambda_j\).
    pub rates: Vector,
    /// Training log-likelihood.
    pub loglik: f64,
}

fn log_poisson(k: f64, lambda: f64) -> f64 {
    if k < 0.0 || !k.is_finite() || !lambda.is_finite() {
        return f64::NEG_INFINITY;
    }
    let lam = lambda.max(1e-12);
    k * lam.ln() - lam - crate::special::ln_gamma(k + 1.0)
}

impl FittedPoissonHmm {
    fn log_emit_seq(&self, x: &Matrix) -> Vec<Vec<f64>> {
        let t = x.nrows();
        let s = self.rates.len();
        let mut out = vec![vec![f64::NEG_INFINITY; s]; t];
        for ti in 0..t {
            let k = x.get(ti, 0).round();
            for j in 0..s {
                out[ti][j] = log_poisson(k, self.rates[j]);
            }
        }
        out
    }

    /// Viterbi path.
    pub fn decode(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let ctx = FitCtx::with_session(session.child("decode"));
        let (path, _) = viterbi_path(&self.start, &self.trans, &self.log_emit_seq(x));
        ctx.finish(path)
    }

    /// Sequence log-likelihood via scaled forward–backward.
    pub fn score(&self, x: &Matrix, session: &Session) -> Result<Qualified<f64>> {
        let mut ctx = FitCtx::with_session(session.child("score"));
        let log_emit = self.log_emit_seq(x);
        let ll = scaled_forward_backward(&mut ctx, &self.start, &self.trans, &log_emit)
            .map(|fb| fb.loglik)
            .unwrap_or(f64::NEG_INFINITY);
        ctx.finish(ll)
    }
}

impl Predict for FittedPoissonHmm {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        self.decode(x, session)
    }
}

impl FitUnsupervised for PoissonHmm {
    type Fitted = FittedPoissonHmm;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedPoissonHmm>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let t_len = x.nrows();
        let k = self.n_states.max(1);
        if t_len == 0 {
            return ctx.finish(FittedPoissonHmm {
                labels: empty_labels(0),
                start: init_start(k),
                trans: init_trans(k),
                rates: Vector::zeros(k),
                loglik: f64::NAN,
            });
        }
        for i in 0..t_len {
            if x.get(i, 0) < 0.0 {
                ctx.push(
                    Issue::builder(IssueCode::NonPositiveSeries)
                        .message(format!("PoissonHMM y[{i}]={} < 0", x.get(i, 0)))
                        .build(),
                );
                break;
            }
        }
        let mean = x.column(0).mean().max(1e-3);
        if x.column(0).std() <= ctx.policy.near_zero_variance {
            ctx.push(emission_degenerate_issue(
                "count series is constant; Poisson rates are not identified across states",
            ));
        }
        let mut rates = Vector::from_iter((0..k).map(|j| mean * (0.5 + j as f64)));
        let mut start = init_start(k);
        let mut trans = init_trans(k);
        if self.left_right {
            enforce_left_right(&mut start, &mut trans);
        }
        let mut loglik = f64::NEG_INFINITY;
        let mut last_gamma: Vec<Vec<f64>> = Vec::new();
        for it in 0..self.max_iter.max(1) {
            let log_emit: Vec<Vec<f64>> = (0..t_len)
                .map(|t| {
                    let obs = x.get(t, 0).round();
                    (0..k).map(|j| log_poisson(obs, rates[j])).collect()
                })
                .collect();
            let Some(fb) = scaled_forward_backward(&mut ctx, &start, &trans, &log_emit) else {
                break;
            };
            loglik = fb.loglik;
            last_gamma = fb.gamma.clone();
            ctx.session.step(it as u64, -loglik, None);
            let mut ns = Vector::zeros(k);
            let mut nt = Matrix::zeros(k, k);
            let mut wr = Vector::zeros(k);
            let mut mass = Vector::zeros(k);
            for j in 0..k {
                ns[j] = fb.gamma[0][j];
            }
            for t in 0..t_len {
                for j in 0..k {
                    wr[j] += fb.gamma[t][j] * x.get(t, 0).max(0.0);
                    mass[j] += fb.gamma[t][j];
                }
            }
            for j in 0..k {
                rates[j] = if mass[j] > 1e-12 {
                    wr[j] / mass[j]
                } else {
                    mean
                }
                .max(1e-6);
            }
            for t in 0..t_len.saturating_sub(1) {
                for i in 0..k {
                    for j in 0..k {
                        nt.set(i, j, nt.get(i, j) + fb.xi[t][i][j]);
                    }
                }
            }
            start = ns;
            trans = nt;
            renormalize_vec(&mut start, TRANS_FLOOR);
            renormalize_rows(&mut trans, TRANS_FLOOR);
            if self.left_right {
                enforce_left_right(&mut start, &mut trans);
            }
        }
        let occup: Vec<f64> = (0..k)
            .map(|j| {
                last_gamma
                    .iter()
                    .map(|g| g.get(j).copied().unwrap_or(0.0))
                    .sum()
            })
            .collect();
        if !last_gamma.is_empty() {
            diagnose_chain(&mut ctx, &start, &trans, &occup);
        }
        let log_emit: Vec<Vec<f64>> = (0..t_len)
            .map(|t| {
                let obs = x.get(t, 0).round();
                (0..k).map(|j| log_poisson(obs, rates[j])).collect()
            })
            .collect();
        let (labels, _) = viterbi_path(&start, &trans, &log_emit);
        if self.left_right {
            enforce_left_right(&mut start, &mut trans);
        }
        ctx.finish(FittedPoissonHmm {
            labels,
            start,
            trans,
            rates,
            loglik,
        })
    }
}

/// Left-right Poisson HMM (hmmlearn `PoissonHMM` with `params` left-right).
pub type PoissonHmmLeftRight = PoissonHmm;

fn digamma(mut x: f64) -> f64 {
    if x <= 0.0 {
        return f64::NAN;
    }
    let mut r = 0.0;
    while x < 6.0 {
        r -= 1.0 / x;
        x += 1.0;
    }
    let inv = 1.0 / x;
    let inv2 = inv * inv;
    r + x.ln() - 0.5 * inv - inv2 / 12.0 + inv2 * inv2 / 120.0
}

/// Mean-field variational Gaussian HMM (Dirichlet + diagonal Normal-Gamma).
#[derive(Clone, Debug)]
pub struct VariationalGaussianHmm {
    /// Number of hidden states.
    pub n_states: usize,
    /// Variational EM iteration cap.
    pub max_iter: usize,
    /// Seed for k-means++ mean initialization.
    pub seed: u64,
}

impl Default for VariationalGaussianHmm {
    fn default() -> Self {
        Self {
            n_states: 2,
            max_iter: 40,
            seed: 0,
        }
    }
}

impl VariationalGaussianHmm {
    /// `n_states` variational Gaussians.
    pub fn new(n_states: usize) -> Self {
        Self {
            n_states,
            ..Self::default()
        }
    }

    /// Fit alias.
    pub fn fit(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedVariationalHmm>> {
        self.fit_unsupervised(x, session)
    }
}

/// Fitted variational Gaussian HMM (posterior means of the factors).
#[derive(Clone, Debug)]
pub struct FittedVariationalHmm {
    /// Viterbi path under the posterior-mean parameters.
    pub labels: Vector,
    /// Number of states.
    pub n_states: usize,
    /// Posterior-mean start distribution.
    pub start: Vector,
    /// Posterior-mean transitions.
    pub trans: Matrix,
    /// Posterior-mean emission means.
    pub means: Matrix,
    /// Posterior-mean diagonal variances.
    pub covs: Matrix,
    /// Evidence lower bound (ELBO) proxy: expected complete log-likelihood.
    pub elbo: f64,
}

impl FittedVariationalHmm {
    fn as_gaussian(&self) -> FittedGaussianHmm {
        FittedGaussianHmm {
            labels: self.labels.clone(),
            n_states: self.n_states,
            start: self.start.clone(),
            trans: self.trans.clone(),
            means: self.means.clone(),
            covs: self.covs.clone(),
            loglik: self.elbo,
        }
    }

    /// Viterbi path.
    pub fn decode(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        self.as_gaussian().decode(x, session)
    }

    /// Scaled-forward log-likelihood under posterior-mean parameters.
    pub fn score(&self, x: &Matrix, session: &Session) -> Result<Qualified<f64>> {
        self.as_gaussian().score(x, session)
    }
}

impl Predict for FittedVariationalHmm {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        self.decode(x, session)
    }
}

impl FitUnsupervised for VariationalGaussianHmm {
    type Fitted = FittedVariationalHmm;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedVariationalHmm>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        inspect_identification(
            &mut ctx.report,
            x.nrows(),
            self.n_states.max(1),
            &ctx.policy,
        );
        let (t_len, d) = x.shape();
        let k = self.n_states.max(1);
        if t_len == 0 || d == 0 {
            return ctx.finish(FittedVariationalHmm {
                labels: empty_labels(t_len),
                n_states: k,
                start: init_start(k),
                trans: init_trans(k),
                means: Matrix::zeros(k, d),
                covs: Matrix::zeros(k, d),
                elbo: f64::NAN,
            });
        }
        if series_zero_variance(x, ctx.policy.near_zero_variance) {
            ctx.push(emission_degenerate_issue(
                "variational HMM: observation series has zero variance",
            ));
        }
        let mut rng = Rng::new(self.seed | 3);
        let mut means = kmeans_pp_rows(x, k.min(t_len), &mut rng);
        if means.nrows() < k {
            let mut padded = Matrix::zeros(k, d);
            for i in 0..means.nrows() {
                for j in 0..d {
                    padded.set(i, j, means.get(i, j));
                }
            }
            means = padded;
        }
        let gvar = global_diag_var(x);
        let mut covs = Matrix::from_fn(k, d, |_, j| gvar[j]);
        let mut alpha0 = Vector::filled(k, 1.0);
        let mut alpha_a = Matrix::from_fn(k, k, |_, _| 1.0);
        let mut elbo = f64::NEG_INFINITY;
        let mut last_gamma: Vec<Vec<f64>> = Vec::new();
        for it in 0..self.max_iter.max(1) {
            let mut start = Vector::zeros(k);
            let sum_a0: f64 = (0..k).map(|j| alpha0[j]).sum();
            for j in 0..k {
                start[j] = (digamma(alpha0[j]) - digamma(sum_a0)).exp();
            }
            renormalize_vec(&mut start, TRANS_FLOOR);
            let mut trans = Matrix::zeros(k, k);
            for i in 0..k {
                let row_s: f64 = (0..k).map(|j| alpha_a.get(i, j)).sum();
                for j in 0..k {
                    trans.set(i, j, (digamma(alpha_a.get(i, j)) - digamma(row_s)).exp());
                }
            }
            renormalize_rows(&mut trans, TRANS_FLOOR);
            let mut log_emit = vec![vec![0.0; k]; t_len];
            for t in 0..t_len {
                for j in 0..k {
                    log_emit[t][j] = log_diag_gauss(x, t, &means, j, &covs);
                }
            }
            let Some(fb) = scaled_forward_backward(&mut ctx, &start, &trans, &log_emit) else {
                break;
            };
            elbo = fb.loglik;
            last_gamma = fb.gamma.clone();
            ctx.session.step(it as u64, -elbo, None);
            for j in 0..k {
                alpha0[j] = 1.0 + fb.gamma[0][j];
            }
            if t_len > 1 {
                for i in 0..k {
                    for j in 0..k {
                        let mut num = 1.0;
                        for t in 0..t_len - 1 {
                            num += fb.xi[t][i][j];
                        }
                        alpha_a.set(i, j, num);
                    }
                }
            }
            let n0 = 1e-3;
            for j in 0..k {
                let mut nj = 0.0;
                for t in 0..t_len {
                    nj += fb.gamma[t][j];
                }
                if nj <= TRANS_FLOOR {
                    ctx.push(
                        Issue::builder(IssueCode::UnreachableState)
                            .message(format!("variational state {j} mass {nj:.3e}"))
                            .build(),
                    );
                    continue;
                }
                for dim in 0..d {
                    let mut m = 0.0;
                    for t in 0..t_len {
                        m += fb.gamma[t][j] * x.get(t, dim);
                    }
                    let xbar = m / nj;
                    means.set(j, dim, (n0 * gvar[dim].sqrt() + nj * xbar) / (n0 + nj));
                    let mut s = n0;
                    for t in 0..t_len {
                        let z = x.get(t, dim) - means.get(j, dim);
                        s += fb.gamma[t][j] * z * z;
                    }
                    covs.set(j, dim, (s / (nj + n0)).max(COV_FLOOR));
                }
            }
            if it + 1 == self.max_iter {
                ctx.push(
                    Issue::builder(IssueCode::MaxIterReached)
                        .message("variational HMM hit max_iter")
                        .build(),
                );
            }
        }
        let occup: Vec<f64> = (0..k)
            .map(|j| {
                last_gamma
                    .iter()
                    .map(|g| g.get(j).copied().unwrap_or(0.0))
                    .sum()
            })
            .collect();
        let mut start = Vector::zeros(k);
        let sum_a0: f64 = (0..k).map(|j| alpha0[j]).sum();
        for j in 0..k {
            start[j] = alpha0[j] / sum_a0.max(1e-12);
        }
        let mut trans = Matrix::zeros(k, k);
        for i in 0..k {
            let row_s: f64 = (0..k).map(|j| alpha_a.get(i, j)).sum();
            for j in 0..k {
                trans.set(i, j, alpha_a.get(i, j) / row_s.max(1e-12));
            }
        }
        diagnose_chain(&mut ctx, &start, &trans, &occup);
        ctx.push(
            Issue::builder(IssueCode::JitterInjected)
                .message("variational HMM uses Dirichlet(1) and Normal-Gamma floors")
                .compromise(NumericalCompromise::new(
                    "unconstrained variational Bayes",
                    "Dirichlet(1) + diagonal variance floor",
                    "zero cells make the sequence probability vanish",
                    "posterior means are shrunk; do not treat them as MLEs",
                ))
                .build(),
        );
        let log_emit: Vec<Vec<f64>> = (0..t_len)
            .map(|t| {
                (0..k)
                    .map(|j| log_diag_gauss(x, t, &means, j, &covs))
                    .collect()
            })
            .collect();
        let (labels, _) = viterbi_path(&start, &trans, &log_emit);
        ctx.finish(FittedVariationalHmm {
            labels,
            n_states: k,
            start,
            trans,
            means,
            covs,
            elbo,
        })
    }
}

/// Mean-field variational categorical HMM (Dirichlet on \(\pi\), \(A\), emissions).
#[derive(Clone, Debug)]
pub struct VariationalCategoricalHmm {
    /// Number of hidden states.
    pub n_states: usize,
    /// Variational EM iteration cap.
    pub max_iter: usize,
}

impl Default for VariationalCategoricalHmm {
    fn default() -> Self {
        Self {
            n_states: 2,
            max_iter: 40,
        }
    }
}

impl VariationalCategoricalHmm {
    /// `n_states` variational categoricals.
    pub fn new(n_states: usize) -> Self {
        Self {
            n_states,
            ..Self::default()
        }
    }

    /// Fit alias.
    pub fn fit(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedVariationalCatHmm>> {
        self.fit_unsupervised(x, session)
    }
}

/// Fitted variational categorical HMM.
#[derive(Clone, Debug)]
pub struct FittedVariationalCatHmm {
    /// Viterbi path under posterior-mean parameters.
    pub labels: Vector,
    /// Number of states.
    pub n_states: usize,
    /// Posterior-mean start.
    pub start: Vector,
    /// Posterior-mean transitions.
    pub trans: Matrix,
    /// Posterior-mean emissions (`n_states` × `n_symbols`).
    pub emission: Matrix,
    /// ELBO proxy (expected complete log-likelihood).
    pub elbo: f64,
}

impl FittedVariationalCatHmm {
    fn as_multi(&self) -> FittedMultinomialHmm {
        FittedMultinomialHmm {
            labels: self.labels.clone(),
            n_states: self.n_states,
            start: self.start.clone(),
            trans: self.trans.clone(),
            emission: self.emission.clone(),
            loglik: self.elbo,
        }
    }

    /// Viterbi path.
    pub fn decode(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        self.as_multi().decode(x, session)
    }
}

impl Predict for FittedVariationalCatHmm {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        self.decode(x, session)
    }
}

impl FitUnsupervised for VariationalCategoricalHmm {
    type Fitted = FittedVariationalCatHmm;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedVariationalCatHmm>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let k = self.n_states.max(1);
        let (codes, n_sym) = codes_from_x(x);
        let t_len = codes.len();
        if t_len == 0 {
            return ctx.finish(FittedVariationalCatHmm {
                labels: empty_labels(0),
                n_states: k,
                start: init_start(k),
                trans: init_trans(k),
                emission: Matrix::zeros(k, 0),
                elbo: f64::NAN,
            });
        }
        if n_sym <= 1 {
            ctx.push(emission_degenerate_issue(
                "variational categorical HMM: only one emission symbol",
            ));
        }
        let mut alpha0 = Vector::filled(k, 1.0);
        let mut alpha_a = Matrix::from_fn(k, k, |_, _| 1.0);
        let mut alpha_e = Matrix::from_fn(k, n_sym, |_, _| 1.0);
        let mut elbo = f64::NEG_INFINITY;
        let mut last_gamma: Vec<Vec<f64>> = Vec::new();
        for it in 0..self.max_iter.max(1) {
            let mut start = Vector::zeros(k);
            let sum_a0: f64 = (0..k).map(|j| alpha0[j]).sum();
            for j in 0..k {
                start[j] = (digamma(alpha0[j]) - digamma(sum_a0)).exp();
            }
            renormalize_vec(&mut start, TRANS_FLOOR);
            let mut trans = Matrix::zeros(k, k);
            for i in 0..k {
                let row_s: f64 = (0..k).map(|j| alpha_a.get(i, j)).sum();
                for j in 0..k {
                    trans.set(i, j, (digamma(alpha_a.get(i, j)) - digamma(row_s)).exp());
                }
            }
            renormalize_rows(&mut trans, TRANS_FLOOR);
            let mut emit = Matrix::zeros(k, n_sym);
            for i in 0..k {
                let row_s: f64 = (0..n_sym).map(|j| alpha_e.get(i, j)).sum();
                for j in 0..n_sym {
                    emit.set(i, j, (digamma(alpha_e.get(i, j)) - digamma(row_s)).exp());
                }
            }
            renormalize_rows(&mut emit, TRANS_FLOOR);
            let log_emit: Vec<Vec<f64>> = (0..t_len)
                .map(|t| {
                    let o = codes[t].min(n_sym.saturating_sub(1));
                    (0..k)
                        .map(|j| {
                            let p = emit.get(j, o).max(TRANS_FLOOR);
                            p.ln()
                        })
                        .collect()
                })
                .collect();
            let Some(fb) = scaled_forward_backward(&mut ctx, &start, &trans, &log_emit) else {
                break;
            };
            elbo = fb.loglik;
            last_gamma = fb.gamma.clone();
            ctx.session.step(it as u64, -elbo, None);
            for j in 0..k {
                alpha0[j] = 1.0 + fb.gamma[0][j];
            }
            if t_len > 1 {
                for i in 0..k {
                    for j in 0..k {
                        let mut num = 1.0;
                        for t in 0..t_len - 1 {
                            num += fb.xi[t][i][j];
                        }
                        alpha_a.set(i, j, num);
                    }
                }
            }
            for j in 0..k {
                for s in 0..n_sym {
                    let mut num = 1.0;
                    for t in 0..t_len {
                        if codes[t] == s {
                            num += fb.gamma[t][j];
                        }
                    }
                    alpha_e.set(j, s, num);
                }
            }
            if it + 1 == self.max_iter {
                ctx.push(
                    Issue::builder(IssueCode::MaxIterReached)
                        .message("variational categorical HMM hit max_iter")
                        .build(),
                );
            }
        }
        let occup: Vec<f64> = (0..k)
            .map(|j| {
                last_gamma
                    .iter()
                    .map(|g| g.get(j).copied().unwrap_or(0.0))
                    .sum()
            })
            .collect();
        let mut start = Vector::zeros(k);
        let sum_a0: f64 = (0..k).map(|j| alpha0[j]).sum();
        for j in 0..k {
            start[j] = alpha0[j] / sum_a0.max(1e-12);
        }
        let mut trans = Matrix::zeros(k, k);
        for i in 0..k {
            let row_s: f64 = (0..k).map(|j| alpha_a.get(i, j)).sum();
            for j in 0..k {
                trans.set(i, j, alpha_a.get(i, j) / row_s.max(1e-12));
            }
        }
        let mut emission = Matrix::zeros(k, n_sym);
        for i in 0..k {
            let row_s: f64 = (0..n_sym).map(|j| alpha_e.get(i, j)).sum();
            for j in 0..n_sym {
                emission.set(i, j, alpha_e.get(i, j) / row_s.max(1e-12));
            }
        }
        diagnose_chain(&mut ctx, &start, &trans, &occup);
        ctx.push(
            Issue::builder(IssueCode::JitterInjected)
                .message("variational categorical HMM uses Dirichlet(1) floors")
                .compromise(NumericalCompromise::new(
                    "unconstrained variational Bayes",
                    "Dirichlet(1) on π, A, and emission rows",
                    "zero cells make the sequence probability vanish",
                    "posterior means are shrunk; do not treat them as MLEs",
                ))
                .build(),
        );
        let log_emit: Vec<Vec<f64>> = (0..t_len)
            .map(|t| {
                let o = codes[t].min(n_sym.saturating_sub(1));
                (0..k)
                    .map(|j| emission.get(j, o).max(TRANS_FLOOR).ln())
                    .collect()
            })
            .collect();
        let (labels, _) = viterbi_path(&start, &trans, &log_emit);
        ctx.finish(FittedVariationalCatHmm {
            labels,
            n_states: k,
            start,
            trans,
            emission,
            elbo,
        })
    }
}

/// Mean-field variational Poisson HMM (Dirichlet on \(\pi\), \(A\); Gamma on rates).
#[derive(Clone, Debug)]
pub struct VariationalPoissonHmm {
    /// Number of hidden states.
    pub n_states: usize,
    /// Variational EM iteration cap.
    pub max_iter: usize,
}

impl Default for VariationalPoissonHmm {
    fn default() -> Self {
        Self {
            n_states: 2,
            max_iter: 40,
        }
    }
}

impl VariationalPoissonHmm {
    /// `n_states` variational Poissons.
    pub fn new(n_states: usize) -> Self {
        Self {
            n_states,
            ..Self::default()
        }
    }

    /// Fit alias.
    pub fn fit(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedVariationalPoissonHmm>> {
        self.fit_unsupervised(x, session)
    }
}

/// Fitted variational Poisson HMM (posterior-mean rates).
#[derive(Clone, Debug)]
pub struct FittedVariationalPoissonHmm {
    /// Viterbi path under the posterior-mean parameters.
    pub labels: Vector,
    /// Number of states.
    pub n_states: usize,
    /// Posterior-mean start distribution.
    pub start: Vector,
    /// Posterior-mean transitions.
    pub trans: Matrix,
    /// Posterior-mean Poisson rates.
    pub rates: Vector,
    /// Evidence lower bound proxy (expected complete log-likelihood).
    pub elbo: f64,
}

impl FittedVariationalPoissonHmm {
    fn as_poisson(&self) -> FittedPoissonHmm {
        FittedPoissonHmm {
            labels: self.labels.clone(),
            start: self.start.clone(),
            trans: self.trans.clone(),
            rates: self.rates.clone(),
            loglik: self.elbo,
        }
    }

    /// Viterbi path.
    pub fn decode(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        self.as_poisson().decode(x, session)
    }

    /// Scaled-forward log-likelihood under posterior-mean rates.
    pub fn score(&self, x: &Matrix, session: &Session) -> Result<Qualified<f64>> {
        self.as_poisson().score(x, session)
    }
}

impl Predict for FittedVariationalPoissonHmm {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        self.decode(x, session)
    }
}

impl FitUnsupervised for VariationalPoissonHmm {
    type Fitted = FittedVariationalPoissonHmm;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedVariationalPoissonHmm>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        inspect_identification(
            &mut ctx.report,
            x.nrows(),
            self.n_states.max(1),
            &ctx.policy,
        );
        let t_len = x.nrows();
        let k = self.n_states.max(1);
        if t_len == 0 {
            return ctx.finish(FittedVariationalPoissonHmm {
                labels: empty_labels(0),
                n_states: k,
                start: init_start(k),
                trans: init_trans(k),
                rates: Vector::zeros(k),
                elbo: f64::NAN,
            });
        }
        for i in 0..t_len {
            if x.get(i, 0) < 0.0 {
                ctx.push(
                    Issue::builder(IssueCode::NonPositiveSeries)
                        .message(format!("variational PoissonHMM y[{i}]={} < 0", x.get(i, 0)))
                        .build(),
                );
                break;
            }
        }
        let mean = x.column(0).mean().max(1e-3);
        if x.column(0).std() <= ctx.policy.near_zero_variance {
            ctx.push(emission_degenerate_issue(
                "variational Poisson: count series is constant; rates are not identified",
            ));
        }
        let mut rates = Vector::from_iter((0..k).map(|j| mean * (0.5 + j as f64)));
        let mut alpha0 = Vector::filled(k, 1.0);
        let mut alpha_a = Matrix::from_fn(k, k, |_, _| 1.0);
        let mut elbo = f64::NEG_INFINITY;
        let mut last_gamma: Vec<Vec<f64>> = Vec::new();
        for it in 0..self.max_iter.max(1) {
            let mut start = Vector::zeros(k);
            let sum_a0: f64 = (0..k).map(|j| alpha0[j]).sum();
            for j in 0..k {
                start[j] = (digamma(alpha0[j]) - digamma(sum_a0)).exp();
            }
            renormalize_vec(&mut start, TRANS_FLOOR);
            let mut trans = Matrix::zeros(k, k);
            for i in 0..k {
                let row_s: f64 = (0..k).map(|j| alpha_a.get(i, j)).sum();
                for j in 0..k {
                    trans.set(i, j, (digamma(alpha_a.get(i, j)) - digamma(row_s)).exp());
                }
            }
            renormalize_rows(&mut trans, TRANS_FLOOR);
            let log_emit: Vec<Vec<f64>> = (0..t_len)
                .map(|t| {
                    let obs = x.get(t, 0).round();
                    (0..k).map(|j| log_poisson(obs, rates[j])).collect()
                })
                .collect();
            let Some(fb) = scaled_forward_backward(&mut ctx, &start, &trans, &log_emit) else {
                break;
            };
            elbo = fb.loglik;
            last_gamma = fb.gamma.clone();
            ctx.session.step(it as u64, -elbo, None);
            for j in 0..k {
                alpha0[j] = 1.0 + fb.gamma[0][j];
            }
            if t_len > 1 {
                for i in 0..k {
                    for j in 0..k {
                        let mut num = 1.0;
                        for t in 0..t_len - 1 {
                            num += fb.xi[t][i][j];
                        }
                        alpha_a.set(i, j, num);
                    }
                }
            }
            let n0 = 1e-3;
            for j in 0..k {
                let mut nj = 0.0;
                let mut wr = 0.0;
                for t in 0..t_len {
                    nj += fb.gamma[t][j];
                    wr += fb.gamma[t][j] * x.get(t, 0).max(0.0);
                }
                if nj <= TRANS_FLOOR {
                    ctx.push(
                        Issue::builder(IssueCode::UnreachableState)
                            .message(format!("variational Poisson state {j} mass {nj:.3e}"))
                            .build(),
                    );
                    continue;
                }
                rates[j] = ((n0 * mean + wr) / (n0 + nj)).max(1e-6);
            }
            if it + 1 == self.max_iter {
                ctx.push(
                    Issue::builder(IssueCode::MaxIterReached)
                        .message("variational Poisson HMM hit max_iter")
                        .build(),
                );
            }
        }
        let occup: Vec<f64> = (0..k)
            .map(|j| {
                last_gamma
                    .iter()
                    .map(|g| g.get(j).copied().unwrap_or(0.0))
                    .sum()
            })
            .collect();
        let mut start = Vector::zeros(k);
        let sum_a0: f64 = (0..k).map(|j| alpha0[j]).sum();
        for j in 0..k {
            start[j] = alpha0[j] / sum_a0.max(1e-12);
        }
        let mut trans = Matrix::zeros(k, k);
        for i in 0..k {
            let row_s: f64 = (0..k).map(|j| alpha_a.get(i, j)).sum();
            for j in 0..k {
                trans.set(i, j, alpha_a.get(i, j) / row_s.max(1e-12));
            }
        }
        diagnose_chain(&mut ctx, &start, &trans, &occup);
        ctx.push(
            Issue::builder(IssueCode::JitterInjected)
                .message("variational Poisson HMM uses Dirichlet(1) and Gamma floors")
                .compromise(NumericalCompromise::new(
                    "unconstrained variational Bayes",
                    "Dirichlet(1) + Gamma rate floor",
                    "zero cells make the sequence probability vanish",
                    "posterior means are shrunk; do not treat them as MLEs",
                ))
                .build(),
        );
        let log_emit: Vec<Vec<f64>> = (0..t_len)
            .map(|t| {
                let obs = x.get(t, 0).round();
                (0..k).map(|j| log_poisson(obs, rates[j])).collect()
            })
            .collect();
        let (labels, _) = viterbi_path(&start, &trans, &log_emit);
        ctx.finish(FittedVariationalPoissonHmm {
            labels,
            n_states: k,
            start,
            trans,
            rates,
            elbo,
        })
    }
}

/// Mean-field variational GMM-HMM (Dirichlet on \(\pi,A,w\); Normal-Gamma on mixtures).
///
/// Mix count is not identification `p`.
#[derive(Clone, Debug)]
pub struct VariationalGmmHmm {
    /// Hidden states.
    pub n_states: usize,
    /// Mixture components per state.
    pub n_mix: usize,
    /// Variational EM iteration cap.
    pub max_iter: usize,
    /// Seed.
    pub seed: u64,
}

impl Default for VariationalGmmHmm {
    fn default() -> Self {
        Self {
            n_states: 2,
            n_mix: 1,
            max_iter: 40,
            seed: 0,
        }
    }
}

impl VariationalGmmHmm {
    /// `n_states` states with `n_mix` variational Gaussians each.
    pub fn new(n_states: usize, n_mix: usize) -> Self {
        Self {
            n_states,
            n_mix,
            ..Self::default()
        }
    }

    /// Fit alias.
    pub fn fit(&mut self, x: &Matrix, session: &Session) -> Result<Qualified<FittedVariationalGmmHmm>> {
        self.fit_unsupervised(x, session)
    }
}

/// Fitted variational GMM-HMM (posterior-mean parameters).
#[derive(Clone, Debug)]
pub struct FittedVariationalGmmHmm {
    /// Viterbi path.
    pub labels: Vector,
    /// Number of states.
    pub n_states: usize,
    /// Mixture size.
    pub n_mix: usize,
    /// Posterior-mean start.
    pub start: Vector,
    /// Posterior-mean transitions.
    pub trans: Matrix,
    /// Mixture weights.
    pub mix_weights: Matrix,
    /// Mixture means.
    pub means: Matrix,
    /// Diagonal variances.
    pub vars: Matrix,
    /// ELBO proxy.
    pub elbo: f64,
}

impl FittedVariationalGmmHmm {
    fn as_gmm(&self) -> FittedGmmHmm {
        FittedGmmHmm {
            labels: self.labels.clone(),
            n_states: self.n_states,
            n_mix: self.n_mix,
            start: self.start.clone(),
            trans: self.trans.clone(),
            mix_weights: self.mix_weights.clone(),
            means: self.means.clone(),
            vars: self.vars.clone(),
            loglik: self.elbo,
        }
    }

    /// Viterbi path.
    pub fn decode(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        self.as_gmm().decode(x, session)
    }

    /// Scaled-forward log-likelihood.
    pub fn score(&self, x: &Matrix, session: &Session) -> Result<Qualified<f64>> {
        self.as_gmm().score(x, session)
    }
}

impl Predict for FittedVariationalGmmHmm {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        self.decode(x, session)
    }
}

impl FitUnsupervised for VariationalGmmHmm {
    type Fitted = FittedVariationalGmmHmm;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedVariationalGmmHmm>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        inspect_identification(
            &mut ctx.report,
            x.nrows(),
            self.n_states.max(1),
            &ctx.policy,
        );
        let (t_len, d) = x.shape();
        let k = self.n_states.max(1);
        let nm = self.n_mix.max(1);
        if t_len == 0 || d == 0 {
            return ctx.finish(FittedVariationalGmmHmm {
                labels: empty_labels(0),
                n_states: k,
                n_mix: nm,
                start: init_start(k),
                trans: init_trans(k),
                mix_weights: Matrix::from_fn(k, nm, |_, _| 1.0 / nm as f64),
                means: Matrix::zeros(k * nm, d),
                vars: Matrix::zeros(k * nm, d),
                elbo: f64::NAN,
            });
        }
        if series_zero_variance(x, ctx.policy.near_zero_variance) {
            ctx.push(emission_degenerate_issue(
                "variational GMM-HMM observation series has zero variance",
            ));
        }
        let mut rng = Rng::new(self.seed | 11);
        let seeds = kmeans_pp_rows(x, (k * nm).min(t_len), &mut rng);
        let mut means = Matrix::zeros(k * nm, d);
        for r in 0..(k * nm).min(seeds.nrows()) {
            for j in 0..d {
                means.set(r, j, seeds.get(r, j));
            }
        }
        let gvar = global_diag_var(x);
        let mut vars = Matrix::from_fn(k * nm, d, |_, j| gvar[j]);
        let mut mix_dir = Matrix::from_fn(k, nm, |_, _| 1.0);
        let mut alpha0 = Vector::filled(k, 1.0);
        let mut alpha_a = Matrix::from_fn(k, k, |_, _| 1.0);
        let mut elbo = f64::NEG_INFINITY;
        let mut last_gamma: Vec<Vec<f64>> = Vec::new();
        for it in 0..self.max_iter.max(1) {
            let mut start = Vector::zeros(k);
            let sum_a0: f64 = (0..k).map(|j| alpha0[j]).sum();
            for j in 0..k {
                start[j] = (digamma(alpha0[j]) - digamma(sum_a0)).exp();
            }
            renormalize_vec(&mut start, TRANS_FLOOR);
            let mut trans = Matrix::zeros(k, k);
            for i in 0..k {
                let row_s: f64 = (0..k).map(|j| alpha_a.get(i, j)).sum();
                for j in 0..k {
                    trans.set(i, j, (digamma(alpha_a.get(i, j)) - digamma(row_s)).exp());
                }
            }
            renormalize_rows(&mut trans, TRANS_FLOOR);
            let mut mix_weights = Matrix::zeros(k, nm);
            for j in 0..k {
                let row_s: f64 = (0..nm).map(|m| mix_dir.get(j, m)).sum();
                for m in 0..nm {
                    mix_weights.set(j, m, (digamma(mix_dir.get(j, m)) - digamma(row_s)).exp());
                }
            }
            let mut log_emit = vec![vec![0.0; k]; t_len];
            let mut log_comp = vec![vec![vec![0.0; nm]; k]; t_len];
            for t in 0..t_len {
                for j in 0..k {
                    let mut parts = vec![0.0; nm];
                    for m in 0..nm {
                        let r = mix_index(j, m, nm);
                        let lp = mix_weights.get(j, m).max(1e-300).ln()
                            + log_diag_gauss(x, t, &means, r, &vars);
                        log_comp[t][j][m] = lp;
                        parts[m] = lp;
                    }
                    log_emit[t][j] = logsumexp(&parts);
                }
            }
            let Some(fb) = scaled_forward_backward(&mut ctx, &start, &trans, &log_emit) else {
                break;
            };
            elbo = fb.loglik;
            last_gamma = fb.gamma.clone();
            ctx.session.step(it as u64, -elbo, None);
            for j in 0..k {
                alpha0[j] = 1.0 + fb.gamma[0][j];
            }
            if t_len > 1 {
                for i in 0..k {
                    for j in 0..k {
                        let mut num = 1.0;
                        for t in 0..t_len - 1 {
                            num += fb.xi[t][i][j];
                        }
                        alpha_a.set(i, j, num);
                    }
                }
            }
            let n0 = 1e-3;
            for j in 0..k {
                for m in 0..nm {
                    let lse = |t: usize| logsumexp(&log_comp[t][j]);
                    let mut nj = 0.0;
                    let mut acc = vec![0.0; d];
                    for t in 0..t_len {
                        let p = if lse(t).is_finite() {
                            (log_comp[t][j][m] - lse(t)).exp()
                        } else {
                            1.0 / nm as f64
                        };
                        let g = fb.gamma[t][j] * p;
                        nj += g;
                        for dim in 0..d {
                            acc[dim] += g * x.get(t, dim);
                        }
                    }
                    mix_dir.set(j, m, 1.0 + nj);
                    if nj <= TRANS_FLOOR {
                        ctx.push(
                            Issue::builder(IssueCode::MixtureWeightCollapsed)
                                .message(format!("variational GMM-HMM state {j} mix {m} collapsed"))
                                .build(),
                        );
                        continue;
                    }
                    let r = mix_index(j, m, nm);
                    for dim in 0..d {
                        let xbar = acc[dim] / nj;
                        means.set(r, dim, (n0 * gvar[dim].sqrt() + nj * xbar) / (n0 + nj));
                        let mut s = n0;
                        for t in 0..t_len {
                            let p = if lse(t).is_finite() {
                                (log_comp[t][j][m] - lse(t)).exp()
                            } else {
                                1.0 / nm as f64
                            };
                            let g = fb.gamma[t][j] * p;
                            let z = x.get(t, dim) - means.get(r, dim);
                            s += g * z * z;
                        }
                        vars.set(r, dim, (s / (nj + n0)).max(COV_FLOOR));
                    }
                }
            }
            if it + 1 == self.max_iter {
                ctx.push(
                    Issue::builder(IssueCode::MaxIterReached)
                        .message("variational GMM-HMM hit max_iter")
                        .build(),
                );
            }
        }
        let occup: Vec<f64> = (0..k)
            .map(|j| {
                last_gamma
                    .iter()
                    .map(|g| g.get(j).copied().unwrap_or(0.0))
                    .sum()
            })
            .collect();
        let mut start = Vector::zeros(k);
        let sum_a0: f64 = (0..k).map(|j| alpha0[j]).sum();
        for j in 0..k {
            start[j] = alpha0[j] / sum_a0.max(1e-12);
        }
        let mut trans = Matrix::zeros(k, k);
        for i in 0..k {
            let row_s: f64 = (0..k).map(|j| alpha_a.get(i, j)).sum();
            for j in 0..k {
                trans.set(i, j, alpha_a.get(i, j) / row_s.max(1e-12));
            }
        }
        let mut mix_weights = Matrix::zeros(k, nm);
        for j in 0..k {
            let row_s: f64 = (0..nm).map(|m| mix_dir.get(j, m)).sum();
            for m in 0..nm {
                mix_weights.set(j, m, mix_dir.get(j, m) / row_s.max(1e-12));
            }
        }
        diagnose_chain(&mut ctx, &start, &trans, &occup);
        ctx.push(
            Issue::builder(IssueCode::JitterInjected)
                .message("variational GMM-HMM uses Dirichlet(1) and Normal-Gamma floors")
                .compromise(NumericalCompromise::new(
                    "unconstrained variational Bayes",
                    "Dirichlet(1) + diagonal variance floor",
                    "zero cells make the sequence probability vanish",
                    "posterior means are shrunk; do not treat them as MLEs",
                ))
                .build(),
        );
        let tmp = FittedGmmHmm {
            labels: Vector::zeros(0),
            n_states: k,
            n_mix: nm,
            start: start.clone(),
            trans: trans.clone(),
            mix_weights: mix_weights.clone(),
            means: means.clone(),
            vars: vars.clone(),
            loglik: elbo,
        };
        let (labels, _) = viterbi_path(&start, &trans, &tmp.log_emit_seq(x));
        ctx.finish(FittedVariationalGmmHmm {
            labels,
            n_states: k,
            n_mix: nm,
            start,
            trans,
            mix_weights,
            means,
            vars,
            elbo,
        })
    }
}

/// Left-right (Bakis) Gaussian HMM (hmmlearn `GaussianHMM` with `left_right`).
///
/// State count is not treated as an extra identification `p` beyond the
/// inner [`GaussianHmm`] fit.
#[derive(Clone, Debug)]
pub struct GaussianHmmLeftRight {
    inner: GaussianHmm,
}

impl Default for GaussianHmmLeftRight {
    fn default() -> Self {
        Self {
            inner: GaussianHmm::left_right(2),
        }
    }
}

impl GaussianHmmLeftRight {
    /// Left-right Gaussian HMM with `n_states` states.
    pub fn new(n_states: usize) -> Self {
        Self {
            inner: GaussianHmm::left_right(n_states),
        }
    }

    /// Fit alias for [`FitUnsupervised::fit_unsupervised`].
    pub fn fit(&mut self, x: &Matrix, session: &Session) -> Result<Qualified<FittedGaussianHmm>> {
        self.fit_unsupervised(x, session)
    }
}

impl FitUnsupervised for GaussianHmmLeftRight {
    type Fitted = FittedGaussianHmm;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedGaussianHmm>> {
        self.inner.fit_unsupervised(x, session)
    }
}

/// Left-right multinomial HMM (hmmlearn `MultinomialHMM` / `CategoricalHMM`).
#[derive(Clone, Debug)]
pub struct MultinomialHmmLeftRight {
    inner: MultinomialHmm,
}

impl Default for MultinomialHmmLeftRight {
    fn default() -> Self {
        Self {
            inner: MultinomialHmm::left_right(2),
        }
    }
}

impl MultinomialHmmLeftRight {
    /// Left-right multinomial HMM with `n_states` states.
    pub fn new(n_states: usize) -> Self {
        Self {
            inner: MultinomialHmm::left_right(n_states),
        }
    }

    /// Fit alias.
    pub fn fit(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedMultinomialHmm>> {
        self.fit_unsupervised(x, session)
    }
}

impl FitUnsupervised for MultinomialHmmLeftRight {
    type Fitted = FittedMultinomialHmm;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedMultinomialHmm>> {
        self.inner.fit_unsupervised(x, session)
    }
}

/// Left-right GMM-HMM (hmmlearn `GMMHMM` with `left_right`).
///
/// State / mixture counts are not extra identification `p` beyond the inner
/// [`GmmHmm`] fit.
#[derive(Clone, Debug)]
pub struct GmmHmmLeftRight {
    inner: GmmHmm,
}

impl Default for GmmHmmLeftRight {
    fn default() -> Self {
        Self {
            inner: GmmHmm::left_right(2, 1),
        }
    }
}

impl GmmHmmLeftRight {
    /// Left-right GMM-HMM with `n_states` states and `n_mix` mixtures.
    pub fn new(n_states: usize, n_mix: usize) -> Self {
        Self {
            inner: GmmHmm::left_right(n_states, n_mix),
        }
    }

    /// Fit alias.
    pub fn fit(&mut self, x: &Matrix, session: &Session) -> Result<Qualified<FittedGmmHmm>> {
        self.fit_unsupervised(x, session)
    }
}

impl FitUnsupervised for GmmHmmLeftRight {
    type Fitted = FittedGmmHmm;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedGmmHmm>> {
        self.inner.fit_unsupervised(x, session)
    }
}

fn log_gamma_emit(y: f64, shape: f64, rate: f64) -> f64 {
    if !y.is_finite() || y <= 0.0 || shape <= 0.0 || rate <= 0.0 {
        return f64::NEG_INFINITY;
    }
    shape * rate.ln() - crate::special::ln_gamma(shape) + (shape - 1.0) * y.ln() - rate * y
}

/// Gamma-emission HMM (hmmlearn-adjacent; positive continuous observations).
///
/// State count is not identification `p`. Distinct from [`PoissonHmm`]
/// (integer counts) and [`GaussianHmm`] (unrestricted support).
#[derive(Clone, Debug)]
pub struct GammaHmm {
    /// Hidden states. Not identification `p`.
    pub n_states: usize,
    /// Baum–Welch cap.
    pub max_iter: usize,
}

impl Default for GammaHmm {
    fn default() -> Self {
        Self {
            n_states: 2,
            max_iter: 40,
        }
    }
}

impl GammaHmm {
    /// `k`-state gamma HMM.
    pub fn new(n_states: usize) -> Self {
        Self {
            n_states,
            ..Self::default()
        }
    }

    /// Fit alias.
    pub fn fit(&mut self, x: &Matrix, session: &Session) -> Result<Qualified<FittedGammaHmm>> {
        self.fit_unsupervised(x, session)
    }
}

/// Fitted gamma HMM.
#[derive(Clone, Debug)]
pub struct FittedGammaHmm {
    /// Viterbi path.
    pub labels: Vector,
    /// Start distribution.
    pub start: Vector,
    /// Transitions.
    pub trans: Matrix,
    /// Shape \(\alpha_j\).
    pub shapes: Vector,
    /// Rate \(\beta_j\).
    pub rates: Vector,
    /// Training log-likelihood.
    pub loglik: f64,
}

impl FittedGammaHmm {
    fn log_emit_seq(&self, x: &Matrix) -> Vec<Vec<f64>> {
        let t = x.nrows();
        let s = self.shapes.len();
        let mut out = vec![vec![f64::NEG_INFINITY; s]; t];
        for ti in 0..t {
            let y = x.get(ti, 0);
            for j in 0..s {
                out[ti][j] = log_gamma_emit(y, self.shapes[j], self.rates[j]);
            }
        }
        out
    }

    /// Viterbi path.
    pub fn decode(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let ctx = FitCtx::with_session(session.child("decode"));
        let (path, _) = viterbi_path(&self.start, &self.trans, &self.log_emit_seq(x));
        ctx.finish(path)
    }
}

impl Predict for FittedGammaHmm {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        self.decode(x, session)
    }
}

impl FitUnsupervised for GammaHmm {
    type Fitted = FittedGammaHmm;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedGammaHmm>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let t_len = x.nrows();
        let k = self.n_states.max(1);
        if t_len == 0 {
            return ctx.finish(FittedGammaHmm {
                labels: empty_labels(0),
                start: init_start(k),
                trans: init_trans(k),
                shapes: Vector::zeros(k),
                rates: Vector::zeros(k),
                loglik: f64::NAN,
            });
        }
        let mut n_pos = 0usize;
        for i in 0..t_len {
            if x.get(i, 0) > 0.0 && x.get(i, 0).is_finite() {
                n_pos += 1;
            }
        }
        if n_pos < t_len {
            ctx.push(
                Issue::builder(IssueCode::NonPositiveSeries)
                    .severity(signlred::Severity::Warning)
                    .message("GammaHmm skipped non-positive observations in the emission")
                    .build(),
            );
        }
        if n_pos < 2 {
            ctx.push(
                Issue::builder(IssueCode::MeaninglessFit)
                    .message("GammaHmm needs at least two positive observations")
                    .meaninglessness(Meaninglessness::vacuous(
                        "gamma HMM",
                        "shape/rate are unidentified on a non-positive series",
                        "collect positive continuous observations",
                    ))
                    .build(),
            );
            return ctx.finish(FittedGammaHmm {
                labels: empty_labels(t_len),
                start: init_start(k),
                trans: init_trans(k),
                shapes: Vector::filled(k, 1.0),
                rates: Vector::filled(k, 1.0),
                loglik: f64::NAN,
            });
        }
        let mean = x
            .column(0)
            .as_slice()
            .iter()
            .filter(|v| **v > 0.0 && v.is_finite())
            .sum::<f64>()
            / n_pos as f64;
        let mut shapes = Vector::from_iter((0..k).map(|j| 1.5 + 0.5 * j as f64));
        let mut rates = Vector::from_iter((0..k).map(|j| shapes[j] / (mean * (0.6 + 0.3 * j as f64)).max(1e-3)));
        let mut start = init_start(k);
        let mut trans = init_trans(k);
        let mut loglik = f64::NEG_INFINITY;
        let mut last_gamma: Vec<Vec<f64>> = Vec::new();
        for it in 0..self.max_iter.max(1) {
            let log_emit: Vec<Vec<f64>> = (0..t_len)
                .map(|t| {
                    let y = x.get(t, 0);
                    (0..k)
                        .map(|j| log_gamma_emit(y, shapes[j], rates[j]))
                        .collect()
                })
                .collect();
            let Some(fb) = scaled_forward_backward(&mut ctx, &start, &trans, &log_emit) else {
                break;
            };
            loglik = fb.loglik;
            last_gamma = fb.gamma.clone();
            ctx.session.step(it as u64, -loglik, None);
            let mut ns = Vector::zeros(k);
            let mut nt = Matrix::zeros(k, k);
            for j in 0..k {
                ns[j] = fb.gamma[0][j];
            }
            for j in 0..k {
                let mut wsum = 0.0_f64;
                let mut wy = 0.0_f64;
                let mut wy2 = 0.0_f64;
                for t in 0..t_len {
                    let y = x.get(t, 0);
                    if y <= 0.0 || !y.is_finite() {
                        continue;
                    }
                    let w = fb.gamma[t][j];
                    wsum += w;
                    wy += w * y;
                    wy2 += w * y * y;
                }
                if wsum > 1e-12 {
                    let mu = wy / wsum;
                    let var = (wy2 / wsum - mu * mu).max(1e-8);
                    shapes[j] = (mu * mu / var).clamp(0.05, 80.0);
                    rates[j] = (mu / var).clamp(1e-6, 80.0);
                }
            }
            for t in 0..t_len.saturating_sub(1) {
                for i in 0..k {
                    for j in 0..k {
                        nt.set(i, j, nt.get(i, j) + fb.xi[t][i][j]);
                    }
                }
            }
            start = ns;
            trans = nt;
            renormalize_vec(&mut start, TRANS_FLOOR);
            renormalize_rows(&mut trans, TRANS_FLOOR);
        }
        let occup: Vec<f64> = (0..k)
            .map(|j| {
                last_gamma
                    .iter()
                    .map(|g| g.get(j).copied().unwrap_or(0.0))
                    .sum()
            })
            .collect();
        if !last_gamma.is_empty() {
            diagnose_chain(&mut ctx, &start, &trans, &occup);
        }
        let log_emit = {
            let dummy = FittedGammaHmm {
                labels: Vector::zeros(0),
                start: start.clone(),
                trans: trans.clone(),
                shapes: shapes.clone(),
                rates: rates.clone(),
                loglik,
            };
            dummy.log_emit_seq(x)
        };
        let (labels, _) = viterbi_path(&start, &trans, &log_emit);
        ctx.finish(FittedGammaHmm {
            labels,
            start,
            trans,
            shapes,
            rates,
            loglik,
        })
    }
}

fn log_ar_gauss(y: f64, ylag: Option<f64>, mu: f64, phi: f64, var: f64) -> f64 {
    if !y.is_finite() {
        return f64::NEG_INFINITY;
    }
    let m = match ylag {
        Some(yl) if yl.is_finite() => mu + phi * yl,
        _ => mu,
    };
    let v = var.max(COV_FLOOR);
    let z = y - m;
    -0.5 * (LN_2PI + v.ln() + z * z / v)
}

/// Autoregressive Gaussian HMM (AR(1) emissions per state).
///
/// \(y_t\mid s_t\sim\mathcal N(\mu_s+\varphi_s y_{t-1},\sigma_s^2)\).
/// State / lag counts are not identification `p`. Distinct from
/// [`GaussianHmm`] (i.i.d. emissions).
#[derive(Clone, Debug)]
pub struct AutoregressiveHmm {
    /// Hidden states. Not identification `p`.
    pub n_states: usize,
    /// Baum–Welch cap.
    pub max_iter: usize,
}

impl Default for AutoregressiveHmm {
    fn default() -> Self {
        Self {
            n_states: 2,
            max_iter: 40,
        }
    }
}

impl AutoregressiveHmm {
    /// `k`-state AR-Gaussian HMM.
    pub fn new(n_states: usize) -> Self {
        Self {
            n_states,
            ..Self::default()
        }
    }

    /// Fit alias.
    pub fn fit(&mut self, x: &Matrix, session: &Session) -> Result<Qualified<FittedArHmm>> {
        self.fit_unsupervised(x, session)
    }
}

/// Fitted AR-Gaussian HMM.
#[derive(Clone, Debug)]
pub struct FittedArHmm {
    /// Viterbi path.
    pub labels: Vector,
    /// Start distribution.
    pub start: Vector,
    /// Transitions.
    pub trans: Matrix,
    /// Intercept \(\mu_j\).
    pub mu: Vector,
    /// Autoregressive slope \(\varphi_j\).
    pub phi: Vector,
    /// Innovation variance.
    pub var: Vector,
    /// Training log-likelihood.
    pub loglik: f64,
}

impl FittedArHmm {
    fn log_emit_seq(&self, x: &Matrix) -> Vec<Vec<f64>> {
        let t = x.nrows();
        let s = self.mu.len();
        let mut out = vec![vec![f64::NEG_INFINITY; s]; t];
        for ti in 0..t {
            let y = x.get(ti, 0);
            let ylag = if ti > 0 {
                Some(x.get(ti - 1, 0))
            } else {
                None
            };
            for j in 0..s {
                out[ti][j] = log_ar_gauss(y, ylag, self.mu[j], self.phi[j], self.var[j]);
            }
        }
        out
    }

    /// Viterbi path.
    pub fn decode(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let ctx = FitCtx::with_session(session.child("decode"));
        let (path, _) = viterbi_path(&self.start, &self.trans, &self.log_emit_seq(x));
        ctx.finish(path)
    }
}

impl Predict for FittedArHmm {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        self.decode(x, session)
    }
}

impl FitUnsupervised for AutoregressiveHmm {
    type Fitted = FittedArHmm;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedArHmm>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let t_len = x.nrows();
        let k = self.n_states.max(1);
        if t_len == 0 {
            return ctx.finish(FittedArHmm {
                labels: empty_labels(0),
                start: init_start(k),
                trans: init_trans(k),
                mu: Vector::zeros(k),
                phi: Vector::zeros(k),
                var: Vector::filled(k, 1.0),
                loglik: f64::NAN,
            });
        }
        let mean = x.column(0).mean();
        let mut mu = Vector::from_iter((0..k).map(|j| mean + (j as f64 - 0.5) * x.column(0).std().max(0.1)));
        let mut phi = Vector::zeros(k);
        let mut var = Vector::filled(k, x.column(0).std().max(0.1).powi(2));
        let mut start = init_start(k);
        let mut trans = init_trans(k);
        let mut loglik = f64::NEG_INFINITY;
        let mut last_gamma: Vec<Vec<f64>> = Vec::new();
        for it in 0..self.max_iter.max(1) {
            let log_emit: Vec<Vec<f64>> = (0..t_len)
                .map(|t| {
                    let y = x.get(t, 0);
                    let ylag = if t > 0 { Some(x.get(t - 1, 0)) } else { None };
                    (0..k)
                        .map(|j| log_ar_gauss(y, ylag, mu[j], phi[j], var[j]))
                        .collect()
                })
                .collect();
            let Some(fb) = scaled_forward_backward(&mut ctx, &start, &trans, &log_emit) else {
                break;
            };
            loglik = fb.loglik;
            last_gamma = fb.gamma.clone();
            ctx.session.step(it as u64, -loglik, None);
            let mut ns = Vector::zeros(k);
            let mut nt = Matrix::zeros(k, k);
            for j in 0..k {
                ns[j] = fb.gamma[0][j];
            }
            for j in 0..k {
                let mut sw = 0.0_f64;
                let mut sx = 0.0_f64;
                let mut sy = 0.0_f64;
                let mut sxx = 0.0_f64;
                let mut sxy = 0.0_f64;
                for t in 1..t_len {
                    let y = x.get(t, 0);
                    let xl = x.get(t - 1, 0);
                    if !y.is_finite() || !xl.is_finite() {
                        continue;
                    }
                    let w = fb.gamma[t][j];
                    sw += w;
                    sx += w * xl;
                    sy += w * y;
                    sxx += w * xl * xl;
                    sxy += w * xl * y;
                }
                let den = sw * sxx - sx * sx;
                if den.abs() > 1e-12 && sw > 1e-12 {
                    mu[j] = (sxx * sy - sx * sxy) / den;
                    phi[j] = ((sw * sxy - sx * sy) / den).clamp(-0.99, 0.99);
                } else if sw > 1e-12 {
                    mu[j] = sy / sw;
                    phi[j] = 0.0;
                }
                let mut sse = 0.0_f64;
                let mut w2 = 0.0_f64;
                for t in 1..t_len {
                    let y = x.get(t, 0);
                    let xl = x.get(t - 1, 0);
                    if !y.is_finite() || !xl.is_finite() {
                        continue;
                    }
                    let w = fb.gamma[t][j];
                    let r = y - mu[j] - phi[j] * xl;
                    sse += w * r * r;
                    w2 += w;
                }
                var[j] = if w2 > 1e-12 {
                    (sse / w2).max(COV_FLOOR)
                } else {
                    COV_FLOOR
                };
            }
            for t in 0..t_len.saturating_sub(1) {
                for i in 0..k {
                    for j in 0..k {
                        nt.set(i, j, nt.get(i, j) + fb.xi[t][i][j]);
                    }
                }
            }
            start = ns;
            trans = nt;
            renormalize_vec(&mut start, TRANS_FLOOR);
            renormalize_rows(&mut trans, TRANS_FLOOR);
        }
        let occup: Vec<f64> = (0..k)
            .map(|j| {
                last_gamma
                    .iter()
                    .map(|g| g.get(j).copied().unwrap_or(0.0))
                    .sum()
            })
            .collect();
        if !last_gamma.is_empty() {
            diagnose_chain(&mut ctx, &start, &trans, &occup);
        }
        let dummy = FittedArHmm {
            labels: Vector::zeros(0),
            start: start.clone(),
            trans: trans.clone(),
            mu: mu.clone(),
            phi: phi.clone(),
            var: var.clone(),
            loglik,
        };
        let (labels, _) = viterbi_path(&start, &trans, &dummy.log_emit_seq(x));
        ctx.finish(FittedArHmm {
            labels,
            start,
            trans,
            mu,
            phi,
            var,
            loglik,
        })
    }
}

const LN_PI: f64 = 1.1447298858494002;

fn log_student_t(y: f64, mu: f64, var: f64, nu: f64) -> f64 {
    if !y.is_finite() || !mu.is_finite() || var <= 0.0 || nu <= 0.0 {
        return f64::NEG_INFINITY;
    }
    let z = (y - mu) * (y - mu) / var;
    crate::special::ln_gamma(0.5 * (nu + 1.0))
        - crate::special::ln_gamma(0.5 * nu)
        - 0.5 * (nu.ln() + LN_PI)
        - 0.5 * var.ln()
        - 0.5 * (nu + 1.0) * (1.0 + z / nu).ln()
}

fn log_exponential(y: f64, rate: f64) -> f64 {
    if !y.is_finite() || y < 0.0 || rate <= 0.0 {
        return f64::NEG_INFINITY;
    }
    rate.ln() - rate * y
}

fn log_invgauss(y: f64, mu: f64, lambda: f64) -> f64 {
    if !y.is_finite() || y <= 0.0 || mu <= 0.0 || lambda <= 0.0 {
        return f64::NEG_INFINITY;
    }
    0.5 * (lambda.ln() - LN_2PI - 3.0 * y.ln()) - lambda * (y - mu) * (y - mu) / (2.0 * mu * mu * y)
}

fn log_i0(kappa: f64) -> f64 {
    let k = kappa.abs();
    if k < 12.0 {
        let mut term = 1.0_f64;
        let mut s = 1.0_f64;
        let x = 0.25 * k * k;
        for n in 1..24 {
            let nf = n as f64;
            term *= x / (nf * nf);
            s += term;
            if term < 1e-16 * s {
                break;
            }
        }
        s.max(1e-300).ln()
    } else {
        k - 0.5 * (LN_2PI + k.ln())
    }
}

fn log_von_mises(y: f64, mu: f64, kappa: f64) -> f64 {
    if !y.is_finite() || !mu.is_finite() || kappa <= 0.0 {
        return f64::NEG_INFINITY;
    }
    kappa * (y - mu).cos() - LN_2PI - log_i0(kappa)
}

/// Student-\(t\) emission HMM (heavy-tailed hmmlearn-adjacent).
///
/// Degrees of freedom are a hyperparameter, not identification `p`. Distinct
/// from [`GaussianHmm`] (finite fourth moment) and [`GammaHmm`] (positive support).
#[derive(Clone, Debug)]
pub struct StudentTHmm {
    /// Hidden states. Not identification `p`.
    pub n_states: usize,
    /// Degrees of freedom \(\nu>0\). Not identification `p`.
    pub df: f64,
    /// Baum–Welch cap.
    pub max_iter: usize,
}

impl Default for StudentTHmm {
    fn default() -> Self {
        Self {
            n_states: 2,
            df: 4.0,
            max_iter: 40,
        }
    }
}

impl StudentTHmm {
    /// `k`-state Student-\(t\) HMM with \(\nu=4\).
    pub fn new(n_states: usize) -> Self {
        Self {
            n_states,
            ..Self::default()
        }
    }

    /// Fit alias.
    pub fn fit(&mut self, x: &Matrix, session: &Session) -> Result<Qualified<FittedStudentTHmm>> {
        self.fit_unsupervised(x, session)
    }
}

/// Fitted Student-\(t\) HMM.
#[derive(Clone, Debug)]
pub struct FittedStudentTHmm {
    /// Viterbi path.
    pub labels: Vector,
    /// Start distribution.
    pub start: Vector,
    /// Transitions.
    pub trans: Matrix,
    /// Location \(\mu_j\).
    pub mu: Vector,
    /// Scale \(\sigma_j^2\).
    pub var: Vector,
    /// Degrees of freedom.
    pub df: f64,
    /// Training log-likelihood.
    pub loglik: f64,
}

impl FittedStudentTHmm {
    fn log_emit_seq(&self, x: &Matrix) -> Vec<Vec<f64>> {
        let t = x.nrows();
        let s = self.mu.len();
        let mut out = vec![vec![f64::NEG_INFINITY; s]; t];
        for ti in 0..t {
            let y = x.get(ti, 0);
            for j in 0..s {
                out[ti][j] = log_student_t(y, self.mu[j], self.var[j], self.df);
            }
        }
        out
    }

    /// Viterbi path.
    pub fn decode(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let ctx = FitCtx::with_session(session.child("decode"));
        let (path, _) = viterbi_path(&self.start, &self.trans, &self.log_emit_seq(x));
        ctx.finish(path)
    }
}

impl Predict for FittedStudentTHmm {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        self.decode(x, session)
    }
}

impl FitUnsupervised for StudentTHmm {
    type Fitted = FittedStudentTHmm;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedStudentTHmm>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let t_len = x.nrows();
        let k = self.n_states.max(1);
        let nu = if self.df.is_finite() && self.df > 0.5 {
            self.df
        } else {
            4.0
        };
        if t_len == 0 {
            return ctx.finish(FittedStudentTHmm {
                labels: empty_labels(0),
                start: init_start(k),
                trans: init_trans(k),
                mu: Vector::zeros(k),
                var: Vector::filled(k, 1.0),
                df: nu,
                loglik: f64::NAN,
            });
        }
        let mean = x.column(0).mean();
        let sd = x.column(0).std().max(0.1);
        let mut mu = Vector::from_iter((0..k).map(|j| mean + (j as f64 - 0.5) * sd));
        let mut var = Vector::filled(k, sd * sd);
        let mut start = init_start(k);
        let mut trans = init_trans(k);
        let mut loglik = f64::NEG_INFINITY;
        let mut last_gamma: Vec<Vec<f64>> = Vec::new();
        for it in 0..self.max_iter.max(1) {
            let log_emit: Vec<Vec<f64>> = (0..t_len)
                .map(|t| {
                    let y = x.get(t, 0);
                    (0..k)
                        .map(|j| log_student_t(y, mu[j], var[j], nu))
                        .collect()
                })
                .collect();
            let Some(fb) = scaled_forward_backward(&mut ctx, &start, &trans, &log_emit) else {
                break;
            };
            loglik = fb.loglik;
            last_gamma = fb.gamma.clone();
            ctx.session.step(it as u64, -loglik, None);
            let mut ns = Vector::zeros(k);
            let mut nt = Matrix::zeros(k, k);
            for j in 0..k {
                ns[j] = fb.gamma[0][j];
            }
            for j in 0..k {
                let mut wu = 0.0_f64;
                let mut wuy = 0.0_f64;
                let mut wsum = 0.0_f64;
                for t in 0..t_len {
                    let y = x.get(t, 0);
                    if !y.is_finite() {
                        continue;
                    }
                    let g = fb.gamma[t][j];
                    let delta = (y - mu[j]) * (y - mu[j]) / var[j].max(COV_FLOOR);
                    let u = (nu + 1.0) / (nu + delta).max(1e-8);
                    wu += g * u;
                    wuy += g * u * y;
                    wsum += g;
                }
                if wu > 1e-12 {
                    mu[j] = wuy / wu;
                }
                let mut sse = 0.0_f64;
                for t in 0..t_len {
                    let y = x.get(t, 0);
                    if !y.is_finite() {
                        continue;
                    }
                    let g = fb.gamma[t][j];
                    let delta = (y - mu[j]) * (y - mu[j]) / var[j].max(COV_FLOOR);
                    let u = (nu + 1.0) / (nu + delta).max(1e-8);
                    let r = y - mu[j];
                    sse += g * u * r * r;
                }
                var[j] = if wsum > 1e-12 {
                    (sse / wsum).max(COV_FLOOR)
                } else {
                    COV_FLOOR
                };
            }
            for t in 0..t_len.saturating_sub(1) {
                for i in 0..k {
                    for j in 0..k {
                        nt.set(i, j, nt.get(i, j) + fb.xi[t][i][j]);
                    }
                }
            }
            start = ns;
            trans = nt;
            renormalize_vec(&mut start, TRANS_FLOOR);
            renormalize_rows(&mut trans, TRANS_FLOOR);
        }
        let occup: Vec<f64> = (0..k)
            .map(|j| {
                last_gamma
                    .iter()
                    .map(|g| g.get(j).copied().unwrap_or(0.0))
                    .sum()
            })
            .collect();
        if !last_gamma.is_empty() {
            diagnose_chain(&mut ctx, &start, &trans, &occup);
        }
        let dummy = FittedStudentTHmm {
            labels: Vector::zeros(0),
            start: start.clone(),
            trans: trans.clone(),
            mu: mu.clone(),
            var: var.clone(),
            df: nu,
            loglik,
        };
        let (labels, _) = viterbi_path(&start, &trans, &dummy.log_emit_seq(x));
        ctx.finish(FittedStudentTHmm {
            labels,
            start,
            trans,
            mu,
            var,
            df: nu,
            loglik,
        })
    }
}

/// Exponential-emission HMM (positive waiting times).
///
/// State count is not identification `p`. Distinct from [`PoissonHmm`]
/// (integer counts) and [`GammaHmm`] (two-parameter shape/rate).
#[derive(Clone, Debug)]
pub struct ExponentialHmm {
    /// Hidden states. Not identification `p`.
    pub n_states: usize,
    /// Baum–Welch cap.
    pub max_iter: usize,
}

impl Default for ExponentialHmm {
    fn default() -> Self {
        Self {
            n_states: 2,
            max_iter: 40,
        }
    }
}

impl ExponentialHmm {
    /// `k`-state exponential HMM.
    pub fn new(n_states: usize) -> Self {
        Self {
            n_states,
            ..Self::default()
        }
    }

    /// Fit alias.
    pub fn fit(&mut self, x: &Matrix, session: &Session) -> Result<Qualified<FittedExponentialHmm>> {
        self.fit_unsupervised(x, session)
    }
}

/// Fitted exponential HMM.
#[derive(Clone, Debug)]
pub struct FittedExponentialHmm {
    /// Viterbi path.
    pub labels: Vector,
    /// Start distribution.
    pub start: Vector,
    /// Transitions.
    pub trans: Matrix,
    /// Rate \(\lambda_j\).
    pub rates: Vector,
    /// Training log-likelihood.
    pub loglik: f64,
}

impl FittedExponentialHmm {
    fn log_emit_seq(&self, x: &Matrix) -> Vec<Vec<f64>> {
        let t = x.nrows();
        let s = self.rates.len();
        let mut out = vec![vec![f64::NEG_INFINITY; s]; t];
        for ti in 0..t {
            let y = x.get(ti, 0);
            for j in 0..s {
                out[ti][j] = log_exponential(y, self.rates[j]);
            }
        }
        out
    }

    /// Viterbi path.
    pub fn decode(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let ctx = FitCtx::with_session(session.child("decode"));
        let (path, _) = viterbi_path(&self.start, &self.trans, &self.log_emit_seq(x));
        ctx.finish(path)
    }
}

impl Predict for FittedExponentialHmm {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        self.decode(x, session)
    }
}

impl FitUnsupervised for ExponentialHmm {
    type Fitted = FittedExponentialHmm;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedExponentialHmm>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let t_len = x.nrows();
        let k = self.n_states.max(1);
        if t_len == 0 {
            return ctx.finish(FittedExponentialHmm {
                labels: empty_labels(0),
                start: init_start(k),
                trans: init_trans(k),
                rates: Vector::filled(k, 1.0),
                loglik: f64::NAN,
            });
        }
        let mut n_pos = 0usize;
        for i in 0..t_len {
            if x.get(i, 0) >= 0.0 && x.get(i, 0).is_finite() {
                n_pos += 1;
            }
        }
        if n_pos < t_len {
            ctx.push(
                Issue::builder(IssueCode::NonPositiveSeries)
                    .severity(signlred::Severity::Warning)
                    .message("ExponentialHmm skipped negative observations in the emission")
                    .build(),
            );
        }
        if n_pos < 2 {
            ctx.push(
                Issue::builder(IssueCode::MeaninglessFit)
                    .message("ExponentialHmm needs at least two non-negative observations")
                    .meaninglessness(Meaninglessness::vacuous(
                        "exponential HMM",
                        "the rate is unidentified on a negative series",
                        "collect non-negative waiting times",
                    ))
                    .build(),
            );
            return ctx.finish(FittedExponentialHmm {
                labels: empty_labels(t_len),
                start: init_start(k),
                trans: init_trans(k),
                rates: Vector::filled(k, 1.0),
                loglik: f64::NAN,
            });
        }
        let mean = x
            .column(0)
            .as_slice()
            .iter()
            .filter(|v| **v >= 0.0 && v.is_finite())
            .sum::<f64>()
            / n_pos as f64;
        let mut rates = Vector::from_iter((0..k).map(|j| {
            1.0 / (mean * (0.6 + 0.4 * j as f64)).max(1e-3)
        }));
        let mut start = init_start(k);
        let mut trans = init_trans(k);
        let mut loglik = f64::NEG_INFINITY;
        let mut last_gamma: Vec<Vec<f64>> = Vec::new();
        for it in 0..self.max_iter.max(1) {
            let log_emit: Vec<Vec<f64>> = (0..t_len)
                .map(|t| {
                    let y = x.get(t, 0);
                    (0..k)
                        .map(|j| log_exponential(y, rates[j]))
                        .collect()
                })
                .collect();
            let Some(fb) = scaled_forward_backward(&mut ctx, &start, &trans, &log_emit) else {
                break;
            };
            loglik = fb.loglik;
            last_gamma = fb.gamma.clone();
            ctx.session.step(it as u64, -loglik, None);
            let mut ns = Vector::zeros(k);
            let mut nt = Matrix::zeros(k, k);
            for j in 0..k {
                ns[j] = fb.gamma[0][j];
            }
            for j in 0..k {
                let mut wsum = 0.0_f64;
                let mut wy = 0.0_f64;
                for t in 0..t_len {
                    let y = x.get(t, 0);
                    if y < 0.0 || !y.is_finite() {
                        continue;
                    }
                    let w = fb.gamma[t][j];
                    wsum += w;
                    wy += w * y;
                }
                if wsum > 1e-12 && wy > 1e-12 {
                    rates[j] = (wsum / wy).clamp(1e-6, 80.0);
                }
            }
            for t in 0..t_len.saturating_sub(1) {
                for i in 0..k {
                    for j in 0..k {
                        nt.set(i, j, nt.get(i, j) + fb.xi[t][i][j]);
                    }
                }
            }
            start = ns;
            trans = nt;
            renormalize_vec(&mut start, TRANS_FLOOR);
            renormalize_rows(&mut trans, TRANS_FLOOR);
        }
        let occup: Vec<f64> = (0..k)
            .map(|j| {
                last_gamma
                    .iter()
                    .map(|g| g.get(j).copied().unwrap_or(0.0))
                    .sum()
            })
            .collect();
        if !last_gamma.is_empty() {
            diagnose_chain(&mut ctx, &start, &trans, &occup);
        }
        let dummy = FittedExponentialHmm {
            labels: Vector::zeros(0),
            start: start.clone(),
            trans: trans.clone(),
            rates: rates.clone(),
            loglik,
        };
        let (labels, _) = viterbi_path(&start, &trans, &dummy.log_emit_seq(x));
        ctx.finish(FittedExponentialHmm {
            labels,
            start,
            trans,
            rates,
            loglik,
        })
    }
}

/// Inverse-Gaussian emission HMM (positive hitting times).
///
/// State count is not identification `p`. Distinct from [`GammaHmm`]
/// (shape/rate) and [`ExponentialHmm`] (memoryless).
#[derive(Clone, Debug)]
pub struct InverseGaussianHmm {
    /// Hidden states. Not identification `p`.
    pub n_states: usize,
    /// Baum–Welch cap.
    pub max_iter: usize,
}

impl Default for InverseGaussianHmm {
    fn default() -> Self {
        Self {
            n_states: 2,
            max_iter: 40,
        }
    }
}

impl InverseGaussianHmm {
    /// `k`-state inverse-Gaussian HMM.
    pub fn new(n_states: usize) -> Self {
        Self {
            n_states,
            ..Self::default()
        }
    }

    /// Fit alias.
    pub fn fit(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedInverseGaussianHmm>> {
        self.fit_unsupervised(x, session)
    }
}

/// Fitted inverse-Gaussian HMM.
#[derive(Clone, Debug)]
pub struct FittedInverseGaussianHmm {
    /// Viterbi path.
    pub labels: Vector,
    /// Start distribution.
    pub start: Vector,
    /// Transitions.
    pub trans: Matrix,
    /// Mean \(\mu_j\).
    pub mu: Vector,
    /// Shape \(\lambda_j\).
    pub lambda: Vector,
    /// Training log-likelihood.
    pub loglik: f64,
}

impl FittedInverseGaussianHmm {
    fn log_emit_seq(&self, x: &Matrix) -> Vec<Vec<f64>> {
        let t = x.nrows();
        let s = self.mu.len();
        let mut out = vec![vec![f64::NEG_INFINITY; s]; t];
        for ti in 0..t {
            let y = x.get(ti, 0);
            for j in 0..s {
                out[ti][j] = log_invgauss(y, self.mu[j], self.lambda[j]);
            }
        }
        out
    }

    /// Viterbi path.
    pub fn decode(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let ctx = FitCtx::with_session(session.child("decode"));
        let (path, _) = viterbi_path(&self.start, &self.trans, &self.log_emit_seq(x));
        ctx.finish(path)
    }
}

impl Predict for FittedInverseGaussianHmm {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        self.decode(x, session)
    }
}

impl FitUnsupervised for InverseGaussianHmm {
    type Fitted = FittedInverseGaussianHmm;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedInverseGaussianHmm>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let t_len = x.nrows();
        let k = self.n_states.max(1);
        if t_len == 0 {
            return ctx.finish(FittedInverseGaussianHmm {
                labels: empty_labels(0),
                start: init_start(k),
                trans: init_trans(k),
                mu: Vector::filled(k, 1.0),
                lambda: Vector::filled(k, 1.0),
                loglik: f64::NAN,
            });
        }
        let mut n_pos = 0usize;
        for i in 0..t_len {
            if x.get(i, 0) > 0.0 && x.get(i, 0).is_finite() {
                n_pos += 1;
            }
        }
        if n_pos < t_len {
            ctx.push(
                Issue::builder(IssueCode::NonPositiveSeries)
                    .severity(signlred::Severity::Warning)
                    .message("InverseGaussianHmm skipped non-positive observations")
                    .build(),
            );
        }
        if n_pos < 2 {
            ctx.push(
                Issue::builder(IssueCode::MeaninglessFit)
                    .message("InverseGaussianHmm needs at least two positive observations")
                    .meaninglessness(Meaninglessness::vacuous(
                        "inverse-Gaussian HMM",
                        "μ and λ are unidentified on a non-positive series",
                        "collect positive hitting times",
                    ))
                    .build(),
            );
            return ctx.finish(FittedInverseGaussianHmm {
                labels: empty_labels(t_len),
                start: init_start(k),
                trans: init_trans(k),
                mu: Vector::filled(k, 1.0),
                lambda: Vector::filled(k, 1.0),
                loglik: f64::NAN,
            });
        }
        let mean = x
            .column(0)
            .as_slice()
            .iter()
            .filter(|v| **v > 0.0 && v.is_finite())
            .sum::<f64>()
            / n_pos as f64;
        let mut mu = Vector::from_iter((0..k).map(|j| (mean * (0.7 + 0.4 * j as f64)).max(1e-3)));
        let mut lambda = Vector::filled(k, mean.max(1e-3));
        let mut start = init_start(k);
        let mut trans = init_trans(k);
        let mut loglik = f64::NEG_INFINITY;
        let mut last_gamma: Vec<Vec<f64>> = Vec::new();
        for it in 0..self.max_iter.max(1) {
            let log_emit: Vec<Vec<f64>> = (0..t_len)
                .map(|t| {
                    let y = x.get(t, 0);
                    (0..k)
                        .map(|j| log_invgauss(y, mu[j], lambda[j]))
                        .collect()
                })
                .collect();
            let Some(fb) = scaled_forward_backward(&mut ctx, &start, &trans, &log_emit) else {
                break;
            };
            loglik = fb.loglik;
            last_gamma = fb.gamma.clone();
            ctx.session.step(it as u64, -loglik, None);
            let mut ns = Vector::zeros(k);
            let mut nt = Matrix::zeros(k, k);
            for j in 0..k {
                ns[j] = fb.gamma[0][j];
            }
            for j in 0..k {
                let mut wsum = 0.0_f64;
                let mut wy = 0.0_f64;
                for t in 0..t_len {
                    let y = x.get(t, 0);
                    if y <= 0.0 || !y.is_finite() {
                        continue;
                    }
                    let w = fb.gamma[t][j];
                    wsum += w;
                    wy += w * y;
                }
                if wsum > 1e-12 {
                    mu[j] = (wy / wsum).max(1e-6);
                }
                let mut inv = 0.0_f64;
                for t in 0..t_len {
                    let y = x.get(t, 0);
                    if y <= 0.0 || !y.is_finite() {
                        continue;
                    }
                    let w = fb.gamma[t][j];
                    let r = y - mu[j];
                    inv += w * r * r / (mu[j] * mu[j] * y);
                }
                lambda[j] = if inv > 1e-12 {
                    (wsum / inv).clamp(1e-6, 1e6)
                } else {
                    1.0
                };
            }
            for t in 0..t_len.saturating_sub(1) {
                for i in 0..k {
                    for j in 0..k {
                        nt.set(i, j, nt.get(i, j) + fb.xi[t][i][j]);
                    }
                }
            }
            start = ns;
            trans = nt;
            renormalize_vec(&mut start, TRANS_FLOOR);
            renormalize_rows(&mut trans, TRANS_FLOOR);
        }
        let occup: Vec<f64> = (0..k)
            .map(|j| {
                last_gamma
                    .iter()
                    .map(|g| g.get(j).copied().unwrap_or(0.0))
                    .sum()
            })
            .collect();
        if !last_gamma.is_empty() {
            diagnose_chain(&mut ctx, &start, &trans, &occup);
        }
        let dummy = FittedInverseGaussianHmm {
            labels: Vector::zeros(0),
            start: start.clone(),
            trans: trans.clone(),
            mu: mu.clone(),
            lambda: lambda.clone(),
            loglik,
        };
        let (labels, _) = viterbi_path(&start, &trans, &dummy.log_emit_seq(x));
        ctx.finish(FittedInverseGaussianHmm {
            labels,
            start,
            trans,
            mu,
            lambda,
            loglik,
        })
    }
}

/// von Mises (circular) emission HMM.
///
/// Concentration \(\kappa\) is not identification `p`. Distinct from
/// [`GaussianHmm`] (unwrapped Euclidean).
#[derive(Clone, Debug)]
pub struct CircularHmm {
    /// Hidden states. Not identification `p`.
    pub n_states: usize,
    /// Baum–Welch cap.
    pub max_iter: usize,
}

impl Default for CircularHmm {
    fn default() -> Self {
        Self {
            n_states: 2,
            max_iter: 40,
        }
    }
}

impl CircularHmm {
    /// `k`-state von Mises HMM.
    pub fn new(n_states: usize) -> Self {
        Self {
            n_states,
            ..Self::default()
        }
    }

    /// Fit alias.
    pub fn fit(&mut self, x: &Matrix, session: &Session) -> Result<Qualified<FittedCircularHmm>> {
        self.fit_unsupervised(x, session)
    }
}

/// Fitted von Mises HMM.
#[derive(Clone, Debug)]
pub struct FittedCircularHmm {
    /// Viterbi path.
    pub labels: Vector,
    /// Start distribution.
    pub start: Vector,
    /// Transitions.
    pub trans: Matrix,
    /// Mean direction \(\mu_j\).
    pub mu: Vector,
    /// Concentration \(\kappa_j\).
    pub kappa: Vector,
    /// Training log-likelihood.
    pub loglik: f64,
}

impl FittedCircularHmm {
    fn log_emit_seq(&self, x: &Matrix) -> Vec<Vec<f64>> {
        let t = x.nrows();
        let s = self.mu.len();
        let mut out = vec![vec![f64::NEG_INFINITY; s]; t];
        for ti in 0..t {
            let y = x.get(ti, 0);
            for j in 0..s {
                out[ti][j] = log_von_mises(y, self.mu[j], self.kappa[j]);
            }
        }
        out
    }

    /// Viterbi path.
    pub fn decode(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let ctx = FitCtx::with_session(session.child("decode"));
        let (path, _) = viterbi_path(&self.start, &self.trans, &self.log_emit_seq(x));
        ctx.finish(path)
    }
}

impl Predict for FittedCircularHmm {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        self.decode(x, session)
    }
}

impl FitUnsupervised for CircularHmm {
    type Fitted = FittedCircularHmm;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedCircularHmm>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let t_len = x.nrows();
        let k = self.n_states.max(1);
        if t_len == 0 {
            return ctx.finish(FittedCircularHmm {
                labels: empty_labels(0),
                start: init_start(k),
                trans: init_trans(k),
                mu: Vector::zeros(k),
                kappa: Vector::filled(k, 1.0),
                loglik: f64::NAN,
            });
        }
        let mut mu = Vector::from_iter((0..k).map(|j| {
            if j == 0 {
                -2.0
            } else {
                2.0
            }
        }));
        let mut kappa = Vector::filled(k, 2.0);
        let mut start = init_start(k);
        let mut trans = init_trans(k);
        let mut loglik = f64::NEG_INFINITY;
        let mut last_gamma: Vec<Vec<f64>> = Vec::new();
        for it in 0..self.max_iter.max(1) {
            let log_emit: Vec<Vec<f64>> = (0..t_len)
                .map(|t| {
                    let y = x.get(t, 0);
                    (0..k)
                        .map(|j| log_von_mises(y, mu[j], kappa[j]))
                        .collect()
                })
                .collect();
            let Some(fb) = scaled_forward_backward(&mut ctx, &start, &trans, &log_emit) else {
                break;
            };
            loglik = fb.loglik;
            last_gamma = fb.gamma.clone();
            ctx.session.step(it as u64, -loglik, None);
            let mut ns = Vector::zeros(k);
            let mut nt = Matrix::zeros(k, k);
            for j in 0..k {
                ns[j] = fb.gamma[0][j];
            }
            for j in 0..k {
                let mut c = 0.0_f64;
                let mut s = 0.0_f64;
                let mut wsum = 0.0_f64;
                for t in 0..t_len {
                    let y = x.get(t, 0);
                    if !y.is_finite() {
                        continue;
                    }
                    let w = fb.gamma[t][j];
                    c += w * y.cos();
                    s += w * y.sin();
                    wsum += w;
                }
                if wsum > 1e-12 {
                    mu[j] = s.atan2(c);
                    let r = (c * c + s * s).sqrt() / wsum;
                    let r = r.clamp(0.0, 0.999);
                    kappa[j] = if r < 1e-6 {
                        1e-3
                    } else {
                        (r * (2.0 - r * r) / (1.0 - r * r).max(1e-8)).clamp(1e-3, 80.0)
                    };
                }
            }
            for t in 0..t_len.saturating_sub(1) {
                for i in 0..k {
                    for j in 0..k {
                        nt.set(i, j, nt.get(i, j) + fb.xi[t][i][j]);
                    }
                }
            }
            start = ns;
            trans = nt;
            renormalize_vec(&mut start, TRANS_FLOOR);
            renormalize_rows(&mut trans, TRANS_FLOOR);
        }
        let occup: Vec<f64> = (0..k)
            .map(|j| {
                last_gamma
                    .iter()
                    .map(|g| g.get(j).copied().unwrap_or(0.0))
                    .sum()
            })
            .collect();
        if !last_gamma.is_empty() {
            diagnose_chain(&mut ctx, &start, &trans, &occup);
        }
        let dummy = FittedCircularHmm {
            labels: Vector::zeros(0),
            start: start.clone(),
            trans: trans.clone(),
            mu: mu.clone(),
            kappa: kappa.clone(),
            loglik,
        };
        let (labels, _) = viterbi_path(&start, &trans, &dummy.log_emit_seq(x));
        ctx.finish(FittedCircularHmm {
            labels,
            start,
            trans,
            mu,
            kappa,
            loglik,
        })
    }
}

/// Sticky HMM: Gaussian emissions with a self-transition bias \(\kappa\).
///
/// Fox–Sudderth–Jordan sticky prior lite: add \(\kappa\) to the diagonal of
/// the transition counts before renormalizing. \(\kappa\) is not identification
/// `p`. Distinct from [`GaussianHmm`] (no sticky bias) and [`Hsmm`]
/// (explicit duration).
#[derive(Clone, Debug)]
pub struct StickyHmm {
    /// Hidden states. Not identification `p`.
    pub n_states: usize,
    /// Sticky self-transition mass. Not identification `p`.
    pub kappa: f64,
    /// Baum–Welch cap.
    pub max_iter: usize,
}

impl Default for StickyHmm {
    fn default() -> Self {
        Self {
            n_states: 2,
            kappa: 4.0,
            max_iter: 40,
        }
    }
}

impl StickyHmm {
    /// `k`-state sticky Gaussian HMM.
    pub fn new(n_states: usize) -> Self {
        Self {
            n_states,
            ..Self::default()
        }
    }

    /// Fit alias.
    pub fn fit(&mut self, x: &Matrix, session: &Session) -> Result<Qualified<FittedStickyHmm>> {
        self.fit_unsupervised(x, session)
    }
}

/// Fitted sticky Gaussian HMM.
#[derive(Clone, Debug)]
pub struct FittedStickyHmm {
    /// Viterbi path.
    pub labels: Vector,
    /// Start distribution.
    pub start: Vector,
    /// Sticky transitions.
    pub trans: Matrix,
    /// Mean \(\mu_j\).
    pub mu: Vector,
    /// Variance \(\sigma_j^2\).
    pub var: Vector,
    /// Sticky mass used at the last M-step.
    pub kappa: f64,
    /// Training log-likelihood.
    pub loglik: f64,
}

impl FittedStickyHmm {
    fn log_emit_seq(&self, x: &Matrix) -> Vec<Vec<f64>> {
        let t = x.nrows();
        let s = self.mu.len();
        let mut out = vec![vec![f64::NEG_INFINITY; s]; t];
        for ti in 0..t {
            let y = x.get(ti, 0);
            for j in 0..s {
                out[ti][j] = log_ar_gauss(y, None, self.mu[j], 0.0, self.var[j]);
            }
        }
        out
    }

    /// Viterbi path.
    pub fn decode(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let ctx = FitCtx::with_session(session.child("decode"));
        let (path, _) = viterbi_path(&self.start, &self.trans, &self.log_emit_seq(x));
        ctx.finish(path)
    }
}

impl Predict for FittedStickyHmm {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        self.decode(x, session)
    }
}

impl FitUnsupervised for StickyHmm {
    type Fitted = FittedStickyHmm;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedStickyHmm>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let t_len = x.nrows();
        let k = self.n_states.max(1);
        let kappa = if self.kappa.is_finite() && self.kappa >= 0.0 {
            self.kappa
        } else {
            4.0
        };
        if t_len == 0 {
            return ctx.finish(FittedStickyHmm {
                labels: empty_labels(0),
                start: init_start(k),
                trans: init_trans(k),
                mu: Vector::zeros(k),
                var: Vector::filled(k, 1.0),
                kappa,
                loglik: f64::NAN,
            });
        }
        let mean = x.column(0).mean();
        let sd = x.column(0).std().max(0.1);
        let mut mu = Vector::from_iter((0..k).map(|j| mean + (j as f64 - 0.5) * sd));
        let mut var = Vector::filled(k, sd * sd);
        let mut start = init_start(k);
        let mut trans = init_trans(k);
        let mut loglik = f64::NEG_INFINITY;
        let mut last_gamma: Vec<Vec<f64>> = Vec::new();
        for it in 0..self.max_iter.max(1) {
            let log_emit: Vec<Vec<f64>> = (0..t_len)
                .map(|t| {
                    let y = x.get(t, 0);
                    (0..k)
                        .map(|j| log_ar_gauss(y, None, mu[j], 0.0, var[j]))
                        .collect()
                })
                .collect();
            let Some(fb) = scaled_forward_backward(&mut ctx, &start, &trans, &log_emit) else {
                break;
            };
            loglik = fb.loglik;
            last_gamma = fb.gamma.clone();
            ctx.session.step(it as u64, -loglik, None);
            let mut ns = Vector::zeros(k);
            let mut nt = Matrix::zeros(k, k);
            for j in 0..k {
                ns[j] = fb.gamma[0][j];
            }
            for j in 0..k {
                let mut wsum = 0.0_f64;
                let mut wy = 0.0_f64;
                let mut wy2 = 0.0_f64;
                for t in 0..t_len {
                    let y = x.get(t, 0);
                    if !y.is_finite() {
                        continue;
                    }
                    let w = fb.gamma[t][j];
                    wsum += w;
                    wy += w * y;
                    wy2 += w * y * y;
                }
                if wsum > 1e-12 {
                    mu[j] = wy / wsum;
                    var[j] = (wy2 / wsum - mu[j] * mu[j]).max(COV_FLOOR);
                }
            }
            for t in 0..t_len.saturating_sub(1) {
                for i in 0..k {
                    for j in 0..k {
                        nt.set(i, j, nt.get(i, j) + fb.xi[t][i][j]);
                    }
                }
            }
            for j in 0..k {
                nt.set(j, j, nt.get(j, j) + kappa);
            }
            start = ns;
            trans = nt;
            renormalize_vec(&mut start, TRANS_FLOOR);
            renormalize_rows(&mut trans, TRANS_FLOOR);
        }
        let occup: Vec<f64> = (0..k)
            .map(|j| {
                last_gamma
                    .iter()
                    .map(|g| g.get(j).copied().unwrap_or(0.0))
                    .sum()
            })
            .collect();
        if !last_gamma.is_empty() {
            diagnose_chain(&mut ctx, &start, &trans, &occup);
        }
        let dummy = FittedStickyHmm {
            labels: Vector::zeros(0),
            start: start.clone(),
            trans: trans.clone(),
            mu: mu.clone(),
            var: var.clone(),
            kappa,
            loglik,
        };
        let (labels, _) = viterbi_path(&start, &trans, &dummy.log_emit_seq(x));
        ctx.finish(FittedStickyHmm {
            labels,
            start,
            trans,
            mu,
            var,
            kappa,
            loglik,
        })
    }
}

fn log_geom_duration(d: usize, rho: f64) -> f64 {
    if d == 0 {
        return f64::NEG_INFINITY;
    }
    let r = rho.clamp(1e-3, 0.99);
    (d as f64 - 1.0) * r.ln() + (1.0 - r).ln()
}

fn hsmm_viterbi(
    x: &Matrix,
    start: &Vector,
    trans: &Matrix,
    mu: &Vector,
    var: &Vector,
    rho: &Vector,
    dmax: usize,
) -> (Vector, f64) {
    let t_len = x.nrows();
    let k = mu.len();
    if t_len == 0 || k == 0 {
        return (empty_labels(t_len), f64::NAN);
    }
    let dcap = dmax.max(1);
    let mut best = vec![vec![f64::NEG_INFINITY; k]; t_len];
    let mut ptr_state = vec![vec![0usize; k]; t_len];
    let mut ptr_dur = vec![vec![1usize; k]; t_len];
    for t in 0..t_len {
        for j in 0..k {
            for d in 1..=dcap.min(t + 1) {
                let t0 = t + 1 - d;
                let mut emit = 0.0_f64;
                for u in t0..=t {
                    emit += log_ar_gauss(x.get(u, 0), None, mu[j], 0.0, var[j]);
                }
                let dur = log_geom_duration(d, rho[j]);
                let (prev, pscore) = if t0 == 0 {
                    (j, start[j].max(TRANS_FLOOR).ln())
                } else {
                    let mut bp = 0usize;
                    let mut bs = f64::NEG_INFINITY;
                    for i in 0..k {
                        if i == j {
                            continue;
                        }
                        let s = best[t0 - 1][i] + trans.get(i, j).max(TRANS_FLOOR).ln();
                        if s > bs {
                            bs = s;
                            bp = i;
                        }
                    }
                    (bp, bs)
                };
                let sc = pscore + dur + emit;
                if sc > best[t][j] {
                    best[t][j] = sc;
                    ptr_state[t][j] = prev;
                    ptr_dur[t][j] = d;
                }
            }
        }
    }
    let mut end_state = 0usize;
    let mut best_end = f64::NEG_INFINITY;
    for j in 0..k {
        if best[t_len - 1][j] > best_end {
            best_end = best[t_len - 1][j];
            end_state = j;
        }
    }
    let mut labels = Vector::zeros(t_len);
    let mut t = t_len - 1;
    let mut j = end_state;
    loop {
        let d = ptr_dur[t][j].max(1);
        let t0 = t + 1 - d;
        for u in t0..=t {
            labels[u] = j as f64;
        }
        if t0 == 0 {
            break;
        }
        let prev = ptr_state[t][j];
        t = t0 - 1;
        j = prev;
    }
    (labels, best_end)
}

/// Explicit-duration hidden semi-Markov model (segmental Viterbi).
///
/// Duration cap is not identification `p`. Self-transitions are forbidden;
/// sojourns follow a truncated geometric. Distinct from [`StickyHmm`]
/// (diagonal bias inside an ordinary HMM).
#[derive(Clone, Debug)]
pub struct Hsmm {
    /// Hidden states. Not identification `p`.
    pub n_states: usize,
    /// Maximum sojourn. Not identification `p`.
    pub max_duration: usize,
    /// Segmental iterations.
    pub max_iter: usize,
}

impl Default for Hsmm {
    fn default() -> Self {
        Self {
            n_states: 2,
            max_duration: 8,
            max_iter: 12,
        }
    }
}

impl Hsmm {
    /// `k`-state explicit-duration HSMM.
    pub fn new(n_states: usize) -> Self {
        Self {
            n_states,
            ..Self::default()
        }
    }

    /// Fit alias.
    pub fn fit(&mut self, x: &Matrix, session: &Session) -> Result<Qualified<FittedHsmm>> {
        self.fit_unsupervised(x, session)
    }
}

/// Fitted explicit-duration HSMM.
#[derive(Clone, Debug)]
pub struct FittedHsmm {
    /// Segmental Viterbi path.
    pub labels: Vector,
    /// Start distribution.
    pub start: Vector,
    /// Off-diagonal next-state kernel.
    pub trans: Matrix,
    /// Emission mean.
    pub mu: Vector,
    /// Emission variance.
    pub var: Vector,
    /// Geometric stay parameter \(\rho_j\).
    pub rho: Vector,
    /// Path score.
    pub loglik: f64,
}

impl FittedHsmm {
    /// Segmental Viterbi path.
    pub fn decode(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let ctx = FitCtx::with_session(session.child("decode"));
        let (path, _) = hsmm_viterbi(
            x,
            &self.start,
            &self.trans,
            &self.mu,
            &self.var,
            &self.rho,
            8,
        );
        ctx.finish(path)
    }
}

impl Predict for FittedHsmm {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        self.decode(x, session)
    }
}

impl FitUnsupervised for Hsmm {
    type Fitted = FittedHsmm;
    fn fit_unsupervised(&mut self, x: &Matrix, session: &Session) -> Result<Qualified<FittedHsmm>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let t_len = x.nrows();
        let k = self.n_states.max(1);
        let dmax = self.max_duration.max(1);
        if t_len == 0 {
            return ctx.finish(FittedHsmm {
                labels: empty_labels(0),
                start: init_start(k),
                trans: init_trans(k),
                mu: Vector::zeros(k),
                var: Vector::filled(k, 1.0),
                rho: Vector::filled(k, 0.7),
                loglik: f64::NAN,
            });
        }
        let mean = x.column(0).mean();
        let sd = x.column(0).std().max(0.1);
        let mut mu = Vector::from_iter((0..k).map(|j| mean + (j as f64 - 0.5) * sd));
        let mut var = Vector::filled(k, sd * sd);
        let mut rho = Vector::filled(k, 0.75);
        let mut start = init_start(k);
        let mut trans = init_trans(k);
        for i in 0..k {
            trans.set(i, i, 0.0);
        }
        renormalize_rows(&mut trans, TRANS_FLOOR);
        for i in 0..k {
            trans.set(i, i, 0.0);
        }
        let mut labels = empty_labels(t_len);
        let mut loglik = f64::NEG_INFINITY;
        for it in 0..self.max_iter.max(1) {
            let (path, score) = hsmm_viterbi(x, &start, &trans, &mu, &var, &rho, dmax);
            labels = path;
            loglik = score;
            ctx.session.step(it as u64, -loglik, None);
            let mut nrun = vec![0.0_f64; k];
            let mut slen = vec![0.0_f64; k];
            let mut wy = vec![0.0_f64; k];
            let mut wy2 = vec![0.0_f64; k];
            let mut wsum = vec![0.0_f64; k];
            let mut nt = Matrix::zeros(k, k);
            let mut t0 = 0usize;
            while t0 < t_len {
                let j = labels[t0] as usize;
                let j = j.min(k.saturating_sub(1));
                let mut t1 = t0 + 1;
                while t1 < t_len && (labels[t1] as usize) == j {
                    t1 += 1;
                }
                let d = (t1 - t0) as f64;
                nrun[j] += 1.0;
                slen[j] += d;
                for t in t0..t1 {
                    let y = x.get(t, 0);
                    if y.is_finite() {
                        wsum[j] += 1.0;
                        wy[j] += y;
                        wy2[j] += y * y;
                    }
                }
                if t1 < t_len {
                    let nxt = (labels[t1] as usize).min(k.saturating_sub(1));
                    if nxt != j {
                        nt.set(j, nxt, nt.get(j, nxt) + 1.0);
                    }
                }
                t0 = t1;
            }
            for j in 0..k {
                if wsum[j] > 0.0 {
                    mu[j] = wy[j] / wsum[j];
                    var[j] = (wy2[j] / wsum[j] - mu[j] * mu[j]).max(COV_FLOOR);
                }
                if nrun[j] > 0.0 {
                    let md = (slen[j] / nrun[j]).max(1.0);
                    rho[j] = (1.0 - 1.0 / md).clamp(1e-3, 0.99);
                }
            }
            trans = nt;
            for i in 0..k {
                trans.set(i, i, 0.0);
            }
            renormalize_rows(&mut trans, TRANS_FLOOR);
            for i in 0..k {
                trans.set(i, i, 0.0);
            }
            if t_len > 0 {
                let s0 = (labels[0] as usize).min(k.saturating_sub(1));
                start = Vector::zeros(k);
                start[s0] = 1.0;
            }
        }
        ctx.push(
            Issue::builder(IssueCode::CausalClaimUnidentified)
                .severity(signlred::Severity::Advisory)
                .message("Hsmm is segmental Viterbi, not a published explicit-duration HSMM")
                .compromise(NumericalCompromise::new(
                    "explicit-duration HSMM",
                    "truncated-geometric sojourns and Gaussian emissions, Viterbi trained",
                    "forward–backward over durations and a nonparametric duration PMF are omitted",
                    "read the path as a segmental sketch",
                ))
                .build(),
        );
        ctx.finish(FittedHsmm {
            labels,
            start,
            trans,
            mu,
            var,
            rho,
            loglik,
        })
    }
}

fn log_negbin(y: f64, r: f64, p: f64) -> f64 {
    if !y.is_finite() || y < 0.0 || r <= 0.0 || p <= 0.0 || p >= 1.0 {
        return f64::NEG_INFINITY;
    }
    crate::special::ln_gamma(y + r) - crate::special::ln_gamma(r) - crate::special::ln_gamma(y + 1.0)
        + r * p.ln()
        + y * (1.0 - p).ln()
}

fn log_zip(y: f64, pi: f64, lam: f64) -> f64 {
    if !y.is_finite() || y < 0.0 || lam <= 0.0 {
        return f64::NEG_INFINITY;
    }
    let pi = pi.clamp(1e-6, 1.0 - 1e-6);
    if y.abs() < 1e-12 {
        let a = pi.ln();
        let b = (1.0 - pi).ln() - lam;
        logsumexp(&[a, b])
    } else {
        (1.0 - pi).ln() + y * lam.ln() - lam - crate::special::ln_gamma(y + 1.0)
    }
}

fn log_dirichlet_row(x: &Matrix, t: usize, alpha: &Vector) -> f64 {
    let d = x.ncols().min(alpha.len());
    let mut asum = 0.0_f64;
    let mut s = 0.0_f64;
    for j in 0..d {
        let a = alpha[j];
        let y = x.get(t, j);
        if !y.is_finite() || y <= 0.0 || a <= 0.0 {
            return f64::NEG_INFINITY;
        }
        asum += a;
        s += (a - 1.0) * y.ln() - crate::special::ln_gamma(a);
    }
    crate::special::ln_gamma(asum) + s
}

/// Negative-binomial emission HMM (overdispersed counts).
///
/// State count is not identification `p`. Distinct from [`PoissonHmm`]
/// (equidispersed) and [`GammaHmm`] (continuous).
#[derive(Clone, Debug)]
pub struct NegativeBinomialHmm {
    /// Hidden states. Not identification `p`.
    pub n_states: usize,
    /// Baum–Welch cap.
    pub max_iter: usize,
}

impl Default for NegativeBinomialHmm {
    fn default() -> Self {
        Self {
            n_states: 2,
            max_iter: 40,
        }
    }
}

impl NegativeBinomialHmm {
    /// `k`-state NB HMM.
    pub fn new(n_states: usize) -> Self {
        Self {
            n_states,
            ..Self::default()
        }
    }

    /// Fit alias.
    pub fn fit(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedNegativeBinomialHmm>> {
        self.fit_unsupervised(x, session)
    }
}

/// Fitted negative-binomial HMM.
#[derive(Clone, Debug)]
pub struct FittedNegativeBinomialHmm {
    /// Viterbi path.
    pub labels: Vector,
    /// Start distribution.
    pub start: Vector,
    /// Transitions.
    pub trans: Matrix,
    /// Success counts \(r_j\).
    pub r: Vector,
    /// Success probability \(p_j\).
    pub p: Vector,
    /// Training log-likelihood.
    pub loglik: f64,
}

impl FittedNegativeBinomialHmm {
    fn log_emit_seq(&self, x: &Matrix) -> Vec<Vec<f64>> {
        let t = x.nrows();
        let s = self.r.len();
        let mut out = vec![vec![f64::NEG_INFINITY; s]; t];
        for ti in 0..t {
            let y = x.get(ti, 0);
            for j in 0..s {
                out[ti][j] = log_negbin(y, self.r[j], self.p[j]);
            }
        }
        out
    }

    /// Viterbi path.
    pub fn decode(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let ctx = FitCtx::with_session(session.child("decode"));
        let (path, _) = viterbi_path(&self.start, &self.trans, &self.log_emit_seq(x));
        ctx.finish(path)
    }
}

impl Predict for FittedNegativeBinomialHmm {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        self.decode(x, session)
    }
}

impl FitUnsupervised for NegativeBinomialHmm {
    type Fitted = FittedNegativeBinomialHmm;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedNegativeBinomialHmm>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let t_len = x.nrows();
        let k = self.n_states.max(1);
        if t_len == 0 {
            return ctx.finish(FittedNegativeBinomialHmm {
                labels: empty_labels(0),
                start: init_start(k),
                trans: init_trans(k),
                r: Vector::filled(k, 1.0),
                p: Vector::filled(k, 0.5),
                loglik: f64::NAN,
            });
        }
        let mean = x.column(0).as_slice().iter().filter(|v| **v >= 0.0).sum::<f64>()
            / t_len.max(1) as f64;
        let mut r = Vector::from_iter((0..k).map(|j| (1.0 + j as f64).max(0.5)));
        let mut p = Vector::from_iter((0..k).map(|j| {
            let rj = r[j];
            (rj / (rj + mean.max(0.1) * (0.6 + 0.4 * j as f64))).clamp(0.05, 0.95)
        }));
        let mut start = init_start(k);
        let mut trans = init_trans(k);
        let mut loglik = f64::NEG_INFINITY;
        let mut last_gamma: Vec<Vec<f64>> = Vec::new();
        for it in 0..self.max_iter.max(1) {
            let log_emit: Vec<Vec<f64>> = (0..t_len)
                .map(|t| {
                    let y = x.get(t, 0);
                    (0..k).map(|j| log_negbin(y, r[j], p[j])).collect()
                })
                .collect();
            let Some(fb) = scaled_forward_backward(&mut ctx, &start, &trans, &log_emit) else {
                break;
            };
            loglik = fb.loglik;
            last_gamma = fb.gamma.clone();
            ctx.session.step(it as u64, -loglik, None);
            let mut ns = Vector::zeros(k);
            let mut nt = Matrix::zeros(k, k);
            for j in 0..k {
                ns[j] = fb.gamma[0][j];
            }
            for j in 0..k {
                let mut wsum = 0.0_f64;
                let mut wy = 0.0_f64;
                let mut wy2 = 0.0_f64;
                for t in 0..t_len {
                    let y = x.get(t, 0);
                    if y < 0.0 || !y.is_finite() {
                        continue;
                    }
                    let w = fb.gamma[t][j];
                    wsum += w;
                    wy += w * y;
                    wy2 += w * y * y;
                }
                if wsum > 1e-12 {
                    let mu = wy / wsum;
                    let var = (wy2 / wsum - mu * mu).max(mu + 1e-4);
                    r[j] = (mu * mu / (var - mu).max(1e-4)).clamp(0.05, 80.0);
                    p[j] = (r[j] / (r[j] + mu.max(1e-6))).clamp(0.02, 0.98);
                }
            }
            for t in 0..t_len.saturating_sub(1) {
                for i in 0..k {
                    for j in 0..k {
                        nt.set(i, j, nt.get(i, j) + fb.xi[t][i][j]);
                    }
                }
            }
            start = ns;
            trans = nt;
            renormalize_vec(&mut start, TRANS_FLOOR);
            renormalize_rows(&mut trans, TRANS_FLOOR);
        }
        let occup: Vec<f64> = (0..k)
            .map(|j| last_gamma.iter().map(|g| g.get(j).copied().unwrap_or(0.0)).sum())
            .collect();
        if !last_gamma.is_empty() {
            diagnose_chain(&mut ctx, &start, &trans, &occup);
        }
        let dummy = FittedNegativeBinomialHmm {
            labels: Vector::zeros(0),
            start: start.clone(),
            trans: trans.clone(),
            r: r.clone(),
            p: p.clone(),
            loglik,
        };
        let (labels, _) = viterbi_path(&start, &trans, &dummy.log_emit_seq(x));
        ctx.finish(FittedNegativeBinomialHmm {
            labels,
            start,
            trans,
            r,
            p,
            loglik,
        })
    }
}

/// Zero-inflated Poisson HMM.
///
/// Zero-mass \(\pi_j\) is not identification `p`. Distinct from [`PoissonHmm`]
/// (no extra zeros) and [`NegativeBinomialHmm`] (overdispersion without a point mass).
#[derive(Clone, Debug)]
pub struct ZeroInflatedPoissonHmm {
    /// Hidden states. Not identification `p`.
    pub n_states: usize,
    /// Baum–Welch cap.
    pub max_iter: usize,
}

impl Default for ZeroInflatedPoissonHmm {
    fn default() -> Self {
        Self {
            n_states: 2,
            max_iter: 40,
        }
    }
}

impl ZeroInflatedPoissonHmm {
    /// `k`-state ZIP HMM.
    pub fn new(n_states: usize) -> Self {
        Self {
            n_states,
            ..Self::default()
        }
    }

    /// Fit alias.
    pub fn fit(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedZeroInflatedPoissonHmm>> {
        self.fit_unsupervised(x, session)
    }
}

/// Fitted ZIP HMM.
#[derive(Clone, Debug)]
pub struct FittedZeroInflatedPoissonHmm {
    /// Viterbi path.
    pub labels: Vector,
    /// Start distribution.
    pub start: Vector,
    /// Transitions.
    pub trans: Matrix,
    /// Extra-zero mass.
    pub pi: Vector,
    /// Poisson rate on the non-zero component.
    pub lam: Vector,
    /// Training log-likelihood.
    pub loglik: f64,
}

impl FittedZeroInflatedPoissonHmm {
    fn log_emit_seq(&self, x: &Matrix) -> Vec<Vec<f64>> {
        let t = x.nrows();
        let s = self.lam.len();
        let mut out = vec![vec![f64::NEG_INFINITY; s]; t];
        for ti in 0..t {
            let y = x.get(ti, 0);
            for j in 0..s {
                out[ti][j] = log_zip(y, self.pi[j], self.lam[j]);
            }
        }
        out
    }

    /// Viterbi path.
    pub fn decode(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let ctx = FitCtx::with_session(session.child("decode"));
        let (path, _) = viterbi_path(&self.start, &self.trans, &self.log_emit_seq(x));
        ctx.finish(path)
    }
}

impl Predict for FittedZeroInflatedPoissonHmm {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        self.decode(x, session)
    }
}

impl FitUnsupervised for ZeroInflatedPoissonHmm {
    type Fitted = FittedZeroInflatedPoissonHmm;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedZeroInflatedPoissonHmm>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let t_len = x.nrows();
        let k = self.n_states.max(1);
        if t_len == 0 {
            return ctx.finish(FittedZeroInflatedPoissonHmm {
                labels: empty_labels(0),
                start: init_start(k),
                trans: init_trans(k),
                pi: Vector::filled(k, 0.1),
                lam: Vector::filled(k, 1.0),
                loglik: f64::NAN,
            });
        }
        let mean = x.column(0).as_slice().iter().filter(|v| **v >= 0.0).sum::<f64>()
            / t_len.max(1) as f64;
        let mut pi = Vector::filled(k, 0.15);
        let mut lam = Vector::from_iter((0..k).map(|j| (mean * (0.5 + j as f64)).max(0.2)));
        let mut start = init_start(k);
        let mut trans = init_trans(k);
        let mut loglik = f64::NEG_INFINITY;
        let mut last_gamma: Vec<Vec<f64>> = Vec::new();
        for it in 0..self.max_iter.max(1) {
            let log_emit: Vec<Vec<f64>> = (0..t_len)
                .map(|t| {
                    let y = x.get(t, 0);
                    (0..k).map(|j| log_zip(y, pi[j], lam[j])).collect()
                })
                .collect();
            let Some(fb) = scaled_forward_backward(&mut ctx, &start, &trans, &log_emit) else {
                break;
            };
            loglik = fb.loglik;
            last_gamma = fb.gamma.clone();
            ctx.session.step(it as u64, -loglik, None);
            let mut ns = Vector::zeros(k);
            let mut nt = Matrix::zeros(k, k);
            for j in 0..k {
                ns[j] = fb.gamma[0][j];
            }
            for j in 0..k {
                let mut wsum = 0.0_f64;
                let mut wzero = 0.0_f64;
                let mut wy = 0.0_f64;
                for t in 0..t_len {
                    let y = x.get(t, 0);
                    if y < 0.0 || !y.is_finite() {
                        continue;
                    }
                    let w = fb.gamma[t][j];
                    wsum += w;
                    wy += w * y;
                    if y.abs() < 1e-12 {
                        wzero += w;
                    }
                }
                if wsum > 1e-12 {
                    let mu = wy / wsum;
                    let zf = (wzero / wsum).clamp(0.0, 0.95);
                    let pois0 = (-lam[j]).exp();
                    pi[j] = ((zf - pois0) / (1.0 - pois0).max(1e-6)).clamp(1e-4, 0.9);
                    lam[j] = (mu / (1.0 - pi[j]).max(1e-3)).max(0.05);
                }
            }
            for t in 0..t_len.saturating_sub(1) {
                for i in 0..k {
                    for j in 0..k {
                        nt.set(i, j, nt.get(i, j) + fb.xi[t][i][j]);
                    }
                }
            }
            start = ns;
            trans = nt;
            renormalize_vec(&mut start, TRANS_FLOOR);
            renormalize_rows(&mut trans, TRANS_FLOOR);
        }
        let occup: Vec<f64> = (0..k)
            .map(|j| last_gamma.iter().map(|g| g.get(j).copied().unwrap_or(0.0)).sum())
            .collect();
        if !last_gamma.is_empty() {
            diagnose_chain(&mut ctx, &start, &trans, &occup);
        }
        let dummy = FittedZeroInflatedPoissonHmm {
            labels: Vector::zeros(0),
            start: start.clone(),
            trans: trans.clone(),
            pi: pi.clone(),
            lam: lam.clone(),
            loglik,
        };
        let (labels, _) = viterbi_path(&start, &trans, &dummy.log_emit_seq(x));
        ctx.finish(FittedZeroInflatedPoissonHmm {
            labels,
            start,
            trans,
            pi,
            lam,
            loglik,
        })
    }
}

/// Dirichlet-emission HMM (simplex / compositional rows).
///
/// Concentration width is not identification `p`. Distinct from
/// [`MultinomialHmm`] (integer codes) and [`CircularHmm`] (von Mises).
#[derive(Clone, Debug)]
pub struct DirichletHmm {
    /// Hidden states. Not identification `p`.
    pub n_states: usize,
    /// Baum–Welch cap.
    pub max_iter: usize,
}

impl Default for DirichletHmm {
    fn default() -> Self {
        Self {
            n_states: 2,
            max_iter: 40,
        }
    }
}

impl DirichletHmm {
    /// `k`-state Dirichlet HMM.
    pub fn new(n_states: usize) -> Self {
        Self {
            n_states,
            ..Self::default()
        }
    }

    /// Fit alias.
    pub fn fit(&mut self, x: &Matrix, session: &Session) -> Result<Qualified<FittedDirichletHmm>> {
        self.fit_unsupervised(x, session)
    }
}

/// Fitted Dirichlet HMM.
#[derive(Clone, Debug)]
pub struct FittedDirichletHmm {
    /// Viterbi path.
    pub labels: Vector,
    /// Start distribution.
    pub start: Vector,
    /// Transitions.
    pub trans: Matrix,
    /// Concentrations \(\alpha_{j\cdot}\).
    pub alpha: Matrix,
    /// Training log-likelihood.
    pub loglik: f64,
}

impl FittedDirichletHmm {
    fn log_emit_seq(&self, x: &Matrix) -> Vec<Vec<f64>> {
        let t = x.nrows();
        let s = self.alpha.nrows();
        let mut out = vec![vec![f64::NEG_INFINITY; s]; t];
        for ti in 0..t {
            for j in 0..s {
                let aj = Vector::from_iter((0..self.alpha.ncols()).map(|c| self.alpha.get(j, c)));
                out[ti][j] = log_dirichlet_row(x, ti, &aj);
            }
        }
        out
    }

    /// Viterbi path.
    pub fn decode(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let ctx = FitCtx::with_session(session.child("decode"));
        let (path, _) = viterbi_path(&self.start, &self.trans, &self.log_emit_seq(x));
        ctx.finish(path)
    }
}

impl Predict for FittedDirichletHmm {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        self.decode(x, session)
    }
}

impl FitUnsupervised for DirichletHmm {
    type Fitted = FittedDirichletHmm;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedDirichletHmm>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let t_len = x.nrows();
        let d = x.ncols().max(1);
        let k = self.n_states.max(1);
        if t_len == 0 {
            return ctx.finish(FittedDirichletHmm {
                labels: empty_labels(0),
                start: init_start(k),
                trans: init_trans(k),
                alpha: Matrix::from_fn(k, d, |_, _| 1.0),
                loglik: f64::NAN,
            });
        }
        let mut n_ok = 0usize;
        for i in 0..t_len {
            let mut s = 0.0_f64;
            let mut pos = true;
            for j in 0..d {
                let v = x.get(i, j);
                if !v.is_finite() || v <= 0.0 {
                    pos = false;
                }
                s += v;
            }
            if pos && s > 1e-12 {
                n_ok += 1;
            }
        }
        if n_ok < t_len {
            ctx.push(
                Issue::builder(IssueCode::NonPositiveSeries)
                    .severity(signlred::Severity::Warning)
                    .message("DirichletHmm skipped non-simplex rows")
                    .build(),
            );
        }
        if n_ok < 2 {
            ctx.push(
                Issue::builder(IssueCode::MeaninglessFit)
                    .message("DirichletHmm needs at least two positive simplex rows")
                    .meaninglessness(Meaninglessness::vacuous(
                        "Dirichlet HMM",
                        "concentrations are unidentified without positive compositions",
                        "collect simplex-valued rows",
                    ))
                    .build(),
            );
            return ctx.finish(FittedDirichletHmm {
                labels: empty_labels(t_len),
                start: init_start(k),
                trans: init_trans(k),
                alpha: Matrix::from_fn(k, d, |_, _| 1.0),
                loglik: f64::NAN,
            });
        }
        let mut alpha = Matrix::from_fn(k, d, |j, c| 0.8 + 0.4 * j as f64 + 0.2 * c as f64);
        let mut start = init_start(k);
        let mut trans = init_trans(k);
        let mut loglik = f64::NEG_INFINITY;
        let mut last_gamma: Vec<Vec<f64>> = Vec::new();
        for it in 0..self.max_iter.max(1) {
            let dummy = FittedDirichletHmm {
                labels: Vector::zeros(0),
                start: start.clone(),
                trans: trans.clone(),
                alpha: alpha.clone(),
                loglik,
            };
            let log_emit = dummy.log_emit_seq(x);
            let Some(fb) = scaled_forward_backward(&mut ctx, &start, &trans, &log_emit) else {
                break;
            };
            loglik = fb.loglik;
            last_gamma = fb.gamma.clone();
            ctx.session.step(it as u64, -loglik, None);
            let mut ns = Vector::zeros(k);
            let mut nt = Matrix::zeros(k, k);
            for j in 0..k {
                ns[j] = fb.gamma[0][j];
            }
            for j in 0..k {
                let mut wsum = 0.0_f64;
                let mut mean = vec![0.0_f64; d];
                let mut var = vec![0.0_f64; d];
                for t in 0..t_len {
                    let mut ok = true;
                    let mut row = vec![0.0_f64; d];
                    let mut rs = 0.0_f64;
                    for c in 0..d {
                        let v = x.get(t, c);
                        if !v.is_finite() || v <= 0.0 {
                            ok = false;
                        }
                        row[c] = v;
                        rs += v;
                    }
                    if !ok || rs <= 1e-12 {
                        continue;
                    }
                    let w = fb.gamma[t][j];
                    wsum += w;
                    for c in 0..d {
                        let z = row[c] / rs;
                        mean[c] += w * z;
                        var[c] += w * z * z;
                    }
                }
                if wsum > 1e-12 {
                    for c in 0..d {
                        mean[c] /= wsum;
                        var[c] = (var[c] / wsum - mean[c] * mean[c]).max(1e-8);
                    }
                    let mut prec = 0.0_f64;
                    let mut pc = 0.0_f64;
                    for c in 0..d {
                        if mean[c] > 1e-8 && mean[c] < 1.0 - 1e-8 {
                            prec += mean[c] * (1.0 - mean[c]) / var[c] - 1.0;
                            pc += 1.0;
                        }
                    }
                    let prec = if pc > 0.0 {
                        (prec / pc).clamp(0.5, 80.0)
                    } else {
                        2.0
                    };
                    for c in 0..d {
                        alpha.set(j, c, (mean[c] * prec).max(0.05));
                    }
                }
            }
            for t in 0..t_len.saturating_sub(1) {
                for i in 0..k {
                    for j in 0..k {
                        nt.set(i, j, nt.get(i, j) + fb.xi[t][i][j]);
                    }
                }
            }
            start = ns;
            trans = nt;
            renormalize_vec(&mut start, TRANS_FLOOR);
            renormalize_rows(&mut trans, TRANS_FLOOR);
        }
        let occup: Vec<f64> = (0..k)
            .map(|j| last_gamma.iter().map(|g| g.get(j).copied().unwrap_or(0.0)).sum())
            .collect();
        if !last_gamma.is_empty() {
            diagnose_chain(&mut ctx, &start, &trans, &occup);
        }
        let dummy = FittedDirichletHmm {
            labels: Vector::zeros(0),
            start: start.clone(),
            trans: trans.clone(),
            alpha: alpha.clone(),
            loglik,
        };
        let (labels, _) = viterbi_path(&start, &trans, &dummy.log_emit_seq(x));
        ctx.finish(FittedDirichletHmm {
            labels,
            start,
            trans,
            alpha,
            loglik,
        })
    }
}

fn gem_start(k: usize, alpha: f64) -> Vector {
    let a = if alpha.is_finite() && alpha > 0.0 {
        alpha
    } else {
        1.0
    };
    if k == 0 {
        return Vector::zeros(0);
    }
    let v = 1.0 / (1.0 + a);
    let stay = a / (1.0 + a);
    let mut s = Vector::zeros(k);
    let mut rem = 1.0_f64;
    for j in 0..k.saturating_sub(1) {
        s[j] = rem * v;
        rem *= stay;
    }
    s[k - 1] = rem;
    s
}

fn log_gauss1(y: f64, mu: f64, var: f64) -> f64 {
    if !y.is_finite() || !mu.is_finite() || var <= 0.0 {
        return f64::NEG_INFINITY;
    }
    -0.5 * (LN_2PI + var.ln() + (y - mu) * (y - mu) / var)
}

fn log_beta_emit(y: f64, alpha: f64, beta: f64) -> f64 {
    if !y.is_finite() || y <= 0.0 || y >= 1.0 || alpha <= 0.0 || beta <= 0.0 {
        return f64::NEG_INFINITY;
    }
    (alpha - 1.0) * y.ln() + (beta - 1.0) * (1.0 - y).ln()
        - crate::special::ln_gamma(alpha)
        - crate::special::ln_gamma(beta)
        + crate::special::ln_gamma(alpha + beta)
}

fn log_lognormal(y: f64, mu: f64, var: f64) -> f64 {
    if !y.is_finite() || y <= 0.0 || var <= 0.0 {
        return f64::NEG_INFINITY;
    }
    let z = y.ln();
    -z + log_gauss1(z, mu, var)
}

const EULER_GAMMA: f64 = 0.5772156649015329;

fn log_weibull(y: f64, shape: f64, scale: f64) -> f64 {
    if !y.is_finite() || y <= 0.0 || shape <= 0.0 || scale <= 0.0 {
        return f64::NEG_INFINITY;
    }
    shape.ln() - shape * scale.ln() + (shape - 1.0) * y.ln() - (y / scale).powf(shape)
}

/// Beta-emission HMM on \((0,1)\) (compositional / rate series).
///
/// Shape counts are not identification `p`. Distinct from [`DirichletHmm`]
/// (simplex rows) and [`GammaHmm`] (positive unbounded support).
#[derive(Clone, Debug)]
pub struct BetaHmm {
    /// Hidden states. Not identification `p`.
    pub n_states: usize,
    /// Baum–Welch cap.
    pub max_iter: usize,
}

impl Default for BetaHmm {
    fn default() -> Self {
        Self {
            n_states: 2,
            max_iter: 40,
        }
    }
}

impl BetaHmm {
    /// `k`-state Beta HMM.
    pub fn new(n_states: usize) -> Self {
        Self {
            n_states,
            ..Self::default()
        }
    }

    /// Fit alias.
    pub fn fit(&mut self, x: &Matrix, session: &Session) -> Result<Qualified<FittedBetaHmm>> {
        self.fit_unsupervised(x, session)
    }
}

/// Fitted Beta HMM.
#[derive(Clone, Debug)]
pub struct FittedBetaHmm {
    /// Viterbi path.
    pub labels: Vector,
    /// Start distribution.
    pub start: Vector,
    /// Transitions.
    pub trans: Matrix,
    /// Shape \(\alpha_j\).
    pub alpha: Vector,
    /// Shape \(\beta_j\).
    pub beta: Vector,
    /// Training log-likelihood.
    pub loglik: f64,
}

impl FittedBetaHmm {
    fn log_emit_seq(&self, x: &Matrix) -> Vec<Vec<f64>> {
        let t = x.nrows();
        let s = self.alpha.len();
        let mut out = vec![vec![f64::NEG_INFINITY; s]; t];
        for ti in 0..t {
            let y = x.get(ti, 0);
            for j in 0..s {
                out[ti][j] = log_beta_emit(y, self.alpha[j], self.beta[j]);
            }
        }
        out
    }

    /// Viterbi path.
    pub fn decode(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let ctx = FitCtx::with_session(session.child("decode"));
        let (path, _) = viterbi_path(&self.start, &self.trans, &self.log_emit_seq(x));
        ctx.finish(path)
    }
}

impl Predict for FittedBetaHmm {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        self.decode(x, session)
    }
}

impl FitUnsupervised for BetaHmm {
    type Fitted = FittedBetaHmm;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedBetaHmm>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let t_len = x.nrows();
        let k = self.n_states.max(1);
        if t_len == 0 {
            return ctx.finish(FittedBetaHmm {
                labels: empty_labels(0),
                start: init_start(k),
                trans: init_trans(k),
                alpha: Vector::filled(k, 2.0),
                beta: Vector::filled(k, 2.0),
                loglik: f64::NAN,
            });
        }
        let mut n_ok = 0usize;
        for i in 0..t_len {
            let y = x.get(i, 0);
            if y.is_finite() && y > 0.0 && y < 1.0 {
                n_ok += 1;
            }
        }
        if n_ok < t_len {
            ctx.push(
                Issue::builder(IssueCode::NonPositiveSeries)
                    .severity(signlred::Severity::Warning)
                    .message("BetaHmm skipped observations outside (0, 1)")
                    .build(),
            );
        }
        if n_ok < 2 {
            ctx.push(
                Issue::builder(IssueCode::MeaninglessFit)
                    .message("BetaHmm needs at least two observations in (0, 1)")
                    .meaninglessness(Meaninglessness::vacuous(
                        "Beta HMM",
                        "α, β are unidentified without unit-interval data",
                        "collect rates or proportions in (0, 1)",
                    ))
                    .build(),
            );
            return ctx.finish(FittedBetaHmm {
                labels: empty_labels(t_len),
                start: init_start(k),
                trans: init_trans(k),
                alpha: Vector::filled(k, 2.0),
                beta: Vector::filled(k, 2.0),
                loglik: f64::NAN,
            });
        }
        let mut alpha = Vector::from_iter((0..k).map(|j| 1.2 + 0.6 * j as f64));
        let mut beta = Vector::from_iter((0..k).map(|j| 2.0 - 0.4 * j as f64));
        let mut start = init_start(k);
        let mut trans = init_trans(k);
        let mut loglik = f64::NEG_INFINITY;
        let mut last_gamma: Vec<Vec<f64>> = Vec::new();
        for it in 0..self.max_iter.max(1) {
            let log_emit: Vec<Vec<f64>> = (0..t_len)
                .map(|t| {
                    let y = x.get(t, 0);
                    (0..k)
                        .map(|j| log_beta_emit(y, alpha[j], beta[j]))
                        .collect()
                })
                .collect();
            let Some(fb) = scaled_forward_backward(&mut ctx, &start, &trans, &log_emit) else {
                break;
            };
            loglik = fb.loglik;
            last_gamma = fb.gamma.clone();
            ctx.session.step(it as u64, -loglik, None);
            let mut ns = Vector::zeros(k);
            let mut nt = Matrix::zeros(k, k);
            for j in 0..k {
                ns[j] = fb.gamma[0][j];
            }
            for j in 0..k {
                let mut wsum = 0.0_f64;
                let mut wy = 0.0_f64;
                let mut wy2 = 0.0_f64;
                for t in 0..t_len {
                    let y = x.get(t, 0);
                    if !y.is_finite() || y <= 0.0 || y >= 1.0 {
                        continue;
                    }
                    let w = fb.gamma[t][j];
                    wsum += w;
                    wy += w * y;
                    wy2 += w * y * y;
                }
                if wsum > 1e-12 {
                    let m = (wy / wsum).clamp(1e-3, 1.0 - 1e-3);
                    let var = (wy2 / wsum - m * m).max(1e-8);
                    let prec = (m * (1.0 - m) / var - 1.0).clamp(0.5, 80.0);
                    alpha[j] = (m * prec).max(0.05);
                    beta[j] = ((1.0 - m) * prec).max(0.05);
                }
            }
            for t in 0..t_len.saturating_sub(1) {
                for i in 0..k {
                    for j in 0..k {
                        nt.set(i, j, nt.get(i, j) + fb.xi[t][i][j]);
                    }
                }
            }
            start = ns;
            trans = nt;
            renormalize_vec(&mut start, TRANS_FLOOR);
            renormalize_rows(&mut trans, TRANS_FLOOR);
        }
        let occup: Vec<f64> = (0..k)
            .map(|j| last_gamma.iter().map(|g| g.get(j).copied().unwrap_or(0.0)).sum())
            .collect();
        if !last_gamma.is_empty() {
            diagnose_chain(&mut ctx, &start, &trans, &occup);
        }
        let dummy = FittedBetaHmm {
            labels: Vector::zeros(0),
            start: start.clone(),
            trans: trans.clone(),
            alpha: alpha.clone(),
            beta: beta.clone(),
            loglik,
        };
        let (labels, _) = viterbi_path(&start, &trans, &dummy.log_emit_seq(x));
        ctx.finish(FittedBetaHmm {
            labels,
            start,
            trans,
            alpha,
            beta,
            loglik,
        })
    }
}

/// Log-normal emission HMM (\(\ln y\sim\mathcal N\)).
///
/// State count is not identification `p`. Distinct from [`GaussianHmm`]
/// (untransformed) and [`GammaHmm`] (shape/rate, not log-mean).
#[derive(Clone, Debug)]
pub struct LogNormalHmm {
    /// Hidden states. Not identification `p`.
    pub n_states: usize,
    /// Baum–Welch cap.
    pub max_iter: usize,
}

impl Default for LogNormalHmm {
    fn default() -> Self {
        Self {
            n_states: 2,
            max_iter: 40,
        }
    }
}

impl LogNormalHmm {
    /// `k`-state log-normal HMM.
    pub fn new(n_states: usize) -> Self {
        Self {
            n_states,
            ..Self::default()
        }
    }

    /// Fit alias.
    pub fn fit(&mut self, x: &Matrix, session: &Session) -> Result<Qualified<FittedLogNormalHmm>> {
        self.fit_unsupervised(x, session)
    }
}

/// Fitted log-normal HMM.
#[derive(Clone, Debug)]
pub struct FittedLogNormalHmm {
    /// Viterbi path.
    pub labels: Vector,
    /// Start distribution.
    pub start: Vector,
    /// Transitions.
    pub trans: Matrix,
    /// Log-mean \(\mu_j\).
    pub mu: Vector,
    /// Log-variance \(\sigma_j^2\).
    pub var: Vector,
    /// Training log-likelihood.
    pub loglik: f64,
}

impl FittedLogNormalHmm {
    fn log_emit_seq(&self, x: &Matrix) -> Vec<Vec<f64>> {
        let t = x.nrows();
        let s = self.mu.len();
        let mut out = vec![vec![f64::NEG_INFINITY; s]; t];
        for ti in 0..t {
            let y = x.get(ti, 0);
            for j in 0..s {
                out[ti][j] = log_lognormal(y, self.mu[j], self.var[j]);
            }
        }
        out
    }

    /// Viterbi path.
    pub fn decode(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let ctx = FitCtx::with_session(session.child("decode"));
        let (path, _) = viterbi_path(&self.start, &self.trans, &self.log_emit_seq(x));
        ctx.finish(path)
    }
}

impl Predict for FittedLogNormalHmm {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        self.decode(x, session)
    }
}

impl FitUnsupervised for LogNormalHmm {
    type Fitted = FittedLogNormalHmm;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedLogNormalHmm>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let t_len = x.nrows();
        let k = self.n_states.max(1);
        if t_len == 0 {
            return ctx.finish(FittedLogNormalHmm {
                labels: empty_labels(0),
                start: init_start(k),
                trans: init_trans(k),
                mu: Vector::zeros(k),
                var: Vector::filled(k, 1.0),
                loglik: f64::NAN,
            });
        }
        let mut n_pos = 0usize;
        for i in 0..t_len {
            if x.get(i, 0) > 0.0 && x.get(i, 0).is_finite() {
                n_pos += 1;
            }
        }
        if n_pos < t_len {
            ctx.push(
                Issue::builder(IssueCode::NonPositiveSeries)
                    .severity(signlred::Severity::Warning)
                    .message("LogNormalHmm skipped non-positive observations")
                    .build(),
            );
        }
        if n_pos < 2 {
            ctx.push(
                Issue::builder(IssueCode::MeaninglessFit)
                    .message("LogNormalHmm needs at least two positive observations")
                    .meaninglessness(Meaninglessness::vacuous(
                        "log-normal HMM",
                        "log-mean is unidentified on a non-positive series",
                        "collect strictly positive observations",
                    ))
                    .build(),
            );
            return ctx.finish(FittedLogNormalHmm {
                labels: empty_labels(t_len),
                start: init_start(k),
                trans: init_trans(k),
                mu: Vector::zeros(k),
                var: Vector::filled(k, 1.0),
                loglik: f64::NAN,
            });
        }
        let mut lsum = 0.0_f64;
        for i in 0..t_len {
            let y = x.get(i, 0);
            if y > 0.0 && y.is_finite() {
                lsum += y.ln();
            }
        }
        let lmean = lsum / n_pos as f64;
        let mut mu = Vector::from_iter((0..k).map(|j| lmean + (j as f64 - 0.5) * 0.3));
        let mut var = Vector::filled(k, 0.25);
        let mut start = init_start(k);
        let mut trans = init_trans(k);
        let mut loglik = f64::NEG_INFINITY;
        let mut last_gamma: Vec<Vec<f64>> = Vec::new();
        for it in 0..self.max_iter.max(1) {
            let log_emit: Vec<Vec<f64>> = (0..t_len)
                .map(|t| {
                    let y = x.get(t, 0);
                    (0..k).map(|j| log_lognormal(y, mu[j], var[j])).collect()
                })
                .collect();
            let Some(fb) = scaled_forward_backward(&mut ctx, &start, &trans, &log_emit) else {
                break;
            };
            loglik = fb.loglik;
            last_gamma = fb.gamma.clone();
            ctx.session.step(it as u64, -loglik, None);
            let mut ns = Vector::zeros(k);
            let mut nt = Matrix::zeros(k, k);
            for j in 0..k {
                ns[j] = fb.gamma[0][j];
            }
            for j in 0..k {
                let mut wsum = 0.0_f64;
                let mut wy = 0.0_f64;
                let mut wy2 = 0.0_f64;
                for t in 0..t_len {
                    let y = x.get(t, 0);
                    if !y.is_finite() || y <= 0.0 {
                        continue;
                    }
                    let z = y.ln();
                    let w = fb.gamma[t][j];
                    wsum += w;
                    wy += w * z;
                    wy2 += w * z * z;
                }
                if wsum > 1e-12 {
                    mu[j] = wy / wsum;
                    var[j] = (wy2 / wsum - mu[j] * mu[j]).max(COV_FLOOR);
                }
            }
            for t in 0..t_len.saturating_sub(1) {
                for i in 0..k {
                    for j in 0..k {
                        nt.set(i, j, nt.get(i, j) + fb.xi[t][i][j]);
                    }
                }
            }
            start = ns;
            trans = nt;
            renormalize_vec(&mut start, TRANS_FLOOR);
            renormalize_rows(&mut trans, TRANS_FLOOR);
        }
        let occup: Vec<f64> = (0..k)
            .map(|j| last_gamma.iter().map(|g| g.get(j).copied().unwrap_or(0.0)).sum())
            .collect();
        if !last_gamma.is_empty() {
            diagnose_chain(&mut ctx, &start, &trans, &occup);
        }
        let dummy = FittedLogNormalHmm {
            labels: Vector::zeros(0),
            start: start.clone(),
            trans: trans.clone(),
            mu: mu.clone(),
            var: var.clone(),
            loglik,
        };
        let (labels, _) = viterbi_path(&start, &trans, &dummy.log_emit_seq(x));
        ctx.finish(FittedLogNormalHmm {
            labels,
            start,
            trans,
            mu,
            var,
            loglik,
        })
    }
}

/// Weibull-emission HMM (shape/scale waiting times).
///
/// Shape is not identification `p`. Distinct from [`ExponentialHmm`]
/// (fixed shape 1) and [`GammaHmm`] (shape/rate, not Weibull).
#[derive(Clone, Debug)]
pub struct WeibullHmm {
    /// Hidden states. Not identification `p`.
    pub n_states: usize,
    /// Baum–Welch cap.
    pub max_iter: usize,
}

impl Default for WeibullHmm {
    fn default() -> Self {
        Self {
            n_states: 2,
            max_iter: 40,
        }
    }
}

impl WeibullHmm {
    /// `k`-state Weibull HMM.
    pub fn new(n_states: usize) -> Self {
        Self {
            n_states,
            ..Self::default()
        }
    }

    /// Fit alias.
    pub fn fit(&mut self, x: &Matrix, session: &Session) -> Result<Qualified<FittedWeibullHmm>> {
        self.fit_unsupervised(x, session)
    }
}

/// Fitted Weibull HMM.
#[derive(Clone, Debug)]
pub struct FittedWeibullHmm {
    /// Viterbi path.
    pub labels: Vector,
    /// Start distribution.
    pub start: Vector,
    /// Transitions.
    pub trans: Matrix,
    /// Shape \(k_j\).
    pub shape: Vector,
    /// Scale \(\lambda_j\).
    pub scale: Vector,
    /// Training log-likelihood.
    pub loglik: f64,
}

impl FittedWeibullHmm {
    fn log_emit_seq(&self, x: &Matrix) -> Vec<Vec<f64>> {
        let t = x.nrows();
        let s = self.shape.len();
        let mut out = vec![vec![f64::NEG_INFINITY; s]; t];
        for ti in 0..t {
            let y = x.get(ti, 0);
            for j in 0..s {
                out[ti][j] = log_weibull(y, self.shape[j], self.scale[j]);
            }
        }
        out
    }

    /// Viterbi path.
    pub fn decode(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let ctx = FitCtx::with_session(session.child("decode"));
        let (path, _) = viterbi_path(&self.start, &self.trans, &self.log_emit_seq(x));
        ctx.finish(path)
    }
}

impl Predict for FittedWeibullHmm {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        self.decode(x, session)
    }
}

impl FitUnsupervised for WeibullHmm {
    type Fitted = FittedWeibullHmm;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedWeibullHmm>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let t_len = x.nrows();
        let k = self.n_states.max(1);
        if t_len == 0 {
            return ctx.finish(FittedWeibullHmm {
                labels: empty_labels(0),
                start: init_start(k),
                trans: init_trans(k),
                shape: Vector::filled(k, 1.0),
                scale: Vector::filled(k, 1.0),
                loglik: f64::NAN,
            });
        }
        let mut n_pos = 0usize;
        for i in 0..t_len {
            if x.get(i, 0) > 0.0 && x.get(i, 0).is_finite() {
                n_pos += 1;
            }
        }
        if n_pos < t_len {
            ctx.push(
                Issue::builder(IssueCode::NonPositiveSeries)
                    .severity(signlred::Severity::Warning)
                    .message("WeibullHmm skipped non-positive observations")
                    .build(),
            );
        }
        if n_pos < 2 {
            ctx.push(
                Issue::builder(IssueCode::MeaninglessFit)
                    .message("WeibullHmm needs at least two positive observations")
                    .meaninglessness(Meaninglessness::vacuous(
                        "Weibull HMM",
                        "shape and scale are unidentified on a non-positive series",
                        "collect strictly positive waiting times",
                    ))
                    .build(),
            );
            return ctx.finish(FittedWeibullHmm {
                labels: empty_labels(t_len),
                start: init_start(k),
                trans: init_trans(k),
                shape: Vector::filled(k, 1.0),
                scale: Vector::filled(k, 1.0),
                loglik: f64::NAN,
            });
        }
        let mut shape = Vector::from_iter((0..k).map(|j| 1.0 + 0.4 * j as f64));
        let mut scale = Vector::from_iter((0..k).map(|j| 1.0 + 0.5 * j as f64));
        let mut start = init_start(k);
        let mut trans = init_trans(k);
        let mut loglik = f64::NEG_INFINITY;
        let mut last_gamma: Vec<Vec<f64>> = Vec::new();
        for it in 0..self.max_iter.max(1) {
            let log_emit: Vec<Vec<f64>> = (0..t_len)
                .map(|t| {
                    let y = x.get(t, 0);
                    (0..k)
                        .map(|j| log_weibull(y, shape[j], scale[j]))
                        .collect()
                })
                .collect();
            let Some(fb) = scaled_forward_backward(&mut ctx, &start, &trans, &log_emit) else {
                break;
            };
            loglik = fb.loglik;
            last_gamma = fb.gamma.clone();
            ctx.session.step(it as u64, -loglik, None);
            let mut ns = Vector::zeros(k);
            let mut nt = Matrix::zeros(k, k);
            for j in 0..k {
                ns[j] = fb.gamma[0][j];
            }
            for j in 0..k {
                let mut wsum = 0.0_f64;
                let mut wy = 0.0_f64;
                let mut wy2 = 0.0_f64;
                for t in 0..t_len {
                    let y = x.get(t, 0);
                    if !y.is_finite() || y <= 0.0 {
                        continue;
                    }
                    let z = y.ln();
                    let w = fb.gamma[t][j];
                    wsum += w;
                    wy += w * z;
                    wy2 += w * z * z;
                }
                if wsum > 1e-12 {
                    let ml = wy / wsum;
                    let vl = (wy2 / wsum - ml * ml).max(1e-8);
                    let sh = (std::f64::consts::PI / (6.0_f64.sqrt() * vl.sqrt())).clamp(0.2, 12.0);
                    shape[j] = sh;
                    scale[j] = (ml + EULER_GAMMA / sh).exp().max(1e-4);
                }
            }
            for t in 0..t_len.saturating_sub(1) {
                for i in 0..k {
                    for j in 0..k {
                        nt.set(i, j, nt.get(i, j) + fb.xi[t][i][j]);
                    }
                }
            }
            start = ns;
            trans = nt;
            renormalize_vec(&mut start, TRANS_FLOOR);
            renormalize_rows(&mut trans, TRANS_FLOOR);
        }
        let occup: Vec<f64> = (0..k)
            .map(|j| last_gamma.iter().map(|g| g.get(j).copied().unwrap_or(0.0)).sum())
            .collect();
        if !last_gamma.is_empty() {
            diagnose_chain(&mut ctx, &start, &trans, &occup);
        }
        let dummy = FittedWeibullHmm {
            labels: Vector::zeros(0),
            start: start.clone(),
            trans: trans.clone(),
            shape: shape.clone(),
            scale: scale.clone(),
            loglik,
        };
        let (labels, _) = viterbi_path(&start, &trans, &dummy.log_emit_seq(x));
        ctx.finish(FittedWeibullHmm {
            labels,
            start,
            trans,
            shape,
            scale,
            loglik,
        })
    }
}

/// Truncated Dirichlet-process Gaussian HMM (GEM / stick-breaking start).
///
/// Concentration \(\alpha\) is not identification `p`. Distinct from
/// [`GaussianHmm`] (uniform start, no GEM) and [`VariationalGaussianHmm`]
/// (variational, no stick-breaking).
#[derive(Clone, Debug)]
pub struct DirichletProcessHmm {
    /// Truncation level. Not identification `p`.
    pub n_states: usize,
    /// DP concentration. Not identification `p`.
    pub alpha: f64,
    /// Baum–Welch cap.
    pub max_iter: usize,
}

impl Default for DirichletProcessHmm {
    fn default() -> Self {
        Self {
            n_states: 4,
            alpha: 1.0,
            max_iter: 40,
        }
    }
}

impl DirichletProcessHmm {
    /// Truncated DP-HMM with `k` atoms and concentration `alpha`.
    pub fn new(n_states: usize, alpha: f64) -> Self {
        Self {
            n_states,
            alpha,
            ..Self::default()
        }
    }

    /// Fit alias.
    pub fn fit(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedDirichletProcessHmm>> {
        self.fit_unsupervised(x, session)
    }
}

/// Fitted truncated DP-HMM.
#[derive(Clone, Debug)]
pub struct FittedDirichletProcessHmm {
    /// Viterbi path.
    pub labels: Vector,
    /// GEM start.
    pub start: Vector,
    /// Transitions.
    pub trans: Matrix,
    /// Emission means.
    pub mu: Vector,
    /// Emission variances.
    pub var: Vector,
    /// Concentration used at fit.
    pub alpha: f64,
    /// Training log-likelihood.
    pub loglik: f64,
}

impl FittedDirichletProcessHmm {
    fn log_emit_seq(&self, x: &Matrix) -> Vec<Vec<f64>> {
        let t = x.nrows();
        let s = self.mu.len();
        let mut out = vec![vec![f64::NEG_INFINITY; s]; t];
        for ti in 0..t {
            let y = x.get(ti, 0);
            for j in 0..s {
                out[ti][j] = log_gauss1(y, self.mu[j], self.var[j]);
            }
        }
        out
    }

    /// Viterbi path.
    pub fn decode(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let ctx = FitCtx::with_session(session.child("decode"));
        let (path, _) = viterbi_path(&self.start, &self.trans, &self.log_emit_seq(x));
        ctx.finish(path)
    }
}

impl Predict for FittedDirichletProcessHmm {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        self.decode(x, session)
    }
}

impl FitUnsupervised for DirichletProcessHmm {
    type Fitted = FittedDirichletProcessHmm;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedDirichletProcessHmm>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let t_len = x.nrows();
        let k = self.n_states.max(1);
        let alpha = if self.alpha.is_finite() && self.alpha > 0.0 {
            self.alpha
        } else {
            1.0
        };
        if t_len == 0 {
            return ctx.finish(FittedDirichletProcessHmm {
                labels: empty_labels(0),
                start: gem_start(k, alpha),
                trans: init_trans(k),
                mu: Vector::zeros(k),
                var: Vector::filled(k, 1.0),
                alpha,
                loglik: f64::NAN,
            });
        }
        let mean = x.column(0).mean();
        let sd = x.column(0).std().max(0.1);
        let mut mu = Vector::from_iter((0..k).map(|j| mean + (j as f64 - 0.5 * (k as f64 - 1.0)) * sd));
        let mut var = Vector::filled(k, sd * sd);
        let mut start = gem_start(k, alpha);
        let mut trans = init_trans(k);
        let mut loglik = f64::NEG_INFINITY;
        let mut last_gamma: Vec<Vec<f64>> = Vec::new();
        for it in 0..self.max_iter.max(1) {
            let log_emit: Vec<Vec<f64>> = (0..t_len)
                .map(|t| {
                    let y = x.get(t, 0);
                    (0..k).map(|j| log_gauss1(y, mu[j], var[j])).collect()
                })
                .collect();
            let Some(fb) = scaled_forward_backward(&mut ctx, &start, &trans, &log_emit) else {
                break;
            };
            loglik = fb.loglik;
            last_gamma = fb.gamma.clone();
            ctx.session.step(it as u64, -loglik, None);
            let mut ns = Vector::zeros(k);
            let mut nt = Matrix::zeros(k, k);
            for j in 0..k {
                ns[j] = fb.gamma[0][j];
            }
            for j in 0..k {
                let mut wsum = 0.0_f64;
                let mut wy = 0.0_f64;
                let mut wy2 = 0.0_f64;
                for t in 0..t_len {
                    let y = x.get(t, 0);
                    if !y.is_finite() {
                        continue;
                    }
                    let w = fb.gamma[t][j];
                    wsum += w;
                    wy += w * y;
                    wy2 += w * y * y;
                }
                if wsum > 1e-12 {
                    mu[j] = wy / wsum;
                    var[j] = (wy2 / wsum - mu[j] * mu[j]).max(COV_FLOOR);
                }
            }
            for t in 0..t_len.saturating_sub(1) {
                for i in 0..k {
                    for j in 0..k {
                        nt.set(i, j, nt.get(i, j) + fb.xi[t][i][j]);
                    }
                }
            }
            start = ns;
            trans = nt;
            renormalize_vec(&mut start, TRANS_FLOOR);
            renormalize_rows(&mut trans, TRANS_FLOOR);
        }
        let occup: Vec<f64> = (0..k)
            .map(|j| last_gamma.iter().map(|g| g.get(j).copied().unwrap_or(0.0)).sum())
            .collect();
        if !last_gamma.is_empty() {
            diagnose_chain(&mut ctx, &start, &trans, &occup);
        }
        let dummy = FittedDirichletProcessHmm {
            labels: Vector::zeros(0),
            start: start.clone(),
            trans: trans.clone(),
            mu: mu.clone(),
            var: var.clone(),
            alpha,
            loglik,
        };
        let (labels, _) = viterbi_path(&start, &trans, &dummy.log_emit_seq(x));
        ctx.finish(FittedDirichletProcessHmm {
            labels,
            start,
            trans,
            mu,
            var,
            alpha,
            loglik,
        })
    }
}

fn log_bernoulli(y: f64, p: f64) -> f64 {
    if !y.is_finite() || y < 0.0 || y > 1.0 || !p.is_finite() {
        return f64::NEG_INFINITY;
    }
    let p = p.clamp(1e-8, 1.0 - 1e-8);
    y * p.ln() + (1.0 - y) * (1.0 - p).ln()
}

/// Bernoulli-emission HMM (binary / fractional codes).
///
/// State count is not identification `p`. Distinct from [`MultinomialHmm`]
/// (integer codes over \(K>2\)) and [`PoissonHmm`] (unbounded counts).
#[derive(Clone, Debug)]
pub struct BernoulliHmm {
    /// Hidden states. Not identification `p`.
    pub n_states: usize,
    /// Baum–Welch cap.
    pub max_iter: usize,
}

impl Default for BernoulliHmm {
    fn default() -> Self {
        Self {
            n_states: 2,
            max_iter: 40,
        }
    }
}

impl BernoulliHmm {
    /// `k`-state Bernoulli HMM.
    pub fn new(n_states: usize) -> Self {
        Self {
            n_states,
            ..Self::default()
        }
    }

    /// Fit alias.
    pub fn fit(&mut self, x: &Matrix, session: &Session) -> Result<Qualified<FittedBernoulliHmm>> {
        self.fit_unsupervised(x, session)
    }
}

/// Fitted Bernoulli HMM.
#[derive(Clone, Debug)]
pub struct FittedBernoulliHmm {
    /// Viterbi path.
    pub labels: Vector,
    /// Start distribution.
    pub start: Vector,
    /// Transitions.
    pub trans: Matrix,
    /// Success probabilities.
    pub p: Vector,
    /// Training log-likelihood.
    pub loglik: f64,
}

impl FittedBernoulliHmm {
    fn log_emit_seq(&self, x: &Matrix) -> Vec<Vec<f64>> {
        let t = x.nrows();
        let s = self.p.len();
        let mut out = vec![vec![f64::NEG_INFINITY; s]; t];
        for ti in 0..t {
            let y = x.get(ti, 0);
            for j in 0..s {
                out[ti][j] = log_bernoulli(y, self.p[j]);
            }
        }
        out
    }

    /// Viterbi path.
    pub fn decode(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let ctx = FitCtx::with_session(session.child("decode"));
        let (path, _) = viterbi_path(&self.start, &self.trans, &self.log_emit_seq(x));
        ctx.finish(path)
    }
}

impl Predict for FittedBernoulliHmm {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        self.decode(x, session)
    }
}

impl FitUnsupervised for BernoulliHmm {
    type Fitted = FittedBernoulliHmm;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedBernoulliHmm>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let t_len = x.nrows();
        let k = self.n_states.max(1);
        if t_len == 0 {
            return ctx.finish(FittedBernoulliHmm {
                labels: empty_labels(0),
                start: init_start(k),
                trans: init_trans(k),
                p: Vector::filled(k, 0.5),
                loglik: f64::NAN,
            });
        }
        let mut p = Vector::from_iter((0..k).map(|j| 0.25 + 0.5 * j as f64 / k.max(1) as f64));
        let mut start = init_start(k);
        let mut trans = init_trans(k);
        let mut loglik = f64::NEG_INFINITY;
        let mut last_gamma: Vec<Vec<f64>> = Vec::new();
        for it in 0..self.max_iter.max(1) {
            let log_emit: Vec<Vec<f64>> = (0..t_len)
                .map(|t| {
                    let y = x.get(t, 0);
                    (0..k).map(|j| log_bernoulli(y, p[j])).collect()
                })
                .collect();
            let Some(fb) = scaled_forward_backward(&mut ctx, &start, &trans, &log_emit) else {
                break;
            };
            loglik = fb.loglik;
            last_gamma = fb.gamma.clone();
            ctx.session.step(it as u64, -loglik, None);
            let mut ns = Vector::zeros(k);
            let mut nt = Matrix::zeros(k, k);
            for j in 0..k {
                ns[j] = fb.gamma[0][j];
            }
            for j in 0..k {
                let mut wsum = 0.0_f64;
                let mut wy = 0.0_f64;
                for t in 0..t_len {
                    let y = x.get(t, 0);
                    if !y.is_finite() || y < 0.0 || y > 1.0 {
                        continue;
                    }
                    let w = fb.gamma[t][j];
                    wsum += w;
                    wy += w * y;
                }
                if wsum > 1e-12 {
                    p[j] = (wy / wsum).clamp(1e-3, 1.0 - 1e-3);
                }
            }
            for t in 0..t_len.saturating_sub(1) {
                for i in 0..k {
                    for j in 0..k {
                        nt.set(i, j, nt.get(i, j) + fb.xi[t][i][j]);
                    }
                }
            }
            start = ns;
            trans = nt;
            renormalize_vec(&mut start, TRANS_FLOOR);
            renormalize_rows(&mut trans, TRANS_FLOOR);
        }
        let occup: Vec<f64> = (0..k)
            .map(|j| last_gamma.iter().map(|g| g.get(j).copied().unwrap_or(0.0)).sum())
            .collect();
        if !last_gamma.is_empty() {
            diagnose_chain(&mut ctx, &start, &trans, &occup);
        }
        let dummy = FittedBernoulliHmm {
            labels: Vector::zeros(0),
            start: start.clone(),
            trans: trans.clone(),
            p: p.clone(),
            loglik,
        };
        let (labels, _) = viterbi_path(&start, &trans, &dummy.log_emit_seq(x));
        ctx.finish(FittedBernoulliHmm {
            labels,
            start,
            trans,
            p,
            loglik,
        })
    }
}

/// Sticky hierarchical-DP Gaussian HMM (GEM start plus self-transition bias).
///
/// Concentration and \(\kappa\) are not identification `p`. Distinct from
/// [`DirichletProcessHmm`] (no sticky bias) and [`StickyHmm`] (uniform start).
#[derive(Clone, Debug)]
pub struct StickyHdpHmm {
    /// Truncation level. Not identification `p`.
    pub n_states: usize,
    /// DP concentration. Not identification `p`.
    pub alpha: f64,
    /// Sticky self-transition mass. Not identification `p`.
    pub kappa: f64,
    /// Baum–Welch cap.
    pub max_iter: usize,
}

impl Default for StickyHdpHmm {
    fn default() -> Self {
        Self {
            n_states: 4,
            alpha: 1.0,
            kappa: 4.0,
            max_iter: 40,
        }
    }
}

impl StickyHdpHmm {
    /// Sticky HDP-HMM with truncation `k` and concentration `alpha`.
    pub fn new(n_states: usize, alpha: f64) -> Self {
        Self {
            n_states,
            alpha,
            ..Self::default()
        }
    }

    /// Fit alias.
    pub fn fit(&mut self, x: &Matrix, session: &Session) -> Result<Qualified<FittedStickyHdpHmm>> {
        self.fit_unsupervised(x, session)
    }
}

/// Fitted sticky HDP-HMM.
#[derive(Clone, Debug)]
pub struct FittedStickyHdpHmm {
    /// Viterbi path.
    pub labels: Vector,
    /// GEM start.
    pub start: Vector,
    /// Sticky transitions.
    pub trans: Matrix,
    /// Emission means.
    pub mu: Vector,
    /// Emission variances.
    pub var: Vector,
    /// Concentration.
    pub alpha: f64,
    /// Sticky mass.
    pub kappa: f64,
    /// Training log-likelihood.
    pub loglik: f64,
}

impl FittedStickyHdpHmm {
    fn log_emit_seq(&self, x: &Matrix) -> Vec<Vec<f64>> {
        let t = x.nrows();
        let s = self.mu.len();
        let mut out = vec![vec![f64::NEG_INFINITY; s]; t];
        for ti in 0..t {
            let y = x.get(ti, 0);
            for j in 0..s {
                out[ti][j] = log_gauss1(y, self.mu[j], self.var[j]);
            }
        }
        out
    }

    /// Viterbi path.
    pub fn decode(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let ctx = FitCtx::with_session(session.child("decode"));
        let (path, _) = viterbi_path(&self.start, &self.trans, &self.log_emit_seq(x));
        ctx.finish(path)
    }
}

impl Predict for FittedStickyHdpHmm {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        self.decode(x, session)
    }
}

impl FitUnsupervised for StickyHdpHmm {
    type Fitted = FittedStickyHdpHmm;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedStickyHdpHmm>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let t_len = x.nrows();
        let k = self.n_states.max(1);
        let alpha = if self.alpha.is_finite() && self.alpha > 0.0 {
            self.alpha
        } else {
            1.0
        };
        let kappa = if self.kappa.is_finite() && self.kappa >= 0.0 {
            self.kappa
        } else {
            4.0
        };
        if t_len == 0 {
            return ctx.finish(FittedStickyHdpHmm {
                labels: empty_labels(0),
                start: gem_start(k, alpha),
                trans: init_trans(k),
                mu: Vector::zeros(k),
                var: Vector::filled(k, 1.0),
                alpha,
                kappa,
                loglik: f64::NAN,
            });
        }
        let mean = x.column(0).mean();
        let sd = x.column(0).std().max(0.1);
        let mut mu = Vector::from_iter((0..k).map(|j| mean + (j as f64 - 0.5 * (k as f64 - 1.0)) * sd));
        let mut var = Vector::filled(k, sd * sd);
        let mut start = gem_start(k, alpha);
        let mut trans = init_trans(k);
        let mut loglik = f64::NEG_INFINITY;
        let mut last_gamma: Vec<Vec<f64>> = Vec::new();
        for it in 0..self.max_iter.max(1) {
            let log_emit: Vec<Vec<f64>> = (0..t_len)
                .map(|t| {
                    let y = x.get(t, 0);
                    (0..k).map(|j| log_gauss1(y, mu[j], var[j])).collect()
                })
                .collect();
            let Some(fb) = scaled_forward_backward(&mut ctx, &start, &trans, &log_emit) else {
                break;
            };
            loglik = fb.loglik;
            last_gamma = fb.gamma.clone();
            ctx.session.step(it as u64, -loglik, None);
            let mut ns = Vector::zeros(k);
            let mut nt = Matrix::zeros(k, k);
            for j in 0..k {
                ns[j] = fb.gamma[0][j];
            }
            for j in 0..k {
                let mut wsum = 0.0_f64;
                let mut wy = 0.0_f64;
                let mut wy2 = 0.0_f64;
                for t in 0..t_len {
                    let y = x.get(t, 0);
                    if !y.is_finite() {
                        continue;
                    }
                    let w = fb.gamma[t][j];
                    wsum += w;
                    wy += w * y;
                    wy2 += w * y * y;
                }
                if wsum > 1e-12 {
                    mu[j] = wy / wsum;
                    var[j] = (wy2 / wsum - mu[j] * mu[j]).max(COV_FLOOR);
                }
            }
            for t in 0..t_len.saturating_sub(1) {
                for i in 0..k {
                    for j in 0..k {
                        nt.set(i, j, nt.get(i, j) + fb.xi[t][i][j]);
                    }
                }
            }
            for j in 0..k {
                nt.set(j, j, nt.get(j, j) + kappa);
            }
            start = ns;
            trans = nt;
            renormalize_vec(&mut start, TRANS_FLOOR);
            renormalize_rows(&mut trans, TRANS_FLOOR);
        }
        let occup: Vec<f64> = (0..k)
            .map(|j| last_gamma.iter().map(|g| g.get(j).copied().unwrap_or(0.0)).sum())
            .collect();
        if !last_gamma.is_empty() {
            diagnose_chain(&mut ctx, &start, &trans, &occup);
        }
        let dummy = FittedStickyHdpHmm {
            labels: Vector::zeros(0),
            start: start.clone(),
            trans: trans.clone(),
            mu: mu.clone(),
            var: var.clone(),
            alpha,
            kappa,
            loglik,
        };
        let (labels, _) = viterbi_path(&start, &trans, &dummy.log_emit_seq(x));
        ctx.finish(FittedStickyHdpHmm {
            labels,
            start,
            trans,
            mu,
            var,
            alpha,
            kappa,
            loglik,
        })
    }
}

/// Left-right explicit-duration HSMM (Bakis sojourns).
///
/// Duration cap is not identification `p`. Distinct from [`Hsmm`] (ergodic
/// next-state kernel) and [`GaussianHmmLeftRight`] (no explicit duration).
#[derive(Clone, Debug)]
pub struct LeftRightHsmm {
    /// Hidden states. Not identification `p`.
    pub n_states: usize,
    /// Maximum sojourn. Not identification `p`.
    pub max_duration: usize,
    /// Segmental iterations.
    pub max_iter: usize,
}

impl Default for LeftRightHsmm {
    fn default() -> Self {
        Self {
            n_states: 2,
            max_duration: 8,
            max_iter: 12,
        }
    }
}

impl LeftRightHsmm {
    /// Left-right HSMM with `k` states.
    pub fn new(n_states: usize) -> Self {
        Self {
            n_states,
            ..Self::default()
        }
    }

    /// Fit alias.
    pub fn fit(&mut self, x: &Matrix, session: &Session) -> Result<Qualified<FittedLeftRightHsmm>> {
        self.fit_unsupervised(x, session)
    }
}

/// Fitted left-right HSMM.
#[derive(Clone, Debug)]
pub struct FittedLeftRightHsmm {
    /// Segmental Viterbi path.
    pub labels: Vector,
    /// Start (mass on state 0).
    pub start: Vector,
    /// Forward-only next-state kernel.
    pub trans: Matrix,
    /// Emission mean.
    pub mu: Vector,
    /// Emission variance.
    pub var: Vector,
    /// Geometric stay parameter.
    pub rho: Vector,
    /// Path score.
    pub loglik: f64,
}

impl FittedLeftRightHsmm {
    /// Segmental Viterbi path.
    pub fn decode(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let ctx = FitCtx::with_session(session.child("decode"));
        let (path, _) = hsmm_viterbi(
            x,
            &self.start,
            &self.trans,
            &self.mu,
            &self.var,
            &self.rho,
            8,
        );
        ctx.finish(path)
    }
}

impl Predict for FittedLeftRightHsmm {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        self.decode(x, session)
    }
}

impl FitUnsupervised for LeftRightHsmm {
    type Fitted = FittedLeftRightHsmm;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedLeftRightHsmm>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let t_len = x.nrows();
        let k = self.n_states.max(1);
        let dmax = self.max_duration.max(1);
        if t_len == 0 {
            return ctx.finish(FittedLeftRightHsmm {
                labels: empty_labels(0),
                start: init_start(k),
                trans: init_trans(k),
                mu: Vector::zeros(k),
                var: Vector::filled(k, 1.0),
                rho: Vector::filled(k, 0.7),
                loglik: f64::NAN,
            });
        }
        let mean = x.column(0).mean();
        let sd = x.column(0).std().max(0.1);
        let mut mu = Vector::from_iter((0..k).map(|j| mean + (j as f64 - 0.5) * sd));
        let mut var = Vector::filled(k, sd * sd);
        let mut rho = Vector::filled(k, 0.75);
        let mut start = init_start(k);
        let mut trans = init_trans(k);
        for i in 0..k {
            trans.set(i, i, 0.0);
            for j in 0..i {
                trans.set(i, j, 0.0);
            }
        }
        enforce_left_right(&mut start, &mut trans);
        for i in 0..k {
            trans.set(i, i, 0.0);
            for j in 0..i {
                trans.set(i, j, 0.0);
            }
        }
        let mut labels = empty_labels(t_len);
        let mut loglik = f64::NEG_INFINITY;
        for it in 0..self.max_iter.max(1) {
            let (path, score) = hsmm_viterbi(x, &start, &trans, &mu, &var, &rho, dmax);
            labels = path;
            loglik = score;
            ctx.session.step(it as u64, -loglik, None);
            let mut nrun = vec![0.0_f64; k];
            let mut slen = vec![0.0_f64; k];
            let mut wy = vec![0.0_f64; k];
            let mut wy2 = vec![0.0_f64; k];
            let mut wsum = vec![0.0_f64; k];
            let mut nt = Matrix::zeros(k, k);
            let mut t0 = 0usize;
            while t0 < t_len {
                let j = labels[t0] as usize;
                let j = j.min(k.saturating_sub(1));
                let mut t1 = t0 + 1;
                while t1 < t_len && (labels[t1] as usize) == j {
                    t1 += 1;
                }
                let d = (t1 - t0) as f64;
                nrun[j] += 1.0;
                slen[j] += d;
                for t in t0..t1 {
                    let y = x.get(t, 0);
                    if y.is_finite() {
                        wsum[j] += 1.0;
                        wy[j] += y;
                        wy2[j] += y * y;
                    }
                }
                if t1 < t_len {
                    let nxt = (labels[t1] as usize).min(k.saturating_sub(1));
                    if nxt != j {
                        nt.set(j, nxt, nt.get(j, nxt) + 1.0);
                    }
                }
                t0 = t1;
            }
            for j in 0..k {
                if wsum[j] > 0.0 {
                    mu[j] = wy[j] / wsum[j];
                    var[j] = (wy2[j] / wsum[j] - mu[j] * mu[j]).max(COV_FLOOR);
                }
                if nrun[j] > 0.0 {
                    let md = (slen[j] / nrun[j]).max(1.0);
                    rho[j] = (1.0 - 1.0 / md).clamp(1e-3, 0.99);
                }
            }
            trans = nt;
            for i in 0..k {
                trans.set(i, i, 0.0);
                for j in 0..i {
                    trans.set(i, j, 0.0);
                }
            }
            for i in 0..k {
                let mut s = 0.0_f64;
                for j in (i + 1)..k {
                    s += trans.get(i, j).max(0.0);
                }
                if s > 0.0 {
                    for j in (i + 1)..k {
                        trans.set(i, j, trans.get(i, j).max(0.0) / s);
                    }
                }
            }
            enforce_left_right(&mut start, &mut trans);
            for i in 0..k {
                trans.set(i, i, 0.0);
                for j in 0..i {
                    trans.set(i, j, 0.0);
                }
            }
        }
        ctx.finish(FittedLeftRightHsmm {
            labels,
            start,
            trans,
            mu,
            var,
            rho,
            loglik,
        })
    }
}

fn log_binomial(y: f64, n: f64, p: f64) -> f64 {
    if !y.is_finite() || y < 0.0 || y > n || n < 1.0 || !p.is_finite() {
        return f64::NEG_INFINITY;
    }
    let p = p.clamp(1e-8, 1.0 - 1e-8);
    crate::special::ln_gamma(n + 1.0) - crate::special::ln_gamma(y + 1.0)
        - crate::special::ln_gamma(n - y + 1.0)
        + y * p.ln()
        + (n - y) * (1.0 - p).ln()
}

fn log_geometric(y: f64, p: f64) -> f64 {
    if !y.is_finite() || y < 0.0 || !p.is_finite() {
        return f64::NEG_INFINITY;
    }
    let p = p.clamp(1e-8, 1.0 - 1e-8);
    y * (1.0 - p).ln() + p.ln()
}

fn log_betabin(y: f64, n: f64, alpha: f64, beta: f64) -> f64 {
    if !y.is_finite() || y < 0.0 || y > n || n < 1.0 || alpha <= 0.0 || beta <= 0.0 {
        return f64::NEG_INFINITY;
    }
    crate::special::ln_gamma(n + 1.0) - crate::special::ln_gamma(y + 1.0)
        - crate::special::ln_gamma(n - y + 1.0)
        + crate::special::ln_gamma(alpha + y)
        + crate::special::ln_gamma(beta + n - y)
        - crate::special::ln_gamma(alpha + beta + n)
        - crate::special::ln_gamma(alpha)
        - crate::special::ln_gamma(beta)
        + crate::special::ln_gamma(alpha + beta)
}

fn log_ordinal(y: f64, mu: f64, n_class: usize) -> f64 {
    if !y.is_finite() || n_class < 2 {
        return f64::NEG_INFINITY;
    }
    let k = y.round().clamp(0.0, (n_class - 1) as f64) as usize;
    let sig = |z: f64| 1.0 / (1.0 + (-z).clamp(-40.0, 40.0).exp());
    let p = if k == 0 {
        sig(0.5 - mu)
    } else if k + 1 == n_class {
        1.0 - sig((n_class as f64 - 1.5) - mu)
    } else {
        sig((k as f64 + 0.5) - mu) - sig((k as f64 - 0.5) - mu)
    };
    p.clamp(1e-12, 1.0).ln()
}

fn hmm_em_trans(
    fb_xi: &[Vec<Vec<f64>>],
    fb_g0: &[f64],
    k: usize,
    t_len: usize,
) -> (Vector, Matrix) {
    let mut ns = Vector::zeros(k);
    let mut nt = Matrix::zeros(k, k);
    for j in 0..k {
        ns[j] = fb_g0.get(j).copied().unwrap_or(0.0);
    }
    for t in 0..t_len.saturating_sub(1) {
        for i in 0..k {
            for j in 0..k {
                nt.set(i, j, nt.get(i, j) + fb_xi[t][i][j]);
            }
        }
    }
    renormalize_vec(&mut ns, TRANS_FLOOR);
    renormalize_rows(&mut nt, TRANS_FLOOR);
    (ns, nt)
}

/// Binomial-emission HMM (fixed trial count).
///
/// Trial count is not identification `p`. Distinct from [`BernoulliHmm`]
/// (\(n=1\)) and [`PoissonHmm`] (unbounded).
#[derive(Clone, Debug)]
pub struct BinomialHmm {
    /// Hidden states. Not identification `p`.
    pub n_states: usize,
    /// Trials per observation. Not identification `p`.
    pub n_trials: f64,
    /// Baum–Welch cap.
    pub max_iter: usize,
}

impl Default for BinomialHmm {
    fn default() -> Self {
        Self {
            n_states: 2,
            n_trials: 8.0,
            max_iter: 40,
        }
    }
}

impl BinomialHmm {
    /// `k`-state binomial HMM with `n_trials` trials.
    pub fn new(n_states: usize, n_trials: f64) -> Self {
        Self {
            n_states,
            n_trials,
            ..Self::default()
        }
    }

    /// Fit alias.
    pub fn fit(&mut self, x: &Matrix, session: &Session) -> Result<Qualified<FittedBinomialHmm>> {
        self.fit_unsupervised(x, session)
    }
}

/// Fitted binomial HMM.
#[derive(Clone, Debug)]
pub struct FittedBinomialHmm {
    /// Viterbi path.
    pub labels: Vector,
    /// Start distribution.
    pub start: Vector,
    /// Transitions.
    pub trans: Matrix,
    /// Success probabilities.
    pub p: Vector,
    /// Trial count.
    pub n_trials: f64,
    /// Training log-likelihood.
    pub loglik: f64,
}

impl FittedBinomialHmm {
    fn log_emit_seq(&self, x: &Matrix) -> Vec<Vec<f64>> {
        let t = x.nrows();
        let s = self.p.len();
        let mut out = vec![vec![f64::NEG_INFINITY; s]; t];
        for ti in 0..t {
            let y = x.get(ti, 0);
            for j in 0..s {
                out[ti][j] = log_binomial(y, self.n_trials, self.p[j]);
            }
        }
        out
    }

    /// Viterbi path.
    pub fn decode(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let ctx = FitCtx::with_session(session.child("decode"));
        let (path, _) = viterbi_path(&self.start, &self.trans, &self.log_emit_seq(x));
        ctx.finish(path)
    }
}

impl Predict for FittedBinomialHmm {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        self.decode(x, session)
    }
}

impl FitUnsupervised for BinomialHmm {
    type Fitted = FittedBinomialHmm;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedBinomialHmm>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let t_len = x.nrows();
        let k = self.n_states.max(1);
        let nt = if self.n_trials.is_finite() && self.n_trials >= 1.0 {
            self.n_trials
        } else {
            8.0
        };
        if t_len == 0 {
            return ctx.finish(FittedBinomialHmm {
                labels: empty_labels(0),
                start: init_start(k),
                trans: init_trans(k),
                p: Vector::filled(k, 0.5),
                n_trials: nt,
                loglik: f64::NAN,
            });
        }
        let mut p = Vector::from_iter((0..k).map(|j| 0.2 + 0.6 * j as f64 / k.max(1) as f64));
        let mut start = init_start(k);
        let mut trans = init_trans(k);
        let mut loglik = f64::NEG_INFINITY;
        let mut last_gamma: Vec<Vec<f64>> = Vec::new();
        for it in 0..self.max_iter.max(1) {
            let dummy = FittedBinomialHmm {
                labels: Vector::zeros(0),
                start: start.clone(),
                trans: trans.clone(),
                p: p.clone(),
                n_trials: nt,
                loglik,
            };
            let Some(fb) = scaled_forward_backward(&mut ctx, &start, &trans, &dummy.log_emit_seq(x))
            else {
                break;
            };
            loglik = fb.loglik;
            last_gamma = fb.gamma.clone();
            ctx.session.step(it as u64, -loglik, None);
            for j in 0..k {
                let mut wsum = 0.0_f64;
                let mut wy = 0.0_f64;
                for t in 0..t_len {
                    let y = x.get(t, 0);
                    if !y.is_finite() || y < 0.0 || y > nt {
                        continue;
                    }
                    wsum += fb.gamma[t][j];
                    wy += fb.gamma[t][j] * y;
                }
                if wsum > 1e-12 {
                    p[j] = (wy / (wsum * nt)).clamp(1e-3, 1.0 - 1e-3);
                }
            }
            let (ns, ntr) = hmm_em_trans(&fb.xi, &fb.gamma[0], k, t_len);
            start = ns;
            trans = ntr;
        }
        if !last_gamma.is_empty() {
            let occup: Vec<f64> = (0..k)
                .map(|j| last_gamma.iter().map(|g| g.get(j).copied().unwrap_or(0.0)).sum())
                .collect();
            diagnose_chain(&mut ctx, &start, &trans, &occup);
        }
        let dummy = FittedBinomialHmm {
            labels: Vector::zeros(0),
            start: start.clone(),
            trans: trans.clone(),
            p: p.clone(),
            n_trials: nt,
            loglik,
        };
        let (labels, _) = viterbi_path(&start, &trans, &dummy.log_emit_seq(x));
        ctx.finish(FittedBinomialHmm {
            labels,
            start,
            trans,
            p,
            n_trials: nt,
            loglik,
        })
    }
}

/// Geometric-emission HMM (discrete waiting times).
///
/// State count is not identification `p`. Distinct from [`ExponentialHmm`]
/// (continuous) and [`PoissonHmm`] (counts without a stop probability).
#[derive(Clone, Debug)]
pub struct GeometricHmm {
    /// Hidden states. Not identification `p`.
    pub n_states: usize,
    /// Baum–Welch cap.
    pub max_iter: usize,
}

impl Default for GeometricHmm {
    fn default() -> Self {
        Self {
            n_states: 2,
            max_iter: 40,
        }
    }
}

impl GeometricHmm {
    /// `k`-state geometric HMM.
    pub fn new(n_states: usize) -> Self {
        Self {
            n_states,
            ..Self::default()
        }
    }

    /// Fit alias.
    pub fn fit(&mut self, x: &Matrix, session: &Session) -> Result<Qualified<FittedGeometricHmm>> {
        self.fit_unsupervised(x, session)
    }
}

/// Fitted geometric HMM.
#[derive(Clone, Debug)]
pub struct FittedGeometricHmm {
    /// Viterbi path.
    pub labels: Vector,
    /// Start distribution.
    pub start: Vector,
    /// Transitions.
    pub trans: Matrix,
    /// Success probabilities.
    pub p: Vector,
    /// Training log-likelihood.
    pub loglik: f64,
}

impl FittedGeometricHmm {
    fn log_emit_seq(&self, x: &Matrix) -> Vec<Vec<f64>> {
        let t = x.nrows();
        let s = self.p.len();
        let mut out = vec![vec![f64::NEG_INFINITY; s]; t];
        for ti in 0..t {
            let y = x.get(ti, 0);
            for j in 0..s {
                out[ti][j] = log_geometric(y, self.p[j]);
            }
        }
        out
    }

    /// Viterbi path.
    pub fn decode(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let ctx = FitCtx::with_session(session.child("decode"));
        let (path, _) = viterbi_path(&self.start, &self.trans, &self.log_emit_seq(x));
        ctx.finish(path)
    }
}

impl Predict for FittedGeometricHmm {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        self.decode(x, session)
    }
}

impl FitUnsupervised for GeometricHmm {
    type Fitted = FittedGeometricHmm;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedGeometricHmm>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let t_len = x.nrows();
        let k = self.n_states.max(1);
        if t_len == 0 {
            return ctx.finish(FittedGeometricHmm {
                labels: empty_labels(0),
                start: init_start(k),
                trans: init_trans(k),
                p: Vector::filled(k, 0.5),
                loglik: f64::NAN,
            });
        }
        let mut p = Vector::from_iter((0..k).map(|j| 0.3 + 0.4 * j as f64 / k.max(1) as f64));
        let mut start = init_start(k);
        let mut trans = init_trans(k);
        let mut loglik = f64::NEG_INFINITY;
        let mut last_gamma: Vec<Vec<f64>> = Vec::new();
        for it in 0..self.max_iter.max(1) {
            let dummy = FittedGeometricHmm {
                labels: Vector::zeros(0),
                start: start.clone(),
                trans: trans.clone(),
                p: p.clone(),
                loglik,
            };
            let Some(fb) = scaled_forward_backward(&mut ctx, &start, &trans, &dummy.log_emit_seq(x))
            else {
                break;
            };
            loglik = fb.loglik;
            last_gamma = fb.gamma.clone();
            ctx.session.step(it as u64, -loglik, None);
            for j in 0..k {
                let mut wsum = 0.0_f64;
                let mut wy = 0.0_f64;
                for t in 0..t_len {
                    let y = x.get(t, 0);
                    if !y.is_finite() || y < 0.0 {
                        continue;
                    }
                    wsum += fb.gamma[t][j];
                    wy += fb.gamma[t][j] * y;
                }
                if wsum > 1e-12 {
                    let mu = wy / wsum;
                    p[j] = (1.0 / (1.0 + mu)).clamp(1e-3, 0.99);
                }
            }
            let (ns, ntr) = hmm_em_trans(&fb.xi, &fb.gamma[0], k, t_len);
            start = ns;
            trans = ntr;
        }
        if !last_gamma.is_empty() {
            let occup: Vec<f64> = (0..k)
                .map(|j| last_gamma.iter().map(|g| g.get(j).copied().unwrap_or(0.0)).sum())
                .collect();
            diagnose_chain(&mut ctx, &start, &trans, &occup);
        }
        let dummy = FittedGeometricHmm {
            labels: Vector::zeros(0),
            start: start.clone(),
            trans: trans.clone(),
            p: p.clone(),
            loglik,
        };
        let (labels, _) = viterbi_path(&start, &trans, &dummy.log_emit_seq(x));
        ctx.finish(FittedGeometricHmm {
            labels,
            start,
            trans,
            p,
            loglik,
        })
    }
}

/// Beta-binomial emission HMM (overdispersed trials).
///
/// Trial count is not identification `p`. Distinct from [`BinomialHmm`]
/// (no overdispersion) and [`NegativeBinomialHmm`] (unbounded).
#[derive(Clone, Debug)]
pub struct BetaBinomialHmm {
    /// Hidden states. Not identification `p`.
    pub n_states: usize,
    /// Trials per observation. Not identification `p`.
    pub n_trials: f64,
    /// Baum–Welch cap.
    pub max_iter: usize,
}

impl Default for BetaBinomialHmm {
    fn default() -> Self {
        Self {
            n_states: 2,
            n_trials: 8.0,
            max_iter: 40,
        }
    }
}

impl BetaBinomialHmm {
    /// `k`-state beta-binomial HMM.
    pub fn new(n_states: usize, n_trials: f64) -> Self {
        Self {
            n_states,
            n_trials,
            ..Self::default()
        }
    }

    /// Fit alias.
    pub fn fit(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedBetaBinomialHmm>> {
        self.fit_unsupervised(x, session)
    }
}

/// Fitted beta-binomial HMM.
#[derive(Clone, Debug)]
pub struct FittedBetaBinomialHmm {
    /// Viterbi path.
    pub labels: Vector,
    /// Start distribution.
    pub start: Vector,
    /// Transitions.
    pub trans: Matrix,
    /// Shape \(\alpha_j\).
    pub alpha: Vector,
    /// Shape \(\beta_j\).
    pub beta: Vector,
    /// Trial count.
    pub n_trials: f64,
    /// Training log-likelihood.
    pub loglik: f64,
}

impl FittedBetaBinomialHmm {
    fn log_emit_seq(&self, x: &Matrix) -> Vec<Vec<f64>> {
        let t = x.nrows();
        let s = self.alpha.len();
        let mut out = vec![vec![f64::NEG_INFINITY; s]; t];
        for ti in 0..t {
            let y = x.get(ti, 0);
            for j in 0..s {
                out[ti][j] = log_betabin(y, self.n_trials, self.alpha[j], self.beta[j]);
            }
        }
        out
    }

    /// Viterbi path.
    pub fn decode(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let ctx = FitCtx::with_session(session.child("decode"));
        let (path, _) = viterbi_path(&self.start, &self.trans, &self.log_emit_seq(x));
        ctx.finish(path)
    }
}

impl Predict for FittedBetaBinomialHmm {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        self.decode(x, session)
    }
}

impl FitUnsupervised for BetaBinomialHmm {
    type Fitted = FittedBetaBinomialHmm;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedBetaBinomialHmm>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let t_len = x.nrows();
        let k = self.n_states.max(1);
        let nt = if self.n_trials.is_finite() && self.n_trials >= 1.0 {
            self.n_trials
        } else {
            8.0
        };
        if t_len == 0 {
            return ctx.finish(FittedBetaBinomialHmm {
                labels: empty_labels(0),
                start: init_start(k),
                trans: init_trans(k),
                alpha: Vector::filled(k, 1.0),
                beta: Vector::filled(k, 1.0),
                n_trials: nt,
                loglik: f64::NAN,
            });
        }
        let mut alpha = Vector::from_iter((0..k).map(|j| 1.0 + 0.5 * j as f64));
        let mut beta = Vector::from_iter((0..k).map(|j| 2.0 - 0.3 * j as f64));
        let mut start = init_start(k);
        let mut trans = init_trans(k);
        let mut loglik = f64::NEG_INFINITY;
        let mut last_gamma: Vec<Vec<f64>> = Vec::new();
        for it in 0..self.max_iter.max(1) {
            let dummy = FittedBetaBinomialHmm {
                labels: Vector::zeros(0),
                start: start.clone(),
                trans: trans.clone(),
                alpha: alpha.clone(),
                beta: beta.clone(),
                n_trials: nt,
                loglik,
            };
            let Some(fb) = scaled_forward_backward(&mut ctx, &start, &trans, &dummy.log_emit_seq(x))
            else {
                break;
            };
            loglik = fb.loglik;
            last_gamma = fb.gamma.clone();
            ctx.session.step(it as u64, -loglik, None);
            for j in 0..k {
                let mut wsum = 0.0_f64;
                let mut wy = 0.0_f64;
                let mut wy2 = 0.0_f64;
                for t in 0..t_len {
                    let y = x.get(t, 0);
                    if !y.is_finite() || y < 0.0 || y > nt {
                        continue;
                    }
                    wsum += fb.gamma[t][j];
                    wy += fb.gamma[t][j] * y;
                    wy2 += fb.gamma[t][j] * y * y;
                }
                if wsum > 1e-12 {
                    let mu = wy / wsum;
                    let var = (wy2 / wsum - mu * mu).max(1e-8);
                    let pi = (mu / nt).clamp(1e-3, 1.0 - 1e-3);
                    let binom = nt * pi * (1.0 - pi);
                    let rho = ((var / binom.max(1e-8) - 1.0) / (nt - 1.0).max(1.0)).clamp(1e-4, 0.9);
                    let conc = (1.0 / rho - 1.0).clamp(0.5, 80.0);
                    alpha[j] = (pi * conc).max(0.05);
                    beta[j] = ((1.0 - pi) * conc).max(0.05);
                }
            }
            let (ns, ntr) = hmm_em_trans(&fb.xi, &fb.gamma[0], k, t_len);
            start = ns;
            trans = ntr;
        }
        if !last_gamma.is_empty() {
            let occup: Vec<f64> = (0..k)
                .map(|j| last_gamma.iter().map(|g| g.get(j).copied().unwrap_or(0.0)).sum())
                .collect();
            diagnose_chain(&mut ctx, &start, &trans, &occup);
        }
        let dummy = FittedBetaBinomialHmm {
            labels: Vector::zeros(0),
            start: start.clone(),
            trans: trans.clone(),
            alpha: alpha.clone(),
            beta: beta.clone(),
            n_trials: nt,
            loglik,
        };
        let (labels, _) = viterbi_path(&start, &trans, &dummy.log_emit_seq(x));
        ctx.finish(FittedBetaBinomialHmm {
            labels,
            start,
            trans,
            alpha,
            beta,
            n_trials: nt,
            loglik,
        })
    }
}

/// Ordered-logit emission HMM (cumulative thresholds).
///
/// Class count is not identification `p`. Distinct from [`MultinomialHmm`]
/// (unordered codes) and [`BernoulliHmm`] (no shared cutpoints).
#[derive(Clone, Debug)]
pub struct OrdinalHmm {
    /// Hidden states. Not identification `p`.
    pub n_states: usize,
    /// Baum–Welch cap.
    pub max_iter: usize,
}

impl Default for OrdinalHmm {
    fn default() -> Self {
        Self {
            n_states: 2,
            max_iter: 40,
        }
    }
}

impl OrdinalHmm {
    /// `k`-state ordinal HMM.
    pub fn new(n_states: usize) -> Self {
        Self {
            n_states,
            ..Self::default()
        }
    }

    /// Fit alias.
    pub fn fit(&mut self, x: &Matrix, session: &Session) -> Result<Qualified<FittedOrdinalHmm>> {
        self.fit_unsupervised(x, session)
    }
}

/// Fitted ordinal HMM.
#[derive(Clone, Debug)]
pub struct FittedOrdinalHmm {
    /// Viterbi path.
    pub labels: Vector,
    /// Start distribution.
    pub start: Vector,
    /// Transitions.
    pub trans: Matrix,
    /// Locations \(\mu_j\).
    pub mu: Vector,
    /// Number of ordered classes.
    pub n_class: usize,
    /// Training log-likelihood.
    pub loglik: f64,
}

impl FittedOrdinalHmm {
    fn log_emit_seq(&self, x: &Matrix) -> Vec<Vec<f64>> {
        let t = x.nrows();
        let s = self.mu.len();
        let mut out = vec![vec![f64::NEG_INFINITY; s]; t];
        for ti in 0..t {
            let y = x.get(ti, 0);
            for j in 0..s {
                out[ti][j] = log_ordinal(y, self.mu[j], self.n_class);
            }
        }
        out
    }

    /// Viterbi path.
    pub fn decode(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let ctx = FitCtx::with_session(session.child("decode"));
        let (path, _) = viterbi_path(&self.start, &self.trans, &self.log_emit_seq(x));
        ctx.finish(path)
    }
}

impl Predict for FittedOrdinalHmm {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        self.decode(x, session)
    }
}

impl FitUnsupervised for OrdinalHmm {
    type Fitted = FittedOrdinalHmm;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedOrdinalHmm>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let t_len = x.nrows();
        let k = self.n_states.max(1);
        let mut ymax = 1.0_f64;
        for i in 0..t_len {
            let y = x.get(i, 0);
            if y.is_finite() {
                ymax = ymax.max(y);
            }
        }
        let n_class = (ymax.round() as usize + 1).max(2);
        if t_len == 0 {
            return ctx.finish(FittedOrdinalHmm {
                labels: empty_labels(0),
                start: init_start(k),
                trans: init_trans(k),
                mu: Vector::zeros(k),
                n_class,
                loglik: f64::NAN,
            });
        }
        let mut mu = Vector::from_iter((0..k).map(|j| j as f64 * 0.8));
        let mut start = init_start(k);
        let mut trans = init_trans(k);
        let mut loglik = f64::NEG_INFINITY;
        let mut last_gamma: Vec<Vec<f64>> = Vec::new();
        for it in 0..self.max_iter.max(1) {
            let dummy = FittedOrdinalHmm {
                labels: Vector::zeros(0),
                start: start.clone(),
                trans: trans.clone(),
                mu: mu.clone(),
                n_class,
                loglik,
            };
            let Some(fb) = scaled_forward_backward(&mut ctx, &start, &trans, &dummy.log_emit_seq(x))
            else {
                break;
            };
            loglik = fb.loglik;
            last_gamma = fb.gamma.clone();
            ctx.session.step(it as u64, -loglik, None);
            for j in 0..k {
                let mut wsum = 0.0_f64;
                let mut wy = 0.0_f64;
                for t in 0..t_len {
                    let y = x.get(t, 0);
                    if !y.is_finite() {
                        continue;
                    }
                    wsum += fb.gamma[t][j];
                    wy += fb.gamma[t][j] * y;
                }
                if wsum > 1e-12 {
                    mu[j] = wy / wsum;
                }
            }
            let (ns, ntr) = hmm_em_trans(&fb.xi, &fb.gamma[0], k, t_len);
            start = ns;
            trans = ntr;
        }
        if !last_gamma.is_empty() {
            let occup: Vec<f64> = (0..k)
                .map(|j| last_gamma.iter().map(|g| g.get(j).copied().unwrap_or(0.0)).sum())
                .collect();
            diagnose_chain(&mut ctx, &start, &trans, &occup);
        }
        let dummy = FittedOrdinalHmm {
            labels: Vector::zeros(0),
            start: start.clone(),
            trans: trans.clone(),
            mu: mu.clone(),
            n_class,
            loglik,
        };
        let (labels, _) = viterbi_path(&start, &trans, &dummy.log_emit_seq(x));
        ctx.finish(FittedOrdinalHmm {
            labels,
            start,
            trans,
            mu,
            n_class,
            loglik,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ojizou_san::Session;

    fn two_gaussian_blocks() -> Matrix {
        // 40 samples near −3, then 40 near +3 (tiny noise so variance is positive).
        Matrix::from_fn(80, 1, |i, _| {
            if i < 40 {
                -3.0 + 0.05 * ((i % 10) as f64 - 4.5)
            } else {
                3.0 + 0.05 * (((i - 40) % 10) as f64 - 4.5)
            }
        })
    }

    #[test]
    fn gaussian_hmm_learns_two_means() {
        let x = two_gaussian_blocks();
        let session = Session::new("gaussian_hmm", "fit");
        let q = GaussianHmm {
            n_states: 2,
            max_iter: 40,
            seed: 2,
            left_right: false,
        }
        .fit(&x, &session)
        .expect("hmm");
        let mut m0 = q.value.means.get(0, 0);
        let mut m1 = q.value.means.get(1, 0);
        if m0 > m1 {
            std::mem::swap(&mut m0, &mut m1);
        }
        assert!(
            (m0 + 3.0).abs() < 0.75,
            "expected a mean near -3, got {m0} and {m1}"
        );
        assert!(
            (m1 - 3.0).abs() < 0.75,
            "expected a mean near +3, got {m0} and {m1}"
        );
        let path = q
            .value
            .decode(&x, &Session::new("gaussian_hmm", "decode"))
            .expect("decode");
        assert_eq!(path.value.len(), 80);
        let sc = q
            .value
            .score(&x, &Session::new("gaussian_hmm", "score"))
            .expect("score");
        assert!(sc.value.is_finite());
        let lr = GaussianHmm::left_right(2)
            .fit(&x, &Session::new("lr_hmm", "fit"))
            .expect("lr");
        assert!(lr.value.trans.get(1, 0) <= 1e-8);
        assert!(lr.value.start[0] >= lr.value.start[1] - 1e-9);
        let glrn = GaussianHmmLeftRight::new(2)
            .fit(&x, &Session::new("glrn_hmm", "fit"))
            .expect("glrn");
        assert!(glrn.value.trans.get(1, 0) <= 1e-8);
        let xpos = Matrix::from_fn(80, 1, |i, _| x.get(i, 0) + 6.0);
        let ghm = GammaHmm::new(2)
            .fit(&xpos, &Session::new("ghm", "fit"))
            .expect("ghm");
        assert_eq!(ghm.value.labels.len(), 80);
        assert!(ghm.value.shapes.as_slice().iter().all(|v| v.is_finite() && *v > 0.0));
        let arh = AutoregressiveHmm::new(2)
            .fit(&x, &Session::new("arh", "fit"))
            .expect("arh");
        assert_eq!(arh.value.labels.len(), 80);
        assert!(arh.value.phi.as_slice().iter().all(|v| v.is_finite()));
        let sth = StudentTHmm::new(2)
            .fit(&x, &Session::new("sth", "fit"))
            .expect("sth");
        assert_eq!(sth.value.labels.len(), 80);
        assert!(sth.value.var.as_slice().iter().all(|v| v.is_finite() && *v > 0.0));
        let exh = ExponentialHmm::new(2)
            .fit(&xpos, &Session::new("exh", "fit"))
            .expect("exh");
        assert_eq!(exh.value.labels.len(), 80);
        assert!(exh.value.rates.as_slice().iter().all(|v| v.is_finite() && *v > 0.0));
        let igh = InverseGaussianHmm::new(2)
            .fit(&xpos, &Session::new("igh", "fit"))
            .expect("igh");
        assert_eq!(igh.value.labels.len(), 80);
        assert!(igh.value.lambda.as_slice().iter().all(|v| v.is_finite() && *v > 0.0));
        let crh = CircularHmm::new(2)
            .fit(&x, &Session::new("crh", "fit"))
            .expect("crh");
        assert_eq!(crh.value.labels.len(), 80);
        assert!(crh.value.kappa.as_slice().iter().all(|v| v.is_finite() && *v > 0.0));
        let skh = StickyHmm::new(2)
            .fit(&x, &Session::new("skh", "fit"))
            .expect("skh");
        assert_eq!(skh.value.labels.len(), 80);
        let hsm = Hsmm::new(2)
            .fit(&x, &Session::new("hsm", "fit"))
            .expect("hsm");
        assert_eq!(hsm.value.labels.len(), 80);
        assert!(hsm.value.rho.as_slice().iter().all(|v| v.is_finite()));
        let dirx = Matrix::from_fn(80, 2, |i, j| {
            let a = (x.get(i, 0) + 6.0).max(0.2);
            let b = (6.0 - x.get(i, 0)).max(0.2);
            let t = a + b;
            if j == 0 {
                a / t
            } else {
                b / t
            }
        });
        let dh = DirichletHmm::new(2)
            .fit(&dirx, &Session::new("dirhmm", "fit"))
            .expect("dirhmm");
        assert_eq!(dh.value.labels.len(), 80);
        assert_eq!(dh.value.alpha.shape(), (2, 2));
        let betx = Matrix::from_fn(80, 1, |i, _| {
            ((x.get(i, 0) + 6.0) / 12.0).clamp(0.02, 0.98)
        });
        let bhm = BetaHmm::new(2)
            .fit(&betx, &Session::new("bhm", "fit"))
            .expect("bhm");
        assert_eq!(bhm.value.labels.len(), 80);
        assert!(bhm.value.alpha.as_slice().iter().all(|v| v.is_finite() && *v > 0.0));
        let lnh = LogNormalHmm::new(2)
            .fit(&xpos, &Session::new("lnh", "fit"))
            .expect("lnh");
        assert_eq!(lnh.value.labels.len(), 80);
        assert!(lnh.value.var.as_slice().iter().all(|v| v.is_finite() && *v > 0.0));
        let wbh = WeibullHmm::new(2)
            .fit(&xpos, &Session::new("wbh", "fit"))
            .expect("wbh");
        assert_eq!(wbh.value.labels.len(), 80);
        assert!(wbh.value.shape.as_slice().iter().all(|v| v.is_finite() && *v > 0.0));
        let dph = DirichletProcessHmm::new(2, 1.0)
            .fit(&x, &Session::new("dph", "fit"))
            .expect("dph");
        assert_eq!(dph.value.labels.len(), 80);
        assert!(dph.value.var.as_slice().iter().all(|v| v.is_finite() && *v > 0.0));
        let hdp = StickyHdpHmm::new(2, 1.0)
            .fit(&x, &Session::new("hdp", "fit"))
            .expect("hdp");
        assert_eq!(hdp.value.labels.len(), 80);
        assert!(hdp.value.var.as_slice().iter().all(|v| v.is_finite() && *v > 0.0));
        let lrhs = LeftRightHsmm::new(2)
            .fit(&x, &Session::new("lrhs", "fit"))
            .expect("lrhs");
        assert_eq!(lrhs.value.labels.len(), 80);
        assert!(lrhs.value.trans.get(1, 0) <= 1e-8);
        let cat = Matrix::from_fn(40, 1, |i, _| if i < 20 { 0.0 } else { 1.0 });
        let brh = BernoulliHmm::new(2)
            .fit(&cat, &Session::new("brh", "fit"))
            .expect("brh");
        assert_eq!(brh.value.labels.len(), 40);
        assert!(brh.value.p.as_slice().iter().all(|v| v.is_finite() && *v > 0.0 && *v < 1.0));
        let orh = OrdinalHmm::new(2)
            .fit(&cat, &Session::new("orh", "fit"))
            .expect("orh");
        assert_eq!(orh.value.labels.len(), 40);
        assert!(orh.value.mu.as_slice().iter().all(|v| v.is_finite()));
        let mlr = MultinomialHmm::left_right(2)
            .fit(&cat, &Session::new("mlr_hmm", "fit"))
            .expect("mlr");
        assert!(mlr.value.trans.get(1, 0) <= 1e-8);
        let mlrn = MultinomialHmmLeftRight::new(2)
            .fit(&cat, &Session::new("mlrn_hmm", "fit"))
            .expect("mlrn");
        assert!(mlrn.value.trans.get(1, 0) <= 1e-8);
        let glr = GmmHmm::left_right(2, 1)
            .fit(&x, &Session::new("glr_hmm", "fit"))
            .expect("glr");
        assert!(glr.value.trans.get(1, 0) <= 1e-8);
        let gmmlr = GmmHmmLeftRight::new(2, 1)
            .fit(&x, &Session::new("gmmlr_hmm", "fit"))
            .expect("gmmlr");
        assert!(gmmlr.value.trans.get(1, 0) <= 1e-8);
        let gsph = GmmHmm::spherical(2, 1)
            .fit(&x, &Session::new("gsph_hmm", "fit"))
            .expect("gsph");
        assert_eq!(gsph.value.vars.ncols(), 1);
        assert!(gsph.value.vars.get(0, 0).is_finite());
        let full = GaussianHmmFull::new(2)
            .fit(&x, &Session::new("full_hmm", "fit"))
            .expect("full");
        let mut fm0 = full.value.means.get(0, 0);
        let mut fm1 = full.value.means.get(1, 0);
        if fm0 > fm1 {
            std::mem::swap(&mut fm0, &mut fm1);
        }
        assert!(
            (fm0 + 3.0).abs() < 0.75,
            "full-cov expected a mean near -3, got {fm0} and {fm1}"
        );
        assert!(
            (fm1 - 3.0).abs() < 0.75,
            "full-cov expected a mean near +3, got {fm0} and {fm1}"
        );
        assert_eq!(full.value.covs.len(), 2);
        assert_eq!(full.value.covs[0].shape(), (1, 1));
        let ann = HmmAnnotator::new(2)
            .fit_unsupervised(&x, &Session::new("ann_hmm", "fit"))
            .expect("ann");
        assert_eq!(ann.value.labels.len(), 80);
        let sph = GaussianHmmSpherical::new(2)
            .fit(&x, &Session::new("sph_hmm", "fit"))
            .expect("sph");
        let mut sm0 = sph.value.means.get(0, 0);
        let mut sm1 = sph.value.means.get(1, 0);
        if sm0 > sm1 {
            std::mem::swap(&mut sm0, &mut sm1);
        }
        assert!(
            (sm0 + 3.0).abs() < 0.75,
            "spherical expected a mean near -3, got {sm0} and {sm1}"
        );
        assert_eq!(sph.value.vars.len(), 2);
        let tied = GaussianHmmTied::new(2)
            .fit(&x, &Session::new("tied_hmm", "fit"))
            .expect("tied");
        let mut tm0 = tied.value.means.get(0, 0);
        let mut tm1 = tied.value.means.get(1, 0);
        if tm0 > tm1 {
            std::mem::swap(&mut tm0, &mut tm1);
        }
        assert!(
            (tm0 + 3.0).abs() < 0.75,
            "tied expected a mean near -3, got {tm0} and {tm1}"
        );
        assert_eq!(tied.value.cov.shape(), (1, 1));
        let gfull = GmmHmmFull::new(2, 1)
            .fit(&x, &Session::new("gfull_hmm", "fit"))
            .expect("gfull");
        let mut gfm0 = gfull.value.means.get(0, 0);
        let mut gfm1 = gfull.value.means.get(1, 0);
        if gfm0 > gfm1 {
            std::mem::swap(&mut gfm0, &mut gfm1);
        }
        assert!(
            (gfm0 + 3.0).abs() < 0.75,
            "GmmHmmFull expected a mean near -3, got {gfm0} and {gfm1}"
        );
        assert!(
            (gfm1 - 3.0).abs() < 0.75,
            "GmmHmmFull expected a mean near +3, got {gfm0} and {gfm1}"
        );
        assert_eq!(gfull.value.covs.len(), 2);
        assert_eq!(gfull.value.covs[0].shape(), (1, 1));
        let gtied = GmmHmmTied::new(2, 1)
            .fit(&x, &Session::new("gtied_hmm", "fit"))
            .expect("gtied");
        let mut gtm0 = gtied.value.means.get(0, 0);
        let mut gtm1 = gtied.value.means.get(1, 0);
        if gtm0 > gtm1 {
            std::mem::swap(&mut gtm0, &mut gtm1);
        }
        assert!(
            (gtm0 + 3.0).abs() < 0.75,
            "GmmHmmTied expected a mean near -3, got {gtm0} and {gtm1}"
        );
        assert_eq!(gtied.value.covs.len(), 2);
        assert_eq!(gtied.value.covs[0].shape(), (1, 1));
        let vg = VariationalGmmHmm::new(2, 1)
            .fit(&x, &Session::new("vgmm_hmm", "fit"))
            .expect("vgmm");
        let mut vm0 = vg.value.means.get(0, 0);
        let mut vm1 = vg.value.means.get(1, 0);
        if vm0 > vm1 {
            std::mem::swap(&mut vm0, &mut vm1);
        }
        assert!(
            (vm0 + 3.0).abs() < 1.2,
            "variational GMM-HMM expected a mean near -3, got {vm0} and {vm1}"
        );
        assert!(
            (vm1 - 3.0).abs() < 1.2,
            "variational GMM-HMM expected a mean near +3, got {vm0} and {vm1}"
        );
        assert_eq!(vg.value.labels.len(), 80);
    }

    #[test]
    fn gaussian_hmm_zero_variance_is_degenerate() {
        let x = Matrix::from_fn(24, 1, |_, _| 2.5);
        let session = Session::new("gaussian_hmm", "fit");
        let err = GaussianHmm {
            n_states: 2,
            max_iter: 5,
            seed: 0,
            left_right: false,
        }
        .fit(&x, &session)
        .unwrap_err();
        assert_eq!(err.primary().code, IssueCode::EmissionDegenerate);
    }

    #[test]
    fn poisson_hmm_two_rates() {
        let x = Matrix::from_fn(40, 1, |i, _| if i < 20 { 1.0 } else { 8.0 });
        let q = PoissonHmm::new(2)
            .fit(&x, &Session::new("phmm", "fit"))
            .expect("phmm");
        let mut rates = q.value.rates.as_slice().to_vec();
        rates.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!(rates[0] < 3.0 && rates[1] > 5.0, "{rates:?}");
        let sc = q
            .value
            .score(&x, &Session::new("phmm", "score"))
            .expect("score");
        assert!(sc.value.is_finite());
        let lr = PoissonHmm::left_right(2)
            .fit(&x, &Session::new("phmm_lr", "fit"))
            .expect("phmm_lr");
        assert!(lr.value.trans.get(1, 0) <= 1e-8);
        assert_eq!(lr.value.rates.len(), 2);
        let vb = VariationalPoissonHmm::new(2)
            .fit(&x, &Session::new("vphmm", "fit"))
            .expect("vphmm");
        let mut vr = vb.value.rates.as_slice().to_vec();
        vr.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!(vr[0] < 3.0 && vr[1] > 5.0, "{vr:?}");
        assert_eq!(vb.value.labels.len(), 40);
        let nbh = NegativeBinomialHmm::new(2)
            .fit(&x, &Session::new("nbhmm", "fit"))
            .expect("nbhmm");
        assert_eq!(nbh.value.labels.len(), 40);
        assert!(nbh.value.r.as_slice().iter().all(|v| v.is_finite() && *v > 0.0));
        let zip = ZeroInflatedPoissonHmm::new(2)
            .fit(&x, &Session::new("ziphmm", "fit"))
            .expect("ziphmm");
        assert_eq!(zip.value.labels.len(), 40);
        assert!(zip.value.lam.as_slice().iter().all(|v| v.is_finite() && *v > 0.0));
        let bnh = BinomialHmm::new(2, 10.0)
            .fit(&x, &Session::new("bnh", "fit"))
            .expect("bnh");
        assert_eq!(bnh.value.labels.len(), 40);
        assert!(bnh.value.p.as_slice().iter().all(|v| v.is_finite() && *v > 0.0 && *v < 1.0));
        let geh = GeometricHmm::new(2)
            .fit(&x, &Session::new("geh", "fit"))
            .expect("geh");
        assert_eq!(geh.value.labels.len(), 40);
        assert!(geh.value.p.as_slice().iter().all(|v| v.is_finite() && *v > 0.0 && *v < 1.0));
        let bbh = BetaBinomialHmm::new(2, 10.0)
            .fit(&x, &Session::new("bbh", "fit"))
            .expect("bbh");
        assert_eq!(bbh.value.labels.len(), 40);
        assert!(bbh.value.alpha.as_slice().iter().all(|v| v.is_finite() && *v > 0.0));
    }

    #[test]
    fn variational_hmm_two_means() {
        let x = two_gaussian_blocks();
        let q = VariationalGaussianHmm {
            n_states: 2,
            max_iter: 30,
            seed: 4,
        }
        .fit(&x, &Session::new("vbhmm", "fit"))
        .expect("vb");
        let mut m0 = q.value.means.get(0, 0);
        let mut m1 = q.value.means.get(1, 0);
        if m0 > m1 {
            std::mem::swap(&mut m0, &mut m1);
        }
        assert!(
            (m0 + 3.0).abs() < 1.2,
            "expected a mean near -3, got {m0} and {m1}"
        );
        assert!(
            (m1 - 3.0).abs() < 1.2,
            "expected a mean near +3, got {m0} and {m1}"
        );
    }

    #[test]
    fn variational_categorical_two_symbols() {
        let x = Matrix::from_fn(40, 1, |i, _| if i < 20 { 0.0 } else { 1.0 });
        let q = VariationalCategoricalHmm {
            n_states: 2,
            max_iter: 25,
        }
        .fit(&x, &Session::new("vchmm", "fit"))
        .expect("vchmm");
        assert_eq!(q.value.emission.nrows(), 2);
        assert_eq!(q.value.labels.len(), 40);
        assert!(q.value.elbo.is_finite() || q.value.elbo.is_infinite());
    }
}
