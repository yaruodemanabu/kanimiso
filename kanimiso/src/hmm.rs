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

fn log_laplace(y: f64, mu: f64, scale: f64) -> f64 {
    if !y.is_finite() || !mu.is_finite() || scale <= 0.0 {
        return f64::NEG_INFINITY;
    }
    -scale.ln() - std::f64::consts::LN_2 - (y - mu).abs() / scale
}

fn log_pareto(y: f64, xmin: f64, alpha: f64) -> f64 {
    if !y.is_finite() || y < xmin || xmin <= 0.0 || alpha <= 0.0 {
        return f64::NEG_INFINITY;
    }
    alpha.ln() + alpha * xmin.ln() - (alpha + 1.0) * y.ln()
}

fn log_logit_normal(y: f64, mu: f64, var: f64) -> f64 {
    if !y.is_finite() || y <= 0.0 || y >= 1.0 || var <= 0.0 {
        return f64::NEG_INFINITY;
    }
    let z = (y / (1.0 - y)).ln();
    log_gauss1(z, mu, var) - y.ln() - (1.0 - y).ln()
}

fn ln_choose(n: f64, k: f64) -> f64 {
    if !n.is_finite() || !k.is_finite() || k < 0.0 || k > n {
        return f64::NEG_INFINITY;
    }
    crate::special::ln_gamma(n + 1.0) - crate::special::ln_gamma(k + 1.0)
        - crate::special::ln_gamma(n - k + 1.0)
}

fn log_hyper(y: f64, n_pop: f64, k_succ: f64, n_draw: f64) -> f64 {
    if !y.is_finite() || n_pop < 1.0 || n_draw < 1.0 || k_succ < 0.0 {
        return f64::NEG_INFINITY;
    }
    let y = y.round();
    ln_choose(k_succ, y) + ln_choose(n_pop - k_succ, n_draw - y) - ln_choose(n_pop, n_draw)
}

fn log_zinb(y: f64, pi: f64, r: f64, p: f64) -> f64 {
    if !y.is_finite() || y < 0.0 || r <= 0.0 || p <= 0.0 || p >= 1.0 {
        return f64::NEG_INFINITY;
    }
    let pi = pi.clamp(1e-6, 1.0 - 1e-6);
    if y.abs() < 1e-12 {
        logsumexp(&[pi.ln(), (1.0 - pi).ln() + log_negbin(0.0, r, p)])
    } else {
        (1.0 - pi).ln() + log_negbin(y, r, p)
    }
}

/// Laplace-emission HMM (double exponential).
///
/// State count is not identification `p`. Distinct from [`GaussianHmm`]
/// (quadratic tails) and [`StudentTHmm`] (degrees of freedom).
#[derive(Clone, Debug)]
pub struct LaplaceHmm {
    /// Hidden states. Not identification `p`.
    pub n_states: usize,
    /// Baum–Welch cap.
    pub max_iter: usize,
}

impl Default for LaplaceHmm {
    fn default() -> Self {
        Self {
            n_states: 2,
            max_iter: 40,
        }
    }
}

impl LaplaceHmm {
    /// `k`-state Laplace HMM.
    pub fn new(n_states: usize) -> Self {
        Self {
            n_states,
            ..Self::default()
        }
    }

    /// Fit alias.
    pub fn fit(&mut self, x: &Matrix, session: &Session) -> Result<Qualified<FittedLaplaceHmm>> {
        self.fit_unsupervised(x, session)
    }
}

/// Fitted Laplace HMM.
#[derive(Clone, Debug)]
pub struct FittedLaplaceHmm {
    /// Viterbi path.
    pub labels: Vector,
    /// Start distribution.
    pub start: Vector,
    /// Transitions.
    pub trans: Matrix,
    /// Locations \(\mu_j\).
    pub mu: Vector,
    /// Scales \(b_j\).
    pub scale: Vector,
    /// Training log-likelihood.
    pub loglik: f64,
}

impl FittedLaplaceHmm {
    fn log_emit_seq(&self, x: &Matrix) -> Vec<Vec<f64>> {
        let t = x.nrows();
        let s = self.mu.len();
        let mut out = vec![vec![f64::NEG_INFINITY; s]; t];
        for ti in 0..t {
            let y = x.get(ti, 0);
            for j in 0..s {
                out[ti][j] = log_laplace(y, self.mu[j], self.scale[j]);
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

impl Predict for FittedLaplaceHmm {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        self.decode(x, session)
    }
}

impl FitUnsupervised for LaplaceHmm {
    type Fitted = FittedLaplaceHmm;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedLaplaceHmm>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let t_len = x.nrows();
        let k = self.n_states.max(1);
        if t_len == 0 {
            return ctx.finish(FittedLaplaceHmm {
                labels: empty_labels(0),
                start: init_start(k),
                trans: init_trans(k),
                mu: Vector::zeros(k),
                scale: Vector::filled(k, 1.0),
                loglik: f64::NAN,
            });
        }
        let mean = x.column(0).mean();
        let sd = x.column(0).std().max(0.1);
        let mut mu = Vector::from_iter((0..k).map(|j| mean + (j as f64 - 0.5) * sd));
        let mut scale = Vector::filled(k, sd / std::f64::consts::SQRT_2);
        let mut start = init_start(k);
        let mut trans = init_trans(k);
        let mut loglik = f64::NEG_INFINITY;
        let mut last_gamma: Vec<Vec<f64>> = Vec::new();
        for it in 0..self.max_iter.max(1) {
            let dummy = FittedLaplaceHmm {
                labels: Vector::zeros(0),
                start: start.clone(),
                trans: trans.clone(),
                mu: mu.clone(),
                scale: scale.clone(),
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
                let mut wad = 0.0_f64;
                for t in 0..t_len {
                    let y = x.get(t, 0);
                    if !y.is_finite() {
                        continue;
                    }
                    let w = fb.gamma[t][j];
                    wsum += w;
                    wy += w * y;
                }
                if wsum > 1e-12 {
                    mu[j] = wy / wsum;
                    for t in 0..t_len {
                        let y = x.get(t, 0);
                        if y.is_finite() {
                            wad += fb.gamma[t][j] * (y - mu[j]).abs();
                        }
                    }
                    scale[j] = (wad / wsum).max(COV_FLOOR.sqrt());
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
        let dummy = FittedLaplaceHmm {
            labels: Vector::zeros(0),
            start: start.clone(),
            trans: trans.clone(),
            mu: mu.clone(),
            scale: scale.clone(),
            loglik,
        };
        let (labels, _) = viterbi_path(&start, &trans, &dummy.log_emit_seq(x));
        ctx.finish(FittedLaplaceHmm {
            labels,
            start,
            trans,
            mu,
            scale,
            loglik,
        })
    }
}

/// Pareto-emission HMM (power-law tails).
///
/// Minimum \(x_m\) is not identification `p`. Distinct from [`ExponentialHmm`]
/// (memoryless) and [`WeibullHmm`] (shape/scale).
#[derive(Clone, Debug)]
pub struct ParetoHmm {
    /// Hidden states. Not identification `p`.
    pub n_states: usize,
    /// Lower endpoint. Not identification `p`.
    pub xmin: f64,
    /// Baum–Welch cap.
    pub max_iter: usize,
}

impl Default for ParetoHmm {
    fn default() -> Self {
        Self {
            n_states: 2,
            xmin: 1.0,
            max_iter: 40,
        }
    }
}

impl ParetoHmm {
    /// `k`-state Pareto HMM with lower endpoint `xmin`.
    pub fn new(n_states: usize, xmin: f64) -> Self {
        Self {
            n_states,
            xmin,
            ..Self::default()
        }
    }

    /// Fit alias.
    pub fn fit(&mut self, x: &Matrix, session: &Session) -> Result<Qualified<FittedParetoHmm>> {
        self.fit_unsupervised(x, session)
    }
}

/// Fitted Pareto HMM.
#[derive(Clone, Debug)]
pub struct FittedParetoHmm {
    /// Viterbi path.
    pub labels: Vector,
    /// Start distribution.
    pub start: Vector,
    /// Transitions.
    pub trans: Matrix,
    /// Shape \(\alpha_j\).
    pub alpha: Vector,
    /// Shared lower endpoint.
    pub xmin: f64,
    /// Training log-likelihood.
    pub loglik: f64,
}

impl FittedParetoHmm {
    fn log_emit_seq(&self, x: &Matrix) -> Vec<Vec<f64>> {
        let t = x.nrows();
        let s = self.alpha.len();
        let mut out = vec![vec![f64::NEG_INFINITY; s]; t];
        for ti in 0..t {
            let y = x.get(ti, 0);
            for j in 0..s {
                out[ti][j] = log_pareto(y, self.xmin, self.alpha[j]);
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

impl Predict for FittedParetoHmm {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        self.decode(x, session)
    }
}

impl FitUnsupervised for ParetoHmm {
    type Fitted = FittedParetoHmm;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedParetoHmm>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let t_len = x.nrows();
        let k = self.n_states.max(1);
        let mut xm = if self.xmin.is_finite() && self.xmin > 0.0 {
            self.xmin
        } else {
            1.0
        };
        let mut n_skip = 0usize;
        for i in 0..t_len {
            let y = x.get(i, 0);
            if y.is_finite() && y > 0.0 {
                xm = xm.min(y);
            } else if y.is_finite() {
                n_skip += 1;
            }
        }
        if n_skip > 0 {
            ctx.push(
                Issue::builder(IssueCode::NonPositiveSeries)
                    .severity(signlred::Severity::Warning)
                    .message(format!("ParetoHmm skipped {n_skip} non-positive observations"))
                    .build(),
            );
        }
        if t_len == 0 {
            return ctx.finish(FittedParetoHmm {
                labels: empty_labels(0),
                start: init_start(k),
                trans: init_trans(k),
                alpha: Vector::filled(k, 1.5),
                xmin: xm,
                loglik: f64::NAN,
            });
        }
        let mut alpha = Vector::from_iter((0..k).map(|j| 1.0 + 0.5 * j as f64));
        let mut start = init_start(k);
        let mut trans = init_trans(k);
        let mut loglik = f64::NEG_INFINITY;
        let mut last_gamma: Vec<Vec<f64>> = Vec::new();
        for it in 0..self.max_iter.max(1) {
            let dummy = FittedParetoHmm {
                labels: Vector::zeros(0),
                start: start.clone(),
                trans: trans.clone(),
                alpha: alpha.clone(),
                xmin: xm,
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
                let mut wlog = 0.0_f64;
                for t in 0..t_len {
                    let y = x.get(t, 0);
                    if !y.is_finite() || y < xm {
                        continue;
                    }
                    let w = fb.gamma[t][j];
                    wsum += w;
                    wlog += w * (y / xm).ln();
                }
                if wsum > 1e-12 && wlog > 1e-12 {
                    alpha[j] = (wsum / wlog).clamp(0.05, 80.0);
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
        let dummy = FittedParetoHmm {
            labels: Vector::zeros(0),
            start: start.clone(),
            trans: trans.clone(),
            alpha: alpha.clone(),
            xmin: xm,
            loglik,
        };
        let (labels, _) = viterbi_path(&start, &trans, &dummy.log_emit_seq(x));
        ctx.finish(FittedParetoHmm {
            labels,
            start,
            trans,
            alpha,
            xmin: xm,
            loglik,
        })
    }
}

/// Logit-normal emission HMM (Gaussian on \(\mathrm{logit}(y)\)).
///
/// State count is not identification `p`. Distinct from [`BetaHmm`]
/// (moment \(\alpha,\beta\)) and [`LogNormalHmm`] (positive support).
#[derive(Clone, Debug)]
pub struct LogitNormalHmm {
    /// Hidden states. Not identification `p`.
    pub n_states: usize,
    /// Baum–Welch cap.
    pub max_iter: usize,
}

impl Default for LogitNormalHmm {
    fn default() -> Self {
        Self {
            n_states: 2,
            max_iter: 40,
        }
    }
}

impl LogitNormalHmm {
    /// `k`-state logit-normal HMM.
    pub fn new(n_states: usize) -> Self {
        Self {
            n_states,
            ..Self::default()
        }
    }

    /// Fit alias.
    pub fn fit(&mut self, x: &Matrix, session: &Session) -> Result<Qualified<FittedLogitNormalHmm>> {
        self.fit_unsupervised(x, session)
    }
}

/// Fitted logit-normal HMM.
#[derive(Clone, Debug)]
pub struct FittedLogitNormalHmm {
    /// Viterbi path.
    pub labels: Vector,
    /// Start distribution.
    pub start: Vector,
    /// Transitions.
    pub trans: Matrix,
    /// Mean of \(\mathrm{logit}(y)\).
    pub mu: Vector,
    /// Variance of \(\mathrm{logit}(y)\).
    pub var: Vector,
    /// Training log-likelihood.
    pub loglik: f64,
}

impl FittedLogitNormalHmm {
    fn log_emit_seq(&self, x: &Matrix) -> Vec<Vec<f64>> {
        let t = x.nrows();
        let s = self.mu.len();
        let mut out = vec![vec![f64::NEG_INFINITY; s]; t];
        for ti in 0..t {
            let y = x.get(ti, 0);
            for j in 0..s {
                out[ti][j] = log_logit_normal(y, self.mu[j], self.var[j]);
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

impl Predict for FittedLogitNormalHmm {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        self.decode(x, session)
    }
}

impl FitUnsupervised for LogitNormalHmm {
    type Fitted = FittedLogitNormalHmm;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedLogitNormalHmm>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let t_len = x.nrows();
        let k = self.n_states.max(1);
        if t_len == 0 {
            return ctx.finish(FittedLogitNormalHmm {
                labels: empty_labels(0),
                start: init_start(k),
                trans: init_trans(k),
                mu: Vector::zeros(k),
                var: Vector::filled(k, 1.0),
                loglik: f64::NAN,
            });
        }
        let mut mu = Vector::from_iter((0..k).map(|j| -0.4 + 0.8 * j as f64));
        let mut var = Vector::filled(k, 0.5);
        let mut start = init_start(k);
        let mut trans = init_trans(k);
        let mut loglik = f64::NEG_INFINITY;
        let mut last_gamma: Vec<Vec<f64>> = Vec::new();
        for it in 0..self.max_iter.max(1) {
            let dummy = FittedLogitNormalHmm {
                labels: Vector::zeros(0),
                start: start.clone(),
                trans: trans.clone(),
                mu: mu.clone(),
                var: var.clone(),
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
                let mut wz = 0.0_f64;
                let mut wz2 = 0.0_f64;
                for t in 0..t_len {
                    let y = x.get(t, 0);
                    if !y.is_finite() || y <= 0.0 || y >= 1.0 {
                        continue;
                    }
                    let z = (y / (1.0 - y)).ln();
                    let w = fb.gamma[t][j];
                    wsum += w;
                    wz += w * z;
                    wz2 += w * z * z;
                }
                if wsum > 1e-12 {
                    mu[j] = wz / wsum;
                    var[j] = (wz2 / wsum - mu[j] * mu[j]).max(COV_FLOOR);
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
        let dummy = FittedLogitNormalHmm {
            labels: Vector::zeros(0),
            start: start.clone(),
            trans: trans.clone(),
            mu: mu.clone(),
            var: var.clone(),
            loglik,
        };
        let (labels, _) = viterbi_path(&start, &trans, &dummy.log_emit_seq(x));
        ctx.finish(FittedLogitNormalHmm {
            labels,
            start,
            trans,
            mu,
            var,
            loglik,
        })
    }
}

/// Hypergeometric-emission HMM (finite-population draws).
///
/// Population and draw size are not identification `p`. Distinct from
/// [`BinomialHmm`] (with-replacement / infinite population) and
/// [`PoissonHmm`] (unbounded).
#[derive(Clone, Debug)]
pub struct HypergeometricHmm {
    /// Hidden states. Not identification `p`.
    pub n_states: usize,
    /// Population size \(N\). Not identification `p`.
    pub n_pop: f64,
    /// Draw size \(n\). Not identification `p`.
    pub n_draw: f64,
    /// Baum–Welch cap.
    pub max_iter: usize,
}

impl Default for HypergeometricHmm {
    fn default() -> Self {
        Self {
            n_states: 2,
            n_pop: 20.0,
            n_draw: 10.0,
            max_iter: 40,
        }
    }
}

impl HypergeometricHmm {
    /// `k`-state hypergeometric HMM.
    pub fn new(n_states: usize, n_pop: f64, n_draw: f64) -> Self {
        Self {
            n_states,
            n_pop,
            n_draw,
            ..Self::default()
        }
    }

    /// Fit alias.
    pub fn fit(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedHypergeometricHmm>> {
        self.fit_unsupervised(x, session)
    }
}

/// Fitted hypergeometric HMM.
#[derive(Clone, Debug)]
pub struct FittedHypergeometricHmm {
    /// Viterbi path.
    pub labels: Vector,
    /// Start distribution.
    pub start: Vector,
    /// Transitions.
    pub trans: Matrix,
    /// Success-state counts \(K_j\).
    pub k_succ: Vector,
    /// Population size.
    pub n_pop: f64,
    /// Draw size.
    pub n_draw: f64,
    /// Training log-likelihood.
    pub loglik: f64,
}

impl FittedHypergeometricHmm {
    fn log_emit_seq(&self, x: &Matrix) -> Vec<Vec<f64>> {
        let t = x.nrows();
        let s = self.k_succ.len();
        let mut out = vec![vec![f64::NEG_INFINITY; s]; t];
        for ti in 0..t {
            let y = x.get(ti, 0);
            for j in 0..s {
                out[ti][j] = log_hyper(y, self.n_pop, self.k_succ[j], self.n_draw);
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

impl Predict for FittedHypergeometricHmm {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        self.decode(x, session)
    }
}

impl FitUnsupervised for HypergeometricHmm {
    type Fitted = FittedHypergeometricHmm;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedHypergeometricHmm>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let t_len = x.nrows();
        let k = self.n_states.max(1);
        let mut np = if self.n_pop.is_finite() && self.n_pop >= 2.0 {
            self.n_pop
        } else {
            20.0
        };
        let nd = if self.n_draw.is_finite() && self.n_draw >= 1.0 && self.n_draw <= np {
            self.n_draw
        } else {
            np.min(10.0)
        };
        let mut ymin = f64::INFINITY;
        let mut ymax = 0.0_f64;
        for i in 0..t_len {
            let y = x.get(i, 0);
            if y.is_finite() && y >= 0.0 {
                ymin = ymin.min(y);
                ymax = ymax.max(y);
            }
        }
        if !ymin.is_finite() {
            ymin = 0.0;
        }
        // Every observed y must be feasible under every K: ymax ≤ K ≤ N − n + ymin.
        let need = ymax - ymin + nd;
        if np < need {
            np = need;
        }
        let k_lo = ymax.clamp(0.0, np);
        let k_hi = (np - nd + ymin).clamp(k_lo, np);
        if t_len == 0 {
            return ctx.finish(FittedHypergeometricHmm {
                labels: empty_labels(0),
                start: init_start(k),
                trans: init_trans(k),
                k_succ: Vector::filled(k, nd * 0.5),
                n_pop: np,
                n_draw: nd,
                loglik: f64::NAN,
            });
        }
        let mut k_succ = Vector::from_iter((0..k).map(|j| {
            (nd * (0.25 + 0.5 * j as f64 / k.max(1) as f64)).clamp(k_lo, k_hi)
        }));
        let mut start = init_start(k);
        let mut trans = init_trans(k);
        let mut loglik = f64::NEG_INFINITY;
        let mut last_gamma: Vec<Vec<f64>> = Vec::new();
        for it in 0..self.max_iter.max(1) {
            let dummy = FittedHypergeometricHmm {
                labels: Vector::zeros(0),
                start: start.clone(),
                trans: trans.clone(),
                k_succ: k_succ.clone(),
                n_pop: np,
                n_draw: nd,
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
                    k_succ[j] = (mu * np / nd.max(1e-8)).clamp(k_lo, k_hi);
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
        let dummy = FittedHypergeometricHmm {
            labels: Vector::zeros(0),
            start: start.clone(),
            trans: trans.clone(),
            k_succ: k_succ.clone(),
            n_pop: np,
            n_draw: nd,
            loglik,
        };
        let (labels, _) = viterbi_path(&start, &trans, &dummy.log_emit_seq(x));
        ctx.finish(FittedHypergeometricHmm {
            labels,
            start,
            trans,
            k_succ,
            n_pop: np,
            n_draw: nd,
            loglik,
        })
    }
}

/// Zero-inflated negative-binomial HMM.
///
/// State count is not identification `p`. Distinct from
/// [`ZeroInflatedPoissonHmm`] (no overdispersion) and
/// [`NegativeBinomialHmm`] (no extra zeros).
#[derive(Clone, Debug)]
pub struct ZeroInflatedNegBinHmm {
    /// Hidden states. Not identification `p`.
    pub n_states: usize,
    /// Baum–Welch cap.
    pub max_iter: usize,
}

impl Default for ZeroInflatedNegBinHmm {
    fn default() -> Self {
        Self {
            n_states: 2,
            max_iter: 40,
        }
    }
}

impl ZeroInflatedNegBinHmm {
    /// `k`-state ZINB HMM.
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
    ) -> Result<Qualified<FittedZeroInflatedNegBinHmm>> {
        self.fit_unsupervised(x, session)
    }
}

/// Fitted ZINB HMM.
#[derive(Clone, Debug)]
pub struct FittedZeroInflatedNegBinHmm {
    /// Viterbi path.
    pub labels: Vector,
    /// Start distribution.
    pub start: Vector,
    /// Transitions.
    pub trans: Matrix,
    /// Extra-zero mass.
    pub pi: Vector,
    /// Success counts \(r_j\).
    pub r: Vector,
    /// Success probability \(p_j\).
    pub p: Vector,
    /// Training log-likelihood.
    pub loglik: f64,
}

impl FittedZeroInflatedNegBinHmm {
    fn log_emit_seq(&self, x: &Matrix) -> Vec<Vec<f64>> {
        let t = x.nrows();
        let s = self.r.len();
        let mut out = vec![vec![f64::NEG_INFINITY; s]; t];
        for ti in 0..t {
            let y = x.get(ti, 0);
            for j in 0..s {
                out[ti][j] = log_zinb(y, self.pi[j], self.r[j], self.p[j]);
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

impl Predict for FittedZeroInflatedNegBinHmm {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        self.decode(x, session)
    }
}

impl FitUnsupervised for ZeroInflatedNegBinHmm {
    type Fitted = FittedZeroInflatedNegBinHmm;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedZeroInflatedNegBinHmm>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let t_len = x.nrows();
        let k = self.n_states.max(1);
        if t_len == 0 {
            return ctx.finish(FittedZeroInflatedNegBinHmm {
                labels: empty_labels(0),
                start: init_start(k),
                trans: init_trans(k),
                pi: Vector::filled(k, 0.1),
                r: Vector::filled(k, 1.0),
                p: Vector::filled(k, 0.5),
                loglik: f64::NAN,
            });
        }
        let mean = x
            .column(0)
            .as_slice()
            .iter()
            .filter(|v| **v >= 0.0)
            .sum::<f64>()
            / t_len.max(1) as f64;
        let mut pi = Vector::filled(k, 0.15);
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
            let dummy = FittedZeroInflatedNegBinHmm {
                labels: Vector::zeros(0),
                start: start.clone(),
                trans: trans.clone(),
                pi: pi.clone(),
                r: r.clone(),
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
                let mut wzero = 0.0_f64;
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
                    if y.abs() < 1e-12 {
                        wzero += w;
                    }
                }
                if wsum > 1e-12 {
                    let mu = (wy / wsum).max(1e-4);
                    let var = (wy2 / wsum - mu * mu).max(mu + 1e-4);
                    let zf = (wzero / wsum).clamp(0.0, 0.95);
                    let nb0 = log_negbin(0.0, r[j], p[j]).exp();
                    pi[j] = ((zf - nb0) / (1.0 - nb0).max(1e-6)).clamp(1e-4, 0.9);
                    let mu_c = (mu / (1.0 - pi[j]).max(1e-3)).max(0.05);
                    r[j] = (mu_c * mu_c / (var - mu).max(1e-4)).clamp(0.05, 80.0);
                    p[j] = (r[j] / (r[j] + mu_c)).clamp(0.02, 0.98);
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
        let dummy = FittedZeroInflatedNegBinHmm {
            labels: Vector::zeros(0),
            start: start.clone(),
            trans: trans.clone(),
            pi: pi.clone(),
            r: r.clone(),
            p: p.clone(),
            loglik,
        };
        let (labels, _) = viterbi_path(&start, &trans, &dummy.log_emit_seq(x));
        ctx.finish(FittedZeroInflatedNegBinHmm {
            labels,
            start,
            trans,
            pi,
            r,
            p,
            loglik,
        })
    }
}

fn log_invgamma(y: f64, shape: f64, scale: f64) -> f64 {
    if !y.is_finite() || y <= 0.0 || shape <= 0.0 || scale <= 0.0 {
        return f64::NEG_INFINITY;
    }
    shape * scale.ln() - crate::special::ln_gamma(shape) - (shape + 1.0) * y.ln() - scale / y
}

fn log_gumbel(y: f64, loc: f64, scale: f64) -> f64 {
    if !y.is_finite() || !loc.is_finite() || scale <= 0.0 {
        return f64::NEG_INFINITY;
    }
    let z = (y - loc) / scale;
    -scale.ln() - z - (-z).exp()
}

fn log_wrapped_cauchy(y: f64, mu: f64, rho: f64) -> f64 {
    if !y.is_finite() || !mu.is_finite() || rho <= 0.0 || rho >= 1.0 {
        return f64::NEG_INFINITY;
    }
    let den = 1.0 + rho * rho - 2.0 * rho * (y - mu).cos();
    if den <= 0.0 {
        return f64::NEG_INFINITY;
    }
    (1.0 - rho * rho).ln() - LN_2PI - den.ln()
}

/// Inverse-gamma emission HMM.
///
/// State count is not identification `p`. Distinct from [`GammaHmm`]
/// (shape/rate on \(y\)) and [`InverseGaussianHmm`] (Wald).
#[derive(Clone, Debug)]
pub struct InverseGammaHmm {
    /// Hidden states. Not identification `p`.
    pub n_states: usize,
    /// Baum–Welch cap.
    pub max_iter: usize,
}

impl Default for InverseGammaHmm {
    fn default() -> Self {
        Self {
            n_states: 2,
            max_iter: 40,
        }
    }
}

impl InverseGammaHmm {
    /// `k`-state inverse-gamma HMM.
    pub fn new(n_states: usize) -> Self {
        Self {
            n_states,
            ..Self::default()
        }
    }

    /// Fit alias.
    pub fn fit(&mut self, x: &Matrix, session: &Session) -> Result<Qualified<FittedInverseGammaHmm>> {
        self.fit_unsupervised(x, session)
    }
}

/// Fitted inverse-gamma HMM.
#[derive(Clone, Debug)]
pub struct FittedInverseGammaHmm {
    /// Viterbi path.
    pub labels: Vector,
    /// Start distribution.
    pub start: Vector,
    /// Transitions.
    pub trans: Matrix,
    /// Shape \(\alpha_j\).
    pub shape: Vector,
    /// Scale \(\beta_j\).
    pub scale: Vector,
    /// Training log-likelihood.
    pub loglik: f64,
}

impl FittedInverseGammaHmm {
    fn log_emit_seq(&self, x: &Matrix) -> Vec<Vec<f64>> {
        let t = x.nrows();
        let s = self.shape.len();
        let mut out = vec![vec![f64::NEG_INFINITY; s]; t];
        for ti in 0..t {
            let y = x.get(ti, 0);
            for j in 0..s {
                out[ti][j] = log_invgamma(y, self.shape[j], self.scale[j]);
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

impl Predict for FittedInverseGammaHmm {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        self.decode(x, session)
    }
}

impl FitUnsupervised for InverseGammaHmm {
    type Fitted = FittedInverseGammaHmm;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedInverseGammaHmm>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let t_len = x.nrows();
        let k = self.n_states.max(1);
        if t_len == 0 {
            return ctx.finish(FittedInverseGammaHmm {
                labels: empty_labels(0),
                start: init_start(k),
                trans: init_trans(k),
                shape: Vector::filled(k, 2.0),
                scale: Vector::filled(k, 1.0),
                loglik: f64::NAN,
            });
        }
        let mut n_skip = 0usize;
        for i in 0..t_len {
            let y = x.get(i, 0);
            if y.is_finite() && y <= 0.0 {
                n_skip += 1;
            }
        }
        if n_skip > 0 {
            ctx.push(
                Issue::builder(IssueCode::NonPositiveSeries)
                    .severity(signlred::Severity::Warning)
                    .message(format!("InverseGammaHmm skipped {n_skip} non-positive observations"))
                    .build(),
            );
        }
        let mut shape = Vector::from_iter((0..k).map(|j| 2.0 + j as f64));
        let mut scale = Vector::from_iter((0..k).map(|j| 1.0 + 0.5 * j as f64));
        let mut start = init_start(k);
        let mut trans = init_trans(k);
        let mut loglik = f64::NEG_INFINITY;
        let mut last_gamma: Vec<Vec<f64>> = Vec::new();
        for it in 0..self.max_iter.max(1) {
            let dummy = FittedInverseGammaHmm {
                labels: Vector::zeros(0),
                start: start.clone(),
                trans: trans.clone(),
                shape: shape.clone(),
                scale: scale.clone(),
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
                let mut wz = 0.0_f64;
                let mut wz2 = 0.0_f64;
                for t in 0..t_len {
                    let y = x.get(t, 0);
                    if !y.is_finite() || y <= 0.0 {
                        continue;
                    }
                    let z = 1.0 / y;
                    let w = fb.gamma[t][j];
                    wsum += w;
                    wz += w * z;
                    wz2 += w * z * z;
                }
                if wsum > 1e-12 {
                    let m = wz / wsum;
                    let v = (wz2 / wsum - m * m).max(1e-8);
                    shape[j] = (m * m / v).clamp(0.2, 80.0);
                    scale[j] = (m * shape[j]).max(1e-4);
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
        let dummy = FittedInverseGammaHmm {
            labels: Vector::zeros(0),
            start: start.clone(),
            trans: trans.clone(),
            shape: shape.clone(),
            scale: scale.clone(),
            loglik,
        };
        let (labels, _) = viterbi_path(&start, &trans, &dummy.log_emit_seq(x));
        ctx.finish(FittedInverseGammaHmm {
            labels,
            start,
            trans,
            shape,
            scale,
            loglik,
        })
    }
}

/// Gumbel-emission HMM (extreme-value location/scale).
///
/// State count is not identification `p`. Distinct from [`LaplaceHmm`]
/// (symmetric) and [`GaussianHmm`] (light tails).
#[derive(Clone, Debug)]
pub struct GumbelHmm {
    /// Hidden states. Not identification `p`.
    pub n_states: usize,
    /// Baum–Welch cap.
    pub max_iter: usize,
}

impl Default for GumbelHmm {
    fn default() -> Self {
        Self {
            n_states: 2,
            max_iter: 40,
        }
    }
}

impl GumbelHmm {
    /// `k`-state Gumbel HMM.
    pub fn new(n_states: usize) -> Self {
        Self {
            n_states,
            ..Self::default()
        }
    }

    /// Fit alias.
    pub fn fit(&mut self, x: &Matrix, session: &Session) -> Result<Qualified<FittedGumbelHmm>> {
        self.fit_unsupervised(x, session)
    }
}

/// Fitted Gumbel HMM.
#[derive(Clone, Debug)]
pub struct FittedGumbelHmm {
    /// Viterbi path.
    pub labels: Vector,
    /// Start distribution.
    pub start: Vector,
    /// Transitions.
    pub trans: Matrix,
    /// Locations \(\mu_j\).
    pub loc: Vector,
    /// Scales \(\beta_j\).
    pub scale: Vector,
    /// Training log-likelihood.
    pub loglik: f64,
}

impl FittedGumbelHmm {
    fn log_emit_seq(&self, x: &Matrix) -> Vec<Vec<f64>> {
        let t = x.nrows();
        let s = self.loc.len();
        let mut out = vec![vec![f64::NEG_INFINITY; s]; t];
        for ti in 0..t {
            let y = x.get(ti, 0);
            for j in 0..s {
                out[ti][j] = log_gumbel(y, self.loc[j], self.scale[j]);
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

impl Predict for FittedGumbelHmm {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        self.decode(x, session)
    }
}

impl FitUnsupervised for GumbelHmm {
    type Fitted = FittedGumbelHmm;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedGumbelHmm>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let t_len = x.nrows();
        let k = self.n_states.max(1);
        if t_len == 0 {
            return ctx.finish(FittedGumbelHmm {
                labels: empty_labels(0),
                start: init_start(k),
                trans: init_trans(k),
                loc: Vector::zeros(k),
                scale: Vector::filled(k, 1.0),
                loglik: f64::NAN,
            });
        }
        let mean = x.column(0).mean();
        let sd = x.column(0).std().max(0.1);
        let mut loc = Vector::from_iter((0..k).map(|j| mean + (j as f64 - 0.5) * sd));
        let mut scale = Vector::filled(k, sd * 6.0_f64.sqrt() / std::f64::consts::PI);
        let mut start = init_start(k);
        let mut trans = init_trans(k);
        let mut loglik = f64::NEG_INFINITY;
        let mut last_gamma: Vec<Vec<f64>> = Vec::new();
        for it in 0..self.max_iter.max(1) {
            let dummy = FittedGumbelHmm {
                labels: Vector::zeros(0),
                start: start.clone(),
                trans: trans.clone(),
                loc: loc.clone(),
                scale: scale.clone(),
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
                let mut wad = 0.0_f64;
                for t in 0..t_len {
                    let y = x.get(t, 0);
                    if !y.is_finite() {
                        continue;
                    }
                    wsum += fb.gamma[t][j];
                    wy += fb.gamma[t][j] * y;
                }
                if wsum > 1e-12 {
                    loc[j] = wy / wsum;
                    for t in 0..t_len {
                        let y = x.get(t, 0);
                        if y.is_finite() {
                            wad += fb.gamma[t][j] * (y - loc[j]).abs();
                        }
                    }
                    scale[j] = (wad / wsum * 6.0_f64.sqrt() / std::f64::consts::PI).max(1e-4);
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
        let dummy = FittedGumbelHmm {
            labels: Vector::zeros(0),
            start: start.clone(),
            trans: trans.clone(),
            loc: loc.clone(),
            scale: scale.clone(),
            loglik,
        };
        let (labels, _) = viterbi_path(&start, &trans, &dummy.log_emit_seq(x));
        ctx.finish(FittedGumbelHmm {
            labels,
            start,
            trans,
            loc,
            scale,
            loglik,
        })
    }
}

/// Wrapped-Cauchy emission HMM (circular, heavier tails than von Mises).
///
/// State count is not identification `p`. Distinct from [`CircularHmm`]
/// (von Mises / Bessel \(I_0\)).
#[derive(Clone, Debug)]
pub struct WrappedCauchyHmm {
    /// Hidden states. Not identification `p`.
    pub n_states: usize,
    /// Baum–Welch cap.
    pub max_iter: usize,
}

impl Default for WrappedCauchyHmm {
    fn default() -> Self {
        Self {
            n_states: 2,
            max_iter: 40,
        }
    }
}

impl WrappedCauchyHmm {
    /// `k`-state wrapped-Cauchy HMM.
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
    ) -> Result<Qualified<FittedWrappedCauchyHmm>> {
        self.fit_unsupervised(x, session)
    }
}

/// Fitted wrapped-Cauchy HMM.
#[derive(Clone, Debug)]
pub struct FittedWrappedCauchyHmm {
    /// Viterbi path.
    pub labels: Vector,
    /// Start distribution.
    pub start: Vector,
    /// Transitions.
    pub trans: Matrix,
    /// Mean directions \(\mu_j\).
    pub mu: Vector,
    /// Concentrations \(\rho_j\in(0,1)\).
    pub rho: Vector,
    /// Training log-likelihood.
    pub loglik: f64,
}

impl FittedWrappedCauchyHmm {
    fn log_emit_seq(&self, x: &Matrix) -> Vec<Vec<f64>> {
        let t = x.nrows();
        let s = self.mu.len();
        let mut out = vec![vec![f64::NEG_INFINITY; s]; t];
        for ti in 0..t {
            let y = x.get(ti, 0);
            for j in 0..s {
                out[ti][j] = log_wrapped_cauchy(y, self.mu[j], self.rho[j]);
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

impl Predict for FittedWrappedCauchyHmm {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        self.decode(x, session)
    }
}

impl FitUnsupervised for WrappedCauchyHmm {
    type Fitted = FittedWrappedCauchyHmm;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedWrappedCauchyHmm>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let t_len = x.nrows();
        let k = self.n_states.max(1);
        if t_len == 0 {
            return ctx.finish(FittedWrappedCauchyHmm {
                labels: empty_labels(0),
                start: init_start(k),
                trans: init_trans(k),
                mu: Vector::zeros(k),
                rho: Vector::filled(k, 0.5),
                loglik: f64::NAN,
            });
        }
        let mut mu = Vector::from_iter((0..k).map(|j| (j as f64 - 0.5) * 0.8));
        let mut rho = Vector::filled(k, 0.4);
        let mut start = init_start(k);
        let mut trans = init_trans(k);
        let mut loglik = f64::NEG_INFINITY;
        let mut last_gamma: Vec<Vec<f64>> = Vec::new();
        for it in 0..self.max_iter.max(1) {
            let dummy = FittedWrappedCauchyHmm {
                labels: Vector::zeros(0),
                start: start.clone(),
                trans: trans.clone(),
                mu: mu.clone(),
                rho: rho.clone(),
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
                let mut cs = 0.0_f64;
                let mut ss = 0.0_f64;
                for t in 0..t_len {
                    let y = x.get(t, 0);
                    if !y.is_finite() {
                        continue;
                    }
                    let w = fb.gamma[t][j];
                    wsum += w;
                    cs += w * y.cos();
                    ss += w * y.sin();
                }
                if wsum > 1e-12 {
                    mu[j] = ss.atan2(cs);
                    rho[j] = ((cs * cs + ss * ss).sqrt() / wsum).clamp(0.05, 0.95);
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
        let dummy = FittedWrappedCauchyHmm {
            labels: Vector::zeros(0),
            start: start.clone(),
            trans: trans.clone(),
            mu: mu.clone(),
            rho: rho.clone(),
            loglik,
        };
        let (labels, _) = viterbi_path(&start, &trans, &dummy.log_emit_seq(x));
        ctx.finish(FittedWrappedCauchyHmm {
            labels,
            start,
            trans,
            mu,
            rho,
            loglik,
        })
    }
}

fn log_cauchy(y: f64, loc: f64, scale: f64) -> f64 {
    if !y.is_finite() || !loc.is_finite() || scale <= 0.0 {
        return f64::NEG_INFINITY;
    }
    let z = (y - loc) / scale;
    -std::f64::consts::PI.ln() - scale.ln() - (1.0 + z * z).ln()
}

fn log_logistic(y: f64, loc: f64, scale: f64) -> f64 {
    if !y.is_finite() || !loc.is_finite() || scale <= 0.0 {
        return f64::NEG_INFINITY;
    }
    let z = ((y - loc) / scale).clamp(-40.0, 40.0);
    -z - scale.ln() - 2.0 * (1.0 + (-z).exp()).ln()
}

fn log_rayleigh(y: f64, sigma: f64) -> f64 {
    if !y.is_finite() || y < 0.0 || sigma <= 0.0 {
        return f64::NEG_INFINITY;
    }
    y.max(1e-300).ln() - 2.0 * sigma.ln() - y * y / (2.0 * sigma * sigma)
}

fn log_rice(y: f64, nu: f64, sigma: f64) -> f64 {
    if !y.is_finite() || y < 0.0 || sigma <= 0.0 || nu < 0.0 {
        return f64::NEG_INFINITY;
    }
    let s2 = (sigma * sigma).max(1e-12);
    y.max(1e-300).ln() - s2.ln() - (y * y + nu * nu) / (2.0 * s2) + log_i0(y * nu / s2)
}

fn log_nakagami(y: f64, m: f64, omega: f64) -> f64 {
    if !y.is_finite() || y < 0.0 || m <= 0.0 || omega <= 0.0 {
        return f64::NEG_INFINITY;
    }
    std::f64::consts::LN_2 + m * m.ln() - crate::special::ln_gamma(m) - m * omega.ln()
        + (2.0 * m - 1.0) * y.max(1e-300).ln()
        - m * y * y / omega
}

fn log_zib(y: f64, pi: f64, n: f64, p: f64) -> f64 {
    if !y.is_finite() || y < 0.0 || n < 1.0 {
        return f64::NEG_INFINITY;
    }
    let pi = pi.clamp(1e-6, 1.0 - 1e-6);
    if y.abs() < 1e-12 {
        logsumexp(&[pi.ln(), (1.0 - pi).ln() + log_binomial(0.0, n, p)])
    } else {
        (1.0 - pi).ln() + log_binomial(y, n, p)
    }
}

/// Cauchy-emission HMM (undefined mean, heavy tails).
///
/// State count is not identification `p`. Distinct from [`LaplaceHmm`]
/// (finite mean) and [`StudentTHmm`] (finite \(\nu\)).
#[derive(Clone, Debug)]
pub struct CauchyHmm {
    /// Hidden states. Not identification `p`.
    pub n_states: usize,
    /// Baum–Welch cap.
    pub max_iter: usize,
}

impl Default for CauchyHmm {
    fn default() -> Self {
        Self {
            n_states: 2,
            max_iter: 40,
        }
    }
}

impl CauchyHmm {
    /// `k`-state Cauchy HMM.
    pub fn new(n_states: usize) -> Self {
        Self {
            n_states,
            ..Self::default()
        }
    }

    /// Fit alias.
    pub fn fit(&mut self, x: &Matrix, session: &Session) -> Result<Qualified<FittedCauchyHmm>> {
        self.fit_unsupervised(x, session)
    }
}

/// Fitted Cauchy HMM.
#[derive(Clone, Debug)]
pub struct FittedCauchyHmm {
    /// Viterbi path.
    pub labels: Vector,
    /// Start distribution.
    pub start: Vector,
    /// Transitions.
    pub trans: Matrix,
    /// Locations.
    pub loc: Vector,
    /// Scales \(\gamma_j\).
    pub scale: Vector,
    /// Training log-likelihood.
    pub loglik: f64,
}

impl FittedCauchyHmm {
    fn log_emit_seq(&self, x: &Matrix) -> Vec<Vec<f64>> {
        let t = x.nrows();
        let s = self.loc.len();
        let mut out = vec![vec![f64::NEG_INFINITY; s]; t];
        for ti in 0..t {
            let y = x.get(ti, 0);
            for j in 0..s {
                out[ti][j] = log_cauchy(y, self.loc[j], self.scale[j]);
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

impl Predict for FittedCauchyHmm {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        self.decode(x, session)
    }
}

impl FitUnsupervised for CauchyHmm {
    type Fitted = FittedCauchyHmm;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedCauchyHmm>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let t_len = x.nrows();
        let k = self.n_states.max(1);
        if t_len == 0 {
            return ctx.finish(FittedCauchyHmm {
                labels: empty_labels(0),
                start: init_start(k),
                trans: init_trans(k),
                loc: Vector::zeros(k),
                scale: Vector::filled(k, 1.0),
                loglik: f64::NAN,
            });
        }
        let mean = x.column(0).mean();
        let sd = x.column(0).std().max(0.1);
        let mut loc = Vector::from_iter((0..k).map(|j| mean + (j as f64 - 0.5) * sd));
        let mut scale = Vector::filled(k, sd);
        let mut start = init_start(k);
        let mut trans = init_trans(k);
        let mut loglik = f64::NEG_INFINITY;
        let mut last_gamma: Vec<Vec<f64>> = Vec::new();
        for it in 0..self.max_iter.max(1) {
            let dummy = FittedCauchyHmm {
                labels: Vector::zeros(0),
                start: start.clone(),
                trans: trans.clone(),
                loc: loc.clone(),
                scale: scale.clone(),
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
                let mut wad = 0.0_f64;
                for t in 0..t_len {
                    let y = x.get(t, 0);
                    if !y.is_finite() {
                        continue;
                    }
                    wsum += fb.gamma[t][j];
                    wy += fb.gamma[t][j] * y;
                }
                if wsum > 1e-12 {
                    loc[j] = wy / wsum;
                    for t in 0..t_len {
                        let y = x.get(t, 0);
                        if y.is_finite() {
                            wad += fb.gamma[t][j] * (y - loc[j]).abs();
                        }
                    }
                    scale[j] = (wad / wsum).max(COV_FLOOR.sqrt());
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
        let dummy = FittedCauchyHmm {
            labels: Vector::zeros(0),
            start: start.clone(),
            trans: trans.clone(),
            loc: loc.clone(),
            scale: scale.clone(),
            loglik,
        };
        let (labels, _) = viterbi_path(&start, &trans, &dummy.log_emit_seq(x));
        ctx.finish(FittedCauchyHmm {
            labels,
            start,
            trans,
            loc,
            scale,
            loglik,
        })
    }
}

/// Logistic-emission HMM (sech-squared tails).
///
/// State count is not identification `p`. Distinct from [`LaplaceHmm`]
/// (exponential tails) and [`GaussianHmm`] (quadratic).
#[derive(Clone, Debug)]
pub struct LogisticHmm {
    /// Hidden states. Not identification `p`.
    pub n_states: usize,
    /// Baum–Welch cap.
    pub max_iter: usize,
}

impl Default for LogisticHmm {
    fn default() -> Self {
        Self {
            n_states: 2,
            max_iter: 40,
        }
    }
}

impl LogisticHmm {
    /// `k`-state logistic HMM.
    pub fn new(n_states: usize) -> Self {
        Self {
            n_states,
            ..Self::default()
        }
    }

    /// Fit alias.
    pub fn fit(&mut self, x: &Matrix, session: &Session) -> Result<Qualified<FittedLogisticHmm>> {
        self.fit_unsupervised(x, session)
    }
}

/// Fitted logistic HMM.
#[derive(Clone, Debug)]
pub struct FittedLogisticHmm {
    /// Viterbi path.
    pub labels: Vector,
    /// Start distribution.
    pub start: Vector,
    /// Transitions.
    pub trans: Matrix,
    /// Locations.
    pub loc: Vector,
    /// Scales.
    pub scale: Vector,
    /// Training log-likelihood.
    pub loglik: f64,
}

impl FittedLogisticHmm {
    fn log_emit_seq(&self, x: &Matrix) -> Vec<Vec<f64>> {
        let t = x.nrows();
        let s = self.loc.len();
        let mut out = vec![vec![f64::NEG_INFINITY; s]; t];
        for ti in 0..t {
            let y = x.get(ti, 0);
            for j in 0..s {
                out[ti][j] = log_logistic(y, self.loc[j], self.scale[j]);
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

impl Predict for FittedLogisticHmm {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        self.decode(x, session)
    }
}

impl FitUnsupervised for LogisticHmm {
    type Fitted = FittedLogisticHmm;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedLogisticHmm>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let t_len = x.nrows();
        let k = self.n_states.max(1);
        if t_len == 0 {
            return ctx.finish(FittedLogisticHmm {
                labels: empty_labels(0),
                start: init_start(k),
                trans: init_trans(k),
                loc: Vector::zeros(k),
                scale: Vector::filled(k, 1.0),
                loglik: f64::NAN,
            });
        }
        let mean = x.column(0).mean();
        let sd = x.column(0).std().max(0.1);
        let mut loc = Vector::from_iter((0..k).map(|j| mean + (j as f64 - 0.5) * sd));
        let mut scale = Vector::filled(k, sd * 3.0_f64.sqrt() / std::f64::consts::PI);
        let mut start = init_start(k);
        let mut trans = init_trans(k);
        let mut loglik = f64::NEG_INFINITY;
        let mut last_gamma: Vec<Vec<f64>> = Vec::new();
        for it in 0..self.max_iter.max(1) {
            let dummy = FittedLogisticHmm {
                labels: Vector::zeros(0),
                start: start.clone(),
                trans: trans.clone(),
                loc: loc.clone(),
                scale: scale.clone(),
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
                let mut wad = 0.0_f64;
                for t in 0..t_len {
                    let y = x.get(t, 0);
                    if !y.is_finite() {
                        continue;
                    }
                    wsum += fb.gamma[t][j];
                    wy += fb.gamma[t][j] * y;
                }
                if wsum > 1e-12 {
                    loc[j] = wy / wsum;
                    for t in 0..t_len {
                        let y = x.get(t, 0);
                        if y.is_finite() {
                            wad += fb.gamma[t][j] * (y - loc[j]).abs();
                        }
                    }
                    scale[j] = (wad / wsum * 3.0_f64.sqrt() / std::f64::consts::PI).max(1e-4);
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
        let dummy = FittedLogisticHmm {
            labels: Vector::zeros(0),
            start: start.clone(),
            trans: trans.clone(),
            loc: loc.clone(),
            scale: scale.clone(),
            loglik,
        };
        let (labels, _) = viterbi_path(&start, &trans, &dummy.log_emit_seq(x));
        ctx.finish(FittedLogisticHmm {
            labels,
            start,
            trans,
            loc,
            scale,
            loglik,
        })
    }
}

/// Rayleigh-emission HMM (zero-mean Rice / Weibull \(k=2\)).
///
/// State count is not identification `p`. Distinct from [`WeibullHmm`]
/// (free shape) and [`ExponentialHmm`] (linear hazard).
#[derive(Clone, Debug)]
pub struct RayleighHmm {
    /// Hidden states. Not identification `p`.
    pub n_states: usize,
    /// Baum–Welch cap.
    pub max_iter: usize,
}

impl Default for RayleighHmm {
    fn default() -> Self {
        Self {
            n_states: 2,
            max_iter: 40,
        }
    }
}

impl RayleighHmm {
    /// `k`-state Rayleigh HMM.
    pub fn new(n_states: usize) -> Self {
        Self {
            n_states,
            ..Self::default()
        }
    }

    /// Fit alias.
    pub fn fit(&mut self, x: &Matrix, session: &Session) -> Result<Qualified<FittedRayleighHmm>> {
        self.fit_unsupervised(x, session)
    }
}

/// Fitted Rayleigh HMM.
#[derive(Clone, Debug)]
pub struct FittedRayleighHmm {
    /// Viterbi path.
    pub labels: Vector,
    /// Start distribution.
    pub start: Vector,
    /// Transitions.
    pub trans: Matrix,
    /// Scales \(\sigma_j\).
    pub sigma: Vector,
    /// Training log-likelihood.
    pub loglik: f64,
}

impl FittedRayleighHmm {
    fn log_emit_seq(&self, x: &Matrix) -> Vec<Vec<f64>> {
        let t = x.nrows();
        let s = self.sigma.len();
        let mut out = vec![vec![f64::NEG_INFINITY; s]; t];
        for ti in 0..t {
            let y = x.get(ti, 0);
            for j in 0..s {
                out[ti][j] = log_rayleigh(y, self.sigma[j]);
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

impl Predict for FittedRayleighHmm {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        self.decode(x, session)
    }
}

impl FitUnsupervised for RayleighHmm {
    type Fitted = FittedRayleighHmm;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedRayleighHmm>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let t_len = x.nrows();
        let k = self.n_states.max(1);
        if t_len == 0 {
            return ctx.finish(FittedRayleighHmm {
                labels: empty_labels(0),
                start: init_start(k),
                trans: init_trans(k),
                sigma: Vector::filled(k, 1.0),
                loglik: f64::NAN,
            });
        }
        let mut n_skip = 0usize;
        for i in 0..t_len {
            if x.get(i, 0).is_finite() && x.get(i, 0) < 0.0 {
                n_skip += 1;
            }
        }
        if n_skip > 0 {
            ctx.push(
                Issue::builder(IssueCode::NonPositiveSeries)
                    .severity(signlred::Severity::Warning)
                    .message(format!("RayleighHmm skipped {n_skip} negative observations"))
                    .build(),
            );
        }
        let mut sigma = Vector::from_iter((0..k).map(|j| 0.5 + j as f64));
        let mut start = init_start(k);
        let mut trans = init_trans(k);
        let mut loglik = f64::NEG_INFINITY;
        let mut last_gamma: Vec<Vec<f64>> = Vec::new();
        for it in 0..self.max_iter.max(1) {
            let dummy = FittedRayleighHmm {
                labels: Vector::zeros(0),
                start: start.clone(),
                trans: trans.clone(),
                sigma: sigma.clone(),
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
                let mut wy2 = 0.0_f64;
                for t in 0..t_len {
                    let y = x.get(t, 0);
                    if !y.is_finite() || y < 0.0 {
                        continue;
                    }
                    wsum += fb.gamma[t][j];
                    wy2 += fb.gamma[t][j] * y * y;
                }
                if wsum > 1e-12 {
                    sigma[j] = (wy2 / (2.0 * wsum)).max(1e-8).sqrt();
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
        let dummy = FittedRayleighHmm {
            labels: Vector::zeros(0),
            start: start.clone(),
            trans: trans.clone(),
            sigma: sigma.clone(),
            loglik,
        };
        let (labels, _) = viterbi_path(&start, &trans, &dummy.log_emit_seq(x));
        ctx.finish(FittedRayleighHmm {
            labels,
            start,
            trans,
            sigma,
            loglik,
        })
    }
}

/// Rician-emission HMM (non-central Rayleigh).
///
/// Non-centrality is not identification `p`. Distinct from [`RayleighHmm`]
/// (\(\nu=0\)) and [`GaussianHmm`] (signed support).
#[derive(Clone, Debug)]
pub struct RiceHmm {
    /// Hidden states. Not identification `p`.
    pub n_states: usize,
    /// Baum–Welch cap.
    pub max_iter: usize,
}

impl Default for RiceHmm {
    fn default() -> Self {
        Self {
            n_states: 2,
            max_iter: 40,
        }
    }
}

impl RiceHmm {
    /// `k`-state Rice HMM.
    pub fn new(n_states: usize) -> Self {
        Self {
            n_states,
            ..Self::default()
        }
    }

    /// Fit alias.
    pub fn fit(&mut self, x: &Matrix, session: &Session) -> Result<Qualified<FittedRiceHmm>> {
        self.fit_unsupervised(x, session)
    }
}

/// Fitted Rice HMM.
#[derive(Clone, Debug)]
pub struct FittedRiceHmm {
    /// Viterbi path.
    pub labels: Vector,
    /// Start distribution.
    pub start: Vector,
    /// Transitions.
    pub trans: Matrix,
    /// Non-centrality \(\nu_j\).
    pub nu: Vector,
    /// Scales \(\sigma_j\).
    pub sigma: Vector,
    /// Training log-likelihood.
    pub loglik: f64,
}

impl FittedRiceHmm {
    fn log_emit_seq(&self, x: &Matrix) -> Vec<Vec<f64>> {
        let t = x.nrows();
        let s = self.nu.len();
        let mut out = vec![vec![f64::NEG_INFINITY; s]; t];
        for ti in 0..t {
            let y = x.get(ti, 0);
            for j in 0..s {
                out[ti][j] = log_rice(y, self.nu[j], self.sigma[j]);
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

impl Predict for FittedRiceHmm {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        self.decode(x, session)
    }
}

impl FitUnsupervised for RiceHmm {
    type Fitted = FittedRiceHmm;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedRiceHmm>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let t_len = x.nrows();
        let k = self.n_states.max(1);
        if t_len == 0 {
            return ctx.finish(FittedRiceHmm {
                labels: empty_labels(0),
                start: init_start(k),
                trans: init_trans(k),
                nu: Vector::zeros(k),
                sigma: Vector::filled(k, 1.0),
                loglik: f64::NAN,
            });
        }
        let mut nu = Vector::from_iter((0..k).map(|j| 0.5 + j as f64));
        let mut sigma = Vector::filled(k, 0.8);
        let mut start = init_start(k);
        let mut trans = init_trans(k);
        let mut loglik = f64::NEG_INFINITY;
        let mut last_gamma: Vec<Vec<f64>> = Vec::new();
        for it in 0..self.max_iter.max(1) {
            let dummy = FittedRiceHmm {
                labels: Vector::zeros(0),
                start: start.clone(),
                trans: trans.clone(),
                nu: nu.clone(),
                sigma: sigma.clone(),
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
                    if !y.is_finite() || y < 0.0 {
                        continue;
                    }
                    wsum += fb.gamma[t][j];
                    wy += fb.gamma[t][j] * y;
                    wy2 += fb.gamma[t][j] * y * y;
                }
                if wsum > 1e-12 {
                    let m1 = wy / wsum;
                    let m2 = wy2 / wsum;
                    nu[j] = m1.max(0.0);
                    sigma[j] = ((m2 - m1 * m1).max(1e-8) * 0.5).sqrt();
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
        let dummy = FittedRiceHmm {
            labels: Vector::zeros(0),
            start: start.clone(),
            trans: trans.clone(),
            nu: nu.clone(),
            sigma: sigma.clone(),
            loglik,
        };
        let (labels, _) = viterbi_path(&start, &trans, &dummy.log_emit_seq(x));
        ctx.finish(FittedRiceHmm {
            labels,
            start,
            trans,
            nu,
            sigma,
            loglik,
        })
    }
}

/// Nakagami-\(m\) emission HMM.
///
/// Shape \(m\) is not identification `p`. Distinct from [`RayleighHmm`]
/// (\(m=1\)) and [`GammaHmm`] (linear, not squared, argument).
#[derive(Clone, Debug)]
pub struct NakagamiHmm {
    /// Hidden states. Not identification `p`.
    pub n_states: usize,
    /// Baum–Welch cap.
    pub max_iter: usize,
}

impl Default for NakagamiHmm {
    fn default() -> Self {
        Self {
            n_states: 2,
            max_iter: 40,
        }
    }
}

impl NakagamiHmm {
    /// `k`-state Nakagami HMM.
    pub fn new(n_states: usize) -> Self {
        Self {
            n_states,
            ..Self::default()
        }
    }

    /// Fit alias.
    pub fn fit(&mut self, x: &Matrix, session: &Session) -> Result<Qualified<FittedNakagamiHmm>> {
        self.fit_unsupervised(x, session)
    }
}

/// Fitted Nakagami HMM.
#[derive(Clone, Debug)]
pub struct FittedNakagamiHmm {
    /// Viterbi path.
    pub labels: Vector,
    /// Start distribution.
    pub start: Vector,
    /// Transitions.
    pub trans: Matrix,
    /// Shapes \(m_j\).
    pub m: Vector,
    /// Spreads \(\Omega_j\).
    pub omega: Vector,
    /// Training log-likelihood.
    pub loglik: f64,
}

impl FittedNakagamiHmm {
    fn log_emit_seq(&self, x: &Matrix) -> Vec<Vec<f64>> {
        let t = x.nrows();
        let s = self.m.len();
        let mut out = vec![vec![f64::NEG_INFINITY; s]; t];
        for ti in 0..t {
            let y = x.get(ti, 0);
            for j in 0..s {
                out[ti][j] = log_nakagami(y, self.m[j], self.omega[j]);
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

impl Predict for FittedNakagamiHmm {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        self.decode(x, session)
    }
}

impl FitUnsupervised for NakagamiHmm {
    type Fitted = FittedNakagamiHmm;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedNakagamiHmm>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let t_len = x.nrows();
        let k = self.n_states.max(1);
        if t_len == 0 {
            return ctx.finish(FittedNakagamiHmm {
                labels: empty_labels(0),
                start: init_start(k),
                trans: init_trans(k),
                m: Vector::filled(k, 1.0),
                omega: Vector::filled(k, 1.0),
                loglik: f64::NAN,
            });
        }
        let mut mpar = Vector::from_iter((0..k).map(|j| 1.0 + 0.4 * j as f64));
        let mut omega = Vector::from_iter((0..k).map(|j| 1.0 + j as f64));
        let mut start = init_start(k);
        let mut trans = init_trans(k);
        let mut loglik = f64::NEG_INFINITY;
        let mut last_gamma: Vec<Vec<f64>> = Vec::new();
        for it in 0..self.max_iter.max(1) {
            let dummy = FittedNakagamiHmm {
                labels: Vector::zeros(0),
                start: start.clone(),
                trans: trans.clone(),
                m: mpar.clone(),
                omega: omega.clone(),
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
                let mut wy2 = 0.0_f64;
                let mut wy4 = 0.0_f64;
                for t in 0..t_len {
                    let y = x.get(t, 0);
                    if !y.is_finite() || y < 0.0 {
                        continue;
                    }
                    let y2 = y * y;
                    wsum += fb.gamma[t][j];
                    wy2 += fb.gamma[t][j] * y2;
                    wy4 += fb.gamma[t][j] * y2 * y2;
                }
                if wsum > 1e-12 {
                    let om = (wy2 / wsum).max(1e-8);
                    omega[j] = om;
                    let v = (wy4 / wsum - om * om).max(1e-8);
                    mpar[j] = (om * om / v).clamp(0.25, 40.0);
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
        let dummy = FittedNakagamiHmm {
            labels: Vector::zeros(0),
            start: start.clone(),
            trans: trans.clone(),
            m: mpar.clone(),
            omega: omega.clone(),
            loglik,
        };
        let (labels, _) = viterbi_path(&start, &trans, &dummy.log_emit_seq(x));
        ctx.finish(FittedNakagamiHmm {
            labels,
            start,
            trans,
            m: mpar,
            omega,
            loglik,
        })
    }
}

/// Zero-inflated binomial HMM.
///
/// Trial count is not identification `p`. Distinct from [`BinomialHmm`]
/// (no extra zeros) and [`ZeroInflatedPoissonHmm`] (unbounded).
#[derive(Clone, Debug)]
pub struct ZeroInflatedBinomialHmm {
    /// Hidden states. Not identification `p`.
    pub n_states: usize,
    /// Trials per observation. Not identification `p`.
    pub n_trials: f64,
    /// Baum–Welch cap.
    pub max_iter: usize,
}

impl Default for ZeroInflatedBinomialHmm {
    fn default() -> Self {
        Self {
            n_states: 2,
            n_trials: 8.0,
            max_iter: 40,
        }
    }
}

impl ZeroInflatedBinomialHmm {
    /// `k`-state ZIB HMM with `n_trials` trials.
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
    ) -> Result<Qualified<FittedZeroInflatedBinomialHmm>> {
        self.fit_unsupervised(x, session)
    }
}

/// Fitted ZIB HMM.
#[derive(Clone, Debug)]
pub struct FittedZeroInflatedBinomialHmm {
    /// Viterbi path.
    pub labels: Vector,
    /// Start distribution.
    pub start: Vector,
    /// Transitions.
    pub trans: Matrix,
    /// Extra-zero mass.
    pub pi: Vector,
    /// Success probabilities.
    pub p: Vector,
    /// Trial count.
    pub n_trials: f64,
    /// Training log-likelihood.
    pub loglik: f64,
}

impl FittedZeroInflatedBinomialHmm {
    fn log_emit_seq(&self, x: &Matrix) -> Vec<Vec<f64>> {
        let t = x.nrows();
        let s = self.p.len();
        let mut out = vec![vec![f64::NEG_INFINITY; s]; t];
        for ti in 0..t {
            let y = x.get(ti, 0);
            for j in 0..s {
                out[ti][j] = log_zib(y, self.pi[j], self.n_trials, self.p[j]);
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

impl Predict for FittedZeroInflatedBinomialHmm {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        self.decode(x, session)
    }
}

impl FitUnsupervised for ZeroInflatedBinomialHmm {
    type Fitted = FittedZeroInflatedBinomialHmm;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedZeroInflatedBinomialHmm>> {
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
            return ctx.finish(FittedZeroInflatedBinomialHmm {
                labels: empty_labels(0),
                start: init_start(k),
                trans: init_trans(k),
                pi: Vector::filled(k, 0.1),
                p: Vector::filled(k, 0.5),
                n_trials: nt,
                loglik: f64::NAN,
            });
        }
        let mut pi = Vector::filled(k, 0.12);
        let mut p = Vector::from_iter((0..k).map(|j| 0.2 + 0.5 * j as f64 / k.max(1) as f64));
        let mut start = init_start(k);
        let mut trans = init_trans(k);
        let mut loglik = f64::NEG_INFINITY;
        let mut last_gamma: Vec<Vec<f64>> = Vec::new();
        for it in 0..self.max_iter.max(1) {
            let dummy = FittedZeroInflatedBinomialHmm {
                labels: Vector::zeros(0),
                start: start.clone(),
                trans: trans.clone(),
                pi: pi.clone(),
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
                let mut wzero = 0.0_f64;
                let mut wy = 0.0_f64;
                for t in 0..t_len {
                    let y = x.get(t, 0);
                    if !y.is_finite() || y < 0.0 || y > nt {
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
                    let zf = (wzero / wsum).clamp(0.0, 0.95);
                    let b0 = log_binomial(0.0, nt, p[j]).exp();
                    pi[j] = ((zf - b0) / (1.0 - b0).max(1e-6)).clamp(1e-4, 0.9);
                    p[j] = ((wy / wsum) / ((1.0 - pi[j]) * nt).max(1e-6)).clamp(1e-3, 1.0 - 1e-3);
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
        let dummy = FittedZeroInflatedBinomialHmm {
            labels: Vector::zeros(0),
            start: start.clone(),
            trans: trans.clone(),
            pi: pi.clone(),
            p: p.clone(),
            n_trials: nt,
            loglik,
        };
        let (labels, _) = viterbi_path(&start, &trans, &dummy.log_emit_seq(x));
        ctx.finish(FittedZeroInflatedBinomialHmm {
            labels,
            start,
            trans,
            pi,
            p,
            n_trials: nt,
            loglik,
        })
    }
}

fn log_asymmetric_laplace(y: f64, loc: f64, left: f64, right: f64) -> f64 {
    if !y.is_finite() || !loc.is_finite() || left <= 0.0 || right <= 0.0 {
        return f64::NEG_INFINITY;
    }
    let norm = -(left + right).ln();
    if y >= loc {
        norm - (y - loc) / right
    } else {
        norm - (loc - y) / left
    }
}

fn log_frechet(y: f64, shape: f64, scale: f64) -> f64 {
    if !y.is_finite() || y <= 0.0 || shape <= 0.0 || scale <= 0.0 {
        return f64::NEG_INFINITY;
    }
    let z = (y / scale).max(1e-12);
    shape.ln() - scale.ln() - (shape + 1.0) * z.ln() - z.powf(-shape)
}

fn log_wrapped_normal(y: f64, mu: f64, sigma: f64) -> f64 {
    if !y.is_finite() || !mu.is_finite() || sigma <= 0.0 {
        return f64::NEG_INFINITY;
    }
    let s2 = (sigma * sigma).max(1e-12);
    let mut terms = [0.0_f64; 5];
    for (k, slot) in (-2i32..=2).zip(terms.iter_mut()) {
        let d = y - mu + 2.0 * std::f64::consts::PI * f64::from(k);
        *slot = -0.5 * (LN_2PI + s2.ln()) - 0.5 * d * d / s2;
    }
    logsumexp(&terms)
}

fn log_kumaraswamy(y: f64, a: f64, b: f64) -> f64 {
    if !y.is_finite() || y <= 0.0 || y >= 1.0 || a <= 0.0 || b <= 0.0 {
        return f64::NEG_INFINITY;
    }
    let ya = y.powf(a);
    if ya >= 1.0 {
        return f64::NEG_INFINITY;
    }
    a.ln() + b.ln() + (a - 1.0) * y.ln() + (b - 1.0) * (1.0 - ya).max(1e-300).ln()
}

fn log_loglogistic(y: f64, scale: f64, shape: f64) -> f64 {
    if !y.is_finite() || y <= 0.0 || scale <= 0.0 || shape <= 0.0 {
        return f64::NEG_INFINITY;
    }
    let z = (y / scale).max(1e-12);
    let zb = z.powf(shape);
    shape.ln() - scale.ln() + (shape - 1.0) * z.ln() - 2.0 * (1.0 + zb).ln()
}

fn log_hurdle_poisson(y: f64, pi: f64, lam: f64) -> f64 {
    if !y.is_finite() || y < 0.0 || lam <= 0.0 {
        return f64::NEG_INFINITY;
    }
    let pi = pi.clamp(1e-6, 1.0 - 1e-6);
    if y.abs() < 1e-12 {
        pi.ln()
    } else {
        let p0 = (-lam).exp();
        (1.0 - pi).ln() + log_poisson(y, lam) - (1.0 - p0).max(1e-12).ln()
    }
}

/// Asymmetric-Laplace HMM (two-sided exponential scales).
///
/// State count is not identification `p`. Distinct from [`LaplaceHmm`]
/// (one scale) and [`LogisticHmm`] (sech-squared).
#[derive(Clone, Debug)]
pub struct AsymmetricLaplaceHmm {
    /// Hidden states. Not identification `p`.
    pub n_states: usize,
    /// Baum–Welch cap.
    pub max_iter: usize,
}

impl Default for AsymmetricLaplaceHmm {
    fn default() -> Self {
        Self {
            n_states: 2,
            max_iter: 40,
        }
    }
}

impl AsymmetricLaplaceHmm {
    /// `k`-state asymmetric-Laplace HMM.
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
    ) -> Result<Qualified<FittedAsymmetricLaplaceHmm>> {
        self.fit_unsupervised(x, session)
    }
}

/// Fitted asymmetric-Laplace HMM.
#[derive(Clone, Debug)]
pub struct FittedAsymmetricLaplaceHmm {
    /// Viterbi path.
    pub labels: Vector,
    /// Start distribution.
    pub start: Vector,
    /// Transitions.
    pub trans: Matrix,
    /// Locations.
    pub loc: Vector,
    /// Left scales.
    pub left: Vector,
    /// Right scales.
    pub right: Vector,
    /// Training log-likelihood.
    pub loglik: f64,
}

impl FittedAsymmetricLaplaceHmm {
    fn log_emit_seq(&self, x: &Matrix) -> Vec<Vec<f64>> {
        let t = x.nrows();
        let s = self.loc.len();
        let mut out = vec![vec![f64::NEG_INFINITY; s]; t];
        for ti in 0..t {
            let y = x.get(ti, 0);
            for j in 0..s {
                out[ti][j] = log_asymmetric_laplace(y, self.loc[j], self.left[j], self.right[j]);
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

impl Predict for FittedAsymmetricLaplaceHmm {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        self.decode(x, session)
    }
}

impl FitUnsupervised for AsymmetricLaplaceHmm {
    type Fitted = FittedAsymmetricLaplaceHmm;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedAsymmetricLaplaceHmm>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let t_len = x.nrows();
        let k = self.n_states.max(1);
        if t_len == 0 {
            return ctx.finish(FittedAsymmetricLaplaceHmm {
                labels: empty_labels(0),
                start: init_start(k),
                trans: init_trans(k),
                loc: Vector::zeros(k),
                left: Vector::filled(k, 1.0),
                right: Vector::filled(k, 1.0),
                loglik: f64::NAN,
            });
        }
        let mean = x.column(0).mean();
        let sd = x.column(0).std().max(0.1);
        let mut loc = Vector::from_iter((0..k).map(|j| mean + (j as f64 - 0.5) * sd));
        let mut left = Vector::filled(k, sd);
        let mut right = Vector::filled(k, sd);
        let mut start = init_start(k);
        let mut trans = init_trans(k);
        let mut loglik = f64::NEG_INFINITY;
        let mut last_gamma: Vec<Vec<f64>> = Vec::new();
        for it in 0..self.max_iter.max(1) {
            let dummy = FittedAsymmetricLaplaceHmm {
                labels: Vector::zeros(0),
                start: start.clone(),
                trans: trans.clone(),
                loc: loc.clone(),
                left: left.clone(),
                right: right.clone(),
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
                    loc[j] = wy / wsum;
                    let mut wl = 0.0_f64;
                    let mut wr = 0.0_f64;
                    let mut nl = 0.0_f64;
                    let mut nr = 0.0_f64;
                    for t in 0..t_len {
                        let y = x.get(t, 0);
                        if !y.is_finite() {
                            continue;
                        }
                        let w = fb.gamma[t][j];
                        if y >= loc[j] {
                            wr += w * (y - loc[j]);
                            nr += w;
                        } else {
                            wl += w * (loc[j] - y);
                            nl += w;
                        }
                    }
                    left[j] = if nl > 1e-12 {
                        (wl / nl).max(COV_FLOOR.sqrt())
                    } else {
                        left[j]
                    };
                    right[j] = if nr > 1e-12 {
                        (wr / nr).max(COV_FLOOR.sqrt())
                    } else {
                        right[j]
                    };
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
        let dummy = FittedAsymmetricLaplaceHmm {
            labels: Vector::zeros(0),
            start: start.clone(),
            trans: trans.clone(),
            loc: loc.clone(),
            left: left.clone(),
            right: right.clone(),
            loglik,
        };
        let (labels, _) = viterbi_path(&start, &trans, &dummy.log_emit_seq(x));
        ctx.finish(FittedAsymmetricLaplaceHmm {
            labels,
            start,
            trans,
            loc,
            left,
            right,
            loglik,
        })
    }
}

/// Fréchet-emission HMM (type-II extreme value).
///
/// State count is not identification `p`. Distinct from [`GumbelHmm`]
/// (unbounded support) and [`WeibullHmm`] (reversed extreme value).
#[derive(Clone, Debug)]
pub struct FrechetHmm {
    /// Hidden states. Not identification `p`.
    pub n_states: usize,
    /// Baum–Welch cap.
    pub max_iter: usize,
}

impl Default for FrechetHmm {
    fn default() -> Self {
        Self {
            n_states: 2,
            max_iter: 40,
        }
    }
}

impl FrechetHmm {
    /// `k`-state Fréchet HMM.
    pub fn new(n_states: usize) -> Self {
        Self {
            n_states,
            ..Self::default()
        }
    }

    /// Fit alias.
    pub fn fit(&mut self, x: &Matrix, session: &Session) -> Result<Qualified<FittedFrechetHmm>> {
        self.fit_unsupervised(x, session)
    }
}

/// Fitted Fréchet HMM.
#[derive(Clone, Debug)]
pub struct FittedFrechetHmm {
    /// Viterbi path.
    pub labels: Vector,
    /// Start distribution.
    pub start: Vector,
    /// Transitions.
    pub trans: Matrix,
    /// Shapes \(\alpha_j\).
    pub shape: Vector,
    /// Scales \(s_j\).
    pub scale: Vector,
    /// Training log-likelihood.
    pub loglik: f64,
}

impl FittedFrechetHmm {
    fn log_emit_seq(&self, x: &Matrix) -> Vec<Vec<f64>> {
        let t = x.nrows();
        let s = self.shape.len();
        let mut out = vec![vec![f64::NEG_INFINITY; s]; t];
        for ti in 0..t {
            let y = x.get(ti, 0);
            for j in 0..s {
                out[ti][j] = log_frechet(y, self.shape[j], self.scale[j]);
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

impl Predict for FittedFrechetHmm {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        self.decode(x, session)
    }
}

impl FitUnsupervised for FrechetHmm {
    type Fitted = FittedFrechetHmm;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedFrechetHmm>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let t_len = x.nrows();
        let k = self.n_states.max(1);
        if t_len == 0 {
            return ctx.finish(FittedFrechetHmm {
                labels: empty_labels(0),
                start: init_start(k),
                trans: init_trans(k),
                shape: Vector::filled(k, 2.0),
                scale: Vector::filled(k, 1.0),
                loglik: f64::NAN,
            });
        }
        let mut n_skip = 0usize;
        for i in 0..t_len {
            if x.get(i, 0).is_finite() && x.get(i, 0) <= 0.0 {
                n_skip += 1;
            }
        }
        if n_skip > 0 {
            ctx.push(
                Issue::builder(IssueCode::NonPositiveSeries)
                    .severity(signlred::Severity::Warning)
                    .message(format!("FrechetHmm skipped {n_skip} non-positive observations"))
                    .build(),
            );
        }
        let mut shape = Vector::from_iter((0..k).map(|j| 2.0 + j as f64));
        let mut scale = Vector::from_iter((0..k).map(|j| 1.0 + 0.5 * j as f64));
        let mut start = init_start(k);
        let mut trans = init_trans(k);
        let mut loglik = f64::NEG_INFINITY;
        let mut last_gamma: Vec<Vec<f64>> = Vec::new();
        for it in 0..self.max_iter.max(1) {
            let dummy = FittedFrechetHmm {
                labels: Vector::zeros(0),
                start: start.clone(),
                trans: trans.clone(),
                shape: shape.clone(),
                scale: scale.clone(),
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
                let mut wln = 0.0_f64;
                let mut wln2 = 0.0_f64;
                for t in 0..t_len {
                    let y = x.get(t, 0);
                    if !y.is_finite() || y <= 0.0 {
                        continue;
                    }
                    let w = fb.gamma[t][j];
                    let z = y.ln();
                    wsum += w;
                    wln += w * z;
                    wln2 += w * z * z;
                }
                if wsum > 1e-12 {
                    let m = wln / wsum;
                    let v = (wln2 / wsum - m * m).max(1e-8);
                    let alpha = (std::f64::consts::PI / (6.0 * v).sqrt()).clamp(0.2, 80.0);
                    shape[j] = alpha;
                    scale[j] = (m - 0.5772156649015329 / alpha).exp().max(1e-4);
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
        let dummy = FittedFrechetHmm {
            labels: Vector::zeros(0),
            start: start.clone(),
            trans: trans.clone(),
            shape: shape.clone(),
            scale: scale.clone(),
            loglik,
        };
        let (labels, _) = viterbi_path(&start, &trans, &dummy.log_emit_seq(x));
        ctx.finish(FittedFrechetHmm {
            labels,
            start,
            trans,
            shape,
            scale,
            loglik,
        })
    }
}

/// Wrapped-normal HMM (Gaussian wraps on the circle).
///
/// State count is not identification `p`. Distinct from [`CircularHmm`]
/// (von Mises) and [`WrappedCauchyHmm`] (heavier circular tails).
#[derive(Clone, Debug)]
pub struct WrappedNormalHmm {
    /// Hidden states. Not identification `p`.
    pub n_states: usize,
    /// Baum–Welch cap.
    pub max_iter: usize,
}

impl Default for WrappedNormalHmm {
    fn default() -> Self {
        Self {
            n_states: 2,
            max_iter: 40,
        }
    }
}

impl WrappedNormalHmm {
    /// `k`-state wrapped-normal HMM.
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
    ) -> Result<Qualified<FittedWrappedNormalHmm>> {
        self.fit_unsupervised(x, session)
    }
}

/// Fitted wrapped-normal HMM.
#[derive(Clone, Debug)]
pub struct FittedWrappedNormalHmm {
    /// Viterbi path.
    pub labels: Vector,
    /// Start distribution.
    pub start: Vector,
    /// Transitions.
    pub trans: Matrix,
    /// Mean directions.
    pub mu: Vector,
    /// Circular scales.
    pub sigma: Vector,
    /// Training log-likelihood.
    pub loglik: f64,
}

impl FittedWrappedNormalHmm {
    fn log_emit_seq(&self, x: &Matrix) -> Vec<Vec<f64>> {
        let t = x.nrows();
        let s = self.mu.len();
        let mut out = vec![vec![f64::NEG_INFINITY; s]; t];
        for ti in 0..t {
            let y = x.get(ti, 0);
            for j in 0..s {
                out[ti][j] = log_wrapped_normal(y, self.mu[j], self.sigma[j]);
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

impl Predict for FittedWrappedNormalHmm {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        self.decode(x, session)
    }
}

impl FitUnsupervised for WrappedNormalHmm {
    type Fitted = FittedWrappedNormalHmm;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedWrappedNormalHmm>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let t_len = x.nrows();
        let k = self.n_states.max(1);
        if t_len == 0 {
            return ctx.finish(FittedWrappedNormalHmm {
                labels: empty_labels(0),
                start: init_start(k),
                trans: init_trans(k),
                mu: Vector::zeros(k),
                sigma: Vector::filled(k, 1.0),
                loglik: f64::NAN,
            });
        }
        let mut mu = Vector::from_iter((0..k).map(|j| (j as f64 - 0.5) * 0.8));
        let mut sigma = Vector::filled(k, 1.0);
        let mut start = init_start(k);
        let mut trans = init_trans(k);
        let mut loglik = f64::NEG_INFINITY;
        let mut last_gamma: Vec<Vec<f64>> = Vec::new();
        for it in 0..self.max_iter.max(1) {
            let dummy = FittedWrappedNormalHmm {
                labels: Vector::zeros(0),
                start: start.clone(),
                trans: trans.clone(),
                mu: mu.clone(),
                sigma: sigma.clone(),
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
                let mut cs = 0.0_f64;
                let mut sn = 0.0_f64;
                for t in 0..t_len {
                    let y = x.get(t, 0);
                    if !y.is_finite() {
                        continue;
                    }
                    let w = fb.gamma[t][j];
                    wsum += w;
                    cs += w * y.cos();
                    sn += w * y.sin();
                }
                if wsum > 1e-12 {
                    mu[j] = sn.atan2(cs);
                    let resultant = (cs * cs + sn * sn).sqrt() / wsum;
                    let r = resultant.clamp(1e-8, 1.0 - 1e-8);
                    sigma[j] = ((-2.0 * r.ln()).max(COV_FLOOR)).sqrt();
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
        let dummy = FittedWrappedNormalHmm {
            labels: Vector::zeros(0),
            start: start.clone(),
            trans: trans.clone(),
            mu: mu.clone(),
            sigma: sigma.clone(),
            loglik,
        };
        let (labels, _) = viterbi_path(&start, &trans, &dummy.log_emit_seq(x));
        ctx.finish(FittedWrappedNormalHmm {
            labels,
            start,
            trans,
            mu,
            sigma,
            loglik,
        })
    }
}

/// Kumaraswamy-emission HMM on \((0,1)\).
///
/// State count is not identification `p`. Distinct from [`BetaHmm`]
/// (gamma-function normalizer) and [`LogitNormalHmm`] (Gaussian on logit).
#[derive(Clone, Debug)]
pub struct KumaraswamyHmm {
    /// Hidden states. Not identification `p`.
    pub n_states: usize,
    /// Baum–Welch cap.
    pub max_iter: usize,
}

impl Default for KumaraswamyHmm {
    fn default() -> Self {
        Self {
            n_states: 2,
            max_iter: 40,
        }
    }
}

impl KumaraswamyHmm {
    /// `k`-state Kumaraswamy HMM.
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
    ) -> Result<Qualified<FittedKumaraswamyHmm>> {
        self.fit_unsupervised(x, session)
    }
}

/// Fitted Kumaraswamy HMM.
#[derive(Clone, Debug)]
pub struct FittedKumaraswamyHmm {
    /// Viterbi path.
    pub labels: Vector,
    /// Start distribution.
    pub start: Vector,
    /// Transitions.
    pub trans: Matrix,
    /// First shapes \(a_j\).
    pub a: Vector,
    /// Second shapes \(b_j\).
    pub b: Vector,
    /// Training log-likelihood.
    pub loglik: f64,
}

impl FittedKumaraswamyHmm {
    fn log_emit_seq(&self, x: &Matrix) -> Vec<Vec<f64>> {
        let t = x.nrows();
        let s = self.a.len();
        let mut out = vec![vec![f64::NEG_INFINITY; s]; t];
        for ti in 0..t {
            let y = x.get(ti, 0);
            for j in 0..s {
                out[ti][j] = log_kumaraswamy(y, self.a[j], self.b[j]);
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

impl Predict for FittedKumaraswamyHmm {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        self.decode(x, session)
    }
}

impl FitUnsupervised for KumaraswamyHmm {
    type Fitted = FittedKumaraswamyHmm;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedKumaraswamyHmm>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let t_len = x.nrows();
        let k = self.n_states.max(1);
        if t_len == 0 {
            return ctx.finish(FittedKumaraswamyHmm {
                labels: empty_labels(0),
                start: init_start(k),
                trans: init_trans(k),
                a: Vector::filled(k, 2.0),
                b: Vector::filled(k, 2.0),
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
                    .message("KumaraswamyHmm skipped observations outside (0, 1)")
                    .build(),
            );
        }
        if n_ok < 2 {
            ctx.push(
                Issue::builder(IssueCode::MeaninglessFit)
                    .message("KumaraswamyHmm needs at least two observations in (0, 1)")
                    .meaninglessness(Meaninglessness::vacuous(
                        "Kumaraswamy HMM",
                        "a, b are unidentified without unit-interval data",
                        "collect rates or proportions in (0, 1)",
                    ))
                    .build(),
            );
            return ctx.finish(FittedKumaraswamyHmm {
                labels: empty_labels(t_len),
                start: init_start(k),
                trans: init_trans(k),
                a: Vector::filled(k, 2.0),
                b: Vector::filled(k, 2.0),
                loglik: f64::NAN,
            });
        }
        let mut ashape = Vector::from_iter((0..k).map(|j| 1.5 + 0.5 * j as f64));
        let mut bshape = Vector::from_iter((0..k).map(|j| 1.5 + 0.25 * j as f64));
        let mut start = init_start(k);
        let mut trans = init_trans(k);
        let mut loglik = f64::NEG_INFINITY;
        let mut last_gamma: Vec<Vec<f64>> = Vec::new();
        for it in 0..self.max_iter.max(1) {
            let dummy = FittedKumaraswamyHmm {
                labels: Vector::zeros(0),
                start: start.clone(),
                trans: trans.clone(),
                a: ashape.clone(),
                b: bshape.clone(),
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
                let mut wln = 0.0_f64;
                for t in 0..t_len {
                    let y = x.get(t, 0);
                    if !y.is_finite() || y <= 0.0 || y >= 1.0 {
                        continue;
                    }
                    wsum += fb.gamma[t][j];
                    wln += fb.gamma[t][j] * y.ln();
                }
                if wsum > 1e-12 && wln < 0.0 {
                    ashape[j] = (-wsum / wln).clamp(0.2, 40.0);
                    let mut w1 = 0.0_f64;
                    for t in 0..t_len {
                        let y = x.get(t, 0);
                        if !y.is_finite() || y <= 0.0 || y >= 1.0 {
                            continue;
                        }
                        let ya = y.powf(ashape[j]);
                        w1 += fb.gamma[t][j] * (-(1.0 - ya).max(1e-12).ln());
                    }
                    if w1 > 1e-12 {
                        bshape[j] = (wsum / w1).clamp(0.2, 40.0);
                    }
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
        let dummy = FittedKumaraswamyHmm {
            labels: Vector::zeros(0),
            start: start.clone(),
            trans: trans.clone(),
            a: ashape.clone(),
            b: bshape.clone(),
            loglik,
        };
        let (labels, _) = viterbi_path(&start, &trans, &dummy.log_emit_seq(x));
        ctx.finish(FittedKumaraswamyHmm {
            labels,
            start,
            trans,
            a: ashape,
            b: bshape,
            loglik,
        })
    }
}

/// Log-logistic emission HMM (Fisk).
///
/// State count is not identification `p`. Distinct from [`LogNormalHmm`]
/// (Gaussian on \(\log y\)) and [`LogisticHmm`] (unbounded support).
#[derive(Clone, Debug)]
pub struct LogLogisticHmm {
    /// Hidden states. Not identification `p`.
    pub n_states: usize,
    /// Baum–Welch cap.
    pub max_iter: usize,
}

impl Default for LogLogisticHmm {
    fn default() -> Self {
        Self {
            n_states: 2,
            max_iter: 40,
        }
    }
}

impl LogLogisticHmm {
    /// `k`-state log-logistic HMM.
    pub fn new(n_states: usize) -> Self {
        Self {
            n_states,
            ..Self::default()
        }
    }

    /// Fit alias.
    pub fn fit(&mut self, x: &Matrix, session: &Session) -> Result<Qualified<FittedLogLogisticHmm>> {
        self.fit_unsupervised(x, session)
    }
}

/// Fitted log-logistic HMM.
#[derive(Clone, Debug)]
pub struct FittedLogLogisticHmm {
    /// Viterbi path.
    pub labels: Vector,
    /// Start distribution.
    pub start: Vector,
    /// Transitions.
    pub trans: Matrix,
    /// Scales \(\alpha_j\).
    pub scale: Vector,
    /// Shapes \(\beta_j\).
    pub shape: Vector,
    /// Training log-likelihood.
    pub loglik: f64,
}

impl FittedLogLogisticHmm {
    fn log_emit_seq(&self, x: &Matrix) -> Vec<Vec<f64>> {
        let t = x.nrows();
        let s = self.scale.len();
        let mut out = vec![vec![f64::NEG_INFINITY; s]; t];
        for ti in 0..t {
            let y = x.get(ti, 0);
            for j in 0..s {
                out[ti][j] = log_loglogistic(y, self.scale[j], self.shape[j]);
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

impl Predict for FittedLogLogisticHmm {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        self.decode(x, session)
    }
}

impl FitUnsupervised for LogLogisticHmm {
    type Fitted = FittedLogLogisticHmm;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedLogLogisticHmm>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let t_len = x.nrows();
        let k = self.n_states.max(1);
        if t_len == 0 {
            return ctx.finish(FittedLogLogisticHmm {
                labels: empty_labels(0),
                start: init_start(k),
                trans: init_trans(k),
                scale: Vector::filled(k, 1.0),
                shape: Vector::filled(k, 2.0),
                loglik: f64::NAN,
            });
        }
        let mut n_skip = 0usize;
        for i in 0..t_len {
            if x.get(i, 0).is_finite() && x.get(i, 0) <= 0.0 {
                n_skip += 1;
            }
        }
        if n_skip > 0 {
            ctx.push(
                Issue::builder(IssueCode::NonPositiveSeries)
                    .severity(signlred::Severity::Warning)
                    .message(format!("LogLogisticHmm skipped {n_skip} non-positive observations"))
                    .build(),
            );
        }
        let mut scale = Vector::from_iter((0..k).map(|j| 1.0 + 0.5 * j as f64));
        let mut shape = Vector::from_iter((0..k).map(|j| 2.0 + j as f64));
        let mut start = init_start(k);
        let mut trans = init_trans(k);
        let mut loglik = f64::NEG_INFINITY;
        let mut last_gamma: Vec<Vec<f64>> = Vec::new();
        for it in 0..self.max_iter.max(1) {
            let dummy = FittedLogLogisticHmm {
                labels: Vector::zeros(0),
                start: start.clone(),
                trans: trans.clone(),
                scale: scale.clone(),
                shape: shape.clone(),
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
                let mut wln = 0.0_f64;
                for t in 0..t_len {
                    let y = x.get(t, 0);
                    if !y.is_finite() || y <= 0.0 {
                        continue;
                    }
                    wsum += fb.gamma[t][j];
                    wln += fb.gamma[t][j] * y.ln();
                }
                if wsum > 1e-12 {
                    let m = wln / wsum;
                    scale[j] = m.exp().max(1e-4);
                    let mut wad = 0.0_f64;
                    for t in 0..t_len {
                        let y = x.get(t, 0);
                        if y.is_finite() && y > 0.0 {
                            wad += fb.gamma[t][j] * (y.ln() - m).abs();
                        }
                    }
                    let s = (wad / wsum).max(1e-4);
                    shape[j] = (std::f64::consts::PI / (s * 3.0_f64.sqrt())).clamp(0.2, 80.0);
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
        let dummy = FittedLogLogisticHmm {
            labels: Vector::zeros(0),
            start: start.clone(),
            trans: trans.clone(),
            scale: scale.clone(),
            shape: shape.clone(),
            loglik,
        };
        let (labels, _) = viterbi_path(&start, &trans, &dummy.log_emit_seq(x));
        ctx.finish(FittedLogLogisticHmm {
            labels,
            start,
            trans,
            scale,
            shape,
            loglik,
        })
    }
}

/// Hurdle Poisson HMM (zero process separate from truncated counts).
///
/// State count is not identification `p`. Distinct from [`ZeroInflatedPoissonHmm`]
/// (mixture zeros) and [`PoissonHmm`] (no hurdle).
#[derive(Clone, Debug)]
pub struct HurdlePoissonHmm {
    /// Hidden states. Not identification `p`.
    pub n_states: usize,
    /// Baum–Welch cap.
    pub max_iter: usize,
}

impl Default for HurdlePoissonHmm {
    fn default() -> Self {
        Self {
            n_states: 2,
            max_iter: 40,
        }
    }
}

impl HurdlePoissonHmm {
    /// `k`-state hurdle-Poisson HMM.
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
    ) -> Result<Qualified<FittedHurdlePoissonHmm>> {
        self.fit_unsupervised(x, session)
    }
}

/// Fitted hurdle-Poisson HMM.
#[derive(Clone, Debug)]
pub struct FittedHurdlePoissonHmm {
    /// Viterbi path.
    pub labels: Vector,
    /// Start distribution.
    pub start: Vector,
    /// Transitions.
    pub trans: Matrix,
    /// Zero-hurdle mass.
    pub pi: Vector,
    /// Truncated-Poisson rates.
    pub lam: Vector,
    /// Training log-likelihood.
    pub loglik: f64,
}

impl FittedHurdlePoissonHmm {
    fn log_emit_seq(&self, x: &Matrix) -> Vec<Vec<f64>> {
        let t = x.nrows();
        let s = self.lam.len();
        let mut out = vec![vec![f64::NEG_INFINITY; s]; t];
        for ti in 0..t {
            let y = x.get(ti, 0);
            for j in 0..s {
                out[ti][j] = log_hurdle_poisson(y, self.pi[j], self.lam[j]);
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

impl Predict for FittedHurdlePoissonHmm {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        self.decode(x, session)
    }
}

impl FitUnsupervised for HurdlePoissonHmm {
    type Fitted = FittedHurdlePoissonHmm;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedHurdlePoissonHmm>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let t_len = x.nrows();
        let k = self.n_states.max(1);
        if t_len == 0 {
            return ctx.finish(FittedHurdlePoissonHmm {
                labels: empty_labels(0),
                start: init_start(k),
                trans: init_trans(k),
                pi: Vector::filled(k, 0.2),
                lam: Vector::filled(k, 1.0),
                loglik: f64::NAN,
            });
        }
        let mut pi = Vector::filled(k, 0.2);
        let mut lam = Vector::from_iter((0..k).map(|j| 1.0 + 2.0 * j as f64));
        let mut start = init_start(k);
        let mut trans = init_trans(k);
        let mut loglik = f64::NEG_INFINITY;
        let mut last_gamma: Vec<Vec<f64>> = Vec::new();
        for it in 0..self.max_iter.max(1) {
            let dummy = FittedHurdlePoissonHmm {
                labels: Vector::zeros(0),
                start: start.clone(),
                trans: trans.clone(),
                pi: pi.clone(),
                lam: lam.clone(),
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
                let mut wzero = 0.0_f64;
                let mut wpos = 0.0_f64;
                let mut wy = 0.0_f64;
                for t in 0..t_len {
                    let y = x.get(t, 0);
                    if !y.is_finite() || y < 0.0 {
                        continue;
                    }
                    let w = fb.gamma[t][j];
                    wsum += w;
                    if y.abs() < 1e-12 {
                        wzero += w;
                    } else {
                        wpos += w;
                        wy += w * y;
                    }
                }
                if wsum > 1e-12 {
                    pi[j] = (wzero / wsum).clamp(1e-6, 1.0 - 1e-6);
                }
                if wpos > 1e-12 {
                    lam[j] = (wy / wpos).max(1e-4);
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
        let dummy = FittedHurdlePoissonHmm {
            labels: Vector::zeros(0),
            start: start.clone(),
            trans: trans.clone(),
            pi: pi.clone(),
            lam: lam.clone(),
            loglik,
        };
        let (labels, _) = viterbi_path(&start, &trans, &dummy.log_emit_seq(x));
        ctx.finish(FittedHurdlePoissonHmm {
            labels,
            start,
            trans,
            pi,
            lam,
            loglik,
        })
    }
}

fn log_com_poisson(y: f64, lam: f64, nu: f64) -> f64 {
    if !y.is_finite() || y < 0.0 || lam <= 0.0 || nu <= 0.0 {
        return f64::NEG_INFINITY;
    }
    let y = y.round();
    let mut terms = [f64::NEG_INFINITY; 40];
    for (k, slot) in terms.iter_mut().enumerate() {
        *slot = k as f64 * lam.ln() - nu * crate::special::ln_gamma(k as f64 + 1.0);
    }
    y * lam.ln() - nu * crate::special::ln_gamma(y + 1.0) - logsumexp(&terms)
}

fn log_gev(y: f64, loc: f64, scale: f64, xi: f64) -> f64 {
    if !y.is_finite() || !loc.is_finite() || scale <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if xi.abs() < 1e-8 {
        return log_gumbel(y, loc, scale);
    }
    let t = (1.0 + xi * (y - loc) / scale).max(1e-8);
    -scale.ln() - (1.0 + 1.0 / xi) * t.ln() - t.powf(-1.0 / xi)
}

fn log_slash(y: f64, loc: f64, scale: f64) -> f64 {
    if !y.is_finite() || !loc.is_finite() || scale <= 0.0 {
        return f64::NEG_INFINITY;
    }
    let z = (y - loc) / scale;
    if z.abs() < 1e-6 {
        return -scale.ln() - 0.5 * LN_2PI - std::f64::consts::LN_2;
    }
    let one_minus = 1.0 - (-0.5 * z * z).exp();
    if one_minus <= 0.0 {
        return f64::NEG_INFINITY;
    }
    -scale.ln() - 0.5 * LN_2PI + one_minus.ln() - 2.0 * z.abs().ln()
}

fn log_skew_normal(y: f64, loc: f64, scale: f64, alpha: f64) -> f64 {
    if !y.is_finite() || !loc.is_finite() || scale <= 0.0 || !alpha.is_finite() {
        return f64::NEG_INFINITY;
    }
    let z = (y - loc) / scale;
    let cdf = crate::special::norm_cdf(alpha * z);
    if cdf <= 0.0 {
        return f64::NEG_INFINITY;
    }
    std::f64::consts::LN_2 + log_gauss1(y, loc, scale * scale) + cdf.ln()
}

fn log_discrete_weibull(y: f64, q: f64, beta: f64) -> f64 {
    if !y.is_finite() || y < 0.0 || q <= 0.0 || q >= 1.0 || beta <= 0.0 {
        return f64::NEG_INFINITY;
    }
    let y = y.round().max(0.0);
    let lnq = q.ln();
    let a = y.powf(beta) * lnq;
    let b = (y + 1.0).powf(beta) * lnq;
    a + (1.0 - (b - a).exp()).max(1e-12).ln()
}

fn log_burr(y: f64, c: f64, k: f64, alpha: f64) -> f64 {
    if !y.is_finite() || y <= 0.0 || c <= 0.0 || k <= 0.0 || alpha <= 0.0 {
        return f64::NEG_INFINITY;
    }
    let z = (y / c).max(1e-12);
    let zk = z.powf(k);
    alpha.ln() + k.ln() - c.ln() + (k - 1.0) * z.ln() - (alpha + 1.0) * (1.0 + zk).ln()
}

/// Conway–Maxwell–Poisson HMM (dispersion \(\nu\)).
///
/// Dispersion is not identification `p`. Distinct from [`PoissonHmm`]
/// (\(\nu=1\)) and [`NegativeBinomialHmm`] (overdispersion only).
#[derive(Clone, Debug)]
pub struct ComPoissonHmm {
    /// Hidden states. Not identification `p`.
    pub n_states: usize,
    /// Baum–Welch cap.
    pub max_iter: usize,
}

impl Default for ComPoissonHmm {
    fn default() -> Self {
        Self {
            n_states: 2,
            max_iter: 40,
        }
    }
}

impl ComPoissonHmm {
    /// `k`-state COM-Poisson HMM.
    pub fn new(n_states: usize) -> Self {
        Self {
            n_states,
            ..Self::default()
        }
    }

    /// Fit alias.
    pub fn fit(&mut self, x: &Matrix, session: &Session) -> Result<Qualified<FittedComPoissonHmm>> {
        self.fit_unsupervised(x, session)
    }
}

/// Fitted COM-Poisson HMM.
#[derive(Clone, Debug)]
pub struct FittedComPoissonHmm {
    /// Viterbi path.
    pub labels: Vector,
    /// Start distribution.
    pub start: Vector,
    /// Transitions.
    pub trans: Matrix,
    /// Intensities \(\lambda_j\).
    pub lam: Vector,
    /// Dispersions \(\nu_j\).
    pub nu: Vector,
    /// Training log-likelihood.
    pub loglik: f64,
}

impl FittedComPoissonHmm {
    fn log_emit_seq(&self, x: &Matrix) -> Vec<Vec<f64>> {
        let t = x.nrows();
        let s = self.lam.len();
        let mut out = vec![vec![f64::NEG_INFINITY; s]; t];
        for ti in 0..t {
            let y = x.get(ti, 0);
            for j in 0..s {
                out[ti][j] = log_com_poisson(y, self.lam[j], self.nu[j]);
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

impl Predict for FittedComPoissonHmm {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        self.decode(x, session)
    }
}

impl FitUnsupervised for ComPoissonHmm {
    type Fitted = FittedComPoissonHmm;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedComPoissonHmm>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let t_len = x.nrows();
        let k = self.n_states.max(1);
        if t_len == 0 {
            return ctx.finish(FittedComPoissonHmm {
                labels: empty_labels(0),
                start: init_start(k),
                trans: init_trans(k),
                lam: Vector::filled(k, 1.0),
                nu: Vector::filled(k, 1.0),
                loglik: f64::NAN,
            });
        }
        let mut lam = Vector::from_iter((0..k).map(|j| 1.0 + 2.0 * j as f64));
        let mut nu = Vector::filled(k, 1.0);
        let mut start = init_start(k);
        let mut trans = init_trans(k);
        let mut loglik = f64::NEG_INFINITY;
        let mut last_gamma: Vec<Vec<f64>> = Vec::new();
        for it in 0..self.max_iter.max(1) {
            let dummy = FittedComPoissonHmm {
                labels: Vector::zeros(0),
                start: start.clone(),
                trans: trans.clone(),
                lam: lam.clone(),
                nu: nu.clone(),
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
                    if !y.is_finite() || y < 0.0 {
                        continue;
                    }
                    let w = fb.gamma[t][j];
                    wsum += w;
                    wy += w * y;
                    wy2 += w * y * y;
                }
                if wsum > 1e-12 {
                    let m = (wy / wsum).max(1e-4);
                    lam[j] = m.min(40.0);
                    let v = (wy2 / wsum - m * m).max(1e-4);
                    nu[j] = (m / v).clamp(0.2, 8.0);
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
        let dummy = FittedComPoissonHmm {
            labels: Vector::zeros(0),
            start: start.clone(),
            trans: trans.clone(),
            lam: lam.clone(),
            nu: nu.clone(),
            loglik,
        };
        let (labels, _) = viterbi_path(&start, &trans, &dummy.log_emit_seq(x));
        ctx.finish(FittedComPoissonHmm {
            labels,
            start,
            trans,
            lam,
            nu,
            loglik,
        })
    }
}

/// Generalized extreme-value HMM (shape \(\xi\) is a hyperparameter).
///
/// \(\xi\) is not identification `p`. Distinct from [`GumbelHmm`] (\(\xi=0\))
/// and [`FrechetHmm`] (type-II only).
#[derive(Clone, Debug)]
pub struct GevHmm {
    /// Hidden states. Not identification `p`.
    pub n_states: usize,
    /// GEV shape. Not identification `p`.
    pub xi: f64,
    /// Baum–Welch cap.
    pub max_iter: usize,
}

impl Default for GevHmm {
    fn default() -> Self {
        Self {
            n_states: 2,
            xi: 0.2,
            max_iter: 40,
        }
    }
}

impl GevHmm {
    /// `k`-state GEV HMM.
    pub fn new(n_states: usize) -> Self {
        Self {
            n_states,
            ..Self::default()
        }
    }

    /// Fit alias.
    pub fn fit(&mut self, x: &Matrix, session: &Session) -> Result<Qualified<FittedGevHmm>> {
        self.fit_unsupervised(x, session)
    }
}

/// Fitted GEV HMM.
#[derive(Clone, Debug)]
pub struct FittedGevHmm {
    /// Viterbi path.
    pub labels: Vector,
    /// Start distribution.
    pub start: Vector,
    /// Transitions.
    pub trans: Matrix,
    /// Locations.
    pub loc: Vector,
    /// Scales.
    pub scale: Vector,
    /// Shared shape.
    pub xi: f64,
    /// Training log-likelihood.
    pub loglik: f64,
}

impl FittedGevHmm {
    fn log_emit_seq(&self, x: &Matrix) -> Vec<Vec<f64>> {
        let t = x.nrows();
        let s = self.loc.len();
        let mut out = vec![vec![f64::NEG_INFINITY; s]; t];
        for ti in 0..t {
            let y = x.get(ti, 0);
            for j in 0..s {
                out[ti][j] = log_gev(y, self.loc[j], self.scale[j], self.xi);
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

impl Predict for FittedGevHmm {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        self.decode(x, session)
    }
}

impl FitUnsupervised for GevHmm {
    type Fitted = FittedGevHmm;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedGevHmm>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let t_len = x.nrows();
        let k = self.n_states.max(1);
        let xi = if self.xi.is_finite() { self.xi } else { 0.2 };
        if t_len == 0 {
            return ctx.finish(FittedGevHmm {
                labels: empty_labels(0),
                start: init_start(k),
                trans: init_trans(k),
                loc: Vector::zeros(k),
                scale: Vector::filled(k, 1.0),
                xi,
                loglik: f64::NAN,
            });
        }
        let mean = x.column(0).mean();
        let sd = x.column(0).std().max(0.1);
        let mut loc = Vector::from_iter((0..k).map(|j| mean + (j as f64 - 0.5) * sd));
        let mut scale = Vector::filled(k, sd);
        let mut start = init_start(k);
        let mut trans = init_trans(k);
        let mut loglik = f64::NEG_INFINITY;
        let mut last_gamma: Vec<Vec<f64>> = Vec::new();
        for it in 0..self.max_iter.max(1) {
            let dummy = FittedGevHmm {
                labels: Vector::zeros(0),
                start: start.clone(),
                trans: trans.clone(),
                loc: loc.clone(),
                scale: scale.clone(),
                xi,
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
                    loc[j] = wy / wsum;
                    let mut wad = 0.0_f64;
                    for t in 0..t_len {
                        let y = x.get(t, 0);
                        if y.is_finite() {
                            wad += fb.gamma[t][j] * (y - loc[j]).abs();
                        }
                    }
                    scale[j] = (wad / wsum).max(COV_FLOOR.sqrt());
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
        let dummy = FittedGevHmm {
            labels: Vector::zeros(0),
            start: start.clone(),
            trans: trans.clone(),
            loc: loc.clone(),
            scale: scale.clone(),
            xi,
            loglik,
        };
        let (labels, _) = viterbi_path(&start, &trans, &dummy.log_emit_seq(x));
        ctx.finish(FittedGevHmm {
            labels,
            start,
            trans,
            loc,
            scale,
            xi,
            loglik,
        })
    }
}

/// Slash-emission HMM (\(N/U\) ratio, heavier than Cauchy near the origin).
///
/// State count is not identification `p`. Distinct from [`CauchyHmm`]
/// (undefined mean) and [`StudentTHmm`] (finite \(\nu\)).
#[derive(Clone, Debug)]
pub struct SlashHmm {
    /// Hidden states. Not identification `p`.
    pub n_states: usize,
    /// Baum–Welch cap.
    pub max_iter: usize,
}

impl Default for SlashHmm {
    fn default() -> Self {
        Self {
            n_states: 2,
            max_iter: 40,
        }
    }
}

impl SlashHmm {
    /// `k`-state slash HMM.
    pub fn new(n_states: usize) -> Self {
        Self {
            n_states,
            ..Self::default()
        }
    }

    /// Fit alias.
    pub fn fit(&mut self, x: &Matrix, session: &Session) -> Result<Qualified<FittedSlashHmm>> {
        self.fit_unsupervised(x, session)
    }
}

/// Fitted slash HMM.
#[derive(Clone, Debug)]
pub struct FittedSlashHmm {
    /// Viterbi path.
    pub labels: Vector,
    /// Start distribution.
    pub start: Vector,
    /// Transitions.
    pub trans: Matrix,
    /// Locations.
    pub loc: Vector,
    /// Scales.
    pub scale: Vector,
    /// Training log-likelihood.
    pub loglik: f64,
}

impl FittedSlashHmm {
    fn log_emit_seq(&self, x: &Matrix) -> Vec<Vec<f64>> {
        let t = x.nrows();
        let s = self.loc.len();
        let mut out = vec![vec![f64::NEG_INFINITY; s]; t];
        for ti in 0..t {
            let y = x.get(ti, 0);
            for j in 0..s {
                out[ti][j] = log_slash(y, self.loc[j], self.scale[j]);
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

impl Predict for FittedSlashHmm {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        self.decode(x, session)
    }
}

impl FitUnsupervised for SlashHmm {
    type Fitted = FittedSlashHmm;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedSlashHmm>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let t_len = x.nrows();
        let k = self.n_states.max(1);
        if t_len == 0 {
            return ctx.finish(FittedSlashHmm {
                labels: empty_labels(0),
                start: init_start(k),
                trans: init_trans(k),
                loc: Vector::zeros(k),
                scale: Vector::filled(k, 1.0),
                loglik: f64::NAN,
            });
        }
        let mean = x.column(0).mean();
        let sd = x.column(0).std().max(0.1);
        let mut loc = Vector::from_iter((0..k).map(|j| mean + (j as f64 - 0.5) * sd));
        let mut scale = Vector::filled(k, sd);
        let mut start = init_start(k);
        let mut trans = init_trans(k);
        let mut loglik = f64::NEG_INFINITY;
        let mut last_gamma: Vec<Vec<f64>> = Vec::new();
        for it in 0..self.max_iter.max(1) {
            let dummy = FittedSlashHmm {
                labels: Vector::zeros(0),
                start: start.clone(),
                trans: trans.clone(),
                loc: loc.clone(),
                scale: scale.clone(),
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
                    loc[j] = wy / wsum;
                    let mut wad = 0.0_f64;
                    for t in 0..t_len {
                        let y = x.get(t, 0);
                        if y.is_finite() {
                            wad += fb.gamma[t][j] * (y - loc[j]).abs();
                        }
                    }
                    scale[j] = (wad / wsum).max(COV_FLOOR.sqrt());
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
        let dummy = FittedSlashHmm {
            labels: Vector::zeros(0),
            start: start.clone(),
            trans: trans.clone(),
            loc: loc.clone(),
            scale: scale.clone(),
            loglik,
        };
        let (labels, _) = viterbi_path(&start, &trans, &dummy.log_emit_seq(x));
        ctx.finish(FittedSlashHmm {
            labels,
            start,
            trans,
            loc,
            scale,
            loglik,
        })
    }
}

/// Skew-normal HMM (Azzalini shape \(\alpha\) is a hyperparameter).
///
/// \(\alpha\) is not identification `p`. Distinct from [`GaussianHmm`]
/// (symmetric) and [`GumbelHmm`] (extreme-value skew).
#[derive(Clone, Debug)]
pub struct SkewNormalHmm {
    /// Hidden states. Not identification `p`.
    pub n_states: usize,
    /// Skewness shape. Not identification `p`.
    pub alpha: f64,
    /// Baum–Welch cap.
    pub max_iter: usize,
}

impl Default for SkewNormalHmm {
    fn default() -> Self {
        Self {
            n_states: 2,
            alpha: 2.0,
            max_iter: 40,
        }
    }
}

impl SkewNormalHmm {
    /// `k`-state skew-normal HMM.
    pub fn new(n_states: usize) -> Self {
        Self {
            n_states,
            ..Self::default()
        }
    }

    /// Fit alias.
    pub fn fit(&mut self, x: &Matrix, session: &Session) -> Result<Qualified<FittedSkewNormalHmm>> {
        self.fit_unsupervised(x, session)
    }
}

/// Fitted skew-normal HMM.
#[derive(Clone, Debug)]
pub struct FittedSkewNormalHmm {
    /// Viterbi path.
    pub labels: Vector,
    /// Start distribution.
    pub start: Vector,
    /// Transitions.
    pub trans: Matrix,
    /// Locations.
    pub loc: Vector,
    /// Scales.
    pub scale: Vector,
    /// Shared skewness.
    pub alpha: f64,
    /// Training log-likelihood.
    pub loglik: f64,
}

impl FittedSkewNormalHmm {
    fn log_emit_seq(&self, x: &Matrix) -> Vec<Vec<f64>> {
        let t = x.nrows();
        let s = self.loc.len();
        let mut out = vec![vec![f64::NEG_INFINITY; s]; t];
        for ti in 0..t {
            let y = x.get(ti, 0);
            for j in 0..s {
                out[ti][j] = log_skew_normal(y, self.loc[j], self.scale[j], self.alpha);
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

impl Predict for FittedSkewNormalHmm {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        self.decode(x, session)
    }
}

impl FitUnsupervised for SkewNormalHmm {
    type Fitted = FittedSkewNormalHmm;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedSkewNormalHmm>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let t_len = x.nrows();
        let k = self.n_states.max(1);
        let alpha = if self.alpha.is_finite() {
            self.alpha
        } else {
            2.0
        };
        if t_len == 0 {
            return ctx.finish(FittedSkewNormalHmm {
                labels: empty_labels(0),
                start: init_start(k),
                trans: init_trans(k),
                loc: Vector::zeros(k),
                scale: Vector::filled(k, 1.0),
                alpha,
                loglik: f64::NAN,
            });
        }
        let mean = x.column(0).mean();
        let sd = x.column(0).std().max(0.1);
        let mut loc = Vector::from_iter((0..k).map(|j| mean + (j as f64 - 0.5) * sd));
        let mut scale = Vector::filled(k, sd);
        let mut start = init_start(k);
        let mut trans = init_trans(k);
        let mut loglik = f64::NEG_INFINITY;
        let mut last_gamma: Vec<Vec<f64>> = Vec::new();
        for it in 0..self.max_iter.max(1) {
            let dummy = FittedSkewNormalHmm {
                labels: Vector::zeros(0),
                start: start.clone(),
                trans: trans.clone(),
                loc: loc.clone(),
                scale: scale.clone(),
                alpha,
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
                    if !y.is_finite() {
                        continue;
                    }
                    let w = fb.gamma[t][j];
                    wsum += w;
                    wy += w * y;
                    wy2 += w * y * y;
                }
                if wsum > 1e-12 {
                    loc[j] = wy / wsum;
                    scale[j] = ((wy2 / wsum - loc[j] * loc[j]).max(COV_FLOOR)).sqrt();
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
        let dummy = FittedSkewNormalHmm {
            labels: Vector::zeros(0),
            start: start.clone(),
            trans: trans.clone(),
            loc: loc.clone(),
            scale: scale.clone(),
            alpha,
            loglik,
        };
        let (labels, _) = viterbi_path(&start, &trans, &dummy.log_emit_seq(x));
        ctx.finish(FittedSkewNormalHmm {
            labels,
            start,
            trans,
            loc,
            scale,
            alpha,
            loglik,
        })
    }
}

/// Discrete-Weibull HMM on non-negative integers.
///
/// Shape \(\beta\) is not identification `p`. Distinct from [`GeometricHmm`]
/// (\(\beta=1\)) and [`WeibullHmm`] (continuous).
#[derive(Clone, Debug)]
pub struct DiscreteWeibullHmm {
    /// Hidden states. Not identification `p`.
    pub n_states: usize,
    /// Baum–Welch cap.
    pub max_iter: usize,
}

impl Default for DiscreteWeibullHmm {
    fn default() -> Self {
        Self {
            n_states: 2,
            max_iter: 40,
        }
    }
}

impl DiscreteWeibullHmm {
    /// `k`-state discrete-Weibull HMM.
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
    ) -> Result<Qualified<FittedDiscreteWeibullHmm>> {
        self.fit_unsupervised(x, session)
    }
}

/// Fitted discrete-Weibull HMM.
#[derive(Clone, Debug)]
pub struct FittedDiscreteWeibullHmm {
    /// Viterbi path.
    pub labels: Vector,
    /// Start distribution.
    pub start: Vector,
    /// Transitions.
    pub trans: Matrix,
    /// Survival bases \(q_j\in(0,1)\).
    pub q: Vector,
    /// Shapes \(\beta_j\).
    pub beta: Vector,
    /// Training log-likelihood.
    pub loglik: f64,
}

impl FittedDiscreteWeibullHmm {
    fn log_emit_seq(&self, x: &Matrix) -> Vec<Vec<f64>> {
        let t = x.nrows();
        let s = self.q.len();
        let mut out = vec![vec![f64::NEG_INFINITY; s]; t];
        for ti in 0..t {
            let y = x.get(ti, 0);
            for j in 0..s {
                out[ti][j] = log_discrete_weibull(y, self.q[j], self.beta[j]);
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

impl Predict for FittedDiscreteWeibullHmm {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        self.decode(x, session)
    }
}

impl FitUnsupervised for DiscreteWeibullHmm {
    type Fitted = FittedDiscreteWeibullHmm;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedDiscreteWeibullHmm>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let t_len = x.nrows();
        let k = self.n_states.max(1);
        if t_len == 0 {
            return ctx.finish(FittedDiscreteWeibullHmm {
                labels: empty_labels(0),
                start: init_start(k),
                trans: init_trans(k),
                q: Vector::filled(k, 0.5),
                beta: Vector::filled(k, 1.0),
                loglik: f64::NAN,
            });
        }
        let mut q = Vector::from_iter((0..k).map(|j| 0.4 + 0.2 * j as f64));
        let mut beta = Vector::filled(k, 1.0);
        let mut start = init_start(k);
        let mut trans = init_trans(k);
        let mut loglik = f64::NEG_INFINITY;
        let mut last_gamma: Vec<Vec<f64>> = Vec::new();
        for it in 0..self.max_iter.max(1) {
            let dummy = FittedDiscreteWeibullHmm {
                labels: Vector::zeros(0),
                start: start.clone(),
                trans: trans.clone(),
                q: q.clone(),
                beta: beta.clone(),
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
                let mut wzero = 0.0_f64;
                let mut wy = 0.0_f64;
                for t in 0..t_len {
                    let y = x.get(t, 0);
                    if !y.is_finite() || y < 0.0 {
                        continue;
                    }
                    let w = fb.gamma[t][j];
                    wsum += w;
                    if y.abs() < 1e-12 {
                        wzero += w;
                    }
                    wy += w * y;
                }
                if wsum > 1e-12 {
                    let p0 = (wzero / wsum).clamp(1e-4, 1.0 - 1e-4);
                    q[j] = (1.0 - p0).clamp(1e-3, 1.0 - 1e-3);
                    let m = (wy / wsum).max(1e-4);
                    beta[j] = (1.0 / m.ln().abs().max(0.2)).clamp(0.2, 8.0);
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
        let dummy = FittedDiscreteWeibullHmm {
            labels: Vector::zeros(0),
            start: start.clone(),
            trans: trans.clone(),
            q: q.clone(),
            beta: beta.clone(),
            loglik,
        };
        let (labels, _) = viterbi_path(&start, &trans, &dummy.log_emit_seq(x));
        ctx.finish(FittedDiscreteWeibullHmm {
            labels,
            start,
            trans,
            q,
            beta,
            loglik,
        })
    }
}

/// Burr type-XII HMM (two shapes on a positive scale).
///
/// Shapes are not identification `p`. Distinct from [`LogLogisticHmm`]
/// (\(\alpha=1\)) and [`ParetoHmm`] (one shape).
#[derive(Clone, Debug)]
pub struct BurrHmm {
    /// Hidden states. Not identification `p`.
    pub n_states: usize,
    /// Baum–Welch cap.
    pub max_iter: usize,
}

impl Default for BurrHmm {
    fn default() -> Self {
        Self {
            n_states: 2,
            max_iter: 40,
        }
    }
}

impl BurrHmm {
    /// `k`-state Burr HMM.
    pub fn new(n_states: usize) -> Self {
        Self {
            n_states,
            ..Self::default()
        }
    }

    /// Fit alias.
    pub fn fit(&mut self, x: &Matrix, session: &Session) -> Result<Qualified<FittedBurrHmm>> {
        self.fit_unsupervised(x, session)
    }
}

/// Fitted Burr HMM.
#[derive(Clone, Debug)]
pub struct FittedBurrHmm {
    /// Viterbi path.
    pub labels: Vector,
    /// Start distribution.
    pub start: Vector,
    /// Transitions.
    pub trans: Matrix,
    /// Scales \(c_j\).
    pub scale: Vector,
    /// First shapes \(k_j\).
    pub k: Vector,
    /// Second shapes \(\alpha_j\).
    pub alpha: Vector,
    /// Training log-likelihood.
    pub loglik: f64,
}

impl FittedBurrHmm {
    fn log_emit_seq(&self, x: &Matrix) -> Vec<Vec<f64>> {
        let t = x.nrows();
        let s = self.scale.len();
        let mut out = vec![vec![f64::NEG_INFINITY; s]; t];
        for ti in 0..t {
            let y = x.get(ti, 0);
            for j in 0..s {
                out[ti][j] = log_burr(y, self.scale[j], self.k[j], self.alpha[j]);
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

impl Predict for FittedBurrHmm {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        self.decode(x, session)
    }
}

impl FitUnsupervised for BurrHmm {
    type Fitted = FittedBurrHmm;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedBurrHmm>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let t_len = x.nrows();
        let kst = self.n_states.max(1);
        if t_len == 0 {
            return ctx.finish(FittedBurrHmm {
                labels: empty_labels(0),
                start: init_start(kst),
                trans: init_trans(kst),
                scale: Vector::filled(kst, 1.0),
                k: Vector::filled(kst, 2.0),
                alpha: Vector::filled(kst, 1.0),
                loglik: f64::NAN,
            });
        }
        let mut n_skip = 0usize;
        for i in 0..t_len {
            if x.get(i, 0).is_finite() && x.get(i, 0) <= 0.0 {
                n_skip += 1;
            }
        }
        if n_skip > 0 {
            ctx.push(
                Issue::builder(IssueCode::NonPositiveSeries)
                    .severity(signlred::Severity::Warning)
                    .message(format!("BurrHmm skipped {n_skip} non-positive observations"))
                    .build(),
            );
        }
        let mut scale = Vector::from_iter((0..kst).map(|j| 1.0 + 0.5 * j as f64));
        let mut kpar = Vector::from_iter((0..kst).map(|j| 1.5 + 0.5 * j as f64));
        let mut alpha = Vector::filled(kst, 1.0);
        let mut start = init_start(kst);
        let mut trans = init_trans(kst);
        let mut loglik = f64::NEG_INFINITY;
        let mut last_gamma: Vec<Vec<f64>> = Vec::new();
        for it in 0..self.max_iter.max(1) {
            let dummy = FittedBurrHmm {
                labels: Vector::zeros(0),
                start: start.clone(),
                trans: trans.clone(),
                scale: scale.clone(),
                k: kpar.clone(),
                alpha: alpha.clone(),
                loglik,
            };
            let Some(fb) = scaled_forward_backward(&mut ctx, &start, &trans, &dummy.log_emit_seq(x))
            else {
                break;
            };
            loglik = fb.loglik;
            last_gamma = fb.gamma.clone();
            ctx.session.step(it as u64, -loglik, None);
            for j in 0..kst {
                let mut wsum = 0.0_f64;
                let mut wln = 0.0_f64;
                for t in 0..t_len {
                    let y = x.get(t, 0);
                    if !y.is_finite() || y <= 0.0 {
                        continue;
                    }
                    wsum += fb.gamma[t][j];
                    wln += fb.gamma[t][j] * y.ln();
                }
                if wsum > 1e-12 {
                    let m = wln / wsum;
                    scale[j] = m.exp().max(1e-4);
                    let mut wad = 0.0_f64;
                    let mut wtail = 0.0_f64;
                    for t in 0..t_len {
                        let y = x.get(t, 0);
                        if !y.is_finite() || y <= 0.0 {
                            continue;
                        }
                        wad += fb.gamma[t][j] * (y.ln() - m).abs();
                        let zk = (y / scale[j]).max(1e-12).powf(kpar[j]);
                        wtail += fb.gamma[t][j] * (1.0 + zk).ln();
                    }
                    let s = (wad / wsum).max(1e-4);
                    kpar[j] = (std::f64::consts::PI / (s * 3.0_f64.sqrt())).clamp(0.2, 40.0);
                    if wtail > 1e-12 {
                        alpha[j] = (wsum / wtail).clamp(0.2, 40.0);
                    }
                }
            }
            let (ns, ntr) = hmm_em_trans(&fb.xi, &fb.gamma[0], kst, t_len);
            start = ns;
            trans = ntr;
        }
        if !last_gamma.is_empty() {
            let occup: Vec<f64> = (0..kst)
                .map(|j| last_gamma.iter().map(|g| g.get(j).copied().unwrap_or(0.0)).sum())
                .collect();
            diagnose_chain(&mut ctx, &start, &trans, &occup);
        }
        let dummy = FittedBurrHmm {
            labels: Vector::zeros(0),
            start: start.clone(),
            trans: trans.clone(),
            scale: scale.clone(),
            k: kpar.clone(),
            alpha: alpha.clone(),
            loglik,
        };
        let (labels, _) = viterbi_path(&start, &trans, &dummy.log_emit_seq(x));
        ctx.finish(FittedBurrHmm {
            labels,
            start,
            trans,
            scale,
            k: kpar,
            alpha,
            loglik,
        })
    }
}

fn log_in(n: i32, x: f64) -> f64 {
    let n = n.unsigned_abs() as usize;
    if n == 0 {
        return log_i0(x);
    }
    if !x.is_finite() || x <= 0.0 {
        return f64::NEG_INFINITY;
    }
    let mut terms = [f64::NEG_INFINITY; 32];
    let lx = (0.25 * x * x).ln();
    let mut lfak = 0.0_f64;
    let mut lfan = crate::special::ln_gamma(n as f64 + 1.0);
    for (k, slot) in terms.iter_mut().enumerate() {
        *slot = k as f64 * lx - lfak - lfan;
        let kf = (k + 1) as f64;
        lfak += kf.ln();
        lfan += (n as f64 + kf).ln();
    }
    n as f64 * (0.5 * x).ln() + logsumexp(&terms)
}

fn log_skellam(k: f64, mu1: f64, mu2: f64) -> f64 {
    if !k.is_finite() || mu1 <= 0.0 || mu2 <= 0.0 {
        return f64::NEG_INFINITY;
    }
    let k = k.round();
    let ki = k as i32;
    -(mu1 + mu2) + 0.5 * k * (mu1 / mu2).ln() + log_in(ki.unsigned_abs() as i32, 2.0 * (mu1 * mu2).sqrt())
}

fn log_levy(y: f64, loc: f64, scale: f64) -> f64 {
    if !y.is_finite() || y <= loc || scale <= 0.0 {
        return f64::NEG_INFINITY;
    }
    let d = y - loc;
    0.5 * (scale.ln() - LN_2PI) - 1.5 * d.ln() - scale / (2.0 * d)
}

fn log_cardioid(y: f64, mu: f64, rho: f64) -> f64 {
    if !y.is_finite() || !mu.is_finite() || rho < 0.0 || rho > 0.5 {
        return f64::NEG_INFINITY;
    }
    let u = 1.0 + 2.0 * rho * (y - mu).cos();
    if u <= 0.0 {
        return f64::NEG_INFINITY;
    }
    -LN_2PI + u.ln()
}

fn log_zig(y: f64, pi: f64, shape: f64, rate: f64) -> f64 {
    if !y.is_finite() || y < 0.0 || shape <= 0.0 || rate <= 0.0 {
        return f64::NEG_INFINITY;
    }
    let pi = pi.clamp(1e-6, 1.0 - 1e-6);
    if y.abs() < 1e-12 {
        pi.ln()
    } else {
        (1.0 - pi).ln() + log_gamma_emit(y, shape, rate)
    }
}

fn log_gengamma(y: f64, scale: f64, d: f64, power: f64) -> f64 {
    if !y.is_finite() || y <= 0.0 || scale <= 0.0 || d <= 0.0 || power <= 0.0 {
        return f64::NEG_INFINITY;
    }
    power.ln() - d * scale.ln() - crate::special::ln_gamma(d / power) + (d - 1.0) * y.ln()
        - (y / scale).powf(power)
}

fn log_gen_poisson(y: f64, lam: f64, xi: f64) -> f64 {
    if !y.is_finite() || y < 0.0 || lam <= 0.0 {
        return f64::NEG_INFINITY;
    }
    let y = y.round();
    let t = lam + y * xi;
    if t <= 0.0 {
        return f64::NEG_INFINITY;
    }
    lam.ln() + (y - 1.0) * t.ln() - t - crate::special::ln_gamma(y + 1.0)
}

/// Skellam HMM (difference of two Poissons).
///
/// State count is not identification `p`. Distinct from [`PoissonHmm`]
/// (one rate) and [`ComPoissonHmm`] (single-count dispersion).
#[derive(Clone, Debug)]
pub struct SkellamHmm {
    /// Hidden states. Not identification `p`.
    pub n_states: usize,
    /// Baum–Welch cap.
    pub max_iter: usize,
}

impl Default for SkellamHmm {
    fn default() -> Self {
        Self {
            n_states: 2,
            max_iter: 40,
        }
    }
}

impl SkellamHmm {
    /// `k`-state Skellam HMM.
    pub fn new(n_states: usize) -> Self {
        Self {
            n_states,
            ..Self::default()
        }
    }

    /// Fit alias.
    pub fn fit(&mut self, x: &Matrix, session: &Session) -> Result<Qualified<FittedSkellamHmm>> {
        self.fit_unsupervised(x, session)
    }
}

/// Fitted Skellam HMM.
#[derive(Clone, Debug)]
pub struct FittedSkellamHmm {
    /// Viterbi path.
    pub labels: Vector,
    /// Start distribution.
    pub start: Vector,
    /// Transitions.
    pub trans: Matrix,
    /// First Poisson means.
    pub mu1: Vector,
    /// Second Poisson means.
    pub mu2: Vector,
    /// Training log-likelihood.
    pub loglik: f64,
}

impl FittedSkellamHmm {
    fn log_emit_seq(&self, x: &Matrix) -> Vec<Vec<f64>> {
        let t = x.nrows();
        let s = self.mu1.len();
        let mut out = vec![vec![f64::NEG_INFINITY; s]; t];
        for ti in 0..t {
            let y = x.get(ti, 0);
            for j in 0..s {
                out[ti][j] = log_skellam(y, self.mu1[j], self.mu2[j]);
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

impl Predict for FittedSkellamHmm {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        self.decode(x, session)
    }
}

impl FitUnsupervised for SkellamHmm {
    type Fitted = FittedSkellamHmm;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedSkellamHmm>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let t_len = x.nrows();
        let k = self.n_states.max(1);
        if t_len == 0 {
            return ctx.finish(FittedSkellamHmm {
                labels: empty_labels(0),
                start: init_start(k),
                trans: init_trans(k),
                mu1: Vector::filled(k, 1.0),
                mu2: Vector::filled(k, 1.0),
                loglik: f64::NAN,
            });
        }
        let mut mu1 = Vector::from_iter((0..k).map(|j| 1.0 + 2.0 * j as f64));
        let mut mu2 = Vector::filled(k, 0.5);
        let mut start = init_start(k);
        let mut trans = init_trans(k);
        let mut loglik = f64::NEG_INFINITY;
        let mut last_gamma: Vec<Vec<f64>> = Vec::new();
        for it in 0..self.max_iter.max(1) {
            let dummy = FittedSkellamHmm {
                labels: Vector::zeros(0),
                start: start.clone(),
                trans: trans.clone(),
                mu1: mu1.clone(),
                mu2: mu2.clone(),
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
                    if !y.is_finite() {
                        continue;
                    }
                    let w = fb.gamma[t][j];
                    wsum += w;
                    wy += w * y;
                    wy2 += w * y * y;
                }
                if wsum > 1e-12 {
                    let m = wy / wsum;
                    let v = (wy2 / wsum - m * m).max(1e-4);
                    let a = (0.5 * (v + m)).max(1e-4);
                    let b = (0.5 * (v - m)).max(1e-4);
                    mu1[j] = a.min(40.0);
                    mu2[j] = b.min(40.0);
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
        let dummy = FittedSkellamHmm {
            labels: Vector::zeros(0),
            start: start.clone(),
            trans: trans.clone(),
            mu1: mu1.clone(),
            mu2: mu2.clone(),
            loglik,
        };
        let (labels, _) = viterbi_path(&start, &trans, &dummy.log_emit_seq(x));
        ctx.finish(FittedSkellamHmm {
            labels,
            start,
            trans,
            mu1,
            mu2,
            loglik,
        })
    }
}

/// Lévy-emission HMM (stable \(\alpha=1/2\), support \(y>\mu\)).
///
/// State count is not identification `p`. Distinct from [`InverseGaussianHmm`]
/// (Wald) and [`InverseGammaHmm`] (reciprocal-gamma).
#[derive(Clone, Debug)]
pub struct LevyHmm {
    /// Hidden states. Not identification `p`.
    pub n_states: usize,
    /// Baum–Welch cap.
    pub max_iter: usize,
}

impl Default for LevyHmm {
    fn default() -> Self {
        Self {
            n_states: 2,
            max_iter: 40,
        }
    }
}

impl LevyHmm {
    /// `k`-state Lévy HMM.
    pub fn new(n_states: usize) -> Self {
        Self {
            n_states,
            ..Self::default()
        }
    }

    /// Fit alias.
    pub fn fit(&mut self, x: &Matrix, session: &Session) -> Result<Qualified<FittedLevyHmm>> {
        self.fit_unsupervised(x, session)
    }
}

/// Fitted Lévy HMM.
#[derive(Clone, Debug)]
pub struct FittedLevyHmm {
    /// Viterbi path.
    pub labels: Vector,
    /// Start distribution.
    pub start: Vector,
    /// Transitions.
    pub trans: Matrix,
    /// Thresholds \(\mu_j\).
    pub loc: Vector,
    /// Scales \(c_j\).
    pub scale: Vector,
    /// Training log-likelihood.
    pub loglik: f64,
}

impl FittedLevyHmm {
    fn log_emit_seq(&self, x: &Matrix) -> Vec<Vec<f64>> {
        let t = x.nrows();
        let s = self.loc.len();
        let mut out = vec![vec![f64::NEG_INFINITY; s]; t];
        for ti in 0..t {
            let y = x.get(ti, 0);
            for j in 0..s {
                out[ti][j] = log_levy(y, self.loc[j], self.scale[j]);
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

impl Predict for FittedLevyHmm {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        self.decode(x, session)
    }
}

impl FitUnsupervised for LevyHmm {
    type Fitted = FittedLevyHmm;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedLevyHmm>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let t_len = x.nrows();
        let k = self.n_states.max(1);
        if t_len == 0 {
            return ctx.finish(FittedLevyHmm {
                labels: empty_labels(0),
                start: init_start(k),
                trans: init_trans(k),
                loc: Vector::filled(k, -1.0),
                scale: Vector::filled(k, 1.0),
                loglik: f64::NAN,
            });
        }
        let mut ymin = f64::INFINITY;
        let mut ymax = f64::NEG_INFINITY;
        let mut n_skip = 0usize;
        for i in 0..t_len {
            let y = x.get(i, 0);
            if !y.is_finite() {
                continue;
            }
            if y < ymin {
                ymin = y;
            }
            if y > ymax {
                ymax = y;
            }
        }
        if !ymin.is_finite() {
            n_skip = t_len;
        }
        if n_skip > 0 {
            ctx.push(
                Issue::builder(IssueCode::NonPositiveSeries)
                    .severity(signlred::Severity::Warning)
                    .message("LevyHmm skipped a series with no finite observations")
                    .build(),
            );
        }
        let span = (ymax - ymin).max(0.1);
        let mut loc = Vector::from_iter((0..k).map(|j| ymin - 0.15 * span - 0.05 * j as f64));
        let mut scale = Vector::filled(k, span);
        let mut start = init_start(k);
        let mut trans = init_trans(k);
        let mut loglik = f64::NEG_INFINITY;
        let mut last_gamma: Vec<Vec<f64>> = Vec::new();
        for it in 0..self.max_iter.max(1) {
            let dummy = FittedLevyHmm {
                labels: Vector::zeros(0),
                start: start.clone(),
                trans: trans.clone(),
                loc: loc.clone(),
                scale: scale.clone(),
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
                let mut winv = 0.0_f64;
                for t in 0..t_len {
                    let y = x.get(t, 0);
                    if !y.is_finite() || y <= loc[j] {
                        continue;
                    }
                    wsum += fb.gamma[t][j];
                    winv += fb.gamma[t][j] / (y - loc[j]);
                }
                if wsum > 1e-12 && winv > 1e-12 {
                    scale[j] = (wsum / winv).max(1e-4);
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
        let dummy = FittedLevyHmm {
            labels: Vector::zeros(0),
            start: start.clone(),
            trans: trans.clone(),
            loc: loc.clone(),
            scale: scale.clone(),
            loglik,
        };
        let (labels, _) = viterbi_path(&start, &trans, &dummy.log_emit_seq(x));
        ctx.finish(FittedLevyHmm {
            labels,
            start,
            trans,
            loc,
            scale,
            loglik,
        })
    }
}

/// Cardioid HMM (cosine tilt on the circle).
///
/// State count is not identification `p`. Distinct from [`CircularHmm`]
/// (von Mises) and [`WrappedNormalHmm`] (wrapped Gaussian).
#[derive(Clone, Debug)]
pub struct CardioidHmm {
    /// Hidden states. Not identification `p`.
    pub n_states: usize,
    /// Baum–Welch cap.
    pub max_iter: usize,
}

impl Default for CardioidHmm {
    fn default() -> Self {
        Self {
            n_states: 2,
            max_iter: 40,
        }
    }
}

impl CardioidHmm {
    /// `k`-state cardioid HMM.
    pub fn new(n_states: usize) -> Self {
        Self {
            n_states,
            ..Self::default()
        }
    }

    /// Fit alias.
    pub fn fit(&mut self, x: &Matrix, session: &Session) -> Result<Qualified<FittedCardioidHmm>> {
        self.fit_unsupervised(x, session)
    }
}

/// Fitted cardioid HMM.
#[derive(Clone, Debug)]
pub struct FittedCardioidHmm {
    /// Viterbi path.
    pub labels: Vector,
    /// Start distribution.
    pub start: Vector,
    /// Transitions.
    pub trans: Matrix,
    /// Mean directions.
    pub mu: Vector,
    /// Concentrations \(\rho_j\in[0,1/2]\).
    pub rho: Vector,
    /// Training log-likelihood.
    pub loglik: f64,
}

impl FittedCardioidHmm {
    fn log_emit_seq(&self, x: &Matrix) -> Vec<Vec<f64>> {
        let t = x.nrows();
        let s = self.mu.len();
        let mut out = vec![vec![f64::NEG_INFINITY; s]; t];
        for ti in 0..t {
            let y = x.get(ti, 0);
            for j in 0..s {
                out[ti][j] = log_cardioid(y, self.mu[j], self.rho[j]);
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

impl Predict for FittedCardioidHmm {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        self.decode(x, session)
    }
}

impl FitUnsupervised for CardioidHmm {
    type Fitted = FittedCardioidHmm;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedCardioidHmm>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let t_len = x.nrows();
        let k = self.n_states.max(1);
        if t_len == 0 {
            return ctx.finish(FittedCardioidHmm {
                labels: empty_labels(0),
                start: init_start(k),
                trans: init_trans(k),
                mu: Vector::zeros(k),
                rho: Vector::filled(k, 0.2),
                loglik: f64::NAN,
            });
        }
        let mut mu = Vector::from_iter((0..k).map(|j| (j as f64 - 0.5) * 0.8));
        let mut rho = Vector::filled(k, 0.2);
        let mut start = init_start(k);
        let mut trans = init_trans(k);
        let mut loglik = f64::NEG_INFINITY;
        let mut last_gamma: Vec<Vec<f64>> = Vec::new();
        for it in 0..self.max_iter.max(1) {
            let dummy = FittedCardioidHmm {
                labels: Vector::zeros(0),
                start: start.clone(),
                trans: trans.clone(),
                mu: mu.clone(),
                rho: rho.clone(),
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
                let mut cs = 0.0_f64;
                let mut sn = 0.0_f64;
                for t in 0..t_len {
                    let y = x.get(t, 0);
                    if !y.is_finite() {
                        continue;
                    }
                    let w = fb.gamma[t][j];
                    wsum += w;
                    cs += w * y.cos();
                    sn += w * y.sin();
                }
                if wsum > 1e-12 {
                    mu[j] = sn.atan2(cs);
                    rho[j] = ((cs * cs + sn * sn).sqrt() / wsum).clamp(1e-4, 0.49);
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
        let dummy = FittedCardioidHmm {
            labels: Vector::zeros(0),
            start: start.clone(),
            trans: trans.clone(),
            mu: mu.clone(),
            rho: rho.clone(),
            loglik,
        };
        let (labels, _) = viterbi_path(&start, &trans, &dummy.log_emit_seq(x));
        ctx.finish(FittedCardioidHmm {
            labels,
            start,
            trans,
            mu,
            rho,
            loglik,
        })
    }
}

/// Zero-inflated gamma HMM.
///
/// State count is not identification `p`. Distinct from [`GammaHmm`]
/// (no extra zeros) and [`ZeroInflatedPoissonHmm`] (counts).
#[derive(Clone, Debug)]
pub struct ZeroInflatedGammaHmm {
    /// Hidden states. Not identification `p`.
    pub n_states: usize,
    /// Baum–Welch cap.
    pub max_iter: usize,
}

impl Default for ZeroInflatedGammaHmm {
    fn default() -> Self {
        Self {
            n_states: 2,
            max_iter: 40,
        }
    }
}

impl ZeroInflatedGammaHmm {
    /// `k`-state ZIG HMM.
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
    ) -> Result<Qualified<FittedZeroInflatedGammaHmm>> {
        self.fit_unsupervised(x, session)
    }
}

/// Fitted zero-inflated gamma HMM.
#[derive(Clone, Debug)]
pub struct FittedZeroInflatedGammaHmm {
    /// Viterbi path.
    pub labels: Vector,
    /// Start distribution.
    pub start: Vector,
    /// Transitions.
    pub trans: Matrix,
    /// Extra-zero mass.
    pub pi: Vector,
    /// Gamma shapes.
    pub shape: Vector,
    /// Gamma rates.
    pub rate: Vector,
    /// Training log-likelihood.
    pub loglik: f64,
}

impl FittedZeroInflatedGammaHmm {
    fn log_emit_seq(&self, x: &Matrix) -> Vec<Vec<f64>> {
        let t = x.nrows();
        let s = self.shape.len();
        let mut out = vec![vec![f64::NEG_INFINITY; s]; t];
        for ti in 0..t {
            let y = x.get(ti, 0);
            for j in 0..s {
                out[ti][j] = log_zig(y, self.pi[j], self.shape[j], self.rate[j]);
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

impl Predict for FittedZeroInflatedGammaHmm {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        self.decode(x, session)
    }
}

impl FitUnsupervised for ZeroInflatedGammaHmm {
    type Fitted = FittedZeroInflatedGammaHmm;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedZeroInflatedGammaHmm>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let t_len = x.nrows();
        let k = self.n_states.max(1);
        if t_len == 0 {
            return ctx.finish(FittedZeroInflatedGammaHmm {
                labels: empty_labels(0),
                start: init_start(k),
                trans: init_trans(k),
                pi: Vector::filled(k, 0.1),
                shape: Vector::filled(k, 2.0),
                rate: Vector::filled(k, 1.0),
                loglik: f64::NAN,
            });
        }
        let mut n_skip = 0usize;
        for i in 0..t_len {
            if x.get(i, 0).is_finite() && x.get(i, 0) < 0.0 {
                n_skip += 1;
            }
        }
        if n_skip > 0 {
            ctx.push(
                Issue::builder(IssueCode::NonPositiveSeries)
                    .severity(signlred::Severity::Warning)
                    .message(format!("ZeroInflatedGammaHmm skipped {n_skip} negative observations"))
                    .build(),
            );
        }
        let mut pi = Vector::filled(k, 0.1);
        let mut shape = Vector::from_iter((0..k).map(|j| 1.5 + j as f64));
        let mut rate = Vector::filled(k, 1.0);
        let mut start = init_start(k);
        let mut trans = init_trans(k);
        let mut loglik = f64::NEG_INFINITY;
        let mut last_gamma: Vec<Vec<f64>> = Vec::new();
        for it in 0..self.max_iter.max(1) {
            let dummy = FittedZeroInflatedGammaHmm {
                labels: Vector::zeros(0),
                start: start.clone(),
                trans: trans.clone(),
                pi: pi.clone(),
                shape: shape.clone(),
                rate: rate.clone(),
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
                let mut wzero = 0.0_f64;
                let mut wpos = 0.0_f64;
                let mut wy = 0.0_f64;
                let mut wln = 0.0_f64;
                for t in 0..t_len {
                    let y = x.get(t, 0);
                    if !y.is_finite() || y < 0.0 {
                        continue;
                    }
                    let w = fb.gamma[t][j];
                    wsum += w;
                    if y.abs() < 1e-12 {
                        wzero += w;
                    } else {
                        wpos += w;
                        wy += w * y;
                        wln += w * y.ln();
                    }
                }
                if wsum > 1e-12 {
                    pi[j] = (wzero / wsum).clamp(1e-6, 1.0 - 1e-6);
                }
                if wpos > 1e-12 {
                    let m = (wy / wpos).max(1e-4);
                    let ml = wln / wpos;
                    shape[j] = (0.5 / (m.ln() - ml).abs().max(1e-3)).clamp(0.2, 40.0);
                    rate[j] = (shape[j] / m).max(1e-4);
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
        let dummy = FittedZeroInflatedGammaHmm {
            labels: Vector::zeros(0),
            start: start.clone(),
            trans: trans.clone(),
            pi: pi.clone(),
            shape: shape.clone(),
            rate: rate.clone(),
            loglik,
        };
        let (labels, _) = viterbi_path(&start, &trans, &dummy.log_emit_seq(x));
        ctx.finish(FittedZeroInflatedGammaHmm {
            labels,
            start,
            trans,
            pi,
            shape,
            rate,
            loglik,
        })
    }
}

/// Generalized-gamma HMM (Stacy; power \(p\) is a hyperparameter).
///
/// Power is not identification `p`. Distinct from [`GammaHmm`] (\(p=1\))
/// and [`WeibullHmm`] (\(d=p\)).
#[derive(Clone, Debug)]
pub struct GenGammaHmm {
    /// Hidden states. Not identification `p`.
    pub n_states: usize,
    /// Power \(p>0\). Not identification `p`.
    pub power: f64,
    /// Baum–Welch cap.
    pub max_iter: usize,
}

impl Default for GenGammaHmm {
    fn default() -> Self {
        Self {
            n_states: 2,
            power: 2.0,
            max_iter: 40,
        }
    }
}

impl GenGammaHmm {
    /// `k`-state generalized-gamma HMM.
    pub fn new(n_states: usize) -> Self {
        Self {
            n_states,
            ..Self::default()
        }
    }

    /// Fit alias.
    pub fn fit(&mut self, x: &Matrix, session: &Session) -> Result<Qualified<FittedGenGammaHmm>> {
        self.fit_unsupervised(x, session)
    }
}

/// Fitted generalized-gamma HMM.
#[derive(Clone, Debug)]
pub struct FittedGenGammaHmm {
    /// Viterbi path.
    pub labels: Vector,
    /// Start distribution.
    pub start: Vector,
    /// Transitions.
    pub trans: Matrix,
    /// Scales \(a_j\).
    pub scale: Vector,
    /// Shape products \(d_j\).
    pub d: Vector,
    /// Shared power.
    pub power: f64,
    /// Training log-likelihood.
    pub loglik: f64,
}

impl FittedGenGammaHmm {
    fn log_emit_seq(&self, x: &Matrix) -> Vec<Vec<f64>> {
        let t = x.nrows();
        let s = self.scale.len();
        let mut out = vec![vec![f64::NEG_INFINITY; s]; t];
        for ti in 0..t {
            let y = x.get(ti, 0);
            for j in 0..s {
                out[ti][j] = log_gengamma(y, self.scale[j], self.d[j], self.power);
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

impl Predict for FittedGenGammaHmm {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        self.decode(x, session)
    }
}

impl FitUnsupervised for GenGammaHmm {
    type Fitted = FittedGenGammaHmm;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedGenGammaHmm>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let t_len = x.nrows();
        let k = self.n_states.max(1);
        let power = if self.power.is_finite() && self.power > 0.0 {
            self.power
        } else {
            2.0
        };
        if t_len == 0 {
            return ctx.finish(FittedGenGammaHmm {
                labels: empty_labels(0),
                start: init_start(k),
                trans: init_trans(k),
                scale: Vector::filled(k, 1.0),
                d: Vector::filled(k, 2.0),
                power,
                loglik: f64::NAN,
            });
        }
        let mut n_skip = 0usize;
        for i in 0..t_len {
            if x.get(i, 0).is_finite() && x.get(i, 0) <= 0.0 {
                n_skip += 1;
            }
        }
        if n_skip > 0 {
            ctx.push(
                Issue::builder(IssueCode::NonPositiveSeries)
                    .severity(signlred::Severity::Warning)
                    .message(format!("GenGammaHmm skipped {n_skip} non-positive observations"))
                    .build(),
            );
        }
        let mut scale = Vector::from_iter((0..k).map(|j| 1.0 + 0.5 * j as f64));
        let mut dpar = Vector::from_iter((0..k).map(|j| 1.5 + 0.5 * j as f64));
        let mut start = init_start(k);
        let mut trans = init_trans(k);
        let mut loglik = f64::NEG_INFINITY;
        let mut last_gamma: Vec<Vec<f64>> = Vec::new();
        for it in 0..self.max_iter.max(1) {
            let dummy = FittedGenGammaHmm {
                labels: Vector::zeros(0),
                start: start.clone(),
                trans: trans.clone(),
                scale: scale.clone(),
                d: dpar.clone(),
                power,
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
                let mut wyp = 0.0_f64;
                for t in 0..t_len {
                    let y = x.get(t, 0);
                    if !y.is_finite() || y <= 0.0 {
                        continue;
                    }
                    let w = fb.gamma[t][j];
                    wsum += w;
                    wy += w * y;
                    wyp += w * y.powf(power);
                }
                if wsum > 1e-12 {
                    let m = (wy / wsum).max(1e-4);
                    scale[j] = (wyp / wsum).max(1e-8).powf(1.0 / power).max(1e-4);
                    dpar[j] = (m / scale[j] * power).clamp(0.2, 40.0);
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
        let dummy = FittedGenGammaHmm {
            labels: Vector::zeros(0),
            start: start.clone(),
            trans: trans.clone(),
            scale: scale.clone(),
            d: dpar.clone(),
            power,
            loglik,
        };
        let (labels, _) = viterbi_path(&start, &trans, &dummy.log_emit_seq(x));
        ctx.finish(FittedGenGammaHmm {
            labels,
            start,
            trans,
            scale,
            d: dpar,
            power,
            loglik,
        })
    }
}

/// Consul–Jain generalized Poisson HMM.
///
/// Dispersion \(\xi\) is not identification `p`. Distinct from [`PoissonHmm`]
/// (\(\xi=0\)) and [`ComPoissonHmm`] (factorial tilt).
#[derive(Clone, Debug)]
pub struct GenPoissonHmm {
    /// Hidden states. Not identification `p`.
    pub n_states: usize,
    /// Baum–Welch cap.
    pub max_iter: usize,
}

impl Default for GenPoissonHmm {
    fn default() -> Self {
        Self {
            n_states: 2,
            max_iter: 40,
        }
    }
}

impl GenPoissonHmm {
    /// `k`-state generalized-Poisson HMM.
    pub fn new(n_states: usize) -> Self {
        Self {
            n_states,
            ..Self::default()
        }
    }

    /// Fit alias.
    pub fn fit(&mut self, x: &Matrix, session: &Session) -> Result<Qualified<FittedGenPoissonHmm>> {
        self.fit_unsupervised(x, session)
    }
}

/// Fitted generalized-Poisson HMM.
#[derive(Clone, Debug)]
pub struct FittedGenPoissonHmm {
    /// Viterbi path.
    pub labels: Vector,
    /// Start distribution.
    pub start: Vector,
    /// Transitions.
    pub trans: Matrix,
    /// Intensities \(\lambda_j\).
    pub lam: Vector,
    /// Dispersions \(\xi_j\).
    pub xi: Vector,
    /// Training log-likelihood.
    pub loglik: f64,
}

impl FittedGenPoissonHmm {
    fn log_emit_seq(&self, x: &Matrix) -> Vec<Vec<f64>> {
        let t = x.nrows();
        let s = self.lam.len();
        let mut out = vec![vec![f64::NEG_INFINITY; s]; t];
        for ti in 0..t {
            let y = x.get(ti, 0);
            for j in 0..s {
                out[ti][j] = log_gen_poisson(y, self.lam[j], self.xi[j]);
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

impl Predict for FittedGenPoissonHmm {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        self.decode(x, session)
    }
}

impl FitUnsupervised for GenPoissonHmm {
    type Fitted = FittedGenPoissonHmm;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedGenPoissonHmm>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let t_len = x.nrows();
        let k = self.n_states.max(1);
        if t_len == 0 {
            return ctx.finish(FittedGenPoissonHmm {
                labels: empty_labels(0),
                start: init_start(k),
                trans: init_trans(k),
                lam: Vector::filled(k, 1.0),
                xi: Vector::zeros(k),
                loglik: f64::NAN,
            });
        }
        let mut lam = Vector::from_iter((0..k).map(|j| 1.0 + 2.0 * j as f64));
        let mut xi = Vector::zeros(k);
        let mut start = init_start(k);
        let mut trans = init_trans(k);
        let mut loglik = f64::NEG_INFINITY;
        let mut last_gamma: Vec<Vec<f64>> = Vec::new();
        for it in 0..self.max_iter.max(1) {
            let dummy = FittedGenPoissonHmm {
                labels: Vector::zeros(0),
                start: start.clone(),
                trans: trans.clone(),
                lam: lam.clone(),
                xi: xi.clone(),
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
                    if !y.is_finite() || y < 0.0 {
                        continue;
                    }
                    let w = fb.gamma[t][j];
                    wsum += w;
                    wy += w * y;
                    wy2 += w * y * y;
                }
                if wsum > 1e-12 {
                    let m = (wy / wsum).max(1e-4);
                    let v = (wy2 / wsum - m * m).max(1e-4);
                    let shrink = (m / v).sqrt().clamp(0.1, 2.0);
                    let xij = (1.0 - shrink).clamp(-0.8, 0.8);
                    xi[j] = xij;
                    lam[j] = (m * (1.0 - xij)).max(1e-4).min(40.0);
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
        let dummy = FittedGenPoissonHmm {
            labels: Vector::zeros(0),
            start: start.clone(),
            trans: trans.clone(),
            lam: lam.clone(),
            xi: xi.clone(),
            loglik,
        };
        let (labels, _) = viterbi_path(&start, &trans, &dummy.log_emit_seq(x));
        ctx.finish(FittedGenPoissonHmm {
            labels,
            start,
            trans,
            lam,
            xi,
            loglik,
        })
    }
}

fn log_dagum(y: f64, scale: f64, a: f64, p: f64) -> f64 {
    if !y.is_finite() || y <= 0.0 || scale <= 0.0 || a <= 0.0 || p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    let z = (y / scale).max(1e-12);
    a.ln() + p.ln() - y.ln() - a * z.ln() - (p + 1.0) * (1.0 + z.powf(-a)).ln()
}

fn log_half_cauchy(y: f64, scale: f64) -> f64 {
    if !y.is_finite() || y < 0.0 || scale <= 0.0 {
        return f64::NEG_INFINITY;
    }
    std::f64::consts::LN_2 - std::f64::consts::PI.ln() - scale.ln() - (1.0 + (y / scale) * (y / scale)).ln()
}

fn log_logarithmic(y: f64, p: f64) -> f64 {
    if !y.is_finite() || y < 1.0 - 1e-9 || p <= 0.0 || p >= 1.0 {
        return f64::NEG_INFINITY;
    }
    let y = y.round().max(1.0);
    y * p.ln() - y.ln() - (-(1.0 - p).ln()).ln()
}

fn log_yule_simon(y: f64, rho: f64) -> f64 {
    if !y.is_finite() || y < 1.0 - 1e-9 || rho <= 0.0 {
        return f64::NEG_INFINITY;
    }
    let y = y.round().max(1.0);
    rho.ln() + crate::special::ln_gamma(y) + crate::special::ln_gamma(rho + 1.0)
        - crate::special::ln_gamma(y + rho + 1.0)
}

/// Dagum HMM (inverse Burr on the positive line).
///
/// Shapes are not identification `p`. Distinct from [`BurrHmm`] (cdf
/// complement) and [`ParetoHmm`] (one shape).
#[derive(Clone, Debug)]
pub struct DagumHmm {
    /// Hidden states. Not identification `p`.
    pub n_states: usize,
    /// Baum–Welch cap.
    pub max_iter: usize,
}

impl Default for DagumHmm {
    fn default() -> Self {
        Self {
            n_states: 2,
            max_iter: 40,
        }
    }
}

impl DagumHmm {
    /// `k`-state Dagum HMM.
    pub fn new(n_states: usize) -> Self {
        Self {
            n_states,
            ..Self::default()
        }
    }

    /// Fit alias.
    pub fn fit(&mut self, x: &Matrix, session: &Session) -> Result<Qualified<FittedDagumHmm>> {
        self.fit_unsupervised(x, session)
    }
}

/// Fitted Dagum HMM.
#[derive(Clone, Debug)]
pub struct FittedDagumHmm {
    /// Viterbi path.
    pub labels: Vector,
    /// Start distribution.
    pub start: Vector,
    /// Transitions.
    pub trans: Matrix,
    /// Scales.
    pub scale: Vector,
    /// First shapes \(a_j\).
    pub a: Vector,
    /// Second shapes \(p_j\).
    pub p: Vector,
    /// Training log-likelihood.
    pub loglik: f64,
}

impl FittedDagumHmm {
    fn log_emit_seq(&self, x: &Matrix) -> Vec<Vec<f64>> {
        let t = x.nrows();
        let s = self.scale.len();
        let mut out = vec![vec![f64::NEG_INFINITY; s]; t];
        for ti in 0..t {
            let y = x.get(ti, 0);
            for j in 0..s {
                out[ti][j] = log_dagum(y, self.scale[j], self.a[j], self.p[j]);
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

impl Predict for FittedDagumHmm {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        self.decode(x, session)
    }
}

impl FitUnsupervised for DagumHmm {
    type Fitted = FittedDagumHmm;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedDagumHmm>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let t_len = x.nrows();
        let k = self.n_states.max(1);
        if t_len == 0 {
            return ctx.finish(FittedDagumHmm {
                labels: empty_labels(0),
                start: init_start(k),
                trans: init_trans(k),
                scale: Vector::filled(k, 1.0),
                a: Vector::filled(k, 2.0),
                p: Vector::filled(k, 1.0),
                loglik: f64::NAN,
            });
        }
        let mut n_skip = 0usize;
        for i in 0..t_len {
            if x.get(i, 0).is_finite() && x.get(i, 0) <= 0.0 {
                n_skip += 1;
            }
        }
        if n_skip > 0 {
            ctx.push(
                Issue::builder(IssueCode::NonPositiveSeries)
                    .severity(signlred::Severity::Warning)
                    .message(format!("DagumHmm skipped {n_skip} non-positive observations"))
                    .build(),
            );
        }
        let mut scale = Vector::from_iter((0..k).map(|j| 1.0 + 0.5 * j as f64));
        let mut ashape = Vector::from_iter((0..k).map(|j| 1.5 + 0.5 * j as f64));
        let mut pshape = Vector::filled(k, 1.0);
        let mut start = init_start(k);
        let mut trans = init_trans(k);
        let mut loglik = f64::NEG_INFINITY;
        let mut last_gamma: Vec<Vec<f64>> = Vec::new();
        for it in 0..self.max_iter.max(1) {
            let dummy = FittedDagumHmm {
                labels: Vector::zeros(0),
                start: start.clone(),
                trans: trans.clone(),
                scale: scale.clone(),
                a: ashape.clone(),
                p: pshape.clone(),
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
                let mut wln = 0.0_f64;
                for t in 0..t_len {
                    let y = x.get(t, 0);
                    if !y.is_finite() || y <= 0.0 {
                        continue;
                    }
                    wsum += fb.gamma[t][j];
                    wln += fb.gamma[t][j] * y.ln();
                }
                if wsum > 1e-12 {
                    let m = wln / wsum;
                    scale[j] = m.exp().max(1e-4);
                    let mut wad = 0.0_f64;
                    for t in 0..t_len {
                        let y = x.get(t, 0);
                        if y.is_finite() && y > 0.0 {
                            wad += fb.gamma[t][j] * (y.ln() - m).abs();
                        }
                    }
                    ashape[j] = (std::f64::consts::PI / ((wad / wsum).max(1e-4) * 3.0_f64.sqrt()))
                        .clamp(0.2, 40.0);
                    pshape[j] = (wsum / (1.0 + wad).max(1e-4)).clamp(0.2, 40.0);
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
        let dummy = FittedDagumHmm {
            labels: Vector::zeros(0),
            start: start.clone(),
            trans: trans.clone(),
            scale: scale.clone(),
            a: ashape.clone(),
            p: pshape.clone(),
            loglik,
        };
        let (labels, _) = viterbi_path(&start, &trans, &dummy.log_emit_seq(x));
        ctx.finish(FittedDagumHmm {
            labels,
            start,
            trans,
            scale,
            a: ashape,
            p: pshape,
            loglik,
        })
    }
}

/// Half-Cauchy HMM on the non-negative line.
///
/// State count is not identification `p`. Distinct from [`CauchyHmm`]
/// (two-sided) and half-normal sketches.
#[derive(Clone, Debug)]
pub struct HalfCauchyHmm {
    /// Hidden states. Not identification `p`.
    pub n_states: usize,
    /// Baum–Welch cap.
    pub max_iter: usize,
}

impl Default for HalfCauchyHmm {
    fn default() -> Self {
        Self {
            n_states: 2,
            max_iter: 40,
        }
    }
}

impl HalfCauchyHmm {
    /// `k`-state half-Cauchy HMM.
    pub fn new(n_states: usize) -> Self {
        Self {
            n_states,
            ..Self::default()
        }
    }

    /// Fit alias.
    pub fn fit(&mut self, x: &Matrix, session: &Session) -> Result<Qualified<FittedHalfCauchyHmm>> {
        self.fit_unsupervised(x, session)
    }
}

/// Fitted half-Cauchy HMM.
#[derive(Clone, Debug)]
pub struct FittedHalfCauchyHmm {
    /// Viterbi path.
    pub labels: Vector,
    /// Start distribution.
    pub start: Vector,
    /// Transitions.
    pub trans: Matrix,
    /// Scales.
    pub scale: Vector,
    /// Training log-likelihood.
    pub loglik: f64,
}

impl FittedHalfCauchyHmm {
    fn log_emit_seq(&self, x: &Matrix) -> Vec<Vec<f64>> {
        let t = x.nrows();
        let s = self.scale.len();
        let mut out = vec![vec![f64::NEG_INFINITY; s]; t];
        for ti in 0..t {
            let y = x.get(ti, 0);
            for j in 0..s {
                out[ti][j] = log_half_cauchy(y, self.scale[j]);
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

impl Predict for FittedHalfCauchyHmm {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        self.decode(x, session)
    }
}

impl FitUnsupervised for HalfCauchyHmm {
    type Fitted = FittedHalfCauchyHmm;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedHalfCauchyHmm>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let t_len = x.nrows();
        let k = self.n_states.max(1);
        if t_len == 0 {
            return ctx.finish(FittedHalfCauchyHmm {
                labels: empty_labels(0),
                start: init_start(k),
                trans: init_trans(k),
                scale: Vector::filled(k, 1.0),
                loglik: f64::NAN,
            });
        }
        let mut n_skip = 0usize;
        for i in 0..t_len {
            if x.get(i, 0).is_finite() && x.get(i, 0) < 0.0 {
                n_skip += 1;
            }
        }
        if n_skip > 0 {
            ctx.push(
                Issue::builder(IssueCode::NonPositiveSeries)
                    .severity(signlred::Severity::Warning)
                    .message(format!("HalfCauchyHmm skipped {n_skip} negative observations"))
                    .build(),
            );
        }
        let mut scale = Vector::from_iter((0..k).map(|j| 1.0 + j as f64));
        let mut start = init_start(k);
        let mut trans = init_trans(k);
        let mut loglik = f64::NEG_INFINITY;
        let mut last_gamma: Vec<Vec<f64>> = Vec::new();
        for it in 0..self.max_iter.max(1) {
            let dummy = FittedHalfCauchyHmm {
                labels: Vector::zeros(0),
                start: start.clone(),
                trans: trans.clone(),
                scale: scale.clone(),
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
                    scale[j] = (wy / wsum).max(COV_FLOOR.sqrt());
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
        let dummy = FittedHalfCauchyHmm {
            labels: Vector::zeros(0),
            start: start.clone(),
            trans: trans.clone(),
            scale: scale.clone(),
            loglik,
        };
        let (labels, _) = viterbi_path(&start, &trans, &dummy.log_emit_seq(x));
        ctx.finish(FittedHalfCauchyHmm {
            labels,
            start,
            trans,
            scale,
            loglik,
        })
    }
}

/// Logarithmic-series HMM on \(\{1,2,\ldots\}\).
///
/// State count is not identification `p`. Distinct from [`GeometricHmm`]
/// (includes zero) and [`PoissonHmm`] (unbounded factorial).
#[derive(Clone, Debug)]
pub struct LogarithmicHmm {
    /// Hidden states. Not identification `p`.
    pub n_states: usize,
    /// Baum–Welch cap.
    pub max_iter: usize,
}

impl Default for LogarithmicHmm {
    fn default() -> Self {
        Self {
            n_states: 2,
            max_iter: 40,
        }
    }
}

impl LogarithmicHmm {
    /// `k`-state log-series HMM.
    pub fn new(n_states: usize) -> Self {
        Self {
            n_states,
            ..Self::default()
        }
    }

    /// Fit alias.
    pub fn fit(&mut self, x: &Matrix, session: &Session) -> Result<Qualified<FittedLogarithmicHmm>> {
        self.fit_unsupervised(x, session)
    }
}

/// Fitted logarithmic-series HMM.
#[derive(Clone, Debug)]
pub struct FittedLogarithmicHmm {
    /// Viterbi path.
    pub labels: Vector,
    /// Start distribution.
    pub start: Vector,
    /// Transitions.
    pub trans: Matrix,
    /// Success parameters \(p_j\in(0,1)\).
    pub p: Vector,
    /// Training log-likelihood.
    pub loglik: f64,
}

impl FittedLogarithmicHmm {
    fn log_emit_seq(&self, x: &Matrix) -> Vec<Vec<f64>> {
        let t = x.nrows();
        let s = self.p.len();
        let mut out = vec![vec![f64::NEG_INFINITY; s]; t];
        for ti in 0..t {
            let y = x.get(ti, 0);
            for j in 0..s {
                out[ti][j] = log_logarithmic(y, self.p[j]);
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

impl Predict for FittedLogarithmicHmm {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        self.decode(x, session)
    }
}

impl FitUnsupervised for LogarithmicHmm {
    type Fitted = FittedLogarithmicHmm;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedLogarithmicHmm>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let t_len = x.nrows();
        let k = self.n_states.max(1);
        if t_len == 0 {
            return ctx.finish(FittedLogarithmicHmm {
                labels: empty_labels(0),
                start: init_start(k),
                trans: init_trans(k),
                p: Vector::filled(k, 0.5),
                loglik: f64::NAN,
            });
        }
        let mut p = Vector::from_iter((0..k).map(|j| 0.3 + 0.2 * j as f64));
        let mut start = init_start(k);
        let mut trans = init_trans(k);
        let mut loglik = f64::NEG_INFINITY;
        let mut last_gamma: Vec<Vec<f64>> = Vec::new();
        for it in 0..self.max_iter.max(1) {
            let dummy = FittedLogarithmicHmm {
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
                    if !y.is_finite() || y < 1.0 - 1e-9 {
                        continue;
                    }
                    wsum += fb.gamma[t][j];
                    wy += fb.gamma[t][j] * y;
                }
                if wsum > 1e-12 {
                    let m = (wy / wsum).max(1.01);
                    p[j] = (1.0 - 1.0 / m).clamp(1e-3, 1.0 - 1e-3);
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
        let dummy = FittedLogarithmicHmm {
            labels: Vector::zeros(0),
            start: start.clone(),
            trans: trans.clone(),
            p: p.clone(),
            loglik,
        };
        let (labels, _) = viterbi_path(&start, &trans, &dummy.log_emit_seq(x));
        ctx.finish(FittedLogarithmicHmm {
            labels,
            start,
            trans,
            p,
            loglik,
        })
    }
}

/// Yule–Simon HMM on \(\{1,2,\ldots\}\).
///
/// Shape \(\rho\) is not identification `p`. Distinct from [`LogarithmicHmm`]
/// (log-series) and zeta/Zipf sketches.
#[derive(Clone, Debug)]
pub struct YuleSimonHmm {
    /// Hidden states. Not identification `p`.
    pub n_states: usize,
    /// Baum–Welch cap.
    pub max_iter: usize,
}

impl Default for YuleSimonHmm {
    fn default() -> Self {
        Self {
            n_states: 2,
            max_iter: 40,
        }
    }
}

impl YuleSimonHmm {
    /// `k`-state Yule–Simon HMM.
    pub fn new(n_states: usize) -> Self {
        Self {
            n_states,
            ..Self::default()
        }
    }

    /// Fit alias.
    pub fn fit(&mut self, x: &Matrix, session: &Session) -> Result<Qualified<FittedYuleSimonHmm>> {
        self.fit_unsupervised(x, session)
    }
}

/// Fitted Yule–Simon HMM.
#[derive(Clone, Debug)]
pub struct FittedYuleSimonHmm {
    /// Viterbi path.
    pub labels: Vector,
    /// Start distribution.
    pub start: Vector,
    /// Transitions.
    pub trans: Matrix,
    /// Shapes \(\rho_j\).
    pub rho: Vector,
    /// Training log-likelihood.
    pub loglik: f64,
}

impl FittedYuleSimonHmm {
    fn log_emit_seq(&self, x: &Matrix) -> Vec<Vec<f64>> {
        let t = x.nrows();
        let s = self.rho.len();
        let mut out = vec![vec![f64::NEG_INFINITY; s]; t];
        for ti in 0..t {
            let y = x.get(ti, 0);
            for j in 0..s {
                out[ti][j] = log_yule_simon(y, self.rho[j]);
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

impl Predict for FittedYuleSimonHmm {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        self.decode(x, session)
    }
}

impl FitUnsupervised for YuleSimonHmm {
    type Fitted = FittedYuleSimonHmm;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedYuleSimonHmm>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let t_len = x.nrows();
        let k = self.n_states.max(1);
        if t_len == 0 {
            return ctx.finish(FittedYuleSimonHmm {
                labels: empty_labels(0),
                start: init_start(k),
                trans: init_trans(k),
                rho: Vector::filled(k, 2.0),
                loglik: f64::NAN,
            });
        }
        let mut rho = Vector::from_iter((0..k).map(|j| 1.5 + j as f64));
        let mut start = init_start(k);
        let mut trans = init_trans(k);
        let mut loglik = f64::NEG_INFINITY;
        let mut last_gamma: Vec<Vec<f64>> = Vec::new();
        for it in 0..self.max_iter.max(1) {
            let dummy = FittedYuleSimonHmm {
                labels: Vector::zeros(0),
                start: start.clone(),
                trans: trans.clone(),
                rho: rho.clone(),
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
                    if !y.is_finite() || y < 1.0 - 1e-9 {
                        continue;
                    }
                    wsum += fb.gamma[t][j];
                    wy += fb.gamma[t][j] * y;
                }
                if wsum > 1e-12 {
                    let m = (wy / wsum).max(1.05);
                    rho[j] = (1.0 / (m - 1.0)).clamp(0.2, 40.0);
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
        let dummy = FittedYuleSimonHmm {
            labels: Vector::zeros(0),
            start: start.clone(),
            trans: trans.clone(),
            rho: rho.clone(),
            loglik,
        };
        let (labels, _) = viterbi_path(&start, &trans, &dummy.log_emit_seq(x));
        ctx.finish(FittedYuleSimonHmm {
            labels,
            start,
            trans,
            rho,
            loglik,
        })
    }
}

fn riemann_zeta(s: f64) -> f64 {
    let s = s.max(1.05);
    let mut acc = 0.0_f64;
    for k in 1..=192 {
        acc += (k as f64).powf(-s);
    }
    acc.max(1e-300)
}

fn log_zeta_emit(y: f64, s: f64, ln_zeta: f64) -> f64 {
    if !y.is_finite() || y < 1.0 - 1e-9 || s <= 1.0 || !ln_zeta.is_finite() {
        return f64::NEG_INFINITY;
    }
    let y = y.round().max(1.0);
    -s * y.ln() - ln_zeta
}

fn log_lomax(y: f64, scale: f64, shape: f64) -> f64 {
    if !y.is_finite() || y < 0.0 || scale <= 0.0 || shape <= 0.0 {
        return f64::NEG_INFINITY;
    }
    shape.ln() - scale.ln() - (shape + 1.0) * (1.0 + y / scale).ln()
}

fn log_half_normal(y: f64, sigma: f64) -> f64 {
    if !y.is_finite() || y < 0.0 || sigma <= 0.0 {
        return f64::NEG_INFINITY;
    }
    0.5 * (2.0 / std::f64::consts::PI).ln() - sigma.ln() - (y * y) / (2.0 * sigma * sigma)
}

fn log_maxwell(y: f64, a: f64) -> f64 {
    if !y.is_finite() || y <= 0.0 || a <= 0.0 {
        return f64::NEG_INFINITY;
    }
    0.5 * (2.0 / std::f64::consts::PI).ln() + 2.0 * y.ln() - 3.0 * a.ln() - (y * y) / (2.0 * a * a)
}

fn log_beta_prime(y: f64, alpha: f64, beta: f64) -> f64 {
    if !y.is_finite() || y <= 0.0 || alpha <= 0.0 || beta <= 0.0 {
        return f64::NEG_INFINITY;
    }
    (alpha - 1.0) * y.ln() - (alpha + beta) * (1.0 + y).ln()
        - crate::special::ln_gamma(alpha)
        - crate::special::ln_gamma(beta)
        + crate::special::ln_gamma(alpha + beta)
}

fn log_gompertz(y: f64, eta: f64, c: f64) -> f64 {
    if !y.is_finite() || y <= 0.0 || eta <= 0.0 || c <= 0.0 {
        return f64::NEG_INFINITY;
    }
    let ey = (eta * y).exp();
    if !ey.is_finite() {
        return f64::NEG_INFINITY;
    }
    c.ln() + eta.ln() + eta * y - c * (ey - 1.0)
}

/// Zipf / zeta HMM on \(\{1,2,\ldots\}\).
///
/// Shape \(s>1\) is not identification `p`. Distinct from [`YuleSimonHmm`]
/// (Beta tail) and [`LogarithmicHmm`] (log-series).
#[derive(Clone, Debug)]
pub struct ZetaHmm {
    /// Hidden states. Not identification `p`.
    pub n_states: usize,
    /// Baum–Welch cap.
    pub max_iter: usize,
}

impl Default for ZetaHmm {
    fn default() -> Self {
        Self {
            n_states: 2,
            max_iter: 40,
        }
    }
}

impl ZetaHmm {
    /// `k`-state zeta HMM.
    pub fn new(n_states: usize) -> Self {
        Self {
            n_states,
            ..Self::default()
        }
    }

    /// Fit alias.
    pub fn fit(&mut self, x: &Matrix, session: &Session) -> Result<Qualified<FittedZetaHmm>> {
        self.fit_unsupervised(x, session)
    }
}

/// Fitted zeta / Zipf HMM.
#[derive(Clone, Debug)]
pub struct FittedZetaHmm {
    /// Viterbi path.
    pub labels: Vector,
    /// Start distribution.
    pub start: Vector,
    /// Transitions.
    pub trans: Matrix,
    /// Exponents \(s_j>1\).
    pub s: Vector,
    /// Training log-likelihood.
    pub loglik: f64,
}

impl FittedZetaHmm {
    fn log_emit_seq(&self, x: &Matrix) -> Vec<Vec<f64>> {
        let t = x.nrows();
        let ns = self.s.len();
        let ln_z: Vec<f64> = (0..ns).map(|j| riemann_zeta(self.s[j]).ln()).collect();
        let mut out = vec![vec![f64::NEG_INFINITY; ns]; t];
        for ti in 0..t {
            let y = x.get(ti, 0);
            for j in 0..ns {
                out[ti][j] = log_zeta_emit(y, self.s[j], ln_z[j]);
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

impl Predict for FittedZetaHmm {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        self.decode(x, session)
    }
}

impl FitUnsupervised for ZetaHmm {
    type Fitted = FittedZetaHmm;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedZetaHmm>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let t_len = x.nrows();
        let k = self.n_states.max(1);
        if t_len == 0 {
            return ctx.finish(FittedZetaHmm {
                labels: empty_labels(0),
                start: init_start(k),
                trans: init_trans(k),
                s: Vector::filled(k, 2.0),
                loglik: f64::NAN,
            });
        }
        let mut s = Vector::from_iter((0..k).map(|j| 1.6 + 0.4 * j as f64));
        let mut start = init_start(k);
        let mut trans = init_trans(k);
        let mut loglik = f64::NEG_INFINITY;
        let mut last_gamma: Vec<Vec<f64>> = Vec::new();
        for it in 0..self.max_iter.max(1) {
            let dummy = FittedZetaHmm {
                labels: Vector::zeros(0),
                start: start.clone(),
                trans: trans.clone(),
                s: s.clone(),
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
                    if !y.is_finite() || y < 1.0 - 1e-9 {
                        continue;
                    }
                    wsum += fb.gamma[t][j];
                    wy += fb.gamma[t][j] * y;
                }
                if wsum > 1e-12 {
                    let m = (wy / wsum).max(1.02);
                    s[j] = (1.0 + 1.0 / (m - 1.0)).clamp(1.15, 12.0);
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
        let dummy = FittedZetaHmm {
            labels: Vector::zeros(0),
            start: start.clone(),
            trans: trans.clone(),
            s: s.clone(),
            loglik,
        };
        let (labels, _) = viterbi_path(&start, &trans, &dummy.log_emit_seq(x));
        ctx.finish(FittedZetaHmm {
            labels,
            start,
            trans,
            s,
            loglik,
        })
    }
}

/// Lomax HMM (Pareto Type II on the non-negative line).
///
/// Shapes are not identification `p`. Distinct from [`ParetoHmm`] (Type I,
/// hard \(x_m\)) and [`DagumHmm`] (two-shape inverse Burr).
#[derive(Clone, Debug)]
pub struct LomaxHmm {
    /// Hidden states. Not identification `p`.
    pub n_states: usize,
    /// Baum–Welch cap.
    pub max_iter: usize,
}

impl Default for LomaxHmm {
    fn default() -> Self {
        Self {
            n_states: 2,
            max_iter: 40,
        }
    }
}

impl LomaxHmm {
    /// `k`-state Lomax HMM.
    pub fn new(n_states: usize) -> Self {
        Self {
            n_states,
            ..Self::default()
        }
    }

    /// Fit alias.
    pub fn fit(&mut self, x: &Matrix, session: &Session) -> Result<Qualified<FittedLomaxHmm>> {
        self.fit_unsupervised(x, session)
    }
}

/// Fitted Lomax HMM.
#[derive(Clone, Debug)]
pub struct FittedLomaxHmm {
    /// Viterbi path.
    pub labels: Vector,
    /// Start distribution.
    pub start: Vector,
    /// Transitions.
    pub trans: Matrix,
    /// Scales \(\theta_j\).
    pub scale: Vector,
    /// Shapes \(\alpha_j\).
    pub shape: Vector,
    /// Training log-likelihood.
    pub loglik: f64,
}

impl FittedLomaxHmm {
    fn log_emit_seq(&self, x: &Matrix) -> Vec<Vec<f64>> {
        let t = x.nrows();
        let ns = self.scale.len();
        let mut out = vec![vec![f64::NEG_INFINITY; ns]; t];
        for ti in 0..t {
            let y = x.get(ti, 0);
            for j in 0..ns {
                out[ti][j] = log_lomax(y, self.scale[j], self.shape[j]);
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

impl Predict for FittedLomaxHmm {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        self.decode(x, session)
    }
}

impl FitUnsupervised for LomaxHmm {
    type Fitted = FittedLomaxHmm;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedLomaxHmm>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let t_len = x.nrows();
        let k = self.n_states.max(1);
        if t_len == 0 {
            return ctx.finish(FittedLomaxHmm {
                labels: empty_labels(0),
                start: init_start(k),
                trans: init_trans(k),
                scale: Vector::filled(k, 1.0),
                shape: Vector::filled(k, 2.0),
                loglik: f64::NAN,
            });
        }
        let mut n_skip = 0usize;
        for i in 0..t_len {
            if x.get(i, 0).is_finite() && x.get(i, 0) < 0.0 {
                n_skip += 1;
            }
        }
        if n_skip > 0 {
            ctx.push(
                Issue::builder(IssueCode::NonPositiveSeries)
                    .severity(signlred::Severity::Warning)
                    .message(format!("LomaxHmm skipped {n_skip} negative observations"))
                    .build(),
            );
        }
        let mut scale = Vector::from_iter((0..k).map(|j| 1.0 + 0.5 * j as f64));
        let mut shape = Vector::from_iter((0..k).map(|j| 2.0 + 0.5 * j as f64));
        let mut start = init_start(k);
        let mut trans = init_trans(k);
        let mut loglik = f64::NEG_INFINITY;
        let mut last_gamma: Vec<Vec<f64>> = Vec::new();
        for it in 0..self.max_iter.max(1) {
            let dummy = FittedLomaxHmm {
                labels: Vector::zeros(0),
                start: start.clone(),
                trans: trans.clone(),
                scale: scale.clone(),
                shape: shape.clone(),
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
                    if !y.is_finite() || y < 0.0 {
                        continue;
                    }
                    wsum += fb.gamma[t][j];
                    wy += fb.gamma[t][j] * y;
                    wy2 += fb.gamma[t][j] * y * y;
                }
                if wsum > 1e-12 {
                    let m = (wy / wsum).max(1e-4);
                    let m2 = (wy2 / wsum).max(m * m);
                    let var = (m2 - m * m).max(1e-8);
                    let cv2 = var / (m * m).max(1e-8);
                    let alph = if cv2 > 1.05 {
                        (2.0 * cv2 / (cv2 - 1.0)).clamp(1.2, 40.0)
                    } else {
                        3.0
                    };
                    shape[j] = alph;
                    scale[j] = (m * (alph - 1.0)).max(1e-4);
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
        let dummy = FittedLomaxHmm {
            labels: Vector::zeros(0),
            start: start.clone(),
            trans: trans.clone(),
            scale: scale.clone(),
            shape: shape.clone(),
            loglik,
        };
        let (labels, _) = viterbi_path(&start, &trans, &dummy.log_emit_seq(x));
        ctx.finish(FittedLomaxHmm {
            labels,
            start,
            trans,
            scale,
            shape,
            loglik,
        })
    }
}

/// Half-normal HMM on the non-negative line.
///
/// State count is not identification `p`. Distinct from [`HalfCauchyHmm`]
/// (Cauchy tails) and [`RayleighHmm`] (Weibull / \(y\)-weighted Gaussian).
#[derive(Clone, Debug)]
pub struct HalfNormalHmm {
    /// Hidden states. Not identification `p`.
    pub n_states: usize,
    /// Baum–Welch cap.
    pub max_iter: usize,
}

impl Default for HalfNormalHmm {
    fn default() -> Self {
        Self {
            n_states: 2,
            max_iter: 40,
        }
    }
}

impl HalfNormalHmm {
    /// `k`-state half-normal HMM.
    pub fn new(n_states: usize) -> Self {
        Self {
            n_states,
            ..Self::default()
        }
    }

    /// Fit alias.
    pub fn fit(&mut self, x: &Matrix, session: &Session) -> Result<Qualified<FittedHalfNormalHmm>> {
        self.fit_unsupervised(x, session)
    }
}

/// Fitted half-normal HMM.
#[derive(Clone, Debug)]
pub struct FittedHalfNormalHmm {
    /// Viterbi path.
    pub labels: Vector,
    /// Start distribution.
    pub start: Vector,
    /// Transitions.
    pub trans: Matrix,
    /// Scales \(\sigma_j\).
    pub scale: Vector,
    /// Training log-likelihood.
    pub loglik: f64,
}

impl FittedHalfNormalHmm {
    fn log_emit_seq(&self, x: &Matrix) -> Vec<Vec<f64>> {
        let t = x.nrows();
        let ns = self.scale.len();
        let mut out = vec![vec![f64::NEG_INFINITY; ns]; t];
        for ti in 0..t {
            let y = x.get(ti, 0);
            for j in 0..ns {
                out[ti][j] = log_half_normal(y, self.scale[j]);
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

impl Predict for FittedHalfNormalHmm {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        self.decode(x, session)
    }
}

impl FitUnsupervised for HalfNormalHmm {
    type Fitted = FittedHalfNormalHmm;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedHalfNormalHmm>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let t_len = x.nrows();
        let k = self.n_states.max(1);
        if t_len == 0 {
            return ctx.finish(FittedHalfNormalHmm {
                labels: empty_labels(0),
                start: init_start(k),
                trans: init_trans(k),
                scale: Vector::filled(k, 1.0),
                loglik: f64::NAN,
            });
        }
        let mut n_skip = 0usize;
        for i in 0..t_len {
            if x.get(i, 0).is_finite() && x.get(i, 0) < 0.0 {
                n_skip += 1;
            }
        }
        if n_skip > 0 {
            ctx.push(
                Issue::builder(IssueCode::NonPositiveSeries)
                    .severity(signlred::Severity::Warning)
                    .message(format!("HalfNormalHmm skipped {n_skip} negative observations"))
                    .build(),
            );
        }
        let mut scale = Vector::from_iter((0..k).map(|j| 1.0 + j as f64));
        let mut start = init_start(k);
        let mut trans = init_trans(k);
        let mut loglik = f64::NEG_INFINITY;
        let mut last_gamma: Vec<Vec<f64>> = Vec::new();
        for it in 0..self.max_iter.max(1) {
            let dummy = FittedHalfNormalHmm {
                labels: Vector::zeros(0),
                start: start.clone(),
                trans: trans.clone(),
                scale: scale.clone(),
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
                let mut wy2 = 0.0_f64;
                for t in 0..t_len {
                    let y = x.get(t, 0);
                    if !y.is_finite() || y < 0.0 {
                        continue;
                    }
                    wsum += fb.gamma[t][j];
                    wy2 += fb.gamma[t][j] * y * y;
                }
                if wsum > 1e-12 {
                    scale[j] = (wy2 / wsum).max(COV_FLOOR).sqrt();
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
        let dummy = FittedHalfNormalHmm {
            labels: Vector::zeros(0),
            start: start.clone(),
            trans: trans.clone(),
            scale: scale.clone(),
            loglik,
        };
        let (labels, _) = viterbi_path(&start, &trans, &dummy.log_emit_seq(x));
        ctx.finish(FittedHalfNormalHmm {
            labels,
            start,
            trans,
            scale,
            loglik,
        })
    }
}

/// Maxwell–Boltzmann HMM on the positive line.
///
/// Scale is not identification `p`. Distinct from [`RayleighHmm`] (\(y\) vs
/// \(y^2\) weight) and [`HalfNormalHmm`] (no \(y^2\) prefactor).
#[derive(Clone, Debug)]
pub struct MaxwellHmm {
    /// Hidden states. Not identification `p`.
    pub n_states: usize,
    /// Baum–Welch cap.
    pub max_iter: usize,
}

impl Default for MaxwellHmm {
    fn default() -> Self {
        Self {
            n_states: 2,
            max_iter: 40,
        }
    }
}

impl MaxwellHmm {
    /// `k`-state Maxwell HMM.
    pub fn new(n_states: usize) -> Self {
        Self {
            n_states,
            ..Self::default()
        }
    }

    /// Fit alias.
    pub fn fit(&mut self, x: &Matrix, session: &Session) -> Result<Qualified<FittedMaxwellHmm>> {
        self.fit_unsupervised(x, session)
    }
}

/// Fitted Maxwell–Boltzmann HMM.
#[derive(Clone, Debug)]
pub struct FittedMaxwellHmm {
    /// Viterbi path.
    pub labels: Vector,
    /// Start distribution.
    pub start: Vector,
    /// Transitions.
    pub trans: Matrix,
    /// Scales \(a_j\).
    pub scale: Vector,
    /// Training log-likelihood.
    pub loglik: f64,
}

impl FittedMaxwellHmm {
    fn log_emit_seq(&self, x: &Matrix) -> Vec<Vec<f64>> {
        let t = x.nrows();
        let ns = self.scale.len();
        let mut out = vec![vec![f64::NEG_INFINITY; ns]; t];
        for ti in 0..t {
            let y = x.get(ti, 0);
            for j in 0..ns {
                out[ti][j] = log_maxwell(y, self.scale[j]);
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

impl Predict for FittedMaxwellHmm {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        self.decode(x, session)
    }
}

impl FitUnsupervised for MaxwellHmm {
    type Fitted = FittedMaxwellHmm;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedMaxwellHmm>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let t_len = x.nrows();
        let k = self.n_states.max(1);
        if t_len == 0 {
            return ctx.finish(FittedMaxwellHmm {
                labels: empty_labels(0),
                start: init_start(k),
                trans: init_trans(k),
                scale: Vector::filled(k, 1.0),
                loglik: f64::NAN,
            });
        }
        let mut n_skip = 0usize;
        for i in 0..t_len {
            if x.get(i, 0).is_finite() && x.get(i, 0) <= 0.0 {
                n_skip += 1;
            }
        }
        if n_skip > 0 {
            ctx.push(
                Issue::builder(IssueCode::NonPositiveSeries)
                    .severity(signlred::Severity::Warning)
                    .message(format!("MaxwellHmm skipped {n_skip} non-positive observations"))
                    .build(),
            );
        }
        let mut scale = Vector::from_iter((0..k).map(|j| 1.0 + 0.4 * j as f64));
        let mut start = init_start(k);
        let mut trans = init_trans(k);
        let mut loglik = f64::NEG_INFINITY;
        let mut last_gamma: Vec<Vec<f64>> = Vec::new();
        for it in 0..self.max_iter.max(1) {
            let dummy = FittedMaxwellHmm {
                labels: Vector::zeros(0),
                start: start.clone(),
                trans: trans.clone(),
                scale: scale.clone(),
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
                let mut wy2 = 0.0_f64;
                for t in 0..t_len {
                    let y = x.get(t, 0);
                    if !y.is_finite() || y <= 0.0 {
                        continue;
                    }
                    wsum += fb.gamma[t][j];
                    wy2 += fb.gamma[t][j] * y * y;
                }
                if wsum > 1e-12 {
                    scale[j] = (wy2 / (3.0 * wsum)).max(COV_FLOOR).sqrt();
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
        let dummy = FittedMaxwellHmm {
            labels: Vector::zeros(0),
            start: start.clone(),
            trans: trans.clone(),
            scale: scale.clone(),
            loglik,
        };
        let (labels, _) = viterbi_path(&start, &trans, &dummy.log_emit_seq(x));
        ctx.finish(FittedMaxwellHmm {
            labels,
            start,
            trans,
            scale,
            loglik,
        })
    }
}

/// Beta-prime (inverted-beta) HMM on the positive line.
///
/// Shapes are not identification `p`. Distinct from [`BetaHmm`] (unit
/// interval) and [`InverseGammaHmm`] (one shape).
#[derive(Clone, Debug)]
pub struct BetaPrimeHmm {
    /// Hidden states. Not identification `p`.
    pub n_states: usize,
    /// Baum–Welch cap.
    pub max_iter: usize,
}

impl Default for BetaPrimeHmm {
    fn default() -> Self {
        Self {
            n_states: 2,
            max_iter: 40,
        }
    }
}

impl BetaPrimeHmm {
    /// `k`-state beta-prime HMM.
    pub fn new(n_states: usize) -> Self {
        Self {
            n_states,
            ..Self::default()
        }
    }

    /// Fit alias.
    pub fn fit(&mut self, x: &Matrix, session: &Session) -> Result<Qualified<FittedBetaPrimeHmm>> {
        self.fit_unsupervised(x, session)
    }
}

/// Fitted beta-prime HMM.
#[derive(Clone, Debug)]
pub struct FittedBetaPrimeHmm {
    /// Viterbi path.
    pub labels: Vector,
    /// Start distribution.
    pub start: Vector,
    /// Transitions.
    pub trans: Matrix,
    /// First shapes \(\alpha_j\).
    pub alpha: Vector,
    /// Second shapes \(\beta_j\).
    pub beta: Vector,
    /// Training log-likelihood.
    pub loglik: f64,
}

impl FittedBetaPrimeHmm {
    fn log_emit_seq(&self, x: &Matrix) -> Vec<Vec<f64>> {
        let t = x.nrows();
        let ns = self.alpha.len();
        let mut out = vec![vec![f64::NEG_INFINITY; ns]; t];
        for ti in 0..t {
            let y = x.get(ti, 0);
            for j in 0..ns {
                out[ti][j] = log_beta_prime(y, self.alpha[j], self.beta[j]);
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

impl Predict for FittedBetaPrimeHmm {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        self.decode(x, session)
    }
}

impl FitUnsupervised for BetaPrimeHmm {
    type Fitted = FittedBetaPrimeHmm;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedBetaPrimeHmm>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let t_len = x.nrows();
        let k = self.n_states.max(1);
        if t_len == 0 {
            return ctx.finish(FittedBetaPrimeHmm {
                labels: empty_labels(0),
                start: init_start(k),
                trans: init_trans(k),
                alpha: Vector::filled(k, 2.0),
                beta: Vector::filled(k, 3.0),
                loglik: f64::NAN,
            });
        }
        let mut n_skip = 0usize;
        for i in 0..t_len {
            if x.get(i, 0).is_finite() && x.get(i, 0) <= 0.0 {
                n_skip += 1;
            }
        }
        if n_skip > 0 {
            ctx.push(
                Issue::builder(IssueCode::NonPositiveSeries)
                    .severity(signlred::Severity::Warning)
                    .message(format!("BetaPrimeHmm skipped {n_skip} non-positive observations"))
                    .build(),
            );
        }
        let mut alpha = Vector::from_iter((0..k).map(|j| 2.0 + j as f64));
        let mut beta = Vector::filled(k, 3.0);
        let mut start = init_start(k);
        let mut trans = init_trans(k);
        let mut loglik = f64::NEG_INFINITY;
        let mut last_gamma: Vec<Vec<f64>> = Vec::new();
        for it in 0..self.max_iter.max(1) {
            let dummy = FittedBetaPrimeHmm {
                labels: Vector::zeros(0),
                start: start.clone(),
                trans: trans.clone(),
                alpha: alpha.clone(),
                beta: beta.clone(),
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
                    if !y.is_finite() || y <= 0.0 {
                        continue;
                    }
                    wsum += fb.gamma[t][j];
                    wy += fb.gamma[t][j] * y;
                }
                if wsum > 1e-12 {
                    let m = (wy / wsum).max(1e-3);
                    beta[j] = 3.0;
                    alpha[j] = (m * (beta[j] - 1.0)).clamp(0.2, 40.0);
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
        let dummy = FittedBetaPrimeHmm {
            labels: Vector::zeros(0),
            start: start.clone(),
            trans: trans.clone(),
            alpha: alpha.clone(),
            beta: beta.clone(),
            loglik,
        };
        let (labels, _) = viterbi_path(&start, &trans, &dummy.log_emit_seq(x));
        ctx.finish(FittedBetaPrimeHmm {
            labels,
            start,
            trans,
            alpha,
            beta,
            loglik,
        })
    }
}

/// Gompertz lifetime HMM on the positive line.
///
/// Growth \(\eta\) is not identification `p`. Distinct from [`GumbelHmm`]
/// (unrestricted support) and [`WeibullHmm`] (power hazard).
#[derive(Clone, Debug)]
pub struct GompertzHmm {
    /// Hidden states. Not identification `p`.
    pub n_states: usize,
    /// Growth rate. Not identification `p`.
    pub eta: f64,
    /// Baum–Welch cap.
    pub max_iter: usize,
}

impl Default for GompertzHmm {
    fn default() -> Self {
        Self {
            n_states: 2,
            eta: 0.2,
            max_iter: 40,
        }
    }
}

impl GompertzHmm {
    /// `k`-state Gompertz HMM.
    pub fn new(n_states: usize) -> Self {
        Self {
            n_states,
            ..Self::default()
        }
    }

    /// Fit alias.
    pub fn fit(&mut self, x: &Matrix, session: &Session) -> Result<Qualified<FittedGompertzHmm>> {
        self.fit_unsupervised(x, session)
    }
}

/// Fitted Gompertz HMM.
#[derive(Clone, Debug)]
pub struct FittedGompertzHmm {
    /// Viterbi path.
    pub labels: Vector,
    /// Start distribution.
    pub start: Vector,
    /// Transitions.
    pub trans: Matrix,
    /// Growth rates \(\eta_j\).
    pub eta: Vector,
    /// Shapes \(c_j\).
    pub c: Vector,
    /// Training log-likelihood.
    pub loglik: f64,
}

impl FittedGompertzHmm {
    fn log_emit_seq(&self, x: &Matrix) -> Vec<Vec<f64>> {
        let t = x.nrows();
        let ns = self.c.len();
        let mut out = vec![vec![f64::NEG_INFINITY; ns]; t];
        for ti in 0..t {
            let y = x.get(ti, 0);
            for j in 0..ns {
                out[ti][j] = log_gompertz(y, self.eta[j], self.c[j]);
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

impl Predict for FittedGompertzHmm {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        self.decode(x, session)
    }
}

impl FitUnsupervised for GompertzHmm {
    type Fitted = FittedGompertzHmm;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedGompertzHmm>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let t_len = x.nrows();
        let k = self.n_states.max(1);
        let eta0 = if self.eta.is_finite() && self.eta > 0.0 {
            self.eta.clamp(0.01, 1.0)
        } else {
            0.2
        };
        if t_len == 0 {
            return ctx.finish(FittedGompertzHmm {
                labels: empty_labels(0),
                start: init_start(k),
                trans: init_trans(k),
                eta: Vector::filled(k, eta0),
                c: Vector::filled(k, 1.0),
                loglik: f64::NAN,
            });
        }
        let mut n_skip = 0usize;
        for i in 0..t_len {
            if x.get(i, 0).is_finite() && x.get(i, 0) <= 0.0 {
                n_skip += 1;
            }
        }
        if n_skip > 0 {
            ctx.push(
                Issue::builder(IssueCode::NonPositiveSeries)
                    .severity(signlred::Severity::Warning)
                    .message(format!("GompertzHmm skipped {n_skip} non-positive observations"))
                    .build(),
            );
        }
        let mut eta = Vector::filled(k, eta0);
        let mut c = Vector::from_iter((0..k).map(|j| 0.8 + 0.3 * j as f64));
        let mut start = init_start(k);
        let mut trans = init_trans(k);
        let mut loglik = f64::NEG_INFINITY;
        let mut last_gamma: Vec<Vec<f64>> = Vec::new();
        for it in 0..self.max_iter.max(1) {
            let dummy = FittedGompertzHmm {
                labels: Vector::zeros(0),
                start: start.clone(),
                trans: trans.clone(),
                eta: eta.clone(),
                c: c.clone(),
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
                let mut wex = 0.0_f64;
                for t in 0..t_len {
                    let y = x.get(t, 0);
                    if !y.is_finite() || y <= 0.0 {
                        continue;
                    }
                    wsum += fb.gamma[t][j];
                    wex += fb.gamma[t][j] * ((eta[j] * y).exp() - 1.0).max(1e-8);
                }
                if wsum > 1e-12 && wex > 1e-12 {
                    c[j] = (wsum / wex).clamp(0.05, 40.0);
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
        let dummy = FittedGompertzHmm {
            labels: Vector::zeros(0),
            start: start.clone(),
            trans: trans.clone(),
            eta: eta.clone(),
            c: c.clone(),
            loglik,
        };
        let (labels, _) = viterbi_path(&start, &trans, &dummy.log_emit_seq(x));
        ctx.finish(FittedGompertzHmm {
            labels,
            start,
            trans,
            eta,
            c,
            loglik,
        })
    }
}

fn ln_cosh(z: f64) -> f64 {
    let az = z.abs();
    az + (-2.0 * az).exp().ln_1p() - std::f64::consts::LN_2
}

fn log_hyperbolic_secant(y: f64, loc: f64, scale: f64) -> f64 {
    if !y.is_finite() || !loc.is_finite() || scale <= 0.0 {
        return f64::NEG_INFINITY;
    }
    let z = std::f64::consts::PI * (y - loc) / (2.0 * scale);
    -((2.0 * scale).ln()) - ln_cosh(z)
}

fn log_moyal(y: f64, loc: f64, scale: f64) -> f64 {
    if !y.is_finite() || !loc.is_finite() || scale <= 0.0 {
        return f64::NEG_INFINITY;
    }
    let z = (y - loc) / scale;
    let ez = (-z).exp();
    if !ez.is_finite() {
        return f64::NEG_INFINITY;
    }
    -0.5 * (z + ez) - scale.ln() - 0.5 * LN_2PI
}

fn log_f_dist(y: f64, d1: f64, d2: f64) -> f64 {
    if !y.is_finite() || y <= 0.0 || d1 <= 0.0 || d2 <= 0.0 {
        return f64::NEG_INFINITY;
    }
    let a = 0.5 * d1;
    let b = 0.5 * d2;
    let ratio = d1 / d2;
    -crate::special::ln_gamma(a) - crate::special::ln_gamma(b)
        + crate::special::ln_gamma(a + b)
        + a * ratio.ln()
        + (a - 1.0) * y.ln()
        - (a + b) * (1.0 + ratio * y).ln()
}

fn log_zipf_mandelbrot(y: f64, s: f64, q: f64, ln_z: f64) -> f64 {
    if !y.is_finite() || y < 1.0 - 1e-9 || s <= 1.0 || q < 0.0 || !ln_z.is_finite() {
        return f64::NEG_INFINITY;
    }
    let y = y.round().max(1.0);
    -s * (y + q).ln() - ln_z
}

fn zipf_mandelbrot_z(s: f64, q: f64) -> f64 {
    let mut acc = 0.0_f64;
    for k in 1..=192 {
        acc += ((k as f64) + q).powf(-s);
    }
    acc.max(1e-300)
}

fn log_hurdle_gamma(y: f64, pi0: f64, shape: f64, scale: f64) -> f64 {
    if !y.is_finite() || pi0 <= 0.0 || pi0 >= 1.0 || shape <= 0.0 || scale <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if y.abs() < 1e-12 {
        return pi0.ln();
    }
    if y < 0.0 {
        return f64::NEG_INFINITY;
    }
    (1.0 - pi0).ln() + log_gamma_emit(y, shape, 1.0 / scale)
}

fn ln_fact(n: i32) -> f64 {
    crate::special::ln_gamma(n as f64 + 1.0)
}

fn log_hermite(y: f64, lam1: f64, lam2: f64) -> f64 {
    if !y.is_finite() || y < -1e-9 || lam1 <= 0.0 || lam2 <= 0.0 {
        return f64::NEG_INFINITY;
    }
    let k = y.round().max(0.0) as i32;
    let mut terms = Vec::with_capacity((k / 2 + 1) as usize);
    let jmax = k / 2;
    for j in 0..=jmax {
        let m = k - 2 * j;
        terms.push(m as f64 * lam1.ln() + j as f64 * lam2.ln() - ln_fact(m) - ln_fact(j));
    }
    logsumexp(&terms) - lam1 - lam2
}

/// Hyperbolic-secant HMM on the real line.
///
/// Scales are not identification `p`. Distinct from [`LogisticHmm`] (logit
/// tails) and [`CauchyHmm`] (polynomial tails).
#[derive(Clone, Debug)]
pub struct HyperbolicSecantHmm {
    /// Hidden states. Not identification `p`.
    pub n_states: usize,
    /// Baum–Welch cap.
    pub max_iter: usize,
}

impl Default for HyperbolicSecantHmm {
    fn default() -> Self {
        Self {
            n_states: 2,
            max_iter: 40,
        }
    }
}

impl HyperbolicSecantHmm {
    /// `k`-state hyperbolic-secant HMM.
    pub fn new(n_states: usize) -> Self {
        Self {
            n_states,
            ..Self::default()
        }
    }

    /// Fit alias.
    pub fn fit(&mut self, x: &Matrix, session: &Session) -> Result<Qualified<FittedHyperbolicSecantHmm>> {
        self.fit_unsupervised(x, session)
    }
}

/// Fitted hyperbolic-secant HMM.
#[derive(Clone, Debug)]
pub struct FittedHyperbolicSecantHmm {
    /// Viterbi path.
    pub labels: Vector,
    /// Start distribution.
    pub start: Vector,
    /// Transitions.
    pub trans: Matrix,
    /// Locations.
    pub loc: Vector,
    /// Scales.
    pub scale: Vector,
    /// Training log-likelihood.
    pub loglik: f64,
}

impl FittedHyperbolicSecantHmm {
    fn log_emit_seq(&self, x: &Matrix) -> Vec<Vec<f64>> {
        let t = x.nrows();
        let ns = self.loc.len();
        let mut out = vec![vec![f64::NEG_INFINITY; ns]; t];
        for ti in 0..t {
            let y = x.get(ti, 0);
            for j in 0..ns {
                out[ti][j] = log_hyperbolic_secant(y, self.loc[j], self.scale[j]);
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

impl Predict for FittedHyperbolicSecantHmm {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        self.decode(x, session)
    }
}

impl FitUnsupervised for HyperbolicSecantHmm {
    type Fitted = FittedHyperbolicSecantHmm;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedHyperbolicSecantHmm>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let t_len = x.nrows();
        let k = self.n_states.max(1);
        if t_len == 0 {
            return ctx.finish(FittedHyperbolicSecantHmm {
                labels: empty_labels(0),
                start: init_start(k),
                trans: init_trans(k),
                loc: Vector::zeros(k),
                scale: Vector::filled(k, 1.0),
                loglik: f64::NAN,
            });
        }
        let mut loc = Vector::from_iter((0..k).map(|j| -1.0 + 2.0 * j as f64));
        let mut scale = Vector::filled(k, 1.0);
        let mut start = init_start(k);
        let mut trans = init_trans(k);
        let mut loglik = f64::NEG_INFINITY;
        let mut last_gamma: Vec<Vec<f64>> = Vec::new();
        for it in 0..self.max_iter.max(1) {
            let dummy = FittedHyperbolicSecantHmm {
                labels: Vector::zeros(0),
                start: start.clone(),
                trans: trans.clone(),
                loc: loc.clone(),
                scale: scale.clone(),
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
                    loc[j] = wy / wsum;
                    let mut wad = 0.0_f64;
                    for t in 0..t_len {
                        let y = x.get(t, 0);
                        if y.is_finite() {
                            wad += fb.gamma[t][j] * (y - loc[j]).abs();
                        }
                    }
                    scale[j] = (wad / wsum * std::f64::consts::FRAC_PI_2).max(COV_FLOOR.sqrt());
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
        let dummy = FittedHyperbolicSecantHmm {
            labels: Vector::zeros(0),
            start: start.clone(),
            trans: trans.clone(),
            loc: loc.clone(),
            scale: scale.clone(),
            loglik,
        };
        let (labels, _) = viterbi_path(&start, &trans, &dummy.log_emit_seq(x));
        ctx.finish(FittedHyperbolicSecantHmm {
            labels,
            start,
            trans,
            loc,
            scale,
            loglik,
        })
    }
}

/// Moyal HMM (extreme-value / Landau-adjacent) on the real line.
///
/// Scales are not identification `p`. Distinct from [`GumbelHmm`] (Gumbel
/// type-I) and [`GevHmm`] (three-parameter).
#[derive(Clone, Debug)]
pub struct MoyalHmm {
    /// Hidden states. Not identification `p`.
    pub n_states: usize,
    /// Baum–Welch cap.
    pub max_iter: usize,
}

impl Default for MoyalHmm {
    fn default() -> Self {
        Self {
            n_states: 2,
            max_iter: 40,
        }
    }
}

impl MoyalHmm {
    /// `k`-state Moyal HMM.
    pub fn new(n_states: usize) -> Self {
        Self {
            n_states,
            ..Self::default()
        }
    }

    /// Fit alias.
    pub fn fit(&mut self, x: &Matrix, session: &Session) -> Result<Qualified<FittedMoyalHmm>> {
        self.fit_unsupervised(x, session)
    }
}

/// Fitted Moyal HMM.
#[derive(Clone, Debug)]
pub struct FittedMoyalHmm {
    /// Viterbi path.
    pub labels: Vector,
    /// Start distribution.
    pub start: Vector,
    /// Transitions.
    pub trans: Matrix,
    /// Locations.
    pub loc: Vector,
    /// Scales.
    pub scale: Vector,
    /// Training log-likelihood.
    pub loglik: f64,
}

impl FittedMoyalHmm {
    fn log_emit_seq(&self, x: &Matrix) -> Vec<Vec<f64>> {
        let t = x.nrows();
        let ns = self.loc.len();
        let mut out = vec![vec![f64::NEG_INFINITY; ns]; t];
        for ti in 0..t {
            let y = x.get(ti, 0);
            for j in 0..ns {
                out[ti][j] = log_moyal(y, self.loc[j], self.scale[j]);
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

impl Predict for FittedMoyalHmm {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        self.decode(x, session)
    }
}

impl FitUnsupervised for MoyalHmm {
    type Fitted = FittedMoyalHmm;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedMoyalHmm>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let t_len = x.nrows();
        let k = self.n_states.max(1);
        if t_len == 0 {
            return ctx.finish(FittedMoyalHmm {
                labels: empty_labels(0),
                start: init_start(k),
                trans: init_trans(k),
                loc: Vector::zeros(k),
                scale: Vector::filled(k, 1.0),
                loglik: f64::NAN,
            });
        }
        let mut loc = Vector::from_iter((0..k).map(|j| -1.0 + 2.0 * j as f64));
        let mut scale = Vector::filled(k, 1.0);
        let mut start = init_start(k);
        let mut trans = init_trans(k);
        let mut loglik = f64::NEG_INFINITY;
        let mut last_gamma: Vec<Vec<f64>> = Vec::new();
        for it in 0..self.max_iter.max(1) {
            let dummy = FittedMoyalHmm {
                labels: Vector::zeros(0),
                start: start.clone(),
                trans: trans.clone(),
                loc: loc.clone(),
                scale: scale.clone(),
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
                    loc[j] = wy / wsum;
                    let mut w2 = 0.0_f64;
                    for t in 0..t_len {
                        let y = x.get(t, 0);
                        if y.is_finite() {
                            let e = y - loc[j];
                            w2 += fb.gamma[t][j] * e * e;
                        }
                    }
                    scale[j] = (w2 / wsum).max(COV_FLOOR).sqrt();
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
        let dummy = FittedMoyalHmm {
            labels: Vector::zeros(0),
            start: start.clone(),
            trans: trans.clone(),
            loc: loc.clone(),
            scale: scale.clone(),
            loglik,
        };
        let (labels, _) = viterbi_path(&start, &trans, &dummy.log_emit_seq(x));
        ctx.finish(FittedMoyalHmm {
            labels,
            start,
            trans,
            loc,
            scale,
            loglik,
        })
    }
}

/// Fisher–Snedecor \(F\) HMM on the positive line.
///
/// Numerator df is not identification `p`. Distinct from [`BetaPrimeHmm`]
/// (no \(d_1/d_2\) scaling) and [`GammaHmm`].
#[derive(Clone, Debug)]
pub struct FDistHmm {
    /// Hidden states. Not identification `p`.
    pub n_states: usize,
    /// Numerator degrees of freedom. Not identification `p`.
    pub d1: f64,
    /// Baum–Welch cap.
    pub max_iter: usize,
}

impl Default for FDistHmm {
    fn default() -> Self {
        Self {
            n_states: 2,
            d1: 4.0,
            max_iter: 40,
        }
    }
}

impl FDistHmm {
    /// `k`-state \(F\) HMM.
    pub fn new(n_states: usize) -> Self {
        Self {
            n_states,
            ..Self::default()
        }
    }

    /// Fit alias.
    pub fn fit(&mut self, x: &Matrix, session: &Session) -> Result<Qualified<FittedFDistHmm>> {
        self.fit_unsupervised(x, session)
    }
}

/// Fitted \(F\) HMM.
#[derive(Clone, Debug)]
pub struct FittedFDistHmm {
    /// Viterbi path.
    pub labels: Vector,
    /// Start distribution.
    pub start: Vector,
    /// Transitions.
    pub trans: Matrix,
    /// Numerator df.
    pub d1: Vector,
    /// Denominator df.
    pub d2: Vector,
    /// Training log-likelihood.
    pub loglik: f64,
}

impl FittedFDistHmm {
    fn log_emit_seq(&self, x: &Matrix) -> Vec<Vec<f64>> {
        let t = x.nrows();
        let ns = self.d2.len();
        let mut out = vec![vec![f64::NEG_INFINITY; ns]; t];
        for ti in 0..t {
            let y = x.get(ti, 0);
            for j in 0..ns {
                out[ti][j] = log_f_dist(y, self.d1[j], self.d2[j]);
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

impl Predict for FittedFDistHmm {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        self.decode(x, session)
    }
}

impl FitUnsupervised for FDistHmm {
    type Fitted = FittedFDistHmm;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedFDistHmm>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let t_len = x.nrows();
        let k = self.n_states.max(1);
        let d1_0 = if self.d1.is_finite() && self.d1 > 0.0 {
            self.d1.clamp(0.5, 40.0)
        } else {
            4.0
        };
        if t_len == 0 {
            return ctx.finish(FittedFDistHmm {
                labels: empty_labels(0),
                start: init_start(k),
                trans: init_trans(k),
                d1: Vector::filled(k, d1_0),
                d2: Vector::filled(k, 6.0),
                loglik: f64::NAN,
            });
        }
        let mut n_skip = 0usize;
        for i in 0..t_len {
            if x.get(i, 0).is_finite() && x.get(i, 0) <= 0.0 {
                n_skip += 1;
            }
        }
        if n_skip > 0 {
            ctx.push(
                Issue::builder(IssueCode::NonPositiveSeries)
                    .severity(signlred::Severity::Warning)
                    .message(format!("FDistHmm skipped {n_skip} non-positive observations"))
                    .build(),
            );
        }
        let mut d1 = Vector::filled(k, d1_0);
        let mut d2 = Vector::from_iter((0..k).map(|j| 5.0 + j as f64));
        let mut start = init_start(k);
        let mut trans = init_trans(k);
        let mut loglik = f64::NEG_INFINITY;
        let mut last_gamma: Vec<Vec<f64>> = Vec::new();
        for it in 0..self.max_iter.max(1) {
            let dummy = FittedFDistHmm {
                labels: Vector::zeros(0),
                start: start.clone(),
                trans: trans.clone(),
                d1: d1.clone(),
                d2: d2.clone(),
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
                    if !y.is_finite() || y <= 0.0 {
                        continue;
                    }
                    wsum += fb.gamma[t][j];
                    wy += fb.gamma[t][j] * y;
                }
                if wsum > 1e-12 {
                    let m = (wy / wsum).max(1.05);
                    d2[j] = (2.0 * m / (m - 1.0)).clamp(2.2, 80.0);
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
        let dummy = FittedFDistHmm {
            labels: Vector::zeros(0),
            start: start.clone(),
            trans: trans.clone(),
            d1: d1.clone(),
            d2: d2.clone(),
            loglik,
        };
        let (labels, _) = viterbi_path(&start, &trans, &dummy.log_emit_seq(x));
        ctx.finish(FittedFDistHmm {
            labels,
            start,
            trans,
            d1,
            d2,
            loglik,
        })
    }
}

/// Zipf–Mandelbrot HMM on \(\{1,2,\ldots\}\).
///
/// Shift \(q\) is not identification `p`. Distinct from [`ZetaHmm`] (\(q=0\)).
#[derive(Clone, Debug)]
pub struct ZipfMandelbrotHmm {
    /// Hidden states. Not identification `p`.
    pub n_states: usize,
    /// Mandelbrot shift. Not identification `p`.
    pub q: f64,
    /// Baum–Welch cap.
    pub max_iter: usize,
}

impl Default for ZipfMandelbrotHmm {
    fn default() -> Self {
        Self {
            n_states: 2,
            q: 1.0,
            max_iter: 40,
        }
    }
}

impl ZipfMandelbrotHmm {
    /// `k`-state Zipf–Mandelbrot HMM.
    pub fn new(n_states: usize) -> Self {
        Self {
            n_states,
            ..Self::default()
        }
    }

    /// Fit alias.
    pub fn fit(&mut self, x: &Matrix, session: &Session) -> Result<Qualified<FittedZipfMandelbrotHmm>> {
        self.fit_unsupervised(x, session)
    }
}

/// Fitted Zipf–Mandelbrot HMM.
#[derive(Clone, Debug)]
pub struct FittedZipfMandelbrotHmm {
    /// Viterbi path.
    pub labels: Vector,
    /// Start distribution.
    pub start: Vector,
    /// Transitions.
    pub trans: Matrix,
    /// Exponents \(s_j>1\).
    pub s: Vector,
    /// Shifts \(q_j\).
    pub q: Vector,
    /// Training log-likelihood.
    pub loglik: f64,
}

impl FittedZipfMandelbrotHmm {
    fn log_emit_seq(&self, x: &Matrix) -> Vec<Vec<f64>> {
        let t = x.nrows();
        let ns = self.s.len();
        let ln_z: Vec<f64> = (0..ns)
            .map(|j| zipf_mandelbrot_z(self.s[j], self.q[j]).ln())
            .collect();
        let mut out = vec![vec![f64::NEG_INFINITY; ns]; t];
        for ti in 0..t {
            let y = x.get(ti, 0);
            for j in 0..ns {
                out[ti][j] = log_zipf_mandelbrot(y, self.s[j], self.q[j], ln_z[j]);
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

impl Predict for FittedZipfMandelbrotHmm {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        self.decode(x, session)
    }
}

impl FitUnsupervised for ZipfMandelbrotHmm {
    type Fitted = FittedZipfMandelbrotHmm;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedZipfMandelbrotHmm>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let t_len = x.nrows();
        let k = self.n_states.max(1);
        let q0 = if self.q.is_finite() && self.q >= 0.0 {
            self.q
        } else {
            1.0
        };
        if t_len == 0 {
            return ctx.finish(FittedZipfMandelbrotHmm {
                labels: empty_labels(0),
                start: init_start(k),
                trans: init_trans(k),
                s: Vector::filled(k, 2.0),
                q: Vector::filled(k, q0),
                loglik: f64::NAN,
            });
        }
        let mut s = Vector::from_iter((0..k).map(|j| 1.6 + 0.3 * j as f64));
        let mut q = Vector::filled(k, q0);
        let mut start = init_start(k);
        let mut trans = init_trans(k);
        let mut loglik = f64::NEG_INFINITY;
        let mut last_gamma: Vec<Vec<f64>> = Vec::new();
        for it in 0..self.max_iter.max(1) {
            let dummy = FittedZipfMandelbrotHmm {
                labels: Vector::zeros(0),
                start: start.clone(),
                trans: trans.clone(),
                s: s.clone(),
                q: q.clone(),
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
                    if !y.is_finite() || y < 1.0 - 1e-9 {
                        continue;
                    }
                    wsum += fb.gamma[t][j];
                    wy += fb.gamma[t][j] * y;
                }
                if wsum > 1e-12 {
                    let m = (wy / wsum).max(1.02);
                    s[j] = (1.0 + 1.0 / (m + q[j] - 1.0).max(0.05)).clamp(1.15, 12.0);
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
        let dummy = FittedZipfMandelbrotHmm {
            labels: Vector::zeros(0),
            start: start.clone(),
            trans: trans.clone(),
            s: s.clone(),
            q: q.clone(),
            loglik,
        };
        let (labels, _) = viterbi_path(&start, &trans, &dummy.log_emit_seq(x));
        ctx.finish(FittedZipfMandelbrotHmm {
            labels,
            start,
            trans,
            s,
            q,
            loglik,
        })
    }
}

/// Hurdle-gamma HMM (zeros only from a point mass).
///
/// State count is not identification `p`. Distinct from [`ZeroInflatedGammaHmm`]
/// (mixture zeros) and [`GammaHmm`] (no atom).
#[derive(Clone, Debug)]
pub struct HurdleGammaHmm {
    /// Hidden states. Not identification `p`.
    pub n_states: usize,
    /// Baum–Welch cap.
    pub max_iter: usize,
}

impl Default for HurdleGammaHmm {
    fn default() -> Self {
        Self {
            n_states: 2,
            max_iter: 40,
        }
    }
}

impl HurdleGammaHmm {
    /// `k`-state hurdle-gamma HMM.
    pub fn new(n_states: usize) -> Self {
        Self {
            n_states,
            ..Self::default()
        }
    }

    /// Fit alias.
    pub fn fit(&mut self, x: &Matrix, session: &Session) -> Result<Qualified<FittedHurdleGammaHmm>> {
        self.fit_unsupervised(x, session)
    }
}

/// Fitted hurdle-gamma HMM.
#[derive(Clone, Debug)]
pub struct FittedHurdleGammaHmm {
    /// Viterbi path.
    pub labels: Vector,
    /// Start distribution.
    pub start: Vector,
    /// Transitions.
    pub trans: Matrix,
    /// Zero masses \(\pi_j\).
    pub pi0: Vector,
    /// Shapes.
    pub shape: Vector,
    /// Scales.
    pub scale: Vector,
    /// Training log-likelihood.
    pub loglik: f64,
}

impl FittedHurdleGammaHmm {
    fn log_emit_seq(&self, x: &Matrix) -> Vec<Vec<f64>> {
        let t = x.nrows();
        let ns = self.pi0.len();
        let mut out = vec![vec![f64::NEG_INFINITY; ns]; t];
        for ti in 0..t {
            let y = x.get(ti, 0);
            for j in 0..ns {
                out[ti][j] = log_hurdle_gamma(y, self.pi0[j], self.shape[j], self.scale[j]);
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

impl Predict for FittedHurdleGammaHmm {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        self.decode(x, session)
    }
}

impl FitUnsupervised for HurdleGammaHmm {
    type Fitted = FittedHurdleGammaHmm;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedHurdleGammaHmm>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let t_len = x.nrows();
        let k = self.n_states.max(1);
        if t_len == 0 {
            return ctx.finish(FittedHurdleGammaHmm {
                labels: empty_labels(0),
                start: init_start(k),
                trans: init_trans(k),
                pi0: Vector::filled(k, 0.1),
                shape: Vector::filled(k, 2.0),
                scale: Vector::filled(k, 1.0),
                loglik: f64::NAN,
            });
        }
        let mut n_skip = 0usize;
        for i in 0..t_len {
            if x.get(i, 0).is_finite() && x.get(i, 0) < 0.0 {
                n_skip += 1;
            }
        }
        if n_skip > 0 {
            ctx.push(
                Issue::builder(IssueCode::NonPositiveSeries)
                    .severity(signlred::Severity::Warning)
                    .message(format!("HurdleGammaHmm skipped {n_skip} negative observations"))
                    .build(),
            );
        }
        let mut pi0 = Vector::filled(k, 0.05);
        let mut shape = Vector::from_iter((0..k).map(|j| 1.5 + 0.5 * j as f64));
        let mut scale = Vector::from_iter((0..k).map(|j| 1.0 + 0.4 * j as f64));
        let mut start = init_start(k);
        let mut trans = init_trans(k);
        let mut loglik = f64::NEG_INFINITY;
        let mut last_gamma: Vec<Vec<f64>> = Vec::new();
        for it in 0..self.max_iter.max(1) {
            let dummy = FittedHurdleGammaHmm {
                labels: Vector::zeros(0),
                start: start.clone(),
                trans: trans.clone(),
                pi0: pi0.clone(),
                shape: shape.clone(),
                scale: scale.clone(),
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
                let mut wz = 0.0_f64;
                let mut wy = 0.0_f64;
                let mut wln = 0.0_f64;
                for t in 0..t_len {
                    let y = x.get(t, 0);
                    if !y.is_finite() || y < 0.0 {
                        continue;
                    }
                    wsum += fb.gamma[t][j];
                    if y.abs() < 1e-12 {
                        wz += fb.gamma[t][j];
                    } else {
                        wy += fb.gamma[t][j] * y;
                        wln += fb.gamma[t][j] * y.ln();
                    }
                }
                if wsum > 1e-12 {
                    pi0[j] = (wz / wsum).clamp(1e-4, 1.0 - 1e-4);
                    let wpos = (wsum - wz).max(1e-12);
                    let m = (wy / wpos).max(1e-4);
                    let ml = wln / wpos;
                    let s2 = (m.ln() - ml).max(1e-4);
                    shape[j] = (0.5 / s2).clamp(0.2, 40.0);
                    scale[j] = (m / shape[j]).max(1e-4);
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
        let dummy = FittedHurdleGammaHmm {
            labels: Vector::zeros(0),
            start: start.clone(),
            trans: trans.clone(),
            pi0: pi0.clone(),
            shape: shape.clone(),
            scale: scale.clone(),
            loglik,
        };
        let (labels, _) = viterbi_path(&start, &trans, &dummy.log_emit_seq(x));
        ctx.finish(FittedHurdleGammaHmm {
            labels,
            start,
            trans,
            pi0,
            shape,
            scale,
            loglik,
        })
    }
}

/// Hermite (Poisson-stopped Poisson) HMM on \(\{0,1,\ldots\}\).
///
/// Rates are not identification `p`. Distinct from [`PoissonHmm`] (one rate)
/// and [`GenPoissonHmm`] (dispersion tilt).
#[derive(Clone, Debug)]
pub struct HermiteHmm {
    /// Hidden states. Not identification `p`.
    pub n_states: usize,
    /// Baum–Welch cap.
    pub max_iter: usize,
}

impl Default for HermiteHmm {
    fn default() -> Self {
        Self {
            n_states: 2,
            max_iter: 40,
        }
    }
}

impl HermiteHmm {
    /// `k`-state Hermite HMM.
    pub fn new(n_states: usize) -> Self {
        Self {
            n_states,
            ..Self::default()
        }
    }

    /// Fit alias.
    pub fn fit(&mut self, x: &Matrix, session: &Session) -> Result<Qualified<FittedHermiteHmm>> {
        self.fit_unsupervised(x, session)
    }
}

/// Fitted Hermite HMM.
#[derive(Clone, Debug)]
pub struct FittedHermiteHmm {
    /// Viterbi path.
    pub labels: Vector,
    /// Start distribution.
    pub start: Vector,
    /// Transitions.
    pub trans: Matrix,
    /// First Poisson rates.
    pub lam1: Vector,
    /// Second Poisson rates.
    pub lam2: Vector,
    /// Training log-likelihood.
    pub loglik: f64,
}

impl FittedHermiteHmm {
    fn log_emit_seq(&self, x: &Matrix) -> Vec<Vec<f64>> {
        let t = x.nrows();
        let ns = self.lam1.len();
        let mut out = vec![vec![f64::NEG_INFINITY; ns]; t];
        for ti in 0..t {
            let y = x.get(ti, 0);
            for j in 0..ns {
                out[ti][j] = log_hermite(y, self.lam1[j], self.lam2[j]);
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

impl Predict for FittedHermiteHmm {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        self.decode(x, session)
    }
}

impl FitUnsupervised for HermiteHmm {
    type Fitted = FittedHermiteHmm;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedHermiteHmm>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let t_len = x.nrows();
        let k = self.n_states.max(1);
        if t_len == 0 {
            return ctx.finish(FittedHermiteHmm {
                labels: empty_labels(0),
                start: init_start(k),
                trans: init_trans(k),
                lam1: Vector::filled(k, 1.0),
                lam2: Vector::filled(k, 0.3),
                loglik: f64::NAN,
            });
        }
        let mut lam1 = Vector::from_iter((0..k).map(|j| 1.0 + j as f64));
        let mut lam2 = Vector::filled(k, 0.4);
        let mut start = init_start(k);
        let mut trans = init_trans(k);
        let mut loglik = f64::NEG_INFINITY;
        let mut last_gamma: Vec<Vec<f64>> = Vec::new();
        for it in 0..self.max_iter.max(1) {
            let dummy = FittedHermiteHmm {
                labels: Vector::zeros(0),
                start: start.clone(),
                trans: trans.clone(),
                lam1: lam1.clone(),
                lam2: lam2.clone(),
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
                    if !y.is_finite() || y < -1e-9 {
                        continue;
                    }
                    wsum += fb.gamma[t][j];
                    wy += fb.gamma[t][j] * y;
                    wy2 += fb.gamma[t][j] * y * y;
                }
                if wsum > 1e-12 {
                    let m = (wy / wsum).max(0.05);
                    let v = (wy2 / wsum - m * m).max(m);
                    let l2 = ((v - m) * 0.5).clamp(0.05, 20.0);
                    let l1 = (m - 2.0 * l2).max(0.05);
                    lam1[j] = l1;
                    lam2[j] = l2;
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
        let dummy = FittedHermiteHmm {
            labels: Vector::zeros(0),
            start: start.clone(),
            trans: trans.clone(),
            lam1: lam1.clone(),
            lam2: lam2.clone(),
            loglik,
        };
        let (labels, _) = viterbi_path(&start, &trans, &dummy.log_emit_seq(x));
        ctx.finish(FittedHermiteHmm {
            labels,
            start,
            trans,
            lam1,
            lam2,
            loglik,
        })
    }
}

fn wrap_tau(y: f64) -> f64 {
    let t = std::f64::consts::TAU;
    let r = y % t;
    if r < 0.0 {
        r + t
    } else {
        r
    }
}

fn log_wrapped_exp(y: f64, rate: f64) -> f64 {
    if !y.is_finite() || rate <= 0.0 {
        return f64::NEG_INFINITY;
    }
    let th = wrap_tau(y);
    let den = 1.0 - (-rate * std::f64::consts::TAU).exp();
    if den <= 1e-15 {
        return f64::NEG_INFINITY;
    }
    rate.ln() - rate * th - den.ln()
}

fn log_chi_emit(y: f64, df: f64) -> f64 {
    if !y.is_finite() || y <= 0.0 || df <= 0.0 {
        return f64::NEG_INFINITY;
    }
    let a = 0.5 * df;
    -(a - 1.0) * std::f64::consts::LN_2 - crate::special::ln_gamma(a) + (df - 1.0) * y.ln()
        - 0.5 * y * y
}

fn log_delaporte(y: f64, lam: f64, alpha: f64, beta: f64) -> f64 {
    if !y.is_finite() || y < -1e-9 || lam <= 0.0 || alpha <= 0.0 || beta <= 0.0 {
        return f64::NEG_INFINITY;
    }
    let k = y.round().max(0.0) as i32;
    let mut terms = Vec::with_capacity((k + 1) as usize);
    let p0 = beta / (1.0 + beta);
    for i in 0..=k {
        let m = k - i;
        let po = m as f64 * lam.ln() - lam - ln_fact(m);
        let nb = crate::special::ln_gamma(alpha + i as f64) - ln_fact(i)
            - crate::special::ln_gamma(alpha)
            + alpha * p0.ln()
            + i as f64 * (1.0 - p0).ln();
        terms.push(po + nb);
    }
    logsumexp(&terms)
}

fn log_neyman_a(y: f64, lam: f64, phi: f64) -> f64 {
    if !y.is_finite() || y < -1e-9 || lam <= 0.0 || phi <= 0.0 {
        return f64::NEG_INFINITY;
    }
    let k = y.round().max(0.0) as i32;
    if k == 0 {
        return -lam + lam * (-phi).exp();
    }
    let jmax = (k + 12).max(8);
    let mut terms = Vec::with_capacity(jmax as usize);
    for j in 1..=jmax {
        let jf = j as f64;
        terms.push(
            -lam + jf * lam.ln() - ln_fact(j) - jf * phi + k as f64 * (jf * phi).ln() - ln_fact(k),
        );
    }
    logsumexp(&terms)
}

/// Wrapped-exponential HMM on the circle.
///
/// Rate is not identification `p`. Distinct from [`CircularHmm`] (von Mises)
/// and [`CardioidHmm`] (cosine tilt).
#[derive(Clone, Debug)]
pub struct WrappedExponentialHmm {
    /// Hidden states. Not identification `p`.
    pub n_states: usize,
    /// Baum–Welch cap.
    pub max_iter: usize,
}

impl Default for WrappedExponentialHmm {
    fn default() -> Self {
        Self {
            n_states: 2,
            max_iter: 40,
        }
    }
}

impl WrappedExponentialHmm {
    /// `k`-state wrapped-exponential HMM.
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
    ) -> Result<Qualified<FittedWrappedExponentialHmm>> {
        self.fit_unsupervised(x, session)
    }
}

/// Fitted wrapped-exponential HMM.
#[derive(Clone, Debug)]
pub struct FittedWrappedExponentialHmm {
    /// Viterbi path.
    pub labels: Vector,
    /// Start distribution.
    pub start: Vector,
    /// Transitions.
    pub trans: Matrix,
    /// Rates.
    pub rate: Vector,
    /// Training log-likelihood.
    pub loglik: f64,
}

impl FittedWrappedExponentialHmm {
    fn log_emit_seq(&self, x: &Matrix) -> Vec<Vec<f64>> {
        let t = x.nrows();
        let ns = self.rate.len();
        let mut out = vec![vec![f64::NEG_INFINITY; ns]; t];
        for ti in 0..t {
            let y = x.get(ti, 0);
            for j in 0..ns {
                out[ti][j] = log_wrapped_exp(y, self.rate[j]);
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

impl Predict for FittedWrappedExponentialHmm {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        self.decode(x, session)
    }
}

impl FitUnsupervised for WrappedExponentialHmm {
    type Fitted = FittedWrappedExponentialHmm;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedWrappedExponentialHmm>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let t_len = x.nrows();
        let k = self.n_states.max(1);
        if t_len == 0 {
            return ctx.finish(FittedWrappedExponentialHmm {
                labels: empty_labels(0),
                start: init_start(k),
                trans: init_trans(k),
                rate: Vector::filled(k, 1.0),
                loglik: f64::NAN,
            });
        }
        let mut rate = Vector::from_iter((0..k).map(|j| 0.6 + 0.4 * j as f64));
        let mut start = init_start(k);
        let mut trans = init_trans(k);
        let mut loglik = f64::NEG_INFINITY;
        let mut last_gamma: Vec<Vec<f64>> = Vec::new();
        for it in 0..self.max_iter.max(1) {
            let dummy = FittedWrappedExponentialHmm {
                labels: Vector::zeros(0),
                start: start.clone(),
                trans: trans.clone(),
                rate: rate.clone(),
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
                    wy += fb.gamma[t][j] * wrap_tau(y);
                }
                if wsum > 1e-12 {
                    rate[j] = (1.0 / (wy / wsum).max(0.05)).clamp(0.05, 20.0);
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
        let dummy = FittedWrappedExponentialHmm {
            labels: Vector::zeros(0),
            start: start.clone(),
            trans: trans.clone(),
            rate: rate.clone(),
            loglik,
        };
        let (labels, _) = viterbi_path(&start, &trans, &dummy.log_emit_seq(x));
        ctx.finish(FittedWrappedExponentialHmm {
            labels,
            start,
            trans,
            rate,
            loglik,
        })
    }
}

/// Chi HMM on the positive line.
///
/// Degrees of freedom are not identification `p`. Distinct from
/// [`NakagamiHmm`] (m, Ω) and [`RayleighHmm`] (df = 2).
#[derive(Clone, Debug)]
pub struct ChiHmm {
    /// Hidden states. Not identification `p`.
    pub n_states: usize,
    /// Baum–Welch cap.
    pub max_iter: usize,
}

impl Default for ChiHmm {
    fn default() -> Self {
        Self {
            n_states: 2,
            max_iter: 40,
        }
    }
}

impl ChiHmm {
    /// `k`-state chi HMM.
    pub fn new(n_states: usize) -> Self {
        Self {
            n_states,
            ..Self::default()
        }
    }

    /// Fit alias.
    pub fn fit(&mut self, x: &Matrix, session: &Session) -> Result<Qualified<FittedChiHmm>> {
        self.fit_unsupervised(x, session)
    }
}

/// Fitted chi HMM.
#[derive(Clone, Debug)]
pub struct FittedChiHmm {
    /// Viterbi path.
    pub labels: Vector,
    /// Start distribution.
    pub start: Vector,
    /// Transitions.
    pub trans: Matrix,
    /// Degrees of freedom.
    pub df: Vector,
    /// Training log-likelihood.
    pub loglik: f64,
}

impl FittedChiHmm {
    fn log_emit_seq(&self, x: &Matrix) -> Vec<Vec<f64>> {
        let t = x.nrows();
        let ns = self.df.len();
        let mut out = vec![vec![f64::NEG_INFINITY; ns]; t];
        for ti in 0..t {
            let y = x.get(ti, 0);
            for j in 0..ns {
                out[ti][j] = log_chi_emit(y, self.df[j]);
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

impl Predict for FittedChiHmm {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        self.decode(x, session)
    }
}

impl FitUnsupervised for ChiHmm {
    type Fitted = FittedChiHmm;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedChiHmm>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let t_len = x.nrows();
        let k = self.n_states.max(1);
        if t_len == 0 {
            return ctx.finish(FittedChiHmm {
                labels: empty_labels(0),
                start: init_start(k),
                trans: init_trans(k),
                df: Vector::filled(k, 2.0),
                loglik: f64::NAN,
            });
        }
        let mut n_skip = 0usize;
        for i in 0..t_len {
            if x.get(i, 0).is_finite() && x.get(i, 0) <= 0.0 {
                n_skip += 1;
            }
        }
        if n_skip > 0 {
            ctx.push(
                Issue::builder(IssueCode::NonPositiveSeries)
                    .severity(signlred::Severity::Warning)
                    .message(format!("ChiHmm skipped {n_skip} non-positive observations"))
                    .build(),
            );
        }
        let mut df = Vector::from_iter((0..k).map(|j| 2.0 + j as f64));
        let mut start = init_start(k);
        let mut trans = init_trans(k);
        let mut loglik = f64::NEG_INFINITY;
        let mut last_gamma: Vec<Vec<f64>> = Vec::new();
        for it in 0..self.max_iter.max(1) {
            let dummy = FittedChiHmm {
                labels: Vector::zeros(0),
                start: start.clone(),
                trans: trans.clone(),
                df: df.clone(),
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
                let mut wy2 = 0.0_f64;
                for t in 0..t_len {
                    let y = x.get(t, 0);
                    if !y.is_finite() || y <= 0.0 {
                        continue;
                    }
                    wsum += fb.gamma[t][j];
                    wy2 += fb.gamma[t][j] * y * y;
                }
                if wsum > 1e-12 {
                    df[j] = (wy2 / wsum).clamp(0.5, 40.0);
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
        let dummy = FittedChiHmm {
            labels: Vector::zeros(0),
            start: start.clone(),
            trans: trans.clone(),
            df: df.clone(),
            loglik,
        };
        let (labels, _) = viterbi_path(&start, &trans, &dummy.log_emit_seq(x));
        ctx.finish(FittedChiHmm {
            labels,
            start,
            trans,
            df,
            loglik,
        })
    }
}

/// Delaporte HMM (Poisson plus negative binomial).
///
/// Extra Poisson rate is not identification `p`. Distinct from
/// [`NegativeBinomialHmm`] (no Poisson offset) and [`HermiteHmm`].
#[derive(Clone, Debug)]
pub struct DelaporteHmm {
    /// Hidden states. Not identification `p`.
    pub n_states: usize,
    /// Baum–Welch cap.
    pub max_iter: usize,
}

impl Default for DelaporteHmm {
    fn default() -> Self {
        Self {
            n_states: 2,
            max_iter: 40,
        }
    }
}

impl DelaporteHmm {
    /// `k`-state Delaporte HMM.
    pub fn new(n_states: usize) -> Self {
        Self {
            n_states,
            ..Self::default()
        }
    }

    /// Fit alias.
    pub fn fit(&mut self, x: &Matrix, session: &Session) -> Result<Qualified<FittedDelaporteHmm>> {
        self.fit_unsupervised(x, session)
    }
}

/// Fitted Delaporte HMM.
#[derive(Clone, Debug)]
pub struct FittedDelaporteHmm {
    /// Viterbi path.
    pub labels: Vector,
    /// Start distribution.
    pub start: Vector,
    /// Transitions.
    pub trans: Matrix,
    /// Poisson offsets.
    pub lam: Vector,
    /// NB shapes.
    pub alpha: Vector,
    /// Training log-likelihood.
    pub loglik: f64,
}

impl FittedDelaporteHmm {
    fn log_emit_seq(&self, x: &Matrix) -> Vec<Vec<f64>> {
        let t = x.nrows();
        let ns = self.lam.len();
        let mut out = vec![vec![f64::NEG_INFINITY; ns]; t];
        for ti in 0..t {
            let y = x.get(ti, 0);
            for j in 0..ns {
                out[ti][j] = log_delaporte(y, self.lam[j], self.alpha[j], 1.0);
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

impl Predict for FittedDelaporteHmm {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        self.decode(x, session)
    }
}

impl FitUnsupervised for DelaporteHmm {
    type Fitted = FittedDelaporteHmm;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedDelaporteHmm>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let t_len = x.nrows();
        let k = self.n_states.max(1);
        if t_len == 0 {
            return ctx.finish(FittedDelaporteHmm {
                labels: empty_labels(0),
                start: init_start(k),
                trans: init_trans(k),
                lam: Vector::filled(k, 1.0),
                alpha: Vector::filled(k, 1.0),
                loglik: f64::NAN,
            });
        }
        let mut lam = Vector::from_iter((0..k).map(|j| 0.8 + 0.6 * j as f64));
        let mut alpha = Vector::from_iter((0..k).map(|j| 1.0 + 0.5 * j as f64));
        let mut start = init_start(k);
        let mut trans = init_trans(k);
        let mut loglik = f64::NEG_INFINITY;
        let mut last_gamma: Vec<Vec<f64>> = Vec::new();
        for it in 0..self.max_iter.max(1) {
            let dummy = FittedDelaporteHmm {
                labels: Vector::zeros(0),
                start: start.clone(),
                trans: trans.clone(),
                lam: lam.clone(),
                alpha: alpha.clone(),
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
                    if !y.is_finite() || y < -1e-9 {
                        continue;
                    }
                    wsum += fb.gamma[t][j];
                    wy += fb.gamma[t][j] * y;
                }
                if wsum > 1e-12 {
                    let m = (wy / wsum).max(0.2);
                    lam[j] = (0.4 * m).max(0.05);
                    alpha[j] = (0.6 * m).max(0.2);
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
        let dummy = FittedDelaporteHmm {
            labels: Vector::zeros(0),
            start: start.clone(),
            trans: trans.clone(),
            lam: lam.clone(),
            alpha: alpha.clone(),
            loglik,
        };
        let (labels, _) = viterbi_path(&start, &trans, &dummy.log_emit_seq(x));
        ctx.finish(FittedDelaporteHmm {
            labels,
            start,
            trans,
            lam,
            alpha,
            loglik,
        })
    }
}

/// Neyman Type A HMM (Poisson-stopped Poisson).
///
/// Cluster rate is not identification `p`. Distinct from [`HermiteHmm`]
/// (two additive Poissons) and [`PoissonHmm`].
#[derive(Clone, Debug)]
pub struct NeymanTypeAHmm {
    /// Hidden states. Not identification `p`.
    pub n_states: usize,
    /// Baum–Welch cap.
    pub max_iter: usize,
}

impl Default for NeymanTypeAHmm {
    fn default() -> Self {
        Self {
            n_states: 2,
            max_iter: 40,
        }
    }
}

impl NeymanTypeAHmm {
    /// `k`-state Neyman Type A HMM.
    pub fn new(n_states: usize) -> Self {
        Self {
            n_states,
            ..Self::default()
        }
    }

    /// Fit alias.
    pub fn fit(&mut self, x: &Matrix, session: &Session) -> Result<Qualified<FittedNeymanTypeAHmm>> {
        self.fit_unsupervised(x, session)
    }
}

/// Fitted Neyman Type A HMM.
#[derive(Clone, Debug)]
pub struct FittedNeymanTypeAHmm {
    /// Viterbi path.
    pub labels: Vector,
    /// Start distribution.
    pub start: Vector,
    /// Transitions.
    pub trans: Matrix,
    /// Cluster Poisson rates.
    pub lam: Vector,
    /// Offspring rates.
    pub phi: Vector,
    /// Training log-likelihood.
    pub loglik: f64,
}

impl FittedNeymanTypeAHmm {
    fn log_emit_seq(&self, x: &Matrix) -> Vec<Vec<f64>> {
        let t = x.nrows();
        let ns = self.lam.len();
        let mut out = vec![vec![f64::NEG_INFINITY; ns]; t];
        for ti in 0..t {
            let y = x.get(ti, 0);
            for j in 0..ns {
                out[ti][j] = log_neyman_a(y, self.lam[j], self.phi[j]);
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

impl Predict for FittedNeymanTypeAHmm {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        self.decode(x, session)
    }
}

impl FitUnsupervised for NeymanTypeAHmm {
    type Fitted = FittedNeymanTypeAHmm;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedNeymanTypeAHmm>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let t_len = x.nrows();
        let k = self.n_states.max(1);
        if t_len == 0 {
            return ctx.finish(FittedNeymanTypeAHmm {
                labels: empty_labels(0),
                start: init_start(k),
                trans: init_trans(k),
                lam: Vector::filled(k, 1.0),
                phi: Vector::filled(k, 1.0),
                loglik: f64::NAN,
            });
        }
        let mut lam = Vector::from_iter((0..k).map(|j| 0.8 + 0.5 * j as f64));
        let mut phi = Vector::filled(k, 1.2);
        let mut start = init_start(k);
        let mut trans = init_trans(k);
        let mut loglik = f64::NEG_INFINITY;
        let mut last_gamma: Vec<Vec<f64>> = Vec::new();
        for it in 0..self.max_iter.max(1) {
            let dummy = FittedNeymanTypeAHmm {
                labels: Vector::zeros(0),
                start: start.clone(),
                trans: trans.clone(),
                lam: lam.clone(),
                phi: phi.clone(),
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
                    if !y.is_finite() || y < -1e-9 {
                        continue;
                    }
                    wsum += fb.gamma[t][j];
                    wy += fb.gamma[t][j] * y;
                    wy2 += fb.gamma[t][j] * y * y;
                }
                if wsum > 1e-12 {
                    let m = (wy / wsum).max(0.2);
                    let v = (wy2 / wsum - m * m).max(m);
                    let ph = (v / m).clamp(0.2, 20.0);
                    phi[j] = ph;
                    lam[j] = (m / ph).max(0.05);
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
        let dummy = FittedNeymanTypeAHmm {
            labels: Vector::zeros(0),
            start: start.clone(),
            trans: trans.clone(),
            lam: lam.clone(),
            phi: phi.clone(),
            loglik,
        };
        let (labels, _) = viterbi_path(&start, &trans, &dummy.log_emit_seq(x));
        ctx.finish(FittedNeymanTypeAHmm {
            labels,
            start,
            trans,
            lam,
            phi,
            loglik,
        })
    }
}

fn log_arcsine(y: f64, alpha: f64) -> f64 {
    if !y.is_finite() || y <= 0.0 || y >= 1.0 || alpha <= 0.0 || alpha >= 1.0 {
        return f64::NEG_INFINITY;
    }
    (alpha - 1.0) * y.ln() - alpha * (1.0 - y).ln()
        - crate::special::ln_gamma(alpha)
        - crate::special::ln_gamma(1.0 - alpha)
}

fn log_power_unit(y: f64, alpha: f64) -> f64 {
    if !y.is_finite() || y <= 0.0 || y >= 1.0 || alpha <= 0.0 {
        return f64::NEG_INFINITY;
    }
    alpha.ln() + (alpha - 1.0) * y.ln()
}

fn log_raised_cosine(y: f64, loc: f64, scale: f64) -> f64 {
    if !y.is_finite() || !loc.is_finite() || scale <= 0.0 {
        return f64::NEG_INFINITY;
    }
    let z = (y - loc) / scale;
    if z.abs() >= 1.0 {
        return f64::NEG_INFINITY;
    }
    -(2.0 * scale).ln() + (1.0 + (std::f64::consts::PI * z).cos()).max(1e-15).ln()
}

fn log_log_uniform(y: f64, lo: f64, hi: f64) -> f64 {
    if !y.is_finite() || y <= 0.0 || lo <= 0.0 || hi <= lo {
        return f64::NEG_INFINITY;
    }
    if y < lo || y > hi {
        return f64::NEG_INFINITY;
    }
    -y.ln() - (hi / lo).ln()
}

/// Generalized-arcsine HMM on \((0,1)\) (\(\mathrm{Beta}(\alpha,1-\alpha)\)).
///
/// Shape \(\alpha\) is not identification `p`. Distinct from [`BetaHmm`]
/// (two free shapes) and [`KumaraswamyHmm`].
#[derive(Clone, Debug)]
pub struct ArcsineHmm {
    /// Hidden states. Not identification `p`.
    pub n_states: usize,
    /// Baum–Welch cap.
    pub max_iter: usize,
}

impl Default for ArcsineHmm {
    fn default() -> Self {
        Self {
            n_states: 2,
            max_iter: 40,
        }
    }
}

impl ArcsineHmm {
    /// `k`-state generalized-arcsine HMM.
    pub fn new(n_states: usize) -> Self {
        Self {
            n_states,
            ..Self::default()
        }
    }

    /// Fit alias.
    pub fn fit(&mut self, x: &Matrix, session: &Session) -> Result<Qualified<FittedArcsineHmm>> {
        self.fit_unsupervised(x, session)
    }
}

/// Fitted generalized-arcsine HMM.
#[derive(Clone, Debug)]
pub struct FittedArcsineHmm {
    /// Viterbi path.
    pub labels: Vector,
    /// Start distribution.
    pub start: Vector,
    /// Transitions.
    pub trans: Matrix,
    /// Shapes \(\alpha_j\in(0,1)\).
    pub alpha: Vector,
    /// Training log-likelihood.
    pub loglik: f64,
}

impl FittedArcsineHmm {
    fn log_emit_seq(&self, x: &Matrix) -> Vec<Vec<f64>> {
        let t = x.nrows();
        let ns = self.alpha.len();
        let mut out = vec![vec![f64::NEG_INFINITY; ns]; t];
        for ti in 0..t {
            let y = x.get(ti, 0);
            for j in 0..ns {
                out[ti][j] = log_arcsine(y, self.alpha[j]);
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

impl Predict for FittedArcsineHmm {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        self.decode(x, session)
    }
}

impl FitUnsupervised for ArcsineHmm {
    type Fitted = FittedArcsineHmm;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedArcsineHmm>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let t_len = x.nrows();
        let k = self.n_states.max(1);
        if t_len == 0 {
            return ctx.finish(FittedArcsineHmm {
                labels: empty_labels(0),
                start: init_start(k),
                trans: init_trans(k),
                alpha: Vector::filled(k, 0.5),
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
        if n_ok < 2 {
            ctx.push(
                Issue::builder(IssueCode::MeaninglessFit)
                    .message("ArcsineHmm needs at least two observations in (0, 1)")
                    .build(),
            );
        }
        let mut alpha = Vector::from_iter((0..k).map(|j| 0.35 + 0.25 * j as f64));
        let mut start = init_start(k);
        let mut trans = init_trans(k);
        let mut loglik = f64::NEG_INFINITY;
        let mut last_gamma: Vec<Vec<f64>> = Vec::new();
        for it in 0..self.max_iter.max(1) {
            let dummy = FittedArcsineHmm {
                labels: Vector::zeros(0),
                start: start.clone(),
                trans: trans.clone(),
                alpha: alpha.clone(),
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
                    if !y.is_finite() || y <= 0.0 || y >= 1.0 {
                        continue;
                    }
                    wsum += fb.gamma[t][j];
                    wy += fb.gamma[t][j] * y;
                }
                if wsum > 1e-12 {
                    alpha[j] = (wy / wsum).clamp(0.05, 0.95);
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
        let dummy = FittedArcsineHmm {
            labels: Vector::zeros(0),
            start: start.clone(),
            trans: trans.clone(),
            alpha: alpha.clone(),
            loglik,
        };
        let (labels, _) = viterbi_path(&start, &trans, &dummy.log_emit_seq(x));
        ctx.finish(FittedArcsineHmm {
            labels,
            start,
            trans,
            alpha,
            loglik,
        })
    }
}

/// Power-function HMM on \((0,1)\).
///
/// Shape is not identification `p`. Distinct from [`ArcsineHmm`] (two-sided
/// Beta tilt) and [`BetaHmm`].
#[derive(Clone, Debug)]
pub struct PowerHmm {
    /// Hidden states. Not identification `p`.
    pub n_states: usize,
    /// Baum–Welch cap.
    pub max_iter: usize,
}

impl Default for PowerHmm {
    fn default() -> Self {
        Self {
            n_states: 2,
            max_iter: 40,
        }
    }
}

impl PowerHmm {
    /// `k`-state power HMM.
    pub fn new(n_states: usize) -> Self {
        Self {
            n_states,
            ..Self::default()
        }
    }

    /// Fit alias.
    pub fn fit(&mut self, x: &Matrix, session: &Session) -> Result<Qualified<FittedPowerHmm>> {
        self.fit_unsupervised(x, session)
    }
}

/// Fitted power-function HMM.
#[derive(Clone, Debug)]
pub struct FittedPowerHmm {
    /// Viterbi path.
    pub labels: Vector,
    /// Start distribution.
    pub start: Vector,
    /// Transitions.
    pub trans: Matrix,
    /// Shapes \(\alpha_j\).
    pub alpha: Vector,
    /// Training log-likelihood.
    pub loglik: f64,
}

impl FittedPowerHmm {
    fn log_emit_seq(&self, x: &Matrix) -> Vec<Vec<f64>> {
        let t = x.nrows();
        let ns = self.alpha.len();
        let mut out = vec![vec![f64::NEG_INFINITY; ns]; t];
        for ti in 0..t {
            let y = x.get(ti, 0);
            for j in 0..ns {
                out[ti][j] = log_power_unit(y, self.alpha[j]);
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

impl Predict for FittedPowerHmm {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        self.decode(x, session)
    }
}

impl FitUnsupervised for PowerHmm {
    type Fitted = FittedPowerHmm;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedPowerHmm>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let t_len = x.nrows();
        let k = self.n_states.max(1);
        if t_len == 0 {
            return ctx.finish(FittedPowerHmm {
                labels: empty_labels(0),
                start: init_start(k),
                trans: init_trans(k),
                alpha: Vector::filled(k, 2.0),
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
        if n_ok < 2 {
            ctx.push(
                Issue::builder(IssueCode::MeaninglessFit)
                    .message("PowerHmm needs at least two observations in (0, 1)")
                    .build(),
            );
        }
        let mut alpha = Vector::from_iter((0..k).map(|j| 1.2 + j as f64));
        let mut start = init_start(k);
        let mut trans = init_trans(k);
        let mut loglik = f64::NEG_INFINITY;
        let mut last_gamma: Vec<Vec<f64>> = Vec::new();
        for it in 0..self.max_iter.max(1) {
            let dummy = FittedPowerHmm {
                labels: Vector::zeros(0),
                start: start.clone(),
                trans: trans.clone(),
                alpha: alpha.clone(),
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
                    if !y.is_finite() || y <= 0.0 || y >= 1.0 {
                        continue;
                    }
                    wsum += fb.gamma[t][j];
                    wy += fb.gamma[t][j] * y;
                }
                if wsum > 1e-12 {
                    let m = (wy / wsum).clamp(0.05, 0.95);
                    alpha[j] = (m / (1.0 - m)).clamp(0.2, 40.0);
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
        let dummy = FittedPowerHmm {
            labels: Vector::zeros(0),
            start: start.clone(),
            trans: trans.clone(),
            alpha: alpha.clone(),
            loglik,
        };
        let (labels, _) = viterbi_path(&start, &trans, &dummy.log_emit_seq(x));
        ctx.finish(FittedPowerHmm {
            labels,
            start,
            trans,
            alpha,
            loglik,
        })
    }
}

/// Raised-cosine HMM on a compact interval around each state mean.
///
/// Width is not identification `p`. Distinct from [`HyperbolicSecantHmm`]
/// (unbounded) and [`CardioidHmm`] (circular).
#[derive(Clone, Debug)]
pub struct RaisedCosineHmm {
    /// Hidden states. Not identification `p`.
    pub n_states: usize,
    /// Baum–Welch cap.
    pub max_iter: usize,
}

impl Default for RaisedCosineHmm {
    fn default() -> Self {
        Self {
            n_states: 2,
            max_iter: 40,
        }
    }
}

impl RaisedCosineHmm {
    /// `k`-state raised-cosine HMM.
    pub fn new(n_states: usize) -> Self {
        Self {
            n_states,
            ..Self::default()
        }
    }

    /// Fit alias.
    pub fn fit(&mut self, x: &Matrix, session: &Session) -> Result<Qualified<FittedRaisedCosineHmm>> {
        self.fit_unsupervised(x, session)
    }
}

/// Fitted raised-cosine HMM.
#[derive(Clone, Debug)]
pub struct FittedRaisedCosineHmm {
    /// Viterbi path.
    pub labels: Vector,
    /// Start distribution.
    pub start: Vector,
    /// Transitions.
    pub trans: Matrix,
    /// Locations.
    pub loc: Vector,
    /// Half-widths.
    pub scale: Vector,
    /// Training log-likelihood.
    pub loglik: f64,
}

impl FittedRaisedCosineHmm {
    fn log_emit_seq(&self, x: &Matrix) -> Vec<Vec<f64>> {
        let t = x.nrows();
        let ns = self.loc.len();
        let mut out = vec![vec![f64::NEG_INFINITY; ns]; t];
        for ti in 0..t {
            let y = x.get(ti, 0);
            for j in 0..ns {
                out[ti][j] = log_raised_cosine(y, self.loc[j], self.scale[j]);
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

impl Predict for FittedRaisedCosineHmm {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        self.decode(x, session)
    }
}

impl FitUnsupervised for RaisedCosineHmm {
    type Fitted = FittedRaisedCosineHmm;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedRaisedCosineHmm>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let t_len = x.nrows();
        let k = self.n_states.max(1);
        if t_len == 0 {
            return ctx.finish(FittedRaisedCosineHmm {
                labels: empty_labels(0),
                start: init_start(k),
                trans: init_trans(k),
                loc: Vector::zeros(k),
                scale: Vector::filled(k, 4.0),
                loglik: f64::NAN,
            });
        }
        let mut loc = Vector::from_iter((0..k).map(|j| -2.0 + 4.0 * j as f64));
        let mut scale = Vector::filled(k, 5.0);
        let mut start = init_start(k);
        let mut trans = init_trans(k);
        let mut loglik = f64::NEG_INFINITY;
        let mut last_gamma: Vec<Vec<f64>> = Vec::new();
        for it in 0..self.max_iter.max(1) {
            let dummy = FittedRaisedCosineHmm {
                labels: Vector::zeros(0),
                start: start.clone(),
                trans: trans.clone(),
                loc: loc.clone(),
                scale: scale.clone(),
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
                let mut mx = 0.0_f64;
                for t in 0..t_len {
                    let y = x.get(t, 0);
                    if !y.is_finite() {
                        continue;
                    }
                    wsum += fb.gamma[t][j];
                    wy += fb.gamma[t][j] * y;
                }
                if wsum > 1e-12 {
                    loc[j] = wy / wsum;
                    for t in 0..t_len {
                        let y = x.get(t, 0);
                        if y.is_finite() {
                            mx = mx.max((y - loc[j]).abs());
                        }
                    }
                    scale[j] = (mx * 1.05 + 0.25).max(0.5);
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
        let dummy = FittedRaisedCosineHmm {
            labels: Vector::zeros(0),
            start: start.clone(),
            trans: trans.clone(),
            loc: loc.clone(),
            scale: scale.clone(),
            loglik,
        };
        let (labels, _) = viterbi_path(&start, &trans, &dummy.log_emit_seq(x));
        ctx.finish(FittedRaisedCosineHmm {
            labels,
            start,
            trans,
            loc,
            scale,
            loglik,
        })
    }
}

/// Log-uniform HMM on a positive interval.
///
/// Endpoints are not identification `p`. Distinct from [`ParetoHmm`]
/// (power tail) and [`LogNormalHmm`] (Gaussian on the log).
#[derive(Clone, Debug)]
pub struct LogUniformHmm {
    /// Hidden states. Not identification `p`.
    pub n_states: usize,
    /// Baum–Welch cap.
    pub max_iter: usize,
}

impl Default for LogUniformHmm {
    fn default() -> Self {
        Self {
            n_states: 2,
            max_iter: 40,
        }
    }
}

impl LogUniformHmm {
    /// `k`-state log-uniform HMM.
    pub fn new(n_states: usize) -> Self {
        Self {
            n_states,
            ..Self::default()
        }
    }

    /// Fit alias.
    pub fn fit(&mut self, x: &Matrix, session: &Session) -> Result<Qualified<FittedLogUniformHmm>> {
        self.fit_unsupervised(x, session)
    }
}

/// Fitted log-uniform HMM.
#[derive(Clone, Debug)]
pub struct FittedLogUniformHmm {
    /// Viterbi path.
    pub labels: Vector,
    /// Start distribution.
    pub start: Vector,
    /// Transitions.
    pub trans: Matrix,
    /// Lower endpoints.
    pub lo: Vector,
    /// Upper endpoints.
    pub hi: Vector,
    /// Training log-likelihood.
    pub loglik: f64,
}

impl FittedLogUniformHmm {
    fn log_emit_seq(&self, x: &Matrix) -> Vec<Vec<f64>> {
        let t = x.nrows();
        let ns = self.lo.len();
        let mut out = vec![vec![f64::NEG_INFINITY; ns]; t];
        for ti in 0..t {
            let y = x.get(ti, 0);
            for j in 0..ns {
                out[ti][j] = log_log_uniform(y, self.lo[j], self.hi[j]);
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

impl Predict for FittedLogUniformHmm {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        self.decode(x, session)
    }
}

impl FitUnsupervised for LogUniformHmm {
    type Fitted = FittedLogUniformHmm;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedLogUniformHmm>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let t_len = x.nrows();
        let k = self.n_states.max(1);
        let mut ymin = f64::INFINITY;
        let mut ymax = 0.0_f64;
        for i in 0..t_len {
            let y = x.get(i, 0);
            if y.is_finite() && y > 0.0 {
                ymin = ymin.min(y);
                ymax = ymax.max(y);
            }
        }
        if t_len == 0 || !ymin.is_finite() {
            return ctx.finish(FittedLogUniformHmm {
                labels: empty_labels(t_len),
                start: init_start(k),
                trans: init_trans(k),
                lo: Vector::filled(k, 1.0),
                hi: Vector::filled(k, 2.0),
                loglik: f64::NAN,
            });
        }
        let mut n_skip = 0usize;
        for i in 0..t_len {
            if x.get(i, 0).is_finite() && x.get(i, 0) <= 0.0 {
                n_skip += 1;
            }
        }
        if n_skip > 0 {
            ctx.push(
                Issue::builder(IssueCode::NonPositiveSeries)
                    .severity(signlred::Severity::Warning)
                    .message(format!("LogUniformHmm skipped {n_skip} non-positive observations"))
                    .build(),
            );
        }
        let pad = (ymax / ymin.max(1e-8)).sqrt().max(1.05);
        let mut lo = Vector::filled(k, (ymin / pad).max(1e-6));
        let mut hi = Vector::filled(k, ymax * pad);
        let mut start = init_start(k);
        let mut trans = init_trans(k);
        let mut loglik = f64::NEG_INFINITY;
        let mut last_gamma: Vec<Vec<f64>> = Vec::new();
        for it in 0..self.max_iter.max(1) {
            let dummy = FittedLogUniformHmm {
                labels: Vector::zeros(0),
                start: start.clone(),
                trans: trans.clone(),
                lo: lo.clone(),
                hi: hi.clone(),
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
                let mut wlo = f64::INFINITY;
                let mut whi = 0.0_f64;
                let mut wsum = 0.0_f64;
                for t in 0..t_len {
                    let y = x.get(t, 0);
                    if !y.is_finite() || y <= 0.0 {
                        continue;
                    }
                    wsum += fb.gamma[t][j];
                    if fb.gamma[t][j] > 1e-8 {
                        wlo = wlo.min(y);
                        whi = whi.max(y);
                    }
                }
                if wsum > 1e-12 && wlo.is_finite() && whi > wlo {
                    lo[j] = (wlo / 1.05).max(1e-6);
                    hi[j] = whi * 1.05;
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
        let dummy = FittedLogUniformHmm {
            labels: Vector::zeros(0),
            start: start.clone(),
            trans: trans.clone(),
            lo: lo.clone(),
            hi: hi.clone(),
            loglik,
        };
        let (labels, _) = viterbi_path(&start, &trans, &dummy.log_emit_seq(x));
        ctx.finish(FittedLogUniformHmm {
            labels,
            start,
            trans,
            lo,
            hi,
            loglik,
        })
    }
}

fn wrap_pi(y: f64) -> f64 {
    let mut t = wrap_tau(y);
    if t > std::f64::consts::PI {
        t -= std::f64::consts::TAU;
    }
    t
}

fn log_triangular(y: f64, loc: f64, scale: f64) -> f64 {
    if !y.is_finite() || !loc.is_finite() || scale <= 0.0 {
        return f64::NEG_INFINITY;
    }
    let a = loc - scale;
    let b = loc + scale;
    if y < a || y > b {
        return f64::NEG_INFINITY;
    }
    let dens = if y <= loc {
        (y - a) / (scale * scale)
    } else {
        (b - y) / (scale * scale)
    };
    dens.max(1e-15).ln()
}

fn log_wigner(y: f64, loc: f64, radius: f64) -> f64 {
    if !y.is_finite() || !loc.is_finite() || radius <= 0.0 {
        return f64::NEG_INFINITY;
    }
    let z = y - loc;
    if z.abs() >= radius {
        return f64::NEG_INFINITY;
    }
    let s = (radius * radius - z * z).max(0.0).sqrt();
    (2.0 / (std::f64::consts::PI * radius * radius)).ln() + s.max(1e-15).ln()
}

fn asinh(w: f64) -> f64 {
    (w + (w * w + 1.0).sqrt()).ln()
}

fn log_johnson_su(y: f64, loc: f64, scale: f64, delta: f64) -> f64 {
    if !y.is_finite() || !loc.is_finite() || scale <= 0.0 || delta <= 0.0 {
        return f64::NEG_INFINITY;
    }
    let w = (y - loc) / scale;
    let z = delta * asinh(w);
    delta.ln() - scale.ln() - 0.5 * LN_2PI - 0.5 * (1.0 + w * w).ln() - 0.5 * z * z
}

fn log_borel(y: f64, mu: f64) -> f64 {
    if !y.is_finite() || y < 1.0 - 1e-9 || mu <= 0.0 || mu >= 1.0 {
        return f64::NEG_INFINITY;
    }
    let k = y.round().max(1.0);
    -mu * k + (k - 1.0) * (mu * k).max(1e-15).ln() - ln_fact(k as i32)
}

fn log_polya_aeppli(y: f64, lam: f64, rho: f64) -> f64 {
    if !y.is_finite() || y < -1e-9 || lam <= 0.0 || rho <= 0.0 || rho >= 1.0 {
        return f64::NEG_INFINITY;
    }
    let k = y.round().max(0.0) as i32;
    if k == 0 {
        return -lam;
    }
    let mut terms = Vec::with_capacity(k as usize);
    for j in 1..=k {
        let logc = ln_fact(k - 1) - ln_fact(j - 1) - ln_fact(k - j);
        terms.push(
            j as f64 * lam.ln() - ln_fact(j) + logc + (k - j) as f64 * rho.ln()
                + j as f64 * (1.0 - rho).ln(),
        );
    }
    -lam + logsumexp(&terms)
}

fn log_wrapped_laplace(y: f64, loc: f64, kappa: f64) -> f64 {
    if !y.is_finite() || !loc.is_finite() || kappa <= 0.0 {
        return f64::NEG_INFINITY;
    }
    let d = wrap_pi(y - loc).abs();
    let den = 2.0 * (1.0 - (-kappa * std::f64::consts::PI).exp());
    if den <= 1e-15 {
        return f64::NEG_INFINITY;
    }
    kappa.ln() - den.ln() - kappa * d
}

fn log_inv_chi2(y: f64, nu: f64, tau: f64) -> f64 {
    if !y.is_finite() || y <= 0.0 || nu <= 0.0 || tau <= 0.0 {
        return f64::NEG_INFINITY;
    }
    let a = 0.5 * nu;
    a * (nu * tau / 2.0).ln() - crate::special::ln_gamma(a) - (a + 1.0) * y.ln()
        - (nu * tau) / (2.0 * y)
}

/// Symmetric triangular HMM on a compact interval.
///
/// Width is not identification `p`. Distinct from [`RaisedCosineHmm`]
/// (cosine bowl) and [`WignerHmm`] (semicircle).
#[derive(Clone, Debug)]
pub struct TriangularHmm {
    /// Hidden states. Not identification `p`.
    pub n_states: usize,
    /// Baum–Welch cap.
    pub max_iter: usize,
}

impl Default for TriangularHmm {
    fn default() -> Self {
        Self {
            n_states: 2,
            max_iter: 40,
        }
    }
}

impl TriangularHmm {
    /// `k`-state triangular HMM.
    pub fn new(n_states: usize) -> Self {
        Self {
            n_states,
            ..Self::default()
        }
    }

    /// Fit alias.
    pub fn fit(&mut self, x: &Matrix, session: &Session) -> Result<Qualified<FittedTriangularHmm>> {
        self.fit_unsupervised(x, session)
    }
}

/// Fitted triangular HMM.
#[derive(Clone, Debug)]
pub struct FittedTriangularHmm {
    /// Viterbi path.
    pub labels: Vector,
    /// Start distribution.
    pub start: Vector,
    /// Transitions.
    pub trans: Matrix,
    /// Modes.
    pub loc: Vector,
    /// Half-widths.
    pub scale: Vector,
    /// Training log-likelihood.
    pub loglik: f64,
}

impl FittedTriangularHmm {
    fn log_emit_seq(&self, x: &Matrix) -> Vec<Vec<f64>> {
        let t = x.nrows();
        let ns = self.loc.len();
        let mut out = vec![vec![f64::NEG_INFINITY; ns]; t];
        for ti in 0..t {
            let y = x.get(ti, 0);
            for j in 0..ns {
                out[ti][j] = log_triangular(y, self.loc[j], self.scale[j]);
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

impl Predict for FittedTriangularHmm {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        self.decode(x, session)
    }
}

impl FitUnsupervised for TriangularHmm {
    type Fitted = FittedTriangularHmm;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedTriangularHmm>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let t_len = x.nrows();
        let k = self.n_states.max(1);
        if t_len == 0 {
            return ctx.finish(FittedTriangularHmm {
                labels: empty_labels(0),
                start: init_start(k),
                trans: init_trans(k),
                loc: Vector::zeros(k),
                scale: Vector::filled(k, 4.0),
                loglik: f64::NAN,
            });
        }
        let mut loc = Vector::from_iter((0..k).map(|j| -2.0 + 4.0 * j as f64));
        let mut scale = Vector::filled(k, 5.0);
        let mut start = init_start(k);
        let mut trans = init_trans(k);
        let mut loglik = f64::NEG_INFINITY;
        let mut last_gamma: Vec<Vec<f64>> = Vec::new();
        for it in 0..self.max_iter.max(1) {
            let dummy = FittedTriangularHmm {
                labels: Vector::zeros(0),
                start: start.clone(),
                trans: trans.clone(),
                loc: loc.clone(),
                scale: scale.clone(),
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
                let mut mx = 0.0_f64;
                for t in 0..t_len {
                    let y = x.get(t, 0);
                    if !y.is_finite() {
                        continue;
                    }
                    wsum += fb.gamma[t][j];
                    wy += fb.gamma[t][j] * y;
                }
                if wsum > 1e-12 {
                    loc[j] = wy / wsum;
                    for t in 0..t_len {
                        let y = x.get(t, 0);
                        if y.is_finite() {
                            mx = mx.max((y - loc[j]).abs());
                        }
                    }
                    scale[j] = (mx * 1.05 + 0.25).max(0.5);
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
        let dummy = FittedTriangularHmm {
            labels: Vector::zeros(0),
            start: start.clone(),
            trans: trans.clone(),
            loc: loc.clone(),
            scale: scale.clone(),
            loglik,
        };
        let (labels, _) = viterbi_path(&start, &trans, &dummy.log_emit_seq(x));
        ctx.finish(FittedTriangularHmm {
            labels,
            start,
            trans,
            loc,
            scale,
            loglik,
        })
    }
}

/// Wigner semicircle HMM on a compact interval.
///
/// Radius is not identification `p`. Distinct from [`TriangularHmm`] (linear
/// roof) and [`RaisedCosineHmm`].
#[derive(Clone, Debug)]
pub struct WignerHmm {
    /// Hidden states. Not identification `p`.
    pub n_states: usize,
    /// Baum–Welch cap.
    pub max_iter: usize,
}

impl Default for WignerHmm {
    fn default() -> Self {
        Self {
            n_states: 2,
            max_iter: 40,
        }
    }
}

impl WignerHmm {
    /// `k`-state Wigner HMM.
    pub fn new(n_states: usize) -> Self {
        Self {
            n_states,
            ..Self::default()
        }
    }

    /// Fit alias.
    pub fn fit(&mut self, x: &Matrix, session: &Session) -> Result<Qualified<FittedWignerHmm>> {
        self.fit_unsupervised(x, session)
    }
}

/// Fitted Wigner semicircle HMM.
#[derive(Clone, Debug)]
pub struct FittedWignerHmm {
    /// Viterbi path.
    pub labels: Vector,
    /// Start distribution.
    pub start: Vector,
    /// Transitions.
    pub trans: Matrix,
    /// Centres.
    pub loc: Vector,
    /// Radii.
    pub radius: Vector,
    /// Training log-likelihood.
    pub loglik: f64,
}

impl FittedWignerHmm {
    fn log_emit_seq(&self, x: &Matrix) -> Vec<Vec<f64>> {
        let t = x.nrows();
        let ns = self.loc.len();
        let mut out = vec![vec![f64::NEG_INFINITY; ns]; t];
        for ti in 0..t {
            let y = x.get(ti, 0);
            for j in 0..ns {
                out[ti][j] = log_wigner(y, self.loc[j], self.radius[j]);
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

impl Predict for FittedWignerHmm {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        self.decode(x, session)
    }
}

impl FitUnsupervised for WignerHmm {
    type Fitted = FittedWignerHmm;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedWignerHmm>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let t_len = x.nrows();
        let k = self.n_states.max(1);
        if t_len == 0 {
            return ctx.finish(FittedWignerHmm {
                labels: empty_labels(0),
                start: init_start(k),
                trans: init_trans(k),
                loc: Vector::zeros(k),
                radius: Vector::filled(k, 4.0),
                loglik: f64::NAN,
            });
        }
        let mut loc = Vector::from_iter((0..k).map(|j| -2.0 + 4.0 * j as f64));
        let mut radius = Vector::filled(k, 5.0);
        let mut start = init_start(k);
        let mut trans = init_trans(k);
        let mut loglik = f64::NEG_INFINITY;
        let mut last_gamma: Vec<Vec<f64>> = Vec::new();
        for it in 0..self.max_iter.max(1) {
            let dummy = FittedWignerHmm {
                labels: Vector::zeros(0),
                start: start.clone(),
                trans: trans.clone(),
                loc: loc.clone(),
                radius: radius.clone(),
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
                let mut mx = 0.0_f64;
                for t in 0..t_len {
                    let y = x.get(t, 0);
                    if !y.is_finite() {
                        continue;
                    }
                    wsum += fb.gamma[t][j];
                    wy += fb.gamma[t][j] * y;
                }
                if wsum > 1e-12 {
                    loc[j] = wy / wsum;
                    for t in 0..t_len {
                        let y = x.get(t, 0);
                        if y.is_finite() {
                            mx = mx.max((y - loc[j]).abs());
                        }
                    }
                    radius[j] = (mx * 1.05 + 0.25).max(0.5);
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
        let dummy = FittedWignerHmm {
            labels: Vector::zeros(0),
            start: start.clone(),
            trans: trans.clone(),
            loc: loc.clone(),
            radius: radius.clone(),
            loglik,
        };
        let (labels, _) = viterbi_path(&start, &trans, &dummy.log_emit_seq(x));
        ctx.finish(FittedWignerHmm {
            labels,
            start,
            trans,
            loc,
            radius,
            loglik,
        })
    }
}

/// Johnson \(S_U\) HMM (sinh-arcsinh Gaussian) on the real line.
///
/// Shape \(\delta\) is not identification `p`. Distinct from [`StudentTHmm`]
/// (polynomial tails) and [`SlashHmm`].
#[derive(Clone, Debug)]
pub struct JohnsonSuHmm {
    /// Hidden states. Not identification `p`.
    pub n_states: usize,
    /// Tail shape. Not identification `p`.
    pub delta: f64,
    /// Baum–Welch cap.
    pub max_iter: usize,
}

impl Default for JohnsonSuHmm {
    fn default() -> Self {
        Self {
            n_states: 2,
            delta: 1.0,
            max_iter: 40,
        }
    }
}

impl JohnsonSuHmm {
    /// `k`-state Johnson \(S_U\) HMM.
    pub fn new(n_states: usize) -> Self {
        Self {
            n_states,
            ..Self::default()
        }
    }

    /// Fit alias.
    pub fn fit(&mut self, x: &Matrix, session: &Session) -> Result<Qualified<FittedJohnsonSuHmm>> {
        self.fit_unsupervised(x, session)
    }
}

/// Fitted Johnson \(S_U\) HMM.
#[derive(Clone, Debug)]
pub struct FittedJohnsonSuHmm {
    /// Viterbi path.
    pub labels: Vector,
    /// Start distribution.
    pub start: Vector,
    /// Transitions.
    pub trans: Matrix,
    /// Locations.
    pub loc: Vector,
    /// Scales.
    pub scale: Vector,
    /// Tail shapes.
    pub delta: Vector,
    /// Training log-likelihood.
    pub loglik: f64,
}

impl FittedJohnsonSuHmm {
    fn log_emit_seq(&self, x: &Matrix) -> Vec<Vec<f64>> {
        let t = x.nrows();
        let ns = self.loc.len();
        let mut out = vec![vec![f64::NEG_INFINITY; ns]; t];
        for ti in 0..t {
            let y = x.get(ti, 0);
            for j in 0..ns {
                out[ti][j] = log_johnson_su(y, self.loc[j], self.scale[j], self.delta[j]);
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

impl Predict for FittedJohnsonSuHmm {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        self.decode(x, session)
    }
}

impl FitUnsupervised for JohnsonSuHmm {
    type Fitted = FittedJohnsonSuHmm;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedJohnsonSuHmm>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let t_len = x.nrows();
        let k = self.n_states.max(1);
        let d0 = if self.delta.is_finite() && self.delta > 0.0 {
            self.delta.clamp(0.2, 8.0)
        } else {
            1.0
        };
        if t_len == 0 {
            return ctx.finish(FittedJohnsonSuHmm {
                labels: empty_labels(0),
                start: init_start(k),
                trans: init_trans(k),
                loc: Vector::zeros(k),
                scale: Vector::filled(k, 1.0),
                delta: Vector::filled(k, d0),
                loglik: f64::NAN,
            });
        }
        let mut loc = Vector::from_iter((0..k).map(|j| -1.0 + 2.0 * j as f64));
        let mut scale = Vector::filled(k, 1.0);
        let mut delta = Vector::filled(k, d0);
        let mut start = init_start(k);
        let mut trans = init_trans(k);
        let mut loglik = f64::NEG_INFINITY;
        let mut last_gamma: Vec<Vec<f64>> = Vec::new();
        for it in 0..self.max_iter.max(1) {
            let dummy = FittedJohnsonSuHmm {
                labels: Vector::zeros(0),
                start: start.clone(),
                trans: trans.clone(),
                loc: loc.clone(),
                scale: scale.clone(),
                delta: delta.clone(),
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
                    loc[j] = wy / wsum;
                    let mut w2 = 0.0_f64;
                    for t in 0..t_len {
                        let y = x.get(t, 0);
                        if y.is_finite() {
                            let e = y - loc[j];
                            w2 += fb.gamma[t][j] * e * e;
                        }
                    }
                    scale[j] = (w2 / wsum).max(COV_FLOOR).sqrt();
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
        let dummy = FittedJohnsonSuHmm {
            labels: Vector::zeros(0),
            start: start.clone(),
            trans: trans.clone(),
            loc: loc.clone(),
            scale: scale.clone(),
            delta: delta.clone(),
            loglik,
        };
        let (labels, _) = viterbi_path(&start, &trans, &dummy.log_emit_seq(x));
        ctx.finish(FittedJohnsonSuHmm {
            labels,
            start,
            trans,
            loc,
            scale,
            delta,
            loglik,
        })
    }
}

/// Borel HMM on \(\{1,2,\ldots\}\).
///
/// Mean parameter is not identification `p`. Distinct from [`PoissonHmm`]
/// (includes zero) and [`LogarithmicHmm`].
#[derive(Clone, Debug)]
pub struct BorelHmm {
    /// Hidden states. Not identification `p`.
    pub n_states: usize,
    /// Baum–Welch cap.
    pub max_iter: usize,
}

impl Default for BorelHmm {
    fn default() -> Self {
        Self {
            n_states: 2,
            max_iter: 40,
        }
    }
}

impl BorelHmm {
    /// `k`-state Borel HMM.
    pub fn new(n_states: usize) -> Self {
        Self {
            n_states,
            ..Self::default()
        }
    }

    /// Fit alias.
    pub fn fit(&mut self, x: &Matrix, session: &Session) -> Result<Qualified<FittedBorelHmm>> {
        self.fit_unsupervised(x, session)
    }
}

/// Fitted Borel HMM.
#[derive(Clone, Debug)]
pub struct FittedBorelHmm {
    /// Viterbi path.
    pub labels: Vector,
    /// Start distribution.
    pub start: Vector,
    /// Transitions.
    pub trans: Matrix,
    /// Parameters \(\mu_j\in(0,1)\).
    pub mu: Vector,
    /// Training log-likelihood.
    pub loglik: f64,
}

impl FittedBorelHmm {
    fn log_emit_seq(&self, x: &Matrix) -> Vec<Vec<f64>> {
        let t = x.nrows();
        let ns = self.mu.len();
        let mut out = vec![vec![f64::NEG_INFINITY; ns]; t];
        for ti in 0..t {
            let y = x.get(ti, 0);
            for j in 0..ns {
                out[ti][j] = log_borel(y, self.mu[j]);
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

impl Predict for FittedBorelHmm {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        self.decode(x, session)
    }
}

impl FitUnsupervised for BorelHmm {
    type Fitted = FittedBorelHmm;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedBorelHmm>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let t_len = x.nrows();
        let k = self.n_states.max(1);
        if t_len == 0 {
            return ctx.finish(FittedBorelHmm {
                labels: empty_labels(0),
                start: init_start(k),
                trans: init_trans(k),
                mu: Vector::filled(k, 0.4),
                loglik: f64::NAN,
            });
        }
        let mut mu = Vector::from_iter((0..k).map(|j| 0.3 + 0.2 * j as f64));
        let mut start = init_start(k);
        let mut trans = init_trans(k);
        let mut loglik = f64::NEG_INFINITY;
        let mut last_gamma: Vec<Vec<f64>> = Vec::new();
        for it in 0..self.max_iter.max(1) {
            let dummy = FittedBorelHmm {
                labels: Vector::zeros(0),
                start: start.clone(),
                trans: trans.clone(),
                mu: mu.clone(),
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
                    if !y.is_finite() || y < 1.0 - 1e-9 {
                        continue;
                    }
                    wsum += fb.gamma[t][j];
                    wy += fb.gamma[t][j] * y;
                }
                if wsum > 1e-12 {
                    let m = (wy / wsum).max(1.05);
                    mu[j] = (1.0 - 1.0 / m).clamp(0.05, 0.95);
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
        let dummy = FittedBorelHmm {
            labels: Vector::zeros(0),
            start: start.clone(),
            trans: trans.clone(),
            mu: mu.clone(),
            loglik,
        };
        let (labels, _) = viterbi_path(&start, &trans, &dummy.log_emit_seq(x));
        ctx.finish(FittedBorelHmm {
            labels,
            start,
            trans,
            mu,
            loglik,
        })
    }
}

/// Pólya–Aeppli HMM (Poisson-stopped geometric).
///
/// Cluster probability is not identification `p`. Distinct from
/// [`NeymanTypeAHmm`] (Poisson offspring) and [`HermiteHmm`].
#[derive(Clone, Debug)]
pub struct PolyaAeppliHmm {
    /// Hidden states. Not identification `p`.
    pub n_states: usize,
    /// Baum–Welch cap.
    pub max_iter: usize,
}

impl Default for PolyaAeppliHmm {
    fn default() -> Self {
        Self {
            n_states: 2,
            max_iter: 40,
        }
    }
}

impl PolyaAeppliHmm {
    /// `k`-state Pólya–Aeppli HMM.
    pub fn new(n_states: usize) -> Self {
        Self {
            n_states,
            ..Self::default()
        }
    }

    /// Fit alias.
    pub fn fit(&mut self, x: &Matrix, session: &Session) -> Result<Qualified<FittedPolyaAeppliHmm>> {
        self.fit_unsupervised(x, session)
    }
}

/// Fitted Pólya–Aeppli HMM.
#[derive(Clone, Debug)]
pub struct FittedPolyaAeppliHmm {
    /// Viterbi path.
    pub labels: Vector,
    /// Start distribution.
    pub start: Vector,
    /// Transitions.
    pub trans: Matrix,
    /// Cluster Poisson rates.
    pub lam: Vector,
    /// Geometric continuation probabilities.
    pub rho: Vector,
    /// Training log-likelihood.
    pub loglik: f64,
}

impl FittedPolyaAeppliHmm {
    fn log_emit_seq(&self, x: &Matrix) -> Vec<Vec<f64>> {
        let t = x.nrows();
        let ns = self.lam.len();
        let mut out = vec![vec![f64::NEG_INFINITY; ns]; t];
        for ti in 0..t {
            let y = x.get(ti, 0);
            for j in 0..ns {
                out[ti][j] = log_polya_aeppli(y, self.lam[j], self.rho[j]);
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

impl Predict for FittedPolyaAeppliHmm {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        self.decode(x, session)
    }
}

impl FitUnsupervised for PolyaAeppliHmm {
    type Fitted = FittedPolyaAeppliHmm;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedPolyaAeppliHmm>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let t_len = x.nrows();
        let k = self.n_states.max(1);
        if t_len == 0 {
            return ctx.finish(FittedPolyaAeppliHmm {
                labels: empty_labels(0),
                start: init_start(k),
                trans: init_trans(k),
                lam: Vector::filled(k, 1.0),
                rho: Vector::filled(k, 0.3),
                loglik: f64::NAN,
            });
        }
        let mut lam = Vector::from_iter((0..k).map(|j| 0.8 + 0.6 * j as f64));
        let mut rho = Vector::filled(k, 0.3);
        let mut start = init_start(k);
        let mut trans = init_trans(k);
        let mut loglik = f64::NEG_INFINITY;
        let mut last_gamma: Vec<Vec<f64>> = Vec::new();
        for it in 0..self.max_iter.max(1) {
            let dummy = FittedPolyaAeppliHmm {
                labels: Vector::zeros(0),
                start: start.clone(),
                trans: trans.clone(),
                lam: lam.clone(),
                rho: rho.clone(),
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
                    if !y.is_finite() || y < -1e-9 {
                        continue;
                    }
                    wsum += fb.gamma[t][j];
                    wy += fb.gamma[t][j] * y;
                    wy2 += fb.gamma[t][j] * y * y;
                }
                if wsum > 1e-12 {
                    let m = (wy / wsum).max(0.2);
                    let v = (wy2 / wsum - m * m).max(m);
                    let r = ((v / m - 1.0) / (v / m + 1.0)).clamp(0.05, 0.9);
                    rho[j] = r;
                    lam[j] = (m * (1.0 - r)).max(0.05);
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
        let dummy = FittedPolyaAeppliHmm {
            labels: Vector::zeros(0),
            start: start.clone(),
            trans: trans.clone(),
            lam: lam.clone(),
            rho: rho.clone(),
            loglik,
        };
        let (labels, _) = viterbi_path(&start, &trans, &dummy.log_emit_seq(x));
        ctx.finish(FittedPolyaAeppliHmm {
            labels,
            start,
            trans,
            lam,
            rho,
            loglik,
        })
    }
}

/// Wrapped-Laplace HMM on the circle.
///
/// Concentration is not identification `p`. Distinct from [`CircularHmm`]
/// (von Mises) and [`WrappedExponentialHmm`] (one-sided).
#[derive(Clone, Debug)]
pub struct WrappedLaplaceHmm {
    /// Hidden states. Not identification `p`.
    pub n_states: usize,
    /// Baum–Welch cap.
    pub max_iter: usize,
}

impl Default for WrappedLaplaceHmm {
    fn default() -> Self {
        Self {
            n_states: 2,
            max_iter: 40,
        }
    }
}

impl WrappedLaplaceHmm {
    /// `k`-state wrapped-Laplace HMM.
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
    ) -> Result<Qualified<FittedWrappedLaplaceHmm>> {
        self.fit_unsupervised(x, session)
    }
}

/// Fitted wrapped-Laplace HMM.
#[derive(Clone, Debug)]
pub struct FittedWrappedLaplaceHmm {
    /// Viterbi path.
    pub labels: Vector,
    /// Start distribution.
    pub start: Vector,
    /// Transitions.
    pub trans: Matrix,
    /// Circular locations.
    pub loc: Vector,
    /// Concentrations.
    pub kappa: Vector,
    /// Training log-likelihood.
    pub loglik: f64,
}

impl FittedWrappedLaplaceHmm {
    fn log_emit_seq(&self, x: &Matrix) -> Vec<Vec<f64>> {
        let t = x.nrows();
        let ns = self.loc.len();
        let mut out = vec![vec![f64::NEG_INFINITY; ns]; t];
        for ti in 0..t {
            let y = x.get(ti, 0);
            for j in 0..ns {
                out[ti][j] = log_wrapped_laplace(y, self.loc[j], self.kappa[j]);
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

impl Predict for FittedWrappedLaplaceHmm {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        self.decode(x, session)
    }
}

impl FitUnsupervised for WrappedLaplaceHmm {
    type Fitted = FittedWrappedLaplaceHmm;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedWrappedLaplaceHmm>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let t_len = x.nrows();
        let k = self.n_states.max(1);
        if t_len == 0 {
            return ctx.finish(FittedWrappedLaplaceHmm {
                labels: empty_labels(0),
                start: init_start(k),
                trans: init_trans(k),
                loc: Vector::zeros(k),
                kappa: Vector::filled(k, 1.0),
                loglik: f64::NAN,
            });
        }
        let mut loc = Vector::from_iter((0..k).map(|j| -1.0 + 2.0 * j as f64));
        let mut kappa = Vector::filled(k, 1.0);
        let mut start = init_start(k);
        let mut trans = init_trans(k);
        let mut loglik = f64::NEG_INFINITY;
        let mut last_gamma: Vec<Vec<f64>> = Vec::new();
        for it in 0..self.max_iter.max(1) {
            let dummy = FittedWrappedLaplaceHmm {
                labels: Vector::zeros(0),
                start: start.clone(),
                trans: trans.clone(),
                loc: loc.clone(),
                kappa: kappa.clone(),
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
                let mut cs = 0.0_f64;
                let mut sn = 0.0_f64;
                for t in 0..t_len {
                    let y = x.get(t, 0);
                    if !y.is_finite() {
                        continue;
                    }
                    wsum += fb.gamma[t][j];
                    cs += fb.gamma[t][j] * wrap_tau(y).cos();
                    sn += fb.gamma[t][j] * wrap_tau(y).sin();
                }
                if wsum > 1e-12 {
                    loc[j] = sn.atan2(cs);
                    let mut wad = 0.0_f64;
                    for t in 0..t_len {
                        let y = x.get(t, 0);
                        if y.is_finite() {
                            wad += fb.gamma[t][j] * wrap_pi(y - loc[j]).abs();
                        }
                    }
                    kappa[j] = (1.0 / (wad / wsum).max(0.05)).clamp(0.1, 20.0);
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
        let dummy = FittedWrappedLaplaceHmm {
            labels: Vector::zeros(0),
            start: start.clone(),
            trans: trans.clone(),
            loc: loc.clone(),
            kappa: kappa.clone(),
            loglik,
        };
        let (labels, _) = viterbi_path(&start, &trans, &dummy.log_emit_seq(x));
        ctx.finish(FittedWrappedLaplaceHmm {
            labels,
            start,
            trans,
            loc,
            kappa,
            loglik,
        })
    }
}

/// Inverse-χ² HMM on the positive line.
///
/// Degrees of freedom are not identification `p`. Distinct from
/// [`InverseGammaHmm`] (two free shapes) and [`ChiHmm`] (not inverted).
#[derive(Clone, Debug)]
pub struct InverseChiSquaredHmm {
    /// Hidden states. Not identification `p`.
    pub n_states: usize,
    /// Degrees of freedom. Not identification `p`.
    pub nu: f64,
    /// Baum–Welch cap.
    pub max_iter: usize,
}

impl Default for InverseChiSquaredHmm {
    fn default() -> Self {
        Self {
            n_states: 2,
            nu: 4.0,
            max_iter: 40,
        }
    }
}

impl InverseChiSquaredHmm {
    /// `k`-state inverse-χ² HMM.
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
    ) -> Result<Qualified<FittedInverseChiSquaredHmm>> {
        self.fit_unsupervised(x, session)
    }
}

/// Fitted inverse-χ² HMM.
#[derive(Clone, Debug)]
pub struct FittedInverseChiSquaredHmm {
    /// Viterbi path.
    pub labels: Vector,
    /// Start distribution.
    pub start: Vector,
    /// Transitions.
    pub trans: Matrix,
    /// Degrees of freedom.
    pub nu: Vector,
    /// Scales \(\tau_j\).
    pub tau: Vector,
    /// Training log-likelihood.
    pub loglik: f64,
}

impl FittedInverseChiSquaredHmm {
    fn log_emit_seq(&self, x: &Matrix) -> Vec<Vec<f64>> {
        let t = x.nrows();
        let ns = self.tau.len();
        let mut out = vec![vec![f64::NEG_INFINITY; ns]; t];
        for ti in 0..t {
            let y = x.get(ti, 0);
            for j in 0..ns {
                out[ti][j] = log_inv_chi2(y, self.nu[j], self.tau[j]);
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

impl Predict for FittedInverseChiSquaredHmm {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        self.decode(x, session)
    }
}

impl FitUnsupervised for InverseChiSquaredHmm {
    type Fitted = FittedInverseChiSquaredHmm;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedInverseChiSquaredHmm>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let t_len = x.nrows();
        let k = self.n_states.max(1);
        let nu0 = if self.nu.is_finite() && self.nu > 2.0 {
            self.nu.clamp(2.2, 40.0)
        } else {
            4.0
        };
        if t_len == 0 {
            return ctx.finish(FittedInverseChiSquaredHmm {
                labels: empty_labels(0),
                start: init_start(k),
                trans: init_trans(k),
                nu: Vector::filled(k, nu0),
                tau: Vector::filled(k, 1.0),
                loglik: f64::NAN,
            });
        }
        let mut n_skip = 0usize;
        for i in 0..t_len {
            if x.get(i, 0).is_finite() && x.get(i, 0) <= 0.0 {
                n_skip += 1;
            }
        }
        if n_skip > 0 {
            ctx.push(
                Issue::builder(IssueCode::NonPositiveSeries)
                    .severity(signlred::Severity::Warning)
                    .message(format!(
                        "InverseChiSquaredHmm skipped {n_skip} non-positive observations"
                    ))
                    .build(),
            );
        }
        let mut nu = Vector::filled(k, nu0);
        let mut tau = Vector::from_iter((0..k).map(|j| 1.0 + 0.4 * j as f64));
        let mut start = init_start(k);
        let mut trans = init_trans(k);
        let mut loglik = f64::NEG_INFINITY;
        let mut last_gamma: Vec<Vec<f64>> = Vec::new();
        for it in 0..self.max_iter.max(1) {
            let dummy = FittedInverseChiSquaredHmm {
                labels: Vector::zeros(0),
                start: start.clone(),
                trans: trans.clone(),
                nu: nu.clone(),
                tau: tau.clone(),
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
                    if !y.is_finite() || y <= 0.0 {
                        continue;
                    }
                    wsum += fb.gamma[t][j];
                    wy += fb.gamma[t][j] * y;
                }
                if wsum > 1e-12 {
                    let m = (wy / wsum).max(1e-4);
                    tau[j] = (m * (nu[j] - 2.0) / nu[j]).max(1e-4);
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
        let dummy = FittedInverseChiSquaredHmm {
            labels: Vector::zeros(0),
            start: start.clone(),
            trans: trans.clone(),
            nu: nu.clone(),
            tau: tau.clone(),
            loglik,
        };
        let (labels, _) = viterbi_path(&start, &trans, &dummy.log_emit_seq(x));
        ctx.finish(FittedInverseChiSquaredHmm {
            labels,
            start,
            trans,
            nu,
            tau,
            loglik,
        })
    }
}

fn log_k1(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if x < 1e-8 {
        return -x.max(1e-16).ln();
    }
    if x < 0.75 {
        let half = 0.5 * x;
        let ln = x.ln() + 0.5772156649015329;
        return (1.0 / x + half * (ln - 0.5)).max(1e-300).ln();
    }
    let asy = 0.5 * (std::f64::consts::PI / (2.0 * x)).ln() - x;
    let t = 1.0 / x;
    asy + (1.0 + 0.375 * t - 0.1171875 * t * t + 0.1025390625 * t * t * t)
        .max(1e-12)
        .ln()
}

fn log_sichel(y: f64, mu: f64, phi: f64) -> f64 {
    if !y.is_finite() || y < -1e-9 || mu <= 0.0 || phi <= 0.0 {
        return f64::NEG_INFINITY;
    }
    let zs = [0.35, 0.7, 1.0, 1.4, 2.2];
    let mut logw = [0.0_f64; 5];
    let mut terms = [0.0_f64; 5];
    for i in 0..5 {
        let z = (mu * zs[i]).max(1e-6);
        logw[i] = 0.5 * (phi / (2.0 * std::f64::consts::PI * z * z * z)).ln()
            - phi * (z - mu) * (z - mu) / (2.0 * mu * mu * z);
        terms[i] = logw[i] + log_poisson(y, z);
    }
    let lw = logsumexp(&logw);
    for t in terms.iter_mut() {
        *t -= lw;
    }
    logsumexp(&terms)
}

const GH5_X: [f64; 5] = [
    -2.0201828704560856,
    -0.9585724646138185,
    0.0,
    0.9585724646138185,
    2.0201828704560856,
];
const GH5_W: [f64; 5] = [
    0.019953242059045913,
    0.39361932315224113,
    0.9453087204829419,
    0.39361932315224113,
    0.019953242059045913,
];

fn log_poisson_lognormal(y: f64, mu: f64, sigma: f64) -> f64 {
    if !y.is_finite() || y < -1e-9 || !mu.is_finite() || sigma <= 0.0 {
        return f64::NEG_INFINITY;
    }
    let k = y.round().max(0.0);
    let ln_sqrt_pi = 0.5 * std::f64::consts::PI.ln();
    let mut terms = [0.0_f64; 5];
    for i in 0..5 {
        let lam = (mu + sigma * std::f64::consts::SQRT_2 * GH5_X[i]).exp().max(1e-8);
        terms[i] = GH5_W[i].ln() - ln_sqrt_pi + log_poisson(k, lam);
    }
    logsumexp(&terms)
}

fn log_nig(y: f64, loc: f64, delta: f64, alpha: f64) -> f64 {
    if !y.is_finite() || !loc.is_finite() || delta <= 0.0 || alpha <= 0.0 {
        return f64::NEG_INFINITY;
    }
    let q = (delta * delta + (y - loc) * (y - loc)).sqrt().max(1e-12);
    (alpha * delta / std::f64::consts::PI).ln() + delta * alpha + log_k1(alpha * q) - q.ln()
}

fn log_variance_gamma(y: f64, loc: f64, scale: f64) -> f64 {
    if !y.is_finite() || !loc.is_finite() || scale <= 0.0 {
        return f64::NEG_INFINITY;
    }
    let z = (y - loc).abs().max(1e-8);
    let g = 1.0 / scale;
    let lam = 1.5_f64;
    2.0 * lam * g.ln()
        - 0.5 * LN_2PI
        - (lam + 0.5) * scale.ln()
        - crate::special::ln_gamma(lam)
        + (lam - 0.5) * z.ln()
        + log_k1(g * z)
}

fn log_jones_pewsey(y: f64, mu: f64, kappa: f64, psi: f64) -> f64 {
    if !y.is_finite() || !mu.is_finite() || kappa <= 0.0 {
        return f64::NEG_INFINITY;
    }
    let d = wrap_pi(y - mu);
    if psi.abs() < 1e-3 {
        return kappa * d.cos() - (2.0 * std::f64::consts::PI).ln() - log_i0(kappa);
    }
    let ckp = (kappa * psi).cosh();
    let skp = (kappa * psi).sinh();
    let core = (ckp + skp * d.cos()).max(1e-15);
    let log_un = -(1.0 / psi) * core.ln();
    let nq = 32usize;
    let mut terms = [0.0_f64; 32];
    for i in 0..nq {
        let th = -std::f64::consts::PI + 2.0 * std::f64::consts::PI * (i as f64 + 0.5) / nq as f64;
        let c = (ckp + skp * th.cos()).max(1e-15);
        terms[i] = -(1.0 / psi) * c.ln();
    }
    let log_z = (2.0 * std::f64::consts::PI / nq as f64).ln() + logsumexp(&terms);
    log_un - log_z
}

fn log_waring(y: f64, rho: f64, alpha: f64) -> f64 {
    if !y.is_finite() || y < -1e-9 || rho <= 1.0 || alpha <= 0.0 {
        return f64::NEG_INFINITY;
    }
    let k = y.round().max(0.0);
    rho.ln() + crate::special::ln_gamma(alpha + k) - crate::special::ln_gamma(alpha)
        + crate::special::ln_gamma(alpha + rho + 1.0)
        - crate::special::ln_gamma(alpha + rho + k + 1.0)
}

/// Sichel / Poisson–inverse-Gaussian HMM (IG-mixed Poisson).
///
/// Shape is not identification `p`. Distinct from [`NegativeBinomialHmm`]
/// (gamma mixture) and [`DelaporteHmm`] (Poisson + gamma).
#[derive(Clone, Debug)]
pub struct SichelHmm {
    /// Hidden states. Not identification `p`.
    pub n_states: usize,
    /// Baum–Welch cap.
    pub max_iter: usize,
}

impl Default for SichelHmm {
    fn default() -> Self {
        Self {
            n_states: 2,
            max_iter: 40,
        }
    }
}

impl SichelHmm {
    /// `k`-state Sichel HMM.
    pub fn new(n_states: usize) -> Self {
        Self {
            n_states,
            ..Self::default()
        }
    }

    /// Fit alias.
    pub fn fit(&mut self, x: &Matrix, session: &Session) -> Result<Qualified<FittedSichelHmm>> {
        self.fit_unsupervised(x, session)
    }
}

/// Fitted Sichel HMM.
#[derive(Clone, Debug)]
pub struct FittedSichelHmm {
    /// Viterbi path.
    pub labels: Vector,
    /// Start distribution.
    pub start: Vector,
    /// Transitions.
    pub trans: Matrix,
    /// Poisson–IG means.
    pub mu: Vector,
    /// IG concentrations.
    pub phi: Vector,
    /// Training log-likelihood.
    pub loglik: f64,
}

impl FittedSichelHmm {
    fn log_emit_seq(&self, x: &Matrix) -> Vec<Vec<f64>> {
        let t = x.nrows();
        let ns = self.mu.len();
        let mut out = vec![vec![f64::NEG_INFINITY; ns]; t];
        for ti in 0..t {
            let y = x.get(ti, 0);
            for j in 0..ns {
                out[ti][j] = log_sichel(y, self.mu[j], self.phi[j]);
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

impl Predict for FittedSichelHmm {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        self.decode(x, session)
    }
}

impl FitUnsupervised for SichelHmm {
    type Fitted = FittedSichelHmm;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedSichelHmm>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let t_len = x.nrows();
        let k = self.n_states.max(1);
        if t_len == 0 {
            return ctx.finish(FittedSichelHmm {
                labels: empty_labels(0),
                start: init_start(k),
                trans: init_trans(k),
                mu: Vector::filled(k, 1.0),
                phi: Vector::filled(k, 2.0),
                loglik: f64::NAN,
            });
        }
        let mut mu = Vector::from_iter((0..k).map(|j| 0.8 + 1.2 * j as f64));
        let mut phi = Vector::filled(k, 2.0);
        let mut start = init_start(k);
        let mut trans = init_trans(k);
        let mut loglik = f64::NEG_INFINITY;
        let mut last_gamma: Vec<Vec<f64>> = Vec::new();
        for it in 0..self.max_iter.max(1) {
            let dummy = FittedSichelHmm {
                labels: Vector::zeros(0),
                start: start.clone(),
                trans: trans.clone(),
                mu: mu.clone(),
                phi: phi.clone(),
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
                    if !y.is_finite() || y < -1e-9 {
                        continue;
                    }
                    wsum += fb.gamma[t][j];
                    wy += fb.gamma[t][j] * y;
                }
                if wsum > 1e-12 {
                    mu[j] = (wy / wsum).max(0.2);
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
        let dummy = FittedSichelHmm {
            labels: Vector::zeros(0),
            start: start.clone(),
            trans: trans.clone(),
            mu: mu.clone(),
            phi: phi.clone(),
            loglik,
        };
        let (labels, _) = viterbi_path(&start, &trans, &dummy.log_emit_seq(x));
        ctx.finish(FittedSichelHmm {
            labels,
            start,
            trans,
            mu,
            phi,
            loglik,
        })
    }
}

/// Poisson–lognormal HMM (Gauss–Hermite mixed Poisson).
///
/// Latent Gaussian scale is not identification `p`. Distinct from
/// [`SichelHmm`] (IG mixer) and [`PoissonHmm`].
#[derive(Clone, Debug)]
pub struct PoissonLognormalHmm {
    /// Hidden states. Not identification `p`.
    pub n_states: usize,
    /// Baum–Welch cap.
    pub max_iter: usize,
}

impl Default for PoissonLognormalHmm {
    fn default() -> Self {
        Self {
            n_states: 2,
            max_iter: 40,
        }
    }
}

impl PoissonLognormalHmm {
    /// `k`-state Poisson–lognormal HMM.
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
    ) -> Result<Qualified<FittedPoissonLognormalHmm>> {
        self.fit_unsupervised(x, session)
    }
}

/// Fitted Poisson–lognormal HMM.
#[derive(Clone, Debug)]
pub struct FittedPoissonLognormalHmm {
    /// Viterbi path.
    pub labels: Vector,
    /// Start distribution.
    pub start: Vector,
    /// Transitions.
    pub trans: Matrix,
    /// Latent Gaussian means.
    pub mu: Vector,
    /// Latent Gaussian scales.
    pub sigma: Vector,
    /// Training log-likelihood.
    pub loglik: f64,
}

impl FittedPoissonLognormalHmm {
    fn log_emit_seq(&self, x: &Matrix) -> Vec<Vec<f64>> {
        let t = x.nrows();
        let ns = self.mu.len();
        let mut out = vec![vec![f64::NEG_INFINITY; ns]; t];
        for ti in 0..t {
            let y = x.get(ti, 0);
            for j in 0..ns {
                out[ti][j] = log_poisson_lognormal(y, self.mu[j], self.sigma[j]);
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

impl Predict for FittedPoissonLognormalHmm {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        self.decode(x, session)
    }
}

impl FitUnsupervised for PoissonLognormalHmm {
    type Fitted = FittedPoissonLognormalHmm;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedPoissonLognormalHmm>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let t_len = x.nrows();
        let k = self.n_states.max(1);
        if t_len == 0 {
            return ctx.finish(FittedPoissonLognormalHmm {
                labels: empty_labels(0),
                start: init_start(k),
                trans: init_trans(k),
                mu: Vector::from_iter((0..k).map(|j| -0.2 + 0.8 * j as f64)),
                sigma: Vector::filled(k, 0.4),
                loglik: f64::NAN,
            });
        }
        let mut mu = Vector::from_iter((0..k).map(|j| 0.2 + 1.0 * j as f64));
        let mut sigma = Vector::filled(k, 0.45);
        let mut start = init_start(k);
        let mut trans = init_trans(k);
        let mut loglik = f64::NEG_INFINITY;
        let mut last_gamma: Vec<Vec<f64>> = Vec::new();
        for it in 0..self.max_iter.max(1) {
            let dummy = FittedPoissonLognormalHmm {
                labels: Vector::zeros(0),
                start: start.clone(),
                trans: trans.clone(),
                mu: mu.clone(),
                sigma: sigma.clone(),
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
                    if !y.is_finite() || y < -1e-9 {
                        continue;
                    }
                    wsum += fb.gamma[t][j];
                    wy += fb.gamma[t][j] * y.max(1e-6).ln();
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
        let dummy = FittedPoissonLognormalHmm {
            labels: Vector::zeros(0),
            start: start.clone(),
            trans: trans.clone(),
            mu: mu.clone(),
            sigma: sigma.clone(),
            loglik,
        };
        let (labels, _) = viterbi_path(&start, &trans, &dummy.log_emit_seq(x));
        ctx.finish(FittedPoissonLognormalHmm {
            labels,
            start,
            trans,
            mu,
            sigma,
            loglik,
        })
    }
}

/// Symmetric normal–inverse-Gaussian HMM.
///
/// Tail heaviness `α` is not identification `p`. Distinct from
/// [`SlashHmm`] (normal/uniform) and [`HyperbolicSecantHmm`].
#[derive(Clone, Debug)]
pub struct NormalInverseGaussianHmm {
    /// Hidden states. Not identification `p`.
    pub n_states: usize,
    /// Baum–Welch cap.
    pub max_iter: usize,
}

impl Default for NormalInverseGaussianHmm {
    fn default() -> Self {
        Self {
            n_states: 2,
            max_iter: 40,
        }
    }
}

impl NormalInverseGaussianHmm {
    /// `k`-state NIG HMM.
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
    ) -> Result<Qualified<FittedNormalInverseGaussianHmm>> {
        self.fit_unsupervised(x, session)
    }
}

/// Fitted NIG HMM.
#[derive(Clone, Debug)]
pub struct FittedNormalInverseGaussianHmm {
    /// Viterbi path.
    pub labels: Vector,
    /// Start distribution.
    pub start: Vector,
    /// Transitions.
    pub trans: Matrix,
    /// Locations.
    pub loc: Vector,
    /// Scales `δ`.
    pub delta: Vector,
    /// Tail parameters `α`.
    pub alpha: Vector,
    /// Training log-likelihood.
    pub loglik: f64,
}

impl FittedNormalInverseGaussianHmm {
    fn log_emit_seq(&self, x: &Matrix) -> Vec<Vec<f64>> {
        let t = x.nrows();
        let ns = self.loc.len();
        let mut out = vec![vec![f64::NEG_INFINITY; ns]; t];
        for ti in 0..t {
            let y = x.get(ti, 0);
            for j in 0..ns {
                out[ti][j] = log_nig(y, self.loc[j], self.delta[j], self.alpha[j]);
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

impl Predict for FittedNormalInverseGaussianHmm {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        self.decode(x, session)
    }
}

impl FitUnsupervised for NormalInverseGaussianHmm {
    type Fitted = FittedNormalInverseGaussianHmm;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedNormalInverseGaussianHmm>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let t_len = x.nrows();
        let k = self.n_states.max(1);
        if t_len == 0 {
            return ctx.finish(FittedNormalInverseGaussianHmm {
                labels: empty_labels(0),
                start: init_start(k),
                trans: init_trans(k),
                loc: Vector::zeros(k),
                delta: Vector::filled(k, 1.0),
                alpha: Vector::filled(k, 1.2),
                loglik: f64::NAN,
            });
        }
        let mut loc = Vector::from_iter((0..k).map(|j| -2.0 + 4.0 * j as f64));
        let mut delta = Vector::filled(k, 1.2);
        let mut alpha = Vector::filled(k, 1.1);
        let mut start = init_start(k);
        let mut trans = init_trans(k);
        let mut loglik = f64::NEG_INFINITY;
        let mut last_gamma: Vec<Vec<f64>> = Vec::new();
        for it in 0..self.max_iter.max(1) {
            let dummy = FittedNormalInverseGaussianHmm {
                labels: Vector::zeros(0),
                start: start.clone(),
                trans: trans.clone(),
                loc: loc.clone(),
                delta: delta.clone(),
                alpha: alpha.clone(),
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
                    loc[j] = wy / wsum;
                    let mut wad = 0.0_f64;
                    for t in 0..t_len {
                        let y = x.get(t, 0);
                        if y.is_finite() {
                            wad += fb.gamma[t][j] * (y - loc[j]).abs();
                        }
                    }
                    delta[j] = (wad / wsum).max(0.2);
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
        let dummy = FittedNormalInverseGaussianHmm {
            labels: Vector::zeros(0),
            start: start.clone(),
            trans: trans.clone(),
            loc: loc.clone(),
            delta: delta.clone(),
            alpha: alpha.clone(),
            loglik,
        };
        let (labels, _) = viterbi_path(&start, &trans, &dummy.log_emit_seq(x));
        ctx.finish(FittedNormalInverseGaussianHmm {
            labels,
            start,
            trans,
            loc,
            delta,
            alpha,
            loglik,
        })
    }
}

/// Symmetric variance-gamma HMM (`λ = 3/2`, modified Bessel \(K_1\)).
///
/// Scale is not identification `p`. Distinct from [`LaplaceHmm`] (`λ = 1`
/// VG) and [`NormalInverseGaussianHmm`].
#[derive(Clone, Debug)]
pub struct VarianceGammaHmm {
    /// Hidden states. Not identification `p`.
    pub n_states: usize,
    /// Baum–Welch cap.
    pub max_iter: usize,
}

impl Default for VarianceGammaHmm {
    fn default() -> Self {
        Self {
            n_states: 2,
            max_iter: 40,
        }
    }
}

impl VarianceGammaHmm {
    /// `k`-state variance-gamma HMM.
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
    ) -> Result<Qualified<FittedVarianceGammaHmm>> {
        self.fit_unsupervised(x, session)
    }
}

/// Fitted variance-gamma HMM.
#[derive(Clone, Debug)]
pub struct FittedVarianceGammaHmm {
    /// Viterbi path.
    pub labels: Vector,
    /// Start distribution.
    pub start: Vector,
    /// Transitions.
    pub trans: Matrix,
    /// Locations.
    pub loc: Vector,
    /// Scales.
    pub scale: Vector,
    /// Training log-likelihood.
    pub loglik: f64,
}

impl FittedVarianceGammaHmm {
    fn log_emit_seq(&self, x: &Matrix) -> Vec<Vec<f64>> {
        let t = x.nrows();
        let ns = self.loc.len();
        let mut out = vec![vec![f64::NEG_INFINITY; ns]; t];
        for ti in 0..t {
            let y = x.get(ti, 0);
            for j in 0..ns {
                out[ti][j] = log_variance_gamma(y, self.loc[j], self.scale[j]);
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

impl Predict for FittedVarianceGammaHmm {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        self.decode(x, session)
    }
}

impl FitUnsupervised for VarianceGammaHmm {
    type Fitted = FittedVarianceGammaHmm;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedVarianceGammaHmm>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let t_len = x.nrows();
        let k = self.n_states.max(1);
        if t_len == 0 {
            return ctx.finish(FittedVarianceGammaHmm {
                labels: empty_labels(0),
                start: init_start(k),
                trans: init_trans(k),
                loc: Vector::zeros(k),
                scale: Vector::filled(k, 1.0),
                loglik: f64::NAN,
            });
        }
        let mut loc = Vector::from_iter((0..k).map(|j| -2.0 + 4.0 * j as f64));
        let mut scale = Vector::filled(k, 1.2);
        let mut start = init_start(k);
        let mut trans = init_trans(k);
        let mut loglik = f64::NEG_INFINITY;
        let mut last_gamma: Vec<Vec<f64>> = Vec::new();
        for it in 0..self.max_iter.max(1) {
            let dummy = FittedVarianceGammaHmm {
                labels: Vector::zeros(0),
                start: start.clone(),
                trans: trans.clone(),
                loc: loc.clone(),
                scale: scale.clone(),
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
                    loc[j] = wy / wsum;
                    let mut wad = 0.0_f64;
                    for t in 0..t_len {
                        let y = x.get(t, 0);
                        if y.is_finite() {
                            wad += fb.gamma[t][j] * (y - loc[j]).abs();
                        }
                    }
                    scale[j] = (wad / wsum).max(0.2);
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
        let dummy = FittedVarianceGammaHmm {
            labels: Vector::zeros(0),
            start: start.clone(),
            trans: trans.clone(),
            loc: loc.clone(),
            scale: scale.clone(),
            loglik,
        };
        let (labels, _) = viterbi_path(&start, &trans, &dummy.log_emit_seq(x));
        ctx.finish(FittedVarianceGammaHmm {
            labels,
            start,
            trans,
            loc,
            scale,
            loglik,
        })
    }
}

/// Jones–Pewsey circular HMM (power of a cosine-hyperbolic kernel).
///
/// Shape `ψ` is not identification `p`. Distinct from [`CircularHmm`]
/// (von Mises, `ψ → 0`) and [`WrappedCauchyHmm`] (`ψ = 1`).
#[derive(Clone, Debug)]
pub struct JonesPewseyHmm {
    /// Hidden states. Not identification `p`.
    pub n_states: usize,
    /// Jones–Pewsey shape. Not identification `p`.
    pub psi: f64,
    /// Baum–Welch cap.
    pub max_iter: usize,
}

impl Default for JonesPewseyHmm {
    fn default() -> Self {
        Self {
            n_states: 2,
            psi: 0.5,
            max_iter: 40,
        }
    }
}

impl JonesPewseyHmm {
    /// `k`-state Jones–Pewsey HMM.
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
    ) -> Result<Qualified<FittedJonesPewseyHmm>> {
        self.fit_unsupervised(x, session)
    }
}

/// Fitted Jones–Pewsey HMM.
#[derive(Clone, Debug)]
pub struct FittedJonesPewseyHmm {
    /// Viterbi path.
    pub labels: Vector,
    /// Start distribution.
    pub start: Vector,
    /// Transitions.
    pub trans: Matrix,
    /// Mean directions.
    pub mu: Vector,
    /// Concentrations.
    pub kappa: Vector,
    /// Shared shape.
    pub psi: f64,
    /// Training log-likelihood.
    pub loglik: f64,
}

impl FittedJonesPewseyHmm {
    fn log_emit_seq(&self, x: &Matrix) -> Vec<Vec<f64>> {
        let t = x.nrows();
        let ns = self.mu.len();
        let mut out = vec![vec![f64::NEG_INFINITY; ns]; t];
        for ti in 0..t {
            let y = x.get(ti, 0);
            for j in 0..ns {
                out[ti][j] = log_jones_pewsey(y, self.mu[j], self.kappa[j], self.psi);
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

impl Predict for FittedJonesPewseyHmm {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        self.decode(x, session)
    }
}

impl FitUnsupervised for JonesPewseyHmm {
    type Fitted = FittedJonesPewseyHmm;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedJonesPewseyHmm>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let t_len = x.nrows();
        let k = self.n_states.max(1);
        let psi = if self.psi.is_finite() && self.psi.abs() > 1e-4 {
            self.psi.clamp(-2.0, 2.0)
        } else {
            0.5
        };
        if t_len == 0 {
            return ctx.finish(FittedJonesPewseyHmm {
                labels: empty_labels(0),
                start: init_start(k),
                trans: init_trans(k),
                mu: Vector::zeros(k),
                kappa: Vector::filled(k, 1.0),
                psi,
                loglik: f64::NAN,
            });
        }
        let mut mu = Vector::from_iter((0..k).map(|j| -1.0 + 2.0 * j as f64));
        let mut kappa = Vector::filled(k, 1.2);
        let mut start = init_start(k);
        let mut trans = init_trans(k);
        let mut loglik = f64::NEG_INFINITY;
        let mut last_gamma: Vec<Vec<f64>> = Vec::new();
        for it in 0..self.max_iter.max(1) {
            let dummy = FittedJonesPewseyHmm {
                labels: Vector::zeros(0),
                start: start.clone(),
                trans: trans.clone(),
                mu: mu.clone(),
                kappa: kappa.clone(),
                psi,
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
                let mut sx = 0.0_f64;
                let mut sy = 0.0_f64;
                for t in 0..t_len {
                    let y = x.get(t, 0);
                    if !y.is_finite() {
                        continue;
                    }
                    wsum += fb.gamma[t][j];
                    sx += fb.gamma[t][j] * y.cos();
                    sy += fb.gamma[t][j] * y.sin();
                }
                if wsum > 1e-12 {
                    mu[j] = sy.atan2(sx);
                    let r = (sx * sx + sy * sy).sqrt() / wsum;
                    kappa[j] = (2.0 * r / (1.0 - r).max(0.05)).clamp(0.15, 8.0);
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
        let dummy = FittedJonesPewseyHmm {
            labels: Vector::zeros(0),
            start: start.clone(),
            trans: trans.clone(),
            mu: mu.clone(),
            kappa: kappa.clone(),
            psi,
            loglik,
        };
        let (labels, _) = viterbi_path(&start, &trans, &dummy.log_emit_seq(x));
        ctx.finish(FittedJonesPewseyHmm {
            labels,
            start,
            trans,
            mu,
            kappa,
            psi,
            loglik,
        })
    }
}

/// Waring HMM on \(\{0,1,\ldots\}\) (Yule with an extra shape).
///
/// Shapes are not identification `p`. Distinct from [`YuleSimonHmm`]
/// (one-parameter, support starts at 1).
#[derive(Clone, Debug)]
pub struct WaringHmm {
    /// Hidden states. Not identification `p`.
    pub n_states: usize,
    /// Baum–Welch cap.
    pub max_iter: usize,
}

impl Default for WaringHmm {
    fn default() -> Self {
        Self {
            n_states: 2,
            max_iter: 40,
        }
    }
}

impl WaringHmm {
    /// `k`-state Waring HMM.
    pub fn new(n_states: usize) -> Self {
        Self {
            n_states,
            ..Self::default()
        }
    }

    /// Fit alias.
    pub fn fit(&mut self, x: &Matrix, session: &Session) -> Result<Qualified<FittedWaringHmm>> {
        self.fit_unsupervised(x, session)
    }
}

/// Fitted Waring HMM.
#[derive(Clone, Debug)]
pub struct FittedWaringHmm {
    /// Viterbi path.
    pub labels: Vector,
    /// Start distribution.
    pub start: Vector,
    /// Transitions.
    pub trans: Matrix,
    /// Tail parameters `ρ > 1`.
    pub rho: Vector,
    /// Extra shapes `α`.
    pub alpha: Vector,
    /// Training log-likelihood.
    pub loglik: f64,
}

impl FittedWaringHmm {
    fn log_emit_seq(&self, x: &Matrix) -> Vec<Vec<f64>> {
        let t = x.nrows();
        let ns = self.rho.len();
        let mut out = vec![vec![f64::NEG_INFINITY; ns]; t];
        for ti in 0..t {
            let y = x.get(ti, 0);
            for j in 0..ns {
                out[ti][j] = log_waring(y, self.rho[j], self.alpha[j]);
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

impl Predict for FittedWaringHmm {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        self.decode(x, session)
    }
}

impl FitUnsupervised for WaringHmm {
    type Fitted = FittedWaringHmm;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedWaringHmm>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let t_len = x.nrows();
        let k = self.n_states.max(1);
        if t_len == 0 {
            return ctx.finish(FittedWaringHmm {
                labels: empty_labels(0),
                start: init_start(k),
                trans: init_trans(k),
                rho: Vector::filled(k, 2.5),
                alpha: Vector::filled(k, 1.0),
                loglik: f64::NAN,
            });
        }
        let mut rho = Vector::from_iter((0..k).map(|j| 2.2 + 0.6 * j as f64));
        let mut alpha = Vector::filled(k, 1.2);
        let mut start = init_start(k);
        let mut trans = init_trans(k);
        let mut loglik = f64::NEG_INFINITY;
        let mut last_gamma: Vec<Vec<f64>> = Vec::new();
        for it in 0..self.max_iter.max(1) {
            let dummy = FittedWaringHmm {
                labels: Vector::zeros(0),
                start: start.clone(),
                trans: trans.clone(),
                rho: rho.clone(),
                alpha: alpha.clone(),
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
                    if !y.is_finite() || y < -1e-9 {
                        continue;
                    }
                    wsum += fb.gamma[t][j];
                    wy += fb.gamma[t][j] * y;
                }
                if wsum > 1e-12 {
                    let m = (wy / wsum).max(0.2);
                    alpha[j] = ((rho[j] - 1.0) * m).max(0.2);
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
        let dummy = FittedWaringHmm {
            labels: Vector::zeros(0),
            start: start.clone(),
            trans: trans.clone(),
            rho: rho.clone(),
            alpha: alpha.clone(),
            loglik,
        };
        let (labels, _) = viterbi_path(&start, &trans, &dummy.log_emit_seq(x));
        ctx.finish(FittedWaringHmm {
            labels,
            start,
            trans,
            rho,
            alpha,
            loglik,
        })
    }
}

fn log_good(y: f64, s: f64, q: f64) -> f64 {
    if !y.is_finite() || y < 1.0 - 1e-9 || s <= 0.0 || q <= 0.0 || q >= 1.0 {
        return f64::NEG_INFINITY;
    }
    let k = y.round().max(1.0);
    let mut z = [0.0_f64; 48];
    for n in 1..=48 {
        z[n - 1] = -s * (n as f64).ln() + (n as f64) * q.ln();
    }
    -s * k.ln() + k * q.ln() - logsumexp(&z)
}

fn log_ncx2(y: f64, lam: f64) -> f64 {
    if !y.is_finite() || y <= 0.0 || lam < 0.0 {
        return f64::NEG_INFINITY;
    }
    -0.5 * y - 0.5 * lam - std::f64::consts::LN_2 + log_i0((lam * y).max(0.0).sqrt())
}

fn log_sine_skew_vm(y: f64, mu: f64, kappa: f64, skew: f64) -> f64 {
    if !y.is_finite() || !mu.is_finite() || kappa <= 0.0 {
        return f64::NEG_INFINITY;
    }
    let d = wrap_pi(y - mu);
    let sk = skew.clamp(-0.95, 0.95);
    kappa * d.cos() + (1.0 + sk * d.sin()).max(1e-12).ln()
        - (2.0 * std::f64::consts::PI).ln()
        - log_i0(kappa)
}

fn log_hyperbolic(y: f64, loc: f64, delta: f64, alpha: f64) -> f64 {
    if !y.is_finite() || !loc.is_finite() || delta <= 0.0 || alpha <= 0.0 {
        return f64::NEG_INFINITY;
    }
    let q = (delta * delta + (y - loc) * (y - loc)).sqrt().max(1e-12);
    (alpha / (2.0 * delta)).ln() - log_k1(alpha * delta) - alpha * q
}

/// Good count HMM \(P(k)\propto k^{-s}q^{k}\) on \(\{1,2,\ldots\}\).
///
/// Exponent `s` is not identification `p`. Distinct from [`ZetaHmm`] (`q = 1`)
/// and [`LogarithmicHmm`] (`s = 1`).
#[derive(Clone, Debug)]
pub struct GoodHmm {
    /// Hidden states. Not identification `p`.
    pub n_states: usize,
    /// Baum–Welch cap.
    pub max_iter: usize,
}

impl Default for GoodHmm {
    fn default() -> Self {
        Self {
            n_states: 2,
            max_iter: 40,
        }
    }
}

impl GoodHmm {
    /// `k`-state Good HMM.
    pub fn new(n_states: usize) -> Self {
        Self {
            n_states,
            ..Self::default()
        }
    }

    /// Fit alias.
    pub fn fit(&mut self, x: &Matrix, session: &Session) -> Result<Qualified<FittedGoodHmm>> {
        self.fit_unsupervised(x, session)
    }
}

/// Fitted Good HMM.
#[derive(Clone, Debug)]
pub struct FittedGoodHmm {
    /// Viterbi path.
    pub labels: Vector,
    /// Start distribution.
    pub start: Vector,
    /// Transitions.
    pub trans: Matrix,
    /// Power exponents.
    pub s: Vector,
    /// Discount factors.
    pub q: Vector,
    /// Training log-likelihood.
    pub loglik: f64,
}

impl FittedGoodHmm {
    fn log_emit_seq(&self, x: &Matrix) -> Vec<Vec<f64>> {
        let t = x.nrows();
        let ns = self.s.len();
        let mut out = vec![vec![f64::NEG_INFINITY; ns]; t];
        for ti in 0..t {
            let y = x.get(ti, 0);
            for j in 0..ns {
                out[ti][j] = log_good(y, self.s[j], self.q[j]);
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

impl Predict for FittedGoodHmm {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        self.decode(x, session)
    }
}

impl FitUnsupervised for GoodHmm {
    type Fitted = FittedGoodHmm;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedGoodHmm>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let t_len = x.nrows();
        let k = self.n_states.max(1);
        if t_len == 0 {
            return ctx.finish(FittedGoodHmm {
                labels: empty_labels(0),
                start: init_start(k),
                trans: init_trans(k),
                s: Vector::filled(k, 1.5),
                q: Vector::filled(k, 0.6),
                loglik: f64::NAN,
            });
        }
        let mut s = Vector::from_iter((0..k).map(|j| 1.2 + 0.4 * j as f64));
        let mut qv = Vector::from_iter((0..k).map(|j| 0.45 + 0.2 * j as f64));
        let mut start = init_start(k);
        let mut trans = init_trans(k);
        let mut loglik = f64::NEG_INFINITY;
        let mut last_gamma: Vec<Vec<f64>> = Vec::new();
        for it in 0..self.max_iter.max(1) {
            let dummy = FittedGoodHmm {
                labels: Vector::zeros(0),
                start: start.clone(),
                trans: trans.clone(),
                s: s.clone(),
                q: qv.clone(),
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
                    if !y.is_finite() || y < 1.0 - 1e-9 {
                        continue;
                    }
                    wsum += fb.gamma[t][j];
                    wy += fb.gamma[t][j] * y;
                }
                if wsum > 1e-12 {
                    let m = (wy / wsum).max(1.05);
                    qv[j] = (1.0 - 1.0 / m).clamp(0.15, 0.9);
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
        let dummy = FittedGoodHmm {
            labels: Vector::zeros(0),
            start: start.clone(),
            trans: trans.clone(),
            s: s.clone(),
            q: qv.clone(),
            loglik,
        };
        let (labels, _) = viterbi_path(&start, &trans, &dummy.log_emit_seq(x));
        ctx.finish(FittedGoodHmm {
            labels,
            start,
            trans,
            s,
            q: qv,
            loglik,
        })
    }
}

/// Non-central χ² HMM with 2 degrees of freedom (Bessel \(I_0\)).
///
/// Non-centrality is not identification `p`. Distinct from [`ChiHmm`]
/// (central) and [`ExponentialHmm`] (central χ²₂).
#[derive(Clone, Debug)]
pub struct NoncentralChiSquaredHmm {
    /// Hidden states. Not identification `p`.
    pub n_states: usize,
    /// Baum–Welch cap.
    pub max_iter: usize,
}

impl Default for NoncentralChiSquaredHmm {
    fn default() -> Self {
        Self {
            n_states: 2,
            max_iter: 40,
        }
    }
}

impl NoncentralChiSquaredHmm {
    /// `k`-state non-central χ² HMM.
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
    ) -> Result<Qualified<FittedNoncentralChiSquaredHmm>> {
        self.fit_unsupervised(x, session)
    }
}

/// Fitted non-central χ² HMM.
#[derive(Clone, Debug)]
pub struct FittedNoncentralChiSquaredHmm {
    /// Viterbi path.
    pub labels: Vector,
    /// Start distribution.
    pub start: Vector,
    /// Transitions.
    pub trans: Matrix,
    /// Non-centralities.
    pub lam: Vector,
    /// Training log-likelihood.
    pub loglik: f64,
}

impl FittedNoncentralChiSquaredHmm {
    fn log_emit_seq(&self, x: &Matrix) -> Vec<Vec<f64>> {
        let t = x.nrows();
        let ns = self.lam.len();
        let mut out = vec![vec![f64::NEG_INFINITY; ns]; t];
        for ti in 0..t {
            let y = x.get(ti, 0);
            for j in 0..ns {
                out[ti][j] = log_ncx2(y, self.lam[j]);
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

impl Predict for FittedNoncentralChiSquaredHmm {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        self.decode(x, session)
    }
}

impl FitUnsupervised for NoncentralChiSquaredHmm {
    type Fitted = FittedNoncentralChiSquaredHmm;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedNoncentralChiSquaredHmm>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let t_len = x.nrows();
        let k = self.n_states.max(1);
        let mut n_skip = 0usize;
        for i in 0..t_len {
            if x.get(i, 0).is_finite() && x.get(i, 0) <= 0.0 {
                n_skip += 1;
            }
        }
        if n_skip > 0 {
            ctx.push(
                Issue::builder(IssueCode::NonPositiveSeries)
                    .severity(signlred::Severity::Warning)
                    .message(format!(
                        "NoncentralChiSquaredHmm skipped {n_skip} non-positive observations"
                    ))
                    .build(),
            );
        }
        if t_len == 0 {
            return ctx.finish(FittedNoncentralChiSquaredHmm {
                labels: empty_labels(0),
                start: init_start(k),
                trans: init_trans(k),
                lam: Vector::filled(k, 1.0),
                loglik: f64::NAN,
            });
        }
        let mut lam = Vector::from_iter((0..k).map(|j| 0.8 + 1.2 * j as f64));
        let mut start = init_start(k);
        let mut trans = init_trans(k);
        let mut loglik = f64::NEG_INFINITY;
        let mut last_gamma: Vec<Vec<f64>> = Vec::new();
        for it in 0..self.max_iter.max(1) {
            let dummy = FittedNoncentralChiSquaredHmm {
                labels: Vector::zeros(0),
                start: start.clone(),
                trans: trans.clone(),
                lam: lam.clone(),
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
                    if !y.is_finite() || y <= 0.0 {
                        continue;
                    }
                    wsum += fb.gamma[t][j];
                    wy += fb.gamma[t][j] * y;
                }
                if wsum > 1e-12 {
                    lam[j] = (wy / wsum - 2.0).max(0.05);
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
        let dummy = FittedNoncentralChiSquaredHmm {
            labels: Vector::zeros(0),
            start: start.clone(),
            trans: trans.clone(),
            lam: lam.clone(),
            loglik,
        };
        let (labels, _) = viterbi_path(&start, &trans, &dummy.log_emit_seq(x));
        ctx.finish(FittedNoncentralChiSquaredHmm {
            labels,
            start,
            trans,
            lam,
            loglik,
        })
    }
}

/// Sine-skewed von Mises HMM (Abe–Pewsey).
///
/// Skew `λ` is not identification `p`. Distinct from [`CircularHmm`]
/// (`λ = 0`) and [`CardioidHmm`].
#[derive(Clone, Debug)]
pub struct SineSkewedVonMisesHmm {
    /// Hidden states. Not identification `p`.
    pub n_states: usize,
    /// Baum–Welch cap.
    pub max_iter: usize,
}

impl Default for SineSkewedVonMisesHmm {
    fn default() -> Self {
        Self {
            n_states: 2,
            max_iter: 40,
        }
    }
}

impl SineSkewedVonMisesHmm {
    /// `k`-state sine-skewed von Mises HMM.
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
    ) -> Result<Qualified<FittedSineSkewedVonMisesHmm>> {
        self.fit_unsupervised(x, session)
    }
}

/// Fitted sine-skewed von Mises HMM.
#[derive(Clone, Debug)]
pub struct FittedSineSkewedVonMisesHmm {
    /// Viterbi path.
    pub labels: Vector,
    /// Start distribution.
    pub start: Vector,
    /// Transitions.
    pub trans: Matrix,
    /// Mean directions.
    pub mu: Vector,
    /// Concentrations.
    pub kappa: Vector,
    /// Skew parameters.
    pub skew: Vector,
    /// Training log-likelihood.
    pub loglik: f64,
}

impl FittedSineSkewedVonMisesHmm {
    fn log_emit_seq(&self, x: &Matrix) -> Vec<Vec<f64>> {
        let t = x.nrows();
        let ns = self.mu.len();
        let mut out = vec![vec![f64::NEG_INFINITY; ns]; t];
        for ti in 0..t {
            let y = x.get(ti, 0);
            for j in 0..ns {
                out[ti][j] = log_sine_skew_vm(y, self.mu[j], self.kappa[j], self.skew[j]);
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

impl Predict for FittedSineSkewedVonMisesHmm {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        self.decode(x, session)
    }
}

impl FitUnsupervised for SineSkewedVonMisesHmm {
    type Fitted = FittedSineSkewedVonMisesHmm;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedSineSkewedVonMisesHmm>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let t_len = x.nrows();
        let k = self.n_states.max(1);
        if t_len == 0 {
            return ctx.finish(FittedSineSkewedVonMisesHmm {
                labels: empty_labels(0),
                start: init_start(k),
                trans: init_trans(k),
                mu: Vector::zeros(k),
                kappa: Vector::filled(k, 1.0),
                skew: Vector::from_iter((0..k).map(|j| -0.2 + 0.4 * j as f64)),
                loglik: f64::NAN,
            });
        }
        let mut mu = Vector::from_iter((0..k).map(|j| -1.0 + 2.0 * j as f64));
        let mut kappa = Vector::filled(k, 1.2);
        let mut skew = Vector::from_iter((0..k).map(|j| -0.25 + 0.5 * j as f64));
        let mut start = init_start(k);
        let mut trans = init_trans(k);
        let mut loglik = f64::NEG_INFINITY;
        let mut last_gamma: Vec<Vec<f64>> = Vec::new();
        for it in 0..self.max_iter.max(1) {
            let dummy = FittedSineSkewedVonMisesHmm {
                labels: Vector::zeros(0),
                start: start.clone(),
                trans: trans.clone(),
                mu: mu.clone(),
                kappa: kappa.clone(),
                skew: skew.clone(),
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
                let mut sx = 0.0_f64;
                let mut sy = 0.0_f64;
                for t in 0..t_len {
                    let y = x.get(t, 0);
                    if !y.is_finite() {
                        continue;
                    }
                    wsum += fb.gamma[t][j];
                    sx += fb.gamma[t][j] * y.cos();
                    sy += fb.gamma[t][j] * y.sin();
                }
                if wsum > 1e-12 {
                    mu[j] = sy.atan2(sx);
                    let r = (sx * sx + sy * sy).sqrt() / wsum;
                    kappa[j] = (2.0 * r / (1.0 - r).max(0.05)).clamp(0.15, 8.0);
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
        let dummy = FittedSineSkewedVonMisesHmm {
            labels: Vector::zeros(0),
            start: start.clone(),
            trans: trans.clone(),
            mu: mu.clone(),
            kappa: kappa.clone(),
            skew: skew.clone(),
            loglik,
        };
        let (labels, _) = viterbi_path(&start, &trans, &dummy.log_emit_seq(x));
        ctx.finish(FittedSineSkewedVonMisesHmm {
            labels,
            start,
            trans,
            mu,
            kappa,
            skew,
            loglik,
        })
    }
}

/// Symmetric hyperbolic HMM (Barndorff-Nielsen; kernel \(e^{-α q}\)).
///
/// Tail `α` is not identification `p`. Distinct from
/// [`NormalInverseGaussianHmm`] (\(K_1(αq)/q\)) and [`HyperbolicSecantHmm`].
#[derive(Clone, Debug)]
pub struct HyperbolicHmm {
    /// Hidden states. Not identification `p`.
    pub n_states: usize,
    /// Baum–Welch cap.
    pub max_iter: usize,
}

impl Default for HyperbolicHmm {
    fn default() -> Self {
        Self {
            n_states: 2,
            max_iter: 40,
        }
    }
}

impl HyperbolicHmm {
    /// `k`-state hyperbolic HMM.
    pub fn new(n_states: usize) -> Self {
        Self {
            n_states,
            ..Self::default()
        }
    }

    /// Fit alias.
    pub fn fit(&mut self, x: &Matrix, session: &Session) -> Result<Qualified<FittedHyperbolicHmm>> {
        self.fit_unsupervised(x, session)
    }
}

/// Fitted hyperbolic HMM.
#[derive(Clone, Debug)]
pub struct FittedHyperbolicHmm {
    /// Viterbi path.
    pub labels: Vector,
    /// Start distribution.
    pub start: Vector,
    /// Transitions.
    pub trans: Matrix,
    /// Locations.
    pub loc: Vector,
    /// Scales `δ`.
    pub delta: Vector,
    /// Tail parameters `α`.
    pub alpha: Vector,
    /// Training log-likelihood.
    pub loglik: f64,
}

impl FittedHyperbolicHmm {
    fn log_emit_seq(&self, x: &Matrix) -> Vec<Vec<f64>> {
        let t = x.nrows();
        let ns = self.loc.len();
        let mut out = vec![vec![f64::NEG_INFINITY; ns]; t];
        for ti in 0..t {
            let y = x.get(ti, 0);
            for j in 0..ns {
                out[ti][j] = log_hyperbolic(y, self.loc[j], self.delta[j], self.alpha[j]);
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

impl Predict for FittedHyperbolicHmm {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        self.decode(x, session)
    }
}

impl FitUnsupervised for HyperbolicHmm {
    type Fitted = FittedHyperbolicHmm;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedHyperbolicHmm>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let t_len = x.nrows();
        let k = self.n_states.max(1);
        if t_len == 0 {
            return ctx.finish(FittedHyperbolicHmm {
                labels: empty_labels(0),
                start: init_start(k),
                trans: init_trans(k),
                loc: Vector::zeros(k),
                delta: Vector::filled(k, 1.0),
                alpha: Vector::filled(k, 1.1),
                loglik: f64::NAN,
            });
        }
        let mut loc = Vector::from_iter((0..k).map(|j| -2.0 + 4.0 * j as f64));
        let mut delta = Vector::filled(k, 1.2);
        let mut alpha = Vector::filled(k, 1.05);
        let mut start = init_start(k);
        let mut trans = init_trans(k);
        let mut loglik = f64::NEG_INFINITY;
        let mut last_gamma: Vec<Vec<f64>> = Vec::new();
        for it in 0..self.max_iter.max(1) {
            let dummy = FittedHyperbolicHmm {
                labels: Vector::zeros(0),
                start: start.clone(),
                trans: trans.clone(),
                loc: loc.clone(),
                delta: delta.clone(),
                alpha: alpha.clone(),
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
                    loc[j] = wy / wsum;
                    let mut wad = 0.0_f64;
                    for t in 0..t_len {
                        let y = x.get(t, 0);
                        if y.is_finite() {
                            wad += fb.gamma[t][j] * (y - loc[j]).abs();
                        }
                    }
                    delta[j] = (wad / wsum).max(0.2);
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
        let dummy = FittedHyperbolicHmm {
            labels: Vector::zeros(0),
            start: start.clone(),
            trans: trans.clone(),
            loc: loc.clone(),
            delta: delta.clone(),
            alpha: alpha.clone(),
            loglik,
        };
        let (labels, _) = viterbi_path(&start, &trans, &dummy.log_emit_seq(x));
        ctx.finish(FittedHyperbolicHmm {
            labels,
            start,
            trans,
            loc,
            delta,
            alpha,
            loglik,
        })
    }
}

fn log_gpd(y: f64, loc: f64, scale: f64, xi: f64) -> f64 {
    if !y.is_finite() || !loc.is_finite() || scale <= 0.0 {
        return f64::NEG_INFINITY;
    }
    let z = (y - loc) / scale;
    if z < 0.0 {
        return f64::NEG_INFINITY;
    }
    if xi.abs() < 1e-8 {
        return -scale.ln() - z;
    }
    let t = 1.0 + xi * z;
    if t <= 1e-15 {
        return f64::NEG_INFINITY;
    }
    -scale.ln() - (1.0 / xi + 1.0) * t.ln()
}

fn tukey_q(p: f64, lam: f64) -> f64 {
    if lam.abs() < 1e-8 {
        p.max(1e-15).ln() - (1.0 - p).max(1e-15).ln()
    } else {
        (p.max(1e-15).powf(lam) - (1.0 - p).max(1e-15).powf(lam)) / lam
    }
}

fn log_tukey_lambda(y: f64, loc: f64, scale: f64, lam: f64) -> f64 {
    if !y.is_finite() || !loc.is_finite() || scale <= 0.0 {
        return f64::NEG_INFINITY;
    }
    let z = (y - loc) / scale;
    let mut lo = 1e-12_f64;
    let mut hi = 1.0 - 1e-12;
    for _ in 0..48 {
        let mid = 0.5 * (lo + hi);
        if tukey_q(mid, lam) < z {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let p = 0.5 * (lo + hi);
    let qp = p.max(1e-15).powf(lam - 1.0) + (1.0 - p).max(1e-15).powf(lam - 1.0);
    -scale.ln() - qp.max(1e-15).ln()
}

fn log_exp_weibull(y: f64, scale: f64, c: f64, alpha: f64) -> f64 {
    if !y.is_finite() || y <= 0.0 || scale <= 0.0 || c <= 0.0 || alpha <= 0.0 {
        return f64::NEG_INFINITY;
    }
    let u = y / scale;
    let z = u.powf(c);
    if !z.is_finite() || z > 60.0 {
        return f64::NEG_INFINITY;
    }
    let surv = (-z).exp();
    let cdf1 = 1.0 - surv;
    if cdf1 <= 1e-15 {
        return f64::NEG_INFINITY;
    }
    alpha.ln() + c.ln() - scale.ln() + (c - 1.0) * u.max(1e-15).ln() - z
        + (alpha - 1.0) * cdf1.ln()
}

fn log_binom_coef(n: i32, k: i32) -> f64 {
    if k < 0 || k > n {
        return f64::NEG_INFINITY;
    }
    ln_fact(n) - ln_fact(k) - ln_fact(n - k)
}

fn log_poisson_binom(y: f64, n1: i32, p1: f64, n2: i32, p2: f64) -> f64 {
    if !y.is_finite()
        || y < -1e-9
        || n1 < 0
        || n2 < 0
        || p1 <= 0.0
        || p1 >= 1.0
        || p2 <= 0.0
        || p2 >= 1.0
    {
        return f64::NEG_INFINITY;
    }
    let k = y.round() as i32;
    if k < 0 || k > n1 + n2 {
        return f64::NEG_INFINITY;
    }
    let jmin = 0.max(k - n2);
    let jmax = k.min(n1);
    let mut terms = Vec::with_capacity((jmax - jmin + 1) as usize);
    for j in jmin..=jmax {
        let i = k - j;
        terms.push(
            log_binom_coef(n1, j)
                + j as f64 * p1.ln()
                + (n1 - j) as f64 * (1.0 - p1).ln()
                + log_binom_coef(n2, i)
                + i as f64 * p2.ln()
                + (n2 - i) as f64 * (1.0 - p2).ln(),
        );
    }
    logsumexp(&terms)
}

fn log_projected_normal(y: f64, mu: f64, tau: f64) -> f64 {
    if !y.is_finite() || !mu.is_finite() || tau < 0.0 {
        return f64::NEG_INFINITY;
    }
    let c = tau * wrap_pi(y - mu).cos();
    let extra = 1.0
        + c * (2.0 * std::f64::consts::PI).sqrt()
            * (0.5 * c * c).exp()
            * crate::special::norm_cdf(c);
    -0.5 * tau * tau - (2.0 * std::f64::consts::PI).ln() + extra.max(1e-15).ln()
}

/// Generalized Pareto HMM (peaks-over-threshold).
///
/// Shape `ξ` is not identification `p`. Distinct from [`GevHmm`] (block
/// maxima) and [`ExponentialHmm`] (`ξ = 0`). Location is pinned below the
/// sample minimum so the support covers every observation.
#[derive(Clone, Debug)]
pub struct GeneralizedParetoHmm {
    /// Hidden states. Not identification `p`.
    pub n_states: usize,
    /// GPD shape. Not identification `p`.
    pub xi: f64,
    /// Baum–Welch cap.
    pub max_iter: usize,
}

impl Default for GeneralizedParetoHmm {
    fn default() -> Self {
        Self {
            n_states: 2,
            xi: 0.15,
            max_iter: 40,
        }
    }
}

impl GeneralizedParetoHmm {
    /// `k`-state GPD HMM.
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
    ) -> Result<Qualified<FittedGeneralizedParetoHmm>> {
        self.fit_unsupervised(x, session)
    }
}

/// Fitted generalized Pareto HMM.
#[derive(Clone, Debug)]
pub struct FittedGeneralizedParetoHmm {
    /// Viterbi path.
    pub labels: Vector,
    /// Start distribution.
    pub start: Vector,
    /// Transitions.
    pub trans: Matrix,
    /// Thresholds (below the sample min).
    pub loc: Vector,
    /// Scales.
    pub scale: Vector,
    /// Shared shape.
    pub xi: f64,
    /// Training log-likelihood.
    pub loglik: f64,
}

impl FittedGeneralizedParetoHmm {
    fn log_emit_seq(&self, x: &Matrix) -> Vec<Vec<f64>> {
        let t = x.nrows();
        let ns = self.loc.len();
        let mut out = vec![vec![f64::NEG_INFINITY; ns]; t];
        for ti in 0..t {
            let y = x.get(ti, 0);
            for j in 0..ns {
                out[ti][j] = log_gpd(y, self.loc[j], self.scale[j], self.xi);
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

impl Predict for FittedGeneralizedParetoHmm {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        self.decode(x, session)
    }
}

impl FitUnsupervised for GeneralizedParetoHmm {
    type Fitted = FittedGeneralizedParetoHmm;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedGeneralizedParetoHmm>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let t_len = x.nrows();
        let k = self.n_states.max(1);
        let xi = if self.xi.is_finite() {
            self.xi.clamp(-0.4, 0.8)
        } else {
            0.15
        };
        let mut ymin = f64::INFINITY;
        for i in 0..t_len {
            let y = x.get(i, 0);
            if y.is_finite() {
                ymin = ymin.min(y);
            }
        }
        let loc0 = if ymin.is_finite() { ymin - 0.25 } else { 0.0 };
        if t_len == 0 {
            return ctx.finish(FittedGeneralizedParetoHmm {
                labels: empty_labels(0),
                start: init_start(k),
                trans: init_trans(k),
                loc: Vector::filled(k, loc0),
                scale: Vector::filled(k, 1.0),
                xi,
                loglik: f64::NAN,
            });
        }
        let loc = Vector::filled(k, loc0);
        let mut scale = Vector::from_iter((0..k).map(|j| 1.0 + 0.4 * j as f64));
        let mut start = init_start(k);
        let mut trans = init_trans(k);
        let mut loglik = f64::NEG_INFINITY;
        let mut last_gamma: Vec<Vec<f64>> = Vec::new();
        for it in 0..self.max_iter.max(1) {
            let dummy = FittedGeneralizedParetoHmm {
                labels: Vector::zeros(0),
                start: start.clone(),
                trans: trans.clone(),
                loc: loc.clone(),
                scale: scale.clone(),
                xi,
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
                    wy += fb.gamma[t][j] * (y - loc[j]).max(0.0);
                }
                if wsum > 1e-12 {
                    scale[j] = (wy / wsum).max(0.2);
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
        let dummy = FittedGeneralizedParetoHmm {
            labels: Vector::zeros(0),
            start: start.clone(),
            trans: trans.clone(),
            loc: loc.clone(),
            scale: scale.clone(),
            xi,
            loglik,
        };
        let (labels, _) = viterbi_path(&start, &trans, &dummy.log_emit_seq(x));
        ctx.finish(FittedGeneralizedParetoHmm {
            labels,
            start,
            trans,
            loc,
            scale,
            xi,
            loglik,
        })
    }
}

/// Tukey-λ HMM (quantile-defined density via bisection).
///
/// Shape `λ ≤ 0` keeps unbounded support. Distinct from [`LogisticHmm`]
/// (`λ = 0`) and [`GaussianHmm`]. `λ` is not identification `p`.
#[derive(Clone, Debug)]
pub struct TukeyLambdaHmm {
    /// Hidden states. Not identification `p`.
    pub n_states: usize,
    /// Tukey shape. Not identification `p`.
    pub lambda: f64,
    /// Baum–Welch cap.
    pub max_iter: usize,
}

impl Default for TukeyLambdaHmm {
    fn default() -> Self {
        Self {
            n_states: 2,
            lambda: -0.5,
            max_iter: 40,
        }
    }
}

impl TukeyLambdaHmm {
    /// `k`-state Tukey-λ HMM.
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
    ) -> Result<Qualified<FittedTukeyLambdaHmm>> {
        self.fit_unsupervised(x, session)
    }
}

/// Fitted Tukey-λ HMM.
#[derive(Clone, Debug)]
pub struct FittedTukeyLambdaHmm {
    /// Viterbi path.
    pub labels: Vector,
    /// Start distribution.
    pub start: Vector,
    /// Transitions.
    pub trans: Matrix,
    /// Locations.
    pub loc: Vector,
    /// Scales.
    pub scale: Vector,
    /// Shared shape.
    pub lambda: f64,
    /// Training log-likelihood.
    pub loglik: f64,
}

impl FittedTukeyLambdaHmm {
    fn log_emit_seq(&self, x: &Matrix) -> Vec<Vec<f64>> {
        let t = x.nrows();
        let ns = self.loc.len();
        let mut out = vec![vec![f64::NEG_INFINITY; ns]; t];
        for ti in 0..t {
            let y = x.get(ti, 0);
            for j in 0..ns {
                out[ti][j] = log_tukey_lambda(y, self.loc[j], self.scale[j], self.lambda);
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

impl Predict for FittedTukeyLambdaHmm {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        self.decode(x, session)
    }
}

impl FitUnsupervised for TukeyLambdaHmm {
    type Fitted = FittedTukeyLambdaHmm;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedTukeyLambdaHmm>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let t_len = x.nrows();
        let k = self.n_states.max(1);
        let lam = if self.lambda.is_finite() && self.lambda <= 0.0 {
            self.lambda.clamp(-1.5, 0.0)
        } else {
            -0.5
        };
        if t_len == 0 {
            return ctx.finish(FittedTukeyLambdaHmm {
                labels: empty_labels(0),
                start: init_start(k),
                trans: init_trans(k),
                loc: Vector::zeros(k),
                scale: Vector::filled(k, 1.0),
                lambda: lam,
                loglik: f64::NAN,
            });
        }
        let mut loc = Vector::from_iter((0..k).map(|j| -2.0 + 4.0 * j as f64));
        let mut scale = Vector::filled(k, 1.2);
        let mut start = init_start(k);
        let mut trans = init_trans(k);
        let mut loglik = f64::NEG_INFINITY;
        let mut last_gamma: Vec<Vec<f64>> = Vec::new();
        for it in 0..self.max_iter.max(1) {
            let dummy = FittedTukeyLambdaHmm {
                labels: Vector::zeros(0),
                start: start.clone(),
                trans: trans.clone(),
                loc: loc.clone(),
                scale: scale.clone(),
                lambda: lam,
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
                    loc[j] = wy / wsum;
                    let mut wad = 0.0_f64;
                    for t in 0..t_len {
                        let y = x.get(t, 0);
                        if y.is_finite() {
                            wad += fb.gamma[t][j] * (y - loc[j]).abs();
                        }
                    }
                    scale[j] = (wad / wsum).max(0.2);
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
        let dummy = FittedTukeyLambdaHmm {
            labels: Vector::zeros(0),
            start: start.clone(),
            trans: trans.clone(),
            loc: loc.clone(),
            scale: scale.clone(),
            lambda: lam,
            loglik,
        };
        let (labels, _) = viterbi_path(&start, &trans, &dummy.log_emit_seq(x));
        ctx.finish(FittedTukeyLambdaHmm {
            labels,
            start,
            trans,
            loc,
            scale,
            lambda: lam,
            loglik,
        })
    }
}

/// Exponentiated-Weibull HMM (Mudholkar–Srivastava).
///
/// Extra power `α` is not identification `p`. Distinct from [`WeibullHmm`]
/// (`α = 1`) and [`FrechetHmm`].
#[derive(Clone, Debug)]
pub struct ExpWeibullHmm {
    /// Hidden states. Not identification `p`.
    pub n_states: usize,
    /// Baum–Welch cap.
    pub max_iter: usize,
}

impl Default for ExpWeibullHmm {
    fn default() -> Self {
        Self {
            n_states: 2,
            max_iter: 40,
        }
    }
}

impl ExpWeibullHmm {
    /// `k`-state exponentiated-Weibull HMM.
    pub fn new(n_states: usize) -> Self {
        Self {
            n_states,
            ..Self::default()
        }
    }

    /// Fit alias.
    pub fn fit(&mut self, x: &Matrix, session: &Session) -> Result<Qualified<FittedExpWeibullHmm>> {
        self.fit_unsupervised(x, session)
    }
}

/// Fitted exponentiated-Weibull HMM.
#[derive(Clone, Debug)]
pub struct FittedExpWeibullHmm {
    /// Viterbi path.
    pub labels: Vector,
    /// Start distribution.
    pub start: Vector,
    /// Transitions.
    pub trans: Matrix,
    /// Scales.
    pub scale: Vector,
    /// Weibull exponents.
    pub c: Vector,
    /// Extra powers.
    pub alpha: Vector,
    /// Training log-likelihood.
    pub loglik: f64,
}

impl FittedExpWeibullHmm {
    fn log_emit_seq(&self, x: &Matrix) -> Vec<Vec<f64>> {
        let t = x.nrows();
        let ns = self.scale.len();
        let mut out = vec![vec![f64::NEG_INFINITY; ns]; t];
        for ti in 0..t {
            let y = x.get(ti, 0);
            for j in 0..ns {
                out[ti][j] = log_exp_weibull(y, self.scale[j], self.c[j], self.alpha[j]);
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

impl Predict for FittedExpWeibullHmm {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        self.decode(x, session)
    }
}

impl FitUnsupervised for ExpWeibullHmm {
    type Fitted = FittedExpWeibullHmm;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedExpWeibullHmm>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let t_len = x.nrows();
        let k = self.n_states.max(1);
        let mut n_skip = 0usize;
        for i in 0..t_len {
            if x.get(i, 0).is_finite() && x.get(i, 0) <= 0.0 {
                n_skip += 1;
            }
        }
        if n_skip > 0 {
            ctx.push(
                Issue::builder(IssueCode::NonPositiveSeries)
                    .severity(signlred::Severity::Warning)
                    .message(format!(
                        "ExpWeibullHmm skipped {n_skip} non-positive observations"
                    ))
                    .build(),
            );
        }
        if t_len == 0 {
            return ctx.finish(FittedExpWeibullHmm {
                labels: empty_labels(0),
                start: init_start(k),
                trans: init_trans(k),
                scale: Vector::filled(k, 1.0),
                c: Vector::filled(k, 1.4),
                alpha: Vector::filled(k, 1.2),
                loglik: f64::NAN,
            });
        }
        let mut scale = Vector::from_iter((0..k).map(|j| 1.0 + 0.6 * j as f64));
        let mut c = Vector::filled(k, 1.4);
        let mut alpha = Vector::from_iter((0..k).map(|j| 1.0 + 0.3 * j as f64));
        let mut start = init_start(k);
        let mut trans = init_trans(k);
        let mut loglik = f64::NEG_INFINITY;
        let mut last_gamma: Vec<Vec<f64>> = Vec::new();
        for it in 0..self.max_iter.max(1) {
            let dummy = FittedExpWeibullHmm {
                labels: Vector::zeros(0),
                start: start.clone(),
                trans: trans.clone(),
                scale: scale.clone(),
                c: c.clone(),
                alpha: alpha.clone(),
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
                    if !y.is_finite() || y <= 0.0 {
                        continue;
                    }
                    wsum += fb.gamma[t][j];
                    wy += fb.gamma[t][j] * y;
                }
                if wsum > 1e-12 {
                    scale[j] = (wy / wsum).max(0.2);
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
        let dummy = FittedExpWeibullHmm {
            labels: Vector::zeros(0),
            start: start.clone(),
            trans: trans.clone(),
            scale: scale.clone(),
            c: c.clone(),
            alpha: alpha.clone(),
            loglik,
        };
        let (labels, _) = viterbi_path(&start, &trans, &dummy.log_emit_seq(x));
        ctx.finish(FittedExpWeibullHmm {
            labels,
            start,
            trans,
            scale,
            c,
            alpha,
            loglik,
        })
    }
}

/// Poisson-binomial HMM (two heterogeneous Bernoulli blocks).
///
/// Trial counts are not identification `p`. Distinct from [`BinomialHmm`]
/// (one success probability).
#[derive(Clone, Debug)]
pub struct PoissonBinomialHmm {
    /// Hidden states. Not identification `p`.
    pub n_states: usize,
    /// Trials in the first block. Not identification `p`.
    pub n1: i32,
    /// Trials in the second block. Not identification `p`.
    pub n2: i32,
    /// Baum–Welch cap.
    pub max_iter: usize,
}

impl Default for PoissonBinomialHmm {
    fn default() -> Self {
        Self {
            n_states: 2,
            n1: 8,
            n2: 8,
            max_iter: 40,
        }
    }
}

impl PoissonBinomialHmm {
    /// `k`-state Poisson-binomial HMM.
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
    ) -> Result<Qualified<FittedPoissonBinomialHmm>> {
        self.fit_unsupervised(x, session)
    }
}

/// Fitted Poisson-binomial HMM.
#[derive(Clone, Debug)]
pub struct FittedPoissonBinomialHmm {
    /// Viterbi path.
    pub labels: Vector,
    /// Start distribution.
    pub start: Vector,
    /// Transitions.
    pub trans: Matrix,
    /// First-block success probabilities.
    pub p1: Vector,
    /// Second-block success probabilities.
    pub p2: Vector,
    /// First-block trial count.
    pub n1: i32,
    /// Second-block trial count.
    pub n2: i32,
    /// Training log-likelihood.
    pub loglik: f64,
}

impl FittedPoissonBinomialHmm {
    fn log_emit_seq(&self, x: &Matrix) -> Vec<Vec<f64>> {
        let t = x.nrows();
        let ns = self.p1.len();
        let mut out = vec![vec![f64::NEG_INFINITY; ns]; t];
        for ti in 0..t {
            let y = x.get(ti, 0);
            for j in 0..ns {
                out[ti][j] = log_poisson_binom(y, self.n1, self.p1[j], self.n2, self.p2[j]);
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

impl Predict for FittedPoissonBinomialHmm {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        self.decode(x, session)
    }
}

impl FitUnsupervised for PoissonBinomialHmm {
    type Fitted = FittedPoissonBinomialHmm;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedPoissonBinomialHmm>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let t_len = x.nrows();
        let k = self.n_states.max(1);
        let n1 = self.n1.max(1);
        let n2 = self.n2.max(1);
        let ntot = (n1 + n2) as f64;
        if t_len == 0 {
            return ctx.finish(FittedPoissonBinomialHmm {
                labels: empty_labels(0),
                start: init_start(k),
                trans: init_trans(k),
                p1: Vector::filled(k, 0.3),
                p2: Vector::filled(k, 0.6),
                n1,
                n2,
                loglik: f64::NAN,
            });
        }
        let mut p1 = Vector::from_iter((0..k).map(|j| 0.25 + 0.2 * j as f64));
        let mut p2 = Vector::from_iter((0..k).map(|j| 0.45 + 0.2 * j as f64));
        let mut start = init_start(k);
        let mut trans = init_trans(k);
        let mut loglik = f64::NEG_INFINITY;
        let mut last_gamma: Vec<Vec<f64>> = Vec::new();
        for it in 0..self.max_iter.max(1) {
            let dummy = FittedPoissonBinomialHmm {
                labels: Vector::zeros(0),
                start: start.clone(),
                trans: trans.clone(),
                p1: p1.clone(),
                p2: p2.clone(),
                n1,
                n2,
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
                    if !y.is_finite() || y < -1e-9 {
                        continue;
                    }
                    wsum += fb.gamma[t][j];
                    wy += fb.gamma[t][j] * y;
                }
                if wsum > 1e-12 {
                    let p = (wy / (wsum * ntot)).clamp(0.05, 0.95);
                    p1[j] = (p * 0.7).clamp(0.05, 0.95);
                    p2[j] = (p * 1.3).clamp(0.05, 0.95);
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
        let dummy = FittedPoissonBinomialHmm {
            labels: Vector::zeros(0),
            start: start.clone(),
            trans: trans.clone(),
            p1: p1.clone(),
            p2: p2.clone(),
            n1,
            n2,
            loglik,
        };
        let (labels, _) = viterbi_path(&start, &trans, &dummy.log_emit_seq(x));
        ctx.finish(FittedPoissonBinomialHmm {
            labels,
            start,
            trans,
            p1,
            p2,
            n1,
            n2,
            loglik,
        })
    }
}

/// Projected-normal circular HMM (angular Gaussian).
///
/// Concentration `τ` is not identification `p`. Distinct from [`CircularHmm`]
/// (von Mises) and [`WrappedNormalHmm`].
#[derive(Clone, Debug)]
pub struct ProjectedNormalHmm {
    /// Hidden states. Not identification `p`.
    pub n_states: usize,
    /// Baum–Welch cap.
    pub max_iter: usize,
}

impl Default for ProjectedNormalHmm {
    fn default() -> Self {
        Self {
            n_states: 2,
            max_iter: 40,
        }
    }
}

impl ProjectedNormalHmm {
    /// `k`-state projected-normal HMM.
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
    ) -> Result<Qualified<FittedProjectedNormalHmm>> {
        self.fit_unsupervised(x, session)
    }
}

/// Fitted projected-normal HMM.
#[derive(Clone, Debug)]
pub struct FittedProjectedNormalHmm {
    /// Viterbi path.
    pub labels: Vector,
    /// Start distribution.
    pub start: Vector,
    /// Transitions.
    pub trans: Matrix,
    /// Mean directions.
    pub mu: Vector,
    /// Concentrations.
    pub tau: Vector,
    /// Training log-likelihood.
    pub loglik: f64,
}

impl FittedProjectedNormalHmm {
    fn log_emit_seq(&self, x: &Matrix) -> Vec<Vec<f64>> {
        let t = x.nrows();
        let ns = self.mu.len();
        let mut out = vec![vec![f64::NEG_INFINITY; ns]; t];
        for ti in 0..t {
            let y = x.get(ti, 0);
            for j in 0..ns {
                out[ti][j] = log_projected_normal(y, self.mu[j], self.tau[j]);
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

impl Predict for FittedProjectedNormalHmm {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        self.decode(x, session)
    }
}

impl FitUnsupervised for ProjectedNormalHmm {
    type Fitted = FittedProjectedNormalHmm;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedProjectedNormalHmm>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let t_len = x.nrows();
        let k = self.n_states.max(1);
        if t_len == 0 {
            return ctx.finish(FittedProjectedNormalHmm {
                labels: empty_labels(0),
                start: init_start(k),
                trans: init_trans(k),
                mu: Vector::zeros(k),
                tau: Vector::filled(k, 1.0),
                loglik: f64::NAN,
            });
        }
        let mut mu = Vector::from_iter((0..k).map(|j| -1.0 + 2.0 * j as f64));
        let mut tau = Vector::filled(k, 1.1);
        let mut start = init_start(k);
        let mut trans = init_trans(k);
        let mut loglik = f64::NEG_INFINITY;
        let mut last_gamma: Vec<Vec<f64>> = Vec::new();
        for it in 0..self.max_iter.max(1) {
            let dummy = FittedProjectedNormalHmm {
                labels: Vector::zeros(0),
                start: start.clone(),
                trans: trans.clone(),
                mu: mu.clone(),
                tau: tau.clone(),
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
                let mut sx = 0.0_f64;
                let mut sy = 0.0_f64;
                for t in 0..t_len {
                    let y = x.get(t, 0);
                    if !y.is_finite() {
                        continue;
                    }
                    wsum += fb.gamma[t][j];
                    sx += fb.gamma[t][j] * y.cos();
                    sy += fb.gamma[t][j] * y.sin();
                }
                if wsum > 1e-12 {
                    mu[j] = sy.atan2(sx);
                    let r = (sx * sx + sy * sy).sqrt() / wsum;
                    tau[j] = (2.0 * r / (1.0 - r).max(0.05)).clamp(0.15, 6.0);
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
        let dummy = FittedProjectedNormalHmm {
            labels: Vector::zeros(0),
            start: start.clone(),
            trans: trans.clone(),
            mu: mu.clone(),
            tau: tau.clone(),
            loglik,
        };
        let (labels, _) = viterbi_path(&start, &trans, &dummy.log_emit_seq(x));
        ctx.finish(FittedProjectedNormalHmm {
            labels,
            start,
            trans,
            mu,
            tau,
            loglik,
        })
    }
}

fn log_kato_jones(y: f64, mu: f64, rho: f64, kappa: f64) -> f64 {
    if !y.is_finite() || !mu.is_finite() || kappa < 0.0 {
        return f64::NEG_INFINITY;
    }
    let rho = rho.clamp(0.0, 0.95);
    let c = wrap_pi(y - mu).cos();
    let den = 1.0 + rho * rho - 2.0 * rho * c;
    if den <= 1e-15 {
        return f64::NEG_INFINITY;
    }
    (1.0 - rho * rho).max(1e-15).ln() - (2.0 * std::f64::consts::PI).ln() - den.ln()
        + kappa * (c - rho) / den
        - log_i0(kappa)
}

fn log_nct(y: f64, loc: f64, scale: f64, ncp: f64) -> f64 {
    if !y.is_finite() || !loc.is_finite() || scale <= 0.0 || !ncp.is_finite() {
        return f64::NEG_INFINITY;
    }
    let z = (y - loc) / scale;
    let ws = [0.25, 0.55, 0.9, 1.4, 2.2];
    let mut logw = [0.0_f64; 5];
    let mut terms = [0.0_f64; 5];
    for i in 0..5 {
        let w: f64 = ws[i];
        logw[i] = 2.0 * 2.0_f64.ln() + w.ln() - 2.0 * w;
        let s = w.sqrt();
        let resid = z * s - ncp;
        terms[i] = logw[i] - 0.5 * LN_2PI - 0.5 * resid * resid + s.ln() - scale.ln();
    }
    let lw = logsumexp(&logw);
    for t in terms.iter_mut() {
        *t -= lw;
    }
    logsumexp(&terms)
}

fn log_hypoexp(y: f64, l1: f64, l2: f64) -> f64 {
    if !y.is_finite() || y <= 0.0 || l1 <= 0.0 || l2 <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if (l1 - l2).abs() < 1e-6 {
        return 2.0 * l1.ln() + y.ln() - l1 * y;
    }
    let (lo, hi) = if l1 < l2 { (l1, l2) } else { (l2, l1) };
    let log_diff = (-lo * y) + (1.0 - (-(hi - lo) * y).exp()).max(1e-15).ln();
    l1.ln() + l2.ln() - (hi - lo).ln() + log_diff
}

/// Kato–Jones circular HMM (Möbius × von Mises).
///
/// Recovers [`WrappedCauchyHmm`] at `κ = 0` and [`CircularHmm`] at `ρ = 0`.
/// `ρ` and `κ` are not identification `p`.
#[derive(Clone, Debug)]
pub struct KatoJonesHmm {
    /// Hidden states. Not identification `p`.
    pub n_states: usize,
    /// Baum–Welch cap.
    pub max_iter: usize,
}

impl Default for KatoJonesHmm {
    fn default() -> Self {
        Self {
            n_states: 2,
            max_iter: 40,
        }
    }
}

impl KatoJonesHmm {
    /// `k`-state Kato–Jones HMM.
    pub fn new(n_states: usize) -> Self {
        Self {
            n_states,
            ..Self::default()
        }
    }

    /// Fit alias.
    pub fn fit(&mut self, x: &Matrix, session: &Session) -> Result<Qualified<FittedKatoJonesHmm>> {
        self.fit_unsupervised(x, session)
    }
}

/// Fitted Kato–Jones HMM.
#[derive(Clone, Debug)]
pub struct FittedKatoJonesHmm {
    /// Viterbi path.
    pub labels: Vector,
    /// Start distribution.
    pub start: Vector,
    /// Transitions.
    pub trans: Matrix,
    /// Mean directions.
    pub mu: Vector,
    /// Möbius radii.
    pub rho: Vector,
    /// Von Mises concentrations.
    pub kappa: Vector,
    /// Training log-likelihood.
    pub loglik: f64,
}

impl FittedKatoJonesHmm {
    fn log_emit_seq(&self, x: &Matrix) -> Vec<Vec<f64>> {
        let t = x.nrows();
        let ns = self.mu.len();
        let mut out = vec![vec![f64::NEG_INFINITY; ns]; t];
        for ti in 0..t {
            let y = x.get(ti, 0);
            for j in 0..ns {
                out[ti][j] = log_kato_jones(y, self.mu[j], self.rho[j], self.kappa[j]);
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

impl Predict for FittedKatoJonesHmm {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        self.decode(x, session)
    }
}

impl FitUnsupervised for KatoJonesHmm {
    type Fitted = FittedKatoJonesHmm;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedKatoJonesHmm>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let t_len = x.nrows();
        let k = self.n_states.max(1);
        if t_len == 0 {
            return ctx.finish(FittedKatoJonesHmm {
                labels: empty_labels(0),
                start: init_start(k),
                trans: init_trans(k),
                mu: Vector::zeros(k),
                rho: Vector::filled(k, 0.3),
                kappa: Vector::filled(k, 0.8),
                loglik: f64::NAN,
            });
        }
        let mut mu = Vector::from_iter((0..k).map(|j| -1.0 + 2.0 * j as f64));
        let mut rho = Vector::from_iter((0..k).map(|j| 0.2 + 0.2 * j as f64));
        let mut kappa = Vector::filled(k, 0.9);
        let mut start = init_start(k);
        let mut trans = init_trans(k);
        let mut loglik = f64::NEG_INFINITY;
        let mut last_gamma: Vec<Vec<f64>> = Vec::new();
        for it in 0..self.max_iter.max(1) {
            let dummy = FittedKatoJonesHmm {
                labels: Vector::zeros(0),
                start: start.clone(),
                trans: trans.clone(),
                mu: mu.clone(),
                rho: rho.clone(),
                kappa: kappa.clone(),
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
                let mut sx = 0.0_f64;
                let mut sy = 0.0_f64;
                for t in 0..t_len {
                    let y = x.get(t, 0);
                    if !y.is_finite() {
                        continue;
                    }
                    wsum += fb.gamma[t][j];
                    sx += fb.gamma[t][j] * y.cos();
                    sy += fb.gamma[t][j] * y.sin();
                }
                if wsum > 1e-12 {
                    mu[j] = sy.atan2(sx);
                    let r = (sx * sx + sy * sy).sqrt() / wsum;
                    rho[j] = (0.5 * r).clamp(0.05, 0.9);
                    kappa[j] = (2.0 * r / (1.0 - r).max(0.05)).clamp(0.1, 6.0);
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
        let dummy = FittedKatoJonesHmm {
            labels: Vector::zeros(0),
            start: start.clone(),
            trans: trans.clone(),
            mu: mu.clone(),
            rho: rho.clone(),
            kappa: kappa.clone(),
            loglik,
        };
        let (labels, _) = viterbi_path(&start, &trans, &dummy.log_emit_seq(x));
        ctx.finish(FittedKatoJonesHmm {
            labels,
            start,
            trans,
            mu,
            rho,
            kappa,
            loglik,
        })
    }
}

/// Non-central \(t\) HMM (χ²₄ scale mixture of a shifted Gaussian).
///
/// Non-centrality is not identification `p`. Distinct from [`StudentTHmm`]
/// (central) and [`NoncentralChiSquaredHmm`].
#[derive(Clone, Debug)]
pub struct NoncentralTHmm {
    /// Hidden states. Not identification `p`.
    pub n_states: usize,
    /// Baum–Welch cap.
    pub max_iter: usize,
}

impl Default for NoncentralTHmm {
    fn default() -> Self {
        Self {
            n_states: 2,
            max_iter: 40,
        }
    }
}

impl NoncentralTHmm {
    /// `k`-state non-central \(t\) HMM.
    pub fn new(n_states: usize) -> Self {
        Self {
            n_states,
            ..Self::default()
        }
    }

    /// Fit alias.
    pub fn fit(&mut self, x: &Matrix, session: &Session) -> Result<Qualified<FittedNoncentralTHmm>> {
        self.fit_unsupervised(x, session)
    }
}

/// Fitted non-central \(t\) HMM.
#[derive(Clone, Debug)]
pub struct FittedNoncentralTHmm {
    /// Viterbi path.
    pub labels: Vector,
    /// Start distribution.
    pub start: Vector,
    /// Transitions.
    pub trans: Matrix,
    /// Locations.
    pub loc: Vector,
    /// Scales.
    pub scale: Vector,
    /// Non-centralities.
    pub ncp: Vector,
    /// Training log-likelihood.
    pub loglik: f64,
}

impl FittedNoncentralTHmm {
    fn log_emit_seq(&self, x: &Matrix) -> Vec<Vec<f64>> {
        let t = x.nrows();
        let ns = self.loc.len();
        let mut out = vec![vec![f64::NEG_INFINITY; ns]; t];
        for ti in 0..t {
            let y = x.get(ti, 0);
            for j in 0..ns {
                out[ti][j] = log_nct(y, self.loc[j], self.scale[j], self.ncp[j]);
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

impl Predict for FittedNoncentralTHmm {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        self.decode(x, session)
    }
}

impl FitUnsupervised for NoncentralTHmm {
    type Fitted = FittedNoncentralTHmm;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedNoncentralTHmm>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let t_len = x.nrows();
        let k = self.n_states.max(1);
        if t_len == 0 {
            return ctx.finish(FittedNoncentralTHmm {
                labels: empty_labels(0),
                start: init_start(k),
                trans: init_trans(k),
                loc: Vector::zeros(k),
                scale: Vector::filled(k, 1.0),
                ncp: Vector::from_iter((0..k).map(|j| -0.4 + 0.8 * j as f64)),
                loglik: f64::NAN,
            });
        }
        let mut loc = Vector::from_iter((0..k).map(|j| -2.0 + 4.0 * j as f64));
        let mut scale = Vector::filled(k, 1.2);
        let mut ncp = Vector::from_iter((0..k).map(|j| -0.3 + 0.6 * j as f64));
        let mut start = init_start(k);
        let mut trans = init_trans(k);
        let mut loglik = f64::NEG_INFINITY;
        let mut last_gamma: Vec<Vec<f64>> = Vec::new();
        for it in 0..self.max_iter.max(1) {
            let dummy = FittedNoncentralTHmm {
                labels: Vector::zeros(0),
                start: start.clone(),
                trans: trans.clone(),
                loc: loc.clone(),
                scale: scale.clone(),
                ncp: ncp.clone(),
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
                    loc[j] = wy / wsum;
                    let mut wad = 0.0_f64;
                    for t in 0..t_len {
                        let y = x.get(t, 0);
                        if y.is_finite() {
                            wad += fb.gamma[t][j] * (y - loc[j]).abs();
                        }
                    }
                    scale[j] = (wad / wsum).max(0.2);
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
        let dummy = FittedNoncentralTHmm {
            labels: Vector::zeros(0),
            start: start.clone(),
            trans: trans.clone(),
            loc: loc.clone(),
            scale: scale.clone(),
            ncp: ncp.clone(),
            loglik,
        };
        let (labels, _) = viterbi_path(&start, &trans, &dummy.log_emit_seq(x));
        ctx.finish(FittedNoncentralTHmm {
            labels,
            start,
            trans,
            loc,
            scale,
            ncp,
            loglik,
        })
    }
}

/// Hypoexponential HMM (sum of two distinct exponential stages).
///
/// Rates are not identification `p`. Distinct from [`GammaHmm`] (equal rates
/// / Erlang) and [`ExponentialHmm`] (one stage).
#[derive(Clone, Debug)]
pub struct HypoexponentialHmm {
    /// Hidden states. Not identification `p`.
    pub n_states: usize,
    /// Baum–Welch cap.
    pub max_iter: usize,
}

impl Default for HypoexponentialHmm {
    fn default() -> Self {
        Self {
            n_states: 2,
            max_iter: 40,
        }
    }
}

impl HypoexponentialHmm {
    /// `k`-state hypoexponential HMM.
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
    ) -> Result<Qualified<FittedHypoexponentialHmm>> {
        self.fit_unsupervised(x, session)
    }
}

/// Fitted hypoexponential HMM.
#[derive(Clone, Debug)]
pub struct FittedHypoexponentialHmm {
    /// Viterbi path.
    pub labels: Vector,
    /// Start distribution.
    pub start: Vector,
    /// Transitions.
    pub trans: Matrix,
    /// First-stage rates.
    pub rate1: Vector,
    /// Second-stage rates.
    pub rate2: Vector,
    /// Training log-likelihood.
    pub loglik: f64,
}

impl FittedHypoexponentialHmm {
    fn log_emit_seq(&self, x: &Matrix) -> Vec<Vec<f64>> {
        let t = x.nrows();
        let ns = self.rate1.len();
        let mut out = vec![vec![f64::NEG_INFINITY; ns]; t];
        for ti in 0..t {
            let y = x.get(ti, 0);
            for j in 0..ns {
                out[ti][j] = log_hypoexp(y, self.rate1[j], self.rate2[j]);
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

impl Predict for FittedHypoexponentialHmm {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        self.decode(x, session)
    }
}

impl FitUnsupervised for HypoexponentialHmm {
    type Fitted = FittedHypoexponentialHmm;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedHypoexponentialHmm>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let t_len = x.nrows();
        let k = self.n_states.max(1);
        let mut n_skip = 0usize;
        for i in 0..t_len {
            if x.get(i, 0).is_finite() && x.get(i, 0) <= 0.0 {
                n_skip += 1;
            }
        }
        if n_skip > 0 {
            ctx.push(
                Issue::builder(IssueCode::NonPositiveSeries)
                    .severity(signlred::Severity::Warning)
                    .message(format!(
                        "HypoexponentialHmm skipped {n_skip} non-positive observations"
                    ))
                    .build(),
            );
        }
        if t_len == 0 {
            return ctx.finish(FittedHypoexponentialHmm {
                labels: empty_labels(0),
                start: init_start(k),
                trans: init_trans(k),
                rate1: Vector::filled(k, 1.0),
                rate2: Vector::filled(k, 2.0),
                loglik: f64::NAN,
            });
        }
        let mut rate1 = Vector::from_iter((0..k).map(|j| 0.6 + 0.3 * j as f64));
        let mut rate2 = Vector::from_iter((0..k).map(|j| 1.4 + 0.4 * j as f64));
        let mut start = init_start(k);
        let mut trans = init_trans(k);
        let mut loglik = f64::NEG_INFINITY;
        let mut last_gamma: Vec<Vec<f64>> = Vec::new();
        for it in 0..self.max_iter.max(1) {
            let dummy = FittedHypoexponentialHmm {
                labels: Vector::zeros(0),
                start: start.clone(),
                trans: trans.clone(),
                rate1: rate1.clone(),
                rate2: rate2.clone(),
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
                    if !y.is_finite() || y <= 0.0 {
                        continue;
                    }
                    wsum += fb.gamma[t][j];
                    wy += fb.gamma[t][j] * y;
                }
                if wsum > 1e-12 {
                    let m = (wy / wsum).max(0.2);
                    rate1[j] = (1.2 / m).clamp(0.1, 8.0);
                    rate2[j] = (2.4 / m).clamp(0.15, 10.0);
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
        let dummy = FittedHypoexponentialHmm {
            labels: Vector::zeros(0),
            start: start.clone(),
            trans: trans.clone(),
            rate1: rate1.clone(),
            rate2: rate2.clone(),
            loglik,
        };
        let (labels, _) = viterbi_path(&start, &trans, &dummy.log_emit_seq(x));
        ctx.finish(FittedHypoexponentialHmm {
            labels,
            start,
            trans,
            rate1,
            rate2,
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
        let lph = LaplaceHmm::new(2)
            .fit(&x, &Session::new("lph", "fit"))
            .expect("lph");
        assert_eq!(lph.value.labels.len(), 80);
        assert!(lph.value.scale.as_slice().iter().all(|v| v.is_finite() && *v > 0.0));
        let pah = ParetoHmm::new(2, 1.0)
            .fit(&xpos, &Session::new("pah", "fit"))
            .expect("pah");
        assert_eq!(pah.value.labels.len(), 80);
        assert!(pah.value.alpha.as_slice().iter().all(|v| v.is_finite() && *v > 0.0));
        let ltn = LogitNormalHmm::new(2)
            .fit(&betx, &Session::new("ltn", "fit"))
            .expect("ltn");
        assert_eq!(ltn.value.labels.len(), 80);
        assert!(ltn.value.var.as_slice().iter().all(|v| v.is_finite() && *v > 0.0));
        let ivg = InverseGammaHmm::new(2)
            .fit(&xpos, &Session::new("ivg", "fit"))
            .expect("ivg");
        assert_eq!(ivg.value.labels.len(), 80);
        assert!(ivg.value.shape.as_slice().iter().all(|v| v.is_finite() && *v > 0.0));
        let gbh = GumbelHmm::new(2)
            .fit(&x, &Session::new("gbh", "fit"))
            .expect("gbh");
        assert_eq!(gbh.value.labels.len(), 80);
        assert!(gbh.value.scale.as_slice().iter().all(|v| v.is_finite() && *v > 0.0));
        let wch = WrappedCauchyHmm::new(2)
            .fit(&x, &Session::new("wch", "fit"))
            .expect("wch");
        assert_eq!(wch.value.labels.len(), 80);
        assert!(wch.value.rho.as_slice().iter().all(|v| v.is_finite() && *v > 0.0 && *v < 1.0));
        let cah = CauchyHmm::new(2)
            .fit(&x, &Session::new("cah", "fit"))
            .expect("cah");
        assert_eq!(cah.value.labels.len(), 80);
        assert!(cah.value.scale.as_slice().iter().all(|v| v.is_finite() && *v > 0.0));
        let lgh = LogisticHmm::new(2)
            .fit(&x, &Session::new("lgh", "fit"))
            .expect("lgh");
        assert_eq!(lgh.value.labels.len(), 80);
        assert!(lgh.value.scale.as_slice().iter().all(|v| v.is_finite() && *v > 0.0));
        let ryh = RayleighHmm::new(2)
            .fit(&xpos, &Session::new("ryh", "fit"))
            .expect("ryh");
        assert_eq!(ryh.value.labels.len(), 80);
        assert!(ryh.value.sigma.as_slice().iter().all(|v| v.is_finite() && *v > 0.0));
        let rch = RiceHmm::new(2)
            .fit(&xpos, &Session::new("rch", "fit"))
            .expect("rch");
        assert_eq!(rch.value.labels.len(), 80);
        assert!(rch.value.sigma.as_slice().iter().all(|v| v.is_finite() && *v > 0.0));
        let nkh = NakagamiHmm::new(2)
            .fit(&xpos, &Session::new("nkh", "fit"))
            .expect("nkh");
        assert_eq!(nkh.value.labels.len(), 80);
        assert!(nkh.value.omega.as_slice().iter().all(|v| v.is_finite() && *v > 0.0));
        let alh = AsymmetricLaplaceHmm::new(2)
            .fit(&x, &Session::new("alh", "fit"))
            .expect("alh");
        assert_eq!(alh.value.labels.len(), 80);
        assert!(alh.value.left.as_slice().iter().all(|v| v.is_finite() && *v > 0.0));
        let frh = FrechetHmm::new(2)
            .fit(&xpos, &Session::new("frh", "fit"))
            .expect("frh");
        assert_eq!(frh.value.labels.len(), 80);
        assert!(frh.value.shape.as_slice().iter().all(|v| v.is_finite() && *v > 0.0));
        let wnh = WrappedNormalHmm::new(2)
            .fit(&x, &Session::new("wnh", "fit"))
            .expect("wnh");
        assert_eq!(wnh.value.labels.len(), 80);
        assert!(wnh.value.sigma.as_slice().iter().all(|v| v.is_finite() && *v > 0.0));
        let kmh = KumaraswamyHmm::new(2)
            .fit(&betx, &Session::new("kmh", "fit"))
            .expect("kmh");
        assert_eq!(kmh.value.labels.len(), 80);
        assert!(kmh.value.a.as_slice().iter().all(|v| v.is_finite() && *v > 0.0));
        let ash = ArcsineHmm::new(2)
            .fit(&betx, &Session::new("ash", "fit"))
            .expect("ash");
        assert_eq!(ash.value.labels.len(), 80);
        assert!(ash.value.alpha.as_slice().iter().all(|v| v.is_finite() && *v > 0.0 && *v < 1.0));
        let pwh = PowerHmm::new(2)
            .fit(&betx, &Session::new("pwh", "fit"))
            .expect("pwh");
        assert_eq!(pwh.value.labels.len(), 80);
        assert!(pwh.value.alpha.as_slice().iter().all(|v| v.is_finite() && *v > 0.0));
        let llh = LogLogisticHmm::new(2)
            .fit(&xpos, &Session::new("llh", "fit"))
            .expect("llh");
        assert_eq!(llh.value.labels.len(), 80);
        assert!(llh.value.shape.as_slice().iter().all(|v| v.is_finite() && *v > 0.0));
        let gev = GevHmm::new(2)
            .fit(&x, &Session::new("gev", "fit"))
            .expect("gev");
        assert_eq!(gev.value.labels.len(), 80);
        assert!(gev.value.scale.as_slice().iter().all(|v| v.is_finite() && *v > 0.0));
        let slh = SlashHmm::new(2)
            .fit(&x, &Session::new("slh", "fit"))
            .expect("slh");
        assert_eq!(slh.value.labels.len(), 80);
        assert!(slh.value.scale.as_slice().iter().all(|v| v.is_finite() && *v > 0.0));
        let snh = SkewNormalHmm::new(2)
            .fit(&x, &Session::new("snh", "fit"))
            .expect("snh");
        assert_eq!(snh.value.labels.len(), 80);
        assert!(snh.value.scale.as_slice().iter().all(|v| v.is_finite() && *v > 0.0));
        let brh = BurrHmm::new(2)
            .fit(&xpos, &Session::new("brh", "fit"))
            .expect("brh");
        assert_eq!(brh.value.labels.len(), 80);
        assert!(brh.value.k.as_slice().iter().all(|v| v.is_finite() && *v > 0.0));
        let lyh = LevyHmm::new(2)
            .fit(&xpos, &Session::new("lyh", "fit"))
            .expect("lyh");
        assert_eq!(lyh.value.labels.len(), 80);
        assert!(lyh.value.scale.as_slice().iter().all(|v| v.is_finite() && *v > 0.0));
        let cdh = CardioidHmm::new(2)
            .fit(&x, &Session::new("cdh", "fit"))
            .expect("cdh");
        assert_eq!(cdh.value.labels.len(), 80);
        assert!(cdh.value.rho.as_slice().iter().all(|v| v.is_finite() && *v > 0.0 && *v <= 0.5));
        let zig = ZeroInflatedGammaHmm::new(2)
            .fit(&xpos, &Session::new("zig", "fit"))
            .expect("zig");
        assert_eq!(zig.value.labels.len(), 80);
        assert!(zig.value.shape.as_slice().iter().all(|v| v.is_finite() && *v > 0.0));
        let ggh = GenGammaHmm::new(2)
            .fit(&xpos, &Session::new("ggh", "fit"))
            .expect("ggh");
        assert_eq!(ggh.value.labels.len(), 80);
        assert!(ggh.value.scale.as_slice().iter().all(|v| v.is_finite() && *v > 0.0));
        let dgh = DagumHmm::new(2)
            .fit(&xpos, &Session::new("dgh", "fit"))
            .expect("dgh");
        assert_eq!(dgh.value.labels.len(), 80);
        assert!(dgh.value.scale.as_slice().iter().all(|v| v.is_finite() && *v > 0.0));
        let hch = HalfCauchyHmm::new(2)
            .fit(&xpos, &Session::new("hch", "fit"))
            .expect("hch");
        assert_eq!(hch.value.labels.len(), 80);
        assert!(hch.value.scale.as_slice().iter().all(|v| v.is_finite() && *v > 0.0));
        let lmh = LomaxHmm::new(2)
            .fit(&xpos, &Session::new("lmh", "fit"))
            .expect("lmh");
        assert_eq!(lmh.value.labels.len(), 80);
        assert!(lmh.value.scale.as_slice().iter().all(|v| v.is_finite() && *v > 0.0));
        let hnh = HalfNormalHmm::new(2)
            .fit(&xpos, &Session::new("hnh", "fit"))
            .expect("hnh");
        assert_eq!(hnh.value.labels.len(), 80);
        assert!(hnh.value.scale.as_slice().iter().all(|v| v.is_finite() && *v > 0.0));
        let mxh = MaxwellHmm::new(2)
            .fit(&xpos, &Session::new("mxh", "fit"))
            .expect("mxh");
        assert_eq!(mxh.value.labels.len(), 80);
        assert!(mxh.value.scale.as_slice().iter().all(|v| v.is_finite() && *v > 0.0));
        let bph = BetaPrimeHmm::new(2)
            .fit(&xpos, &Session::new("bph", "fit"))
            .expect("bph");
        assert_eq!(bph.value.labels.len(), 80);
        assert!(bph.value.alpha.as_slice().iter().all(|v| v.is_finite() && *v > 0.0));
        let gzh = GompertzHmm::new(2)
            .fit(&xpos, &Session::new("gzh", "fit"))
            .expect("gzh");
        assert_eq!(gzh.value.labels.len(), 80);
        assert!(gzh.value.c.as_slice().iter().all(|v| v.is_finite() && *v > 0.0));
        let fdh = FDistHmm::new(2)
            .fit(&xpos, &Session::new("fdh", "fit"))
            .expect("fdh");
        assert_eq!(fdh.value.labels.len(), 80);
        assert!(fdh.value.d2.as_slice().iter().all(|v| v.is_finite() && *v > 2.0));
        let hgh = HurdleGammaHmm::new(2)
            .fit(&xpos, &Session::new("hgh", "fit"))
            .expect("hgh");
        assert_eq!(hgh.value.labels.len(), 80);
        assert!(hgh.value.scale.as_slice().iter().all(|v| v.is_finite() && *v > 0.0));
        let hsh = HyperbolicSecantHmm::new(2)
            .fit(&x, &Session::new("hsh", "fit"))
            .expect("hsh");
        assert_eq!(hsh.value.labels.len(), 80);
        assert!(hsh.value.scale.as_slice().iter().all(|v| v.is_finite() && *v > 0.0));
        let myh = MoyalHmm::new(2)
            .fit(&x, &Session::new("myh", "fit"))
            .expect("myh");
        assert_eq!(myh.value.labels.len(), 80);
        assert!(myh.value.scale.as_slice().iter().all(|v| v.is_finite() && *v > 0.0));
        let weh = WrappedExponentialHmm::new(2)
            .fit(&x, &Session::new("weh", "fit"))
            .expect("weh");
        assert_eq!(weh.value.labels.len(), 80);
        assert!(weh.value.rate.as_slice().iter().all(|v| v.is_finite() && *v > 0.0));
        let cih = ChiHmm::new(2)
            .fit(&xpos, &Session::new("cih", "fit"))
            .expect("cih");
        assert_eq!(cih.value.labels.len(), 80);
        assert!(cih.value.df.as_slice().iter().all(|v| v.is_finite() && *v > 0.0));
        let rch = RaisedCosineHmm::new(2)
            .fit(&x, &Session::new("rch", "fit"))
            .expect("rch");
        assert_eq!(rch.value.labels.len(), 80);
        assert!(rch.value.scale.as_slice().iter().all(|v| v.is_finite() && *v > 0.0));
        let luh = LogUniformHmm::new(2)
            .fit(&xpos, &Session::new("luh", "fit"))
            .expect("luh");
        assert_eq!(luh.value.labels.len(), 80);
        assert!(luh.value.lo.as_slice().iter().all(|v| v.is_finite() && *v > 0.0));
        let trh = TriangularHmm::new(2)
            .fit(&x, &Session::new("trh", "fit"))
            .expect("trh");
        assert_eq!(trh.value.labels.len(), 80);
        assert!(trh.value.scale.as_slice().iter().all(|v| v.is_finite() && *v > 0.0));
        let wgh = WignerHmm::new(2)
            .fit(&x, &Session::new("wgh", "fit"))
            .expect("wgh");
        assert_eq!(wgh.value.labels.len(), 80);
        assert!(wgh.value.radius.as_slice().iter().all(|v| v.is_finite() && *v > 0.0));
        let jsu = JohnsonSuHmm::new(2)
            .fit(&x, &Session::new("jsu", "fit"))
            .expect("jsu");
        assert_eq!(jsu.value.labels.len(), 80);
        assert!(jsu.value.delta.as_slice().iter().all(|v| v.is_finite() && *v > 0.0));
        let wlh = WrappedLaplaceHmm::new(2)
            .fit(&x, &Session::new("wlh", "fit"))
            .expect("wlh");
        assert_eq!(wlh.value.labels.len(), 80);
        assert!(wlh.value.kappa.as_slice().iter().all(|v| v.is_finite() && *v > 0.0));
        let ics = InverseChiSquaredHmm::new(2)
            .fit(&xpos, &Session::new("ics", "fit"))
            .expect("ics");
        assert_eq!(ics.value.labels.len(), 80);
        assert!(ics.value.tau.as_slice().iter().all(|v| v.is_finite() && *v > 0.0));
        let nig = NormalInverseGaussianHmm::new(2)
            .fit(&x, &Session::new("nig", "fit"))
            .expect("nig");
        assert_eq!(nig.value.labels.len(), 80);
        assert!(nig.value.delta.as_slice().iter().all(|v| v.is_finite() && *v > 0.0));
        let vgh = VarianceGammaHmm::new(2)
            .fit(&x, &Session::new("vgh", "fit"))
            .expect("vgh");
        assert_eq!(vgh.value.labels.len(), 80);
        assert!(vgh.value.scale.as_slice().iter().all(|v| v.is_finite() && *v > 0.0));
        let jph = JonesPewseyHmm::new(2)
            .fit(&x, &Session::new("jph", "fit"))
            .expect("jph");
        assert_eq!(jph.value.labels.len(), 80);
        assert!(jph.value.kappa.as_slice().iter().all(|v| v.is_finite() && *v > 0.0));
        let hbh = HyperbolicHmm::new(2)
            .fit(&x, &Session::new("hbh", "fit"))
            .expect("hbh");
        assert_eq!(hbh.value.labels.len(), 80);
        assert!(hbh.value.delta.as_slice().iter().all(|v| v.is_finite() && *v > 0.0));
        let skv = SineSkewedVonMisesHmm::new(2)
            .fit(&x, &Session::new("skv", "fit"))
            .expect("skv");
        assert_eq!(skv.value.labels.len(), 80);
        assert!(skv.value.kappa.as_slice().iter().all(|v| v.is_finite() && *v > 0.0));
        let nch = NoncentralChiSquaredHmm::new(2)
            .fit(&xpos, &Session::new("nch", "fit"))
            .expect("nch");
        assert_eq!(nch.value.labels.len(), 80);
        assert!(nch.value.lam.as_slice().iter().all(|v| v.is_finite() && *v >= 0.0));
        let gpd = GeneralizedParetoHmm::new(2)
            .fit(&xpos, &Session::new("gpd", "fit"))
            .expect("gpd");
        assert_eq!(gpd.value.labels.len(), 80);
        assert!(gpd.value.scale.as_slice().iter().all(|v| v.is_finite() && *v > 0.0));
        let tlk = TukeyLambdaHmm::new(2)
            .fit(&x, &Session::new("tlk", "fit"))
            .expect("tlk");
        assert_eq!(tlk.value.labels.len(), 80);
        assert!(tlk.value.scale.as_slice().iter().all(|v| v.is_finite() && *v > 0.0));
        let ewh = ExpWeibullHmm::new(2)
            .fit(&xpos, &Session::new("ewh", "fit"))
            .expect("ewh");
        assert_eq!(ewh.value.labels.len(), 80);
        assert!(ewh.value.scale.as_slice().iter().all(|v| v.is_finite() && *v > 0.0));
        let pnh = ProjectedNormalHmm::new(2)
            .fit(&x, &Session::new("pnh", "fit"))
            .expect("pnh");
        assert_eq!(pnh.value.labels.len(), 80);
        assert!(pnh.value.tau.as_slice().iter().all(|v| v.is_finite() && *v > 0.0));
        let kjh = KatoJonesHmm::new(2)
            .fit(&x, &Session::new("kjh", "fit"))
            .expect("kjh");
        assert_eq!(kjh.value.labels.len(), 80);
        assert!(kjh.value.kappa.as_slice().iter().all(|v| v.is_finite() && *v >= 0.0));
        let nct = NoncentralTHmm::new(2)
            .fit(&x, &Session::new("nct", "fit"))
            .expect("nct");
        assert_eq!(nct.value.labels.len(), 80);
        assert!(nct.value.scale.as_slice().iter().all(|v| v.is_finite() && *v > 0.0));
        let hoe = HypoexponentialHmm::new(2)
            .fit(&xpos, &Session::new("hoe", "fit"))
            .expect("hoe");
        assert_eq!(hoe.value.labels.len(), 80);
        assert!(hoe.value.rate1.as_slice().iter().all(|v| v.is_finite() && *v > 0.0));
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
        let zin = ZeroInflatedNegBinHmm::new(2)
            .fit(&x, &Session::new("zin", "fit"))
            .expect("zin");
        assert_eq!(zin.value.labels.len(), 40);
        assert!(zin.value.r.as_slice().iter().all(|v| v.is_finite() && *v > 0.0));
        let hyh = HypergeometricHmm::new(2, 20.0, 10.0)
            .fit(&x, &Session::new("hyh", "fit"))
            .expect("hyh");
        assert_eq!(hyh.value.labels.len(), 40);
        assert!(hyh.value.k_succ.as_slice().iter().all(|v| v.is_finite() && *v > 0.0));
        let zib = ZeroInflatedBinomialHmm::new(2, 10.0)
            .fit(&x, &Session::new("zib", "fit"))
            .expect("zib");
        assert_eq!(zib.value.labels.len(), 40);
        assert!(zib.value.p.as_slice().iter().all(|v| v.is_finite() && *v > 0.0 && *v < 1.0));
        let hph = HurdlePoissonHmm::new(2)
            .fit(&x, &Session::new("hph", "fit"))
            .expect("hph");
        assert_eq!(hph.value.labels.len(), 40);
        assert!(hph.value.lam.as_slice().iter().all(|v| v.is_finite() && *v > 0.0));
        let cph = ComPoissonHmm::new(2)
            .fit(&x, &Session::new("cph", "fit"))
            .expect("cph");
        assert_eq!(cph.value.labels.len(), 40);
        assert!(cph.value.lam.as_slice().iter().all(|v| v.is_finite() && *v > 0.0));
        let dvw = DiscreteWeibullHmm::new(2)
            .fit(&x, &Session::new("dvw", "fit"))
            .expect("dvw");
        assert_eq!(dvw.value.labels.len(), 40);
        assert!(dvw.value.q.as_slice().iter().all(|v| v.is_finite() && *v > 0.0 && *v < 1.0));
        let skh = SkellamHmm::new(2)
            .fit(&x, &Session::new("skh", "fit"))
            .expect("skh");
        assert_eq!(skh.value.labels.len(), 40);
        assert!(skh.value.mu1.as_slice().iter().all(|v| v.is_finite() && *v > 0.0));
        let gph = GenPoissonHmm::new(2)
            .fit(&x, &Session::new("gph", "fit"))
            .expect("gph");
        assert_eq!(gph.value.labels.len(), 40);
        assert!(gph.value.lam.as_slice().iter().all(|v| v.is_finite() && *v > 0.0));
        let lsh = LogarithmicHmm::new(2)
            .fit(&x, &Session::new("lsh", "fit"))
            .expect("lsh");
        assert_eq!(lsh.value.labels.len(), 40);
        assert!(lsh.value.p.as_slice().iter().all(|v| v.is_finite() && *v > 0.0 && *v < 1.0));
        let ysh = YuleSimonHmm::new(2)
            .fit(&x, &Session::new("ysh", "fit"))
            .expect("ysh");
        assert_eq!(ysh.value.labels.len(), 40);
        assert!(ysh.value.rho.as_slice().iter().all(|v| v.is_finite() && *v > 0.0));
        let zth = ZetaHmm::new(2)
            .fit(&x, &Session::new("zth", "fit"))
            .expect("zth");
        assert_eq!(zth.value.labels.len(), 40);
        assert!(zth.value.s.as_slice().iter().all(|v| v.is_finite() && *v > 1.0));
        let zmh = ZipfMandelbrotHmm::new(2)
            .fit(&x, &Session::new("zmh", "fit"))
            .expect("zmh");
        assert_eq!(zmh.value.labels.len(), 40);
        assert!(zmh.value.s.as_slice().iter().all(|v| v.is_finite() && *v > 1.0));
        let hrh = HermiteHmm::new(2)
            .fit(&x, &Session::new("hrh", "fit"))
            .expect("hrh");
        assert_eq!(hrh.value.labels.len(), 40);
        assert!(hrh.value.lam1.as_slice().iter().all(|v| v.is_finite() && *v > 0.0));
        let dlh = DelaporteHmm::new(2)
            .fit(&x, &Session::new("dlh", "fit"))
            .expect("dlh");
        assert_eq!(dlh.value.labels.len(), 40);
        assert!(dlh.value.lam.as_slice().iter().all(|v| v.is_finite() && *v > 0.0));
        let nta = NeymanTypeAHmm::new(2)
            .fit(&x, &Session::new("nta", "fit"))
            .expect("nta");
        assert_eq!(nta.value.labels.len(), 40);
        assert!(nta.value.lam.as_slice().iter().all(|v| v.is_finite() && *v > 0.0));
        let boh = BorelHmm::new(2)
            .fit(&x, &Session::new("boh", "fit"))
            .expect("boh");
        assert_eq!(boh.value.labels.len(), 40);
        assert!(boh.value.mu.as_slice().iter().all(|v| v.is_finite() && *v > 0.0 && *v < 1.0));
        let ply = PolyaAeppliHmm::new(2)
            .fit(&x, &Session::new("ply", "fit"))
            .expect("ply");
        assert_eq!(ply.value.labels.len(), 40);
        assert!(ply.value.lam.as_slice().iter().all(|v| v.is_finite() && *v > 0.0));
        let sch = SichelHmm::new(2)
            .fit(&x, &Session::new("sch", "fit"))
            .expect("sch");
        assert_eq!(sch.value.labels.len(), 40);
        assert!(sch.value.mu.as_slice().iter().all(|v| v.is_finite() && *v > 0.0));
        let pln = PoissonLognormalHmm::new(2)
            .fit(&x, &Session::new("pln", "fit"))
            .expect("pln");
        assert_eq!(pln.value.labels.len(), 40);
        assert!(pln.value.sigma.as_slice().iter().all(|v| v.is_finite() && *v > 0.0));
        let wrh = WaringHmm::new(2)
            .fit(&x, &Session::new("wrh", "fit"))
            .expect("wrh");
        assert_eq!(wrh.value.labels.len(), 40);
        assert!(wrh.value.rho.as_slice().iter().all(|v| v.is_finite() && *v > 1.0));
        let gdh = GoodHmm::new(2)
            .fit(&x, &Session::new("gdh", "fit"))
            .expect("gdh");
        assert_eq!(gdh.value.labels.len(), 40);
        assert!(gdh.value.q.as_slice().iter().all(|v| v.is_finite() && *v > 0.0 && *v < 1.0));
        let pbh = PoissonBinomialHmm::new(2)
            .fit(&x, &Session::new("pbh", "fit"))
            .expect("pbh");
        assert_eq!(pbh.value.labels.len(), 40);
        assert!(pbh.value.p1.as_slice().iter().all(|v| v.is_finite() && *v > 0.0 && *v < 1.0));
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
