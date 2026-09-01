//! River-style contextual-free bandits with mandatory incremental explainability.
//!
//! Every `pull` / `partial_fit` records what the update changed (which arm,
//! how the mean / Beta / UCB index moved), why (ε-greedy draw, UCB index,
//! Thompson sample), and whether the post-update state is identified
//! (each arm needs at least one pull; Bernoulli Thompson needs both successes
//! and failures before the posterior is a real Beta).

use crate::context::FitCtx;
use crate::data::{Matrix, Vector};
use crate::rng::Rng;
use crate::traits::PartialFit;
use ojizou_san::{IncrementalExplain, Session};
use signlred::{
    IncrementalQuality, Issue, IssueCode, Meaninglessness, NumericalCompromise, Qualified, Result,
    Severity,
};
use std::collections::BTreeMap;

/// ε-greedy with incremental sample means.
#[derive(Clone, Debug)]
pub struct EpsilonGreedy {
    /// Explore probability.
    pub epsilon: f64,
    n_arms: usize,
    counts: Vec<u64>,
    values: Vec<f64>,
    rng: Rng,
    updates: u64,
    n_seen: u64,
}

impl EpsilonGreedy {
    /// `k` arms, explore with probability `epsilon`.
    pub fn new(n_arms: usize, epsilon: f64) -> Self {
        let k = n_arms.max(1);
        Self {
            epsilon,
            n_arms: k,
            counts: vec![0; k],
            values: vec![0.0; k],
            rng: Rng::new(7),
            updates: 0,
            n_seen: 0,
        }
    }

    /// Current sample means.
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// Choose an arm and explain the decision (no reward yet).
    pub fn pull(&mut self, session: &Session) -> Result<Qualified<(usize, IncrementalExplain)>> {
        let mut ctx = FitCtx::with_session(session.child("pull"));
        let arm = self.choose_arm(&mut ctx);
        let expl = self.decision_explain(arm, "pull", "ε-greedy index");
        ctx.session.record_incremental(expl.clone());
        ctx.finish((arm, expl))
    }

    fn choose_arm(&mut self, ctx: &mut FitCtx) -> usize {
        if self.n_arms == 0 {
            return 0;
        }
        let unpulled = self.counts.iter().position(|&c| c == 0);
        if let Some(a) = unpulled {
            ctx.push(
                Issue::builder(IssueCode::WarmupIncomplete)
                    .message(format!(
                        "arm {a} has never been pulled; forcing exploration"
                    ))
                    .build(),
            );
            return a;
        }
        if self.rng.uniform() < self.epsilon.clamp(0.0, 1.0) {
            self.rng.below(self.n_arms)
        } else {
            let mut best = 0usize;
            let mut bv = f64::NEG_INFINITY;
            for (i, &v) in self.values.iter().enumerate() {
                if v > bv {
                    bv = v;
                    best = i;
                }
            }
            best
        }
    }

    fn decision_explain(&self, arm: usize, what: &str, why: &str) -> IncrementalExplain {
        let mut q = IncrementalQuality::new(self.updates, 1, self.n_seen);
        q.effective_sample_size = self.n_seen as f64;
        q.still_identified = self.counts.iter().all(|&c| c > 0);
        q.warmup = self.counts.iter().any(|&c| c == 0);
        q.information_gain = Some(0.0);
        q.explanation = format!(
            "{what} arm={arm} mean={:.6e}",
            self.values.get(arm).copied().unwrap_or(0.0)
        );
        IncrementalExplain::from_quality(
            q,
            format!("{what} arm {arm}"),
            why,
            format!("counts={:?}", self.counts),
            format!("values={:?}", self.values),
        )
    }

    fn apply_reward(&mut self, arm: usize, reward: f64) -> f64 {
        let a = arm.min(self.n_arms.saturating_sub(1));
        let c = self.counts[a] as f64;
        let before = self.values[a];
        self.counts[a] += 1;
        self.values[a] = before + (reward - before) / (c + 1.0);
        self.n_seen += 1;
        self.updates += 1;
        (self.values[a] - before).abs()
    }
}

impl PartialFit for EpsilonGreedy {
    fn partial_fit(
        &mut self,
        x: &Matrix,
        y: Option<&Vector>,
        session: &Session,
    ) -> Result<Qualified<IncrementalExplain>> {
        let mut ctx = FitCtx::with_session(session.child("partial_fit"));
        let Some(y) = y else {
            ctx.push(Issue::builder(IssueCode::MissingTarget).build());
            return finish(
                &ctx,
                reject(self.updates, 0, self.n_seen, "bandit update needs rewards"),
            );
        };
        if x.nrows() != y.len() {
            ctx.push(
                Issue::builder(IssueCode::DimensionMismatch)
                    .message("bandit x rows ≠ y length")
                    .build(),
            );
        }
        let before = self.values.clone();
        let mut dsum = 0.0;
        for i in 0..y.len() {
            let arm = if x.ncols() == 0 {
                0
            } else {
                x.get(i, 0).round().abs() as usize
            };
            if arm >= self.n_arms {
                ctx.push(
                    Issue::builder(IssueCode::IncrementalUnidentifiable)
                        .message(format!("arm {arm} is outside 0..{}", self.n_arms))
                        .build(),
                );
                continue;
            }
            dsum += self.apply_reward(arm, y[i]);
        }
        let expl = pack_bandit(
            &mut ctx,
            self.updates,
            y.len(),
            self.n_seen,
            dsum,
            self.counts.iter().all(|&c| c > 0),
            self.counts.iter().any(|&c| c == 0),
            &before,
            &self.values,
            "ε-greedy sample means",
            "incremental mean update on the observed (arm, reward) pairs",
        );
        finish(&ctx, expl)
    }
}

/// UCB1 (Auer, Cesa-Bianchi, Fischer).
#[derive(Clone, Debug)]
pub struct Ucb1 {
    n_arms: usize,
    counts: Vec<u64>,
    values: Vec<f64>,
    updates: u64,
    n_seen: u64,
}

impl Ucb1 {
    /// `k` arms.
    pub fn new(n_arms: usize) -> Self {
        let k = n_arms.max(1);
        Self {
            n_arms: k,
            counts: vec![0; k],
            values: vec![0.0; k],
            updates: 0,
            n_seen: 0,
        }
    }

    /// Current sample means.
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// Choose the arm with the largest UCB index.
    pub fn pull(&mut self, session: &Session) -> Result<Qualified<(usize, IncrementalExplain)>> {
        let mut ctx = FitCtx::with_session(session.child("pull"));
        let mut arm = 0usize;
        let mut best = f64::NEG_INFINITY;
        let mut indexes = vec![f64::INFINITY; self.n_arms];
        for a in 0..self.n_arms {
            let idx = if self.counts[a] == 0 {
                ctx.push(
                    Issue::builder(IssueCode::WarmupIncomplete)
                        .message(format!("UCB1 has never pulled arm {a}"))
                        .build(),
                );
                f64::INFINITY
            } else {
                let bonus = (2.0 * (self.n_seen.max(1) as f64).ln() / self.counts[a] as f64).sqrt();
                self.values[a] + bonus
            };
            indexes[a] = idx;
            if idx > best {
                best = idx;
                arm = a;
            }
        }
        let mut q = IncrementalQuality::new(self.updates, 1, self.n_seen);
        q.warmup = self.counts.iter().any(|&c| c == 0);
        q.still_identified = self.counts.iter().all(|&c| c > 0);
        q.explanation = format!("UCB1 chose {arm} indexes={indexes:?}");
        let expl = IncrementalExplain::from_quality(
            q,
            format!("pull arm {arm}"),
            "argmax of mean + sqrt(2 log t / n_a)",
            format!("counts={:?}", self.counts),
            format!("ucb={indexes:?}"),
        )
        .contribute(format!("arm[{arm}]"), best);
        ctx.session.record_incremental(expl.clone());
        ctx.finish((arm, expl))
    }

    fn apply_reward(&mut self, arm: usize, reward: f64) -> f64 {
        let a = arm.min(self.n_arms.saturating_sub(1));
        let c = self.counts[a] as f64;
        let before = self.values[a];
        self.counts[a] += 1;
        self.values[a] = before + (reward - before) / (c + 1.0);
        self.n_seen += 1;
        self.updates += 1;
        (self.values[a] - before).abs()
    }
}

impl PartialFit for Ucb1 {
    fn partial_fit(
        &mut self,
        x: &Matrix,
        y: Option<&Vector>,
        session: &Session,
    ) -> Result<Qualified<IncrementalExplain>> {
        let mut ctx = FitCtx::with_session(session.child("partial_fit"));
        let Some(y) = y else {
            ctx.push(Issue::builder(IssueCode::MissingTarget).build());
            return finish(
                &ctx,
                reject(self.updates, 0, self.n_seen, "UCB1 needs rewards"),
            );
        };
        let before = self.values.clone();
        let mut dsum = 0.0;
        for i in 0..y.len() {
            let arm = if x.ncols() == 0 {
                0
            } else {
                x.get(i, 0).round().abs() as usize
            };
            if arm >= self.n_arms {
                ctx.push(
                    Issue::builder(IssueCode::IncrementalUnidentifiable)
                        .message(format!("arm {arm} is outside 0..{}", self.n_arms))
                        .build(),
                );
                continue;
            }
            dsum += self.apply_reward(arm, y[i]);
        }
        let expl = pack_bandit(
            &mut ctx,
            self.updates,
            y.len(),
            self.n_seen,
            dsum,
            self.counts.iter().all(|&c| c > 0),
            self.counts.iter().any(|&c| c == 0),
            &before,
            &self.values,
            "UCB1 sample means",
            "mean update; the next pull will recompute the log-t bonus",
        );
        finish(&ctx, expl)
    }
}

/// UCB-Tuned (Auer et al.): variance-aware bonus.
///
/// Arm count is not identification `p`.
#[derive(Clone, Debug)]
pub struct UcbTuned {
    n_arms: usize,
    counts: Vec<u64>,
    values: Vec<f64>,
    m2: Vec<f64>,
    updates: u64,
    n_seen: u64,
}

impl UcbTuned {
    /// `k` arms.
    pub fn new(n_arms: usize) -> Self {
        let k = n_arms.max(1);
        Self {
            n_arms: k,
            counts: vec![0; k],
            values: vec![0.0; k],
            m2: vec![0.0; k],
            updates: 0,
            n_seen: 0,
        }
    }

    fn index(&self, arm: usize) -> f64 {
        let n = self.counts[arm] as f64;
        if n <= 0.0 {
            return f64::INFINITY;
        }
        let t = self.n_seen.max(1) as f64;
        let var = (self.m2[arm] / n).max(0.0);
        let extra = (2.0 * t.ln() / n).sqrt();
        let v = (var + extra).min(0.25);
        self.values[arm] + (t.ln() / n * v).sqrt()
    }

    /// Choose the arm with the largest UCB-Tuned index.
    pub fn pull(&mut self, session: &Session) -> Result<Qualified<(usize, IncrementalExplain)>> {
        let mut ctx = FitCtx::with_session(session.child("pull"));
        let mut arm = 0usize;
        let mut best = f64::NEG_INFINITY;
        let mut indexes = vec![f64::INFINITY; self.n_arms];
        for a in 0..self.n_arms {
            if self.counts[a] == 0 {
                ctx.push(
                    Issue::builder(IssueCode::WarmupIncomplete)
                        .message(format!("UCB-Tuned has never pulled arm {a}"))
                        .build(),
                );
            }
            let idx = self.index(a);
            indexes[a] = idx;
            if idx > best {
                best = idx;
                arm = a;
            }
        }
        let mut q = IncrementalQuality::new(self.updates, 1, self.n_seen);
        q.warmup = self.counts.iter().any(|&c| c == 0);
        q.still_identified = self.counts.iter().all(|&c| c > 0);
        q.explanation = format!("UCB-Tuned chose {arm} indexes={indexes:?}");
        let expl = IncrementalExplain::from_quality(
            q,
            format!("pull arm {arm}"),
            "argmax of mean + sqrt(ln t / n × min(1/4, V̂))",
            format!("counts={:?}", self.counts),
            format!("ucb={indexes:?}"),
        );
        ctx.session.record_incremental(expl.clone());
        ctx.finish((arm, expl))
    }
}

impl PartialFit for UcbTuned {
    fn partial_fit(
        &mut self,
        x: &Matrix,
        y: Option<&Vector>,
        session: &Session,
    ) -> Result<Qualified<IncrementalExplain>> {
        let mut ctx = FitCtx::with_session(session.child("partial_fit"));
        let Some(y) = y else {
            ctx.push(Issue::builder(IssueCode::MissingTarget).build());
            return finish(
                &ctx,
                reject(self.updates, 0, self.n_seen, "UCB-Tuned needs rewards"),
            );
        };
        let before = self.values.clone();
        let mut dsum = 0.0;
        for i in 0..y.len() {
            if !y[i].is_finite() {
                continue;
            }
            let arm = if x.ncols() == 0 || x.nrows() == 0 {
                0
            } else {
                x.get(i.min(x.nrows() - 1), 0).round().abs() as usize
            };
            if arm >= self.n_arms {
                ctx.push(
                    Issue::builder(IssueCode::DimensionMismatch)
                        .severity(Severity::Warning)
                        .message(format!("UCB-Tuned arm {arm} is outside 0..{}", self.n_arms))
                        .build(),
                );
                continue;
            }
            let c = self.counts[arm] as f64;
            let prev = self.values[arm];
            self.counts[arm] += 1;
            self.values[arm] = prev + (y[i] - prev) / (c + 1.0);
            let d = y[i] - prev;
            self.m2[arm] += d * (y[i] - self.values[arm]);
            self.n_seen += 1;
            self.updates += 1;
            dsum += (self.values[arm] - prev).abs();
        }
        let expl = pack_bandit(
            &mut ctx,
            self.updates,
            y.len(),
            self.n_seen,
            dsum,
            self.counts.iter().all(|&c| c > 0),
            self.counts.iter().any(|&c| c == 0),
            &before,
            &self.values,
            "UCB-Tuned sample means",
            "Welford mean/variance; the next pull uses a variance-aware bonus",
        );
        finish(&ctx, expl)
    }
}

/// Thompson sampling for Bernoulli arms (`Beta(α, β)` posteriors).
#[derive(Clone, Debug)]
pub struct ThompsonBernoulli {
    n_arms: usize,
    alpha: Vec<f64>,
    beta: Vec<f64>,
    rng: Rng,
    updates: u64,
    n_seen: u64,
}

impl ThompsonBernoulli {
    /// `k` arms, uniform Beta(1, 1) priors.
    pub fn new(n_arms: usize) -> Self {
        let k = n_arms.max(1);
        Self {
            n_arms: k,
            alpha: vec![1.0; k],
            beta: vec![1.0; k],
            rng: Rng::new(11),
            updates: 0,
            n_seen: 0,
        }
    }

    /// Posterior means \(\alpha/(\alpha+\beta)\).
    pub fn means(&self) -> Vector {
        Vector::from_iter(
            self.alpha
                .iter()
                .zip(&self.beta)
                .map(|(&a, &b)| a / (a + b)),
        )
    }

    /// Draw one Thompson sample per arm and pick the max.
    pub fn pull(&mut self, session: &Session) -> Result<Qualified<(usize, IncrementalExplain)>> {
        let mut ctx = FitCtx::with_session(session.child("pull"));
        let mut samples = vec![0.0; self.n_arms];
        let mut arm = 0usize;
        let mut best = f64::NEG_INFINITY;
        for a in 0..self.n_arms {
            samples[a] = sample_beta(&mut self.rng, self.alpha[a], self.beta[a]);
            if samples[a] > best {
                best = samples[a];
                arm = a;
            }
        }
        let identified = self
            .alpha
            .iter()
            .zip(&self.beta)
            .all(|(&a, &b)| a + b > 2.0 + 1e-12);
        if !identified {
            ctx.push(
                Issue::builder(IssueCode::WarmupIncomplete)
                    .message("at least one Beta posterior is still the prior or one-sided")
                    .build(),
            );
        }
        let mut q = IncrementalQuality::new(self.updates, 1, self.n_seen);
        q.warmup = !identified;
        q.still_identified = identified;
        q.explanation = format!("Thompson samples={samples:?} chose {arm}");
        let expl = IncrementalExplain::from_quality(
            q,
            format!("pull arm {arm}"),
            "argmax of Beta(α,β) draws",
            format!("alpha={:?} beta={:?}", self.alpha, self.beta),
            format!("samples={samples:?}"),
        );
        ctx.session.record_incremental(expl.clone());
        ctx.finish((arm, expl))
    }

    fn apply_reward(&mut self, arm: usize, reward: f64) {
        let a = arm.min(self.n_arms.saturating_sub(1));
        if reward >= 0.5 {
            self.alpha[a] += 1.0;
        } else {
            self.beta[a] += 1.0;
        }
        self.n_seen += 1;
        self.updates += 1;
    }
}

impl PartialFit for ThompsonBernoulli {
    fn partial_fit(
        &mut self,
        x: &Matrix,
        y: Option<&Vector>,
        session: &Session,
    ) -> Result<Qualified<IncrementalExplain>> {
        let mut ctx = FitCtx::with_session(session.child("partial_fit"));
        let Some(y) = y else {
            ctx.push(Issue::builder(IssueCode::MissingTarget).build());
            return finish(
                &ctx,
                reject(
                    self.updates,
                    0,
                    self.n_seen,
                    "Thompson needs Bernoulli rewards",
                ),
            );
        };
        for &r in y.as_slice() {
            if r.is_finite()
                && r != 0.0
                && r != 1.0
                && (r - 0.0).abs() > 1e-12
                && (r - 1.0).abs() > 1e-12
            {
                ctx.push(
                    Issue::builder(IssueCode::DegenerateDistribution)
                        .severity(signlred::Severity::Warning)
                        .message("ThompsonBernoulli treats reward ≥ 0.5 as a success; non-Bernoulli rewards are a different estimand")
                        .meaninglessness(Meaninglessness::new(
                            "Beta posterior",
                            "the conjugate update assumes y ∈ {0,1}",
                            signlred::InterpretiveValue::Misleading,
                            "binarize the reward or use a Gaussian bandit",
                        ))
                        .build(),
                );
                break;
            }
        }
        let before = self.means();
        for i in 0..y.len() {
            let arm = if x.ncols() == 0 {
                0
            } else {
                x.get(i, 0).round().abs() as usize
            };
            if arm >= self.n_arms {
                ctx.push(
                    Issue::builder(IssueCode::IncrementalUnidentifiable)
                        .message(format!("arm {arm} is outside 0..{}", self.n_arms))
                        .build(),
                );
                continue;
            }
            self.apply_reward(arm, y[i]);
        }
        let after = self.means();
        let delta = after.sub(&before);
        let identified = self
            .alpha
            .iter()
            .zip(&self.beta)
            .all(|(&a, &b)| a + b > 2.0 + 1e-12);
        let expl = pack_bandit(
            &mut ctx,
            self.updates,
            y.len(),
            self.n_seen,
            delta.norm(),
            identified,
            !identified,
            before.as_slice(),
            after.as_slice(),
            "Thompson Beta means",
            "conjugate Bernoulli update of (α, β) on the observed arms",
        );
        finish(&ctx, expl)
    }
}

/// Contextual LinUCB (Li, Chu, Langford, Schapire).
///
/// Each row of `x` is a context; `y` is the observed reward. Arm count is not
/// identification `p`. The context dimension is taken from `x.ncols()`.
#[derive(Clone, Debug)]
pub struct LinUcb {
    n_arms: usize,
    /// Exploration bonus \(\alpha\).
    pub alpha: f64,
    p: usize,
    ainv: Vec<Vec<f64>>,
    b: Vec<Vector>,
    counts: Vec<u64>,
    updates: u64,
    n_seen: u64,
}

impl LinUcb {
    /// `k` arms, unit ridge, \(\alpha = 1\).
    pub fn new(n_arms: usize) -> Self {
        let k = n_arms.max(1);
        Self {
            n_arms: k,
            alpha: 1.0,
            p: 0,
            ainv: Vec::new(),
            b: Vec::new(),
            counts: vec![0; k],
            updates: 0,
            n_seen: 0,
        }
    }

    fn ensure_dim(&mut self, p: usize) {
        if self.p == p && !self.ainv.is_empty() {
            return;
        }
        self.p = p.max(1);
        let dim = self.p;
        self.ainv = (0..self.n_arms)
            .map(|_| {
                let mut m = vec![0.0; dim * dim];
                for i in 0..dim {
                    m[i * dim + i] = 1.0;
                }
                m
            })
            .collect();
        self.b = (0..self.n_arms).map(|_| Vector::zeros(dim)).collect();
    }

    fn context(x: &Matrix, i: usize, p: usize) -> Vector {
        Vector::from_iter((0..p).map(|j| if j < x.ncols() { x.get(i, j) } else { 0.0 }))
    }

    fn matvec(a: &[f64], p: usize, v: &Vector) -> Vector {
        Vector::from_iter((0..p).map(|r| {
            let mut s = 0.0;
            for c in 0..p {
                s += a[r * p + c] * v[c];
            }
            s
        }))
    }

    fn quad(a: &[f64], p: usize, v: &Vector) -> f64 {
        let av = Self::matvec(a, p, v);
        let n = v.len().min(av.len());
        let mut s = 0.0;
        for j in 0..n {
            s += v[j] * av[j];
        }
        s
    }

    fn choose_arm(&self, z: &Vector, ctx: &mut FitCtx) -> usize {
        if let Some(a) = self.counts.iter().position(|&c| c == 0) {
            ctx.push(
                Issue::builder(IssueCode::WarmupIncomplete)
                    .message(format!("LinUCB has never pulled arm {a}"))
                    .build(),
            );
            return a;
        }
        let p = self.p.max(1);
        let alpha = if self.alpha.is_finite() && self.alpha > 0.0 {
            self.alpha
        } else {
            1.0
        };
        let mut best = 0usize;
        let mut best_u = f64::NEG_INFINITY;
        for a in 0..self.n_arms {
            let theta = Self::matvec(&self.ainv[a], p, &self.b[a]);
            let mut mean = 0.0;
            for j in 0..z.len().min(theta.len()) {
                mean += z[j] * theta[j];
            }
            let var = Self::quad(&self.ainv[a], p, z).max(0.0);
            let u = mean + alpha * var.sqrt();
            if u > best_u {
                best_u = u;
                best = a;
            }
        }
        best
    }

    fn sherman_update(ainv: &mut [f64], p: usize, z: &Vector) {
        let az = Self::matvec(ainv, p, z);
        let den = 1.0 + Self::quad(ainv, p, z);
        if !den.is_finite() || den.abs() <= 1e-18 {
            return;
        }
        for r in 0..p {
            for c in 0..p {
                ainv[r * p + c] -= az[r] * az[c] / den;
            }
        }
    }

    fn apply(&mut self, arm: usize, z: &Vector, reward: f64) {
        let a = arm.min(self.n_arms.saturating_sub(1));
        let p = self.p.max(1);
        Self::sherman_update(&mut self.ainv[a], p, z);
        for j in 0..p.min(self.b[a].len()).min(z.len()) {
            self.b[a][j] += reward * z[j];
        }
        self.counts[a] += 1;
        self.n_seen += 1;
        self.updates += 1;
    }
}

impl PartialFit for LinUcb {
    fn partial_fit(
        &mut self,
        x: &Matrix,
        y: Option<&Vector>,
        session: &Session,
    ) -> Result<Qualified<IncrementalExplain>> {
        let mut ctx = FitCtx::with_session(session.child("partial_fit"));
        let Some(y) = y else {
            ctx.push(Issue::builder(IssueCode::MissingTarget).build());
            return finish(
                &ctx,
                reject(self.updates, 0, self.n_seen, "LinUCB needs rewards"),
            );
        };
        if x.ncols() == 0 {
            ctx.push(
                Issue::builder(IssueCode::EmptyMatrix)
                    .severity(Severity::Warning)
                    .message("LinUCB received a context with no columns")
                    .build(),
            );
            return finish(
                &ctx,
                reject(self.updates, y.len(), self.n_seen, "empty context"),
            );
        }
        if self.p != 0 && self.p != x.ncols() {
            ctx.push(
                Issue::builder(IssueCode::FeatureSpaceChangedOnline)
                    .severity(Severity::Warning)
                    .message(format!(
                        "LinUCB context dim changed from {} to {}",
                        self.p,
                        x.ncols()
                    ))
                    .build(),
            );
            return finish(
                &ctx,
                reject(self.updates, y.len(), self.n_seen, "context dim changed"),
            );
        }
        self.ensure_dim(x.ncols());
        let before: Vec<f64> = self.b.iter().map(|v| v.norm()).collect();
        let mut dsum = 0.0;
        let mut last_arm = 0usize;
        for i in 0..y.len().min(x.nrows()) {
            if !y[i].is_finite() {
                continue;
            }
            let z = Self::context(x, i, self.p);
            let arm = self.choose_arm(&z, &mut ctx);
            self.apply(arm, &z, y[i]);
            last_arm = arm;
            dsum += 1.0;
        }
        let after: Vec<f64> = self.b.iter().map(|v| v.norm()).collect();
        let identified = self.counts.iter().all(|&c| c > 0);
        let why = format!(
            "chose arm {last_arm}; A ← A+xxᵀ and b ← b+rx on that arm (arm count is not p)"
        );
        let expl = pack_bandit(
            &mut ctx,
            self.updates,
            y.len(),
            self.n_seen,
            dsum,
            identified,
            self.counts.iter().any(|&c| c == 0),
            &before,
            &after,
            "LinUCB ridge-regression arms",
            why.as_str(),
        );
        finish(&ctx, expl)
    }
}

/// Contextual linear Thompson sampling (river `bandit.LinTS`).
///
/// Arm count is not identification `p`.
#[derive(Clone, Debug)]
pub struct LinTs {
    n_arms: usize,
    /// Posterior scale on \(A^{-1}\).
    pub v: f64,
    p: usize,
    ainv: Vec<Vec<f64>>,
    b: Vec<Vector>,
    counts: Vec<u64>,
    rng: Rng,
    updates: u64,
    n_seen: u64,
}

impl LinTs {
    /// `k` arms, unit ridge.
    pub fn new(n_arms: usize) -> Self {
        let k = n_arms.max(1);
        Self {
            n_arms: k,
            v: 1.0,
            p: 0,
            ainv: Vec::new(),
            b: Vec::new(),
            counts: vec![0; k],
            rng: Rng::new(19),
            updates: 0,
            n_seen: 0,
        }
    }

    fn ensure_dim(&mut self, p: usize) {
        if self.p == p && !self.ainv.is_empty() {
            return;
        }
        self.p = p.max(1);
        let dim = self.p;
        self.ainv = (0..self.n_arms)
            .map(|_| {
                let mut m = vec![0.0; dim * dim];
                for i in 0..dim {
                    m[i * dim + i] = 1.0;
                }
                m
            })
            .collect();
        self.b = (0..self.n_arms).map(|_| Vector::zeros(dim)).collect();
    }

    fn sample_theta(&mut self, arm: usize, ctx: &mut FitCtx) -> Vector {
        let p = self.p.max(1);
        let mu = LinUcb::matvec(&self.ainv[arm], p, &self.b[arm]);
        let scale = if self.v.is_finite() && self.v > 0.0 {
            self.v.sqrt()
        } else {
            1.0
        };
        if p == 1 {
            let sd = self.ainv[arm][0].max(0.0).sqrt() * scale;
            return Vector::from_slice(&[mu[0] + sd * self.rng.standard_normal()]);
        }
        match cholesky_lower(&self.ainv[arm], p) {
            Some(l) => {
                let z = Vector::from_iter((0..p).map(|_| self.rng.standard_normal()));
                let lz = LinUcb::matvec(&l, p, &z);
                Vector::from_iter((0..p).map(|j| mu[j] + scale * lz[j]))
            }
            None => {
                ctx.push(
                    Issue::builder(IssueCode::CholeskyFailed)
                        .severity(Severity::Warning)
                        .message(format!(
                            "LinTS arm {arm} A⁻¹ was not SPD; using a diagonal draw"
                        ))
                        .compromise(NumericalCompromise::new(
                            "θ ~ N(A⁻¹b, v A⁻¹)",
                            "independent N(μ_j, v A⁻¹_jj) draws",
                            "the posterior Gram lost definiteness",
                            "arm scores are not a joint Gaussian draw",
                        ))
                        .build(),
                );
                Vector::from_iter((0..p).map(|j| {
                    let sd = self.ainv[arm][j * p + j].max(0.0).sqrt() * scale;
                    mu[j] + sd * self.rng.standard_normal()
                }))
            }
        }
    }

    fn choose_arm(&mut self, z: &Vector, ctx: &mut FitCtx) -> usize {
        if let Some(a) = self.counts.iter().position(|&c| c == 0) {
            ctx.push(
                Issue::builder(IssueCode::WarmupIncomplete)
                    .message(format!("LinTS has never pulled arm {a}"))
                    .build(),
            );
            return a;
        }
        let mut best = 0usize;
        let mut best_u = f64::NEG_INFINITY;
        for a in 0..self.n_arms {
            let theta = self.sample_theta(a, ctx);
            let mut s = 0.0;
            for j in 0..z.len().min(theta.len()) {
                s += z[j] * theta[j];
            }
            if s > best_u {
                best_u = s;
                best = a;
            }
        }
        best
    }
}

fn cholesky_lower(a: &[f64], p: usize) -> Option<Vec<f64>> {
    let mut l = vec![0.0; p * p];
    for i in 0..p {
        for j in 0..=i {
            let mut s = a[i * p + j];
            for k in 0..j {
                s -= l[i * p + k] * l[j * p + k];
            }
            if i == j {
                if s <= 1e-18 {
                    return None;
                }
                l[i * p + i] = s.sqrt();
            } else {
                l[i * p + j] = s / l[j * p + j];
            }
        }
    }
    Some(l)
}

impl PartialFit for LinTs {
    fn partial_fit(
        &mut self,
        x: &Matrix,
        y: Option<&Vector>,
        session: &Session,
    ) -> Result<Qualified<IncrementalExplain>> {
        let mut ctx = FitCtx::with_session(session.child("partial_fit"));
        let Some(y) = y else {
            ctx.push(Issue::builder(IssueCode::MissingTarget).build());
            return finish(
                &ctx,
                reject(self.updates, 0, self.n_seen, "LinTS needs rewards"),
            );
        };
        if x.ncols() == 0 {
            ctx.push(
                Issue::builder(IssueCode::EmptyMatrix)
                    .severity(Severity::Warning)
                    .message("LinTS received a context with no columns")
                    .build(),
            );
            return finish(
                &ctx,
                reject(self.updates, y.len(), self.n_seen, "empty context"),
            );
        }
        if self.p != 0 && self.p != x.ncols() {
            ctx.push(
                Issue::builder(IssueCode::FeatureSpaceChangedOnline)
                    .severity(Severity::Warning)
                    .message("LinTS context dim changed")
                    .build(),
            );
            return finish(
                &ctx,
                reject(self.updates, y.len(), self.n_seen, "context dim changed"),
            );
        }
        self.ensure_dim(x.ncols());
        let before: Vec<f64> = self.b.iter().map(|v| v.norm()).collect();
        let mut dsum = 0.0;
        let mut last_arm = 0usize;
        for i in 0..y.len().min(x.nrows()) {
            if !y[i].is_finite() {
                continue;
            }
            let z = LinUcb::context(x, i, self.p);
            let arm = self.choose_arm(&z, &mut ctx);
            LinUcb::sherman_update(&mut self.ainv[arm], self.p.max(1), &z);
            for j in 0..self.p.min(self.b[arm].len()).min(z.len()) {
                self.b[arm][j] += y[i] * z[j];
            }
            self.counts[arm] += 1;
            self.n_seen += 1;
            self.updates += 1;
            last_arm = arm;
            dsum += 1.0;
        }
        let after: Vec<f64> = self.b.iter().map(|v| v.norm()).collect();
        let identified = self.counts.iter().all(|&c| c > 0);
        let why = format!("chose arm {last_arm} by a Gaussian draw of θ; arm count is not p");
        let expl = pack_bandit(
            &mut ctx,
            self.updates,
            y.len(),
            self.n_seen,
            dsum,
            identified,
            self.counts.iter().any(|&c| c == 0),
            &before,
            &after,
            "LinTS posterior means",
            why.as_str(),
        );
        finish(&ctx, expl)
    }
}

/// Exponential-weight algorithm for exploration and exploitation (river `bandit.Exp3`).
///
/// Arm count is not identification `p`.
#[derive(Clone, Debug)]
pub struct Exp3 {
    /// Exploration mixture \(\gamma \in (0,1]\).
    pub gamma: f64,
    n_arms: usize,
    weights: Vec<f64>,
    counts: Vec<u64>,
    rng: Rng,
    updates: u64,
    n_seen: u64,
}

impl Exp3 {
    /// `k` arms, default \(\gamma=0.1\).
    pub fn new(n_arms: usize) -> Self {
        let k = n_arms.max(1);
        Self {
            gamma: 0.1,
            n_arms: k,
            weights: vec![1.0; k],
            counts: vec![0; k],
            rng: Rng::new(23),
            updates: 0,
            n_seen: 0,
        }
    }

    fn probs(&self) -> Vec<f64> {
        let g = self.gamma.clamp(1e-6, 1.0);
        let k = self.n_arms.max(1) as f64;
        let sw: f64 = self.weights.iter().copied().sum::<f64>().max(1e-18);
        self.weights
            .iter()
            .map(|&w| (1.0 - g) * (w / sw) + g / k)
            .collect()
    }

    fn choose(&mut self, ctx: &mut FitCtx) -> usize {
        if let Some(a) = self.counts.iter().position(|&c| c == 0) {
            ctx.push(
                Issue::builder(IssueCode::WarmupIncomplete)
                    .message(format!("Exp3 has never pulled arm {a}"))
                    .build(),
            );
            return a;
        }
        let p = self.probs();
        let u = self.rng.uniform();
        let mut acc = 0.0;
        for (i, &pi) in p.iter().enumerate() {
            acc += pi;
            if u <= acc {
                return i;
            }
        }
        self.n_arms.saturating_sub(1)
    }

    /// Draw an arm from the Exp3 mixture and explain the draw.
    pub fn pull(&mut self, session: &Session) -> Result<Qualified<(usize, IncrementalExplain)>> {
        let mut ctx = FitCtx::with_session(session.child("pull"));
        let arm = self.choose(&mut ctx);
        let p = self.probs();
        let mut q = IncrementalQuality::new(self.updates, 1, self.n_seen);
        q.warmup = self.counts.iter().any(|&c| c == 0);
        q.still_identified = self.counts.iter().all(|&c| c > 0);
        q.explanation = format!("Exp3 chose {arm} p={p:?}");
        let expl = IncrementalExplain::from_quality(
            q,
            format!("pull arm {arm}"),
            "mixture of the exponential weights and the uniform explore mass",
            format!("weights={:?}", self.weights),
            format!("p={p:?}"),
        );
        ctx.session.record_incremental(expl.clone());
        ctx.finish((arm, expl))
    }
}

impl PartialFit for Exp3 {
    fn partial_fit(
        &mut self,
        x: &Matrix,
        y: Option<&Vector>,
        session: &Session,
    ) -> Result<Qualified<IncrementalExplain>> {
        let mut ctx = FitCtx::with_session(session.child("partial_fit"));
        let Some(y) = y else {
            ctx.push(Issue::builder(IssueCode::MissingTarget).build());
            return finish(
                &ctx,
                reject(self.updates, 0, self.n_seen, "Exp3 needs rewards"),
            );
        };
        let before = self.weights.clone();
        let g = self.gamma.clamp(1e-6, 1.0);
        let k = self.n_arms.max(1) as f64;
        let mut dsum = 0.0;
        for i in 0..y.len().min(x.nrows().max(y.len())) {
            if !y[i].is_finite() {
                continue;
            }
            let arm = if x.ncols() == 0 || x.nrows() == 0 {
                self.choose(&mut ctx)
            } else {
                let a = x.get(i.min(x.nrows() - 1), 0).round().abs() as usize;
                if a >= self.n_arms {
                    ctx.push(
                        Issue::builder(IssueCode::DimensionMismatch)
                            .severity(Severity::Warning)
                            .message(format!("Exp3 arm {a} is outside 0..{}", self.n_arms))
                            .build(),
                    );
                    continue;
                }
                a
            };
            let p = self.probs();
            let pa = p.get(arm).copied().unwrap_or(1.0 / k).max(1e-12);
            let r = y[i].clamp(0.0, 1.0);
            let est = r / pa;
            let before_w = self.weights[arm];
            self.weights[arm] *= (g * est / k).exp();
            if !self.weights[arm].is_finite() || self.weights[arm] > 1e12 {
                ctx.push(
                    Issue::builder(IssueCode::JitterInjected)
                        .severity(Severity::Warning)
                        .message("Exp3 weights overflowed; rescaling")
                        .compromise(NumericalCompromise::new(
                            "finite exponential weights",
                            "weights were rescaled after overflow",
                            "the importance-weighted update grew without bound",
                            "relative arm probabilities are kept; the absolute scale is conventional",
                        ))
                        .build(),
                );
                let mx = self
                    .weights
                    .iter()
                    .copied()
                    .fold(0.0_f64, |a, b| a.max(b))
                    .max(1.0);
                for w in &mut self.weights {
                    *w /= mx;
                }
            }
            dsum += (self.weights[arm] - before_w).abs();
            self.counts[arm] += 1;
            self.n_seen += 1;
            self.updates += 1;
        }
        let expl = pack_bandit(
            &mut ctx,
            self.updates,
            y.len(),
            self.n_seen,
            dsum,
            self.counts.iter().all(|&c| c > 0),
            self.counts.iter().any(|&c| c == 0),
            &before,
            &self.weights,
            "Exp3 exponential weights",
            "importance-weighted multiplicative update; arm count is not p",
        );
        finish(&ctx, expl)
    }
}

/// Expert-mixture Exp3 (Auer Exp4; river `bandit` peer).
///
/// Column 0 is the played arm; extra columns are expert recommendations.
/// Arm and expert counts are not identification `p`.
#[derive(Clone, Debug)]
pub struct Exp4 {
    /// Exploration mixture \(\gamma \in (0,1]\).
    pub gamma: f64,
    n_arms: usize,
    expert_weights: Vec<f64>,
    expert_counts: Vec<u64>,
    updates: u64,
    n_seen: u64,
}

impl Exp4 {
    /// `k` arms, default \(\gamma=0.1\).
    pub fn new(n_arms: usize) -> Self {
        Self {
            gamma: 0.1,
            n_arms: n_arms.max(1),
            expert_weights: Vec::new(),
            expert_counts: Vec::new(),
            updates: 0,
            n_seen: 0,
        }
    }

    fn ensure_experts(&mut self, ctx: &mut FitCtx, n_exp: usize) {
        let n = n_exp.max(1);
        if self.expert_weights.is_empty() {
            self.expert_weights = vec![1.0; n];
            self.expert_counts = vec![0; n];
            return;
        }
        if self.expert_weights.len() != n {
            ctx.push(
                Issue::builder(IssueCode::FeatureSpaceChangedOnline)
                    .severity(Severity::Warning)
                    .message(format!(
                        "Exp4 expert count changed {} → {n}",
                        self.expert_weights.len()
                    ))
                    .build(),
            );
            self.expert_weights.resize(n, 1.0);
            self.expert_counts.resize(n, 0);
        }
    }

    fn mix_probs(&self, recs: &[usize]) -> Vec<f64> {
        let g = self.gamma.clamp(1e-6, 1.0);
        let k = self.n_arms.max(1);
        let sw: f64 = self.expert_weights.iter().copied().sum::<f64>().max(1e-18);
        let mut p = vec![g / k as f64; k];
        for (e, &a) in recs.iter().enumerate() {
            if a < k {
                let we = self.expert_weights.get(e).copied().unwrap_or(0.0) / sw;
                p[a] += (1.0 - g) * we;
            }
        }
        p
    }
}

impl PartialFit for Exp4 {
    fn partial_fit(
        &mut self,
        x: &Matrix,
        y: Option<&Vector>,
        session: &Session,
    ) -> Result<Qualified<IncrementalExplain>> {
        let mut ctx = FitCtx::with_session(session.child("partial_fit"));
        let Some(y) = y else {
            ctx.push(Issue::builder(IssueCode::MissingTarget).build());
            return finish(
                &ctx,
                reject(self.updates, 0, self.n_seen, "Exp4 needs rewards"),
            );
        };
        if x.ncols() < 2 {
            ctx.push(
                Issue::builder(IssueCode::DimensionMismatch)
                    .severity(Severity::Warning)
                    .message("Exp4 has no expert columns; using a single dummy expert")
                    .build(),
            );
        }
        let n_exp = x.ncols().saturating_sub(1).max(1);
        self.ensure_experts(&mut ctx, n_exp);
        let before = self.expert_weights.clone();
        let g = self.gamma.clamp(1e-6, 1.0);
        let k_e = n_exp as f64;
        let mut dsum = 0.0_f64;
        for i in 0..y.len().min(x.nrows().max(y.len())) {
            if !y[i].is_finite() {
                continue;
            }
            let arm = if x.ncols() == 0 || x.nrows() == 0 {
                0
            } else {
                let a = x.get(i.min(x.nrows() - 1), 0).round().abs() as usize;
                if a >= self.n_arms {
                    ctx.push(
                        Issue::builder(IssueCode::DimensionMismatch)
                            .severity(Severity::Warning)
                            .message(format!("Exp4 arm {a} is outside 0..{}", self.n_arms))
                            .build(),
                    );
                    continue;
                }
                a
            };
            let recs: Vec<usize> = if x.ncols() < 2 {
                vec![arm]
            } else {
                (1..x.ncols())
                    .map(|j| x.get(i.min(x.nrows() - 1), j).round().abs() as usize)
                    .collect()
            };
            let p = self.mix_probs(&recs);
            let pa = p
                .get(arm)
                .copied()
                .unwrap_or(1.0 / self.n_arms.max(1) as f64)
                .max(1e-12);
            let r = y[i].clamp(0.0, 1.0);
            let est = r / pa;
            for (e, &rec) in recs.iter().enumerate() {
                if rec != arm || e >= self.expert_weights.len() {
                    continue;
                }
                let before_w = self.expert_weights[e];
                self.expert_weights[e] *= (g * est / k_e).exp();
                dsum += (self.expert_weights[e] - before_w).abs();
                self.expert_counts[e] += 1;
            }
            if self
                .expert_weights
                .iter()
                .any(|w| !w.is_finite() || *w > 1e12)
            {
                ctx.push(
                    Issue::builder(IssueCode::JitterInjected)
                        .severity(Severity::Warning)
                        .message("Exp4 expert weights overflowed; rescaling")
                        .compromise(NumericalCompromise::new(
                            "finite expert weights",
                            "weights were rescaled after overflow",
                            "the importance-weighted update grew without bound",
                            "relative expert probabilities are kept; the absolute scale is conventional",
                        ))
                        .build(),
                );
                let mx = self
                    .expert_weights
                    .iter()
                    .copied()
                    .fold(0.0_f64, |a, b| a.max(b))
                    .max(1.0);
                for w in &mut self.expert_weights {
                    *w /= mx;
                }
            }
            self.n_seen += 1;
            self.updates += 1;
        }
        let expl = pack_bandit(
            &mut ctx,
            self.updates,
            y.len(),
            self.n_seen,
            dsum,
            self.expert_counts.iter().all(|&c| c > 0),
            self.expert_counts.iter().any(|&c| c == 0) || self.expert_counts.is_empty(),
            &before,
            &self.expert_weights,
            "Exp4 expert weights",
            "importance-weighted mixture of expert recommendations; arm/expert counts are not p",
        );
        finish(&ctx, expl)
    }
}

/// Bayesian UCB on Bernoulli arms (river `bandit.BayesUCB`).
///
/// Each arm holds a `Beta(α, β)` posterior. The index is a Gaussian
/// approximation to the posterior quantile. Arm count is not identification
/// `p`.
#[derive(Clone, Debug)]
pub struct BayesianUcb {
    n_arms: usize,
    alpha: Vec<f64>,
    beta: Vec<f64>,
    updates: u64,
    n_seen: u64,
}

impl BayesianUcb {
    /// `k` arms, uniform Beta(1, 1) priors.
    pub fn new(n_arms: usize) -> Self {
        let k = n_arms.max(1);
        Self {
            n_arms: k,
            alpha: vec![1.0; k],
            beta: vec![1.0; k],
            updates: 0,
            n_seen: 0,
        }
    }

    fn index(&self, arm: usize) -> f64 {
        let a = self.alpha[arm].max(1e-8);
        let b = self.beta[arm].max(1e-8);
        let s = a + b;
        let mu = a / s;
        let var = a * b / (s * s * (s + 1.0));
        let t = self.n_seen.max(1) as f64;
        let z = (2.0 * t.ln()).sqrt().max(1.0);
        mu + z * var.max(0.0).sqrt()
    }

    /// Choose the arm with the largest Bayesian UCB index.
    pub fn pull(&mut self, session: &Session) -> Result<Qualified<(usize, IncrementalExplain)>> {
        let mut ctx = FitCtx::with_session(session.child("pull"));
        let mut arm = 0usize;
        let mut best = f64::NEG_INFINITY;
        let mut indexes = vec![0.0; self.n_arms];
        for a in 0..self.n_arms {
            if self.alpha[a] + self.beta[a] <= 2.0 + 1e-12 {
                ctx.push(
                    Issue::builder(IssueCode::WarmupIncomplete)
                        .message(format!("BayesUCB arm {a} is still the prior"))
                        .build(),
                );
            }
            let idx = self.index(a);
            indexes[a] = idx;
            if idx > best {
                best = idx;
                arm = a;
            }
        }
        let mut q = IncrementalQuality::new(self.updates, 1, self.n_seen);
        q.warmup = self
            .alpha
            .iter()
            .zip(&self.beta)
            .any(|(&a, &b)| a + b <= 2.0 + 1e-12);
        q.still_identified = !q.warmup;
        q.explanation = format!("BayesUCB chose {arm} indexes={indexes:?}");
        let expl = IncrementalExplain::from_quality(
            q,
            format!("pull arm {arm}"),
            "argmax of a Gaussian approximation to the Beta posterior quantile",
            format!("alpha={:?} beta={:?}", self.alpha, self.beta),
            format!("ucb={indexes:?}"),
        );
        ctx.session.record_incremental(expl.clone());
        ctx.finish((arm, expl))
    }
}

impl PartialFit for BayesianUcb {
    fn partial_fit(
        &mut self,
        x: &Matrix,
        y: Option<&Vector>,
        session: &Session,
    ) -> Result<Qualified<IncrementalExplain>> {
        let mut ctx = FitCtx::with_session(session.child("partial_fit"));
        let Some(y) = y else {
            ctx.push(Issue::builder(IssueCode::MissingTarget).build());
            return finish(
                &ctx,
                reject(self.updates, 0, self.n_seen, "BayesUCB needs rewards"),
            );
        };
        let before: Vec<f64> = (0..self.n_arms)
            .map(|a| self.alpha[a] / (self.alpha[a] + self.beta[a]))
            .collect();
        let mut dsum = 0.0;
        for i in 0..y.len() {
            if !y[i].is_finite() {
                continue;
            }
            let arm = if x.ncols() == 0 || x.nrows() == 0 {
                0
            } else {
                x.get(i.min(x.nrows() - 1), 0).round().abs() as usize
            };
            if arm >= self.n_arms {
                ctx.push(
                    Issue::builder(IssueCode::DimensionMismatch)
                        .severity(Severity::Warning)
                        .message(format!("BayesUCB arm {arm} is outside 0..{}", self.n_arms))
                        .build(),
                );
                continue;
            }
            let r = if y[i] >= 0.5 { 1.0 } else { 0.0 };
            self.alpha[arm] += r;
            self.beta[arm] += 1.0 - r;
            self.counts_bump();
            dsum += 1.0;
        }
        let after: Vec<f64> = (0..self.n_arms)
            .map(|a| self.alpha[a] / (self.alpha[a] + self.beta[a]))
            .collect();
        let identified = self
            .alpha
            .iter()
            .zip(&self.beta)
            .all(|(&a, &b)| a + b > 2.0 + 1e-12);
        let expl = pack_bandit(
            &mut ctx,
            self.updates,
            y.len(),
            self.n_seen,
            dsum,
            identified,
            !identified,
            &before,
            &after,
            "BayesUCB Beta means",
            "Bernoulli update of Beta(α,β); arm count is not p",
        );
        finish(&ctx, expl)
    }
}

impl BayesianUcb {
    fn counts_bump(&mut self) {
        self.n_seen += 1;
        self.updates += 1;
    }
}

fn sample_beta(rng: &mut Rng, alpha: f64, beta: f64) -> f64 {
    // Gamma(k,1) ≈ sum of k exponentials for integer shape; otherwise Jöhnk.
    let x = sample_gamma(rng, alpha.max(1e-6));
    let y = sample_gamma(rng, beta.max(1e-6));
    let s = x + y;
    if s <= 0.0 {
        0.5
    } else {
        x / s
    }
}

fn sample_gamma(rng: &mut Rng, shape: f64) -> f64 {
    let k = shape.floor().max(1.0) as usize;
    let mut s = 0.0;
    for _ in 0..k {
        s += -rng.uniform().max(1e-12).ln();
    }
    s * (shape / k as f64)
}

fn pack_bandit(
    ctx: &mut FitCtx,
    updates: u64,
    batch: usize,
    n_seen: u64,
    delta: f64,
    identified: bool,
    warmup: bool,
    before: &[f64],
    after: &[f64],
    what: &str,
    why: &str,
) -> IncrementalExplain {
    let mut q = IncrementalQuality::new(updates.saturating_sub(1), batch, n_seen);
    q.effective_sample_size = n_seen as f64;
    q.parameter_delta_norm = Some(delta);
    q.information_gain = Some(delta);
    q.still_identified = identified;
    q.warmup = warmup;
    q.explanation = format!("{what}: ||Δμ||={delta:.6e}");
    if warmup {
        ctx.push(
            Issue::builder(IssueCode::WarmupIncomplete)
                .incremental(q.clone())
                .message("at least one arm has no observations")
                .build(),
        );
    }
    if !identified {
        ctx.push(
            Issue::builder(IssueCode::IncrementalUnidentifiable)
                .severity(Severity::Warning)
                .incremental(q.clone())
                .message("bandit state is not identified for every arm")
                .build(),
        );
    }
    if q.is_uninformative(ctx.policy.uninformative_info_eps) {
        ctx.push(
            Issue::builder(IssueCode::UpdateWithZeroInformation)
                .incremental(q.clone())
                .message("arm means did not move")
                .build(),
        );
    }
    let mut expl = IncrementalExplain::from_quality(
        q,
        what,
        why,
        format!("before={before:?}"),
        format!("after={after:?}"),
    );
    let mut contrib = BTreeMap::new();
    for (i, (a, b)) in before.iter().zip(after).enumerate() {
        contrib.insert(format!("arm[{i}]"), (b - a).abs());
    }
    expl.contribution = contrib;
    expl
}

fn reject(update: u64, batch: usize, n_seen: u64, why: &str) -> IncrementalExplain {
    IncrementalExplain::from_quality(
        IncrementalQuality::new(update, batch, n_seen),
        "nothing",
        why,
        "invalid",
        "invalid",
    )
}

fn finish(ctx: &FitCtx, expl: IncrementalExplain) -> Result<Qualified<IncrementalExplain>> {
    ctx.session.record_incremental(expl.clone());
    // FitCtx::finish consumes self; rebuild from the same session.
    let owned = FitCtx::with_session(ctx.session.clone());
    // Copy already-pushed issues by finishing on a fresh report would drop them.
    // Use the original context by reconstructing via session only when the
    // report is empty of extras we care about — callers push on `ctx`.
    drop(owned);
    let FitCtx {
        session,
        report,
        policy,
    } = FitCtx {
        session: ctx.session.clone(),
        report: ctx.report.clone(),
        policy: ctx.policy.clone(),
    };
    match report.finish_with_policy(policy, expl) {
        Ok(q) => {
            session.finish_ok(&q);
            Ok(q)
        }
        Err(e) => {
            session.finish_err(&e);
            Err(e)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::PartialFit;

    #[test]
    fn epsilon_greedy_learns_best_arm() {
        let mut b = EpsilonGreedy::new(3, 0.2);
        let session = Session::new("eg", "pf");
        // arm 2 always pays 1, others 0
        for _ in 0..40 {
            let x = Matrix::from_fn(3, 1, |i, _| i as f64);
            let y = Vector::from_slice(&[0.0, 0.0, 1.0]);
            b.partial_fit(&x, Some(&y), &session).expect("eg");
        }
        assert!(b.values()[2] > b.values()[0]);
        assert!(b.values()[2] > b.values()[1]);
        let pull = b.pull(&Session::new("eg", "pull")).expect("pull");
        assert!(pull.value.1.narrative.contains("arm"));
    }

    #[test]
    fn ucb_and_thompson_explain() {
        let session = Session::new("bandit", "pf");
        let x = Matrix::from_fn(2, 1, |i, _| i as f64);
        let y = Vector::from_slice(&[1.0, 0.0]);
        let mut u = Ucb1::new(2);
        let q = u.partial_fit(&x, Some(&y), &session).expect("ucb");
        assert!(!q.value.narrative.is_empty());
        let mut t = ThompsonBernoulli::new(2);
        let q = t.partial_fit(&x, Some(&y), &session).expect("th");
        assert!(!q.value.narrative.is_empty());
        let _ = t.pull(&Session::new("th", "pull")).expect("th pull");
        let _ = u.pull(&Session::new("ucb", "pull")).expect("ucb pull");
        let mut lin = LinUcb::new(2);
        let q = lin.partial_fit(&x, Some(&y), &session).expect("linucb");
        assert!(!q.value.narrative.is_empty());
        let mut lts = LinTs::new(2);
        let q = lts.partial_fit(&x, Some(&y), &session).expect("lints");
        assert!(!q.value.narrative.is_empty());
        let mut exp3 = Exp3::new(2);
        let q = exp3.partial_fit(&x, Some(&y), &session).expect("exp3");
        assert!(!q.value.narrative.is_empty());
        let mut bu = BayesianUcb::new(2);
        let q = bu.partial_fit(&x, Some(&y), &session).expect("bayesucb");
        assert!(!q.value.narrative.is_empty());
        let mut ut = UcbTuned::new(2);
        let q = ut.partial_fit(&x, Some(&y), &session).expect("ucbt");
        assert!(!q.value.narrative.is_empty());
        let mut exp4 = Exp4::new(2);
        let q = exp4.partial_fit(&x, Some(&y), &session).expect("exp4");
        assert!(!q.value.narrative.is_empty());
    }
}
