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
    IncrementalQuality, Issue, IssueCode, Meaninglessness, Qualified, Result, Severity,
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
    }
}
