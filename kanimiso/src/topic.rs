//! Latent Dirichlet allocation (Blei, Ng, Jordan) via batch variational Bayes.
//!
//! The design is a document–term count matrix (`n` documents × `V` terms).
//! Empty documents or a vocabulary of all-zero columns leave topics
//! unidentified. Negative counts abort.

use crate::context::FitCtx;
use crate::data::{Matrix, Vector};
use crate::special::ln_gamma;
use crate::traits::FitUnsupervised;
use crate::validate::inspect_xy;
use ojizou_san::Session;
use signlred::{Issue, IssueCode, Meaninglessness, NumericalCompromise, Qualified, Result};

/// Batch variational LDA.
#[derive(Clone, Debug)]
pub(crate) struct LatentDirichletAllocation {
    /// Topics \(K\).
    pub n_topics: usize,
    /// Document-topic Dirichlet \(\alpha\).
    pub alpha: f64,
    /// Topic-word Dirichlet \(\eta\).
    pub eta: f64,
    /// VB iterations.
    pub max_iter: usize,
    /// PRNG seed for the initial topics.
    pub seed: u64,
}

impl Default for LatentDirichletAllocation {
    fn default() -> Self {
        Self {
            n_topics: 2,
            alpha: 0.1,
            eta: 0.1,
            max_iter: 20,
            seed: 1,
        }
    }
}

impl LatentDirichletAllocation {
    /// `k`-topic LDA.
    pub(crate) fn new(n_topics: usize) -> Self {
        Self {
            n_topics,
            ..Self::default()
        }
    }

    /// Fit alias.
    pub(crate) fn fit(&self, x: &Matrix, session: &Session) -> Result<Qualified<FittedLda>> {
        self.fit_unsupervised(x, session)
    }
}

/// Fitted LDA (variational point).
#[derive(Clone, Debug)]
pub(crate) struct FittedLda {
    /// Topic-word Dirichlet means (`K` × `V`).
    pub components: Matrix,
    /// Document-topic means on the training corpus (`n` × `K`).
    pub doc_topic: Matrix,
    /// Evidence lower bound on the last pass (nats).
    pub bound: f64,
}

fn digamma(mut x: f64) -> f64 {
    // Recurrence to x>6 plus Stirling.
    if x <= 0.0 || !x.is_finite() {
        return f64::NEG_INFINITY;
    }
    let mut acc = 0.0;
    while x < 6.0 {
        acc -= 1.0 / x;
        x += 1.0;
    }
    let inv = 1.0 / x;
    let inv2 = inv * inv;
    acc + x.ln() - 0.5 * inv - inv2 / 12.0 + inv2 * inv2 / 120.0
}

impl FitUnsupervised for LatentDirichletAllocation {
    type Fitted = FittedLda;
    fn fit_unsupervised(&self, x: &Matrix, session: &Session) -> Result<Qualified<FittedLda>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let (n, v) = x.shape();
        let k = self.n_topics.max(1);
        if k > n {
            ctx.push(
                Issue::builder(IssueCode::Overparameterized)
                    .message(format!("K={k} topics on n={n} documents"))
                    .build(),
            );
        }
        if n == 0 || v == 0 {
            return ctx.finish(FittedLda {
                components: Matrix::zeros(k, v),
                doc_topic: Matrix::zeros(n, k),
                bound: f64::NAN,
            });
        }
        let mut neg = false;
        let mut empty_docs = 0usize;
        let mut empty_terms = 0usize;
        for i in 0..n {
            let mut s = 0.0;
            for j in 0..v {
                let c = x.get(i, j);
                if c < 0.0 {
                    neg = true;
                }
                s += c;
            }
            if s <= 0.0 {
                empty_docs += 1;
            }
        }
        for j in 0..v {
            let mut s = 0.0;
            for i in 0..n {
                s += x.get(i, j);
            }
            if s <= 0.0 {
                empty_terms += 1;
            }
        }
        if neg {
            ctx.push(
                Issue::builder(IssueCode::NonPositiveSeries)
                    .message("LDA saw a negative count")
                    .build(),
            );
        }
        if empty_docs > 0 {
            ctx.push(
                Issue::builder(IssueCode::EmptyCluster)
                    .message(format!("{empty_docs} documents have zero tokens"))
                    .meaninglessness(Meaninglessness::new(
                        "document-topic Dirichlet",
                        "a zero-length document has no likelihood and γ is the prior",
                        signlred::InterpretiveValue::Misleading,
                        "drop empty documents before interpreting those θ rows",
                    ))
                    .build(),
            );
        }
        if empty_terms == v {
            ctx.push(
                Issue::builder(IssueCode::UnidentifiedModel)
                    .message("every term column is zero; topics are unidentified")
                    .build(),
            );
            return ctx.finish(FittedLda {
                components: Matrix::zeros(k, v),
                doc_topic: Matrix::zeros(n, k),
                bound: f64::NAN,
            });
        }
        let mut rng = crate::rng::Rng::new(self.seed);
        let mut lambda = Matrix::from_fn(k, v, |_, _| self.eta.max(1e-8) + 0.1 * rng.uniform());
        let mut gamma = Matrix::from_fn(n, k, |_, _| self.alpha.max(1e-8));
        let mut bound = f64::NEG_INFINITY;
        for it in 0..self.max_iter.max(1) {
            let mut psi_lambda = Matrix::zeros(k, v);
            let mut psi_lambda_sum = Vector::zeros(k);
            for t in 0..k {
                let mut s = 0.0;
                for w in 0..v {
                    s += lambda.get(t, w);
                }
                psi_lambda_sum[t] = digamma(s.max(1e-12));
                for w in 0..v {
                    psi_lambda.set(t, w, digamma(lambda.get(t, w).max(1e-12)));
                }
            }
            let mut new_lambda = Matrix::from_fn(k, v, |_, _| self.eta.max(1e-8));
            for d in 0..n {
                let mut g =
                    Vector::from_iter((0..k).map(|t| gamma.get(d, t).max(self.alpha.max(1e-8))));
                for _inner in 0..8 {
                    let gsum = g.as_slice().iter().sum::<f64>().max(1e-12);
                    let psi_gs = digamma(gsum);
                    let mut ng = Vector::filled(k, self.alpha.max(1e-8));
                    for w in 0..v {
                        let cnt = x.get(d, w);
                        if cnt <= 0.0 {
                            continue;
                        }
                        let mut log_phi = vec![0.0; k];
                        for t in 0..k {
                            log_phi[t] =
                                digamma(g[t]) - psi_gs + psi_lambda.get(t, w) - psi_lambda_sum[t];
                        }
                        let m = log_phi.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                        let mut z = 0.0;
                        for t in 0..k {
                            log_phi[t] = (log_phi[t] - m).exp();
                            z += log_phi[t];
                        }
                        z = z.max(1e-15);
                        for t in 0..k {
                            let phi = log_phi[t] / z;
                            ng[t] += cnt * phi;
                            new_lambda.set(t, w, new_lambda.get(t, w) + cnt * phi);
                        }
                    }
                    g = ng;
                }
                for t in 0..k {
                    gamma.set(d, t, g[t]);
                }
            }
            lambda = new_lambda;
            // ELBO fragment: E[log p(w|z,β)] using mean β.
            let mut elbo = 0.0;
            for t in 0..k {
                let mut s = 0.0;
                for w in 0..v {
                    s += lambda.get(t, w);
                }
                elbo += ln_gamma(self.eta.max(1e-8) * v as f64) - ln_gamma(s.max(1e-12));
            }
            bound = elbo;
            ctx.session.step(it as u64, -bound, None);
        }
        ctx.session.converged("LDA batch VB", self.max_iter as u64);
        let components = Matrix::from_fn(k, v, |t, w| {
            let mut s = 0.0;
            for u in 0..v {
                s += lambda.get(t, u);
            }
            lambda.get(t, w) / s.max(1e-12)
        });
        let doc_topic = Matrix::from_fn(n, k, |d, t| {
            let mut s = 0.0;
            for u in 0..k {
                s += gamma.get(d, u);
            }
            gamma.get(d, t) / s.max(1e-12)
        });
        ctx.push(
            Issue::builder(IssueCode::LocalMinimumUnstable)
                .severity(signlred::Severity::Advisory)
                .message("LDA VB is locally optimal; topics are identified only up to permutation")
                .compromise(NumericalCompromise::new(
                    "exact posterior over (θ, β, z)",
                    "mean-field batch variational Bayes",
                    "the evidence is not jointly concave",
                    "do not treat a single run as the unique topic decomposition",
                ))
                .build(),
        );
        ctx.finish(FittedLda {
            components,
            doc_topic,
            bound,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lda_two_topics_on_block_counts() {
        let x = Matrix::from_fn(8, 6, |i, j| {
            if i < 4 {
                if j < 3 {
                    4.0
                } else {
                    0.0
                }
            } else if j >= 3 {
                4.0
            } else {
                0.0
            }
        });
        let q = LatentDirichletAllocation::new(2)
            .fit(&x, &Session::new("lda", "fit"))
            .expect("lda");
        assert_eq!(q.value.components.shape(), (2, 6));
        assert_eq!(q.value.doc_topic.shape(), (8, 2));
        // Each topic should concentrate on one vocabulary block.
        let mut mass_left = [0.0; 2];
        for t in 0..2 {
            for j in 0..3 {
                mass_left[t] += q.value.components.get(t, j);
            }
        }
        mass_left.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!(mass_left[1] > 0.6, "{mass_left:?}");
    }
}
