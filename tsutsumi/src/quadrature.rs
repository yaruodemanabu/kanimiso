//! Gaussian expectation quadrature using the Golub–Welsch Jacobi matrix.

use crate::{FitCtx, Matrix};
use ojizou_san::Session;
use signlred::{Issue, IssueCode, Policy, Qualified, Result};

/// Nodes and positive weights for expectations under a standard normal law.
#[derive(Clone, Debug)]
pub struct NormalQuadrature {
    /// Increasing standard-normal nodes (not physicists' Hermite coordinates).
    pub nodes: Vec<f64>,
    /// Corresponding weights, summing to one up to floating-point error.
    pub weights: Vec<f64>,
}

impl NormalQuadrature {
    /// Construct an order-`points` Gaussian rule for 2 through 128 points.
    ///
    /// The Jacobi off-diagonal is sqrt(k), and weights are squared first
    /// components of normalized eigenvectors. Polynomials through degree
    /// `2*points-1` are integrated exactly in real arithmetic.
    pub fn new(points: usize, policy: &Policy, session: &Session) -> Result<Qualified<Self>> {
        let mut ctx = FitCtx::with_session(session.clone());
        ctx.policy = policy.clone();
        if !(2..=128).contains(&points)
            || !policy.probability_sum_tol.is_finite()
            || policy.probability_sum_tol <= 0.0
        {
            ctx.push(
                Issue::builder(IssueCode::InvalidParameter)
                    .message("normal quadrature requires 2..=128 points and a positive finite probability_sum_tol")
                    .build(),
            );
            return Err(ctx.finish_failure());
        }
        let jacobi = Matrix::from_fn(points, points, |i, j| {
            if i.abs_diff(j) == 1 {
                (i.max(j) as f64).sqrt()
            } else {
                0.0
            }
        });
        let Some((nodes, vectors)) =
            crate::linalg::symmetric_eigen(&mut ctx.report, jacobi.inner(), policy)
        else {
            return Err(ctx.finish_failure());
        };
        let mut pairs: Vec<_> = nodes
            .into_iter()
            .enumerate()
            .map(|(j, node)| (node, vectors[(0, j)].powi(2)))
            .collect();
        pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
        let weights: Vec<_> = pairs.iter().map(|v| v.1).collect();
        let total: f64 = weights.iter().sum();
        if weights.iter().any(|w| !w.is_finite() || *w <= 0.0)
            || (total - 1.0).abs() > policy.probability_sum_tol
        {
            ctx.push(
                Issue::builder(IssueCode::NonFiniteOutput)
                    .message("quadrature eigenweights are not a finite normalized positive rule")
                    .build(),
            );
            return Err(ctx.finish_failure());
        }
        ctx.finish(Self {
            nodes: pairs.into_iter().map(|v| v.0).collect(),
            weights,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn normal_rule_matches_independent_exact_moments() {
        let rule =
            NormalQuadrature::new(8, &Policy::default(), &Session::new("normal_quad", "test"))
                .unwrap()
                .value;
        let mut worst: f64 = 0.0;
        for (power, expected) in [
            (0, 1.0),
            (1, 0.0),
            (2, 1.0),
            (3, 0.0),
            (4, 3.0),
            (6, 15.0),
            (8, 105.0),
        ] {
            let actual: f64 = rule
                .nodes
                .iter()
                .zip(&rule.weights)
                .map(|(&x, &w)| w * x.powi(power))
                .sum();
            worst = worst.max((actual - expected).abs() / expected.max(1.0));
            assert!(
                // Measured 3.08e-15 scaled on 2026-09-05, fourfold margin.
                (actual - expected).abs() <= 1.24e-14 * expected.max(1.0),
                "moment {power}: {actual} vs {expected}"
            );
        }
        eprintln!("normal moment maximum scaled error: {worst:e}");
    }
    #[test]
    fn normal_nodes_and_weights_have_reflection_symmetry() {
        let rule =
            NormalQuadrature::new(16, &Policy::default(), &Session::new("normal_quad", "test"))
                .unwrap()
                .value;
        for i in 0..16 {
            assert!((rule.nodes[i] + rule.nodes[15 - i]).abs() <= 64.0 * f64::EPSILON);
            assert!((rule.weights[i] - rule.weights[15 - i]).abs() <= 64.0 * f64::EPSILON);
        }
    }
}
