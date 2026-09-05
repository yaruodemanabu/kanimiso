//! Random-intercept Gaussian and generalized linear mixed models.

use crate::annotation::{basic_annotations, context, note, stopped};
use crate::regression::{design_matrix, failure, validate_prediction, validate_training};
use crate::{
    AnalysisOptions, Annotation, Family, GeneralizedLinearModel, Matrix, Qualified, Result,
    Session, Topic, Vector,
};
use signlred::{Issue, IssueCode, NumericalCompromise, Report};
use std::collections::BTreeMap;
use tsutsumi::linalg::least_squares_with_diagnostics;
use tsutsumi::optimize::{NelderMead, OptimizationTermination};
use tsutsumi::quadrature::NormalQuadrature;

/// Likelihood criterion for Gaussian variance-component fitting.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Likelihood {
    /// Marginal maximum likelihood.
    Maximum,
    /// Restricted maximum likelihood, Gaussian models only.
    #[default]
    Restricted,
}

/// Random-intercept LMM or GLMM with the family chosen at runtime.
#[derive(Clone, Debug)]
pub struct MixedModel {
    /// Gaussian, Bernoulli-logit, or Poisson-log observations.
    pub family: Family,
    /// ML or Gaussian REML.
    pub likelihood: Likelihood,
    /// Normal quadrature order for GLMM integration, 2..=64; doubled for checking.
    pub quadrature_points: usize,
    /// Intercept, optimizer iterations, and numerical policy.
    pub options: AnalysisOptions,
}

impl Default for MixedModel {
    fn default() -> Self {
        Self {
            family: Family::Gaussian,
            likelihood: Likelihood::Restricted,
            quadrature_points: 32,
            options: AnalysisOptions::default(),
        }
    }
}

/// Gaussian random-intercept model by default.
pub type LinearMixedModel = MixedModel;
/// Use `MixedModel::generalized` for a canonical non-Gaussian random-intercept model.
pub type GeneralizedMixedModel = MixedModel;

/// Fitted random-intercept model with explicit population/group predictions.
#[derive(Clone, Debug)]
pub struct FittedMixed {
    /// Observation family.
    pub family: Family,
    /// Fixed coefficients including an added intercept when requested.
    pub beta: Vector,
    /// Gaussian residual variance; not a free parameter for Bernoulli/Poisson.
    pub residual_variance: Option<f64>,
    /// Gaussian random-intercept variance.
    pub random_intercept_variance: f64,
    /// Sorted group identifiers and BLUP/posterior mean intercepts.
    pub random_effects: Vec<(u64, f64)>,
    /// Maximized marginal or restricted log likelihood.
    pub log_likelihood: f64,
    /// Likelihood convention used.
    pub likelihood: Likelihood,
    /// Absolute likelihood change when the GLMM quadrature order is doubled.
    pub quadrature_difference: Option<f64>,
    /// Statistical, approximation, and prediction notes.
    pub annotations: Vec<Annotation>,
    features: usize,
    options: AnalysisOptions,
    quadrature: Option<NormalQuadrature>,
}

impl MixedModel {
    /// Construct a marginal-ML GLMM; Gaussian remains a valid ML choice.
    pub fn generalized(family: Family) -> Self {
        Self {
            family,
            likelihood: Likelihood::Maximum,
            ..Self::default()
        }
    }

    /// Fit independent groups with one Gaussian random intercept per group.
    ///
    /// Group identifiers are integers and never rounded from floating point.
    /// Random slopes and crossed/nested variance components are not inferred.
    pub fn fit(
        &self,
        x: &Matrix,
        y: &Vector,
        groups: &[u64],
        session: &Session,
    ) -> Result<Qualified<FittedMixed>> {
        let mut ctx = context(session, &self.options);
        validate_training(&mut ctx, x, y, self.family, self.options.fit_intercept);
        if groups.len() != y.len() {
            ctx.push(
                Issue::builder(IssueCode::DimensionMismatch)
                    .message("groups length differs from response length")
                    .build(),
            );
        }
        if self.family != Family::Gaussian && self.likelihood == Likelihood::Restricted {
            ctx.push(
                Issue::builder(IssueCode::InvalidParameter)
                    .message("REML is implemented for Gaussian LMM only; GLMM requires marginal ML")
                    .build(),
            );
        }
        if !(2..=64).contains(&self.quadrature_points) {
            ctx.push(
                Issue::builder(IssueCode::InvalidParameter)
                    .message("GLMM quadrature_points must be 2..=64")
                    .build(),
            );
        }
        if stopped(&ctx) {
            return Err(ctx.finish_failure());
        }
        let mut grouped: BTreeMap<u64, Vec<usize>> = BTreeMap::new();
        for (row, &group) in groups.iter().enumerate() {
            grouped.entry(group).or_default().push(row);
        }
        if grouped.len() < 2 || grouped.values().all(|rows| rows.len() == 1) {
            ctx.push(Issue::builder(IssueCode::UnidentifiedModel).message("random-intercept variance requires multiple groups and within-group replication").build());
            return Err(ctx.finish_failure());
        }
        let mut annotations = basic_annotations();
        annotations.push(Annotation::new(Topic::Assumptions,
            format!("{} independent groups, exchangeable Gaussian random intercepts, and correctly specified fixed effects; random intercepts are independent of included covariates.",grouped.len()),
            "Inspect within-group replication and covariate/random-effect dependence. Few groups, informative cluster sizes, or omitted random slopes can invalidate interpretation."));
        annotations.push(Annotation::new(Topic::Inference,
            "Variance components are constrained at zero. Ordinary chi-square likelihood-ratio and Wald approximations are not generally valid at that boundary; fixed-effect/variance p-values are withheld.",
            "Use a design-appropriate parametric bootstrap or a verified profile-likelihood analysis; compare REML likelihoods only for identical fixed-effect designs."));
        let design = design_matrix(x, self.options.fit_intercept);
        let optimizer = NelderMead {
            max_iterations: self
                .options
                .max_iterations
                .saturating_mul(design.ncols() + 2),
            policy: self.options.policy.clone(),
            ..NelderMead::default()
        };
        if self.family == Family::Gaussian {
            let objective = |parameters: &[f64]| {
                gaussian_profile(
                    &design,
                    y,
                    &grouped,
                    parameters[0].exp(),
                    self.likelihood,
                    &self.options,
                )
                .map_or(f64::INFINITY, |fit| fit.objective)
            };
            let search = optimizer.minimize(
                &[Vector::from_slice(&[-1.0]), Vector::from_slice(&[0.0])],
                objective,
                &session.child("variance_profile"),
            )?;
            ctx.report.merge(search.report);
            if search.value.termination != OptimizationTermination::Converged {
                return Err(failure(&mut ctx, "LMM variance search did not converge"));
            }
            let mut ratio = search.value.point[0].exp();
            let mut fitted =
                gaussian_profile(&design, y, &grouped, ratio, self.likelihood, &self.options)
                    .ok_or_else(|| {
                        failure(&mut ctx, "Gaussian variance profile is unrepresentable")
                    })?;
            if let Some(boundary) =
                gaussian_profile(&design, y, &grouped, 0.0, self.likelihood, &self.options)
            {
                if boundary.objective <= fitted.objective {
                    ratio = 0.0;
                    fitted = boundary;
                }
            }
            ctx.report.merge(fitted.report);
            let variance = ratio * fitted.sigma_squared;
            if !variance.is_finite() {
                return Err(failure(&mut ctx, "random-intercept variance overflow"));
            }
            let residual = y.sub(&design.matvec(&fitted.beta));
            let random_effects: Vec<(u64, f64)> = grouped
                .iter()
                .map(|(&group, rows)| {
                    let shrink = if ratio == 0.0 {
                        0.0
                    } else {
                        1.0 / (1.0 / ratio + rows.len() as f64)
                    };
                    (
                        group,
                        shrink * rows.iter().map(|&i| residual[i]).sum::<f64>(),
                    )
                })
                .collect();
            if random_effects.iter().any(|(_, value)| !value.is_finite()) {
                return Err(failure(
                    &mut ctx,
                    "Gaussian conditional effects are not representable",
                ));
            }
            annotations.push(Annotation::new(Topic::Computation,format!("Gaussian {:?} profiles fixed effects and residual scale; variance-ratio optimum={ratio}. The exact zero-variance face was evaluated separately.",self.likelihood),
                "Inspect variance-component boundary behavior and sensitivity to fixed effects."));
            note(&mut ctx,IssueCode::PValueUnreliable,"LMM reports variance components and BLUPs; unverified Wald/component tests are deliberately absent");
            if ratio == 0.0 {
                note(&mut ctx,IssueCode::DegenerateDistribution,"random-intercept variance is on the zero boundary; fitted population model reduces to a linear model");
            }
            return ctx.finish(FittedMixed {
                family: self.family,
                beta: fitted.beta,
                residual_variance: Some(fitted.sigma_squared),
                random_intercept_variance: variance,
                random_effects,
                log_likelihood: -fitted.objective,
                likelihood: self.likelihood,
                quadrature_difference: None,
                annotations,
                features: x.ncols(),
                options: self.options.clone(),
                quadrature: None,
            });
        }
        let baseline = GeneralizedLinearModel {
            family: self.family,
            options: self.options.clone(),
        }
        .fit(x, y, &session.child("initial_glm"))?;
        ctx.report.merge(baseline.report);
        let rule = NormalQuadrature::new(
            self.quadrature_points,
            &ctx.policy,
            &session.child("normal_rule"),
        )?;
        ctx.report.merge(rule.report);
        let rule = rule.value;
        let mut start = baseline.value.beta.as_slice().to_vec();
        start.push(0.5_f64.ln());
        let mut simplex = vec![Vector::from_slice(&start)];
        for j in 0..start.len() {
            let mut vertex = start.clone();
            vertex[j] += 0.25;
            simplex.push(Vector::from_slice(&vertex));
        }
        let objective = |parameters: &[f64]| {
            let beta = Vector::from_slice(&parameters[..design.ncols()]);
            -integrated_loglik(
                self.family,
                &design,
                y,
                &grouped,
                &beta,
                parameters[design.ncols()].exp(),
                &rule,
            )
        };
        let search =
            optimizer.minimize(&simplex, objective, &session.child("marginal_likelihood"))?;
        ctx.report.merge(search.report);
        if search.value.termination != OptimizationTermination::Converged {
            return Err(failure(
                &mut ctx,
                "GLMM marginal likelihood search did not converge",
            ));
        }
        let mut beta = Vector::from_slice(&search.value.point.as_slice()[..design.ncols()]);
        let mut sd = search.value.point[design.ncols()].exp();
        if !(sd * sd).is_finite() {
            return Err(failure(&mut ctx, "GLMM variance is not representable"));
        }
        let boundary = integrated_loglik(
            self.family,
            &design,
            y,
            &grouped,
            &baseline.value.beta,
            0.0,
            &rule,
        );
        let mut loglik = -search.value.value;
        if boundary >= loglik {
            beta = baseline.value.beta;
            sd = 0.0;
            loglik = boundary;
        }
        let refined = NormalQuadrature::new(
            2 * self.quadrature_points,
            &ctx.policy,
            &session.child("quadrature_check"),
        )?;
        ctx.report.merge(refined.report);
        let difference =
            (integrated_loglik(self.family, &design, y, &grouped, &beta, sd, &refined.value)
                - loglik)
                .abs();
        if !difference.is_finite() || difference > ctx.policy.residual_tol * (1.0 + loglik.abs()) {
            return Err(failure(&mut ctx,"GLMM quadrature did not stabilize when its order was doubled; increase quadrature_points"));
        }
        let eta = design.matvec(&beta);
        let random_effects: Vec<(u64, f64)> = grouped
            .iter()
            .map(|(&group, rows)| {
                let logs = group_log_terms(self.family, &eta, y, rows, sd, &rule);
                let norm = tsutsumi::special::logsumexp(&logs);
                (
                    group,
                    logs.iter()
                        .zip(&rule.nodes)
                        .map(|(&value, &z)| (value - norm).exp() * sd * z)
                        .sum(),
                )
            })
            .collect();
        if random_effects.iter().any(|(_, value)| !value.is_finite()) {
            return Err(failure(
                &mut ctx,
                "GLMM posterior effects are not representable",
            ));
        }
        ctx.push(Issue::builder(IssueCode::PValueUnreliable).severity(signlred::Severity::Advisory)
            .message(format!("GLMM integration is approximate; doubled-order log-likelihood difference={difference}"))
            .compromise(NumericalCompromise::new("exact marginal integration over Gaussian random intercepts","fixed Gauss–Hermite normal quadrature with a doubled-order check","non-Gaussian likelihood has no closed-form Gaussian integral","quadrature difference is an empirical diagnostic, not a rigorous integration error bound"))
            .build());
        annotations.push(Annotation::new(Topic::Computation,format!("Marginal ML uses {}-point normal quadrature. Doubled-order log-likelihood change={difference}; variance={}. A local optimum is not proof of a global optimum.",self.quadrature_points,sd*sd),
            "Check higher quadrature orders, alternative starting values, and profile or bootstrap uncertainty."));
        if sd == 0.0 {
            note(&mut ctx,IssueCode::DegenerateDistribution,"random-intercept variance is at zero; the fitted population model reduces to a GLM");
        }
        ctx.finish(FittedMixed {
            family: self.family,
            beta,
            residual_variance: None,
            random_intercept_variance: sd * sd,
            random_effects,
            log_likelihood: loglik,
            likelihood: self.likelihood,
            quadrature_difference: Some(difference),
            annotations,
            features: x.ncols(),
            options: self.options.clone(),
            quadrature: Some(rule),
        })
    }
}

impl FittedMixed {
    /// Population response mean, integrating over a new group's random intercept.
    pub fn predict_marginal(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = context(session, &self.options);
        validate_prediction(&mut ctx, x, self.features);
        if stopped(&ctx) {
            return Err(ctx.finish_failure());
        }
        let eta = design_matrix(x, self.options.fit_intercept).matvec(&self.beta);
        if eta.as_slice().iter().any(|v| !v.is_finite()) {
            return Err(failure(&mut ctx, "mixed-model linear predictor overflow"));
        }
        let sd = self.random_intercept_variance.sqrt();
        let output = Vector::from_iter(eta.as_slice().iter().map(|&value| match self.family {
            Family::Gaussian => value,
            Family::Poisson => (value + 0.5 * self.random_intercept_variance).exp(),
            Family::Binomial => {
                let rule = self
                    .quadrature
                    .as_ref()
                    .expect("GLMM stores integration rule");
                rule.nodes
                    .iter()
                    .zip(&rule.weights)
                    .map(|(&z, &w)| w * self.family.mean(value + sd * z))
                    .sum()
            }
        }));
        checked_prediction(ctx, output)
    }

    /// Plug-in response mean using each known group's BLUP/posterior mean.
    ///
    /// Nonlinear inverse links at posterior mean effects are not posterior
    /// predictive means. Unknown group IDs are an error; use predict_marginal.
    pub fn predict_conditional(
        &self,
        x: &Matrix,
        groups: &[u64],
        session: &Session,
    ) -> Result<Qualified<Vector>> {
        let mut ctx = context(session, &self.options);
        validate_prediction(&mut ctx, x, self.features);
        if groups.len() != x.nrows() {
            ctx.push(
                Issue::builder(IssueCode::DimensionMismatch)
                    .message("prediction groups length mismatch")
                    .build(),
            );
        }
        if stopped(&ctx) {
            return Err(ctx.finish_failure());
        }
        let eta = design_matrix(x, self.options.fit_intercept).matvec(&self.beta);
        let mut output = Vector::zeros(x.nrows());
        for (row, &group) in groups.iter().enumerate() {
            let Ok(index) = self
                .random_effects
                .binary_search_by_key(&group, |entry| entry.0)
            else {
                ctx.push(
                    Issue::builder(IssueCode::InvalidParameter)
                        .message(format!(
                            "unknown group {group}; request a marginal prediction for new groups"
                        ))
                        .build(),
                );
                return Err(ctx.finish_failure());
            };
            let predictor = eta[row] + self.random_effects[index].1;
            if !predictor.is_finite() {
                return Err(failure(&mut ctx, "conditional linear predictor overflow"));
            }
            output[row] = self.family.mean(predictor);
        }
        note(&mut ctx,IssueCode::PValueUnreliable,"conditional prediction uses plug-in group effects and does not integrate their posterior uncertainty");
        checked_prediction(ctx, output)
    }
}

struct GaussianProfile {
    beta: Vector,
    sigma_squared: f64,
    objective: f64,
    report: Report,
}

fn gaussian_profile(
    x: &Matrix,
    y: &Vector,
    groups: &BTreeMap<u64, Vec<usize>>,
    ratio: f64,
    likelihood: Likelihood,
    options: &AnalysisOptions,
) -> Option<GaussianProfile> {
    if !ratio.is_finite() || ratio < 0.0 {
        return None;
    }
    let mut xs = x.clone();
    let mut ys = y.clone();
    let mut logdet = 0.0;
    for rows in groups.values() {
        let m = rows.len() as f64;
        let scale = (1.0 + m * ratio).sqrt();
        if !scale.is_finite() {
            return None;
        }
        logdet += (m * ratio).ln_1p();
        let factor = (1.0 / scale - 1.0) / m;
        let sum_y: f64 = rows.iter().map(|&i| y[i]).sum();
        let sum_x: Vec<f64> = (0..x.ncols())
            .map(|j| rows.iter().map(|&i| x.get(i, j)).sum())
            .collect();
        for &i in rows {
            ys[i] = y[i] + factor * sum_y;
            for (j, &sum) in sum_x.iter().enumerate() {
                xs.set(i, j, x.get(i, j) + factor * sum);
            }
        }
    }
    let mut report = Report::new("number_ruler.LMM", "profile");
    let solution = least_squares_with_diagnostics(&mut report, &xs, &ys, &options.policy)?;
    if solution.rank != x.ncols() {
        return None;
    }
    let residual = ys.sub(&xs.matvec(&solution.coefficients));
    let sse = residual.dot(&residual);
    let df = if likelihood == Likelihood::Restricted {
        y.len().checked_sub(x.ncols())?
    } else {
        y.len()
    } as f64;
    let sigma_squared = sse / df;
    if df <= 0.0 || !sigma_squared.is_finite() || sigma_squared <= 0.0 {
        return None;
    }
    if likelihood == Likelihood::Restricted {
        logdet += 2.0
            * solution
                .decomposition
                .singular_values
                .iter()
                .map(|s| s.ln())
                .sum::<f64>();
    }
    let objective =
        0.5 * (df * ((2.0 * std::f64::consts::PI).ln() + 1.0 + sigma_squared.ln()) + logdet);
    objective.is_finite().then_some(GaussianProfile {
        beta: solution.coefficients,
        sigma_squared,
        objective,
        report,
    })
}

fn group_log_terms(
    family: Family,
    eta: &Vector,
    y: &Vector,
    rows: &[usize],
    sd: f64,
    rule: &NormalQuadrature,
) -> Vec<f64> {
    rule.nodes
        .iter()
        .zip(&rule.weights)
        .map(|(&z, &w)| {
            w.ln()
                + rows
                    .iter()
                    .map(|&i| family.log_density(y[i], eta[i] + sd * z))
                    .sum::<f64>()
        })
        .collect()
}
fn integrated_loglik(
    family: Family,
    x: &Matrix,
    y: &Vector,
    groups: &BTreeMap<u64, Vec<usize>>,
    beta: &Vector,
    sd: f64,
    rule: &NormalQuadrature,
) -> f64 {
    if !sd.is_finite() {
        return f64::NEG_INFINITY;
    }
    let eta = x.matvec(beta);
    let result = groups
        .values()
        .map(|rows| tsutsumi::special::logsumexp(&group_log_terms(family, &eta, y, rows, sd, rule)))
        .sum::<f64>();
    if result.is_finite() {
        result
    } else {
        f64::NEG_INFINITY
    }
}
fn checked_prediction(mut ctx: tsutsumi::FitCtx, output: Vector) -> Result<Qualified<Vector>> {
    if output.as_slice().iter().any(|v| !v.is_finite()) {
        ctx.push(
            Issue::builder(IssueCode::NonFiniteOutput)
                .message("mixed-model prediction overflow")
                .build(),
        );
        return Err(ctx.finish_failure());
    }
    ctx.finish(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_parameter_integrals_match_independent_adaptive_quadrature() {
        let data: serde_json::Value =
            serde_json::from_str(include_str!("../golden/regression.json")).unwrap();
        let rule = NormalQuadrature::new(
            128,
            &crate::Policy::default(),
            &Session::new("quad", "test"),
        )
        .unwrap()
        .value;
        for case in data["cases"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|c| c["kind"] == "glmm")
        {
            let family = if case["family"] == "Binomial" {
                Family::Binomial
            } else {
                Family::Poisson
            };
            let y = Vector::from_iter(
                case["y"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_f64().unwrap()),
            );
            let x = Matrix::from_fn(y.len(), 2, |i, j| {
                if j == 0 {
                    1.0
                } else {
                    case["x"][i][0].as_f64().unwrap()
                }
            });
            let mut groups: BTreeMap<u64, Vec<usize>> = BTreeMap::new();
            for (i, g) in case["groups"].as_array().unwrap().iter().enumerate() {
                groups.entry(g.as_u64().unwrap()).or_default().push(i);
            }
            let mut worst: f64 = 0.0;
            for probe in case["probes"].as_array().unwrap() {
                let beta = Vector::from_iter(
                    probe["beta"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|v| v.as_f64().unwrap()),
                );
                let actual = integrated_loglik(
                    family,
                    &x,
                    &y,
                    &groups,
                    &beta,
                    probe["sd"].as_f64().unwrap(),
                    &rule,
                );
                worst = worst.max((actual - probe["loglik"].as_f64().unwrap()).abs());
            }
            eprintln!("{family:?} fixed integral absolute error: {worst:e}");
            // Measured 2026-09-05: 1.43e-14 (Bernoulli), 5.83e-7
            // (Poisson), with a fourfold margin for this fixed-order probe.
            assert!(
                worst
                    < if family == Family::Binomial {
                        5.8e-14
                    } else {
                        2.34e-6
                    }
            );
        }
    }
}
