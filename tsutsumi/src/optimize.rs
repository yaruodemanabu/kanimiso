//! Derivative-free optimization with explicit numerical-quality reporting.
//!
//! The Nelder–Mead branch order follows the independently reviewed argmin
//! 0.11 implementation while remaining a first-party Pure Rust kernel:
//! <https://docs.rs/argmin/0.11.0/src/argmin/solver/neldermead/mod.rs.html>.
//! This implementation treats positive infinity as an ordered domain barrier,
//! rejects NaN and negative infinity, and requires both objective spread and
//! full-simplex diameter for successful convergence.
//! A simplex that reaches parameter resolution first stops as an explicitly
//! compromised partial result instead of being reported as converged.

use crate::context::FitCtx;
use crate::data::{Matrix, Vector};
use crate::linalg::thin_svd;
use ojizou_san::Session;
use signlred::{Issue, IssueCode, Policy, Qualified, Report, Result, Severity};

const ALGORITHM: &str = "optimize.NelderMead";

/// Runtime configuration for one unconstrained Nelder–Mead search.
///
/// The coefficient domains match argmin 0.11: `reflection > 0`,
/// `expansion > 1`, `0 < contraction <= 0.5`, and `0 < shrink <= 1`.
#[derive(Clone, Debug, PartialEq)]
pub struct NelderMead {
    /// Reflection coefficient (traditionally α).
    pub reflection: f64,
    /// Expansion coefficient (traditionally γ).
    pub expansion: f64,
    /// Inside/outside contraction coefficient (traditionally ρ).
    pub contraction: f64,
    /// Shrink coefficient (traditionally σ).
    pub shrink: f64,
    /// Maximum number of simplex iterations after initial evaluation.
    pub max_iterations: usize,
    /// Numerical-quality and convergence policy.
    pub policy: Policy,
}

impl Default for NelderMead {
    fn default() -> Self {
        Self {
            reflection: 1.0,
            expansion: 2.0,
            contraction: 0.5,
            shrink: 0.5,
            max_iterations: 1_000,
            policy: Policy::default(),
        }
    }
}

/// Why a Nelder–Mead search stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OptimizationTermination {
    /// Objective spread and full-simplex diameter both satisfied policy.
    Converged,
    /// The configured iteration cap was reached first.
    MaxIterations,
    /// The simplex stopped moving before objective values agreed.
    SimplexCollapsed,
}

/// Best point plus convergence evidence from a Nelder–Mead search.
#[derive(Clone, Debug, PartialEq)]
pub struct OptimizationResult {
    /// Best point in the final simplex.
    pub point: Vector,
    /// Objective value at `point`.
    pub value: f64,
    /// Number of completed simplex iterations.
    pub iterations: usize,
    /// Total objective evaluations, including initial vertices.
    pub evaluations: usize,
    /// Explicit stopping reason.
    pub termination: OptimizationTermination,
    /// Sample standard deviation of final simplex objective values.
    pub objective_std: f64,
    /// Maximum Euclidean distance between any final simplex vertices.
    pub simplex_diameter: f64,
}

#[derive(Clone, Debug, PartialEq)]
struct Vertex {
    point: Vec<f64>,
    value: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum IterationAction {
    Reflection,
    Expansion,
    ExpansionFallbackReflection,
    OutsideContraction,
    InsideContraction,
    ShrinkAfterOutsideContraction,
    ShrinkAfterInsideContraction,
}

impl NelderMead {
    /// Minimize `objective` from an explicit `dimension + 1` vertex simplex.
    ///
    /// Supplying the simplex makes initialization reproducible and exposes
    /// degenerate geometry before evaluation. Positive infinity is accepted as
    /// a dominated domain barrier when the initial simplex contains at least
    /// one finite value. NaN, negative infinity, or an all-barrier initial
    /// simplex is a [`IssueCode::LossIsNan`] failure.
    pub fn minimize<F>(
        &self,
        simplex: &[Vector],
        objective: F,
        session: &Session,
    ) -> Result<Qualified<OptimizationResult>>
    where
        F: FnMut(&[f64]) -> f64,
    {
        self.minimize_impl(simplex, objective, session, true)
    }

    /// Nested variant: iteration breadcrumbs share `session`, while the
    /// enclosing computation records the merged report and terminal event.
    pub fn minimize_nested<F>(
        &self,
        simplex: &[Vector],
        objective: F,
        session: &Session,
    ) -> Result<Qualified<OptimizationResult>>
    where
        F: FnMut(&[f64]) -> f64,
    {
        self.minimize_impl(simplex, objective, session, false)
    }

    fn minimize_impl<F>(
        &self,
        simplex: &[Vector],
        mut objective: F,
        session: &Session,
        record_completion: bool,
    ) -> Result<Qualified<OptimizationResult>>
    where
        F: FnMut(&[f64]) -> f64,
    {
        let mut ctx = FitCtx::with_session(session.child("minimize"));
        ctx.policy = self.policy.clone();
        if !record_completion {
            ctx.suppress_completion_recording();
        }

        if let Some(issue) = self.validation_issue(simplex) {
            ctx.push(issue);
            return Err(ctx.finish_failure());
        }

        let dimension = simplex[0].len();
        ctx.report.set_sample_shape(simplex.len(), dimension);
        ctx.report.set_n_parameters(dimension);
        let mut evaluations = 0usize;
        let mut vertices = Vec::with_capacity(simplex.len());
        for point in simplex {
            let value = match evaluate(&mut objective, point.as_slice(), &mut evaluations, &mut ctx)
            {
                Some(value) => value,
                None => return Err(ctx.finish_failure()),
            };
            vertices.push(Vertex {
                point: point.as_slice().to_vec(),
                value,
            });
        }
        sort_vertices(&mut vertices);
        if !vertices[0].value.is_finite() {
            ctx.push(
                Issue::builder(IssueCode::LossIsNan)
                    .message(
                        "every initial Nelder–Mead vertex evaluated to positive infinity; at least one finite objective value is required",
                    )
                    .metric("evaluations", evaluations as f64)
                    .build(),
            );
            return Err(ctx.finish_failure());
        }

        let initial_metrics = simplex_metrics(&vertices);
        if converged(&vertices, initial_metrics, &self.policy) {
            ctx.session
                .converged("objective spread and simplex diameter", 0);
            return ctx.finish(result(
                &vertices,
                0,
                evaluations,
                OptimizationTermination::Converged,
                initial_metrics,
            ));
        }
        if parameter_converged(&vertices, initial_metrics.1, &self.policy) {
            return finish_collapsed(&vertices, 0, evaluations, initial_metrics, ctx);
        }

        for iteration in 1..=self.max_iterations {
            if next_iteration(
                self,
                &mut vertices,
                &mut objective,
                &mut evaluations,
                &mut ctx,
            )
            .is_none()
            {
                return Err(ctx.finish_failure());
            }
            let metrics = simplex_metrics(&vertices);
            ctx.session.step(iteration as u64, vertices[0].value, None);

            if converged(&vertices, metrics, &self.policy) {
                ctx.session.converged(
                    "objective spread and full-simplex diameter",
                    iteration as u64,
                );
                return ctx.finish(result(
                    &vertices,
                    iteration,
                    evaluations,
                    OptimizationTermination::Converged,
                    metrics,
                ));
            }
            if parameter_converged(&vertices, metrics.1, &self.policy) {
                return finish_collapsed(&vertices, iteration, evaluations, metrics, ctx);
            }
        }

        let metrics = simplex_metrics(&vertices);
        ctx.push(
            Issue::builder(IssueCode::MaxIterReached)
                .message(format!(
                    "Nelder–Mead reached max_iterations={} before convergence",
                    self.max_iterations
                ))
                .metric("iterations", self.max_iterations as f64)
                .metric("objective_std", metrics.0)
                .metric("simplex_diameter", metrics.1)
                .build(),
        );
        ctx.session.diverged("Nelder–Mead iteration cap reached");
        ctx.finish(result(
            &vertices,
            self.max_iterations,
            evaluations,
            OptimizationTermination::MaxIterations,
            metrics,
        ))
    }

    fn validation_issue(&self, simplex: &[Vector]) -> Option<Issue> {
        if !self.reflection.is_finite() || self.reflection <= 0.0 {
            return Some(invalid_parameter("reflection must be finite and positive"));
        }
        if !self.expansion.is_finite() || self.expansion <= 1.0 {
            return Some(invalid_parameter(
                "expansion must be finite and greater than one",
            ));
        }
        if !self.contraction.is_finite() || self.contraction <= 0.0 || self.contraction > 0.5 {
            return Some(invalid_parameter(
                "contraction must be finite and in (0, 0.5]",
            ));
        }
        if !self.shrink.is_finite() || self.shrink <= 0.0 || self.shrink > 1.0 {
            return Some(invalid_parameter("shrink must be finite and in (0, 1]"));
        }
        if self.max_iterations == 0 {
            return Some(invalid_parameter("max_iterations must be positive"));
        }
        if !valid_tolerance(self.policy.optimizer_objective_tol) {
            return Some(invalid_parameter(
                "Policy::optimizer_objective_tol must be finite and positive",
            ));
        }
        if !valid_tolerance(self.policy.optimizer_parameter_tol) {
            return Some(invalid_parameter(
                "Policy::optimizer_parameter_tol must be finite and positive",
            ));
        }
        if !self.policy.rank_tol_relative.is_finite()
            || self.policy.rank_tol_relative <= 0.0
            || self.policy.rank_tol_relative >= 1.0
        {
            return Some(invalid_parameter(
                "Policy::rank_tol_relative must be finite and in (0, 1)",
            ));
        }
        if simplex.is_empty() {
            return Some(
                Issue::builder(IssueCode::EmptyMatrix)
                    .message("Nelder–Mead requires a non-empty simplex")
                    .build(),
            );
        }
        let dimension = simplex[0].len();
        if dimension == 0 {
            return Some(
                Issue::builder(IssueCode::EmptyMatrix)
                    .message("Nelder–Mead vertices must have positive dimension")
                    .build(),
            );
        }
        if simplex.len() != dimension + 1 {
            return Some(
                Issue::builder(IssueCode::DimensionMismatch)
                    .message(format!(
                        "a {dimension}-dimensional Nelder–Mead simplex needs {} vertices, got {}",
                        dimension + 1,
                        simplex.len()
                    ))
                    .metric("dimension", dimension as f64)
                    .metric("vertices", simplex.len() as f64)
                    .build(),
            );
        }
        for (vertex, point) in simplex.iter().enumerate() {
            if point.len() != dimension {
                return Some(
                    Issue::builder(IssueCode::DimensionMismatch)
                        .message(format!(
                            "simplex vertex {vertex} has dimension {}, expected {dimension}",
                            point.len()
                        ))
                        .metric("vertex", vertex as f64)
                        .build(),
                );
            }
            if point
                .as_slice()
                .iter()
                .any(|coordinate| !coordinate.is_finite())
            {
                return Some(
                    Issue::builder(IssueCode::NonFiniteInput)
                        .message(format!("simplex vertex {vertex} is not finite"))
                        .metric("vertex", vertex as f64)
                        .build(),
                );
            }
        }

        let coordinate_scale = simplex
            .iter()
            .flat_map(|point| point.as_slice())
            .fold(0.0_f64, |scale, coordinate| scale.max(coordinate.abs()));
        let coordinate_scale = if coordinate_scale == 0.0 {
            1.0
        } else {
            coordinate_scale
        };
        let edges = Matrix::from_fn(dimension, dimension, |row, column| {
            simplex[column + 1][row] / coordinate_scale - simplex[0][row] / coordinate_scale
        });
        let mut report = Report::new(ALGORITHM, "validate_simplex");
        let Some(svd) = thin_svd(&mut report, &edges, &self.policy) else {
            return report.primary().cloned().or_else(|| {
                Some(
                    Issue::builder(IssueCode::SvdDidNotConverge)
                        .message("simplex rank validation failed")
                        .build(),
                )
            });
        };
        let rank = svd.rank(self.policy.rank_tol_relative);
        if rank != dimension {
            return Some(
                Issue::builder(IssueCode::InvalidParameter)
                    .message(format!(
                        "initial simplex is affinely rank deficient: rank {rank}, expected {dimension}"
                    ))
                    .metric("rank", rank as f64)
                    .metric("dimension", dimension as f64)
                    .build(),
            );
        }
        None
    }
}

fn next_iteration<F>(
    config: &NelderMead,
    vertices: &mut [Vertex],
    objective: &mut F,
    evaluations: &mut usize,
    ctx: &mut FitCtx,
) -> Option<IterationAction>
where
    F: FnMut(&[f64]) -> f64,
{
    let last = vertices.len() - 1;
    let centroid = centroid_without_worst(vertices);
    let best_value = vertices[0].value;
    let second_worst_value = vertices[last - 1].value;
    let worst = vertices[last].clone();

    let reflected_point = affine(&centroid, &worst.point, -config.reflection);
    let reflected_value = evaluate(objective, &reflected_point, evaluations, ctx)?;
    let reflected = Vertex {
        point: reflected_point,
        value: reflected_value,
    };

    let action = if reflected.value < second_worst_value && reflected.value >= best_value {
        vertices[last] = reflected;
        IterationAction::Reflection
    } else if reflected.value < best_value {
        let expanded_point = affine(&centroid, &reflected.point, config.expansion);
        let expanded_value = evaluate(objective, &expanded_point, evaluations, ctx)?;
        if expanded_value < reflected.value {
            vertices[last] = Vertex {
                point: expanded_point,
                value: expanded_value,
            };
            IterationAction::Expansion
        } else {
            vertices[last] = reflected;
            IterationAction::ExpansionFallbackReflection
        }
    } else if reflected.value < worst.value {
        let contracted_point = affine(&centroid, &reflected.point, config.contraction);
        let contracted_value = evaluate(objective, &contracted_point, evaluations, ctx)?;
        if contracted_value <= reflected.value {
            vertices[last] = Vertex {
                point: contracted_point,
                value: contracted_value,
            };
            IterationAction::OutsideContraction
        } else {
            shrink(config, vertices, objective, evaluations, ctx)?;
            IterationAction::ShrinkAfterOutsideContraction
        }
    } else {
        let contracted_point = affine(&centroid, &worst.point, config.contraction);
        let contracted_value = evaluate(objective, &contracted_point, evaluations, ctx)?;
        if contracted_value < worst.value {
            vertices[last] = Vertex {
                point: contracted_point,
                value: contracted_value,
            };
            IterationAction::InsideContraction
        } else {
            shrink(config, vertices, objective, evaluations, ctx)?;
            IterationAction::ShrinkAfterInsideContraction
        }
    };
    sort_vertices(vertices);
    Some(action)
}

fn shrink<F>(
    config: &NelderMead,
    vertices: &mut [Vertex],
    objective: &mut F,
    evaluations: &mut usize,
    ctx: &mut FitCtx,
) -> Option<()>
where
    F: FnMut(&[f64]) -> f64,
{
    let best = vertices[0].point.clone();
    for vertex in vertices.iter_mut().skip(1) {
        let point = affine(&best, &vertex.point, config.shrink);
        let value = evaluate(objective, &point, evaluations, ctx)?;
        vertex.point = point;
        vertex.value = value;
    }
    Some(())
}

fn evaluate<F>(
    objective: &mut F,
    point: &[f64],
    evaluations: &mut usize,
    ctx: &mut FitCtx,
) -> Option<f64>
where
    F: FnMut(&[f64]) -> f64,
{
    if point.iter().any(|coordinate| !coordinate.is_finite()) {
        ctx.push(
            Issue::builder(IssueCode::NonFiniteOutput)
                .severity(Severity::Fatal)
                .message("Nelder–Mead generated a non-finite point before objective evaluation")
                .build(),
        );
        return None;
    }
    let value = objective(point);
    *evaluations += 1;
    if value.is_finite() || value == f64::INFINITY {
        Some(value)
    } else {
        ctx.push(
            Issue::builder(IssueCode::LossIsNan)
                .message(format!(
                    "objective evaluation {} returned NaN or negative infinity ({value})",
                    *evaluations
                ))
                .metric("evaluation", *evaluations as f64)
                .build(),
        );
        None
    }
}

fn invalid_parameter(message: &'static str) -> Issue {
    Issue::builder(IssueCode::InvalidParameter)
        .message(message)
        .build()
}

fn valid_tolerance(value: f64) -> bool {
    value.is_finite() && value > 0.0
}

/// Encode an interior finite value as an unconstrained log-odds coordinate.
pub fn encode_open_interval(value: f64, lower: f64, upper: f64) -> Option<f64> {
    let width = upper - lower;
    let lower_gap = value - lower;
    let upper_gap = upper - value;
    if !width.is_finite()
        || width <= 0.0
        || !lower_gap.is_finite()
        || lower_gap <= 0.0
        || !upper_gap.is_finite()
        || upper_gap <= 0.0
    {
        return None;
    }
    let encoded = lower_gap.ln() - upper_gap.ln();
    encoded.is_finite().then_some(encoded)
}

/// Decode log odds, returning `None` when an interior value is unrepresentable.
pub fn decode_open_interval(value: f64, lower: f64, upper: f64) -> Option<f64> {
    let width = upper - lower;
    if !value.is_finite() || !width.is_finite() || width <= 0.0 {
        return None;
    }
    let interior_lower = next_up(lower);
    let interior_upper = next_down(upper);
    if interior_lower > interior_upper {
        return None;
    }
    let decoded = if value >= 0.0 {
        let tail = (-value).exp();
        upper - width * (tail / (1.0 + tail))
    } else {
        let head = value.exp();
        lower + width * (head / (1.0 + head))
    };
    decoded
        .is_finite()
        .then_some(decoded.clamp(interior_lower, interior_upper))
}

fn next_up(value: f64) -> f64 {
    if value == f64::INFINITY {
        return value;
    }
    if value == 0.0 {
        return f64::from_bits(1);
    }
    let bits = value.to_bits();
    if value > 0.0 {
        f64::from_bits(bits + 1)
    } else {
        f64::from_bits(bits - 1)
    }
}

fn next_down(value: f64) -> f64 {
    if value == f64::NEG_INFINITY {
        return value;
    }
    if value == 0.0 {
        return f64::from_bits((1_u64 << 63) | 1);
    }
    let bits = value.to_bits();
    if value > 0.0 {
        f64::from_bits(bits - 1)
    } else {
        f64::from_bits(bits + 1)
    }
}

fn affine(origin: &[f64], target: &[f64], scale: f64) -> Vec<f64> {
    origin
        .iter()
        .zip(target)
        .map(|(origin, target)| {
            let coordinate_scale = origin.abs().max(target.abs());
            if coordinate_scale == 0.0 {
                0.0
            } else {
                coordinate_scale
                    * (origin / coordinate_scale
                        + scale * (target / coordinate_scale - origin / coordinate_scale))
            }
        })
        .collect()
}

fn centroid_without_worst(vertices: &[Vertex]) -> Vec<f64> {
    let dimension = vertices[0].point.len();
    let count = vertices.len() - 1;
    let mut centroid = vec![0.0; dimension];
    for (column, coordinate) in centroid.iter_mut().enumerate() {
        let scale = vertices
            .iter()
            .take(count)
            .map(|vertex| vertex.point[column].abs())
            .fold(0.0_f64, f64::max);
        if scale != 0.0 {
            *coordinate = vertices
                .iter()
                .take(count)
                .map(|vertex| vertex.point[column] / scale)
                .sum::<f64>()
                / count as f64
                * scale;
        }
    }
    centroid
}

fn sort_vertices(vertices: &mut [Vertex]) {
    // `sort_by` is stable, matching argmin's value-only ordering: caller order
    // is retained when non-NaN objective values are exactly equal.
    vertices.sort_by(|left, right| {
        left.value
            .partial_cmp(&right.value)
            .expect("optimizer stores no NaN objective values")
    });
}

fn simplex_metrics(vertices: &[Vertex]) -> (f64, f64) {
    let count = vertices.len() as f64;
    let has_barrier = vertices.iter().any(|vertex| vertex.value == f64::INFINITY);
    let objective_scale = vertices
        .iter()
        .filter(|vertex| vertex.value.is_finite())
        .map(|vertex| vertex.value.abs())
        .fold(0.0_f64, f64::max);
    let objective_std = if has_barrier {
        f64::INFINITY
    } else if objective_scale == 0.0 {
        0.0
    } else {
        let mean = vertices
            .iter()
            .map(|vertex| vertex.value / objective_scale)
            .sum::<f64>()
            / count;
        let scaled_std = (vertices
            .iter()
            .map(|vertex| {
                let residual = vertex.value / objective_scale - mean;
                residual * residual
            })
            .sum::<f64>()
            / (count - 1.0))
            .sqrt();
        objective_scale * scaled_std
    };

    let mut diameter = 0.0_f64;
    for left in 0..vertices.len() {
        for right in left + 1..vertices.len() {
            let distance = stable_distance(&vertices[left].point, &vertices[right].point);
            diameter = diameter.max(distance);
        }
    }
    (objective_std, diameter)
}

fn stable_distance(left: &[f64], right: &[f64]) -> f64 {
    let mut scale = 0.0_f64;
    let mut sum_squares = 1.0_f64;
    for (left, right) in left.iter().zip(right) {
        let absolute = (left - right).abs();
        if absolute.is_infinite() {
            return f64::INFINITY;
        }
        if absolute == 0.0 {
            continue;
        }
        if scale < absolute {
            let ratio = scale / absolute;
            sum_squares = 1.0 + sum_squares * ratio * ratio;
            scale = absolute;
        } else {
            let ratio = absolute / scale;
            sum_squares += ratio * ratio;
        }
    }
    if scale == 0.0 {
        0.0
    } else {
        scale * sum_squares.sqrt()
    }
}

fn converged(vertices: &[Vertex], metrics: (f64, f64), policy: &Policy) -> bool {
    objective_converged(vertices, metrics.0, policy)
        && parameter_converged(vertices, metrics.1, policy)
}

fn objective_converged(vertices: &[Vertex], objective_std: f64, policy: &Policy) -> bool {
    objective_std / (1.0 + vertices[0].value.abs()) <= policy.optimizer_objective_tol
}

fn parameter_converged(vertices: &[Vertex], diameter: f64, policy: &Policy) -> bool {
    let scale = vertices
        .iter()
        .flat_map(|vertex| vertex.point.iter())
        .fold(0.0_f64, |maximum, coordinate| maximum.max(coordinate.abs()));
    diameter / (1.0 + scale) <= policy.optimizer_parameter_tol
}

fn result(
    vertices: &[Vertex],
    iterations: usize,
    evaluations: usize,
    termination: OptimizationTermination,
    metrics: (f64, f64),
) -> OptimizationResult {
    OptimizationResult {
        point: Vector::from_slice(&vertices[0].point),
        value: vertices[0].value,
        iterations,
        evaluations,
        termination,
        objective_std: metrics.0,
        simplex_diameter: metrics.1,
    }
}

fn finish_collapsed(
    vertices: &[Vertex],
    iterations: usize,
    evaluations: usize,
    metrics: (f64, f64),
    mut ctx: FitCtx,
) -> Result<Qualified<OptimizationResult>> {
    ctx.push(
        Issue::builder(IssueCode::StepSizeCollapsed)
            .message(
                "the full simplex collapsed before objective values satisfied the convergence tolerance",
            )
            .metric("iteration", iterations as f64)
            .metric("objective_std", metrics.0)
            .metric("simplex_diameter", metrics.1)
            .build(),
    );
    ctx.session
        .diverged("simplex collapsed before objective convergence");
    ctx.finish(result(
        vertices,
        iterations,
        evaluations,
        OptimizationTermination::SimplexCollapsed,
        metrics,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn branch_simplex() -> Vec<Vertex> {
        vec![
            Vertex {
                point: vec![0.0, 0.0],
                value: 0.0,
            },
            Vertex {
                point: vec![2.0, 0.0],
                value: 10.0,
            },
            Vertex {
                point: vec![0.0, 2.0],
                value: 20.0,
            },
        ]
    }

    fn branch_step(costs: &[(&[f64], f64)]) -> (IterationAction, Vec<Vertex>, Vec<Vec<f64>>) {
        let config = NelderMead::default();
        let mut vertices = branch_simplex();
        let mut trace = Vec::new();
        let mut objective = |point: &[f64]| {
            trace.push(point.to_vec());
            costs
                .iter()
                .find(|(expected, _)| *expected == point)
                .map(|(_, value)| *value)
                .unwrap_or_else(|| panic!("unexpected point {point:?}"))
        };
        let mut evaluations = 0;
        let mut ctx = FitCtx::new(ALGORITHM, "branch_test");
        let action = next_iteration(
            &config,
            &mut vertices,
            &mut objective,
            &mut evaluations,
            &mut ctx,
        )
        .expect("finite scripted objective");
        assert_eq!(evaluations, trace.len());
        (action, vertices, trace)
    }

    fn assert_vertices(actual: &[Vertex], expected: &[(&[f64], f64)]) {
        assert_eq!(actual.len(), expected.len());
        for (actual, (point, value)) in actual.iter().zip(expected) {
            assert_eq!(actual.point, *point);
            assert_eq!(actual.value, *value);
        }
    }

    #[test]
    fn argmin_trace_reflection_branch() {
        let (action, vertices, trace) = branch_step(&[(&[2.0, -2.0], 5.0)]);
        assert_eq!(action, IterationAction::Reflection);
        assert_eq!(trace, vec![vec![2.0, -2.0]]);
        assert_vertices(
            &vertices,
            &[(&[0.0, 0.0], 0.0), (&[2.0, -2.0], 5.0), (&[2.0, 0.0], 10.0)],
        );
    }

    #[test]
    fn argmin_trace_expansion_and_strict_fallback() {
        let expanded = branch_step(&[(&[2.0, -2.0], -1.0), (&[3.0, -4.0], -2.0)]);
        assert_eq!(expanded.0, IterationAction::Expansion);
        assert_eq!(expanded.2, vec![vec![2.0, -2.0], vec![3.0, -4.0]]);
        assert_vertices(
            &expanded.1,
            &[
                (&[3.0, -4.0], -2.0),
                (&[0.0, 0.0], 0.0),
                (&[2.0, 0.0], 10.0),
            ],
        );

        let fallback = branch_step(&[(&[2.0, -2.0], -1.0), (&[3.0, -4.0], -1.0)]);
        assert_eq!(fallback.0, IterationAction::ExpansionFallbackReflection);
        assert_vertices(
            &fallback.1,
            &[
                (&[2.0, -2.0], -1.0),
                (&[0.0, 0.0], 0.0),
                (&[2.0, 0.0], 10.0),
            ],
        );
    }

    #[test]
    fn argmin_trace_outside_and_inside_contraction() {
        let outside = branch_step(&[(&[2.0, -2.0], 15.0), (&[1.5, -1.0], 15.0)]);
        assert_eq!(outside.0, IterationAction::OutsideContraction);
        assert_vertices(
            &outside.1,
            &[
                (&[0.0, 0.0], 0.0),
                (&[2.0, 0.0], 10.0),
                (&[1.5, -1.0], 15.0),
            ],
        );

        let inside = branch_step(&[(&[2.0, -2.0], 25.0), (&[0.5, 1.0], 15.0)]);
        assert_eq!(inside.0, IterationAction::InsideContraction);
        assert_vertices(
            &inside.1,
            &[(&[0.0, 0.0], 0.0), (&[2.0, 0.0], 10.0), (&[0.5, 1.0], 15.0)],
        );
    }

    #[test]
    fn argmin_trace_failed_contractions_shrink_without_moving_best() {
        let outside = branch_step(&[
            (&[2.0, -2.0], 15.0),
            (&[1.5, -1.0], 16.0),
            (&[1.0, 0.0], 4.0),
            (&[0.0, 1.0], 6.0),
        ]);
        assert_eq!(outside.0, IterationAction::ShrinkAfterOutsideContraction);
        assert_eq!(
            outside.2,
            vec![
                vec![2.0, -2.0],
                vec![1.5, -1.0],
                vec![1.0, 0.0],
                vec![0.0, 1.0]
            ]
        );
        assert_vertices(
            &outside.1,
            &[(&[0.0, 0.0], 0.0), (&[1.0, 0.0], 4.0), (&[0.0, 1.0], 6.0)],
        );

        let inside = branch_step(&[
            (&[2.0, -2.0], 25.0),
            (&[0.5, 1.0], 20.0),
            (&[1.0, 0.0], 4.0),
            (&[0.0, 1.0], 6.0),
        ]);
        assert_eq!(inside.0, IterationAction::ShrinkAfterInsideContraction);
        assert_vertices(
            &inside.1,
            &[(&[0.0, 0.0], 0.0), (&[1.0, 0.0], 4.0), (&[0.0, 1.0], 6.0)],
        );
    }

    #[test]
    fn equal_objectives_preserve_caller_vertex_order() {
        let mut vertices = vec![
            Vertex {
                point: vec![2.0],
                value: -0.0,
            },
            Vertex {
                point: vec![0.0],
                value: 0.0,
            },
            Vertex {
                point: vec![1.0],
                value: -0.0,
            },
        ];
        sort_vertices(&mut vertices);
        assert_eq!(vertices[0].point, vec![2.0]);
        assert_eq!(vertices[1].point, vec![0.0]);
        assert_eq!(vertices[2].point, vec![1.0]);
    }

    #[test]
    fn extreme_finite_values_do_not_make_geometry_metrics_nan() {
        let large = 0.75 * f64::MAX;
        let vertices = vec![
            Vertex {
                point: vec![large, f64::MAX],
                value: -f64::MAX,
            },
            Vertex {
                point: vec![large, -f64::MAX],
                value: 0.0,
            },
            Vertex {
                point: vec![f64::MAX, 0.0],
                value: f64::MAX,
            },
        ];
        let centroid = centroid_without_worst(&vertices);
        assert_eq!(centroid, vec![large, 0.0]);

        let metrics = simplex_metrics(&vertices);
        assert!(!metrics.0.is_nan());
        assert!(!metrics.1.is_nan());
        assert!(!objective_converged(
            &vertices,
            metrics.0,
            &Policy::default()
        ));

        let representable = stable_distance(&[f64::MAX / 2.0, f64::MAX / 2.0], &[0.0, 0.0]);
        assert!(representable.is_finite());
    }

    #[test]
    fn positive_infinity_is_an_ordered_barrier_not_a_nan() {
        let simplex = vec![Vector::from_slice(&[0.0]), Vector::from_slice(&[-1.0])];
        let fitted = NelderMead::default()
            .minimize(
                &simplex,
                |point| {
                    if point[0] < 0.0 {
                        f64::INFINITY
                    } else {
                        (point[0] - 1.0).powi(2)
                    }
                },
                &Session::new(ALGORITHM, "positive-infinity-barrier"),
            )
            .expect("a finite initial incumbent can escape a positive-infinity barrier");
        assert!(fitted.value.value.is_finite());
        assert!((fitted.value.point[0] - 1.0).abs() <= 4.0 * f64::EPSILON);
        assert!(!fitted.report.contains(IssueCode::LossIsNan));

        let vertices = vec![
            Vertex {
                point: vec![0.0],
                value: 1.0,
            },
            Vertex {
                point: vec![-1.0],
                value: f64::INFINITY,
            },
        ];
        let metrics = simplex_metrics(&vertices);
        assert_eq!(metrics.0, f64::INFINITY);
        assert!(!metrics.0.is_nan());
        assert!(!objective_converged(
            &vertices,
            metrics.0,
            &Policy::default()
        ));
    }

    #[test]
    fn open_interval_transform_is_stable_and_rejects_invalid_domains() {
        let lower = 0.5;
        let upper = 0.999;
        for value in [0.500_001, 0.75, 0.94, 0.998_999] {
            let encoded = encode_open_interval(value, lower, upper).expect("interior value");
            let decoded = decode_open_interval(encoded, lower, upper).expect("finite encoding");
            // Measured error was 0.0 on 2026-09-02; four unit-scale ulps allow libm variation.
            assert!((decoded - value).abs() <= 4.0 * f64::EPSILON);
        }
        let near_lower = decode_open_interval(-1_000.0, lower, upper).expect("lower interior");
        let near_upper = decode_open_interval(1_000.0, lower, upper).expect("upper interior");
        assert!(lower < near_lower && near_lower < upper);
        assert!(lower < near_upper && near_upper < upper);
        assert_eq!(near_lower, next_up(lower));
        assert_eq!(near_upper, next_down(upper));
        assert!(encode_open_interval(lower, lower, upper).is_none());
        assert!(encode_open_interval(upper, lower, upper).is_none());
        assert!(decode_open_interval(f64::INFINITY, lower, upper).is_none());
        assert!(decode_open_interval(0.0, upper, lower).is_none());
    }

    #[test]
    fn generated_non_finite_point_aborts_before_objective_call() {
        let large = 0.75 * f64::MAX;
        let simplex = vec![
            Vector::from_slice(&[f64::MAX]),
            Vector::from_slice(&[large]),
        ];
        let mut evaluations = 0usize;
        let failure = NelderMead::default()
            .minimize(
                &simplex,
                |point| {
                    evaluations += 1;
                    if point[0] == f64::MAX {
                        0.0
                    } else {
                        1.0
                    }
                },
                &Session::new(ALGORITHM, "overflow"),
            )
            .unwrap_err();
        assert_eq!(failure.primary.code, IssueCode::NonFiniteOutput);
        assert_eq!(evaluations, 2, "overflow point reached the objective");
    }

    fn quadratic(point: &[f64]) -> f64 {
        let first = point[0] - 1.25;
        let second = point[1] + 0.75;
        first * first + 3.0 * second * second
    }

    fn quadratic_simplex() -> Vec<Vector> {
        vec![
            Vector::from_slice(&[-2.0, 2.0]),
            Vector::from_slice(&[-1.0, 2.0]),
            Vector::from_slice(&[-2.0, 3.0]),
        ]
    }

    #[test]
    fn quadratic_converges_and_best_value_is_monotone() {
        let config = NelderMead::default();
        let qualified = config
            .minimize(
                &quadratic_simplex(),
                quadratic,
                &Session::new(ALGORITHM, "test"),
            )
            .expect("quadratic must converge");
        assert_eq!(
            qualified.value.termination,
            OptimizationTermination::Converged
        );
        assert!(!qualified.report.has_warning());
        let error = ((qualified.value.point[0] - 1.25).powi(2)
            + (qualified.value.point[1] + 0.75).powi(2))
        .sqrt();
        // Measured 7.94e-9 on 2026-09-02; tolerance is about 4x that error.
        assert!(error <= 3.2e-8, "quadratic parameter error {error:e}");

        let mut vertices = quadratic_simplex()
            .into_iter()
            .map(|point| Vertex {
                value: quadratic(point.as_slice()),
                point: point.as_slice().to_vec(),
            })
            .collect::<Vec<_>>();
        sort_vertices(&mut vertices);
        let mut evaluations = vertices.len();
        let mut ctx = FitCtx::new(ALGORITHM, "monotonicity_test");
        for _ in 0..64 {
            let before = vertices[0].value;
            next_iteration(
                &config,
                &mut vertices,
                &mut quadratic,
                &mut evaluations,
                &mut ctx,
            )
            .expect("finite quadratic");
            assert!(vertices[0].value <= before);
        }
    }

    #[test]
    fn rosenbrock_converges_to_closed_form_minimum() {
        let simplex = vec![
            Vector::from_slice(&[-1.2, 1.0]),
            Vector::from_slice(&[-0.2, 1.0]),
            Vector::from_slice(&[-1.2, 2.0]),
        ];
        let result = NelderMead::default()
            .minimize(
                &simplex,
                |point| {
                    let residual = point[1] - point[0] * point[0];
                    let location = 1.0 - point[0];
                    100.0 * residual * residual + location * location
                },
                &Session::new(ALGORITHM, "test"),
            )
            .expect("Rosenbrock must converge")
            .value;
        assert_eq!(result.termination, OptimizationTermination::Converged);
        let error = ((result.point[0] - 1.0).powi(2) + (result.point[1] - 1.0).powi(2)).sqrt();
        // Measured 3.86e-9 on 2026-09-02; tolerance is about 4x that error.
        assert!(error <= 1.6e-8, "Rosenbrock parameter error {error:e}");
    }

    #[test]
    fn translated_and_permuted_quadratics_preserve_solution() {
        let config = NelderMead::default();
        let original = config
            .minimize(
                &quadratic_simplex(),
                quadratic,
                &Session::new(ALGORITHM, "original"),
            )
            .expect("original")
            .value;
        let shift = [8.0, -5.0];
        let shifted_simplex = quadratic_simplex()
            .iter()
            .map(|point| Vector::from_slice(&[point[0] + shift[0], point[1] + shift[1]]))
            .collect::<Vec<_>>();
        let shifted = config
            .minimize(
                &shifted_simplex,
                |point| quadratic(&[point[0] - shift[0], point[1] - shift[1]]),
                &Session::new(ALGORITHM, "shifted"),
            )
            .expect("shifted")
            .value;
        // Measured max translation delta 1.515e-8 on 2026-09-02; this is 3.9x.
        let tolerance = 4.0 * config.policy.optimizer_parameter_tol;
        assert!((shifted.point[0] - shift[0] - original.point[0]).abs() <= tolerance);
        assert!((shifted.point[1] - shift[1] - original.point[1]).abs() <= tolerance);

        let permuted_simplex = quadratic_simplex()
            .iter()
            .map(|point| Vector::from_slice(&[point[1], point[0]]))
            .collect::<Vec<_>>();
        let permuted = config
            .minimize(
                &permuted_simplex,
                |point| quadratic(&[point[1], point[0]]),
                &Session::new(ALGORITHM, "permuted"),
            )
            .expect("permuted")
            .value;
        assert!((permuted.point[0] - original.point[1]).abs() <= tolerance);
        assert!((permuted.point[1] - original.point[0]).abs() <= tolerance);
    }

    #[test]
    fn invalid_configuration_simplex_and_objective_fail_explicitly() {
        let session = Session::new(ALGORITHM, "test");
        let invalid = NelderMead {
            contraction: 0.75,
            ..NelderMead::default()
        };
        let failure = invalid
            .minimize(&quadratic_simplex(), quadratic, &session)
            .unwrap_err();
        assert_eq!(failure.primary.code, IssueCode::InvalidParameter);

        let rank_deficient = vec![
            Vector::from_slice(&[0.0, 0.0]),
            Vector::from_slice(&[1.0, 1.0]),
            Vector::from_slice(&[2.0, 2.0]),
        ];
        let failure = NelderMead::default()
            .minimize(&rank_deficient, quadratic, &session)
            .unwrap_err();
        assert_eq!(failure.primary.code, IssueCode::InvalidParameter);

        let failure = NelderMead::default()
            .minimize(&quadratic_simplex(), |_| f64::NAN, &session)
            .unwrap_err();
        assert_eq!(failure.primary.code, IssueCode::LossIsNan);
        assert!(failure.report.contains(IssueCode::LossIsNan));

        let failure = NelderMead::default()
            .minimize(&quadratic_simplex(), |_| f64::NEG_INFINITY, &session)
            .unwrap_err();
        assert_eq!(failure.primary.code, IssueCode::LossIsNan);

        let failure = NelderMead::default()
            .minimize(&quadratic_simplex(), |_| f64::INFINITY, &session)
            .unwrap_err();
        assert_eq!(failure.primary.code, IssueCode::LossIsNan);
        assert!(failure
            .primary
            .message
            .contains("at least one finite objective value"));
    }

    #[test]
    fn iteration_cap_is_a_qualified_nonconvergence() {
        let config = NelderMead {
            max_iterations: 1,
            ..NelderMead::default()
        };
        let qualified = config
            .minimize(
                &quadratic_simplex(),
                quadratic,
                &Session::new(ALGORITHM, "test"),
            )
            .expect("iteration cap is a reportable partial result");
        assert_eq!(
            qualified.value.termination,
            OptimizationTermination::MaxIterations
        );
        assert!(qualified.report.contains(IssueCode::MaxIterReached));
        assert!(qualified.is_compromised());
        assert_eq!(qualified.value.iterations, 1);
        assert_eq!(qualified.value.evaluations, 5);
    }

    #[test]
    fn parameter_resolution_without_objective_agreement_is_not_convergence() {
        let policy = Policy {
            optimizer_parameter_tol: 1.0e6,
            optimizer_objective_tol: f64::MIN_POSITIVE,
            ..Policy::default()
        };
        let config = NelderMead {
            policy,
            ..NelderMead::default()
        };
        let qualified = config
            .minimize(
                &quadratic_simplex(),
                quadratic,
                &Session::new(ALGORITHM, "test"),
            )
            .expect("simplex collapse is a reportable partial result");
        assert_eq!(
            qualified.value.termination,
            OptimizationTermination::SimplexCollapsed
        );
        assert!(qualified.report.contains(IssueCode::StepSizeCollapsed));
        assert!(qualified.is_compromised());
        assert_eq!(qualified.value.iterations, 0);
        assert_eq!(qualified.value.evaluations, 3);
    }

    #[test]
    fn flat_objective_does_not_converge_from_objective_spread_alone() {
        let config = NelderMead {
            max_iterations: 1,
            ..NelderMead::default()
        };
        let qualified = config
            .minimize(
                &quadratic_simplex(),
                |_| 7.0,
                &Session::new(ALGORITHM, "test"),
            )
            .expect("flat objective returns a qualified partial result");
        assert_eq!(qualified.value.objective_std, 0.0);
        assert_eq!(
            qualified.value.termination,
            OptimizationTermination::MaxIterations
        );
        assert!(qualified.value.simplex_diameter > config.policy.optimizer_parameter_tol);
    }
}
