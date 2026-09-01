//! Projection-based Wasserstein distances and sliced transport plans.

use crate::error::{Error, Result};
use crate::exact;
use crate::metrics::{self, Metric};
use crate::result::{SolverStatus, TransportPlan};
use crate::validate;
use faer::Mat;

/// Options for Monte-Carlo sliced Wasserstein calculations.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SlicedOptions {
    /// Number of random unit directions.
    pub projections: usize,
    /// Wasserstein exponent, at least one.
    pub p: f64,
    /// Deterministic pseudo-random seed.
    pub seed: u64,
}

impl Default for SlicedOptions {
    fn default() -> Self {
        Self {
            projections: 50,
            p: 2.0,
            seed: 0,
        }
    }
}

/// Projection-induced couplings and their original-space costs.
#[derive(Clone, Debug)]
pub struct SlicedPlans {
    /// Dense coupling induced by each projection.
    pub plans: Vec<Mat<f64>>,
    /// Original-space transport cost of each coupling.
    pub costs: Vec<f64>,
    /// Projection directions used to construct the couplings.
    pub projections: Mat<f64>,
}

/// Metric and temperature options for an expected sliced plan.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExpectedSlicedPlanOptions {
    /// Metric used to score projection-induced plans in the original space.
    pub metric: Metric,
    /// Minkowski exponent and one-dimensional transport exponent.
    pub p: f64,
    /// Inverse temperature; zero selects uniform projection weights.
    pub beta: f64,
}

impl Default for ExpectedSlicedPlanOptions {
    fn default() -> Self {
        Self {
            metric: Metric::SquaredEuclidean,
            p: 2.0,
            beta: 0.0,
        }
    }
}

fn validate_options(options: SlicedOptions) -> Result<()> {
    if options.projections == 0 {
        return Err(Error::InvalidParameter {
            name: "projections",
            requirement: "positive",
        });
    }
    if !options.p.is_finite() || options.p < 1.0 {
        return Err(Error::InvalidParameter {
            name: "p",
            requirement: "finite and at least one",
        });
    }
    Ok(())
}

#[derive(Clone, Debug)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e3779b97f4a7c15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
        value ^ (value >> 31)
    }

    fn uniform_open(&mut self) -> f64 {
        let bits = self.next_u64() >> 11;
        ((bits as f64) + 0.5) * (1.0 / ((1_u64 << 53) as f64))
    }

    fn normal_pair(&mut self) -> (f64, f64) {
        let radius = (-2.0 * self.uniform_open().ln()).sqrt();
        let angle = std::f64::consts::TAU * self.uniform_open();
        (radius * angle.cos(), radius * angle.sin())
    }
}

/// Generate deterministic, normally distributed unit projection directions.
pub fn random_projections(dimensions: usize, count: usize, seed: u64) -> Result<Mat<f64>> {
    if dimensions == 0 {
        return Err(Error::InvalidParameter {
            name: "dimensions",
            requirement: "positive",
        });
    }
    if count == 0 {
        return Err(Error::InvalidParameter {
            name: "count",
            requirement: "positive",
        });
    }
    let mut rng = SplitMix64::new(seed);
    let mut output = Mat::<f64>::zeros(dimensions, count);
    for column in 0..count {
        let mut row = 0;
        while row < dimensions {
            let (first, second) = rng.normal_pair();
            output[(row, column)] = first;
            if row + 1 < dimensions {
                output[(row + 1, column)] = second;
            }
            row += 2;
        }
        let norm = (0..dimensions)
            .map(|index| output[(index, column)].powi(2))
            .sum::<f64>()
            .sqrt();
        if norm == 0.0 || !norm.is_finite() {
            return Err(Error::Infeasible {
                context: "random projection has zero norm",
            });
        }
        for index in 0..dimensions {
            output[(index, column)] /= norm;
        }
    }
    Ok(output)
}

fn validate_samples(source: &Mat<f64>, target: &Mat<f64>) -> Result<()> {
    validate::samples(source, "source samples")?;
    validate::samples(target, "target samples")?;
    if source.ncols() != target.ncols() {
        return Err(Error::ShapeMismatch {
            context: "sliced Wasserstein feature dimensions",
            left: (source.nrows(), source.ncols()),
            right: (target.nrows(), target.ncols()),
        });
    }
    Ok(())
}

fn project(samples: &Mat<f64>, directions: &Mat<f64>, direction: usize) -> Vec<f64> {
    (0..samples.nrows())
        .map(|row| {
            (0..samples.ncols())
                .map(|column| samples[(row, column)] * directions[(column, direction)])
                .sum()
        })
        .collect()
}

fn weights<'a>(
    supplied: Option<&'a [f64]>,
    rows: usize,
    storage: &'a mut Vec<f64>,
) -> Result<&'a [f64]> {
    match supplied {
        Some(weights) => {
            validate::distribution(weights, "sliced weights")?;
            if weights.len() != rows {
                return Err(Error::ShapeMismatch {
                    context: "sliced samples and weights",
                    left: (rows, 1),
                    right: (weights.len(), 1),
                });
            }
            Ok(weights)
        }
        None => {
            *storage = validate::uniform(rows)?;
            Ok(storage)
        }
    }
}

/// Monte-Carlo approximation of the sliced `p`-Wasserstein distance.
pub fn sliced_wasserstein(
    source: &Mat<f64>,
    target: &Mat<f64>,
    source_weights: Option<&[f64]>,
    target_weights: Option<&[f64]>,
    options: SlicedOptions,
) -> Result<f64> {
    validate_samples(source, target)?;
    validate_options(options)?;
    let mut source_storage = Vec::new();
    let mut target_storage = Vec::new();
    let source_weights = weights(source_weights, source.nrows(), &mut source_storage)?;
    let target_weights = weights(target_weights, target.nrows(), &mut target_storage)?;
    validate::balanced_distributions(source_weights, target_weights)?;
    let directions = random_projections(source.ncols(), options.projections, options.seed)?;
    sliced_wasserstein_with_projections(
        source,
        target,
        source_weights,
        target_weights,
        &directions,
        options.p,
    )
}

/// Sliced Wasserstein using caller-supplied unit or non-unit directions.
pub fn sliced_wasserstein_with_projections(
    source: &Mat<f64>,
    target: &Mat<f64>,
    source_weights: &[f64],
    target_weights: &[f64],
    projections: &Mat<f64>,
    p: f64,
) -> Result<f64> {
    validate_samples(source, target)?;
    validate::balanced_distributions(source_weights, target_weights)?;
    if source_weights.len() != source.nrows() || target_weights.len() != target.nrows() {
        return Err(Error::ShapeMismatch {
            context: "sliced samples and weights",
            left: (source.nrows(), target.nrows()),
            right: (source_weights.len(), target_weights.len()),
        });
    }
    if projections.nrows() != source.ncols() || projections.ncols() == 0 {
        return Err(Error::ShapeMismatch {
            context: "sliced projections",
            left: (projections.nrows(), projections.ncols()),
            right: (source.ncols(), 1),
        });
    }
    validate::finite_positive(p, "p", "finite and strictly positive")?;
    let mut sum = 0.0;
    for direction in 0..projections.ncols() {
        let source_projection = project(source, projections, direction);
        let target_projection = project(target, projections, direction);
        sum += exact::wasserstein_1d(
            &source_projection,
            &target_projection,
            Some(source_weights),
            Some(target_weights),
            p,
        )?;
    }
    Ok((sum / projections.ncols() as f64).powf(1.0 / p))
}

/// Maximum projected `p`-Wasserstein distance over random directions.
pub fn max_sliced_wasserstein(
    source: &Mat<f64>,
    target: &Mat<f64>,
    source_weights: Option<&[f64]>,
    target_weights: Option<&[f64]>,
    options: SlicedOptions,
) -> Result<f64> {
    validate_samples(source, target)?;
    validate_options(options)?;
    let mut source_storage = Vec::new();
    let mut target_storage = Vec::new();
    let source_weights = weights(source_weights, source.nrows(), &mut source_storage)?;
    let target_weights = weights(target_weights, target.nrows(), &mut target_storage)?;
    validate::balanced_distributions(source_weights, target_weights)?;
    let directions = random_projections(source.ncols(), options.projections, options.seed)?;
    let mut maximum = 0.0_f64;
    for direction in 0..directions.ncols() {
        let source_projection = project(source, &directions, direction);
        let target_projection = project(target, &directions, direction);
        maximum = maximum.max(exact::wasserstein_1d(
            &source_projection,
            &target_projection,
            Some(source_weights),
            Some(target_weights),
            options.p,
        )?);
    }
    Ok(maximum.powf(1.0 / options.p))
}

fn sliced_plan_metric(metric: Metric, p: f64) -> Result<(Metric, f64)> {
    if !p.is_finite() || p < 1.0 {
        return Err(Error::InvalidParameter {
            name: "p",
            requirement: "finite and at least one",
        });
    }
    match metric {
        Metric::SquaredEuclidean => Ok((metric, p)),
        Metric::Euclidean => Ok((metric, 2.0)),
        Metric::Manhattan => Ok((metric, 1.0)),
        Metric::Minkowski(_) => Ok((Metric::Minkowski(p), p)),
        _ => Err(Error::InvalidParameter {
            name: "sliced plan metric",
            requirement: "squared Euclidean, Euclidean, Manhattan, or Minkowski",
        }),
    }
}

fn dense_plan_cost(plan: &Mat<f64>, cost: &Mat<f64>) -> f64 {
    let mut value = 0.0;
    for i in 0..plan.nrows() {
        for j in 0..plan.ncols() {
            value += plan[(i, j)] * cost[(i, j)];
        }
    }
    value
}

fn dense_plan_residual(plan: &Mat<f64>, source: &[f64], target: &[f64]) -> f64 {
    let mut residual = 0.0_f64;
    for i in 0..plan.nrows() {
        let marginal = (0..plan.ncols()).map(|j| plan[(i, j)]).sum::<f64>();
        residual = residual.max((marginal - source[i]).abs());
    }
    for j in 0..plan.ncols() {
        let marginal = (0..plan.nrows()).map(|i| plan[(i, j)]).sum::<f64>();
        residual = residual.max((marginal - target[j]).abs());
    }
    residual
}

/// Construct every random projection-induced coupling and original-space cost.
pub fn sliced_plans(
    source: &Mat<f64>,
    target: &Mat<f64>,
    source_weights: Option<&[f64]>,
    target_weights: Option<&[f64]>,
    options: SlicedOptions,
    metric: Metric,
) -> Result<SlicedPlans> {
    validate_samples(source, target)?;
    validate_options(options)?;
    let projections = random_projections(source.ncols(), options.projections, options.seed)?;
    sliced_plans_with_projections(
        source,
        target,
        source_weights,
        target_weights,
        &projections,
        metric,
        options.p,
    )
}

/// Construct projection-induced couplings using caller-supplied directions.
pub fn sliced_plans_with_projections(
    source: &Mat<f64>,
    target: &Mat<f64>,
    source_weights: Option<&[f64]>,
    target_weights: Option<&[f64]>,
    projections: &Mat<f64>,
    metric: Metric,
    p: f64,
) -> Result<SlicedPlans> {
    validate_samples(source, target)?;
    validate::samples(projections, "sliced projections")?;
    if projections.nrows() != source.ncols() {
        return Err(Error::ShapeMismatch {
            context: "sliced projections",
            left: (projections.nrows(), projections.ncols()),
            right: (source.ncols(), 1),
        });
    }
    let (metric, projection_power) = sliced_plan_metric(metric, p)?;
    let mut source_storage = Vec::new();
    let mut target_storage = Vec::new();
    let source_weights = weights(source_weights, source.nrows(), &mut source_storage)?;
    let target_weights = weights(target_weights, target.nrows(), &mut target_storage)?;
    validate::balanced_distributions(source_weights, target_weights)?;
    let original_cost = metrics::pairwise(source, target, metric)?;
    let mut plans = Vec::with_capacity(projections.ncols());
    let mut costs = Vec::with_capacity(projections.ncols());
    for direction in 0..projections.ncols() {
        let source_projection = project(source, projections, direction);
        let target_projection = project(target, projections, direction);
        let plan = exact::emd_1d(
            &source_projection,
            &target_projection,
            source_weights,
            target_weights,
            projection_power,
        )?;
        costs.push(dense_plan_cost(&plan.plan, &original_cost));
        plans.push(plan.plan);
    }
    Ok(SlicedPlans {
        plans,
        costs,
        projections: projections.clone(),
    })
}

/// Average of exact one-dimensional plans over random projections.
pub fn expected_sliced_plan(
    source: &Mat<f64>,
    target: &Mat<f64>,
    source_weights: Option<&[f64]>,
    target_weights: Option<&[f64]>,
    options: SlicedOptions,
) -> Result<Mat<f64>> {
    validate_samples(source, target)?;
    validate_options(options)?;
    let projections = random_projections(source.ncols(), options.projections, options.seed)?;
    Ok(expected_sliced_plan_with_projections(
        source,
        target,
        source_weights,
        target_weights,
        &projections,
        ExpectedSlicedPlanOptions {
            p: options.p,
            ..ExpectedSlicedPlanOptions::default()
        },
    )?
    .plan)
}

/// Compute a temperature-weighted expected sliced plan and its original-space cost.
///
/// `beta = 0` gives a uniform average. Positive `beta` increasingly favors
/// projection-induced plans with smaller original-space cost.
pub fn expected_sliced_plan_with_projections(
    source: &Mat<f64>,
    target: &Mat<f64>,
    source_weights: Option<&[f64]>,
    target_weights: Option<&[f64]>,
    projections: &Mat<f64>,
    options: ExpectedSlicedPlanOptions,
) -> Result<TransportPlan> {
    if !options.beta.is_finite() {
        return Err(Error::InvalidParameter {
            name: "beta",
            requirement: "finite",
        });
    }
    let mut source_storage = Vec::new();
    let mut target_storage = Vec::new();
    let source_weights = weights(source_weights, source.nrows(), &mut source_storage)?;
    let target_weights = weights(target_weights, target.nrows(), &mut target_storage)?;
    validate::balanced_distributions(source_weights, target_weights)?;
    let candidates = sliced_plans_with_projections(
        source,
        target,
        Some(source_weights),
        Some(target_weights),
        projections,
        options.metric,
        options.p,
    )?;
    let plan_weights = if options.beta == 0.0 {
        vec![1.0 / candidates.plans.len() as f64; candidates.plans.len()]
    } else {
        let logits = candidates
            .costs
            .iter()
            .map(|&cost| -options.beta * cost)
            .collect::<Vec<_>>();
        if logits.iter().any(|value| !value.is_finite()) {
            return Err(Error::InvalidParameter {
                name: "beta",
                requirement: "finite at the observed cost scale",
            });
        }
        let maximum = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let normalizer = logits
            .iter()
            .map(|&value| (value - maximum).exp())
            .sum::<f64>();
        if !normalizer.is_finite() || normalizer <= 0.0 {
            return Err(Error::Infeasible {
                context: "expected sliced plan weights could not be normalized",
            });
        }
        logits
            .iter()
            .map(|&value| (value - maximum).exp() / normalizer)
            .collect()
    };
    let mut average = Mat::<f64>::zeros(source.nrows(), target.nrows());
    for (plan, &weight) in candidates.plans.iter().zip(&plan_weights) {
        for i in 0..average.nrows() {
            for j in 0..average.ncols() {
                average[(i, j)] += weight * plan[(i, j)];
            }
        }
    }
    let original_cost = metrics::pairwise(
        source,
        target,
        sliced_plan_metric(options.metric, options.p)?.0,
    )?;
    let value = dense_plan_cost(&average, &original_cost);
    let residual = dense_plan_residual(&average, source_weights, target_weights);
    Ok(TransportPlan {
        plan: average,
        value,
        potentials: None,
        iterations: candidates.plans.len(),
        residual,
        status: SolverStatus::Converged,
    })
}

/// Plan whose random projection-induced coupling has the lowest original-space cost.
pub fn min_sliced_transport_plan(
    source: &Mat<f64>,
    target: &Mat<f64>,
    source_weights: Option<&[f64]>,
    target_weights: Option<&[f64]>,
    options: SlicedOptions,
) -> Result<TransportPlan> {
    validate_samples(source, target)?;
    validate_options(options)?;
    let directions = random_projections(source.ncols(), options.projections, options.seed)?;
    min_sliced_transport_plan_with_projections(
        source,
        target,
        source_weights,
        target_weights,
        &directions,
        options.p,
    )
}

/// Select a sliced coupling using caller-supplied projection directions.
///
/// Each projection induces a one-dimensional monotone coupling. Candidates
/// are scored using squared Euclidean costs in the original sample space,
/// matching POT's default min-sliced transport-plan quantity.
pub fn min_sliced_transport_plan_with_projections(
    source: &Mat<f64>,
    target: &Mat<f64>,
    source_weights: Option<&[f64]>,
    target_weights: Option<&[f64]>,
    projections: &Mat<f64>,
    p: f64,
) -> Result<TransportPlan> {
    min_sliced_transport_plan_with_metric(
        source,
        target,
        source_weights,
        target_weights,
        projections,
        Metric::SquaredEuclidean,
        p,
    )
}

/// Select a sliced coupling with an explicit original-space metric.
pub fn min_sliced_transport_plan_with_metric(
    source: &Mat<f64>,
    target: &Mat<f64>,
    source_weights: Option<&[f64]>,
    target_weights: Option<&[f64]>,
    projections: &Mat<f64>,
    metric: Metric,
    p: f64,
) -> Result<TransportPlan> {
    let mut source_storage = Vec::new();
    let mut target_storage = Vec::new();
    let source_weights = weights(source_weights, source.nrows(), &mut source_storage)?;
    let target_weights = weights(target_weights, target.nrows(), &mut target_storage)?;
    let candidates = sliced_plans_with_projections(
        source,
        target,
        Some(source_weights),
        Some(target_weights),
        projections,
        metric,
        p,
    )?;
    let best = candidates
        .costs
        .iter()
        .enumerate()
        .min_by(|(_, left), (_, right)| left.total_cmp(right))
        .map(|(index, &cost)| (index, cost))
        .ok_or(Error::Infeasible {
            context: "no sliced projections were generated",
        })?;
    let plan = candidates.plans[best.0].clone();
    let residual = dense_plan_residual(&plan, source_weights, target_weights);
    Ok(TransportPlan {
        plan,
        value: best.1,
        potentials: None,
        iterations: candidates.plans.len(),
        residual,
        status: SolverStatus::Converged,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn projections_have_unit_norm() {
        let projections = random_projections(5, 7, 42).unwrap();
        for j in 0..projections.ncols() {
            let norm = (0..projections.nrows())
                .map(|i| projections[(i, j)].powi(2))
                .sum::<f64>()
                .sqrt();
            assert!((norm - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn sliced_distance_vanishes_on_identical_samples() {
        let samples = Mat::<f64>::from_fn(4, 2, |i, j| (i + j) as f64);
        let options = SlicedOptions {
            projections: 10,
            ..SlicedOptions::default()
        };
        let value = sliced_wasserstein(&samples, &samples, None, None, options).unwrap();
        assert!(value < 1e-12, "value={value}");
        let maximum = max_sliced_wasserstein(&samples, &samples, None, None, options).unwrap();
        assert!(maximum < 1e-12, "maximum={maximum}");
    }

    #[test]
    fn expected_plan_preserves_marginals() {
        let source = Mat::<f64>::from_fn(2, 2, |i, j| (i + j) as f64);
        let target = Mat::<f64>::from_fn(3, 2, |i, j| (i * 2 + j) as f64);
        let plan = expected_sliced_plan(
            &source,
            &target,
            None,
            None,
            SlicedOptions {
                projections: 8,
                ..SlicedOptions::default()
            },
        )
        .unwrap();
        for i in 0..2 {
            let sum = (0..3).map(|j| plan[(i, j)]).sum::<f64>();
            assert!((sum - 0.5).abs() < 1e-12);
        }
        for j in 0..3 {
            let sum = (0..2).map(|i| plan[(i, j)]).sum::<f64>();
            assert!((sum - 1.0 / 3.0).abs() < 1e-12);
        }
    }

    #[test]
    fn minimum_plan_is_scored_in_original_space() {
        let source = Mat::<f64>::from_fn(2, 2, |i, j| [[3.0, 3.0], [1.0, 1.0]][i][j]);
        let target = Mat::<f64>::from_fn(2, 2, |i, j| [[2.0, 2.5], [3.0, 2.0]][i][j]);
        let projections = Mat::<f64>::from_fn(2, 2, |i, j| if i == j { 1.0 } else { 0.0 });
        let candidates = sliced_plans_with_projections(
            &source,
            &target,
            None,
            None,
            &projections,
            Metric::SquaredEuclidean,
            2.0,
        )
        .unwrap();
        assert_eq!(candidates.costs, vec![2.125, 3.125]);
        let result = min_sliced_transport_plan_with_projections(
            &source,
            &target,
            None,
            None,
            &projections,
            2.0,
        )
        .unwrap();
        assert!((result.value - 2.125).abs() < 1e-12);
        assert_eq!(result.plan[(0, 1)], 0.5);
        assert_eq!(result.plan[(1, 0)], 0.5);

        let expected = expected_sliced_plan_with_projections(
            &source,
            &target,
            None,
            None,
            &projections,
            ExpectedSlicedPlanOptions {
                beta: 1.5,
                ..ExpectedSlicedPlanOptions::default()
            },
        )
        .unwrap();
        assert!((expected.value - 2.3074255238063563).abs() < 1e-12);
        assert!((expected.plan[(0, 0)] - 0.09121276190317816).abs() < 1e-12);
        assert!((expected.plan[(0, 1)] - 0.4087872380968218).abs() < 1e-12);
    }
}
