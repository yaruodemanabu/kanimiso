//! Gaussian-mixture densities and mixture optimal transport.

use crate::error::{Error, Result};
use crate::exact;
use crate::gaussian;
use crate::gaussian::GaussianBarycenterOptions;
use crate::result::TransportPlan;
use crate::validate;
use faer::{Mat, Side};

/// A finite Gaussian mixture with dense `faer` parameters.
#[derive(Clone, Debug)]
pub struct GaussianMixture {
    /// Component means, one component per row.
    pub means: Mat<f64>,
    /// Component covariance matrices.
    pub covariances: Vec<Mat<f64>>,
    /// Non-negative component masses.
    pub weights: Vec<f64>,
}

impl GaussianMixture {
    /// Validate and construct a Gaussian mixture.
    pub fn new(means: Mat<f64>, covariances: Vec<Mat<f64>>, weights: Vec<f64>) -> Result<Self> {
        validate::samples(&means, "GMM means")?;
        validate::distribution(&weights, "GMM weights")?;
        if means.nrows() != covariances.len() || means.nrows() != weights.len() {
            return Err(Error::ShapeMismatch {
                context: "GMM component counts",
                left: (means.nrows(), covariances.len()),
                right: (weights.len(), means.nrows()),
            });
        }
        for covariance in &covariances {
            validate::samples(covariance, "GMM covariance")?;
            if covariance.nrows() != means.ncols() || covariance.ncols() != means.ncols() {
                return Err(Error::ShapeMismatch {
                    context: "GMM covariance dimensions",
                    left: (covariance.nrows(), covariance.ncols()),
                    right: (means.ncols(), means.ncols()),
                });
            }
            // Force a PSD validation even when only density is not requested.
            gaussian::psd_sqrt(covariance)?;
        }
        Ok(Self {
            means,
            covariances,
            weights,
        })
    }

    /// Number of mixture components.
    pub fn components(&self) -> usize {
        self.means.nrows()
    }

    /// Feature dimension.
    pub fn dimensions(&self) -> usize {
        self.means.ncols()
    }

    fn mean(&self, component: usize) -> Vec<f64> {
        (0..self.dimensions())
            .map(|coordinate| self.means[(component, coordinate)])
            .collect()
    }

    /// Log-density of one component at each sample row.
    pub fn component_log_pdf(&self, samples: &Mat<f64>, component: usize) -> Result<Vec<f64>> {
        if component >= self.components() {
            return Err(Error::InvalidParameter {
                name: "component",
                requirement: "within the GMM component range",
            });
        }
        validate::samples(samples, "GMM density samples")?;
        if samples.ncols() != self.dimensions() {
            return Err(Error::ShapeMismatch {
                context: "GMM density dimensions",
                left: (samples.nrows(), samples.ncols()),
                right: (self.components(), self.dimensions()),
            });
        }
        gaussian_log_pdf(samples, &self.mean(component), &self.covariances[component])
    }

    /// Mixture probability density at each sample row.
    pub fn pdf(&self, samples: &Mat<f64>) -> Result<Vec<f64>> {
        let component_logs = (0..self.components())
            .map(|component| self.component_log_pdf(samples, component))
            .collect::<Result<Vec<_>>>()?;
        Ok((0..samples.nrows())
            .map(|row| {
                (0..self.components())
                    .map(|component| self.weights[component] * component_logs[component][row].exp())
                    .sum()
            })
            .collect())
    }
}

/// Log density of a multivariate Gaussian at each sample row.
pub fn gaussian_log_pdf(
    samples: &Mat<f64>,
    mean: &[f64],
    covariance: &Mat<f64>,
) -> Result<Vec<f64>> {
    validate::samples(samples, "Gaussian density samples")?;
    if samples.ncols() != mean.len()
        || covariance.nrows() != mean.len()
        || covariance.ncols() != mean.len()
    {
        return Err(Error::ShapeMismatch {
            context: "Gaussian density parameters",
            left: (samples.nrows(), samples.ncols()),
            right: (covariance.nrows(), covariance.ncols()),
        });
    }
    let eigenvalues = covariance
        .self_adjoint_eigenvalues(Side::Lower)
        .map_err(|_| Error::LinearAlgebra {
            operation: "Gaussian covariance eigenvalues",
        })?;
    if eigenvalues.iter().any(|&value| value <= 0.0) {
        return Err(Error::InvalidParameter {
            name: "Gaussian covariance",
            requirement: "positive definite for density evaluation",
        });
    }
    let log_determinant = eigenvalues.iter().map(|value| value.ln()).sum::<f64>();
    let inverse_root = gaussian::psd_inverse_sqrt(covariance)?;
    let normalization = mean.len() as f64 * std::f64::consts::TAU.ln() + log_determinant;
    Ok((0..samples.nrows())
        .map(|row| {
            let difference = (0..samples.ncols())
                .map(|column| samples[(row, column)] - mean[column])
                .collect::<Vec<_>>();
            let squared = (0..samples.ncols())
                .map(|i| {
                    (0..samples.ncols())
                        .map(|j| inverse_root[(i, j)] * difference[j])
                        .sum::<f64>()
                        .powi(2)
                })
                .sum::<f64>();
            -0.5 * (normalization + squared)
        })
        .collect())
}

/// Gaussian probability density at each sample row.
pub fn gaussian_pdf(samples: &Mat<f64>, mean: &[f64], covariance: &Mat<f64>) -> Result<Vec<f64>> {
    Ok(gaussian_log_pdf(samples, mean, covariance)?
        .into_iter()
        .map(f64::exp)
        .collect())
}

/// Pairwise squared `W₂` costs between Gaussian mixture components.
pub fn component_cost(source: &GaussianMixture, target: &GaussianMixture) -> Result<Mat<f64>> {
    if source.dimensions() != target.dimensions() {
        return Err(Error::ShapeMismatch {
            context: "GMM feature dimensions",
            left: (source.components(), source.dimensions()),
            right: (target.components(), target.dimensions()),
        });
    }
    let mut cost = Mat::<f64>::zeros(source.components(), target.components());
    for i in 0..source.components() {
        let source_mean = source.mean(i);
        for j in 0..target.components() {
            let distance = gaussian::bures_wasserstein_distance(
                &source_mean,
                &target.mean(j),
                &source.covariances[i],
                &target.covariances[j],
            )?;
            cost[(i, j)] = distance * distance;
        }
    }
    Ok(cost)
}

/// Exact optimal transport plan between Gaussian mixture components.
pub fn gmm_ot_plan(source: &GaussianMixture, target: &GaussianMixture) -> Result<TransportPlan> {
    let cost = component_cost(source, target)?;
    exact::emd(&source.weights, &target.weights, &cost)
}

/// Gaussian mixture optimal transport objective.
pub fn gmm_ot_loss(source: &GaussianMixture, target: &GaussianMixture) -> Result<f64> {
    Ok(gmm_ot_plan(source, target)?.value)
}

/// Apply the deterministic barycentric Gaussian-mixture transport map.
pub fn apply_barycentric_map(
    samples: &Mat<f64>,
    source: &GaussianMixture,
    target: &GaussianMixture,
    supplied_plan: Option<&Mat<f64>>,
) -> Result<Mat<f64>> {
    validate::samples(samples, "GMM map samples")?;
    if samples.ncols() != source.dimensions() || source.dimensions() != target.dimensions() {
        return Err(Error::ShapeMismatch {
            context: "GMM map feature dimensions",
            left: (samples.nrows(), samples.ncols()),
            right: (source.components(), source.dimensions()),
        });
    }
    let owned_plan;
    let plan = match supplied_plan {
        Some(plan) => plan,
        None => {
            owned_plan = gmm_ot_plan(source, target)?.plan;
            &owned_plan
        }
    };
    if plan.nrows() != source.components() || plan.ncols() != target.components() {
        return Err(Error::ShapeMismatch {
            context: "GMM transport plan",
            left: (plan.nrows(), plan.ncols()),
            right: (source.components(), target.components()),
        });
    }
    let component_logs = (0..source.components())
        .map(|component| source.component_log_pdf(samples, component))
        .collect::<Result<Vec<_>>>()?;
    let mappings = (0..source.components())
        .map(|i| {
            (0..target.components())
                .map(|j| {
                    gaussian::bures_wasserstein_mapping(
                        &source.mean(i),
                        &target.mean(j),
                        &source.covariances[i],
                        &target.covariances[j],
                    )
                })
                .collect::<Result<Vec<_>>>()
        })
        .collect::<Result<Vec<_>>>()?;
    let mut output = Mat::<f64>::zeros(samples.nrows(), samples.ncols());
    for row in 0..samples.nrows() {
        let maximum = component_logs
            .iter()
            .map(|values| values[row])
            .fold(f64::NEG_INFINITY, f64::max);
        let scaled_density = component_logs
            .iter()
            .enumerate()
            .map(|(component, values)| source.weights[component] * (values[row] - maximum).exp())
            .collect::<Vec<_>>();
        let mixture_density = scaled_density.iter().sum::<f64>();
        if mixture_density <= 0.0 {
            return Err(Error::Infeasible {
                context: "GMM posterior density vanished",
            });
        }
        let sample = (0..samples.ncols())
            .map(|column| samples[(row, column)])
            .collect::<Vec<_>>();
        for i in 0..source.components() {
            if source.weights[i] == 0.0 {
                continue;
            }
            let posterior = scaled_density[i] / mixture_density;
            for j in 0..target.components() {
                if plan[(i, j)] == 0.0 {
                    continue;
                }
                let conditional = plan[(i, j)] / source.weights[i];
                let mapped = mappings[i][j].apply(&sample)?;
                for coordinate in 0..samples.ncols() {
                    output[(row, coordinate)] += posterior * conditional * mapped[coordinate];
                }
            }
        }
    }
    Ok(output)
}

/// Density of the GMM-OT coupling on pairs of source and target points.
///
/// The exact Gaussian component coupling is concentrated on affine transport
/// maps. As in POT, `tolerance` thickens each map support so the density can be
/// represented on a finite point grid.
pub fn gmm_ot_plan_density(
    source_samples: &Mat<f64>,
    target_samples: &Mat<f64>,
    source: &GaussianMixture,
    target: &GaussianMixture,
    supplied_plan: Option<&Mat<f64>>,
    tolerance: f64,
) -> Result<Mat<f64>> {
    validate::samples(source_samples, "GMM plan-density source samples")?;
    validate::samples(target_samples, "GMM plan-density target samples")?;
    if source_samples.ncols() != target_samples.ncols()
        || source_samples.ncols() != source.dimensions()
        || source.dimensions() != target.dimensions()
    {
        return Err(Error::ShapeMismatch {
            context: "GMM plan-density dimensions",
            left: (source_samples.nrows(), source_samples.ncols()),
            right: (target_samples.nrows(), target_samples.ncols()),
        });
    }
    validate::finite_positive(tolerance, "tolerance", "finite and strictly positive")?;
    let owned_plan;
    let plan = match supplied_plan {
        Some(plan) => plan,
        None => {
            owned_plan = gmm_ot_plan(source, target)?.plan;
            &owned_plan
        }
    };
    if plan.nrows() != source.components() || plan.ncols() != target.components() {
        return Err(Error::ShapeMismatch {
            context: "GMM plan-density transport plan",
            left: (plan.nrows(), plan.ncols()),
            right: (source.components(), target.components()),
        });
    }
    for j in 0..plan.ncols() {
        for i in 0..plan.nrows() {
            if !plan[(i, j)].is_finite() || plan[(i, j)] < 0.0 {
                return Err(Error::InvalidWeight {
                    index: i * plan.ncols() + j,
                    value: plan[(i, j)],
                });
            }
        }
    }
    let densities = (0..source.components())
        .map(|component| {
            gaussian_pdf(
                source_samples,
                &source.mean(component),
                &source.covariances[component],
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let mappings = (0..source.components())
        .map(|i| {
            (0..target.components())
                .map(|j| {
                    gaussian::bures_wasserstein_mapping(
                        &source.mean(i),
                        &target.mean(j),
                        &source.covariances[i],
                        &target.covariances[j],
                    )
                })
                .collect::<Result<Vec<_>>>()
        })
        .collect::<Result<Vec<_>>>()?;
    let mut density = Mat::<f64>::zeros(source_samples.nrows(), target_samples.nrows());
    for source_row in 0..source_samples.nrows() {
        let point = (0..source_samples.ncols())
            .map(|coordinate| source_samples[(source_row, coordinate)])
            .collect::<Vec<_>>();
        for i in 0..source.components() {
            for j in 0..target.components() {
                if plan[(i, j)] == 0.0 {
                    continue;
                }
                let mapped = mappings[i][j].apply(&point)?;
                for target_row in 0..target_samples.nrows() {
                    let distance = (0..target_samples.ncols())
                        .map(|coordinate| {
                            (mapped[coordinate] - target_samples[(target_row, coordinate)]).powi(2)
                        })
                        .sum::<f64>()
                        .sqrt();
                    if distance < tolerance {
                        density[(source_row, target_row)] +=
                            plan[(i, j)] * densities[i][source_row];
                    }
                }
            }
        }
    }
    Ok(density)
}

/// Component selection used by the GMM barycenter fixed-point update.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GmmBarycentricProjection {
    /// Average selected component means and covariances in parameter space.
    Euclidean,
    /// Compute a Bures-Wasserstein barycenter of selected components.
    Bures,
}

/// Options for the fixed-point Gaussian-mixture barycenter.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GmmBarycenterOptions {
    /// Number of transport-and-projection iterations.
    pub iterations: usize,
    /// Component selection rule.
    pub projection: GmmBarycentricProjection,
    /// Inner Gaussian barycenter stopping options.
    pub gaussian: GaussianBarycenterOptions,
}

impl Default for GmmBarycenterOptions {
    fn default() -> Self {
        Self {
            iterations: 100,
            projection: GmmBarycentricProjection::Euclidean,
            gaussian: GaussianBarycenterOptions::default(),
        }
    }
}

/// Gaussian-mixture barycenter and iteration diagnostics.
#[derive(Clone, Debug)]
pub struct GmmBarycenter {
    /// Resulting mixture; component weights remain fixed at their initial values.
    pub mixture: GaussianMixture,
    /// Number of completed fixed-point iterations.
    pub iterations: usize,
}

fn selected_gaussian(
    mixture: &GaussianMixture,
    plan: &Mat<f64>,
    row: usize,
    row_mass: f64,
    projection: GmmBarycentricProjection,
    options: GaussianBarycenterOptions,
) -> Result<(Vec<f64>, Mat<f64>)> {
    let conditional = (0..mixture.components())
        .map(|component| plan[(row, component)] / row_mass)
        .collect::<Vec<_>>();
    match projection {
        GmmBarycentricProjection::Euclidean => {
            let mean = (0..mixture.dimensions())
                .map(|coordinate| {
                    (0..mixture.components())
                        .map(|component| {
                            conditional[component] * mixture.means[(component, coordinate)]
                        })
                        .sum()
                })
                .collect::<Vec<_>>();
            let covariance =
                Mat::<f64>::from_fn(mixture.dimensions(), mixture.dimensions(), |i, j| {
                    (0..mixture.components())
                        .map(|component| {
                            conditional[component] * mixture.covariances[component][(i, j)]
                        })
                        .sum()
                });
            Ok((mean, covariance))
        }
        GmmBarycentricProjection::Bures => {
            let means = (0..mixture.components())
                .map(|component| mixture.mean(component))
                .collect::<Vec<_>>();
            let selected = gaussian::bures_wasserstein_barycenter(
                &means,
                &mixture.covariances,
                Some(&conditional),
                options,
            )?;
            Ok((selected.mean, selected.covariance))
        }
    }
}

/// Fixed-point Gaussian-mixture OT barycenter with fixed component weights.
pub fn gmm_barycenter_fixed_point(
    mixtures: &[GaussianMixture],
    initial: &GaussianMixture,
    mixture_weights: Option<&[f64]>,
    options: GmmBarycenterOptions,
) -> Result<GmmBarycenter> {
    if mixtures.is_empty() {
        return Err(Error::EmptyInput {
            name: "GMM barycenter mixtures",
        });
    }
    if options.iterations == 0 {
        return Err(Error::InvalidParameter {
            name: "iterations",
            requirement: "positive",
        });
    }
    if initial.weights.iter().any(|&weight| weight <= 0.0) {
        return Err(Error::InvalidParameter {
            name: "initial GMM component weights",
            requirement: "strictly positive",
        });
    }
    for mixture in mixtures {
        if mixture.dimensions() != initial.dimensions() {
            return Err(Error::ShapeMismatch {
                context: "GMM barycenter dimensions",
                left: (initial.components(), initial.dimensions()),
                right: (mixture.components(), mixture.dimensions()),
            });
        }
    }
    let weights_storage;
    let mixture_weights = match mixture_weights {
        Some(weights) => weights,
        None => {
            weights_storage = validate::uniform(mixtures.len())?;
            &weights_storage
        }
    };
    if mixture_weights.len() != mixtures.len() {
        return Err(Error::ShapeMismatch {
            context: "GMM barycenter coefficients",
            left: (mixture_weights.len(), 1),
            right: (mixtures.len(), 1),
        });
    }
    validate::distribution(mixture_weights, "GMM barycenter coefficients")?;
    let mut current = initial.clone();
    for _ in 0..options.iterations {
        let plans = mixtures
            .iter()
            .map(|mixture| gmm_ot_plan(&current, mixture).map(|result| result.plan))
            .collect::<Result<Vec<_>>>()?;
        let mut means = Mat::<f64>::zeros(current.components(), current.dimensions());
        let mut covariances = Vec::with_capacity(current.components());
        for component in 0..current.components() {
            let selections = mixtures
                .iter()
                .zip(&plans)
                .map(|(mixture, plan)| {
                    selected_gaussian(
                        mixture,
                        plan,
                        component,
                        current.weights[component],
                        options.projection,
                        options.gaussian,
                    )
                })
                .collect::<Result<Vec<_>>>()?;
            let selected_means = selections
                .iter()
                .map(|selection| selection.0.clone())
                .collect::<Vec<_>>();
            let selected_covariances = selections
                .iter()
                .map(|selection| selection.1.clone())
                .collect::<Vec<_>>();
            let barycenter = gaussian::bures_wasserstein_barycenter(
                &selected_means,
                &selected_covariances,
                Some(mixture_weights),
                options.gaussian,
            )?;
            for coordinate in 0..current.dimensions() {
                means[(component, coordinate)] = barycenter.mean[coordinate];
            }
            covariances.push(barycenter.covariance);
        }
        current = GaussianMixture::new(means, covariances, current.weights.clone())?;
    }
    Ok(GmmBarycenter {
        mixture: current,
        iterations: options.iterations,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn one_component(mean: f64, variance: f64) -> GaussianMixture {
        GaussianMixture::new(
            Mat::<f64>::from_fn(1, 1, |_, _| mean),
            vec![Mat::<f64>::from_fn(1, 1, |_, _| variance)],
            vec![1.0],
        )
        .unwrap()
    }

    #[test]
    fn standard_normal_density_matches_constant() {
        let samples = Mat::<f64>::from_fn(1, 1, |_, _| 0.0);
        let density =
            gaussian_pdf(&samples, &[0.0], &Mat::<f64>::from_fn(1, 1, |_, _| 1.0)).unwrap();
        assert!((density[0] - 1.0 / std::f64::consts::TAU.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn one_component_loss_is_gaussian_wasserstein() {
        let source = one_component(0.0, 4.0);
        let target = one_component(2.0, 9.0);
        assert!((gmm_ot_loss(&source, &target).unwrap() - 5.0).abs() < 1e-12);
    }

    #[test]
    fn one_component_map_matches_affine_gaussian_map() {
        let source = one_component(1.0, 4.0);
        let target = one_component(5.0, 9.0);
        let samples = Mat::<f64>::from_fn(2, 1, |i, _| [1.0, 3.0][i]);
        let mapped = apply_barycentric_map(&samples, &source, &target, None).unwrap();
        assert!((mapped[(0, 0)] - 5.0).abs() < 1e-12);
        assert!((mapped[(1, 0)] - 8.0).abs() < 1e-12);
    }

    #[test]
    fn one_component_plan_density_lies_on_affine_map() {
        let source = one_component(0.0, 1.0);
        let target = one_component(1.0, 4.0);
        let source_samples = Mat::<f64>::from_fn(3, 1, |i, _| [-1.0, 0.0, 1.0][i]);
        let target_samples = Mat::<f64>::from_fn(3, 1, |i, _| [-1.0, 1.0, 3.0][i]);
        let density = gmm_ot_plan_density(
            &source_samples,
            &target_samples,
            &source,
            &target,
            None,
            1e-2,
        )
        .unwrap();
        assert!((density[(0, 0)] - 0.24197072451914337).abs() < 1e-12);
        assert!((density[(1, 1)] - 0.3989422804014327).abs() < 1e-12);
        assert!((density[(2, 2)] - 0.24197072451914337).abs() < 1e-12);
        assert_eq!(density[(0, 1)], 0.0);
    }

    #[test]
    fn gmm_barycenter_matches_symmetric_one_dimensional_case() {
        let first = GaussianMixture::new(
            Mat::<f64>::from_fn(2, 1, |i, _| [0.0, 4.0][i]),
            vec![
                Mat::<f64>::from_fn(1, 1, |_, _| 1.0),
                Mat::<f64>::from_fn(1, 1, |_, _| 4.0),
            ],
            vec![0.5, 0.5],
        )
        .unwrap();
        let second = GaussianMixture::new(
            Mat::<f64>::from_fn(2, 1, |i, _| [2.0, 6.0][i]),
            vec![
                Mat::<f64>::from_fn(1, 1, |_, _| 9.0),
                Mat::<f64>::from_fn(1, 1, |_, _| 1.0),
            ],
            vec![0.5, 0.5],
        )
        .unwrap();
        let initial = GaussianMixture::new(
            Mat::<f64>::from_fn(2, 1, |i, _| [1.0, 5.0][i]),
            vec![
                Mat::<f64>::from_fn(1, 1, |_, _| 2.0),
                Mat::<f64>::from_fn(1, 1, |_, _| 2.0),
            ],
            vec![0.5, 0.5],
        )
        .unwrap();
        let result = gmm_barycenter_fixed_point(
            &[first, second],
            &initial,
            Some(&[0.25, 0.75]),
            GmmBarycenterOptions {
                iterations: 2,
                ..GmmBarycenterOptions::default()
            },
        )
        .unwrap();
        assert!((result.mixture.means[(0, 0)] - 1.5).abs() < 1e-12);
        assert!((result.mixture.means[(1, 0)] - 5.5).abs() < 1e-12);
        assert!((result.mixture.covariances[0][(0, 0)] - 6.25).abs() < 1e-7);
        assert!((result.mixture.covariances[1][(0, 0)] - 1.5625).abs() < 1e-7);
    }
}
