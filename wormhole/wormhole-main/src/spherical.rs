//! Sliced optimal transport for probability measures on spheres.

use crate::circle;
use crate::error::{Error, Result};
use crate::sliced::{self, SlicedOptions};
use crate::validate;
use faer::Mat;

/// Orthonormal two-frames used to project a sphere onto great circles.
#[derive(Clone, Debug)]
pub struct SphereProjections {
    /// First basis vector of each frame, stored by column.
    pub first: Mat<f64>,
    /// Second basis vector of each frame, stored by column.
    pub second: Mat<f64>,
}

impl SphereProjections {
    /// Number of projection frames.
    pub fn len(&self) -> usize {
        self.first.ncols()
    }

    /// Whether there are no projection frames.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Generate deterministic random orthonormal two-frames.
pub fn random_sphere_projections(
    dimensions: usize,
    count: usize,
    seed: u64,
) -> Result<SphereProjections> {
    if dimensions < 2 {
        return Err(Error::InvalidParameter {
            name: "sphere dimensions",
            requirement: "at least two",
        });
    }
    let raw = sliced::random_projections(dimensions, count.saturating_mul(2), seed)?;
    let mut first = Mat::<f64>::zeros(dimensions, count);
    let mut second = Mat::<f64>::zeros(dimensions, count);
    for projection in 0..count {
        let first_column = 2 * projection;
        let second_column = first_column + 1;
        let dot = (0..dimensions)
            .map(|i| raw[(i, first_column)] * raw[(i, second_column)])
            .sum::<f64>();
        let norm = (0..dimensions)
            .map(|i| (raw[(i, second_column)] - dot * raw[(i, first_column)]).powi(2))
            .sum::<f64>()
            .sqrt();
        if norm <= 1e-12 {
            return Err(Error::Infeasible {
                context: "random sphere frame is numerically rank deficient",
            });
        }
        for i in 0..dimensions {
            first[(i, projection)] = raw[(i, first_column)];
            second[(i, projection)] =
                (raw[(i, second_column)] - dot * raw[(i, first_column)]) / norm;
        }
    }
    Ok(SphereProjections { first, second })
}

fn validate_sphere(samples: &Mat<f64>, name: &'static str) -> Result<()> {
    validate::samples(samples, name)?;
    if samples.ncols() < 2 {
        return Err(Error::InvalidParameter {
            name: "sphere dimensions",
            requirement: "at least two",
        });
    }
    for i in 0..samples.nrows() {
        let squared_norm = (0..samples.ncols())
            .map(|j| samples[(i, j)].powi(2))
            .sum::<f64>();
        if (squared_norm - 1.0).abs() > 1e-4 {
            return Err(Error::InvalidParameter {
                name: "sphere samples",
                requirement: "unit norm within 1e-4",
            });
        }
    }
    Ok(())
}

fn validate_projections(samples: &Mat<f64>, projections: &SphereProjections) -> Result<()> {
    if projections.is_empty()
        || projections.first.nrows() != samples.ncols()
        || projections.second.nrows() != samples.ncols()
        || projections.first.ncols() != projections.second.ncols()
    {
        return Err(Error::ShapeMismatch {
            context: "sphere samples and projections",
            left: (samples.ncols(), samples.ncols()),
            right: (projections.first.nrows(), projections.second.nrows()),
        });
    }
    Ok(())
}

fn projected_coordinates(
    samples: &Mat<f64>,
    projections: &SphereProjections,
    projection: usize,
) -> Result<Vec<f64>> {
    let mut output = Vec::with_capacity(samples.nrows());
    for row in 0..samples.nrows() {
        let first = (0..samples.ncols())
            .map(|column| samples[(row, column)] * projections.first[(column, projection)])
            .sum::<f64>();
        let second = (0..samples.ncols())
            .map(|column| samples[(row, column)] * projections.second[(column, projection)])
            .sum::<f64>();
        let norm = (first * first + second * second).sqrt();
        if norm <= 1e-14 {
            return Err(Error::Infeasible {
                context: "sphere point is orthogonal to a projection plane",
            });
        }
        output.push(circle::coordinate(first / norm, second / norm)?);
    }
    Ok(output)
}

/// Spherical sliced-Wasserstein using caller-supplied projection frames.
pub fn sliced_wasserstein_sphere_with_projections(
    source: &Mat<f64>,
    target: &Mat<f64>,
    source_weights: Option<&[f64]>,
    target_weights: Option<&[f64]>,
    projections: &SphereProjections,
    p: f64,
) -> Result<f64> {
    validate_sphere(source, "source sphere samples")?;
    validate_sphere(target, "target sphere samples")?;
    if source.ncols() != target.ncols() {
        return Err(Error::ShapeMismatch {
            context: "sphere feature dimensions",
            left: (source.nrows(), source.ncols()),
            right: (target.nrows(), target.ncols()),
        });
    }
    validate_projections(source, projections)?;
    if !p.is_finite() || p < 1.0 {
        return Err(Error::InvalidParameter {
            name: "p",
            requirement: "finite and at least one",
        });
    }
    let mut objective = 0.0;
    for projection in 0..projections.len() {
        let source_projection = projected_coordinates(source, projections, projection)?;
        let target_projection = projected_coordinates(target, projections, projection)?;
        objective += circle::wasserstein_circle(
            &source_projection,
            &target_projection,
            source_weights,
            target_weights,
            p,
        )?;
    }
    Ok((objective / projections.len() as f64).powf(1.0 / p))
}

/// Monte-Carlo spherical sliced-Wasserstein distance.
pub fn sliced_wasserstein_sphere(
    source: &Mat<f64>,
    target: &Mat<f64>,
    source_weights: Option<&[f64]>,
    target_weights: Option<&[f64]>,
    options: SlicedOptions,
) -> Result<f64> {
    let projections = random_sphere_projections(source.ncols(), options.projections, options.seed)?;
    sliced_wasserstein_sphere_with_projections(
        source,
        target,
        source_weights,
        target_weights,
        &projections,
        options.p,
    )
}

/// Spherical sliced `W₂` from a discrete measure to uniform sphere measure.
pub fn sliced_wasserstein_sphere_uniform(
    source: &Mat<f64>,
    source_weights: Option<&[f64]>,
    projections: &SphereProjections,
) -> Result<f64> {
    validate_sphere(source, "sphere samples")?;
    validate_projections(source, projections)?;
    let mut objective = 0.0;
    for projection in 0..projections.len() {
        let projected = projected_coordinates(source, projections, projection)?;
        objective += circle::semidiscrete_wasserstein2_uniform_circle(&projected, source_weights)?;
    }
    Ok((objective / projections.len() as f64).sqrt())
}

/// Linear spherical sliced OT based on linear circular embeddings.
pub fn linear_sliced_wasserstein_sphere(
    source: &Mat<f64>,
    target: Option<&Mat<f64>>,
    source_weights: Option<&[f64]>,
    target_weights: Option<&[f64]>,
    projections: &SphereProjections,
) -> Result<f64> {
    validate_sphere(source, "source sphere samples")?;
    validate_projections(source, projections)?;
    if let Some(target) = target {
        validate_sphere(target, "target sphere samples")?;
        if target.ncols() != source.ncols() {
            return Err(Error::ShapeMismatch {
                context: "linear spherical feature dimensions",
                left: (source.nrows(), source.ncols()),
                right: (target.nrows(), target.ncols()),
            });
        }
    }
    let mut objective = 0.0;
    for projection in 0..projections.len() {
        let projected_source = projected_coordinates(source, projections, projection)?;
        let projected_target = target
            .map(|samples| projected_coordinates(samples, projections, projection))
            .transpose()?;
        objective += circle::linear_circular_ot(
            &projected_source,
            projected_target.as_deref(),
            source_weights,
            target_weights,
        )?;
    }
    Ok((objective / projections.len() as f64).sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn circle_samples() -> Mat<f64> {
        Mat::<f64>::from_fn(4, 3, |i, j| {
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
            ][i][j]
        })
    }

    #[test]
    fn generated_frames_are_orthonormal() {
        let projections = random_sphere_projections(5, 8, 7).unwrap();
        for j in 0..projections.len() {
            let first_norm = (0..5)
                .map(|i| projections.first[(i, j)].powi(2))
                .sum::<f64>();
            let second_norm = (0..5)
                .map(|i| projections.second[(i, j)].powi(2))
                .sum::<f64>();
            let dot = (0..5)
                .map(|i| projections.first[(i, j)] * projections.second[(i, j)])
                .sum::<f64>();
            assert!((first_norm - 1.0).abs() < 1e-12);
            assert!((second_norm - 1.0).abs() < 1e-12);
            assert!(dot.abs() < 1e-12);
        }
    }

    #[test]
    fn spherical_distances_vanish_on_identity() {
        let samples = circle_samples();
        let options = SlicedOptions {
            projections: 10,
            seed: 4,
            ..SlicedOptions::default()
        };
        let distance = sliced_wasserstein_sphere(&samples, &samples, None, None, options).unwrap();
        assert!(distance < 1e-9, "distance={distance}");
        let projections = random_sphere_projections(3, 10, 4).unwrap();
        let linear =
            linear_sliced_wasserstein_sphere(&samples, Some(&samples), None, None, &projections)
                .unwrap();
        assert!(linear < 1e-12);
    }
}
