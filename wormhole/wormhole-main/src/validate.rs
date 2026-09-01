use crate::error::{Error, Result};
use faer::Mat;

pub(crate) const DEFAULT_TOLERANCE: f64 = 1e-9;

pub(crate) fn distribution(weights: &[f64], name: &'static str) -> Result<f64> {
    if weights.is_empty() {
        return Err(Error::EmptyInput { name });
    }
    let mut mass = 0.0;
    for (index, value) in weights.iter().copied().enumerate() {
        if !value.is_finite() || value < 0.0 {
            return Err(Error::InvalidWeight { index, value });
        }
        mass += value;
    }
    if !mass.is_finite() || mass <= 0.0 {
        return Err(Error::InvalidParameter {
            name,
            requirement: "finite, non-negative, and have positive total mass",
        });
    }
    Ok(mass)
}

pub(crate) fn balanced_distributions(source: &[f64], target: &[f64]) -> Result<f64> {
    let source_mass = distribution(source, "source distribution")?;
    let target_mass = distribution(target, "target distribution")?;
    let scale = source_mass.abs().max(target_mass.abs()).max(1.0);
    if (source_mass - target_mass).abs() > DEFAULT_TOLERANCE * scale {
        return Err(Error::MassMismatch {
            source: source_mass,
            target: target_mass,
        });
    }
    Ok(0.5 * (source_mass + target_mass))
}

pub(crate) fn cost_matrix(cost: &Mat<f64>, source_len: usize, target_len: usize) -> Result<()> {
    if cost.nrows() != source_len || cost.ncols() != target_len {
        return Err(Error::ShapeMismatch {
            context: "cost matrix and marginals",
            left: (cost.nrows(), cost.ncols()),
            right: (source_len, target_len),
        });
    }
    for j in 0..cost.ncols() {
        for i in 0..cost.nrows() {
            let value = cost[(i, j)];
            if !value.is_finite() {
                return Err(Error::InvalidCost {
                    row: i,
                    column: j,
                    value,
                });
            }
        }
    }
    Ok(())
}

pub(crate) fn samples(samples: &Mat<f64>, name: &'static str) -> Result<()> {
    if samples.nrows() == 0 || samples.ncols() == 0 {
        return Err(Error::EmptyInput { name });
    }
    for j in 0..samples.ncols() {
        for i in 0..samples.nrows() {
            let value = samples[(i, j)];
            if !value.is_finite() {
                return Err(Error::InvalidCost {
                    row: i,
                    column: j,
                    value,
                });
            }
        }
    }
    Ok(())
}

pub(crate) fn uniform(length: usize) -> Result<Vec<f64>> {
    if length == 0 {
        return Err(Error::EmptyInput {
            name: "uniform distribution length",
        });
    }
    Ok(vec![1.0 / length as f64; length])
}

pub(crate) fn finite_positive(
    value: f64,
    name: &'static str,
    requirement: &'static str,
) -> Result<()> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(Error::InvalidParameter { name, requirement })
    }
}

pub(crate) fn finite_non_negative(
    value: f64,
    name: &'static str,
    requirement: &'static str,
) -> Result<()> {
    if value.is_finite() && value >= 0.0 {
        Ok(())
    } else {
        Err(Error::InvalidParameter { name, requirement })
    }
}
