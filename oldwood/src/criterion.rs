use crate::numeric::{CompensatedSum, WeightedMoments};
use crate::Result;

/// Runtime classification impurity criterion.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ClassificationCriterion {
    /// Sum of `p * (1 - p)` over classes.
    #[default]
    Gini,
    /// Shannon entropy in bits, with `0 * log2(0) = 0`.
    Entropy,
}

/// Runtime regression impurity criterion.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RegressionCriterion {
    /// Weighted population variance.
    #[default]
    SquaredError,
}

pub(crate) fn classification_impurity(
    criterion: ClassificationCriterion,
    counts: &[f64],
) -> Result<f64> {
    let mut total = CompensatedSum::default();
    for &count in counts {
        total.add(count, "class-weight summation")?;
    }
    let total = total.total("class-weight summation")?;
    if total == 0.0 {
        return Ok(0.0);
    }

    let mut impurity = CompensatedSum::default();
    for &count in counts {
        if count == 0.0 {
            continue;
        }
        let probability = count / total;
        let term = match criterion {
            ClassificationCriterion::Gini => probability * (1.0 - probability),
            ClassificationCriterion::Entropy => -probability * probability.log2(),
        };
        impurity.add(term, "classification impurity summation")?;
    }
    impurity.total("classification impurity summation")
}

pub(crate) fn regression_impurity(criterion: RegressionCriterion, moments: WeightedMoments) -> f64 {
    match criterion {
        RegressionCriterion::SquaredError => moments.variance(),
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::float_cmp)]

    use super::*;

    #[test]
    fn analytical_classification_impurities() {
        let balanced = [1.0, 1.0];
        assert_eq!(
            classification_impurity(ClassificationCriterion::Gini, &balanced).unwrap(),
            0.5
        );
        assert_eq!(
            classification_impurity(ClassificationCriterion::Entropy, &balanced).unwrap(),
            1.0
        );
        assert_eq!(
            classification_impurity(ClassificationCriterion::Entropy, &[1.0; 4]).unwrap(),
            2.0
        );
        assert_eq!(
            classification_impurity(ClassificationCriterion::Gini, &[9.0, 0.0]).unwrap(),
            0.0
        );
    }
}
