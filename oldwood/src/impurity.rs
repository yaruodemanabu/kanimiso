//! Gini / MSE impurities and weighted class counts.

/// Index of `lab` in the sorted `classes` slice.
pub fn class_index(lab: i64, classes: &[i64]) -> Option<usize> {
    classes.iter().position(|&c| c == lab)
}

/// Majority label; ties break toward the smaller class id.
pub fn majority(classes: &[i64], counts: &[f64]) -> i64 {
    let mut best_i = 0usize;
    let mut best = f64::NEG_INFINITY;
    for (i, &c) in counts.iter().enumerate() {
        if c > best + 1e-15 || ((c - best).abs() <= 1e-15 && classes[i] < classes[best_i]) {
            best = c;
            best_i = i;
        }
    }
    classes[best_i]
}

/// Gini impurity \(1 - \sum p_k^2\).
pub fn gini(counts: &[f64]) -> f64 {
    let tot: f64 = counts.iter().sum();
    if tot <= 0.0 {
        return 0.0;
    }
    let mut s = 0.0;
    for &c in counts {
        let p = c / tot;
        s += p * p;
    }
    1.0 - s
}

/// Weighted class histogram over `idx`.
pub fn weighted_counts(y: &[i64], classes: &[i64], idx: &[usize], weights: &[f64]) -> Vec<f64> {
    let mut counts = vec![0.0; classes.len()];
    for &i in idx {
        if let Some(k) = class_index(y[i], classes) {
            counts[k] += weights[i];
        }
    }
    counts
}

/// Weighted mean, SSE, and weight sum of `ys[idx]`.
pub fn mse_of(ys: &[f64], idx: &[usize], weights: &[f64]) -> (f64, f64, f64) {
    let mut wsum = 0.0;
    let mut s = 0.0;
    for &i in idx {
        wsum += weights[i];
        s += weights[i] * ys[i];
    }
    if wsum <= 0.0 {
        return (0.0, 0.0, 0.0);
    }
    let mean = s / wsum;
    let mut sse = 0.0;
    for &i in idx {
        let d = ys[i] - mean;
        sse += weights[i] * d * d;
    }
    (mean, sse, wsum)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gini_pure_and_balanced() {
        assert_eq!(gini(&[3.0, 0.0]), 0.0);
        assert!((gini(&[1.0, 1.0]) - 0.5).abs() < 1e-15);
        assert_eq!(gini(&[0.0, 0.0]), 0.0);
    }

    #[test]
    fn majority_breaks_ties_toward_smaller_id() {
        assert_eq!(majority(&[0, 1], &[2.0, 2.0]), 0);
        assert_eq!(majority(&[3, 7], &[1.0, 4.0]), 7);
    }
}
