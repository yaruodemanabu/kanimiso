//! Bootstrap / extra-trees forests on [`oldwood`] trees.

use faer::Mat;
use oldwood::{
    class_index, grow_class, grow_reg, majority, predict_class_one, predict_reg_one, ClassNode,
    GrowSpec, RegNode, Rng,
};

/// Hyperparameters shared by random forests and extra-trees.
#[derive(Clone, Copy, Debug)]
pub struct ForestSpec {
    /// Number of trees.
    pub n_estimators: usize,
    /// CART grow options (set `extra` / `sqrt_features` here).
    pub grow: GrowSpec,
    /// Draw a bootstrap sample per tree.
    pub bootstrap: bool,
    /// PRNG seed.
    pub seed: u64,
}

impl Default for ForestSpec {
    fn default() -> Self {
        Self {
            n_estimators: 20,
            grow: GrowSpec {
                sqrt_features: true,
                ..GrowSpec::default()
            },
            bootstrap: true,
            seed: 0,
        }
    }
}

/// Fitted classification forest.
#[derive(Clone, Debug)]
pub struct ForestClassifier {
    /// Grown trees.
    pub trees: Vec<ClassNode>,
    /// Sorted unique training labels.
    pub classes: Vec<i64>,
    /// Training feature count.
    pub n_features: usize,
}

impl ForestClassifier {
    /// Majority vote for row `i`.
    pub fn vote_row(&self, x: &Mat<f64>, i: usize) -> i64 {
        let k = self.classes.len();
        let mut votes = vec![0.0; k];
        for t in &self.trees {
            let lab = predict_class_one(t, x, i);
            if let Some(j) = class_index(lab, &self.classes) {
                votes[j] += 1.0;
            }
        }
        if k == 0 {
            0
        } else {
            majority(&self.classes, &votes)
        }
    }

    /// Predicted labels for every row.
    pub fn predict_labels(&self, x: &Mat<f64>) -> Vec<i64> {
        (0..x.nrows()).map(|i| self.vote_row(x, i)).collect()
    }
}

/// Grow a classification forest (random forest or extra-trees).
pub fn grow_forest_class(
    x: &Mat<f64>,
    y: &[i64],
    classes: &[i64],
    spec: &ForestSpec,
) -> ForestClassifier {
    let w = vec![1.0; x.nrows()];
    let mut rng = Rng::new(spec.seed);
    let mut trees = Vec::with_capacity(spec.n_estimators.max(1));
    if classes.is_empty() {
        return ForestClassifier {
            trees,
            classes: classes.to_vec(),
            n_features: x.ncols(),
        };
    }
    for _ in 0..spec.n_estimators.max(1) {
        let mut trng = Rng::new(rng.next_u64());
        let idx = if spec.bootstrap && x.nrows() > 0 {
            trng.bootstrap_idx(x.nrows())
        } else {
            (0..x.nrows()).collect()
        };
        trees.push(grow_class(x, y, classes, &idx, &w, &spec.grow, &mut trng));
    }
    ForestClassifier {
        trees,
        classes: classes.to_vec(),
        n_features: x.ncols(),
    }
}

/// Fitted regression forest.
#[derive(Clone, Debug)]
pub struct ForestRegressor {
    /// Grown trees.
    pub trees: Vec<RegNode>,
    /// Training feature count.
    pub n_features: usize,
}

impl ForestRegressor {
    /// Mean of tree predictions for every row.
    pub fn predict(&self, x: &Mat<f64>) -> Vec<f64> {
        let mut out = vec![0.0; x.nrows()];
        if self.trees.is_empty() {
            return out;
        }
        let inv = 1.0 / self.trees.len() as f64;
        for i in 0..x.nrows() {
            let mut s = 0.0;
            for t in &self.trees {
                s += predict_reg_one(t, x, i);
            }
            out[i] = s * inv;
        }
        out
    }
}

/// Grow a regression forest.
pub fn grow_forest_reg(x: &Mat<f64>, ys: &[f64], spec: &ForestSpec) -> ForestRegressor {
    let w = vec![1.0; x.nrows()];
    let mut rng = Rng::new(spec.seed);
    let mut trees = Vec::new();
    let n_est = spec.n_estimators.max(1);
    let full: Vec<usize> = (0..x.nrows()).collect();
    for _ in 0..n_est {
        let mut trng = Rng::new(rng.next_u64());
        let idx = if spec.bootstrap && x.nrows() > 0 {
            trng.bootstrap_idx(x.nrows())
        } else {
            full.clone()
        };
        trees.push(grow_reg(x, ys, &idx, &w, &spec.grow, &mut trng));
    }
    ForestRegressor {
        trees,
        n_features: x.ncols(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_blob_forest_votes() {
        let x = Mat::<f64>::from_fn(8, 2, |i, j| {
            let c = if i < 4 { 0.0 } else { 4.0 };
            c + 0.01 * (j as f64)
        });
        let y = [0_i64, 0, 0, 0, 1, 1, 1, 1];
        let classes = [0_i64, 1];
        let spec = ForestSpec {
            n_estimators: 8,
            seed: 2,
            grow: GrowSpec {
                max_depth: 4,
                sqrt_features: true,
                ..GrowSpec::default()
            },
            bootstrap: true,
        };
        let forest = grow_forest_class(&x, &y, &classes, &spec);
        let pred = forest.predict_labels(&x);
        let ok = pred.iter().zip(y.iter()).filter(|(a, b)| *a == *b).count();
        assert!(ok >= 6, "ok={ok} pred={pred:?}");
    }
}
