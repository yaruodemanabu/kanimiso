//! Isolation Forest and random-trees embedding on [`oldwood`] isolation trees.

use faer::Mat;
use oldwood::{grow_iso, iso_leaf_code, iso_path, isolation_c_factor, IsoNode, Rng};

/// Isolation Forest hyperparameters.
#[derive(Clone, Copy, Debug)]
pub struct IsolationSpec {
    /// Number of isolation trees.
    pub n_trees: usize,
    /// PRNG seed.
    pub seed: u64,
}

impl Default for IsolationSpec {
    fn default() -> Self {
        Self {
            n_trees: 50,
            seed: 0,
        }
    }
}

/// Fitted isolation forest.
#[derive(Clone, Debug)]
pub struct FittedIsolation {
    /// Isolation trees.
    pub trees: Vec<IsoNode>,
    /// Subsample size used to grow each tree (and in \(c(n)\)).
    pub max_samples: usize,
    /// Training feature count.
    pub n_features: usize,
    /// \(c(\texttt{max_samples})\).
    pub c_norm: f64,
}

impl FittedIsolation {
    /// Mean path length of row `i`.
    pub fn average_path_length(&self, x: &Mat<f64>, i: usize) -> f64 {
        if self.trees.is_empty() {
            return 0.0;
        }
        let mut s = 0.0;
        for t in &self.trees {
            s += iso_path(t, x, i);
        }
        s / self.trees.len() as f64
    }

    /// Liu et al. anomaly score \(s(x,n)=2^{-E(h)/c(n)}\) (higher = more anomalous).
    pub fn score_samples(&self, x: &Mat<f64>) -> Vec<f64> {
        let c = if self.c_norm > 0.0 { self.c_norm } else { 1.0 };
        (0..x.nrows())
            .map(|i| {
                let eh = self.average_path_length(x, i);
                2.0_f64.powf(-eh / c)
            })
            .collect()
    }
}

/// Grow an isolation forest (Liu, Ting, Zhou).
pub fn fit_isolation(x: &Mat<f64>, spec: &IsolationSpec) -> FittedIsolation {
    let n = x.nrows();
    if n == 0 {
        return FittedIsolation {
            trees: Vec::new(),
            max_samples: 0,
            n_features: x.ncols(),
            c_norm: 0.0,
        };
    }
    let max_samples = n.min(256);
    let max_depth = (max_samples as f64).log2().ceil().max(1.0) as usize;
    let mut rng = Rng::new(spec.seed);
    let mut trees = Vec::new();
    let n_trees = spec.n_trees.max(1);
    for _ in 0..n_trees {
        let mut trng = Rng::new(rng.next_u64());
        let idx = if n > max_samples {
            trng.sample_indices(n, max_samples)
        } else {
            (0..n).collect()
        };
        trees.push(grow_iso(x, &idx, 0, max_depth, &mut trng));
    }
    FittedIsolation {
        trees,
        max_samples,
        n_features: x.ncols(),
        c_norm: isolation_c_factor(max_samples as f64),
    }
}

/// Completely-random tree leaf embedding hyperparameters.
#[derive(Clone, Copy, Debug)]
pub struct EmbeddingSpec {
    /// Number of random trees.
    pub n_estimators: usize,
    /// Hashed leaf-code width.
    pub n_components: usize,
    /// PRNG seed.
    pub seed: u64,
}

/// Fitted random-tree leaf embedding.
#[derive(Clone, Debug)]
pub struct FittedEmbedding {
    /// Isolation-style random trees.
    pub trees: Vec<IsoNode>,
    /// Hash width.
    pub n_components: usize,
    /// Training feature count.
    pub n_features: usize,
}

impl FittedEmbedding {
    /// Hashed leaf-count embedding (`n × n_components`).
    pub fn transform(&self, x: &Mat<f64>) -> Mat<f64> {
        let m = self.n_components.max(1);
        let mut out = Mat::<f64>::zeros(x.nrows(), m);
        for i in 0..x.nrows() {
            for (t, tree) in self.trees.iter().enumerate() {
                let code = iso_leaf_code(tree, x, i).wrapping_add(t as u64);
                let bin = (code as usize) % m;
                out[(i, bin)] += 1.0;
            }
        }
        out
    }
}

/// Fit a random-trees embedding (sklearn `RandomTreesEmbedding`).
pub fn fit_embedding(x: &Mat<f64>, spec: &EmbeddingSpec) -> FittedEmbedding {
    let n = x.nrows();
    let n_est = spec.n_estimators.max(1);
    let n_comp = spec.n_components.max(1);
    if n == 0 || x.ncols() == 0 {
        return FittedEmbedding {
            trees: Vec::new(),
            n_components: n_comp,
            n_features: x.ncols(),
        };
    }
    let max_depth = (n as f64).log2().ceil().max(1.0) as usize;
    let mut rng = Rng::new(spec.seed ^ 0xA11CE);
    let mut trees = Vec::with_capacity(n_est);
    for _ in 0..n_est {
        let mut trng = Rng::new(rng.next_u64());
        let idx: Vec<usize> = (0..n).collect();
        trees.push(grow_iso(x, &idx, 0, max_depth.max(2), &mut trng));
    }
    FittedEmbedding {
        trees,
        n_components: n_comp,
        n_features: x.ncols(),
    }
}
