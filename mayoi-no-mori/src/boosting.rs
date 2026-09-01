//! Gradient boosting and AdaBoost on [`oldwood`] trees.

use faer::Mat;
use oldwood::{
    class_index, grow_class, grow_reg, majority, predict_class_one, predict_class_proba,
    predict_reg, predict_reg_one, rewrite_logistic_leaves, ClassNode, GrowSpec, RegNode, Rng,
};

/// Floor used when taking `ln` of a probability in ensemble scores.
const PROB_LN_FLOOR: f64 = 1e-15;

fn sigmoid(z: f64) -> f64 {
    if z >= 0.0 {
        let e = (-z).exp();
        1.0 / (1.0 + e)
    } else {
        let e = z.exp();
        e / (1.0 + e)
    }
}

fn softmax_row(scores: &[f64]) -> Vec<f64> {
    let m = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut e: Vec<f64> = scores.iter().map(|s| (s - m).exp()).collect();
    let z: f64 = e.iter().sum::<f64>().max(PROB_LN_FLOOR);
    for v in &mut e {
        *v /= z;
    }
    e
}

/// Squared-error gradient booster hyperparameters.
#[derive(Clone, Copy, Debug)]
pub struct GbrSpec {
    /// Number of sequential trees.
    pub n_estimators: usize,
    /// Shrinkage \(\nu\).
    pub learning_rate: f64,
    /// Weak-learner grow options.
    pub grow: GrowSpec,
    /// PRNG seed.
    pub seed: u64,
}

/// Why a booster stopped.
#[derive(Clone, Debug, PartialEq)]
pub enum BoostStop {
    /// Finished all stages (or residuals vanished).
    Finished {
        /// Last stage index (0-based), if any tree was grown.
        stages: usize,
        /// Residuals fell under `eps`.
        residuals_vanished: bool,
    },
    /// `learning_rate` is not a positive finite number.
    InvalidLearningRate,
}

/// Fitted squared-error gradient booster.
#[derive(Clone, Debug)]
pub struct FittedGbr {
    /// Initial constant (training mean).
    pub intercept: f64,
    /// Sequential trees.
    pub trees: Vec<RegNode>,
    /// Shrinkage used at fit time.
    pub learning_rate: f64,
    /// Training feature count.
    pub n_features: usize,
    /// Stop reason.
    pub stop: BoostStop,
}

impl FittedGbr {
    /// Additive prediction.
    pub fn predict(&self, x: &Mat<f64>) -> Vec<f64> {
        let mut out = vec![self.intercept; x.nrows()];
        for t in &self.trees {
            for i in 0..x.nrows() {
                out[i] += self.learning_rate * predict_reg_one(t, x, i);
            }
        }
        out
    }
}

/// Fit Friedman GBR (squared error).
pub fn fit_gbr(x: &Mat<f64>, y: &[f64], spec: &GbrSpec) -> FittedGbr {
    let n = x.nrows();
    let intercept = if n == 0 {
        0.0
    } else {
        y.iter().sum::<f64>() / n as f64
    };
    let mut residual: Vec<f64> = y.iter().map(|v| v - intercept).collect();
    let w = vec![1.0; n];
    let idx: Vec<usize> = (0..n).collect();
    let mut rng = Rng::new(spec.seed);
    let mut trees = Vec::new();
    let nu = spec.learning_rate;
    if nu <= 0.0 || !nu.is_finite() {
        return FittedGbr {
            intercept,
            trees,
            learning_rate: nu,
            n_features: x.ncols(),
            stop: BoostStop::InvalidLearningRate,
        };
    }
    let mut vanished = false;
    let n_est = spec.n_estimators.max(1);
    for _m in 0..n_est {
        let mut trng = Rng::new(rng.next_u64());
        let tree = grow_reg(x, &residual, &idx, &w, &spec.grow, &mut trng);
        let mut sse = 0.0;
        for i in 0..n {
            let step = nu * predict_reg_one(&tree, x, i);
            residual[i] -= step;
            sse += residual[i] * residual[i];
        }
        trees.push(tree);
        if sse <= spec.grow.eps {
            vanished = true;
            break;
        }
    }
    FittedGbr {
        intercept,
        trees,
        learning_rate: nu,
        n_features: x.ncols(),
        stop: BoostStop::Finished {
            stages: n_est,
            residuals_vanished: vanished,
        },
    }
}

/// Log-loss gradient booster hyperparameters.
#[derive(Clone, Copy, Debug)]
pub struct GbcSpec {
    /// Number of sequential stages.
    pub n_estimators: usize,
    /// Shrinkage \(\nu\).
    pub learning_rate: f64,
    /// Weak-learner grow options.
    pub grow: GrowSpec,
    /// PRNG seed.
    pub seed: u64,
}

/// Fitted log-loss gradient booster.
#[derive(Clone, Debug)]
pub struct FittedGbc {
    /// Sorted unique training labels.
    pub classes: Vec<i64>,
    /// Per-class initial scores.
    pub intercept: Vec<f64>,
    /// Stages; for binary, each stage has one tree (positive class).
    pub trees: Vec<Vec<RegNode>>,
    /// Shrinkage used at fit time.
    pub learning_rate: f64,
    /// Training feature count.
    pub n_features: usize,
}

impl FittedGbc {
    fn scores_row(&self, x: &Mat<f64>, i: usize) -> Vec<f64> {
        let k = self.classes.len();
        let mut f = self.intercept.clone();
        if f.len() != k {
            f.resize(k, 0.0);
        }
        let binary = k <= 2;
        for stage in &self.trees {
            if binary {
                if let Some(t) = stage.first() {
                    let step = self.learning_rate * predict_reg_one(t, x, i);
                    if k == 2 {
                        f[1] += step;
                    } else if k == 1 {
                        f[0] += step;
                    }
                }
            } else {
                for (c, t) in stage.iter().enumerate().take(k) {
                    f[c] += self.learning_rate * predict_reg_one(t, x, i);
                }
            }
        }
        f
    }

    /// Predicted labels.
    pub fn predict_labels(&self, x: &Mat<f64>) -> Vec<i64> {
        let k = self.classes.len();
        (0..x.nrows())
            .map(|i| {
                if k == 0 {
                    return 0;
                }
                if k == 2 {
                    let f = self.scores_row(x, i);
                    let p = sigmoid(
                        f.get(1).copied().unwrap_or(0.0) - f.first().copied().unwrap_or(0.0),
                    );
                    if p >= 0.5 {
                        self.classes[1]
                    } else {
                        self.classes[0]
                    }
                } else {
                    let f = self.scores_row(x, i);
                    let mut best = 0usize;
                    for c in 1..f.len() {
                        if f[c] > f[best] {
                            best = c;
                        }
                    }
                    self.classes[best]
                }
            })
            .collect()
    }
}

/// Fit Friedman GBC (binomial / multinomial log-loss).
pub fn fit_gbc(x: &Mat<f64>, y: &[i64], classes: &[i64], spec: &GbcSpec) -> FittedGbc {
    let n = x.nrows();
    let k = classes.len();
    let w = vec![1.0; n];
    let idx: Vec<usize> = (0..n).collect();
    let mut rng = Rng::new(spec.seed);
    if k < 2 {
        return FittedGbc {
            classes: classes.to_vec(),
            intercept: vec![0.0],
            trees: Vec::new(),
            learning_rate: spec.learning_rate,
            n_features: x.ncols(),
        };
    }
    let mut counts = vec![0.0; k];
    for &lab in y {
        if let Some(c) = class_index(lab, classes) {
            counts[c] += 1.0;
        }
    }
    let ntot = counts.iter().sum::<f64>().max(1.0);
    let mut intercept = vec![0.0; k];
    if k == 2 {
        let p1 = counts[1] / ntot;
        let p1 = p1.clamp(PROB_LN_FLOOR, 1.0 - PROB_LN_FLOOR);
        intercept[1] = (p1 / (1.0 - p1)).ln();
    }
    let mut scores = vec![vec![0.0; k]; n];
    for i in 0..n {
        scores[i].clone_from(&intercept);
    }
    let mut trees: Vec<Vec<RegNode>> = Vec::new();
    let nu = spec.learning_rate;
    for _m in 0..spec.n_estimators.max(1) {
        let mut stage = Vec::new();
        if k == 2 {
            let mut r = vec![0.0; n];
            let mut p = vec![0.0; n];
            for i in 0..n {
                let logit = scores[i][1] - scores[i][0];
                p[i] = sigmoid(logit);
                let yi = if y[i] == classes[1] { 1.0 } else { 0.0 };
                r[i] = yi - p[i];
            }
            let mut trng = Rng::new(rng.next_u64());
            let mut tree = grow_reg(x, &r, &idx, &w, &spec.grow, &mut trng);
            rewrite_logistic_leaves(&mut tree, x, &r, &p, &idx);
            for i in 0..n {
                scores[i][1] += nu * predict_reg_one(&tree, x, i);
            }
            stage.push(tree);
        } else {
            let mut probs = vec![vec![0.0; k]; n];
            for i in 0..n {
                probs[i] = softmax_row(&scores[i]);
            }
            for c in 0..k {
                let mut r = vec![0.0; n];
                for i in 0..n {
                    let yi = if y[i] == classes[c] { 1.0 } else { 0.0 };
                    r[i] = yi - probs[i][c];
                }
                let mut trng = Rng::new(rng.next_u64());
                let tree = grow_reg(x, &r, &idx, &w, &spec.grow, &mut trng);
                for i in 0..n {
                    scores[i][c] += nu * predict_reg_one(&tree, x, i);
                }
                stage.push(tree);
            }
        }
        trees.push(stage);
    }
    FittedGbc {
        classes: classes.to_vec(),
        intercept,
        trees,
        learning_rate: nu,
        n_features: x.ncols(),
    }
}

/// AdaBoost label-update scheme.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdaBoostAlgorithm {
    /// Discrete SAMME (Zhu et al.).
    Samme,
    /// Real SAMME.R using class probabilities.
    SammeR,
}

/// AdaBoost classifier hyperparameters.
#[derive(Clone, Copy, Debug)]
pub struct AdaBoostSpec {
    /// Number of weak learners.
    pub n_estimators: usize,
    /// Shrinkage on the additive model.
    pub learning_rate: f64,
    /// Weak-learner grow options.
    pub grow: GrowSpec,
    /// `SAMME` or `SAMME.R`.
    pub algorithm: AdaBoostAlgorithm,
    /// PRNG seed.
    pub seed: u64,
}

/// Why AdaBoost stopped.
#[derive(Clone, Debug, PartialEq)]
pub enum AdaBoostStop {
    /// Completed usable stages.
    Finished {
        /// Number of weak learners kept.
        stages: usize,
    },
    /// SAMME weighted error at or worse than chance.
    WeakNotBetterThanChance {
        /// Stage index.
        stage: usize,
        /// Weighted error.
        err: f64,
    },
    /// No usable weak learner.
    Empty,
}

/// Fitted AdaBoost classifier.
#[derive(Clone, Debug)]
pub struct FittedAdaBoost {
    /// Weak learners.
    pub trees: Vec<ClassNode>,
    /// SAMME weights (`α_m`); empty when using SAMME.R.
    pub alphas: Vec<f64>,
    /// Algorithm used at fit time.
    pub algorithm: AdaBoostAlgorithm,
    /// Shrinkage.
    pub learning_rate: f64,
    /// Sorted unique training labels.
    pub classes: Vec<i64>,
    /// Training feature count.
    pub n_features: usize,
    /// Stop reason.
    pub stop: AdaBoostStop,
}

impl FittedAdaBoost {
    /// Predicted labels.
    pub fn predict_labels(&self, x: &Mat<f64>) -> Vec<i64> {
        let k = self.classes.len();
        (0..x.nrows())
            .map(|i| {
                if k == 0 {
                    return 0;
                }
                let mut scores = vec![0.0; k];
                match self.algorithm {
                    AdaBoostAlgorithm::Samme => {
                        for (t, &alpha) in self.trees.iter().zip(&self.alphas) {
                            let lab = predict_class_one(t, x, i);
                            if let Some(j) = class_index(lab, &self.classes) {
                                scores[j] += alpha;
                            }
                        }
                    }
                    AdaBoostAlgorithm::SammeR => {
                        let km1 = (k as f64 - 1.0).max(1.0);
                        for t in &self.trees {
                            let p = predict_class_proba(t, x, i, k);
                            let mut lp: Vec<f64> =
                                p.iter().map(|v| v.max(PROB_LN_FLOOR).ln()).collect();
                            let mean = lp.iter().sum::<f64>() / k as f64;
                            for c in 0..k {
                                lp[c] -= mean;
                                scores[c] += self.learning_rate * km1 * lp[c];
                            }
                        }
                    }
                }
                majority(&self.classes, &scores)
            })
            .collect()
    }
}

/// Fit SAMME / SAMME.R.
pub fn fit_adaboost(
    x: &Mat<f64>,
    y: &[i64],
    classes: &[i64],
    spec: &AdaBoostSpec,
) -> FittedAdaBoost {
    let n = x.nrows();
    let k = classes.len();
    if k < 2 || n == 0 {
        return FittedAdaBoost {
            trees: Vec::new(),
            alphas: Vec::new(),
            algorithm: spec.algorithm,
            learning_rate: spec.learning_rate,
            classes: classes.to_vec(),
            n_features: x.ncols(),
            stop: AdaBoostStop::Empty,
        };
    }
    let mut w = vec![1.0 / n as f64; n];
    let idx: Vec<usize> = (0..n).collect();
    let mut rng = Rng::new(spec.seed);
    let mut trees = Vec::new();
    let mut alphas = Vec::new();
    let k_f = k as f64;
    let mut stop = AdaBoostStop::Finished { stages: 0 };
    for m in 0..spec.n_estimators.max(1) {
        let mut trng = Rng::new(rng.next_u64());
        let tree = grow_class(x, y, classes, &idx, &w, &spec.grow, &mut trng);
        match spec.algorithm {
            AdaBoostAlgorithm::Samme => {
                let mut err = 0.0;
                let mut wsum = 0.0;
                for i in 0..n {
                    wsum += w[i];
                    if predict_class_one(&tree, x, i) != y[i] {
                        err += w[i];
                    }
                }
                err = if wsum > 0.0 { err / wsum } else { 1.0 };
                if err >= 1.0 - 1.0 / k_f {
                    stop = AdaBoostStop::WeakNotBetterThanChance { stage: m, err };
                    break;
                }
                let alpha = spec.learning_rate
                    * (((1.0 - err) / err.max(PROB_LN_FLOOR)).ln() + (k_f - 1.0).ln());
                for i in 0..n {
                    if predict_class_one(&tree, x, i) != y[i] {
                        w[i] *= alpha.exp();
                    }
                }
                let z: f64 = w.iter().sum::<f64>().max(PROB_LN_FLOOR);
                for wi in &mut w {
                    *wi /= z;
                }
                alphas.push(alpha);
                trees.push(tree);
            }
            AdaBoostAlgorithm::SammeR => {
                let factor = (k_f - 1.0) / k_f;
                for i in 0..n {
                    let p = predict_class_proba(&tree, x, i, k);
                    if let Some(c) = class_index(y[i], classes) {
                        let lp = p[c].max(PROB_LN_FLOOR).ln();
                        w[i] *= (-spec.learning_rate * factor * lp).exp();
                    }
                }
                let z: f64 = w.iter().sum::<f64>().max(PROB_LN_FLOOR);
                for wi in &mut w {
                    *wi /= z;
                }
                trees.push(tree);
            }
        }
    }
    if trees.is_empty() {
        stop = AdaBoostStop::Empty;
    } else if matches!(stop, AdaBoostStop::Finished { .. }) {
        stop = AdaBoostStop::Finished {
            stages: trees.len(),
        };
    }
    FittedAdaBoost {
        trees,
        alphas,
        algorithm: spec.algorithm,
        learning_rate: spec.learning_rate,
        classes: classes.to_vec(),
        n_features: x.ncols(),
        stop,
    }
}

/// AdaBoost.R2 hyperparameters.
#[derive(Clone, Copy, Debug)]
pub struct AdaBoostR2Spec {
    /// Number of weak learners.
    pub n_estimators: usize,
    /// Shrinkage on \(\ln(1/\beta)\).
    pub learning_rate: f64,
    /// Weak-learner grow options.
    pub grow: GrowSpec,
    /// PRNG seed.
    pub seed: u64,
}

/// Why AdaBoost.R2 stopped.
#[derive(Clone, Debug, PartialEq)]
pub enum AdaBoostR2Stop {
    /// Completed usable stages.
    Finished {
        /// Number of weak learners kept.
        stages: usize,
    },
    /// Weighted loss ≥ 1/2.
    WeightedLossGeHalf {
        /// Stage index.
        stage: usize,
        /// Weighted loss.
        loss: f64,
        /// At least one prior tree was kept.
        had_prior: bool,
    },
    /// No usable weak learner.
    Empty,
}

/// Fitted AdaBoost.R2 model.
#[derive(Clone, Debug)]
pub struct FittedAdaBoostR2 {
    /// Weak learners.
    pub trees: Vec<RegNode>,
    /// Stage weights \(\ln(1/\beta_m)\).
    pub alphas: Vec<f64>,
    /// Training feature count.
    pub n_features: usize,
    /// Stop reason.
    pub stop: AdaBoostR2Stop,
}

impl FittedAdaBoostR2 {
    /// Weighted-median prediction.
    pub fn predict(&self, x: &Mat<f64>) -> Vec<f64> {
        (0..x.nrows())
            .map(|i| {
                let mut pairs: Vec<(f64, f64)> = self
                    .trees
                    .iter()
                    .zip(&self.alphas)
                    .map(|(t, a)| (predict_reg_one(t, x, i), *a))
                    .collect();
                if pairs.is_empty() {
                    return 0.0;
                }
                pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
                let tot: f64 = pairs.iter().map(|(_, a)| *a).sum();
                let mut acc = 0.0;
                for (v, a) in pairs {
                    acc += a;
                    if acc >= 0.5 * tot {
                        return v;
                    }
                }
                0.0
            })
            .collect()
    }
}

/// Fit AdaBoost.R2 (Drucker 1997).
pub fn fit_adaboost_r2(
    x: &Mat<f64>,
    ys: &[f64],
    spec: &AdaBoostR2Spec,
    eps: f64,
) -> FittedAdaBoostR2 {
    let n = x.nrows();
    if n == 0 {
        return FittedAdaBoostR2 {
            trees: Vec::new(),
            alphas: Vec::new(),
            n_features: x.ncols(),
            stop: AdaBoostR2Stop::Empty,
        };
    }
    let mut w = vec![1.0 / n as f64; n];
    let mut rng = Rng::new(spec.seed);
    let mut trees = Vec::new();
    let mut alphas = Vec::new();
    let mut stop = AdaBoostR2Stop::Finished { stages: 0 };
    for m in 0..spec.n_estimators.max(1) {
        let mut trng = Rng::new(rng.next_u64());
        let sample = trng.weighted_bootstrap(&w);
        let unit = vec![1.0; n];
        let tree = grow_reg(x, ys, &sample, &unit, &spec.grow, &mut trng);
        let pred = predict_reg(&tree, x);
        let mut max_e = 0.0;
        let mut err = vec![0.0; n];
        for i in 0..n {
            err[i] = (ys[i] - pred[i]).abs();
            if err[i] > max_e {
                max_e = err[i];
            }
        }
        if max_e <= eps {
            trees.push(tree);
            alphas.push(1.0);
            break;
        }
        let mut lbar = 0.0;
        for i in 0..n {
            lbar += w[i] * (err[i] / max_e);
        }
        if lbar >= 0.5 {
            stop = AdaBoostR2Stop::WeightedLossGeHalf {
                stage: m,
                loss: lbar,
                had_prior: !trees.is_empty(),
            };
            break;
        }
        let beta = (lbar / (1.0 - lbar).max(PROB_LN_FLOOR)).max(1e-12);
        let alpha = spec.learning_rate * (1.0 / beta).ln();
        for i in 0..n {
            w[i] *= beta.powf(1.0 - err[i] / max_e);
        }
        let z: f64 = w.iter().sum::<f64>().max(PROB_LN_FLOOR);
        for wi in &mut w {
            *wi /= z;
        }
        trees.push(tree);
        alphas.push(alpha);
    }
    if trees.is_empty() {
        stop = AdaBoostR2Stop::Empty;
    } else if matches!(stop, AdaBoostR2Stop::Finished { .. }) {
        stop = AdaBoostR2Stop::Finished {
            stages: trees.len(),
        };
    }
    FittedAdaBoostR2 {
        trees,
        alphas,
        n_features: x.ncols(),
        stop,
    }
}
