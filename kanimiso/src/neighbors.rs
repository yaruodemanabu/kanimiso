//! Neighbors methods: k-NN, radius neighbors, LOF, Gaussian KDE, nearest centroid.
//!
//! Every `fit` / `predict` talks to [`crate::context::FitCtx`] so `signlred`
//! can abort on `SingleClass` / `EmptyClass` / `ConstantTarget` and
//! `ojizou-san` records the session.

use crate::context::FitCtx;
use crate::data::{Matrix, Vector};
use crate::traits::{Fit, FitUnsupervised, Predict};
use crate::validate::{inspect_classes, inspect_identification, inspect_xy};
use ojizou_san::Session;
use signlred::{Issue, IssueCode, Meaninglessness, Qualified, Result};

fn labels_of(y: &Vector) -> Vec<i64> {
    y.as_slice()
        .iter()
        .map(|&v| if v.is_finite() { v.round() as i64 } else { 0 })
        .collect()
}

fn class_index(lab: i64, classes: &[i64]) -> Option<usize> {
    classes.iter().position(|&c| c == lab)
}

fn majority(classes: &[i64], counts: &[f64]) -> i64 {
    let mut best_i = 0usize;
    let mut best = f64::NEG_INFINITY;
    for (i, &c) in counts.iter().enumerate() {
        if c > best + 1e-15 || ((c - best).abs() <= 1e-15 && classes[i] < classes[best_i]) {
            best = c;
            best_i = i;
        }
    }
    classes.get(best_i).copied().unwrap_or(0)
}

fn sq_dist_row(a: &Matrix, i: usize, b: &Matrix, t: usize) -> f64 {
    let p = a.ncols().min(b.ncols());
    let mut s = 0.0;
    for j in 0..p {
        let d = a.get(i, j) - b.get(t, j);
        s += d * d;
    }
    s
}

fn sq_dist_vec(a: &Matrix, i: usize, z: &Vector) -> f64 {
    let p = a.ncols().min(z.len());
    let mut s = 0.0;
    for j in 0..p {
        let d = a.get(i, j) - z[j];
        s += d * d;
    }
    s
}

fn knn_order(train: &Matrix, query: &Matrix, i: usize) -> Vec<(f64, usize)> {
    let mut d: Vec<(f64, usize)> = (0..train.nrows())
        .map(|t| (sq_dist_row(train, t, query, i), t))
        .collect();
    d.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    d
}

fn logsumexp(xs: &[f64]) -> f64 {
    let m = xs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !m.is_finite() {
        return m;
    }
    let mut s = 0.0;
    for &v in xs {
        s += (v - m).exp();
    }
    m + s.ln()
}

fn diagnose_constant_predictions(ctx: &mut FitCtx, pred: &Vector, y: &Vector) {
    let pst = signlred::slice_stats(pred.as_slice());
    let yst = signlred::slice_stats(y.as_slice());
    if pst.is_constant(ctx.policy.near_zero_variance) && !yst.is_constant(ctx.policy.near_zero_variance)
    {
        ctx.push(
            Issue::builder(IssueCode::PredictionsAreConstant)
                .message("neighbor predictor collapsed to a constant")
                .meaninglessness(Meaninglessness::vacuous(
                    "nearest-neighbor predictor",
                    "every in-sample prediction is the same while y varies",
                    "increase k diversity or inspect feature scaling",
                ))
                .build(),
        );
    }
}

fn predict_shape_guard(ctx: &mut FitCtx, x: &Matrix, n_features: usize) {
    inspect_xy(&mut ctx.report, x, None, &ctx.policy);
    if x.ncols() != n_features {
        ctx.push(
            Issue::builder(IssueCode::DimensionMismatch)
                .message(format!(
                    "predict X is n×{} but the model was fit on {} features",
                    x.ncols(),
                    n_features
                ))
                .build(),
        );
    }
}

/// k-nearest-neighbor classifier (majority vote, Euclidean).
#[derive(Clone, Debug)]
pub struct KNeighborsClassifier {
    /// Number of neighbors.
    pub k: usize,
}

impl Default for KNeighborsClassifier {
    fn default() -> Self {
        Self { k: 5 }
    }
}

impl KNeighborsClassifier {
    /// Classifier with `k` neighbors.
    pub fn new(k: usize) -> Self {
        Self { k }
    }
}

/// Fitted k-NN classifier (stores the training set).
#[derive(Clone, Debug)]
pub struct FittedKnnClassifier {
    /// Training features.
    pub x_train: Matrix,
    /// Training labels (rounded).
    pub y_train: Vec<i64>,
    /// Sorted unique labels.
    pub classes: Vec<i64>,
    /// Neighbors used at predict time.
    pub k: usize,
}

impl Predict for FittedKnnClassifier {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        predict_shape_guard(&mut ctx, x, self.x_train.ncols());
        let k = self.k.max(1).min(self.x_train.nrows().max(1));
        if self.k == 0 {
            ctx.push(
                Issue::builder(IssueCode::InvalidWeight)
                    .message("k=0 is not a neighbor count; using 1")
                    .build(),
            );
        }
        if self.x_train.nrows() == 0 {
            ctx.push(
                Issue::builder(IssueCode::EmptyMatrix)
                    .message("k-NN has an empty training set")
                    .build(),
            );
            return ctx.finish(Vector::zeros(x.nrows()));
        }
        let mut out = Vector::zeros(x.nrows());
        for i in 0..x.nrows() {
            let order = knn_order(&self.x_train, x, i);
            let mut counts = vec![0.0; self.classes.len()];
            for &(_, t) in order.iter().take(k) {
                if let Some(c) = class_index(self.y_train[t], &self.classes) {
                    counts[c] += 1.0;
                }
            }
            out[i] = majority(&self.classes, &counts) as f64;
        }
        ctx.finish(out)
    }
}

impl Fit for KNeighborsClassifier {
    type Fitted = FittedKnnClassifier;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedKnnClassifier>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        let counts = inspect_classes(&mut ctx.report, y, &ctx.policy);
        let classes: Vec<i64> = counts.iter().map(|(k, _)| *k).collect();
        if self.k == 0 {
            ctx.push(
                Issue::builder(IssueCode::InvalidWeight)
                    .message("KNeighborsClassifier.k = 0")
                    .build(),
            );
        }
        if self.k > x.nrows() && x.nrows() > 0 {
            ctx.push(
                Issue::builder(IssueCode::InsufficientSample)
                    .message(format!("k={} > n={}; predict will clip k to n", self.k, x.nrows()))
                    .metric("k", self.k as f64)
                    .metric("n", x.nrows() as f64)
                    .build(),
            );
        }
        let fitted = FittedKnnClassifier {
            x_train: x.clone(),
            y_train: labels_of(y),
            classes,
            k: self.k.max(1),
        };
        if !y.is_empty() && x.nrows() > 0 {
            let pred = {
                let k = fitted.k.min(x.nrows());
                let mut out = Vector::zeros(x.nrows());
                for i in 0..x.nrows() {
                    let order = knn_order(&fitted.x_train, x, i);
                    let mut cts = vec![0.0; fitted.classes.len()];
                    for &(_, t) in order.iter().take(k) {
                        if let Some(c) = class_index(fitted.y_train[t], &fitted.classes) {
                            cts[c] += 1.0;
                        }
                    }
                    out[i] = majority(&fitted.classes, &cts) as f64;
                }
                out
            };
            diagnose_constant_predictions(&mut ctx, &pred, y);
        }
        ctx.finish(fitted)
    }
}

/// k-nearest-neighbor regressor (mean of neighbors).
#[derive(Clone, Debug)]
pub struct KNeighborsRegressor {
    /// Number of neighbors.
    pub k: usize,
}

impl Default for KNeighborsRegressor {
    fn default() -> Self {
        Self { k: 5 }
    }
}

impl KNeighborsRegressor {
    /// Regressor with `k` neighbors.
    pub fn new(k: usize) -> Self {
        Self { k }
    }
}

/// Fitted k-NN regressor.
#[derive(Clone, Debug)]
pub struct FittedKnnRegressor {
    /// Training features.
    pub x_train: Matrix,
    /// Training response.
    pub y_train: Vector,
    /// Neighbors used at predict time.
    pub k: usize,
}

impl Predict for FittedKnnRegressor {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        predict_shape_guard(&mut ctx, x, self.x_train.ncols());
        if self.x_train.nrows() == 0 {
            ctx.push(Issue::builder(IssueCode::EmptyMatrix).message("empty k-NN training set").build());
            return ctx.finish(Vector::zeros(x.nrows()));
        }
        let k = self.k.max(1).min(self.x_train.nrows());
        let mut out = Vector::zeros(x.nrows());
        for i in 0..x.nrows() {
            let order = knn_order(&self.x_train, x, i);
            let mut s = 0.0;
            for &(_, t) in order.iter().take(k) {
                s += self.y_train[t];
            }
            out[i] = s / k as f64;
        }
        ctx.finish(out)
    }
}

impl Fit for KNeighborsRegressor {
    type Fitted = FittedKnnRegressor;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedKnnRegressor>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        if self.k == 0 {
            ctx.push(
                Issue::builder(IssueCode::InvalidWeight)
                    .message("KNeighborsRegressor.k = 0")
                    .build(),
            );
        }
        ctx.finish(FittedKnnRegressor {
            x_train: x.clone(),
            y_train: y.clone(),
            k: self.k.max(1),
        })
    }
}

/// Radius-neighbors classifier (majority vote inside a Euclidean ball).
#[derive(Clone, Debug)]
pub struct RadiusNeighborsClassifier {
    /// Inclusive radius (Euclidean).
    pub radius: f64,
}

impl Default for RadiusNeighborsClassifier {
    fn default() -> Self {
        Self { radius: 1.0 }
    }
}

impl RadiusNeighborsClassifier {
    /// Classifier with the given radius.
    pub fn new(radius: f64) -> Self {
        Self { radius }
    }
}

/// Fitted radius-neighbors classifier.
#[derive(Clone, Debug)]
pub struct FittedRadiusNeighbors {
    /// Training features.
    pub x_train: Matrix,
    /// Training labels.
    pub y_train: Vec<i64>,
    /// Sorted unique labels.
    pub classes: Vec<i64>,
    /// Ball radius.
    pub radius: f64,
    /// Global majority used when a query has an empty neighborhood.
    pub fallback: i64,
}

impl Predict for FittedRadiusNeighbors {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        predict_shape_guard(&mut ctx, x, self.x_train.ncols());
        let r2 = (self.radius.max(0.0)).powi(2);
        let mut empty = 0usize;
        let mut out = Vector::zeros(x.nrows());
        for i in 0..x.nrows() {
            let mut counts = vec![0.0; self.classes.len()];
            let mut hit = 0usize;
            for t in 0..self.x_train.nrows() {
                if sq_dist_row(&self.x_train, t, x, i) <= r2 {
                    hit += 1;
                    if let Some(c) = class_index(self.y_train[t], &self.classes) {
                        counts[c] += 1.0;
                    }
                }
            }
            if hit == 0 {
                empty += 1;
                out[i] = self.fallback as f64;
            } else {
                out[i] = majority(&self.classes, &counts) as f64;
            }
        }
        if empty > 0 {
            ctx.push(
                Issue::builder(IssueCode::InsufficientSample)
                    .message(format!(
                        "{empty} / {} queries have an empty radius-{:.4} neighborhood; majority fallback used",
                        x.nrows(),
                        self.radius
                    ))
                    .metric("empty_neighborhoods", empty as f64)
                    .build(),
            );
        }
        if empty == x.nrows() && x.nrows() > 0 {
            ctx.push(
                Issue::builder(IssueCode::MeaninglessFit)
                    .message("every query had an empty neighborhood")
                    .meaninglessness(Meaninglessness::vacuous(
                        "radius-neighbors labels",
                        "the radius isolated every query from the training set",
                        "increase the radius or rescale features",
                    ))
                    .build(),
            );
        }
        ctx.finish(out)
    }
}

impl Fit for RadiusNeighborsClassifier {
    type Fitted = FittedRadiusNeighbors;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedRadiusNeighbors>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        let counts = inspect_classes(&mut ctx.report, y, &ctx.policy);
        let classes: Vec<i64> = counts.iter().map(|(k, _)| *k).collect();
        if !self.radius.is_finite() || self.radius < 0.0 {
            ctx.push(
                Issue::builder(IssueCode::InvalidWeight)
                    .message(format!("radius={} is not a finite non-negative number", self.radius))
                    .build(),
            );
        }
        let fallback = counts
            .iter()
            .max_by_key(|(_, c)| *c)
            .map(|(k, _)| *k)
            .unwrap_or(0);
        ctx.finish(FittedRadiusNeighbors {
            x_train: x.clone(),
            y_train: labels_of(y),
            classes,
            radius: self.radius,
            fallback,
        })
    }
}

/// Local outlier factor (Breunig et al.).
#[derive(Clone, Debug)]
pub struct LocalOutlierFactor {
    /// Neighborhood size.
    pub k: usize,
}

impl Default for LocalOutlierFactor {
    fn default() -> Self {
        Self { k: 5 }
    }
}

impl LocalOutlierFactor {
    /// LOF with `k` neighbors.
    pub fn new(k: usize) -> Self {
        Self { k }
    }
}

/// Fitted LOF model (novelty scores against the training cloud).
#[derive(Clone, Debug)]
pub struct FittedLof {
    /// Training features.
    pub x_train: Matrix,
    /// Neighbor count actually used (`min(k, n-1)`).
    pub k: usize,
    /// Local reachability density of each training row.
    pub lrd: Vector,
    /// k-distance of each training row.
    pub k_dist: Vector,
}

fn lof_neighbors(x: &Matrix, i: usize, k: usize) -> (Vec<usize>, f64) {
    let mut d: Vec<(f64, usize)> = (0..x.nrows())
        .filter(|&t| t != i)
        .map(|t| (sq_dist_row(x, i, x, t).sqrt(), t))
        .collect();
    d.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let take = k.min(d.len());
    if take == 0 {
        return (Vec::new(), 0.0);
    }
    let kdist = d[take - 1].0;
    let nbrs = d.into_iter().take(take).map(|(_, t)| t).collect();
    (nbrs, kdist)
}

impl FittedLof {
    /// LOF scores for the rows of `x` (higher = more outlying). Training-row
    /// queries use the stored densities; novel rows are scored against the cloud.
    pub fn score_samples(&self, x: &Matrix) -> Vector {
        let same = x.nrows() == self.x_train.nrows()
            && x.ncols() == self.x_train.ncols()
            && (0..x.nrows()).all(|i| {
                (0..x.ncols()).all(|j| (x.get(i, j) - self.x_train.get(i, j)).abs() < 1e-15)
            });
        if same {
            return self.training_scores();
        }
        Vector::from_iter((0..x.nrows()).map(|i| self.score_query(x, i)))
    }

    fn training_scores(&self) -> Vector {
        let k = self.k.max(1);
        Vector::from_iter((0..self.x_train.nrows()).map(|i| {
            let (nbrs, _) = lof_neighbors(&self.x_train, i, k);
            if nbrs.is_empty() || self.lrd[i] <= 0.0 {
                return 1.0;
            }
            let mut s = 0.0;
            for &t in &nbrs {
                s += self.lrd[t] / self.lrd[i];
            }
            s / nbrs.len() as f64
        }))
    }

    fn score_query(&self, x: &Matrix, i: usize) -> f64 {
        let k = self.k.max(1);
        let mut d: Vec<(f64, usize)> = (0..self.x_train.nrows())
            .map(|t| (sq_dist_row(&self.x_train, t, x, i).sqrt(), t))
            .collect();
        d.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let take = k.min(d.len());
        if take == 0 {
            return f64::INFINITY;
        }
        let mut reach_sum = 0.0;
        let mut lrd_nbr = 0.0;
        for &(dist, t) in d.iter().take(take) {
            reach_sum += dist.max(self.k_dist[t]);
            lrd_nbr += self.lrd[t];
        }
        let lrd_q = if reach_sum > 0.0 {
            take as f64 / reach_sum
        } else {
            f64::INFINITY
        };
        if !lrd_q.is_finite() || lrd_q <= 0.0 {
            return 1.0;
        }
        (lrd_nbr / take as f64) / lrd_q
    }
}

impl Predict for FittedLof {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        predict_shape_guard(&mut ctx, x, self.x_train.ncols());
        ctx.finish(self.score_samples(x))
    }
}

impl FitUnsupervised for LocalOutlierFactor {
    type Fitted = FittedLof;
    fn fit_unsupervised(&mut self, x: &Matrix, session: &Session) -> Result<Qualified<FittedLof>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let n = x.nrows();
        let mut k = self.k.max(1);
        if n <= 1 {
            ctx.push(
                Issue::builder(IssueCode::InsufficientSample)
                    .message("LOF needs at least two points")
                    .build(),
            );
            return ctx.finish(FittedLof {
                x_train: x.clone(),
                k: 1,
                lrd: Vector::zeros(n),
                k_dist: Vector::zeros(n),
            });
        }
        if k >= n {
            ctx.push(
                Issue::builder(IssueCode::InsufficientSample)
                    .message(format!("LOF k={k} ≥ n={n}; using n-1"))
                    .build(),
            );
            k = n - 1;
        }
        let mut k_dist = Vector::zeros(n);
        let mut nbrs: Vec<Vec<usize>> = Vec::with_capacity(n);
        for i in 0..n {
            let (nn, kd) = lof_neighbors(x, i, k);
            k_dist[i] = kd;
            nbrs.push(nn);
        }
        let mut lrd = Vector::zeros(n);
        for i in 0..n {
            let mut reach = 0.0;
            for &t in &nbrs[i] {
                let d = sq_dist_row(x, i, x, t).sqrt();
                reach += d.max(k_dist[t]);
            }
            lrd[i] = if reach > 0.0 && !nbrs[i].is_empty() {
                nbrs[i].len() as f64 / reach
            } else {
                0.0
            };
        }
        if lrd.as_slice().iter().all(|v| *v <= 0.0) {
            ctx.push(
                Issue::builder(IssueCode::DegenerateDistribution)
                    .message("all local reachability densities are zero")
                    .build(),
            );
        }
        ctx.finish(FittedLof {
            x_train: x.clone(),
            k,
            lrd,
            k_dist,
        })
    }
}

/// Gaussian kernel density estimator.
#[derive(Clone, Debug)]
pub struct KernelDensity {
    /// Bandwidth \(h > 0\).
    pub bandwidth: f64,
}

impl Default for KernelDensity {
    fn default() -> Self {
        Self { bandwidth: 1.0 }
    }
}

impl KernelDensity {
    /// Gaussian KDE with the given bandwidth.
    pub fn new(bandwidth: f64) -> Self {
        Self { bandwidth }
    }
}

/// Fitted Gaussian KDE.
#[derive(Clone, Debug)]
pub struct FittedKde {
    /// Training features.
    pub x_train: Matrix,
    /// Bandwidth.
    pub bandwidth: f64,
}

impl FittedKde {
    /// Log-density of each row of `x`.
    pub fn log_density(&self, x: &Matrix) -> Vector {
        let n = self.x_train.nrows();
        let p = self.x_train.ncols() as f64;
        let h = self.bandwidth.max(1e-15);
        let log_const = -0.5 * p * (2.0 * std::f64::consts::PI).ln() - p * h.ln();
        Vector::from_iter((0..x.nrows()).map(|i| {
            if n == 0 {
                return f64::NEG_INFINITY;
            }
            let mut terms = Vec::with_capacity(n);
            for t in 0..n {
                let d2 = sq_dist_row(&self.x_train, t, x, i);
                terms.push(log_const - d2 / (2.0 * h * h));
            }
            logsumexp(&terms) - (n as f64).ln()
        }))
    }
}

impl Predict for FittedKde {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        predict_shape_guard(&mut ctx, x, self.x_train.ncols());
        ctx.finish(self.log_density(x))
    }
}

impl FitUnsupervised for KernelDensity {
    type Fitted = FittedKde;
    fn fit_unsupervised(&mut self, x: &Matrix, session: &Session) -> Result<Qualified<FittedKde>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        if !self.bandwidth.is_finite() || self.bandwidth <= 0.0 {
            ctx.push(
                Issue::builder(IssueCode::InvalidWeight)
                    .message(format!("KDE bandwidth={} is not a positive finite number", self.bandwidth))
                    .build(),
            );
        }
        if self.bandwidth > 0.0 && self.bandwidth < 1e-8 {
            ctx.push(
                Issue::builder(IssueCode::NumericalUnderflow)
                    .message("KDE bandwidth is tiny; log-density may underflow to −∞")
                    .build(),
            );
        }
        ctx.finish(FittedKde {
            x_train: x.clone(),
            bandwidth: self.bandwidth,
        })
    }
}

/// Nearest-centroid (Rocchio) classifier.
#[derive(Clone, Debug, Default)]
pub struct NearestCentroid;

impl NearestCentroid {
    /// Default nearest-centroid classifier.
    pub fn new() -> Self {
        Self
    }
}

/// Fitted class centroids.
#[derive(Clone, Debug)]
pub struct FittedNearestCentroid {
    /// Sorted unique labels.
    pub classes: Vec<i64>,
    /// Centroid matrix (`K × p`).
    pub centroids: Matrix,
}

impl Predict for FittedNearestCentroid {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        predict_shape_guard(&mut ctx, x, self.centroids.ncols());
        if self.classes.is_empty() {
            ctx.push(
                Issue::builder(IssueCode::EmptyClass)
                    .message("nearest centroid has no classes")
                    .build(),
            );
            return ctx.finish(Vector::zeros(x.nrows()));
        }
        let mut out = Vector::zeros(x.nrows());
        for i in 0..x.nrows() {
            let mut best = 0usize;
            let mut best_d = f64::INFINITY;
            for c in 0..self.centroids.nrows() {
                let row = self.centroids.row(c);
                let d = sq_dist_vec(x, i, &row);
                if d < best_d {
                    best_d = d;
                    best = c;
                }
            }
            out[i] = self.classes[best] as f64;
        }
        ctx.finish(out)
    }
}

impl Fit for NearestCentroid {
    type Fitted = FittedNearestCentroid;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedNearestCentroid>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        let counts = inspect_classes(&mut ctx.report, y, &ctx.policy);
        inspect_identification(&mut ctx.report, x.nrows(), x.ncols(), &ctx.policy);
        let classes: Vec<i64> = counts.iter().map(|(k, _)| *k).collect();
        let ylab = labels_of(y);
        let k = classes.len();
        let p = x.ncols();
        let mut centroids = Matrix::zeros(k.max(1), p);
        for (c, &(lab, n_c)) in counts.iter().enumerate() {
            if n_c == 0 {
                ctx.push(
                    Issue::builder(IssueCode::EmptyClass)
                        .message(format!("class {lab} has zero finite rows"))
                        .build(),
                );
                continue;
            }
            for i in 0..x.nrows() {
                if ylab[i] == lab {
                    for j in 0..p {
                        centroids.set(c, j, centroids.get(c, j) + x.get(i, j));
                    }
                }
            }
            let inv = 1.0 / n_c as f64;
            for j in 0..p {
                centroids.set(c, j, centroids.get(c, j) * inv);
            }
        }
        let fitted = FittedNearestCentroid { classes, centroids };
        if x.nrows() > 0 && !fitted.classes.is_empty() {
            let mut pred = Vector::zeros(x.nrows());
            for i in 0..x.nrows() {
                let mut best = 0usize;
                let mut best_d = f64::INFINITY;
                for c in 0..fitted.centroids.nrows() {
                    let row = fitted.centroids.row(c);
                    let d = sq_dist_vec(x, i, &row);
                    if d < best_d {
                        best_d = d;
                        best = c;
                    }
                }
                pred[i] = fitted.classes.get(best).copied().unwrap_or(0) as f64;
            }
            diagnose_constant_predictions(&mut ctx, &pred, y);
        }
        ctx.finish(fitted)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;
    use ojizou_san::Session;

    fn accuracy(pred: &Vector, y: &Vector) -> f64 {
        let mut ok = 0usize;
        for i in 0..y.len() {
            if (pred[i].round() - y[i].round()).abs() < 0.5 {
                ok += 1;
            }
        }
        ok as f64 / y.len() as f64
    }

    fn two_class(n_per: usize) -> (Matrix, Vector) {
        let mut rng = Rng::new(11);
        let n = n_per * 2;
        let mut data = vec![0.0; n * 2];
        let mut y = vec![0.0; n];
        for i in 0..n {
            let c = if i < n_per { 0.0 } else { 3.0 };
            data[i * 2] = c + 0.15 * rng.standard_normal();
            data[i * 2 + 1] = c + 0.15 * rng.standard_normal();
            y[i] = if i < n_per { 0.0 } else { 1.0 };
        }
        (Matrix::from_row_major(n, 2, &data), Vector::from_slice(&y))
    }

    #[test]
    fn knn_obvious_two_class() {
        let (x, y) = two_class(16);
        let q = KNeighborsClassifier { k: 3 }
            .fit(&x, &y, &Session::new("knn", "fit"))
            .expect("knn");
        let pred = q
            .value
            .predict(&x, &Session::new("knn", "predict"))
            .expect("pred")
            .value;
        assert!(accuracy(&pred, &y) > 0.9, "acc={}", accuracy(&pred, &y));
    }

    #[test]
    fn knn_constant_y_errors() {
        let x = Matrix::from_fn(8, 2, |i, j| (i + j) as f64);
        let y = Vector::filled(8, 1.0);
        let err = KNeighborsClassifier { k: 3 }
            .fit(&x, &y, &Session::new("knn", "fit"))
            .unwrap_err();
        assert!(
            err.primary().code == IssueCode::ConstantTarget
                || err.primary().code == IssueCode::SingleClass
        );
    }

    #[test]
    fn nearest_centroid_separates_blobs() {
        let (x, y) = two_class(12);
        let q = NearestCentroid::new()
            .fit(&x, &y, &Session::new("nc", "fit"))
            .expect("nc");
        let pred = q.value.predict(&x, &Session::new("nc", "predict")).unwrap().value;
        assert!(accuracy(&pred, &y) > 0.9);
    }

    #[test]
    fn lof_flags_far_point() {
        let (x, _) = two_class(12);
        let q = LocalOutlierFactor { k: 4 }
            .fit_unsupervised(&x, &Session::new("lof", "fit"))
            .expect("lof");
        let mut far = Matrix::zeros(1, 2);
        far.set(0, 0, 30.0);
        far.set(0, 1, -30.0);
        let s = q
            .value
            .predict(&far, &Session::new("lof", "predict"))
            .unwrap()
            .value;
        assert!(s[0] > 1.2, "LOF of far point = {}", s[0]);
    }

    #[test]
    fn kde_higher_at_mode() {
        let x = Matrix::from_fn(20, 1, |i, _| 0.1 * (i as f64 - 10.0));
        let q = KernelDensity { bandwidth: 0.5 }
            .fit_unsupervised(&x, &Session::new("kde", "fit"))
            .expect("kde");
        let mut at0 = Matrix::zeros(1, 1);
        at0.set(0, 0, 0.0);
        let mut far = Matrix::zeros(1, 1);
        far.set(0, 0, 20.0);
        let l0 = q.value.predict(&at0, &Session::new("kde", "predict")).unwrap().value[0];
        let lf = q.value.predict(&far, &Session::new("kde", "predict")).unwrap().value[0];
        assert!(l0 > lf, "log-density at 0 ({l0}) should exceed far ({lf})");
    }
}
