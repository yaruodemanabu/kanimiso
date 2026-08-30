//! Feature selection, random Fourier features, and simple time-series maps.
//!
//! Supervised selectors that score against the full `y` (no cross-validation
//! note) raise [`IssueCode::TargetLeakageSuspected`]. Constant columns are
//! dropped by [`VarianceThreshold`] and ignored by [`SelectKBest`].

use crate::cluster::Linkage;
use crate::context::FitCtx;
use crate::data::{Matrix, Vector};
use crate::linalg::{least_squares, ridge_solve};
use crate::model_selection::{take_rows, take_vec, KFold};
use crate::rng::Rng;
use crate::special::{chi2_pvalue, digamma, f_pvalue};
use crate::traits::{Fit, FitUnsupervised, Transform};
use crate::validate::{inspect_classes, inspect_xy};
use faer::Mat;
use ojizou_san::Session;
use signlred::{
    slice_stats, Issue, IssueCode, Meaninglessness, Qualified, Result, Severity, SliceStats,
};

/// Re-export of [`crate::preprocess::PolynomialFeatures`] for a sklearn-like
/// `feature_extraction` surface.
pub use crate::preprocess::PolynomialFeatures;

/// Drop columns whose sample variance is at or below `threshold`.
#[derive(Clone, Debug)]
pub struct VarianceThreshold {
    /// Variance cutoff (0 drops exact constants).
    pub threshold: f64,
    variances: Vector,
    support: Vec<bool>,
    fitted: bool,
}

impl Default for VarianceThreshold {
    fn default() -> Self {
        Self {
            threshold: 0.0,
            variances: Vector::zeros(0),
            support: Vec::new(),
            fitted: false,
        }
    }
}

impl VarianceThreshold {
    /// Drop columns with variance ≤ `threshold`.
    pub fn new(threshold: f64) -> Self {
        Self {
            threshold,
            ..Self::default()
        }
    }

    /// Fitted per-column variances.
    pub fn variances(&self) -> &Vector {
        &self.variances
    }

    /// Mask of kept columns.
    pub fn support(&self) -> &[bool] {
        &self.support
    }
}

impl FitUnsupervised for VarianceThreshold {
    type Fitted = Self;
    fn fit_unsupervised(&mut self, x: &Matrix, session: &Session) -> Result<Qualified<Self>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let (n, p) = x.shape();
        self.variances = Vector::zeros(p);
        self.support = vec![false; p];
        for j in 0..p {
            let col: Vec<f64> = (0..n).map(|i| x.get(i, j)).collect();
            let st = slice_stats(&col);
            self.variances[j] = st.variance;
            self.support[j] = st.count >= 2 && st.variance > self.threshold;
            if st.count > 0 && st.is_constant(ctx.policy.near_zero_variance) {
                ctx.push(constant_col_issue(j, st));
            }
        }
        if self.support.iter().all(|s| !*s) {
            ctx.push(
                Issue::builder(IssueCode::MeaninglessFit)
                    .message("VarianceThreshold dropped every column")
                    .meaninglessness(Meaninglessness::vacuous(
                        "feature mask",
                        "no column exceeded the variance cutoff",
                        "lower the threshold or inspect the design",
                    ))
                    .build(),
            );
        }
        self.fitted = true;
        ctx.finish(self.clone())
    }
}

impl Transform for VarianceThreshold {
    fn transform(&self, x: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
        let mut ctx = FitCtx::with_session(session.child("transform"));
        if !self.fitted {
            ctx.push(Issue::builder(IssueCode::StaleState).build());
            return ctx.finish(x.clone());
        }
        ctx.finish(select_columns(x, &self.support))
    }
}

/// Keep the `k` features with largest f-regression scores (`corr²` with `y`).
#[derive(Clone, Debug)]
pub struct SelectKBest {
    /// Number of features to keep.
    pub k: usize,
    scores: Vector,
    support: Vec<bool>,
    fitted: bool,
}

impl Default for SelectKBest {
    fn default() -> Self {
        Self {
            k: 1,
            scores: Vector::zeros(0),
            support: Vec::new(),
            fitted: false,
        }
    }
}

impl SelectKBest {
    /// Keep `k` features.
    pub fn new(k: usize) -> Self {
        Self {
            k: k.max(1),
            ..Self::default()
        }
    }

    /// `corr²` scores (NaN for constant / unidentified columns).
    pub fn scores(&self) -> &Vector {
        &self.scores
    }

    /// Mask of kept columns.
    pub fn support(&self) -> &[bool] {
        &self.support
    }
}

impl Fit for SelectKBest {
    type Fitted = Self;
    fn fit(&mut self, x: &Matrix, y: &Vector, session: &Session) -> Result<Qualified<Self>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        ctx.push(
            Issue::builder(IssueCode::TargetLeakageSuspected)
                .severity(Severity::Advisory)
                .message(
                    "SelectKBest scored every feature against the full y; without a CV note this can leak the target into the selected set",
                )
                .build(),
        );
        let (n, p) = x.shape();
        self.scores = Vector::zeros(p);
        let yst = slice_stats(y.as_slice());
        for j in 0..p {
            let col: Vec<f64> = (0..n).map(|i| x.get(i, j)).collect();
            let xst = slice_stats(&col);
            self.scores[j] = pearson_sq(&col, xst, y.as_slice(), yst);
        }
        let mut order: Vec<usize> = (0..p).collect();
        order.sort_by(|a, b| {
            self.scores[*b]
                .partial_cmp(&self.scores[*a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        self.support = vec![false; p];
        for &j in order.iter().take(self.k.min(p)) {
            if self.scores[j].is_finite() {
                self.support[j] = true;
            }
        }
        if self.support.iter().all(|s| !*s) && p > 0 {
            self.support[order[0]] = true;
        }
        self.fitted = true;
        ctx.finish(self.clone())
    }
}

impl Transform for SelectKBest {
    fn transform(&self, x: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
        let mut ctx = FitCtx::with_session(session.child("transform"));
        if !self.fitted {
            ctx.push(Issue::builder(IssueCode::StaleState).build());
            return ctx.finish(x.clone());
        }
        ctx.finish(select_columns(x, &self.support))
    }
}

/// Recursive feature elimination wrapping a plain OLS score (`|β|`).
#[derive(Clone, Debug)]
pub struct Rfe {
    /// Features to keep.
    pub n_features_to_select: usize,
    support: Vec<bool>,
    ranking: Vec<usize>,
    fitted: bool,
}

impl Default for Rfe {
    fn default() -> Self {
        Self {
            n_features_to_select: 1,
            support: Vec::new(),
            ranking: Vec::new(),
            fitted: false,
        }
    }
}

impl Rfe {
    /// Eliminate down to `n_features_to_select` columns.
    pub fn new(n_features_to_select: usize) -> Self {
        Self {
            n_features_to_select: n_features_to_select.max(1),
            ..Self::default()
        }
    }

    /// Mask of kept columns.
    pub fn support(&self) -> &[bool] {
        &self.support
    }

    /// Elimination rank (1 = kept in the final set).
    pub fn ranking(&self) -> &[usize] {
        &self.ranking
    }
}

impl Fit for Rfe {
    type Fitted = Self;
    fn fit(&mut self, x: &Matrix, y: &Vector, session: &Session) -> Result<Qualified<Self>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        ctx.push(
            Issue::builder(IssueCode::TargetLeakageSuspected)
                .severity(Severity::Advisory)
                .message(
                    "RFE refits OLS on the full y at every elimination step; this is not a CV-safe selector",
                )
                .build(),
        );
        let p = x.ncols();
        let keep = self.n_features_to_select.min(p.max(1));
        let mut active: Vec<usize> = (0..p).collect();
        self.ranking = vec![0; p];
        let mut rank = p;
        while active.len() > keep {
            let sub = take_columns(x, &active);
            let Some(beta) = least_squares(&mut ctx.report, &sub, y, &ctx.policy) else {
                ctx.push(
                    Issue::builder(IssueCode::UnidentifiedModel)
                        .message("OLS score inside RFE failed; stopping elimination")
                        .build(),
                );
                break;
            };
            let mut worst = 0usize;
            let mut worst_abs = f64::INFINITY;
            for (t, _) in active.iter().enumerate() {
                let a = beta[t].abs();
                if a < worst_abs {
                    worst_abs = a;
                    worst = t;
                }
            }
            let dropped = active.remove(worst);
            self.ranking[dropped] = rank;
            rank -= 1;
        }
        self.support = vec![false; p];
        for &j in &active {
            self.support[j] = true;
            self.ranking[j] = 1;
        }
        self.fitted = true;
        ctx.finish(self.clone())
    }
}

impl Transform for Rfe {
    fn transform(&self, x: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
        let mut ctx = FitCtx::with_session(session.child("transform"));
        if !self.fitted {
            ctx.push(Issue::builder(IssueCode::StaleState).build());
            return ctx.finish(x.clone());
        }
        ctx.finish(select_columns(x, &self.support))
    }
}

/// Keep the top `percentile` of features by f-regression score.
#[derive(Clone, Debug)]
pub struct SelectPercentile {
    /// Percentile in `(0, 100]`.
    pub percentile: f64,
    scores: Vector,
    support: Vec<bool>,
    fitted: bool,
}

impl Default for SelectPercentile {
    fn default() -> Self {
        Self {
            percentile: 50.0,
            scores: Vector::zeros(0),
            support: Vec::new(),
            fitted: false,
        }
    }
}

impl SelectPercentile {
    /// Keep the top `percentile` percent of columns.
    pub fn new(percentile: f64) -> Self {
        Self {
            percentile,
            ..Self::default()
        }
    }

    /// `corr²` scores.
    pub fn scores(&self) -> &Vector {
        &self.scores
    }

    /// Mask of kept columns.
    pub fn support(&self) -> &[bool] {
        &self.support
    }
}

impl Fit for SelectPercentile {
    type Fitted = Self;
    fn fit(&mut self, x: &Matrix, y: &Vector, session: &Session) -> Result<Qualified<Self>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        if !self.percentile.is_finite() || self.percentile <= 0.0 || self.percentile > 100.0 {
            ctx.push(
                Issue::builder(IssueCode::InvalidWeight)
                    .severity(Severity::Warning)
                    .message(format!(
                        "SelectPercentile percentile={} not in (0, 100]; using 50",
                        self.percentile
                    ))
                    .build(),
            );
        }
        ctx.push(
            Issue::builder(IssueCode::TargetLeakageSuspected)
                .severity(Severity::Advisory)
                .message("SelectPercentile scored every feature against the full y")
                .build(),
        );
        let (n, p) = x.shape();
        self.scores = Vector::zeros(p);
        let yst = slice_stats(y.as_slice());
        for j in 0..p {
            let col: Vec<f64> = (0..n).map(|i| x.get(i, j)).collect();
            let xst = slice_stats(&col);
            self.scores[j] = pearson_sq(&col, xst, y.as_slice(), yst);
        }
        let pct = self.percentile.clamp(1e-6, 100.0);
        let k = ((p as f64) * pct / 100.0).ceil() as usize;
        let k = k.clamp(1, p.max(1));
        let mut order: Vec<usize> = (0..p).collect();
        order.sort_by(|a, b| {
            self.scores[*b]
                .partial_cmp(&self.scores[*a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        self.support = vec![false; p];
        for &j in order.iter().take(k) {
            if self.scores[j].is_finite() {
                self.support[j] = true;
            }
        }
        if self.support.iter().all(|s| !*s) && p > 0 {
            self.support[order[0]] = true;
        }
        self.fitted = true;
        ctx.finish(self.clone())
    }
}

impl Transform for SelectPercentile {
    fn transform(&self, x: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
        let mut ctx = FitCtx::with_session(session.child("transform"));
        if !self.fitted {
            ctx.push(Issue::builder(IssueCode::StaleState).build());
            return ctx.finish(x.clone());
        }
        ctx.finish(select_columns(x, &self.support))
    }
}

fn ols_mse(x: &Matrix, y: &Vector, policy: &signlred::Policy) -> f64 {
    let mut scratch = signlred::Report::new("sfs", "ols");
    match least_squares(&mut scratch, x, y, policy) {
        Some(beta) => {
            let r = y.sub(&x.matvec(&beta));
            r.dot(&r) / y.len().max(1) as f64
        }
        None => f64::INFINITY,
    }
}

/// Recursive feature elimination with K-fold OLS MSE to pick the subset size.
#[derive(Clone, Debug)]
pub struct RfeCv {
    /// Number of CV folds.
    pub n_splits: usize,
    support: Vec<bool>,
    n_features: usize,
    fitted: bool,
}

impl Default for RfeCv {
    fn default() -> Self {
        Self {
            n_splits: 3,
            support: Vec::new(),
            n_features: 1,
            fitted: false,
        }
    }
}

impl RfeCv {
    /// RFECV with `n_splits` folds.
    pub fn new(n_splits: usize) -> Self {
        Self {
            n_splits: n_splits.max(2),
            ..Self::default()
        }
    }

    /// Mask of kept columns.
    pub fn support(&self) -> &[bool] {
        &self.support
    }

    /// Chosen number of features.
    pub fn n_features(&self) -> usize {
        self.n_features
    }
}

impl Fit for RfeCv {
    type Fitted = Self;
    fn fit(&mut self, x: &Matrix, y: &Vector, session: &Session) -> Result<Qualified<Self>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        ctx.push(
            Issue::builder(IssueCode::TargetLeakageSuspected)
                .severity(Severity::Advisory)
                .message("RFECV still ranks features with full-data OLS between CV scores")
                .build(),
        );
        let p = x.ncols();
        if p == 0 {
            self.support.clear();
            self.fitted = true;
            return ctx.finish(self.clone());
        }
        let kf = KFold::new(self.n_splits.max(2));
        let folds = match kf.split(x.nrows(), &session.child("kfold")) {
            Ok(q) => q.value,
            Err(_) => Vec::new(),
        };
        let mut best_k = 1usize;
        let mut best_mse = f64::INFINITY;
        for k in 1..=p {
            let mut mses = Vec::new();
            for fold in &folds {
                let xtr = take_rows(x, &fold.train);
                let ytr = take_vec(y, &fold.train);
                let xte = take_rows(x, &fold.test);
                let yte = take_vec(y, &fold.test);
                let mut rfe = Rfe::new(k);
                let Ok(q) = rfe.fit(&xtr, &ytr, &session.child("rfe")) else {
                    continue;
                };
                let ztr = select_columns(&xte, q.value.support());
                if ztr.ncols() == 0 {
                    continue;
                }
                let mut scratch = signlred::Report::new("rfecv", "ols");
                let Some(beta) = least_squares(
                    &mut scratch,
                    &select_columns(&xtr, q.value.support()),
                    &ytr,
                    &ctx.policy,
                ) else {
                    continue;
                };
                let pred = ztr.matvec(&beta);
                let mut sse = 0.0;
                for i in 0..yte.len() {
                    let e = yte[i] - pred[i];
                    sse += e * e;
                }
                mses.push(sse / yte.len().max(1) as f64);
            }
            if mses.is_empty() {
                continue;
            }
            let m = mses.iter().sum::<f64>() / mses.len() as f64;
            if m < best_mse {
                best_mse = m;
                best_k = k;
            }
        }
        let mut final_rfe = Rfe::new(best_k);
        let q = final_rfe.fit(x, y, &session.child("rfe-final"))?;
        self.support = q.value.support().to_vec();
        self.n_features = self.support.iter().filter(|s| **s).count().max(1);
        self.fitted = true;
        ctx.finish(self.clone())
    }
}

impl Transform for RfeCv {
    fn transform(&self, x: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
        let mut ctx = FitCtx::with_session(session.child("transform"));
        if !self.fitted {
            ctx.push(Issue::builder(IssueCode::StaleState).build());
            return ctx.finish(x.clone());
        }
        ctx.finish(select_columns(x, &self.support))
    }
}

/// Forward sequential selection by in-sample OLS MSE (sklearn `SequentialFeatureSelector`).
#[derive(Clone, Debug)]
pub struct SequentialFeatureSelector {
    /// Features to keep.
    pub n_features_to_select: usize,
    support: Vec<bool>,
    fitted: bool,
}

impl Default for SequentialFeatureSelector {
    fn default() -> Self {
        Self {
            n_features_to_select: 1,
            support: Vec::new(),
            fitted: false,
        }
    }
}

impl SequentialFeatureSelector {
    /// Keep `n` features.
    pub fn new(n: usize) -> Self {
        Self {
            n_features_to_select: n.max(1),
            ..Self::default()
        }
    }

    /// Mask of kept columns.
    pub fn support(&self) -> &[bool] {
        &self.support
    }
}

impl Fit for SequentialFeatureSelector {
    type Fitted = Self;
    fn fit(&mut self, x: &Matrix, y: &Vector, session: &Session) -> Result<Qualified<Self>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        ctx.push(
            Issue::builder(IssueCode::TargetLeakageSuspected)
                .severity(Severity::Advisory)
                .message("SequentialFeatureSelector scored subsets on the full y")
                .build(),
        );
        let p = x.ncols();
        let keep = self.n_features_to_select.min(p.max(1));
        let mut chosen: Vec<usize> = Vec::new();
        let mut remaining: Vec<usize> = (0..p).collect();
        while chosen.len() < keep && !remaining.is_empty() {
            let mut best_j = 0usize;
            let mut best_mse = f64::INFINITY;
            for (t, &j) in remaining.iter().enumerate() {
                let mut cols = chosen.clone();
                cols.push(j);
                let sub = take_columns(x, &cols);
                let mse = ols_mse(&sub.with_intercept(), y, &ctx.policy);
                if !mse.is_finite() {
                    continue;
                }
                // Collinear copies of the same signal (x and 0.01 x) share MSE;
                // keep the earlier column rather than letting 1e-16 noise flip the pick.
                let tol = 1e-12 * (1.0 + best_mse.abs());
                if mse + tol < best_mse {
                    best_mse = mse;
                    best_j = t;
                }
            }
            chosen.push(remaining.remove(best_j));
        }
        self.support = vec![false; p];
        for j in chosen {
            self.support[j] = true;
        }
        if self.support.iter().all(|s| !*s) && p > 0 {
            self.support[0] = true;
        }
        self.fitted = true;
        ctx.finish(self.clone())
    }
}

impl Transform for SequentialFeatureSelector {
    fn transform(&self, x: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
        let mut ctx = FitCtx::with_session(session.child("transform"));
        if !self.fitted {
            ctx.push(Issue::builder(IssueCode::StaleState).build());
            return ctx.finish(x.clone());
        }
        ctx.finish(select_columns(x, &self.support))
    }
}

fn apply_univariate_mask(
    scores: &FeatureScores,
    keep: impl Fn(usize, f64) -> bool,
    ctx: &mut FitCtx,
) -> Vec<bool> {
    let p = scores.pvalues.len();
    let mut support = vec![false; p];
    for j in 0..p {
        if keep(j, scores.pvalues[j]) {
            support[j] = true;
        }
    }
    if support.iter().all(|s| !*s) && p > 0 {
        let mut best = 0usize;
        let mut bp = f64::INFINITY;
        for j in 0..p {
            if scores.pvalues[j] < bp {
                bp = scores.pvalues[j];
                best = j;
            }
        }
        support[best] = true;
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .severity(Severity::Warning)
                .message("no feature passed the p-value cutoff; keeping the smallest p")
                .build(),
        );
    }
    support
}

/// Keep columns with F-test p-value below `alpha` (sklearn `SelectFpr`).
#[derive(Clone, Debug)]
pub struct SelectFpr {
    /// Family-wise type-I rate for a single test.
    pub alpha: f64,
    support: Vec<bool>,
    fitted: bool,
}

impl Default for SelectFpr {
    fn default() -> Self {
        Self {
            alpha: 0.05,
            support: Vec::new(),
            fitted: false,
        }
    }
}

impl SelectFpr {
    /// FPR selector with the given `α`.
    pub fn new(alpha: f64) -> Self {
        Self {
            alpha,
            ..Self::default()
        }
    }

    /// Mask of kept columns.
    pub fn support(&self) -> &[bool] {
        &self.support
    }
}

impl Fit for SelectFpr {
    type Fitted = Self;
    fn fit(&mut self, x: &Matrix, y: &Vector, session: &Session) -> Result<Qualified<Self>> {
        let mut ctx = FitCtx::with_session(session.clone());
        let alpha = if self.alpha.is_finite() && self.alpha > 0.0 && self.alpha < 1.0 {
            self.alpha
        } else {
            ctx.push(
                Issue::builder(IssueCode::InvalidWeight)
                    .severity(Severity::Warning)
                    .message(format!(
                        "SelectFpr α={} not in (0, 1); using 0.05",
                        self.alpha
                    ))
                    .build(),
            );
            0.05
        };
        let scores = match f_regression(x, y, &session.child("fpr")) {
            Ok(q) => q.value,
            Err(e) => {
                ctx.push(e.primary);
                self.support = vec![true; x.ncols().min(1)];
                self.fitted = true;
                return ctx.finish(self.clone());
            }
        };
        self.support = apply_univariate_mask(&scores, |_, p| p <= alpha, &mut ctx);
        self.fitted = true;
        ctx.finish(self.clone())
    }
}

impl Transform for SelectFpr {
    fn transform(&self, x: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
        let mut ctx = FitCtx::with_session(session.child("transform"));
        if !self.fitted {
            ctx.push(Issue::builder(IssueCode::StaleState).build());
            return ctx.finish(x.clone());
        }
        ctx.finish(select_columns(x, &self.support))
    }
}

/// Benjamini–Hochberg FDR control (sklearn `SelectFdr`).
#[derive(Clone, Debug)]
pub struct SelectFdr {
    /// Target false-discovery rate.
    pub alpha: f64,
    support: Vec<bool>,
    fitted: bool,
}

impl Default for SelectFdr {
    fn default() -> Self {
        Self {
            alpha: 0.05,
            support: Vec::new(),
            fitted: false,
        }
    }
}

impl SelectFdr {
    /// FDR selector with the given `α`.
    pub fn new(alpha: f64) -> Self {
        Self {
            alpha,
            ..Self::default()
        }
    }

    /// Mask of kept columns.
    pub fn support(&self) -> &[bool] {
        &self.support
    }
}

impl Fit for SelectFdr {
    type Fitted = Self;
    fn fit(&mut self, x: &Matrix, y: &Vector, session: &Session) -> Result<Qualified<Self>> {
        let mut ctx = FitCtx::with_session(session.clone());
        let alpha = if self.alpha.is_finite() && self.alpha > 0.0 && self.alpha < 1.0 {
            self.alpha
        } else {
            ctx.push(
                Issue::builder(IssueCode::InvalidWeight)
                    .severity(Severity::Warning)
                    .message(format!(
                        "SelectFdr α={} not in (0, 1); using 0.05",
                        self.alpha
                    ))
                    .build(),
            );
            0.05
        };
        let scores = match f_regression(x, y, &session.child("fdr")) {
            Ok(q) => q.value,
            Err(e) => {
                ctx.push(e.primary);
                self.support = vec![x.ncols() > 0];
                if x.ncols() > 1 {
                    self.support.resize(x.ncols(), false);
                }
                self.fitted = true;
                return ctx.finish(self.clone());
            }
        };
        let p = scores.pvalues.len();
        let mut order: Vec<usize> = (0..p).collect();
        order.sort_by(|a, b| {
            scores.pvalues[*a]
                .partial_cmp(&scores.pvalues[*b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut cutoff = None;
        for (rank, &j) in order.iter().enumerate() {
            let thresh = alpha * (rank + 1) as f64 / p.max(1) as f64;
            if scores.pvalues[j] <= thresh {
                cutoff = Some(rank);
            }
        }
        let mut keep_idx = vec![false; p];
        if let Some(last) = cutoff {
            for &j in order.iter().take(last + 1) {
                keep_idx[j] = true;
            }
        }
        self.support = apply_univariate_mask(&scores, |j, _| keep_idx[j], &mut ctx);
        self.fitted = true;
        ctx.finish(self.clone())
    }
}

impl Transform for SelectFdr {
    fn transform(&self, x: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
        let mut ctx = FitCtx::with_session(session.child("transform"));
        if !self.fitted {
            ctx.push(Issue::builder(IssueCode::StaleState).build());
            return ctx.finish(x.clone());
        }
        ctx.finish(select_columns(x, &self.support))
    }
}

/// Bonferroni family-wise error selector (sklearn `SelectFwe`).
#[derive(Clone, Debug)]
pub struct SelectFwe {
    /// Family-wise type-I rate.
    pub alpha: f64,
    support: Vec<bool>,
    fitted: bool,
}

impl Default for SelectFwe {
    fn default() -> Self {
        Self {
            alpha: 0.05,
            support: Vec::new(),
            fitted: false,
        }
    }
}

impl SelectFwe {
    /// FWE selector with the given `α`.
    pub fn new(alpha: f64) -> Self {
        Self {
            alpha,
            ..Self::default()
        }
    }

    /// Mask of kept columns.
    pub fn support(&self) -> &[bool] {
        &self.support
    }
}

impl Fit for SelectFwe {
    type Fitted = Self;
    fn fit(&mut self, x: &Matrix, y: &Vector, session: &Session) -> Result<Qualified<Self>> {
        let mut ctx = FitCtx::with_session(session.clone());
        let alpha = if self.alpha.is_finite() && self.alpha > 0.0 && self.alpha < 1.0 {
            self.alpha
        } else {
            ctx.push(
                Issue::builder(IssueCode::InvalidWeight)
                    .severity(Severity::Warning)
                    .message(format!(
                        "SelectFwe α={} not in (0, 1); using 0.05",
                        self.alpha
                    ))
                    .build(),
            );
            0.05
        };
        let scores = match f_regression(x, y, &session.child("fwe")) {
            Ok(q) => q.value,
            Err(e) => {
                ctx.push(e.primary);
                self.support = vec![x.ncols() > 0];
                if x.ncols() > 1 {
                    self.support.resize(x.ncols(), false);
                }
                self.fitted = true;
                return ctx.finish(self.clone());
            }
        };
        let p = scores.pvalues.len().max(1);
        let thresh = alpha / p as f64;
        self.support = apply_univariate_mask(&scores, |_, pv| pv <= thresh, &mut ctx);
        self.fitted = true;
        ctx.finish(self.clone())
    }
}

impl Transform for SelectFwe {
    fn transform(&self, x: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
        let mut ctx = FitCtx::with_session(session.child("transform"));
        if !self.fitted {
            ctx.push(Issue::builder(IssueCode::StaleState).build());
            return ctx.finish(x.clone());
        }
        ctx.finish(select_columns(x, &self.support))
    }
}

/// Univariate selector with a named strategy (sklearn `GenericUnivariateSelect`).
#[derive(Clone, Debug)]
pub struct GenericUnivariateSelect {
    /// One of `k_best`, `percentile`, `fpr`, `fwe`.
    pub mode: UnivariateMode,
    /// `k`, percentile, or `α` depending on `mode`.
    pub param: f64,
    support: Vec<bool>,
    fitted: bool,
}

/// Strategy for [`GenericUnivariateSelect`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnivariateMode {
    /// Keep the `k` highest F scores.
    KBest,
    /// Keep the top percentile of F scores.
    Percentile,
    /// Keep p-values below `α`.
    Fpr,
    /// Bonferroni: keep p-values below `α / p`.
    Fwe,
}

impl Default for GenericUnivariateSelect {
    fn default() -> Self {
        Self {
            mode: UnivariateMode::KBest,
            param: 1.0,
            support: Vec::new(),
            fitted: false,
        }
    }
}

impl GenericUnivariateSelect {
    /// Selector with the given mode and parameter.
    pub fn new(mode: UnivariateMode, param: f64) -> Self {
        Self {
            mode,
            param,
            ..Self::default()
        }
    }

    /// Mask of kept columns.
    pub fn support(&self) -> &[bool] {
        &self.support
    }
}

impl Fit for GenericUnivariateSelect {
    type Fitted = Self;
    fn fit(&mut self, x: &Matrix, y: &Vector, session: &Session) -> Result<Qualified<Self>> {
        self.support = match self.mode {
            UnivariateMode::KBest => {
                let mut s = SelectKBest::new(self.param.max(1.0) as usize);
                s.fit(x, y, session)?.value.support().to_vec()
            }
            UnivariateMode::Percentile => {
                let mut s = SelectPercentile::new(self.param);
                s.fit(x, y, session)?.value.support().to_vec()
            }
            UnivariateMode::Fpr => {
                let mut s = SelectFpr::new(self.param);
                s.fit(x, y, session)?.value.support().to_vec()
            }
            UnivariateMode::Fwe => {
                let mut s = SelectFwe::new(self.param);
                s.fit(x, y, session)?.value.support().to_vec()
            }
        };
        self.fitted = true;
        let mut ctx = FitCtx::with_session(session.child("finish"));
        ctx.finish(self.clone())
    }
}

impl Transform for GenericUnivariateSelect {
    fn transform(&self, x: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
        let mut ctx = FitCtx::with_session(session.child("transform"));
        if !self.fitted {
            ctx.push(Issue::builder(IssueCode::StaleState).build());
            return ctx.finish(x.clone());
        }
        ctx.finish(select_columns(x, &self.support))
    }
}

/// Random Fourier features for an RBF kernel (Rahimi & Recht).
#[derive(Clone, Debug)]
pub struct RbfSampler {
    /// Number of Monte-Carlo frequencies.
    pub n_components: usize,
    /// RBF `γ` in `exp(-γ‖x−x'‖²)`.
    pub gamma: f64,
    /// PRNG seed.
    pub seed: u64,
    weights: Matrix,
    offset: Vector,
    fitted: bool,
}

impl Default for RbfSampler {
    fn default() -> Self {
        Self {
            n_components: 100,
            gamma: 1.0,
            seed: 0,
            weights: Matrix::zeros(0, 0),
            offset: Vector::zeros(0),
            fitted: false,
        }
    }
}

impl RbfSampler {
    /// Sampler with `n_components` features and RBF `γ`.
    pub fn new(n_components: usize, gamma: f64) -> Self {
        Self {
            n_components: n_components.max(1),
            gamma,
            ..Self::default()
        }
    }
}

impl FitUnsupervised for RbfSampler {
    type Fitted = Self;
    fn fit_unsupervised(&mut self, x: &Matrix, session: &Session) -> Result<Qualified<Self>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let p = x.ncols();
        let m = self.n_components;
        let mut rng = Rng::new(self.seed);
        let sd = (2.0 * self.gamma).sqrt();
        self.weights = Matrix::from_fn(p, m, |_, _| rng.standard_normal() * sd);
        self.offset =
            Vector::from_iter((0..m).map(|_| rng.uniform_range(0.0, 2.0 * std::f64::consts::PI)));
        self.fitted = true;
        ctx.finish(self.clone())
    }
}

impl Transform for RbfSampler {
    fn transform(&self, x: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
        let mut ctx = FitCtx::with_session(session.child("transform"));
        if !self.fitted {
            ctx.push(Issue::builder(IssueCode::StaleState).build());
            return ctx.finish(x.clone());
        }
        if x.ncols() != self.weights.nrows() {
            ctx.push(
                Issue::builder(IssueCode::DimensionMismatch)
                    .message("RbfSampler transform p ≠ fitted p")
                    .build(),
            );
        }
        let scale = (2.0 / self.n_components as f64).sqrt();
        let m = self.offset.len();
        let out = Matrix::from_fn(x.nrows(), m, |i, k| {
            let mut s = self.offset[k];
            for j in 0..x.ncols().min(self.weights.nrows()) {
                s += x.get(i, j) * self.weights.get(j, k);
            }
            scale * s.cos()
        });
        ctx.finish(out)
    }
}

/// Additive χ² kernel map (Vedaldi & Zisserman / sklearn `AdditiveChi2Sampler`).
///
/// Each non-negative coordinate `x_j` is mapped to
/// `√x_j {cos,sin}(ω_k log(x_j+ε))`. Do not pass the output dimension to
/// identification — a 20×2 table with 4 Fourier features is identified.
/// Negative entries are a [`IssueCode::NonPositiveSeries`] warning.
#[derive(Clone, Debug)]
pub struct AdditiveChi2Sampler {
    /// Frequencies per input column.
    pub n_components: usize,
    /// Spacing of `ω_k = (k+1) · sample_interval`.
    pub sample_interval: f64,
    fitted: bool,
}

impl Default for AdditiveChi2Sampler {
    fn default() -> Self {
        Self {
            n_components: 2,
            sample_interval: 1.0,
            fitted: false,
        }
    }
}

impl AdditiveChi2Sampler {
    /// Sampler with `n_components` frequencies per column.
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components: n_components.max(1),
            ..Self::default()
        }
    }
}

impl FitUnsupervised for AdditiveChi2Sampler {
    type Fitted = Self;
    fn fit_unsupervised(&mut self, x: &Matrix, session: &Session) -> Result<Qualified<Self>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        if self.sample_interval <= 0.0 || !self.sample_interval.is_finite() {
            ctx.push(
                Issue::builder(IssueCode::InvalidWeight)
                    .severity(Severity::Warning)
                    .message(format!(
                        "AdditiveChi2Sampler interval={} is not positive; using 1",
                        self.sample_interval
                    ))
                    .build(),
            );
            self.sample_interval = 1.0;
        }
        let mut neg = false;
        for i in 0..x.nrows() {
            for j in 0..x.ncols() {
                if x.get(i, j) < 0.0 {
                    neg = true;
                }
            }
        }
        if neg {
            ctx.push(
                Issue::builder(IssueCode::NonPositiveSeries)
                    .severity(Severity::Warning)
                    .message("AdditiveChi2Sampler saw negative entries; they are clipped to 0")
                    .build(),
            );
        }
        self.fitted = true;
        ctx.finish(self.clone())
    }
}

impl Transform for AdditiveChi2Sampler {
    fn transform(&self, x: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
        let mut ctx = FitCtx::with_session(session.child("transform"));
        if !self.fitted {
            ctx.push(Issue::builder(IssueCode::StaleState).build());
            return ctx.finish(x.clone());
        }
        let m = self.n_components.max(1);
        let interval = if self.sample_interval > 0.0 {
            self.sample_interval
        } else {
            1.0
        };
        let out_p = x.ncols() * m * 2;
        let out = Matrix::from_fn(x.nrows(), out_p, |i, c| {
            let j = c / (2 * m);
            let rem = c % (2 * m);
            let k = rem / 2;
            let use_sin = rem % 2 == 1;
            let v = x.get(i, j).max(0.0);
            let omega = (k + 1) as f64 * interval;
            let arg = omega * (v + 1e-8).ln();
            let amp = v.sqrt();
            if use_sin {
                amp * arg.sin()
            } else {
                amp * arg.cos()
            }
        });
        ctx.finish(out)
    }
}

/// Skewed χ² kernel map (sklearn `SkewedChi2Sampler`).
///
/// `z = log(x + c)` is mapped with random Fourier features. Entries with
/// `x + c ≤ 0` are a [`IssueCode::NonPositiveSeries`] warning and are clipped.
/// Do not pass the output dimension to identification.
#[derive(Clone, Debug)]
pub struct SkewedChi2Sampler {
    /// Frequencies.
    pub n_components: usize,
    /// Offset `c > 0`.
    pub skewedness: f64,
    /// PRNG seed.
    pub seed: u64,
    weights: Matrix,
    offset: Vector,
    fitted: bool,
}

impl Default for SkewedChi2Sampler {
    fn default() -> Self {
        Self {
            n_components: 8,
            skewedness: 1.0,
            seed: 1,
            weights: Matrix::zeros(0, 0),
            offset: Vector::zeros(0),
            fitted: false,
        }
    }
}

impl SkewedChi2Sampler {
    /// Sampler with `n_components` frequencies.
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components: n_components.max(1),
            ..Self::default()
        }
    }
}

impl FitUnsupervised for SkewedChi2Sampler {
    type Fitted = Self;
    fn fit_unsupervised(&mut self, x: &Matrix, session: &Session) -> Result<Qualified<Self>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        if self.skewedness <= 0.0 || !self.skewedness.is_finite() {
            ctx.push(
                Issue::builder(IssueCode::InvalidWeight)
                    .severity(Severity::Warning)
                    .message(format!(
                        "SkewedChi2Sampler c={} is not positive; using 1",
                        self.skewedness
                    ))
                    .build(),
            );
            self.skewedness = 1.0;
        }
        let mut bad = false;
        for i in 0..x.nrows() {
            for j in 0..x.ncols() {
                if x.get(i, j) + self.skewedness <= 0.0 {
                    bad = true;
                }
            }
        }
        if bad {
            ctx.push(
                Issue::builder(IssueCode::NonPositiveSeries)
                    .severity(Severity::Warning)
                    .message("SkewedChi2Sampler saw x+c≤0; those entries are clipped")
                    .build(),
            );
        }
        let m = self.n_components.max(1);
        let mut rng = Rng::new(self.seed);
        self.weights = Matrix::from_fn(x.ncols(), m, |_, _| rng.standard_normal());
        self.offset =
            Vector::from_iter((0..m).map(|_| rng.uniform_range(0.0, 2.0 * std::f64::consts::PI)));
        self.fitted = true;
        ctx.finish(self.clone())
    }
}

impl Transform for SkewedChi2Sampler {
    fn transform(&self, x: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
        let mut ctx = FitCtx::with_session(session.child("transform"));
        if !self.fitted {
            ctx.push(Issue::builder(IssueCode::StaleState).build());
            return ctx.finish(x.clone());
        }
        let c = self.skewedness.max(1e-12);
        let m = self.offset.len();
        let scale = (2.0 / m.max(1) as f64).sqrt();
        let out = Matrix::from_fn(x.nrows(), m, |i, k| {
            let mut s = self.offset[k];
            for j in 0..x.ncols().min(self.weights.nrows()) {
                let z = (x.get(i, j) + c).max(1e-12).ln();
                s += z * self.weights.get(j, k);
            }
            scale * s.cos()
        });
        ctx.finish(out)
    }
}

/// Tensor / count sketch for a polynomial kernel (sklearn `PolynomialCountSketch`).
///
/// Degree-2 features are hashed into `n_components` bins. Do not identify on
/// the sketch width.
#[derive(Clone, Debug)]
pub struct PolynomialCountSketch {
    /// Sketch width.
    pub n_components: usize,
    /// Polynomial degree (`2` = pairwise).
    pub degree: usize,
    /// PRNG seed for hash signs.
    pub seed: u64,
    hash_idx: Vec<usize>,
    hash_sign: Vec<f64>,
    fitted: bool,
}

impl Default for PolynomialCountSketch {
    fn default() -> Self {
        Self {
            n_components: 16,
            degree: 2,
            seed: 1,
            hash_idx: Vec::new(),
            hash_sign: Vec::new(),
            fitted: false,
        }
    }
}

impl PolynomialCountSketch {
    /// Sketch of width `n_components`.
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components: n_components.max(2),
            ..Self::default()
        }
    }
}

impl FitUnsupervised for PolynomialCountSketch {
    type Fitted = Self;
    fn fit_unsupervised(&mut self, x: &Matrix, session: &Session) -> Result<Qualified<Self>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let p = x.ncols();
        let m = self.n_components.max(2);
        let mut rng = Rng::new(self.seed);
        self.hash_idx = (0..p).map(|_| rng.below(m)).collect();
        self.hash_sign = (0..p)
            .map(|_| if rng.uniform() < 0.5 { -1.0 } else { 1.0 })
            .collect();
        self.fitted = true;
        ctx.finish(self.clone())
    }
}

impl Transform for PolynomialCountSketch {
    fn transform(&self, x: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
        let mut ctx = FitCtx::with_session(session.child("transform"));
        if !self.fitted {
            ctx.push(Issue::builder(IssueCode::StaleState).build());
            return ctx.finish(x.clone());
        }
        let m = self.n_components.max(2);
        let p = x.ncols().min(self.hash_idx.len());
        let mut out = Matrix::zeros(x.nrows(), m);
        for i in 0..x.nrows() {
            let mut sketch = vec![0.0; m];
            for j in 0..p {
                sketch[self.hash_idx[j]] += self.hash_sign[j] * x.get(i, j);
            }
            if self.degree >= 2 {
                let linear = sketch.clone();
                for a in 0..m {
                    for b in 0..m {
                        sketch[(a + b) % m] += 0.5 * linear[a] * linear[b];
                    }
                }
            }
            for k in 0..m {
                out.set(i, k, sketch[k] / (m as f64).sqrt());
            }
        }
        ctx.finish(out)
    }
}

/// Signed-hash feature map (sklearn `FeatureHasher`).
///
/// Column indices are hashed into `n_features` bins. Output width is not
/// identification `p`.
#[derive(Clone, Debug)]
pub struct FeatureHasher {
    /// Hash-bin count.
    pub n_features: usize,
}

impl Default for FeatureHasher {
    fn default() -> Self {
        Self { n_features: 16 }
    }
}

impl FeatureHasher {
    /// Hasher with `n_features` bins.
    pub fn new(n_features: usize) -> Self {
        Self {
            n_features: n_features.max(1),
        }
    }

    fn bin(j: usize, n: usize) -> (usize, f64) {
        let h = (j as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let sign = if (h >> 1) & 1 == 0 { 1.0 } else { -1.0 };
        ((h as usize) % n.max(1), sign)
    }
}

impl Transform for FeatureHasher {
    fn transform(&self, x: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
        let mut ctx = FitCtx::with_session(session.child("transform"));
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let m = self.n_features.max(1);
        let mut out = Matrix::zeros(x.nrows(), m);
        for i in 0..x.nrows() {
            for j in 0..x.ncols() {
                let (b, s) = Self::bin(j, m);
                out.set(i, b, out.get(i, b) + s * x.get(i, j));
            }
        }
        ctx.finish(out)
    }
}

/// Nyström approximation to an RBF kernel map.
#[derive(Clone, Debug)]
pub struct Nystroem {
    /// Number of landmark rows.
    pub n_components: usize,
    /// RBF `γ`.
    pub gamma: f64,
    /// PRNG seed for landmark sampling.
    pub seed: u64,
    landmarks: Matrix,
    norm: Mat<f64>,
    fitted: bool,
}

impl Default for Nystroem {
    fn default() -> Self {
        Self {
            n_components: 10,
            gamma: 1.0,
            seed: 1,
            landmarks: Matrix::zeros(0, 0),
            norm: Mat::<f64>::zeros(0, 0),
            fitted: false,
        }
    }
}

impl Nystroem {
    /// Nyström map with `n_components` landmarks.
    pub fn new(n_components: usize, gamma: f64) -> Self {
        Self {
            n_components: n_components.max(1),
            gamma,
            ..Self::default()
        }
    }
}

impl FitUnsupervised for Nystroem {
    type Fitted = Self;
    fn fit_unsupervised(&mut self, x: &Matrix, session: &Session) -> Result<Qualified<Self>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let m = self.n_components.min(x.nrows()).max(1);
        let mut rng = Rng::new(self.seed);
        let idx = rng.sample_indices(x.nrows(), m);
        self.landmarks = Matrix::from_fn(m, x.ncols(), |r, c| x.get(idx[r], c));
        let mut k = Mat::<f64>::zeros(m, m);
        for i in 0..m {
            for j in 0..=i {
                let kij = rbf_entry(&self.landmarks, i, &self.landmarks, j, self.gamma);
                k[(i, j)] = kij;
                k[(j, i)] = kij;
            }
        }
        let Some((vals, u)) = crate::linalg::symmetric_eigen(&mut ctx.report, &k, &ctx.policy)
        else {
            ctx.push(
                Issue::builder(IssueCode::KernelNotPd)
                    .message("Nyström landmark kernel eigendecomposition failed")
                    .build(),
            );
            self.norm = Mat::<f64>::zeros(m, m);
            self.fitted = true;
            return ctx.finish(self.clone());
        };
        let mut kept = 0usize;
        for &λ in &vals {
            if λ > ctx.policy.rank_tol_relative {
                kept += 1;
            } else if λ < 0.0 {
                ctx.push(
                    Issue::builder(IssueCode::NegativeEigenvalueDropped)
                        .message(format!("Nyström kernel eigenvalue {λ:.3e} < 0"))
                        .build(),
                );
            }
        }
        if kept == 0 {
            ctx.push(
                Issue::builder(IssueCode::RankZero)
                    .message("Nyström kernel has no positive eigenvalue")
                    .build(),
            );
        }
        // W^{+}½ ≈ U diag(λ⁺½) — columns of `norm` are the embedding directions.
        let r = vals.len().min(u.ncols());
        self.norm = Mat::<f64>::zeros(m, r);
        for c in 0..r {
            let s = if vals[c] > ctx.policy.rank_tol_relative {
                1.0 / vals[c].sqrt()
            } else {
                0.0
            };
            for i in 0..m {
                self.norm[(i, c)] = u[(i, c)] * s;
            }
        }
        self.fitted = true;
        ctx.finish(self.clone())
    }
}

impl Transform for Nystroem {
    fn transform(&self, x: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
        let mut ctx = FitCtx::with_session(session.child("transform"));
        if !self.fitted {
            ctx.push(Issue::builder(IssueCode::StaleState).build());
            return ctx.finish(x.clone());
        }
        let m = self.landmarks.nrows();
        let r = self.norm.ncols();
        let out = Matrix::from_fn(x.nrows(), r, |i, c| {
            let mut s = 0.0;
            for t in 0..m {
                let kxt = rbf_entry(x, i, &self.landmarks, t, self.gamma);
                s += kxt * self.norm[(t, c)];
            }
            s
        });
        ctx.finish(out)
    }
}

/// Histogram mutual-information classifier feature scorer / selector.
#[derive(Clone, Debug)]
pub struct MutualInfoClassif {
    /// Histogram bins per feature.
    pub n_bins: usize,
    /// Keep the top `k` features (all if `k` exceeds `p`).
    pub k: usize,
    scores: Vector,
    support: Vec<bool>,
    fitted: bool,
}

impl Default for MutualInfoClassif {
    fn default() -> Self {
        Self {
            n_bins: 8,
            k: 1,
            scores: Vector::zeros(0),
            support: Vec::new(),
            fitted: false,
        }
    }
}

impl MutualInfoClassif {
    /// Selector with `n_bins` and top-`k`.
    pub fn new(n_bins: usize, k: usize) -> Self {
        Self {
            n_bins: n_bins.max(2),
            k: k.max(1),
            ..Self::default()
        }
    }

    /// Estimated I(X_j; Y) in nats.
    pub fn scores(&self) -> &Vector {
        &self.scores
    }

    /// Mask of kept columns.
    pub fn support(&self) -> &[bool] {
        &self.support
    }
}

impl Fit for MutualInfoClassif {
    type Fitted = Self;
    fn fit(&mut self, x: &Matrix, y: &Vector, session: &Session) -> Result<Qualified<Self>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        ctx.push(
            Issue::builder(IssueCode::TargetLeakageSuspected)
                .severity(Severity::Advisory)
                .message(
                    "MutualInfoClassif estimates I(X;Y) on the same labels later used for training",
                )
                .build(),
        );
        let (n, p) = x.shape();
        self.scores = Vector::zeros(p);
        for j in 0..p {
            let col: Vec<f64> = (0..n).map(|i| x.get(i, j)).collect();
            self.scores[j] = histogram_mi(&col, y.as_slice(), self.n_bins);
        }
        let mut order: Vec<usize> = (0..p).collect();
        order.sort_by(|a, b| {
            self.scores[*b]
                .partial_cmp(&self.scores[*a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        self.support = vec![false; p];
        for &j in order.iter().take(self.k.min(p)) {
            self.support[j] = true;
        }
        self.fitted = true;
        ctx.finish(self.clone())
    }
}

impl Transform for MutualInfoClassif {
    fn transform(&self, x: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
        let mut ctx = FitCtx::with_session(session.child("transform"));
        if !self.fitted {
            ctx.push(Issue::builder(IssueCode::StaleState).build());
            return ctx.finish(x.clone());
        }
        ctx.finish(select_columns(x, &self.support))
    }
}

/// Build a design whose column `j` is `y[t - lags[j]]` (leading rows are NaN).
pub fn lag_features(y: &Vector, lags: &[usize]) -> Matrix {
    let n = y.len();
    let p = lags.len();
    Matrix::from_fn(n, p, |i, j| {
        let lag = lags[j];
        if i >= lag {
            y[i - lag]
        } else {
            f64::NAN
        }
    })
}

/// Trailing rolling mean of window `window` (NaN until the window is full).
pub fn rolling_mean(y: &Vector, window: usize) -> Vector {
    let w = window.max(1);
    let mut out = Vector::zeros(y.len());
    let mut run = 0.0;
    let mut q: Vec<f64> = Vec::new();
    for i in 0..y.len() {
        let v = y[i];
        q.push(v);
        if v.is_finite() {
            run += v;
        }
        if q.len() > w {
            let old = q.remove(0);
            if old.is_finite() {
                run -= old;
            }
        }
        if q.len() == w {
            let k = q.iter().filter(|x| x.is_finite()).count() as f64;
            out[i] = if k > 0.0 { run / k } else { f64::NAN };
        } else {
            out[i] = f64::NAN;
        }
    }
    out
}

fn select_columns(x: &Matrix, support: &[bool]) -> Matrix {
    let cols: Vec<usize> = support
        .iter()
        .enumerate()
        .filter_map(|(j, s)| if *s { Some(j) } else { None })
        .collect();
    if cols.is_empty() {
        return Matrix::zeros(x.nrows(), 0);
    }
    Matrix::from_fn(x.nrows(), cols.len(), |i, t| x.get(i, cols[t]))
}

fn take_columns(x: &Matrix, cols: &[usize]) -> Matrix {
    Matrix::from_fn(x.nrows(), cols.len(), |i, t| x.get(i, cols[t]))
}

fn pearson_sq(a: &[f64], sa: signlred::SliceStats, b: &[f64], sb: signlred::SliceStats) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 || sa.std() == 0.0 || sb.std() == 0.0 {
        return 0.0;
    }
    let mut s = 0.0;
    let mut k = 0.0;
    for i in 0..n {
        if a[i].is_finite() && b[i].is_finite() {
            s += (a[i] - sa.mean) * (b[i] - sb.mean);
            k += 1.0;
        }
    }
    if k < 2.0 {
        return 0.0;
    }
    let r = s / ((k - 1.0) * sa.std() * sb.std());
    if r.is_finite() {
        r * r
    } else {
        0.0
    }
}

fn rbf_entry(a: &Matrix, ia: usize, b: &Matrix, ib: usize, gamma: f64) -> f64 {
    let mut d2 = 0.0;
    for c in 0..a.ncols().min(b.ncols()) {
        let d = a.get(ia, c) - b.get(ib, c);
        d2 += d * d;
    }
    (-gamma * d2).exp()
}

fn constant_col_issue(index: usize, stats: SliceStats) -> Issue {
    Issue::builder(IssueCode::ConstantFeature)
        .message(format!(
            "feature {index} is constant on [{:.6e}, {:.6e}]",
            stats.min, stats.max
        ))
        .metric("feature_index", index as f64)
        .metric("feature_std", stats.std())
        .build()
}

fn histogram_mi(x: &[f64], y: &[f64], n_bins: usize) -> f64 {
    let n = x.len().min(y.len());
    if n == 0 {
        return 0.0;
    }
    let st = slice_stats(x);
    let mut classes: Vec<i64> = Vec::new();
    for &v in y.iter().take(n) {
        if v.is_finite() {
            let lab = v.round() as i64;
            if !classes.contains(&lab) {
                classes.push(lab);
            }
        }
    }
    classes.sort_unstable();
    if classes.is_empty() || st.count < 2 {
        return 0.0;
    }
    let nb = n_bins.max(2);
    let span = (st.max - st.min).max(1e-15);
    let mut joint = vec![0.0; nb * classes.len()];
    let mut mx = vec![0.0; nb];
    let mut my = vec![0.0; classes.len()];
    let mut tot: f64 = 0.0;
    for i in 0..n {
        if !x[i].is_finite() || !y[i].is_finite() {
            continue;
        }
        let mut b = ((x[i] - st.min) / span * nb as f64).floor() as usize;
        if b >= nb {
            b = nb - 1;
        }
        let lab = y[i].round() as i64;
        let c = match classes.iter().position(|k| *k == lab) {
            Some(c) => c,
            None => continue,
        };
        joint[b * classes.len() + c] += 1.0;
        mx[b] += 1.0;
        my[c] += 1.0;
        tot += 1.0;
    }
    if tot <= 0.0 {
        return 0.0;
    }
    let mut mi: f64 = 0.0;
    for b in 0..nb {
        for c in 0..classes.len() {
            let pxy = joint[b * classes.len() + c] / tot;
            let px = mx[b] / tot;
            let py = my[c] / tot;
            if pxy > 0.0 && px > 0.0 && py > 0.0 {
                mi += pxy * (pxy / (px * py)).ln();
            }
        }
    }
    if !mi.is_finite() {
        0.0
    } else {
        mi
    }
}

/// Supervised feature scores plus p-values.
#[derive(Clone, Debug)]
pub struct FeatureScores {
    /// Test statistic per column.
    pub scores: Vector,
    /// Upper-tail p-value per column.
    pub pvalues: Vector,
}

/// One-way ANOVA F of each column against a class label (`f_classif`).
pub fn f_classif(x: &Matrix, y: &Vector, session: &Session) -> Result<Qualified<FeatureScores>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
    let counts = inspect_classes(&mut ctx.report, y, &ctx.policy);
    ctx.push(
        Issue::builder(IssueCode::TargetLeakageSuspected)
            .severity(Severity::Advisory)
            .message("f_classif scores every column against the full y")
            .build(),
    );
    let k = counts.len();
    let n = x.nrows();
    let mut scores = Vector::zeros(x.ncols());
    let mut pvalues = Vector::zeros(x.ncols());
    if k < 2 {
        return ctx.finish(FeatureScores { scores, pvalues });
    }
    let dfb = (k - 1) as f64;
    let dfw = (n.saturating_sub(k)) as f64;
    for j in 0..x.ncols() {
        let mut ss_w = 0.0;
        let mut ss_b = 0.0;
        let grand = x.column(j).mean();
        for (lab, cnt) in &counts {
            let mut s = 0.0;
            let mut q = 0.0;
            let mut m = 0.0;
            for i in 0..n {
                if y[i].is_finite() && y[i].round() as i64 == *lab {
                    let v = x.get(i, j);
                    s += v;
                    q += v * v;
                    m += 1.0;
                }
            }
            let mean = if m > 0.0 { s / m } else { 0.0 };
            ss_w += q - mean * s;
            let d = mean - grand;
            ss_b += (*cnt as f64) * d * d;
        }
        let msb = ss_b / dfb.max(1.0);
        let msw = ss_w / dfw.max(1.0);
        if msw <= ctx.policy.near_zero_variance {
            if msb <= ctx.policy.near_zero_variance {
                scores[j] = 0.0;
                pvalues[j] = 1.0;
            } else {
                ctx.push(
                    Issue::builder(IssueCode::DegenerateDistribution)
                        .message(format!("feature {j} has zero within-class variance"))
                        .build(),
                );
                scores[j] = f64::INFINITY;
                pvalues[j] = 0.0;
            }
        } else {
            let f = msb / msw;
            scores[j] = f;
            pvalues[j] = f_pvalue(f.max(0.0), dfb, dfw.max(1.0));
        }
    }
    ctx.finish(FeatureScores { scores, pvalues })
}

/// F-regression: \(F = r^2/(1-r^2)\,(n-2)\) of each column against `y`.
pub fn f_regression(x: &Matrix, y: &Vector, session: &Session) -> Result<Qualified<FeatureScores>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
    ctx.push(
        Issue::builder(IssueCode::TargetLeakageSuspected)
            .severity(Severity::Advisory)
            .message("f_regression scores every column against the full y")
            .build(),
    );
    let n = x.nrows() as f64;
    let yst = slice_stats(y.as_slice());
    let mut scores = Vector::zeros(x.ncols());
    let mut pvalues = Vector::zeros(x.ncols());
    for j in 0..x.ncols() {
        let col: Vec<f64> = (0..x.nrows()).map(|i| x.get(i, j)).collect();
        let xst = slice_stats(&col);
        let r2 = pearson_sq(&col, xst, y.as_slice(), yst);
        if r2 >= 1.0 - 1e-15 {
            scores[j] = f64::INFINITY;
            pvalues[j] = 0.0;
        } else {
            let f = (r2 / (1.0 - r2).max(1e-18)) * (n - 2.0).max(1.0);
            scores[j] = f;
            pvalues[j] = f_pvalue(f.max(0.0), 1.0, (n - 2.0).max(1.0));
        }
    }
    ctx.finish(FeatureScores { scores, pvalues })
}

/// k-NN mutual information of each column with `y` (sklearn `mutual_info_regression`).
///
/// Kraskov I^(2) with Chebyshev neighbours. The scores use the full `y`
/// ([`IssueCode::TargetLeakageSuspected`]).
pub fn mutual_info_regression(
    x: &Matrix,
    y: &Vector,
    session: &Session,
) -> Result<Qualified<FeatureScores>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
    ctx.push(
        Issue::builder(IssueCode::TargetLeakageSuspected)
            .severity(Severity::Advisory)
            .message("mutual_info_regression scores every column against the full y")
            .build(),
    );
    let n = x.nrows().min(y.len());
    let k = 3usize.min(n.saturating_sub(1)).max(1);
    let mut scores = Vector::zeros(x.ncols());
    let mut pvalues = Vector::zeros(x.ncols());
    if n < 6 {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .severity(Severity::Warning)
                .message(format!("mutual_info_regression n={n} is thin for k-NN MI"))
                .build(),
        );
    }
    for j in 0..x.ncols() {
        let mut mi = 0.0;
        let mut used = 0.0;
        for i in 0..n {
            if !x.get(i, j).is_finite() || !y[i].is_finite() {
                continue;
            }
            let mut dists: Vec<f64> = (0..n)
                .filter(|&u| u != i)
                .map(|u| {
                    let dx = (x.get(u, j) - x.get(i, j)).abs();
                    let dy = (y[u] - y[i]).abs();
                    dx.max(dy)
                })
                .collect();
            if dists.len() < k {
                continue;
            }
            dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let eps = dists[k - 1];
            let mut nx = 0usize;
            let mut ny = 0usize;
            for u in 0..n {
                if u == i {
                    continue;
                }
                if (x.get(u, j) - x.get(i, j)).abs() <= eps {
                    nx += 1;
                }
                if (y[u] - y[i]).abs() <= eps {
                    ny += 1;
                }
            }
            mi += digamma(k as f64) - digamma(nx as f64 + 1.0) - digamma(ny as f64 + 1.0)
                + digamma(n as f64);
            used += 1.0;
        }
        scores[j] = if used > 0.0 {
            (mi / used).max(0.0)
        } else {
            0.0
        };
        pvalues[j] = f64::NAN;
    }
    ctx.finish(FeatureScores { scores, pvalues })
}

/// Discrete-label Kraskov / Ross I^(2) mutual information per column.
///
/// For each row the k-th Chebyshev neighbour is taken **inside its class**;
/// `m_i` counts every sample (any class) inside that radius. Scores use the
/// full `y` ([`IssueCode::TargetLeakageSuspected`]).
pub fn mutual_info_classif(
    x: &Matrix,
    y: &Vector,
    session: &Session,
) -> Result<Qualified<FeatureScores>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
    let _ = inspect_classes(&mut ctx.report, y, &ctx.policy);
    ctx.push(
        Issue::builder(IssueCode::TargetLeakageSuspected)
            .severity(Severity::Advisory)
            .message("mutual_info_classif scores every column against the full y")
            .build(),
    );
    let n = x.nrows().min(y.len());
    let k = 3usize.min(n.saturating_sub(1)).max(1);
    let mut scores = Vector::zeros(x.ncols());
    let mut pvalues = Vector::zeros(x.ncols());
    if n < 6 {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .severity(Severity::Warning)
                .message(format!("mutual_info_classif n={n} is thin for k-NN MI"))
                .build(),
        );
    }
    let labs: Vec<i64> = (0..n)
        .map(|i| {
            if y[i].is_finite() {
                y[i].round() as i64
            } else {
                0
            }
        })
        .collect();
    for j in 0..x.ncols() {
        let mut mi = 0.0;
        let mut used = 0.0;
        for i in 0..n {
            if !x.get(i, j).is_finite() || !y[i].is_finite() {
                continue;
            }
            let yi = labs[i];
            let n_y = labs.iter().filter(|&&l| l == yi).count();
            let mut d_same: Vec<f64> = (0..n)
                .filter(|&u| u != i && labs[u] == yi)
                .map(|u| (x.get(u, j) - x.get(i, j)).abs())
                .collect();
            if d_same.len() < k {
                continue;
            }
            d_same.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let eps = d_same[k - 1];
            let mut m = 0usize;
            for u in 0..n {
                if u == i {
                    continue;
                }
                if (x.get(u, j) - x.get(i, j)).abs() <= eps {
                    m += 1;
                }
            }
            mi += digamma(n as f64) - digamma(n_y as f64) + digamma(k as f64)
                - digamma((m as f64).max(1.0));
            used += 1.0;
        }
        scores[j] = if used > 0.0 {
            (mi / used).max(0.0)
        } else {
            0.0
        };
        pvalues[j] = f64::NAN;
    }
    ctx.finish(FeatureScores { scores, pvalues })
}

/// χ² scores of non-negative columns against a class label.
pub fn chi2(x: &Matrix, y: &Vector, session: &Session) -> Result<Qualified<FeatureScores>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
    let counts = inspect_classes(&mut ctx.report, y, &ctx.policy);
    ctx.push(
        Issue::builder(IssueCode::TargetLeakageSuspected)
            .severity(Severity::Advisory)
            .message("chi2 scores every column against the full y")
            .build(),
    );
    let n = x.nrows() as f64;
    let k = counts.len();
    let mut scores = Vector::zeros(x.ncols());
    let mut pvalues = Vector::zeros(x.ncols());
    let mut any_neg = false;
    for j in 0..x.ncols() {
        for i in 0..x.nrows() {
            if x.get(i, j) < 0.0 {
                any_neg = true;
                break;
            }
        }
    }
    if any_neg {
        ctx.push(
            Issue::builder(IssueCode::InvalidWeight)
                .severity(Severity::Warning)
                .message("chi2 requires non-negative X; negative columns are scored as NaN")
                .build(),
        );
    }
    if k < 2 {
        return ctx.finish(FeatureScores { scores, pvalues });
    }
    for j in 0..x.ncols() {
        let mut neg = false;
        let mut col_sum = 0.0;
        for i in 0..x.nrows() {
            let v = x.get(i, j);
            if v < 0.0 {
                neg = true;
                break;
            }
            col_sum += v;
        }
        if neg || col_sum <= 0.0 {
            scores[j] = f64::NAN;
            pvalues[j] = f64::NAN;
            continue;
        }
        let mut stat = 0.0;
        for (lab, cnt) in &counts {
            let mut obs = 0.0;
            for i in 0..x.nrows() {
                if y[i].is_finite() && y[i].round() as i64 == *lab {
                    obs += x.get(i, j);
                }
            }
            let exp = col_sum * (*cnt as f64 / n.max(1.0));
            if exp > 1e-18 {
                let d = obs - exp;
                stat += d * d / exp;
            }
        }
        scores[j] = stat;
        pvalues[j] = chi2_pvalue(stat.max(0.0), (k.saturating_sub(1)) as f64);
    }
    ctx.finish(FeatureScores { scores, pvalues })
}

/// Agglomerative clustering on **columns**, then average members (sklearn
/// `FeatureAgglomeration`).
#[derive(Clone, Debug)]
pub struct FeatureAgglomeration {
    /// Number of output features (clusters of columns).
    pub n_clusters: usize,
    /// Linkage on column Euclidean distances.
    pub linkage: Linkage,
    labels: Vector,
    fitted: bool,
}

impl Default for FeatureAgglomeration {
    fn default() -> Self {
        Self {
            n_clusters: 2,
            linkage: Linkage::Average,
            labels: Vector::zeros(0),
            fitted: false,
        }
    }
}

impl FeatureAgglomeration {
    /// Keep `n_clusters` agglomerated features.
    pub fn new(n_clusters: usize) -> Self {
        Self {
            n_clusters: n_clusters.max(1),
            ..Self::default()
        }
    }

    /// Cluster id per original column.
    pub fn labels(&self) -> &Vector {
        &self.labels
    }
}

impl FitUnsupervised for FeatureAgglomeration {
    type Fitted = Self;
    fn fit_unsupervised(&mut self, x: &Matrix, session: &Session) -> Result<Qualified<Self>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let p = x.ncols();
        let mut want = self.n_clusters.max(1);
        if want > p && p > 0 {
            ctx.push(
                Issue::builder(IssueCode::Overparameterized)
                    .message(format!("n_clusters={want} > p={p}"))
                    .build(),
            );
            want = p;
        }
        if p == 0 {
            self.labels = Vector::zeros(0);
            self.fitted = true;
            return ctx.finish(self.clone());
        }
        let dist = Matrix::from_fn(p, p, |i, j| {
            if i == j {
                0.0
            } else {
                let mut s = 0.0;
                for r in 0..x.nrows() {
                    let d = x.get(r, i) - x.get(r, j);
                    s += d * d;
                }
                s.sqrt()
            }
        });
        let mut clusters: Vec<Vec<usize>> = (0..p).map(|i| vec![i]).collect();
        while clusters.len() > want {
            let mut bi = 0usize;
            let mut bj = 1usize;
            let mut best = f64::INFINITY;
            for i in 0..clusters.len() {
                for j in (i + 1)..clusters.len() {
                    let d = feature_link(&clusters[i], &clusters[j], &dist, self.linkage);
                    if d < best {
                        best = d;
                        bi = i;
                        bj = j;
                    }
                }
            }
            let mut merged = clusters[bi].clone();
            merged.extend_from_slice(&clusters[bj]);
            if bi > bj {
                clusters.remove(bi);
                clusters.remove(bj);
            } else {
                clusters.remove(bj);
                clusters.remove(bi);
            }
            clusters.push(merged);
        }
        let mut labels = Vector::zeros(p);
        for (c, members) in clusters.iter().enumerate() {
            for &j in members {
                labels[j] = c as f64;
            }
        }
        self.labels = labels;
        self.fitted = true;
        ctx.finish(self.clone())
    }
}

impl Transform for FeatureAgglomeration {
    fn transform(&self, x: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
        let mut ctx = FitCtx::with_session(session.child("transform"));
        if !self.fitted {
            ctx.push(Issue::builder(IssueCode::StaleState).build());
            return ctx.finish(x.clone());
        }
        if x.ncols() != self.labels.len() {
            ctx.push(
                Issue::builder(IssueCode::DimensionMismatch)
                    .message("FeatureAgglomeration transform column count ≠ fitted p")
                    .build(),
            );
        }
        let k = self.n_clusters.max(1);
        let mut out = Matrix::zeros(x.nrows(), k);
        let mut den = vec![0.0; k];
        for j in 0..x.ncols().min(self.labels.len()) {
            let c = self.labels[j].round().clamp(0.0, (k - 1) as f64) as usize;
            den[c] += 1.0;
            for i in 0..x.nrows() {
                out.set(i, c, out.get(i, c) + x.get(i, j));
            }
        }
        for c in 0..k {
            if den[c] <= 0.0 {
                ctx.push(
                    Issue::builder(IssueCode::EmptyCluster)
                        .message(format!("feature cluster {c} is empty"))
                        .build(),
                );
                continue;
            }
            for i in 0..x.nrows() {
                out.set(i, c, out.get(i, c) / den[c]);
            }
        }
        ctx.finish(out)
    }
}

fn feature_link(a: &[usize], b: &[usize], dist: &Matrix, linkage: Linkage) -> f64 {
    match linkage {
        Linkage::Single => {
            let mut m = f64::INFINITY;
            for &i in a {
                for &j in b {
                    m = m.min(dist.get(i, j));
                }
            }
            m
        }
        Linkage::Complete => {
            let mut m = f64::NEG_INFINITY;
            for &i in a {
                for &j in b {
                    m = m.max(dist.get(i, j));
                }
            }
            m
        }
        Linkage::Average => {
            let mut s = 0.0;
            let mut c = 0.0;
            for &i in a {
                for &j in b {
                    s += dist.get(i, j);
                    c += 1.0;
                }
            }
            if c == 0.0 {
                f64::INFINITY
            } else {
                s / c
            }
        }
    }
}

/// Keep features whose ridge |coef| is at least `threshold` × max |coef|
/// (sklearn `SelectFromModel` with a linear base).
#[derive(Clone, Debug)]
pub struct SelectFromModel {
    /// Fraction of the largest absolute coefficient.
    pub threshold: f64,
    /// Ridge penalty used to score features.
    pub alpha: f64,
    support: Vec<bool>,
    coef: Vector,
    fitted: bool,
}

impl Default for SelectFromModel {
    fn default() -> Self {
        Self {
            threshold: 0.1,
            alpha: 1.0,
            support: Vec::new(),
            coef: Vector::zeros(0),
            fitted: false,
        }
    }
}

impl SelectFromModel {
    /// Selector with the given relative threshold.
    pub fn new(threshold: f64) -> Self {
        Self {
            threshold,
            ..Self::default()
        }
    }

    /// Boolean mask of kept columns.
    pub fn support(&self) -> &[bool] {
        &self.support
    }

    /// Ridge coefficients used for ranking.
    pub fn coef(&self) -> &Vector {
        &self.coef
    }
}

impl Fit for SelectFromModel {
    type Fitted = Self;
    fn fit(&mut self, x: &Matrix, y: &Vector, session: &Session) -> Result<Qualified<Self>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        let (xc, _) = x.centered();
        let ymean = y.mean();
        let yc = Vector::from_iter(y.as_slice().iter().map(|v| v - ymean));
        let coef = ridge_solve(&mut ctx.report, &xc, &yc, self.alpha.max(0.0), &ctx.policy)
            .unwrap_or_else(|| Vector::zeros(x.ncols()));
        let max_abs = coef.max_abs();
        let cut = self.threshold.max(0.0) * max_abs;
        self.support = (0..coef.len())
            .map(|j| coef[j].abs() >= cut && max_abs > 0.0)
            .collect();
        if !self.support.iter().any(|s| *s) && x.ncols() > 0 {
            ctx.push(
                Issue::builder(IssueCode::InterceptOnlyCollapse)
                    .message("SelectFromModel kept 0 columns")
                    .build(),
            );
            if !self.support.is_empty() {
                let mut best = 0usize;
                for j in 1..coef.len() {
                    if coef[j].abs() > coef[best].abs() {
                        best = j;
                    }
                }
                self.support[best] = true;
            }
        }
        ctx.push(
            Issue::builder(IssueCode::TargetLeakageSuspected)
                .message("SelectFromModel scored features on the full y")
                .build(),
        );
        self.coef = coef;
        self.fitted = true;
        ctx.finish(self.clone())
    }
}

impl Transform for SelectFromModel {
    fn transform(&self, x: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
        let mut ctx = FitCtx::with_session(session.child("transform"));
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        if !self.fitted {
            ctx.push(Issue::builder(IssueCode::PartialFitBeforeInit).build());
            return ctx.finish(x.clone());
        }
        if x.ncols() != self.support.len() {
            ctx.push(
                Issue::builder(IssueCode::DimensionMismatch)
                    .message("SelectFromModel column count ≠ fitted p")
                    .build(),
            );
        }
        let keep: Vec<usize> = self
            .support
            .iter()
            .enumerate()
            .filter_map(|(j, s)| if *s { Some(j) } else { None })
            .collect();
        let out = Matrix::from_fn(x.nrows(), keep.len(), |i, k| {
            let j = keep[k];
            if j < x.ncols() {
                x.get(i, j)
            } else {
                0.0
            }
        });
        ctx.finish(out)
    }
}

/// A 12-feature catch22-style sketch of a univariate series (sktime / pycatch22).
///
/// Features: mean, std, ACF(1), OLS slope on time, zero-crossing rate,
/// 5-bin histogram mode, outlier fraction \(|z|>2\), mean |Δ|, longest run
/// above the mean, 3-mean residual stderr, argmax of |ACF|, ACF(2).
pub fn catch22(y: &Vector, session: &Session) -> Result<Qualified<Vector>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(
        &mut ctx.report,
        &Matrix::from_vector(y),
        Some(y),
        &ctx.policy,
    );
    let n = y.len();
    if n < 4 {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .severity(Severity::Warning)
                .message("catch22 on n<4 is a sketch, not a catch22 identification")
                .build(),
        );
    }
    let mean = y.mean();
    let std = y.std();
    let acf_at = |lag: usize| -> f64 {
        if n <= lag || std <= 0.0 {
            return f64::NAN;
        }
        let mut s = 0.0;
        let mut k = 0.0;
        for t in lag..n {
            s += (y[t] - mean) * (y[t - lag] - mean);
            k += 1.0;
        }
        if k > 0.0 {
            s / (k * std * std)
        } else {
            f64::NAN
        }
    };
    let acf1 = acf_at(1);
    let acf2 = acf_at(2);
    let mut xtx = 0.0;
    let mut xty = 0.0;
    let tmean = (n.saturating_sub(1)) as f64 / 2.0;
    for i in 0..n {
        let t = i as f64 - tmean;
        xtx += t * t;
        xty += t * (y[i] - mean);
    }
    let slope = if xtx > 0.0 { xty / xtx } else { 0.0 };
    let mut crossings = 0.0;
    for i in 1..n {
        if (y[i] - mean) * (y[i - 1] - mean) < 0.0 {
            crossings += 1.0;
        }
    }
    let zcr = if n > 1 {
        crossings / (n - 1) as f64
    } else {
        0.0
    };
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for &v in y.as_slice() {
        if v.is_finite() {
            lo = lo.min(v);
            hi = hi.max(v);
        }
    }
    let mut hist = [0usize; 5];
    if hi > lo {
        for &v in y.as_slice() {
            if !v.is_finite() {
                continue;
            }
            let b = (((v - lo) / (hi - lo)) * 4.999).floor() as usize;
            hist[b.min(4)] += 1;
        }
    }
    let mode_bin = hist
        .iter()
        .enumerate()
        .max_by_key(|(_, c)| *c)
        .map(|(i, _)| i)
        .unwrap_or(0) as f64;
    let mut outliers = 0.0;
    if std > 0.0 {
        for &v in y.as_slice() {
            if ((v - mean) / std).abs() > 2.0 {
                outliers += 1.0;
            }
        }
    }
    let outlier_frac = if n > 0 { outliers / n as f64 } else { 0.0 };
    let mut mad = 0.0;
    for i in 1..n {
        mad += (y[i] - y[i - 1]).abs();
    }
    mad = if n > 1 { mad / (n - 1) as f64 } else { 0.0 };
    let mut best_run = 0usize;
    let mut run = 0usize;
    for &v in y.as_slice() {
        if v >= mean {
            run += 1;
            best_run = best_run.max(run);
        } else {
            run = 0;
        }
    }
    let mut sse3 = 0.0;
    let mut k3 = 0.0;
    for i in 1..n.saturating_sub(1) {
        let m = (y[i - 1] + y[i] + y[i + 1]) / 3.0;
        let e = y[i] - m;
        sse3 += e * e;
        k3 += 1.0;
    }
    let se3 = if k3 > 1.0 {
        (sse3 / (k3 - 1.0)).sqrt()
    } else {
        0.0
    };
    let mut acf_peak = 1.0;
    let mut peak = acf1.abs();
    for lag in 1..=n.min(8).saturating_sub(1).max(1) {
        let a = acf_at(lag).abs();
        if a > peak {
            peak = a;
            acf_peak = lag as f64;
        }
    }
    let feat = Vector::from_slice(&[
        mean,
        std,
        acf1,
        slope,
        zcr,
        mode_bin,
        outlier_frac,
        mad,
        best_run as f64,
        se3,
        acf_peak,
        acf2,
    ]);
    ctx.finish(feat)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::{Fit, FitUnsupervised, Transform};
    use ojizou_san::Session;

    #[test]
    fn variance_threshold_drops_constant() {
        let x = Matrix::from_fn(10, 2, |i, j| if j == 0 { 5.0 } else { i as f64 });
        let session = Session::new("vt", "fit");
        let mut vt = VarianceThreshold::new(0.0);
        vt.fit_unsupervised(&x, &session).expect("fit");
        let z = vt.transform(&x, &session).expect("transform").value;
        assert_eq!(z.ncols(), 1);
        assert!((z.get(3, 0) - 3.0).abs() < 1e-12);
        assert!(!vt.support()[0]);
        assert!(vt.support()[1]);
    }

    #[test]
    fn select_k_best_keeps_signal() {
        let x = Matrix::from_fn(20, 3, |i, j| match j {
            0 => i as f64,
            1 => ((i * 17 + 5) % 7) as f64,
            _ => ((i * 3) % 2) as f64,
        });
        let y = Vector::from_iter((0..20).map(|i| i as f64));
        let session = Session::new("skb", "fit");
        let mut sel = SelectKBest::new(1);
        sel.fit(&x, &y, &session).expect("fit");
        assert!(sel.support()[0], "scores={:?}", sel.scores().as_slice());
        assert_eq!(sel.support().iter().filter(|s| **s).count(), 1);
        let z = sel.transform(&x, &session).expect("transform").value;
        assert_eq!(z.ncols(), 1);
        assert!((z.get(5, 0) - 5.0).abs() < 1e-12);
        assert!(session.ledger().len() > 0);
    }

    #[test]
    fn f_scores_and_agglomeration() {
        let x = Matrix::from_fn(20, 3, |i, j| match j {
            0 => i as f64,
            1 => ((i * 17) % 5) as f64,
            _ => {
                if i < 10 {
                    0.0
                } else {
                    1.0
                }
            }
        });
        let y = Vector::from_iter((0..20).map(|i| if i < 10 { 0.0 } else { 1.0 }));
        let fc = f_classif(&x, &y, &Session::new("f", "c")).unwrap();
        assert!(fc.value.scores[2] > fc.value.scores[1]);
        let fr = f_regression(&x, &y, &Session::new("f", "r")).unwrap();
        assert!(fr.value.scores[0].is_finite() || fr.value.scores[2].is_finite());
        let mi = mutual_info_regression(&x, &y, &Session::new("f", "mi")).unwrap();
        assert!(mi.value.scores[0] >= mi.value.scores[1] - 1e-6 || mi.value.scores[2].is_finite());
        let mic = mutual_info_classif(&x, &y, &Session::new("f", "mic")).unwrap();
        assert!(mic.value.scores.as_slice().iter().all(|v| v.is_finite()));
        assert!(mic.value.scores[2] >= 0.0);
        let xnn = Matrix::from_fn(20, 2, |i, j| {
            if j == 0 {
                if i < 10 {
                    3.0
                } else {
                    1.0
                }
            } else {
                1.0
            }
        });
        let ch = chi2(&xnn, &y, &Session::new("f", "chi")).unwrap();
        assert!(ch.value.scores[0] > 0.0);
        let mut fa = FeatureAgglomeration::new(2);
        fa.fit_unsupervised(&x, &Session::new("fa", "fit")).unwrap();
        let z = fa.transform(&x, &Session::new("fa", "t")).unwrap().value;
        assert_eq!(z.ncols(), 2);
        assert_eq!(z.nrows(), 20);
    }

    #[test]
    fn select_from_model_and_catch22() {
        let x = Matrix::from_fn(24, 3, |i, j| match j {
            0 => i as f64,
            1 => ((i * 13) % 5) as f64,
            _ => 0.01 * (i as f64),
        });
        let y = Vector::from_iter((0..24).map(|i| 2.0 * i as f64));
        let mut sel = SelectFromModel::new(0.2);
        sel.fit(&x, &y, &Session::new("sfm", "fit")).unwrap();
        assert!(sel.support()[0]);
        let z = sel.transform(&x, &Session::new("sfm", "t")).unwrap().value;
        assert!(z.ncols() >= 1);
        let yts = Vector::from_iter((0..32).map(|i| (i as f64).sin() + 0.1 * i as f64));
        let c = catch22(&yts, &Session::new("c22", "fit")).unwrap().value;
        assert_eq!(c.len(), 12);
        assert!(c.as_slice().iter().all(|v| v.is_finite()));
        let mut sp = SelectPercentile::new(40.0);
        sp.fit(&x, &y, &Session::new("sp", "fit")).unwrap();
        assert!(sp.support()[0]);
        let mut sfs = SequentialFeatureSelector::new(1);
        sfs.fit(&x, &y, &Session::new("sfs", "fit")).unwrap();
        assert!(sfs.support()[0]);
        let mut rfecv = RfeCv::new(3);
        rfecv.fit(&x, &y, &Session::new("rfecv", "fit")).unwrap();
        assert!(rfecv.support().iter().any(|s| *s));
        let mut fpr = SelectFpr::new(0.05);
        fpr.fit(&x, &y, &Session::new("fpr", "fit")).unwrap();
        assert!(fpr.support()[0]);
        let mut fdr = SelectFdr::new(0.05);
        fdr.fit(&x, &y, &Session::new("fdr", "fit")).unwrap();
        assert!(fdr.support()[0]);
        let xnn = Matrix::from_fn(16, 2, |i, j| 0.2 + (i + j) as f64);
        let mut chi = AdditiveChi2Sampler::new(2);
        chi.fit_unsupervised(&xnn, &Session::new("achi", "fit"))
            .unwrap();
        let z = chi
            .transform(&xnn, &Session::new("achi", "t"))
            .unwrap()
            .value;
        assert_eq!(z.ncols(), 8);
        assert!(z.get(0, 0).is_finite());
        let mut sk = SkewedChi2Sampler::new(6);
        sk.fit_unsupervised(&xnn, &Session::new("sch", "fit"))
            .unwrap();
        let zs = sk.transform(&xnn, &Session::new("sch", "t")).unwrap().value;
        assert_eq!(zs.ncols(), 6);
        assert!(zs.get(0, 0).is_finite());
        let mut pcs = PolynomialCountSketch::new(8);
        pcs.fit_unsupervised(&xnn, &Session::new("pcs", "fit"))
            .unwrap();
        let zp = pcs
            .transform(&xnn, &Session::new("pcs", "t"))
            .unwrap()
            .value;
        assert_eq!(zp.ncols(), 8);
        assert!(zp.get(0, 0).is_finite());
        let mut fwe = SelectFwe::new(0.05);
        fwe.fit(&x, &y, &Session::new("fwe", "fit")).unwrap();
        assert!(fwe.support()[0]);
        let mut guni = GenericUnivariateSelect::new(UnivariateMode::KBest, 1.0);
        guni.fit(&x, &y, &Session::new("guni", "fit")).unwrap();
        assert!(guni.support().iter().any(|s| *s));
        let zh = FeatureHasher::new(8)
            .transform(&xnn, &Session::new("fh", "t"))
            .unwrap()
            .value;
        assert_eq!(zh.ncols(), 8);
        assert!(zh.get(0, 0).is_finite());
    }
}
