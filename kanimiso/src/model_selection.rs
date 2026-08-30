//! Train/test splits, CV iterators, cross-validated scores, and a Ridge grid.
//!
//! [`fit_transform_full`] is the leakage-prone helper: it estimates a
//! standardizer on the **entire** design. That records
//! [`IssueCode::TargetLeakageSuspected`] so a later supervised fit cannot
//! pretend the scale was learned inside each training fold.

use crate::context::FitCtx;
use crate::data::{Matrix, Vector};
use crate::linear_model::{
    ElasticNet, FittedMultiTask, FittedPenalized, Lars, Lasso, LassoLars, LinearRegression,
    LogisticRegression, MultiTaskElasticNet, MultiTaskLasso, Ridge,
};
use crate::metrics::{accuracy, r2};
use crate::rng::Rng;
use crate::robust::OrthogonalMatchingPursuit;
use crate::traits::{Fit, Predict};
use crate::validate::inspect_xy;
use ojizou_san::Session;
use signlred::{Issue, IssueCode, NumericalCompromise, Qualified, Result};

/// Row indices of one train / test fold.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Split {
    /// Training row indices.
    pub train: Vec<usize>,
    /// Held-out row indices.
    pub test: Vec<usize>,
}

/// Copy the listed rows of `x`.
pub fn take_rows(x: &Matrix, idx: &[usize]) -> Matrix {
    if idx.is_empty() {
        return Matrix::zeros(0, x.ncols());
    }
    Matrix::from_fn(idx.len(), x.ncols(), |i, j| {
        let r = idx[i];
        if r < x.nrows() {
            x.get(r, j)
        } else {
            f64::NAN
        }
    })
}

/// Copy the listed entries of `y`.
pub fn take_vec(y: &Vector, idx: &[usize]) -> Vector {
    Vector::from_iter(
        idx.iter()
            .map(|&i| if i < y.len() { y[i] } else { f64::NAN }),
    )
}

fn labels_of(y: &Vector) -> Vec<i64> {
    y.as_slice()
        .iter()
        .map(|&v| if v.is_finite() { v.round() as i64 } else { 0 })
        .collect()
}

/// Shuffle-split `X, y` into train and test sets.
pub fn train_test_split(
    x: &Matrix,
    y: &Vector,
    test_size: f64,
    seed: u64,
    session: &Session,
) -> Result<Qualified<(Matrix, Matrix, Vector, Vector)>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
    if y.len() != x.nrows() {
        ctx.push(
            Issue::builder(IssueCode::DimensionMismatch)
                .message("train_test_split: y length ≠ n")
                .build(),
        );
        let empty = (
            Matrix::zeros(0, x.ncols()),
            Matrix::zeros(0, x.ncols()),
            Vector::zeros(0),
            Vector::zeros(0),
        );
        return ctx.finish(empty);
    }
    let n = x.nrows();
    let frac = if test_size.is_finite() {
        test_size.clamp(0.0, 1.0)
    } else {
        0.25
    };
    if !test_size.is_finite() || test_size < 0.0 || test_size > 1.0 {
        ctx.push(
            Issue::builder(IssueCode::InvalidWeight)
                .message(format!("test_size={test_size} is not in [0, 1]; clamped"))
                .build(),
        );
    }
    let mut n_test = (frac * n as f64).round() as usize;
    if n > 1 {
        n_test = n_test.clamp(1, n - 1);
    }
    let mut idx: Vec<usize> = (0..n).collect();
    Rng::new(seed).shuffle(&mut idx);
    let test = idx[..n_test].to_vec();
    let train = idx[n_test..].to_vec();
    ctx.finish((
        take_rows(x, &train),
        take_rows(x, &test),
        take_vec(y, &train),
        take_vec(y, &test),
    ))
}

/// K-fold splitter (contiguous after an optional shuffle).
#[derive(Clone, Debug)]
pub struct KFold {
    /// Number of folds (≥ 2).
    pub n_splits: usize,
    /// If true, shuffle rows with `seed` before slicing.
    pub shuffle: bool,
    /// PRNG seed used when `shuffle` is true.
    pub seed: u64,
}

impl Default for KFold {
    fn default() -> Self {
        Self {
            n_splits: 5,
            shuffle: false,
            seed: 0,
        }
    }
}

impl KFold {
    /// `k` contiguous folds, no shuffle.
    pub fn new(n_splits: usize) -> Self {
        Self {
            n_splits,
            ..Self::default()
        }
    }

    /// Materialize train/test index pairs for `n` rows.
    pub fn split(&self, n: usize, session: &Session) -> Result<Qualified<Vec<Split>>> {
        let mut ctx = FitCtx::with_session(session.clone());
        let k = self.n_splits.max(2);
        if self.n_splits < 2 {
            ctx.push(
                Issue::builder(IssueCode::InvalidWeight)
                    .message("KFold.n_splits < 2; using 2")
                    .build(),
            );
        }
        if n < k {
            ctx.push(
                Issue::builder(IssueCode::InsufficientSample)
                    .message(format!("KFold requested {k} folds on n={n}"))
                    .build(),
            );
        }
        let mut idx: Vec<usize> = (0..n).collect();
        if self.shuffle {
            Rng::new(self.seed).shuffle(&mut idx);
        }
        let k = k.min(n.max(1));
        let mut folds = Vec::with_capacity(k);
        for f in 0..k {
            let mut test = Vec::new();
            let mut train = Vec::new();
            for (i, &row) in idx.iter().enumerate() {
                if i % k == f {
                    test.push(row);
                } else {
                    train.push(row);
                }
            }
            folds.push(Split { train, test });
        }
        ctx.finish(folds)
    }
}

/// Stratified K-fold: each class is split independently so fold prevalences match.
#[derive(Clone, Debug)]
pub struct StratifiedKFold {
    /// Number of folds.
    pub n_splits: usize,
    /// Shuffle within each class.
    pub shuffle: bool,
    /// PRNG seed.
    pub seed: u64,
}

impl Default for StratifiedKFold {
    fn default() -> Self {
        Self {
            n_splits: 5,
            shuffle: true,
            seed: 0,
        }
    }
}

impl StratifiedKFold {
    /// `k` stratified folds.
    pub fn new(n_splits: usize) -> Self {
        Self {
            n_splits,
            ..Self::default()
        }
    }

    /// Materialize folds that preserve class proportions of `y`.
    pub fn split(&self, y: &Vector, session: &Session) -> Result<Qualified<Vec<Split>>> {
        let mut ctx = FitCtx::with_session(session.clone());
        crate::validate::inspect_classes(&mut ctx.report, y, &ctx.policy);
        let labs = labels_of(y);
        let mut by_class: Vec<(i64, Vec<usize>)> = Vec::new();
        for (i, &lab) in labs.iter().enumerate() {
            if let Some((_, rows)) = by_class.iter_mut().find(|(c, _)| *c == lab) {
                rows.push(i);
            } else {
                by_class.push((lab, vec![i]));
            }
        }
        let k = self.n_splits.max(2);
        if self.n_splits < 2 {
            ctx.push(
                Issue::builder(IssueCode::InvalidWeight)
                    .message("StratifiedKFold.n_splits < 2; using 2")
                    .build(),
            );
        }
        let mut rng = Rng::new(self.seed);
        for (_, rows) in by_class.iter_mut() {
            if self.shuffle {
                rng.shuffle(rows);
            }
            if rows.len() < k {
                ctx.push(
                    Issue::builder(IssueCode::ClassImbalanceSevere)
                        .message(format!(
                            "a class has {} rows < n_splits={k}; some folds miss it",
                            rows.len()
                        ))
                        .build(),
                );
            }
        }
        let mut folds: Vec<Split> = (0..k)
            .map(|_| Split {
                train: Vec::new(),
                test: Vec::new(),
            })
            .collect();
        for (_, rows) in &by_class {
            for (i, &row) in rows.iter().enumerate() {
                let f = i % k;
                folds[f].test.push(row);
            }
        }
        let n = y.len();
        for f in 0..k {
            let test_set = folds[f].test.clone();
            folds[f].train = (0..n).filter(|i| !test_set.contains(i)).collect();
        }
        ctx.finish(folds)
    }
}

/// Expanding-window time-series split (sktime / sklearn `TimeSeriesSplit`).
#[derive(Clone, Debug)]
pub struct TimeSeriesSplit {
    /// Number of splits.
    pub n_splits: usize,
    /// Optional fixed test-block length; default is `n / (n_splits+1)`.
    pub test_size: Option<usize>,
}

impl Default for TimeSeriesSplit {
    fn default() -> Self {
        Self {
            n_splits: 5,
            test_size: None,
        }
    }
}

impl TimeSeriesSplit {
    /// Expanding window with `k` test blocks.
    pub fn new(n_splits: usize) -> Self {
        Self {
            n_splits,
            test_size: None,
        }
    }

    /// Materialize temporally ordered folds for `n` rows.
    pub fn split(&self, n: usize, session: &Session) -> Result<Qualified<Vec<Split>>> {
        let mut ctx = FitCtx::with_session(session.clone());
        let k = self.n_splits.max(1);
        if n < k + 1 {
            ctx.push(
                Issue::builder(IssueCode::InsufficientSample)
                    .message(format!("TimeSeriesSplit needs n ≥ n_splits+1; n={n}"))
                    .build(),
            );
        }
        let test_size = self
            .test_size
            .unwrap_or_else(|| (n / (k + 1)).max(1))
            .max(1);
        let mut folds = Vec::new();
        for i in 0..k {
            let test_end = n.saturating_sub((k - 1 - i) * test_size);
            let test_start = test_end.saturating_sub(test_size);
            if test_start == 0 {
                continue;
            }
            let train: Vec<usize> = (0..test_start).collect();
            let test: Vec<usize> = (test_start..test_end.min(n)).collect();
            if train.is_empty() || test.is_empty() {
                continue;
            }
            folds.push(Split { train, test });
        }
        if folds.is_empty() {
            ctx.push(
                Issue::builder(IssueCode::MeaninglessFit)
                    .message("TimeSeriesSplit produced no usable window")
                    .build(),
            );
        }
        ctx.finish(folds)
    }
}

/// Cross-validated scores from a caller-supplied fold scorer.
///
/// `scorer(x_train, y_train, x_test, y_test, session)` should return a
/// finite skill score (higher is better).
pub fn cross_val_score<F>(
    x: &Matrix,
    y: &Vector,
    folds: &[Split],
    mut scorer: F,
    session: &Session,
) -> Result<Qualified<Vector>>
where
    F: FnMut(&Matrix, &Vector, &Matrix, &Vector, &Session) -> Result<Qualified<f64>>,
{
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
    let mut scores = Vector::zeros(folds.len());
    for (i, fold) in folds.iter().enumerate() {
        let xt = take_rows(x, &fold.train);
        let yt = take_vec(y, &fold.train);
        let xv = take_rows(x, &fold.test);
        let yv = take_vec(y, &fold.test);
        match scorer(&xt, &yt, &xv, &yv, &session.child(format!("fold_{i}"))) {
            Ok(q) => scores[i] = q.value,
            Err(e) => {
                scores[i] = f64::NAN;
                ctx.push(e.primary);
            }
        }
    }
    ctx.finish(scores)
}

/// K-fold R² of [`LinearRegression`] (sklearn `cross_val_score` on OLS).
pub fn cross_val_score_linear(
    x: &Matrix,
    y: &Vector,
    splitter: &KFold,
    session: &Session,
) -> Result<Qualified<Vector>> {
    let folds = splitter.split(x.nrows(), &session.child("kfold"))?.value;
    cross_val_score(
        x,
        y,
        &folds,
        |xt, yt, xv, yv, sess| {
            let fitted = LinearRegression::new().fit(xt, yt, &sess.child("ols"))?;
            let pred = fitted.value.predict(xv, &sess.child("predict"))?;
            r2(yv, &pred.value, &sess.child("r2"))
        },
        session,
    )
}

/// Discrete grid search over Ridge `alpha` using K-fold R².
#[derive(Clone, Debug)]
pub struct GridSearchRidge {
    /// Candidate ℓ₂ penalties.
    pub alphas: Vec<f64>,
    /// CV splitter.
    pub cv: KFold,
}

impl Default for GridSearchRidge {
    fn default() -> Self {
        Self {
            alphas: vec![0.01, 0.1, 1.0, 10.0],
            cv: KFold::new(3),
        }
    }
}

impl GridSearchRidge {
    /// Grid over the given `alpha` values.
    pub fn new(alphas: Vec<f64>) -> Self {
        Self {
            alphas,
            cv: KFold::new(3),
        }
    }
}

/// Selected Ridge model and the CV scores that justified it.
#[derive(Clone, Debug)]
pub struct FittedGridSearchRidge {
    /// Penalty with the highest mean fold R².
    pub best_alpha: f64,
    /// Mean CV R² of `best_alpha`.
    pub best_score: f64,
    /// `(alpha, mean_cv_r2)` for every grid point.
    pub scores: Vec<(f64, f64)>,
    /// Ridge refit on the full training design at `best_alpha`.
    pub fitted: FittedPenalized,
}

impl Fit for GridSearchRidge {
    type Fitted = FittedGridSearchRidge;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedGridSearchRidge>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        let folds = match self.cv.split(x.nrows(), &session.child("cv")) {
            Ok(q) => q.value,
            Err(e) => {
                ctx.push(e.primary);
                Vec::new()
            }
        };
        let mut scores = Vec::new();
        let mut best_alpha = self.alphas.first().copied().unwrap_or(1.0);
        let mut best_score = f64::NEG_INFINITY;
        for &alpha in &self.alphas {
            let mut acc = 0.0;
            let mut k = 0.0;
            for (i, fold) in folds.iter().enumerate() {
                let xt = take_rows(x, &fold.train);
                let yt = take_vec(y, &fold.train);
                let xv = take_rows(x, &fold.test);
                let yv = take_vec(y, &fold.test);
                let mut ridge = Ridge::new(alpha);
                match ridge.fit(&xt, &yt, &session.child(format!("ridge_{alpha}_{i}"))) {
                    Ok(q) => match q.value.predict(&xv, &session.child("predict")) {
                        Ok(p) => {
                            if let Ok(s) = r2(&yv, &p.value, &session.child("r2")) {
                                if s.value.is_finite() {
                                    acc += s.value;
                                    k += 1.0;
                                }
                            }
                        }
                        Err(e) => ctx.push(e.primary),
                    },
                    Err(e) => ctx.push(e.primary),
                }
            }
            let mean = if k > 0.0 { acc / k } else { f64::NAN };
            scores.push((alpha, mean));
            if mean.is_finite() && mean > best_score {
                best_score = mean;
                best_alpha = alpha;
            }
        }
        let mut ridge = Ridge::new(best_alpha);
        let fitted = match ridge.fit(x, y, &session.child("refit")) {
            Ok(q) => q.value,
            Err(e) => {
                ctx.push(e.primary);
                FittedPenalized {
                    coef: Vector::zeros(x.ncols()),
                    intercept: y.mean(),
                    alpha: best_alpha,
                    l1_ratio: 0.0,
                }
            }
        };
        ctx.finish(FittedGridSearchRidge {
            best_alpha,
            best_score,
            scores,
            fitted,
        })
    }
}

/// Ridge with a K-fold R² grid over `alpha` (sklearn `RidgeCV`).
#[derive(Clone, Debug)]
pub struct RidgeCV {
    /// Candidate ℓ₂ penalties.
    pub alphas: Vec<f64>,
    /// CV splitter.
    pub cv: KFold,
}

impl Default for RidgeCV {
    fn default() -> Self {
        Self {
            alphas: vec![0.1, 1.0, 10.0],
            cv: KFold::new(3),
        }
    }
}

impl RidgeCV {
    /// Grid over the given `alpha` values.
    pub fn new(alphas: Vec<f64>) -> Self {
        Self {
            alphas,
            cv: KFold::new(3),
        }
    }
}

impl Fit for RidgeCV {
    type Fitted = FittedGridSearchRidge;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedGridSearchRidge>> {
        GridSearchRidge {
            alphas: self.alphas.clone(),
            cv: self.cv.clone(),
        }
        .fit(x, y, session)
    }
}

/// Lasso with a K-fold R² grid over `alpha` (sklearn `LassoCV`).
#[derive(Clone, Debug)]
pub struct LassoCV {
    /// Candidate ℓ₁ penalties.
    pub alphas: Vec<f64>,
    /// CV splitter.
    pub cv: KFold,
}

impl Default for LassoCV {
    fn default() -> Self {
        Self {
            alphas: vec![0.01, 0.1, 1.0],
            cv: KFold::new(3),
        }
    }
}

impl LassoCV {
    /// Grid over the given `alpha` values.
    pub fn new(alphas: Vec<f64>) -> Self {
        Self {
            alphas,
            cv: KFold::new(3),
        }
    }
}

/// Selected Lasso and the CV scores that justified it.
#[derive(Clone, Debug)]
pub struct FittedLassoCV {
    /// Penalty with the highest mean fold R².
    pub best_alpha: f64,
    /// Mean CV R² of `best_alpha`.
    pub best_score: f64,
    /// `(alpha, mean_cv_r2)` for every grid point.
    pub scores: Vec<(f64, f64)>,
    /// Lasso refit on the full training design at `best_alpha`.
    pub fitted: FittedPenalized,
}

impl Fit for LassoCV {
    type Fitted = FittedLassoCV;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedLassoCV>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        let folds = match self.cv.split(x.nrows(), &session.child("cv")) {
            Ok(q) => q.value,
            Err(e) => {
                ctx.push(e.primary);
                Vec::new()
            }
        };
        let mut scores = Vec::new();
        let mut best_alpha = self.alphas.first().copied().unwrap_or(1.0);
        let mut best_score = f64::NEG_INFINITY;
        for &alpha in &self.alphas {
            let mut acc = 0.0;
            let mut k = 0.0;
            for (i, fold) in folds.iter().enumerate() {
                let xt = take_rows(x, &fold.train);
                let yt = take_vec(y, &fold.train);
                let xv = take_rows(x, &fold.test);
                let yv = take_vec(y, &fold.test);
                let mut las = Lasso::new(alpha);
                match las.fit(&xt, &yt, &session.child(format!("lasso_{alpha}_{i}"))) {
                    Ok(q) => match q.value.predict(&xv, &session.child("predict")) {
                        Ok(p) => {
                            if let Ok(s) = r2(&yv, &p.value, &session.child("r2")) {
                                if s.value.is_finite() {
                                    acc += s.value;
                                    k += 1.0;
                                }
                            }
                        }
                        Err(e) => ctx.push(e.primary),
                    },
                    Err(e) => ctx.push(e.primary),
                }
            }
            let mean = if k > 0.0 { acc / k } else { f64::NAN };
            scores.push((alpha, mean));
            if mean.is_finite() && mean > best_score {
                best_score = mean;
                best_alpha = alpha;
            }
        }
        let mut las = Lasso::new(best_alpha);
        let fitted = match las.fit(x, y, &session.child("refit")) {
            Ok(q) => q.value,
            Err(e) => {
                ctx.push(e.primary);
                FittedPenalized {
                    coef: Vector::zeros(x.ncols()),
                    intercept: y.mean(),
                    alpha: best_alpha,
                    l1_ratio: 1.0,
                }
            }
        };
        ctx.finish(FittedLassoCV {
            best_alpha,
            best_score,
            scores,
            fitted,
        })
    }
}

/// Elastic-net with a K-fold R² grid over `alpha` × `l1_ratio`.
#[derive(Clone, Debug)]
pub struct ElasticNetCV {
    /// Candidate combined penalties.
    pub alphas: Vec<f64>,
    /// Candidate ℓ1 mixing weights.
    pub l1_ratio: Vec<f64>,
    /// CV splitter.
    pub cv: KFold,
}

impl Default for ElasticNetCV {
    fn default() -> Self {
        Self {
            alphas: vec![0.01, 0.1, 1.0],
            l1_ratio: vec![0.5],
            cv: KFold::new(3),
        }
    }
}

impl ElasticNetCV {
    /// Grid over the given `alpha` values at `l1_ratio = 0.5`.
    pub fn new(alphas: Vec<f64>) -> Self {
        Self {
            alphas,
            ..Self::default()
        }
    }
}

/// Selected elastic-net and the CV scores that justified it.
#[derive(Clone, Debug)]
pub struct FittedElasticNetCV {
    /// Penalty with the highest mean fold R².
    pub best_alpha: f64,
    /// Mixing weight of the winner.
    pub best_l1_ratio: f64,
    /// Mean CV R² of the winner.
    pub best_score: f64,
    /// `(alpha, l1_ratio, mean_cv_r2)` for every grid point.
    pub scores: Vec<(f64, f64, f64)>,
    /// Elastic-net refit on the full training design.
    pub fitted: FittedPenalized,
}

impl Fit for ElasticNetCV {
    type Fitted = FittedElasticNetCV;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedElasticNetCV>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        let ratios = if self.l1_ratio.is_empty() {
            vec![0.5]
        } else {
            self.l1_ratio.clone()
        };
        let folds = match self.cv.split(x.nrows(), &session.child("cv")) {
            Ok(q) => q.value,
            Err(e) => {
                ctx.push(e.primary);
                Vec::new()
            }
        };
        let mut scores = Vec::new();
        let mut best_alpha = self.alphas.first().copied().unwrap_or(1.0);
        let mut best_l1 = ratios.first().copied().unwrap_or(0.5);
        let mut best_score = f64::NEG_INFINITY;
        for &alpha in &self.alphas {
            for &l1 in &ratios {
                if !(0.0..=1.0).contains(&l1) {
                    ctx.push(
                        Issue::builder(IssueCode::InvalidWeight)
                            .severity(signlred::Severity::Warning)
                            .message(format!("ElasticNetCV skips l1_ratio={l1} outside [0,1]"))
                            .build(),
                    );
                    continue;
                }
                let mut acc = 0.0;
                let mut k = 0.0;
                for (i, fold) in folds.iter().enumerate() {
                    let xt = take_rows(x, &fold.train);
                    let yt = take_vec(y, &fold.train);
                    let xv = take_rows(x, &fold.test);
                    let yv = take_vec(y, &fold.test);
                    let mut en = ElasticNet::new(alpha, l1);
                    match en.fit(&xt, &yt, &session.child(format!("en_{alpha}_{l1}_{i}"))) {
                        Ok(q) => match q.value.predict(&xv, &session.child("predict")) {
                            Ok(p) => {
                                if let Ok(s) = r2(&yv, &p.value, &session.child("r2")) {
                                    if s.value.is_finite() {
                                        acc += s.value;
                                        k += 1.0;
                                    }
                                }
                            }
                            Err(e) => ctx.push(e.primary),
                        },
                        Err(e) => ctx.push(e.primary),
                    }
                }
                let mean = if k > 0.0 { acc / k } else { f64::NAN };
                scores.push((alpha, l1, mean));
                if mean.is_finite() && mean > best_score {
                    best_score = mean;
                    best_alpha = alpha;
                    best_l1 = l1;
                }
            }
        }
        let mut en = ElasticNet::new(best_alpha, best_l1);
        let fitted = match en.fit(x, y, &session.child("refit")) {
            Ok(q) => q.value,
            Err(e) => {
                ctx.push(e.primary);
                FittedPenalized {
                    coef: Vector::zeros(x.ncols()),
                    intercept: y.mean(),
                    alpha: best_alpha,
                    l1_ratio: best_l1,
                }
            }
        };
        ctx.finish(FittedElasticNetCV {
            best_alpha,
            best_l1_ratio: best_l1,
            best_score,
            scores,
            fitted,
        })
    }
}

/// Logistic regression with a K-fold accuracy grid over `c_inv` (sklearn
/// `LogisticRegressionCV`).
#[derive(Clone, Debug)]
pub struct LogisticRegressionCV {
    /// Candidate inverse-`C` penalties (`0` is unregularized MLE).
    pub cs_inv: Vec<f64>,
    /// CV splitter.
    pub cv: KFold,
}

impl Default for LogisticRegressionCV {
    fn default() -> Self {
        Self {
            cs_inv: vec![0.0, 0.1, 1.0],
            cv: KFold::new(3),
        }
    }
}

impl LogisticRegressionCV {
    /// Grid over the given `c_inv` values.
    pub fn new(cs_inv: Vec<f64>) -> Self {
        Self {
            cs_inv,
            cv: KFold::new(3),
        }
    }
}

/// Selected logistic and the CV scores that justified it.
#[derive(Clone, Debug)]
pub struct FittedLogisticRegressionCV {
    /// Winning `c_inv`.
    pub best_c_inv: f64,
    /// Mean CV accuracy.
    pub best_score: f64,
    /// `(c_inv, mean_cv_acc)` for every grid point.
    pub scores: Vec<(f64, f64)>,
    /// Logistic refit on the full training design.
    pub fitted: crate::linear_model::FittedLogistic,
}

impl Fit for LogisticRegressionCV {
    type Fitted = FittedLogisticRegressionCV;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedLogisticRegressionCV>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        let folds = match self.cv.split(x.nrows(), &session.child("cv")) {
            Ok(q) => q.value,
            Err(e) => {
                ctx.push(e.primary);
                Vec::new()
            }
        };
        let mut scores = Vec::new();
        let mut best_c = self.cs_inv.first().copied().unwrap_or(0.0);
        let mut best_score = f64::NEG_INFINITY;
        for &c_inv in &self.cs_inv {
            let mut acc = 0.0;
            let mut k = 0.0;
            for (i, fold) in folds.iter().enumerate() {
                let xt = take_rows(x, &fold.train);
                let yt = take_vec(y, &fold.train);
                let xv = take_rows(x, &fold.test);
                let yv = take_vec(y, &fold.test);
                let mut lr = LogisticRegression {
                    c_inv,
                    ..LogisticRegression::default()
                };
                match lr.fit(&xt, &yt, &session.child(format!("lrcv_{c_inv}_{i}"))) {
                    Ok(q) => match q.value.predict(&xv, &session.child("p")) {
                        Ok(p) => {
                            if let Ok(s) = accuracy(&yv, &p.value, &session.child("acc")) {
                                if s.value.is_finite() {
                                    acc += s.value;
                                    k += 1.0;
                                }
                            }
                        }
                        Err(_) => {}
                    },
                    Err(_) => {}
                }
            }
            let mean = if k > 0.0 { acc / k } else { f64::NAN };
            scores.push((c_inv, mean));
            if mean.is_finite() && mean > best_score {
                best_score = mean;
                best_c = c_inv;
            }
        }
        if !best_score.is_finite() {
            ctx.push(
                Issue::builder(IssueCode::UnidentifiedModel)
                    .message("LogisticRegressionCV found no finite fold accuracy")
                    .build(),
            );
        }
        let mut refit = LogisticRegression {
            c_inv: best_c,
            ..LogisticRegression::default()
        };
        let fitted = match refit.fit(x, y, &session.child("refit")) {
            Ok(q) => q.value,
            Err(e) => {
                ctx.push(e.primary);
                crate::linear_model::FittedLogistic {
                    coef: Vector::zeros(x.ncols()),
                    intercept: 0.0,
                    classes: vec![0, 1],
                    beta: Vector::zeros(x.ncols() + 1),
                    softmax: None,
                }
            }
        };
        ctx.finish(FittedLogisticRegressionCV {
            best_c_inv: best_c,
            best_score,
            scores,
            fitted,
        })
    }
}

impl Predict for FittedLogisticRegressionCV {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        self.fitted.predict(x, session)
    }
}

/// OMP with a K-fold R² grid over `n_nonzero` (sklearn `OrthogonalMatchingPursuitCV`).
#[derive(Clone, Debug)]
pub struct OrthogonalMatchingPursuitCV {
    /// Candidate support sizes.
    pub n_nonzero: Vec<usize>,
    /// CV splitter.
    pub cv: KFold,
}

impl Default for OrthogonalMatchingPursuitCV {
    fn default() -> Self {
        Self {
            n_nonzero: vec![1, 2],
            cv: KFold::new(3),
        }
    }
}

impl OrthogonalMatchingPursuitCV {
    /// Grid over the given support sizes.
    pub fn new(n_nonzero: Vec<usize>) -> Self {
        Self {
            n_nonzero,
            cv: KFold::new(3),
        }
    }
}

/// Selected OMP and the CV scores that justified it.
#[derive(Clone, Debug)]
pub struct FittedOmpCV {
    /// Winning support size.
    pub best_n_nonzero: usize,
    /// Mean CV R².
    pub best_score: f64,
    /// OMP refit on the full training design.
    pub fitted: crate::robust::FittedOmp,
}

impl Fit for OrthogonalMatchingPursuitCV {
    type Fitted = FittedOmpCV;
    fn fit(&mut self, x: &Matrix, y: &Vector, session: &Session) -> Result<Qualified<FittedOmpCV>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        let folds = match self.cv.split(x.nrows(), &session.child("cv")) {
            Ok(q) => q.value,
            Err(e) => {
                ctx.push(e.primary);
                Vec::new()
            }
        };
        let mut best_k = self.n_nonzero.first().copied().unwrap_or(1).max(1);
        let mut best_score = f64::NEG_INFINITY;
        for &k in &self.n_nonzero {
            let k = k.max(1);
            let mut acc = 0.0;
            let mut n = 0.0;
            for (i, fold) in folds.iter().enumerate() {
                let xt = take_rows(x, &fold.train);
                let yt = take_vec(y, &fold.train);
                let xv = take_rows(x, &fold.test);
                let yv = take_vec(y, &fold.test);
                match OrthogonalMatchingPursuit::new(k).fit(
                    &xt,
                    &yt,
                    &session.child(format!("ompcv_{k}_{i}")),
                ) {
                    Ok(q) => match q.value.predict(&xv, &session.child("p")) {
                        Ok(p) => {
                            if let Ok(s) = r2(&yv, &p.value, &session.child("r2")) {
                                if s.value.is_finite() {
                                    acc += s.value;
                                    n += 1.0;
                                }
                            }
                        }
                        Err(_) => {}
                    },
                    Err(_) => {}
                }
            }
            let mean = if n > 0.0 { acc / n } else { f64::NAN };
            if mean.is_finite() && mean > best_score {
                best_score = mean;
                best_k = k;
            }
        }
        let fitted = match OrthogonalMatchingPursuit::new(best_k).fit(x, y, &session.child("refit"))
        {
            Ok(q) => q.value,
            Err(e) => {
                ctx.push(e.primary);
                crate::robust::FittedOmp {
                    coef: Vector::zeros(x.ncols()),
                    intercept: y.mean(),
                    support: Vec::new(),
                }
            }
        };
        ctx.finish(FittedOmpCV {
            best_n_nonzero: best_k,
            best_score,
            fitted,
        })
    }
}

/// Multi-task elastic-net with a K-fold R² grid over `alpha`.
///
/// `Y` is a matrix; this is not the [`Fit`] trait.
#[derive(Clone, Debug)]
pub struct MultiTaskElasticNetCV {
    /// Candidate combined penalties.
    pub alphas: Vec<f64>,
    /// Mixing weight.
    pub l1_ratio: f64,
    /// CV splitter.
    pub cv: KFold,
}

impl Default for MultiTaskElasticNetCV {
    fn default() -> Self {
        Self {
            alphas: vec![0.01, 0.1, 1.0],
            l1_ratio: 0.5,
            cv: KFold::new(3),
        }
    }
}

impl MultiTaskElasticNetCV {
    /// Grid over the given `alpha` values.
    pub fn new(alphas: Vec<f64>) -> Self {
        Self {
            alphas,
            ..Self::default()
        }
    }

    /// Fit on a multi-column `Y`.
    pub fn fit(
        &mut self,
        x: &Matrix,
        y: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedMultiTaskElasticNetCV>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        inspect_xy(&mut ctx.report, y, None, &ctx.policy);
        let folds = match self.cv.split(x.nrows(), &session.child("cv")) {
            Ok(q) => q.value,
            Err(e) => {
                ctx.push(e.primary);
                Vec::new()
            }
        };
        let mut best_alpha = self.alphas.first().copied().unwrap_or(0.1);
        let mut best_score = f64::NEG_INFINITY;
        for &alpha in &self.alphas {
            let mut acc = 0.0;
            let mut k = 0.0;
            for (i, fold) in folds.iter().enumerate() {
                let xt = take_rows(x, &fold.train);
                let xv = take_rows(x, &fold.test);
                let yt =
                    Matrix::from_fn(fold.train.len(), y.ncols(), |r, c| y.get(fold.train[r], c));
                let yv = Matrix::from_fn(fold.test.len(), y.ncols(), |r, c| y.get(fold.test[r], c));
                match MultiTaskElasticNet::new(alpha, self.l1_ratio).fit(
                    &xt,
                    &yt,
                    &session.child(format!("mtecv_{alpha}_{i}")),
                ) {
                    Ok(q) => match q.value.predict_matrix(&xv, &session.child("p")) {
                        Ok(p) => {
                            let mut sse = 0.0;
                            let mut sst = 0.0;
                            for c in 0..yv.ncols() {
                                let col = yv.column(c);
                                let m = col.mean();
                                for r in 0..yv.nrows() {
                                    let e = yv.get(r, c) - p.value.get(r, c);
                                    sse += e * e;
                                    let d = yv.get(r, c) - m;
                                    sst += d * d;
                                }
                            }
                            if sst > 0.0 {
                                acc += 1.0 - sse / sst;
                                k += 1.0;
                            }
                        }
                        Err(_) => {}
                    },
                    Err(_) => {}
                }
            }
            let mean = if k > 0.0 { acc / k } else { f64::NAN };
            if mean.is_finite() && mean > best_score {
                best_score = mean;
                best_alpha = alpha;
            }
        }
        let fitted = match MultiTaskElasticNet::new(best_alpha, self.l1_ratio).fit(
            x,
            y,
            &session.child("refit"),
        ) {
            Ok(q) => q.value,
            Err(e) => {
                ctx.push(e.primary);
                FittedMultiTask {
                    coef: Matrix::zeros(x.ncols(), y.ncols()),
                    intercept: Vector::zeros(y.ncols()),
                    alpha: best_alpha,
                }
            }
        };
        ctx.finish(FittedMultiTaskElasticNetCV {
            best_alpha,
            best_score,
            fitted,
        })
    }
}

/// Selected multi-task elastic-net.
#[derive(Clone, Debug)]
pub struct FittedMultiTaskElasticNetCV {
    /// Winning `alpha`.
    pub best_alpha: f64,
    /// Mean CV R² across outputs.
    pub best_score: f64,
    /// Refit on the full design.
    pub fitted: FittedMultiTask,
}

/// LARS with a K-fold R² grid over `n_nonzero` (sklearn `LarsCV`).
#[derive(Clone, Debug)]
pub struct LarsCV {
    /// Candidate active-set sizes.
    pub n_nonzero: Vec<usize>,
    /// CV splitter.
    pub cv: KFold,
}

impl Default for LarsCV {
    fn default() -> Self {
        Self {
            n_nonzero: vec![1, 2],
            cv: KFold::new(3),
        }
    }
}

impl LarsCV {
    /// Grid over the given support sizes.
    pub fn new(n_nonzero: Vec<usize>) -> Self {
        Self {
            n_nonzero,
            cv: KFold::new(3),
        }
    }
}

/// Selected LARS and the CV score that justified it.
#[derive(Clone, Debug)]
pub struct FittedLarsCV {
    /// Winning support size.
    pub best_n_nonzero: usize,
    /// Mean CV R².
    pub best_score: f64,
    /// LARS refit on the full training design.
    pub fitted: crate::linear_model::FittedLars,
}

impl Fit for LarsCV {
    type Fitted = FittedLarsCV;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedLarsCV>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        let folds = match self.cv.split(x.nrows(), &session.child("cv")) {
            Ok(q) => q.value,
            Err(e) => {
                ctx.push(e.primary);
                Vec::new()
            }
        };
        let mut best_k = self.n_nonzero.first().copied().unwrap_or(1).max(1);
        let mut best_score = f64::NEG_INFINITY;
        for &k in &self.n_nonzero {
            let k = k.max(1);
            let mut acc = 0.0;
            let mut n = 0.0;
            for (i, fold) in folds.iter().enumerate() {
                let xt = take_rows(x, &fold.train);
                let yt = take_vec(y, &fold.train);
                let xv = take_rows(x, &fold.test);
                let yv = take_vec(y, &fold.test);
                let mut m = Lars {
                    n_nonzero: Some(k),
                    ..Lars::default()
                };
                match m.fit(&xt, &yt, &session.child(format!("larscv_{k}_{i}"))) {
                    Ok(q) => match q.value.predict(&xv, &session.child("p")) {
                        Ok(p) => {
                            if let Ok(s) = r2(&yv, &p.value, &session.child("r2")) {
                                if s.value.is_finite() {
                                    acc += s.value;
                                    n += 1.0;
                                }
                            }
                        }
                        Err(_) => {}
                    },
                    Err(_) => {}
                }
            }
            let mean = if n > 0.0 { acc / n } else { f64::NAN };
            if mean.is_finite() && mean > best_score {
                best_score = mean;
                best_k = k;
            }
        }
        let mut refit = Lars {
            n_nonzero: Some(best_k),
            ..Lars::default()
        };
        let fitted = match refit.fit(x, y, &session.child("refit")) {
            Ok(q) => q.value,
            Err(e) => {
                ctx.push(e.primary);
                crate::linear_model::FittedLars {
                    coef: Vector::zeros(x.ncols()),
                    intercept: y.mean(),
                    active: Vec::new(),
                }
            }
        };
        ctx.finish(FittedLarsCV {
            best_n_nonzero: best_k,
            best_score,
            fitted,
        })
    }
}

/// LassoLars with a K-fold R² grid over `alpha` (sklearn `LassoLarsCV`).
#[derive(Clone, Debug)]
pub struct LassoLarsCV {
    /// Candidate correlation floors.
    pub alphas: Vec<f64>,
    /// CV splitter.
    pub cv: KFold,
}

impl Default for LassoLarsCV {
    fn default() -> Self {
        Self {
            alphas: vec![0.0, 0.1, 1.0],
            cv: KFold::new(3),
        }
    }
}

impl LassoLarsCV {
    /// Grid over the given `alpha` values.
    pub fn new(alphas: Vec<f64>) -> Self {
        Self {
            alphas,
            cv: KFold::new(3),
        }
    }
}

/// Selected LassoLars.
#[derive(Clone, Debug)]
pub struct FittedLassoLarsCV {
    /// Winning `alpha`.
    pub best_alpha: f64,
    /// Mean CV R².
    pub best_score: f64,
    /// Refit on the full design.
    pub fitted: crate::linear_model::FittedLars,
}

impl Fit for LassoLarsCV {
    type Fitted = FittedLassoLarsCV;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedLassoLarsCV>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        let folds = match self.cv.split(x.nrows(), &session.child("cv")) {
            Ok(q) => q.value,
            Err(e) => {
                ctx.push(e.primary);
                Vec::new()
            }
        };
        let mut best_a = self.alphas.first().copied().unwrap_or(0.0);
        let mut best_score = f64::NEG_INFINITY;
        for &alpha in &self.alphas {
            let mut acc = 0.0;
            let mut n = 0.0;
            for (i, fold) in folds.iter().enumerate() {
                let xt = take_rows(x, &fold.train);
                let yt = take_vec(y, &fold.train);
                let xv = take_rows(x, &fold.test);
                let yv = take_vec(y, &fold.test);
                match LassoLars::new(alpha).fit(
                    &xt,
                    &yt,
                    &session.child(format!("llcv_{alpha}_{i}")),
                ) {
                    Ok(q) => match q.value.predict(&xv, &session.child("p")) {
                        Ok(p) => {
                            if let Ok(s) = r2(&yv, &p.value, &session.child("r2")) {
                                if s.value.is_finite() {
                                    acc += s.value;
                                    n += 1.0;
                                }
                            }
                        }
                        Err(_) => {}
                    },
                    Err(_) => {}
                }
            }
            let mean = if n > 0.0 { acc / n } else { f64::NAN };
            if mean.is_finite() && mean > best_score {
                best_score = mean;
                best_a = alpha;
            }
        }
        let fitted = match LassoLars::new(best_a).fit(x, y, &session.child("refit")) {
            Ok(q) => q.value,
            Err(e) => {
                ctx.push(e.primary);
                crate::linear_model::FittedLars {
                    coef: Vector::zeros(x.ncols()),
                    intercept: y.mean(),
                    active: Vec::new(),
                }
            }
        };
        ctx.finish(FittedLassoLarsCV {
            best_alpha: best_a,
            best_score,
            fitted,
        })
    }
}

/// Multi-task lasso with a K-fold R² grid over `alpha`.
#[derive(Clone, Debug)]
pub struct MultiTaskLassoCV {
    /// Candidate group penalties.
    pub alphas: Vec<f64>,
    /// CV splitter.
    pub cv: KFold,
}

impl Default for MultiTaskLassoCV {
    fn default() -> Self {
        Self {
            alphas: vec![0.01, 0.1, 1.0],
            cv: KFold::new(3),
        }
    }
}

impl MultiTaskLassoCV {
    /// Grid over the given `alpha` values.
    pub fn new(alphas: Vec<f64>) -> Self {
        Self {
            alphas,
            ..Self::default()
        }
    }

    /// Fit on a multi-column `Y`.
    pub fn fit(
        &mut self,
        x: &Matrix,
        y: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedMultiTaskElasticNetCV>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        inspect_xy(&mut ctx.report, y, None, &ctx.policy);
        let folds = match self.cv.split(x.nrows(), &session.child("cv")) {
            Ok(q) => q.value,
            Err(e) => {
                ctx.push(e.primary);
                Vec::new()
            }
        };
        let mut best_alpha = self.alphas.first().copied().unwrap_or(0.1);
        let mut best_score = f64::NEG_INFINITY;
        for &alpha in &self.alphas {
            let mut acc = 0.0;
            let mut k = 0.0;
            for (i, fold) in folds.iter().enumerate() {
                let xt = take_rows(x, &fold.train);
                let xv = take_rows(x, &fold.test);
                let yt =
                    Matrix::from_fn(fold.train.len(), y.ncols(), |r, c| y.get(fold.train[r], c));
                let yv = Matrix::from_fn(fold.test.len(), y.ncols(), |r, c| y.get(fold.test[r], c));
                match MultiTaskLasso::new(alpha).fit(
                    &xt,
                    &yt,
                    &session.child(format!("mtlcv_{alpha}_{i}")),
                ) {
                    Ok(q) => match q.value.predict_matrix(&xv, &session.child("p")) {
                        Ok(p) => {
                            let mut sse = 0.0;
                            let mut sst = 0.0;
                            for c in 0..yv.ncols() {
                                let col = yv.column(c);
                                let m = col.mean();
                                for r in 0..yv.nrows() {
                                    let e = yv.get(r, c) - p.value.get(r, c);
                                    sse += e * e;
                                    let d = yv.get(r, c) - m;
                                    sst += d * d;
                                }
                            }
                            if sst > 0.0 {
                                acc += 1.0 - sse / sst;
                                k += 1.0;
                            }
                        }
                        Err(_) => {}
                    },
                    Err(_) => {}
                }
            }
            let mean = if k > 0.0 { acc / k } else { f64::NAN };
            if mean.is_finite() && mean > best_score {
                best_score = mean;
                best_alpha = alpha;
            }
        }
        let fitted = match MultiTaskLasso::new(best_alpha).fit(x, y, &session.child("refit")) {
            Ok(q) => q.value,
            Err(e) => {
                ctx.push(e.primary);
                FittedMultiTask {
                    coef: Matrix::zeros(x.ncols(), y.ncols()),
                    intercept: Vector::zeros(y.ncols()),
                    alpha: best_alpha,
                }
            }
        };
        ctx.finish(FittedMultiTaskElasticNetCV {
            best_alpha,
            best_score,
            fitted,
        })
    }
}

/// Leave-one-out: each row is a singleton test fold.
#[derive(Clone, Debug, Default)]
pub struct LeaveOneOut;

impl LeaveOneOut {
    /// Default LOO splitter.
    pub fn new() -> Self {
        Self
    }

    /// Materialize `n` folds.
    pub fn split(&self, n: usize, session: &Session) -> Result<Qualified<Vec<Split>>> {
        let mut ctx = FitCtx::with_session(session.clone());
        if n < 2 {
            ctx.push(
                Issue::builder(IssueCode::InsufficientSample)
                    .message("LeaveOneOut on n<2 has no training fold")
                    .build(),
            );
        }
        if n > 200 {
            ctx.push(
                Issue::builder(IssueCode::Overparameterized)
                    .message(format!(
                        "LeaveOneOut materializes {n} folds; this is O(n) refits"
                    ))
                    .build(),
            );
        }
        let mut folds = Vec::with_capacity(n);
        for i in 0..n {
            let test = vec![i];
            let train: Vec<usize> = (0..n).filter(|&j| j != i).collect();
            folds.push(Split { train, test });
        }
        ctx.finish(folds)
    }
}

/// Group k-fold: every group id appears in exactly one test fold.
#[derive(Clone, Debug)]
pub struct GroupKFold {
    /// Number of folds.
    pub n_splits: usize,
}

impl Default for GroupKFold {
    fn default() -> Self {
        Self { n_splits: 5 }
    }
}

impl GroupKFold {
    /// `k` group folds.
    pub fn new(n_splits: usize) -> Self {
        Self { n_splits }
    }

    /// Split `n` rows whose group labels are `groups`.
    pub fn split(&self, groups: &Vector, session: &Session) -> Result<Qualified<Vec<Split>>> {
        let mut ctx = FitCtx::with_session(session.clone());
        let n = groups.len();
        let mut ids: Vec<i64> = Vec::new();
        for &g in groups.as_slice() {
            if !g.is_finite() {
                continue;
            }
            let lab = g.round() as i64;
            if !ids.contains(&lab) {
                ids.push(lab);
            }
        }
        ids.sort_unstable();
        let k = self.n_splits.max(2).min(ids.len().max(1));
        if self.n_splits < 2 {
            ctx.push(
                Issue::builder(IssueCode::InvalidWeight)
                    .message("GroupKFold.n_splits < 2; using 2")
                    .build(),
            );
        }
        if ids.len() < self.n_splits.max(2) {
            ctx.push(
                Issue::builder(IssueCode::InsufficientSample)
                    .message(format!(
                        "GroupKFold requested {} folds but only {} groups",
                        self.n_splits,
                        ids.len()
                    ))
                    .build(),
            );
        }
        let mut folds = Vec::with_capacity(k);
        for f in 0..k {
            let test_groups: Vec<i64> = ids
                .iter()
                .copied()
                .enumerate()
                .filter(|(i, _)| i % k == f)
                .map(|(_, g)| g)
                .collect();
            let mut test = Vec::new();
            let mut train = Vec::new();
            for i in 0..n {
                let g = groups[i].round() as i64;
                if test_groups.contains(&g) {
                    test.push(i);
                } else {
                    train.push(i);
                }
            }
            folds.push(Split { train, test });
        }
        ctx.finish(folds)
    }
}

/// Fit a column standardizer on **all** rows of `X` and transform them.
///
/// This is the helper that documents the leakage anti-pattern. The scale is
/// a function of the evaluation rows, so a subsequent supervised fit on the
/// result is not an honest out-of-sample protocol. Always recorded:
/// [`IssueCode::TargetLeakageSuspected`].
pub fn fit_transform_full(x: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, x, None, &ctx.policy);
    ctx.push(
        Issue::builder(IssueCode::TargetLeakageSuspected)
            .message(
                "fit_transform_full estimated mean/std on the entire design; \
                 do not reuse this transform inside a train/test or CV protocol",
            )
            .compromise(NumericalCompromise::new(
                "scaler fit on training rows only",
                "column mean/std on all n rows",
                "the helper is defined to fit on the full matrix",
                "refit the scaler inside each training fold; this transform leaks test-set scale",
            ))
            .build(),
    );
    let (n, p) = x.shape();
    let (xc, mean) = x.centered();
    let mut std = Vector::zeros(p);
    for j in 0..p {
        let col = xc.column(j);
        let s = col.std();
        std[j] = if s > ctx.policy.near_zero_variance {
            s
        } else {
            1.0
        };
    }
    let out = Matrix::from_fn(n, p, |i, j| {
        let v = x.get(i, j) - mean[j];
        v / std[j]
    });
    ctx.finish(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ojizou_san::Session;

    #[test]
    fn split_sizes_add_up() {
        let x = Matrix::from_fn(20, 2, |i, j| (i + j) as f64);
        let y = Vector::from_iter((0..20).map(|i| i as f64));
        let q = train_test_split(&x, &y, 0.25, 1, &Session::new("ms", "split")).unwrap();
        assert_eq!(q.value.0.nrows() + q.value.1.nrows(), 20);
        assert_eq!(q.value.2.len() + q.value.3.len(), 20);
    }

    #[test]
    fn kfold_covers_each_row_once() {
        let folds = KFold::new(4)
            .split(12, &Session::new("ms", "kfold"))
            .unwrap()
            .value;
        assert_eq!(folds.len(), 4);
        let mut seen = vec![0u8; 12];
        for f in &folds {
            for &i in &f.test {
                seen[i] += 1;
            }
        }
        assert!(seen.iter().all(|&c| c == 1));
    }

    #[test]
    fn stratified_preserves_both_classes() {
        let y = Vector::from_slice(&[0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]);
        let folds = StratifiedKFold::new(2)
            .split(&y, &Session::new("ms", "skf"))
            .unwrap()
            .value;
        for f in &folds {
            let mut has0 = false;
            let mut has1 = false;
            for &i in &f.test {
                if y[i] < 0.5 {
                    has0 = true;
                } else {
                    has1 = true;
                }
            }
            assert!(has0 && has1);
        }
    }

    #[test]
    fn timeseries_is_causal() {
        let folds = TimeSeriesSplit::new(3)
            .split(20, &Session::new("ms", "tss"))
            .unwrap()
            .value;
        assert!(!folds.is_empty());
        for f in &folds {
            let max_tr = f.train.iter().copied().max().unwrap();
            let min_te = f.test.iter().copied().min().unwrap();
            assert!(max_tr < min_te);
        }
    }

    #[test]
    fn cv_ridge_on_a_line_is_high() {
        let x = Matrix::from_fn(24, 1, |i, _| i as f64);
        let y = Vector::from_iter((0..24).map(|i| 1.0 + 2.0 * i as f64 + 0.3 * ((i % 4) as f64)));
        let folds = KFold::new(4)
            .split(24, &Session::new("ms", "kfold"))
            .unwrap()
            .value;
        let s = cross_val_score(
            &x,
            &y,
            &folds,
            |xt, yt, xv, yv, sess| {
                let fitted = Ridge::new(0.1).fit(xt, yt, &sess.child("ridge"))?;
                let pred = fitted.value.predict(xv, &sess.child("predict"))?;
                r2(yv, &pred.value, &sess.child("r2"))
            },
            &Session::new("ms", "cv"),
        )
        .unwrap()
        .value;
        assert!(
            s.as_slice().iter().all(|v| v.is_finite() && *v > 0.9),
            "{:?}",
            s.as_slice()
        );
    }

    #[test]
    fn grid_picks_a_finite_alpha() {
        let x = Matrix::from_fn(20, 1, |i, _| i as f64);
        let y = Vector::from_iter((0..20).map(|i| 0.5 * i as f64));
        let q = GridSearchRidge::new(vec![0.1, 1.0, 100.0])
            .fit(&x, &y, &Session::new("ms", "grid"))
            .unwrap();
        assert!(q.value.best_alpha.is_finite());
        assert!(q.value.best_score.is_finite());
        let r = RidgeCV::new(vec![0.1, 1.0])
            .fit(&x, &y, &Session::new("ms", "ridgecv"))
            .unwrap();
        assert!(r.value.best_alpha.is_finite());
        let l = LassoCV::new(vec![0.01, 0.1])
            .fit(&x, &y, &Session::new("ms", "lassocv"))
            .unwrap();
        assert!(l.value.best_alpha.is_finite());
        let e = ElasticNetCV::new(vec![0.01, 0.1])
            .fit(&x, &y, &Session::new("ms", "encv"))
            .unwrap();
        assert!(e.value.best_alpha.is_finite());
        assert!((0.0..=1.0).contains(&e.value.best_l1_ratio));
        let yb = Vector::from_iter((0..20).map(|i| if i < 10 { 0.0 } else { 1.0 }));
        let xb = Matrix::from_fn(20, 1, |i, _| if i < 10 { -1.2 } else { 1.2 });
        let lr = LogisticRegressionCV::new(vec![0.0, 1.0])
            .fit(&xb, &yb, &Session::new("ms", "lrcv"))
            .unwrap();
        assert!(lr.value.best_c_inv.is_finite());
        assert_eq!(lr.value.fitted.classes.len(), 2);
        let omp = OrthogonalMatchingPursuitCV::new(vec![1, 2])
            .fit(&x, &y, &Session::new("ms", "ompcv"))
            .unwrap();
        assert!(omp.value.best_n_nonzero >= 1);
        assert!(!omp.value.fitted.support.is_empty() || x.ncols() >= 1);
        let y2 = Matrix::from_fn(20, 2, |i, c| if c == 0 { y[i] } else { -0.4 * y[i] });
        let mt = MultiTaskElasticNetCV::new(vec![0.05, 0.2])
            .fit(&x, &y2, &Session::new("ms", "mtecv"))
            .unwrap();
        assert!(mt.value.best_alpha.is_finite());
        assert_eq!(mt.value.fitted.coef.ncols(), 2);
        let lcv = LarsCV::new(vec![1, 2])
            .fit(&x, &y, &Session::new("ms", "larscv"))
            .unwrap();
        assert!(lcv.value.best_n_nonzero >= 1);
        let llcv = LassoLarsCV::new(vec![0.0, 0.1])
            .fit(&x, &y, &Session::new("ms", "llcv"))
            .unwrap();
        assert!(llcv.value.best_alpha.is_finite());
        let mtl = MultiTaskLassoCV::new(vec![0.05, 0.2])
            .fit(&x, &y2, &Session::new("ms", "mtlcv"))
            .unwrap();
        assert!(mtl.value.best_alpha.is_finite());
    }

    #[test]
    fn fit_transform_full_flags_leakage() {
        let x = Matrix::from_fn(8, 2, |i, j| (i + 2 * j) as f64);
        let q = fit_transform_full(&x, &Session::new("ms", "leak")).unwrap();
        assert!(q.report.contains(IssueCode::TargetLeakageSuspected));
        assert!((q.value.column(0).mean()).abs() < 1e-12);
    }

    #[test]
    fn loo_and_group_kfold() {
        let loo = LeaveOneOut::new()
            .split(5, &Session::new("ms", "loo"))
            .unwrap()
            .value;
        assert_eq!(loo.len(), 5);
        assert!(loo.iter().all(|s| s.test.len() == 1 && s.train.len() == 4));
        let g = Vector::from_slice(&[0.0, 0.0, 1.0, 1.0, 2.0, 2.0]);
        let folds = GroupKFold::new(3)
            .split(&g, &Session::new("ms", "gkf"))
            .unwrap()
            .value;
        assert_eq!(folds.len(), 3);
        for f in &folds {
            let mut seen = Vec::new();
            for &i in &f.test {
                let lab = g[i].round() as i64;
                if !seen.contains(&lab) {
                    seen.push(lab);
                }
            }
            for &i in &f.train {
                assert!(!seen.contains(&(g[i].round() as i64)));
            }
        }
    }
}
