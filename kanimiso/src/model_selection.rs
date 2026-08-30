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
use signlred::{Issue, IssueCode, NumericalCompromise, Qualified, Result, Severity};

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

/// Repeated K-fold: `n_repeats` shuffled K-fold partitions.
#[derive(Clone, Debug)]
pub struct RepeatedKFold {
    /// Folds per repeat.
    pub n_splits: usize,
    /// Independent shuffles.
    pub n_repeats: usize,
    /// Base PRNG seed.
    pub seed: u64,
}

impl Default for RepeatedKFold {
    fn default() -> Self {
        Self {
            n_splits: 5,
            n_repeats: 2,
            seed: 0,
        }
    }
}

impl RepeatedKFold {
    /// `n_splits` folds, `n_repeats` times.
    pub fn new(n_splits: usize, n_repeats: usize) -> Self {
        Self {
            n_splits,
            n_repeats,
            seed: 0,
        }
    }

    /// Materialize all train/test index pairs.
    pub fn split(&self, n: usize, session: &Session) -> Result<Qualified<Vec<Split>>> {
        let mut ctx = FitCtx::with_session(session.clone());
        let k = self.n_splits.max(2);
        let r = self.n_repeats.max(1);
        if self.n_splits < 2 {
            ctx.push(
                Issue::builder(IssueCode::InvalidWeight)
                    .severity(Severity::Warning)
                    .message("RepeatedKFold.n_splits < 2; using 2")
                    .build(),
            );
        }
        if self.n_repeats < 1 {
            ctx.push(
                Issue::builder(IssueCode::InvalidWeight)
                    .severity(Severity::Warning)
                    .message("RepeatedKFold.n_repeats < 1; using 1")
                    .build(),
            );
        }
        let mut folds = Vec::new();
        for rep in 0..r {
            let inner = KFold {
                n_splits: k,
                shuffle: true,
                seed: self.seed.wrapping_add(rep as u64),
            };
            match inner.split(n, &session.child(format!("rep_{rep}"))) {
                Ok(q) => folds.extend(q.value),
                Err(e) => ctx.push(e.primary),
            }
        }
        ctx.finish(folds)
    }
}

/// Independent random train/test draws (sklearn `ShuffleSplit`).
#[derive(Clone, Debug)]
pub struct ShuffleSplit {
    /// Number of splits.
    pub n_splits: usize,
    /// Test fraction.
    pub test_size: f64,
    /// PRNG seed.
    pub seed: u64,
}

impl Default for ShuffleSplit {
    fn default() -> Self {
        Self {
            n_splits: 5,
            test_size: 0.25,
            seed: 0,
        }
    }
}

impl ShuffleSplit {
    /// `n_splits` random partitions.
    pub fn new(n_splits: usize) -> Self {
        Self {
            n_splits,
            ..Self::default()
        }
    }

    /// Materialize train/test index pairs for `n` rows.
    pub fn split(&self, n: usize, session: &Session) -> Result<Qualified<Vec<Split>>> {
        let mut ctx = FitCtx::with_session(session.clone());
        let k = self.n_splits.max(1);
        if self.n_splits < 1 {
            ctx.push(
                Issue::builder(IssueCode::InvalidWeight)
                    .severity(Severity::Warning)
                    .message("ShuffleSplit.n_splits < 1; using 1")
                    .build(),
            );
        }
        let frac = if self.test_size.is_finite() {
            self.test_size.clamp(0.05, 0.5)
        } else {
            0.25
        };
        if !self.test_size.is_finite() || self.test_size <= 0.0 || self.test_size >= 1.0 {
            ctx.push(
                Issue::builder(IssueCode::InvalidWeight)
                    .severity(Severity::Warning)
                    .message(format!(
                        "ShuffleSplit.test_size={} is not in (0,1); clamped",
                        self.test_size
                    ))
                    .build(),
            );
        }
        let mut n_test = (frac * n as f64).round() as usize;
        if n > 1 {
            n_test = n_test.clamp(1, n - 1);
        }
        let mut rng = Rng::new(self.seed);
        let mut folds = Vec::with_capacity(k);
        for _ in 0..k {
            let mut idx: Vec<usize> = (0..n).collect();
            rng.shuffle(&mut idx);
            folds.push(Split {
                test: idx[..n_test].to_vec(),
                train: idx[n_test..].to_vec(),
            });
        }
        ctx.finish(folds)
    }
}

/// Stratified random train/test draws (sklearn `StratifiedShuffleSplit`).
#[derive(Clone, Debug)]
pub struct StratifiedShuffleSplit {
    /// Number of splits.
    pub n_splits: usize,
    /// Test fraction.
    pub test_size: f64,
    /// PRNG seed.
    pub seed: u64,
}

impl Default for StratifiedShuffleSplit {
    fn default() -> Self {
        Self {
            n_splits: 5,
            test_size: 0.25,
            seed: 0,
        }
    }
}

impl StratifiedShuffleSplit {
    /// `n_splits` stratified random partitions.
    pub fn new(n_splits: usize) -> Self {
        Self {
            n_splits,
            ..Self::default()
        }
    }

    /// Materialize train/test index pairs that preserve class proportions of `y`.
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
        let k = self.n_splits.max(1);
        if self.n_splits < 1 {
            ctx.push(
                Issue::builder(IssueCode::InvalidWeight)
                    .severity(Severity::Warning)
                    .message("StratifiedShuffleSplit.n_splits < 1; using 1")
                    .build(),
            );
        }
        let frac = if self.test_size.is_finite() {
            self.test_size.clamp(0.05, 0.5)
        } else {
            0.25
        };
        if !self.test_size.is_finite() || self.test_size <= 0.0 || self.test_size >= 1.0 {
            ctx.push(
                Issue::builder(IssueCode::InvalidWeight)
                    .severity(Severity::Warning)
                    .message(format!(
                        "StratifiedShuffleSplit.test_size={} is not in (0,1); clamped",
                        self.test_size
                    ))
                    .build(),
            );
        }
        let mut rng = Rng::new(self.seed);
        let mut folds = Vec::with_capacity(k);
        for _ in 0..k {
            let mut test = Vec::new();
            let mut train = Vec::new();
            for (_, rows) in &mut by_class {
                rng.shuffle(rows);
                let mut n_te = (frac * rows.len() as f64).round() as usize;
                if rows.len() > 1 {
                    n_te = n_te.clamp(1, rows.len() - 1);
                }
                test.extend_from_slice(&rows[..n_te]);
                train.extend_from_slice(&rows[n_te..]);
            }
            folds.push(Split { train, test });
        }
        ctx.finish(folds)
    }
}

/// Repeated stratified K-fold.
#[derive(Clone, Debug)]
pub struct RepeatedStratifiedKFold {
    /// Folds per repeat.
    pub n_splits: usize,
    /// Independent shuffles.
    pub n_repeats: usize,
    /// Base PRNG seed.
    pub seed: u64,
}

impl Default for RepeatedStratifiedKFold {
    fn default() -> Self {
        Self {
            n_splits: 5,
            n_repeats: 2,
            seed: 0,
        }
    }
}

impl RepeatedStratifiedKFold {
    /// `n_splits` folds, `n_repeats` times.
    pub fn new(n_splits: usize, n_repeats: usize) -> Self {
        Self {
            n_splits,
            n_repeats,
            seed: 0,
        }
    }

    /// Materialize all stratified train/test index pairs.
    pub fn split(&self, y: &Vector, session: &Session) -> Result<Qualified<Vec<Split>>> {
        let mut ctx = FitCtx::with_session(session.clone());
        let k = self.n_splits.max(2);
        let r = self.n_repeats.max(1);
        if self.n_splits < 2 {
            ctx.push(
                Issue::builder(IssueCode::InvalidWeight)
                    .severity(Severity::Warning)
                    .message("RepeatedStratifiedKFold.n_splits < 2; using 2")
                    .build(),
            );
        }
        let mut folds = Vec::new();
        for rep in 0..r {
            let inner = StratifiedKFold {
                n_splits: k,
                shuffle: true,
                seed: self.seed.wrapping_add(rep as u64),
            };
            match inner.split(y, &session.child(format!("rskf_{rep}"))) {
                Ok(q) => folds.extend(q.value),
                Err(e) => ctx.push(e.primary),
            }
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

/// Out-of-fold Ridge predictions (sklearn `cross_val_predict`).
///
/// Each fold's predictions come from a model that did not see those rows.
/// Scoring against the full `y` after this map is still a leakage risk if
/// a later selector treats the OOF vector as a feature — that is recorded.
pub fn cross_val_predict(
    x: &Matrix,
    y: &Vector,
    splitter: &KFold,
    session: &Session,
) -> Result<Qualified<Vector>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
    let folds = match splitter.split(x.nrows(), &session.child("kfold")) {
        Ok(q) => q.value,
        Err(e) => {
            ctx.push(e.primary);
            Vec::new()
        }
    };
    let mut out = Vector::zeros(y.len());
    let mut seen = vec![false; y.len()];
    for (i, fold) in folds.iter().enumerate() {
        let xt = take_rows(x, &fold.train);
        let yt = take_vec(y, &fold.train);
        let xv = take_rows(x, &fold.test);
        match Ridge::new(0.1).fit(&xt, &yt, &session.child(format!("cvp_{i}"))) {
            Ok(q) => match q.value.predict(&xv, &session.child("p")) {
                Ok(p) => {
                    for (k, &row) in fold.test.iter().enumerate() {
                        if row < out.len() && k < p.value.len() {
                            out[row] = p.value[k];
                            seen[row] = true;
                        }
                    }
                }
                Err(e) => ctx.push(e.primary),
            },
            Err(e) => ctx.push(e.primary),
        }
    }
    if seen.iter().any(|s| !*s) {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .severity(Severity::Warning)
                .message("cross_val_predict left some rows without an OOF prediction")
                .build(),
        );
    }
    ctx.finish(out)
}

/// Train/test R² as a function of training-set size (sklearn `learning_curve`).
///
/// Each point is a Ridge(α=10⁻³) diagnostic, not a generic estimator protocol.
#[derive(Clone, Debug)]
pub struct LearningCurve {
    /// Absolute training sizes used.
    pub train_sizes: Vec<usize>,
    /// Mean in-sample R².
    pub train_scores: Vector,
    /// Mean held-out R².
    pub test_scores: Vector,
}

/// Learning curve of a lightly-penalized ridge over `train_sizes` fractions.
pub fn learning_curve(
    x: &Matrix,
    y: &Vector,
    train_sizes: &[f64],
    n_splits: usize,
    session: &Session,
) -> Result<Qualified<LearningCurve>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
    let k = n_splits.max(2);
    if n_splits < 2 {
        ctx.push(
            Issue::builder(IssueCode::InvalidWeight)
                .severity(Severity::Warning)
                .message("learning_curve n_splits < 2; using 2")
                .build(),
        );
    }
    let folds = match (KFold {
        n_splits: k,
        shuffle: true,
        seed: 3,
    })
    .split(x.nrows(), &session.child("lc_kfold"))
    {
        Ok(q) => q.value,
        Err(e) => {
            ctx.push(e.primary);
            Vec::new()
        }
    };
    let mut sizes = Vec::new();
    let mut tr_sc = Vec::new();
    let mut te_sc = Vec::new();
    for &frac in train_sizes {
        let f = if frac.is_finite() {
            frac.clamp(0.1, 1.0)
        } else {
            1.0
        };
        let mut tr_acc = 0.0;
        let mut te_acc = 0.0;
        let mut used = 0.0;
        let mut ntr_used = 0usize;
        for (i, fold) in folds.iter().enumerate() {
            let ntr = ((f * fold.train.len() as f64).round() as usize).clamp(2, fold.train.len());
            ntr_used = ntr;
            let idx = &fold.train[..ntr];
            let xt = take_rows(x, idx);
            let yt = take_vec(y, idx);
            let xv = take_rows(x, &fold.test);
            let yv = take_vec(y, &fold.test);
            match Ridge::new(1e-3).fit(&xt, &yt, &session.child(format!("lc_{i}"))) {
                Ok(q) => {
                    if let (Ok(ptr), Ok(pte)) = (
                        q.value.predict(&xt, &session.child("lctr")),
                        q.value.predict(&xv, &session.child("lcte")),
                    ) {
                        if let (Ok(rtr), Ok(rte)) = (
                            r2(&yt, &ptr.value, &session.child("r2tr")),
                            r2(&yv, &pte.value, &session.child("r2te")),
                        ) {
                            tr_acc += rtr.value;
                            te_acc += rte.value;
                            used += 1.0;
                        }
                    }
                }
                Err(e) => {
                    if !matches!(
                        e.primary.code,
                        IssueCode::ResidualTooLarge
                            | IssueCode::NearSingular
                            | IssueCode::RankZero
                            | IssueCode::R2IsOne
                    ) {
                        ctx.push(e.primary);
                    }
                }
            }
        }
        sizes.push(ntr_used);
        tr_sc.push(if used > 0.0 { tr_acc / used } else { f64::NAN });
        te_sc.push(if used > 0.0 { te_acc / used } else { f64::NAN });
    }
    ctx.finish(LearningCurve {
        train_sizes: sizes,
        train_scores: Vector::from_iter(tr_sc),
        test_scores: Vector::from_iter(te_sc),
    })
}

/// Ridge validation curve over a grid of `alpha` (sklearn `validation_curve`).
#[derive(Clone, Debug)]
pub struct ValidationCurve {
    /// Penalty values.
    pub param_values: Vector,
    /// Mean in-sample R².
    pub train_scores: Vector,
    /// Mean held-out R².
    pub test_scores: Vector,
}

/// Sklearn `validation_curve` for Ridge `alpha`.
pub fn validation_curve(
    x: &Matrix,
    y: &Vector,
    alphas: &[f64],
    n_splits: usize,
    session: &Session,
) -> Result<Qualified<ValidationCurve>> {
    validation_curve_ridge(x, y, alphas, n_splits, session)
}

/// Validation curve of Ridge over `alphas`.
pub fn validation_curve_ridge(
    x: &Matrix,
    y: &Vector,
    alphas: &[f64],
    n_splits: usize,
    session: &Session,
) -> Result<Qualified<ValidationCurve>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
    let k = n_splits.max(2);
    let folds = match (KFold {
        n_splits: k,
        shuffle: true,
        seed: 5,
    })
    .split(x.nrows(), &session.child("vc_kfold"))
    {
        Ok(q) => q.value,
        Err(e) => {
            ctx.push(e.primary);
            Vec::new()
        }
    };
    let mut tr = Vec::new();
    let mut te = Vec::new();
    for &a in alphas {
        let alpha = if a.is_finite() && a >= 0.0 { a } else { 0.1 };
        let mut tr_acc = 0.0;
        let mut te_acc = 0.0;
        let mut used = 0.0;
        for (i, fold) in folds.iter().enumerate() {
            let xt = take_rows(x, &fold.train);
            let yt = take_vec(y, &fold.train);
            let xv = take_rows(x, &fold.test);
            let yv = take_vec(y, &fold.test);
            match Ridge::new(alpha).fit(&xt, &yt, &session.child(format!("vc_{i}"))) {
                Ok(q) => {
                    if let (Ok(ptr), Ok(pte)) = (
                        q.value.predict(&xt, &session.child("vctr")),
                        q.value.predict(&xv, &session.child("vcte")),
                    ) {
                        if let (Ok(rtr), Ok(rte)) = (
                            r2(&yt, &ptr.value, &session.child("r2tr")),
                            r2(&yv, &pte.value, &session.child("r2te")),
                        ) {
                            tr_acc += rtr.value;
                            te_acc += rte.value;
                            used += 1.0;
                        }
                    }
                }
                Err(e) => {
                    if !matches!(
                        e.primary.code,
                        IssueCode::ResidualTooLarge
                            | IssueCode::NearSingular
                            | IssueCode::RankZero
                            | IssueCode::R2IsOne
                    ) {
                        ctx.push(e.primary);
                    }
                }
            }
        }
        tr.push(if used > 0.0 { tr_acc / used } else { f64::NAN });
        te.push(if used > 0.0 { te_acc / used } else { f64::NAN });
    }
    ctx.finish(ValidationCurve {
        param_values: Vector::from_iter(alphas.iter().copied()),
        train_scores: Vector::from_iter(tr),
        test_scores: Vector::from_iter(te),
    })
}

/// Column-wise permutation importance (mean and std of R² drop).
#[derive(Clone, Debug)]
pub struct PermutationImportance {
    /// Mean R² drop per column.
    pub importances_mean: Vector,
    /// Sample std of the drop over repeats.
    pub importances_std: Vector,
}

/// Permute each column of a full-sample Ridge fit and record the R² drop.
///
/// The baseline uses the full `y` ([`IssueCode::TargetLeakageSuspected`]).
pub fn permutation_importance(
    x: &Matrix,
    y: &Vector,
    n_repeats: usize,
    session: &Session,
) -> Result<Qualified<PermutationImportance>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
    ctx.push(
        Issue::builder(IssueCode::TargetLeakageSuspected)
            .severity(Severity::Advisory)
            .message("permutation_importance fits Ridge on the full (X, y)")
            .build(),
    );
    let reps = n_repeats.max(1);
    if n_repeats < 1 {
        ctx.push(
            Issue::builder(IssueCode::InvalidWeight)
                .severity(Severity::Warning)
                .message("permutation_importance n_repeats < 1; using 1")
                .build(),
        );
    }
    let fitted = match Ridge::new(1e-3).fit(x, y, &session.child("pi_fit")) {
        Ok(q) => q.value,
        Err(e) => {
            ctx.push(e.primary);
            return ctx.finish(PermutationImportance {
                importances_mean: Vector::zeros(x.ncols()),
                importances_std: Vector::zeros(x.ncols()),
            });
        }
    };
    let pred0 = match fitted.predict(x, &session.child("pi_p0")) {
        Ok(q) => q.value,
        Err(e) => {
            ctx.push(e.primary);
            return ctx.finish(PermutationImportance {
                importances_mean: Vector::zeros(x.ncols()),
                importances_std: Vector::zeros(x.ncols()),
            });
        }
    };
    let baseline = match r2(y, &pred0, &session.child("pi_r2")) {
        Ok(q) => q.value,
        Err(_) => f64::NAN,
    };
    let mut mean = Vector::zeros(x.ncols());
    let mut stdv = Vector::zeros(x.ncols());
    let mut rng = Rng::new(11);
    for j in 0..x.ncols() {
        let mut drops = Vec::new();
        for _ in 0..reps {
            let mut col: Vec<f64> = (0..x.nrows()).map(|i| x.get(i, j)).collect();
            rng.shuffle(&mut col);
            let xp = Matrix::from_fn(x.nrows(), x.ncols(), |r, c| {
                if c == j {
                    col[r]
                } else {
                    x.get(r, c)
                }
            });
            if let Ok(p) = fitted.predict(&xp, &session.child("pi_p")) {
                if let Ok(s) = r2(y, &p.value, &session.child("pi_rs")) {
                    if baseline.is_finite() && s.value.is_finite() {
                        drops.push(baseline - s.value);
                    }
                }
            }
        }
        if drops.is_empty() {
            continue;
        }
        let m = drops.iter().sum::<f64>() / drops.len() as f64;
        let var = if drops.len() > 1 {
            drops.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / (drops.len() as f64 - 1.0)
        } else {
            0.0
        };
        mean[j] = m;
        stdv[j] = var.max(0.0).sqrt();
    }
    ctx.finish(PermutationImportance {
        importances_mean: mean,
        importances_std: stdv,
    })
}

/// One-way partial dependence of a full-sample Ridge (sklearn `partial_dependence`).
#[derive(Clone, Debug)]
pub struct PartialDependence {
    /// Grid of the chosen column.
    pub grid: Vector,
    /// Mean prediction at each grid value.
    pub average: Vector,
}

/// Average Ridge prediction after pinning `feature` to each `grid` value.
pub fn partial_dependence(
    x: &Matrix,
    y: &Vector,
    feature: usize,
    grid: &Vector,
    session: &Session,
) -> Result<Qualified<PartialDependence>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
    ctx.push(
        Issue::builder(IssueCode::TargetLeakageSuspected)
            .severity(Severity::Advisory)
            .message("partial_dependence fits Ridge on the full (X, y)")
            .build(),
    );
    if feature >= x.ncols() {
        ctx.push(
            Issue::builder(IssueCode::DimensionMismatch)
                .message(format!(
                    "partial_dependence feature={feature} ≥ p={}",
                    x.ncols()
                ))
                .build(),
        );
        return ctx.finish(PartialDependence {
            grid: grid.clone(),
            average: Vector::zeros(grid.len()),
        });
    }
    let fitted = match Ridge::new(1e-3).fit(x, y, &session.child("pd_fit")) {
        Ok(q) => q.value,
        Err(e) => {
            ctx.push(e.primary);
            return ctx.finish(PartialDependence {
                grid: grid.clone(),
                average: Vector::zeros(grid.len()),
            });
        }
    };
    let mut avg = Vector::zeros(grid.len());
    for (t, &g) in grid.as_slice().iter().enumerate() {
        let xp = Matrix::from_fn(x.nrows(), x.ncols(), |i, j| {
            if j == feature {
                g
            } else {
                x.get(i, j)
            }
        });
        if let Ok(p) = fitted.predict(&xp, &session.child("pd_p")) {
            avg[t] = p.value.mean();
        }
    }
    ctx.finish(PartialDependence {
        grid: grid.clone(),
        average: avg,
    })
}

/// sklearn-style grid search over Ridge / Lasso `alpha` using K-fold R².
///
/// Lives here (not `linear_model`) to avoid a module cycle.
#[derive(Clone, Debug)]
pub struct GridSearchCV {
    /// Candidate penalties.
    pub alphas: Vec<f64>,
    /// `0` ⇒ Ridge, `1` ⇒ Lasso.
    pub l1_ratio: f64,
    /// CV splitter.
    pub cv: KFold,
}

impl Default for GridSearchCV {
    fn default() -> Self {
        Self {
            alphas: vec![0.01, 0.1, 1.0, 10.0],
            l1_ratio: 0.0,
            cv: KFold::new(3),
        }
    }
}

impl GridSearchCV {
    /// Grid over the given `alpha` values (Ridge unless `l1_ratio` is set).
    pub fn new(alphas: Vec<f64>) -> Self {
        Self {
            alphas,
            l1_ratio: 0.0,
            cv: KFold::new(3),
        }
    }
}

/// Selected linear model and the CV scores that justified it.
#[derive(Clone, Debug)]
pub struct FittedGridSearchCV {
    /// Penalty with the highest mean fold R².
    pub best_alpha: f64,
    /// Mean CV R² of `best_alpha`.
    pub best_score: f64,
    /// `(alpha, mean_cv_r2)` for every grid point.
    pub scores: Vec<(f64, f64)>,
    /// Refit on the full training design.
    pub fitted: FittedPenalized,
}

impl Fit for GridSearchCV {
    type Fitted = FittedGridSearchCV;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedGridSearchCV>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        let folds = match self.cv.split(x.nrows(), &session.child("cv")) {
            Ok(q) => q.value,
            Err(e) => {
                ctx.push(e.primary);
                Vec::new()
            }
        };
        let l1 = self.l1_ratio.clamp(0.0, 1.0);
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
                let fitted = if l1 >= 0.5 {
                    Lasso::new(alpha).fit(&xt, &yt, &session.child(format!("gs_{alpha}_{i}")))
                } else {
                    Ridge::new(alpha).fit(&xt, &yt, &session.child(format!("gs_{alpha}_{i}")))
                };
                match fitted {
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
        let fitted = if l1 >= 0.5 {
            Lasso::new(best_alpha).fit(x, y, &session.child("refit"))
        } else {
            Ridge::new(best_alpha).fit(x, y, &session.child("refit"))
        };
        let fitted = match fitted {
            Ok(q) => q.value,
            Err(e) => {
                ctx.push(e.primary);
                FittedPenalized {
                    coef: Vector::zeros(x.ncols()),
                    intercept: y.mean(),
                    alpha: best_alpha,
                    l1_ratio: l1,
                }
            }
        };
        ctx.finish(FittedGridSearchCV {
            best_alpha,
            best_score,
            scores,
            fitted,
        })
    }
}

/// Randomized search over a log-uniform Ridge `alpha` range.
#[derive(Clone, Debug)]
pub struct RandomizedSearchCV {
    /// Number of random `alpha` draws.
    pub n_iter: usize,
    /// Inclusive log10 lower bound.
    pub alpha_log_low: f64,
    /// Inclusive log10 upper bound.
    pub alpha_log_high: f64,
    /// PRNG seed.
    pub seed: u64,
    /// CV splitter.
    pub cv: KFold,
}

impl Default for RandomizedSearchCV {
    fn default() -> Self {
        Self {
            n_iter: 6,
            alpha_log_low: -2.0,
            alpha_log_high: 1.0,
            seed: 3,
            cv: KFold::new(3),
        }
    }
}

impl RandomizedSearchCV {
    /// `n_iter` log-uniform Ridge penalties.
    pub fn new(n_iter: usize) -> Self {
        Self {
            n_iter: n_iter.max(1),
            ..Self::default()
        }
    }
}

impl Fit for RandomizedSearchCV {
    type Fitted = FittedGridSearchCV;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedGridSearchCV>> {
        let mut ctx = FitCtx::with_session(session.clone());
        if self.n_iter < 1 {
            ctx.push(
                Issue::builder(IssueCode::InvalidWeight)
                    .severity(Severity::Warning)
                    .message("RandomizedSearchCV.n_iter < 1; using 1")
                    .build(),
            );
        }
        let mut rng = Rng::new(self.seed);
        let lo = self.alpha_log_low.min(self.alpha_log_high);
        let hi = self.alpha_log_low.max(self.alpha_log_high);
        let n_iter = self.n_iter.max(1);
        let alphas: Vec<f64> = (0..n_iter)
            .map(|_| 10.0_f64.powf(lo + (hi - lo) * rng.uniform()))
            .collect();
        let mut inner = GridSearchCV {
            alphas,
            l1_ratio: 0.0,
            cv: self.cv.clone(),
        };
        match inner.fit(x, y, session) {
            Ok(q) => {
                for issue in q.report.issues() {
                    ctx.push(issue.clone());
                }
                ctx.finish(q.value)
            }
            Err(e) => {
                ctx.push(e.primary);
                ctx.finish(FittedGridSearchCV {
                    best_alpha: 1.0,
                    best_score: f64::NAN,
                    scores: Vec::new(),
                    fitted: FittedPenalized {
                        coef: Vector::zeros(x.ncols()),
                        intercept: y.mean(),
                        alpha: 1.0,
                        l1_ratio: 0.0,
                    },
                })
            }
        }
    }
}

/// Successive-halving grid search over Ridge `alpha`.
///
/// Resource is a growing training prefix. Candidate count is not identification
/// `p`. Inner Ridge residual issues are not promoted.
#[derive(Clone, Debug)]
pub struct HalvingGridSearchCV {
    /// Candidate penalties.
    pub alphas: Vec<f64>,
    /// Reduction factor \(\eta \ge 2\).
    pub factor: usize,
    /// Smallest training prefix.
    pub min_resources: usize,
}

impl Default for HalvingGridSearchCV {
    fn default() -> Self {
        Self {
            alphas: vec![0.01, 0.1, 1.0, 10.0],
            factor: 2,
            min_resources: 8,
        }
    }
}

impl HalvingGridSearchCV {
    /// Halving search over the given `alpha` values.
    pub fn new(alphas: Vec<f64>) -> Self {
        Self {
            alphas,
            ..Self::default()
        }
    }
}

fn score_ridge_prefix(
    x: &Matrix,
    y: &Vector,
    alpha: f64,
    n_train: usize,
    session: &Session,
) -> Option<f64> {
    let n = x.nrows().min(y.len());
    let n_tr = n_train.max(2).min(n.saturating_sub(1).max(2));
    if n_tr >= n {
        return None;
    }
    let xt = take_rows(x, &(0..n_tr).collect::<Vec<_>>());
    let yt = take_vec(y, &(0..n_tr).collect::<Vec<_>>());
    let xv = take_rows(x, &(n_tr..n).collect::<Vec<_>>());
    let yv = take_vec(y, &(n_tr..n).collect::<Vec<_>>());
    let fitted = Ridge::new(alpha).fit(&xt, &yt, &session.child("halving_fit"));
    match fitted {
        Ok(q) => match q.value.predict(&xv, &session.child("halving_p")) {
            Ok(p) => r2(&yv, &p.value, &session.child("halving_r2"))
                .ok()
                .map(|s| s.value)
                .filter(|v| v.is_finite()),
            Err(_) => None,
        },
        Err(_) => None,
    }
}

impl Fit for HalvingGridSearchCV {
    type Fitted = FittedGridSearchCV;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedGridSearchCV>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        let eta = self.factor.max(2);
        if self.factor < 2 {
            ctx.push(
                Issue::builder(IssueCode::InvalidWeight)
                    .severity(Severity::Warning)
                    .message(format!(
                        "HalvingGridSearchCV.factor={} < 2; using 2",
                        self.factor
                    ))
                    .build(),
            );
        }
        let n = x.nrows().min(y.len());
        let mut resource = self.min_resources.max(4).min(n.saturating_sub(1).max(4));
        let mut candidates = self.alphas.clone();
        if candidates.is_empty() {
            candidates.push(1.0);
            ctx.push(
                Issue::builder(IssueCode::InvalidWeight)
                    .severity(Severity::Warning)
                    .message("HalvingGridSearchCV had an empty grid; using α=1")
                    .build(),
            );
        }
        let mut scores = Vec::new();
        let mut best_alpha = candidates[0];
        let mut best_score = f64::NEG_INFINITY;
        while candidates.len() > 1 && resource < n {
            let mut ranked: Vec<(f64, f64)> = Vec::new();
            for &alpha in &candidates {
                let mean = score_ridge_prefix(x, y, alpha, resource, session).unwrap_or(f64::NAN);
                ranked.push((alpha, mean));
                if mean.is_finite() && mean > best_score {
                    best_score = mean;
                    best_alpha = alpha;
                }
            }
            ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let keep = (candidates.len() / eta).max(1);
            candidates = ranked.iter().take(keep).map(|(a, _)| *a).collect();
            scores = ranked;
            let next = resource.saturating_mul(eta);
            if next <= resource {
                break;
            }
            resource = next.min(n.saturating_sub(1).max(resource));
        }
        if let Some(&alpha) = candidates.first() {
            if let Some(s) = score_ridge_prefix(x, y, alpha, n.saturating_sub(1).max(2), session) {
                if s > best_score {
                    best_score = s;
                    best_alpha = alpha;
                }
            } else {
                best_alpha = alpha;
            }
        }
        let fitted = match Ridge::new(best_alpha).fit(x, y, &session.child("halving_refit")) {
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
        ctx.finish(FittedGridSearchCV {
            best_alpha,
            best_score,
            scores,
            fitted,
        })
    }
}

/// Successive-halving randomized search over a log-uniform Ridge `alpha`.
#[derive(Clone, Debug)]
pub struct HalvingRandomSearchCV {
    /// Number of random `alpha` draws at the first rung.
    pub n_candidates: usize,
    /// Inclusive log10 lower bound.
    pub alpha_log_low: f64,
    /// Inclusive log10 upper bound.
    pub alpha_log_high: f64,
    /// Reduction factor.
    pub factor: usize,
    /// PRNG seed.
    pub seed: u64,
}

impl Default for HalvingRandomSearchCV {
    fn default() -> Self {
        Self {
            n_candidates: 8,
            alpha_log_low: -2.0,
            alpha_log_high: 1.0,
            factor: 2,
            seed: 5,
        }
    }
}

impl HalvingRandomSearchCV {
    /// `n_candidates` log-uniform Ridge penalties.
    pub fn new(n_candidates: usize) -> Self {
        Self {
            n_candidates: n_candidates.max(2),
            ..Self::default()
        }
    }
}

impl Fit for HalvingRandomSearchCV {
    type Fitted = FittedGridSearchCV;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedGridSearchCV>> {
        let mut ctx = FitCtx::with_session(session.clone());
        if self.n_candidates < 2 {
            ctx.push(
                Issue::builder(IssueCode::InvalidWeight)
                    .severity(Severity::Warning)
                    .message("HalvingRandomSearchCV.n_candidates < 2; using 2")
                    .build(),
            );
        }
        let mut rng = Rng::new(self.seed);
        let lo = self.alpha_log_low.min(self.alpha_log_high);
        let hi = self.alpha_log_low.max(self.alpha_log_high);
        let n_c = self.n_candidates.max(2);
        let alphas: Vec<f64> = (0..n_c)
            .map(|_| 10.0_f64.powf(lo + (hi - lo) * rng.uniform()))
            .collect();
        let mut inner = HalvingGridSearchCV {
            alphas,
            factor: self.factor.max(2),
            min_resources: 8,
        };
        match inner.fit(x, y, session) {
            Ok(q) => {
                for issue in q.report.issues() {
                    ctx.push(issue.clone());
                }
                ctx.finish(q.value)
            }
            Err(e) => {
                ctx.push(e.primary);
                ctx.finish(FittedGridSearchCV {
                    best_alpha: 1.0,
                    best_score: f64::NAN,
                    scores: Vec::new(),
                    fitted: FittedPenalized {
                        coef: Vector::zeros(x.ncols()),
                        intercept: y.mean(),
                        alpha: 1.0,
                        l1_ratio: 0.0,
                    },
                })
            }
        }
    }
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

/// Random train/test draws that keep groups intact (sklearn `GroupShuffleSplit`).
#[derive(Clone, Debug)]
pub struct GroupShuffleSplit {
    /// Number of splits.
    pub n_splits: usize,
    /// Test fraction of **groups**.
    pub test_size: f64,
    /// PRNG seed.
    pub seed: u64,
}

impl Default for GroupShuffleSplit {
    fn default() -> Self {
        Self {
            n_splits: 5,
            test_size: 0.25,
            seed: 0,
        }
    }
}

impl GroupShuffleSplit {
    /// `n_splits` group-wise random partitions.
    pub fn new(n_splits: usize) -> Self {
        Self {
            n_splits,
            ..Self::default()
        }
    }

    /// Split rows whose group labels are `groups`.
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
        if ids.len() < 2 {
            ctx.push(
                Issue::builder(IssueCode::InsufficientSample)
                    .severity(Severity::Warning)
                    .message("GroupShuffleSplit needs at least two groups")
                    .build(),
            );
        }
        let k = self.n_splits.max(1);
        let frac = if self.test_size.is_finite() {
            self.test_size.clamp(0.05, 0.5)
        } else {
            0.25
        };
        if !self.test_size.is_finite() || self.test_size <= 0.0 || self.test_size >= 1.0 {
            ctx.push(
                Issue::builder(IssueCode::InvalidWeight)
                    .severity(Severity::Warning)
                    .message(format!(
                        "GroupShuffleSplit.test_size={} is not in (0,1); clamped",
                        self.test_size
                    ))
                    .build(),
            );
        }
        let mut n_te_g = (frac * ids.len() as f64).round() as usize;
        if ids.len() > 1 {
            n_te_g = n_te_g.clamp(1, ids.len() - 1);
        }
        let mut rng = Rng::new(self.seed);
        let mut folds = Vec::with_capacity(k);
        for _ in 0..k {
            rng.shuffle(&mut ids);
            let test_g = ids[..n_te_g.min(ids.len())].to_vec();
            let mut test = Vec::new();
            let mut train = Vec::new();
            for i in 0..n {
                let g = groups[i].round() as i64;
                if test_g.contains(&g) {
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

/// Sliding window of fixed length (sktime `SlidingWindowSplitter`).
///
/// Train is `[t, t+window)`, test is `[t+window, t+window+fh)`. Window length
/// is not identification `p`.
#[derive(Clone, Debug)]
pub struct SlidingWindowSplitter {
    /// Training window length.
    pub window_length: usize,
    /// Forecast horizon (test length).
    pub fh: usize,
    /// Step between consecutive windows.
    pub step: usize,
}

impl Default for SlidingWindowSplitter {
    fn default() -> Self {
        Self {
            window_length: 8,
            fh: 1,
            step: 1,
        }
    }
}

impl SlidingWindowSplitter {
    /// Window `window_length`, horizon `fh`.
    pub fn new(window_length: usize, fh: usize) -> Self {
        Self {
            window_length,
            fh,
            step: 1,
        }
    }

    /// Materialize causal sliding windows for `n` rows.
    pub fn split(&self, n: usize, session: &Session) -> Result<Qualified<Vec<Split>>> {
        let mut ctx = FitCtx::with_session(session.clone());
        let w = self.window_length.max(1);
        let h = self.fh.max(1);
        let step = self.step.max(1);
        if self.window_length < 2 {
            ctx.push(
                Issue::builder(IssueCode::WindowTooShort)
                    .message(format!(
                        "SlidingWindowSplitter.window_length={} < 2",
                        self.window_length
                    ))
                    .build(),
            );
        }
        let mut folds = Vec::new();
        let mut start = 0usize;
        while start + w + h <= n {
            folds.push(Split {
                train: (start..start + w).collect(),
                test: (start + w..start + w + h).collect(),
            });
            start += step;
        }
        if folds.is_empty() {
            ctx.push(
                Issue::builder(IssueCode::InsufficientSample)
                    .severity(Severity::Warning)
                    .message(format!(
                        "SlidingWindowSplitter produced no folds for n={n} window={w} fh={h}"
                    ))
                    .build(),
            );
        }
        ctx.finish(folds)
    }
}

/// Expanding training window (sktime `ExpandingWindowSplitter`).
///
/// Train is `[0, t)`, test is `[t, t+fh)`. Initial window length is not
/// identification `p`.
#[derive(Clone, Debug)]
pub struct ExpandingWindowSplitter {
    /// First training length.
    pub initial_window: usize,
    /// Forecast horizon (test length).
    pub fh: usize,
    /// Step between consecutive origins.
    pub step: usize,
}

impl Default for ExpandingWindowSplitter {
    fn default() -> Self {
        Self {
            initial_window: 8,
            fh: 1,
            step: 1,
        }
    }
}

impl ExpandingWindowSplitter {
    /// Expanding window starting at `initial_window` with horizon `fh`.
    pub fn new(initial_window: usize, fh: usize) -> Self {
        Self {
            initial_window,
            fh,
            step: 1,
        }
    }

    /// Materialize causal expanding windows for `n` rows.
    pub fn split(&self, n: usize, session: &Session) -> Result<Qualified<Vec<Split>>> {
        let mut ctx = FitCtx::with_session(session.clone());
        let w0 = self.initial_window.max(2);
        let h = self.fh.max(1);
        let step = self.step.max(1);
        if self.initial_window < 2 {
            ctx.push(
                Issue::builder(IssueCode::WindowTooShort)
                    .message(format!(
                        "ExpandingWindowSplitter.initial_window={} < 2",
                        self.initial_window
                    ))
                    .build(),
            );
        }
        let mut folds = Vec::new();
        let mut origin = w0;
        while origin + h <= n {
            folds.push(Split {
                train: (0..origin).collect(),
                test: (origin..origin + h).collect(),
            });
            origin += step;
        }
        if folds.is_empty() {
            ctx.push(
                Issue::builder(IssueCode::InsufficientSample)
                    .severity(Severity::Warning)
                    .message(format!(
                        "ExpandingWindowSplitter produced no folds for n={n} initial={w0} fh={h}"
                    ))
                    .build(),
            );
        }
        ctx.finish(folds)
    }
}

fn combinations_limited(n: usize, p: usize, limit: usize) -> Vec<Vec<usize>> {
    let mut out = Vec::new();
    if p == 0 || p > n || limit == 0 {
        return out;
    }
    let mut cur = Vec::with_capacity(p);
    fn rec(
        start: usize,
        n: usize,
        p: usize,
        cur: &mut Vec<usize>,
        out: &mut Vec<Vec<usize>>,
        limit: usize,
    ) {
        if out.len() >= limit {
            return;
        }
        if cur.len() == p {
            out.push(cur.clone());
            return;
        }
        let need = p - cur.len();
        if start + need > n {
            return;
        }
        for i in start..=n - need {
            cur.push(i);
            rec(i + 1, n, p, cur, out, limit);
            cur.pop();
            if out.len() >= limit {
                return;
            }
        }
    }
    rec(0, n, p, &mut cur, &mut out, limit);
    out
}

/// Leave-`p`-out: every `p`-subset is a test fold (sklearn `LeavePOut`).
///
/// Combination count is not identification `p`. More than 128 folds are
/// truncated and recorded.
#[derive(Clone, Debug)]
pub struct LeavePOut {
    /// Test-set cardinality.
    pub p: usize,
}

impl Default for LeavePOut {
    fn default() -> Self {
        Self { p: 2 }
    }
}

impl LeavePOut {
    /// Leave-`p`-out splitter.
    pub fn new(p: usize) -> Self {
        Self { p }
    }

    /// Materialize leave-`p`-out folds for `n` rows.
    pub fn split(&self, n: usize, session: &Session) -> Result<Qualified<Vec<Split>>> {
        let mut ctx = FitCtx::with_session(session.clone());
        let p = self.p.max(1);
        if self.p == 0 || self.p >= n {
            ctx.push(
                Issue::builder(IssueCode::InvalidWeight)
                    .severity(Severity::Warning)
                    .message(format!(
                        "LeavePOut.p={} is not in 1..n={n}; using {}",
                        self.p,
                        p.min(n.saturating_sub(1).max(1))
                    ))
                    .build(),
            );
        }
        let p = p.min(n.saturating_sub(1).max(1));
        const LIMIT: usize = 128;
        let combos = combinations_limited(n, p, LIMIT + 1);
        if combos.len() > LIMIT {
            ctx.push(
                Issue::builder(IssueCode::Overparameterized)
                    .message(format!(
                        "LeavePOut({p}) on n={n} exceeds {LIMIT} folds; the iterator is truncated"
                    ))
                    .build(),
            );
        }
        let folds: Vec<Split> = combos
            .into_iter()
            .take(LIMIT)
            .map(|test| {
                let train: Vec<usize> = (0..n).filter(|i| !test.contains(i)).collect();
                Split { train, test }
            })
            .collect();
        if folds.is_empty() {
            ctx.push(
                Issue::builder(IssueCode::InsufficientSample)
                    .severity(Severity::Warning)
                    .message(format!("LeavePOut produced no folds for n={n} p={p}"))
                    .build(),
            );
        }
        ctx.finish(folds)
    }
}

/// User-supplied fold ids (sklearn `PredefinedSplit`).
///
/// A value of `-1` keeps the row in every training set. Other integers are
/// test-fold labels. Fold-id cardinality is not identification `p`.
#[derive(Clone, Debug)]
pub struct PredefinedSplit {
    /// Per-row test-fold id (`-1` = always train).
    pub test_fold: Vector,
}

impl PredefinedSplit {
    /// Splitter from a fold-id vector.
    pub fn new(test_fold: Vector) -> Self {
        Self { test_fold }
    }

    /// Materialize one fold per distinct non-negative id.
    pub fn split(&self, session: &Session) -> Result<Qualified<Vec<Split>>> {
        let mut ctx = FitCtx::with_session(session.clone());
        let n = self.test_fold.len();
        let mut ids: Vec<i64> = Vec::new();
        for &v in self.test_fold.as_slice() {
            if !v.is_finite() {
                continue;
            }
            let lab = v.round() as i64;
            if lab >= 0 && !ids.contains(&lab) {
                ids.push(lab);
            }
        }
        ids.sort_unstable();
        if ids.is_empty() {
            ctx.push(
                Issue::builder(IssueCode::InsufficientSample)
                    .severity(Severity::Warning)
                    .message("PredefinedSplit has no non-negative fold ids")
                    .build(),
            );
        }
        let mut folds = Vec::with_capacity(ids.len());
        for &lab in &ids {
            let mut test = Vec::new();
            let mut train = Vec::new();
            for i in 0..n {
                let v = self.test_fold[i];
                if !v.is_finite() {
                    continue;
                }
                let id = v.round() as i64;
                if id == lab {
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

/// Single causal hold-out (sktime `TemporalTrainTestSplitter`).
///
/// `test_size` in `(0, 1)` is a fraction; otherwise it is a row count.
#[derive(Clone, Debug)]
pub struct TemporalTrainTestSplitter {
    /// Test length or fraction.
    pub test_size: f64,
}

impl Default for TemporalTrainTestSplitter {
    fn default() -> Self {
        Self { test_size: 0.25 }
    }
}

impl TemporalTrainTestSplitter {
    /// Causal splitter with the given test size.
    pub fn new(test_size: f64) -> Self {
        Self { test_size }
    }

    /// One expanding-origin hold-out on `n` rows.
    pub fn split(&self, n: usize, session: &Session) -> Result<Qualified<Vec<Split>>> {
        let mut ctx = FitCtx::with_session(session.clone());
        if n < 2 {
            ctx.push(
                Issue::builder(IssueCode::InsufficientSample)
                    .severity(Severity::Warning)
                    .message(format!("TemporalTrainTestSplitter on n={n} < 2"))
                    .build(),
            );
            return ctx.finish(Vec::new());
        }
        let mut h = if self.test_size > 0.0 && self.test_size < 1.0 {
            (n as f64 * self.test_size).round() as usize
        } else {
            self.test_size.round() as usize
        };
        if self.test_size <= 0.0 || (self.test_size >= 1.0 && h >= n) {
            ctx.push(
                Issue::builder(IssueCode::InvalidWeight)
                    .severity(Severity::Warning)
                    .message(format!(
                        "TemporalTrainTestSplitter.test_size={} is not a valid hold-out",
                        self.test_size
                    ))
                    .build(),
            );
        }
        h = h.clamp(1, n.saturating_sub(1).max(1));
        ctx.finish(vec![Split {
            train: (0..n - h).collect(),
            test: (n - h..n).collect(),
        }])
    }
}

/// Leave-one-group-out (sklearn `LeaveOneGroupOut`).
///
/// Group count is not identification `p`.
#[derive(Clone, Debug, Default)]
pub struct LeaveOneGroupOut;

impl LeaveOneGroupOut {
    /// Default LOGO splitter.
    pub fn new() -> Self {
        Self
    }

    /// One test fold per distinct group id.
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
        if ids.len() < 2 {
            ctx.push(
                Issue::builder(IssueCode::InsufficientSample)
                    .severity(Severity::Warning)
                    .message(format!("LeaveOneGroupOut has {} groups", ids.len()))
                    .build(),
            );
        }
        let mut folds = Vec::with_capacity(ids.len());
        for &lab in &ids {
            let mut test = Vec::new();
            let mut train = Vec::new();
            for i in 0..n {
                if groups[i].round() as i64 == lab {
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

/// Stratified group k-fold (sklearn `StratifiedGroupKFold`).
///
/// Whole groups stay together; folds are filled to balance the majority
/// class of each group. Group count is not identification `p`.
#[derive(Clone, Debug)]
pub struct StratifiedGroupKFold {
    /// Number of folds.
    pub n_splits: usize,
}

impl Default for StratifiedGroupKFold {
    fn default() -> Self {
        Self { n_splits: 5 }
    }
}

impl StratifiedGroupKFold {
    /// `k` stratified group folds.
    pub fn new(n_splits: usize) -> Self {
        Self { n_splits }
    }

    /// Split using labels `y` and group ids `groups`.
    pub fn split(
        &self,
        y: &Vector,
        groups: &Vector,
        session: &Session,
    ) -> Result<Qualified<Vec<Split>>> {
        let mut ctx = FitCtx::with_session(session.clone());
        let n = y.len().min(groups.len());
        if y.len() != groups.len() {
            ctx.push(
                Issue::builder(IssueCode::DimensionMismatch)
                    .message("StratifiedGroupKFold: y length ≠ groups length")
                    .build(),
            );
        }
        let k = self.n_splits.max(2);
        if self.n_splits < 2 {
            ctx.push(
                Issue::builder(IssueCode::InvalidWeight)
                    .severity(Severity::Warning)
                    .message("StratifiedGroupKFold.n_splits < 2; using 2")
                    .build(),
            );
        }
        let mut ids: Vec<i64> = Vec::new();
        for i in 0..n {
            if !groups[i].is_finite() {
                continue;
            }
            let lab = groups[i].round() as i64;
            if !ids.contains(&lab) {
                ids.push(lab);
            }
        }
        if ids.len() < k {
            ctx.push(
                Issue::builder(IssueCode::InsufficientSample)
                    .severity(Severity::Warning)
                    .message(format!(
                        "StratifiedGroupKFold requested {k} folds but only {} groups",
                        ids.len()
                    ))
                    .build(),
            );
        }
        let mut majority: Vec<(i64, i64, usize)> = Vec::new();
        for &g in &ids {
            let mut counts: Vec<(i64, usize)> = Vec::new();
            let mut size = 0usize;
            for i in 0..n {
                if groups[i].round() as i64 != g {
                    continue;
                }
                size += 1;
                if !y[i].is_finite() {
                    continue;
                }
                let lab = y[i].round() as i64;
                if let Some(slot) = counts.iter_mut().find(|(c, _)| *c == lab) {
                    slot.1 += 1;
                } else {
                    counts.push((lab, 1));
                }
            }
            counts.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
            let maj = counts.first().map(|c| c.0).unwrap_or(0);
            majority.push((g, maj, size));
        }
        majority.sort_by(|a, b| b.2.cmp(&a.2).then(a.0.cmp(&b.0)));
        let k_use = k.min(ids.len().max(1));
        let mut fold_groups: Vec<Vec<i64>> = vec![Vec::new(); k_use];
        let mut fold_load = vec![0usize; k_use];
        let mut fold_class: Vec<Vec<(i64, usize)>> = vec![Vec::new(); k_use];
        for &(g, maj, size) in &majority {
            let mut best = 0usize;
            let mut best_key = (usize::MAX, usize::MAX);
            for f in 0..k_use {
                let class_n = fold_class[f]
                    .iter()
                    .find(|(c, _)| *c == maj)
                    .map(|s| s.1)
                    .unwrap_or(0);
                let key = (class_n, fold_load[f]);
                if key < best_key {
                    best_key = key;
                    best = f;
                }
            }
            fold_groups[best].push(g);
            fold_load[best] += size;
            if let Some(slot) = fold_class[best].iter_mut().find(|(c, _)| *c == maj) {
                slot.1 += size;
            } else {
                fold_class[best].push((maj, size));
            }
        }
        let mut folds = Vec::with_capacity(k_use);
        for f in 0..k_use {
            let mut test = Vec::new();
            let mut train = Vec::new();
            for i in 0..n {
                let g = groups[i].round() as i64;
                if fold_groups[f].contains(&g) {
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

/// Observed score, permutation scores, and a Monte Carlo *p* (sklearn
/// `permutation_test_score`).
#[derive(Clone, Debug)]
pub struct PermutationTestScore {
    /// Mean KFold Ridge \(R^2\) on the true labels.
    pub score: f64,
    /// The same score after each label permutation.
    pub permutation_scores: Vector,
    /// \((1 + \#\{s_\pi \ge s\}) / (n_{\mathrm{perm}}+1)\).
    pub pvalue: f64,
}

/// Permutation test of a Ridge \(R^2\) against shuffled labels.
///
/// Inner residual / rank failures are not promoted. The Monte Carlo *p* is
/// recorded as unreliable (it is not an exact permutation tail).
pub fn permutation_test_score(
    x: &Matrix,
    y: &Vector,
    n_permutations: usize,
    n_splits: usize,
    session: &Session,
) -> Result<Qualified<PermutationTestScore>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
    let k = n_splits.max(2);
    if n_splits < 2 {
        ctx.push(
            Issue::builder(IssueCode::InvalidWeight)
                .severity(Severity::Warning)
                .message("permutation_test_score n_splits < 2; using 2")
                .build(),
        );
    }
    let nperm = n_permutations.max(1);
    if n_permutations < 1 {
        ctx.push(
            Issue::builder(IssueCode::InvalidWeight)
                .severity(Severity::Warning)
                .message("permutation_test_score n_permutations < 1; using 1")
                .build(),
        );
    }
    let folds = match KFold::new(k).split(x.nrows(), &session.child("pts_k")) {
        Ok(q) => q.value,
        Err(e) => {
            if !matches!(
                e.primary.code,
                IssueCode::ResidualTooLarge
                    | IssueCode::NearSingular
                    | IssueCode::RankZero
                    | IssueCode::R2IsOne
            ) {
                ctx.push(e.primary);
            }
            return ctx.finish(PermutationTestScore {
                score: f64::NAN,
                permutation_scores: Vector::zeros(0),
                pvalue: f64::NAN,
            });
        }
    };
    let score_of = |yy: &Vector, sess: &Session| -> f64 {
        let mut acc = 0.0;
        let mut used: f64 = 0.0;
        for (fi, fold) in folds.iter().enumerate() {
            let xt = take_rows(x, &fold.train);
            let yt = take_vec(yy, &fold.train);
            let xv = take_rows(x, &fold.test);
            let yv = take_vec(yy, &fold.test);
            let child = sess.child(format!("f{fi}"));
            match Ridge::new(0.1).fit(&xt, &yt, &child) {
                Ok(q) => {
                    if let Ok(pred) = q.value.predict(&xv, &sess.child("p")) {
                        if let Ok(s) = r2(&yv, &pred.value, &sess.child("r")) {
                            if s.value.is_finite() {
                                acc += s.value;
                                used += 1.0;
                            }
                        }
                    }
                }
                Err(_) => {}
            }
        }
        if used > 0.0 {
            acc / used
        } else {
            f64::NAN
        }
    };
    let observed = score_of(y, &session.child("obs"));
    let mut rng = Rng::new(11);
    let mut yb = y.as_slice().to_vec();
    let mut perm = Vec::with_capacity(nperm);
    for _ in 0..nperm {
        rng.shuffle(&mut yb);
        perm.push(score_of(&Vector::from_slice(&yb), &session.child("perm")));
    }
    let ge = perm
        .iter()
        .filter(|&&s| s.is_finite() && observed.is_finite() && s >= observed)
        .count();
    let pvalue = (1 + ge) as f64 / (nperm + 1) as f64;
    ctx.push(
        Issue::builder(IssueCode::PValueUnreliable)
            .severity(Severity::Advisory)
            .message("permutation_test_score p is a Monte Carlo tail, not an exact permutation p")
            .compromise(NumericalCompromise::new(
                "exact permutation tail",
                format!("{nperm} label shuffles of a Ridge KFold R²"),
                "the enumerator includes the observed score (plus-one smoothing)",
                "read p as a Monte Carlo upper bound on the exchangeability test",
            ))
            .build(),
    );
    ctx.finish(PermutationTestScore {
        score: observed,
        permutation_scores: Vector::from_iter(perm),
        pvalue,
    })
}

/// Cutoff-origin splitter (sktime `CutoffSplitter`).
///
/// Train is `[0, cutoff)`, test is `[cutoff, cutoff+fh)`. Cutoff count is not
/// identification `p`.
#[derive(Clone, Debug)]
pub struct CutoffSplitter {
    /// Forecast horizon.
    pub fh: usize,
    /// Inclusive origins at which the test window starts.
    pub cutoffs: Vec<usize>,
}

impl CutoffSplitter {
    /// Splitter with horizon `fh` at the given cutoffs.
    pub fn new(fh: usize, cutoffs: Vec<usize>) -> Self {
        Self { fh, cutoffs }
    }

    /// Materialize one fold per cutoff that fits in `n`.
    pub fn split(&self, n: usize, session: &Session) -> Result<Qualified<Vec<Split>>> {
        let mut ctx = FitCtx::with_session(session.clone());
        let h = self.fh.max(1);
        if self.fh == 0 {
            ctx.push(
                Issue::builder(IssueCode::WindowTooShort)
                    .message("CutoffSplitter.fh=0; using 1")
                    .build(),
            );
        }
        let mut folds = Vec::new();
        for &c in &self.cutoffs {
            if c == 0 || c + h > n {
                ctx.push(
                    Issue::builder(IssueCode::InsufficientSample)
                        .severity(Severity::Warning)
                        .message(format!(
                            "CutoffSplitter cutoff={c} fh={h} does not fit n={n}"
                        ))
                        .build(),
                );
                continue;
            }
            folds.push(Split {
                train: (0..c).collect(),
                test: (c..c + h).collect(),
            });
        }
        if folds.is_empty() {
            ctx.push(
                Issue::builder(IssueCode::InsufficientSample)
                    .severity(Severity::Warning)
                    .message("CutoffSplitter produced no folds")
                    .build(),
            );
        }
        ctx.finish(folds)
    }
}

/// Single causal window (sktime `SingleWindowSplitter`).
#[derive(Clone, Debug)]
pub struct SingleWindowSplitter {
    /// Training window length.
    pub window_length: usize,
    /// Forecast horizon.
    pub fh: usize,
}

impl Default for SingleWindowSplitter {
    fn default() -> Self {
        Self {
            window_length: 8,
            fh: 1,
        }
    }
}

impl SingleWindowSplitter {
    /// One window of length `window_length` and horizon `fh`.
    pub fn new(window_length: usize, fh: usize) -> Self {
        Self { window_length, fh }
    }

    /// The last admissible causal window on `n` rows.
    pub fn split(&self, n: usize, session: &Session) -> Result<Qualified<Vec<Split>>> {
        let mut ctx = FitCtx::with_session(session.clone());
        let w = self.window_length.max(1);
        let h = self.fh.max(1);
        if self.window_length < 2 {
            ctx.push(
                Issue::builder(IssueCode::WindowTooShort)
                    .message(format!(
                        "SingleWindowSplitter.window_length={} < 2",
                        self.window_length
                    ))
                    .build(),
            );
        }
        if w + h > n {
            ctx.push(
                Issue::builder(IssueCode::InsufficientSample)
                    .severity(Severity::Warning)
                    .message(format!(
                        "SingleWindowSplitter window={w} fh={h} does not fit n={n}"
                    ))
                    .build(),
            );
            return ctx.finish(Vec::new());
        }
        let start = n - w - h;
        ctx.finish(vec![Split {
            train: (start..start + w).collect(),
            test: (start + w..start + w + h).collect(),
        }])
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
        let rk = RepeatedKFold::new(3, 2)
            .split(12, &Session::new("ms", "rkf"))
            .unwrap()
            .value;
        assert_eq!(rk.len(), 6);
        let ss = ShuffleSplit::new(4)
            .split(16, &Session::new("ms", "shs"))
            .unwrap()
            .value;
        assert_eq!(ss.len(), 4);
        let x = Matrix::from_fn(24, 1, |i, _| i as f64);
        let y = Vector::from_iter((0..24).map(|i| 1.0 + 2.0 * i as f64 + 0.2 * ((i % 3) as f64)));
        let oof = cross_val_predict(&x, &y, &KFold::new(4), &Session::new("ms", "cvp"))
            .unwrap()
            .value;
        assert_eq!(oof.len(), 24);
        assert!(oof.as_slice().iter().all(|v| v.is_finite()));
        let gs = GridSearchCV::new(vec![0.01, 0.1, 1.0])
            .fit(&x, &y, &Session::new("ms", "gscv"))
            .unwrap();
        assert!(gs.value.best_alpha.is_finite());
        let rs = RandomizedSearchCV::new(4)
            .fit(&x, &y, &Session::new("ms", "rscv"))
            .unwrap();
        assert!(rs.value.best_alpha.is_finite());
        let yb = Vector::from_iter((0..16).map(|i| if i < 8 { 0.0 } else { 1.0 }));
        let sss = StratifiedShuffleSplit::new(3)
            .split(&yb, &Session::new("ms", "sss"))
            .unwrap()
            .value;
        assert_eq!(sss.len(), 3);
        let rsk = RepeatedStratifiedKFold::new(2, 2)
            .split(&yb, &Session::new("ms", "rsk"))
            .unwrap()
            .value;
        assert_eq!(rsk.len(), 4);
        let gg = Vector::from_iter((0..16).map(|i| (i / 4) as f64));
        let gss = GroupShuffleSplit::new(2)
            .split(&gg, &Session::new("ms", "gss"))
            .unwrap()
            .value;
        assert_eq!(gss.len(), 2);
        let sl = SlidingWindowSplitter::new(8, 2)
            .split(24, &Session::new("ms", "sw"))
            .unwrap()
            .value;
        assert!(!sl.is_empty());
        assert!(sl[0].train.iter().max().unwrap() < sl[0].test.iter().min().unwrap());
        let ew = ExpandingWindowSplitter::new(8, 2)
            .split(24, &Session::new("ms", "ew"))
            .unwrap()
            .value;
        assert!(!ew.is_empty());
        assert_eq!(ew[0].train[0], 0);
        assert!(ew[0].train.len() < ew.last().unwrap().train.len());
        let hg = HalvingGridSearchCV::new(vec![0.01, 0.1, 1.0, 10.0])
            .fit(&x, &y, &Session::new("ms", "hgs"))
            .unwrap();
        assert!(hg.value.best_alpha.is_finite());
        let hr = HalvingRandomSearchCV::new(6)
            .fit(&x, &y, &Session::new("ms", "hrs"))
            .unwrap();
        assert!(hr.value.best_alpha.is_finite());
        let lc = learning_curve(&x, &y, &[0.5, 1.0], 3, &Session::new("ms", "lc")).unwrap();
        assert_eq!(lc.value.train_scores.len(), 2);
        assert!(lc
            .value
            .test_scores
            .as_slice()
            .iter()
            .all(|v| v.is_finite()));
        let vc = validation_curve_ridge(&x, &y, &[0.1, 1.0], 3, &Session::new("ms", "vc")).unwrap();
        assert_eq!(vc.value.param_values.len(), 2);
        let pi = permutation_importance(&x, &y, 3, &Session::new("ms", "pi")).unwrap();
        assert_eq!(pi.value.importances_mean.len(), 1);
        let grid = Vector::from_slice(&[0.0, 8.0, 16.0]);
        let pd = partial_dependence(&x, &y, 0, &grid, &Session::new("ms", "pd")).unwrap();
        assert_eq!(pd.value.average.len(), 3);
        assert!(pd.value.average.as_slice().iter().all(|v| v.is_finite()));
        let lpo = LeavePOut::new(2)
            .split(6, &Session::new("ms", "lpo"))
            .unwrap()
            .value;
        assert_eq!(lpo.len(), 15);
        assert!(lpo.iter().all(|s| s.test.len() == 2 && s.train.len() == 4));
        let tf = Vector::from_slice(&[-1.0, 0.0, 0.0, 1.0, 1.0, -1.0]);
        let pre = PredefinedSplit::new(tf)
            .split(&Session::new("ms", "pre"))
            .unwrap()
            .value;
        assert_eq!(pre.len(), 2);
        let tt = TemporalTrainTestSplitter::new(0.25)
            .split(20, &Session::new("ms", "ttt"))
            .unwrap()
            .value;
        assert_eq!(tt.len(), 1);
        assert_eq!(tt[0].test.len(), 5);
        assert!(tt[0].train.iter().max().unwrap() < tt[0].test.iter().min().unwrap());
        let g = Vector::from_slice(&[0.0, 0.0, 1.0, 1.0, 2.0, 2.0]);
        let logo = LeaveOneGroupOut::new()
            .split(&g, &Session::new("ms", "logo"))
            .unwrap()
            .value;
        assert_eq!(logo.len(), 3);
        let yg = Vector::from_iter((0..16).map(|i| if i % 4 < 2 { 0.0 } else { 1.0 }));
        let gg = Vector::from_iter((0..16).map(|i| (i / 4) as f64));
        let sgk = StratifiedGroupKFold::new(2)
            .split(&yg, &gg, &Session::new("ms", "sgk"))
            .unwrap()
            .value;
        assert_eq!(sgk.len(), 2);
        let pts = permutation_test_score(&x, &y, 8, 3, &Session::new("ms", "pts")).unwrap();
        assert!(pts.value.score.is_finite());
        assert_eq!(pts.value.permutation_scores.len(), 8);
        assert!(pts.value.pvalue <= 1.0);
        let cut = CutoffSplitter::new(2, vec![8, 12])
            .split(20, &Session::new("ms", "cut"))
            .unwrap()
            .value;
        assert_eq!(cut.len(), 2);
        assert!(cut[0].train.iter().max().unwrap() < cut[0].test.iter().min().unwrap());
        let sw1 = SingleWindowSplitter::new(8, 2)
            .split(20, &Session::new("ms", "sw1"))
            .unwrap()
            .value;
        assert_eq!(sw1.len(), 1);
        assert_eq!(sw1[0].train.len(), 8);
        assert_eq!(sw1[0].test.len(), 2);
    }
}
