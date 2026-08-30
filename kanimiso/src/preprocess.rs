//! sklearn-style preprocessing: scalers, encoders, imputers, and expansions.
//!
//! Every fit inspects columns with [`signlred`]. An all-missing column is
//! [`IssueCode::ImputationUndefined`] (the fill value is not a statistic of
//! observed data). A constant column is a warning. Polynomial maps that
//! explode `p` relative to `n` raise [`IssueCode::PolynomialExplosion`].

use crate::context::FitCtx;
use crate::data::{Matrix, Vector};
use crate::special::norm_cdf;
use crate::traits::{FitSeries, FitUnsupervised, PartialFit, Transform};
use crate::validate::inspect_classes;
use ojizou_san::{IncrementalExplain, Session};
use signlred::{
    slice_stats, IncrementalQuality, Issue, IssueCode, Meaninglessness, Qualified, Result,
    SliceStats,
};

/// Column-wise mean / std scaler (sklearn `StandardScaler`).
#[derive(Clone, Debug)]
pub struct StandardScaler {
    /// Subtract the column mean.
    pub with_mean: bool,
    /// Divide by the sample standard deviation.
    pub with_std: bool,
    mean: Vector,
    scale: Vector,
    count: Vector,
    m2: Vector,
    fitted: bool,
}

impl Default for StandardScaler {
    fn default() -> Self {
        Self {
            with_mean: true,
            with_std: true,
            mean: Vector::zeros(0),
            scale: Vector::zeros(0),
            count: Vector::zeros(0),
            m2: Vector::zeros(0),
            fitted: false,
        }
    }
}

impl StandardScaler {
    /// Mean-and-std scaler.
    pub fn new() -> Self {
        Self::default()
    }

    /// Fitted column means (NaN where a column had no finite values).
    pub fn mean(&self) -> &Vector {
        &self.mean
    }

    /// Fitted scales (1 when a column is constant so the transform stays finite).
    pub fn scale(&self) -> &Vector {
        &self.scale
    }

    fn ensure_p(&mut self, p: usize) {
        if self.mean.len() != p {
            self.mean = Vector::zeros(p);
            self.scale = Vector::filled(p, 1.0);
            self.count = Vector::zeros(p);
            self.m2 = Vector::zeros(p);
        }
    }

    fn absorb_matrix(&mut self, x: &Matrix) {
        self.ensure_p(x.ncols());
        for j in 0..x.ncols() {
            for i in 0..x.nrows() {
                let v = x.get(i, j);
                if !v.is_finite() {
                    continue;
                }
                let n = self.count[j] + 1.0;
                let d = v - self.mean[j];
                self.mean[j] += d / n;
                self.m2[j] += d * (v - self.mean[j]);
                self.count[j] = n;
            }
        }
        self.refresh_scale();
        self.fitted = true;
    }

    fn refresh_scale(&mut self) {
        for j in 0..self.mean.len() {
            let n = self.count[j];
            let std = if n >= 2.0 {
                (self.m2[j] / (n - 1.0)).max(0.0).sqrt()
            } else {
                0.0
            };
            self.scale[j] = if self.with_std && std > 0.0 { std } else { 1.0 };
        }
    }
}

impl FitUnsupervised for StandardScaler {
    type Fitted = Self;
    fn fit_unsupervised(&mut self, x: &Matrix, session: &Session) -> Result<Qualified<Self>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_columns_for_preprocess(&mut ctx, x);
        *self = Self {
            with_mean: self.with_mean,
            with_std: self.with_std,
            ..Self::default()
        };
        self.absorb_matrix(x);
        warn_constant_fitted(&mut ctx, &col_stats(x));
        ctx.finish(self.clone())
    }
}

impl PartialFit for StandardScaler {
    fn partial_fit(
        &mut self,
        x: &Matrix,
        _y: Option<&Vector>,
        session: &Session,
    ) -> Result<Qualified<IncrementalExplain>> {
        let mut ctx = FitCtx::with_session(session.child("partial_fit"));
        if x.nrows() == 0 || x.ncols() == 0 {
            if !self.fitted {
                ctx.push(Issue::builder(IssueCode::PartialFitBeforeInit).build());
            }
            return finish_explain(ctx, reject_explain(0, x.nrows(), 0, "empty batch"));
        }
        if self.fitted && self.mean.len() != x.ncols() {
            ctx.push(
                Issue::builder(IssueCode::FeatureSpaceChangedOnline)
                    .message(format!(
                        "scaler has {} columns; batch has {}",
                        self.mean.len(),
                        x.ncols()
                    ))
                    .build(),
            );
            return finish_explain(
                ctx,
                reject_explain(
                    0,
                    x.nrows(),
                    self.count.as_slice().iter().sum::<f64>() as u64,
                    "feature space changed",
                ),
            );
        }
        inspect_columns_for_preprocess(&mut ctx, x);
        let before = self.mean.clone();
        self.absorb_matrix(x);
        let n_seen = self.count.as_slice().iter().cloned().fold(0.0, f64::max) as u64;
        let delta = if before.len() == self.mean.len() {
            self.mean.sub(&before).norm()
        } else {
            self.mean.norm()
        };
        let mut q = IncrementalQuality::new(n_seen.saturating_sub(1), x.nrows(), n_seen);
        q.effective_sample_size = n_seen as f64;
        q.parameter_delta_norm = Some(delta);
        q.information_gain = Some(delta);
        q.still_identified = n_seen >= 2;
        q.warmup = n_seen < 2;
        q.explanation = format!(
            "Welford update of column means/stds on {} rows; ||Δμ||={delta:.4e}",
            x.nrows()
        );
        flag_uninformative(&mut ctx, &q);
        finish_explain(
            ctx,
            IncrementalExplain::from_quality(
                q,
                "column means and scales",
                "online Welford moments",
                "previous running moments",
                "updated running moments",
            ),
        )
    }
}

impl Transform for StandardScaler {
    fn transform(&self, x: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
        let mut ctx = FitCtx::with_session(session.child("transform"));
        if !self.fitted {
            ctx.push(Issue::builder(IssueCode::PartialFitBeforeInit).build());
            return ctx.finish(x.clone());
        }
        if x.ncols() != self.mean.len() {
            ctx.push(
                Issue::builder(IssueCode::DimensionMismatch)
                    .message("StandardScaler transform column count ≠ fitted p")
                    .build(),
            );
        }
        let p = x.ncols().min(self.mean.len());
        let out = Matrix::from_fn(x.nrows(), x.ncols(), |i, j| {
            let v = x.get(i, j);
            if !v.is_finite() || j >= p {
                return v;
            }
            let mut z = v;
            if self.with_mean {
                z -= self.mean[j];
            }
            if self.with_std {
                z /= self.scale[j];
            }
            z
        });
        ctx.finish(out)
    }
}

/// Map each column onto `[0, 1]` from the fitted min/max.
#[derive(Clone, Debug)]
pub struct MinMaxScaler {
    /// Target range lower bound.
    pub feature_min: f64,
    /// Target range upper bound.
    pub feature_max: f64,
    data_min: Vector,
    data_max: Vector,
    fitted: bool,
}

impl Default for MinMaxScaler {
    fn default() -> Self {
        Self {
            feature_min: 0.0,
            feature_max: 1.0,
            data_min: Vector::zeros(0),
            data_max: Vector::zeros(0),
            fitted: false,
        }
    }
}

impl MinMaxScaler {
    /// Unit-interval scaler.
    pub fn new() -> Self {
        Self::default()
    }
}

impl FitUnsupervised for MinMaxScaler {
    type Fitted = Self;
    fn fit_unsupervised(&mut self, x: &Matrix, session: &Session) -> Result<Qualified<Self>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_columns_for_preprocess(&mut ctx, x);
        let stats = col_stats(x);
        self.data_min = Vector::from_iter(stats.iter().map(|s| s.min));
        self.data_max = Vector::from_iter(stats.iter().map(|s| s.max));
        self.fitted = true;
        warn_constant_fitted(&mut ctx, &stats);
        ctx.finish(self.clone())
    }
}

impl Transform for MinMaxScaler {
    fn transform(&self, x: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
        let mut ctx = FitCtx::with_session(session.child("transform"));
        if !self.fitted {
            ctx.push(Issue::builder(IssueCode::PartialFitBeforeInit).build());
            return ctx.finish(x.clone());
        }
        let lo = self.feature_min;
        let hi = self.feature_max;
        let p = x.ncols().min(self.data_min.len());
        let out = Matrix::from_fn(x.nrows(), x.ncols(), |i, j| {
            let v = x.get(i, j);
            if !v.is_finite() || j >= p {
                return v;
            }
            let span = self.data_max[j] - self.data_min[j];
            if span.abs() <= f64::EPSILON {
                return lo;
            }
            lo + (hi - lo) * (v - self.data_min[j]) / span
        });
        ctx.finish(out)
    }
}

/// Median / IQR scaler (sklearn `RobustScaler`).
#[derive(Clone, Debug)]
pub struct RobustScaler {
    /// Subtract the column median.
    pub with_centering: bool,
    /// Divide by the interquartile range.
    pub with_scaling: bool,
    center: Vector,
    scale: Vector,
    fitted: bool,
}

impl Default for RobustScaler {
    fn default() -> Self {
        Self {
            with_centering: true,
            with_scaling: true,
            center: Vector::zeros(0),
            scale: Vector::zeros(0),
            fitted: false,
        }
    }
}

impl RobustScaler {
    /// Median / IQR scaler.
    pub fn new() -> Self {
        Self::default()
    }
}

impl FitUnsupervised for RobustScaler {
    type Fitted = Self;
    fn fit_unsupervised(&mut self, x: &Matrix, session: &Session) -> Result<Qualified<Self>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_columns_for_preprocess(&mut ctx, x);
        let (n, p) = x.shape();
        self.center = Vector::zeros(p);
        self.scale = Vector::filled(p, 1.0);
        for j in 0..p {
            let col: Vec<f64> = (0..n).map(|i| x.get(i, j)).collect();
            let med = quantile(&col, 0.5);
            let iqr = quantile(&col, 0.75) - quantile(&col, 0.25);
            self.center[j] = med;
            self.scale[j] = if iqr.is_finite() && iqr > 0.0 {
                iqr
            } else {
                1.0
            };
            if !iqr.is_finite() || iqr == 0.0 {
                let st = slice_stats(&col);
                if st.count > 0 {
                    ctx.push(constant_col_issue(j, st));
                }
            }
        }
        self.fitted = true;
        ctx.finish(self.clone())
    }
}

impl Transform for RobustScaler {
    fn transform(&self, x: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
        let mut ctx = FitCtx::with_session(session.child("transform"));
        if !self.fitted {
            ctx.push(Issue::builder(IssueCode::PartialFitBeforeInit).build());
            return ctx.finish(x.clone());
        }
        let p = x.ncols().min(self.center.len());
        let out = Matrix::from_fn(x.nrows(), x.ncols(), |i, j| {
            let v = x.get(i, j);
            if !v.is_finite() || j >= p {
                return v;
            }
            let mut z = v;
            if self.with_centering {
                z -= self.center[j];
            }
            if self.with_scaling {
                z /= self.scale[j];
            }
            z
        });
        ctx.finish(out)
    }
}

/// Divide each column by its maximum absolute value.
#[derive(Clone, Debug)]
pub struct MaxAbsScaler {
    scale: Vector,
    fitted: bool,
}

impl Default for MaxAbsScaler {
    fn default() -> Self {
        Self {
            scale: Vector::zeros(0),
            fitted: false,
        }
    }
}

impl MaxAbsScaler {
    /// Max-abs scaler.
    pub fn new() -> Self {
        Self::default()
    }
}

impl FitUnsupervised for MaxAbsScaler {
    type Fitted = Self;
    fn fit_unsupervised(&mut self, x: &Matrix, session: &Session) -> Result<Qualified<Self>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_columns_for_preprocess(&mut ctx, x);
        let stats = col_stats(x);
        self.scale = Vector::from_iter(stats.iter().map(|s| {
            let m = s.min.abs().max(s.max.abs());
            if m > 0.0 && m.is_finite() {
                m
            } else {
                1.0
            }
        }));
        self.fitted = true;
        warn_constant_fitted(&mut ctx, &stats);
        ctx.finish(self.clone())
    }
}

impl Transform for MaxAbsScaler {
    fn transform(&self, x: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
        let mut ctx = FitCtx::with_session(session.child("transform"));
        if !self.fitted {
            ctx.push(Issue::builder(IssueCode::PartialFitBeforeInit).build());
            return ctx.finish(x.clone());
        }
        let p = x.ncols().min(self.scale.len());
        let out = Matrix::from_fn(x.nrows(), x.ncols(), |i, j| {
            let v = x.get(i, j);
            if !v.is_finite() || j >= p {
                v
            } else {
                v / self.scale[j]
            }
        });
        ctx.finish(out)
    }
}

/// Row-wise vector normalization.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NormKind {
    /// ℓ₁.
    L1,
    /// ℓ₂.
    L2,
    /// Max-abs.
    Max,
}

/// sklearn `Normalizer`.
#[derive(Clone, Debug)]
pub struct Normalizer {
    /// Row norm.
    pub norm: NormKind,
}

impl Default for Normalizer {
    fn default() -> Self {
        Self { norm: NormKind::L2 }
    }
}

impl Normalizer {
    /// ℓ₂ row normalizer.
    pub fn new() -> Self {
        Self::default()
    }
}

impl FitUnsupervised for Normalizer {
    type Fitted = Self;
    fn fit_unsupervised(&mut self, x: &Matrix, session: &Session) -> Result<Qualified<Self>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_columns_for_preprocess(&mut ctx, x);
        ctx.finish(self.clone())
    }
}

impl Transform for Normalizer {
    fn transform(&self, x: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
        let ctx = FitCtx::with_session(session.child("transform"));
        let out = Matrix::from_fn(x.nrows(), x.ncols(), |i, j| {
            let row_norm = match self.norm {
                NormKind::L1 => (0..x.ncols())
                    .map(|c| x.get(i, c).abs())
                    .filter(|v| v.is_finite())
                    .sum::<f64>(),
                NormKind::L2 => (0..x.ncols())
                    .map(|c| {
                        let v = x.get(i, c);
                        if v.is_finite() {
                            v * v
                        } else {
                            0.0
                        }
                    })
                    .sum::<f64>()
                    .sqrt(),
                NormKind::Max => (0..x.ncols())
                    .map(|c| x.get(i, c).abs())
                    .filter(|v| v.is_finite())
                    .fold(0.0, f64::max),
            };
            let v = x.get(i, j);
            if !v.is_finite() || row_norm <= 0.0 {
                v
            } else {
                v / row_norm
            }
        });
        ctx.finish(out)
    }
}

/// One-hot dummy expansion.
#[derive(Clone, Debug)]
pub struct OneHotEncoder {
    /// Drop the first level of each feature (needed for a full-rank design with intercept).
    pub drop_first: bool,
    levels: Vec<Vec<f64>>,
    fitted: bool,
}

impl Default for OneHotEncoder {
    fn default() -> Self {
        Self {
            drop_first: false,
            levels: Vec::new(),
            fitted: false,
        }
    }
}

impl OneHotEncoder {
    /// Keep every level (will warn about the dummy trap).
    pub fn new() -> Self {
        Self::default()
    }
}

impl FitUnsupervised for OneHotEncoder {
    type Fitted = Self;
    fn fit_unsupervised(&mut self, x: &Matrix, session: &Session) -> Result<Qualified<Self>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_columns_for_preprocess(&mut ctx, x);
        self.levels = (0..x.ncols())
            .map(|j| unique_sorted(&(0..x.nrows()).map(|i| x.get(i, j)).collect::<Vec<_>>()))
            .collect();
        if !self.drop_first {
            ctx.push(
                Issue::builder(IssueCode::OneHotFullRankViolation)
                    .message(
                        "drop_first=false: one-hot columns plus an intercept are linearly dependent",
                    )
                    .build(),
            );
        }
        self.fitted = true;
        ctx.finish(self.clone())
    }
}

impl Transform for OneHotEncoder {
    fn transform(&self, x: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
        let mut ctx = FitCtx::with_session(session.child("transform"));
        if !self.fitted {
            ctx.push(Issue::builder(IssueCode::PartialFitBeforeInit).build());
            return ctx.finish(Matrix::zeros(x.nrows(), 0));
        }
        let widths: Vec<usize> = self
            .levels
            .iter()
            .map(|lv| {
                let n = lv.len();
                if self.drop_first {
                    n.saturating_sub(1)
                } else {
                    n
                }
            })
            .collect();
        let p_out: usize = widths.iter().sum();
        let out = Matrix::from_fn(x.nrows(), p_out, |i, jj| {
            let (col, local) = decode_block(&widths, jj);
            let v = x.get(i, col);
            let lv = &self.levels[col];
            let start = if self.drop_first { 1 } else { 0 };
            if start + local >= lv.len() {
                return 0.0;
            }
            if almost_eq(v, lv[start + local]) {
                1.0
            } else {
                0.0
            }
        });
        ctx.finish(out)
    }
}

/// Map each column's observed values onto `0..k`.
#[derive(Clone, Debug)]
pub struct OrdinalEncoder {
    levels: Vec<Vec<f64>>,
    fitted: bool,
}

impl Default for OrdinalEncoder {
    fn default() -> Self {
        Self {
            levels: Vec::new(),
            fitted: false,
        }
    }
}

impl OrdinalEncoder {
    /// Empty encoder.
    pub fn new() -> Self {
        Self::default()
    }
}

impl FitUnsupervised for OrdinalEncoder {
    type Fitted = Self;
    fn fit_unsupervised(&mut self, x: &Matrix, session: &Session) -> Result<Qualified<Self>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_columns_for_preprocess(&mut ctx, x);
        self.levels = (0..x.ncols())
            .map(|j| unique_sorted(&(0..x.nrows()).map(|i| x.get(i, j)).collect::<Vec<_>>()))
            .collect();
        self.fitted = true;
        ctx.finish(self.clone())
    }
}

impl Transform for OrdinalEncoder {
    fn transform(&self, x: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
        let mut ctx = FitCtx::with_session(session.child("transform"));
        if !self.fitted {
            ctx.push(Issue::builder(IssueCode::PartialFitBeforeInit).build());
            return ctx.finish(x.clone());
        }
        let p = x.ncols().min(self.levels.len());
        let out = Matrix::from_fn(x.nrows(), x.ncols(), |i, j| {
            if j >= p {
                return f64::NAN;
            }
            let v = x.get(i, j);
            match self.levels[j].iter().position(|u| almost_eq(*u, v)) {
                Some(k) => k as f64,
                None => f64::NAN,
            }
        });
        ctx.finish(out)
    }
}

/// Encode a 1-d label vector as `0..k-1`.
#[derive(Clone, Debug, Default)]
pub struct LabelEncoder {
    classes: Vec<i64>,
    fitted: bool,
}

impl LabelEncoder {
    /// Empty encoder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sorted class ids.
    pub fn classes(&self) -> &[i64] {
        &self.classes
    }
}

impl FitSeries for LabelEncoder {
    type Fitted = Self;
    fn fit_series(&mut self, y: &Vector, session: &Session) -> Result<Qualified<Self>> {
        let mut ctx = FitCtx::with_session(session.clone());
        let counts = inspect_classes(&mut ctx.report, y, &ctx.policy);
        self.classes = counts.into_iter().map(|(k, _)| k).collect();
        self.fitted = true;
        ctx.finish(self.clone())
    }
}

impl Transform for LabelEncoder {
    fn transform(&self, x: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
        let mut ctx = FitCtx::with_session(session.child("transform"));
        if !self.fitted {
            ctx.push(Issue::builder(IssueCode::PartialFitBeforeInit).build());
            return ctx.finish(Matrix::zeros(x.nrows(), 1));
        }
        // Encode column 0 (or every column independently against the same map).
        let out = Matrix::from_fn(x.nrows(), x.ncols(), |i, j| {
            let lab = x.get(i, j).round() as i64;
            match self.classes.iter().position(|c| *c == lab) {
                Some(k) => k as f64,
                None => f64::NAN,
            }
        });
        ctx.finish(out)
    }
}

/// How [`SimpleImputer`] fills missing entries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImputeStrategy {
    /// Column mean of finite values.
    Mean,
    /// Column median of finite values.
    Median,
    /// Modal finite value (ties → smallest).
    MostFrequent,
}

/// Univariate imputer.
#[derive(Clone, Debug)]
pub struct SimpleImputer {
    /// Fill strategy.
    pub strategy: ImputeStrategy,
    statistics: Vector,
    fitted: bool,
}

impl Default for SimpleImputer {
    fn default() -> Self {
        Self {
            strategy: ImputeStrategy::Mean,
            statistics: Vector::zeros(0),
            fitted: false,
        }
    }
}

impl SimpleImputer {
    /// Mean imputer.
    pub fn new() -> Self {
        Self::default()
    }
}

impl FitUnsupervised for SimpleImputer {
    type Fitted = Self;
    fn fit_unsupervised(&mut self, x: &Matrix, session: &Session) -> Result<Qualified<Self>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_columns_for_preprocess(&mut ctx, x);
        let (n, p) = x.shape();
        self.statistics = Vector::zeros(p);
        for j in 0..p {
            let col: Vec<f64> = (0..n).map(|i| x.get(i, j)).collect();
            let st = slice_stats(&col);
            if st.count == 0 {
                ctx.push(imputation_undefined(j));
                self.statistics[j] = f64::NAN;
                continue;
            }
            self.statistics[j] = match self.strategy {
                ImputeStrategy::Mean => st.mean,
                ImputeStrategy::Median => quantile(&col, 0.5),
                ImputeStrategy::MostFrequent => mode_finite(&col),
            };
        }
        self.fitted = true;
        ctx.finish(self.clone())
    }
}

impl Transform for SimpleImputer {
    fn transform(&self, x: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
        let mut ctx = FitCtx::with_session(session.child("transform"));
        if !self.fitted {
            ctx.push(Issue::builder(IssueCode::PartialFitBeforeInit).build());
            return ctx.finish(x.clone());
        }
        let p = x.ncols().min(self.statistics.len());
        let out = Matrix::from_fn(x.nrows(), x.ncols(), |i, j| {
            let v = x.get(i, j);
            if v.is_finite() || j >= p {
                v
            } else {
                self.statistics[j]
            }
        });
        ctx.finish(out)
    }
}

/// k-nearest-neighbour imputer (nan-Euclidean distance).
#[derive(Clone, Debug)]
pub struct KnnImputer {
    /// Neighbours used to fill a missing coordinate.
    pub n_neighbors: usize,
    train: Matrix,
    fitted: bool,
}

impl Default for KnnImputer {
    fn default() -> Self {
        Self {
            n_neighbors: 5,
            train: Matrix::zeros(0, 0),
            fitted: false,
        }
    }
}

impl KnnImputer {
    /// Imputer with `k` neighbours.
    pub fn new(n_neighbors: usize) -> Self {
        Self {
            n_neighbors: n_neighbors.max(1),
            ..Self::default()
        }
    }
}

impl FitUnsupervised for KnnImputer {
    type Fitted = Self;
    fn fit_unsupervised(&mut self, x: &Matrix, session: &Session) -> Result<Qualified<Self>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_columns_for_preprocess(&mut ctx, x);
        self.train = x.clone();
        self.fitted = true;
        ctx.finish(self.clone())
    }
}

impl Transform for KnnImputer {
    fn transform(&self, x: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
        let mut ctx = FitCtx::with_session(session.child("transform"));
        if !self.fitted {
            ctx.push(Issue::builder(IssueCode::PartialFitBeforeInit).build());
            return ctx.finish(x.clone());
        }
        let (n, p) = x.shape();
        let mut out = x.clone();
        for i in 0..n {
            for j in 0..p {
                if out.get(i, j).is_finite() {
                    continue;
                }
                let mut neigh: Vec<(f64, usize)> = Vec::new();
                for t in 0..self.train.nrows() {
                    if let Some(d) = nan_euclidean(x, i, &self.train, t) {
                        if self.train.get(t, j).is_finite() {
                            neigh.push((d, t));
                        }
                    }
                }
                neigh.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
                neigh.truncate(self.n_neighbors);
                if neigh.is_empty() {
                    ctx.push(imputation_undefined(j));
                    continue;
                }
                let mut s = 0.0;
                let mut k = 0.0;
                for (_, t) in neigh {
                    s += self.train.get(t, j);
                    k += 1.0;
                }
                out.set(i, j, s / k);
            }
        }
        ctx.finish(out)
    }
}

/// Polynomial / interaction expansion.
#[derive(Clone, Debug)]
pub struct PolynomialFeatures {
    /// Maximum total degree.
    pub degree: usize,
    /// If true, only products of distinct features (no powers).
    pub interaction_only: bool,
    /// Include a column of ones.
    pub include_bias: bool,
    n_features_in: usize,
    monomials: Vec<Vec<usize>>,
    fitted: bool,
}

impl Default for PolynomialFeatures {
    fn default() -> Self {
        Self {
            degree: 2,
            interaction_only: false,
            include_bias: true,
            n_features_in: 0,
            monomials: Vec::new(),
            fitted: false,
        }
    }
}

impl PolynomialFeatures {
    /// Degree-`d` expander.
    pub fn new(degree: usize) -> Self {
        Self {
            degree: degree.max(1),
            ..Self::default()
        }
    }

    /// Number of output columns after `fit`.
    pub fn n_output_features(&self) -> usize {
        self.monomials.len()
    }
}

impl FitUnsupervised for PolynomialFeatures {
    type Fitted = Self;
    fn fit_unsupervised(&mut self, x: &Matrix, session: &Session) -> Result<Qualified<Self>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_columns_for_preprocess(&mut ctx, x);
        self.n_features_in = x.ncols();
        self.monomials = enumerate_monomials(
            x.ncols(),
            self.degree,
            self.interaction_only,
            self.include_bias,
        );
        let p_out = self.monomials.len();
        if p_out > x.nrows() && x.nrows() > 0 {
            ctx.push(
                Issue::builder(IssueCode::PolynomialExplosion)
                    .message(format!(
                        "polynomial map sends n={} rows to p={p_out} columns (degree {})",
                        x.nrows(),
                        self.degree
                    ))
                    .metric("n", x.nrows() as f64)
                    .metric("p_out", p_out as f64)
                    .meaninglessness(Meaninglessness::vacuous(
                        "polynomial feature map",
                        "expanded p ≫ n is interpolation, not a feature",
                        "lower the degree or collect more rows",
                    ))
                    .build(),
            );
        }
        self.fitted = true;
        ctx.finish(self.clone())
    }
}

impl Transform for PolynomialFeatures {
    fn transform(&self, x: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
        let mut ctx = FitCtx::with_session(session.child("transform"));
        if !self.fitted {
            ctx.push(Issue::builder(IssueCode::PartialFitBeforeInit).build());
            return ctx.finish(x.clone());
        }
        if x.ncols() != self.n_features_in {
            ctx.push(
                Issue::builder(IssueCode::DimensionMismatch)
                    .message("PolynomialFeatures column count ≠ fitted p")
                    .build(),
            );
        }
        let out = Matrix::from_fn(x.nrows(), self.monomials.len(), |i, k| {
            let mut v = 1.0;
            for &j in &self.monomials[k] {
                if j < x.ncols() {
                    v *= x.get(i, j);
                }
            }
            v
        });
        ctx.finish(out)
    }
}

/// Power transform family.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PowerMethod {
    /// Yeo–Johnson (defined on ℝ).
    YeoJohnson,
    /// Box–Cox (requires strictly positive data).
    BoxCox,
}

/// sklearn `PowerTransformer`.
#[derive(Clone, Debug)]
pub struct PowerTransformer {
    /// Yeo–Johnson or Box–Cox.
    pub method: PowerMethod,
    /// Standardize after the power map.
    pub standardize: bool,
    lambdas: Vector,
    mean: Vector,
    scale: Vector,
    fitted: bool,
}

impl Default for PowerTransformer {
    fn default() -> Self {
        Self {
            method: PowerMethod::YeoJohnson,
            standardize: true,
            lambdas: Vector::zeros(0),
            mean: Vector::zeros(0),
            scale: Vector::zeros(0),
            fitted: false,
        }
    }
}

impl PowerTransformer {
    /// Yeo–Johnson transformer.
    pub fn new() -> Self {
        Self::default()
    }

    /// Fitted λ per column.
    pub fn lambdas(&self) -> &Vector {
        &self.lambdas
    }
}

impl FitUnsupervised for PowerTransformer {
    type Fitted = Self;
    fn fit_unsupervised(&mut self, x: &Matrix, session: &Session) -> Result<Qualified<Self>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_columns_for_preprocess(&mut ctx, x);
        let (n, p) = x.shape();
        self.lambdas = Vector::zeros(p);
        self.mean = Vector::zeros(p);
        self.scale = Vector::filled(p, 1.0);
        for j in 0..p {
            let col: Vec<f64> = (0..n).map(|i| x.get(i, j)).collect();
            if matches!(self.method, PowerMethod::BoxCox)
                && col.iter().any(|v| v.is_finite() && *v <= 0.0)
            {
                ctx.push(
                    Issue::builder(IssueCode::NonPositiveSeries)
                        .message(format!(
                            "Box–Cox requires x>0; column {j} has non-positive values"
                        ))
                        .metric("feature_index", j as f64)
                        .build(),
                );
                continue;
            }
            let st = slice_stats(&col);
            if st.count == 0 {
                ctx.push(imputation_undefined(j));
                continue;
            }
            self.lambdas[j] = match self.method {
                PowerMethod::YeoJohnson => best_yeo_johnson(&col),
                PowerMethod::BoxCox => best_box_cox(&col),
            };
            let mapped: Vec<f64> = col
                .iter()
                .map(|&v| match self.method {
                    PowerMethod::YeoJohnson => yeo_johnson(v, self.lambdas[j]),
                    PowerMethod::BoxCox => box_cox(v, self.lambdas[j]),
                })
                .collect();
            let ms = slice_stats(&mapped);
            self.mean[j] = ms.mean;
            self.scale[j] = if ms.std() > 0.0 { ms.std() } else { 1.0 };
        }
        self.fitted = true;
        ctx.finish(self.clone())
    }
}

impl Transform for PowerTransformer {
    fn transform(&self, x: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
        let mut ctx = FitCtx::with_session(session.child("transform"));
        if !self.fitted {
            ctx.push(Issue::builder(IssueCode::PartialFitBeforeInit).build());
            return ctx.finish(x.clone());
        }
        let p = x.ncols().min(self.lambdas.len());
        let out = Matrix::from_fn(x.nrows(), x.ncols(), |i, j| {
            if j >= p {
                return x.get(i, j);
            }
            let v = x.get(i, j);
            let z = match self.method {
                PowerMethod::YeoJohnson => yeo_johnson(v, self.lambdas[j]),
                PowerMethod::BoxCox => box_cox(v, self.lambdas[j]),
            };
            if self.standardize {
                (z - self.mean[j]) / self.scale[j]
            } else {
                z
            }
        });
        ctx.finish(out)
    }
}

/// Output law of [`QuantileTransformer`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutputDistribution {
    /// Map through Φ⁻¹.
    Normal,
    /// Leave as the empirical CDF (uniform on (0, 1)).
    Uniform,
}

/// Empirical-CDF transformer.
#[derive(Clone, Debug)]
pub struct QuantileTransformer {
    /// Number of interpolation knots stored per column.
    pub n_quantiles: usize,
    /// Target distribution.
    pub output_distribution: OutputDistribution,
    references: Vec<Vec<f64>>,
    fitted: bool,
}

impl Default for QuantileTransformer {
    fn default() -> Self {
        Self {
            n_quantiles: 1000,
            output_distribution: OutputDistribution::Uniform,
            references: Vec::new(),
            fitted: false,
        }
    }
}

impl QuantileTransformer {
    /// Uniform empirical-CDF map.
    pub fn new() -> Self {
        Self::default()
    }
}

impl FitUnsupervised for QuantileTransformer {
    type Fitted = Self;
    fn fit_unsupervised(&mut self, x: &Matrix, session: &Session) -> Result<Qualified<Self>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_columns_for_preprocess(&mut ctx, x);
        let (n, p) = x.shape();
        self.references.clear();
        let nq = self.n_quantiles.max(2).min(n.max(2));
        for j in 0..p {
            let col: Vec<f64> = (0..n).map(|i| x.get(i, j)).collect();
            let st = slice_stats(&col);
            if st.count == 0 {
                ctx.push(imputation_undefined(j));
                self.references.push(Vec::new());
                continue;
            }
            let knots: Vec<f64> = (0..nq)
                .map(|k| {
                    let q = k as f64 / (nq - 1) as f64;
                    quantile(&col, q)
                })
                .collect();
            self.references.push(knots);
        }
        self.fitted = true;
        ctx.finish(self.clone())
    }
}

impl Transform for QuantileTransformer {
    fn transform(&self, x: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
        let mut ctx = FitCtx::with_session(session.child("transform"));
        if !self.fitted {
            ctx.push(Issue::builder(IssueCode::PartialFitBeforeInit).build());
            return ctx.finish(x.clone());
        }
        let p = x.ncols().min(self.references.len());
        let out = Matrix::from_fn(x.nrows(), x.ncols(), |i, j| {
            if j >= p {
                return x.get(i, j);
            }
            let u = empirical_cdf(x.get(i, j), &self.references[j]);
            match self.output_distribution {
                OutputDistribution::Uniform => u,
                OutputDistribution::Normal => norm_ppf(u),
            }
        });
        ctx.finish(out)
    }
}

/// Binning strategy for [`KBinsDiscretizer`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KBinsStrategy {
    /// Equal-width bins on `[min, max]`.
    Uniform,
    /// Equal-count (quantile) bins.
    Quantile,
    /// 1-d k-means centres.
    KMeans,
}

/// sklearn `KBinsDiscretizer` (ordinal encoding of bins).
#[derive(Clone, Debug)]
pub struct KBinsDiscretizer {
    /// Number of bins per column.
    pub n_bins: usize,
    /// Edge construction.
    pub strategy: KBinsStrategy,
    edges: Vec<Vec<f64>>,
    fitted: bool,
}

impl Default for KBinsDiscretizer {
    fn default() -> Self {
        Self {
            n_bins: 5,
            strategy: KBinsStrategy::Quantile,
            edges: Vec::new(),
            fitted: false,
        }
    }
}

impl KBinsDiscretizer {
    /// Quantile discretizer with `n_bins` bins.
    pub fn new(n_bins: usize) -> Self {
        Self {
            n_bins: n_bins.max(2),
            ..Self::default()
        }
    }
}

impl FitUnsupervised for KBinsDiscretizer {
    type Fitted = Self;
    fn fit_unsupervised(&mut self, x: &Matrix, session: &Session) -> Result<Qualified<Self>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_columns_for_preprocess(&mut ctx, x);
        let (n, p) = x.shape();
        self.edges.clear();
        for j in 0..p {
            let col: Vec<f64> = (0..n).map(|i| x.get(i, j)).collect();
            let st = slice_stats(&col);
            if st.count == 0 {
                ctx.push(imputation_undefined(j));
                self.edges.push(Vec::new());
                continue;
            }
            if st.is_constant(ctx.policy.near_zero_variance) {
                ctx.push(constant_col_issue(j, st));
            }
            let nb = self.n_bins.max(2);
            let e = match self.strategy {
                KBinsStrategy::Uniform => (0..=nb)
                    .map(|k| st.min + (st.max - st.min) * k as f64 / nb as f64)
                    .collect(),
                KBinsStrategy::Quantile => (0..=nb)
                    .map(|k| quantile(&col, k as f64 / nb as f64))
                    .collect(),
                KBinsStrategy::KMeans => kmeans_1d_edges(&col, nb),
            };
            self.edges.push(e);
        }
        self.fitted = true;
        ctx.finish(self.clone())
    }
}

impl Transform for KBinsDiscretizer {
    fn transform(&self, x: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
        let mut ctx = FitCtx::with_session(session.child("transform"));
        if !self.fitted {
            ctx.push(Issue::builder(IssueCode::PartialFitBeforeInit).build());
            return ctx.finish(x.clone());
        }
        let p = x.ncols().min(self.edges.len());
        let out = Matrix::from_fn(x.nrows(), x.ncols(), |i, j| {
            if j >= p || self.edges[j].len() < 2 {
                return f64::NAN;
            }
            bin_index(x.get(i, j), &self.edges[j]) as f64
        });
        ctx.finish(out)
    }
}

/// Threshold map `1[x > t]`.
#[derive(Clone, Debug)]
pub struct Binarizer {
    /// Values strictly greater than this become 1.
    pub threshold: f64,
}

impl Default for Binarizer {
    fn default() -> Self {
        Self { threshold: 0.0 }
    }
}

impl Binarizer {
    /// Binarizer at `threshold`.
    pub fn new(threshold: f64) -> Self {
        Self { threshold }
    }
}

impl FitUnsupervised for Binarizer {
    type Fitted = Self;
    fn fit_unsupervised(&mut self, x: &Matrix, session: &Session) -> Result<Qualified<Self>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_columns_for_preprocess(&mut ctx, x);
        ctx.finish(self.clone())
    }
}

impl Transform for Binarizer {
    fn transform(&self, x: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
        let ctx = FitCtx::with_session(session.child("transform"));
        let out = Matrix::from_fn(x.nrows(), x.ncols(), |i, j| {
            let v = x.get(i, j);
            if v.is_finite() && v > self.threshold {
                1.0
            } else if v.is_finite() {
                0.0
            } else {
                v
            }
        });
        ctx.finish(out)
    }
}

fn inspect_columns_for_preprocess(ctx: &mut FitCtx, x: &Matrix) {
    let (n, p) = x.shape();
    ctx.report.set_sample_shape(n, p);
    if n == 0 || p == 0 {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message(format!("preprocess design is {n}×{p}"))
                .build(),
        );
        return;
    }
    for j in 0..p {
        let col: Vec<f64> = (0..n).map(|i| x.get(i, j)).collect();
        let st = slice_stats(&col);
        if st.count == 0 {
            ctx.push(imputation_undefined(j));
        } else if st.is_constant(ctx.policy.near_zero_variance) {
            ctx.push(constant_col_issue(j, st));
        }
    }
}

fn col_stats(x: &Matrix) -> Vec<signlred::SliceStats> {
    (0..x.ncols())
        .map(|j| {
            let col: Vec<f64> = (0..x.nrows()).map(|i| x.get(i, j)).collect();
            slice_stats(&col)
        })
        .collect()
}

fn warn_constant_fitted(ctx: &mut FitCtx, stats: &[signlred::SliceStats]) {
    for (j, st) in stats.iter().enumerate() {
        if st.count > 0 && st.is_constant(ctx.policy.near_zero_variance) {
            // already pushed during inspect; keep this a no-op if present
            let _ = j;
        }
    }
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

fn imputation_undefined(j: usize) -> Issue {
    Issue::builder(IssueCode::ImputationUndefined)
        .message(format!(
            "column {j} is all-missing; the fill / scale is not a statistic of observed data"
        ))
        .metric("feature_index", j as f64)
        .meaninglessness(Meaninglessness::vacuous(
            "column statistic",
            "zero finite observations; mean/std/impute value is undefined",
            "drop the column or collect observations",
        ))
        .build()
}

fn quantile(xs: &[f64], q: f64) -> f64 {
    let mut v: Vec<f64> = xs.iter().copied().filter(|x| x.is_finite()).collect();
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    if v.len() == 1 {
        return v[0];
    }
    let q = q.clamp(0.0, 1.0);
    let pos = q * (v.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    let w = pos - lo as f64;
    v[lo] * (1.0 - w) + v[hi] * w
}

fn mode_finite(xs: &[f64]) -> f64 {
    let mut v: Vec<f64> = xs.iter().copied().filter(|x| x.is_finite()).collect();
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mut best = v[0];
    let mut best_c = 1usize;
    let mut cur = v[0];
    let mut c = 1usize;
    for &x in v.iter().skip(1) {
        if almost_eq(x, cur) {
            c += 1;
        } else {
            if c > best_c {
                best_c = c;
                best = cur;
            }
            cur = x;
            c = 1;
        }
    }
    if c > best_c {
        best = cur;
    }
    best
}

fn unique_sorted(xs: &[f64]) -> Vec<f64> {
    let mut v: Vec<f64> = xs.iter().copied().filter(|x| x.is_finite()).collect();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    v.dedup_by(|a, b| almost_eq(*a, *b));
    v
}

fn almost_eq(a: f64, b: f64) -> bool {
    if a == b {
        return true;
    }
    (a - b).abs() <= 1e-12 * (1.0 + a.abs().max(b.abs()))
}

fn decode_block(widths: &[usize], jj: usize) -> (usize, usize) {
    let mut acc = 0;
    for (c, w) in widths.iter().enumerate() {
        if jj < acc + *w {
            return (c, jj - acc);
        }
        acc += *w;
    }
    (widths.len().saturating_sub(1), 0)
}

fn nan_euclidean(a: &Matrix, ia: usize, b: &Matrix, ib: usize) -> Option<f64> {
    let p = a.ncols().min(b.ncols());
    let mut s = 0.0;
    let mut shared = 0.0;
    for j in 0..p {
        let ua = a.get(ia, j);
        let ub = b.get(ib, j);
        if ua.is_finite() && ub.is_finite() {
            let d = ua - ub;
            s += d * d;
            shared += 1.0;
        }
    }
    if shared == 0.0 {
        return None;
    }
    Some((s * (p as f64 / shared)).sqrt())
}

fn enumerate_monomials(
    p: usize,
    degree: usize,
    interaction_only: bool,
    include_bias: bool,
) -> Vec<Vec<usize>> {
    let mut out = Vec::new();
    if include_bias {
        out.push(Vec::new());
    }
    fn rec(
        start: usize,
        remaining: usize,
        cur: &mut Vec<usize>,
        out: &mut Vec<Vec<usize>>,
        p: usize,
        interaction_only: bool,
    ) {
        if remaining == 0 {
            return;
        }
        for j in start..p {
            cur.push(j);
            out.push(cur.clone());
            let next = if interaction_only { j + 1 } else { j };
            rec(next, remaining - 1, cur, out, p, interaction_only);
            cur.pop();
        }
    }
    rec(0, degree, &mut Vec::new(), &mut out, p, interaction_only);
    out
}

fn yeo_johnson(x: f64, lambda: f64) -> f64 {
    if !x.is_finite() {
        return x;
    }
    if x >= 0.0 {
        if (lambda - 0.0).abs() < 1e-12 {
            (x + 1.0).ln()
        } else {
            ((x + 1.0).powf(lambda) - 1.0) / lambda
        }
    } else if (lambda - 2.0).abs() < 1e-12 {
        -(-x + 1.0).ln()
    } else {
        -((-x + 1.0).powf(2.0 - lambda) - 1.0) / (2.0 - lambda)
    }
}

fn box_cox(x: f64, lambda: f64) -> f64 {
    if !x.is_finite() || x <= 0.0 {
        return f64::NAN;
    }
    if lambda.abs() < 1e-12 {
        x.ln()
    } else {
        (x.powf(lambda) - 1.0) / lambda
    }
}

fn yeo_johnson_ll(xs: &[f64], lambda: f64) -> f64 {
    let mapped: Vec<f64> = xs
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .map(|v| yeo_johnson(v, lambda))
        .collect();
    let st = slice_stats(&mapped);
    if st.count < 2 || st.variance <= 0.0 {
        return f64::NEG_INFINITY;
    }
    let n = st.count as f64;
    let mut jac = 0.0;
    for &x in xs {
        if !x.is_finite() {
            continue;
        }
        jac += if x >= 0.0 {
            (lambda - 1.0) * (x + 1.0).ln()
        } else {
            (1.0 - lambda) * (-x + 1.0).ln()
        };
    }
    -0.5 * n * st.variance.ln() + jac
}

fn box_cox_ll(xs: &[f64], lambda: f64) -> f64 {
    let mapped: Vec<f64> = xs
        .iter()
        .copied()
        .filter(|v| v.is_finite() && *v > 0.0)
        .map(|v| box_cox(v, lambda))
        .collect();
    let st = slice_stats(&mapped);
    if st.count < 2 || st.variance <= 0.0 {
        return f64::NEG_INFINITY;
    }
    let n = st.count as f64;
    let jac: f64 = xs
        .iter()
        .filter(|v| v.is_finite() && **v > 0.0)
        .map(|v| (lambda - 1.0) * v.ln())
        .sum();
    -0.5 * n * st.variance.ln() + jac
}

fn best_yeo_johnson(xs: &[f64]) -> f64 {
    let mut best_l = 1.0;
    let mut best = f64::NEG_INFINITY;
    let mut lam = -2.0;
    while lam <= 2.0 + 1e-12 {
        let ll = yeo_johnson_ll(xs, lam);
        if ll > best {
            best = ll;
            best_l = lam;
        }
        lam += 0.2;
    }
    best_l
}

fn best_box_cox(xs: &[f64]) -> f64 {
    let mut best_l = 1.0;
    let mut best = f64::NEG_INFINITY;
    let mut lam = -2.0;
    while lam <= 2.0 + 1e-12 {
        let ll = box_cox_ll(xs, lam);
        if ll > best {
            best = ll;
            best_l = lam;
        }
        lam += 0.2;
    }
    best_l
}

fn empirical_cdf(x: f64, knots: &[f64]) -> f64 {
    if knots.is_empty() || !x.is_finite() {
        return f64::NAN;
    }
    if x <= knots[0] {
        return 1e-6;
    }
    if x >= *knots.last().unwrap() {
        return 1.0 - 1e-6;
    }
    for i in 0..knots.len() - 1 {
        if x <= knots[i + 1] {
            let span = knots[i + 1] - knots[i];
            let t = if span.abs() <= 0.0 {
                0.0
            } else {
                (x - knots[i]) / span
            };
            let u0 = i as f64 / (knots.len() - 1) as f64;
            let u1 = (i + 1) as f64 / (knots.len() - 1) as f64;
            return (u0 + t * (u1 - u0)).clamp(1e-6, 1.0 - 1e-6);
        }
    }
    1.0 - 1e-6
}

/// Rational approximation to Φ⁻¹ (Peter Acklam).
fn norm_ppf(p: f64) -> f64 {
    let p = p.clamp(1e-12, 1.0 - 1e-12);
    // Use the fact that Φ(z) = norm_cdf(z); bisection is robust and dependency-free.
    let mut lo = -8.0;
    let mut hi = 8.0;
    for _ in 0..80 {
        let mid = 0.5 * (lo + hi);
        if norm_cdf(mid) < p {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

fn kmeans_1d_edges(xs: &[f64], n_bins: usize) -> Vec<f64> {
    let mut pts: Vec<f64> = xs.iter().copied().filter(|v| v.is_finite()).collect();
    if pts.is_empty() {
        return Vec::new();
    }
    pts.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let k = n_bins.min(pts.len()).max(1);
    let mut centers: Vec<f64> = (0..k)
        .map(|i| pts[i * (pts.len() - 1) / k.max(1).min(pts.len())])
        .collect();
    for _ in 0..25 {
        let mut sum = vec![0.0; k];
        let mut cnt = vec![0.0; k];
        for &x in &pts {
            let mut bi = 0usize;
            let mut bd = f64::INFINITY;
            for (c, &mu) in centers.iter().enumerate() {
                let d = (x - mu).abs();
                if d < bd {
                    bd = d;
                    bi = c;
                }
            }
            sum[bi] += x;
            cnt[bi] += 1.0;
        }
        for c in 0..k {
            if cnt[c] > 0.0 {
                centers[c] = sum[c] / cnt[c];
            }
        }
    }
    centers.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mut edges = Vec::with_capacity(k + 1);
    edges.push(pts[0]);
    for i in 0..k - 1 {
        edges.push(0.5 * (centers[i] + centers[i + 1]));
    }
    edges.push(*pts.last().unwrap());
    edges
}

fn bin_index(x: f64, edges: &[f64]) -> usize {
    if !x.is_finite() || edges.len() < 2 {
        return 0;
    }
    for i in 0..edges.len() - 1 {
        if x <= edges[i + 1] {
            return i;
        }
    }
    edges.len() - 2
}

fn reject_explain(update: u64, batch: usize, n_seen: u64, why: &str) -> IncrementalExplain {
    IncrementalExplain::from_quality(
        IncrementalQuality::new(update, batch, n_seen),
        "nothing",
        why,
        "invalid",
        "invalid",
    )
}

fn flag_uninformative(ctx: &mut FitCtx, q: &IncrementalQuality) {
    if q.is_uninformative(ctx.policy.uninformative_info_eps) {
        ctx.push(
            Issue::builder(IssueCode::UpdateWithZeroInformation)
                .incremental(q.clone())
                .message("this scaler batch did not move the running moments")
                .build(),
        );
    }
}

fn finish_explain(ctx: FitCtx, expl: IncrementalExplain) -> Result<Qualified<IncrementalExplain>> {
    ctx.session.record_incremental(expl.clone());
    ctx.finish(expl)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::{FitUnsupervised, Transform};
    use ojizou_san::Session;
    use signlred::IssueCode;

    #[test]
    fn standard_scaler_centers_and_scales() {
        let x = Matrix::from_fn(30, 2, |i, j| (i as f64) + 3.0 * (j as f64));
        let session = Session::new("scaler", "fit");
        let mut s = StandardScaler::new();
        s.fit_unsupervised(&x, &session).expect("fit");
        let z = s.transform(&x, &session).expect("transform").value;
        for j in 0..2 {
            let col = z.column(j);
            assert!(col.mean().abs() < 1e-10, "mean {}", col.mean());
            assert!((col.std() - 1.0).abs() < 1e-10, "std {}", col.std());
        }
    }

    #[test]
    fn constant_column_warns() {
        let x = Matrix::from_fn(12, 2, |i, j| if j == 0 { 4.0 } else { i as f64 });
        let session = Session::new("scaler", "fit");
        let q = StandardScaler::new()
            .fit_unsupervised(&x, &session)
            .expect("constant is a warning, not an abort");
        assert!(q.report.contains(IssueCode::ConstantFeature));
    }

    #[test]
    fn all_nan_column_errors() {
        let x = Matrix::from_fn(8, 2, |i, j| if j == 0 { f64::NAN } else { i as f64 });
        let session = Session::new("scaler", "fit");
        let err = StandardScaler::new()
            .fit_unsupervised(&x, &session)
            .unwrap_err();
        assert_eq!(err.primary().code, IssueCode::ImputationUndefined);
    }
}
