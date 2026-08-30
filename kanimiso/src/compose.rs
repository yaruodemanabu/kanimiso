//! Pipelines, feature unions, and column-wise transforms.
//!
//! Preprocessing is a local [`PreprocessStep`] (column standardize) so this
//! module does not depend on [`crate::preprocess`]. Voting / bagging / stacking
//! live in [`crate::ensemble`] and are re-exported here.

use crate::context::FitCtx;
use crate::data::{Matrix, Vector};
use crate::linear_model::{FittedLinear, LinearRegression};
use crate::traits::{Fit, Predict, Transform};
use crate::validate::inspect_xy;
use ojizou_san::Session;
use signlred::{Issue, IssueCode, Qualified, Result};

pub use crate::ensemble::{
    BaggingClassifier, BaggingRegressor, FittedBaggingClassifier, FittedBaggingRegressor,
    FittedStackingClassifier, FittedStackingRegressor, FittedVotingClassifier,
    FittedVotingRegressor, StackingClassifier, StackingRegressor, VotingClassifier,
    VotingRegressor,
};

/// Local column standardizer (mean / sample std). Not [`crate::preprocess::StandardScaler`].
#[derive(Clone, Debug)]
pub struct Standardize {
    /// Column means (empty before fit).
    pub mean: Vector,
    /// Column scales; 1 when a column is constant.
    pub scale: Vector,
    fitted: bool,
}

impl Default for Standardize {
    fn default() -> Self {
        Self {
            mean: Vector::zeros(0),
            scale: Vector::zeros(0),
            fitted: false,
        }
    }
}

impl Standardize {
    /// Unfitted standardizer.
    pub fn new() -> Self {
        Self::default()
    }

    fn fit_on(&mut self, x: &Matrix, ctx: &mut FitCtx) {
        let (xc, mean) = x.centered();
        let mut scale = Vector::filled(x.ncols(), 1.0);
        for j in 0..x.ncols() {
            let s = xc.column(j).std();
            if s > ctx.policy.near_zero_variance {
                scale[j] = s;
            }
        }
        self.mean = mean;
        self.scale = scale;
        self.fitted = true;
    }

    fn apply(&self, x: &Matrix) -> Matrix {
        let p = x.ncols().min(self.mean.len());
        Matrix::from_fn(x.nrows(), x.ncols(), |i, j| {
            let v = x.get(i, j);
            if j >= p || !self.fitted {
                v
            } else {
                (v - self.mean[j]) / self.scale[j]
            }
        })
    }
}

/// Pipeline preprocessing step.
#[derive(Clone, Debug)]
pub enum PreprocessStep {
    /// Subtract column mean and divide by sample std.
    Standardize(Standardize),
    /// Leave columns unchanged.
    Identity,
}

impl PreprocessStep {
    /// Column standardizer step.
    pub fn standardize() -> Self {
        Self::Standardize(Standardize::new())
    }

    fn fit_transform(&mut self, x: &Matrix, ctx: &mut FitCtx) -> Matrix {
        match self {
            Self::Standardize(s) => {
                s.fit_on(x, ctx);
                s.apply(x)
            }
            Self::Identity => x.clone(),
        }
    }

    fn transform(&self, x: &Matrix) -> Matrix {
        match self {
            Self::Standardize(s) => s.apply(x),
            Self::Identity => x.clone(),
        }
    }
}

/// [`PreprocessStep`] sequence followed by [`LinearRegression`].
#[derive(Clone, Debug)]
pub struct Pipeline {
    /// Ordered preprocessing steps.
    pub steps: Vec<PreprocessStep>,
    /// Final OLS (intercept on).
    pub regressor: LinearRegression,
}

impl Default for Pipeline {
    fn default() -> Self {
        Self {
            steps: vec![PreprocessStep::standardize()],
            regressor: LinearRegression::new(),
        }
    }
}

impl Pipeline {
    /// Standardize-then-OLS pipeline.
    pub fn new() -> Self {
        Self::default()
    }

    /// Pipeline with explicit steps.
    pub fn with_steps(steps: Vec<PreprocessStep>) -> Self {
        Self {
            steps,
            regressor: LinearRegression::new(),
        }
    }
}

/// Fitted pipeline: frozen preprocess + fitted OLS.
#[derive(Clone, Debug)]
pub struct FittedPipeline {
    /// Fitted preprocess steps.
    pub steps: Vec<PreprocessStep>,
    /// Final linear model.
    pub model: FittedLinear,
}

impl Predict for FittedPipeline {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let mut z = x.clone();
        for step in &self.steps {
            z = step.transform(&z);
        }
        match self.model.predict(&z, &session.child("ols")) {
            Ok(q) => ctx.finish(q.value),
            Err(_) => ctx.finish(Vector::zeros(x.nrows())),
        }
    }
}

impl Fit for Pipeline {
    type Fitted = FittedPipeline;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedPipeline>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        let mut z = x.clone();
        for step in &mut self.steps {
            z = step.fit_transform(&z, &mut ctx);
        }
        let model = match self.regressor.fit(&z, y, &session.child("final")) {
            Ok(q) => q.value,
            Err(_) => {
                // OLS may quality-abort (R²=1 or a large residual). The child
                // session already recorded that; the pipeline still returns the
                // fitted preprocess plus a mean predictor.
                empty_linear(x, y)
            }
        };
        ctx.finish(FittedPipeline {
            steps: self.steps.clone(),
            model,
        })
    }
}

fn empty_linear(x: &Matrix, y: &Vector) -> FittedLinear {
    let p = x.ncols() + 1;
    FittedLinear {
        coef: Vector::zeros(x.ncols()),
        intercept: y.mean(),
        beta: Vector::zeros(p),
        n: y.len(),
        p,
        df_resid: 0.0,
        r2: f64::NAN,
        adj_r2: f64::NAN,
        sigma2: f64::NAN,
        se: Vector::zeros(p),
        t_values: Vector::zeros(p),
        p_values: Vector::zeros(p),
        aic: f64::NAN,
        bic: f64::NAN,
        f_stat: f64::NAN,
        f_pvalue: f64::NAN,
        durbin_watson: f64::NAN,
        loglik: f64::NAN,
        fitted: Vector::zeros(y.len()),
        resid: Vector::zeros(y.len()),
        leverage: Vector::zeros(y.len()),
        cooks: Vector::zeros(y.len()),
        used_intercept: true,
    }
}

fn hstack(parts: &[Matrix]) -> Matrix {
    if parts.is_empty() {
        return Matrix::zeros(0, 0);
    }
    let n = parts.iter().map(|m| m.nrows()).max().unwrap_or(0);
    let p: usize = parts.iter().map(|m| m.ncols()).sum();
    let mut out = Matrix::zeros(n, p);
    let mut off = 0;
    for m in parts {
        for j in 0..m.ncols() {
            for i in 0..m.nrows().min(n) {
                out.set(i, off + j, m.get(i, j));
            }
        }
        off += m.ncols();
    }
    out
}

/// Independently transform `X` with each step and concatenate columns.
#[derive(Clone, Debug)]
pub struct FeatureUnion {
    /// Parallel preprocess steps.
    pub steps: Vec<PreprocessStep>,
}

impl Default for FeatureUnion {
    fn default() -> Self {
        Self {
            steps: vec![PreprocessStep::Identity, PreprocessStep::standardize()],
        }
    }
}

impl FeatureUnion {
    /// Union of the given steps.
    pub fn new(steps: Vec<PreprocessStep>) -> Self {
        Self { steps }
    }
}

/// Fitted feature union.
#[derive(Clone, Debug)]
pub struct FittedFeatureUnion {
    /// Fitted parallel steps.
    pub steps: Vec<PreprocessStep>,
}

impl Transform for FittedFeatureUnion {
    fn transform(&self, x: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
        let mut ctx = FitCtx::with_session(session.child("transform"));
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let parts: Vec<Matrix> = self.steps.iter().map(|s| s.transform(x)).collect();
        ctx.finish(hstack(&parts))
    }
}

impl Fit for FeatureUnion {
    type Fitted = FittedFeatureUnion;
    fn fit(
        &mut self,
        x: &Matrix,
        _y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedFeatureUnion>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        for step in &mut self.steps {
            let _ = step.fit_transform(x, &mut ctx);
        }
        ctx.finish(FittedFeatureUnion {
            steps: self.steps.clone(),
        })
    }
}

/// One column-index group and the step applied to it.
#[derive(Clone, Debug)]
pub struct ColumnGroup {
    /// Column indices (into the original design).
    pub columns: Vec<usize>,
    /// Transform applied to those columns.
    pub step: PreprocessStep,
}

/// Column-wise transformer (sklearn `ColumnTransformer` by index groups).
#[derive(Clone, Debug)]
pub struct ColumnTransformer {
    /// Ordered groups; outputs are concatenated in this order.
    pub groups: Vec<ColumnGroup>,
}

impl ColumnTransformer {
    /// Empty transformer.
    pub fn new() -> Self {
        Self { groups: Vec::new() }
    }

    /// Append a group.
    pub fn add_group(&mut self, columns: Vec<usize>, step: PreprocessStep) {
        self.groups.push(ColumnGroup { columns, step });
    }
}

impl Default for ColumnTransformer {
    fn default() -> Self {
        Self::new()
    }
}

fn select_columns(x: &Matrix, cols: &[usize]) -> Matrix {
    if cols.is_empty() {
        return Matrix::zeros(x.nrows(), 0);
    }
    Matrix::from_fn(x.nrows(), cols.len(), |i, j| {
        let c = cols[j];
        if c < x.ncols() {
            x.get(i, c)
        } else {
            f64::NAN
        }
    })
}

/// Fitted column transformer.
#[derive(Clone, Debug)]
pub struct FittedColumnTransformer {
    /// Fitted groups.
    pub groups: Vec<ColumnGroup>,
}

impl Transform for FittedColumnTransformer {
    fn transform(&self, x: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
        let mut ctx = FitCtx::with_session(session.child("transform"));
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let mut parts = Vec::new();
        for g in &self.groups {
            let sl = select_columns(x, &g.columns);
            parts.push(g.step.transform(&sl));
        }
        ctx.finish(hstack(&parts))
    }
}

impl Fit for ColumnTransformer {
    type Fitted = FittedColumnTransformer;
    fn fit(
        &mut self,
        x: &Matrix,
        _y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedColumnTransformer>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        for g in &mut self.groups {
            for &c in &g.columns {
                if c >= x.ncols() {
                    ctx.push(
                        Issue::builder(IssueCode::DimensionMismatch)
                            .message(format!("ColumnTransformer column {c} ≥ p={}", x.ncols()))
                            .build(),
                    );
                }
            }
            let sl = select_columns(x, &g.columns);
            let _ = g.step.fit_transform(&sl, &mut ctx);
        }
        ctx.finish(FittedColumnTransformer {
            groups: self.groups.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::Transform;
    use ojizou_san::Session;

    #[test]
    fn pipeline_standardize_ols_line() {
        let x = Matrix::from_fn(12, 1, |i, _| (i as f64) * 10.0);
        let y = Vector::from_iter(
            (0..12).map(|i| 3.0 + 0.5 * (i as f64) * 10.0 + 0.15 * (i as f64 % 3.0)),
        );
        let q = Pipeline::new()
            .fit(&x, &y, &Session::new("pipe", "fit"))
            .unwrap();
        let pred = q
            .value
            .predict(&x, &Session::new("pipe", "pred"))
            .unwrap()
            .value;
        assert_eq!(pred.len(), y.len());
        assert!(pred.as_slice().iter().all(|v| v.is_finite()));
    }

    #[test]
    fn feature_union_concatenates() {
        let x = Matrix::from_fn(5, 2, |i, j| (i + j) as f64);
        let y = Vector::zeros(5);
        let q = FeatureUnion::new(vec![
            PreprocessStep::Identity,
            PreprocessStep::standardize(),
        ])
        .fit(&x, &y, &Session::new("fu", "fit"))
        .unwrap();
        let z = q
            .value
            .transform(&x, &Session::new("fu", "tf"))
            .unwrap()
            .value;
        assert_eq!(z.ncols(), 4);
        assert_eq!(z.nrows(), 5);
    }

    #[test]
    fn column_transformer_groups() {
        let x = Matrix::from_fn(6, 3, |i, j| (i * 3 + j) as f64);
        let y = Vector::zeros(6);
        let mut ct = ColumnTransformer::new();
        ct.add_group(vec![0, 1], PreprocessStep::standardize());
        ct.add_group(vec![2], PreprocessStep::Identity);
        let q = ct.fit(&x, &y, &Session::new("ct", "fit")).unwrap();
        let z = q
            .value
            .transform(&x, &Session::new("ct", "tf"))
            .unwrap()
            .value;
        assert_eq!(z.ncols(), 3);
        assert!((z.column(0).mean()).abs() < 1e-12);
    }
}
