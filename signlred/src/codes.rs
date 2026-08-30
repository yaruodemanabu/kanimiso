//! Machine-readable quality issue codes.

use crate::domain::Domain;
use crate::severity::Severity;
use core::fmt;

/// Stable identifier for a quality failure or warning.
///
/// Codes are the API. Messages may be localized or expanded; codes must remain
/// comparable across versions so callers can branch on them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IssueCode {
    // --- data integrity ---
    /// Design matrix or series has a zero dimension.
    EmptyMatrix,
    /// Two operands disagree on a required axis length.
    DimensionMismatch,
    /// NaN or ±∞ in the input that the algorithm cannot absorb.
    NonFiniteInput,
    /// NaN or ±∞ produced by the algorithm.
    NonFiniteOutput,
    /// A required response / label vector is missing.
    MissingTarget,
    /// A required design matrix is missing.
    MissingFeatures,
    /// Entire column is missing / imputed-from-nothing.
    AllMissing,
    /// Duplicate row indices or timestamps.
    DuplicateIndex,
    /// Negative weights or frequencies.
    InvalidWeight,

    // --- linear algebra ---
    /// Exact singularity (rank 0 or a zero pivot at working precision).
    SingularMatrix,
    /// Numerically singular; a solve would be noise.
    NearSingular,
    /// Condition number in the warning band.
    IllConditioned,
    /// Detected rank strictly below the nominal dimension.
    RankDeficient,
    /// All singular values are ~0.
    RankZero,
    /// SPD factorization requested on a matrix that is not SPD.
    NonPositiveDefinite,
    /// Symmetric solve requested on an indefinite matrix.
    Indefinite,
    /// Underflow that flushed information to zero.
    NumericalUnderflow,
    /// Overflow; subsequent arithmetic is meaningless.
    NumericalOverflow,
    /// Residual of a linear solve exceeds the policy tolerance.
    ResidualTooLarge,
    /// Orthogonal factor lost orthogonality (iterative / TSQR).
    LossOfOrthogonality,
    /// A pivot was replaced or dropped.
    PivotTooSmall,
    /// Cholesky refused the matrix.
    CholeskyFailed,
    /// SVD iteration failed to converge.
    SvdDidNotConverge,
    /// Eigensolver failed to converge.
    EigenDidNotConverge,
    /// Algorithm substituted a Moore–Penrose / truncated inverse.
    PseudoinverseUsed,
    /// Algorithm added ridge / jitter that the caller did not request.
    RidgeFallbackUsed,
    /// Components were truncated to the numerical rank.
    TruncatedSvdUsed,
    /// A tiny diagonal jitter was added to restore definiteness.
    JitterInjected,
    /// Least-squares problem is underdetermined without extra constraints.
    UnderdeterminedSystem,
    /// Least-squares problem is inconsistent; only a minimizer exists.
    InconsistentSystem,

    // --- statistical inference ---
    /// n is too small for the claimed model.
    InsufficientSample,
    /// n ≤ p (or n ≤ p+1 when an intercept is present) for an unregularized model.
    SampleSmallerThanFeatures,
    /// Residual degrees of freedom ≤ 0.
    DegreesOfFreedomNonPositive,
    /// Target has zero variance.
    ConstantTarget,
    /// A feature has zero variance.
    ConstantFeature,
    /// A feature is numerically constant.
    NearZeroVariance,
    /// Linear dependence among columns is exact.
    PerfectCollinearity,
    /// VIF / condition indicates unstable partial effects.
    HighMulticollinearity,
    /// Logistic / discrete model separated perfectly.
    PerfectSeparation,
    /// Almost-separated discrete model; MLE diverges.
    QuasiCompleteSeparation,
    /// A class required by the estimator is empty.
    EmptyClass,
    /// Only one class present; a classifier is a constant.
    SingleClass,
    /// Class counts so unbalanced that default metrics lie.
    ClassImbalanceSevere,
    /// Likelihood / density collapsed to a point mass.
    DegenerateDistribution,
    /// Likelihood is identically zero (or −∞).
    ZeroLikelihood,
    /// Prior is improper and the posterior is not integrable.
    ImproperPrior,
    /// Residual diagnostics reject the homoscedastic-Gaussian story.
    Heteroscedasticity,
    /// Residual autocorrelation violates the i.i.d. assumption.
    AutocorrelatedResiduals,
    /// Residual normality is untenable for the claimed interval.
    NonNormalResiduals,
    /// High-leverage rows dominate the fit.
    LeveragePoint,
    /// Cook / DFFITS says one row owns the model.
    InfluentialPoint,
    /// Outliers dominate the loss.
    OutlierDominated,
    /// R² is 1 because the model interpolated or the target is constant.
    R2IsOne,
    /// R² is 0; the model is the null model.
    R2IsZero,
    /// Out-of-sample or uncentered R² is negative.
    R2Negative,
    /// Observed information / Hessian is singular.
    InformationMatrixSingular,
    /// Newton step is not a descent direction.
    HessianNotPositiveDefinite,
    /// Parameters are not identified from the observed information.
    UnidentifiedModel,
    /// More free parameters than the data can support.
    Overparameterized,
    /// p-values are reported without a valid null / regularity.
    PValueUnreliable,
    /// Interval collapsed to a point or exploded to ℝ.
    ConfidenceIntervalDegenerate,
    /// Multiple comparisons without correction.
    MultipleTestingUncorrected,
    /// Fit succeeded algebraically but has no interpretive content.
    MeaninglessFit,
    /// The fitted predictor is a constant (null model after collapse).
    PredictionsAreConstant,
    /// Only the intercept survived after collapse / selection.
    InterceptOnlyCollapse,
    /// Labels appear independent of features (e.g. shuffled / null).
    FeatureTargetIndependence,
    /// Claimed causal / effect interpretation is not identified.
    CausalClaimUnidentified,

    // --- optimization ---
    /// Stopped without satisfying the convergence test.
    DidNotConverge,
    /// Hit the iteration cap.
    MaxIterReached,
    /// Line search could not find an acceptable step.
    LineSearchFailed,
    /// Step length underflowed.
    StepSizeCollapsed,
    /// Gradient / update exploded.
    GradientExploded,
    /// Objective became non-finite.
    LossIsNan,
    /// Curvature suggests a saddle, not a minimizer.
    SaddlePointSuspected,
    /// Solution is sensitive to initialization.
    LocalMinimumUnstable,
    /// Step size / learning rate is too large for the curvature.
    LearningRateTooLarge,
    /// Progress is numerically zero because the rate is tiny.
    LearningRateTooSmall,

    // --- online / incremental ---
    /// `partial_fit` was called on a stale or deserialized-without-state model.
    StaleState,
    /// Effective sample size (after forgetting) is too small.
    InsufficientEffectiveSample,
    /// Drift detector fired.
    ConceptDriftDetected,
    /// Input distribution drifted without a label-conditional change.
    VirtualDriftDetected,
    /// Forgetting factor erased earlier identification.
    ForgettingErasedIdentification,
    /// Batch carried ~0 Fisher information / residual reduction.
    UpdateWithZeroInformation,
    /// Online update is not identified (e.g. new one-hot level only).
    IncrementalUnidentifiable,
    /// Parameter jump larger than the policy allows.
    ParameterJumpAnomalous,
    /// Sliding window is shorter than the model order.
    WindowTooShort,
    /// Estimator is still in warm-up; predictions are not inferential.
    WarmupIncomplete,
    /// New data overwrote earlier structure without a drift flag.
    CatastrophicForgetting,
    /// `partial_fit` called before the first initializing fit.
    PartialFitBeforeInit,
    /// Label / class space grew online; earlier probabilities are incomparable.
    LabelSpaceExpandedOnline,
    /// Feature space grew or shrank online.
    FeatureSpaceChangedOnline,

    // --- time series ---
    /// Unit-root / non-stationarity makes the claimed estimator inconsistent.
    NonStationary,
    /// Seasonal unit root / insufficient seasonal cycles.
    SeasonalUnitRoot,
    /// Documented or detected structural break inside the estimation window.
    StructuralBreak,
    /// Too few complete seasonal cycles to identify seasonality.
    InsufficientSeasonalCycles,
    /// Sampling frequency does not match the claimed seasonal period.
    FrequencyMismatch,
    /// Horizon is longer than what the identified dynamics can support.
    ForecastHorizonExceedsIdentifiability,
    /// Log / multiplicative model on a non-positive series.
    NonPositiveSeries,
    /// Invertibility of an MA polynomial is violated.
    InvertibilityViolated,
    /// Causality of an AR polynomial is violated.
    CausalityViolated,
    /// Series too short for the claimed ARIMA order.
    ShortSeriesForArima,
    /// Spectral leakage / aliasing undermines a periodogram claim.
    SpectralLeakage,

    // --- clustering / DR ---
    /// A cluster contains no points after assignment.
    EmptyCluster,
    /// All points collapsed into one cluster.
    DegenerateClusters,
    /// A cluster is a single point; covariance is undefined.
    SinglePointCluster,
    /// Requested components exceed numerical rank.
    ComponentsExceedRank,
    /// Negative eigenvalues were dropped from a supposed Gram / Laplacian.
    NegativeEigenvalueDropped,
    /// Embedding stress / reconstruction error is unacceptable.
    EmbeddingUnstable,
    /// Kernel matrix is not positive semidefinite.
    KernelNotPd,

    // --- probabilistic / HMM ---
    /// Forward / backward probabilities underflowed to zero.
    ForwardUnderflow,
    /// Every scale factor is zero; the sequence has probability 0.
    ScaleFactorZero,
    /// Chain is absorbing; likelihood of long sequences is a delta.
    AbsorbingStateOnly,
    /// A state is unreachable from the start distribution.
    UnreachableState,
    /// Emission distribution has zero variance / empty support.
    EmissionDegenerate,
    /// Mixture weight is zero; component is unidentified.
    MixtureWeightCollapsed,

    // --- features / composition ---
    /// One-hot without dropping a level in a model with intercept.
    OneHotFullRankViolation,
    /// Polynomial / interaction expansion exploded dimension vs n.
    PolynomialExplosion,
    /// Transformer / target used information from the test fold.
    TargetLeakageSuspected,
    /// Imputer filled a column using a statistic that was itself undefined.
    ImputationUndefined,
    /// Pipeline step order makes a later step's assumptions false.
    PipelineAssumptionBroken,
}

impl IssueCode {
    /// Default scientific domain for this code.
    pub const fn default_domain(self) -> Domain {
        use IssueCode::*;
        match self {
            EmptyMatrix | DimensionMismatch | NonFiniteInput | NonFiniteOutput | MissingTarget
            | MissingFeatures | AllMissing | DuplicateIndex | InvalidWeight => {
                Domain::DataIntegrity
            }
            SingularMatrix
            | NearSingular
            | IllConditioned
            | RankDeficient
            | RankZero
            | NonPositiveDefinite
            | Indefinite
            | NumericalUnderflow
            | NumericalOverflow
            | ResidualTooLarge
            | LossOfOrthogonality
            | PivotTooSmall
            | CholeskyFailed
            | SvdDidNotConverge
            | EigenDidNotConverge
            | PseudoinverseUsed
            | RidgeFallbackUsed
            | TruncatedSvdUsed
            | JitterInjected
            | UnderdeterminedSystem
            | InconsistentSystem => Domain::LinearAlgebra,
            InsufficientSample
            | SampleSmallerThanFeatures
            | DegreesOfFreedomNonPositive
            | ConstantTarget
            | ConstantFeature
            | NearZeroVariance
            | PerfectCollinearity
            | HighMulticollinearity
            | PerfectSeparation
            | QuasiCompleteSeparation
            | EmptyClass
            | SingleClass
            | ClassImbalanceSevere
            | DegenerateDistribution
            | ZeroLikelihood
            | ImproperPrior
            | Heteroscedasticity
            | AutocorrelatedResiduals
            | NonNormalResiduals
            | LeveragePoint
            | InfluentialPoint
            | OutlierDominated
            | R2IsOne
            | R2IsZero
            | R2Negative
            | InformationMatrixSingular
            | HessianNotPositiveDefinite
            | UnidentifiedModel
            | Overparameterized
            | PValueUnreliable
            | ConfidenceIntervalDegenerate
            | MultipleTestingUncorrected
            | MeaninglessFit
            | PredictionsAreConstant
            | InterceptOnlyCollapse
            | FeatureTargetIndependence
            | CausalClaimUnidentified => Domain::StatisticalInference,
            DidNotConverge | MaxIterReached | LineSearchFailed | StepSizeCollapsed
            | GradientExploded | LossIsNan | SaddlePointSuspected | LocalMinimumUnstable
            | LearningRateTooLarge | LearningRateTooSmall => Domain::Optimization,
            StaleState
            | InsufficientEffectiveSample
            | ConceptDriftDetected
            | VirtualDriftDetected
            | ForgettingErasedIdentification
            | UpdateWithZeroInformation
            | IncrementalUnidentifiable
            | ParameterJumpAnomalous
            | WindowTooShort
            | WarmupIncomplete
            | CatastrophicForgetting
            | PartialFitBeforeInit
            | LabelSpaceExpandedOnline
            | FeatureSpaceChangedOnline => Domain::OnlineLearning,
            NonStationary
            | SeasonalUnitRoot
            | StructuralBreak
            | InsufficientSeasonalCycles
            | FrequencyMismatch
            | ForecastHorizonExceedsIdentifiability
            | NonPositiveSeries
            | InvertibilityViolated
            | CausalityViolated
            | ShortSeriesForArima
            | SpectralLeakage => Domain::TimeSeries,
            EmptyCluster | DegenerateClusters | SinglePointCluster => Domain::Clustering,
            ComponentsExceedRank | NegativeEigenvalueDropped | EmbeddingUnstable | KernelNotPd => {
                Domain::DimensionalityReduction
            }
            ForwardUnderflow
            | ScaleFactorZero
            | AbsorbingStateOnly
            | UnreachableState
            | EmissionDegenerate
            | MixtureWeightCollapsed => Domain::ProbabilisticModel,
            OneHotFullRankViolation
            | PolynomialExplosion
            | TargetLeakageSuspected
            | ImputationUndefined
            | PipelineAssumptionBroken => Domain::FeatureEngineering,
        }
    }

    /// Default severity before [`crate::Policy`] rewrites it.
    pub const fn default_severity(self) -> Severity {
        use IssueCode::*;
        match self {
            EmptyMatrix | DimensionMismatch | NonFiniteInput | MissingTarget | MissingFeatures
            | InvalidWeight | RankZero | CholeskyFailed | SvdDidNotConverge
            | EigenDidNotConverge | LossIsNan | PartialFitBeforeInit | ScaleFactorZero => {
                Severity::Fatal
            }
            NonFiniteOutput
            | SingularMatrix
            | NearSingular
            | NonPositiveDefinite
            | NumericalOverflow
            | ResidualTooLarge
            | InsufficientSample
            | SampleSmallerThanFeatures
            | DegreesOfFreedomNonPositive
            | ConstantTarget
            | PerfectCollinearity
            | PerfectSeparation
            | EmptyClass
            | SingleClass
            | ZeroLikelihood
            | InformationMatrixSingular
            | UnidentifiedModel
            | MeaninglessFit
            | PredictionsAreConstant
            | GradientExploded
            | StaleState
            | IncrementalUnidentifiable
            | FeatureSpaceChangedOnline
            | NonPositiveSeries
            | ShortSeriesForArima
            | DegenerateClusters
            | EmissionDegenerate
            | ImputationUndefined
            | PipelineAssumptionBroken => Severity::Error,
            IllConditioned
            | RankDeficient
            | Indefinite
            | NumericalUnderflow
            | PivotTooSmall
            | LossOfOrthogonality
            | PseudoinverseUsed
            | RidgeFallbackUsed
            | TruncatedSvdUsed
            | JitterInjected
            | UnderdeterminedSystem
            | InconsistentSystem
            | ConstantFeature
            | NearZeroVariance
            | HighMulticollinearity
            | QuasiCompleteSeparation
            | ClassImbalanceSevere
            | DegenerateDistribution
            | ImproperPrior
            | Heteroscedasticity
            | AutocorrelatedResiduals
            | NonNormalResiduals
            | LeveragePoint
            | InfluentialPoint
            | OutlierDominated
            | R2IsOne
            | R2IsZero
            | R2Negative
            | HessianNotPositiveDefinite
            | Overparameterized
            | PValueUnreliable
            | ConfidenceIntervalDegenerate
            | InterceptOnlyCollapse
            | FeatureTargetIndependence
            | CausalClaimUnidentified
            | DidNotConverge
            | MaxIterReached
            | LineSearchFailed
            | StepSizeCollapsed
            | SaddlePointSuspected
            | LocalMinimumUnstable
            | LearningRateTooLarge
            | InsufficientEffectiveSample
            | ConceptDriftDetected
            | VirtualDriftDetected
            | ForgettingErasedIdentification
            | UpdateWithZeroInformation
            | ParameterJumpAnomalous
            | WindowTooShort
            | WarmupIncomplete
            | CatastrophicForgetting
            | LabelSpaceExpandedOnline
            | NonStationary
            | SeasonalUnitRoot
            | StructuralBreak
            | InsufficientSeasonalCycles
            | FrequencyMismatch
            | ForecastHorizonExceedsIdentifiability
            | InvertibilityViolated
            | CausalityViolated
            | SpectralLeakage
            | EmptyCluster
            | SinglePointCluster
            | ComponentsExceedRank
            | NegativeEigenvalueDropped
            | EmbeddingUnstable
            | KernelNotPd
            | ForwardUnderflow
            | AbsorbingStateOnly
            | UnreachableState
            | MixtureWeightCollapsed
            | OneHotFullRankViolation
            | PolynomialExplosion
            | TargetLeakageSuspected
            | AllMissing
            | DuplicateIndex => Severity::Warning,
            LearningRateTooSmall | MultipleTestingUncorrected => Severity::Advisory,
        }
    }

    /// Default remediation shown to the caller.
    pub const fn default_remediation(self) -> &'static str {
        use IssueCode::*;
        match self {
            EmptyMatrix => "provide a matrix with positive rows and columns",
            DimensionMismatch => "align operand shapes before the linear algebra step",
            NonFiniteInput => "drop, impute, or repair NaN/Inf before fitting; do not silently skip",
            NonFiniteOutput => "the algorithm diverged; reduce step size, regularize, or rescale",
            MissingTarget => "pass the response / label vector required by this estimator",
            MissingFeatures => "pass a design matrix or feature frame",
            AllMissing => "do not impute a column that has zero observed values",
            DuplicateIndex => "deduplicate timestamps / row ids or aggregate them explicitly",
            InvalidWeight => "weights must be finite and non-negative",
            SingularMatrix => "remove dependent columns, add identified regularization, or use a reduced-rank model",
            NearSingular => "treat the solve as unidentified; regularize or drop columns",
            IllConditioned => "center/scale features, drop collinear columns, or use a regularized estimator",
            RankDeficient => "reduce the parameter dimension to the numerical rank",
            RankZero => "the matrix carries no information; stop",
            NonPositiveDefinite => "do not use a Cholesky / Mahalanobis interpretation on this matrix",
            Indefinite => "use an indefinite solver or a projected PSD reconstruction, and record the compromise",
            NumericalUnderflow => "rescale, use log-space, or a compensated summation",
            NumericalOverflow => "rescale features / targets; check the link function",
            ResidualTooLarge => "do not treat the solve as successful; inspect rank and scaling",
            LossOfOrthogonality => "reorthogonalize or switch to a stable factorization",
            PivotTooSmall => "a column was dropped or jittered; coefficients for that direction are unidentified",
            CholeskyFailed => "matrix is not SPD at working precision; use a fallback and record it",
            SvdDidNotConverge => "retry with a different algorithm or scale; do not use a partial SVD silently",
            EigenDidNotConverge => "do not use the returned eigenpairs",
            PseudoinverseUsed => "coefficients in the null space are unidentified; report the null dimension",
            RidgeFallbackUsed => "the extra ridge changes the estimand; do not market it as OLS",
            TruncatedSvdUsed => "dropped components are not estimated; downstream k must be reduced",
            JitterInjected => "definiteness was manufactured; eigenvalues near the jitter are artifacts",
            UnderdeterminedSystem => "infinite solutions exist; pick an identified criterion (min-norm, ridge) and say so",
            InconsistentSystem => "only a least-squares compromise exists; report the residual",
            InsufficientSample => "collect more observations or reduce model order",
            SampleSmallerThanFeatures => "unregularized p-parameter inference is unidentified when n ≤ p",
            DegreesOfFreedomNonPositive => "variance, σ², p-values, and AIC are undefined",
            ConstantTarget => "stop; there is nothing to explain",
            ConstantFeature => "drop the column; its coefficient is not identified",
            NearZeroVariance => "the column is numerically constant after scaling",
            PerfectCollinearity => "drop dependent columns; partial effects do not exist",
            HighMulticollinearity => "do not interpret individual coefficients; report a joint effect or regularize",
            PerfectSeparation => "MLE for logistic/probit diverges; use Firth / exact / regularized likelihood",
            QuasiCompleteSeparation => "finite MLE is an illusion of the iteration cap; regularize",
            EmptyClass => "cannot estimate a class-conditional distribution with zero counts",
            SingleClass => "the classifier is a constant; accuracy is not a skill score",
            ClassImbalanceSevere => "do not report raw accuracy as skill; use proper scores / prevalence-aware metrics",
            DegenerateDistribution => "moments / entropy / KL are undefined or infinite",
            ZeroLikelihood => "parameters are outside the support of the data",
            ImproperPrior => "posterior expectations are not defined",
            Heteroscedasticity => "OLS SEs are inconsistent; use WLS / robust / GLM variance",
            AutocorrelatedResiduals => "i.i.d. SEs and tests are invalid; use HAC / GLS / a time-series model",
            NonNormalResiduals => "Gaussian intervals and AIC comparisons are approximate at best",
            LeveragePoint => "the fit is owned by extreme x; report influence diagnostics",
            InfluentialPoint => "one row moves the estimate; show leave-one-out",
            OutlierDominated => "use a robust loss or disclose that the mean is a tail functional",
            R2IsOne => "interpolation or a constant target; this is not confirmatory skill",
            R2IsZero => "the model did not beat the mean; do not over-interpret coefficients",
            R2Negative => "the model lost to the mean; it is not a useful predictor",
            InformationMatrixSingular => "Wald / SE / p-values cannot be formed",
            HessianNotPositiveDefinite => "the critical point is not a local MLE",
            UnidentifiedModel => "many parameter values give the same predictive distribution",
            Overparameterized => "reduce order, add regularization, or collect data",
            PValueUnreliable => "do not report stars; regularity conditions failed",
            ConfidenceIntervalDegenerate => "do not publish the interval",
            MultipleTestingUncorrected => "control FWER or FDR before claiming discoveries",
            MeaninglessFit => "discard the numeric output for inferential use",
            PredictionsAreConstant => "the model learned no input-dependent relationship",
            InterceptOnlyCollapse => "all slopes vanished; you are publishing a mean",
            FeatureTargetIndependence => "features carry no detectable information about y",
            CausalClaimUnidentified => "the design does not identify the claimed effect",
            DidNotConverge => "do not treat the last iterate as the solution",
            MaxIterReached => "increase iterations or change the solver; disclose the cap",
            LineSearchFailed => "step was rejected; the update is not a descent step",
            StepSizeCollapsed => "no progress is possible at working precision",
            GradientExploded => "rescale, clip, or reduce the learning rate",
            LossIsNan => "abort; subsequent iterates are garbage",
            SaddlePointSuspected => "perturb and retry; do not claim a minimum",
            LocalMinimumUnstable => "report multiple random restarts",
            LearningRateTooLarge => "decrease the rate; the discrete gradient step is unstable",
            LearningRateTooSmall => "increase the rate or you will stop on a plateau",
            StaleState => "re-initialize the online estimator; the sufficient statistics are invalid",
            InsufficientEffectiveSample => "the forgetting factor left too little mass for identification",
            ConceptDriftDetected => "reset, decay, or switch models; old parameters no longer describe the stream",
            VirtualDriftDetected => "P(X) changed; calibrate or adapt the preprocessor",
            ForgettingErasedIdentification => "raise λ or widen the window; the model is unidentified again",
            UpdateWithZeroInformation => "this partial_fit did not change the estimand; say so",
            IncrementalUnidentifiable => "the new batch does not identify the new parameters",
            ParameterJumpAnomalous => "inspect the batch for contamination or a true break",
            WindowTooShort => "lengthen the window past the model order",
            WarmupIncomplete => "do not serve predictions as if the model were fitted",
            CatastrophicForgetting => "the update overwrote earlier structure without a drift declaration",
            PartialFitBeforeInit => "call fit or an initializing partial_fit first",
            LabelSpaceExpandedOnline => "old probability vectors are not comparable to new ones",
            FeatureSpaceChangedOnline => "coefficients from the previous space are not the same estimand",
            NonStationary => "difference, detrend, or use a unit-root-consistent procedure",
            SeasonalUnitRoot => "apply seasonal differences or a seasonal cointegration model",
            StructuralBreak => "split the sample or use a break-robust estimator",
            InsufficientSeasonalCycles => "do not estimate a seasonal pattern from < 2 cycles",
            FrequencyMismatch => "resample or change the seasonal period",
            ForecastHorizonExceedsIdentifiability => "shorten the horizon or state that forecasts are extrapolations of a weak identification",
            NonPositiveSeries => "do not take logs / multiplicative seasonality on non-positive data",
            InvertibilityViolated => "the MA representation is not invertible; impulse responses are not identified that way",
            CausalityViolated => "the AR polynomial has roots inside the unit circle; the process is not causal",
            ShortSeriesForArima => "reduce (p,d,q) or collect a longer series",
            SpectralLeakage => "taper / change window length before claiming a peak",
            EmptyCluster => "re-seed or reduce k; empty clusters make centroids undefined",
            DegenerateClusters => "k is not identified; the partition is a single blob",
            SinglePointCluster => "within-cluster covariance / silhouette for that cluster is undefined",
            ComponentsExceedRank => "set n_components ≤ numerical rank",
            NegativeEigenvalueDropped => "the similarity / covariance is not PSD; dropped directions are artifacts",
            EmbeddingUnstable => "do not interpret distances in the embedding",
            KernelNotPd => "project to PSD or change the kernel; SVM / GP math does not apply",
            ForwardUnderflow => "use scaled forward-backward; unscaled probabilities are zero",
            ScaleFactorZero => "the observation sequence is impossible under the current HMM",
            AbsorbingStateOnly => "long-run occupancy is a delta; dynamics are not identified",
            UnreachableState => "that state's emission parameters are unidentified",
            EmissionDegenerate => "emission covariance / support collapsed; decoding is a hard assignment",
            MixtureWeightCollapsed => "the component is gone; its parameters must not be interpreted",
            OneHotFullRankViolation => "drop a reference level when an intercept is present",
            PolynomialExplosion => "n ≪ expanded p; the map is interpolation, not a feature",
            TargetLeakageSuspected => "refit the transformer inside each training fold",
            ImputationUndefined => "the fill value is not a statistic of observed data",
            PipelineAssumptionBroken => "reorder steps or change the estimator; documented assumptions are false",
        }
    }

    /// Snake-case name, stable for logs and serialization.
    pub const fn as_str(self) -> &'static str {
        use IssueCode::*;
        match self {
            EmptyMatrix => "empty_matrix",
            DimensionMismatch => "dimension_mismatch",
            NonFiniteInput => "non_finite_input",
            NonFiniteOutput => "non_finite_output",
            MissingTarget => "missing_target",
            MissingFeatures => "missing_features",
            AllMissing => "all_missing",
            DuplicateIndex => "duplicate_index",
            InvalidWeight => "invalid_weight",
            SingularMatrix => "singular_matrix",
            NearSingular => "near_singular",
            IllConditioned => "ill_conditioned",
            RankDeficient => "rank_deficient",
            RankZero => "rank_zero",
            NonPositiveDefinite => "non_positive_definite",
            Indefinite => "indefinite",
            NumericalUnderflow => "numerical_underflow",
            NumericalOverflow => "numerical_overflow",
            ResidualTooLarge => "residual_too_large",
            LossOfOrthogonality => "loss_of_orthogonality",
            PivotTooSmall => "pivot_too_small",
            CholeskyFailed => "cholesky_failed",
            SvdDidNotConverge => "svd_did_not_converge",
            EigenDidNotConverge => "eigen_did_not_converge",
            PseudoinverseUsed => "pseudoinverse_used",
            RidgeFallbackUsed => "ridge_fallback_used",
            TruncatedSvdUsed => "truncated_svd_used",
            JitterInjected => "jitter_injected",
            UnderdeterminedSystem => "underdetermined_system",
            InconsistentSystem => "inconsistent_system",
            InsufficientSample => "insufficient_sample",
            SampleSmallerThanFeatures => "sample_smaller_than_features",
            DegreesOfFreedomNonPositive => "degrees_of_freedom_non_positive",
            ConstantTarget => "constant_target",
            ConstantFeature => "constant_feature",
            NearZeroVariance => "near_zero_variance",
            PerfectCollinearity => "perfect_collinearity",
            HighMulticollinearity => "high_multicollinearity",
            PerfectSeparation => "perfect_separation",
            QuasiCompleteSeparation => "quasi_complete_separation",
            EmptyClass => "empty_class",
            SingleClass => "single_class",
            ClassImbalanceSevere => "class_imbalance_severe",
            DegenerateDistribution => "degenerate_distribution",
            ZeroLikelihood => "zero_likelihood",
            ImproperPrior => "improper_prior",
            Heteroscedasticity => "heteroscedasticity",
            AutocorrelatedResiduals => "autocorrelated_residuals",
            NonNormalResiduals => "non_normal_residuals",
            LeveragePoint => "leverage_point",
            InfluentialPoint => "influential_point",
            OutlierDominated => "outlier_dominated",
            R2IsOne => "r2_is_one",
            R2IsZero => "r2_is_zero",
            R2Negative => "r2_negative",
            InformationMatrixSingular => "information_matrix_singular",
            HessianNotPositiveDefinite => "hessian_not_positive_definite",
            UnidentifiedModel => "unidentified_model",
            Overparameterized => "overparameterized",
            PValueUnreliable => "p_value_unreliable",
            ConfidenceIntervalDegenerate => "confidence_interval_degenerate",
            MultipleTestingUncorrected => "multiple_testing_uncorrected",
            MeaninglessFit => "meaningless_fit",
            PredictionsAreConstant => "predictions_are_constant",
            InterceptOnlyCollapse => "intercept_only_collapse",
            FeatureTargetIndependence => "feature_target_independence",
            CausalClaimUnidentified => "causal_claim_unidentified",
            DidNotConverge => "did_not_converge",
            MaxIterReached => "max_iter_reached",
            LineSearchFailed => "line_search_failed",
            StepSizeCollapsed => "step_size_collapsed",
            GradientExploded => "gradient_exploded",
            LossIsNan => "loss_is_nan",
            SaddlePointSuspected => "saddle_point_suspected",
            LocalMinimumUnstable => "local_minimum_unstable",
            LearningRateTooLarge => "learning_rate_too_large",
            LearningRateTooSmall => "learning_rate_too_small",
            StaleState => "stale_state",
            InsufficientEffectiveSample => "insufficient_effective_sample",
            ConceptDriftDetected => "concept_drift_detected",
            VirtualDriftDetected => "virtual_drift_detected",
            ForgettingErasedIdentification => "forgetting_erased_identification",
            UpdateWithZeroInformation => "update_with_zero_information",
            IncrementalUnidentifiable => "incremental_unidentifiable",
            ParameterJumpAnomalous => "parameter_jump_anomalous",
            WindowTooShort => "window_too_short",
            WarmupIncomplete => "warmup_incomplete",
            CatastrophicForgetting => "catastrophic_forgetting",
            PartialFitBeforeInit => "partial_fit_before_init",
            LabelSpaceExpandedOnline => "label_space_expanded_online",
            FeatureSpaceChangedOnline => "feature_space_changed_online",
            NonStationary => "non_stationary",
            SeasonalUnitRoot => "seasonal_unit_root",
            StructuralBreak => "structural_break",
            InsufficientSeasonalCycles => "insufficient_seasonal_cycles",
            FrequencyMismatch => "frequency_mismatch",
            ForecastHorizonExceedsIdentifiability => "forecast_horizon_exceeds_identifiability",
            NonPositiveSeries => "non_positive_series",
            InvertibilityViolated => "invertibility_violated",
            CausalityViolated => "causality_violated",
            ShortSeriesForArima => "short_series_for_arima",
            SpectralLeakage => "spectral_leakage",
            EmptyCluster => "empty_cluster",
            DegenerateClusters => "degenerate_clusters",
            SinglePointCluster => "single_point_cluster",
            ComponentsExceedRank => "components_exceed_rank",
            NegativeEigenvalueDropped => "negative_eigenvalue_dropped",
            EmbeddingUnstable => "embedding_unstable",
            KernelNotPd => "kernel_not_pd",
            ForwardUnderflow => "forward_underflow",
            ScaleFactorZero => "scale_factor_zero",
            AbsorbingStateOnly => "absorbing_state_only",
            UnreachableState => "unreachable_state",
            EmissionDegenerate => "emission_degenerate",
            MixtureWeightCollapsed => "mixture_weight_collapsed",
            OneHotFullRankViolation => "one_hot_full_rank_violation",
            PolynomialExplosion => "polynomial_explosion",
            TargetLeakageSuspected => "target_leakage_suspected",
            ImputationUndefined => "imputation_undefined",
            PipelineAssumptionBroken => "pipeline_assumption_broken",
        }
    }

    /// Every known code. Used by exhaustive tests and coverage ledgers.
    pub const ALL: &'static [IssueCode] = &[
        Self::EmptyMatrix,
        Self::DimensionMismatch,
        Self::NonFiniteInput,
        Self::NonFiniteOutput,
        Self::MissingTarget,
        Self::MissingFeatures,
        Self::AllMissing,
        Self::DuplicateIndex,
        Self::InvalidWeight,
        Self::SingularMatrix,
        Self::NearSingular,
        Self::IllConditioned,
        Self::RankDeficient,
        Self::RankZero,
        Self::NonPositiveDefinite,
        Self::Indefinite,
        Self::NumericalUnderflow,
        Self::NumericalOverflow,
        Self::ResidualTooLarge,
        Self::LossOfOrthogonality,
        Self::PivotTooSmall,
        Self::CholeskyFailed,
        Self::SvdDidNotConverge,
        Self::EigenDidNotConverge,
        Self::PseudoinverseUsed,
        Self::RidgeFallbackUsed,
        Self::TruncatedSvdUsed,
        Self::JitterInjected,
        Self::UnderdeterminedSystem,
        Self::InconsistentSystem,
        Self::InsufficientSample,
        Self::SampleSmallerThanFeatures,
        Self::DegreesOfFreedomNonPositive,
        Self::ConstantTarget,
        Self::ConstantFeature,
        Self::NearZeroVariance,
        Self::PerfectCollinearity,
        Self::HighMulticollinearity,
        Self::PerfectSeparation,
        Self::QuasiCompleteSeparation,
        Self::EmptyClass,
        Self::SingleClass,
        Self::ClassImbalanceSevere,
        Self::DegenerateDistribution,
        Self::ZeroLikelihood,
        Self::ImproperPrior,
        Self::Heteroscedasticity,
        Self::AutocorrelatedResiduals,
        Self::NonNormalResiduals,
        Self::LeveragePoint,
        Self::InfluentialPoint,
        Self::OutlierDominated,
        Self::R2IsOne,
        Self::R2IsZero,
        Self::R2Negative,
        Self::InformationMatrixSingular,
        Self::HessianNotPositiveDefinite,
        Self::UnidentifiedModel,
        Self::Overparameterized,
        Self::PValueUnreliable,
        Self::ConfidenceIntervalDegenerate,
        Self::MultipleTestingUncorrected,
        Self::MeaninglessFit,
        Self::PredictionsAreConstant,
        Self::InterceptOnlyCollapse,
        Self::FeatureTargetIndependence,
        Self::CausalClaimUnidentified,
        Self::DidNotConverge,
        Self::MaxIterReached,
        Self::LineSearchFailed,
        Self::StepSizeCollapsed,
        Self::GradientExploded,
        Self::LossIsNan,
        Self::SaddlePointSuspected,
        Self::LocalMinimumUnstable,
        Self::LearningRateTooLarge,
        Self::LearningRateTooSmall,
        Self::StaleState,
        Self::InsufficientEffectiveSample,
        Self::ConceptDriftDetected,
        Self::VirtualDriftDetected,
        Self::ForgettingErasedIdentification,
        Self::UpdateWithZeroInformation,
        Self::IncrementalUnidentifiable,
        Self::ParameterJumpAnomalous,
        Self::WindowTooShort,
        Self::WarmupIncomplete,
        Self::CatastrophicForgetting,
        Self::PartialFitBeforeInit,
        Self::LabelSpaceExpandedOnline,
        Self::FeatureSpaceChangedOnline,
        Self::NonStationary,
        Self::SeasonalUnitRoot,
        Self::StructuralBreak,
        Self::InsufficientSeasonalCycles,
        Self::FrequencyMismatch,
        Self::ForecastHorizonExceedsIdentifiability,
        Self::NonPositiveSeries,
        Self::InvertibilityViolated,
        Self::CausalityViolated,
        Self::ShortSeriesForArima,
        Self::SpectralLeakage,
        Self::EmptyCluster,
        Self::DegenerateClusters,
        Self::SinglePointCluster,
        Self::ComponentsExceedRank,
        Self::NegativeEigenvalueDropped,
        Self::EmbeddingUnstable,
        Self::KernelNotPd,
        Self::ForwardUnderflow,
        Self::ScaleFactorZero,
        Self::AbsorbingStateOnly,
        Self::UnreachableState,
        Self::EmissionDegenerate,
        Self::MixtureWeightCollapsed,
        Self::OneHotFullRankViolation,
        Self::PolynomialExplosion,
        Self::TargetLeakageSuspected,
        Self::ImputationUndefined,
        Self::PipelineAssumptionBroken,
    ];
}

impl fmt::Display for IssueCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_codes_have_unique_slugs() {
        let mut slugs: Vec<_> = IssueCode::ALL.iter().map(|c| c.as_str()).collect();
        slugs.sort_unstable();
        let before = slugs.len();
        slugs.dedup();
        assert_eq!(before, slugs.len());
    }
}
