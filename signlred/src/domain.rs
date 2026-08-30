//! Scientific domain of a quality issue.

use core::fmt;

/// Which part of the statistical / numerical stack produced the issue.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Domain {
    /// Dense or sparse linear algebra (condition, rank, definiteness).
    LinearAlgebra,
    /// Point estimates, intervals, tests, identification.
    StatisticalInference,
    /// Iterative solvers, Newton, coordinate descent, backprop.
    Optimization,
    /// Incremental / streaming / partial_fit algorithms.
    OnlineLearning,
    /// Feature maps, encodings, imputations, expansions.
    FeatureEngineering,
    /// Temporal dependence, stationarity, seasonality, forecasts.
    TimeSeries,
    /// Likelihoods, HMMs, mixtures, Bayesian updates.
    ProbabilisticModel,
    /// Partition quality, empty clusters, degenerate kernels.
    Clustering,
    /// Embeddings, eigenvalues dropped, stress.
    DimensionalityReduction,
    /// Input dataset integrity.
    DataIntegrity,
    /// Model composition / pipelines / leakage.
    Composition,
}

impl Domain {
    /// Human label used in logs.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::LinearAlgebra => "linear_algebra",
            Self::StatisticalInference => "statistical_inference",
            Self::Optimization => "optimization",
            Self::OnlineLearning => "online_learning",
            Self::FeatureEngineering => "feature_engineering",
            Self::TimeSeries => "time_series",
            Self::ProbabilisticModel => "probabilistic_model",
            Self::Clustering => "clustering",
            Self::DimensionalityReduction => "dimensionality_reduction",
            Self::DataIntegrity => "data_integrity",
            Self::Composition => "composition",
        }
    }
}

impl fmt::Display for Domain {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}
