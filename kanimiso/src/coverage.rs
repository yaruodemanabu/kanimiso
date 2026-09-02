//! Verification ledger for the active v0.2 surface.
//!
//! [`inventory`] is intentionally not a graveyard of historical names.  It
//! lists the algorithms that the reconstitution currently claims or actively
//! tracks.  The generated v0.1 ledger is preserved in
//! `generated-v0.1-archive/coverage.rs.txt`; README claims follow
//! [`verified`], never the archive.

/// Verification state of one active ledger entry (AGENTS.md D10).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CoverageStatus {
    /// Oracle-backed (Tier 0/1) with a recorded tolerance.
    Verified,
    /// Implemented, but not yet cross-implementation verified.
    Experimental,
    /// Generated compatibility surface; active v0.2 code must not use this.
    Generated,
    /// Incomplete placeholder; active v0.2 code must not use this.
    Stub,
}

/// One algorithm or public computational entry point tracked by v0.2.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Algorithm {
    /// Rust path (`"linear_model.LinearRegression"`).
    pub name: &'static str,
    /// Independent implementation or specification used as a reference.
    pub python_equiv: &'static str,
    /// Broad algorithm category.
    pub kind: &'static str,
    status: CoverageStatus,
}

impl Algorithm {
    /// Current evidence level, recorded explicitly rather than inferred from
    /// a substring in the algorithm name.
    pub const fn status(&self) -> CoverageStatus {
        self.status
    }
}

const fn entry(
    name: &'static str,
    python_equiv: &'static str,
    kind: &'static str,
    status: CoverageStatus,
) -> Algorithm {
    Algorithm {
        name,
        python_equiv,
        kind,
        status,
    }
}

const fn v(name: &'static str, python_equiv: &'static str, kind: &'static str) -> Algorithm {
    entry(name, python_equiv, kind, CoverageStatus::Verified)
}

const fn a(name: &'static str, python_equiv: &'static str, kind: &'static str) -> Algorithm {
    entry(name, python_equiv, kind, CoverageStatus::Experimental)
}

const INVENTORY: &[Algorithm] = &[
    v(
        "linear_model.LinearRegression",
        "sklearn.linear_model.LinearRegression",
        "estimator",
    ),
    v(
        "online.LinearRegression",
        "river.linear_model.LinearRegression",
        "online",
    ),
    v("online.OnlineAutoCorr", "river.stats.AutoCorr", "online"),
    v("online.OnlineCount", "river.stats.Count", "online"),
    v("online.OnlineCovariance", "river.stats.Cov", "online"),
    v("online.OnlineEwMean", "river.stats.EWMean", "online"),
    v("online.OnlineEwVar", "river.stats.EWVar", "online"),
    v("online.OnlineMean", "river.stats.Mean", "online"),
    v("online.OnlineSum", "river.stats.Sum", "online"),
    v("online.OnlineVar", "river.stats.Var", "online"),
    v(
        "online.OnlineVarianceThreshold",
        "river.feature_selection.VarianceThreshold",
        "online",
    ),
    v(
        "online.OnlineWeightedMean",
        "river.stats.WeightedMean",
        "online",
    ),
    v("special.betainc_reg", "scipy.special.betainc", "function"),
    v("special.chi2_cdf", "scipy.stats.chi2.cdf", "function"),
    v("special.chi2_pvalue", "scipy.stats.chi2.sf", "function"),
    v("special.digamma", "scipy.special.digamma", "function"),
    v("special.erf", "scipy.special.erf", "function"),
    v("special.f_cdf", "scipy.stats.f.cdf", "function"),
    v("special.f_pvalue", "scipy.stats.f.sf", "function"),
    v("special.gamma_p", "scipy.special.gammainc", "function"),
    v("special.ln_gamma", "scipy.special.gammaln", "function"),
    v("special.norm_cdf", "scipy.stats.norm.cdf", "function"),
    v(
        "special.norm_pvalue_two_sided",
        "scipy.stats.norm.sf",
        "function",
    ),
    v("special.student_t_cdf", "scipy.stats.t.cdf", "function"),
    v("special.student_t_pvalue", "scipy.stats.t.sf", "function"),
    v(
        "state_space.LinearGaussianStateSpace",
        "statsmodels.tsa.statespace.kalman_filter.KalmanFilter",
        "timeseries",
    ),
    v(
        "filters.bk_filter",
        "statsmodels.tsa.filters.bk_filter.bkfilter",
        "forecast",
    ),
    v(
        "filters.cf_filter",
        "statsmodels.tsa.filters.cf_filter.cffilter",
        "forecast",
    ),
    v(
        "filters.convolution_filter",
        "statsmodels.tsa.filters.filtertools.convolution_filter",
        "forecast",
    ),
    v(
        "filters.lfilter",
        "statsmodels.tsa.filters.filtertools.lfilter",
        "forecast",
    ),
    v(
        "filters.LocalLinearTrend",
        "statsmodels.tsa.statespace.structural.UnobservedComponents",
        "forecast",
    ),
    v(
        "filters.miso_lfilter",
        "statsmodels.tsa.filters.filtertools.miso_lfilter",
        "forecast",
    ),
    v(
        "filters.recursive_filter",
        "statsmodels.tsa.filters.filtertools.recursive_filter",
        "forecast",
    ),
    v(
        "tsa.arma_acf",
        "statsmodels.tsa.arima_process.arma_acf",
        "forecast",
    ),
    v(
        "tsa.arma_acovf",
        "statsmodels.tsa.arima_process.arma_acovf",
        "forecast",
    ),
    v(
        "tsa.arma_impulse_response",
        "statsmodels.tsa.arima_process.arma2ma",
        "forecast",
    ),
    v(
        "tsa.arma2ar",
        "statsmodels.tsa.arima_process.arma2ar",
        "forecast",
    ),
    v(
        "tsa.arma2ma",
        "statsmodels.tsa.arima_process.arma2ma",
        "forecast",
    ),
    v("tsa.Fiegarch", "BollerslevMikkelsen1996.Eq11", "forecast"),
    v(
        "tsa.Figarch",
        "BaillieBollerslevMikkelsen1996.FIGARCH11",
        "forecast",
    ),
    v("tsa.Garch11", "arch.univariate.GARCH", "forecast"),
    v(
        "stats.process_mle",
        "statsmodels.regression.process.ProcessMLE",
        "stochastic-process",
    ),
    a("hmm.HiddenMarkovModel", "hmmlearn.base.BaseHMM", "hmm"),
    a("hmm.GaussianEmission", "hmmlearn.hmm.GaussianHMM", "hmm"),
    a(
        "hmm.CategoricalEmission",
        "hmmlearn.hmm.CategoricalHMM",
        "hmm",
    ),
    a("hmm.PoissonEmission", "hmmlearn.hmm.PoissonHMM", "hmm"),
    a(
        "anomaly.KnnDistanceAnomaly",
        "sklearn.neighbors.NearestNeighbors",
        "anomaly",
    ),
    a(
        "optimize.NelderMead",
        "argmin.solver.NelderMead",
        "optimizer",
    ),
    a("tsa.EwmaVol", "arch.univariate.EWMAVariance", "forecast"),
    a("tsa.Egarch", "arch.univariate.EGARCH", "forecast"),
    a(
        "tsa.arma_generate_sample",
        "statsmodels.tsa.arima_process.arma_generate_sample",
        "forecast",
    ),
];

/// Return the active v0.2 verification ledger.
pub const fn inventory() -> &'static [Algorithm] {
    INVENTORY
}

/// Iterate over entries backed by the recorded Tier 0/1 evidence.
pub fn verified() -> impl Iterator<Item = &'static Algorithm> {
    INVENTORY
        .iter()
        .filter(|algorithm| algorithm.status() == CoverageStatus::Verified)
}

#[cfg(test)]
mod tests {
    use super::*;

    const VERIFIED_NAMES: &[&str] = &[
        "linear_model.LinearRegression",
        "online.LinearRegression",
        "online.OnlineAutoCorr",
        "online.OnlineCount",
        "online.OnlineCovariance",
        "online.OnlineEwMean",
        "online.OnlineEwVar",
        "online.OnlineMean",
        "online.OnlineSum",
        "online.OnlineVar",
        "online.OnlineVarianceThreshold",
        "online.OnlineWeightedMean",
        "special.betainc_reg",
        "special.chi2_cdf",
        "special.chi2_pvalue",
        "special.digamma",
        "special.erf",
        "special.f_cdf",
        "special.f_pvalue",
        "special.gamma_p",
        "special.ln_gamma",
        "special.norm_cdf",
        "special.norm_pvalue_two_sided",
        "special.student_t_cdf",
        "special.student_t_pvalue",
        "state_space.LinearGaussianStateSpace",
        "filters.bk_filter",
        "filters.cf_filter",
        "filters.convolution_filter",
        "filters.lfilter",
        "filters.LocalLinearTrend",
        "filters.miso_lfilter",
        "filters.recursive_filter",
        "tsa.arma_acf",
        "tsa.arma_acovf",
        "tsa.arma_impulse_response",
        "tsa.arma2ar",
        "tsa.arma2ma",
        "tsa.Fiegarch",
        "tsa.Figarch",
        "tsa.Garch11",
        "stats.process_mle",
    ];

    #[test]
    fn names_are_unique_and_statuses_are_explicit() {
        let mut names: Vec<&str> = inventory().iter().map(|item| item.name).collect();
        let count = names.len();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), count);
        assert!(inventory().iter().all(|item| matches!(
            item.status(),
            CoverageStatus::Verified | CoverageStatus::Experimental
        )));
    }

    #[test]
    fn verified_surface_matches_the_documented_allowlist() {
        let actual: Vec<&str> = verified().map(|item| item.name).collect();
        assert_eq!(actual, VERIFIED_NAMES);
    }

    #[test]
    fn registered_paths_link_to_active_symbols() {
        fn type_exists<T>() {}

        type_exists::<crate::linear_model::LinearRegression>();
        type_exists::<crate::online::LinearRegression>();
        type_exists::<crate::online::OnlineAutoCorr>();
        type_exists::<crate::online::OnlineCount>();
        type_exists::<crate::online::OnlineCovariance>();
        type_exists::<crate::online::OnlineEwMean>();
        type_exists::<crate::online::OnlineEwVar>();
        type_exists::<crate::online::OnlineMean>();
        type_exists::<crate::online::OnlineSum>();
        type_exists::<crate::online::OnlineVar>();
        type_exists::<crate::online::OnlineVarianceThreshold>();
        type_exists::<crate::online::OnlineWeightedMean>();
        type_exists::<crate::state_space::LinearGaussianStateSpace>();
        type_exists::<crate::tsa::Fiegarch>();
        type_exists::<crate::tsa::Figarch>();
        type_exists::<crate::tsa::Garch11>();
        type_exists::<crate::stats::ProcessMleFit>();
        type_exists::<crate::hmm::GaussianEmission>();
        type_exists::<crate::hmm::CategoricalEmission>();
        type_exists::<crate::hmm::PoissonEmission>();
        type_exists::<crate::hmm::HiddenMarkovModel<crate::hmm::GaussianEmission>>();
        type_exists::<crate::anomaly::KnnDistanceAnomaly>();
        type_exists::<crate::optimize::NelderMead>();
        type_exists::<crate::tsa::EwmaVol>();
        type_exists::<crate::tsa::Egarch>();

        let _ = crate::special::betainc_reg;
        let _ = crate::special::chi2_cdf;
        let _ = crate::special::chi2_pvalue;
        let _ = crate::special::digamma;
        let _ = crate::special::erf;
        let _ = crate::special::f_cdf;
        let _ = crate::special::f_pvalue;
        let _ = crate::special::gamma_p;
        let _ = crate::special::ln_gamma;
        let _ = crate::special::norm_cdf;
        let _ = crate::special::norm_pvalue_two_sided;
        let _ = crate::special::student_t_cdf;
        let _ = crate::special::student_t_pvalue;
        let _ = crate::filters::bk_filter;
        let _ = crate::filters::cf_filter;
        let _ = crate::filters::convolution_filter;
        let _ = crate::filters::lfilter;
        let _ = crate::filters::miso_lfilter;
        let _ = crate::filters::recursive_filter;
        let _ = crate::tsa::arma_acf;
        let _ = crate::tsa::arma_acovf;
        let _ = crate::tsa::arma_generate_sample;
        let _ = crate::tsa::arma_impulse_response;
        let _ = crate::tsa::arma2ar;
        let _ = crate::tsa::arma2ma;
        let _ = crate::stats::process_mle;
    }
}
