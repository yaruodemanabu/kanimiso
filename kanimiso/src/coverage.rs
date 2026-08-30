//! Coverage ledger: every estimator and public function mapped to its Python peer.
//!
//! [`inventory`] is the sklearn / statsmodels / sktime / tslearn / hmmlearn /
//! river surface that this crate implements. New algorithms must be appended
//! here so the ledger stays the source of truth.

/// One implemented algorithm or public computational entry point.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Algorithm {
    /// Rust path (`"linear_model.LinearRegression"`).
    pub name: &'static str,
    /// Python equivalent (`"sklearn.linear_model.LinearRegression"`).
    pub python_equiv: &'static str,
    /// Kind: `estimator`, `metric`, `function`, `splitter`, `transformer`,
    /// `online`, `forecast`, `hmm`, `manifold`, `covariance`, `anomaly`,
    /// `neural`, `timeseries`, `stats`.
    pub kind: &'static str,
}

const fn a(name: &'static str, python_equiv: &'static str, kind: &'static str) -> Algorithm {
    Algorithm {
        name,
        python_equiv,
        kind,
    }
}

const INVENTORY: &[Algorithm] = &[
    // linear_model
    a(
        "linear_model.LinearRegression",
        "sklearn.linear_model.LinearRegression",
        "estimator",
    ),
    a(
        "linear_model.Wls",
        "statsmodels.regression.linear_model.WLS",
        "estimator",
    ),
    a(
        "linear_model.Ridge",
        "sklearn.linear_model.Ridge",
        "estimator",
    ),
    a(
        "linear_model.Lasso",
        "sklearn.linear_model.Lasso",
        "estimator",
    ),
    a(
        "linear_model.ElasticNet",
        "sklearn.linear_model.ElasticNet",
        "estimator",
    ),
    a(
        "linear_model.LogisticRegression",
        "sklearn.linear_model.LogisticRegression",
        "estimator",
    ),
    a(
        "multinomial.MultinomialLogistic",
        "sklearn.linear_model.LogisticRegression",
        "estimator",
    ),
    a(
        "linear_model.Lars",
        "sklearn.linear_model.Lars",
        "estimator",
    ),
    a(
        "linear_model.TweedieRegressor",
        "sklearn.linear_model.TweedieRegressor",
        "estimator",
    ),
    a(
        "classification.PlattCalibrator",
        "sklearn.calibration.CalibratedClassifierCV",
        "estimator",
    ),
    a(
        "linear_model.SgdRegressor",
        "sklearn.linear_model.SGDRegressor",
        "estimator",
    ),
    a(
        "linear_model.HuberRegressor",
        "sklearn.linear_model.HuberRegressor",
        "estimator",
    ),
    a(
        "linear_model.IsotonicRegression",
        "sklearn.isotonic.IsotonicRegression",
        "estimator",
    ),
    a(
        "linear_model.KernelRidge",
        "sklearn.kernel_ridge.KernelRidge",
        "estimator",
    ),
    a(
        "linear_model.PlsRegression",
        "sklearn.cross_decomposition.PLSRegression",
        "estimator",
    ),
    a(
        "linear_model.QuantileRegressor",
        "sklearn.linear_model.QuantileRegressor",
        "estimator",
    ),
    a(
        "linear_model.PoissonRegressor",
        "sklearn.linear_model.PoissonRegressor",
        "estimator",
    ),
    a(
        "linear_model.DummyRegressor",
        "sklearn.dummy.DummyRegressor",
        "estimator",
    ),
    a(
        "robust.RansacRegressor",
        "sklearn.linear_model.RANSACRegressor",
        "estimator",
    ),
    a(
        "robust.TheilSenRegressor",
        "sklearn.linear_model.TheilSenRegressor",
        "estimator",
    ),
    a(
        "robust.Gls",
        "statsmodels.regression.linear_model.GLS",
        "estimator",
    ),
    a(
        "robust.GammaRegressor",
        "sklearn.linear_model.GammaRegressor",
        "estimator",
    ),
    a(
        "robust.OrthogonalMatchingPursuit",
        "sklearn.linear_model.OrthogonalMatchingPursuit",
        "estimator",
    ),
    a(
        "mixed.MixedLM",
        "statsmodels.regression.mixed_linear_model.MixedLM",
        "estimator",
    ),
    a(
        "vecm.Johansen",
        "statsmodels.tsa.vector_ar.vecm.coint_johansen",
        "forecast",
    ),
    a(
        "vecm.Vecm",
        "statsmodels.tsa.vector_ar.vecm.VECM",
        "forecast",
    ),
    a(
        "cluster.MeanShift",
        "sklearn.cluster.MeanShift",
        "estimator",
    ),
    a("cluster.Optics", "sklearn.cluster.OPTICS", "estimator"),
    a("cluster.Birch", "sklearn.cluster.Birch", "estimator"),
    a(
        "tslearn.Rocket",
        "sktime.transformations.panel.rocket.Rocket",
        "timeseries",
    ),
    // classification extras
    a(
        "classification.RidgeClassifier",
        "sklearn.linear_model.RidgeClassifier",
        "estimator",
    ),
    a(
        "classification.Perceptron",
        "sklearn.linear_model.Perceptron",
        "estimator",
    ),
    a(
        "classification.PassiveAggressive",
        "sklearn.linear_model.PassiveAggressiveClassifier",
        "estimator",
    ),
    a(
        "classification.DummyClassifier",
        "sklearn.dummy.DummyClassifier",
        "estimator",
    ),
    // tree
    a(
        "tree.DecisionTreeClassifier",
        "sklearn.tree.DecisionTreeClassifier",
        "estimator",
    ),
    a(
        "tree.DecisionTreeRegressor",
        "sklearn.tree.DecisionTreeRegressor",
        "estimator",
    ),
    a(
        "tree.RandomForestClassifier",
        "sklearn.ensemble.RandomForestClassifier",
        "estimator",
    ),
    a(
        "tree.RandomForestRegressor",
        "sklearn.ensemble.RandomForestRegressor",
        "estimator",
    ),
    a(
        "tree.ExtraTreesClassifier",
        "sklearn.ensemble.ExtraTreesClassifier",
        "estimator",
    ),
    a(
        "tree.ExtraTreesRegressor",
        "sklearn.ensemble.ExtraTreesRegressor",
        "estimator",
    ),
    a(
        "histgb.HistGradientBoostingRegressor",
        "sklearn.ensemble.HistGradientBoostingRegressor",
        "estimator",
    ),
    a(
        "histgb.HistGradientBoostingClassifier",
        "sklearn.ensemble.HistGradientBoostingClassifier",
        "estimator",
    ),
    a(
        "tree.GradientBoostingRegressor",
        "sklearn.ensemble.GradientBoostingRegressor",
        "estimator",
    ),
    a(
        "tree.GradientBoostingClassifier",
        "sklearn.ensemble.GradientBoostingClassifier",
        "estimator",
    ),
    a(
        "tree.AdaBoostClassifier",
        "sklearn.ensemble.AdaBoostClassifier",
        "estimator",
    ),
    a(
        "tree.IsolationForest",
        "sklearn.ensemble.IsolationForest",
        "estimator",
    ),
    // neighbors
    a(
        "neighbors.KNeighborsClassifier",
        "sklearn.neighbors.KNeighborsClassifier",
        "estimator",
    ),
    a(
        "neighbors.KNeighborsRegressor",
        "sklearn.neighbors.KNeighborsRegressor",
        "estimator",
    ),
    a(
        "neighbors.RadiusNeighborsClassifier",
        "sklearn.neighbors.RadiusNeighborsClassifier",
        "estimator",
    ),
    a(
        "neighbors.LocalOutlierFactor",
        "sklearn.neighbors.LocalOutlierFactor",
        "estimator",
    ),
    a(
        "neighbors.KernelDensity",
        "sklearn.neighbors.KernelDensity",
        "estimator",
    ),
    a(
        "neighbors.NearestCentroid",
        "sklearn.neighbors.NearestCentroid",
        "estimator",
    ),
    // svm
    a("svm.LinearSvc", "sklearn.svm.LinearSVC", "estimator"),
    a("svm.LinearSvr", "sklearn.svm.LinearSVR", "estimator"),
    a("svm.Svc", "sklearn.svm.SVC", "estimator"),
    a("svm.Svr", "sklearn.svm.SVR", "estimator"),
    a("svm.OneClassSvm", "sklearn.svm.OneClassSVM", "estimator"),
    // naive_bayes
    a(
        "naive_bayes.GaussianNB",
        "sklearn.naive_bayes.GaussianNB",
        "estimator",
    ),
    a(
        "naive_bayes.MultinomialNB",
        "sklearn.naive_bayes.MultinomialNB",
        "estimator",
    ),
    a(
        "naive_bayes.BernoulliNB",
        "sklearn.naive_bayes.BernoulliNB",
        "estimator",
    ),
    a(
        "naive_bayes.ComplementNB",
        "sklearn.naive_bayes.ComplementNB",
        "estimator",
    ),
    // preprocess
    a(
        "preprocess.StandardScaler",
        "sklearn.preprocessing.StandardScaler",
        "transformer",
    ),
    a(
        "preprocess.MinMaxScaler",
        "sklearn.preprocessing.MinMaxScaler",
        "transformer",
    ),
    a(
        "preprocess.RobustScaler",
        "sklearn.preprocessing.RobustScaler",
        "transformer",
    ),
    a(
        "preprocess.MaxAbsScaler",
        "sklearn.preprocessing.MaxAbsScaler",
        "transformer",
    ),
    a(
        "preprocess.Normalizer",
        "sklearn.preprocessing.Normalizer",
        "transformer",
    ),
    a(
        "preprocess.OneHotEncoder",
        "sklearn.preprocessing.OneHotEncoder",
        "transformer",
    ),
    a(
        "preprocess.OrdinalEncoder",
        "sklearn.preprocessing.OrdinalEncoder",
        "transformer",
    ),
    a(
        "preprocess.LabelEncoder",
        "sklearn.preprocessing.LabelEncoder",
        "transformer",
    ),
    a(
        "preprocess.SimpleImputer",
        "sklearn.impute.SimpleImputer",
        "transformer",
    ),
    a(
        "preprocess.KnnImputer",
        "sklearn.impute.KNNImputer",
        "transformer",
    ),
    a(
        "preprocess.PolynomialFeatures",
        "sklearn.preprocessing.PolynomialFeatures",
        "transformer",
    ),
    a(
        "preprocess.PowerTransformer",
        "sklearn.preprocessing.PowerTransformer",
        "transformer",
    ),
    a(
        "preprocess.QuantileTransformer",
        "sklearn.preprocessing.QuantileTransformer",
        "transformer",
    ),
    a(
        "preprocess.KBinsDiscretizer",
        "sklearn.preprocessing.KBinsDiscretizer",
        "transformer",
    ),
    a(
        "preprocess.Binarizer",
        "sklearn.preprocessing.Binarizer",
        "transformer",
    ),
    // feature
    a(
        "feature.VarianceThreshold",
        "sklearn.feature_selection.VarianceThreshold",
        "transformer",
    ),
    a(
        "feature.SelectKBest",
        "sklearn.feature_selection.SelectKBest",
        "transformer",
    ),
    a(
        "feature.Rfe",
        "sklearn.feature_selection.RFE",
        "transformer",
    ),
    a(
        "feature.RbfSampler",
        "sklearn.kernel_approximation.RBFSampler",
        "transformer",
    ),
    a(
        "feature.Nystroem",
        "sklearn.kernel_approximation.Nystroem",
        "transformer",
    ),
    a(
        "feature.MutualInfoClassif",
        "sklearn.feature_selection.mutual_info_classif",
        "function",
    ),
    a(
        "feature.lag_features",
        "sktime.transformations.series.lag.Lag",
        "function",
    ),
    a(
        "feature.rolling_mean",
        "sktime.transformations.series.rolling.RollingMean",
        "function",
    ),
    // decompose
    a("decompose.Pca", "sklearn.decomposition.PCA", "estimator"),
    a(
        "decompose.IncrementalPca",
        "sklearn.decomposition.IncrementalPCA",
        "estimator",
    ),
    a(
        "decompose.TruncatedSvd",
        "sklearn.decomposition.TruncatedSVD",
        "estimator",
    ),
    a("decompose.Nmf", "sklearn.decomposition.NMF", "estimator"),
    a(
        "decompose.FastIca",
        "sklearn.decomposition.FastICA",
        "estimator",
    ),
    a(
        "decompose.FactorAnalysis",
        "sklearn.decomposition.FactorAnalysis",
        "estimator",
    ),
    a(
        "decompose.Cca",
        "sklearn.cross_decomposition.CCA",
        "estimator",
    ),
    a(
        "decompose.SparsePca",
        "sklearn.decomposition.SparsePCA",
        "estimator",
    ),
    a(
        "decompose.DictionaryLearning",
        "sklearn.decomposition.DictionaryLearning",
        "estimator",
    ),
    // cluster
    a("cluster.KMeans", "sklearn.cluster.KMeans", "estimator"),
    a(
        "cluster.MiniBatchKMeans",
        "sklearn.cluster.MiniBatchKMeans",
        "estimator",
    ),
    a(
        "cluster.StreamKMeans",
        "river.cluster.STREAMKMeans",
        "online",
    ),
    a("cluster.Dbscan", "sklearn.cluster.DBSCAN", "estimator"),
    a(
        "cluster.Agglomerative",
        "sklearn.cluster.AgglomerativeClustering",
        "estimator",
    ),
    a(
        "cluster.GaussianMixture",
        "sklearn.mixture.GaussianMixture",
        "estimator",
    ),
    a(
        "cluster.SpectralClustering",
        "sklearn.cluster.SpectralClustering",
        "estimator",
    ),
    a(
        "cluster.AffinityPropagation",
        "sklearn.cluster.AffinityPropagation",
        "estimator",
    ),
    // hmm
    a("hmm.GaussianHmm", "hmmlearn.hmm.GaussianHMM", "hmm"),
    a("hmm.MultinomialHmm", "hmmlearn.hmm.CategoricalHMM", "hmm"),
    a("hmm.GmmHmm", "hmmlearn.hmm.GMMHMM", "hmm"),
    // tsa / sktime
    a("tsa.acf", "statsmodels.tsa.stattools.acf", "forecast"),
    a("tsa.pacf", "statsmodels.tsa.stattools.pacf", "forecast"),
    a(
        "tsa.seasonal_decompose",
        "statsmodels.tsa.seasonal.seasonal_decompose",
        "forecast",
    ),
    a("tsa.stl_like", "statsmodels.tsa.seasonal.STL", "forecast"),
    a(
        "tsa.HoltWinters",
        "statsmodels.tsa.holtwinters.ExponentialSmoothing",
        "forecast",
    ),
    a("tsa.Arima", "statsmodels.tsa.arima.model.ARIMA", "forecast"),
    a(
        "tsa.Sarima",
        "statsmodels.tsa.statespace.sarimax.SARIMAX",
        "forecast",
    ),
    a(
        "tsa.Var",
        "statsmodels.tsa.vector_ar.var_model.VAR",
        "forecast",
    ),
    a(
        "tsa.ExponentialSmoothing",
        "sktime.forecasting.exp_smoothing.ExponentialSmoothing",
        "forecast",
    ),
    a(
        "tsa.Naive",
        "sktime.forecasting.naive.NaiveForecaster",
        "forecast",
    ),
    a(
        "tsa.SeasonalNaive",
        "sktime.forecasting.naive.NaiveForecaster",
        "forecast",
    ),
    a(
        "tsa.Drift",
        "sktime.forecasting.trend.TrendForecaster",
        "forecast",
    ),
    a(
        "tsa.Theta",
        "sktime.forecasting.theta.ThetaForecaster",
        "forecast",
    ),
    a(
        "tsa.kalman_level",
        "sktime.forecasting.kalman_filter",
        "forecast",
    ),
    a(
        "tsa.hp_filter",
        "statsmodels.tsa.filters.hp_filter.hpfilter",
        "forecast",
    ),
    a("tsa.Garch11", "arch.univariate.GARCH", "forecast"),
    a(
        "tsa.Croston",
        "sktime.forecasting.croston.Croston",
        "forecast",
    ),
    // stats
    a("stats.describe", "scipy.stats.describe", "stats"),
    a("stats.pearson", "scipy.stats.pearsonr", "stats"),
    a("stats.spearman", "scipy.stats.spearmanr", "stats"),
    a("stats.kendall", "scipy.stats.kendalltau", "stats"),
    a("stats.corrcoef", "numpy.corrcoef", "stats"),
    a("stats.partial_corr", "statsmodels.stats.stattools", "stats"),
    a(
        "stats.vif",
        "statsmodels.stats.outliers_influence.variance_inflation_factor",
        "stats",
    ),
    a("stats.ttest_1samp", "scipy.stats.ttest_1samp", "stats"),
    a("stats.ttest_ind", "scipy.stats.ttest_ind", "stats"),
    a("stats.ttest_rel", "scipy.stats.ttest_rel", "stats"),
    a("stats.anova_oneway", "scipy.stats.f_oneway", "stats"),
    a(
        "stats.chi2_independence",
        "scipy.stats.chi2_contingency",
        "stats",
    ),
    a(
        "stats.chi2_contingency",
        "scipy.stats.chi2_contingency",
        "stats",
    ),
    a("stats.ks_2samp", "scipy.stats.ks_2samp", "stats"),
    a("stats.shapiro_francia", "scipy.stats.shapiro", "stats"),
    a("stats.levene", "scipy.stats.levene", "stats"),
    a("stats.mannwhitneyu", "scipy.stats.mannwhitneyu", "stats"),
    a("stats.wilcoxon_signed", "scipy.stats.wilcoxon", "stats"),
    a("stats.kruskal", "scipy.stats.kruskal", "stats"),
    a(
        "stats.jarque_bera",
        "statsmodels.stats.stattools.jarque_bera",
        "stats",
    ),
    a(
        "stats.durbin_watson",
        "statsmodels.stats.stattools.durbin_watson",
        "stats",
    ),
    a(
        "stats.breusch_pagan",
        "statsmodels.stats.diagnostic.het_breuschpagan",
        "stats",
    ),
    a(
        "stats.ljung_box",
        "statsmodels.stats.diagnostic.acorr_ljungbox",
        "stats",
    ),
    a(
        "stats.adfuller",
        "statsmodels.tsa.stattools.adfuller",
        "stats",
    ),
    a("stats.kpss", "statsmodels.tsa.stattools.kpss", "stats"),
    a(
        "stats.granger_causality",
        "statsmodels.tsa.stattools.grangercausalitytests",
        "stats",
    ),
    a(
        "stats.multipletests",
        "statsmodels.stats.multitest.multipletests",
        "stats",
    ),
    a("stats.bootstrap_mean", "scipy.stats.bootstrap", "stats"),
    a("stats.gaussian_kde", "scipy.stats.gaussian_kde", "stats"),
    a(
        "stats.lowess",
        "statsmodels.nonparametric.smoothers_lowess.lowess",
        "stats",
    ),
    a(
        "stats.ttest_power",
        "statsmodels.stats.power.TTestPower",
        "stats",
    ),
    a("stats.KaplanMeier", "lifelines.KaplanMeierFitter", "stats"),
    a("stats.CoxPH", "lifelines.CoxPHFitter", "stats"),
    a("stats.norm_ppf", "scipy.stats.norm.ppf", "function"),
    // online / river
    a(
        "online.LinearRegression",
        "river.linear_model.LinearRegression",
        "online",
    ),
    a(
        "online.LogisticRegression",
        "river.linear_model.LogisticRegression",
        "online",
    ),
    a(
        "online.Perceptron",
        "river.linear_model.Perceptron",
        "online",
    ),
    a(
        "online.PassiveAggressive",
        "river.linear_model.PAClassifier",
        "online",
    ),
    a(
        "online.HoeffdingTree",
        "river.tree.HoeffdingTreeClassifier",
        "online",
    ),
    a("online.Adwin", "river.drift.ADWIN", "online"),
    a("online.Ddm", "river.drift.binary.DDM", "online"),
    a("online.PageHinkley", "river.drift.PageHinkley", "online"),
    a(
        "online.AdaptiveRandomForest",
        "river.forest.ARFClassifier",
        "online",
    ),
    a(
        "online.OnlineStandardScaler",
        "river.preprocessing.StandardScaler",
        "online",
    ),
    a(
        "online.StreamKMeans",
        "river.cluster.STREAMKMeans",
        "online",
    ),
    a(
        "online.HalfSpaceTrees",
        "river.anomaly.HalfSpaceTrees",
        "online",
    ),
    a("online.OnlineAccuracy", "river.metrics.Accuracy", "online"),
    a("online.OnlineMse", "river.metrics.MSE", "online"),
    a("online.OnlineR2", "river.metrics.R2", "online"),
    a(
        "online.HoltWintersOnline",
        "river.time_series.HoltWinters",
        "online",
    ),
    a("online.AdaptiveModelRules", "river.rules.AMRules", "online"),
    a(
        "bandit.EpsilonGreedy",
        "river.bandit.EpsilonGreedy",
        "online",
    ),
    a("bandit.Ucb1", "river.bandit.UCB", "online"),
    a(
        "bandit.ThompsonBernoulli",
        "river.bandit.ThompsonSampling",
        "online",
    ),
    // linalg / special / validate
    a("linalg.least_squares", "numpy.linalg.lstsq", "function"),
    a(
        "linalg.ridge_solve",
        "sklearn.linear_model.Ridge",
        "function",
    ),
    a("linalg.thin_svd", "numpy.linalg.svd", "function"),
    a("linalg.symmetric_eigen", "numpy.linalg.eigh", "function"),
    a("linalg.chol_solve", "scipy.linalg.cho_solve", "function"),
    a("special.erf", "scipy.special.erf", "function"),
    a("special.norm_cdf", "scipy.stats.norm.cdf", "function"),
    a(
        "special.norm_pvalue_two_sided",
        "scipy.stats.norm.sf",
        "function",
    ),
    a("special.ln_gamma", "scipy.special.gammaln", "function"),
    a("special.gamma_p", "scipy.special.gammainc", "function"),
    a("special.chi2_cdf", "scipy.stats.chi2.cdf", "function"),
    a("special.chi2_pvalue", "scipy.stats.chi2.sf", "function"),
    a("special.betainc_reg", "scipy.special.betainc", "function"),
    a("special.student_t_cdf", "scipy.stats.t.cdf", "function"),
    a("special.student_t_pvalue", "scipy.stats.t.sf", "function"),
    a("special.f_cdf", "scipy.stats.f.cdf", "function"),
    a("special.f_pvalue", "scipy.stats.f.sf", "function"),
    a(
        "tree.isolation_c_factor",
        "sklearn.ensemble._iforest._average_path_length",
        "function",
    ),
    a(
        "validate.inspect_xy",
        "sklearn.utils.check_array",
        "function",
    ),
    a(
        "validate.inspect_classes",
        "sklearn.utils.multiclass.type_of_target",
        "function",
    ),
    a(
        "validate.inspect_identification",
        "statsmodels.stats.stattools",
        "function",
    ),
    // metrics
    a(
        "metrics.accuracy",
        "sklearn.metrics.accuracy_score",
        "metric",
    ),
    a(
        "metrics.precision_recall_f1",
        "sklearn.metrics.precision_recall_fscore_support",
        "metric",
    ),
    a("metrics.roc_auc", "sklearn.metrics.roc_auc_score", "metric"),
    a("metrics.log_loss", "sklearn.metrics.log_loss", "metric"),
    a(
        "metrics.confusion_matrix",
        "sklearn.metrics.confusion_matrix",
        "metric",
    ),
    a(
        "metrics.mse",
        "sklearn.metrics.mean_squared_error",
        "metric",
    ),
    a(
        "metrics.mae",
        "sklearn.metrics.mean_absolute_error",
        "metric",
    ),
    a("metrics.r2", "sklearn.metrics.r2_score", "metric"),
    a(
        "metrics.mape",
        "sklearn.metrics.mean_absolute_percentage_error",
        "metric",
    ),
    a(
        "metrics.medae",
        "sklearn.metrics.median_absolute_error",
        "metric",
    ),
    a(
        "metrics.silhouette",
        "sklearn.metrics.silhouette_score",
        "metric",
    ),
    a(
        "metrics.adjusted_rand",
        "sklearn.metrics.adjusted_rand_score",
        "metric",
    ),
    a(
        "metrics.mase",
        "sktime.performance_metrics.forecasting.mean_absolute_scaled_error",
        "metric",
    ),
    a(
        "metrics.smape",
        "sktime.performance_metrics.forecasting.mean_absolute_percentage_error",
        "metric",
    ),
    // model_selection
    a(
        "model_selection.train_test_split",
        "sklearn.model_selection.train_test_split",
        "splitter",
    ),
    a(
        "model_selection.KFold",
        "sklearn.model_selection.KFold",
        "splitter",
    ),
    a(
        "model_selection.StratifiedKFold",
        "sklearn.model_selection.StratifiedKFold",
        "splitter",
    ),
    a(
        "model_selection.TimeSeriesSplit",
        "sklearn.model_selection.TimeSeriesSplit",
        "splitter",
    ),
    a(
        "model_selection.cross_val_score",
        "sklearn.model_selection.cross_val_score",
        "function",
    ),
    a(
        "model_selection.cross_val_score_linear",
        "sklearn.model_selection.cross_val_score",
        "function",
    ),
    a("model_selection.take_rows", "numpy.take", "function"),
    a(
        "model_selection.GridSearchRidge",
        "sklearn.model_selection.GridSearchCV",
        "estimator",
    ),
    a(
        "model_selection.fit_transform_full",
        "sklearn.pipeline.Pipeline.fit_transform",
        "function",
    ),
    // compose / ensemble
    a(
        "compose.Standardize",
        "sklearn.preprocessing.StandardScaler",
        "transformer",
    ),
    a("compose.Pipeline", "sklearn.pipeline.Pipeline", "estimator"),
    a(
        "compose.FeatureUnion",
        "sklearn.pipeline.FeatureUnion",
        "transformer",
    ),
    a(
        "compose.ColumnTransformer",
        "sklearn.compose.ColumnTransformer",
        "transformer",
    ),
    a(
        "ensemble.VotingClassifier",
        "sklearn.ensemble.VotingClassifier",
        "estimator",
    ),
    a(
        "ensemble.VotingRegressor",
        "sklearn.ensemble.VotingRegressor",
        "estimator",
    ),
    a(
        "ensemble.BaggingRegressor",
        "sklearn.ensemble.BaggingRegressor",
        "estimator",
    ),
    a(
        "ensemble.StackingRegressor",
        "sklearn.ensemble.StackingRegressor",
        "estimator",
    ),
    // covariance
    a(
        "covariance.EmpiricalCovariance",
        "sklearn.covariance.EmpiricalCovariance",
        "covariance",
    ),
    a(
        "covariance.LedoitWolf",
        "sklearn.covariance.LedoitWolf",
        "covariance",
    ),
    a("covariance.Oas", "sklearn.covariance.OAS", "covariance"),
    a(
        "covariance.MinCovDet",
        "sklearn.covariance.MinCovDet",
        "covariance",
    ),
    a(
        "covariance.GraphicalLasso",
        "sklearn.covariance.GraphicalLasso",
        "covariance",
    ),
    a(
        "covariance.EllipticEnvelope",
        "sklearn.covariance.EllipticEnvelope",
        "anomaly",
    ),
    // anomaly
    a(
        "anomaly.IsolationForest",
        "sklearn.ensemble.IsolationForest",
        "anomaly",
    ),
    a(
        "anomaly.LocalOutlierFactor",
        "sklearn.neighbors.LocalOutlierFactor",
        "anomaly",
    ),
    a(
        "anomaly.OneClassHypersphere",
        "sklearn.svm.OneClassSVM",
        "anomaly",
    ),
    a(
        "anomaly.EllipticEnvelope",
        "sklearn.covariance.EllipticEnvelope",
        "anomaly",
    ),
    // manifold
    a("manifold.MDS", "sklearn.manifold.MDS", "manifold"),
    a("manifold.Isomap", "sklearn.manifold.Isomap", "manifold"),
    a(
        "manifold.SpectralEmbedding",
        "sklearn.manifold.SpectralEmbedding",
        "manifold",
    ),
    a(
        "manifold.LocallyLinearEmbedding",
        "sklearn.manifold.LocallyLinearEmbedding",
        "manifold",
    ),
    a("manifold.TSNE", "sklearn.manifold.TSNE", "manifold"),
    // discriminant
    a(
        "discriminant.LinearDiscriminantAnalysis",
        "sklearn.discriminant_analysis.LinearDiscriminantAnalysis",
        "estimator",
    ),
    a(
        "discriminant.QuadraticDiscriminantAnalysis",
        "sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis",
        "estimator",
    ),
    // semi
    a(
        "semi.LabelPropagation",
        "sklearn.semi_supervised.LabelPropagation",
        "estimator",
    ),
    a(
        "semi.LabelSpreading",
        "sklearn.semi_supervised.LabelSpreading",
        "estimator",
    ),
    // neural
    a(
        "neural.MLPRegressor",
        "sklearn.neural_network.MLPRegressor",
        "neural",
    ),
    a(
        "neural.MLPClassifier",
        "sklearn.neural_network.MLPClassifier",
        "neural",
    ),
    a(
        "neural.BernoulliRBM",
        "sklearn.neural_network.BernoulliRBM",
        "neural",
    ),
    // tslearn
    a("tslearn.dtw", "tslearn.metrics.dtw", "timeseries"),
    a(
        "tslearn.cdist_dtw",
        "tslearn.metrics.cdist_dtw",
        "timeseries",
    ),
    a("tslearn.softdtw", "tslearn.metrics.soft_dtw", "timeseries"),
    a(
        "tslearn.dtw_barycenter",
        "tslearn.barycenters.dtw_barycenter_averaging",
        "timeseries",
    ),
    a(
        "tslearn.TimeSeriesKMeans",
        "tslearn.clustering.TimeSeriesKMeans",
        "timeseries",
    ),
    a("tslearn.KShape", "tslearn.clustering.KShape", "timeseries"),
    a(
        "tslearn.paa",
        "tslearn.piecewise.PiecewiseAggregateApproximation",
        "timeseries",
    ),
    a(
        "tslearn.sax",
        "tslearn.piecewise.SymbolicAggregateApproximation",
        "timeseries",
    ),
    a(
        "tslearn.shapelet_distance",
        "tslearn.shapelets.LearningShapelets",
        "timeseries",
    ),
    a(
        "tslearn.TimeSeriesSvm",
        "tslearn.svm.TimeSeriesSVC",
        "timeseries",
    ),
    a(
        "tslearn.TimeSeriesForestClassifier",
        "sktime.classification.interval_based.TimeSeriesForestClassifier",
        "timeseries",
    ),
    a("coverage.inventory", "sklearn.show_versions", "function"),
    a(
        "gp.GaussianProcessRegressor",
        "sklearn.gaussian_process.GaussianProcessRegressor",
        "estimator",
    ),
    a(
        "gp.GaussianProcessClassifier",
        "sklearn.gaussian_process.GaussianProcessClassifier",
        "estimator",
    ),
    a(
        "bayes.BayesianRidge",
        "sklearn.linear_model.BayesianRidge",
        "estimator",
    ),
    a(
        "iv.TwoSls",
        "statsmodels.sandbox.regression.gmm.IV2SLS",
        "estimator",
    ),
    a(
        "iv.newey_west",
        "statsmodels.stats.sandwich_covariance.cov_hac",
        "function",
    ),
    a(
        "iv.CointEngleGranger",
        "statsmodels.tsa.stattools.coint",
        "forecast",
    ),
    a("hmm.PoissonHmm", "hmmlearn.hmm.PoissonHMM", "hmm"),
    a(
        "glm.ProbitRegression",
        "statsmodels.discrete.discrete_model.Probit",
        "estimator",
    ),
    a(
        "glm.NegativeBinomialRegressor",
        "statsmodels.discrete.discrete_model.NegativeBinomial",
        "estimator",
    ),
    a(
        "glm.SgdClassifier",
        "sklearn.linear_model.SGDClassifier",
        "estimator",
    ),
    a(
        "glm.PassiveAggressiveRegressor",
        "sklearn.linear_model.PassiveAggressiveRegressor",
        "estimator",
    ),
    a(
        "tree.AdaBoostRegressor",
        "sklearn.ensemble.AdaBoostRegressor",
        "estimator",
    ),
    a(
        "ensemble.BaggingClassifier",
        "sklearn.ensemble.BaggingClassifier",
        "estimator",
    ),
    a(
        "ensemble.StackingClassifier",
        "sklearn.ensemble.StackingClassifier",
        "estimator",
    ),
    a(
        "naive_bayes.CategoricalNB",
        "sklearn.naive_bayes.CategoricalNB",
        "estimator",
    ),
    a(
        "kernel_pca.KernelPca",
        "sklearn.decomposition.KernelPCA",
        "estimator",
    ),
    a(
        "topic.LatentDirichletAllocation",
        "sklearn.decomposition.LatentDirichletAllocation",
        "estimator",
    ),
    a("online.Kswin", "river.drift.KSWIN", "online"),
    a("online.Hddm", "river.drift.HDDM_A", "online"),
    a("online.Alma", "river.linear_model.ALMAClassifier", "online"),
    a(
        "model_selection.LeaveOneOut",
        "sklearn.model_selection.LeaveOneOut",
        "splitter",
    ),
    a(
        "model_selection.GroupKFold",
        "sklearn.model_selection.GroupKFold",
        "splitter",
    ),
    a(
        "metrics.brier",
        "sklearn.metrics.brier_score_loss",
        "metric",
    ),
    a(
        "metrics.average_precision",
        "sklearn.metrics.average_precision_score",
        "metric",
    ),
    a(
        "metrics.explained_variance",
        "sklearn.metrics.explained_variance_score",
        "metric",
    ),
    a("metrics.hamming", "sklearn.metrics.hamming_loss", "metric"),
    a(
        "metrics.mutual_info",
        "sklearn.metrics.mutual_info_score",
        "metric",
    ),
    a(
        "filters.bk_filter",
        "statsmodels.tsa.filters.bk_filter.bkfilter",
        "forecast",
    ),
    a(
        "filters.LocalLinearTrend",
        "statsmodels.tsa.statespace.structural.UnobservedComponents",
        "forecast",
    ),
    a(
        "reducer.RecursiveReducer",
        "sktime.forecasting.compose.make_reduction",
        "forecast",
    ),
    a(
        "tslearn.RocketClassifier",
        "sktime.classification.kernel_based.RocketClassifier",
        "timeseries",
    ),
    a(
        "tslearn.KNeighborsTimeSeries",
        "sktime.classification.distance_based.KNeighborsTimeSeriesClassifier",
        "timeseries",
    ),
    a(
        "tslearn.TimeSeriesForestRegressor",
        "sktime.regression.interval_based.TimeSeriesForestRegressor",
        "timeseries",
    ),
    a(
        "bayes.ArdRegression",
        "sklearn.linear_model.ARDRegression",
        "estimator",
    ),
    a(
        "classification.CalibratedClassifierCV",
        "sklearn.calibration.CalibratedClassifierCV",
        "estimator",
    ),
    a(
        "semi.SelfTrainingClassifier",
        "sklearn.semi_supervised.SelfTrainingClassifier",
        "estimator",
    ),
    a(
        "multioutput.MultiOutputRegressor",
        "sklearn.multioutput.MultiOutputRegressor",
        "estimator",
    ),
    a(
        "multioutput.ClassifierChain",
        "sklearn.multioutput.ClassifierChain",
        "estimator",
    ),
    a(
        "feature.f_classif",
        "sklearn.feature_selection.f_classif",
        "function",
    ),
    a(
        "feature.f_regression",
        "sklearn.feature_selection.f_regression",
        "function",
    ),
    a("feature.chi2", "sklearn.feature_selection.chi2", "function"),
    a(
        "feature.FeatureAgglomeration",
        "sklearn.cluster.FeatureAgglomeration",
        "transformer",
    ),
    a(
        "filters.cf_filter",
        "statsmodels.tsa.filters.cf_filter.cffilter",
        "forecast",
    ),
    a(
        "reducer.DirectReducer",
        "sktime.forecasting.compose.make_reduction",
        "forecast",
    ),
    a(
        "iv.hc0",
        "statsmodels.stats.sandwich_covariance.cov_hc0",
        "function",
    ),
    a(
        "iv.hc3",
        "statsmodels.stats.sandwich_covariance.cov_hc3",
        "function",
    ),
    a(
        "stats.het_white",
        "statsmodels.stats.diagnostic.het_white",
        "stats",
    ),
    a(
        "glm.OrderedLogit",
        "statsmodels.miscmodels.ordinal_model.OrderedModel",
        "estimator",
    ),
    a(
        "glm.Gee",
        "statsmodels.genmod.generalized_estimating_equations.GEE",
        "estimator",
    ),
    a(
        "tsa.AutoArima",
        "sktime.forecasting.arima.AutoARIMA",
        "forecast",
    ),
    a(
        "linear_model.RidgeCV",
        "sklearn.linear_model.RidgeCV",
        "estimator",
    ),
    a(
        "linear_model.LassoCV",
        "sklearn.linear_model.LassoCV",
        "estimator",
    ),
    a(
        "tslearn.cdist_softdtw",
        "tslearn.metrics.cdist_soft_dtw",
        "timeseries",
    ),
    a(
        "tslearn.KernelKMeans",
        "tslearn.clustering.KernelKMeans",
        "timeseries",
    ),
    a(
        "tslearn.TimeSeriesScalerMeanVariance",
        "tslearn.preprocessing.TimeSeriesScalerMeanVariance",
        "timeseries",
    ),
    a("online.Eddm", "river.drift.binary.EDDM", "online"),
    a("online.HddmW", "river.drift.HDDM_W", "online"),
    a(
        "online.LeveragingBagging",
        "river.ensemble.LeveragingBaggingClassifier",
        "online",
    ),
];

/// Return the static coverage ledger (every public estimator / function).
pub fn inventory() -> &'static [Algorithm] {
    INVENTORY
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ledger_is_extensive() {
        assert!(
            inventory().len() >= 100,
            "coverage ledger has {} entries",
            inventory().len()
        );
        let names: Vec<&str> = inventory().iter().map(|a| a.name).collect();
        for must in [
            "metrics.accuracy",
            "model_selection.KFold",
            "compose.Pipeline",
            "covariance.LedoitWolf",
            "anomaly.IsolationForest",
            "manifold.TSNE",
            "discriminant.LinearDiscriminantAnalysis",
            "ensemble.BaggingRegressor",
            "semi.LabelPropagation",
            "classification.RidgeClassifier",
            "neural.MLPRegressor",
            "tslearn.dtw",
            "bayes.ArdRegression",
            "glm.OrderedLogit",
            "tsa.AutoArima",
            "online.LeveragingBagging",
        ] {
            assert!(names.contains(&must), "missing {must}");
        }
    }

    #[test]
    fn names_are_unique() {
        let mut v: Vec<&str> = inventory().iter().map(|a| a.name).collect();
        v.sort_unstable();
        let n = v.len();
        v.dedup();
        assert_eq!(v.len(), n);
    }
}
