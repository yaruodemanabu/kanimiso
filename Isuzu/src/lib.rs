//! **Isuzu** — Pure-Rust simulation, inference, filtering, control, and
//! high-frequency statistics for stochastic processes. YUIMA-class workflows
//! plus the models, estimators, Kalman / particle filters, and HFT / Malliavin
//! / control tools YUIMA does not ship, with a scikit-learn-style
//! `Estimator` / `Dataset` / `Pipeline` API.
//!
//! All dependencies are Pure Rust (no BLAS / LAPACK / GSL / C bindings).
//! Linear algebra is [`faer`]; random generation lives in the companion crate
//! [`amatsuki`]. The lockfile is audited in [`audit`].
//!
//! ```
//! use isuzu::prelude::*;
//!
//! let model = GeometricBrownianMotion::new(0.05, 0.2).unwrap();
//! let sampling = Sampling::from_terminal(1.0, 250).unwrap();
//! let mut rng = seed_rng(1);
//! let path = simulate(&model, &sampling, &[100.0], &mut rng, &SimConfig::default()).unwrap();
//! assert_eq!(path.n_steps(), 250);
//! ```

#![forbid(unsafe_code)]

pub mod api;
pub mod audit;
pub mod control;
pub mod datasets;
pub mod error;
pub mod expansion;
pub mod filter;
pub mod finance;
pub mod hft;
pub mod highfreq;
pub mod infer;
pub mod infer_more;
pub mod linalg;
pub mod malliavin;
pub mod model;
pub mod models;
pub mod noise;
pub mod npbayes;
pub mod optimize;
pub mod path;
pub mod qmc;
pub mod rng;
pub mod sampling;
pub mod scheme;
pub mod simulate;
pub mod yuima;

pub use api::{Estimator, EulerSimulator, Pipeline, QmleEstimator, RecoveryReport};
pub use error::{Error, Result};
pub use faer::{Col, Mat};
pub use model::{FnSde, LinearStateSpace, ParametricSde, Sde};
pub use path::{AsyncData, Ensemble, Path, TickSeries};
pub use rng::{seed_rng, Rng, SeededRng};
pub use sampling::Sampling;
pub use scheme::Scheme;
pub use simulate::{simulate, simulate_n, SimConfig};
pub use yuima::Yuima;

pub use npbayes::{
    dp_gaussian_mixture_gibbs, finite_beta_bernoulli_gibbs, hdp_crf_gaussian_gibbs,
    ibp_linear_gaussian_gibbs, sample_bernoulli_process, sample_beta_process_finite,
    sample_crp_assignments, sample_ibp_sequential, sample_stick_breaking, BetaProcessParams,
    FeatureMatrix, IbpParams, StickBreakingKind,
};

/// Common imports for interactive / example use.
pub mod prelude {
    pub use crate::api::{
        recover, Estimator, EulerSimulator, Pipeline, QmleEstimator, RecoveryReport,
    };
    pub use crate::control::{
        hjb_1d, hjb_1d_implicit, kelly_fraction, kushner_dupuis_1d, MertonPortfolio,
    };
    pub use crate::datasets::{
        make_cir, make_dp_gaussians, make_gbm, make_hawkes, make_ibp_linear_gaussian, make_ou,
    };
    pub use crate::error::{Error, Result};
    pub use crate::expansion::{ito_taylor_expectation, mc_functional, ScalarJet};
    pub use crate::filter::{
        adaptive_kalman, auxiliary_particle_filter, continuous_discrete_ekf, cubature_kalman,
        ensemble_kalman, extended_kalman, extended_rts_smoother, gaussian_sum_filter,
        information_filter, iterated_ekf, kalman, kalman_bucy, particle_filter, particle_smoother,
        regularized_particle_filter, rts_smoother, second_order_ekf, sis_filter,
        square_root_kalman, unscented_kalman, unscented_particle_filter, AdaptiveKalmanConfig,
        DiscreteSsm, EnkfConfig, FnSsm, GaussianFilter, GaussianSmoother, IekfConfig, KalmanBucy,
        LinearGaussian, ParticleConfig, ParticleFilter, RegularizedConfig, ResamplingScheme,
        SdeSsm, UkfParams,
    };
    pub use crate::finance::{
        bs_call, bs_put, crr_price, price_sde, EuropeanCall, EuropeanPut, FlatCurve,
        MonteCarloEstimate, PricingMeasure,
    };
    pub use crate::hft::{
        ofi, preaveraged_hy, realized_kernel, roll_spread, two_scale_rv, Acd11, AlmgrenChriss,
        LobSnapshot,
    };
    pub use crate::highfreq::{
        bns_jump_test, bns_ratio_test, cce, hayashi_yoshida, hy_avar, lead_lag, lead_lag_grid,
        lee_mykland, realized_covariance,
    };
    pub use crate::infer::{
        adaptive_bayes, change_point_qv, euler_residuals, lasso_qmle, lse, qmle, quasi_loglik, Fit,
    };
    pub use crate::infer_more::{
        cic, cogarch_gmm, hurst_mmfrac, hurst_qgv, kessler_qmle, threshold_qmle, two_stage_qmle,
    };
    pub use crate::malliavin::{
        asymptotic_term, characteristic_function, kernel_density_mc, malliavin_density,
        malliavin_greeks, moment_summary, moments, Greeks,
    };
    pub use crate::model::{FnSde, LinearStateSpace, ParametricSde, Sde};
    pub use crate::models::{
        Bates, Bessel, BlackKarasinski, BrownianBridge, Carma, CarmaHawkes, CarteaFigueroa, Cir,
        Ckls, Cogarch, ExponentialHawkes, FractionalGbm, FractionalOu, GeometricBrownianMotion,
        GibsonSchwartz, Heston, HomogeneousPoisson, HullWhite, Jacobi, KouJumpDiffusion,
        LuciaSchwartz, MertonJumpDiffusion, MultivariateHawkes, MultivariateHawkes2,
        OrnsteinUhlenbeck, PowerLawHawkes, RegimeSwitchingDiffusion, Sabr, SchwartzOneFactor,
        SchwartzSmith, SteinStein, ThreeHalves, Vasicek, WeibullRenewal,
    };
    pub use crate::noise::{JumpLaw, LevyMeasure};
    pub use crate::npbayes::{
        dp_gaussian_mixture_gibbs, finite_beta_bernoulli_gibbs, hdp_crf_gaussian_gibbs,
        ibp_linear_gaussian_gibbs, sample_bernoulli_process, sample_beta_process_finite,
        sample_crp_assignments, sample_ibp_sequential, sample_ibp_stick_breaking,
        sample_pitman_yor_crp, sample_stick_breaking, BetaProcessParams, FeatureMatrix, IbpParams,
        PitmanYorParams, StickBreakingKind,
    };
    pub use crate::optimize::{LbfgsOptions, OptOptions};
    pub use crate::path::{AsyncData, Ensemble, Path, TickSeries};
    pub use crate::qmc::{brownian_bridge_path, sobol_brownian, Sobol};
    pub use crate::rng::{seed_rng, Rng, SeededRng};
    pub use crate::sampling::Sampling;
    pub use crate::scheme::Scheme;
    pub use crate::simulate::{
        poisson_random_sampling, simulate, simulate_cogarch, simulate_gbm_exact, simulate_n,
        simulate_ou_exact, SimConfig,
    };
    pub use crate::yuima::Yuima;
    pub use faer::{Col, Mat};
}

pub use optimize::{LbfgsOptions, OptOptions};
