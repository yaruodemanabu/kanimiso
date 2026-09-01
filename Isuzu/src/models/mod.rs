//! Named parametric processes (the catalogue YUIMA users reach for first).

pub mod carma;
pub mod catalogue;
pub mod cogarch;
pub mod diffusion;
pub mod energy;
pub mod hawkes_nd;
pub mod levy_paths;
pub mod point;
pub mod point_more;
pub mod util;

pub use carma::Carma;
pub use catalogue::{
    Bates, Bessel, BlackKarasinski, BrownianBridge, Cev, Ckls, FractionalOu, HullWhite, Jacobi,
    Sabr, SteinStein, ThreeHalves,
};
pub use cogarch::Cogarch;
pub use diffusion::{
    BlackScholes, Cir, FractionalGbm, GeometricBrownianMotion, Heston, KouJumpDiffusion,
    MertonJumpDiffusion, OrnsteinUhlenbeck, Vasicek,
};
pub use energy::{
    CarteaFigueroa, GibsonSchwartz, LuciaSchwartz, RegimeSwitchingDiffusion, SchwartzOneFactor,
    SchwartzSmith, SparkSpread,
};
pub use hawkes_nd::MultivariateHawkes;
pub use levy_paths::{gamma_process, levy_path};
pub use point::{ExponentialHawkes, HomogeneousPoisson, InhomogeneousPoisson};
pub use point_more::{
    CarmaHawkes, CoxCir, GammaRenewal, InhibitingHawkes, MarkedHawkes, MultivariateHawkes2,
    PowerLawHawkes, SelfCorrecting, WeibullRenewal,
};
