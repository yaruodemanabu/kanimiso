//! **amatsuki** — generation algorithms and RNG interfaces for Isuzu.
//!
//! This companion crate owns:
//! - the [`Rng`] / [`SeedableRng`] / [`Distribution`] interfaces
//! - the ChaCha8 stream generator ([`ChaCha8Rng`])
//! - samplers for the laws used by SDE / point-process simulation
//!
//! `isuzu` depends on this crate for all random-number generation. There is
//! no `rand` / `rand_chacha` / `rand_distr` dependency.
//!
//! The default `distributions` feature preserves the complete sampling API.
//! Consumers that only need the deterministic ChaCha8 stream can disable
//! default features and use [`ChaCha8Rng`], [`Rng`], and [`SeedableRng`]
//! without compiling the distribution implementations.

#![forbid(unsafe_code)]

pub mod chacha;
#[cfg(feature = "distributions")]
pub mod discrete;
#[cfg(feature = "distributions")]
pub mod dist;
pub mod rng;
#[cfg(feature = "distributions")]
pub mod ziggurat;

pub use chacha::{seed_rng, ChaCha8Rng, SeededRng};
#[cfg(feature = "distributions")]
pub use discrete::{
    sample_binomial, sample_multinomial, Bernoulli, Binomial, Categorical, Multinomial,
};
#[cfg(feature = "distributions")]
pub use dist::{
    sample_beta, sample_chi2, sample_dirichlet, sample_gamma, sample_inverse_gaussian,
    sample_noncentral_chi2, sample_stable_cms, Beta, DistError, Exp, Exp1, Exponential, Gamma,
    InverseGaussian, Normal, Open01, OpenClosed01, Poisson, StandardNormal, StudentT, Uniform,
};
pub use rng::{Distribution, Rng, SeedableRng};
#[cfg(feature = "distributions")]
pub use ziggurat::{sample_normal_ziggurat, StandardNormalZiggurat};
