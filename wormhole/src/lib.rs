//! `wormhole` — Pure-Rust distances and optimal transport.
//!
//! Dense numerical data is represented by [`faer::Mat`].  Kernel quantities
//! are delegated to the sibling [`coronel`] crate and waveform quantities such
//! as DTW are delegated to [`jelly_wave`].

#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub mod barycenter;
pub mod circle;
pub mod coot;
pub mod error;
pub mod exact;
pub mod factored;
pub mod gaussian;
pub mod gmm;
pub mod gromov;
pub mod lowrank;
pub mod metrics;
pub mod optim;
pub mod partial;
pub mod result;
pub mod sinkhorn;
pub mod sliced;
pub mod solvers;
pub mod spherical;
pub mod unbalanced;
pub mod utils;
pub mod weak;

mod validate;

pub use error::{Error, Result};
pub use exact::{emd, emd2, emd_1d, quantiles, wasserstein_1d};
pub use faer;
pub use metrics::{distance, pairwise, pairwise_batch, pairwise_self, BatchMetric, Metric};
pub use result::{BarycenterResult, DualPotentials, SolverStatus, TransportPlan};
pub use sinkhorn::{greenkhorn, sinkhorn, sinkhorn2};
pub use solvers::{solve, solve_samples, MarginalConstraint, Regularization, SolveOptions};
pub use unbalanced::{barycenter_unbalanced, sinkhorn_unbalanced, sinkhorn_unbalanced2};

/// Kernel functionality maintained as the dedicated sibling crate.
pub use coronel;

/// Waveform and DTW functionality maintained as the dedicated sibling crate.
pub use jelly_wave;

/// Crate version.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
