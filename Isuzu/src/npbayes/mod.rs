//! Bayesian nonparametric processes and their inference algorithms.
//!
//! This module is the catalogue YUIMA does not ship: Dirichlet / Pitman–Yor
//! processes, the Beta–Bernoulli process and Indian buffet process (IBP),
//! hierarchical Dirichlet processes, and Gibbs samplers for the conjugate
//! mixture / linear-Gaussian feature models.
//!
//! Literature and every deviation from the cited originals are written in
//! the Obsidian vault under `docs/` (see `docs/npbayes/`).

mod beta_bernoulli;
mod dirichlet;
mod infer;

pub use beta_bernoulli::{
    expected_ibp_features, sample_bernoulli_process, sample_beta_process_finite,
    sample_ibp_sequential, sample_ibp_stick_breaking, BetaProcessParams, FeatureMatrix, IbpParams,
};
pub use dirichlet::{
    expected_crp_tables, sample_crp_assignments, sample_pitman_yor_crp, sample_stick_breaking,
    ClusterSizes, PitmanYorParams, StickBreaking, StickBreakingKind,
};
pub use infer::{
    dp_gaussian_mixture_gibbs, finite_beta_bernoulli_gibbs, hdp_crf_gaussian_gibbs,
    ibp_linear_gaussian_gibbs, ibp_linear_gaussian_gibbs_ex, DpGaussianFit, HdpGaussianFit,
    IbpLinearGaussianFit,
};
