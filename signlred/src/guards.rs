//! Slice-level numerical guards used before a factorization or fit.
//!
//! These helpers do not depend on `faer`. Matrix crates scan their storage
//! into slices (or walk entries) and call these predicates.

use crate::codes::IssueCode;
use crate::issue::Issue;
use crate::meaningless::{InterpretiveValue, Meaninglessness};
use crate::policy::Policy;
use crate::severity::Severity;

/// Result of walking a float buffer for non-finite values.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FiniteScan {
    /// Count of NaNs.
    pub nans: usize,
    /// Count of +∞.
    pub pos_infs: usize,
    /// Count of −∞.
    pub neg_infs: usize,
    /// First offending linear index, if any.
    pub first_bad: Option<usize>,
}

impl FiniteScan {
    /// True when the buffer is safe for ordinary real arithmetic.
    pub fn ok(&self) -> bool {
        self.nans == 0 && self.pos_infs == 0 && self.neg_infs == 0
    }

    /// Convert to an issue, or `None` if clean.
    pub fn to_issue(&self, what: &str) -> Option<Issue> {
        if self.ok() {
            return None;
        }
        Some(
            Issue::builder(IssueCode::NonFiniteInput)
                .message(format!(
                    "{what} contains {} NaN, {} +∞, {} −∞ (first linear index {:?})",
                    self.nans, self.pos_infs, self.neg_infs, self.first_bad
                ))
                .metric("nans", self.nans as f64)
                .metric("pos_infs", self.pos_infs as f64)
                .metric("neg_infs", self.neg_infs as f64)
                .build(),
        )
    }
}

/// Walk `data` and count non-finite values.
pub fn scan_finite(data: &[f64]) -> FiniteScan {
    let mut scan = FiniteScan {
        nans: 0,
        pos_infs: 0,
        neg_infs: 0,
        first_bad: None,
    };
    for (i, &x) in data.iter().enumerate() {
        if x.is_nan() {
            scan.nans += 1;
            if scan.first_bad.is_none() {
                scan.first_bad = Some(i);
            }
        } else if x.is_infinite() {
            if x.is_sign_positive() {
                scan.pos_infs += 1;
            } else {
                scan.neg_infs += 1;
            }
            if scan.first_bad.is_none() {
                scan.first_bad = Some(i);
            }
        }
    }
    scan
}

/// One-pass mean / variance / min / max for a slice (skips non-finite).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SliceStats {
    /// Finite count.
    pub count: usize,
    /// Arithmetic mean of finite values.
    pub mean: f64,
    /// Unbiased sample variance (n−1), 0 when count < 2.
    pub variance: f64,
    /// Minimum finite value.
    pub min: f64,
    /// Maximum finite value.
    pub max: f64,
}

impl SliceStats {
    /// Population (n) standard deviation.
    pub fn std(&self) -> f64 {
        if self.count < 2 {
            0.0
        } else {
            self.variance.max(0.0).sqrt()
        }
    }

    /// True when all finite values are equal at working precision.
    pub fn is_constant(&self, eps: f64) -> bool {
        self.count == 0 || (self.max - self.min).abs() <= eps
    }
}

/// Welford stats over finite entries.
pub fn slice_stats(data: &[f64]) -> SliceStats {
    let mut count = 0usize;
    let mut mean = 0.0;
    let mut m2 = 0.0;
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    for &x in data {
        if !x.is_finite() {
            continue;
        }
        count += 1;
        let d = x - mean;
        mean += d / count as f64;
        let d2 = x - mean;
        m2 += d * d2;
        if x < min {
            min = x;
        }
        if x > max {
            max = x;
        }
    }
    let variance = if count >= 2 {
        m2 / (count - 1) as f64
    } else {
        0.0
    };
    if count == 0 {
        min = f64::NAN;
        max = f64::NAN;
        mean = f64::NAN;
    }
    SliceStats {
        count,
        mean,
        variance,
        min,
        max,
    }
}

/// How a condition number maps onto issue codes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RankHint {
    /// Well-conditioned enough for a dense solve.
    Full,
    /// Warning-band ill-conditioning.
    Ill,
    /// Error-band near-singularity.
    NearSingular,
    /// Rank is numerically zero.
    Zero,
}

/// Classify `κ = σ_max / σ_min` (pass `+∞` when `σ_min == 0`).
pub fn classify_condition_number(kappa: f64, policy: &Policy) -> RankHint {
    if !kappa.is_finite() || kappa.is_infinite() {
        return RankHint::Zero;
    }
    if kappa >= policy.condition_number_error {
        RankHint::NearSingular
    } else if kappa >= policy.condition_number_warn {
        RankHint::Ill
    } else {
        RankHint::Full
    }
}

/// Build the standard ill-conditioned / near-singular issue from `κ`.
pub fn condition_issue(kappa: f64, policy: &Policy) -> Option<Issue> {
    match classify_condition_number(kappa, policy) {
        RankHint::Full => None,
        RankHint::Ill => Some(
            Issue::builder(IssueCode::IllConditioned)
                .message(format!(
                    "condition number κ={kappa:.4e} ≥ warn threshold {}",
                    policy.condition_number_warn
                ))
                .metric("condition_number", kappa)
                .build(),
        ),
        RankHint::NearSingular => Some(
            Issue::builder(IssueCode::NearSingular)
                .message(format!(
                    "condition number κ={kappa:.4e} ≥ error threshold {}",
                    policy.condition_number_error
                ))
                .metric("condition_number", kappa)
                .build(),
        ),
        RankHint::Zero => Some(
            Issue::builder(IssueCode::RankZero)
                .message("condition number is non-finite; every singular value is ~0")
                .metric("condition_number", f64::INFINITY)
                .meaninglessness(Meaninglessness::vacuous(
                    "linear solve / inverse",
                    "the operator is the zero map at working precision",
                    "stop; do not invert or interpret coefficients",
                ))
                .build(),
        ),
    }
}

/// Standard insufficient-sample issue.
pub fn insufficient_sample(n: usize, p: usize, policy: &Policy) -> Option<Issue> {
    if p == 0 {
        return Some(
            Issue::builder(IssueCode::EmptyMatrix)
                .message("feature count p=0")
                .metric("n", n as f64)
                .metric("p", 0.0)
                .build(),
        );
    }
    if n == 0 {
        return Some(
            Issue::builder(IssueCode::EmptyMatrix)
                .message("sample count n=0")
                .metric("n", 0.0)
                .metric("p", p as f64)
                .build(),
        );
    }
    if (n as f64) < policy.min_samples_per_parameter * p as f64 {
        let code = if n <= p {
            IssueCode::SampleSmallerThanFeatures
        } else {
            IssueCode::InsufficientSample
        };
        return Some(
            Issue::builder(code)
                .message(format!(
                    "n={n} p={p} requires at least {} observations per parameter",
                    policy.min_samples_per_parameter
                ))
                .metric("n", n as f64)
                .metric("p", p as f64)
                .build(),
        );
    }
    None
}

/// Constant-target issue with meaninglessness attached.
pub fn constant_target_issue(stats: SliceStats) -> Issue {
    Issue::builder(IssueCode::ConstantTarget)
        .severity(Severity::Error)
        .message(format!(
            "target is constant on [{:.6e}, {:.6e}] (std={:.3e})",
            stats.min,
            stats.max,
            stats.std()
        ))
        .metric("target_std", stats.std())
        .metric("target_mean", stats.mean)
        .meaninglessness(Meaninglessness {
            what_was_computed: "supervised fit against a constant response".into(),
            why_meaningless:
                "there is no variation to explain; slopes, R², and skill scores are vacuous".into(),
            interpretive_value: InterpretiveValue::Vacuous,
            suggested_action: "inspect target construction; do not publish coefficients".into(),
        })
        .build()
}

/// Constant-feature issue.
pub fn constant_feature_issue(index: usize, stats: SliceStats) -> Issue {
    Issue::builder(IssueCode::ConstantFeature)
        .message(format!(
            "feature {index} is constant on [{:.6e}, {:.6e}]",
            stats.min, stats.max
        ))
        .metric("feature_index", index as f64)
        .metric("feature_std", stats.std())
        .build()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scan_finds_nan() {
        let s = scan_finite(&[1.0, f64::NAN, 3.0]);
        assert_eq!(s.nans, 1);
        assert_eq!(s.first_bad, Some(1));
        assert!(s.to_issue("X").is_some());
    }

    #[test]
    fn constant_slice() {
        let st = slice_stats(&[2.0, 2.0, 2.0]);
        assert!(st.is_constant(0.0));
        assert_eq!(st.std(), 0.0);
    }
}
