//! Dataset guards that feed [`signlred::Report`] before any factorization.

use crate::data::{Matrix, Vector};
use signlred::{
    constant_feature_issue, constant_target_issue, insufficient_sample, scan_finite, slice_stats,
    Issue, IssueCode, Meaninglessness, Policy, Report, SliceStats,
};

/// Scan `X` (and optional `y`) and push data-quality issues into `report`.
pub fn inspect_xy(report: &mut Report, x: &Matrix, y: Option<&Vector>, policy: &Policy) {
    let (n, p) = x.shape();
    report.set_sample_shape(n, p);

    if n == 0 || p == 0 {
        report.push_with_policy(
            policy.clone(),
            Issue::builder(IssueCode::EmptyMatrix)
                .message(format!("design is {n}×{p}"))
                .metric("n", n as f64)
                .metric("p", p as f64)
                .build(),
        );
        return;
    }

    let mut buf = Vec::with_capacity(n * p);
    for j in 0..p {
        for i in 0..n {
            buf.push(x.get(i, j));
        }
    }
    if let Some(issue) = scan_finite(&buf).to_issue("X") {
        report.push_with_policy(policy.clone(), issue);
    }

    for j in 0..p {
        let col: Vec<f64> = (0..n).map(|i| x.get(i, j)).collect();
        let st = slice_stats(&col);
        if st.count == 0 {
            report.push_with_policy(
                policy.clone(),
                Issue::builder(IssueCode::AllMissing)
                    .message(format!("column {j} has no finite values"))
                    .metric("feature_index", j as f64)
                    .build(),
            );
        } else if st.is_constant(policy.near_zero_variance) {
            report.push_with_policy(policy.clone(), constant_feature_issue(j, st));
        } else if st.std() <= policy.near_zero_variance {
            report.push_with_policy(
                policy.clone(),
                Issue::builder(IssueCode::NearZeroVariance)
                    .message(format!("column {j} std={:.3e}", st.std()))
                    .metric("feature_index", j as f64)
                    .metric("feature_std", st.std())
                    .build(),
            );
        }
    }

    if let Some(y) = y {
        if y.len() != n {
            report.push_with_policy(
                policy.clone(),
                Issue::builder(IssueCode::DimensionMismatch)
                    .message(format!("y.len()={} but X has {n} rows", y.len()))
                    .metric("n_x", n as f64)
                    .metric("n_y", y.len() as f64)
                    .build(),
            );
            return;
        }
        if let Some(issue) = scan_finite(y.as_slice()).to_issue("y") {
            report.push_with_policy(policy.clone(), issue);
        }
        let st = slice_stats(y.as_slice());
        if st.count >= 1 && st.is_constant(policy.near_zero_variance) {
            report.push_with_policy(policy.clone(), constant_target_issue(st));
        }
    }
}

/// Unregularized identification check: `n` vs `p` (after intercept).
pub fn inspect_identification(report: &mut Report, n: usize, p: usize, policy: &Policy) {
    report.set_n_parameters(p);
    if let Some(issue) = insufficient_sample(n, p, policy) {
        report.push_with_policy(policy.clone(), issue);
    }
}

/// Pairwise |corr| scan for collinearity (O(p² n)).
pub fn inspect_collinearity(report: &mut Report, x: &Matrix, policy: &Policy) {
    let (n, p) = x.shape();
    if n < 3 || p < 2 {
        return;
    }
    let mut stats: Vec<SliceStats> = Vec::with_capacity(p);
    let mut cols: Vec<Vec<f64>> = Vec::with_capacity(p);
    for j in 0..p {
        let col: Vec<f64> = (0..n).map(|i| x.get(i, j)).collect();
        stats.push(slice_stats(&col));
        cols.push(col);
    }
    for a in 0..p {
        for b in (a + 1)..p {
            let sa = stats[a];
            let sb = stats[b];
            if sa.std() <= policy.near_zero_variance || sb.std() <= policy.near_zero_variance {
                continue;
            }
            let corr = pearson(&cols[a], sa, &cols[b], sb);
            if corr.abs() >= policy.collinearity_corr {
                report.push_with_policy(
                    policy.clone(),
                    Issue::builder(IssueCode::PerfectCollinearity)
                        .message(format!("columns {a} and {b} have |corr|={corr:.6e}"))
                        .metric("i", a as f64)
                        .metric("j", b as f64)
                        .metric("corr", corr)
                        .meaninglessness(Meaninglessness::vacuous(
                            "partial regression coefficients",
                            "exact collinearity: the two columns are the same direction",
                            "drop one column; do not interpret either coefficient",
                        ))
                        .build(),
                );
            } else if corr.abs() >= 0.999 {
                report.push_with_policy(
                    policy.clone(),
                    Issue::builder(IssueCode::HighMulticollinearity)
                        .message(format!("columns {a} and {b} have |corr|={corr:.6e}"))
                        .metric("i", a as f64)
                        .metric("j", b as f64)
                        .metric("corr", corr)
                        .build(),
                );
            }
        }
    }
}

fn pearson(a: &[f64], sa: SliceStats, b: &[f64], sb: SliceStats) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 || sa.std() == 0.0 || sb.std() == 0.0 {
        return f64::NAN;
    }
    let mut s = 0.0;
    let mut k = 0.0;
    for i in 0..n {
        if a[i].is_finite() && b[i].is_finite() {
            s += (a[i] - sa.mean) * (b[i] - sb.mean);
            k += 1.0;
        }
    }
    if k < 2.0 {
        return f64::NAN;
    }
    s / ((k - 1.0) * sa.std() * sb.std())
}

/// Class-count diagnostics for classification.
pub fn inspect_classes(report: &mut Report, y: &Vector, policy: &Policy) -> Vec<(i64, usize)> {
    let mut counts: Vec<(i64, usize)> = Vec::new();
    for &v in y.as_slice() {
        if !v.is_finite() {
            continue;
        }
        let lab = v.round() as i64;
        if let Some(c) = counts.iter_mut().find(|(k, _)| *k == lab) {
            c.1 += 1;
        } else {
            counts.push((lab, 1));
        }
    }
    counts.sort_by_key(|(k, _)| *k);
    if counts.is_empty() {
        report.push_with_policy(
            policy.clone(),
            Issue::builder(IssueCode::EmptyClass)
                .message("no finite labels")
                .build(),
        );
        return counts;
    }
    if counts.len() == 1 {
        report.push_with_policy(
            policy.clone(),
            Issue::builder(IssueCode::SingleClass)
                .message(format!("only class {} is present", counts[0].0))
                .meaninglessness(Meaninglessness::vacuous(
                    "classifier",
                    "a single class makes every decision rule a constant",
                    "collect the other class; do not report accuracy",
                ))
                .build(),
        );
    }
    let n: usize = counts.iter().map(|(_, c)| *c).sum();
    if n > 0 {
        let min_c = counts.iter().map(|(_, c)| *c).min().unwrap_or(0);
        let frac = min_c as f64 / n as f64;
        if frac < policy.imbalance_warn && counts.len() > 1 {
            report.push_with_policy(
                policy.clone(),
                Issue::builder(IssueCode::ClassImbalanceSevere)
                    .message(format!(
                        "minority class fraction {frac:.4} < {}",
                        policy.imbalance_warn
                    ))
                    .metric("minority_fraction", frac)
                    .build(),
            );
        }
    }
    counts
}
