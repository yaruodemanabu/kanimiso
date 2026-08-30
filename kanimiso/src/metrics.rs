//! Classification, regression, clustering, and forecast scores.
//!
//! Every public computation opens a [`crate::context::FitCtx`]. A constant
//! `y` is diagnosed as [`IssueCode::MeaninglessFit`] (and, for classifiers,
//! [`IssueCode::ClassImbalanceSevere`]). A predictor that loses to the mean
//! raises [`IssueCode::R2Negative`].

use crate::context::FitCtx;
use crate::data::{Matrix, Vector};
use crate::validate::{inspect_classes, inspect_xy};
use ojizou_san::Session;
use signlred::{
    Issue, IssueCode, Meaninglessness, NumericalCompromise, Qualified, Result, Severity,
};

/// Precision, recall, and F1 (binary or macro-averaged).
#[derive(Clone, Debug, PartialEq)]
pub struct PrecisionRecallF1 {
    /// Precision (binary positive class, or macro mean).
    pub precision: f64,
    /// Recall (binary positive class, or macro mean).
    pub recall: f64,
    /// Harmonic mean of precision and recall.
    pub f1: f64,
    /// `(class, precision, recall, f1)` for every observed label.
    pub per_class: Vec<(i64, f64, f64, f64)>,
}

fn scan_pair(ctx: &mut FitCtx, y_true: &Vector, y_pred: &Vector, what: &str) -> bool {
    if y_true.len() != y_pred.len() {
        ctx.push(
            Issue::builder(IssueCode::DimensionMismatch)
                .message(format!(
                    "{what}: y_true.len()={} y_pred.len()={}",
                    y_true.len(),
                    y_pred.len()
                ))
                .build(),
        );
        return false;
    }
    if let Some(issue) = signlred::scan_finite(y_true.as_slice()).to_issue("y_true") {
        ctx.push(issue);
    }
    if let Some(issue) = signlred::scan_finite(y_pred.as_slice()).to_issue("y_pred") {
        ctx.push(issue);
    }
    true
}

fn y_is_constant(y: &Vector, tol: f64) -> bool {
    let st = signlred::slice_stats(y.as_slice());
    st.count >= 1 && st.is_constant(tol)
}

fn warn_constant_target(ctx: &mut FitCtx, y: &Vector, classification: bool) {
    if !y_is_constant(y, ctx.policy.near_zero_variance) {
        return;
    }
    ctx.push(
        Issue::builder(IssueCode::MeaninglessFit)
            .message("y is constant; the score has no skill interpretation")
            .meaninglessness(Meaninglessness::vacuous(
                "supervised metric against a constant response",
                "there is no variation to explain; accuracy, R², and proper scores are vacuous",
                "inspect target construction; do not publish the score as skill",
            ))
            .build(),
    );
    if classification {
        ctx.push(
            Issue::builder(IssueCode::ClassImbalanceSevere)
                .message("constant y is a one-class sample; minority fraction is 0")
                .metric("minority_fraction", 0.0)
                .build(),
        );
    }
}

fn labels_of(y: &Vector) -> Vec<i64> {
    y.as_slice()
        .iter()
        .map(|&v| if v.is_finite() { v.round() as i64 } else { 0 })
        .collect()
}

fn unique_sorted(labels: &[i64]) -> Vec<i64> {
    let mut u = labels.to_vec();
    u.sort_unstable();
    u.dedup();
    u
}

fn prf1_one(tp: f64, fp: f64, fn_: f64) -> (f64, f64, f64) {
    let p = if tp + fp > 0.0 { tp / (tp + fp) } else { 0.0 };
    let r = if tp + fn_ > 0.0 { tp / (tp + fn_) } else { 0.0 };
    let f = if p + r > 0.0 {
        2.0 * p * r / (p + r)
    } else {
        0.0
    };
    (p, r, f)
}

/// Fraction of exact (rounded) label matches.
pub fn accuracy(y_true: &Vector, y_pred: &Vector, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if !scan_pair(&mut ctx, y_true, y_pred, "accuracy") {
        return ctx.finish(f64::NAN);
    }
    inspect_classes(&mut ctx.report, y_true, &ctx.policy);
    warn_constant_target(&mut ctx, y_true, true);
    if y_true.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message("accuracy on an empty pair")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    let mut ok = 0usize;
    for i in 0..y_true.len() {
        if (y_true[i].round() - y_pred[i].round()).abs() < 0.5 {
            ok += 1;
        }
    }
    ctx.finish(ok as f64 / y_true.len() as f64)
}

/// Binary F1 when exactly two classes are present; otherwise macro-averaged F1.
pub fn precision_recall_f1(
    y_true: &Vector,
    y_pred: &Vector,
    session: &Session,
) -> Result<Qualified<PrecisionRecallF1>> {
    let mut ctx = FitCtx::with_session(session.clone());
    let empty = PrecisionRecallF1 {
        precision: f64::NAN,
        recall: f64::NAN,
        f1: f64::NAN,
        per_class: Vec::new(),
    };
    if !scan_pair(&mut ctx, y_true, y_pred, "precision_recall_f1") {
        return ctx.finish(empty);
    }
    inspect_classes(&mut ctx.report, y_true, &ctx.policy);
    warn_constant_target(&mut ctx, y_true, true);
    let yt = labels_of(y_true);
    let yp = labels_of(y_pred);
    let mut classes = unique_sorted(&yt);
    for &c in &yp {
        if !classes.contains(&c) {
            classes.push(c);
        }
    }
    classes.sort_unstable();
    if classes.is_empty() {
        return ctx.finish(empty);
    }
    let mut per = Vec::with_capacity(classes.len());
    let mut mp = 0.0;
    let mut mr = 0.0;
    let mut mf = 0.0;
    for &c in &classes {
        let mut tp = 0.0;
        let mut fp = 0.0;
        let mut fn_ = 0.0;
        for i in 0..yt.len() {
            let t = yt[i] == c;
            let p = yp[i] == c;
            if t && p {
                tp += 1.0;
            } else if !t && p {
                fp += 1.0;
            } else if t && !p {
                fn_ += 1.0;
            }
        }
        let (p, r, f) = prf1_one(tp, fp, fn_);
        per.push((c, p, r, f));
        mp += p;
        mr += r;
        mf += f;
    }
    let k = classes.len() as f64;
    let (precision, recall, f1) = if classes.len() == 2 {
        // Positive class = the larger label (sklearn-style binary).
        let pos = per.iter().max_by_key(|t| t.0).copied().unwrap_or(per[1]);
        (pos.1, pos.2, pos.3)
    } else {
        (mp / k, mr / k, mf / k)
    };
    ctx.finish(PrecisionRecallF1 {
        precision,
        recall,
        f1,
        per_class: per,
    })
}

/// Rank ROC-AUC via the Wilcoxon–Mann–Whitney statistic (ties get mid-ranks).
pub fn roc_auc(y_true: &Vector, scores: &Vector, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if !scan_pair(&mut ctx, y_true, scores, "roc_auc") {
        return ctx.finish(f64::NAN);
    }
    inspect_classes(&mut ctx.report, y_true, &ctx.policy);
    warn_constant_target(&mut ctx, y_true, true);
    let yt = labels_of(y_true);
    let classes = unique_sorted(&yt);
    if classes.len() != 2 {
        ctx.push(
            Issue::builder(IssueCode::MeaninglessFit)
                .message(format!(
                    "ROC-AUC is defined for two classes; found {}",
                    classes.len()
                ))
                .meaninglessness(Meaninglessness::vacuous(
                    "ROC-AUC",
                    "the rank AUC is a two-class functional",
                    "use a one-vs-rest reduction or a proper multiclass score",
                ))
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    let pos = classes[1];
    let mut pairs: Vec<(f64, bool)> = (0..yt.len())
        .filter(|&i| y_true[i].is_finite() && scores[i].is_finite())
        .map(|i| (scores[i], yt[i] == pos))
        .collect();
    if pairs.is_empty() {
        return ctx.finish(f64::NAN);
    }
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let n = pairs.len();
    let mut ranks = vec![0.0; n];
    let mut i = 0;
    while i < n {
        let mut j = i + 1;
        while j < n && (pairs[j].0 - pairs[i].0).abs() <= 0.0 {
            j += 1;
        }
        let mean_rank = (i + 1 + j) as f64 / 2.0;
        for r in ranks.iter_mut().take(j).skip(i) {
            *r = mean_rank;
        }
        i = j;
    }
    let mut n_pos = 0.0;
    let mut n_neg = 0.0;
    let mut sum_pos = 0.0;
    for k in 0..n {
        if pairs[k].1 {
            n_pos += 1.0;
            sum_pos += ranks[k];
        } else {
            n_neg += 1.0;
        }
    }
    if n_pos <= 0.0 || n_neg <= 0.0 {
        ctx.push(
            Issue::builder(IssueCode::ClassImbalanceSevere)
                .message("ROC-AUC has an empty class after filtering")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    let auc = (sum_pos - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg);
    ctx.finish(auc)
}

/// Mean Bernoulli / binary cross-entropy. `p` is clipped to `(eps, 1-eps)`.
pub fn log_loss(y_true: &Vector, p: &Vector, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if !scan_pair(&mut ctx, y_true, p, "log_loss") {
        return ctx.finish(f64::NAN);
    }
    inspect_classes(&mut ctx.report, y_true, &ctx.policy);
    warn_constant_target(&mut ctx, y_true, true);
    if y_true.is_empty() {
        return ctx.finish(f64::NAN);
    }
    let eps = 1e-15;
    let mut s = 0.0;
    let mut k = 0.0;
    for i in 0..y_true.len() {
        if !y_true[i].is_finite() || !p[i].is_finite() {
            continue;
        }
        let yi = if y_true[i] >= 0.5 { 1.0 } else { 0.0 };
        let pi = p[i].clamp(eps, 1.0 - eps);
        s += -(yi * pi.ln() + (1.0 - yi) * (1.0 - pi).ln());
        k += 1.0;
    }
    if k <= 0.0 {
        return ctx.finish(f64::NAN);
    }
    ctx.finish(s / k)
}

/// Confusion matrix with rows = true class, columns = predicted class.
///
/// Classes are the sorted union of rounded labels in `y_true` and `y_pred`.
pub fn confusion_matrix(
    y_true: &Vector,
    y_pred: &Vector,
    session: &Session,
) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if !scan_pair(&mut ctx, y_true, y_pred, "confusion_matrix") {
        return ctx.finish(Matrix::zeros(0, 0));
    }
    inspect_classes(&mut ctx.report, y_true, &ctx.policy);
    warn_constant_target(&mut ctx, y_true, true);
    let yt = labels_of(y_true);
    let yp = labels_of(y_pred);
    let mut classes = unique_sorted(&yt);
    for &c in &yp {
        if !classes.contains(&c) {
            classes.push(c);
        }
    }
    classes.sort_unstable();
    let k = classes.len();
    let mut m = Matrix::zeros(k, k);
    for i in 0..yt.len() {
        let r = classes.iter().position(|&c| c == yt[i]);
        let c = classes.iter().position(|&c| c == yp[i]);
        if let (Some(r), Some(c)) = (r, c) {
            m.set(r, c, m.get(r, c) + 1.0);
        }
    }
    ctx.finish(m)
}

fn residual_stats(y_true: &Vector, y_pred: &Vector) -> (f64, f64, f64, f64, f64, usize) {
    let mut sse = 0.0;
    let mut sae = 0.0;
    let mut ape = 0.0;
    let mut n_ape = 0usize;
    let mut abs_err = Vec::new();
    let mut n = 0usize;
    for i in 0..y_true.len().min(y_pred.len()) {
        if !y_true[i].is_finite() || !y_pred[i].is_finite() {
            continue;
        }
        let e = y_true[i] - y_pred[i];
        sse += e * e;
        sae += e.abs();
        abs_err.push(e.abs());
        if y_true[i].abs() > 0.0 {
            ape += e.abs() / y_true[i].abs();
            n_ape += 1;
        }
        n += 1;
    }
    let mse = if n > 0 { sse / n as f64 } else { f64::NAN };
    let mae = if n > 0 { sae / n as f64 } else { f64::NAN };
    let mape = if n_ape > 0 {
        ape / n_ape as f64
    } else {
        f64::NAN
    };
    abs_err.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let medae = if abs_err.is_empty() {
        f64::NAN
    } else if abs_err.len() % 2 == 1 {
        abs_err[abs_err.len() / 2]
    } else {
        let m = abs_err.len() / 2;
        0.5 * (abs_err[m - 1] + abs_err[m])
    };
    (mse, mae, mape, medae, sse, n)
}

fn sst_of(y: &Vector) -> (f64, f64) {
    let st = signlred::slice_stats(y.as_slice());
    let mut sst = 0.0;
    for &v in y.as_slice() {
        if v.is_finite() {
            let d = v - st.mean;
            sst += d * d;
        }
    }
    (sst, st.mean)
}

/// Mean squared error.
pub fn mse(y_true: &Vector, y_pred: &Vector, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if !scan_pair(&mut ctx, y_true, y_pred, "mse") {
        return ctx.finish(f64::NAN);
    }
    warn_constant_target(&mut ctx, y_true, false);
    let (mse, _, _, _, _, _) = residual_stats(y_true, y_pred);
    ctx.finish(mse)
}

/// Mean absolute error.
pub fn mae(y_true: &Vector, y_pred: &Vector, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if !scan_pair(&mut ctx, y_true, y_pred, "mae") {
        return ctx.finish(f64::NAN);
    }
    warn_constant_target(&mut ctx, y_true, false);
    let (_, mae, _, _, _, _) = residual_stats(y_true, y_pred);
    ctx.finish(mae)
}

/// Coefficient of determination \(1 - \mathrm{SSE}/\mathrm{SST}\).
pub fn r2(y_true: &Vector, y_pred: &Vector, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if !scan_pair(&mut ctx, y_true, y_pred, "r2") {
        return ctx.finish(f64::NAN);
    }
    warn_constant_target(&mut ctx, y_true, false);
    let (_, _, _, _, sse, n) = residual_stats(y_true, y_pred);
    let (sst, _) = sst_of(y_true);
    if n == 0 {
        return ctx.finish(f64::NAN);
    }
    if sst <= ctx.policy.r2_zero_tol {
        ctx.push(
            Issue::builder(IssueCode::MeaninglessFit)
                .message("SST≈0; R² is undefined")
                .meaninglessness(Meaninglessness::vacuous(
                    "R²",
                    "a constant target makes SST = 0",
                    "do not report R²",
                ))
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    let r2 = 1.0 - sse / sst;
    if r2 < -ctx.policy.r2_zero_tol {
        ctx.push(
            Issue::builder(IssueCode::R2Negative)
                .message(format!("R²={r2:.4e} < 0; the predictor lost to the mean"))
                .metric("r2", r2)
                .build(),
        );
    }
    ctx.finish(r2)
}

/// Mean absolute percentage error (undefined rows with \(y=0\) are skipped).
pub fn mape(y_true: &Vector, y_pred: &Vector, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if !scan_pair(&mut ctx, y_true, y_pred, "mape") {
        return ctx.finish(f64::NAN);
    }
    warn_constant_target(&mut ctx, y_true, false);
    let (_, _, mape, _, _, _) = residual_stats(y_true, y_pred);
    ctx.finish(mape)
}

/// Median absolute error.
pub fn medae(y_true: &Vector, y_pred: &Vector, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if !scan_pair(&mut ctx, y_true, y_pred, "medae") {
        return ctx.finish(f64::NAN);
    }
    warn_constant_target(&mut ctx, y_true, false);
    let (_, _, _, medae, _, _) = residual_stats(y_true, y_pred);
    ctx.finish(medae)
}

fn euclid(x: &Matrix, a: usize, b: usize) -> f64 {
    let mut s = 0.0;
    for j in 0..x.ncols() {
        let d = x.get(a, j) - x.get(b, j);
        s += d * d;
    }
    s.sqrt()
}

/// Mean silhouette coefficient with pairwise Euclidean distances.
pub fn silhouette(x: &Matrix, labels: &Vector, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    crate::validate::inspect_xy(&mut ctx.report, x, Some(labels), &ctx.policy);
    inspect_classes(&mut ctx.report, labels, &ctx.policy);
    warn_constant_target(&mut ctx, labels, true);
    if x.nrows() != labels.len() {
        ctx.push(
            Issue::builder(IssueCode::DimensionMismatch)
                .message("silhouette: rows(X) ≠ len(labels)")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    let labs = labels_of(labels);
    let classes = unique_sorted(&labs);
    if classes.len() < 2 {
        ctx.push(
            Issue::builder(IssueCode::MeaninglessFit)
                .message("silhouette needs at least two clusters")
                .meaninglessness(Meaninglessness::vacuous(
                    "silhouette",
                    "b(i) is undefined when every point shares a label",
                    "use a partition with K≥2",
                ))
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    let n = x.nrows();
    if n < 2 {
        return ctx.finish(f64::NAN);
    }
    let mut sil = 0.0;
    let mut counted = 0.0;
    for i in 0..n {
        let ci = labs[i];
        let mut a_sum = 0.0;
        let mut a_n = 0.0;
        let mut b = f64::INFINITY;
        for &c in &classes {
            let mut s = 0.0;
            let mut k = 0.0;
            for j in 0..n {
                if i == j || labs[j] != c {
                    continue;
                }
                s += euclid(x, i, j);
                k += 1.0;
            }
            if k <= 0.0 {
                continue;
            }
            let mean = s / k;
            if c == ci {
                a_sum = mean;
                a_n = k;
            } else if mean < b {
                b = mean;
            }
        }
        if a_n <= 0.0 || !b.is_finite() {
            continue;
        }
        let a = a_sum;
        let denom = a.max(b);
        if denom > 0.0 {
            sil += (b - a) / denom;
            counted += 1.0;
        }
    }
    ctx.finish(if counted > 0.0 {
        sil / counted
    } else {
        f64::NAN
    })
}

/// Per-sample silhouette (sklearn `silhouette_samples`).
///
/// Cluster count is not identification `p`. A one-class partition is vacuous
/// and aborts, matching [`silhouette`].
pub fn silhouette_samples(
    x: &Matrix,
    labels: &Vector,
    session: &Session,
) -> Result<Qualified<Vector>> {
    let mut ctx = FitCtx::with_session(session.clone());
    crate::validate::inspect_xy(&mut ctx.report, x, Some(labels), &ctx.policy);
    inspect_classes(&mut ctx.report, labels, &ctx.policy);
    warn_constant_target(&mut ctx, labels, true);
    if x.nrows() != labels.len() {
        ctx.push(
            Issue::builder(IssueCode::DimensionMismatch)
                .message("silhouette_samples: rows(X) ≠ len(labels)")
                .build(),
        );
        return ctx.finish(Vector::filled(x.nrows(), f64::NAN));
    }
    let labs = labels_of(labels);
    let classes = unique_sorted(&labs);
    if classes.len() < 2 {
        ctx.push(
            Issue::builder(IssueCode::MeaninglessFit)
                .message("silhouette_samples needs at least two clusters")
                .meaninglessness(Meaninglessness::vacuous(
                    "silhouette_samples",
                    "b(i) is undefined when every point shares a label",
                    "use a partition with K≥2",
                ))
                .build(),
        );
        return ctx.finish(Vector::filled(x.nrows(), f64::NAN));
    }
    let n = x.nrows();
    let mut out = Vector::zeros(n);
    for i in 0..n {
        let ci = labs[i];
        let mut a = 0.0;
        let mut a_n = 0.0;
        let mut b = f64::INFINITY;
        for &c in &classes {
            let mut s = 0.0;
            let mut k = 0.0;
            for j in 0..n {
                if i == j || labs[j] != c {
                    continue;
                }
                s += euclid(x, i, j);
                k += 1.0;
            }
            if k <= 0.0 {
                continue;
            }
            let mean = s / k;
            if c == ci {
                a = mean;
                a_n = k;
            } else if mean < b {
                b = mean;
            }
        }
        if a_n <= 0.0 || !b.is_finite() {
            out[i] = f64::NAN;
            continue;
        }
        let denom = a.max(b);
        out[i] = if denom > 0.0 { (b - a) / denom } else { 0.0 };
    }
    ctx.finish(out)
}

/// Pairwise Euclidean distances (sklearn `pairwise_distances` metric=`euclidean`).
///
/// Row counts are not identification `p`.
pub fn pairwise_distances(a: &Matrix, b: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, a, None, &ctx.policy);
    inspect_xy(&mut ctx.report, b, None, &ctx.policy);
    if a.ncols() != b.ncols() {
        ctx.push(
            Issue::builder(IssueCode::DimensionMismatch)
                .message(format!(
                    "pairwise_distances a.ncols()={} b.ncols()={}",
                    a.ncols(),
                    b.ncols()
                ))
                .build(),
        );
        return ctx.finish(Matrix::zeros(a.nrows(), b.nrows()));
    }
    let out = Matrix::from_fn(a.nrows(), b.nrows(), |i, j| {
        let mut s = 0.0;
        for k in 0..a.ncols() {
            let d = a.get(i, k) - b.get(j, k);
            s += d * d;
        }
        s.sqrt()
    });
    ctx.finish(out)
}

/// Pairwise Euclidean distances (sklearn `euclidean_distances`).
///
/// Row counts are not identification `p`.
pub fn euclidean_distances(a: &Matrix, b: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
    pairwise_distances(a, b, session)
}

/// Pairwise Manhattan distances (sklearn `manhattan_distances`).
pub fn manhattan_distances(a: &Matrix, b: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, a, None, &ctx.policy);
    inspect_xy(&mut ctx.report, b, None, &ctx.policy);
    if a.ncols() != b.ncols() {
        ctx.push(
            Issue::builder(IssueCode::DimensionMismatch)
                .message(format!(
                    "manhattan_distances a.ncols()={} b.ncols()={}",
                    a.ncols(),
                    b.ncols()
                ))
                .build(),
        );
        return ctx.finish(Matrix::zeros(a.nrows(), b.nrows()));
    }
    let out = Matrix::from_fn(a.nrows(), b.nrows(), |i, j| {
        let mut s = 0.0;
        for k in 0..a.ncols() {
            s += (a.get(i, k) - b.get(j, k)).abs();
        }
        s
    });
    ctx.finish(out)
}

/// Cosine similarity matrix (sklearn `cosine_similarity`).
///
/// A zero-norm row is recorded as a numerical compromise and contributes 0.
pub fn cosine_similarity(a: &Matrix, b: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, a, None, &ctx.policy);
    inspect_xy(&mut ctx.report, b, None, &ctx.policy);
    if a.ncols() != b.ncols() {
        ctx.push(
            Issue::builder(IssueCode::DimensionMismatch)
                .message(format!(
                    "cosine_similarity a.ncols()={} b.ncols()={}",
                    a.ncols(),
                    b.ncols()
                ))
                .build(),
        );
        return ctx.finish(Matrix::zeros(a.nrows(), b.nrows()));
    }
    let na = (0..a.nrows())
        .map(|i| {
            let mut s = 0.0;
            for k in 0..a.ncols() {
                let v = a.get(i, k);
                s += v * v;
            }
            s.sqrt()
        })
        .collect::<Vec<_>>();
    let nb = (0..b.nrows())
        .map(|j| {
            let mut s = 0.0;
            for k in 0..b.ncols() {
                let v = b.get(j, k);
                s += v * v;
            }
            s.sqrt()
        })
        .collect::<Vec<_>>();
    if na.iter().any(|v| *v <= 1e-18) || nb.iter().any(|v| *v <= 1e-18) {
        ctx.push(
            Issue::builder(IssueCode::JitterInjected)
                .message("cosine_similarity saw a zero-norm row; that entry is 0")
                .compromise(NumericalCompromise::new(
                    "unit-length rows",
                    "zero-norm cosine set to 0",
                    "the inner product is undefined after L2 normalisation",
                    "do not read a zero as orthogonal when a row vanished",
                ))
                .build(),
        );
    }
    let out = Matrix::from_fn(a.nrows(), b.nrows(), |i, j| {
        if na[i] <= 1e-18 || nb[j] <= 1e-18 {
            return 0.0;
        }
        let mut s = 0.0;
        for k in 0..a.ncols() {
            s += a.get(i, k) * b.get(j, k);
        }
        s / (na[i] * nb[j])
    });
    ctx.finish(out)
}

/// RBF kernel \(K_{ij}=\exp(-\gamma\|x_i-y_j\|^2)\) (sklearn `rbf_kernel`).
///
/// \(\gamma\) is not identification `p`.
pub fn rbf_kernel(
    a: &Matrix,
    b: &Matrix,
    gamma: f64,
    session: &Session,
) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, a, None, &ctx.policy);
    inspect_xy(&mut ctx.report, b, None, &ctx.policy);
    let g = if gamma.is_finite() && gamma > 0.0 {
        gamma
    } else {
        ctx.push(
            Issue::builder(IssueCode::InvalidWeight)
                .severity(Severity::Warning)
                .message(format!("rbf_kernel γ={gamma} is not positive; using 1"))
                .build(),
        );
        1.0
    };
    if a.ncols() != b.ncols() {
        ctx.push(
            Issue::builder(IssueCode::DimensionMismatch)
                .message(format!(
                    "rbf_kernel a.ncols()={} b.ncols()={}",
                    a.ncols(),
                    b.ncols()
                ))
                .build(),
        );
        return ctx.finish(Matrix::zeros(a.nrows(), b.nrows()));
    }
    let out = Matrix::from_fn(a.nrows(), b.nrows(), |i, j| {
        let mut s = 0.0;
        for k in 0..a.ncols() {
            let d = a.get(i, k) - b.get(j, k);
            s += d * d;
        }
        (-g * s).exp()
    });
    ctx.finish(out)
}

/// Linear kernel \(K=XY^\top\) (sklearn `linear_kernel`).
pub fn linear_kernel(a: &Matrix, b: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, a, None, &ctx.policy);
    inspect_xy(&mut ctx.report, b, None, &ctx.policy);
    if a.ncols() != b.ncols() {
        ctx.push(
            Issue::builder(IssueCode::DimensionMismatch)
                .message(format!(
                    "linear_kernel a.ncols()={} b.ncols()={}",
                    a.ncols(),
                    b.ncols()
                ))
                .build(),
        );
        return ctx.finish(Matrix::zeros(a.nrows(), b.nrows()));
    }
    let out = Matrix::from_fn(a.nrows(), b.nrows(), |i, j| {
        let mut s = 0.0;
        for k in 0..a.ncols() {
            s += a.get(i, k) * b.get(j, k);
        }
        s
    });
    ctx.finish(out)
}

/// Polynomial kernel \((γ\langle x,y\rangle+c_0)^d\) (sklearn `polynomial_kernel`).
///
/// Degree is not identification `p`.
pub fn polynomial_kernel(
    a: &Matrix,
    b: &Matrix,
    degree: usize,
    gamma: f64,
    coef0: f64,
    session: &Session,
) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, a, None, &ctx.policy);
    inspect_xy(&mut ctx.report, b, None, &ctx.policy);
    let d = if degree >= 1 {
        degree
    } else {
        ctx.push(
            Issue::builder(IssueCode::InvalidWeight)
                .severity(Severity::Warning)
                .message(format!("polynomial_kernel degree={degree} < 1; using 1"))
                .build(),
        );
        1
    };
    let g = if gamma.is_finite() {
        gamma
    } else {
        ctx.push(
            Issue::builder(IssueCode::InvalidWeight)
                .severity(Severity::Warning)
                .message(format!(
                    "polynomial_kernel γ={gamma} is not finite; using 1"
                ))
                .build(),
        );
        1.0
    };
    let c0 = if coef0.is_finite() { coef0 } else { 0.0 };
    if a.ncols() != b.ncols() {
        ctx.push(
            Issue::builder(IssueCode::DimensionMismatch)
                .message("polynomial_kernel column mismatch")
                .build(),
        );
        return ctx.finish(Matrix::zeros(a.nrows(), b.nrows()));
    }
    let out = Matrix::from_fn(a.nrows(), b.nrows(), |i, j| {
        let mut s = 0.0;
        for k in 0..a.ncols() {
            s += a.get(i, k) * b.get(j, k);
        }
        (g * s + c0).powi(d as i32)
    });
    ctx.finish(out)
}

/// Laplacian kernel \(\exp(-\gamma\|x-y\|_1)\) (sklearn `laplacian_kernel`).
pub fn laplacian_kernel(
    a: &Matrix,
    b: &Matrix,
    gamma: f64,
    session: &Session,
) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, a, None, &ctx.policy);
    inspect_xy(&mut ctx.report, b, None, &ctx.policy);
    let g = if gamma.is_finite() && gamma > 0.0 {
        gamma
    } else {
        ctx.push(
            Issue::builder(IssueCode::InvalidWeight)
                .severity(Severity::Warning)
                .message(format!("laplacian_kernel γ={gamma} is not positive; using 1"))
                .build(),
        );
        1.0
    };
    if a.ncols() != b.ncols() {
        ctx.push(
            Issue::builder(IssueCode::DimensionMismatch)
                .message("laplacian_kernel column mismatch")
                .build(),
        );
        return ctx.finish(Matrix::zeros(a.nrows(), b.nrows()));
    }
    let out = Matrix::from_fn(a.nrows(), b.nrows(), |i, j| {
        let mut s = 0.0;
        for k in 0..a.ncols() {
            s += (a.get(i, k) - b.get(j, k)).abs();
        }
        (-g * s).exp()
    });
    ctx.finish(out)
}

/// Sigmoid kernel \(\tanh(\gamma\langle x,y\rangle+c_0)\) (sklearn `sigmoid_kernel`).
pub fn sigmoid_kernel(
    a: &Matrix,
    b: &Matrix,
    gamma: f64,
    coef0: f64,
    session: &Session,
) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, a, None, &ctx.policy);
    inspect_xy(&mut ctx.report, b, None, &ctx.policy);
    let g = if gamma.is_finite() { gamma } else { 1.0 };
    let c0 = if coef0.is_finite() { coef0 } else { 0.0 };
    if a.ncols() != b.ncols() {
        ctx.push(
            Issue::builder(IssueCode::DimensionMismatch)
                .message("sigmoid_kernel column mismatch")
                .build(),
        );
        return ctx.finish(Matrix::zeros(a.nrows(), b.nrows()));
    }
    let out = Matrix::from_fn(a.nrows(), b.nrows(), |i, j| {
        let mut s = 0.0;
        for k in 0..a.ncols() {
            s += a.get(i, k) * b.get(j, k);
        }
        (g * s + c0).tanh()
    });
    ctx.finish(out)
}

/// χ² kernel \(\exp(-\gamma\sum(x-y)^2/(x+y))\) (sklearn `chi2_kernel`).
///
/// Negative coordinates are skipped with a warning; they are not identification
/// `p`.
pub fn chi2_kernel(
    a: &Matrix,
    b: &Matrix,
    gamma: f64,
    session: &Session,
) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, a, None, &ctx.policy);
    inspect_xy(&mut ctx.report, b, None, &ctx.policy);
    let g = if gamma.is_finite() && gamma > 0.0 {
        gamma
    } else {
        ctx.push(
            Issue::builder(IssueCode::InvalidWeight)
                .severity(Severity::Warning)
                .message(format!("chi2_kernel γ={gamma} is not positive; using 1"))
                .build(),
        );
        1.0
    };
    if a.ncols() != b.ncols() {
        ctx.push(
            Issue::builder(IssueCode::DimensionMismatch)
                .message("chi2_kernel column mismatch")
                .build(),
        );
        return ctx.finish(Matrix::zeros(a.nrows(), b.nrows()));
    }
    let mut saw_neg = false;
    for i in 0..a.nrows() {
        for k in 0..a.ncols() {
            if a.get(i, k) < 0.0 {
                saw_neg = true;
            }
        }
    }
    for i in 0..b.nrows() {
        for k in 0..b.ncols() {
            if b.get(i, k) < 0.0 {
                saw_neg = true;
            }
        }
    }
    if saw_neg {
        ctx.push(
            Issue::builder(IssueCode::NonPositiveSeries)
                .severity(Severity::Warning)
                .message("chi2_kernel skipped negative coordinates")
                .build(),
        );
    }
    let out = Matrix::from_fn(a.nrows(), b.nrows(), |i, j| {
        let mut s = 0.0;
        for k in 0..a.ncols() {
            let u = a.get(i, k);
            let v = b.get(j, k);
            if u < 0.0 || v < 0.0 {
                continue;
            }
            let den = u + v;
            if den > 1e-18 {
                let d = u - v;
                s += d * d / den;
            }
        }
        (-g * s).exp()
    });
    ctx.finish(out)
}

/// Haversine distances in kilometres (sklearn `haversine_distances` × Earth radius).
///
/// Rows are `(lat, lon)` in degrees. Column count must be 2.
pub fn haversine_distances(a: &Matrix, b: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, a, None, &ctx.policy);
    inspect_xy(&mut ctx.report, b, None, &ctx.policy);
    if a.ncols() != 2 || b.ncols() != 2 {
        ctx.push(
            Issue::builder(IssueCode::DimensionMismatch)
                .message(format!(
                    "haversine_distances needs 2 columns, got {} and {}",
                    a.ncols(),
                    b.ncols()
                ))
                .build(),
        );
        return ctx.finish(Matrix::zeros(a.nrows(), b.nrows()));
    }
    const R: f64 = 6371.0;
    let out = Matrix::from_fn(a.nrows(), b.nrows(), |i, j| {
        let lat1 = a.get(i, 0).to_radians();
        let lon1 = a.get(i, 1).to_radians();
        let lat2 = b.get(j, 0).to_radians();
        let lon2 = b.get(j, 1).to_radians();
        let dlat = lat2 - lat1;
        let dlon = lon2 - lon1;
        let h = (dlat * 0.5).sin().powi(2)
            + lat1.cos() * lat2.cos() * (dlon * 0.5).sin().powi(2);
        let h = h.clamp(0.0, 1.0);
        2.0 * R * h.sqrt().asin()
    });
    ctx.finish(out)
}

/// Maximum residual (sklearn `max_error`).
pub fn max_error(y_true: &Vector, y_pred: &Vector, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if !scan_pair(&mut ctx, y_true, y_pred, "max_error") {
        return ctx.finish(f64::NAN);
    }
    warn_constant_target(&mut ctx, y_true, false);
    let mut m: f64 = 0.0;
    let mut n = 0.0;
    for i in 0..y_true.len().min(y_pred.len()) {
        if y_true[i].is_finite() && y_pred[i].is_finite() {
            m = m.max((y_true[i] - y_pred[i]).abs());
            n += 1.0;
        }
    }
    ctx.finish(if n > 0.0 { m } else { f64::NAN })
}

/// \(F_\beta\) score (sklearn `fbeta_score`).
///
/// Binary positive class is the larger label; otherwise macro-averaged.
/// \(\beta\) is not identification `p`.
pub fn fbeta_score(
    y_true: &Vector,
    y_pred: &Vector,
    beta: f64,
    session: &Session,
) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if !scan_pair(&mut ctx, y_true, y_pred, "fbeta_score") {
        return ctx.finish(f64::NAN);
    }
    inspect_classes(&mut ctx.report, y_true, &ctx.policy);
    warn_constant_target(&mut ctx, y_true, true);
    let b = if beta.is_finite() && beta > 0.0 {
        beta
    } else {
        ctx.push(
            Issue::builder(IssueCode::InvalidWeight)
                .severity(Severity::Warning)
                .message(format!("fbeta β={beta} is not positive; using 1"))
                .build(),
        );
        1.0
    };
    let pr = match precision_recall_f1(y_true, y_pred, &session.child("prf1")) {
        Ok(q) => {
            for issue in q.report.issues() {
                if issue.code == IssueCode::MeaninglessFit {
                    continue;
                }
                ctx.push(issue.clone());
            }
            q.value
        }
        Err(e) => {
            if e.primary.code != IssueCode::MeaninglessFit {
                ctx.push(e.primary);
            }
            return ctx.finish(f64::NAN);
        }
    };
    let b2 = b * b;
    let den = b2 * pr.precision + pr.recall;
    ctx.finish(if den.abs() <= 1e-18 {
        0.0
    } else {
        (1.0 + b2) * pr.precision * pr.recall / den
    })
}

/// Unadjusted Rand index (sklearn `rand_score`).
pub fn rand_score(y_true: &Vector, y_pred: &Vector, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if !scan_pair(&mut ctx, y_true, y_pred, "rand_score") {
        return ctx.finish(f64::NAN);
    }
    inspect_classes(&mut ctx.report, y_true, &ctx.policy);
    warn_constant_target(&mut ctx, y_true, true);
    let yt = labels_of(y_true);
    let yp = labels_of(y_pred);
    let n = yt.len().min(yp.len());
    let cn = comb2(n as f64);
    if cn <= 0.0 {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .severity(Severity::Warning)
                .message("rand_score is undefined for n<2")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    let mut a = 0.0;
    let mut b = 0.0;
    for i in 0..n {
        for j in (i + 1)..n {
            let st = yt[i] == yt[j];
            let sp = yp[i] == yp[j];
            if st && sp {
                a += 1.0;
            } else if !st && !sp {
                b += 1.0;
            }
        }
    }
    ctx.finish((a + b) / cn)
}

/// Neighborhood trustworthiness of an embedding (sklearn `manifold.trustworthiness`).
///
/// Neighbor count is not identification `p`.
pub fn trustworthiness(
    x: &Matrix,
    embedded: &Matrix,
    n_neighbors: usize,
    session: &Session,
) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, x, None, &ctx.policy);
    inspect_xy(&mut ctx.report, embedded, None, &ctx.policy);
    if x.nrows() != embedded.nrows() {
        ctx.push(
            Issue::builder(IssueCode::DimensionMismatch)
                .message(format!(
                    "trustworthiness x.nrows()={} embedded.nrows()={}",
                    x.nrows(),
                    embedded.nrows()
                ))
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    let n = x.nrows();
    let k = if n_neighbors >= 1 && n_neighbors < n {
        n_neighbors
    } else {
        ctx.push(
            Issue::builder(IssueCode::InvalidWeight)
                .severity(Severity::Warning)
                .message(format!(
                    "trustworthiness n_neighbors={n_neighbors} is not in 1..n-1 (n={n}); using 1"
                ))
                .build(),
        );
        1.min(n.saturating_sub(1)).max(1)
    };
    if n < 3 {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .severity(Severity::Warning)
                .message("trustworthiness needs n≥3")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    let mut acc = 0.0;
    for i in 0..n {
        let mut orig: Vec<(f64, usize)> = (0..n)
            .filter(|&j| j != i)
            .map(|j| (euclid(x, i, j), j))
            .collect();
        orig.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let mut rank = vec![0usize; n];
        for (r, &(_, j)) in orig.iter().enumerate() {
            rank[j] = r + 1;
        }
        let mut emb: Vec<(f64, usize)> = (0..n)
            .filter(|&j| j != i)
            .map(|j| {
                let mut s = 0.0;
                for c in 0..embedded.ncols() {
                    let d = embedded.get(i, c) - embedded.get(j, c);
                    s += d * d;
                }
                (s.sqrt(), j)
            })
            .collect();
        emb.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        for &(_, j) in emb.iter().take(k) {
            let r = rank[j];
            if r > k {
                acc += (r - k) as f64;
            }
        }
    }
    let nk = n as f64 * k as f64 * (2.0 * n as f64 - 3.0 * k as f64 - 1.0);
    ctx.finish(if nk.abs() <= 1e-12 {
        1.0
    } else {
        (1.0 - 2.0 / nk * acc).clamp(0.0, 1.0)
    })
}

/// Per-label 2×2 tables stacked as \(L\times 4\) `[tn, fp, fn, tp]`
/// (sklearn `multilabel_confusion_matrix`).
pub fn multilabel_confusion_matrix(
    y_true: &Matrix,
    y_pred: &Matrix,
    session: &Session,
) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if y_true.nrows() != y_pred.nrows() || y_true.ncols() != y_pred.ncols() {
        ctx.push(
            Issue::builder(IssueCode::DimensionMismatch)
                .message(format!(
                    "multilabel_confusion_matrix y_true is {}×{} y_pred is {}×{}",
                    y_true.nrows(),
                    y_true.ncols(),
                    y_pred.nrows(),
                    y_pred.ncols()
                ))
                .build(),
        );
        return ctx.finish(Matrix::zeros(0, 4));
    }
    inspect_xy(&mut ctx.report, y_true, None, &ctx.policy);
    inspect_xy(&mut ctx.report, y_pred, None, &ctx.policy);
    let l = y_true.ncols();
    let n = y_true.nrows();
    let out = Matrix::from_fn(l, 4, |j, k| {
        let mut tn = 0.0;
        let mut fp = 0.0;
        let mut fn_ = 0.0;
        let mut tp = 0.0;
        for i in 0..n {
            let t = y_true.get(i, j) > 0.5;
            let p = y_pred.get(i, j) > 0.5;
            match (t, p) {
                (false, false) => tn += 1.0,
                (false, true) => fp += 1.0,
                (true, false) => fn_ += 1.0,
                (true, true) => tp += 1.0,
            }
        }
        match k {
            0 => tn,
            1 => fp,
            2 => fn_,
            _ => tp,
        }
    });
    ctx.finish(out)
}

fn comb2(n: f64) -> f64 {
    if n < 2.0 {
        0.0
    } else {
        n * (n - 1.0) / 2.0
    }
}

/// Adjusted Rand index from pair-counting on the label contingency.
pub fn adjusted_rand(
    y_true: &Vector,
    y_pred: &Vector,
    session: &Session,
) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if !scan_pair(&mut ctx, y_true, y_pred, "adjusted_rand") {
        return ctx.finish(f64::NAN);
    }
    inspect_classes(&mut ctx.report, y_true, &ctx.policy);
    warn_constant_target(&mut ctx, y_true, true);
    let yt = labels_of(y_true);
    let yp = labels_of(y_pred);
    let ct = unique_sorted(&yt);
    let cp = unique_sorted(&yp);
    let n = yt.len() as f64;
    let cn = comb2(n);
    if cn <= 0.0 {
        ctx.push(
            Issue::builder(IssueCode::MeaninglessFit)
                .message("ARI is undefined for n<2")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    let mut table = vec![vec![0.0; cp.len()]; ct.len()];
    for i in 0..yt.len() {
        let r = ct.iter().position(|&c| c == yt[i]);
        let c = cp.iter().position(|&c| c == yp[i]);
        if let (Some(r), Some(c)) = (r, c) {
            table[r][c] += 1.0;
        }
    }
    let mut sum_c = 0.0;
    let mut row = vec![0.0; ct.len()];
    let mut col = vec![0.0; cp.len()];
    for i in 0..ct.len() {
        for j in 0..cp.len() {
            sum_c += comb2(table[i][j]);
            row[i] += table[i][j];
            col[j] += table[i][j];
        }
    }
    let sum_a: f64 = row.iter().map(|&v| comb2(v)).sum();
    let sum_b: f64 = col.iter().map(|&v| comb2(v)).sum();
    let expected = sum_a * sum_b / cn;
    let denom = 0.5 * (sum_a + sum_b) - expected;
    if denom.abs() <= 1e-15 {
        return ctx.finish(1.0);
    }
    ctx.finish((sum_c - expected) / denom)
}

/// Mean absolute scaled error versus the one-step naive residual.
pub fn mase(y_true: &Vector, y_pred: &Vector, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if !scan_pair(&mut ctx, y_true, y_pred, "mase") {
        return ctx.finish(f64::NAN);
    }
    warn_constant_target(&mut ctx, y_true, false);
    let (_, mae, _, _, _, _) = residual_stats(y_true, y_pred);
    let mut scale = 0.0;
    let mut k = 0.0;
    for i in 1..y_true.len() {
        if y_true[i].is_finite() && y_true[i - 1].is_finite() {
            scale += (y_true[i] - y_true[i - 1]).abs();
            k += 1.0;
        }
    }
    if k <= 0.0 || scale <= 0.0 {
        ctx.push(
            Issue::builder(IssueCode::MeaninglessFit)
                .message("MASE scale (naive MAE) is zero; the series is constant")
                .meaninglessness(Meaninglessness::vacuous(
                    "MASE",
                    "the naive in-sample scale vanished",
                    "do not compare forecasts on a flat series with MASE",
                ))
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    ctx.finish(mae / (scale / k))
}

/// Symmetric MAPE: \(\mathrm{mean}\, 2|y-\hat y|/(|y|+|\hat y|)\).
pub fn smape(y_true: &Vector, y_pred: &Vector, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if !scan_pair(&mut ctx, y_true, y_pred, "smape") {
        return ctx.finish(f64::NAN);
    }
    warn_constant_target(&mut ctx, y_true, false);
    let mut s = 0.0;
    let mut k = 0.0;
    for i in 0..y_true.len() {
        if !y_true[i].is_finite() || !y_pred[i].is_finite() {
            continue;
        }
        let den = y_true[i].abs() + y_pred[i].abs();
        if den > 0.0 {
            s += 2.0 * (y_true[i] - y_pred[i]).abs() / den;
            k += 1.0;
        }
    }
    ctx.finish(if k > 0.0 { s / k } else { f64::NAN })
}

/// Brier score \(\mathrm{mean}(p-y)^2\) for binary probabilities.
pub fn brier(y_true: &Vector, p: &Vector, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if !scan_pair(&mut ctx, y_true, p, "brier") {
        return ctx.finish(f64::NAN);
    }
    warn_constant_target(&mut ctx, y_true, true);
    let mut s = 0.0;
    let mut n = 0.0;
    for i in 0..y_true.len() {
        if !y_true[i].is_finite() || !p[i].is_finite() {
            continue;
        }
        let e = p[i] - y_true[i];
        s += e * e;
        n += 1.0;
    }
    ctx.finish(if n > 0.0 { s / n } else { f64::NAN })
}

/// Average precision (binary PR-AUC via the step-function interpolation).
pub fn average_precision(
    y_true: &Vector,
    scores: &Vector,
    session: &Session,
) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if !scan_pair(&mut ctx, y_true, scores, "average_precision") {
        return ctx.finish(f64::NAN);
    }
    warn_constant_target(&mut ctx, y_true, true);
    let mut pairs: Vec<(f64, f64)> = y_true
        .as_slice()
        .iter()
        .zip(scores.as_slice())
        .filter(|(y, s)| y.is_finite() && s.is_finite())
        .map(|(y, s)| (*s, *y))
        .collect();
    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    let n_pos = pairs.iter().filter(|(_, y)| *y >= 0.5).count() as f64;
    if n_pos <= 0.0 {
        ctx.push(
            Issue::builder(IssueCode::EmptyClass)
                .message("average_precision has no positive labels")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    let mut tp = 0.0;
    let mut fp = 0.0;
    let mut ap = 0.0;
    let mut prev_rec = 0.0;
    for (_, y) in pairs {
        if y >= 0.5 {
            tp += 1.0;
        } else {
            fp += 1.0;
        }
        let rec = tp / n_pos;
        let prec = tp / (tp + fp).max(1e-15);
        ap += (rec - prev_rec) * prec;
        prev_rec = rec;
    }
    ctx.finish(ap)
}

/// Explained variance \(1 - \mathrm{Var}(y-\hat y)/\mathrm{Var}(y)\).
pub fn explained_variance(
    y_true: &Vector,
    y_pred: &Vector,
    session: &Session,
) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if !scan_pair(&mut ctx, y_true, y_pred, "explained_variance") {
        return ctx.finish(f64::NAN);
    }
    warn_constant_target(&mut ctx, y_true, false);
    let (sst, _) = sst_of(y_true);
    if sst <= ctx.policy.r2_zero_tol {
        ctx.push(
            Issue::builder(IssueCode::MeaninglessFit)
                .message("SST≈0; explained variance is undefined")
                .meaninglessness(Meaninglessness::vacuous(
                    "explained variance",
                    "a constant target makes Var(y)=0",
                    "do not report explained_variance",
                ))
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    let resid = Vector::from_iter(
        y_true
            .as_slice()
            .iter()
            .zip(y_pred.as_slice())
            .map(|(a, b)| a - b),
    );
    let (ssr, _) = sst_of(&resid);
    ctx.finish(1.0 - ssr / sst)
}

/// Hamming loss (fraction of labels that differ after rounding).
pub fn hamming(y_true: &Vector, y_pred: &Vector, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if !scan_pair(&mut ctx, y_true, y_pred, "hamming") {
        return ctx.finish(f64::NAN);
    }
    warn_constant_target(&mut ctx, y_true, true);
    let mut bad = 0.0;
    let mut n = 0.0;
    for i in 0..y_true.len() {
        if !y_true[i].is_finite() || !y_pred[i].is_finite() {
            continue;
        }
        if y_true[i].round() != y_pred[i].round() {
            bad += 1.0;
        }
        n += 1.0;
    }
    ctx.finish(if n > 0.0 { bad / n } else { f64::NAN })
}

/// Discrete mutual information of rounded labels (nats).
pub fn mutual_info(y_true: &Vector, y_pred: &Vector, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if !scan_pair(&mut ctx, y_true, y_pred, "mutual_info") {
        return ctx.finish(f64::NAN);
    }
    let _ = inspect_classes(&mut ctx.report, y_true, &ctx.policy);
    let mut joint: Vec<((i64, i64), f64)> = Vec::new();
    let mut py: Vec<(i64, f64)> = Vec::new();
    let mut pz: Vec<(i64, f64)> = Vec::new();
    let mut n = 0.0;
    for i in 0..y_true.len() {
        if !y_true[i].is_finite() || !y_pred[i].is_finite() {
            continue;
        }
        let a = y_true[i].round() as i64;
        let b = y_pred[i].round() as i64;
        bump(&mut joint, (a, b));
        bump(&mut py, a);
        bump(&mut pz, b);
        n += 1.0;
    }
    if n <= 0.0 {
        return ctx.finish(f64::NAN);
    }
    let mut mi = 0.0;
    for ((a, b), c) in &joint {
        let pab = *c / n;
        let pa = py
            .iter()
            .find(|(k, _)| *k == *a)
            .map(|(_, v)| *v / n)
            .unwrap_or(0.0);
        let pb = pz
            .iter()
            .find(|(k, _)| *k == *b)
            .map(|(_, v)| *v / n)
            .unwrap_or(0.0);
        if pab > 0.0 && pa > 0.0 && pb > 0.0 {
            mi += pab * (pab / (pa * pb)).ln();
        }
    }
    ctx.finish(mi)
}

/// Cohen's κ for rounded labels.
pub fn cohen_kappa(y_true: &Vector, y_pred: &Vector, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if !scan_pair(&mut ctx, y_true, y_pred, "cohen_kappa") {
        return ctx.finish(f64::NAN);
    }
    warn_constant_target(&mut ctx, y_true, true);
    let labs_t = labels_of(y_true);
    let labs_p = labels_of(y_pred);
    let mut classes = unique_sorted(&labs_t);
    for &c in &labs_p {
        if !classes.contains(&c) {
            classes.push(c);
        }
    }
    classes.sort_unstable();
    let k = classes.len();
    let n = labs_t.len().min(labs_p.len()) as f64;
    if n <= 0.0 || k == 0 {
        return ctx.finish(f64::NAN);
    }
    let mut row = vec![0.0; k];
    let mut col = vec![0.0; k];
    let mut agree = 0.0;
    for i in 0..labs_t.len().min(labs_p.len()) {
        let a = classes.iter().position(|&c| c == labs_t[i]).unwrap_or(0);
        let b = classes.iter().position(|&c| c == labs_p[i]).unwrap_or(0);
        row[a] += 1.0;
        col[b] += 1.0;
        if a == b {
            agree += 1.0;
        }
    }
    let po = agree / n;
    let pe: f64 = (0..k).map(|c| (row[c] / n) * (col[c] / n)).sum();
    if (1.0 - pe).abs() <= 1e-15 {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .severity(Severity::Warning)
                .message("Cohen κ chance agreement is 1; the statistic is undefined")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    ctx.finish((po - pe) / (1.0 - pe))
}

/// Matthews correlation (binary or multiclass).
pub fn matthews_corrcoef(
    y_true: &Vector,
    y_pred: &Vector,
    session: &Session,
) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if !scan_pair(&mut ctx, y_true, y_pred, "matthews_corrcoef") {
        return ctx.finish(f64::NAN);
    }
    warn_constant_target(&mut ctx, y_true, true);
    let labs_t = labels_of(y_true);
    let labs_p = labels_of(y_pred);
    let mut classes = unique_sorted(&labs_t);
    for &c in &labs_p {
        if !classes.contains(&c) {
            classes.push(c);
        }
    }
    classes.sort_unstable();
    let k = classes.len();
    let n = labs_t.len().min(labs_p.len());
    if n == 0 || k == 0 {
        return ctx.finish(f64::NAN);
    }
    let mut cm = vec![vec![0.0; k]; k];
    for i in 0..n {
        let a = classes.iter().position(|&c| c == labs_t[i]).unwrap_or(0);
        let b = classes.iter().position(|&c| c == labs_p[i]).unwrap_or(0);
        cm[a][b] += 1.0;
    }
    let s = n as f64;
    let mut c = 0.0;
    let mut t = vec![0.0; k];
    let mut p = vec![0.0; k];
    for i in 0..k {
        c += cm[i][i];
        for j in 0..k {
            t[i] += cm[i][j];
            p[j] += cm[i][j];
        }
    }
    let sum_pt: f64 = (0..k).map(|i| p[i] * t[i]).sum();
    let sum_p2: f64 = p.iter().map(|v| v * v).sum();
    let sum_t2: f64 = t.iter().map(|v| v * v).sum();
    let den = ((s * s - sum_p2) * (s * s - sum_t2)).sqrt();
    if den <= 1e-18 {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .severity(Severity::Warning)
                .message("MCC denominator vanished")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    ctx.finish((c * s - sum_pt) / den)
}

/// Calinski–Harabasz variance-ratio index.
pub fn calinski_harabasz(x: &Matrix, labels: &Vector, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, x, None, &ctx.policy);
    let labs = labels_of(labels);
    let classes = unique_sorted(&labs);
    let k = classes.len();
    let n = x.nrows().min(labels.len());
    if k < 2 || n <= k {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .severity(Severity::Warning)
                .message(format!(
                    "Calinski–Harabasz needs ≥2 clusters and n>k; k={k} n={n}"
                ))
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    let p = x.ncols();
    let mut global = vec![0.0; p];
    for i in 0..n {
        for j in 0..p {
            global[j] += x.get(i, j);
        }
    }
    for g in global.iter_mut() {
        *g /= n as f64;
    }
    let mut ssb = 0.0;
    let mut ssw = 0.0;
    for (ci, &lab) in classes.iter().enumerate() {
        let mut mu = vec![0.0; p];
        let mut nk = 0.0;
        for i in 0..n {
            if labs[i] == lab {
                nk += 1.0;
                for j in 0..p {
                    mu[j] += x.get(i, j);
                }
            }
        }
        if nk <= 0.0 {
            let _ = ci;
            continue;
        }
        for j in 0..p {
            mu[j] /= nk;
            let d = mu[j] - global[j];
            ssb += nk * d * d;
        }
        for i in 0..n {
            if labs[i] != lab {
                continue;
            }
            for j in 0..p {
                let d = x.get(i, j) - mu[j];
                ssw += d * d;
            }
        }
    }
    if ssw <= 1e-18 {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .severity(Severity::Warning)
                .message("Calinski–Harabasz within-cluster SS is 0")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    let ch = (ssb / (k as f64 - 1.0)) / (ssw / (n as f64 - k as f64));
    ctx.finish(ch)
}

/// Davies–Bouldin index (lower is tighter, better-separated clusters).
pub fn davies_bouldin(x: &Matrix, labels: &Vector, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, x, None, &ctx.policy);
    let labs = labels_of(labels);
    let classes = unique_sorted(&labs);
    let k = classes.len();
    let n = x.nrows().min(labels.len());
    if k < 2 {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .severity(Severity::Warning)
                .message("Davies–Bouldin needs at least two clusters")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    let p = x.ncols();
    let mut mu = vec![vec![0.0; p]; k];
    let mut nk = vec![0.0; k];
    for i in 0..n {
        if let Some(c) = classes.iter().position(|&lab| lab == labs[i]) {
            nk[c] += 1.0;
            for j in 0..p {
                mu[c][j] += x.get(i, j);
            }
        }
    }
    for c in 0..k {
        if nk[c] > 0.0 {
            for j in 0..p {
                mu[c][j] /= nk[c];
            }
        }
    }
    let mut s = vec![0.0; k];
    for i in 0..n {
        if let Some(c) = classes.iter().position(|&lab| lab == labs[i]) {
            let mut d2 = 0.0;
            for j in 0..p {
                let d = x.get(i, j) - mu[c][j];
                d2 += d * d;
            }
            s[c] += d2.sqrt();
        }
    }
    for c in 0..k {
        if nk[c] > 0.0 {
            s[c] /= nk[c];
        }
    }
    let mut acc = 0.0;
    for i in 0..k {
        let mut best = 0.0;
        for j in 0..k {
            if i == j {
                continue;
            }
            let mut dij = 0.0;
            for t in 0..p {
                let d = mu[i][t] - mu[j][t];
                dij += d * d;
            }
            dij = dij.sqrt();
            if dij <= 1e-18 {
                continue;
            }
            let r = (s[i] + s[j]) / dij;
            if r > best {
                best = r;
            }
        }
        acc += best;
    }
    ctx.finish(acc / k as f64)
}

/// Pinball (quantile) loss at level `tau`.
pub fn pinball_loss(
    y_true: &Vector,
    y_pred: &Vector,
    tau: f64,
    session: &Session,
) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if !scan_pair(&mut ctx, y_true, y_pred, "pinball_loss") {
        return ctx.finish(f64::NAN);
    }
    warn_constant_target(&mut ctx, y_true, false);
    let t = if tau.is_finite() && tau > 0.0 && tau < 1.0 {
        tau
    } else {
        ctx.push(
            Issue::builder(IssueCode::InvalidWeight)
                .severity(Severity::Warning)
                .message(format!("pinball τ={tau} is not in (0,1); using 0.5"))
                .build(),
        );
        0.5
    };
    let mut s = 0.0;
    let mut n = 0.0;
    for i in 0..y_true.len() {
        if !y_true[i].is_finite() || !y_pred[i].is_finite() {
            continue;
        }
        let e = y_true[i] - y_pred[i];
        s += if e >= 0.0 { t * e } else { (t - 1.0) * e };
        n += 1.0;
    }
    ctx.finish(if n > 0.0 { s / n } else { f64::NAN })
}

/// Reliability-diagram bins (sklearn `calibration_curve`).
#[derive(Clone, Debug)]
pub struct CalibrationCurve {
    /// Fraction of positives in each occupied bin.
    pub prob_true: Vector,
    /// Mean predicted probability in each occupied bin.
    pub prob_pred: Vector,
    /// Occupancy of each occupied bin.
    pub counts: Vector,
}

/// Equal-width calibration curve on \([0,1]\) predicted probabilities.
///
/// Bin count is not identification `p`. A constant score or a single occupied
/// bin cannot show miscalibration across the score range.
pub fn calibration_curve(
    y_true: &Vector,
    prob: &Vector,
    n_bins: usize,
    session: &Session,
) -> Result<Qualified<CalibrationCurve>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if !scan_pair(&mut ctx, y_true, prob, "calibration_curve") {
        return ctx.finish(CalibrationCurve {
            prob_true: Vector::zeros(0),
            prob_pred: Vector::zeros(0),
            counts: Vector::zeros(0),
        });
    }
    warn_constant_target(&mut ctx, y_true, true);
    let bins = n_bins.max(2);
    if n_bins < 2 {
        ctx.push(
            Issue::builder(IssueCode::InvalidWeight)
                .severity(Severity::Warning)
                .message(format!("calibration_curve n_bins={n_bins} < 2; using 2"))
                .build(),
        );
    }
    let mut sum_y = vec![0.0; bins];
    let mut sum_p = vec![0.0; bins];
    let mut cnt = vec![0.0; bins];
    let mut out_of_range = 0usize;
    for i in 0..y_true.len() {
        let p = prob[i];
        if !p.is_finite() || !y_true[i].is_finite() {
            continue;
        }
        if p < 0.0 || p > 1.0 {
            out_of_range += 1;
        }
        let pc = p.clamp(0.0, 1.0);
        let mut b = (pc * bins as f64).floor() as usize;
        if b >= bins {
            b = bins - 1;
        }
        sum_y[b] += y_true[i];
        sum_p[b] += pc;
        cnt[b] += 1.0;
    }
    if out_of_range > 0 {
        ctx.push(
            Issue::builder(IssueCode::InvalidWeight)
                .severity(Severity::Warning)
                .message(format!(
                    "{out_of_range} predicted probabilities were outside [0,1] and were clipped"
                ))
                .build(),
        );
    }
    let mut pt = Vec::new();
    let mut pp = Vec::new();
    let mut cc = Vec::new();
    for b in 0..bins {
        if cnt[b] <= 0.0 {
            continue;
        }
        pt.push(sum_y[b] / cnt[b]);
        pp.push(sum_p[b] / cnt[b]);
        cc.push(cnt[b]);
    }
    if pt.len() <= 1 {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .severity(Severity::Warning)
                .message("calibration curve occupied fewer than two bins")
                .build(),
        );
    }
    ctx.finish(CalibrationCurve {
        prob_true: Vector::from_slice(&pt),
        prob_pred: Vector::from_slice(&pp),
        counts: Vector::from_slice(&cc),
    })
}

/// ROC curve points (sklearn `roc_curve`).
#[derive(Clone, Debug)]
pub struct RocCurve {
    /// False-positive rates, high-score first then descending thresholds.
    pub fpr: Vector,
    /// True-positive rates.
    pub tpr: Vector,
    /// Score thresholds (one per point, plus a leading `+∞` sentinel dropped).
    pub thresholds: Vector,
}

/// Binary ROC curve from scores. Constant `y` is vacuous.
pub fn roc_curve(
    y_true: &Vector,
    scores: &Vector,
    session: &Session,
) -> Result<Qualified<RocCurve>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if !scan_pair(&mut ctx, y_true, scores, "roc_curve") {
        return ctx.finish(RocCurve {
            fpr: Vector::zeros(0),
            tpr: Vector::zeros(0),
            thresholds: Vector::zeros(0),
        });
    }
    inspect_classes(&mut ctx.report, y_true, &ctx.policy);
    warn_constant_target(&mut ctx, y_true, true);
    let yt = labels_of(y_true);
    let classes = unique_sorted(&yt);
    if classes.len() != 2 {
        ctx.push(
            Issue::builder(IssueCode::MeaninglessFit)
                .message(format!(
                    "roc_curve is defined for two classes; found {}",
                    classes.len()
                ))
                .meaninglessness(Meaninglessness::vacuous(
                    "ROC curve",
                    "the curve is a two-class functional",
                    "use a one-vs-rest reduction",
                ))
                .build(),
        );
        return ctx.finish(RocCurve {
            fpr: Vector::zeros(0),
            tpr: Vector::zeros(0),
            thresholds: Vector::zeros(0),
        });
    }
    let pos = classes[1];
    let mut pairs: Vec<(f64, bool)> = (0..yt.len())
        .filter(|&i| y_true[i].is_finite() && scores[i].is_finite())
        .map(|i| (scores[i], yt[i] == pos))
        .collect();
    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    let n_pos = pairs.iter().filter(|p| p.1).count() as f64;
    let n_neg = pairs.len() as f64 - n_pos;
    if n_pos <= 0.0 || n_neg <= 0.0 {
        ctx.push(
            Issue::builder(IssueCode::EmptyClass)
                .message("roc_curve needs both a positive and a negative class")
                .build(),
        );
        return ctx.finish(RocCurve {
            fpr: Vector::zeros(0),
            tpr: Vector::zeros(0),
            thresholds: Vector::zeros(0),
        });
    }
    let mut fpr = vec![0.0];
    let mut tpr = vec![0.0];
    let mut thr = vec![f64::INFINITY];
    let mut tp: f64 = 0.0;
    let mut fp: f64 = 0.0;
    let mut i = 0;
    while i < pairs.len() {
        let t = pairs[i].0;
        while i < pairs.len() && (pairs[i].0 - t).abs() <= 0.0 {
            if pairs[i].1 {
                tp += 1.0;
            } else {
                fp += 1.0;
            }
            i += 1;
        }
        fpr.push(fp / n_neg);
        tpr.push(tp / n_pos);
        thr.push(t);
    }
    ctx.finish(RocCurve {
        fpr: Vector::from_slice(&fpr),
        tpr: Vector::from_slice(&tpr),
        thresholds: Vector::from_slice(&thr),
    })
}

/// Precision–recall curve (sklearn `precision_recall_curve`).
#[derive(Clone, Debug)]
pub struct PrecisionRecallCurve {
    /// Precision at each threshold.
    pub precision: Vector,
    /// Recall at each threshold.
    pub recall: Vector,
    /// Score thresholds.
    pub thresholds: Vector,
}

/// Binary precision–recall curve from scores.
pub fn precision_recall_curve(
    y_true: &Vector,
    scores: &Vector,
    session: &Session,
) -> Result<Qualified<PrecisionRecallCurve>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if !scan_pair(&mut ctx, y_true, scores, "precision_recall_curve") {
        return ctx.finish(PrecisionRecallCurve {
            precision: Vector::zeros(0),
            recall: Vector::zeros(0),
            thresholds: Vector::zeros(0),
        });
    }
    inspect_classes(&mut ctx.report, y_true, &ctx.policy);
    warn_constant_target(&mut ctx, y_true, true);
    let yt = labels_of(y_true);
    let classes = unique_sorted(&yt);
    if classes.len() != 2 {
        ctx.push(
            Issue::builder(IssueCode::MeaninglessFit)
                .message(format!(
                    "precision_recall_curve is defined for two classes; found {}",
                    classes.len()
                ))
                .meaninglessness(Meaninglessness::vacuous(
                    "precision–recall curve",
                    "the curve is a two-class functional",
                    "use a one-vs-rest reduction",
                ))
                .build(),
        );
        return ctx.finish(PrecisionRecallCurve {
            precision: Vector::zeros(0),
            recall: Vector::zeros(0),
            thresholds: Vector::zeros(0),
        });
    }
    let pos = classes[1];
    let mut pairs: Vec<(f64, bool)> = (0..yt.len())
        .filter(|&i| y_true[i].is_finite() && scores[i].is_finite())
        .map(|i| (scores[i], yt[i] == pos))
        .collect();
    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    let n_pos = pairs.iter().filter(|p| p.1).count() as f64;
    if n_pos <= 0.0 {
        ctx.push(
            Issue::builder(IssueCode::EmptyClass)
                .message("precision_recall_curve has no positive labels")
                .build(),
        );
        return ctx.finish(PrecisionRecallCurve {
            precision: Vector::zeros(0),
            recall: Vector::zeros(0),
            thresholds: Vector::zeros(0),
        });
    }
    let mut prec = Vec::new();
    let mut rec = Vec::new();
    let mut thr = Vec::new();
    let mut tp: f64 = 0.0;
    let mut fp: f64 = 0.0;
    let mut i = 0;
    while i < pairs.len() {
        let t = pairs[i].0;
        while i < pairs.len() && (pairs[i].0 - t).abs() <= 0.0 {
            if pairs[i].1 {
                tp += 1.0;
            } else {
                fp += 1.0;
            }
            i += 1;
        }
        prec.push(tp / (tp + fp).max(1e-12));
        rec.push(tp / n_pos);
        thr.push(t);
    }
    ctx.finish(PrecisionRecallCurve {
        precision: Vector::from_slice(&prec),
        recall: Vector::from_slice(&rec),
        thresholds: Vector::from_slice(&thr),
    })
}

/// Mean hinge loss \(\max(0, 1 - y s)\) with \(y\in\{\pm 1\}\).
pub fn hinge_loss(y_true: &Vector, scores: &Vector, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if !scan_pair(&mut ctx, y_true, scores, "hinge_loss") {
        return ctx.finish(f64::NAN);
    }
    warn_constant_target(&mut ctx, y_true, true);
    let mut acc = 0.0;
    let mut n = 0.0;
    for i in 0..y_true.len().min(scores.len()) {
        if !y_true[i].is_finite() || !scores[i].is_finite() {
            continue;
        }
        let y = if y_true[i] > 0.5 { 1.0 } else { -1.0 };
        acc += (1.0 - y * scores[i]).max(0.0);
        n += 1.0;
    }
    ctx.finish(if n > 0.0 { acc / n } else { f64::NAN })
}

/// Zero–one loss \(1-\mathrm{accuracy}\).
pub fn zero_one_loss(
    y_true: &Vector,
    y_pred: &Vector,
    session: &Session,
) -> Result<Qualified<f64>> {
    let q = accuracy(y_true, y_pred, session)?;
    let mut ctx = FitCtx::with_session(session.child("zo"));
    for issue in q.report.issues() {
        ctx.push(issue.clone());
    }
    ctx.finish(1.0 - q.value)
}

/// Binary Jaccard index (sklearn `jaccard_score`).
pub fn jaccard_score(
    y_true: &Vector,
    y_pred: &Vector,
    session: &Session,
) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if !scan_pair(&mut ctx, y_true, y_pred, "jaccard_score") {
        return ctx.finish(f64::NAN);
    }
    inspect_classes(&mut ctx.report, y_true, &ctx.policy);
    warn_constant_target(&mut ctx, y_true, true);
    let mut inter = 0.0;
    let mut union = 0.0;
    for i in 0..y_true.len().min(y_pred.len()) {
        let a = y_true[i] > 0.5;
        let b = y_pred[i] > 0.5;
        if a && b {
            inter += 1.0;
        }
        if a || b {
            union += 1.0;
        }
    }
    if union <= 0.0 {
        ctx.push(
            Issue::builder(IssueCode::EmptyClass)
                .severity(Severity::Warning)
                .message("jaccard_score union is empty; both sides are all-negative")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    ctx.finish(inter / union)
}

fn tweedie_deviance(y: f64, mu: f64, power: f64) -> f64 {
    let mu = mu.max(1e-12);
    let y = y.max(0.0);
    if (power - 0.0).abs() < 1e-12 {
        let e = y - mu;
        return e * e;
    }
    if (power - 1.0).abs() < 1e-12 {
        let yl = if y > 0.0 { y * (y / mu).ln() } else { 0.0 };
        return 2.0 * (yl - (y - mu));
    }
    if (power - 2.0).abs() < 1e-12 {
        let yl = if y > 0.0 { (mu / y).ln() } else { mu.ln() };
        return 2.0 * (yl + y / mu - 1.0);
    }
    let p = power;
    2.0 * (y.max(0.0).powf(2.0 - p) / ((1.0 - p) * (2.0 - p)) - y * mu.powf(1.0 - p) / (1.0 - p)
        + mu.powf(2.0 - p) / (2.0 - p))
}

/// Mean Tweedie deviance (sklearn `mean_tweedie_deviance` / Poisson).
pub fn mean_poisson_deviance(
    y_true: &Vector,
    y_pred: &Vector,
    session: &Session,
) -> Result<Qualified<f64>> {
    mean_tweedie_deviance(y_true, y_pred, 1.0, session)
}

/// Mean Tweedie deviance at power `p`.
pub fn mean_tweedie_deviance(
    y_true: &Vector,
    y_pred: &Vector,
    power: f64,
    session: &Session,
) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if !scan_pair(&mut ctx, y_true, y_pred, "mean_tweedie_deviance") {
        return ctx.finish(f64::NAN);
    }
    if !power.is_finite() {
        ctx.push(
            Issue::builder(IssueCode::InvalidWeight)
                .severity(Severity::Warning)
                .message(format!("Tweedie power={power} is not finite; using 0"))
                .build(),
        );
    }
    let p = if power.is_finite() { power } else { 0.0 };
    let mut acc = 0.0;
    let mut n = 0.0;
    for i in 0..y_true.len().min(y_pred.len()) {
        if !y_true[i].is_finite() || !y_pred[i].is_finite() {
            continue;
        }
        if y_true[i] < 0.0 || (p >= 1.0 && y_pred[i] <= 0.0) {
            ctx.push(
                Issue::builder(IssueCode::InvalidWeight)
                    .severity(Severity::Warning)
                    .message(format!(
                        "Tweedie deviance skipped y={}, μ={} at power {p}",
                        y_true[i], y_pred[i]
                    ))
                    .build(),
            );
            continue;
        }
        acc += tweedie_deviance(y_true[i], y_pred[i], p);
        n += 1.0;
    }
    ctx.finish(if n > 0.0 { acc / n } else { f64::NAN })
}

/// \(D^2\) Tweedie score (sklearn `d2_tweedie_score`).
pub fn d2_tweedie_score(
    y_true: &Vector,
    y_pred: &Vector,
    power: f64,
    session: &Session,
) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    let d_model = mean_tweedie_deviance(y_true, y_pred, power, &session.child("d_model"))?;
    for issue in d_model.report.issues() {
        if issue.code == IssueCode::MeaninglessFit {
            continue;
        }
        ctx.push(issue.clone());
    }
    let mean = y_true.mean();
    let null = Vector::from_iter((0..y_true.len()).map(|_| mean));
    let d_null = mean_tweedie_deviance(y_true, &null, power, &session.child("d_null"))?;
    if !d_null.value.is_finite() || d_null.value.abs() <= 1e-18 {
        ctx.push(
            Issue::builder(IssueCode::R2IsZero)
                .message("D² Tweedie null deviance is ~0; the score is undefined")
                .compromise(NumericalCompromise::new(
                    "positive null Tweedie deviance",
                    "D² set to NaN",
                    "the intercept-only deviance vanished",
                    "do not read a missing D² as a perfect fit",
                ))
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    ctx.finish(1.0 - d_model.value / d_null.value)
}

/// Balanced accuracy: mean of per-class recalls (sklearn `balanced_accuracy_score`).
pub fn balanced_accuracy(
    y_true: &Vector,
    y_pred: &Vector,
    session: &Session,
) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if !scan_pair(&mut ctx, y_true, y_pred, "balanced_accuracy") {
        return ctx.finish(f64::NAN);
    }
    inspect_classes(&mut ctx.report, y_true, &ctx.policy);
    warn_constant_target(&mut ctx, y_true, true);
    let yt = labels_of(y_true);
    let yp = labels_of(y_pred);
    let classes = unique_sorted(&yt);
    if classes.is_empty() {
        return ctx.finish(f64::NAN);
    }
    let mut acc = 0.0;
    for &c in &classes {
        let mut tp = 0.0;
        let mut n = 0.0;
        for i in 0..yt.len().min(yp.len()) {
            if yt[i] == c {
                n += 1.0;
                if yp[i] == c {
                    tp += 1.0;
                }
            }
        }
        if n > 0.0 {
            acc += tp / n;
        }
    }
    ctx.finish(acc / classes.len() as f64)
}

/// Mean Gamma deviance (Tweedie power 2).
pub fn mean_gamma_deviance(
    y_true: &Vector,
    y_pred: &Vector,
    session: &Session,
) -> Result<Qualified<f64>> {
    mean_tweedie_deviance(y_true, y_pred, 2.0, session)
}

/// \(D^2\) absolute-error score (sklearn `d2_absolute_error_score`).
pub fn d2_absolute_error_score(
    y_true: &Vector,
    y_pred: &Vector,
    session: &Session,
) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if !scan_pair(&mut ctx, y_true, y_pred, "d2_absolute_error_score") {
        return ctx.finish(f64::NAN);
    }
    warn_constant_target(&mut ctx, y_true, false);
    let mut xs: Vec<f64> = y_true
        .as_slice()
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .collect();
    if xs.is_empty() {
        return ctx.finish(f64::NAN);
    }
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = xs.len() / 2;
    let med = if xs.len() % 2 == 0 {
        0.5 * (xs[mid - 1] + xs[mid])
    } else {
        xs[mid]
    };
    let mut mae_m = 0.0;
    let mut mae_n = 0.0;
    let mut n = 0.0;
    for i in 0..y_true.len().min(y_pred.len()) {
        if !y_true[i].is_finite() || !y_pred[i].is_finite() {
            continue;
        }
        mae_m += (y_true[i] - y_pred[i]).abs();
        mae_n += (y_true[i] - med).abs();
        n += 1.0;
    }
    if n <= 0.0 || mae_n <= 1e-18 {
        ctx.push(
            Issue::builder(IssueCode::R2IsZero)
                .message("D² absolute-error null MAE is ~0; the score is undefined")
                .compromise(NumericalCompromise::new(
                    "positive null MAE to the median",
                    "D² set to NaN",
                    "the median-only absolute error vanished",
                    "do not read a missing D² as a perfect fit",
                ))
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    ctx.finish(1.0 - mae_m / mae_n)
}

fn dcg_at(rels: &[f64]) -> f64 {
    let mut s = 0.0;
    for (i, &r) in rels.iter().enumerate() {
        let gain = (2.0_f64).powf(r) - 1.0;
        s += gain / ((i as f64) + 2.0).log2();
    }
    s
}

/// Discounted cumulative gain (sklearn `dcg_score`).
pub fn dcg_score(y_true: &Vector, y_score: &Vector, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if !scan_pair(&mut ctx, y_true, y_score, "dcg_score") {
        return ctx.finish(f64::NAN);
    }
    let mut pairs: Vec<(f64, f64)> = (0..y_true.len().min(y_score.len()))
        .filter(|&i| y_true[i].is_finite() && y_score[i].is_finite())
        .map(|i| (y_score[i], y_true[i]))
        .collect();
    if pairs.is_empty() {
        return ctx.finish(f64::NAN);
    }
    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    let rels: Vec<f64> = pairs.iter().map(|p| p.1).collect();
    ctx.finish(dcg_at(&rels))
}

/// Normalized DCG (sklearn `ndcg_score`).
pub fn ndcg_score(y_true: &Vector, y_score: &Vector, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    let dcg = dcg_score(y_true, y_score, &session.child("dcg"))?;
    for issue in dcg.report.issues() {
        if issue.code == IssueCode::MeaninglessFit {
            continue;
        }
        ctx.push(issue.clone());
    }
    let mut ideal: Vec<f64> = y_true
        .as_slice()
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .collect();
    ideal.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let idcg = dcg_at(&ideal);
    if !idcg.is_finite() || idcg.abs() <= 1e-18 {
        ctx.push(
            Issue::builder(IssueCode::R2IsZero)
                .message("nDCG ideal DCG is ~0; the score is undefined")
                .compromise(NumericalCompromise::new(
                    "positive ideal DCG",
                    "nDCG set to NaN",
                    "all relevances are zero",
                    "do not read a missing nDCG as a perfect ranking",
                ))
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    ctx.finish(dcg.value / idcg)
}

fn entropy_counts(counts: &[f64]) -> f64 {
    let n: f64 = counts.iter().sum();
    if n <= 0.0 {
        return 0.0;
    }
    let mut h = 0.0;
    for &c in counts {
        if c > 0.0 {
            let p = c / n;
            h -= p * p.ln();
        }
    }
    h
}

fn contingency(yt: &[i64], yp: &[i64]) -> (Vec<Vec<f64>>, Vec<f64>, Vec<f64>, f64) {
    let ct = unique_sorted(yt);
    let cp = unique_sorted(yp);
    let mut table = vec![vec![0.0; cp.len()]; ct.len()];
    let mut n = 0.0;
    for i in 0..yt.len().min(yp.len()) {
        let r = ct.iter().position(|&c| c == yt[i]);
        let c = cp.iter().position(|&c| c == yp[i]);
        if let (Some(r), Some(c)) = (r, c) {
            table[r][c] += 1.0;
            n += 1.0;
        }
    }
    let mut row = vec![0.0; ct.len()];
    let mut col = vec![0.0; cp.len()];
    for i in 0..ct.len() {
        for j in 0..cp.len() {
            row[i] += table[i][j];
            col[j] += table[i][j];
        }
    }
    (table, row, col, n)
}

fn mi_from_table(table: &[Vec<f64>], row: &[f64], col: &[f64], n: f64) -> f64 {
    if n <= 0.0 {
        return 0.0;
    }
    let mut mi = 0.0;
    for i in 0..row.len() {
        for j in 0..col.len() {
            let nij = table[i][j];
            if nij <= 0.0 || row[i] <= 0.0 || col[j] <= 0.0 {
                continue;
            }
            mi += (nij / n) * (n * nij / (row[i] * col[j])).ln();
        }
    }
    mi
}

fn ln_comb(n: f64, k: f64) -> f64 {
    if k < -1e-12 || k > n + 1e-12 {
        return f64::NEG_INFINITY;
    }
    let k = k.clamp(0.0, n);
    crate::special::ln_gamma(n + 1.0)
        - crate::special::ln_gamma(k + 1.0)
        - crate::special::ln_gamma(n - k + 1.0)
}

fn expected_mi(row: &[f64], col: &[f64], n: f64) -> f64 {
    if n <= 0.0 {
        return 0.0;
    }
    let mut emi = 0.0;
    for &ai in row {
        for &bj in col {
            if ai <= 0.0 || bj <= 0.0 {
                continue;
            }
            let lo = (ai + bj - n).max(1.0);
            let hi = ai.min(bj);
            let mut nij = lo.ceil();
            while nij <= hi + 1e-9 {
                let log_p = ln_comb(ai, nij) + ln_comb(n - ai, bj - nij) - ln_comb(n, bj);
                if log_p.is_finite() {
                    let p = log_p.exp();
                    let term = (nij / n) * (n * nij / (ai * bj)).ln();
                    if term.is_finite() {
                        emi += p * term;
                    }
                }
                nij += 1.0;
            }
        }
    }
    emi
}

/// Homogeneity of a clustering (sklearn `homogeneity_score`).
pub fn homogeneity_score(
    y_true: &Vector,
    y_pred: &Vector,
    session: &Session,
) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if !scan_pair(&mut ctx, y_true, y_pred, "homogeneity_score") {
        return ctx.finish(f64::NAN);
    }
    inspect_classes(&mut ctx.report, y_true, &ctx.policy);
    warn_constant_target(&mut ctx, y_true, true);
    let yt = labels_of(y_true);
    let yp = labels_of(y_pred);
    let (table, row, col, n) = contingency(&yt, &yp);
    let hc = entropy_counts(&row);
    if hc <= 1e-18 {
        return ctx.finish(1.0);
    }
    let mi = mi_from_table(&table, &row, &col, n);
    ctx.finish((mi / hc).clamp(0.0, 1.0))
}

/// Completeness of a clustering (sklearn `completeness_score`).
pub fn completeness_score(
    y_true: &Vector,
    y_pred: &Vector,
    session: &Session,
) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if !scan_pair(&mut ctx, y_true, y_pred, "completeness_score") {
        return ctx.finish(f64::NAN);
    }
    inspect_classes(&mut ctx.report, y_true, &ctx.policy);
    warn_constant_target(&mut ctx, y_true, true);
    let yt = labels_of(y_true);
    let yp = labels_of(y_pred);
    let (table, row, col, n) = contingency(&yt, &yp);
    let hk = entropy_counts(&col);
    if hk <= 1e-18 {
        return ctx.finish(1.0);
    }
    let mi = mi_from_table(&table, &row, &col, n);
    ctx.finish((mi / hk).clamp(0.0, 1.0))
}

/// V-measure (harmonic mean of homogeneity and completeness).
pub fn v_measure_score(
    y_true: &Vector,
    y_pred: &Vector,
    session: &Session,
) -> Result<Qualified<f64>> {
    let h = homogeneity_score(y_true, y_pred, &session.child("h"))?;
    let c = completeness_score(y_true, y_pred, &session.child("c"))?;
    let mut ctx = FitCtx::with_session(session.clone());
    for issue in h.report.issues().iter().chain(c.report.issues()) {
        if issue.code == IssueCode::MeaninglessFit {
            continue;
        }
        ctx.push(issue.clone());
    }
    let den = h.value + c.value;
    ctx.finish(if den.abs() <= 1e-18 {
        0.0
    } else {
        2.0 * h.value * c.value / den
    })
}

/// Normalized mutual information (sklearn `normalized_mutual_info_score`).
pub fn normalized_mutual_info_score(
    y_true: &Vector,
    y_pred: &Vector,
    session: &Session,
) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if !scan_pair(&mut ctx, y_true, y_pred, "normalized_mutual_info_score") {
        return ctx.finish(f64::NAN);
    }
    inspect_classes(&mut ctx.report, y_true, &ctx.policy);
    warn_constant_target(&mut ctx, y_true, true);
    let yt = labels_of(y_true);
    let yp = labels_of(y_pred);
    let (table, row, col, n) = contingency(&yt, &yp);
    let hc = entropy_counts(&row);
    let hk = entropy_counts(&col);
    let mi = mi_from_table(&table, &row, &col, n);
    let den = (hc * hk).sqrt();
    if den <= 1e-18 {
        return ctx.finish(1.0);
    }
    ctx.finish((mi / den).clamp(0.0, 1.0))
}

/// Adjusted mutual information (sklearn `adjusted_mutual_info_score`).
pub fn adjusted_mutual_info_score(
    y_true: &Vector,
    y_pred: &Vector,
    session: &Session,
) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if !scan_pair(&mut ctx, y_true, y_pred, "adjusted_mutual_info_score") {
        return ctx.finish(f64::NAN);
    }
    inspect_classes(&mut ctx.report, y_true, &ctx.policy);
    warn_constant_target(&mut ctx, y_true, true);
    let yt = labels_of(y_true);
    let yp = labels_of(y_pred);
    let (table, row, col, n) = contingency(&yt, &yp);
    let hc = entropy_counts(&row);
    let hk = entropy_counts(&col);
    let mi = mi_from_table(&table, &row, &col, n);
    let emi = expected_mi(&row, &col, n);
    let den = 0.5 * (hc + hk) - emi;
    if den.abs() <= 1e-12 {
        return ctx.finish(1.0);
    }
    ctx.finish(((mi - emi) / den).clamp(-1.0, 1.0))
}

/// Fowlkes–Mallows index (sklearn `fowlkes_mallows_score`).
pub fn fowlkes_mallows_score(
    y_true: &Vector,
    y_pred: &Vector,
    session: &Session,
) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if !scan_pair(&mut ctx, y_true, y_pred, "fowlkes_mallows_score") {
        return ctx.finish(f64::NAN);
    }
    inspect_classes(&mut ctx.report, y_true, &ctx.policy);
    warn_constant_target(&mut ctx, y_true, true);
    let yt = labels_of(y_true);
    let yp = labels_of(y_pred);
    let (table, row, col, _) = contingency(&yt, &yp);
    let mut tp = 0.0;
    for r in &table {
        for &v in r {
            tp += comb2(v);
        }
    }
    let fp: f64 = col.iter().map(|&v| comb2(v)).sum::<f64>() - tp;
    let fn_: f64 = row.iter().map(|&v| comb2(v)).sum::<f64>() - tp;
    let den = ((tp + fp) * (tp + fn_)).sqrt();
    if den <= 1e-18 {
        ctx.push(
            Issue::builder(IssueCode::PValueUnreliable)
                .message("Fowlkes–Mallows denominator vanished; no pair is co-clustered")
                .build(),
        );
        return ctx.finish(0.0);
    }
    ctx.finish(tp / den)
}

/// Mean log-cosh error (sklearn `log_cosh`).
pub fn log_cosh_error(
    y_true: &Vector,
    y_pred: &Vector,
    session: &Session,
) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if !scan_pair(&mut ctx, y_true, y_pred, "log_cosh_error") {
        return ctx.finish(f64::NAN);
    }
    warn_constant_target(&mut ctx, y_true, false);
    let mut acc = 0.0;
    let mut n = 0.0;
    for i in 0..y_true.len().min(y_pred.len()) {
        if !y_true[i].is_finite() || !y_pred[i].is_finite() {
            continue;
        }
        let x = (y_pred[i] - y_true[i]).abs();
        acc += x + ((-2.0 * x).exp() + 1.0).ln() - std::f64::consts::LN_2;
        n += 1.0;
    }
    ctx.finish(if n > 0.0 { acc / n } else { f64::NAN })
}

/// Mean squared logarithmic error (sklearn `mean_squared_log_error`).
pub fn mean_squared_log_error(
    y_true: &Vector,
    y_pred: &Vector,
    session: &Session,
) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if !scan_pair(&mut ctx, y_true, y_pred, "mean_squared_log_error") {
        return ctx.finish(f64::NAN);
    }
    warn_constant_target(&mut ctx, y_true, false);
    let mut acc = 0.0;
    let mut n = 0.0;
    for i in 0..y_true.len().min(y_pred.len()) {
        if !y_true[i].is_finite() || !y_pred[i].is_finite() {
            continue;
        }
        if y_true[i] < 0.0 || y_pred[i] < 0.0 {
            ctx.push(
                Issue::builder(IssueCode::InvalidWeight)
                    .severity(Severity::Warning)
                    .message(format!(
                        "MSLE skipped negative pair y={}, ŷ={}",
                        y_true[i], y_pred[i]
                    ))
                    .build(),
            );
            continue;
        }
        let e = (1.0 + y_true[i]).ln() - (1.0 + y_pred[i]).ln();
        acc += e * e;
        n += 1.0;
    }
    ctx.finish(if n > 0.0 { acc / n } else { f64::NAN })
}

/// Root mean squared error.
pub fn root_mean_squared_error(
    y_true: &Vector,
    y_pred: &Vector,
    session: &Session,
) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if !scan_pair(&mut ctx, y_true, y_pred, "root_mean_squared_error") {
        return ctx.finish(f64::NAN);
    }
    warn_constant_target(&mut ctx, y_true, false);
    let (_, _, _, _, sse, n) = residual_stats(y_true, y_pred);
    ctx.finish(if n > 0 {
        (sse / n as f64).sqrt()
    } else {
        f64::NAN
    })
}

/// Top-k accuracy from a score matrix (sklearn `top_k_accuracy_score`).
///
/// Column `j` is the score of class `j` after labels are mapped onto
/// `0..ncols`. `k` is not identification `p`.
pub fn top_k_accuracy(
    y_true: &Vector,
    scores: &Matrix,
    k: usize,
    session: &Session,
) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, scores, None, &ctx.policy);
    warn_constant_target(&mut ctx, y_true, true);
    if y_true.len() != scores.nrows() {
        ctx.push(
            Issue::builder(IssueCode::DimensionMismatch)
                .message(format!(
                    "top_k_accuracy y.len()={} scores.nrows()={}",
                    y_true.len(),
                    scores.nrows()
                ))
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    let kk = if k >= 1 {
        k
    } else {
        ctx.push(
            Issue::builder(IssueCode::InvalidWeight)
                .severity(Severity::Warning)
                .message(format!("top_k_accuracy k={k} < 1; using 1"))
                .build(),
        );
        1
    };
    let p = scores.ncols().max(1);
    let kk = kk.min(p);
    let mut hit = 0.0;
    let mut n = 0.0;
    for i in 0..y_true.len() {
        if !y_true[i].is_finite() {
            continue;
        }
        let lab = y_true[i].round() as i64;
        let want = if lab >= 0 && (lab as usize) < p {
            lab as usize
        } else {
            continue;
        };
        let mut idx: Vec<usize> = (0..p).collect();
        idx.sort_by(|a, b| {
            scores
                .get(i, *b)
                .partial_cmp(&scores.get(i, *a))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        if idx.iter().take(kk).any(|&j| j == want) {
            hit += 1.0;
        }
        n += 1.0;
    }
    ctx.finish(if n > 0.0 { hit / n } else { f64::NAN })
}

fn inspect_indicator_scores(
    ctx: &mut FitCtx,
    y_true: &Matrix,
    scores: &Matrix,
    what: &str,
) -> bool {
    if y_true.nrows() != scores.nrows() || y_true.ncols() != scores.ncols() {
        ctx.push(
            Issue::builder(IssueCode::DimensionMismatch)
                .message(format!(
                    "{what}: y_true is {}×{} scores are {}×{}",
                    y_true.nrows(),
                    y_true.ncols(),
                    scores.nrows(),
                    scores.ncols()
                ))
                .build(),
        );
        return false;
    }
    inspect_xy(&mut ctx.report, scores, None, &ctx.policy);
    inspect_xy(&mut ctx.report, y_true, None, &ctx.policy);
    true
}

fn rank_desc(scores_row: impl Fn(usize) -> f64, p: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..p).collect();
    idx.sort_by(|a, b| {
        scores_row(*b)
            .partial_cmp(&scores_row(*a))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    idx
}

/// Label-ranking average precision (sklearn `label_ranking_average_precision_score`).
///
/// `y_true` is a 0/1 indicator matrix; column `j` is label `j`. Label count
/// is not identification `p`.
pub fn label_ranking_average_precision(
    y_true: &Matrix,
    scores: &Matrix,
    session: &Session,
) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if !inspect_indicator_scores(&mut ctx, y_true, scores, "label_ranking_average_precision") {
        return ctx.finish(f64::NAN);
    }
    let n = y_true.nrows();
    let p = y_true.ncols();
    let mut acc = 0.0;
    let mut used = 0.0;
    for i in 0..n {
        let mut rel = Vec::new();
        for j in 0..p {
            if y_true.get(i, j) > 0.5 {
                rel.push(j);
            }
        }
        if rel.is_empty() {
            continue;
        }
        let order = rank_desc(|j| scores.get(i, j), p);
        let mut rank_of = vec![0usize; p];
        for (r, &j) in order.iter().enumerate() {
            rank_of[j] = r + 1;
        }
        let mut s = 0.0;
        for &j in &rel {
            let rj = rank_of[j];
            let mut hit = 0.0;
            for &k in &rel {
                if rank_of[k] <= rj {
                    hit += 1.0;
                }
            }
            s += hit / rj as f64;
        }
        acc += s / rel.len() as f64;
        used += 1.0;
    }
    if used <= 0.0 {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .severity(Severity::Warning)
                .message("LRAP: no sample has a relevant label")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    ctx.finish(acc / used)
}

/// Coverage error (sklearn `coverage_error`): mean rank of the last relevant label.
pub fn coverage_error(
    y_true: &Matrix,
    scores: &Matrix,
    session: &Session,
) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if !inspect_indicator_scores(&mut ctx, y_true, scores, "coverage_error") {
        return ctx.finish(f64::NAN);
    }
    let n = y_true.nrows();
    let p = y_true.ncols();
    let mut acc = 0.0;
    let mut used = 0.0;
    for i in 0..n {
        let order = rank_desc(|j| scores.get(i, j), p);
        let mut rank_of = vec![0usize; p];
        for (r, &j) in order.iter().enumerate() {
            rank_of[j] = r + 1;
        }
        let mut last = 0usize;
        let mut any = false;
        for j in 0..p {
            if y_true.get(i, j) > 0.5 {
                any = true;
                last = last.max(rank_of[j]);
            }
        }
        if !any {
            continue;
        }
        acc += last as f64;
        used += 1.0;
    }
    if used <= 0.0 {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .severity(Severity::Warning)
                .message("coverage_error: no sample has a relevant label")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    ctx.finish(acc / used)
}

/// Label-ranking loss (sklearn `label_ranking_loss`): inverted relevant/irrelevant pairs.
pub fn label_ranking_loss(
    y_true: &Matrix,
    scores: &Matrix,
    session: &Session,
) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if !inspect_indicator_scores(&mut ctx, y_true, scores, "label_ranking_loss") {
        return ctx.finish(f64::NAN);
    }
    let n = y_true.nrows();
    let p = y_true.ncols();
    let mut acc = 0.0;
    let mut used = 0.0;
    for i in 0..n {
        let mut rel = Vec::new();
        let mut irr = Vec::new();
        for j in 0..p {
            if y_true.get(i, j) > 0.5 {
                rel.push(j);
            } else {
                irr.push(j);
            }
        }
        if rel.is_empty() || irr.is_empty() {
            continue;
        }
        let mut bad = 0.0;
        for &r in &rel {
            for &u in &irr {
                let sr = scores.get(i, r);
                let su = scores.get(i, u);
                if sr < su {
                    bad += 1.0;
                } else if (sr - su).abs() <= 0.0 {
                    bad += 0.5;
                }
            }
        }
        acc += bad / (rel.len() as f64 * irr.len() as f64);
        used += 1.0;
    }
    if used <= 0.0 {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .severity(Severity::Warning)
                .message("label_ranking_loss: every row is all-relevant or all-irrelevant")
                .build(),
        );
        return ctx.finish(0.0);
    }
    ctx.finish(acc / used)
}

fn empirical_quantile(xs: &[f64], tau: f64) -> f64 {
    if xs.is_empty() {
        return f64::NAN;
    }
    let mut v = xs.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let t = tau.clamp(0.0, 1.0) * (v.len() - 1) as f64;
    let lo = t.floor() as usize;
    let hi = t.ceil() as usize;
    let w = t - lo as f64;
    (1.0 - w) * v[lo] + w * v[hi.min(v.len() - 1)]
}

fn pinball_sum(y_true: &Vector, y_pred: &Vector, tau: f64) -> (f64, f64) {
    let mut s = 0.0;
    let mut n = 0.0;
    for i in 0..y_true.len().min(y_pred.len()) {
        if !y_true[i].is_finite() || !y_pred[i].is_finite() {
            continue;
        }
        let e = y_true[i] - y_pred[i];
        s += if e >= 0.0 { tau * e } else { (tau - 1.0) * e };
        n += 1.0;
    }
    (s, n)
}

/// \(D^2\) pinball score (sklearn `d2_pinball_score`).
pub fn d2_pinball_score(
    y_true: &Vector,
    y_pred: &Vector,
    tau: f64,
    session: &Session,
) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if !scan_pair(&mut ctx, y_true, y_pred, "d2_pinball_score") {
        return ctx.finish(f64::NAN);
    }
    warn_constant_target(&mut ctx, y_true, false);
    let t = if tau.is_finite() && tau > 0.0 && tau < 1.0 {
        tau
    } else {
        ctx.push(
            Issue::builder(IssueCode::InvalidWeight)
                .severity(Severity::Warning)
                .message(format!("d2_pinball τ={tau} is not in (0,1); using 0.5"))
                .build(),
        );
        0.5
    };
    let xs: Vec<f64> = y_true
        .as_slice()
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .collect();
    let q = empirical_quantile(&xs, t);
    let null = Vector::filled(y_true.len(), q);
    let (s_m, n) = pinball_sum(y_true, y_pred, t);
    let (s_n, _) = pinball_sum(y_true, &null, t);
    if n <= 0.0 || s_n.abs() <= 1e-18 {
        ctx.push(
            Issue::builder(IssueCode::R2IsZero)
                .message("D² pinball null loss is ~0; the score is undefined")
                .compromise(NumericalCompromise::new(
                    "positive null pinball to the empirical quantile",
                    "D² set to NaN",
                    "the quantile-only pinball vanished",
                    "do not read a missing D² as a perfect quantile fit",
                ))
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    ctx.finish(1.0 - s_m / s_n)
}

/// Root mean squared logarithmic error (sklearn `root_mean_squared_log_error`).
pub fn root_mean_squared_log_error(
    y_true: &Vector,
    y_pred: &Vector,
    session: &Session,
) -> Result<Qualified<f64>> {
    let q = mean_squared_log_error(y_true, y_pred, session)?;
    let v = if q.value.is_finite() && q.value >= 0.0 {
        q.value.sqrt()
    } else {
        f64::NAN
    };
    Ok(q.map(|_| v))
}

/// Pinball loss under the sklearn `mean_pinball_loss` name.
pub fn mean_pinball_loss(
    y_true: &Vector,
    y_pred: &Vector,
    tau: f64,
    session: &Session,
) -> Result<Qualified<f64>> {
    pinball_loss(y_true, y_pred, tau, session)
}

fn wilcoxon_auc(pos: &[bool], scores: &[f64]) -> f64 {
    let n = pos.len().min(scores.len());
    let mut pairs: Vec<(f64, bool)> = (0..n)
        .filter(|&i| scores[i].is_finite())
        .map(|i| (scores[i], pos[i]))
        .collect();
    if pairs.is_empty() {
        return f64::NAN;
    }
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let m = pairs.len();
    let mut ranks = vec![0.0; m];
    let mut i = 0;
    while i < m {
        let mut j = i + 1;
        while j < m && (pairs[j].0 - pairs[i].0).abs() <= 0.0 {
            j += 1;
        }
        let mean_rank = (i + 1 + j) as f64 / 2.0;
        for r in ranks.iter_mut().take(j).skip(i) {
            *r = mean_rank;
        }
        i = j;
    }
    let mut n_pos = 0.0;
    let mut n_neg = 0.0;
    let mut sum_pos = 0.0;
    for k in 0..m {
        if pairs[k].1 {
            n_pos += 1.0;
            sum_pos += ranks[k];
        } else {
            n_neg += 1.0;
        }
    }
    if n_pos <= 0.0 || n_neg <= 0.0 {
        return f64::NAN;
    }
    (sum_pos - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg)
}

/// One-vs-rest ROC-AUC from a score matrix (sklearn `roc_auc_score` `ovr`).
///
/// Column `j` is the score of class `j` after labels are mapped onto
/// `0..ncols`. Class count is not identification `p`.
pub fn roc_auc_ovr(y_true: &Vector, scores: &Matrix, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, scores, None, &ctx.policy);
    warn_constant_target(&mut ctx, y_true, true);
    if y_true.len() != scores.nrows() {
        ctx.push(
            Issue::builder(IssueCode::DimensionMismatch)
                .message(format!(
                    "roc_auc_ovr y.len()={} scores.nrows()={}",
                    y_true.len(),
                    scores.nrows()
                ))
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    let yt = labels_of(y_true);
    let classes = unique_sorted(&yt);
    let c = scores.ncols().max(1);
    if classes.len() < 2 {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .severity(Severity::Warning)
                .message("roc_auc_ovr needs two classes after filtering")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    let mut acc = 0.0;
    let mut k = 0.0;
    for (j, &lab) in classes.iter().enumerate() {
        let col = if j < c { j } else { c - 1 };
        let pos: Vec<bool> = yt.iter().map(|&v| v == lab).collect();
        let sc: Vec<f64> = (0..scores.nrows()).map(|i| scores.get(i, col)).collect();
        let a = wilcoxon_auc(&pos, &sc);
        if a.is_finite() {
            acc += a;
            k += 1.0;
        }
    }
    if k <= 0.0 {
        ctx.push(
            Issue::builder(IssueCode::PValueUnreliable)
                .message("every one-vs-rest AUC was undefined")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    ctx.finish(acc / k)
}

/// One-vs-one ROC-AUC from a score matrix (sklearn `roc_auc_score` `ovo`).
///
/// Class count is not identification `p`.
pub fn roc_auc_ovo(
    y_true: &Vector,
    scores: &Matrix,
    session: &Session,
) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, scores, None, &ctx.policy);
    warn_constant_target(&mut ctx, y_true, true);
    if y_true.len() != scores.nrows() {
        ctx.push(
            Issue::builder(IssueCode::DimensionMismatch)
                .message(format!(
                    "roc_auc_ovo y.len()={} scores.nrows()={}",
                    y_true.len(),
                    scores.nrows()
                ))
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    let yt = labels_of(y_true);
    let classes = unique_sorted(&yt);
    if classes.len() < 2 {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .severity(Severity::Warning)
                .message("roc_auc_ovo needs two classes")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    let c = scores.ncols().max(1);
    let mut acc = 0.0;
    let mut k = 0.0;
    for (i, &c1) in classes.iter().enumerate() {
        for &c2 in classes.iter().skip(i + 1) {
            let mut pos = Vec::new();
            let mut sc = Vec::new();
            for r in 0..yt.len() {
                if yt[r] != c1 && yt[r] != c2 {
                    continue;
                }
                let col = classes.iter().position(|&z| z == c1).unwrap_or(0).min(c - 1);
                pos.push(yt[r] == c1);
                sc.push(scores.get(r, col));
            }
            let a = wilcoxon_auc(&pos, &sc);
            if a.is_finite() {
                acc += a;
                k += 1.0;
            }
        }
    }
    if k <= 0.0 {
        ctx.push(
            Issue::builder(IssueCode::PValueUnreliable)
                .message("every one-vs-one AUC was undefined")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    ctx.finish(acc / k)
}

/// Per-class precision / recall / F1 (sklearn `classification_report` numbers).
pub fn classification_report(
    y_true: &Vector,
    y_pred: &Vector,
    session: &Session,
) -> Result<Qualified<PrecisionRecallF1>> {
    precision_recall_f1(y_true, y_pred, session)
}

/// Detection-error-tradeoff curve (sklearn `det_curve`).
///
/// Returns false-positive rate, false-negative rate, and thresholds.
pub fn det_curve(
    y_true: &Vector,
    scores: &Vector,
    session: &Session,
) -> Result<Qualified<RocCurve>> {
    let roc = roc_curve(y_true, scores, session)?;
    let fnr = Vector::from_iter(roc.value.tpr.as_slice().iter().map(|t| 1.0 - *t));
    Ok(roc.map(|r| RocCurve {
        fpr: r.fpr,
        tpr: fnr,
        thresholds: r.thresholds,
    }))
}

fn bump<K: PartialEq>(xs: &mut Vec<(K, f64)>, key: K) {
    if let Some(e) = xs.iter_mut().find(|(k, _)| *k == key) {
        e.1 += 1.0;
    } else {
        xs.push((key, 1.0));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ojizou_san::Session;

    #[test]
    fn accuracy_and_prf1_binary() {
        let yt = Vector::from_slice(&[0.0, 0.0, 1.0, 1.0, 1.0]);
        let yp = Vector::from_slice(&[0.0, 1.0, 1.0, 1.0, 0.0]);
        let s = Session::new("metrics", "accuracy");
        let acc = accuracy(&yt, &yp, &s).unwrap().value;
        assert!((acc - 0.6).abs() < 1e-12);
        let pr = precision_recall_f1(&yt, &yp, &Session::new("metrics", "f1"))
            .unwrap()
            .value;
        assert!((pr.precision - 2.0 / 3.0).abs() < 1e-12);
        assert!((pr.recall - 2.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn roc_auc_separable_is_one() {
        let y = Vector::from_slice(&[0.0, 0.0, 1.0, 1.0]);
        let s = Vector::from_slice(&[0.1, 0.2, 0.8, 0.9]);
        let auc = roc_auc(&y, &s, &Session::new("metrics", "auc"))
            .unwrap()
            .value;
        assert!((auc - 1.0).abs() < 1e-12);
        let roc = roc_curve(&y, &s, &Session::new("metrics", "roc"))
            .unwrap()
            .value;
        assert!(roc.tpr.len() >= 2);
        assert!(roc.tpr.as_slice().iter().any(|v| *v >= 1.0 - 1e-12));
        let prc = precision_recall_curve(&y, &s, &Session::new("metrics", "prc"))
            .unwrap()
            .value;
        assert!(prc.recall.as_slice().last().copied().unwrap_or(0.0) >= 1.0 - 1e-12);
        let hl = hinge_loss(
            &Vector::from_slice(&[-1.0, -1.0, 1.0, 1.0]),
            &s,
            &Session::new("metrics", "hinge"),
        )
        .unwrap()
        .value;
        assert!(hl >= 0.0 && hl.is_finite());
    }

    #[test]
    fn r2_perfect_and_negative() {
        let y = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0]);
        let hat = y.clone();
        let r = r2(&y, &hat, &Session::new("metrics", "r2")).unwrap().value;
        assert!((r - 1.0).abs() < 1e-12);
        let worse = Vector::from_slice(&[10.0, -4.0, 12.0, -8.0]);
        let q = r2(&y, &worse, &Session::new("metrics", "r2_neg")).unwrap();
        assert!(q.value < 0.0);
        assert!(q.report.contains(IssueCode::R2Negative));
    }

    #[test]
    fn constant_y_is_meaningless() {
        let y = Vector::filled(6, 3.0);
        let p = Vector::filled(6, 3.0);
        let err = accuracy(&y, &p, &Session::new("metrics", "const")).unwrap_err();
        assert!(
            err.report.contains(IssueCode::MeaninglessFit)
                || err.primary().code == IssueCode::MeaninglessFit
                || err.report.contains(IssueCode::SingleClass)
                || err.primary().code == IssueCode::SingleClass
        );
        assert!(
            err.report.contains(IssueCode::ClassImbalanceSevere)
                || err.report.contains(IssueCode::SingleClass)
        );
    }

    #[test]
    fn silhouette_two_blobs_positive() {
        let x = Matrix::from_fn(6, 1, |i, _| if i < 3 { 0.0 } else { 10.0 });
        let y = Vector::from_iter((0..6).map(|i| if i < 3 { 0.0 } else { 1.0 }));
        let s = silhouette(&x, &y, &Session::new("metrics", "sil"))
            .unwrap()
            .value;
        assert!(s > 0.8, "sil={s}");
    }

    #[test]
    fn ari_identity_is_one() {
        let y = Vector::from_slice(&[0.0, 0.0, 1.0, 1.0, 2.0]);
        let a = adjusted_rand(&y, &y, &Session::new("metrics", "ari"))
            .unwrap()
            .value;
        assert!((a - 1.0).abs() < 1e-12);
    }

    #[test]
    fn mase_and_smape_on_trend() {
        let y = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0]);
        let hat = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0]);
        let m = mase(&y, &hat, &Session::new("metrics", "mase"))
            .unwrap()
            .value;
        assert!(m.abs() < 1e-12);
        let sm = smape(&y, &hat, &Session::new("metrics", "smape"))
            .unwrap()
            .value;
        assert!(sm.abs() < 1e-12);
    }

    #[test]
    fn log_loss_and_confusion() {
        let y = Vector::from_slice(&[0.0, 1.0, 1.0, 0.0]);
        let p = Vector::from_slice(&[0.1, 0.9, 0.8, 0.2]);
        let ll = log_loss(&y, &p, &Session::new("metrics", "ll"))
            .unwrap()
            .value;
        assert!(ll > 0.0 && ll < 0.3);
        let cm = confusion_matrix(
            &y,
            &Vector::from_slice(&[0.0, 1.0, 1.0, 0.0]),
            &Session::new("metrics", "cm"),
        )
        .unwrap()
        .value;
        assert_eq!(cm.nrows(), 2);
        assert!((cm.get(0, 0) - 2.0).abs() < 1e-12);
        assert!((cm.get(1, 1) - 2.0).abs() < 1e-12);
        let _ = (mse, mae, mape, medae);
        let y2 = Vector::from_slice(&[1.0, 2.0, 3.0]);
        let h2 = Vector::from_slice(&[1.0, 2.0, 2.0]);
        assert!(mse(&y2, &h2, &Session::new("m", "mse")).unwrap().value > 0.0);
        assert!(mae(&y2, &h2, &Session::new("m", "mae")).unwrap().value > 0.0);
        assert!(mape(&y2, &h2, &Session::new("m", "mape")).unwrap().value > 0.0);
        assert!(medae(&y2, &h2, &Session::new("m", "med")).unwrap().value >= 0.0);
        let br = brier(&y, &p, &Session::new("m", "brier")).unwrap().value;
        assert!(br > 0.0 && br < 0.1);
        let ap = average_precision(&y, &p, &Session::new("m", "ap"))
            .unwrap()
            .value;
        assert!(ap > 0.8);
        let ev = explained_variance(&y2, &h2, &Session::new("m", "ev"))
            .unwrap()
            .value;
        assert!(ev.is_finite());
        let ham = hamming(
            &y,
            &Vector::from_slice(&[0.0, 1.0, 1.0, 0.0]),
            &Session::new("m", "ham"),
        )
        .unwrap()
        .value;
        assert!(ham.abs() < 1e-12);
        let mi = mutual_info(&y, &y, &Session::new("m", "mi")).unwrap().value;
        assert!(mi > 0.0);
        let kap = cohen_kappa(&y, &y, &Session::new("m", "kap"))
            .unwrap()
            .value;
        assert!((kap - 1.0).abs() < 1e-12);
        let mcc = matthews_corrcoef(&y, &y, &Session::new("m", "mcc"))
            .unwrap()
            .value;
        assert!((mcc - 1.0).abs() < 1e-12);
        let xb = Matrix::from_fn(8, 1, |i, _| {
            if i < 4 {
                0.1 * i as f64
            } else {
                10.0 + 0.1 * (i as f64 - 4.0)
            }
        });
        let lb = Vector::from_iter((0..8).map(|i| if i < 4 { 0.0 } else { 1.0 }));
        let ch = calinski_harabasz(&xb, &lb, &Session::new("m", "ch"))
            .unwrap()
            .value;
        assert!(ch > 1.0, "ch={ch}");
        let db = davies_bouldin(&xb, &lb, &Session::new("m", "db"))
            .unwrap()
            .value;
        assert!(db.is_finite() && db >= 0.0);
        let pb = pinball_loss(&y2, &h2, 0.5, &Session::new("m", "pin"))
            .unwrap()
            .value;
        assert!(pb >= 0.0);
        let cal = calibration_curve(&y, &p, 4, &Session::new("m", "cal"))
            .unwrap()
            .value;
        assert!(cal.prob_true.len() >= 2);
        assert_eq!(cal.prob_true.len(), cal.prob_pred.len());
        let zo = zero_one_loss(
            &y,
            &Vector::from_slice(&[0.0, 1.0, 1.0, 0.0]),
            &Session::new("m", "zo"),
        )
        .unwrap()
        .value;
        assert!(zo.abs() < 1e-12);
        let jac = jaccard_score(
            &y,
            &Vector::from_slice(&[0.0, 1.0, 1.0, 0.0]),
            &Session::new("m", "jac"),
        )
        .unwrap()
        .value;
        assert!((jac - 1.0).abs() < 1e-12);
        let pd = mean_poisson_deviance(&y2, &h2, &Session::new("m", "pois"))
            .unwrap()
            .value;
        assert!(pd.is_finite() && pd >= 0.0);
        let td = mean_tweedie_deviance(&y2, &h2, 0.0, &Session::new("m", "tw"))
            .unwrap()
            .value;
        assert!(td.is_finite() && td >= 0.0);
        let d2 = d2_tweedie_score(&y2, &h2, 0.0, &Session::new("m", "d2"))
            .unwrap()
            .value;
        assert!(d2.is_finite());
        let ba = balanced_accuracy(
            &y,
            &Vector::from_slice(&[0.0, 1.0, 1.0, 0.0]),
            &Session::new("m", "ba"),
        )
        .unwrap()
        .value;
        assert!((ba - 1.0).abs() < 1e-12);
        let gd = mean_gamma_deviance(&y2, &h2, &Session::new("m", "gd"))
            .unwrap()
            .value;
        assert!(gd.is_finite() && gd >= 0.0);
        let d2a = d2_absolute_error_score(&y2, &h2, &Session::new("m", "d2a"))
            .unwrap()
            .value;
        assert!(d2a.is_finite());
        let rel = Vector::from_slice(&[3.0, 2.0, 1.0, 0.0]);
        let sc = Vector::from_slice(&[0.9, 0.8, 0.1, 0.0]);
        let nd = ndcg_score(&rel, &sc, &Session::new("m", "ndcg"))
            .unwrap()
            .value;
        assert!((nd - 1.0).abs() < 1e-12);
        let dc = dcg_score(&rel, &sc, &Session::new("m", "dcg"))
            .unwrap()
            .value;
        assert!(dc.is_finite() && dc > 0.0);
        let hom = homogeneity_score(&y, &y, &Session::new("m", "hom"))
            .unwrap()
            .value;
        assert!((hom - 1.0).abs() < 1e-12);
        let vm = v_measure_score(&y, &y, &Session::new("m", "vm"))
            .unwrap()
            .value;
        assert!((vm - 1.0).abs() < 1e-12);
        let nmi = normalized_mutual_info_score(&y, &y, &Session::new("m", "nmi"))
            .unwrap()
            .value;
        assert!((nmi - 1.0).abs() < 1e-12);
        let ami = adjusted_mutual_info_score(&y, &y, &Session::new("m", "ami"))
            .unwrap()
            .value;
        assert!(ami > 0.5, "ami={ami}");
        let fms = fowlkes_mallows_score(&y, &y, &Session::new("m", "fms"))
            .unwrap()
            .value;
        assert!((fms - 1.0).abs() < 1e-12);
        let lc = log_cosh_error(&y2, &h2, &Session::new("m", "lch"))
            .unwrap()
            .value;
        assert!(lc.is_finite() && lc >= 0.0);
        let msle = mean_squared_log_error(&y2, &h2, &Session::new("m", "msle"))
            .unwrap()
            .value;
        assert!(msle.is_finite() && msle >= 0.0);
        let rmse = root_mean_squared_error(&y2, &h2, &Session::new("m", "rmse"))
            .unwrap()
            .value;
        assert!(rmse.is_finite() && rmse >= 0.0);
        let sc2 = Matrix::from_fn(4, 2, |i, j| {
            let want = if y[i] > 0.5 { 1 } else { 0 };
            if j == want {
                0.9
            } else {
                0.1
            }
        });
        let tk = top_k_accuracy(&y, &sc2, 1, &Session::new("m", "tk"))
            .unwrap()
            .value;
        assert!((tk - 1.0).abs() < 1e-12);
        let yt = Matrix::from_fn(3, 3, |i, j| if i == j { 1.0 } else { 0.0 });
        let ys = Matrix::from_fn(3, 3, |i, j| if i == j { 0.9 } else { 0.1 });
        let lrap = label_ranking_average_precision(&yt, &ys, &Session::new("m", "lrap"))
            .unwrap()
            .value;
        assert!((lrap - 1.0).abs() < 1e-12);
        let cov = coverage_error(&yt, &ys, &Session::new("m", "cov"))
            .unwrap()
            .value;
        assert!((cov - 1.0).abs() < 1e-12);
        let lrl = label_ranking_loss(&yt, &ys, &Session::new("m", "lrl"))
            .unwrap()
            .value;
        assert!(lrl.abs() < 1e-12);
        let d2p = d2_pinball_score(&y2, &h2, 0.5, &Session::new("m", "d2p"))
            .unwrap()
            .value;
        assert!(d2p.is_finite());
        let ss = silhouette_samples(&xb, &lb, &Session::new("m", "sils"))
            .unwrap()
            .value;
        assert_eq!(ss.len(), 8);
        assert!(ss.as_slice().iter().all(|v| v.is_finite()));
        let pd = pairwise_distances(&xb, &xb, &Session::new("m", "pwd"))
            .unwrap()
            .value;
        assert_eq!(pd.shape(), (8, 8));
        assert!(pd.get(0, 0).abs() < 1e-12);
        let me = max_error(&y2, &h2, &Session::new("m", "maxe"))
            .unwrap()
            .value;
        assert!((me - 1.0).abs() < 1e-12);
        let fb = fbeta_score(
            &y,
            &Vector::from_slice(&[0.0, 1.0, 1.0, 0.0]),
            1.0,
            &Session::new("m", "fb"),
        )
        .unwrap()
        .value;
        assert!((fb - 1.0).abs() < 1e-12);
        let rs = rand_score(&y, &y, &Session::new("m", "ri")).unwrap().value;
        assert!((rs - 1.0).abs() < 1e-12);
        let tw = trustworthiness(&xb, &xb, 2, &Session::new("m", "tw"))
            .unwrap()
            .value;
        assert!(tw > 0.8, "trust={tw}");
        let mlcm = multilabel_confusion_matrix(&yt, &yt, &Session::new("m", "mlcm"))
            .unwrap()
            .value;
        assert_eq!(mlcm.shape(), (3, 4));
        assert!((mlcm.get(0, 3) - 1.0).abs() < 1e-12);
        let ed = euclidean_distances(&xb, &xb, &Session::new("m", "eu"))
            .unwrap()
            .value;
        assert!(ed.get(0, 0).abs() < 1e-12);
        let md = manhattan_distances(&xb, &xb, &Session::new("m", "man"))
            .unwrap()
            .value;
        assert!(md.get(0, 0).abs() < 1e-12);
        let cs = cosine_similarity(&xb, &xb, &Session::new("m", "cos"))
            .unwrap()
            .value;
        assert!((cs.get(1, 1) - 1.0).abs() < 1e-9);
        let rk = rbf_kernel(&xb, &xb, 0.5, &Session::new("m", "rbf"))
            .unwrap()
            .value;
        assert!((rk.get(0, 0) - 1.0).abs() < 1e-12);
        let lk = linear_kernel(&xb, &xb, &Session::new("m", "lin"))
            .unwrap()
            .value;
        assert!(lk.get(0, 0).is_finite());
        let pk = polynomial_kernel(&xb, &xb, 2, 1.0, 1.0, &Session::new("m", "poly"))
            .unwrap()
            .value;
        assert!(pk.get(0, 0).is_finite());
        let rmsle = root_mean_squared_log_error(&y2, &h2, &Session::new("m", "rmsle"))
            .unwrap()
            .value;
        assert!(rmsle.is_finite() && rmsle >= 0.0);
        let mpl = mean_pinball_loss(&y2, &h2, 0.5, &Session::new("m", "mpl"))
            .unwrap()
            .value;
        assert!(mpl >= 0.0);
        let ovr = roc_auc_ovr(&y, &sc2, &Session::new("m", "ovr"))
            .unwrap()
            .value;
        assert!((ovr - 1.0).abs() < 1e-12);
        let det = det_curve(&y, &p, &Session::new("m", "det")).unwrap().value;
        assert!(det.fpr.len() >= 2);
        assert!(det.tpr.as_slice().iter().all(|v| v.is_finite()));
        let lap = laplacian_kernel(&xb, &xb, 0.5, &Session::new("m", "lap"))
            .unwrap()
            .value;
        assert!((lap.get(1, 1) - 1.0).abs() < 1e-12);
        let sg = sigmoid_kernel(&xb, &xb, 0.1, 0.0, &Session::new("m", "sig"))
            .unwrap()
            .value;
        assert!(sg.get(1, 1).is_finite());
        let ck = chi2_kernel(&xb, &xb, 0.5, &Session::new("m", "chi2"))
            .unwrap()
            .value;
        assert!((ck.get(1, 1) - 1.0).abs() < 1e-12);
        let geo = Matrix::from_fn(2, 2, |i, j| if j == 0 { 0.0 } else { i as f64 });
        let hv = haversine_distances(&geo, &geo, &Session::new("m", "hav"))
            .unwrap()
            .value;
        assert_eq!(hv.shape(), (2, 2));
        assert!(hv.get(0, 0).abs() < 1e-9);
        assert!(hv.get(0, 1) > 100.0 && hv.get(0, 1) < 120.0);
        let ovo = roc_auc_ovo(&y, &sc2, &Session::new("m", "ovo"))
            .unwrap()
            .value;
        assert!((ovo - 1.0).abs() < 1e-12);
        let cr = classification_report(&y, &y, &Session::new("m", "cr"))
            .unwrap()
            .value;
        assert!((cr.precision - 1.0).abs() < 1e-12);
        assert!((cr.f1 - 1.0).abs() < 1e-12);
    }
}
