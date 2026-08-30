//! Classification, regression, clustering, and forecast scores.
//!
//! Every public computation opens a [`crate::context::FitCtx`]. A constant
//! `y` is diagnosed as [`IssueCode::MeaninglessFit`] (and, for classifiers,
//! [`IssueCode::ClassImbalanceSevere`]). A predictor that loses to the mean
//! raises [`IssueCode::R2Negative`].

use crate::context::FitCtx;
use crate::data::{Matrix, Vector};
use crate::validate::inspect_classes;
use ojizou_san::Session;
use signlred::{Issue, IssueCode, Meaninglessness, Qualified, Result};

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
    }
}
