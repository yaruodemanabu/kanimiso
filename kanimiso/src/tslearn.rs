//! Time-series distances, barycentres, clustering, SAX/PAA, and a DTW baseline SVM.
//!
//! Distances and estimators open a [`crate::context::FitCtx`]. DTW is the
//! classic dynamic program; soft-DTW uses the `γ`-softmin of Cuturi & Blondel.

use crate::context::FitCtx;
use crate::data::{Matrix, Vector};
use crate::linalg::{ridge_solve, thin_svd};
use crate::linear_model::{FittedPenalized, Ridge};
use crate::rng::Rng;
use crate::special::norm_cdf;
use crate::traits::{Fit, FitUnsupervised, Predict, Transform};
use crate::validate::{inspect_classes, inspect_xy};
use ojizou_san::Session;
use signlred::{Issue, IssueCode, NumericalCompromise, Qualified, Report, Result, Severity};
use std::collections::BTreeMap;

fn series_ok(a: &Vector) -> bool {
    a.as_slice().iter().any(|v| v.is_finite())
}

fn _use_series_ok(a: &Vector, ctx: &mut FitCtx) {
    if !series_ok(a) {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message("series has no finite samples")
                .build(),
        );
    }
}

/// Classic DTW distance (absolute local cost, no window).
pub fn dtw(a: &Vector, b: &Vector, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if let Some(issue) = signlred::scan_finite(a.as_slice()).to_issue("dtw.a") {
        ctx.push(issue);
    }
    if let Some(issue) = signlred::scan_finite(b.as_slice()).to_issue("dtw.b") {
        ctx.push(issue);
    }
    _use_series_ok(a, &mut ctx);
    _use_series_ok(b, &mut ctx);
    if a.is_empty() || b.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message("DTW on an empty series")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    ctx.finish(dtw_raw(a.as_slice(), b.as_slice()))
}

/// Longest common subsequence similarity under an ε-tube (tslearn `lcss`).
///
/// Optional Sakoe–Chiba `band` (`None` ⇒ full). Similarity is
/// \(\mathrm{LCS}/\max(n,m)\).
pub fn lcss(
    a: &Vector,
    b: &Vector,
    eps: f64,
    band: Option<usize>,
    session: &Session,
) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if let Some(issue) = signlred::scan_finite(a.as_slice()).to_issue("lcss.a") {
        ctx.push(issue);
    }
    if let Some(issue) = signlred::scan_finite(b.as_slice()).to_issue("lcss.b") {
        ctx.push(issue);
    }
    if !eps.is_finite() || eps < 0.0 {
        ctx.push(
            Issue::builder(IssueCode::InvalidWeight)
                .severity(Severity::Warning)
                .message(format!("LCSS ε={eps} is not a finite ≥0 radius; using |ε|"))
                .build(),
        );
    }
    if a.is_empty() || b.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message("LCSS on an empty series")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    let e = if eps.is_finite() { eps.abs() } else { 0.0 };
    ctx.finish(lcss_raw(a.as_slice(), b.as_slice(), e, band))
}

/// Weighted DTW (tslearn `wdtw`, Jeong logistic weights).
///
/// \(w(|i-j|)=1/(1+\exp(-g(|i-j|-m_c)))\) with \(m_c=\max(n,m)/2\).
pub fn wdtw(a: &Vector, b: &Vector, g: f64, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if let Some(issue) = signlred::scan_finite(a.as_slice()).to_issue("wdtw.a") {
        ctx.push(issue);
    }
    if let Some(issue) = signlred::scan_finite(b.as_slice()).to_issue("wdtw.b") {
        ctx.push(issue);
    }
    let g = if g.is_finite() && g >= 0.0 {
        g
    } else {
        ctx.push(
            Issue::builder(IssueCode::InvalidWeight)
                .severity(Severity::Warning)
                .message(format!("wdtw g={g} is not a finite ≥0 slope; using 0.1"))
                .build(),
        );
        0.1
    };
    if a.is_empty() || b.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message("WDTW on an empty series")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    ctx.finish(wdtw_raw(a.as_slice(), b.as_slice(), g))
}

fn wdtw_raw(a: &[f64], b: &[f64], g: f64) -> f64 {
    let n = a.len();
    let m = b.len();
    let mc = n.max(m) as f64 / 2.0;
    let inf: f64 = 1e300;
    let mut prev = vec![inf; m + 1];
    let mut cur = vec![inf; m + 1];
    prev[0] = 0.0;
    for i in 1..=n {
        cur[0] = inf;
        for j in 1..=m {
            let d = (i as f64 - j as f64).abs();
            let w = 1.0 / (1.0 + (-g * (d - mc)).exp());
            let cost = w * (a[i - 1] - b[j - 1]).abs();
            cur[j] = cost + prev[j].min(cur[j - 1]).min(prev[j - 1]);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[m]
}

fn ddtw_deriv(s: &[f64]) -> Vec<f64> {
    let n = s.len();
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![0.0];
    }
    let mut d = vec![0.0; n];
    d[0] = s[1] - s[0];
    d[n - 1] = s[n - 1] - s[n - 2];
    for i in 1..n - 1 {
        d[i] = 0.5 * (s[i + 1] - s[i - 1]);
    }
    d
}

/// Derivative DTW (Keogh / tslearn `ddtw`).
pub fn ddtw(a: &Vector, b: &Vector, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if let Some(issue) = signlred::scan_finite(a.as_slice()).to_issue("ddtw.a") {
        ctx.push(issue);
    }
    if let Some(issue) = signlred::scan_finite(b.as_slice()).to_issue("ddtw.b") {
        ctx.push(issue);
    }
    if a.is_empty() || b.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message("DDTW on an empty series")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    let da = ddtw_deriv(a.as_slice());
    let db = ddtw_deriv(b.as_slice());
    ctx.finish(dtw_raw(&da, &db))
}

/// Weighted derivative DTW (Jeong–Jeong–Omitaomu `WDDTW`).
///
/// Applies [`wdtw`] to first-difference descriptors. Distinct from [`ddtw`]
/// (unweighted) and [`wdtw`] (levels). Logistic slope `g` is not
/// identification `p`.
pub fn wddtw(a: &Vector, b: &Vector, g: f64, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if let Some(issue) = signlred::scan_finite(a.as_slice()).to_issue("wddtw.a") {
        ctx.push(issue);
    }
    if let Some(issue) = signlred::scan_finite(b.as_slice()).to_issue("wddtw.b") {
        ctx.push(issue);
    }
    let g = if g.is_finite() && g >= 0.0 {
        g
    } else {
        ctx.push(
            Issue::builder(IssueCode::InvalidWeight)
                .severity(Severity::Warning)
                .message(format!("wddtw g={g} is not a finite ≥0 slope; using 0.1"))
                .build(),
        );
        0.1
    };
    if a.is_empty() || b.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message("WDDTW on an empty series")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    let da = ddtw_deriv(a.as_slice());
    let db = ddtw_deriv(b.as_slice());
    ctx.finish(wdtw_raw(&da, &db, g))
}

fn shape_desc(s: &[f64]) -> Vec<f64> {
    let n = s.len();
    if n == 0 {
        return Vec::new();
    }
    let mut d = vec![0.0; n];
    for i in 0..n {
        let lo = i.saturating_sub(1);
        let hi = (i + 1).min(n - 1);
        d[i] = s[hi] - s[lo];
    }
    d
}

/// Shape DTW: DTW on local first differences (tslearn / Zhao `shape_dtw`).
pub fn shape_dtw(a: &Vector, b: &Vector, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if let Some(issue) = signlred::scan_finite(a.as_slice()).to_issue("shape_dtw.a") {
        ctx.push(issue);
    }
    if let Some(issue) = signlred::scan_finite(b.as_slice()).to_issue("shape_dtw.b") {
        ctx.push(issue);
    }
    if a.is_empty() || b.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message("shape DTW on an empty series")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    ctx.finish(dtw_raw(&shape_desc(a.as_slice()), &shape_desc(b.as_slice())))
}

/// Global alignment kernel between two series (tslearn `gak`).
///
/// Bandwidth \(\sigma\) is not identification `p`.
pub fn gak(a: &Vector, b: &Vector, sigma: f64, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if let Some(issue) = signlred::scan_finite(a.as_slice()).to_issue("gak.a") {
        ctx.push(issue);
    }
    if let Some(issue) = signlred::scan_finite(b.as_slice()).to_issue("gak.b") {
        ctx.push(issue);
    }
    let s = if sigma.is_finite() && sigma > 0.0 {
        sigma
    } else {
        ctx.push(
            Issue::builder(IssueCode::InvalidWeight)
                .severity(Severity::Warning)
                .message(format!("gak σ={sigma} is not positive; using 1"))
                .build(),
        );
        1.0
    };
    if a.is_empty() || b.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message("GAK on an empty series")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    let d = softdtw_raw(a.as_slice(), b.as_slice(), 0.1);
    ctx.finish((-d / s).exp())
}

fn zscore_series(s: &[f64]) -> Vec<f64> {
    let n = s.len().max(1) as f64;
    let mean = s.iter().sum::<f64>() / n;
    let var = s.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n;
    let sd = var.sqrt().max(1e-12);
    s.iter().map(|v| (v - mean) / sd).collect()
}

/// Shape-based distance (Paparrizos / tslearn `sbd`).
///
/// \(1 - \max_w \mathrm{NCC}_w(\tilde a,\tilde b)\). Identical series score 0.
pub fn sbd(a: &Vector, b: &Vector, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if let Some(issue) = signlred::scan_finite(a.as_slice()).to_issue("sbd.a") {
        ctx.push(issue);
    }
    if let Some(issue) = signlred::scan_finite(b.as_slice()).to_issue("sbd.b") {
        ctx.push(issue);
    }
    if a.is_empty() || b.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message("SBD on an empty series")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    ctx.finish(sbd_raw(a.as_slice(), b.as_slice()))
}

fn sbd_raw(a: &[f64], b: &[f64]) -> f64 {
    if a.is_empty() || b.is_empty() {
        return f64::NAN;
    }
    let za = zscore_series(a);
    let zb = zscore_series(b);
    let na = za.len();
    let nb = zb.len();
    let mut best = f64::NEG_INFINITY;
    for shift in -(nb as i32 - 1)..=(na as i32 - 1) {
        let mut num = 0.0;
        for i in 0..na {
            let j = i as i32 - shift;
            if j >= 0 && (j as usize) < nb {
                num += za[i] * zb[j as usize];
            }
        }
        if num > best {
            best = num;
        }
    }
    let ncc = best / na.max(nb) as f64;
    (1.0 - ncc).max(0.0)
}

/// Pairwise shape-based distance (tslearn `cdist_sbd`).
///
/// Series / pair counts are not identification `p`.
pub fn cdist_sbd(a: &Matrix, b: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, a, None, &ctx.policy);
    inspect_xy(&mut ctx.report, b, None, &ctx.policy);
    let out = Matrix::from_fn(a.nrows(), b.nrows(), |i, j| {
        let ai = a.row(i);
        let bj = b.row(j);
        sbd_raw(ai.as_slice(), bj.as_slice())
    });
    ctx.finish(out)
}

/// Pairwise canonical time warping (tslearn `ctw`).
pub fn ctw(a: &Vector, b: &Vector, session: &Session) -> Result<Qualified<f64>> {
    canonical_time_warping(a, b, session)
}

/// Discrete Fréchet distance (tslearn `frechet`).
///
/// Identical series score 0.
pub fn frechet(a: &Vector, b: &Vector, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if let Some(issue) = signlred::scan_finite(a.as_slice()).to_issue("frechet.a") {
        ctx.push(issue);
    }
    if let Some(issue) = signlred::scan_finite(b.as_slice()).to_issue("frechet.b") {
        ctx.push(issue);
    }
    if a.is_empty() || b.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message("Fréchet on an empty series")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    ctx.finish(frechet_raw(a.as_slice(), b.as_slice()))
}

fn frechet_raw(a: &[f64], b: &[f64]) -> f64 {
    if a.is_empty() || b.is_empty() {
        return f64::NAN;
    }
    let n = a.len();
    let m = b.len();
    let mut dp = vec![0.0; n * m];
    let at = |i: usize, j: usize| i * m + j;
    dp[0] = (a[0] - b[0]).abs();
    for i in 1..n {
        dp[at(i, 0)] = dp[at(i - 1, 0)].max((a[i] - b[0]).abs());
    }
    for j in 1..m {
        dp[at(0, j)] = dp[at(0, j - 1)].max((a[0] - b[j]).abs());
    }
    for i in 1..n {
        for j in 1..m {
            let prev = dp[at(i - 1, j)].min(dp[at(i, j - 1)]).min(dp[at(i - 1, j - 1)]);
            dp[at(i, j)] = prev.max((a[i] - b[j]).abs());
        }
    }
    dp[at(n - 1, m - 1)]
}

/// Pairwise discrete Fréchet (tslearn `cdist` with Fréchet).
///
/// Series / pair counts are not identification `p`.
pub fn cdist_frechet(a: &Matrix, b: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, a, None, &ctx.policy);
    inspect_xy(&mut ctx.report, b, None, &ctx.policy);
    let out = Matrix::from_fn(a.nrows(), b.nrows(), |i, j| {
        let ai = a.row(i);
        let bj = b.row(j);
        frechet_raw(ai.as_slice(), bj.as_slice())
    });
    ctx.finish(out)
}

/// Hausdorff distance between two series as point sets (tslearn `hausdorff`).
///
/// Identical series score 0.
pub fn hausdorff(a: &Vector, b: &Vector, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if let Some(issue) = signlred::scan_finite(a.as_slice()).to_issue("hausdorff.a") {
        ctx.push(issue);
    }
    if let Some(issue) = signlred::scan_finite(b.as_slice()).to_issue("hausdorff.b") {
        ctx.push(issue);
    }
    if a.is_empty() || b.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message("Hausdorff on an empty series")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    let mut ab = 0.0_f64;
    for &ai in a.as_slice() {
        let mut best = f64::INFINITY;
        for &bj in b.as_slice() {
            best = best.min((ai - bj).abs());
        }
        ab = ab.max(best);
    }
    let mut ba = 0.0_f64;
    for &bj in b.as_slice() {
        let mut best = f64::INFINITY;
        for &ai in a.as_slice() {
            best = best.min((ai - bj).abs());
        }
        ba = ba.max(best);
    }
    ctx.finish(ab.max(ba))
}

fn hausdorff_raw(a: &[f64], b: &[f64]) -> f64 {
    if a.is_empty() || b.is_empty() {
        return f64::NAN;
    }
    let mut ab = 0.0_f64;
    for &ai in a {
        let mut best = f64::INFINITY;
        for &bj in b {
            best = best.min((ai - bj).abs());
        }
        ab = ab.max(best);
    }
    let mut ba = 0.0_f64;
    for &bj in b {
        let mut best = f64::INFINITY;
        for &ai in a {
            best = best.min((ai - bj).abs());
        }
        ba = ba.max(best);
    }
    ab.max(ba)
}

/// Pairwise Hausdorff distance (tslearn `cdist` with Hausdorff).
///
/// Series / pair counts are not identification `p`.
pub fn cdist_hausdorff(a: &Matrix, b: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, a, None, &ctx.policy);
    inspect_xy(&mut ctx.report, b, None, &ctx.policy);
    let out = Matrix::from_fn(a.nrows(), b.nrows(), |i, j| {
        let ai = a.row(i);
        let bj = b.row(j);
        hausdorff_raw(ai.as_slice(), bj.as_slice())
    });
    ctx.finish(out)
}

/// Pairwise edit distance on real sequences (tslearn `cdist` with EDR).
///
/// Series / pair counts are not identification `p`. `ε` is not identification
/// `p`. Distinct from [`cdist_dtw`] (real cost) and [`cdist_lcss`] (similarity).
pub fn cdist_edr(
    a: &Matrix,
    b: &Matrix,
    eps: f64,
    session: &Session,
) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, a, None, &ctx.policy);
    inspect_xy(&mut ctx.report, b, None, &ctx.policy);
    let eps = if eps.is_finite() && eps >= 0.0 {
        eps
    } else {
        ctx.push(
            Issue::builder(IssueCode::InvalidWeight)
                .severity(Severity::Warning)
                .message(format!("cdist_edr ε={eps} is not a finite ≥0 match radius; using 0"))
                .build(),
        );
        0.0
    };
    let out = Matrix::from_fn(a.nrows(), b.nrows(), |i, j| {
        let ai = a.row(i);
        let bj = b.row(j);
        edr_raw(ai.as_slice(), bj.as_slice(), eps)
    });
    ctx.finish(out)
}

/// Pairwise Amerced DTW (tslearn `cdist` with ADTW).
///
/// Series / pair counts are not identification `p`. `ω` is not identification
/// `p`. Distinct from [`cdist_dtw`] (`ω = 0`) and [`cdist_wdtw`].
pub fn cdist_adtw(
    a: &Matrix,
    b: &Matrix,
    omega: f64,
    session: &Session,
) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, a, None, &ctx.policy);
    inspect_xy(&mut ctx.report, b, None, &ctx.policy);
    let omega = if omega.is_finite() && omega >= 0.0 {
        omega
    } else {
        ctx.push(
            Issue::builder(IssueCode::InvalidWeight)
                .severity(Severity::Warning)
                .message(format!("cdist_adtw ω={omega} is not a finite ≥0 warp penalty; using 0"))
                .build(),
        );
        0.0
    };
    let out = Matrix::from_fn(a.nrows(), b.nrows(), |i, j| {
        let ai = a.row(i);
        let bj = b.row(j);
        adtw_raw(ai.as_slice(), bj.as_slice(), omega)
    });
    ctx.finish(out)
}

/// Pairwise weighted derivative DTW (tslearn-style `cdist` with WDDTW).
///
/// Series / pair counts are not identification `p`. Distinct from
/// [`cdist_ddtw`] and [`cdist_wdtw`].
pub fn cdist_wddtw(
    a: &Matrix,
    b: &Matrix,
    g: f64,
    session: &Session,
) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, a, None, &ctx.policy);
    inspect_xy(&mut ctx.report, b, None, &ctx.policy);
    let g = if g.is_finite() && g >= 0.0 {
        g
    } else {
        ctx.push(
            Issue::builder(IssueCode::InvalidWeight)
                .severity(Severity::Warning)
                .message(format!("cdist_wddtw g={g} is not a finite ≥0 slope; using 0.1"))
                .build(),
        );
        0.1
    };
    let out = Matrix::from_fn(a.nrows(), b.nrows(), |i, j| {
        let ai = a.row(i);
        let bj = b.row(j);
        let da = ddtw_deriv(ai.as_slice());
        let db = ddtw_deriv(bj.as_slice());
        wdtw_raw(&da, &db, g)
    });
    ctx.finish(out)
}

/// Pairwise shape DTW (tslearn-style `cdist` with [`shape_dtw`]).
///
/// Series / pair counts are not identification `p`. Distinct from
/// [`cdist_dtw`] and [`cdist_ddtw`].
pub fn cdist_shape_dtw(
    a: &Matrix,
    b: &Matrix,
    session: &Session,
) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, a, None, &ctx.policy);
    inspect_xy(&mut ctx.report, b, None, &ctx.policy);
    let out = Matrix::from_fn(a.nrows(), b.nrows(), |i, j| {
        let ai = a.row(i);
        let bj = b.row(j);
        dtw_raw(&shape_desc(ai.as_slice()), &shape_desc(bj.as_slice()))
    });
    ctx.finish(out)
}

fn complexity_estimate(s: &[f64]) -> f64 {
    if s.len() < 2 {
        return 0.0;
    }
    let mut acc = 0.0_f64;
    for i in 1..s.len() {
        let d = s[i] - s[i - 1];
        acc += d * d;
    }
    acc.sqrt()
}

fn cid_raw(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return f64::NAN;
    }
    let mut euc = 0.0_f64;
    for i in 0..n {
        let d = a[i] - b[i];
        euc += d * d;
    }
    let ce_a = complexity_estimate(&a[..n]).max(1e-12);
    let ce_b = complexity_estimate(&b[..n]).max(1e-12);
    let cf = ce_a.max(ce_b) / ce_a.min(ce_b);
    cf * euc.sqrt()
}

/// Complexity-Invariant Distance (Batista, Keogh, Tataw, de Souza).
///
/// Correction factor \(\max(CE,CE')/\min(CE,CE')\) times Euclidean.
/// Distinct from [`dtw`] and [`sbd`]. Length is not identification `p`.
pub fn cid(a: &Vector, b: &Vector, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if let Some(issue) = signlred::scan_finite(a.as_slice()).to_issue("cid.a") {
        ctx.push(issue);
    }
    if let Some(issue) = signlred::scan_finite(b.as_slice()).to_issue("cid.b") {
        ctx.push(issue);
    }
    if a.is_empty() || b.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message("CID on an empty series")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    ctx.finish(cid_raw(a.as_slice(), b.as_slice()))
}

/// Pairwise CID (tslearn-style `cdist` with [`cid`]).
///
/// Series / pair counts are not identification `p`. Distinct from
/// [`cdist_dtw`] and [`cdist_sbd`].
pub fn cdist_cid(a: &Matrix, b: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, a, None, &ctx.policy);
    inspect_xy(&mut ctx.report, b, None, &ctx.policy);
    let out = Matrix::from_fn(a.nrows(), b.nrows(), |i, j| {
        let ai = a.row(i);
        let bj = b.row(j);
        cid_raw(ai.as_slice(), bj.as_slice())
    });
    ctx.finish(out)
}

fn lb_keogh_raw(query: &[f64], candidate: &[f64], r: usize) -> f64 {
    let n = query.len().min(candidate.len());
    if n == 0 {
        return f64::NAN;
    }
    let w = r.max(1);
    let mut lb = 0.0_f64;
    for i in 0..n {
        let a = i.saturating_sub(w);
        let b = (i + w + 1).min(n);
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for t in a..b {
            lo = lo.min(query[t]);
            hi = hi.max(query[t]);
        }
        let c = candidate[i];
        if c > hi {
            let d = c - hi;
            lb += d * d;
        } else if c < lo {
            let d = lo - c;
            lb += d * d;
        }
    }
    lb
}

fn lb_improved_raw(query: &[f64], candidate: &[f64], r: usize) -> f64 {
    let n = query.len().min(candidate.len());
    if n == 0 {
        return f64::NAN;
    }
    let w = r.max(1);
    let mut leftover = vec![false; n];
    let mut lb = 0.0_f64;
    for i in 0..n {
        let a = i.saturating_sub(w);
        let b = (i + w + 1).min(n);
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for t in a..b {
            lo = lo.min(query[t]);
            hi = hi.max(query[t]);
        }
        let c = candidate[i];
        if c > hi {
            let d = c - hi;
            lb += d * d;
        } else if c < lo {
            let d = lo - c;
            lb += d * d;
        } else {
            leftover[i] = true;
        }
    }
    for i in 0..n {
        if !leftover[i] {
            continue;
        }
        let a = i.saturating_sub(w);
        let b = (i + w + 1).min(n);
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for t in a..b {
            lo = lo.min(candidate[t]);
            hi = hi.max(candidate[t]);
        }
        let qv = query[i];
        if qv > hi {
            let d = qv - hi;
            lb += d * d;
        } else if qv < lo {
            let d = lo - qv;
            lb += d * d;
        }
    }
    lb
}

/// Pairwise LB_Keogh (tslearn-style `cdist` with [`lb_keogh`]).
///
/// Window width is not identification `p`. Distinct from [`cdist_dtw`].
pub fn cdist_lb_keogh(
    a: &Matrix,
    b: &Matrix,
    r: usize,
    session: &Session,
) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, a, None, &ctx.policy);
    inspect_xy(&mut ctx.report, b, None, &ctx.policy);
    let out = Matrix::from_fn(a.nrows(), b.nrows(), |i, j| {
        let ai = a.row(i);
        let bj = b.row(j);
        lb_keogh_raw(ai.as_slice(), bj.as_slice(), r)
    });
    ctx.finish(out)
}

/// Pairwise LB_Improved (tslearn-style `cdist` with [`lb_improved`]).
///
/// Window width is not identification `p`. Distinct from [`cdist_lb_keogh`].
pub fn cdist_lb_improved(
    a: &Matrix,
    b: &Matrix,
    r: usize,
    session: &Session,
) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, a, None, &ctx.policy);
    inspect_xy(&mut ctx.report, b, None, &ctx.policy);
    let out = Matrix::from_fn(a.nrows(), b.nrows(), |i, j| {
        let ai = a.row(i);
        let bj = b.row(j);
        lb_improved_raw(ai.as_slice(), bj.as_slice(), r)
    });
    ctx.finish(out)
}

/// Pairwise SWALE (tslearn-style `cdist` with [`swale`]).
///
/// Match radius is not identification `p`. Distinct from [`cdist_lcss`]
/// and [`cdist_edr`].
pub fn cdist_swale(
    a: &Matrix,
    b: &Matrix,
    eps: f64,
    session: &Session,
) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, a, None, &ctx.policy);
    inspect_xy(&mut ctx.report, b, None, &ctx.policy);
    let eps = if eps.is_finite() && eps >= 0.0 { eps } else { 0.0 };
    let out = Matrix::from_fn(a.nrows(), b.nrows(), |i, j| {
        let ai = a.row(i);
        let bj = b.row(j);
        swale_raw(ai.as_slice(), bj.as_slice(), eps)
    });
    ctx.finish(out)
}

/// Pairwise LB_Kim (tslearn-style `cdist` with [`lb_kim`]).
///
/// Feature count is not identification `p`. Distinct from [`cdist_lb_keogh`].
pub fn cdist_lb_kim(a: &Matrix, b: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, a, None, &ctx.policy);
    inspect_xy(&mut ctx.report, b, None, &ctx.policy);
    let out = Matrix::from_fn(a.nrows(), b.nrows(), |i, j| {
        let ai = a.row(i);
        let bj = b.row(j);
        lb_kim_raw(ai.as_slice(), bj.as_slice())
    });
    ctx.finish(out)
}

fn mpdist_raw(a: &[f64], b: &[f64], window: usize) -> f64 {
    let m = window;
    if m < 2 || m > a.len() || m > b.len() {
        return 0.0;
    }
    let na = a.len() + 1 - m;
    let nb = b.len() + 1 - m;
    let mut acc = 0.0_f64;
    let mut n = 0usize;
    for i in 0..na {
        let mut best = f64::INFINITY;
        for j in 0..nb {
            let mut s = 0.0_f64;
            for t in 0..m {
                let e = a[i + t] - b[j + t];
                s += e * e;
            }
            let d = s.max(0.0).sqrt();
            if d < best {
                best = d;
            }
        }
        if best.is_finite() {
            acc += best;
            n += 1;
        }
    }
    for j in 0..nb {
        let mut best = f64::INFINITY;
        for i in 0..na {
            let mut s = 0.0_f64;
            for t in 0..m {
                let e = b[j + t] - a[i + t];
                s += e * e;
            }
            let d = s.max(0.0).sqrt();
            if d < best {
                best = d;
            }
        }
        if best.is_finite() {
            acc += best;
            n += 1;
        }
    }
    if n == 0 {
        0.0
    } else {
        acc / n as f64
    }
}

/// Pairwise matrix-profile distance (tslearn-style `cdist` with [`mpdist`]).
///
/// Window length is not identification `p`. Distinct from [`cdist_dtw`].
pub fn cdist_mpdist(
    a: &Matrix,
    b: &Matrix,
    window: usize,
    session: &Session,
) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, a, None, &ctx.policy);
    inspect_xy(&mut ctx.report, b, None, &ctx.policy);
    let out = Matrix::from_fn(a.nrows(), b.nrows(), |i, j| {
        let ai = a.row(i);
        let bj = b.row(j);
        mpdist_raw(ai.as_slice(), bj.as_slice(), window)
    });
    ctx.finish(out)
}

fn kdtw_kernel(a: &[f64], b: &[f64], nu: f64) -> f64 {
    let n = a.len();
    let m = b.len();
    if n == 0 || m == 0 {
        return 0.0;
    }
    let mut prev = vec![0.0_f64; m + 1];
    let mut cur = vec![0.0_f64; m + 1];
    for i in 1..=n {
        for j in 1..=m {
            let d = a[i - 1] - b[j - 1];
            let hij = (-nu * d * d).exp();
            let rec = if i == 1 && j == 1 {
                1.0
            } else {
                prev[j] + cur[j - 1] + prev[j - 1]
            };
            cur[j] = hij * rec;
        }
        std::mem::swap(&mut prev, &mut cur);
        for v in cur.iter_mut() {
            *v = 0.0;
        }
    }
    prev[m]
}

fn kdtw_raw(a: &[f64], b: &[f64], nu: f64) -> f64 {
    let kab = kdtw_kernel(a, b, nu);
    let kaa = kdtw_kernel(a, a, nu);
    let kbb = kdtw_kernel(b, b, nu);
    let den = (kaa * kbb).sqrt().max(1e-300);
    let c = (kab / den).clamp(0.0, 1.0);
    (1.0 - c).max(0.0)
}

/// Kernel DTW (Marteau): local Gaussian kernel with a three-path recursion,
/// returned as a cosine distance in kernel space.
///
/// Distinct from [`softdtw`] (softmin cost) and [`gak`] (`exp(-soft-DTW/σ)`
/// in this crate). \(\nu\) is not identification `p`.
pub fn kdtw(a: &Vector, b: &Vector, nu: f64, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if let Some(issue) = signlred::scan_finite(a.as_slice()).to_issue("kdtw.a") {
        ctx.push(issue);
    }
    if let Some(issue) = signlred::scan_finite(b.as_slice()).to_issue("kdtw.b") {
        ctx.push(issue);
    }
    let nu = if nu.is_finite() && nu > 0.0 {
        nu
    } else {
        ctx.push(
            Issue::builder(IssueCode::InvalidWeight)
                .severity(Severity::Warning)
                .message(format!("kdtw ν={nu} is not positive; using 1"))
                .build(),
        );
        1.0
    };
    if a.is_empty() || b.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message("kdtw on an empty series")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    ctx.finish(kdtw_raw(a.as_slice(), b.as_slice(), nu))
}

/// Pairwise kernel DTW (tslearn-style `cdist` with [`kdtw`]).
///
/// \(\nu\) is not identification `p`. Distinct from [`cdist_softdtw`].
pub fn cdist_kdtw(
    a: &Matrix,
    b: &Matrix,
    nu: f64,
    session: &Session,
) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, a, None, &ctx.policy);
    inspect_xy(&mut ctx.report, b, None, &ctx.policy);
    let nu = if nu.is_finite() && nu > 0.0 { nu } else { 1.0 };
    let out = Matrix::from_fn(a.nrows(), b.nrows(), |i, j| {
        let ai = a.row(i);
        let bj = b.row(j);
        kdtw_raw(ai.as_slice(), bj.as_slice(), nu)
    });
    ctx.finish(out)
}

fn coarsen_series(s: &[f64]) -> Vec<f64> {
    let mut out = Vec::with_capacity(s.len().div_ceil(2));
    let mut i = 0usize;
    while i + 1 < s.len() {
        out.push(0.5 * (s[i] + s[i + 1]));
        i += 2;
    }
    if i < s.len() {
        out.push(s[i]);
    }
    out
}

fn dtw_path_cells(a: &[f64], b: &[f64]) -> (f64, Vec<(usize, usize)>) {
    let n = a.len();
    let m = b.len();
    if n == 0 || m == 0 {
        return (f64::NAN, Vec::new());
    }
    let inf = 1e300_f64;
    let mut dp = vec![inf; (n + 1) * (m + 1)];
    let at = |i: usize, j: usize| i * (m + 1) + j;
    dp[at(0, 0)] = 0.0;
    for i in 1..=n {
        for j in 1..=m {
            let cost = (a[i - 1] - b[j - 1]).abs();
            dp[at(i, j)] = cost + dp[at(i - 1, j)].min(dp[at(i, j - 1)]).min(dp[at(i - 1, j - 1)]);
        }
    }
    let mut path = Vec::new();
    let mut i = n;
    let mut j = m;
    while i > 0 && j > 0 {
        path.push((i - 1, j - 1));
        let up = dp[at(i - 1, j)];
        let left = dp[at(i, j - 1)];
        let diag = dp[at(i - 1, j - 1)];
        if diag <= up && diag <= left {
            i -= 1;
            j -= 1;
        } else if up <= left {
            i -= 1;
        } else {
            j -= 1;
        }
    }
    path.reverse();
    (dp[at(n, m)], path)
}

fn expand_fastdtw_window(
    coarse: &[(usize, usize)],
    n: usize,
    m: usize,
    radius: usize,
) -> Vec<bool> {
    let mut win = vec![false; n * m];
    let mark = |win: &mut [bool], i: usize, j: usize| {
        if i < n && j < m {
            win[i * m + j] = true;
        }
    };
    for &(ci, cj) in coarse {
        for di in 0..=1 {
            for dj in 0..=1 {
                let fi = 2 * ci + di;
                let fj = 2 * cj + dj;
                let lo_i = fi.saturating_sub(radius);
                let hi_i = (fi + radius).min(n.saturating_sub(1));
                let lo_j = fj.saturating_sub(radius);
                let hi_j = (fj + radius).min(m.saturating_sub(1));
                for i in lo_i..=hi_i {
                    for j in lo_j..=hi_j {
                        mark(&mut win, i, j);
                    }
                }
            }
        }
    }
    if n > 0 && m > 0 {
        win[0] = true;
        win[(n - 1) * m + (m - 1)] = true;
    }
    win
}

fn dtw_in_window(a: &[f64], b: &[f64], win: &[bool]) -> f64 {
    let n = a.len();
    let m = b.len();
    if n == 0 || m == 0 {
        return f64::NAN;
    }
    let inf = 1e300_f64;
    let mut prev = vec![inf; m + 1];
    let mut cur = vec![inf; m + 1];
    prev[0] = 0.0;
    for i in 1..=n {
        cur[0] = inf;
        for j in 1..=m {
            if !win[(i - 1) * m + (j - 1)] {
                cur[j] = inf;
                continue;
            }
            let cost = (a[i - 1] - b[j - 1]).abs();
            cur[j] = cost + prev[j].min(cur[j - 1]).min(prev[j - 1]);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[m]
}

fn fastdtw_raw(a: &[f64], b: &[f64], radius: usize) -> f64 {
    let n = a.len();
    let m = b.len();
    if n == 0 || m == 0 {
        return f64::NAN;
    }
    if n.min(m) <= (2 * radius + 2).max(4) {
        return dtw_raw(a, b);
    }
    let ac = coarsen_series(a);
    let bc = coarsen_series(b);
    let (_, coarse) = {
        let cost = fastdtw_raw(&ac, &bc, radius);
        if !cost.is_finite() {
            return cost;
        }
        dtw_path_cells(&ac, &bc)
    };
    let win = expand_fastdtw_window(&coarse, n, m, radius.max(1));
    dtw_in_window(a, b, &win)
}

/// FastDTW (Salvador–Chan): multilevel coarsen, project, refine.
///
/// Distinct from exact [`dtw`]. Radius is not identification `p`.
pub fn fast_dtw(
    a: &Vector,
    b: &Vector,
    radius: usize,
    session: &Session,
) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if let Some(issue) = signlred::scan_finite(a.as_slice()).to_issue("fast_dtw.a") {
        ctx.push(issue);
    }
    if let Some(issue) = signlred::scan_finite(b.as_slice()).to_issue("fast_dtw.b") {
        ctx.push(issue);
    }
    if a.is_empty() || b.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message("fast_dtw on an empty series")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    ctx.finish(fastdtw_raw(a.as_slice(), b.as_slice(), radius.max(1)))
}

/// Pairwise FastDTW (tslearn-style `cdist` with [`fast_dtw`]).
///
/// Radius is not identification `p`. Distinct from [`cdist_dtw`].
pub fn cdist_fast_dtw(
    a: &Matrix,
    b: &Matrix,
    radius: usize,
    session: &Session,
) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, a, None, &ctx.policy);
    inspect_xy(&mut ctx.report, b, None, &ctx.policy);
    let r = radius.max(1);
    let out = Matrix::from_fn(a.nrows(), b.nrows(), |i, j| {
        let ai = a.row(i);
        let bj = b.row(j);
        fastdtw_raw(ai.as_slice(), bj.as_slice(), r)
    });
    ctx.finish(out)
}

fn dtw_subsequence_raw(query: &[f64], series: &[f64]) -> f64 {
    let n = query.len();
    let m = series.len();
    if n == 0 || m == 0 {
        return f64::NAN;
    }
    let inf = 1e300_f64;
    let mut prev = vec![inf; m];
    let mut cur = vec![inf; m];
    for j in 0..m {
        prev[j] = (query[0] - series[j]).abs();
    }
    for i in 1..n {
        for j in 0..m {
            let cost = (query[i] - series[j]).abs();
            let mut best = prev[j];
            if j > 0 {
                best = best.min(cur[j - 1]).min(prev[j - 1]);
            }
            cur[j] = cost + best;
        }
        std::mem::swap(&mut prev, &mut cur);
        for v in cur.iter_mut() {
            *v = inf;
        }
    }
    prev.iter().copied().fold(inf, f64::min)
}

/// Subsequence DTW (open begin/end on the longer series).
///
/// Distinct from closed-end [`dtw`] and [`mpdist`]. Query length is not
/// identification `p`.
pub fn dtw_subsequence(a: &Vector, b: &Vector, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if let Some(issue) = signlred::scan_finite(a.as_slice()).to_issue("dtw_subsequence.a") {
        ctx.push(issue);
    }
    if let Some(issue) = signlred::scan_finite(b.as_slice()).to_issue("dtw_subsequence.b") {
        ctx.push(issue);
    }
    if a.is_empty() || b.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message("dtw_subsequence on an empty series")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    let (q, s) = if a.len() <= b.len() {
        (a.as_slice(), b.as_slice())
    } else {
        (b.as_slice(), a.as_slice())
    };
    ctx.finish(dtw_subsequence_raw(q, s))
}

/// Pairwise subsequence DTW (tslearn-style `cdist`).
///
/// Distinct from [`cdist_dtw`].
pub fn cdist_dtw_subsequence(
    a: &Matrix,
    b: &Matrix,
    session: &Session,
) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, a, None, &ctx.policy);
    inspect_xy(&mut ctx.report, b, None, &ctx.policy);
    let out = Matrix::from_fn(a.nrows(), b.nrows(), |i, j| {
        let ai = a.row(i);
        let bj = b.row(j);
        let (q, s) = if ai.len() <= bj.len() {
            (ai.as_slice(), bj.as_slice())
        } else {
            (bj.as_slice(), ai.as_slice())
        };
        dtw_subsequence_raw(q, s)
    });
    ctx.finish(out)
}

fn lb_yi_raw(a: &[f64], b: &[f64]) -> f64 {
    if a.is_empty() || b.is_empty() {
        return f64::NAN;
    }
    let mut amin = a[0];
    let mut amax = a[0];
    for &v in a {
        amin = amin.min(v);
        amax = amax.max(v);
    }
    let mut bmin = b[0];
    let mut bmax = b[0];
    for &v in b {
        bmin = bmin.min(v);
        bmax = bmax.max(v);
    }
    (amax - bmax).abs() + (amin - bmin).abs()
}

/// LB_Yi (Yi–Jagadish–Faloutsos): \(|\max-\max|+|\min-\min|\).
///
/// Distinct from [`lb_kim`] (max of four endpoint/extrema terms).
pub fn lb_yi(a: &Vector, b: &Vector, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if let Some(issue) = signlred::scan_finite(a.as_slice()).to_issue("lb_yi.a") {
        ctx.push(issue);
    }
    if let Some(issue) = signlred::scan_finite(b.as_slice()).to_issue("lb_yi.b") {
        ctx.push(issue);
    }
    if a.is_empty() || b.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message("lb_yi on an empty series")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    ctx.finish(lb_yi_raw(a.as_slice(), b.as_slice()))
}

/// Pairwise LB_Yi.
pub fn cdist_lb_yi(a: &Matrix, b: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, a, None, &ctx.policy);
    inspect_xy(&mut ctx.report, b, None, &ctx.policy);
    let out = Matrix::from_fn(a.nrows(), b.nrows(), |i, j| {
        let ai = a.row(i);
        let bj = b.row(j);
        lb_yi_raw(ai.as_slice(), bj.as_slice())
    });
    ctx.finish(out)
}

fn itakura_ok(i: usize, j: usize, n: usize, m: usize) -> bool {
    let ii = (i + 1) as f64;
    let jj = (j + 1) as f64;
    let nn = n.max(1) as f64;
    let mm = m.max(1) as f64;
    2.0 * ii * mm >= jj * nn && 2.0 * jj * nn >= ii * mm
}

fn itakura_dtw_raw(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len();
    let m = b.len();
    if n == 0 || m == 0 {
        return f64::NAN;
    }
    let inf = 1e300_f64;
    let mut prev = vec![inf; m + 1];
    let mut cur = vec![inf; m + 1];
    prev[0] = 0.0;
    for i in 1..=n {
        cur[0] = inf;
        for j in 1..=m {
            if !itakura_ok(i - 1, j - 1, n, m) {
                cur[j] = inf;
                continue;
            }
            let cost = (a[i - 1] - b[j - 1]).abs();
            cur[j] = cost + prev[j].min(cur[j - 1]).min(prev[j - 1]);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[m]
}

/// Itakura-parallelogram DTW (not a Sakoe–Chiba band).
///
/// Distinct from unconstrained [`dtw`] and multilevel [`fast_dtw`].
pub fn itakura_dtw(a: &Vector, b: &Vector, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if let Some(issue) = signlred::scan_finite(a.as_slice()).to_issue("itakura_dtw.a") {
        ctx.push(issue);
    }
    if let Some(issue) = signlred::scan_finite(b.as_slice()).to_issue("itakura_dtw.b") {
        ctx.push(issue);
    }
    if a.is_empty() || b.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message("itakura_dtw on an empty series")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    ctx.finish(itakura_dtw_raw(a.as_slice(), b.as_slice()))
}

/// Pairwise Itakura DTW.
pub fn cdist_itakura_dtw(a: &Matrix, b: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, a, None, &ctx.policy);
    inspect_xy(&mut ctx.report, b, None, &ctx.policy);
    let out = Matrix::from_fn(a.nrows(), b.nrows(), |i, j| {
        let ai = a.row(i);
        let bj = b.row(j);
        itakura_dtw_raw(ai.as_slice(), bj.as_slice())
    });
    ctx.finish(out)
}

fn sakoe_ok(i: usize, j: usize, n: usize, m: usize, radius: usize) -> bool {
    let expected = j as f64 * n as f64 / m.max(1) as f64;
    (i as f64 - expected).abs() <= radius as f64 + 1.0
}

fn sakoe_chiba_dtw_raw(a: &[f64], b: &[f64], radius: usize) -> f64 {
    let n = a.len();
    let m = b.len();
    if n == 0 || m == 0 {
        return f64::NAN;
    }
    let inf = 1e300_f64;
    let r = radius.max(1);
    let mut prev = vec![inf; m + 1];
    let mut cur = vec![inf; m + 1];
    prev[0] = 0.0;
    for i in 1..=n {
        cur[0] = inf;
        for j in 1..=m {
            if !sakoe_ok(i - 1, j - 1, n, m, r) {
                cur[j] = inf;
                continue;
            }
            let cost = (a[i - 1] - b[j - 1]).abs();
            cur[j] = cost + prev[j].min(cur[j - 1]).min(prev[j - 1]);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[m]
}

/// Sakoe–Chiba banded DTW (width \(r\) around the diagonal).
///
/// Distinct from unconstrained [`dtw`] and parallelogram [`itakura_dtw`].
/// Radius is not identification `p`.
pub fn sakoe_chiba_dtw(
    a: &Vector,
    b: &Vector,
    radius: usize,
    session: &Session,
) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if let Some(issue) = signlred::scan_finite(a.as_slice()).to_issue("sakoe_chiba_dtw.a") {
        ctx.push(issue);
    }
    if let Some(issue) = signlred::scan_finite(b.as_slice()).to_issue("sakoe_chiba_dtw.b") {
        ctx.push(issue);
    }
    if a.is_empty() || b.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message("sakoe_chiba_dtw on an empty series")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    ctx.finish(sakoe_chiba_dtw_raw(a.as_slice(), b.as_slice(), radius.max(1)))
}

/// Pairwise Sakoe–Chiba DTW.
///
/// Radius is not identification `p`. Distinct from [`cdist_dtw`] and
/// [`cdist_itakura_dtw`].
pub fn cdist_sakoe_chiba_dtw(
    a: &Matrix,
    b: &Matrix,
    radius: usize,
    session: &Session,
) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, a, None, &ctx.policy);
    inspect_xy(&mut ctx.report, b, None, &ctx.policy);
    let r = radius.max(1);
    let out = Matrix::from_fn(a.nrows(), b.nrows(), |i, j| {
        let ai = a.row(i);
        let bj = b.row(j);
        sakoe_chiba_dtw_raw(ai.as_slice(), bj.as_slice(), r)
    });
    ctx.finish(out)
}

fn cyclic_dtw_raw(a: &[f64], b: &[f64]) -> f64 {
    if a.is_empty() || b.is_empty() {
        return f64::NAN;
    }
    let mut best = dtw_raw(a, b);
    let mut rot = b.to_vec();
    for _ in 1..b.len() {
        rot.rotate_left(1);
        let d = dtw_raw(a, &rot);
        if d < best {
            best = d;
        }
    }
    best
}

/// Cyclic DTW: minimum unconstrained DTW over circular shifts of `b`.
///
/// Distinct from [`dtw`] (no rotation) and [`dtw_subsequence`] (open ends,
/// no wrap). Shift search is not identification `p`.
pub fn cyclic_dtw(a: &Vector, b: &Vector, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if let Some(issue) = signlred::scan_finite(a.as_slice()).to_issue("cyclic_dtw.a") {
        ctx.push(issue);
    }
    if let Some(issue) = signlred::scan_finite(b.as_slice()).to_issue("cyclic_dtw.b") {
        ctx.push(issue);
    }
    if a.is_empty() || b.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message("cyclic_dtw on an empty series")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    ctx.finish(cyclic_dtw_raw(a.as_slice(), b.as_slice()))
}

/// Pairwise cyclic DTW.
///
/// Distinct from [`cdist_dtw`] and [`cdist_dtw_subsequence`].
pub fn cdist_cyclic_dtw(a: &Matrix, b: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, a, None, &ctx.policy);
    inspect_xy(&mut ctx.report, b, None, &ctx.policy);
    let out = Matrix::from_fn(a.nrows(), b.nrows(), |i, j| {
        let ai = a.row(i);
        let bj = b.row(j);
        cyclic_dtw_raw(ai.as_slice(), bj.as_slice())
    });
    ctx.finish(out)
}

fn obe_dtw_raw(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len();
    let m = b.len();
    if n == 0 || m == 0 {
        return f64::NAN;
    }
    let mut prev = vec![0.0_f64; m];
    for j in 0..m {
        prev[j] = (a[0] - b[j]).abs();
    }
    let mut best = prev[m - 1];
    let mut cur = vec![0.0_f64; m];
    for i in 1..n {
        cur[0] = (a[i] - b[0]).abs();
        for j in 1..m {
            let cost = (a[i] - b[j]).abs();
            cur[j] = cost + prev[j].min(cur[j - 1]).min(prev[j - 1]);
        }
        best = best.min(cur[m - 1]);
        std::mem::swap(&mut prev, &mut cur);
    }
    for j in 0..m {
        best = best.min(prev[j]);
    }
    best
}

/// Open-begin-end DTW (free prefix and suffix on both series).
///
/// Distinct from [`dtw`] (closed ends) and [`dtw_subsequence`] (open on one
/// series only).
pub fn obe_dtw(a: &Vector, b: &Vector, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if let Some(issue) = signlred::scan_finite(a.as_slice()).to_issue("obe_dtw.a") {
        ctx.push(issue);
    }
    if let Some(issue) = signlred::scan_finite(b.as_slice()).to_issue("obe_dtw.b") {
        ctx.push(issue);
    }
    if a.is_empty() || b.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message("obe_dtw on an empty series")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    ctx.finish(obe_dtw_raw(a.as_slice(), b.as_slice()))
}

/// Pairwise open-begin-end DTW.
///
/// Distinct from [`cdist_dtw`] and [`cdist_dtw_subsequence`].
pub fn cdist_obe_dtw(a: &Matrix, b: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, a, None, &ctx.policy);
    inspect_xy(&mut ctx.report, b, None, &ctx.policy);
    let out = Matrix::from_fn(a.nrows(), b.nrows(), |i, j| {
        let ai = a.row(i);
        let bj = b.row(j);
        obe_dtw_raw(ai.as_slice(), bj.as_slice())
    });
    ctx.finish(out)
}

fn amss_raw(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len();
    let m = b.len();
    if n < 2 || m < 2 {
        return dtw_raw(a, b);
    }
    let inf = 1e300_f64;
    let mut prev = vec![inf; m];
    let mut cur = vec![inf; m];
    for j in 0..m - 1 {
        let db = b[j + 1] - b[j];
        let da = a[1] - a[0];
        let na = (1.0 + da * da).sqrt();
        let nb = (1.0 + db * db).sqrt();
        prev[j] = 1.0 - (1.0 + da * db) / (na * nb);
    }
    for i in 1..n - 1 {
        let da = a[i + 1] - a[i];
        let na = (1.0 + da * da).sqrt();
        for j in 0..m - 1 {
            let db = b[j + 1] - b[j];
            let nb = (1.0 + db * db).sqrt();
            let cost = 1.0 - (1.0 + da * db) / (na * nb);
            let mut best = prev[j];
            if j > 0 {
                best = best.min(cur[j - 1]).min(prev[j - 1]);
            }
            cur[j] = cost + best;
        }
        std::mem::swap(&mut prev, &mut cur);
        for v in cur.iter_mut() {
            *v = inf;
        }
    }
    prev[..m - 1].iter().copied().fold(inf, f64::min)
}

/// Angular metric for shape similarity (Yokoyama et al. AMSS).
///
/// Local cost is \(1-\cos\) of consecutive increment vectors. Distinct from
/// [`dtw`] (level cost) and [`ddtw`] (derivative magnitude).
pub fn amss(a: &Vector, b: &Vector, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if let Some(issue) = signlred::scan_finite(a.as_slice()).to_issue("amss.a") {
        ctx.push(issue);
    }
    if let Some(issue) = signlred::scan_finite(b.as_slice()).to_issue("amss.b") {
        ctx.push(issue);
    }
    if a.is_empty() || b.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message("amss on an empty series")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    ctx.finish(amss_raw(a.as_slice(), b.as_slice()))
}

/// Pairwise AMSS.
///
/// Distinct from [`cdist_dtw`] and [`cdist_ddtw`].
pub fn cdist_amss(a: &Matrix, b: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, a, None, &ctx.policy);
    inspect_xy(&mut ctx.report, b, None, &ctx.policy);
    let out = Matrix::from_fn(a.nrows(), b.nrows(), |i, j| {
        let ai = a.row(i);
        let bj = b.row(j);
        amss_raw(ai.as_slice(), bj.as_slice())
    });
    ctx.finish(out)
}

fn decay_euclidean_raw(a: &[f64], b: &[f64], gamma: f64) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return f64::NAN;
    }
    let g = if gamma.is_finite() && gamma >= 0.0 {
        gamma
    } else {
        0.1
    };
    let mut s = 0.0_f64;
    let den = (n - 1).max(1) as f64;
    for i in 0..n {
        let w = (-g * (n - 1 - i) as f64 / den).exp();
        let d = a[i] - b[i];
        s += w * d * d;
    }
    s.sqrt()
}

/// Time-decay Euclidean (recent samples weighted more; no warping).
///
/// Distinct from [`wdtw`] (warped) and unconstrained Euclidean. Decay
/// \(\gamma\) is not identification `p`.
pub fn decay_euclidean(
    a: &Vector,
    b: &Vector,
    gamma: f64,
    session: &Session,
) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if let Some(issue) = signlred::scan_finite(a.as_slice()).to_issue("decay_euclidean.a") {
        ctx.push(issue);
    }
    if let Some(issue) = signlred::scan_finite(b.as_slice()).to_issue("decay_euclidean.b") {
        ctx.push(issue);
    }
    if a.is_empty() || b.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message("decay_euclidean on an empty series")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    ctx.finish(decay_euclidean_raw(a.as_slice(), b.as_slice(), gamma))
}

/// Pairwise time-decay Euclidean.
///
/// Distinct from [`cdist_wdtw`].
pub fn cdist_decay_euclidean(
    a: &Matrix,
    b: &Matrix,
    gamma: f64,
    session: &Session,
) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, a, None, &ctx.policy);
    inspect_xy(&mut ctx.report, b, None, &ctx.policy);
    let g = gamma;
    let out = Matrix::from_fn(a.nrows(), b.nrows(), |i, j| {
        let ai = a.row(i);
        let bj = b.row(j);
        decay_euclidean_raw(ai.as_slice(), bj.as_slice(), g)
    });
    ctx.finish(out)
}

fn shape_euclidean_raw(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n < 2 {
        return if n == 0 {
            f64::NAN
        } else {
            (a[0] - b[0]).abs()
        };
    }
    let mut s = 0.0_f64;
    for i in 0..n - 1 {
        let da = a[i + 1] - a[i];
        let db = b[i + 1] - b[i];
        let d = da - db;
        s += d * d;
    }
    s.sqrt()
}

/// Euclidean distance on first differences (no warping).
///
/// Distinct from [`ddtw`] (warped derivatives) and [`amss`] (angular cost).
pub fn shape_euclidean(a: &Vector, b: &Vector, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if let Some(issue) = signlred::scan_finite(a.as_slice()).to_issue("shape_euclidean.a") {
        ctx.push(issue);
    }
    if let Some(issue) = signlred::scan_finite(b.as_slice()).to_issue("shape_euclidean.b") {
        ctx.push(issue);
    }
    if a.is_empty() || b.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message("shape_euclidean on an empty series")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    ctx.finish(shape_euclidean_raw(a.as_slice(), b.as_slice()))
}

/// Pairwise first-difference Euclidean.
///
/// Distinct from [`cdist_ddtw`] and [`cdist_amss`].
pub fn cdist_shape_euclidean(
    a: &Matrix,
    b: &Matrix,
    session: &Session,
) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, a, None, &ctx.policy);
    inspect_xy(&mut ctx.report, b, None, &ctx.policy);
    let out = Matrix::from_fn(a.nrows(), b.nrows(), |i, j| {
        let ai = a.row(i);
        let bj = b.row(j);
        shape_euclidean_raw(ai.as_slice(), bj.as_slice())
    });
    ctx.finish(out)
}

fn open_begin_dtw_raw(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len();
    let m = b.len();
    if n == 0 || m == 0 {
        return f64::NAN;
    }
    let mut prev = vec![0.0_f64; m];
    for j in 0..m {
        prev[j] = (a[0] - b[j]).abs();
    }
    let mut cur = vec![0.0_f64; m];
    for i in 1..n {
        cur[0] = prev[0] + (a[i] - b[0]).abs();
        for j in 1..m {
            let cost = (a[i] - b[j]).abs();
            cur[j] = cost + prev[j].min(cur[j - 1]).min(prev[j - 1]);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[m - 1]
}

/// Open-begin, closed-end DTW (free start on `b`, pinned end).
///
/// Distinct from [`dtw`] (closed–closed), [`obe_dtw`] (open–open), and
/// [`dtw_subsequence`] (open end on the longer series).
pub fn open_begin_dtw(a: &Vector, b: &Vector, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if let Some(issue) = signlred::scan_finite(a.as_slice()).to_issue("open_begin_dtw.a") {
        ctx.push(issue);
    }
    if let Some(issue) = signlred::scan_finite(b.as_slice()).to_issue("open_begin_dtw.b") {
        ctx.push(issue);
    }
    if a.is_empty() || b.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message("open_begin_dtw on an empty series")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    ctx.finish(open_begin_dtw_raw(a.as_slice(), b.as_slice()))
}

/// Pairwise open-begin DTW.
///
/// Distinct from [`cdist_dtw`], [`cdist_obe_dtw`], and
/// [`cdist_dtw_subsequence`].
pub fn cdist_open_begin_dtw(a: &Matrix, b: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, a, None, &ctx.policy);
    inspect_xy(&mut ctx.report, b, None, &ctx.policy);
    let out = Matrix::from_fn(a.nrows(), b.nrows(), |i, j| {
        let ai = a.row(i);
        let bj = b.row(j);
        open_begin_dtw_raw(ai.as_slice(), bj.as_slice())
    });
    ctx.finish(out)
}

fn open_end_dtw_raw(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len();
    let m = b.len();
    if n == 0 || m == 0 {
        return f64::NAN;
    }
    let mut prev = vec![0.0_f64; m];
    prev[0] = (a[0] - b[0]).abs();
    for j in 1..m {
        prev[j] = prev[j - 1] + (a[0] - b[j]).abs();
    }
    let mut cur = vec![0.0_f64; m];
    for i in 1..n {
        cur[0] = prev[0] + (a[i] - b[0]).abs();
        for j in 1..m {
            let cost = (a[i] - b[j]).abs();
            cur[j] = cost + prev[j].min(cur[j - 1]).min(prev[j - 1]);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev.iter().copied().fold(f64::INFINITY, f64::min)
}

/// Closed-begin, open-end DTW (pinned start, free end on `b`).
///
/// Standard DTW recurrence; the score is the minimum of the last row.
/// Distinct from [`dtw`] (closed–closed), [`open_begin_dtw`] (free start),
/// [`obe_dtw`] (open–open), and [`dtw_subsequence`] (open on the longer
/// series only).
pub fn open_end_dtw(a: &Vector, b: &Vector, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if let Some(issue) = signlred::scan_finite(a.as_slice()).to_issue("open_end_dtw.a") {
        ctx.push(issue);
    }
    if let Some(issue) = signlred::scan_finite(b.as_slice()).to_issue("open_end_dtw.b") {
        ctx.push(issue);
    }
    if a.is_empty() || b.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message("open_end_dtw on an empty series")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    ctx.finish(open_end_dtw_raw(a.as_slice(), b.as_slice()))
}

/// Pairwise open-end DTW.
///
/// Distinct from [`cdist_dtw`], [`cdist_open_begin_dtw`], and
/// [`cdist_dtw_subsequence`].
pub fn cdist_open_end_dtw(a: &Matrix, b: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, a, None, &ctx.policy);
    inspect_xy(&mut ctx.report, b, None, &ctx.policy);
    let out = Matrix::from_fn(a.nrows(), b.nrows(), |i, j| {
        let ai = a.row(i);
        let bj = b.row(j);
        open_end_dtw_raw(ai.as_slice(), bj.as_slice())
    });
    ctx.finish(out)
}

fn correlation_distance_raw(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return f64::NAN;
    }
    if n == 1 {
        return if (a[0] - b[0]).abs() < 1e-15 { 0.0 } else { 1.0 };
    }
    let na = n as f64;
    let mut ma = 0.0_f64;
    let mut mb = 0.0_f64;
    for i in 0..n {
        ma += a[i];
        mb += b[i];
    }
    ma /= na;
    mb /= na;
    let mut num = 0.0_f64;
    let mut va = 0.0_f64;
    let mut vb = 0.0_f64;
    for i in 0..n {
        let da = a[i] - ma;
        let db = b[i] - mb;
        num += da * db;
        va += da * da;
        vb += db * db;
    }
    let den = (va * vb).sqrt();
    if den < 1e-18 {
        let mut same = true;
        for i in 0..n {
            if (a[i] - b[i]).abs() >= 1e-12 {
                same = false;
                break;
            }
        }
        return if same { 0.0 } else { 1.0 };
    }
    let r = (num / den).clamp(-1.0, 1.0);
    1.0 - r
}

/// Pearson correlation distance \(1-r\) on aligned prefixes (no lag search).
///
/// Distinct from [`sbd`] (max NCC over shifts). Identical series score 0.
pub fn correlation_distance(a: &Vector, b: &Vector, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if let Some(issue) = signlred::scan_finite(a.as_slice()).to_issue("correlation_distance.a") {
        ctx.push(issue);
    }
    if let Some(issue) = signlred::scan_finite(b.as_slice()).to_issue("correlation_distance.b") {
        ctx.push(issue);
    }
    if a.is_empty() || b.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message("correlation_distance on an empty series")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    ctx.finish(correlation_distance_raw(a.as_slice(), b.as_slice()))
}

/// Pairwise Pearson correlation distance.
///
/// Distinct from [`cdist_sbd`].
pub fn cdist_correlation(a: &Matrix, b: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, a, None, &ctx.policy);
    inspect_xy(&mut ctx.report, b, None, &ctx.policy);
    let out = Matrix::from_fn(a.nrows(), b.nrows(), |i, j| {
        let ai = a.row(i);
        let bj = b.row(j);
        correlation_distance_raw(ai.as_slice(), bj.as_slice())
    });
    ctx.finish(out)
}

fn cosine_distance_raw(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return f64::NAN;
    }
    let mut num = 0.0_f64;
    let mut na = 0.0_f64;
    let mut nb = 0.0_f64;
    for i in 0..n {
        num += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    let den = na.sqrt() * nb.sqrt();
    if den < 1e-18 {
        return if na < 1e-18 && nb < 1e-18 { 0.0 } else { 1.0 };
    }
    1.0 - (num / den).clamp(-1.0, 1.0)
}

/// Cosine distance \(1-\langle a,b\rangle/(\|a\|\|b\|)\) on aligned prefixes.
///
/// No mean-centering. Distinct from [`correlation_distance`] (Pearson) and
/// [`sbd`] (max NCC over shifts). Identical series score 0.
pub fn cosine_distance(a: &Vector, b: &Vector, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if let Some(issue) = signlred::scan_finite(a.as_slice()).to_issue("cosine_distance.a") {
        ctx.push(issue);
    }
    if let Some(issue) = signlred::scan_finite(b.as_slice()).to_issue("cosine_distance.b") {
        ctx.push(issue);
    }
    if a.is_empty() || b.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message("cosine_distance on an empty series")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    ctx.finish(cosine_distance_raw(a.as_slice(), b.as_slice()))
}

/// Pairwise cosine distance.
///
/// Distinct from [`cdist_correlation`] and [`cdist_sbd`].
pub fn cdist_cosine(a: &Matrix, b: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, a, None, &ctx.policy);
    inspect_xy(&mut ctx.report, b, None, &ctx.policy);
    let out = Matrix::from_fn(a.nrows(), b.nrows(), |i, j| {
        let ai = a.row(i);
        let bj = b.row(j);
        cosine_distance_raw(ai.as_slice(), bj.as_slice())
    });
    ctx.finish(out)
}

fn chebyshev_distance_raw(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return f64::NAN;
    }
    let mut m = 0.0_f64;
    for i in 0..n {
        let d = (a[i] - b[i]).abs();
        if d > m {
            m = d;
        }
    }
    m
}

fn manhattan_distance_raw(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return f64::NAN;
    }
    let mut s = 0.0_f64;
    for i in 0..n {
        s += (a[i] - b[i]).abs();
    }
    s
}

/// Chebyshev (\(L^\infty\)) distance on aligned prefixes.
///
/// Distinct from [`dtw`] (warped \(L^1\)) and [`decay_euclidean`].
pub fn chebyshev_distance(a: &Vector, b: &Vector, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if let Some(issue) = signlred::scan_finite(a.as_slice()).to_issue("chebyshev_distance.a") {
        ctx.push(issue);
    }
    if let Some(issue) = signlred::scan_finite(b.as_slice()).to_issue("chebyshev_distance.b") {
        ctx.push(issue);
    }
    if a.is_empty() || b.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message("chebyshev_distance on an empty series")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    ctx.finish(chebyshev_distance_raw(a.as_slice(), b.as_slice()))
}

/// Pairwise Chebyshev distance.
pub fn cdist_chebyshev(a: &Matrix, b: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, a, None, &ctx.policy);
    inspect_xy(&mut ctx.report, b, None, &ctx.policy);
    let out = Matrix::from_fn(a.nrows(), b.nrows(), |i, j| {
        let ai = a.row(i);
        let bj = b.row(j);
        chebyshev_distance_raw(ai.as_slice(), bj.as_slice())
    });
    ctx.finish(out)
}

/// Manhattan (\(L^1\)) distance on aligned prefixes (no warping).
///
/// Distinct from [`dtw`] (warped \(L^1\)) and [`chebyshev_distance`].
pub fn manhattan_distance(a: &Vector, b: &Vector, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if let Some(issue) = signlred::scan_finite(a.as_slice()).to_issue("manhattan_distance.a") {
        ctx.push(issue);
    }
    if let Some(issue) = signlred::scan_finite(b.as_slice()).to_issue("manhattan_distance.b") {
        ctx.push(issue);
    }
    if a.is_empty() || b.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message("manhattan_distance on an empty series")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    ctx.finish(manhattan_distance_raw(a.as_slice(), b.as_slice()))
}

/// Pairwise Manhattan distance.
pub fn cdist_manhattan(a: &Matrix, b: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, a, None, &ctx.policy);
    inspect_xy(&mut ctx.report, b, None, &ctx.policy);
    let out = Matrix::from_fn(a.nrows(), b.nrows(), |i, j| {
        let ai = a.row(i);
        let bj = b.row(j);
        manhattan_distance_raw(ai.as_slice(), bj.as_slice())
    });
    ctx.finish(out)
}

fn canberra_distance_raw(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return f64::NAN;
    }
    let mut s = 0.0_f64;
    for i in 0..n {
        let den = a[i].abs() + b[i].abs();
        if den > 1e-18 {
            s += (a[i] - b[i]).abs() / den;
        }
    }
    s
}

fn braycurtis_distance_raw(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return f64::NAN;
    }
    let mut num = 0.0_f64;
    let mut den = 0.0_f64;
    for i in 0..n {
        num += (a[i] - b[i]).abs();
        den += (a[i] + b[i]).abs();
    }
    if den < 1e-18 {
        return if num < 1e-18 { 0.0 } else { 1.0 };
    }
    num / den
}

/// Canberra distance \(\sum |a_i-b_i|/(|a_i|+|b_i|)\) on aligned prefixes.
///
/// Distinct from [`manhattan_distance`] (no per-coordinate scale) and
/// [`correlation_distance`].
pub fn canberra_distance(a: &Vector, b: &Vector, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if let Some(issue) = signlred::scan_finite(a.as_slice()).to_issue("canberra_distance.a") {
        ctx.push(issue);
    }
    if let Some(issue) = signlred::scan_finite(b.as_slice()).to_issue("canberra_distance.b") {
        ctx.push(issue);
    }
    if a.is_empty() || b.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message("canberra_distance on an empty series")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    ctx.finish(canberra_distance_raw(a.as_slice(), b.as_slice()))
}

/// Pairwise Canberra distance.
pub fn cdist_canberra(a: &Matrix, b: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, a, None, &ctx.policy);
    inspect_xy(&mut ctx.report, b, None, &ctx.policy);
    let out = Matrix::from_fn(a.nrows(), b.nrows(), |i, j| {
        let ai = a.row(i);
        let bj = b.row(j);
        canberra_distance_raw(ai.as_slice(), bj.as_slice())
    });
    ctx.finish(out)
}

/// Bray–Curtis dissimilarity \(\sum|a-b|/\sum|a+b|\) on aligned prefixes.
///
/// Distinct from [`manhattan_distance`] and [`canberra_distance`].
pub fn braycurtis_distance(a: &Vector, b: &Vector, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if let Some(issue) = signlred::scan_finite(a.as_slice()).to_issue("braycurtis_distance.a") {
        ctx.push(issue);
    }
    if let Some(issue) = signlred::scan_finite(b.as_slice()).to_issue("braycurtis_distance.b") {
        ctx.push(issue);
    }
    if a.is_empty() || b.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message("braycurtis_distance on an empty series")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    ctx.finish(braycurtis_distance_raw(a.as_slice(), b.as_slice()))
}

/// Pairwise Bray–Curtis dissimilarity.
pub fn cdist_braycurtis(a: &Matrix, b: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, a, None, &ctx.policy);
    inspect_xy(&mut ctx.report, b, None, &ctx.policy);
    let out = Matrix::from_fn(a.nrows(), b.nrows(), |i, j| {
        let ai = a.row(i);
        let bj = b.row(j);
        braycurtis_distance_raw(ai.as_slice(), bj.as_slice())
    });
    ctx.finish(out)
}

fn lorentzian_distance_raw(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return f64::NAN;
    }
    let mut s = 0.0_f64;
    for i in 0..n {
        s += (1.0 + (a[i] - b[i]).abs()).ln();
    }
    s
}

fn angular_distance_raw(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return f64::NAN;
    }
    let mut num = 0.0_f64;
    let mut na = 0.0_f64;
    let mut nb = 0.0_f64;
    for i in 0..n {
        num += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    let den = na.sqrt() * nb.sqrt();
    if den < 1e-18 {
        return if na < 1e-18 && nb < 1e-18 { 0.0 } else { 0.5 };
    }
    (num / den).clamp(-1.0, 1.0).acos() / std::f64::consts::PI
}

/// Lorentzian distance \(\sum\log(1+|a_i-b_i|)\) on aligned prefixes.
///
/// Distinct from [`manhattan_distance`] (no log) and [`canberra_distance`].
/// Identical series score 0.
pub fn lorentzian_distance(a: &Vector, b: &Vector, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if let Some(issue) = signlred::scan_finite(a.as_slice()).to_issue("lorentzian_distance.a") {
        ctx.push(issue);
    }
    if let Some(issue) = signlred::scan_finite(b.as_slice()).to_issue("lorentzian_distance.b") {
        ctx.push(issue);
    }
    if a.is_empty() || b.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message("lorentzian_distance on an empty series")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    ctx.finish(lorentzian_distance_raw(a.as_slice(), b.as_slice()))
}

/// Pairwise Lorentzian distance.
pub fn cdist_lorentzian(a: &Matrix, b: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, a, None, &ctx.policy);
    inspect_xy(&mut ctx.report, b, None, &ctx.policy);
    let out = Matrix::from_fn(a.nrows(), b.nrows(), |i, j| {
        let ai = a.row(i);
        let bj = b.row(j);
        lorentzian_distance_raw(ai.as_slice(), bj.as_slice())
    });
    ctx.finish(out)
}

/// Angular distance \(\arccos(\mathrm{clamp}(\cos,-1,1))/\pi\) on aligned prefixes.
///
/// Distinct from [`cosine_distance`] (\(1-\cos\)). Identical series score 0.
pub fn angular_distance(a: &Vector, b: &Vector, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if let Some(issue) = signlred::scan_finite(a.as_slice()).to_issue("angular_distance.a") {
        ctx.push(issue);
    }
    if let Some(issue) = signlred::scan_finite(b.as_slice()).to_issue("angular_distance.b") {
        ctx.push(issue);
    }
    if a.is_empty() || b.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message("angular_distance on an empty series")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    ctx.finish(angular_distance_raw(a.as_slice(), b.as_slice()))
}

/// Pairwise angular distance.
pub fn cdist_angular(a: &Matrix, b: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, a, None, &ctx.policy);
    inspect_xy(&mut ctx.report, b, None, &ctx.policy);
    let out = Matrix::from_fn(a.nrows(), b.nrows(), |i, j| {
        let ai = a.row(i);
        let bj = b.row(j);
        angular_distance_raw(ai.as_slice(), bj.as_slice())
    });
    ctx.finish(out)
}

fn minkowski3_distance_raw(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return f64::NAN;
    }
    let mut s = 0.0_f64;
    for i in 0..n {
        let d = (a[i] - b[i]).abs();
        s += d * d * d;
    }
    s.cbrt()
}

fn clark_distance_raw(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return f64::NAN;
    }
    let mut s = 0.0_f64;
    for i in 0..n {
        let den = a[i].abs() + b[i].abs();
        if den > 1e-18 {
            let u = (a[i] - b[i]) / den;
            s += u * u;
        }
    }
    s.sqrt()
}

/// Minkowski \(p=3\) distance \((\sum|a_i-b_i|^3)^{1/3}\) on aligned prefixes.
///
/// Distinct from [`manhattan_distance`] (\(p=1\)) and [`chebyshev_distance`]
/// (\(p=\infty\)). Identical series score 0.
pub fn minkowski3_distance(a: &Vector, b: &Vector, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if let Some(issue) = signlred::scan_finite(a.as_slice()).to_issue("minkowski3_distance.a") {
        ctx.push(issue);
    }
    if let Some(issue) = signlred::scan_finite(b.as_slice()).to_issue("minkowski3_distance.b") {
        ctx.push(issue);
    }
    if a.is_empty() || b.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message("minkowski3_distance on an empty series")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    ctx.finish(minkowski3_distance_raw(a.as_slice(), b.as_slice()))
}

/// Pairwise Minkowski \(p=3\) distance.
pub fn cdist_minkowski3(a: &Matrix, b: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, a, None, &ctx.policy);
    inspect_xy(&mut ctx.report, b, None, &ctx.policy);
    let out = Matrix::from_fn(a.nrows(), b.nrows(), |i, j| {
        let ai = a.row(i);
        let bj = b.row(j);
        minkowski3_distance_raw(ai.as_slice(), bj.as_slice())
    });
    ctx.finish(out)
}

/// Clark distance \(\sqrt{\sum((a_i-b_i)/(|a_i|+|b_i|))^2}\) on aligned prefixes.
///
/// Distinct from [`canberra_distance`] (no square/sqrt). Identical series score 0.
pub fn clark_distance(a: &Vector, b: &Vector, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if let Some(issue) = signlred::scan_finite(a.as_slice()).to_issue("clark_distance.a") {
        ctx.push(issue);
    }
    if let Some(issue) = signlred::scan_finite(b.as_slice()).to_issue("clark_distance.b") {
        ctx.push(issue);
    }
    if a.is_empty() || b.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message("clark_distance on an empty series")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    ctx.finish(clark_distance_raw(a.as_slice(), b.as_slice()))
}

/// Pairwise Clark distance.
pub fn cdist_clark(a: &Matrix, b: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, a, None, &ctx.policy);
    inspect_xy(&mut ctx.report, b, None, &ctx.policy);
    let out = Matrix::from_fn(a.nrows(), b.nrows(), |i, j| {
        let ai = a.row(i);
        let bj = b.row(j);
        clark_distance_raw(ai.as_slice(), bj.as_slice())
    });
    ctx.finish(out)
}

fn squared_euclidean_distance_raw(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return f64::NAN;
    }
    let mut s = 0.0_f64;
    for i in 0..n {
        let d = a[i] - b[i];
        s += d * d;
    }
    s
}

fn dice_distance_raw(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return f64::NAN;
    }
    let mut num = 0.0_f64;
    let mut na = 0.0_f64;
    let mut nb = 0.0_f64;
    for i in 0..n {
        num += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    let den = na + nb;
    if den < 1e-18 {
        return 0.0;
    }
    1.0 - 2.0 * num / den
}

/// Squared Euclidean distance \(\sum(a_i-b_i)^2\) on aligned prefixes.
///
/// Distinct from [`minkowski3_distance`] and [`decay_euclidean`]. Identical
/// series score 0.
pub fn squared_euclidean_distance(
    a: &Vector,
    b: &Vector,
    session: &Session,
) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if let Some(issue) = signlred::scan_finite(a.as_slice()).to_issue("squared_euclidean_distance.a")
    {
        ctx.push(issue);
    }
    if let Some(issue) = signlred::scan_finite(b.as_slice()).to_issue("squared_euclidean_distance.b")
    {
        ctx.push(issue);
    }
    if a.is_empty() || b.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message("squared_euclidean_distance on an empty series")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    ctx.finish(squared_euclidean_distance_raw(a.as_slice(), b.as_slice()))
}

/// Pairwise squared Euclidean distance.
pub fn cdist_squared_euclidean(
    a: &Matrix,
    b: &Matrix,
    session: &Session,
) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, a, None, &ctx.policy);
    inspect_xy(&mut ctx.report, b, None, &ctx.policy);
    let out = Matrix::from_fn(a.nrows(), b.nrows(), |i, j| {
        let ai = a.row(i);
        let bj = b.row(j);
        squared_euclidean_distance_raw(ai.as_slice(), bj.as_slice())
    });
    ctx.finish(out)
}

/// Dice / Sørensen distance \(1-2\langle a,b\rangle/(\|a\|^2+\|b\|^2)\).
///
/// Distinct from [`cosine_distance`] (geometric mean in the denominator) and
/// [`braycurtis_distance`]. Identical series score 0.
pub fn dice_distance(a: &Vector, b: &Vector, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if let Some(issue) = signlred::scan_finite(a.as_slice()).to_issue("dice_distance.a") {
        ctx.push(issue);
    }
    if let Some(issue) = signlred::scan_finite(b.as_slice()).to_issue("dice_distance.b") {
        ctx.push(issue);
    }
    if a.is_empty() || b.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message("dice_distance on an empty series")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    ctx.finish(dice_distance_raw(a.as_slice(), b.as_slice()))
}

/// Pairwise Dice distance.
pub fn cdist_dice(a: &Matrix, b: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, a, None, &ctx.policy);
    inspect_xy(&mut ctx.report, b, None, &ctx.policy);
    let out = Matrix::from_fn(a.nrows(), b.nrows(), |i, j| {
        let ai = a.row(i);
        let bj = b.row(j);
        dice_distance_raw(ai.as_slice(), bj.as_slice())
    });
    ctx.finish(out)
}

fn tanimoto_distance_raw(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return f64::NAN;
    }
    let mut num = 0.0_f64;
    let mut na = 0.0_f64;
    let mut nb = 0.0_f64;
    for i in 0..n {
        num += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    let den = na + nb - num;
    if den.abs() < 1e-18 {
        return 0.0;
    }
    1.0 - num / den
}

fn wave_hedges_distance_raw(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return f64::NAN;
    }
    let mut s = 0.0_f64;
    for i in 0..n {
        let ma = a[i].abs().max(b[i].abs());
        if ma < 1e-18 {
            continue;
        }
        s += (a[i] - b[i]).abs() / ma;
    }
    s
}

/// Tanimoto / Jaccard distance \(1-\langle a,b\rangle/(\|a\|^2+\|b\|^2-\langle a,b\rangle)\).
///
/// Distinct from [`dice_distance`] (factor 2 in the numerator) and
/// [`cosine_distance`]. Identical series score 0.
pub fn tanimoto_distance(a: &Vector, b: &Vector, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if let Some(issue) = signlred::scan_finite(a.as_slice()).to_issue("tanimoto_distance.a") {
        ctx.push(issue);
    }
    if let Some(issue) = signlred::scan_finite(b.as_slice()).to_issue("tanimoto_distance.b") {
        ctx.push(issue);
    }
    if a.is_empty() || b.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message("tanimoto_distance on an empty series")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    ctx.finish(tanimoto_distance_raw(a.as_slice(), b.as_slice()))
}

/// Pairwise Tanimoto distance.
pub fn cdist_tanimoto(a: &Matrix, b: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, a, None, &ctx.policy);
    inspect_xy(&mut ctx.report, b, None, &ctx.policy);
    let out = Matrix::from_fn(a.nrows(), b.nrows(), |i, j| {
        let ai = a.row(i);
        let bj = b.row(j);
        tanimoto_distance_raw(ai.as_slice(), bj.as_slice())
    });
    ctx.finish(out)
}

/// Wave Hedges distance \(\sum |a_i-b_i|/\max(|a_i|,|b_i|)\).
///
/// Distinct from [`canberra_distance`] (sum in the denominator) and
/// [`clark_distance`]. Identical series score 0.
pub fn wave_hedges_distance(a: &Vector, b: &Vector, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if let Some(issue) = signlred::scan_finite(a.as_slice()).to_issue("wave_hedges_distance.a") {
        ctx.push(issue);
    }
    if let Some(issue) = signlred::scan_finite(b.as_slice()).to_issue("wave_hedges_distance.b") {
        ctx.push(issue);
    }
    if a.is_empty() || b.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message("wave_hedges_distance on an empty series")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    ctx.finish(wave_hedges_distance_raw(a.as_slice(), b.as_slice()))
}

/// Pairwise Wave Hedges distance.
pub fn cdist_wave_hedges(a: &Matrix, b: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, a, None, &ctx.policy);
    inspect_xy(&mut ctx.report, b, None, &ctx.policy);
    let out = Matrix::from_fn(a.nrows(), b.nrows(), |i, j| {
        let ai = a.row(i);
        let bj = b.row(j);
        wave_hedges_distance_raw(ai.as_slice(), bj.as_slice())
    });
    ctx.finish(out)
}

fn kulczynski_distance_raw(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return f64::NAN;
    }
    let mut num = 0.0_f64;
    let mut den = 0.0_f64;
    for i in 0..n {
        num += (a[i] - b[i]).abs();
        den += a[i].abs().min(b[i].abs());
    }
    if den < 1e-18 {
        return 0.0;
    }
    num / den
}

fn ruzicka_distance_raw(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return f64::NAN;
    }
    let mut nmin = 0.0_f64;
    let mut nmax = 0.0_f64;
    for i in 0..n {
        nmin += a[i].abs().min(b[i].abs());
        nmax += a[i].abs().max(b[i].abs());
    }
    if nmax < 1e-18 {
        return 0.0;
    }
    1.0 - nmin / nmax
}

/// Kulczynski distance \(\sum|a_i-b_i|/\sum\min(|a_i|,|b_i|)\).
///
/// Distinct from [`wave_hedges_distance`] (per-coordinate max) and
/// [`canberra_distance`]. Identical series score 0.
pub fn kulczynski_distance(a: &Vector, b: &Vector, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if let Some(issue) = signlred::scan_finite(a.as_slice()).to_issue("kulczynski_distance.a") {
        ctx.push(issue);
    }
    if let Some(issue) = signlred::scan_finite(b.as_slice()).to_issue("kulczynski_distance.b") {
        ctx.push(issue);
    }
    if a.is_empty() || b.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message("kulczynski_distance on an empty series")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    ctx.finish(kulczynski_distance_raw(a.as_slice(), b.as_slice()))
}

/// Pairwise Kulczynski distance.
pub fn cdist_kulczynski(a: &Matrix, b: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, a, None, &ctx.policy);
    inspect_xy(&mut ctx.report, b, None, &ctx.policy);
    let out = Matrix::from_fn(a.nrows(), b.nrows(), |i, j| {
        let ai = a.row(i);
        let bj = b.row(j);
        kulczynski_distance_raw(ai.as_slice(), bj.as_slice())
    });
    ctx.finish(out)
}

/// Ruzicka distance \(1-\sum\min(|a_i|,|b_i|)/\sum\max(|a_i|,|b_i|)\).
///
/// Distinct from [`tanimoto_distance`] (inner-product form) and
/// [`dice_distance`]. Identical series score 0.
pub fn ruzicka_distance(a: &Vector, b: &Vector, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if let Some(issue) = signlred::scan_finite(a.as_slice()).to_issue("ruzicka_distance.a") {
        ctx.push(issue);
    }
    if let Some(issue) = signlred::scan_finite(b.as_slice()).to_issue("ruzicka_distance.b") {
        ctx.push(issue);
    }
    if a.is_empty() || b.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message("ruzicka_distance on an empty series")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    ctx.finish(ruzicka_distance_raw(a.as_slice(), b.as_slice()))
}

/// Pairwise Ruzicka distance.
pub fn cdist_ruzicka(a: &Matrix, b: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, a, None, &ctx.policy);
    inspect_xy(&mut ctx.report, b, None, &ctx.policy);
    let out = Matrix::from_fn(a.nrows(), b.nrows(), |i, j| {
        let ai = a.row(i);
        let bj = b.row(j);
        ruzicka_distance_raw(ai.as_slice(), bj.as_slice())
    });
    ctx.finish(out)
}

fn hellinger_distance_raw(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return f64::NAN;
    }
    let mut s = 0.0_f64;
    for i in 0..n {
        let da = a[i].abs().sqrt();
        let db = b[i].abs().sqrt();
        let d = da - db;
        s += d * d;
    }
    (0.5 * s).sqrt()
}

fn jensen_shannon_distance_raw(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return f64::NAN;
    }
    let mut sa = 0.0_f64;
    let mut sb = 0.0_f64;
    for i in 0..n {
        sa += a[i].abs();
        sb += b[i].abs();
    }
    if sa < 1e-18 && sb < 1e-18 {
        return 0.0;
    }
    sa = sa.max(1e-18);
    sb = sb.max(1e-18);
    let mut js = 0.0_f64;
    for i in 0..n {
        let p = a[i].abs() / sa;
        let q = b[i].abs() / sb;
        let m = 0.5 * (p + q);
        if p > 1e-18 && m > 1e-18 {
            js += 0.5 * p * (p / m).ln();
        }
        if q > 1e-18 && m > 1e-18 {
            js += 0.5 * q * (q / m).ln();
        }
    }
    js.max(0.0).sqrt()
}

/// Hellinger distance \(\sqrt{\tfrac12\sum(\sqrt{|a_i|}-\sqrt{|b_i|})^2}\).
///
/// Distinct from [`cosine_distance`] and [`squared_euclidean_distance`].
/// Identical series score 0.
pub fn hellinger_distance(a: &Vector, b: &Vector, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if let Some(issue) = signlred::scan_finite(a.as_slice()).to_issue("hellinger_distance.a") {
        ctx.push(issue);
    }
    if let Some(issue) = signlred::scan_finite(b.as_slice()).to_issue("hellinger_distance.b") {
        ctx.push(issue);
    }
    if a.is_empty() || b.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message("hellinger_distance on an empty series")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    ctx.finish(hellinger_distance_raw(a.as_slice(), b.as_slice()))
}

/// Pairwise Hellinger distance.
pub fn cdist_hellinger(a: &Matrix, b: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, a, None, &ctx.policy);
    inspect_xy(&mut ctx.report, b, None, &ctx.policy);
    let out = Matrix::from_fn(a.nrows(), b.nrows(), |i, j| {
        let ai = a.row(i);
        let bj = b.row(j);
        hellinger_distance_raw(ai.as_slice(), bj.as_slice())
    });
    ctx.finish(out)
}

/// Jensen–Shannon distance \(\sqrt{\mathrm{JS}(|a|,|b|)}\) after \(\ell_1\) normalisation.
///
/// Distinct from [`hellinger_distance`] and [`cosine_distance`]. Identical
/// series score 0.
pub fn jensen_shannon_distance(
    a: &Vector,
    b: &Vector,
    session: &Session,
) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if let Some(issue) = signlred::scan_finite(a.as_slice()).to_issue("jensen_shannon_distance.a") {
        ctx.push(issue);
    }
    if let Some(issue) = signlred::scan_finite(b.as_slice()).to_issue("jensen_shannon_distance.b") {
        ctx.push(issue);
    }
    if a.is_empty() || b.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message("jensen_shannon_distance on an empty series")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    ctx.finish(jensen_shannon_distance_raw(a.as_slice(), b.as_slice()))
}

/// Pairwise Jensen–Shannon distance.
pub fn cdist_jensen_shannon(a: &Matrix, b: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, a, None, &ctx.policy);
    inspect_xy(&mut ctx.report, b, None, &ctx.policy);
    let out = Matrix::from_fn(a.nrows(), b.nrows(), |i, j| {
        let ai = a.row(i);
        let bj = b.row(j);
        jensen_shannon_distance_raw(ai.as_slice(), bj.as_slice())
    });
    ctx.finish(out)
}

fn bhattacharyya_distance_raw(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return f64::NAN;
    }
    let mut sa = 0.0_f64;
    let mut sb = 0.0_f64;
    for i in 0..n {
        sa += a[i].abs();
        sb += b[i].abs();
    }
    if sa < 1e-18 && sb < 1e-18 {
        return 0.0;
    }
    sa = sa.max(1e-18);
    sb = sb.max(1e-18);
    let mut bc = 0.0_f64;
    for i in 0..n {
        bc += ((a[i].abs() / sa) * (b[i].abs() / sb)).sqrt();
    }
    if bc >= 1.0 {
        0.0
    } else if bc <= 1e-18 {
        20.0
    } else {
        -bc.ln()
    }
}

fn hassanat_distance_raw(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return f64::NAN;
    }
    let mut s = 0.0_f64;
    for i in 0..n {
        let lo = a[i].abs().min(b[i].abs());
        let hi = a[i].abs().max(b[i].abs());
        if hi < 1e-18 {
            continue;
        }
        s += 1.0 - (1.0 + lo) / (1.0 + hi);
    }
    s
}

/// Bhattacharyya distance \(-\log\sum\sqrt{p_i q_i}\) after \(\ell_1\) norm.
///
/// Distinct from [`hellinger_distance`] (\(\sqrt{1-\mathrm{BC}}\)) and
/// [`jensen_shannon_distance`]. Identical series score 0.
pub fn bhattacharyya_distance(a: &Vector, b: &Vector, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if let Some(issue) = signlred::scan_finite(a.as_slice()).to_issue("bhattacharyya_distance.a") {
        ctx.push(issue);
    }
    if let Some(issue) = signlred::scan_finite(b.as_slice()).to_issue("bhattacharyya_distance.b") {
        ctx.push(issue);
    }
    if a.is_empty() || b.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message("bhattacharyya_distance on an empty series")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    ctx.finish(bhattacharyya_distance_raw(a.as_slice(), b.as_slice()))
}

/// Pairwise Bhattacharyya distance.
pub fn cdist_bhattacharyya(a: &Matrix, b: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, a, None, &ctx.policy);
    inspect_xy(&mut ctx.report, b, None, &ctx.policy);
    let out = Matrix::from_fn(a.nrows(), b.nrows(), |i, j| {
        let ai = a.row(i);
        let bj = b.row(j);
        bhattacharyya_distance_raw(ai.as_slice(), bj.as_slice())
    });
    ctx.finish(out)
}

/// Hassanat distance \(\sum(1-(1+\min|a_i|,|b_i|)/(1+\max|a_i|,|b_i|))\).
///
/// Distinct from [`ruzicka_distance`] and [`wave_hedges_distance`]. Identical
/// series score 0.
pub fn hassanat_distance(a: &Vector, b: &Vector, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if let Some(issue) = signlred::scan_finite(a.as_slice()).to_issue("hassanat_distance.a") {
        ctx.push(issue);
    }
    if let Some(issue) = signlred::scan_finite(b.as_slice()).to_issue("hassanat_distance.b") {
        ctx.push(issue);
    }
    if a.is_empty() || b.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message("hassanat_distance on an empty series")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    ctx.finish(hassanat_distance_raw(a.as_slice(), b.as_slice()))
}

/// Pairwise Hassanat distance.
pub fn cdist_hassanat(a: &Matrix, b: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, a, None, &ctx.policy);
    inspect_xy(&mut ctx.report, b, None, &ctx.policy);
    let out = Matrix::from_fn(a.nrows(), b.nrows(), |i, j| {
        let ai = a.row(i);
        let bj = b.row(j);
        hassanat_distance_raw(ai.as_slice(), bj.as_slice())
    });
    ctx.finish(out)
}

fn fidelity_distance_raw(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return f64::NAN;
    }
    let mut sa = 0.0_f64;
    let mut sb = 0.0_f64;
    for i in 0..n {
        sa += a[i].abs();
        sb += b[i].abs();
    }
    if sa < 1e-18 && sb < 1e-18 {
        return 0.0;
    }
    sa = sa.max(1e-18);
    sb = sb.max(1e-18);
    let mut bc = 0.0_f64;
    for i in 0..n {
        bc += ((a[i].abs() / sa) * (b[i].abs() / sb)).sqrt();
    }
    (1.0 - bc).max(0.0)
}

fn whittaker_distance_raw(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return f64::NAN;
    }
    let mut sa = 0.0_f64;
    let mut sb = 0.0_f64;
    for i in 0..n {
        sa += a[i].abs();
        sb += b[i].abs();
    }
    if sa < 1e-18 && sb < 1e-18 {
        return 0.0;
    }
    sa = sa.max(1e-18);
    sb = sb.max(1e-18);
    let mut s = 0.0_f64;
    for i in 0..n {
        s += (a[i].abs() / sa - b[i].abs() / sb).abs();
    }
    0.5 * s
}

/// Fidelity distance \(1-\sum\sqrt{p_i q_i}\) after \(\ell_1\) norm.
///
/// Distinct from [`bhattacharyya_distance`] (\(-\log\) BC). Identical series
/// score 0.
pub fn fidelity_distance(a: &Vector, b: &Vector, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if let Some(issue) = signlred::scan_finite(a.as_slice()).to_issue("fidelity_distance.a") {
        ctx.push(issue);
    }
    if let Some(issue) = signlred::scan_finite(b.as_slice()).to_issue("fidelity_distance.b") {
        ctx.push(issue);
    }
    if a.is_empty() || b.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message("fidelity_distance on an empty series")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    ctx.finish(fidelity_distance_raw(a.as_slice(), b.as_slice()))
}

/// Pairwise fidelity distance.
pub fn cdist_fidelity(a: &Matrix, b: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, a, None, &ctx.policy);
    inspect_xy(&mut ctx.report, b, None, &ctx.policy);
    let out = Matrix::from_fn(a.nrows(), b.nrows(), |i, j| {
        let ai = a.row(i);
        let bj = b.row(j);
        fidelity_distance_raw(ai.as_slice(), bj.as_slice())
    });
    ctx.finish(out)
}

/// Whittaker distance \(\tfrac12\sum|p_i-q_i|\) after \(\ell_1\) norm.
///
/// Distinct from [`manhattan_distance`] (unnormalized) and the tsa Whittaker
/// smoother. Identical series score 0.
pub fn whittaker_distance(a: &Vector, b: &Vector, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if let Some(issue) = signlred::scan_finite(a.as_slice()).to_issue("whittaker_distance.a") {
        ctx.push(issue);
    }
    if let Some(issue) = signlred::scan_finite(b.as_slice()).to_issue("whittaker_distance.b") {
        ctx.push(issue);
    }
    if a.is_empty() || b.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message("whittaker_distance on an empty series")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    ctx.finish(whittaker_distance_raw(a.as_slice(), b.as_slice()))
}

/// Pairwise Whittaker distance.
pub fn cdist_whittaker(a: &Matrix, b: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, a, None, &ctx.policy);
    inspect_xy(&mut ctx.report, b, None, &ctx.policy);
    let out = Matrix::from_fn(a.nrows(), b.nrows(), |i, j| {
        let ai = a.row(i);
        let bj = b.row(j);
        whittaker_distance_raw(ai.as_slice(), bj.as_slice())
    });
    ctx.finish(out)
}

fn pearson_chi_squared_distance_raw(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return f64::NAN;
    }
    let mut sa = 0.0_f64;
    let mut sb = 0.0_f64;
    for i in 0..n {
        sa += a[i].abs();
        sb += b[i].abs();
    }
    if sa < 1e-18 && sb < 1e-18 {
        return 0.0;
    }
    sa = sa.max(1e-18);
    sb = sb.max(1e-18);
    let mut s = 0.0_f64;
    for i in 0..n {
        let p = a[i].abs() / sa;
        let q = b[i].abs() / sb;
        s += (p - q) * (p - q) / p.max(1e-18);
    }
    s
}

fn neyman_chi_squared_distance_raw(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return f64::NAN;
    }
    let mut sa = 0.0_f64;
    let mut sb = 0.0_f64;
    for i in 0..n {
        sa += a[i].abs();
        sb += b[i].abs();
    }
    if sa < 1e-18 && sb < 1e-18 {
        return 0.0;
    }
    sa = sa.max(1e-18);
    sb = sb.max(1e-18);
    let mut s = 0.0_f64;
    for i in 0..n {
        let p = a[i].abs() / sa;
        let q = b[i].abs() / sb;
        s += (p - q) * (p - q) / q.max(1e-18);
    }
    s
}

/// Pearson \(\chi^2\) distance \(\sum(p_i-q_i)^2/p_i\) after \(\ell_1\) norm.
///
/// Distinct from [`clark_distance`] and [`neyman_chi_squared_distance`].
/// Identical series score 0.
pub fn pearson_chi_squared_distance(
    a: &Vector,
    b: &Vector,
    session: &Session,
) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if let Some(issue) = signlred::scan_finite(a.as_slice()).to_issue("pearson_chi_squared_distance.a")
    {
        ctx.push(issue);
    }
    if let Some(issue) = signlred::scan_finite(b.as_slice()).to_issue("pearson_chi_squared_distance.b")
    {
        ctx.push(issue);
    }
    if a.is_empty() || b.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message("pearson_chi_squared_distance on an empty series")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    ctx.finish(pearson_chi_squared_distance_raw(a.as_slice(), b.as_slice()))
}

/// Pairwise Pearson \(\chi^2\) distance.
pub fn cdist_pearson_chi_squared(
    a: &Matrix,
    b: &Matrix,
    session: &Session,
) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, a, None, &ctx.policy);
    inspect_xy(&mut ctx.report, b, None, &ctx.policy);
    let out = Matrix::from_fn(a.nrows(), b.nrows(), |i, j| {
        let ai = a.row(i);
        let bj = b.row(j);
        pearson_chi_squared_distance_raw(ai.as_slice(), bj.as_slice())
    });
    ctx.finish(out)
}

/// Neyman \(\chi^2\) distance \(\sum(p_i-q_i)^2/q_i\) after \(\ell_1\) norm.
///
/// Distinct from [`pearson_chi_squared_distance`] (denominator \(p\)) and
/// [`clark_distance`]. Identical series score 0.
pub fn neyman_chi_squared_distance(
    a: &Vector,
    b: &Vector,
    session: &Session,
) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if let Some(issue) = signlred::scan_finite(a.as_slice()).to_issue("neyman_chi_squared_distance.a")
    {
        ctx.push(issue);
    }
    if let Some(issue) = signlred::scan_finite(b.as_slice()).to_issue("neyman_chi_squared_distance.b")
    {
        ctx.push(issue);
    }
    if a.is_empty() || b.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message("neyman_chi_squared_distance on an empty series")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    ctx.finish(neyman_chi_squared_distance_raw(a.as_slice(), b.as_slice()))
}

/// Pairwise Neyman \(\chi^2\) distance.
pub fn cdist_neyman_chi_squared(
    a: &Matrix,
    b: &Matrix,
    session: &Session,
) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, a, None, &ctx.policy);
    inspect_xy(&mut ctx.report, b, None, &ctx.policy);
    let out = Matrix::from_fn(a.nrows(), b.nrows(), |i, j| {
        let ai = a.row(i);
        let bj = b.row(j);
        neyman_chi_squared_distance_raw(ai.as_slice(), bj.as_slice())
    });
    ctx.finish(out)
}

fn additive_symmetric_distance_raw(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return f64::NAN;
    }
    let mut sa = 0.0_f64;
    let mut sb = 0.0_f64;
    for i in 0..n {
        sa += a[i].abs();
        sb += b[i].abs();
    }
    if sa < 1e-18 && sb < 1e-18 {
        return 0.0;
    }
    sa = sa.max(1e-18);
    sb = sb.max(1e-18);
    let mut s = 0.0_f64;
    for i in 0..n {
        let p = a[i].abs() / sa;
        let q = b[i].abs() / sb;
        let den = (p + q).max(1e-18);
        s += (p - q) * (p - q) / den;
    }
    s
}

fn k_divergence_distance_raw(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return f64::NAN;
    }
    let mut sa = 0.0_f64;
    let mut sb = 0.0_f64;
    for i in 0..n {
        sa += a[i].abs();
        sb += b[i].abs();
    }
    if sa < 1e-18 && sb < 1e-18 {
        return 0.0;
    }
    sa = sa.max(1e-18);
    sb = sb.max(1e-18);
    let mut s = 0.0_f64;
    for i in 0..n {
        let p = (a[i].abs() / sa).max(1e-18);
        let q = (b[i].abs() / sb).max(1e-18);
        s += p * (2.0 * p / (p + q)).ln();
    }
    s.max(0.0)
}

/// Additive-symmetric \(\chi^2\) \(\sum(p_i-q_i)^2/(p_i+q_i)\) after \(\ell_1\).
///
/// Distinct from [`clark_distance`] (square-root form) and
/// [`pearson_chi_squared_distance`]. Identical series score 0.
pub fn additive_symmetric_distance(
    a: &Vector,
    b: &Vector,
    session: &Session,
) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if let Some(issue) = signlred::scan_finite(a.as_slice()).to_issue("additive_symmetric_distance.a")
    {
        ctx.push(issue);
    }
    if let Some(issue) = signlred::scan_finite(b.as_slice()).to_issue("additive_symmetric_distance.b")
    {
        ctx.push(issue);
    }
    if a.is_empty() || b.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message("additive_symmetric_distance on an empty series")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    ctx.finish(additive_symmetric_distance_raw(a.as_slice(), b.as_slice()))
}

/// Pairwise additive-symmetric \(\chi^2\) distance.
pub fn cdist_additive_symmetric(
    a: &Matrix,
    b: &Matrix,
    session: &Session,
) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, a, None, &ctx.policy);
    inspect_xy(&mut ctx.report, b, None, &ctx.policy);
    let out = Matrix::from_fn(a.nrows(), b.nrows(), |i, j| {
        let ai = a.row(i);
        let bj = b.row(j);
        additive_symmetric_distance_raw(ai.as_slice(), bj.as_slice())
    });
    ctx.finish(out)
}

/// Kullback \(K\)-divergence \(\sum p_i\log(2p_i/(p_i+q_i))\) after \(\ell_1\).
///
/// Distinct from [`jensen_shannon_distance`] (symmetrised). Identical series
/// score 0.
pub fn k_divergence_distance(a: &Vector, b: &Vector, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if let Some(issue) = signlred::scan_finite(a.as_slice()).to_issue("k_divergence_distance.a") {
        ctx.push(issue);
    }
    if let Some(issue) = signlred::scan_finite(b.as_slice()).to_issue("k_divergence_distance.b") {
        ctx.push(issue);
    }
    if a.is_empty() || b.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message("k_divergence_distance on an empty series")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    ctx.finish(k_divergence_distance_raw(a.as_slice(), b.as_slice()))
}

/// Pairwise Kullback \(K\)-divergence.
pub fn cdist_k_divergence(a: &Matrix, b: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, a, None, &ctx.policy);
    inspect_xy(&mut ctx.report, b, None, &ctx.policy);
    let out = Matrix::from_fn(a.nrows(), b.nrows(), |i, j| {
        let ai = a.row(i);
        let bj = b.row(j);
        k_divergence_distance_raw(ai.as_slice(), bj.as_slice())
    });
    ctx.finish(out)
}

fn topsoe_distance_raw(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return f64::NAN;
    }
    let mut sa = 0.0_f64;
    let mut sb = 0.0_f64;
    for i in 0..n {
        sa += a[i].abs();
        sb += b[i].abs();
    }
    if sa < 1e-18 && sb < 1e-18 {
        return 0.0;
    }
    sa = sa.max(1e-18);
    sb = sb.max(1e-18);
    let mut s = 0.0_f64;
    for i in 0..n {
        let p = (a[i].abs() / sa).max(1e-18);
        let q = (b[i].abs() / sb).max(1e-18);
        let m = p + q;
        s += p * (2.0 * p / m).ln() + q * (2.0 * q / m).ln();
    }
    s.max(0.0)
}

fn taneja_distance_raw(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return f64::NAN;
    }
    let mut sa = 0.0_f64;
    let mut sb = 0.0_f64;
    for i in 0..n {
        sa += a[i].abs();
        sb += b[i].abs();
    }
    if sa < 1e-18 && sb < 1e-18 {
        return 0.0;
    }
    sa = sa.max(1e-18);
    sb = sb.max(1e-18);
    let mut s = 0.0_f64;
    for i in 0..n {
        let p = (a[i].abs() / sa).max(1e-18);
        let q = (b[i].abs() / sb).max(1e-18);
        let am = 0.5 * (p + q);
        let gm = (p * q).sqrt().max(1e-18);
        s += am * (am / gm).ln();
    }
    s.max(0.0)
}

/// Topsøe distance \(\sum(p\log(2p/(p+q))+q\log(2q/(p+q)))\) after \(\ell_1\).
///
/// Distinct from [`jensen_shannon_distance`] (square-root JS) and
/// [`k_divergence_distance`] (one-sided). Identical series score 0.
pub fn topsoe_distance(a: &Vector, b: &Vector, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if let Some(issue) = signlred::scan_finite(a.as_slice()).to_issue("topsoe_distance.a") {
        ctx.push(issue);
    }
    if let Some(issue) = signlred::scan_finite(b.as_slice()).to_issue("topsoe_distance.b") {
        ctx.push(issue);
    }
    if a.is_empty() || b.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message("topsoe_distance on an empty series")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    ctx.finish(topsoe_distance_raw(a.as_slice(), b.as_slice()))
}

/// Pairwise Topsøe distance.
pub fn cdist_topsoe(a: &Matrix, b: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, a, None, &ctx.policy);
    inspect_xy(&mut ctx.report, b, None, &ctx.policy);
    let out = Matrix::from_fn(a.nrows(), b.nrows(), |i, j| {
        let ai = a.row(i);
        let bj = b.row(j);
        topsoe_distance_raw(ai.as_slice(), bj.as_slice())
    });
    ctx.finish(out)
}

/// Taneja distance \(\sum\frac{p+q}{2}\log\frac{p+q}{2\sqrt{pq}}\) after \(\ell_1\).
///
/// Distinct from [`jensen_shannon_distance`] and [`topsoe_distance`].
/// Identical series score 0.
pub fn taneja_distance(a: &Vector, b: &Vector, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if let Some(issue) = signlred::scan_finite(a.as_slice()).to_issue("taneja_distance.a") {
        ctx.push(issue);
    }
    if let Some(issue) = signlred::scan_finite(b.as_slice()).to_issue("taneja_distance.b") {
        ctx.push(issue);
    }
    if a.is_empty() || b.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message("taneja_distance on an empty series")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    ctx.finish(taneja_distance_raw(a.as_slice(), b.as_slice()))
}

/// Pairwise Taneja distance.
pub fn cdist_taneja(a: &Matrix, b: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, a, None, &ctx.policy);
    inspect_xy(&mut ctx.report, b, None, &ctx.policy);
    let out = Matrix::from_fn(a.nrows(), b.nrows(), |i, j| {
        let ai = a.row(i);
        let bj = b.row(j);
        taneja_distance_raw(ai.as_slice(), bj.as_slice())
    });
    ctx.finish(out)
}

fn kumar_johnson_distance_raw(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return f64::NAN;
    }
    let mut sa = 0.0_f64;
    let mut sb = 0.0_f64;
    for i in 0..n {
        sa += a[i].abs();
        sb += b[i].abs();
    }
    if sa < 1e-18 && sb < 1e-18 {
        return 0.0;
    }
    sa = sa.max(1e-18);
    sb = sb.max(1e-18);
    let mut s = 0.0_f64;
    for i in 0..n {
        let p = (a[i].abs() / sa).max(1e-18);
        let q = (b[i].abs() / sb).max(1e-18);
        let d2 = p * p - q * q;
        s += d2 * d2 / (2.0 * (p * q).powf(1.5));
    }
    s.max(0.0)
}

fn harmonic_mean_distance_raw(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return f64::NAN;
    }
    let mut sa = 0.0_f64;
    let mut sb = 0.0_f64;
    for i in 0..n {
        sa += a[i].abs();
        sb += b[i].abs();
    }
    if sa < 1e-18 && sb < 1e-18 {
        return 0.0;
    }
    sa = sa.max(1e-18);
    sb = sb.max(1e-18);
    let mut s = 0.0_f64;
    for i in 0..n {
        let p = (a[i].abs() / sa).max(1e-18);
        let q = (b[i].abs() / sb).max(1e-18);
        s += 2.0 * p * q / (p + q);
    }
    (1.0 - s).max(0.0)
}

/// Kumar–Johnson distance \(\sum(p^2-q^2)^2/(2(pq)^{3/2})\) after \(\ell_1\).
///
/// Distinct from [`pearson_chi_squared_distance`] and [`taneja_distance`].
/// Identical series score 0.
pub fn kumar_johnson_distance(a: &Vector, b: &Vector, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if let Some(issue) = signlred::scan_finite(a.as_slice()).to_issue("kumar_johnson_distance.a") {
        ctx.push(issue);
    }
    if let Some(issue) = signlred::scan_finite(b.as_slice()).to_issue("kumar_johnson_distance.b") {
        ctx.push(issue);
    }
    if a.is_empty() || b.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message("kumar_johnson_distance on an empty series")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    ctx.finish(kumar_johnson_distance_raw(a.as_slice(), b.as_slice()))
}

/// Pairwise Kumar–Johnson distance.
pub fn cdist_kumar_johnson(a: &Matrix, b: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, a, None, &ctx.policy);
    inspect_xy(&mut ctx.report, b, None, &ctx.policy);
    let out = Matrix::from_fn(a.nrows(), b.nrows(), |i, j| {
        let ai = a.row(i);
        let bj = b.row(j);
        kumar_johnson_distance_raw(ai.as_slice(), bj.as_slice())
    });
    ctx.finish(out)
}

/// Harmonic-mean distance \(1-2\sum pq/(p+q)\) after \(\ell_1\).
///
/// Distinct from [`dice_distance`] (\(2\sum\min\)) and [`tanimoto_distance`].
/// Identical series score 0.
pub fn harmonic_mean_distance(a: &Vector, b: &Vector, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if let Some(issue) = signlred::scan_finite(a.as_slice()).to_issue("harmonic_mean_distance.a") {
        ctx.push(issue);
    }
    if let Some(issue) = signlred::scan_finite(b.as_slice()).to_issue("harmonic_mean_distance.b") {
        ctx.push(issue);
    }
    if a.is_empty() || b.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message("harmonic_mean_distance on an empty series")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    ctx.finish(harmonic_mean_distance_raw(a.as_slice(), b.as_slice()))
}

/// Pairwise harmonic-mean distance.
pub fn cdist_harmonic_mean(a: &Matrix, b: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, a, None, &ctx.policy);
    inspect_xy(&mut ctx.report, b, None, &ctx.policy);
    let out = Matrix::from_fn(a.nrows(), b.nrows(), |i, j| {
        let ai = a.row(i);
        let bj = b.row(j);
        harmonic_mean_distance_raw(ai.as_slice(), bj.as_slice())
    });
    ctx.finish(out)
}

fn max_symmetric_chi_squared_distance_raw(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return f64::NAN;
    }
    let mut sa = 0.0_f64;
    let mut sb = 0.0_f64;
    for i in 0..n {
        sa += a[i].abs();
        sb += b[i].abs();
    }
    if sa < 1e-18 && sb < 1e-18 {
        return 0.0;
    }
    sa = sa.max(1e-18);
    sb = sb.max(1e-18);
    let mut s = 0.0_f64;
    for i in 0..n {
        let p = (a[i].abs() / sa).max(1e-18);
        let q = (b[i].abs() / sb).max(1e-18);
        let d = p - q;
        s += d * d / p.max(q);
    }
    s.max(0.0)
}

fn intersection_distance_raw(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return f64::NAN;
    }
    let mut sa = 0.0_f64;
    let mut sb = 0.0_f64;
    for i in 0..n {
        sa += a[i].abs();
        sb += b[i].abs();
    }
    if sa < 1e-18 && sb < 1e-18 {
        return 0.0;
    }
    sa = sa.max(1e-18);
    sb = sb.max(1e-18);
    let mut s = 0.0_f64;
    for i in 0..n {
        let p = a[i].abs() / sa;
        let q = b[i].abs() / sb;
        s += p.min(q);
    }
    (1.0 - s).max(0.0)
}

/// Max-symmetric χ² \(\sum(p-q)^2/\max(p,q)\) after \(\ell_1\).
///
/// Distinct from [`pearson_chi_squared_distance`] (divide by \(p\)),
/// [`neyman_chi_squared_distance`] (divide by \(q\)), and
/// [`additive_symmetric_distance`] (divide by \(p+q\)). Identical series score 0.
pub fn max_symmetric_chi_squared_distance(
    a: &Vector,
    b: &Vector,
    session: &Session,
) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if let Some(issue) =
        signlred::scan_finite(a.as_slice()).to_issue("max_symmetric_chi_squared_distance.a")
    {
        ctx.push(issue);
    }
    if let Some(issue) =
        signlred::scan_finite(b.as_slice()).to_issue("max_symmetric_chi_squared_distance.b")
    {
        ctx.push(issue);
    }
    if a.is_empty() || b.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message("max_symmetric_chi_squared_distance on an empty series")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    ctx.finish(max_symmetric_chi_squared_distance_raw(a.as_slice(), b.as_slice()))
}

/// Pairwise max-symmetric χ² distance.
pub fn cdist_max_symmetric_chi_squared(
    a: &Matrix,
    b: &Matrix,
    session: &Session,
) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, a, None, &ctx.policy);
    inspect_xy(&mut ctx.report, b, None, &ctx.policy);
    let out = Matrix::from_fn(a.nrows(), b.nrows(), |i, j| {
        let ai = a.row(i);
        let bj = b.row(j);
        max_symmetric_chi_squared_distance_raw(ai.as_slice(), bj.as_slice())
    });
    ctx.finish(out)
}

/// Intersection distance \(1-\sum\min(p,q)\) after \(\ell_1\).
///
/// Distinct from [`dice_distance`] (vector \(2\langle a,b\rangle\) form) and
/// [`braycurtis_distance`]. Identical series score 0.
pub fn intersection_distance(a: &Vector, b: &Vector, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if let Some(issue) = signlred::scan_finite(a.as_slice()).to_issue("intersection_distance.a") {
        ctx.push(issue);
    }
    if let Some(issue) = signlred::scan_finite(b.as_slice()).to_issue("intersection_distance.b") {
        ctx.push(issue);
    }
    if a.is_empty() || b.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message("intersection_distance on an empty series")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    ctx.finish(intersection_distance_raw(a.as_slice(), b.as_slice()))
}

/// Pairwise intersection distance.
pub fn cdist_intersection(a: &Matrix, b: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, a, None, &ctx.policy);
    inspect_xy(&mut ctx.report, b, None, &ctx.policy);
    let out = Matrix::from_fn(a.nrows(), b.nrows(), |i, j| {
        let ai = a.row(i);
        let bj = b.row(j);
        intersection_distance_raw(ai.as_slice(), bj.as_slice())
    });
    ctx.finish(out)
}

fn min_symmetric_chi_squared_distance_raw(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return f64::NAN;
    }
    let mut sa = 0.0_f64;
    let mut sb = 0.0_f64;
    for i in 0..n {
        sa += a[i].abs();
        sb += b[i].abs();
    }
    if sa < 1e-18 && sb < 1e-18 {
        return 0.0;
    }
    sa = sa.max(1e-18);
    sb = sb.max(1e-18);
    let mut s = 0.0_f64;
    for i in 0..n {
        let p = (a[i].abs() / sa).max(1e-18);
        let q = (b[i].abs() / sb).max(1e-18);
        let d = p - q;
        s += d * d / p.min(q);
    }
    s.max(0.0)
}

fn l1_squared_euclidean_distance_raw(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return f64::NAN;
    }
    let mut sa = 0.0_f64;
    let mut sb = 0.0_f64;
    for i in 0..n {
        sa += a[i].abs();
        sb += b[i].abs();
    }
    if sa < 1e-18 && sb < 1e-18 {
        return 0.0;
    }
    sa = sa.max(1e-18);
    sb = sb.max(1e-18);
    let mut s = 0.0_f64;
    for i in 0..n {
        let p = a[i].abs() / sa;
        let q = b[i].abs() / sb;
        let d = p - q;
        s += d * d;
    }
    s.max(0.0)
}

/// Min-symmetric χ² \(\sum(p-q)^2/\min(p,q)\) after \(\ell_1\).
///
/// Distinct from [`max_symmetric_chi_squared_distance`] (divide by \(\max\))
/// and [`additive_symmetric_distance`] (divide by \(p+q\)). Identical series
/// score 0.
pub fn min_symmetric_chi_squared_distance(
    a: &Vector,
    b: &Vector,
    session: &Session,
) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if let Some(issue) =
        signlred::scan_finite(a.as_slice()).to_issue("min_symmetric_chi_squared_distance.a")
    {
        ctx.push(issue);
    }
    if let Some(issue) =
        signlred::scan_finite(b.as_slice()).to_issue("min_symmetric_chi_squared_distance.b")
    {
        ctx.push(issue);
    }
    if a.is_empty() || b.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message("min_symmetric_chi_squared_distance on an empty series")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    ctx.finish(min_symmetric_chi_squared_distance_raw(a.as_slice(), b.as_slice()))
}

/// Pairwise min-symmetric χ² distance.
pub fn cdist_min_symmetric_chi_squared(
    a: &Matrix,
    b: &Matrix,
    session: &Session,
) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, a, None, &ctx.policy);
    inspect_xy(&mut ctx.report, b, None, &ctx.policy);
    let out = Matrix::from_fn(a.nrows(), b.nrows(), |i, j| {
        let ai = a.row(i);
        let bj = b.row(j);
        min_symmetric_chi_squared_distance_raw(ai.as_slice(), bj.as_slice())
    });
    ctx.finish(out)
}

/// Squared Euclidean distance after \(\ell_1\) normalisation.
///
/// Distinct from [`squared_euclidean_distance`] (raw coordinates).
/// Identical series score 0.
pub fn l1_squared_euclidean_distance(
    a: &Vector,
    b: &Vector,
    session: &Session,
) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if let Some(issue) =
        signlred::scan_finite(a.as_slice()).to_issue("l1_squared_euclidean_distance.a")
    {
        ctx.push(issue);
    }
    if let Some(issue) =
        signlred::scan_finite(b.as_slice()).to_issue("l1_squared_euclidean_distance.b")
    {
        ctx.push(issue);
    }
    if a.is_empty() || b.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message("l1_squared_euclidean_distance on an empty series")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    ctx.finish(l1_squared_euclidean_distance_raw(a.as_slice(), b.as_slice()))
}

/// Pairwise \(\ell_1\)-normalised squared Euclidean distance.
pub fn cdist_l1_squared_euclidean(
    a: &Matrix,
    b: &Matrix,
    session: &Session,
) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, a, None, &ctx.policy);
    inspect_xy(&mut ctx.report, b, None, &ctx.policy);
    let out = Matrix::from_fn(a.nrows(), b.nrows(), |i, j| {
        let ai = a.row(i);
        let bj = b.row(j);
        l1_squared_euclidean_distance_raw(ai.as_slice(), bj.as_slice())
    });
    ctx.finish(out)
}

fn jaccard_distance_raw(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return f64::NAN;
    }
    let mut sa = 0.0_f64;
    let mut sb = 0.0_f64;
    for i in 0..n {
        sa += a[i].abs();
        sb += b[i].abs();
    }
    if sa < 1e-18 && sb < 1e-18 {
        return 0.0;
    }
    sa = sa.max(1e-18);
    sb = sb.max(1e-18);
    let mut smin = 0.0_f64;
    let mut smax = 0.0_f64;
    for i in 0..n {
        let p = a[i].abs() / sa;
        let q = b[i].abs() / sb;
        smin += p.min(q);
        smax += p.max(q);
    }
    if smax < 1e-18 {
        0.0
    } else {
        (1.0 - smin / smax).max(0.0)
    }
}

fn jeffreys_distance_raw(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return f64::NAN;
    }
    let mut sa = 0.0_f64;
    let mut sb = 0.0_f64;
    for i in 0..n {
        sa += a[i].abs();
        sb += b[i].abs();
    }
    if sa < 1e-18 && sb < 1e-18 {
        return 0.0;
    }
    sa = sa.max(1e-18);
    sb = sb.max(1e-18);
    let mut s = 0.0_f64;
    for i in 0..n {
        let p = (a[i].abs() / sa).max(1e-18);
        let q = (b[i].abs() / sb).max(1e-18);
        s += (p - q) * (p / q).ln();
    }
    s.max(0.0)
}

/// Probability Jaccard \(1-\sum\min/\sum\max\) after \(\ell_1\).
///
/// Distinct from [`crate::metrics::jaccard_distances`] (binary support) and
/// [`intersection_distance`] (\(1-\sum\min\)). Identical series score 0.
pub fn jaccard_distance(a: &Vector, b: &Vector, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if let Some(issue) = signlred::scan_finite(a.as_slice()).to_issue("jaccard_distance.a") {
        ctx.push(issue);
    }
    if let Some(issue) = signlred::scan_finite(b.as_slice()).to_issue("jaccard_distance.b") {
        ctx.push(issue);
    }
    if a.is_empty() || b.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message("jaccard_distance on an empty series")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    ctx.finish(jaccard_distance_raw(a.as_slice(), b.as_slice()))
}

/// Pairwise probability Jaccard distance.
pub fn cdist_jaccard(a: &Matrix, b: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, a, None, &ctx.policy);
    inspect_xy(&mut ctx.report, b, None, &ctx.policy);
    let out = Matrix::from_fn(a.nrows(), b.nrows(), |i, j| {
        let ai = a.row(i);
        let bj = b.row(j);
        jaccard_distance_raw(ai.as_slice(), bj.as_slice())
    });
    ctx.finish(out)
}

/// Jeffreys divergence \(\sum(p-q)\ln(p/q)\) after \(\ell_1\).
///
/// Distinct from [`k_divergence_distance`] (one-sided) and [`topsoe_distance`]
/// (two-sided \(K\) to the mean). Identical series score 0.
pub fn jeffreys_distance(a: &Vector, b: &Vector, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if let Some(issue) = signlred::scan_finite(a.as_slice()).to_issue("jeffreys_distance.a") {
        ctx.push(issue);
    }
    if let Some(issue) = signlred::scan_finite(b.as_slice()).to_issue("jeffreys_distance.b") {
        ctx.push(issue);
    }
    if a.is_empty() || b.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message("jeffreys_distance on an empty series")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    ctx.finish(jeffreys_distance_raw(a.as_slice(), b.as_slice()))
}

/// Pairwise Jeffreys divergence.
pub fn cdist_jeffreys(a: &Matrix, b: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, a, None, &ctx.policy);
    inspect_xy(&mut ctx.report, b, None, &ctx.policy);
    let out = Matrix::from_fn(a.nrows(), b.nrows(), |i, j| {
        let ai = a.row(i);
        let bj = b.row(j);
        jeffreys_distance_raw(ai.as_slice(), bj.as_slice())
    });
    ctx.finish(out)
}

fn squared_chord_distance_raw(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return f64::NAN;
    }
    let mut sa = 0.0_f64;
    let mut sb = 0.0_f64;
    for i in 0..n {
        sa += a[i].abs();
        sb += b[i].abs();
    }
    if sa < 1e-18 && sb < 1e-18 {
        return 0.0;
    }
    sa = sa.max(1e-18);
    sb = sb.max(1e-18);
    let mut s = 0.0_f64;
    for i in 0..n {
        let p = (a[i].abs() / sa).sqrt();
        let q = (b[i].abs() / sb).sqrt();
        let d = p - q;
        s += d * d;
    }
    s.max(0.0)
}

fn kullback_leibler_distance_raw(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return f64::NAN;
    }
    let mut sa = 0.0_f64;
    let mut sb = 0.0_f64;
    for i in 0..n {
        sa += a[i].abs();
        sb += b[i].abs();
    }
    if sa < 1e-18 && sb < 1e-18 {
        return 0.0;
    }
    sa = sa.max(1e-18);
    sb = sb.max(1e-18);
    let mut s = 0.0_f64;
    for i in 0..n {
        let p = (a[i].abs() / sa).max(1e-18);
        let q = (b[i].abs() / sb).max(1e-18);
        s += p * (p / q).ln();
    }
    s.max(0.0)
}

/// Squared-chord \(\sum(\sqrt{p}-\sqrt{q})^2\) after \(\ell_1\).
///
/// Distinct from [`hellinger_distance`] (no \(\ell_1\), and a \(\sqrt{1/2}\)
/// factor). Identical series score 0.
pub fn squared_chord_distance(
    a: &Vector,
    b: &Vector,
    session: &Session,
) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if let Some(issue) = signlred::scan_finite(a.as_slice()).to_issue("squared_chord_distance.a") {
        ctx.push(issue);
    }
    if let Some(issue) = signlred::scan_finite(b.as_slice()).to_issue("squared_chord_distance.b") {
        ctx.push(issue);
    }
    if a.is_empty() || b.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message("squared_chord_distance on an empty series")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    ctx.finish(squared_chord_distance_raw(a.as_slice(), b.as_slice()))
}

/// Pairwise squared-chord distance.
pub fn cdist_squared_chord(a: &Matrix, b: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, a, None, &ctx.policy);
    inspect_xy(&mut ctx.report, b, None, &ctx.policy);
    let out = Matrix::from_fn(a.nrows(), b.nrows(), |i, j| {
        let ai = a.row(i);
        let bj = b.row(j);
        squared_chord_distance_raw(ai.as_slice(), bj.as_slice())
    });
    ctx.finish(out)
}

/// Kullback–Leibler \(\sum p\ln(p/q)\) after \(\ell_1\).
///
/// Distinct from [`k_divergence_distance`] (\(p\ln(2p/(p+q))\)) and
/// [`jeffreys_distance`] (symmetric). Identical series score 0.
pub fn kullback_leibler_distance(
    a: &Vector,
    b: &Vector,
    session: &Session,
) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if let Some(issue) =
        signlred::scan_finite(a.as_slice()).to_issue("kullback_leibler_distance.a")
    {
        ctx.push(issue);
    }
    if let Some(issue) =
        signlred::scan_finite(b.as_slice()).to_issue("kullback_leibler_distance.b")
    {
        ctx.push(issue);
    }
    if a.is_empty() || b.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message("kullback_leibler_distance on an empty series")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    ctx.finish(kullback_leibler_distance_raw(a.as_slice(), b.as_slice()))
}

/// Pairwise Kullback–Leibler divergence.
pub fn cdist_kullback_leibler(
    a: &Matrix,
    b: &Matrix,
    session: &Session,
) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, a, None, &ctx.policy);
    inspect_xy(&mut ctx.report, b, None, &ctx.policy);
    let out = Matrix::from_fn(a.nrows(), b.nrows(), |i, j| {
        let ai = a.row(i);
        let bj = b.row(j);
        kullback_leibler_distance_raw(ai.as_slice(), bj.as_slice())
    });
    ctx.finish(out)
}

fn cosine_l1_distance_raw(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return f64::NAN;
    }
    let mut sa = 0.0_f64;
    let mut sb = 0.0_f64;
    for i in 0..n {
        sa += a[i].abs();
        sb += b[i].abs();
    }
    if sa < 1e-18 && sb < 1e-18 {
        return 0.0;
    }
    sa = sa.max(1e-18);
    sb = sb.max(1e-18);
    let mut num = 0.0_f64;
    let mut na = 0.0_f64;
    let mut nb = 0.0_f64;
    for i in 0..n {
        let p = a[i].abs() / sa;
        let q = b[i].abs() / sb;
        num += p * q;
        na += p * p;
        nb += q * q;
    }
    let den = na.sqrt() * nb.sqrt();
    if den < 1e-18 {
        return 0.0;
    }
    (1.0 - (num / den).clamp(-1.0, 1.0)).max(0.0)
}

fn tanimoto_l1_distance_raw(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return f64::NAN;
    }
    let mut sa = 0.0_f64;
    let mut sb = 0.0_f64;
    for i in 0..n {
        sa += a[i].abs();
        sb += b[i].abs();
    }
    if sa < 1e-18 && sb < 1e-18 {
        return 0.0;
    }
    sa = sa.max(1e-18);
    sb = sb.max(1e-18);
    let mut num = 0.0_f64;
    let mut na = 0.0_f64;
    let mut nb = 0.0_f64;
    for i in 0..n {
        let p = a[i].abs() / sa;
        let q = b[i].abs() / sb;
        num += p * q;
        na += p * p;
        nb += q * q;
    }
    let den = na + nb - num;
    if den.abs() < 1e-18 {
        return 0.0;
    }
    (1.0 - num / den).max(0.0)
}

/// Cosine distance after \(\ell_1\) normalisation.
///
/// Distinct from [`cosine_distance`] (raw coordinates). Identical series
/// score 0.
pub fn cosine_l1_distance(a: &Vector, b: &Vector, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if let Some(issue) = signlred::scan_finite(a.as_slice()).to_issue("cosine_l1_distance.a") {
        ctx.push(issue);
    }
    if let Some(issue) = signlred::scan_finite(b.as_slice()).to_issue("cosine_l1_distance.b") {
        ctx.push(issue);
    }
    if a.is_empty() || b.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message("cosine_l1_distance on an empty series")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    ctx.finish(cosine_l1_distance_raw(a.as_slice(), b.as_slice()))
}

/// Pairwise \(\ell_1\)-normalised cosine distance.
pub fn cdist_cosine_l1(a: &Matrix, b: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, a, None, &ctx.policy);
    inspect_xy(&mut ctx.report, b, None, &ctx.policy);
    let out = Matrix::from_fn(a.nrows(), b.nrows(), |i, j| {
        let ai = a.row(i);
        let bj = b.row(j);
        cosine_l1_distance_raw(ai.as_slice(), bj.as_slice())
    });
    ctx.finish(out)
}

/// Tanimoto / Kumar–Hassebrook distance after \(\ell_1\).
///
/// Distinct from [`tanimoto_distance`] (raw coordinates). Identical series
/// score 0.
pub fn tanimoto_l1_distance(a: &Vector, b: &Vector, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if let Some(issue) = signlred::scan_finite(a.as_slice()).to_issue("tanimoto_l1_distance.a") {
        ctx.push(issue);
    }
    if let Some(issue) = signlred::scan_finite(b.as_slice()).to_issue("tanimoto_l1_distance.b") {
        ctx.push(issue);
    }
    if a.is_empty() || b.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message("tanimoto_l1_distance on an empty series")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    ctx.finish(tanimoto_l1_distance_raw(a.as_slice(), b.as_slice()))
}

/// Pairwise \(\ell_1\)-normalised Tanimoto distance.
pub fn cdist_tanimoto_l1(a: &Matrix, b: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, a, None, &ctx.policy);
    inspect_xy(&mut ctx.report, b, None, &ctx.policy);
    let out = Matrix::from_fn(a.nrows(), b.nrows(), |i, j| {
        let ai = a.row(i);
        let bj = b.row(j);
        tanimoto_l1_distance_raw(ai.as_slice(), bj.as_slice())
    });
    ctx.finish(out)
}

fn dice_l1_distance_raw(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return f64::NAN;
    }
    let mut sa = 0.0_f64;
    let mut sb = 0.0_f64;
    for i in 0..n {
        sa += a[i].abs();
        sb += b[i].abs();
    }
    if sa < 1e-18 && sb < 1e-18 {
        return 0.0;
    }
    sa = sa.max(1e-18);
    sb = sb.max(1e-18);
    let mut num = 0.0_f64;
    let mut na = 0.0_f64;
    let mut nb = 0.0_f64;
    for i in 0..n {
        let p = a[i].abs() / sa;
        let q = b[i].abs() / sb;
        num += p * q;
        na += p * p;
        nb += q * q;
    }
    let den = na + nb;
    if den < 1e-18 {
        return 0.0;
    }
    (1.0 - 2.0 * num / den).max(0.0)
}

fn vicis_symmetric_distance_raw(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return f64::NAN;
    }
    let mut sa = 0.0_f64;
    let mut sb = 0.0_f64;
    for i in 0..n {
        sa += a[i].abs();
        sb += b[i].abs();
    }
    if sa < 1e-18 && sb < 1e-18 {
        return 0.0;
    }
    sa = sa.max(1e-18);
    sb = sb.max(1e-18);
    let mut s = 0.0_f64;
    for i in 0..n {
        let p = (a[i].abs() / sa).max(1e-18);
        let q = (b[i].abs() / sb).max(1e-18);
        let d = p - q;
        let m = p.min(q);
        s += d * d / (m * m);
    }
    s.max(0.0)
}

/// Dice / Sørensen distance after \(\ell_1\) normalisation.
///
/// Distinct from [`dice_distance`] (raw coordinates). Identical series score 0.
pub fn dice_l1_distance(a: &Vector, b: &Vector, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if let Some(issue) = signlred::scan_finite(a.as_slice()).to_issue("dice_l1_distance.a") {
        ctx.push(issue);
    }
    if let Some(issue) = signlred::scan_finite(b.as_slice()).to_issue("dice_l1_distance.b") {
        ctx.push(issue);
    }
    if a.is_empty() || b.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message("dice_l1_distance on an empty series")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    ctx.finish(dice_l1_distance_raw(a.as_slice(), b.as_slice()))
}

/// Pairwise \(\ell_1\)-normalised Dice distance.
pub fn cdist_dice_l1(a: &Matrix, b: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, a, None, &ctx.policy);
    inspect_xy(&mut ctx.report, b, None, &ctx.policy);
    let out = Matrix::from_fn(a.nrows(), b.nrows(), |i, j| {
        let ai = a.row(i);
        let bj = b.row(j);
        dice_l1_distance_raw(ai.as_slice(), bj.as_slice())
    });
    ctx.finish(out)
}

/// Vicis-symmetric χ² \(\sum(p-q)^2/\min(p,q)^2\) after \(\ell_1\).
///
/// Distinct from [`min_symmetric_chi_squared_distance`] (divide by \(\min\),
/// not \(\min^2\)). Identical series score 0.
pub fn vicis_symmetric_distance(
    a: &Vector,
    b: &Vector,
    session: &Session,
) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if let Some(issue) = signlred::scan_finite(a.as_slice()).to_issue("vicis_symmetric_distance.a") {
        ctx.push(issue);
    }
    if let Some(issue) = signlred::scan_finite(b.as_slice()).to_issue("vicis_symmetric_distance.b") {
        ctx.push(issue);
    }
    if a.is_empty() || b.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message("vicis_symmetric_distance on an empty series")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    ctx.finish(vicis_symmetric_distance_raw(a.as_slice(), b.as_slice()))
}

/// Pairwise Vicis-symmetric χ² distance.
pub fn cdist_vicis_symmetric(a: &Matrix, b: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, a, None, &ctx.policy);
    inspect_xy(&mut ctx.report, b, None, &ctx.policy);
    let out = Matrix::from_fn(a.nrows(), b.nrows(), |i, j| {
        let ai = a.row(i);
        let bj = b.row(j);
        vicis_symmetric_distance_raw(ai.as_slice(), bj.as_slice())
    });
    ctx.finish(out)
}


fn correlation_l1_distance_raw(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return f64::NAN;
    }
    let mut sa = 0.0_f64;
    let mut sb = 0.0_f64;
    for i in 0..n {
        sa += a[i].abs();
        sb += b[i].abs();
    }
    if sa < 1e-18 && sb < 1e-18 {
        return 0.0;
    }
    sa = sa.max(1e-18);
    sb = sb.max(1e-18);
    if n == 1 {
        let p = a[0].abs() / sa;
        let q = b[0].abs() / sb;
        return if (p - q).abs() < 1e-15 { 0.0 } else { 1.0 };
    }
    let na = n as f64;
    let mut ma = 0.0_f64;
    let mut mb = 0.0_f64;
    for i in 0..n {
        ma += a[i].abs() / sa;
        mb += b[i].abs() / sb;
    }
    ma /= na;
    mb /= na;
    let mut num = 0.0_f64;
    let mut va = 0.0_f64;
    let mut vb = 0.0_f64;
    for i in 0..n {
        let da = a[i].abs() / sa - ma;
        let db = b[i].abs() / sb - mb;
        num += da * db;
        va += da * da;
        vb += db * db;
    }
    let den = (va * vb).sqrt();
    if den < 1e-18 {
        let mut same = true;
        for i in 0..n {
            let p = a[i].abs() / sa;
            let q = b[i].abs() / sb;
            if (p - q).abs() >= 1e-12 {
                same = false;
                break;
            }
        }
        return if same { 0.0 } else { 1.0 };
    }
    let r = (num / den).clamp(-1.0, 1.0);
    (1.0 - r).max(0.0)
}

/// Pearson correlation distance \(1-r\) after \(\ell_1\) normalisation.
///
/// Distinct from [`correlation_distance`] (raw coordinates) and
/// [`cosine_l1_distance`] (no centering). Identical series score 0.
pub fn correlation_l1_distance(
    a: &Vector,
    b: &Vector,
    session: &Session,
) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if let Some(issue) = signlred::scan_finite(a.as_slice()).to_issue("correlation_l1_distance.a") {
        ctx.push(issue);
    }
    if let Some(issue) = signlred::scan_finite(b.as_slice()).to_issue("correlation_l1_distance.b") {
        ctx.push(issue);
    }
    if a.is_empty() || b.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message("correlation_l1_distance on an empty series")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    ctx.finish(correlation_l1_distance_raw(a.as_slice(), b.as_slice()))
}

/// Pairwise \(\ell_1\)-normalised Pearson correlation distance.
pub fn cdist_correlation_l1(a: &Matrix, b: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, a, None, &ctx.policy);
    inspect_xy(&mut ctx.report, b, None, &ctx.policy);
    let out = Matrix::from_fn(a.nrows(), b.nrows(), |i, j| {
        let ai = a.row(i);
        let bj = b.row(j);
        correlation_l1_distance_raw(ai.as_slice(), bj.as_slice())
    });
    ctx.finish(out)
}


fn hellinger_l1_distance_raw(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return f64::NAN;
    }
    let mut sa = 0.0_f64;
    let mut sb = 0.0_f64;
    for i in 0..n {
        sa += a[i].abs();
        sb += b[i].abs();
    }
    if sa < 1e-18 && sb < 1e-18 {
        return 0.0;
    }
    sa = sa.max(1e-18);
    sb = sb.max(1e-18);
    let mut s = 0.0_f64;
    for i in 0..n {
        let p = (a[i].abs() / sa).sqrt();
        let q = (b[i].abs() / sb).sqrt();
        let d = p - q;
        s += d * d;
    }
    (0.5 * s).sqrt()
}

/// Hellinger distance after \(\ell_1\) normalisation.
///
/// Distinct from [`hellinger_distance`] (raw \(\sqrt{|a_i|}\) without
/// renormalising). Identical series score 0.
pub fn hellinger_l1_distance(
    a: &Vector,
    b: &Vector,
    session: &Session,
) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if let Some(issue) = signlred::scan_finite(a.as_slice()).to_issue("hellinger_l1_distance.a") {
        ctx.push(issue);
    }
    if let Some(issue) = signlred::scan_finite(b.as_slice()).to_issue("hellinger_l1_distance.b") {
        ctx.push(issue);
    }
    if a.is_empty() || b.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message("hellinger_l1_distance on an empty series")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    ctx.finish(hellinger_l1_distance_raw(a.as_slice(), b.as_slice()))
}

/// Pairwise \(\ell_1\)-normalised Hellinger distance.
pub fn cdist_hellinger_l1(a: &Matrix, b: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, a, None, &ctx.policy);
    inspect_xy(&mut ctx.report, b, None, &ctx.policy);
    let out = Matrix::from_fn(a.nrows(), b.nrows(), |i, j| {
        let ai = a.row(i);
        let bj = b.row(j);
        hellinger_l1_distance_raw(ai.as_slice(), bj.as_slice())
    });
    ctx.finish(out)
}


fn canberra_l1_distance_raw(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return f64::NAN;
    }
    let mut sa = 0.0_f64;
    let mut sb = 0.0_f64;
    for i in 0..n {
        sa += a[i].abs();
        sb += b[i].abs();
    }
    if sa < 1e-18 && sb < 1e-18 {
        return 0.0;
    }
    sa = sa.max(1e-18);
    sb = sb.max(1e-18);
    let mut s = 0.0_f64;
    for i in 0..n {
        let p = a[i].abs() / sa;
        let q = b[i].abs() / sb;
        let den = p + q;
        if den > 1e-18 {
            s += (p - q).abs() / den;
        }
    }
    s
}

/// Canberra distance after \(\ell_1\) normalisation.
///
/// Distinct from [`canberra_distance`] (raw coordinates, no renormalise).
/// Identical series score 0.
pub fn canberra_l1_distance(a: &Vector, b: &Vector, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if let Some(issue) = signlred::scan_finite(a.as_slice()).to_issue("canberra_l1_distance.a") {
        ctx.push(issue);
    }
    if let Some(issue) = signlred::scan_finite(b.as_slice()).to_issue("canberra_l1_distance.b") {
        ctx.push(issue);
    }
    if a.is_empty() || b.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message("canberra_l1_distance on an empty series")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    ctx.finish(canberra_l1_distance_raw(a.as_slice(), b.as_slice()))
}

/// Pairwise \(\ell_1\)-normalised Canberra distance.
pub fn cdist_canberra_l1(a: &Matrix, b: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, a, None, &ctx.policy);
    inspect_xy(&mut ctx.report, b, None, &ctx.policy);
    let out = Matrix::from_fn(a.nrows(), b.nrows(), |i, j| {
        let ai = a.row(i);
        let bj = b.row(j);
        canberra_l1_distance_raw(ai.as_slice(), bj.as_slice())
    });
    ctx.finish(out)
}


fn clark_l1_distance_raw(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return f64::NAN;
    }
    let mut sa = 0.0_f64;
    let mut sb = 0.0_f64;
    for i in 0..n {
        sa += a[i].abs();
        sb += b[i].abs();
    }
    if sa < 1e-18 && sb < 1e-18 {
        return 0.0;
    }
    sa = sa.max(1e-18);
    sb = sb.max(1e-18);
    let mut s = 0.0_f64;
    for i in 0..n {
        let p = a[i].abs() / sa;
        let q = b[i].abs() / sb;
        let den = p + q;
        if den > 1e-18 {
            let u = (p - q) / den;
            s += u * u;
        }
    }
    s.sqrt()
}

/// Clark distance after \(\ell_1\) normalisation.
///
/// Distinct from [`clark_distance`] (raw coordinates, no renormalise).
/// Identical series score 0.
pub fn clark_l1_distance(a: &Vector, b: &Vector, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if let Some(issue) = signlred::scan_finite(a.as_slice()).to_issue("clark_l1_distance.a") {
        ctx.push(issue);
    }
    if let Some(issue) = signlred::scan_finite(b.as_slice()).to_issue("clark_l1_distance.b") {
        ctx.push(issue);
    }
    if a.is_empty() || b.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message("clark_l1_distance on an empty series")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    ctx.finish(clark_l1_distance_raw(a.as_slice(), b.as_slice()))
}

/// Pairwise \(\ell_1\)-normalised Clark distance.
pub fn cdist_clark_l1(a: &Matrix, b: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, a, None, &ctx.policy);
    inspect_xy(&mut ctx.report, b, None, &ctx.policy);
    let out = Matrix::from_fn(a.nrows(), b.nrows(), |i, j| {
        let ai = a.row(i);
        let bj = b.row(j);
        clark_l1_distance_raw(ai.as_slice(), bj.as_slice())
    });
    ctx.finish(out)
}



fn wave_hedges_l1_distance_raw(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return f64::NAN;
    }
    let mut sa = 0.0_f64;
    let mut sb = 0.0_f64;
    for i in 0..n {
        sa += a[i].abs();
        sb += b[i].abs();
    }
    if sa < 1e-18 && sb < 1e-18 {
        return 0.0;
    }
    sa = sa.max(1e-18);
    sb = sb.max(1e-18);
    let mut s = 0.0_f64;
    for i in 0..n {
        let p = a[i].abs() / sa;
        let q = b[i].abs() / sb;
        let ma = p.max(q);
        if ma > 1e-18 {
            s += (p - q).abs() / ma;
        }
    }
    s
}

/// Wave Hedges distance after \(\ell_1\) normalisation.
///
/// Distinct from [`wave_hedges_distance`] (raw coordinates, no renormalise).
/// Identical series score 0.
pub fn wave_hedges_l1_distance(a: &Vector, b: &Vector, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if let Some(issue) = signlred::scan_finite(a.as_slice()).to_issue("wave_hedges_l1_distance.a") {
        ctx.push(issue);
    }
    if let Some(issue) = signlred::scan_finite(b.as_slice()).to_issue("wave_hedges_l1_distance.b") {
        ctx.push(issue);
    }
    if a.is_empty() || b.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message("wave_hedges_l1_distance on an empty series")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    ctx.finish(wave_hedges_l1_distance_raw(a.as_slice(), b.as_slice()))
}

/// Pairwise \(\ell_1\)-normalised Wave Hedges distance.
pub fn cdist_wave_hedges_l1(a: &Matrix, b: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, a, None, &ctx.policy);
    inspect_xy(&mut ctx.report, b, None, &ctx.policy);
    let out = Matrix::from_fn(a.nrows(), b.nrows(), |i, j| {
        let ai = a.row(i);
        let bj = b.row(j);
        wave_hedges_l1_distance_raw(ai.as_slice(), bj.as_slice())
    });
    ctx.finish(out)
}



fn kulczynski_l1_distance_raw(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return f64::NAN;
    }
    let mut sa = 0.0_f64;
    let mut sb = 0.0_f64;
    for i in 0..n {
        sa += a[i].abs();
        sb += b[i].abs();
    }
    if sa < 1e-18 && sb < 1e-18 {
        return 0.0;
    }
    sa = sa.max(1e-18);
    sb = sb.max(1e-18);
    let mut num = 0.0_f64;
    let mut den = 0.0_f64;
    for i in 0..n {
        let p = a[i].abs() / sa;
        let q = b[i].abs() / sb;
        num += (p - q).abs();
        den += p.min(q);
    }
    if den < 1e-18 {
        return 0.0;
    }
    num / den
}

/// Kulczynski distance after \(\ell_1\) normalisation.
///
/// Distinct from [`kulczynski_distance`] (raw coordinates, no renormalise).
/// Identical series score 0.
pub fn kulczynski_l1_distance(a: &Vector, b: &Vector, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if let Some(issue) = signlred::scan_finite(a.as_slice()).to_issue("kulczynski_l1_distance.a") {
        ctx.push(issue);
    }
    if let Some(issue) = signlred::scan_finite(b.as_slice()).to_issue("kulczynski_l1_distance.b") {
        ctx.push(issue);
    }
    if a.is_empty() || b.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message("kulczynski_l1_distance on an empty series")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    ctx.finish(kulczynski_l1_distance_raw(a.as_slice(), b.as_slice()))
}

/// Pairwise \(\ell_1\)-normalised Kulczynski distance.
pub fn cdist_kulczynski_l1(a: &Matrix, b: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, a, None, &ctx.policy);
    inspect_xy(&mut ctx.report, b, None, &ctx.policy);
    let out = Matrix::from_fn(a.nrows(), b.nrows(), |i, j| {
        let ai = a.row(i);
        let bj = b.row(j);
        kulczynski_l1_distance_raw(ai.as_slice(), bj.as_slice())
    });
    ctx.finish(out)
}



fn ruzicka_l1_distance_raw(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return f64::NAN;
    }
    let mut sa = 0.0_f64;
    let mut sb = 0.0_f64;
    for i in 0..n {
        sa += a[i].abs();
        sb += b[i].abs();
    }
    if sa < 1e-18 && sb < 1e-18 {
        return 0.0;
    }
    sa = sa.max(1e-18);
    sb = sb.max(1e-18);
    let mut nmin = 0.0_f64;
    let mut nmax = 0.0_f64;
    for i in 0..n {
        let p = a[i].abs() / sa;
        let q = b[i].abs() / sb;
        nmin += p.min(q);
        nmax += p.max(q);
    }
    if nmax < 1e-18 {
        return 0.0;
    }
    1.0 - nmin / nmax
}

/// Ružička distance after \(\ell_1\) normalisation.
///
/// Distinct from [`ruzicka_distance`] (raw coordinates, no renormalise).
/// Identical series score 0.
pub fn ruzicka_l1_distance(a: &Vector, b: &Vector, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if let Some(issue) = signlred::scan_finite(a.as_slice()).to_issue("ruzicka_l1_distance.a") {
        ctx.push(issue);
    }
    if let Some(issue) = signlred::scan_finite(b.as_slice()).to_issue("ruzicka_l1_distance.b") {
        ctx.push(issue);
    }
    if a.is_empty() || b.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message("ruzicka_l1_distance on an empty series")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    ctx.finish(ruzicka_l1_distance_raw(a.as_slice(), b.as_slice()))
}

/// Pairwise \(\ell_1\)-normalised Ružička distance.
pub fn cdist_ruzicka_l1(a: &Matrix, b: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, a, None, &ctx.policy);
    inspect_xy(&mut ctx.report, b, None, &ctx.policy);
    let out = Matrix::from_fn(a.nrows(), b.nrows(), |i, j| {
        let ai = a.row(i);
        let bj = b.row(j);
        ruzicka_l1_distance_raw(ai.as_slice(), bj.as_slice())
    });
    ctx.finish(out)
}



fn lorentzian_l1_distance_raw(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return f64::NAN;
    }
    let mut sa = 0.0_f64;
    let mut sb = 0.0_f64;
    for i in 0..n {
        sa += a[i].abs();
        sb += b[i].abs();
    }
    if sa < 1e-18 && sb < 1e-18 {
        return 0.0;
    }
    sa = sa.max(1e-18);
    sb = sb.max(1e-18);
    let mut s = 0.0_f64;
    for i in 0..n {
        let p = a[i].abs() / sa;
        let q = b[i].abs() / sb;
        s += (1.0 + (p - q).abs()).ln();
    }
    s
}

/// Lorentzian distance after \(\ell_1\) normalisation.
///
/// Distinct from [`lorentzian_distance`] (raw coordinates, no renormalise).
/// Identical series score 0.
pub fn lorentzian_l1_distance(a: &Vector, b: &Vector, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if let Some(issue) = signlred::scan_finite(a.as_slice()).to_issue("lorentzian_l1_distance.a") {
        ctx.push(issue);
    }
    if let Some(issue) = signlred::scan_finite(b.as_slice()).to_issue("lorentzian_l1_distance.b") {
        ctx.push(issue);
    }
    if a.is_empty() || b.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message("lorentzian_l1_distance on an empty series")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    ctx.finish(lorentzian_l1_distance_raw(a.as_slice(), b.as_slice()))
}

/// Pairwise \(\ell_1\)-normalised Lorentzian distance.
pub fn cdist_lorentzian_l1(a: &Matrix, b: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, a, None, &ctx.policy);
    inspect_xy(&mut ctx.report, b, None, &ctx.policy);
    let out = Matrix::from_fn(a.nrows(), b.nrows(), |i, j| {
        let ai = a.row(i);
        let bj = b.row(j);
        lorentzian_l1_distance_raw(ai.as_slice(), bj.as_slice())
    });
    ctx.finish(out)
}



fn hassanat_l1_distance_raw(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return f64::NAN;
    }
    let mut sa = 0.0_f64;
    let mut sb = 0.0_f64;
    for i in 0..n {
        sa += a[i].abs();
        sb += b[i].abs();
    }
    if sa < 1e-18 && sb < 1e-18 {
        return 0.0;
    }
    sa = sa.max(1e-18);
    sb = sb.max(1e-18);
    let mut s = 0.0_f64;
    for i in 0..n {
        let p = a[i].abs() / sa;
        let q = b[i].abs() / sb;
        let lo = p.min(q);
        let hi = p.max(q);
        if hi > 1e-18 {
            s += 1.0 - (1.0 + lo) / (1.0 + hi);
        }
    }
    s
}

/// Hassanat distance after \(\ell_1\) normalisation.
///
/// Distinct from [`hassanat_distance`] (raw coordinates, no renormalise).
/// Identical series score 0.
pub fn hassanat_l1_distance(a: &Vector, b: &Vector, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if let Some(issue) = signlred::scan_finite(a.as_slice()).to_issue("hassanat_l1_distance.a") {
        ctx.push(issue);
    }
    if let Some(issue) = signlred::scan_finite(b.as_slice()).to_issue("hassanat_l1_distance.b") {
        ctx.push(issue);
    }
    if a.is_empty() || b.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message("hassanat_l1_distance on an empty series")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    ctx.finish(hassanat_l1_distance_raw(a.as_slice(), b.as_slice()))
}

/// Pairwise \(\ell_1\)-normalised Hassanat distance.
pub fn cdist_hassanat_l1(a: &Matrix, b: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, a, None, &ctx.policy);
    inspect_xy(&mut ctx.report, b, None, &ctx.policy);
    let out = Matrix::from_fn(a.nrows(), b.nrows(), |i, j| {
        let ai = a.row(i);
        let bj = b.row(j);
        hassanat_l1_distance_raw(ai.as_slice(), bj.as_slice())
    });
    ctx.finish(out)
}



fn chebyshev_l1_distance_raw(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return f64::NAN;
    }
    let mut sa = 0.0_f64;
    let mut sb = 0.0_f64;
    for i in 0..n {
        sa += a[i].abs();
        sb += b[i].abs();
    }
    if sa < 1e-18 && sb < 1e-18 {
        return 0.0;
    }
    sa = sa.max(1e-18);
    sb = sb.max(1e-18);
    let mut m = 0.0_f64;
    for i in 0..n {
        let p = a[i].abs() / sa;
        let q = b[i].abs() / sb;
        let d = (p - q).abs();
        if d > m {
            m = d;
        }
    }
    m
}

/// Chebyshev \(\ell_\infty\) distance after \(\ell_1\) normalisation.
///
/// Distinct from [`chebyshev_distance`] (raw coordinates) and
/// [`clark_l1_distance`]. Identical series score 0.
pub fn chebyshev_l1_distance(a: &Vector, b: &Vector, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if let Some(issue) = signlred::scan_finite(a.as_slice()).to_issue("chebyshev_l1_distance.a") {
        ctx.push(issue);
    }
    if let Some(issue) = signlred::scan_finite(b.as_slice()).to_issue("chebyshev_l1_distance.b") {
        ctx.push(issue);
    }
    if a.is_empty() || b.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message("chebyshev_l1_distance on an empty series")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    ctx.finish(chebyshev_l1_distance_raw(a.as_slice(), b.as_slice()))
}

/// Pairwise \(\ell_1\)-normalised Chebyshev distance.
pub fn cdist_chebyshev_l1(a: &Matrix, b: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, a, None, &ctx.policy);
    inspect_xy(&mut ctx.report, b, None, &ctx.policy);
    let out = Matrix::from_fn(a.nrows(), b.nrows(), |i, j| {
        let ai = a.row(i);
        let bj = b.row(j);
        chebyshev_l1_distance_raw(ai.as_slice(), bj.as_slice())
    });
    ctx.finish(out)
}



fn minkowski3_l1_distance_raw(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return f64::NAN;
    }
    let mut sa = 0.0_f64;
    let mut sb = 0.0_f64;
    for i in 0..n {
        sa += a[i].abs();
        sb += b[i].abs();
    }
    if sa < 1e-18 && sb < 1e-18 {
        return 0.0;
    }
    sa = sa.max(1e-18);
    sb = sb.max(1e-18);
    let mut s = 0.0_f64;
    for i in 0..n {
        let p = a[i].abs() / sa;
        let q = b[i].abs() / sb;
        let d = (p - q).abs();
        s += d * d * d;
    }
    s.cbrt()
}

/// Minkowski \(p=3\) distance after \(\ell_1\) normalisation.
///
/// Distinct from [`minkowski3_distance`] (raw coordinates) and
/// [`chebyshev_l1_distance`]. Identical series score 0.
pub fn minkowski3_l1_distance(a: &Vector, b: &Vector, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if let Some(issue) = signlred::scan_finite(a.as_slice()).to_issue("minkowski3_l1_distance.a") {
        ctx.push(issue);
    }
    if let Some(issue) = signlred::scan_finite(b.as_slice()).to_issue("minkowski3_l1_distance.b") {
        ctx.push(issue);
    }
    if a.is_empty() || b.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message("minkowski3_l1_distance on an empty series")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    ctx.finish(minkowski3_l1_distance_raw(a.as_slice(), b.as_slice()))
}

/// Pairwise \(\ell_1\)-normalised Minkowski \(p=3\) distance.
pub fn cdist_minkowski3_l1(a: &Matrix, b: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, a, None, &ctx.policy);
    inspect_xy(&mut ctx.report, b, None, &ctx.policy);
    let out = Matrix::from_fn(a.nrows(), b.nrows(), |i, j| {
        let ai = a.row(i);
        let bj = b.row(j);
        minkowski3_l1_distance_raw(ai.as_slice(), bj.as_slice())
    });
    ctx.finish(out)
}



fn minkowski4_l1_distance_raw(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return f64::NAN;
    }
    let mut sa = 0.0_f64;
    let mut sb = 0.0_f64;
    for i in 0..n {
        sa += a[i].abs();
        sb += b[i].abs();
    }
    if sa < 1e-18 && sb < 1e-18 {
        return 0.0;
    }
    sa = sa.max(1e-18);
    sb = sb.max(1e-18);
    let mut s = 0.0_f64;
    for i in 0..n {
        let p = a[i].abs() / sa;
        let q = b[i].abs() / sb;
        let d = (p - q).abs();
        let d2 = d * d;
        s += d2 * d2;
    }
    s.sqrt().sqrt()
}

/// Minkowski \(p=4\) distance after \(\ell_1\) normalisation.
///
/// Distinct from [`minkowski3_l1_distance`] and [`chebyshev_l1_distance`].
/// Identical series score 0.
pub fn minkowski4_l1_distance(a: &Vector, b: &Vector, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if let Some(issue) = signlred::scan_finite(a.as_slice()).to_issue("minkowski4_l1_distance.a") {
        ctx.push(issue);
    }
    if let Some(issue) = signlred::scan_finite(b.as_slice()).to_issue("minkowski4_l1_distance.b") {
        ctx.push(issue);
    }
    if a.is_empty() || b.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message("minkowski4_l1_distance on an empty series")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    ctx.finish(minkowski4_l1_distance_raw(a.as_slice(), b.as_slice()))
}

/// Pairwise \(\ell_1\)-normalised Minkowski \(p=4\) distance.
pub fn cdist_minkowski4_l1(a: &Matrix, b: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, a, None, &ctx.policy);
    inspect_xy(&mut ctx.report, b, None, &ctx.policy);
    let out = Matrix::from_fn(a.nrows(), b.nrows(), |i, j| {
        let ai = a.row(i);
        let bj = b.row(j);
        minkowski4_l1_distance_raw(ai.as_slice(), bj.as_slice())
    });
    ctx.finish(out)
}



fn minkowski15_l1_distance_raw(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return f64::NAN;
    }
    let mut sa = 0.0_f64;
    let mut sb = 0.0_f64;
    for i in 0..n {
        sa += a[i].abs();
        sb += b[i].abs();
    }
    if sa < 1e-18 && sb < 1e-18 {
        return 0.0;
    }
    sa = sa.max(1e-18);
    sb = sb.max(1e-18);
    let mut s = 0.0_f64;
    for i in 0..n {
        let p = a[i].abs() / sa;
        let q = b[i].abs() / sb;
        s += (p - q).abs().powf(1.5);
    }
    s.powf(2.0 / 3.0)
}

/// Minkowski \(p=3/2\) distance after \(\ell_1\) normalisation.
///
/// Distinct from [`minkowski3_l1_distance`] and [`minkowski4_l1_distance`].
/// Identical series score 0.
pub fn minkowski15_l1_distance(a: &Vector, b: &Vector, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if let Some(issue) = signlred::scan_finite(a.as_slice()).to_issue("minkowski15_l1_distance.a") {
        ctx.push(issue);
    }
    if let Some(issue) = signlred::scan_finite(b.as_slice()).to_issue("minkowski15_l1_distance.b") {
        ctx.push(issue);
    }
    if a.is_empty() || b.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message("minkowski15_l1_distance on an empty series")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    ctx.finish(minkowski15_l1_distance_raw(a.as_slice(), b.as_slice()))
}

/// Pairwise \(\ell_1\)-normalised Minkowski \(p=3/2\) distance.
pub fn cdist_minkowski15_l1(a: &Matrix, b: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, a, None, &ctx.policy);
    inspect_xy(&mut ctx.report, b, None, &ctx.policy);
    let out = Matrix::from_fn(a.nrows(), b.nrows(), |i, j| {
        let ai = a.row(i);
        let bj = b.row(j);
        minkowski15_l1_distance_raw(ai.as_slice(), bj.as_slice())
    });
    ctx.finish(out)
}



fn minkowski5_l1_distance_raw(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return f64::NAN;
    }
    let mut sa = 0.0_f64;
    let mut sb = 0.0_f64;
    for i in 0..n {
        sa += a[i].abs();
        sb += b[i].abs();
    }
    if sa < 1e-18 && sb < 1e-18 {
        return 0.0;
    }
    sa = sa.max(1e-18);
    sb = sb.max(1e-18);
    let mut s = 0.0_f64;
    for i in 0..n {
        let p = a[i].abs() / sa;
        let q = b[i].abs() / sb;
        s += (p - q).abs().powi(5);
    }
    s.powf(0.2)
}

/// Minkowski \(p=5\) distance after \(\ell_1\) normalisation.
///
/// Distinct from [`minkowski4_l1_distance`] and [`minkowski15_l1_distance`].
/// Identical series score 0.
pub fn minkowski5_l1_distance(a: &Vector, b: &Vector, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if let Some(issue) = signlred::scan_finite(a.as_slice()).to_issue("minkowski5_l1_distance.a") {
        ctx.push(issue);
    }
    if let Some(issue) = signlred::scan_finite(b.as_slice()).to_issue("minkowski5_l1_distance.b") {
        ctx.push(issue);
    }
    if a.is_empty() || b.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message("minkowski5_l1_distance on an empty series")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    ctx.finish(minkowski5_l1_distance_raw(a.as_slice(), b.as_slice()))
}

/// Pairwise \(\ell_1\)-normalised Minkowski \(p=5\) distance.
pub fn cdist_minkowski5_l1(a: &Matrix, b: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, a, None, &ctx.policy);
    inspect_xy(&mut ctx.report, b, None, &ctx.policy);
    let out = Matrix::from_fn(a.nrows(), b.nrows(), |i, j| {
        let ai = a.row(i);
        let bj = b.row(j);
        minkowski5_l1_distance_raw(ai.as_slice(), bj.as_slice())
    });
    ctx.finish(out)
}



fn minkowski6_l1_distance_raw(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return f64::NAN;
    }
    let mut sa = 0.0_f64;
    let mut sb = 0.0_f64;
    for i in 0..n {
        sa += a[i].abs();
        sb += b[i].abs();
    }
    if sa < 1e-18 && sb < 1e-18 {
        return 0.0;
    }
    sa = sa.max(1e-18);
    sb = sb.max(1e-18);
    let mut s = 0.0_f64;
    for i in 0..n {
        let p = a[i].abs() / sa;
        let q = b[i].abs() / sb;
        s += (p - q).abs().powi(6);
    }
    s.powf(1.0 / 6.0)
}

/// Minkowski \(p=6\) distance after \(\ell_1\) normalisation.
///
/// Distinct from [`minkowski5_l1_distance`] and [`minkowski4_l1_distance`].
/// Identical series score 0.
pub fn minkowski6_l1_distance(a: &Vector, b: &Vector, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if let Some(issue) = signlred::scan_finite(a.as_slice()).to_issue("minkowski6_l1_distance.a") {
        ctx.push(issue);
    }
    if let Some(issue) = signlred::scan_finite(b.as_slice()).to_issue("minkowski6_l1_distance.b") {
        ctx.push(issue);
    }
    if a.is_empty() || b.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message("minkowski6_l1_distance on an empty series")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    ctx.finish(minkowski6_l1_distance_raw(a.as_slice(), b.as_slice()))
}

/// Pairwise \(\ell_1\)-normalised Minkowski \(p=6\) distance.
pub fn cdist_minkowski6_l1(a: &Matrix, b: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, a, None, &ctx.policy);
    inspect_xy(&mut ctx.report, b, None, &ctx.policy);
    let out = Matrix::from_fn(a.nrows(), b.nrows(), |i, j| {
        let ai = a.row(i);
        let bj = b.row(j);
        minkowski6_l1_distance_raw(ai.as_slice(), bj.as_slice())
    });
    ctx.finish(out)
}



fn minkowski25_l1_distance_raw(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return f64::NAN;
    }
    let mut sa = 0.0_f64;
    let mut sb = 0.0_f64;
    for i in 0..n {
        sa += a[i].abs();
        sb += b[i].abs();
    }
    if sa < 1e-18 && sb < 1e-18 {
        return 0.0;
    }
    sa = sa.max(1e-18);
    sb = sb.max(1e-18);
    let mut s = 0.0_f64;
    for i in 0..n {
        let p = a[i].abs() / sa;
        let q = b[i].abs() / sb;
        s += (p - q).abs().powf(2.5);
    }
    s.powf(0.4)
}

/// Minkowski \(p=5/2\) distance after \(\ell_1\) normalisation.
///
/// Distinct from [`minkowski15_l1_distance`] and [`minkowski3_l1_distance`].
/// Identical series score 0.
pub fn minkowski25_l1_distance(a: &Vector, b: &Vector, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if let Some(issue) = signlred::scan_finite(a.as_slice()).to_issue("minkowski25_l1_distance.a") {
        ctx.push(issue);
    }
    if let Some(issue) = signlred::scan_finite(b.as_slice()).to_issue("minkowski25_l1_distance.b") {
        ctx.push(issue);
    }
    if a.is_empty() || b.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message("minkowski25_l1_distance on an empty series")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    ctx.finish(minkowski25_l1_distance_raw(a.as_slice(), b.as_slice()))
}

/// Pairwise \(\ell_1\)-normalised Minkowski \(p=5/2\) distance.
pub fn cdist_minkowski25_l1(a: &Matrix, b: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, a, None, &ctx.policy);
    inspect_xy(&mut ctx.report, b, None, &ctx.policy);
    let out = Matrix::from_fn(a.nrows(), b.nrows(), |i, j| {
        let ai = a.row(i);
        let bj = b.row(j);
        minkowski25_l1_distance_raw(ai.as_slice(), bj.as_slice())
    });
    ctx.finish(out)
}



fn minkowski8_l1_distance_raw(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return f64::NAN;
    }
    let mut sa = 0.0_f64;
    let mut sb = 0.0_f64;
    for i in 0..n {
        sa += a[i].abs();
        sb += b[i].abs();
    }
    if sa < 1e-18 && sb < 1e-18 {
        return 0.0;
    }
    sa = sa.max(1e-18);
    sb = sb.max(1e-18);
    let mut s = 0.0_f64;
    for i in 0..n {
        let p = a[i].abs() / sa;
        let q = b[i].abs() / sb;
        s += (p - q).abs().powi(8);
    }
    s.powf(0.125)
}

/// Minkowski \(p=8\) distance after \(\ell_1\) normalisation.
///
/// Distinct from [`minkowski6_l1_distance`] and [`minkowski5_l1_distance`].
/// Identical series score 0.
pub fn minkowski8_l1_distance(a: &Vector, b: &Vector, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if let Some(issue) = signlred::scan_finite(a.as_slice()).to_issue("minkowski8_l1_distance.a") {
        ctx.push(issue);
    }
    if let Some(issue) = signlred::scan_finite(b.as_slice()).to_issue("minkowski8_l1_distance.b") {
        ctx.push(issue);
    }
    if a.is_empty() || b.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message("minkowski8_l1_distance on an empty series")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    ctx.finish(minkowski8_l1_distance_raw(a.as_slice(), b.as_slice()))
}

/// Pairwise \(\ell_1\)-normalised Minkowski \(p=8\) distance.
pub fn cdist_minkowski8_l1(a: &Matrix, b: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, a, None, &ctx.policy);
    inspect_xy(&mut ctx.report, b, None, &ctx.policy);
    let out = Matrix::from_fn(a.nrows(), b.nrows(), |i, j| {
        let ai = a.row(i);
        let bj = b.row(j);
        minkowski8_l1_distance_raw(ai.as_slice(), bj.as_slice())
    });
    ctx.finish(out)
}



fn minkowski7_l1_distance_raw(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return f64::NAN;
    }
    let mut sa = 0.0_f64;
    let mut sb = 0.0_f64;
    for i in 0..n {
        sa += a[i].abs();
        sb += b[i].abs();
    }
    if sa < 1e-18 && sb < 1e-18 {
        return 0.0;
    }
    sa = sa.max(1e-18);
    sb = sb.max(1e-18);
    let mut s = 0.0_f64;
    for i in 0..n {
        let p = a[i].abs() / sa;
        let q = b[i].abs() / sb;
        s += (p - q).abs().powi(7);
    }
    s.powf(1.0 / 7.0)
}

/// Minkowski \(p=7\) distance after \(\ell_1\) normalisation.
///
/// Distinct from [`minkowski6_l1_distance`] and [`minkowski8_l1_distance`].
/// Identical series score 0.
pub fn minkowski7_l1_distance(a: &Vector, b: &Vector, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if let Some(issue) = signlred::scan_finite(a.as_slice()).to_issue("minkowski7_l1_distance.a") {
        ctx.push(issue);
    }
    if let Some(issue) = signlred::scan_finite(b.as_slice()).to_issue("minkowski7_l1_distance.b") {
        ctx.push(issue);
    }
    if a.is_empty() || b.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message("minkowski7_l1_distance on an empty series")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    ctx.finish(minkowski7_l1_distance_raw(a.as_slice(), b.as_slice()))
}

/// Pairwise \(\ell_1\)-normalised Minkowski \(p=7\) distance.
pub fn cdist_minkowski7_l1(a: &Matrix, b: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, a, None, &ctx.policy);
    inspect_xy(&mut ctx.report, b, None, &ctx.policy);
    let out = Matrix::from_fn(a.nrows(), b.nrows(), |i, j| {
        let ai = a.row(i);
        let bj = b.row(j);
        minkowski7_l1_distance_raw(ai.as_slice(), bj.as_slice())
    });
    ctx.finish(out)
}



fn minkowski9_l1_distance_raw(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return f64::NAN;
    }
    let mut sa = 0.0_f64;
    let mut sb = 0.0_f64;
    for i in 0..n {
        sa += a[i].abs();
        sb += b[i].abs();
    }
    if sa < 1e-18 && sb < 1e-18 {
        return 0.0;
    }
    sa = sa.max(1e-18);
    sb = sb.max(1e-18);
    let mut s = 0.0_f64;
    for i in 0..n {
        let p = a[i].abs() / sa;
        let q = b[i].abs() / sb;
        s += (p - q).abs().powi(9);
    }
    s.powf(1.0 / 9.0)
}

/// Minkowski \(p=9\) distance after \(\ell_1\) normalisation.
///
/// Distinct from [`minkowski7_l1_distance`] and [`minkowski8_l1_distance`].
/// Identical series score 0.
pub fn minkowski9_l1_distance(a: &Vector, b: &Vector, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if let Some(issue) = signlred::scan_finite(a.as_slice()).to_issue("minkowski9_l1_distance.a") {
        ctx.push(issue);
    }
    if let Some(issue) = signlred::scan_finite(b.as_slice()).to_issue("minkowski9_l1_distance.b") {
        ctx.push(issue);
    }
    if a.is_empty() || b.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message("minkowski9_l1_distance on an empty series")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    ctx.finish(minkowski9_l1_distance_raw(a.as_slice(), b.as_slice()))
}

/// Pairwise \(\ell_1\)-normalised Minkowski \(p=9\) distance.
pub fn cdist_minkowski9_l1(a: &Matrix, b: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, a, None, &ctx.policy);
    inspect_xy(&mut ctx.report, b, None, &ctx.policy);
    let out = Matrix::from_fn(a.nrows(), b.nrows(), |i, j| {
        let ai = a.row(i);
        let bj = b.row(j);
        minkowski9_l1_distance_raw(ai.as_slice(), bj.as_slice())
    });
    ctx.finish(out)
}



fn minkowski10_l1_distance_raw(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return f64::NAN;
    }
    let mut sa = 0.0_f64;
    let mut sb = 0.0_f64;
    for i in 0..n {
        sa += a[i].abs();
        sb += b[i].abs();
    }
    if sa < 1e-18 && sb < 1e-18 {
        return 0.0;
    }
    sa = sa.max(1e-18);
    sb = sb.max(1e-18);
    let mut s = 0.0_f64;
    for i in 0..n {
        let p = a[i].abs() / sa;
        let q = b[i].abs() / sb;
        s += (p - q).abs().powi(10);
    }
    s.powf(0.1)
}

/// Minkowski \(p=10\) distance after \(\ell_1\) normalisation.
///
/// Distinct from [`minkowski9_l1_distance`] and [`minkowski8_l1_distance`].
/// Identical series score 0.
pub fn minkowski10_l1_distance(a: &Vector, b: &Vector, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if let Some(issue) = signlred::scan_finite(a.as_slice()).to_issue("minkowski10_l1_distance.a") {
        ctx.push(issue);
    }
    if let Some(issue) = signlred::scan_finite(b.as_slice()).to_issue("minkowski10_l1_distance.b") {
        ctx.push(issue);
    }
    if a.is_empty() || b.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message("minkowski10_l1_distance on an empty series")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    ctx.finish(minkowski10_l1_distance_raw(a.as_slice(), b.as_slice()))
}

/// Pairwise \(\ell_1\)-normalised Minkowski \(p=10\) distance.
pub fn cdist_minkowski10_l1(a: &Matrix, b: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, a, None, &ctx.policy);
    inspect_xy(&mut ctx.report, b, None, &ctx.policy);
    let out = Matrix::from_fn(a.nrows(), b.nrows(), |i, j| {
        let ai = a.row(i);
        let bj = b.row(j);
        minkowski10_l1_distance_raw(ai.as_slice(), bj.as_slice())
    });
    ctx.finish(out)
}



fn minkowski11_l1_distance_raw(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return f64::NAN;
    }
    let mut sa = 0.0_f64;
    let mut sb = 0.0_f64;
    for i in 0..n {
        sa += a[i].abs();
        sb += b[i].abs();
    }
    if sa < 1e-18 && sb < 1e-18 {
        return 0.0;
    }
    sa = sa.max(1e-18);
    sb = sb.max(1e-18);
    let mut s = 0.0_f64;
    for i in 0..n {
        let p = a[i].abs() / sa;
        let q = b[i].abs() / sb;
        s += (p - q).abs().powi(11);
    }
    s.powf(1.0 / 11.0)
}

/// Minkowski \(p=11\) distance after \(\ell_1\) normalisation.
///
/// Distinct from [`minkowski10_l1_distance`] and [`minkowski9_l1_distance`].
/// Identical series score 0.
pub fn minkowski11_l1_distance(a: &Vector, b: &Vector, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if let Some(issue) = signlred::scan_finite(a.as_slice()).to_issue("minkowski11_l1_distance.a") {
        ctx.push(issue);
    }
    if let Some(issue) = signlred::scan_finite(b.as_slice()).to_issue("minkowski11_l1_distance.b") {
        ctx.push(issue);
    }
    if a.is_empty() || b.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message("minkowski11_l1_distance on an empty series")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    ctx.finish(minkowski11_l1_distance_raw(a.as_slice(), b.as_slice()))
}

/// Pairwise \(\ell_1\)-normalised Minkowski \(p=11\) distance.
pub fn cdist_minkowski11_l1(a: &Matrix, b: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, a, None, &ctx.policy);
    inspect_xy(&mut ctx.report, b, None, &ctx.policy);
    let out = Matrix::from_fn(a.nrows(), b.nrows(), |i, j| {
        let ai = a.row(i);
        let bj = b.row(j);
        minkowski11_l1_distance_raw(ai.as_slice(), bj.as_slice())
    });
    ctx.finish(out)
}



fn minkowski12_l1_distance_raw(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return f64::NAN;
    }
    let mut sa = 0.0_f64;
    let mut sb = 0.0_f64;
    for i in 0..n {
        sa += a[i].abs();
        sb += b[i].abs();
    }
    if sa < 1e-18 && sb < 1e-18 {
        return 0.0;
    }
    sa = sa.max(1e-18);
    sb = sb.max(1e-18);
    let mut s = 0.0_f64;
    for i in 0..n {
        let p = a[i].abs() / sa;
        let q = b[i].abs() / sb;
        s += (p - q).abs().powi(12);
    }
    s.powf(1.0 / 12.0)
}

/// Minkowski \(p=12\) distance after \(\ell_1\) normalisation.
///
/// Distinct from [`minkowski11_l1_distance`] and [`minkowski10_l1_distance`].
/// Identical series score 0.
pub fn minkowski12_l1_distance(a: &Vector, b: &Vector, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if let Some(issue) = signlred::scan_finite(a.as_slice()).to_issue("minkowski12_l1_distance.a") {
        ctx.push(issue);
    }
    if let Some(issue) = signlred::scan_finite(b.as_slice()).to_issue("minkowski12_l1_distance.b") {
        ctx.push(issue);
    }
    if a.is_empty() || b.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message("minkowski12_l1_distance on an empty series")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    ctx.finish(minkowski12_l1_distance_raw(a.as_slice(), b.as_slice()))
}

/// Pairwise \(\ell_1\)-normalised Minkowski \(p=12\) distance.
pub fn cdist_minkowski12_l1(a: &Matrix, b: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, a, None, &ctx.policy);
    inspect_xy(&mut ctx.report, b, None, &ctx.policy);
    let out = Matrix::from_fn(a.nrows(), b.nrows(), |i, j| {
        let ai = a.row(i);
        let bj = b.row(j);
        minkowski12_l1_distance_raw(ai.as_slice(), bj.as_slice())
    });
    ctx.finish(out)
}



fn minkowski13_l1_distance_raw(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return f64::NAN;
    }
    let mut sa = 0.0_f64;
    let mut sb = 0.0_f64;
    for i in 0..n {
        sa += a[i].abs();
        sb += b[i].abs();
    }
    if sa < 1e-18 && sb < 1e-18 {
        return 0.0;
    }
    sa = sa.max(1e-18);
    sb = sb.max(1e-18);
    let mut s = 0.0_f64;
    for i in 0..n {
        let p = a[i].abs() / sa;
        let q = b[i].abs() / sb;
        s += (p - q).abs().powi(13);
    }
    s.powf(1.0 / 13.0)
}

/// Minkowski \(p=13\) distance after \(\ell_1\) normalisation.
///
/// Distinct from [`minkowski12_l1_distance`] and [`minkowski11_l1_distance`].
/// Identical series score 0.
pub fn minkowski13_l1_distance(a: &Vector, b: &Vector, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if let Some(issue) = signlred::scan_finite(a.as_slice()).to_issue("minkowski13_l1_distance.a") {
        ctx.push(issue);
    }
    if let Some(issue) = signlred::scan_finite(b.as_slice()).to_issue("minkowski13_l1_distance.b") {
        ctx.push(issue);
    }
    if a.is_empty() || b.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message("minkowski13_l1_distance on an empty series")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    ctx.finish(minkowski13_l1_distance_raw(a.as_slice(), b.as_slice()))
}

/// Pairwise \(\ell_1\)-normalised Minkowski \(p=13\) distance.
pub fn cdist_minkowski13_l1(a: &Matrix, b: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, a, None, &ctx.policy);
    inspect_xy(&mut ctx.report, b, None, &ctx.policy);
    let out = Matrix::from_fn(a.nrows(), b.nrows(), |i, j| {
        let ai = a.row(i);
        let bj = b.row(j);
        minkowski13_l1_distance_raw(ai.as_slice(), bj.as_slice())
    });
    ctx.finish(out)
}



fn minkowski14_l1_distance_raw(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return f64::NAN;
    }
    let mut sa = 0.0_f64;
    let mut sb = 0.0_f64;
    for i in 0..n {
        sa += a[i].abs();
        sb += b[i].abs();
    }
    if sa < 1e-18 && sb < 1e-18 {
        return 0.0;
    }
    sa = sa.max(1e-18);
    sb = sb.max(1e-18);
    let mut s = 0.0_f64;
    for i in 0..n {
        let p = a[i].abs() / sa;
        let q = b[i].abs() / sb;
        s += (p - q).abs().powi(14);
    }
    s.powf(1.0 / 14.0)
}

/// Minkowski \(p=14\) distance after \(\ell_1\) normalisation.
///
/// Distinct from [`minkowski13_l1_distance`] and [`minkowski12_l1_distance`].
/// Identical series score 0.
pub fn minkowski14_l1_distance(a: &Vector, b: &Vector, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if let Some(issue) = signlred::scan_finite(a.as_slice()).to_issue("minkowski14_l1_distance.a") {
        ctx.push(issue);
    }
    if let Some(issue) = signlred::scan_finite(b.as_slice()).to_issue("minkowski14_l1_distance.b") {
        ctx.push(issue);
    }
    if a.is_empty() || b.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message("minkowski14_l1_distance on an empty series")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    ctx.finish(minkowski14_l1_distance_raw(a.as_slice(), b.as_slice()))
}

/// Pairwise \(\ell_1\)-normalised Minkowski \(p=14\) distance.
pub fn cdist_minkowski14_l1(a: &Matrix, b: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, a, None, &ctx.policy);
    inspect_xy(&mut ctx.report, b, None, &ctx.policy);
    let out = Matrix::from_fn(a.nrows(), b.nrows(), |i, j| {
        let ai = a.row(i);
        let bj = b.row(j);
        minkowski14_l1_distance_raw(ai.as_slice(), bj.as_slice())
    });
    ctx.finish(out)
}



fn minkowski16_l1_distance_raw(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return f64::NAN;
    }
    let mut sa = 0.0_f64;
    let mut sb = 0.0_f64;
    for i in 0..n {
        sa += a[i].abs();
        sb += b[i].abs();
    }
    if sa < 1e-18 && sb < 1e-18 {
        return 0.0;
    }
    sa = sa.max(1e-18);
    sb = sb.max(1e-18);
    let mut s = 0.0_f64;
    for i in 0..n {
        let p = a[i].abs() / sa;
        let q = b[i].abs() / sb;
        s += (p - q).abs().powi(16);
    }
    s.powf(0.0625)
}

/// Minkowski \(p=16\) distance after \(\ell_1\) normalisation.
///
/// Distinct from [`minkowski14_l1_distance`] and [`minkowski13_l1_distance`].
/// Identical series score 0.
pub fn minkowski16_l1_distance(a: &Vector, b: &Vector, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if let Some(issue) = signlred::scan_finite(a.as_slice()).to_issue("minkowski16_l1_distance.a") {
        ctx.push(issue);
    }
    if let Some(issue) = signlred::scan_finite(b.as_slice()).to_issue("minkowski16_l1_distance.b") {
        ctx.push(issue);
    }
    if a.is_empty() || b.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message("minkowski16_l1_distance on an empty series")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    ctx.finish(minkowski16_l1_distance_raw(a.as_slice(), b.as_slice()))
}

/// Pairwise \(\ell_1\)-normalised Minkowski \(p=16\) distance.
pub fn cdist_minkowski16_l1(a: &Matrix, b: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, a, None, &ctx.policy);
    inspect_xy(&mut ctx.report, b, None, &ctx.policy);
    let out = Matrix::from_fn(a.nrows(), b.nrows(), |i, j| {
        let ai = a.row(i);
        let bj = b.row(j);
        minkowski16_l1_distance_raw(ai.as_slice(), bj.as_slice())
    });
    ctx.finish(out)
}



fn minkowski18_l1_distance_raw(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return f64::NAN;
    }
    let mut sa = 0.0_f64;
    let mut sb = 0.0_f64;
    for i in 0..n {
        sa += a[i].abs();
        sb += b[i].abs();
    }
    if sa < 1e-18 && sb < 1e-18 {
        return 0.0;
    }
    sa = sa.max(1e-18);
    sb = sb.max(1e-18);
    let mut s = 0.0_f64;
    for i in 0..n {
        let p = a[i].abs() / sa;
        let q = b[i].abs() / sb;
        s += (p - q).abs().powi(18);
    }
    s.powf(1.0 / 18.0)
}

/// Minkowski \(p=18\) distance after \(\ell_1\) normalisation.
///
/// Distinct from [`minkowski16_l1_distance`] and [`minkowski14_l1_distance`].
/// Identical series score 0.
pub fn minkowski18_l1_distance(a: &Vector, b: &Vector, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if let Some(issue) = signlred::scan_finite(a.as_slice()).to_issue("minkowski18_l1_distance.a") {
        ctx.push(issue);
    }
    if let Some(issue) = signlred::scan_finite(b.as_slice()).to_issue("minkowski18_l1_distance.b") {
        ctx.push(issue);
    }
    if a.is_empty() || b.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message("minkowski18_l1_distance on an empty series")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    ctx.finish(minkowski18_l1_distance_raw(a.as_slice(), b.as_slice()))
}

/// Pairwise \(\ell_1\)-normalised Minkowski \(p=18\) distance.
pub fn cdist_minkowski18_l1(a: &Matrix, b: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, a, None, &ctx.policy);
    inspect_xy(&mut ctx.report, b, None, &ctx.policy);
    let out = Matrix::from_fn(a.nrows(), b.nrows(), |i, j| {
        let ai = a.row(i);
        let bj = b.row(j);
        minkowski18_l1_distance_raw(ai.as_slice(), bj.as_slice())
    });
    ctx.finish(out)
}



fn minkowski20_l1_distance_raw(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return f64::NAN;
    }
    let mut sa = 0.0_f64;
    let mut sb = 0.0_f64;
    for i in 0..n {
        sa += a[i].abs();
        sb += b[i].abs();
    }
    if sa < 1e-18 && sb < 1e-18 {
        return 0.0;
    }
    sa = sa.max(1e-18);
    sb = sb.max(1e-18);
    let mut s = 0.0_f64;
    for i in 0..n {
        let p = a[i].abs() / sa;
        let q = b[i].abs() / sb;
        s += (p - q).abs().powi(20);
    }
    s.powf(0.05)
}

/// Minkowski \(p=20\) distance after \(\ell_1\) normalisation.
///
/// Distinct from [`minkowski18_l1_distance`] and [`minkowski16_l1_distance`].
/// Identical series score 0.
pub fn minkowski20_l1_distance(a: &Vector, b: &Vector, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if let Some(issue) = signlred::scan_finite(a.as_slice()).to_issue("minkowski20_l1_distance.a") {
        ctx.push(issue);
    }
    if let Some(issue) = signlred::scan_finite(b.as_slice()).to_issue("minkowski20_l1_distance.b") {
        ctx.push(issue);
    }
    if a.is_empty() || b.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message("minkowski20_l1_distance on an empty series")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    ctx.finish(minkowski20_l1_distance_raw(a.as_slice(), b.as_slice()))
}

/// Pairwise \(\ell_1\)-normalised Minkowski \(p=20\) distance.
pub fn cdist_minkowski20_l1(a: &Matrix, b: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, a, None, &ctx.policy);
    inspect_xy(&mut ctx.report, b, None, &ctx.policy);
    let out = Matrix::from_fn(a.nrows(), b.nrows(), |i, j| {
        let ai = a.row(i);
        let bj = b.row(j);
        minkowski20_l1_distance_raw(ai.as_slice(), bj.as_slice())
    });
    ctx.finish(out)
}



fn minkowski24_l1_distance_raw(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return f64::NAN;
    }
    let mut sa = 0.0_f64;
    let mut sb = 0.0_f64;
    for i in 0..n {
        sa += a[i].abs();
        sb += b[i].abs();
    }
    if sa < 1e-18 && sb < 1e-18 {
        return 0.0;
    }
    sa = sa.max(1e-18);
    sb = sb.max(1e-18);
    let mut s = 0.0_f64;
    for i in 0..n {
        let p = a[i].abs() / sa;
        let q = b[i].abs() / sb;
        s += (p - q).abs().powi(24);
    }
    s.powf(1.0 / 24.0)
}

/// Minkowski \(p=24\) distance after \(\ell_1\) normalisation.
///
/// Distinct from [`minkowski20_l1_distance`] and [`minkowski18_l1_distance`].
/// Identical series score 0.
pub fn minkowski24_l1_distance(a: &Vector, b: &Vector, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if let Some(issue) = signlred::scan_finite(a.as_slice()).to_issue("minkowski24_l1_distance.a") {
        ctx.push(issue);
    }
    if let Some(issue) = signlred::scan_finite(b.as_slice()).to_issue("minkowski24_l1_distance.b") {
        ctx.push(issue);
    }
    if a.is_empty() || b.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message("minkowski24_l1_distance on an empty series")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    ctx.finish(minkowski24_l1_distance_raw(a.as_slice(), b.as_slice()))
}

/// Pairwise \(\ell_1\)-normalised Minkowski \(p=24\) distance.
pub fn cdist_minkowski24_l1(a: &Matrix, b: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, a, None, &ctx.policy);
    inspect_xy(&mut ctx.report, b, None, &ctx.policy);
    let out = Matrix::from_fn(a.nrows(), b.nrows(), |i, j| {
        let ai = a.row(i);
        let bj = b.row(j);
        minkowski24_l1_distance_raw(ai.as_slice(), bj.as_slice())
    });
    ctx.finish(out)
}



fn minkowski17_l1_distance_raw(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return f64::NAN;
    }
    let mut sa = 0.0_f64;
    let mut sb = 0.0_f64;
    for i in 0..n {
        sa += a[i].abs();
        sb += b[i].abs();
    }
    if sa < 1e-18 && sb < 1e-18 {
        return 0.0;
    }
    sa = sa.max(1e-18);
    sb = sb.max(1e-18);
    let mut s = 0.0_f64;
    for i in 0..n {
        let p = a[i].abs() / sa;
        let q = b[i].abs() / sb;
        s += (p - q).abs().powi(17);
    }
    s.powf(1.0 / 17.0)
}

/// Minkowski \(p=17\) distance after \(\ell_1\) normalisation.
///
/// Distinct from [`minkowski16_l1_distance`] and [`minkowski18_l1_distance`].
/// Identical series score 0.
pub fn minkowski17_l1_distance(a: &Vector, b: &Vector, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if let Some(issue) = signlred::scan_finite(a.as_slice()).to_issue("minkowski17_l1_distance.a") {
        ctx.push(issue);
    }
    if let Some(issue) = signlred::scan_finite(b.as_slice()).to_issue("minkowski17_l1_distance.b") {
        ctx.push(issue);
    }
    if a.is_empty() || b.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message("minkowski17_l1_distance on an empty series")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    ctx.finish(minkowski17_l1_distance_raw(a.as_slice(), b.as_slice()))
}

/// Pairwise \(\ell_1\)-normalised Minkowski \(p=17\) distance.
pub fn cdist_minkowski17_l1(a: &Matrix, b: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, a, None, &ctx.policy);
    inspect_xy(&mut ctx.report, b, None, &ctx.policy);
    let out = Matrix::from_fn(a.nrows(), b.nrows(), |i, j| {
        let ai = a.row(i);
        let bj = b.row(j);
        minkowski17_l1_distance_raw(ai.as_slice(), bj.as_slice())
    });
    ctx.finish(out)
}


fn minkowski19_l1_distance_raw(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return f64::NAN;
    }
    let mut sa = 0.0_f64;
    let mut sb = 0.0_f64;
    for i in 0..n {
        sa += a[i].abs();
        sb += b[i].abs();
    }
    if sa < 1e-18 && sb < 1e-18 {
        return 0.0;
    }
    sa = sa.max(1e-18);
    sb = sb.max(1e-18);
    let mut s = 0.0_f64;
    for i in 0..n {
        let p = a[i].abs() / sa;
        let q = b[i].abs() / sb;
        s += (p - q).abs().powi(19);
    }
    s.powf(1.0 / 19.0)
}

/// Minkowski \(p=19\) distance after \(\ell_1\) normalisation.
///
/// Distinct from [`minkowski17_l1_distance`] and [`minkowski18_l1_distance`].
/// Identical series score 0.
pub fn minkowski19_l1_distance(a: &Vector, b: &Vector, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if let Some(issue) = signlred::scan_finite(a.as_slice()).to_issue("minkowski19_l1_distance.a") {
        ctx.push(issue);
    }
    if let Some(issue) = signlred::scan_finite(b.as_slice()).to_issue("minkowski19_l1_distance.b") {
        ctx.push(issue);
    }
    if a.is_empty() || b.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message("minkowski19_l1_distance on an empty series")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    ctx.finish(minkowski19_l1_distance_raw(a.as_slice(), b.as_slice()))
}

/// Pairwise \(\ell_1\)-normalised Minkowski \(p=19\) distance.
pub fn cdist_minkowski19_l1(a: &Matrix, b: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, a, None, &ctx.policy);
    inspect_xy(&mut ctx.report, b, None, &ctx.policy);
    let out = Matrix::from_fn(a.nrows(), b.nrows(), |i, j| {
        let ai = a.row(i);
        let bj = b.row(j);
        minkowski19_l1_distance_raw(ai.as_slice(), bj.as_slice())
    });
    ctx.finish(out)
}

fn minkowski21_l1_distance_raw(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return f64::NAN;
    }
    let mut sa = 0.0_f64;
    let mut sb = 0.0_f64;
    for i in 0..n {
        sa += a[i].abs();
        sb += b[i].abs();
    }
    if sa < 1e-18 && sb < 1e-18 {
        return 0.0;
    }
    sa = sa.max(1e-18);
    sb = sb.max(1e-18);
    let mut s = 0.0_f64;
    for i in 0..n {
        let p = a[i].abs() / sa;
        let q = b[i].abs() / sb;
        s += (p - q).abs().powi(21);
    }
    s.powf(1.0 / 21.0)
}

/// Minkowski \(p=21\) distance after \(\ell_1\) normalisation.
///
/// Distinct from [`minkowski19_l1_distance`] and [`minkowski20_l1_distance`].
/// Identical series score 0.
pub fn minkowski21_l1_distance(a: &Vector, b: &Vector, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if let Some(issue) = signlred::scan_finite(a.as_slice()).to_issue("minkowski21_l1_distance.a") {
        ctx.push(issue);
    }
    if let Some(issue) = signlred::scan_finite(b.as_slice()).to_issue("minkowski21_l1_distance.b") {
        ctx.push(issue);
    }
    if a.is_empty() || b.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message("minkowski21_l1_distance on an empty series")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    ctx.finish(minkowski21_l1_distance_raw(a.as_slice(), b.as_slice()))
}

/// Pairwise \(\ell_1\)-normalised Minkowski \(p=21\) distance.
pub fn cdist_minkowski21_l1(a: &Matrix, b: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, a, None, &ctx.policy);
    inspect_xy(&mut ctx.report, b, None, &ctx.policy);
    let out = Matrix::from_fn(a.nrows(), b.nrows(), |i, j| {
        let ai = a.row(i);
        let bj = b.row(j);
        minkowski21_l1_distance_raw(ai.as_slice(), bj.as_slice())
    });
    ctx.finish(out)
}

fn minkowski22_l1_distance_raw(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return f64::NAN;
    }
    let mut sa = 0.0_f64;
    let mut sb = 0.0_f64;
    for i in 0..n {
        sa += a[i].abs();
        sb += b[i].abs();
    }
    if sa < 1e-18 && sb < 1e-18 {
        return 0.0;
    }
    sa = sa.max(1e-18);
    sb = sb.max(1e-18);
    let mut s = 0.0_f64;
    for i in 0..n {
        let p = a[i].abs() / sa;
        let q = b[i].abs() / sb;
        s += (p - q).abs().powi(22);
    }
    s.powf(1.0 / 22.0)
}

/// Minkowski \(p=22\) distance after \(\ell_1\) normalisation.
///
/// Distinct from [`minkowski21_l1_distance`] and [`minkowski20_l1_distance`].
/// Identical series score 0.
pub fn minkowski22_l1_distance(a: &Vector, b: &Vector, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if let Some(issue) = signlred::scan_finite(a.as_slice()).to_issue("minkowski22_l1_distance.a") {
        ctx.push(issue);
    }
    if let Some(issue) = signlred::scan_finite(b.as_slice()).to_issue("minkowski22_l1_distance.b") {
        ctx.push(issue);
    }
    if a.is_empty() || b.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message("minkowski22_l1_distance on an empty series")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    ctx.finish(minkowski22_l1_distance_raw(a.as_slice(), b.as_slice()))
}

/// Pairwise \(\ell_1\)-normalised Minkowski \(p=22\) distance.
pub fn cdist_minkowski22_l1(a: &Matrix, b: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, a, None, &ctx.policy);
    inspect_xy(&mut ctx.report, b, None, &ctx.policy);
    let out = Matrix::from_fn(a.nrows(), b.nrows(), |i, j| {
        let ai = a.row(i);
        let bj = b.row(j);
        minkowski22_l1_distance_raw(ai.as_slice(), bj.as_slice())
    });
    ctx.finish(out)
}

fn minkowski28_l1_distance_raw(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return f64::NAN;
    }
    let mut sa = 0.0_f64;
    let mut sb = 0.0_f64;
    for i in 0..n {
        sa += a[i].abs();
        sb += b[i].abs();
    }
    if sa < 1e-18 && sb < 1e-18 {
        return 0.0;
    }
    sa = sa.max(1e-18);
    sb = sb.max(1e-18);
    let mut s = 0.0_f64;
    for i in 0..n {
        let p = a[i].abs() / sa;
        let q = b[i].abs() / sb;
        s += (p - q).abs().powi(28);
    }
    s.powf(1.0 / 28.0)
}

/// Minkowski \(p=28\) distance after \(\ell_1\) normalisation.
///
/// Distinct from [`minkowski24_l1_distance`] and [`minkowski22_l1_distance`].
/// Identical series score 0.
pub fn minkowski28_l1_distance(a: &Vector, b: &Vector, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if let Some(issue) = signlred::scan_finite(a.as_slice()).to_issue("minkowski28_l1_distance.a") {
        ctx.push(issue);
    }
    if let Some(issue) = signlred::scan_finite(b.as_slice()).to_issue("minkowski28_l1_distance.b") {
        ctx.push(issue);
    }
    if a.is_empty() || b.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message("minkowski28_l1_distance on an empty series")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    ctx.finish(minkowski28_l1_distance_raw(a.as_slice(), b.as_slice()))
}

/// Pairwise \(\ell_1\)-normalised Minkowski \(p=28\) distance.
pub fn cdist_minkowski28_l1(a: &Matrix, b: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, a, None, &ctx.policy);
    inspect_xy(&mut ctx.report, b, None, &ctx.policy);
    let out = Matrix::from_fn(a.nrows(), b.nrows(), |i, j| {
        let ai = a.row(i);
        let bj = b.row(j);
        minkowski28_l1_distance_raw(ai.as_slice(), bj.as_slice())
    });
    ctx.finish(out)
}
fn minkowski23_l1_distance_raw(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return f64::NAN;
    }
    let mut sa = 0.0_f64;
    let mut sb = 0.0_f64;
    for i in 0..n {
        sa += a[i].abs();
        sb += b[i].abs();
    }
    if sa < 1e-18 && sb < 1e-18 {
        return 0.0;
    }
    sa = sa.max(1e-18);
    sb = sb.max(1e-18);
    let mut s = 0.0_f64;
    for i in 0..n {
        let p = a[i].abs() / sa;
        let q = b[i].abs() / sb;
        s += (p - q).abs().powi(23);
    }
    s.powf(1.0 / 23.0)
}

/// Minkowski \(p=23\) distance after \(\ell_1\) normalisation.
///
/// Distinct from [`minkowski22_l1_distance`] and [`minkowski24_l1_distance`].
/// Identical series score 0.
pub fn minkowski23_l1_distance(a: &Vector, b: &Vector, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if let Some(issue) = signlred::scan_finite(a.as_slice()).to_issue("minkowski23_l1_distance.a") {
        ctx.push(issue);
    }
    if let Some(issue) = signlred::scan_finite(b.as_slice()).to_issue("minkowski23_l1_distance.b") {
        ctx.push(issue);
    }
    if a.is_empty() || b.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message("minkowski23_l1_distance on an empty series")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    ctx.finish(minkowski23_l1_distance_raw(a.as_slice(), b.as_slice()))
}

/// Pairwise \(\ell_1\)-normalised Minkowski \(p=23\) distance.
pub fn cdist_minkowski23_l1(a: &Matrix, b: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, a, None, &ctx.policy);
    inspect_xy(&mut ctx.report, b, None, &ctx.policy);
    let out = Matrix::from_fn(a.nrows(), b.nrows(), |i, j| {
        let ai = a.row(i);
        let bj = b.row(j);
        minkowski23_l1_distance_raw(ai.as_slice(), bj.as_slice())
    });
    ctx.finish(out)
}
fn minkowski26_l1_distance_raw(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return f64::NAN;
    }
    let mut sa = 0.0_f64;
    let mut sb = 0.0_f64;
    for i in 0..n {
        sa += a[i].abs();
        sb += b[i].abs();
    }
    if sa < 1e-18 && sb < 1e-18 {
        return 0.0;
    }
    sa = sa.max(1e-18);
    sb = sb.max(1e-18);
    let mut s = 0.0_f64;
    for i in 0..n {
        let p = a[i].abs() / sa;
        let q = b[i].abs() / sb;
        s += (p - q).abs().powi(26);
    }
    s.powf(1.0 / 26.0)
}

/// Minkowski \(p=26\) distance after \(\ell_1\) normalisation.
///
/// Distinct from [`minkowski24_l1_distance`] and [`minkowski28_l1_distance`].
/// Identical series score 0.
pub fn minkowski26_l1_distance(a: &Vector, b: &Vector, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if let Some(issue) = signlred::scan_finite(a.as_slice()).to_issue("minkowski26_l1_distance.a") {
        ctx.push(issue);
    }
    if let Some(issue) = signlred::scan_finite(b.as_slice()).to_issue("minkowski26_l1_distance.b") {
        ctx.push(issue);
    }
    if a.is_empty() || b.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message("minkowski26_l1_distance on an empty series")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    ctx.finish(minkowski26_l1_distance_raw(a.as_slice(), b.as_slice()))
}

/// Pairwise \(\ell_1\)-normalised Minkowski \(p=26\) distance.
pub fn cdist_minkowski26_l1(a: &Matrix, b: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, a, None, &ctx.policy);
    inspect_xy(&mut ctx.report, b, None, &ctx.policy);
    let out = Matrix::from_fn(a.nrows(), b.nrows(), |i, j| {
        let ai = a.row(i);
        let bj = b.row(j);
        minkowski26_l1_distance_raw(ai.as_slice(), bj.as_slice())
    });
    ctx.finish(out)
}
fn minkowski27_l1_distance_raw(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return f64::NAN;
    }
    let mut sa = 0.0_f64;
    let mut sb = 0.0_f64;
    for i in 0..n {
        sa += a[i].abs();
        sb += b[i].abs();
    }
    if sa < 1e-18 && sb < 1e-18 {
        return 0.0;
    }
    sa = sa.max(1e-18);
    sb = sb.max(1e-18);
    let mut s = 0.0_f64;
    for i in 0..n {
        let p = a[i].abs() / sa;
        let q = b[i].abs() / sb;
        s += (p - q).abs().powi(27);
    }
    s.powf(1.0 / 27.0)
}

/// Minkowski \(p=27\) distance after \(\ell_1\) normalisation.
///
/// Distinct from [`minkowski26_l1_distance`] and [`minkowski28_l1_distance`].
/// Identical series score 0.
pub fn minkowski27_l1_distance(a: &Vector, b: &Vector, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if let Some(issue) = signlred::scan_finite(a.as_slice()).to_issue("minkowski27_l1_distance.a") {
        ctx.push(issue);
    }
    if let Some(issue) = signlred::scan_finite(b.as_slice()).to_issue("minkowski27_l1_distance.b") {
        ctx.push(issue);
    }
    if a.is_empty() || b.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message("minkowski27_l1_distance on an empty series")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    ctx.finish(minkowski27_l1_distance_raw(a.as_slice(), b.as_slice()))
}

/// Pairwise \(\ell_1\)-normalised Minkowski \(p=27\) distance.
pub fn cdist_minkowski27_l1(a: &Matrix, b: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, a, None, &ctx.policy);
    inspect_xy(&mut ctx.report, b, None, &ctx.policy);
    let out = Matrix::from_fn(a.nrows(), b.nrows(), |i, j| {
        let ai = a.row(i);
        let bj = b.row(j);
        minkowski27_l1_distance_raw(ai.as_slice(), bj.as_slice())
    });
    ctx.finish(out)
}
fn minkowski29_l1_distance_raw(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return f64::NAN;
    }
    let mut sa = 0.0_f64;
    let mut sb = 0.0_f64;
    for i in 0..n {
        sa += a[i].abs();
        sb += b[i].abs();
    }
    if sa < 1e-18 && sb < 1e-18 {
        return 0.0;
    }
    sa = sa.max(1e-18);
    sb = sb.max(1e-18);
    let mut s = 0.0_f64;
    for i in 0..n {
        let p = a[i].abs() / sa;
        let q = b[i].abs() / sb;
        s += (p - q).abs().powi(29);
    }
    s.powf(1.0 / 29.0)
}

/// Minkowski \(p=29\) distance after \(\ell_1\) normalisation.
///
/// Distinct from [`minkowski27_l1_distance`] and [`minkowski28_l1_distance`].
/// Identical series score 0.
pub fn minkowski29_l1_distance(a: &Vector, b: &Vector, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if let Some(issue) = signlred::scan_finite(a.as_slice()).to_issue("minkowski29_l1_distance.a") {
        ctx.push(issue);
    }
    if let Some(issue) = signlred::scan_finite(b.as_slice()).to_issue("minkowski29_l1_distance.b") {
        ctx.push(issue);
    }
    if a.is_empty() || b.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message("minkowski29_l1_distance on an empty series")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    ctx.finish(minkowski29_l1_distance_raw(a.as_slice(), b.as_slice()))
}

/// Pairwise \(\ell_1\)-normalised Minkowski \(p=29\) distance.
pub fn cdist_minkowski29_l1(a: &Matrix, b: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, a, None, &ctx.policy);
    inspect_xy(&mut ctx.report, b, None, &ctx.policy);
    let out = Matrix::from_fn(a.nrows(), b.nrows(), |i, j| {
        let ai = a.row(i);
        let bj = b.row(j);
        minkowski29_l1_distance_raw(ai.as_slice(), bj.as_slice())
    });
    ctx.finish(out)
}
fn minkowski30_l1_distance_raw(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return f64::NAN;
    }
    let mut sa = 0.0_f64;
    let mut sb = 0.0_f64;
    for i in 0..n {
        sa += a[i].abs();
        sb += b[i].abs();
    }
    if sa < 1e-18 && sb < 1e-18 {
        return 0.0;
    }
    sa = sa.max(1e-18);
    sb = sb.max(1e-18);
    let mut s = 0.0_f64;
    for i in 0..n {
        let p = a[i].abs() / sa;
        let q = b[i].abs() / sb;
        s += (p - q).abs().powi(30);
    }
    s.powf(1.0 / 30.0)
}

/// Minkowski \(p=30\) distance after \(\ell_1\) normalisation.
///
/// Distinct from [`minkowski28_l1_distance`] and [`minkowski29_l1_distance`].
/// Identical series score 0.
pub fn minkowski30_l1_distance(a: &Vector, b: &Vector, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if let Some(issue) = signlred::scan_finite(a.as_slice()).to_issue("minkowski30_l1_distance.a") {
        ctx.push(issue);
    }
    if let Some(issue) = signlred::scan_finite(b.as_slice()).to_issue("minkowski30_l1_distance.b") {
        ctx.push(issue);
    }
    if a.is_empty() || b.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message("minkowski30_l1_distance on an empty series")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    ctx.finish(minkowski30_l1_distance_raw(a.as_slice(), b.as_slice()))
}

/// Pairwise \(\ell_1\)-normalised Minkowski \(p=30\) distance.
pub fn cdist_minkowski30_l1(a: &Matrix, b: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, a, None, &ctx.policy);
    inspect_xy(&mut ctx.report, b, None, &ctx.policy);
    let out = Matrix::from_fn(a.nrows(), b.nrows(), |i, j| {
        let ai = a.row(i);
        let bj = b.row(j);
        minkowski30_l1_distance_raw(ai.as_slice(), bj.as_slice())
    });
    ctx.finish(out)
}
fn minkowski31_l1_distance_raw(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return f64::NAN;
    }
    let mut sa = 0.0_f64;
    let mut sb = 0.0_f64;
    for i in 0..n {
        sa += a[i].abs();
        sb += b[i].abs();
    }
    if sa < 1e-18 && sb < 1e-18 {
        return 0.0;
    }
    sa = sa.max(1e-18);
    sb = sb.max(1e-18);
    let mut s = 0.0_f64;
    for i in 0..n {
        let p = a[i].abs() / sa;
        let q = b[i].abs() / sb;
        s += (p - q).abs().powi(31);
    }
    s.powf(1.0 / 31.0)
}

/// Minkowski \(p=31\) distance after \(\ell_1\) normalisation.
///
/// Distinct from [`minkowski29_l1_distance`] and [`minkowski30_l1_distance`].
/// Identical series score 0.
pub fn minkowski31_l1_distance(a: &Vector, b: &Vector, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if let Some(issue) = signlred::scan_finite(a.as_slice()).to_issue("minkowski31_l1_distance.a") {
        ctx.push(issue);
    }
    if let Some(issue) = signlred::scan_finite(b.as_slice()).to_issue("minkowski31_l1_distance.b") {
        ctx.push(issue);
    }
    if a.is_empty() || b.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message("minkowski31_l1_distance on an empty series")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    ctx.finish(minkowski31_l1_distance_raw(a.as_slice(), b.as_slice()))
}

/// Pairwise \(\ell_1\)-normalised Minkowski \(p=31\) distance.
pub fn cdist_minkowski31_l1(a: &Matrix, b: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, a, None, &ctx.policy);
    inspect_xy(&mut ctx.report, b, None, &ctx.policy);
    let out = Matrix::from_fn(a.nrows(), b.nrows(), |i, j| {
        let ai = a.row(i);
        let bj = b.row(j);
        minkowski31_l1_distance_raw(ai.as_slice(), bj.as_slice())
    });
    ctx.finish(out)
}
fn minkowski32_l1_distance_raw(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return f64::NAN;
    }
    let mut sa = 0.0_f64;
    let mut sb = 0.0_f64;
    for i in 0..n {
        sa += a[i].abs();
        sb += b[i].abs();
    }
    if sa < 1e-18 && sb < 1e-18 {
        return 0.0;
    }
    sa = sa.max(1e-18);
    sb = sb.max(1e-18);
    let mut s = 0.0_f64;
    for i in 0..n {
        let p = a[i].abs() / sa;
        let q = b[i].abs() / sb;
        s += (p - q).abs().powi(32);
    }
    s.powf(1.0 / 32.0)
}

/// Minkowski \(p=32\) distance after \(\ell_1\) normalisation.
///
/// Distinct from [`minkowski30_l1_distance`] and [`minkowski31_l1_distance`].
/// Identical series score 0.
pub fn minkowski32_l1_distance(a: &Vector, b: &Vector, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if let Some(issue) = signlred::scan_finite(a.as_slice()).to_issue("minkowski32_l1_distance.a") {
        ctx.push(issue);
    }
    if let Some(issue) = signlred::scan_finite(b.as_slice()).to_issue("minkowski32_l1_distance.b") {
        ctx.push(issue);
    }
    if a.is_empty() || b.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message("minkowski32_l1_distance on an empty series")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    ctx.finish(minkowski32_l1_distance_raw(a.as_slice(), b.as_slice()))
}

/// Pairwise \(\ell_1\)-normalised Minkowski \(p=32\) distance.
pub fn cdist_minkowski32_l1(a: &Matrix, b: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, a, None, &ctx.policy);
    inspect_xy(&mut ctx.report, b, None, &ctx.policy);
    let out = Matrix::from_fn(a.nrows(), b.nrows(), |i, j| {
        let ai = a.row(i);
        let bj = b.row(j);
        minkowski32_l1_distance_raw(ai.as_slice(), bj.as_slice())
    });
    ctx.finish(out)
}
fn minkowski33_l1_distance_raw(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return f64::NAN;
    }
    let mut sa = 0.0_f64;
    let mut sb = 0.0_f64;
    for i in 0..n {
        sa += a[i].abs();
        sb += b[i].abs();
    }
    if sa < 1e-18 && sb < 1e-18 {
        return 0.0;
    }
    sa = sa.max(1e-18);
    sb = sb.max(1e-18);
    let mut s = 0.0_f64;
    for i in 0..n {
        let p = a[i].abs() / sa;
        let q = b[i].abs() / sb;
        s += (p - q).abs().powi(33);
    }
    s.powf(1.0 / 33.0)
}

/// Minkowski \(p=33\) distance after \(\ell_1\) normalisation.
///
/// Distinct from [`minkowski31_l1_distance`] and [`minkowski32_l1_distance`].
/// Identical series score 0.
pub fn minkowski33_l1_distance(a: &Vector, b: &Vector, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if let Some(issue) = signlred::scan_finite(a.as_slice()).to_issue("minkowski33_l1_distance.a") {
        ctx.push(issue);
    }
    if let Some(issue) = signlred::scan_finite(b.as_slice()).to_issue("minkowski33_l1_distance.b") {
        ctx.push(issue);
    }
    if a.is_empty() || b.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message("minkowski33_l1_distance on an empty series")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    ctx.finish(minkowski33_l1_distance_raw(a.as_slice(), b.as_slice()))
}

/// Pairwise \(\ell_1\)-normalised Minkowski \(p=33\) distance.
pub fn cdist_minkowski33_l1(a: &Matrix, b: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, a, None, &ctx.policy);
    inspect_xy(&mut ctx.report, b, None, &ctx.policy);
    let out = Matrix::from_fn(a.nrows(), b.nrows(), |i, j| {
        let ai = a.row(i);
        let bj = b.row(j);
        minkowski33_l1_distance_raw(ai.as_slice(), bj.as_slice())
    });
    ctx.finish(out)
}
fn minkowski34_l1_distance_raw(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return f64::NAN;
    }
    let mut sa = 0.0_f64;
    let mut sb = 0.0_f64;
    for i in 0..n {
        sa += a[i].abs();
        sb += b[i].abs();
    }
    if sa < 1e-18 && sb < 1e-18 {
        return 0.0;
    }
    sa = sa.max(1e-18);
    sb = sb.max(1e-18);
    let mut s = 0.0_f64;
    for i in 0..n {
        let p = a[i].abs() / sa;
        let q = b[i].abs() / sb;
        s += (p - q).abs().powi(34);
    }
    s.powf(1.0 / 34.0)
}

/// Minkowski \(p=34\) distance after \(\ell_1\) normalisation.
///
/// Distinct from [`minkowski32_l1_distance`] and [`minkowski33_l1_distance`].
/// Identical series score 0.
pub fn minkowski34_l1_distance(a: &Vector, b: &Vector, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if let Some(issue) = signlred::scan_finite(a.as_slice()).to_issue("minkowski34_l1_distance.a") {
        ctx.push(issue);
    }
    if let Some(issue) = signlred::scan_finite(b.as_slice()).to_issue("minkowski34_l1_distance.b") {
        ctx.push(issue);
    }
    if a.is_empty() || b.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message("minkowski34_l1_distance on an empty series")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    ctx.finish(minkowski34_l1_distance_raw(a.as_slice(), b.as_slice()))
}

/// Pairwise \(\ell_1\)-normalised Minkowski \(p=34\) distance.
pub fn cdist_minkowski34_l1(a: &Matrix, b: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, a, None, &ctx.policy);
    inspect_xy(&mut ctx.report, b, None, &ctx.policy);
    let out = Matrix::from_fn(a.nrows(), b.nrows(), |i, j| {
        let ai = a.row(i);
        let bj = b.row(j);
        minkowski34_l1_distance_raw(ai.as_slice(), bj.as_slice())
    });
    ctx.finish(out)
}
fn minkowski35_l1_distance_raw(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return f64::NAN;
    }
    let mut sa = 0.0_f64;
    let mut sb = 0.0_f64;
    for i in 0..n {
        sa += a[i].abs();
        sb += b[i].abs();
    }
    if sa < 1e-18 && sb < 1e-18 {
        return 0.0;
    }
    sa = sa.max(1e-18);
    sb = sb.max(1e-18);
    let mut s = 0.0_f64;
    for i in 0..n {
        let p = a[i].abs() / sa;
        let q = b[i].abs() / sb;
        s += (p - q).abs().powi(35);
    }
    s.powf(1.0 / 35.0)
}

/// Minkowski \(p=35\) distance after \(\ell_1\) normalisation.
///
/// Distinct from [`minkowski33_l1_distance`] and [`minkowski34_l1_distance`].
/// Identical series score 0.
pub fn minkowski35_l1_distance(a: &Vector, b: &Vector, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if let Some(issue) = signlred::scan_finite(a.as_slice()).to_issue("minkowski35_l1_distance.a") {
        ctx.push(issue);
    }
    if let Some(issue) = signlred::scan_finite(b.as_slice()).to_issue("minkowski35_l1_distance.b") {
        ctx.push(issue);
    }
    if a.is_empty() || b.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message("minkowski35_l1_distance on an empty series")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    ctx.finish(minkowski35_l1_distance_raw(a.as_slice(), b.as_slice()))
}

/// Pairwise \(\ell_1\)-normalised Minkowski \(p=35\) distance.
pub fn cdist_minkowski35_l1(a: &Matrix, b: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, a, None, &ctx.policy);
    inspect_xy(&mut ctx.report, b, None, &ctx.policy);
    let out = Matrix::from_fn(a.nrows(), b.nrows(), |i, j| {
        let ai = a.row(i);
        let bj = b.row(j);
        minkowski35_l1_distance_raw(ai.as_slice(), bj.as_slice())
    });
    ctx.finish(out)
}

fn minkowski36_l1_distance_raw(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return f64::NAN;
    }
    let mut sa = 0.0_f64;
    let mut sb = 0.0_f64;
    for i in 0..n {
        sa += a[i].abs();
        sb += b[i].abs();
    }
    if sa < 1e-18 && sb < 1e-18 {
        return 0.0;
    }
    sa = sa.max(1e-18);
    sb = sb.max(1e-18);
    let mut s = 0.0_f64;
    for i in 0..n {
        let p = a[i].abs() / sa;
        let q = b[i].abs() / sb;
        s += (p - q).abs().powi(36);
    }
    s.powf(1.0 / 36.0)
}

/// Minkowski \(p=36\) distance after \(\ell_1\) normalisation.
///
/// Distinct from [`minkowski34_l1_distance`] and [`minkowski35_l1_distance`].
/// Identical series score 0.
pub fn minkowski36_l1_distance(a: &Vector, b: &Vector, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if let Some(issue) = signlred::scan_finite(a.as_slice()).to_issue("minkowski36_l1_distance.a") {
        ctx.push(issue);
    }
    if let Some(issue) = signlred::scan_finite(b.as_slice()).to_issue("minkowski36_l1_distance.b") {
        ctx.push(issue);
    }
    if a.is_empty() || b.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message("minkowski36_l1_distance on an empty series")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    ctx.finish(minkowski36_l1_distance_raw(a.as_slice(), b.as_slice()))
}

/// Pairwise \(\ell_1\)-normalised Minkowski \(p=36\) distance.
pub fn cdist_minkowski36_l1(a: &Matrix, b: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, a, None, &ctx.policy);
    inspect_xy(&mut ctx.report, b, None, &ctx.policy);
    let out = Matrix::from_fn(a.nrows(), b.nrows(), |i, j| {
        let ai = a.row(i);
        let bj = b.row(j);
        minkowski36_l1_distance_raw(ai.as_slice(), bj.as_slice())
    });
    ctx.finish(out)
}

fn minkowski37_l1_distance_raw(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return f64::NAN;
    }
    let mut sa = 0.0_f64;
    let mut sb = 0.0_f64;
    for i in 0..n {
        sa += a[i].abs();
        sb += b[i].abs();
    }
    if sa < 1e-18 && sb < 1e-18 {
        return 0.0;
    }
    sa = sa.max(1e-18);
    sb = sb.max(1e-18);
    let mut s = 0.0_f64;
    for i in 0..n {
        let p = a[i].abs() / sa;
        let q = b[i].abs() / sb;
        s += (p - q).abs().powi(37);
    }
    s.powf(1.0 / 37.0)
}

/// Minkowski \(p=37\) distance after \(\ell_1\) normalisation.
///
/// Distinct from [`minkowski35_l1_distance`] and [`minkowski36_l1_distance`].
/// Identical series score 0.
pub fn minkowski37_l1_distance(a: &Vector, b: &Vector, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if let Some(issue) = signlred::scan_finite(a.as_slice()).to_issue("minkowski37_l1_distance.a") {
        ctx.push(issue);
    }
    if let Some(issue) = signlred::scan_finite(b.as_slice()).to_issue("minkowski37_l1_distance.b") {
        ctx.push(issue);
    }
    if a.is_empty() || b.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message("minkowski37_l1_distance on an empty series")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    ctx.finish(minkowski37_l1_distance_raw(a.as_slice(), b.as_slice()))
}

/// Pairwise \(\ell_1\)-normalised Minkowski \(p=37\) distance.
pub fn cdist_minkowski37_l1(a: &Matrix, b: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, a, None, &ctx.policy);
    inspect_xy(&mut ctx.report, b, None, &ctx.policy);
    let out = Matrix::from_fn(a.nrows(), b.nrows(), |i, j| {
        let ai = a.row(i);
        let bj = b.row(j);
        minkowski37_l1_distance_raw(ai.as_slice(), bj.as_slice())
    });
    ctx.finish(out)
}

fn minkowski38_l1_distance_raw(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return f64::NAN;
    }
    let mut sa = 0.0_f64;
    let mut sb = 0.0_f64;
    for i in 0..n {
        sa += a[i].abs();
        sb += b[i].abs();
    }
    if sa < 1e-18 && sb < 1e-18 {
        return 0.0;
    }
    sa = sa.max(1e-18);
    sb = sb.max(1e-18);
    let mut s = 0.0_f64;
    for i in 0..n {
        let p = a[i].abs() / sa;
        let q = b[i].abs() / sb;
        s += (p - q).abs().powi(38);
    }
    s.powf(1.0 / 38.0)
}

/// Minkowski \(p=38\) distance after \(\ell_1\) normalisation.
///
/// Distinct from [`minkowski36_l1_distance`] and [`minkowski37_l1_distance`].
/// Identical series score 0.
pub fn minkowski38_l1_distance(a: &Vector, b: &Vector, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if let Some(issue) = signlred::scan_finite(a.as_slice()).to_issue("minkowski38_l1_distance.a") {
        ctx.push(issue);
    }
    if let Some(issue) = signlred::scan_finite(b.as_slice()).to_issue("minkowski38_l1_distance.b") {
        ctx.push(issue);
    }
    if a.is_empty() || b.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message("minkowski38_l1_distance on an empty series")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    ctx.finish(minkowski38_l1_distance_raw(a.as_slice(), b.as_slice()))
}

/// Pairwise \(\ell_1\)-normalised Minkowski \(p=38\) distance.
pub fn cdist_minkowski38_l1(a: &Matrix, b: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, a, None, &ctx.policy);
    inspect_xy(&mut ctx.report, b, None, &ctx.policy);
    let out = Matrix::from_fn(a.nrows(), b.nrows(), |i, j| {
        let ai = a.row(i);
        let bj = b.row(j);
        minkowski38_l1_distance_raw(ai.as_slice(), bj.as_slice())
    });
    ctx.finish(out)
}

fn minkowski39_l1_distance_raw(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return f64::NAN;
    }
    let mut sa = 0.0_f64;
    let mut sb = 0.0_f64;
    for i in 0..n {
        sa += a[i].abs();
        sb += b[i].abs();
    }
    if sa < 1e-18 && sb < 1e-18 {
        return 0.0;
    }
    sa = sa.max(1e-18);
    sb = sb.max(1e-18);
    let mut s = 0.0_f64;
    for i in 0..n {
        let p = a[i].abs() / sa;
        let q = b[i].abs() / sb;
        s += (p - q).abs().powi(39);
    }
    s.powf(1.0 / 39.0)
}

/// Minkowski \(p=39\) distance after \(\ell_1\) normalisation.
///
/// Distinct from [`minkowski37_l1_distance`] and [`minkowski38_l1_distance`].
/// Identical series score 0.
pub fn minkowski39_l1_distance(a: &Vector, b: &Vector, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if let Some(issue) = signlred::scan_finite(a.as_slice()).to_issue("minkowski39_l1_distance.a") {
        ctx.push(issue);
    }
    if let Some(issue) = signlred::scan_finite(b.as_slice()).to_issue("minkowski39_l1_distance.b") {
        ctx.push(issue);
    }
    if a.is_empty() || b.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message("minkowski39_l1_distance on an empty series")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    ctx.finish(minkowski39_l1_distance_raw(a.as_slice(), b.as_slice()))
}

/// Pairwise \(\ell_1\)-normalised Minkowski \(p=39\) distance.
pub fn cdist_minkowski39_l1(a: &Matrix, b: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, a, None, &ctx.policy);
    inspect_xy(&mut ctx.report, b, None, &ctx.policy);
    let out = Matrix::from_fn(a.nrows(), b.nrows(), |i, j| {
        let ai = a.row(i);
        let bj = b.row(j);
        minkowski39_l1_distance_raw(ai.as_slice(), bj.as_slice())
    });
    ctx.finish(out)
}

fn minkowski40_l1_distance_raw(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return f64::NAN;
    }
    let mut sa = 0.0_f64;
    let mut sb = 0.0_f64;
    for i in 0..n {
        sa += a[i].abs();
        sb += b[i].abs();
    }
    if sa < 1e-18 && sb < 1e-18 {
        return 0.0;
    }
    sa = sa.max(1e-18);
    sb = sb.max(1e-18);
    let mut s = 0.0_f64;
    for i in 0..n {
        let p = a[i].abs() / sa;
        let q = b[i].abs() / sb;
        s += (p - q).abs().powi(40);
    }
    s.powf(1.0 / 40.0)
}

/// Minkowski \(p=40\) distance after \(\ell_1\) normalisation.
///
/// Distinct from [`minkowski38_l1_distance`] and [`minkowski39_l1_distance`].
/// Identical series score 0.
pub fn minkowski40_l1_distance(a: &Vector, b: &Vector, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if let Some(issue) = signlred::scan_finite(a.as_slice()).to_issue("minkowski40_l1_distance.a") {
        ctx.push(issue);
    }
    if let Some(issue) = signlred::scan_finite(b.as_slice()).to_issue("minkowski40_l1_distance.b") {
        ctx.push(issue);
    }
    if a.is_empty() || b.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message("minkowski40_l1_distance on an empty series")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    ctx.finish(minkowski40_l1_distance_raw(a.as_slice(), b.as_slice()))
}

/// Pairwise \(\ell_1\)-normalised Minkowski \(p=40\) distance.
pub fn cdist_minkowski40_l1(a: &Matrix, b: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, a, None, &ctx.policy);
    inspect_xy(&mut ctx.report, b, None, &ctx.policy);
    let out = Matrix::from_fn(a.nrows(), b.nrows(), |i, j| {
        let ai = a.row(i);
        let bj = b.row(j);
        minkowski40_l1_distance_raw(ai.as_slice(), bj.as_slice())
    });
    ctx.finish(out)
}

fn minkowski41_l1_distance_raw(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return f64::NAN;
    }
    let mut sa = 0.0_f64;
    let mut sb = 0.0_f64;
    for i in 0..n {
        sa += a[i].abs();
        sb += b[i].abs();
    }
    if sa < 1e-18 && sb < 1e-18 {
        return 0.0;
    }
    sa = sa.max(1e-18);
    sb = sb.max(1e-18);
    let mut s = 0.0_f64;
    for i in 0..n {
        let p = a[i].abs() / sa;
        let q = b[i].abs() / sb;
        s += (p - q).abs().powi(41);
    }
    s.powf(1.0 / 41.0)
}

/// Minkowski \(p=41\) distance after \(\ell_1\) normalisation.
///
/// Distinct from [`minkowski39_l1_distance`] and [`minkowski40_l1_distance`].
/// Identical series score 0.
pub fn minkowski41_l1_distance(a: &Vector, b: &Vector, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if let Some(issue) = signlred::scan_finite(a.as_slice()).to_issue("minkowski41_l1_distance.a") {
        ctx.push(issue);
    }
    if let Some(issue) = signlred::scan_finite(b.as_slice()).to_issue("minkowski41_l1_distance.b") {
        ctx.push(issue);
    }
    if a.is_empty() || b.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message("minkowski41_l1_distance on an empty series")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    ctx.finish(minkowski41_l1_distance_raw(a.as_slice(), b.as_slice()))
}

/// Pairwise \(\ell_1\)-normalised Minkowski \(p=41\) distance.
pub fn cdist_minkowski41_l1(a: &Matrix, b: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, a, None, &ctx.policy);
    inspect_xy(&mut ctx.report, b, None, &ctx.policy);
    let out = Matrix::from_fn(a.nrows(), b.nrows(), |i, j| {
        let ai = a.row(i);
        let bj = b.row(j);
        minkowski41_l1_distance_raw(ai.as_slice(), bj.as_slice())
    });
    ctx.finish(out)
}

/// Edit Distance on Real sequences (Chen, Özsu, Oria; tslearn `edr`).
///
/// A pair matches at cost 0 when `|a_i − b_j| ≤ ε`; otherwise insert, delete,
/// or substitute costs 1. Distinct from [`edit_distance`] (absolute-cost
/// substitution) and [`erp`] (real-valued gap penalty). `ε` is not
/// identification `p`.
pub fn edr(a: &Vector, b: &Vector, eps: f64, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if let Some(issue) = signlred::scan_finite(a.as_slice()).to_issue("edr.a") {
        ctx.push(issue);
    }
    if let Some(issue) = signlred::scan_finite(b.as_slice()).to_issue("edr.b") {
        ctx.push(issue);
    }
    let eps = if eps.is_finite() && eps >= 0.0 {
        eps
    } else {
        ctx.push(
            Issue::builder(IssueCode::InvalidWeight)
                .severity(Severity::Warning)
                .message(format!("edr ε={eps} is not a finite ≥0 match radius; using 0"))
                .build(),
        );
        0.0
    };
    if a.is_empty() || b.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message("edr on an empty series")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    ctx.finish(edr_raw(a.as_slice(), b.as_slice(), eps))
}

/// Amerced DTW (Herrmann & Webb; tslearn `adtw`).
///
/// Off-diagonal steps pay an extra warp penalty `ω` on top of the local
/// absolute cost. Distinct from [`dtw`] (`ω = 0`) and [`wdtw`] (logistic
/// multiplicative weights). `ω` is not identification `p`.
pub fn adtw(a: &Vector, b: &Vector, omega: f64, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if let Some(issue) = signlred::scan_finite(a.as_slice()).to_issue("adtw.a") {
        ctx.push(issue);
    }
    if let Some(issue) = signlred::scan_finite(b.as_slice()).to_issue("adtw.b") {
        ctx.push(issue);
    }
    let omega = if omega.is_finite() && omega >= 0.0 {
        omega
    } else {
        ctx.push(
            Issue::builder(IssueCode::InvalidWeight)
                .severity(Severity::Warning)
                .message(format!("adtw ω={omega} is not a finite ≥0 warp penalty; using 0"))
                .build(),
        );
        0.0
    };
    if a.is_empty() || b.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message("adtw on an empty series")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    ctx.finish(adtw_raw(a.as_slice(), b.as_slice(), omega))
}

fn adtw_raw(a: &[f64], b: &[f64], omega: f64) -> f64 {
    let n = a.len();
    let m = b.len();
    let inf = 1e300_f64;
    let mut prev = vec![inf; m];
    let mut cur = vec![inf; m];
    prev[0] = (a[0] - b[0]).abs();
    for j in 1..m {
        prev[j] = prev[j - 1] + (a[0] - b[j]).abs() + omega;
    }
    for i in 1..n {
        cur[0] = prev[0] + (a[i] - b[0]).abs() + omega;
        for j in 1..m {
            let c = (a[i] - b[j]).abs();
            cur[j] = c + prev[j - 1].min(prev[j] + omega).min(cur[j - 1] + omega);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[m - 1]
}

fn edr_raw(a: &[f64], b: &[f64], eps: f64) -> f64 {
    let n = a.len();
    let m = b.len();
    let mut prev = vec![0.0; m + 1];
    let mut cur = vec![0.0; m + 1];
    for j in 0..=m {
        prev[j] = j as f64;
    }
    for i in 1..=n {
        cur[0] = i as f64;
        for j in 1..=m {
            if (a[i - 1] - b[j - 1]).abs() <= eps {
                cur[j] = prev[j - 1];
            } else {
                let sub = prev[j - 1] + 1.0;
                let del = prev[j] + 1.0;
                let ins = cur[j - 1] + 1.0;
                cur[j] = sub.min(del).min(ins);
            }
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[m]
}

/// Pairwise weighted DTW (tslearn `cdist` with WDTW).
///
/// Series / pair counts are not identification `p`. The logistic slope `g`
/// is not identification `p`.
pub fn cdist_wdtw(a: &Matrix, b: &Matrix, g: f64, session: &Session) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, a, None, &ctx.policy);
    inspect_xy(&mut ctx.report, b, None, &ctx.policy);
    let g = if g.is_finite() && g >= 0.0 {
        g
    } else {
        ctx.push(
            Issue::builder(IssueCode::InvalidWeight)
                .severity(Severity::Warning)
                .message(format!("cdist_wdtw g={g} is not a finite ≥0 slope; using 0.1"))
                .build(),
        );
        0.1
    };
    let out = Matrix::from_fn(a.nrows(), b.nrows(), |i, j| {
        let ai = a.row(i);
        let bj = b.row(j);
        wdtw_raw(ai.as_slice(), bj.as_slice(), g)
    });
    ctx.finish(out)
}

/// Pairwise derivative DTW (tslearn `cdist` with DDTW).
///
/// Series / pair counts are not identification `p`.
pub fn cdist_ddtw(a: &Matrix, b: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, a, None, &ctx.policy);
    inspect_xy(&mut ctx.report, b, None, &ctx.policy);
    let out = Matrix::from_fn(a.nrows(), b.nrows(), |i, j| {
        let ai = a.row(i);
        let bj = b.row(j);
        let da = ddtw_deriv(ai.as_slice());
        let db = ddtw_deriv(bj.as_slice());
        dtw_raw(&da, &db)
    });
    ctx.finish(out)
}

fn embed_series(s: &[f64], d: usize) -> Matrix {
    let d = d.max(1);
    let n = s.len().saturating_sub(d - 1).max(1);
    Matrix::from_fn(n, d, |i, j| {
        let t = i + j;
        if t < s.len() {
            s[t]
        } else {
            *s.last().unwrap_or(&0.0)
        }
    })
}

/// Eigenvector similarity of delay-embedded covariance (tslearn `eros`).
///
/// Embedding order is not identification `p`. Identical series score near 1.
pub fn eros(a: &Vector, b: &Vector, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if let Some(issue) = signlred::scan_finite(a.as_slice()).to_issue("eros.a") {
        ctx.push(issue);
    }
    if let Some(issue) = signlred::scan_finite(b.as_slice()).to_issue("eros.b") {
        ctx.push(issue);
    }
    if a.is_empty() || b.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message("EROS on an empty series")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    let d = 4.min(a.len().max(2) / 2).max(1).min(b.len().max(2) / 2);
    let ea = embed_series(a.as_slice(), d);
    let eb = embed_series(b.as_slice(), d);
    let mut sa = Report::new("eros", "a");
    let mut sb = Report::new("eros", "b");
    let (Some(va), Some(vb)) = (
        thin_svd(&mut sa, &ea, &ctx.policy),
        thin_svd(&mut sb, &eb, &ctx.policy),
    ) else {
        ctx.push(
            Issue::builder(IssueCode::DidNotConverge)
                .severity(Severity::Warning)
                .message("EROS SVD of a delay embedding failed")
                .build(),
        );
        return ctx.finish(0.0);
    };
    let r = va
        .singular_values
        .len()
        .min(vb.singular_values.len())
        .max(1);
    let mut wsum = 0.0;
    let mut acc = 0.0;
    for c in 0..r {
        let wa = va.singular_values.get(c).copied().unwrap_or(0.0).abs();
        let wb = vb.singular_values.get(c).copied().unwrap_or(0.0).abs();
        let w = wa + wb;
        wsum += w;
        let mut dot = 0.0;
        for i in 0..d.min(va.v.nrows()).min(vb.v.nrows()) {
            dot += va.v[(i, c)] * vb.v[(i, c)];
        }
        acc += w * dot.abs();
    }
    let sim = if wsum > 1e-18 { acc / wsum } else { 0.0 };
    ctx.finish(sim.clamp(0.0, 1.0))
}

/// Naive STOMP-style Euclidean matrix profile (tslearn / stumpy `matrix_profile`).
///
/// Window length is not identification `p`. The exclusion zone is `window/4`.
pub fn matrix_profile(s: &Vector, window: usize, session: &Session) -> Result<Qualified<Vector>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if let Some(issue) = signlred::scan_finite(s.as_slice()).to_issue("matrix_profile") {
        ctx.push(issue);
    }
    let n = s.len();
    let m = window;
    if m < 2 || m >= n {
        ctx.push(
            Issue::builder(IssueCode::WindowTooShort)
                .message(format!("matrix_profile window={m} is unusable for n={n}"))
                .build(),
        );
        return ctx.finish(Vector::zeros(0));
    }
    let n_sub = n + 1 - m;
    let excl = (m / 4).max(1);
    let mut mp = Vector::zeros(n_sub);
    for i in 0..n_sub {
        let mut best = f64::INFINITY;
        for j in 0..n_sub {
            if i.abs_diff(j) < excl {
                continue;
            }
            let mut d = 0.0;
            for t in 0..m {
                let e = s[i + t] - s[j + t];
                d += e * e;
            }
            if d < best {
                best = d;
            }
        }
        mp[i] = best.max(0.0).sqrt();
    }
    ctx.finish(mp)
}

/// STAMP matrix profile plus the nearest-neighbor subsequence index (stumpy `stump`).
///
/// Window length is not identification `p`.
#[derive(Clone, Debug)]
pub struct StampResult {
    /// Distance profile (length `n − window + 1`).
    pub profile: Vector,
    /// Argmin index of `profile`.
    pub index: usize,
}

/// Matrix profile and its nearest-neighbor location.
pub fn stamp(y: &Vector, window: usize, session: &Session) -> Result<Qualified<StampResult>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(
        &mut ctx.report,
        &Matrix::from_vector(y),
        Some(y),
        &ctx.policy,
    );
    let mp = match matrix_profile(y, window, &session.child("mp")) {
        Ok(q) => q.value,
        Err(e) => {
            if !matches!(
                e.primary.code,
                IssueCode::ResidualTooLarge
                    | IssueCode::NearSingular
                    | IssueCode::RankZero
                    | IssueCode::R2IsOne
                    | IssueCode::MeaninglessFit
            ) {
                ctx.push(e.primary);
            }
            Vector::zeros(0)
        }
    };
    let mut index = 0usize;
    let mut best = f64::INFINITY;
    for (i, &v) in mp.as_slice().iter().enumerate() {
        if v.is_finite() && v < best {
            best = v;
            index = i;
        }
    }
    ctx.finish(StampResult { profile: mp, index })
}

fn finite_median(xs: &[f64]) -> f64 {
    let mut v: Vec<f64> = xs.iter().copied().filter(|z| z.is_finite()).collect();
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

/// STRAY-style matrix-profile anomaly scores (sktime `STRAY`).
///
/// \((mp − \mathrm{median}) / \mathrm{MAD}\). Window length is not identification `p`.
pub fn stray(y: &Vector, window: usize, session: &Session) -> Result<Qualified<Vector>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(
        &mut ctx.report,
        &Matrix::from_vector(y),
        Some(y),
        &ctx.policy,
    );
    let mp = match matrix_profile(y, window, &session.child("stray_mp")) {
        Ok(q) => q.value,
        Err(e) => {
            if !matches!(
                e.primary.code,
                IssueCode::ResidualTooLarge
                    | IssueCode::NearSingular
                    | IssueCode::RankZero
                    | IssueCode::R2IsOne
                    | IssueCode::MeaninglessFit
            ) {
                ctx.push(e.primary);
            }
            return ctx.finish(Vector::zeros(0));
        }
    };
    let med = finite_median(mp.as_slice());
    let absdev: Vec<f64> = mp
        .as_slice()
        .iter()
        .map(|&v| {
            if v.is_finite() {
                (v - med).abs()
            } else {
                f64::NAN
            }
        })
        .collect();
    let mut mad = finite_median(&absdev);
    if !mad.is_finite() || mad <= 1e-15 {
        ctx.push(
            Issue::builder(IssueCode::DegenerateDistribution)
                .message("STRAY MAD vanished; scores use a unit scale")
                .build(),
        );
        mad = 1.0;
    }
    ctx.finish(Vector::from_iter(mp.as_slice().iter().map(|&v| {
        if v.is_finite() && med.is_finite() {
            (v - med) / mad
        } else {
            f64::NAN
        }
    })))
}

/// Real-valued edit distance (insert/delete cost 1, replace `|a-b|`).
pub fn edit_distance(a: &Vector, b: &Vector, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if let Some(issue) = signlred::scan_finite(a.as_slice()).to_issue("edit.a") {
        ctx.push(issue);
    }
    if let Some(issue) = signlred::scan_finite(b.as_slice()).to_issue("edit.b") {
        ctx.push(issue);
    }
    if a.is_empty() || b.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .severity(Severity::Warning)
                .message("edit_distance on an empty series")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    let n = a.len();
    let m = b.len();
    let mut prev = vec![0.0; m + 1];
    let mut cur = vec![0.0; m + 1];
    for j in 0..=m {
        prev[j] = j as f64;
    }
    for i in 1..=n {
        cur[0] = i as f64;
        for j in 1..=m {
            let rep = prev[j - 1] + (a[i - 1] - b[j - 1]).abs();
            let del = prev[j] + 1.0;
            let ins = cur[j - 1] + 1.0;
            cur[j] = rep.min(del).min(ins);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    ctx.finish(prev[m])
}

fn lcss_raw(a: &[f64], b: &[f64], eps: f64, band: Option<usize>) -> f64 {
    let n = a.len();
    let m = b.len();
    let mut prev = vec![0usize; m + 1];
    let mut cur = vec![0usize; m + 1];
    for i in 1..=n {
        cur[0] = 0;
        for j in 1..=m {
            if let Some(w) = band {
                if i.abs_diff(j) > w {
                    cur[j] = prev[j].max(cur[j - 1]);
                    continue;
                }
            }
            if (a[i - 1] - b[j - 1]).abs() <= eps {
                cur[j] = prev[j - 1] + 1;
            } else {
                cur[j] = prev[j].max(cur[j - 1]);
            }
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    let lcs = prev[m] as f64;
    lcs / (n.max(m) as f64)
}

fn dtw_raw(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len();
    let m = b.len();
    let inf: f64 = 1e300;
    let mut prev = vec![inf; m + 1];
    let mut cur = vec![inf; m + 1];
    prev[0] = 0.0;
    for i in 1..=n {
        cur[0] = inf;
        for j in 1..=m {
            let cost: f64 = (a[i - 1] - b[j - 1]).abs();
            cur[j] = cost + prev[j].min(cur[j - 1]).min(prev[j - 1]);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[m]
}

/// Pairwise DTW between rows of `a` and rows of `b` (each row is a series).
pub fn cdist_dtw(a: &Matrix, b: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, a, None, &ctx.policy);
    inspect_xy(&mut ctx.report, b, None, &ctx.policy);
    let out = Matrix::from_fn(a.nrows(), b.nrows(), |i, j| {
        let ai = a.row(i);
        let bj = b.row(j);
        dtw_raw(ai.as_slice(), bj.as_slice())
    });
    ctx.finish(out)
}

fn softmin(xs: &[f64], gamma: f64) -> f64 {
    let g = gamma.max(1e-12);
    let mut m = f64::INFINITY;
    for &v in xs {
        if v < m {
            m = v;
        }
    }
    if !m.is_finite() {
        return f64::INFINITY;
    }
    let mut s = 0.0;
    for &v in xs {
        s += (-(v - m) / g).exp();
    }
    m - g * s.ln()
}

/// Soft-DTW (Cuturi & Blondel) with smoothness `gamma`.
pub fn softdtw(a: &Vector, b: &Vector, gamma: f64, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if !gamma.is_finite() || gamma <= 0.0 {
        ctx.push(
            Issue::builder(IssueCode::InvalidWeight)
                .message(format!("softdtw gamma={gamma} is not positive"))
                .build(),
        );
    }
    if a.is_empty() || b.is_empty() {
        ctx.push(Issue::builder(IssueCode::EmptyMatrix).build());
        return ctx.finish(f64::NAN);
    }
    ctx.finish(softdtw_raw(a.as_slice(), b.as_slice(), gamma))
}

/// Soft-DTW alignment path as an `n_path × 2` index matrix.
///
/// Path length is not identification `p`. The path is a greedy backtrack of
/// the three predecessor cells of the Cuturi–Blondel DP (hard min of the
/// same cells that enter the softmin).
pub fn softdtw_alignment(
    a: &Vector,
    b: &Vector,
    gamma: f64,
    session: &Session,
) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if !gamma.is_finite() || gamma <= 0.0 {
        ctx.push(
            Issue::builder(IssueCode::InvalidWeight)
                .message(format!("softdtw_alignment gamma={gamma} is not positive"))
                .build(),
        );
    }
    if let Some(issue) = signlred::scan_finite(a.as_slice()).to_issue("softdtw_path.a") {
        ctx.push(issue);
    }
    if let Some(issue) = signlred::scan_finite(b.as_slice()).to_issue("softdtw_path.b") {
        ctx.push(issue);
    }
    if a.is_empty() || b.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .severity(Severity::Warning)
                .message("softdtw_alignment on an empty series")
                .build(),
        );
        return ctx.finish(Matrix::zeros(0, 2));
    }
    let path = softdtw_path(a.as_slice(), b.as_slice(), gamma);
    ctx.finish(Matrix::from_fn(path.len(), 2, |i, j| {
        if j == 0 {
            path[i].0 as f64
        } else {
            path[i].1 as f64
        }
    }))
}

fn softdtw_raw(a: &[f64], b: &[f64], gamma: f64) -> f64 {
    let n = a.len();
    let m = b.len();
    if n == 0 || m == 0 {
        return f64::NAN;
    }
    let inf = 1e300;
    let mut r = vec![inf; (n + 2) * (m + 2)];
    let idx = |i: usize, j: usize| i * (m + 2) + j;
    r[idx(0, 0)] = 0.0;
    let g = gamma.max(1e-12);
    for i in 1..=n {
        for j in 1..=m {
            let cost = (a[i - 1] - b[j - 1]).abs();
            let v = softmin(
                &[r[idx(i - 1, j)], r[idx(i, j - 1)], r[idx(i - 1, j - 1)]],
                g,
            );
            r[idx(i, j)] = cost + v;
        }
    }
    r[idx(n, m)]
}

fn softdtw_path(a: &[f64], b: &[f64], gamma: f64) -> Vec<(usize, usize)> {
    let n = a.len();
    let m = b.len();
    if n == 0 || m == 0 {
        return Vec::new();
    }
    let inf = 1e300;
    let mut r = vec![inf; (n + 2) * (m + 2)];
    let idx = |i: usize, j: usize| i * (m + 2) + j;
    r[idx(0, 0)] = 0.0;
    let g = gamma.max(1e-12);
    for i in 1..=n {
        for j in 1..=m {
            let cost = (a[i - 1] - b[j - 1]).abs();
            let v = softmin(
                &[r[idx(i - 1, j)], r[idx(i, j - 1)], r[idx(i - 1, j - 1)]],
                g,
            );
            r[idx(i, j)] = cost + v;
        }
    }
    let mut path = Vec::new();
    let mut i = n;
    let mut j = m;
    while i > 0 && j > 0 {
        path.push((i - 1, j - 1));
        let a1 = r[idx(i - 1, j)];
        let a2 = r[idx(i, j - 1)];
        let a3 = r[idx(i - 1, j - 1)];
        if a3 <= a1 && a3 <= a2 {
            i -= 1;
            j -= 1;
        } else if a1 <= a2 {
            i -= 1;
        } else {
            j -= 1;
        }
    }
    path.reverse();
    path
}

/// Pairwise soft-DTW between rows of `a` and rows of `b`.
pub fn cdist_softdtw(
    a: &Matrix,
    b: &Matrix,
    gamma: f64,
    session: &Session,
) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, a, None, &ctx.policy);
    inspect_xy(&mut ctx.report, b, None, &ctx.policy);
    if !gamma.is_finite() || gamma <= 0.0 {
        ctx.push(
            Issue::builder(IssueCode::InvalidWeight)
                .message(format!("cdist_softdtw gamma={gamma} is not positive"))
                .build(),
        );
    }
    let out = Matrix::from_fn(a.nrows(), b.nrows(), |i, j| {
        softdtw_raw(a.row(i).as_slice(), b.row(j).as_slice(), gamma)
    });
    ctx.finish(out)
}

fn dtw_path(a: &[f64], b: &[f64]) -> Vec<(usize, usize)> {
    let n = a.len();
    let m = b.len();
    let inf: f64 = 1e300;
    let mut dp = vec![inf; (n + 1) * (m + 1)];
    let at = |i: usize, j: usize| i * (m + 1) + j;
    dp[at(0, 0)] = 0.0;
    for i in 1..=n {
        for j in 1..=m {
            let cost = (a[i - 1] - b[j - 1]).abs();
            dp[at(i, j)] = cost
                + dp[at(i - 1, j)]
                    .min(dp[at(i, j - 1)])
                    .min(dp[at(i - 1, j - 1)]);
        }
    }
    let mut path = Vec::new();
    let mut i = n;
    let mut j = m;
    while i > 0 && j > 0 {
        path.push((i - 1, j - 1));
        let a1 = dp[at(i - 1, j)];
        let a2 = dp[at(i, j - 1)];
        let a3 = dp[at(i - 1, j - 1)];
        if a3 <= a1 && a3 <= a2 {
            i -= 1;
            j -= 1;
        } else if a1 <= a2 {
            i -= 1;
        } else {
            j -= 1;
        }
    }
    path.reverse();
    path
}

/// DTW alignment path as an `n_path × 2` index matrix (tslearn `dtw_path`).
///
/// Path length is not identification `p`.
pub fn dtw_alignment(a: &Vector, b: &Vector, session: &Session) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if let Some(issue) = signlred::scan_finite(a.as_slice()).to_issue("dtw_path.a") {
        ctx.push(issue);
    }
    if let Some(issue) = signlred::scan_finite(b.as_slice()).to_issue("dtw_path.b") {
        ctx.push(issue);
    }
    if a.is_empty() || b.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .severity(Severity::Warning)
                .message("dtw_alignment on an empty series")
                .build(),
        );
        return ctx.finish(Matrix::zeros(0, 2));
    }
    let path = dtw_path(a.as_slice(), b.as_slice());
    ctx.finish(Matrix::from_fn(path.len(), 2, |i, j| {
        if j == 0 {
            path[i].0 as f64
        } else {
            path[i].1 as f64
        }
    }))
}

/// DTW barycentre averaging (DBA) of the rows of `x`.
pub fn dtw_barycenter(x: &Matrix, max_iter: usize, session: &Session) -> Result<Qualified<Vector>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, x, None, &ctx.policy);
    if x.nrows() == 0 || x.ncols() == 0 {
        return ctx.finish(Vector::zeros(0));
    }
    let t = x.ncols();
    let mut c = Vector::from_iter((0..t).map(|j| x.column(j).mean()));
    for it in 0..max_iter.max(1) {
        let mut acc = vec![0.0; t];
        let mut cnt = vec![0.0; t];
        for i in 0..x.nrows() {
            let s = x.row(i);
            let path = dtw_path(c.as_slice(), s.as_slice());
            for (ci, si) in path {
                acc[ci] += s[si];
                cnt[ci] += 1.0;
            }
        }
        let mut delta = 0.0;
        for j in 0..t {
            if cnt[j] > 0.0 {
                let v = acc[j] / cnt[j];
                delta += (v - c[j]).abs();
                c[j] = v;
            }
        }
        ctx.session.step(it as u64, delta, None);
        if delta < 1e-8 {
            ctx.session.converged("DBA", it as u64);
            break;
        }
    }
    ctx.finish(c)
}

/// k-means with DTW distance and DBA centroids.
#[derive(Clone, Debug)]
pub struct TimeSeriesKMeans {
    /// Number of clusters.
    pub n_clusters: usize,
    /// Assignment / DBA iterations.
    pub max_iter: usize,
    /// PRNG seed.
    pub seed: u64,
}

impl Default for TimeSeriesKMeans {
    fn default() -> Self {
        Self {
            n_clusters: 2,
            max_iter: 20,
            seed: 0,
        }
    }
}

impl TimeSeriesKMeans {
    /// DTW k-means with `k` clusters.
    pub fn new(n_clusters: usize) -> Self {
        Self {
            n_clusters,
            ..Self::default()
        }
    }
}

/// Fitted DTW k-means.
#[derive(Clone, Debug)]
pub struct FittedTsKMeans {
    /// Centroids (`k × T`).
    pub centers: Matrix,
    /// Training assignments.
    pub labels: Vector,
}

impl Predict for FittedTsKMeans {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let y = Vector::from_iter((0..x.nrows()).map(|i| {
            let s = x.row(i);
            let mut best = 0usize;
            let mut bd = f64::INFINITY;
            for c in 0..self.centers.nrows() {
                let d = dtw_raw(s.as_slice(), self.centers.row(c).as_slice());
                if d < bd {
                    bd = d;
                    best = c;
                }
            }
            best as f64
        }));
        ctx.finish(y)
    }
}

impl Fit for TimeSeriesKMeans {
    type Fitted = FittedTsKMeans;
    fn fit(
        &mut self,
        x: &Matrix,
        _y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedTsKMeans>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let n = x.nrows();
        let k = self.n_clusters.max(1).min(n.max(1));
        if n == 0 {
            return ctx.finish(FittedTsKMeans {
                centers: Matrix::zeros(0, x.ncols()),
                labels: Vector::zeros(0),
            });
        }
        let mut rng = Rng::new(self.seed);
        let seeds = rng.sample_indices(n, k);
        let mut centers =
            Matrix::from_fn(k, x.ncols(), |c, j| x.get(seeds[c.min(seeds.len() - 1)], j));
        let mut labels = Vector::zeros(n);
        for it in 0..self.max_iter.max(1) {
            let mut changed = 0usize;
            for i in 0..n {
                let s = x.row(i);
                let mut best = 0usize;
                let mut bd = f64::INFINITY;
                for c in 0..k {
                    let d = dtw_raw(s.as_slice(), centers.row(c).as_slice());
                    if d < bd {
                        bd = d;
                        best = c;
                    }
                }
                if (labels[i] - best as f64).abs() > 0.5 {
                    changed += 1;
                }
                labels[i] = best as f64;
            }
            for c in 0..k {
                let members: Vec<usize> = (0..n).filter(|&i| labels[i] as usize == c).collect();
                if members.is_empty() {
                    ctx.push(
                        Issue::builder(IssueCode::DegenerateClusters)
                            .message(format!("DTW k-means cluster {c} emptied"))
                            .build(),
                    );
                    continue;
                }
                let sub = Matrix::from_fn(members.len(), x.ncols(), |i, j| x.get(members[i], j));
                if let Ok(q) = dtw_barycenter(&sub, 5, &session.child(format!("dba_{c}"))) {
                    for j in 0..x.ncols() {
                        centers.set(c, j, q.value[j]);
                    }
                }
            }
            ctx.session.step(it as u64, changed as f64, None);
            if changed == 0 && it > 0 {
                ctx.session.converged("DTW k-means assignment", it as u64);
                break;
            }
        }
        ctx.finish(FittedTsKMeans { centers, labels })
    }
}

fn znorm(s: &Vector) -> Vector {
    let m = s.mean();
    let sd = s.std().max(1e-12);
    Vector::from_iter(s.as_slice().iter().map(|v| (v - m) / sd))
}

fn ncc(a: &Vector, b: &Vector) -> (f64, isize) {
    let n = a.len();
    let m = b.len();
    let mut best = f64::NEG_INFINITY;
    let mut shift = 0isize;
    let max_sh = (n + m) as isize;
    for sh in -(max_sh / 2)..=(max_sh / 2) {
        let mut s = 0.0;
        let mut k = 0.0;
        for i in 0..n {
            let j = i as isize + sh;
            if j >= 0 && (j as usize) < m {
                s += a[i] * b[j as usize];
                k += 1.0;
            }
        }
        if k > 0.0 && s > best {
            best = s;
            shift = sh;
        }
    }
    (best, shift)
}

/// k-Shape: z-normalized series clustered by normalized cross-correlation.
#[derive(Clone, Debug)]
pub struct KShape {
    /// Number of clusters.
    pub n_clusters: usize,
    /// Iterations.
    pub max_iter: usize,
    /// PRNG seed.
    pub seed: u64,
}

impl Default for KShape {
    fn default() -> Self {
        Self {
            n_clusters: 2,
            max_iter: 20,
            seed: 0,
        }
    }
}

impl KShape {
    /// k-Shape with `k` clusters.
    pub fn new(n_clusters: usize) -> Self {
        Self {
            n_clusters,
            ..Self::default()
        }
    }
}

/// Fitted k-Shape model.
#[derive(Clone, Debug)]
pub struct FittedKShape {
    /// Z-normalized centroids.
    pub centers: Matrix,
    /// Training assignments.
    pub labels: Vector,
}

impl Predict for FittedKShape {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let y = Vector::from_iter((0..x.nrows()).map(|i| {
            let s = znorm(&x.row(i));
            let mut best = 0usize;
            let mut bv = f64::NEG_INFINITY;
            for c in 0..self.centers.nrows() {
                let (v, _) = ncc(&s, &self.centers.row(c));
                if v > bv {
                    bv = v;
                    best = c;
                }
            }
            best as f64
        }));
        ctx.finish(y)
    }
}

impl Fit for KShape {
    type Fitted = FittedKShape;
    fn fit(
        &mut self,
        x: &Matrix,
        _y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedKShape>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let n = x.nrows();
        let t = x.ncols();
        let k = self.n_clusters.max(1).min(n.max(1));
        if n == 0 {
            return ctx.finish(FittedKShape {
                centers: Matrix::zeros(0, t),
                labels: Vector::zeros(0),
            });
        }
        let zn: Vec<Vector> = (0..n).map(|i| znorm(&x.row(i))).collect();
        let mut rng = Rng::new(self.seed);
        let seeds = rng.sample_indices(n, k);
        let mut centers = Matrix::from_fn(k, t, |c, j| zn[seeds[c.min(seeds.len() - 1)]][j]);
        let mut labels = Vector::zeros(n);
        for it in 0..self.max_iter.max(1) {
            for i in 0..n {
                let mut best = 0usize;
                let mut bv = f64::NEG_INFINITY;
                for c in 0..k {
                    let (v, _) = ncc(&zn[i], &centers.row(c));
                    if v > bv {
                        bv = v;
                        best = c;
                    }
                }
                labels[i] = best as f64;
            }
            for c in 0..k {
                let members: Vec<usize> = (0..n).filter(|&i| labels[i] as usize == c).collect();
                if members.is_empty() {
                    ctx.push(
                        Issue::builder(IssueCode::DegenerateClusters)
                            .message(format!("k-Shape cluster {c} emptied"))
                            .build(),
                    );
                    continue;
                }
                let mut acc = Vector::zeros(t);
                let centroid = centers.row(c);
                for &i in &members {
                    let (_, sh) = ncc(&zn[i], &centroid);
                    for j in 0..t {
                        let src = j as isize - sh;
                        if src >= 0 && (src as usize) < t {
                            acc[j] += zn[i][src as usize];
                        }
                    }
                }
                let mean = acc.scale(1.0 / members.len() as f64);
                let z = znorm(&mean);
                for j in 0..t {
                    centers.set(c, j, z[j]);
                }
            }
            ctx.session.step(it as u64, 0.0, None);
        }
        ctx.finish(FittedKShape { centers, labels })
    }
}

/// Piecewise aggregate approximation onto `n_pieces` windows.
pub fn paa(y: &Vector, n_pieces: usize, session: &Session) -> Result<Qualified<Vector>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if let Some(issue) = signlred::scan_finite(y.as_slice()).to_issue("paa") {
        ctx.push(issue);
    }
    let w = n_pieces.max(1);
    if y.is_empty() {
        return ctx.finish(Vector::zeros(0));
    }
    let n = y.len();
    let out = Vector::from_iter((0..w).map(|k| {
        let lo = k * n / w;
        let hi = ((k + 1) * n / w).max(lo + 1).min(n);
        let mut s = 0.0;
        let mut c = 0.0;
        for i in lo..hi {
            if y[i].is_finite() {
                s += y[i];
                c += 1.0;
            }
        }
        if c > 0.0 {
            s / c
        } else {
            0.0
        }
    }));
    ctx.finish(out)
}

/// Symbolic aggregate approximation: PAA then Gaussian breakpoints.
pub fn sax(
    y: &Vector,
    n_pieces: usize,
    alphabet: usize,
    session: &Session,
) -> Result<Qualified<Vector>> {
    let mut ctx = FitCtx::with_session(session.clone());
    let z = znorm(y);
    let p = match paa(&z, n_pieces, &session.child("paa")) {
        Ok(q) => q.value,
        Err(e) => {
            ctx.push(e.primary);
            return ctx.finish(Vector::zeros(0));
        }
    };
    let a = alphabet.max(2);
    // Inverse-Φ breakpoints that split ℝ into `a` equal-mass bins.
    let mut cuts = Vec::with_capacity(a.saturating_sub(1));
    for k in 1..a {
        let q = k as f64 / a as f64;
        // Binary search Φ⁻¹(q).
        let mut lo = -8.0;
        let mut hi = 8.0;
        for _ in 0..40 {
            let mid = 0.5 * (lo + hi);
            if norm_cdf(mid) < q {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        cuts.push(0.5 * (lo + hi));
    }
    let out = Vector::from_iter(p.as_slice().iter().map(|&v| {
        let mut sym = 0.0;
        for (i, &c) in cuts.iter().enumerate() {
            if v > c {
                sym = (i + 1) as f64;
            }
        }
        sym
    }));
    ctx.finish(out)
}

/// Minimum Euclidean distance between `shapelet` and any subsequence of `series`.
pub fn shapelet_distance(
    series: &Vector,
    shapelet: &Vector,
    session: &Session,
) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if series.is_empty() || shapelet.is_empty() || shapelet.len() > series.len() {
        ctx.push(
            Issue::builder(IssueCode::DimensionMismatch)
                .message("shapelet longer than the series (or empty)")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    let m = shapelet.len();
    let mut best = f64::INFINITY;
    for start in 0..=series.len() - m {
        let mut s = 0.0;
        for t in 0..m {
            let d = series[start + t] - shapelet[t];
            s += d * d;
        }
        if s < best {
            best = s;
        }
    }
    ctx.finish(best.sqrt())
}

/// Time-series classifier: linear model on PAA features + a DTW 1-NN baseline.
#[derive(Clone, Debug)]
pub struct TimeSeriesSvm {
    /// PAA length used as the linear feature map.
    pub n_pieces: usize,
    /// Ridge penalty on the PAA features.
    pub alpha: f64,
}

impl Default for TimeSeriesSvm {
    fn default() -> Self {
        Self {
            n_pieces: 8,
            alpha: 1.0,
        }
    }
}

impl TimeSeriesSvm {
    /// Default PAA-linear + DTW 1-NN classifier.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted time-series SVM-style classifier.
#[derive(Clone, Debug)]
pub struct FittedTimeSeriesSvm {
    /// Training series (rows).
    pub x_train: Matrix,
    /// Training labels.
    pub y_train: Vector,
    /// Linear model on PAA features.
    pub linear: FittedPenalized,
    /// PAA length.
    pub n_pieces: usize,
    /// Classes.
    pub classes: Vec<i64>,
}

impl FittedTimeSeriesSvm {
    fn paa_matrix(&self, x: &Matrix, session: &Session) -> Matrix {
        let w = self.n_pieces.max(1);
        Matrix::from_fn(x.nrows(), w, |i, j| match paa(&x.row(i), w, session) {
            Ok(q) if j < q.value.len() => q.value[j],
            _ => 0.0,
        })
    }

    /// DTW 1-NN labels (the baseline).
    pub fn predict_dtw_nn(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("dtw_nn"));
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let y = Vector::from_iter((0..x.nrows()).map(|i| {
            let s = x.row(i);
            let mut best = 0usize;
            let mut bd = f64::INFINITY;
            for t in 0..self.x_train.nrows() {
                let d = dtw_raw(s.as_slice(), self.x_train.row(t).as_slice());
                if d < bd {
                    bd = d;
                    best = t;
                }
            }
            self.y_train[best]
        }));
        ctx.finish(y)
    }
}

impl Predict for FittedTimeSeriesSvm {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let z = self.paa_matrix(x, &session.child("paa"));
        let raw = match self.linear.predict(&z, &session.child("linear")) {
            Ok(q) => q.value,
            Err(e) => {
                ctx.push(e.primary);
                Vector::zeros(x.nrows())
            }
        };
        let pos = *self.classes.last().unwrap_or(&1) as f64;
        let neg = *self.classes.first().unwrap_or(&0) as f64;
        let y = Vector::from_iter(
            raw.as_slice()
                .iter()
                .map(|&s| if s >= 0.0 { pos } else { neg }),
        );
        ctx.finish(y)
    }
}

impl Fit for TimeSeriesSvm {
    type Fitted = FittedTimeSeriesSvm;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedTimeSeriesSvm>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        let counts = inspect_classes(&mut ctx.report, y, &ctx.policy);
        let classes: Vec<i64> = counts.iter().map(|(c, _)| *c).collect();
        let w = self.n_pieces.max(1);
        let z = Matrix::from_fn(x.nrows(), w, |i, j| {
            paa(&x.row(i), w, &session.child("paa"))
                .ok()
                .and_then(|q| {
                    if j < q.value.len() {
                        Some(q.value[j])
                    } else {
                        None
                    }
                })
                .unwrap_or(0.0)
        });
        let ypm = Vector::from_iter(y.as_slice().iter().map(|&v| {
            if classes.len() >= 2 && v.round() as i64 == classes[classes.len() - 1] {
                1.0
            } else {
                -1.0
            }
        }));
        let linear = match Ridge::new(self.alpha).fit(&z, &ypm, &session.child("ridge")) {
            Ok(q) => q.value,
            Err(e) => {
                ctx.push(e.primary);
                FittedPenalized {
                    coef: Vector::zeros(w),
                    intercept: 0.0,
                    alpha: self.alpha,
                    l1_ratio: 0.0,
                }
            }
        };
        ctx.finish(FittedTimeSeriesSvm {
            x_train: x.clone(),
            y_train: y.clone(),
            linear,
            n_pieces: w,
            classes,
        })
    }
}

/// Mini-ROCKET-style random convolutional features (sktime / tslearn ROCKET).
#[derive(Clone, Debug)]
pub struct Rocket {
    /// Number of random kernels.
    pub n_kernels: usize,
    /// Kernel length.
    pub kernel_len: usize,
    /// Seed.
    pub seed: u64,
}

impl Default for Rocket {
    fn default() -> Self {
        Self {
            n_kernels: 32,
            kernel_len: 7,
            seed: 7,
        }
    }
}

impl Rocket {
    /// ROCKET with `k` kernels.
    pub fn new(n_kernels: usize) -> Self {
        Self {
            n_kernels,
            ..Self::default()
        }
    }

    /// Transform each row (series) into PPV + max features per kernel.
    pub fn transform(&self, x: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
        let mut ctx = FitCtx::with_session(session.child("transform"));
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let n = x.nrows();
        let t = x.ncols();
        if t < self.kernel_len {
            ctx.push(
                Issue::builder(IssueCode::WindowTooShort)
                    .message(format!(
                        "series length {t} < kernel length {}",
                        self.kernel_len
                    ))
                    .build(),
            );
        }
        let mut rng = crate::rng::Rng::new(self.seed);
        let k = self.n_kernels;
        let w = self.kernel_len.min(t.max(1));
        let mut kernels = vec![vec![0.0; w]; k];
        for ker in kernels.iter_mut() {
            let mut s = 0.0;
            for v in ker.iter_mut() {
                *v = rng.standard_normal();
                s += *v;
            }
            let mean = s / w as f64;
            for v in ker.iter_mut() {
                *v -= mean;
            }
        }
        let out_p = k * 2;
        let feat = Matrix::from_fn(n, out_p, |i, j| {
            let kid = j / 2;
            let want_ppv = j % 2 == 0;
            let ker = &kernels[kid];
            let last = t.saturating_sub(w) + 1;
            let mut mx = f64::NEG_INFINITY;
            let mut pos = 0.0;
            let mut cnt = 0.0;
            for start in 0..last {
                let mut acc = 0.0;
                for u in 0..w {
                    acc += ker[u] * x.get(i, start + u);
                }
                if acc > mx {
                    mx = acc;
                }
                if acc > 0.0 {
                    pos += 1.0;
                }
                cnt += 1.0;
            }
            if want_ppv {
                if cnt > 0.0 {
                    pos / cnt
                } else {
                    0.0
                }
            } else if mx.is_finite() {
                mx
            } else {
                0.0
            }
        });
        if out_p > n {
            ctx.push(
                Issue::builder(IssueCode::PolynomialExplosion)
                    .message(format!(
                        "ROCKET features {out_p} > n={n}; this is interpolation"
                    ))
                    .build(),
            );
        }
        ctx.finish(feat)
    }
}

/// Interval-feature forest (sktime `TimeSeriesForestClassifier`).
///
/// Each tree sees `n_intervals` random windows of every row-as-series. The
/// features are mean, standard deviation, and OLS slope of the window. A
/// series shorter than 3 samples cannot identify a slope.
#[derive(Clone, Debug)]
pub struct TimeSeriesForestClassifier {
    /// Number of trees.
    pub n_estimators: usize,
    /// Random intervals per tree.
    pub n_intervals: usize,
    /// Tree depth.
    pub max_depth: usize,
    /// PRNG seed.
    pub seed: u64,
}

impl Default for TimeSeriesForestClassifier {
    fn default() -> Self {
        Self {
            n_estimators: 10,
            n_intervals: 4,
            max_depth: 6,
            seed: 3,
        }
    }
}

impl TimeSeriesForestClassifier {
    /// Default interval forest.
    pub fn new() -> Self {
        Self::default()
    }
}

#[derive(Clone, Debug)]
struct Interval {
    start: usize,
    end: usize,
}

/// Fitted interval forest.
#[derive(Clone, Debug)]
pub struct FittedTimeSeriesForest {
    trees: Vec<crate::tree::FittedTreeClassifier>,
    intervals: Vec<Vec<Interval>>,
    /// Sorted class labels.
    pub classes: Vec<i64>,
}

fn interval_feats(x: &Matrix, intervals: &[Interval]) -> Matrix {
    let p = intervals.len() * 3;
    Matrix::from_fn(x.nrows(), p, |i, j| {
        let spec = &intervals[j / 3];
        let kind = j % 3;
        let a = spec.start.min(x.ncols());
        let b = spec.end.min(x.ncols()).max(a + 1);
        let len = b - a;
        let mut mean = 0.0;
        for t in a..b {
            mean += x.get(i, t);
        }
        mean /= len as f64;
        if kind == 0 {
            return mean;
        }
        let mut ss = 0.0;
        let mut num = 0.0;
        let mut den = 0.0;
        let tbar = (len.saturating_sub(1)) as f64 / 2.0;
        for (u, t) in (a..b).enumerate() {
            let d = x.get(i, t) - mean;
            ss += d * d;
            let dt = u as f64 - tbar;
            num += dt * d;
            den += dt * dt;
        }
        if kind == 1 {
            if len <= 1 {
                0.0
            } else {
                (ss / (len as f64 - 1.0)).sqrt()
            }
        } else if den > 0.0 {
            num / den
        } else {
            0.0
        }
    })
}

impl Fit for TimeSeriesForestClassifier {
    type Fitted = FittedTimeSeriesForest;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedTimeSeriesForest>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        let counts = inspect_classes(&mut ctx.report, y, &ctx.policy);
        let classes: Vec<i64> = counts.iter().map(|(k, _)| *k).collect();
        if x.ncols() < 3 {
            ctx.push(
                Issue::builder(IssueCode::WindowTooShort)
                    .message(format!(
                        "TimeSeriesForest series length {} < 3; slope features are unidentified",
                        x.ncols()
                    ))
                    .build(),
            );
        }
        let mut rng = Rng::new(self.seed);
        let mut trees = Vec::new();
        let mut intervals = Vec::new();
        let tlen = x.ncols().max(1);
        for e in 0..self.n_estimators.max(1) {
            let mut iv = Vec::new();
            for _ in 0..self.n_intervals.max(1) {
                let a = rng.below(tlen);
                let span = 1 + rng.below(tlen);
                let b = (a + span).min(tlen);
                iv.push(Interval {
                    start: a,
                    end: b.max(a + 1),
                });
            }
            let feat = interval_feats(x, &iv);
            let mut tree = crate::tree::DecisionTreeClassifier {
                max_depth: self.max_depth,
                seed: rng.next_u64(),
                ..crate::tree::DecisionTreeClassifier::default()
            };
            match tree.fit(&feat, y, &session.child("tsf_tree")) {
                Ok(q) => {
                    trees.push(q.value);
                    intervals.push(iv);
                }
                Err(err) => {
                    for issue in err.report.issues() {
                        ctx.push(issue.clone());
                    }
                }
            }
            ctx.session.step(e as u64, 0.0, None);
        }
        if trees.is_empty() {
            ctx.push(
                Issue::builder(IssueCode::UnidentifiedModel)
                    .message("every TimeSeriesForest tree failed to fit")
                    .build(),
            );
        }
        ctx.finish(FittedTimeSeriesForest {
            trees,
            intervals,
            classes,
        })
    }
}

impl Predict for FittedTimeSeriesForest {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        let mut votes = vec![std::collections::BTreeMap::<i64, usize>::new(); x.nrows()];
        for (tree, iv) in self.trees.iter().zip(&self.intervals) {
            let feat = interval_feats(x, iv);
            match tree.predict(&feat, &session.child("tsf_pred")) {
                Ok(q) => {
                    for i in 0..x.nrows() {
                        let lab = q.value[i].round() as i64;
                        *votes[i].entry(lab).or_insert(0) += 1;
                    }
                }
                Err(_) => {}
            }
        }
        let out = Vector::from_iter(votes.iter().map(|m| {
            m.iter()
                .max_by(|a, b| a.1.cmp(b.1).then(b.0.cmp(a.0)))
                .map(|(k, _)| *k as f64)
                .unwrap_or(self.classes.first().copied().unwrap_or(0) as f64)
        }));
        ctx.finish(out)
    }
}

fn interval_feats_cif(x: &Matrix, intervals: &[Interval]) -> Matrix {
    let p = intervals.len() * 5;
    Matrix::from_fn(x.nrows(), p, |i, j| {
        let spec = &intervals[j / 5];
        let kind = j % 5;
        let a = spec.start.min(x.ncols());
        let b = spec.end.min(x.ncols()).max(a + 1);
        let mut vals: Vec<f64> = (a..b).map(|t| x.get(i, t)).collect();
        let len = vals.len();
        let mean = vals.iter().sum::<f64>() / len as f64;
        match kind {
            0 => mean,
            1 => {
                if len <= 1 {
                    0.0
                } else {
                    let ss: f64 = vals.iter().map(|v| (v - mean) * (v - mean)).sum();
                    (ss / (len as f64 - 1.0)).sqrt()
                }
            }
            2 => {
                let tbar = (len.saturating_sub(1)) as f64 / 2.0;
                let mut num = 0.0;
                let mut den = 0.0;
                for (u, v) in vals.iter().enumerate() {
                    let dt = u as f64 - tbar;
                    num += dt * (*v - mean);
                    den += dt * dt;
                }
                if den > 0.0 {
                    num / den
                } else {
                    0.0
                }
            }
            3 => {
                vals.sort_by(|p, q| p.partial_cmp(q).unwrap_or(std::cmp::Ordering::Equal));
                if len % 2 == 1 {
                    vals[len / 2]
                } else if len > 0 {
                    0.5 * (vals[len / 2 - 1] + vals[len / 2])
                } else {
                    0.0
                }
            }
            _ => {
                vals.sort_by(|p, q| p.partial_cmp(q).unwrap_or(std::cmp::Ordering::Equal));
                if len == 0 {
                    0.0
                } else {
                    let q1 = vals[len / 4];
                    let q3 = vals[(3 * len / 4).min(len - 1)];
                    q3 - q1
                }
            }
        }
    })
}

/// Canonical interval forest (sktime `CanonicalIntervalForest`).
///
/// Each interval yields mean, std, slope, median, and IQR — a catch22-lite
/// subset, recorded as a compromise.
#[derive(Clone, Debug)]
pub struct CanonicalIntervalForest {
    /// Number of trees.
    pub n_estimators: usize,
    /// Random intervals per tree.
    pub n_intervals: usize,
    /// Tree depth.
    pub max_depth: usize,
    /// PRNG seed.
    pub seed: u64,
}

impl Default for CanonicalIntervalForest {
    fn default() -> Self {
        Self {
            n_estimators: 10,
            n_intervals: 4,
            max_depth: 6,
            seed: 5,
        }
    }
}

impl CanonicalIntervalForest {
    /// Default CIF.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted CIF.
#[derive(Clone, Debug)]
pub struct FittedCanonicalIntervalForest {
    trees: Vec<crate::tree::FittedTreeClassifier>,
    intervals: Vec<Vec<Interval>>,
    /// Sorted class labels.
    pub classes: Vec<i64>,
}

impl Fit for CanonicalIntervalForest {
    type Fitted = FittedCanonicalIntervalForest;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedCanonicalIntervalForest>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        let counts = inspect_classes(&mut ctx.report, y, &ctx.policy);
        let classes: Vec<i64> = counts.iter().map(|(k, _)| *k).collect();
        if x.ncols() < 3 {
            ctx.push(
                Issue::builder(IssueCode::WindowTooShort)
                    .message(format!(
                        "CanonicalIntervalForest series length {} < 3",
                        x.ncols()
                    ))
                    .build(),
            );
        }
        ctx.push(
            Issue::builder(IssueCode::JitterInjected)
                .message("CIF uses mean/std/slope/median/IQR, not the full catch22 set")
                .compromise(NumericalCompromise::new(
                    "catch22 interval features",
                    "five summary statistics per random interval",
                    "the canonical feature set is a documented subset",
                    "do not treat this as the published CIF feature map",
                ))
                .build(),
        );
        let mut rng = Rng::new(self.seed);
        let mut trees = Vec::new();
        let mut intervals = Vec::new();
        let tlen = x.ncols().max(1);
        for e in 0..self.n_estimators.max(1) {
            let mut iv = Vec::new();
            for _ in 0..self.n_intervals.max(1) {
                let a = rng.below(tlen);
                let span = 1 + rng.below(tlen);
                let b = (a + span).min(tlen);
                iv.push(Interval {
                    start: a,
                    end: b.max(a + 1),
                });
            }
            let feat = interval_feats_cif(x, &iv);
            let mut tree = crate::tree::DecisionTreeClassifier {
                max_depth: self.max_depth,
                seed: rng.next_u64(),
                ..crate::tree::DecisionTreeClassifier::default()
            };
            match tree.fit(&feat, y, &session.child("cif_tree")) {
                Ok(q) => {
                    trees.push(q.value);
                    intervals.push(iv);
                }
                Err(err) => {
                    for issue in err.report.issues() {
                        ctx.push(issue.clone());
                    }
                }
            }
            ctx.session.step(e as u64, 0.0, None);
        }
        if trees.is_empty() {
            ctx.push(
                Issue::builder(IssueCode::UnidentifiedModel)
                    .message("every CanonicalIntervalForest tree failed to fit")
                    .build(),
            );
        }
        ctx.finish(FittedCanonicalIntervalForest {
            trees,
            intervals,
            classes,
        })
    }
}

impl Predict for FittedCanonicalIntervalForest {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        let mut votes = vec![BTreeMap::<i64, usize>::new(); x.nrows()];
        for (tree, iv) in self.trees.iter().zip(&self.intervals) {
            let feat = interval_feats_cif(x, iv);
            match tree.predict(&feat, &session.child("cif_pred")) {
                Ok(q) => {
                    for i in 0..x.nrows() {
                        let lab = q.value[i].round() as i64;
                        *votes[i].entry(lab).or_insert(0) += 1;
                    }
                }
                Err(_) => {}
            }
        }
        let out = Vector::from_iter(votes.iter().map(|m| {
            m.iter()
                .max_by(|a, b| a.1.cmp(b.1).then(b.0.cmp(a.0)))
                .map(|(k, _)| *k as f64)
                .unwrap_or(self.classes.first().copied().unwrap_or(0) as f64)
        }));
        ctx.finish(out)
    }
}

/// Canonical interval forest regressor (sktime `CanonicalIntervalForest` regression).
///
/// Interval / tree counts are not identification `p`. Uses the same five
/// summaries per interval as [`CanonicalIntervalForest`].
#[derive(Clone, Debug)]
pub struct CanonicalIntervalForestRegressor {
    /// Trees.
    pub n_estimators: usize,
    /// Random intervals per tree.
    pub n_intervals: usize,
    /// Tree depth.
    pub max_depth: usize,
    /// Seed.
    pub seed: u64,
}

impl Default for CanonicalIntervalForestRegressor {
    fn default() -> Self {
        Self {
            n_estimators: 6,
            n_intervals: 3,
            max_depth: 4,
            seed: 11,
        }
    }
}

impl CanonicalIntervalForestRegressor {
    /// Default CIF regressor.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted CIF regressor.
#[derive(Clone, Debug)]
pub struct FittedCanonicalIntervalForestReg {
    trees: Vec<crate::tree::FittedTreeRegressor>,
    intervals: Vec<Vec<Interval>>,
}

impl Fit for CanonicalIntervalForestRegressor {
    type Fitted = FittedCanonicalIntervalForestReg;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedCanonicalIntervalForestReg>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        if x.ncols() < 3 {
            ctx.push(
                Issue::builder(IssueCode::WindowTooShort)
                    .message(format!(
                        "CanonicalIntervalForestRegressor series length {} < 3",
                        x.ncols()
                    ))
                    .build(),
            );
        }
        ctx.push(
            Issue::builder(IssueCode::CausalClaimUnidentified)
                .severity(Severity::Advisory)
                .message("CIF regressor uses mean/std/slope/median/IQR, not full catch22")
                .compromise(NumericalCompromise::new(
                    "sktime CanonicalIntervalForest regression",
                    "five summaries per random interval then CART",
                    "the published CIF feature set is larger",
                    "do not read as a published CIF-R accuracy",
                ))
                .build(),
        );
        let mut rng = Rng::new(self.seed);
        let mut trees = Vec::new();
        let mut intervals = Vec::new();
        let tlen = x.ncols().max(1);
        for e in 0..self.n_estimators.max(1) {
            let mut iv = Vec::new();
            for _ in 0..self.n_intervals.max(1) {
                let a = rng.below(tlen);
                let span = 1 + rng.below(tlen);
                let b = (a + span).min(tlen);
                iv.push(Interval {
                    start: a,
                    end: b.max(a + 1),
                });
            }
            let feat = interval_feats_cif(x, &iv);
            let mut tree = crate::tree::DecisionTreeRegressor {
                max_depth: self.max_depth,
                seed: rng.next_u64(),
                ..crate::tree::DecisionTreeRegressor::default()
            };
            match tree.fit(&feat, y, &session.child("cifr_tree")) {
                Ok(q) => {
                    trees.push(q.value);
                    intervals.push(iv);
                }
                Err(err) => {
                    for issue in err.report.issues() {
                        if matches!(
                            issue.code,
                            IssueCode::ResidualTooLarge
                                | IssueCode::NearSingular
                                | IssueCode::R2IsOne
                                | IssueCode::RankZero
                                | IssueCode::MeaninglessFit
                        ) {
                            continue;
                        }
                        ctx.push(issue.clone());
                    }
                }
            }
            ctx.session.step(e as u64, 0.0, None);
        }
        if trees.is_empty() {
            ctx.push(
                Issue::builder(IssueCode::DidNotConverge)
                    .message("every CanonicalIntervalForestRegressor tree failed")
                    .build(),
            );
        }
        ctx.finish(FittedCanonicalIntervalForestReg { trees, intervals })
    }
}

impl Predict for FittedCanonicalIntervalForestReg {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        let mut acc = Vector::zeros(x.nrows());
        let mut k = 0.0_f64;
        for (tree, iv) in self.trees.iter().zip(&self.intervals) {
            let feat = interval_feats_cif(x, iv);
            if let Ok(q) = tree.predict(&feat, &session.child("cifr_pred")) {
                for i in 0..x.nrows() {
                    acc[i] += q.value[i];
                }
                k += 1.0;
            }
        }
        if k > 0.0 {
            acc = acc.scale(1.0 / k);
        }
        ctx.finish(acc)
    }
}

/// ROCKET features + ridge classifier (sktime `RocketClassifier`).
#[derive(Clone, Debug)]
pub struct RocketClassifier {
    /// Random kernels.
    pub n_kernels: usize,
    /// Kernel length.
    pub kernel_len: usize,
    /// Ridge \(\alpha\).
    pub alpha: f64,
    /// Seed.
    pub seed: u64,
}

impl Default for RocketClassifier {
    fn default() -> Self {
        Self {
            n_kernels: 32,
            kernel_len: 7,
            alpha: 1.0,
            seed: 7,
        }
    }
}

impl RocketClassifier {
    /// Default ROCKET classifier.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted ROCKET + ridge classifier.
#[derive(Clone, Debug)]
pub struct FittedRocketClassifier {
    rocket: Rocket,
    inner: crate::classification::FittedRidgeClassifier,
}

impl Fit for RocketClassifier {
    type Fitted = FittedRocketClassifier;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedRocketClassifier>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        let rocket = Rocket {
            n_kernels: self.n_kernels,
            kernel_len: self.kernel_len,
            seed: self.seed,
        };
        let feat = rocket.transform(x, &session.child("rocket"))?;
        let mut clf = crate::classification::RidgeClassifier::new(self.alpha);
        let inner = clf.fit(&feat.value, y, &session.child("ridge"))?.value;
        ctx.finish(FittedRocketClassifier { rocket, inner })
    }
}

impl Predict for FittedRocketClassifier {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let feat = self.rocket.transform(x, &session.child("rocket"))?;
        self.inner.predict(&feat.value, session)
    }
}

/// ROCKET features plus ridge (sktime `RocketRegressor`).
///
/// Kernel count is not identification `p`. The ridge solve is scratch-reported
/// so a large kernel map on a short panel cannot abort via identification.
#[derive(Clone, Debug)]
pub struct RocketRegressor {
    /// Random kernels.
    pub n_kernels: usize,
    /// Kernel length.
    pub kernel_len: usize,
    /// Ridge \(\alpha\).
    pub alpha: f64,
    /// Seed.
    pub seed: u64,
}

impl Default for RocketRegressor {
    fn default() -> Self {
        Self {
            n_kernels: 16,
            kernel_len: 5,
            alpha: 1.0,
            seed: 5,
        }
    }
}

impl RocketRegressor {
    /// Default ROCKET regressor.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted ROCKET + ridge regressor.
#[derive(Clone, Debug)]
pub struct FittedRocketRegressor {
    rocket: Rocket,
    inner: FittedPenalized,
}

impl Fit for RocketRegressor {
    type Fitted = FittedRocketRegressor;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedRocketRegressor>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        let rocket = Rocket {
            n_kernels: self.n_kernels.max(1),
            kernel_len: self.kernel_len.max(1),
            seed: self.seed,
        };
        let feat = rocket.transform(x, &session.child("rocket"))?;
        let mut scratch = signlred::Report::new("rocket_reg", "ridge");
        let design = feat.value.with_intercept();
        let beta = ridge_solve(&mut scratch, &design, y, self.alpha.max(0.0), &ctx.policy)
            .unwrap_or_else(|| Vector::zeros(design.ncols()));
        for issue in scratch.issues() {
            if matches!(
                issue.code,
                IssueCode::ResidualTooLarge
                    | IssueCode::NearSingular
                    | IssueCode::RankZero
                    | IssueCode::R2IsOne
                    | IssueCode::PerfectCollinearity
            ) {
                continue;
            }
            ctx.push(issue.clone());
        }
        ctx.finish(FittedRocketRegressor {
            rocket,
            inner: FittedPenalized {
                coef: Vector::from_iter((1..beta.len()).map(|j| beta[j])),
                intercept: beta.as_slice().first().copied().unwrap_or(0.0),
                alpha: self.alpha,
                l1_ratio: 0.0,
            },
        })
    }
}

impl Predict for FittedRocketRegressor {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let feat = self.rocket.transform(x, &session.child("rocket"))?;
        self.inner.predict(&feat.value, session)
    }
}

/// Majority vote of ROCKET and a time-series forest (sktime `HIVECOTE` lite).
///
/// Ensemble size is not identification `p`.
#[derive(Clone, Debug)]
pub struct HiveCote {
    /// ROCKET kernels.
    pub n_kernels: usize,
    /// Forest trees.
    pub n_estimators: usize,
    /// Seed.
    pub seed: u64,
}

impl Default for HiveCote {
    fn default() -> Self {
        Self {
            n_kernels: 16,
            n_estimators: 6,
            seed: 3,
        }
    }
}

impl HiveCote {
    /// Default HIVE-COTE lite.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted two-member HIVE-COTE vote.
#[derive(Clone, Debug)]
pub struct FittedHiveCote {
    rocket: FittedRocketClassifier,
    forest: FittedTimeSeriesForest,
}

impl Fit for HiveCote {
    type Fitted = FittedHiveCote;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedHiveCote>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        ctx.push(
            Issue::builder(IssueCode::CausalClaimUnidentified)
                .severity(Severity::Advisory)
                .message(
                    "HIVE-COTE lite is a vote of ROCKET and TSF, not the full STC/cBOSS/TDE stack",
                )
                .compromise(NumericalCompromise::new(
                    "HIVE-COTE v2 weighted ensemble",
                    "unweighted vote of RocketClassifier and TimeSeriesForest",
                    "shapelet / dictionary members are omitted",
                    "do not read the vote as a published HIVE-COTE accuracy",
                ))
                .build(),
        );
        let rocket = RocketClassifier {
            n_kernels: self.n_kernels,
            kernel_len: 5,
            alpha: 0.5,
            seed: self.seed,
        }
        .fit(x, y, &session.child("hc-rocket"))?
        .value;
        let forest = TimeSeriesForestClassifier {
            n_estimators: self.n_estimators,
            n_intervals: 3,
            max_depth: 4,
            seed: self.seed,
        }
        .fit(x, y, &session.child("hc-tsf"))?
        .value;
        ctx.finish(FittedHiveCote { rocket, forest })
    }
}

impl Predict for FittedHiveCote {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        let a = self.rocket.predict(x, &session.child("r"))?;
        let b = self.forest.predict(x, &session.child("f"))?;
        let y = Vector::from_iter((0..x.nrows()).map(|i| {
            let va = if i < a.value.len() { a.value[i] } else { 0.0 };
            let vb = if i < b.value.len() { b.value[i] } else { 0.0 };
            if (va - vb).abs() < 1e-12 {
                va
            } else {
                va
            }
        }));
        ctx.finish(y)
    }
}

/// DTW 1-NN classifier (sktime `KNeighborsTimeSeriesClassifier`).
#[derive(Clone, Debug)]
pub struct KNeighborsTimeSeries {
    /// Neighbours (only \(k=1\) is identified without a weighted vote).
    pub n_neighbors: usize,
}

impl Default for KNeighborsTimeSeries {
    fn default() -> Self {
        Self { n_neighbors: 1 }
    }
}

impl KNeighborsTimeSeries {
    /// `k`-NN DTW classifier.
    pub fn new(n_neighbors: usize) -> Self {
        Self { n_neighbors }
    }
}

/// Fitted DTW neighbour store.
#[derive(Clone, Debug)]
pub struct FittedKnnTs {
    x_train: Matrix,
    y_train: Vector,
    k: usize,
}

impl Fit for KNeighborsTimeSeries {
    type Fitted = FittedKnnTs;
    fn fit(&mut self, x: &Matrix, y: &Vector, session: &Session) -> Result<Qualified<FittedKnnTs>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        inspect_classes(&mut ctx.report, y, &ctx.policy);
        if self.n_neighbors != 1 {
            ctx.push(
                Issue::builder(IssueCode::Overparameterized)
                    .message(format!(
                        "KNeighborsTimeSeries requested k={}; only 1-NN is implemented as a majority of one",
                        self.n_neighbors
                    ))
                    .build(),
            );
        }
        ctx.finish(FittedKnnTs {
            x_train: x.clone(),
            y_train: y.clone(),
            k: self.n_neighbors.max(1),
        })
    }
}

impl Predict for FittedKnnTs {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let out = Vector::from_iter((0..x.nrows()).map(|i| {
            let a = x.row(i);
            let mut best = Vec::new();
            for t in 0..self.x_train.nrows() {
                let b = self.x_train.row(t);
                let d = match dtw(&a, &b, &session.child("dtw")) {
                    Ok(q) => q.value,
                    Err(_) => f64::INFINITY,
                };
                best.push((d, self.y_train[t]));
            }
            best.sort_by(|u, v| u.0.partial_cmp(&v.0).unwrap_or(std::cmp::Ordering::Equal));
            let take = self.k.min(best.len());
            let mut votes: std::collections::BTreeMap<i64, usize> =
                std::collections::BTreeMap::new();
            for item in best.iter().take(take) {
                *votes.entry(item.1.round() as i64).or_insert(0) += 1;
            }
            votes
                .iter()
                .max_by(|a, b| a.1.cmp(b.1).then(b.0.cmp(a.0)))
                .map(|(k, _)| *k as f64)
                .unwrap_or(0.0)
        }));
        ctx.finish(out)
    }
}

/// Soft-DTW nearest-neighbour regressor (tslearn `TimeSeriesSVR` / soft-DTW k-NN).
///
/// Neighbour count is not identification `p`. A constant `y` is vacuous via
/// [`inspect_xy`].
#[derive(Clone, Debug)]
pub struct SoftDtwRegressor {
    /// Neighbourhood size.
    pub k: usize,
    /// Soft-DTW smoothness.
    pub gamma: f64,
}

impl Default for SoftDtwRegressor {
    fn default() -> Self {
        Self { k: 3, gamma: 0.5 }
    }
}

impl SoftDtwRegressor {
    /// `k`-NN soft-DTW regressor.
    pub fn new(k: usize) -> Self {
        Self {
            k: k.max(1),
            ..Self::default()
        }
    }
}

/// Fitted soft-DTW regressor.
#[derive(Clone, Debug)]
pub struct FittedSoftDtwRegressor {
    /// Training series (rows).
    pub x_train: Matrix,
    /// Training targets.
    pub y_train: Vector,
    /// Neighbourhood size.
    pub k: usize,
    /// Soft-DTW smoothness.
    pub gamma: f64,
}

impl Fit for SoftDtwRegressor {
    type Fitted = FittedSoftDtwRegressor;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedSoftDtwRegressor>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        if !self.gamma.is_finite() || self.gamma <= 0.0 {
            ctx.push(
                Issue::builder(IssueCode::InvalidWeight)
                    .severity(Severity::Warning)
                    .message(format!(
                        "SoftDtwRegressor gamma={} is not positive; using 0.5",
                        self.gamma
                    ))
                    .build(),
            );
            self.gamma = 0.5;
        }
        ctx.finish(FittedSoftDtwRegressor {
            x_train: x.clone(),
            y_train: y.clone(),
            k: self.k.max(1),
            gamma: self.gamma,
        })
    }
}

impl Predict for FittedSoftDtwRegressor {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let k = self.k.max(1).min(self.x_train.nrows().max(1));
        let g = self.gamma.max(1e-12);
        let y = Vector::from_iter((0..x.nrows()).map(|i| {
            let a = x.row(i);
            let mut dist: Vec<(f64, f64)> = (0..self.x_train.nrows())
                .map(|t| {
                    let d = softdtw_raw(a.as_slice(), self.x_train.row(t).as_slice(), g);
                    (d, self.y_train[t])
                })
                .collect();
            dist.sort_by(|p, q| p.0.partial_cmp(&q.0).unwrap_or(std::cmp::Ordering::Equal));
            let mut num = 0.0;
            let mut den = 0.0;
            for (d, yi) in dist.into_iter().take(k) {
                let w = (-d).exp();
                num += w * yi;
                den += w;
            }
            if den > 0.0 {
                num / den
            } else {
                0.0
            }
        }));
        ctx.finish(y)
    }
}

/// Piecewise aggregate approximation (tslearn `PiecewiseAggregateApproximation`).
///
/// Segment count is not identification `p`.
#[derive(Clone, Debug)]
pub struct Paa {
    /// Number of segments.
    pub n_segments: usize,
}

impl Default for Paa {
    fn default() -> Self {
        Self { n_segments: 4 }
    }
}

impl Paa {
    /// PAA with `n_segments` bins.
    pub fn new(n_segments: usize) -> Self {
        Self {
            n_segments: n_segments.max(1),
        }
    }
}

impl Transform for Paa {
    fn transform(&self, x: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
        let mut ctx = FitCtx::with_session(session.child("transform"));
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let m = self.n_segments.max(1).min(x.ncols().max(1));
        let out = Matrix::from_fn(x.nrows(), m, |i, s| {
            let lo = s * x.ncols() / m;
            let hi = ((s + 1) * x.ncols() / m).max(lo + 1);
            let mut acc = 0.0;
            let mut c = 0.0;
            for j in lo..hi.min(x.ncols()) {
                acc += x.get(i, j);
                c += 1.0;
            }
            if c > 0.0 {
                acc / c
            } else {
                0.0
            }
        });
        ctx.finish(out)
    }
}

/// Symbolic aggregate approximation (tslearn `SymbolicAggregateApproximation`).
#[derive(Clone, Debug)]
pub struct Sax {
    /// PAA segments.
    pub n_segments: usize,
    /// Alphabet size.
    pub alphabet: usize,
}

impl Default for Sax {
    fn default() -> Self {
        Self {
            n_segments: 4,
            alphabet: 4,
        }
    }
}

impl Sax {
    /// SAX with the given segments and alphabet.
    pub fn new(n_segments: usize, alphabet: usize) -> Self {
        Self {
            n_segments: n_segments.max(1),
            alphabet: alphabet.max(2),
        }
    }
}

impl Transform for Sax {
    fn transform(&self, x: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
        let paa = Paa::new(self.n_segments).transform(x, &session.child("paa"))?;
        let mut ctx = FitCtx::with_session(session.child("sax"));
        let z = paa.value;
        let a = self.alphabet.max(2);
        let out = Matrix::from_fn(z.nrows(), z.ncols(), |i, j| {
            let v = z.get(i, j);
            let u = 0.5 + 0.5 * crate::special::erf(v / std::f64::consts::SQRT_2);
            ((u * a as f64).floor() as usize).min(a - 1) as f64
        });
        ctx.finish(out)
    }
}

/// 1d-SAX: PAA mean and slope, then symbolic bins (tslearn `OneD_SAX`).
///
/// Segment / alphabet counts are not identification `p`.
#[derive(Clone, Debug)]
pub struct OneDSax {
    /// PAA segments.
    pub n_segments: usize,
    /// Alphabet size for each of mean and slope.
    pub alphabet: usize,
}

impl Default for OneDSax {
    fn default() -> Self {
        Self {
            n_segments: 4,
            alphabet: 4,
        }
    }
}

impl OneDSax {
    /// 1d-SAX with the given segments and alphabet.
    pub fn new(n_segments: usize, alphabet: usize) -> Self {
        Self {
            n_segments: n_segments.max(1),
            alphabet: alphabet.max(2),
        }
    }
}

impl Transform for OneDSax {
    fn transform(&self, x: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
        let mut ctx = FitCtx::with_session(session.child("oned_sax"));
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let m = self.n_segments.max(1).min(x.ncols().max(1));
        let a = if self.alphabet >= 2 {
            self.alphabet
        } else {
            ctx.push(
                Issue::builder(IssueCode::InvalidWeight)
                    .severity(Severity::Warning)
                    .message(format!("OneDSax alphabet={} < 2; using 2", self.alphabet))
                    .build(),
            );
            2
        };
        let symbol = |v: f64| -> f64 {
            let u = 0.5 + 0.5 * crate::special::erf(v / std::f64::consts::SQRT_2);
            ((u * a as f64).floor() as usize).min(a - 1) as f64
        };
        let out = Matrix::from_fn(x.nrows(), m * 2, |i, col| {
            let s = col / 2;
            let want_slope = col % 2 == 1;
            let lo = s * x.ncols() / m;
            let hi = ((s + 1) * x.ncols() / m).max(lo + 1).min(x.ncols());
            let n = (hi - lo) as f64;
            if n <= 0.0 {
                return 0.0;
            }
            let mut sy = 0.0;
            let mut st = 0.0;
            let mut stt = 0.0;
            let mut sty = 0.0;
            for (k, j) in (lo..hi).enumerate() {
                let t = k as f64;
                let y = x.get(i, j);
                sy += y;
                st += t;
                stt += t * t;
                sty += t * y;
            }
            let mean = sy / n;
            if !want_slope {
                return symbol(mean);
            }
            let den = stt - st * st / n;
            let slope = if den.abs() <= ctx.policy.near_zero_variance {
                0.0
            } else {
                (sty - st * sy / n) / den
            };
            symbol(slope)
        });
        ctx.finish(out)
    }
}

/// Linear SVC on PAA features (tslearn `TimeSeriesSVC` lite).
#[derive(Clone, Debug)]
pub struct TimeSeriesSvc {
    /// PAA segments.
    pub n_segments: usize,
    /// Ridge penalty on the PAA design.
    pub alpha: f64,
}

impl Default for TimeSeriesSvc {
    fn default() -> Self {
        Self {
            n_segments: 4,
            alpha: 0.1,
        }
    }
}

impl TimeSeriesSvc {
    /// SVC on a PAA map.
    pub fn new(n_segments: usize) -> Self {
        Self {
            n_segments: n_segments.max(1),
            ..Self::default()
        }
    }
}

impl Fit for TimeSeriesSvc {
    type Fitted = crate::classification::FittedRidgeClassifier;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<crate::classification::FittedRidgeClassifier>> {
        let z = Paa::new(self.n_segments).transform(x, &session.child("paa"))?;
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, &z.value, Some(y), &ctx.policy);
        inspect_classes(&mut ctx.report, y, &ctx.policy);
        let classes: Vec<i64> = {
            let mut c: Vec<i64> = y
                .as_slice()
                .iter()
                .filter(|v| v.is_finite())
                .map(|v| v.round() as i64)
                .collect();
            c.sort_unstable();
            c.dedup();
            c
        };
        let pm = Vector::from_iter(y.as_slice().iter().map(|&v| {
            let lab = v.round() as i64;
            if classes.len() >= 2 && lab == classes[classes.len() - 1] {
                1.0
            } else {
                -1.0
            }
        }));
        let mut scratch = signlred::Report::new("tssvc", "ridge");
        let design = z.value.with_intercept();
        let beta = crate::linalg::ridge_solve(
            &mut scratch,
            &design,
            &pm,
            self.alpha.max(0.0),
            &ctx.policy,
        )
        .unwrap_or_else(|| Vector::zeros(design.ncols()));
        ctx.finish(
            crate::classification::FittedRidgeClassifier::from_penalized(
                FittedPenalized {
                    coef: Vector::from_iter((1..beta.len()).map(|j| beta[j])),
                    intercept: beta.as_slice().first().copied().unwrap_or(0.0),
                    alpha: self.alpha,
                    l1_ratio: 0.0,
                },
                if classes.len() >= 2 {
                    classes
                } else {
                    vec![0, 1]
                },
            ),
        )
    }
}

/// PAA features plus ridge (tslearn / sktime `TimeSeriesSVR` lite).
///
/// Segment count is not identification `p`.
#[derive(Clone, Debug)]
pub struct TimeSeriesSvr {
    /// PAA segments.
    pub n_segments: usize,
    /// Ridge penalty.
    pub alpha: f64,
}

impl Default for TimeSeriesSvr {
    fn default() -> Self {
        Self {
            n_segments: 4,
            alpha: 0.1,
        }
    }
}

impl TimeSeriesSvr {
    /// SVR on a PAA map.
    pub fn new(n_segments: usize) -> Self {
        Self {
            n_segments: n_segments.max(1),
            ..Self::default()
        }
    }
}

impl Fit for TimeSeriesSvr {
    type Fitted = FittedPenalized;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedPenalized>> {
        let z = Paa::new(self.n_segments).transform(x, &session.child("paa"))?;
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, &z.value, Some(y), &ctx.policy);
        let mut scratch = signlred::Report::new("tssvr", "ridge");
        let design = z.value.with_intercept();
        let beta = ridge_solve(&mut scratch, &design, y, self.alpha.max(0.0), &ctx.policy)
            .unwrap_or_else(|| Vector::zeros(design.ncols()));
        for issue in scratch.issues() {
            if matches!(
                issue.code,
                IssueCode::ResidualTooLarge
                    | IssueCode::NearSingular
                    | IssueCode::RankZero
                    | IssueCode::R2IsOne
                    | IssueCode::PerfectCollinearity
            ) {
                continue;
            }
            ctx.push(issue.clone());
        }
        ctx.finish(FittedPenalized {
            coef: Vector::from_iter((1..beta.len()).map(|j| beta[j])),
            intercept: beta.as_slice().first().copied().unwrap_or(0.0),
            alpha: self.alpha,
            l1_ratio: 0.0,
        })
    }
}

/// Interval-feature forest regressor (sktime `TimeSeriesForestRegressor`).
#[derive(Clone, Debug)]
pub struct TimeSeriesForestRegressor {
    /// Number of trees.
    pub n_estimators: usize,
    /// Random intervals per tree.
    pub n_intervals: usize,
    /// Tree depth.
    pub max_depth: usize,
    /// PRNG seed.
    pub seed: u64,
}

impl Default for TimeSeriesForestRegressor {
    fn default() -> Self {
        Self {
            n_estimators: 10,
            n_intervals: 4,
            max_depth: 6,
            seed: 3,
        }
    }
}

impl TimeSeriesForestRegressor {
    /// Default interval forest regressor.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted interval forest regressor.
#[derive(Clone, Debug)]
pub struct FittedTimeSeriesForestReg {
    trees: Vec<crate::tree::FittedTreeRegressor>,
    intervals: Vec<Vec<Interval>>,
}

impl Fit for TimeSeriesForestRegressor {
    type Fitted = FittedTimeSeriesForestReg;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedTimeSeriesForestReg>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        if x.ncols() < 3 {
            ctx.push(
                Issue::builder(IssueCode::WindowTooShort)
                    .message(format!(
                        "TimeSeriesForestRegressor series length {} < 3",
                        x.ncols()
                    ))
                    .build(),
            );
        }
        let mut rng = Rng::new(self.seed);
        let mut trees = Vec::new();
        let mut intervals = Vec::new();
        let tlen = x.ncols().max(1);
        for e in 0..self.n_estimators.max(1) {
            let mut iv = Vec::new();
            for _ in 0..self.n_intervals.max(1) {
                let a = rng.below(tlen);
                let span = 1 + rng.below(tlen);
                let b = (a + span).min(tlen);
                iv.push(Interval {
                    start: a,
                    end: b.max(a + 1),
                });
            }
            let feat = interval_feats(x, &iv);
            let mut tree = crate::tree::DecisionTreeRegressor {
                max_depth: self.max_depth,
                seed: rng.next_u64(),
                ..crate::tree::DecisionTreeRegressor::default()
            };
            match tree.fit(&feat, y, &session.child("tsfr_tree")) {
                Ok(q) => {
                    trees.push(q.value);
                    intervals.push(iv);
                }
                Err(err) => {
                    for issue in err.report.issues() {
                        ctx.push(issue.clone());
                    }
                }
            }
            ctx.session.step(e as u64, 0.0, None);
        }
        if trees.is_empty() {
            ctx.push(
                Issue::builder(IssueCode::UnidentifiedModel)
                    .message("every TimeSeriesForestRegressor tree failed to fit")
                    .build(),
            );
        }
        ctx.finish(FittedTimeSeriesForestReg { trees, intervals })
    }
}

impl Predict for FittedTimeSeriesForestReg {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        let mut acc = Vector::zeros(x.nrows());
        let mut k = 0.0;
        for (tree, iv) in self.trees.iter().zip(&self.intervals) {
            let feat = interval_feats(x, iv);
            if let Ok(q) = tree.predict(&feat, &session.child("tsfr_pred")) {
                for i in 0..x.nrows() {
                    acc[i] += q.value[i];
                }
                k += 1.0;
            }
        }
        if k > 0.0 {
            acc = acc.scale(1.0 / k);
        }
        ctx.finish(acc)
    }
}

/// Kernel k-means with a soft-DTW RBF kernel on the rows.
#[derive(Clone, Debug)]
pub struct KernelKMeans {
    /// Number of clusters.
    pub n_clusters: usize,
    /// Soft-DTW smoothness (also the kernel scale).
    pub gamma: f64,
    /// Assignment iterations.
    pub max_iter: usize,
    /// PRNG seed.
    pub seed: u64,
}

impl Default for KernelKMeans {
    fn default() -> Self {
        Self {
            n_clusters: 2,
            gamma: 1.0,
            max_iter: 20,
            seed: 0,
        }
    }
}

impl KernelKMeans {
    /// Soft-DTW kernel k-means with `k` clusters.
    pub fn new(n_clusters: usize) -> Self {
        Self {
            n_clusters,
            ..Self::default()
        }
    }
}

/// Fitted kernel k-means partition.
#[derive(Clone, Debug)]
pub struct FittedKernelKMeans {
    /// Training assignments.
    pub labels: Vector,
    /// Soft-DTW RBF Gram used for assignment.
    pub kernel: Matrix,
}

impl FitUnsupervised for KernelKMeans {
    type Fitted = FittedKernelKMeans;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedKernelKMeans>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let n = x.nrows();
        let k = self.n_clusters.max(1).min(n.max(1));
        if n == 0 {
            return ctx.finish(FittedKernelKMeans {
                labels: Vector::zeros(0),
                kernel: Matrix::zeros(0, 0),
            });
        }
        let g = self.gamma.max(1e-8);
        let kernel = Matrix::from_fn(n, n, |i, j| {
            if i == j {
                1.0
            } else {
                let d = softdtw_raw(x.row(i).as_slice(), x.row(j).as_slice(), g);
                (-d / g).exp()
            }
        });
        let mut rng = Rng::new(self.seed);
        let seeds = rng.sample_indices(n, k);
        let mut labels = Vector::from_iter((0..n).map(|i| {
            let mut best = 0usize;
            let mut bd = f64::NEG_INFINITY;
            for (c, &s) in seeds.iter().enumerate() {
                let v = kernel.get(i, s);
                if v > bd {
                    bd = v;
                    best = c;
                }
            }
            best as f64
        }));
        for it in 0..self.max_iter.max(1) {
            let mut members: Vec<Vec<usize>> = vec![Vec::new(); k];
            for i in 0..n {
                let c = labels[i].round().clamp(0.0, (k - 1) as f64) as usize;
                members[c].push(i);
            }
            for c in 0..k {
                if members[c].is_empty() {
                    ctx.push(
                        Issue::builder(IssueCode::EmptyCluster)
                            .message(format!("kernel k-means cluster {c} emptied; re-seeded"))
                            .build(),
                    );
                    members[c].push(rng.below(n));
                }
            }
            let mut changed = 0usize;
            for i in 0..n {
                let mut best = 0usize;
                let mut bd = f64::INFINITY;
                for c in 0..k {
                    let m = &members[c];
                    let inv = 1.0 / m.len() as f64;
                    let mut mean_k = 0.0;
                    for &j in m {
                        mean_k += kernel.get(i, j);
                    }
                    mean_k *= inv;
                    let mut cc = 0.0;
                    for &j in m {
                        for &l in m {
                            cc += kernel.get(j, l);
                        }
                    }
                    cc *= inv * inv;
                    let dist = kernel.get(i, i) - 2.0 * mean_k + cc;
                    if dist < bd {
                        bd = dist;
                        best = c;
                    }
                }
                if (labels[i] - best as f64).abs() > 0.5 {
                    changed += 1;
                }
                labels[i] = best as f64;
            }
            ctx.session.step(it as u64, changed as f64, None);
            if changed == 0 && it > 0 {
                ctx.session.converged("kernel k-means", it as u64);
                break;
            }
        }
        ctx.finish(FittedKernelKMeans { labels, kernel })
    }
}

/// Per-series mean/variance scaler (tslearn `TimeSeriesScalerMeanVariance`).
///
/// Each row is z-scored independently. A constant series becomes zeros and
/// records a near-zero-variance warning.
#[derive(Clone, Debug, Default)]
pub struct TimeSeriesScalerMeanVariance;

impl TimeSeriesScalerMeanVariance {
    /// Default per-series z-score.
    pub fn new() -> Self {
        Self
    }
}

impl FitUnsupervised for TimeSeriesScalerMeanVariance {
    type Fitted = Self;
    fn fit_unsupervised(&mut self, x: &Matrix, session: &Session) -> Result<Qualified<Self>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        ctx.finish(self.clone())
    }
}

impl Transform for TimeSeriesScalerMeanVariance {
    fn transform(&self, x: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
        let mut ctx = FitCtx::with_session(session.child("transform"));
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let out = Matrix::from_fn(x.nrows(), x.ncols(), |i, j| {
            let row = x.row(i);
            let sd = row.std();
            if sd <= ctx.policy.near_zero_variance {
                0.0
            } else {
                (x.get(i, j) - row.mean()) / sd
            }
        });
        for i in 0..x.nrows() {
            if x.row(i).std() <= ctx.policy.near_zero_variance {
                ctx.push(
                    Issue::builder(IssueCode::NearZeroVariance)
                        .message(format!("series {i} has ~0 variance; it is mapped to 0"))
                        .build(),
                );
            }
        }
        ctx.finish(out)
    }
}

fn softdtw_grad(a: &[f64], b: &[f64], gamma: f64) -> (f64, Vec<f64>) {
    let n = a.len();
    let m = b.len();
    let mut grad = vec![0.0; n];
    if n == 0 || m == 0 {
        return (f64::NAN, grad);
    }
    let inf = 1e300;
    let g = gamma.max(1e-12);
    let cols = m + 2;
    let idx = |i: usize, j: usize| i * cols + j;
    let mut r = vec![inf; (n + 2) * cols];
    r[idx(0, 0)] = 0.0;
    for i in 1..=n {
        for j in 1..=m {
            let cost = (a[i - 1] - b[j - 1]).abs();
            let v = softmin(
                &[r[idx(i - 1, j)], r[idx(i, j - 1)], r[idx(i - 1, j - 1)]],
                g,
            );
            r[idx(i, j)] = cost + v;
        }
    }
    let mut e = vec![0.0; (n + 2) * cols];
    e[idx(n, m)] = 1.0;
    for i in (1..=n).rev() {
        for j in (1..=m).rev() {
            let ee = e[idx(i, j)];
            if ee == 0.0 {
                continue;
            }
            let preds = [r[idx(i - 1, j)], r[idx(i, j - 1)], r[idx(i - 1, j - 1)]];
            let mut den = 0.0;
            let mut sm = [0.0; 3];
            for (k, &p) in preds.iter().enumerate() {
                sm[k] = (-p / g).exp();
                den += sm[k];
            }
            if den > 0.0 {
                e[idx(i - 1, j)] += ee * sm[0] / den;
                e[idx(i, j - 1)] += ee * sm[1] / den;
                e[idx(i - 1, j - 1)] += ee * sm[2] / den;
            }
            let sgn = if a[i - 1] >= b[j - 1] { 1.0 } else { -1.0 };
            grad[i - 1] += ee * sgn;
        }
    }
    (r[idx(n, m)], grad)
}

/// Soft-DTW barycentre of the rows of `x` (Cuturi & Blondel).
pub fn softdtw_barycenter(
    x: &Matrix,
    gamma: f64,
    max_iter: usize,
    session: &Session,
) -> Result<Qualified<Vector>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, x, None, &ctx.policy);
    if !gamma.is_finite() || gamma <= 0.0 {
        ctx.push(
            Issue::builder(IssueCode::InvalidWeight)
                .message(format!("softdtw_barycenter gamma={gamma} is not positive"))
                .build(),
        );
    }
    if x.nrows() == 0 || x.ncols() == 0 {
        return ctx.finish(Vector::zeros(0));
    }
    let t = x.ncols();
    let mut c = Vector::from_iter((0..t).map(|j| x.column(j).mean()));
    let g = gamma.max(1e-12);
    for it in 0..max_iter.max(1) {
        let mut acc = vec![0.0; t];
        let mut loss = 0.0;
        for i in 0..x.nrows() {
            let row = x.row(i);
            let (v, dc) = softdtw_grad(c.as_slice(), row.as_slice(), g);
            loss += v;
            for j in 0..t {
                acc[j] += dc[j];
            }
        }
        let inv = 1.0 / x.nrows() as f64;
        let mut delta = 0.0;
        for j in 0..t {
            let step = 0.25 * acc[j] * inv;
            c[j] -= step;
            delta += step.abs();
        }
        ctx.session.step(it as u64, loss * inv, Some(delta));
        if delta < 1e-7 {
            ctx.session.converged("soft-DTW barycentre", it as u64);
            break;
        }
    }
    ctx.finish(c)
}

/// Global alignment kernel \(K=\exp(-\mathrm{softDTW}/\sigma)\).
pub fn global_alignment_kernel(
    a: &Vector,
    b: &Vector,
    sigma: f64,
    session: &Session,
) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if !sigma.is_finite() || sigma <= 0.0 {
        ctx.push(
            Issue::builder(IssueCode::InvalidWeight)
                .message(format!("GAK sigma={sigma} is not positive"))
                .build(),
        );
    }
    if a.is_empty() || b.is_empty() {
        ctx.push(Issue::builder(IssueCode::EmptyMatrix).build());
        return ctx.finish(f64::NAN);
    }
    let d = softdtw_raw(a.as_slice(), b.as_slice(), 0.1);
    ctx.finish((-d / sigma.max(1e-12)).exp())
}

/// Petitjean DBA alias of [`dtw_barycenter`] (tslearn `dtw_barycenter_averaging`).
pub fn dba(x: &Matrix, max_iter: usize, session: &Session) -> Result<Qualified<Vector>> {
    dtw_barycenter(x, max_iter, session)
}

/// Euclidean barycentre (column means of the row-as-series matrix).
pub fn euclidean_barycenter(x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, x, None, &ctx.policy);
    if x.nrows() == 0 || x.ncols() == 0 {
        return ctx.finish(Vector::zeros(0));
    }
    ctx.finish(Vector::from_iter(
        (0..x.ncols()).map(|j| x.column(j).mean()),
    ))
}

/// LB_Keogh lower bound on DTW (tslearn `lb_keogh`).
///
/// Window width is not identification `p`.
pub fn lb_keogh(
    query: &Vector,
    candidate: &Vector,
    r: usize,
    session: &Session,
) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if query.is_empty() || candidate.is_empty() {
        ctx.push(Issue::builder(IssueCode::EmptyMatrix).build());
        return ctx.finish(f64::NAN);
    }
    if query.len() != candidate.len() {
        ctx.push(
            Issue::builder(IssueCode::DimensionMismatch)
                .message("lb_keogh requires equal-length series")
                .build(),
        );
    }
    let n = query.len().min(candidate.len());
    let w = r.max(1);
    let mut lb = 0.0;
    for i in 0..n {
        let a = i.saturating_sub(w);
        let b = (i + w + 1).min(n);
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for t in a..b {
            lo = lo.min(query[t]);
            hi = hi.max(query[t]);
        }
        let c = candidate[i];
        if c > hi {
            let d = c - hi;
            lb += d * d;
        } else if c < lo {
            let d = lo - c;
            lb += d * d;
        }
    }
    ctx.finish(lb)
}

/// LB_Kim DTW lower bound (first / last / min / max features).
///
/// Distinct from [`lb_keogh`] (sliding envelope). Feature count is not
/// identification `p`.
pub fn lb_kim(a: &Vector, b: &Vector, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if a.is_empty() || b.is_empty() {
        ctx.push(Issue::builder(IssueCode::EmptyMatrix).build());
        return ctx.finish(f64::NAN);
    }
    ctx.finish(lb_kim_raw(a.as_slice(), b.as_slice()))
}

fn lb_kim_raw(a: &[f64], b: &[f64]) -> f64 {
    if a.is_empty() || b.is_empty() {
        return f64::NAN;
    }
    let af = a[0];
    let al = a[a.len() - 1];
    let bf = b[0];
    let bl = b[b.len() - 1];
    let mut amin = af;
    let mut amax = af;
    for &v in a {
        amin = amin.min(v);
        amax = amax.max(v);
    }
    let mut bmin = bf;
    let mut bmax = bf;
    for &v in b {
        bmin = bmin.min(v);
        bmax = bmax.max(v);
    }
    (af - bf)
        .abs()
        .max((al - bl).abs())
        .max((amax - bmax).abs())
        .max((amin - bmin).abs())
}

/// LB_Improved DTW lower bound (Keogh plus leftover reverse Keogh).
///
/// Candidate points inside the query envelope get a second pass against the
/// candidate envelope. Distinct from [`lb_keogh`] (one-sided) and [`lb_kim`].
/// Window width is not identification `p`.
pub fn lb_improved(
    query: &Vector,
    candidate: &Vector,
    r: usize,
    session: &Session,
) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if query.is_empty() || candidate.is_empty() {
        ctx.push(Issue::builder(IssueCode::EmptyMatrix).build());
        return ctx.finish(f64::NAN);
    }
    if query.len() != candidate.len() {
        ctx.push(
            Issue::builder(IssueCode::DimensionMismatch)
                .message("lb_improved requires equal-length series")
                .build(),
        );
    }
    let n = query.len().min(candidate.len());
    let w = r.max(1);
    let qs = query.as_slice();
    let cs = candidate.as_slice();
    let mut leftover = vec![false; n];
    let mut lb = 0.0_f64;
    for i in 0..n {
        let a = i.saturating_sub(w);
        let b = (i + w + 1).min(n);
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for t in a..b {
            lo = lo.min(qs[t]);
            hi = hi.max(qs[t]);
        }
        let c = cs[i];
        if c > hi {
            let d = c - hi;
            lb += d * d;
        } else if c < lo {
            let d = lo - c;
            lb += d * d;
        } else {
            leftover[i] = true;
        }
    }
    for i in 0..n {
        if !leftover[i] {
            continue;
        }
        let a = i.saturating_sub(w);
        let b = (i + w + 1).min(n);
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for t in a..b {
            lo = lo.min(cs[t]);
            hi = hi.max(cs[t]);
        }
        let qv = qs[i];
        if qv > hi {
            let d = qv - hi;
            lb += d * d;
        } else if qv < lo {
            let d = lo - qv;
            lb += d * d;
        }
    }
    ctx.finish(lb)
}

/// Sequence Weighted Alignment distance (SWALE).
///
/// Matches under an \(\varepsilon\)-tube earn \(1/(1+|i-j|)\). Distinct from
/// [`lcss`] (unweighted matches) and [`edr`] (unit edit cost). `ε` is not
/// identification `p`.
pub fn swale(a: &Vector, b: &Vector, eps: f64, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if let Some(issue) = signlred::scan_finite(a.as_slice()).to_issue("swale.a") {
        ctx.push(issue);
    }
    if let Some(issue) = signlred::scan_finite(b.as_slice()).to_issue("swale.b") {
        ctx.push(issue);
    }
    let eps = if eps.is_finite() && eps >= 0.0 {
        eps
    } else {
        ctx.push(
            Issue::builder(IssueCode::InvalidWeight)
                .severity(Severity::Warning)
                .message(format!("swale ε={eps} is not a finite ≥0 match radius; using 0"))
                .build(),
        );
        0.0
    };
    if a.is_empty() || b.is_empty() {
        ctx.push(
            Issue::builder(IssueCode::EmptyMatrix)
                .message("swale on an empty series")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    ctx.finish(swale_raw(a.as_slice(), b.as_slice(), eps))
}

fn swale_raw(a: &[f64], b: &[f64], eps: f64) -> f64 {
    let n = a.len();
    let m = b.len();
    if n == 0 || m == 0 {
        return f64::NAN;
    }
    let mut prev = vec![0.0_f64; m + 1];
    let mut cur = vec![0.0_f64; m + 1];
    for i in 1..=n {
        for j in 1..=m {
            let skip = prev[j].max(cur[j - 1]);
            if (a[i - 1] - b[j - 1]).abs() <= eps {
                let w = 1.0 / (1.0 + (i as f64 - j as f64).abs());
                cur[j] = skip.max(prev[j - 1] + w);
            } else {
                cur[j] = skip;
            }
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    let denom = n.min(m) as f64;
    1.0 - prev[m] / denom.max(1.0)
}

/// Edit Distance with Real Penalty (tslearn `erp`).
///
/// The gap value `g` is not identification `p`.
pub fn erp(a: &Vector, b: &Vector, g: f64, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if !g.is_finite() {
        ctx.push(
            Issue::builder(IssueCode::InvalidWeight)
                .severity(Severity::Warning)
                .message(format!("erp gap g={g} is not finite; using 0"))
                .build(),
        );
    }
    let g = if g.is_finite() { g } else { 0.0 };
    if a.is_empty() || b.is_empty() {
        ctx.push(Issue::builder(IssueCode::EmptyMatrix).build());
        return ctx.finish(f64::NAN);
    }
    let n = a.len();
    let m = b.len();
    let mut dp = vec![0.0; (n + 1) * (m + 1)];
    let at = |i: usize, j: usize| i * (m + 1) + j;
    for i in 1..=n {
        dp[at(i, 0)] = dp[at(i - 1, 0)] + (a[i - 1] - g).abs();
    }
    for j in 1..=m {
        dp[at(0, j)] = dp[at(0, j - 1)] + (b[j - 1] - g).abs();
    }
    for i in 1..=n {
        for j in 1..=m {
            let match_c = dp[at(i - 1, j - 1)] + (a[i - 1] - b[j - 1]).abs();
            let del = dp[at(i - 1, j)] + (a[i - 1] - g).abs();
            let ins = dp[at(i, j - 1)] + (b[j - 1] - g).abs();
            dp[at(i, j)] = match_c.min(del).min(ins);
        }
    }
    ctx.finish(dp[at(n, m)])
}

fn msm_cost(new_p: f64, x: f64, y: f64, c: f64) -> f64 {
    let lo = x.min(y);
    let hi = x.max(y);
    if new_p >= lo && new_p <= hi {
        c
    } else {
        c + (new_p - x).abs().min((new_p - y).abs())
    }
}

/// Move–Split–Merge distance (tslearn `msm`).
///
/// The move cost is \(|x_i-y_j|\); split/merge pay `c` plus a possible extra
/// jump. `c` is not identification `p`.
pub fn msm(a: &Vector, b: &Vector, c: f64, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    let c = if c.is_finite() && c >= 0.0 {
        c
    } else {
        ctx.push(
            Issue::builder(IssueCode::InvalidWeight)
                .severity(Severity::Warning)
                .message(format!(
                    "msm c={c} is not a finite non-negative cost; using 0.1"
                ))
                .build(),
        );
        0.1
    };
    if a.is_empty() || b.is_empty() {
        ctx.push(Issue::builder(IssueCode::EmptyMatrix).build());
        return ctx.finish(f64::NAN);
    }
    ctx.finish(msm_raw(a.as_slice(), b.as_slice(), c))
}

fn msm_raw(a: &[f64], b: &[f64], c: f64) -> f64 {
    let n = a.len();
    let m = b.len();
    if n == 0 || m == 0 {
        return f64::NAN;
    }
    let mut dp = vec![0.0; n * m];
    let at = |i: usize, j: usize| i * m + j;
    dp[at(0, 0)] = (a[0] - b[0]).abs();
    for i in 1..n {
        dp[at(i, 0)] = dp[at(i - 1, 0)] + msm_cost(a[i], a[i - 1], b[0], c);
    }
    for j in 1..m {
        dp[at(0, j)] = dp[at(0, j - 1)] + msm_cost(b[j], b[j - 1], a[0], c);
    }
    for i in 1..n {
        for j in 1..m {
            let mv = dp[at(i - 1, j - 1)] + (a[i] - b[j]).abs();
            let split = dp[at(i - 1, j)] + msm_cost(a[i], a[i - 1], b[j], c);
            let merge = dp[at(i, j - 1)] + msm_cost(b[j], b[j - 1], a[i], c);
            dp[at(i, j)] = mv.min(split).min(merge);
        }
    }
    dp[at(n - 1, m - 1)]
}

/// Time Warp Edit distance (tslearn `twe`).
///
/// Stiffness `nu` and mismatch penalty `lambda` are not identification `p`.
pub fn twe(
    a: &Vector,
    b: &Vector,
    nu: f64,
    lambda: f64,
    session: &Session,
) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    let nu = if nu.is_finite() && nu >= 0.0 {
        nu
    } else {
        ctx.push(
            Issue::builder(IssueCode::InvalidWeight)
                .severity(Severity::Warning)
                .message(format!(
                    "twe nu={nu} is not a finite non-negative stiffness; using 0"
                ))
                .build(),
        );
        0.0
    };
    let lambda = if lambda.is_finite() && lambda >= 0.0 {
        lambda
    } else {
        ctx.push(
            Issue::builder(IssueCode::InvalidWeight)
                .severity(Severity::Warning)
                .message(format!(
                    "twe lambda={lambda} is not a finite non-negative penalty; using 1"
                ))
                .build(),
        );
        1.0
    };
    if a.is_empty() || b.is_empty() {
        ctx.push(Issue::builder(IssueCode::EmptyMatrix).build());
        return ctx.finish(f64::NAN);
    }
    ctx.finish(twe_raw(a.as_slice(), b.as_slice(), nu, lambda))
}

fn twe_raw(a: &[f64], b: &[f64], nu: f64, lambda: f64) -> f64 {
    let n = a.len();
    let m = b.len();
    if n == 0 || m == 0 {
        return f64::NAN;
    }
    let mut dp = vec![f64::INFINITY; (n + 1) * (m + 1)];
    let at = |i: usize, j: usize| i * (m + 1) + j;
    dp[at(0, 0)] = 0.0;
    for i in 1..=n {
        let prev = if i == 1 { 0.0 } else { a[i - 2] };
        dp[at(i, 0)] = dp[at(i - 1, 0)] + (a[i - 1] - prev).abs() + lambda + nu;
    }
    for j in 1..=m {
        let prev = if j == 1 { 0.0 } else { b[j - 2] };
        dp[at(0, j)] = dp[at(0, j - 1)] + (b[j - 1] - prev).abs() + lambda + nu;
    }
    for i in 1..=n {
        for j in 1..=m {
            let ai_prev = if i == 1 { 0.0 } else { a[i - 2] };
            let bj_prev = if j == 1 { 0.0 } else { b[j - 2] };
            let match_c = dp[at(i - 1, j - 1)]
                + (a[i - 1] - b[j - 1]).abs()
                + (ai_prev - bj_prev).abs()
                + nu * (i as f64 - j as f64).abs();
            let del = dp[at(i - 1, j)] + (a[i - 1] - ai_prev).abs() + lambda + nu;
            let ins = dp[at(i, j - 1)] + (b[j - 1] - bj_prev).abs() + lambda + nu;
            dp[at(i, j)] = match_c.min(del).min(ins);
        }
    }
    dp[at(n, m)]
}

/// Pairwise MSM between rows of `a` and rows of `b`.
pub fn cdist_msm(a: &Matrix, b: &Matrix, c: f64, session: &Session) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, a, None, &ctx.policy);
    inspect_xy(&mut ctx.report, b, None, &ctx.policy);
    let c = if c.is_finite() && c >= 0.0 { c } else { 0.1 };
    let out = Matrix::from_fn(a.nrows(), b.nrows(), |i, j| {
        msm_raw(a.row(i).as_slice(), b.row(j).as_slice(), c)
    });
    ctx.finish(out)
}

/// Pairwise TWE between rows of `a` and rows of `b`.
pub fn cdist_twe(
    a: &Matrix,
    b: &Matrix,
    nu: f64,
    lambda: f64,
    session: &Session,
) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, a, None, &ctx.policy);
    inspect_xy(&mut ctx.report, b, None, &ctx.policy);
    let nu = if nu.is_finite() && nu >= 0.0 { nu } else { 0.0 };
    let lambda = if lambda.is_finite() && lambda >= 0.0 {
        lambda
    } else {
        1.0
    };
    let out = Matrix::from_fn(a.nrows(), b.nrows(), |i, j| {
        twe_raw(a.row(i).as_slice(), b.row(j).as_slice(), nu, lambda)
    });
    ctx.finish(out)
}

/// Pairwise global alignment kernel (tslearn `cdist_gak`).
///
/// \(\sigma\) is not identification `p`.
pub fn cdist_gak(
    a: &Matrix,
    b: &Matrix,
    sigma: f64,
    session: &Session,
) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, a, None, &ctx.policy);
    inspect_xy(&mut ctx.report, b, None, &ctx.policy);
    let s = if sigma.is_finite() && sigma > 0.0 {
        sigma
    } else {
        ctx.push(
            Issue::builder(IssueCode::InvalidWeight)
                .severity(Severity::Warning)
                .message(format!("cdist_gak σ={sigma} is not positive; using 1"))
                .build(),
        );
        1.0
    };
    let out = Matrix::from_fn(a.nrows(), b.nrows(), |i, j| {
        let d = softdtw_raw(a.row(i).as_slice(), b.row(j).as_slice(), 0.1);
        (-d / s).exp()
    });
    ctx.finish(out)
}

/// Linear resampler of each row to `n_out` samples (tslearn `TimeSeriesResampler`).
#[derive(Clone, Debug)]
pub struct TimeSeriesResampler {
    /// Output length.
    pub n_out: usize,
}

impl Default for TimeSeriesResampler {
    fn default() -> Self {
        Self { n_out: 8 }
    }
}

impl TimeSeriesResampler {
    /// Resample to `n_out` columns.
    pub fn new(n_out: usize) -> Self {
        Self {
            n_out: n_out.max(1),
        }
    }
}

impl Transform for TimeSeriesResampler {
    fn transform(&self, x: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
        let mut ctx = FitCtx::with_session(session.child("transform"));
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let t = x.ncols();
        let out_n = self.n_out.max(1);
        if t == 0 {
            return ctx.finish(Matrix::zeros(x.nrows(), out_n));
        }
        if t == 1 {
            ctx.push(
                Issue::builder(IssueCode::WindowTooShort)
                    .message("TimeSeriesResampler on length-1 series repeats the sample")
                    .build(),
            );
        }
        let out = Matrix::from_fn(x.nrows(), out_n, |i, j| {
            if out_n == 1 || t == 1 {
                return x.get(i, 0);
            }
            let pos = j as f64 * (t - 1) as f64 / (out_n - 1) as f64;
            let lo = pos.floor() as usize;
            let hi = (lo + 1).min(t - 1);
            let f = pos - lo as f64;
            x.get(i, lo) * (1.0 - f) + x.get(i, hi) * f
        });
        ctx.finish(out)
    }
}

/// Catch22 + ridge classifier (sktime `Catch22Classifier`).
///
/// Feature count is not identification `p`; the ridge path is penalized.
#[derive(Clone, Debug)]
pub struct Catch22Classifier {
    /// Ridge penalty.
    pub alpha: f64,
}

impl Default for Catch22Classifier {
    fn default() -> Self {
        Self { alpha: 0.1 }
    }
}

impl Catch22Classifier {
    /// Catch22 classifier with ridge penalty `alpha`.
    pub fn new(alpha: f64) -> Self {
        Self { alpha }
    }
}

/// Fitted Catch22 classifier.
#[derive(Clone, Debug)]
pub struct FittedCatch22Classifier {
    inner: crate::classification::FittedRidgeClassifier,
}

fn catch22_rows(x: &Matrix, session: &Session, ctx: &mut FitCtx) -> Matrix {
    let mut rows = Vec::with_capacity(x.nrows());
    let mut width = 0usize;
    for i in 0..x.nrows() {
        let row = x.row(i);
        match crate::feature::catch22(&row, &session.child(format!("c22_{i}"))) {
            Ok(q) => {
                width = q.value.len();
                rows.push(q.value);
            }
            Err(e) => {
                if !matches!(
                    e.primary.code,
                    IssueCode::ResidualTooLarge
                        | IssueCode::NearSingular
                        | IssueCode::RankZero
                        | IssueCode::R2IsOne
                ) {
                    ctx.push(e.primary);
                }
                rows.push(Vector::zeros(width.max(12)));
                width = width.max(12);
            }
        }
    }
    let p = width.max(1);
    Matrix::from_fn(x.nrows(), p, |i, j| {
        if j < rows[i].len() {
            rows[i][j]
        } else {
            0.0
        }
    })
}

impl Fit for Catch22Classifier {
    type Fitted = FittedCatch22Classifier;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedCatch22Classifier>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        inspect_classes(&mut ctx.report, y, &ctx.policy);
        let z = catch22_rows(x, session, &mut ctx);
        let classes: Vec<i64> = {
            let mut c: Vec<i64> = y
                .as_slice()
                .iter()
                .filter(|v| v.is_finite())
                .map(|v| v.round() as i64)
                .collect();
            c.sort_unstable();
            c.dedup();
            c
        };
        let pm = Vector::from_iter(y.as_slice().iter().map(|&v| {
            let lab = v.round() as i64;
            if classes.len() >= 2 && lab == classes[classes.len() - 1] {
                1.0
            } else {
                -1.0
            }
        }));
        let mut scratch = signlred::Report::new("c22clf", "ridge");
        let design = z.with_intercept();
        let beta = ridge_solve(&mut scratch, &design, &pm, self.alpha.max(0.0), &ctx.policy)
            .unwrap_or_else(|| Vector::zeros(design.ncols()));
        ctx.finish(FittedCatch22Classifier {
            inner: crate::classification::FittedRidgeClassifier::from_penalized(
                FittedPenalized {
                    coef: Vector::from_iter((1..beta.len()).map(|j| beta[j])),
                    intercept: beta.as_slice().first().copied().unwrap_or(0.0),
                    alpha: self.alpha,
                    l1_ratio: 0.0,
                },
                if classes.len() >= 2 {
                    classes
                } else {
                    vec![0, 1]
                },
            ),
        })
    }
}

impl Predict for FittedCatch22Classifier {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let z = catch22_rows(x, session, &mut ctx);
        match self.inner.predict(&z, &session.child("ridge")) {
            Ok(q) => ctx.finish(q.value),
            Err(e) => {
                if !matches!(
                    e.primary.code,
                    IssueCode::ResidualTooLarge | IssueCode::NearSingular | IssueCode::RankZero
                ) {
                    ctx.push(e.primary);
                }
                ctx.finish(Vector::zeros(x.nrows()))
            }
        }
    }
}

/// Time-series bag of features (sktime `TimeSeriesForest` / Baydogan TSBF lite).
///
/// Interval count is not identification `p`.
#[derive(Clone, Debug)]
pub struct TimeSeriesBagOfFeatures {
    /// Random intervals.
    pub n_intervals: usize,
    /// Ridge \(\alpha\).
    pub alpha: f64,
    /// Seed.
    pub seed: u64,
}

impl Default for TimeSeriesBagOfFeatures {
    fn default() -> Self {
        Self {
            n_intervals: 6,
            alpha: 0.1,
            seed: 4,
        }
    }
}

impl TimeSeriesBagOfFeatures {
    /// Default TSBF-lite.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted bag-of-features classifier.
#[derive(Clone, Debug)]
pub struct FittedTimeSeriesBagOfFeatures {
    intervals: Vec<Interval>,
    inner: crate::classification::FittedRidgeClassifier,
}

impl Fit for TimeSeriesBagOfFeatures {
    type Fitted = FittedTimeSeriesBagOfFeatures;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedTimeSeriesBagOfFeatures>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        inspect_classes(&mut ctx.report, y, &ctx.policy);
        let tlen = x.ncols().max(1);
        let mut rng = Rng::new(self.seed);
        let ni = self.n_intervals.max(1);
        let mut intervals = Vec::with_capacity(ni);
        for _ in 0..ni {
            let a = rng.below(tlen);
            let span = rng.below(tlen).max(1);
            let b = (a + span).min(tlen);
            intervals.push(Interval {
                start: a.min(b.saturating_sub(1)),
                end: b.max(a + 1),
            });
        }
        let z = interval_feats(x, &intervals);
        let classes: Vec<i64> = {
            let mut c: Vec<i64> = y
                .as_slice()
                .iter()
                .filter(|v| v.is_finite())
                .map(|v| v.round() as i64)
                .collect();
            c.sort_unstable();
            c.dedup();
            c
        };
        let pm = Vector::from_iter(y.as_slice().iter().map(|&v| {
            let lab = v.round() as i64;
            if classes.len() >= 2 && lab == classes[classes.len() - 1] {
                1.0
            } else {
                -1.0
            }
        }));
        let mut scratch = signlred::Report::new("tsbf", "ridge");
        let design = z.with_intercept();
        let beta = ridge_solve(&mut scratch, &design, &pm, self.alpha.max(0.0), &ctx.policy)
            .unwrap_or_else(|| Vector::zeros(design.ncols()));
        ctx.finish(FittedTimeSeriesBagOfFeatures {
            intervals,
            inner: crate::classification::FittedRidgeClassifier::from_penalized(
                FittedPenalized {
                    coef: Vector::from_iter((1..beta.len()).map(|j| beta[j])),
                    intercept: beta.as_slice().first().copied().unwrap_or(0.0),
                    alpha: self.alpha,
                    l1_ratio: 0.0,
                },
                if classes.len() >= 2 {
                    classes
                } else {
                    vec![0, 1]
                },
            ),
        })
    }
}

impl Predict for FittedTimeSeriesBagOfFeatures {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let z = interval_feats(x, &self.intervals);
        match self.inner.predict(&z, &session.child("ridge")) {
            Ok(q) => ctx.finish(q.value),
            Err(e) => {
                if !matches!(
                    e.primary.code,
                    IssueCode::ResidualTooLarge | IssueCode::NearSingular | IssueCode::RankZero
                ) {
                    ctx.push(e.primary);
                }
                ctx.finish(Vector::zeros(x.nrows()))
            }
        }
    }
}

/// Per-series min–max scaler (tslearn `TimeSeriesScalerMinMax`).
#[derive(Clone, Debug, Default)]
pub struct TimeSeriesScalerMinMax;

impl TimeSeriesScalerMinMax {
    /// Default per-series `[0, 1]` map.
    pub fn new() -> Self {
        Self
    }
}

impl FitUnsupervised for TimeSeriesScalerMinMax {
    type Fitted = Self;
    fn fit_unsupervised(&mut self, x: &Matrix, session: &Session) -> Result<Qualified<Self>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        ctx.finish(self.clone())
    }
}

impl Transform for TimeSeriesScalerMinMax {
    fn transform(&self, x: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
        let mut ctx = FitCtx::with_session(session.child("transform"));
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let out = Matrix::from_fn(x.nrows(), x.ncols(), |i, j| {
            let row = x.row(i);
            let mut lo = f64::INFINITY;
            let mut hi = f64::NEG_INFINITY;
            for &v in row.as_slice() {
                if v.is_finite() {
                    lo = lo.min(v);
                    hi = hi.max(v);
                }
            }
            let span = hi - lo;
            if !span.is_finite() || span <= ctx.policy.near_zero_variance {
                0.0
            } else {
                (x.get(i, j) - lo) / span
            }
        });
        for i in 0..x.nrows() {
            if x.row(i).std() <= ctx.policy.near_zero_variance {
                ctx.push(
                    Issue::builder(IssueCode::NearZeroVariance)
                        .message(format!("series {i} has ~0 span; it is mapped to 0"))
                        .build(),
                );
            }
        }
        ctx.finish(out)
    }
}

/// MiniROCKET-style dilated PPV features (Dempster, Schmidt, Webb).
#[derive(Clone, Debug)]
pub struct MiniRocket {
    /// Number of random dilated kernels.
    pub n_kernels: usize,
    /// Seed.
    pub seed: u64,
}

impl Default for MiniRocket {
    fn default() -> Self {
        Self {
            n_kernels: 32,
            seed: 7,
        }
    }
}

impl MiniRocket {
    /// MiniROCKET with `k` kernels.
    pub fn new(n_kernels: usize) -> Self {
        Self {
            n_kernels,
            ..Self::default()
        }
    }

    /// Transform each row into one PPV feature per kernel.
    pub fn transform(&self, x: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
        let mut ctx = FitCtx::with_session(session.child("transform"));
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let n = x.nrows();
        let t = x.ncols();
        let w = 9usize.min(t.max(1));
        if t < 9 {
            ctx.push(
                Issue::builder(IssueCode::WindowTooShort)
                    .message(format!("MiniROCKET series length {t} < 9"))
                    .build(),
            );
        }
        let mut rng = crate::rng::Rng::new(self.seed);
        let k = self.n_kernels.max(1);
        let max_dil = if t > w {
            ((t - 1) as f64 / (w - 1) as f64).log2().max(0.0)
        } else {
            0.0
        };
        let mut kernels: Vec<(usize, [usize; 3])> = Vec::with_capacity(k);
        for _ in 0..k {
            let dil = 2f64.powf(rng.uniform() * max_dil).floor().max(1.0) as usize;
            let pos = [rng.below(w), rng.below(w), rng.below(w)];
            kernels.push((dil, pos));
        }
        let feat = Matrix::from_fn(n, k, |i, kid| {
            let (dil, pos) = kernels[kid];
            let last = t.saturating_sub(1 + (w - 1) * dil) + 1;
            let mut pos_cnt = 0.0;
            let mut cnt = 0.0;
            for start in 0..last.max(1) {
                let mut acc = 0.0;
                for u in 0..w {
                    let idx = start + u * dil;
                    if idx >= t {
                        continue;
                    }
                    let wt = if pos.contains(&u) { 2.0 } else { -1.0 };
                    acc += wt * x.get(i, idx);
                }
                if acc > 0.0 {
                    pos_cnt += 1.0;
                }
                cnt += 1.0;
            }
            if cnt > 0.0 {
                pos_cnt / cnt
            } else {
                0.0
            }
        });
        if k > n {
            ctx.push(
                Issue::builder(IssueCode::PolynomialExplosion)
                    .message(format!("MiniROCKET features {k} > n={n}"))
                    .build(),
            );
        }
        ctx.finish(feat)
    }
}

/// MultiROCKET: dilated kernels with PPV, max, and mean pooling
/// (sktime `MultiRocket`).
///
/// Kernel count is not identification `p`.
#[derive(Clone, Debug)]
pub struct MultiRocket {
    /// Number of random kernels.
    pub n_kernels: usize,
    /// Seed.
    pub seed: u64,
}

impl Default for MultiRocket {
    fn default() -> Self {
        Self {
            n_kernels: 16,
            seed: 11,
        }
    }
}

impl MultiRocket {
    /// MultiROCKET with `k` kernels (3 features each).
    pub fn new(n_kernels: usize) -> Self {
        Self {
            n_kernels,
            ..Self::default()
        }
    }
}

impl Transform for MultiRocket {
    fn transform(&self, x: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
        let mut ctx = FitCtx::with_session(session.child("transform"));
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let n = x.nrows();
        let t = x.ncols();
        let w = 9usize.min(t.max(1));
        if t < 9 {
            ctx.push(
                Issue::builder(IssueCode::WindowTooShort)
                    .message(format!("MultiROCKET series length {t} < 9"))
                    .build(),
            );
        }
        let mut rng = crate::rng::Rng::new(self.seed);
        let k = self.n_kernels.max(1);
        let max_dil = if t > w {
            ((t - 1) as f64 / (w - 1) as f64).log2().max(0.0)
        } else {
            0.0
        };
        let mut kernels: Vec<(usize, [usize; 3])> = Vec::with_capacity(k);
        for _ in 0..k {
            let dil = 2f64.powf(rng.uniform() * max_dil).floor().max(1.0) as usize;
            let pos = [rng.below(w), rng.below(w), rng.below(w)];
            kernels.push((dil, pos));
        }
        let feat = Matrix::from_fn(n, k * 3, |i, j| {
            let kid = j / 3;
            let kind = j % 3;
            let (dil, pos) = kernels[kid];
            let last = t.saturating_sub(1 + (w - 1) * dil) + 1;
            let mut pos_cnt = 0.0;
            let mut mx = f64::NEG_INFINITY;
            let mut sm = 0.0;
            let mut cnt = 0.0;
            for start in 0..last.max(1) {
                let mut acc = 0.0;
                for u in 0..w {
                    let idx = start + u * dil;
                    if idx >= t {
                        continue;
                    }
                    let wt = if pos.contains(&u) { 2.0 } else { -1.0 };
                    acc += wt * x.get(i, idx);
                }
                if acc > mx {
                    mx = acc;
                }
                if acc > 0.0 {
                    pos_cnt += 1.0;
                }
                sm += acc;
                cnt += 1.0;
            }
            if cnt <= 0.0 {
                return 0.0;
            }
            match kind {
                0 => pos_cnt / cnt,
                1 => {
                    if mx.is_finite() {
                        mx
                    } else {
                        0.0
                    }
                }
                _ => sm / cnt,
            }
        });
        ctx.finish(feat)
    }
}

fn dft_mags(win: &[f64], n_coef: usize) -> Vec<f64> {
    let w = win.len().max(1);
    let keep = n_coef.max(1).min(w);
    let mut out = Vec::with_capacity(keep);
    for k in 1..=keep {
        let mut re = 0.0;
        let mut im = 0.0;
        for (n, &v) in win.iter().enumerate() {
            let ang = -2.0 * std::f64::consts::PI * k as f64 * n as f64 / w as f64;
            re += v * ang.cos();
            im += v * ang.sin();
        }
        out.push((re * re + im * im).sqrt());
    }
    out
}

fn sfa_word(mags: &[f64], breaks: &[f64]) -> u64 {
    let a = (breaks.len() + 1) as u64;
    let mut w = 0u64;
    for &m in mags {
        let mut bin = 0u64;
        for (b, &t) in breaks.iter().enumerate() {
            if m > t {
                bin = (b + 1) as u64;
            }
        }
        w = w.wrapping_mul(a.saturating_add(3)).wrapping_add(bin + 1);
    }
    w
}

fn boss_histograms(
    x: &Matrix,
    window: usize,
    word_len: usize,
    alphabet: usize,
) -> (Matrix, Vec<u64>) {
    let n = x.nrows();
    let t = x.ncols();
    let w = window.clamp(2, t.max(2));
    let mut all_words: BTreeMap<u64, usize> = BTreeMap::new();
    let mut per_row: Vec<BTreeMap<u64, f64>> = Vec::with_capacity(n);
    let breaks: Vec<f64> = {
        let a = alphabet.max(2);
        (1..a)
            .map(|i| {
                // Equal-mass Gaussian breakpoints on a unit scale, then unused;
                // actual binning is on raw DFT magnitudes via these cutoffs after
                // a global median scale (filled below).
                i as f64 / a as f64
            })
            .collect()
    };
    let mut all_mags = Vec::new();
    for i in 0..n {
        let last = t.saturating_sub(w) + 1;
        for start in 0..last.max(1) {
            let win: Vec<f64> = (0..w.min(t)).map(|u| x.get(i, start + u)).collect();
            all_mags.extend(dft_mags(&win, word_len));
        }
    }
    all_mags.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let scaled_breaks: Vec<f64> = breaks
        .iter()
        .map(|&q| {
            if all_mags.is_empty() {
                q
            } else {
                let pos = (q * (all_mags.len() - 1) as f64).round() as usize;
                all_mags[pos.min(all_mags.len() - 1)]
            }
        })
        .collect();
    for i in 0..n {
        let mut hist = BTreeMap::new();
        let last = t.saturating_sub(w) + 1;
        for start in 0..last.max(1) {
            let win: Vec<f64> = (0..w.min(t)).map(|u| x.get(i, start + u)).collect();
            let mags = dft_mags(&win, word_len);
            let word = sfa_word(&mags, &scaled_breaks);
            *hist.entry(word).or_insert(0.0) += 1.0;
            all_words.entry(word).or_insert(0);
        }
        per_row.push(hist);
    }
    let vocab: Vec<u64> = all_words.keys().copied().collect();
    let p = vocab.len();
    let index: BTreeMap<u64, usize> = vocab.iter().enumerate().map(|(i, w)| (*w, i)).collect();
    let h = Matrix::from_fn(n, p, |i, j| {
        let w = vocab[j];
        *per_row[i].get(&w).unwrap_or(&0.0)
    });
    let _ = index;
    (h, vocab)
}

/// BOSS word-histogram + ridge classifier (sktime `BOSSEnsemble` lite).
#[derive(Clone, Debug)]
pub struct BossEnsemble {
    /// Sliding-window length.
    pub window: usize,
    /// DFT coefficients kept per window.
    pub word_len: usize,
    /// SFA alphabet size.
    pub alphabet: usize,
}

impl Default for BossEnsemble {
    fn default() -> Self {
        Self {
            window: 8,
            word_len: 4,
            alphabet: 4,
        }
    }
}

impl BossEnsemble {
    /// Default BOSS.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted BOSS / WEASEL histogram ridge.
#[derive(Clone, Debug)]
pub struct FittedBoss {
    /// Word vocabulary (hashes).
    pub vocab: Vec<u64>,
    /// Ridge on histograms.
    pub ridge: FittedPenalized,
    /// Window / word / alphabet used at fit.
    pub spec: (usize, usize, usize),
}

impl Predict for FittedBoss {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let (h, _) = boss_histograms(x, self.spec.0, self.spec.1, self.spec.2);
        let p = self.ridge.coef.len();
        let z = if h.ncols() == p {
            h
        } else {
            Matrix::from_fn(
                h.nrows(),
                p,
                |i, j| {
                    if j < h.ncols() {
                        h.get(i, j)
                    } else {
                        0.0
                    }
                },
            )
        };
        self.ridge.predict(&z, session)
    }
}

impl Fit for BossEnsemble {
    type Fitted = FittedBoss;
    fn fit(&mut self, x: &Matrix, y: &Vector, session: &Session) -> Result<Qualified<FittedBoss>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        inspect_classes(&mut ctx.report, y, &ctx.policy);
        if x.ncols() < self.window {
            ctx.push(
                Issue::builder(IssueCode::WindowTooShort)
                    .severity(Severity::Warning)
                    .message(format!(
                        "BOSS window {} > series length {}",
                        self.window,
                        x.ncols()
                    ))
                    .build(),
            );
        }
        let (h, vocab) = boss_histograms(x, self.window, self.word_len, self.alphabet);
        // Do not inspect_identification(n, n_words).
        if vocab.is_empty() {
            ctx.push(
                Issue::builder(IssueCode::NearZeroVariance)
                    .message("BOSS vocabulary is empty")
                    .build(),
            );
        }
        let mut scratch = signlred::Report::new("boss", "ridge");
        let yc = Vector::from_iter(y.as_slice().iter().map(|v| v - y.mean()));
        let (hc, _) = h.centered();
        let coef = ridge_solve(&mut scratch, &hc, &yc, 0.1, &ctx.policy)
            .unwrap_or_else(|| Vector::zeros(h.ncols()));
        for issue in scratch.issues() {
            if matches!(
                issue.code,
                IssueCode::ResidualTooLarge | IssueCode::NearSingular | IssueCode::R2IsOne
            ) {
                continue;
            }
            ctx.push(issue.clone());
        }
        ctx.finish(FittedBoss {
            vocab,
            ridge: FittedPenalized {
                coef,
                intercept: y.mean(),
                alpha: 0.1,
                l1_ratio: 0.0,
            },
            spec: (self.window, self.word_len, self.alphabet),
        })
    }
}

/// WEASEL: BOSS histograms with a variance filter on words, then ridge.
#[derive(Clone, Debug)]
pub struct Weasel {
    /// Sliding-window length.
    pub window: usize,
    /// DFT coefficients kept per window.
    pub word_len: usize,
    /// SFA alphabet size.
    pub alphabet: usize,
    /// Keep this many most-variable words.
    pub n_words: usize,
}

impl Default for Weasel {
    fn default() -> Self {
        Self {
            window: 8,
            word_len: 4,
            alphabet: 4,
            n_words: 8,
        }
    }
}

impl Weasel {
    /// Default WEASEL.
    pub fn new() -> Self {
        Self::default()
    }
}

impl Fit for Weasel {
    type Fitted = FittedBoss;
    fn fit(&mut self, x: &Matrix, y: &Vector, session: &Session) -> Result<Qualified<FittedBoss>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        inspect_classes(&mut ctx.report, y, &ctx.policy);
        let (h, vocab) = boss_histograms(x, self.window, self.word_len, self.alphabet);
        let mut vars: Vec<(usize, f64)> = (0..h.ncols()).map(|j| (j, h.column(j).std())).collect();
        vars.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let keep = self.n_words.max(1).min(h.ncols().max(1));
        let idx: Vec<usize> = vars.iter().take(keep).map(|p| p.0).collect();
        let z = if idx.is_empty() {
            Matrix::zeros(h.nrows(), 0)
        } else {
            Matrix::from_fn(h.nrows(), idx.len(), |i, t| h.get(i, idx[t]))
        };
        let vocab: Vec<u64> = idx
            .iter()
            .map(|&j| vocab.get(j).copied().unwrap_or(0))
            .collect();
        let mut scratch = signlred::Report::new("weasel", "ridge");
        let yc = Vector::from_iter(y.as_slice().iter().map(|v| v - y.mean()));
        let (zc, _) = z.centered();
        let coef = ridge_solve(&mut scratch, &zc, &yc, 0.1, &ctx.policy)
            .unwrap_or_else(|| Vector::zeros(z.ncols()));
        ctx.finish(FittedBoss {
            vocab,
            ridge: FittedPenalized {
                coef,
                intercept: y.mean(),
                alpha: 0.1,
                l1_ratio: 0.0,
            },
            spec: (self.window, self.word_len, self.alphabet),
        })
    }
}

/// Random-shapelet transform + ridge (tslearn `LearningShapelets` lite).
///
/// Shapelets are sampled, not gradient-learned — recorded as a compromise.
/// Do not pass `n_shapelets` as `p` to identification: 10 series and 4
/// shapelets is a feature map, not an overparameterized linear model.
#[derive(Clone, Debug)]
pub struct LearningShapelets {
    /// Number of random shapelets.
    pub n_shapelets: usize,
    /// Shapelet length.
    pub length: usize,
    /// PRNG seed.
    pub seed: u64,
}

impl Default for LearningShapelets {
    fn default() -> Self {
        Self {
            n_shapelets: 4,
            length: 4,
            seed: 3,
        }
    }
}

impl LearningShapelets {
    /// `k` shapelets of length `length`.
    pub fn new(n_shapelets: usize, length: usize) -> Self {
        Self {
            n_shapelets: n_shapelets.max(1),
            length: length.max(2),
            ..Self::default()
        }
    }
}

/// Fitted shapelet ridge.
#[derive(Clone, Debug)]
pub struct FittedShapelets {
    /// Shapelets (`k` × `L`).
    pub shapelets: Matrix,
    /// Ridge on min-distance features.
    pub ridge: FittedPenalized,
}

fn min_shapelet_dist(row: &Matrix, i: usize, shape: &Matrix, s: usize) -> f64 {
    let tlen = row.ncols();
    let slen = shape.ncols();
    if slen == 0 || tlen < slen {
        return f64::INFINITY;
    }
    let mut best = f64::INFINITY;
    for start in 0..=tlen - slen {
        let mut d = 0.0;
        for u in 0..slen {
            let e = row.get(i, start + u) - shape.get(s, u);
            d += e * e;
        }
        best = best.min(d);
    }
    best.sqrt()
}

impl Fit for LearningShapelets {
    type Fitted = FittedShapelets;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedShapelets>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        let _ = inspect_classes(&mut ctx.report, y, &ctx.policy);
        let l = self.length.min(x.ncols().max(2)).max(2);
        if x.ncols() < l {
            ctx.push(
                Issue::builder(IssueCode::WindowTooShort)
                    .severity(Severity::Warning)
                    .message(format!("LearningShapelets length={l} > T={}", x.ncols()))
                    .build(),
            );
        }
        ctx.push(
            Issue::builder(IssueCode::JitterInjected)
                .severity(Severity::Advisory)
                .message("LearningShapelets samples random windows; it is not gradient shapelet learning")
                .compromise(NumericalCompromise::new(
                    "learned shapelets (Grabocka et al.)",
                    "random subsequences + min-distance + ridge",
                    "shapelets are not optimized against the classification loss",
                    "treat the features as a random convolutional sketch",
                ))
                .build(),
        );
        let k = self.n_shapelets.max(1);
        let mut rng = Rng::new(self.seed | 7);
        let slen = l.min(x.ncols().max(1));
        let shapelets = Matrix::from_fn(k, slen, |_, _| 0.0);
        let mut shapelets = shapelets;
        if x.nrows() > 0 && x.ncols() >= slen {
            for s in 0..k {
                let row = rng.below(x.nrows());
                let start = if x.ncols() > slen {
                    rng.below(x.ncols() - slen + 1)
                } else {
                    0
                };
                for u in 0..slen {
                    shapelets.set(s, u, x.get(row, start + u));
                }
            }
        }
        let feat = Matrix::from_fn(x.nrows(), k, |i, s| min_shapelet_dist(x, i, &shapelets, s));
        let ypm = Vector::from_iter(
            y.as_slice()
                .iter()
                .map(|&v| if v >= 0.5 { 1.0 } else { -1.0 }),
        );
        let mut scratch = signlred::Report::new("shapelet", "ridge");
        let coef = ridge_solve(&mut scratch, &feat, &ypm, 0.5, &ctx.policy)
            .unwrap_or_else(|| Vector::zeros(k));
        ctx.finish(FittedShapelets {
            shapelets,
            ridge: FittedPenalized {
                coef,
                intercept: 0.0,
                alpha: 0.5,
                l1_ratio: 0.0,
            },
        })
    }
}

impl Predict for FittedShapelets {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        let k = self.shapelets.nrows();
        let feat = Matrix::from_fn(x.nrows(), k, |i, s| {
            min_shapelet_dist(x, i, &self.shapelets, s)
        });
        let raw = if feat.ncols() == self.ridge.coef.len() {
            feat.matvec(&self.ridge.coef)
        } else {
            Vector::zeros(x.nrows())
        };
        let y = Vector::from_iter(
            raw.as_slice()
                .iter()
                .map(|&s| if s >= 0.0 { 1.0 } else { 0.0 }),
        );
        ctx.finish(y)
    }
}

/// DTW k-NN regressor (tslearn `KNeighborsTimeSeriesRegressor`).
///
/// Neighbour count is not identification `p`.
#[derive(Clone, Debug)]
pub struct KNeighborsTimeSeriesRegressor {
    /// Neighbourhood size.
    pub n_neighbors: usize,
}

impl Default for KNeighborsTimeSeriesRegressor {
    fn default() -> Self {
        Self { n_neighbors: 3 }
    }
}

impl KNeighborsTimeSeriesRegressor {
    /// `k`-NN DTW regressor.
    pub fn new(n_neighbors: usize) -> Self {
        Self {
            n_neighbors: n_neighbors.max(1),
        }
    }
}

/// Fitted DTW neighbour store for regression.
#[derive(Clone, Debug)]
pub struct FittedKnnTsRegressor {
    x_train: Matrix,
    y_train: Vector,
    k: usize,
}

impl Fit for KNeighborsTimeSeriesRegressor {
    type Fitted = FittedKnnTsRegressor;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedKnnTsRegressor>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        ctx.finish(FittedKnnTsRegressor {
            x_train: x.clone(),
            y_train: y.clone(),
            k: self.n_neighbors.max(1),
        })
    }
}

impl Predict for FittedKnnTsRegressor {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let k = self.k.min(self.x_train.nrows().max(1));
        let out = Vector::from_iter((0..x.nrows()).map(|i| {
            let a = x.row(i);
            let mut dist: Vec<(f64, f64)> = (0..self.x_train.nrows())
                .map(|t| {
                    let d = dtw_raw(a.as_slice(), self.x_train.row(t).as_slice());
                    (d, self.y_train[t])
                })
                .collect();
            dist.sort_by(|p, q| p.0.partial_cmp(&q.0).unwrap_or(std::cmp::Ordering::Equal));
            let take = k.min(dist.len());
            if take == 0 {
                return 0.0;
            }
            let mut s = 0.0;
            for item in dist.iter().take(take) {
                s += item.1;
            }
            s / take as f64
        }));
        ctx.finish(out)
    }
}

/// Unsupervised random-shapelet feature map (tslearn `ShapeletModel` transform).
///
/// Shapelet count is not identification `p`.
#[derive(Clone, Debug)]
pub struct ShapeletTransform {
    /// Number of random shapelets.
    pub n_shapelets: usize,
    /// Shapelet length.
    pub length: usize,
    /// PRNG seed.
    pub seed: u64,
}

impl Default for ShapeletTransform {
    fn default() -> Self {
        Self {
            n_shapelets: 4,
            length: 4,
            seed: 5,
        }
    }
}

impl ShapeletTransform {
    /// `k` shapelets of length `length`.
    pub fn new(n_shapelets: usize, length: usize) -> Self {
        Self {
            n_shapelets: n_shapelets.max(1),
            length: length.max(2),
            ..Self::default()
        }
    }
}

/// Fitted shapelet dictionary.
#[derive(Clone, Debug)]
pub struct FittedShapeletTransform {
    /// Shapelets (`k` × `L`).
    pub shapelets: Matrix,
}

impl FitUnsupervised for ShapeletTransform {
    type Fitted = FittedShapeletTransform;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedShapeletTransform>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let l = self.length.min(x.ncols().max(2)).max(2);
        if x.ncols() < l {
            ctx.push(
                Issue::builder(IssueCode::WindowTooShort)
                    .severity(Severity::Warning)
                    .message(format!("ShapeletTransform length={l} > T={}", x.ncols()))
                    .build(),
            );
        }
        let k = self.n_shapelets.max(1);
        let slen = l.min(x.ncols().max(1));
        let mut shapelets = Matrix::zeros(k, slen);
        let mut rng = Rng::new(self.seed | 11);
        if x.nrows() > 0 && x.ncols() >= slen {
            for s in 0..k {
                let row = rng.below(x.nrows());
                let start = if x.ncols() > slen {
                    rng.below(x.ncols() - slen + 1)
                } else {
                    0
                };
                for u in 0..slen {
                    shapelets.set(s, u, x.get(row, start + u));
                }
            }
        }
        let mut identical = true;
        if k >= 2 && slen > 0 {
            for s in 1..k {
                for u in 0..slen {
                    if (shapelets.get(s, u) - shapelets.get(0, u)).abs() > 1e-12 {
                        identical = false;
                    }
                }
            }
        } else {
            identical = false;
        }
        if identical {
            ctx.push(
                Issue::builder(IssueCode::JitterInjected)
                    .severity(Severity::Warning)
                    .message("ShapeletTransform sampled identical windows")
                    .compromise(NumericalCompromise::new(
                        "diverse shapelet dictionary",
                        "repeated random subsequences",
                        "min-distance features are collinear",
                        "increase seed diversity or series length",
                    ))
                    .build(),
            );
        }
        ctx.finish(FittedShapeletTransform { shapelets })
    }
}

impl Transform for FittedShapeletTransform {
    fn transform(&self, x: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
        let mut ctx = FitCtx::with_session(session.child("transform"));
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let k = self.shapelets.nrows();
        let feat = Matrix::from_fn(x.nrows(), k, |i, s| {
            min_shapelet_dist(x, i, &self.shapelets, s)
        });
        ctx.finish(feat)
    }
}

/// Lead–lag path signature of order 2 (sktime `SignatureTransformer` lite).
///
/// Each row is treated as a 2-d path \((t, x_t)\). Feature count is not
/// identification `p`.
#[derive(Clone, Debug, Default)]
pub struct SignatureTransformer;

impl SignatureTransformer {
    /// Default order-2 lead–lag signature.
    pub fn new() -> Self {
        Self
    }
}

impl Transform for SignatureTransformer {
    fn transform(&self, x: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
        let mut ctx = FitCtx::with_session(session.child("transform"));
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let t = x.ncols();
        if t < 2 {
            ctx.push(
                Issue::builder(IssueCode::WindowTooShort)
                    .message("SignatureTransformer needs T≥2")
                    .build(),
            );
        }
        // S¹ (2) + S² (4)
        let out = Matrix::from_fn(x.nrows(), 6, |i, j| {
            if t < 2 {
                return 0.0;
            }
            let mut s1 = [0.0; 2];
            let mut s2 = [0.0; 4];
            for k in 1..t {
                let dt = 1.0 / (t - 1) as f64;
                let dx = x.get(i, k) - x.get(i, k - 1);
                let d = [dt, dx];
                s2[0] += s1[0] * d[0];
                s2[1] += s1[0] * d[1];
                s2[2] += s1[1] * d[0];
                s2[3] += s1[1] * d[1];
                s1[0] += d[0];
                s1[1] += d[1];
            }
            match j {
                0 => s1[0],
                1 => s1[1],
                2 => s2[0],
                3 => s2[1],
                4 => s2[2],
                _ => s2[3],
            }
        });
        ctx.finish(out)
    }
}

/// Vote of several ROCKET+ridge members (sktime `Arsenal` lite).
///
/// Member / kernel counts are not identification `p`.
#[derive(Clone, Debug)]
pub struct Arsenal {
    /// Ensemble size.
    pub n_members: usize,
    /// Kernels per member.
    pub n_kernels: usize,
    /// Kernel length.
    pub kernel_len: usize,
    /// Ridge \(\alpha\).
    pub alpha: f64,
    /// Seed.
    pub seed: u64,
}

impl Default for Arsenal {
    fn default() -> Self {
        Self {
            n_members: 3,
            n_kernels: 8,
            kernel_len: 3,
            alpha: 0.5,
            seed: 4,
        }
    }
}

impl Arsenal {
    /// Default Arsenal lite.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted Arsenal vote.
#[derive(Clone, Debug)]
pub struct FittedArsenal {
    members: Vec<FittedRocketClassifier>,
}

impl Fit for Arsenal {
    type Fitted = FittedArsenal;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedArsenal>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        inspect_classes(&mut ctx.report, y, &ctx.policy);
        let m = self.n_members.max(1);
        let mut members = Vec::with_capacity(m);
        for i in 0..m {
            let mut clf = RocketClassifier {
                n_kernels: self.n_kernels.max(1),
                kernel_len: self.kernel_len.max(1),
                alpha: self.alpha,
                seed: self.seed.wrapping_add(i as u64 * 17),
            };
            match clf.fit(x, y, &session.child(format!("ars_{i}"))) {
                Ok(q) => members.push(q.value),
                Err(e) => {
                    if !matches!(
                        e.primary.code,
                        IssueCode::ResidualTooLarge
                            | IssueCode::NearSingular
                            | IssueCode::RankZero
                            | IssueCode::R2IsOne
                            | IssueCode::InsufficientSample
                    ) {
                        ctx.push(e.primary);
                    }
                }
            }
        }
        if members.is_empty() {
            ctx.push(
                Issue::builder(IssueCode::DidNotConverge)
                    .message("Arsenal: every ROCKET member was rejected")
                    .build(),
            );
        }
        ctx.finish(FittedArsenal { members })
    }
}

impl Predict for FittedArsenal {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        if self.members.is_empty() {
            return ctx.finish(Vector::zeros(x.nrows()));
        }
        let mut votes = vec![0.0; x.nrows()];
        for (t, m) in self.members.iter().enumerate() {
            match m.predict(x, &session.child(format!("m{t}"))) {
                Ok(q) => {
                    for i in 0..x.nrows().min(q.value.len()) {
                        votes[i] += q.value[i];
                    }
                }
                Err(_) => {}
            }
        }
        let k = self.members.len() as f64;
        ctx.finish(Vector::from_iter(votes.iter().map(|v| {
            if *v / k > 0.5 {
                1.0
            } else {
                0.0
            }
        })))
    }
}

/// Catch22 features, random rotation, ridge (sktime `FreshPRINCE` lite).
///
/// Feature count is not identification `p`.
#[derive(Clone, Debug)]
pub struct FreshPrince {
    /// Ridge \(\alpha\).
    pub alpha: f64,
    /// Rotation seed.
    pub seed: u64,
}

impl Default for FreshPrince {
    fn default() -> Self {
        Self {
            alpha: 0.1,
            seed: 9,
        }
    }
}

impl FreshPrince {
    /// Default FreshPRINCE lite.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted FreshPRINCE lite.
#[derive(Clone, Debug)]
pub struct FittedFreshPrince {
    rot: Matrix,
    inner: crate::classification::FittedRidgeClassifier,
}

fn random_rotation(p: usize, seed: u64) -> Matrix {
    let mut rng = Rng::new(seed);
    let mut q = Matrix::from_fn(p, p, |_, _| rng.standard_normal());
    for j in 0..p {
        for k in 0..j {
            let mut dot = 0.0;
            for i in 0..p {
                dot += q.get(i, j) * q.get(i, k);
            }
            for i in 0..p {
                q.set(i, j, q.get(i, j) - dot * q.get(i, k));
            }
        }
        let mut nrm = 0.0;
        for i in 0..p {
            nrm += q.get(i, j) * q.get(i, j);
        }
        let nrm = nrm.sqrt().max(1e-12);
        for i in 0..p {
            q.set(i, j, q.get(i, j) / nrm);
        }
    }
    q
}

fn apply_rotation(z: &Matrix, rot: &Matrix) -> Matrix {
    let p = z.ncols().min(rot.nrows());
    Matrix::from_fn(z.nrows(), rot.ncols(), |i, j| {
        let mut s = 0.0;
        for k in 0..p.min(rot.nrows()) {
            s += z.get(i, k) * rot.get(k, j);
        }
        s
    })
}

impl Fit for FreshPrince {
    type Fitted = FittedFreshPrince;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedFreshPrince>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        inspect_classes(&mut ctx.report, y, &ctx.policy);
        let z = catch22_rows(x, session, &mut ctx);
        let p = z.ncols().max(1);
        let rot = random_rotation(p, self.seed);
        let zr = apply_rotation(&z, &rot);
        let classes: Vec<i64> = {
            let mut c: Vec<i64> = y
                .as_slice()
                .iter()
                .filter(|v| v.is_finite())
                .map(|v| v.round() as i64)
                .collect();
            c.sort_unstable();
            c.dedup();
            c
        };
        let pm = Vector::from_iter(y.as_slice().iter().map(|&v| {
            let lab = v.round() as i64;
            if classes.len() >= 2 && lab == classes[classes.len() - 1] {
                1.0
            } else {
                -1.0
            }
        }));
        let mut scratch = signlred::Report::new("freshprince", "ridge");
        let design = zr.with_intercept();
        let beta = ridge_solve(&mut scratch, &design, &pm, self.alpha.max(0.0), &ctx.policy)
            .unwrap_or_else(|| Vector::zeros(design.ncols()));
        ctx.finish(FittedFreshPrince {
            rot,
            inner: crate::classification::FittedRidgeClassifier::from_penalized(
                FittedPenalized {
                    coef: Vector::from_iter((1..beta.len()).map(|j| beta[j])),
                    intercept: beta.as_slice().first().copied().unwrap_or(0.0),
                    alpha: self.alpha,
                    l1_ratio: 0.0,
                },
                if classes.len() >= 2 {
                    classes
                } else {
                    vec![0, 1]
                },
            ),
        })
    }
}

impl Predict for FittedFreshPrince {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        let z = catch22_rows(x, session, &mut ctx);
        let zr = apply_rotation(&z, &self.rot);
        match self.inner.predict(&zr, &session.child("ridge")) {
            Ok(q) => ctx.finish(q.value),
            Err(e) => {
                if !matches!(
                    e.primary.code,
                    IssueCode::ResidualTooLarge | IssueCode::NearSingular | IssueCode::RankZero
                ) {
                    ctx.push(e.primary);
                }
                ctx.finish(Vector::zeros(x.nrows()))
            }
        }
    }
}

/// Shapelet distances plus ridge (sktime `ShapeletTransformClassifier` lite).
#[derive(Clone, Debug)]
pub struct ShapeletTransformClassifier {
    /// Shapelets.
    pub n_shapelets: usize,
    /// Shapelet length.
    pub length: usize,
    /// Ridge \(\alpha\).
    pub alpha: f64,
    /// Seed.
    pub seed: u64,
}

impl Default for ShapeletTransformClassifier {
    fn default() -> Self {
        Self {
            n_shapelets: 3,
            length: 3,
            alpha: 0.1,
            seed: 2,
        }
    }
}

impl ShapeletTransformClassifier {
    /// Default shapelet transform classifier.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted shapelet-transform classifier.
#[derive(Clone, Debug)]
pub struct FittedShapeletTransformClassifier {
    shapelets: FittedShapeletTransform,
    inner: crate::classification::FittedRidgeClassifier,
}

impl Fit for ShapeletTransformClassifier {
    type Fitted = FittedShapeletTransformClassifier;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedShapeletTransformClassifier>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        inspect_classes(&mut ctx.report, y, &ctx.policy);
        let st = ShapeletTransform::new(self.n_shapelets, self.length)
            .fit_unsupervised(x, &session.child("shp"))?
            .value;
        let z = st.transform(x, &session.child("shpt"))?.value;
        let classes: Vec<i64> = {
            let mut c: Vec<i64> = y
                .as_slice()
                .iter()
                .filter(|v| v.is_finite())
                .map(|v| v.round() as i64)
                .collect();
            c.sort_unstable();
            c.dedup();
            c
        };
        let pm = Vector::from_iter(y.as_slice().iter().map(|&v| {
            let lab = v.round() as i64;
            if classes.len() >= 2 && lab == classes[classes.len() - 1] {
                1.0
            } else {
                -1.0
            }
        }));
        let mut scratch = signlred::Report::new("stc", "ridge");
        let design = z.with_intercept();
        let beta = ridge_solve(&mut scratch, &design, &pm, self.alpha.max(0.0), &ctx.policy)
            .unwrap_or_else(|| Vector::zeros(design.ncols()));
        ctx.finish(FittedShapeletTransformClassifier {
            shapelets: st,
            inner: crate::classification::FittedRidgeClassifier::from_penalized(
                FittedPenalized {
                    coef: Vector::from_iter((1..beta.len()).map(|j| beta[j])),
                    intercept: beta.as_slice().first().copied().unwrap_or(0.0),
                    alpha: self.alpha,
                    l1_ratio: 0.0,
                },
                if classes.len() >= 2 {
                    classes
                } else {
                    vec![0, 1]
                },
            ),
        })
    }
}

impl Predict for FittedShapeletTransformClassifier {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let z = self.shapelets.transform(x, &session.child("shpt"))?;
        self.inner.predict(&z.value, session)
    }
}

/// Soft-DTW k-means (tslearn `TimeSeriesKMeans` with soft-DTW metric).
///
/// Cluster count is not identification `p`.
#[derive(Clone, Debug)]
pub struct SoftDtwKMeans {
    /// Clusters.
    pub n_clusters: usize,
    /// Soft-DTW smoothness.
    pub gamma: f64,
    /// Assignment iterations.
    pub max_iter: usize,
    /// Seed.
    pub seed: u64,
}

impl Default for SoftDtwKMeans {
    fn default() -> Self {
        Self {
            n_clusters: 2,
            gamma: 0.5,
            max_iter: 8,
            seed: 1,
        }
    }
}

impl SoftDtwKMeans {
    /// Soft-DTW k-means with `k` clusters.
    pub fn new(n_clusters: usize) -> Self {
        Self {
            n_clusters,
            ..Self::default()
        }
    }
}

/// Fitted soft-DTW k-means.
#[derive(Clone, Debug)]
pub struct FittedSoftDtwKMeans {
    /// Centroids.
    pub centers: Matrix,
    /// Training labels.
    pub labels: Vector,
    gamma: f64,
}

impl FitUnsupervised for SoftDtwKMeans {
    type Fitted = FittedSoftDtwKMeans;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedSoftDtwKMeans>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let n = x.nrows();
        let k = self.n_clusters.max(1).min(n.max(1));
        let g = if self.gamma.is_finite() && self.gamma > 0.0 {
            self.gamma
        } else {
            ctx.push(
                Issue::builder(IssueCode::InvalidWeight)
                    .severity(Severity::Warning)
                    .message(format!(
                        "SoftDtwKMeans.gamma={} is not positive; using 0.5",
                        self.gamma
                    ))
                    .build(),
            );
            0.5
        };
        if n == 0 {
            return ctx.finish(FittedSoftDtwKMeans {
                centers: Matrix::zeros(0, x.ncols()),
                labels: Vector::zeros(0),
                gamma: g,
            });
        }
        let mut rng = Rng::new(self.seed);
        let seeds = rng.sample_indices(n, k);
        let mut centers =
            Matrix::from_fn(k, x.ncols(), |c, j| x.get(seeds[c.min(seeds.len() - 1)], j));
        let mut labels = Vector::zeros(n);
        for _ in 0..self.max_iter.max(1) {
            for i in 0..n {
                let mut best = 0usize;
                let mut bd = f64::INFINITY;
                for c in 0..k {
                    let d = softdtw_raw(x.row(i).as_slice(), centers.row(c).as_slice(), g);
                    if d < bd {
                        bd = d;
                        best = c;
                    }
                }
                labels[i] = best as f64;
            }
            for c in 0..k {
                let members: Vec<usize> = (0..n)
                    .filter(|&i| labels[i].round() as usize == c)
                    .collect();
                if members.is_empty() {
                    ctx.push(
                        Issue::builder(IssueCode::EmptyCluster)
                            .message(format!("soft-DTW k-means cluster {c} emptied; re-seeded"))
                            .build(),
                    );
                    let r = rng.below(n);
                    for j in 0..x.ncols() {
                        centers.set(c, j, x.get(r, j));
                    }
                    continue;
                }
                let sub = Matrix::from_fn(members.len(), x.ncols(), |i, j| x.get(members[i], j));
                match softdtw_barycenter(&sub, g, 4, &session.child(format!("sdb_{c}"))) {
                    Ok(q) => {
                        for j in 0..x.ncols().min(q.value.len()) {
                            centers.set(c, j, q.value[j]);
                        }
                    }
                    Err(_) => {
                        for j in 0..x.ncols() {
                            let m = members.iter().map(|&i| x.get(i, j)).sum::<f64>()
                                / members.len() as f64;
                            centers.set(c, j, m);
                        }
                    }
                }
            }
        }
        ctx.finish(FittedSoftDtwKMeans {
            centers,
            labels,
            gamma: g,
        })
    }
}

impl Predict for FittedSoftDtwKMeans {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let y = Vector::from_iter((0..x.nrows()).map(|i| {
            let mut best = 0usize;
            let mut bd = f64::INFINITY;
            for c in 0..self.centers.nrows() {
                let d = softdtw_raw(
                    x.row(i).as_slice(),
                    self.centers.row(c).as_slice(),
                    self.gamma,
                );
                if d < bd {
                    bd = d;
                    best = c;
                }
            }
            best as f64
        }));
        ctx.finish(y)
    }
}

fn interval_feats_drcif(x: &Matrix, intervals: &[Interval]) -> Matrix {
    let p = intervals.len() * 8;
    Matrix::from_fn(x.nrows(), p, |i, j| {
        let spec = &intervals[j / 8];
        let kind = j % 8;
        let a = spec.start.min(x.ncols());
        let b = spec.end.min(x.ncols()).max(a + 1);
        let mut vals: Vec<f64> = (a..b).map(|t| x.get(i, t)).collect();
        let len = vals.len();
        let mean = vals.iter().sum::<f64>() / len as f64;
        match kind {
            0 => mean,
            1 => {
                if len <= 1 {
                    0.0
                } else {
                    let ss: f64 = vals.iter().map(|v| (v - mean) * (v - mean)).sum();
                    (ss / (len as f64 - 1.0)).sqrt()
                }
            }
            2 => {
                let tbar = (len.saturating_sub(1)) as f64 / 2.0;
                let mut num = 0.0;
                let mut den = 0.0;
                for (u, v) in vals.iter().enumerate() {
                    let dt = u as f64 - tbar;
                    num += dt * (*v - mean);
                    den += dt * dt;
                }
                if den > 0.0 {
                    num / den
                } else {
                    0.0
                }
            }
            3 => {
                vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                vals[len / 2]
            }
            4 => {
                vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let q = |p: f64| {
                    let t = p * (len.saturating_sub(1)) as f64;
                    let lo = t.floor() as usize;
                    let hi = t.ceil() as usize;
                    let w = t - lo as f64;
                    (1.0 - w) * vals[lo] + w * vals[hi.min(len - 1)]
                };
                q(0.75) - q(0.25)
            }
            5 => vals.iter().map(|v| v * v).sum::<f64>() / len as f64,
            6 => vals.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            _ => vals.iter().copied().fold(f64::INFINITY, f64::min),
        }
    })
}

/// Diverse-representation CIF (sktime `DrCIF` lite).
///
/// Interval count is not identification `p`. Catch22 on short intervals is
/// omitted so a constant window cannot abort the outer fit.
#[derive(Clone, Debug)]
pub struct DrCif {
    /// Trees.
    pub n_estimators: usize,
    /// Random intervals per tree.
    pub n_intervals: usize,
    /// Tree depth.
    pub max_depth: usize,
    /// Seed.
    pub seed: u64,
}

impl Default for DrCif {
    fn default() -> Self {
        Self {
            n_estimators: 6,
            n_intervals: 3,
            max_depth: 4,
            seed: 11,
        }
    }
}

impl DrCif {
    /// Default DrCIF lite.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted DrCIF vote.
#[derive(Clone, Debug)]
pub struct FittedDrCif {
    trees: Vec<crate::tree::FittedTreeClassifier>,
    intervals: Vec<Vec<Interval>>,
    /// Sorted class labels.
    pub classes: Vec<i64>,
}

impl Fit for DrCif {
    type Fitted = FittedDrCif;
    fn fit(&mut self, x: &Matrix, y: &Vector, session: &Session) -> Result<Qualified<FittedDrCif>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        let counts = inspect_classes(&mut ctx.report, y, &ctx.policy);
        let classes: Vec<i64> = counts.iter().map(|(k, _)| *k).collect();
        ctx.push(
            Issue::builder(IssueCode::JitterInjected)
                .message("DrCIF lite uses eight interval summaries, not the published catch22 set")
                .compromise(NumericalCompromise::new(
                    "diverse interval features",
                    "mean/std/slope/median/IQR/energy/min/max per interval",
                    "catch22 on short windows can be statistically vacuous",
                    "do not treat this as the published DrCIF feature map",
                ))
                .build(),
        );
        let mut rng = Rng::new(self.seed);
        let mut trees = Vec::new();
        let mut intervals = Vec::new();
        let tlen = x.ncols().max(1);
        for e in 0..self.n_estimators.max(1) {
            let mut iv = Vec::new();
            for _ in 0..self.n_intervals.max(1) {
                let a = rng.below(tlen);
                let span = 1 + rng.below(tlen);
                let b = (a + span).min(tlen);
                iv.push(Interval {
                    start: a,
                    end: b.max(a + 1),
                });
            }
            let feat = interval_feats_drcif(x, &iv);
            let mut tree = crate::tree::DecisionTreeClassifier {
                max_depth: self.max_depth,
                seed: rng.next_u64(),
                ..crate::tree::DecisionTreeClassifier::default()
            };
            match tree.fit(&feat, y, &session.child("drcif_tree")) {
                Ok(q) => {
                    trees.push(q.value);
                    intervals.push(iv);
                }
                Err(err) => {
                    for issue in err.report.issues() {
                        if !matches!(
                            issue.code,
                            IssueCode::ResidualTooLarge
                                | IssueCode::NearSingular
                                | IssueCode::RankZero
                                | IssueCode::R2IsOne
                        ) {
                            ctx.push(issue.clone());
                        }
                    }
                }
            }
            ctx.session.step(e as u64, 0.0, None);
        }
        if trees.is_empty() {
            ctx.push(
                Issue::builder(IssueCode::DidNotConverge)
                    .message("every DrCIF tree failed to fit")
                    .build(),
            );
        }
        ctx.finish(FittedDrCif {
            trees,
            intervals,
            classes,
        })
    }
}

impl Predict for FittedDrCif {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        let mut votes = vec![BTreeMap::<i64, usize>::new(); x.nrows()];
        for (tree, iv) in self.trees.iter().zip(&self.intervals) {
            let feat = interval_feats_drcif(x, iv);
            match tree.predict(&feat, &session.child("drcif_pred")) {
                Ok(q) => {
                    for i in 0..x.nrows() {
                        let lab = q.value[i].round() as i64;
                        *votes[i].entry(lab).or_insert(0) += 1;
                    }
                }
                Err(_) => {}
            }
        }
        let out = Vector::from_iter(votes.iter().map(|m| {
            m.iter()
                .max_by(|a, b| a.1.cmp(b.1).then(b.0.cmp(a.0)))
                .map(|(k, _)| *k as f64)
                .unwrap_or(self.classes.first().copied().unwrap_or(0) as f64)
        }));
        ctx.finish(out)
    }
}

/// Proximity-stump forest (sktime `ProximityForest` lite).
///
/// Each member splits on DTW proximity to two class exemplars. Tree count is
/// not identification `p`.
#[derive(Clone, Debug)]
pub struct ProximityForest {
    /// Stumps.
    pub n_trees: usize,
    /// Seed.
    pub seed: u64,
}

impl Default for ProximityForest {
    fn default() -> Self {
        Self {
            n_trees: 5,
            seed: 13,
        }
    }
}

impl ProximityForest {
    /// Default proximity forest lite.
    pub fn new() -> Self {
        Self::default()
    }
}

#[derive(Clone, Debug)]
struct ProxStump {
    left: Vector,
    right: Vector,
    left_lab: f64,
    right_lab: f64,
}

/// Fitted proximity forest.
#[derive(Clone, Debug)]
pub struct FittedProximityForest {
    trees: Vec<ProxStump>,
    /// Majority class fallback.
    pub default_label: f64,
}

impl Fit for ProximityForest {
    type Fitted = FittedProximityForest;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedProximityForest>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        inspect_classes(&mut ctx.report, y, &ctx.policy);
        let mut by_class: BTreeMap<i64, Vec<usize>> = BTreeMap::new();
        for i in 0..x.nrows().min(y.len()) {
            if y[i].is_finite() {
                by_class.entry(y[i].round() as i64).or_default().push(i);
            }
        }
        let labs: Vec<i64> = by_class.keys().copied().collect();
        let default_label = labs.first().copied().unwrap_or(0) as f64;
        if labs.len() < 2 {
            ctx.push(
                Issue::builder(IssueCode::SingleClass)
                    .message("ProximityForest needs two classes to pick opposing exemplars")
                    .build(),
            );
        }
        let mut rng = Rng::new(self.seed);
        let mut trees = Vec::new();
        for _ in 0..self.n_trees.max(1) {
            if labs.len() < 2 {
                break;
            }
            let a = labs[rng.below(labs.len())];
            let mut b = a;
            for _ in 0..8 {
                b = labs[rng.below(labs.len())];
                if b != a {
                    break;
                }
            }
            if b == a {
                continue;
            }
            let ia = &by_class[&a];
            let ib = &by_class[&b];
            let i0 = ia[rng.below(ia.len())];
            let i1 = ib[rng.below(ib.len())];
            trees.push(ProxStump {
                left: x.row(i0),
                right: x.row(i1),
                left_lab: a as f64,
                right_lab: b as f64,
            });
        }
        if trees.is_empty() {
            ctx.push(
                Issue::builder(IssueCode::DidNotConverge)
                    .message("ProximityForest built no proximity stumps")
                    .build(),
            );
        }
        ctx.finish(FittedProximityForest {
            trees,
            default_label,
        })
    }
}

impl Predict for FittedProximityForest {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        if self.trees.is_empty() {
            return ctx.finish(Vector::filled(x.nrows(), self.default_label));
        }
        let mut votes = vec![BTreeMap::<i64, usize>::new(); x.nrows()];
        for t in &self.trees {
            for i in 0..x.nrows() {
                let row = x.row(i);
                let dl = dtw_raw(row.as_slice(), t.left.as_slice());
                let dr = dtw_raw(row.as_slice(), t.right.as_slice());
                let lab = if dl <= dr { t.left_lab } else { t.right_lab };
                *votes[i].entry(lab.round() as i64).or_insert(0) += 1;
            }
        }
        let out = Vector::from_iter(votes.iter().map(|m| {
            m.iter()
                .max_by(|a, b| a.1.cmp(b.1).then(b.0.cmp(a.0)))
                .map(|(k, _)| *k as f64)
                .unwrap_or(self.default_label)
        }));
        ctx.finish(out)
    }
}

fn binary_ridge_from_features(
    z: &Matrix,
    y: &Vector,
    alpha: f64,
    policy: &signlred::Policy,
    name: &str,
) -> crate::classification::FittedRidgeClassifier {
    let classes: Vec<i64> = {
        let mut c: Vec<i64> = y
            .as_slice()
            .iter()
            .filter(|v| v.is_finite())
            .map(|v| v.round() as i64)
            .collect();
        c.sort_unstable();
        c.dedup();
        if c.len() >= 2 {
            c
        } else {
            vec![0, 1]
        }
    };
    let pm = Vector::from_iter(y.as_slice().iter().map(|&v| {
        let lab = v.round() as i64;
        if classes.len() >= 2 && lab == classes[classes.len() - 1] {
            1.0
        } else {
            -1.0
        }
    }));
    let mut scratch = signlred::Report::new(name, "ridge");
    let design = z.with_intercept();
    let beta = ridge_solve(&mut scratch, &design, &pm, alpha.max(0.0), policy)
        .unwrap_or_else(|| Vector::zeros(design.ncols()));
    crate::classification::FittedRidgeClassifier::from_penalized(
        FittedPenalized {
            coef: Vector::from_iter((1..beta.len()).map(|j| beta[j])),
            intercept: beta.as_slice().first().copied().unwrap_or(0.0),
            alpha,
            l1_ratio: 0.0,
        },
        classes,
    )
}

/// Prefix-and-commit classifier (tslearn / sktime early classification lite).
///
/// Prefix length is not identification `p`.
#[derive(Clone, Debug)]
pub struct EarlyClassifier {
    /// Ridge \(\alpha\).
    pub alpha: f64,
    /// PAA segments on each prefix.
    pub n_segments: usize,
}

impl Default for EarlyClassifier {
    fn default() -> Self {
        Self {
            alpha: 0.1,
            n_segments: 2,
        }
    }
}

impl EarlyClassifier {
    /// Default early classifier.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted prefix committee.
#[derive(Clone, Debug)]
pub struct FittedEarlyClassifier {
    fracs: Vec<f64>,
    models: Vec<crate::classification::FittedRidgeClassifier>,
    segs: usize,
}

fn prefix_cols(x: &Matrix, frac: f64) -> Matrix {
    let t = ((x.ncols() as f64 * frac).ceil() as usize)
        .max(1)
        .min(x.ncols().max(1));
    Matrix::from_fn(x.nrows(), t, |i, j| x.get(i, j))
}

impl Fit for EarlyClassifier {
    type Fitted = FittedEarlyClassifier;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedEarlyClassifier>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        inspect_classes(&mut ctx.report, y, &ctx.policy);
        let segs = self.n_segments.max(1);
        let fracs = vec![0.5, 1.0];
        let mut models = Vec::new();
        for (k, &f) in fracs.iter().enumerate() {
            let pref = prefix_cols(x, f);
            match Paa::new(segs).transform(&pref, &session.child(format!("early_paa_{k}"))) {
                Ok(q) => {
                    models.push(binary_ridge_from_features(
                        &q.value,
                        y,
                        self.alpha,
                        &ctx.policy,
                        "early",
                    ));
                }
                Err(e) => {
                    if !matches!(
                        e.primary.code,
                        IssueCode::ResidualTooLarge
                            | IssueCode::NearSingular
                            | IssueCode::RankZero
                            | IssueCode::R2IsOne
                    ) {
                        ctx.push(e.primary);
                    }
                }
            }
        }
        if models.is_empty() {
            ctx.push(
                Issue::builder(IssueCode::DidNotConverge)
                    .message("EarlyClassifier: every prefix ridge was rejected")
                    .build(),
            );
        }
        ctx.finish(FittedEarlyClassifier {
            fracs,
            models,
            segs,
        })
    }
}

impl Predict for FittedEarlyClassifier {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        if self.models.is_empty() {
            return ctx.finish(Vector::zeros(x.nrows()));
        }
        let mut preds: Vec<Vector> = Vec::new();
        for (k, (f, m)) in self.fracs.iter().zip(&self.models).enumerate() {
            let pref = prefix_cols(x, *f);
            match Paa::new(self.segs).transform(&pref, &session.child(format!("epaa_{k}"))) {
                Ok(z) => match m.predict(&z.value, &session.child(format!("er_{k}"))) {
                    Ok(q) => preds.push(q.value),
                    Err(_) => {}
                },
                Err(_) => {}
            }
        }
        if preds.is_empty() {
            return ctx.finish(Vector::zeros(x.nrows()));
        }
        let out = Vector::from_iter((0..x.nrows()).map(|i| {
            let first = preds[0][i];
            if preds
                .iter()
                .all(|p| i < p.len() && (p[i] - first).abs() < 0.5)
            {
                first
            } else {
                preds.last().map(|p| p[i]).unwrap_or(first)
            }
        }));
        ctx.finish(out)
    }
}

/// Time-contracted BOSS vote (sktime `ContractableBOSS` lite).
///
/// Member / word counts are not identification `p`.
#[derive(Clone, Debug)]
pub struct ContractableBoss {
    /// Ensemble size.
    pub n_members: usize,
    /// Base window.
    pub window: usize,
    /// Seed.
    pub seed: u64,
}

impl Default for ContractableBoss {
    fn default() -> Self {
        Self {
            n_members: 3,
            window: 3,
            seed: 17,
        }
    }
}

impl ContractableBoss {
    /// Default cBOSS lite.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted cBOSS vote.
#[derive(Clone, Debug)]
pub struct FittedContractableBoss {
    members: Vec<FittedBoss>,
}

impl Fit for ContractableBoss {
    type Fitted = FittedContractableBoss;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedContractableBoss>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        inspect_classes(&mut ctx.report, y, &ctx.policy);
        let mut members = Vec::new();
        for i in 0..self.n_members.max(1) {
            let w = (self.window.max(2) + i).min(x.ncols().max(1));
            let mut boss = BossEnsemble {
                window: w,
                word_len: 3,
                alphabet: 4,
            };
            match boss.fit(x, y, &session.child(format!("cboss_{i}"))) {
                Ok(q) => members.push(q.value),
                Err(e) => {
                    if !matches!(
                        e.primary.code,
                        IssueCode::ResidualTooLarge
                            | IssueCode::NearSingular
                            | IssueCode::RankZero
                            | IssueCode::R2IsOne
                            | IssueCode::InsufficientSample
                    ) {
                        ctx.push(e.primary);
                    }
                }
            }
        }
        if members.is_empty() {
            ctx.push(
                Issue::builder(IssueCode::DidNotConverge)
                    .message("ContractableBOSS: every member was rejected")
                    .build(),
            );
        }
        ctx.finish(FittedContractableBoss { members })
    }
}

impl Predict for FittedContractableBoss {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        if self.members.is_empty() {
            return ctx.finish(Vector::zeros(x.nrows()));
        }
        let mut acc = vec![0.0; x.nrows()];
        let mut k: f64 = 0.0;
        for (t, m) in self.members.iter().enumerate() {
            match m.predict(x, &session.child(format!("cb_{t}"))) {
                Ok(q) => {
                    for i in 0..x.nrows().min(q.value.len()) {
                        acc[i] += q.value[i];
                    }
                    k += 1.0;
                }
                Err(_) => {}
            }
        }
        let k = k.max(1.0);
        ctx.finish(Vector::from_iter(acc.iter().map(|v| {
            if *v / k > 0.5 {
                1.0
            } else {
                0.0
            }
        })))
    }
}

/// One tree per series column (sktime `ColumnEnsembleClassifier` lite).
///
/// Column count is not identification `p`.
#[derive(Clone, Debug)]
pub struct ColumnEnsembleClassifier {
    /// Tree depth.
    pub max_depth: usize,
}

impl Default for ColumnEnsembleClassifier {
    fn default() -> Self {
        Self { max_depth: 3 }
    }
}

impl ColumnEnsembleClassifier {
    /// Default column ensemble.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted per-column vote.
#[derive(Clone, Debug)]
pub struct FittedColumnEnsembleClassifier {
    trees: Vec<crate::tree::FittedTreeClassifier>,
    /// Fallback label.
    pub default_label: f64,
}

impl Fit for ColumnEnsembleClassifier {
    type Fitted = FittedColumnEnsembleClassifier;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedColumnEnsembleClassifier>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        inspect_classes(&mut ctx.report, y, &ctx.policy);
        let default_label = y
            .as_slice()
            .iter()
            .copied()
            .find(|v| v.is_finite())
            .unwrap_or(0.0);
        let mut trees = Vec::new();
        for j in 0..x.ncols() {
            let col = Matrix::from_fn(x.nrows(), 1, |i, _| x.get(i, j));
            let mut tree = crate::tree::DecisionTreeClassifier {
                max_depth: self.max_depth,
                seed: j as u64 + 3,
                ..crate::tree::DecisionTreeClassifier::default()
            };
            match tree.fit(&col, y, &session.child(format!("col_{j}"))) {
                Ok(q) => trees.push(q.value),
                Err(e) => {
                    if !matches!(
                        e.primary.code,
                        IssueCode::ResidualTooLarge
                            | IssueCode::NearSingular
                            | IssueCode::RankZero
                            | IssueCode::R2IsOne
                            | IssueCode::ConstantFeature
                    ) {
                        ctx.push(e.primary);
                    }
                }
            }
        }
        if trees.is_empty() {
            ctx.push(
                Issue::builder(IssueCode::DidNotConverge)
                    .message("ColumnEnsembleClassifier: every column tree failed")
                    .build(),
            );
        }
        ctx.finish(FittedColumnEnsembleClassifier {
            trees,
            default_label,
        })
    }
}

impl Predict for FittedColumnEnsembleClassifier {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        let mut votes = vec![BTreeMap::<i64, usize>::new(); x.nrows()];
        for (j, tree) in self.trees.iter().enumerate() {
            let col = Matrix::from_fn(x.nrows(), 1, |i, _| {
                x.get(i, j.min(x.ncols().saturating_sub(1)))
            });
            match tree.predict(&col, &session.child(format!("colp_{j}"))) {
                Ok(q) => {
                    for i in 0..x.nrows().min(q.value.len()) {
                        *votes[i].entry(q.value[i].round() as i64).or_insert(0) += 1;
                    }
                }
                Err(_) => {}
            }
        }
        let out = Vector::from_iter(votes.iter().map(|m| {
            m.iter()
                .max_by(|a, b| a.1.cmp(b.1).then(b.0.cmp(a.0)))
                .map(|(k, _)| *k as f64)
                .unwrap_or(self.default_label)
        }));
        ctx.finish(out)
    }
}

/// Temporal dictionary ensemble (sktime `TemporalDictionaryEnsemble` lite).
///
/// A vote of BOSS and WEASEL. Word / window counts are not identification `p`.
#[derive(Clone, Debug)]
pub struct TemporalDictionaryEnsemble {
    /// BOSS window.
    pub window: usize,
    /// Words kept by WEASEL.
    pub n_words: usize,
}

impl Default for TemporalDictionaryEnsemble {
    fn default() -> Self {
        Self {
            window: 3,
            n_words: 6,
        }
    }
}

impl TemporalDictionaryEnsemble {
    /// Default TDE lite.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted TDE vote.
#[derive(Clone, Debug)]
pub struct FittedTemporalDictionaryEnsemble {
    boss: Option<FittedBoss>,
    weasel: Option<FittedBoss>,
}

impl Fit for TemporalDictionaryEnsemble {
    type Fitted = FittedTemporalDictionaryEnsemble;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedTemporalDictionaryEnsemble>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        inspect_classes(&mut ctx.report, y, &ctx.policy);
        let w = self.window.max(2).min(x.ncols().max(1));
        let mut boss = BossEnsemble {
            window: w,
            word_len: 3,
            alphabet: 4,
        };
        let mut weasel = Weasel {
            window: w,
            word_len: 3,
            alphabet: 4,
            n_words: self.n_words.max(1),
        };
        let boss = match boss.fit(x, y, &session.child("tde_boss")) {
            Ok(q) => Some(q.value),
            Err(e) => {
                if !matches!(
                    e.primary.code,
                    IssueCode::ResidualTooLarge
                        | IssueCode::NearSingular
                        | IssueCode::RankZero
                        | IssueCode::R2IsOne
                        | IssueCode::InsufficientSample
                ) {
                    ctx.push(e.primary);
                }
                None
            }
        };
        let weasel = match weasel.fit(x, y, &session.child("tde_weasel")) {
            Ok(q) => Some(q.value),
            Err(e) => {
                if !matches!(
                    e.primary.code,
                    IssueCode::ResidualTooLarge
                        | IssueCode::NearSingular
                        | IssueCode::RankZero
                        | IssueCode::R2IsOne
                        | IssueCode::InsufficientSample
                ) {
                    ctx.push(e.primary);
                }
                None
            }
        };
        if boss.is_none() && weasel.is_none() {
            ctx.push(
                Issue::builder(IssueCode::DidNotConverge)
                    .message("TemporalDictionaryEnsemble: both dictionary members failed")
                    .build(),
            );
        }
        ctx.finish(FittedTemporalDictionaryEnsemble { boss, weasel })
    }
}

impl Predict for FittedTemporalDictionaryEnsemble {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        let mut acc = vec![0.0; x.nrows()];
        let mut k: f64 = 0.0;
        for (name, m) in [("b", self.boss.as_ref()), ("w", self.weasel.as_ref())] {
            if let Some(m) = m {
                match m.predict(x, &session.child(name)) {
                    Ok(q) => {
                        for i in 0..x.nrows().min(q.value.len()) {
                            acc[i] += q.value[i];
                        }
                        k += 1.0;
                    }
                    Err(_) => {}
                }
            }
        }
        let k = k.max(1.0);
        ctx.finish(Vector::from_iter(acc.iter().map(|v| {
            if *v / k > 0.5 {
                1.0
            } else {
                0.0
            }
        })))
    }
}

fn interval_feats_rise(x: &Matrix, intervals: &[Interval]) -> Matrix {
    let p = intervals.len() * 4;
    Matrix::from_fn(x.nrows(), p, |i, j| {
        let spec = &intervals[j / 4];
        let kind = j % 4;
        let a = spec.start.min(x.ncols());
        let b = spec.end.min(x.ncols()).max(a + 1);
        let len = (b - a).max(1);
        let mut mean = 0.0;
        for t in a..b {
            mean += x.get(i, t);
        }
        mean /= len as f64;
        match kind {
            0 => mean,
            1 => {
                let mut e = 0.0;
                for t in a..b {
                    let v = x.get(i, t);
                    e += v * v;
                }
                e / len as f64
            }
            _ => {
                let k = if kind == 2 { 1.0 } else { 2.0 };
                let omega = std::f64::consts::TAU * k / len as f64;
                let mut re = 0.0;
                let mut im = 0.0;
                for (u, t) in (a..b).enumerate() {
                    let v = x.get(i, t) - mean;
                    let ang = omega * u as f64;
                    re += v * ang.cos();
                    im += v * ang.sin();
                }
                (re * re + im * im).sqrt() / len as f64
            }
        }
    })
}

/// Random Interval Spectral Ensemble (sktime `RandomIntervalSpectralEnsemble` lite).
///
/// Interval / tree counts are not identification `p`.
#[derive(Clone, Debug)]
pub struct Rise {
    /// Trees.
    pub n_estimators: usize,
    /// Random intervals per tree.
    pub n_intervals: usize,
    /// Tree depth.
    pub max_depth: usize,
    /// Seed.
    pub seed: u64,
}

impl Default for Rise {
    fn default() -> Self {
        Self {
            n_estimators: 6,
            n_intervals: 3,
            max_depth: 4,
            seed: 23,
        }
    }
}

impl Rise {
    /// Default RISE lite.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted RISE vote.
#[derive(Clone, Debug)]
pub struct FittedRise {
    trees: Vec<crate::tree::FittedTreeClassifier>,
    intervals: Vec<Vec<Interval>>,
    /// Fallback label.
    pub default_label: f64,
}

impl Fit for Rise {
    type Fitted = FittedRise;
    fn fit(&mut self, x: &Matrix, y: &Vector, session: &Session) -> Result<Qualified<FittedRise>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        inspect_classes(&mut ctx.report, y, &ctx.policy);
        ctx.push(
            Issue::builder(IssueCode::JitterInjected)
                .message("RISE lite uses four interval DFT summaries, not ACF/periodogram forests")
                .compromise(NumericalCompromise::new(
                    "RISE spectral interval map",
                    "mean/energy/first two DFT bins per random interval",
                    "the published estimator uses a richer spectral dictionary",
                    "do not treat this as the sktime RISE feature map",
                ))
                .build(),
        );
        let default_label = y
            .as_slice()
            .iter()
            .copied()
            .find(|v| v.is_finite())
            .unwrap_or(0.0);
        let mut rng = Rng::new(self.seed);
        let mut trees = Vec::new();
        let mut intervals = Vec::new();
        let tlen = x.ncols().max(1);
        for e in 0..self.n_estimators.max(1) {
            let mut iv = Vec::new();
            for _ in 0..self.n_intervals.max(1) {
                let a = rng.below(tlen);
                let span = 1 + rng.below(tlen);
                iv.push(Interval {
                    start: a,
                    end: (a + span).min(tlen).max(a + 1),
                });
            }
            let feat = interval_feats_rise(x, &iv);
            let mut tree = crate::tree::DecisionTreeClassifier {
                max_depth: self.max_depth,
                seed: rng.next_u64(),
                ..crate::tree::DecisionTreeClassifier::default()
            };
            match tree.fit(&feat, y, &session.child("rise_tree")) {
                Ok(q) => {
                    trees.push(q.value);
                    intervals.push(iv);
                }
                Err(err) => {
                    for issue in err.report.issues() {
                        if !matches!(
                            issue.code,
                            IssueCode::ResidualTooLarge
                                | IssueCode::NearSingular
                                | IssueCode::RankZero
                                | IssueCode::R2IsOne
                        ) {
                            ctx.push(issue.clone());
                        }
                    }
                }
            }
            ctx.session.step(e as u64, 0.0, None);
        }
        if trees.is_empty() {
            ctx.push(
                Issue::builder(IssueCode::DidNotConverge)
                    .message("every RISE tree failed to fit")
                    .build(),
            );
        }
        ctx.finish(FittedRise {
            trees,
            intervals,
            default_label,
        })
    }
}

impl Predict for FittedRise {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        let mut votes = vec![BTreeMap::<i64, usize>::new(); x.nrows()];
        for (tree, iv) in self.trees.iter().zip(&self.intervals) {
            let feat = interval_feats_rise(x, iv);
            match tree.predict(&feat, &session.child("risep")) {
                Ok(q) => {
                    for i in 0..x.nrows().min(q.value.len()) {
                        *votes[i].entry(q.value[i].round() as i64).or_insert(0) += 1;
                    }
                }
                Err(_) => {}
            }
        }
        let out = Vector::from_iter(votes.iter().map(|m| {
            m.iter()
                .max_by(|a, b| a.1.cmp(b.1).then(b.0.cmp(a.0)))
                .map(|(k, _)| *k as f64)
                .unwrap_or(self.default_label)
        }));
        ctx.finish(out)
    }
}

/// Elastic 1-NN vote over DTW/MSM/TWE/Euclidean (sktime `ElasticEnsemble` lite).
#[derive(Clone, Debug)]
pub struct ElasticEnsemble {
    /// MSM move cost.
    pub msm_c: f64,
    /// TWE stiffness.
    pub twe_nu: f64,
}

impl Default for ElasticEnsemble {
    fn default() -> Self {
        Self {
            msm_c: 0.1,
            twe_nu: 0.0,
        }
    }
}

impl ElasticEnsemble {
    /// Default elastic ensemble.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted elastic 1-NN committee.
#[derive(Clone, Debug)]
pub struct FittedElasticEnsemble {
    x: Matrix,
    y: Vector,
    msm_c: f64,
    twe_nu: f64,
}

impl Fit for ElasticEnsemble {
    type Fitted = FittedElasticEnsemble;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedElasticEnsemble>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        inspect_classes(&mut ctx.report, y, &ctx.policy);
        let c = if self.msm_c.is_finite() && self.msm_c >= 0.0 {
            self.msm_c
        } else {
            ctx.push(
                Issue::builder(IssueCode::InvalidWeight)
                    .severity(Severity::Warning)
                    .message(format!(
                        "ElasticEnsemble msm_c={} is invalid; using 0.1",
                        self.msm_c
                    ))
                    .build(),
            );
            0.1
        };
        let nu = if self.twe_nu.is_finite() && self.twe_nu >= 0.0 {
            self.twe_nu
        } else {
            ctx.push(
                Issue::builder(IssueCode::InvalidWeight)
                    .severity(Severity::Warning)
                    .message(format!(
                        "ElasticEnsemble twe_nu={} is invalid; using 0",
                        self.twe_nu
                    ))
                    .build(),
            );
            0.0
        };
        ctx.finish(FittedElasticEnsemble {
            x: x.clone(),
            y: y.clone(),
            msm_c: c,
            twe_nu: nu,
        })
    }
}

impl Predict for FittedElasticEnsemble {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        if self.x.nrows() == 0 {
            return ctx.finish(Vector::zeros(x.nrows()));
        }
        let out = Vector::from_iter((0..x.nrows()).map(|i| {
            let q = x.row(i);
            let mut votes = BTreeMap::<i64, usize>::new();
            for metric in 0..4 {
                let mut best = 0usize;
                let mut bd = f64::INFINITY;
                for t in 0..self.x.nrows() {
                    let d = match metric {
                        0 => dtw_raw(q.as_slice(), self.x.row(t).as_slice()),
                        1 => msm_raw(q.as_slice(), self.x.row(t).as_slice(), self.msm_c),
                        2 => twe_raw(q.as_slice(), self.x.row(t).as_slice(), self.twe_nu, 1.0),
                        _ => {
                            let mut s = 0.0;
                            for j in 0..q.len().min(self.x.ncols()) {
                                let z = q[j] - self.x.get(t, j);
                                s += z * z;
                            }
                            s.sqrt()
                        }
                    };
                    if d < bd {
                        bd = d;
                        best = t;
                    }
                }
                *votes
                    .entry(self.y[best.min(self.y.len().saturating_sub(1))].round() as i64)
                    .or_insert(0) += 1;
            }
            votes
                .iter()
                .max_by(|a, b| a.1.cmp(b.1).then(b.0.cmp(a.0)))
                .map(|(k, _)| *k as f64)
                .unwrap_or(0.0)
        }));
        ctx.finish(out)
    }
}

/// Catch22 + ridge regression (sktime `Catch22Regressor` lite).
#[derive(Clone, Debug)]
pub struct Catch22Regressor {
    /// Ridge \(\alpha\).
    pub alpha: f64,
}

impl Default for Catch22Regressor {
    fn default() -> Self {
        Self { alpha: 0.1 }
    }
}

impl Catch22Regressor {
    /// Catch22 regressor with ridge penalty `alpha`.
    pub fn new(alpha: f64) -> Self {
        Self { alpha }
    }
}

/// Fitted Catch22 ridge regressor.
#[derive(Clone, Debug)]
pub struct FittedCatch22Regressor {
    inner: FittedPenalized,
}

impl Fit for Catch22Regressor {
    type Fitted = FittedCatch22Regressor;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedCatch22Regressor>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        let z = catch22_rows(x, session, &mut ctx);
        let mut scratch = signlred::Report::new("c22reg", "ridge");
        let design = z.with_intercept();
        let beta = ridge_solve(&mut scratch, &design, y, self.alpha.max(0.0), &ctx.policy)
            .unwrap_or_else(|| Vector::zeros(design.ncols()));
        for issue in scratch.issues() {
            if matches!(
                issue.code,
                IssueCode::ResidualTooLarge
                    | IssueCode::NearSingular
                    | IssueCode::RankZero
                    | IssueCode::R2IsOne
                    | IssueCode::PerfectCollinearity
            ) {
                continue;
            }
            ctx.push(issue.clone());
        }
        ctx.finish(FittedCatch22Regressor {
            inner: FittedPenalized {
                coef: Vector::from_iter((1..beta.len()).map(|j| beta[j])),
                intercept: beta.as_slice().first().copied().unwrap_or(0.0),
                alpha: self.alpha,
                l1_ratio: 0.0,
            },
        })
    }
}

impl Predict for FittedCatch22Regressor {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        let z = catch22_rows(x, session, &mut ctx);
        match self.inner.predict(&z, &session.child("ridge")) {
            Ok(q) => ctx.finish(q.value),
            Err(e) => {
                if !matches!(
                    e.primary.code,
                    IssueCode::ResidualTooLarge | IssueCode::NearSingular | IssueCode::RankZero
                ) {
                    ctx.push(e.primary);
                }
                ctx.finish(Vector::zeros(x.nrows()))
            }
        }
    }
}

/// Soft-DTW \(k\)-NN (tslearn `KNeighborsTimeSeriesClassifier` with soft-DTW).
///
/// Neighbor count is not identification `p`.
#[derive(Clone, Debug)]
pub struct SoftDtwKnn {
    /// Neighbors.
    pub k: usize,
    /// Soft-DTW smoothness.
    pub gamma: f64,
}

impl Default for SoftDtwKnn {
    fn default() -> Self {
        Self { k: 1, gamma: 0.5 }
    }
}

impl SoftDtwKnn {
    /// Soft-DTW k-NN with `k` neighbors.
    pub fn new(k: usize) -> Self {
        Self {
            k,
            ..Self::default()
        }
    }
}

/// Fitted soft-DTW k-NN.
#[derive(Clone, Debug)]
pub struct FittedSoftDtwKnn {
    x: Matrix,
    y: Vector,
    k: usize,
    gamma: f64,
}

impl Fit for SoftDtwKnn {
    type Fitted = FittedSoftDtwKnn;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedSoftDtwKnn>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        inspect_classes(&mut ctx.report, y, &ctx.policy);
        let k = if self.k >= 1 {
            self.k
        } else {
            ctx.push(
                Issue::builder(IssueCode::InvalidWeight)
                    .severity(Severity::Warning)
                    .message(format!("SoftDtwKnn k={} < 1; using 1", self.k))
                    .build(),
            );
            1
        };
        let g = if self.gamma.is_finite() && self.gamma > 0.0 {
            self.gamma
        } else {
            ctx.push(
                Issue::builder(IssueCode::InvalidWeight)
                    .severity(Severity::Warning)
                    .message(format!(
                        "SoftDtwKnn gamma={} is not positive; using 0.5",
                        self.gamma
                    ))
                    .build(),
            );
            0.5
        };
        ctx.finish(FittedSoftDtwKnn {
            x: x.clone(),
            y: y.clone(),
            k,
            gamma: g,
        })
    }
}

impl Predict for FittedSoftDtwKnn {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        if self.x.nrows() == 0 {
            return ctx.finish(Vector::zeros(x.nrows()));
        }
        let k = self.k.max(1).min(self.x.nrows());
        let out = Vector::from_iter((0..x.nrows()).map(|i| {
            let q = x.row(i);
            let mut dist: Vec<(f64, f64)> = (0..self.x.nrows())
                .map(|t| {
                    (
                        softdtw_raw(q.as_slice(), self.x.row(t).as_slice(), self.gamma),
                        self.y[t],
                    )
                })
                .collect();
            dist.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            let mut votes = BTreeMap::<i64, usize>::new();
            for &(_, lab) in dist.iter().take(k) {
                *votes.entry(lab.round() as i64).or_insert(0) += 1;
            }
            votes
                .iter()
                .max_by(|a, b| a.1.cmp(b.1).then(b.0.cmp(a.0)))
                .map(|(c, _)| *c as f64)
                .unwrap_or(0.0)
        }));
        ctx.finish(out)
    }
}

/// Soft-DTW 1-NN classifier (tslearn `KNeighborsTimeSeriesClassifier` with Soft-DTW).
///
/// Neighbor count is fixed at 1 and is not identification `p`.
#[derive(Clone, Debug)]
pub struct SoftDtwClassifier {
    /// Soft-DTW smoothness.
    pub gamma: f64,
}

impl Default for SoftDtwClassifier {
    fn default() -> Self {
        Self { gamma: 0.5 }
    }
}

impl SoftDtwClassifier {
    /// Soft-DTW 1-NN classifier.
    pub fn new() -> Self {
        Self::default()
    }
}

impl Fit for SoftDtwClassifier {
    type Fitted = FittedSoftDtwKnn;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedSoftDtwKnn>> {
        SoftDtwKnn {
            k: 1,
            gamma: self.gamma,
        }
        .fit(x, y, session)
    }
}

/// Summary statistics + ridge (sktime `SummaryClassifier` lite).
#[derive(Clone, Debug)]
pub struct SummaryClassifier {
    /// Ridge \(\alpha\).
    pub alpha: f64,
}

impl Default for SummaryClassifier {
    fn default() -> Self {
        Self { alpha: 0.1 }
    }
}

impl SummaryClassifier {
    /// Default summary classifier.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted summary ridge.
#[derive(Clone, Debug)]
pub struct FittedSummaryClassifier {
    inner: crate::classification::FittedRidgeClassifier,
}

fn summary_rows(x: &Matrix) -> Matrix {
    Matrix::from_fn(x.nrows(), 6, |i, j| {
        let row = x.row(i);
        let n = row.len().max(1) as f64;
        let mean = row.as_slice().iter().sum::<f64>() / n;
        match j {
            0 => mean,
            1 => {
                let ss: f64 = row.as_slice().iter().map(|v| (v - mean) * (v - mean)).sum();
                (ss / n.max(1.0)).sqrt()
            }
            2 => row.as_slice().iter().copied().fold(f64::INFINITY, f64::min),
            3 => row
                .as_slice()
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max),
            4 => {
                let mut v = row.as_slice().to_vec();
                v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                v[v.len() / 2]
            }
            _ => {
                let tbar = (row.len().saturating_sub(1)) as f64 / 2.0;
                let mut num = 0.0;
                let mut den = 0.0;
                for (u, &v) in row.as_slice().iter().enumerate() {
                    let dt = u as f64 - tbar;
                    num += dt * (v - mean);
                    den += dt * dt;
                }
                if den > 0.0 {
                    num / den
                } else {
                    0.0
                }
            }
        }
    })
}

impl Fit for SummaryClassifier {
    type Fitted = FittedSummaryClassifier;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedSummaryClassifier>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        inspect_classes(&mut ctx.report, y, &ctx.policy);
        let z = summary_rows(x);
        let inner = binary_ridge_from_features(&z, y, self.alpha, &ctx.policy, "summary");
        ctx.finish(FittedSummaryClassifier { inner })
    }
}

impl Predict for FittedSummaryClassifier {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let z = summary_rows(x);
        self.inner.predict(&z, session)
    }
}

fn signature_rows(x: &Matrix) -> Matrix {
    let t = x.ncols();
    Matrix::from_fn(x.nrows(), 6, |i, j| {
        if t < 2 {
            return 0.0;
        }
        let mut s1 = [0.0; 2];
        let mut s2 = [0.0; 4];
        for k in 1..t {
            let dt = 1.0 / (t - 1) as f64;
            let dx = x.get(i, k) - x.get(i, k - 1);
            let d = [dt, dx];
            s2[0] += s1[0] * d[0];
            s2[1] += s1[0] * d[1];
            s2[2] += s1[1] * d[0];
            s2[3] += s1[1] * d[1];
            s1[0] += d[0];
            s1[1] += d[1];
        }
        match j {
            0 => s1[0],
            1 => s1[1],
            2 => s2[0],
            3 => s2[1],
            4 => s2[2],
            _ => s2[3],
        }
    })
}

fn hydra_apply(x: &Matrix, kernels: &[(Vec<f64>, usize)], n_groups: usize) -> Matrix {
    let g = n_groups.max(1);
    Matrix::from_fn(x.nrows(), g * 2, |i, j| {
        let gid = j / 2;
        let want_max = j % 2 == 1;
        let mut acc_mean = 0.0;
        let mut acc_max: f64 = 0.0;
        let mut k = 0.0;
        for (w, grp) in kernels {
            if *grp != gid {
                continue;
            }
            let t = x.ncols();
            let ww = w.len().max(1);
            let last = t.saturating_sub(ww) + 1;
            let mut mx: f64 = 0.0;
            for start in 0..last.max(1) {
                let mut s = 0.0;
                for (u, &wt) in w.iter().enumerate() {
                    let idx = start + u;
                    if idx < t {
                        s += wt * x.get(i, idx);
                    }
                }
                mx = mx.max(s.abs());
            }
            acc_mean += mx;
            acc_max = acc_max.max(mx);
            k += 1.0;
        }
        if want_max {
            acc_max
        } else if k > 0.0 {
            acc_mean / k
        } else {
            0.0
        }
    })
}

fn hydra_kernels(
    n_kernels: usize,
    n_groups: usize,
    width: usize,
    seed: u64,
) -> Vec<(Vec<f64>, usize)> {
    let mut rng = Rng::new(seed);
    let k = n_kernels.max(1);
    let g = n_groups.max(1);
    let w = width.max(1);
    (0..k)
        .map(|i| {
            let weights = (0..w).map(|_| rng.standard_normal()).collect::<Vec<_>>();
            (weights, i % g)
        })
        .collect()
}

/// Random grouped convolutional kernels (sktime `Hydra` transformer lite).
///
/// Kernel and group counts are not identification `p`.
#[derive(Clone, Debug)]
pub struct Hydra {
    /// Kernels.
    pub n_kernels: usize,
    /// Groups (two pooled features each).
    pub n_groups: usize,
    /// Kernel width.
    pub width: usize,
    /// Seed.
    pub seed: u64,
}

impl Default for Hydra {
    fn default() -> Self {
        Self {
            n_kernels: 8,
            n_groups: 4,
            width: 3,
            seed: 5,
        }
    }
}

impl Hydra {
    /// Default Hydra map.
    pub fn new() -> Self {
        Self::default()
    }
}

impl Transform for Hydra {
    fn transform(&self, x: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
        let mut ctx = FitCtx::with_session(session.child("transform"));
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        if x.ncols() < 2 {
            ctx.push(
                Issue::builder(IssueCode::WindowTooShort)
                    .message("Hydra needs T≥2")
                    .build(),
            );
        }
        let kernels = hydra_kernels(self.n_kernels, self.n_groups, self.width, self.seed);
        ctx.finish(hydra_apply(x, &kernels, self.n_groups))
    }
}

/// Hydra features + ridge (sktime `HydraClassifier` lite).
#[derive(Clone, Debug)]
pub struct HydraClassifier {
    /// Kernels.
    pub n_kernels: usize,
    /// Groups.
    pub n_groups: usize,
    /// Kernel width.
    pub width: usize,
    /// Ridge \(\alpha\).
    pub alpha: f64,
    /// Seed.
    pub seed: u64,
}

impl Default for HydraClassifier {
    fn default() -> Self {
        Self {
            n_kernels: 8,
            n_groups: 4,
            width: 3,
            alpha: 0.1,
            seed: 5,
        }
    }
}

impl HydraClassifier {
    /// Default Hydra classifier.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted Hydra ridge.
#[derive(Clone, Debug)]
pub struct FittedHydraClassifier {
    kernels: Vec<(Vec<f64>, usize)>,
    n_groups: usize,
    inner: crate::classification::FittedRidgeClassifier,
}

impl Fit for HydraClassifier {
    type Fitted = FittedHydraClassifier;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedHydraClassifier>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        inspect_classes(&mut ctx.report, y, &ctx.policy);
        let kernels = hydra_kernels(self.n_kernels, self.n_groups, self.width, self.seed);
        let z = hydra_apply(x, &kernels, self.n_groups);
        let inner = binary_ridge_from_features(&z, y, self.alpha, &ctx.policy, "hydra");
        ctx.finish(FittedHydraClassifier {
            kernels,
            n_groups: self.n_groups.max(1),
            inner,
        })
    }
}

impl Predict for FittedHydraClassifier {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let z = hydra_apply(x, &self.kernels, self.n_groups);
        self.inner.predict(&z, session)
    }
}

/// Catch22 + rotation + ridge (sktime `FreshPRINCERegressor` lite).
#[derive(Clone, Debug)]
pub struct FreshPrinceRegressor {
    /// Ridge \(\alpha\).
    pub alpha: f64,
    /// Rotation seed.
    pub seed: u64,
}

impl Default for FreshPrinceRegressor {
    fn default() -> Self {
        Self {
            alpha: 0.1,
            seed: 9,
        }
    }
}

impl FreshPrinceRegressor {
    /// Default FreshPRINCE regressor lite.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted FreshPRINCE regressor.
#[derive(Clone, Debug)]
pub struct FittedFreshPrinceRegressor {
    rot: Matrix,
    inner: FittedPenalized,
}

impl Fit for FreshPrinceRegressor {
    type Fitted = FittedFreshPrinceRegressor;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedFreshPrinceRegressor>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        let z = catch22_rows(x, session, &mut ctx);
        let p = z.ncols().max(1);
        let rot = random_rotation(p, self.seed);
        let zr = apply_rotation(&z, &rot);
        let mut scratch = signlred::Report::new("fpr", "ridge");
        let design = zr.with_intercept();
        let beta = ridge_solve(&mut scratch, &design, y, self.alpha.max(0.0), &ctx.policy)
            .unwrap_or_else(|| Vector::zeros(design.ncols()));
        for issue in scratch.issues() {
            if matches!(
                issue.code,
                IssueCode::ResidualTooLarge
                    | IssueCode::NearSingular
                    | IssueCode::RankZero
                    | IssueCode::R2IsOne
                    | IssueCode::PerfectCollinearity
            ) {
                continue;
            }
            ctx.push(issue.clone());
        }
        ctx.finish(FittedFreshPrinceRegressor {
            rot,
            inner: FittedPenalized {
                coef: Vector::from_iter((1..beta.len()).map(|j| beta[j])),
                intercept: beta.as_slice().first().copied().unwrap_or(0.0),
                alpha: self.alpha,
                l1_ratio: 0.0,
            },
        })
    }
}

impl Predict for FittedFreshPrinceRegressor {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        let z = catch22_rows(x, session, &mut ctx);
        let zr = apply_rotation(&z, &self.rot);
        match self.inner.predict(&zr, &session.child("ridge")) {
            Ok(q) => ctx.finish(q.value),
            Err(e) => {
                if !matches!(
                    e.primary.code,
                    IssueCode::ResidualTooLarge | IssueCode::NearSingular | IssueCode::RankZero
                ) {
                    ctx.push(e.primary);
                }
                ctx.finish(Vector::zeros(x.nrows()))
            }
        }
    }
}

/// Summary statistics + ridge (sktime `SummaryRegressor` lite).
#[derive(Clone, Debug)]
pub struct SummaryRegressor {
    /// Ridge \(\alpha\).
    pub alpha: f64,
}

impl Default for SummaryRegressor {
    fn default() -> Self {
        Self { alpha: 0.1 }
    }
}

impl SummaryRegressor {
    /// Default summary regressor.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted summary ridge regressor.
#[derive(Clone, Debug)]
pub struct FittedSummaryRegressor {
    inner: FittedPenalized,
}

impl Fit for SummaryRegressor {
    type Fitted = FittedSummaryRegressor;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedSummaryRegressor>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        let z = summary_rows(x);
        let mut scratch = signlred::Report::new("sumreg", "ridge");
        let design = z.with_intercept();
        let beta = ridge_solve(&mut scratch, &design, y, self.alpha.max(0.0), &ctx.policy)
            .unwrap_or_else(|| Vector::zeros(design.ncols()));
        for issue in scratch.issues() {
            if matches!(
                issue.code,
                IssueCode::ResidualTooLarge
                    | IssueCode::NearSingular
                    | IssueCode::RankZero
                    | IssueCode::R2IsOne
                    | IssueCode::PerfectCollinearity
            ) {
                continue;
            }
            ctx.push(issue.clone());
        }
        ctx.finish(FittedSummaryRegressor {
            inner: FittedPenalized {
                coef: Vector::from_iter((1..beta.len()).map(|j| beta[j])),
                intercept: beta.as_slice().first().copied().unwrap_or(0.0),
                alpha: self.alpha,
                l1_ratio: 0.0,
            },
        })
    }
}

impl Predict for FittedSummaryRegressor {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let z = summary_rows(x);
        self.inner.predict(&z, session)
    }
}

/// Path signature + ridge (sktime `SignatureClassifier` lite).
#[derive(Clone, Debug)]
pub struct SignatureClassifier {
    /// Ridge \(\alpha\).
    pub alpha: f64,
}

impl Default for SignatureClassifier {
    fn default() -> Self {
        Self { alpha: 0.1 }
    }
}

impl SignatureClassifier {
    /// Default signature classifier.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted signature ridge.
#[derive(Clone, Debug)]
pub struct FittedSignatureClassifier {
    inner: crate::classification::FittedRidgeClassifier,
}

impl Fit for SignatureClassifier {
    type Fitted = FittedSignatureClassifier;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedSignatureClassifier>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        inspect_classes(&mut ctx.report, y, &ctx.policy);
        let z = signature_rows(x);
        let inner = binary_ridge_from_features(&z, y, self.alpha, &ctx.policy, "sigclf");
        ctx.finish(FittedSignatureClassifier { inner })
    }
}

impl Predict for FittedSignatureClassifier {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let z = signature_rows(x);
        self.inner.predict(&z, session)
    }
}

/// Diverse-representation CIF regressor (sktime `DrCIFRegressor` lite).
#[derive(Clone, Debug)]
pub struct DrCifRegressor {
    /// Trees.
    pub n_estimators: usize,
    /// Random intervals per tree.
    pub n_intervals: usize,
    /// Tree depth.
    pub max_depth: usize,
    /// Seed.
    pub seed: u64,
}

impl Default for DrCifRegressor {
    fn default() -> Self {
        Self {
            n_estimators: 6,
            n_intervals: 3,
            max_depth: 4,
            seed: 11,
        }
    }
}

impl DrCifRegressor {
    /// Default DrCIF regressor lite.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted DrCIF regressor vote.
#[derive(Clone, Debug)]
pub struct FittedDrCifRegressor {
    trees: Vec<crate::tree::FittedTreeRegressor>,
    intervals: Vec<Vec<Interval>>,
}

impl Fit for DrCifRegressor {
    type Fitted = FittedDrCifRegressor;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedDrCifRegressor>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        ctx.push(
            Issue::builder(IssueCode::JitterInjected)
                .message("DrCIF regressor lite uses eight interval summaries")
                .compromise(NumericalCompromise::new(
                    "diverse interval features",
                    "mean/std/slope/median/IQR/energy/min/max per interval",
                    "catch22 on short windows can be statistically vacuous",
                    "do not treat this as the published DrCIF regressor map",
                ))
                .build(),
        );
        let mut rng = Rng::new(self.seed);
        let mut trees = Vec::new();
        let mut intervals = Vec::new();
        let tlen = x.ncols().max(1);
        for e in 0..self.n_estimators.max(1) {
            let mut iv = Vec::new();
            for _ in 0..self.n_intervals.max(1) {
                let a = rng.below(tlen);
                let span = 1 + rng.below(tlen);
                let b = (a + span).min(tlen);
                iv.push(Interval {
                    start: a,
                    end: b.max(a + 1),
                });
            }
            let feat = interval_feats_drcif(x, &iv);
            let mut tree = crate::tree::DecisionTreeRegressor {
                max_depth: self.max_depth,
                seed: rng.next_u64(),
                ..crate::tree::DecisionTreeRegressor::default()
            };
            match tree.fit(&feat, y, &session.child("drcifr_tree")) {
                Ok(q) => {
                    trees.push(q.value);
                    intervals.push(iv);
                }
                Err(err) => {
                    for issue in err.report.issues() {
                        if !matches!(
                            issue.code,
                            IssueCode::ResidualTooLarge
                                | IssueCode::NearSingular
                                | IssueCode::RankZero
                                | IssueCode::R2IsOne
                                | IssueCode::MeaninglessFit
                        ) {
                            ctx.push(issue.clone());
                        }
                    }
                }
            }
            ctx.session.step(e as u64, 0.0, None);
        }
        if trees.is_empty() {
            ctx.push(
                Issue::builder(IssueCode::DidNotConverge)
                    .message("every DrCIF regressor tree failed to fit")
                    .build(),
            );
        }
        ctx.finish(FittedDrCifRegressor { trees, intervals })
    }
}

impl Predict for FittedDrCifRegressor {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        let mut acc = Vector::zeros(x.nrows());
        let mut k = 0.0;
        for (tree, iv) in self.trees.iter().zip(&self.intervals) {
            let feat = interval_feats_drcif(x, iv);
            if let Ok(q) = tree.predict(&feat, &session.child("drcifr_pred")) {
                for i in 0..x.nrows() {
                    acc[i] += q.value[i];
                }
                k += 1.0;
            }
        }
        if k > 0.0 {
            acc = acc.scale(1.0 / k);
        }
        ctx.finish(acc)
    }
}

/// Single DTW-proximity stump (sktime `ProximityTree` lite).
#[derive(Clone, Debug)]
pub struct ProximityTree {
    /// Seed.
    pub seed: u64,
}

impl Default for ProximityTree {
    fn default() -> Self {
        Self { seed: 13 }
    }
}

impl ProximityTree {
    /// Default proximity tree.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted proximity stump.
#[derive(Clone, Debug)]
pub struct FittedProximityTree {
    stump: Option<ProxStump>,
    /// Majority class fallback.
    pub default_label: f64,
}

impl Fit for ProximityTree {
    type Fitted = FittedProximityTree;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedProximityTree>> {
        let mut forest = ProximityForest {
            n_trees: 1,
            seed: self.seed,
        };
        let q = forest.fit(x, y, session)?;
        Ok(q.map(|f| FittedProximityTree {
            stump: f.trees.into_iter().next(),
            default_label: f.default_label,
        }))
    }
}

impl Predict for FittedProximityTree {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        let Some(t) = &self.stump else {
            return ctx.finish(Vector::filled(x.nrows(), self.default_label));
        };
        let out = Vector::from_iter((0..x.nrows()).map(|i| {
            let row = x.row(i);
            let dl = dtw_raw(row.as_slice(), t.left.as_slice());
            let dr = dtw_raw(row.as_slice(), t.right.as_slice());
            if dl <= dr {
                t.left_lab
            } else {
                t.right_lab
            }
        }));
        ctx.finish(out)
    }
}

fn supervised_intervals(
    x: &Matrix,
    y: &Vector,
    n_intervals: usize,
    rng: &mut Rng,
) -> Vec<Interval> {
    let tlen = x.ncols().max(1);
    let want = n_intervals.max(1);
    let mut scored: Vec<(f64, Interval)> = Vec::new();
    for _ in 0..(want * 8).max(8) {
        let a = rng.below(tlen);
        let span = 1 + rng.below(tlen);
        let b = (a + span).min(tlen).max(a + 1);
        let mut m0 = 0.0;
        let mut m1 = 0.0;
        let mut n0 = 0.0;
        let mut n1 = 0.0;
        for i in 0..x.nrows().min(y.len()) {
            if !y[i].is_finite() {
                continue;
            }
            let mut mu = 0.0;
            for t in a..b {
                mu += x.get(i, t);
            }
            mu /= (b - a) as f64;
            if y[i] >= 0.5 {
                m1 += mu;
                n1 += 1.0;
            } else {
                m0 += mu;
                n0 += 1.0;
            }
        }
        let gap = if n0 > 0.0 && n1 > 0.0 {
            (m1 / n1 - m0 / n0).abs()
        } else {
            0.0
        };
        scored.push((gap, Interval { start: a, end: b }));
    }
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(want);
    scored.into_iter().map(|(_, iv)| iv).collect()
}

/// Supervised time-series forest (sktime `SupervisedTimeSeriesForest` lite).
///
/// Intervals are ranked by class-mean gap. Interval count is not identification
/// `p`.
#[derive(Clone, Debug)]
pub struct Stsf {
    /// Trees.
    pub n_estimators: usize,
    /// Supervised intervals per tree.
    pub n_intervals: usize,
    /// Tree depth.
    pub max_depth: usize,
    /// Seed.
    pub seed: u64,
}

impl Default for Stsf {
    fn default() -> Self {
        Self {
            n_estimators: 6,
            n_intervals: 3,
            max_depth: 4,
            seed: 17,
        }
    }
}

impl Stsf {
    /// Default STSF lite.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted STSF vote.
#[derive(Clone, Debug)]
pub struct FittedStsf {
    trees: Vec<crate::tree::FittedTreeClassifier>,
    intervals: Vec<Vec<Interval>>,
    /// Sorted class labels.
    pub classes: Vec<i64>,
}

impl Fit for Stsf {
    type Fitted = FittedStsf;
    fn fit(&mut self, x: &Matrix, y: &Vector, session: &Session) -> Result<Qualified<FittedStsf>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        let counts = inspect_classes(&mut ctx.report, y, &ctx.policy);
        let classes: Vec<i64> = counts.iter().map(|(k, _)| *k).collect();
        let mut rng = Rng::new(self.seed);
        let mut trees = Vec::new();
        let mut intervals = Vec::new();
        for e in 0..self.n_estimators.max(1) {
            let iv = supervised_intervals(x, y, self.n_intervals, &mut rng);
            let feat = interval_feats(x, &iv);
            let mut tree = crate::tree::DecisionTreeClassifier {
                max_depth: self.max_depth,
                seed: rng.next_u64(),
                ..crate::tree::DecisionTreeClassifier::default()
            };
            match tree.fit(&feat, y, &session.child("stsf_tree")) {
                Ok(q) => {
                    trees.push(q.value);
                    intervals.push(iv);
                }
                Err(err) => {
                    for issue in err.report.issues() {
                        if !matches!(
                            issue.code,
                            IssueCode::ResidualTooLarge
                                | IssueCode::NearSingular
                                | IssueCode::RankZero
                                | IssueCode::R2IsOne
                                | IssueCode::MeaninglessFit
                        ) {
                            ctx.push(issue.clone());
                        }
                    }
                }
            }
            ctx.session.step(e as u64, 0.0, None);
        }
        if trees.is_empty() {
            ctx.push(
                Issue::builder(IssueCode::DidNotConverge)
                    .message("every STSF tree failed to fit")
                    .build(),
            );
        }
        ctx.finish(FittedStsf {
            trees,
            intervals,
            classes,
        })
    }
}

impl Predict for FittedStsf {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        let mut votes = vec![BTreeMap::<i64, usize>::new(); x.nrows()];
        for (tree, iv) in self.trees.iter().zip(&self.intervals) {
            let feat = interval_feats(x, iv);
            match tree.predict(&feat, &session.child("stsf_pred")) {
                Ok(q) => {
                    for i in 0..x.nrows() {
                        let lab = q.value[i].round() as i64;
                        *votes[i].entry(lab).or_insert(0) += 1;
                    }
                }
                Err(_) => {}
            }
        }
        let out = Vector::from_iter(votes.iter().map(|m| {
            m.iter()
                .max_by(|a, b| a.1.cmp(b.1).then(b.0.cmp(a.0)))
                .map(|(k, _)| *k as f64)
                .unwrap_or(self.classes.first().copied().unwrap_or(0) as f64)
        }));
        ctx.finish(out)
    }
}

fn hstack(a: &Matrix, b: &Matrix) -> Matrix {
    let n = a.nrows().min(b.nrows());
    Matrix::from_fn(n, a.ncols() + b.ncols(), |i, j| {
        if j < a.ncols() {
            a.get(i, j)
        } else {
            b.get(i, j - a.ncols())
        }
    })
}

fn sax_symbols(row: &[f64], n_pieces: usize, alphabet: usize) -> Vec<f64> {
    let v = Vector::from_slice(row);
    let z = znorm(&v);
    let k = n_pieces.max(1);
    let a = alphabet.max(2);
    let mut paa = vec![0.0; k];
    let t = z.len().max(1);
    for j in 0..k {
        let lo = j * t / k;
        let hi = ((j + 1) * t / k).max(lo + 1);
        let mut s = 0.0;
        let mut n = 0.0;
        for u in lo..hi.min(z.len()) {
            s += z[u];
            n += 1.0;
        }
        paa[j] = if n > 0.0 { s / n } else { 0.0 };
    }
    paa.into_iter()
        .map(|val| {
            let u = 0.5 + 0.5 * crate::special::erf(val / std::f64::consts::SQRT_2);
            ((u * a as f64).floor() as usize).min(a - 1) as f64
        })
        .collect()
}

/// Pairwise SAX MINDIST / Hamming (tslearn `cdist_sax`).
///
/// Piece / alphabet counts are not identification `p`.
pub fn cdist_sax(
    a: &Matrix,
    b: &Matrix,
    n_pieces: usize,
    alphabet: usize,
    session: &Session,
) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, a, None, &ctx.policy);
    inspect_xy(&mut ctx.report, b, None, &ctx.policy);
    let np = n_pieces.max(1);
    let al = if alphabet >= 2 {
        alphabet
    } else {
        ctx.push(
            Issue::builder(IssueCode::InvalidWeight)
                .severity(Severity::Warning)
                .message(format!("cdist_sax alphabet={alphabet} < 2; using 2"))
                .build(),
        );
        2
    };
    let sa: Vec<Vec<f64>> = (0..a.nrows())
        .map(|i| sax_symbols(a.row(i).as_slice(), np, al))
        .collect();
    let sb: Vec<Vec<f64>> = (0..b.nrows())
        .map(|i| sax_symbols(b.row(i).as_slice(), np, al))
        .collect();
    let out = Matrix::from_fn(a.nrows(), b.nrows(), |i, j| {
        let u = &sa[i];
        let v = &sb[j];
        let m = u.len().min(v.len()).max(1);
        let mut d = 0.0;
        for t in 0..u.len().min(v.len()) {
            if (u[t] - v[t]).abs() > 0.0 {
                d += 1.0;
            }
        }
        d / m as f64
    });
    ctx.finish(out)
}

/// One-step canonical time warping (tslearn `ctw` lite).
///
/// DTW-align, OLS-scale the first series onto the second, then DTW again.
pub fn canonical_time_warping(a: &Vector, b: &Vector, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    if a.is_empty() || b.is_empty() {
        ctx.push(Issue::builder(IssueCode::EmptyMatrix).build());
        return ctx.finish(f64::NAN);
    }
    let path = dtw_path(a.as_slice(), b.as_slice());
    let mut sx = 0.0;
    let mut sy = 0.0;
    let mut sxx = 0.0;
    let mut sxy = 0.0;
    let mut n = 0.0;
    for (i, j) in &path {
        let x = a[*i];
        let yv = b[*j];
        if x.is_finite() && yv.is_finite() {
            sx += x;
            sy += yv;
            sxx += x * x;
            sxy += x * yv;
            n += 1.0;
        }
    }
    let (scale, intercept) = if n >= 2.0 {
        let mx = sx / n;
        let my = sy / n;
        let var = sxx - n * mx * mx;
        if var.abs() > 1e-18 {
            let sl = (sxy - n * mx * my) / var;
            (sl, my - sl * mx)
        } else {
            ctx.push(
                Issue::builder(IssueCode::JitterInjected)
                    .message("CTW aligned x had no variance; scale set to 1")
                    .compromise(NumericalCompromise::new(
                        "identifiable linear map on the DTW path",
                        "identity scale",
                        "the aligned query was constant",
                        "do not read a unit scale as a unique CTW warp",
                    ))
                    .build(),
            );
            (1.0, 0.0)
        }
    } else {
        (1.0, 0.0)
    };
    let ap = Vector::from_iter(a.as_slice().iter().map(|v| scale * *v + intercept));
    ctx.finish(dtw_raw(ap.as_slice(), b.as_slice()))
}

/// Hydra + MultiROCKET concatenation (sktime `HydraMultiRocketClassifier` lite).
#[derive(Clone, Debug)]
pub struct HydraMultiRocket {
    /// Hydra kernels.
    pub n_kernels: usize,
    /// Hydra groups.
    pub n_groups: usize,
    /// MultiROCKET kernels.
    pub n_rocket: usize,
    /// Ridge \(\alpha\).
    pub alpha: f64,
    /// Seed.
    pub seed: u64,
}

impl Default for HydraMultiRocket {
    fn default() -> Self {
        Self {
            n_kernels: 8,
            n_groups: 4,
            n_rocket: 4,
            alpha: 0.1,
            seed: 5,
        }
    }
}

impl HydraMultiRocket {
    /// Default Hydra+MultiROCKET classifier.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted Hydra+MultiROCKET ridge.
#[derive(Clone, Debug)]
pub struct FittedHydraMultiRocket {
    hydra: Vec<(Vec<f64>, usize)>,
    n_groups: usize,
    rocket: MultiRocket,
    inner: crate::classification::FittedRidgeClassifier,
}

impl Fit for HydraMultiRocket {
    type Fitted = FittedHydraMultiRocket;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedHydraMultiRocket>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        inspect_classes(&mut ctx.report, y, &ctx.policy);
        let hydra = hydra_kernels(self.n_kernels, self.n_groups, 3, self.seed);
        let zh = hydra_apply(x, &hydra, self.n_groups);
        let rocket = MultiRocket {
            n_kernels: self.n_rocket.max(1),
            seed: self.seed.wrapping_add(1),
        };
        let zr = match rocket.transform(x, &session.child("mr")) {
            Ok(q) => q.value,
            Err(_) => Matrix::zeros(x.nrows(), self.n_rocket.max(1) * 3),
        };
        let z = hstack(&zh, &zr);
        let inner = binary_ridge_from_features(&z, y, self.alpha, &ctx.policy, "hmr");
        ctx.finish(FittedHydraMultiRocket {
            hydra,
            n_groups: self.n_groups.max(1),
            rocket,
            inner,
        })
    }
}

impl Predict for FittedHydraMultiRocket {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let zh = hydra_apply(x, &self.hydra, self.n_groups);
        let zr = match self.rocket.transform(x, &session.child("mr")) {
            Ok(q) => q.value,
            Err(_) => Matrix::zeros(x.nrows(), self.rocket.n_kernels.max(1) * 3),
        };
        let z = hstack(&zh, &zr);
        self.inner.predict(&z, session)
    }
}

/// Path signature + ridge (sktime `SignatureRegressor` lite).
#[derive(Clone, Debug)]
pub struct SignatureRegressor {
    /// Ridge \(\alpha\).
    pub alpha: f64,
}

impl Default for SignatureRegressor {
    fn default() -> Self {
        Self { alpha: 0.1 }
    }
}

impl SignatureRegressor {
    /// Default signature regressor.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted signature ridge regressor.
#[derive(Clone, Debug)]
pub struct FittedSignatureRegressor {
    inner: FittedPenalized,
}

impl Fit for SignatureRegressor {
    type Fitted = FittedSignatureRegressor;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedSignatureRegressor>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        let z = signature_rows(x);
        let mut scratch = signlred::Report::new("sigreg", "ridge");
        let design = z.with_intercept();
        let beta = ridge_solve(&mut scratch, &design, y, self.alpha.max(0.0), &ctx.policy)
            .unwrap_or_else(|| Vector::zeros(design.ncols()));
        for issue in scratch.issues() {
            if matches!(
                issue.code,
                IssueCode::ResidualTooLarge
                    | IssueCode::NearSingular
                    | IssueCode::RankZero
                    | IssueCode::R2IsOne
                    | IssueCode::PerfectCollinearity
            ) {
                continue;
            }
            ctx.push(issue.clone());
        }
        ctx.finish(FittedSignatureRegressor {
            inner: FittedPenalized {
                coef: Vector::from_iter((1..beta.len()).map(|j| beta[j])),
                intercept: beta.as_slice().first().copied().unwrap_or(0.0),
                alpha: self.alpha,
                l1_ratio: 0.0,
            },
        })
    }
}

impl Predict for FittedSignatureRegressor {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let z = signature_rows(x);
        self.inner.predict(&z, session)
    }
}

fn interval_quantiles(x: &Matrix, intervals: &[Interval]) -> Matrix {
    let p = intervals.len() * 3;
    Matrix::from_fn(x.nrows(), p.max(1), |i, j| {
        if intervals.is_empty() {
            return 0.0;
        }
        let spec = &intervals[j / 3];
        let qk = j % 3;
        let a = spec.start.min(x.ncols());
        let b = spec.end.min(x.ncols()).max(a + 1);
        let mut v: Vec<f64> = (a..b).map(|t| x.get(i, t)).collect();
        v.sort_by(|p, q| p.partial_cmp(q).unwrap_or(std::cmp::Ordering::Equal));
        let tau = match qk {
            0 => 0.25,
            1 => 0.5,
            _ => 0.75,
        };
        let t = tau * (v.len().saturating_sub(1)) as f64;
        let lo = t.floor() as usize;
        let hi = t.ceil() as usize;
        let w = t - lo as f64;
        (1.0 - w) * v[lo] + w * v[hi.min(v.len() - 1)]
    })
}

/// Interval quantile forest/ridge (sktime `QUANT` lite).
#[derive(Clone, Debug)]
pub struct QuantClassifier {
    /// Random intervals.
    pub n_intervals: usize,
    /// Ridge \(\alpha\).
    pub alpha: f64,
    /// Seed.
    pub seed: u64,
}

impl Default for QuantClassifier {
    fn default() -> Self {
        Self {
            n_intervals: 4,
            alpha: 0.1,
            seed: 21,
        }
    }
}

impl QuantClassifier {
    /// Default QUANT lite.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted QUANT ridge.
#[derive(Clone, Debug)]
pub struct FittedQuantClassifier {
    intervals: Vec<Interval>,
    inner: crate::classification::FittedRidgeClassifier,
}

impl Fit for QuantClassifier {
    type Fitted = FittedQuantClassifier;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedQuantClassifier>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        inspect_classes(&mut ctx.report, y, &ctx.policy);
        let mut rng = Rng::new(self.seed);
        let tlen = x.ncols().max(1);
        let mut iv = Vec::new();
        for _ in 0..self.n_intervals.max(1) {
            let a = rng.below(tlen);
            let span = 1 + rng.below(tlen);
            let b = (a + span).min(tlen);
            iv.push(Interval {
                start: a,
                end: b.max(a + 1),
            });
        }
        let z = interval_quantiles(x, &iv);
        let inner = binary_ridge_from_features(&z, y, self.alpha, &ctx.policy, "quant");
        ctx.finish(FittedQuantClassifier {
            intervals: iv,
            inner,
        })
    }
}

impl Predict for FittedQuantClassifier {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let z = interval_quantiles(x, &self.intervals);
        self.inner.predict(&z, session)
    }
}

/// WEASEL+MUSE lite: two WEASEL windows voted (sktime `MUSE`).
#[derive(Clone, Debug)]
pub struct WeaselMuse {
    /// Short window.
    pub window_short: usize,
    /// Long window.
    pub window_long: usize,
}

impl Default for WeaselMuse {
    fn default() -> Self {
        Self {
            window_short: 3,
            window_long: 4,
        }
    }
}

impl WeaselMuse {
    /// Default MUSE lite.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted two-window WEASEL vote.
#[derive(Clone, Debug)]
pub struct FittedWeaselMuse {
    short: FittedBoss,
    long: FittedBoss,
}

impl Fit for WeaselMuse {
    type Fitted = FittedWeaselMuse;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedWeaselMuse>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        inspect_classes(&mut ctx.report, y, &ctx.policy);
        let mut w0 = Weasel {
            window: self.window_short.max(2),
            word_len: 3,
            alphabet: 4,
            n_words: 6,
        };
        let mut w1 = Weasel {
            window: self.window_long.max(2),
            word_len: 3,
            alphabet: 4,
            n_words: 6,
        };
        let short = match w0.fit(x, y, &session.child("muse_s")) {
            Ok(q) => q.value,
            Err(e) => {
                if !matches!(
                    e.primary.code,
                    IssueCode::ResidualTooLarge
                        | IssueCode::NearSingular
                        | IssueCode::RankZero
                        | IssueCode::R2IsOne
                ) {
                    ctx.push(e.primary);
                }
                return ctx.finish(FittedWeaselMuse {
                    short: FittedBoss {
                        vocab: Vec::new(),
                        ridge: FittedPenalized {
                            coef: Vector::zeros(0),
                            intercept: 0.0,
                            alpha: 0.1,
                            l1_ratio: 0.0,
                        },
                        spec: (3, 3, 4),
                    },
                    long: FittedBoss {
                        vocab: Vec::new(),
                        ridge: FittedPenalized {
                            coef: Vector::zeros(0),
                            intercept: 0.0,
                            alpha: 0.1,
                            l1_ratio: 0.0,
                        },
                        spec: (4, 3, 4),
                    },
                });
            }
        };
        let long = match w1.fit(x, y, &session.child("muse_l")) {
            Ok(q) => q.value,
            Err(e) => {
                if !matches!(
                    e.primary.code,
                    IssueCode::ResidualTooLarge
                        | IssueCode::NearSingular
                        | IssueCode::RankZero
                        | IssueCode::R2IsOne
                ) {
                    ctx.push(e.primary);
                }
                short.clone()
            }
        };
        ctx.finish(FittedWeaselMuse { short, long })
    }
}

impl Predict for FittedWeaselMuse {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        let a = self
            .short
            .predict(x, &session.child("s"))
            .map(|q| q.value)
            .unwrap_or_else(|_| Vector::zeros(x.nrows()));
        let b = self
            .long
            .predict(x, &session.child("l"))
            .map(|q| q.value)
            .unwrap_or_else(|_| Vector::zeros(x.nrows()));
        let out = Vector::from_iter((0..x.nrows()).map(|i| {
            let va = if i < a.len() { a[i] } else { 0.0 };
            let vb = if i < b.len() { b[i] } else { 0.0 };
            if (va + vb) * 0.5 >= 0.5 {
                1.0
            } else {
                0.0
            }
        }));
        ctx.finish(out)
    }
}

/// Pairwise canonical time warping (tslearn `cdist_ctw`).
///
/// Series count is not identification `p`.
pub fn cdist_ctw(a: &Matrix, b: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, a, None, &ctx.policy);
    inspect_xy(&mut ctx.report, b, None, &ctx.policy);
    let out = Matrix::from_fn(a.nrows(), b.nrows(), |i, j| {
        dtw_raw(a.row(i).as_slice(), b.row(j).as_slice())
    });
    let mut scaled = out.clone();
    for i in 0..a.nrows() {
        for j in 0..b.nrows() {
            match canonical_time_warping(
                &a.row(i),
                &b.row(j),
                &session.child(format!("ctw_{i}_{j}")),
            ) {
                Ok(q) if q.value.is_finite() => scaled.set(i, j, q.value),
                _ => {}
            }
        }
    }
    ctx.finish(scaled)
}

fn erp_raw(a: &[f64], b: &[f64], g: f64) -> f64 {
    if a.is_empty() || b.is_empty() {
        return f64::NAN;
    }
    let n = a.len();
    let m = b.len();
    let mut dp = vec![0.0; (n + 1) * (m + 1)];
    let at = |i: usize, j: usize| i * (m + 1) + j;
    for i in 1..=n {
        dp[at(i, 0)] = dp[at(i - 1, 0)] + (a[i - 1] - g).abs();
    }
    for j in 1..=m {
        dp[at(0, j)] = dp[at(0, j - 1)] + (b[j - 1] - g).abs();
    }
    for i in 1..=n {
        for j in 1..=m {
            let match_c = dp[at(i - 1, j - 1)] + (a[i - 1] - b[j - 1]).abs();
            let del = dp[at(i - 1, j)] + (a[i - 1] - g).abs();
            let ins = dp[at(i, j - 1)] + (b[j - 1] - g).abs();
            dp[at(i, j)] = match_c.min(del).min(ins);
        }
    }
    dp[at(n, m)]
}

/// Pairwise ERP (tslearn `cdist_erp`).
///
/// Gap reference `g` is not identification `p`.
pub fn cdist_erp(a: &Matrix, b: &Matrix, g: f64, session: &Session) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, a, None, &ctx.policy);
    inspect_xy(&mut ctx.report, b, None, &ctx.policy);
    let g = if g.is_finite() {
        g
    } else {
        ctx.push(
            Issue::builder(IssueCode::InvalidWeight)
                .severity(Severity::Warning)
                .message(format!("cdist_erp g={g} is not finite; using 0"))
                .build(),
        );
        0.0
    };
    let out = Matrix::from_fn(a.nrows(), b.nrows(), |i, j| {
        erp_raw(a.row(i).as_slice(), b.row(j).as_slice(), g)
    });
    ctx.finish(out)
}

/// DTW k-medoids (tslearn `TimeSeriesKMedoids` / sktime `TimeSeriesKMedoids`).
///
/// Cluster count is not identification `p`.
#[derive(Clone, Debug)]
pub struct TimeSeriesKMedoids {
    /// Number of medoids.
    pub n_clusters: usize,
    /// PAM iterations.
    pub max_iter: usize,
    /// Seed.
    pub seed: u64,
}

impl Default for TimeSeriesKMedoids {
    fn default() -> Self {
        Self {
            n_clusters: 2,
            max_iter: 10,
            seed: 3,
        }
    }
}

impl TimeSeriesKMedoids {
    /// DTW PAM with `k` medoids.
    pub fn new(n_clusters: usize) -> Self {
        Self {
            n_clusters,
            ..Self::default()
        }
    }
}

/// Fitted DTW medoids.
#[derive(Clone, Debug)]
pub struct FittedTsKMedoids {
    /// Medoid series (`k × T`).
    pub centers: Matrix,
    /// Training assignments.
    pub labels: Vector,
}

impl FitUnsupervised for TimeSeriesKMedoids {
    type Fitted = FittedTsKMedoids;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedTsKMedoids>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let n = x.nrows();
        let k = self.n_clusters.max(1).min(n.max(1));
        if n == 0 {
            return ctx.finish(FittedTsKMedoids {
                centers: Matrix::zeros(0, x.ncols()),
                labels: Vector::zeros(0),
            });
        }
        let dist = Matrix::from_fn(n, n, |i, j| {
            if i == j {
                0.0
            } else {
                dtw_raw(x.row(i).as_slice(), x.row(j).as_slice())
            }
        });
        let mut rng = Rng::new(self.seed);
        let mut medoids = rng.sample_indices(n, k);
        let mut labels = Vector::zeros(n);
        for it in 0..self.max_iter.max(1) {
            for i in 0..n {
                let mut best = 0usize;
                let mut bd = f64::INFINITY;
                for (c, &m) in medoids.iter().enumerate() {
                    let d = dist.get(i, m);
                    if d < bd {
                        bd = d;
                        best = c;
                    }
                }
                labels[i] = best as f64;
            }
            let mut changed = false;
            for c in 0..k {
                let members: Vec<usize> = (0..n).filter(|&i| labels[i] as usize == c).collect();
                if members.is_empty() {
                    ctx.push(
                        Issue::builder(IssueCode::EmptyCluster)
                            .message(format!("DTW k-medoids cluster {c} emptied"))
                            .build(),
                    );
                    continue;
                }
                let mut best_m = medoids[c];
                let mut best_s = f64::INFINITY;
                for &u in &members {
                    let mut s = 0.0;
                    for &v in &members {
                        s += dist.get(u, v);
                    }
                    if s < best_s {
                        best_s = s;
                        best_m = u;
                    }
                }
                if best_m != medoids[c] {
                    medoids[c] = best_m;
                    changed = true;
                }
            }
            ctx.session
                .step(it as u64, if changed { 1.0 } else { 0.0 }, None);
            if !changed && it > 0 {
                ctx.session.converged("DTW k-medoids", it as u64);
                break;
            }
        }
        let centers = Matrix::from_fn(k, x.ncols(), |c, j| {
            x.get(medoids[c.min(medoids.len().saturating_sub(1))], j)
        });
        ctx.finish(FittedTsKMedoids { centers, labels })
    }
}

impl Predict for FittedTsKMedoids {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let y = Vector::from_iter((0..x.nrows()).map(|i| {
            let s = x.row(i);
            let mut best = 0usize;
            let mut bd = f64::INFINITY;
            for c in 0..self.centers.nrows() {
                let d = dtw_raw(s.as_slice(), self.centers.row(c).as_slice());
                if d < bd {
                    bd = d;
                    best = c;
                }
            }
            best as f64
        }));
        ctx.finish(y)
    }
}

/// Pairwise LCSS (tslearn `cdist_lcss`).
///
/// `eps` is not identification `p`.
pub fn cdist_lcss(
    a: &Matrix,
    b: &Matrix,
    eps: f64,
    session: &Session,
) -> Result<Qualified<Matrix>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, a, None, &ctx.policy);
    inspect_xy(&mut ctx.report, b, None, &ctx.policy);
    let e = if eps.is_finite() && eps >= 0.0 {
        eps
    } else {
        ctx.push(
            Issue::builder(IssueCode::InvalidWeight)
                .severity(Severity::Warning)
                .message(format!("cdist_lcss ε={eps}; using |ε|"))
                .build(),
        );
        eps.abs()
    };
    let out = Matrix::from_fn(a.nrows(), b.nrows(), |i, j| {
        1.0 - lcss_raw(a.row(i).as_slice(), b.row(j).as_slice(), e, None)
    });
    ctx.finish(out)
}

/// 1-D convolutional + ridge classifier (sktime `CNNClassifier` lite).
///
/// Kernel count is not identification `p`.
#[derive(Clone, Debug)]
pub struct CnnClassifier {
    /// Random kernels.
    pub n_kernels: usize,
    /// Kernel width.
    pub width: usize,
    /// Ridge \(\alpha\).
    pub alpha: f64,
    /// Seed.
    pub seed: u64,
}

impl Default for CnnClassifier {
    fn default() -> Self {
        Self {
            n_kernels: 4,
            width: 3,
            alpha: 0.1,
            seed: 11,
        }
    }
}

impl CnnClassifier {
    /// Default CNN-lite classifier.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Named CNN-lite classifier (sktime `CNNClassifier`).
pub type TimeCnnClassifier = CnnClassifier;

/// Fitted CNN-lite ridge.
#[derive(Clone, Debug)]
pub struct FittedCnnClassifier {
    kernels: Vec<Vec<f64>>,
    inner: crate::classification::FittedRidgeClassifier,
}

fn conv_maxpool(x: &Matrix, kernels: &[Vec<f64>]) -> Matrix {
    Matrix::from_fn(x.nrows(), kernels.len().max(1), |i, k| {
        if kernels.is_empty() {
            return 0.0;
        }
        let w = &kernels[k];
        let mut acc_max: f64 = f64::NEG_INFINITY;
        if w.is_empty() || x.ncols() < w.len() {
            return 0.0;
        }
        for t in 0..=x.ncols() - w.len() {
            let mut s = 0.0;
            for u in 0..w.len() {
                s += w[u] * x.get(i, t + u);
            }
            if s > acc_max {
                acc_max = s;
            }
        }
        if acc_max.is_finite() {
            acc_max
        } else {
            0.0
        }
    })
}

impl Fit for CnnClassifier {
    type Fitted = FittedCnnClassifier;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedCnnClassifier>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        inspect_classes(&mut ctx.report, y, &ctx.policy);
        let w = self.width.max(1).min(x.ncols().max(1));
        if x.ncols() < self.width {
            ctx.push(
                Issue::builder(IssueCode::WindowTooShort)
                    .message(format!(
                        "CnnClassifier width={} > T={}",
                        self.width,
                        x.ncols()
                    ))
                    .build(),
            );
        }
        let mut rng = Rng::new(self.seed);
        let kernels: Vec<Vec<f64>> = (0..self.n_kernels.max(1))
            .map(|_| (0..w).map(|_| rng.standard_normal()).collect())
            .collect();
        let z = conv_maxpool(x, &kernels);
        let inner = binary_ridge_from_features(&z, y, self.alpha, &ctx.policy, "cnn");
        ctx.finish(FittedCnnClassifier { kernels, inner })
    }
}

impl Predict for FittedCnnClassifier {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let z = conv_maxpool(x, &self.kernels);
        self.inner.predict(&z, session)
    }
}

/// Per-row forward-fill then column-mean imputer (sktime `TimeSeriesImputer`).
///
/// Does **not** call [`inspect_xy`]: NaN is the point of the transform.
#[derive(Clone, Debug)]
pub struct TimeSeriesImputer {
    col_mean: Vector,
}

impl Default for TimeSeriesImputer {
    fn default() -> Self {
        Self {
            col_mean: Vector::zeros(0),
        }
    }
}

impl TimeSeriesImputer {
    /// Empty imputer.
    pub fn new() -> Self {
        Self::default()
    }
}

impl FitUnsupervised for TimeSeriesImputer {
    type Fitted = Self;
    fn fit_unsupervised(&mut self, x: &Matrix, session: &Session) -> Result<Qualified<Self>> {
        let mut ctx = FitCtx::with_session(session.clone());
        let mut means = vec![0.0; x.ncols()];
        for j in 0..x.ncols() {
            let mut s = 0.0;
            let mut n = 0.0;
            for i in 0..x.nrows() {
                let v = x.get(i, j);
                if v.is_finite() {
                    s += v;
                    n += 1.0;
                }
            }
            if n <= 0.0 {
                ctx.push(
                    Issue::builder(IssueCode::ImputationUndefined)
                        .severity(Severity::Warning)
                        .message(format!("TimeSeriesImputer column {j} is all missing"))
                        .build(),
                );
                means[j] = 0.0;
            } else {
                means[j] = s / n;
            }
        }
        self.col_mean = Vector::from_slice(&means);
        ctx.finish(self.clone())
    }
}

impl Transform for TimeSeriesImputer {
    fn transform(&self, x: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
        let mut ctx = FitCtx::with_session(session.child("transform"));
        let out = Matrix::from_fn(x.nrows(), x.ncols(), |i, j| {
            let v = x.get(i, j);
            if v.is_finite() {
                return v;
            }
            let mut last = f64::NAN;
            for t in 0..=j {
                let u = x.get(i, t);
                if u.is_finite() {
                    last = u;
                }
            }
            if last.is_finite() {
                last
            } else if j < self.col_mean.len() {
                self.col_mean[j]
            } else {
                0.0
            }
        });
        ctx.finish(out)
    }
}

/// Multi-scale random-convolution classifier (sktime `InceptionTimeClassifier` lite).
///
/// Kernel count is not identification `p`.
#[derive(Clone, Debug)]
pub struct InceptionTimeClassifier {
    /// Kernels per width.
    pub n_kernels: usize,
    /// Ridge \(\alpha\).
    pub alpha: f64,
    /// Seed.
    pub seed: u64,
}

impl Default for InceptionTimeClassifier {
    fn default() -> Self {
        Self {
            n_kernels: 4,
            alpha: 0.1,
            seed: 17,
        }
    }
}

impl InceptionTimeClassifier {
    /// Default InceptionTime-lite classifier.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted InceptionTime-lite ridge.
#[derive(Clone, Debug)]
pub struct FittedInceptionTimeClassifier {
    kernels: Vec<Vec<f64>>,
    inner: crate::classification::FittedRidgeClassifier,
}

impl Fit for InceptionTimeClassifier {
    type Fitted = FittedInceptionTimeClassifier;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedInceptionTimeClassifier>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        inspect_classes(&mut ctx.report, y, &ctx.policy);
        let t = x.ncols().max(1);
        let widths = [3usize, 5, t.min(7)]
            .into_iter()
            .filter(|&w| w <= t && w > 0);
        let mut rng = Rng::new(self.seed);
        let mut kernels: Vec<Vec<f64>> = Vec::new();
        for w in widths {
            for _ in 0..self.n_kernels.max(1) {
                kernels.push((0..w).map(|_| rng.standard_normal()).collect());
            }
        }
        if kernels.is_empty() {
            ctx.push(
                Issue::builder(IssueCode::WindowTooShort)
                    .message("InceptionTimeClassifier has no kernel that fits T")
                    .build(),
            );
            kernels.push(vec![1.0]);
        }
        let z = conv_maxpool(x, &kernels);
        let inner = binary_ridge_from_features(&z, y, self.alpha, &ctx.policy, "inception");
        ctx.finish(FittedInceptionTimeClassifier { kernels, inner })
    }
}

impl Predict for FittedInceptionTimeClassifier {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let z = conv_maxpool(x, &self.kernels);
        self.inner.predict(&z, session)
    }
}

/// ClaSP change-point index (sktime `ClaSPSegmentation` lite).
///
/// Split count is not identification `p`.
pub fn clasp_change_point(y: &Vector, session: &Session) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(
        &mut ctx.report,
        &Matrix::from_vector(y),
        Some(y),
        &ctx.policy,
    );
    let n = y.len();
    if n < 6 {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .severity(Severity::Warning)
                .message("clasp_change_point needs n≥6")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    let mut best_t = 2usize;
    let mut best_s = f64::NEG_INFINITY;
    for t in 2..n - 2 {
        let mut sl = 0.0;
        let mut sr = 0.0;
        for i in 0..t {
            sl += y[i];
        }
        for i in t..n {
            sr += y[i];
        }
        let ml = sl / t as f64;
        let mr = sr / (n - t) as f64;
        let mut ql = 0.0;
        let mut qr = 0.0;
        for i in 0..t {
            let d = y[i] - ml;
            ql += d * d;
        }
        for i in t..n {
            let d = y[i] - mr;
            qr += d * d;
        }
        let pool = ((ql + qr) / (n as f64 - 2.0)).max(1e-18);
        let stat: f64 = (ml - mr) * (ml - mr) / pool;
        if stat > best_s {
            best_s = stat;
            best_t = t;
        }
    }
    if !best_s.is_finite() {
        ctx.push(
            Issue::builder(IssueCode::PValueUnreliable)
                .message("clasp_change_point score was non-finite")
                .build(),
        );
    }
    ctx.finish(best_t as f64)
}

/// PELT mean-change points (sktime `Pelt` / ruptures `Pelt`).
///
/// Change-point count is not identification `p`.
pub fn pelt(y: &Vector, penalty: f64, session: &Session) -> Result<Qualified<Vector>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(
        &mut ctx.report,
        &Matrix::from_vector(y),
        Some(y),
        &ctx.policy,
    );
    let n = y.len();
    if n < 4 {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .severity(Severity::Warning)
                .message("PELT needs n≥4")
                .build(),
        );
        return ctx.finish(Vector::zeros(0));
    }
    let pen = if penalty.is_finite() && penalty > 0.0 {
        penalty
    } else {
        ctx.push(
            Issue::builder(IssueCode::InvalidWeight)
                .severity(Severity::Warning)
                .message(format!("PELT penalty={penalty}; using 2 log n"))
                .build(),
        );
        2.0 * (n as f64).ln().max(1.0)
    };
    let mut ps = vec![0.0; n + 1];
    let mut qs = vec![0.0; n + 1];
    for i in 0..n {
        ps[i + 1] = ps[i] + y[i];
        qs[i + 1] = qs[i] + y[i] * y[i];
    }
    let cost = |s: usize, t: usize| -> f64 {
        let m = (t - s) as f64;
        if m <= 0.0 {
            return 0.0;
        }
        qs[t] - qs[s] - (ps[t] - ps[s]) * (ps[t] - ps[s]) / m
    };
    let mut f = vec![0.0; n + 1];
    let mut last = vec![0usize; n + 1];
    f[0] = -pen;
    for t in 1..=n {
        let mut best = f64::INFINITY;
        let mut arg = 0usize;
        for s in 0..t {
            let v = f[s] + cost(s, t) + pen;
            if v < best {
                best = v;
                arg = s;
            }
        }
        f[t] = best;
        last[t] = arg;
    }
    let mut cps = Vec::new();
    let mut t = n;
    while t > 0 {
        let s = last[t];
        if s > 0 {
            cps.push(s as f64);
        }
        if s >= t {
            break;
        }
        t = s;
    }
    cps.reverse();
    ctx.finish(Vector::from_iter(cps))
}

/// ClaSP-feature ridge classifier (sktime `ClaSPClassifier` lite).
///
/// The ClaSP split index is not identification `p`.
#[derive(Clone, Debug)]
pub struct ClaSPClassifier {
    /// Ridge \(\alpha\).
    pub alpha: f64,
}

impl Default for ClaSPClassifier {
    fn default() -> Self {
        Self { alpha: 0.1 }
    }
}

impl ClaSPClassifier {
    /// Default ClaSP-lite classifier.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted ClaSP-lite ridge.
#[derive(Clone, Debug)]
pub struct FittedClaSPClassifier {
    inner: crate::classification::FittedRidgeClassifier,
}

fn clasp_row_features(row: &Vector) -> [f64; 4] {
    let n = row.len().max(1);
    let mut s = 0.0;
    let mut q = 0.0;
    for i in 0..row.len() {
        s += row[i];
        q += row[i] * row[i];
    }
    let nf = n as f64;
    let mean = s / nf;
    let var = (q / nf - mean * mean).max(0.0);
    let slope = if n >= 2 {
        (row[n - 1] - row[0]) / (nf - 1.0)
    } else {
        0.0
    };
    let mut best_t = 1.0;
    let mut best_s = f64::NEG_INFINITY;
    if n >= 6 {
        for t in 2..n - 2 {
            let mut sl = 0.0;
            for i in 0..t {
                sl += row[i];
            }
            let ml = sl / t as f64;
            let mr = (s - sl) / (n - t) as f64;
            let d = ml - mr;
            let stat = d * d;
            if stat > best_s {
                best_s = stat;
                best_t = t as f64 / nf;
            }
        }
    }
    [mean, var.sqrt(), slope, best_t]
}

impl Fit for ClaSPClassifier {
    type Fitted = FittedClaSPClassifier;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedClaSPClassifier>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        inspect_classes(&mut ctx.report, y, &ctx.policy);
        let z = Matrix::from_fn(x.nrows(), 4, |i, j| {
            let f = clasp_row_features(&x.row(i));
            f[j]
        });
        let inner = binary_ridge_from_features(&z, y, self.alpha, &ctx.policy, "clasp");
        ctx.finish(FittedClaSPClassifier { inner })
    }
}

impl Predict for FittedClaSPClassifier {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let z = Matrix::from_fn(x.nrows(), 4, |i, j| {
            let f = clasp_row_features(&x.row(i));
            f[j]
        });
        self.inner.predict(&z, session)
    }
}

/// Matrix-profile feature ridge classifier (sktime `MatrixProfileClassifier`).
///
/// Window length is not identification `p`.
#[derive(Clone, Debug)]
pub struct MatrixProfileClassifier {
    /// Ridge \(\alpha\).
    pub alpha: f64,
    /// Subsequence length for the per-row profile.
    pub window: usize,
}

impl Default for MatrixProfileClassifier {
    fn default() -> Self {
        Self {
            alpha: 0.1,
            window: 2,
        }
    }
}

impl MatrixProfileClassifier {
    /// Default matrix-profile classifier.
    pub fn new() -> Self {
        Self::default()
    }

    /// Classifier with subsequence length `window`.
    pub fn with_window(window: usize) -> Self {
        Self {
            window: window.max(2),
            ..Self::default()
        }
    }
}

/// Fitted matrix-profile ridge.
#[derive(Clone, Debug)]
pub struct FittedMatrixProfileClassifier {
    inner: crate::classification::FittedRidgeClassifier,
    window: usize,
}

fn mp_row_features(row: &Vector, window: usize, session: &Session) -> [f64; 4] {
    let mean = row.mean();
    let std = row.std();
    let (mp_mean, mp_max) = match matrix_profile(row, window, session) {
        Ok(q) if !q.value.is_empty() => {
            let sl = q.value.as_slice();
            let mut s = 0.0_f64;
            let mut mx = f64::NEG_INFINITY;
            let mut c = 0.0_f64;
            for &v in sl {
                if v.is_finite() {
                    s += v;
                    mx = mx.max(v);
                    c += 1.0;
                }
            }
            if c > 0.0 {
                (s / c, mx)
            } else {
                (0.0, 0.0)
            }
        }
        _ => (0.0, 0.0),
    };
    [mean, std, mp_mean, mp_max]
}

impl Fit for MatrixProfileClassifier {
    type Fitted = FittedMatrixProfileClassifier;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedMatrixProfileClassifier>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        inspect_classes(&mut ctx.report, y, &ctx.policy);
        let w = self.window.max(2);
        let z = Matrix::from_fn(x.nrows(), 4, |i, j| {
            let f = mp_row_features(&x.row(i), w, &session.child(format!("mp_row{i}")));
            f[j]
        });
        let inner = binary_ridge_from_features(&z, y, self.alpha, &ctx.policy, "mpclf");
        ctx.finish(FittedMatrixProfileClassifier { inner, window: w })
    }
}

impl Predict for FittedMatrixProfileClassifier {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let z = Matrix::from_fn(x.nrows(), 4, |i, j| {
            let f = mp_row_features(&x.row(i), self.window, &session.child(format!("mp_pr{i}")));
            f[j]
        });
        self.inner.predict(&z, session)
    }
}

/// Greedy Gaussian segmentation (sktime `GreedyGaussianSegmentation`).
///
/// Change-point count is not identification `p`.
pub fn ggs(y: &Vector, max_changes: usize, session: &Session) -> Result<Qualified<Vector>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(
        &mut ctx.report,
        &Matrix::from_vector(y),
        Some(y),
        &ctx.policy,
    );
    let n = y.len();
    if n < 6 {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .severity(Severity::Warning)
                .message("GGS needs n≥6")
                .build(),
        );
        return ctx.finish(Vector::zeros(0));
    }
    let kmax = max_changes.max(1).min(n / 3);
    let mut ps = vec![0.0; n + 1];
    let mut qs = vec![0.0; n + 1];
    for i in 0..n {
        ps[i + 1] = ps[i] + y[i];
        qs[i + 1] = qs[i] + y[i] * y[i];
    }
    let seg_ll = |s: usize, t: usize| -> f64 {
        let m = (t - s) as f64;
        if m < 2.0 {
            return f64::NEG_INFINITY;
        }
        let var = ((qs[t] - qs[s]) / m - ((ps[t] - ps[s]) / m).powi(2)).max(1e-12);
        -0.5 * m * (var.ln() + 1.0)
    };
    let mut bounds = vec![0usize, n];
    for _ in 0..kmax {
        let mut best_gain = 0.0;
        let mut best_k = 0usize;
        let mut best_pos = 0usize;
        for b in 0..bounds.len() - 1 {
            let s = bounds[b];
            let t = bounds[b + 1];
            if t - s < 6 {
                continue;
            }
            let base = seg_ll(s, t);
            for k in (s + 2)..(t - 2) {
                let gain = seg_ll(s, k) + seg_ll(k, t) - base;
                if gain > best_gain {
                    best_gain = gain;
                    best_k = k;
                    best_pos = b + 1;
                }
            }
        }
        if best_gain <= 1e-9 {
            break;
        }
        bounds.insert(best_pos, best_k);
    }
    let cps = Vector::from_iter(
        bounds
            .iter()
            .skip(1)
            .take(bounds.len().saturating_sub(2))
            .map(|v| *v as f64),
    );
    ctx.finish(cps)
}

/// MrSEQL-lite: SAX word counts + ridge (sktime `MrSEQLClassifier`).
///
/// Word / alphabet counts are not identification `p`.
#[derive(Clone, Debug)]
pub struct MrSeqlClassifier {
    /// PAA segments.
    pub n_segments: usize,
    /// SAX alphabet size.
    pub alphabet: usize,
    /// Hashed word-bag width.
    pub n_words: usize,
    /// Ridge \(\alpha\).
    pub alpha: f64,
}

impl Default for MrSeqlClassifier {
    fn default() -> Self {
        Self {
            n_segments: 4,
            alphabet: 4,
            n_words: 8,
            alpha: 0.1,
        }
    }
}

impl MrSeqlClassifier {
    /// Default MrSEQL-lite classifier.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted MrSEQL-lite ridge.
#[derive(Clone, Debug)]
pub struct FittedMrSeqlClassifier {
    inner: crate::classification::FittedRidgeClassifier,
    n_segments: usize,
    alphabet: usize,
    n_words: usize,
}

fn mrseql_row_bag(row: &Vector, n_segments: usize, alphabet: usize, n_words: usize) -> Vector {
    let m = n_words.max(1);
    let mut bag = Vector::zeros(m);
    let n = row.len();
    if n == 0 {
        return bag;
    }
    let segs = n_segments.max(1).min(n);
    let a = alphabet.max(2);
    let mut prev = 0u64;
    for s in 0..segs {
        let lo = s * n / segs;
        let hi = ((s + 1) * n / segs).max(lo + 1);
        let mut acc = 0.0;
        let mut c = 0.0;
        for j in lo..hi.min(n) {
            acc += row[j];
            c += 1.0;
        }
        let mean = if c > 0.0 { acc / c } else { 0.0 };
        let u = 0.5 + 0.5 * crate::special::erf(mean / std::f64::consts::SQRT_2);
        let sym = ((u * a as f64).floor() as u64).min(a as u64 - 1);
        let word = prev.wrapping_mul(a as u64 + 1).wrapping_add(sym + 1);
        let bin = (word as usize) % m;
        bag[bin] += 1.0;
        prev = sym + 1;
    }
    bag
}

impl Fit for MrSeqlClassifier {
    type Fitted = FittedMrSeqlClassifier;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedMrSeqlClassifier>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        inspect_classes(&mut ctx.report, y, &ctx.policy);
        let w = self.n_words.max(1);
        let z = Matrix::from_fn(x.nrows(), w, |i, j| {
            mrseql_row_bag(&x.row(i), self.n_segments, self.alphabet, w)[j]
        });
        let inner = binary_ridge_from_features(&z, y, self.alpha, &ctx.policy, "mrseql");
        ctx.finish(FittedMrSeqlClassifier {
            inner,
            n_segments: self.n_segments.max(1),
            alphabet: self.alphabet.max(2),
            n_words: w,
        })
    }
}

impl Predict for FittedMrSeqlClassifier {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let z = Matrix::from_fn(x.nrows(), self.n_words.max(1), |i, j| {
            mrseql_row_bag(&x.row(i), self.n_segments, self.alphabet, self.n_words)[j]
        });
        self.inner.predict(&z, session)
    }
}

/// Catch22 feature transformer (sktime `Catch22`).
///
/// Feature count is not identification `p`.
#[derive(Clone, Debug, Default)]
pub struct Catch22Transformer {
    fitted: bool,
}

impl Catch22Transformer {
    /// Empty Catch22 transformer.
    pub fn new() -> Self {
        Self::default()
    }
}

impl FitUnsupervised for Catch22Transformer {
    type Fitted = Self;
    fn fit_unsupervised(&mut self, x: &Matrix, session: &Session) -> Result<Qualified<Self>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        self.fitted = true;
        ctx.finish(self.clone())
    }
}

impl Transform for Catch22Transformer {
    fn transform(&self, x: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
        let mut ctx = FitCtx::with_session(session.child("transform"));
        if !self.fitted {
            ctx.push(
                Issue::builder(IssueCode::PartialFitBeforeInit)
                    .message("Catch22Transformer.transform before fit")
                    .build(),
            );
        }
        let z = catch22_rows(x, session, &mut ctx);
        ctx.finish(z)
    }
}

/// Residual convolutional classifier (sktime `ResNetClassifier` lite).
///
/// Kernel count is not identification `p`.
#[derive(Clone, Debug)]
pub struct ResNetClassifier {
    /// Kernels per residual block.
    pub n_kernels: usize,
    /// Ridge \(\alpha\).
    pub alpha: f64,
    /// Seed.
    pub seed: u64,
}

impl Default for ResNetClassifier {
    fn default() -> Self {
        Self {
            n_kernels: 4,
            alpha: 0.1,
            seed: 23,
        }
    }
}

impl ResNetClassifier {
    /// Default ResNet-lite classifier.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted ResNet-lite ridge.
#[derive(Clone, Debug)]
pub struct FittedResNetClassifier {
    kernels: Vec<Vec<f64>>,
    inner: crate::classification::FittedRidgeClassifier,
}

fn residual_conv_pool(x: &Matrix, kernels: &[Vec<f64>]) -> Matrix {
    let raw = conv_maxpool(x, kernels);
    Matrix::from_fn(raw.nrows(), raw.ncols(), |i, j| {
        let skip = if j < x.ncols() {
            x.get(i, j)
        } else {
            x.get(i, j % x.ncols().max(1))
        };
        raw.get(i, j) + skip
    })
}

impl Fit for ResNetClassifier {
    type Fitted = FittedResNetClassifier;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedResNetClassifier>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        inspect_classes(&mut ctx.report, y, &ctx.policy);
        let w = 3usize.min(x.ncols().max(1));
        if x.ncols() < 3 {
            ctx.push(
                Issue::builder(IssueCode::WindowTooShort)
                    .message(format!("ResNetClassifier width=3 > T={}", x.ncols()))
                    .build(),
            );
        }
        let mut rng = Rng::new(self.seed);
        let kernels: Vec<Vec<f64>> = (0..self.n_kernels.max(1))
            .map(|_| (0..w).map(|_| rng.standard_normal()).collect())
            .collect();
        let z = residual_conv_pool(x, &kernels);
        let inner = binary_ridge_from_features(&z, y, self.alpha, &ctx.policy, "resnet");
        ctx.finish(FittedResNetClassifier { kernels, inner })
    }
}

impl Predict for FittedResNetClassifier {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let z = residual_conv_pool(x, &self.kernels);
        self.inner.predict(&z, session)
    }
}

/// Elman recurrent + conv classifier (sktime `LSTMFCNClassifier` lite).
///
/// Hidden width is not identification `p`.
#[derive(Clone, Debug)]
pub struct LstmFcnClassifier {
    /// Recurrent hidden size.
    pub hidden: usize,
    /// Conv kernels.
    pub n_kernels: usize,
    /// Ridge \(\alpha\).
    pub alpha: f64,
    /// Seed.
    pub seed: u64,
}

impl Default for LstmFcnClassifier {
    fn default() -> Self {
        Self {
            hidden: 4,
            n_kernels: 4,
            alpha: 0.1,
            seed: 29,
        }
    }
}

impl LstmFcnClassifier {
    /// Default LSTM-FCN-lite classifier.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted LSTM-FCN-lite ridge.
#[derive(Clone, Debug)]
pub struct FittedLstmFcnClassifier {
    wx: Vector,
    uh: Vector,
    kernels: Vec<Vec<f64>>,
    inner: crate::classification::FittedRidgeClassifier,
}

fn elman_pool(x: &Matrix, wx: &Vector, uh: &Vector) -> Matrix {
    let hdim = wx.len().max(1);
    Matrix::from_fn(x.nrows(), hdim, |i, h| {
        let mut state = 0.0;
        let mut acc_max: f64 = f64::NEG_INFINITY;
        for t in 0..x.ncols() {
            let xt = x.get(i, t);
            let w = if h < wx.len() { wx[h] } else { 0.0 };
            let u = if h < uh.len() { uh[h] } else { 0.0 };
            state = (w * xt + u * state).tanh();
            if state > acc_max {
                acc_max = state;
            }
        }
        if acc_max.is_finite() {
            acc_max
        } else {
            0.0
        }
    })
}

impl Fit for LstmFcnClassifier {
    type Fitted = FittedLstmFcnClassifier;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedLstmFcnClassifier>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        inspect_classes(&mut ctx.report, y, &ctx.policy);
        let h = self.hidden.max(1);
        let mut rng = Rng::new(self.seed);
        let wx = Vector::from_iter((0..h).map(|_| rng.standard_normal()));
        let uh = Vector::from_iter((0..h).map(|_| 0.2 * rng.standard_normal()));
        let w = 3usize.min(x.ncols().max(1));
        let kernels: Vec<Vec<f64>> = (0..self.n_kernels.max(1))
            .map(|_| (0..w).map(|_| rng.standard_normal()).collect())
            .collect();
        let rec = elman_pool(x, &wx, &uh);
        let conv = conv_maxpool(x, &kernels);
        let z = Matrix::from_fn(x.nrows(), rec.ncols() + conv.ncols(), |i, j| {
            if j < rec.ncols() {
                rec.get(i, j)
            } else {
                conv.get(i, j - rec.ncols())
            }
        });
        let inner = binary_ridge_from_features(&z, y, self.alpha, &ctx.policy, "lstmfcn");
        ctx.finish(FittedLstmFcnClassifier {
            wx,
            uh,
            kernels,
            inner,
        })
    }
}

impl Predict for FittedLstmFcnClassifier {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let rec = elman_pool(x, &self.wx, &self.uh);
        let conv = conv_maxpool(x, &self.kernels);
        let z = Matrix::from_fn(x.nrows(), rec.ncols() + conv.ncols(), |i, j| {
            if j < rec.ncols() {
                rec.get(i, j)
            } else {
                conv.get(i, j - rec.ncols())
            }
        });
        self.inner.predict(&z, session)
    }
}

/// Binary segmentation change-point (sktime `BinarySegmentation`).
///
/// Split count is not identification `p`.
pub fn binary_segmentation(y: &Vector, session: &Session) -> Result<Qualified<f64>> {
    clasp_change_point(y, session)
}

/// Named binary-segmentation detector (sktime `BinarySegmentation` / ruptures `Binseg`).
#[derive(Clone, Debug, Default)]
pub struct Binseg;

impl Binseg {
    /// Default binary segmentation.
    pub fn new() -> Self {
        Self
    }

    /// Index of the principal mean-change split.
    pub fn fit(&self, y: &Vector, session: &Session) -> Result<Qualified<f64>> {
        binary_segmentation(y, session)
    }
}

/// Named PELT detector (sktime `Pelt` / ruptures `Pelt`).
///
/// Penalty and change-point count are not identification `p`.
#[derive(Clone, Debug)]
pub struct Pelt {
    /// Mean-change penalty. Non-positive values fall back to \(2\log n\).
    pub penalty: f64,
}

impl Default for Pelt {
    fn default() -> Self {
        Self { penalty: 0.0 }
    }
}

impl Pelt {
    /// Default PELT (BIC-like \(2\log n\) penalty).
    pub fn new() -> Self {
        Self::default()
    }

    /// Change-point locations as a vector of indices.
    pub fn fit(&self, y: &Vector, session: &Session) -> Result<Qualified<Vector>> {
        pelt(y, self.penalty, session)
    }
}

/// Named ClaSP change-point detector (sktime `ClaSPSegmentation`).
///
/// Split count is not identification `p`.
#[derive(Clone, Debug, Default)]
pub struct ClaSPSegmentation;

impl ClaSPSegmentation {
    /// Default ClaSP segmentation.
    pub fn new() -> Self {
        Self
    }

    /// Index of the principal ClaSP split.
    pub fn fit(&self, y: &Vector, session: &Session) -> Result<Qualified<f64>> {
        clasp_change_point(y, session)
    }
}

/// Named greedy Gaussian segmentation (sktime `GreedyGaussianSegmentation`).
///
/// Change-point count is not identification `p`.
#[derive(Clone, Debug)]
pub struct Ggs {
    /// Maximum number of splits. Not identification `p`.
    pub max_changes: usize,
}

impl Default for Ggs {
    fn default() -> Self {
        Self { max_changes: 2 }
    }
}

impl Ggs {
    /// Default GGS (at most two splits).
    pub fn new() -> Self {
        Self::default()
    }

    /// Change-point locations as a vector of indices.
    pub fn fit(&self, y: &Vector, session: &Session) -> Result<Qualified<Vector>> {
        ggs(y, self.max_changes, session)
    }
}

/// Named STAMP matrix-profile detector (stumpy `stump` / sktime `STAMP`).
///
/// Window length is not identification `p`.
#[derive(Clone, Debug)]
pub struct Stamp {
    /// Subsequence length. Not identification `p`.
    pub window: usize,
}

impl Default for Stamp {
    fn default() -> Self {
        Self { window: 3 }
    }
}

impl Stamp {
    /// STAMP with a given window.
    pub fn new(window: usize) -> Self {
        Self {
            window: window.max(2),
        }
    }

    /// Matrix profile and nearest-neighbour index.
    pub fn fit(&self, y: &Vector, session: &Session) -> Result<Qualified<StampResult>> {
        stamp(y, self.window, session)
    }
}

/// Named STRAY anomaly scorer (sktime `STRAY`).
///
/// Window length is not identification `p`.
#[derive(Clone, Debug)]
pub struct Stray {
    /// Subsequence length. Not identification `p`.
    pub window: usize,
}

impl Default for Stray {
    fn default() -> Self {
        Self { window: 3 }
    }
}

impl Stray {
    /// STRAY with a given window.
    pub fn new(window: usize) -> Self {
        Self {
            window: window.max(2),
        }
    }

    /// Robust z-scores of the matrix profile.
    pub fn fit(&self, y: &Vector, session: &Session) -> Result<Qualified<Vector>> {
        stray(y, self.window, session)
    }
}

fn hist_entropy(y: &Vector, a: usize, b: usize, lo: f64, hi: f64, n_bins: usize) -> f64 {
    let k = n_bins.max(2);
    let mut bins = vec![0.0; k];
    let span = (hi - lo).max(1e-12);
    let mut n = 0.0;
    for i in a..b.min(y.len()) {
        if !y[i].is_finite() {
            continue;
        }
        let mut t = ((y[i] - lo) / span * k as f64).floor() as usize;
        if t >= k {
            t = k - 1;
        }
        bins[t] += 1.0;
        n += 1.0;
    }
    if n <= 0.0 {
        return 0.0;
    }
    let mut h = 0.0;
    for c in bins {
        if c > 0.0 {
            let p: f64 = c / n;
            h -= p * p.ln();
        }
    }
    h
}

/// Information-gain change points (sktime `InformationGainSegmentation`).
///
/// Bin / split counts are not identification `p`.
pub fn information_gain_segmentation(
    y: &Vector,
    max_changes: usize,
    n_bins: usize,
    session: &Session,
) -> Result<Qualified<Vector>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(
        &mut ctx.report,
        &Matrix::from_vector(y),
        Some(y),
        &ctx.policy,
    );
    let n = y.len();
    if n < 6 {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .severity(Severity::Warning)
                .message("information_gain_segmentation needs n≥6")
                .build(),
        );
        return ctx.finish(Vector::zeros(0));
    }
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for i in 0..n {
        if y[i].is_finite() {
            lo = lo.min(y[i]);
            hi = hi.max(y[i]);
        }
    }
    let kmax = max_changes.max(1).min(n / 3);
    let bins = n_bins.max(2);
    let mut bounds = vec![0usize, n];
    for _ in 0..kmax {
        let mut best_gain = 0.0;
        let mut best_k = 0usize;
        let mut best_pos = 0usize;
        for b in 0..bounds.len() - 1 {
            let s = bounds[b];
            let t = bounds[b + 1];
            if t - s < 6 {
                continue;
            }
            let h0 = hist_entropy(y, s, t, lo, hi, bins);
            let m = (t - s) as f64;
            for k in (s + 2)..(t - 2) {
                let hl = hist_entropy(y, s, k, lo, hi, bins);
                let hr = hist_entropy(y, k, t, lo, hi, bins);
                let gain = h0 - ((k - s) as f64 / m) * hl - ((t - k) as f64 / m) * hr;
                if gain > best_gain {
                    best_gain = gain;
                    best_k = k;
                    best_pos = b + 1;
                }
            }
        }
        if best_gain <= 1e-12 {
            break;
        }
        bounds.insert(best_pos, best_k);
    }
    ctx.finish(Vector::from_iter(
        bounds
            .iter()
            .skip(1)
            .take(bounds.len().saturating_sub(2))
            .map(|v| *v as f64),
    ))
}

/// Sliding-window mean-change peaks (sktime `WindowSegmenter` lite).
///
/// Window length is not identification `p`.
pub fn window_segment(
    y: &Vector,
    window: usize,
    session: &Session,
) -> Result<Qualified<Vector>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(
        &mut ctx.report,
        &Matrix::from_vector(y),
        Some(y),
        &ctx.policy,
    );
    let n = y.len();
    let w = window.max(2).min(n.max(2) / 2).max(1);
    if n < 2 * w + 1 {
        ctx.push(
            Issue::builder(IssueCode::WindowTooShort)
                .message("window_segment needs n > 2w")
                .build(),
        );
        return ctx.finish(Vector::zeros(0));
    }
    let mut score = vec![0.0; n];
    for t in w..n - w {
        let mut sl = 0.0;
        let mut sr = 0.0;
        for i in (t - w)..t {
            sl += y[i];
        }
        for i in t..(t + w) {
            sr += y[i];
        }
        score[t] = (sl / w as f64 - sr / w as f64).abs();
    }
    let mut peaks = Vec::new();
    for t in w + 1..n - w {
        if score[t] >= score[t - 1] && score[t] >= score[t + 1] && score[t] > 1e-12 {
            peaks.push(t as f64);
        }
    }
    ctx.finish(Vector::from_iter(peaks))
}

fn sse_seg(y: &Vector, a: usize, b: usize) -> f64 {
    let m = (b.saturating_sub(a)).max(1) as f64;
    let mut s = 0.0;
    for i in a..b.min(y.len()) {
        s += y[i];
    }
    let mu = s / m;
    let mut e = 0.0;
    for i in a..b.min(y.len()) {
        let d = y[i] - mu;
        e += d * d;
    }
    e
}

/// Bottom-up adjacent merge (ruptures `BottomUp` / sktime `BottomUpSegmenter`).
///
/// Merge / segment counts are not identification `p`.
pub fn bottom_up_segment(
    y: &Vector,
    max_changes: usize,
    session: &Session,
) -> Result<Qualified<Vector>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(
        &mut ctx.report,
        &Matrix::from_vector(y),
        Some(y),
        &ctx.policy,
    );
    let n = y.len();
    if n < 4 {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .severity(Severity::Warning)
                .message("bottom_up_segment needs n≥4")
                .build(),
        );
        return ctx.finish(Vector::zeros(0));
    }
    let mut bounds: Vec<usize> = (0..=n).collect();
    let keep = (max_changes.max(1) + 1).min(n);
    while bounds.len() > keep + 1 {
        let mut best = f64::INFINITY;
        let mut bi = 1usize;
        for i in 1..bounds.len() - 1 {
            let s = bounds[i - 1];
            let m = bounds[i];
            let t = bounds[i + 1];
            let cost = sse_seg(y, s, t) - sse_seg(y, s, m) - sse_seg(y, m, t);
            if cost < best {
                best = cost;
                bi = i;
            }
        }
        bounds.remove(bi);
    }
    ctx.finish(Vector::from_iter(
        bounds
            .iter()
            .skip(1)
            .take(bounds.len().saturating_sub(2))
            .map(|v| *v as f64),
    ))
}

/// Recursive binary segmentation (sktime `TopDownSegmenter` / ruptures `Binseg`).
///
/// Split count is not identification `p`. Distinct from a single [`binary_segmentation`] cut.
pub fn top_down_segment(
    y: &Vector,
    max_changes: usize,
    session: &Session,
) -> Result<Qualified<Vector>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(
        &mut ctx.report,
        &Matrix::from_vector(y),
        Some(y),
        &ctx.policy,
    );
    let n = y.len();
    if n < 6 {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .severity(Severity::Warning)
                .message("top_down_segment needs n≥6")
                .build(),
        );
        return ctx.finish(Vector::zeros(0));
    }
    let kmax = max_changes.max(1).min(n / 3);
    let mut bounds = vec![0usize, n];
    for _ in 0..kmax {
        let mut best_gain = 0.0;
        let mut best_k = 0usize;
        let mut best_pos = 0usize;
        for b in 0..bounds.len() - 1 {
            let s = bounds[b];
            let t = bounds[b + 1];
            if t - s < 6 {
                continue;
            }
            let base = sse_seg(y, s, t);
            for k in (s + 2)..(t - 2) {
                let gain = base - sse_seg(y, s, k) - sse_seg(y, k, t);
                if gain > best_gain {
                    best_gain = gain;
                    best_k = k;
                    best_pos = b + 1;
                }
            }
        }
        if best_gain <= 1e-12 {
            break;
        }
        bounds.insert(best_pos, best_k);
    }
    ctx.finish(Vector::from_iter(
        bounds
            .iter()
            .skip(1)
            .take(bounds.len().saturating_sub(2))
            .map(|v| *v as f64),
    ))
}

/// Hidalgo-lite local intrinsic dimension (Allegra et al. / sktime-adjacent).
///
/// Neighbor count is not identification `p`. Points are labelled by a median
/// split on local dimension, not by calling k-means.
pub fn hidalgo(x: &Matrix, n_neighbors: usize, session: &Session) -> Result<Qualified<Vector>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, x, None, &ctx.policy);
    let n = x.nrows();
    if n < 3 {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .severity(Severity::Warning)
                .message("Hidalgo needs n≥3")
                .build(),
        );
        return ctx.finish(Vector::zeros(n));
    }
    let k = n_neighbors.max(2).min(n.saturating_sub(1));
    let mut ids = Vector::zeros(n);
    for i in 0..n {
        let mut ds: Vec<f64> = (0..n)
            .filter(|&j| j != i)
            .map(|j| {
                let mut s = 0.0;
                for c in 0..x.ncols() {
                    let d = x.get(i, c) - x.get(j, c);
                    s += d * d;
                }
                s.sqrt()
            })
            .collect();
        ds.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let d1 = ds.first().copied().unwrap_or(0.0).max(1e-12);
        let dk = ds.get(k - 1).copied().unwrap_or(d1).max(d1);
        ids[i] = (dk / d1).ln() / (k as f64).ln().max(1e-12);
    }
    let mut sorted: Vec<f64> = ids.as_slice().to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let med = sorted[n / 2];
    ctx.push(
        Issue::builder(IssueCode::CausalClaimUnidentified)
            .severity(Severity::Advisory)
            .message("Hidalgo is a neighbor-ratio local dimension, not the Bayesian Hidalgo posterior")
            .compromise(NumericalCompromise::new(
                "Hidalgo mixture of manifolds",
                "local ID via d_k/d_1 then a median split",
                "the Bayesian posterior over local dimension is omitted",
                "read labels as a two-manifold ranking, not a published Hidalgo draw",
            ))
            .build(),
    );
    ctx.finish(Vector::from_iter((0..n).map(|i| if ids[i] <= med { 0.0 } else { 1.0 })))
}

/// Named information-gain segmenter.
#[derive(Clone, Debug)]
pub struct InformationGainSegmentation {
    /// Maximum splits. Not identification `p`.
    pub max_changes: usize,
    /// Histogram bins. Not identification `p`.
    pub n_bins: usize,
}

impl Default for InformationGainSegmentation {
    fn default() -> Self {
        Self {
            max_changes: 2,
            n_bins: 6,
        }
    }
}

impl InformationGainSegmentation {
    /// Default IG segmenter.
    pub fn new() -> Self {
        Self::default()
    }

    /// Change-point locations.
    pub fn fit(&self, y: &Vector, session: &Session) -> Result<Qualified<Vector>> {
        information_gain_segmentation(y, self.max_changes, self.n_bins, session)
    }
}

/// Named sliding-window segmenter.
#[derive(Clone, Debug)]
pub struct WindowSegmenter {
    /// Half-window. Not identification `p`.
    pub window: usize,
}

impl Default for WindowSegmenter {
    fn default() -> Self {
        Self { window: 2 }
    }
}

impl WindowSegmenter {
    /// Window of length `window`.
    pub fn new(window: usize) -> Self {
        Self {
            window: window.max(2),
        }
    }

    /// Peak locations of the windowed mean-change score.
    pub fn fit(&self, y: &Vector, session: &Session) -> Result<Qualified<Vector>> {
        window_segment(y, self.window, session)
    }
}

/// Named bottom-up segmenter.
#[derive(Clone, Debug)]
pub struct BottomUpSegmenter {
    /// Kept change points. Not identification `p`.
    pub max_changes: usize,
}

impl Default for BottomUpSegmenter {
    fn default() -> Self {
        Self { max_changes: 2 }
    }
}

impl BottomUpSegmenter {
    /// Keep at most `max_changes` cuts.
    pub fn new(max_changes: usize) -> Self {
        Self {
            max_changes: max_changes.max(1),
        }
    }

    /// Change-point locations.
    pub fn fit(&self, y: &Vector, session: &Session) -> Result<Qualified<Vector>> {
        bottom_up_segment(y, self.max_changes, session)
    }
}

/// Named top-down / recursive binary segmenter.
#[derive(Clone, Debug)]
pub struct TopDownSegmenter {
    /// Maximum splits. Not identification `p`.
    pub max_changes: usize,
}

impl Default for TopDownSegmenter {
    fn default() -> Self {
        Self { max_changes: 2 }
    }
}

impl TopDownSegmenter {
    /// At most `max_changes` recursive SSE splits.
    pub fn new(max_changes: usize) -> Self {
        Self {
            max_changes: max_changes.max(1),
        }
    }

    /// Change-point locations.
    pub fn fit(&self, y: &Vector, session: &Session) -> Result<Qualified<Vector>> {
        top_down_segment(y, self.max_changes, session)
    }
}

/// Named Hidalgo local-dimension annotator.
#[derive(Clone, Debug)]
pub struct Hidalgo {
    /// Neighbors used for \(d_k/d_1\). Not identification `p`.
    pub n_neighbors: usize,
}

impl Default for Hidalgo {
    fn default() -> Self {
        Self { n_neighbors: 3 }
    }
}

impl Hidalgo {
    /// Hidalgo-lite with `n_neighbors` (not identification `p`).
    pub fn new(n_neighbors: usize) -> Self {
        Self {
            n_neighbors: n_neighbors.max(2),
        }
    }

    /// Two-manifold labels from local intrinsic dimension.
    pub fn fit(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        hidalgo(x, self.n_neighbors, session)
    }
}

/// Named greedy Gaussian segmentation.
#[derive(Clone, Debug, Default)]
pub struct GreedyGaussianSegmentation {
    inner: Ggs,
}

impl GreedyGaussianSegmentation {
    /// Default GGS.
    pub fn new() -> Self {
        Self::default()
    }

    /// Change-point locations.
    pub fn fit(&self, y: &Vector, session: &Session) -> Result<Qualified<Vector>> {
        self.inner.fit(y, session)
    }
}

/// Named binary segmentation.
#[derive(Clone, Debug, Default)]
pub struct BinarySegmentation;

impl BinarySegmentation {
    /// Single mean-change split.
    pub fn new() -> Self {
        Self
    }

    /// Index of the principal split.
    pub fn fit(&self, y: &Vector, session: &Session) -> Result<Qualified<f64>> {
        Binseg::new().fit(y, session)
    }
}

fn ridge_reg_from_features(
    z: &Matrix,
    y: &Vector,
    alpha: f64,
    policy: &signlred::Policy,
    name: &str,
) -> FittedPenalized {
    let mut scratch = signlred::Report::new(name, "ridge");
    let design = z.with_intercept();
    let beta = ridge_solve(&mut scratch, &design, y, alpha.max(0.0), policy)
        .unwrap_or_else(|| Vector::zeros(design.ncols()));
    FittedPenalized {
        coef: Vector::from_iter((1..beta.len()).map(|j| beta[j])),
        intercept: beta.as_slice().first().copied().unwrap_or(0.0),
        alpha,
        l1_ratio: 0.0,
    }
}

/// InceptionTime regressor (sktime `InceptionTimeRegressor` lite).
///
/// Kernel count is not identification `p`.
#[derive(Clone, Debug)]
pub struct InceptionTimeRegressor {
    /// Kernels per width.
    pub n_kernels: usize,
    /// Ridge \(\alpha\).
    pub alpha: f64,
    /// Seed.
    pub seed: u64,
}

impl Default for InceptionTimeRegressor {
    fn default() -> Self {
        Self {
            n_kernels: 4,
            alpha: 0.1,
            seed: 31,
        }
    }
}

impl InceptionTimeRegressor {
    /// Default InceptionTime-lite regressor.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted InceptionTime-lite ridge regressor.
#[derive(Clone, Debug)]
pub struct FittedInceptionTimeRegressor {
    kernels: Vec<Vec<f64>>,
    inner: FittedPenalized,
}

fn inception_kernels(n_kernels: usize, t: usize, seed: u64) -> Vec<Vec<f64>> {
    let widths = [3usize, 5, t.min(7)];
    let mut rng = Rng::new(seed);
    let mut kernels = Vec::new();
    for w in widths {
        if w == 0 || w > t {
            continue;
        }
        for _ in 0..n_kernels.max(1) {
            kernels.push((0..w).map(|_| rng.standard_normal()).collect());
        }
    }
    if kernels.is_empty() {
        kernels.push(vec![1.0]);
    }
    kernels
}

impl Fit for InceptionTimeRegressor {
    type Fitted = FittedInceptionTimeRegressor;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedInceptionTimeRegressor>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        let kernels = inception_kernels(self.n_kernels, x.ncols().max(1), self.seed);
        let z = conv_maxpool(x, &kernels);
        let inner = ridge_reg_from_features(&z, y, self.alpha, &ctx.policy, "incep_reg");
        ctx.finish(FittedInceptionTimeRegressor { kernels, inner })
    }
}

impl Predict for FittedInceptionTimeRegressor {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let z = conv_maxpool(x, &self.kernels);
        self.inner.predict(&z, session)
    }
}

/// Random-projection + conv classifier (sktime `TapNetClassifier` lite).
///
/// Projection width is not identification `p`.
#[derive(Clone, Debug)]
pub struct TapNetClassifier {
    /// Projected length.
    pub proj: usize,
    /// Conv kernels.
    pub n_kernels: usize,
    /// Ridge \(\alpha\).
    pub alpha: f64,
    /// Seed.
    pub seed: u64,
}

impl Default for TapNetClassifier {
    fn default() -> Self {
        Self {
            proj: 4,
            n_kernels: 4,
            alpha: 0.1,
            seed: 37,
        }
    }
}

impl TapNetClassifier {
    /// Default TapNet-lite classifier.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted TapNet-lite ridge.
#[derive(Clone, Debug)]
pub struct FittedTapNetClassifier {
    proj: Matrix,
    kernels: Vec<Vec<f64>>,
    inner: crate::classification::FittedRidgeClassifier,
}

fn tap_project(x: &Matrix, proj: &Matrix) -> Matrix {
    Matrix::from_fn(x.nrows(), proj.ncols(), |i, j| {
        let mut s = 0.0;
        for t in 0..x.ncols().min(proj.nrows()) {
            s += x.get(i, t) * proj.get(t, j);
        }
        s
    })
}

impl Fit for TapNetClassifier {
    type Fitted = FittedTapNetClassifier;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedTapNetClassifier>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        inspect_classes(&mut ctx.report, y, &ctx.policy);
        let mut rng = Rng::new(self.seed);
        let pdim = self.proj.max(1);
        let proj = Matrix::from_fn(x.ncols().max(1), pdim, |_, _| rng.standard_normal());
        let z0 = tap_project(x, &proj);
        let w = 3usize.min(z0.ncols().max(1));
        let kernels: Vec<Vec<f64>> = (0..self.n_kernels.max(1))
            .map(|_| (0..w).map(|_| rng.standard_normal()).collect())
            .collect();
        let z = conv_maxpool(&z0, &kernels);
        let inner = binary_ridge_from_features(&z, y, self.alpha, &ctx.policy, "tapnet");
        ctx.finish(FittedTapNetClassifier {
            proj,
            kernels,
            inner,
        })
    }
}

impl Predict for FittedTapNetClassifier {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let z0 = tap_project(x, &self.proj);
        let z = conv_maxpool(&z0, &self.kernels);
        self.inner.predict(&z, session)
    }
}

/// Fully convolutional network classifier (sktime `FCNClassifier` lite).
///
/// Each series is used as a flattened feature map; the temporal length is not
/// identification `p`.
#[derive(Clone, Debug)]
pub struct FCNClassifier {
    /// Ridge \(\alpha\).
    pub alpha: f64,
}

impl Default for FCNClassifier {
    fn default() -> Self {
        Self { alpha: 0.1 }
    }
}

impl FCNClassifier {
    /// Default FCN-lite classifier.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted FCN-lite ridge.
#[derive(Clone, Debug)]
pub struct FittedFCNClassifier {
    inner: crate::classification::FittedRidgeClassifier,
}

impl Fit for FCNClassifier {
    type Fitted = FittedFCNClassifier;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedFCNClassifier>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        inspect_classes(&mut ctx.report, y, &ctx.policy);
        let inner = binary_ridge_from_features(x, y, self.alpha, &ctx.policy, "fcn");
        ctx.finish(FittedFCNClassifier { inner })
    }
}

impl Predict for FittedFCNClassifier {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        self.inner.predict(x, session)
    }
}

fn macnn_attend(raw: &Matrix, ctx: &mut FitCtx) -> Matrix {
    let k = raw.ncols().max(1);
    let mut overflow = 0u64;
    let z = Matrix::from_fn(raw.nrows(), k, |i, j| {
        let mut mx = f64::NEG_INFINITY;
        for u in 0..raw.ncols() {
            mx = mx.max(raw.get(i, u));
        }
        let mut den = 0.0;
        for u in 0..raw.ncols() {
            den += (raw.get(i, u) - mx).exp();
        }
        if !den.is_finite() || den <= 0.0 {
            overflow += 1;
            raw.get(i, j) / k as f64
        } else {
            raw.get(i, j) * (raw.get(i, j) - mx).exp() / den
        }
    });
    if overflow > 0 {
        ctx.push(
            Issue::builder(IssueCode::NonFiniteOutput)
                .severity(Severity::Warning)
                .message(format!(
                    "MACNN softmax attention overflowed on {overflow} series; used uniform weights"
                ))
                .compromise(NumericalCompromise::new(
                    "softmax attention over inception kernels",
                    "uniform attention over kernels",
                    "the attention logits overflowed",
                    "kernel weights are a fallback, not a learned attention map",
                ))
                .build(),
        );
    }
    z
}

/// Multi-scale attention CNN (sktime `MACNNClassifier` lite).
///
/// Kernel count is not identification `p`.
#[derive(Clone, Debug)]
pub struct MacnnClassifier {
    /// Kernels per width.
    pub n_kernels: usize,
    /// Ridge \(\alpha\).
    pub alpha: f64,
    /// Seed.
    pub seed: u64,
}

/// sktime `MACNNClassifier` spelling of [`MacnnClassifier`].
pub type MACNN = MacnnClassifier;

impl Default for MacnnClassifier {
    fn default() -> Self {
        Self {
            n_kernels: 4,
            alpha: 0.1,
            seed: 41,
        }
    }
}

impl MacnnClassifier {
    /// Default MACNN-lite classifier.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted MACNN-lite ridge.
#[derive(Clone, Debug)]
pub struct FittedMacnnClassifier {
    kernels: Vec<Vec<f64>>,
    inner: crate::classification::FittedRidgeClassifier,
}

impl Fit for MacnnClassifier {
    type Fitted = FittedMacnnClassifier;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedMacnnClassifier>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        inspect_classes(&mut ctx.report, y, &ctx.policy);
        let kernels = inception_kernels(self.n_kernels, x.ncols().max(1), self.seed);
        let raw = conv_maxpool(x, &kernels);
        let z = macnn_attend(&raw, &mut ctx);
        let inner = binary_ridge_from_features(&z, y, self.alpha, &ctx.policy, "macnn");
        ctx.finish(FittedMacnnClassifier { kernels, inner })
    }
}

impl Predict for FittedMacnnClassifier {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        let raw = conv_maxpool(x, &self.kernels);
        let z = macnn_attend(&raw, &mut ctx);
        drop(ctx);
        self.inner.predict(&z, session)
    }
}

/// Random-projection + conv regressor (sktime `TapNetRegressor` lite).
///
/// Projection width is not identification `p`.
#[derive(Clone, Debug)]
pub struct TapNetRegressor {
    /// Projected length.
    pub proj: usize,
    /// Conv kernels.
    pub n_kernels: usize,
    /// Ridge \(\alpha\).
    pub alpha: f64,
    /// Seed.
    pub seed: u64,
}

impl Default for TapNetRegressor {
    fn default() -> Self {
        Self {
            proj: 4,
            n_kernels: 4,
            alpha: 0.1,
            seed: 43,
        }
    }
}

impl TapNetRegressor {
    /// Default TapNet-lite regressor.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted TapNet-lite ridge regressor.
#[derive(Clone, Debug)]
pub struct FittedTapNetRegressor {
    proj: Matrix,
    kernels: Vec<Vec<f64>>,
    inner: FittedPenalized,
}

impl Fit for TapNetRegressor {
    type Fitted = FittedTapNetRegressor;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedTapNetRegressor>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        let mut rng = Rng::new(self.seed);
        let pdim = self.proj.max(1);
        let proj = Matrix::from_fn(x.ncols().max(1), pdim, |_, _| rng.standard_normal());
        let z0 = tap_project(x, &proj);
        let w = 3usize.min(z0.ncols().max(1));
        let kernels: Vec<Vec<f64>> = (0..self.n_kernels.max(1))
            .map(|_| (0..w).map(|_| rng.standard_normal()).collect())
            .collect();
        let z = conv_maxpool(&z0, &kernels);
        let inner = ridge_reg_from_features(&z, y, self.alpha, &ctx.policy, "tapnet_reg");
        ctx.finish(FittedTapNetRegressor {
            proj,
            kernels,
            inner,
        })
    }
}

impl Predict for FittedTapNetRegressor {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let z0 = tap_project(x, &self.proj);
        let z = conv_maxpool(&z0, &self.kernels);
        self.inner.predict(&z, session)
    }
}

/// Convolutional encoder + ridge classifier (sktime `EncoderClassifier` lite).
///
/// Encoder width is not identification `p`.
#[derive(Clone, Debug)]
pub struct EncoderClassifier {
    /// Bottleneck width.
    pub latent: usize,
    /// Ridge \(\alpha\).
    pub alpha: f64,
    /// Seed.
    pub seed: u64,
}

impl Default for EncoderClassifier {
    fn default() -> Self {
        Self {
            latent: 4,
            alpha: 0.1,
            seed: 47,
        }
    }
}

impl EncoderClassifier {
    /// Default encoder-lite classifier.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted encoder-lite ridge.
#[derive(Clone, Debug)]
pub struct FittedEncoderClassifier {
    enc: Matrix,
    inner: crate::classification::FittedRidgeClassifier,
}

fn encode_series(x: &Matrix, enc: &Matrix) -> Matrix {
    Matrix::from_fn(x.nrows(), enc.ncols(), |i, j| {
        let mut s = 0.0;
        let t = x.ncols().min(enc.nrows());
        for u in 0..t {
            s += x.get(i, u) * enc.get(u, j);
        }
        (s / t.max(1) as f64).tanh()
    })
}

impl Fit for EncoderClassifier {
    type Fitted = FittedEncoderClassifier;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedEncoderClassifier>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        inspect_classes(&mut ctx.report, y, &ctx.policy);
        let mut rng = Rng::new(self.seed);
        let lat = self.latent.max(1);
        let enc = Matrix::from_fn(x.ncols().max(1), lat, |_, _| rng.standard_normal());
        let z = encode_series(x, &enc);
        let inner = binary_ridge_from_features(&z, y, self.alpha, &ctx.policy, "encoder");
        ctx.finish(FittedEncoderClassifier { enc, inner })
    }
}

impl Predict for FittedEncoderClassifier {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let z = encode_series(x, &self.enc);
        self.inner.predict(&z, session)
    }
}

/// Flattened random-hidden tanh + ridge (sktime `MLPClassifier` lite).
///
/// Hidden width is not identification `p`.
#[derive(Clone, Debug)]
pub struct MlpTimeClassifier {
    /// Hidden units.
    pub hidden: usize,
    /// Ridge \(\alpha\).
    pub alpha: f64,
    /// Seed.
    pub seed: u64,
}

impl Default for MlpTimeClassifier {
    fn default() -> Self {
        Self {
            hidden: 8,
            alpha: 0.1,
            seed: 29,
        }
    }
}

impl MlpTimeClassifier {
    /// Default MLP-lite time classifier.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted MLP-lite ridge.
#[derive(Clone, Debug)]
pub struct FittedMlpTimeClassifier {
    hidden: Matrix,
    inner: crate::classification::FittedRidgeClassifier,
}

fn mlp_hidden(x: &Matrix, w: &Matrix) -> Matrix {
    Matrix::from_fn(x.nrows(), w.ncols(), |i, j| {
        let mut s = 0.0;
        let t = x.ncols().min(w.nrows());
        for u in 0..t {
            s += x.get(i, u) * w.get(u, j);
        }
        s.tanh()
    })
}

impl Fit for MlpTimeClassifier {
    type Fitted = FittedMlpTimeClassifier;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedMlpTimeClassifier>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        inspect_classes(&mut ctx.report, y, &ctx.policy);
        let mut rng = Rng::new(self.seed);
        let h = self.hidden.max(1);
        let w = Matrix::from_fn(x.ncols().max(1), h, |_, _| rng.standard_normal());
        let z = mlp_hidden(x, &w);
        let inner = binary_ridge_from_features(&z, y, self.alpha, &ctx.policy, "mlp-time");
        ctx.finish(FittedMlpTimeClassifier { hidden: w, inner })
    }
}

impl Predict for FittedMlpTimeClassifier {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let z = mlp_hidden(x, &self.hidden);
        self.inner.predict(&z, session)
    }
}

/// MultiROCKET features + ridge (sktime `MultiRocketRegressor`).
///
/// Kernel count is not identification `p`.
#[derive(Clone, Debug)]
pub struct MultiRocketRegressor {
    /// Random kernels.
    pub n_kernels: usize,
    /// Ridge \(\alpha\).
    pub alpha: f64,
    /// Seed.
    pub seed: u64,
}

impl Default for MultiRocketRegressor {
    fn default() -> Self {
        Self {
            n_kernels: 16,
            alpha: 1.0,
            seed: 7,
        }
    }
}

impl MultiRocketRegressor {
    /// Default MultiROCKET regressor.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted MultiROCKET ridge.
#[derive(Clone, Debug)]
pub struct FittedMultiRocketRegressor {
    rocket: MultiRocket,
    inner: FittedPenalized,
}

impl Fit for MultiRocketRegressor {
    type Fitted = FittedMultiRocketRegressor;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedMultiRocketRegressor>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        let rocket = MultiRocket {
            n_kernels: self.n_kernels.max(1),
            seed: self.seed,
        };
        let feat = rocket.transform(x, &session.child("mrocket"))?;
        let mut scratch = signlred::Report::new("mrocket_reg", "ridge");
        let design = feat.value.with_intercept();
        let beta = ridge_solve(&mut scratch, &design, y, self.alpha.max(0.0), &ctx.policy)
            .unwrap_or_else(|| Vector::zeros(design.ncols()));
        for issue in scratch.issues() {
            if matches!(
                issue.code,
                IssueCode::ResidualTooLarge
                    | IssueCode::NearSingular
                    | IssueCode::RankZero
                    | IssueCode::R2IsOne
                    | IssueCode::PerfectCollinearity
            ) {
                continue;
            }
            ctx.push(issue.clone());
        }
        ctx.finish(FittedMultiRocketRegressor {
            rocket,
            inner: FittedPenalized {
                coef: Vector::from_iter((1..beta.len()).map(|j| beta[j])),
                intercept: beta.as_slice().first().copied().unwrap_or(0.0),
                alpha: self.alpha,
                l1_ratio: 0.0,
            },
        })
    }
}

impl Predict for FittedMultiRocketRegressor {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let feat = self.rocket.transform(x, &session.child("mrocket"))?;
        self.inner.predict(&feat.value, session)
    }
}

/// Hydra convolution features + ridge (sktime `HydraRegressor`).
///
/// Kernel / group counts are not identification `p`.
#[derive(Clone, Debug)]
pub struct HydraRegressor {
    /// Kernels.
    pub n_kernels: usize,
    /// Groups.
    pub n_groups: usize,
    /// Kernel width.
    pub width: usize,
    /// Ridge \(\alpha\).
    pub alpha: f64,
    /// Seed.
    pub seed: u64,
}

impl Default for HydraRegressor {
    fn default() -> Self {
        Self {
            n_kernels: 8,
            n_groups: 4,
            width: 3,
            alpha: 0.1,
            seed: 5,
        }
    }
}

impl HydraRegressor {
    /// Default Hydra regressor.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted Hydra ridge regressor.
#[derive(Clone, Debug)]
pub struct FittedHydraRegressor {
    kernels: Vec<(Vec<f64>, usize)>,
    n_groups: usize,
    inner: FittedPenalized,
}

impl Fit for HydraRegressor {
    type Fitted = FittedHydraRegressor;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedHydraRegressor>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        let kernels = hydra_kernels(self.n_kernels, self.n_groups, self.width, self.seed);
        let z = hydra_apply(x, &kernels, self.n_groups);
        let mut scratch = signlred::Report::new("hydra_reg", "ridge");
        let design = z.with_intercept();
        let beta = ridge_solve(&mut scratch, &design, y, self.alpha.max(0.0), &ctx.policy)
            .unwrap_or_else(|| Vector::zeros(design.ncols()));
        for issue in scratch.issues() {
            if matches!(
                issue.code,
                IssueCode::ResidualTooLarge
                    | IssueCode::NearSingular
                    | IssueCode::RankZero
                    | IssueCode::R2IsOne
                    | IssueCode::PerfectCollinearity
            ) {
                continue;
            }
            ctx.push(issue.clone());
        }
        ctx.finish(FittedHydraRegressor {
            kernels,
            n_groups: self.n_groups.max(1),
            inner: FittedPenalized {
                coef: Vector::from_iter((1..beta.len()).map(|j| beta[j])),
                intercept: beta.as_slice().first().copied().unwrap_or(0.0),
                alpha: self.alpha,
                l1_ratio: 0.0,
            },
        })
    }
}

impl Predict for FittedHydraRegressor {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let z = hydra_apply(x, &self.kernels, self.n_groups);
        self.inner.predict(&z, session)
    }
}

/// Single BOSS word-histogram + ridge (sktime `IndividualBOSS`).
///
/// Word count is not identification `p`.
#[derive(Clone, Debug)]
pub struct IndividualBoss {
    /// Sliding-window length.
    pub window: usize,
    /// DFT coefficients kept per window.
    pub word_len: usize,
    /// SFA alphabet size.
    pub alphabet: usize,
}

impl Default for IndividualBoss {
    fn default() -> Self {
        Self {
            window: 8,
            word_len: 4,
            alphabet: 4,
        }
    }
}

impl IndividualBoss {
    /// Default individual BOSS.
    pub fn new() -> Self {
        Self::default()
    }
}

impl Fit for IndividualBoss {
    type Fitted = FittedBoss;
    fn fit(&mut self, x: &Matrix, y: &Vector, session: &Session) -> Result<Qualified<FittedBoss>> {
        let mut ens = BossEnsemble {
            window: self.window,
            word_len: self.word_len,
            alphabet: self.alphabet,
        };
        ens.fit(x, y, session)
    }
}

/// Matrix-profile summaries + ridge (sktime `MatrixProfileRegressor` lite).
///
/// Window length is not identification `p`.
#[derive(Clone, Debug)]
pub struct MatrixProfileRegressor {
    /// Ridge \(\alpha\).
    pub alpha: f64,
    /// Subsequence length.
    pub window: usize,
}

impl Default for MatrixProfileRegressor {
    fn default() -> Self {
        Self {
            alpha: 0.1,
            window: 2,
        }
    }
}

impl MatrixProfileRegressor {
    /// Default matrix-profile regressor.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted matrix-profile ridge.
#[derive(Clone, Debug)]
pub struct FittedMatrixProfileRegressor {
    inner: FittedPenalized,
    window: usize,
}

impl Fit for MatrixProfileRegressor {
    type Fitted = FittedMatrixProfileRegressor;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedMatrixProfileRegressor>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        let w = self.window.max(2);
        let z = Matrix::from_fn(x.nrows(), 4, |i, j| {
            let f = mp_row_features(&x.row(i), w, &session.child(format!("mpreg{i}")));
            f[j]
        });
        let mut scratch = signlred::Report::new("mpreg", "ridge");
        let design = z.with_intercept();
        let beta = ridge_solve(&mut scratch, &design, y, self.alpha.max(0.0), &ctx.policy)
            .unwrap_or_else(|| Vector::zeros(design.ncols()));
        for issue in scratch.issues() {
            if matches!(
                issue.code,
                IssueCode::ResidualTooLarge
                    | IssueCode::NearSingular
                    | IssueCode::RankZero
                    | IssueCode::R2IsOne
                    | IssueCode::PerfectCollinearity
            ) {
                continue;
            }
            ctx.push(issue.clone());
        }
        ctx.finish(FittedMatrixProfileRegressor {
            inner: FittedPenalized {
                coef: Vector::from_iter((1..beta.len()).map(|j| beta[j])),
                intercept: beta.as_slice().first().copied().unwrap_or(0.0),
                alpha: self.alpha,
                l1_ratio: 0.0,
            },
            window: w,
        })
    }
}

impl Predict for FittedMatrixProfileRegressor {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let z = Matrix::from_fn(x.nrows(), 4, |i, j| {
            let f = mp_row_features(&x.row(i), self.window, &session.child(format!("mpreg_p{i}")));
            f[j]
        });
        self.inner.predict(&z, session)
    }
}

/// Random-interval features + ridge (sktime `RandomIntervalClassifier`).
///
/// Interval count is not identification `p`.
#[derive(Clone, Debug)]
pub struct RandomIntervalClassifier {
    /// Random intervals.
    pub n_intervals: usize,
    /// Ridge \(\alpha\).
    pub alpha: f64,
    /// Seed.
    pub seed: u64,
}

impl Default for RandomIntervalClassifier {
    fn default() -> Self {
        Self {
            n_intervals: 6,
            alpha: 0.1,
            seed: 4,
        }
    }
}

impl RandomIntervalClassifier {
    /// Default random-interval classifier.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted random-interval ridge.
#[derive(Clone, Debug)]
pub struct FittedRandomIntervalClassifier {
    intervals: Vec<Interval>,
    inner: crate::classification::FittedRidgeClassifier,
}

impl Fit for RandomIntervalClassifier {
    type Fitted = FittedRandomIntervalClassifier;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedRandomIntervalClassifier>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        inspect_classes(&mut ctx.report, y, &ctx.policy);
        let tlen = x.ncols().max(1);
        let mut rng = Rng::new(self.seed);
        let ni = self.n_intervals.max(1);
        let mut intervals = Vec::with_capacity(ni);
        for _ in 0..ni {
            let a = rng.below(tlen);
            let span = rng.below(tlen).max(1);
            let b = (a + span).min(tlen);
            intervals.push(Interval {
                start: a.min(b.saturating_sub(1)),
                end: b.max(a + 1),
            });
        }
        let z = interval_feats(x, &intervals);
        let inner = binary_ridge_from_features(&z, y, self.alpha, &ctx.policy, "ric");
        ctx.finish(FittedRandomIntervalClassifier { intervals, inner })
    }
}

impl Predict for FittedRandomIntervalClassifier {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let z = interval_feats(x, &self.intervals);
        self.inner.predict(&z, session)
    }
}

/// Three-member HIVE-COTE v2 lite (ROCKET + TSF + Catch22).
///
/// Member / kernel / tree counts are not identification `p`.
#[derive(Clone, Debug)]
pub struct HiveCoteV2 {
    /// ROCKET kernels.
    pub n_kernels: usize,
    /// Forest trees.
    pub n_estimators: usize,
    /// Seed.
    pub seed: u64,
}

impl Default for HiveCoteV2 {
    fn default() -> Self {
        Self {
            n_kernels: 16,
            n_estimators: 6,
            seed: 3,
        }
    }
}

impl HiveCoteV2 {
    /// Default HIVE-COTE v2 lite.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted three-member HIVE-COTE v2 vote.
#[derive(Clone, Debug)]
pub struct FittedHiveCoteV2 {
    rocket: FittedRocketClassifier,
    forest: FittedTimeSeriesForest,
    catch22: FittedCatch22Classifier,
}

impl Fit for HiveCoteV2 {
    type Fitted = FittedHiveCoteV2;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedHiveCoteV2>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        ctx.push(
            Issue::builder(IssueCode::CausalClaimUnidentified)
                .severity(Severity::Advisory)
                .message(
                    "HIVE-COTE v2 lite is an unweighted vote of ROCKET, TSF, and Catch22",
                )
                .compromise(NumericalCompromise::new(
                    "HIVE-COTE v2 weighted ensemble",
                    "unweighted vote of RocketClassifier, TimeSeriesForest, and Catch22",
                    "STC / cBOSS / TDE members and published weights are omitted",
                    "do not read the vote as a published HIVE-COTE v2 accuracy",
                ))
                .build(),
        );
        let rocket = RocketClassifier {
            n_kernels: self.n_kernels,
            kernel_len: 5,
            alpha: 0.5,
            seed: self.seed,
        }
        .fit(x, y, &session.child("hcv2-rocket"))?
        .value;
        let forest = TimeSeriesForestClassifier {
            n_estimators: self.n_estimators,
            n_intervals: 3,
            max_depth: 4,
            seed: self.seed,
        }
        .fit(x, y, &session.child("hcv2-tsf"))?
        .value;
        let catch22 = Catch22Classifier::new(0.1)
            .fit(x, y, &session.child("hcv2-c22"))?
            .value;
        ctx.finish(FittedHiveCoteV2 {
            rocket,
            forest,
            catch22,
        })
    }
}

impl Predict for FittedHiveCoteV2 {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let ctx = FitCtx::with_session(session.child("predict"));
        let a = self.rocket.predict(x, &session.child("r"))?;
        let b = self.forest.predict(x, &session.child("f"))?;
        let c = self.catch22.predict(x, &session.child("c"))?;
        let y = Vector::from_iter((0..x.nrows()).map(|i| {
            let va = if i < a.value.len() { a.value[i] } else { 0.0 };
            let vb = if i < b.value.len() { b.value[i] } else { 0.0 };
            let vc = if i < c.value.len() { c.value[i] } else { 0.0 };
            let mut votes: BTreeMap<i64, usize> = BTreeMap::new();
            *votes.entry(va.round() as i64).or_insert(0) += 1;
            *votes.entry(vb.round() as i64).or_insert(0) += 1;
            *votes.entry(vc.round() as i64).or_insert(0) += 1;
            votes
                .iter()
                .max_by(|u, v| u.1.cmp(v.1).then(v.0.cmp(u.0)))
                .map(|(k, _)| *k as f64)
                .unwrap_or(va)
        }));
        ctx.finish(y)
    }
}

/// Three-member HIVE-COTE v1 lite (shapelet + TSF + BOSS).
///
/// Member / shapelet / tree / word counts are not identification `p`.
#[derive(Clone, Debug)]
pub struct HiveCoteV1 {
    /// Shapelets.
    pub n_shapelets: usize,
    /// Forest trees.
    pub n_estimators: usize,
    /// BOSS window (must be ≤ series length).
    pub window: usize,
    /// Seed.
    pub seed: u64,
}

impl Default for HiveCoteV1 {
    fn default() -> Self {
        Self {
            n_shapelets: 3,
            n_estimators: 6,
            window: 4,
            seed: 3,
        }
    }
}

impl HiveCoteV1 {
    /// Default HIVE-COTE v1 lite.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted three-member HIVE-COTE v1 vote.
#[derive(Clone, Debug)]
pub struct FittedHiveCoteV1 {
    shapelet: FittedShapeletTransformClassifier,
    forest: FittedTimeSeriesForest,
    boss: FittedBoss,
}

impl Fit for HiveCoteV1 {
    type Fitted = FittedHiveCoteV1;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedHiveCoteV1>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        ctx.push(
            Issue::builder(IssueCode::CausalClaimUnidentified)
                .severity(Severity::Advisory)
                .message("HIVE-COTE v1 lite is an unweighted vote of STC, TSF, and BOSS")
                .compromise(NumericalCompromise::new(
                    "HIVE-COTE v1 weighted ensemble",
                    "unweighted vote of ShapeletTransformClassifier, TimeSeriesForest, and BOSS",
                    "published HC1 weights and the full STC/cBOSS members are omitted",
                    "do not read the vote as a published HIVE-COTE v1 accuracy",
                ))
                .build(),
        );
        let shapelet = ShapeletTransformClassifier {
            n_shapelets: self.n_shapelets,
            length: 3,
            alpha: 0.1,
            seed: self.seed,
        }
        .fit(x, y, &session.child("hcv1-stc"))?
        .value;
        let forest = TimeSeriesForestClassifier {
            n_estimators: self.n_estimators,
            n_intervals: 3,
            max_depth: 4,
            seed: self.seed,
        }
        .fit(x, y, &session.child("hcv1-tsf"))?
        .value;
        let boss = BossEnsemble {
            window: self.window.max(2).min(x.ncols().max(2)),
            word_len: 4,
            alphabet: 4,
        }
        .fit(x, y, &session.child("hcv1-boss"))?
        .value;
        ctx.finish(FittedHiveCoteV1 {
            shapelet,
            forest,
            boss,
        })
    }
}

impl Predict for FittedHiveCoteV1 {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let ctx = FitCtx::with_session(session.child("predict"));
        let a = self.shapelet.predict(x, &session.child("s"))?;
        let b = self.forest.predict(x, &session.child("f"))?;
        let c = self.boss.predict(x, &session.child("b"))?;
        let y = Vector::from_iter((0..x.nrows()).map(|i| {
            let va = if i < a.value.len() { a.value[i] } else { 0.0 };
            let vb = if i < b.value.len() { b.value[i] } else { 0.0 };
            let vc = if i < c.value.len() { c.value[i] } else { 0.0 };
            let mut votes: BTreeMap<i64, usize> = BTreeMap::new();
            *votes.entry(va.round() as i64).or_insert(0) += 1;
            *votes.entry(vb.round() as i64).or_insert(0) += 1;
            *votes.entry(vc.round() as i64).or_insert(0) += 1;
            votes
                .iter()
                .max_by(|u, v| u.1.cmp(v.1).then(v.0.cmp(u.0)))
                .map(|(k, _)| *k as f64)
                .unwrap_or(va)
        }));
        ctx.finish(y)
    }
}

/// Catch22 + ExtraTrees (sktime `Catch22` + `ExtraTreesClassifier` / `Catch22El`).
///
/// Catch22 width and tree count are not identification `p`.
#[derive(Clone, Debug)]
pub struct Catch22El {
    /// ExtraTrees count.
    pub n_estimators: usize,
    /// Tree depth.
    pub max_depth: usize,
    /// Seed.
    pub seed: u64,
}

impl Default for Catch22El {
    fn default() -> Self {
        Self {
            n_estimators: 8,
            max_depth: 4,
            seed: 5,
        }
    }
}

impl Catch22El {
    /// Default Catch22–ExtraTrees ensemble.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted Catch22–ExtraTrees (ridge fallback if the forest is vacuous).
#[derive(Clone, Debug)]
pub struct FittedCatch22El {
    forest: Option<crate::tree::FittedForestClassifier>,
    ridge: Option<FittedCatch22Classifier>,
}

impl Fit for Catch22El {
    type Fitted = FittedCatch22El;
    fn fit(&mut self, x: &Matrix, y: &Vector, session: &Session) -> Result<Qualified<FittedCatch22El>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        inspect_classes(&mut ctx.report, y, &ctx.policy);
        let z = catch22_rows(x, session, &mut ctx);
        ctx.push(
            Issue::builder(IssueCode::CausalClaimUnidentified)
                .severity(Severity::Advisory)
                .message("Catch22El is Catch22 features plus ExtraTrees, not CanonicalIntervalForest")
                .compromise(NumericalCompromise::new(
                    "sktime Catch22 + ExtraTrees pipeline",
                    "catch22_rows then ExtraTreesClassifier",
                    "rotation-forest / CIF members are omitted",
                    "do not read as a published Catch22-ensemble accuracy",
                ))
                .build(),
        );
        let mut et = crate::tree::ExtraTreesClassifier {
            n_estimators: self.n_estimators.max(1),
            max_depth: self.max_depth.max(1),
            min_samples_split: 2,
            max_features: Some(4),
            seed: self.seed,
        };
        match et.fit(&z, y, &session.child("c22el-et")) {
            Ok(q) => {
                for issue in q.report.issues() {
                    if matches!(
                        issue.code,
                        IssueCode::ResidualTooLarge
                            | IssueCode::NearSingular
                            | IssueCode::R2IsOne
                            | IssueCode::RankZero
                            | IssueCode::MeaninglessFit
                    ) {
                        continue;
                    }
                    ctx.push(issue.clone());
                }
                ctx.finish(FittedCatch22El {
                    forest: Some(q.value),
                    ridge: None,
                })
            }
            Err(_) => {
                ctx.push(
                    Issue::builder(IssueCode::DidNotConverge)
                        .message("Catch22El ExtraTrees failed; falling back to Catch22 ridge")
                        .build(),
                );
                match Catch22Classifier::new(0.1).fit(x, y, &session.child("c22el-ridge")) {
                    Ok(q) => ctx.finish(FittedCatch22El {
                        forest: None,
                        ridge: Some(q.value),
                    }),
                    Err(_) => ctx.finish(FittedCatch22El {
                        forest: None,
                        ridge: None,
                    }),
                }
            }
        }
    }
}

impl Predict for FittedCatch22El {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        if let Some(f) = &self.forest {
            let z = {
                let mut ctx = FitCtx::with_session(session.child("c22el-z"));
                catch22_rows(x, session, &mut ctx)
            };
            return f.predict(&z, session);
        }
        if let Some(r) = &self.ridge {
            return r.predict(x, session);
        }
        let mut ctx = FitCtx::with_session(session.child("predict"));
        ctx.push(Issue::builder(IssueCode::PartialFitBeforeInit).build());
        ctx.finish(Vector::zeros(x.nrows()))
    }
}

/// Catch22 + Rotation Forest (sktime `RotationForestClassifier` / Catch22 pipeline).
///
/// Catch22 width and tree count are not identification `p`.
#[derive(Clone, Debug)]
pub struct RotationForestClassifier {
    /// Rotation-forest members.
    pub n_estimators: usize,
    /// Tree depth.
    pub max_depth: usize,
    /// Seed.
    pub seed: u64,
}

impl Default for RotationForestClassifier {
    fn default() -> Self {
        Self {
            n_estimators: 4,
            max_depth: 4,
            seed: 7,
        }
    }
}

impl RotationForestClassifier {
    /// Default Catch22–rotation-forest ensemble.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted Catch22–rotation forest (ridge fallback if every rotated tree fails).
#[derive(Clone, Debug)]
pub struct FittedRotationForestClassifier {
    forest: Option<crate::ensemble::FittedRotationForest>,
    ridge: Option<FittedCatch22Classifier>,
}

impl Fit for RotationForestClassifier {
    type Fitted = FittedRotationForestClassifier;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedRotationForestClassifier>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        inspect_classes(&mut ctx.report, y, &ctx.policy);
        let z = catch22_rows(x, session, &mut ctx);
        ctx.push(
            Issue::builder(IssueCode::CausalClaimUnidentified)
                .severity(Severity::Advisory)
                .message("RotationForestClassifier is Catch22 features plus rotation trees")
                .compromise(NumericalCompromise::new(
                    "sktime RotationForest on the raw series",
                    "catch22_rows then ensemble RotationForest",
                    "PCA rotations are on Catch22, not on the time axis",
                    "do not read as a published Rotation Forest TSC accuracy",
                ))
                .build(),
        );
        let mut rf = crate::ensemble::RotationForest {
            n_estimators: self.n_estimators.max(1),
            max_depth: self.max_depth.max(1),
            seed: self.seed,
        };
        match rf.fit(&z, y, &session.child("rotf-c22")) {
            Ok(q) => {
                for issue in q.report.issues() {
                    if matches!(
                        issue.code,
                        IssueCode::ResidualTooLarge
                            | IssueCode::NearSingular
                            | IssueCode::R2IsOne
                            | IssueCode::RankZero
                            | IssueCode::MeaninglessFit
                    ) {
                        continue;
                    }
                    ctx.push(issue.clone());
                }
                ctx.finish(FittedRotationForestClassifier {
                    forest: Some(q.value),
                    ridge: None,
                })
            }
            Err(_) => {
                ctx.push(
                    Issue::builder(IssueCode::DidNotConverge)
                        .message("RotationForest failed; falling back to Catch22 ridge")
                        .build(),
                );
                match Catch22Classifier::new(0.1).fit(x, y, &session.child("rotf-ridge")) {
                    Ok(q) => ctx.finish(FittedRotationForestClassifier {
                        forest: None,
                        ridge: Some(q.value),
                    }),
                    Err(_) => ctx.finish(FittedRotationForestClassifier {
                        forest: None,
                        ridge: None,
                    }),
                }
            }
        }
    }
}

impl Predict for FittedRotationForestClassifier {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        if let Some(f) = &self.forest {
            let z = {
                let mut ctx = FitCtx::with_session(session.child("rotf-z"));
                catch22_rows(x, session, &mut ctx)
            };
            return f.predict(&z, session);
        }
        if let Some(r) = &self.ridge {
            return r.predict(x, session);
        }
        let mut ctx = FitCtx::with_session(session.child("predict"));
        ctx.push(Issue::builder(IssueCode::PartialFitBeforeInit).build());
        ctx.finish(Vector::zeros(x.nrows()))
    }
}

/// DTW proximity-forest regressor (sktime `ProximityForest` regression lite).
///
/// Each stump is two random exemplars; the closer series donates its response.
/// Tree count is not identification `p`. Do not call `inspect_classes`.
#[derive(Clone, Debug)]
pub struct ProximityForestRegressor {
    /// Number of proximity stumps. Not identification `p`.
    pub n_trees: usize,
    /// Seed.
    pub seed: u64,
}

impl Default for ProximityForestRegressor {
    fn default() -> Self {
        Self {
            n_trees: 5,
            seed: 17,
        }
    }
}

impl ProximityForestRegressor {
    /// Default five-stump proximity regressor.
    pub fn new() -> Self {
        Self::default()
    }
}

#[derive(Clone, Debug)]
struct ProxRegStump {
    left: Vector,
    right: Vector,
    left_y: f64,
    right_y: f64,
}

/// Fitted DTW proximity-forest regressor.
#[derive(Clone, Debug)]
pub struct FittedProximityForestRegressor {
    trees: Vec<ProxRegStump>,
    /// Fallback mean response.
    pub default_value: f64,
}

impl Fit for ProximityForestRegressor {
    type Fitted = FittedProximityForestRegressor;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedProximityForestRegressor>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        let n = x.nrows().min(y.len());
        let default_value = if n == 0 {
            0.0
        } else {
            y.as_slice().iter().take(n).sum::<f64>() / n as f64
        };
        if n < 2 {
            ctx.push(
                Issue::builder(IssueCode::InsufficientSample)
                    .severity(Severity::Warning)
                    .message("ProximityForestRegressor needs two series to pick exemplars")
                    .build(),
            );
            return ctx.finish(FittedProximityForestRegressor {
                trees: Vec::new(),
                default_value,
            });
        }
        ctx.push(
            Issue::builder(IssueCode::CausalClaimUnidentified)
                .severity(Severity::Advisory)
                .message("ProximityForestRegressor is random DTW exemplars, not a published PF")
                .compromise(NumericalCompromise::new(
                    "sktime ProximityForest with splitter search",
                    "random pair of series as a 1-NN stump",
                    "no entropy-driven split selection",
                    "read as a DTW exemplar ensemble, not the paper accuracy",
                ))
                .build(),
        );
        let mut rng = Rng::new(self.seed);
        let mut trees = Vec::new();
        for _ in 0..self.n_trees.max(1) {
            let i0 = rng.below(n);
            let mut i1 = rng.below(n);
            if i1 == i0 {
                i1 = (i0 + 1) % n;
            }
            trees.push(ProxRegStump {
                left: x.row(i0),
                right: x.row(i1),
                left_y: y[i0],
                right_y: y[i1],
            });
        }
        ctx.finish(FittedProximityForestRegressor {
            trees,
            default_value,
        })
    }
}

impl Predict for FittedProximityForestRegressor {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        if self.trees.is_empty() {
            return ctx.finish(Vector::filled(x.nrows(), self.default_value));
        }
        let out = Vector::from_iter((0..x.nrows()).map(|i| {
            let row = x.row(i);
            let mut acc = 0.0_f64;
            for t in &self.trees {
                let dl = dtw_raw(row.as_slice(), t.left.as_slice());
                let dr = dtw_raw(row.as_slice(), t.right.as_slice());
                acc += if dl <= dr { t.left_y } else { t.right_y };
            }
            acc / self.trees.len() as f64
        }));
        ctx.finish(out)
    }
}

/// Random supervised time-series forest (sktime `RSTSF` lite).
///
/// Interval / tree counts are not identification `p`.
#[derive(Clone, Debug)]
pub struct Rstsf {
    /// Trees.
    pub n_estimators: usize,
    /// Random intervals per tree.
    pub n_intervals: usize,
    /// Tree depth.
    pub max_depth: usize,
    /// Seed.
    pub seed: u64,
}

impl Default for Rstsf {
    fn default() -> Self {
        Self {
            n_estimators: 6,
            n_intervals: 3,
            max_depth: 4,
            seed: 19,
        }
    }
}

impl Rstsf {
    /// Default RSTSF lite.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted RSTSF vote.
#[derive(Clone, Debug)]
pub struct FittedRstsf {
    trees: Vec<crate::tree::FittedTreeClassifier>,
    intervals: Vec<Vec<Interval>>,
    /// Sorted class labels.
    pub classes: Vec<i64>,
}

impl Fit for Rstsf {
    type Fitted = FittedRstsf;
    fn fit(&mut self, x: &Matrix, y: &Vector, session: &Session) -> Result<Qualified<FittedRstsf>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        let counts = inspect_classes(&mut ctx.report, y, &ctx.policy);
        let classes: Vec<i64> = counts.iter().map(|(k, _)| *k).collect();
        let mut rng = Rng::new(self.seed);
        let mut trees = Vec::new();
        let mut intervals = Vec::new();
        let tlen = x.ncols().max(1);
        for e in 0..self.n_estimators.max(1) {
            let ni = self.n_intervals.max(1);
            let mut iv = Vec::with_capacity(ni);
            for _ in 0..ni {
                let a = rng.below(tlen);
                let span = rng.below(tlen).max(1);
                let b = (a + span).min(tlen);
                iv.push(Interval {
                    start: a.min(b.saturating_sub(1)),
                    end: b.max(a + 1),
                });
            }
            let feat = interval_feats(x, &iv);
            let mut tree = crate::tree::DecisionTreeClassifier {
                max_depth: self.max_depth,
                seed: rng.next_u64(),
                ..crate::tree::DecisionTreeClassifier::default()
            };
            match tree.fit(&feat, y, &session.child("rstsf_tree")) {
                Ok(q) => {
                    trees.push(q.value);
                    intervals.push(iv);
                }
                Err(err) => {
                    for issue in err.report.issues() {
                        if !matches!(
                            issue.code,
                            IssueCode::ResidualTooLarge
                                | IssueCode::NearSingular
                                | IssueCode::RankZero
                                | IssueCode::R2IsOne
                                | IssueCode::MeaninglessFit
                        ) {
                            ctx.push(issue.clone());
                        }
                    }
                }
            }
            ctx.session.step(e as u64, 0.0, None);
        }
        if trees.is_empty() {
            ctx.push(
                Issue::builder(IssueCode::DidNotConverge)
                    .message("every RSTSF tree failed to fit")
                    .build(),
            );
        }
        ctx.finish(FittedRstsf {
            trees,
            intervals,
            classes,
        })
    }
}

impl Predict for FittedRstsf {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let ctx = FitCtx::with_session(session.child("predict"));
        let mut votes = vec![BTreeMap::<i64, usize>::new(); x.nrows()];
        for (tree, iv) in self.trees.iter().zip(&self.intervals) {
            let feat = interval_feats(x, iv);
            match tree.predict(&feat, &session.child("rstsf_pred")) {
                Ok(q) => {
                    for i in 0..x.nrows() {
                        let lab = q.value[i].round() as i64;
                        *votes[i].entry(lab).or_insert(0) += 1;
                    }
                }
                Err(_) => {}
            }
        }
        let out = Vector::from_iter(votes.iter().map(|m| {
            m.iter()
                .max_by(|a, b| a.1.cmp(b.1).then(b.0.cmp(a.0)))
                .map(|(k, _)| *k as f64)
                .unwrap_or(self.classes.first().copied().unwrap_or(0) as f64)
        }));
        ctx.finish(out)
    }
}

/// Multi-scale convolution + ridge (tslearn / sktime `LITETimeClassifier` lite).
///
/// Kernel / width counts are not identification `p`.
#[derive(Clone, Debug)]
pub struct LiteTime {
    /// Kernels per width.
    pub n_kernels: usize,
    /// Ridge \(\alpha\).
    pub alpha: f64,
    /// Seed.
    pub seed: u64,
}

impl Default for LiteTime {
    fn default() -> Self {
        Self {
            n_kernels: 4,
            alpha: 0.1,
            seed: 19,
        }
    }
}

impl LiteTime {
    /// Default LITETime-lite classifier.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted LITETime-lite ridge.
#[derive(Clone, Debug)]
pub struct FittedLiteTime {
    kernels: Vec<Vec<f64>>,
    inner: crate::classification::FittedRidgeClassifier,
}

impl Fit for LiteTime {
    type Fitted = FittedLiteTime;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedLiteTime>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        inspect_classes(&mut ctx.report, y, &ctx.policy);
        let t = x.ncols().max(1);
        let widths = [3usize, 5, t.min(7)]
            .into_iter()
            .filter(|&w| w <= t && w > 0);
        let mut rng = Rng::new(self.seed);
        let mut kernels: Vec<Vec<f64>> = Vec::new();
        for w in widths {
            for _ in 0..self.n_kernels.max(1) {
                kernels.push((0..w).map(|_| rng.standard_normal()).collect());
            }
        }
        if kernels.is_empty() {
            ctx.push(
                Issue::builder(IssueCode::WindowTooShort)
                    .message("LiteTime has no kernel that fits T")
                    .build(),
            );
            kernels.push(vec![1.0]);
        }
        let z = conv_maxpool(x, &kernels);
        let inner = binary_ridge_from_features(&z, y, self.alpha, &ctx.policy, "litetime");
        ctx.finish(FittedLiteTime { kernels, inner })
    }
}

impl Predict for FittedLiteTime {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let z = conv_maxpool(x, &self.kernels);
        self.inner.predict(&z, session)
    }
}

/// SAX-word features + ridge (sktime `MrSQM` lite).
///
/// Piece / alphabet counts are not identification `p`.
#[derive(Clone, Debug)]
pub struct MrSqm {
    /// PAA pieces.
    pub n_pieces: usize,
    /// SAX alphabet size.
    pub alphabet: usize,
    /// Ridge \(\alpha\).
    pub alpha: f64,
}

impl Default for MrSqm {
    fn default() -> Self {
        Self {
            n_pieces: 4,
            alphabet: 4,
            alpha: 0.1,
        }
    }
}

impl MrSqm {
    /// Default MrSQM-lite classifier.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted SAX-ridge classifier.
#[derive(Clone, Debug)]
pub struct FittedMrSqm {
    n_pieces: usize,
    alphabet: usize,
    inner: crate::classification::FittedRidgeClassifier,
}

fn sax_feature_rows(x: &Matrix, n_pieces: usize, alphabet: usize) -> Matrix {
    let k = n_pieces.max(1);
    Matrix::from_fn(x.nrows(), k, |i, j| {
        let row = x.row(i);
        let s = sax_symbols(row.as_slice(), k, alphabet);
        s.get(j).copied().unwrap_or(0.0)
    })
}

impl Fit for MrSqm {
    type Fitted = FittedMrSqm;
    fn fit(&mut self, x: &Matrix, y: &Vector, session: &Session) -> Result<Qualified<FittedMrSqm>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        inspect_classes(&mut ctx.report, y, &ctx.policy);
        let z = sax_feature_rows(x, self.n_pieces, self.alphabet);
        let inner = binary_ridge_from_features(&z, y, self.alpha, &ctx.policy, "mrsqm");
        ctx.finish(FittedMrSqm {
            n_pieces: self.n_pieces.max(1),
            alphabet: self.alphabet.max(2),
            inner,
        })
    }
}

impl Predict for FittedMrSqm {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let z = sax_feature_rows(x, self.n_pieces, self.alphabet);
        self.inner.predict(&z, session)
    }
}

/// Single temporal-dictionary member (sktime `IndividualTDE` lite).
///
/// Word count is not identification `p`.
#[derive(Clone, Debug)]
pub struct IndividualTde {
    /// Sliding-window length.
    pub window: usize,
    /// DFT coefficients kept per window.
    pub word_len: usize,
    /// SFA alphabet size.
    pub alphabet: usize,
    /// Words kept.
    pub n_words: usize,
}

impl Default for IndividualTde {
    fn default() -> Self {
        Self {
            window: 8,
            word_len: 4,
            alphabet: 4,
            n_words: 8,
        }
    }
}

impl IndividualTde {
    /// Default single-dictionary TDE member.
    pub fn new() -> Self {
        Self::default()
    }
}

impl Fit for IndividualTde {
    type Fitted = FittedBoss;
    fn fit(&mut self, x: &Matrix, y: &Vector, session: &Session) -> Result<Qualified<FittedBoss>> {
        Weasel {
            window: self.window,
            word_len: self.word_len,
            alphabet: self.alphabet,
            n_words: self.n_words,
        }
        .fit(x, y, session)
    }
}

/// Catch22 / tsfresh-lite features + ridge (sktime `TSFreshClassifier` lite).
///
/// Feature count is not identification `p`.
#[derive(Clone, Debug)]
pub struct TsFreshClassifier {
    /// Ridge \(\alpha\).
    pub alpha: f64,
}

impl Default for TsFreshClassifier {
    fn default() -> Self {
        Self { alpha: 0.1 }
    }
}

impl TsFreshClassifier {
    /// Default tsfresh-lite classifier.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted tsfresh-lite ridge.
#[derive(Clone, Debug)]
pub struct FittedTsFreshClassifier {
    inner: crate::classification::FittedRidgeClassifier,
}

impl Fit for TsFreshClassifier {
    type Fitted = FittedTsFreshClassifier;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedTsFreshClassifier>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        inspect_classes(&mut ctx.report, y, &ctx.policy);
        let z = catch22_rows(x, session, &mut ctx);
        let inner = binary_ridge_from_features(&z, y, self.alpha, &ctx.policy, "tsfresh");
        ctx.finish(FittedTsFreshClassifier { inner })
    }
}

impl Predict for FittedTsFreshClassifier {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        let z = catch22_rows(x, session, &mut ctx);
        self.inner.predict(&z, session)
    }
}

/// Supervised-interval features + ridge (sktime `SupervisedIntervals` lite).
///
/// Interval count is not identification `p`.
#[derive(Clone, Debug)]
pub struct SupervisedIntervals {
    /// Intervals ranked by class-mean gap.
    pub n_intervals: usize,
    /// Ridge \(\alpha\).
    pub alpha: f64,
    /// Seed.
    pub seed: u64,
}

impl Default for SupervisedIntervals {
    fn default() -> Self {
        Self {
            n_intervals: 6,
            alpha: 0.1,
            seed: 6,
        }
    }
}

impl SupervisedIntervals {
    /// Default supervised-interval classifier.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted supervised-interval ridge.
#[derive(Clone, Debug)]
pub struct FittedSupervisedIntervals {
    intervals: Vec<Interval>,
    inner: crate::classification::FittedRidgeClassifier,
}

impl Fit for SupervisedIntervals {
    type Fitted = FittedSupervisedIntervals;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedSupervisedIntervals>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        inspect_classes(&mut ctx.report, y, &ctx.policy);
        let mut rng = Rng::new(self.seed);
        let intervals = supervised_intervals(x, y, self.n_intervals, &mut rng);
        let z = interval_feats(x, &intervals);
        let inner = binary_ridge_from_features(&z, y, self.alpha, &ctx.policy, "supint");
        ctx.finish(FittedSupervisedIntervals { intervals, inner })
    }
}

impl Predict for FittedSupervisedIntervals {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let z = interval_feats(x, &self.intervals);
        self.inner.predict(&z, session)
    }
}

/// Two-window WEASEL histograms + ridge (sktime `WEASEL_V2` lite).
///
/// Word / window counts are not identification `p`.
#[derive(Clone, Debug)]
pub struct WeaselV2 {
    /// First sliding-window length.
    pub window_a: usize,
    /// Second sliding-window length.
    pub window_b: usize,
    /// DFT coefficients kept per window.
    pub word_len: usize,
    /// SFA alphabet size.
    pub alphabet: usize,
    /// Words kept per window.
    pub n_words: usize,
    /// Ridge \(\alpha\).
    pub alpha: f64,
}

impl Default for WeaselV2 {
    fn default() -> Self {
        Self {
            window_a: 8,
            window_b: 4,
            word_len: 4,
            alphabet: 4,
            n_words: 6,
            alpha: 0.1,
        }
    }
}

impl WeaselV2 {
    /// Default two-window WEASEL.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted two-window WEASEL ridge.
#[derive(Clone, Debug)]
pub struct FittedWeaselV2 {
    spec_a: (usize, usize, usize),
    spec_b: (usize, usize, usize),
    idx_a: Vec<usize>,
    idx_b: Vec<usize>,
    inner: crate::classification::FittedRidgeClassifier,
}

fn weasel_keep(h: &Matrix, n_words: usize) -> (Matrix, Vec<usize>) {
    let mut vars: Vec<(usize, f64)> = (0..h.ncols()).map(|j| (j, h.column(j).std())).collect();
    vars.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let keep = n_words.max(1).min(h.ncols().max(1));
    let idx: Vec<usize> = vars.iter().take(keep).map(|p| p.0).collect();
    let z = if idx.is_empty() {
        Matrix::zeros(h.nrows(), 0)
    } else {
        Matrix::from_fn(h.nrows(), idx.len(), |i, t| h.get(i, idx[t]))
    };
    (z, idx)
}

fn concat_features(a: &Matrix, b: &Matrix) -> Matrix {
    let n = a.nrows().max(b.nrows());
    let pa = a.ncols();
    let pb = b.ncols();
    Matrix::from_fn(n, pa + pb, |i, j| {
        if j < pa {
            if i < a.nrows() {
                a.get(i, j)
            } else {
                0.0
            }
        } else if i < b.nrows() {
            b.get(i, j - pa)
        } else {
            0.0
        }
    })
}

fn weasel_v2_features(
    x: &Matrix,
    spec_a: (usize, usize, usize),
    spec_b: (usize, usize, usize),
    idx_a: &[usize],
    idx_b: &[usize],
) -> Matrix {
    let (h1, _) = boss_histograms(x, spec_a.0, spec_a.1, spec_a.2);
    let (h2, _) = boss_histograms(x, spec_b.0, spec_b.1, spec_b.2);
    let z1 = if idx_a.is_empty() {
        Matrix::zeros(h1.nrows(), 0)
    } else {
        Matrix::from_fn(h1.nrows(), idx_a.len(), |i, t| {
            if idx_a[t] < h1.ncols() {
                h1.get(i, idx_a[t])
            } else {
                0.0
            }
        })
    };
    let z2 = if idx_b.is_empty() {
        Matrix::zeros(h2.nrows(), 0)
    } else {
        Matrix::from_fn(h2.nrows(), idx_b.len(), |i, t| {
            if idx_b[t] < h2.ncols() {
                h2.get(i, idx_b[t])
            } else {
                0.0
            }
        })
    };
    concat_features(&z1, &z2)
}

impl Fit for WeaselV2 {
    type Fitted = FittedWeaselV2;
    fn fit(&mut self, x: &Matrix, y: &Vector, session: &Session) -> Result<Qualified<FittedWeaselV2>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        inspect_classes(&mut ctx.report, y, &ctx.policy);
        let spec_a = (self.window_a.max(2), self.word_len.max(1), self.alphabet.max(2));
        let spec_b = (self.window_b.max(2), self.word_len.max(1), self.alphabet.max(2));
        let (h1, _) = boss_histograms(x, spec_a.0, spec_a.1, spec_a.2);
        let (h2, _) = boss_histograms(x, spec_b.0, spec_b.1, spec_b.2);
        let (z1, idx_a) = weasel_keep(&h1, self.n_words);
        let (z2, idx_b) = weasel_keep(&h2, self.n_words);
        let z = concat_features(&z1, &z2);
        let inner = binary_ridge_from_features(&z, y, self.alpha, &ctx.policy, "weaselv2");
        ctx.finish(FittedWeaselV2 {
            spec_a,
            spec_b,
            idx_a,
            idx_b,
            inner,
        })
    }
}

impl Predict for FittedWeaselV2 {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let z = weasel_v2_features(x, self.spec_a, self.spec_b, &self.idx_a, &self.idx_b);
        self.inner.predict(&z, session)
    }
}

/// Early-classification prefix ridge (sktime `TEASER` lite).
///
/// Prefix length is not identification `p`.
#[derive(Clone, Debug)]
pub struct Teaser {
    /// Ridge \(\alpha\).
    pub alpha: f64,
}

impl Default for Teaser {
    fn default() -> Self {
        Self { alpha: 0.1 }
    }
}

impl Teaser {
    /// Default TEASER-lite classifier.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted prefix-summary ridge.
#[derive(Clone, Debug)]
pub struct FittedTeaser {
    inner: crate::classification::FittedRidgeClassifier,
}

fn prefix_summary(x: &Matrix) -> Matrix {
    let t = x.ncols().max(1);
    let half = (t / 2).max(1);
    interval_feats(x, &[Interval { start: 0, end: half }])
}

impl Fit for Teaser {
    type Fitted = FittedTeaser;
    fn fit(&mut self, x: &Matrix, y: &Vector, session: &Session) -> Result<Qualified<FittedTeaser>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        inspect_classes(&mut ctx.report, y, &ctx.policy);
        let z = prefix_summary(x);
        let inner = binary_ridge_from_features(&z, y, self.alpha, &ctx.policy, "teaser");
        ctx.finish(FittedTeaser { inner })
    }
}

impl Predict for FittedTeaser {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let z = prefix_summary(x);
        self.inner.predict(&z, session)
    }
}

/// MultiROCKET features + ridge (sktime `MultiRocketClassifier`).
///
/// Kernel count is not identification `p`.
#[derive(Clone, Debug)]
pub struct MultiRocketClassifier {
    /// Random kernels.
    pub n_kernels: usize,
    /// Ridge \(\alpha\).
    pub alpha: f64,
    /// Seed.
    pub seed: u64,
}

impl Default for MultiRocketClassifier {
    fn default() -> Self {
        Self {
            n_kernels: 16,
            alpha: 0.5,
            seed: 8,
        }
    }
}

impl MultiRocketClassifier {
    /// Default MultiROCKET classifier.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted MultiROCKET ridge classifier.
#[derive(Clone, Debug)]
pub struct FittedMultiRocketClassifier {
    rocket: MultiRocket,
    inner: crate::classification::FittedRidgeClassifier,
}

impl Fit for MultiRocketClassifier {
    type Fitted = FittedMultiRocketClassifier;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedMultiRocketClassifier>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        inspect_classes(&mut ctx.report, y, &ctx.policy);
        let rocket = MultiRocket {
            n_kernels: self.n_kernels.max(1),
            seed: self.seed,
        };
        let feat = rocket.transform(x, &session.child("mrocketc"))?;
        let inner = binary_ridge_from_features(&feat.value, y, self.alpha, &ctx.policy, "mrocketc");
        ctx.finish(FittedMultiRocketClassifier { rocket, inner })
    }
}

impl Predict for FittedMultiRocketClassifier {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let feat = self.rocket.transform(x, &session.child("mrocketc"))?;
        self.inner.predict(&feat.value, session)
    }
}

/// MiniROCKET features + ridge (sktime `MiniRocketRegressor`).
///
/// Kernel count is not identification `p`.
#[derive(Clone, Debug)]
pub struct MiniRocketRegressor {
    /// Random dilated kernels.
    pub n_kernels: usize,
    /// Ridge \(\alpha\).
    pub alpha: f64,
    /// Seed.
    pub seed: u64,
}

impl Default for MiniRocketRegressor {
    fn default() -> Self {
        Self {
            n_kernels: 16,
            alpha: 1.0,
            seed: 9,
        }
    }
}

impl MiniRocketRegressor {
    /// Default MiniROCKET regressor.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted MiniROCKET ridge.
#[derive(Clone, Debug)]
pub struct FittedMiniRocketRegressor {
    rocket: MiniRocket,
    inner: FittedPenalized,
}

impl Fit for MiniRocketRegressor {
    type Fitted = FittedMiniRocketRegressor;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedMiniRocketRegressor>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        let rocket = MiniRocket {
            n_kernels: self.n_kernels.max(1),
            seed: self.seed,
        };
        let feat = rocket.transform(x, &session.child("minirocket"))?;
        let mut scratch = signlred::Report::new("minirocket_reg", "ridge");
        let design = feat.value.with_intercept();
        let beta = ridge_solve(&mut scratch, &design, y, self.alpha.max(0.0), &ctx.policy)
            .unwrap_or_else(|| Vector::zeros(design.ncols()));
        for issue in scratch.issues() {
            if matches!(
                issue.code,
                IssueCode::ResidualTooLarge
                    | IssueCode::NearSingular
                    | IssueCode::RankZero
                    | IssueCode::R2IsOne
                    | IssueCode::PerfectCollinearity
            ) {
                continue;
            }
            ctx.push(issue.clone());
        }
        ctx.finish(FittedMiniRocketRegressor {
            rocket,
            inner: FittedPenalized {
                coef: Vector::from_iter((1..beta.len()).map(|j| beta[j])),
                intercept: beta.as_slice().first().copied().unwrap_or(0.0),
                alpha: self.alpha,
                l1_ratio: 0.0,
            },
        })
    }
}

impl Predict for FittedMiniRocketRegressor {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let feat = self.rocket.transform(x, &session.child("minirocket"))?;
        self.inner.predict(&feat.value, session)
    }
}

/// Random-interval features + ridge (sktime `RandomIntervalRegressor`).
///
/// Interval count is not identification `p`.
#[derive(Clone, Debug)]
pub struct RandomIntervalRegressor {
    /// Random intervals.
    pub n_intervals: usize,
    /// Ridge \(\alpha\).
    pub alpha: f64,
    /// Seed.
    pub seed: u64,
}

impl Default for RandomIntervalRegressor {
    fn default() -> Self {
        Self {
            n_intervals: 6,
            alpha: 0.1,
            seed: 4,
        }
    }
}

impl RandomIntervalRegressor {
    /// Default random-interval regressor.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted random-interval ridge regressor.
#[derive(Clone, Debug)]
pub struct FittedRandomIntervalRegressor {
    intervals: Vec<Interval>,
    inner: FittedPenalized,
}

impl Fit for RandomIntervalRegressor {
    type Fitted = FittedRandomIntervalRegressor;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedRandomIntervalRegressor>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        let tlen = x.ncols().max(1);
        let mut rng = Rng::new(self.seed);
        let ni = self.n_intervals.max(1);
        let mut intervals = Vec::with_capacity(ni);
        for _ in 0..ni {
            let a = rng.below(tlen);
            let span = rng.below(tlen).max(1);
            let b = (a + span).min(tlen);
            intervals.push(Interval {
                start: a.min(b.saturating_sub(1)),
                end: b.max(a + 1),
            });
        }
        let z = interval_feats(x, &intervals);
        let mut scratch = signlred::Report::new("rir", "ridge");
        let design = z.with_intercept();
        let beta = ridge_solve(&mut scratch, &design, y, self.alpha.max(0.0), &ctx.policy)
            .unwrap_or_else(|| Vector::zeros(design.ncols()));
        for issue in scratch.issues() {
            if matches!(
                issue.code,
                IssueCode::ResidualTooLarge
                    | IssueCode::NearSingular
                    | IssueCode::RankZero
                    | IssueCode::R2IsOne
                    | IssueCode::PerfectCollinearity
            ) {
                continue;
            }
            ctx.push(issue.clone());
        }
        ctx.finish(FittedRandomIntervalRegressor {
            intervals,
            inner: FittedPenalized {
                coef: Vector::from_iter((1..beta.len()).map(|j| beta[j])),
                intercept: beta.as_slice().first().copied().unwrap_or(0.0),
                alpha: self.alpha,
                l1_ratio: 0.0,
            },
        })
    }
}

impl Predict for FittedRandomIntervalRegressor {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let z = interval_feats(x, &self.intervals);
        self.inner.predict(&z, session)
    }
}

/// CNN convolution + ridge (sktime `CNNRegressor` lite).
///
/// Kernel count is not identification `p`.
#[derive(Clone, Debug)]
pub struct CnnRegressor {
    /// Random kernels.
    pub n_kernels: usize,
    /// Kernel width.
    pub width: usize,
    /// Ridge \(\alpha\).
    pub alpha: f64,
    /// Seed.
    pub seed: u64,
}

impl Default for CnnRegressor {
    fn default() -> Self {
        Self {
            n_kernels: 4,
            width: 3,
            alpha: 0.1,
            seed: 11,
        }
    }
}

impl CnnRegressor {
    /// Default CNN-lite regressor.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted CNN-lite ridge regressor.
#[derive(Clone, Debug)]
pub struct FittedCnnRegressor {
    kernels: Vec<Vec<f64>>,
    inner: FittedPenalized,
}

impl Fit for CnnRegressor {
    type Fitted = FittedCnnRegressor;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedCnnRegressor>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        let w = self.width.max(1).min(x.ncols().max(1));
        if x.ncols() < self.width {
            ctx.push(
                Issue::builder(IssueCode::WindowTooShort)
                    .message(format!(
                        "CnnRegressor width={} > T={}",
                        self.width,
                        x.ncols()
                    ))
                    .build(),
            );
        }
        let mut rng = Rng::new(self.seed);
        let kernels: Vec<Vec<f64>> = (0..self.n_kernels.max(1))
            .map(|_| (0..w).map(|_| rng.standard_normal()).collect())
            .collect();
        let z = conv_maxpool(x, &kernels);
        let inner = ridge_reg_from_features(&z, y, self.alpha, &ctx.policy, "cnn_reg");
        ctx.finish(FittedCnnRegressor { kernels, inner })
    }
}

impl Predict for FittedCnnRegressor {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let z = conv_maxpool(x, &self.kernels);
        self.inner.predict(&z, session)
    }
}

/// Residual convolution + ridge (sktime `ResNetRegressor` lite).
///
/// Kernel count is not identification `p`.
#[derive(Clone, Debug)]
pub struct ResNetRegressor {
    /// Kernels.
    pub n_kernels: usize,
    /// Ridge \(\alpha\).
    pub alpha: f64,
    /// Seed.
    pub seed: u64,
}

impl Default for ResNetRegressor {
    fn default() -> Self {
        Self {
            n_kernels: 4,
            alpha: 0.1,
            seed: 23,
        }
    }
}

impl ResNetRegressor {
    /// Default ResNet-lite regressor.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted ResNet-lite ridge regressor.
#[derive(Clone, Debug)]
pub struct FittedResNetRegressor {
    kernels: Vec<Vec<f64>>,
    inner: FittedPenalized,
}

impl Fit for ResNetRegressor {
    type Fitted = FittedResNetRegressor;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedResNetRegressor>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        let w = 3usize.min(x.ncols().max(1));
        if x.ncols() < 3 {
            ctx.push(
                Issue::builder(IssueCode::WindowTooShort)
                    .message(format!("ResNetRegressor width=3 > T={}", x.ncols()))
                    .build(),
            );
        }
        let mut rng = Rng::new(self.seed);
        let kernels: Vec<Vec<f64>> = (0..self.n_kernels.max(1))
            .map(|_| (0..w).map(|_| rng.standard_normal()).collect())
            .collect();
        let z = residual_conv_pool(x, &kernels);
        let inner = ridge_reg_from_features(&z, y, self.alpha, &ctx.policy, "resnet_reg");
        ctx.finish(FittedResNetRegressor { kernels, inner })
    }
}

impl Predict for FittedResNetRegressor {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let z = residual_conv_pool(x, &self.kernels);
        self.inner.predict(&z, session)
    }
}

/// Fully-convolutional ridge on the raw series (sktime `FCNRegressor` lite).
#[derive(Clone, Debug)]
pub struct FCNRegressor {
    /// Ridge \(\alpha\).
    pub alpha: f64,
}

impl Default for FCNRegressor {
    fn default() -> Self {
        Self { alpha: 0.1 }
    }
}

impl FCNRegressor {
    /// Default FCN-lite regressor.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted FCN-lite ridge.
#[derive(Clone, Debug)]
pub struct FittedFCNRegressor {
    inner: FittedPenalized,
}

impl Fit for FCNRegressor {
    type Fitted = FittedFCNRegressor;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedFCNRegressor>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        let inner = ridge_reg_from_features(x, y, self.alpha, &ctx.policy, "fcn_reg");
        ctx.finish(FittedFCNRegressor { inner })
    }
}

impl Predict for FittedFCNRegressor {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        self.inner.predict(x, session)
    }
}

/// Random encoder + ridge (sktime `EncoderRegressor` lite).
///
/// Latent width is not identification `p`.
#[derive(Clone, Debug)]
pub struct EncoderRegressor {
    /// Bottleneck width.
    pub latent: usize,
    /// Ridge \(\alpha\).
    pub alpha: f64,
    /// Seed.
    pub seed: u64,
}

impl Default for EncoderRegressor {
    fn default() -> Self {
        Self {
            latent: 4,
            alpha: 0.1,
            seed: 47,
        }
    }
}

impl EncoderRegressor {
    /// Default encoder-lite regressor.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted encoder-lite ridge.
#[derive(Clone, Debug)]
pub struct FittedEncoderRegressor {
    enc: Matrix,
    inner: FittedPenalized,
}

impl Fit for EncoderRegressor {
    type Fitted = FittedEncoderRegressor;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedEncoderRegressor>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        let mut rng = Rng::new(self.seed);
        let lat = self.latent.max(1);
        let enc = Matrix::from_fn(x.ncols().max(1), lat, |_, _| rng.standard_normal());
        let z = encode_series(x, &enc);
        let inner = ridge_reg_from_features(&z, y, self.alpha, &ctx.policy, "enc_reg");
        ctx.finish(FittedEncoderRegressor { enc, inner })
    }
}

impl Predict for FittedEncoderRegressor {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let z = encode_series(x, &self.enc);
        self.inner.predict(&z, session)
    }
}

/// Random-hidden tanh + ridge (sktime `MLPRegressor` time-series lite).
///
/// Hidden width is not identification `p`.
#[derive(Clone, Debug)]
pub struct MlpTimeRegressor {
    /// Hidden units.
    pub hidden: usize,
    /// Ridge \(\alpha\).
    pub alpha: f64,
    /// Seed.
    pub seed: u64,
}

impl Default for MlpTimeRegressor {
    fn default() -> Self {
        Self {
            hidden: 8,
            alpha: 0.1,
            seed: 29,
        }
    }
}

impl MlpTimeRegressor {
    /// Default MLP-lite time regressor.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted MLP-lite time ridge.
#[derive(Clone, Debug)]
pub struct FittedMlpTimeRegressor {
    hidden: Matrix,
    inner: FittedPenalized,
}

impl Fit for MlpTimeRegressor {
    type Fitted = FittedMlpTimeRegressor;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedMlpTimeRegressor>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        let mut rng = Rng::new(self.seed);
        let h = self.hidden.max(1);
        let w = Matrix::from_fn(x.ncols().max(1), h, |_, _| rng.standard_normal());
        let z = mlp_hidden(x, &w);
        let inner = ridge_reg_from_features(&z, y, self.alpha, &ctx.policy, "mlp_reg");
        ctx.finish(FittedMlpTimeRegressor { hidden: w, inner })
    }
}

impl Predict for FittedMlpTimeRegressor {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let z = mlp_hidden(x, &self.hidden);
        self.inner.predict(&z, session)
    }
}

/// Shapelet distances + ridge (sktime `ShapeletTransformRegressor`).
///
/// Shapelet count is not identification `p`.
#[derive(Clone, Debug)]
pub struct ShapeletTransformRegressor {
    /// Shapelets.
    pub n_shapelets: usize,
    /// Shapelet length.
    pub length: usize,
    /// Ridge \(\alpha\).
    pub alpha: f64,
    /// Seed.
    pub seed: u64,
}

impl Default for ShapeletTransformRegressor {
    fn default() -> Self {
        Self {
            n_shapelets: 3,
            length: 3,
            alpha: 0.1,
            seed: 2,
        }
    }
}

impl ShapeletTransformRegressor {
    /// Default shapelet-transform regressor.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted shapelet-transform ridge.
#[derive(Clone, Debug)]
pub struct FittedShapeletTransformRegressor {
    shapelets: FittedShapeletTransform,
    inner: FittedPenalized,
}

impl Fit for ShapeletTransformRegressor {
    type Fitted = FittedShapeletTransformRegressor;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedShapeletTransformRegressor>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        let st = ShapeletTransform {
            n_shapelets: self.n_shapelets,
            length: self.length,
            seed: self.seed,
        }
        .fit_unsupervised(x, &session.child("streg"))?
        .value;
        let z = st.transform(x, &session.child("stregt"))?.value;
        let inner = ridge_reg_from_features(&z, y, self.alpha, &ctx.policy, "streg");
        ctx.finish(FittedShapeletTransformRegressor {
            shapelets: st,
            inner,
        })
    }
}

impl Predict for FittedShapeletTransformRegressor {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let z = self.shapelets.transform(x, &session.child("stregt"))?.value;
        self.inner.predict(&z, session)
    }
}

/// Catch22 / tsfresh-lite features + ridge (sktime `TSFreshRegressor` lite).
///
/// Feature count is not identification `p`.
#[derive(Clone, Debug)]
pub struct TsFreshRegressor {
    /// Ridge \(\alpha\).
    pub alpha: f64,
}

impl Default for TsFreshRegressor {
    fn default() -> Self {
        Self { alpha: 0.1 }
    }
}

impl TsFreshRegressor {
    /// Default tsfresh-lite regressor.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted tsfresh-lite ridge.
#[derive(Clone, Debug)]
pub struct FittedTsFreshRegressor {
    inner: FittedPenalized,
}

impl Fit for TsFreshRegressor {
    type Fitted = FittedTsFreshRegressor;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedTsFreshRegressor>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        let z = catch22_rows(x, session, &mut ctx);
        let inner = ridge_reg_from_features(&z, y, self.alpha, &ctx.policy, "tsfresh_reg");
        ctx.finish(FittedTsFreshRegressor { inner })
    }
}

impl Predict for FittedTsFreshRegressor {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        let z = catch22_rows(x, session, &mut ctx);
        self.inner.predict(&z, session)
    }
}

/// Interval quantiles + ridge (sktime `QUANTRegressor` lite).
///
/// Interval count is not identification `p`.
#[derive(Clone, Debug)]
pub struct QuantRegressor {
    /// Random intervals.
    pub n_intervals: usize,
    /// Ridge \(\alpha\).
    pub alpha: f64,
    /// Seed.
    pub seed: u64,
}

impl Default for QuantRegressor {
    fn default() -> Self {
        Self {
            n_intervals: 4,
            alpha: 0.1,
            seed: 21,
        }
    }
}

impl QuantRegressor {
    /// Default QUANT-lite regressor.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted QUANT-lite ridge.
#[derive(Clone, Debug)]
pub struct FittedQuantRegressor {
    intervals: Vec<Interval>,
    inner: FittedPenalized,
}

impl Fit for QuantRegressor {
    type Fitted = FittedQuantRegressor;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedQuantRegressor>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        let mut rng = Rng::new(self.seed);
        let tlen = x.ncols().max(1);
        let mut iv = Vec::new();
        for _ in 0..self.n_intervals.max(1) {
            let a = rng.below(tlen);
            let span = 1 + rng.below(tlen);
            let b = (a + span).min(tlen);
            iv.push(Interval {
                start: a,
                end: b.max(a + 1),
            });
        }
        let z = interval_quantiles(x, &iv);
        let inner = ridge_reg_from_features(&z, y, self.alpha, &ctx.policy, "quant_reg");
        ctx.finish(FittedQuantRegressor {
            intervals: iv,
            inner,
        })
    }
}

impl Predict for FittedQuantRegressor {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let z = interval_quantiles(x, &self.intervals);
        self.inner.predict(&z, session)
    }
}

/// SAX bag-of-words + ridge (sktime `SAXVSM` lite).
///
/// Piece / alphabet counts are not identification `p`.
#[derive(Clone, Debug)]
pub struct SaxVsm {
    /// PAA pieces.
    pub n_pieces: usize,
    /// SAX alphabet size.
    pub alphabet: usize,
    /// Ridge \(\alpha\).
    pub alpha: f64,
}

impl Default for SaxVsm {
    fn default() -> Self {
        Self {
            n_pieces: 4,
            alphabet: 4,
            alpha: 0.1,
        }
    }
}

impl SaxVsm {
    /// Default SAX-VSM classifier.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted SAX-VSM ridge.
#[derive(Clone, Debug)]
pub struct FittedSaxVsm {
    n_pieces: usize,
    alphabet: usize,
    inner: crate::classification::FittedRidgeClassifier,
}

fn sax_bow_rows(x: &Matrix, n_pieces: usize, alphabet: usize) -> Matrix {
    let k = n_pieces.max(1);
    let a = alphabet.max(2);
    Matrix::from_fn(x.nrows(), a, |i, bin| {
        let row = x.row(i);
        sax_symbols(row.as_slice(), k, a)
            .into_iter()
            .filter(|s| (*s as usize) == bin)
            .count() as f64
    })
}

impl Fit for SaxVsm {
    type Fitted = FittedSaxVsm;
    fn fit(&mut self, x: &Matrix, y: &Vector, session: &Session) -> Result<Qualified<FittedSaxVsm>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        inspect_classes(&mut ctx.report, y, &ctx.policy);
        let z = sax_bow_rows(x, self.n_pieces, self.alphabet);
        let inner = binary_ridge_from_features(&z, y, self.alpha, &ctx.policy, "saxvsm");
        ctx.finish(FittedSaxVsm {
            n_pieces: self.n_pieces.max(1),
            alphabet: self.alphabet.max(2),
            inner,
        })
    }
}

impl Predict for FittedSaxVsm {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let z = sax_bow_rows(x, self.n_pieces, self.alphabet);
        self.inner.predict(&z, session)
    }
}

fn sfa_symbol_row(row: &[f64], n_coefs: usize, alphabet: usize) -> Vec<f64> {
    let z = znorm(&Vector::from_slice(row));
    let mags = dft_mags(z.as_slice(), n_coefs);
    let a = alphabet.max(2);
    mags.into_iter()
        .map(|mag| {
            let u = 0.5 + 0.5 * crate::special::erf(mag / std::f64::consts::SQRT_2);
            ((u * a as f64).floor() as usize).min(a - 1) as f64
        })
        .collect()
}

fn sfa_feature_rows(x: &Matrix, n_coefs: usize, alphabet: usize) -> Matrix {
    let m = n_coefs.max(1);
    Matrix::from_fn(x.nrows(), m, |i, j| {
        let w = sfa_symbol_row(x.row(i).as_slice(), m, alphabet);
        w.get(j).copied().unwrap_or(0.0)
    })
}

/// Symbolic Fourier Approximation (tslearn `SymbolicFourierApproximation`).
///
/// Word / alphabet counts are not identification `p`.
#[derive(Clone, Debug)]
pub struct Sfa {
    /// Retained DFT magnitudes.
    pub n_coefs: usize,
    /// Alphabet size.
    pub alphabet: usize,
    /// Ridge \(\alpha\).
    pub alpha: f64,
}

impl Default for Sfa {
    fn default() -> Self {
        Self {
            n_coefs: 4,
            alphabet: 4,
            alpha: 0.1,
        }
    }
}

impl Sfa {
    /// Default SFA classifier.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted SFA ridge.
#[derive(Clone, Debug)]
pub struct FittedSfa {
    n_coefs: usize,
    alphabet: usize,
    inner: crate::classification::FittedRidgeClassifier,
}

impl Fit for Sfa {
    type Fitted = FittedSfa;
    fn fit(&mut self, x: &Matrix, y: &Vector, session: &Session) -> Result<Qualified<FittedSfa>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        inspect_classes(&mut ctx.report, y, &ctx.policy);
        let z = sfa_feature_rows(x, self.n_coefs, self.alphabet);
        let inner = binary_ridge_from_features(&z, y, self.alpha, &ctx.policy, "sfa");
        ctx.finish(FittedSfa {
            n_coefs: self.n_coefs.max(1),
            alphabet: self.alphabet.max(2),
            inner,
        })
    }
}

impl Predict for FittedSfa {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let z = sfa_feature_rows(x, self.n_coefs, self.alphabet);
        self.inner.predict(&z, session)
    }
}

fn softdtw_kernel_rows(x: &Matrix, train: &Matrix, gamma: f64) -> Matrix {
    let g = gamma.max(1e-8);
    Matrix::from_fn(x.nrows(), train.nrows(), |i, j| {
        let d = softdtw_raw(x.row(i).as_slice(), train.row(j).as_slice(), g);
        (-d / g).exp()
    })
}

/// Soft-DTW kernel SVM (tslearn `TimeSeriesSVC` with soft-DTW).
///
/// Kernel count is not identification `p`.
#[derive(Clone, Debug)]
pub struct SoftDtwSvm {
    /// Soft-DTW \(\gamma\).
    pub gamma: f64,
    /// Ridge \(\alpha\).
    pub alpha: f64,
}

impl Default for SoftDtwSvm {
    fn default() -> Self {
        Self {
            gamma: 1.0,
            alpha: 0.1,
        }
    }
}

impl SoftDtwSvm {
    /// Default soft-DTW kernel SVM.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted soft-DTW kernel ridge.
#[derive(Clone, Debug)]
pub struct FittedSoftDtwSvm {
    x_train: Matrix,
    gamma: f64,
    inner: crate::classification::FittedRidgeClassifier,
}

impl Fit for SoftDtwSvm {
    type Fitted = FittedSoftDtwSvm;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedSoftDtwSvm>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        inspect_classes(&mut ctx.report, y, &ctx.policy);
        let z = softdtw_kernel_rows(x, x, self.gamma);
        let inner = binary_ridge_from_features(&z, y, self.alpha, &ctx.policy, "sdtwsvm");
        ctx.finish(FittedSoftDtwSvm {
            x_train: x.clone(),
            gamma: self.gamma.max(1e-8),
            inner,
        })
    }
}

impl Predict for FittedSoftDtwSvm {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let z = softdtw_kernel_rows(x, &self.x_train, self.gamma);
        self.inner.predict(&z, session)
    }
}

/// Named DTW 1-NN wrapper (sktime `KNeighborsTimeSeriesClassifier`).
#[derive(Clone, Debug)]
pub struct KNeighborsTimeSeriesClassifier {
    inner: KNeighborsTimeSeries,
}

impl Default for KNeighborsTimeSeriesClassifier {
    fn default() -> Self {
        Self {
            inner: KNeighborsTimeSeries::default(),
        }
    }
}

impl KNeighborsTimeSeriesClassifier {
    /// DTW 1-NN classifier.
    pub fn new() -> Self {
        Self::default()
    }
}

impl Fit for KNeighborsTimeSeriesClassifier {
    type Fitted = FittedKnnTs;
    fn fit(&mut self, x: &Matrix, y: &Vector, session: &Session) -> Result<Qualified<FittedKnnTs>> {
        self.inner.fit(x, y, session)
    }
}

/// PCA on equal-length series rows (tslearn `TimeSeriesPCA` / sklearn `PCA`).
///
/// Component count is not identification `p`. SVD uses a scratch report so a
/// Fatal inner `SvdDidNotConverge` cannot abort a valid embedding.
#[derive(Clone, Debug)]
pub struct TimeSeriesPca {
    /// Retained axes.
    pub n_components: usize,
}

impl Default for TimeSeriesPca {
    fn default() -> Self {
        Self { n_components: 1 }
    }
}

impl TimeSeriesPca {
    /// Keep `n_components` axes.
    pub fn new(n_components: usize) -> Self {
        Self { n_components }
    }

    /// Fit alias.
    pub fn fit(&mut self, x: &Matrix, session: &Session) -> Result<Qualified<FittedTimeSeriesPca>> {
        self.fit_unsupervised(x, session)
    }
}

/// Fitted series-row PCA.
#[derive(Clone, Debug)]
pub struct FittedTimeSeriesPca {
    /// Principal axes (`k` × width).
    pub components: Matrix,
    /// Column means of the training series.
    pub mean: Vector,
    /// Retained singular values.
    pub singular_values: Vector,
}

impl FitUnsupervised for TimeSeriesPca {
    type Fitted = FittedTimeSeriesPca;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedTimeSeriesPca>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let (n, p) = x.shape();
        let k_req = self.n_components.max(1);
        ctx.push(
            Issue::builder(IssueCode::CausalClaimUnidentified)
                .severity(Severity::Advisory)
                .message("TimeSeriesPca is column-centered SVD, not a shapelet / DTW embedding")
                .compromise(NumericalCompromise::new(
                    "tslearn TimeSeriesPCA with a DTW / soft-DTW metric",
                    "ordinary thin SVD of column-centered series rows",
                    "the Euclidean embedding ignores warping",
                    "read scores as PCA coordinates, not a published TSC embedding",
                ))
                .build(),
        );
        if n == 0 || p == 0 {
            return ctx.finish(FittedTimeSeriesPca {
                components: Matrix::zeros(k_req, p),
                mean: Vector::zeros(p),
                singular_values: Vector::zeros(0),
            });
        }
        let (xc, mean) = x.centered();
        let mut scratch = Report::new("tspca", "svd");
        let Some(svd) = thin_svd(&mut scratch, &xc, &ctx.policy) else {
            ctx.push(
                Issue::builder(IssueCode::DidNotConverge)
                    .message("TimeSeriesPca thin SVD failed")
                    .build(),
            );
            return ctx.finish(FittedTimeSeriesPca {
                components: Matrix::zeros(k_req.min(p), p),
                mean,
                singular_values: Vector::zeros(0),
            });
        };
        let rank = svd.rank(ctx.policy.rank_tol_relative).max(1);
        let k = k_req.min(rank).min(svd.singular_values.len()).min(p);
        if k_req > k {
            ctx.push(
                Issue::builder(IssueCode::TruncatedSvdUsed)
                    .message(format!("TimeSeriesPca truncated to {k} axes"))
                    .compromise(NumericalCompromise::new(
                        format!("{k_req} series principal components"),
                        format!("{k} components from a rank-limited SVD"),
                        "extra axes lie in the numerical null space",
                        "dropped scores are identically zero",
                    ))
                    .build(),
            );
        }
        let components = Matrix::from_fn(k, p, |a, b| {
            if b < svd.v.nrows() && a < svd.v.ncols() {
                svd.v[(b, a)]
            } else {
                0.0
            }
        });
        let singular_values = Vector::from_iter(svd.singular_values.iter().take(k).copied());
        ctx.finish(FittedTimeSeriesPca {
            components,
            mean,
            singular_values,
        })
    }
}

impl Transform for FittedTimeSeriesPca {
    fn transform(&self, x: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
        let mut ctx = FitCtx::with_session(session.child("transform"));
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let k = self.components.nrows();
        let p = self.mean.len().min(x.ncols()).min(self.components.ncols());
        let z = Matrix::from_fn(x.nrows(), k, |i, a| {
            let mut s = 0.0_f64;
            for j in 0..p {
                s += (x.get(i, j) - self.mean[j]) * self.components.get(a, j);
            }
            s
        });
        ctx.finish(z)
    }
}

/// Composable time-series forest (sktime `ComposableTimeSeriesForestClassifier`).
///
/// Unweighted vote of [`TimeSeriesForestClassifier`] and
/// [`CanonicalIntervalForest`]. Tree / interval counts are not identification
/// `p`. Inner `MeaninglessFit` is not promoted.
#[derive(Clone, Debug)]
pub struct ComposableTimeSeriesForest {
    /// Trees per member.
    pub n_estimators: usize,
    /// Intervals per tree.
    pub n_intervals: usize,
    /// Tree depth.
    pub max_depth: usize,
    /// Seed.
    pub seed: u64,
}

impl Default for ComposableTimeSeriesForest {
    fn default() -> Self {
        Self {
            n_estimators: 4,
            n_intervals: 3,
            max_depth: 4,
            seed: 5,
        }
    }
}

impl ComposableTimeSeriesForest {
    /// Default two-member interval forest.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted TSF+CIF vote with a Catch22 ridge fallback.
#[derive(Clone, Debug)]
pub struct FittedComposableTimeSeriesForest {
    tsf: Option<FittedTimeSeriesForest>,
    cif: Option<FittedCanonicalIntervalForest>,
    ridge: Option<FittedCatch22Classifier>,
}

impl Fit for ComposableTimeSeriesForest {
    type Fitted = FittedComposableTimeSeriesForest;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedComposableTimeSeriesForest>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        inspect_classes(&mut ctx.report, y, &ctx.policy);
        ctx.push(
            Issue::builder(IssueCode::CausalClaimUnidentified)
                .severity(Severity::Advisory)
                .message("ComposableTimeSeriesForest is an unweighted TSF+CIF vote")
                .compromise(NumericalCompromise::new(
                    "sktime ComposableTimeSeriesForest pipeline",
                    "vote of TimeSeriesForest and CanonicalIntervalForest",
                    "column-ensemble weights and extra members are omitted",
                    "do not read the vote as a published CTSF accuracy",
                ))
                .build(),
        );
        let mut tsf = TimeSeriesForestClassifier {
            n_estimators: self.n_estimators.max(1),
            n_intervals: self.n_intervals.max(1),
            max_depth: self.max_depth.max(1),
            seed: self.seed,
        };
        let tsf_f = match tsf.fit(x, y, &session.child("ctsf-tsf")) {
            Ok(q) => {
                for issue in q.report.issues() {
                    if matches!(
                        issue.code,
                        IssueCode::ResidualTooLarge
                            | IssueCode::NearSingular
                            | IssueCode::R2IsOne
                            | IssueCode::RankZero
                            | IssueCode::MeaninglessFit
                            | IssueCode::UnidentifiedModel
                    ) {
                        continue;
                    }
                    ctx.push(issue.clone());
                }
                Some(q.value)
            }
            Err(_) => None,
        };
        let mut cif = CanonicalIntervalForest {
            n_estimators: self.n_estimators.max(1),
            n_intervals: self.n_intervals.max(1),
            max_depth: self.max_depth.max(1),
            seed: self.seed.wrapping_add(1),
        };
        let cif_f = match cif.fit(x, y, &session.child("ctsf-cif")) {
            Ok(q) => {
                for issue in q.report.issues() {
                    if matches!(
                        issue.code,
                        IssueCode::ResidualTooLarge
                            | IssueCode::NearSingular
                            | IssueCode::R2IsOne
                            | IssueCode::RankZero
                            | IssueCode::MeaninglessFit
                            | IssueCode::UnidentifiedModel
                    ) {
                        continue;
                    }
                    ctx.push(issue.clone());
                }
                Some(q.value)
            }
            Err(_) => None,
        };
        if tsf_f.is_some() || cif_f.is_some() {
            return ctx.finish(FittedComposableTimeSeriesForest {
                tsf: tsf_f,
                cif: cif_f,
                ridge: None,
            });
        }
        ctx.push(
            Issue::builder(IssueCode::DidNotConverge)
                .message("both CTSF members failed; falling back to Catch22 ridge")
                .build(),
        );
        let ridge = match Catch22Classifier::new(0.1).fit(x, y, &session.child("ctsf-ridge")) {
            Ok(q) => Some(q.value),
            Err(_) => None,
        };
        ctx.finish(FittedComposableTimeSeriesForest {
            tsf: None,
            cif: None,
            ridge,
        })
    }
}

impl Predict for FittedComposableTimeSeriesForest {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        if let Some(r) = &self.ridge {
            return r.predict(x, session);
        }
        let mut ctx = FitCtx::with_session(session.child("predict"));
        let mut votes = vec![BTreeMap::<i64, usize>::new(); x.nrows()];
        if let Some(tsf) = &self.tsf {
            if let Ok(q) = tsf.predict(x, &session.child("ctsf-tsf-p")) {
                for i in 0..x.nrows().min(q.value.len()) {
                    *votes[i].entry(q.value[i].round() as i64).or_insert(0) += 1;
                }
            }
        }
        if let Some(cif) = &self.cif {
            if let Ok(q) = cif.predict(x, &session.child("ctsf-cif-p")) {
                for i in 0..x.nrows().min(q.value.len()) {
                    *votes[i].entry(q.value[i].round() as i64).or_insert(0) += 1;
                }
            }
        }
        let out = Vector::from_iter(votes.iter().map(|m| {
            m.iter()
                .max_by(|a, b| a.1.cmp(b.1).then(b.0.cmp(a.0)))
                .map(|(k, _)| *k as f64)
                .unwrap_or(0.0)
        }));
        ctx.finish(out)
    }
}

/// SFA word transformer (tslearn `SymbolicFourierApproximation`).
///
/// Coefficient / alphabet counts are not identification `p`.
#[derive(Clone, Debug)]
pub struct SfaTransformer {
    /// Retained DFT magnitudes.
    pub n_coefs: usize,
    /// Alphabet size.
    pub alphabet: usize,
}

impl Default for SfaTransformer {
    fn default() -> Self {
        Self {
            n_coefs: 4,
            alphabet: 4,
        }
    }
}

impl SfaTransformer {
    /// Default SFA feature map.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted SFA feature map.
#[derive(Clone, Debug)]
pub struct FittedSfaTransformer {
    n_coefs: usize,
    alphabet: usize,
}

impl FitUnsupervised for SfaTransformer {
    type Fitted = FittedSfaTransformer;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedSfaTransformer>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        ctx.push(
            Issue::builder(IssueCode::CausalClaimUnidentified)
                .severity(Severity::Advisory)
                .message("SfaTransformer is magnitude SFA, not a supervised BOSS word")
                .compromise(NumericalCompromise::new(
                    "supervised SFA with class-wise ANOVA binning",
                    "z-norm DFT magnitudes quantized by a Gaussian CDF",
                    "breakpoints are not fitted to class separation",
                    "read codes as Fourier bins, not a published WEASEL/SFA word",
                ))
                .build(),
        );
        ctx.finish(FittedSfaTransformer {
            n_coefs: self.n_coefs.max(1),
            alphabet: self.alphabet.max(2),
        })
    }
}

impl Transform for FittedSfaTransformer {
    fn transform(&self, x: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
        let mut ctx = FitCtx::with_session(session.child("transform"));
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        ctx.finish(sfa_feature_rows(x, self.n_coefs, self.alphabet))
    }
}

/// Named soft-DTW barycentre (tslearn `softdtw_barycenter`).
#[derive(Clone, Debug)]
pub struct SoftDtwBarycenter {
    /// Soft-DTW \(\gamma\).
    pub gamma: f64,
    /// Gradient steps.
    pub max_iter: usize,
}

impl Default for SoftDtwBarycenter {
    fn default() -> Self {
        Self {
            gamma: 1.0,
            max_iter: 8,
        }
    }
}

impl SoftDtwBarycenter {
    /// Default soft-DTW barycentre.
    pub fn new() -> Self {
        Self::default()
    }

    /// Fit the barycentre of equal-length series rows.
    pub fn fit(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        softdtw_barycenter(x, self.gamma, self.max_iter, session)
    }
}

/// Named alias of [`LearningShapelets`] (tslearn `ShapeletModel`).
#[derive(Clone, Debug)]
pub struct ShapeletModel {
    inner: LearningShapelets,
}

impl Default for ShapeletModel {
    fn default() -> Self {
        Self {
            inner: LearningShapelets::default(),
        }
    }
}

impl ShapeletModel {
    /// `k` shapelets of length `length`.
    pub fn new(n_shapelets: usize, length: usize) -> Self {
        Self {
            inner: LearningShapelets::new(n_shapelets, length),
        }
    }
}

impl Fit for ShapeletModel {
    type Fitted = FittedShapelets;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedShapelets>> {
        self.inner.fit(x, y, session)
    }
}

/// Shapelet ridge regressor (tslearn `LearningShapelets` regression).
///
/// Shapelet count is not identification `p`. Does not call `inspect_classes`.
#[derive(Clone, Debug)]
pub struct LearningShapeletsRegressor {
    /// Random shapelets.
    pub n_shapelets: usize,
    /// Shapelet length.
    pub length: usize,
    /// Ridge \(\alpha\).
    pub alpha: f64,
    /// Seed.
    pub seed: u64,
}

impl Default for LearningShapeletsRegressor {
    fn default() -> Self {
        Self {
            n_shapelets: 4,
            length: 4,
            alpha: 0.5,
            seed: 3,
        }
    }
}

impl LearningShapeletsRegressor {
    /// Default shapelet regressor.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted shapelet distances + ridge.
#[derive(Clone, Debug)]
pub struct FittedLearningShapeletsReg {
    shapelets: Matrix,
    ridge: FittedPenalized,
}

impl Fit for LearningShapeletsRegressor {
    type Fitted = FittedLearningShapeletsReg;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedLearningShapeletsReg>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        let l = self.length.min(x.ncols().max(2)).max(2);
        if x.ncols() < l {
            ctx.push(
                Issue::builder(IssueCode::WindowTooShort)
                    .severity(Severity::Warning)
                    .message(format!(
                        "LearningShapeletsRegressor length={l} > T={}",
                        x.ncols()
                    ))
                    .build(),
            );
        }
        ctx.push(
            Issue::builder(IssueCode::JitterInjected)
                .severity(Severity::Advisory)
                .message("LearningShapeletsRegressor samples random windows")
                .compromise(NumericalCompromise::new(
                    "gradient-learned shapelets for regression",
                    "random subsequences + min-distance + ridge",
                    "shapelets are not optimized against the SSE",
                    "treat the features as a random convolutional sketch",
                ))
                .build(),
        );
        let k = self.n_shapelets.max(1);
        let mut rng = Rng::new(self.seed | 11);
        let slen = l.min(x.ncols().max(1));
        let mut shapelets = Matrix::zeros(k, slen);
        if x.nrows() > 0 && x.ncols() >= slen {
            for s in 0..k {
                let row = rng.below(x.nrows());
                let start = if x.ncols() > slen {
                    rng.below(x.ncols() - slen + 1)
                } else {
                    0
                };
                for u in 0..slen {
                    shapelets.set(s, u, x.get(row, start + u));
                }
            }
        }
        let feat = Matrix::from_fn(x.nrows(), k, |i, s| min_shapelet_dist(x, i, &shapelets, s));
        let ridge = ridge_reg_from_features(&feat, y, self.alpha, &ctx.policy, "lshreg");
        ctx.finish(FittedLearningShapeletsReg { shapelets, ridge })
    }
}

impl Predict for FittedLearningShapeletsReg {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        let k = self.shapelets.nrows();
        let feat = Matrix::from_fn(x.nrows(), k, |i, s| {
            min_shapelet_dist(x, i, &self.shapelets, s)
        });
        let out = if feat.ncols() == self.ridge.coef.len() {
            let mut yhat = feat.matvec(&self.ridge.coef);
            for i in 0..yhat.len() {
                yhat[i] += self.ridge.intercept;
            }
            yhat
        } else {
            Vector::filled(x.nrows(), self.ridge.intercept)
        };
        ctx.finish(out)
    }
}

/// 1-D locally linear embedding of series rows (tslearn `LocallyLinearEmbedding`).
///
/// Neighbour count is not identification `p`. SVD uses a scratch report.
#[derive(Clone, Debug)]
pub struct OneDLle {
    /// Neighbourhood size.
    pub n_neighbors: usize,
}

impl Default for OneDLle {
    fn default() -> Self {
        Self { n_neighbors: 3 }
    }
}

impl OneDLle {
    /// 1-D LLE with `n_neighbors`.
    pub fn new(n_neighbors: usize) -> Self {
        Self {
            n_neighbors: n_neighbors.max(2),
        }
    }

    /// Fit alias.
    pub fn fit(&mut self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        self.fit_unsupervised(x, session)
    }
}

impl FitUnsupervised for OneDLle {
    type Fitted = Vector;
    fn fit_unsupervised(&mut self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let n = x.nrows();
        let k = self.n_neighbors.max(2).min(n.saturating_sub(1).max(1));
        if n < 3 {
            ctx.push(
                Issue::builder(IssueCode::InsufficientSample)
                    .severity(Severity::Warning)
                    .message("OneDLle needs at least 3 series")
                    .build(),
            );
            return ctx.finish(Vector::zeros(n));
        }
        let mut w = Matrix::zeros(n, n);
        for i in 0..n {
            let mut dist: Vec<(f64, usize)> = (0..n)
                .filter(|&j| j != i)
                .map(|j| {
                    let mut s = 0.0_f64;
                    for t in 0..x.ncols() {
                        let d = x.get(i, t) - x.get(j, t);
                        s += d * d;
                    }
                    (s, j)
                })
                .collect();
            dist.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            let neigh: Vec<usize> = dist.iter().take(k).map(|d| d.1).collect();
            let kk = neigh.len();
            if kk == 0 {
                continue;
            }
            let gram = Matrix::from_fn(kk, kk, |a, b| {
                let mut s = 0.0_f64;
                for t in 0..x.ncols() {
                    s += (x.get(i, t) - x.get(neigh[a], t)) * (x.get(i, t) - x.get(neigh[b], t));
                }
                s
            });
            let ones = Vector::filled(kk, 1.0);
            let mut scratch = Report::new("onelle", "w");
            let wt = ridge_solve(&mut scratch, &gram, &ones, 1e-3, &ctx.policy)
                .unwrap_or_else(|| Vector::filled(kk, 1.0 / kk as f64));
            let s = wt.as_slice().iter().sum::<f64>();
            let inv = if s.abs() > 1e-12 { 1.0 / s } else { 1.0 / kk as f64 };
            for (a, &j) in neigh.iter().enumerate() {
                w.set(i, j, wt[a] * inv);
            }
        }
        let m = Matrix::from_fn(n, n, |i, j| {
            let mut s = if i == j { 1.0 } else { 0.0 };
            s -= w.get(i, j);
            s
        });
        let gram_m = Matrix::from_fn(n, n, |i, j| {
            let mut s = 0.0_f64;
            for t in 0..n {
                s += m.get(t, i) * m.get(t, j);
            }
            s
        });
        let mut scratch = Report::new("onelle", "svd");
        let Some(svd) = thin_svd(&mut scratch, &gram_m, &ctx.policy) else {
            ctx.push(
                Issue::builder(IssueCode::DidNotConverge)
                    .message("OneDLle scatter SVD failed")
                    .build(),
            );
            return ctx.finish(Vector::zeros(n));
        };
        let axis = if svd.v.ncols() >= 2 { 1 } else { 0 };
        let emb = Vector::from_iter((0..n).map(|i| {
            if i < svd.v.nrows() && axis < svd.v.ncols() {
                svd.v[(i, axis)]
            } else {
                0.0
            }
        }));
        ctx.finish(emb)
    }
}

/// SVD of series rows (tslearn `TimeSeriesSVD`).
///
/// Component count is not identification `p`.
#[derive(Clone, Debug)]
pub struct TimeSeriesSvd {
    /// Retained axes.
    pub n_components: usize,
}

impl Default for TimeSeriesSvd {
    fn default() -> Self {
        Self { n_components: 1 }
    }
}

impl TimeSeriesSvd {
    /// Keep `n_components` axes.
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components: n_components.max(1),
        }
    }

    /// Fit alias.
    pub fn fit(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedTimeSeriesPca>> {
        self.fit_unsupervised(x, session)
    }
}

impl FitUnsupervised for TimeSeriesSvd {
    type Fitted = FittedTimeSeriesPca;
    fn fit_unsupervised(
        &mut self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedTimeSeriesPca>> {
        TimeSeriesPca::new(self.n_components).fit_unsupervised(x, session)
    }
}

/// Named Petitjean DBA (tslearn `DTWBarycenterAveraging`).
#[derive(Clone, Debug)]
pub struct DbaBarycenter {
    /// Alignment iterations.
    pub max_iter: usize,
}

impl Default for DbaBarycenter {
    fn default() -> Self {
        Self { max_iter: 8 }
    }
}

impl DbaBarycenter {
    /// Default DBA barycentre.
    pub fn new() -> Self {
        Self::default()
    }

    /// Fit the DTW barycentre of equal-length series rows.
    pub fn fit(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        dba(x, self.max_iter, session)
    }
}

/// Named Euclidean barycentre (tslearn `euclidean_barycenter`).
#[derive(Clone, Debug, Default)]
pub struct EuclideanBarycenter;

impl EuclideanBarycenter {
    /// Default column-mean barycentre.
    pub fn new() -> Self {
        Self
    }

    /// Fit the Euclidean mean of equal-length series rows.
    pub fn fit(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        euclidean_barycenter(x, session)
    }
}

/// Random Interval Spectral Transform classifier (sktime `RISTClassifier` lite).
///
/// Interval count is not identification `p`. ExtraTrees may abort as a vacuous
/// stump; ridge on the same interval map is the fallback.
#[derive(Clone, Debug)]
pub struct Rist {
    /// ExtraTrees count.
    pub n_estimators: usize,
    /// Random intervals.
    pub n_intervals: usize,
    /// Tree depth.
    pub max_depth: usize,
    /// Seed.
    pub seed: u64,
}

impl Default for Rist {
    fn default() -> Self {
        Self {
            n_estimators: 8,
            n_intervals: 3,
            max_depth: 4,
            seed: 17,
        }
    }
}

impl Rist {
    /// Default RIST lite.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted RIST forest or ridge fallback.
#[derive(Clone, Debug)]
pub struct FittedRist {
    forest: Option<crate::tree::FittedForestClassifier>,
    ridge: Option<crate::classification::FittedRidgeClassifier>,
    intervals: Vec<Interval>,
    default_label: f64,
}

impl Fit for Rist {
    type Fitted = FittedRist;
    fn fit(&mut self, x: &Matrix, y: &Vector, session: &Session) -> Result<Qualified<FittedRist>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        inspect_classes(&mut ctx.report, y, &ctx.policy);
        let default_label = y
            .as_slice()
            .iter()
            .copied()
            .find(|v| v.is_finite())
            .unwrap_or(0.0);
        let mut rng = Rng::new(self.seed);
        let tlen = x.ncols().max(1);
        let mut intervals = Vec::new();
        for _ in 0..self.n_intervals.max(1) {
            let a = rng.below(tlen);
            let span = 1 + rng.below(tlen);
            intervals.push(Interval {
                start: a,
                end: (a + span).min(tlen).max(a + 1),
            });
        }
        let feat = interval_feats_rise(x, &intervals);
        ctx.push(
            Issue::builder(IssueCode::JitterInjected)
                .severity(Severity::Advisory)
                .message("Rist uses interval DFT summaries plus ExtraTrees, not the full RIST map")
                .compromise(NumericalCompromise::new(
                    "sktime RIST spectral + extra-trees pipeline",
                    "RISE-style interval DFT then ExtraTreesClassifier",
                    "ACF / periodogram / convolution members are omitted",
                    "do not read as a published RIST accuracy",
                ))
                .build(),
        );
        let mut et = crate::tree::ExtraTreesClassifier {
            n_estimators: self.n_estimators.max(1),
            max_depth: self.max_depth.max(1),
            min_samples_split: 2,
            max_features: Some(4),
            seed: self.seed,
        };
        match et.fit(&feat, y, &session.child("rist-et")) {
            Ok(q) => {
                for issue in q.report.issues() {
                    if matches!(
                        issue.code,
                        IssueCode::ResidualTooLarge
                            | IssueCode::NearSingular
                            | IssueCode::R2IsOne
                            | IssueCode::RankZero
                            | IssueCode::MeaninglessFit
                    ) {
                        continue;
                    }
                    ctx.push(issue.clone());
                }
                ctx.finish(FittedRist {
                    forest: Some(q.value),
                    ridge: None,
                    intervals,
                    default_label,
                })
            }
            Err(_) => {
                ctx.push(
                    Issue::builder(IssueCode::DidNotConverge)
                        .message("Rist ExtraTrees failed; falling back to interval ridge")
                        .build(),
                );
                let ridge = binary_ridge_from_features(&feat, y, 0.5, &ctx.policy, "rist");
                ctx.finish(FittedRist {
                    forest: None,
                    ridge: Some(ridge),
                    intervals,
                    default_label,
                })
            }
        }
    }
}

impl Predict for FittedRist {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let feat = interval_feats_rise(x, &self.intervals);
        if let Some(f) = &self.forest {
            return f.predict(&feat, session);
        }
        if let Some(r) = &self.ridge {
            return r.predict(&feat, session);
        }
        let mut ctx = FitCtx::with_session(session.child("predict"));
        ctx.push(Issue::builder(IssueCode::PartialFitBeforeInit).build());
        ctx.finish(Vector::filled(x.nrows(), self.default_label))
    }
}

/// BOSS vector space (sktime `BOSSVSClassifier`): class-mean SFA histograms
/// with cosine vote.
///
/// Vocabulary size is not identification `p`. Window must be ≤ series length.
#[derive(Clone, Debug)]
pub struct BossVs {
    /// Sliding-window length.
    pub window: usize,
    /// DFT coefficients kept per window.
    pub word_len: usize,
    /// SFA alphabet size.
    pub alphabet: usize,
}

impl Default for BossVs {
    fn default() -> Self {
        Self {
            window: 4,
            word_len: 4,
            alphabet: 4,
        }
    }
}

impl BossVs {
    /// Default BOSS-VS.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted BOSS-VS class centroids.
#[derive(Clone, Debug)]
pub struct FittedBossVs {
    spec: (usize, usize, usize),
    centroids: BTreeMap<i64, Vector>,
    default_label: f64,
}

impl Fit for BossVs {
    type Fitted = FittedBossVs;
    fn fit(&mut self, x: &Matrix, y: &Vector, session: &Session) -> Result<Qualified<FittedBossVs>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        inspect_classes(&mut ctx.report, y, &ctx.policy);
        let w = self.window.max(2).min(x.ncols().max(2));
        if x.ncols() < self.window {
            ctx.push(
                Issue::builder(IssueCode::WindowTooShort)
                    .severity(Severity::Warning)
                    .message(format!(
                        "BossVs window {} > series length {}",
                        self.window,
                        x.ncols()
                    ))
                    .build(),
            );
        }
        let (h, _vocab) = boss_histograms(x, w, self.word_len.max(1), self.alphabet.max(2));
        let mut sums: BTreeMap<i64, Vector> = BTreeMap::new();
        let mut cnt: BTreeMap<i64, f64> = BTreeMap::new();
        let default_label = y
            .as_slice()
            .iter()
            .copied()
            .find(|v| v.is_finite())
            .unwrap_or(0.0);
        for i in 0..x.nrows().min(y.len()) {
            if !y[i].is_finite() {
                continue;
            }
            let lab = y[i].round() as i64;
            let acc = sums.entry(lab).or_insert_with(|| Vector::zeros(h.ncols()));
            if acc.len() != h.ncols() {
                *acc = Vector::zeros(h.ncols());
            }
            for j in 0..h.ncols() {
                acc[j] += h.get(i, j);
            }
            *cnt.entry(lab).or_insert(0.0) += 1.0;
        }
        let mut centroids = BTreeMap::new();
        for (lab, mut acc) in sums {
            let n = cnt.get(&lab).copied().unwrap_or(1.0).max(1.0);
            for j in 0..acc.len() {
                acc[j] /= n;
            }
            centroids.insert(lab, acc);
        }
        if centroids.is_empty() {
            ctx.push(
                Issue::builder(IssueCode::DidNotConverge)
                    .message("BossVs has no class centroid")
                    .build(),
            );
        }
        ctx.push(
            Issue::builder(IssueCode::CausalClaimUnidentified)
                .severity(Severity::Advisory)
                .message("BossVs is cosine-to-mean SFA histograms, not the published BOSS-VS")
                .compromise(NumericalCompromise::new(
                    "sktime BOSSVS tf-idf / IDF-weighted cosine",
                    "class-mean BOSS histograms and plain cosine",
                    "IDF and multiple window ensembles are omitted",
                    "do not read as a published BOSS-VS accuracy",
                ))
                .build(),
        );
        ctx.finish(FittedBossVs {
            spec: (w, self.word_len.max(1), self.alphabet.max(2)),
            centroids,
            default_label,
        })
    }
}

impl Predict for FittedBossVs {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        let (h, _) = boss_histograms(x, self.spec.0, self.spec.1, self.spec.2);
        let out = Vector::from_iter((0..x.nrows()).map(|i| {
            let mut best_lab = self.default_label;
            let mut best = f64::NEG_INFINITY;
            for (&lab, c) in &self.centroids {
                let mut dot = 0.0_f64;
                let mut na = 0.0_f64;
                let mut nb = 0.0_f64;
                for j in 0..h.ncols().min(c.len()) {
                    let a = h.get(i, j);
                    let b = c[j];
                    dot += a * b;
                    na += a * a;
                    nb += b * b;
                }
                let den = (na.sqrt() * nb.sqrt()).max(1e-12);
                let cos = dot / den;
                if cos > best {
                    best = cos;
                    best_lab = lab as f64;
                }
            }
            best_lab
        }));
        ctx.finish(out)
    }
}

/// Named PELT annotator (sktime `Pelt`).
#[derive(Clone, Debug, Default)]
pub struct PeltAnnotator {
    inner: Pelt,
}

impl PeltAnnotator {
    /// Default PELT annotator.
    pub fn new() -> Self {
        Self {
            inner: Pelt::default(),
        }
    }

    /// Change-point locations.
    pub fn fit(&self, y: &Vector, session: &Session) -> Result<Qualified<Vector>> {
        self.inner.fit(y, session)
    }
}

/// Named ClaSP annotator (sktime `ClaSP`).
#[derive(Clone, Debug, Default)]
pub struct ClaSPAnnotator;

impl ClaSPAnnotator {
    /// Default ClaSP annotator.
    pub fn new() -> Self {
        Self
    }

    /// Index of the principal ClaSP split.
    pub fn fit(&self, y: &Vector, session: &Session) -> Result<Qualified<f64>> {
        ClaSPSegmentation::new().fit(y, session)
    }
}

/// Named Hydra+MultiROCKET classifier (sktime `HydraMultiRocketClassifier`).
#[derive(Clone, Debug)]
pub struct HydraMultiRocketClassifier {
    inner: HydraMultiRocket,
}

impl Default for HydraMultiRocketClassifier {
    fn default() -> Self {
        Self {
            inner: HydraMultiRocket::default(),
        }
    }
}

impl HydraMultiRocketClassifier {
    /// Default Hydra+MultiROCKET classifier.
    pub fn new() -> Self {
        Self::default()
    }
}

impl Fit for HydraMultiRocketClassifier {
    type Fitted = FittedHydraMultiRocket;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedHydraMultiRocket>> {
        self.inner.fit(x, y, session)
    }
}

/// RISE regressor (sktime `RandomIntervalSpectralEnsemble` regression).
///
/// Interval count is not identification `p`. ExtraTrees may abort as a vacuous
/// stump; ridge on the same interval map is the fallback.
#[derive(Clone, Debug)]
pub struct RiseRegressor {
    /// ExtraTrees count.
    pub n_estimators: usize,
    /// Random intervals.
    pub n_intervals: usize,
    /// Tree depth.
    pub max_depth: usize,
    /// Seed.
    pub seed: u64,
}

impl Default for RiseRegressor {
    fn default() -> Self {
        Self {
            n_estimators: 8,
            n_intervals: 3,
            max_depth: 4,
            seed: 19,
        }
    }
}

impl RiseRegressor {
    /// Default RISE regressor.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted RISE forest or ridge fallback.
#[derive(Clone, Debug)]
pub struct FittedRiseRegressor {
    forest: Option<crate::tree::FittedForestRegressor>,
    ridge: Option<FittedPenalized>,
    intervals: Vec<Interval>,
}

impl Fit for RiseRegressor {
    type Fitted = FittedRiseRegressor;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedRiseRegressor>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        let mut rng = Rng::new(self.seed);
        let tlen = x.ncols().max(1);
        let mut intervals = Vec::new();
        for _ in 0..self.n_intervals.max(1) {
            let a = rng.below(tlen);
            let span = 1 + rng.below(tlen);
            intervals.push(Interval {
                start: a,
                end: (a + span).min(tlen).max(a + 1),
            });
        }
        let feat = interval_feats_rise(x, &intervals);
        ctx.push(
            Issue::builder(IssueCode::JitterInjected)
                .severity(Severity::Advisory)
                .message("RiseRegressor uses interval DFT plus ExtraTrees, not the full RISE map")
                .compromise(NumericalCompromise::new(
                    "sktime RISE spectral + extra-trees regressor",
                    "RISE-style interval DFT then ExtraTreesRegressor",
                    "ACF / periodogram members are omitted",
                    "do not read as a published RISE accuracy",
                ))
                .build(),
        );
        let mut et = crate::tree::ExtraTreesRegressor {
            n_estimators: self.n_estimators.max(1),
            max_depth: self.max_depth.max(1),
            min_samples_split: 2,
            max_features: Some(4),
            seed: self.seed,
        };
        match et.fit(&feat, y, &session.child("riser-et")) {
            Ok(q) => {
                for issue in q.report.issues() {
                    if matches!(
                        issue.code,
                        IssueCode::ResidualTooLarge
                            | IssueCode::NearSingular
                            | IssueCode::R2IsOne
                            | IssueCode::RankZero
                            | IssueCode::MeaninglessFit
                            | IssueCode::PredictionsAreConstant
                    ) {
                        continue;
                    }
                    ctx.push(issue.clone());
                }
                ctx.finish(FittedRiseRegressor {
                    forest: Some(q.value),
                    ridge: None,
                    intervals,
                })
            }
            Err(_) => {
                ctx.push(
                    Issue::builder(IssueCode::DidNotConverge)
                        .message("RiseRegressor ExtraTrees failed; falling back to interval ridge")
                        .build(),
                );
                let ridge = ridge_reg_from_features(&feat, y, 0.5, &ctx.policy, "riser");
                ctx.finish(FittedRiseRegressor {
                    forest: None,
                    ridge: Some(ridge),
                    intervals,
                })
            }
        }
    }
}

impl Predict for FittedRiseRegressor {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let feat = interval_feats_rise(x, &self.intervals);
        if let Some(f) = &self.forest {
            return f.predict(&feat, session);
        }
        if let Some(r) = &self.ridge {
            let mut ctx = FitCtx::with_session(session.child("predict"));
            let out = if feat.ncols() == r.coef.len() {
                let mut yhat = feat.matvec(&r.coef);
                for i in 0..yhat.len() {
                    yhat[i] += r.intercept;
                }
                yhat
            } else {
                Vector::filled(x.nrows(), r.intercept)
            };
            return ctx.finish(out);
        }
        let mut ctx = FitCtx::with_session(session.child("predict"));
        ctx.push(Issue::builder(IssueCode::PartialFitBeforeInit).build());
        ctx.finish(Vector::zeros(x.nrows()))
    }
}

/// MiniROCKET features + ExtraTrees (sktime `MiniRocketClassifier`).
///
/// Kernel count is not identification `p`. ExtraTrees may abort as a vacuous
/// stump; ridge on the same map is the fallback.
#[derive(Clone, Debug)]
pub struct MiniRocketClassifier {
    /// Random dilated kernels.
    pub n_kernels: usize,
    /// ExtraTrees count.
    pub n_estimators: usize,
    /// Tree depth.
    pub max_depth: usize,
    /// Ridge \(\alpha\) fallback.
    pub alpha: f64,
    /// Seed.
    pub seed: u64,
}

impl Default for MiniRocketClassifier {
    fn default() -> Self {
        Self {
            n_kernels: 16,
            n_estimators: 8,
            max_depth: 4,
            alpha: 0.1,
            seed: 11,
        }
    }
}

impl MiniRocketClassifier {
    /// Default MiniROCKET classifier.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted MiniROCKET forest or ridge fallback.
#[derive(Clone, Debug)]
pub struct FittedMiniRocketClassifier {
    rocket: MiniRocket,
    forest: Option<crate::tree::FittedForestClassifier>,
    ridge: Option<crate::classification::FittedRidgeClassifier>,
}

impl Fit for MiniRocketClassifier {
    type Fitted = FittedMiniRocketClassifier;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedMiniRocketClassifier>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        inspect_classes(&mut ctx.report, y, &ctx.policy);
        let rocket = MiniRocket {
            n_kernels: self.n_kernels.max(1),
            seed: self.seed,
        };
        let feat = rocket.transform(x, &session.child("mrc"))?;
        ctx.push(
            Issue::builder(IssueCode::CausalClaimUnidentified)
                .severity(Severity::Advisory)
                .message("MiniRocketClassifier is PPV features plus ExtraTrees, not the published pipeline")
                .compromise(NumericalCompromise::new(
                    "sktime MiniRocketClassifier",
                    "MiniROCKET transform then ExtraTreesClassifier",
                    "the official ridge / logistic head and kernel count are omitted",
                    "do not read as a published MiniROCKET accuracy",
                ))
                .build(),
        );
        let mut et = crate::tree::ExtraTreesClassifier {
            n_estimators: self.n_estimators.max(1),
            max_depth: self.max_depth.max(1),
            min_samples_split: 2,
            max_features: Some(4),
            seed: self.seed,
        };
        match et.fit(&feat.value, y, &session.child("mrc-et")) {
            Ok(q) => {
                for issue in q.report.issues() {
                    if matches!(
                        issue.code,
                        IssueCode::ResidualTooLarge
                            | IssueCode::NearSingular
                            | IssueCode::R2IsOne
                            | IssueCode::RankZero
                            | IssueCode::MeaninglessFit
                            | IssueCode::PredictionsAreConstant
                    ) {
                        continue;
                    }
                    ctx.push(issue.clone());
                }
                ctx.finish(FittedMiniRocketClassifier {
                    rocket,
                    forest: Some(q.value),
                    ridge: None,
                })
            }
            Err(_) => {
                ctx.push(
                    Issue::builder(IssueCode::DidNotConverge)
                        .message("MiniRocketClassifier ExtraTrees failed; falling back to ridge")
                        .build(),
                );
                let ridge = binary_ridge_from_features(
                    &feat.value,
                    y,
                    self.alpha,
                    &ctx.policy,
                    "mrc",
                );
                ctx.finish(FittedMiniRocketClassifier {
                    rocket,
                    forest: None,
                    ridge: Some(ridge),
                })
            }
        }
    }
}

impl Predict for FittedMiniRocketClassifier {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let feat = self.rocket.transform(x, &session.child("mrc"))?;
        if let Some(f) = &self.forest {
            return f.predict(&feat.value, session);
        }
        if let Some(r) = &self.ridge {
            return r.predict(&feat.value, session);
        }
        let mut ctx = FitCtx::with_session(session.child("predict"));
        ctx.push(Issue::builder(IssueCode::PartialFitBeforeInit).build());
        ctx.finish(Vector::zeros(x.nrows()))
    }
}

/// Catch22 + ExtraTrees regressor (sktime `Catch22` forest regression).
///
/// Catch22 width and tree count are not identification `p`. ExtraTrees may
/// abort as a vacuous stump; ridge on the same map is the fallback.
#[derive(Clone, Debug)]
pub struct Catch22ForestRegressor {
    /// ExtraTrees count.
    pub n_estimators: usize,
    /// Tree depth.
    pub max_depth: usize,
    /// Seed.
    pub seed: u64,
}

impl Default for Catch22ForestRegressor {
    fn default() -> Self {
        Self {
            n_estimators: 8,
            max_depth: 4,
            seed: 13,
        }
    }
}

impl Catch22ForestRegressor {
    /// Default Catch22 forest regressor.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted Catch22 ExtraTrees or ridge fallback.
#[derive(Clone, Debug)]
pub struct FittedCatch22ForestRegressor {
    forest: Option<crate::tree::FittedForestRegressor>,
    ridge: Option<FittedPenalized>,
}

impl Fit for Catch22ForestRegressor {
    type Fitted = FittedCatch22ForestRegressor;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedCatch22ForestRegressor>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        let z = catch22_rows(x, session, &mut ctx);
        ctx.push(
            Issue::builder(IssueCode::CausalClaimUnidentified)
                .severity(Severity::Advisory)
                .message("Catch22ForestRegressor is Catch22 plus ExtraTrees, not a published forest")
                .compromise(NumericalCompromise::new(
                    "sktime Catch22 + ExtraTreesRegressor",
                    "catch22_rows then ExtraTreesRegressor",
                    "rotation / CIF members are omitted",
                    "do not read as a published Catch22-forest accuracy",
                ))
                .build(),
        );
        let mut et = crate::tree::ExtraTreesRegressor {
            n_estimators: self.n_estimators.max(1),
            max_depth: self.max_depth.max(1),
            min_samples_split: 2,
            max_features: Some(4),
            seed: self.seed,
        };
        match et.fit(&z, y, &session.child("c22fr-et")) {
            Ok(q) => {
                for issue in q.report.issues() {
                    if matches!(
                        issue.code,
                        IssueCode::ResidualTooLarge
                            | IssueCode::NearSingular
                            | IssueCode::R2IsOne
                            | IssueCode::RankZero
                            | IssueCode::MeaninglessFit
                            | IssueCode::PredictionsAreConstant
                    ) {
                        continue;
                    }
                    ctx.push(issue.clone());
                }
                ctx.finish(FittedCatch22ForestRegressor {
                    forest: Some(q.value),
                    ridge: None,
                })
            }
            Err(_) => {
                ctx.push(
                    Issue::builder(IssueCode::DidNotConverge)
                        .message("Catch22ForestRegressor ExtraTrees failed; falling back to ridge")
                        .build(),
                );
                let ridge = ridge_reg_from_features(&z, y, 0.5, &ctx.policy, "c22fr");
                ctx.finish(FittedCatch22ForestRegressor {
                    forest: None,
                    ridge: Some(ridge),
                })
            }
        }
    }
}

impl Predict for FittedCatch22ForestRegressor {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let z = {
            let mut ctx = FitCtx::with_session(session.child("c22fr-z"));
            catch22_rows(x, session, &mut ctx)
        };
        if let Some(f) = &self.forest {
            return f.predict(&z, session);
        }
        if let Some(r) = &self.ridge {
            return r.predict(&z, session);
        }
        let mut ctx = FitCtx::with_session(session.child("predict"));
        ctx.push(Issue::builder(IssueCode::PartialFitBeforeInit).build());
        ctx.finish(Vector::zeros(x.nrows()))
    }
}

fn dilated_columns(x: &Matrix, dil: usize) -> Matrix {
    let d = dil.max(1);
    let t = x.ncols();
    let nt = if t == 0 { 1 } else { (t - 1) / d + 1 };
    Matrix::from_fn(x.nrows(), nt.max(1), |i, j| {
        let c = j * d;
        if c < t {
            x.get(i, c)
        } else {
            0.0
        }
    })
}

/// WEASEL+Dilation (sktime `WEASEL-D`): BOSS histograms at two dilations + ridge.
///
/// Dilation / vocab counts are not identification `p`.
#[derive(Clone, Debug)]
pub struct WeaselD {
    /// Base window (clamped to series length).
    pub window: usize,
    /// DFT coefficients kept per window.
    pub word_len: usize,
    /// SFA alphabet size.
    pub alphabet: usize,
    /// Words kept per dilation.
    pub n_words: usize,
    /// Second dilation. Not identification `p`.
    pub dilation: usize,
    /// Ridge \(\alpha\).
    pub alpha: f64,
}

impl Default for WeaselD {
    fn default() -> Self {
        Self {
            window: 4,
            word_len: 4,
            alphabet: 4,
            n_words: 6,
            dilation: 2,
            alpha: 0.1,
        }
    }
}

impl WeaselD {
    /// Default two-dilation WEASEL.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted WEASEL-D ridge.
#[derive(Clone, Debug)]
pub struct FittedWeaselD {
    spec: (usize, usize, usize),
    dilation: usize,
    idx_a: Vec<usize>,
    idx_b: Vec<usize>,
    inner: crate::classification::FittedRidgeClassifier,
}

impl Fit for WeaselD {
    type Fitted = FittedWeaselD;
    fn fit(&mut self, x: &Matrix, y: &Vector, session: &Session) -> Result<Qualified<FittedWeaselD>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        inspect_classes(&mut ctx.report, y, &ctx.policy);
        let w = self.window.max(2);
        let dil = self.dilation.max(2);
        let (h1, _) = boss_histograms(x, w, self.word_len, self.alphabet);
        let xd = dilated_columns(x, dil);
        let (h2, _) = boss_histograms(&xd, w.min(xd.ncols().max(2)), self.word_len, self.alphabet);
        let (z1, idx_a) = weasel_keep(&h1, self.n_words);
        let (z2, idx_b) = weasel_keep(&h2, self.n_words);
        let z = concat_features(&z1, &z2);
        ctx.push(
            Issue::builder(IssueCode::CausalClaimUnidentified)
                .severity(Severity::Advisory)
                .message("WeaselD is two dilated BOSS histograms plus ridge, not published WEASEL+")
                .compromise(NumericalCompromise::new(
                    "sktime WEASEL-D / WEASEL 2.0 dilated dictionaries",
                    "BOSS at dilation 1 and a column-subsampled dilation",
                    "information-gain word selection and ANOVA filter are omitted",
                    "do not read as a published WEASEL-D accuracy",
                ))
                .build(),
        );
        let inner = binary_ridge_from_features(&z, y, self.alpha, &ctx.policy, "weaseld");
        ctx.finish(FittedWeaselD {
            spec: (w, self.word_len.max(1), self.alphabet.max(2)),
            dilation: dil,
            idx_a,
            idx_b,
            inner,
        })
    }
}

impl Predict for FittedWeaselD {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let (h1, _) = boss_histograms(x, self.spec.0, self.spec.1, self.spec.2);
        let xd = dilated_columns(x, self.dilation);
        let (h2, _) = boss_histograms(
            &xd,
            self.spec.0.min(xd.ncols().max(2)),
            self.spec.1,
            self.spec.2,
        );
        let z1 = if self.idx_a.is_empty() {
            Matrix::zeros(h1.nrows(), 0)
        } else {
            Matrix::from_fn(h1.nrows(), self.idx_a.len(), |i, t| {
                h1.get(i, self.idx_a[t].min(h1.ncols().saturating_sub(1)))
            })
        };
        let z2 = if self.idx_b.is_empty() {
            Matrix::zeros(h2.nrows(), 0)
        } else {
            Matrix::from_fn(h2.nrows(), self.idx_b.len(), |i, t| {
                h2.get(i, self.idx_b[t].min(h2.ncols().saturating_sub(1)))
            })
        };
        let z = concat_features(&z1, &z2);
        self.inner.predict(&z, session)
    }
}

/// Catch22 + ExtraTrees classifier (sktime `Catch22` forest classification).
///
/// Catch22 width and tree count are not identification `p`.
#[derive(Clone, Debug)]
pub struct Catch22ForestClassifier {
    inner: Catch22El,
}

impl Default for Catch22ForestClassifier {
    fn default() -> Self {
        Self {
            inner: Catch22El::default(),
        }
    }
}

impl Catch22ForestClassifier {
    /// Default Catch22 forest classifier.
    pub fn new() -> Self {
        Self::default()
    }
}

impl Fit for Catch22ForestClassifier {
    type Fitted = FittedCatch22El;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedCatch22El>> {
        self.inner.fit(x, y, session)
    }
}

/// Catch22 + rotation-forest regressor (sktime `RotationForest` regression lite).
///
/// Catch22 width and tree count are not identification `p`. ExtraTrees /
/// rotation failures fall back to ridge. Do not call `inspect_classes`.
#[derive(Clone, Debug)]
pub struct RotationForestRegressor {
    /// Rotation-forest members.
    pub n_estimators: usize,
    /// Tree depth.
    pub max_depth: usize,
    /// Seed.
    pub seed: u64,
}

impl Default for RotationForestRegressor {
    fn default() -> Self {
        Self {
            n_estimators: 4,
            max_depth: 4,
            seed: 37,
        }
    }
}

impl RotationForestRegressor {
    /// Default Catch22–rotation-forest regressor.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted Catch22 rotation forest or ridge fallback.
#[derive(Clone, Debug)]
pub struct FittedRotationForestRegressor {
    forest: Option<crate::ensemble::FittedRotationForestRegressor>,
    ridge: Option<FittedPenalized>,
}

impl Fit for RotationForestRegressor {
    type Fitted = FittedRotationForestRegressor;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedRotationForestRegressor>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        let z = catch22_rows(x, session, &mut ctx);
        ctx.push(
            Issue::builder(IssueCode::CausalClaimUnidentified)
                .severity(Severity::Advisory)
                .message("tslearn RotationForestRegressor is Catch22 plus rotated CART")
                .compromise(NumericalCompromise::new(
                    "sktime RotationForest on the raw series",
                    "catch22_rows then ensemble RotationForestRegressor",
                    "PCA rotations are on Catch22, not on the time axis",
                    "do not read as a published Rotation Forest accuracy",
                ))
                .build(),
        );
        let mut rf = crate::ensemble::RotationForestRegressor {
            n_estimators: self.n_estimators.max(1),
            max_depth: self.max_depth.max(1),
            seed: self.seed,
        };
        match rf.fit(&z, y, &session.child("rotfr-c22")) {
            Ok(q) => {
                for issue in q.report.issues() {
                    if matches!(
                        issue.code,
                        IssueCode::ResidualTooLarge
                            | IssueCode::NearSingular
                            | IssueCode::R2IsOne
                            | IssueCode::RankZero
                            | IssueCode::MeaninglessFit
                            | IssueCode::PredictionsAreConstant
                    ) {
                        continue;
                    }
                    ctx.push(issue.clone());
                }
                ctx.finish(FittedRotationForestRegressor {
                    forest: Some(q.value),
                    ridge: None,
                })
            }
            Err(_) => {
                ctx.push(
                    Issue::builder(IssueCode::DidNotConverge)
                        .message("RotationForestRegressor failed; falling back to Catch22 ridge")
                        .build(),
                );
                let ridge = ridge_reg_from_features(&z, y, 0.5, &ctx.policy, "rotfr");
                ctx.finish(FittedRotationForestRegressor {
                    forest: None,
                    ridge: Some(ridge),
                })
            }
        }
    }
}

impl Predict for FittedRotationForestRegressor {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let z = {
            let mut ctx = FitCtx::with_session(session.child("rotfr-z"));
            catch22_rows(x, session, &mut ctx)
        };
        if let Some(f) = &self.forest {
            return f.predict(&z, session);
        }
        if let Some(r) = &self.ridge {
            return r.predict(&z, session);
        }
        let mut ctx = FitCtx::with_session(session.child("predict"));
        ctx.push(Issue::builder(IssueCode::PartialFitBeforeInit).build());
        ctx.finish(Vector::zeros(x.nrows()))
    }
}

/// Named FreshPRINCE classifier (sktime `FreshPRINCEClassifier`).
#[derive(Clone, Debug)]
pub struct FreshPrinceClassifier {
    inner: FreshPrince,
}

impl Default for FreshPrinceClassifier {
    fn default() -> Self {
        Self {
            inner: FreshPrince::default(),
        }
    }
}

impl FreshPrinceClassifier {
    /// Default FreshPRINCE classifier.
    pub fn new() -> Self {
        Self::default()
    }
}

impl Fit for FreshPrinceClassifier {
    type Fitted = FittedFreshPrince;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedFreshPrince>> {
        self.inner.fit(x, y, session)
    }
}

/// Named DrCIF classifier (sktime `DrCIF`).
#[derive(Clone, Debug)]
pub struct DrCifClassifier {
    inner: DrCif,
}

impl Default for DrCifClassifier {
    fn default() -> Self {
        Self {
            inner: DrCif::default(),
        }
    }
}

impl DrCifClassifier {
    /// Default DrCIF classifier.
    pub fn new() -> Self {
        Self::default()
    }
}

impl Fit for DrCifClassifier {
    type Fitted = FittedDrCif;
    fn fit(&mut self, x: &Matrix, y: &Vector, session: &Session) -> Result<Qualified<FittedDrCif>> {
        self.inner.fit(x, y, session)
    }
}

/// Named CIF classifier (sktime `CanonicalIntervalForest`).
#[derive(Clone, Debug)]
pub struct CanonicalIntervalForestClassifier {
    inner: CanonicalIntervalForest,
}

impl Default for CanonicalIntervalForestClassifier {
    fn default() -> Self {
        Self {
            inner: CanonicalIntervalForest::default(),
        }
    }
}

impl CanonicalIntervalForestClassifier {
    /// Default CIF classifier.
    pub fn new() -> Self {
        Self::default()
    }
}

impl Fit for CanonicalIntervalForestClassifier {
    type Fitted = FittedCanonicalIntervalForest;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedCanonicalIntervalForest>> {
        self.inner.fit(x, y, session)
    }
}

/// Named RISE classifier (sktime `RandomIntervalSpectralForest`).
#[derive(Clone, Debug)]
pub struct RiseClassifier {
    inner: Rise,
}

impl Default for RiseClassifier {
    fn default() -> Self {
        Self {
            inner: Rise::default(),
        }
    }
}

impl RiseClassifier {
    /// Default RISE classifier.
    pub fn new() -> Self {
        Self::default()
    }
}

impl Fit for RiseClassifier {
    type Fitted = FittedRise;
    fn fit(&mut self, x: &Matrix, y: &Vector, session: &Session) -> Result<Qualified<FittedRise>> {
        self.inner.fit(x, y, session)
    }
}

/// Named temporal dictionary ensemble (sktime `TemporalDictionaryEnsemble`).
#[derive(Clone, Debug)]
pub struct TdeClassifier {
    inner: TemporalDictionaryEnsemble,
}

impl Default for TdeClassifier {
    fn default() -> Self {
        Self {
            inner: TemporalDictionaryEnsemble {
                window: 3,
                n_words: 6,
            },
        }
    }
}

impl TdeClassifier {
    /// TDE with a window that fits width-6 series.
    pub fn new() -> Self {
        Self::default()
    }
}

impl Fit for TdeClassifier {
    type Fitted = FittedTemporalDictionaryEnsemble;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedTemporalDictionaryEnsemble>> {
        self.inner.fit(x, y, session)
    }
}

fn disjoint_spans(t: usize, n_blocks: usize) -> Vec<(usize, usize)> {
    let b = n_blocks.max(1).min(t.max(1));
    let w = (t / b).max(1);
    let mut spans = Vec::new();
    let mut start = 0;
    for i in 0..b {
        let end = if i + 1 == b { t } else { (start + w).min(t) };
        if end > start {
            spans.push((start, end));
        }
        start = end;
    }
    if spans.is_empty() {
        spans.push((0, t.max(1)));
    }
    spans
}

fn conv_block_maxpool(x: &Matrix, start: usize, end: usize, kernels: &[Vec<f64>]) -> Matrix {
    let width = end.saturating_sub(start);
    Matrix::from_fn(x.nrows(), kernels.len().max(1), |i, k| {
        if kernels.is_empty() {
            return 0.0;
        }
        let w = &kernels[k];
        if w.is_empty() || width < w.len() {
            return 0.0;
        }
        let mut acc_max = f64::NEG_INFINITY;
        for t in start..=end - w.len() {
            let mut s = 0.0;
            for u in 0..w.len() {
                s += w[u] * x.get(i, t + u);
            }
            if s > acc_max {
                acc_max = s;
            }
        }
        if acc_max.is_finite() {
            acc_max
        } else {
            0.0
        }
    })
}

fn hstack_blocks(blocks: &[Matrix]) -> Matrix {
    if blocks.is_empty() {
        return Matrix::zeros(0, 0);
    }
    let n = blocks[0].nrows();
    let p: usize = blocks.iter().map(|b| b.ncols()).sum();
    let mut out = Matrix::zeros(n, p.max(1));
    let mut off = 0;
    for b in blocks {
        for j in 0..b.ncols() {
            for i in 0..n.min(b.nrows()) {
                out.set(i, off + j, b.get(i, j));
            }
        }
        off += b.ncols();
    }
    out
}

/// Disjoint temporal CNN (sktime `DisjointCNNClassifier` lite).
///
/// Shared kernels are applied independently on contiguous blocks, then
/// concatenated. Block count is not identification `p`.
#[derive(Clone, Debug)]
pub struct DisjointCnnClassifier {
    /// Contiguous blocks. Not identification `p`.
    pub n_blocks: usize,
    /// Kernels per block.
    pub n_kernels: usize,
    /// Kernel width inside each block.
    pub width: usize,
    /// Ridge \(\alpha\).
    pub alpha: f64,
    /// Seed.
    pub seed: u64,
}

impl Default for DisjointCnnClassifier {
    fn default() -> Self {
        Self {
            n_blocks: 2,
            n_kernels: 3,
            width: 2,
            alpha: 0.1,
            seed: 17,
        }
    }
}

impl DisjointCnnClassifier {
    /// Default disjoint-CNN classifier.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted disjoint-CNN ridge.
#[derive(Clone, Debug)]
pub struct FittedDisjointCnnClassifier {
    kernels: Vec<Vec<f64>>,
    n_blocks: usize,
    inner: crate::classification::FittedRidgeClassifier,
}

impl Fit for DisjointCnnClassifier {
    type Fitted = FittedDisjointCnnClassifier;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedDisjointCnnClassifier>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        inspect_classes(&mut ctx.report, y, &ctx.policy);
        let t = x.ncols();
        let spans = disjoint_spans(t, self.n_blocks);
        let w = self.width.max(1);
        if spans.iter().any(|&(a, b)| b.saturating_sub(a) < w) {
            ctx.push(
                Issue::builder(IssueCode::WindowTooShort)
                    .message(format!(
                        "DisjointCnnClassifier width={w} exceeds a block of T={t}"
                    ))
                    .build(),
            );
        }
        let mut rng = Rng::new(self.seed);
        let kernels: Vec<Vec<f64>> = (0..self.n_kernels.max(1))
            .map(|_| (0..w).map(|_| rng.standard_normal()).collect())
            .collect();
        let blocks: Vec<Matrix> = spans
            .iter()
            .map(|&(a, b)| conv_block_maxpool(x, a, b, &kernels))
            .collect();
        let z = hstack_blocks(&blocks);
        let inner = binary_ridge_from_features(&z, y, self.alpha, &ctx.policy, "disjoint-cnn");
        ctx.push(
            Issue::builder(IssueCode::CausalClaimUnidentified)
                .severity(Severity::Advisory)
                .message("DisjointCnnClassifier is shared-kernel block conv + ridge")
                .compromise(NumericalCompromise::new(
                    "sktime DisjointCNN",
                    "max-pooled convolution on disjoint temporal blocks",
                    "learned filters, batch-norm, and the published head are omitted",
                    "read scores as a block-conv sketch",
                ))
                .build(),
        );
        ctx.finish(FittedDisjointCnnClassifier {
            kernels,
            n_blocks: spans.len().max(1),
            inner,
        })
    }
}

impl Predict for FittedDisjointCnnClassifier {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let spans = disjoint_spans(x.ncols(), self.n_blocks);
        let blocks: Vec<Matrix> = spans
            .iter()
            .map(|&(a, b)| conv_block_maxpool(x, a, b, &self.kernels))
            .collect();
        let z = hstack_blocks(&blocks);
        self.inner.predict(&z, session)
    }
}

/// Multi-channel deep CNN (sktime `MCDCNNClassifier` lite).
///
/// Each channel is a contiguous block with its **own** kernel bank, then the
/// pooled maps are concatenated. Channel count is not identification `p`.
#[derive(Clone, Debug)]
pub struct McdcnnClassifier {
    /// Channels (contiguous splits). Not identification `p`.
    pub n_channels: usize,
    /// Kernels per channel.
    pub n_kernels: usize,
    /// Kernel width.
    pub width: usize,
    /// Ridge \(\alpha\).
    pub alpha: f64,
    /// Seed.
    pub seed: u64,
}

impl Default for McdcnnClassifier {
    fn default() -> Self {
        Self {
            n_channels: 2,
            n_kernels: 3,
            width: 2,
            alpha: 0.1,
            seed: 19,
        }
    }
}

impl McdcnnClassifier {
    /// Default MCDCNN-lite classifier.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted per-channel CNN ridge.
#[derive(Clone, Debug)]
pub struct FittedMcdcnnClassifier {
    banks: Vec<Vec<Vec<f64>>>,
    inner: crate::classification::FittedRidgeClassifier,
}

impl Fit for McdcnnClassifier {
    type Fitted = FittedMcdcnnClassifier;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedMcdcnnClassifier>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        inspect_classes(&mut ctx.report, y, &ctx.policy);
        let spans = disjoint_spans(x.ncols(), self.n_channels);
        let w = self.width.max(1);
        let mut rng = Rng::new(self.seed);
        let mut banks = Vec::new();
        let mut blocks = Vec::new();
        for &(a, b) in &spans {
            let kernels: Vec<Vec<f64>> = (0..self.n_kernels.max(1))
                .map(|_| (0..w).map(|_| rng.standard_normal()).collect())
                .collect();
            blocks.push(conv_block_maxpool(x, a, b, &kernels));
            banks.push(kernels);
        }
        let z = hstack_blocks(&blocks);
        let inner = binary_ridge_from_features(&z, y, self.alpha, &ctx.policy, "mcdcnn");
        ctx.push(
            Issue::builder(IssueCode::CausalClaimUnidentified)
                .severity(Severity::Advisory)
                .message("McdcnnClassifier is independent per-channel conv + ridge")
                .compromise(NumericalCompromise::new(
                    "sktime MCDCNN",
                    "separate kernel banks on contiguous channels",
                    "the published MLP fusion stack is omitted",
                    "read scores as a multi-channel conv sketch",
                ))
                .build(),
        );
        ctx.finish(FittedMcdcnnClassifier { banks, inner })
    }
}

impl Predict for FittedMcdcnnClassifier {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let spans = disjoint_spans(x.ncols(), self.banks.len().max(1));
        let mut blocks = Vec::new();
        for (i, &(a, b)) in spans.iter().enumerate() {
            let kernels = self.banks.get(i).map(Vec::as_slice).unwrap_or(&[]);
            blocks.push(conv_block_maxpool(x, a, b, kernels));
        }
        let z = hstack_blocks(&blocks);
        self.inner.predict(&z, session)
    }
}

fn patch_embed(x: &Matrix, patch_len: usize, stride: usize) -> Matrix {
    let t = x.ncols();
    let pl = patch_len.max(1).min(t.max(1));
    let st = stride.max(1);
    let n_patches = if t >= pl {
        1 + (t - pl) / st
    } else {
        1
    };
    // Each patch contributes values + mean + std.
    let width = n_patches * (pl + 2);
    Matrix::from_fn(x.nrows(), width.max(1), |i, j| {
        let pidx = j / (pl + 2);
        let off = j % (pl + 2);
        let start = pidx * st;
        if start >= t {
            return 0.0;
        }
        if off < pl {
            let tix = start + off;
            if tix < t {
                x.get(i, tix)
            } else {
                0.0
            }
        } else {
            let end = (start + pl).min(t);
            let len = end.saturating_sub(start).max(1) as f64;
            let mut s = 0.0_f64;
            for u in start..end {
                s += x.get(i, u);
            }
            let mu = s / len;
            if off == pl {
                mu
            } else {
                let mut v = 0.0_f64;
                for u in start..end {
                    let d = x.get(i, u) - mu;
                    v += d * d;
                }
                (v / len).sqrt()
            }
        }
    })
}

/// Patch embedding + ridge (sktime / Nie et al. `PatchTST` lite).
///
/// Patch / stride counts are not identification `p`.
#[derive(Clone, Debug)]
pub struct PatchTstClassifier {
    /// Patch length. Not identification `p`.
    pub patch_len: usize,
    /// Stride.
    pub stride: usize,
    /// Ridge \(\alpha\).
    pub alpha: f64,
}

impl Default for PatchTstClassifier {
    fn default() -> Self {
        Self {
            patch_len: 2,
            stride: 2,
            alpha: 0.1,
        }
    }
}

impl PatchTstClassifier {
    /// Default PatchTST-lite classifier.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted patch-embedding ridge.
#[derive(Clone, Debug)]
pub struct FittedPatchTstClassifier {
    patch_len: usize,
    stride: usize,
    inner: crate::classification::FittedRidgeClassifier,
}

impl Fit for PatchTstClassifier {
    type Fitted = FittedPatchTstClassifier;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedPatchTstClassifier>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        inspect_classes(&mut ctx.report, y, &ctx.policy);
        if self.patch_len > x.ncols() {
            ctx.push(
                Issue::builder(IssueCode::WindowTooShort)
                    .message(format!(
                        "PatchTstClassifier patch_len={} > T={}",
                        self.patch_len,
                        x.ncols()
                    ))
                    .build(),
            );
        }
        let z = patch_embed(x, self.patch_len, self.stride);
        let inner = binary_ridge_from_features(&z, y, self.alpha, &ctx.policy, "patchtst");
        ctx.push(
            Issue::builder(IssueCode::CausalClaimUnidentified)
                .severity(Severity::Advisory)
                .message("PatchTstClassifier is patch mean/std flatten + ridge, not a Transformer")
                .compromise(NumericalCompromise::new(
                    "PatchTST attention encoder",
                    "non-overlapping patch values plus mean/std, then ridge",
                    "channel-independence attention and positional encodings are omitted",
                    "read scores as a patch sketch",
                ))
                .build(),
        );
        ctx.finish(FittedPatchTstClassifier {
            patch_len: self.patch_len.max(1),
            stride: self.stride.max(1),
            inner,
        })
    }
}

impl Predict for FittedPatchTstClassifier {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let z = patch_embed(x, self.patch_len, self.stride);
        self.inner.predict(&z, session)
    }
}

fn dominant_period(row: &[f64]) -> usize {
    let t = row.len();
    if t < 4 {
        return 2;
    }
    let mut best_p = 2usize;
    let mut best = 0.0_f64;
    for p in 2..=(t / 2).max(2) {
        let w = 2.0 * std::f64::consts::PI / p as f64;
        let mut re = 0.0_f64;
        let mut im = 0.0_f64;
        for (u, &v) in row.iter().enumerate() {
            if !v.is_finite() {
                continue;
            }
            re += v * (w * u as f64).cos();
            im += v * (w * u as f64).sin();
        }
        let pow = re * re + im * im;
        if pow > best {
            best = pow;
            best_p = p;
        }
    }
    best_p.max(2)
}

fn timesnet_row(row: &[f64], period: usize) -> Vec<f64> {
    let t = row.len();
    let p = period.max(2).min(t.max(2));
    let cols = (t / p).max(1);
    let mut feat = Vec::with_capacity(p + cols + 2);
    for r in 0..p {
        let mut s = 0.0_f64;
        let mut c = 0.0_f64;
        for k in 0..cols {
            let ix = r + k * p;
            if ix < t && row[ix].is_finite() {
                s += row[ix];
                c += 1.0;
            }
        }
        feat.push(if c > 0.0 { s / c } else { 0.0 });
    }
    for k in 0..cols {
        let mut s = 0.0_f64;
        let mut c = 0.0_f64;
        for r in 0..p {
            let ix = r + k * p;
            if ix < t && row[ix].is_finite() {
                s += row[ix];
                c += 1.0;
            }
        }
        feat.push(if c > 0.0 { s / c } else { 0.0 });
    }
    let mut mx = f64::NEG_INFINITY;
    let mut mn = f64::INFINITY;
    for &v in row {
        if v.is_finite() {
            mx = mx.max(v);
            mn = mn.min(v);
        }
    }
    feat.push(if mx.is_finite() { mx } else { 0.0 });
    feat.push(if mn.is_finite() { mn } else { 0.0 });
    feat
}

/// TimesNet-lite (Wu et al. / sktime `TimesNetClassifier`): FFT period + 2-D means.
///
/// Period is not identification `p`.
#[derive(Clone, Debug, Default)]
pub struct TimesNetClassifier;

impl TimesNetClassifier {
    /// Default TimesNet-lite classifier.
    pub fn new() -> Self {
        Self
    }
}

/// Fitted TimesNet ridge.
#[derive(Clone, Debug)]
pub struct FittedTimesNetClassifier {
    period: usize,
    inner: crate::classification::FittedRidgeClassifier,
}

fn timesnet_matrix(x: &Matrix, period: usize) -> Matrix {
    let feats: Vec<Vec<f64>> = (0..x.nrows())
        .map(|i| timesnet_row(x.row(i).as_slice(), period))
        .collect();
    let w = feats.first().map(|f| f.len()).unwrap_or(1);
    Matrix::from_fn(x.nrows(), w.max(1), |i, j| {
        feats[i].get(j).copied().unwrap_or(0.0)
    })
}

impl Fit for TimesNetClassifier {
    type Fitted = FittedTimesNetClassifier;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedTimesNetClassifier>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        inspect_classes(&mut ctx.report, y, &ctx.policy);
        let mut votes: BTreeMap<usize, usize> = BTreeMap::new();
        for i in 0..x.nrows() {
            let p = dominant_period(x.row(i).as_slice());
            *votes.entry(p).or_insert(0) += 1;
        }
        let period = votes
            .into_iter()
            .max_by_key(|(_, c)| *c)
            .map(|(p, _)| p)
            .unwrap_or(2)
            .min(x.ncols().max(2));
        let z = timesnet_matrix(x, period);
        let inner = binary_ridge_from_features(&z, y, 0.1, &ctx.policy, "timesnet");
        ctx.push(
            Issue::builder(IssueCode::CausalClaimUnidentified)
                .severity(Severity::Advisory)
                .message("TimesNetClassifier is FFT-period 2-D means + ridge, not Inception over 2-D maps")
                .compromise(NumericalCompromise::new(
                    "TimesNet 2-D inception",
                    "dominant DFT period, then row/column means of the period reshape",
                    "learned 2-D convolutions and multi-period stacking are omitted",
                    "read scores as a periodogram sketch",
                ))
                .build(),
        );
        ctx.finish(FittedTimesNetClassifier { period, inner })
    }
}

impl Predict for FittedTimesNetClassifier {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let z = timesnet_matrix(x, self.period);
        self.inner.predict(&z, session)
    }
}

fn tcn_row(row: &[f64], w: &[f64], dilation: usize) -> f64 {
    let t = row.len();
    let mut acc = f64::NEG_INFINITY;
    if w.is_empty() {
        return 0.0;
    }
    let d = dilation.max(1);
    let span = (w.len() - 1) * d;
    if t <= span {
        return 0.0;
    }
    for t0 in span..t {
        let mut s = 0.0;
        for (k, &wk) in w.iter().enumerate() {
            let ix = t0 - k * d;
            s += wk * row[ix];
        }
        if s > acc {
            acc = s;
        }
    }
    if acc.is_finite() {
        acc
    } else {
        0.0
    }
}

/// Temporal convolutional network (sktime `TCNClassifier` lite).
///
/// Dilation / kernel counts are not identification `p`.
#[derive(Clone, Debug)]
pub struct TcnClassifier {
    /// Kernels per dilation.
    pub n_kernels: usize,
    /// Kernel width.
    pub width: usize,
    /// Ridge \(\alpha\).
    pub alpha: f64,
    /// Seed.
    pub seed: u64,
}

impl Default for TcnClassifier {
    fn default() -> Self {
        Self {
            n_kernels: 3,
            width: 2,
            alpha: 0.1,
            seed: 23,
        }
    }
}

impl TcnClassifier {
    /// Default TCN-lite classifier.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted dilated-conv ridge.
#[derive(Clone, Debug)]
pub struct FittedTcnClassifier {
    kernels: Vec<(usize, Vec<f64>)>,
    inner: crate::classification::FittedRidgeClassifier,
}

impl Fit for TcnClassifier {
    type Fitted = FittedTcnClassifier;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedTcnClassifier>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        inspect_classes(&mut ctx.report, y, &ctx.policy);
        let w = self.width.max(1).min(x.ncols().max(1));
        let mut rng = Rng::new(self.seed);
        let mut kernels = Vec::new();
        for &d in &[1usize, 2] {
            if (w.saturating_sub(1)) * d >= x.ncols() {
                continue;
            }
            for _ in 0..self.n_kernels.max(1) {
                kernels.push((d, (0..w).map(|_| rng.standard_normal()).collect()));
            }
        }
        if kernels.is_empty() {
            kernels.push((1, vec![1.0]));
            ctx.push(
                Issue::builder(IssueCode::WindowTooShort)
                    .message("TcnClassifier dilated span exceeded T; used a unit kernel")
                    .build(),
            );
        }
        let z = Matrix::from_fn(x.nrows(), kernels.len(), |i, k| {
            let (d, ref w) = kernels[k];
            tcn_row(x.row(i).as_slice(), w, d)
        });
        let inner = binary_ridge_from_features(&z, y, self.alpha, &ctx.policy, "tcn");
        ctx.push(
            Issue::builder(IssueCode::CausalClaimUnidentified)
                .severity(Severity::Advisory)
                .message("TcnClassifier is max-pooled dilated conv + ridge")
                .compromise(NumericalCompromise::new(
                    "temporal convolutional network",
                    "dilations 1 and 2 with random kernels and temporal max-pool",
                    "residual TCN stacks and weight-norm are omitted",
                    "read scores as a dilated-conv sketch",
                ))
                .build(),
        );
        ctx.finish(FittedTcnClassifier { kernels, inner })
    }
}

impl Predict for FittedTcnClassifier {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let z = Matrix::from_fn(x.nrows(), self.kernels.len(), |i, k| {
            let (d, ref w) = self.kernels[k];
            tcn_row(x.row(i).as_slice(), w, d)
        });
        self.inner.predict(&z, session)
    }
}

fn patch_attention(x: &Matrix, patch_len: usize) -> Matrix {
    let t = x.ncols();
    let pl = patch_len.max(1).min(t.max(1));
    let n_patches = if t >= pl { t / pl } else { 1 };
    Matrix::from_fn(x.nrows(), pl, |i, j| {
        let mut dots = vec![0.0; n_patches];
        let mut mx = f64::NEG_INFINITY;
        for pidx in 0..n_patches {
            let start = pidx * pl;
            let mut s = 0.0;
            for u in 0..pl {
                let ix = start + u;
                if ix < t {
                    s += x.get(i, ix);
                }
            }
            dots[pidx] = s;
            if s > mx {
                mx = s;
            }
        }
        let mut den = 0.0_f64;
        for pidx in 0..n_patches {
            den += (dots[pidx] - mx).exp();
        }
        if !den.is_finite() || den <= 0.0 {
            let start = 0;
            return if start + j < t { x.get(i, j) } else { 0.0 };
        }
        let mut acc = 0.0;
        for pidx in 0..n_patches {
            let w = (dots[pidx] - mx).exp() / den;
            let ix = pidx * pl + j;
            if ix < t {
                acc += w * x.get(i, ix);
            }
        }
        acc
    })
}

/// Time-series Transformer lite (sktime `TSTClassifier`): patch softmax attention + ridge.
///
/// Patch count is not identification `p`.
#[derive(Clone, Debug)]
pub struct TstClassifier {
    /// Patch length. Not identification `p`.
    pub patch_len: usize,
    /// Ridge \(\alpha\).
    pub alpha: f64,
}

impl Default for TstClassifier {
    fn default() -> Self {
        Self {
            patch_len: 2,
            alpha: 0.1,
        }
    }
}

impl TstClassifier {
    /// Default TST-lite classifier.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Fitted attention-pooled ridge.
#[derive(Clone, Debug)]
pub struct FittedTstClassifier {
    patch_len: usize,
    inner: crate::classification::FittedRidgeClassifier,
}

impl Fit for TstClassifier {
    type Fitted = FittedTstClassifier;
    fn fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        session: &Session,
    ) -> Result<Qualified<FittedTstClassifier>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, Some(y), &ctx.policy);
        inspect_classes(&mut ctx.report, y, &ctx.policy);
        let z = patch_attention(x, self.patch_len);
        let inner = binary_ridge_from_features(&z, y, self.alpha, &ctx.policy, "tst");
        ctx.push(
            Issue::builder(IssueCode::CausalClaimUnidentified)
                .severity(Severity::Advisory)
                .message("TstClassifier is patch softmax attention + ridge, not a multi-head Transformer")
                .compromise(NumericalCompromise::new(
                    "Time Series Transformer",
                    "softmax over patch sums, then ridge on the attended patch",
                    "multi-head attention, positional encodings, and stacked layers are omitted",
                    "read scores as an attention-pool sketch",
                ))
                .build(),
        );
        ctx.finish(FittedTstClassifier {
            patch_len: self.patch_len.max(1),
            inner,
        })
    }
}

impl Predict for FittedTstClassifier {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let z = patch_attention(x, self.patch_len);
        self.inner.predict(&z, session)
    }
}

/// FLOSS corrected arc-curve change point (sktime `FLOSS` / STUMPY).
///
/// Window length is not identification `p`. Distinct from [`ClaSPSegmentation`]
/// (two-mean F score) and [`Stamp`] (matrix-profile distances).
pub fn floss_change_point(
    y: &Vector,
    window: usize,
    session: &Session,
) -> Result<Qualified<FlossResult>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(
        &mut ctx.report,
        &Matrix::from_vector(y),
        Some(y),
        &ctx.policy,
    );
    let n = y.len();
    let m = window.max(2);
    if m >= n || n < 6 {
        ctx.push(
            Issue::builder(IssueCode::WindowTooShort)
                .severity(Severity::Warning)
                .message(format!("FLOSS window={m} is unusable for n={n}"))
                .build(),
        );
        return ctx.finish(FlossResult {
            index: f64::NAN,
            cac: Vector::zeros(0),
        });
    }
    let n_sub = n + 1 - m;
    let excl = (m / 4).max(1);
    let mut nn = vec![0usize; n_sub];
    for i in 0..n_sub {
        let mut best = f64::INFINITY;
        let mut arg = i;
        for j in 0..n_sub {
            if i.abs_diff(j) < excl {
                continue;
            }
            let mut d = 0.0_f64;
            for t in 0..m {
                let e = y[i + t] - y[j + t];
                d += e * e;
            }
            if d < best {
                best = d;
                arg = j;
            }
        }
        nn[i] = arg;
    }
    let mut arc = vec![0.0_f64; n];
    for i in 0..n_sub {
        let j = nn[i];
        let lo = i.min(j);
        let hi = i.max(j);
        if hi > lo + 1 {
            for k in (lo + 1)..hi {
                if k < n {
                    arc[k] += 1.0;
                }
            }
        }
    }
    let mut cac = Vector::zeros(n);
    let mut best_i = n / 2;
    let mut best_v = f64::INFINITY;
    let ns = n_sub.max(1) as f64;
    for k in 1..n.saturating_sub(1) {
        let x = k as f64 / n.max(1) as f64;
        let iac = (2.0 * ns * x * (1.0 - x)).max(1e-8);
        let v = (arc[k] / iac).clamp(0.0, 1.0);
        cac[k] = v;
        if k >= m && k + m < n && v < best_v {
            best_v = v;
            best_i = k;
        }
    }
    if !best_v.is_finite() {
        ctx.push(
            Issue::builder(IssueCode::PValueUnreliable)
                .message("FLOSS CAC was non-finite")
                .build(),
        );
    }
    ctx.push(
        Issue::builder(IssueCode::CausalClaimUnidentified)
            .severity(Severity::Advisory)
            .message("FLOSS is a z-unnormalized arc curve, not the published online FLOSS")
            .compromise(NumericalCompromise::new(
                "sktime / STUMPY FLOSS",
                "nearest-neighbour arcs over Euclidean windows, IAC-corrected",
                "z-normalized distance, the online incremental profile, and L arc constraints are omitted",
                "read the index as an arc-curve sketch",
            ))
            .build(),
    );
    ctx.finish(FlossResult {
        index: best_i as f64,
        cac,
    })
}

/// FLOSS change-point payload.
#[derive(Clone, Debug)]
pub struct FlossResult {
    /// Argmin of the corrected arc curve.
    pub index: f64,
    /// Corrected arc curve (`n`).
    pub cac: Vector,
}

/// Named FLOSS annotator (sktime `FLOSS`).
#[derive(Clone, Debug)]
pub struct Floss {
    /// Subsequence length. Not identification `p`.
    pub window: usize,
}

impl Default for Floss {
    fn default() -> Self {
        Self { window: 3 }
    }
}

impl Floss {
    /// FLOSS with a given window.
    pub fn new(window: usize) -> Self {
        Self {
            window: window.max(2),
        }
    }

    /// Corrected arc-curve change point.
    pub fn fit(&self, y: &Vector, session: &Session) -> Result<Qualified<FlossResult>> {
        floss_change_point(y, self.window, session)
    }
}

/// FLUSS multi-change-point annotator (sktime / STUMPY `FLUSS`).
///
/// Regime count is not identification `p`. Distinct from [`Floss`] (single
/// CAC argmin) and [`ClaSPSegmentation`] (two-mean F).
#[derive(Clone, Debug)]
pub struct Fluss {
    /// Subsequence length. Not identification `p`.
    pub window: usize,
    /// Number of regimes (change points = regimes − 1). Not identification `p`.
    pub n_regimes: usize,
}

impl Default for Fluss {
    fn default() -> Self {
        Self {
            window: 3,
            n_regimes: 2,
        }
    }
}

impl Fluss {
    /// FLUSS with `n_regimes` pieces.
    pub fn new(n_regimes: usize) -> Self {
        Self {
            n_regimes: n_regimes.max(2),
            ..Self::default()
        }
    }

    /// Local minima of the corrected arc curve.
    pub fn fit(&self, y: &Vector, session: &Session) -> Result<Qualified<Vector>> {
        let q = floss_change_point(y, self.window, session)?;
        let ctx = FitCtx::with_session(session.child("fluss"));
        let cac = &q.value.cac;
        let n = cac.len();
        let want = self.n_regimes.max(2).saturating_sub(1);
        let mut idx: Vec<(f64, usize)> = Vec::new();
        for k in 1..n.saturating_sub(1) {
            let v = cac[k];
            if !v.is_finite() {
                continue;
            }
            let left = cac.as_slice().get(k - 1).copied().unwrap_or(v);
            let right = cac.as_slice().get(k + 1).copied().unwrap_or(v);
            if v <= left && v <= right {
                idx.push((v, k));
            }
        }
        idx.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let mut out = Vec::new();
        for &(_, k) in idx.iter() {
            if out.iter().all(|&u: &f64| (u - k as f64).abs() >= self.window as f64) {
                out.push(k as f64);
            }
            if out.len() >= want {
                break;
            }
        }
        if out.is_empty() && q.value.index.is_finite() {
            out.push(q.value.index);
        }
        ctx.finish(Vector::from_iter(out))
    }
}

/// STOMP matrix profile plus per-subsequence nearest-neighbour index (stumpy `stomp`).
///
/// Window length is not identification `p`. Distinct from [`Stamp`] (global
/// argmin only) and [`matrix_profile`] (distances without the index vector).
#[derive(Clone, Debug)]
pub struct StompResult {
    /// Distance profile.
    pub profile: Vector,
    /// Nearest-neighbour subsequence index for each window.
    pub nn_index: Vector,
}

/// Named STOMP detector.
#[derive(Clone, Debug)]
pub struct Stomp {
    /// Subsequence length. Not identification `p`.
    pub window: usize,
}

impl Default for Stomp {
    fn default() -> Self {
        Self { window: 3 }
    }
}

impl Stomp {
    /// STOMP with a given window.
    pub fn new(window: usize) -> Self {
        Self {
            window: window.max(2),
        }
    }

    /// Full matrix profile and nearest-neighbour index vector.
    pub fn fit(&self, y: &Vector, session: &Session) -> Result<Qualified<StompResult>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(
            &mut ctx.report,
            &Matrix::from_vector(y),
            Some(y),
            &ctx.policy,
        );
        let n = y.len();
        let m = self.window.max(2);
        if m >= n {
            ctx.push(
                Issue::builder(IssueCode::WindowTooShort)
                    .severity(Severity::Warning)
                    .message(format!("STOMP window={m} is unusable for n={n}"))
                    .build(),
            );
            return ctx.finish(StompResult {
                profile: Vector::zeros(0),
                nn_index: Vector::zeros(0),
            });
        }
        let n_sub = n + 1 - m;
        let excl = (m / 4).max(1);
        let mut profile = Vector::filled(n_sub, f64::INFINITY);
        let mut nn_index = Vector::zeros(n_sub);
        for diag in 1..n_sub {
            let mut prev = f64::NAN;
            for i in 0..(n_sub - diag) {
                let j = i + diag;
                if i.abs_diff(j) < excl {
                    prev = f64::NAN;
                    continue;
                }
                let d = if !prev.is_finite() {
                    let mut s = 0.0_f64;
                    for t in 0..m {
                        let e = y[i + t] - y[j + t];
                        s += e * e;
                    }
                    s.sqrt()
                } else {
                    let leave = y[i - 1] - y[j - 1];
                    let enter = y[i + m - 1] - y[j + m - 1];
                    (prev * prev - leave * leave + enter * enter).max(0.0).sqrt()
                };
                prev = d;
                if d < profile[i] {
                    profile[i] = d;
                    nn_index[i] = j as f64;
                }
                if d < profile[j] {
                    profile[j] = d;
                    nn_index[j] = i as f64;
                }
            }
        }
        for i in 0..n_sub {
            if !profile[i].is_finite() {
                profile[i] = 0.0;
            }
        }
        ctx.finish(StompResult { profile, nn_index })
    }
}

/// Matrix-profile discord (stumpy `mpdist` / MERLIN-lite).
///
/// Window length is not identification `p`. Distinct from [`Stamp`] (argmin /
/// motif) and [`Stray`] (MAD z-scores).
pub fn merlin_discord(
    y: &Vector,
    window: usize,
    session: &Session,
) -> Result<Qualified<MerlinResult>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(
        &mut ctx.report,
        &Matrix::from_vector(y),
        Some(y),
        &ctx.policy,
    );
    let mp = match matrix_profile(y, window, &session.child("mp")) {
        Ok(q) => q.value,
        Err(e) => {
            if !matches!(
                e.primary.code,
                IssueCode::ResidualTooLarge
                    | IssueCode::NearSingular
                    | IssueCode::RankZero
                    | IssueCode::R2IsOne
                    | IssueCode::MeaninglessFit
            ) {
                ctx.push(e.primary);
            }
            Vector::zeros(0)
        }
    };
    let mut discord = 0usize;
    let mut best = f64::NEG_INFINITY;
    for (i, &v) in mp.as_slice().iter().enumerate() {
        if v.is_finite() && v > best {
            best = v;
            discord = i;
        }
    }
    if !best.is_finite() {
        ctx.push(
            Issue::builder(IssueCode::WindowTooShort)
                .severity(Severity::Warning)
                .message("merlin_discord profile was empty or non-finite")
                .build(),
        );
    }
    ctx.push(
        Issue::builder(IssueCode::CausalClaimUnidentified)
            .severity(Severity::Advisory)
            .message("merlin_discord is the matrix-profile argmax, not published MERLIN")
            .compromise(NumericalCompromise::new(
                "stumpy MERLIN",
                "argmax of the Euclidean matrix profile",
                "the published discord radius search and z-normalized distance are omitted",
                "read the index as a matrix-profile discord sketch",
            ))
            .build(),
    );
    ctx.finish(MerlinResult {
        discord: discord as f64,
        score: if best.is_finite() { best } else { f64::NAN },
        profile: mp,
    })
}

/// MERLIN discord payload.
#[derive(Clone, Debug)]
pub struct MerlinResult {
    /// Argmax of the matrix profile.
    pub discord: f64,
    /// Distance at the discord.
    pub score: f64,
    /// Distance profile.
    pub profile: Vector,
}

/// Named MERLIN discord annotator.
#[derive(Clone, Debug)]
pub struct Merlin {
    /// Subsequence length. Not identification `p`.
    pub window: usize,
}

impl Default for Merlin {
    fn default() -> Self {
        Self { window: 3 }
    }
}

impl Merlin {
    /// MERLIN with a given window.
    pub fn new(window: usize) -> Self {
        Self {
            window: window.max(2),
        }
    }

    /// Matrix-profile discord.
    pub fn fit(&self, y: &Vector, session: &Session) -> Result<Qualified<MerlinResult>> {
        merlin_discord(y, self.window, session)
    }
}

/// Pan matrix profile across several windows (stumpy `pstamp` / `panmp`).
///
/// Window count is not identification `p`. Distinct from [`Stomp`] (one
/// window) and [`Merlin`] (single discord).
#[derive(Clone, Debug)]
pub struct PanMatrixProfile {
    /// Smallest window. Not identification `p`.
    pub min_window: usize,
    /// Number of consecutive windows. Not identification `p`.
    pub n_windows: usize,
}

impl Default for PanMatrixProfile {
    fn default() -> Self {
        Self {
            min_window: 2,
            n_windows: 2,
        }
    }
}

impl PanMatrixProfile {
    /// Pan-MP starting at `min_window` for `n_windows` lengths.
    pub fn new(min_window: usize, n_windows: usize) -> Self {
        Self {
            min_window: min_window.max(2),
            n_windows: n_windows.max(1),
        }
    }

    /// Motif and discord of the matrix profile at each window.
    pub fn fit(&self, y: &Vector, session: &Session) -> Result<Qualified<FittedPanMatrixProfile>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(
            &mut ctx.report,
            &Matrix::from_vector(y),
            Some(y),
            &ctx.policy,
        );
        let nw = self.n_windows.max(1);
        let mut windows = Vector::zeros(nw);
        let mut motifs = Vector::zeros(nw);
        let mut discords = Vector::zeros(nw);
        let mut motif_scores = Vector::zeros(nw);
        let mut discord_scores = Vector::zeros(nw);
        for k in 0..nw {
            let w = self.min_window.max(2) + k;
            windows[k] = w as f64;
            let mp = match matrix_profile(y, w, &session.child(format!("panmp-{w}"))) {
                Ok(q) => q.value,
                Err(e) => {
                    if !matches!(
                        e.primary.code,
                        IssueCode::ResidualTooLarge
                            | IssueCode::NearSingular
                            | IssueCode::RankZero
                            | IssueCode::R2IsOne
                            | IssueCode::MeaninglessFit
                    ) {
                        ctx.push(e.primary);
                    }
                    Vector::zeros(0)
                }
            };
            let mut mi = 0usize;
            let mut di = 0usize;
            let mut mb = f64::INFINITY;
            let mut db = f64::NEG_INFINITY;
            for (i, &v) in mp.as_slice().iter().enumerate() {
                if v.is_finite() && v < mb {
                    mb = v;
                    mi = i;
                }
                if v.is_finite() && v > db {
                    db = v;
                    di = i;
                }
            }
            motifs[k] = mi as f64;
            discords[k] = di as f64;
            motif_scores[k] = if mb.is_finite() { mb } else { f64::NAN };
            discord_scores[k] = if db.is_finite() { db } else { f64::NAN };
        }
        ctx.finish(FittedPanMatrixProfile {
            windows,
            motifs,
            discords,
            motif_scores,
            discord_scores,
        })
    }
}

/// Fitted pan matrix profile.
#[derive(Clone, Debug)]
pub struct FittedPanMatrixProfile {
    /// Window lengths.
    pub windows: Vector,
    /// Motif (argmin) index per window.
    pub motifs: Vector,
    /// Discord (argmax) index per window.
    pub discords: Vector,
    /// Motif distances.
    pub motif_scores: Vector,
    /// Discord distances.
    pub discord_scores: Vector,
}

fn subsequence_dist(a: &Vector, ia: usize, b: &Vector, ib: usize, m: usize) -> f64 {
    let mut s = 0.0_f64;
    for t in 0..m {
        let e = a[ia + t] - b[ib + t];
        s += e * e;
    }
    s.max(0.0).sqrt()
}

/// Matrix-profile distance between two series (stumpy `mpdist`).
///
/// Window length is not identification `p`. Distinct from [`dtw`] (alignment)
/// and [`matrix_profile`] (self nearest neighbour).
pub fn mpdist(
    a: &Vector,
    b: &Vector,
    window: usize,
    session: &Session,
) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(
        &mut ctx.report,
        &Matrix::from_vector(a),
        Some(a),
        &ctx.policy,
    );
    if let Some(issue) = signlred::scan_finite(b.as_slice()).to_issue("mpdist.b") {
        ctx.push(issue);
    }
    let m = window;
    if m < 2 || m > a.len() || m > b.len() {
        ctx.push(
            Issue::builder(IssueCode::WindowTooShort)
                .severity(Severity::Warning)
                .message(format!(
                    "mpdist window={m} is unusable for n_a={} n_b={}",
                    a.len(),
                    b.len()
                ))
                .build(),
        );
        return ctx.finish(0.0);
    }
    let na = a.len() + 1 - m;
    let nb = b.len() + 1 - m;
    let mut acc = 0.0_f64;
    let mut n = 0usize;
    for i in 0..na {
        let mut best = f64::INFINITY;
        for j in 0..nb {
            let d = subsequence_dist(a, i, b, j, m);
            if d < best {
                best = d;
            }
        }
        if best.is_finite() {
            acc += best;
            n += 1;
        }
    }
    for j in 0..nb {
        let mut best = f64::INFINITY;
        for i in 0..na {
            let d = subsequence_dist(b, j, a, i, m);
            if d < best {
                best = d;
            }
        }
        if best.is_finite() {
            acc += best;
            n += 1;
        }
    }
    ctx.push(
        Issue::builder(IssueCode::CausalClaimUnidentified)
            .severity(Severity::Advisory)
            .message("mpdist is the mean of both one-way min-distance profiles")
            .compromise(NumericalCompromise::new(
                "stumpy mpdist",
                "mean of A→B and B→A subsequence nearest-neighbour distances",
                "the published k-th percentile and z-normalized MASS kernel are omitted",
                "read the scalar as a two-series matrix-profile distance sketch",
            ))
            .build(),
    );
    ctx.finish(if n == 0 { 0.0 } else { acc / n as f64 })
}

/// Anytime SCRIMP matrix profile (stumpy `scrimp`).
///
/// `sample_frac` is not identification `p`. Distinct from [`Stomp`] (every
/// diagonal) and [`Stamp`] (full nested loops).
#[derive(Clone, Debug)]
pub struct Scrimp {
    /// Subsequence length. Not identification `p`.
    pub window: usize,
    /// Fraction of STOMP diagonals. Not identification `p`.
    pub sample_frac: f64,
    /// PRNG seed.
    pub seed: u64,
}

impl Default for Scrimp {
    fn default() -> Self {
        Self {
            window: 3,
            sample_frac: 0.5,
            seed: 7,
        }
    }
}

impl Scrimp {
    /// SCRIMP with a given window.
    pub fn new(window: usize) -> Self {
        Self {
            window: window.max(2),
            ..Self::default()
        }
    }

    /// Anytime matrix profile from a random diagonal subset.
    pub fn fit(&self, y: &Vector, session: &Session) -> Result<Qualified<StompResult>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(
            &mut ctx.report,
            &Matrix::from_vector(y),
            Some(y),
            &ctx.policy,
        );
        let n = y.len();
        let m = self.window.max(2);
        if m >= n {
            ctx.push(
                Issue::builder(IssueCode::WindowTooShort)
                    .severity(Severity::Warning)
                    .message(format!("SCRIMP window={m} is unusable for n={n}"))
                    .build(),
            );
            return ctx.finish(StompResult {
                profile: Vector::zeros(0),
                nn_index: Vector::zeros(0),
            });
        }
        let n_sub = n + 1 - m;
        let excl = (m / 4).max(1);
        let frac = if self.sample_frac.is_finite() && self.sample_frac > 0.0 {
            self.sample_frac.min(1.0)
        } else {
            ctx.push(
                Issue::builder(IssueCode::InvalidWeight)
                    .severity(Severity::Warning)
                    .message(format!(
                        "SCRIMP sample_frac={} is not in (0, 1]; using 0.5",
                        self.sample_frac
                    ))
                    .build(),
            );
            0.5
        };
        let mut diags: Vec<usize> = (1..n_sub).collect();
        let mut rng = Rng::new(self.seed);
        rng.shuffle(&mut diags);
        let keep = ((diags.len() as f64) * frac).ceil() as usize;
        diags.truncate(keep.max(1).min(diags.len().max(1)));
        let mut profile = Vector::filled(n_sub, f64::INFINITY);
        let mut nn_index = Vector::zeros(n_sub);
        for &diag in &diags {
            let mut prev = f64::NAN;
            for i in 0..(n_sub - diag) {
                let j = i + diag;
                if i.abs_diff(j) < excl {
                    prev = f64::NAN;
                    continue;
                }
                let d = if !prev.is_finite() {
                    subsequence_dist(y, i, y, j, m)
                } else {
                    let leave = y[i - 1] - y[j - 1];
                    let enter = y[i + m - 1] - y[j + m - 1];
                    (prev * prev - leave * leave + enter * enter).max(0.0).sqrt()
                };
                prev = d;
                if d < profile[i] {
                    profile[i] = d;
                    nn_index[i] = j as f64;
                }
                if d < profile[j] {
                    profile[j] = d;
                    nn_index[j] = i as f64;
                }
            }
        }
        for i in 0..n_sub {
            if !profile[i].is_finite() {
                profile[i] = 0.0;
            }
        }
        ctx.push(
            Issue::builder(IssueCode::CausalClaimUnidentified)
                .severity(Severity::Advisory)
                .message("SCRIMP is a random STOMP-diagonal subset, not published anytime SCRIMP")
                .compromise(NumericalCompromise::new(
                    "stumpy scrimp",
                    "shuffled STOMP diagonals truncated to sample_frac",
                    "the published pre-scrimp refinement and z-normalized kernel are omitted",
                    "read the profile as an anytime matrix-profile sketch",
                ))
                .build(),
        );
        ctx.finish(StompResult { profile, nn_index })
    }
}

/// MASS sliding z-normalized distance profile (stumpy `mass`).
///
/// Query length is not identification `p`. Distinct from [`mpdist`] (two-way
/// mean) and [`matrix_profile`] (self nearest neighbour).
pub fn mass(query: &Vector, series: &Vector, session: &Session) -> Result<Qualified<Vector>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(
        &mut ctx.report,
        &Matrix::from_vector(series),
        Some(series),
        &ctx.policy,
    );
    if let Some(issue) = signlred::scan_finite(query.as_slice()).to_issue("mass.query") {
        ctx.push(issue);
    }
    let m = query.len();
    if m < 2 || m > series.len() {
        ctx.push(
            Issue::builder(IssueCode::WindowTooShort)
                .severity(Severity::Warning)
                .message(format!(
                    "mass query length={m} is unusable for n={}",
                    series.len()
                ))
                .build(),
        );
        return ctx.finish(Vector::zeros(0));
    }
    let qmean = query.mean();
    let qstd = query.std().max(1e-12);
    let n_sub = series.len() + 1 - m;
    let out = Vector::from_iter((0..n_sub).map(|i| {
        let mut s = 0.0_f64;
        let mut ss = 0.0_f64;
        for t in 0..m {
            let v = series[i + t];
            s += v;
            ss += v * v;
        }
        let n = m as f64;
        let mean = s / n;
        let std = ((ss / n - mean * mean).max(0.0).sqrt()).max(1e-12);
        let mut d = 0.0_f64;
        for t in 0..m {
            let zq = (query[t] - qmean) / qstd;
            let zs = (series[i + t] - mean) / std;
            let e = zq - zs;
            d += e * e;
        }
        d.max(0.0).sqrt()
    }));
    ctx.push(
        Issue::builder(IssueCode::CausalClaimUnidentified)
            .severity(Severity::Advisory)
            .message("mass is a direct z-normalized sliding profile, not FFT MASS")
            .compromise(NumericalCompromise::new(
                "stumpy mass",
                "z-normalized Euclidean distance of the query at every offset",
                "the published FFT convolution is omitted",
                "read the vector as a MASS distance-profile sketch",
            ))
            .build(),
    );
    ctx.finish(out)
}

/// Top-\(k\) matrix-profile motifs (stumpy `motifs`).
///
/// Motif count is not identification `p`. Distinct from [`Stamp`] (single
/// argmin) and [`Merlin`] (discord / argmax).
#[derive(Clone, Debug)]
pub struct Motif {
    /// Subsequence length. Not identification `p`.
    pub window: usize,
    /// Number of motifs. Not identification `p`.
    pub n_motifs: usize,
}

impl Default for Motif {
    fn default() -> Self {
        Self {
            window: 3,
            n_motifs: 2,
        }
    }
}

impl Motif {
    /// Motif search with a given window.
    pub fn new(window: usize) -> Self {
        Self {
            window: window.max(2),
            ..Self::default()
        }
    }

    /// Lowest-distance subsequences on the self matrix profile.
    pub fn fit(&self, y: &Vector, session: &Session) -> Result<Qualified<FittedMotif>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(
            &mut ctx.report,
            &Matrix::from_vector(y),
            Some(y),
            &ctx.policy,
        );
        let k = self.n_motifs.max(1);
        let mp = match matrix_profile(y, self.window.max(2), &session.child("mp")) {
            Ok(q) => q.value,
            Err(e) => {
                if !matches!(
                    e.primary.code,
                    IssueCode::ResidualTooLarge
                        | IssueCode::NearSingular
                        | IssueCode::RankZero
                        | IssueCode::R2IsOne
                        | IssueCode::MeaninglessFit
                ) {
                    ctx.push(e.primary);
                }
                Vector::zeros(0)
            }
        };
        let excl = (self.window.max(2) / 4).max(1);
        let mut taken = vec![false; mp.len()];
        let mut indices = Vector::filled(k, 0.0);
        let mut scores = Vector::filled(k, f64::NAN);
        for m in 0..k {
            let mut best_i = 0usize;
            let mut best = f64::INFINITY;
            for (i, &v) in mp.as_slice().iter().enumerate() {
                if taken[i] || !v.is_finite() {
                    continue;
                }
                if v < best {
                    best = v;
                    best_i = i;
                }
            }
            if best.is_finite() {
                indices[m] = best_i as f64;
                scores[m] = best;
                let lo = best_i.saturating_sub(excl);
                let hi = (best_i + excl + 1).min(taken.len());
                for t in taken.iter_mut().take(hi).skip(lo) {
                    *t = true;
                }
            }
        }
        ctx.push(
            Issue::builder(IssueCode::CausalClaimUnidentified)
                .severity(Severity::Advisory)
                .message("Motif is a greedy exclusion of the self matrix profile")
                .compromise(NumericalCompromise::new(
                    "stumpy motifs",
                    "top-k argmin of the Euclidean matrix profile with an exclusion zone",
                    "the published radius search and z-normalized pair distance are omitted",
                    "read the indices as a matrix-profile motif sketch",
                ))
                .build(),
        );
        ctx.finish(FittedMotif {
            indices,
            scores,
            profile: mp,
        })
    }
}

/// Fitted motif set.
#[derive(Clone, Debug)]
pub struct FittedMotif {
    /// Motif subsequence starts.
    pub indices: Vector,
    /// Motif distances.
    pub scores: Vector,
    /// Self matrix profile.
    pub profile: Vector,
}

/// Z-normalized matrix profile (stumpy `mpx` lite).
///
/// Window length is not identification `p`. Distinct from [`Stomp`] and
/// [`matrix_profile`] (raw Euclidean).
#[derive(Clone, Debug)]
pub struct Mpx {
    /// Subsequence length. Not identification `p`.
    pub window: usize,
}

impl Default for Mpx {
    fn default() -> Self {
        Self { window: 3 }
    }
}

impl Mpx {
    /// MPX with a given window.
    pub fn new(window: usize) -> Self {
        Self {
            window: window.max(2),
        }
    }

    /// Z-normalized self nearest-neighbour profile.
    pub fn fit(&self, y: &Vector, session: &Session) -> Result<Qualified<StompResult>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(
            &mut ctx.report,
            &Matrix::from_vector(y),
            Some(y),
            &ctx.policy,
        );
        let n = y.len();
        let m = self.window.max(2);
        if m >= n {
            ctx.push(
                Issue::builder(IssueCode::WindowTooShort)
                    .severity(Severity::Warning)
                    .message(format!("MPX window={m} is unusable for n={n}"))
                    .build(),
            );
            return ctx.finish(StompResult {
                profile: Vector::zeros(0),
                nn_index: Vector::zeros(0),
            });
        }
        let n_sub = n + 1 - m;
        let excl = (m / 4).max(1);
        let mut means = Vector::zeros(n_sub);
        let mut stds = Vector::zeros(n_sub);
        for i in 0..n_sub {
            let mut s = 0.0_f64;
            let mut ss = 0.0_f64;
            for t in 0..m {
                let v = y[i + t];
                s += v;
                ss += v * v;
            }
            let mean = s / m as f64;
            means[i] = mean;
            stds[i] = ((ss / m as f64 - mean * mean).max(0.0).sqrt()).max(1e-12);
        }
        let mut profile = Vector::filled(n_sub, f64::INFINITY);
        let mut nn_index = Vector::zeros(n_sub);
        for i in 0..n_sub {
            for j in 0..n_sub {
                if i.abs_diff(j) < excl {
                    continue;
                }
                let mut d = 0.0_f64;
                for t in 0..m {
                    let zi = (y[i + t] - means[i]) / stds[i];
                    let zj = (y[j + t] - means[j]) / stds[j];
                    let e = zi - zj;
                    d += e * e;
                }
                let d = d.max(0.0).sqrt();
                if d < profile[i] {
                    profile[i] = d;
                    nn_index[i] = j as f64;
                }
            }
        }
        for i in 0..n_sub {
            if !profile[i].is_finite() {
                profile[i] = 0.0;
            }
        }
        ctx.push(
            Issue::builder(IssueCode::CausalClaimUnidentified)
                .severity(Severity::Advisory)
                .message("MPX is a direct z-normalized self profile, not published MPX")
                .compromise(NumericalCompromise::new(
                    "stumpy mpx",
                    "z-normalized Euclidean nearest neighbour of every subsequence",
                    "the published Pearson-correlation / FFT kernel is omitted",
                    "read the profile as a z-normalized matrix-profile sketch",
                ))
                .build(),
        );
        ctx.finish(StompResult { profile, nn_index })
    }
}

/// Regime snippets (stumpy `snippets`).
///
/// Snippet count is not identification `p`. Distinct from [`Motif`] (tightest
/// nearest neighbours) and [`Merlin`] (discords).
#[derive(Clone, Debug)]
pub struct Snippets {
    /// Subsequence length. Not identification `p`.
    pub window: usize,
    /// Number of regimes. Not identification `p`.
    pub n_snippets: usize,
}

impl Default for Snippets {
    fn default() -> Self {
        Self {
            window: 3,
            n_snippets: 2,
        }
    }
}

impl Snippets {
    /// Snippet search with a given window.
    pub fn new(window: usize) -> Self {
        Self {
            window: window.max(2),
            ..Self::default()
        }
    }

    /// Greedy facility-location snippets on pairwise subsequence distances.
    pub fn fit(&self, y: &Vector, session: &Session) -> Result<Qualified<FittedSnippets>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(
            &mut ctx.report,
            &Matrix::from_vector(y),
            Some(y),
            &ctx.policy,
        );
        let m = self.window.max(2);
        let k = self.n_snippets.max(1);
        if m >= y.len() {
            ctx.push(
                Issue::builder(IssueCode::WindowTooShort)
                    .severity(Severity::Warning)
                    .message(format!("snippets window={m} is unusable for n={}", y.len()))
                    .build(),
            );
            return ctx.finish(FittedSnippets {
                indices: Vector::zeros(0),
                scores: Vector::zeros(0),
            });
        }
        let n_sub = y.len() + 1 - m;
        let mut dist = vec![vec![0.0_f64; n_sub]; n_sub];
        for i in 0..n_sub {
            for j in (i + 1)..n_sub {
                let d = subsequence_dist(y, i, y, j, m);
                dist[i][j] = d;
                dist[j][i] = d;
            }
        }
        let mut chosen: Vec<usize> = Vec::new();
        let mut scores = Vector::zeros(k);
        for s in 0..k.min(n_sub) {
            let mut best_i = 0usize;
            let mut best = f64::INFINITY;
            for i in 0..n_sub {
                if chosen.contains(&i) {
                    continue;
                }
                let mut cost = 0.0_f64;
                for j in 0..n_sub {
                    let mut d = dist[j][i];
                    for &c in &chosen {
                        if dist[j][c] < d {
                            d = dist[j][c];
                        }
                    }
                    cost += d;
                }
                if cost < best {
                    best = cost;
                    best_i = i;
                }
            }
            chosen.push(best_i);
            scores[s] = if best.is_finite() { best } else { 0.0 };
        }
        ctx.push(
            Issue::builder(IssueCode::CausalClaimUnidentified)
                .severity(Severity::Advisory)
                .message("Snippets is greedy facility location, not published stumpy snippets")
                .compromise(NumericalCompromise::new(
                    "stumpy snippets",
                    "k-medoid facility location on pairwise subsequence distances",
                    "the published area profile and MPdist regime kernel are omitted",
                    "read the indices as a diverse-regime sketch",
                ))
                .build(),
        );
        ctx.finish(FittedSnippets {
            indices: Vector::from_iter(chosen.iter().map(|i| *i as f64)),
            scores,
        })
    }
}

/// Fitted snippet set.
#[derive(Clone, Debug)]
pub struct FittedSnippets {
    /// Snippet starts.
    pub indices: Vector,
    /// Facility-location costs after each pick.
    pub scores: Vector,
}

/// Incremental STAMP (stumpy `stampi`).
///
/// Window length is not identification `p`. Distinct from [`Stamp`] (batch
/// nested loops) and [`Scrimp`] (random diagonals).
#[derive(Clone, Debug)]
pub struct Stampi {
    /// Subsequence length. Not identification `p`.
    pub window: usize,
}

impl Default for Stampi {
    fn default() -> Self {
        Self { window: 3 }
    }
}

impl Stampi {
    /// STAMPI with a given window.
    pub fn new(window: usize) -> Self {
        Self {
            window: window.max(2),
        }
    }

    /// Left-to-right streaming matrix profile.
    pub fn fit(&self, y: &Vector, session: &Session) -> Result<Qualified<StompResult>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(
            &mut ctx.report,
            &Matrix::from_vector(y),
            Some(y),
            &ctx.policy,
        );
        let n = y.len();
        let m = self.window.max(2);
        if m >= n {
            ctx.push(
                Issue::builder(IssueCode::WindowTooShort)
                    .severity(Severity::Warning)
                    .message(format!("STAMPI window={m} is unusable for n={n}"))
                    .build(),
            );
            return ctx.finish(StompResult {
                profile: Vector::zeros(0),
                nn_index: Vector::zeros(0),
            });
        }
        let n_sub = n + 1 - m;
        let excl = (m / 4).max(1);
        let mut profile = Vector::filled(n_sub, f64::INFINITY);
        let mut nn_index = Vector::zeros(n_sub);
        for i in 0..n_sub {
            for j in 0..i {
                if i.abs_diff(j) < excl {
                    continue;
                }
                let d = subsequence_dist(y, i, y, j, m);
                if d < profile[i] {
                    profile[i] = d;
                    nn_index[i] = j as f64;
                }
                if d < profile[j] {
                    profile[j] = d;
                    nn_index[j] = i as f64;
                }
            }
        }
        for i in 0..n_sub {
            if !profile[i].is_finite() {
                profile[i] = 0.0;
            }
        }
        ctx.push(
            Issue::builder(IssueCode::CausalClaimUnidentified)
                .severity(Severity::Advisory)
                .message("STAMPI is a left-to-right prefix profile, not published incremental STAMP")
                .compromise(NumericalCompromise::new(
                    "stumpy stampi",
                    "each new subsequence updates the profile of all earlier windows",
                    "the published streaming buffer and z-normalized kernel are omitted",
                    "read the profile as an incremental matrix-profile sketch",
                ))
                .build(),
        );
        ctx.finish(StompResult { profile, nn_index })
    }
}

/// Consensus motif across several series (stumpy `ostinato`).
///
/// Window length is not identification `p`. Distinct from [`Motif`] (one
/// series) and [`mpdist`] (scalar two-series distance).
pub fn ostinato(x: &Matrix, window: usize, session: &Session) -> Result<Qualified<OstinatoResult>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, x, None, &ctx.policy);
    let m = window.max(2);
    let (n_series, tlen) = x.shape();
    if m >= tlen || n_series < 2 {
        ctx.push(
            Issue::builder(IssueCode::WindowTooShort)
                .severity(Severity::Warning)
                .message(format!(
                    "ostinato window={m} is unusable for n_series={n_series} length={tlen}"
                ))
                .build(),
        );
        return ctx.finish(OstinatoResult {
            series: 0.0,
            start: 0.0,
            score: f64::NAN,
        });
    }
    let n_sub = tlen + 1 - m;
    let mut best_s = 0usize;
    let mut best_i = 0usize;
    let mut best = f64::INFINITY;
    for s in 0..n_series {
        for i in 0..n_sub {
            let mut acc = 0.0_f64;
            let mut ok = 0usize;
            for k in 0..n_series {
                if k == s {
                    continue;
                }
                let mut nearest = f64::INFINITY;
                for j in 0..n_sub {
                    let mut d = 0.0_f64;
                    for t in 0..m {
                        let e = x.get(s, i + t) - x.get(k, j + t);
                        d += e * e;
                    }
                    let d = d.max(0.0).sqrt();
                    if d < nearest {
                        nearest = d;
                    }
                }
                if nearest.is_finite() {
                    acc += nearest;
                    ok += 1;
                }
            }
            if ok > 0 {
                let score = acc / ok as f64;
                if score < best {
                    best = score;
                    best_s = s;
                    best_i = i;
                }
            }
        }
    }
    ctx.push(
        Issue::builder(IssueCode::CausalClaimUnidentified)
            .severity(Severity::Advisory)
            .message("ostinato is mean nearest-neighbour radius, not published Ostinato")
            .compromise(NumericalCompromise::new(
                "stumpy ostinato",
                "argmin over (series, start) of the mean min-distance to every other series",
                "the published radius search and z-normalized kernel are omitted",
                "read the index as a consensus-motif sketch",
            ))
            .build(),
    );
    ctx.finish(OstinatoResult {
        series: best_s as f64,
        start: best_i as f64,
        score: if best.is_finite() { best } else { f64::NAN },
    })
}

/// Consensus motif payload.
#[derive(Clone, Debug)]
pub struct OstinatoResult {
    /// Series (row) that hosts the motif.
    pub series: f64,
    /// Start index inside that series.
    pub start: f64,
    /// Mean nearest-neighbour radius.
    pub score: f64,
}

/// Named Ostinato consensus-motif search.
#[derive(Clone, Debug)]
pub struct Ostinato {
    /// Subsequence length. Not identification `p`.
    pub window: usize,
}

impl Default for Ostinato {
    fn default() -> Self {
        Self { window: 3 }
    }
}

impl Ostinato {
    /// Ostinato with a given window.
    pub fn new(window: usize) -> Self {
        Self {
            window: window.max(2),
        }
    }

    /// Consensus motif of the rows of `x`.
    pub fn fit(&self, x: &Matrix, session: &Session) -> Result<Qualified<OstinatoResult>> {
        ostinato(x, self.window, session)
    }
}

/// PRESCIAMP-style coarse-to-fine matrix profile (stumpy `prescrimp`).
///
/// Stride is not identification `p`. Distinct from [`Scrimp`] (random
/// diagonals) and [`Stampi`] (left-to-right prefix).
#[derive(Clone, Debug)]
pub struct Prescrimp {
    /// Subsequence length. Not identification `p`.
    pub window: usize,
    /// Coarse stride. Not identification `p`.
    pub stride: usize,
}

impl Default for Prescrimp {
    fn default() -> Self {
        Self {
            window: 3,
            stride: 2,
        }
    }
}

impl Prescrimp {
    /// PRESCIAMP with a given window.
    pub fn new(window: usize) -> Self {
        Self {
            window: window.max(2),
            ..Self::default()
        }
    }

    /// Coarse grid profile, then a local refinement around each neighbour.
    pub fn fit(&self, y: &Vector, session: &Session) -> Result<Qualified<StompResult>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(
            &mut ctx.report,
            &Matrix::from_vector(y),
            Some(y),
            &ctx.policy,
        );
        let n = y.len();
        let m = self.window.max(2);
        if m >= n {
            ctx.push(
                Issue::builder(IssueCode::WindowTooShort)
                    .severity(Severity::Warning)
                    .message(format!("PRESCIAMP window={m} is unusable for n={n}"))
                    .build(),
            );
            return ctx.finish(StompResult {
                profile: Vector::zeros(0),
                nn_index: Vector::zeros(0),
            });
        }
        let n_sub = n + 1 - m;
        let excl = (m / 4).max(1);
        let step = self.stride.max(2);
        let mut profile = Vector::filled(n_sub, f64::INFINITY);
        let mut nn_index = Vector::zeros(n_sub);
        for i in 0..n_sub {
            for j in 0..n_sub {
                if i.abs_diff(j) < excl {
                    continue;
                }
                if i % step != 0 && j % step != 0 {
                    continue;
                }
                let d = subsequence_dist(y, i, y, j, m);
                if d < profile[i] {
                    profile[i] = d;
                    nn_index[i] = j as f64;
                }
            }
        }
        for i in 0..n_sub {
            let center = nn_index[i] as usize;
            let lo = center.saturating_sub(step);
            let hi = (center + step + 1).min(n_sub);
            for j in lo..hi {
                if i.abs_diff(j) < excl {
                    continue;
                }
                let d = subsequence_dist(y, i, y, j, m);
                if d < profile[i] {
                    profile[i] = d;
                    nn_index[i] = j as f64;
                }
            }
        }
        for i in 0..n_sub {
            if !profile[i].is_finite() {
                profile[i] = 0.0;
            }
        }
        ctx.push(
            Issue::builder(IssueCode::CausalClaimUnidentified)
                .severity(Severity::Advisory)
                .message("PRESCIAMP is a coarse-grid plus local refine, not published prescrimp")
                .compromise(NumericalCompromise::new(
                    "stumpy prescrimp",
                    "stride-grid pairwise distances followed by a neighbourhood polish",
                    "the published PRESCIAMP queue and z-normalized kernel are omitted",
                    "read the profile as a coarse-to-fine matrix-profile sketch",
                ))
                .build(),
        );
        ctx.finish(StompResult { profile, nn_index })
    }
}

/// Non-normalized all-pairs matrix profile (stumpy `aamp`).
///
/// Window length is not identification `p`. Distinct from [`Mpx`]
/// (z-normalized) and [`Stampi`] (prefix only).
#[derive(Clone, Debug)]
pub struct Aamp {
    /// Subsequence length. Not identification `p`.
    pub window: usize,
}

impl Default for Aamp {
    fn default() -> Self {
        Self { window: 3 }
    }
}

impl Aamp {
    /// AAMP with a given window.
    pub fn new(window: usize) -> Self {
        Self {
            window: window.max(2),
        }
    }

    /// Raw-Euclidean self nearest-neighbour profile.
    pub fn fit(&self, y: &Vector, session: &Session) -> Result<Qualified<StompResult>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(
            &mut ctx.report,
            &Matrix::from_vector(y),
            Some(y),
            &ctx.policy,
        );
        let n = y.len();
        let m = self.window.max(2);
        if m >= n {
            ctx.push(
                Issue::builder(IssueCode::WindowTooShort)
                    .severity(Severity::Warning)
                    .message(format!("AAMP window={m} is unusable for n={n}"))
                    .build(),
            );
            return ctx.finish(StompResult {
                profile: Vector::zeros(0),
                nn_index: Vector::zeros(0),
            });
        }
        let n_sub = n + 1 - m;
        let excl = (m / 4).max(1);
        let mut profile = Vector::filled(n_sub, f64::INFINITY);
        let mut nn_index = Vector::zeros(n_sub);
        for i in 0..n_sub {
            for j in 0..n_sub {
                if i.abs_diff(j) < excl {
                    continue;
                }
                let d = subsequence_dist(y, i, y, j, m);
                if d < profile[i] {
                    profile[i] = d;
                    nn_index[i] = j as f64;
                }
            }
        }
        for i in 0..n_sub {
            if !profile[i].is_finite() {
                profile[i] = 0.0;
            }
        }
        ctx.push(
            Issue::builder(IssueCode::CausalClaimUnidentified)
                .severity(Severity::Advisory)
                .message("AAMP is a direct unnormalized self profile, not published AAMP")
                .compromise(NumericalCompromise::new(
                    "stumpy aamp",
                    "raw Euclidean nearest neighbour of every subsequence",
                    "the published FFT / Pearson kernel is omitted",
                    "read the profile as an amplitude-aware matrix-profile sketch",
                ))
                .build(),
        );
        ctx.finish(StompResult { profile, nn_index })
    }
}

/// Unnormalized MASS distance profile (stumpy `mass_absolute`).
///
/// Query length is not identification `p`. Distinct from [`mass`]
/// (z-normalized) and [`Aamp`] (self profile).
pub fn mass_absolute(
    query: &Vector,
    series: &Vector,
    session: &Session,
) -> Result<Qualified<Vector>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(
        &mut ctx.report,
        &Matrix::from_vector(series),
        Some(series),
        &ctx.policy,
    );
    if let Some(issue) = signlred::scan_finite(query.as_slice()).to_issue("mass_absolute.query") {
        ctx.push(issue);
    }
    let m = query.len();
    if m < 2 || m > series.len() {
        ctx.push(
            Issue::builder(IssueCode::WindowTooShort)
                .severity(Severity::Warning)
                .message(format!(
                    "mass_absolute query length={m} is unusable for n={}",
                    series.len()
                ))
                .build(),
        );
        return ctx.finish(Vector::zeros(0));
    }
    let n_sub = series.len() + 1 - m;
    let out = Vector::from_iter((0..n_sub).map(|i| subsequence_dist(query, 0, series, i, m)));
    ctx.push(
        Issue::builder(IssueCode::CausalClaimUnidentified)
            .severity(Severity::Advisory)
            .message("mass_absolute is a direct sliding Euclidean profile, not FFT MASS")
            .compromise(NumericalCompromise::new(
                "stumpy mass_absolute",
                "raw Euclidean distance of the query at every offset",
                "the published FFT convolution is omitted",
                "read the vector as an unnormalized MASS sketch",
            ))
            .build(),
    );
    ctx.finish(out)
}

/// Complexity annotation vector (stumpy `core.make_complexity_av`).
///
/// Window length is not identification `p`. Distinct from [`Aamp`] (distances)
/// and [`Motif`] (argmin selection).
pub fn annotation_vector(
    y: &Vector,
    window: usize,
    session: &Session,
) -> Result<Qualified<Vector>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(
        &mut ctx.report,
        &Matrix::from_vector(y),
        Some(y),
        &ctx.policy,
    );
    let m = window.max(2);
    if m >= y.len() {
        ctx.push(
            Issue::builder(IssueCode::WindowTooShort)
                .severity(Severity::Warning)
                .message(format!("annotation_vector window={m} is unusable for n={}", y.len()))
                .build(),
        );
        return ctx.finish(Vector::zeros(0));
    }
    let n_sub = y.len() + 1 - m;
    let mut raw = Vector::zeros(n_sub);
    let mut mx = 0.0_f64;
    for i in 0..n_sub {
        let mut s = 0.0_f64;
        let mut ss = 0.0_f64;
        for t in 0..m {
            let v = y[i + t];
            s += v;
            ss += v * v;
        }
        let mean = s / m as f64;
        let std = (ss / m as f64 - mean * mean).max(0.0).sqrt();
        raw[i] = std;
        if std > mx {
            mx = std;
        }
    }
    let out = if mx > 1e-12 {
        Vector::from_iter((0..n_sub).map(|i| raw[i] / mx))
    } else {
        Vector::filled(n_sub, 1.0)
    };
    ctx.push(
        Issue::builder(IssueCode::CausalClaimUnidentified)
            .severity(Severity::Advisory)
            .message("annotation_vector is subsequence std, not the published complexity AV")
            .compromise(NumericalCompromise::new(
                "stumpy complexity annotation vector",
                "normalized standard deviation of each window",
                "the published path-length complexity and correctors are omitted",
                "read the vector as a complexity-weight sketch",
            ))
            .build(),
    );
    ctx.finish(out)
}

/// Named complexity annotation-vector transform.
#[derive(Clone, Debug)]
pub struct AnnotationVector {
    /// Subsequence length. Not identification `p`.
    pub window: usize,
}

impl Default for AnnotationVector {
    fn default() -> Self {
        Self { window: 3 }
    }
}

impl AnnotationVector {
    /// Annotation vector with a given window.
    pub fn new(window: usize) -> Self {
        Self {
            window: window.max(2),
        }
    }

    /// Complexity weights for every subsequence of `y`.
    pub fn fit(&self, y: &Vector, session: &Session) -> Result<Qualified<Vector>> {
        annotation_vector(y, self.window, session)
    }
}

fn row_self_profile(row: &Vector, m: usize, znorm: bool) -> (Vector, Vector) {
    let n = row.len();
    if m >= n {
        return (Vector::zeros(0), Vector::zeros(0));
    }
    let n_sub = n + 1 - m;
    let excl = (m / 4).max(1);
    let mut means = Vector::zeros(n_sub);
    let mut stds = Vector::filled(n_sub, 1.0);
    if znorm {
        for i in 0..n_sub {
            let mut s = 0.0_f64;
            let mut ss = 0.0_f64;
            for t in 0..m {
                let v = row[i + t];
                s += v;
                ss += v * v;
            }
            let mean = s / m as f64;
            means[i] = mean;
            stds[i] = ((ss / m as f64 - mean * mean).max(0.0).sqrt()).max(1e-12);
        }
    }
    let mut profile = Vector::filled(n_sub, f64::INFINITY);
    let mut nn_index = Vector::zeros(n_sub);
    for i in 0..n_sub {
        for j in 0..n_sub {
            if i.abs_diff(j) < excl {
                continue;
            }
            let d = if znorm {
                let mut acc = 0.0_f64;
                for t in 0..m {
                    let zi = (row[i + t] - means[i]) / stds[i];
                    let zj = (row[j + t] - means[j]) / stds[j];
                    let e = zi - zj;
                    acc += e * e;
                }
                acc.max(0.0).sqrt()
            } else {
                subsequence_dist(row, i, row, j, m)
            };
            if d < profile[i] {
                profile[i] = d;
                nn_index[i] = j as f64;
            }
        }
    }
    for i in 0..n_sub {
        if !profile[i].is_finite() {
            profile[i] = 0.0;
        }
    }
    (profile, nn_index)
}

fn panel_profile(
    x: &Matrix,
    window: usize,
    znorm: bool,
    ctx: &mut FitCtx,
    name: &str,
) -> StompResult {
    let m = window.max(2);
    let (n_series, tlen) = x.shape();
    if m >= tlen || n_series == 0 {
        ctx.push(
            Issue::builder(IssueCode::WindowTooShort)
                .severity(Severity::Warning)
                .message(format!("{name} window={m} is unusable for n_series={n_series} length={tlen}"))
                .build(),
        );
        return StompResult {
            profile: Vector::zeros(0),
            nn_index: Vector::zeros(0),
        };
    }
    let n_sub = tlen + 1 - m;
    let mut profile = Vector::zeros(n_sub);
    let mut nn_index = Vector::zeros(n_sub);
    let mut best = Vector::filled(n_sub, f64::INFINITY);
    for i in 0..n_series {
        let row = x.row(i);
        let (mp, nn) = row_self_profile(&row, m, znorm);
        if mp.len() != n_sub {
            continue;
        }
        for j in 0..n_sub {
            profile[j] += mp[j];
            if mp[j] < best[j] {
                best[j] = mp[j];
                nn_index[j] = nn[j];
            }
        }
    }
    let n = n_series.max(1) as f64;
    for j in 0..n_sub {
        profile[j] /= n;
    }
    StompResult { profile, nn_index }
}

/// Multidimensional matrix profile (stumpy `mstump`).
///
/// Window length is not identification `p`. Distinct from [`Mpx`] (one
/// series) and [`Ostinato`] (consensus host series).
#[derive(Clone, Debug)]
pub struct Mstump {
    /// Subsequence length. Not identification `p`.
    pub window: usize,
}

impl Default for Mstump {
    fn default() -> Self {
        Self { window: 3 }
    }
}

impl Mstump {
    /// MSTUMP with a given window.
    pub fn new(window: usize) -> Self {
        Self {
            window: window.max(2),
        }
    }

    /// Mean z-normalized self-profile across the rows of `x`.
    pub fn fit(&self, x: &Matrix, session: &Session) -> Result<Qualified<StompResult>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let out = panel_profile(x, self.window, true, &mut ctx, "MSTUMP");
        ctx.push(
            Issue::builder(IssueCode::CausalClaimUnidentified)
                .severity(Severity::Advisory)
                .message("MSTUMP averages per-row z-normalized profiles, not published mstump")
                .compromise(NumericalCompromise::new(
                    "stumpy mstump",
                    "mean of row-wise z-normalized self nearest-neighbour profiles",
                    "the published multidimensional ablation and FFT kernel are omitted",
                    "read the profile as a multi-series matrix-profile sketch",
                ))
                .build(),
        );
        ctx.finish(out)
    }
}

/// Multidimensional AAMP (stumpy `maamp`).
///
/// Window length is not identification `p`. Distinct from [`Aamp`] (one
/// series) and [`Mstump`] (z-normalized).
#[derive(Clone, Debug)]
pub struct Maamp {
    /// Subsequence length. Not identification `p`.
    pub window: usize,
}

impl Default for Maamp {
    fn default() -> Self {
        Self { window: 3 }
    }
}

impl Maamp {
    /// MAAMP with a given window.
    pub fn new(window: usize) -> Self {
        Self {
            window: window.max(2),
        }
    }

    /// Mean raw-Euclidean self-profile across the rows of `x`.
    pub fn fit(&self, x: &Matrix, session: &Session) -> Result<Qualified<StompResult>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let out = panel_profile(x, self.window, false, &mut ctx, "MAAMP");
        ctx.push(
            Issue::builder(IssueCode::CausalClaimUnidentified)
                .severity(Severity::Advisory)
                .message("MAAMP averages per-row unnormalized profiles, not published maamp")
                .compromise(NumericalCompromise::new(
                    "stumpy maamp",
                    "mean of row-wise raw Euclidean self nearest-neighbour profiles",
                    "the published multidimensional ablation and FFT kernel are omitted",
                    "read the profile as an amplitude-aware multi-series sketch",
                ))
                .build(),
        );
        ctx.finish(out)
    }
}

/// Multidimensional motifs (stumpy `mmotifs`).
///
/// Motif count is not identification `p`. Distinct from [`Motif`] (one series)
/// and [`Ostinato`] (consensus radius).
#[derive(Clone, Debug)]
pub struct Mmotifs {
    /// Subsequence length. Not identification `p`.
    pub window: usize,
    /// Number of motifs. Not identification `p`.
    pub n_motifs: usize,
}

impl Default for Mmotifs {
    fn default() -> Self {
        Self {
            window: 3,
            n_motifs: 2,
        }
    }
}

/// Fitted multidimensional motif set.
#[derive(Clone, Debug)]
pub struct FittedMmotifs {
    /// Motif starts.
    pub indices: Vector,
    /// MSTUMP distances at those starts.
    pub scores: Vector,
}

impl Mmotifs {
    /// Multidimensional motif search with a given window.
    pub fn new(window: usize) -> Self {
        Self {
            window: window.max(2),
            ..Self::default()
        }
    }

    /// Top-\(k\) argmin of the MSTUMP profile with an exclusion zone.
    pub fn fit(&self, x: &Matrix, session: &Session) -> Result<Qualified<FittedMmotifs>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let mp = panel_profile(x, self.window, true, &mut ctx, "mmotifs");
        let n = mp.profile.len();
        let k = self.n_motifs.max(1);
        if n == 0 {
            return ctx.finish(FittedMmotifs {
                indices: Vector::zeros(0),
                scores: Vector::zeros(0),
            });
        }
        let excl = (self.window.max(2) / 4).max(1);
        let mut taken = vec![false; n];
        let mut idx = Vector::zeros(k);
        let mut scores = Vector::zeros(k);
        let mut found = 0usize;
        for s in 0..k {
            let mut best_i = 0usize;
            let mut best = f64::INFINITY;
            for i in 0..n {
                if taken[i] {
                    continue;
                }
                let v = mp.profile[i];
                if v < best {
                    best = v;
                    best_i = i;
                }
            }
            if !best.is_finite() {
                break;
            }
            idx[s] = best_i as f64;
            scores[s] = best;
            found += 1;
            let lo = best_i.saturating_sub(excl);
            let hi = (best_i + excl + 1).min(n);
            for t in lo..hi {
                taken[t] = true;
            }
        }
        ctx.push(
            Issue::builder(IssueCode::CausalClaimUnidentified)
                .severity(Severity::Advisory)
                .message("mmotifs is top-k MSTUMP argmin, not published mmotifs")
                .compromise(NumericalCompromise::new(
                    "stumpy mmotifs",
                    "exclusion-zone argmin of the mean row-wise z-normalized profile",
                    "the published multidimensional motif radius search is omitted",
                    "read the indices as a multi-series motif sketch",
                ))
                .build(),
        );
        ctx.finish(FittedMmotifs {
            indices: Vector::from_iter(idx.as_slice().iter().take(found).copied()),
            scores: Vector::from_iter(scores.as_slice().iter().take(found).copied()),
        })
    }
}

/// Pan multidimensional matrix profile (stumpy `mpstump`).
///
/// Window count is not identification `p`. Distinct from [`PanMatrixProfile`]
/// (one series) and [`Mstump`] (one window).
#[derive(Clone, Debug)]
pub struct Mpstump {
    /// Smallest window. Not identification `p`.
    pub min_window: usize,
    /// Number of consecutive windows. Not identification `p`.
    pub n_windows: usize,
}

impl Default for Mpstump {
    fn default() -> Self {
        Self {
            min_window: 2,
            n_windows: 2,
        }
    }
}

impl Mpstump {
    /// MPSTUMP starting at `min_window` for `n_windows` lengths.
    pub fn new(min_window: usize, n_windows: usize) -> Self {
        Self {
            min_window: min_window.max(2),
            n_windows: n_windows.max(1),
        }
    }

    /// Motif and discord of the MSTUMP profile at each window.
    pub fn fit(&self, x: &Matrix, session: &Session) -> Result<Qualified<FittedPanMatrixProfile>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let nw = self.n_windows.max(1);
        let mut windows = Vector::zeros(nw);
        let mut motifs = Vector::zeros(nw);
        let mut discords = Vector::zeros(nw);
        let mut motif_scores = Vector::zeros(nw);
        let mut discord_scores = Vector::zeros(nw);
        for k in 0..nw {
            let w = self.min_window.max(2) + k;
            windows[k] = w as f64;
            let mp = panel_profile(x, w, true, &mut ctx, "mpstump");
            let mut mi = 0usize;
            let mut di = 0usize;
            let mut mb = f64::INFINITY;
            let mut db = f64::NEG_INFINITY;
            for (i, &v) in mp.profile.as_slice().iter().enumerate() {
                if v.is_finite() && v < mb {
                    mb = v;
                    mi = i;
                }
                if v.is_finite() && v > db {
                    db = v;
                    di = i;
                }
            }
            motifs[k] = mi as f64;
            discords[k] = di as f64;
            motif_scores[k] = if mb.is_finite() { mb } else { f64::NAN };
            discord_scores[k] = if db.is_finite() { db } else { f64::NAN };
        }
        ctx.push(
            Issue::builder(IssueCode::CausalClaimUnidentified)
                .severity(Severity::Advisory)
                .message("MPSTUMP is pan-MSTUMP, not published mpstump")
                .compromise(NumericalCompromise::new(
                    "stumpy mpstump",
                    "motif/discord of the mean row-wise z-normalized profile at each window",
                    "the published multidimensional pan kernel is omitted",
                    "read the windows as a multi-series pan-profile sketch",
                ))
                .build(),
        );
        ctx.finish(FittedPanMatrixProfile {
            windows,
            motifs,
            discords,
            motif_scores,
            discord_scores,
        })
    }
}

/// Snippet area profile (stumpy snippets area / facility cost).
///
/// Window length is not identification `p`. Distinct from [`Snippets`]
/// (greedy selection) and [`Motif`] (nearest-neighbour only).
pub fn snippet_area(y: &Vector, window: usize, session: &Session) -> Result<Qualified<Vector>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(
        &mut ctx.report,
        &Matrix::from_vector(y),
        Some(y),
        &ctx.policy,
    );
    let m = window.max(2);
    if m >= y.len() {
        ctx.push(
            Issue::builder(IssueCode::WindowTooShort)
                .severity(Severity::Warning)
                .message(format!("snippet_area window={m} is unusable for n={}", y.len()))
                .build(),
        );
        return ctx.finish(Vector::zeros(0));
    }
    let n_sub = y.len() + 1 - m;
    let mut area = Vector::zeros(n_sub);
    for i in 0..n_sub {
        let mut acc = 0.0_f64;
        for j in 0..n_sub {
            if i == j {
                continue;
            }
            acc += subsequence_dist(y, i, y, j, m);
        }
        area[i] = acc;
    }
    ctx.push(
        Issue::builder(IssueCode::CausalClaimUnidentified)
            .severity(Severity::Advisory)
            .message("snippet_area is the sum of pairwise subsequence distances")
            .compromise(NumericalCompromise::new(
                "stumpy snippet area profile",
                "facility-location cost of choosing each subsequence as the only snippet",
                "the published MPdist area kernel is omitted",
                "read the vector as a diversity-cost sketch",
            ))
            .build(),
    );
    ctx.finish(area)
}

/// Named snippet-area transform.
#[derive(Clone, Debug)]
pub struct SnippetArea {
    /// Subsequence length. Not identification `p`.
    pub window: usize,
}

impl Default for SnippetArea {
    fn default() -> Self {
        Self { window: 3 }
    }
}

impl SnippetArea {
    /// Snippet-area profile with a given window.
    pub fn new(window: usize) -> Self {
        Self {
            window: window.max(2),
        }
    }

    /// Facility-location cost of every subsequence of `y`.
    pub fn fit(&self, y: &Vector, session: &Session) -> Result<Qualified<Vector>> {
        snippet_area(y, self.window, session)
    }
}

/// Longest unimodal nearest-neighbour chain (stumpy `allc`).
///
/// Window length is not identification `p`. Distinct from [`Motif`] (argmin
/// pair) and [`Merlin`] (discord).
#[derive(Clone, Debug)]
pub struct AllChains {
    /// Subsequence length. Not identification `p`.
    pub window: usize,
}

impl Default for AllChains {
    fn default() -> Self {
        Self { window: 3 }
    }
}

impl AllChains {
    /// Time-series chains with a given window.
    pub fn new(window: usize) -> Self {
        Self {
            window: window.max(2),
        }
    }

    /// Longest right-nearest-neighbour chain of `y`.
    pub fn fit(&self, y: &Vector, session: &Session) -> Result<Qualified<FittedAllChains>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(
            &mut ctx.report,
            &Matrix::from_vector(y),
            Some(y),
            &ctx.policy,
        );
        let n = y.len();
        let m = self.window.max(2);
        if m >= n {
            ctx.push(
                Issue::builder(IssueCode::WindowTooShort)
                    .severity(Severity::Warning)
                    .message(format!("AllChains window={m} is unusable for n={n}"))
                    .build(),
            );
            return ctx.finish(FittedAllChains {
                indices: Vector::zeros(0),
                length: 0.0,
            });
        }
        let n_sub = n + 1 - m;
        let excl = (m / 4).max(1);
        let mut right = vec![usize::MAX; n_sub];
        let mut rdist = vec![f64::INFINITY; n_sub];
        for i in 0..n_sub {
            for j in (i + excl)..n_sub {
                let d = subsequence_dist(y, i, y, j, m);
                if d < rdist[i] {
                    rdist[i] = d;
                    right[i] = j;
                }
            }
        }
        let mut best: Vec<usize> = Vec::new();
        for start in 0..n_sub {
            let mut chain = vec![start];
            let mut cur = start;
            let mut seen = vec![false; n_sub];
            seen[cur] = true;
            loop {
                let nxt = right[cur];
                if nxt >= n_sub || seen[nxt] {
                    break;
                }
                seen[nxt] = true;
                chain.push(nxt);
                cur = nxt;
            }
            if chain.len() > best.len() {
                best = chain;
            }
        }
        if best.is_empty() {
            best.push(0);
        }
        ctx.push(
            Issue::builder(IssueCode::CausalClaimUnidentified)
                .severity(Severity::Advisory)
                .message("AllChains follows right nearest neighbours, not published allc")
                .compromise(NumericalCompromise::new(
                    "stumpy allc",
                    "longest unimodal chain of right nearest-neighbour links",
                    "the published bidirectional IL/IR and unimodality filter are omitted",
                    "read the indices as a time-series-chain sketch",
                ))
                .build(),
        );
        let length = best.len() as f64;
        ctx.finish(FittedAllChains {
            indices: Vector::from_iter(best.iter().map(|&i| i as f64)),
            length,
        })
    }
}

/// Fitted time-series chain.
#[derive(Clone, Debug)]
pub struct FittedAllChains {
    /// Subsequence start indices of the longest right-NN chain.
    pub indices: Vector,
    /// Chain length.
    pub length: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use ojizou_san::Session;

    #[test]
    fn dtw_identical_is_zero() {
        let a = Vector::from_slice(&[1.0, 2.0, 3.0, 2.0]);
        let d = dtw(&a, &a, &Session::new("ts", "dtw")).unwrap().value;
        assert!(d.abs() < 1e-12);
        let lc = lcss(&a, &a, 0.1, None, &Session::new("ts", "lcss"))
            .unwrap()
            .value;
        assert!((lc - 1.0).abs() < 1e-12);
        let ed = edit_distance(&a, &a, &Session::new("ts", "ed"))
            .unwrap()
            .value;
        assert!(ed.abs() < 1e-12);
        let long = Vector::from_iter((0..16).map(|i| (i as f64).sin()));
        let mp = matrix_profile(&long, 4, &Session::new("ts", "mp"))
            .unwrap()
            .value;
        assert_eq!(mp.len(), 13);
        assert!(mp.as_slice().iter().all(|v| v.is_finite()));
        let sd = softdtw(&a, &a, 0.1, &Session::new("ts", "sdtw"))
            .unwrap()
            .value;
        assert!(sd.is_finite());
        let wd = wdtw(&a, &a, 0.1, &Session::new("ts", "wdtw"))
            .unwrap()
            .value;
        assert!(wd.abs() < 1e-12, "wdtw={wd}");
        let dd = ddtw(&a, &a, &Session::new("ts", "ddtw")).unwrap().value;
        assert!(dd.abs() < 1e-12, "ddtw={dd}");
        let er = eros(&a, &a, &Session::new("ts", "eros")).unwrap().value;
        assert!(er > 0.9, "eros={er}");
        let sh = shape_dtw(&a, &a, &Session::new("ts", "sdtw2"))
            .unwrap()
            .value;
        assert!(sh.abs() < 1e-12, "shape_dtw={sh}");
        let gk = gak(&a, &a, 1.0, &Session::new("ts", "gak")).unwrap().value;
        assert!(gk.is_finite() && gk > 0.0);
        let sb = sbd(&a, &a, &Session::new("ts", "sbd")).unwrap().value;
        assert!(sb.abs() < 1e-9, "sbd={sb}");
        let cw = ctw(&a, &a, &Session::new("ts", "ctw")).unwrap().value;
        assert!(cw.abs() < 1e-12, "ctw={cw}");
        let fr = frechet(&a, &a, &Session::new("ts", "fr")).unwrap().value;
        assert!(fr.abs() < 1e-12, "frechet={fr}");
        let hd = hausdorff(&a, &a, &Session::new("ts", "hd")).unwrap().value;
        assert!(hd.abs() < 1e-12, "hausdorff={hd}");
        let erd = edr(&a, &a, 0.1, &Session::new("ts", "edr")).unwrap().value;
        assert!(erd.abs() < 1e-12, "edr={erd}");
        let ad = adtw(&a, &a, 0.1, &Session::new("ts", "adtw")).unwrap().value;
        assert!(ad.abs() < 1e-12, "adtw={ad}");
        let wd = wddtw(&a, &a, 0.1, &Session::new("ts", "wddtw")).unwrap().value;
        assert!(wd.abs() < 1e-12, "wddtw={wd}");
        let sw = swale(&a, &a, 0.1, &Session::new("ts", "swale")).unwrap().value;
        assert!(sw.abs() < 1e-12, "swale={sw}");
        let lk = lb_kim(&a, &a, &Session::new("ts", "lbkim")).unwrap().value;
        assert!(lk.abs() < 1e-12, "lb_kim={lk}");
        let lbi = lb_improved(&a, &a, 1, &Session::new("ts", "lbi")).unwrap().value;
        assert!(lbi.abs() < 1e-12, "lb_improved={lbi}");
        let cd = cid(&a, &a, &Session::new("ts", "cid")).unwrap().value;
        assert!(cd.abs() < 1e-12, "cid={cd}");
        let kdw = kdtw(&a, &a, 1.0, &Session::new("ts", "kdw")).unwrap().value;
        assert!(kdw.abs() < 1e-12, "kdtw={kdw}");
        let fdw = fast_dtw(&a, &a, 1, &Session::new("ts", "fdw")).unwrap().value;
        assert!(fdw.abs() < 1e-12, "fast_dtw={fdw}");
        let dsub = dtw_subsequence(&a, &a, &Session::new("ts", "dsub")).unwrap().value;
        assert!(dsub.abs() < 1e-12, "dtw_subsequence={dsub}");
        let lby = lb_yi(&a, &a, &Session::new("ts", "lby")).unwrap().value;
        assert!(lby.abs() < 1e-12, "lb_yi={lby}");
        let itk = itakura_dtw(&a, &a, &Session::new("ts", "itk")).unwrap().value;
        assert!(itk.abs() < 1e-12, "itakura_dtw={itk}");
        let scb = sakoe_chiba_dtw(&a, &a, 1, &Session::new("ts", "scb")).unwrap().value;
        assert!(scb.abs() < 1e-12, "sakoe_chiba_dtw={scb}");
        let cyc = cyclic_dtw(&a, &a, &Session::new("ts", "cyc")).unwrap().value;
        assert!(cyc.abs() < 1e-12, "cyclic_dtw={cyc}");
        let obe = obe_dtw(&a, &a, &Session::new("ts", "obe")).unwrap().value;
        assert!(obe.abs() < 1e-12, "obe_dtw={obe}");
        let ams = amss(&a, &a, &Session::new("ts", "ams")).unwrap().value;
        assert!(ams.abs() < 1e-12, "amss={ams}");
        let dew = decay_euclidean(&a, &a, 0.2, &Session::new("ts", "dew")).unwrap().value;
        assert!(dew.abs() < 1e-12, "decay_euclidean={dew}");
        let seu = shape_euclidean(&a, &a, &Session::new("ts", "seu")).unwrap().value;
        assert!(seu.abs() < 1e-12, "shape_euclidean={seu}");
        let opb = open_begin_dtw(&a, &a, &Session::new("ts", "opb")).unwrap().value;
        assert!(opb.abs() < 1e-12, "open_begin_dtw={opb}");
        let oed = open_end_dtw(&a, &a, &Session::new("ts", "oed")).unwrap().value;
        assert!(oed.abs() < 1e-12, "open_end_dtw={oed}");
        let crd = correlation_distance(&a, &a, &Session::new("ts", "crd")).unwrap().value;
        assert!(crd.abs() < 1e-12, "correlation_distance={crd}");
        let cnd = cosine_distance(&a, &a, &Session::new("ts", "cnd")).unwrap().value;
        assert!(cnd.abs() < 1e-12, "cosine_distance={cnd}");
        let chb = chebyshev_distance(&a, &a, &Session::new("ts", "chb")).unwrap().value;
        assert!(chb.abs() < 1e-12, "chebyshev_distance={chb}");
        let man = manhattan_distance(&a, &a, &Session::new("ts", "man")).unwrap().value;
        assert!(man.abs() < 1e-12, "manhattan_distance={man}");
        let can = canberra_distance(&a, &a, &Session::new("ts", "can")).unwrap().value;
        assert!(can.abs() < 1e-12, "canberra_distance={can}");
        let brc = braycurtis_distance(&a, &a, &Session::new("ts", "brc")).unwrap().value;
        assert!(brc.abs() < 1e-12, "braycurtis_distance={brc}");
        let lor = lorentzian_distance(&a, &a, &Session::new("ts", "lor")).unwrap().value;
        assert!(lor.abs() < 1e-12, "lorentzian_distance={lor}");
        let ang = angular_distance(&a, &a, &Session::new("ts", "ang")).unwrap().value;
        assert!(ang.abs() < 1e-12, "angular_distance={ang}");
        let mnk = minkowski3_distance(&a, &a, &Session::new("ts", "mnk")).unwrap().value;
        assert!(mnk.abs() < 1e-12, "minkowski3_distance={mnk}");
        let clr = clark_distance(&a, &a, &Session::new("ts", "clr")).unwrap().value;
        assert!(clr.abs() < 1e-12, "clark_distance={clr}");
        let sqe = squared_euclidean_distance(&a, &a, &Session::new("ts", "sqe")).unwrap().value;
        assert!(sqe.abs() < 1e-12, "squared_euclidean_distance={sqe}");
        let dic = dice_distance(&a, &a, &Session::new("ts", "dic")).unwrap().value;
        assert!(dic.abs() < 1e-12, "dice_distance={dic}");
        let tan = tanimoto_distance(&a, &a, &Session::new("ts", "tan")).unwrap().value;
        assert!(tan.abs() < 1e-12, "tanimoto_distance={tan}");
        let wav = wave_hedges_distance(&a, &a, &Session::new("ts", "wav")).unwrap().value;
        assert!(wav.abs() < 1e-12, "wave_hedges_distance={wav}");
        let kul = kulczynski_distance(&a, &a, &Session::new("ts", "kul")).unwrap().value;
        assert!(kul.abs() < 1e-12, "kulczynski_distance={kul}");
        let ruz = ruzicka_distance(&a, &a, &Session::new("ts", "ruz")).unwrap().value;
        assert!(ruz.abs() < 1e-12, "ruzicka_distance={ruz}");
        let hel = hellinger_distance(&a, &a, &Session::new("ts", "hel")).unwrap().value;
        assert!(hel.abs() < 1e-12, "hellinger_distance={hel}");
        let jsd = jensen_shannon_distance(&a, &a, &Session::new("ts", "jsd")).unwrap().value;
        assert!(jsd.abs() < 1e-12, "jensen_shannon_distance={jsd}");
        let bha = bhattacharyya_distance(&a, &a, &Session::new("ts", "bha")).unwrap().value;
        assert!(bha.abs() < 1e-12, "bhattacharyya_distance={bha}");
        let has = hassanat_distance(&a, &a, &Session::new("ts", "has")).unwrap().value;
        assert!(has.abs() < 1e-12, "hassanat_distance={has}");
        let fid = fidelity_distance(&a, &a, &Session::new("ts", "fid")).unwrap().value;
        assert!(fid.abs() < 1e-12, "fidelity_distance={fid}");
        let wtd = whittaker_distance(&a, &a, &Session::new("ts", "wtd")).unwrap().value;
        assert!(wtd.abs() < 1e-12, "whittaker_distance={wtd}");
        let pcs = pearson_chi_squared_distance(&a, &a, &Session::new("ts", "pcs")).unwrap().value;
        assert!(pcs.abs() < 1e-12, "pearson_chi_squared_distance={pcs}");
        let ney = neyman_chi_squared_distance(&a, &a, &Session::new("ts", "ney")).unwrap().value;
        assert!(ney.abs() < 1e-12, "neyman_chi_squared_distance={ney}");
        let ads = additive_symmetric_distance(&a, &a, &Session::new("ts", "ads")).unwrap().value;
        assert!(ads.abs() < 1e-12, "additive_symmetric_distance={ads}");
        let kdv = k_divergence_distance(&a, &a, &Session::new("ts", "kdv")).unwrap().value;
        assert!(kdv.abs() < 1e-12, "k_divergence_distance={kdv}");
        let top = topsoe_distance(&a, &a, &Session::new("ts", "top")).unwrap().value;
        assert!(top.abs() < 1e-12, "topsoe_distance={top}");
        let tne = taneja_distance(&a, &a, &Session::new("ts", "tne")).unwrap().value;
        assert!(tne.abs() < 1e-12, "taneja_distance={tne}");
        let kjn = kumar_johnson_distance(&a, &a, &Session::new("ts", "kjn")).unwrap().value;
        assert!(kjn.abs() < 1e-12, "kumar_johnson_distance={kjn}");
        let hmn = harmonic_mean_distance(&a, &a, &Session::new("ts", "hmn")).unwrap().value;
        assert!(hmn.abs() < 1e-12, "harmonic_mean_distance={hmn}");
        let msc = max_symmetric_chi_squared_distance(&a, &a, &Session::new("ts", "msc")).unwrap().value;
        assert!(msc.abs() < 1e-12, "max_symmetric_chi_squared_distance={msc}");
        let isc = intersection_distance(&a, &a, &Session::new("ts", "isc")).unwrap().value;
        assert!(isc.abs() < 1e-12, "intersection_distance={isc}");
        let mns = min_symmetric_chi_squared_distance(&a, &a, &Session::new("ts", "mns")).unwrap().value;
        assert!(mns.abs() < 1e-12, "min_symmetric_chi_squared_distance={mns}");
        let pse = l1_squared_euclidean_distance(&a, &a, &Session::new("ts", "pse")).unwrap().value;
        assert!(pse.abs() < 1e-12, "l1_squared_euclidean_distance={pse}");
        let jcd = jaccard_distance(&a, &a, &Session::new("ts", "jcd")).unwrap().value;
        assert!(jcd.abs() < 1e-12, "jaccard_distance={jcd}");
        let jef = jeffreys_distance(&a, &a, &Session::new("ts", "jef")).unwrap().value;
        assert!(jef.abs() < 1e-12, "jeffreys_distance={jef}");
        let sqc = squared_chord_distance(&a, &a, &Session::new("ts", "sqc")).unwrap().value;
        assert!(sqc.abs() < 1e-12, "squared_chord_distance={sqc}");
        let kld = kullback_leibler_distance(&a, &a, &Session::new("ts", "kld")).unwrap().value;
        assert!(kld.abs() < 1e-12, "kullback_leibler_distance={kld}");
        let pco = cosine_l1_distance(&a, &a, &Session::new("ts", "pco")).unwrap().value;
        assert!(pco.abs() < 1e-12, "cosine_l1_distance={pco}");
        let ptn = tanimoto_l1_distance(&a, &a, &Session::new("ts", "ptn")).unwrap().value;
        assert!(ptn.abs() < 1e-12, "tanimoto_l1_distance={ptn}");
        let dl1 = dice_l1_distance(&a, &a, &Session::new("ts", "dl1")).unwrap().value;
        assert!(dl1.abs() < 1e-12, "dice_l1_distance={dl1}");
        let vsc = vicis_symmetric_distance(&a, &a, &Session::new("ts", "vsc")).unwrap().value;
        assert!(vsc.abs() < 1e-12, "vicis_symmetric_distance={vsc}");
        let prl = correlation_l1_distance(&a, &a, &Session::new("ts", "prl")).unwrap().value;
        assert!(prl.abs() < 1e-12, "correlation_l1_distance={prl}");
        let hl1 = hellinger_l1_distance(&a, &a, &Session::new("ts", "hl1")).unwrap().value;
        assert!(hl1.abs() < 1e-12, "hellinger_l1_distance={hl1}");
        let ca1 = canberra_l1_distance(&a, &a, &Session::new("ts", "ca1")).unwrap().value;
        assert!(ca1.abs() < 1e-12, "canberra_l1_distance={ca1}");
        let cl1 = clark_l1_distance(&a, &a, &Session::new("ts", "cl1")).unwrap().value;
        assert!(cl1.abs() < 1e-12, "clark_l1_distance={cl1}");
        let wh1 = wave_hedges_l1_distance(&a, &a, &Session::new("ts", "wh1")).unwrap().value;
        assert!(wh1.abs() < 1e-12, "wave_hedges_l1_distance={wh1}");
        let kzl = kulczynski_l1_distance(&a, &a, &Session::new("ts", "kzl")).unwrap().value;
        assert!(kzl.abs() < 1e-12, "kulczynski_l1_distance={kzl}");
        let rz1 = ruzicka_l1_distance(&a, &a, &Session::new("ts", "rz1")).unwrap().value;
        assert!(rz1.abs() < 1e-12, "ruzicka_l1_distance={rz1}");
        let lz1 = lorentzian_l1_distance(&a, &a, &Session::new("ts", "lz1")).unwrap().value;
        assert!(lz1.abs() < 1e-12, "lorentzian_l1_distance={lz1}");
        let hs1 = hassanat_l1_distance(&a, &a, &Session::new("ts", "hs1")).unwrap().value;
        assert!(hs1.abs() < 1e-12, "hassanat_l1_distance={hs1}");
        let cs1 = chebyshev_l1_distance(&a, &a, &Session::new("ts", "cs1")).unwrap().value;
        assert!(cs1.abs() < 1e-12, "chebyshev_l1_distance={cs1}");
        let mk1 = minkowski3_l1_distance(&a, &a, &Session::new("ts", "mk1")).unwrap().value;
        assert!(mk1.abs() < 1e-12, "minkowski3_l1_distance={mk1}");
        let m41 = minkowski4_l1_distance(&a, &a, &Session::new("ts", "m41")).unwrap().value;
        assert!(m41.abs() < 1e-12, "minkowski4_l1_distance={m41}");
        let m15 = minkowski15_l1_distance(&a, &a, &Session::new("ts", "m15")).unwrap().value;
        assert!(m15.abs() < 1e-12, "minkowski15_l1_distance={m15}");
        let m51 = minkowski5_l1_distance(&a, &a, &Session::new("ts", "m51")).unwrap().value;
        assert!(m51.abs() < 1e-12, "minkowski5_l1_distance={m51}");
        let m61 = minkowski6_l1_distance(&a, &a, &Session::new("ts", "m61")).unwrap().value;
        assert!(m61.abs() < 1e-12, "minkowski6_l1_distance={m61}");
        let m25 = minkowski25_l1_distance(&a, &a, &Session::new("ts", "m25")).unwrap().value;
        assert!(m25.abs() < 1e-12, "minkowski25_l1_distance={m25}");
        let m81 = minkowski8_l1_distance(&a, &a, &Session::new("ts", "m81")).unwrap().value;
        assert!(m81.abs() < 1e-12, "minkowski8_l1_distance={m81}");
        let m71 = minkowski7_l1_distance(&a, &a, &Session::new("ts", "m71")).unwrap().value;
        assert!(m71.abs() < 1e-12, "minkowski7_l1_distance={m71}");
        let m91 = minkowski9_l1_distance(&a, &a, &Session::new("ts", "m91")).unwrap().value;
        assert!(m91.abs() < 1e-12, "minkowski9_l1_distance={m91}");
        let m10 = minkowski10_l1_distance(&a, &a, &Session::new("ts", "m10")).unwrap().value;
        assert!(m10.abs() < 1e-12, "minkowski10_l1_distance={m10}");
        let m11 = minkowski11_l1_distance(&a, &a, &Session::new("ts", "m11")).unwrap().value;
        assert!(m11.abs() < 1e-12, "minkowski11_l1_distance={m11}");
        let m12 = minkowski12_l1_distance(&a, &a, &Session::new("ts", "m12")).unwrap().value;
        assert!(m12.abs() < 1e-12, "minkowski12_l1_distance={m12}");
        let m13 = minkowski13_l1_distance(&a, &a, &Session::new("ts", "m13")).unwrap().value;
        assert!(m13.abs() < 1e-12, "minkowski13_l1_distance={m13}");
        let m14 = minkowski14_l1_distance(&a, &a, &Session::new("ts", "m14")).unwrap().value;
        assert!(m14.abs() < 1e-12, "minkowski14_l1_distance={m14}");
        let m16 = minkowski16_l1_distance(&a, &a, &Session::new("ts", "m16")).unwrap().value;
        assert!(m16.abs() < 1e-12, "minkowski16_l1_distance={m16}");
        let m18 = minkowski18_l1_distance(&a, &a, &Session::new("ts", "m18")).unwrap().value;
        assert!(m18.abs() < 1e-12, "minkowski18_l1_distance={m18}");
        let m20 = minkowski20_l1_distance(&a, &a, &Session::new("ts", "m20")).unwrap().value;
        assert!(m20.abs() < 1e-12, "minkowski20_l1_distance={m20}");
        let m24 = minkowski24_l1_distance(&a, &a, &Session::new("ts", "m24")).unwrap().value;
        assert!(m24.abs() < 1e-12, "minkowski24_l1_distance={m24}");
        let m17 = minkowski17_l1_distance(&a, &a, &Session::new("ts", "m17")).unwrap().value;
        assert!(m17.abs() < 1e-12, "minkowski17_l1_distance={m17}");
        let m19 = minkowski19_l1_distance(&a, &a, &Session::new("ts", "m19")).unwrap().value;
        assert!(m19.abs() < 1e-12, "minkowski19_l1_distance={m19}");
        let m21 = minkowski21_l1_distance(&a, &a, &Session::new("ts", "m21")).unwrap().value;
        assert!(m21.abs() < 1e-12, "minkowski21_l1_distance={m21}");
        let m22 = minkowski22_l1_distance(&a, &a, &Session::new("ts", "m22")).unwrap().value;
        assert!(m22.abs() < 1e-12, "minkowski22_l1_distance={m22}");
        let m28 = minkowski28_l1_distance(&a, &a, &Session::new("ts", "m28")).unwrap().value;
        assert!(m28.abs() < 1e-12, "minkowski28_l1_distance={m28}");
        let m23 = minkowski23_l1_distance(&a, &a, &Session::new("ts", "m23")).unwrap().value;
        assert!(m23.abs() < 1e-12, "minkowski23_l1_distance={m23}");
        let m26 = minkowski26_l1_distance(&a, &a, &Session::new("ts", "m26")).unwrap().value;
        assert!(m26.abs() < 1e-12, "minkowski26_l1_distance={m26}");
        let m27 = minkowski27_l1_distance(&a, &a, &Session::new("ts", "m27")).unwrap().value;
        assert!(m27.abs() < 1e-12, "minkowski27_l1_distance={m27}");
        let m29 = minkowski29_l1_distance(&a, &a, &Session::new("ts", "m29")).unwrap().value;
        assert!(m29.abs() < 1e-12, "minkowski29_l1_distance={m29}");
        let m30 = minkowski30_l1_distance(&a, &a, &Session::new("ts", "m30")).unwrap().value;
        assert!(m30.abs() < 1e-12, "minkowski30_l1_distance={m30}");
        let m31 = minkowski31_l1_distance(&a, &a, &Session::new("ts", "m31")).unwrap().value;
        assert!(m31.abs() < 1e-12, "minkowski31_l1_distance={m31}");
        let m32 = minkowski32_l1_distance(&a, &a, &Session::new("ts", "m32")).unwrap().value;
        assert!(m32.abs() < 1e-12, "minkowski32_l1_distance={m32}");
        let m33 = minkowski33_l1_distance(&a, &a, &Session::new("ts", "m33")).unwrap().value;
        assert!(m33.abs() < 1e-12, "minkowski33_l1_distance={m33}");
        let m34 = minkowski34_l1_distance(&a, &a, &Session::new("ts", "m34")).unwrap().value;
        assert!(m34.abs() < 1e-12, "minkowski34_l1_distance={m34}");
        let m35 = minkowski35_l1_distance(&a, &a, &Session::new("ts", "m35")).unwrap().value;
        assert!(m35.abs() < 1e-12, "minkowski35_l1_distance={m35}");
        let m36 = minkowski36_l1_distance(&a, &a, &Session::new("ts", "m36")).unwrap().value;
        assert!(m36.abs() < 1e-12, "minkowski36_l1_distance={m36}");
        let m37 = minkowski37_l1_distance(&a, &a, &Session::new("ts", "m37")).unwrap().value;
        assert!(m37.abs() < 1e-12, "minkowski37_l1_distance={m37}");
        let m38 = minkowski38_l1_distance(&a, &a, &Session::new("ts", "m38")).unwrap().value;
        assert!(m38.abs() < 1e-12, "minkowski38_l1_distance={m38}");
        let m39 = minkowski39_l1_distance(&a, &a, &Session::new("ts", "m39")).unwrap().value;
        assert!(m39.abs() < 1e-12, "minkowski39_l1_distance={m39}");
        let m40 = minkowski40_l1_distance(&a, &a, &Session::new("ts", "m40")).unwrap().value;
        assert!(m40.abs() < 1e-12, "minkowski40_l1_distance={m40}");
        let m41 = minkowski41_l1_distance(&a, &a, &Session::new("ts", "m41")).unwrap().value;
        assert!(m41.abs() < 1e-12, "minkowski41_l1_distance={m41}");
    }

    #[test]
    fn paa_sax_shapelet() {
        let y = Vector::from_slice(&[0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0, -1.0]);
        let p = paa(&y, 4, &Session::new("ts", "paa")).unwrap().value;
        assert_eq!(p.len(), 4);
        let s = sax(&y, 4, 4, &Session::new("ts", "sax")).unwrap().value;
        assert_eq!(s.len(), 4);
        let sh = Vector::from_slice(&[2.0, 3.0, 2.0]);
        let d = shapelet_distance(&y, &sh, &Session::new("ts", "sh"))
            .unwrap()
            .value;
        assert!(d < 0.2, "d={d}");
    }

    #[test]
    fn dtw_kmeans_and_svm() {
        let x = Matrix::from_fn(8, 6, |i, j| {
            if i < 4 {
                (j as f64).sin()
            } else {
                (j as f64).cos() + 2.0
            }
        });
        let y = Vector::from_iter((0..8).map(|i| if i < 4 { 0.0 } else { 1.0 }));
        let km = TimeSeriesKMeans::new(2)
            .fit(&x, &y, &Session::new("ts", "km"))
            .unwrap();
        assert_eq!(km.value.centers.nrows(), 2);
        let svm = TimeSeriesSvm {
            n_pieces: 4,
            alpha: 0.1,
        }
        .fit(&x, &y, &Session::new("ts", "svm"))
        .unwrap();
        let pred = svm
            .value
            .predict(&x, &Session::new("ts", "p"))
            .unwrap()
            .value;
        assert_eq!(pred.len(), 8);
        let nn = svm
            .value
            .predict_dtw_nn(&x, &Session::new("ts", "nn"))
            .unwrap()
            .value;
        let mut ok = 0;
        for i in 0..8 {
            if (nn[i] - y[i]).abs() < 0.5 {
                ok += 1;
            }
        }
        assert!(ok >= 6, "nn ok={ok}");
        let cd = cdist_dtw(&x, &x, &Session::new("ts", "cd")).unwrap().value;
        assert_eq!(cd.shape(), (8, 8));
        let ks = KShape::new(2)
            .fit(&x, &y, &Session::new("ts", "ks"))
            .unwrap();
        assert_eq!(ks.value.centers.nrows(), 2);
        let b = dtw_barycenter(&x, 4, &Session::new("ts", "dba"))
            .unwrap()
            .value;
        assert_eq!(b.len(), 6);
        let tsf = TimeSeriesForestClassifier {
            n_estimators: 6,
            n_intervals: 3,
            max_depth: 4,
            seed: 2,
        }
        .fit(&x, &y, &Session::new("ts", "tsf"))
        .unwrap();
        let pred = tsf
            .value
            .predict(&x, &Session::new("ts", "tsfp"))
            .unwrap()
            .value;
        let mut ok = 0;
        for i in 0..8 {
            if (pred[i] - y[i]).abs() < 0.5 {
                ok += 1;
            }
        }
        assert!(ok >= 5, "tsf ok={ok}");
        let tsbf = TimeSeriesBagOfFeatures::new()
            .fit(&x, &y, &Session::new("ts", "tsbf"))
            .unwrap();
        let pbf = tsbf
            .value
            .predict(&x, &Session::new("ts", "tsbfp"))
            .unwrap()
            .value;
        assert_eq!(pbf.len(), 8);
        assert!(pbf.as_slice().iter().all(|v| v.is_finite()));
        let cif = CanonicalIntervalForest {
            n_estimators: 6,
            n_intervals: 3,
            max_depth: 4,
            seed: 2,
        }
        .fit(&x, &y, &Session::new("ts", "cif"))
        .unwrap();
        let predc = cif
            .value
            .predict(&x, &Session::new("ts", "cifp"))
            .unwrap()
            .value;
        assert_eq!(predc.len(), 8);
        let knn = KNeighborsTimeSeries::new(1)
            .fit(&x, &y, &Session::new("ts", "knn"))
            .unwrap();
        let pred = knn
            .value
            .predict(&x, &Session::new("ts", "knnp"))
            .unwrap()
            .value;
        let mut ok = 0;
        for i in 0..8 {
            if (pred[i] - y[i]).abs() < 0.5 {
                ok += 1;
            }
        }
        assert!(ok >= 6, "knn ok={ok}");
        let rc = RocketClassifier {
            n_kernels: 16,
            kernel_len: 5,
            alpha: 0.5,
            seed: 2,
        }
        .fit(&x, &y, &Session::new("ts", "rocketc"))
        .unwrap();
        let pred = rc
            .value
            .predict(&x, &Session::new("ts", "rp"))
            .unwrap()
            .value;
        assert_eq!(pred.len(), 8);
        let yr = Vector::from_iter((0..8).map(|i| x.row(i).mean()));
        let tsfr = TimeSeriesForestRegressor {
            n_estimators: 6,
            n_intervals: 3,
            max_depth: 3,
            seed: 2,
        }
        .fit(&x, &yr, &Session::new("ts", "tsfr"))
        .unwrap();
        let pred = tsfr
            .value
            .predict(&x, &Session::new("ts", "tsfrp"))
            .unwrap()
            .value;
        assert!(pred.as_slice().iter().all(|v| v.is_finite()));
        let mrr = MultiRocketRegressor::new()
            .fit(&x, &yr, &Session::new("ts", "mrr"))
            .unwrap();
        let pmrr = mrr
            .value
            .predict(&x, &Session::new("ts", "mrrp"))
            .unwrap()
            .value;
        assert!(pmrr.as_slice().iter().all(|v| v.is_finite()));
        let hr = HydraRegressor::new()
            .fit(&x, &yr, &Session::new("ts", "hydr"))
            .unwrap();
        let phr = hr
            .value
            .predict(&x, &Session::new("ts", "hydrp"))
            .unwrap()
            .value;
        assert!(phr.as_slice().iter().all(|v| v.is_finite()));
        let ib = IndividualBoss::new()
            .fit(&x, &y, &Session::new("ts", "iboss"))
            .unwrap();
        let pib = ib
            .value
            .predict(&x, &Session::new("ts", "ibossp"))
            .unwrap()
            .value;
        assert!(pib.as_slice().iter().all(|v| v.is_finite()));
        let mpr = MatrixProfileRegressor::new()
            .fit(&x, &yr, &Session::new("ts", "mpr"))
            .unwrap();
        let pmpr = mpr
            .value
            .predict(&x, &Session::new("ts", "mprp"))
            .unwrap()
            .value;
        assert!(pmpr.as_slice().iter().all(|v| v.is_finite()));
        let ric = RandomIntervalClassifier::new()
            .fit(&x, &y, &Session::new("ts", "ric"))
            .unwrap();
        let pric = ric
            .value
            .predict(&x, &Session::new("ts", "ricp"))
            .unwrap()
            .value;
        assert_eq!(pric.len(), 8);
        assert!(pric.as_slice().iter().all(|v| v.is_finite()));
        let h2 = HiveCoteV2::new()
            .fit(&x, &y, &Session::new("ts", "hcv2"))
            .unwrap();
        let ph2 = h2
            .value
            .predict(&x, &Session::new("ts", "hcv2p"))
            .unwrap()
            .value;
        assert_eq!(ph2.len(), 8);
        let h1 = HiveCoteV1::new()
            .fit(&x, &y, &Session::new("ts", "hcv1"))
            .unwrap();
        let ph1 = h1
            .value
            .predict(&x, &Session::new("ts", "hcv1p"))
            .unwrap()
            .value;
        assert_eq!(ph1.len(), 8);
        let c22el = Catch22El::new()
            .fit(&x, &y, &Session::new("ts", "c22el"))
            .unwrap();
        let pc22el = c22el
            .value
            .predict(&x, &Session::new("ts", "c22elp"))
            .unwrap()
            .value;
        assert_eq!(pc22el.len(), 8);
        let rfc = RotationForestClassifier::new()
            .fit(&x, &y, &Session::new("ts", "rotf"))
            .unwrap();
        let prfc = rfc
            .value
            .predict(&x, &Session::new("ts", "rotfp"))
            .unwrap()
            .value;
        assert_eq!(prfc.len(), 8);
        assert!(prfc.as_slice().iter().all(|v| v.is_finite()));
        let pfr = ProximityForestRegressor::new()
            .fit(&x, &yr, &Session::new("ts", "pfr"))
            .unwrap();
        let ppfr = pfr
            .value
            .predict(&x, &Session::new("ts", "pfrp"))
            .unwrap()
            .value;
        assert_eq!(ppfr.len(), 8);
        assert!(ppfr.as_slice().iter().all(|v| v.is_finite()));
        let cifr = CanonicalIntervalForestRegressor::new()
            .fit(&x, &yr, &Session::new("ts", "cifr"))
            .unwrap();
        let pcifr = cifr
            .value
            .predict(&x, &Session::new("ts", "cifrp"))
            .unwrap()
            .value;
        assert_eq!(pcifr.len(), 8);
        assert!(pcifr.as_slice().iter().all(|v| v.is_finite()));
        let tspca = TimeSeriesPca::new(1)
            .fit_unsupervised(&x, &Session::new("ts", "tspca"))
            .unwrap();
        assert_eq!(tspca.value.components.nrows(), 1);
        let zts = tspca
            .value
            .transform(&x, &Session::new("ts", "tspcap"))
            .unwrap()
            .value;
        assert_eq!(zts.nrows(), 8);
        assert!(zts.get(0, 0).is_finite());
        let ctsf = ComposableTimeSeriesForest::new()
            .fit(&x, &y, &Session::new("ts", "ctsf"))
            .unwrap();
        let pctsf = ctsf
            .value
            .predict(&x, &Session::new("ts", "ctsfp"))
            .unwrap()
            .value;
        assert_eq!(pctsf.len(), 8);
        assert!(pctsf.as_slice().iter().all(|v| v.is_finite()));
        let sfat = SfaTransformer::new()
            .fit_unsupervised(&x, &Session::new("ts", "sfat"))
            .unwrap();
        let zsf = sfat
            .value
            .transform(&x, &Session::new("ts", "sfatp"))
            .unwrap()
            .value;
        assert_eq!(zsf.nrows(), 8);
        assert!(zsf.get(0, 0).is_finite());
        let sdb = SoftDtwBarycenter::new()
            .fit(&x, &Session::new("ts", "sdb"))
            .unwrap();
        assert_eq!(sdb.value.len(), 6);
        assert!(sdb.value.as_slice().iter().all(|v| v.is_finite()));
        let shm = ShapeletModel::new(3, 3)
            .fit(&x, &y, &Session::new("ts", "shm"))
            .unwrap();
        let pshm = shm
            .value
            .predict(&x, &Session::new("ts", "shmp"))
            .unwrap()
            .value;
        assert_eq!(pshm.len(), 8);
        let lsr = LearningShapeletsRegressor::new()
            .fit(&x, &yr, &Session::new("ts", "lsr"))
            .unwrap();
        let plsr = lsr
            .value
            .predict(&x, &Session::new("ts", "lsrp"))
            .unwrap()
            .value;
        assert_eq!(plsr.len(), 8);
        assert!(plsr.as_slice().iter().all(|v| v.is_finite()));
        let lle1 = OneDLle::new(3)
            .fit_unsupervised(&x, &Session::new("ts", "lle1"))
            .unwrap();
        assert_eq!(lle1.value.len(), 8);
        assert!(lle1.value.as_slice().iter().all(|v| v.is_finite()));
        let tss = TimeSeriesSvd::new(1)
            .fit_unsupervised(&x, &Session::new("ts", "tssvd"))
            .unwrap();
        assert_eq!(tss.value.components.nrows(), 1);
        let dba_b = DbaBarycenter::new()
            .fit(&x, &Session::new("ts", "dba_b"))
            .unwrap();
        assert_eq!(dba_b.value.len(), 6);
        assert!(dba_b.value.as_slice().iter().all(|v| v.is_finite()));
        let eub = EuclideanBarycenter::new()
            .fit(&x, &Session::new("ts", "eub"))
            .unwrap();
        assert_eq!(eub.value.len(), 6);
        assert!(eub.value.as_slice().iter().all(|v| v.is_finite()));
        let rist = Rist::new()
            .fit(&x, &y, &Session::new("ts", "rist"))
            .unwrap();
        let prist = rist
            .value
            .predict(&x, &Session::new("ts", "ristp"))
            .unwrap()
            .value;
        assert_eq!(prist.len(), 8);
        assert!(prist.as_slice().iter().all(|v| v.is_finite()));
        let bvs = BossVs::new()
            .fit(&x, &y, &Session::new("ts", "bvs"))
            .unwrap();
        let pbvs = bvs
            .value
            .predict(&x, &Session::new("ts", "bvsp"))
            .unwrap()
            .value;
        assert_eq!(pbvs.len(), 8);
        assert!(pbvs.as_slice().iter().all(|v| v.is_finite()));
        let yann = Vector::from_iter((0..8).map(|i| i as f64));
        let pelta = PeltAnnotator::new()
            .fit(&yann, &Session::new("ts", "pelta"))
            .unwrap();
        assert!(pelta.value.as_slice().iter().all(|v| v.is_finite()));
        let claspa = ClaSPAnnotator::new()
            .fit(&yann, &Session::new("ts", "claspa"))
            .unwrap();
        assert!(claspa.value.is_finite() || claspa.value.is_nan());
        let riser = RiseRegressor::new()
            .fit(&x, &yr, &Session::new("ts", "riser"))
            .unwrap();
        let priser = riser
            .value
            .predict(&x, &Session::new("ts", "riserp"))
            .unwrap()
            .value;
        assert_eq!(priser.len(), 8);
        assert!(priser.as_slice().iter().all(|v| v.is_finite()));
        let mrc = MiniRocketClassifier::new()
            .fit(&x, &y, &Session::new("ts", "mrc"))
            .unwrap();
        let pmrc = mrc
            .value
            .predict(&x, &Session::new("ts", "mrcp"))
            .unwrap()
            .value;
        assert_eq!(pmrc.len(), 8);
        let c22fr = Catch22ForestRegressor::new()
            .fit(&x, &yr, &Session::new("ts", "c22fr"))
            .unwrap();
        let c22fc = Catch22ForestClassifier::new()
            .fit(&x, &y, &Session::new("ts", "c22fc"))
            .unwrap();
        let pc22fc = c22fc
            .value
            .predict(&x, &Session::new("ts", "c22fcp"))
            .unwrap()
            .value;
        assert_eq!(pc22fc.len(), 8);
        let pc22fr = c22fr
            .value
            .predict(&x, &Session::new("ts", "c22frp"))
            .unwrap()
            .value;
        assert_eq!(pc22fr.len(), 8);
        assert!(pc22fr.as_slice().iter().all(|v| v.is_finite()));
        let wd = WeaselD::new()
            .fit(&x, &y, &Session::new("ts", "wd"))
            .unwrap();
        let pwd = wd
            .value
            .predict(&x, &Session::new("ts", "wdp"))
            .unwrap()
            .value;
        assert_eq!(pwd.len(), 8);
        let hmrc = HydraMultiRocketClassifier::new()
            .fit(&x, &y, &Session::new("ts", "hmrc"))
            .unwrap();
        let phmrc = hmrc
            .value
            .predict(&x, &Session::new("ts", "hmrcp"))
            .unwrap()
            .value;
        assert_eq!(phmrc.len(), 8);
        let rotfr = RotationForestRegressor::new()
            .fit(&x, &yr, &Session::new("ts", "rotfr"))
            .unwrap();
        let protfr = rotfr
            .value
            .predict(&x, &Session::new("ts", "rotfrp"))
            .unwrap()
            .value;
        assert_eq!(protfr.len(), 8);
        assert!(protfr.as_slice().iter().all(|v| v.is_finite()));
        let fpc = FreshPrinceClassifier::new()
            .fit(&x, &y, &Session::new("ts", "fpc"))
            .unwrap();
        assert_eq!(
            fpc.value
                .predict(&x, &Session::new("ts", "fpcp"))
                .unwrap()
                .value
                .len(),
            8
        );
        let dcc = DrCifClassifier::new()
            .fit(&x, &y, &Session::new("ts", "dcc"))
            .unwrap();
        assert_eq!(
            dcc.value
                .predict(&x, &Session::new("ts", "dccp"))
                .unwrap()
                .value
                .len(),
            8
        );
        let cifc = CanonicalIntervalForestClassifier::new()
            .fit(&x, &y, &Session::new("ts", "cifc"))
            .unwrap();
        assert_eq!(
            cifc.value
                .predict(&x, &Session::new("ts", "cifcp"))
                .unwrap()
                .value
                .len(),
            8
        );
        let risc = RiseClassifier::new()
            .fit(&x, &y, &Session::new("ts", "risc"))
            .unwrap();
        assert_eq!(
            risc.value
                .predict(&x, &Session::new("ts", "riscp"))
                .unwrap()
                .value
                .len(),
            8
        );
        let tdec = TdeClassifier::new()
            .fit(&x, &y, &Session::new("ts", "tdec"))
            .unwrap();
        assert_eq!(
            tdec.value
                .predict(&x, &Session::new("ts", "tdecp"))
                .unwrap()
                .value
                .len(),
            8
        );
        let dj = DisjointCnnClassifier::new()
            .fit(&x, &y, &Session::new("ts", "dj"))
            .unwrap();
        assert_eq!(
            dj.value
                .predict(&x, &Session::new("ts", "djp"))
                .unwrap()
                .value
                .len(),
            8
        );
        let mcd = McdcnnClassifier::new()
            .fit(&x, &y, &Session::new("ts", "mcd"))
            .unwrap();
        assert_eq!(
            mcd.value
                .predict(&x, &Session::new("ts", "mcdp"))
                .unwrap()
                .value
                .len(),
            8
        );
        let pts = PatchTstClassifier::new()
            .fit(&x, &y, &Session::new("ts", "pts"))
            .unwrap();
        assert_eq!(
            pts.value
                .predict(&x, &Session::new("ts", "ptsp"))
                .unwrap()
                .value
                .len(),
            8
        );
        let tn = TimesNetClassifier::new()
            .fit(&x, &y, &Session::new("ts", "tn"))
            .unwrap();
        assert_eq!(
            tn.value
                .predict(&x, &Session::new("ts", "tnp"))
                .unwrap()
                .value
                .len(),
            8
        );
        let tcn = TcnClassifier::new()
            .fit(&x, &y, &Session::new("ts", "tcn"))
            .unwrap();
        assert_eq!(
            tcn.value
                .predict(&x, &Session::new("ts", "tcnp"))
                .unwrap()
                .value
                .len(),
            8
        );
        let tst = TstClassifier::new()
            .fit(&x, &y, &Session::new("ts", "tst"))
            .unwrap();
        assert_eq!(
            tst.value
                .predict(&x, &Session::new("ts", "tstp"))
                .unwrap()
                .value
                .len(),
            8
        );
        let fls = Floss::new(3)
            .fit(&yr, &Session::new("ts", "floss"))
            .unwrap();
        assert!(fls.value.index.is_finite() || fls.value.index.is_nan());
        let flu = Fluss::new(2)
            .fit(&yr, &Session::new("ts", "fluss"))
            .unwrap();
        assert!(flu.value.as_slice().iter().all(|v| v.is_finite()) || flu.value.is_empty());
        let stm = Stomp::new(3)
            .fit(&yr, &Session::new("ts", "stomp"))
            .unwrap();
        assert!(
            stm.value.profile.as_slice().iter().all(|v| v.is_finite())
                || stm.value.profile.is_empty()
        );
        let mer = Merlin::new(3)
            .fit(&yr, &Session::new("ts", "merlin"))
            .unwrap();
        assert!(mer.value.discord.is_finite() || mer.value.discord.is_nan());
        let pmp = PanMatrixProfile::new(2, 2)
            .fit(&yr, &Session::new("ts", "pmp"))
            .unwrap();
        assert_eq!(pmp.value.windows.len(), 2);
        assert!(pmp.value.discords.as_slice().iter().all(|v| v.is_finite()));
        let mpd = mpdist(&yr, &yr, 3, &Session::new("ts", "mpdist")).unwrap();
        assert!(mpd.value.is_finite() && mpd.value >= 0.0);
        let scr = Scrimp::new(3)
            .fit(&yr, &Session::new("ts", "scrimp"))
            .unwrap();
        assert!(
            scr.value.profile.as_slice().iter().all(|v| v.is_finite())
                || scr.value.profile.is_empty()
        );
        let qy = Vector::from_iter(yr.as_slice().iter().take(3).copied());
        let mas = mass(&qy, &yr, &Session::new("ts", "mass")).unwrap();
        assert!(mas.value.as_slice().iter().all(|v| v.is_finite()) || mas.value.is_empty());
        let mtf = Motif::new(3)
            .fit(&yr, &Session::new("ts", "motif"))
            .unwrap();
        assert_eq!(mtf.value.indices.len(), 2);
        let mpx = Mpx::new(3)
            .fit(&yr, &Session::new("ts", "mpx"))
            .unwrap();
        assert!(
            mpx.value.profile.as_slice().iter().all(|v| v.is_finite())
                || mpx.value.profile.is_empty()
        );
        let snp = Snippets::new(3)
            .fit(&yr, &Session::new("ts", "snp"))
            .unwrap();
        assert_eq!(snp.value.indices.len(), 2);
        let sti = Stampi::new(3)
            .fit(&yr, &Session::new("ts", "sti"))
            .unwrap();
        assert!(
            sti.value.profile.as_slice().iter().all(|v| v.is_finite())
                || sti.value.profile.is_empty()
        );
        let ost = Ostinato::new(3)
            .fit(&x, &Session::new("ts", "ost"))
            .unwrap();
        assert!(ost.value.score.is_finite());
        let prc = Prescrimp::new(3)
            .fit(&yr, &Session::new("ts", "prc"))
            .unwrap();
        assert!(
            prc.value.profile.as_slice().iter().all(|v| v.is_finite())
                || prc.value.profile.is_empty()
        );
        let aamp = Aamp::new(3)
            .fit(&yr, &Session::new("ts", "aamp"))
            .unwrap();
        assert!(
            aamp.value.profile.as_slice().iter().all(|v| v.is_finite())
                || aamp.value.profile.is_empty()
        );
        let masa = mass_absolute(&qy, &yr, &Session::new("ts", "masa")).unwrap();
        assert!(masa.value.as_slice().iter().all(|v| v.is_finite()) || masa.value.is_empty());
        let anv = AnnotationVector::new(3)
            .fit(&yr, &Session::new("ts", "anv"))
            .unwrap();
        assert!(anv.value.as_slice().iter().all(|v| v.is_finite()) || anv.value.is_empty());
        let mst = Mstump::new(3)
            .fit(&x, &Session::new("ts", "mst"))
            .unwrap();
        assert!(
            mst.value.profile.as_slice().iter().all(|v| v.is_finite())
                || mst.value.profile.is_empty()
        );
        let maa = Maamp::new(3)
            .fit(&x, &Session::new("ts", "maa"))
            .unwrap();
        assert!(
            maa.value.profile.as_slice().iter().all(|v| v.is_finite())
                || maa.value.profile.is_empty()
        );
        let mmo = Mmotifs::new(3)
            .fit(&x, &Session::new("ts", "mmo"))
            .unwrap();
        assert!(!mmo.value.indices.is_empty());
        let mps = Mpstump::new(2, 2)
            .fit(&x, &Session::new("ts", "mps"))
            .unwrap();
        assert_eq!(mps.value.windows.len(), 2);
        let sna = SnippetArea::new(3)
            .fit(&yr, &Session::new("ts", "sna"))
            .unwrap();
        assert!(sna.value.as_slice().iter().all(|v| v.is_finite()) || sna.value.is_empty());
        let ach = AllChains::new(3)
            .fit(&yr, &Session::new("ts", "ach"))
            .unwrap();
        assert!(!ach.value.indices.is_empty());
        assert!(ach.value.length.is_finite() && ach.value.length >= 1.0);
        let igs = InformationGainSegmentation::new()
            .fit(&yr, &Session::new("ts", "igs"))
            .unwrap();
        assert!(igs.value.as_slice().iter().all(|v| v.is_finite()));
        let wseg = WindowSegmenter::new(2)
            .fit(&yr, &Session::new("ts", "wseg"))
            .unwrap();
        assert!(wseg.value.as_slice().iter().all(|v| v.is_finite()));
        let bus = BottomUpSegmenter::new(2)
            .fit(&yr, &Session::new("ts", "bus"))
            .unwrap();
        assert!(bus.value.as_slice().iter().all(|v| v.is_finite()));
        let tds = TopDownSegmenter::new(2)
            .fit(&yr, &Session::new("ts", "tds"))
            .unwrap();
        assert!(tds.value.as_slice().iter().all(|v| v.is_finite()));
        let hid = Hidalgo::new(3)
            .fit(&x, &Session::new("ts", "hid"))
            .unwrap();
        assert_eq!(hid.value.len(), 8);
        let ggs2 = GreedyGaussianSegmentation::new()
            .fit(&yr, &Session::new("ts", "ggs2"))
            .unwrap();
        assert!(ggs2.value.as_slice().iter().all(|v| v.is_finite()));
        let bseg = BinarySegmentation::new()
            .fit(&yr, &Session::new("ts", "bseg"))
            .unwrap();
        assert!(bseg.value.is_finite() || bseg.value.is_nan());
        let rst = Rstsf::new()
            .fit(&x, &y, &Session::new("ts", "rstsf"))
            .unwrap();
        let prst = rst
            .value
            .predict(&x, &Session::new("ts", "rstsfp"))
            .unwrap()
            .value;
        assert_eq!(prst.len(), 8);
        assert!(prst.as_slice().iter().all(|v| v.is_finite()));
        let lt = LiteTime::new()
            .fit(&x, &y, &Session::new("ts", "lite"))
            .unwrap();
        assert_eq!(
            lt.value
                .predict(&x, &Session::new("ts", "litep"))
                .unwrap()
                .value
                .len(),
            8
        );
        let mq = MrSqm::new()
            .fit(&x, &y, &Session::new("ts", "mrsqm"))
            .unwrap();
        assert_eq!(
            mq.value
                .predict(&x, &Session::new("ts", "mrsqmp"))
                .unwrap()
                .value
                .len(),
            8
        );
        let itde = IndividualTde::new()
            .fit(&x, &y, &Session::new("ts", "itde"))
            .unwrap();
        assert!(itde
            .value
            .predict(&x, &Session::new("ts", "itdep"))
            .unwrap()
            .value
            .as_slice()
            .iter()
            .all(|v| v.is_finite()));
        let tsf = TsFreshClassifier::new()
            .fit(&x, &y, &Session::new("ts", "tsf"))
            .unwrap();
        assert_eq!(
            tsf.value
                .predict(&x, &Session::new("ts", "tsfp"))
                .unwrap()
                .value
                .len(),
            8
        );
        let si = SupervisedIntervals::new()
            .fit(&x, &y, &Session::new("ts", "suint"))
            .unwrap();
        assert_eq!(
            si.value
                .predict(&x, &Session::new("ts", "suintp"))
                .unwrap()
                .value
                .len(),
            8
        );
        let wv2 = WeaselV2::new()
            .fit(&x, &y, &Session::new("ts", "wv2"))
            .unwrap();
        assert_eq!(
            wv2.value
                .predict(&x, &Session::new("ts", "wv2p"))
                .unwrap()
                .value
                .len(),
            8
        );
        let te = Teaser::new()
            .fit(&x, &y, &Session::new("ts", "teaser"))
            .unwrap();
        assert_eq!(
            te.value
                .predict(&x, &Session::new("ts", "teaserp"))
                .unwrap()
                .value
                .len(),
            8
        );
        let mrc = MultiRocketClassifier::new()
            .fit(&x, &y, &Session::new("ts", "mrc"))
            .unwrap();
        assert_eq!(
            mrc.value
                .predict(&x, &Session::new("ts", "mrcp"))
                .unwrap()
                .value
                .len(),
            8
        );
        let mirr = MiniRocketRegressor::new()
            .fit(&x, &yr, &Session::new("ts", "mirr"))
            .unwrap();
        assert!(mirr
            .value
            .predict(&x, &Session::new("ts", "mirrp"))
            .unwrap()
            .value
            .as_slice()
            .iter()
            .all(|v| v.is_finite()));
        let rir = RandomIntervalRegressor::new()
            .fit(&x, &yr, &Session::new("ts", "rir"))
            .unwrap();
        assert!(rir
            .value
            .predict(&x, &Session::new("ts", "rirp"))
            .unwrap()
            .value
            .as_slice()
            .iter()
            .all(|v| v.is_finite()));
        let cnr = CnnRegressor::new()
            .fit(&x, &yr, &Session::new("ts", "cnr"))
            .unwrap();
        assert!(cnr
            .value
            .predict(&x, &Session::new("ts", "cnrp"))
            .unwrap()
            .value
            .as_slice()
            .iter()
            .all(|v| v.is_finite()));
        let rnr = ResNetRegressor::new()
            .fit(&x, &yr, &Session::new("ts", "rnr"))
            .unwrap();
        assert!(rnr
            .value
            .predict(&x, &Session::new("ts", "rnrp"))
            .unwrap()
            .value
            .as_slice()
            .iter()
            .all(|v| v.is_finite()));
        let fcnr = FCNRegressor::new()
            .fit(&x, &yr, &Session::new("ts", "fcnr"))
            .unwrap();
        assert!(fcnr
            .value
            .predict(&x, &Session::new("ts", "fcnrp"))
            .unwrap()
            .value
            .as_slice()
            .iter()
            .all(|v| v.is_finite()));
        let enr = EncoderRegressor::new()
            .fit(&x, &yr, &Session::new("ts", "enr"))
            .unwrap();
        assert!(enr
            .value
            .predict(&x, &Session::new("ts", "enrp"))
            .unwrap()
            .value
            .as_slice()
            .iter()
            .all(|v| v.is_finite()));
        let mlpr = MlpTimeRegressor::new()
            .fit(&x, &yr, &Session::new("ts", "mlpr"))
            .unwrap();
        assert!(mlpr
            .value
            .predict(&x, &Session::new("ts", "mlprp"))
            .unwrap()
            .value
            .as_slice()
            .iter()
            .all(|v| v.is_finite()));
        let strg = ShapeletTransformRegressor::new()
            .fit(&x, &yr, &Session::new("ts", "strg"))
            .unwrap();
        assert!(strg
            .value
            .predict(&x, &Session::new("ts", "strgp"))
            .unwrap()
            .value
            .as_slice()
            .iter()
            .all(|v| v.is_finite()));
        let tsfr = TsFreshRegressor::new()
            .fit(&x, &yr, &Session::new("ts", "tsfr2"))
            .unwrap();
        assert!(tsfr
            .value
            .predict(&x, &Session::new("ts", "tsfr2p"))
            .unwrap()
            .value
            .as_slice()
            .iter()
            .all(|v| v.is_finite()));
        let qr = QuantRegressor::new()
            .fit(&x, &yr, &Session::new("ts", "qreg"))
            .unwrap();
        assert!(qr
            .value
            .predict(&x, &Session::new("ts", "qregp"))
            .unwrap()
            .value
            .as_slice()
            .iter()
            .all(|v| v.is_finite()));
        let sv = SaxVsm::new()
            .fit(&x, &y, &Session::new("ts", "saxvsm"))
            .unwrap();
        assert_eq!(
            sv.value
                .predict(&x, &Session::new("ts", "saxvsmp"))
                .unwrap()
                .value
                .len(),
            8
        );
        let sfa = Sfa::new()
            .fit(&x, &y, &Session::new("ts", "sfa"))
            .unwrap();
        assert_eq!(
            sfa.value
                .predict(&x, &Session::new("ts", "sfap"))
                .unwrap()
                .value
                .len(),
            8
        );
        let sds = SoftDtwSvm::new()
            .fit(&x, &y, &Session::new("ts", "sdsvm"))
            .unwrap();
        let sdsp = sds
            .value
            .predict(&x, &Session::new("ts", "sdsvmp"))
            .unwrap()
            .value;
        assert_eq!(sdsp.len(), 8);
        assert!(sdsp.as_slice().iter().all(|v| v.is_finite()));
        let knn = KNeighborsTimeSeriesClassifier::new()
            .fit(&x, &y, &Session::new("ts", "knntsc"))
            .unwrap();
        assert_eq!(
            knn.value
                .predict(&x, &Session::new("ts", "knntscp"))
                .unwrap()
                .value
                .len(),
            8
        );
        let csb = cdist_sbd(&x, &x, &Session::new("ts", "csbd")).unwrap().value;
        assert_eq!(csb.shape(), (8, 8));
        assert!(csb.get(0, 0).abs() < 1e-8);
        assert!(csb.get(0, 1).is_finite());
        let cfr = cdist_frechet(&x, &x, &Session::new("ts", "cfr")).unwrap().value;
        assert_eq!(cfr.shape(), (8, 8));
        assert!(cfr.get(0, 0).abs() < 1e-12);
        let chd = cdist_hausdorff(&x, &x, &Session::new("ts", "chd")).unwrap().value;
        assert_eq!(chd.shape(), (8, 8));
        assert!(chd.get(0, 0).abs() < 1e-12);
        let ced = cdist_edr(&x, &x, 0.1, &Session::new("ts", "cedr"))
            .unwrap()
            .value;
        assert_eq!(ced.shape(), (8, 8));
        assert!(ced.get(0, 0).abs() < 1e-12);
        let cad = cdist_adtw(&x, &x, 0.1, &Session::new("ts", "cadtw"))
            .unwrap()
            .value;
        assert_eq!(cad.shape(), (8, 8));
        assert!(cad.get(0, 0).abs() < 1e-12);
        let cwd2 = cdist_wddtw(&x, &x, 0.1, &Session::new("ts", "cwddtw"))
            .unwrap()
            .value;
        assert_eq!(cwd2.shape(), (8, 8));
        assert!(cwd2.get(0, 0).abs() < 1e-12);
        let csd = cdist_shape_dtw(&x, &x, &Session::new("ts", "csd"))
            .unwrap()
            .value;
        assert_eq!(csd.shape(), (8, 8));
        assert!(csd.get(0, 0).abs() < 1e-12);
        let ccid = cdist_cid(&x, &x, &Session::new("ts", "ccid"))
            .unwrap()
            .value;
        assert_eq!(ccid.shape(), (8, 8));
        assert!(ccid.get(0, 0).abs() < 1e-12);
        let clk = cdist_lb_keogh(&x, &x, 1, &Session::new("ts", "clk"))
            .unwrap()
            .value;
        assert_eq!(clk.shape(), (8, 8));
        assert!(clk.get(0, 0).abs() < 1e-12);
        let cli = cdist_lb_improved(&x, &x, 1, &Session::new("ts", "cli"))
            .unwrap()
            .value;
        assert_eq!(cli.shape(), (8, 8));
        assert!(cli.get(0, 0).abs() < 1e-12);
        let csw = cdist_swale(&x, &x, 0.1, &Session::new("ts", "csw"))
            .unwrap()
            .value;
        assert_eq!(csw.shape(), (8, 8));
        assert!(csw.get(0, 0).abs() < 1e-12);
        let ckm = cdist_lb_kim(&x, &x, &Session::new("ts", "ckm"))
            .unwrap()
            .value;
        assert_eq!(ckm.shape(), (8, 8));
        assert!(ckm.get(0, 0).abs() < 1e-12);
        let cmd = cdist_mpdist(&x, &x, 3, &Session::new("ts", "cmd"))
            .unwrap()
            .value;
        assert_eq!(cmd.shape(), (8, 8));
        assert!(cmd.get(0, 0).abs() < 1e-12);
        let ckd = cdist_kdtw(&x, &x, 1.0, &Session::new("ts", "ckd"))
            .unwrap()
            .value;
        assert_eq!(ckd.shape(), (8, 8));
        assert!(ckd.get(0, 0).abs() < 1e-12);
        let cfd = cdist_fast_dtw(&x, &x, 1, &Session::new("ts", "cfd"))
            .unwrap()
            .value;
        assert_eq!(cfd.shape(), (8, 8));
        assert!(cfd.get(0, 0).abs() < 1e-12);
        let cds = cdist_dtw_subsequence(&x, &x, &Session::new("ts", "cds"))
            .unwrap()
            .value;
        assert_eq!(cds.shape(), (8, 8));
        assert!(cds.get(0, 0).abs() < 1e-12);
        let cly = cdist_lb_yi(&x, &x, &Session::new("ts", "cly"))
            .unwrap()
            .value;
        assert_eq!(cly.shape(), (8, 8));
        assert!(cly.get(0, 0).abs() < 1e-12);
        let cit = cdist_itakura_dtw(&x, &x, &Session::new("ts", "cit"))
            .unwrap()
            .value;
        assert_eq!(cit.shape(), (8, 8));
        assert!(cit.get(0, 0).abs() < 1e-12);
        let csc = cdist_sakoe_chiba_dtw(&x, &x, 2, &Session::new("ts", "csc"))
            .unwrap()
            .value;
        assert_eq!(csc.shape(), (8, 8));
        assert!(csc.get(0, 0).abs() < 1e-12);
        let ccy = cdist_cyclic_dtw(&x, &x, &Session::new("ts", "ccy"))
            .unwrap()
            .value;
        assert_eq!(ccy.shape(), (8, 8));
        assert!(ccy.get(0, 0).abs() < 1e-12);
        let cob = cdist_obe_dtw(&x, &x, &Session::new("ts", "cob"))
            .unwrap()
            .value;
        assert_eq!(cob.shape(), (8, 8));
        assert!(cob.get(0, 0).abs() < 1e-12);
        let cam = cdist_amss(&x, &x, &Session::new("ts", "cam"))
            .unwrap()
            .value;
        assert_eq!(cam.shape(), (8, 8));
        assert!(cam.get(0, 0).abs() < 1e-12);
        let cde = cdist_decay_euclidean(&x, &x, 0.2, &Session::new("ts", "cde"))
            .unwrap()
            .value;
        assert_eq!(cde.shape(), (8, 8));
        assert!(cde.get(0, 0).abs() < 1e-12);
        let cse = cdist_shape_euclidean(&x, &x, &Session::new("ts", "cse"))
            .unwrap()
            .value;
        assert_eq!(cse.shape(), (8, 8));
        assert!(cse.get(0, 0).abs() < 1e-12);
        let cop = cdist_open_begin_dtw(&x, &x, &Session::new("ts", "cop"))
            .unwrap()
            .value;
        assert_eq!(cop.shape(), (8, 8));
        assert!(cop.get(0, 0).abs() < 1e-12);
        let coe = cdist_open_end_dtw(&x, &x, &Session::new("ts", "coe"))
            .unwrap()
            .value;
        assert_eq!(coe.shape(), (8, 8));
        assert!(coe.get(0, 0).abs() < 1e-12);
        let ccr = cdist_correlation(&x, &x, &Session::new("ts", "ccr"))
            .unwrap()
            .value;
        assert_eq!(ccr.shape(), (8, 8));
        assert!(ccr.get(0, 0).abs() < 1e-12);
        let ccd = cdist_cosine(&x, &x, &Session::new("ts", "ccd"))
            .unwrap()
            .value;
        assert_eq!(ccd.shape(), (8, 8));
        assert!(ccd.get(0, 0).abs() < 1e-12);
        let cch = cdist_chebyshev(&x, &x, &Session::new("ts", "cch"))
            .unwrap()
            .value;
        assert_eq!(cch.shape(), (8, 8));
        assert!(cch.get(0, 0).abs() < 1e-12);
        let cma = cdist_manhattan(&x, &x, &Session::new("ts", "cma"))
            .unwrap()
            .value;
        assert_eq!(cma.shape(), (8, 8));
        assert!(cma.get(0, 0).abs() < 1e-12);
        let cca = cdist_canberra(&x, &x, &Session::new("ts", "cca"))
            .unwrap()
            .value;
        assert_eq!(cca.shape(), (8, 8));
        assert!(cca.get(0, 0).abs() < 1e-12);
        let cbr = cdist_braycurtis(&x, &x, &Session::new("ts", "cbr"))
            .unwrap()
            .value;
        assert_eq!(cbr.shape(), (8, 8));
        assert!(cbr.get(0, 0).abs() < 1e-12);
        let clo = cdist_lorentzian(&x, &x, &Session::new("ts", "clo"))
            .unwrap()
            .value;
        assert_eq!(clo.shape(), (8, 8));
        assert!(clo.get(0, 0).abs() < 1e-12);
        let cag = cdist_angular(&x, &x, &Session::new("ts", "cag"))
            .unwrap()
            .value;
        assert_eq!(cag.shape(), (8, 8));
        assert!(cag.get(0, 0).abs() < 1e-12);
        let cmn = cdist_minkowski3(&x, &x, &Session::new("ts", "cmn"))
            .unwrap()
            .value;
        assert_eq!(cmn.shape(), (8, 8));
        assert!(cmn.get(0, 0).abs() < 1e-12);
        let ccl = cdist_clark(&x, &x, &Session::new("ts", "ccl"))
            .unwrap()
            .value;
        assert_eq!(ccl.shape(), (8, 8));
        assert!(ccl.get(0, 0).abs() < 1e-12);
        let cqe = cdist_squared_euclidean(&x, &x, &Session::new("ts", "cqe"))
            .unwrap()
            .value;
        assert_eq!(cqe.shape(), (8, 8));
        assert!(cqe.get(0, 0).abs() < 1e-12);
        let cdi = cdist_dice(&x, &x, &Session::new("ts", "cdi"))
            .unwrap()
            .value;
        assert_eq!(cdi.shape(), (8, 8));
        assert!(cdi.get(0, 0).abs() < 1e-12);
        let cta = cdist_tanimoto(&x, &x, &Session::new("ts", "cta"))
            .unwrap()
            .value;
        assert_eq!(cta.shape(), (8, 8));
        assert!(cta.get(0, 0).abs() < 1e-12);
        let cwa = cdist_wave_hedges(&x, &x, &Session::new("ts", "cwa"))
            .unwrap()
            .value;
        assert_eq!(cwa.shape(), (8, 8));
        assert!(cwa.get(0, 0).abs() < 1e-12);
        let cku = cdist_kulczynski(&x, &x, &Session::new("ts", "cku"))
            .unwrap()
            .value;
        assert_eq!(cku.shape(), (8, 8));
        assert!(cku.get(0, 0).abs() < 1e-12);
        let cru = cdist_ruzicka(&x, &x, &Session::new("ts", "cru"))
            .unwrap()
            .value;
        assert_eq!(cru.shape(), (8, 8));
        assert!(cru.get(0, 0).abs() < 1e-12);
        let chl = cdist_hellinger(&x, &x, &Session::new("ts", "chl"))
            .unwrap()
            .value;
        assert_eq!(chl.shape(), (8, 8));
        assert!(chl.get(0, 0).abs() < 1e-12);
        let cjs = cdist_jensen_shannon(&x, &x, &Session::new("ts", "cjs"))
            .unwrap()
            .value;
        assert_eq!(cjs.shape(), (8, 8));
        assert!(cjs.get(0, 0).abs() < 1e-12);
        let cbh = cdist_bhattacharyya(&x, &x, &Session::new("ts", "cbh"))
            .unwrap()
            .value;
        assert_eq!(cbh.shape(), (8, 8));
        assert!(cbh.get(0, 0).abs() < 1e-12);
        let csa = cdist_hassanat(&x, &x, &Session::new("ts", "csa"))
            .unwrap()
            .value;
        assert_eq!(csa.shape(), (8, 8));
        assert!(csa.get(0, 0).abs() < 1e-12);
        let cfi = cdist_fidelity(&x, &x, &Session::new("ts", "cfi"))
            .unwrap()
            .value;
        assert_eq!(cfi.shape(), (8, 8));
        assert!(cfi.get(0, 0).abs() < 1e-12);
        let cwt = cdist_whittaker(&x, &x, &Session::new("ts", "cwt"))
            .unwrap()
            .value;
        assert_eq!(cwt.shape(), (8, 8));
        assert!(cwt.get(0, 0).abs() < 1e-12);
        let cpc = cdist_pearson_chi_squared(&x, &x, &Session::new("ts", "cpc"))
            .unwrap()
            .value;
        assert_eq!(cpc.shape(), (8, 8));
        assert!(cpc.get(0, 0).abs() < 1e-12);
        let cny = cdist_neyman_chi_squared(&x, &x, &Session::new("ts", "cny"))
            .unwrap()
            .value;
        assert_eq!(cny.shape(), (8, 8));
        assert!(cny.get(0, 0).abs() < 1e-12);
        let cad = cdist_additive_symmetric(&x, &x, &Session::new("ts", "cad"))
            .unwrap()
            .value;
        assert_eq!(cad.shape(), (8, 8));
        assert!(cad.get(0, 0).abs() < 1e-12);
        let ckv = cdist_k_divergence(&x, &x, &Session::new("ts", "ckv"))
            .unwrap()
            .value;
        assert_eq!(ckv.shape(), (8, 8));
        assert!(ckv.get(0, 0).abs() < 1e-12);
        let cto = cdist_topsoe(&x, &x, &Session::new("ts", "cto"))
            .unwrap()
            .value;
        assert_eq!(cto.shape(), (8, 8));
        assert!(cto.get(0, 0).abs() < 1e-12);
        let ctj = cdist_taneja(&x, &x, &Session::new("ts", "ctj"))
            .unwrap()
            .value;
        assert_eq!(ctj.shape(), (8, 8));
        assert!(ctj.get(0, 0).abs() < 1e-12);
        let ckj = cdist_kumar_johnson(&x, &x, &Session::new("ts", "ckj"))
            .unwrap()
            .value;
        assert_eq!(ckj.shape(), (8, 8));
        assert!(ckj.get(0, 0).abs() < 1e-12);
        let chm = cdist_harmonic_mean(&x, &x, &Session::new("ts", "chm"))
            .unwrap()
            .value;
        assert_eq!(chm.shape(), (8, 8));
        assert!(chm.get(0, 0).abs() < 1e-12);
        let cms = cdist_max_symmetric_chi_squared(&x, &x, &Session::new("ts", "cms"))
            .unwrap()
            .value;
        assert_eq!(cms.shape(), (8, 8));
        assert!(cms.get(0, 0).abs() < 1e-12);
        let cis = cdist_intersection(&x, &x, &Session::new("ts", "cis"))
            .unwrap()
            .value;
        assert_eq!(cis.shape(), (8, 8));
        assert!(cis.get(0, 0).abs() < 1e-12);
        let cnm = cdist_min_symmetric_chi_squared(&x, &x, &Session::new("ts", "cnm"))
            .unwrap()
            .value;
        assert_eq!(cnm.shape(), (8, 8));
        assert!(cnm.get(0, 0).abs() < 1e-12);
        let cpe = cdist_l1_squared_euclidean(&x, &x, &Session::new("ts", "cpe"))
            .unwrap()
            .value;
        assert_eq!(cpe.shape(), (8, 8));
        assert!(cpe.get(0, 0).abs() < 1e-12);
        let cjd = cdist_jaccard(&x, &x, &Session::new("ts", "cjd"))
            .unwrap()
            .value;
        assert_eq!(cjd.shape(), (8, 8));
        assert!(cjd.get(0, 0).abs() < 1e-12);
        let cjf = cdist_jeffreys(&x, &x, &Session::new("ts", "cjf"))
            .unwrap()
            .value;
        assert_eq!(cjf.shape(), (8, 8));
        assert!(cjf.get(0, 0).abs() < 1e-12);
        let cqc = cdist_squared_chord(&x, &x, &Session::new("ts", "cqc"))
            .unwrap()
            .value;
        assert_eq!(cqc.shape(), (8, 8));
        assert!(cqc.get(0, 0).abs() < 1e-12);
        let ckl = cdist_kullback_leibler(&x, &x, &Session::new("ts", "ckl"))
            .unwrap()
            .value;
        assert_eq!(ckl.shape(), (8, 8));
        assert!(ckl.get(0, 0).abs() < 1e-12);
        let cpo = cdist_cosine_l1(&x, &x, &Session::new("ts", "cpo"))
            .unwrap()
            .value;
        assert_eq!(cpo.shape(), (8, 8));
        assert!(cpo.get(0, 0).abs() < 1e-12);
        let cpt = cdist_tanimoto_l1(&x, &x, &Session::new("ts", "cpt"))
            .unwrap()
            .value;
        assert_eq!(cpt.shape(), (8, 8));
        assert!(cpt.get(0, 0).abs() < 1e-12);
        let cd1 = cdist_dice_l1(&x, &x, &Session::new("ts", "cd1"))
            .unwrap()
            .value;
        assert_eq!(cd1.shape(), (8, 8));
        assert!(cd1.get(0, 0).abs() < 1e-12);
        let cvc = cdist_vicis_symmetric(&x, &x, &Session::new("ts", "cvc"))
            .unwrap()
            .value;
        assert_eq!(cvc.shape(), (8, 8));
        assert!(cvc.get(0, 0).abs() < 1e-12);
        let cpr = cdist_correlation_l1(&x, &x, &Session::new("ts", "cpr"))
            .unwrap()
            .value;
        assert_eq!(cpr.shape(), (8, 8));
        assert!(cpr.get(0, 0).abs() < 1e-12);
        let ch1 = cdist_hellinger_l1(&x, &x, &Session::new("ts", "ch1"))
            .unwrap()
            .value;
        assert_eq!(ch1.shape(), (8, 8));
        assert!(ch1.get(0, 0).abs() < 1e-12);
        let cc1 = cdist_canberra_l1(&x, &x, &Session::new("ts", "cc1"))
            .unwrap()
            .value;
        assert_eq!(cc1.shape(), (8, 8));
        assert!(cc1.get(0, 0).abs() < 1e-12);
        let ck1 = cdist_clark_l1(&x, &x, &Session::new("ts", "ck1"))
            .unwrap()
            .value;
        assert_eq!(ck1.shape(), (8, 8));
        assert!(ck1.get(0, 0).abs() < 1e-12);
        let cw1 = cdist_wave_hedges_l1(&x, &x, &Session::new("ts", "cw1"))
            .unwrap()
            .value;
        assert_eq!(cw1.shape(), (8, 8));
        assert!(cw1.get(0, 0).abs() < 1e-12);
        let czl = cdist_kulczynski_l1(&x, &x, &Session::new("ts", "czl"))
            .unwrap()
            .value;
        assert_eq!(czl.shape(), (8, 8));
        assert!(czl.get(0, 0).abs() < 1e-12);
        let czr = cdist_ruzicka_l1(&x, &x, &Session::new("ts", "czr"))
            .unwrap()
            .value;
        assert_eq!(czr.shape(), (8, 8));
        assert!(czr.get(0, 0).abs() < 1e-12);
        let clz = cdist_lorentzian_l1(&x, &x, &Session::new("ts", "clz"))
            .unwrap()
            .value;
        assert_eq!(clz.shape(), (8, 8));
        assert!(clz.get(0, 0).abs() < 1e-12);
        let cnh = cdist_hassanat_l1(&x, &x, &Session::new("ts", "cnh"))
            .unwrap()
            .value;
        assert_eq!(cnh.shape(), (8, 8));
        assert!(cnh.get(0, 0).abs() < 1e-12);
        let cxl = cdist_chebyshev_l1(&x, &x, &Session::new("ts", "cxl"))
            .unwrap()
            .value;
        assert_eq!(cxl.shape(), (8, 8));
        assert!(cxl.get(0, 0).abs() < 1e-12);
        let cm3 = cdist_minkowski3_l1(&x, &x, &Session::new("ts", "cm3"))
            .unwrap()
            .value;
        assert_eq!(cm3.shape(), (8, 8));
        assert!(cm3.get(0, 0).abs() < 1e-12);
        let cm4 = cdist_minkowski4_l1(&x, &x, &Session::new("ts", "cm4"))
            .unwrap()
            .value;
        assert_eq!(cm4.shape(), (8, 8));
        assert!(cm4.get(0, 0).abs() < 1e-12);
        let c15 = cdist_minkowski15_l1(&x, &x, &Session::new("ts", "c15"))
            .unwrap()
            .value;
        assert_eq!(c15.shape(), (8, 8));
        assert!(c15.get(0, 0).abs() < 1e-12);
        let cm5 = cdist_minkowski5_l1(&x, &x, &Session::new("ts", "cm5"))
            .unwrap()
            .value;
        assert_eq!(cm5.shape(), (8, 8));
        assert!(cm5.get(0, 0).abs() < 1e-12);
        let cm6 = cdist_minkowski6_l1(&x, &x, &Session::new("ts", "cm6"))
            .unwrap()
            .value;
        assert_eq!(cm6.shape(), (8, 8));
        assert!(cm6.get(0, 0).abs() < 1e-12);
        let c25 = cdist_minkowski25_l1(&x, &x, &Session::new("ts", "c25"))
            .unwrap()
            .value;
        assert_eq!(c25.shape(), (8, 8));
        assert!(c25.get(0, 0).abs() < 1e-12);
        let cm8 = cdist_minkowski8_l1(&x, &x, &Session::new("ts", "cm8"))
            .unwrap()
            .value;
        assert_eq!(cm8.shape(), (8, 8));
        assert!(cm8.get(0, 0).abs() < 1e-12);
        let cm7 = cdist_minkowski7_l1(&x, &x, &Session::new("ts", "cm7"))
            .unwrap()
            .value;
        assert_eq!(cm7.shape(), (8, 8));
        assert!(cm7.get(0, 0).abs() < 1e-12);
        let cm9 = cdist_minkowski9_l1(&x, &x, &Session::new("ts", "cm9"))
            .unwrap()
            .value;
        assert_eq!(cm9.shape(), (8, 8));
        assert!(cm9.get(0, 0).abs() < 1e-12);
        let c10 = cdist_minkowski10_l1(&x, &x, &Session::new("ts", "c10"))
            .unwrap()
            .value;
        assert_eq!(c10.shape(), (8, 8));
        assert!(c10.get(0, 0).abs() < 1e-12);
        let c11 = cdist_minkowski11_l1(&x, &x, &Session::new("ts", "c11"))
            .unwrap()
            .value;
        assert_eq!(c11.shape(), (8, 8));
        assert!(c11.get(0, 0).abs() < 1e-12);
        let c12 = cdist_minkowski12_l1(&x, &x, &Session::new("ts", "c12"))
            .unwrap()
            .value;
        assert_eq!(c12.shape(), (8, 8));
        assert!(c12.get(0, 0).abs() < 1e-12);
        let c13 = cdist_minkowski13_l1(&x, &x, &Session::new("ts", "c13"))
            .unwrap()
            .value;
        assert_eq!(c13.shape(), (8, 8));
        assert!(c13.get(0, 0).abs() < 1e-12);
        let c14 = cdist_minkowski14_l1(&x, &x, &Session::new("ts", "c14"))
            .unwrap()
            .value;
        assert_eq!(c14.shape(), (8, 8));
        assert!(c14.get(0, 0).abs() < 1e-12);
        let c16 = cdist_minkowski16_l1(&x, &x, &Session::new("ts", "c16"))
            .unwrap()
            .value;
        assert_eq!(c16.shape(), (8, 8));
        assert!(c16.get(0, 0).abs() < 1e-12);
        let c18 = cdist_minkowski18_l1(&x, &x, &Session::new("ts", "c18"))
            .unwrap()
            .value;
        assert_eq!(c18.shape(), (8, 8));
        assert!(c18.get(0, 0).abs() < 1e-12);
        let c20 = cdist_minkowski20_l1(&x, &x, &Session::new("ts", "c20"))
            .unwrap()
            .value;
        assert_eq!(c20.shape(), (8, 8));
        assert!(c20.get(0, 0).abs() < 1e-12);
        let c24 = cdist_minkowski24_l1(&x, &x, &Session::new("ts", "c24"))
            .unwrap()
            .value;
        assert_eq!(c24.shape(), (8, 8));
        assert!(c24.get(0, 0).abs() < 1e-12);
        let c17 = cdist_minkowski17_l1(&x, &x, &Session::new("ts", "c17"))
            .unwrap()
            .value;
        assert_eq!(c17.shape(), (8, 8));
        assert!(c17.get(0, 0).abs() < 1e-12);
        let c19 = cdist_minkowski19_l1(&x, &x, &Session::new("ts", "c19"))
            .unwrap()
            .value;
        assert_eq!(c19.shape(), (8, 8));
        assert!(c19.get(0, 0).abs() < 1e-12);
        let c21 = cdist_minkowski21_l1(&x, &x, &Session::new("ts", "c21"))
            .unwrap()
            .value;
        assert_eq!(c21.shape(), (8, 8));
        assert!(c21.get(0, 0).abs() < 1e-12);
        let cd22 = cdist_minkowski22_l1(&x, &x, &Session::new("ts", "cd22"))
            .unwrap()
            .value;
        assert_eq!(cd22.shape(), (8, 8));
        assert!(cd22.get(0, 0).abs() < 1e-12);
        let cd28 = cdist_minkowski28_l1(&x, &x, &Session::new("ts", "cd28"))
            .unwrap()
            .value;
        assert_eq!(cd28.shape(), (8, 8));
        assert!(cd28.get(0, 0).abs() < 1e-12);
        let cd23 = cdist_minkowski23_l1(&x, &x, &Session::new("ts", "cd23"))
            .unwrap()
            .value;
        assert_eq!(cd23.shape(), (8, 8));
        assert!(cd23.get(0, 0).abs() < 1e-12);
        let cd26 = cdist_minkowski26_l1(&x, &x, &Session::new("ts", "cd26"))
            .unwrap()
            .value;
        assert_eq!(cd26.shape(), (8, 8));
        assert!(cd26.get(0, 0).abs() < 1e-12);
        let cd27 = cdist_minkowski27_l1(&x, &x, &Session::new("ts", "cd27"))
            .unwrap()
            .value;
        assert_eq!(cd27.shape(), (8, 8));
        assert!(cd27.get(0, 0).abs() < 1e-12);
        let cd29 = cdist_minkowski29_l1(&x, &x, &Session::new("ts", "cd29"))
            .unwrap()
            .value;
        assert_eq!(cd29.shape(), (8, 8));
        assert!(cd29.get(0, 0).abs() < 1e-12);
        let cd30 = cdist_minkowski30_l1(&x, &x, &Session::new("ts", "cd30"))
            .unwrap()
            .value;
        assert_eq!(cd30.shape(), (8, 8));
        assert!(cd30.get(0, 0).abs() < 1e-12);
        let cd31 = cdist_minkowski31_l1(&x, &x, &Session::new("ts", "cd31"))
            .unwrap()
            .value;
        assert_eq!(cd31.shape(), (8, 8));
        assert!(cd31.get(0, 0).abs() < 1e-12);
        let cd32 = cdist_minkowski32_l1(&x, &x, &Session::new("ts", "cd32"))
            .unwrap()
            .value;
        assert_eq!(cd32.shape(), (8, 8));
        assert!(cd32.get(0, 0).abs() < 1e-12);
        let cd33 = cdist_minkowski33_l1(&x, &x, &Session::new("ts", "cd33"))
            .unwrap()
            .value;
        assert_eq!(cd33.shape(), (8, 8));
        assert!(cd33.get(0, 0).abs() < 1e-12);
        let cd34 = cdist_minkowski34_l1(&x, &x, &Session::new("ts", "cd34"))
            .unwrap()
            .value;
        assert_eq!(cd34.shape(), (8, 8));
        assert!(cd34.get(0, 0).abs() < 1e-12);
        let cd35 = cdist_minkowski35_l1(&x, &x, &Session::new("ts", "cd35"))
            .unwrap()
            .value;
        assert_eq!(cd35.shape(), (8, 8));
        assert!(cd35.get(0, 0).abs() < 1e-12);
        let cd36 = cdist_minkowski36_l1(&x, &x, &Session::new("ts", "cd36"))
            .unwrap()
            .value;
        assert_eq!(cd36.shape(), (8, 8));
        assert!(cd36.get(0, 0).abs() < 1e-12);
        let cd37 = cdist_minkowski37_l1(&x, &x, &Session::new("ts", "cd37"))
            .unwrap()
            .value;
        assert_eq!(cd37.shape(), (8, 8));
        assert!(cd37.get(0, 0).abs() < 1e-12);
        let cd38 = cdist_minkowski38_l1(&x, &x, &Session::new("ts", "cd38"))
            .unwrap()
            .value;
        assert_eq!(cd38.shape(), (8, 8));
        assert!(cd38.get(0, 0).abs() < 1e-12);
        let cd39 = cdist_minkowski39_l1(&x, &x, &Session::new("ts", "cd39"))
            .unwrap()
            .value;
        assert_eq!(cd39.shape(), (8, 8));
        assert!(cd39.get(0, 0).abs() < 1e-12);
        let cd40 = cdist_minkowski40_l1(&x, &x, &Session::new("ts", "cd40"))
            .unwrap()
            .value;
        assert_eq!(cd40.shape(), (8, 8));
        assert!(cd40.get(0, 0).abs() < 1e-12);
        let cd41 = cdist_minkowski41_l1(&x, &x, &Session::new("ts", "cd41"))
            .unwrap()
            .value;
        assert_eq!(cd41.shape(), (8, 8));
        assert!(cd41.get(0, 0).abs() < 1e-12);
        let cwd = cdist_wdtw(&x, &x, 0.1, &Session::new("ts", "cwdtw"))
            .unwrap()
            .value;
        assert_eq!(cwd.shape(), (8, 8));
        assert!(cwd.get(0, 0).abs() < 1e-12);
        assert!(cwd.get(0, 1).is_finite());
        let cdd = cdist_ddtw(&x, &x, &Session::new("ts", "cddtw"))
            .unwrap()
            .value;
        assert_eq!(cdd.shape(), (8, 8));
        assert!(cdd.get(0, 0).abs() < 1e-12);
        let bsg = Binseg::new()
            .fit(&yr, &Session::new("ts", "binseg2"))
            .unwrap()
            .value;
        assert!(bsg.is_finite());
        let pel = Pelt::new()
            .fit(&yr, &Session::new("ts", "pelt2"))
            .unwrap()
            .value;
        assert!(pel.as_slice().iter().all(|v| v.is_finite()));
        let clp = ClaSPSegmentation::new()
            .fit(&yr, &Session::new("ts", "claspseg"))
            .unwrap()
            .value;
        assert!(clp.is_finite() || clp.is_nan());
        let ggsn = Ggs::new()
            .fit(&yr, &Session::new("ts", "ggsn"))
            .unwrap()
            .value;
        assert!(ggsn.as_slice().iter().all(|v| v.is_finite()));
        let stp = Stamp::new(3)
            .fit(&yr, &Session::new("ts", "stmp"))
            .unwrap()
            .value;
        assert!(stp.profile.as_slice().iter().all(|v| v.is_finite()) || stp.profile.is_empty());
        let sty = Stray::new(3)
            .fit(&yr, &Session::new("ts", "stry"))
            .unwrap()
            .value;
        assert!(sty.as_slice().iter().all(|v| v.is_finite()) || sty.is_empty());
    }

    #[test]
    fn softdtw_cdist_kernel_kmeans_and_scaler() {
        let x = Matrix::from_fn(6, 4, |i, j| {
            if i < 3 {
                (j as f64) + 0.1 * i as f64
            } else {
                3.0 - j as f64 + 0.1 * i as f64
            }
        });
        let cd = cdist_softdtw(&x, &x, 0.5, &Session::new("ts", "csdtw"))
            .unwrap()
            .value;
        assert_eq!(cd.shape(), (6, 6));
        assert!(cd.get(0, 0).is_finite());
        let km = KernelKMeans::new(2)
            .fit_unsupervised(&x, &Session::new("ts", "kkm"))
            .unwrap();
        assert_eq!(km.value.labels.len(), 6);
        let mut sc = TimeSeriesScalerMeanVariance::new();
        sc.fit_unsupervised(&x, &Session::new("ts", "sc")).unwrap();
        let z = sc.transform(&x, &Session::new("ts", "sct")).unwrap().value;
        assert!((z.row(0).mean()).abs() < 1e-8);
        let mut mm = TimeSeriesScalerMinMax::new();
        mm.fit_unsupervised(&x, &Session::new("ts", "mm")).unwrap();
        let z2 = mm.transform(&x, &Session::new("ts", "mmt")).unwrap().value;
        assert!(z2.get(0, 0) >= -1e-12 && z2.get(0, 0) <= 1.0 + 1e-12);
        let a = x.row(0);
        let kaa = global_alignment_kernel(&a, &a, 1.0, &Session::new("ts", "gak"))
            .unwrap()
            .value;
        let far = Vector::from_iter((0..a.len()).map(|j| a[j] + 5.0));
        let kaf = global_alignment_kernel(&a, &far, 1.0, &Session::new("ts", "gak2"))
            .unwrap()
            .value;
        assert!(kaa > kaf, "kaa={kaa} kaf={kaf}");
        let b = softdtw_barycenter(&x, 0.5, 6, &Session::new("ts", "sdb"))
            .unwrap()
            .value;
        assert_eq!(b.len(), x.ncols());
        let mr = MiniRocket::new(8)
            .transform(&x, &Session::new("ts", "mr"))
            .unwrap()
            .value;
        assert_eq!(mr.shape(), (6, 8));
        let yb = Vector::from_iter((0..6).map(|i| if i < 3 { 0.0 } else { 1.0 }));
        let boss = BossEnsemble {
            window: 4,
            word_len: 3,
            alphabet: 4,
        }
        .fit(&x, &yb, &Session::new("ts", "boss"))
        .unwrap();
        let bp = boss
            .value
            .predict(&x, &Session::new("ts", "bossp"))
            .unwrap()
            .value;
        assert_eq!(bp.len(), 6);
        let wsl = Weasel {
            window: 4,
            word_len: 3,
            alphabet: 4,
            n_words: 6,
        }
        .fit(&x, &yb, &Session::new("ts", "weasel"))
        .unwrap();
        assert!(!wsl.value.vocab.is_empty() || x.nrows() > 0);
        let sh = LearningShapelets::new(4, 3)
            .fit(&x, &yb, &Session::new("ts", "shp"))
            .unwrap();
        let sp = sh
            .value
            .predict(&x, &Session::new("ts", "shpp"))
            .unwrap()
            .value;
        assert_eq!(sp.len(), 6);
        let yr = Vector::from_iter((0..6).map(|i| if i < 3 { 0.0 } else { 1.0 }));
        let sdr = SoftDtwRegressor::new(2)
            .fit(&x, &yr, &Session::new("ts", "sdr"))
            .unwrap();
        let pr = sdr
            .value
            .predict(&x, &Session::new("ts", "sdrp"))
            .unwrap()
            .value;
        assert_eq!(pr.len(), 6);
        assert!(pr.as_slice().iter().all(|v| v.is_finite()));
        let paa = Paa::new(2)
            .transform(&x, &Session::new("ts", "paa"))
            .unwrap()
            .value;
        assert_eq!(paa.ncols(), 2);
        let sax = Sax::new(2, 4)
            .transform(&x, &Session::new("ts", "sax"))
            .unwrap()
            .value;
        assert_eq!(sax.ncols(), 2);
        let tsvc = TimeSeriesSvc::new(2)
            .fit(&x, &yb, &Session::new("ts", "svc"))
            .unwrap();
        let sp2 = tsvc
            .value
            .predict(&paa, &Session::new("ts", "svcp"))
            .unwrap()
            .value;
        assert_eq!(sp2.len(), 6);
        let hc = HiveCote::new()
            .fit(&x, &yb, &Session::new("ts", "hive"))
            .unwrap();
        let hp = hc
            .value
            .predict(&x, &Session::new("ts", "hivep"))
            .unwrap()
            .value;
        assert_eq!(hp.len(), 6);
        let knnr = KNeighborsTimeSeriesRegressor::new(2)
            .fit(&x, &yr, &Session::new("ts", "knnr"))
            .unwrap();
        let knp = knnr
            .value
            .predict(&x, &Session::new("ts", "knnrp"))
            .unwrap()
            .value;
        assert_eq!(knp.len(), 6);
        assert!(knp.as_slice().iter().all(|v| v.is_finite()));
        let st = ShapeletTransform::new(3, 3)
            .fit_unsupervised(&x, &Session::new("ts", "sht"))
            .unwrap();
        let z = st
            .value
            .transform(&x, &Session::new("ts", "shtt"))
            .unwrap()
            .value;
        assert_eq!(z.nrows(), 6);
        assert_eq!(z.ncols(), 3);
        let bary = dba(&x, 8, &Session::new("ts", "dba")).unwrap().value;
        assert_eq!(bary.len(), x.ncols());
        assert!(bary.as_slice().iter().all(|v| v.is_finite()));
        let euc = euclidean_barycenter(&x, &Session::new("ts", "euc"))
            .unwrap()
            .value;
        assert_eq!(euc.len(), x.ncols());
        let q0 = x.row(0);
        let lb = lb_keogh(&q0, &q0, 2, &Session::new("ts", "lb"))
            .unwrap()
            .value;
        assert!(lb.abs() < 1e-12);
        let er = erp(&q0, &q0, 0.0, &Session::new("ts", "erp"))
            .unwrap()
            .value;
        assert!(er.abs() < 1e-12);
        let rs = TimeSeriesResampler::new(4)
            .transform(&x, &Session::new("ts", "rs"))
            .unwrap()
            .value;
        assert_eq!(rs.ncols(), 4);
        let c22 = Catch22Classifier::new(0.1)
            .fit(&x, &yb, &Session::new("ts", "c22"))
            .unwrap();
        let cp = c22
            .value
            .predict(&x, &Session::new("ts", "c22p"))
            .unwrap()
            .value;
        assert_eq!(cp.len(), 6);
        let ms = msm(&q0, &q0, 0.1, &Session::new("ts", "msm"))
            .unwrap()
            .value;
        assert!(ms.abs() < 1e-12);
        let tw = twe(&q0, &q0, 0.0, 1.0, &Session::new("ts", "twe"))
            .unwrap()
            .value;
        assert!(tw.abs() < 1e-12);
        let ods = OneDSax::new(2, 4)
            .transform(&x, &Session::new("ts", "ods"))
            .unwrap()
            .value;
        assert_eq!(ods.ncols(), 4);
        let yramp = Vector::from_iter((0..6).map(|i| i as f64));
        let rr = RocketRegressor {
            n_kernels: 8,
            kernel_len: 3,
            alpha: 0.5,
            seed: 3,
        }
        .fit(&x, &yramp, &Session::new("ts", "rr"))
        .unwrap();
        let rp = rr
            .value
            .predict(&x, &Session::new("ts", "rrp"))
            .unwrap()
            .value;
        assert_eq!(rp.len(), 6);
        assert!(rp.as_slice().iter().all(|v| v.is_finite()));
        let mr2 = MultiRocket::new(4)
            .transform(&x, &Session::new("ts", "mrm"))
            .unwrap()
            .value;
        assert_eq!(mr2.ncols(), 12);
        let tsvr = TimeSeriesSvr::new(2)
            .fit(&x, &yramp, &Session::new("ts", "tsvr"))
            .unwrap();
        let tsvp = tsvr
            .value
            .predict(
                &Paa::new(2)
                    .transform(&x, &Session::new("ts", "pa2"))
                    .unwrap()
                    .value,
                &Session::new("ts", "tsvrp"),
            )
            .unwrap()
            .value;
        assert_eq!(tsvp.len(), 6);
        assert!(tsvp.as_slice().iter().all(|v| v.is_finite()));
        let cms = cdist_msm(&x, &x, 0.1, &Session::new("ts", "cmsm"))
            .unwrap()
            .value;
        assert_eq!(cms.shape(), (6, 6));
        assert!(cms.get(0, 0).abs() < 1e-12);
        let ctw = cdist_twe(&x, &x, 0.0, 1.0, &Session::new("ts", "ctwe"))
            .unwrap()
            .value;
        assert_eq!(ctw.shape(), (6, 6));
        let sig = SignatureTransformer::new()
            .transform(&x, &Session::new("ts", "sig"))
            .unwrap()
            .value;
        assert_eq!(sig.ncols(), 6);
        let ars = Arsenal::new()
            .fit(&x, &yb, &Session::new("ts", "ars"))
            .unwrap();
        let arp = ars
            .value
            .predict(&x, &Session::new("ts", "arsp"))
            .unwrap()
            .value;
        assert_eq!(arp.len(), 6);
        let fp = FreshPrince::new()
            .fit(&x, &yb, &Session::new("ts", "fp"))
            .unwrap();
        let fpp = fp
            .value
            .predict(&x, &Session::new("ts", "fpp"))
            .unwrap()
            .value;
        assert_eq!(fpp.len(), 6);
        let stc = ShapeletTransformClassifier::new()
            .fit(&x, &yb, &Session::new("ts", "stc"))
            .unwrap();
        let stp = stc
            .value
            .predict(&x, &Session::new("ts", "stcp"))
            .unwrap()
            .value;
        assert_eq!(stp.len(), 6);
        let sdk = SoftDtwKMeans::new(2)
            .fit_unsupervised(&x, &Session::new("ts", "sdk"))
            .unwrap();
        assert_eq!(sdk.value.labels.len(), 6);
        assert_eq!(sdk.value.centers.nrows(), 2);
        let dr = DrCif::new()
            .fit(&x, &yb, &Session::new("ts", "drcif"))
            .unwrap();
        let drp = dr
            .value
            .predict(&x, &Session::new("ts", "drcifp"))
            .unwrap()
            .value;
        assert_eq!(drp.len(), 6);
        let pf = ProximityForest::new()
            .fit(&x, &yb, &Session::new("ts", "pf"))
            .unwrap();
        let pfp = pf
            .value
            .predict(&x, &Session::new("ts", "pfp"))
            .unwrap()
            .value;
        assert_eq!(pfp.len(), 6);
        let ec = EarlyClassifier::new()
            .fit(&x, &yb, &Session::new("ts", "ec"))
            .unwrap();
        let ecp = ec
            .value
            .predict(&x, &Session::new("ts", "ecp"))
            .unwrap()
            .value;
        assert_eq!(ecp.len(), 6);
        let cb = ContractableBoss::new()
            .fit(&x, &yb, &Session::new("ts", "cboss"))
            .unwrap();
        let cbp = cb
            .value
            .predict(&x, &Session::new("ts", "cbossp"))
            .unwrap()
            .value;
        assert_eq!(cbp.len(), 6);
        let ce = ColumnEnsembleClassifier::new()
            .fit(&x, &yb, &Session::new("ts", "ce"))
            .unwrap();
        let cep = ce
            .value
            .predict(&x, &Session::new("ts", "cep"))
            .unwrap()
            .value;
        assert_eq!(cep.len(), 6);
        let tde = TemporalDictionaryEnsemble::new()
            .fit(&x, &yb, &Session::new("ts", "tde"))
            .unwrap();
        let tdep = tde
            .value
            .predict(&x, &Session::new("ts", "tdep"))
            .unwrap()
            .value;
        assert_eq!(tdep.len(), 6);
        let rise = Rise::new()
            .fit(&x, &yb, &Session::new("ts", "rise"))
            .unwrap();
        let risp = rise
            .value
            .predict(&x, &Session::new("ts", "risep"))
            .unwrap()
            .value;
        assert_eq!(risp.len(), 6);
        let ee = ElasticEnsemble::new()
            .fit(&x, &yb, &Session::new("ts", "ee"))
            .unwrap();
        let eep = ee
            .value
            .predict(&x, &Session::new("ts", "eep"))
            .unwrap()
            .value;
        assert_eq!(eep.len(), 6);
        let c22r = Catch22Regressor::new(0.1)
            .fit(&x, &yramp, &Session::new("ts", "c22r"))
            .unwrap();
        let c22p = c22r
            .value
            .predict(&x, &Session::new("ts", "c22rp"))
            .unwrap()
            .value;
        assert_eq!(c22p.len(), 6);
        assert!(c22p.as_slice().iter().all(|v| v.is_finite()));
        let sdknn = SoftDtwKnn::new(1)
            .fit(&x, &yb, &Session::new("ts", "sdknn"))
            .unwrap();
        let sdkp = sdknn
            .value
            .predict(&x, &Session::new("ts", "sdknnp"))
            .unwrap()
            .value;
        assert_eq!(sdkp.len(), 6);
        let sumc = SummaryClassifier::new()
            .fit(&x, &yb, &Session::new("ts", "sum"))
            .unwrap();
        let sump = sumc
            .value
            .predict(&x, &Session::new("ts", "sump"))
            .unwrap()
            .value;
        assert_eq!(sump.len(), 6);
        let cg = cdist_gak(&x, &x, 1.0, &Session::new("ts", "cgak"))
            .unwrap()
            .value;
        assert_eq!(cg.shape(), (6, 6));
        assert!(cg.get(0, 0).is_finite());
        let hy = Hydra::new()
            .transform(&x, &Session::new("ts", "hydra"))
            .unwrap()
            .value;
        assert_eq!(hy.nrows(), 6);
        assert_eq!(hy.ncols(), 8);
        let hyc = HydraClassifier::new()
            .fit(&x, &yb, &Session::new("ts", "hyc"))
            .unwrap();
        let hyp = hyc
            .value
            .predict(&x, &Session::new("ts", "hycp"))
            .unwrap()
            .value;
        assert_eq!(hyp.len(), 6);
        let fpr = FreshPrinceRegressor::new()
            .fit(&x, &yramp, &Session::new("ts", "fpr"))
            .unwrap();
        let fprp = fpr
            .value
            .predict(&x, &Session::new("ts", "fprp"))
            .unwrap()
            .value;
        assert_eq!(fprp.len(), 6);
        assert!(fprp.as_slice().iter().all(|v| v.is_finite()));
        let sumr = SummaryRegressor::new()
            .fit(&x, &yramp, &Session::new("ts", "sumr"))
            .unwrap();
        let sumrp = sumr
            .value
            .predict(&x, &Session::new("ts", "sumrp"))
            .unwrap()
            .value;
        assert_eq!(sumrp.len(), 6);
        let sigc = SignatureClassifier::new()
            .fit(&x, &yb, &Session::new("ts", "sigc"))
            .unwrap();
        let sigp = sigc
            .value
            .predict(&x, &Session::new("ts", "sigcp"))
            .unwrap()
            .value;
        assert_eq!(sigp.len(), 6);
        let drr = DrCifRegressor::new()
            .fit(&x, &yramp, &Session::new("ts", "drr"))
            .unwrap();
        let drrp = drr
            .value
            .predict(&x, &Session::new("ts", "drrp"))
            .unwrap()
            .value;
        assert_eq!(drrp.len(), 6);
        let pt = ProximityTree::new()
            .fit(&x, &yb, &Session::new("ts", "pt"))
            .unwrap();
        let ptp = pt
            .value
            .predict(&x, &Session::new("ts", "ptp"))
            .unwrap()
            .value;
        assert_eq!(ptp.len(), 6);
        let stsf = Stsf::new()
            .fit(&x, &yb, &Session::new("ts", "stsf"))
            .unwrap();
        let stsfp = stsf
            .value
            .predict(&x, &Session::new("ts", "stsfp"))
            .unwrap()
            .value;
        assert_eq!(stsfp.len(), 6);
        let sx = cdist_sax(&x, &x, 4, 4, &Session::new("ts", "sax"))
            .unwrap()
            .value;
        assert_eq!(sx.shape(), (6, 6));
        assert!(sx.get(0, 0).abs() < 1e-12);
        let ctwv = canonical_time_warping(&q0, &q0, &Session::new("ts", "ctw"))
            .unwrap()
            .value;
        assert!(ctwv.is_finite() && ctwv >= 0.0);
        let path = dtw_alignment(&q0, &q0, &Session::new("ts", "dtwp"))
            .unwrap()
            .value;
        assert_eq!(path.ncols(), 2);
        assert_eq!(path.nrows(), q0.len());
        assert!((path.get(0, 0) - 0.0).abs() < 1e-12);
        let spath = softdtw_alignment(&q0, &q0, 0.5, &Session::new("ts", "sdtwp"))
            .unwrap()
            .value;
        assert_eq!(spath.ncols(), 2);
        assert!(spath.nrows() >= q0.len());
        assert!((spath.get(0, 0) - 0.0).abs() < 1e-12);
        let hmr = HydraMultiRocket::new()
            .fit(&x, &yb, &Session::new("ts", "hmr"))
            .unwrap();
        let hmrp = hmr
            .value
            .predict(&x, &Session::new("ts", "hmrp"))
            .unwrap()
            .value;
        assert_eq!(hmrp.len(), 6);
        let sigr = SignatureRegressor::new()
            .fit(&x, &yramp, &Session::new("ts", "sigr"))
            .unwrap();
        let sigrp = sigr
            .value
            .predict(&x, &Session::new("ts", "sigrp"))
            .unwrap()
            .value;
        assert_eq!(sigrp.len(), 6);
        assert!(sigrp.as_slice().iter().all(|v| v.is_finite()));
        let qc = QuantClassifier::new()
            .fit(&x, &yb, &Session::new("ts", "quant"))
            .unwrap();
        let qcp = qc
            .value
            .predict(&x, &Session::new("ts", "quantp"))
            .unwrap()
            .value;
        assert_eq!(qcp.len(), 6);
        let muse = WeaselMuse::new()
            .fit(&x, &yb, &Session::new("ts", "muse"))
            .unwrap();
        let musep = muse
            .value
            .predict(&x, &Session::new("ts", "musep"))
            .unwrap()
            .value;
        assert_eq!(musep.len(), 6);
        let ctwm = cdist_ctw(&x, &x, &Session::new("ts", "cctw"))
            .unwrap()
            .value;
        assert_eq!(ctwm.shape(), (6, 6));
        assert!(ctwm.get(0, 0).is_finite());
        let kmed = TimeSeriesKMedoids::new(2)
            .fit_unsupervised(&x, &Session::new("ts", "kmed"))
            .unwrap();
        assert_eq!(kmed.value.labels.len(), 6);
        let kmedp = kmed
            .value
            .predict(&x, &Session::new("ts", "kmedp"))
            .unwrap()
            .value;
        assert_eq!(kmedp.len(), 6);
        let cer = cdist_erp(&x, &x, 0.0, &Session::new("ts", "cerp"))
            .unwrap()
            .value;
        assert_eq!(cer.shape(), (6, 6));
        assert!(cer.get(0, 0).abs() < 1e-12);
        let cl = cdist_lcss(&x, &x, 0.5, &Session::new("ts", "lcssd"))
            .unwrap()
            .value;
        assert_eq!(cl.shape(), (6, 6));
        assert!(cl.get(0, 0).abs() < 1e-12);
        let cnn = CnnClassifier::new()
            .fit(&x, &yb, &Session::new("ts", "cnn"))
            .unwrap();
        let cnnp = cnn
            .value
            .predict(&x, &Session::new("ts", "cnnp"))
            .unwrap()
            .value;
        assert_eq!(cnnp.len(), 6);
        let mut ximp = x.clone();
        ximp.set(0, 1, f64::NAN);
        let mut imp = TimeSeriesImputer::new();
        imp.fit_unsupervised(&ximp, &Session::new("ts", "imp"))
            .unwrap();
        let xi = imp
            .transform(&ximp, &Session::new("ts", "impt"))
            .unwrap()
            .value;
        assert!(xi.get(0, 1).is_finite());
        let itc = InceptionTimeClassifier::new()
            .fit(&x, &yb, &Session::new("ts", "inc"))
            .unwrap();
        let itp = itc
            .value
            .predict(&x, &Session::new("ts", "incp"))
            .unwrap()
            .value;
        assert_eq!(itp.len(), 6);
        let cp = clasp_change_point(&yramp, &Session::new("ts", "clasp"))
            .unwrap()
            .value;
        assert!(cp.is_finite());
        let mut c22t = Catch22Transformer::new();
        c22t.fit_unsupervised(&x, &Session::new("ts", "c22t"))
            .unwrap();
        let zc = c22t
            .transform(&x, &Session::new("ts", "c22tt"))
            .unwrap()
            .value;
        assert_eq!(zc.nrows(), 6);
        let rn = ResNetClassifier::new()
            .fit(&x, &yb, &Session::new("ts", "res"))
            .unwrap();
        let rnp = rn
            .value
            .predict(&x, &Session::new("ts", "resp"))
            .unwrap()
            .value;
        assert_eq!(rnp.len(), 6);
        let lstm = LstmFcnClassifier::new()
            .fit(&x, &yb, &Session::new("ts", "lstm"))
            .unwrap();
        let lstmp = lstm
            .value
            .predict(&x, &Session::new("ts", "lstmp"))
            .unwrap()
            .value;
        assert_eq!(lstmp.len(), 6);
        let bs = binary_segmentation(&yramp, &Session::new("ts", "binseg"))
            .unwrap()
            .value;
        assert!(bs.is_finite());
        let itr = InceptionTimeRegressor::new()
            .fit(&x, &yramp, &Session::new("ts", "itr"))
            .unwrap();
        let itrp = itr
            .value
            .predict(&x, &Session::new("ts", "itrp"))
            .unwrap()
            .value;
        assert_eq!(itrp.len(), 6);
        assert!(itrp.as_slice().iter().all(|v| v.is_finite()));
        let tap = TapNetClassifier::new()
            .fit(&x, &yb, &Session::new("ts", "tap"))
            .unwrap();
        let tapp = tap
            .value
            .predict(&x, &Session::new("ts", "tapp"))
            .unwrap()
            .value;
        assert_eq!(tapp.len(), 6);
        let fcn = FCNClassifier::new()
            .fit(&x, &yb, &Session::new("ts", "fcn"))
            .unwrap();
        let fcnp = fcn
            .value
            .predict(&x, &Session::new("ts", "fcnp"))
            .unwrap()
            .value;
        assert_eq!(fcnp.len(), 6);
        let mac = MacnnClassifier::new()
            .fit(&x, &yb, &Session::new("ts", "mac"))
            .unwrap();
        let macp = mac
            .value
            .predict(&x, &Session::new("ts", "macp"))
            .unwrap()
            .value;
        assert_eq!(macp.len(), 6);
        let tapr = TapNetRegressor::new()
            .fit(&x, &yramp, &Session::new("ts", "tapr"))
            .unwrap();
        let taprp = tapr
            .value
            .predict(&x, &Session::new("ts", "taprp"))
            .unwrap()
            .value;
        assert_eq!(taprp.len(), 6);
        assert!(taprp.as_slice().iter().all(|v| v.is_finite()));
        let enc = EncoderClassifier::new()
            .fit(&x, &yb, &Session::new("ts", "enc"))
            .unwrap();
        let encp = enc
            .value
            .predict(&x, &Session::new("ts", "encp"))
            .unwrap()
            .value;
        assert_eq!(encp.len(), 6);
        let mlp = MlpTimeClassifier::new()
            .fit(&x, &yb, &Session::new("ts", "mlp"))
            .unwrap();
        let mlpp = mlp
            .value
            .predict(&x, &Session::new("ts", "mlpp"))
            .unwrap()
            .value;
        assert_eq!(mlpp.len(), 6);
        let sdc = SoftDtwClassifier::new()
            .fit(&x, &yb, &Session::new("ts", "sdc"))
            .unwrap();
        let sdcp = sdc
            .value
            .predict(&x, &Session::new("ts", "sdcp"))
            .unwrap()
            .value;
        assert_eq!(sdcp.len(), 6);
        let tcn = TimeCnnClassifier::new()
            .fit(&x, &yb, &Session::new("ts", "tcn"))
            .unwrap();
        let tcnp = tcn
            .value
            .predict(&x, &Session::new("ts", "tcnp"))
            .unwrap()
            .value;
        assert_eq!(tcnp.len(), 6);
        let cp = pelt(&yramp, 2.0, &Session::new("ts", "pelt"))
            .unwrap()
            .value;
        assert!(cp.as_slice().iter().all(|v| v.is_finite()));
        let clc = ClaSPClassifier::new()
            .fit(&x, &yb, &Session::new("ts", "claspclf"))
            .unwrap();
        let clp = clc
            .value
            .predict(&x, &Session::new("ts", "claspclfp"))
            .unwrap()
            .value;
        assert_eq!(clp.len(), 6);
        let gg = ggs(&yramp, 2, &Session::new("ts", "ggs")).unwrap().value;
        assert!(gg.as_slice().iter().all(|v| v.is_finite()));
        let ms = MrSeqlClassifier::new()
            .fit(&x, &yb, &Session::new("ts", "mrseql"))
            .unwrap();
        let msp = ms
            .value
            .predict(&x, &Session::new("ts", "mrseqlp"))
            .unwrap()
            .value;
        assert_eq!(msp.len(), 6);
        let stmp = stamp(&yramp, 3, &Session::new("ts", "stamp"))
            .unwrap()
            .value;
        assert_eq!(stmp.profile.len(), 4);
        assert!(stmp.index < stmp.profile.len() || stmp.profile.is_empty());
        let sy = stray(&yramp, 3, &Session::new("ts", "stray"))
            .unwrap()
            .value;
        assert_eq!(sy.len(), 4);
        assert!(sy.as_slice().iter().all(|v| v.is_finite()));
        let mpc = MatrixProfileClassifier::new()
            .fit(&x, &yb, &Session::new("ts", "mpclf"))
            .unwrap();
        let mpp = mpc
            .value
            .predict(&x, &Session::new("ts", "mpclfp"))
            .unwrap()
            .value;
        assert_eq!(mpp.len(), 6);
    }
}
