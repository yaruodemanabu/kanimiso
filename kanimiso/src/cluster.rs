//! Clustering: k-means, mini-batch / STREAM k-means, DBSCAN, agglomerative,
//! Gaussian mixtures, spectral clustering, and affinity propagation.
//!
//! Every estimator talks to [`FitCtx`], inspects the design with
//! [`inspect_xy`] / [`inspect_identification`], and records empty-cluster,
//! degeneracy, rank, NaN, and meaningless-fit issues. Linear algebra goes
//! through [`crate::linalg`] or [`Matrix::inner`].

use crate::context::FitCtx;
use crate::data::{Matrix, Vector};
use crate::linalg::{symmetric_eigen, thin_svd};
use crate::rng::Rng;
use crate::traits::{FitUnsupervised, PartialFit, Predict};
use crate::validate::{inspect_identification, inspect_xy};
use faer::Mat;
use ojizou_san::{IncrementalExplain, Session};
use signlred::{
    IncrementalQuality, Issue, IssueCode, Meaninglessness, Qualified, Result, Severity,
};

const COV_FLOOR: f64 = 1e-6;
const EMPTY_RESEED_TOL: f64 = 1e-15;

/// Squared Euclidean distance between row `i` of `x` and row `c` of `centers`.
fn sq_dist_rc(x: &Matrix, i: usize, centers: &Matrix, c: usize) -> f64 {
    let p = x.ncols().min(centers.ncols());
    let mut s = 0.0;
    for j in 0..p {
        let d = x.get(i, j) - centers.get(c, j);
        s += d * d;
    }
    s
}

/// Squared Euclidean distance between two rows of the same matrix.
fn sq_dist_rows(x: &Matrix, a: usize, b: usize) -> f64 {
    let p = x.ncols();
    let mut s = 0.0;
    for j in 0..p {
        let d = x.get(a, j) - x.get(b, j);
        s += d * d;
    }
    s
}

/// Copy row `r` of `src` into row `d` of `dst`.
fn copy_row(dst: &mut Matrix, d: usize, src: &Matrix, r: usize) {
    let p = dst.ncols().min(src.ncols());
    for j in 0..p {
        dst.set(d, j, src.get(r, j));
    }
}

/// True when every row equals the first row at working precision.
fn all_rows_identical(x: &Matrix, tol: f64) -> bool {
    let (n, p) = x.shape();
    if n <= 1 {
        return true;
    }
    for i in 1..n {
        for j in 0..p {
            if (x.get(i, j) - x.get(0, j)).abs() > tol {
                return false;
            }
        }
    }
    true
}

/// k-means++ seed (Arthur & Vassilvitskii).
fn kmeans_plus_plus(x: &Matrix, k: usize, rng: &mut Rng) -> Matrix {
    let (n, p) = x.shape();
    let k = k.max(1).min(n.max(1));
    let mut centers = Matrix::zeros(k, p);
    if n == 0 || p == 0 {
        return centers;
    }
    copy_row(&mut centers, 0, x, rng.below(n));
    let mut d2 = vec![f64::INFINITY; n];
    for c in 1..k {
        for i in 0..n {
            let d = sq_dist_rc(x, i, &centers, c - 1);
            if d < d2[i] {
                d2[i] = d;
            }
        }
        let sum: f64 = d2.iter().copied().filter(|v| v.is_finite()).sum();
        if !sum.is_finite() || sum <= 0.0 {
            copy_row(&mut centers, c, x, rng.below(n));
            continue;
        }
        let mut tick = rng.uniform() * sum;
        let mut chosen = n - 1;
        for i in 0..n {
            tick -= d2[i].max(0.0);
            if tick <= 0.0 {
                chosen = i;
                break;
            }
        }
        copy_row(&mut centers, c, x, chosen);
    }
    centers
}

/// Nearest-centroid assignment. Returns `(labels, counts, inertia)`.
fn assign_lloyd(x: &Matrix, centers: &Matrix) -> (Vector, Vec<usize>, f64) {
    let n = x.nrows();
    let k = centers.nrows();
    let mut labels = Vector::zeros(n);
    let mut counts = vec![0usize; k.max(1)];
    let mut inertia = 0.0;
    if n == 0 || k == 0 {
        return (labels, counts, 0.0);
    }
    for i in 0..n {
        let mut best = 0usize;
        let mut best_d = f64::INFINITY;
        for c in 0..k {
            let d = sq_dist_rc(x, i, centers, c);
            if d < best_d {
                best_d = d;
                best = c;
            }
        }
        labels[i] = best as f64;
        counts[best] += 1;
        inertia += best_d;
    }
    (labels, counts, inertia)
}

/// Mean of assigned rows; empty clusters keep their previous center.
fn update_means(x: &Matrix, labels: &Vector, centers: &mut Matrix, counts: &[usize]) {
    let (n, p) = x.shape();
    let k = centers.nrows();
    let mut acc = Matrix::zeros(k, p);
    for i in 0..n {
        let c = labels[i] as usize;
        if c >= k {
            continue;
        }
        for j in 0..p {
            acc.set(c, j, acc.get(c, j) + x.get(i, j));
        }
    }
    for c in 0..k {
        if counts[c] == 0 {
            continue;
        }
        let inv = 1.0 / counts[c] as f64;
        for j in 0..p {
            centers.set(c, j, acc.get(c, j) * inv);
        }
    }
}

/// Re-seed every empty centroid from a distant observation and warn.
fn reseed_empty(
    ctx: &mut FitCtx,
    x: &Matrix,
    centers: &mut Matrix,
    counts: &[usize],
    rng: &mut Rng,
) {
    let (n, k) = (x.nrows(), centers.nrows());
    if n == 0 || k == 0 {
        return;
    }
    for c in 0..k {
        if counts[c] > 0 {
            continue;
        }
        ctx.push(
            Issue::builder(IssueCode::EmptyCluster)
                .message(format!(
                    "cluster {c} received 0 points; re-seeding from a distant row"
                ))
                .metric("cluster", c as f64)
                .build(),
        );
        let mut best_i = rng.below(n);
        let mut best_d = -1.0;
        for i in 0..n {
            let mut dmin = f64::INFINITY;
            for j in 0..k {
                if counts[j] == 0 {
                    continue;
                }
                let d = sq_dist_rc(x, i, centers, j);
                if d < dmin {
                    dmin = d;
                }
            }
            if !dmin.is_finite() {
                dmin = 0.0;
            }
            if dmin > best_d {
                best_d = dmin;
                best_i = i;
            }
        }
        copy_row(centers, c, x, best_i);
    }
}

fn push_if_nonfinite_vec(ctx: &mut FitCtx, v: &Vector, what: &str) {
    if v.as_slice().iter().any(|z| !z.is_finite()) {
        ctx.push(
            Issue::builder(IssueCode::NonFiniteOutput)
                .message(format!("{what} contains NaN/Inf"))
                .build(),
        );
    }
}

fn push_if_nonfinite_mat(ctx: &mut FitCtx, m: &Matrix, what: &str) {
    for j in 0..m.ncols() {
        for i in 0..m.nrows() {
            if !m.get(i, j).is_finite() {
                ctx.push(
                    Issue::builder(IssueCode::NonFiniteOutput)
                        .message(format!("{what} contains NaN/Inf"))
                        .build(),
                );
                return;
            }
        }
    }
}

fn degenerate_cluster_issue(why: impl Into<String>) -> Issue {
    Issue::builder(IssueCode::DegenerateClusters)
        .message(why)
        .meaninglessness(Meaninglessness::vacuous(
            "cluster labels",
            "every observation is the same point; a k-way partition is not identified",
            "do not interpret centroids or cluster ids; collect variation first",
        ))
        .build()
}

/// Lloyd iteration with empty-cluster re-seeding. Returns labels, inertia, iters.
fn run_lloyd(
    ctx: &mut FitCtx,
    x: &Matrix,
    mut centers: Matrix,
    max_iter: usize,
    rng: &mut Rng,
) -> (Matrix, Vector, Vec<usize>, f64, usize) {
    let mut last_labels = Vector::zeros(x.nrows());
    let mut last_counts = vec![0usize; centers.nrows().max(1)];
    let mut last_inertia = f64::INFINITY;
    let mut used = 0usize;
    for it in 0..max_iter.max(1) {
        used = it + 1;
        let (labels, counts, inertia) = assign_lloyd(x, &centers);
        reseed_empty(ctx, x, &mut centers, &counts, rng);
        update_means(x, &labels, &mut centers, &counts);
        let mut moved = false;
        if last_labels.len() == labels.len() {
            for i in 0..labels.len() {
                if (last_labels[i] - labels[i]).abs() > 0.0 {
                    moved = true;
                    break;
                }
            }
        } else {
            moved = true;
        }
        last_labels = labels;
        last_counts = counts;
        last_inertia = inertia;
        if !moved && it > 0 {
            ctx.session.converged("lloyd_assignment_stable", it as u64);
            break;
        }
        if it + 1 == max_iter.max(1) {
            ctx.push(
                Issue::builder(IssueCode::MaxIterReached)
                    .message(format!("Lloyd iteration cap {max_iter} reached"))
                    .metric("max_iter", max_iter as f64)
                    .build(),
            );
        }
    }
    (centers, last_labels, last_counts, last_inertia, used)
}

fn unique_label_count(labels: &Vector) -> usize {
    let mut seen: Vec<i64> = Vec::new();
    for &v in labels.as_slice() {
        if !v.is_finite() {
            continue;
        }
        let k = v.round() as i64;
        if !seen.contains(&k) {
            seen.push(k);
        }
    }
    seen.len()
}

fn finish_centroid_fit(
    ctx: &mut FitCtx,
    x: &Matrix,
    centers: Matrix,
    labels: Vector,
    counts: Vec<usize>,
    inertia: f64,
    n_iter: usize,
) -> FittedKMeans {
    if all_rows_identical(x, EMPTY_RESEED_TOL) && x.nrows() > 0 {
        ctx.push(degenerate_cluster_issue(
            "all rows are identical at working precision",
        ));
    } else if unique_label_count(&labels) <= 1 && x.nrows() > 1 && centers.nrows() > 1 {
        ctx.push(
            Issue::builder(IssueCode::DegenerateClusters)
                .message("assignment collapsed to a single cluster")
                .meaninglessness(Meaninglessness::vacuous(
                    "k-way labels",
                    "only one cluster is occupied; k is not identified",
                    "reduce k or change initialization",
                ))
                .build(),
        );
    }
    push_if_nonfinite_vec(ctx, &labels, "labels");
    push_if_nonfinite_mat(ctx, &centers, "centroids");
    if !inertia.is_finite() {
        ctx.push(
            Issue::builder(IssueCode::LossIsNan)
                .message("k-means inertia is not finite")
                .build(),
        );
    }
    let count_v = Vector::from_iter(counts.iter().map(|c| *c as f64));
    FittedKMeans {
        labels,
        centroids: centers,
        inertia,
        n_iter,
        counts: count_v,
    }
}

fn empty_kmeans(k: usize, p: usize, n: usize) -> FittedKMeans {
    FittedKMeans {
        labels: Vector::zeros(n),
        centroids: Matrix::zeros(k, p),
        inertia: f64::NAN,
        n_iter: 0,
        counts: Vector::zeros(k),
    }
}

fn dummy_explain(update: u64, batch: usize, n_seen: u64) -> IncrementalExplain {
    IncrementalExplain::from_quality(
        IncrementalQuality::new(update, batch, n_seen),
        "nothing",
        "the update was rejected",
        "invalid",
        "invalid",
    )
}

fn logsumexp(xs: &[f64]) -> f64 {
    let m = xs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !m.is_finite() {
        return m;
    }
    let mut s = 0.0;
    for &v in xs {
        s += (v - m).exp();
    }
    m + s.ln()
}

/// Batch k-means with k-means++ initialization and Lloyd updates.
#[derive(Clone, Debug)]
pub(crate) struct KMeans {
    /// Number of clusters.
    pub k: usize,
    /// Lloyd iteration cap per restart.
    pub max_iter: usize,
    /// PRNG seed.
    pub seed: u64,
    /// Independent k-means++ restarts; the lowest-inertia run is kept.
    pub n_init: usize,
}

impl Default for KMeans {
    fn default() -> Self {
        Self {
            k: 8,
            max_iter: 300,
            seed: 0,
            n_init: 4,
        }
    }
}

impl KMeans {
    /// `k` clusters with default iteration / restart policy.
    pub(crate) fn new(k: usize) -> Self {
        Self {
            k,
            ..Self::default()
        }
    }

    /// Fit alias for [`FitUnsupervised::fit_unsupervised`].
    pub(crate) fn fit(&self, x: &Matrix, session: &Session) -> Result<Qualified<FittedKMeans>> {
        self.fit_unsupervised(x, session)
    }
}

/// Fitted centroid model (k-means family).
#[derive(Clone, Debug)]
pub(crate) struct FittedKMeans {
    /// Cluster id per row (`0 .. k-1` as `f64`).
    pub labels: Vector,
    /// `k` × `p` matrix of centroids.
    pub centroids: Matrix,
    /// Within-cluster sum of squared Euclidean distances.
    pub inertia: f64,
    /// Lloyd iterations used by the winning restart.
    pub n_iter: usize,
    /// Occupancy of each centroid.
    pub counts: Vector,
}

impl Predict for FittedKMeans {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        if x.ncols() != self.centroids.ncols() {
            ctx.push(
                Issue::builder(IssueCode::DimensionMismatch)
                    .message(format!(
                        "predict X is n×{} but centroids are k×{}",
                        x.ncols(),
                        self.centroids.ncols()
                    ))
                    .build(),
            );
        }
        let (labels, _, _) = assign_lloyd(x, &self.centroids);
        ctx.finish(labels)
    }
}

impl FitUnsupervised for KMeans {
    type Fitted = FittedKMeans;
    fn fit_unsupervised(&self, x: &Matrix, session: &Session) -> Result<Qualified<FittedKMeans>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        inspect_identification(&mut ctx.report, x.nrows(), self.k.max(1), &ctx.policy);
        if x.nrows() == 0 || x.ncols() == 0 {
            return ctx.finish(empty_kmeans(self.k, x.ncols(), x.nrows()));
        }
        if self.k == 0 {
            ctx.push(
                Issue::builder(IssueCode::UnidentifiedModel)
                    .message("k-means requires k ≥ 1")
                    .meaninglessness(Meaninglessness::vacuous(
                        "cluster labels",
                        "zero clusters is not a partition of the sample",
                        "set k ≥ 1",
                    ))
                    .build(),
            );
            return ctx.finish(empty_kmeans(0, x.ncols(), x.nrows()));
        }
        let k = self.k.min(x.nrows());
        if k < self.k {
            ctx.push(
                Issue::builder(IssueCode::Overparameterized)
                    .message(format!(
                        "requested k={} > n={}; clamping",
                        self.k,
                        x.nrows()
                    ))
                    .metric("k", self.k as f64)
                    .metric("n", x.nrows() as f64)
                    .build(),
            );
        }
        if all_rows_identical(x, ctx.policy.near_zero_variance) {
            ctx.push(degenerate_cluster_issue(
                "all observations are the same vector; k-means centroids are not identified",
            ));
        }
        let n_init = self.n_init.max(1);
        let mut best: Option<(f64, Matrix, Vector, Vec<usize>, usize)> = None;
        for r in 0..n_init {
            let mut rng = Rng::new(self.seed.wrapping_add(r as u64 * 17 + 1));
            let init = kmeans_plus_plus(x, k, &mut rng);
            let (centers, labels, counts, inertia, n_iter) =
                run_lloyd(&mut ctx, x, init, self.max_iter, &mut rng);
            let better = match &best {
                None => true,
                Some((b, _, _, _, _)) => inertia.is_finite() && inertia < *b,
            };
            if better {
                best = Some((inertia, centers, labels, counts, n_iter));
            }
        }
        let (inertia, centers, labels, counts, n_iter) = best.unwrap_or_else(|| {
            (
                f64::NAN,
                Matrix::zeros(k, x.ncols()),
                Vector::zeros(x.nrows()),
                vec![0; k],
                0,
            )
        });
        let fitted = finish_centroid_fit(&mut ctx, x, centers, labels, counts, inertia, n_iter);
        ctx.finish(fitted)
    }
}

/// Mini-batch k-means with mandatory incremental explainability.
#[derive(Clone, Debug)]
pub(crate) struct MiniBatchKMeans {
    /// Number of clusters.
    pub k: usize,
    /// Passes over sampled mini-batches when using [`FitUnsupervised`].
    pub max_iter: usize,
    /// PRNG seed.
    pub seed: u64,
    /// Mini-batch length.
    pub batch_size: usize,
    centroids: Option<Matrix>,
    counts: Vec<f64>,
    n_seen: u64,
    updates: u64,
    initialized: bool,
}

impl Default for MiniBatchKMeans {
    fn default() -> Self {
        Self {
            k: 8,
            max_iter: 100,
            seed: 0,
            batch_size: 32,
            centroids: None,
            counts: Vec::new(),
            n_seen: 0,
            updates: 0,
            initialized: false,
        }
    }
}

impl MiniBatchKMeans {
    /// `k` clusters.
    pub(crate) fn new(k: usize) -> Self {
        Self {
            k,
            ..Self::default()
        }
    }

    /// Current centroids, if initialized.
    pub(crate) fn centroids(&self) -> Option<&Matrix> {
        self.centroids.as_ref()
    }

    fn take_batch(x: &Matrix, idx: &[usize]) -> Matrix {
        let p = x.ncols();
        Matrix::from_fn(idx.len(), p, |i, j| x.get(idx[i], j))
    }

    fn apply_batch(
        &mut self,
        ctx: &mut FitCtx,
        batch: &Matrix,
    ) -> (Vector, f64, f64, Vec<(String, f64)>) {
        let k = self.k.max(1);
        let p = batch.ncols();
        if !self.initialized || self.centroids.as_ref().map(|c| c.ncols()) != Some(p) {
            if self.initialized && self.centroids.as_ref().map(|c| c.ncols()) != Some(p) {
                ctx.push(
                    Issue::builder(IssueCode::FeatureSpaceChangedOnline)
                        .message("mini-batch k-means saw a different column count")
                        .build(),
                );
            }
            let mut rng = Rng::new(self.seed.wrapping_add(self.updates + 1));
            let kk = k.min(batch.nrows().max(1));
            self.centroids = Some(kmeans_plus_plus(batch, kk, &mut rng));
            if kk < k {
                let mut padded = Matrix::zeros(k, p);
                if let Some(c) = &self.centroids {
                    for i in 0..c.nrows() {
                        copy_row(&mut padded, i, c, i);
                    }
                    for i in c.nrows()..k {
                        copy_row(&mut padded, i, batch, rng.below(batch.nrows().max(1)));
                    }
                }
                self.centroids = Some(padded);
            }
            self.counts = vec![0.0; k];
            self.initialized = true;
        }
        let mut centers = self
            .centroids
            .clone()
            .unwrap_or_else(|| Matrix::zeros(k, p));
        let before = centers.clone();
        let (labels, _, inertia_before) = assign_lloyd(batch, &centers);
        let mut moved = Vec::new();
        for i in 0..batch.nrows() {
            let c = labels[i] as usize;
            if c >= k {
                continue;
            }
            self.counts[c] += 1.0;
            let eta = 1.0 / self.counts[c];
            for j in 0..p {
                let v = (1.0 - eta) * centers.get(c, j) + eta * batch.get(i, j);
                centers.set(c, j, v);
            }
        }
        for c in 0..k {
            let mut d2 = 0.0;
            for j in 0..p {
                let d = centers.get(c, j) - before.get(c, j);
                d2 += d * d;
            }
            moved.push((format!("centroid[{c}]"), d2.sqrt()));
        }
        let (_, _, inertia_after) = assign_lloyd(batch, &centers);
        self.centroids = Some(centers);
        self.n_seen += batch.nrows() as u64;
        self.updates += 1;
        (labels, inertia_before, inertia_after, moved)
    }
}

impl FitUnsupervised for MiniBatchKMeans {
    type Fitted = FittedKMeans;
    fn fit_unsupervised(&self, x: &Matrix, session: &Session) -> Result<Qualified<FittedKMeans>> {
        let mut this = self.clone();
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        inspect_identification(&mut ctx.report, x.nrows(), this.k.max(1), &ctx.policy);
        if x.nrows() == 0 || x.ncols() == 0 {
            return ctx.finish(empty_kmeans(this.k, x.ncols(), x.nrows()));
        }
        let mut rng = Rng::new(this.seed);
        let bs = this.batch_size.max(1).min(x.nrows());
        for _ in 0..this.max_iter.max(1) {
            let idx = rng.sample_indices(x.nrows(), bs);
            let batch = Self::take_batch(x, &idx);
            let _ = this.apply_batch(&mut ctx, &batch);
        }
        let centers = this
            .centroids
            .clone()
            .unwrap_or_else(|| Matrix::zeros(this.k, x.ncols()));
        let (labels, counts, inertia) = assign_lloyd(x, &centers);
        let fitted =
            finish_centroid_fit(&mut ctx, x, centers, labels, counts, inertia, this.max_iter);
        ctx.finish(fitted)
    }
}

impl PartialFit for MiniBatchKMeans {
    fn partial_fit(
        &mut self,
        x: &Matrix,
        _y: Option<&Vector>,
        session: &Session,
    ) -> Result<Qualified<IncrementalExplain>> {
        let mut ctx = FitCtx::with_session(session.child("partial_fit"));
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        inspect_identification(&mut ctx.report, x.nrows(), self.k.max(1), &ctx.policy);
        if x.nrows() == 0 || x.ncols() == 0 {
            return ctx.finish(dummy_explain(self.updates, 0, self.n_seen));
        }
        let counts_before: Vec<f64> = self.counts.clone();
        let (labels, loss_b, loss_a, moved) = self.apply_batch(&mut ctx, x);
        let n_eff: f64 = self.counts.iter().sum();
        let mut assign = vec![0.0; self.k.max(1)];
        for i in 0..labels.len() {
            let c = labels[i] as usize;
            if c < assign.len() {
                assign[c] += 1.0;
            }
        }
        let delta_norm = moved.iter().map(|(_, d)| d * d).sum::<f64>().sqrt();
        let mut q = IncrementalQuality::new(self.updates.saturating_sub(1), x.nrows(), self.n_seen);
        q.effective_sample_size = n_eff;
        q.parameter_delta_norm = Some(delta_norm);
        q.parameter_delta_max = moved
            .iter()
            .map(|(_, d)| *d)
            .fold(None, |a, b| Some(a.unwrap_or(0.0).max(b)));
        q.top_moved_parameters = moved.clone();
        q.loss_before = Some(loss_b);
        q.loss_after = Some(loss_a);
        q.information_gain = Some((loss_b - loss_a).abs().max(delta_norm));
        q.still_identified = n_eff > 0.0 && self.initialized;
        q.warmup = self.n_seen < self.k as u64 * 3;
        q.explanation = format!(
            "mini-batch k-means: assignment counts {:?}; centroid L2 moves {:?}; n_eff={n_eff:.1}",
            assign, moved
        );
        if q.is_uninformative(ctx.policy.uninformative_info_eps) {
            ctx.push(
                Issue::builder(IssueCode::UpdateWithZeroInformation)
                    .incremental(q.clone())
                    .message("mini-batch did not move centroids")
                    .build(),
            );
        }
        for (c, cnt) in assign.iter().enumerate() {
            if *cnt == 0.0 && counts_before.get(c).copied().unwrap_or(0.0) == 0.0 {
                ctx.push(
                    Issue::builder(IssueCode::EmptyCluster)
                        .message(format!(
                            "cluster {c} unused in this mini-batch and historically empty"
                        ))
                        .metric("cluster", c as f64)
                        .build(),
                );
            }
        }
        let expl = IncrementalExplain::from_quality(
            q,
            format!("centroids moved: {moved:?}; assignment counts {assign:?}"),
            "online centroid update with η = 1/count[c] after nearest-centroid assignment",
            format!("batch inertia={loss_b:.6e}; counts_before={counts_before:?}"),
            format!("batch inertia={loss_a:.6e}; n_eff={n_eff:.4}"),
        )
        .contribute("n_eff", n_eff);
        ctx.session.record_incremental(expl.clone());
        ctx.finish(expl)
    }
}

/// STREAM-style weighted online k-means (chunk means + count weights).
#[derive(Clone, Debug)]
pub(crate) struct StreamKMeans {
    /// Number of clusters.
    pub k: usize,
    /// PRNG seed used for the first initializing chunk.
    pub seed: u64,
    centroids: Option<Matrix>,
    weights: Vec<f64>,
    n_seen: u64,
    updates: u64,
    initialized: bool,
}

impl Default for StreamKMeans {
    fn default() -> Self {
        Self {
            k: 8,
            seed: 0,
            centroids: None,
            weights: Vec::new(),
            n_seen: 0,
            updates: 0,
            initialized: false,
        }
    }
}

impl StreamKMeans {
    /// `k` streaming centers.
    pub(crate) fn new(k: usize) -> Self {
        Self {
            k,
            ..Self::default()
        }
    }

    /// Current centers.
    pub(crate) fn centroids(&self) -> Option<&Matrix> {
        self.centroids.as_ref()
    }

    /// Fit alias.
    pub(crate) fn fit(&self, x: &Matrix, session: &Session) -> Result<Qualified<FittedKMeans>> {
        self.fit_unsupervised(x, session)
    }
}

impl FitUnsupervised for StreamKMeans {
    type Fitted = FittedKMeans;
    fn fit_unsupervised(&self, x: &Matrix, session: &Session) -> Result<Qualified<FittedKMeans>> {
        let mut this = self.clone();
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        inspect_identification(&mut ctx.report, x.nrows(), this.k.max(1), &ctx.policy);
        if x.nrows() == 0 || x.ncols() == 0 {
            return ctx.finish(empty_kmeans(this.k, x.ncols(), x.nrows()));
        }
        let _ = this.partial_fit(x, None, &session.child("stream_init"));
        let centers = this
            .centroids
            .clone()
            .unwrap_or_else(|| Matrix::zeros(this.k, x.ncols()));
        let (labels, counts, inertia) = assign_lloyd(x, &centers);
        let fitted = finish_centroid_fit(&mut ctx, x, centers, labels, counts, inertia, 1);
        ctx.finish(fitted)
    }
}

impl PartialFit for StreamKMeans {
    fn partial_fit(
        &mut self,
        x: &Matrix,
        _y: Option<&Vector>,
        session: &Session,
    ) -> Result<Qualified<IncrementalExplain>> {
        let mut ctx = FitCtx::with_session(session.child("partial_fit"));
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        inspect_identification(&mut ctx.report, x.nrows(), self.k.max(1), &ctx.policy);
        if x.nrows() == 0 || x.ncols() == 0 {
            return ctx.finish(dummy_explain(self.updates, 0, self.n_seen));
        }
        let k = self.k.max(1);
        let p = x.ncols();
        if !self.initialized {
            let mut rng = Rng::new(self.seed | 1);
            self.centroids = Some(kmeans_plus_plus(x, k.min(x.nrows()), &mut rng));
            if let Some(c) = &self.centroids {
                if c.nrows() < k {
                    let mut padded = Matrix::zeros(k, p);
                    for i in 0..c.nrows() {
                        copy_row(&mut padded, i, c, i);
                    }
                    self.centroids = Some(padded);
                }
            }
            self.weights = vec![0.0; k];
            self.initialized = true;
        } else if self.centroids.as_ref().map(|c| c.ncols()) != Some(p) {
            ctx.push(
                Issue::builder(IssueCode::FeatureSpaceChangedOnline)
                    .message("STREAM k-means feature dimension changed")
                    .build(),
            );
            return ctx.finish(dummy_explain(self.updates, x.nrows(), self.n_seen));
        }
        let mut centers = self
            .centroids
            .clone()
            .unwrap_or_else(|| Matrix::zeros(k, p));
        let before = centers.clone();
        let (labels, assign_counts, loss_b) = assign_lloyd(x, &centers);
        let mut batch_acc = Matrix::zeros(k, p);
        for i in 0..x.nrows() {
            let c = labels[i] as usize;
            if c >= k {
                continue;
            }
            for j in 0..p {
                batch_acc.set(c, j, batch_acc.get(c, j) + x.get(i, j));
            }
        }
        let mut moved = Vec::new();
        for c in 0..k {
            let w_old = self.weights[c];
            let w_new = assign_counts[c] as f64;
            if w_new > 0.0 {
                let tot = w_old + w_new;
                for j in 0..p {
                    let mean_new = batch_acc.get(c, j) / w_new;
                    let v = if tot > 0.0 {
                        (w_old * centers.get(c, j) + w_new * mean_new) / tot
                    } else {
                        mean_new
                    };
                    centers.set(c, j, v);
                }
                self.weights[c] = tot;
            }
            let mut d2 = 0.0;
            for j in 0..p {
                let d = centers.get(c, j) - before.get(c, j);
                d2 += d * d;
            }
            moved.push((format!("centroid[{c}]"), d2.sqrt()));
            if assign_counts[c] == 0 {
                ctx.push(
                    Issue::builder(IssueCode::EmptyCluster)
                        .message(format!("STREAM chunk assigned 0 points to cluster {c}"))
                        .metric("cluster", c as f64)
                        .build(),
                );
            }
        }
        let (_, _, loss_a) = assign_lloyd(x, &centers);
        self.centroids = Some(centers);
        self.n_seen += x.nrows() as u64;
        self.updates += 1;
        let n_eff: f64 = self.weights.iter().sum();
        let delta_norm = moved.iter().map(|(_, d)| d * d).sum::<f64>().sqrt();
        let mut q = IncrementalQuality::new(self.updates - 1, x.nrows(), self.n_seen);
        q.effective_sample_size = n_eff;
        q.parameter_delta_norm = Some(delta_norm);
        q.parameter_delta_max = moved
            .iter()
            .map(|(_, d)| *d)
            .fold(None, |a, b| Some(a.unwrap_or(0.0).max(b)));
        q.top_moved_parameters = moved.clone();
        q.loss_before = Some(loss_b);
        q.loss_after = Some(loss_a);
        q.information_gain = Some((loss_b - loss_a).abs().max(delta_norm));
        q.still_identified = n_eff >= self.k as f64;
        q.warmup = self.n_seen < 10;
        q.explanation = format!(
            "STREAM k-means weighted merge: assignment {assign_counts:?}; moves {moved:?}; n_eff={n_eff:.1}"
        );
        let expl = IncrementalExplain::from_quality(
            q,
            format!("weighted centroids moved {moved:?}; chunk assignment {assign_counts:?}"),
            "STREAM-style merge of chunk means into count-weighted centers",
            format!("chunk inertia={loss_b:.6e}"),
            format!("chunk inertia={loss_a:.6e}; n_eff={n_eff:.4}"),
        )
        .contribute("n_eff", n_eff);
        ctx.session.record_incremental(expl.clone());
        ctx.finish(expl)
    }
}

impl Predict for StreamKMeans {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let Some(c) = &self.centroids else {
            ctx.push(Issue::builder(IssueCode::PartialFitBeforeInit).build());
            return ctx.finish(Vector::zeros(x.nrows()));
        };
        let (labels, _, _) = assign_lloyd(x, c);
        ctx.finish(labels)
    }
}

/// DBSCAN (Ester et al.): density-connected components plus noise (`-1`).
#[derive(Clone, Debug)]
pub(crate) struct Dbscan {
    /// Neighborhood radius.
    pub eps: f64,
    /// Minimum neighborhood size (including the point itself) to be a core point.
    pub min_samples: usize,
}

impl Default for Dbscan {
    fn default() -> Self {
        Self {
            eps: 0.5,
            min_samples: 5,
        }
    }
}

impl Dbscan {
    /// DBSCAN with the given radius and core-point threshold.
    pub(crate) fn new(eps: f64, min_samples: usize) -> Self {
        Self { eps, min_samples }
    }

    /// Fit alias.
    pub(crate) fn fit(&self, x: &Matrix, session: &Session) -> Result<Qualified<FittedDbscan>> {
        self.fit_unsupervised(x, session)
    }
}

/// Fitted DBSCAN partition.
#[derive(Clone, Debug)]
pub(crate) struct FittedDbscan {
    /// Cluster ids; noise is `-1.0`.
    pub labels: Vector,
    /// Number of non-noise clusters.
    pub n_clusters: usize,
    /// `1` if the row is a core point, else `0`.
    pub core: Vector,
}

impl FitUnsupervised for Dbscan {
    type Fitted = FittedDbscan;
    fn fit_unsupervised(&self, x: &Matrix, session: &Session) -> Result<Qualified<FittedDbscan>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        inspect_identification(&mut ctx.report, x.nrows(), 1, &ctx.policy);
        let n = x.nrows();
        if n == 0 || x.ncols() == 0 {
            return ctx.finish(FittedDbscan {
                labels: Vector::zeros(0),
                n_clusters: 0,
                core: Vector::zeros(0),
            });
        }
        if !(self.eps.is_finite() && self.eps > 0.0) {
            ctx.push(
                Issue::builder(IssueCode::InvalidWeight)
                    .message(format!(
                        "DBSCAN eps={} is not a positive finite radius",
                        self.eps
                    ))
                    .build(),
            );
            return ctx.finish(FittedDbscan {
                labels: Vector::filled(n, -1.0),
                n_clusters: 0,
                core: Vector::zeros(n),
            });
        }
        let eps2 = self.eps * self.eps;
        let min_s = self.min_samples.max(1);
        let mut neigh: Vec<Vec<usize>> = vec![Vec::new(); n];
        for i in 0..n {
            for j in i..n {
                if sq_dist_rows(x, i, j) <= eps2 {
                    neigh[i].push(j);
                    if i != j {
                        neigh[j].push(i);
                    }
                }
            }
        }
        let mut labels = vec![-2i64; n]; // -2 unseen, -1 noise
        let mut core = Vector::zeros(n);
        for i in 0..n {
            if neigh[i].len() >= min_s {
                core[i] = 1.0;
            }
        }
        let mut cid = 0i64;
        for i in 0..n {
            if labels[i] != -2 {
                continue;
            }
            if core[i] < 0.5 {
                labels[i] = -1;
                continue;
            }
            let mut stack = vec![i];
            labels[i] = cid;
            while let Some(p) = stack.pop() {
                for &q in &neigh[p] {
                    if labels[q] == -1 {
                        labels[q] = cid;
                    }
                    if labels[q] != -2 {
                        continue;
                    }
                    labels[q] = cid;
                    if core[q] > 0.5 {
                        stack.push(q);
                    }
                }
            }
            cid += 1;
        }
        let n_clusters = cid as usize;
        if n_clusters == 0 {
            ctx.push(degenerate_cluster_issue(
                "DBSCAN produced only noise; no core point exists at this (eps, min_samples)",
            ));
        }
        if all_rows_identical(x, ctx.policy.near_zero_variance) && n > 1 {
            ctx.push(degenerate_cluster_issue("DBSCAN input rows are identical"));
        }
        let lab = Vector::from_iter(labels.iter().map(|v| *v as f64));
        push_if_nonfinite_vec(&mut ctx, &lab, "dbscan labels");
        ctx.finish(FittedDbscan {
            labels: lab,
            n_clusters,
            core,
        })
    }
}

/// HDBSCAN-lite: mutual-reachability MST, then a single longest-edge split
/// (Campello / McInnes extraction reduced to the dominant cut).
///
/// Core distance is the distance to the `min_samples`-th neighbour. Mutual
/// reachability is \(\max(d_k(i), d_k(j), d(i,j))\). Components smaller than
/// `min_cluster_size` are labelled noise (`-1`). Cluster count is **not**
/// passed to [`inspect_identification`].
#[derive(Clone, Debug)]
pub(crate) struct Hdbscan {
    /// Neighbours used for the core distance.
    pub min_samples: usize,
    /// Minimum component size to keep as a cluster.
    pub min_cluster_size: usize,
}

impl Default for Hdbscan {
    fn default() -> Self {
        Self {
            min_samples: 5,
            min_cluster_size: 5,
        }
    }
}

impl Hdbscan {
    /// HDBSCAN with the given core-distance and cluster-size floors.
    pub(crate) fn new(min_samples: usize, min_cluster_size: usize) -> Self {
        Self {
            min_samples: min_samples.max(1),
            min_cluster_size: min_cluster_size.max(1),
        }
    }

    /// Fit alias.
    pub(crate) fn fit(&self, x: &Matrix, session: &Session) -> Result<Qualified<FittedHdbscan>> {
        self.fit_unsupervised(x, session)
    }
}

/// Fitted HDBSCAN partition.
#[derive(Clone, Debug)]
pub(crate) struct FittedHdbscan {
    /// Cluster ids; noise is `-1.0`.
    pub labels: Vector,
    /// Number of non-noise clusters.
    pub n_clusters: usize,
}

impl FitUnsupervised for Hdbscan {
    type Fitted = FittedHdbscan;
    fn fit_unsupervised(&self, x: &Matrix, session: &Session) -> Result<Qualified<FittedHdbscan>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        inspect_identification(&mut ctx.report, x.nrows(), 1, &ctx.policy);
        let n = x.nrows();
        if n == 0 || x.ncols() == 0 {
            return ctx.finish(FittedHdbscan {
                labels: Vector::zeros(0),
                n_clusters: 0,
            });
        }
        if all_rows_identical(x, ctx.policy.near_zero_variance) && n > 1 {
            ctx.push(degenerate_cluster_issue("HDBSCAN input rows are identical"));
            return ctx.finish(FittedHdbscan {
                labels: Vector::zeros(n),
                n_clusters: 1,
            });
        }
        let k = self.min_samples.max(1).min(n.saturating_sub(1).max(1));
        let mut core = vec![0.0f64; n];
        let mut dist = vec![vec![0.0f64; n]; n];
        for i in 0..n {
            let mut row = Vec::with_capacity(n);
            for j in 0..n {
                let d = sq_dist_rows(x, i, j).sqrt();
                dist[i][j] = d;
                if i != j {
                    row.push(d);
                }
            }
            row.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            core[i] = row.get(k.saturating_sub(1)).copied().unwrap_or(0.0);
        }
        let mut mrd = vec![vec![0.0f64; n]; n];
        for i in 0..n {
            for j in i..n {
                let v = core[i].max(core[j]).max(dist[i][j]);
                mrd[i][j] = v;
                mrd[j][i] = v;
            }
        }
        // Prim MST on mutual reachability.
        let mut in_tree = vec![false; n];
        let mut parent = vec![0usize; n];
        let mut best = vec![f64::INFINITY; n];
        best[0] = 0.0;
        let mut edges: Vec<(usize, usize, f64)> = Vec::new();
        for _ in 0..n {
            let mut u = 0usize;
            let mut bu = f64::INFINITY;
            for i in 0..n {
                if !in_tree[i] && best[i] < bu {
                    bu = best[i];
                    u = i;
                }
            }
            in_tree[u] = true;
            if bu.is_finite() && bu > 0.0 {
                edges.push((parent[u], u, bu));
            }
            for v in 0..n {
                if !in_tree[v] && mrd[u][v] < best[v] {
                    best[v] = mrd[u][v];
                    parent[v] = u;
                }
            }
        }
        // Remove the longest MST edge if both sides meet min_cluster_size.
        edges.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        let min_c = self.min_cluster_size.max(1);
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
        for &(a, b, _) in &edges {
            adj[a].push(b);
            adj[b].push(a);
        }
        let mut labels = vec![-1i64; n];
        let mut n_clusters = 0usize;
        if let Some(&(a, b, _)) = edges.first() {
            // Drop the longest edge and grow the two sides.
            adj[a].retain(|&v| v != b);
            adj[b].retain(|&v| v != a);
            for (seed, cid) in [(a, 0i64), (b, 1i64)] {
                let mut stack = vec![seed];
                let mut comp = Vec::new();
                let mut seen = vec![false; n];
                seen[seed] = true;
                while let Some(p) = stack.pop() {
                    comp.push(p);
                    for &q in &adj[p] {
                        if !seen[q] {
                            seen[q] = true;
                            stack.push(q);
                        }
                    }
                }
                if comp.len() >= min_c {
                    for i in comp {
                        labels[i] = cid;
                    }
                    n_clusters += 1;
                }
            }
        }
        if n_clusters == 0 {
            // Fallback: one cluster if the whole sample is large enough.
            if n >= min_c {
                for i in 0..n {
                    labels[i] = 0;
                }
                n_clusters = 1;
                ctx.push(
                    Issue::builder(IssueCode::EmptyCluster)
                        .message("HDBSCAN found no stable cut; the sample is one cluster")
                        .build(),
                );
            } else {
                ctx.push(degenerate_cluster_issue(
                    "HDBSCAN produced only noise at this min_cluster_size",
                ));
            }
        }
        let lab = Vector::from_iter(labels.iter().map(|v| *v as f64));
        push_if_nonfinite_vec(&mut ctx, &lab, "hdbscan labels");
        ctx.finish(FittedHdbscan {
            labels: lab,
            n_clusters,
        })
    }
}

/// Hierarchical linkage rule on pairwise Euclidean distances.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Linkage {
    /// Unweighted average of cross-pair distances (UPGMA).
    Average,
    /// Maximum cross-pair distance (complete / farthest neighbor).
    Complete,
    /// Minimum cross-pair distance (single / nearest neighbor).
    Single,
    /// Variance-minimizing Ward merge (size-weighted squared mean gap).
    Ward,
}

/// Agglomerative clustering on the full Euclidean distance matrix.
#[derive(Clone, Debug)]
pub(crate) struct Agglomerative {
    /// Requested number of clusters.
    pub n_clusters: usize,
    /// Linkage.
    pub linkage: Linkage,
}

impl Default for Agglomerative {
    fn default() -> Self {
        Self {
            n_clusters: 2,
            linkage: Linkage::Average,
        }
    }
}

impl Agglomerative {
    /// `n_clusters` with average linkage.
    pub(crate) fn new(n_clusters: usize) -> Self {
        Self {
            n_clusters,
            linkage: Linkage::Average,
        }
    }

    /// Fit alias.
    pub(crate) fn fit(
        &self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedAgglomerative>> {
        self.fit_unsupervised(x, session)
    }
}

/// Named sklearn `AgglomerativeClustering` (default Ward linkage).
#[derive(Clone, Debug)]
pub(crate) struct AgglomerativeClustering {
    inner: Agglomerative,
}

impl Default for AgglomerativeClustering {
    fn default() -> Self {
        Self {
            inner: Agglomerative {
                n_clusters: 2,
                linkage: Linkage::Ward,
            },
        }
    }
}

impl AgglomerativeClustering {
    /// Ward agglomeration into `n_clusters` groups.
    pub(crate) fn new(n_clusters: usize) -> Self {
        Self {
            inner: Agglomerative {
                n_clusters: n_clusters.max(1),
                linkage: Linkage::Ward,
            },
        }
    }

    /// Fit alias.
    pub(crate) fn fit(
        &self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedAgglomerative>> {
        self.inner.fit(x, session)
    }
}

impl FitUnsupervised for AgglomerativeClustering {
    type Fitted = FittedAgglomerative;
    fn fit_unsupervised(
        &self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedAgglomerative>> {
        self.inner.fit_unsupervised(x, session)
    }
}

/// Fitted agglomerative partition.
#[derive(Clone, Debug)]
pub(crate) struct FittedAgglomerative {
    /// Cluster id per row.
    pub labels: Vector,
    /// Linkage that produced the tree.
    pub linkage: Linkage,
}

fn cluster_link(a: &[usize], b: &[usize], dist: &Matrix, linkage: Linkage) -> f64 {
    match linkage {
        Linkage::Single => {
            let mut m = f64::INFINITY;
            for &i in a {
                for &j in b {
                    m = m.min(dist.get(i, j));
                }
            }
            m
        }
        Linkage::Complete => {
            let mut m = f64::NEG_INFINITY;
            for &i in a {
                for &j in b {
                    m = m.max(dist.get(i, j));
                }
            }
            m
        }
        Linkage::Average => {
            let mut s = 0.0;
            let mut c = 0.0;
            for &i in a {
                for &j in b {
                    s += dist.get(i, j);
                    c += 1.0;
                }
            }
            if c == 0.0 {
                f64::INFINITY
            } else {
                s / c
            }
        }
        Linkage::Ward => {
            let na = a.len() as f64;
            let nb = b.len() as f64;
            let mut s = 0.0;
            let mut c = 0.0;
            for &i in a {
                for &j in b {
                    s += dist.get(i, j);
                    c += 1.0;
                }
            }
            if c == 0.0 {
                f64::INFINITY
            } else {
                let d = s / c;
                (na * nb / (na + nb).max(1.0)) * d * d
            }
        }
    }
}

impl FitUnsupervised for Agglomerative {
    type Fitted = FittedAgglomerative;
    fn fit_unsupervised(
        &self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedAgglomerative>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        inspect_identification(
            &mut ctx.report,
            x.nrows(),
            self.n_clusters.max(1),
            &ctx.policy,
        );
        let n = x.nrows();
        if n == 0 || x.ncols() == 0 {
            return ctx.finish(FittedAgglomerative {
                labels: Vector::zeros(0),
                linkage: self.linkage,
            });
        }
        if all_rows_identical(x, ctx.policy.near_zero_variance) {
            ctx.push(degenerate_cluster_issue(
                "agglomerative input is a single repeated point",
            ));
        }
        let mut want = self.n_clusters.max(1);
        if want > n {
            ctx.push(
                Issue::builder(IssueCode::Overparameterized)
                    .message(format!("n_clusters={want} > n={n}"))
                    .build(),
            );
            want = n;
        }
        let dist = Matrix::from_fn(n, n, |i, j| {
            if i == j {
                0.0
            } else {
                sq_dist_rows(x, i, j).sqrt()
            }
        });
        let mut clusters: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();
        while clusters.len() > want {
            let mut bi = 0usize;
            let mut bj = 1usize;
            let mut best = f64::INFINITY;
            for i in 0..clusters.len() {
                for j in (i + 1)..clusters.len() {
                    let d = cluster_link(&clusters[i], &clusters[j], &dist, self.linkage);
                    if d < best {
                        best = d;
                        bi = i;
                        bj = j;
                    }
                }
            }
            if !best.is_finite() {
                ctx.push(
                    Issue::builder(IssueCode::NonFiniteOutput)
                        .message("agglomerative linkage produced a non-finite merge distance")
                        .build(),
                );
                break;
            }
            let mut merged = clusters[bi].clone();
            merged.extend_from_slice(&clusters[bj]);
            if bi > bj {
                clusters.remove(bi);
                clusters.remove(bj);
            } else {
                clusters.remove(bj);
                clusters.remove(bi);
            }
            clusters.push(merged);
        }
        let mut labels = Vector::zeros(n);
        for (c, members) in clusters.iter().enumerate() {
            if members.len() == 1 {
                ctx.push(
                    Issue::builder(IssueCode::SinglePointCluster)
                        .message(format!("agglomerative cluster {c} is a singleton"))
                        .metric("cluster", c as f64)
                        .build(),
                );
            }
            for &i in members {
                labels[i] = c as f64;
            }
        }
        ctx.finish(FittedAgglomerative {
            labels,
            linkage: self.linkage,
        })
    }
}

/// Diagonal-covariance Gaussian mixture via EM.
#[derive(Clone, Debug)]
pub(crate) struct GaussianMixture {
    /// Number of mixture components.
    pub n_components: usize,
    /// EM iteration cap.
    pub max_iter: usize,
    /// PRNG seed for k-means++ mean initialization.
    pub seed: u64,
}

impl Default for GaussianMixture {
    fn default() -> Self {
        Self {
            n_components: 2,
            max_iter: 100,
            seed: 0,
        }
    }
}

impl GaussianMixture {
    /// `k` components.
    pub(crate) fn new(n_components: usize) -> Self {
        Self {
            n_components,
            ..Self::default()
        }
    }

    /// Fit alias.
    pub(crate) fn fit(&self, x: &Matrix, session: &Session) -> Result<Qualified<FittedGmm>> {
        self.fit_unsupervised(x, session)
    }
}

/// Fitted diagonal Gaussian mixture.
#[derive(Clone, Debug)]
pub(crate) struct FittedGmm {
    /// Hard labels (`argmax` responsibility).
    pub labels: Vector,
    /// Mixing weights (length `k`).
    pub weights: Vector,
    /// Component means (`k` × `p`).
    pub means: Matrix,
    /// Diagonal variances (`k` × `p`).
    pub covariances: Matrix,
    /// Final average log-likelihood.
    pub loglik: f64,
}

fn diag_gauss_logpdf(x: &Matrix, i: usize, mean: &Matrix, c: usize, var: &Matrix) -> f64 {
    let p = x.ncols().min(mean.ncols()).min(var.ncols());
    let mut s = 0.0;
    let ln2pi = (2.0 * std::f64::consts::PI).ln();
    for j in 0..p {
        let v = var.get(c, j).max(COV_FLOOR);
        let d = x.get(i, j) - mean.get(c, j);
        s += d * d / v + v.ln() + ln2pi;
    }
    -0.5 * s
}

impl FitUnsupervised for GaussianMixture {
    type Fitted = FittedGmm;
    fn fit_unsupervised(&self, x: &Matrix, session: &Session) -> Result<Qualified<FittedGmm>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        inspect_identification(
            &mut ctx.report,
            x.nrows(),
            self.n_components.max(1),
            &ctx.policy,
        );
        let (n, p) = x.shape();
        let k = self.n_components.max(1).min(n.max(1));
        if n == 0 || p == 0 {
            return ctx.finish(FittedGmm {
                labels: Vector::zeros(0),
                weights: Vector::zeros(self.n_components),
                means: Matrix::zeros(self.n_components, p),
                covariances: Matrix::zeros(self.n_components, p),
                loglik: f64::NAN,
            });
        }
        if all_rows_identical(x, ctx.policy.near_zero_variance) {
            ctx.push(degenerate_cluster_issue(
                "GMM data have zero spread; component covariances collapse",
            ));
            ctx.push(
                Issue::builder(IssueCode::EmissionDegenerate)
                    .message("every row is identical; Gaussian emissions have zero variance")
                    .meaninglessness(Meaninglessness::vacuous(
                        "mixture parameters",
                        "a point-mass sample does not identify a covariance",
                        "do not interpret component scales",
                    ))
                    .build(),
            );
        }
        if k < self.n_components {
            ctx.push(
                Issue::builder(IssueCode::Overparameterized)
                    .message("n_components exceeds n; clamping")
                    .build(),
            );
        }
        let mut rng = Rng::new(self.seed | 1);
        let mut means = kmeans_plus_plus(x, k, &mut rng);
        let mut weights = Vector::filled(k, 1.0 / k as f64);
        let mut vars = Matrix::zeros(k, p);
        for j in 0..p {
            let col = x.column(j);
            let v = col.std().max(COV_FLOOR);
            for c in 0..k {
                vars.set(c, j, v * v);
            }
        }
        let mut loglik = f64::NEG_INFINITY;
        let mut resp = vec![vec![0.0; k]; n];
        for it in 0..self.max_iter.max(1) {
            let mut ll = 0.0;
            for i in 0..n {
                let mut logp = vec![0.0; k];
                for c in 0..k {
                    let lw = if weights[c] > 0.0 {
                        weights[c].ln()
                    } else {
                        f64::NEG_INFINITY
                    };
                    logp[c] = lw + diag_gauss_logpdf(x, i, &means, c, &vars);
                }
                let lse = logsumexp(&logp);
                ll += lse;
                for c in 0..k {
                    resp[i][c] = if lse.is_finite() {
                        (logp[c] - lse).exp()
                    } else {
                        1.0 / k as f64
                    };
                }
            }
            loglik = ll / n as f64;
            ctx.session.step(it as u64, -loglik, None);
            let mut nk = vec![0.0; k];
            for i in 0..n {
                for c in 0..k {
                    nk[c] += resp[i][c];
                }
            }
            for c in 0..k {
                if nk[c] <= 1e-8 {
                    ctx.push(
                        Issue::builder(IssueCode::MixtureWeightCollapsed)
                            .message(format!("component {c} weight collapsed; reseeding mean"))
                            .metric("component", c as f64)
                            .metric("n_eff", nk[c])
                            .build(),
                    );
                    copy_row(&mut means, c, x, rng.below(n));
                    weights[c] = 1.0 / k as f64;
                    for j in 0..p {
                        vars.set(c, j, vars.get(c, j).max(COV_FLOOR) * 2.0);
                    }
                    continue;
                }
                weights[c] = nk[c] / n as f64;
                for j in 0..p {
                    let mut m = 0.0;
                    for i in 0..n {
                        m += resp[i][c] * x.get(i, j);
                    }
                    means.set(c, j, m / nk[c]);
                }
                for j in 0..p {
                    let mut s = 0.0;
                    for i in 0..n {
                        let d = x.get(i, j) - means.get(c, j);
                        s += resp[i][c] * d * d;
                    }
                    let raw = s / nk[c];
                    if raw <= ctx.policy.near_zero_variance {
                        ctx.push(
                            Issue::builder(IssueCode::EmissionDegenerate)
                                .message(format!(
                                    "component {c} feature {j} variance {raw:.3e} collapsed"
                                ))
                                .metric("component", c as f64)
                                .metric("feature", j as f64)
                                .meaninglessness(Meaninglessness::vacuous(
                                    "component covariance",
                                    "a collapsed Gaussian is a hard assignment, not a density",
                                    "drop the component or add a covariance floor and stop interpreting scale",
                                ))
                                .build(),
                        );
                    }
                    vars.set(c, j, raw.max(COV_FLOOR));
                }
                if nk[c] < 1.5 {
                    ctx.push(
                        Issue::builder(IssueCode::SinglePointCluster)
                            .message(format!("component {c} effective count {nk:.3}", nk = nk[c]))
                            .metric("component", c as f64)
                            .build(),
                    );
                }
            }
            let wsum: f64 = (0..k).map(|c| weights[c]).sum();
            if wsum > 0.0 {
                for c in 0..k {
                    weights[c] /= wsum;
                }
            }
            if it + 1 == self.max_iter {
                ctx.push(
                    Issue::builder(IssueCode::MaxIterReached)
                        .message("GMM EM hit max_iter")
                        .build(),
                );
            }
        }
        let mut labels = Vector::zeros(n);
        for i in 0..n {
            let mut b = 0usize;
            let mut bv = f64::NEG_INFINITY;
            for c in 0..k {
                if resp[i][c] > bv {
                    bv = resp[i][c];
                    b = c;
                }
            }
            labels[i] = b as f64;
        }
        push_if_nonfinite_vec(&mut ctx, &labels, "gmm labels");
        push_if_nonfinite_mat(&mut ctx, &means, "gmm means");
        if !loglik.is_finite() {
            ctx.push(
                Issue::builder(IssueCode::LossIsNan)
                    .message("GMM log-likelihood is not finite")
                    .build(),
            );
        }
        ctx.finish(FittedGmm {
            labels,
            weights,
            means,
            covariances: vars,
            loglik,
        })
    }
}

impl Predict for FittedGmm {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        if x.ncols() != self.means.ncols() {
            ctx.push(
                Issue::builder(IssueCode::DimensionMismatch)
                    .message("GMM predict column count does not match the fitted means")
                    .build(),
            );
        }
        let k = self.means.nrows();
        let mut labels = Vector::zeros(x.nrows());
        for i in 0..x.nrows() {
            let mut b = 0usize;
            let mut bv = f64::NEG_INFINITY;
            for c in 0..k {
                let lp = self.weights[c].max(1e-300).ln()
                    + diag_gauss_logpdf(x, i, &self.means, c, &self.covariances);
                if lp > bv {
                    bv = lp;
                    b = c;
                }
            }
            labels[i] = b as f64;
        }
        ctx.finish(labels)
    }
}

/// Variational / MAP diagonal Bayesian Gaussian mixture (sklearn
/// `BayesianGaussianMixture`).
///
/// Weights have a Dirichlet prior; means shrink toward the data mean. This is
/// a mean-field MAP-EM, not a full collapsed Gibbs sampler — recorded as a
/// numerical compromise. Do not pass `n_components` as `p` to
/// [`inspect_identification`]: a 2-component mixture on 40 rows is identified.
#[derive(Clone, Debug)]
pub(crate) struct BayesianGaussianMixture {
    /// Number of mixture components.
    pub n_components: usize,
    /// Dirichlet concentration (`α₀`).
    pub weight_concentration: f64,
    /// EM iteration cap.
    pub max_iter: usize,
    /// PRNG seed.
    pub seed: u64,
}

impl Default for BayesianGaussianMixture {
    fn default() -> Self {
        Self {
            n_components: 2,
            weight_concentration: 1.0,
            max_iter: 80,
            seed: 2,
        }
    }
}

impl BayesianGaussianMixture {
    /// `k` components with `α₀ = 1`.
    pub(crate) fn new(n_components: usize) -> Self {
        Self {
            n_components,
            ..Self::default()
        }
    }

    /// Fit alias.
    pub(crate) fn fit(&self, x: &Matrix, session: &Session) -> Result<Qualified<FittedGmm>> {
        self.fit_unsupervised(x, session)
    }
}

impl FitUnsupervised for BayesianGaussianMixture {
    type Fitted = FittedGmm;
    fn fit_unsupervised(&self, x: &Matrix, session: &Session) -> Result<Qualified<FittedGmm>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let (n, p) = x.shape();
        let k = self.n_components.max(1).min(n.max(1));
        if n == 0 || p == 0 {
            return ctx.finish(FittedGmm {
                labels: Vector::zeros(0),
                weights: Vector::zeros(self.n_components),
                means: Matrix::zeros(self.n_components, p),
                covariances: Matrix::zeros(self.n_components, p),
                loglik: f64::NAN,
            });
        }
        if all_rows_identical(x, ctx.policy.near_zero_variance) {
            ctx.push(degenerate_cluster_issue(
                "Bayesian GMM data have zero spread; component covariances collapse",
            ));
        }
        if k < self.n_components {
            ctx.push(
                Issue::builder(IssueCode::Overparameterized)
                    .message("BayesianGaussianMixture n_components exceeds n; clamping")
                    .build(),
            );
        }
        ctx.push(
            Issue::builder(IssueCode::JitterInjected)
                .severity(Severity::Advisory)
                .message("BayesianGaussianMixture uses MAP-EM, not a full variational posterior")
                .build(),
        );
        let alpha0 = self.weight_concentration.max(1e-3);
        let mut rng = Rng::new(self.seed | 5);
        let mut means = kmeans_plus_plus(x, k, &mut rng);
        let mut weights = Vector::filled(k, 1.0 / k as f64);
        let mut vars = Matrix::zeros(k, p);
        let mut prior_mean = Vector::zeros(p);
        for j in 0..p {
            let col = x.column(j);
            prior_mean[j] = col.mean();
            let v = col.std().max(COV_FLOOR);
            for c in 0..k {
                vars.set(c, j, v * v);
            }
        }
        let mut loglik = f64::NEG_INFINITY;
        let mut resp = vec![vec![0.0; k]; n];
        for it in 0..self.max_iter.max(1) {
            let mut ll = 0.0;
            for i in 0..n {
                let mut logp = vec![0.0; k];
                for c in 0..k {
                    let lw = if weights[c] > 0.0 {
                        weights[c].ln()
                    } else {
                        f64::NEG_INFINITY
                    };
                    logp[c] = lw + diag_gauss_logpdf(x, i, &means, c, &vars);
                }
                let lse = logsumexp(&logp);
                ll += lse;
                for c in 0..k {
                    resp[i][c] = if lse.is_finite() {
                        (logp[c] - lse).exp()
                    } else {
                        1.0 / k as f64
                    };
                }
            }
            loglik = ll / n as f64;
            ctx.session.step(it as u64, -loglik, None);
            let mut nk = vec![0.0; k];
            for i in 0..n {
                for c in 0..k {
                    nk[c] += resp[i][c];
                }
            }
            let den = n as f64 + k as f64 * alpha0;
            for c in 0..k {
                weights[c] = (nk[c] + alpha0) / den;
                if nk[c] <= 1e-8 {
                    ctx.push(
                        Issue::builder(IssueCode::MixtureWeightCollapsed)
                            .severity(Severity::Warning)
                            .message(format!("Bayesian GMM component {c} weight collapsed"))
                            .metric("component", c as f64)
                            .build(),
                    );
                    copy_row(&mut means, c, x, rng.below(n));
                    continue;
                }
                let shrink = nk[c] / (nk[c] + 1.0);
                for j in 0..p {
                    let mut m = 0.0;
                    for i in 0..n {
                        m += resp[i][c] * x.get(i, j);
                    }
                    let mle = m / nk[c];
                    means.set(c, j, shrink * mle + (1.0 - shrink) * prior_mean[j]);
                }
                for j in 0..p {
                    let mut s = 0.0;
                    for i in 0..n {
                        let d = x.get(i, j) - means.get(c, j);
                        s += resp[i][c] * d * d;
                    }
                    vars.set(c, j, (s / nk[c]).max(COV_FLOOR));
                }
            }
            let wsum: f64 = (0..k).map(|c| weights[c]).sum();
            if wsum > 0.0 {
                for c in 0..k {
                    weights[c] /= wsum;
                }
            }
        }
        let mut labels = Vector::zeros(n);
        for i in 0..n {
            let mut b = 0usize;
            let mut bv = f64::NEG_INFINITY;
            for c in 0..k {
                if resp[i][c] > bv {
                    bv = resp[i][c];
                    b = c;
                }
            }
            labels[i] = b as f64;
        }
        push_if_nonfinite_vec(&mut ctx, &labels, "bgmm labels");
        ctx.finish(FittedGmm {
            labels,
            weights,
            means,
            covariances: vars,
            loglik,
        })
    }
}

/// Spectral clustering: Gaussian affinity → Laplacian eigenmap → k-means.
#[derive(Clone, Debug)]
pub(crate) struct SpectralClustering {
    /// Number of clusters / eigenvectors kept.
    pub n_clusters: usize,
    /// Seed for the k-means stage.
    pub seed: u64,
    /// RBF coefficient `γ` in `exp(−γ‖x−y‖²)`. `None` uses `1 / median ‖x−y‖²`.
    pub gamma: Option<f64>,
}

impl Default for SpectralClustering {
    fn default() -> Self {
        Self {
            n_clusters: 2,
            seed: 0,
            gamma: None,
        }
    }
}

impl SpectralClustering {
    /// `n_clusters` with automatic RBF bandwidth.
    pub(crate) fn new(n_clusters: usize) -> Self {
        Self {
            n_clusters,
            ..Self::default()
        }
    }

    /// Fit alias.
    pub(crate) fn fit(&self, x: &Matrix, session: &Session) -> Result<Qualified<FittedSpectral>> {
        self.fit_unsupervised(x, session)
    }
}

/// Fitted spectral embedding and labels.
#[derive(Clone, Debug)]
pub(crate) struct FittedSpectral {
    /// Cluster id per row.
    pub labels: Vector,
    /// Rows are the Laplacian eigenvectors used as features.
    pub embedding: Matrix,
}

impl FitUnsupervised for SpectralClustering {
    type Fitted = FittedSpectral;
    fn fit_unsupervised(&self, x: &Matrix, session: &Session) -> Result<Qualified<FittedSpectral>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        inspect_identification(
            &mut ctx.report,
            x.nrows(),
            self.n_clusters.max(1),
            &ctx.policy,
        );
        let n = x.nrows();
        if n == 0 || x.ncols() == 0 {
            return ctx.finish(FittedSpectral {
                labels: Vector::zeros(0),
                embedding: Matrix::zeros(0, self.n_clusters),
            });
        }
        if all_rows_identical(x, ctx.policy.near_zero_variance) {
            ctx.push(degenerate_cluster_issue(
                "spectral clustering on identical rows; the Laplacian is unidentified",
            ));
        }
        let mut d2s = Vec::with_capacity(n * n / 2 + 1);
        for i in 0..n {
            for j in (i + 1)..n {
                d2s.push(sq_dist_rows(x, i, j));
            }
        }
        d2s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let med = if d2s.is_empty() {
            1.0
        } else {
            d2s[d2s.len() / 2].max(1e-12)
        };
        let gamma = self
            .gamma
            .filter(|g| g.is_finite() && *g > 0.0)
            .unwrap_or(1.0 / med);
        let w = Matrix::from_fn(n, n, |i, j| {
            if i == j {
                0.0
            } else {
                (-gamma * sq_dist_rows(x, i, j)).exp()
            }
        });
        let mut deg = vec![0.0; n];
        for i in 0..n {
            let mut s = 0.0;
            for j in 0..n {
                s += w.get(i, j);
            }
            if s <= ctx.policy.near_zero_variance {
                ctx.push(
                    Issue::builder(IssueCode::EmptyCluster)
                        .message(format!(
                            "row {i} has degree {s:.3e}; isolated in the affinity graph"
                        ))
                        .metric("row", i as f64)
                        .build(),
                );
            }
            deg[i] = s.max(1e-12);
        }
        // Normalized Laplacian L = I − D^{-1/2} W D^{-1/2}.
        let mut l = Mat::<f64>::zeros(n, n);
        for i in 0..n {
            let di = deg[i].sqrt();
            for j in 0..n {
                let dj = deg[j].sqrt();
                let off = w.get(i, j) / (di * dj);
                l[(i, j)] = if i == j { 1.0 - off } else { -off };
            }
        }
        let Some((vals, vecs)) = symmetric_eigen(&mut ctx.report, &l, &ctx.policy) else {
            ctx.push(
                Issue::builder(IssueCode::EigenDidNotConverge)
                    .message("Laplacian eigensolve failed")
                    .build(),
            );
            return ctx.finish(FittedSpectral {
                labels: Vector::zeros(n),
                embedding: Matrix::zeros(n, self.n_clusters),
            });
        };
        let mut order: Vec<usize> = (0..vals.len()).collect();
        order.sort_by(|&a, &b| {
            vals[a]
                .partial_cmp(&vals[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        for &i in &order {
            if vals[i] < -ctx.policy.rank_tol_relative {
                ctx.push(
                    Issue::builder(IssueCode::NegativeEigenvalueDropped)
                        .message(format!("Laplacian eigenvalue {ev:.3e} < 0", ev = vals[i]))
                        .metric("eigenvalue", vals[i])
                        .build(),
                );
            }
        }
        let k = self.n_clusters.max(1).min(n).min(vals.len());
        if k < self.n_clusters {
            ctx.push(
                Issue::builder(IssueCode::ComponentsExceedRank)
                    .message("requested more spectral components than Laplacian rank")
                    .build(),
            );
        }
        let embedding = Matrix::from_fn(n, k, |i, c| {
            let col = order[c];
            if i < vecs.nrows() && col < vecs.ncols() {
                vecs[(i, col)]
            } else {
                0.0
            }
        });
        // Row-normalize the embedding (Ng–Jordan–Weiss).
        let mut emb = embedding.clone();
        for i in 0..n {
            let mut nrm = 0.0;
            for c in 0..k {
                nrm += emb.get(i, c) * emb.get(i, c);
            }
            nrm = nrm.sqrt();
            if nrm > 0.0 {
                for c in 0..k {
                    emb.set(i, c, emb.get(i, c) / nrm);
                }
            }
        }
        let mut rng = Rng::new(self.seed | 1);
        let init = kmeans_plus_plus(&emb, k, &mut rng);
        let (centers, labels, counts, _inertia, _) = run_lloyd(&mut ctx, &emb, init, 80, &mut rng);
        let _ = (centers, counts);
        if unique_label_count(&labels) <= 1 && k > 1 {
            ctx.push(degenerate_cluster_issue(
                "spectral k-means collapsed to one cluster",
            ));
        }
        ctx.finish(FittedSpectral {
            labels,
            embedding: emb,
        })
    }
}

/// Simplified affinity propagation (Frey & Dueck): responsibility / availability.
#[derive(Clone, Debug)]
pub(crate) struct AffinityPropagation {
    /// Message-passing iteration cap.
    pub max_iter: usize,
    /// Damping in `(0, 1)` (higher is more conservative).
    pub damping: f64,
    /// Seed reserved for tie-breaking preference jitter.
    pub seed: u64,
}

impl Default for AffinityPropagation {
    fn default() -> Self {
        Self {
            max_iter: 50,
            damping: 0.5,
            seed: 0,
        }
    }
}

impl AffinityPropagation {
    /// Default preference (median similarity) and damping.
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Fit alias.
    pub(crate) fn fit(&self, x: &Matrix, session: &Session) -> Result<Qualified<FittedAffinity>> {
        self.fit_unsupervised(x, session)
    }
}

/// Fitted affinity-propagation exemplars.
#[derive(Clone, Debug)]
pub(crate) struct FittedAffinity {
    /// Cluster id per row (exemplar index, remapped to `0 .. n_exemplars-1`).
    pub labels: Vector,
    /// Exemplar row indices as `f64`.
    pub exemplars: Vector,
}

impl FitUnsupervised for AffinityPropagation {
    type Fitted = FittedAffinity;
    fn fit_unsupervised(&self, x: &Matrix, session: &Session) -> Result<Qualified<FittedAffinity>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        inspect_identification(&mut ctx.report, x.nrows(), 1, &ctx.policy);
        let n = x.nrows();
        if n == 0 || x.ncols() == 0 {
            return ctx.finish(FittedAffinity {
                labels: Vector::zeros(0),
                exemplars: Vector::zeros(0),
            });
        }
        if all_rows_identical(x, ctx.policy.near_zero_variance) {
            ctx.push(degenerate_cluster_issue(
                "affinity propagation on identical rows yields a single exemplar",
            ));
        }
        let damp = self.damping.clamp(0.05, 0.95);
        let mut s = Matrix::from_fn(n, n, |i, j| -sq_dist_rows(x, i, j));
        let mut off: Vec<f64> = Vec::with_capacity(n * n);
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    off.push(s.get(i, j));
                }
            }
        }
        off.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let pref = if off.is_empty() {
            0.0
        } else {
            off[off.len() / 2]
        };
        let mut rng = Rng::new(self.seed | 1);
        for i in 0..n {
            s.set(i, i, pref + 1e-12 * rng.standard_normal());
        }
        let mut r = Matrix::zeros(n, n);
        let mut a = Matrix::zeros(n, n);
        for it in 0..self.max_iter.max(1) {
            let old_r = r.clone();
            let old_a = a.clone();
            for i in 0..n {
                for k in 0..n {
                    let mut mx = f64::NEG_INFINITY;
                    for kp in 0..n {
                        if kp == k {
                            continue;
                        }
                        mx = mx.max(a.get(i, kp) + s.get(i, kp));
                    }
                    let val = s.get(i, k) - mx;
                    r.set(i, k, damp * old_r.get(i, k) + (1.0 - damp) * val);
                }
            }
            for i in 0..n {
                for k in 0..n {
                    let val = if i == k {
                        let mut sm = 0.0;
                        for ip in 0..n {
                            if ip != k {
                                sm += r.get(ip, k).max(0.0);
                            }
                        }
                        sm
                    } else {
                        let mut sm = 0.0;
                        for ip in 0..n {
                            if ip != i && ip != k {
                                sm += r.get(ip, k).max(0.0);
                            }
                        }
                        (r.get(k, k) + sm).min(0.0)
                    };
                    a.set(i, k, damp * old_a.get(i, k) + (1.0 - damp) * val);
                }
            }
            if it + 1 == self.max_iter {
                ctx.push(
                    Issue::builder(IssueCode::MaxIterReached)
                        .message("affinity propagation hit max_iter")
                        .build(),
                );
            }
        }
        let mut exemplar_of = vec![0usize; n];
        let mut exemplars: Vec<usize> = Vec::new();
        for i in 0..n {
            let mut bk = 0usize;
            let mut bv = f64::NEG_INFINITY;
            for k in 0..n {
                let v = a.get(i, k) + r.get(i, k);
                if v > bv {
                    bv = v;
                    bk = k;
                }
            }
            exemplar_of[i] = bk;
            if a.get(bk, bk) + r.get(bk, bk) > 0.0 && !exemplars.contains(&bk) {
                exemplars.push(bk);
            }
        }
        if exemplars.is_empty() {
            for &e in &exemplar_of {
                if !exemplars.contains(&e) {
                    exemplars.push(e);
                }
            }
        }
        if exemplars.len() <= 1 && n > 1 {
            ctx.push(
                Issue::builder(IssueCode::DegenerateClusters)
                    .message("affinity propagation found a single exemplar")
                    .build(),
            );
        }
        let mut labels = Vector::zeros(n);
        for i in 0..n {
            let e = exemplar_of[i];
            let id = exemplars.iter().position(|&z| z == e).unwrap_or(0);
            labels[i] = id as f64;
        }
        ctx.finish(FittedAffinity {
            labels,
            exemplars: Vector::from_iter(exemplars.iter().map(|e| *e as f64)),
        })
    }
}

/// Mean-shift clustering (flat kernel).
#[derive(Clone, Debug)]
pub(crate) struct MeanShift {
    /// Kernel bandwidth.
    pub bandwidth: f64,
    /// Max shift iterations.
    pub max_iter: usize,
    /// Merge distance for modes.
    pub merge: f64,
}

impl Default for MeanShift {
    fn default() -> Self {
        Self {
            bandwidth: 1.0,
            max_iter: 40,
            merge: 0.25,
        }
    }
}

impl MeanShift {
    /// Mean-shift with the given bandwidth.
    pub(crate) fn new(bandwidth: f64) -> Self {
        Self {
            bandwidth,
            ..Self::default()
        }
    }

    /// Fit.
    pub(crate) fn fit(&self, x: &Matrix, session: &Session) -> Result<Qualified<FittedMeanShift>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        inspect_identification(&mut ctx.report, x.nrows(), x.ncols(), &ctx.policy);
        let n = x.nrows();
        let p = x.ncols();
        if self.bandwidth <= 0.0 {
            ctx.push(
                Issue::builder(IssueCode::InvalidWeight)
                    .message("mean-shift bandwidth must be positive")
                    .build(),
            );
        }
        let mut modes = Matrix::from_fn(n, p, |i, j| x.get(i, j));
        let bw2 = self.bandwidth * self.bandwidth;
        for it in 0..self.max_iter {
            let mut max_shift = 0.0;
            let mut next = modes.clone();
            for i in 0..n {
                let mut num = vec![0.0; p];
                let mut den = 0.0;
                for t in 0..n {
                    let mut d2 = 0.0;
                    for j in 0..p {
                        let d = modes.get(i, j) - x.get(t, j);
                        d2 += d * d;
                    }
                    if d2 <= bw2 {
                        den += 1.0;
                        for j in 0..p {
                            num[j] += x.get(t, j);
                        }
                    }
                }
                if den <= 0.0 {
                    ctx.push(
                        Issue::builder(IssueCode::EmptyCluster)
                            .message(format!("mean-shift point {i} has an empty neighborhood"))
                            .build(),
                    );
                    continue;
                }
                let mut sh = 0.0;
                for j in 0..p {
                    let m = num[j] / den;
                    sh += (m - modes.get(i, j)).abs();
                    next.set(i, j, m);
                }
                if sh > max_shift {
                    max_shift = sh;
                }
            }
            modes = next;
            ctx.session.step(it as u64, max_shift, None);
            if max_shift < 1e-6 {
                ctx.session.converged("mean-shift modes stable", it as u64);
                break;
            }
        }
        // Merge modes
        let mut centers: Vec<Vec<f64>> = Vec::new();
        let mut labels = Vector::zeros(n);
        for i in 0..n {
            let row: Vec<f64> = (0..p).map(|j| modes.get(i, j)).collect();
            let mut found = None;
            for (c, ctr) in centers.iter().enumerate() {
                let mut d2 = 0.0;
                for j in 0..p {
                    let d = row[j] - ctr[j];
                    d2 += d * d;
                }
                if d2.sqrt() <= self.merge {
                    found = Some(c);
                    break;
                }
            }
            let id = match found {
                Some(c) => c,
                None => {
                    centers.push(row);
                    centers.len() - 1
                }
            };
            labels[i] = id as f64;
        }
        if centers.len() <= 1 && n > 1 {
            ctx.push(
                Issue::builder(IssueCode::DegenerateClusters)
                    .message("mean-shift collapsed to one mode")
                    .build(),
            );
        }
        let k = centers.len();
        let ctr = Matrix::from_fn(k, p, |i, j| centers[i][j]);
        ctx.finish(FittedMeanShift {
            labels,
            centers: ctr,
        })
    }
}

/// Fitted mean-shift.
#[derive(Clone, Debug)]
pub(crate) struct FittedMeanShift {
    /// Labels.
    pub labels: Vector,
    /// Distinct modes.
    pub centers: Matrix,
}

/// OPTICS reachability (simplified; extracts DBSCAN-like clusters from the ordering).
#[derive(Clone, Debug)]
pub(crate) struct Optics {
    /// Neighborhood radius.
    pub eps: f64,
    /// Min points to be a core.
    pub min_samples: usize,
}

impl Default for Optics {
    fn default() -> Self {
        Self {
            eps: 0.5,
            min_samples: 5,
        }
    }
}

impl Optics {
    /// OPTICS with `eps` and `min_samples`.
    pub(crate) fn new(eps: f64, min_samples: usize) -> Self {
        Self { eps, min_samples }
    }

    /// Fit: produce an ordering and extract clusters where reachability ≤ eps.
    pub(crate) fn fit(&self, x: &Matrix, session: &Session) -> Result<Qualified<FittedOptics>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let n = x.nrows();
        let mut reach = vec![f64::INFINITY; n];
        let mut core = vec![f64::INFINITY; n];
        let mut processed = vec![false; n];
        let mut order = Vec::new();
        let dist = |a: usize, b: usize| {
            let mut s = 0.0;
            for j in 0..x.ncols() {
                let d = x.get(a, j) - x.get(b, j);
                s += d * d;
            }
            s.sqrt()
        };
        for i in 0..n {
            let mut neigh = Vec::new();
            for j in 0..n {
                let d = dist(i, j);
                if d <= self.eps {
                    neigh.push((d, j));
                }
            }
            neigh.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            if neigh.len() >= self.min_samples {
                core[i] = neigh[self.min_samples - 1].0;
            }
        }
        for seed in 0..n {
            if processed[seed] {
                continue;
            }
            let mut seeds = vec![seed];
            while let Some(p) = seeds.pop() {
                if processed[p] {
                    continue;
                }
                processed[p] = true;
                order.push(p);
                if !core[p].is_finite() {
                    continue;
                }
                for j in 0..n {
                    if processed[j] {
                        continue;
                    }
                    let d = dist(p, j);
                    if d > self.eps {
                        continue;
                    }
                    let r = core[p].max(d);
                    if r < reach[j] {
                        reach[j] = r;
                        seeds.push(j);
                    }
                }
            }
        }
        let mut labels = Vector::filled(n, -1.0);
        let mut cid = 0.0;
        let mut in_cluster = false;
        for &i in &order {
            if reach[i] <= self.eps || core[i].is_finite() {
                if !in_cluster {
                    cid += 1.0;
                    in_cluster = true;
                }
                labels[i] = cid;
            } else {
                in_cluster = false;
            }
        }
        if cid <= 0.0 && n > 0 {
            ctx.push(
                Issue::builder(IssueCode::DegenerateClusters)
                    .message("OPTICS extracted no clusters at this eps")
                    .build(),
            );
        }
        ctx.finish(FittedOptics {
            labels,
            ordering: Vector::from_iter(order.iter().map(|i| *i as f64)),
            reachability: Vector::from_iter(reach),
        })
    }
}

/// BIRCH: sequential clustering features, then optional k-means on CF centroids.
///
/// A leaf CF is \((N, LS, SS)\). Absorbing a point is allowed only when the
/// resulting radius stays at or below `threshold`. Groups with \(n_g=1\) have
/// an undefined radius and are kept as singleton CFs. Asking for more clusters
/// than CFs is overparameterized.
#[derive(Clone, Debug)]
pub(crate) struct Birch {
    /// Radius threshold \(T\).
    pub threshold: f64,
    /// Global clusters after the CF pass (`None` keeps every CF).
    pub n_clusters: Option<usize>,
}

impl Default for Birch {
    fn default() -> Self {
        Self {
            threshold: 0.5,
            n_clusters: Some(3),
        }
    }
}

impl Birch {
    /// BIRCH with radius `threshold` and optional global `k`.
    pub(crate) fn new(threshold: f64, n_clusters: Option<usize>) -> Self {
        Self {
            threshold,
            n_clusters,
        }
    }
}

#[derive(Clone, Debug)]
struct ClusteringFeature {
    n: f64,
    ls: Vector,
    ss: f64,
}

impl ClusteringFeature {
    fn new(p: usize) -> Self {
        Self {
            n: 0.0,
            ls: Vector::zeros(p),
            ss: 0.0,
        }
    }

    fn from_row(x: &Matrix, i: usize) -> Self {
        let mut ls = Vector::zeros(x.ncols());
        let mut ss = 0.0;
        for j in 0..x.ncols() {
            let v = x.get(i, j);
            ls[j] = v;
            ss += v * v;
        }
        Self { n: 1.0, ls, ss }
    }

    fn radius_after(&self, x: &Matrix, i: usize) -> f64 {
        let n = self.n + 1.0;
        let mut ss = self.ss;
        let mut nrm = 0.0;
        for j in 0..self.ls.len() {
            let v = x.get(i, j);
            ss += v * v;
            let m = (self.ls[j] + v) / n;
            nrm += m * m;
        }
        (ss / n - nrm).max(0.0).sqrt()
    }

    fn absorb(&mut self, x: &Matrix, i: usize) {
        self.n += 1.0;
        for j in 0..self.ls.len() {
            let v = x.get(i, j);
            self.ls[j] += v;
            self.ss += v * v;
        }
    }

    fn centroid(&self) -> Vector {
        if self.n <= 0.0 {
            self.ls.clone()
        } else {
            self.ls.scale(1.0 / self.n)
        }
    }

    fn dist_row(&self, x: &Matrix, i: usize) -> f64 {
        let c = self.centroid();
        let mut s = 0.0;
        for j in 0..c.len().min(x.ncols()) {
            let d = x.get(i, j) - c[j];
            s += d * d;
        }
        s.sqrt()
    }
}

impl FitUnsupervised for Birch {
    type Fitted = FittedBirch;
    fn fit_unsupervised(&self, x: &Matrix, session: &Session) -> Result<Qualified<FittedBirch>> {
        let mut this = self.clone();
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        if x.nrows() == 0 || x.ncols() == 0 {
            return ctx.finish(FittedBirch {
                labels: Vector::zeros(x.nrows()),
                centroids: Matrix::zeros(0, x.ncols()),
                n_cf: 0,
            });
        }
        if this.threshold < 0.0 || !this.threshold.is_finite() {
            ctx.push(
                Issue::builder(IssueCode::InvalidWeight)
                    .message(format!(
                        "BIRCH threshold={} is not a finite ≥0 radius",
                        this.threshold
                    ))
                    .build(),
            );
        }
        if all_rows_identical(x, ctx.policy.near_zero_variance) {
            ctx.push(degenerate_cluster_issue(
                "all observations are the same vector; BIRCH CFs collapse to one point",
            ));
        }
        let mut cfs: Vec<ClusteringFeature> = Vec::new();
        for i in 0..x.nrows() {
            let mut best = None;
            let mut best_d = f64::INFINITY;
            for (k, cf) in cfs.iter().enumerate() {
                let d = cf.dist_row(x, i);
                if d < best_d {
                    best_d = d;
                    best = Some(k);
                }
            }
            match best {
                Some(k) if cfs[k].radius_after(x, i) <= this.threshold => {
                    cfs[k].absorb(x, i);
                }
                _ => cfs.push(ClusteringFeature::from_row(x, i)),
            }
        }
        if cfs.is_empty() {
            cfs.push(ClusteringFeature::new(x.ncols()));
        }
        let n_cf = cfs.len();
        if n_cf == 1 && x.nrows() > 1 {
            ctx.push(
                Issue::builder(IssueCode::DegenerateClusters)
                    .severity(signlred::Severity::Warning)
                    .message("BIRCH produced a single CF; the threshold swallowed the whole sample")
                    .meaninglessness(Meaninglessness::new(
                        "BIRCH partition",
                        "one micro-cluster is a constant assignment",
                        signlred::InterpretiveValue::Misleading,
                        "lower the threshold or skip the global k-means step",
                    ))
                    .build(),
            );
        }
        let cf_cent = Matrix::from_fn(n_cf, x.ncols(), |i, j| cfs[i].centroid()[j]);
        let k_req = this.n_clusters.unwrap_or(n_cf).max(1);
        let k = k_req.min(n_cf);
        if k < k_req {
            ctx.push(
                Issue::builder(IssueCode::Overparameterized)
                    .message(format!(
                        "requested {k_req} clusters but only {n_cf} CFs exist"
                    ))
                    .build(),
            );
        }
        let centroids = if k == n_cf {
            cf_cent.clone()
        } else {
            let mut km = KMeans {
                k,
                max_iter: 40,
                seed: 3,
                n_init: 2,
            };
            match km.fit(&cf_cent, &session.child("birch_kmeans")) {
                Ok(q) => q.value.centroids,
                Err(_) => cf_cent.clone(),
            }
        };
        let (labels, _, _) = assign_lloyd(x, &centroids);
        ctx.finish(FittedBirch {
            labels,
            centroids,
            n_cf,
        })
    }
}

impl Birch {
    /// Fit alias for [`FitUnsupervised::fit_unsupervised`].
    pub(crate) fn fit(&self, x: &Matrix, session: &Session) -> Result<Qualified<FittedBirch>> {
        self.fit_unsupervised(x, session)
    }
}

/// Fitted BIRCH partition.
#[derive(Clone, Debug)]
pub(crate) struct FittedBirch {
    /// Cluster id per row.
    pub labels: Vector,
    /// Global centroids (`k` × `p`).
    pub centroids: Matrix,
    /// Number of clustering features before the global k-means step.
    pub n_cf: usize,
}

impl Predict for FittedBirch {
    type Output = Vector;
    fn predict(&self, x: &Matrix, session: &Session) -> Result<Qualified<Vector>> {
        let mut ctx = FitCtx::with_session(session.child("predict"));
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        if x.ncols() != self.centroids.ncols() {
            ctx.push(
                Issue::builder(IssueCode::DimensionMismatch)
                    .message(format!(
                        "predict X is n×{} but BIRCH centroids are k×{}",
                        x.ncols(),
                        self.centroids.ncols()
                    ))
                    .build(),
            );
        }
        let (labels, _, _) = assign_lloyd(x, &self.centroids);
        ctx.finish(labels)
    }
}

/// Fitted OPTICS.
#[derive(Clone, Debug)]
pub(crate) struct FittedOptics {
    /// Cluster labels (−1 = noise).
    pub labels: Vector,
    /// Processing order (indices).
    pub ordering: Vector,
    /// Reachability distances.
    pub reachability: Vector,
}

/// Bisecting k-means: recursively split the highest-SSE cluster with 2-means.
#[derive(Clone, Debug)]
pub(crate) struct BisectingKMeans {
    /// Target number of clusters.
    pub k: usize,
    /// Lloyd iterations per bisection.
    pub max_iter: usize,
    /// PRNG seed.
    pub seed: u64,
}

impl Default for BisectingKMeans {
    fn default() -> Self {
        Self {
            k: 8,
            max_iter: 50,
            seed: 0,
        }
    }
}

impl BisectingKMeans {
    /// Bisect until `k` clusters.
    pub(crate) fn new(k: usize) -> Self {
        Self {
            k,
            ..Self::default()
        }
    }

    /// Fit alias.
    pub(crate) fn fit(&self, x: &Matrix, session: &Session) -> Result<Qualified<FittedKMeans>> {
        self.fit_unsupervised(x, session)
    }
}

impl FitUnsupervised for BisectingKMeans {
    type Fitted = FittedKMeans;
    fn fit_unsupervised(&self, x: &Matrix, session: &Session) -> Result<Qualified<FittedKMeans>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let (n, p) = x.shape();
        let want = self.k.max(1);
        if n == 0 || p == 0 {
            return ctx.finish(empty_kmeans(want, p, n));
        }
        if want > n {
            ctx.push(
                Issue::builder(IssueCode::Overparameterized)
                    .message(format!("bisecting k={want} > n={n}"))
                    .build(),
            );
        }
        let k = want.min(n.max(1));
        let mut rng = Rng::new(self.seed | 1);
        let mut groups: Vec<Vec<usize>> = vec![(0..n).collect()];
        let mut n_iter = 0usize;
        while groups.len() < k {
            let mut best = 0usize;
            let mut best_sse = -1.0;
            for (g, idx) in groups.iter().enumerate() {
                if idx.len() < 2 {
                    continue;
                }
                let mut mean = vec![0.0; p];
                for &i in idx {
                    for j in 0..p {
                        mean[j] += x.get(i, j);
                    }
                }
                let inv = 1.0 / idx.len() as f64;
                for m in mean.iter_mut() {
                    *m *= inv;
                }
                let mut sse = 0.0;
                for &i in idx {
                    for j in 0..p {
                        let d = x.get(i, j) - mean[j];
                        sse += d * d;
                    }
                }
                if sse > best_sse {
                    best_sse = sse;
                    best = g;
                }
            }
            if best_sse <= 0.0 {
                ctx.push(
                    Issue::builder(IssueCode::EmptyCluster)
                        .message("no cluster with ≥2 distinct points remains to bisect")
                        .build(),
                );
                break;
            }
            let idx = groups[best].clone();
            let sub = Matrix::from_fn(idx.len(), p, |i, j| x.get(idx[i], j));
            let mut centers = kmeans_plus_plus(&sub, 2, &mut rng);
            let mut last = Vector::zeros(idx.len());
            for it in 0..self.max_iter.max(1) {
                let (labels, counts, _) = assign_lloyd(&sub, &centers);
                update_means(&sub, &labels, &mut centers, &counts);
                reseed_empty(&mut ctx, &sub, &mut centers, &counts, &mut rng);
                n_iter += 1;
                let mut changed = false;
                for i in 0..labels.len() {
                    if (labels[i] - last[i]).abs() > 0.0 {
                        changed = true;
                    }
                }
                last = labels;
                ctx.session.step(it as u64, 0.0, None);
                if !changed && it > 0 {
                    break;
                }
            }
            let mut a = Vec::new();
            let mut b = Vec::new();
            for (i, &row) in idx.iter().enumerate() {
                if last[i] < 0.5 {
                    a.push(row);
                } else {
                    b.push(row);
                }
            }
            if a.is_empty() || b.is_empty() {
                ctx.push(
                    Issue::builder(IssueCode::EmptyCluster)
                        .message("a bisection produced an empty child")
                        .build(),
                );
                break;
            }
            groups[best] = a;
            groups.push(b);
        }
        let kk = groups.len();
        let mut centroids = Matrix::zeros(kk, p);
        let mut labels = Vector::zeros(n);
        let mut counts = Vector::zeros(kk);
        let mut inertia = 0.0;
        for (c, idx) in groups.iter().enumerate() {
            counts[c] = idx.len() as f64;
            if idx.is_empty() {
                continue;
            }
            for j in 0..p {
                let mut s = 0.0;
                for &i in idx {
                    s += x.get(i, j);
                }
                centroids.set(c, j, s / idx.len() as f64);
            }
            for &i in idx {
                labels[i] = c as f64;
                inertia += sq_dist_rc(x, i, &centroids, c);
            }
        }
        if kk < 2 && n > 1 && !all_rows_identical(x, ctx.policy.near_zero_variance) {
            ctx.push(
                Issue::builder(IssueCode::DegenerateClusters)
                    .message("bisecting k-means collapsed to one cluster")
                    .build(),
            );
        }
        ctx.finish(FittedKMeans {
            labels,
            centroids,
            inertia,
            n_iter,
            counts,
        })
    }
}

/// Spectral co-clustering (Dhillon): normalize `A`, SVD, k-means on the
/// stacked singular vectors (sklearn `SpectralCoclustering`).
///
/// `n_clusters` is not passed to [`inspect_identification`]. An all-zero
/// table is vacuous.
#[derive(Clone, Debug)]
pub(crate) struct SpectralCoclustering {
    /// Number of row/column clusters.
    pub n_clusters: usize,
    /// PRNG seed for the k-means stage.
    pub seed: u64,
}

impl Default for SpectralCoclustering {
    fn default() -> Self {
        Self {
            n_clusters: 2,
            seed: 1,
        }
    }
}

impl SpectralCoclustering {
    /// `k` co-clusters.
    pub(crate) fn new(n_clusters: usize) -> Self {
        Self {
            n_clusters: n_clusters.max(2),
            ..Self::default()
        }
    }

    /// Fit alias.
    pub(crate) fn fit(&self, x: &Matrix, session: &Session) -> Result<Qualified<FittedCocluster>> {
        self.fit_unsupervised(x, session)
    }
}

/// Fitted spectral co-clustering.
#[derive(Clone, Debug)]
pub(crate) struct FittedCocluster {
    /// Row labels.
    pub row_labels: Vector,
    /// Column labels.
    pub col_labels: Vector,
}

fn local_kmeans(x: &Matrix, k: usize, seed: u64) -> Vector {
    let n = x.nrows();
    let k = k.max(1).min(n.max(1));
    if n == 0 {
        return Vector::zeros(0);
    }
    let mut rng = Rng::new(seed | 11);
    let mut cents = kmeans_plus_plus(x, k, &mut rng);
    let mut labels = Vector::zeros(n);
    for _ in 0..25 {
        for i in 0..n {
            let mut b = 0usize;
            let mut bd = f64::INFINITY;
            for c in 0..k {
                let d = sq_dist_rc(x, i, &cents, c);
                if d < bd {
                    bd = d;
                    b = c;
                }
            }
            labels[i] = b as f64;
        }
        for c in 0..k {
            let mut cnt = 0.0;
            for j in 0..x.ncols() {
                cents.set(c, j, 0.0);
            }
            for i in 0..n {
                if labels[i] as usize == c {
                    cnt += 1.0;
                    for j in 0..x.ncols() {
                        cents.set(c, j, cents.get(c, j) + x.get(i, j));
                    }
                }
            }
            if cnt > 0.0 {
                for j in 0..x.ncols() {
                    cents.set(c, j, cents.get(c, j) / cnt);
                }
            }
        }
    }
    labels
}

impl FitUnsupervised for SpectralCoclustering {
    type Fitted = FittedCocluster;
    fn fit_unsupervised(
        &self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedCocluster>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let (n, p) = x.shape();
        if n == 0 || p == 0 {
            return ctx.finish(FittedCocluster {
                row_labels: Vector::zeros(n),
                col_labels: Vector::zeros(p),
            });
        }
        let mut energy = 0.0f64;
        let mut neg = false;
        for i in 0..n {
            for j in 0..p {
                let v = x.get(i, j);
                energy = energy.max(v.abs());
                if v < 0.0 {
                    neg = true;
                }
            }
        }
        if energy <= ctx.policy.near_zero_variance {
            ctx.push(
                Issue::builder(IssueCode::MeaninglessFit)
                    .message("co-clustering table is the zero map")
                    .meaninglessness(Meaninglessness::vacuous(
                        "co-cluster labels",
                        "a zero table has no row/column association to partition",
                        "supply a non-negative contingency / count matrix",
                    ))
                    .build(),
            );
            return ctx.finish(FittedCocluster {
                row_labels: Vector::zeros(n),
                col_labels: Vector::zeros(p),
            });
        }
        if neg {
            ctx.push(
                Issue::builder(IssueCode::NonPositiveSeries)
                    .severity(Severity::Warning)
                    .message("SpectralCoclustering saw negative entries; treating them as 0")
                    .build(),
            );
        }
        let mut rsum = vec![0.0; n];
        let mut csum = vec![0.0; p];
        for i in 0..n {
            for j in 0..p {
                let v = x.get(i, j).max(0.0);
                rsum[i] += v;
                csum[j] += v;
            }
        }
        let an = Matrix::from_fn(n, p, |i, j| {
            let d = (rsum[i].max(1e-12) * csum[j].max(1e-12)).sqrt();
            x.get(i, j).max(0.0) / d
        });
        let mut scratch = signlred::Report::new("coclust", "svd");
        let Some(svd) = thin_svd(&mut scratch, &an, &ctx.policy) else {
            ctx.push(
                Issue::builder(IssueCode::SvdDidNotConverge)
                    .message("co-clustering SVD failed")
                    .build(),
            );
            return ctx.finish(FittedCocluster {
                row_labels: Vector::zeros(n),
                col_labels: Vector::zeros(p),
            });
        };
        let k = self.n_clusters.max(2);
        let rkeep = svd.singular_values.len().min(k).max(1);
        let zrow = Matrix::from_fn(n, rkeep, |i, c| svd.u[(i, c)]);
        let zcol = Matrix::from_fn(p, rkeep, |j, c| svd.v[(j, c)]);
        ctx.finish(FittedCocluster {
            row_labels: local_kmeans(&zrow, k, self.seed),
            col_labels: local_kmeans(&zcol, k, self.seed.wrapping_add(1)),
        })
    }
}

/// Spectral bi-clustering (sklearn `SpectralBiclustering`, Kluger-style).
///
/// Rows and columns are clustered independently in the SVD embedding.
/// Cluster count is not identification `p`.
#[derive(Clone, Debug)]
pub(crate) struct SpectralBiclustering {
    /// Number of row clusters.
    pub n_clusters: usize,
    /// PRNG seed.
    pub seed: u64,
}

impl Default for SpectralBiclustering {
    fn default() -> Self {
        Self {
            n_clusters: 2,
            seed: 2,
        }
    }
}

impl SpectralBiclustering {
    /// `k` row/column clusters.
    pub(crate) fn new(n_clusters: usize) -> Self {
        Self {
            n_clusters: n_clusters.max(2),
            ..Self::default()
        }
    }

    /// Fit alias.
    pub(crate) fn fit(&self, x: &Matrix, session: &Session) -> Result<Qualified<FittedCocluster>> {
        self.fit_unsupervised(x, session)
    }
}

impl FitUnsupervised for SpectralBiclustering {
    type Fitted = FittedCocluster;
    fn fit_unsupervised(
        &self,
        x: &Matrix,
        session: &Session,
    ) -> Result<Qualified<FittedCocluster>> {
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let (n, p) = x.shape();
        if n == 0 || p == 0 {
            return ctx.finish(FittedCocluster {
                row_labels: Vector::zeros(n),
                col_labels: Vector::zeros(p),
            });
        }
        let mut rsum = vec![0.0; n];
        let mut csum = vec![0.0; p];
        let mut energy: f64 = 0.0;
        for i in 0..n {
            for j in 0..p {
                let v = x.get(i, j).abs();
                energy = energy.max(v);
                rsum[i] += v;
                csum[j] += v;
            }
        }
        if energy <= ctx.policy.near_zero_variance {
            ctx.push(
                Issue::builder(IssueCode::MeaninglessFit)
                    .message("bi-clustering table is the zero map")
                    .meaninglessness(Meaninglessness::vacuous(
                        "bi-cluster labels",
                        "a zero table has no row/column association to partition",
                        "supply a non-zero matrix",
                    ))
                    .build(),
            );
            return ctx.finish(FittedCocluster {
                row_labels: Vector::zeros(n),
                col_labels: Vector::zeros(p),
            });
        }
        let an = Matrix::from_fn(n, p, |i, j| {
            x.get(i, j) / (rsum[i].max(1e-12) * csum[j].max(1e-12)).sqrt()
        });
        let mut scratch = signlred::Report::new("biclust", "svd");
        let Some(svd) = thin_svd(&mut scratch, &an, &ctx.policy) else {
            ctx.push(
                Issue::builder(IssueCode::SvdDidNotConverge)
                    .message("bi-clustering SVD failed")
                    .build(),
            );
            return ctx.finish(FittedCocluster {
                row_labels: Vector::zeros(n),
                col_labels: Vector::zeros(p),
            });
        };
        let k = self.n_clusters.max(2);
        let rkeep = svd.singular_values.len().min(k).max(1);
        let zrow = Matrix::from_fn(n, rkeep, |i, c| svd.u[(i, c)]);
        let zcol = Matrix::from_fn(p, rkeep, |j, c| svd.v[(j, c)]);
        ctx.finish(FittedCocluster {
            row_labels: local_kmeans(&zrow, k, self.seed),
            col_labels: local_kmeans(&zcol, k, self.seed.wrapping_add(3)),
        })
    }
}

/// Quantile of pairwise Euclidean distances (sklearn `estimate_bandwidth`).
///
/// `quantile` is not identification `p`. Neighbor count is not `p`.
pub(crate) fn estimate_bandwidth(
    x: &Matrix,
    quantile: f64,
    session: &Session,
) -> Result<Qualified<f64>> {
    let mut ctx = FitCtx::with_session(session.clone());
    inspect_xy(&mut ctx.report, x, None, &ctx.policy);
    let q = if quantile.is_finite() && quantile > 0.0 && quantile < 1.0 {
        quantile
    } else {
        ctx.push(
            Issue::builder(IssueCode::InvalidWeight)
                .severity(Severity::Warning)
                .message(format!(
                    "estimate_bandwidth quantile={quantile} not in (0,1); using 0.3"
                ))
                .build(),
        );
        0.3
    };
    let n = x.nrows();
    if n < 2 {
        ctx.push(
            Issue::builder(IssueCode::InsufficientSample)
                .severity(Severity::Warning)
                .message("estimate_bandwidth needs at least two rows")
                .build(),
        );
        return ctx.finish(f64::NAN);
    }
    let mut dists = Vec::with_capacity(n * (n - 1) / 2);
    for i in 0..n {
        for j in (i + 1)..n {
            let mut s = 0.0;
            for k in 0..x.ncols() {
                let d = x.get(i, k) - x.get(j, k);
                s += d * d;
            }
            dists.push(s.sqrt());
        }
    }
    dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let pos = q * (dists.len().saturating_sub(1)) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    let t = pos - lo as f64;
    let bw = if dists.is_empty() {
        f64::NAN
    } else {
        (1.0 - t) * dists[lo] + t * dists[hi.min(dists.len() - 1)]
    };
    if bw.is_finite() && bw <= 1e-18 {
        ctx.push(
            Issue::builder(IssueCode::NearZeroVariance)
                .message("estimate_bandwidth collapsed to 0; every pair is coincident")
                .build(),
        );
    }
    ctx.finish(bw)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ojizou_san::Session;

    fn two_blobs() -> Matrix {
        // 20 points near (−4, −4) and 20 near (4, 4).
        Matrix::from_fn(40, 2, |i, j| {
            if i < 20 {
                -4.0 + 0.05 * (i as f64) + 0.02 * j as f64
            } else {
                4.0 + 0.05 * ((i - 20) as f64) + 0.02 * j as f64
            }
        })
    }

    #[test]
    fn kmeans_recovers_two_blobs() {
        let x = two_blobs();
        let session = Session::new("kmeans", "fit");
        let q = KMeans {
            k: 2,
            max_iter: 40,
            seed: 3,
            n_init: 4,
        }
        .fit(&x, &session)
        .expect("kmeans");
        let lab = q.value.labels.as_slice();
        let a = lab[0];
        let b = lab[20];
        assert_ne!(a, b, "blobs must receive different labels: {lab:?}");
        for i in 0..20 {
            assert_eq!(lab[i], a, "blob A row {i}");
        }
        for i in 20..40 {
            assert_eq!(lab[i], b, "blob B row {i}");
        }
        let pred = q
            .value
            .predict(&x, &Session::new("kmeans", "predict"))
            .expect("predict");
        assert_eq!(pred.value.len(), 40);
        let bw = estimate_bandwidth(&x, 0.3, &Session::new("bw", "fit"))
            .expect("bw")
            .value;
        assert!(bw.is_finite() && bw > 0.0);
    }

    #[test]
    fn kmeans_empty_errors() {
        let x = Matrix::zeros(0, 2);
        let session = Session::new("kmeans", "fit");
        let err = KMeans::new(2).fit(&x, &session).unwrap_err();
        assert_eq!(err.primary().code, IssueCode::EmptyMatrix);
    }

    #[test]
    fn minibatch_explains_partial_fit() {
        let x = two_blobs();
        let session = Session::new("mbkmeans", "partial_fit");
        let mut m = MiniBatchKMeans {
            k: 2,
            max_iter: 5,
            seed: 1,
            batch_size: 16,
            ..MiniBatchKMeans::default()
        };
        let q = m.partial_fit(&x, None, &session).expect("pf");
        assert!(!q.value.narrative.is_empty());
        assert!(q.value.quality.effective_sample_size > 0.0);
    }

    #[test]
    fn bisecting_kmeans_recovers_two_blobs() {
        let x = two_blobs();
        let q = BisectingKMeans::new(2)
            .fit(&x, &Session::new("bisect", "fit"))
            .expect("bisect");
        assert_eq!(q.value.centroids.nrows(), 2);
        let lab = q.value.labels.as_slice();
        assert_ne!(lab[0], lab[20], "blobs must differ: {lab:?}");
    }

    #[test]
    fn birch_separates_two_blobs() {
        let x = two_blobs();
        let q = Birch {
            threshold: 1.5,
            n_clusters: Some(2),
        }
        .fit(&x, &Session::new("birch", "fit"))
        .expect("birch");
        assert!(q.value.n_cf >= 2, "n_cf={}", q.value.n_cf);
        let lab = q.value.labels.as_slice();
        assert_ne!(lab[0], lab[20], "blobs must differ: {lab:?}");
    }

    #[test]
    fn hdbscan_separates_two_blobs() {
        let x = two_blobs();
        let q = Hdbscan::new(5, 5)
            .fit(&x, &Session::new("hdb", "fit"))
            .expect("hdb");
        assert!(q.value.n_clusters >= 1, "n={}", q.value.n_clusters);
        let lab = q.value.labels.as_slice();
        if q.value.n_clusters >= 2 {
            assert_ne!(lab[0], lab[20], "blobs must differ: {lab:?}");
        }
    }

    #[test]
    fn bayesian_gmm_separates_two_blobs() {
        let x = two_blobs();
        let q = BayesianGaussianMixture::new(2)
            .fit(&x, &Session::new("bgmm", "fit"))
            .expect("bgmm");
        let lab = q.value.labels.as_slice();
        assert_ne!(lab[0], lab[20], "blobs must differ: {lab:?}");
        assert_eq!(q.value.weights.len(), 2);
    }

    #[test]
    fn spectral_coclustering_block_matrix() {
        let x = Matrix::from_fn(12, 8, |i, j| {
            if (i < 6 && j < 4) || (i >= 6 && j >= 4) {
                2.0
            } else {
                0.1
            }
        });
        let q = SpectralCoclustering::new(2)
            .fit(&x, &Session::new("scc", "fit"))
            .expect("scc");
        assert_eq!(q.value.row_labels.len(), 12);
        assert_eq!(q.value.col_labels.len(), 8);
        assert_ne!(q.value.row_labels[0], q.value.row_labels[8]);
        let bq = SpectralBiclustering::new(2)
            .fit(&x, &Session::new("sbc", "fit"))
            .expect("sbc");
        assert_eq!(bq.value.row_labels.len(), 12);
        assert_eq!(bq.value.col_labels.len(), 8);
        let x2 = two_blobs();
        let agg = AgglomerativeClustering::new(2)
            .fit(&x2, &Session::new("aggc", "fit"))
            .expect("aggc");
        assert_eq!(agg.value.labels.len(), 40);
    }
}
