//! Waveform distances, alignments, and waveform barycenters.
//!
//! The crate deliberately owns DTW and related dynamic-programming quantities;
//! generic sample distances and optimal transport live in `wormhole`, while
//! kernel quantities live in `coronel`.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

use faer::Mat;
use std::fmt;

/// Errors returned by waveform calculations.
#[derive(Clone, Debug, PartialEq)]
pub enum Error {
    /// An input waveform has no samples.
    EmptyInput,
    /// A sample is NaN or infinity.
    NonFiniteInput {
        /// Index of the invalid sample.
        index: usize,
    },
    /// An option is outside its mathematical domain.
    InvalidParameter(&'static str),
    /// Matrices expected to have the same shape differ.
    ShapeMismatch {
        /// Shape of the first input.
        left: (usize, usize),
        /// Shape of the second input.
        right: (usize, usize),
    },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyInput => write!(f, "waveforms must be non-empty"),
            Self::NonFiniteInput { index } => {
                write!(f, "waveform sample {index} is not finite")
            }
            Self::InvalidParameter(parameter) => {
                write!(f, "invalid waveform parameter: {parameter}")
            }
            Self::ShapeMismatch { left, right } => {
                write!(f, "waveform matrix shapes differ: {left:?} != {right:?}")
            }
        }
    }
}

impl std::error::Error for Error {}

/// Scalar local cost used by dynamic-programming alignments.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum LocalCost {
    /// Absolute difference, matching the original `kanimiso` DTW convention.
    #[default]
    Absolute,
    /// Squared difference.
    Squared,
}

impl LocalCost {
    fn evaluate(self, left: f64, right: f64) -> f64 {
        let difference = left - right;
        match self {
            Self::Absolute => difference.abs(),
            Self::Squared => difference * difference,
        }
    }
}

/// Configuration shared by hard DTW calculations.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DtwOptions {
    /// Optional Sakoe-Chiba radius around the diagonal.
    pub window: Option<usize>,
    /// Non-negative additive cost for horizontal and vertical moves.
    pub warp_penalty: f64,
    /// Pointwise cost.
    pub local_cost: LocalCost,
    /// Divide the accumulated cost by the selected path length.
    pub normalize: bool,
}

impl Default for DtwOptions {
    fn default() -> Self {
        Self {
            window: None,
            warp_penalty: 0.0,
            local_cost: LocalCost::Absolute,
            normalize: false,
        }
    }
}

impl DtwOptions {
    fn validate(self) -> Result<(), Error> {
        if !self.warp_penalty.is_finite() || self.warp_penalty < 0.0 {
            Err(Error::InvalidParameter(
                "warp_penalty must be finite and non-negative",
            ))
        } else {
            Ok(())
        }
    }
}

/// A hard DTW alignment and its accumulated cost.
#[derive(Clone, Debug, PartialEq)]
pub struct DtwAlignment {
    /// DTW distance under the requested normalization.
    pub distance: f64,
    /// Zero-based pairs `(left_index, right_index)` from start to end.
    pub path: Vec<(usize, usize)>,
}

fn validate_series(series: &[f64]) -> Result<(), Error> {
    if series.is_empty() {
        return Err(Error::EmptyInput);
    }
    for (index, value) in series.iter().enumerate() {
        if !value.is_finite() {
            return Err(Error::NonFiniteInput { index });
        }
    }
    Ok(())
}

/// Classic unwindowed DTW distance with absolute local cost.
pub fn dtw(left: &[f64], right: &[f64]) -> Result<f64, Error> {
    Ok(dtw_with_options(left, right, DtwOptions::default())?.distance)
}

/// Compute hard DTW with a deterministic alignment path.
pub fn dtw_with_options(
    left: &[f64],
    right: &[f64],
    options: DtwOptions,
) -> Result<DtwAlignment, Error> {
    validate_series(left)?;
    validate_series(right)?;
    options.validate()?;
    hard_dtw(left.len(), right.len(), options, |i, j| {
        options.local_cost.evaluate(left[i], right[j])
    })
}

fn hard_dtw(
    n: usize,
    m: usize,
    options: DtwOptions,
    mut local_cost: impl FnMut(usize, usize) -> f64,
) -> Result<DtwAlignment, Error> {
    let columns = m + 1;
    let mut values = vec![f64::INFINITY; (n + 1) * columns];
    let mut steps = vec![usize::MAX; (n + 1) * columns];
    let mut predecessors = vec![0_u8; (n + 1) * columns];
    let at = |i: usize, j: usize| i * columns + j;
    values[at(0, 0)] = 0.0;
    steps[at(0, 0)] = 0;
    let radius = options.window.unwrap_or(n.max(m)).max(n.abs_diff(m));
    for i in 1..=n {
        let center = i.saturating_mul(m).saturating_add(n / 2) / n;
        let start = center.saturating_sub(radius).max(1);
        let end = center.saturating_add(radius).min(m);
        for j in start..=end {
            let diagonal = values[at(i - 1, j - 1)];
            let vertical = values[at(i - 1, j)] + options.warp_penalty;
            let horizontal = values[at(i, j - 1)] + options.warp_penalty;
            let (best, predecessor) = if diagonal <= vertical && diagonal <= horizontal {
                (diagonal, 1_u8)
            } else if vertical <= horizontal {
                (vertical, 2_u8)
            } else {
                (horizontal, 3_u8)
            };
            if best.is_finite() {
                values[at(i, j)] = best + local_cost(i - 1, j - 1);
                steps[at(i, j)] = match predecessor {
                    1 => steps[at(i - 1, j - 1)].saturating_add(1),
                    2 => steps[at(i - 1, j)].saturating_add(1),
                    _ => steps[at(i, j - 1)].saturating_add(1),
                };
                predecessors[at(i, j)] = predecessor;
            }
        }
    }
    if !values[at(n, m)].is_finite() {
        return Err(Error::InvalidParameter(
            "window excludes every complete alignment",
        ));
    }
    let mut path = Vec::with_capacity(steps[at(n, m)]);
    let (mut i, mut j) = (n, m);
    while i > 0 && j > 0 {
        path.push((i - 1, j - 1));
        match predecessors[at(i, j)] {
            1 => {
                i -= 1;
                j -= 1;
            }
            2 => i -= 1,
            3 => j -= 1,
            _ => {
                return Err(Error::InvalidParameter(
                    "window produced an incomplete predecessor chain",
                ));
            }
        }
    }
    path.reverse();
    let mut distance = values[at(n, m)];
    if options.normalize {
        distance /= path.len() as f64;
    }
    Ok(DtwAlignment { distance, path })
}

fn validate_multivariate_series(series: &Mat<f64>) -> Result<(), Error> {
    if series.nrows() == 0 || series.ncols() == 0 {
        return Err(Error::EmptyInput);
    }
    for row in 0..series.nrows() {
        for column in 0..series.ncols() {
            if !series[(row, column)].is_finite() {
                return Err(Error::NonFiniteInput {
                    index: row * series.ncols() + column,
                });
            }
        }
    }
    Ok(())
}

/// DTW between variable-length multivariate series.
///
/// Rows are time steps and columns are channels. [`LocalCost::Absolute`]
/// sums absolute channel differences; [`LocalCost::Squared`] sums squared
/// channel differences.
pub fn dtw_multivariate(left: &Mat<f64>, right: &Mat<f64>) -> Result<f64, Error> {
    Ok(dtw_multivariate_with_options(left, right, DtwOptions::default())?.distance)
}

/// Multivariate DTW with a deterministic path and explicit options.
pub fn dtw_multivariate_with_options(
    left: &Mat<f64>,
    right: &Mat<f64>,
    options: DtwOptions,
) -> Result<DtwAlignment, Error> {
    validate_multivariate_series(left)?;
    validate_multivariate_series(right)?;
    options.validate()?;
    if left.ncols() != right.ncols() {
        return Err(Error::ShapeMismatch {
            left: (left.nrows(), left.ncols()),
            right: (right.nrows(), right.ncols()),
        });
    }
    hard_dtw(left.nrows(), right.nrows(), options, |i, j| {
        (0..left.ncols())
            .map(|channel| {
                options
                    .local_cost
                    .evaluate(left[(i, channel)], right[(j, channel)])
            })
            .sum()
    })
}

fn soft_min(values: [f64; 3], gamma: f64) -> f64 {
    let minimum = values[0].min(values[1]).min(values[2]);
    if !minimum.is_finite() {
        return f64::INFINITY;
    }
    let sum = values
        .iter()
        .map(|value| (-(value - minimum) / gamma).exp())
        .sum::<f64>();
    minimum - gamma * sum.ln()
}

/// Soft-DTW of Cuturi and Blondel.
///
/// The returned regularized objective can be negative.  Use
/// [`soft_dtw_divergence`] for a non-negative, self-debiased quantity.
pub fn soft_dtw(
    left: &[f64],
    right: &[f64],
    gamma: f64,
    local_cost: LocalCost,
) -> Result<f64, Error> {
    validate_series(left)?;
    validate_series(right)?;
    soft_dtw_with_local_cost(left.len(), right.len(), gamma, |i, j| {
        local_cost.evaluate(left[i], right[j])
    })
}

fn soft_dtw_with_local_cost(
    n: usize,
    m: usize,
    gamma: f64,
    mut local_cost: impl FnMut(usize, usize) -> f64,
) -> Result<f64, Error> {
    if !gamma.is_finite() || gamma <= 0.0 {
        return Err(Error::InvalidParameter(
            "gamma must be finite and strictly positive",
        ));
    }
    let columns = m + 1;
    let at = |i: usize, j: usize| i * columns + j;
    let mut values = vec![f64::INFINITY; (n + 1) * columns];
    values[at(0, 0)] = 0.0;
    for i in 1..=n {
        for j in 1..=m {
            values[at(i, j)] = local_cost(i - 1, j - 1)
                + soft_min(
                    [
                        values[at(i - 1, j)],
                        values[at(i, j - 1)],
                        values[at(i - 1, j - 1)],
                    ],
                    gamma,
                );
        }
    }
    Ok(values[at(n, m)])
}

/// Soft-DTW between variable-length multivariate series.
///
/// Rows are time steps and columns are channels. Channel costs are summed
/// according to `local_cost`.
pub fn soft_dtw_multivariate(
    left: &Mat<f64>,
    right: &Mat<f64>,
    gamma: f64,
    local_cost: LocalCost,
) -> Result<f64, Error> {
    validate_multivariate_series(left)?;
    validate_multivariate_series(right)?;
    if left.ncols() != right.ncols() {
        return Err(Error::ShapeMismatch {
            left: (left.nrows(), left.ncols()),
            right: (right.nrows(), right.ncols()),
        });
    }
    soft_dtw_with_local_cost(left.nrows(), right.nrows(), gamma, |i, j| {
        (0..left.ncols())
            .map(|channel| local_cost.evaluate(left[(i, channel)], right[(j, channel)]))
            .sum()
    })
}

/// Self-debiased Soft-DTW divergence.
pub fn soft_dtw_divergence(
    left: &[f64],
    right: &[f64],
    gamma: f64,
    local_cost: LocalCost,
) -> Result<f64, Error> {
    let cross = soft_dtw(left, right, gamma, local_cost)?;
    let left_self = soft_dtw(left, left, gamma, local_cost)?;
    let right_self = soft_dtw(right, right, gamma, local_cost)?;
    Ok((cross - 0.5 * (left_self + right_self)).max(0.0))
}

fn matrix_row(matrix: &Mat<f64>, row: usize) -> Vec<f64> {
    (0..matrix.ncols())
        .map(|column| matrix[(row, column)])
        .collect()
}

/// Pairwise DTW distances between rows of two matrices.
pub fn pairwise_dtw(
    left: &Mat<f64>,
    right: &Mat<f64>,
    options: DtwOptions,
) -> Result<Mat<f64>, Error> {
    if left.nrows() == 0 || left.ncols() == 0 || right.nrows() == 0 || right.ncols() == 0 {
        return Err(Error::EmptyInput);
    }
    options.validate()?;
    let left_rows: Result<Vec<_>, _> = (0..left.nrows())
        .map(|i| {
            let row = matrix_row(left, i);
            validate_series(&row)?;
            Ok(row)
        })
        .collect();
    let right_rows: Result<Vec<_>, _> = (0..right.nrows())
        .map(|i| {
            let row = matrix_row(right, i);
            validate_series(&row)?;
            Ok(row)
        })
        .collect();
    let left_rows = left_rows?;
    let right_rows = right_rows?;
    let mut output = Mat::<f64>::zeros(left.nrows(), right.nrows());
    for i in 0..left.nrows() {
        for j in 0..right.nrows() {
            output[(i, j)] = dtw_with_options(&left_rows[i], &right_rows[j], options)?.distance;
        }
    }
    Ok(output)
}

/// Pairwise DTW for collections of independently sized scalar series.
pub fn pairwise_dtw_series(
    left: &[Vec<f64>],
    right: &[Vec<f64>],
    options: DtwOptions,
) -> Result<Mat<f64>, Error> {
    if left.is_empty() || right.is_empty() {
        return Err(Error::EmptyInput);
    }
    options.validate()?;
    let mut output = Mat::<f64>::zeros(left.len(), right.len());
    for (i, left_series) in left.iter().enumerate() {
        for (j, right_series) in right.iter().enumerate() {
            output[(i, j)] = dtw_with_options(left_series, right_series, options)?.distance;
        }
    }
    Ok(output)
}

/// Pairwise DTW for collections of variable-length multivariate series.
pub fn pairwise_multivariate_dtw(
    left: &[Mat<f64>],
    right: &[Mat<f64>],
    options: DtwOptions,
) -> Result<Mat<f64>, Error> {
    if left.is_empty() || right.is_empty() {
        return Err(Error::EmptyInput);
    }
    options.validate()?;
    let mut output = Mat::<f64>::zeros(left.len(), right.len());
    for (i, left_series) in left.iter().enumerate() {
        for (j, right_series) in right.iter().enumerate() {
            output[(i, j)] =
                dtw_multivariate_with_options(left_series, right_series, options)?.distance;
        }
    }
    Ok(output)
}

/// DTW barycenter averaging for equal-length rows.
///
/// The initialization is the pointwise mean.  Each iteration aligns every
/// series to the current center and averages observations assigned to each
/// center position.
pub fn dtw_barycenter_averaging(
    series: &Mat<f64>,
    max_iterations: usize,
    tolerance: f64,
    options: DtwOptions,
) -> Result<Vec<f64>, Error> {
    if series.nrows() == 0 || series.ncols() == 0 {
        return Err(Error::EmptyInput);
    }
    if !tolerance.is_finite() || tolerance < 0.0 {
        return Err(Error::InvalidParameter(
            "tolerance must be finite and non-negative",
        ));
    }
    let rows: Result<Vec<_>, _> = (0..series.nrows())
        .map(|i| {
            let row = matrix_row(series, i);
            validate_series(&row)?;
            Ok(row)
        })
        .collect();
    let rows = rows?;
    let length = series.ncols();
    let mut center = vec![0.0; length];
    for row in &rows {
        for (target, value) in center.iter_mut().zip(row) {
            *target += *value / rows.len() as f64;
        }
    }
    for _ in 0..max_iterations {
        let mut sums = vec![0.0; length];
        let mut counts = vec![0_usize; length];
        for row in &rows {
            let alignment = dtw_with_options(&center, row, options)?;
            for (center_index, row_index) in alignment.path {
                sums[center_index] += row[row_index];
                counts[center_index] += 1;
            }
        }
        let mut delta = 0.0;
        for index in 0..length {
            if counts[index] > 0 {
                let next = sums[index] / counts[index] as f64;
                let difference = next - center[index];
                delta += difference * difference;
                center[index] = next;
            }
        }
        if delta.sqrt() <= tolerance {
            break;
        }
    }
    Ok(center)
}

/// Edit distance with real penalty (ERP).
pub fn erp_distance(left: &[f64], right: &[f64], gap: f64) -> Result<f64, Error> {
    validate_series(left)?;
    validate_series(right)?;
    if !gap.is_finite() {
        return Err(Error::InvalidParameter("gap must be finite"));
    }
    let mut previous = vec![0.0; right.len() + 1];
    for j in 1..=right.len() {
        previous[j] = previous[j - 1] + (right[j - 1] - gap).abs();
    }
    let mut current = vec![0.0; right.len() + 1];
    for i in 1..=left.len() {
        current[0] = previous[0] + (left[i - 1] - gap).abs();
        for j in 1..=right.len() {
            let substitute = previous[j - 1] + (left[i - 1] - right[j - 1]).abs();
            let delete = previous[j] + (left[i - 1] - gap).abs();
            let insert = current[j - 1] + (right[j - 1] - gap).abs();
            current[j] = substitute.min(delete).min(insert);
        }
        std::mem::swap(&mut previous, &mut current);
    }
    Ok(previous[right.len()])
}

/// Discrete Fréchet distance between scalar waveforms.
pub fn discrete_frechet(left: &[f64], right: &[f64]) -> Result<f64, Error> {
    validate_series(left)?;
    validate_series(right)?;
    let columns = right.len();
    let at = |i: usize, j: usize| i * columns + j;
    let mut values = vec![0.0; left.len() * columns];
    for i in 0..left.len() {
        for j in 0..right.len() {
            let local = (left[i] - right[j]).abs();
            values[at(i, j)] = if i == 0 && j == 0 {
                local
            } else if i == 0 {
                values[at(i, j - 1)].max(local)
            } else if j == 0 {
                values[at(i - 1, j)].max(local)
            } else {
                values[at(i - 1, j)]
                    .min(values[at(i - 1, j - 1)])
                    .min(values[at(i, j - 1)])
                    .max(local)
            };
        }
    }
    Ok(values[at(left.len() - 1, right.len() - 1)])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_has_zero_hard_dtw() {
        let series = [1.0, 2.0, 3.0, 2.0];
        let result =
            dtw_with_options(&series, &series, DtwOptions::default()).expect("valid alignment");
        assert_eq!(result.distance, 0.0);
        assert_eq!(result.path.first(), Some(&(0, 0)));
        assert_eq!(result.path.last(), Some(&(3, 3)));
    }

    #[test]
    fn soft_divergence_is_zero_on_identity() {
        let series = [0.0, 1.0, 2.0];
        let result =
            soft_dtw_divergence(&series, &series, 0.1, LocalCost::Squared).expect("valid soft DTW");
        assert!(result.abs() < 1e-12);
    }

    #[test]
    fn constrained_path_respects_window() {
        let left = [0.0, 1.0, 2.0, 3.0];
        let right = [0.0, 1.0, 2.0, 3.0];
        let result = dtw_with_options(
            &left,
            &right,
            DtwOptions {
                window: Some(0),
                ..DtwOptions::default()
            },
        )
        .expect("diagonal alignment");
        assert!(result.path.iter().all(|(i, j)| i == j));
    }

    #[test]
    fn dba_preserves_identical_rows() {
        let input = Mat::<f64>::from_fn(3, 3, |_, j| j as f64);
        let center = dtw_barycenter_averaging(&input, 10, 1e-12, DtwOptions::default())
            .expect("valid barycenter");
        assert_eq!(center, vec![0.0, 1.0, 2.0]);
    }

    #[test]
    fn multivariate_dtw_supports_different_lengths() {
        let left = Mat::<f64>::from_fn(3, 2, |i, j| [[0.0, 0.0], [1.0, 2.0], [2.0, 4.0]][i][j]);
        let right = Mat::<f64>::from_fn(4, 2, |i, j| {
            [[0.0, 0.0], [1.0, 2.0], [1.0, 2.0], [2.0, 4.0]][i][j]
        });
        let alignment = dtw_multivariate_with_options(
            &left,
            &right,
            DtwOptions {
                local_cost: LocalCost::Squared,
                ..DtwOptions::default()
            },
        )
        .unwrap();
        assert_eq!(alignment.distance, 0.0);
        assert_eq!(alignment.path.last(), Some(&(2, 3)));
    }

    #[test]
    fn variable_length_pairwise_adapter_returns_all_pairs() {
        let left = vec![vec![0.0, 1.0], vec![0.0, 1.0, 2.0]];
        let right = vec![vec![0.0], vec![1.0, 2.0]];
        let distances = pairwise_dtw_series(&left, &right, DtwOptions::default()).unwrap();
        assert_eq!((distances.nrows(), distances.ncols()), (2, 2));
        assert!(distances[(0, 0)].is_finite());
        assert!(distances[(1, 1)].is_finite());
    }

    #[test]
    fn erp_and_discrete_frechet_match_hand_calculations() {
        let left = [0.0, 1.0, 2.0];
        let right = [0.0, 2.0];
        assert!((erp_distance(&left, &right, 0.0).unwrap() - 1.0).abs() < 1e-12);
        assert!((discrete_frechet(&left, &right).unwrap() - 1.0).abs() < 1e-12);
    }
}
