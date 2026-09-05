use crate::numeric::split_threshold;

/// Stable identity of the node currently asking for split candidates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SplitContext {
    /// Arena index assigned before candidate generation.
    pub node_id: usize,
    /// Root is depth zero.
    pub depth: usize,
    /// Number of positive-weight rows at the node.
    pub sample_count: usize,
}

/// Candidate policy used by the single CART gain evaluator.
///
/// Implementations may use randomness externally, but `oldwood` validates,
/// sorts, and deduplicates every returned candidate. This keeps all impurity
/// calculations, tie-breaking, and arena construction in one implementation.
pub trait SplitStrategy {
    /// Appends feature indices to inspect. Returning none makes a leaf.
    fn features(&mut self, context: SplitContext, total_features: usize, output: &mut Vec<usize>);

    /// Appends finite thresholds in `[first, last)`. `unique_values` is sorted
    /// and contains at least two values; prediction sends `value <= threshold`
    /// left, so the half-open contract also supports adjacent binary64 values.
    fn thresholds(
        &mut self,
        context: SplitContext,
        feature: usize,
        unique_values: &[f64],
        output: &mut Vec<f64>,
    );
}

/// Exhaustive deterministic CART candidate policy.
#[derive(Clone, Copy, Debug, Default)]
pub struct Exhaustive;

impl SplitStrategy for Exhaustive {
    fn features(&mut self, _context: SplitContext, total_features: usize, output: &mut Vec<usize>) {
        output.extend(0..total_features);
    }

    fn thresholds(
        &mut self,
        _context: SplitContext,
        _feature: usize,
        unique_values: &[f64],
        output: &mut Vec<f64>,
    ) {
        output.extend(
            unique_values
                .windows(2)
                .map(|pair| split_threshold(pair[0], pair[1])),
        );
    }
}
