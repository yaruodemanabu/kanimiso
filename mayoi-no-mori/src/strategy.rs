//! Node-local candidate policies delegated to `oldwood`'s gain evaluator.

use amatsuki::{seed_rng, Rng};
use oldwood::{Exhaustive, SplitContext, SplitStrategy};

use crate::random::shuffle;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ThresholdPolicy {
    Exhaustive,
    OneUniform,
}

#[derive(Clone, Debug)]
pub(crate) struct RandomSplitStrategy {
    seed: u64,
    feature_count: usize,
    threshold_policy: ThresholdPolicy,
}

impl RandomSplitStrategy {
    pub(crate) fn new(seed: u64, feature_count: usize, threshold_policy: ThresholdPolicy) -> Self {
        Self {
            seed,
            feature_count,
            threshold_policy,
        }
    }

    fn stream(&self, context: SplitContext, feature: Option<usize>) -> amatsuki::ChaCha8Rng {
        let mut rng = seed_rng(self.seed);
        let feature = feature.map_or(u64::MAX, stream_word);
        let stream = stream_word(context.node_id)
            ^ stream_word(context.depth).rotate_left(17)
            ^ stream_word(context.sample_count).rotate_left(33)
            ^ feature.rotate_left(49);
        rng.set_stream(stream);
        rng
    }
}

fn stream_word(value: usize) -> u64 {
    u64::try_from(value).expect("supported pointer widths fit into u64")
}

impl SplitStrategy for RandomSplitStrategy {
    fn features(&mut self, context: SplitContext, total_features: usize, output: &mut Vec<usize>) {
        let mut candidates: Vec<usize> = (0..total_features).collect();
        let mut rng = self.stream(context, None);
        shuffle(&mut rng, &mut candidates);
        candidates.truncate(self.feature_count.min(total_features));
        output.extend(candidates);
    }

    fn thresholds(
        &mut self,
        context: SplitContext,
        feature: usize,
        unique_values: &[f64],
        output: &mut Vec<f64>,
    ) {
        match self.threshold_policy {
            ThresholdPolicy::Exhaustive => {
                Exhaustive.thresholds(context, feature, unique_values, output);
            }
            ThresholdPolicy::OneUniform => {
                let low = unique_values[0];
                let high = unique_values[unique_values.len() - 1];
                let mut rng = self.stream(context, Some(feature));
                let unit = loop {
                    let value = rng.next_f64();
                    if value > 0.0 {
                        break value;
                    }
                };
                let direct = low + (high - low) * unit;
                let threshold = if direct.is_finite() {
                    direct
                } else {
                    low * (1.0 - unit) + high * unit
                };
                if threshold.is_finite() {
                    output.push(if low <= threshold && threshold < high {
                        threshold
                    } else {
                        low
                    });
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn proposals_are_reproducible_per_node_and_not_call_order() {
        let context = SplitContext {
            node_id: 5,
            depth: 2,
            sample_count: 17,
        };
        let mut first = RandomSplitStrategy::new(42, 3, ThresholdPolicy::OneUniform);
        let mut second = first.clone();
        let mut features_a = Vec::new();
        let mut features_b = Vec::new();
        first.features(context, 7, &mut features_a);
        second.features(context, 7, &mut features_b);
        assert_eq!(features_a, features_b);

        let mut thresholds_a = Vec::new();
        let mut thresholds_b = Vec::new();
        second.thresholds(context, 3, &[1.0, 2.0, 8.0], &mut thresholds_b);
        first.thresholds(context, 3, &[1.0, 2.0, 8.0], &mut thresholds_a);
        assert_eq!(thresholds_a, thresholds_b);
        assert!(1.0 < thresholds_a[0] && thresholds_a[0] < 8.0);
    }

    #[test]
    fn one_uniform_candidate_handles_adjacent_binary64_values() {
        let low = 1.0_f64;
        let high = f64::from_bits(low.to_bits() + 1);
        let context = SplitContext {
            node_id: 0,
            depth: 0,
            sample_count: 2,
        };
        let mut strategy = RandomSplitStrategy::new(9, 1, ThresholdPolicy::OneUniform);
        let mut thresholds = Vec::new();
        strategy.thresholds(context, 0, &[low, high], &mut thresholds);
        assert_eq!(thresholds, vec![low]);
    }
}
