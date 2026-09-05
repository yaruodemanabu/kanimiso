use crate::{Error, Result};
use core::cmp::Ordering;

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct CompensatedSum {
    sum: f64,
    compensation: f64,
}

impl CompensatedSum {
    pub(crate) fn add(&mut self, value: f64, operation: &'static str) -> Result<()> {
        let next = self.sum + value;
        if !next.is_finite() {
            return Err(Error::NumericalOverflow { operation });
        }
        if self.sum.abs() >= value.abs() {
            self.compensation += (self.sum - next) + value;
        } else {
            self.compensation += (value - next) + self.sum;
        }
        if !self.compensation.is_finite() {
            return Err(Error::NumericalOverflow { operation });
        }
        self.sum = next;
        Ok(())
    }

    pub(crate) fn total(self, operation: &'static str) -> Result<f64> {
        let total = self.sum + self.compensation;
        if total.is_finite() {
            Ok(total)
        } else {
            Err(Error::NumericalOverflow { operation })
        }
    }

    fn scale(&mut self, factor: f64, operation: &'static str) -> Result<()> {
        self.sum *= factor;
        self.compensation *= factor;
        if self.sum.is_finite() && self.compensation.is_finite() {
            Ok(())
        } else {
            Err(Error::NumericalOverflow { operation })
        }
    }

    pub(crate) fn merge(&mut self, other: Self, operation: &'static str) -> Result<()> {
        self.add(other.sum, operation)?;
        self.add(other.compensation, operation)
    }

    pub(crate) fn cmp(self, other: Self) -> Ordering {
        let mut difference = self;
        difference
            .add(-other.sum, "compensated comparison")
            .expect("finite expansion difference");
        difference
            .add(-other.compensation, "compensated comparison")
            .expect("finite expansion difference");
        difference
            .sum
            .total_cmp(&0.0)
            .then_with(|| difference.compensation.total_cmp(&0.0))
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct ScaledSum {
    scale: f64,
    normalized: CompensatedSum,
}

impl ScaledSum {
    fn add(&mut self, value: f64, operation: &'static str) -> Result<()> {
        if !value.is_finite() {
            return Err(Error::NumericalOverflow { operation });
        }
        if value == 0.0 {
            return Ok(());
        }
        let magnitude = value.abs();
        if self.scale == 0.0 {
            self.scale = magnitude;
        } else if magnitude > self.scale {
            self.normalized.scale(self.scale / magnitude, operation)?;
            self.scale = magnitude;
        }
        self.normalized.add(value / self.scale, operation)
    }

    fn scale_by_ratio(
        &mut self,
        numerator: f64,
        denominator: f64,
        operation: &'static str,
    ) -> Result<()> {
        if self.scale > 0.0 {
            self.scale = multiply_then_divide(self.scale, numerator, denominator, operation)?;
        }
        Ok(())
    }

    fn merge(&mut self, mut other: Self, operation: &'static str) -> Result<()> {
        if other.scale == 0.0 {
            return Ok(());
        }
        if self.scale == 0.0 {
            *self = other;
            return Ok(());
        }
        if other.scale > self.scale {
            self.normalized.scale(self.scale / other.scale, operation)?;
            self.scale = other.scale;
        } else {
            other
                .normalized
                .scale(other.scale / self.scale, operation)?;
        }
        self.normalized.merge(other.normalized, operation)
    }

    fn divided_by(self, denominator: f64, operation: &'static str) -> Result<f64> {
        if self.scale == 0.0 {
            return Ok(0.0);
        }
        multiply_then_divide(
            self.normalized.total(operation)?,
            self.scale,
            denominator,
            operation,
        )
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct StableWeightedMean {
    weight_scale: f64,
    numerator: ScaledSum,
    denominator: CompensatedSum,
    minimum: f64,
    maximum: f64,
    populated: bool,
}

impl StableWeightedMean {
    fn add(&mut self, value: f64, weight: f64) -> Result<()> {
        if weight == 0.0 {
            return Ok(());
        }
        if self.weight_scale == 0.0 {
            self.weight_scale = weight;
        }
        let try_at_current_scale = || -> Result<(ScaledSum, CompensatedSum)> {
            let contribution =
                multiply_then_divide(value, weight, self.weight_scale, "weighted mean numerator")?;
            let normalized_weight = weight / self.weight_scale;
            if !normalized_weight.is_finite() {
                return Err(Error::NumericalOverflow {
                    operation: "weighted mean denominator",
                });
            }
            let mut numerator = self.numerator;
            numerator.add(contribution, "weighted mean numerator")?;
            let mut denominator = self.denominator;
            denominator.add(normalized_weight, "weighted mean denominator")?;
            Ok((numerator, denominator))
        };
        if let Ok((numerator, denominator)) = try_at_current_scale() {
            self.numerator = numerator;
            self.denominator = denominator;
        } else {
            debug_assert!(weight > self.weight_scale);
            self.numerator
                .scale_by_ratio(self.weight_scale, weight, "weighted mean rescaling")?;
            self.denominator
                .scale(self.weight_scale / weight, "weighted mean rescaling")?;
            self.weight_scale = weight;
            let contribution =
                multiply_then_divide(value, weight, self.weight_scale, "weighted mean numerator")?;
            self.numerator
                .add(contribution, "weighted mean numerator")?;
            self.denominator
                .add(weight / self.weight_scale, "weighted mean denominator")?;
        }
        if self.populated {
            self.minimum = self.minimum.min(value);
            self.maximum = self.maximum.max(value);
        } else {
            self.minimum = value;
            self.maximum = value;
            self.populated = true;
        }
        Ok(())
    }

    fn merge(&mut self, mut other: Self) -> Result<()> {
        if !other.populated {
            return Ok(());
        }
        if !self.populated {
            *self = other;
            return Ok(());
        }
        let scale = self.weight_scale.max(other.weight_scale);
        self.numerator
            .scale_by_ratio(self.weight_scale, scale, "weighted mean merge")?;
        self.denominator
            .scale(self.weight_scale / scale, "weighted mean merge")?;
        other
            .numerator
            .scale_by_ratio(other.weight_scale, scale, "weighted mean merge")?;
        other
            .denominator
            .scale(other.weight_scale / scale, "weighted mean merge")?;
        self.numerator
            .merge(other.numerator, "weighted mean merge")?;
        self.denominator
            .merge(other.denominator, "weighted mean merge")?;
        self.weight_scale = scale;
        self.minimum = self.minimum.min(other.minimum);
        self.maximum = self.maximum.max(other.maximum);
        Ok(())
    }

    fn mean(self) -> Result<f64> {
        let denominator = self.denominator.total("weighted mean denominator")?;
        Ok(self
            .numerator
            .divided_by(denominator, "weighted mean numerator")?
            .clamp(self.minimum, self.maximum))
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct WeightedMoments {
    pub(crate) weight: f64,
    pub(crate) mean: f64,
    weight_scale: f64,
    normalized_weight: CompensatedSum,
    normalized_m2: ScaledSum,
    mean_accumulator: StableWeightedMean,
    variance: f64,
}

impl WeightedMoments {
    pub(crate) fn add(&mut self, value: f64, weight: f64) -> Result<()> {
        if weight == 0.0 {
            return Ok(());
        }
        if self.weight == 0.0 {
            self.weight = weight;
            self.mean = value;
            self.weight_scale = weight;
            self.normalized_weight.add(1.0, "weighted moment weight")?;
            self.mean_accumulator.add(value, weight)?;
            return Ok(());
        }
        let total = self.weight + weight;
        if !total.is_finite() {
            return Err(Error::NumericalOverflow {
                operation: "weighted moment update",
            });
        }
        if weight > self.weight_scale {
            self.normalized_weight.scale(
                self.weight_scale / weight,
                "weighted moment weight rescaling",
            )?;
            self.normalized_m2.scale_by_ratio(
                self.weight_scale,
                weight,
                "weighted moment weight rescaling",
            )?;
            self.weight_scale = weight;
        }
        let old_mean = self.mean;
        self.mean_accumulator.add(value, weight)?;
        let mean = self.mean_accumulator.mean()?;
        self.normalized_weight
            .add(weight / self.weight_scale, "weighted moment weight")?;
        let value_scale = value.abs().max(old_mean.abs()).max(mean.abs());
        if value_scale > 0.0 {
            let left_delta = (value / value_scale - old_mean / value_scale).abs();
            let right_delta = (value / value_scale - mean / value_scale).abs();
            let contribution = scaled_product_ratio(
                &[weight, left_delta, right_delta, value_scale, value_scale],
                &[self.weight_scale],
                "weighted second central moment",
            )?;
            self.normalized_m2
                .add(contribution, "weighted second central moment")?;
        }
        self.weight = total;
        self.mean = mean;
        self.variance = normalized_variance(self.normalized_m2, self.normalized_weight)?;
        Ok(())
    }

    pub(crate) fn merge(self, other: Self) -> Result<Self> {
        if self.weight == 0.0 {
            return Ok(other);
        }
        if other.weight == 0.0 {
            return Ok(self);
        }
        let total = self.weight + other.weight;
        if !total.is_finite() {
            return Err(Error::NumericalOverflow {
                operation: "weighted moment merge",
            });
        }
        let weight_scale = self.weight_scale.max(other.weight_scale);
        let mut left_weight = self.normalized_weight;
        left_weight.scale(self.weight_scale / weight_scale, "weighted moment merge")?;
        let mut right_weight = other.normalized_weight;
        right_weight.scale(other.weight_scale / weight_scale, "weighted moment merge")?;
        let mut normalized_weight = left_weight;
        normalized_weight.merge(right_weight, "weighted moment merge")?;

        let mut normalized_m2 = self.normalized_m2;
        normalized_m2.scale_by_ratio(self.weight_scale, weight_scale, "weighted moment merge")?;
        let mut right_m2 = other.normalized_m2;
        right_m2.scale_by_ratio(other.weight_scale, weight_scale, "weighted moment merge")?;
        normalized_m2.merge(right_m2, "weighted moment merge")?;
        let value_scale = self.mean.abs().max(other.mean.abs());
        if value_scale > 0.0 {
            let delta = (other.mean / value_scale - self.mean / value_scale).abs();
            let cross = scaled_product_ratio(
                &[
                    self.weight,
                    other.weight,
                    delta,
                    delta,
                    value_scale,
                    value_scale,
                ],
                &[total, weight_scale],
                "weighted moment merge cross term",
            )?;
            normalized_m2.add(cross, "weighted moment merge cross term")?;
        }
        let mut mean_accumulator = self.mean_accumulator;
        mean_accumulator.merge(other.mean_accumulator)?;
        let mean = mean_accumulator.mean()?;
        let variance = normalized_variance(normalized_m2, normalized_weight)?;
        Ok(Self {
            weight: total,
            mean,
            weight_scale,
            normalized_weight,
            normalized_m2,
            mean_accumulator,
            variance,
        })
    }

    pub(crate) fn variance(self) -> f64 {
        if self.weight == 0.0 {
            0.0
        } else {
            self.variance
        }
    }
}

fn normalized_variance(normalized_m2: ScaledSum, normalized_weight: CompensatedSum) -> Result<f64> {
    let weight = normalized_weight.total("weighted moment weight")?;
    normalized_m2.divided_by(weight, "weighted variance")
}

fn multiply_then_divide(
    left: f64,
    right: f64,
    denominator: f64,
    operation: &'static str,
) -> Result<f64> {
    if !left.is_finite() || !right.is_finite() || !denominator.is_finite() || denominator <= 0.0 {
        return Err(Error::NumericalOverflow { operation });
    }
    let product = left * right;
    if product.is_finite() && (product != 0.0 || left == 0.0 || right == 0.0) {
        return Ok(product / denominator);
    }
    scaled_product_ratio(&[left, right], &[denominator], operation)
}

fn scaled_product_ratio(
    numerators: &[f64],
    denominators: &[f64],
    operation: &'static str,
) -> Result<f64> {
    let mut sign_negative = false;
    let mut mantissa = 1.0;
    let mut exponent = 0_i64;
    for &value in numerators {
        if !value.is_finite() {
            return Err(Error::NumericalOverflow { operation });
        }
        if value == 0.0 {
            return Ok(value);
        }
        sign_negative ^= value.is_sign_negative();
        let (factor, factor_exponent) = decompose_positive(value.abs());
        mantissa *= factor;
        exponent += i64::from(factor_exponent);
        normalize(&mut mantissa, &mut exponent);
    }
    for &value in denominators {
        if !value.is_finite() || value <= 0.0 {
            return Err(Error::NumericalOverflow { operation });
        }
        let (factor, factor_exponent) = decompose_positive(value);
        mantissa /= factor;
        exponent -= i64::from(factor_exponent);
        normalize(&mut mantissa, &mut exponent);
    }
    let magnitude = compose_positive(mantissa, exponent, operation)?;
    Ok(if sign_negative { -magnitude } else { magnitude })
}

#[allow(clippy::cast_possible_wrap, clippy::cast_precision_loss)]
fn decompose_positive(value: f64) -> (f64, i32) {
    debug_assert!(value.is_finite() && value > 0.0);
    let bits = value.to_bits();
    let exponent_bits = ((bits >> 52) & 0x7ff) as i32;
    let fraction = bits & ((1_u64 << 52) - 1);
    if exponent_bits == 0 {
        let leading = 63_i32 - fraction.leading_zeros() as i32;
        let unit = (1_u64 << leading) as f64;
        (fraction as f64 / unit, leading - 1074)
    } else {
        (
            1.0 + fraction as f64 / (1_u64 << 52) as f64,
            exponent_bits - 1023,
        )
    }
}

fn normalize(mantissa: &mut f64, exponent: &mut i64) {
    if *mantissa >= 2.0 {
        *mantissa *= 0.5;
        *exponent += 1;
    } else if *mantissa < 1.0 {
        *mantissa *= 2.0;
        *exponent -= 1;
    }
}

#[allow(clippy::cast_sign_loss)]
fn compose_positive(mantissa: f64, exponent: i64, operation: &'static str) -> Result<f64> {
    let value = if exponent > 1023 {
        f64::INFINITY
    } else if exponent >= -1022 {
        let factor = f64::from_bits(((exponent + 1023) as u64) << 52);
        mantissa * factor
    } else if exponent >= -1074 {
        let factor = f64::from_bits(1_u64 << (exponent + 1074));
        mantissa * factor
    } else if exponent == -1075 {
        (mantissa * 0.5) * f64::from_bits(1)
    } else {
        0.0
    };
    if value.is_finite() {
        Ok(value)
    } else {
        Err(Error::NumericalOverflow { operation })
    }
}

pub(crate) fn split_threshold(lower: f64, upper: f64) -> f64 {
    debug_assert!(lower.is_finite() && upper.is_finite() && lower < upper);
    let midpoint = if lower.is_sign_negative() == upper.is_sign_negative() {
        lower + (upper - lower) / 2.0
    } else {
        lower / 2.0 + upper / 2.0
    };
    if midpoint.is_finite() && lower < midpoint && midpoint < upper {
        midpoint
    } else {
        lower
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::float_cmp)]

    use super::*;

    #[test]
    fn midpoint_avoids_opposite_sign_overflow() {
        assert_eq!(split_threshold(-f64::MAX, f64::MAX), 0.0);
    }

    #[test]
    fn adjacent_floats_use_the_lower_value_as_a_half_open_threshold() {
        let lower = 1.0_f64;
        let upper = f64::from_bits(lower.to_bits() + 1);
        assert_eq!(split_threshold(lower, upper), lower);
    }

    #[test]
    fn weighted_moments_match_closed_form() {
        let mut moments = WeightedMoments::default();
        moments.add(1.0, 1.0).unwrap();
        moments.add(3.0, 1.0).unwrap();
        assert_eq!(moments.weight, 2.0);
        assert_eq!(moments.mean, 2.0);
        assert_eq!(moments.variance(), 1.0);
    }

    #[test]
    fn weighted_moments_preserve_cancellation_in_every_relevant_order() {
        let large = 8.0e153;
        for values in [
            [large, 1.0, -large],
            [large, -large, 1.0],
            [-large, 1.0, large],
        ] {
            let mut moments = WeightedMoments::default();
            for value in values {
                moments.add(value, 1.0).expect("finite moments");
            }
            assert_eq!(moments.mean, 1.0 / 3.0);
            assert!(moments.variance().is_finite());
            assert!(
                (moments.variance() - (1.28e308 / 3.0)) / moments.variance() <= 8.0 * f64::EPSILON
            );
        }
    }

    #[test]
    fn weighted_moment_merge_matches_the_streaming_result() {
        let large = 8.0e153;
        let mut all = WeightedMoments::default();
        for value in [large, 1.0, -large] {
            all.add(value, 1.0).expect("all");
        }
        let mut left = WeightedMoments::default();
        left.add(large, 1.0).expect("left");
        left.add(1.0, 1.0).expect("left");
        let mut right = WeightedMoments::default();
        right.add(-large, 1.0).expect("right");
        let merged = left.merge(right).expect("merge");
        assert_eq!(merged.mean, all.mean);
        assert!((merged.variance() - all.variance()).abs() / all.variance() <= 8.0 * f64::EPSILON);
    }

    #[test]
    fn mixed_extreme_scales_keep_representable_mean_and_variance() {
        let tiny = f64::from_bits(1);
        let mut moments = WeightedMoments::default();
        moments.add(0.0, f64::MAX).expect("dominant zero");
        moments
            .add(f64::MAX, tiny)
            .expect("tiny extreme observation");
        assert_eq!(moments.mean, tiny);
        assert_eq!(moments.variance(), f64::MAX * tiny);
    }
}
