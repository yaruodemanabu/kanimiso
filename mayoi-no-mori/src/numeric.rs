//! Internal floating-point accumulators for ensemble reductions.

use core::cmp::Ordering;

use crate::{Error, Result};

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct CompensatedSum {
    high: f64,
    low: f64,
}

impl CompensatedSum {
    pub(crate) fn add(&mut self, value: f64, operation: &'static str) -> Result<()> {
        if !value.is_finite() {
            return Err(Error::NumericalOverflow { operation });
        }
        let (sum, first_error) = two_sum(self.high, value);
        let (correction, second_error) = two_sum(self.low, first_error);
        let (high, third_error) = two_sum(sum, correction);
        let (low, fourth_error) = two_sum(third_error, second_error);
        let (high, low) = two_sum(high, low + fourth_error);
        if !high.is_finite() || !low.is_finite() {
            return Err(Error::NumericalOverflow { operation });
        }
        self.high = high;
        self.low = low;
        Ok(())
    }

    pub(crate) fn scale(&mut self, factor: f64, operation: &'static str) -> Result<()> {
        let high = self.high * factor;
        let low = self.low * factor;
        let (high, low) = two_sum(high, low);
        if !high.is_finite() || !low.is_finite() {
            return Err(Error::NumericalOverflow { operation });
        }
        self.high = high;
        self.low = low;
        Ok(())
    }

    pub(crate) fn merge(&mut self, other: Self, operation: &'static str) -> Result<()> {
        self.add(other.high, operation)?;
        self.add(other.low, operation)
    }

    pub(crate) fn value(self, operation: &'static str) -> Result<f64> {
        let value = self.high + self.low;
        if value.is_finite() {
            Ok(value)
        } else {
            Err(Error::NumericalOverflow { operation })
        }
    }

    pub(crate) fn cmp(self, other: Self) -> Ordering {
        let mut difference = self;
        // Both operands are already finite expansions, so cancellation cannot
        // overflow. Preserve the low component instead of first rounding each
        // expansion to a single f64.
        difference
            .add(-other.high, "compensated comparison")
            .expect("finite expansion difference");
        difference
            .add(-other.low, "compensated comparison")
            .expect("finite expansion difference");
        difference
            .high
            .total_cmp(&0.0)
            .then_with(|| difference.low.total_cmp(&0.0))
    }

    pub(crate) fn ratio(self, denominator: Self, operation: &'static str) -> Result<f64> {
        if denominator.high <= 0.0 {
            return Err(Error::NumericalOverflow { operation });
        }
        let leading = self.high / denominator.high;
        let mut remainder = CompensatedSum::default();
        remainder.add((-leading).mul_add(denominator.high, self.high), operation)?;
        remainder.add(self.low, operation)?;
        remainder.add(-leading * denominator.low, operation)?;
        let correction = remainder.value(operation)? / denominator.high;
        let rounded = leading + correction;
        // Nearest rounding can erase a strict sub-ULP ordering at an
        // algorithmic boundary (for example SAMME's chance limit). When the
        // retained expansion proves the exact quotient lies on one side of
        // `leading`, return the adjacent value on that side.
        let value = if correction < 0.0 && rounded >= leading {
            next_down(leading)
        } else if correction > 0.0 && rounded <= leading {
            next_up(leading)
        } else {
            rounded
        };
        if value.is_finite() {
            Ok(value)
        } else {
            Err(Error::NumericalOverflow { operation })
        }
    }
}

fn next_up(value: f64) -> f64 {
    if value == f64::INFINITY {
        value
    } else if value == -0.0 {
        f64::from_bits(1)
    } else if value >= 0.0 {
        f64::from_bits(value.to_bits() + 1)
    } else {
        f64::from_bits(value.to_bits() - 1)
    }
}

fn next_down(value: f64) -> f64 {
    -next_up(-value)
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct ScaledSum {
    scale: f64,
    normalized: CompensatedSum,
}

impl ScaledSum {
    pub(crate) fn add(&mut self, value: f64, operation: &'static str) -> Result<()> {
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

    pub(crate) fn value(self, operation: &'static str) -> Result<f64> {
        if self.scale == 0.0 {
            return Ok(0.0);
        }
        let value = self.normalized.value(operation)? * self.scale;
        if value.is_finite() {
            Ok(value)
        } else {
            Err(Error::NumericalOverflow { operation })
        }
    }

    pub(crate) fn merge(&mut self, other: Self, operation: &'static str) -> Result<()> {
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
            self.normalized.merge(other.normalized, operation)
        } else {
            let mut normalized = other.normalized;
            normalized.scale(other.scale / self.scale, operation)?;
            self.normalized.merge(normalized, operation)
        }
    }

    pub(crate) fn components(self, operation: &'static str) -> Result<(f64, f64)> {
        Ok((self.normalized.value(operation)?, self.scale))
    }

    fn scale_by_ratio(
        &mut self,
        numerator: f64,
        denominator: f64,
        operation: &'static str,
    ) -> Result<()> {
        if self.scale == 0.0 {
            return Ok(());
        }
        self.scale = multiply_then_divide(self.scale, numerator, denominator, operation)?;
        Ok(())
    }

    fn divided_by(self, denominator: f64, operation: &'static str) -> Result<f64> {
        if self.scale == 0.0 {
            return Ok(0.0);
        }
        multiply_then_divide(
            self.normalized.value(operation)?,
            self.scale,
            denominator,
            operation,
        )
    }

    pub(crate) fn mean(self, count: usize, operation: &'static str) -> Result<f64> {
        if count == 0 || self.scale == 0.0 {
            return Ok(0.0);
        }
        self.divided_by(crate::options::count_as_f64(count), operation)
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct StableWeightedMean {
    weight_scale: f64,
    numerator: ScaledSum,
    denominator: CompensatedSum,
    minimum: f64,
    maximum: f64,
    populated: bool,
}

impl StableWeightedMean {
    pub(crate) fn add(&mut self, value: f64, weight: f64, operation: &'static str) -> Result<()> {
        if weight <= 0.0 {
            return Ok(());
        }
        if !value.is_finite() || !weight.is_finite() {
            return Err(Error::NumericalOverflow { operation });
        }
        if weight > self.weight_scale {
            if self.weight_scale > 0.0 {
                self.numerator
                    .scale_by_ratio(self.weight_scale, weight, operation)?;
                self.denominator
                    .scale(self.weight_scale / weight, operation)?;
            }
            self.weight_scale = weight;
        }
        let normalized_weight = weight / self.weight_scale;
        let contribution = multiply_then_divide(value, weight, self.weight_scale, operation)?;
        self.numerator.add(contribution, operation)?;
        self.denominator.add(normalized_weight, operation)?;
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

    pub(crate) fn mean(self, operation: &'static str) -> Result<Option<f64>> {
        if !self.populated {
            return Ok(None);
        }
        let denominator = self.denominator.value(operation)?;
        if denominator <= 0.0 {
            return Err(Error::NumericalOverflow { operation });
        }
        let value = self.numerator.divided_by(denominator, operation)?;
        if !value.is_finite() {
            return Err(Error::NumericalOverflow { operation });
        }
        // A weighted mean is in the convex hull. This only trims a last-bit
        // excursion introduced by the final division/multiplication.
        Ok(Some(value.clamp(self.minimum, self.maximum)))
    }
}

fn multiply_then_divide(
    left: f64,
    right: f64,
    denominator: f64,
    operation: &'static str,
) -> Result<f64> {
    scaled_product_ratio(&[left, right], &[denominator], operation)
}

pub(crate) fn scaled_product_ratio(
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

fn two_sum(left: f64, right: f64) -> (f64, f64) {
    let sum = left + right;
    let right_virtual = sum - left;
    let error = (left - (sum - right_virtual)) + (right - right_virtual);
    (sum, error)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::float_cmp)]

    use super::*;

    #[test]
    fn scaled_sum_preserves_a_small_cancellation_residual() {
        let large = 8.0e153;
        let mut sum = ScaledSum::default();
        for value in [large, 1.0, -large] {
            sum.add(value, "test sum").expect("representable sum");
        }
        assert_eq!(
            sum.value("test sum").expect("sum").to_bits(),
            1.0f64.to_bits()
        );
        assert_eq!(sum.mean(3, "test mean").expect("mean"), 1.0 / 3.0);
    }

    #[test]
    fn stable_weighted_mean_preserves_cancellation_and_extreme_equal_values() {
        let large = 8.0e153;
        for values in [[large, 1.0, -large], [large, -large, 1.0]] {
            let mut mean = StableWeightedMean::default();
            for value in values {
                mean.add(value, 1.0, "test mean").expect("add");
            }
            assert_eq!(mean.mean("test mean").expect("mean"), Some(1.0 / 3.0));
        }

        let mut mean = StableWeightedMean::default();
        mean.add(f64::MAX, f64::MAX, "test mean").expect("first");
        mean.add(f64::MAX, f64::MAX, "test mean").expect("second");
        assert_eq!(mean.mean("test mean").expect("mean"), Some(f64::MAX));

        let tiny = f64::from_bits(1);
        let mut tiny_mean = StableWeightedMean::default();
        tiny_mean.add(0.0, f64::MAX, "test mean").expect("zero");
        tiny_mean
            .add(f64::MAX, tiny, "test mean")
            .expect("tiny mass");
        assert_eq!(tiny_mean.mean("test mean").expect("mean"), Some(tiny));

        let mut subnormal = StableWeightedMean::default();
        subnormal.add(tiny, tiny, "test mean").expect("subnormal");
        assert_eq!(subnormal.mean("test mean").expect("mean"), Some(tiny));
    }

    #[test]
    fn scaled_mean_rescues_a_subnormal_normalized_residual() {
        let tiny = f64::from_bits(1);
        let mut sum = ScaledSum::default();
        for value in [f64::MAX, -f64::MAX, f64::MAX * tiny] {
            sum.add(value, "test mean").expect("add");
        }
        let expected = (f64::MAX * tiny) / 3.0;
        assert_eq!(sum.mean(3, "test mean").expect("mean"), expected);
    }

    #[test]
    fn compensated_comparison_keeps_a_sub_ulp_difference() {
        let mut left = CompensatedSum::default();
        left.add(1.0, "test").expect("one");
        let mut right = left;
        right.add(2.0f64.powi(-53), "test").expect("residual");
        assert_eq!(left.cmp(right), Ordering::Less);
    }
}
