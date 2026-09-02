//! Tiny deterministic PRNG (xorshift64*). Pure Rust, no `rand` crate.

const ZERO_SEED_FALLBACK: u64 = 0x9e37_79b9_7f4a_7c15;
const XORSHIFT_MULTIPLIER: u64 = 0x2545_f491_4f6c_dd1d;
const F64_SIGNIFICAND_BITS: u32 = 53;
const F64_UNIT_SCALE: f64 = 1.0 / ((1u64 << F64_SIGNIFICAND_BITS) as f64);

// Constants from Hoermann's transformed-rejection (PTRS) Poisson sampler.
// The notation follows the original algorithm so its branches can be audited
// directly against the paper and independent implementations.
const PTRS_MIN_LAMBDA: f64 = 10.0;
const PTRS_B_OFFSET: f64 = 0.931;
const PTRS_B_SCALE: f64 = 2.53;
const PTRS_A_OFFSET: f64 = -0.059;
const PTRS_A_SCALE: f64 = 0.02483;
const PTRS_INV_ALPHA_OFFSET: f64 = 1.1239;
const PTRS_INV_ALPHA_SCALE: f64 = 1.1328;
const PTRS_INV_ALPHA_SHIFT: f64 = 3.4;
const PTRS_V_R_OFFSET: f64 = 0.9277;
const PTRS_V_R_SCALE: f64 = 3.6224;
const PTRS_V_R_SHIFT: f64 = 2.0;
const PTRS_U_SHIFT: f64 = 0.5;
const PTRS_K_SHIFT: f64 = 0.43;
const PTRS_FAST_U: f64 = 0.07;
const PTRS_REJECT_U: f64 = 0.013;
// At this rate the standard deviation is 2^16, while every integer around the
// mass remains exactly representable by f64. The first non-consecutive f64
// integer is over 10^11 standard deviations away, and the log-PMF cancellation
// scale stays below roughly 3e-5 in binary64.
const MAX_RELIABLE_POISSON_RATE: f64 = 4_294_967_296.0;
const MAX_CONSECUTIVE_F64_INTEGER: f64 = 9_007_199_254_740_991.0;

/// Reproducible xorshift64* generator used by stochastic algorithms.
#[derive(Clone, Debug)]
pub struct Rng {
    state: u64,
}

impl Rng {
    /// Seed the generator.
    ///
    /// Every nonzero seed denotes a distinct xorshift state. Zero alone is
    /// replaced by a fixed nonzero state because zero is absorbing for
    /// xorshift64*. This means, in particular, that adjacent even and odd seeds
    /// no longer select the same stream.
    pub fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { ZERO_SEED_FALLBACK } else { seed },
        }
    }

    /// Next `u64`.
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(XORSHIFT_MULTIPLIER)
    }

    /// Uniform in `[0, 1)`.
    pub fn uniform(&mut self) -> f64 {
        let u = self.next_u64() >> (u64::BITS - F64_SIGNIFICAND_BITS);
        (u as f64) * F64_UNIT_SCALE
    }

    /// Uniform in the finite, non-empty interval `[lo, hi)`.
    ///
    /// Opposite-sign endpoints whose mathematical width exceeds `f64::MAX`
    /// are interpolated as a convex combination, avoiding overflow in
    /// `hi - lo`. If final rounding reaches the excluded upper endpoint, the
    /// immediately preceding representable value is returned. Invalid bounds
    /// panic before consuming random state because this infallible legacy API
    /// cannot carry a structured parameter error.
    #[track_caller]
    pub fn uniform_range(&mut self, lo: f64, hi: f64) -> f64 {
        assert!(
            lo.is_finite() && hi.is_finite() && lo < hi,
            "uniform range bounds must be finite and satisfy lo < hi"
        );
        let unit = self.uniform();
        let width = hi - lo;
        let value = if width.is_finite() {
            lo + width * unit
        } else {
            lo * (1.0 - unit) + hi * unit
        };
        debug_assert!(value.is_finite() && value >= lo);
        if value < hi {
            value
        } else {
            previous_f64(hi)
        }
    }

    /// Standard normal via Box–Muller.
    pub fn standard_normal(&mut self) -> f64 {
        // `1 - U` is in `(0, 1]`, so Box–Muller never evaluates `ln(0)` and
        // does not need to floor a valid random variate.
        let u = 1.0 - self.uniform();
        let v = self.uniform();
        (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
    }

    /// Draw from the `Poisson(λ)` mass function using floating-point
    /// transformed rejection.
    ///
    /// Knuth's inversion method is used for small rates and Hoermann's PTRS
    /// transformed-rejection method for larger rates. Unlike a rounded-normal
    /// approximation, PTRS preserves the Poisson mass function and skewness.
    ///
    /// `λ = 0` is the valid degenerate distribution at zero. This method
    /// panics before consuming random state when `λ` is negative, non-finite,
    /// or above `2^32`. The upper limit keeps the distribution's effective
    /// support inside the consecutive-integer range of `f64`; it is not a claim
    /// that a finite-word generator can represent an unbounded distribution
    /// with real-arithmetic exactness. The existing `u64` return type cannot
    /// carry a structured invalid-parameter error.
    #[track_caller]
    pub fn poisson(&mut self, lambda: f64) -> u64 {
        assert!(
            lambda.is_finite() && lambda >= 0.0 && lambda <= MAX_RELIABLE_POISSON_RATE,
            "Poisson rate must be finite and in 0..={MAX_RELIABLE_POISSON_RATE}"
        );
        if lambda == 0.0 {
            return 0;
        }
        if lambda >= PTRS_MIN_LAMBDA {
            return self.poisson_ptrs(lambda);
        }

        let l = (-lambda).exp();
        let mut k = 0u64;
        let mut p = 1.0;
        loop {
            p *= self.uniform();
            if p <= l {
                return k;
            }
            k += 1;
        }
    }

    /// Integer in `0..n`.
    ///
    /// Rejection removes the incomplete residue classes that make `% n`
    /// biased whenever `n` does not divide the generator's period. xorshift64*
    /// visits every *nonzero* word exactly once per period, so subtracting one
    /// first maps that source to the contiguous range `0..u64::MAX` before the
    /// incomplete upper residue classes are discarded. Supplying an empty
    /// range is a caller error.
    #[track_caller]
    pub fn below(&mut self, n: usize) -> usize {
        assert!(n > 0, "cannot sample from the empty range 0..0");
        let bound = n as u64;
        let complete_zone = u64::MAX - u64::MAX % bound;
        loop {
            let zero_based_word = self.next_u64() - 1;
            if zero_based_word < complete_zone {
                return (zero_based_word % bound) as usize;
            }
        }
    }

    /// Fisher–Yates shuffle.
    pub fn shuffle<T>(&mut self, xs: &mut [T]) {
        let n = xs.len();
        for i in (1..n).rev() {
            let j = self.below(i + 1);
            xs.swap(i, j);
        }
    }

    /// Sample `k` distinct indices from `0..n`.
    pub fn sample_indices(&mut self, n: usize, k: usize) -> Vec<usize> {
        let mut idx: Vec<usize> = (0..n).collect();
        self.shuffle(&mut idx);
        idx.truncate(k.min(n));
        idx
    }

    fn poisson_ptrs(&mut self, lambda: f64) -> u64 {
        let sqrt_lambda = lambda.sqrt();
        let log_lambda = lambda.ln();
        let b = PTRS_B_OFFSET + PTRS_B_SCALE * sqrt_lambda;
        let a = PTRS_A_OFFSET + PTRS_A_SCALE * b;
        let inverse_alpha =
            PTRS_INV_ALPHA_OFFSET + PTRS_INV_ALPHA_SCALE / (b - PTRS_INV_ALPHA_SHIFT);
        let fast_acceptance = PTRS_V_R_OFFSET - PTRS_V_R_SCALE / (b - PTRS_V_R_SHIFT);

        loop {
            let u = self.uniform() - PTRS_U_SHIFT;
            let v = self.uniform();
            let u_distance = PTRS_U_SHIFT - u.abs();
            if u_distance == 0.0 {
                continue;
            }

            let candidate = ((2.0 * a / u_distance + b) * u + lambda + PTRS_K_SHIFT).floor();
            if candidate < 0.0 || candidate > MAX_CONSECUTIVE_F64_INTEGER {
                continue;
            }
            if u_distance >= PTRS_FAST_U && v <= fast_acceptance {
                return candidate as u64;
            }
            if u_distance < PTRS_REJECT_U && v > u_distance {
                continue;
            }

            let log_proposal = if v == 0.0 {
                f64::NEG_INFINITY
            } else {
                (v * inverse_alpha / (a / (u_distance * u_distance) + b)).ln()
            };
            let log_mass =
                -lambda + candidate * log_lambda - crate::special::ln_gamma(candidate + 1.0);
            if log_proposal <= log_mass {
                return candidate as u64;
            }
        }
    }
}

fn previous_f64(value: f64) -> f64 {
    debug_assert!(value.is_finite());
    if value == 0.0 {
        -f64::from_bits(1)
    } else if value.is_sign_positive() {
        f64::from_bits(value.to_bits() - 1)
    } else {
        f64::from_bits(value.to_bits() + 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seeds_are_deterministic_without_aliasing_even_and_odd_states() {
        let mut a = Rng::new(7);
        let mut b = Rng::new(7);
        assert_eq!(a.next_u64(), b.next_u64());

        let mut even = Rng::new(2);
        let mut odd = Rng::new(3);
        let even_stream: Vec<_> = (0..8).map(|_| even.next_u64()).collect();
        let odd_stream: Vec<_> = (0..8).map(|_| odd.next_u64()).collect();
        assert_ne!(even_stream, odd_stream);

        let mut zero = Rng::new(0);
        let mut fallback = Rng::new(ZERO_SEED_FALLBACK);
        for _ in 0..8 {
            assert_eq!(zero.next_u64(), fallback.next_u64());
        }
    }

    #[test]
    fn uniform_and_bounded_draws_respect_their_support() {
        let mut rng = Rng::new(0x7b1d_4a53);
        for _ in 0..100_000 {
            let u = rng.uniform();
            assert!((0.0..1.0).contains(&u));
        }
        for &bound in &[1, 2, 3, 7, 256, 65_537, usize::MAX] {
            for _ in 0..10_000 {
                assert!(rng.below(bound) < bound);
            }
        }
    }

    #[test]
    fn uniform_range_handles_extreme_and_adjacent_finite_endpoints() {
        let mut extreme = Rng::new(0x2a61_7e5d);
        let mut saw_negative = false;
        let mut saw_positive = false;
        for _ in 0..100_000 {
            let value = extreme.uniform_range(-f64::MAX, f64::MAX);
            assert!(value.is_finite());
            assert!((-f64::MAX..f64::MAX).contains(&value));
            saw_negative |= value.is_sign_negative();
            saw_positive |= value.is_sign_positive() && value != 0.0;
        }
        assert!(saw_negative && saw_positive);

        let lo = 1.0_f64;
        let hi = f64::from_bits(lo.to_bits() + 1);
        let mut adjacent = Rng::new(0x7c01_2026);
        for _ in 0..10_000 {
            // `lo` is the only representable member of this half-open range.
            assert_eq!(adjacent.uniform_range(lo, hi).to_bits(), lo.to_bits());
        }

        let mut direct = Rng::new(0x63ab_419d);
        let mut ranged = direct.clone();
        for _ in 0..1_000 {
            let expected = 2.0 + 3.0 * direct.uniform();
            assert_eq!(ranged.uniform_range(2.0, 5.0).to_bits(), expected.to_bits());
        }
    }

    #[test]
    fn invalid_uniform_ranges_do_not_advance_state() {
        let invalid = [
            (f64::NAN, 1.0),
            (0.0, f64::INFINITY),
            (1.0, 1.0),
            (2.0, -2.0),
        ];
        for (lo, hi) in invalid {
            let mut rng = Rng::new(0x5be1_09a7);
            let mut unchanged = rng.clone();
            let failed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                rng.uniform_range(lo, hi)
            }));
            assert!(failed.is_err());
            assert_eq!(rng.next_u64(), unchanged.next_u64());
        }
    }

    #[test]
    fn rejection_mapping_has_equal_residue_class_sizes() {
        // Exhaustive eight-bit xorshift analogue of `below`: the source has
        // period 255 and emits each nonzero word once. After translating to
        // zero and rejecting the incomplete suffix, every class is equiprobable.
        for bound in 1u16..=255 {
            let complete_zone = 255 - 255 % bound;
            let mut counts = vec![0usize; bound as usize];
            for word in 1..=255 {
                let zero_based_word = word - 1;
                if zero_based_word < complete_zone {
                    counts[(zero_based_word % bound) as usize] += 1;
                }
            }
            assert!(counts.windows(2).all(|pair| pair[0] == pair[1]));
        }
    }

    #[test]
    #[should_panic(expected = "empty range")]
    fn below_rejects_an_empty_range() {
        Rng::new(1).below(0);
    }

    #[test]
    fn poisson_zero_is_degenerate_and_invalid_rates_do_not_advance_state() {
        let invalid = [
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            -1.0,
            MAX_RELIABLE_POISSON_RATE * 2.0,
        ];
        for lambda in invalid {
            let mut rng = Rng::new(91);
            let mut unchanged = rng.clone();
            let failed =
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| rng.poisson(lambda)));
            assert!(failed.is_err());
            assert_eq!(rng.next_u64(), unchanged.next_u64());
        }

        let mut rng = Rng::new(91);
        let mut unchanged = rng.clone();
        assert_eq!(rng.poisson(0.0), 0);
        assert_eq!(rng.next_u64(), unchanged.next_u64());
    }

    #[test]
    fn poisson_sample_moments_match_the_exact_distribution() {
        const DRAWS: usize = 400_000;
        // Measured maxima on 2026-09-03 were 0.012145 for the mean and
        // 0.0596626896 for the variance; these tolerances are 3.9x wider.
        const MEAN_TOLERANCE: f64 = 0.048;
        const VARIANCE_TOLERANCE: f64 = 0.235;
        let mut rng = Rng::new(0x5a17_2026);
        for &lambda in &[0.25, 3.0, 10.0, 21.0, 100.0] {
            let mut sum = 0.0;
            let mut sum_squares = 0.0;
            for _ in 0..DRAWS {
                let value = rng.poisson(lambda) as f64;
                sum += value;
                sum_squares += value * value;
            }
            let mean = sum / DRAWS as f64;
            let variance = sum_squares / DRAWS as f64 - mean * mean;
            assert!((mean - lambda).abs() <= MEAN_TOLERANCE);
            assert!((variance - lambda).abs() <= VARIANCE_TOLERANCE);
        }
    }

    #[test]
    fn transformed_rejection_retains_poisson_skewness() {
        const DRAWS: usize = 600_000;
        const LAMBDA: f64 = 25.0;
        // Measured error on 2026-09-03 was 0.1000024703; tolerance is 3.9x.
        const THIRD_MOMENT_TOLERANCE: f64 = 0.391;
        let mut rng = Rng::new(0x600d_f00d);
        let mut values = Vec::with_capacity(DRAWS);
        let mut sum = 0.0;
        for _ in 0..DRAWS {
            let value = rng.poisson(LAMBDA) as f64;
            values.push(value);
            sum += value;
        }
        let mean = sum / DRAWS as f64;
        let third_central_moment = values
            .iter()
            .map(|value| (value - mean).powi(3))
            .sum::<f64>()
            / DRAWS as f64;
        assert!((third_central_moment - LAMBDA).abs() <= THIRD_MOMENT_TOLERANCE);
    }
}
