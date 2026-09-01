//! Core generation interfaces.

/// A source of uniformly distributed bits.
///
/// Implementors only need [`Rng::next_u64`]; the remaining methods have
/// default bodies built from that stream.
pub trait Rng {
    /// Next 64 uniform bits.
    fn next_u64(&mut self) -> u64;

    /// Next 32 uniform bits.
    fn next_u32(&mut self) -> u32 {
        (self.next_u64() >> 32) as u32
    }

    /// Fill `dest` with random bytes (little-endian words).
    fn fill_bytes(&mut self, dest: &mut [u8]) {
        let mut i = 0;
        while i + 8 <= dest.len() {
            dest[i..i + 8].copy_from_slice(&self.next_u64().to_le_bytes());
            i += 8;
        }
        if i < dest.len() {
            let n = dest.len() - i;
            let rest = self.next_u64().to_le_bytes();
            dest[i..].copy_from_slice(&rest[..n]);
        }
    }

    /// Uniform sample in `[0, 1)`.
    fn next_f64(&mut self) -> f64 {
        // 53-bit mantissa, matching IEEE-754 unit interval convention.
        (self.next_u64() >> 11) as f64 * (1.0 / ((1u64 << 53) as f64))
    }

    /// Draw a value from `dist`.
    fn sample<D: Distribution>(&mut self, dist: D) -> D::Value {
        dist.sample(self)
    }
}

impl<T: Rng + ?Sized> Rng for &mut T {
    fn next_u64(&mut self) -> u64 {
        (**self).next_u64()
    }
    fn next_u32(&mut self) -> u32 {
        (**self).next_u32()
    }
    fn fill_bytes(&mut self, dest: &mut [u8]) {
        (**self).fill_bytes(dest);
    }
    fn next_f64(&mut self) -> f64 {
        (**self).next_f64()
    }
}

/// Construct an RNG from a 64-bit seed.
pub trait SeedableRng: Sized {
    /// Deterministic construction from a `u64` seed.
    fn seed_from_u64(seed: u64) -> Self;
}

/// A sampling law that can draw values from any [`Rng`].
pub trait Distribution {
    /// Sampled type.
    type Value;
    /// Draw one value.
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Self::Value;
}
