//! Tiny deterministic PRNG (xorshift64*). Pure Rust, no `rand` crate.

/// Reproducible xorshift64* generator used by stochastic tree splits.
#[derive(Clone, Debug)]
pub struct Rng {
    state: u64,
}

impl Rng {
    /// Seed; 0 is replaced by a fixed odd constant so the generator is never stuck.
    pub fn new(seed: u64) -> Self {
        Self { state: seed | 1 }
    }

    /// Next `u64`.
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }

    /// Uniform in `[0, 1)`.
    pub fn uniform(&mut self) -> f64 {
        let u = self.next_u64() >> 11;
        (u as f64) * (1.0 / ((1u64 << 53) as f64))
    }

    /// Uniform in `[lo, hi)`.
    pub fn uniform_range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.uniform()
    }

    /// Integer in `0..n`.
    pub fn below(&mut self, n: usize) -> usize {
        if n == 0 {
            return 0;
        }
        (self.next_u64() as usize) % n
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

    /// Bootstrap sample of length `n` from `0..n`.
    pub fn bootstrap_idx(&mut self, n: usize) -> Vec<usize> {
        (0..n).map(|_| self.below(n)).collect()
    }

    /// Weighted bootstrap of length `w.len()`.
    pub fn weighted_bootstrap(&mut self, w: &[f64]) -> Vec<usize> {
        let n = w.len();
        let mut cdf = vec![0.0; n];
        let mut acc = 0.0;
        for i in 0..n {
            acc += w[i].max(0.0);
            cdf[i] = acc;
        }
        if acc <= 0.0 || n == 0 {
            return (0..n).collect();
        }
        (0..n)
            .map(|_| {
                let u = self.uniform() * acc;
                cdf.iter().position(|&c| c >= u).unwrap_or(n - 1)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic() {
        let mut a = Rng::new(7);
        let mut b = Rng::new(7);
        assert_eq!(a.next_u64(), b.next_u64());
    }
}
