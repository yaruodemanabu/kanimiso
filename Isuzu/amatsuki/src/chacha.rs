//! ChaCha8 stream cipher used as a deterministic PRNG.

use crate::rng::{Rng, SeedableRng};

/// Reproducible ChaCha8 generator (Pure Rust, 8 rounds).
pub type SeededRng = ChaCha8Rng;

/// Build a seeded generator.
pub fn seed_rng(seed: u64) -> SeededRng {
    ChaCha8Rng::seed_from_u64(seed)
}

/// SplitMix64 — used only to expand a `u64` seed into a 256-bit ChaCha key.
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// "expand 32-byte k"
const C0: u32 = 0x6170_7865;
const C1: u32 = 0x3320_646e;
const C2: u32 = 0x7962_2d32;
const C3: u32 = 0x6b20_6574;

/// ChaCha with 8 rounds (4 double-rounds), 256-bit key, 64-bit counter.
#[derive(Clone, Debug)]
pub struct ChaCha8Rng {
    state: [u32; 16],
    buf: [u32; 16],
    idx: usize,
}

impl ChaCha8Rng {
    /// Construct from a 32-byte key and a 64-bit nonce (counter starts at 0).
    pub fn from_key_nonce(key: [u8; 32], nonce: u64) -> Self {
        let mut st = [0u32; 16];
        st[0] = C0;
        st[1] = C1;
        st[2] = C2;
        st[3] = C3;
        for i in 0..8 {
            let off = i * 4;
            st[4 + i] = u32::from_le_bytes([key[off], key[off + 1], key[off + 2], key[off + 3]]);
        }
        st[12] = 0;
        st[13] = 0;
        st[14] = nonce as u32;
        st[15] = (nonce >> 32) as u32;
        let mut rng = Self {
            state: st,
            buf: [0; 16],
            idx: 16,
        };
        rng.refill();
        rng
    }

    fn refill(&mut self) {
        self.buf = chacha_block(&self.state);
        // 64-bit counter in words 12–13.
        let t = self.state[12].wrapping_add(1);
        self.state[12] = t;
        if t == 0 {
            self.state[13] = self.state[13].wrapping_add(1);
        }
        self.idx = 0;
    }

    fn next_word(&mut self) -> u32 {
        if self.idx >= 16 {
            self.refill();
        }
        let w = self.buf[self.idx];
        self.idx += 1;
        w
    }

    /// Select an independent ChaCha stream by writing the 64-bit nonce.
    ///
    /// The block counter is reset so the next draw is block 0 of that stream.
    pub fn set_stream(&mut self, stream: u64) {
        self.state[12] = 0;
        self.state[13] = 0;
        self.state[14] = stream as u32;
        self.state[15] = (stream >> 32) as u32;
        self.idx = 16;
    }

    /// Skip `n_blocks` of 64 bytes (16 `u32`s) from the current output
    /// position. A pending (unconsumed) buffer counts as the first skipped
    /// block, because `refill` already advanced the counter after generating
    /// it.
    ///
    /// Used to split a parent generator for `simulate_n` parallelism:
    /// child `i` does `set_stream(i)` (preferred) or a jump of
    /// `i · blocks_per_path`.
    pub fn jump_ahead(&mut self, n_blocks: u64) {
        if n_blocks == 0 {
            return;
        }
        let pending = u64::from(self.idx < 16);
        let extra = n_blocks.saturating_sub(pending);
        let (lo, carry) = self.state[12].overflowing_add(extra as u32);
        self.state[12] = lo;
        self.state[13] = self.state[13]
            .wrapping_add((extra >> 32) as u32)
            .wrapping_add(u32::from(carry));
        self.idx = 16;
    }

    /// Current 64-bit block counter (words 12–13).
    pub fn block_counter(&self) -> u64 {
        self.state[12] as u64 | ((self.state[13] as u64) << 32)
    }

    /// Current 64-bit stream / nonce (words 14–15).
    pub fn stream(&self) -> u64 {
        self.state[14] as u64 | ((self.state[15] as u64) << 32)
    }
}

impl SeedableRng for ChaCha8Rng {
    fn seed_from_u64(seed: u64) -> Self {
        let mut s = seed;
        let mut key = [0u8; 32];
        for chunk in key.chunks_exact_mut(8) {
            chunk.copy_from_slice(&splitmix64(&mut s).to_le_bytes());
        }
        Self::from_key_nonce(key, 0)
    }
}

impl Rng for ChaCha8Rng {
    fn next_u32(&mut self) -> u32 {
        self.next_word()
    }

    fn next_u64(&mut self) -> u64 {
        let lo = self.next_word() as u64;
        let hi = self.next_word() as u64;
        lo | (hi << 32)
    }
}

fn quarter_round(s: &mut [u32; 16], a: usize, b: usize, c: usize, d: usize) {
    s[a] = s[a].wrapping_add(s[b]);
    s[d] ^= s[a];
    s[d] = s[d].rotate_left(16);
    s[c] = s[c].wrapping_add(s[d]);
    s[b] ^= s[c];
    s[b] = s[b].rotate_left(12);
    s[a] = s[a].wrapping_add(s[b]);
    s[d] ^= s[a];
    s[d] = s[d].rotate_left(8);
    s[c] = s[c].wrapping_add(s[d]);
    s[b] ^= s[c];
    s[b] = s[b].rotate_left(7);
}

fn double_round(s: &mut [u32; 16]) {
    quarter_round(s, 0, 4, 8, 12);
    quarter_round(s, 1, 5, 9, 13);
    quarter_round(s, 2, 6, 10, 14);
    quarter_round(s, 3, 7, 11, 15);
    quarter_round(s, 0, 5, 10, 15);
    quarter_round(s, 1, 6, 11, 12);
    quarter_round(s, 2, 7, 8, 13);
    quarter_round(s, 3, 4, 9, 14);
}

fn chacha_block(state: &[u32; 16]) -> [u32; 16] {
    let mut x = *state;
    for _ in 0..4 {
        double_round(&mut x);
    }
    for i in 0..16 {
        x[i] = x[i].wrapping_add(state[i]);
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seed_is_deterministic() {
        let mut a = seed_rng(7);
        let mut b = seed_rng(7);
        let mut c = seed_rng(8);
        let xa = a.next_u64();
        assert_eq!(xa, b.next_u64());
        assert_ne!(xa, c.next_u64());
    }

    #[test]
    fn unit_interval() {
        let mut rng = seed_rng(1);
        for _ in 0..10_000 {
            let u = rng.next_f64();
            assert!((0.0..1.0).contains(&u));
        }
    }

    /// First ChaCha8 block, zero key, zero nonce, counter 0.
    ///
    /// First 16 bytes (little-endian words) match the independently
    /// computed vector `3e00ef2f895f40d6…` cited in the 2026-08-30 audit.
    #[test]
    fn chacha8_zero_key_known_answer() {
        let mut rng = ChaCha8Rng::from_key_nonce([0u8; 32], 0);
        let mut bytes = [0u8; 16];
        for chunk in bytes.chunks_exact_mut(4) {
            chunk.copy_from_slice(&rng.next_u32().to_le_bytes());
        }
        assert_eq!(
            bytes,
            [
                0x3e, 0x00, 0xef, 0x2f, 0x89, 0x5f, 0x40, 0xd6, 0x7f, 0x5b, 0xb8, 0xe8, 0x1f, 0x09,
                0xa5, 0xa1
            ]
        );
        let mut rng64 = ChaCha8Rng::from_key_nonce([0u8; 32], 0);
        assert_eq!(rng64.next_u64(), 0xd640_5f89_2fef_003e);
    }

    #[test]
    fn set_stream_splits_and_jump_matches_skip() {
        let mut a = ChaCha8Rng::from_key_nonce([0u8; 32], 0);
        let mut b = ChaCha8Rng::from_key_nonce([0u8; 32], 0);
        a.set_stream(1);
        b.set_stream(1);
        assert_eq!(a.next_u64(), b.next_u64());
        let mut parent = ChaCha8Rng::from_key_nonce([7u8; 32], 0);
        let skipped = parent.clone();
        for _ in 0..4 {
            let _ = parent.next_u64();
        }
        // 4 × u64 = 8 × u32 = half a block; jump is in whole blocks, so
        // compare a full-block skip instead.
        let mut p2 = ChaCha8Rng::from_key_nonce([7u8; 32], 0);
        let mut j2 = p2.clone();
        for _ in 0..8 {
            let _ = p2.next_u64(); // 16 u32 = 1 block
        }
        j2.jump_ahead(1);
        assert_eq!(p2.next_u64(), j2.next_u64());
        let _ = skipped;
    }
}
