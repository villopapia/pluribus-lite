//! Deterministic, dependency-free RNG: xoshiro256** seeded through SplitMix64.
//! The full state can be read and restored, so checkpoints resume bit-identically.

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rng {
    s: [u64; 4],
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        let mut z = seed;
        let mut s = [0u64; 4];
        for x in s.iter_mut() {
            z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut r = z;
            r = (r ^ (r >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            r = (r ^ (r >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            *x = r ^ (r >> 31);
        }
        Rng { s }
    }

    pub fn from_state(s: [u64; 4]) -> Self {
        Rng { s }
    }

    pub fn state(&self) -> [u64; 4] {
        self.s
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let result = self.s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// Uniform in [0, 1).
    #[inline]
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Sample an index proportionally to `weights` (need not sum to 1).
    #[inline]
    pub fn sample_index(&mut self, weights: &[f64]) -> usize {
        let total: f64 = weights.iter().sum();
        let r = self.next_f64() * total;
        let mut acc = 0.0;
        let mut last_pos = 0;
        for (i, &w) in weights.iter().enumerate() {
            if w > 0.0 {
                acc += w;
                last_pos = i;
                if r < acc {
                    return i;
                }
            }
        }
        last_pos
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_for_seed() {
        let mut a = Rng::new(42);
        let mut b = Rng::new(42);
        for _ in 0..1000 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
        assert_ne!(Rng::new(1).next_u64(), Rng::new(2).next_u64());
    }

    #[test]
    fn state_roundtrip() {
        let mut a = Rng::new(7);
        for _ in 0..10 {
            a.next_u64();
        }
        let mut b = Rng::from_state(a.state());
        for _ in 0..100 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn f64_range_and_mean() {
        let mut r = Rng::new(3);
        let n = 200_000;
        let mut sum = 0.0;
        for _ in 0..n {
            let x = r.next_f64();
            assert!((0.0..1.0).contains(&x));
            sum += x;
        }
        assert!((sum / n as f64 - 0.5).abs() < 0.005);
    }

    #[test]
    fn sample_index_frequencies() {
        let mut r = Rng::new(11);
        let w = [0.2, 0.0, 0.5, 0.3];
        let mut counts = [0usize; 4];
        let n = 200_000;
        for _ in 0..n {
            counts[r.sample_index(&w)] += 1;
        }
        assert_eq!(counts[1], 0);
        for i in [0, 2, 3] {
            assert!((counts[i] as f64 / n as f64 - w[i]).abs() < 0.01);
        }
    }
}
