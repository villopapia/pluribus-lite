//! Minimal Fx hasher (as used in rustc) for u64 infoset keys. Not DoS-resistant,
//! which is irrelevant here. Implemented inline to keep the crate dependency-free.

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

const K: u64 = 0x517c_c1b7_2722_0a95;

#[derive(Default, Clone, Copy)]
pub struct FxHasher {
    hash: u64,
}

impl FxHasher {
    #[inline]
    fn add(&mut self, i: u64) {
        self.hash = (self.hash.rotate_left(5) ^ i).wrapping_mul(K);
    }
}

impl Hasher for FxHasher {
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.add(b as u64);
        }
    }
    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.add(i);
    }
    #[inline]
    fn finish(&self) -> u64 {
        self.hash
    }
}

pub type FxBuildHasher = BuildHasherDefault<FxHasher>;
pub type FxHashMap<K, V> = HashMap<K, V, FxBuildHasher>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn map_works() {
        let mut m: FxHashMap<u64, u64> = FxHashMap::default();
        for i in 0..10_000u64 {
            m.insert(i * 31, i);
        }
        for i in 0..10_000u64 {
            assert_eq!(m[&(i * 31)], i);
        }
    }
}
