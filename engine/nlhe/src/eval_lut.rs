//! Lookup-table hand evaluator (fast path). `eval.rs` stays the reference oracle and builds
//! these tables.
//!
//! A hand is accumulated card by card in an [`LHand`]:
//! * `key`: sum of 5^rank per card. Base-5 digits are rank counts (each <= 4, so there are no
//!   carries), which makes the key unique per rank multiset.
//! * `suits`: four 4-bit counters starting at 3; a nibble reaches 8 (bit 3 set) exactly when
//!   that suit has 5+ cards.
//! * `mask`: the suit-lane bitmask from `eval.rs`, used only to read a flush suit's ranks.
//!
//! Lookups return a dense **class** 0..7461 (higher = stronger, equal = tie), shared by 5-,
//! 6- and 7-card hands. `class_strength` maps a class back to the oracle's strength value.
//!
//! Tables: an open-addressed key -> class table (2^18 slots of 8 bytes = 2 MiB, ~0.28 load),
//! an 8,192-entry flush table (16 KiB) and 7,462 class strengths. Built from the oracle in
//! well under a second.

use crate::eval::{card_bit, eval, Hand};
use std::sync::OnceLock;

pub const N_CLASSES: usize = 7462;
const SLOTS: usize = 1 << 18;
const SLOT_MASK: usize = SLOTS - 1;
pub const POW5: [u32; 13] = {
    let mut p = [1u32; 13];
    let mut i = 1;
    while i < 13 {
        p[i] = p[i - 1] * 5;
        i += 1;
    }
    p
};
const SUIT_INIT: u32 = 0x3333;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LHand {
    pub key: u32,
    pub suits: u32,
    pub mask: Hand,
}

impl Default for LHand {
    fn default() -> Self {
        LHand::EMPTY
    }
}

impl LHand {
    pub const EMPTY: LHand = LHand { key: 0, suits: SUIT_INIT, mask: 0 };

    #[inline]
    pub fn add(self, c: u8) -> LHand {
        LHand {
            key: self.key + POW5[(c / 4) as usize],
            suits: self.suits + (1 << (4 * (c % 4) as u32)),
            mask: self.mask | card_bit(c),
        }
    }

    pub fn of(cards: &[u8]) -> LHand {
        cards.iter().fold(LHand::EMPTY, |h, &c| h.add(c))
    }
}

pub struct Lut {
    slot_key: Vec<u32>,
    slot_class: Vec<u16>,
    flush: Vec<u16>,
    pub class_strength: Vec<u32>,
}

#[inline]
fn slot_of(key: u32) -> usize {
    (key.wrapping_mul(0x9E37_79B1) >> (32 - 18)) as usize
}

impl Lut {
    fn build() -> Lut {
        // 1. all distinct strengths, from every 5-card hand -> dense classes
        let mut strengths: Vec<u32> = Vec::with_capacity(N_CLASSES);
        {
            let mut seen = std::collections::HashSet::new();
            for a in 0..52u8 {
                for b in a + 1..52 {
                    for c in b + 1..52 {
                        for d in c + 1..52 {
                            for e in d + 1..52 {
                                let v = eval(card_bit(a) | card_bit(b) | card_bit(c) | card_bit(d) | card_bit(e));
                                if seen.insert(v) {
                                    strengths.push(v);
                                }
                            }
                        }
                    }
                }
            }
        }
        strengths.sort_unstable();
        assert_eq!(strengths.len(), N_CLASSES);
        let class_of = |v: u32| strengths.binary_search(&v).expect("strength not in class list") as u16;

        // 2. non-flush rank multisets of size 5..=7, counts <= 4, suits assigned round-robin
        //    (at most 2 cards per suit, so never a flush)
        let mut slot_key = vec![0u32; SLOTS];
        let mut slot_class = vec![0u16; SLOTS];
        let mut counts = [0u8; 13];
        fn rec(
            r: usize,
            left: u8,
            counts: &mut [u8; 13],
            f: &mut dyn FnMut(&[u8; 13]),
        ) {
            if r == 13 {
                f(counts);
                return;
            }
            for c in 0..=left.min(4) {
                counts[r] = c;
                rec(r + 1, left - c, counts, f);
            }
            counts[r] = 0;
        }
        let mut insert = |cnt: &[u8; 13]| {
            let n: u8 = cnt.iter().sum();
            if n < 5 {
                return;
            }
            let mut cards = Vec::with_capacity(7);
            let mut k = 0u8;
            for (rank, &c) in cnt.iter().enumerate() {
                for _ in 0..c {
                    cards.push(rank as u8 * 4 + (k % 4));
                    k += 1;
                }
            }
            let h = LHand::of(&cards);
            let class = class_of(eval(h.mask));
            let mut s = slot_of(h.key);
            while slot_key[s] != 0 {
                assert_ne!(slot_key[s], h.key, "duplicate key");
                s = (s + 1) & SLOT_MASK;
            }
            slot_key[s] = h.key;
            slot_class[s] = class;
        };
        rec(0, 7, &mut counts, &mut insert);

        // 3. flush table: any 13-bit suited-rank mask with >= 5 bits
        let mut flush = vec![0u16; 1 << 13];
        for m in 0u32..(1 << 13) {
            if m.count_ones() >= 5 {
                flush[m as usize] = class_of(eval(m as u64)); // suit 0 lane
            }
        }
        Lut { slot_key, slot_class, flush, class_strength: strengths }
    }

    pub fn get() -> &'static Lut {
        static LUT: OnceLock<Lut> = OnceLock::new();
        LUT.get_or_init(Lut::build)
    }

    /// Dense class (0..7461) of a 5–7 card hand.
    #[inline]
    pub fn class(&self, h: LHand) -> u16 {
        let f = h.suits & 0x8888;
        if f != 0 {
            let suit = f.trailing_zeros() / 4;
            return self.flush[((h.mask >> (16 * suit)) & 0x1fff) as usize];
        }
        let mut s = slot_of(h.key);
        loop {
            let k = self.slot_key[s];
            if k == h.key {
                return self.slot_class[s];
            }
            debug_assert!(k != 0, "rank multiset not in table");
            s = (s + 1) & SLOT_MASK;
        }
    }

    /// Class of a hand known to contain no flush, from its rank key alone.
    #[inline]
    pub fn class_nonflush(&self, key: u32) -> u16 {
        let mut s = slot_of(key);
        loop {
            let k = self.slot_key[s];
            if k == key {
                return self.slot_class[s];
            }
            debug_assert!(k != 0, "rank multiset not in table");
            s = (s + 1) & SLOT_MASK;
        }
    }

    /// Class of a flush given the 13-bit rank mask of the flush suit (>= 5 bits).
    #[inline]
    pub fn class_flush(&self, suited_ranks: u32) -> u16 {
        self.flush[suited_ranks as usize]
    }

    /// Strength in the oracle's encoding (for comparisons with `eval::eval`).
    #[inline]
    pub fn strength(&self, h: LHand) -> u32 {
        self.class_strength[self.class(h) as usize]
    }

    /// Longest probe sequence in the open-addressed table (diagnostic).
    pub fn max_probe(&self) -> usize {
        let mut worst = 0;
        for (s, &k) in self.slot_key.iter().enumerate() {
            if k != 0 {
                let home = slot_of(k);
                worst = worst.max((s + SLOTS - home) & SLOT_MASK);
            }
        }
        worst + 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::hand_of;

    #[test]
    fn every_five_card_hand_matches_oracle() {
        let lut = Lut::get();
        for a in 0..52u8 {
            for b in a + 1..52 {
                for c in b + 1..52 {
                    for d in c + 1..52 {
                        for e in d + 1..52 {
                            let cards = [a, b, c, d, e];
                            assert_eq!(lut.strength(LHand::of(&cards)), eval(hand_of(&cards)), "{cards:?}");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn every_six_card_hand_matches_oracle() {
        let lut = Lut::get();
        let mut n = 0u64;
        for a in 0..52u8 {
            let ha = LHand::EMPTY.add(a);
            for b in a + 1..52 {
                let hb = ha.add(b);
                for c in b + 1..52 {
                    let hc = hb.add(c);
                    for d in c + 1..52 {
                        let hd = hc.add(d);
                        for e in d + 1..52 {
                            let he = hd.add(e);
                            for f in e + 1..52 {
                                let h = he.add(f);
                                assert_eq!(lut.strength(h), eval(h.mask));
                                n += 1;
                            }
                        }
                    }
                }
            }
        }
        assert_eq!(n, 20_358_520);
    }

    /// All 133,784,560 seven-card hands (the "large sample" is the full population).
    #[test]
    fn every_seven_card_hand_matches_oracle() {
        let lut = Lut::get();
        let mut n = 0u64;
        for a in 0..52u8 {
            let ha = LHand::EMPTY.add(a);
            for b in a + 1..52 {
                let hb = ha.add(b);
                for c in b + 1..52 {
                    let hc = hb.add(c);
                    for d in c + 1..52 {
                        let hd = hc.add(d);
                        for e in d + 1..52 {
                            let he = hd.add(e);
                            for f in e + 1..52 {
                                let hf = he.add(f);
                                for g in f + 1..52 {
                                    let h = hf.add(g);
                                    if lut.strength(h) != eval(h.mask) {
                                        panic!("mismatch {:?}", [a, b, c, d, e, f, g]);
                                    }
                                    n += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
        assert_eq!(n, 133_784_560);
    }

    #[test]
    fn classes_are_ordered_and_table_is_sane() {
        let lut = Lut::get();
        assert!(lut.class_strength.windows(2).all(|w| w[0] < w[1]));
        assert!(lut.max_probe() <= 16, "max probe {}", lut.max_probe());
        let filled = lut.slot_key.iter().filter(|&&k| k != 0).count();
        // multisets of 13 ranks, counts <= 4, sizes 5, 6, 7
        assert_eq!(filled, 6_175 + 18_395 + 49_205);
    }
}
