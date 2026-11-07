//! Direct (non-isomorphic) indexing of (hole cards, board) for bucket tables.
//!
//! index = combo_index(hole) * C(50, k) + colex(board renumbered to skip the two hole cards)
//!
//! Suit isomorphism is deliberately not used. Lookups are a few integer ops, and the table
//! cost over canonical indexing is small: flop 24.8 MiB vs 1.2, turn 291 MiB vs 53, river
//! 2.62 GiB vs 2.26 (u8 entries).

use crate::equity::{combo_index, COMBOS};

pub const FLOP_ENTRIES: u64 = COMBOS as u64 * 19_600; // C(50,3)
pub const TURN_ENTRIES: u64 = COMBOS as u64 * 230_300; // C(50,4)
pub const RIVER_ENTRIES: u64 = COMBOS as u64 * 2_118_760; // C(50,5)

/// Table entries for street 1..=3 (flop, turn, river).
pub const ENTRIES: [u64; 4] = [169, FLOP_ENTRIES, TURN_ENTRIES, RIVER_ENTRIES];

const fn binom_table() -> [[u32; 6]; 52] {
    let mut t = [[0u32; 6]; 52];
    let mut n = 0;
    while n < 52 {
        t[n][0] = 1;
        let mut k = 1;
        while k < 6 {
            t[n][k] = if n == 0 { 0 } else { t[n - 1][k - 1] + t[n - 1][k] };
            k += 1;
        }
        n += 1;
    }
    t
}
static BINOM: [[u32; 6]; 52] = binom_table();

#[inline]
pub fn binom(n: usize, k: usize) -> u64 {
    BINOM[n][k] as u64
}

/// Colex rank of a set of distinct cards in 0..n (any order), using C(c, i + 1).
#[inline]
pub fn colex(cards: &mut [u8]) -> u64 {
    cards.sort_unstable();
    cards.iter().enumerate().map(|(i, &c)| BINOM[c as usize][i + 1] as u64).sum()
}

/// Index of (hole, board) where board has 3, 4 or 5 cards disjoint from hole.
#[inline]
pub fn hand_index(hole: (u8, u8), board: &[u8]) -> u64 {
    let (a, b) = if hole.0 < hole.1 { hole } else { (hole.1, hole.0) };
    let mut buf = [0u8; 5];
    for (i, &c) in board.iter().enumerate() {
        debug_assert!(c != a && c != b);
        buf[i] = c - (c > a) as u8 - (c > b) as u8;
    }
    let k = board.len();
    combo_index(a, b) as u64 * binom(50, k) + colex(&mut buf[..k])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes() {
        assert_eq!(binom(50, 3), 19_600);
        assert_eq!(binom(50, 4), 230_300);
        assert_eq!(binom(50, 5), 2_118_760);
        assert_eq!(FLOP_ENTRIES, 25_989_600);
        assert_eq!(TURN_ENTRIES, 305_377_800);
        assert_eq!(RIVER_ENTRIES, 2_809_475_760);
    }

    /// Every (hole, flop) maps to a distinct index in range: a bijection onto 0..FLOP_ENTRIES.
    #[test]
    fn flop_index_is_a_bijection() {
        let mut seen = vec![0u64; (FLOP_ENTRIES as usize + 63) / 64];
        let mut n = 0u64;
        for a in 0..52u8 {
            for b in a + 1..52 {
                for x in 0..52u8 {
                    for y in x + 1..52 {
                        for z in y + 1..52 {
                            if [x, y, z].iter().any(|&c| c == a || c == b) {
                                continue;
                            }
                            let i = hand_index((a, b), &[z, x, y]);
                            assert!(i < FLOP_ENTRIES);
                            let (w, bit) = ((i / 64) as usize, i % 64);
                            assert_eq!(seen[w] >> bit & 1, 0, "collision");
                            seen[w] |= 1 << bit;
                            n += 1;
                        }
                    }
                }
            }
        }
        assert_eq!(n, FLOP_ENTRIES);
    }

    #[test]
    fn turn_and_river_indices_in_range_and_order_free() {
        let mut rng = cfr::rng::Rng::new(4);
        for _ in 0..100_000 {
            let mut deck: Vec<u8> = (0..52).collect();
            for i in 0..7 {
                let j = i + (rng.next_u64() % (52 - i as u64)) as usize;
                deck.swap(i, j);
            }
            let hole = (deck[0], deck[1]);
            let t = hand_index(hole, &deck[2..6]);
            assert!(t < TURN_ENTRIES);
            assert_eq!(t, hand_index((hole.1, hole.0), &[deck[5], deck[3], deck[2], deck[4]]));
            let r = hand_index(hole, &deck[2..7]);
            assert!(r < RIVER_ENTRIES);
        }
        // extremes
        assert_eq!(hand_index((0, 1), &[2, 3, 4, 5, 6]), 0);
        assert_eq!(hand_index((50, 51), &[45, 46, 47, 48, 49]), RIVER_ENTRIES - 1);
    }
}
