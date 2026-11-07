//! Table-free 5–7 card hand evaluator.
//!
//! Cards are 0..52 with rank = c / 4 (0 = deuce .. 12 = ace) and suit = c % 4. A hand is a
//! `Hand` bitmask with one 16-bit lane per suit (bit = suit * 16 + rank), so per-suit and
//! per-rank masks are shifts and ANDs. `eval` returns a strength where a larger value means
//! a better hand and equal values are exact ties.
//!
//! Strength layout: `category << 26 | a << 13 | b`, with a and b 13-bit rank masks. Comparing
//! masks with equal popcount as integers is the same as comparing their sorted ranks, so
//! kickers compare correctly.

pub type Hand = u64;

pub const HIGH_CARD: u32 = 0;
pub const PAIR: u32 = 1;
pub const TWO_PAIR: u32 = 2;
pub const TRIPS: u32 = 3;
pub const STRAIGHT: u32 = 4;
pub const FLUSH: u32 = 5;
pub const FULL_HOUSE: u32 = 6;
pub const QUADS: u32 = 7;
pub const STRAIGHT_FLUSH: u32 = 8;

const RANKS: u64 = 0x1fff;

#[inline]
pub fn card_bit(c: u8) -> Hand {
    1u64 << ((c % 4) as u32 * 16 + (c / 4) as u32)
}

pub fn hand_of(cards: &[u8]) -> Hand {
    cards.iter().fold(0, |h, &c| h | card_bit(c))
}

#[inline]
pub fn category(strength: u32) -> u32 {
    strength >> 26
}

/// Keep the `k` highest set bits.
#[inline]
fn top(mut m: u32, k: u32) -> u32 {
    while m.count_ones() > k {
        m &= m - 1;
    }
    m
}

#[inline]
fn high_bit(m: u32) -> u32 {
    1 << (31 - m.leading_zeros())
}

/// Highest straight in a 13-bit rank mask, as the rank of its top card (wheel = 3), if any.
#[inline]
fn straight_high(r: u32) -> Option<u32> {
    let m = (r << 1) | ((r >> 12) & 1); // bit 0 = ace-low
    let s = m & (m >> 1) & (m >> 2) & (m >> 3) & (m >> 4);
    if s == 0 {
        None
    } else {
        Some(31 - s.leading_zeros() + 3)
    }
}

#[inline]
fn make(cat: u32, a: u32, b: u32) -> u32 {
    (cat << 26) | (a << 13) | b
}

/// Strength of the best 5-card hand within 5–7 cards.
#[inline]
pub fn eval(h: Hand) -> u32 {
    let s0 = (h & RANKS) as u32;
    let s1 = ((h >> 16) & RANKS) as u32;
    let s2 = ((h >> 32) & RANKS) as u32;
    let s3 = ((h >> 48) & RANKS) as u32;

    let mut flush = 0u32;
    for s in [s0, s1, s2, s3] {
        if s.count_ones() >= 5 {
            flush = s;
        }
    }
    if flush != 0 {
        if let Some(hi) = straight_high(flush) {
            return make(STRAIGHT_FLUSH, 0, hi);
        }
    }

    let ranks = s0 | s1 | s2 | s3;
    let quads = s0 & s1 & s2 & s3;
    let three = (s0 & s1 & s2) | (s0 & s1 & s3) | (s0 & s2 & s3) | (s1 & s2 & s3);
    let two = (s0 & s1) | (s0 & s2) | (s0 & s3) | (s1 & s2) | (s1 & s3) | (s2 & s3);

    if quads != 0 {
        let q = high_bit(quads);
        return make(QUADS, q, top(ranks & !q, 1));
    }
    let trips = three; // no quads here, so these are exactly three-of-a-kind ranks
    let pairs = two & !three;
    if trips != 0 {
        let t = high_bit(trips);
        let rest = (trips & !t) | pairs;
        if rest != 0 {
            return make(FULL_HOUSE, t, high_bit(rest));
        }
    }
    if flush != 0 {
        return make(FLUSH, 0, top(flush, 5));
    }
    if let Some(hi) = straight_high(ranks) {
        return make(STRAIGHT, 0, hi);
    }
    if trips != 0 {
        let t = high_bit(trips);
        return make(TRIPS, t, top(ranks & !t, 2));
    }
    if pairs.count_ones() >= 2 {
        let tp = top(pairs, 2);
        return make(TWO_PAIR, tp, top(ranks & !tp, 1));
    }
    if pairs != 0 {
        return make(PAIR, pairs, top(ranks & !pairs, 3));
    }
    make(HIGH_CARD, 0, top(ranks, 5))
}

/// Straightforward 5-card evaluator (sort and count), used only as a test oracle.
pub fn eval5_reference(cards: [u8; 5]) -> u32 {
    let mut r: Vec<u32> = cards.iter().map(|&c| (c / 4) as u32).collect();
    r.sort_unstable_by(|a, b| b.cmp(a));
    let is_flush = cards.iter().all(|&c| c % 4 == cards[0] % 4);
    let mut uniq = r.clone();
    uniq.dedup();
    let straight = if uniq.len() == 5 && r[0] - r[4] == 4 {
        Some(r[0])
    } else if uniq == vec![12, 3, 2, 1, 0] {
        Some(3)
    } else {
        None
    };
    let mask = |v: &[u32]| v.iter().fold(0u32, |m, &x| m | (1 << x));
    // group by count desc, then rank desc
    let mut groups: Vec<(u32, u32)> = uniq.iter().map(|&x| (r.iter().filter(|&&y| y == x).count() as u32, x)).collect();
    groups.sort_unstable_by(|a, b| b.cmp(a));
    let counts: Vec<u32> = groups.iter().map(|g| g.0).collect();
    let g = |i: usize| groups[i].1;
    match (straight, is_flush, counts.as_slice()) {
        (Some(hi), true, _) => make(STRAIGHT_FLUSH, 0, hi),
        (_, _, [4, 1]) => make(QUADS, 1 << g(0), 1 << g(1)),
        (_, _, [3, 2]) => make(FULL_HOUSE, 1 << g(0), 1 << g(1)),
        (_, true, _) => make(FLUSH, 0, mask(&r)),
        (Some(hi), false, _) => make(STRAIGHT, 0, hi),
        (_, _, [3, 1, 1]) => make(TRIPS, 1 << g(0), mask(&[g(1), g(2)])),
        (_, _, [2, 2, 1]) => make(TWO_PAIR, mask(&[g(0), g(1)]), 1 << g(2)),
        (_, _, [2, 1, 1, 1]) => make(PAIR, 1 << g(0), mask(&[g(1), g(2), g(3)])),
        _ => make(HIGH_CARD, 0, mask(&r)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    fn cards(s: &str) -> Vec<u8> {
        s.split_whitespace()
            .map(|t| {
                let b = t.as_bytes();
                let r = b"23456789TJQKA".iter().position(|&x| x == b[0]).unwrap() as u8;
                let su = b"cdhs".iter().position(|&x| x == b[1]).unwrap() as u8;
                r * 4 + su
            })
            .collect()
    }

    fn e(s: &str) -> u32 {
        eval(hand_of(&cards(s)))
    }

    #[test]
    fn categories_of_known_hands() {
        assert_eq!(category(e("As Ks Qs Js Ts 2d 3c")), STRAIGHT_FLUSH);
        assert_eq!(category(e("Ah 2h 3h 4h 5h Kd Kc")), STRAIGHT_FLUSH);
        assert_eq!(category(e("9c 9d 9h 9s 2c 3d 4h")), QUADS);
        assert_eq!(category(e("9c 9d 9h 2s 2c 3d 4h")), FULL_HOUSE);
        assert_eq!(category(e("9c 9d 9h 2s 2c 2d 4h")), FULL_HOUSE); // two trips
        assert_eq!(category(e("2c 7c 9c Jc Kc Ad Ah")), FLUSH);
        assert_eq!(category(e("Ad 2c 3h 4s 5d Kc Kh")), STRAIGHT);
        assert_eq!(category(e("7d 7c 7h 2s 4d Kc Jh")), TRIPS);
        assert_eq!(category(e("7d 7c 2h 2s 4d 4c Jh")), TWO_PAIR); // three pairs
        assert_eq!(category(e("7d 7c 2h 3s 9d Kc Jh")), PAIR);
        assert_eq!(category(e("7d 8c 2h 3s 9d Kc Jh")), HIGH_CARD);
    }

    #[test]
    fn orderings_and_ties() {
        assert!(e("As Ad Kc Kd 2h 3c 4d") > e("As Ad Qc Qd Jh Tc 9d")); // higher second pair
        assert!(e("Ad 2c 3h 4s 5d 9c 9h") < e("2d 3c 4h 5s 6d 9c 9h")); // wheel < six-high
        assert_eq!(e("Ah Kd Qc Js 9h 3c 2d"), e("Ad Kc Qh Jd 9s 4c 2h")); // kicker beyond 5 cards ignored
        assert!(e("7d 7c 2h 2s 4d 4c Jh") > e("7d 7c 4d 4c Th 2s 3c")); // best kicker counts
        assert_eq!(e("9c 9d 9h 2s 2c 3d 3h"), e("9s 9d 9h 3s 3c 2d 4h")); // 999-33 both
        assert!(e("9c 9d 9h 3s 3c 2d 4h") > e("9c 9d 9h 2s 2c 3d 4h"));
    }

    /// Category counts over all 2,598,960 five-card hands, and 7,462 distinct strengths.
    #[test]
    fn exhaustive_five_card() {
        let mut counts = [0u64; 9];
        let mut distinct = HashSet::new();
        for a in 0..52u8 {
            for b in a + 1..52 {
                for c in b + 1..52 {
                    for d in c + 1..52 {
                        for f in d + 1..52 {
                            let v = eval(hand_of(&[a, b, c, d, f]));
                            assert_eq!(v, eval5_reference([a, b, c, d, f]));
                            counts[category(v) as usize] += 1;
                            distinct.insert(v);
                        }
                    }
                }
            }
        }
        assert_eq!(counts, [1_302_540, 1_098_240, 123_552, 54_912, 10_200, 5_108, 3_744, 624, 40]);
        assert_eq!(distinct.len(), 7_462);
    }

    /// Category counts over all 133,784,560 seven-card hands, and 4,824 distinct strengths.
    #[test]
    fn exhaustive_seven_card() {
        let mut counts = [0u64; 9];
        let mut distinct = HashSet::new();
        let bits: Vec<Hand> = (0..52u8).map(card_bit).collect();
        for a in 0..52 {
            for b in a + 1..52 {
                let hb = bits[a] | bits[b];
                for c in b + 1..52 {
                    let hc = hb | bits[c];
                    for d in c + 1..52 {
                        let hd = hc | bits[d];
                        for f in d + 1..52 {
                            let hf = hd | bits[f];
                            for g in f + 1..52 {
                                let hg = hf | bits[g];
                                for i in g + 1..52 {
                                    let v = eval(hg | bits[i]);
                                    counts[category(v) as usize] += 1;
                                    if distinct.len() < 5_000 {
                                        distinct.insert(v);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        assert_eq!(
            counts,
            [23_294_460, 58_627_800, 31_433_400, 6_461_620, 6_180_020, 4_047_644, 3_473_184, 224_848, 41_584]
        );
        assert_eq!(distinct.len(), 4_824);
    }

    /// 7-card eval equals the best of its 21 five-card subsets (reference oracle), sampled.
    #[test]
    fn seven_card_matches_best_five_subset() {
        let mut rng = cfr::rng::Rng::new(9);
        for _ in 0..50_000 {
            let mut deck: Vec<u8> = (0..52).collect();
            for i in 0..7 {
                let j = i + (rng.next_u64() % (52 - i as u64)) as usize;
                deck.swap(i, j);
            }
            let c = &deck[..7];
            let mut best = 0;
            for skip1 in 0..7 {
                for skip2 in skip1 + 1..7 {
                    let five: Vec<u8> = (0..7).filter(|&k| k != skip1 && k != skip2).map(|k| c[k]).collect();
                    best = best.max(eval5_reference([five[0], five[1], five[2], five[3], five[4]]));
                }
            }
            assert_eq!(eval(hand_of(c)), best, "{c:?}");
        }
    }
}
