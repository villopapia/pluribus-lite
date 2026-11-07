//! River equity against one uniformly random opponent hand (ties count half, card removal
//! applied: the opponent cannot hold the hero's or the board's cards).
//!
//! Two ways to compute it:
//! * [`river_equity_single`]: one hero hand, enumerating the 990 opponent hands (990 evals).
//! * [`RiverBatch::compute`]: all 1,081 hero hands on one board at once. 1,081 evals and a
//!   sort, then each hand's win/tie counts come from per-card prefix counts in O(1).
//!   This is what MCCFR needs per sampled board: one call covers all six players.

use crate::eval::{card_bit, eval, Hand};
use crate::eval_lut::{LHand, Lut, POW5};

/// Number of 2-card combos from 52 cards.
pub const COMBOS: usize = 1326;

/// combo index -> (low card, high card), ordered by (a, b) with a < b.
pub fn combo_cards() -> Vec<(u8, u8)> {
    let mut v = Vec::with_capacity(COMBOS);
    for a in 0..52u8 {
        for b in a + 1..52 {
            v.push((a, b));
        }
    }
    v
}

/// Index of the combo {a, b} in `combo_cards()` order.
#[inline]
pub fn combo_index(a: u8, b: u8) -> usize {
    let (a, b) = if a < b { (a as usize, b as usize) } else { (b as usize, a as usize) };
    // combos before row a: sum_{i<a} (51 - i)
    a * (103 - a) / 2 + (b - a - 1)
}

pub fn board_mask(board: &[u8]) -> Hand {
    board.iter().fold(0, |h, &c| h | card_bit(c))
}

/// Equity of `hole` on a 5-card `board` against one random opponent hand.
pub fn river_equity_single(board: &[u8; 5], hole: [u8; 2]) -> f64 {
    let bm = board_mask(board);
    let hero = eval(bm | card_bit(hole[0]) | card_bit(hole[1]));
    let dead = bm | card_bit(hole[0]) | card_bit(hole[1]);
    let (mut win, mut tie, mut n) = (0u32, 0u32, 0u32);
    for a in 0..52u8 {
        let ba = card_bit(a);
        if dead & ba != 0 {
            continue;
        }
        for b in a + 1..52 {
            let bb = card_bit(b);
            if dead & bb != 0 {
                continue;
            }
            let v = eval(bm | ba | bb);
            n += 1;
            if hero > v {
                win += 1;
            } else if hero == v {
                tie += 1;
            }
        }
    }
    debug_assert_eq!(n, 990);
    (win as f64 + 0.5 * tie as f64) / n as f64
}

/// Reusable scratch space for whole-board river equities.
pub struct RiverBatch {
    combos: Vec<(u8, u8)>,
    /// oracle path only
    keys: Vec<u64>,
    /// (class << 11 | combo index), and a radix-sort buffer
    k32: Vec<u32>,
    tmp: Vec<u32>,
    /// equity per combo index; NaN for combos that use a board card
    pub equity: Vec<f32>,
}

impl Default for RiverBatch {
    fn default() -> Self {
        Self::new()
    }
}

/// Stable LSD radix sort of `v` by bits [shift, shift + bits) using `tmp`.
#[inline]
fn radix_pass(v: &mut Vec<u32>, tmp: &mut Vec<u32>, shift: u32, bits: u32) {
    let nb = 1usize << bits;
    let mask = (nb - 1) as u32;
    let mut count = [0u32; 128];
    for &x in v.iter() {
        count[((x >> shift) & mask) as usize] += 1;
    }
    let mut sum = 0;
    for c in count.iter_mut().take(nb) {
        let t = *c;
        *c = sum;
        sum += t;
    }
    tmp.resize(v.len(), 0);
    for &x in v.iter() {
        let b = ((x >> shift) & mask) as usize;
        tmp[count[b] as usize] = x;
        count[b] += 1;
    }
    std::mem::swap(v, tmp);
}

impl RiverBatch {
    pub fn new() -> Self {
        Lut::get(); // build tables up front, not inside the first timed call
        RiverBatch {
            combos: combo_cards(),
            keys: Vec::with_capacity(1081),
            k32: Vec::with_capacity(1081),
            tmp: Vec::with_capacity(1081),
            equity: vec![f32::NAN; COMBOS],
        }
    }

    /// Fill `self.equity` for every combo on this board (1,081 valid combos).
    ///
    /// * Non-flush hands: on a fixed board the class depends only on the two hole ranks, so
    ///   91 rank-pair lookups replace 1,081 hand lookups.
    /// * Flush hands: only possible if the board has 3+ cards of one suit; those use the
    ///   8,192-entry flush table.
    /// * Ordering: two-pass LSD radix sort of (class << 11 | index), then the same
    ///   card-removal counting as the oracle path.
    pub fn compute(&mut self, board: &[u8; 5]) {
        let lut = Lut::get();
        let bm = board_mask(board);
        let bh = LHand::of(board);

        // rank-pair classes (non-flush); impossible pairs (rank count > 4) are never used
        let mut rank_cnt = [0u8; 13];
        for &c in board {
            rank_cnt[(c / 4) as usize] += 1;
        }
        let mut pair_class = [[0u16; 13]; 13];
        for rx in 0..13 {
            for ry in rx..13 {
                let need_x = rank_cnt[rx] + 1 + (rx == ry) as u8;
                let need_y = rank_cnt[ry] + 1;
                if need_x > 4 || need_y > 4 {
                    continue;
                }
                let c = lut.class_nonflush(bh.key + POW5[rx] + POW5[ry]);
                pair_class[rx][ry] = c;
                pair_class[ry][rx] = c;
            }
        }
        // flush suit candidate: a suit with >= 3 board cards (at most one on a 5-card board)
        let mut flush_suit = 4u8;
        let mut suit_cnt = [0u8; 4];
        for &c in board {
            suit_cnt[(c % 4) as usize] += 1;
        }
        for s in 0..4 {
            if suit_cnt[s] >= 3 {
                flush_suit = s as u8;
            }
        }
        let board_suit_ranks =
            if flush_suit < 4 { ((bm >> (16 * flush_suit as u32)) & 0x1fff) as u32 } else { 0 };

        self.k32.clear();
        for (i, &(a, b)) in self.combos.iter().enumerate() {
            if bm & (card_bit(a) | card_bit(b)) != 0 {
                self.equity[i] = f32::NAN;
                continue;
            }
            let mut cls = pair_class[(a / 4) as usize][(b / 4) as usize];
            if flush_suit < 4 {
                let n_suited = suit_cnt[flush_suit as usize] + (a % 4 == flush_suit) as u8 + (b % 4 == flush_suit) as u8;
                if n_suited >= 5 {
                    let mut m = board_suit_ranks;
                    if a % 4 == flush_suit {
                        m |= 1 << (a / 4);
                    }
                    if b % 4 == flush_suit {
                        m |= 1 << (b / 4);
                    }
                    cls = lut.class_flush(m);
                }
            }
            self.k32.push(((cls as u32) << 11) | i as u32);
        }
        debug_assert_eq!(self.k32.len(), 1081);
        radix_pass(&mut self.k32, &mut self.tmp, 11, 7);
        radix_pass(&mut self.k32, &mut self.tmp, 18, 6);

        let n = self.k32.len();
        let total_opp = 990.0f32;
        let mut less_total = 0u32;
        let mut less_card = [0u32; 52];
        // per-group card counts; only touched entries are updated and reset
        let mut grp_card = [0u32; 52];
        let mut g = 0;
        while g < n {
            let cls = self.k32[g] >> 11;
            let mut end = g;
            while end < n && self.k32[end] >> 11 == cls {
                let (a, b) = self.combos[(self.k32[end] & 0x7ff) as usize];
                grp_card[a as usize] += 1;
                grp_card[b as usize] += 1;
                end += 1;
            }
            let grp = (end - g) as u32;
            for k in g..end {
                let idx = (self.k32[k] & 0x7ff) as usize;
                let (a, b) = self.combos[idx];
                let (a, b) = (a as usize, b as usize);
                let win = less_total - less_card[a] - less_card[b];
                let tie = grp + 1 - grp_card[a] - grp_card[b];
                self.equity[idx] = (win as f32 + 0.5 * tie as f32) / total_opp;
            }
            for k in g..end {
                let (a, b) = self.combos[(self.k32[k] & 0x7ff) as usize];
                let (a, b) = (a as usize, b as usize);
                less_card[a] += 1;
                less_card[b] += 1;
                grp_card[a] = 0;
                grp_card[b] = 0;
            }
            less_total += grp;
            g = end;
        }
    }

    /// Fill per-combo classes for this board (rank-pair + flush tables; no sorting).
    /// Invalid combos (sharing a board card) get class u16::MAX and an all-ones mask.
    fn fill_classes(&mut self, board: &[u8; 5], classes: &mut [u16; COMBOS], masks: &mut [u64; COMBOS]) {
        let lut = Lut::get();
        let bm = board_mask(board);
        let bh = LHand::of(board);
        let mut rank_cnt = [0u8; 13];
        let mut suit_cnt = [0u8; 4];
        for &c in board {
            rank_cnt[(c / 4) as usize] += 1;
            suit_cnt[(c % 4) as usize] += 1;
        }
        let mut pair_class = [[0u16; 13]; 13];
        for rx in 0..13 {
            for ry in rx..13 {
                if rank_cnt[rx] + 1 + (rx == ry) as u8 > 4 || rank_cnt[ry] + 1 > 4 {
                    continue;
                }
                let c = lut.class_nonflush(bh.key + POW5[rx] + POW5[ry]);
                pair_class[rx][ry] = c;
                pair_class[ry][rx] = c;
            }
        }
        let flush_suit = (0..4u8).find(|&s| suit_cnt[s as usize] >= 3).unwrap_or(4);
        let bsr = if flush_suit < 4 { ((bm >> (16 * flush_suit as u32)) & 0x1fff) as u32 } else { 0 };
        for (i, &(a, b)) in self.combos.iter().enumerate() {
            let m = card_bit(a) | card_bit(b);
            if bm & m != 0 {
                classes[i] = u16::MAX;
                masks[i] = u64::MAX;
                continue;
            }
            masks[i] = m;
            let mut cls = pair_class[(a / 4) as usize][(b / 4) as usize];
            if flush_suit < 4 {
                let sa = (a % 4 == flush_suit) as u8;
                let sb = (b % 4 == flush_suit) as u8;
                if suit_cnt[flush_suit as usize] + sa + sb >= 5 {
                    let mut fm = bsr;
                    if sa == 1 {
                        fm |= 1 << (a / 4);
                    }
                    if sb == 1 {
                        fm |= 1 << (b / 4);
                    }
                    cls = lut.class_flush(fm);
                }
            }
            classes[i] = cls;
        }
    }

    /// River equities of only the given hero hands (what one MCCFR iteration needs: at most
    /// six). Cost: one class fill for the board plus one branchless 1,326-entry pass per hero.
    pub fn compute_heroes(&mut self, board: &[u8; 5], heroes: &[(u8, u8)], out: &mut [f32]) {
        let mut classes = [0u16; COMBOS];
        let mut masks = [0u64; COMBOS];
        self.fill_classes(board, &mut classes, &mut masks);
        for (h, &(a, b)) in heroes.iter().enumerate() {
            let hi = combo_index(a, b);
            let hc = classes[hi];
            let hm = masks[hi];
            debug_assert!(hc != u16::MAX, "hero shares a card with the board");
            let (mut win, mut tie) = (0u32, 0u32);
            for j in 0..COMBOS {
                let ok = (masks[j] & hm == 0) as u32; // disjoint from hero, valid
                win += ok & (classes[j] < hc) as u32;
                tie += ok & (classes[j] == hc) as u32;
            }
            out[h] = (win as f32 + 0.5 * tie as f32) / 990.0;
        }
    }

    /// Previous implementation (oracle evaluator + comparison sort), kept for tests/benchmarks.
    pub fn compute_oracle(&mut self, board: &[u8; 5]) {
        let bm = board_mask(board);
        self.keys.clear();
        for (i, &(a, b)) in self.combos.iter().enumerate() {
            let h = card_bit(a) | card_bit(b);
            if bm & h != 0 {
                self.equity[i] = f32::NAN;
                continue;
            }
            self.keys.push(((eval(bm | h) as u64) << 11) | i as u64);
        }
        self.keys.sort_unstable();
        let total_opp = 990.0f32;
        let mut less_total = 0u32;
        let mut less_card = [0u32; 52];
        let mut g = 0;
        while g < self.keys.len() {
            let strength = self.keys[g] >> 11;
            let mut end = g;
            let mut grp_card = [0u32; 52];
            while end < self.keys.len() && self.keys[end] >> 11 == strength {
                let (a, b) = self.combos[(self.keys[end] & 0x7ff) as usize];
                grp_card[a as usize] += 1;
                grp_card[b as usize] += 1;
                end += 1;
            }
            let grp = (end - g) as u32;
            for k in g..end {
                let idx = (self.keys[k] & 0x7ff) as usize;
                let (a, b) = self.combos[idx];
                let (a, b) = (a as usize, b as usize);
                // opponents below: exclude hands sharing a card with hero (none contain both)
                let win = less_total - less_card[a] - less_card[b];
                // opponents in the same group, excluding hero itself (the only hand with both cards)
                let tie = grp + 1 - grp_card[a] - grp_card[b];
                self.equity[idx] = (win as f32 + 0.5 * tie as f32) / total_opp;
            }
            for c in 0..52 {
                less_card[c] += grp_card[c];
            }
            less_total += grp;
            g = end;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cfr::rng::Rng;

    fn random_cards(rng: &mut Rng, n: usize) -> Vec<u8> {
        let mut deck: Vec<u8> = (0..52).collect();
        for i in 0..n {
            let j = i + (rng.next_u64() % (52 - i as u64)) as usize;
            deck.swap(i, j);
        }
        deck[..n].to_vec()
    }

    #[test]
    fn combo_index_matches_enumeration() {
        for (i, &(a, b)) in combo_cards().iter().enumerate() {
            assert_eq!(combo_index(a, b), i);
            assert_eq!(combo_index(b, a), i);
        }
    }

    #[test]
    fn batch_equals_single_on_random_boards() {
        let mut rng = Rng::new(5);
        let mut batch = RiverBatch::new();
        for _ in 0..30 {
            let c = random_cards(&mut rng, 5);
            let board = [c[0], c[1], c[2], c[3], c[4]];
            batch.compute(&board);
            let mut checked = 0;
            for (i, &(a, b)) in combo_cards().iter().enumerate() {
                if board.contains(&a) || board.contains(&b) {
                    assert!(batch.equity[i].is_nan());
                    continue;
                }
                if i % 7 == 0 {
                    let s = river_equity_single(&board, [a, b]);
                    assert!((s - batch.equity[i] as f64).abs() < 1e-6, "{board:?} {a} {b}: {s} vs {}", batch.equity[i]);
                    checked += 1;
                }
            }
            assert!(checked > 100);
        }
    }

    #[test]
    fn lut_batch_equals_oracle_batch_exactly() {
        let mut rng = Rng::new(15);
        let (mut a, mut b) = (RiverBatch::new(), RiverBatch::new());
        for _ in 0..2_000 {
            let c = random_cards(&mut rng, 5);
            let board = [c[0], c[1], c[2], c[3], c[4]];
            a.compute(&board);
            b.compute_oracle(&board);
            for i in 0..COMBOS {
                assert!(a.equity[i] == b.equity[i] || (a.equity[i].is_nan() && b.equity[i].is_nan()));
            }
        }
    }

    #[test]
    fn hero_path_equals_batch() {
        let mut rng = Rng::new(16);
        let mut rb = RiverBatch::new();
        let mut out = [0f32; 6];
        for _ in 0..2_000 {
            let c = random_cards(&mut rng, 17);
            let board = [c[0], c[1], c[2], c[3], c[4]];
            let heroes: Vec<(u8, u8)> = (0..6).map(|p| (c[5 + 2 * p], c[6 + 2 * p])).collect();
            rb.compute_heroes(&board, &heroes, &mut out);
            rb.compute(&board);
            for (p, &(a, b)) in heroes.iter().enumerate() {
                assert_eq!(out[p], rb.equity[combo_index(a, b)]);
            }
        }
    }

    #[test]
    fn mean_equity_is_one_half() {
        // every disjoint (hero, opponent) pair splits one point, so the mean over hands is 1/2
        let mut rng = Rng::new(6);
        let mut batch = RiverBatch::new();
        for _ in 0..20 {
            let c = random_cards(&mut rng, 5);
            batch.compute(&[c[0], c[1], c[2], c[3], c[4]]);
            let v: Vec<f64> = batch.equity.iter().filter(|x| !x.is_nan()).map(|&x| x as f64).collect();
            assert_eq!(v.len(), 1081);
            let mean = v.iter().sum::<f64>() / v.len() as f64;
            assert!((mean - 0.5).abs() < 1e-6, "{mean}");
        }
    }

    #[test]
    fn royal_flush_board_is_all_ties() {
        // As Ks Qs Js Ts: everyone plays the board
        let board = [12 * 4 + 3, 11 * 4 + 3, 10 * 4 + 3, 9 * 4 + 3, 8 * 4 + 3];
        let mut batch = RiverBatch::new();
        batch.compute(&board);
        assert!(batch.equity.iter().filter(|x| !x.is_nan()).all(|&x| (x - 0.5).abs() < 1e-6));
    }

    #[test]
    fn nuts_has_equity_near_one() {
        // board 2c 7d 9h Js 4c: no flush or straight possible except 8T (7-8-9-T-J); TT/8T...
        // Quads impossible; the unique nuts here is 8T (straight). Check 8T > 0.99.
        let card = |r: u8, s: u8| r * 4 + s;
        let board = [card(0, 0), card(5, 1), card(7, 2), card(9, 3), card(2, 0)];
        let eq = river_equity_single(&board, [card(6, 0), card(8, 1)]); // 8c Td
        assert!(eq > 0.98, "{eq}");
    }
}
