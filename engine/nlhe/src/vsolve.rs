//! M4 stage 4 (NLHE, heads-up): vector-based Linear CFR on a subgame (Supp pp. 21–22,
//! `docs/m4_design.md` §4).
//!
//! * One pass per traverser over the subgame's public tree with a reach vector over the 1,326
//!   hands per player; counterfactual values per hand come back up. Lossless: an infoset is
//!   (node, board, hand) on every street of the subgame (D-SRCH-15).
//! * Chance (a street change inside the subgame, or an all-in run-out) enumerates every card
//!   (exact; heads-up turn subgames have 44 rivers), instead of one sampled board per thread
//!   (D-SRCH-12).
//! * Terminals in O(n): folds from total and per-card opponent reach (card removal by
//!   inclusion–exclusion); showdowns by a sweep over hands sorted by strength.
//! * Linear CFR: iteration t's regrets and average-strategy contributions weighted by t [38].
//!   `play` = the final iteration (what the searcher plays), `avg` = the weighted average (σ).
//! * Frozen (node, hand) entries keep fixed probabilities for our actual hand (Algorithm 2).
//! Heads-up only; subgames with leaves (multiway flop, preflop) are not supported here.

use crate::betting::Rules;
use crate::equity::{combo_cards, COMBOS};
use crate::eval_lut::{LHand, Lut};
use crate::subgame::{Kind, Subgame};
use std::collections::HashMap;

const NO_CARD: u64 = 63;

/// Regrets and average-strategy sums of one (node, later cards) block, f32 to halve memory
/// (a heads-up turn subgame holds ~9k blocks: ~280 MB in f32).
struct Block {
    reg: Vec<f32>,
    sum: Vec<f32>,
}

pub struct VectorSolver {
    pub g: Subgame,
    pub rules: Rules,
    pub board: Vec<u8>,
    pub ranges: [Vec<f64>; 2],
    /// (node, combo) → fixed probabilities over that node's actions (our actual hand)
    pub frozen: HashMap<(u32, u16), Vec<f64>>,
    combos: Vec<(u8, u8)>,
    blocks: HashMap<u64, Block>,
    ranks: HashMap<u64, (Vec<u16>, Vec<u16>)>,
    pool: Vec<Vec<f64>>,
    /// normaliser: total weight of card-compatible hand pairs at the root
    pub z: f64,
    pub iterations: u64,
}

fn mask(cards: &[u8]) -> u64 {
    cards.iter().fold(0u64, |m, &c| m | 1 << c)
}

impl VectorSolver {
    pub fn new(g: Subgame, rules: Rules, board: &[u8], ranges: [Vec<f64>; 2]) -> Self {
        assert_eq!(rules.players, 2, "the vector solver is heads-up only");
        assert!(g.nodes.iter().all(|n| n.kind != Kind::Leaf), "subgames with leaves are not supported");
        let combos = combo_cards();
        let bm = mask(board);
        let mut ranges = ranges;
        for r in ranges.iter_mut() {
            for (h, x) in r.iter_mut().enumerate() {
                if bm & mask(&[combos[h].0, combos[h].1]) != 0 {
                    *x = 0.0;
                }
            }
        }
        let mut s = VectorSolver { g, rules, board: board.to_vec(), ranges, frozen: HashMap::new(), combos, blocks: HashMap::new(), ranks: HashMap::new(), pool: Vec::new(), z: 1.0, iterations: 0 };
        let mut c = vec![0f64; COMBOS];
        s.compat_into(&s.ranges[1], 1.0, &mut c);
        s.z = c.iter().zip(&s.ranges[0]).map(|(c, r)| c * r).sum();
        assert!(s.z > 0.0, "no card-compatible pair of hands has positive belief");
        s
    }

    /// A zeroed buffer of length `n` from the pool. The hot path allocates nothing after warm-up:
    /// with one solver per thread, per-node allocations serialised on the Windows heap lock
    /// (20 concurrent searches ran ~50x slower than one).
    fn take(&mut self, n: usize) -> Vec<f64> {
        let mut v = self.pool.pop().unwrap_or_default();
        v.clear();
        v.resize(n, 0.0);
        v
    }

    /// out[h] = scale · Σ_o reach[o] over opponent hands o sharing no card with h.
    fn compat_into(&self, reach: &[f64], scale: f64, out: &mut [f64]) {
        let total: f64 = reach.iter().sum();
        let mut per = [0f64; 52];
        for (h, &r) in reach.iter().enumerate() {
            let (a, b) = self.combos[h];
            per[a as usize] += r;
            per[b as usize] += r;
        }
        for h in 0..COMBOS {
            let (a, b) = self.combos[h];
            out[h] = scale * (total - per[a as usize] - per[b as usize] + reach[h]);
        }
    }

    #[cfg(test)]
    fn compat_sums(&self, reach: &[f64]) -> Vec<f64> {
        let mut v = vec![0f64; COMBOS];
        self.compat_into(reach, 1.0, &mut v);
        v
    }

    /// Hand ranks on a full board (0 = hand blocked by the board) and the unblocked hands sorted
    /// by rank, cached per board.
    fn ensure_ranks(&mut self, board: &[u8]) -> u64 {
        let key = mask(board);
        if !self.ranks.contains_key(&key) {
            let lut = Lut::get();
            let bh = LHand::of(board);
            let ranks: Vec<u16> = self.combos.iter().map(|&(a, b)| if key & mask(&[a, b]) != 0 { 0 } else { lut.class(bh.add(a).add(b)) + 1 }).collect();
            let mut order: Vec<u16> = (0..COMBOS as u16).filter(|&h| ranks[h as usize] > 0).collect();
            order.sort_by_key(|&h| ranks[h as usize]);
            self.ranks.insert(key, (ranks, order));
        }
        key
    }

    /// out[h] += scale · Σ_o reach[o]·compat(h, o)·sign(rank h − rank o) on a full board, by a
    /// sweep over hands sorted by strength. `out` must be zeroed.
    fn showdown_into(&mut self, board: &[u8], reach: &[f64], scale: f64, out: &mut [f64]) {
        let key = self.ensure_ranks(board);
        let (ranks, order) = &self.ranks[&key];
        let combos = &self.combos;
        let rk = |k: usize| ranks[order[k] as usize];
        // weaker: ascending sweep, groups of equal rank
        let (mut tot, mut per) = (0f64, [0f64; 52]);
        let mut i = 0;
        while i < order.len() {
            let mut j = i;
            while j < order.len() && rk(j) == rk(i) {
                j += 1;
            }
            for &h in &order[i..j] {
                let (a, b) = combos[h as usize];
                out[h as usize] += tot - per[a as usize] - per[b as usize];
            }
            for &h in &order[i..j] {
                let (a, b) = combos[h as usize];
                let r = reach[h as usize];
                tot += r;
                per[a as usize] += r;
                per[b as usize] += r;
            }
            i = j;
        }
        // stronger: descending sweep
        let (mut tot, mut per) = (0f64, [0f64; 52]);
        let mut i = order.len();
        while i > 0 {
            let mut j = i;
            while j > 0 && rk(j - 1) == rk(i - 1) {
                j -= 1;
            }
            for &h in &order[j..i] {
                let (a, b) = combos[h as usize];
                out[h as usize] -= tot - per[a as usize] - per[b as usize];
            }
            for &h in &order[j..i] {
                let (a, b) = combos[h as usize];
                let r = reach[h as usize];
                tot += r;
                per[a as usize] += r;
                per[b as usize] += r;
            }
            i = j;
        }
        if scale != 1.0 {
            out.iter_mut().for_each(|x| *x *= scale);
        }
    }

    #[cfg(test)]
    fn showdown(&mut self, board: &[u8], reach: &[f64]) -> Vec<f64> {
        let mut v = vec![0f64; COMBOS];
        self.showdown_into(board, reach, 1.0, &mut v);
        v
    }

    fn key(node: u32, board: &[u8], root_len: usize) -> u64 {
        // decisions after the root's street are keyed by the cards dealt since the root
        let mut k = node as u64;
        for &c in &board[root_len..] {
            k = k * 64 + c as u64;
        }
        if board.len() == root_len {
            k = k * 64 + NO_CARD;
        }
        k
    }

    /// Strategy at (node, board) for every hand, row-major n per hand: regret matching
    /// (`avg` false) or the weighted average (regret matching where it is still empty).
    fn fill_strategy(&self, key: u64, n: usize, avg: bool, s: &mut [f64]) {
        let u = 1.0 / n as f64;
        let Some(b) = self.blocks.get(&key) else {
            s.fill(u);
            return;
        };
        for h in 0..COMBOS {
            let row = &mut s[h * n..(h + 1) * n];
            if avg {
                let sm = &b.sum[h * n..(h + 1) * n];
                let tot: f64 = sm.iter().map(|&x| x as f64).sum();
                if tot > 0.0 {
                    for a in 0..n {
                        row[a] = sm[a] as f64 / tot;
                    }
                    continue;
                }
            }
            let r = &b.reg[h * n..(h + 1) * n];
            let pos: f64 = r.iter().map(|&x| x.max(0.0) as f64).sum();
            if pos > 0.0 {
                for a in 0..n {
                    row[a] = r[a].max(0.0) as f64 / pos;
                }
            } else {
                row.fill(u);
            }
        }
    }

    fn apply_frozen(&self, node: u32, n: usize, s: &mut [f64]) {
        for (&(nd, h), p) in &self.frozen {
            if nd == node {
                s[h as usize * n..(h as usize + 1) * n].copy_from_slice(p);
            }
        }
    }

    /// Counterfactual values for player `i` (`br`: best response for `i` against `avg`
    /// (true) or final (false) strategies; otherwise a CFR pass at iteration weight `t`).
    /// The returned buffer comes from the pool; callers hand it back.
    fn walk(&mut self, node: u32, board: &mut Vec<u8>, reach: &[Vec<f64>; 2], i: usize, t: f64, br: Option<bool>) -> Vec<f64> {
        let nd = self.g.nodes[node as usize];
        let o = 1 - i;
        match nd.kind {
            Kind::FoldEnd => {
                let pay = if nd.state.folded & (1 << i) != 0 { -(nd.state.total[i] as f64) } else { nd.state.total[o] as f64 };
                let mut v = self.take(COMBOS);
                self.compat_into(&reach[o], pay / self.z, &mut v);
                v
            }
            Kind::Showdown if board.len() == 5 => {
                let scale = nd.state.total[o] as f64 / self.z;
                let mut v = self.take(COMBOS);
                self.showdown_into(board, &reach[o], scale, &mut v);
                v
            }
            Kind::Showdown | Kind::Chance => {
                // deal the next card: every card not on the board (and not in either hand)
                let child = if nd.kind == Kind::Chance { Some(self.g.child(node, 0)) } else { None };
                let w = 1.0 / (52 - board.len() - 4) as f64;
                let mut v = self.take(COMBOS);
                let mut r2 = [self.take(COMBOS), self.take(COMBOS)];
                let bm = mask(board);
                for c in (0..52u8).filter(|&c| bm & (1 << c) == 0) {
                    r2[0].copy_from_slice(&reach[0]);
                    r2[1].copy_from_slice(&reach[1]);
                    for (h, &(a, b)) in self.combos.iter().enumerate() {
                        if a == c || b == c {
                            r2[0][h] = 0.0;
                            r2[1][h] = 0.0;
                        }
                    }
                    board.push(c);
                    let vc = self.walk(child.unwrap_or(node), board, &r2, i, t, br);
                    board.pop();
                    for (h, &(a, b)) in self.combos.iter().enumerate() {
                        if a != c && b != c {
                            v[h] += vc[h] * w;
                        }
                    }
                    self.pool.push(vc);
                }
                self.pool.extend(r2);
                v
            }
            Kind::Leaf => unreachable!(),
            Kind::Decision => {
                let p = nd.state.to_act as usize;
                let n = nd.n as usize;
                let key = Self::key(node, board, self.board.len());
                let mut sig = self.take(COMBOS * n);
                self.fill_strategy(key, n, br == Some(true), &mut sig);
                self.apply_frozen(node, n, &mut sig);
                let mut v = self.take(COMBOS);
                if p == i {
                    let mut va = Vec::with_capacity(n);
                    for a in 0..n {
                        let ch = self.g.child(node, a);
                        va.push(self.walk(ch, board, reach, i, t, br));
                    }
                    if br.is_some() {
                        for h in 0..COMBOS {
                            v[h] = (0..n).map(|a| va[a][h]).fold(f64::NEG_INFINITY, f64::max);
                        }
                    } else {
                        for h in 0..COMBOS {
                            v[h] = (0..n).map(|a| sig[h * n + a] * va[a][h]).sum();
                        }
                        let frozen: Vec<u16> = self.frozen.keys().filter(|(nd2, _)| *nd2 == node).map(|&(_, h)| h).collect();
                        let blk = self.blocks.entry(key).or_insert_with(|| Block { reg: vec![0.0; COMBOS * n], sum: vec![0.0; COMBOS * n] });
                        for h in 0..COMBOS {
                            if reach[i][h] == 0.0 && v[h] == 0.0 || frozen.contains(&(h as u16)) {
                                continue;
                            }
                            for a in 0..n {
                                blk.reg[h * n + a] += (t * (va[a][h] - v[h])) as f32;
                                blk.sum[h * n + a] += (t * reach[i][h] * sig[h * n + a]) as f32;
                            }
                        }
                    }
                    self.pool.extend(va);
                } else {
                    let mut r2 = [self.take(COMBOS), self.take(COMBOS)];
                    for a in 0..n {
                        r2[0].copy_from_slice(&reach[0]);
                        r2[1].copy_from_slice(&reach[1]);
                        for h in 0..COMBOS {
                            r2[p][h] *= sig[h * n + a];
                        }
                        let ch = self.g.child(node, a);
                        let va = self.walk(ch, board, &r2, i, t, br);
                        for h in 0..COMBOS {
                            v[h] += va[h];
                        }
                        self.pool.push(va);
                    }
                    self.pool.extend(r2);
                }
                self.pool.push(sig);
                v
            }
        }
    }

    /// Run `iters` more iterations of Linear CFR.
    pub fn run(&mut self, iters: u64) {
        let reach = self.ranges.clone();
        for _ in 0..iters {
            self.iterations += 1;
            let t = self.iterations as f64;
            for i in 0..2 {
                let mut b = self.board.clone();
                let v = self.walk(0, &mut b, &reach, i, t, None);
                self.pool.push(v);
            }
        }
    }

    /// Strategy at (node, cards dealt since the root) for one hand: final iteration or average.
    pub fn strategy(&self, node: u32, later_cards: &[u8], combo: usize, avg: bool) -> Vec<f64> {
        let n = self.g.nodes[node as usize].n as usize;
        let mut b = self.board.clone();
        b.extend_from_slice(later_cards);
        let key = Self::key(node, &b, self.board.len());
        let mut s = vec![0f64; COMBOS * n];
        self.fill_strategy(key, n, avg, &mut s);
        if let Some(f) = self.frozen.get(&(node, combo as u16)) {
            return f.clone();
        }
        s[combo * n..(combo + 1) * n].to_vec()
    }

    /// Strategy of every hand at (node, cards dealt since the root), row-major (n per hand),
    /// with frozen entries applied.
    pub fn strategy_all(&self, node: u32, later_cards: &[u8], avg: bool) -> Vec<f64> {
        let n = self.g.nodes[node as usize].n as usize;
        let mut b = self.board.clone();
        b.extend_from_slice(later_cards);
        let key = Self::key(node, &b, self.board.len());
        let mut s = vec![0f64; COMBOS * n];
        self.fill_strategy(key, n, avg, &mut s);
        self.apply_frozen(node, n, &mut s);
        s
    }

    /// Value for player `i` of a best response against the other's average (or final) strategy.
    pub fn best_response_value(&mut self, i: usize, avg: bool) -> f64 {
        let reach = self.ranges.clone();
        let mut b = self.board.clone();
        let v = self.walk(0, &mut b, &reach, i, 0.0, Some(avg));
        v.iter().zip(&self.ranges[i]).map(|(x, r)| x * r).sum()
    }

    /// Exploitability of the average (or final) strategy pair within the subgame, chips/hand.
    pub fn exploitability(&mut self, avg: bool) -> f64 {
        (self.best_response_value(0, avg) + self.best_response_value(1, avg)) / 2.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::betting::{Action, Next};
    use crate::config::AbstractionConfig;
    use crate::equity::combo_index;
    use crate::subgame::round_start;
    use cfr::rng::Rng;

    fn river_root(rules: &Rules) -> crate::betting::BState {
        let mut t = [0u32; 6];
        t[0] = 500;
        t[1] = 500;
        round_start(rules, 3, t, 0)
    }

    #[test]
    fn terminal_evaluations_match_brute_force() {
        let rules = Rules { players: 2, ..Rules::default() };
        let cfg = AbstractionConfig::parse("allin = true").unwrap();
        let board = [2u8, 17, 30, 41, 51];
        let g = Subgame::build(&rules, &cfg, river_root(&rules));
        let mut rng = Rng::new(4);
        let r: Vec<f64> = (0..COMBOS).map(|_| rng.next_f64()).collect();
        let mut s = VectorSolver::new(g, rules, &board, [r.clone(), r.clone()]);
        let reach = s.ranges[1].clone();
        let sd = s.showdown(&board, &reach);
        let cs = s.compat_sums(&reach);
        let combos = combo_cards();
        let lut = Lut::get();
        let bh = LHand::of(&board);
        let rank = |h: usize| lut.class(bh.add(combos[h].0).add(combos[h].1));
        let bm = mask(&board);
        for h in (0..COMBOS).step_by(37).filter(|&h| bm & mask(&[combos[h].0, combos[h].1]) == 0) {
            let (mut want_sd, mut want_cs) = (0f64, 0f64);
            for o in 0..COMBOS {
                if mask(&[combos[o].0, combos[o].1]) & mask(&[combos[h].0, combos[h].1]) != 0 {
                    continue;
                }
                want_cs += reach[o];
                if bm & mask(&[combos[o].0, combos[o].1]) == 0 {
                    want_sd += reach[o] * (rank(h).cmp(&rank(o)) as i32) as f64; // ties count 0
                }
            }
            assert!((sd[h] - want_sd).abs() < 1e-9, "showdown {h}: {} vs {want_sd}", sd[h]);
            assert!((cs[h] - want_cs).abs() < 1e-9, "compat {h}");
        }
    }

    #[test]
    fn a_known_winner_gets_exactly_the_pot_already_in() {
        // one hand each, both known: aces vs a weaker hand on a dry board. The loser never puts
        // more chips in (folds to a bet, checks behind), so the winner's value is the 500 the
        // loser already contributed.
        let rules = Rules { players: 2, ..Rules::default() };
        let cfg = AbstractionConfig::parse("allin = true\nriver.0 = 1\nriver.1+ = 1").unwrap();
        let board = [46u8, 30, 13, 7, 39]; // K 9 5 3 J
        let g = Subgame::build(&rules, &cfg, river_root(&rules));
        let (mut r0, mut r1) = (vec![0.0; COMBOS], vec![0.0; COMBOS]);
        r0[combo_index(20, 1)] = 1.0; // 7-2 (BB acts first heads-up post-flop: seat 1)
        r1[combo_index(48, 49)] = 1.0; // aces
        let mut s = VectorSolver::new(g, rules, &board, [r0, r1]);
        s.run(400);
        let br0 = s.best_response_value(0, true);
        let br1 = s.best_response_value(1, true);
        assert!((br1 - 500.0).abs() < 1.0 && (br0 + 500.0).abs() < 1.0, "{br0} {br1}");
    }

    #[test]
    fn river_and_turn_subgames_converge() {
        let rules = Rules { players: 2, ..Rules::default() };
        let cfg = AbstractionConfig::parse("allin = true\nturn.0 = 1\nriver.0 = 1\nriver.1+ = 1").unwrap();
        let mut rng = Rng::new(9);
        let r: Vec<f64> = (0..COMBOS).map(|_| rng.next_f64()).collect();
        let g = Subgame::build(&rules, &cfg, river_root(&rules));
        let mut s = VectorSolver::new(g, rules, &[2, 17, 30, 41, 51], [r.clone(), r.clone()]);
        s.run(20);
        let e1 = s.exploitability(true);
        s.run(280);
        let e2 = s.exploitability(true);
        eprintln!("river: exploitability {e1:.3} -> {e2:.3} chips (pot 1000)");
        assert!(e2 < e1 / 3.0 && e2 < 10.0, "{e1} -> {e2}");
        // turn root: the river is enumerated (44 cards)
        let mut t = [0u32; 6];
        t[0] = 500;
        t[1] = 500;
        let root = round_start(&rules, 2, t, 0);
        let g = Subgame::build(&rules, &cfg, root);
        let mut s = VectorSolver::new(g, rules, &[2, 17, 30, 41], [r.clone(), r]);
        s.run(60);
        let e = s.exploitability(true);
        eprintln!("turn: exploitability {e:.3} after 60 iterations");
        assert!(e < 25.0, "{e}");
    }

    #[test]
    fn frozen_hand_keeps_its_probabilities() {
        let rules = Rules { players: 2, ..Rules::default() };
        let cfg = AbstractionConfig::parse("allin = true").unwrap();
        let g = Subgame::build(&rules, &cfg, river_root(&rules));
        let mut s = VectorSolver::new(g, rules, &[2, 17, 30, 41, 51], [vec![1.0; COMBOS], vec![1.0; COMBOS]]);
        let h = combo_index(48, 49);
        s.frozen.insert((0, h as u16), vec![0.25, 0.75]);
        s.run(50);
        assert_eq!(s.strategy(0, &[], h, true), vec![0.25, 0.75]);
        assert_eq!(s.strategy(0, &[], h, false), vec![0.25, 0.75]);
        let _ = (Action::Call, Next::Terminal);
    }
}
