//! M4 6-max, stage 1: multiway subgame solver for the river (`docs/m4_6max.md`).
//!
//! Pluribus solves large or early subgames with "Monte Carlo Linear CFR … just as it was for the
//! blueprint" (Supp p. 21); vector CFR's O(n) terminals do not extend to 3+ players' card
//! removal. Here: external-sampling Linear MCCFR on a [`Subgame`] without chance nodes or leaves
//! (a river subgame).
//! * Each iteration deals one hand to every player still in the hand, jointly from the beliefs:
//!   independent draws from the ranges, card-conflicting deals rejected, so a deal's probability
//!   is ∝ Π ranges (D-SRCH-1). Folded players get no hand and no say: their contributions stay
//!   in the pot (their cards are unknown to the searcher, D-SRCH-14).
//! * Information set = (node, the actor's exact hand): lossless on the current round.
//! * Linear CFR: iteration t's regrets and average-strategy contributions weighted by t [38];
//!   the traverser explores all its actions, the others sample one from their current strategy
//!   and add it to their average (external sampling).
//! * Frozen (node, hand) entries keep fixed probabilities (our earlier actions, Algorithm 2).
//! * Equal starting stacks: every player still in at showdown has the same contribution, so
//!   there are no side pots (asserted).

use crate::betting::Rules;
use crate::equity::{combo_cards, COMBOS};
use crate::eval_lut::{LHand, Lut};
use crate::subgame::{Kind, Subgame};
use cfr::rng::Rng;
use std::collections::HashMap;

const NO_HAND: u16 = u16::MAX;

struct Block {
    reg: Vec<f32>,
    sum: Vec<f32>,
}

pub struct MultiSolver {
    pub g: Subgame,
    pub rules: Rules,
    pub board: Vec<u8>,
    /// (node, combo) → fixed probabilities over that node's actions (our actual hand)
    pub frozen: HashMap<(u32, u16), Vec<f64>>,
    /// seats still in the hand at the root
    pub live: Vec<usize>,
    /// per seat: cumulative range over combos (board-blocked combos removed); empty if folded
    cdf: Vec<Vec<f64>>,
    combos: Vec<(u8, u8)>,
    blocks: HashMap<u32, Block>,
    rng: Rng,
    pub iterations: u64,
    /// joint deals rejected for sharing a card
    pub rejected: u64,
}

impl MultiSolver {
    /// `ranges[p]` for every seat (ignored for folded seats); `board` = the five river cards.
    pub fn new(g: Subgame, rules: Rules, board: &[u8], ranges: &[Vec<f64>], seed: u64) -> Self {
        assert_eq!(board.len(), 5, "river subgames only (no chance nodes)");
        assert!(g.nodes.iter().all(|n| !matches!(n.kind, Kind::Chance | Kind::Leaf)), "subgame has chance nodes or leaves");
        let combos = combo_cards();
        let folded = g.nodes[0].state.folded;
        let live: Vec<usize> = (0..rules.players).filter(|&p| folded & (1 << p) == 0).collect();
        let mut cdf = vec![Vec::new(); rules.players];
        for &p in &live {
            let mut acc = 0.0;
            cdf[p] = (0..COMBOS)
                .map(|h| {
                    let (a, b) = combos[h];
                    if !board.contains(&a) && !board.contains(&b) {
                        acc += ranges[p][h];
                    }
                    acc
                })
                .collect();
            assert!(acc > 0.0, "seat {p} has no possible hand");
        }
        MultiSolver { g, rules, board: board.to_vec(), frozen: HashMap::new(), live, cdf, combos, blocks: HashMap::new(), rng: Rng::new(seed), iterations: 0, rejected: 0 }
    }

    /// One joint deal: a combo per live seat, NO_HAND for folded seats.
    pub fn deal(&mut self) -> Vec<u16> {
        for _ in 0..1_000_000 {
            let mut hands = vec![NO_HAND; self.rules.players];
            let mut used = 0u64;
            let mut ok = true;
            for &p in &self.live {
                let c = &self.cdf[p];
                let x = self.rng.next_f64() * c[COMBOS - 1];
                let h = c.partition_point(|&v| v <= x).min(COMBOS - 1);
                let (a, b) = self.combos[h];
                let m = (1u64 << a) | (1u64 << b);
                if used & m != 0 {
                    ok = false;
                    break;
                }
                used |= m;
                hands[p] = h as u16;
            }
            if ok {
                return hands;
            }
            self.rejected += 1;
        }
        panic!("no card-compatible deal in 1,000,000 tries");
    }

    /// Chips won (or lost) by seat `i` at terminal `node` with these hands.
    pub fn payoff(&self, node: u32, hands: &[u16], i: usize) -> f64 {
        let st = self.g.nodes[node as usize].state;
        let n = self.rules.players;
        let pot: f64 = st.total[..n].iter().map(|&x| x as f64).sum();
        let mine = st.total[i] as f64;
        if st.folded & (1 << i) != 0 {
            return -mine;
        }
        let contenders: Vec<usize> = (0..n).filter(|&q| st.folded & (1 << q) == 0).collect();
        if contenders.len() == 1 {
            return pot - mine;
        }
        debug_assert!(contenders.iter().all(|&q| st.total[q] == st.total[i]), "side pot");
        let lut = Lut::get();
        let bh = LHand::of(&self.board);
        let rank = |q: usize| {
            let (a, b) = self.combos[hands[q] as usize];
            lut.class(bh.add(a).add(b))
        };
        let best = contenders.iter().map(|&q| rank(q)).max().unwrap();
        let winners = contenders.iter().filter(|&&q| rank(q) == best).count() as f64;
        if rank(i) == best {
            pot / winners - mine
        } else {
            -mine
        }
    }

    /// Current (regret matching) or average strategy at (node, combo), frozen entries applied.
    pub fn strategy(&self, node: u32, h: usize, avg: bool) -> Vec<f64> {
        if let Some(f) = self.frozen.get(&(node, h as u16)) {
            return f.clone();
        }
        let n = self.g.nodes[node as usize].n as usize;
        let uniform = vec![1.0 / n as f64; n];
        let Some(b) = self.blocks.get(&node) else { return uniform };
        if avg {
            let s = &b.sum[h * n..(h + 1) * n];
            let tot: f64 = s.iter().map(|&x| x as f64).sum();
            if tot > 0.0 {
                return s.iter().map(|&x| x as f64 / tot).collect();
            }
        }
        let r = &b.reg[h * n..(h + 1) * n];
        let pos: f64 = r.iter().map(|&x| x.max(0.0) as f64).sum();
        if pos > 0.0 {
            r.iter().map(|&x| x.max(0.0) as f64 / pos).collect()
        } else {
            uniform
        }
    }

    fn block(&mut self, node: u32) -> &mut Block {
        let n = self.g.nodes[node as usize].n as usize;
        self.blocks.entry(node).or_insert_with(|| Block { reg: vec![0.0; COMBOS * n], sum: vec![0.0; COMBOS * n] })
    }

    fn traverse(&mut self, node: u32, hands: &[u16], i: usize, t: f64) -> f64 {
        let nd = self.g.nodes[node as usize];
        match nd.kind {
            Kind::FoldEnd | Kind::Showdown => self.payoff(node, hands, i),
            Kind::Chance | Kind::Leaf => unreachable!(),
            Kind::Decision => {
                let p = nd.state.to_act as usize;
                let n = nd.n as usize;
                let h = hands[p] as usize;
                let sig = self.strategy(node, h, false);
                let frozen = self.frozen.contains_key(&(node, h as u16));
                if p == i {
                    let va: Vec<f64> = (0..n).map(|a| self.traverse(self.g.child(node, a), hands, i, t)).collect();
                    let v: f64 = sig.iter().zip(&va).map(|(s, x)| s * x).sum();
                    if !frozen {
                        let b = self.block(node);
                        for a in 0..n {
                            b.reg[h * n + a] += (t * (va[a] - v)) as f32;
                        }
                    }
                    v
                } else {
                    if !frozen {
                        let b = self.block(node);
                        for a in 0..n {
                            b.sum[h * n + a] += (t * sig[a]) as f32;
                        }
                    }
                    let a = self.rng.sample_index(&sig);
                    self.traverse(self.g.child(node, a), hands, i, t)
                }
            }
        }
    }

    /// Run `iters` more iterations: one joint deal each, every live seat traverses once.
    pub fn run(&mut self, iters: u64) {
        for _ in 0..iters {
            self.iterations += 1;
            let t = self.iterations as f64;
            let hands = self.deal();
            for k in 0..self.live.len() {
                let i = self.live[k];
                self.traverse(0, &hands, i, t);
            }
        }
    }

    /// Bytes held by the regret and average tables (memory per search).
    pub fn table_bytes(&self) -> usize {
        self.blocks.values().map(|b| (b.reg.len() + b.sum.len()) * 4).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::belief::Beliefs;
    use crate::config::AbstractionConfig;
    use crate::equity::combo_index;
    use crate::subgame::round_start;

    const BOARD: [u8; 5] = [2, 17, 30, 41, 51];

    fn ranges(players: usize, support: usize, seed: u64) -> Vec<Vec<f64>> {
        let mut rng = Rng::new(seed);
        (0..players)
            .map(|_| {
                let mut r = vec![0f64; COMBOS];
                while r.iter().filter(|&&x| x > 0.0).count() < support {
                    let (a, b) = ((rng.next_u64() % 52) as u8, (rng.next_u64() % 52) as u8);
                    if a != b && !BOARD.contains(&a) && !BOARD.contains(&b) {
                        r[combo_index(a, b)] = 0.2 + rng.next_f64();
                    }
                }
                r
            })
            .collect()
    }

    fn river_game(players: usize, folded: u8, cfg: &str) -> (Rules, Subgame) {
        let rules = Rules { players, ..Rules::default() };
        let c = AbstractionConfig::parse(cfg).unwrap();
        let mut t = [0u32; 6];
        for (p, x) in t.iter_mut().enumerate().take(players) {
            *x = if folded & (1 << p) != 0 { 100 } else { 300 };
        }
        (rules, Subgame::build_with(&rules, &c, &c, round_start(&rules, 3, t, folded), 200_000))
    }

    #[test]
    fn beliefs_stay_normalised_for_six_players() {
        let mut b = Beliefs::uniform(6);
        b.observe_cards(&BOARD);
        let mut rng = Rng::new(4);
        for p in 0..6 {
            let lik: Vec<f64> = (0..COMBOS).map(|_| rng.next_f64()).collect();
            b.observe_action(p, &lik);
        }
        for (p, r) in b.ranges.iter().enumerate() {
            let s: f64 = r.iter().sum();
            assert!((s - 1.0).abs() < 1e-9, "seat {p}: {s}");
            for (h, &(x, y)) in combo_cards().iter().enumerate() {
                if BOARD.contains(&x) || BOARD.contains(&y) {
                    assert_eq!(r[h], 0.0, "board-blocked combo kept");
                }
            }
        }
    }

    #[test]
    fn joint_deals_follow_the_product_of_ranges() {
        // two live seats with 3 hands each, one pair conflicting: frequencies ∝ r0 · r1 · compat
        let (rules, g) = river_game(3, 0b100, "allin = true\nriver.0 = 1");
        let mut r = vec![vec![0f64; COMBOS]; 3];
        let h0 = [combo_index(0, 1), combo_index(3, 4), combo_index(5, 6)];
        let h1 = [combo_index(0, 7), combo_index(8, 9), combo_index(10, 11)];
        for (k, &h) in h0.iter().enumerate() {
            r[0][h] = 1.0 + k as f64;
        }
        for (k, &h) in h1.iter().enumerate() {
            r[1][h] = 3.0 - k as f64;
        }
        r[2] = vec![1.0; COMBOS]; // folded: ignored
        let mut s = MultiSolver::new(g, rules, &BOARD, &r, 9);
        assert_eq!(s.live, vec![0, 1]);
        let n = 200_000;
        let mut count = HashMap::new();
        for _ in 0..n {
            let d = s.deal();
            assert_eq!(d[2], NO_HAND, "a folded seat got a hand");
            *count.entry((d[0], d[1])).or_insert(0u32) += 1;
        }
        let b = Beliefs { ranges: r.clone(), zero_prob_events: 0 };
        let combos = combo_cards();
        let w = |x: usize, y: usize| b.joint_weight(&[combos[x], combos[y]]);
        let z: f64 = h0.iter().flat_map(|&x| h1.iter().map(move |&y| (x, y))).map(|(x, y)| w(x, y)).sum();
        for &x in &h0 {
            for &y in &h1 {
                let want = w(x, y) / z;
                let got = *count.get(&(x as u16, y as u16)).unwrap_or(&0) as f64 / n as f64;
                assert!((got - want).abs() < 0.005, "{x},{y}: {got} vs {want}");
            }
        }
        assert!(count.get(&(h0[0] as u16, h1[0] as u16)).is_none(), "conflicting deal sampled");
    }

    #[test]
    fn folded_players_drop_out_and_payoffs_are_zero_sum() {
        // six seats, seats 1 and 3 folded earlier (100 chips each in the pot)
        let (rules, g) = river_game(6, 0b1010, "allin = true\nriver.0 = 1\nriver.1+ = 1");
        assert!(g.nodes.iter().filter(|n| n.kind == Kind::Decision).all(|n| n.state.folded & (1 << n.state.to_act) == 0), "a folded seat acts");
        let r = ranges(6, 40, 1);
        let mut s = MultiSolver::new(g, rules, &BOARD, &r, 2);
        assert_eq!(s.live, vec![0, 2, 4, 5]);
        for _ in 0..200 {
            let d = s.deal();
            assert!(d[1] == NO_HAND && d[3] == NO_HAND);
            for node in (0..s.g.nodes.len() as u32).filter(|&k| matches!(s.g.nodes[k as usize].kind, Kind::FoldEnd | Kind::Showdown)) {
                let tot: f64 = (0..6).map(|i| s.payoff(node, &d, i)).sum();
                assert!(tot.abs() < 1e-9, "not zero-sum at node {node}: {tot}");
                assert_eq!(s.payoff(node, &d, 1), -100.0);
            }
        }
        s.run(50);
        assert!(s.blocks.keys().all(|&k| s.g.nodes[k as usize].state.to_act != 1 && s.g.nodes[k as usize].state.to_act != 3));
    }

    #[test]
    fn small_budget_search_reproduces_the_converged_strategy() {
        // "Nothing changed": same subgame and beliefs as the reference (a long solve standing in
        // for a blueprint computed for this spot); a small-budget search with another seed picks
        // the same most likely action at the root for most of the first actor's range.
        let (rules, g1) = river_game(3, 0, "allin = true\nriver.0 = 1\nriver.1+ = 1");
        let (_, g2) = river_game(3, 0, "allin = true\nriver.0 = 1\nriver.1+ = 1");
        let r = ranges(3, 30, 7);
        let mut long = MultiSolver::new(g1, rules, &BOARD, &r, 11);
        let mut short = MultiSolver::new(g2, rules, &BOARD, &r, 12);
        long.run(40_000);
        let t0 = std::time::Instant::now();
        short.run(4_000);
        let p = long.g.nodes[0].state.to_act as usize;
        let argmax = |v: Vec<f64>| v.iter().enumerate().fold(0, |m, (k, &x)| if x > v[m] { k } else { m });
        let (mut agree, mut tot) = (0.0, 0.0);
        for h in 0..COMBOS {
            if r[p][h] > 0.0 {
                tot += r[p][h];
                if argmax(long.strategy(0, h, true)) == argmax(short.strategy(0, h, true)) {
                    agree += r[p][h];
                }
            }
        }
        eprintln!("3-player river: {} nodes; agreement {:.3}; 4,000 iterations in {:.2}s, tables {} KB, rejected deals {}", long.g.nodes.len(), agree / tot, t0.elapsed().as_secs_f64(), short.table_bytes() / 1024, short.rejected);
        assert!(agree / tot > 0.8, "agreement {}", agree / tot);
    }
}
