//! M4 stage 4 (Leduc): solving a subgame (Supp pp. 21–22, `docs/m4_design.md` §4).
//!
//! * [`Solver::Mccfr`]: Monte Carlo Linear CFR "just as it was for the blueprint" (M1's
//!   Algorithm 1 implementation with the Pluribus schedule scaled to the iteration budget),
//!   run on the subgame as a [`Game`].
//! * [`Solver::Vector`]: vector-based Linear CFR. One pass per traverser over the *public* tree
//!   with a vector over the 6 cards per player (reach down, counterfactual values up); the deal at
//!   the root is never sampled. Public chance (the board card) is sampled once per iteration and
//!   reweighted to stay unbiased (Johanson et al. 2011's public chance sampling; one board per
//!   iteration on one thread, D-SRCH-12). Linear CFR: iteration t's regrets and average-strategy
//!   contributions are weighted by t [38].
//! * Both return the **final iteration's** strategy (what Pluribus plays) and the **weighted
//!   average** (what σ, beliefs and freezing use) (p. 22; D-SRCH-11).
//! Frozen infosets are fixed by the subgame itself (chance nodes in the `Game` view; skipped
//! regret updates in the vector pass).

use crate::game::{Game, NodeKind, MAX_ACTIONS};
use crate::leduc::{key_for, Leduc, LeducState};
use crate::leduc_subgame::{public_key, replay, LeducSubgame, LEAF_TAG};
use crate::mccfr::{Mccfr, MccfrConfig};
use crate::rng::Rng;
use crate::table::{Policy, Table, TabularPolicy};
use std::collections::HashMap;

const C: usize = 6;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Solver {
    Mccfr { iters: u64, seed: u64 },
    Vector { iters: u64, seed: u64 },
}

pub struct Solution {
    /// final iteration: what the searcher plays
    pub play: TabularPolicy,
    /// weighted average: σ for beliefs and freezing
    pub avg: TabularPolicy,
}

pub fn solve<P: Policy>(sub: LeducSubgame<P>, solver: Solver) -> Solution {
    match solver {
        Solver::Mccfr { iters, seed } => {
            let cfg = MccfrConfig::pluribus_scaled(iters, sub.max_abs_utility(), seed);
            let mut m = Mccfr::new(sub, cfg);
            m.run(iters);
            Solution { play: m.current_policy(), avg: m.average_policy() }
        }
        Solver::Vector { iters, seed } => {
            let mut v = VectorCfr::new(&sub, seed);
            for t in 1..=iters {
                for i in 0..2 {
                    let root = v.root;
                    let reach = v.reach0;
                    v.walk(&root, &reach, i, t as f64);
                }
            }
            Solution { play: v.table.current_policy(), avg: v.table.average_policy() }
        }
    }
}

struct VectorCfr<'s, 'a, P: Policy> {
    sub: &'s LeducSubgame<'a, P>,
    table: Table,
    root: LeducState,
    reach0: [[f64; C]; 2],
    /// normaliser of the root deal distribution, so values match the subgame's utilities
    z: f64,
    /// leaf public state → value for player 0 of every (c0, c1, k0, k1)
    leaf_values: HashMap<u64, Vec<f64>>,
    rng: Rng,
}

impl<'s, 'a, P: Policy> VectorCfr<'s, 'a, P> {
    fn new(sub: &'s LeducSubgame<'a, P>, seed: u64) -> Self {
        let rb = replay(&sub.public, 0, 1).board;
        let board = (rb < C as u8).then_some(rb);
        let blocked = |c: usize| Some(c as u8) == board;
        let mut reach0 = [[0.0; C]; 2];
        for p in 0..2 {
            for c in 0..C {
                reach0[p][c] = if blocked(c) { 0.0 } else { sub.ranges[p][c] };
            }
        }
        let z: f64 = (0..C).flat_map(|a| (0..C).map(move |b| (a, b))).filter(|(a, b)| a != b).map(|(a, b)| reach0[0][a] * reach0[1][b]).sum();
        assert!(z > 0.0, "no card-compatible deal has positive belief");
        // placeholder hole cards: betting structure does not depend on them
        let (pa, pb) = {
            let free: Vec<u8> = (0..C as u8).filter(|&c| !blocked(c as usize)).collect();
            (free[0], free[1])
        };
        VectorCfr { sub, table: Table::new(), root: replay(&sub.public, pa, pb), reach0, z, leaf_values: HashMap::new(), rng: Rng::new(seed) }
    }

    /// Counterfactual values of traverser `i` per own card (opponent reach and chance included).
    fn walk(&mut self, s: &LeducState, reach: &[[f64; C]; 2], i: usize, t: f64) -> [f64; C] {
        let g = Leduc;
        match g.kind(s) {
            NodeKind::Terminal => {
                let o = 1 - i;
                let mut v = [0.0; C];
                for (ci, vi) in v.iter_mut().enumerate() {
                    if s.board as usize == ci {
                        continue;
                    }
                    for co in 0..C {
                        if co == ci || reach[o][co] == 0.0 {
                            continue;
                        }
                        let mut st = *s;
                        st.cards[i] = ci as u8;
                        st.cards[o] = co as u8;
                        *vi += reach[o][co] * g.utility(&st, i);
                    }
                    *vi /= self.z;
                }
                v
            }
            NodeKind::Chance if self.sub.depth_limit && s.round > self.sub.root_round => self.leaf(s, reach, i, t),
            NodeKind::Chance => {
                // public chance sampling: board b with q = 1/6; P(b | hands) = 1/4 if compatible
                let b = (self.rng.next_u64() % C as u64) as usize;
                let mut r2 = *reach;
                r2[0][b] = 0.0;
                r2[1][b] = 0.0;
                let mut v = self.walk(&g.apply(s, b as u8), &r2, i, t);
                for (c, x) in v.iter_mut().enumerate() {
                    *x = if c == b { 0.0 } else { *x * 1.5 };
                }
                v
            }
            NodeKind::Player(p) => {
                let acts = self.sub.betting_actions(s);
                let n = acts.len();
                let mut sig = [[0.0; MAX_ACTIONS]; C];
                let mut keys = [0u64; C];
                let mut frozen = [false; C];
                for c in 0..C {
                    let mut st = *s;
                    st.cards[p] = c as u8;
                    keys[c] = g.infoset_key(&st);
                    match self.sub.frozen.get(&keys[c]) {
                        Some(f) => {
                            sig[c][..n].copy_from_slice(f);
                            frozen[c] = true;
                        }
                        None => self.table.strategy(keys[c], n, &mut sig[c]),
                    }
                }
                let mut v = [0.0; C];
                if p == i {
                    let mut va = [[0.0; C]; MAX_ACTIONS];
                    for a in 0..n {
                        va[a] = self.walk(&g.apply(s, acts.as_slice()[a]), reach, i, t);
                        for c in 0..C {
                            v[c] += sig[c][a] * va[a][c];
                        }
                    }
                    for c in (0..C).filter(|&c| !frozen[c] && s.board as usize != c) {
                        let inf = self.table.get_or_insert(keys[c], n);
                        for a in 0..n {
                            inf.regret[a] += t * (va[a][c] - v[c]);
                            inf.strat_sum[a] += t * reach[i][c] * sig[c][a];
                        }
                    }
                } else {
                    for a in 0..n {
                        let mut r2 = *reach;
                        for c in 0..C {
                            r2[p][c] *= sig[c][a];
                        }
                        let va = self.walk(&g.apply(s, acts.as_slice()[a]), &r2, i, t);
                        for c in 0..C {
                            v[c] += va[c];
                        }
                    }
                }
                v
            }
        }
    }

    /// Leaf: each player picks a continuation strategy per own card, without seeing the other's
    /// choice (stage 3); values from the exact continuation table.
    fn leaf(&mut self, s: &LeducState, reach: &[[f64; C]; 2], i: usize, t: f64) -> [f64; C] {
        let nc = self.sub.n_cont;
        let pk = public_key(s);
        if !self.leaf_values.contains_key(&pk) {
            let mut u = vec![0.0; C * C * nc * nc];
            for c0 in 0..C {
                for c1 in (0..C).filter(|&c| c != c0) {
                    let mut st = *s;
                    st.cards = [c0 as u8, c1 as u8];
                    for k0 in 0..nc {
                        for k1 in 0..nc {
                            u[((c0 * C + c1) * nc + k0) * nc + k1] = self.sub.cont_value(&st, [k0 as u8, k1 as u8]);
                        }
                    }
                }
            }
            self.leaf_values.insert(pk, u);
        }
        let u = &self.leaf_values[&pk];
        let o = 1 - i;
        // strategies over continuations, per player and card
        let mut sig = [[[0.0; MAX_ACTIONS]; C]; 2];
        let mut keys = [[0u64; C]; 2];
        for p in 0..2 {
            for c in 0..C {
                let mut st = *s;
                st.cards[p] = c as u8;
                keys[p][c] = LEAF_TAG | key_for(&st, p) << 1 | p as u64;
                if nc == 1 {
                    sig[p][c][0] = 1.0;
                } else {
                    self.table.strategy(keys[p][c], nc, &mut sig[p][c]);
                }
            }
        }
        let val = |ci: usize, co: usize, ki: usize, ko: usize| {
            let (c0, c1, k0, k1) = if i == 0 { (ci, co, ki, ko) } else { (co, ci, ko, ki) };
            let v0 = u[((c0 * C + c1) * nc + k0) * nc + k1];
            if i == 0 {
                v0
            } else {
                -v0
            }
        };
        let mut v = [0.0; C];
        let mut vk = [[0.0; MAX_ACTIONS]; C];
        for ci in 0..C {
            for ki in 0..nc {
                let mut x = 0.0;
                for co in (0..C).filter(|&co| co != ci && reach[o][co] > 0.0) {
                    for ko in 0..nc {
                        x += reach[o][co] * sig[o][co][ko] * val(ci, co, ki, ko);
                    }
                }
                vk[ci][ki] = x / self.z;
                v[ci] += sig[i][ci][ki] * vk[ci][ki];
            }
        }
        if nc > 1 {
            for ci in 0..C {
                let inf = self.table.get_or_insert(keys[i][ci], nc);
                for k in 0..nc {
                    inf.regret[k] += t * (vk[ci][k] - v[ci]);
                    inf.strat_sum[k] += t * reach[i][ci] * sig[i][ci][k];
                }
            }
        }
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::br::{expected_values, exploitability};
    use crate::leduc::{CALL, RAISE};
    use crate::table::UniformPolicy;

    fn uniform() -> [Vec<f64>; 2] {
        [vec![1.0 / 6.0; 6], vec![1.0 / 6.0; 6]]
    }

    #[test]
    fn vector_linear_cfr_solves_full_leduc() {
        // No depth limit from the start of the hand: the subgame is Leduc, with the board card
        // sampled once per iteration (public chance sampling). Measured: exploitability 0.142 /
        // 0.097 / 0.036 at 2k / 8k / 32k iterations (sampling 1 of 6 boards is noisy); with every
        // board enumerated instead, 0.006 at 2k, so the vector pass itself is exact. The
        // validation subgames contain no public chance (round-1 leaves come before the board).
        let uni = UniformPolicy;
        let mut e = Vec::new();
        for iters in [2000, 16_000] {
            let mut sub = LeducSubgame::new(vec![], uniform(), &uni, 1, None);
            sub.depth_limit = false;
            let sol = solve(sub, Solver::Vector { iters, seed: 1 });
            let v = expected_values(&Leduc, &sol.avg)[0];
            e.push(exploitability(&Leduc, &sol.avg));
            eprintln!("vector Linear CFR, full Leduc, {iters} iterations: exploitability {:.5}, value {v:.5}", e[e.len() - 1]);
            assert!((v + 0.0856).abs() < 0.01, "value {v}");
        }
        assert!(e[1] < e[0] / 2.0 && e[1] < 0.08, "{e:?}");
    }

    #[test]
    fn both_solvers_solve_a_depth_limited_subgame() {
        let bp = UniformPolicy;
        for solver in [Solver::Vector { iters: 1000, seed: 1 }, Solver::Mccfr { iters: 50_000, seed: 1 }] {
            let sub = LeducSubgame::new(vec![], uniform(), &bp, 4, Some(3));
            let sol = solve(sub, solver);
            let sub = LeducSubgame::new(vec![], uniform(), &bp, 4, Some(3));
            let e = exploitability(&sub, &sol.avg);
            eprintln!("{solver:?}: exploitability in the depth-limited subgame {e:.5}");
            assert!(e < 0.03, "{solver:?}: {e}");
        }
    }

    #[test]
    fn frozen_infosets_stay_fixed_and_round1_roots_work() {
        // round-1 root after raise-call, board card 2; player 0's K (cards 4/5) frozen to "check"
        let bp = UniformPolicy;
        let public = vec![RAISE, CALL, 2];
        let (ranges, _) = crate::leduc_subgame::leduc_beliefs(&public, &bp);
        let mut s = replay(&public, 4, 0);
        s.cards[0] = 4;
        let key = Leduc.infoset_key(&s);
        for solver in [Solver::Vector { iters: 1000, seed: 2 }, Solver::Mccfr { iters: 20_000, seed: 2 }] {
            let mut sub = LeducSubgame::new(public.clone(), ranges.clone(), &bp, 4, None);
            sub.frozen.insert(key, vec![1.0, 0.0]);
            let sol = solve(sub, solver);
            assert!(!sol.avg.map.contains_key(&key) && !sol.play.map.contains_key(&key), "{solver:?} updated a frozen infoset");
            let mut sub = LeducSubgame::new(public.clone(), ranges.clone(), &bp, 4, None);
            sub.frozen.insert(key, vec![1.0, 0.0]);
            let e = exploitability(&sub, &sol.avg);
            eprintln!("{solver:?}: round-1 subgame with a frozen infoset, exploitability {e:.5}");
            assert!(e < 0.05, "{solver:?}: {e}");
        }
    }
}
