//! Depth-limited Leduc subgame (M4 stages 1–3, `docs/m4_design.md`): the toy-game counterpart of
//! the NLHE search, tested against exact answers and later used to validate the search.
//!
//! * Beliefs ([`leduc_beliefs`]): one range over the 6 cards per player, uniform at the start,
//!   updated by Bayes' rule from σ for every public action and board card (Supp p. 21).
//! * Root: a chance node over deals (c0, c1) with weight r0[c0]·r1[c1] for card-compatible deals
//!   (the paper's π^σ(h) / Σ π^σ(h′), Supp p. 20; D-SRCH-1), dealt as c0 then c1 | c0 to stay
//!   within `MAX_CHANCE`. Then the real game from the public history up to the start of the
//!   current round.
//! * Depth limit (Supp p. 21, round-1 rule): a root on round 0 has leaves at the chance node
//!   before round 1; a root on round 1 runs to the end of the game.
//! * Leaves (Supp pp. 23–24): each player chooses one of `n_cont` continuation strategies
//!   (1 = blueprint only, 4 = Pluribus' set), in an infoset that holds only its own card and the
//!   public history, so it does not see the other's choice. The leaf value is the exact expected
//!   value of the rest of the game with player p following strategy k_p, either the exact biased
//!   distribution or its compressed form (one sampled action per infoset, `compress = Some(seed)`).
//!
//! Because this implements [`Game`], M1's CFR, MCCFR and exact best response run on it unchanged.

use crate::game::{Action, ActionList, ChanceList, Game, NodeKind, MAX_ACTIONS};
use crate::leduc::{key_for, Leduc, LeducState, CALL, FOLD, RAISE};
use crate::search::{bayes, bias, sample_fixed, ActKind, N_CONT};
use crate::table::Policy;
use std::collections::{HashMap, HashSet};

/// Set on the infoset keys of leaf choices (Leduc keys are below 4^13).
pub const LEAF_TAG: u64 = 1 << 62;
const CARDS: usize = 6;

#[derive(Clone, Debug)]
pub enum SubState {
    /// chance: player 0's card
    Deal0,
    /// chance: player 1's card, given player 0's
    Deal1(u8),
    Play(LeducState),
    /// leaf: player `next` chooses a continuation strategy; `next == 2`: all chosen (terminal)
    Leaf { s: LeducState, k: [u8; 2], next: u8 },
}

pub struct LeducSubgame<'a, P: Policy> {
    /// Public actions from the start of the game to the subgame root (bets and the board card).
    pub public: Vec<Action>,
    /// Beliefs at the root, one range over the 6 cards per player.
    pub ranges: [Vec<f64>; 2],
    pub blueprint: &'a P,
    /// Continuation strategies offered at a leaf: 1 (blueprint only) or 4.
    pub n_cont: usize,
    /// `Some(seed)`: compressed continuation strategies; `None`: exact biased distributions.
    pub compress: Option<u64>,
    /// Paper rule by default (leaves before round 1 when the root is on round 0).
    pub depth_limit: bool,
    pub root_round: u8,
    /// Algorithm 2's frozen infosets: key → fixed action probabilities. The node becomes a chance
    /// node for that hand, so every solver keeps it fixed (stage 5).
    pub frozen: HashMap<u64, Vec<f64>>,
    /// Search action abstraction without re-raises (to exercise off-tree actions on Leduc,
    /// whose only sizes are fold/call/raise): a raise facing a raise is legal only at the
    /// public states in `added` (Algorithm 2's AddAction).
    pub no_reraise: bool,
    pub added: HashSet<u64>,
    board: Option<u8>,
}

fn kind_of(a: Action) -> ActKind {
    match a {
        FOLD => ActKind::Fold,
        CALL => ActKind::Call,
        RAISE => ActKind::Raise,
        _ => unreachable!(),
    }
}

/// The Leduc state after dealing (c0, c1) and replaying `public`.
pub fn replay(public: &[Action], c0: u8, c1: u8) -> LeducState {
    let g = Leduc;
    let s = g.apply(&g.apply(&g.root(), c0), c1);
    public.iter().fold(s, |s, &a| g.apply(&s, a))
}

/// Beliefs after the public history `public` when both players follow `sigma` (Bayes' rule per
/// action and board card, starting from 1/6 each). Returns the ranges and the probability of
/// each observation in order (0 marks one that σ rules out, D-SRCH-2).
pub fn leduc_beliefs<P: Policy>(public: &[Action], sigma: &P) -> ([Vec<f64>; 2], Vec<f64>) {
    let mut r = [vec![1.0 / CARDS as f64; CARDS], vec![1.0 / CARDS as f64; CARDS]];
    let obs = update_beliefs(&mut r, &[], public, sigma);
    (r, obs)
}

/// Bayes updates of `r` for the public `actions` that follow the public history `prefix`, with
/// both players following `sigma` (used per round by the nested search, stage 5). Returns each
/// observation's probability (0 = ruled out by σ, prior kept, D-SRCH-2).
pub fn update_beliefs<P: Policy>(r: &mut [Vec<f64>; 2], prefix: &[Action], actions: &[Action], sigma: &P) -> Vec<f64> {
    let g = Leduc;
    let mut obs = Vec::new();
    // placeholder hole cards: infoset keys only read the actor's card, which we vary below
    let mut s = replay(prefix, 0, 1);
    for &a in actions {
        let mut lik = [0.0; CARDS];
        match g.kind(&s) {
            NodeKind::Chance => {
                for (c, l) in lik.iter_mut().enumerate() {
                    *l = if c as u8 == a { 0.0 } else { 1.0 };
                }
                obs.push(bayes(&mut r[0], &lik) * bayes(&mut r[1], &lik));
            }
            NodeKind::Player(p) => {
                let acts = g.legal_actions(&s);
                let idx = acts.as_slice().iter().position(|&x| x == a).expect("illegal public action");
                let mut probs = [0.0; MAX_ACTIONS];
                for (c, l) in lik.iter_mut().enumerate() {
                    let mut t = s;
                    t.cards[p] = c as u8;
                    sigma.action_probs(g.infoset_key(&t), acts.len(), &mut probs);
                    *l = probs[idx];
                }
                obs.push(bayes(&mut r[p], &lik));
            }
            NodeKind::Terminal => panic!("public history goes past the end of the hand"),
        }
        s = g.apply(&s, a);
    }
    obs
}

/// Key of the public state at `s` (betting history and board rank), shared by both players.
pub fn public_key(s: &LeducState) -> u64 {
    key_for(s, 0) >> 2
}

impl<'a, P: Policy> LeducSubgame<'a, P> {
    pub fn new(public: Vec<Action>, ranges: [Vec<f64>; 2], blueprint: &'a P, n_cont: usize, compress: Option<u64>) -> Self {
        assert!(n_cont == 1 || n_cont == N_CONT);
        let s = replay(&public, 0, 1);
        assert!(
            matches!(Leduc.kind(&s), NodeKind::Player(0)) && s.round_actions == 0,
            "the subgame root must be the start of a betting round"
        );
        let board = (s.round == 1).then_some(s.board);
        LeducSubgame {
            public,
            ranges,
            blueprint,
            n_cont,
            compress,
            depth_limit: s.round == 0,
            root_round: s.round,
            frozen: HashMap::new(),
            no_reraise: false,
            added: HashSet::new(),
            board,
        }
    }

    /// Betting actions of the search abstraction at a Leduc decision state.
    pub fn betting_actions(&self, s: &LeducState) -> ActionList {
        let all = Leduc.legal_actions(s);
        if !self.no_reraise || s.raises == 0 || self.added.contains(&public_key(s)) {
            return all;
        }
        ActionList::from_slice(&all.as_slice().iter().copied().filter(|&a| a != RAISE).collect::<Vec<_>>())
    }

    /// Frozen probabilities at a Leduc decision state, if its infoset is frozen.
    pub fn frozen_at(&self, s: &LeducState) -> Option<&Vec<f64>> {
        if self.frozen.is_empty() {
            return None;
        }
        self.frozen.get(&Leduc.infoset_key(s))
    }

    /// Weight of the deal (c0, c1) at the root, before normalisation.
    fn weight(&self, c0: usize, c1: usize) -> f64 {
        let blocked = |c: usize| Some(c as u8) == self.board;
        if c0 == c1 || blocked(c0) || blocked(c1) {
            return 0.0;
        }
        self.ranges[0][c0] * self.ranges[1][c1]
    }

    fn wrap(&self, s: LeducState) -> SubState {
        if self.depth_limit && Leduc.kind(&s) == NodeKind::Chance && s.round > self.root_round {
            let next = if self.n_cont == 1 { 2 } else { 0 };
            SubState::Leaf { s, k: [0; 2], next }
        } else {
            SubState::Play(s)
        }
    }

    /// Exact value for player 0 of the rest of the game from `s`, player p following
    /// continuation strategy k[p] (compressed if `self.compress` is set).
    pub fn cont_value(&self, s: &LeducState, k: [u8; 2]) -> f64 {
        let g = Leduc;
        match g.kind(s) {
            NodeKind::Terminal => g.utility(s, 0),
            NodeKind::Chance => {
                let c = g.chance_outcomes(s);
                c.actions().iter().zip(c.probs()).map(|(&a, &p)| p * self.cont_value(&g.apply(s, a), k)).sum()
            }
            NodeKind::Player(q) => {
                let acts = g.legal_actions(s);
                let n = acts.len();
                let key = g.infoset_key(s);
                let mut probs = [0.0; MAX_ACTIONS];
                self.blueprint.action_probs(key, n, &mut probs);
                let kinds: Vec<ActKind> = acts.as_slice().iter().map(|&a| kind_of(a)).collect();
                let mut b = [0.0; MAX_ACTIONS];
                bias(&probs[..n], &kinds, k[q] as usize, &mut b);
                match self.compress {
                    Some(seed) => {
                        let i = sample_fixed(seed, key, k[q] as usize, &b[..n]);
                        self.cont_value(&g.apply(s, acts.as_slice()[i]), k)
                    }
                    None => (0..n)
                        .filter(|&i| b[i] > 0.0)
                        .map(|i| b[i] * self.cont_value(&g.apply(s, acts.as_slice()[i]), k))
                        .sum(),
                }
            }
        }
    }
}

impl<'a, P: Policy> Game for LeducSubgame<'a, P> {
    type State = SubState;

    fn name(&self) -> &'static str {
        "leduc-subgame"
    }
    fn num_players(&self) -> usize {
        2
    }
    fn num_rounds(&self) -> u8 {
        2
    }
    fn max_abs_utility(&self) -> f64 {
        13.0
    }

    fn root(&self) -> SubState {
        SubState::Deal0
    }

    fn kind(&self, s: &SubState) -> NodeKind {
        match s {
            SubState::Deal0 | SubState::Deal1(_) => NodeKind::Chance,
            SubState::Play(st) => match Leduc.kind(st) {
                NodeKind::Player(_) if self.frozen_at(st).is_some() => NodeKind::Chance,
                k => k,
            },
            SubState::Leaf { next, .. } if *next < 2 => NodeKind::Player(*next as usize),
            SubState::Leaf { .. } => NodeKind::Terminal,
        }
    }

    fn utility(&self, s: &SubState, player: usize) -> f64 {
        let v0 = match s {
            SubState::Play(st) => Leduc.utility(st, 0),
            SubState::Leaf { s, k, next: 2 } => self.cont_value(s, *k),
            _ => panic!("utility at a non-terminal state"),
        };
        if player == 0 {
            v0
        } else {
            -v0
        }
    }

    fn chance_outcomes(&self, s: &SubState) -> ChanceList {
        let mut c = ChanceList::new();
        match s {
            SubState::Deal0 => {
                let m: Vec<f64> = (0..CARDS).map(|c0| (0..CARDS).map(|c1| self.weight(c0, c1)).sum()).collect();
                let total: f64 = m.iter().sum();
                assert!(total > 0.0, "no card-compatible deal has positive belief");
                for (c0, &w) in m.iter().enumerate() {
                    if w > 0.0 {
                        c.push(c0 as u8, w / total);
                    }
                }
            }
            SubState::Deal1(c0) => {
                let total: f64 = (0..CARDS).map(|c1| self.weight(*c0 as usize, c1)).sum();
                for c1 in 0..CARDS {
                    let w = self.weight(*c0 as usize, c1);
                    if w > 0.0 {
                        c.push(c1 as u8, w / total);
                    }
                }
            }
            SubState::Play(st) => match self.frozen_at(st) {
                Some(f) => {
                    for (&a, &p) in self.betting_actions(st).as_slice().iter().zip(f) {
                        if p > 0.0 {
                            c.push(a, p);
                        }
                    }
                }
                None => return Leduc.chance_outcomes(st),
            },
            SubState::Leaf { .. } => panic!("a leaf is not a chance node"),
        }
        c
    }

    fn legal_actions(&self, s: &SubState) -> ActionList {
        match s {
            SubState::Play(st) => self.betting_actions(st),
            SubState::Leaf { .. } => ActionList::from_slice(&[0, 1, 2, 3][..self.n_cont]),
            _ => panic!("no player acts at a chance node"),
        }
    }

    fn apply(&self, s: &SubState, a: Action) -> SubState {
        match s {
            SubState::Deal0 => SubState::Deal1(a),
            SubState::Deal1(c0) => self.wrap(replay(&self.public, *c0, a)),
            SubState::Play(st) => self.wrap(Leduc.apply(st, a)),
            SubState::Leaf { s, k, next } => {
                let mut k = *k;
                k[*next as usize] = a;
                SubState::Leaf { s: *s, k, next: next + 1 }
            }
        }
    }

    fn infoset_key(&self, s: &SubState) -> u64 {
        match s {
            SubState::Play(st) => Leduc.infoset_key(st),
            // own card + public leaf history only: the other player's choice is not observed
            SubState::Leaf { s, next, .. } => LEAF_TAG | key_for(s, *next as usize) << 1 | *next as u64,
            _ => panic!("no infoset at a chance node"),
        }
    }

    fn round(&self, s: &SubState) -> u8 {
        match s {
            SubState::Deal0 | SubState::Deal1(_) => self.root_round,
            SubState::Play(st) | SubState::Leaf { s: st, .. } => st.round,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::br::{expected_values, exploitability, nash_conv};
    use crate::cfr::Cfr;
    use crate::rng::Rng;
    use crate::table::{TabularPolicy, UniformPolicy};
    use std::collections::{HashMap, HashSet};

    /// Every Leduc infoset key with its number of actions.
    fn leduc_infosets() -> Vec<(u64, usize)> {
        fn walk(g: &Leduc, s: &LeducState, out: &mut HashMap<u64, usize>) {
            match g.kind(s) {
                NodeKind::Terminal => {}
                NodeKind::Chance => {
                    for &a in g.chance_outcomes(s).actions() {
                        walk(g, &g.apply(s, a), out);
                    }
                }
                NodeKind::Player(_) => {
                    let acts = g.legal_actions(s);
                    out.insert(g.infoset_key(s), acts.len());
                    for &a in acts.as_slice() {
                        walk(g, &g.apply(s, a), out);
                    }
                }
            }
        }
        let mut m = HashMap::new();
        walk(&Leduc, &Leduc.root(), &mut m);
        let mut v: Vec<_> = m.into_iter().collect();
        v.sort_unstable();
        v
    }

    fn random_policy(seed: u64, deterministic: bool) -> TabularPolicy {
        let mut rng = Rng::new(seed);
        let mut p = TabularPolicy::default();
        for (key, n) in leduc_infosets() {
            let mut v: Vec<f64> = (0..n).map(|_| 0.05 + rng.next_f64()).collect();
            if deterministic {
                let best = (0..n).fold(0, |b, i| if v[i] > v[b] { i } else { b });
                v = (0..n).map(|i| (i == best) as u8 as f64).collect();
            }
            let s: f64 = v.iter().sum();
            p.map.insert(key, v.into_iter().map(|x| x / s).collect());
        }
        p
    }

    fn round0(key: u64) -> bool {
        (key / 4) % 4 == 0 // no board rank in the key
    }

    /// `first` on round 0, `then` on round 1.
    fn combine(first: &TabularPolicy, then: &TabularPolicy) -> TabularPolicy {
        let mut p = TabularPolicy::default();
        for (&k, v) in &then.map {
            p.map.insert(k, if round0(k) { first.map[&k].clone() } else { v.clone() });
        }
        p
    }

    /// Wraps a policy; every leaf choice plays continuation `k` (or the inner policy if None).
    struct Choose<'a>(&'a TabularPolicy, Option<usize>);
    impl Policy for Choose<'_> {
        fn action_probs(&self, key: u64, n: usize, out: &mut [f64]) {
            match (key & LEAF_TAG != 0, self.1) {
                (true, Some(k)) => (0..n).for_each(|i| out[i] = (i == k) as u8 as f64),
                _ => self.0.action_probs(key, n, out),
            }
        }
    }

    fn uniform_ranges() -> [Vec<f64>; 2] {
        [vec![1.0 / 6.0; 6], vec![1.0 / 6.0; 6]]
    }

    fn close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() < tol, "{a} vs {b}");
    }

    /// Exact posterior over deals given the public history, by brute force over all histories.
    fn exact_posterior(public: &[Action], sigma: &TabularPolicy) -> [[f64; 6]; 6] {
        let g = Leduc;
        let mut w = [[0.0; 6]; 6];
        for c0 in 0..6u8 {
            for c1 in (0..6u8).filter(|&c| c != c0) {
                let mut s = g.apply(&g.apply(&g.root(), c0), c1);
                let mut p = 1.0 / 30.0;
                for &a in public {
                    match g.kind(&s) {
                        NodeKind::Chance => {
                            let c = g.chance_outcomes(&s);
                            p *= c.actions().iter().zip(c.probs()).find(|(&x, _)| x == a).map_or(0.0, |(_, &q)| q);
                        }
                        NodeKind::Player(_) => {
                            let acts = g.legal_actions(&s);
                            let mut probs = [0.0; MAX_ACTIONS];
                            sigma.action_probs(g.infoset_key(&s), acts.len(), &mut probs);
                            p *= probs[acts.as_slice().iter().position(|&x| x == a).unwrap()];
                        }
                        NodeKind::Terminal => unreachable!(),
                    }
                    if p == 0.0 {
                        break;
                    }
                    s = g.apply(&s, a);
                }
                w[c0 as usize][c1 as usize] = p;
            }
        }
        let total: f64 = w.iter().flatten().sum();
        w.iter_mut().flatten().for_each(|x| *x /= total);
        w
    }

    #[test]
    fn stage1_root_distribution_equals_the_brute_force_posterior() {
        let sigma = random_policy(11, false);
        // round-1 roots (board card 3 or 0) after different round-0 lines, and the game start
        for public in [vec![RAISE, CALL, 3], vec![CALL, RAISE, RAISE, CALL, 0], vec![CALL, CALL, 5], vec![]] {
            let (ranges, obs) = leduc_beliefs(&public, &sigma);
            assert!(obs.iter().all(|&o| o > 0.0));
            let sub = LeducSubgame::new(public.clone(), ranges, &sigma, 4, None);
            let exact = exact_posterior(&public, &sigma);
            let mut got = [[0.0; 6]; 6];
            let d0 = sub.chance_outcomes(&SubState::Deal0);
            for (&c0, &p0) in d0.actions().iter().zip(d0.probs()) {
                let d1 = sub.chance_outcomes(&SubState::Deal1(c0));
                for (&c1, &p1) in d1.actions().iter().zip(d1.probs()) {
                    got[c0 as usize][c1 as usize] = p0 * p1;
                }
            }
            for c0 in 0..6 {
                for c1 in 0..6 {
                    close(got[c0][c1], exact[c0][c1], 1e-12);
                }
            }
        }
    }

    #[test]
    fn stage1_beliefs_block_the_board_card_and_report_impossible_actions() {
        let (r, _) = leduc_beliefs(&[CALL, CALL, 4], &UniformPolicy);
        assert_eq!(r[0][4], 0.0);
        assert_eq!(r[1][4], 0.0);
        close(r[0].iter().sum(), 1.0, 1e-12);
        // σ in which player 0 always checks first: an opening bet is an impossible observation
        let mut sigma = random_policy(3, false);
        for c in 0..6u8 {
            let mut t = replay(&[], c, (c + 1) % 6);
            t.cards[0] = c;
            sigma.map.insert(Leduc.infoset_key(&t), vec![1.0, 0.0]); // [CALL, RAISE]
        }
        let (r, obs) = leduc_beliefs(&[RAISE], &sigma);
        assert_eq!(obs, vec![0.0]);
        assert_eq!(r[0], vec![1.0 / 6.0; 6]); // prior kept
    }

    #[test]
    fn stage2_without_depth_limit_the_subgame_is_leduc() {
        let uni = UniformPolicy;
        let mut sub = LeducSubgame::new(vec![], uniform_ranges(), &uni, 1, None);
        sub.depth_limit = false;
        // OpenSpiel reference for uniform Leduc, as in br.rs
        close(nash_conv(&sub, &UniformPolicy), 4.747222222222222, 1e-9);
        let pol = random_policy(5, false);
        let (a, b) = (expected_values(&sub, &pol), expected_values(&Leduc, &pol));
        close(a[0], b[0], 1e-12);
        close(nash_conv(&sub, &pol), nash_conv(&Leduc, &pol), 1e-9);
    }

    /// Walk the subgame; collect decision keys and leaf-choice keys.
    fn keys(sub: &LeducSubgame<impl Policy>) -> (HashSet<u64>, HashSet<u64>) {
        fn walk<P: Policy>(g: &LeducSubgame<P>, s: &SubState, dec: &mut HashSet<u64>, leaf: &mut HashSet<u64>) {
            match g.kind(s) {
                NodeKind::Terminal => {}
                NodeKind::Chance => {
                    for &a in g.chance_outcomes(s).actions() {
                        walk(g, &g.apply(s, a), dec, leaf);
                    }
                }
                NodeKind::Player(_) => {
                    let k = g.infoset_key(s);
                    if k & LEAF_TAG != 0 { leaf.insert(k) } else { dec.insert(k) };
                    for &a in g.legal_actions(s).as_slice() {
                        walk(g, &g.apply(s, a), dec, leaf);
                    }
                }
            }
        }
        let (mut d, mut l) = (HashSet::new(), HashSet::new());
        walk(sub, &sub.root(), &mut d, &mut l);
        (d, l)
    }

    #[test]
    fn stage2_and_3_depth_limited_structure() {
        let b = random_policy(1, false);
        let sub = LeducSubgame::new(vec![], uniform_ranges(), &b, 4, Some(9));
        let (dec, leaf) = keys(&sub);
        // decisions: exactly Leduc's round-0 infosets
        let r0: HashSet<u64> = leduc_infosets().into_iter().map(|(k, _)| k).filter(|&k| round0(k)).collect();
        assert_eq!(dec, r0);
        // leaf choices: 5 round-0 lines that reach round 1 (cc, crc, crrc, rc, rrc) x 3 ranks x 2
        // players; one infoset per (line, own rank, player), not per opponent card or choice
        assert_eq!(leaf.len(), 30);
        // a round-1 root has no depth limit: no leaves, Leduc's round-1 infosets after that line
        let (ranges, _) = leduc_beliefs(&[RAISE, CALL, 2], &b);
        let sub1 = LeducSubgame::new(vec![RAISE, CALL, 2], ranges, &b, 4, Some(9));
        let (dec1, leaf1) = keys(&sub1);
        assert!(leaf1.is_empty());
        assert!(!dec1.is_empty() && dec1.iter().all(|&k| !round0(k)));
    }

    #[test]
    fn stage3_blueprint_leaves_equal_the_full_game_value() {
        let b = random_policy(1, false);
        let pi = random_policy(2, false);
        let full = expected_values(&Leduc, &combine(&pi, &b))[0];
        // blueprint-only leaves, exact continuation: value = pi on round 0, blueprint after
        let sub = LeducSubgame::new(vec![], uniform_ranges(), &b, 1, None);
        close(expected_values(&sub, &pi)[0], full, 1e-12);
        // four continuations, both players choosing the blueprint: same value
        let sub4 = LeducSubgame::new(vec![], uniform_ranges(), &b, 4, None);
        close(expected_values(&sub4, &Choose(&pi, Some(0)))[0], full, 1e-12);
        // both choosing strategy k: value = round 1 played with the k-biased blueprint
        for k in 1..4 {
            let mut bk = TabularPolicy::default();
            let g = Leduc;
            // biased blueprint, built from each infoset's legal actions
            fn walk(g: &Leduc, s: &LeducState, b: &TabularPolicy, k: usize, out: &mut TabularPolicy) {
                match g.kind(s) {
                    NodeKind::Terminal => {}
                    NodeKind::Chance => g.chance_outcomes(s).actions().iter().for_each(|&a| walk(g, &g.apply(s, a), b, k, out)),
                    NodeKind::Player(_) => {
                        let acts = g.legal_actions(s);
                        let key = g.infoset_key(s);
                        let kinds: Vec<ActKind> = acts.as_slice().iter().map(|&a| kind_of(a)).collect();
                        let mut v = vec![0.0; acts.len()];
                        bias(&b.map[&key], &kinds, k, &mut v);
                        out.map.insert(key, v);
                        acts.as_slice().iter().for_each(|&a| walk(g, &g.apply(s, a), b, k, out));
                    }
                }
            }
            walk(&g, &g.root(), &b, k, &mut bk);
            let want = expected_values(&Leduc, &combine(&pi, &bk))[0];
            close(expected_values(&sub4, &Choose(&pi, Some(k)))[0], want, 1e-12);
        }
    }

    #[test]
    fn stage3_compression_is_exact_for_a_deterministic_blueprint() {
        let b = random_policy(4, true);
        let pi = random_policy(2, false);
        let exact = LeducSubgame::new(vec![], uniform_ranges(), &b, 4, None);
        let comp = LeducSubgame::new(vec![], uniform_ranges(), &b, 4, Some(123));
        for k in 0..4 {
            close(
                expected_values(&comp, &Choose(&pi, Some(k)))[0],
                expected_values(&exact, &Choose(&pi, Some(k)))[0],
                1e-12,
            );
        }
        // with a mixed blueprint, compression changes values but is reproducible
        let bm = random_policy(1, false);
        let c1 = LeducSubgame::new(vec![], uniform_ranges(), &bm, 4, Some(5));
        let c2 = LeducSubgame::new(vec![], uniform_ranges(), &bm, 4, Some(5));
        let e = LeducSubgame::new(vec![], uniform_ranges(), &bm, 4, None);
        let v = |g: &LeducSubgame<TabularPolicy>| expected_values(g, &Choose(&pi, Some(3)))[0];
        assert_eq!(v(&c1), v(&c2));
        assert!((v(&c1) - v(&e)).abs() > 1e-6);
    }

    #[test]
    fn stage3_cfr_solves_the_depth_limited_game() {
        // the depth-limited game with simultaneous leaf choices is a well-formed two-player
        // zero-sum game: CFR drives its exploitability toward 0.
        // Measured: 0.185 / 0.0227 / 0.0117 at 20 / 1000 / 4000 iterations, i.e. the 1/sqrt(T)
        // rate of vanilla CFR (full Leduc: 0.0097 at 3k, cfr.rs).
        let b = random_policy(1, false);
        let sub = LeducSubgame::new(vec![], uniform_ranges(), &b, 4, Some(7));
        let mut cfr = Cfr::new(sub);
        let mut e = Vec::new();
        for target in [20, 1000, 4000] {
            cfr.run(target - cfr.iteration);
            e.push(exploitability(&cfr.game, &cfr.average_policy()));
        }
        eprintln!("exploitability at 20/1000/4000 iterations: {e:?}");
        assert!(e[1] < e[0] / 5.0 && e[2] < e[1] && e[2] < 0.015, "{e:?}");
    }
}
