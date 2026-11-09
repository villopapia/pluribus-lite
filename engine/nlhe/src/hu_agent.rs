//! M4 stage 5 on NLHE (heads-up): a blueprint agent and an Algorithm 2 search agent that
//! searches the turn and river, plus one real hand between two agents (`docs/m4_design.md` §5,
//! `docs/m4_validation.md` NLHE section).
//!
//! * [`BpTracker`] follows the blueprint tree through the real hand. On-tree actions map exactly;
//!   an off-tree raise is mapped with **randomized pseudo-harmonic** translation [39] between the
//!   two neighbouring blueprint sizes, in the D-ACT-5 pot convention (D-SRCH-10). The blueprint's
//!   abstract raises are played as the same pot fraction of the real pot.
//! * [`SearchAgent`] plays the blueprint preflop and on the flop, then Algorithm 2: at each new
//!   round (turn, river) Bayes-update both ranges with σ for the finished round (blueprint, or the
//!   previous search's weighted average) and the new card, root a subgame at the round start
//!   (search sizes `search_root.cfg` / `search_later.cfg`), solve it with vector Linear CFR
//!   (`vsolve`), play the final iteration; freeze σ(I) (average) for our actual hand when we act;
//!   an opponent action missing from the subgame is added at its state (AddAction) and the
//!   subgame is searched again from the round root.
//! * [`play_hand`]: one hand; returns each seat's chips and the AIVAT-style chance correction
//!   (equity value function, D-EVAL-2).

use crate::belief::{action_likelihood, Beliefs};
use crate::betting::{Action, BState, Next, Rules};
use crate::blueprint::{Deal, SnapshotAverage, Storage, ThreadStats, Tree, MAX_A};
use crate::buckets::CardAbstraction;
use crate::config::AbstractionConfig;
use crate::equity::combo_index;
use crate::eval_lut::{LHand, Lut};
use crate::subgame::Subgame;
use crate::vsolve::VectorSolver;
use cfr::rng::Rng;

/// A trained blueprint: preflop average strategy, post-flop snapshot average (or current
/// strategy when no snapshot exists).
pub struct Blueprint<'a> {
    pub tree: &'a Tree,
    pub st: &'a Storage,
    pub snap: Option<&'a SnapshotAverage>,
    pub abs: &'a dyn CardAbstraction,
    pub cfg: &'a AbstractionConfig,
    pub rules: Rules,
}

impl Blueprint<'_> {
    pub fn policy(&self, node: u32, bucket: usize, out: &mut [f64; MAX_A]) {
        if self.tree.nodes[node as usize].street > 0 {
            if let Some(s) = self.snap {
                return s.policy(self.tree, node, bucket, out);
            }
        }
        self.st.policy(self.tree, node, bucket, out)
    }
}

fn pot_frac(rules: &Rules, s: &BState, to: u32) -> f64 {
    let max_bet = *s.street_bet[..rules.players].iter().max().unwrap();
    let p = s.to_act as usize;
    let to_call = max_bet - s.street_bet[p];
    (to as f64 - max_bet as f64) / (rules.pot(s) + to_call) as f64
}

fn all_in_to(rules: &Rules, s: &BState) -> u32 {
    let p = s.to_act as usize;
    s.street_bet[p] + rules.stack - s.total[p]
}

/// Position in the blueprint tree, kept in step with the real hand.
#[derive(Clone)]
pub struct BpTracker {
    pub node: u32,
    pub state: BState,
    /// the abstract hand ended or desynchronised (the agent then checks/calls)
    pub lost: bool,
}

impl BpTracker {
    pub fn new(rules: &Rules) -> Self {
        BpTracker { node: 0, state: rules.root(), lost: false }
    }

    /// Follow a real action taken at real state `real`.
    pub fn observe(&mut self, bp: &Blueprint, real: &BState, a: Action, rng: &mut Rng) -> bool {
        if self.lost {
            return false;
        }
        let acts = bp.rules.actions(&self.state, bp.cfg);
        let pick = |x: Action| acts.iter().position(|&y| y == x);
        let idx = match a {
            Action::Fold => pick(Action::Fold).or_else(|| pick(Action::Call)),
            Action::Call => pick(Action::Call),
            Action::RaiseTo(x) => pick(a).or_else(|| {
                let raises: Vec<(usize, f64, bool)> = acts
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &b)| match b {
                        Action::RaiseTo(t) => Some((i, pot_frac(&bp.rules, &self.state, t), t == all_in_to(&bp.rules, &self.state))),
                        _ => None,
                    })
                    .collect();
                if raises.is_empty() {
                    return pick(Action::Call);
                }
                if x == all_in_to(&bp.rules, real) {
                    if let Some(&(i, _, true)) = raises.iter().find(|r| r.2) {
                        return Some(i);
                    }
                }
                let f = pot_frac(&bp.rules, real, x);
                let below = raises.iter().filter(|r| r.1 <= f).last();
                let above = raises.iter().find(|r| r.1 > f);
                Some(match (below, above) {
                    (Some(b), None) => b.0,
                    (None, Some(c)) => c.0,
                    (Some(b), Some(c)) => {
                        // randomized pseudo-harmonic: P(A) = (B - x)(1 + A) / ((B - A)(1 + x))
                        let (aa, bb) = (b.1, c.1);
                        let p_a = (bb - f) * (1.0 + aa) / ((bb - aa) * (1.0 + f));
                        if rng.next_f64() < p_a {
                            b.0
                        } else {
                            c.0
                        }
                    }
                    (None, None) => unreachable!(),
                })
            }),
        };
        let Some(i) = idx else {
            self.lost = true;
            return false;
        };
        let on_tree = acts[i] == a;
        self.node = bp.tree.child(self.node, i);
        match bp.rules.apply(&self.state, acts[i]) {
            Next::Decision(n) => self.state = n,
            Next::Terminal(_) => self.lost = true,
        }
        on_tree
    }

    /// Advance by abstract action index `i` (our own choice: no re-translation).
    pub fn advance(&mut self, bp: &Blueprint, i: usize) {
        if self.lost {
            return;
        }
        let acts = bp.rules.actions(&self.state, bp.cfg);
        self.node = bp.tree.child(self.node, i);
        match bp.rules.apply(&self.state, acts[i]) {
            Next::Decision(n) => self.state = n,
            Next::Terminal(_) => self.lost = true,
        }
    }

    /// The blueprint's action for `hole` at real state `real` (board = cards dealt so far), and
    /// the abstract action index it came from (None if the tracker is lost).
    pub fn act(&self, bp: &Blueprint, real: &BState, hole: (u8, u8), board: &[u8], d: &mut Deal, rng: &mut Rng) -> (Action, Option<usize>) {
        let p = real.to_act as usize;
        if self.lost || self.state.to_act != real.to_act || self.state.street != real.street {
            return (Action::Call, None);
        }
        let nd = bp.tree.nodes[self.node as usize];
        let mut b5 = [0u8; 5];
        b5[..board.len()].copy_from_slice(board);
        let mut fill = (0..52u8).filter(|c| !board.contains(c) && *c != hole.0 && *c != hole.1);
        for x in b5.iter_mut().skip(board.len()) {
            *x = fill.next().unwrap(); // unused by buckets before the river
        }
        d.set(&[hole], b5);
        let mut ts = ThreadStats::default();
        let bucket = d.bucket(bp.abs, 0, nd.street as usize, &mut ts);
        let mut pr = [0.0; MAX_A];
        bp.policy(self.node, bucket, &mut pr);
        let acts = bp.rules.actions(&self.state, bp.cfg);
        let k = rng.sample_index(&pr[..acts.len()]);
        let max_bet = *real.street_bet[..bp.rules.players].iter().max().unwrap();
        let a = match acts[k] {
            Action::Fold if max_bet > real.street_bet[p] => Action::Fold,
            Action::Fold | Action::Call => Action::Call,
            Action::RaiseTo(t) => {
                let ai = all_in_to(&bp.rules, real);
                if t == all_in_to(&bp.rules, &self.state) {
                    return (Action::RaiseTo(ai), Some(k));
                }
                let f = pot_frac(&bp.rules, &self.state, t);
                let to_call = max_bet - real.street_bet[p];
                let min_to = max_bet + real.last_inc.max(bp.rules.bb);
                let to = (max_bet as f64 + f * (bp.rules.pot(real) + to_call) as f64).round() as u32;
                let to = to.max(min_to);
                if to >= ai || bp.rules.stack - real.total[p] <= to_call {
                    if bp.rules.stack - real.total[p] <= to_call {
                        Action::Call
                    } else {
                        Action::RaiseTo(ai)
                    }
                } else {
                    Action::RaiseTo(to)
                }
            }
        };
        (a, Some(k))
    }
}

/// What an agent sees: the real state before each action, and the board so far.
pub trait Agent {
    fn act(&mut self, s: &BState, board: &[u8]) -> Action;
    /// every action at the table, including our own, after it is taken
    fn observe(&mut self, before: &BState, a: Action, board: &[u8]);
    /// a new round starts with `board` (cards dealt so far) at state `s`
    fn new_round(&mut self, s: &BState, board: &[u8]);
}

pub struct BlueprintAgent<'a> {
    pub bp: &'a Blueprint<'a>,
    pub hole: (u8, u8),
    pub tr: BpTracker,
    pub rng: Rng,
    pub d: Deal,
    pub off_tree: u64,
    mine: Option<usize>,
}

impl<'a> BlueprintAgent<'a> {
    pub fn new(bp: &'a Blueprint<'a>, hole: (u8, u8), seed: u64) -> Self {
        BlueprintAgent { bp, hole, tr: BpTracker::new(&bp.rules), rng: Rng::new(seed), d: Deal::new(), off_tree: 0, mine: None }
    }
}

impl Agent for BlueprintAgent<'_> {
    fn act(&mut self, s: &BState, board: &[u8]) -> Action {
        let (a, i) = self.tr.act(self.bp, s, self.hole, board, &mut self.d, &mut self.rng);
        self.mine = i;
        a
    }
    fn observe(&mut self, before: &BState, a: Action, _board: &[u8]) {
        if let Some(i) = self.mine.take() {
            self.tr.advance(self.bp, i);
        } else if !self.tr.observe(self.bp, before, a, &mut self.rng) {
            self.off_tree += 1;
        }
    }
    fn new_round(&mut self, _s: &BState, _board: &[u8]) {}
}

#[derive(Clone, Debug)]
pub struct SearchSettings {
    pub turn_iters: u64,
    pub river_iters: u64,
    /// play the weighted average instead of the final iteration (diagnostic, not the paper)
    pub play_average: bool,
}

#[derive(Clone, Debug, Default)]
pub struct SearchStats {
    pub searches: u64,
    pub re_searches: u64,
    pub zero_prob_events: u64,
    /// exploitability (chips) of the played strategy within each river subgame, final / average
    pub river_expl: Vec<(f64, f64)>,
    pub search_secs: f64,
}

pub struct SearchAgent<'a> {
    pub bp: &'a Blueprint<'a>,
    pub root_cfg: &'a AbstractionConfig,
    pub later_cfg: &'a AbstractionConfig,
    pub set: SearchSettings,
    pub seat: usize,
    pub hole: (u8, u8),
    combo: usize,
    tr: BpTracker,
    rng: Rng,
    d: Deal,
    /// pre-turn history for the Bayes update: (actor, blueprint node, action index, board)
    pre: Vec<(usize, u32, usize, Vec<u8>)>,
    beliefs: Option<Beliefs>,
    root: Option<BState>,
    board: Vec<u8>,
    /// actions since the round root: (state before, action)
    round: Vec<(BState, Action)>,
    extra: Vec<(BState, Action)>,
    /// our frozen decisions this round: (number of round actions before it, σ average)
    frozen: Vec<(usize, Vec<f64>)>,
    solver: Option<VectorSolver>,
    mine: Option<usize>,
    pub stats: SearchStats,
    pub measure_expl: bool,
}

impl<'a> SearchAgent<'a> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(bp: &'a Blueprint<'a>, root_cfg: &'a AbstractionConfig, later_cfg: &'a AbstractionConfig, set: SearchSettings, seat: usize, hole: (u8, u8), seed: u64) -> Self {
        SearchAgent {
            bp,
            root_cfg,
            later_cfg,
            set,
            seat,
            hole,
            combo: combo_index(hole.0, hole.1),
            tr: BpTracker::new(&bp.rules),
            rng: Rng::new(seed),
            d: Deal::new(),
            pre: Vec::new(),
            beliefs: None,
            root: None,
            board: Vec::new(),
            round: Vec::new(),
            extra: Vec::new(),
            frozen: Vec::new(),
            solver: None,
            mine: None,
            stats: SearchStats::default(),
            measure_expl: false,
        }
    }

    /// Subgame node reached by the first `k` actions of this round.
    fn node_after(&self, k: usize) -> Option<u32> {
        let s = self.solver.as_ref()?;
        let mut node = 0u32;
        for (_, a) in &self.round[..k] {
            let nd = s.g.nodes[node as usize];
            let i = (0..nd.n as usize).find(|&i| s.g.action(node, i) == Some(*a))?;
            node = s.g.child(node, i);
        }
        Some(node)
    }

    fn search(&mut self) {
        let t0 = std::time::Instant::now();
        let root = self.root.unwrap();
        let g = Subgame::build_extra(&self.bp.rules, self.root_cfg, self.later_cfg, root, self.extra.clone());
        let ranges = self.beliefs.as_ref().unwrap().ranges.clone();
        let mut s = VectorSolver::new(g, self.bp.rules, &self.board, [ranges[0].clone(), ranges[1].clone()]);
        self.solver = Some(s);
        // frozen: our earlier decisions this round, at their nodes in the new subgame
        let mut fz = Vec::new();
        for (k, p) in &self.frozen {
            if let Some(n) = self.node_after(*k) {
                fz.push(((n, self.combo as u16), p.clone()));
            }
        }
        s = self.solver.take().unwrap();
        s.frozen = fz.into_iter().collect();
        let iters = if self.board.len() == 4 { self.set.turn_iters } else { self.set.river_iters };
        s.run(iters);
        if self.measure_expl && self.board.len() == 5 {
            let e_fin = s.exploitability(false);
            let e_avg = s.exploitability(true);
            self.stats.river_expl.push((e_fin, e_avg));
        }
        self.solver = Some(s);
        self.stats.searches += 1;
        self.stats.search_secs += t0.elapsed().as_secs_f64();
        if std::env::var_os("HU_DEBUG").is_some() { let s = self.solver.as_ref().unwrap(); eprintln!("search board {} nodes {} extra {} iters {} root {:?} {:.1}s", self.board.len(), s.g.nodes.len(), self.extra.len(), s.iterations, root.total, t0.elapsed().as_secs_f64()); }
    }

    /// Likelihood of each hand taking action index `ai` at subgame node `node` (σ = average).
    fn search_likelihood(&self, node: u32, ai: usize) -> Vec<f64> {
        let s = self.solver.as_ref().unwrap();
        let n = s.g.nodes[node as usize].n as usize;
        let all = s.strategy_all(node, &[], true);
        (0..crate::equity::COMBOS).map(|h| all[h * n + ai]).collect()
    }
}

impl Agent for SearchAgent<'_> {
    fn act(&mut self, s: &BState, board: &[u8]) -> Action {
        if s.street < 2 || self.solver.is_none() {
            let (a, i) = self.tr.act(self.bp, s, self.hole, board, &mut self.d, &mut self.rng);
            self.mine = i;
            return a;
        }
        let k = self.round.len();
        let node = self.node_after(k).expect("our state is in the subgame");
        let sol = self.solver.as_ref().unwrap();
        let play = sol.strategy(node, &[], self.combo, self.set.play_average);
        let avg = sol.strategy(node, &[], self.combo, true);
        let i = self.rng.sample_index(&play);
        let a = sol.g.action(node, i).unwrap();
        self.frozen.push((k, avg));
        a
    }

    fn observe(&mut self, before: &BState, a: Action, board: &[u8]) {
        if before.street < 2 {
            // blueprint rounds: remember the blueprint node for the Bayes update
            let node = self.tr.node;
            let acts = self.bp.rules.actions(&self.tr.state, self.bp.cfg);
            let idx = match self.mine.take() {
                Some(i) => {
                    self.tr.advance(self.bp, i);
                    Some(i)
                }
                None => {
                    let idx = acts.iter().position(|&x| x == a);
                    self.tr.observe(self.bp, before, a, &mut self.rng);
                    idx
                }
            };
            if let Some(i) = idx {
                self.pre.push((before.to_act as usize, node, i, board.to_vec()));
            }
            return;
        }
        if self.solver.is_none() {
            return;
        }
        let k = self.round.len();
        if before.to_act as usize != self.seat {
            // OpponentTurn: an action missing from the subgame is added, then search again
            let node = self.node_after(k);
            let present = node.is_some_and(|n| {
                let s = self.solver.as_ref().unwrap();
                (0..s.g.nodes[n as usize].n as usize).any(|i| s.g.action(n, i) == Some(a))
            });
            if !present {
                self.extra.push((*before, a));
                self.round.push((*before, a));
                self.stats.re_searches += 1;
                self.search();
                return;
            }
        }
        self.round.push((*before, a));
    }

    fn new_round(&mut self, s: &BState, board: &[u8]) {
        if s.street < 2 {
            return;
        }
        // CheckNewRound: Bayes update for the finished round, then search from the new root
        let mut bel = match self.beliefs.take() {
            Some(b) => b,
            None => {
                let mut b = Beliefs::uniform(2);
                let pol = |n: u32, bk: usize, out: &mut [f64; MAX_A]| self.bp.policy(n, bk, out);
                for (p, node, ai, bd) in &self.pre {
                    let lik = action_likelihood(self.bp.tree, self.bp.abs, &pol, *node, *ai, bd);
                    b.observe_action(*p, &lik);
                }
                b
            }
        };
        if self.solver.is_some() && !self.round.is_empty() {
            for k in 0..self.round.len() {
                let (st, a) = self.round[k];
                if let Some(node) = self.node_after(k) {
                    let s = self.solver.as_ref().unwrap();
                    let ai = (0..s.g.nodes[node as usize].n as usize).find(|&i| s.g.action(node, i) == Some(a)).unwrap();
                    let lik = self.search_likelihood(node, ai);
                    bel.observe_action(st.to_act as usize, &lik);
                }
            }
        }
        bel.observe_cards(board);
        self.stats.zero_prob_events = bel.zero_prob_events;
        self.beliefs = Some(bel);
        self.root = Some(*s);
        self.board = board.to_vec();
        self.round.clear();
        self.extra.clear();
        self.frozen.clear();
        self.solver = None;
        let can_act = (0..2).filter(|&p| s.folded & (1 << p) == 0 && s.allin & (1 << p) == 0).count();
        if can_act >= 2 {
            self.search();
        }
    }
}

/// Showdown value for `me` at contributions `total`, averaged over the remaining board cards.
pub fn equity_value(holes: [(u8, u8); 2], board: &[u8], total: [u32; 2], me: usize) -> f64 {
    let lut = Lut::get();
    let used: Vec<u8> = board.iter().copied().chain([holes[0].0, holes[0].1, holes[1].0, holes[1].1]).collect();
    let rest: Vec<u8> = (0..52u8).filter(|c| !used.contains(c)).collect();
    let need = 5 - board.len();
    let (mut sum, mut n) = (0f64, 0u64);
    let mut idx: Vec<usize> = (0..need).collect();
    loop {
        let mut b = board.to_vec();
        b.extend(idx.iter().map(|&i| rest[i]));
        let bh = LHand::of(&b);
        let r0 = lut.class(bh.add(holes[0].0).add(holes[0].1));
        let r1 = lut.class(bh.add(holes[1].0).add(holes[1].1));
        let o = 1 - me;
        let (rm, ro) = if me == 0 { (r0, r1) } else { (r1, r0) };
        sum += match rm.cmp(&ro) {
            std::cmp::Ordering::Greater => total[o] as f64,
            std::cmp::Ordering::Less => -(total[me] as f64),
            std::cmp::Ordering::Equal => 0.0,
        };
        n += 1;
        // next combination of `need` indices out of rest.len()
        let mut k = need;
        loop {
            if k == 0 {
                return sum / n as f64;
            }
            k -= 1;
            if idx[k] < rest.len() - need + k {
                idx[k] += 1;
                for j in k + 1..need {
                    idx[j] = idx[j - 1] + 1;
                }
                break;
            }
        }
        if need == 0 {
            return sum / n as f64;
        }
    }
}

pub struct HandResult {
    pub chips: [f64; 2],
    /// AIVAT-style chance correction for each seat: Σ over dealt streets of
    /// v(board after the deal) − v(board before), v = equity value at the contributions then
    pub correction: [f64; 2],
    pub reached_turn: bool,
}

/// Play one heads-up hand. `board` holds all five cards (dealt as the hand reaches them).
pub fn play_hand(rules: &Rules, holes: [(u8, u8); 2], board: [u8; 5], agents: &mut [&mut dyn Agent; 2]) -> HandResult {
    let mut s = rules.root();
    let mut shown = 0usize;
    let mut corr = [0f64; 2];
    let mut reached_turn = false;
    let deal = |s: &BState, shown: &mut usize, corr: &mut [f64; 2], to: usize| {
        let total = [s.total[0], s.total[1]];
        for me in 0..2 {
            let before = equity_value(holes, &board[..*shown], total, me);
            let after = equity_value(holes, &board[..to], total, me);
            corr[me] += after - before;
        }
        *shown = to;
    };
    loop {
        let before = s;
        let p = s.to_act as usize;
        let a = agents[p].act(&s, &board[..shown]);
        for ag in agents.iter_mut() {
            ag.observe(&before, a, &board[..shown]);
        }
        match rules.apply(&s, a) {
            Next::Decision(n) => {
                if n.street != s.street {
                    let to = [0, 3, 4, 5][n.street as usize];
                    deal(&n, &mut shown, &mut corr, to);
                    reached_turn |= n.street >= 2;
                    for ag in agents.iter_mut() {
                        ag.new_round(&n, &board[..shown]);
                    }
                }
                s = n;
            }
            Next::Terminal(n) => {
                let alive = (0..2).filter(|&q| n.folded & (1 << q) == 0).count();
                let mut chips = [0f64; 2];
                if alive == 1 {
                    for q in 0..2 {
                        chips[q] = if n.folded & (1 << q) != 0 { -(n.total[q] as f64) } else { n.total[1 - q] as f64 };
                    }
                } else {
                    // run out the board (chance corrections street by street) and show down
                    while shown < 5 {
                        let to = if shown == 0 { 3 } else { shown + 1 };
                        reached_turn |= to >= 4;
                        deal(&n, &mut shown, &mut corr, to);
                    }
                    for q in 0..2 {
                        chips[q] = equity_value(holes, &board, [n.total[0], n.total[1]], q);
                    }
                }
                return HandResult { chips, correction: corr, reached_turn };
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn equity_value_is_exact_on_simple_cases() {
        // aces vs 7-2 on K 9 5 3 J: aces win 500
        let holes = [(48u8, 49u8), (20u8, 1u8)];
        assert_eq!(equity_value(holes, &[46, 30, 13, 7, 39], [500, 500], 0), 500.0);
        assert_eq!(equity_value(holes, &[46, 30, 13, 7, 39], [500, 500], 1), -500.0);
        // turn: average over the 44 rivers. 9-8 vs aces on 7 6 K 2: exactly 8 outs (four 5s,
        // four tens make a straight), so aces get 500 * (36 - 8) / 44
        let holes2 = [(48u8, 49u8), (28u8, 25u8)];
        let v = equity_value(holes2, &[22, 19, 46, 0], [500, 500], 0);
        assert!((v - 500.0 * 28.0 / 44.0).abs() < 1e-9, "{v}");
        // zero-sum
        assert!((v + equity_value(holes2, &[22, 19, 46, 0], [500, 500], 1)).abs() < 1e-9);
    }

    #[test]
    fn pseudo_harmonic_probability_matches_the_formula() {
        // P(A) for A = 0.5, B = 1, x = 0.75: (1 - 0.75)(1.5) / (0.5 * 1.75) = 0.42857
        let (a, b, x) = (0.5f64, 1.0f64, 0.75f64);
        let p = (b - x) * (1.0 + a) / ((b - a) * (1.0 + x));
        assert!((p - 0.428_571_428_571).abs() < 1e-9);
        // threshold x* = (A + B + 2AB) / (A + B + 2) gives P = 1/2
        let xs = (a + b + 2.0 * a * b) / (a + b + 2.0);
        assert!(((b - xs) * (1.0 + a) / ((b - a) * (1.0 + xs)) - 0.5).abs() < 1e-12);
    }
}
