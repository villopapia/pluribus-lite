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
//! * [`play_hand`]: one hand; returns each seat's chips and the AIVAT corrections [Burch et al.
//!   2018]: chance terms (equity value function) and decision terms at both agents' decisions,
//!   which use the actor's exact action distribution (`Agent::last_policy`) and ṽ = the search's
//!   value of the full history where the search agent acts in a solved subgame, else the equity
//!   value (D-EVAL-2).

use crate::belief::{action_likelihood, Beliefs};
use crate::betting::{Action, BState, Next, Rules, MAX_PLAYERS};
use crate::blueprint::{Deal, SnapshotAverage, Storage, ThreadStats, Tree, MAX_A};
use crate::buckets::CardAbstraction;
use crate::config::AbstractionConfig;
use crate::equity::{combo_index, COMBOS};
use crate::eval_lut::{LHand, Lut};
use crate::subgame::Subgame;
use crate::vsolve::VectorSolver;
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
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

    /// The blueprint's action for `hole` at real state `real` (board = cards dealt so far), the
    /// abstract action index it came from (None if the tracker is lost), and the distribution
    /// it was sampled from, over real actions (for AIVAT).
    pub fn act(&self, bp: &Blueprint, real: &BState, hole: (u8, u8), board: &[u8], d: &mut Deal, rng: &mut Rng) -> (Action, Option<usize>, Vec<(Action, f64)>) {
        let p = real.to_act as usize;
        if self.lost || self.state.to_act != real.to_act || self.state.street != real.street {
            return (Action::Call, None, vec![(Action::Call, 1.0)]);
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
        let real_of = |x: Action| match x {
            Action::Fold if max_bet > real.street_bet[p] => Action::Fold,
            Action::Fold | Action::Call => Action::Call,
            Action::RaiseTo(t) => {
                let ai = all_in_to(&bp.rules, real);
                if t == all_in_to(&bp.rules, &self.state) {
                    return Action::RaiseTo(ai);
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
        let tot: f64 = pr[..acts.len()].iter().sum();
        let mut dist: Vec<(Action, f64)> = Vec::new();
        for (j, &x) in acts.iter().enumerate() {
            if pr[j] > 0.0 {
                let r = real_of(x);
                match dist.iter_mut().find(|e| e.0 == r) {
                    Some(e) => e.1 += pr[j] / tot,
                    None => dist.push((r, pr[j] / tot)),
                }
            }
        }
        (real_of(acts[k]), Some(k), dist)
    }
}

/// What an agent sees: the real state before each action, and the board so far.
pub trait Agent {
    fn act(&mut self, s: &BState, board: &[u8]) -> Action;
    /// every action at the table, including our own, after it is taken
    fn observe(&mut self, before: &BState, a: Action, board: &[u8]);
    /// a new round starts with `board` (cards dealt so far) at state `s`
    fn new_round(&mut self, s: &BState, board: &[u8]);
    /// the distribution the last `act` sampled from, over real actions
    fn last_policy(&self) -> Vec<(Action, f64)>;
    /// AIVAT ṽ: chips for seat 0 after each of `acts` at the last decision, both hands known
    /// (None: the evaluator uses the equity value)
    fn action_values(&mut self, _acts: &[Action], _holes: [(u8, u8); 2]) -> Option<Vec<f64>> {
        None
    }
}

pub struct BlueprintAgent<'a> {
    pub bp: &'a Blueprint<'a>,
    pub hole: (u8, u8),
    pub tr: BpTracker,
    pub rng: Rng,
    pub d: Deal,
    pub off_tree: u64,
    mine: Option<usize>,
    last: Vec<(Action, f64)>,
}

impl<'a> BlueprintAgent<'a> {
    pub fn new(bp: &'a Blueprint<'a>, hole: (u8, u8), seed: u64) -> Self {
        BlueprintAgent { bp, hole, tr: BpTracker::new(&bp.rules), rng: Rng::new(seed), d: Deal::new(), off_tree: 0, mine: None, last: Vec::new() }
    }
}

impl Agent for BlueprintAgent<'_> {
    fn act(&mut self, s: &BState, board: &[u8]) -> Action {
        let (a, i, dist) = self.tr.act(self.bp, s, self.hole, board, &mut self.d, &mut self.rng);
        self.mine = i;
        self.last = dist;
        a
    }
    fn last_policy(&self) -> Vec<(Action, f64)> {
        self.last.clone()
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
    /// enumerate chance in the solver instead of sampling one board per iteration (the paper)
    pub enumerate: bool,
    /// boards sampled per iteration when sampling (Pluribus: one per thread)
    pub boards_per_iter: usize,
}

#[derive(Clone, Debug, Default)]
pub struct SearchStats {
    pub searches: u64,
    pub re_searches: u64,
    pub zero_prob_events: u64,
    /// exploitability (chips) within each river subgame: final, average, pot at the root
    pub river_expl: Vec<(f64, f64, f64)>,
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
    /// record every subgame searched (`capture_line`), for offline experiments
    pub capture: bool,
    pub captured: Vec<String>,
    last: Vec<(Action, f64)>,
    /// subgame node of our last decision, when it came from the search
    last_node: Option<u32>,
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
            capture: false,
            captured: Vec::new(),
            last: Vec::new(),
            last_node: None,
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
        if !self.set.enumerate {
            s = s.with_sampling(self.rng.next_u64()).with_boards_per_iter(self.set.boards_per_iter);
        }
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
        if self.capture {
            self.captured.push(capture_line(&root, &self.extra, &self.board, &s.ranges, &s.frozen));
        }
        let iters = if self.board.len() == 4 { self.set.turn_iters } else { self.set.river_iters };
        s.run(iters);
        if self.measure_expl && self.board.len() == 5 {
            let e_fin = s.exploitability(false);
            let e_avg = s.exploitability(true);
            self.stats.river_expl.push((e_fin, e_avg, rules_pot(&root)));
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
            let (a, i, dist) = self.tr.act(self.bp, s, self.hole, board, &mut self.d, &mut self.rng);
            self.mine = i;
            self.last = dist;
            self.last_node = None;
            return a;
        }
        let k = self.round.len();
        let node = self.node_after(k).expect("our state is in the subgame");
        let sol = self.solver.as_ref().unwrap();
        let play = sol.strategy(node, &[], self.combo, self.set.play_average);
        let avg = sol.strategy(node, &[], self.combo, true);
        let i = self.rng.sample_index(&play);
        let a = sol.g.action(node, i).unwrap();
        let tot: f64 = play.iter().sum();
        self.last = (0..play.len()).filter(|&j| play[j] > 0.0).map(|j| (sol.g.action(node, j).unwrap(), play[j] / tot)).collect();
        self.last_node = Some(node);
        self.frozen.push((k, avg));
        a
    }

    fn last_policy(&self) -> Vec<(Action, f64)> {
        self.last.clone()
    }

    fn action_values(&mut self, acts: &[Action], holes: [(u8, u8); 2]) -> Option<Vec<f64>> {
        let node = self.last_node?;
        let c = [combo_index(holes[0].0, holes[0].1), combo_index(holes[1].0, holes[1].1)];
        let avg = self.set.play_average;
        let s = self.solver.as_mut()?;
        let n = s.g.nodes[node as usize].n as usize;
        acts.iter()
            .map(|&a| {
                let i = (0..n).find(|&i| s.g.action(node, i) == Some(a))?;
                let ch = s.g.child(node, i);
                Some(s.pair_value(ch, &[], c, avg))
            })
            .collect()
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

fn rules_pot(s: &BState) -> f64 {
    s.total.iter().map(|&x| x as f64).sum()
}

/// (P(seat 0 wins), P(seat 0 loses)) over the remaining board cards, both hands known. The
/// pre-flop case (1.7M boards) is cached per hand pair.
pub fn win_lose(holes: [(u8, u8); 2], board: &[u8]) -> (f64, f64) {
    static PRE: OnceLock<Mutex<HashMap<[u8; 4], (f64, f64)>>> = OnceLock::new();
    let key = [holes[0].0, holes[0].1, holes[1].0, holes[1].1];
    if board.is_empty() {
        if let Some(&v) = PRE.get_or_init(Default::default).lock().unwrap().get(&key) {
            return v;
        }
    }
    let lut = Lut::get();
    let used: Vec<u8> = board.iter().copied().chain(key).collect();
    let rest: Vec<u8> = (0..52u8).filter(|c| !used.contains(c)).collect();
    let need = 5 - board.len();
    let (mut w, mut l, mut n) = (0u64, 0u64, 0u64);
    let mut idx: Vec<usize> = (0..need).collect();
    'outer: loop {
        let mut b = board.to_vec();
        b.extend(idx.iter().map(|&i| rest[i]));
        let bh = LHand::of(&b);
        let r0 = lut.class(bh.add(holes[0].0).add(holes[0].1));
        let r1 = lut.class(bh.add(holes[1].0).add(holes[1].1));
        match r0.cmp(&r1) {
            std::cmp::Ordering::Greater => w += 1,
            std::cmp::Ordering::Less => l += 1,
            std::cmp::Ordering::Equal => {}
        }
        n += 1;
        // next combination of `need` indices out of rest.len()
        let mut k = need;
        loop {
            if k == 0 {
                break 'outer;
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
    }
    let v = (w as f64 / n as f64, l as f64 / n as f64);
    if board.is_empty() {
        PRE.get_or_init(Default::default).lock().unwrap().insert(key, v);
    }
    v
}

/// Showdown value for `me` at contributions `total`, averaged over the remaining board cards.
pub fn equity_value(holes: [(u8, u8); 2], board: &[u8], total: [u32; 2], me: usize) -> f64 {
    let (w, l) = win_lose(holes, board);
    let v0 = w * total[1] as f64 - l * total[0] as f64;
    if me == 0 {
        v0
    } else {
        -v0
    }
}

pub struct HandResult {
    pub chips: [f64; 2],
    /// AIVAT chance terms for each seat: Σ over dealt streets of v(board after the deal) −
    /// v(board before), v = equity value at the contributions then
    pub correction: [f64; 2],
    /// AIVAT decision terms for each seat: Σ over both agents' decisions of
    /// ṽ(action taken) − Σ_a π(a) ṽ(a)
    pub dec_correction: [f64; 2],
    pub reached_turn: bool,
}

/// Play one heads-up hand. `board` holds all five cards (dealt as the hand reaches them).
pub fn play_hand(rules: &Rules, holes: [(u8, u8); 2], board: [u8; 5], agents: &mut [&mut dyn Agent; 2]) -> HandResult {
    let mut s = rules.root();
    let mut shown = 0usize;
    let mut corr = [0f64; 2];
    let mut dec0 = 0f64;
    let mut reached_turn = false;
    // seat 0's equity value at `shown` board cards, (win, lose) cached per board length
    let mut wl: [Option<(f64, f64)>; 6] = [None; 6];
    let mut v0 = |shown: usize, total: [u32; MAX_PLAYERS]| {
        let (w, l) = *wl[shown].get_or_insert_with(|| win_lose(holes, &board[..shown]));
        // an outstanding (uncalled) bet counts as returned: both seats at the smaller contribution
        (w - l) * total[0].min(total[1]) as f64
    };
    loop {
        let before = s;
        let p = s.to_act as usize;
        let a = agents[p].act(&s, &board[..shown]);
        // AIVAT decision term (zero mean under the actor's own distribution)
        let dist = agents[p].last_policy();
        let acts: Vec<Action> = dist.iter().map(|e| e.0).collect();
        let vals = match agents[p].action_values(&acts, holes) {
            Some(v) => v,
            None => acts
                .iter()
                .map(|&x| match rules.apply(&s, x) {
                    Next::Terminal(n) if n.folded != 0 => {
                        if n.folded & 1 != 0 {
                            -(n.total[0] as f64)
                        } else {
                            n.total[1] as f64
                        }
                    }
                    Next::Terminal(n) | Next::Decision(n) => v0(shown, n.total),
                })
                .collect(),
        };
        let taken = acts.iter().position(|&x| x == a).expect("the action taken is in the actor's distribution");
        dec0 += vals[taken] - dist.iter().zip(&vals).map(|(e, v)| e.1 * v).sum::<f64>();
        for ag in agents.iter_mut() {
            ag.observe(&before, a, &board[..shown]);
        }
        match rules.apply(&s, a) {
            Next::Decision(n) => {
                if n.street != s.street {
                    let to = [0, 3, 4, 5][n.street as usize];
                    let d = v0(to, n.total) - v0(shown, n.total);
                    corr[0] += d;
                    corr[1] -= d;
                    shown = to;
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
                        let d = v0(to, n.total) - v0(shown, n.total);
                        corr[0] += d;
                        corr[1] -= d;
                        shown = to;
                    }
                    chips[0] = v0(5, n.total);
                    chips[1] = -chips[0];
                }
                return HandResult { chips, correction: corr, dec_correction: [dec0, -dec0], reached_turn };
            }
        }
    }
}

/// One searched subgame as a text line (offline experiments: `iter_sweep`): board, added
/// actions, frozen entries, root state, both belief ranges.
pub fn capture_line(root: &BState, extra: &[(BState, Action)], board: &[u8], ranges: &[Vec<f64>; 2], frozen: &HashMap<(u32, u16), Vec<f64>>) -> String {
    use std::fmt::Write;
    let mut o = String::new();
    let st = |o: &mut String, s: &BState| {
        write!(o, " {} {} {} {} {} {} {}", s.street, s.to_act, s.folded, s.allin, s.need, s.raises, s.last_inc).unwrap();
        for x in s.total.iter().chain(&s.street_bet) {
            write!(o, " {x}").unwrap();
        }
    };
    write!(o, "{}", board.len()).unwrap();
    for c in board {
        write!(o, " {c}").unwrap();
    }
    write!(o, " {}", extra.len()).unwrap();
    for (s, a) in extra {
        st(&mut o, s);
        let (k, x) = match a {
            Action::Fold => (0, 0),
            Action::Call => (1, 0),
            Action::RaiseTo(x) => (2, *x),
        };
        write!(o, " {k} {x}").unwrap();
    }
    let mut fz: Vec<_> = frozen.iter().collect();
    fz.sort_by_key(|e| *e.0);
    write!(o, " {}", fz.len()).unwrap();
    for ((n, h), p) in fz {
        write!(o, " {n} {h} {}", p.len()).unwrap();
        for x in p {
            write!(o, " {x}").unwrap();
        }
    }
    st(&mut o, root);
    for r in ranges {
        for x in r {
            write!(o, " {x}").unwrap();
        }
    }
    o
}

/// A subgame read back from [`capture_line`].
pub struct Captured {
    pub root: BState,
    pub extra: Vec<(BState, Action)>,
    pub board: Vec<u8>,
    pub ranges: [Vec<f64>; 2],
    pub frozen: HashMap<(u32, u16), Vec<f64>>,
}

pub fn parse_capture(line: &str) -> Captured {
    let toks: Vec<&str> = line.split_whitespace().collect();
    let mut i = 0usize;
    let mut next = || {
        i += 1;
        toks[i - 1]
    };
    fn st<'a>(next: &mut impl FnMut() -> &'a str) -> BState {
        let mut v = [0u32; 19];
        for x in v.iter_mut() {
            *x = next().parse().unwrap();
        }
        let mut total = [0u32; MAX_PLAYERS];
        let mut street_bet = [0u32; MAX_PLAYERS];
        total.copy_from_slice(&v[7..13]);
        street_bet.copy_from_slice(&v[13..19]);
        BState { street: v[0] as u8, to_act: v[1] as u8, folded: v[2] as u8, allin: v[3] as u8, need: v[4] as u8, raises: v[5] as u8, last_inc: v[6], total, street_bet }
    }
    let nb: usize = next().parse().unwrap();
    let board: Vec<u8> = (0..nb).map(|_| next().parse().unwrap()).collect();
    let nx: usize = next().parse().unwrap();
    let mut extra = Vec::new();
    for _ in 0..nx {
        let s = st(&mut next);
        let k: u32 = next().parse().unwrap();
        let x: u32 = next().parse().unwrap();
        extra.push((s, [Action::Fold, Action::Call, Action::RaiseTo(x)][k as usize]));
    }
    let nf: usize = next().parse().unwrap();
    let mut frozen = HashMap::new();
    for _ in 0..nf {
        let n: u32 = next().parse().unwrap();
        let h: u16 = next().parse().unwrap();
        let k: usize = next().parse().unwrap();
        frozen.insert((n, h), (0..k).map(|_| next().parse().unwrap()).collect());
    }
    let root = st(&mut next);
    let r0: Vec<f64> = (0..COMBOS).map(|_| next().parse().unwrap()).collect();
    let r1: Vec<f64> = (0..COMBOS).map(|_| next().parse().unwrap()).collect();
    assert_eq!(i, toks.len(), "trailing tokens in capture line");
    Captured { root, extra, board, ranges: [r0, r1], frozen }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Fixed-probability agent over the legal actions of a fold/call/pot/all-in menu.
    struct RandomAgent {
        cfg: AbstractionConfig,
        rules: Rules,
        rng: Rng,
        last: Vec<(Action, f64)>,
    }

    impl Agent for RandomAgent {
        fn act(&mut self, s: &BState, _board: &[u8]) -> Action {
            let acts = self.rules.actions(s, &self.cfg);
            let w: Vec<f64> = acts.iter().map(|a| match a { Action::Fold => 1.0, Action::Call => 3.0, Action::RaiseTo(_) => 1.5 }).collect();
            let t: f64 = w.iter().sum();
            self.last = acts.iter().zip(&w).map(|(&a, &x)| (a, x / t)).collect();
            acts[self.rng.sample_index(&w)]
        }
        fn observe(&mut self, _: &BState, _: Action, _: &[u8]) {}
        fn new_round(&mut self, _: &BState, _: &[u8]) {}
        fn last_policy(&self) -> Vec<(Action, f64)> {
            self.last.clone()
        }
    }

    #[test]
    fn aivat_terms_are_unbiased_and_reduce_variance() {
        // toy agents with known action probabilities on real heads-up hands: the corrections
        // must have mean 0 (within 4 s.e.) and the corrected results lower variance
        let rules = Rules { players: 2, ..Rules::default() };
        let cfg = AbstractionConfig::parse("allin = true
preflop.0 = 1
preflop.1+ = 1
flop.0 = 1
flop.1+ = 1
turn.0 = 1
turn.1+ = 1
river.0 = 1
river.1+ = 1").unwrap();
        let mut rng = Rng::new(11);
        // a few fixed match-ups (pre-flop equity is cached per pair), random boards
        let pairs = [[(48u8, 49u8), (20u8, 1u8)], [(28, 25), (44, 41)], [(51, 47), (50, 46)], [(12, 13), (36, 32)]];
        let n = 3000;
        let (mut raw, mut chance, mut full) = (Vec::new(), Vec::new(), Vec::new());
        let mut terms = Vec::new();
        for k in 0..n {
            let holes = pairs[k % pairs.len()];
            let used = [holes[0].0, holes[0].1, holes[1].0, holes[1].1];
            let mut deck: Vec<u8> = (0..52u8).filter(|c| !used.contains(c)).collect();
            for i in 0..5 {
                let j = i + (rng.next_u64() % (deck.len() - i) as u64) as usize;
                deck.swap(i, j);
            }
            let board = [deck[0], deck[1], deck[2], deck[3], deck[4]];
            let mut a0 = RandomAgent { cfg: cfg.clone(), rules, rng: Rng::new(rng.next_u64()), last: vec![] };
            let mut a1 = RandomAgent { cfg: cfg.clone(), rules, rng: Rng::new(rng.next_u64()), last: vec![] };
            let r = play_hand(&rules, holes, board, &mut [&mut a0, &mut a1]);
            assert!((r.chips[0] + r.chips[1]).abs() < 1e-9 && (r.dec_correction[0] + r.dec_correction[1]).abs() < 1e-9);
            raw.push(r.chips[0]);
            chance.push(r.chips[0] - r.correction[0]);
            full.push(r.chips[0] - r.correction[0] - r.dec_correction[0]);
            terms.push(r.correction[0] + r.dec_correction[0]);
        }
        let ms = |v: &[f64]| {
            let m = v.iter().sum::<f64>() / v.len() as f64;
            (m, v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (v.len() - 1) as f64)
        };
        let (mt, vt) = ms(&terms);
        let (mr, vr) = ms(&raw);
        let (_, vc) = ms(&chance);
        let (mf, vf) = ms(&full);
        eprintln!("raw {mr:.1} (var {vr:.0}), chance-only var {vc:.0}, full {mf:.1} (var {vf:.0}); mean correction {mt:.2} ± {:.2}", (vt / n as f64).sqrt());
        assert!(mt.abs() < 4.0 * (vt / n as f64).sqrt(), "corrections biased: {mt}");
        assert!(vf < vc && vc < vr, "variance must fall: raw {vr}, chance {vc}, full {vf}");
    }

    #[test]
    fn capture_lines_round_trip() {
        let rules = Rules { players: 2, ..Rules::default() };
        let mut root = rules.root();
        root.street = 2;
        root.total[0] = 400;
        root.total[1] = 400;
        let extra = vec![(root, Action::RaiseTo(350)), (root, Action::Call)];
        let mut rng = Rng::new(5);
        let ranges = [(0..COMBOS).map(|_| rng.next_f64()).collect::<Vec<f64>>(), (0..COMBOS).map(|_| rng.next_f64() * 1e-7).collect()];
        let mut frozen = HashMap::new();
        frozen.insert((7u32, 123u16), vec![0.1, 0.2, 0.7]);
        let line = capture_line(&root, &extra, &[1, 2, 3, 4], &ranges, &frozen);
        let c = parse_capture(&line);
        assert_eq!(c.root, root);
        assert_eq!(c.extra, extra);
        assert_eq!(c.board, vec![1, 2, 3, 4]);
        assert_eq!(c.ranges, ranges);
        assert_eq!(c.frozen, frozen);
    }

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
