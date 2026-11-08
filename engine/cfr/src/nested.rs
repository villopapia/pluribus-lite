//! M4 stage 5 (Leduc): Algorithm 2, nested search (Supp pp. 20–23, Algorithm 2 p. 25;
//! `docs/m4_design.md` §5).
//!
//! ```text
//! I ← ∅;  G_root ← G(I);  σ ← σ_blueprint
//! OpponentTurn(a): if a ∉ abstraction(I): AddAction at G(I); σ ← Search(G_root)
//!                  I ← I·a; CheckNewRound()
//! OurTurn():       a ~ σ(I); frozen(I) ← true; I ← I·a; CheckNewRound()
//! CheckNewRound(): if round(G(I)) > round(G_root): G_root ← G(I); σ ← Search(G_root)
//! ```
//! * σ (beliefs, frozen values) is each search's weighted average; the agent plays the final
//!   iteration (p. 22; D-SRCH-11). Frozen infosets keep σ(I) as it was when we acted, for our
//!   actual hand only; opponents are never frozen (p. 23).
//! * At a new round, beliefs are Bayes-updated for the finished round's actions with σ (and the
//!   board card), then the new round's start is the root (p. 21). Zero-probability observations
//!   are counted (D-SRCH-2).
//! * Validation option `search_round0`: also search at the start of the hand. Algorithm 2 only
//!   searches round 1 after an off-tree opponent action (Pluribus' preflop rule, p. 15); Leduc has
//!   no off-tree sizes, so without this option no depth-limited subgame would ever be built.
//!   `search_round1 = false` keeps the blueprint after round 1 (isolates the leaves' effect).
//!
//! The agent's strategy at an infoset is a deterministic function of the infoset (fixed solver
//! seeds; boards canonicalised by suit), computed by replaying Algorithm 2 along the infoset's
//! public history ([`Agent::act`]); [`agent_policy`] tabulates it for exact best response.

use crate::game::{Action, Game, NodeKind, MAX_ACTIONS};
use crate::leduc::{key_for, Leduc, LeducState, RAISE};
use crate::leduc_subgame::{cont_probs, public_key, replay, update_beliefs, LeducSubgame, LEAF_TAG};
use crate::subgame_solve::{solve, Solution, Solver};
use crate::table::{Policy, TabularPolicy};
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

#[derive(Clone, Debug)]
pub struct SearchConfig {
    /// continuation strategies at leaves: 4 (Pluribus) or 1 (blueprint only)
    pub n_cont: usize,
    /// compression seed of the continuation strategies (D-SRCH-6); None = exact distributions
    pub compress: Option<u64>,
    pub search_round0: bool,
    pub search_round1: bool,
    /// solver for round-0 ("early") subgames, and for round-1 subgames
    pub early: Solver,
    pub late: Solver,
    /// search abstraction without re-raises (exercises off-tree actions)
    pub no_reraise: bool,
    /// diagnostic, not the paper: play the weighted average instead of the final iteration
    pub play_average: bool,
    /// diagnostic, not the paper: with `search_round1 = false`, play round 2 by the continuation
    /// strategy chosen at the round-1 leaf (mixture → behaviour strategy by Bayes over our own
    /// round-2 actions), so round 1 is played as the depth-limited solution assumed
    pub follow_continuation: bool,
}

/// σ seen by the belief update: frozen infosets, then the latest search's average, then the
/// blueprint; vectors over the search abstraction are padded to Leduc's actions (RAISE last).
struct SigmaView<'v> {
    frozen: &'v HashMap<u64, Vec<f64>>,
    avg: Option<&'v TabularPolicy>,
    bp: &'v TabularPolicy,
}

fn pad(v: &[f64], n: usize, out: &mut [f64]) {
    out[..v.len()].copy_from_slice(v);
    out[v.len()..n].iter_mut().for_each(|x| *x = 0.0);
}

impl Policy for SigmaView<'_> {
    fn action_probs(&self, key: u64, n: usize, out: &mut [f64]) {
        if let Some(v) = self.frozen.get(&key).or_else(|| self.avg.and_then(|a| a.map.get(&key))) {
            pad(v, n, out);
        } else {
            self.bp.action_probs(key, n, out);
        }
    }
}

pub struct Agent<'a> {
    pub bp: &'a TabularPolicy,
    pub cfg: SearchConfig,
    cache: HashMap<Vec<u64>, Rc<Solution>>,
    /// searches actually solved (cache misses), off-tree re-searches, zero-probability events
    pub searches: u64,
    pub offtree: u64,
    pub zero_prob_events: u64,
    /// frozen map passed to the most recent search (for tests)
    pub last_frozen: HashMap<u64, Vec<f64>>,
}

impl<'a> Agent<'a> {
    pub fn new(bp: &'a TabularPolicy, cfg: SearchConfig) -> Self {
        Agent { bp, cfg, cache: HashMap::new(), searches: 0, offtree: 0, zero_prob_events: 0, last_frozen: HashMap::new() }
    }

    fn search(&mut self, root: &[Action], ranges: &[Vec<f64>; 2], frozen: &HashMap<u64, Vec<f64>>, added: &HashSet<u64>) -> Rc<Solution> {
        self.last_frozen = frozen.clone();
        let mut key: Vec<u64> = root.iter().map(|&a| a as u64).collect();
        key.push(u64::MAX);
        key.extend(ranges.iter().flatten().map(|x| x.to_bits()));
        let mut fz: Vec<_> = frozen.iter().collect();
        fz.sort_by_key(|(k, _)| **k);
        for (k, v) in fz {
            key.push(*k);
            key.extend(v.iter().map(|x| x.to_bits()));
        }
        let mut ad: Vec<_> = added.iter().copied().collect();
        ad.sort_unstable();
        key.push(u64::MAX);
        key.extend(ad);
        if let Some(s) = self.cache.get(&key) {
            return s.clone();
        }
        self.searches += 1;
        let mut sub = LeducSubgame::new(root.to_vec(), ranges.clone(), self.bp, self.cfg.n_cont, self.cfg.compress);
        sub.frozen = frozen.clone();
        sub.no_reraise = self.cfg.no_reraise;
        sub.added = added.clone();
        let solver = if sub.root_round == 0 { self.cfg.early } else { self.cfg.late };
        let sol = Rc::new(solve(sub, solver));
        self.cache.insert(key, sol.clone());
        sol
    }

    /// Probabilities over the search abstraction's actions at `s` from `pol` (renormalised if
    /// the abstraction drops an action the policy has).
    fn over_abstraction(&self, pol: &dyn Policy, s: &LeducState, added: &HashSet<u64>) -> Vec<f64> {
        let full = Leduc.legal_actions(s);
        let mut p = [0.0; MAX_ACTIONS];
        pol.action_probs(Leduc.infoset_key(s), full.len(), &mut p);
        let drop = self.cfg.no_reraise && s.raises > 0 && !added.contains(&public_key(s)) && full.as_slice().contains(&RAISE);
        if !drop {
            return p[..full.len()].to_vec(); // bit-identical to σ, so beliefs and caches agree
        }
        let n = full.len() - 1;
        let z: f64 = p[..n].iter().sum();
        p[..n].iter().map(|x| if z > 0.0 { x / z } else { 1.0 / n as f64 }).collect()
    }

    /// The agent's strategy (over Leduc's legal actions) when it sits in `seat` with `card` and
    /// the public actions since the start of the hand are `public` (bets and the board card).
    pub fn act(&mut self, seat: usize, card: u8, public: &[Action]) -> Vec<f64> {
        let g = Leduc;
        // canonical suits: the board card (if any) is even; swap both suits otherwise
        let mut public = public.to_vec();
        let mut card = card;
        if let Some(bi) = board_index(&public) {
            if public[bi] % 2 == 1 {
                public[bi] ^= 1;
                card ^= 1;
            }
        }
        let board = {
            let s = replay(&public, 0, 1);
            (s.round == 1).then_some(s.board)
        };
        let opp = (0..6u8).find(|&c| c != card && Some(c) != board).unwrap();
        let mut cards = [opp, opp];
        cards[seat] = card;

        let mut ranges = [vec![1.0 / 6.0; 6], vec![1.0 / 6.0; 6]];
        let mut root: Vec<Action> = Vec::new();
        let mut round_start = 0usize;
        let mut frozen: HashMap<u64, Vec<f64>> = HashMap::new();
        let mut added: HashSet<u64> = HashSet::new();
        let mut current: Option<Rc<Solution>> = None;
        if self.cfg.search_round0 {
            current = Some(self.search(&root, &ranges, &frozen, &added));
        }
        // for `follow_continuation`: the round-1 leaf state with the search in force there, and
        // our round-2 decisions so far
        let mut leaf: Option<(LeducState, Option<Rc<Solution>>)> = None;
        let mut own_r2: Vec<(LeducState, Action)> = Vec::new();
        let mut s = g.apply(&g.apply(&g.root(), cards[0]), cards[1]);
        for (idx, &a) in public.iter().enumerate() {
            match g.kind(&s) {
                NodeKind::Chance => {
                    // CheckNewRound: Bayes update for the finished round with σ, then the board
                    let view = SigmaView { frozen: &frozen, avg: current.as_ref().map(|c| &c.avg), bp: self.bp };
                    let obs = update_beliefs(&mut ranges, &public[..round_start], &public[round_start..=idx], &view);
                    self.zero_prob_events += obs.iter().filter(|&&o| o == 0.0).count() as u64;
                    leaf = Some((s, current.clone()));
                    s = g.apply(&s, a);
                    root = public[..=idx].to_vec();
                    round_start = idx + 1;
                    frozen.clear(); // everything before the new root is fixed by the root itself
                    added.clear();
                    current = if self.cfg.search_round1 { Some(self.search(&root, &ranges, &frozen, &added)) } else { None };
                }
                NodeKind::Player(p) => {
                    if p == seat && s.round == 1 {
                        own_r2.push((s, a));
                    }
                    if p == seat {
                        // OurTurn: freeze σ(I) for our actual hand
                        let sig: &dyn Policy = match &current {
                            Some(c) => &c.avg,
                            None => self.bp,
                        };
                        let probs = self.over_abstraction(sig, &s, &added);
                        frozen.insert(g.infoset_key(&s), probs);
                    } else {
                        // OpponentTurn: an action outside the search abstraction is added and
                        // the subgame is searched again from the root
                        let off = self.cfg.no_reraise && a == RAISE && s.raises > 0 && !added.contains(&public_key(&s));
                        if off {
                            added.insert(public_key(&s));
                            self.offtree += 1;
                            current = Some(self.search(&root, &ranges, &frozen, &added));
                        }
                    }
                    s = g.apply(&s, a);
                }
                NodeKind::Terminal => panic!("public history goes past the end of the hand"),
            }
        }
        assert_eq!(g.kind(&s), NodeKind::Player(seat), "not our turn");
        let full = g.legal_actions(&s).len();
        let mut out = vec![0.0; full];
        match &current {
            Some(c) => {
                let v = self.over_abstraction(if self.cfg.play_average { &c.avg } else { &c.play }, &s, &added);
                pad(&v, full, &mut out);
            }
            None => match &leaf {
                Some((ls, Some(r1))) if self.cfg.follow_continuation => {
                    // mixture over continuations chosen at the leaf, conditioned on our own
                    // round-2 actions so far (Kuhn: mixed → behaviour strategy)
                    let nc = self.cfg.n_cont;
                    let lkey = LEAF_TAG | key_for(ls, seat) << 1 | seat as u64;
                    let pol = if self.cfg.play_average { &r1.avg } else { &r1.play };
                    let mut prior = vec![0.0; nc];
                    pol.action_probs(lkey, nc, &mut prior);
                    let mut w = prior.clone();
                    for (st, a) in &own_r2 {
                        let i = g.legal_actions(st).as_slice().iter().position(|x| x == a).unwrap();
                        for (k, wk) in w.iter_mut().enumerate() {
                            *wk *= cont_probs(self.bp, st, k, self.cfg.compress)[i];
                        }
                    }
                    if w.iter().sum::<f64>() == 0.0 {
                        w = prior; // a path our own continuations never take: any choice is fine
                    }
                    let z: f64 = w.iter().sum();
                    for (k, wk) in w.iter().enumerate() {
                        for (o, p) in out.iter_mut().zip(cont_probs(self.bp, &s, k, self.cfg.compress)) {
                            *o += wk / z * p;
                        }
                    }
                }
                _ => self.bp.action_probs(g.infoset_key(&s), full, &mut out),
            },
        }
        out
    }
}

/// Index of the board card in a public sequence, if it has been dealt.
fn board_index(public: &[Action]) -> Option<usize> {
    let g = Leduc;
    let mut s = replay(&[], 0, 1);
    for (i, &a) in public.iter().enumerate() {
        if g.kind(&s) == NodeKind::Chance {
            return Some(i);
        }
        s = g.apply(&s, a);
    }
    None
}

/// The agent's strategy at every Leduc infoset of both seats, plus each key's player.
pub fn agent_policy(agent: &mut Agent) -> (TabularPolicy, HashMap<u64, usize>) {
    fn walk(g: &Leduc, s: &LeducState, public: &mut Vec<Action>, agent: &mut Agent, pol: &mut TabularPolicy, owner: &mut HashMap<u64, usize>) {
        match g.kind(s) {
            NodeKind::Terminal => {}
            NodeKind::Chance => {
                for &a in g.chance_outcomes(s).actions() {
                    let dealing_holes = s.dealt < 2;
                    if !dealing_holes {
                        public.push(a);
                    }
                    walk(g, &g.apply(s, a), public, agent, pol, owner);
                    if !dealing_holes {
                        public.pop();
                    }
                }
            }
            NodeKind::Player(p) => {
                let key = g.infoset_key(s);
                if let std::collections::hash_map::Entry::Vacant(e) = pol.map.entry(key) {
                    e.insert(agent.act(p, s.cards[p], public));
                    owner.insert(key, p);
                }
                for &a in g.legal_actions(s).as_slice() {
                    public.push(a);
                    walk(g, &g.apply(s, a), public, agent, pol, owner);
                    public.pop();
                }
            }
        }
    }
    let (mut pol, mut owner) = (TabularPolicy::default(), HashMap::new());
    walk(&Leduc, &Leduc.root(), &mut Vec::new(), agent, &mut pol, &mut owner);
    (pol, owner)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::leduc::CALL;
    use crate::mccfr::{Mccfr, MccfrConfig};

    fn blueprint() -> TabularPolicy {
        let mut m = Mccfr::new(Leduc, MccfrConfig::pluribus_scaled(5_000, 13.0, 1));
        m.run(5_000);
        m.average_policy()
    }

    fn cfg(search_round0: bool, no_reraise: bool) -> SearchConfig {
        let v = Solver::Vector { iters: 300, seed: 1 };
        SearchConfig { n_cont: 4, compress: Some(1), search_round0, search_round1: true, early: v, late: v, no_reraise, play_average: false, follow_continuation: false }
    }

    fn close(a: &[f64], b: &[f64]) {
        assert_eq!(a.len(), b.len());
        assert!(a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-12), "{a:?} vs {b:?}");
    }

    #[test]
    fn literal_algorithm2_plays_the_blueprint_on_round_one_and_searches_at_the_board() {
        let bp = blueprint();
        let mut ag = Agent::new(&bp, cfg(false, false));
        let s = replay(&[CALL], 4, 0);
        let mut want = [0.0; 2];
        bp.action_probs(Leduc.infoset_key(&s), 2, &mut want);
        close(&ag.act(1, 0, &[CALL]), &want);
        assert_eq!(ag.searches, 0);
        let p = ag.act(0, 4, &[CALL, CALL, 2]);
        assert_eq!(ag.searches, 1);
        assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn search_at_the_start_plays_the_final_iteration_of_that_search() {
        let bp = blueprint();
        let mut ag = Agent::new(&bp, cfg(true, false));
        let got = ag.act(0, 2, &[]);
        let sub = LeducSubgame::new(vec![], [vec![1.0 / 6.0; 6], vec![1.0 / 6.0; 6]], &bp, 4, Some(1));
        let sol = solve(sub, Solver::Vector { iters: 300, seed: 1 });
        let s = replay(&[], 2, 0);
        close(&got, &sol.play.map[&Leduc.infoset_key(&s)]);
    }

    #[test]
    fn off_tree_reraise_is_added_and_searched_again_with_our_action_frozen() {
        let bp = blueprint();
        let mut ag = Agent::new(&bp, cfg(true, true));
        // we (seat 0, a king) raise; the opponent re-raises, which the abstraction lacks
        let first = ag.act(0, 4, &[]);
        assert_eq!(first.len(), 2);
        let s0 = replay(&[], 4, 0);
        let k0 = Leduc.infoset_key(&s0);
        let p = ag.act(0, 4, &[RAISE, RAISE]);
        assert_eq!((ag.offtree, ag.searches), (1, 2));
        assert_eq!(p.len(), 2); // Leduc caps at two raises: fold or call
        // the re-search kept our first decision at σ(I) = the first search's average
        let sub = LeducSubgame::new(vec![], [vec![1.0 / 6.0; 6], vec![1.0 / 6.0; 6]], &bp, 4, Some(1));
        let mut sub = sub;
        sub.no_reraise = true;
        let sol = solve(sub, Solver::Vector { iters: 300, seed: 1 });
        close(&ag.last_frozen[&k0], &sol.avg.map[&k0]);
        assert_eq!(ag.last_frozen.len(), 1);
    }

    #[test]
    fn impossible_opponent_actions_are_counted() {
        let mut bp = blueprint();
        // the opponent (seat 1) never bets after a check, whatever its card
        for c in 0..6u8 {
            let s = replay(&[CALL], (c + 1) % 6, c);
            bp.map.insert(Leduc.infoset_key(&s), vec![1.0, 0.0]);
        }
        let mut ag = Agent::new(&bp, cfg(false, false));
        ag.act(0, 4, &[CALL, RAISE, CALL, 2]);
        assert_eq!(ag.zero_prob_events, 1);
    }

    #[test]
    fn following_the_blueprint_continuation_plays_the_compressed_blueprint() {
        // one continuation (the blueprint): on round 2 the agent plays its compressed form
        let bp = blueprint();
        let mut c = cfg(true, false);
        c.n_cont = 1;
        c.search_round1 = false;
        c.follow_continuation = true;
        let mut ag = Agent::new(&bp, c);
        let public = [CALL, CALL, 2];
        let mut s = replay(&public, 4, 0);
        s.cards[0] = 4;
        close(&ag.act(0, 4, &public), &cont_probs(&bp, &s, 0, Some(1)));
    }

    #[test]
    fn agent_policy_covers_every_infoset() {
        let bp = blueprint();
        let mut ag = Agent::new(&bp, cfg(true, false));
        let (pol, owner) = agent_policy(&mut ag);
        assert_eq!(pol.map.len(), 288);
        assert_eq!(owner.len(), 288);
        assert!(pol.map.values().all(|v| (v.iter().sum::<f64>() - 1.0).abs() < 1e-9));
        // one round-1 search, then one per (round-1 line, canonical board rank)
        assert!(ag.searches <= 1 + 5 * 3, "{} searches", ag.searches);
    }
}
