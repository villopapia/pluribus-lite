//! Expected values, exact best response and exploitability (two-player zero-sum).
//!
//! Best response for player `b` against a fixed policy:
//! 1. Walk the full tree, grouping every history where `b` acts by infoset,
//!    together with its counterfactual reach (chance x opponent probabilities).
//! 2. The BR action at infoset I is argmax_a sum_{h in I} reach(h) * V(h.a), where V
//!    uses BR actions at deeper infosets (well-defined by perfect recall). Infoset
//!    decisions are memoised.

use crate::fxhash::FxHashMap;
use crate::game::{Action, Game, NodeKind, MAX_ACTIONS};
use crate::table::Policy;

/// Expected utility of every player when all play `policy`.
pub fn expected_values<G: Game, P: Policy>(g: &G, policy: &P) -> Vec<f64> {
    let mut out = vec![0.0; g.num_players()];
    ev_rec(g, &g.root(), policy, 1.0, &mut out);
    out
}

fn ev_rec<G: Game, P: Policy>(g: &G, s: &G::State, pol: &P, w: f64, out: &mut [f64]) {
    match g.kind(s) {
        NodeKind::Terminal => {
            for (p, o) in out.iter_mut().enumerate() {
                *o += w * g.utility(s, p);
            }
        }
        NodeKind::Chance => {
            let c = g.chance_outcomes(s);
            for (&a, &p) in c.actions().iter().zip(c.probs()) {
                ev_rec(g, &g.apply(s, a), pol, w * p, out);
            }
        }
        NodeKind::Player(_) => {
            let acts = g.legal_actions(s);
            let n = acts.len();
            let mut probs = [0.0; MAX_ACTIONS];
            pol.action_probs(g.infoset_key(s), n, &mut probs);
            for k in 0..n {
                if probs[k] > 0.0 {
                    ev_rec(g, &g.apply(s, acts.as_slice()[k]), pol, w * probs[k], out);
                }
            }
        }
    }
}

struct BestResponse<'a, G: Game, P: Policy> {
    g: &'a G,
    pol: &'a P,
    br: usize,
    infosets: FxHashMap<u64, Vec<(G::State, f64)>>,
    choice: FxHashMap<u64, Action>,
}

impl<'a, G: Game, P: Policy> BestResponse<'a, G, P> {
    fn collect(&mut self, s: &G::State, reach: f64) {
        match self.g.kind(s) {
            NodeKind::Terminal => {}
            NodeKind::Chance => {
                let c = self.g.chance_outcomes(s);
                for (&a, &p) in c.actions().iter().zip(c.probs()) {
                    self.collect(&self.g.apply(s, a), reach * p);
                }
            }
            NodeKind::Player(p) => {
                let acts = self.g.legal_actions(s);
                if p == self.br {
                    self.infosets
                        .entry(self.g.infoset_key(s))
                        .or_default()
                        .push((s.clone(), reach));
                    for &a in acts.as_slice() {
                        self.collect(&self.g.apply(s, a), reach);
                    }
                } else {
                    let mut probs = [0.0; MAX_ACTIONS];
                    self.pol.action_probs(self.g.infoset_key(s), acts.len(), &mut probs);
                    for (k, &a) in acts.as_slice().iter().enumerate() {
                        // keep zero-reach branches: the BR still needs a defined action there
                        self.collect(&self.g.apply(s, a), reach * probs[k]);
                    }
                }
            }
        }
    }

    fn best_action(&mut self, key: u64) -> Action {
        if let Some(&a) = self.choice.get(&key) {
            return a;
        }
        let entries = self.infosets[&key].clone();
        let acts = self.g.legal_actions(&entries[0].0);
        let mut best = (f64::NEG_INFINITY, acts.as_slice()[0]);
        for &a in acts.as_slice() {
            let mut q = 0.0;
            for (h, reach) in &entries {
                q += reach * self.value(&self.g.apply(h, a));
            }
            if q > best.0 + 1e-15 {
                best = (q, a);
            }
        }
        self.choice.insert(key, best.1);
        best.1
    }

    fn value(&mut self, s: &G::State) -> f64 {
        match self.g.kind(s) {
            NodeKind::Terminal => self.g.utility(s, self.br),
            NodeKind::Chance => {
                let c = self.g.chance_outcomes(s);
                let mut v = 0.0;
                for (&a, &p) in c.actions().iter().zip(c.probs()) {
                    v += p * self.value(&self.g.apply(s, a));
                }
                v
            }
            NodeKind::Player(p) => {
                if p == self.br {
                    let a = self.best_action(self.g.infoset_key(s));
                    self.value(&self.g.apply(s, a))
                } else {
                    let acts = self.g.legal_actions(s);
                    let mut probs = [0.0; MAX_ACTIONS];
                    self.pol.action_probs(self.g.infoset_key(s), acts.len(), &mut probs);
                    let mut v = 0.0;
                    for (k, &a) in acts.as_slice().iter().enumerate() {
                        if probs[k] > 0.0 {
                            v += probs[k] * self.value(&self.g.apply(s, a));
                        }
                    }
                    v
                }
            }
        }
    }
}

/// Value that player `br` obtains by best-responding to `policy` (others fixed).
pub fn best_response_value<G: Game, P: Policy>(g: &G, policy: &P, br: usize) -> f64 {
    let mut b = BestResponse {
        g,
        pol: policy,
        br,
        infosets: FxHashMap::default(),
        choice: FxHashMap::default(),
    };
    let root = g.root();
    b.collect(&root, 1.0);
    b.value(&root)
}

/// NashConv = sum_i (BR_i value - value_i(policy)).
pub fn nash_conv<G: Game, P: Policy>(g: &G, policy: &P) -> f64 {
    let ev = expected_values(g, policy);
    (0..g.num_players())
        .map(|p| best_response_value(g, policy, p) - ev[p])
        .sum()
}

/// Exploitability = NashConv / num_players (OpenSpiel convention). For two-player
/// zero-sum games this is the mean of the two best-response values, in chips per hand.
pub fn exploitability<G: Game, P: Policy>(g: &G, policy: &P) -> f64 {
    nash_conv(g, policy) / g.num_players() as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kuhn::{self, Kuhn};
    use crate::leduc::Leduc;
    use crate::table::{TabularPolicy, UniformPolicy};

    /// Kuhn equilibrium family with player-0 parameter alpha in [0, 1/3].
    pub fn kuhn_equilibrium(alpha: f64) -> TabularPolicy {
        let mut p = TabularPolicy::default();
        let (j, q, k) = (0u8, 1u8, 2u8);
        let mut set = |card, hist: &[u8], bet: f64| {
            p.map.insert(kuhn::key(card, hist), vec![1.0 - bet, bet]);
        };
        // player 0, first action
        set(j, &[], alpha);
        set(q, &[], 0.0);
        set(k, &[], 3.0 * alpha);
        // player 0 after pass-bet
        set(j, &[0, 1], 0.0);
        set(q, &[0, 1], alpha + 1.0 / 3.0);
        set(k, &[0, 1], 1.0);
        // player 1 after pass
        set(j, &[0], 1.0 / 3.0);
        set(q, &[0], 0.0);
        set(k, &[0], 1.0);
        // player 1 after bet
        set(j, &[1], 0.0);
        set(q, &[1], 1.0 / 3.0);
        set(k, &[1], 1.0);
        p
    }

    #[test]
    fn kuhn_known_equilibria_have_zero_exploitability() {
        for alpha in [0.0, 0.1, 1.0 / 3.0] {
            let pol = kuhn_equilibrium(alpha);
            let ev = expected_values(&Kuhn, &pol);
            assert!((ev[0] + 1.0 / 18.0).abs() < 1e-12, "alpha {alpha}: {ev:?}");
            assert!((ev[0] + ev[1]).abs() < 1e-12);
            let e = exploitability(&Kuhn, &pol);
            assert!(e.abs() < 1e-12, "alpha {alpha}: exploitability {e}");
        }
    }

    #[test]
    fn kuhn_uniform_nash_conv_matches_reference() {
        // Reference: OpenSpiel reports NashConv 0.916666... for uniform Kuhn.
        let nc = nash_conv(&Kuhn, &UniformPolicy);
        assert!((nc - 11.0 / 12.0).abs() < 1e-12, "nash_conv {nc}");
    }

    #[test]
    fn leduc_uniform_nash_conv_matches_reference() {
        // Reference: OpenSpiel reports NashConv 4.747222... for uniform Leduc.
        let nc = nash_conv(&Leduc, &UniformPolicy);
        assert!((nc - 4.747222222222222).abs() < 1e-9, "nash_conv {nc}");
    }

    #[test]
    fn perturbed_equilibrium_is_exploitable() {
        let mut pol = kuhn_equilibrium(0.0);
        pol.map.insert(kuhn::key(2, &[1]), vec![1.0, 0.0]); // P1 folds K to a bet
        assert!(exploitability(&Kuhn, &pol) > 0.05);
    }
}
