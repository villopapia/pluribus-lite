//! M4 stage 2: the subgame tree (Supp pp. 20–21, `docs/m4_design.md` §2).
//!
//! The root is the start of the current betting round (nested search "always solves starting at
//! the beginning of the current betting round"), given as the `BState` reached by replaying the
//! actual actions, so off-tree sizes earlier in the hand are exact. Leaves (Supp p. 21):
//! * root on round 1 (preflop): leaves at the chance node starting round 2;
//! * root on round 2 (flop) with more than two players at the start of the round: leaves at the
//!   chance node starting round 3 or immediately after the second raise of the round, whichever
//!   is earlier (D-SRCH-4);
//! * otherwise the subgame runs to the end of the game; a street change is then a `Chance` node
//!   with one child, where the solver deals the next card(s).
//!
//! Raise sizes come from the search's own configs (D-SRCH-3): one for the root's street, a coarser
//! one for the streets after it (`build_with`).

use crate::betting::{Action, BState, Next, Rules, MAX_PLAYERS};
use crate::config::AbstractionConfig;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Decision,
    /// one player left
    FoldEnd,
    /// river showdown, or everyone but one all-in (run-out)
    Showdown,
    /// depth limit: players choose continuation strategies here (stage 3)
    Leaf,
    /// street change inside the subgame; one child
    Chance,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LeafRule {
    EndOfRound,
    EndOfRoundOrSecondRaise,
    EndOfGame,
}

impl LeafRule {
    /// The paper's rule for a subgame rooted at `root` (the start of a round).
    pub fn for_root(rules: &Rules, root: &BState) -> LeafRule {
        match root.street {
            0 => LeafRule::EndOfRound,
            1 if rules.n_in_hand(root) > 2 => LeafRule::EndOfRoundOrSecondRaise,
            _ => LeafRule::EndOfGame,
        }
    }
}

/// Betting state at the start of post-flop `street` with each seat's total contribution and the
/// folded seats (as after the previous round closed).
pub fn round_start(rules: &Rules, street: u8, total: [u32; MAX_PLAYERS], folded: u8) -> BState {
    assert!((1..=3).contains(&street));
    let all = ((1u16 << rules.players) - 1) as u8;
    let allin = (0..rules.players).filter(|&p| total[p] == rules.stack).fold(0u8, |m, p| m | 1 << p) & !folded;
    let need = all & !folded & !allin;
    let first = if rules.players == 2 { 1 } else { 0 };
    let to_act = (0..rules.players).map(|k| (first + k) % rules.players).find(|&q| need & (1 << q) != 0).expect("nobody can act");
    BState { street, to_act: to_act as u8, folded, allin, need, raises: 0, last_inc: rules.bb, total, street_bet: [0; MAX_PLAYERS] }
}

/// Action sequences in a subgame: (history, action) pairs over its decision nodes, players'
/// actions only (Supp p. 13; the blueprint's 664,845,654 uses the same unit).
pub fn action_sequences(g: &Subgame) -> u64 {
    g.nodes.iter().filter(|n| n.kind == Kind::Decision).map(|n| n.n as u64).sum()
}

#[derive(Clone, Copy, Debug)]
pub struct SNode {
    pub kind: Kind,
    /// betting state at this node (for a leaf or chance node: the state after the action)
    pub state: BState,
    pub first: u32,
    pub n: u8,
}

pub struct Subgame {
    pub rule: LeafRule,
    pub root_street: u8,
    pub nodes: Vec<SNode>,
    pub children: Vec<u32>,
    /// action on each edge, parallel to `children` (`None` for the edge out of a chance node)
    pub edges: Vec<Option<Action>>,
    /// set when `build_capped` stopped expanding at its node cap (the tree is then incomplete)
    pub truncated: bool,
    cap: usize,
    /// seats whose decisions are forced folds (fold if legal, else check): the preflop
    /// "folded players fixed" reading of a round-start root (D-SRCH-14)
    pub forced_fold: u8,
    /// Algorithm 2's AddAction: (state, action) pairs added to the abstraction (stage 5)
    pub extra: Vec<(BState, Action)>,
}

impl Subgame {
    /// One size config for every street of the subgame.
    pub fn build(rules: &Rules, cfg: &AbstractionConfig, root: BState) -> Subgame {
        Self::build_with(rules, cfg, cfg, root, usize::MAX)
    }

    /// `root_cfg` sizes on the root's street, `later_cfg` on the streets after it (D-SRCH-3);
    /// stops expanding once `cap` nodes exist (`truncated`; for size measurements only).
    pub fn build_with(rules: &Rules, root_cfg: &AbstractionConfig, later_cfg: &AbstractionConfig, root: BState, cap: usize) -> Subgame {
        Self::build_inner(rules, root_cfg, later_cfg, root, cap, 0, vec![])
    }

    /// Preflop subgame from the start of the hand in which the seats in `forced_fold` only fold
    /// (or check when folding is not legal): the "folded players fixed" reading (D-SRCH-14).
    pub fn build_preflop_fixed(rules: &Rules, cfg: &AbstractionConfig, forced_fold: u8, cap: usize) -> Subgame {
        Self::build_inner(rules, cfg, cfg, rules.root(), cap, forced_fold, vec![])
    }

    /// `build_with` plus Algorithm 2's added off-tree actions.
    pub fn build_extra(rules: &Rules, root_cfg: &AbstractionConfig, later_cfg: &AbstractionConfig, root: BState, extra: Vec<(BState, Action)>) -> Subgame {
        Self::build_inner(rules, root_cfg, later_cfg, root, usize::MAX, 0, extra)
    }

    fn build_inner(rules: &Rules, root_cfg: &AbstractionConfig, later_cfg: &AbstractionConfig, root: BState, cap: usize, forced_fold: u8, extra: Vec<(BState, Action)>) -> Subgame {
        if root.street == 0 {
            assert_eq!(root, rules.root(), "a preflop subgame starts at the start of the hand");
        } else {
            assert!(root.raises == 0 && root.street_bet.iter().all(|&b| b == 0), "the root must be the start of a round");
        }
        let rule = LeafRule::for_root(rules, &root);
        let mut g = Subgame { rule, root_street: root.street, nodes: Vec::new(), children: Vec::new(), edges: Vec::new(), truncated: false, cap, forced_fold, extra };
        g.add(rules, [root_cfg, later_cfg], root, Kind::Decision);
        g
    }

    /// Actions at a decision state: the config's, filtered for forced folds, plus Algorithm 2's
    /// added off-tree actions at this exact state (raises in increasing order).
    fn actions_at(&self, rules: &Rules, cfgs: [&AbstractionConfig; 2], s: &BState, forced: u8) -> Vec<Action> {
        let mut acts: Vec<Action> = rules
            .actions(s, cfgs[(s.street != self.root_street) as usize])
            .into_iter()
            .filter(|&a| forced & (1 << s.to_act) == 0 || a == Action::Fold || (a == Action::Call && !rules.actions(s, cfgs[0]).contains(&Action::Fold)))
            .collect();
        for (st, a) in &self.extra {
            if st == s && !acts.contains(a) {
                acts.push(*a);
            }
        }
        let rank = |a: &Action| match a {
            Action::Fold => 0,
            Action::Call => 1,
            Action::RaiseTo(t) => 2 + *t as u64,
        };
        acts.sort_by_key(rank);
        acts
    }

    fn add(&mut self, rules: &Rules, cfgs: [&AbstractionConfig; 2], s: BState, kind: Kind) -> u32 {
        let id = self.nodes.len() as u32;
        self.nodes.push(SNode { kind, state: s, first: 0, n: 0 });
        if self.nodes.len() >= self.cap {
            self.truncated = true;
            return id;
        }
        let forced = self.forced_fold;
        let kids: Vec<(Option<Action>, u32)> = match kind {
            Kind::Decision => self
                .actions_at(rules, cfgs, &s, forced)
                .into_iter()
                .map(|a| {
                    let child = match rules.apply(&s, a) {
                        Next::Terminal(n) => {
                            let k = if rules.n_in_hand(&n) == 1 { Kind::FoldEnd } else { Kind::Showdown };
                            self.add(rules, cfgs, n, k)
                        }
                        Next::Decision(n) => {
                            let new_street = n.street > s.street;
                            let leaf = match self.rule {
                                LeafRule::EndOfRound => new_street,
                                LeafRule::EndOfRoundOrSecondRaise => new_street || n.raises >= 2,
                                LeafRule::EndOfGame => false,
                            };
                            let k = if leaf {
                                Kind::Leaf
                            } else if new_street {
                                Kind::Chance
                            } else {
                                Kind::Decision
                            };
                            self.add(rules, cfgs, n, k)
                        }
                    };
                    (Some(a), child)
                })
                .collect(),
            Kind::Chance => vec![(None, self.add(rules, cfgs, s, Kind::Decision))],
            _ => vec![],
        };
        let first = self.children.len() as u32;
        for (a, c) in kids.iter() {
            self.children.push(*c);
            self.edges.push(*a);
        }
        let nd = &mut self.nodes[id as usize];
        nd.first = first;
        nd.n = kids.len() as u8;
        id
    }

    #[inline]
    pub fn child(&self, node: u32, i: usize) -> u32 {
        self.children[self.nodes[node as usize].first as usize + i]
    }

    #[inline]
    pub fn action(&self, node: u32, i: usize) -> Option<Action> {
        self.edges[self.nodes[node as usize].first as usize + i]
    }

    pub fn count(&self, kind: Kind) -> usize {
        self.nodes.iter().filter(|n| n.kind == kind).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blueprint::{Tree, DECISION};

    fn cfg(text: &str) -> AbstractionConfig {
        AbstractionConfig::parse(text).unwrap()
    }

    fn play(rules: &Rules, actions: &[Action]) -> BState {
        actions.iter().fold(rules.root(), |s, &a| match rules.apply(&s, a) {
            Next::Decision(n) => n,
            Next::Terminal(_) => panic!("hand ended"),
        })
    }

    /// Independent count of decision nodes to the end of the game (no leaf logic).
    fn decisions_to_end(rules: &Rules, cfg: &AbstractionConfig, s: &BState) -> usize {
        1 + rules
            .actions(s, cfg)
            .into_iter()
            .map(|a| match rules.apply(s, a) {
                Next::Decision(n) => decisions_to_end(rules, cfg, &n),
                Next::Terminal(_) => 0,
            })
            .sum::<usize>()
    }

    const CFG: &str = "preflop.0 = 0.5\nflop.0 = 0.5, 1.0\nflop.1+ = 1.0\nturn.0 = 1.0\nriver.0 = 0.5, 1.0";
    use Action::{Call, RaiseTo};

    #[test]
    fn river_tree_matches_a_hand_count() {
        let r = Rules { players: 2, ..Rules::default() };
        // limp, check, then check down to the river
        let root = play(&r, &[Call, Call, Call, Call, Call, Call]);
        assert_eq!(root.street, 3);
        let g = Subgame::build(&r, &cfg("allin = true"), root);
        assert_eq!(g.rule, LeafRule::EndOfGame);
        // check/all-in; after check: check/all-in; facing all-in: fold/call
        assert_eq!(g.count(Kind::Decision), 4);
        assert_eq!(g.count(Kind::FoldEnd), 2);
        assert_eq!(g.count(Kind::Showdown), 3);
        assert_eq!(g.count(Kind::Leaf) + g.count(Kind::Chance), 0);
        assert_eq!(g.action(0, 0), Some(Call));
        assert_eq!(g.action(0, 1), Some(RaiseTo(9_900)));
    }

    #[test]
    fn preflop_subgame_is_the_blueprint_preflop_with_leaves_at_the_flop() {
        let r = Rules { players: 4, ..Rules::default() };
        let c = cfg(CFG);
        let g = Subgame::build(&r, &c, r.root());
        assert_eq!(g.rule, LeafRule::EndOfRound);
        let t = Tree::build(&r, &c, [169, 1, 1, 1]);
        assert_eq!(g.count(Kind::Decision) as u64, t.decision_nodes[0]);
        // leaves = edges from a preflop decision to a flop decision in the full tree
        let flop_starts = (0..t.nodes.len() as u32)
            .filter(|&i| t.nodes[i as usize].kind == DECISION && t.nodes[i as usize].street == 0)
            .flat_map(|i| (0..t.nodes[i as usize].n as usize).map(move |a| (i, a)))
            .filter(|&(i, a)| {
                let ch = &t.nodes[t.child(i, a) as usize];
                ch.kind == DECISION && ch.street == 1
            })
            .count();
        assert_eq!(g.count(Kind::Leaf), flop_starts);
        assert!(g.nodes.iter().all(|n| n.state.street == 0 || n.kind == Kind::Leaf || n.kind == Kind::Showdown));
        assert!(g.nodes.iter().filter(|n| n.kind == Kind::Leaf).all(|n| n.state.street == 1));
        assert_eq!(g.count(Kind::Chance), 0);
    }

    #[test]
    fn multiway_flop_leaves_at_the_turn_or_after_the_second_raise() {
        let r = Rules { players: 3, ..Rules::default() };
        let root = play(&r, &[Call, Call, Call]); // UTG limps, SB completes, BB checks
        assert_eq!((root.street, r.n_in_hand(&root)), (1, 3));
        let g = Subgame::build(&r, &cfg(CFG), root);
        assert_eq!(g.rule, LeafRule::EndOfRoundOrSecondRaise);
        let (mut at_turn, mut after_raise) = (0, 0);
        for n in &g.nodes {
            match n.kind {
                Kind::Leaf if n.state.street == 2 => at_turn += 1,
                Kind::Leaf => {
                    assert_eq!((n.state.street, n.state.raises), (1, 2));
                    after_raise += 1;
                }
                Kind::Decision => assert!(n.state.street == 1 && n.state.raises <= 1),
                Kind::Chance => panic!("no chance node before a leaf"),
                _ => {}
            }
        }
        assert!(at_turn > 0 && after_raise > 0, "{at_turn} {after_raise}");
        // a flop raise sequence: bet, raise -> leaf right after the raise
        let pot = r.pot(&root);
        let bet = g.child(0, 2); // [check, 0.5 pot, 1 pot, all-in]
        assert_eq!(g.action(0, 2), Some(RaiseTo(pot)));
        let after = g.child(bet, 2); // next player: [fold, call, 1 pot, all-in]
        assert!(matches!(g.action(bet, 2), Some(RaiseTo(_))));
        assert_eq!(g.nodes[after as usize].kind, Kind::Leaf);
    }

    #[test]
    fn added_off_tree_actions_appear_at_their_state_only() {
        let r = Rules { players: 2, ..Rules::default() };
        let root = play(&r, &[Call, Call, Call, Call, Call, Call]); // heads-up river, pot 200
        let c = cfg("allin = true");
        let base = Subgame::build(&r, &c, root);
        // Algorithm 2's AddAction: the opponent bet 130 (off-tree) at the river root
        let g = Subgame::build_extra(&r, &c, &c, root, vec![(root, RaiseTo(130))]);
        assert_eq!((0..g.nodes[0].n as usize).map(|i| g.action(0, i).unwrap()).collect::<Vec<_>>(), vec![Call, RaiseTo(130), RaiseTo(9_900)]);
        // the new branch has its own subtree; other states are unchanged (only check / all-in)
        assert!(action_sequences(&g) > action_sequences(&base));
        let after_check = g.child(0, 0);
        assert_eq!(g.nodes[after_check as usize].n, 2);
    }

    #[test]
    fn forced_folds_leave_one_action_for_folded_seats() {
        let r = Rules { players: 4, ..Rules::default() };
        let c = cfg(CFG);
        let full = Subgame::build(&r, &c, r.root());
        // seats 2 and 3 (first to act preflop) fixed to fold: exactly one action, Fold, wherever
        // they act; the blinds keep their full tree
        let g = Subgame::build_preflop_fixed(&r, &c, 0b1100, usize::MAX);
        let mut blind_decisions = 0;
        for (i, n) in g.nodes.iter().enumerate().filter(|(_, n)| n.kind == Kind::Decision) {
            if n.state.to_act >= 2 {
                assert_eq!((n.n, g.action(i as u32, 0)), (1, Some(Action::Fold)));
            } else {
                blind_decisions += 1;
            }
        }
        assert!(blind_decisions > 0);
        assert!(action_sequences(&g) < action_sequences(&full));
    }

    #[test]
    fn later_streets_use_the_later_config() {
        let r = Rules { players: 2, ..Rules::default() };
        let root = play(&r, &[Call, Call, Call, Call]); // heads-up turn
        let (rc, lc) = (cfg(CFG), cfg("allin = true\nriver.0 =\nriver.1+ ="));
        let g = Subgame::build_with(&r, &rc, &lc, root, usize::MAX);
        for n in g.nodes.iter().filter(|n| n.kind == Kind::Decision) {
            let want = r.actions(&n.state, if n.state.street == 2 { &rc } else { &lc });
            let got: Vec<Action> = (0..n.n as usize).map(|i| g.edges[n.first as usize + i].unwrap()).collect();
            assert_eq!(got, want);
            if n.state.street == 3 {
                // river: check/call/fold or all-in only
                assert!(got.iter().all(|a| !matches!(a, RaiseTo(t) if *t < n.state.street_bet[n.state.to_act as usize] + r.stack - n.state.total[n.state.to_act as usize])));
            }
        }
        assert!(action_sequences(&g) < action_sequences(&Subgame::build(&r, &rc, root)));
        // a cap truncates and flags the tree
        assert!(Subgame::build_with(&r, &rc, &rc, root, 10).truncated);
    }

    #[test]
    fn heads_up_flop_and_turn_run_to_the_end_of_the_game() {
        let r = Rules { players: 2, ..Rules::default() };
        let c = cfg(CFG);
        for actions in [&[Call, Call][..], &[Call, Call, Call, Call][..]] {
            let root = play(&r, actions);
            let g = Subgame::build(&r, &c, root);
            assert_eq!(g.rule, LeafRule::EndOfGame);
            assert_eq!(g.count(Kind::Leaf), 0);
            assert!(g.count(Kind::Chance) > 0);
            assert_eq!(g.count(Kind::Decision), decisions_to_end(&r, &c, &root));
            // every chance node changes the street and has one decision child
            for (i, n) in g.nodes.iter().enumerate().filter(|(_, n)| n.kind == Kind::Chance) {
                let ch = &g.nodes[g.child(i as u32, 0) as usize];
                assert_eq!((n.n, ch.kind, ch.state.street), (1, Kind::Decision, n.state.street));
            }
        }
        // a three-way pot where the flop started heads-up (one folded preflop) also runs to the end
        let r3 = Rules { players: 3, ..Rules::default() };
        let root = play(&r3, &[Action::Fold, Call, Call]);
        assert_eq!(Subgame::build(&r3, &c, root).rule, LeafRule::EndOfGame);
    }
}
