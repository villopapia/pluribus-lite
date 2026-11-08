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
//! Raise sizes come from the search's own `AbstractionConfig` (D-SRCH-3).

use crate::betting::{Action, BState, Next, Rules};
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
    pub nodes: Vec<SNode>,
    pub children: Vec<u32>,
    /// action on each edge, parallel to `children` (`None` for the edge out of a chance node)
    pub edges: Vec<Option<Action>>,
}

impl Subgame {
    pub fn build(rules: &Rules, cfg: &AbstractionConfig, root: BState) -> Subgame {
        if root.street == 0 {
            assert_eq!(root, rules.root(), "a preflop subgame starts at the start of the hand");
        } else {
            assert!(root.raises == 0 && root.street_bet.iter().all(|&b| b == 0), "the root must be the start of a round");
        }
        let mut g = Subgame { rule: LeafRule::for_root(rules, &root), nodes: Vec::new(), children: Vec::new(), edges: Vec::new() };
        g.add(rules, cfg, root, Kind::Decision);
        g
    }

    fn add(&mut self, rules: &Rules, cfg: &AbstractionConfig, s: BState, kind: Kind) -> u32 {
        let id = self.nodes.len() as u32;
        self.nodes.push(SNode { kind, state: s, first: 0, n: 0 });
        let kids: Vec<(Option<Action>, u32)> = match kind {
            Kind::Decision => rules
                .actions(&s, cfg)
                .into_iter()
                .map(|a| {
                    let child = match rules.apply(&s, a) {
                        Next::Terminal(n) => {
                            let k = if rules.n_in_hand(&n) == 1 { Kind::FoldEnd } else { Kind::Showdown };
                            self.add(rules, cfg, n, k)
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
                            self.add(rules, cfg, n, k)
                        }
                    };
                    (Some(a), child)
                })
                .collect(),
            Kind::Chance => vec![(None, self.add(rules, cfg, s, Kind::Decision))],
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
