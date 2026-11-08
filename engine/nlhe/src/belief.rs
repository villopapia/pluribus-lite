//! M4 stage 1: beliefs over the 1,326 private-card pairs of every player (Supp pp. 19–21,
//! `docs/m4_design.md` §1).
//!
//! Each player has a range (combo order of `equity::combo_cards`), uniform at the start of the
//! hand. Bayes' rule (`cfr::search::bayes`) updates it for each action from σ (the blueprint, or
//! the previous search's average strategy) and for each public card. The subgame root weights a
//! joint deal by the product of the ranges, restricted to hands that share no card (D-SRCH-1);
//! this equals the paper's π^σ(h) / Σ π^σ(h′), checked exactly on Leduc
//! (`cfr::leduc_subgame`).

use crate::blueprint::{Deal, ThreadStats, Tree, MAX_A};
use crate::buckets::CardAbstraction;
use crate::equity::{combo_cards, COMBOS};
use cfr::search::bayes;

/// Blueprint (or search) strategy at an abstract infoset: (node, bucket) → action probabilities.
pub type PolicyFn<'a> = &'a dyn Fn(u32, usize, &mut [f64; MAX_A]);

pub struct Beliefs {
    pub ranges: Vec<Vec<f64>>,
}

/// 1 for combos that share no card with `cards`, else 0.
pub fn live_mask(cards: &[u8]) -> Vec<f64> {
    combo_cards().iter().map(|&(a, b)| (!cards.contains(&a) && !cards.contains(&b)) as u8 as f64).collect()
}

impl Beliefs {
    /// 1/1326 for every combo of every player.
    pub fn uniform(players: usize) -> Self {
        Beliefs { ranges: vec![vec![1.0 / COMBOS as f64; COMBOS]; players] }
    }

    /// Bayes update of player `p`'s range for an observed action; `lik[c]` = σ(action | combo c).
    /// Returns the probability of the observation (0: impossible under σ, prior kept, D-SRCH-2).
    pub fn observe_action(&mut self, p: usize, lik: &[f64]) -> f64 {
        bayes(&mut self.ranges[p], lik)
    }

    /// Public cards: every combo holding one of them becomes impossible, for every player.
    pub fn observe_cards(&mut self, cards: &[u8]) {
        let m = live_mask(cards);
        for r in &mut self.ranges {
            bayes(r, &m);
        }
    }

    /// Unnormalised root weight of a joint deal: product of the ranges, 0 if two hands share a card.
    pub fn joint_weight(&self, hands: &[(u8, u8)]) -> f64 {
        let mut used = 0u64;
        let mut w = 1.0;
        for (p, &(a, b)) in hands.iter().enumerate() {
            let m = (1u64 << a) | (1u64 << b);
            if a == b || used & m != 0 {
                return 0.0;
            }
            used |= m;
            w *= self.ranges[p][crate::equity::combo_index(a, b)];
        }
        w
    }
}

/// σ(action `a` | blueprint `node`, bucket of each combo) for all 1,326 combos; 0 for combos that
/// share a card with `board`. `board` holds the cards dealt so far on the node's street
/// (0, 3, 4 or 5 cards).
pub fn action_likelihood(tree: &Tree, abs: &dyn CardAbstraction, policy: PolicyFn, node: u32, a: usize, board: &[u8]) -> Vec<f64> {
    let street = tree.nodes[node as usize].street as usize;
    assert_eq!(board.len(), [0, 3, 4, 5][street], "board does not match the node's street");
    let mut d = Deal::new();
    let mut b5 = [0u8; 5];
    b5[..board.len()].copy_from_slice(board);
    if street == 3 {
        d.set(&[], b5); // river buckets need the board's features
    } else {
        d.board = b5; // flop/turn buckets read only the dealt cards
    }
    let mut ts = ThreadStats::default();
    let mut probs = [0.0; MAX_A];
    combo_cards()
        .into_iter()
        .map(|h| {
            if board.contains(&h.0) || board.contains(&h.1) {
                return 0.0;
            }
            d.hole[0] = h;
            d.bucket[0] = [crate::buckets::preflop_class(h.0, h.1), u16::MAX, u16::MAX, u16::MAX];
            let b = d.bucket(abs, 0, street, &mut ts);
            policy(node, b, &mut probs);
            probs[a]
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::betting::{Action, Next, Rules};
    use crate::buckets::{preflop_class, StrengthAbstraction};
    use crate::config::AbstractionConfig;
    use crate::equity::combo_index;

    /// A synthetic strategy that depends on the bucket, so a wrong bucket shows up.
    fn policy(node: u32, bucket: usize, out: &mut [f64; MAX_A], n: usize) {
        let mut s = 0.0;
        for (a, o) in out.iter_mut().enumerate().take(n) {
            *o = 1.0 + ((bucket + 3 * a + node as usize) % 5) as f64;
            s += *o;
        }
        out.iter_mut().take(n).for_each(|o| *o /= s);
    }

    fn small_tree(players: usize) -> Tree {
        Tree::build(&Rules { players, ..Rules::default() }, &small_cfg(), [169, 8, 8, 8])
    }

    fn small_cfg() -> AbstractionConfig {
        AbstractionConfig::parse("buckets = 169, 8, 8, 8\npreflop.0 = 0.5\nflop.0 = 0.5, 1.0\nturn.0 = 1.0\nriver.0 = 1.0").unwrap()
    }

    /// First node of `street` reached when everyone checks or calls.
    fn calling_line(tree: &Tree, street: u8) -> u32 {
        let (rules, cfg) = (Rules { players: tree.players, ..Rules::default() }, small_cfg());
        let (mut node, mut s) = (0u32, rules.root());
        while s.street < street {
            let i = rules.actions(&s, &cfg).iter().position(|&a| a == Action::Call).unwrap();
            node = tree.child(node, i);
            s = match rules.apply(&s, Action::Call) {
                Next::Decision(n) => n,
                Next::Terminal(_) => panic!("hand ended on the calling line"),
            };
        }
        node
    }

    #[test]
    fn uniform_start_and_board_blocking() {
        let mut b = Beliefs::uniform(3);
        assert!(b.ranges.iter().all(|r| r.len() == 1326 && r.iter().all(|&x| x == 1.0 / 1326.0)));
        b.observe_cards(&[0, 17, 51]); // a flop
        for r in &b.ranges {
            assert_eq!(r.iter().filter(|&&x| x > 0.0).count(), 1176); // C(49, 2)
            assert!((r.iter().sum::<f64>() - 1.0).abs() < 1e-12);
            assert!(r.iter().filter(|&&x| x > 0.0).all(|&x| (x - 1.0 / 1176.0).abs() < 1e-15));
        }
        b.observe_cards(&[30]); // the turn
        assert_eq!(b.ranges[0].iter().filter(|&&x| x > 0.0).count(), 1128); // C(48, 2)
    }

    #[test]
    fn joint_weight_is_the_product_or_zero_on_shared_cards() {
        let mut b = Beliefs::uniform(2);
        b.ranges[0][combo_index(0, 1)] = 0.3;
        b.ranges[1][combo_index(2, 3)] = 0.2;
        assert!((b.joint_weight(&[(0, 1), (2, 3)]) - 0.06).abs() < 1e-15);
        assert_eq!(b.joint_weight(&[(0, 1), (1, 3)]), 0.0);
    }

    #[test]
    fn likelihood_equals_the_policy_at_independently_computed_buckets() {
        let tree = small_tree(3);
        let abs = StrengthAbstraction { k: 8 };
        let pol = |node: u32, b: usize, out: &mut [f64; MAX_A]| policy(node, b, out, tree.nodes[node as usize].n as usize);
        let combos = combo_cards();
        // flop, turn and river nodes on the check/call line
        let boards: [&[u8]; 3] = [&[4, 21, 38], &[4, 21, 38, 50], &[4, 21, 38, 50, 9]];
        for (street, board) in [1u8, 2, 3].into_iter().zip(boards) {
            let node = calling_line(&tree, street);
            let n = tree.nodes[node as usize].n as usize;
            for a in 0..n {
                let lik = action_likelihood(&tree, &abs, &pol, node, a, board);
                for (c, &h) in combos.iter().enumerate() {
                    if board.contains(&h.0) || board.contains(&h.1) {
                        assert_eq!(lik[c], 0.0);
                        continue;
                    }
                    let mut cards = vec![h.0, h.1];
                    cards.extend_from_slice(board);
                    let b = match street {
                        1 => abs.flop(h, &cards[2..5].try_into().unwrap()),
                        2 => abs.turn(h, &cards[2..6].try_into().unwrap()),
                        _ => abs.river(h, &cards[2..7].try_into().unwrap(), &[]),
                    };
                    let mut want = [0.0; MAX_A];
                    pol(node, b as usize, &mut want);
                    assert_eq!(lik[c], want[a], "street {street} combo {h:?}");
                }
            }
        }
        // preflop: the 169 lossless classes
        let lik = action_likelihood(&tree, &abs, &pol, 0, 0, &[]);
        let mut want = [0.0; MAX_A];
        for (c, &h) in combos.iter().enumerate() {
            pol(0, preflop_class(h.0, h.1) as usize, &mut want);
            assert_eq!(lik[c], want[0]);
        }
    }

    #[test]
    fn bayes_update_is_proportional_to_the_likelihood() {
        let tree = small_tree(3);
        let abs = StrengthAbstraction { k: 8 };
        let pol = |node: u32, b: usize, out: &mut [f64; MAX_A]| policy(node, b, out, tree.nodes[node as usize].n as usize);
        let mut bel = Beliefs::uniform(3);
        let lik = action_likelihood(&tree, &abs, &pol, 0, 1, &[]); // UTG's first action
        let z = bel.observe_action(2, &lik);
        assert!((z - lik.iter().sum::<f64>() / 1326.0).abs() < 1e-12);
        for c in [0, 100, 777, 1325] {
            assert!((bel.ranges[2][c] - lik[c] / 1326.0 / z).abs() < 1e-15);
        }
        assert_eq!(bel.ranges[0], vec![1.0 / 1326.0; 1326]); // other players unchanged
    }
}
