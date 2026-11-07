//! Kuhn poker: 3 cards (J=0, Q=1, K=2), ante 1, one betting round, bet size 1.
//! Actions: 0 = pass (check/fold), 1 = bet (bet/call). Player 0 acts first.
//! Equilibrium value for player 0 is -1/18.

use crate::game::{Action, ActionList, ChanceList, Game, NodeKind};

#[derive(Clone, Copy, Debug, Default)]
pub struct Kuhn;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KuhnState {
    pub cards: [u8; 2],
    pub dealt: u8,
    pub hist: [u8; 3],
    pub hl: u8,
}

pub const PASS: Action = 0;
pub const BET: Action = 1;

/// Infoset key for a player holding `card` after betting history `hist`.
pub fn key(card: u8, hist: &[u8]) -> u64 {
    let mut k: u64 = 1;
    for &a in hist {
        k = k * 2 + a as u64;
    }
    k * 4 + card as u64
}

impl KuhnState {
    fn h(&self) -> &[u8] {
        &self.hist[..self.hl as usize]
    }
}

impl Game for Kuhn {
    type State = KuhnState;

    fn name(&self) -> &'static str {
        "kuhn"
    }
    fn num_players(&self) -> usize {
        2
    }
    fn num_rounds(&self) -> u8 {
        1
    }
    fn max_abs_utility(&self) -> f64 {
        2.0
    }

    fn root(&self) -> KuhnState {
        KuhnState { cards: [0; 2], dealt: 0, hist: [0; 3], hl: 0 }
    }

    fn kind(&self, s: &KuhnState) -> NodeKind {
        if s.dealt < 2 {
            return NodeKind::Chance;
        }
        match s.h() {
            [PASS, PASS] | [BET, PASS] | [BET, BET] | [_, _, _] => NodeKind::Terminal,
            h => NodeKind::Player(h.len() % 2),
        }
    }

    fn utility(&self, s: &KuhnState, player: usize) -> f64 {
        let showdown = |amount: f64| if s.cards[0] > s.cards[1] { amount } else { -amount };
        let u0 = match s.h() {
            [PASS, PASS] => showdown(1.0),
            [BET, BET] | [PASS, BET, BET] => showdown(2.0),
            [BET, PASS] => 1.0,
            [PASS, BET, PASS] => -1.0,
            h => panic!("utility at non-terminal history {h:?}"),
        };
        if player == 0 {
            u0
        } else {
            -u0
        }
    }

    fn chance_outcomes(&self, s: &KuhnState) -> ChanceList {
        let mut c = ChanceList::new();
        let remaining = 3 - s.dealt as usize;
        for card in 0..3u8 {
            if s.cards[..s.dealt as usize].contains(&card) {
                continue;
            }
            c.push(card, 1.0 / remaining as f64);
        }
        c
    }

    fn legal_actions(&self, _s: &KuhnState) -> ActionList {
        ActionList::from_slice(&[PASS, BET])
    }

    fn apply(&self, s: &KuhnState, a: Action) -> KuhnState {
        let mut n = *s;
        if s.dealt < 2 {
            n.cards[s.dealt as usize] = a;
            n.dealt += 1;
        } else {
            n.hist[s.hl as usize] = a;
            n.hl += 1;
        }
        n
    }

    fn infoset_key(&self, s: &KuhnState) -> u64 {
        let p = s.hl as usize % 2;
        key(s.cards[p], s.h())
    }

    fn round(&self, _s: &KuhnState) -> u8 {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn deal(c0: u8, c1: u8) -> KuhnState {
        let g = Kuhn;
        let s = g.apply(&g.root(), c0);
        g.apply(&s, c1)
    }

    #[test]
    fn root_is_chance_with_three_outcomes() {
        let g = Kuhn;
        assert_eq!(g.kind(&g.root()), NodeKind::Chance);
        let c = g.chance_outcomes(&g.root());
        assert_eq!(c.actions(), &[0, 1, 2]);
        let s = g.apply(&g.root(), 1);
        let c = g.chance_outcomes(&s);
        assert_eq!(c.actions(), &[0, 2]);
        assert_eq!(c.probs(), &[0.5, 0.5]);
    }

    #[test]
    fn terminal_payoffs() {
        let g = Kuhn;
        let s = deal(2, 0); // K vs J
        let play = |s: &KuhnState, h: &[u8]| h.iter().fold(*s, |st, &a| g.apply(&st, a));
        let cases: &[(&[u8], f64)] = &[
            (&[PASS, PASS], 1.0),
            (&[BET, BET], 2.0),
            (&[PASS, BET, BET], 2.0),
            (&[BET, PASS], 1.0),
            (&[PASS, BET, PASS], -1.0),
        ];
        for (h, u) in cases {
            let t = play(&s, h);
            assert_eq!(g.kind(&t), NodeKind::Terminal, "{h:?}");
            assert_eq!(g.utility(&t, 0), *u, "{h:?}");
            assert_eq!(g.utility(&t, 1), -*u, "{h:?}");
        }
        // J vs K: showdowns flip sign, folds do not.
        let s = deal(0, 2);
        assert_eq!(g.utility(&play(&s, &[PASS, PASS]), 0), -1.0);
        assert_eq!(g.utility(&play(&s, &[BET, PASS]), 0), 1.0);
    }

    #[test]
    fn players_alternate_and_keys_unique() {
        let g = Kuhn;
        let s = deal(1, 2);
        assert_eq!(g.kind(&s), NodeKind::Player(0));
        let s1 = g.apply(&s, PASS);
        assert_eq!(g.kind(&s1), NodeKind::Player(1));
        let s2 = g.apply(&s1, BET);
        assert_eq!(g.kind(&s2), NodeKind::Player(0));
        let mut keys = std::collections::HashSet::new();
        for c in 0..3 {
            for h in [&[][..], &[0], &[1], &[0, 1]] {
                assert!(keys.insert(key(c, h)));
            }
        }
        assert_eq!(keys.len(), 12);
    }
}
