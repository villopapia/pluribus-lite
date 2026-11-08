//! Leduc hold'em: 6 cards (J,J,Q,Q,K,K; card index c has rank c/2), ante 1 each,
//! two betting rounds with fixed raise sizes 2 then 4, at most 2 raises per round,
//! player 0 acts first in each round. One public board card is dealt after round 1.
//! Showdown: pairing the board wins, else higher rank, else split.
//!
//! Actions: 0 = fold, 1 = check/call, 2 = bet/raise.
//! Infoset keys use card *ranks* (suits are strategically irrelevant): 288 infosets.

use crate::game::{Action, ActionList, ChanceList, Game, NodeKind};

pub const FOLD: Action = 0;
pub const CALL: Action = 1;
pub const RAISE: Action = 2;
const NONE: u8 = 255;
const HIST_LEN: usize = 10;

#[derive(Clone, Copy, Debug, Default)]
pub struct Leduc;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LeducState {
    pub cards: [u8; 2],
    pub board: u8,
    pub dealt: u8,
    pub round: u8,
    pub contrib: [i32; 2],
    pub raises: u8,
    pub round_actions: u8,
    pub to_act: u8,
    pub folded: u8,
    pub showdown: bool,
    /// Action history: action+1 for actions, 0 as round separator.
    pub hist: [u8; HIST_LEN],
    pub hl: u8,
}

#[inline]
fn rank(card: u8) -> u8 {
    card / 2
}

/// What player `p` knows at `s` (betting history, board rank, own rank). Equals the infoset key
/// when `p` is to act. Keys are below 4^13.
pub fn key_for(s: &LeducState, p: usize) -> u64 {
    let mut k: u64 = 1;
    for &h in &s.hist[..s.hl as usize] {
        k = k * 4 + h as u64;
    }
    let board_code = if s.board == NONE { 0 } else { rank(s.board) as u64 + 1 };
    (k * 4 + board_code) * 4 + rank(s.cards[p]) as u64
}

impl LeducState {
    fn outstanding(&self) -> bool {
        let me = self.to_act as usize;
        self.contrib[1 - me] > self.contrib[me]
    }
    fn used(&self, card: u8) -> bool {
        self.cards[..self.dealt as usize].contains(&card) || self.board == card
    }
}

impl Game for Leduc {
    type State = LeducState;

    fn name(&self) -> &'static str {
        "leduc"
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

    fn root(&self) -> LeducState {
        LeducState {
            cards: [0; 2],
            board: NONE,
            dealt: 0,
            round: 0,
            contrib: [1, 1],
            raises: 0,
            round_actions: 0,
            to_act: 0,
            folded: NONE,
            showdown: false,
            hist: [0; HIST_LEN],
            hl: 0,
        }
    }

    fn kind(&self, s: &LeducState) -> NodeKind {
        if s.folded != NONE || s.showdown {
            NodeKind::Terminal
        } else if s.dealt < 2 || (s.round == 1 && s.board == NONE) {
            NodeKind::Chance
        } else {
            NodeKind::Player(s.to_act as usize)
        }
    }

    fn utility(&self, s: &LeducState, player: usize) -> f64 {
        let u0: f64 = if s.folded != NONE {
            let loser = s.folded as usize;
            let amt = s.contrib[loser] as f64;
            if loser == 0 {
                -amt
            } else {
                amt
            }
        } else {
            assert!(s.showdown, "utility at non-terminal state");
            let b = rank(s.board);
            let strength = |c: u8| if rank(c) == b { 10 + rank(c) } else { rank(c) };
            let (a, c) = (strength(s.cards[0]), strength(s.cards[1]));
            let amt = s.contrib[0] as f64; // equal at showdown
            match a.cmp(&c) {
                std::cmp::Ordering::Greater => amt,
                std::cmp::Ordering::Less => -amt,
                std::cmp::Ordering::Equal => 0.0,
            }
        };
        if player == 0 {
            u0
        } else {
            -u0
        }
    }

    fn chance_outcomes(&self, s: &LeducState) -> ChanceList {
        let mut c = ChanceList::new();
        let remaining = (0..6u8).filter(|&x| !s.used(x)).count();
        for card in 0..6u8 {
            if !s.used(card) {
                c.push(card, 1.0 / remaining as f64);
            }
        }
        c
    }

    fn legal_actions(&self, s: &LeducState) -> ActionList {
        if s.outstanding() {
            if s.raises < 2 {
                ActionList::from_slice(&[FOLD, CALL, RAISE])
            } else {
                ActionList::from_slice(&[FOLD, CALL])
            }
        } else {
            ActionList::from_slice(&[CALL, RAISE])
        }
    }

    fn apply(&self, s: &LeducState, a: Action) -> LeducState {
        let mut n = *s;
        if s.dealt < 2 {
            n.cards[s.dealt as usize] = a;
            n.dealt += 1;
            return n;
        }
        if s.round == 1 && s.board == NONE {
            n.board = a;
            return n;
        }
        let me = s.to_act as usize;
        let opp = 1 - me;
        n.hist[n.hl as usize] = a + 1;
        n.hl += 1;
        match a {
            FOLD => {
                debug_assert!(s.outstanding());
                n.folded = me as u8;
            }
            CALL => {
                let outstanding = s.outstanding();
                n.contrib[me] = s.contrib[opp];
                let ends = outstanding || s.round_actions >= 1;
                n.round_actions += 1;
                if ends {
                    if s.round == 0 {
                        n.round = 1;
                        n.raises = 0;
                        n.round_actions = 0;
                        n.to_act = 0;
                        n.hist[n.hl as usize] = 0;
                        n.hl += 1;
                    } else {
                        n.showdown = true;
                    }
                } else {
                    n.to_act = opp as u8;
                }
            }
            RAISE => {
                debug_assert!(s.raises < 2);
                let size = if s.round == 0 { 2 } else { 4 };
                n.contrib[me] = s.contrib[opp] + size;
                n.raises += 1;
                n.round_actions += 1;
                n.to_act = opp as u8;
            }
            _ => panic!("illegal action {a}"),
        }
        n
    }

    fn infoset_key(&self, s: &LeducState) -> u64 {
        key_for(s, s.to_act as usize)
    }

    fn round(&self, s: &LeducState) -> u8 {
        s.round
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    fn play(s: &LeducState, actions: &[Action]) -> LeducState {
        let g = Leduc;
        actions.iter().fold(*s, |st, &a| g.apply(&st, a))
    }

    #[test]
    fn deal_and_round_structure() {
        let g = Leduc;
        let s = play(&g.root(), &[0, 4]); // J vs K
        assert_eq!(g.kind(&s), NodeKind::Player(0));
        assert_eq!(g.legal_actions(&s).as_slice(), &[CALL, RAISE]);
        let s = play(&s, &[CALL, CALL]); // check-check
        assert_eq!(g.kind(&s), NodeKind::Chance);
        assert_eq!(g.chance_outcomes(&s).len(), 4);
        let s = g.apply(&s, 1); // board J -> player 0 pairs
        assert_eq!(g.kind(&s), NodeKind::Player(0));
        let s = play(&s, &[RAISE, RAISE]);
        assert_eq!(g.kind(&s), NodeKind::Player(0));
        // after two raises no more raising
        assert_eq!(g.legal_actions(&s).as_slice(), &[FOLD, CALL]);
        let s = g.apply(&s, CALL);
        assert_eq!(g.kind(&s), NodeKind::Terminal);
        // ante 1; p0 bets 4 -> 5; p1 raises 4 more -> 9; p0 calls -> 9
        assert_eq!(s.contrib, [9, 9]);
        assert_eq!(g.utility(&s, 0), 9.0);
    }

    #[test]
    fn raise_cap_round_one() {
        let g = Leduc;
        let s = play(&g.root(), &[0, 2]);
        let s = play(&s, &[RAISE, RAISE]);
        assert_eq!(s.contrib, [3, 5]);
        assert_eq!(g.legal_actions(&s).as_slice(), &[FOLD, CALL]);
        let t = g.apply(&s, FOLD);
        assert_eq!(g.kind(&t), NodeKind::Terminal);
        assert_eq!(g.utility(&t, 0), -3.0);
        assert_eq!(g.utility(&t, 1), 3.0);
    }

    #[test]
    fn showdown_split_and_high_card() {
        let g = Leduc;
        // Q vs Q, board K -> split
        let s = play(&g.root(), &[2, 3, CALL, CALL, 4, CALL, CALL]);
        assert_eq!(g.kind(&s), NodeKind::Terminal);
        assert_eq!(g.utility(&s, 0), 0.0);
        // J vs Q, board K -> Q wins 1
        let s = play(&g.root(), &[0, 2, CALL, CALL, 4, CALL, CALL]);
        assert_eq!(g.utility(&s, 0), -1.0);
        // J vs K, board J -> pair of J wins over K
        let s = play(&g.root(), &[0, 4, CALL, CALL, 1, CALL, CALL]);
        assert_eq!(g.utility(&s, 0), 1.0);
    }

    fn walk(g: &Leduc, s: &LeducState, keys: &mut HashSet<u64>, terminals: &mut usize) {
        match g.kind(s) {
            NodeKind::Terminal => *terminals += 1,
            NodeKind::Chance => {
                let c = g.chance_outcomes(s);
                let total: f64 = c.probs().iter().sum();
                assert!((total - 1.0).abs() < 1e-12);
                for &a in c.actions() {
                    walk(g, &g.apply(s, a), keys, terminals);
                }
            }
            NodeKind::Player(_) => {
                keys.insert(g.infoset_key(s));
                for &a in g.legal_actions(s).as_slice() {
                    walk(g, &g.apply(s, a), keys, terminals);
                }
            }
        }
    }

    #[test]
    fn infoset_count_is_288() {
        let g = Leduc;
        let mut keys = HashSet::new();
        let mut terminals = 0;
        walk(&g, &g.root(), &mut keys, &mut terminals);
        assert_eq!(keys.len(), 288);
        assert!(terminals > 0);
    }
}
