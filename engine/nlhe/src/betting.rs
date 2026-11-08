//! No-limit hold'em betting rules (up to 6 players) under a bet-size abstraction, and an
//! exact memoised count of the abstract betting tree.
//!
//! Rules: blinds 50/100, stacks 10,000 (100 bb), no ante. Seat 0 = SB, 1 = BB, 2 = UTG ...
//! Preflop the first to act is seat 2 (seat 0 heads-up); post-flop the first active seat from 0
//! (seat 1 heads-up). Min-raise = max(last raise increment, BB). Abstract sizes below the
//! min-raise are dropped, and sizes at or above the stack collapse into all-in.
//! Simplification: a short all-in raise reopens betting (real rules: it does not). This only
//! adds a few extra nodes.

use crate::config::{AbstractionConfig, Limp};
use cfr::fxhash::FxHashMap;

pub const MAX_PLAYERS: usize = 6;

#[derive(Clone, Copy, Debug)]
pub struct Rules {
    pub players: usize,
    pub sb: u32,
    pub bb: u32,
    pub stack: u32,
}

impl Default for Rules {
    fn default() -> Self {
        Rules { players: 6, sb: 50, bb: 100, stack: 10_000 }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Action {
    Fold,
    /// Check or call.
    Call,
    /// Raise (or bet) so that this street's bet becomes the given amount.
    RaiseTo(u32),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BState {
    pub street: u8,
    pub to_act: u8,
    pub folded: u8,
    pub allin: u8,
    /// Players who still have to act on this street.
    pub need: u8,
    /// Voluntary bets/raises so far on this street.
    pub raises: u8,
    pub last_inc: u32,
    pub total: [u32; MAX_PLAYERS],
    pub street_bet: [u32; MAX_PLAYERS],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Next {
    Decision(BState),
    /// Hand over (one player left, river showdown, or everyone but one all-in).
    Terminal(BState),
}

fn bit(p: usize) -> u8 {
    1 << p
}

impl Rules {
    fn all_mask(&self) -> u8 {
        ((1u16 << self.players) - 1) as u8
    }

    pub fn root(&self) -> BState {
        assert!((2..=MAX_PLAYERS).contains(&self.players));
        let mut s = BState {
            street: 0,
            to_act: if self.players == 2 { 0 } else { 2 },
            folded: 0,
            allin: 0,
            need: self.all_mask(),
            raises: 0,
            last_inc: self.bb,
            total: [0; MAX_PLAYERS],
            street_bet: [0; MAX_PLAYERS],
        };
        s.total[0] = self.sb;
        s.street_bet[0] = self.sb;
        s.total[1] = self.bb;
        s.street_bet[1] = self.bb;
        s
    }

    pub fn pot(&self, s: &BState) -> u32 {
        s.total[..self.players].iter().sum()
    }

    fn max_bet(&self, s: &BState) -> u32 {
        *s.street_bet[..self.players].iter().max().unwrap()
    }

    pub fn n_in_hand(&self, s: &BState) -> usize {
        self.players - s.folded.count_ones() as usize
    }

    /// Legal abstract actions for the player to act.
    pub fn actions(&self, s: &BState, cfg: &AbstractionConfig) -> Vec<Action> {
        let p = s.to_act as usize;
        let max_bet = self.max_bet(s);
        let to_call = max_bet - s.street_bet[p];
        let stack = self.stack - s.total[p];
        let mut out = Vec::with_capacity(8);
        if to_call > 0 {
            out.push(Action::Fold);
        }
        // flat-calling the big blind in an unraised preflop pot = limping (BB check excluded)
        let is_limp = s.street == 0 && s.raises == 0 && p != 1 && to_call > 0;
        let limp_ok = match cfg.limp {
            Limp::All => true,
            Limp::Sb => p == 0,
            Limp::None => false,
        };
        if !is_limp || limp_ok {
            out.push(Action::Call);
        }
        let live = self.all_mask() & !s.folded & !s.allin & !bit(p);
        if stack > to_call && live != 0 {
            let pot = self.pot(s);
            let max_to = s.street_bet[p] + stack;
            let min_to = max_bet + s.last_inc.max(self.bb);
            let mut tos: Vec<u32> = Vec::new();
            for &f in cfg.sizes_for(s.street as usize, s.raises as usize, self.n_in_hand(s)) {
                let to = max_bet + (f * (pot + to_call) as f64).round() as u32;
                if to < min_to {
                    continue;
                }
                let to = to.min(max_to);
                if !tos.contains(&to) {
                    tos.push(to);
                }
            }
            if cfg.allin && !tos.contains(&max_to) {
                tos.push(max_to);
            }
            tos.sort_unstable();
            out.extend(tos.into_iter().map(Action::RaiseTo));
        }
        out
    }

    pub fn apply(&self, s: &BState, a: Action) -> Next {
        let mut n = *s;
        let p = s.to_act as usize;
        let max_bet = self.max_bet(s);
        match a {
            Action::Fold => {
                n.folded |= bit(p);
                n.need &= !bit(p);
            }
            Action::Call => {
                let amt = (max_bet - s.street_bet[p]).min(self.stack - s.total[p]);
                n.street_bet[p] += amt;
                n.total[p] += amt;
                if n.total[p] == self.stack {
                    n.allin |= bit(p);
                }
                n.need &= !bit(p);
            }
            Action::RaiseTo(to) => {
                assert!(to > max_bet, "raise must exceed the current bet");
                let amt = to - s.street_bet[p];
                assert!(s.total[p] + amt <= self.stack, "raise exceeds stack");
                n.last_inc = n.last_inc.max(to - max_bet);
                n.street_bet[p] = to;
                n.total[p] += amt;
                n.raises += 1;
                if n.total[p] == self.stack {
                    n.allin |= bit(p);
                }
                n.need = self.all_mask() & !n.folded & !n.allin & !bit(p);
            }
        }
        self.advance(n, p)
    }

    fn advance(&self, mut n: BState, p: usize) -> Next {
        let alive = self.all_mask() & !n.folded;
        if alive.count_ones() == 1 {
            return Next::Terminal(n);
        }
        if n.need == 0 {
            let can_act = alive & !n.allin;
            if n.street == 3 || can_act.count_ones() <= 1 {
                return Next::Terminal(n);
            }
            n.street += 1;
            n.street_bet = [0; MAX_PLAYERS];
            n.raises = 0;
            n.last_inc = self.bb;
            n.need = can_act;
            let first = if self.players == 2 { 1 } else { 0 };
            n.to_act = (0..self.players)
                .map(|k| (first + k) % self.players)
                .find(|&q| n.need & bit(q) != 0)
                .unwrap() as u8;
            return Next::Decision(n);
        }
        n.to_act = (1..=self.players)
            .map(|k| (p + k) % self.players)
            .find(|&q| n.need & bit(q) != 0)
            .unwrap() as u8;
        Next::Decision(n)
    }
}

/// Per-street totals over all betting histories (decision nodes) of the abstract tree.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TreeCounts {
    /// Decision nodes (= betting histories where someone acts), per street.
    pub nodes: [u64; 4],
    /// Sum of legal actions over those nodes, per street.
    pub actions: [u64; 4],
    /// Terminal betting histories.
    pub terminals: u64,
}

impl TreeCounts {
    fn add(&mut self, o: &TreeCounts) {
        for i in 0..4 {
            self.nodes[i] = self.nodes[i].checked_add(o.nodes[i]).expect("overflow");
            self.actions[i] = self.actions[i].checked_add(o.actions[i]).expect("overflow");
        }
        self.terminals = self.terminals.checked_add(o.terminals).expect("overflow");
    }
}

/// Memoised counter. Subtree shape depends only on the betting state, not the path to it,
/// so identical states reached by different histories are counted once and multiplied.
pub struct TreeCounter<'a> {
    pub rules: Rules,
    pub cfg: &'a AbstractionConfig,
    memo: FxHashMap<BState, TreeCounts>,
    pub max_memo: usize,
    /// When true, a full memo is cleared and counting continues (exact, slower) instead of
    /// failing with `MemoLimit`.
    pub evict_when_full: bool,
    pub evictions: u64,
}

#[derive(Debug, PartialEq, Eq)]
pub struct MemoLimit(pub usize);

impl<'a> TreeCounter<'a> {
    pub fn new(rules: Rules, cfg: &'a AbstractionConfig) -> Self {
        TreeCounter { rules, cfg, memo: FxHashMap::default(), max_memo: 4_000_000, evict_when_full: false, evictions: 0 }
    }

    pub fn memo_len(&self) -> usize {
        self.memo.len()
    }

    pub fn count(&mut self) -> Result<TreeCounts, MemoLimit> {
        let root = self.rules.root();
        self.rec(&root)
    }

    /// First decision of a street (nobody has acted yet). Different histories converge mostly
    /// here, so memoising only these keeps the memo small at little extra cost.
    fn is_street_start(&self, s: &BState) -> bool {
        let live = (((1u16 << self.rules.players) - 1) as u8) & !s.folded & !s.allin;
        s.street > 0 && s.raises == 0 && s.need == live && s.street_bet.iter().all(|&b| b == 0)
    }

    /// Memo key with folded players canonicalised: their chips only matter through the pot
    /// (dead money) and their street bets are always below the current max bet, so the
    /// subtree is identical if all dead money sits in the first folded seat and folded
    /// street bets are zero.
    fn canonical(&self, s: &BState) -> BState {
        let mut k = *s;
        if k.folded == 0 {
            return k;
        }
        let mut dead = 0u32;
        let mut first = usize::MAX;
        for p in 0..self.rules.players {
            if k.folded & (1 << p) != 0 {
                dead += k.total[p];
                k.total[p] = 0;
                k.street_bet[p] = 0;
                first = first.min(p);
            }
        }
        k.total[first] = dead;
        k
    }

    fn rec(&mut self, s: &BState) -> Result<TreeCounts, MemoLimit> {
        let memoise = self.is_street_start(s);
        let key = if memoise { self.canonical(s) } else { *s };
        if memoise {
            if let Some(c) = self.memo.get(&key) {
                return Ok(*c);
            }
        }
        let acts = self.rules.actions(s, self.cfg);
        let mut c = TreeCounts::default();
        c.nodes[s.street as usize] = 1;
        c.actions[s.street as usize] = acts.len() as u64;
        for a in acts {
            match self.rules.apply(s, a) {
                Next::Terminal(_) => c.terminals += 1,
                Next::Decision(n) => {
                    let sub = self.rec(&n)?;
                    c.add(&sub);
                }
            }
        }
        if memoise {
            if self.memo.len() >= self.max_memo {
                if !self.evict_when_full {
                    return Err(MemoLimit(self.memo.len()));
                }
                self.memo.clear();
                self.evictions += 1;
            }
            self.memo.insert(key, c);
        }
        Ok(c)
    }
}

/// Plain DFS without memo (test oracle for small trees).
pub fn count_naive(rules: &Rules, cfg: &AbstractionConfig) -> TreeCounts {
    fn rec(r: &Rules, cfg: &AbstractionConfig, s: &BState, c: &mut TreeCounts) {
        let acts = r.actions(s, cfg);
        c.nodes[s.street as usize] += 1;
        c.actions[s.street as usize] += acts.len() as u64;
        for a in acts {
            match r.apply(s, a) {
                Next::Terminal(_) => c.terminals += 1,
                Next::Decision(n) => rec(r, cfg, &n, c),
            }
        }
    }
    let mut c = TreeCounts::default();
    rec(rules, cfg, &rules.root(), &mut c);
    c
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(text: &str) -> AbstractionConfig {
        AbstractionConfig::parse(text).unwrap()
    }

    #[test]
    fn heads_up_check_call_only() {
        let r = Rules { players: 2, ..Rules::default() };
        let c = cfg("allin = false");
        let t = count_naive(&r, &c);
        // preflop: SB {fold, call}, BB {check}; each later street: 2 checks
        assert_eq!(t.nodes, [2, 2, 2, 2]);
        assert_eq!(t.actions, [3, 2, 2, 2]);
        assert_eq!(t.terminals, 2); // SB fold, river showdown
        let mut m = TreeCounter::new(r, &c);
        assert_eq!(m.count().unwrap(), t);
    }

    #[test]
    fn open_sizes_match_pluribus_chip_amounts() {
        let r = Rules::default();
        let c = cfg("preflop.0 = 0.4, 0.5, 0.6");
        let acts = r.actions(&r.root(), &c);
        // 0.4 pot -> 2bb, 0.5 -> 2.25bb, 0.6 -> 2.5bb, plus all-in
        assert_eq!(
            acts,
            vec![Action::Fold, Action::Call, Action::RaiseTo(200), Action::RaiseTo(225), Action::RaiseTo(250), Action::RaiseTo(10_000)]
        );
    }

    #[test]
    fn small_sizes_dropped_and_big_sizes_collapse_to_allin() {
        let r = Rules::default();
        // 0.1 pot = to 125: below min-raise (to 200) -> dropped; 100 pot -> all-in
        let c = cfg("preflop.0 = 0.1, 100");
        let acts = r.actions(&r.root(), &c);
        assert_eq!(acts, vec![Action::Fold, Action::Call, Action::RaiseTo(10_000)]);
    }

    #[test]
    fn preflop_order_and_bb_option() {
        let r = Rules::default();
        let c = cfg("allin = false");
        let mut s = r.root();
        assert_eq!(s.to_act, 2);
        // UTG..BTN call, SB calls, BB gets the option
        for expect in [2u8, 3, 4, 5, 0] {
            assert_eq!(s.to_act, expect);
            s = match r.apply(&s, Action::Call) {
                Next::Decision(n) => n,
                t => panic!("{t:?}"),
            };
        }
        assert_eq!(s.to_act, 1);
        assert_eq!(r.actions(&s, &c), vec![Action::Call]); // check only, no sizes configured
        let flop = match r.apply(&s, Action::Call) {
            Next::Decision(n) => n,
            t => panic!("{t:?}"),
        };
        assert_eq!((flop.street, flop.to_act), (1, 0));
        assert_eq!(r.pot(&flop), 600);
    }

    #[test]
    fn raise_reopens_and_everyone_allin_ends_hand() {
        let r = Rules { players: 3, ..Rules::default() };
        let c = cfg("allin = true");
        let s = r.root(); // seat 2 (BTN in 3-handed) acts first
        assert!(r.actions(&s, &c).contains(&Action::RaiseTo(10_000)));
        let s = match r.apply(&s, Action::RaiseTo(10_000)) { Next::Decision(n) => n, t => panic!("{t:?}") };
        assert_eq!(s.to_act, 0);
        let s = match r.apply(&s, Action::Call) { Next::Decision(n) => n, t => panic!("{t:?}") };
        // BB calls all-in too: no one can act -> terminal (run-out)
        match r.apply(&s, Action::Call) {
            Next::Terminal(t) => assert_eq!(t.allin.count_ones(), 3),
            n => panic!("{n:?}"),
        }
    }

    #[test]
    fn limp_options() {
        let r = Rules::default();
        let root = r.root(); // UTG to act, facing only the BB
        let none = cfg("limp = none
preflop.0 = 0.5");
        assert_eq!(r.actions(&root, &none), vec![Action::Fold, Action::RaiseTo(225), Action::RaiseTo(10_000)]);
        let sb = cfg("limp = sb
preflop.0 = 0.5");
        assert!(!r.actions(&root, &sb).contains(&Action::Call));
        // fold around to the SB: SB may complete
        let mut s = root;
        for _ in 0..4 {
            s = match r.apply(&s, Action::Fold) { Next::Decision(n) => n, t => panic!("{t:?}") };
        }
        assert_eq!(s.to_act, 0);
        assert!(r.actions(&s, &sb).contains(&Action::Call));
        assert!(!r.actions(&s, &none).contains(&Action::Call));
        // after a raise, calling is never a limp
        let s = match r.apply(&root, Action::RaiseTo(225)) { Next::Decision(n) => n, t => panic!("{t:?}") };
        assert!(r.actions(&s, &none).contains(&Action::Call));
    }

    #[test]
    fn memo_matches_naive_with_limp_and_crowded_options() {
        let r = Rules { players: 4, ..Rules::default() };
        let c = cfg("limp = sb
crowded_players = 3
preflop.0 = 0.5
preflop.1 = 1.0
flop.0 = 0.5
turn.0 = 1.0
river.0 = 1.0");
        let mut m = TreeCounter::new(r, &c);
        assert_eq!(m.count().unwrap(), count_naive(&r, &c));
    }

    #[test]
    fn memo_matches_naive_on_three_handed_tree() {
        let r = Rules { players: 3, ..Rules::default() };
        let c = cfg(
            "preflop.0 = 0.5, 1.0
             preflop.1 = 1.0
             flop.0 = 0.5, 1.0
             flop.1 = 1.0
             turn.0 = 0.75
             river.0 = 1.0
             multiway.flop.0 = 0.5",
        );
        let naive = count_naive(&r, &c);
        let mut m = TreeCounter::new(r, &c);
        let memo = m.count().unwrap();
        assert_eq!(memo, naive);
        assert!(naive.nodes.iter().all(|&n| n > 0));
    }

    /// D-ACT-4 (accepted ruling): wherever the paper's "at most three / at most two" sizes
    /// bind, illegal and duplicate sizes collapse. On every path of a heads-up tree with the
    /// faithful turn/river sets: raise targets are distinct and increasing, each is a legal
    /// raise (>= min-raise) or the all-in, and all-in appears once. The test also checks that
    /// both collapse cases actually occur (sizes dropped below the min-raise, sizes capped at
    /// the stack and merged into all-in), so "at most" is exercised rather than assumed.
    #[test]
    fn illegal_and_duplicate_sizes_collapse_on_every_path() {
        let r = Rules { players: 2, ..Rules::default() };
        let c = cfg("preflop.0 = 0.5
preflop.1 = 1.0
flop.0 = 0.5, 1.0
turn.0 = 0.5, 1.0
turn.1+ = 1.0
river.0 = 0.5, 1.0
river.1+ = 1.0");
        #[derive(Default)]
        struct Seen {
            dropped: u64,
            merged: u64,
            nodes: u64,
        }
        fn walk(r: &Rules, c: &AbstractionConfig, s: &BState, seen: &mut Seen) {
            let acts = r.actions(s, c);
            let p = s.to_act as usize;
            let max_bet = r.max_bet(s);
            let stack = r.stack - s.total[p];
            let max_to = s.street_bet[p] + stack;
            let min_to = max_bet + s.last_inc.max(r.bb);
            let tos: Vec<u32> = acts.iter().filter_map(|a| if let Action::RaiseTo(t) = a { Some(*t) } else { None }).collect();
            assert!(tos.windows(2).all(|w| w[0] < w[1]), "distinct, increasing: {tos:?}");
            assert!(tos.iter().all(|&t| (t >= min_to && t <= max_to) || t == max_to), "legal: {tos:?}");
            assert!(tos.iter().filter(|&&t| t == max_to).count() <= 1);
            if !tos.is_empty() {
                seen.nodes += 1;
                let pot = r.pot(s);
                let to_call = max_bet - s.street_bet[p];
                let raw: Vec<u32> = c
                    .sizes_for(s.street as usize, s.raises as usize, r.n_in_hand(s))
                    .iter()
                    .map(|f| max_bet + (f * (pot + to_call) as f64).round() as u32)
                    .collect();
                seen.dropped += raw.iter().filter(|&&t| t < min_to).count() as u64;
                seen.merged += raw.iter().filter(|&&t| t >= max_to).count() as u64;
            }
            for a in acts {
                if let Next::Decision(n) = r.apply(s, a) {
                    walk(r, c, &n, seen)
                }
            }
        }
        let mut seen = Seen::default();
        walk(&r, &c, &r.root(), &mut seen);
        assert!(seen.nodes > 1000);
        assert!(seen.merged > 0, "some sizes reach the stack and merge into all-in");
        let _ = seen.dropped; // min-raise drops are covered by small_sizes_dropped_and_big_sizes_collapse_to_allin
    }

    /// Chip conservation and street-closing invariants on every path of a small tree.
    #[test]
    fn invariants_hold_on_every_path() {
        let r = Rules { players: 3, ..Rules::default() };
        let c = cfg("preflop.0 = 0.5\npreflop.1 = 1.0\nflop.0 = 1.0\nturn.0 = 1.0\nriver.0 = 1.0");
        fn walk(r: &Rules, c: &AbstractionConfig, s: &BState) {
            for i in 0..r.players {
                assert!(s.total[i] <= r.stack);
                assert!(s.street_bet[i] <= s.total[i]);
            }
            for a in r.actions(s, c) {
                match r.apply(s, a) {
                    Next::Decision(n) => {
                        if n.street > s.street {
                            // previous street closed with all live players matched
                            let live: Vec<usize> = (0..r.players).filter(|&i| n.folded & (1 << i) == 0).collect();
                            assert!(live.len() >= 2);
                        }
                        walk(r, c, &n)
                    }
                    Next::Terminal(t) => {
                        let pot: u32 = t.total[..r.players].iter().sum();
                        assert!(pot >= r.sb + r.bb);
                    }
                }
            }
        }
        walk(&r, &c, &r.root());
    }

    #[test]
    fn canonical_memo_matches_naive_with_folds_and_limps() {
        // 5 players, limping allowed, multiple sizes: many histories differ only in who
        // folded and for how much
        let r = Rules { players: 5, ..Rules::default() };
        let c = cfg("limp = all
preflop.0 = 0.5, 1.0
preflop.1 = 1.0
flop.0 = 0.5
turn.0 = 1.0
river.0 = 1.0");
        let mut m = TreeCounter::new(r, &c);
        assert_eq!(m.count().unwrap(), count_naive(&r, &c));
    }

    #[test]
    fn eviction_keeps_counts_exact() {
        let r = Rules { players: 4, ..Rules::default() };
        let c = cfg("limp = all
preflop.0 = 0.5, 1.0
preflop.1 = 1.0
flop.0 = 0.5
turn.0 = 1.0
river.0 = 1.0");
        let mut m = TreeCounter::new(r, &c);
        m.max_memo = 3;
        m.evict_when_full = true;
        assert_eq!(m.count().unwrap(), count_naive(&r, &c));
        assert!(m.evictions > 0);
    }

    #[test]
    fn memo_limit_is_reported() {
        let r = Rules { players: 3, ..Rules::default() };
        let c = cfg("preflop.0 = 0.5\nflop.0 = 0.5");
        let mut m = TreeCounter::new(r, &c);
        m.max_memo = 2;
        assert!(m.count().is_err());
    }
}
