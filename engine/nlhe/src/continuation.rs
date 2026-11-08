//! M4 stage 3 (NLHE): continuation strategies at subgame leaves (Supp pp. 23–24,
//! `docs/m4_design.md` §3).
//!
//! * Four strategies per abstract infoset (blueprint node, bucket): the blueprint, and the
//!   blueprint with fold, call or every raise × 5, renormalised (`cfr::search::bias`).
//! * Compressed to one sampled action per (abstract infoset, strategy), computed on demand
//!   (`cfr::search::sample_fixed`, D-SRCH-6), so it is identical at every leaf.
//! * [`Continuations::rollout`] plays the rest of the hand from a blueprint node with each player
//!   on its chosen strategy: the leaf value sample a Monte Carlo solver needs (D-SRCH-8).
//! * [`blueprint_node`] maps a leaf's action history onto the blueprint tree; off-tree histories
//!   give `None` until stage 6 adds the deterministic pseudo-harmonic mapping.

use crate::belief::PolicyFn;
use crate::betting::{Action, Next, Rules};
use crate::blueprint::{Deal, ThreadStats, Tree, DECISION, MAX_A, MAX_P};
use crate::buckets::CardAbstraction;
use crate::config::AbstractionConfig;
use cfr::rng::Rng;
use cfr::search::{bias, sample_fixed, ActKind};

/// Class of action `a` at blueprint decision `node`, read from the child's contributions: fold if
/// the actor folded, raise if it put in more than the largest total so far, else call (or check).
/// Exact because all stacks start equal, so a call is never short.
pub fn action_kind(tree: &Tree, node: u32, a: usize) -> ActKind {
    let nd = &tree.nodes[node as usize];
    let ch = &tree.nodes[tree.child(node, a) as usize];
    let p = nd.player as usize;
    if ch.folded & (1 << p) != 0 {
        return ActKind::Fold;
    }
    let max = *nd.total[..tree.players].iter().max().unwrap();
    if ch.total[p] > max {
        ActKind::Raise
    } else {
        ActKind::Call
    }
}

/// The blueprint node reached by `history` (every action since the start of the hand), or `None`
/// if an action is not in the blueprint's action abstraction.
pub fn blueprint_node(tree: &Tree, rules: &Rules, cfg: &AbstractionConfig, history: &[Action]) -> Option<u32> {
    let (mut node, mut s) = (0u32, rules.root());
    for (k, &a) in history.iter().enumerate() {
        let i = rules.actions(&s, cfg).iter().position(|&x| x == a)?;
        node = tree.child(node, i);
        match rules.apply(&s, a) {
            Next::Decision(n) => s = n,
            Next::Terminal(_) => assert_eq!(k + 1, history.len(), "history continues after the hand ended"),
        }
    }
    Some(node)
}

pub struct Continuations<'a> {
    pub tree: &'a Tree,
    /// blueprint strategy at (node, bucket)
    pub policy: PolicyFn<'a>,
    /// fixed for a whole match, like a stored table
    pub seed: u64,
}

impl Continuations<'_> {
    /// Continuation strategy `k` at abstract infoset (node, bucket), uncompressed.
    pub fn probs(&self, node: u32, bucket: usize, k: usize, out: &mut [f64; MAX_A]) {
        let n = self.tree.nodes[node as usize].n as usize;
        let mut p = [0.0; MAX_A];
        (self.policy)(node, bucket, &mut p);
        let kinds: Vec<ActKind> = (0..n).map(|a| action_kind(self.tree, node, a)).collect();
        bias(&p[..n], &kinds, k, &mut out[..n]);
    }

    /// Compressed continuation strategy `k`: the one action stored for (node, bucket).
    pub fn action(&self, node: u32, bucket: usize, k: usize) -> usize {
        let n = self.tree.nodes[node as usize].n as usize;
        let mut b = [0.0; MAX_A];
        self.probs(node, bucket, k, &mut b);
        sample_fixed(self.seed, (node as u64) << 16 | bucket as u64, k, &b[..n])
    }

    /// One play-out of the rest of the hand from blueprint `node`: the board cards not in `board`
    /// are dealt at random, then player p follows compressed strategy `choice[p]`. Returns the
    /// hand's chip result per player (as `Tree::payoff`).
    #[allow(clippy::too_many_arguments)]
    pub fn rollout(
        &self,
        abs: &dyn CardAbstraction,
        d: &mut Deal,
        node: u32,
        hole: &[(u8, u8)],
        board: &[u8],
        choice: &[u8],
        rng: &mut Rng,
        ts: &mut ThreadStats,
    ) -> [f64; MAX_P] {
        let tree = self.tree;
        let mut used = 0u64;
        for &c in board.iter().chain(hole.iter().flat_map(|h| [&h.0, &h.1])) {
            used |= 1 << c;
        }
        let mut full = [0u8; 5];
        full[..board.len()].copy_from_slice(board);
        for slot in full.iter_mut().skip(board.len()) {
            let c = loop {
                let c = (rng.next_u64() % 52) as u8;
                if used & (1 << c) == 0 {
                    break c;
                }
            };
            used |= 1 << c;
            *slot = c;
        }
        d.set(hole, full);
        let mut node = node;
        loop {
            let nd = tree.nodes[node as usize];
            if nd.kind != DECISION {
                let mut out = [0.0; MAX_P];
                for (i, o) in out.iter_mut().enumerate().take(tree.players) {
                    *o = tree.payoff(node, i, &d.sd);
                }
                return out;
            }
            let p = nd.player as usize;
            let b = d.bucket(abs, p, nd.street as usize, ts);
            node = tree.child(node, self.action(node, b, choice[p] as usize));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buckets::StrengthAbstraction;
    use Action::{Call, Fold, RaiseTo};

    fn cfg() -> AbstractionConfig {
        AbstractionConfig::parse("buckets = 169, 8, 8, 8\npreflop.0 = 0.5\nflop.0 = 0.5, 1.0\nturn.0 = 1.0\nriver.0 = 1.0").unwrap()
    }

    #[test]
    fn action_kinds_match_the_betting_rules_on_a_whole_tree() {
        let (r, c) = (Rules { players: 3, ..Rules::default() }, cfg());
        let tree = Tree::build(&r, &c, [169, 8, 8, 8]);
        fn walk(tree: &Tree, r: &Rules, c: &AbstractionConfig, node: u32, s: crate::betting::BState, seen: &mut usize) {
            for (i, a) in r.actions(&s, c).into_iter().enumerate() {
                let want = match a {
                    Fold => ActKind::Fold,
                    Call => ActKind::Call,
                    RaiseTo(_) => ActKind::Raise,
                };
                assert_eq!(action_kind(tree, node, i), want, "node {node} action {a:?}");
                *seen += 1;
                if let Next::Decision(n) = r.apply(&s, a) {
                    walk(tree, r, c, tree.child(node, i), n, seen);
                }
            }
        }
        let mut seen = 0;
        walk(&tree, &r, &c, 0, r.root(), &mut seen);
        assert_eq!(seen as u64, tree.actions.iter().sum::<u64>());
    }

    #[test]
    fn blueprint_node_replays_on_tree_histories_and_rejects_off_tree_ones() {
        let (r, c) = (Rules { players: 2, ..Rules::default() }, cfg());
        let tree = Tree::build(&r, &c, [169, 8, 8, 8]);
        // SB [fold, limp, 0.5 pot, all-in] -> limp (1); BB [check, raises] -> check (0)
        let flop = tree.child(tree.child(0, 1), 0);
        assert_eq!(blueprint_node(&tree, &r, &c, &[Call, Call]), Some(flop));
        let nd = tree.nodes[flop as usize];
        assert_eq!((nd.street, nd.player), (1, 1));
        assert_eq!(blueprint_node(&tree, &r, &c, &[RaiseTo(333)]), None);
        assert_eq!(blueprint_node(&tree, &r, &c, &[]), Some(0));
    }

    /// Always check/call, whatever the bucket.
    fn calls(tree: &Tree) -> impl Fn(u32, usize, &mut [f64; MAX_A]) + '_ {
        move |node, _b, out| {
            for (a, o) in out.iter_mut().enumerate().take(tree.nodes[node as usize].n as usize) {
                *o = (action_kind(tree, node, a) == ActKind::Call) as u8 as f64;
            }
        }
    }

    #[test]
    fn rollout_with_an_always_call_blueprint_gives_the_exact_showdown() {
        let (r, c) = (Rules { players: 2, ..Rules::default() }, cfg());
        let tree = Tree::build(&r, &c, [169, 8, 8, 8]);
        let pol = calls(&tree);
        let cont = Continuations { tree: &tree, policy: &pol, seed: 1 };
        let abs = StrengthAbstraction { k: 8 };
        let flop = blueprint_node(&tree, &r, &c, &[Call, Call]).unwrap();
        let (mut d, mut ts, mut rng) = (Deal::new(), ThreadStats::default(), Rng::new(3));
        // card = 4 * rank + suit, rank 0 = deuce. Seat 0: AA; seat 1: 7-2 offsuit.
        let hole = [(48, 49), (20, 1)];
        // K 9 5 3 J, no flush or straight: aces win the 200-chip pot
        let board = [46, 30, 13, 7, 39];
        for k in 0..4 {
            let v = cont.rollout(&abs, &mut d, flop, &hole, &board, &[k, k], &mut rng, &mut ts);
            assert_eq!(&v[..2], &[100.0, -100.0]);
        }
        // royal flush on the board: split pot
        let v = cont.rollout(&abs, &mut d, flop, &hole, &[51, 47, 43, 39, 35], &[0, 3], &mut rng, &mut ts);
        assert_eq!(&v[..2], &[0.0, 0.0]);
        // flop only: turn and river are dealt at random, never colliding with known cards
        let mut wins = 0;
        for _ in 0..2000 {
            let v = cont.rollout(&abs, &mut d, flop, &hole, &board[..3], &[0, 0], &mut rng, &mut ts);
            assert_eq!(v[0] + v[1], 0.0);
            assert!(d.board[..3] == board[..3] && d.board[3] != d.board[4]);
            assert!(!d.board.iter().any(|c| [48, 49, 20, 1].contains(c)));
            wins += (v[0] > 0.0) as u32;
        }
        assert!(wins > 1700, "aces won {wins} of 2000"); // AA vs 72o on this flop: ~93%
    }

    #[test]
    fn compressed_strategies_follow_the_biased_distribution() {
        let (r, c) = (Rules { players: 3, ..Rules::default() }, cfg());
        let tree = Tree::build(&r, &c, [169, 8, 8, 8]);
        // UTG preflop: [fold, limp, 0.5 pot, all-in]
        let pol = |_n: u32, _b: usize, out: &mut [f64; MAX_A]| out[..4].copy_from_slice(&[0.4, 0.3, 0.2, 0.1]);
        let cont = Continuations { tree: &tree, policy: &pol, seed: 77 };
        for k in 0..4 {
            let mut want = [0.0; MAX_A];
            cont.probs(0, 0, k, &mut want);
            let n = 40_000;
            let mut cnt = [0usize; 4];
            for bucket in 0..n {
                let a = cont.action(0, bucket, k);
                assert_eq!(a, cont.action(0, bucket, k)); // fixed per infoset
                cnt[a] += 1;
            }
            for a in 0..4 {
                assert!((cnt[a] as f64 / n as f64 - want[a]).abs() < 0.01, "k {k} action {a}");
            }
        }
        // the raise-biased strategy at UTG: raises (0.2, 0.1) x 5
        let mut p = [0.0; MAX_A];
        cont.probs(0, 0, 3, &mut p);
        let s = 0.4 + 0.3 + 1.0 + 0.5;
        assert!((p[2] - 1.0 / s).abs() < 1e-12 && (p[0] - 0.4 / s).abs() < 1e-12);
    }
}
