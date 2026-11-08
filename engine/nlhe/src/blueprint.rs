//! M3 blueprint trainer (6-max NLHE, abstract game).
//!
//! * Tree: the explicit abstract betting tree of a config (one node per betting history),
//!   with per-node contributions so payoffs need no replay.
//! * Storage: flat `i32` regrets for every (node, bucket, action) and `u32` average-strategy
//!   counts for preflop nodes only. Shared across threads with relaxed atomics; concurrent
//!   read-modify-write races may drop an update, as in the paper's lock-free training.
//! * Algorithm: external-sampling MCCFR with linear discounting and negative-regret pruning,
//!   structured like M1's `mccfr.rs` (Algorithm 1 of the Pluribus supplement, from memory
//!   pending the PDF check), with integer regrets (`cfr::intregret`).
//!   - Iteration t, for each player i: fresh deal; UPDATE-STRATEGY on the preflop every
//!     `strategy_interval` iterations; traverse pruned with probability 1 - explore_prob once
//!     t > prune_after, else unpruned.
//!   - Discount regrets and counts by k/(k+1) whenever t is a multiple of discount_every,
//!     while t < lcfr_until.
//!   - A traverser that has folded gets its fixed payoff immediately.
//! * Determinism: each (t, player) uses its own RNG stream, so single-threaded runs are
//!   exactly reproducible and resume bit-identically. Multi-threaded runs are not.

use crate::betting::{BState, Next, Rules};
use crate::buckets::{preflop_class, CardAbstraction};
use crate::config::AbstractionConfig;
use crate::equity::{HeroRiver, RiverBatch};
use crate::eval_lut::{LHand, Lut};
use cfr::intregret::{add_regret, discount_count, discount_regret, RegretStats};
use cfr::rng::Rng;
use std::fs::{self, File};
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicPtr, AtomicU32, AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::Instant;

pub const DECISION: u8 = 0;
pub const FOLD_END: u8 = 1;
pub const SHOWDOWN: u8 = 2;
pub const MAX_P: usize = 6;
const MAX_A: usize = 8;

#[derive(Clone, Copy, Debug)]
pub struct TNode {
    pub first: u32,
    pub n: u8,
    pub street: u8,
    pub player: u8,
    pub folded: u8,
    pub kind: u8,
    pub total: [u16; MAX_P],
}

pub struct Tree {
    pub players: usize,
    pub nodes: Vec<TNode>,
    pub children: Vec<u32>,
    /// average-strategy offset per node (preflop decision nodes; u64::MAX otherwise)
    pub aoff: Vec<u64>,
    pub n_regrets: u64,
    pub n_avg: u64,
    pub buckets: [u32; 4],
    pub decision_nodes: [u64; 4],
    /// actions summed over decision nodes, per street
    pub actions: [u64; 4],
}

impl Tree {
    pub fn build(rules: &Rules, cfg: &AbstractionConfig, buckets: [u32; 4]) -> Tree {
        assert_eq!(buckets[0], 169);
        let mut t = Tree {
            players: rules.players,
            nodes: Vec::new(),
            children: Vec::new(),
            aoff: Vec::new(),
            n_regrets: 0,
            n_avg: 0,
            buckets,
            decision_nodes: [0; 4],
            actions: [0; 4],
        };
        let root = rules.root();
        t.add(rules, cfg, &root, DECISION);
        let (mut r, mut a) = (0u64, 0u64);
        t.aoff = vec![u64::MAX; t.nodes.len()];
        for (i, n) in t.nodes.iter().enumerate() {
            if n.kind != DECISION {
                continue;
            }
            r += n.n as u64 * buckets[n.street as usize] as u64;
            if n.street == 0 {
                t.aoff[i] = a;
                a += n.n as u64 * 169;
            }
        }
        t.n_regrets = r;
        t.n_avg = a;
        t
    }

    fn add(&mut self, rules: &Rules, cfg: &AbstractionConfig, s: &BState, kind: u8) -> u32 {
        let id = self.nodes.len() as u32;
        let mut total = [0u16; MAX_P];
        for p in 0..rules.players {
            total[p] = s.total[p] as u16;
        }
        self.nodes.push(TNode { first: 0, n: 0, street: s.street, player: s.to_act, folded: s.folded, kind, total });
        if kind != DECISION {
            return id;
        }
        let acts = rules.actions(s, cfg);
        assert!(acts.len() <= MAX_A);
        self.decision_nodes[s.street as usize] += 1;
        self.actions[s.street as usize] += acts.len() as u64;
        let mut kids = Vec::with_capacity(acts.len());
        for a in acts {
            let child = match rules.apply(s, a) {
                Next::Decision(n) => self.add(rules, cfg, &n, DECISION),
                Next::Terminal(n) => {
                    let alive = rules.players as u32 - n.folded.count_ones();
                    self.add(rules, cfg, &n, if alive == 1 { FOLD_END } else { SHOWDOWN })
                }
            };
            kids.push(child);
        }
        let first = self.children.len() as u32;
        self.children.extend(kids.iter());
        let node = &mut self.nodes[id as usize];
        node.first = first;
        node.n = kids.len() as u8;
        id
    }

    #[inline]
    pub fn child(&self, node: u32, a: usize) -> u32 {
        self.children[self.nodes[node as usize].first as usize + a]
    }

    /// Payoff for player i at a terminal (or at any node once i has folded).
    #[inline]
    pub fn payoff(&self, node: u32, i: usize, sd: &[u16; MAX_P]) -> f64 {
        let n = &self.nodes[node as usize];
        let ti = n.total[i] as f64;
        if n.folded & (1 << i) != 0 {
            return -ti;
        }
        let pot: f64 = n.total[..self.players].iter().map(|&x| x as f64).sum();
        if n.kind == FOLD_END {
            return pot - ti;
        }
        debug_assert_eq!(n.kind, SHOWDOWN);
        let mut best = 0u16;
        for p in 0..self.players {
            if n.folded & (1 << p) == 0 {
                best = best.max(sd[p]);
            }
        }
        if sd[i] < best {
            return -ti;
        }
        let winners = (0..self.players).filter(|&p| n.folded & (1 << p) == 0 && sd[p] == best).count();
        pot / winners as f64 - ti
    }
}

/// Regret storage with first-use allocation (supplement p. 4): preflop blocks are allocated
/// up front; a post-flop node's block (actions x buckets i32) is allocated the first time
/// the traverser updates it. Unallocated nodes read as zero regret (uniform strategy).
/// Blocks live in a chunked arena (16 Mi slots = 64 MiB per chunk) that grows on demand.
pub struct Storage {
    /// per tree node: physical base + 1 (0 = unallocated)
    node_off: Vec<AtomicU64>,
    chunks: Vec<AtomicPtr<AtomicI32>>,
    next: Mutex<u64>,
    allocated: AtomicU64,
    pub avg: Vec<AtomicU32>,
}

const CHUNK_BITS: u32 = 24;
const CHUNK: u64 = 1 << CHUNK_BITS;

impl Drop for Storage {
    fn drop(&mut self) {
        for c in &self.chunks {
            let ptr = c.load(Ordering::Acquire);
            if !ptr.is_null() {
                // SAFETY: allocated in `alloc_chunk` as a boxed slice of CHUNK elements.
                unsafe { drop(Box::from_raw(std::ptr::slice_from_raw_parts_mut(ptr, CHUNK as usize))) };
            }
        }
    }
}

impl Storage {
    /// First-use allocation (paper): preflop allocated now, post-flop on first update.
    pub fn new(tree: &Tree) -> Self {
        let st = Self::empty(tree);
        for (id, nd) in tree.nodes.iter().enumerate() {
            if nd.kind == DECISION && nd.street == 0 {
                st.ensure(tree, id as u32);
            }
        }
        st
    }

    /// Everything allocated up front (the pre-M3-fidelity behaviour; for comparisons).
    pub fn new_dense(tree: &Tree) -> Self {
        let st = Self::empty(tree);
        for (id, nd) in tree.nodes.iter().enumerate() {
            if nd.kind == DECISION {
                st.ensure(tree, id as u32);
            }
        }
        st
    }

    fn empty(tree: &Tree) -> Self {
        // worst case: every block allocated, plus at most one block of padding per chunk
        let max_chunks = (tree.n_regrets / CHUNK + 2) as usize * 2;
        Storage {
            node_off: (0..tree.nodes.len()).map(|_| AtomicU64::new(0)).collect(),
            chunks: (0..max_chunks).map(|_| AtomicPtr::new(std::ptr::null_mut())).collect(),
            next: Mutex::new(0),
            allocated: AtomicU64::new(0),
            avg: (0..tree.n_avg).map(|_| AtomicU32::new(0)).collect(),
        }
    }

    fn alloc_chunk(&self, c: usize) {
        if self.chunks[c].load(Ordering::Acquire).is_null() {
            let v: Box<[AtomicI32]> = (0..CHUNK).map(|_| AtomicI32::new(0)).collect();
            let ptr = Box::into_raw(v) as *mut AtomicI32;
            self.chunks[c].store(ptr, Ordering::Release);
        }
    }

    /// Physical base of a node's block, if allocated.
    #[inline]
    pub fn base(&self, node: u32) -> Option<u64> {
        let v = self.node_off[node as usize].load(Ordering::Acquire);
        if v == 0 {
            None
        } else {
            Some(v - 1)
        }
    }

    /// Physical base of a node's block, allocating it on first use.
    pub fn ensure(&self, tree: &Tree, node: u32) -> u64 {
        if let Some(b) = self.base(node) {
            return b;
        }
        let nd = &tree.nodes[node as usize];
        let size = nd.n as u64 * tree.buckets[nd.street as usize] as u64;
        let mut next = self.next.lock().unwrap();
        if let Some(b) = self.base(node) {
            return b; // another thread won the race
        }
        if (*next % CHUNK) + size > CHUNK {
            *next = (*next / CHUNK + 1) * CHUNK; // blocks never straddle chunks
        }
        let base = *next;
        self.alloc_chunk((base >> CHUNK_BITS) as usize);
        *next += size;
        self.allocated.fetch_add(size, Ordering::Relaxed);
        self.node_off[node as usize].store(base + 1, Ordering::Release);
        base
    }

    #[inline]
    pub fn slot(&self, phys: u64) -> &AtomicI32 {
        let ptr = self.chunks[(phys >> CHUNK_BITS) as usize].load(Ordering::Acquire);
        // SAFETY: phys lies in an allocated block; chunk pointers are never freed while
        // `self` lives, and the chunk was published before the node offset.
        unsafe { &*ptr.add((phys & (CHUNK - 1)) as usize) }
    }

    /// Regret slots allocated so far (excluding chunk padding).
    pub fn allocated_slots(&self) -> u64 {
        self.allocated.load(Ordering::Relaxed)
    }

    /// Physical slot range in use (including padding), for whole-table passes.
    pub fn used_range(&self) -> u64 {
        *self.next.lock().unwrap()
    }

    /// Regret at (node, bucket, action); 0 if the node is unallocated.
    pub fn regret(&self, tree: &Tree, node: u32, bucket: usize, a: usize) -> i32 {
        let n = tree.nodes[node as usize].n as usize;
        self.base(node).map_or(0, |b| self.slot(b + (bucket * n + a) as u64).load(Ordering::Relaxed))
    }

    /// Set a regret (allocating the node). For tests and loaders.
    pub fn set_regret(&self, tree: &Tree, node: u32, bucket: usize, a: usize, v: i32) {
        let n = tree.nodes[node as usize].n as usize;
        let b = self.ensure(tree, node);
        self.slot(b + (bucket * n + a) as u64).store(v, Ordering::Relaxed);
    }

    /// All regrets in the logical dense layout (node order; zeros where unallocated).
    pub fn export_dense(&self, tree: &Tree) -> Vec<i32> {
        let mut out = Vec::with_capacity(tree.n_regrets as usize);
        for (id, nd) in tree.nodes.iter().enumerate() {
            if nd.kind != DECISION {
                continue;
            }
            let size = nd.n as usize * tree.buckets[nd.street as usize] as usize;
            match self.base(id as u32) {
                Some(b) => out.extend((0..size).map(|k| self.slot(b + k as u64).load(Ordering::Relaxed))),
                None => out.extend(std::iter::repeat(0).take(size)),
            }
        }
        out
    }

    /// Regret-matching strategy from `n` physical slots starting at `base`.
    #[inline]
    pub fn current_phys(&self, base: u64, n: usize, out: &mut [f64; MAX_A]) {
        let mut pos = 0.0;
        for a in 0..n {
            let r = self.slot(base + a as u64).load(Ordering::Relaxed);
            let v = if r > 0 { r as f64 } else { 0.0 };
            out[a] = v;
            pos += v;
        }
        if pos > 0.0 {
            for o in out.iter_mut().take(n) {
                *o /= pos;
            }
        } else {
            for o in out.iter_mut().take(n) {
                *o = 1.0 / n as f64;
            }
        }
    }

    /// Current (regret-matching) strategy at (node, bucket); uniform if unallocated.
    #[inline]
    pub fn current_at(&self, tree: &Tree, node: u32, bucket: usize, out: &mut [f64; MAX_A]) {
        let n = tree.nodes[node as usize].n as usize;
        match self.base(node) {
            Some(b) => self.current_phys(b + (bucket * n) as u64, n, out),
            None => {
                for o in out.iter_mut().take(n) {
                    *o = 1.0 / n as f64;
                }
            }
        }
    }

    /// Blueprint policy: preflop average strategy (falls back to current if unvisited),
    /// post-flop current strategy (see `snapshot_*` for the paper's post-flop average).
    #[inline]
    pub fn policy(&self, tree: &Tree, node: u32, bucket: usize, out: &mut [f64; MAX_A]) {
        let nd = &tree.nodes[node as usize];
        let n = nd.n as usize;
        if nd.street == 0 {
            let base = tree.aoff[node as usize] as usize + bucket * n;
            let s: u64 = (0..n).map(|a| self.avg[base + a].load(Ordering::Relaxed) as u64).sum();
            if s > 0 {
                for a in 0..n {
                    out[a] = self.avg[base + a].load(Ordering::Relaxed) as f64 / s as f64;
                }
                return;
            }
        }
        self.current_at(tree, node, bucket, out);
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Schedule {
    pub strategy_interval: u64,
    pub prune_after: u64,
    pub lcfr_until: u64,
    pub discount_every: u64,
    pub explore_prob: f64,
    pub prune_c: i32,
    pub floor: i32,
    /// Paper (supplement pp. 3, 14): pruning never skips actions on the final betting round
    /// or actions that immediately lead to a terminal node. Default true (paper-faithful).
    pub prune_skip_river_and_terminal: bool,
    /// First discount boundary (k = 1). 0 means "discount_every" (boundaries at multiples).
    pub discount_base: u64,
    /// Post-flop snapshot averaging (paper p. 15: every 200 min after the first 800 min).
    /// `snapshot_every == 0` disables it.
    pub snapshot_after: u64,
    pub snapshot_every: u64,
    /// Preflop average-strategy sampling starts here (paper p. 15: "after the initial 800
    /// minutes"; Algorithm 1 itself starts at t = 0, see docs/deviations.md).
    pub avg_start: u64,
}

impl Schedule {
    /// Pluribus' wall-clock schedule as fractions of `planned` iterations (prune 200, LCFR 400,
    /// discount 10 of ~11,520 minutes). Used for tests and short runs.
    pub fn pluribus_scaled(planned: u64, strategy_interval: u64) -> Schedule {
        let f = |m: f64| ((planned as f64 * m / 11_520.0).round() as u64).max(1);
        Schedule {
            strategy_interval,
            prune_after: f(200.0),
            lcfr_until: f(400.0),
            discount_every: f(10.0),
            explore_prob: 0.05,
            prune_c: cfr::intregret::PRUNE_THRESHOLD,
            floor: cfr::intregret::REGRET_FLOOR,
            prune_skip_river_and_terminal: true,
            discount_base: 0,
            snapshot_after: f(800.0),
            snapshot_every: f(200.0),
            avg_start: f(800.0),
        }
    }

    /// Schedule used during the warm-up: no pruning, no discounting, no snapshots.
    pub fn warmup(strategy_interval: u64) -> Schedule {
        Schedule {
            strategy_interval,
            prune_after: u64::MAX,
            lcfr_until: 0,
            discount_every: 1,
            explore_prob: 0.05,
            prune_c: cfr::intregret::PRUNE_THRESHOLD,
            floor: cfr::intregret::REGRET_FLOOR,
            prune_skip_river_and_terminal: true,
            discount_base: 0,
            snapshot_after: u64::MAX,
            snapshot_every: 0,
            avg_start: u64::MAX,
        }
    }

    fn base(&self) -> u64 {
        if self.discount_base > 0 { self.discount_base } else { self.discount_every }
    }

    /// Discount step k due after completing iteration t (paper: t < LCFR threshold).
    pub fn discount_k(&self, t: u64) -> Option<u64> {
        let b = self.base();
        if t == 0 || t >= self.lcfr_until || t < b || (t - b) % self.discount_every != 0 {
            return None;
        }
        Some((t - b) / self.discount_every + 1)
    }

    /// Next discount boundary strictly after t, if any.
    pub fn next_discount(&self, t: u64) -> Option<u64> {
        let b = self.base();
        let n = if t < b { b } else { b + ((t - b) / self.discount_every + 1) * self.discount_every };
        (n < self.lcfr_until).then_some(n)
    }

    pub fn snapshot_due(&self, t: u64) -> bool {
        self.snapshot_every > 0 && t >= self.snapshot_after && (t - self.snapshot_after) % self.snapshot_every == 0
    }

    pub fn next_snapshot(&self, t: u64) -> Option<u64> {
        if self.snapshot_every == 0 || self.snapshot_after == u64::MAX {
            return None;
        }
        let a = self.snapshot_after;
        Some(if t < a { a } else { a + ((t - a) / self.snapshot_every + 1) * self.snapshot_every })
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct ThreadStats {
    pub iterations: u64,
    pub traverse_ns: u64,
    pub river_ns: u64,
    pub river_calls: u64,
    pub nodes: u64,
    pub pruned_actions: u64,
    /// traversals, and traversals that reached at least one river decision node
    pub traversals: u64,
    pub river_reached: u64,
    pub regret: RegretStats,
}

impl ThreadStats {
    pub fn add(&mut self, o: &ThreadStats) {
        self.iterations += o.iterations;
        self.traverse_ns += o.traverse_ns;
        self.river_ns += o.river_ns;
        self.river_calls += o.river_calls;
        self.nodes += o.nodes;
        self.pruned_actions += o.pruned_actions;
        self.traversals += o.traversals;
        self.river_reached += o.river_reached;
        self.regret.saturated += o.regret.saturated;
        self.regret.floored += o.regret.floored;
    }
}

/// Per-thread deal and lazily computed buckets.
pub struct Deal {
    pub hole: [(u8, u8); MAX_P],
    pub board: [u8; 5],
    pub bucket: [[u16; 4]; MAX_P],
    pub sd: [u16; MAX_P],
    river_ready: bool,
    /// set when the current traversal visits a river decision node
    pub reached_river: bool,
    rb: RiverBatch,
    hr: HeroRiver,
    pub rng: Rng,
}

impl Default for Deal {
    fn default() -> Self {
        Self::new()
    }
}

#[inline]
fn mix(seed: u64, t: u64, i: u64) -> u64 {
    let mut z = seed ^ t.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ i.wrapping_mul(0xD1B5_4A32_D192_ED03);
    z = (z ^ (z >> 31)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z ^ (z >> 29)
}

impl Deal {
    pub fn new() -> Self {
        Deal {
            hole: [(0, 0); MAX_P],
            board: [0; 5],
            bucket: [[0; 4]; MAX_P],
            sd: [0; MAX_P],
            river_ready: false,
            reached_river: false,
            rb: RiverBatch::new(),
            hr: HeroRiver::new(),
            rng: Rng::new(0),
        }
    }

    /// Deal for (seed, t, i): hole cards for `players`, a full board; preflop buckets and
    /// showdown classes now, post-flop buckets lazily.
    pub fn deal(&mut self, seed: u64, t: u64, i: u64, players: usize) {
        self.rng = Rng::new(mix(seed, t, i));
        let mut deck: [u8; 52] = std::array::from_fn(|k| k as u8);
        let need = 2 * players + 5;
        for k in 0..need {
            let j = k + (self.rng.next_u64() % (52 - k) as u64) as usize;
            deck.swap(k, j);
        }
        self.board.copy_from_slice(&deck[2 * players..2 * players + 5]);
        let lut = Lut::get();
        let bh = LHand::of(&self.board);
        for p in 0..players {
            let h = (deck[2 * p], deck[2 * p + 1]);
            self.hole[p] = h;
            self.bucket[p] = [preflop_class(h.0, h.1), u16::MAX, u16::MAX, u16::MAX];
            self.sd[p] = lut.class(bh.add(h.0).add(h.1));
        }
        self.river_ready = false;
    }

    #[inline]
    pub fn bucket(&mut self, abs: &dyn CardAbstraction, p: usize, street: usize, st: &mut ThreadStats) -> usize {
        let b = self.bucket[p][street];
        if b != u16::MAX {
            return b as usize;
        }
        let h = self.hole[p];
        let v = match street {
            1 => abs.flop(h, &[self.board[0], self.board[1], self.board[2]]),
            2 => abs.turn(h, &[self.board[0], self.board[1], self.board[2], self.board[3]]),
            _ => {
                let t0 = Instant::now();
                let eq = if abs.river_needs_equity() {
                    if !self.river_ready {
                        self.hr.prepare(&mut self.rb, &self.board);
                        self.river_ready = true;
                    }
                    self.hr.equity(h.0, h.1)
                } else {
                    0.0
                };
                let v = abs.river(h, &self.board, eq);
                st.river_ns += t0.elapsed().as_nanos() as u64;
                st.river_calls += 1;
                v
            }
        };
        self.bucket[p][street] = v;
        v as usize
    }
}

pub struct Trainer<'a> {
    pub tree: &'a Tree,
    pub st: &'a Storage,
    pub abs: &'a dyn CardAbstraction,
    pub sched: &'a Schedule,
    pub seed: u64,
}

impl<'a> Trainer<'a> {
    pub fn iteration(&self, t: u64, d: &mut Deal, ts: &mut ThreadStats) {
        let t0 = Instant::now();
        for i in 0..self.tree.players {
            d.deal(self.seed, t, i as u64, self.tree.players);
            if t >= self.sched.avg_start && t % self.sched.strategy_interval == 0 {
                self.update_strategy(0, i, d, ts);
            }
            let prune = t > self.sched.prune_after && d.rng.next_f64() >= self.sched.explore_prob;
            d.reached_river = false;
            self.traverse(0, i, prune, d, ts);
            ts.traversals += 1;
            ts.river_reached += d.reached_river as u64;
        }
        ts.iterations += 1;
        ts.traverse_ns += t0.elapsed().as_nanos() as u64;
    }

    fn update_strategy(&self, node: u32, i: usize, d: &mut Deal, ts: &mut ThreadStats) {
        let nd = self.tree.nodes[node as usize];
        if nd.kind != DECISION || nd.folded & (1 << i) != 0 || nd.street > 0 {
            return;
        }
        let n = nd.n as usize;
        let p = nd.player as usize;
        if p == i {
            let b = d.bucket(self.abs, p, 0, ts);
            let mut s = [0.0; MAX_A];
            self.st.current_at(self.tree, node, b, &mut s);
            let a = d.rng.sample_index(&s[..n]);
            self.st.avg[self.tree.aoff[node as usize] as usize + b * n + a].fetch_add(1, Ordering::Relaxed);
            self.update_strategy(self.tree.child(node, a), i, d, ts);
        } else {
            for a in 0..n {
                self.update_strategy(self.tree.child(node, a), i, d, ts);
            }
        }
    }

    fn traverse(&self, node: u32, i: usize, prune: bool, d: &mut Deal, ts: &mut ThreadStats) -> f64 {
        ts.nodes += 1;
        let nd = self.tree.nodes[node as usize];
        if nd.kind != DECISION || nd.folded & (1 << i) != 0 {
            return self.tree.payoff(node, i, &d.sd);
        }
        let n = nd.n as usize;
        let p = nd.player as usize;
        let street = nd.street as usize;
        d.reached_river |= street == 3;
        let b = d.bucket(self.abs, p, street, ts);
        let mut s = [0.0; MAX_A];
        if p != i {
            // opponents only read: never allocates
            self.st.current_at(self.tree, node, b, &mut s);
            let a = d.rng.sample_index(&s[..n]);
            return self.traverse(self.tree.child(node, a), i, prune, d, ts);
        }
        let base = self.st.ensure(self.tree, node) + (b * n) as u64;
        self.st.current_phys(base, n, &mut s);
        let mut va = [0.0; MAX_A];
        let mut explored = [true; MAX_A];
        let mut v = 0.0;
        for a in 0..n {
            let child = self.tree.child(node, a);
            if prune {
                let r = self.st.slot(base + a as u64).load(Ordering::Relaxed);
                let exempt = self.sched.prune_skip_river_and_terminal
                    && (street == 3 || self.tree.nodes[child as usize].kind != DECISION);
                if r <= self.sched.prune_c && !exempt {
                    explored[a] = false;
                    ts.pruned_actions += 1;
                    continue;
                }
            }
            va[a] = self.traverse(child, i, prune, d, ts);
            v += s[a] * va[a];
        }
        for a in 0..n {
            if explored[a] {
                let cell = self.st.slot(base + a as u64);
                let mut r = cell.load(Ordering::Relaxed);
                add_regret(&mut r, va[a] - v, self.sched.floor, &mut ts.regret);
                cell.store(r, Ordering::Relaxed);
            }
        }
        v
    }

    /// Run iterations from+1 ..= to on `threads` threads (stop early if `stop` is set).
    /// Returns the last completed iteration and merged stats.
    pub fn run(&self, from: u64, to: u64, threads: usize, stop: &AtomicBool) -> (u64, ThreadStats) {
        let next = AtomicU64::new(from + 1);
        let done_max = AtomicU64::new(from);
        let merged = std::sync::Mutex::new(ThreadStats::default());
        std::thread::scope(|sc| {
            for _ in 0..threads.max(1) {
                sc.spawn(|| {
                    let mut d = Deal::new();
                    let mut ts = ThreadStats::default();
                    loop {
                        if stop.load(Ordering::Relaxed) {
                            break;
                        }
                        let t = next.fetch_add(1, Ordering::Relaxed);
                        if t > to {
                            break;
                        }
                        self.iteration(t, &mut d, &mut ts);
                        done_max.fetch_max(t, Ordering::Relaxed);
                    }
                    merged.lock().unwrap().add(&ts);
                });
            }
        });
        // With an early stop, iterations below done_max may be unfinished only if stop was
        // set; callers then use the count of completed iterations, which is exact for one
        // thread and a close lower bound otherwise.
        let st = merged.into_inner().unwrap();
        let last = if stop.load(Ordering::Relaxed) { from + st.iterations } else { to };
        (last, st)
    }

    /// Linear-CFR discount step k (d = k / (k + 1)) on all regrets and counts.
    pub fn discount(&self, k: u64, threads: usize) {
        let st = self.st;
        let used = st.used_range() as usize;
        crate::abstraction::parallel_chunks(used, threads, 1 << 20, |a, b| {
            for i in a..b {
                let x = st.slot(i as u64);
                x.store(discount_regret(x.load(Ordering::Relaxed), k), Ordering::Relaxed);
            }
        });
        let c = &self.st.avg;
        crate::abstraction::parallel_chunks(c.len(), threads, 1 << 20, |a, b| {
            for x in &c[a..b] {
                x.store(discount_count(x.load(Ordering::Relaxed), k), Ordering::Relaxed);
            }
        });
    }
}

// ------------------------------------------------------------------ checkpoints

const MAGIC: &[u8; 4] = b"PLBP";
const VERSION: u32 = 3;

#[derive(Clone, Debug, PartialEq)]
pub struct CkptMeta {
    pub config: String,
    pub buckets: [u32; 4],
    pub iteration: u64,
    pub seed: u64,
    pub elapsed_secs: f64,
    pub sched: Schedule,
    /// post-flop snapshots accumulated so far
    pub snapshots: u64,
}

struct Fnv(u64);
impl Fnv {
    fn new() -> Self {
        Fnv(0xcbf2_9ce4_8422_2325)
    }
    fn feed(&mut self, b: &[u8]) {
        for &x in b {
            self.0 ^= x as u64;
            self.0 = self.0.wrapping_mul(0x0100_0000_01b3);
        }
    }
}

pub fn save_checkpoint(path: &Path, meta: &CkptMeta, tree: &Tree, st: &Storage) -> io::Result<u64> {
    if let Some(d) = path.parent() {
        fs::create_dir_all(d)?;
    }
    let tmp = path.with_extension("tmp");
    let mut h = Fnv::new();
    {
        let mut w = BufWriter::with_capacity(1 << 22, File::create(&tmp)?);
        w.write_all(MAGIC)?;
        w.write_all(&VERSION.to_le_bytes())?;
        w.write_all(&(meta.config.len() as u32).to_le_bytes())?;
        w.write_all(meta.config.as_bytes())?;
        for b in meta.buckets {
            w.write_all(&b.to_le_bytes())?;
        }
        let s = &meta.sched;
        for x in [meta.iteration, meta.seed, tree.n_regrets, tree.n_avg, s.strategy_interval, s.prune_after, s.lcfr_until, s.discount_every] {
            w.write_all(&x.to_le_bytes())?;
        }
        for x in [meta.elapsed_secs, s.explore_prob] {
            w.write_all(&x.to_le_bytes())?;
        }
        for x in [s.prune_c, s.floor, s.prune_skip_river_and_terminal as i32] {
            w.write_all(&x.to_le_bytes())?;
        }
        for x in [s.discount_base, s.snapshot_after, s.snapshot_every, meta.snapshots, s.avg_start] {
            w.write_all(&x.to_le_bytes())?;
        }
        // v3: allocated blocks only, as (u32 node id, block) records in node order, streamed
        let allocated: u64 = (0..tree.nodes.len() as u32).filter(|&id| st.base(id).is_some()).count() as u64;
        w.write_all(&allocated.to_le_bytes())?;
        let mut buf = Vec::with_capacity(1 << 16);
        for (id, nd) in tree.nodes.iter().enumerate() {
            let Some(base) = st.base(id as u32) else { continue };
            let size = nd.n as u64 * tree.buckets[nd.street as usize] as u64;
            buf.clear();
            buf.extend_from_slice(&(id as u32).to_le_bytes());
            for k in 0..size {
                buf.extend_from_slice(&st.slot(base + k).load(Ordering::Relaxed).to_le_bytes());
            }
            h.feed(&buf);
            w.write_all(&buf)?;
        }
        for chunk in st.avg.chunks(1 << 18) {
            buf.clear();
            for x in chunk {
                buf.extend_from_slice(&x.load(Ordering::Relaxed).to_le_bytes());
            }
            h.feed(&buf);
            w.write_all(&buf)?;
        }
        w.write_all(&h.0.to_le_bytes())?;
        w.flush()?;
        w.get_ref().sync_all()?;
    }
    fs::rename(&tmp, path)?;
    Ok(fs::metadata(path)?.len())
}

pub fn load_checkpoint(path: &Path, tree: &Tree, st: &Storage) -> io::Result<CkptMeta> {
    let bad = |m: String| io::Error::new(io::ErrorKind::InvalidData, m);
    let mut r = BufReader::with_capacity(1 << 22, File::open(path)?);
    let mut b4 = [0u8; 4];
    let mut b8 = [0u8; 8];
    r.read_exact(&mut b4)?;
    if &b4 != MAGIC {
        return Err(bad("bad magic".into()));
    }
    r.read_exact(&mut b4)?;
    let version = u32::from_le_bytes(b4);
    if !(1..=3).contains(&version) {
        return Err(bad(format!("unsupported checkpoint version {version}")));
    }
    r.read_exact(&mut b4)?;
    let mut name = vec![0u8; u32::from_le_bytes(b4) as usize];
    r.read_exact(&mut name)?;
    let mut buckets = [0u32; 4];
    for b in buckets.iter_mut() {
        r.read_exact(&mut b4)?;
        *b = u32::from_le_bytes(b4);
    }
    let mut u = [0u64; 8];
    for x in u.iter_mut() {
        r.read_exact(&mut b8)?;
        *x = u64::from_le_bytes(b8);
    }
    let mut f = [0f64; 2];
    for x in f.iter_mut() {
        r.read_exact(&mut b8)?;
        *x = f64::from_le_bytes(b8);
    }
    let mut i = [0i32; 3];
    for x in i.iter_mut() {
        r.read_exact(&mut b4)?;
        *x = i32::from_le_bytes(b4);
    }
    // v2: discount_base, snapshot_after, snapshot_every, snapshots; v3: + avg_start
    let mut v2 = [0u64; 5];
    let nv2 = match version {
        1 => 0,
        2 => 4,
        _ => 5,
    };
    for x in v2.iter_mut().take(nv2) {
        r.read_exact(&mut b8)?;
        *x = u64::from_le_bytes(b8);
    }
    if u[2] != tree.n_regrets || u[3] != tree.n_avg || buckets != tree.buckets {
        return Err(bad(format!("checkpoint shape {:?}/{}/{} does not match tree {:?}/{}/{}", buckets, u[2], u[3], tree.buckets, tree.n_regrets, tree.n_avg)));
    }
    let mut h = Fnv::new();
    let mut buf = vec![0u8; 1 << 20];
    let mut fill_block = |r: &mut BufReader<File>, h: &mut Fnv, base: u64, size: u64| -> io::Result<()> {
        let mut done = 0u64;
        while done < size {
            let m = (size - done).min(1 << 18);
            let bytes = &mut buf[..m as usize * 4];
            r.read_exact(bytes)?;
            h.feed(bytes);
            for (k, c) in bytes.chunks_exact(4).enumerate() {
                st.slot(base + done + k as u64).store(i32::from_le_bytes([c[0], c[1], c[2], c[3]]), Ordering::Relaxed);
            }
            done += m;
        }
        Ok(())
    };
    if version <= 2 {
        // dense layout: every decision node's block, in node order
        for (id, nd) in tree.nodes.iter().enumerate() {
            if nd.kind != DECISION {
                continue;
            }
            let size = nd.n as u64 * tree.buckets[nd.street as usize] as u64;
            let base = st.ensure(tree, id as u32);
            fill_block(&mut r, &mut h, base, size)?;
        }
    } else {
        r.read_exact(&mut b8)?;
        let allocated = u64::from_le_bytes(b8);
        for _ in 0..allocated {
            r.read_exact(&mut b4)?;
            h.feed(&b4);
            let id = u32::from_le_bytes(b4);
            let nd = tree.nodes.get(id as usize).ok_or_else(|| bad(format!("bad node id {id}")))?;
            if nd.kind != DECISION {
                return Err(bad(format!("node {id} is not a decision node")));
            }
            let size = nd.n as u64 * tree.buckets[nd.street as usize] as u64;
            let base = st.ensure(tree, id);
            fill_block(&mut r, &mut h, base, size)?;
        }
    }
    let mut buf = vec![0u8; 1 << 20];
    for chunk in st.avg.chunks(1 << 18) {
        let bytes = &mut buf[..chunk.len() * 4];
        r.read_exact(bytes)?;
        h.feed(bytes);
        for (x, c) in chunk.iter().zip(bytes.chunks_exact(4)) {
            x.store(u32::from_le_bytes([c[0], c[1], c[2], c[3]]), Ordering::Relaxed);
        }
    }
    r.read_exact(&mut b8)?;
    if u64::from_le_bytes(b8) != h.0 {
        return Err(bad("checksum mismatch".into()));
    }
    Ok(CkptMeta {
        config: String::from_utf8(name).map_err(|_| bad("utf8".into()))?,
        buckets,
        iteration: u[0],
        seed: u[1],
        elapsed_secs: f[0],
        sched: Schedule {
            strategy_interval: u[4],
            prune_after: u[5],
            lcfr_until: u[6],
            discount_every: u[7],
            explore_prob: f[1],
            prune_c: i[0],
            floor: i[1],
            prune_skip_river_and_terminal: i[2] != 0,
            discount_base: v2[0],
            snapshot_after: if version >= 2 { v2[1] } else { u64::MAX },
            snapshot_every: v2[2],
            avg_start: if version >= 3 { v2[4] } else { 0 },
        },
        snapshots: v2[3],
    })
}

// ------------------------------------------------------------------ post-flop snapshots

const SNAP_MAGIC: &[u8; 4] = b"PLS2";

/// Quantised regret-matching strategy (sum ~ 65535) at (node, bucket); uniform if unallocated.
#[inline]
fn quantised_current(st: &Storage, tree: &Tree, node: u32, bucket: usize, out: &mut [u32; MAX_A]) {
    let n = tree.nodes[node as usize].n as usize;
    let mut s = [0.0; MAX_A];
    st.current_at(tree, node, bucket, &mut s);
    for a in 0..n {
        out[a] = (s[a] * 65_535.0).round() as u32;
    }
}

struct SnapReader {
    r: BufReader<File>,
    remaining: u64,
}

impl SnapReader {
    /// Next record: (node, snapshots included, sums), or None at the end.
    fn next(&mut self, tree: &Tree) -> io::Result<Option<(u32, u32, Vec<u32>)>> {
        if self.remaining == 0 {
            return Ok(None);
        }
        self.remaining -= 1;
        let mut b4 = [0u8; 4];
        self.r.read_exact(&mut b4)?;
        let node = u32::from_le_bytes(b4);
        self.r.read_exact(&mut b4)?;
        let cnt = u32::from_le_bytes(b4);
        let nd = &tree.nodes[node as usize];
        let size = nd.n as usize * tree.buckets[nd.street as usize] as usize;
        let mut v = vec![0u32; size];
        for x in v.iter_mut() {
            self.r.read_exact(&mut b4)?;
            *x = u32::from_le_bytes(b4);
        }
        Ok(Some((node, cnt, v)))
    }
}

fn snap_open(path: &Path, tree: &Tree) -> io::Result<Option<(SnapReader, u64)>> {
    let bad = |m: &str| io::Error::new(io::ErrorKind::InvalidData, m.to_string());
    match File::open(path) {
        Ok(f) => {
            let mut r = BufReader::with_capacity(1 << 22, f);
            let mut b4 = [0u8; 4];
            let mut b8 = [0u8; 8];
            r.read_exact(&mut b4)?;
            if &b4 != SNAP_MAGIC {
                return Err(bad("bad snapshot magic"));
            }
            r.read_exact(&mut b8)?;
            if u64::from_le_bytes(b8) != tree.nodes.len() as u64 {
                return Err(bad("snapshot file does not match the tree"));
            }
            r.read_exact(&mut b8)?;
            let total = u64::from_le_bytes(b8);
            r.read_exact(&mut b8)?;
            let records = u64::from_le_bytes(b8);
            Ok(Some((SnapReader { r, remaining: records }, total)))
        }
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e),
    }
}

/// Add one snapshot of the post-flop current strategy (paper p. 15) to the sparse
/// running-sum file at `path`. Records exist for post-flop nodes that are allocated now or
/// were recorded before; each record stores how many snapshots it includes, so nodes first
/// allocated later are averaged with uniform for the snapshots they missed. Streams the old
/// file into a new one and renames (crash-safe, no table-sized buffer).
/// Returns the total number of snapshots.
pub fn snapshot_add(path: &Path, tree: &Tree, st: &Storage) -> io::Result<u64> {
    let mut old = snap_open(path, tree)?;
    let total = old.as_ref().map_or(0, |o| o.1) + 1;
    let tmp = path.with_extension("tmp");
    let mut records: Vec<u32> = Vec::new(); // node ids written (counted at the end)
    {
        let mut w = BufWriter::with_capacity(1 << 22, File::create(&tmp)?);
        w.write_all(SNAP_MAGIC)?;
        w.write_all(&(tree.nodes.len() as u64).to_le_bytes())?;
        w.write_all(&total.to_le_bytes())?;
        w.write_all(&0u64.to_le_bytes())?; // record count, patched below
        let mut pending = match old.as_mut() {
            Some((rd, _)) => rd.next(tree)?,
            None => None,
        };
        let mut q = [0u32; MAX_A];
        for (id, nd) in tree.nodes.iter().enumerate() {
            if nd.kind != DECISION || nd.street == 0 {
                continue;
            }
            let id = id as u32;
            let prev = if pending.as_ref().is_some_and(|p| p.0 == id) {
                let p = pending.take();
                pending = match old.as_mut() {
                    Some((rd, _)) => rd.next(tree)?,
                    None => None,
                };
                p
            } else {
                None
            };
            if prev.is_none() && st.base(id).is_none() {
                continue;
            }
            let n = nd.n as usize;
            let nb = tree.buckets[nd.street as usize] as usize;
            let (cnt, mut sums) = match prev {
                Some((_, c, v)) => (c, v),
                None => (0, vec![0u32; n * nb]),
            };
            for b in 0..nb {
                quantised_current(st, tree, id, b, &mut q);
                for a in 0..n {
                    sums[b * n + a] += q[a];
                }
            }
            w.write_all(&id.to_le_bytes())?;
            w.write_all(&(cnt + 1).to_le_bytes())?;
            for x in &sums {
                w.write_all(&x.to_le_bytes())?;
            }
            records.push(id);
        }
        w.flush()?;
        let mut f = w.into_inner().map_err(|e| e.into_error())?;
        use std::io::{Seek, SeekFrom};
        f.seek(SeekFrom::Start(4 + 8 + 8))?;
        f.write_all(&(records.len() as u64).to_le_bytes())?;
        f.sync_all()?;
    }
    drop(old);
    fs::rename(&tmp, path)?;
    Ok(total)
}

/// Post-flop snapshot average loaded for evaluation.
pub struct SnapshotAverage {
    pub total: u64,
    pub recs: cfr::fxhash::FxHashMap<u32, (u32, Vec<u32>)>,
}

impl SnapshotAverage {
    pub fn load(path: &Path, tree: &Tree) -> io::Result<Self> {
        let (mut rd, total) = snap_open(path, tree)?.ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "no snapshot file"))?;
        let mut recs = cfr::fxhash::FxHashMap::default();
        while let Some((id, c, v)) = rd.next(tree)? {
            recs.insert(id, (c, v));
        }
        Ok(SnapshotAverage { total, recs })
    }

    /// Averaged post-flop strategy at (node, bucket): missing snapshots count as uniform.
    pub fn policy(&self, tree: &Tree, node: u32, bucket: usize, out: &mut [f64; MAX_A]) {
        let n = tree.nodes[node as usize].n as usize;
        let uni = 65_535.0 / n as f64;
        match self.recs.get(&node) {
            Some((c, v)) => {
                let missing = (self.total - *c as u64) as f64;
                let mut sum = 0.0;
                for a in 0..n {
                    out[a] = v[bucket * n + a] as f64 + missing * uni;
                    sum += out[a];
                }
                for o in out.iter_mut().take(n) {
                    *o /= sum;
                }
            }
            None => out.iter_mut().take(n).for_each(|o| *o = 1.0 / n as f64),
        }
    }
}

// ------------------------------------------------------------------ metrics

#[derive(Clone, Debug, Default)]
pub struct RegretSummary {
    /// per street: infosets, infosets with any nonzero regret, sum of positive regret
    pub infosets: [u64; 4],
    pub touched: [u64; 4],
    pub pos_sum: [f64; 4],
}

pub fn regret_summary(tree: &Tree, st: &Storage) -> RegretSummary {
    let mut s = RegretSummary::default();
    for (i, nd) in tree.nodes.iter().enumerate() {
        if nd.kind != DECISION {
            continue;
        }
        let street = nd.street as usize;
        let n = nd.n as usize;
        let nb = tree.buckets[street] as usize;
        let Some(base) = st.base(i as u32) else {
            s.infosets[street] += nb as u64;
            continue;
        };
        for b in 0..nb {
            let mut any = false;
            let mut pos = 0i64;
            for a in 0..n {
                let r = st.slot(base + (b * n + a) as u64).load(Ordering::Relaxed);
                any |= r != 0;
                pos += r.max(0) as i64;
            }
            s.infosets[street] += 1;
            s.touched[street] += any as u64;
            s.pos_sum[street] += pos as f64;
        }
    }
    s
}

/// A fixed random sample of infosets (node, bucket) for strategy-change tracking.
pub fn sample_infosets(tree: &Tree, n: usize, seed: u64, preflop: bool) -> Vec<(u32, u32)> {
    let ids: Vec<u32> = (0..tree.nodes.len() as u32)
        .filter(|&i| {
            let nd = &tree.nodes[i as usize];
            nd.kind == DECISION && ((nd.street == 0) == preflop)
        })
        .collect();
    let mut rng = Rng::new(seed);
    (0..n)
        .map(|_| {
            let node = ids[(rng.next_u64() % ids.len() as u64) as usize];
            let nb = tree.buckets[tree.nodes[node as usize].street as usize];
            (node, (rng.next_u64() % nb as u64) as u32)
        })
        .collect()
}

/// Concatenated policy vectors (preflop: average strategy; post-flop: current) at samples.
pub fn policy_snapshot(tree: &Tree, st: &Storage, samples: &[(u32, u32)]) -> Vec<f32> {
    let mut out = Vec::new();
    let mut s = [0.0; MAX_A];
    for &(node, b) in samples {
        let n = tree.nodes[node as usize].n as usize;
        st.policy(tree, node, b as usize, &mut s);
        out.extend(s[..n].iter().map(|&x| x as f32));
    }
    out
}

/// Mean over sampled infosets of the L1 distance between two snapshots (0..2).
pub fn snapshot_change(tree: &Tree, samples: &[(u32, u32)], a: &[f32], b: &[f32]) -> f64 {
    let mut off = 0;
    let mut sum = 0.0;
    for &(node, _) in samples {
        let n = tree.nodes[node as usize].n as usize;
        sum += a[off..off + n].iter().zip(&b[off..off + n]).map(|(x, y)| (x - y).abs() as f64).sum::<f64>();
        off += n;
    }
    sum / samples.len() as f64
}

// ------------------------------------------------------------------ head-to-head

/// Play one hand of the abstract game with per-seat policies; returns chip payoffs.
pub fn play_hand(tree: &Tree, abs: &dyn CardAbstraction, seats: &[&Storage], d: &mut Deal, act_seed: u64, ts: &mut ThreadStats) -> [f64; MAX_P] {
    let mut rng = Rng::new(act_seed);
    let mut node = 0u32;
    let mut s = [0.0; MAX_A];
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
        seats[p].policy(tree, node, b, &mut s);
        let a = rng.sample_index(&s[..nd.n as usize]);
        node = tree.child(node, a);
    }
}

/// Duplicate head-to-head: for each deal, `a` sits in each seat in turn against `b` in all
/// other seats, with common random numbers. Returns (mean chips/hand for `a`, std error).
pub fn head_to_head(tree: &Tree, abs: &dyn CardAbstraction, a: &Storage, b: &Storage, deals: u64, seed: u64, threads: usize) -> (f64, f64) {
    let sums = std::sync::Mutex::new((0.0f64, 0.0f64, 0u64));
    let next = AtomicU64::new(0);
    std::thread::scope(|sc| {
        for _ in 0..threads.max(1) {
            sc.spawn(|| {
                let mut d = Deal::new();
                let mut ts = ThreadStats::default();
                let (mut s1, mut s2, mut n) = (0.0, 0.0, 0u64);
                loop {
                    let k = next.fetch_add(1, Ordering::Relaxed);
                    if k >= deals {
                        break;
                    }
                    d.deal(seed, k, 0, tree.players);
                    let mut per_deal = 0.0;
                    for seat in 0..tree.players {
                        let seats: Vec<&Storage> = (0..tree.players).map(|p| if p == seat { a } else { b }).collect();
                        let r = play_hand(tree, abs, &seats, &mut d, mix(seed ^ 0xA5A5, k, 0), &mut ts);
                        per_deal += r[seat];
                    }
                    let x = per_deal / tree.players as f64;
                    s1 += x;
                    s2 += x * x;
                    n += 1;
                }
                let mut g = sums.lock().unwrap();
                g.0 += s1;
                g.1 += s2;
                g.2 += n;
            });
        }
    });
    let (s1, s2, n) = sums.into_inner().unwrap();
    let mean = s1 / n as f64;
    let var = (s2 / n as f64 - mean * mean).max(0.0);
    (mean, (var / n as f64).sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::betting::TreeCounter;
    use crate::buckets::StrengthAbstraction;

    fn small() -> (Rules, AbstractionConfig) {
        let rules = Rules { players: 3, ..Rules::default() };
        let cfg = AbstractionConfig::parse(
            "limp = sb\npreflop.0 = 0.5\npreflop.1 = 1.0\nflop.0 = 0.5, 1.0\nturn.0 = 1.0\nriver.0 = 1.0",
        )
        .unwrap();
        (rules, cfg)
    }

    fn sched() -> Schedule {
        Schedule {
            strategy_interval: 7,
            prune_after: 300,
            lcfr_until: 400,
            discount_every: 100,
            explore_prob: 0.05,
            prune_c: -2_000,
            floor: -3_000,
            prune_skip_river_and_terminal: true,
            discount_base: 0,
            snapshot_after: u64::MAX,
            snapshot_every: 0,
            avg_start: 0,
        }
    }

    #[test]
    fn tree_matches_counter() {
        let (rules, cfg) = small();
        let tree = Tree::build(&rules, &cfg, [169, 4, 4, 4]);
        let counts = TreeCounter::new(rules, &cfg).count().unwrap();
        assert_eq!(tree.decision_nodes, counts.nodes);
        assert_eq!(tree.actions, counts.actions);
        let expect: u64 = (0..4).map(|s| counts.actions[s] * [169, 4, 4, 4][s]).sum();
        assert_eq!(tree.n_regrets, expect);
        assert_eq!(tree.n_avg, counts.actions[0] * 169);
    }

    #[test]
    fn payoffs_are_zero_sum_at_every_terminal() {
        let (rules, cfg) = small();
        let tree = Tree::build(&rules, &cfg, [169, 4, 4, 4]);
        let mut d = Deal::new();
        for k in 0..50 {
            d.deal(3, k, 0, 3);
            for (id, nd) in tree.nodes.iter().enumerate() {
                if nd.kind != DECISION {
                    let s: f64 = (0..3).map(|i| tree.payoff(id as u32, i, &d.sd)).sum();
                    assert!(s.abs() < 1e-9, "node {id}: sum {s}");
                }
            }
        }
    }

    fn train(iters: u64, resume_at: Option<u64>) -> (Vec<i32>, Vec<u32>) {
        let (rules, cfg) = small();
        let abs = StrengthAbstraction { k: 4 };
        let tree = Tree::build(&rules, &cfg, abs.buckets());
        let st = Storage::new(&tree);
        let sc = sched();
        let tr = Trainer { tree: &tree, st: &st, abs: &abs, sched: &sc, seed: 11 };
        let stop = AtomicBool::new(false);
        let mut t = 0;
        let step = |to: u64, t: &mut u64| {
            while *t < to {
                let next = (*t / sc.discount_every + 1) * sc.discount_every;
                let end = next.min(to);
                tr.run(*t, end, 1, &stop);
                *t = end;
                if let Some(k) = sc.discount_k(end) {
                    tr.discount(k, 1);
                }
            }
        };
        if let Some(mid) = resume_at {
            step(mid, &mut t);
            let dir = std::env::temp_dir().join(format!("bp_ckpt_{}_{}", std::process::id(), mid));
            let path = dir.join("c.ckpt");
            let meta = CkptMeta { config: "t".into(), buckets: tree.buckets, iteration: t, seed: 11, elapsed_secs: 0.0, sched: sc.clone(), snapshots: 3 };
            save_checkpoint(&path, &meta, &tree, &st).unwrap();
            let st2 = Storage::new(&tree);
            let m2 = load_checkpoint(&path, &tree, &st2).unwrap();
            assert_eq!(m2, meta);
            fs::remove_dir_all(dir).ok();
            // continue on the restored storage
            let tr2 = Trainer { tree: &tree, st: &st2, abs: &abs, sched: &sc, seed: 11 };
            let mut t2 = m2.iteration;
            while t2 < iters {
                let next = (t2 / sc.discount_every + 1) * sc.discount_every;
                let end = next.min(iters);
                tr2.run(t2, end, 1, &stop);
                t2 = end;
                if let Some(k) = sc.discount_k(end) {
                    tr2.discount(k, 1);
                }
            }
            return (st2.export_dense(&tree), st2.avg.iter().map(|x| x.load(Ordering::Relaxed)).collect());
        }
        step(iters, &mut t);
        (st.export_dense(&tree), st.avg.iter().map(|x| x.load(Ordering::Relaxed)).collect())
    }

    #[test]
    fn single_thread_is_deterministic_and_resume_is_bit_identical() {
        let a = train(600, None);
        let b = train(600, None);
        assert_eq!(a, b);
        assert!(a.0.iter().any(|&r| r != 0), "regrets never updated");
        assert!(a.1.iter().any(|&c| c != 0), "average strategy never updated");
        let c = train(600, Some(250)); // checkpoint mid-run (after a discount, before pruning)
        assert_eq!(a, c);
    }

    #[test]
    fn self_play_head_to_head_is_exactly_zero() {
        let (rules, cfg) = small();
        let abs = StrengthAbstraction { k: 4 };
        let tree = Tree::build(&rules, &cfg, abs.buckets());
        let st = Storage::new(&tree);
        let sc = sched();
        let tr = Trainer { tree: &tree, st: &st, abs: &abs, sched: &sc, seed: 5 };
        tr.run(0, 200, 2, &AtomicBool::new(false));
        let (m, _) = head_to_head(&tree, &abs, &st, &st, 2_000, 9, 2);
        assert!(m.abs() < 1e-9, "{m}");
    }

    #[test]
    fn trained_beats_untrained_small_game() {
        let (rules, cfg) = small();
        let abs = StrengthAbstraction { k: 4 };
        let tree = Tree::build(&rules, &cfg, abs.buckets());
        let trained = Storage::new(&tree);
        let untrained = Storage::new(&tree); // uniform random policy
        let sc = Schedule { prune_after: u64::MAX, lcfr_until: 2_000, discount_every: 200, ..sched() };
        let tr = Trainer { tree: &tree, st: &trained, abs: &abs, sched: &sc, seed: 5 };
        tr.run(0, 20_000, 4, &AtomicBool::new(false));
        let (m, se) = head_to_head(&tree, &abs, &trained, &untrained, 20_000, 9, 4);
        assert!(m > 3.0 * se && m > 0.0, "trained vs uniform: {m} +- {se}");
    }

    #[test]
    fn discount_halves_at_k1() {
        let (rules, cfg) = small();
        let abs = StrengthAbstraction { k: 4 };
        let tree = Tree::build(&rules, &cfg, abs.buckets());
        let st = Storage::new(&tree);
        st.set_regret(&tree, 0, 0, 0, 7);
        st.set_regret(&tree, 0, 0, 1, -5);
        st.avg[0].store(9, Ordering::Relaxed);
        let sc = sched();
        Trainer { tree: &tree, st: &st, abs: &abs, sched: &sc, seed: 0 }.discount(1, 2);
        assert_eq!(st.regret(&tree, 0, 0, 0), 4); // 3.5 -> 4 (half-even)
        assert_eq!(st.regret(&tree, 0, 0, 1), -2); // -2.5 -> -2
        assert_eq!(st.avg[0].load(Ordering::Relaxed), 4); // 4.5 -> 4
    }

    #[test]
    fn paper_fraction_schedule_boundaries() {
        // "paper minute" = planned run / 11,520
        let s = Schedule::pluribus_scaled(11_520_000, 10_000); // 1,000 iterations per paper minute
        assert_eq!(s.discount_every, 10_000);
        assert_eq!(s.prune_after, 200_000);
        assert_eq!(s.lcfr_until, 400_000);
        assert_eq!(s.avg_start, 800_000);
        assert_eq!(s.snapshot_after, 800_000);
        assert_eq!(s.snapshot_every, 200_000);
        assert_eq!(s.discount_k(10_000), Some(1));
        assert_eq!(s.discount_k(390_000), Some(39));
        assert_eq!(s.discount_k(400_000), None); // t < LCFR threshold only
        assert_eq!(s.next_discount(0), Some(10_000));
        assert_eq!(s.next_discount(390_000), None);
        assert!(s.snapshot_due(800_000) && s.snapshot_due(1_000_000) && !s.snapshot_due(900_000));
        assert_eq!(s.next_snapshot(800_000), Some(1_000_000));
        assert!(s.prune_skip_river_and_terminal);
        assert!(Schedule::warmup(10_000).discount_k(5).is_none());
    }

    #[test]
    fn pruning_exception_spares_river_and_terminal_actions() {
        let (rules, cfg) = small();
        let abs = StrengthAbstraction { k: 4 };
        let tree = Tree::build(&rules, &cfg, abs.buckets());
        // regrets below C only where `select(node, action)` is true
        let run = |skip: bool, select: &dyn Fn(usize, usize) -> bool| {
            let st = Storage::new(&tree);
            for (id, nd) in tree.nodes.iter().enumerate() {
                if nd.kind != DECISION {
                    continue;
                }
                let n = nd.n as usize;
                for b in 0..tree.buckets[nd.street as usize] as usize {
                    for a in 0..n {
                        if select(id, a) {
                            st.set_regret(&tree, id as u32, b, a, -1_500_000_000);
                        }
                    }
                }
            }
            // only the selected regrets are below C; natural regrets never get near -1e9
            let sc = Schedule {
                prune_after: 0,
                explore_prob: 0.0,
                prune_c: -1_000_000_000,
                floor: -2_000_000_000,
                prune_skip_river_and_terminal: skip,
                ..sched()
            };
            let (_, ts) = Trainer { tree: &tree, st: &st, abs: &abs, sched: &sc, seed: 3 }.run(0, 300, 1, &AtomicBool::new(false));
            ts
        };
        let river = |id: usize, _a: usize| tree.nodes[id].street == 3;
        let to_terminal = |id: usize, a: usize| tree.nodes[tree.child(id as u32, a) as usize].kind != DECISION;
        let (r_on, r_off) = (run(true, &river), run(false, &river));
        assert_eq!(r_on.pruned_actions, 0, "river actions must never be pruned");
        assert!(r_off.pruned_actions > 0);
        assert!(r_on.river_reached > 0);
        let (t_on, t_off) = (run(true, &to_terminal), run(false, &to_terminal));
        assert_eq!(t_on.pruned_actions, 0, "terminal-leading actions must never be pruned");
        assert!(t_off.pruned_actions > 0);
    }

    #[test]
    fn snapshot_accumulator_sums_quantised_current_strategies() {
        let (rules, cfg) = small();
        let abs = StrengthAbstraction { k: 4 };
        let tree = Tree::build(&rules, &cfg, abs.buckets());
        let st = Storage::new(&tree);
        let sc = sched();
        let tr = Trainer { tree: &tree, st: &st, abs: &abs, sched: &sc, seed: 4 };
        tr.run(0, 300, 1, &AtomicBool::new(false));
        let dir = std::env::temp_dir().join(format!("bp_snap_{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("snap.bin");
        // remember snapshot 1's quantised strategies for allocated post-flop nodes
        let mut first = std::collections::HashMap::new();
        let mut q = [0u32; MAX_A];
        for (id, nd) in tree.nodes.iter().enumerate() {
            if nd.kind == DECISION && nd.street > 0 && st.base(id as u32).is_some() {
                let n = nd.n as usize;
                let mut v = Vec::new();
                for b in 0..4 {
                    quantised_current(&st, &tree, id as u32, b, &mut q);
                    v.extend_from_slice(&q[..n]);
                }
                first.insert(id as u32, v);
            }
        }
        assert_eq!(snapshot_add(&path, &tree, &st).unwrap(), 1);
        tr.run(300, 900, 1, &AtomicBool::new(false));
        assert_eq!(snapshot_add(&path, &tree, &st).unwrap(), 2);
        let avg = SnapshotAverage::load(&path, &tree).unwrap();
        assert_eq!(avg.total, 2);
        let mut grew = 0;
        for (id, (cnt, sums)) in &avg.recs {
            let nd = &tree.nodes[*id as usize];
            let n = nd.n as usize;
            assert!(nd.street > 0);
            for b in 0..4 {
                quantised_current(&st, &tree, *id, b, &mut q);
                for a in 0..n {
                    let want = q[a] + first.get(id).map_or(0, |v| v[b * n + a]);
                    assert_eq!(sums[b * n + a], want);
                }
            }
            assert_eq!(*cnt, if first.contains_key(id) { 2 } else { 1 });
            grew += (*cnt == 1) as u32;
        }
        assert_eq!(avg.recs.len(), first.len() + grew as usize);
        // a node first allocated after snapshot 1 is averaged with uniform for snapshot 1
        let mut out = [0.0; MAX_A];
        if let Some((id, _)) = avg.recs.iter().find(|(_, (c, _))| *c == 1) {
            avg.policy(&tree, *id, 0, &mut out);
            let n = tree.nodes[*id as usize].n as usize;
            assert!((out[..n].iter().sum::<f64>() - 1.0).abs() < 1e-9);
        }
        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn first_use_allocation_saves_memory_and_matches_dense() {
        let (rules, cfg) = small();
        let abs = StrengthAbstraction { k: 4 };
        let tree = Tree::build(&rules, &cfg, abs.buckets());
        let lazy = Storage::new(&tree);
        let dense = Storage::new_dense(&tree);
        let pre: u64 = tree.n_avg / 169 * 169; // preflop regret slots = preflop avg slots
        assert_eq!(lazy.allocated_slots(), pre);
        assert_eq!(dense.allocated_slots(), tree.n_regrets);
        let sc = sched();
        Trainer { tree: &tree, st: &lazy, abs: &abs, sched: &sc, seed: 6 }.run(0, 500, 1, &AtomicBool::new(false));
        Trainer { tree: &tree, st: &dense, abs: &abs, sched: &sc, seed: 6 }.run(0, 500, 1, &AtomicBool::new(false));
        // identical training (allocation never changes values) ...
        assert_eq!(lazy.export_dense(&tree), dense.export_dense(&tree));
        // ... with fewer slots allocated
        assert!(lazy.allocated_slots() < tree.n_regrets, "{} of {}", lazy.allocated_slots(), tree.n_regrets);
    }

    #[test]
    fn dense_v2_checkpoint_loads_into_lazy_storage() {
        // checkpoints from before first-use allocation stored every block in node order
        let (rules, cfg) = small();
        let abs = StrengthAbstraction { k: 4 };
        let tree = Tree::build(&rules, &cfg, abs.buckets());
        let st = Storage::new(&tree);
        let sc = sched();
        Trainer { tree: &tree, st: &st, abs: &abs, sched: &sc, seed: 8 }.run(0, 200, 1, &AtomicBool::new(false));
        let dir = std::env::temp_dir().join(format!("bp_v3_{}", std::process::id()));
        let path = dir.join("c.ckpt");
        let meta = CkptMeta { config: "t".into(), buckets: tree.buckets, iteration: 200, seed: 8, elapsed_secs: 1.0, sched: sc.clone(), snapshots: 0 };
        let bytes = save_checkpoint(&path, &meta, &tree, &st).unwrap();
        let st2 = Storage::new(&tree);
        assert_eq!(load_checkpoint(&path, &tree, &st2).unwrap(), meta);
        assert_eq!(st2.export_dense(&tree), st.export_dense(&tree));
        // sparse file is smaller than a dense one would be
        assert!(bytes < 4 * (tree.n_regrets + tree.n_avg));
        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn regret_summary_and_snapshots() {
        let (rules, cfg) = small();
        let abs = StrengthAbstraction { k: 4 };
        let tree = Tree::build(&rules, &cfg, abs.buckets());
        let st = Storage::new(&tree);
        let s0 = regret_summary(&tree, &st);
        assert_eq!(s0.touched, [0; 4]);
        let samples = sample_infosets(&tree, 500, 1, false);
        let a = policy_snapshot(&tree, &st, &samples);
        let sc = sched();
        Trainer { tree: &tree, st: &st, abs: &abs, sched: &sc, seed: 2 }.run(0, 300, 2, &AtomicBool::new(false));
        let s1 = regret_summary(&tree, &st);
        assert!(s1.touched.iter().all(|&x| x > 0));
        let b = policy_snapshot(&tree, &st, &samples);
        assert_eq!(snapshot_change(&tree, &samples, &a, &a), 0.0);
        assert!(snapshot_change(&tree, &samples, &a, &b) > 0.0);
    }
}
