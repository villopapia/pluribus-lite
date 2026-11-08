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
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU32, AtomicU64, Ordering};
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
    /// regret offset per node (decision nodes)
    pub roff: Vec<u64>,
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
            roff: Vec::new(),
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
        t.roff = vec![0; t.nodes.len()];
        t.aoff = vec![u64::MAX; t.nodes.len()];
        for (i, n) in t.nodes.iter().enumerate() {
            if n.kind != DECISION {
                continue;
            }
            t.roff[i] = r;
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

pub struct Storage {
    pub regrets: Vec<AtomicI32>,
    pub avg: Vec<AtomicU32>,
}

impl Storage {
    pub fn new(tree: &Tree) -> Self {
        Storage {
            regrets: (0..tree.n_regrets).map(|_| AtomicI32::new(0)).collect(),
            avg: (0..tree.n_avg).map(|_| AtomicU32::new(0)).collect(),
        }
    }

    /// Regret-matching strategy at (base, n).
    #[inline]
    pub fn current(&self, base: u64, n: usize, out: &mut [f64; MAX_A]) {
        let mut pos = 0.0;
        for a in 0..n {
            let r = self.regrets[base as usize + a].load(Ordering::Relaxed);
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

    /// Blueprint policy: preflop average strategy (falls back to current if unvisited),
    /// post-flop current strategy (Pluribus kept no post-flop average).
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
        self.current(tree.roff[node as usize] + (bucket * n) as u64, n, out);
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
    /// Unverified paper detail (pending the PDF): skip pruning on the river and on actions
    /// that end the hand. Default off = same behaviour as M1's mccfr.rs.
    pub prune_skip_river_and_terminal: bool,
}

impl Schedule {
    /// Pluribus' wall-clock schedule (unverified: prune after 200 min, LCFR until 400 min,
    /// discount every 10 min, of an ~11,520 min run) as fractions of `planned` iterations.
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
            prune_skip_river_and_terminal: false,
        }
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
            if t % self.sched.strategy_interval == 0 {
                self.update_strategy(0, i, d, ts);
            }
            let prune = t > self.sched.prune_after && d.rng.next_f64() >= self.sched.explore_prob;
            self.traverse(0, i, prune, d, ts);
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
            self.st.current(self.tree.roff[node as usize] + (b * n) as u64, n, &mut s);
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
        let b = d.bucket(self.abs, p, street, ts);
        let base = self.tree.roff[node as usize] + (b * n) as u64;
        let mut s = [0.0; MAX_A];
        self.st.current(base, n, &mut s);
        if p != i {
            let a = d.rng.sample_index(&s[..n]);
            return self.traverse(self.tree.child(node, a), i, prune, d, ts);
        }
        let mut va = [0.0; MAX_A];
        let mut explored = [true; MAX_A];
        let mut v = 0.0;
        for a in 0..n {
            let child = self.tree.child(node, a);
            if prune {
                let r = self.st.regrets[base as usize + a].load(Ordering::Relaxed);
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
                let cell = &self.st.regrets[base as usize + a];
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
        let r = &self.st.regrets;
        crate::abstraction::parallel_chunks(r.len(), threads, 1 << 20, |a, b| {
            for x in &r[a..b] {
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
const VERSION: u32 = 1;

#[derive(Clone, Debug, PartialEq)]
pub struct CkptMeta {
    pub config: String,
    pub buckets: [u32; 4],
    pub iteration: u64,
    pub seed: u64,
    pub elapsed_secs: f64,
    pub sched: Schedule,
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
        let mut buf = Vec::with_capacity(1 << 20);
        for chunk in st.regrets.chunks(1 << 18) {
            buf.clear();
            for x in chunk {
                buf.extend_from_slice(&x.load(Ordering::Relaxed).to_le_bytes());
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
    if u32::from_le_bytes(b4) != VERSION {
        return Err(bad("bad version".into()));
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
    if u[2] != tree.n_regrets || u[3] != tree.n_avg || buckets != tree.buckets {
        return Err(bad(format!("checkpoint shape {:?}/{}/{} does not match tree {:?}/{}/{}", buckets, u[2], u[3], tree.buckets, tree.n_regrets, tree.n_avg)));
    }
    let mut h = Fnv::new();
    let mut buf = vec![0u8; 1 << 20];
    for chunk in st.regrets.chunks(1 << 18) {
        let bytes = &mut buf[..chunk.len() * 4];
        r.read_exact(bytes)?;
        h.feed(bytes);
        for (x, c) in chunk.iter().zip(bytes.chunks_exact(4)) {
            x.store(i32::from_le_bytes([c[0], c[1], c[2], c[3]]), Ordering::Relaxed);
        }
    }
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
        },
    })
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
        let base = tree.roff[i] as usize;
        for b in 0..nb {
            let mut any = false;
            let mut pos = 0i64;
            for a in 0..n {
                let r = st.regrets[base + b * n + a].load(Ordering::Relaxed);
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
            prune_skip_river_and_terminal: false,
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
                if end % sc.discount_every == 0 && end < sc.lcfr_until {
                    tr.discount(end / sc.discount_every, 1);
                }
            }
        };
        if let Some(mid) = resume_at {
            step(mid, &mut t);
            let dir = std::env::temp_dir().join(format!("bp_ckpt_{}_{}", std::process::id(), mid));
            let path = dir.join("c.ckpt");
            let meta = CkptMeta { config: "t".into(), buckets: tree.buckets, iteration: t, seed: 11, elapsed_secs: 0.0, sched: sc.clone() };
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
                if end % sc.discount_every == 0 && end < sc.lcfr_until {
                    tr2.discount(end / sc.discount_every, 1);
                }
            }
            return (
                st2.regrets.iter().map(|x| x.load(Ordering::Relaxed)).collect(),
                st2.avg.iter().map(|x| x.load(Ordering::Relaxed)).collect(),
            );
        }
        step(iters, &mut t);
        (
            st.regrets.iter().map(|x| x.load(Ordering::Relaxed)).collect(),
            st.avg.iter().map(|x| x.load(Ordering::Relaxed)).collect(),
        )
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
        st.regrets[0].store(7, Ordering::Relaxed);
        st.regrets[1].store(-5, Ordering::Relaxed);
        st.avg[0].store(9, Ordering::Relaxed);
        let sc = sched();
        Trainer { tree: &tree, st: &st, abs: &abs, sched: &sc, seed: 0 }.discount(1, 2);
        assert_eq!(st.regrets[0].load(Ordering::Relaxed), 4); // 3.5 -> 4 (half-even)
        assert_eq!(st.regrets[1].load(Ordering::Relaxed), -2); // -2.5 -> -2
        assert_eq!(st.avg[0].load(Ordering::Relaxed), 4); // 4.5 -> 4
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
