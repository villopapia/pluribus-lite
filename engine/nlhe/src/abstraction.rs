//! M2 card abstraction: flop/turn k-means on equity distributions and river equity cut-offs.
//!
//! * Flop feature: for one (hole, flop), the 1,081 river equities over all (turn, river)
//!   run-outs, summarised as `FLOP_DIM` quantiles (u8, 0..255 = equity 0..1).
//! * Turn feature: the 46 river equities over all rivers, sorted (u8).
//! * Distance: L1 between features. For equal-size sorted samples or equal-count quantiles
//!   this is the 1-D earth mover's distance (times a constant).
//! * Clustering: k-medians (L1 assignment, per-coordinate median update = W1 barycentre),
//!   k-means++ seeding, fixed seed, deterministic for any thread count.
//! * River: 1-D k-means on the exact distribution of river equity over all (board, hand)
//!   pairs. The k-1 midpoints between sorted centroids are the bucket cut-offs.
//! * Buckets are relabelled so a higher id means a higher mean equity.

use crate::equity::{combo_cards, RiverBatch, COMBOS};
use crate::index::hand_index;
use cfr::rng::Rng;
use std::fs::{self, File};
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;
use std::sync::atomic::{AtomicU16, AtomicU8, AtomicUsize, Ordering};

pub const FLOP_DIM: usize = 50;
pub const TURN_DIM: usize = 46;
pub const RIVER_BINS: usize = 65_536;

#[inline]
pub fn q8(eq: f32) -> u8 {
    (eq * 255.0).round() as u8
}

#[inline]
pub fn q16(eq: f32) -> u16 {
    (eq * 65_535.0).round() as u16
}

// ---------------------------------------------------------------- features

pub struct FeatureScratch {
    pub rb: RiverBatch,
    hist: Vec<[u16; 256]>,
    samples: Vec<[u8; 48]>,
    count: Vec<u8>,
    combos: Vec<(u8, u8)>,
}

impl Default for FeatureScratch {
    fn default() -> Self {
        Self::new()
    }
}

impl FeatureScratch {
    pub fn new() -> Self {
        FeatureScratch {
            rb: RiverBatch::new(),
            hist: vec![[0; 256]; COMBOS],
            samples: vec![[0; 48]; COMBOS],
            count: vec![0; COMBOS],
            combos: combo_cards(),
        }
    }
}

/// Flop features for every hand disjoint from `board`. Calls `f(combo_index, feature)`.
pub fn flop_features(board: [u8; 3], s: &mut FeatureScratch, mut f: impl FnMut(usize, &[u8; FLOP_DIM])) {
    for h in s.hist.iter_mut() {
        *h = [0; 256];
    }
    let rest: Vec<u8> = (0..52u8).filter(|c| !board.contains(c)).collect();
    for i in 0..rest.len() {
        for j in i + 1..rest.len() {
            let b5 = [board[0], board[1], board[2], rest[i], rest[j]];
            s.rb.compute(&b5);
            for (ci, &e) in s.rb.equity.iter().enumerate() {
                if !e.is_nan() {
                    s.hist[ci][q8(e) as usize] += 1;
                }
            }
        }
    }
    let mut feat = [0u8; FLOP_DIM];
    for (ci, &(a, b)) in s.combos.iter().enumerate() {
        if board.contains(&a) || board.contains(&b) {
            continue;
        }
        let h = &s.hist[ci];
        let total: u32 = 1081; // C(47, 2) run-outs
        debug_assert_eq!(h.iter().map(|&x| x as u32).sum::<u32>(), total);
        let mut cum = 0u32;
        let mut v = 0usize;
        for (k, slot) in feat.iter_mut().enumerate() {
            // value at rank floor((k + 0.5) * total / FLOP_DIM)
            let target = ((2 * k as u32 + 1) * total) / (2 * FLOP_DIM as u32);
            while cum + h[v] as u32 <= target {
                cum += h[v] as u32;
                v += 1;
            }
            *slot = v as u8;
        }
        f(ci, &feat);
    }
}

/// Turn features (46 sorted river equities) for every hand disjoint from `board`.
pub fn turn_features(board: [u8; 4], s: &mut FeatureScratch, mut f: impl FnMut(usize, &[u8; TURN_DIM])) {
    for c in s.count.iter_mut() {
        *c = 0;
    }
    for r in 0..52u8 {
        if board.contains(&r) {
            continue;
        }
        s.rb.compute(&[board[0], board[1], board[2], board[3], r]);
        for (ci, &e) in s.rb.equity.iter().enumerate() {
            if !e.is_nan() {
                let n = s.count[ci] as usize;
                s.samples[ci][n] = q8(e);
                s.count[ci] += 1;
            }
        }
    }
    let mut feat = [0u8; TURN_DIM];
    for (ci, &(a, b)) in s.combos.iter().enumerate() {
        if board.contains(&a) || board.contains(&b) {
            continue;
        }
        debug_assert_eq!(s.count[ci] as usize, TURN_DIM);
        feat.copy_from_slice(&s.samples[ci][..TURN_DIM]);
        feat.sort_unstable();
        f(ci, &feat);
    }
}

// ---------------------------------------------------------------- k-medians

#[inline]
pub fn l1(a: &[u8], b: &[u8]) -> u32 {
    let mut s = 0u32;
    let mut ca = a.chunks_exact(16);
    let mut cb = b.chunks_exact(16);
    for (x, y) in (&mut ca).zip(&mut cb) {
        let mut acc = 0u16;
        for i in 0..16 {
            acc += x[i].abs_diff(y[i]) as u16;
        }
        s += acc as u32;
    }
    for (x, y) in ca.remainder().iter().zip(cb.remainder()) {
        s += x.abs_diff(*y) as u32;
    }
    s
}

#[derive(Clone, Debug, PartialEq)]
pub struct Centroids {
    pub k: usize,
    pub dim: usize,
    pub data: Vec<u8>,
}

impl Centroids {
    #[inline]
    pub fn get(&self, c: usize) -> &[u8] {
        &self.data[c * self.dim..(c + 1) * self.dim]
    }

    /// Nearest centroid (lowest index on ties) and its distance.
    #[inline]
    pub fn nearest(&self, x: &[u8]) -> (usize, u32) {
        let mut best = (0usize, u32::MAX);
        for c in 0..self.k {
            let d = l1(x, self.get(c));
            if d < best.1 {
                best = (c, d);
            }
        }
        best
    }

    /// Relabel so centroid ids increase with mean value (mean equity).
    pub fn sort_by_mean(&mut self) {
        let mut ids: Vec<usize> = (0..self.k).collect();
        let sum = |c: usize| self.get(c).iter().map(|&v| v as u32).sum::<u32>();
        ids.sort_by_key(|&c| (sum(c), c));
        let mut data = Vec::with_capacity(self.data.len());
        for c in ids {
            data.extend_from_slice(self.get(c));
        }
        self.data = data;
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct FitStats {
    pub iterations: usize,
    pub changed_last: usize,
    pub mean_l1: f64,
    pub empty_reseeds: usize,
}

pub fn parallel_chunks(n: usize, threads: usize, chunk: usize, f: impl Fn(usize, usize) + Sync) {
    let next = AtomicUsize::new(0);
    std::thread::scope(|s| {
        for _ in 0..threads.max(1) {
            s.spawn(|| loop {
                let start = next.fetch_add(chunk, Ordering::Relaxed);
                if start >= n {
                    break;
                }
                f(start, (start + chunk).min(n));
            });
        }
    });
}

/// k-medians fit. `points` is n * dim bytes. Deterministic for a given seed and any
/// thread count (integer reductions only).
pub fn fit_kmedians(points: &[u8], dim: usize, k: usize, max_iters: usize, seed: u64, threads: usize) -> (Centroids, FitStats) {
    let n = points.len() / dim;
    assert!(n >= k, "need at least k points");
    let pt = |i: usize| &points[i * dim..(i + 1) * dim];
    let mut rng = Rng::new(seed);

    // k-means++ seeding on a subsample (L1 distance, D(x) weighting)
    let sub_n = n.min(20_000);
    let sub: Vec<usize> = if sub_n == n {
        (0..n).collect()
    } else {
        (0..sub_n).map(|_| (rng.next_u64() % n as u64) as usize).collect()
    };
    let mut cent = Centroids { k, dim, data: Vec::with_capacity(k * dim) };
    cent.data.extend_from_slice(pt(sub[(rng.next_u64() % sub_n as u64) as usize]));
    let mut dmin: Vec<f64> = sub.iter().map(|&i| l1(pt(i), cent.get(0)) as f64).collect();
    for c in 1..k {
        let total: f64 = dmin.iter().sum();
        let pick = if total > 0.0 { rng.sample_index(&dmin) } else { (rng.next_u64() % sub_n as u64) as usize };
        cent.data.extend_from_slice(pt(sub[pick]));
        for (j, &i) in sub.iter().enumerate() {
            let d = l1(pt(i), cent.get(c)) as f64;
            if d < dmin[j] {
                dmin[j] = d;
            }
        }
    }

    let assign = vec![0u16; n];
    let dist = vec![0u32; n];
    let assign_a: Vec<AtomicU16> = assign.into_iter().map(AtomicU16::new).collect();
    let dist_a: Vec<std::sync::atomic::AtomicU32> = dist.into_iter().map(std::sync::atomic::AtomicU32::new).collect();
    let mut stats = FitStats::default();
    for it in 0..max_iters {
        // assignment
        let changed = AtomicUsize::new(0);
        {
            let cent = &cent;
            parallel_chunks(n, threads, 4096, |a, b| {
                let mut ch = 0;
                for i in a..b {
                    let (c, d) = cent.nearest(pt(i));
                    if it == 0 || assign_a[i].load(Ordering::Relaxed) != c as u16 {
                        ch += 1;
                    }
                    assign_a[i].store(c as u16, Ordering::Relaxed);
                    dist_a[i].store(d, Ordering::Relaxed);
                }
                changed.fetch_add(ch, Ordering::Relaxed);
            });
        }
        let changed = changed.into_inner();
        stats.iterations = it + 1;
        stats.changed_last = changed;
        stats.mean_l1 = dist_a.iter().map(|d| d.load(Ordering::Relaxed) as f64).sum::<f64>() / n as f64;
        // median update from per-(cluster, dim) value histograms
        let mut hist = vec![0u32; k * dim * 256];
        let mut sizes = vec![0u32; k];
        for i in 0..n {
            let c = assign_a[i].load(Ordering::Relaxed) as usize;
            sizes[c] += 1;
            let base = c * dim * 256;
            for (d, &v) in pt(i).iter().enumerate() {
                hist[base + d * 256 + v as usize] += 1;
            }
        }
        let mut new = cent.data.clone();
        let mut empties: Vec<usize> = Vec::new();
        for c in 0..k {
            if sizes[c] == 0 {
                empties.push(c);
                continue;
            }
            let half = (sizes[c] + 1) / 2; // lower median
            for d in 0..dim {
                let h = &hist[c * dim * 256 + d * 256..c * dim * 256 + d * 256 + 256];
                let mut cum = 0u32;
                for (v, &cnt) in h.iter().enumerate() {
                    cum += cnt;
                    if cum >= half {
                        new[c * dim + d] = v as u8;
                        break;
                    }
                }
            }
        }
        // empty clusters: reseed at the currently worst-fitted points (deterministic order)
        if !empties.is_empty() {
            let mut far: Vec<(u32, usize)> = (0..n).map(|i| (dist_a[i].load(Ordering::Relaxed), i)).collect();
            far.sort_unstable_by(|a, b| b.cmp(a));
            for (j, &c) in empties.iter().enumerate() {
                new[c * dim..(c + 1) * dim].copy_from_slice(pt(far[j].1));
            }
            stats.empty_reseeds += empties.len();
        }
        let converged = new == cent.data;
        cent.data = new;
        if converged || (it > 0 && empties.is_empty() && changed * 1000 <= n) {
            break;
        }
    }
    cent.sort_by_mean();
    (cent, stats)
}

// ---------------------------------------------------------------- river cut-offs

/// 1-D k-means on a weighted histogram of quantised equity (bins over [0, 1]).
/// Returns k-1 increasing cut-offs in equity units; bucket(e) = #cut-offs <= e.
pub fn river_cutoffs(hist: &[u64], k: usize, max_iters: usize) -> Vec<f32> {
    let nb = hist.len();
    let total: u64 = hist.iter().sum();
    assert!(total > 0 && k >= 2);
    let x = |b: usize| b as f64 / (nb - 1) as f64;
    // init: weighted quantile midpoints
    let mut cent = Vec::with_capacity(k);
    let mut cum = 0u64;
    let mut b = 0usize;
    for c in 0..k {
        let target = ((2 * c as u64 + 1) * total) / (2 * k as u64);
        while b < nb - 1 && cum + hist[b] <= target {
            cum += hist[b];
            b += 1;
        }
        cent.push(x(b));
    }
    for _ in 0..max_iters {
        cent.sort_by(|a, b| a.partial_cmp(b).unwrap());
        cent.dedup();
        let cuts: Vec<f64> = cent.windows(2).map(|w| 0.5 * (w[0] + w[1])).collect();
        let mut sw = vec![0f64; cent.len()];
        let mut sx = vec![0f64; cent.len()];
        let mut c = 0usize;
        for (bi, &w) in hist.iter().enumerate() {
            if w == 0 {
                continue;
            }
            let v = x(bi);
            while c < cuts.len() && v >= cuts[c] {
                c += 1;
            }
            sw[c] += w as f64;
            sx[c] += w as f64 * v;
        }
        let new: Vec<f64> = (0..cent.len()).map(|i| if sw[i] > 0.0 { sx[i] / sw[i] } else { cent[i] }).collect();
        let moved = new.iter().zip(&cent).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        cent = new;
        if moved < 1e-9 {
            break;
        }
    }
    cent.sort_by(|a, b| a.partial_cmp(b).unwrap());
    cent.dedup();
    cent.windows(2).map(|w| (0.5 * (w[0] + w[1])) as f32).collect()
}

#[inline]
pub fn river_bucket(cutoffs: &[f32], equity: f32) -> usize {
    cutoffs.partition_point(|&c| c <= equity)
}

// ---------------------------------------------------------------- tables and files

/// Bucket table writable from many threads (u8 if k <= 256, else u16).
pub enum BucketTable {
    U8(Vec<AtomicU8>),
    U16(Vec<AtomicU16>),
}

impl BucketTable {
    pub fn new(entries: usize, k: usize) -> Self {
        if k <= 256 {
            BucketTable::U8((0..entries).map(|_| AtomicU8::new(0)).collect())
        } else {
            BucketTable::U16((0..entries).map(|_| AtomicU16::new(0)).collect())
        }
    }
    #[inline]
    pub fn set(&self, i: u64, v: usize) {
        match self {
            BucketTable::U8(t) => t[i as usize].store(v as u8, Ordering::Relaxed),
            BucketTable::U16(t) => t[i as usize].store(v as u16, Ordering::Relaxed),
        }
    }
    pub fn width(&self) -> u8 {
        match self {
            BucketTable::U8(_) => 1,
            BucketTable::U16(_) => 2,
        }
    }
    pub fn to_bytes(&self) -> Vec<u8> {
        match self {
            BucketTable::U8(t) => t.iter().map(|x| x.load(Ordering::Relaxed)).collect(),
            BucketTable::U16(t) => t.iter().flat_map(|x| x.load(Ordering::Relaxed).to_le_bytes()).collect(),
        }
    }
}

pub fn fnv1a64(data: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in data {
        h ^= b as u64;
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h
}

const MAGIC: &[u8; 4] = b"PLAB";
const VERSION: u32 = 1;

#[derive(Clone, Debug, PartialEq)]
pub struct AbsHeader {
    pub kind: String,
    pub config: String,
    pub k: u32,
    pub width: u8,
    pub entries: u64,
    pub seed: u64,
    pub checksum: u64,
}

/// Write header + payload atomically (tmp file + rename).
pub fn write_abs_file(path: &Path, h: &AbsHeader, payload: &[u8]) -> io::Result<()> {
    if let Some(d) = path.parent() {
        fs::create_dir_all(d)?;
    }
    let tmp = path.with_extension("tmp");
    {
        let mut w = BufWriter::new(File::create(&tmp)?);
        w.write_all(MAGIC)?;
        w.write_all(&VERSION.to_le_bytes())?;
        for s in [&h.kind, &h.config] {
            w.write_all(&(s.len() as u32).to_le_bytes())?;
            w.write_all(s.as_bytes())?;
        }
        w.write_all(&h.k.to_le_bytes())?;
        w.write_all(&[h.width])?;
        w.write_all(&h.entries.to_le_bytes())?;
        w.write_all(&h.seed.to_le_bytes())?;
        w.write_all(&fnv1a64(payload).to_le_bytes())?;
        w.write_all(&(payload.len() as u64).to_le_bytes())?;
        w.write_all(payload)?;
        w.flush()?;
        w.get_ref().sync_all()?;
    }
    fs::rename(&tmp, path)
}

/// Read and verify (magic, version, checksum). Returns header and payload.
pub fn read_abs_file(path: &Path) -> io::Result<(AbsHeader, Vec<u8>)> {
    let bad = |m: &str| io::Error::new(io::ErrorKind::InvalidData, m.to_string());
    let mut r = BufReader::new(File::open(path)?);
    let mut b4 = [0u8; 4];
    let mut b8 = [0u8; 8];
    r.read_exact(&mut b4)?;
    if &b4 != MAGIC {
        return Err(bad("bad magic"));
    }
    r.read_exact(&mut b4)?;
    if u32::from_le_bytes(b4) != VERSION {
        return Err(bad("bad version"));
    }
    let mut strs = Vec::new();
    for _ in 0..2 {
        r.read_exact(&mut b4)?;
        let mut s = vec![0u8; u32::from_le_bytes(b4) as usize];
        r.read_exact(&mut s)?;
        strs.push(String::from_utf8(s).map_err(|_| bad("utf8"))?);
    }
    r.read_exact(&mut b4)?;
    let k = u32::from_le_bytes(b4);
    let mut w1 = [0u8; 1];
    r.read_exact(&mut w1)?;
    r.read_exact(&mut b8)?;
    let entries = u64::from_le_bytes(b8);
    r.read_exact(&mut b8)?;
    let seed = u64::from_le_bytes(b8);
    r.read_exact(&mut b8)?;
    let checksum = u64::from_le_bytes(b8);
    r.read_exact(&mut b8)?;
    let mut payload = vec![0u8; u64::from_le_bytes(b8) as usize];
    r.read_exact(&mut payload)?;
    if fnv1a64(&payload) != checksum {
        return Err(bad("checksum mismatch"));
    }
    let h = AbsHeader { kind: strs[0].clone(), config: strs[1].clone(), k, width: w1[0], entries, seed, checksum };
    Ok((h, payload))
}

/// Assign every (hole, board) of one street into `table` using `cent`.
/// `boards` lists the boards to process; `feat` computes features for one board.
pub fn assign_table<const B: usize, const D: usize>(
    boards: &[[u8; B]],
    cent: &Centroids,
    table: &BucketTable,
    threads: usize,
    feat: fn([u8; B], &mut FeatureScratch, &mut dyn FnMut(usize, &[u8; D])),
) -> Vec<u64> {
    let counts: Vec<std::sync::atomic::AtomicU64> = (0..cent.k).map(|_| std::sync::atomic::AtomicU64::new(0)).collect();
    let combos = combo_cards();
    parallel_chunks(boards.len(), threads, 16, |a, b| {
        let mut s = FeatureScratch::new();
        let mut local = vec![0u64; cent.k];
        for board in &boards[a..b] {
            feat(*board, &mut s, &mut |ci, x| {
                let (c, _) = cent.nearest(x);
                table.set(hand_index(combos[ci], board), c);
                local[c] += 1;
            });
        }
        for (c, v) in local.iter().enumerate() {
            counts[c].fetch_add(*v, Ordering::Relaxed);
        }
    });
    counts.into_iter().map(|c| c.into_inner()).collect()
}

pub fn flop_feat_dyn(b: [u8; 3], s: &mut FeatureScratch, f: &mut dyn FnMut(usize, &[u8; FLOP_DIM])) {
    flop_features(b, s, f)
}
pub fn turn_feat_dyn(b: [u8; 4], s: &mut FeatureScratch, f: &mut dyn FnMut(usize, &[u8; TURN_DIM])) {
    turn_features(b, s, f)
}

/// All k-card boards in colex order.
pub fn all_boards<const B: usize>() -> Vec<[u8; B]> {
    let mut out = Vec::new();
    let mut cur = [0u8; B];
    fn rec<const B: usize>(start: u8, depth: usize, cur: &mut [u8; B], out: &mut Vec<[u8; B]>) {
        if depth == B {
            out.push(*cur);
            return;
        }
        for c in start..52 {
            cur[depth] = c;
            rec(c + 1, depth + 1, cur, out);
        }
    }
    rec(0, 0, &mut cur, &mut out);
    out
}

/// Gather features of all hands on a sample of boards into one flat point array.
pub fn sample_points<const B: usize, const D: usize>(
    boards: &[[u8; B]],
    threads: usize,
    feat: fn([u8; B], &mut FeatureScratch, &mut dyn FnMut(usize, &[u8; D])),
) -> Vec<u8> {
    let parts: Vec<std::sync::Mutex<Vec<u8>>> = boards.iter().map(|_| std::sync::Mutex::new(Vec::new())).collect();
    parallel_chunks(boards.len(), threads, 1, |a, b| {
        let mut s = FeatureScratch::new();
        for i in a..b {
            let mut v = Vec::new();
            feat(boards[i], &mut s, &mut |_, x| v.extend_from_slice(x));
            *parts[i].lock().unwrap() = v;
        }
    });
    parts.into_iter().flat_map(|m| m.into_inner().unwrap()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::equity::{combo_index, river_equity_single};

    #[test]
    fn l1_matches_naive() {
        let mut rng = Rng::new(1);
        for len in [3usize, 16, 46, 50, 64] {
            let a: Vec<u8> = (0..len).map(|_| rng.next_u64() as u8).collect();
            let b: Vec<u8> = (0..len).map(|_| rng.next_u64() as u8).collect();
            let naive: u32 = a.iter().zip(&b).map(|(x, y)| (*x as i32 - *y as i32).unsigned_abs()).sum();
            assert_eq!(l1(&a, &b), naive);
        }
    }

    #[test]
    fn kmedians_recovers_separated_clusters_deterministically() {
        let mut rng = Rng::new(2);
        let dim = 8;
        let centers = [20u8, 120, 220];
        let mut pts = Vec::new();
        for i in 0..3000 {
            let c = centers[i % 3];
            for _ in 0..dim {
                pts.push(c.saturating_add((rng.next_u64() % 9) as u8).saturating_sub(4));
            }
        }
        let (c1, st) = fit_kmedians(&pts, dim, 3, 50, 7, 1);
        let (c4, _) = fit_kmedians(&pts, dim, 3, 50, 7, 4);
        assert_eq!(c1, c4, "thread count changed the result");
        let means: Vec<u32> = (0..3).map(|c| c1.get(c).iter().map(|&v| v as u32).sum::<u32>() / dim as u32).collect();
        for (m, want) in means.iter().zip([20u32, 120, 220]) {
            assert!(m.abs_diff(want) <= 2, "{means:?}");
        }
        assert!(st.mean_l1 < 3.0 * dim as f64);
    }

    #[test]
    fn median_update_is_coordinatewise_median() {
        // one cluster: centroid must be the per-coordinate lower median
        let pts = vec![1u8, 50, 9, 3, 2, 10, 7, 60, 5, 8, 4, 0];
        let (c, _) = fit_kmedians(&pts, 2, 1, 5, 0, 1);
        // coords: [1,9,2,7,5,4] -> sorted 1,2,4,5,7,9 lower median 4 ; [50,3,10,60,8,0] -> 0,3,8,10,50,60 -> 8
        assert_eq!(c.data, vec![4, 8]);
    }

    #[test]
    fn turn_features_match_direct_computation() {
        let mut s = FeatureScratch::new();
        let board = [0u8, 13, 26, 39]; // 2c 5d 8h Js
        let mut got = std::collections::HashMap::new();
        turn_features(board, &mut s, |ci, x| {
            got.insert(ci, *x);
        });
        assert_eq!(got.len(), 1128);
        for hole in [(48u8, 49u8), (4, 9), (50, 51)] {
            let mut want: Vec<u8> = (0..52u8)
                .filter(|r| !board.contains(r) && *r != hole.0 && *r != hole.1)
                .map(|r| q8(river_equity_single(&[board[0], board[1], board[2], board[3], r], [hole.0, hole.1]) as f32))
                .collect();
            want.sort_unstable();
            assert_eq!(&got[&combo_index(hole.0, hole.1)][..], &want[..]);
        }
    }

    #[test]
    fn flop_features_match_direct_quantiles() {
        let mut s = FeatureScratch::new();
        let board = [51u8, 30, 7]; // As 9h 3s
        let mut got = std::collections::HashMap::new();
        flop_features(board, &mut s, |ci, x| {
            got.insert(ci, *x);
        });
        assert_eq!(got.len(), 1176);
        let hole = (48u8, 44u8); // Ac Kc
        let rest: Vec<u8> = (0..52u8).filter(|c| !board.contains(c) && *c != hole.0 && *c != hole.1).collect();
        let mut eqs = Vec::new();
        for i in 0..rest.len() {
            for j in i + 1..rest.len() {
                let b5 = [board[0], board[1], board[2], rest[i], rest[j]];
                eqs.push(q8(river_equity_single(&b5, [hole.0, hole.1]) as f32));
            }
        }
        eqs.sort_unstable();
        assert_eq!(eqs.len(), 1081);
        let want: Vec<u8> = (0..FLOP_DIM).map(|k| eqs[((2 * k + 1) * 1081) / (2 * FLOP_DIM)]).collect();
        assert_eq!(&got[&combo_index(hole.0, hole.1)][..], &want[..]);
        // features are monotone (quantiles)
        assert!(got.values().all(|f| f.windows(2).all(|w| w[0] <= w[1])));
    }

    #[test]
    fn river_cutoffs_split_point_masses() {
        let mut h = vec![0u64; 1001];
        h[100] = 500;
        h[500] = 300;
        h[900] = 200;
        let cuts = river_cutoffs(&h, 3, 100);
        assert_eq!(cuts.len(), 2);
        assert!((cuts[0] - 0.3).abs() < 1e-6 && (cuts[1] - 0.7).abs() < 1e-6, "{cuts:?}");
        assert_eq!(river_bucket(&cuts, 0.1), 0);
        assert_eq!(river_bucket(&cuts, 0.5), 1);
        assert_eq!(river_bucket(&cuts, 0.95), 2);
    }

    #[test]
    fn abs_file_roundtrip_and_corruption_detected() {
        let dir = std::env::temp_dir().join(format!("plab_{}", std::process::id()));
        let p = dir.join("t.bkt");
        let h = AbsHeader { kind: "flop".into(), config: "m2_11g".into(), k: 256, width: 1, entries: 5, seed: 9, checksum: 0 };
        write_abs_file(&p, &h, &[1, 2, 3, 4, 5]).unwrap();
        let (h2, pl) = read_abs_file(&p).unwrap();
        assert_eq!(pl, vec![1, 2, 3, 4, 5]);
        assert_eq!((h2.kind.as_str(), h2.k, h2.entries), ("flop", 256, 5));
        // corrupt the last payload byte
        let mut bytes = fs::read(&p).unwrap();
        *bytes.last_mut().unwrap() ^= 0xff;
        fs::write(&p, bytes).unwrap();
        assert!(read_abs_file(&p).is_err());
        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn all_boards_counts() {
        assert_eq!(all_boards::<3>().len(), 22_100);
        assert_eq!(all_boards::<4>().len(), 270_725);
    }

    #[test]
    fn sort_by_mean_relabels() {
        let mut c = Centroids { k: 3, dim: 2, data: vec![200, 200, 10, 10, 100, 100] };
        c.sort_by_mean();
        assert_eq!(c.data, vec![10, 10, 100, 100, 200, 200]);
    }
}
