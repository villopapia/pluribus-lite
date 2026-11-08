//! M2 card abstraction following Johanson, Burch, Valenzano & Bowling (AAMAS 2013), the
//! method Pluribus cites for its information abstraction ([26], supplement p. 13).
//!
//! * Flop: for one (hole, flop), the **hand-strength histogram** over all (turn, river)
//!   run-outs (1,081 river equities), `HIST_BINS` equal-width bins on [0, 1].
//! * Turn: the hand-strength histogram over the 46 rivers.
//! * Distance: **earth mover's distance**. Points are stored as normalised CDFs, so EMD = L1
//!   between CDFs (times the bin width) and the k-means mean centroid is the CDF of the mean
//!   histogram. Clustering: k-means (`kmeans.rs`), k-means++ seeding.
//! * River: **OCHS**: equity against each of Johanson's 8 opponent clusters (`ochs.rs`),
//!   k-means with L2 distance. Buckets are computed at training time from the board (no table).
//! * Buckets are labelled in order of coordinate sum (for CDFs: higher id = lower expected
//!   strength... see `KMeans::from_centroids`; for OCHS: higher id = stronger).
//!
//! The exact feature choices (bin count, single k-means run, fit on a 1,000-board sample) are
//! recorded as deviations in docs/deviations.md (D-INFO-1, D-INFO-2).

use crate::equity::{combo_cards, HeroRiver, RiverBatch, COMBOS};
use crate::index::hand_index;
use crate::kmeans::KMeans;
use crate::ochs::{combo_clusters, N_CLUSTERS};
use std::fs::{self, File};
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;
use std::sync::atomic::{AtomicU16, AtomicU8, AtomicUsize, Ordering};

/// Histogram bins for flop/turn hand-strength distributions.
pub const HIST_BINS: usize = 50;
pub const RIVER_BINS: usize = 65_536;

#[inline]
pub fn hs_bin(eq: f32) -> usize {
    ((eq * HIST_BINS as f32) as usize).min(HIST_BINS - 1)
}

#[inline]
pub fn q16(eq: f32) -> u16 {
    (eq * 65_535.0).round() as u16
}

// ---------------------------------------------------------------- features

pub struct FeatureScratch {
    pub rb: RiverBatch,
    hr: HeroRiver,
    hist: Vec<[u16; HIST_BINS]>,
    combos: Vec<(u8, u8)>,
    clusters: Vec<u8>,
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
            hr: HeroRiver::new(),
            hist: vec![[0; HIST_BINS]; COMBOS],
            combos: combo_cards(),
            clusters: combo_clusters(),
        }
    }
}

/// Normalised CDF of a histogram with `total` samples.
#[inline]
fn cdf(h: &[u16; HIST_BINS], total: f32, out: &mut [f32; HIST_BINS]) {
    let mut c = 0u32;
    for b in 0..HIST_BINS {
        c += h[b] as u32;
        out[b] = c as f32 / total;
    }
}

/// Flop features (hand-strength histogram CDF over 1,081 run-outs) for every hand disjoint
/// from `board`. Calls `f(combo_index, feature)`.
pub fn flop_features(board: [u8; 3], s: &mut FeatureScratch, mut f: impl FnMut(usize, &[f32; HIST_BINS])) {
    s.hist.iter_mut().for_each(|h| *h = [0; HIST_BINS]);
    let rest: Vec<u8> = (0..52u8).filter(|c| !board.contains(c)).collect();
    for i in 0..rest.len() {
        for j in i + 1..rest.len() {
            s.rb.compute(&[board[0], board[1], board[2], rest[i], rest[j]]);
            for (ci, &e) in s.rb.equity.iter().enumerate() {
                if !e.is_nan() {
                    s.hist[ci][hs_bin(e)] += 1;
                }
            }
        }
    }
    let mut feat = [0f32; HIST_BINS];
    for (ci, &(a, b)) in s.combos.iter().enumerate() {
        if board.contains(&a) || board.contains(&b) {
            continue;
        }
        cdf(&s.hist[ci], 1081.0, &mut feat); // C(47, 2) run-outs
        f(ci, &feat);
    }
}

/// Turn features (hand-strength histogram CDF over the 46 rivers).
pub fn turn_features(board: [u8; 4], s: &mut FeatureScratch, mut f: impl FnMut(usize, &[f32; HIST_BINS])) {
    s.hist.iter_mut().for_each(|h| *h = [0; HIST_BINS]);
    for r in 0..52u8 {
        if board.contains(&r) {
            continue;
        }
        s.rb.compute(&[board[0], board[1], board[2], board[3], r]);
        for (ci, &e) in s.rb.equity.iter().enumerate() {
            if !e.is_nan() {
                s.hist[ci][hs_bin(e)] += 1;
            }
        }
    }
    let mut feat = [0f32; HIST_BINS];
    for (ci, &(a, b)) in s.combos.iter().enumerate() {
        if board.contains(&a) || board.contains(&b) {
            continue;
        }
        cdf(&s.hist[ci], 46.0, &mut feat);
        f(ci, &feat);
    }
}

/// River OCHS features (equity against Johanson's 8 opponent clusters) for every hand.
pub fn river_features(board: [u8; 5], s: &mut FeatureScratch, mut f: impl FnMut(usize, &[f32; N_CLUSTERS])) {
    s.hr.prepare(&mut s.rb, &board);
    let mut feat = [0f32; N_CLUSTERS];
    for (ci, &(a, b)) in s.combos.iter().enumerate() {
        if board.contains(&a) || board.contains(&b) {
            continue;
        }
        s.hr.ochs(a, b, &s.clusters, &mut feat);
        f(ci, &feat);
    }
}

pub fn flop_feat_dyn(b: [u8; 3], s: &mut FeatureScratch, f: &mut dyn FnMut(usize, &[f32; HIST_BINS])) {
    flop_features(b, s, f)
}
pub fn turn_feat_dyn(b: [u8; 4], s: &mut FeatureScratch, f: &mut dyn FnMut(usize, &[f32; HIST_BINS])) {
    turn_features(b, s, f)
}
pub fn river_feat_dyn(b: [u8; 5], s: &mut FeatureScratch, f: &mut dyn FnMut(usize, &[f32; N_CLUSTERS])) {
    river_features(b, s, f)
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

// ---------------------------------------------------------------- legacy 1-D river cut-offs

/// Legacy river abstraction (1-D equity against a random hand), kept only to measure the
/// OCHS deviation (D-INFO-2). 1-D k-means (Lloyd) on a weighted histogram of quantised
/// equity, with empty/duplicate centroids reseeded at the highest-error bin, so there are
/// exactly k distinct non-empty clusters when the histogram has >= k non-empty bins.
/// Returns k-1 increasing cut-offs; bucket(e) = #cut-offs <= e.
pub fn river_cutoffs(hist: &[u64], k: usize, max_iters: usize) -> Vec<f32> {
    let nb = hist.len();
    let total: u64 = hist.iter().sum();
    assert!(total > 0 && k >= 2);
    let nonempty: Vec<usize> = (0..nb).filter(|&b| hist[b] > 0).collect();
    let k = k.min(nonempty.len());
    let x = |b: usize| b as f64 / (nb - 1) as f64;
    let mut cent: Vec<f64> = Vec::with_capacity(k);
    let mut cum = 0u64;
    let mut bi = 0usize;
    for c in 0..k {
        let target = ((2 * c as u64 + 1) * total) / (2 * k as u64);
        while bi < nb - 1 && cum + hist[bi] <= target {
            cum += hist[bi];
            bi += 1;
        }
        cent.push(x(bi));
    }
    let assign = |cent: &[f64]| -> Vec<usize> {
        let cuts: Vec<f64> = cent.windows(2).map(|w| 0.5 * (w[0] + w[1])).collect();
        let mut c = 0usize;
        nonempty
            .iter()
            .map(|&b| {
                let v = x(b);
                while c < cuts.len() && v >= cuts[c] {
                    c += 1;
                }
                c
            })
            .collect()
    };
    for _ in 0..max_iters {
        cent.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let owner = assign(&cent);
        let mut sw = vec![0f64; k];
        let mut sx = vec![0f64; k];
        for (j, &b) in nonempty.iter().enumerate() {
            sw[owner[j]] += hist[b] as f64;
            sx[owner[j]] += hist[b] as f64 * x(b);
        }
        let mut new: Vec<f64> = (0..k).map(|i| if sw[i] > 0.0 { sx[i] / sw[i] } else { f64::NAN }).collect();
        let mut bad: Vec<usize> = (0..k).filter(|&i| new[i].is_nan()).collect();
        {
            let mut idx: Vec<usize> = (0..k).filter(|&i| !new[i].is_nan()).collect();
            idx.sort_by(|&a, &b| new[a].partial_cmp(&new[b]).unwrap());
            for w in idx.windows(2) {
                if new[w[1]] - new[w[0]] < 1e-12 {
                    bad.push(w[1]);
                }
            }
        }
        let repaired = !bad.is_empty();
        if repaired {
            let mut err: Vec<(f64, usize)> = nonempty
                .iter()
                .enumerate()
                .map(|(j, &b)| (hist[b] as f64 * (x(b) - cent[owner[j]]).powi(2), b))
                .collect();
            err.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap().then(a.1.cmp(&b.1)));
            let mut used: Vec<f64> = new.iter().copied().filter(|v| !v.is_nan()).collect();
            let mut e = err.into_iter();
            for &i in &bad {
                for (_, b) in e.by_ref() {
                    let v = x(b);
                    if used.iter().all(|&u| (u - v).abs() > 1e-12) {
                        new[i] = v;
                        used.push(v);
                        break;
                    }
                }
            }
        }
        let moved = new.iter().zip(&cent).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        cent = new;
        if !repaired && moved < 1e-12 {
            break;
        }
    }
    cent.sort_by(|a, b| a.partial_cmp(b).unwrap());
    cent.dedup_by(|a, b| (*a - *b).abs() < 1e-12);
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

pub fn centroids_to_bytes(c: &[f32]) -> Vec<u8> {
    c.iter().flat_map(|x| x.to_le_bytes()).collect()
}

pub fn centroids_from_bytes(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

/// Assign every (hole, board) of one street into `table` using the fitted k-means.
pub fn assign_table<const B: usize, const D: usize>(
    boards: &[[u8; B]],
    km: &KMeans,
    table: &BucketTable,
    threads: usize,
    feat: fn([u8; B], &mut FeatureScratch, &mut dyn FnMut(usize, &[f32; D])),
) -> Vec<u64> {
    let counts: Vec<std::sync::atomic::AtomicU64> = (0..km.k).map(|_| std::sync::atomic::AtomicU64::new(0)).collect();
    let combos = combo_cards();
    parallel_chunks(boards.len(), threads, 16, |a, b| {
        let mut s = FeatureScratch::new();
        let mut local = vec![0u64; km.k];
        for board in &boards[a..b] {
            feat(*board, &mut s, &mut |ci, x| {
                let (c, _) = km.nearest(x);
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

/// Bucket sizes of a fitted k-means over all hands of the given boards (no table written).
pub fn bucket_sizes<const B: usize, const D: usize>(
    boards: &[[u8; B]],
    km: &KMeans,
    threads: usize,
    feat: fn([u8; B], &mut FeatureScratch, &mut dyn FnMut(usize, &[f32; D])),
) -> Vec<u64> {
    let counts: Vec<std::sync::atomic::AtomicU64> = (0..km.k).map(|_| std::sync::atomic::AtomicU64::new(0)).collect();
    parallel_chunks(boards.len(), threads, 16, |a, b| {
        let mut s = FeatureScratch::new();
        let mut local = vec![0u64; km.k];
        for board in &boards[a..b] {
            feat(*board, &mut s, &mut |_, x| local[km.nearest(x).0] += 1);
        }
        for (c, v) in local.iter().enumerate() {
            counts[c].fetch_add(*v, Ordering::Relaxed);
        }
    });
    counts.into_iter().map(|c| c.into_inner()).collect()
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
    feat: fn([u8; B], &mut FeatureScratch, &mut dyn FnMut(usize, &[f32; D])),
) -> Vec<f32> {
    let parts: Vec<std::sync::Mutex<Vec<f32>>> = boards.iter().map(|_| std::sync::Mutex::new(Vec::new())).collect();
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
    use crate::eval::{card_bit, eval};
    use crate::ochs::cluster_of;

    fn cdf_of(eqs: &[f64]) -> Vec<f32> {
        let mut h = [0u16; HIST_BINS];
        for &e in eqs {
            h[hs_bin(e as f32)] += 1;
        }
        let mut out = [0f32; HIST_BINS];
        cdf(&h, eqs.len() as f32, &mut out);
        out.to_vec()
    }

    #[test]
    fn turn_features_match_direct_histogram() {
        let mut s = FeatureScratch::new();
        let board = [0u8, 13, 26, 39]; // 2c 5d 8h Js
        let mut got = std::collections::HashMap::new();
        turn_features(board, &mut s, |ci, x| {
            got.insert(ci, *x);
        });
        assert_eq!(got.len(), 1128);
        for hole in [(48u8, 49u8), (4, 9), (50, 51)] {
            let eqs: Vec<f64> = (0..52u8)
                .filter(|r| !board.contains(r) && *r != hole.0 && *r != hole.1)
                .map(|r| river_equity_single(&[board[0], board[1], board[2], board[3], r], [hole.0, hole.1]))
                .collect();
            let want = cdf_of(&eqs);
            let g = &got[&combo_index(hole.0, hole.1)];
            for b in 0..HIST_BINS {
                assert!((g[b] - want[b]).abs() < 1e-6, "bin {b}");
            }
        }
    }

    #[test]
    fn flop_features_match_direct_histogram() {
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
                eqs.push(river_equity_single(&[board[0], board[1], board[2], rest[i], rest[j]], [hole.0, hole.1]));
            }
        }
        assert_eq!(eqs.len(), 1081);
        let want = cdf_of(&eqs);
        let g = &got[&combo_index(hole.0, hole.1)];
        for b in 0..HIST_BINS {
            assert!((g[b] - want[b]).abs() < 1e-6, "bin {b}");
        }
        // CDFs are monotone and end at 1
        assert!(got.values().all(|f| f.windows(2).all(|w| w[0] <= w[1]) && (f[HIST_BINS - 1] - 1.0).abs() < 1e-6));
    }

    #[test]
    fn river_ochs_matches_brute_force() {
        let mut s = FeatureScratch::new();
        let board = [3u8, 17, 22, 40, 47]; // arbitrary river board
        let mut got = std::collections::HashMap::new();
        river_features(board, &mut s, |ci, x| {
            got.insert(ci, *x);
        });
        assert_eq!(got.len(), 1081);
        let bm = board.iter().fold(0u64, |m, &c| m | card_bit(c));
        for hole in [(48u8, 49u8), (0, 5), (30, 34)] {
            let hv = eval(bm | card_bit(hole.0) | card_bit(hole.1));
            let mut win = [0f64; 8];
            let mut cnt = [0f64; 8];
            for (a, b) in combo_cards() {
                let m = card_bit(a) | card_bit(b);
                if m & (bm | card_bit(hole.0) | card_bit(hole.1)) != 0 {
                    continue;
                }
                let c = cluster_of(a, b) as usize;
                let ov = eval(bm | m);
                cnt[c] += 1.0;
                win[c] += if hv > ov { 1.0 } else if hv == ov { 0.5 } else { 0.0 };
            }
            let g = &got[&combo_index(hole.0, hole.1)];
            for c in 0..8 {
                assert!((g[c] as f64 - win[c] / cnt[c]).abs() < 1e-6, "cluster {c}");
            }
        }
    }

    #[test]
    fn legacy_river_cutoffs_split_point_masses_and_give_k_buckets() {
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
    fn centroid_bytes_roundtrip() {
        let c = vec![0.5f32, -1.25, 3.0];
        assert_eq!(centroids_from_bytes(&centroids_to_bytes(&c)), c);
    }
}
