//! k-means (Lloyd) with mean centroids, k-means++ seeding, L1 or squared-L2 distance.
//!
//! Used for the Johanson et al. (AAMAS 2013) features cited by Pluribus [26]:
//! * flop/turn: hand-strength histograms compared with earth mover's distance. Points are
//!   stored as **CDFs**: in 1-D, EMD between histograms = L1 between their CDFs, and the
//!   mean of CDFs is the CDF of the mean histogram. So "k-means with EMD" = L1 assignment,
//!   mean update.
//! * river: OCHS vectors (8 equities), L2 distance.
//!
//! Determinism: partial sums are kept per fixed-size chunk and merged in chunk order, so the
//! result is identical for any thread count.
//!
//! Nearest-centroid search is exact but pruned: centroids are kept sorted by coordinate sum
//! S(c), and |S(x) - S(c)| <= L1(x, c), (S(x) - S(c))^2 / dim <= L2^2(x, c). The search starts
//! at the closest sum and stops on each side once the bound exceeds the best distance.

use crate::abstraction::parallel_chunks;
use cfr::rng::Rng;
use std::sync::Mutex;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Metric {
    /// L1 (EMD when points are CDFs)
    L1,
    /// squared L2
    L2,
}

#[derive(Clone, Debug, PartialEq)]
pub struct KMeans {
    pub k: usize,
    pub dim: usize,
    pub metric: Metric,
    /// centroids, k x dim, sorted by coordinate sum (bucket 0 = smallest sum)
    pub c: Vec<f32>,
    sums: Vec<f32>,
}

#[inline]
fn dist(m: Metric, a: &[f32], b: &[f32]) -> f32 {
    match m {
        Metric::L1 => a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum(),
        Metric::L2 => a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum(),
    }
}

impl KMeans {
    pub fn from_centroids(mut c: Vec<f32>, dim: usize, metric: Metric) -> Self {
        let k = c.len() / dim;
        let mut idx: Vec<usize> = (0..k).collect();
        let sum = |i: usize, c: &[f32]| c[i * dim..(i + 1) * dim].iter().sum::<f32>();
        idx.sort_by(|&a, &b| sum(a, &c).partial_cmp(&sum(b, &c)).unwrap().then(a.cmp(&b)));
        let sorted: Vec<f32> = idx.iter().flat_map(|&i| c[i * dim..(i + 1) * dim].to_vec()).collect();
        c = sorted;
        let sums = (0..k).map(|i| sum(i, &c)).collect();
        KMeans { k, dim, metric, c, sums }
    }

    #[inline]
    pub fn centroid(&self, i: usize) -> &[f32] {
        &self.c[i * self.dim..(i + 1) * self.dim]
    }

    #[inline]
    fn bound(&self, ds: f32) -> f32 {
        match self.metric {
            Metric::L1 => ds.abs(),
            Metric::L2 => ds * ds / self.dim as f32,
        }
    }

    /// Exact nearest centroid (lowest index on ties) and its distance.
    pub fn nearest(&self, x: &[f32]) -> (usize, f32) {
        let s: f32 = x.iter().sum();
        let start = self.sums.partition_point(|&v| v < s);
        let mut best = (usize::MAX, f32::INFINITY);
        let (mut lo, mut hi) = (start as isize - 1, start);
        loop {
            let bl = if lo >= 0 { self.bound(s - self.sums[lo as usize]) } else { f32::INFINITY };
            let bh = if hi < self.k { self.bound(self.sums[hi] - s) } else { f32::INFINITY };
            let (i, b) = if bl <= bh { (lo as usize, bl) } else { (hi, bh) };
            if b > best.1 || (b.is_infinite() && b == f32::INFINITY && lo < 0 && hi >= self.k) {
                break;
            }
            let d = dist(self.metric, x, self.centroid(i));
            if d < best.1 || (d == best.1 && i < best.0) {
                best = (i, d);
            }
            if i == hi && hi < self.k && (bl > bh || lo < 0) {
                hi += 1;
            } else {
                lo -= 1;
            }
            if lo < 0 && hi >= self.k {
                break;
            }
        }
        best
    }

    /// Brute-force nearest (test oracle).
    pub fn nearest_brute(&self, x: &[f32]) -> (usize, f32) {
        let mut best = (0, f32::INFINITY);
        for i in 0..self.k {
            let d = dist(self.metric, x, self.centroid(i));
            if d < best.1 {
                best = (i, d);
            }
        }
        best
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct FitStats {
    pub iterations: usize,
    pub changed_last: usize,
    pub mean_dist: f64,
    pub empty_reseeds: usize,
}

const CHUNK: usize = 4096;

/// Lloyd's k-means with k-means++ seeding (on a subsample of up to 50,000 points).
/// Empty clusters are reseeded at the farthest points. Stops when fewer than 0.1% of points
/// change cluster, or after `max_iters`.
pub fn fit(points: &[f32], dim: usize, k: usize, metric: Metric, max_iters: usize, seed: u64, threads: usize) -> (KMeans, FitStats) {
    let n = points.len() / dim;
    assert!(n >= k);
    let pt = |i: usize| &points[i * dim..(i + 1) * dim];
    let mut rng = Rng::new(seed);
    // k-means++ on a subsample
    let sub_n = n.min(50_000);
    let sub: Vec<usize> = if sub_n == n { (0..n).collect() } else { (0..sub_n).map(|_| (rng.next_u64() % n as u64) as usize).collect() };
    let mut c: Vec<f32> = pt(sub[(rng.next_u64() % sub_n as u64) as usize]).to_vec();
    let mut dmin: Vec<f64> = sub.iter().map(|&i| dist(metric, pt(i), &c[0..dim]) as f64).collect();
    for j in 1..k {
        let total: f64 = dmin.iter().sum();
        let pick = if total > 0.0 { rng.sample_index(&dmin) } else { (rng.next_u64() % sub_n as u64) as usize };
        c.extend_from_slice(pt(sub[pick]));
        for (q, &i) in sub.iter().enumerate() {
            let d = dist(metric, pt(i), &c[j * dim..(j + 1) * dim]) as f64;
            if d < dmin[q] {
                dmin[q] = d;
            }
        }
    }
    let mut km = KMeans::from_centroids(c, dim, metric);
    let mut assign = vec![u32::MAX; n];
    let mut stats = FitStats::default();
    let n_chunks = n.div_ceil(CHUNK);
    for it in 0..max_iters {
        // assignment + per-chunk partial sums (merged in chunk order: deterministic)
        let partial: Vec<Mutex<Option<(Vec<f64>, Vec<u64>, usize, f64, Vec<(f32, usize)>)>>> = (0..n_chunks).map(|_| Mutex::new(None)).collect();
        let assign_ref = &assign;
        let new_assign: Vec<Mutex<Vec<u32>>> = (0..n_chunks).map(|_| Mutex::new(Vec::new())).collect();
        {
            let km = &km;
            parallel_chunks(n_chunks, threads, 1, |a, b| {
                for ch in a..b {
                    let lo = ch * CHUNK;
                    let hi = (lo + CHUNK).min(n);
                    let mut sums = vec![0f64; k * dim];
                    let mut cnt = vec![0u64; k];
                    let mut changed = 0usize;
                    let mut dsum = 0f64;
                    let mut far: Vec<(f32, usize)> = Vec::new();
                    let mut asg = Vec::with_capacity(hi - lo);
                    for i in lo..hi {
                        let (j, d) = km.nearest(pt(i));
                        if assign_ref[i] != j as u32 {
                            changed += 1;
                        }
                        asg.push(j as u32);
                        cnt[j] += 1;
                        dsum += d as f64;
                        for (s, &v) in sums[j * dim..(j + 1) * dim].iter_mut().zip(pt(i)) {
                            *s += v as f64;
                        }
                        far.push((d, i));
                    }
                    far.sort_by(|x, y| y.0.partial_cmp(&x.0).unwrap().then(x.1.cmp(&y.1)));
                    far.truncate(k);
                    *partial[ch].lock().unwrap() = Some((sums, cnt, changed, dsum, far));
                    *new_assign[ch].lock().unwrap() = asg;
                }
            });
        }
        let mut sums = vec![0f64; k * dim];
        let mut cnt = vec![0u64; k];
        let mut changed = 0usize;
        let mut dsum = 0f64;
        let mut far: Vec<(f32, usize)> = Vec::new();
        for (ch, p) in partial.into_iter().enumerate() {
            let (s, c2, ch_changed, d, f) = p.into_inner().unwrap().unwrap();
            for (a, b) in sums.iter_mut().zip(&s) {
                *a += b;
            }
            for (a, b) in cnt.iter_mut().zip(&c2) {
                *a += b;
            }
            changed += ch_changed;
            dsum += d;
            far.extend(f);
            let asg = new_assign[ch].lock().unwrap();
            assign[ch * CHUNK..ch * CHUNK + asg.len()].copy_from_slice(&asg);
        }
        far.sort_by(|x, y| y.0.partial_cmp(&x.0).unwrap().then(x.1.cmp(&y.1)));
        stats.iterations = it + 1;
        stats.changed_last = changed;
        stats.mean_dist = dsum / n as f64;
        // mean update; empty clusters -> farthest points
        let mut newc = vec![0f32; k * dim];
        let mut far_iter = far.into_iter();
        for j in 0..k {
            if cnt[j] == 0 {
                let (_, i) = far_iter.next().expect("enough points");
                newc[j * dim..(j + 1) * dim].copy_from_slice(pt(i));
                stats.empty_reseeds += 1;
            } else {
                for d in 0..dim {
                    newc[j * dim + d] = (sums[j * dim + d] / cnt[j] as f64) as f32;
                }
            }
        }
        let had_empty = cnt.iter().any(|&x| x == 0);
        km = KMeans::from_centroids(newc, dim, metric);
        if it > 0 && !had_empty && changed * 1000 <= n {
            break;
        }
    }
    (km, stats)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn random_points(n: usize, dim: usize, seed: u64) -> Vec<f32> {
        let mut rng = Rng::new(seed);
        (0..n * dim).map(|_| rng.next_f64() as f32 * 100.0).collect()
    }

    #[test]
    fn pruned_nearest_equals_brute_force() {
        for metric in [Metric::L1, Metric::L2] {
            let c = random_points(200, 8, 1);
            let km = KMeans::from_centroids(c, 8, metric);
            let xs = random_points(5_000, 8, 2);
            for x in xs.chunks(8) {
                let (a, da) = km.nearest(x);
                let (b, db) = km.nearest_brute(x);
                assert_eq!(da, db, "{metric:?}");
                assert_eq!(a, b, "{metric:?}");
            }
        }
    }

    #[test]
    fn recovers_separated_clusters_and_is_thread_independent() {
        let mut rng = Rng::new(3);
        let centers = [10.0f32, 50.0, 90.0];
        let mut pts = Vec::new();
        for i in 0..6_000 {
            for _ in 0..4 {
                pts.push(centers[i % 3] + (rng.next_f64() as f32 - 0.5) * 4.0);
            }
        }
        for metric in [Metric::L1, Metric::L2] {
            let (a, _) = fit(&pts, 4, 3, metric, 50, 9, 1);
            let (b, st) = fit(&pts, 4, 3, metric, 50, 9, 4);
            assert_eq!(a, b, "thread count changed the result");
            for (j, want) in centers.iter().enumerate() {
                assert!((a.centroid(j)[0] - want).abs() < 1.0, "{:?}", a.c);
            }
            assert!(st.mean_dist < 10.0);
        }
    }

    #[test]
    fn mean_update_on_one_cluster() {
        let pts = vec![1.0f32, 3.0, 5.0, 7.0];
        let (km, _) = fit(&pts, 2, 1, Metric::L2, 5, 0, 1);
        assert_eq!(km.c, vec![3.0, 5.0]);
    }
}
