//! D-INFO-6 (plan `docs/m4_info6_plan.md`): potential-aware clustering with earth mover's distance
//! (Ganzfried & Sandholm 2014 [27], Algorithm 1 per round, Algorithm 2 for the EMD), the
//! per-round step of Pluribus' per-flop search abstraction ([28], Supp pp. 13–14).
//!
//! A point is a hand's **sparse histogram over next-round clusters** (where chance's next card
//! sends it). Distance: EMD whose ground distance is the distance between next-round cluster
//! means. [27]'s Algorithm 2 approximates it greedily: for each point entry, take mass from the
//! mean's entries in order of increasing ground distance. (The paper's pseudocode zeroes
//! `targets[j]` before subtracting it from `meanRemaining`; we subtract first, as intended.)
//!
//! [`kmeans`]: k-means++ seeding (D² with this EMD) and Lloyd iterations; options for the
//! plan's reductions: fit on a sample (`sample`), Hamerly bounds (`prune`), means truncated to
//! their largest entries (`truncate`). Deterministic for a given seed and thread count.

use cfr::rng::Rng;

/// Sparse histogram over next-round clusters; weights sum to 1.
#[derive(Clone, Debug, PartialEq)]
pub struct Hist {
    pub idx: Vec<u16>,
    pub w: Vec<f32>,
}

impl Hist {
    /// From next-round cluster ids, one per chance outcome (equal weights, duplicates merged).
    pub fn from_outcomes(ids: &[u16]) -> Hist {
        let mut v = ids.to_vec();
        v.sort_unstable();
        let (mut idx, mut w) = (Vec::new(), Vec::new());
        for &c in &v {
            if idx.last() == Some(&c) {
                *w.last_mut().unwrap() += 1.0;
            } else {
                idx.push(c);
                w.push(1.0);
            }
        }
        let n = ids.len() as f32;
        w.iter_mut().for_each(|x| *x /= n);
        Hist { idx, w }
    }
}

/// A cluster mean prepared for Algorithm 2: its nonzero entries and, for every next-round
/// cluster i, those entries ordered by ground distance from i.
pub struct Mean {
    pub nz: Vec<u16>,
    pub mass: Vec<f32>,
    ord: Vec<u16>,
}

impl Mean {
    /// From a dense distribution over `c` clusters; `truncate`: keep the largest entries only
    /// (renormalised).
    pub fn new(dense: &[f32], ground: &[f32], c: usize, truncate: Option<usize>) -> Mean {
        let mut nz: Vec<u16> = (0..c as u16).filter(|&i| dense[i as usize] > 0.0).collect();
        if let Some(t) = truncate {
            if nz.len() > t {
                nz.sort_by(|&a, &b| dense[b as usize].total_cmp(&dense[a as usize]).then(a.cmp(&b)));
                nz.truncate(t);
                nz.sort_unstable();
            }
        }
        let tot: f32 = nz.iter().map(|&i| dense[i as usize]).sum();
        let mass: Vec<f32> = nz.iter().map(|&i| dense[i as usize] / tot).collect();
        let q = nz.len();
        let mut ord = vec![0u16; c * q];
        let mut tmp: Vec<u16> = (0..q as u16).collect();
        for i in 0..c {
            tmp.sort_by(|&a, &b| ground[i * c + nz[a as usize] as usize].total_cmp(&ground[i * c + nz[b as usize] as usize]).then(a.cmp(&b)));
            ord[i * q..(i + 1) * q].copy_from_slice(&tmp);
        }
        Mean { nz, mass, ord }
    }

    pub fn of_hist(h: &Hist, ground: &[f32], c: usize) -> Mean {
        let mut d = vec![0f32; c];
        for (&i, &w) in h.idx.iter().zip(&h.w) {
            d[i as usize] = w;
        }
        Mean::new(&d, ground, c, None)
    }
}

/// [27] Algorithm 2: approximate EMD between point `x` and mean `m`. `rem` is scratch.
pub fn emd(x: &Hist, m: &Mean, ground: &[f32], c: usize, rem: &mut Vec<f32>) -> f32 {
    let q = m.nz.len();
    rem.clear();
    rem.extend_from_slice(&m.mass);
    let n = x.idx.len();
    let mut small = [0f32; 64];
    let mut big = Vec::new();
    let tgt: &mut [f32] = if n <= 64 {
        small[..n].copy_from_slice(&x.w);
        &mut small[..n]
    } else {
        big.extend_from_slice(&x.w); // a mean taken as a point (Hamerly drift)
        &mut big[..]
    };
    let mut left = n;
    let mut cost = 0f32;
    for r in 0..q {
        if left == 0 {
            break;
        }
        for j in 0..n {
            if tgt[j] <= 0.0 {
                continue;
            }
            let pc = x.idx[j] as usize;
            let pos = m.ord[pc * q + r] as usize;
            let amt = rem[pos];
            if amt <= 0.0 {
                continue;
            }
            let d = ground[pc * c + m.nz[pos] as usize];
            if amt < tgt[j] {
                cost += amt * d;
                tgt[j] -= amt;
                rem[pos] = 0.0;
            } else {
                cost += tgt[j] * d;
                rem[pos] -= tgt[j];
                tgt[j] = 0.0;
                left -= 1;
            }
        }
    }
    cost
}

#[derive(Clone, Debug, Default)]
pub struct Options {
    pub k: usize,
    pub max_iters: usize,
    pub seed: u64,
    pub threads: usize,
    /// fit on this many points (sampled without replacement), then assign all once
    pub sample: Option<usize>,
    /// Hamerly bounds to skip assignments (exact only if the approximate EMD is a metric)
    pub prune: bool,
    /// keep only the largest entries of each mean
    pub truncate: Option<usize>,
}

#[derive(Clone, Debug, Default)]
pub struct Stats {
    pub iterations: usize,
    pub emd_evals: u64,
    pub last_changed: usize,
}

fn par_map<T: Send, F: Fn(usize) -> T + Sync>(n: usize, threads: usize, f: F) -> Vec<T> {
    let threads = threads.max(1).min(n.max(1));
    let chunk = n.div_ceil(threads);
    let mut parts: Vec<Vec<T>> = Vec::new();
    std::thread::scope(|s| {
        let hs: Vec<_> = (0..threads)
            .map(|t| {
                let f = &f;
                s.spawn(move || (t * chunk..((t + 1) * chunk).min(n)).map(f).collect::<Vec<T>>())
            })
            .collect();
        parts = hs.into_iter().map(|h| h.join().unwrap()).collect();
    });
    parts.into_iter().flatten().collect()
}

/// Dense means of the assigned points (exact averages, untruncated).
pub fn dense_means(points: &[Hist], assign: &[u16], k: usize, c: usize) -> Vec<Vec<f32>> {
    let mut sum = vec![vec![0f64; c]; k];
    let mut cnt = vec![0usize; k];
    for (p, &a) in points.iter().zip(assign) {
        cnt[a as usize] += 1;
        for (&i, &w) in p.idx.iter().zip(&p.w) {
            sum[a as usize][i as usize] += w as f64;
        }
    }
    sum.into_iter().zip(cnt).map(|(s, n)| s.into_iter().map(|x| if n > 0 { (x / n as f64) as f32 } else { 0.0 }).collect()).collect()
}

/// Mean EMD of every point to the exact mean of its cluster (the quality measure).
pub fn within_emd(points: &[Hist], assign: &[u16], k: usize, ground: &[f32], c: usize, threads: usize) -> f64 {
    let dm = dense_means(points, assign, k, c);
    let means: Vec<Option<Mean>> = par_map(k, threads, |i| dm[i].iter().any(|&x| x > 0.0).then(|| Mean::new(&dm[i], ground, c, None)));
    let d = par_map(points.len(), threads, |i| {
        let mut rem = Vec::new();
        emd(&points[i], means[assign[i] as usize].as_ref().unwrap(), ground, c, &mut rem) as f64
    });
    d.iter().sum::<f64>() / points.len() as f64
}

fn nearest(p: &Hist, means: &[Mean], ground: &[f32], c: usize, rem: &mut Vec<f32>) -> (u16, f32, f32) {
    let (mut b, mut bd, mut sd) = (0u16, f32::MAX, f32::MAX);
    for (j, m) in means.iter().enumerate() {
        let d = emd(p, m, ground, c, rem);
        if d < bd {
            sd = bd;
            bd = d;
            b = j as u16;
        } else if d < sd {
            sd = d;
        }
    }
    (b, bd, sd)
}

/// k-means over histograms with [27]'s approximate EMD. Returns the assignment of **all**
/// points, the means (prepared), and statistics.
pub fn kmeans(points: &[Hist], ground: &[f32], c: usize, o: &Options) -> (Vec<u16>, Vec<Mean>, Stats) {
    let mut st = Stats::default();
    let mut rng = Rng::new(o.seed);
    // fit set
    let fit: Vec<usize> = match o.sample {
        Some(s) if s < points.len() => {
            let mut idx: Vec<usize> = (0..points.len()).collect();
            for i in 0..s {
                let j = i + (rng.next_u64() % (points.len() - i) as u64) as usize;
                idx.swap(i, j);
            }
            idx.truncate(s);
            idx.sort_unstable();
            idx
        }
        _ => (0..points.len()).collect(),
    };
    let fp: Vec<&Hist> = fit.iter().map(|&i| &points[i]).collect();
    let n = fp.len();
    // k-means++ seeding with D^2 under this EMD
    let mut centers = vec![(rng.next_u64() % n as u64) as usize];
    let mut dmin: Vec<f32> = vec![f32::MAX; n];
    while centers.len() < o.k {
        let m = Mean::of_hist(fp[*centers.last().unwrap()], ground, c);
        let d = par_map(n, o.threads, |i| {
            let mut rem = Vec::new();
            emd(fp[i], &m, ground, c, &mut rem)
        });
        st.emd_evals += n as u64;
        let mut tot = 0f64;
        for i in 0..n {
            dmin[i] = dmin[i].min(d[i]);
            tot += (dmin[i] as f64).powi(2);
        }
        let mut r = rng.next_f64() * tot;
        let mut pick = n - 1;
        for i in 0..n {
            r -= (dmin[i] as f64).powi(2);
            if r < 0.0 {
                pick = i;
                break;
            }
        }
        centers.push(pick);
    }
    let mut means: Vec<Mean> = centers.iter().map(|&i| Mean::of_hist(fp[i], ground, c)).collect();
    // Lloyd
    let mut assign = vec![u16::MAX; n];
    let (mut upper, mut lower) = (vec![f32::MAX; n], vec![0f32; n]);
    for it in 0..o.max_iters {
        let res = par_map(n, o.threads, |i| {
            let mut rem = Vec::new();
            if o.prune && assign[i] != u16::MAX && upper[i] <= lower[i] {
                return (assign[i], upper[i], lower[i], 0u64);
            }
            let (b, bd, sd) = nearest(fp[i], &means, ground, c, &mut rem);
            (b, bd, sd, o.k as u64)
        });
        let mut changed = 0;
        for (i, (b, bd, sd, ev)) in res.into_iter().enumerate() {
            st.emd_evals += ev;
            if b != assign[i] {
                changed += 1;
            }
            assign[i] = b;
            if ev > 0 {
                upper[i] = bd;
                lower[i] = sd;
            }
        }
        st.iterations = it + 1;
        st.last_changed = changed;
        // update means; empty clusters keep their previous mean
        let fit_points: Vec<Hist> = fp.iter().map(|&h| h.clone()).collect();
        let dm = dense_means(&fit_points, &assign, o.k, c);
        let new: Vec<Mean> = par_map(o.k, o.threads, |j| {
            if dm[j].iter().any(|&x| x > 0.0) {
                Mean::new(&dm[j], ground, c, o.truncate)
            } else {
                Mean { nz: means[j].nz.clone(), mass: means[j].mass.clone(), ord: means[j].ord.clone() }
            }
        });
        if o.prune {
            // drift of each mean (EMD between old and new mean, the old one taken as a point)
            let drift: Vec<f32> = par_map(o.k, o.threads, |j| {
                let mut rem = Vec::new();
                emd(&Hist { idx: means[j].nz.clone(), w: means[j].mass.clone() }, &new[j], ground, c, &mut rem)
            });
            st.emd_evals += o.k as u64;
            let maxd = drift.iter().cloned().fold(0f32, f32::max);
            for i in 0..n {
                upper[i] += drift[assign[i] as usize];
                lower[i] -= maxd;
            }
        }
        means = new;
        if changed * 1000 <= n {
            break;
        }
    }
    // assign every point (the fit set keeps its assignment)
    let all = if fit.len() == points.len() {
        assign
    } else {
        par_map(points.len(), o.threads, |i| {
            let mut rem = Vec::new();
            nearest(&points[i], &means, ground, c, &mut rem).0
        })
    };
    if fit.len() != points.len() {
        st.emd_evals += (points.len() * o.k) as u64;
    }
    (all, means, st)
}

/// Adjusted Rand index between two labelings (1 = identical partitions, ~0 = chance).
pub fn ari(a: &[u16], b: &[u16]) -> f64 {
    use std::collections::HashMap;
    let c2 = |x: u64| (x * x.saturating_sub(1)) as f64 / 2.0;
    let mut nij: HashMap<(u16, u16), u64> = HashMap::new();
    let (mut ai, mut bj): (HashMap<u16, u64>, HashMap<u16, u64>) = (HashMap::new(), HashMap::new());
    for (&x, &y) in a.iter().zip(b) {
        *nij.entry((x, y)).or_default() += 1;
        *ai.entry(x).or_default() += 1;
        *bj.entry(y).or_default() += 1;
    }
    let s: f64 = nij.values().map(|&v| c2(v)).sum();
    let sa: f64 = ai.values().map(|&v| c2(v)).sum();
    let sb: f64 = bj.values().map(|&v| c2(v)).sum();
    let exp = sa * sb / c2(a.len() as u64);
    (s - exp) / (0.5 * (sa + sb) - exp)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 1-D ground distance |i - j|: EMD = L1 between CDFs, the exact answer.
    fn line(c: usize) -> Vec<f32> {
        (0..c * c).map(|k| ((k / c) as f32 - (k % c) as f32).abs()).collect()
    }

    fn cdf_l1(a: &Hist, b: &Hist, c: usize) -> f32 {
        let (mut da, mut db) = (vec![0f32; c], vec![0f32; c]);
        a.idx.iter().zip(&a.w).for_each(|(&i, &w)| da[i as usize] += w);
        b.idx.iter().zip(&b.w).for_each(|(&i, &w)| db[i as usize] += w);
        let (mut ca, mut cb, mut s) = (0f32, 0f32, 0f32);
        for i in 0..c {
            ca += da[i];
            cb += db[i];
            s += (ca - cb).abs();
        }
        s
    }

    #[test]
    fn emd_exact_on_simple_cases() {
        let c = 10;
        let g = line(c);
        let mut rem = Vec::new();
        // all mass at 2 vs all at 7: 5
        let a = Hist::from_outcomes(&[2, 2]);
        assert_eq!(emd(&a, &Mean::of_hist(&Hist::from_outcomes(&[7]), &g, c), &g, c, &mut rem), 5.0);
        // identical histograms: 0
        let b = Hist::from_outcomes(&[1, 3, 3, 8]);
        assert_eq!(emd(&b, &Mean::of_hist(&b, &g, c), &g, c, &mut rem), 0.0);
        // duplicates merge into weights
        assert_eq!(b.idx, vec![1, 3, 8]);
        assert_eq!(b.w, vec![0.25, 0.5, 0.25]);
    }

    #[test]
    fn emd_approximation_vs_exact_1d() {
        // greedy nearest-first transport is not always optimal; on a line it is exact for
        // single-entry points and close otherwise: measure it
        let c = 12;
        let g = line(c);
        let mut rng = Rng::new(5);
        let mut rem = Vec::new();
        let (mut worst, mut mean_err) = (0f32, 0f32);
        for _ in 0..500 {
            let ids = |r: &mut Rng, n| (0..n).map(|_| (r.next_u64() % c as u64) as u16).collect::<Vec<_>>();
            let a = Hist::from_outcomes(&ids(&mut rng, 6));
            let b = Hist::from_outcomes(&ids(&mut rng, 9));
            let ex = cdf_l1(&a, &b, c);
            let ap = emd(&a, &Mean::of_hist(&b, &g, c), &g, c, &mut rem);
            assert!(ap >= ex - 1e-4, "approximation below the optimum: {ap} < {ex}");
            worst = worst.max(ap - ex);
            mean_err += (ap - ex) / ex.max(1e-6);
        }
        eprintln!("1-D: worst excess {worst:.3}, mean relative excess {:.3}", mean_err / 500.0);
        assert!(mean_err / 500.0 < 0.5);
    }

    #[test]
    fn kmeans_recovers_separated_groups_and_ari_scores_them() {
        let c = 20;
        let g = line(c);
        let mut pts = Vec::new();
        let mut truth = Vec::new();
        let mut rng = Rng::new(1);
        for k in 0..4u16 {
            for _ in 0..50 {
                let ids: Vec<u16> = (0..5).map(|_| k * 5 + (rng.next_u64() % 2) as u16).collect();
                pts.push(Hist::from_outcomes(&ids));
                truth.push(k);
            }
        }
        let o = Options { k: 4, max_iters: 50, seed: 3, threads: 4, ..Default::default() };
        let (a, _, _) = kmeans(&pts, &g, c, &o);
        assert!((ari(&a, &truth) - 1.0).abs() < 1e-9);
        let (ap, _, _) = kmeans(&pts, &g, c, &Options { prune: true, sample: Some(120), truncate: Some(3), ..o.clone() });
        assert!(ari(&ap, &truth) > 0.99);
        assert!(within_emd(&pts, &a, 4, &g, c, 2) < 0.6);
        // ari: permutation-invariant, chance-corrected
        let perm: Vec<u16> = truth.iter().map(|&x| 3 - x).collect();
        assert!((ari(&truth, &perm) - 1.0).abs() < 1e-12);
    }
}
