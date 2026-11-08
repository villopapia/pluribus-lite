//! D-INFO-6 one-flop benchmark (`docs/m4_info6_plan.md` §4 step 1): the per-flop 500-bucket
//! potential-aware abstraction at full settings vs each proposed reduction.
//!
//! `cargo run --release --bin info6_bench -- [--flop 44,22,1] [--k 500] [--threads 12]
//!   [--river-restarts 5] [--turn-restarts 3] [--seed 1]`
//!
//! * River: OCHS (8 equities) for all 1,176 × 1,081 = 1,271,256 (river board, hand) of the flop;
//!   k-means L2, k clusters. Full fit vs fit on a 200,000-point sample (then all assigned).
//!   Quality: within-cluster mean squared L2 over all points; agreement: adjusted Rand index.
//! * Turn: for all 49 × 1,128 = 55,272 (turn, hand), the histogram over the river buckets of the
//!   46 rivers (full river clustering as the common base); k-means with [27]'s approximate EMD,
//!   ground distance = L2 between river centroids. Full (all points, exact means, no pruning) vs
//!   each reduction: fit on 10,000 points; Hamerly pruning; means truncated to 64 entries; all
//!   three. Quality: within-cluster EMD over all points (exact means); agreement: ARI vs full.
//!   Restarts: the best of `--turn-restarts` seeds (lowest within-cluster EMD), as [27] (25);
//!   reduction '5 restarts' = the first 5 of those seeds. Same for the river (`--river-restarts`).

use nlhe::abstraction::{river_features, FeatureScratch};
use nlhe::equity::{combo_cards, combo_index, COMBOS};
use nlhe::kmeans::{self, Metric};
use nlhe::ochs::N_CLUSTERS;
use nlhe::potential::{ari, kmeans as emd_kmeans, within_emd, Hist, Options};
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let get = |k: &str, d: &str| args.iter().position(|a| a == k).map_or(d.to_string(), |i| args[i + 1].clone());
    let flop: Vec<u8> = get("--flop", "44,22,1").split(',').map(|x| x.parse().unwrap()).collect();
    let k: usize = get("--k", "500").parse().unwrap();
    let threads: usize = get("--threads", "12").parse().unwrap();
    let rr: usize = get("--river-restarts", "25").parse().unwrap();
    let tr: usize = get("--turn-restarts", "25").parse().unwrap();
    let seed: u64 = get("--seed", "1").parse().unwrap();
    let only_turn_full = args.iter().any(|a| a == "--probe");
    let t_all = Instant::now();
    println!("flop {flop:?}, k {k}, threads {threads}, seed {seed}");

    // ---------------------------------------------------------------- river
    let rest: Vec<u8> = (0..52u8).filter(|c| !flop.contains(c)).collect(); // 49 cards
    let mut boards: Vec<(u8, u8)> = Vec::new();
    for i in 0..rest.len() {
        for j in i + 1..rest.len() {
            boards.push((rest[i], rest[j]));
        }
    }
    let bid = |a: u8, b: u8| -> usize {
        let (x, y) = if a < b { (a, b) } else { (b, a) };
        boards.iter().position(|&p| p == (x, y)).unwrap()
    };
    let t0 = Instant::now();
    let mut feats: Vec<f32> = Vec::with_capacity(boards.len() * 1081 * N_CLUSTERS);
    let mut pid = vec![u32::MAX; boards.len() * COMBOS];
    let mut s = FeatureScratch::new();
    for (b, &(x, y)) in boards.iter().enumerate() {
        let board = [flop[0], flop[1], flop[2], x, y];
        river_features(board, &mut s, |ci, f| {
            pid[b * COMBOS + ci] = (feats.len() / N_CLUSTERS) as u32;
            feats.extend_from_slice(f);
        });
    }
    let np = feats.len() / N_CLUSTERS;
    println!("river: {} boards, {np} points, features {:.1}s", boards.len(), t0.elapsed().as_secs_f64());

    let t0 = Instant::now();
    let (km, fs, rs) = kmeans::fit_restarts(&feats, N_CLUSTERS, k, Metric::L2, 300, seed, threads, rr);
    let t_full = t0.elapsed().as_secs_f64();
    let lab_full: Vec<u16> = (0..np).map(|i| km.nearest(&feats[i * N_CLUSTERS..(i + 1) * N_CLUSTERS]).0 as u16).collect();
    let sse_full = kmeans::within_sse(&feats, &km, threads) / np as f64;
    println!(
        "river full: {rr} restarts, {:.1}s, {} iterations (best), within mean sq. L2 {:.6}; restart spread: best {:.1} vs first {:.1}",
        t_full, fs.iterations, sse_full, rs.sse[rs.best], rs.sse[0]
    );
    if !only_turn_full {
        let t0 = Instant::now();
        let (km5, _, _) = kmeans::fit_restarts(&feats, N_CLUSTERS, k, Metric::L2, 300, seed, threads, 5);
        let lab5: Vec<u16> = (0..np).map(|i| km5.nearest(&feats[i * N_CLUSTERS..(i + 1) * N_CLUSTERS]).0 as u16).collect();
        let sse5 = kmeans::within_sse(&feats, &km5, threads) / np as f64;
        println!(
            "river reduction '5 restarts': {:.1}s, within {:.6} ({:+.2}% vs full), ARI vs full {:.4}",
            t0.elapsed().as_secs_f64(),
            sse5,
            100.0 * (sse5 / sse_full - 1.0),
            ari(&lab5, &lab_full)
        );
        let t0 = Instant::now();
        let mut rng = cfr::rng::Rng::new(seed ^ 0xA5A5);
        let mut idx: Vec<usize> = (0..np).collect();
        let m = 200_000.min(np);
        for i in 0..m {
            let j = i + (rng.next_u64() % (np - i) as u64) as usize;
            idx.swap(i, j);
        }
        let sample: Vec<f32> = idx[..m].iter().flat_map(|&i| feats[i * N_CLUSTERS..(i + 1) * N_CLUSTERS].iter().copied()).collect();
        let (kms, _, _) = kmeans::fit_restarts(&sample, N_CLUSTERS, k, Metric::L2, 300, seed, threads, rr);
        let lab: Vec<u16> = (0..np).map(|i| kms.nearest(&feats[i * N_CLUSTERS..(i + 1) * N_CLUSTERS]).0 as u16).collect();
        let sse = kmeans::within_sse(&feats, &kms, threads) / np as f64;
        println!(
            "river reduction 'fit on 200k sample': {:.1}s ({:.1}x faster), within {:.6} ({:+.2}% vs full), ARI vs full {:.4}",
            t0.elapsed().as_secs_f64(),
            t_full / t0.elapsed().as_secs_f64(),
            sse,
            100.0 * (sse / sse_full - 1.0),
            ari(&lab, &lab_full)
        );
    }

    // ---------------------------------------------------------------- turn
    let c = k;
    let mut ground = vec![0f32; c * c];
    for i in 0..c {
        for j in 0..c {
            let (a, b) = (km.centroid(i), km.centroid(j));
            ground[i * c + j] = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f32>().sqrt();
        }
    }
    let combos = combo_cards();
    let mut pts: Vec<Hist> = Vec::new();
    for &t in &rest {
        for &(a, b) in &combos {
            if flop.contains(&a) || flop.contains(&b) || a == t || b == t {
                continue;
            }
            let ci = combo_index(a, b);
            let ids: Vec<u16> = rest
                .iter()
                .filter(|&&r| r != t && r != a && r != b)
                .map(|&r| lab_full[pid[bid(t, r) * COMBOS + ci] as usize])
                .collect();
            pts.push(Hist::from_outcomes(&ids));
        }
    }
    let avg_n = pts.iter().map(|p| p.idx.len()).sum::<usize>() as f64 / pts.len() as f64;
    println!("turn: {} points, {:.1} river buckets per histogram on average", pts.len(), avg_n);

    let run = |name: &str, o: Options, restarts: usize, base: Option<&(Vec<u16>, f64, f64)>| -> (Vec<u16>, f64, f64) {
        let t0 = Instant::now();
        let mut best: Option<(Vec<u16>, f64)> = None;
        let mut evals = 0u64;
        let mut its = Vec::new();
        let mut scores = Vec::new();
        for r in 0..restarts {
            let (lab, _, st) = emd_kmeans(&pts, &ground, c, &Options { seed: kmeans::restart_seed(o.seed, r), ..o.clone() });
            evals += st.emd_evals;
            its.push(st.iterations);
            let w = within_emd(&pts, &lab, k, &ground, c, o.threads);
            scores.push(w);
            if best.as_ref().is_none_or(|b| w < b.1) {
                best = Some((lab, w));
            }
        }
        let secs = t0.elapsed().as_secs_f64();
        let (lab, w) = best.unwrap();
        match base {
            None => println!("turn {name}: {secs:.1}s, {restarts} restarts, iterations {its:?}, EMD evals {evals}, within EMD {w:.5} (restarts {scores:.5?})"),
            Some(b) => println!(
                "turn {name}: {secs:.1}s ({:.1}x faster), iterations {its:?}, EMD evals {evals}, within EMD {w:.5} ({:+.2}% vs full), ARI vs full {:.4}",
                b.2 / secs,
                100.0 * (w / b.1 - 1.0),
                ari(&lab, &b.0)
            ),
        }
        (lab, w, secs)
    };
    let base_o = Options { k, max_iters: 100, seed, threads, ..Default::default() };
    let full = run("full", base_o.clone(), tr, None);
    if !only_turn_full {
        run("reduction '5 restarts'", base_o.clone(), 5, Some(&full));
        run("reduction 'fit on 10k sample'", Options { sample: Some(10_000), ..base_o.clone() }, tr, Some(&full));
        run("reduction 'Hamerly pruning'", Options { prune: true, ..base_o.clone() }, tr, Some(&full));
        run("reduction 'means truncated to 64'", Options { truncate: Some(64), ..base_o.clone() }, tr, Some(&full));
        run("all three reductions", Options { sample: Some(10_000), prune: true, truncate: Some(64), ..base_o.clone() }, tr, Some(&full));
    }
    println!("total {:.1}s", t_all.elapsed().as_secs_f64());
}
