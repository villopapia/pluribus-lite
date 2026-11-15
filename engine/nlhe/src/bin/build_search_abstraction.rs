//! D-INFO-6 builder: the per-flop 500-bucket potential-aware search abstraction for all 1,755
//! canonical flops (`docs/m4_info6_plan.md` §4 step 4; format in `nlhe::search_abs`).
//!
//! `build_search_abstraction --out ../data/search_abstraction [--threads 12] [--seed 1] [--k 500]
//!   [--river-restarts 5] [--river-sample 200000] [--turn-restarts 5] [--max-secs N]
//!   [--stop-file PATH] [--limit N]`
//!
//! Per flop, as `info6_bench` with the accepted reductions: river OCHS k-means (L2) fitted on a
//! sample, all points assigned; turn histograms over the river buckets, EMD k-means with Hamerly
//! bounds (exact), best of the restarts by within-cluster EMD. Resumable: flops whose file
//! exists and checks are skipped. No new flop starts after `--max-secs` or once the stop file
//! exists; a flop in progress finishes (~1–2 min) unless the process is killed, which is safe
//! (files are written to .tmp and renamed). One line per flop is appended to `build.tsv`.

use nlhe::abstraction::{river_features, FeatureScratch};
use nlhe::equity::{combo_cards, combo_index, COMBOS};
use nlhe::kmeans::{self, Metric};
use nlhe::ochs::N_CLUSTERS;
use nlhe::potential::{kmeans as emd_kmeans, within_emd, Hist, Options};
use nlhe::search_abs::{self, pair_index, Table, BOARDS, REST};
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let get = |k: &str| args.iter().position(|a| a == k).map(|i| args[i + 1].clone());
    let num = |k: &str, d: u64| get(k).map_or(d, |v| v.parse().expect(k));
    let out = PathBuf::from(get("--out").expect("--out DIR"));
    let threads = num("--threads", 12) as usize;
    let seed = num("--seed", 1);
    let k = num("--k", 500) as usize;
    let rr = num("--river-restarts", 5) as usize;
    let rs = num("--river-sample", 200_000) as usize;
    let tr = num("--turn-restarts", 5) as usize;
    let max_secs = num("--max-secs", u64::MAX);
    let limit = num("--limit", u64::MAX) as usize;
    let stop = get("--stop-file").map(PathBuf::from);
    std::fs::create_dir_all(&out).expect("create --out");
    let t_all = Instant::now();
    let flops = search_abs::canonical_flops();
    let done = flops.iter().filter(|f| search_abs::read(&out.join(search_abs::file_name(**f))).is_ok()).count();
    println!(
        "{} canonical flops, {done} done; k {k}, seed {seed}, river {rr} restarts on {rs} points, turn {tr} restarts (Hamerly), threads {threads}",
        flops.len()
    );
    let log_path = out.join("build.tsv");
    if !log_path.exists() {
        std::fs::write(&log_path, "flop\tsecs\triver_within_sq_l2\tturn_within_emd\tseed\tk\triver_restarts\triver_sample\tturn_restarts\n").unwrap();
    }
    let mut built = 0;
    for (ci, &flop) in flops.iter().enumerate() {
        if built >= limit || t_all.elapsed().as_secs() >= max_secs || stop.as_ref().is_some_and(|s| s.exists()) {
            break;
        }
        if search_abs::read(&out.join(search_abs::file_name(flop))).is_ok() {
            continue;
        }
        let t0 = Instant::now();
        let fseed = seed ^ (ci as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let (table, sse, wemd) = build(flop, k, fseed, rr, rs, tr, threads);
        search_abs::write(&out, &Table { seed: fseed, ..table }).expect("write table");
        let secs = t0.elapsed().as_secs_f64();
        let mut log = std::fs::OpenOptions::new().append(true).open(&log_path).unwrap();
        writeln!(log, "{:?}\t{secs:.1}\t{sse:.6}\t{wemd:.5}\t{fseed}\t{k}\t{rr}\t{rs}\t{tr}", flop).unwrap();
        built += 1;
        println!("flop {ci} {flop:?}: {secs:.1}s ({} of {} done)", done + built, flops.len());
    }
    println!("stopped after {built} flops, {:.0}s", t_all.elapsed().as_secs_f64());
}

fn build(flop: [u8; 3], k: usize, seed: u64, rr: usize, rs: usize, tr: usize, threads: usize) -> (Table, f64, f64) {
    let rest: Vec<u8> = (0..52u8).filter(|c| !flop.contains(c)).collect();
    // river: OCHS features of every (board, hand)
    let mut feats: Vec<f32> = Vec::with_capacity(BOARDS * 1081 * N_CLUSTERS);
    let mut pid = vec![u32::MAX; BOARDS * COMBOS];
    let mut s = FeatureScratch::new();
    for i in 0..REST {
        for j in i + 1..REST {
            let b = pair_index(i, j);
            river_features([flop[0], flop[1], flop[2], rest[i], rest[j]], &mut s, |ci, f| {
                pid[b * COMBOS + ci] = (feats.len() / N_CLUSTERS) as u32;
                feats.extend_from_slice(f);
            });
        }
    }
    let np = feats.len() / N_CLUSTERS;
    let mut rng = cfr::rng::Rng::new(seed ^ 0xA5A5);
    let mut idx: Vec<usize> = (0..np).collect();
    let m = rs.min(np);
    for i in 0..m {
        let j = i + (rng.next_u64() % (np - i) as u64) as usize;
        idx.swap(i, j);
    }
    let sample: Vec<f32> = idx[..m].iter().flat_map(|&i| feats[i * N_CLUSTERS..(i + 1) * N_CLUSTERS].iter().copied()).collect();
    let (km, _, _) = kmeans::fit_restarts(&sample, N_CLUSTERS, k, Metric::L2, 300, seed, threads, rr);
    let lab: Vec<u16> = (0..np).map(|i| km.nearest(&feats[i * N_CLUSTERS..(i + 1) * N_CLUSTERS]).0 as u16).collect();
    let sse = kmeans::within_sse(&feats, &km, threads) / np as f64;
    let river: Vec<u16> = pid.iter().map(|&p| if p == u32::MAX { u16::MAX } else { lab[p as usize] }).collect();

    // turn: histogram over the river buckets of the 46 rivers; EMD k-means
    let mut ground = vec![0f32; k * k];
    for i in 0..k {
        for j in 0..k {
            let (a, b) = (km.centroid(i), km.centroid(j));
            ground[i * k + j] = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f32>().sqrt();
        }
    }
    let mut pts: Vec<Hist> = Vec::new();
    let mut slot: Vec<(usize, usize)> = Vec::new(); // (turn position, combo)
    for (ti, &t) in rest.iter().enumerate() {
        for &(a, b) in &combo_cards() {
            if flop.contains(&a) || flop.contains(&b) || a == t || b == t {
                continue;
            }
            let ci = combo_index(a, b);
            let ids: Vec<u16> = (0..REST)
                .filter(|&ri| ri != ti && rest[ri] != a && rest[ri] != b)
                .map(|ri| river[pair_index(ti.min(ri), ti.max(ri)) * COMBOS + ci])
                .collect();
            pts.push(Hist::from_outcomes(&ids));
            slot.push((ti, ci));
        }
    }
    let o = Options { k, max_iters: 100, seed, threads, prune: true, ..Default::default() };
    let mut best: Option<(Vec<u16>, f64)> = None;
    for r in 0..tr {
        let (l, _, _) = emd_kmeans(&pts, &ground, k, &Options { seed: kmeans::restart_seed(seed, r), ..o.clone() });
        let w = within_emd(&pts, &l, k, &ground, k, threads);
        if best.as_ref().is_none_or(|b| w < b.1) {
            best = Some((l, w));
        }
    }
    let (tl, wemd) = best.unwrap();
    let mut turn = vec![u16::MAX; REST * COMBOS];
    for (&(ti, ci), &b) in slot.iter().zip(&tl) {
        turn[ti * COMBOS + ci] = b;
    }
    let table = Table { flop, k: k as u16, seed, river_restarts: rr as u32, river_sample: m as u32, turn_restarts: tr as u32, river, turn };
    (table, sse, wemd)
}
