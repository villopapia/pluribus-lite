//! Build the M2 card abstraction (Johanson et al. 2013 features): flop/turn EMD k-means on
//! hand-strength histograms (bucket tables), river OCHS k-means (centroids only).
//!
//! `cargo run --release --bin build_abstraction -- --config nlhe/configs/m2_11g.cfg
//!  --out ../data/abstraction/m2_11g [--threads 12] [--seed 1] [--fit-boards 1000]
//!  [--max-iters 30] [--restarts 25] [--stages river,flop,turn]`
//!
//! k-means++ is run `--restarts` times per street (default 25, docs/deviations.md D-INFO-1,
//! following Ganzfried & Sandholm AAAI 2014); the fit with the lowest within-cluster sum of
//! squared distances is kept. The SSE of restart 0 (= the old single fit) and of the best
//! restart are logged.
//!
//! Each stage writes one checksummed file; a stage whose valid output already exists is
//! skipped (stage-level resume). Timings and bucket-size statistics go to stdout and are
//! appended to <out>/build_report.md.

use cfr::rng::Rng;
use nlhe::abstraction::*;
use nlhe::kmeans::{self, Metric};
use nlhe::ochs::N_CLUSTERS;
use nlhe::config::AbstractionConfig;
use nlhe::equity::RiverBatch;
use nlhe::index::{FLOP_ENTRIES, TURN_ENTRIES};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Instant;

struct Args {
    config: PathBuf,
    out: PathBuf,
    threads: usize,
    seed: u64,
    fit_boards: usize,
    max_iters: usize,
    restarts: usize,
    stages: Vec<String>,
    /// "ochs" (default: Johanson 2013 OCHS k-means) or "equity1d" (legacy, for comparison)
    river_method: String,
}

/// "SSE restart 0 (single fit) X, best of N Y (restart r), -Z%; min/median/max ...".
fn restart_line(rs: &kmeans::RestartStats) -> String {
    let mut v = rs.sse.clone();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let best = rs.sse[rs.best];
    format!(
        "  - k-means++ restarts {}: within-cluster SSE restart 0 (= single fit) {:.4}, best {:.4} (restart {}), {:+.2}% vs single; min/median/max {:.4} / {:.4} / {:.4}",
        rs.sse.len(),
        rs.sse[0],
        best,
        rs.best,
        (best / rs.sse[0] - 1.0) * 100.0,
        v[0],
        v[v.len() / 2],
        v[v.len() - 1]
    )
}

fn parse_args() -> Args {
    let mut a = Args {
        config: PathBuf::new(),
        out: PathBuf::new(),
        threads: std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1),
        seed: 1,
        fit_boards: 1000,
        max_iters: 30,
        restarts: 25,
        stages: vec!["river".into(), "flop".into(), "turn".into()],
        river_method: "ochs".into(),
    };
    let v: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < v.len() {
        let val = v.get(i + 1).cloned().unwrap_or_default();
        match v[i].as_str() {
            "--config" => a.config = val.into(),
            "--out" => a.out = val.into(),
            "--threads" => a.threads = val.parse().expect("--threads"),
            "--seed" => a.seed = val.parse().expect("--seed"),
            "--fit-boards" => a.fit_boards = val.parse().expect("--fit-boards"),
            "--max-iters" => a.max_iters = val.parse().expect("--max-iters"),
            "--restarts" => a.restarts = val.parse().expect("--restarts"),
            "--stages" => a.stages = val.split(',').map(|s| s.trim().to_string()).collect(),
            "--river-method" => {
                assert!(val == "ochs" || val == "equity1d", "--river-method ochs|equity1d");
                a.river_method = val
            }
            other => panic!("unknown argument {other}"),
        }
        i += 2;
    }
    assert!(
        !a.config.as_os_str().is_empty() && !a.out.as_os_str().is_empty(),
        "--config and --out are required"
    );
    a
}

fn dist_stats(counts: &[u64]) -> String {
    let mut c: Vec<u64> = counts.to_vec();
    c.sort_unstable();
    let total: u64 = c.iter().sum();
    let mean = total as f64 / c.len() as f64;
    let empty = c.iter().filter(|&&x| x == 0).count();
    format!(
        "buckets {}, entries {}, empty {}, min {} ({:.2}x mean), p10 {}, median {}, p90 {}, max {} ({:.2}x mean)",
        c.len(),
        total,
        empty,
        c[0],
        c[0] as f64 / mean,
        c[c.len() / 10],
        c[c.len() / 2],
        c[c.len() * 9 / 10],
        c[c.len() - 1],
        c[c.len() - 1] as f64 / mean
    )
}

fn valid_existing(path: &Path, kind: &str, cfg: &AbstractionConfig, k: usize) -> bool {
    match read_abs_file(path) {
        Ok((h, _)) => h.kind == kind && h.config == cfg.name && h.k as usize == k,
        Err(_) => false,
    }
}

/// n distinct boards, chosen with a seeded partial Fisher-Yates, returned in colex order.
fn sample_boards<const B: usize>(all: &[[u8; B]], n: usize, rng: &mut Rng) -> Vec<[u8; B]> {
    let mut idx: Vec<usize> = (0..all.len()).collect();
    let n = n.min(all.len());
    for i in 0..n {
        let j = i + (rng.next_u64() % (all.len() - i) as u64) as usize;
        idx.swap(i, j);
    }
    let mut pick: Vec<usize> = idx[..n].to_vec();
    pick.sort_unstable();
    pick.into_iter().map(|i| all[i]).collect()
}

struct Log(String);

impl Log {
    fn line(&mut self, s: String) {
        println!("{s}");
        self.0.push_str(&s);
        self.0.push('\n');
    }
}

/// River: OCHS k-means (default) or the legacy 1-D equity cut-offs (`--river-method equity1d`,
/// kept only to measure deviation D-INFO-2).
fn river_stage(a: &Args, cfg: &AbstractionConfig, log: &mut Log) {
    let k = cfg.buckets[3] as usize;
    if a.river_method == "equity1d" {
        return river_stage_equity1d(a, cfg, log, k);
    }
    let path = a.out.join("river.ochs");
    if valid_existing(&path, "river-ochs", cfg, k) {
        log.line("- river (OCHS): valid output exists, skipped (resume)".into());
        return;
    }
    let mut rng = Rng::new(a.seed ^ 0x51CE);
    let t0 = Instant::now();
    let all = all_boards::<5>();
    let fitb = sample_boards(&all, a.fit_boards, &mut rng);
    let pts = sample_points::<5, N_CLUSTERS>(&fitb, a.threads, river_feat_dyn);
    let t_feat = t0.elapsed().as_secs_f64();
    let t1 = Instant::now();
    let (km, st, rs) = kmeans::fit_restarts(&pts, N_CLUSTERS, k, Metric::L2, a.max_iters, a.seed, a.threads, a.restarts);
    let t_fit = t1.elapsed().as_secs_f64();
    drop(pts);
    // bucket sizes on a second, independent board sample (no river table is stored)
    let evalb = sample_boards(&all, a.fit_boards, &mut rng);
    let sizes = bucket_sizes::<5, N_CLUSTERS>(&evalb, &km, a.threads, river_feat_dyn);
    let h = AbsHeader {
        kind: "river-ochs".into(),
        config: cfg.name.clone(),
        k: k as u32,
        width: N_CLUSTERS as u8,
        entries: k as u64,
        seed: a.seed,
        checksum: 0,
    };
    write_abs_file(&path, &h, &centroids_to_bytes(&km.c)).expect("write river.ochs");
    log.line(format!(
        "- river (OCHS, Johanson 2013 opponent clusters): fit sample {} boards / {} points, features {t_feat:.1}s, k-means L2 {t_fit:.1}s ({} iterations, last changed {}, mean squared L2 {:.5}, empty reseeds {})",
        fitb.len(),
        fitb.len() * 1081,
        st.iterations,
        st.changed_last,
        st.mean_dist,
        st.empty_reseeds
    ));
    log.line(restart_line(&rs));
    log.line(format!("  - bucket sizes on {} independent river boards: {}", evalb.len(), dist_stats(&sizes)));
}

fn river_stage_equity1d(a: &Args, cfg: &AbstractionConfig, log: &mut Log, k: usize) {
    let path = a.out.join("river.cut");
    if valid_existing(&path, "river", cfg, k) {
        log.line("- river (legacy 1-D equity): valid output exists, skipped (resume)".into());
        return;
    }
    let t0 = Instant::now();
    let boards = all_boards::<5>();
    let hist = Mutex::new(vec![0u64; RIVER_BINS]);
    parallel_chunks(boards.len(), a.threads, 2048, |s, e| {
        let mut rb = RiverBatch::new();
        let mut h = vec![0u64; RIVER_BINS];
        for b in &boards[s..e] {
            rb.compute(b);
            for &x in &rb.equity {
                if !x.is_nan() {
                    h[q16(x) as usize] += 1;
                }
            }
        }
        let mut g = hist.lock().unwrap();
        for (i, v) in h.iter().enumerate() {
            g[i] += v;
        }
    });
    let hist = hist.into_inner().unwrap();
    let cuts = river_cutoffs(&hist, k, 500);
    let mut mass = vec![0u64; cuts.len() + 1];
    for (bi, &w) in hist.iter().enumerate() {
        mass[river_bucket(&cuts, bi as f32 / (RIVER_BINS - 1) as f32)] += w;
    }
    let h = AbsHeader {
        kind: "river".into(),
        config: cfg.name.clone(),
        k: k as u32,
        width: 4,
        entries: cuts.len() as u64,
        seed: a.seed,
        checksum: 0,
    };
    write_abs_file(&path, &h, &centroids_to_bytes(&cuts)).expect("write river");
    log.line(format!("- river (legacy 1-D equity k-means): {} buckets in {:.1}s", cuts.len() + 1, t0.elapsed().as_secs_f64()));
    log.line(format!("  - mass per bucket: {}", dist_stats(&mass)));
}

/// Flop/turn: k-means with EMD on hand-strength histograms (L1 on CDFs), then assignment of
/// every (hole, board) into a direct-indexed table.
#[allow(clippy::too_many_arguments)]
fn table_stage<const B: usize, const D: usize>(
    a: &Args,
    cfg: &AbstractionConfig,
    log: &mut Log,
    stage: &str,
    k: usize,
    entries: u64,
    salt: u64,
    feat: fn([u8; B], &mut FeatureScratch, &mut dyn FnMut(usize, &[f32; D])),
) {
    let path = a.out.join(format!("{stage}.bkt"));
    if valid_existing(&path, stage, cfg, k) {
        log.line(format!("- {stage}: valid output exists, skipped (resume)"));
        return;
    }
    let mut rng = Rng::new(a.seed ^ salt);
    let t0 = Instant::now();
    let all = all_boards::<B>();
    let fitb = sample_boards(&all, a.fit_boards, &mut rng);
    let pts = sample_points::<B, D>(&fitb, a.threads, feat);
    let t_feat = t0.elapsed().as_secs_f64();
    let t1 = Instant::now();
    let (km, st, rs) = kmeans::fit_restarts(&pts, D, k, Metric::L1, a.max_iters, a.seed, a.threads, a.restarts);
    let t_fit = t1.elapsed().as_secs_f64();
    log.line(format!(
        "- {stage}: hand-strength histograms ({D} bins), EMD k-means; fit sample {} boards / {} points, features {t_feat:.1}s, k-means {t_fit:.1}s ({} iterations, last changed {}, mean EMD {:.3} bins, empty reseeds {})",
        fitb.len(),
        pts.len() / D,
        st.iterations,
        st.changed_last,
        st.mean_dist,
        st.empty_reseeds
    ));
    log.line(restart_line(&rs));
    drop(pts);
    let t2 = Instant::now();
    let table = BucketTable::new(entries as usize, k);
    let counts = assign_table::<B, D>(&all, &km, &table, a.threads, feat);
    log.line(format!("  - assignment of all {} boards: {:.1}s", all.len(), t2.elapsed().as_secs_f64()));
    let t3 = Instant::now();
    let payload = table.to_bytes();
    drop(table);
    let h = AbsHeader {
        kind: stage.into(),
        config: cfg.name.clone(),
        k: k as u32,
        width: if k <= 256 { 1 } else { 2 },
        entries,
        seed: a.seed,
        checksum: 0,
    };
    write_abs_file(&path, &h, &payload).expect("write table");
    let ch = AbsHeader {
        kind: format!("{stage}-centroids"),
        config: cfg.name.clone(),
        k: k as u32,
        width: D as u8,
        entries: k as u64,
        seed: a.seed,
        checksum: 0,
    };
    write_abs_file(&a.out.join(format!("{stage}.centroids")), &ch, &centroids_to_bytes(&km.c)).expect("write centroids");
    log.line(format!(
        "  - write {:.1} MiB: {:.1}s; stage total {:.1}s",
        payload.len() as f64 / 1048576.0,
        t3.elapsed().as_secs_f64(),
        t0.elapsed().as_secs_f64()
    ));
    log.line(format!("  - bucket sizes over all (hole, board) pairs: {}", dist_stats(&counts)));
}

fn main() {
    let a = parse_args();
    let cfg = AbstractionConfig::parse(&std::fs::read_to_string(&a.config).expect("read config"))
        .expect("parse config");
    std::fs::create_dir_all(&a.out).expect("create out dir");
    let t_all = Instant::now();
    let mut log = Log(String::new());
    log.line(format!(
        "\n## build {} (threads {}, seed {}, fit boards {}, max iters {}, restarts {})\n",
        cfg.name, a.threads, a.seed, a.fit_boards, a.max_iters, a.restarts
    ));
    for stage in &a.stages {
        match stage.as_str() {
            "river" => river_stage(&a, &cfg, &mut log),
            "flop" => table_stage::<3, HIST_BINS>(&a, &cfg, &mut log, "flop", cfg.buckets[1] as usize, FLOP_ENTRIES, 0xF10F, flop_feat_dyn),
            "turn" => table_stage::<4, HIST_BINS>(&a, &cfg, &mut log, "turn", cfg.buckets[2] as usize, TURN_ENTRIES, 0x7A2A, turn_feat_dyn),
            other => panic!("unknown stage {other}"),
        }
    }
    log.line(format!("- total wall time {:.1}s", t_all.elapsed().as_secs_f64()));
    let rp = a.out.join("build_report.md");
    let prev = std::fs::read_to_string(&rp).unwrap_or_default();
    std::fs::write(&rp, prev + &log.0).expect("write report");
}
