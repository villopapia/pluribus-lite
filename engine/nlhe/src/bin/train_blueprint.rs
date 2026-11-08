//! M3 blueprint trainer: 6-max MCCFR on the abstract game (Algorithm 1 of the Pluribus
//! supplement; see docs/alg1_diff.md).
//!
//! `train_blueprint --config nlhe/configs/m2_11g.cfg --abstraction-dir ../data/abstraction/m2_11g
//!   --buckets 256 --run-dir ../data/runs/NAME --threads 12 --minutes 1440
//!   [--warmup-minutes 10] [--schedule-scale 1.0] [--strategy-interval 10000]
//!   [--checkpoint-minutes 120] [--keep-checkpoints 3] [--log-seconds 60]
//!   [--stop-file PATH] [--seed 1] [--bench-seconds S]`
//!
//! * Resumes automatically from the newest `ckpt_*.bin` in the run dir (schedule, seed and
//!   snapshot count come from the checkpoint).
//! * Fresh runs train a warm-up of `--warmup-minutes` with no pruning/discounting, measure
//!   the iteration rate over its second half, and then fix the paper's schedule in
//!   iterations: discount every 10 min until 400 min (the first one exactly at the end of
//!   the warm-up), prune after 200 min, post-flop snapshots every 200 min after 800 min.
//!   `--schedule-scale` multiplies all paper minutes (1.0 = paper; < 1 for short tests).
//! * Stops on the time budget or when the stop file exists, always writing a final checkpoint.
//!   Keeps the newest `--keep-checkpoints` checkpoint files.
//! * Checkpoints stream to disk in 1 MiB chunks (no table-sized copy in memory).
//! * Metrics: `<run-dir>/metrics.tsv`; schedule: `<run-dir>/schedule.txt`.

use nlhe::betting::Rules;
use nlhe::blueprint::*;
use nlhe::buckets::{CardAbstraction, TableAbstraction};
use nlhe::config::AbstractionConfig;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::time::Instant;

struct Args {
    config: PathBuf,
    abs_dir: PathBuf,
    buckets: u32,
    run_dir: PathBuf,
    threads: usize,
    minutes: f64,
    warmup_minutes: f64,
    schedule_scale: f64,
    strategy_interval: u64,
    checkpoint_minutes: f64,
    keep_checkpoints: usize,
    log_seconds: f64,
    stop_file: Option<PathBuf>,
    seed: u64,
    bench_seconds: Option<f64>,
}

fn parse() -> Args {
    let mut a = Args {
        config: PathBuf::new(),
        abs_dir: PathBuf::new(),
        buckets: 256,
        run_dir: PathBuf::new(),
        threads: std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1),
        minutes: 60.0,
        warmup_minutes: 10.0,
        schedule_scale: 1.0,
        strategy_interval: 10_000,
        checkpoint_minutes: 120.0,
        keep_checkpoints: 3,
        log_seconds: 60.0,
        stop_file: None,
        seed: 1,
        bench_seconds: None,
    };
    let v: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < v.len() {
        let x = v.get(i + 1).cloned().unwrap_or_default();
        match v[i].as_str() {
            "--config" => a.config = x.into(),
            "--abstraction-dir" => a.abs_dir = x.into(),
            "--buckets" => a.buckets = x.parse().unwrap(),
            "--run-dir" => a.run_dir = x.into(),
            "--threads" => a.threads = x.parse().unwrap(),
            "--minutes" => a.minutes = x.parse().unwrap(),
            "--warmup-minutes" => a.warmup_minutes = x.parse().unwrap(),
            "--schedule-scale" => a.schedule_scale = x.parse().unwrap(),
            "--strategy-interval" => a.strategy_interval = x.parse().unwrap(),
            "--checkpoint-minutes" => a.checkpoint_minutes = x.parse().unwrap(),
            "--keep-checkpoints" => a.keep_checkpoints = x.parse().unwrap(),
            "--log-seconds" => a.log_seconds = x.parse().unwrap(),
            "--stop-file" => a.stop_file = Some(x.into()),
            "--seed" => a.seed = x.parse().unwrap(),
            "--bench-seconds" => a.bench_seconds = Some(x.parse().unwrap()),
            o => panic!("unknown argument {o}"),
        }
        i += 2;
    }
    a
}

fn checkpoints(dir: &Path) -> Vec<(u64, PathBuf)> {
    let mut v: Vec<(u64, PathBuf)> = std::fs::read_dir(dir)
        .map(|rd| {
            rd.filter_map(|e| e.ok())
                .filter_map(|e| {
                    let name = e.file_name().into_string().ok()?;
                    let t: u64 = name.strip_prefix("ckpt_")?.strip_suffix(".bin")?.parse().ok()?;
                    Some((t, e.path()))
                })
                .collect()
        })
        .unwrap_or_default();
    v.sort();
    v
}

struct Logger {
    file: std::fs::File,
    pre_s: Vec<(u32, u32)>,
    post_s: Vec<(u32, u32)>,
    prev_pre: Vec<f32>,
    prev_post: Vec<f32>,
    last: Instant,
}

fn main() {
    let a = parse();
    let cfg = AbstractionConfig::parse(&std::fs::read_to_string(&a.config).expect("config")).expect("parse config");
    std::fs::create_dir_all(&a.run_dir).expect("run dir");
    let t_setup = Instant::now();
    let abs = TableAbstraction::load(&a.abs_dir, a.buckets).expect("load abstraction");
    let tree = Tree::build(&Rules::default(), &cfg, abs.buckets());
    let st = Storage::new(&tree);
    println!(
        "setup {:.1}s: {} nodes ({} decision), {} regrets ({:.2} GiB), {} avg counts ({:.2} GiB), buckets {:?}, threads {}",
        t_setup.elapsed().as_secs_f64(),
        tree.nodes.len(),
        tree.decision_nodes.iter().sum::<u64>(),
        tree.n_regrets,
        tree.n_regrets as f64 * 4.0 / 1073741824.0,
        tree.n_avg,
        tree.n_avg as f64 * 4.0 / 1073741824.0,
        tree.buckets,
        a.threads
    );
    let stop = AtomicBool::new(false);
    let snapshot_path = a.run_dir.join("postflop_snapshots.bin");

    // resume or start fresh (warm-up first)
    let resumed = checkpoints(&a.run_dir).pop();
    let (mut t, mut elapsed, mut sched, seed, mut snapshots, mut in_warmup) = if let Some((_, p)) = resumed {
        let t0 = Instant::now();
        let m = load_checkpoint(&p, &tree, &st).expect("load checkpoint");
        assert_eq!(m.config, cfg.name, "checkpoint is for another config");
        let warm = m.sched.lcfr_until == 0 && m.sched.prune_after == u64::MAX;
        println!(
            "resumed {} at iteration {} ({:.1} min trained, {} snapshots{}) in {:.1}s",
            p.display(),
            m.iteration,
            m.elapsed_secs / 60.0,
            m.snapshots,
            if warm { ", warm-up incomplete: restarting the warm-up timer" } else { "" },
            t0.elapsed().as_secs_f64()
        );
        (m.iteration, m.elapsed_secs, m.sched, m.seed, m.snapshots, warm)
    } else {
        (0, 0.0, Schedule::warmup(a.strategy_interval), a.seed, 0, true)
    };

    if let Some(bs) = a.bench_seconds {
        let tr = Trainer { tree: &tree, st: &st, abs: &abs, sched: &sched, seed };
        let t0 = Instant::now();
        let mut agg = ThreadStats::default();
        let mut tt = t;
        while t0.elapsed().as_secs_f64() < bs {
            let (last, s) = tr.run(tt, tt + 5_000, a.threads, &stop);
            tt = last;
            agg.add(&s);
        }
        let secs = t0.elapsed().as_secs_f64();
        println!(
            "bench threads={} iterations={} it/s={:.1} river_share={:.3} nodes/it={:.0} river_reach={:.3} ns/node={:.1}",
            a.threads,
            agg.iterations,
            agg.iterations as f64 / secs,
            agg.river_ns as f64 / agg.traverse_ns.max(1) as f64,
            agg.nodes as f64 / agg.iterations.max(1) as f64,
            agg.river_reached as f64 / agg.traversals.max(1) as f64,
            agg.traverse_ns as f64 / agg.nodes.max(1) as f64
        );
        return;
    }

    let metrics_path = a.run_dir.join("metrics.tsv");
    let new_file = !metrics_path.exists();
    let mut lg = Logger {
        file: std::fs::OpenOptions::new().create(true).append(true).open(&metrics_path).unwrap(),
        pre_s: sample_infosets(&tree, 50_000, 77, true),
        post_s: sample_infosets(&tree, 50_000, 78, false),
        prev_pre: Vec::new(),
        prev_post: Vec::new(),
        last: Instant::now(),
    };
    lg.prev_pre = policy_snapshot(&tree, &st, &lg.pre_s);
    lg.prev_post = policy_snapshot(&tree, &st, &lg.post_s);
    if new_file {
        writeln!(lg.file, "wall_s\ttrain_min\titeration\tit_per_s\tnodes_per_it\tns_per_node\triver_reach\triver_share\tpruned_per_it\tpos_regret_per_infoset_pre\tpos_regret_per_infoset_flop\tpos_regret_per_infoset_turn\tpos_regret_per_infoset_river\ttouched_frac_pre\ttouched_frac_flop\ttouched_frac_turn\ttouched_frac_river\tstrat_change_pre_avg\tstrat_change_post_cur\tsaturated\tfloored\tsnapshots\tphase").unwrap();
    }

    let wall0 = Instant::now();
    let mut last_ckpt = Instant::now();
    let mut interval = ThreadStats::default();
    let mut total = ThreadStats::default();
    let chunk = 20_000u64;
    let warm_start = Instant::now();
    let warm_t0 = t;
    let mut warm_mid: Option<(f64, u64)> = None;

    let save = |t: u64, elapsed: f64, sched: &Schedule, snapshots: u64| {
        let meta = CkptMeta { config: cfg.name.clone(), buckets: tree.buckets, iteration: t, seed, elapsed_secs: elapsed, sched: sched.clone(), snapshots };
        let p = a.run_dir.join(format!("ckpt_{t:012}.bin"));
        let t0 = Instant::now();
        let n = save_checkpoint(&p, &meta, &tree, &st).expect("save checkpoint");
        println!("checkpoint {} ({:.2} GiB) written in {:.1}s", p.display(), n as f64 / 1073741824.0, t0.elapsed().as_secs_f64());
        let all = checkpoints(&a.run_dir);
        if all.len() > a.keep_checkpoints {
            for (_, old) in &all[..all.len() - a.keep_checkpoints] {
                if std::fs::remove_file(old).is_ok() {
                    println!("removed old checkpoint {}", old.display());
                }
            }
        }
    };

    loop {
        // next epoch end: chunk, next discount, next snapshot
        let mut to = t + chunk;
        if !in_warmup {
            if let Some(d) = sched.next_discount(t) {
                to = to.min(d);
            }
            if let Some(s) = sched.next_snapshot(t) {
                to = to.min(s);
            }
        }
        let tr = Trainer { tree: &tree, st: &st, abs: &abs, sched: &sched, seed };
        let e0 = Instant::now();
        let (last, s) = tr.run(t, to, a.threads, &stop);
        elapsed += e0.elapsed().as_secs_f64();
        t = last;
        interval.add(&s);
        total.add(&s);

        let mut phase = if in_warmup { "warmup" } else { "train" };
        if in_warmup {
            let w = warm_start.elapsed().as_secs_f64();
            if warm_mid.is_none() && w >= a.warmup_minutes * 30.0 {
                warm_mid = Some((w, t));
            }
            if w >= a.warmup_minutes * 60.0 {
                let (wm, tm) = warm_mid.unwrap_or((0.0, warm_t0));
                let rate = (t - tm) as f64 / (w - wm).max(1e-9); // it/s over the second half
                sched = Schedule::paper_minutes(t, rate * 60.0, a.strategy_interval, a.schedule_scale);
                in_warmup = false;
                phase = "warmup_end";
                let txt = format!(
                    "warm-up: {:.1} min, {} iterations, rate over second half {:.0} it/s\nschedule (paper minutes x {}): {:?}\n",
                    w / 60.0, t, rate, a.schedule_scale, sched
                );
                print!("{txt}");
                std::fs::write(a.run_dir.join("schedule.txt"), txt).unwrap();
            }
        }
        if !in_warmup {
            if let Some(k) = sched.discount_k(t) {
                let d0 = Instant::now();
                Trainer { tree: &tree, st: &st, abs: &abs, sched: &sched, seed }.discount(k, a.threads);
                elapsed += d0.elapsed().as_secs_f64();
                println!("discount k={k} at iteration {t} in {:.1}s", d0.elapsed().as_secs_f64());
            }
            if sched.snapshot_due(t) {
                let s0 = Instant::now();
                snapshots = snapshot_add(&snapshot_path, &tree, &st).expect("snapshot");
                println!("post-flop snapshot {snapshots} at iteration {t} in {:.1}s", s0.elapsed().as_secs_f64());
            }
        }

        let stop_req = a.stop_file.as_ref().is_some_and(|p| p.exists());
        let out_of_time = wall0.elapsed().as_secs_f64() >= a.minutes * 60.0;
        let finishing = stop_req || out_of_time;
        if finishing || last_ckpt.elapsed().as_secs_f64() >= a.checkpoint_minutes * 60.0 {
            save(t, elapsed, &sched, snapshots);
            last_ckpt = Instant::now();
        }
        if finishing || phase == "warmup_end" || lg.last.elapsed().as_secs_f64() >= a.log_seconds {
            let secs = lg.last.elapsed().as_secs_f64();
            let rs = regret_summary(&tree, &st);
            let cur_pre = policy_snapshot(&tree, &st, &lg.pre_s);
            let cur_post = policy_snapshot(&tree, &st, &lg.post_s);
            let ch_pre = snapshot_change(&tree, &lg.pre_s, &lg.prev_pre, &cur_pre);
            let ch_post = snapshot_change(&tree, &lg.post_s, &lg.prev_post, &cur_post);
            lg.prev_pre = cur_pre;
            lg.prev_post = cur_post;
            let it = interval.iterations.max(1) as f64;
            let mut row = format!(
                "{:.0}\t{:.2}\t{}\t{:.1}\t{:.0}\t{:.1}\t{:.4}\t{:.4}\t{:.2}",
                wall0.elapsed().as_secs_f64(),
                elapsed / 60.0,
                t,
                interval.iterations as f64 / secs,
                interval.nodes as f64 / it,
                interval.traverse_ns as f64 / interval.nodes.max(1) as f64,
                interval.river_reached as f64 / interval.traversals.max(1) as f64,
                interval.river_ns as f64 / interval.traverse_ns.max(1) as f64,
                interval.pruned_actions as f64 / it
            );
            for sidx in 0..4 {
                row += &format!("\t{:.1}", rs.pos_sum[sidx] / rs.infosets[sidx].max(1) as f64);
            }
            for sidx in 0..4 {
                row += &format!("\t{:.4}", rs.touched[sidx] as f64 / rs.infosets[sidx].max(1) as f64);
            }
            row += &format!(
                "\t{:.5}\t{:.5}\t{}\t{}\t{}\t{}",
                ch_pre, ch_post, total.regret.saturated, total.regret.floored, snapshots, phase
            );
            println!("{row}");
            writeln!(lg.file, "{row}").unwrap();
            lg.file.flush().unwrap();
            interval = ThreadStats::default();
            lg.last = Instant::now();
        }
        if finishing {
            println!("stopping at iteration {t} ({})", if stop_req { "stop file" } else { "time budget" });
            break;
        }
    }
}
