//! M3 blueprint trainer (skeleton): 6-max MCCFR on the abstract game.
//!
//! `train_blueprint --config nlhe/configs/m2_11g.cfg --abstraction-dir ../data/abstraction/m2_11g
//!   --buckets 32 --run-dir ../data/runs/smoke --threads 12 --minutes 45 [--plan-minutes 45]
//!   [--strategy-interval 1000] [--checkpoint-minutes 10] [--log-seconds 60]
//!   [--stop-file PATH] [--seed 1] [--bench-seconds S]`
//!
//! * Resumes automatically from the newest `ckpt_*.bin` in the run dir (schedule and seed come
//!   from the checkpoint).
//! * Fresh runs calibrate throughput for ~15 s on a throwaway table, then set the schedule
//!   (prune / LCFR / discount) as Pluribus' wall-clock fractions of `--plan-minutes`.
//! * Stops on time budget or when the stop file exists, always writing a final checkpoint.
//! * `--bench-seconds`: throughput only (resumes from a checkpoint if present), no writes.
//! * Metrics: `<run-dir>/metrics.tsv`.

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
    plan_minutes: Option<f64>,
    strategy_interval: u64,
    checkpoint_minutes: f64,
    log_seconds: f64,
    stop_file: Option<PathBuf>,
    seed: u64,
    bench_seconds: Option<f64>,
}

fn parse() -> Args {
    let mut a = Args {
        config: PathBuf::new(),
        abs_dir: PathBuf::new(),
        buckets: 32,
        run_dir: PathBuf::new(),
        threads: std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1),
        minutes: 30.0,
        plan_minutes: None,
        strategy_interval: 1000,
        checkpoint_minutes: 10.0,
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
            "--plan-minutes" => a.plan_minutes = Some(x.parse().unwrap()),
            "--strategy-interval" => a.strategy_interval = x.parse().unwrap(),
            "--checkpoint-minutes" => a.checkpoint_minutes = x.parse().unwrap(),
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

fn latest_checkpoint(dir: &Path) -> Option<(u64, PathBuf)> {
    let rd = std::fs::read_dir(dir).ok()?;
    rd.filter_map(|e| e.ok())
        .filter_map(|e| {
            let name = e.file_name().into_string().ok()?;
            let t: u64 = name.strip_prefix("ckpt_")?.strip_suffix(".bin")?.parse().ok()?;
            Some((t, e.path()))
        })
        .max_by_key(|x| x.0)
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
        "setup {:.1}s: {} nodes ({} decision), {} regrets ({:.2} GiB), {} avg counts ({:.2} GiB), buckets {:?}",
        t_setup.elapsed().as_secs_f64(),
        tree.nodes.len(),
        tree.decision_nodes.iter().sum::<u64>(),
        tree.n_regrets,
        tree.n_regrets as f64 * 4.0 / 1073741824.0,
        tree.n_avg,
        tree.n_avg as f64 * 4.0 / 1073741824.0,
        tree.buckets
    );
    let stop = AtomicBool::new(false);

    // resume or calibrate
    let (mut t, mut elapsed, sched, seed) = if let Some((_, p)) = latest_checkpoint(&a.run_dir) {
        let t0 = Instant::now();
        let m = load_checkpoint(&p, &tree, &st).expect("load checkpoint");
        assert_eq!(m.config, cfg.name, "checkpoint is for another config");
        println!("resumed {} at iteration {} ({:.1} min trained) in {:.1}s", p.display(), m.iteration, m.elapsed_secs / 60.0, t0.elapsed().as_secs_f64());
        (m.iteration, m.elapsed_secs, m.sched, m.seed)
    } else {
        let plan = a.plan_minutes.unwrap_or(a.minutes);
        let scratch = Storage::new(&tree);
        let tmp_sched = Schedule::pluribus_scaled(u64::MAX / 4, a.strategy_interval);
        let tr = Trainer { tree: &tree, st: &scratch, abs: &abs, sched: &tmp_sched, seed: a.seed };
        let t0 = Instant::now();
        let mut n = 0u64;
        while t0.elapsed().as_secs_f64() < 15.0 {
            tr.run(n, n + 2_000, a.threads, &stop);
            n += 2_000;
        }
        let rate = n as f64 / t0.elapsed().as_secs_f64();
        drop(scratch);
        let planned = (rate * plan * 60.0) as u64;
        let s = Schedule::pluribus_scaled(planned, a.strategy_interval);
        println!("calibration: {rate:.0} it/s on {} threads -> planned {planned} iterations over {plan} min; schedule {s:?}", a.threads);
        (0, 0.0, s, a.seed)
    };
    let tr = Trainer { tree: &tree, st: &st, abs: &abs, sched: &sched, seed };

    if let Some(bs) = a.bench_seconds {
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
            "bench threads={} iterations={} it/s={:.1} river_share={:.3} nodes/it={:.0}",
            a.threads,
            agg.iterations,
            agg.iterations as f64 / secs,
            agg.river_ns as f64 / agg.traverse_ns.max(1) as f64,
            agg.nodes as f64 / agg.iterations.max(1) as f64
        );
        return;
    }

    let metrics_path = a.run_dir.join("metrics.tsv");
    let new_file = !metrics_path.exists();
    let mut mf = std::fs::OpenOptions::new().create(true).append(true).open(&metrics_path).unwrap();
    if new_file {
        writeln!(mf, "wall_s\ttrain_min\titeration\tit_per_s\triver_share\tnodes_per_it\tpruned_per_it\tpos_regret_per_infoset_pre\tpos_regret_per_infoset_flop\tpos_regret_per_infoset_turn\tpos_regret_per_infoset_river\ttouched_frac_pre\ttouched_frac_flop\ttouched_frac_turn\ttouched_frac_river\tstrat_change_pre_avg\tstrat_change_post_cur\tsaturated\tfloored\tcheckpoint_bytes").unwrap();
    }
    let pre_s = sample_infosets(&tree, 50_000, 77, true);
    let post_s = sample_infosets(&tree, 50_000, 78, false);
    let mut prev_pre = policy_snapshot(&tree, &st, &pre_s);
    let mut prev_post = policy_snapshot(&tree, &st, &post_s);

    let wall0 = Instant::now();
    let mut last_log = Instant::now();
    let mut last_ckpt = Instant::now();
    let mut interval = ThreadStats::default();
    let mut total = ThreadStats::default();
    let mut ckpt_bytes = 0u64;
    let chunk = 20_000u64;
    let save = |t: u64, elapsed: f64| -> u64 {
        let meta = CkptMeta { config: cfg.name.clone(), buckets: tree.buckets, iteration: t, seed, elapsed_secs: elapsed, sched: sched.clone() };
        let p = a.run_dir.join(format!("ckpt_{t:012}.bin"));
        let t0 = Instant::now();
        let n = save_checkpoint(&p, &meta, &tree, &st).expect("save checkpoint");
        println!("checkpoint {} ({:.2} GiB) in {:.1}s", p.display(), n as f64 / 1073741824.0, t0.elapsed().as_secs_f64());
        n
    };
    loop {
        let mut to = t + chunk;
        if t < sched.lcfr_until {
            let nd = (t / sched.discount_every + 1) * sched.discount_every;
            to = to.min(nd);
        }
        let e0 = Instant::now();
        let (last, s) = tr.run(t, to, a.threads, &stop);
        elapsed += e0.elapsed().as_secs_f64();
        t = last;
        interval.add(&s);
        total.add(&s);
        if t % sched.discount_every == 0 && t < sched.lcfr_until {
            let d0 = Instant::now();
            tr.discount(t / sched.discount_every, a.threads);
            elapsed += d0.elapsed().as_secs_f64();
        }
        let stop_req = a.stop_file.as_ref().is_some_and(|p| p.exists());
        let out_of_time = wall0.elapsed().as_secs_f64() >= a.minutes * 60.0;
        let finishing = stop_req || out_of_time;
        if finishing || last_ckpt.elapsed().as_secs_f64() >= a.checkpoint_minutes * 60.0 {
            ckpt_bytes = save(t, elapsed);
            last_ckpt = Instant::now();
        }
        if finishing || last_log.elapsed().as_secs_f64() >= a.log_seconds {
            let secs = last_log.elapsed().as_secs_f64();
            let rs = regret_summary(&tree, &st);
            let cur_pre = policy_snapshot(&tree, &st, &pre_s);
            let cur_post = policy_snapshot(&tree, &st, &post_s);
            let ch_pre = snapshot_change(&tree, &pre_s, &prev_pre, &cur_pre);
            let ch_post = snapshot_change(&tree, &post_s, &prev_post, &cur_post);
            prev_pre = cur_pre;
            prev_post = cur_post;
            let it = interval.iterations.max(1) as f64;
            let mut row = format!(
                "{:.0}\t{:.2}\t{}\t{:.1}\t{:.4}\t{:.0}\t{:.1}",
                wall0.elapsed().as_secs_f64(),
                elapsed / 60.0,
                t,
                interval.iterations as f64 / secs,
                interval.river_ns as f64 / interval.traverse_ns.max(1) as f64,
                interval.nodes as f64 / it,
                interval.pruned_actions as f64 / it
            );
            for sidx in 0..4 {
                row += &format!("\t{:.1}", rs.pos_sum[sidx] / rs.infosets[sidx].max(1) as f64);
            }
            for sidx in 0..4 {
                row += &format!("\t{:.4}", rs.touched[sidx] as f64 / rs.infosets[sidx].max(1) as f64);
            }
            row += &format!("\t{:.5}\t{:.5}\t{}\t{}\t{}", ch_pre, ch_post, total.regret.saturated, total.regret.floored, ckpt_bytes);
            println!("{row}");
            writeln!(mf, "{row}").unwrap();
            mf.flush().unwrap();
            interval = ThreadStats::default();
            last_log = Instant::now();
        }
        if finishing {
            println!("stopping at iteration {t} ({})", if stop_req { "stop file" } else { "time budget" });
            break;
        }
    }
}
