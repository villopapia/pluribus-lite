//! M3 blueprint trainer: 6-max MCCFR on the abstract game (Algorithm 1 of the Pluribus
//! supplement; see docs/alg1_diff.md).
//!
//! `train_blueprint --config nlhe/configs/m2_11g.cfg --abstraction-dir ../data/abstraction/m2_11g
//!   --buckets 256 --run-dir ../data/runs/NAME --threads 12 --minutes 1440
//!   [--plan-minutes 23040] [--warmup-minutes 10] [--strategy-interval 10000]
//!   [--checkpoint-minutes 120] [--keep-checkpoints 3] [--log-seconds 60]
//!   [--stop-file PATH] [--seed 1] [--bench-seconds S]
//!   [--river-method on_the_fly|precomputed_table] [--river-table-mode file|ram]
//!   [--phi-start-paper-minute 0] [--prune-exempt-fold true] [--prune-exempt-runout false]`
//!
//! * Ablation flags (docs/deviations.md; defaults = rulings of 2026-11-08), applied to fresh
//!   and resumed runs alike: `--phi-start-paper-minute` (D-ALG-5; 0 = phi from t = 0 with
//!   discounting, 800 = the text's start), `--prune-exempt-fold` / `--prune-exempt-runout`
//!   (D-ALG-9).
//!
//! * River buckets: `river_method` from the config unless `--river-method` overrides it;
//!   `precomputed_table` reads `<abstraction-dir>/river.tbl` (`file`: positioned reads via the
//!   OS page cache, default; `ram`: loaded into memory). Identical buckets either way (tested).
//!
//! * Resumes automatically from the newest `ckpt_*.bin` in the run dir (schedule, seed and
//!   snapshot count come from the checkpoint).
//! * Fresh runs train a warm-up of `--warmup-minutes` with no pruning/discounting, measure
//!   the iteration rate over its second half, and plan `--plan-minutes` (default: --minutes)
//!   of training at that rate. A "paper minute" is 1/11,520 of the planned iterations
//!   (Pluribus' ~8-day run; docs/deviations.md D-SCHED-1): discount every 10 until 400,
//!   prune after 200, preflop average sampling and post-flop snapshots (every 200) from 800.
//! * Stops on the time budget or when the stop file exists, always writing a final checkpoint.
//!   Retention is delete-before-write: at most `--keep-checkpoints` (>= 2) files exist at any
//!   moment, *including* the one being written, so one complete checkpoint always exists.
//!   If the newest checkpoint cannot be read, the next older one is used.
//! * Checkpoints stream to disk in 1 MiB chunks (no table-sized copy in memory).
//! * Metrics: `<run-dir>/metrics.tsv`; schedule: `<run-dir>/schedule.txt`.

use nlhe::betting::Rules;
use nlhe::blueprint::*;
use nlhe::buckets::{CardAbstraction, TableAbstraction};
use nlhe::config::{AbstractionConfig, RiverMethod};
use nlhe::river_table::RiverTableMode;
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
    plan_minutes: Option<f64>,
    strategy_interval: u64,
    checkpoint_minutes: f64,
    keep_checkpoints: usize,
    log_seconds: f64,
    stop_file: Option<PathBuf>,
    seed: u64,
    bench_seconds: Option<f64>,
    river_method: Option<RiverMethod>,
    river_table_mode: RiverTableMode,
    phi_start_paper_minute: f64,
    prune_exempt_fold: bool,
    prune_exempt_runout: bool,
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
        plan_minutes: None,
        strategy_interval: 10_000,
        checkpoint_minutes: 120.0,
        keep_checkpoints: 3,
        log_seconds: 60.0,
        stop_file: None,
        seed: 1,
        bench_seconds: None,
        river_method: None,
        river_table_mode: RiverTableMode::File,
        phi_start_paper_minute: 0.0,
        prune_exempt_fold: true,
        prune_exempt_runout: false,
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
            "--plan-minutes" => a.plan_minutes = Some(x.parse().unwrap()),
            "--strategy-interval" => a.strategy_interval = x.parse().unwrap(),
            "--checkpoint-minutes" => a.checkpoint_minutes = x.parse().unwrap(),
            "--keep-checkpoints" => a.keep_checkpoints = x.parse().unwrap(),
            "--log-seconds" => a.log_seconds = x.parse().unwrap(),
            "--stop-file" => a.stop_file = Some(x.into()),
            "--seed" => a.seed = x.parse().unwrap(),
            "--bench-seconds" => a.bench_seconds = Some(x.parse().unwrap()),
            "--river-method" => {
                a.river_method = Some(match x.as_str() {
                    "on_the_fly" => RiverMethod::OnTheFly,
                    "precomputed_table" => RiverMethod::PrecomputedTable,
                    _ => panic!("--river-method on_the_fly|precomputed_table"),
                })
            }
            "--river-table-mode" => a.river_table_mode = x.parse().unwrap(),
            "--phi-start-paper-minute" => a.phi_start_paper_minute = x.parse().unwrap(),
            "--prune-exempt-fold" => a.prune_exempt_fold = x.parse().unwrap(),
            "--prune-exempt-runout" => a.prune_exempt_runout = x.parse().unwrap(),
            o => panic!("unknown argument {o}"),
        }
        i += 2;
    }
    a
}

/// Apply the ablation flags to a schedule. `planned` = planned iterations (None in the
/// warm-up, when paper minutes are not yet defined: phi then runs only if it starts at 0).
fn apply_flags(s: &mut Schedule, a: &Args, planned: Option<u64>) {
    s.prune_exempt_fold = a.prune_exempt_fold;
    s.prune_exempt_runout = a.prune_exempt_runout;
    s.avg_start = if a.phi_start_paper_minute <= 0.0 {
        0
    } else {
        match planned {
            Some(p) => (p as f64 * a.phi_start_paper_minute / 11_520.0).round() as u64,
            None => u64::MAX,
        }
    };
}

/// Planned iterations of a resumed schedule (discount_every = 10 paper minutes).
fn planned_of(s: &Schedule) -> Option<u64> {
    (s.lcfr_until > 0 && s.prune_after != u64::MAX).then(|| s.discount_every * 1_152)
}

fn unix_now() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

/// "YYYY-MM-DD HH:MM" (UTC) from Unix seconds (civil-from-days, H. Hinnant).
fn utc(secs: u64) -> String {
    let days = (secs / 86_400) as i64;
    let (h, m) = ((secs % 86_400) / 3600, (secs % 3600) / 60);
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let mo = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + (mo <= 2) as i64;
    format!("{y:04}-{mo:02}-{d:02} {h:02}:{m:02}")
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
    let mut abs = TableAbstraction::load(&a.abs_dir, a.buckets).expect("load abstraction");
    let river_method = a.river_method.unwrap_or(cfg.river_method);
    if river_method == RiverMethod::PrecomputedTable {
        let t0 = Instant::now();
        abs.load_river_table(&a.abs_dir, a.river_table_mode, false).expect("load river table");
        println!("river: precomputed table ({:?} mode) loaded in {:.1}s", a.river_table_mode, t0.elapsed().as_secs_f64());
    } else {
        println!("river: on the fly (OCHS + nearest centroid)");
    }
    let tree = Tree::build(&Rules { players: cfg.players, ..Rules::default() }, &cfg, abs.buckets());
    let mut st = Storage::new(&tree);
    println!(
        "setup {:.1}s: {} nodes ({} decision), {} regret slots if dense ({:.2} GiB), {} preflop slots allocated, {} avg counts ({:.2} GiB), buckets {:?}, threads {}",
        t_setup.elapsed().as_secs_f64(),
        tree.nodes.len(),
        tree.decision_nodes.iter().sum::<u64>(),
        tree.n_regrets,
        tree.n_regrets as f64 * 4.0 / 1073741824.0,
        st.allocated_slots(),
        tree.n_avg,
        tree.n_avg as f64 * 4.0 / 1073741824.0,
        tree.buckets,
        a.threads
    );
    let stop = AtomicBool::new(false);
    let snapshot_path = a.run_dir.join("postflop_snapshots.bin");

    // resume or start fresh (warm-up first)
    assert!(a.keep_checkpoints >= 2, "--keep-checkpoints must be >= 2 (one complete file while writing)");
    let mut resumed = None;
    for (_, p) in checkpoints(&a.run_dir).into_iter().rev() {
        let t0 = Instant::now();
        match load_checkpoint(&p, &tree, &st) {
            Ok(m) => {
                resumed = Some((p, m, t0));
                break;
            }
            Err(e) => {
                println!("checkpoint {} unreadable ({e}); trying an older one", p.display());
                st = Storage::new(&tree); // discard a partial load
            }
        }
    }
    let resumed_any = resumed.is_some();
    let (mut t, mut elapsed, mut sched, seed, mut snapshots, mut in_warmup) = if let Some((p, m, t0)) = resumed {
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
    } else if let Some(p) = cfg.planned_total_iterations {
        // explicit plan (D-SCHED-2): no warm-up measurement; the paper's schedule from t = 0
        let s = Schedule::pluribus_scaled(p, a.strategy_interval);
        let txt = format!(
            "planned_total_iterations = {p} (explicit, from the run config; no warm-up calibration)
planned: {} min -> planned rate {:.0} it/s; paper minute = {:.0} iterations
schedule: {s:?}
",
            a.plan_minutes.unwrap_or(a.minutes),
            p as f64 / (a.plan_minutes.unwrap_or(a.minutes) * 60.0),
            p as f64 / 11_520.0
        );
        print!("{txt}");
        if a.bench_seconds.is_none() {
            std::fs::write(a.run_dir.join("schedule.txt"), txt).unwrap();
        }
        (0, 0.0, s, a.seed, 0, false)
    } else {
        (0, 0.0, Schedule::warmup(a.strategy_interval), a.seed, 0, true)
    };
    let planned_total = cfg.planned_total_iterations;
    {
        let before = (sched.avg_start, sched.prune_exempt_fold, sched.prune_exempt_runout);
        let planned = planned_total.or_else(|| planned_of(&sched));
        apply_flags(&mut sched, &a, planned);
        println!(
            "flags: phi start iteration {} (paper minute {}), prune exempt fold {}, run-out {}{}",
            if sched.avg_start == u64::MAX { "after warm-up".to_string() } else { sched.avg_start.to_string() },
            a.phi_start_paper_minute,
            sched.prune_exempt_fold,
            sched.prune_exempt_runout,
            if before != (sched.avg_start, sched.prune_exempt_fold, sched.prune_exempt_runout) && resumed_any {
                format!(" (changed on resume from {before:?}; phi accumulates from iteration {t})")
            } else {
                String::new()
            }
        );
    }

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
    const HEADER: &str = "wall_s\ttrain_min\titeration\tit_per_s\tnodes_per_it\tns_per_node\triver_reach\triver_share\tpruned_per_it\tpos_regret_per_infoset_pre\tpos_regret_per_infoset_flop\tpos_regret_per_infoset_turn\tpos_regret_per_infoset_river\ttouched_frac_pre\ttouched_frac_flop\ttouched_frac_turn\ttouched_frac_river\tstrat_change_pre_avg\tstrat_change_post_cur\tsaturated\tfloored\tsnapshots\talloc_gib\talloc_frac_of_dense\tseq_reached\tseq_reached_frac\tphase	pruned_rate";
    // a metrics file with an older column set is kept aside, not appended to
    if let Ok(old) = std::fs::read_to_string(&metrics_path) {
        if old.lines().next() != Some(HEADER) {
            let mut k = 1;
            while a.run_dir.join(format!("metrics_v{k}.tsv")).exists() {
                k += 1;
            }
            std::fs::rename(&metrics_path, a.run_dir.join(format!("metrics_v{k}.tsv"))).expect("rotate metrics");
        }
    }
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
        writeln!(lg.file, "{HEADER}").unwrap();
    }

    let wall0 = Instant::now();
    const PROGRESS_SECS: f64 = 6.0 * 3600.0;
    let prog_t0 = t;
    let mut prog_last = (Instant::now(), t);
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
        // delete-before-write: leftovers of an interrupted write, then the oldest complete
        // files, so that with the new one at most keep_checkpoints exist
        if let Ok(rd) = std::fs::read_dir(&a.run_dir) {
            for e in rd.flatten() {
                let name = e.file_name().to_string_lossy().into_owned();
                if name.starts_with("ckpt_") && name.ends_with(".tmp") && std::fs::remove_file(e.path()).is_ok() {
                    println!("removed partial checkpoint {name}");
                }
            }
        }
        for old in retention_victims(&checkpoints(&a.run_dir), t, a.keep_checkpoints) {
            if std::fs::remove_file(&old).is_ok() {
                println!("removed old checkpoint {}", old.display());
            }
        }
        let t0 = Instant::now();
        let n = save_checkpoint(&p, &meta, &tree, &st).expect("save checkpoint");
        println!("checkpoint {} ({:.2} GiB) written in {:.1}s", p.display(), n as f64 / 1073741824.0, t0.elapsed().as_secs_f64());
    };

    loop {
        // next epoch end: chunk, next discount, next snapshot
        let mut to = t + chunk;
        if let Some(p) = planned_total {
            to = to.min(p.max(t + 1));
        }
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
                let plan = a.plan_minutes.unwrap_or(a.minutes);
                // planned run = warm-up iterations + the rest of the plan at the measured rate
                let planned = t + (rate * (plan * 60.0 - w).max(0.0)) as u64;
                sched = Schedule::pluribus_scaled(planned, a.strategy_interval);
                apply_flags(&mut sched, &a, Some(planned));
                in_warmup = false;
                phase = "warmup_end";
                let skipped = (1u64..).take_while(|k| k * sched.discount_every <= t).count();
                let txt = format!(
                    "warm-up: {:.1} min, {} iterations, rate over second half {:.0} it/s\nplanned: {} min -> {} iterations; paper minute = {:.0} iterations\nschedule: {:?}\ndiscount boundaries passed during the warm-up (skipped): {}\n",
                    w / 60.0,
                    t,
                    rate,
                    plan,
                    planned,
                    planned as f64 / 11_520.0,
                    sched,
                    skipped
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
                let k = sched.snapshot_index(t);
                let ss = snapshot_add(&snapshot_path, &tree, &st, k, a.threads).expect("snapshot");
                snapshots = ss.total;
                println!("post-flop snapshot {k} at iteration {t} in {:.1}s: {ss:?}", s0.elapsed().as_secs_f64());
            }
        }

        let stop_req = a.stop_file.as_ref().is_some_and(|p| p.exists());
        let out_of_time = wall0.elapsed().as_secs_f64() >= a.minutes * 60.0;
        let plan_done = planned_total.is_some_and(|p| t >= p);
        // measured vs planned speed every 6 h, with the projected finish date
        if let Some(p) = planned_total {
            if prog_last.0.elapsed().as_secs_f64() >= PROGRESS_SECS || plan_done {
                let dt = prog_last.0.elapsed().as_secs_f64();
                let rate = (t - prog_last.1) as f64 / dt.max(1e-9);
                let since = (t - prog_t0) as f64 / wall0.elapsed().as_secs_f64().max(1e-9);
                let planned_rate = p as f64 / (a.plan_minutes.unwrap_or(a.minutes) * 60.0);
                let remain = p.saturating_sub(t) as f64 / rate.max(1e-9);
                let now = unix_now();
                let line = format!(
                    "{}	{t}	{:.4}	{rate:.0}	{since:.0}	{planned_rate:.0}	{:.3}	{}",
                    utc(now),
                    t as f64 / p as f64,
                    rate / planned_rate,
                    utc(now + remain as u64)
                );
                let path = a.run_dir.join("progress.tsv");
                let new = !path.exists();
                let mut f = std::fs::OpenOptions::new().create(true).append(true).open(path).unwrap();
                if new {
                    writeln!(f, "utc	iteration	frac_done	it_per_s_last_window	it_per_s_this_process	planned_it_per_s	measured_over_planned	projected_finish_utc").unwrap();
                }
                writeln!(f, "{line}").unwrap();
                println!("progress: {line}");
                prog_last = (Instant::now(), t);
            }
        }
        let finishing = stop_req || out_of_time || plan_done;
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
                "\t{:.5}\t{:.5}\t{}\t{}\t{}\t{:.3}\t{:.4}\t{}\t{:.4}\t{}",
                ch_pre,
                ch_post,
                total.regret.saturated,
                total.regret.floored,
                snapshots,
                st.allocated_slots() as f64 * 4.0 / 1073741824.0,
                st.allocated_slots() as f64 / tree.n_regrets as f64,
                st.allocated_action_sequences(),
                st.allocated_action_sequences() as f64 / tree.actions.iter().sum::<u64>() as f64,
                phase
            );
            // share of traverser actions skipped by pruning, in pruning iterations
            row += &format!("	{:.5}", interval.pruned_actions as f64 / interval.prune_considered.max(1) as f64);
            println!("{row}");
            writeln!(lg.file, "{row}").unwrap();
            lg.file.flush().unwrap();
            interval = ThreadStats::default();
            lg.last = Instant::now();
        }
        if finishing {
            println!("stopping at iteration {t} ({})", if stop_req { "stop file" } else if plan_done { "planned iterations done" } else { "time budget" });
            break;
        }
    }
}
