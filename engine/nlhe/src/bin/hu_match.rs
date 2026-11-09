//! M4 NLHE validation (heads-up): search agent (Algorithm 2 on turn and river) vs blueprint agent,
//! head-to-head on duplicate deals (`docs/m4_validation.md`).
//!
//! `hu_match --run-dir ../data/runs/hu_dev --config nlhe/configs/hu_dev.cfg
//!   --abstraction-dir ../data/abstraction/j63 --buckets 63 [--deals 400] [--threads 12]
//!   [--turn-iters 150] [--river-iters 400] [--seed 1] [--play-average] [--measure-expl]
//!   [--out ../data/hu_match.tsv] [--max-secs 6600] [--capture ../data/subgames.txt]`
//!
//! Each deal (two hands, five board cards) is played twice with the seats swapped (duplicate).
//! One TSV row per finished deal goes to `--out`; a rerun with the same `--out` skips finished
//! deals (resumable; `--max-secs` stops taking new deals, for ≤ 2 h chunks) and the summary is
//! always computed from the whole file. Reported per deal: the search agent's mean result over
//! both seats, raw, with AIVAT chance terms, and with full AIVAT (chance + both agents'
//! decisions, D-EVAL-2), in mbb per hand (1 bb = 100 chips), with standard errors over deals.

use cfr::rng::Rng;
use nlhe::betting::Rules;
use nlhe::blueprint::{load_checkpoint, SnapshotAverage, Storage, Tree};
use nlhe::buckets::{CardAbstraction, TableAbstraction};
use nlhe::config::AbstractionConfig;
use nlhe::hu_agent::{play_hand, Agent, Blueprint, BlueprintAgent, SearchAgent, SearchSettings};
use std::collections::HashSet;
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;

fn mix(a: u64, b: u64, c: u64) -> u64 {
    let mut z = a ^ b.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ c.wrapping_mul(0xD1B5_4A32_D192_ED03);
    z = (z ^ (z >> 31)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z ^ (z >> 29)
}

fn mean_se(v: &[f64]) -> (f64, f64) {
    let n = v.len() as f64;
    let m = v.iter().sum::<f64>() / n;
    let var = v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (n - 1.0).max(1.0);
    (m, (var / n).sqrt())
}

const HEADER: &str = "deal\traw\tchance\tfull\tturn\tsearches\tre\tzero\tsecs\tbp_off\texpl";

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let get = |k: &str, d: &str| args.iter().position(|a| a == k).map_or(d.to_string(), |i| args[i + 1].clone());
    let flag = |k: &str| args.iter().any(|a| a == k);
    let load = |p: &str| AbstractionConfig::parse(&std::fs::read_to_string(p).expect("read config")).expect("parse config");
    let run_dir = PathBuf::from(get("--run-dir", "../data/runs/hu_dev"));
    let cfg = load(&get("--config", "nlhe/configs/hu_dev.cfg"));
    let root_cfg = load(&get("--search-root", "nlhe/configs/search_root.cfg"));
    let later_cfg = load(&get("--search-later", "nlhe/configs/search_later.cfg"));
    let deals: u64 = get("--deals", "400").parse().unwrap();
    let threads: usize = get("--threads", "12").parse().unwrap();
    let seed: u64 = get("--seed", "1").parse().unwrap();
    let max_secs: f64 = get("--max-secs", "1e18").parse().unwrap();
    let out_path = PathBuf::from(get("--out", "../data/hu_match.tsv"));
    let capture = args.iter().position(|a| a == "--capture").map(|i| PathBuf::from(&args[i + 1]));
    let set = SearchSettings {
        turn_iters: get("--turn-iters", "150").parse().unwrap(),
        river_iters: get("--river-iters", "400").parse().unwrap(),
        play_average: flag("--play-average"),
    };
    let measure = flag("--measure-expl");

    let t0 = Instant::now();
    let rules = Rules { players: cfg.players, ..Rules::default() };
    assert_eq!(rules.players, 2, "heads-up only");
    let abs = TableAbstraction::load(&PathBuf::from(get("--abstraction-dir", "../data/abstraction/j63")), get("--buckets", "63").parse().unwrap()).expect("abstraction");
    let tree = Tree::build(&rules, &cfg, abs.buckets());
    let st = Storage::new(&tree);
    let mut ckpts: Vec<PathBuf> = std::fs::read_dir(&run_dir)
        .expect("run dir")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.starts_with("ckpt_") && n.ends_with(".bin")))
        .collect();
    ckpts.sort();
    let ck = ckpts.last().expect("no checkpoint in the run dir");
    let meta = load_checkpoint(ck, &tree, &st).expect("load checkpoint");
    let snap = SnapshotAverage::load(&run_dir.join("postflop_snapshots.bin"), &tree).ok();
    println!(
        "blueprint {} at iteration {} ({:.1} min), post-flop snapshots: {}; loaded in {:.1}s",
        ck.display(),
        meta.iteration,
        meta.elapsed_secs / 60.0,
        snap.as_ref().map_or(0, |s| s.total),
        t0.elapsed().as_secs_f64()
    );
    let bp = Blueprint { tree: &tree, st: &st, snap: snap.as_ref(), abs: &abs, cfg: &cfg, rules };
    println!(
        "search: turn {} / river {} iterations, plays {}; deals {deals} (x2 seats), threads {threads}, seed {seed}; out {}",
        set.turn_iters,
        set.river_iters,
        if set.play_average { "the average (ablation)" } else { "the final iteration (paper)" },
        out_path.display()
    );

    // resume: deals already in the file are skipped
    let done: HashSet<u64> = std::fs::read_to_string(&out_path)
        .map(|t| t.lines().skip(1).filter_map(|l| l.split('\t').next()?.parse().ok()).collect())
        .unwrap_or_default();
    if done.is_empty() {
        std::fs::write(&out_path, format!("{HEADER}\n")).expect("write out");
    }
    let todo: Vec<u64> = (0..deals).filter(|d| !done.contains(d)).collect();
    println!("{} deals already done, {} to play", done.len(), todo.len());
    let out = Mutex::new(std::fs::OpenOptions::new().append(true).open(&out_path).expect("open out"));
    let caps: Mutex<Vec<(u64, usize, usize, String)>> = Mutex::new(Vec::new());
    let next = AtomicUsize::new(0);
    let finished = AtomicU64::new(0);
    std::thread::scope(|sc| {
        for _ in 0..threads {
            sc.spawn(|| loop {
                if t0.elapsed().as_secs_f64() > max_secs {
                    break;
                }
                let Some(&d) = todo.get(next.fetch_add(1, Ordering::Relaxed)) else { break };
                let mut rng = Rng::new(mix(seed, d, 0));
                let mut deck: Vec<u8> = (0..52).collect();
                for i in 0..9 {
                    let j = i + (rng.next_u64() % (52 - i) as u64) as usize;
                    deck.swap(i, j);
                }
                let holes = [(deck[0], deck[1]), (deck[2], deck[3])];
                let board = [deck[4], deck[5], deck[6], deck[7], deck[8]];
                let (mut raw, mut chance, mut full) = (0f64, 0f64, 0f64);
                let (mut turn, mut searches, mut re, mut zero, mut secs, mut off) = (false, 0u64, 0u64, 0u64, 0f64, 0u64);
                let mut expl = Vec::new();
                for s_seat in 0..2 {
                    let b_seat = 1 - s_seat;
                    let mut sa = SearchAgent::new(&bp, &root_cfg, &later_cfg, set.clone(), s_seat, holes[s_seat], mix(seed, d, 10 + s_seat as u64));
                    sa.measure_expl = measure;
                    sa.capture = capture.is_some();
                    let mut ba = BlueprintAgent::new(&bp, holes[b_seat], mix(seed, d, 20 + b_seat as u64));
                    let res = {
                        let mut agents: [&mut dyn Agent; 2] = if s_seat == 0 { [&mut sa, &mut ba] } else { [&mut ba, &mut sa] };
                        play_hand(&rules, holes, board, &mut agents)
                    };
                    let c = res.chips[s_seat];
                    raw += c / 2.0;
                    chance += (c - res.correction[s_seat]) / 2.0;
                    full += (c - res.correction[s_seat] - res.dec_correction[s_seat]) / 2.0;
                    turn |= res.reached_turn;
                    searches += sa.stats.searches;
                    re += sa.stats.re_searches;
                    zero += sa.stats.zero_prob_events;
                    secs += sa.stats.search_secs;
                    off += ba.off_tree;
                    expl.extend(sa.stats.river_expl.iter().map(|e| format!("{:.4},{:.4},{}", e.0, e.1, e.2)));
                    let mut cp = caps.lock().unwrap();
                    for (k, l) in sa.captured.drain(..).enumerate() {
                        cp.push((d, s_seat, k, l));
                    }
                }
                let row = format!("{d}\t{raw}\t{chance}\t{full}\t{}\t{searches}\t{re}\t{zero}\t{secs:.2}\t{off}\t{}\n", turn as u8, expl.join(";"));
                let mut f = out.lock().unwrap();
                f.write_all(row.as_bytes()).unwrap();
                f.flush().unwrap();
                let n = finished.fetch_add(1, Ordering::Relaxed) + 1;
                if n % 25 == 0 {
                    eprintln!("progress: {} / {deals} deals ({:.0} s)", done.len() as u64 + n, t0.elapsed().as_secs_f64());
                }
            });
        }
    });
    if let Some(p) = &capture {
        let mut cp = caps.into_inner().unwrap();
        cp.sort_by_key(|x| (x.0, x.1, x.2));
        let mut f = std::fs::OpenOptions::new().create(true).append(true).open(p).expect("capture file");
        for (d, s, k, l) in cp {
            writeln!(f, "{d} {s} {k} | {l}").unwrap();
        }
    }

    // summary over the whole file
    let text = std::fs::read_to_string(&out_path).expect("read out");
    let rows: Vec<Vec<&str>> = text.lines().skip(1).map(|l| l.split('\t').collect()).collect();
    let col = |i: usize| -> Vec<f64> { rows.iter().map(|r| r[i].parse::<f64>().unwrap() * 10.0).collect() };
    let n = rows.len();
    println!("\nsearch agent vs blueprint agent, {n} deals ({} hands) of {deals}; {} deals reached the turn", 2 * n, rows.iter().filter(|r| r[4] == "1").count());
    let (_, se_raw) = mean_se(&col(1));
    for (name, i) in [("raw", 1), ("AIVAT chance terms", 2), ("AIVAT full (chance + decisions)", 3)] {
        let (m, se) = mean_se(&col(i));
        println!("  {name:32} {m:+8.1} mbb/hand, s.e. {se:6.1}, 95% CI [{:+.0}, {:+.0}], variance x{:.2} lower than raw", m - 1.96 * se, m + 1.96 * se, (se_raw / se).powi(2));
    }
    let sum = |i: usize| -> f64 { rows.iter().map(|r| r[i].parse::<f64>().unwrap()).sum() };
    println!(
        "  searches {} (re-searches after off-tree opponent actions {}), {:.2}s per search, zero-probability observations {}; blueprint agent: off-tree actions translated {}",
        sum(5),
        sum(6),
        sum(8) / sum(5).max(1.0),
        sum(7),
        sum(9)
    );
    let ex: Vec<(f64, f64, f64)> = rows
        .iter()
        .flat_map(|r| r.get(10).copied().unwrap_or("").split(';').filter(|s| !s.is_empty()).map(|s| {
            let v: Vec<f64> = s.split(',').map(|x| x.parse().unwrap()).collect();
            (v[0], v[1], v[2])
        }).collect::<Vec<_>>())
        .collect();
    if !ex.is_empty() {
        let k = ex.len() as f64;
        let m = |f: &dyn Fn(&(f64, f64, f64)) -> f64| ex.iter().map(f).sum::<f64>() / k;
        println!(
            "  river subgames ({} searches), exploitability within the subgame: final iteration {:.1} chips = {:.0} mbb = {:.2}% of pot; average {:.1} chips = {:.0} mbb = {:.2}% of pot (pot = both contributions at the root; per-search % averaged)",
            ex.len(),
            m(&|e| e.0),
            m(&|e| e.0) * 10.0,
            m(&|e| 100.0 * e.0 / e.2),
            m(&|e| e.1),
            m(&|e| e.1) * 10.0,
            m(&|e| 100.0 * e.1 / e.2)
        );
    }
    println!("total {:.1}s", t0.elapsed().as_secs_f64());
}
