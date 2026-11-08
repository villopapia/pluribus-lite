//! M4 NLHE validation (heads-up): search agent (Algorithm 2 on turn and river) vs blueprint agent,
//! head-to-head on duplicate deals (`docs/m4_validation.md`).
//!
//! `hu_match --run-dir ../data/runs/hu_dev --config nlhe/configs/hu_dev.cfg
//!   --abstraction-dir ../data/abstraction/j63 --buckets 63 [--deals 400] [--threads 20]
//!   [--turn-iters 150] [--river-iters 400] [--seed 1] [--play-average] [--measure-expl]`
//!
//! Each deal (two hands, five board cards) is played twice with the seats swapped (duplicate).
//! Reported per deal: the search agent's mean result over both seats, raw and with the
//! AIVAT-style chance correction (equity value function at every flop/turn/river deal, D-EVAL-2),
//! in mbb per hand (1 bb = 100 chips), with standard errors over deals.

use cfr::rng::Rng;
use nlhe::betting::Rules;
use nlhe::blueprint::{load_checkpoint, SnapshotAverage, Storage, Tree};
use nlhe::buckets::{CardAbstraction, TableAbstraction};
use nlhe::config::AbstractionConfig;
use nlhe::hu_agent::{play_hand, Agent, Blueprint, BlueprintAgent, SearchAgent, SearchSettings};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
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
    let threads: usize = get("--threads", "20").parse().unwrap();
    let seed: u64 = get("--seed", "1").parse().unwrap();
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
    println!("search: turn {} / river {} iterations, plays {}; deals {deals} (x2 seats), threads {threads}, seed {seed}", set.turn_iters, set.river_iters, if set.play_average { "the average (diagnostic)" } else { "the final iteration (paper)" });

    struct Row {
        raw: f64,
        adj: f64,
        turn: bool,
        searches: u64,
        re: u64,
        zero: u64,
        secs: f64,
        expl: Vec<(f64, f64)>,
        bp_off: u64,
    }
    let rows: Mutex<Vec<(u64, Row)>> = Mutex::new(Vec::new());
    let next = AtomicU64::new(0);
    std::thread::scope(|sc| {
        for _ in 0..threads {
            sc.spawn(|| loop {
                let d = next.fetch_add(1, Ordering::Relaxed);
                if d >= deals {
                    break;
                }
                let mut rng = Rng::new(mix(seed, d, 0));
                let mut deck: Vec<u8> = (0..52).collect();
                for i in 0..9 {
                    let j = i + (rng.next_u64() % (52 - i) as u64) as usize;
                    deck.swap(i, j);
                }
                let holes = [(deck[0], deck[1]), (deck[2], deck[3])];
                let board = [deck[4], deck[5], deck[6], deck[7], deck[8]];
                let mut r = Row { raw: 0.0, adj: 0.0, turn: false, searches: 0, re: 0, zero: 0, secs: 0.0, expl: vec![], bp_off: 0 };
                for s_seat in 0..2 {
                    let b_seat = 1 - s_seat;
                    let mut sa = SearchAgent::new(&bp, &root_cfg, &later_cfg, set.clone(), s_seat, holes[s_seat], mix(seed, d, 10 + s_seat as u64));
                    sa.measure_expl = measure;
                    let mut ba = BlueprintAgent::new(&bp, holes[b_seat], mix(seed, d, 20 + b_seat as u64));
                    let res = {
                        let mut agents: [&mut dyn Agent; 2] = if s_seat == 0 { [&mut sa, &mut ba] } else { [&mut ba, &mut sa] };
                        play_hand(&rules, holes, board, &mut agents)
                    };
                    r.raw += res.chips[s_seat] / 2.0;
                    r.adj += (res.chips[s_seat] - res.correction[s_seat]) / 2.0;
                    r.turn |= res.reached_turn;
                    r.searches += sa.stats.searches;
                    r.re += sa.stats.re_searches;
                    r.zero += sa.stats.zero_prob_events;
                    r.secs += sa.stats.search_secs;
                    r.expl.extend(sa.stats.river_expl.iter().copied());
                    r.bp_off += ba.off_tree;
                }
                rows.lock().unwrap().push((d, r));
            });
        }
    });
    let mut rows = rows.into_inner().unwrap();
    rows.sort_by_key(|x| x.0);
    let mbb = 10.0; // chips -> mbb (1 bb = 100 chips)
    let raw: Vec<f64> = rows.iter().map(|x| x.1.raw * mbb).collect();
    let adj: Vec<f64> = rows.iter().map(|x| x.1.adj * mbb).collect();
    let (mr, sr) = mean_se(&raw);
    let (ma, sa) = mean_se(&adj);
    let turn = rows.iter().filter(|x| x.1.turn).count();
    let searches: u64 = rows.iter().map(|x| x.1.searches).sum();
    let re: u64 = rows.iter().map(|x| x.1.re).sum();
    let zero: u64 = rows.iter().map(|x| x.1.zero).sum();
    let secs: f64 = rows.iter().map(|x| x.1.secs).sum();
    let off: u64 = rows.iter().map(|x| x.1.bp_off).sum();
    println!("\nsearch agent vs blueprint agent, {} deals ({} hands), {turn} deals reached the turn", rows.len(), 2 * rows.len());
    println!("  raw:                      {mr:+.1} mbb/hand, s.e. {sr:.1}");
    println!("  AIVAT-style (chance):     {ma:+.1} mbb/hand, s.e. {sa:.1}  (variance x{:.2} lower)", (sr / sa).powi(2));
    println!("  searches {searches} (re-searches after off-tree opponent actions {re}), {:.2}s per search, zero-probability observations {zero}", secs / searches.max(1) as f64);
    println!("  blueprint agent: off-tree actions translated {off}");
    let ex: Vec<(f64, f64)> = rows.iter().flat_map(|x| x.1.expl.iter().copied()).collect();
    if !ex.is_empty() {
        let n = ex.len() as f64;
        println!(
            "  river subgames: exploitability within the subgame, final iteration {:.1} / average {:.1} chips (mean of {} searches)",
            ex.iter().map(|e| e.0).sum::<f64>() / n,
            ex.iter().map(|e| e.1).sum::<f64>() / n,
            ex.len()
        );
    }
    println!("total {:.1}s", t0.elapsed().as_secs_f64());
}
