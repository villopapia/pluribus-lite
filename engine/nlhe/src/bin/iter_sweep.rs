//! Iteration sweep on a fixed set of captured heads-up subgames (`hu_match --capture`):
//! exploitability within the subgame of the final iteration vs the weighted average, at
//! increasing iteration budgets, with the solve time per search (`docs/m4_validation.md`).
//!
//! `iter_sweep --subgames ../data/subgames.txt --out ../data/iter_sweep.tsv [--river 200]
//!   [--turn 100] [--river-levels 400,1600,6400,25600] [--turn-levels 150,600,2400,9600]
//!   [--threads 12] [--max-secs 3600] [--dump FILE]`  /  `iter_sweep --summary --out FILE`
//!
//! The first `--river` distinct river and `--turn` distinct turn subgames of the file (in file
//! order) are solved,
//! longest jobs first; one TSV row per (subgame, level). Resumable: subgames with all levels in
//! `--out` are skipped; `--max-secs` stops starting new subgames (≤ 2 h chunks). `--dump` writes
//! the root strategies (average and final, every hand) at the first level, for the f32/f64
//! comparison (build with `--features f64-tables`). The solver samples one board per iteration
//! (the paper); `--enumerate` enumerates chance (the first sweep); `--boards-per-iter k` samples
//! k boards per iteration.

use nlhe::betting::Rules;
use nlhe::config::AbstractionConfig;
use nlhe::hu_agent::parse_capture;
use nlhe::subgame::Subgame;
use nlhe::vsolve::VectorSolver;
use std::collections::{HashMap, HashSet};
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;

const HEADER: &str = "id\tstreet\tlevel\tpot\texpl_final\texpl_avg\tvalue_final\tvalue_avg\tsolve_secs";

fn summary(path: &PathBuf) {
    let text = std::fs::read_to_string(path).expect("read out");
    // last row wins per (id, level)
    let mut rows: HashMap<(String, u64), Vec<f64>> = HashMap::new();
    let mut street: HashMap<String, String> = HashMap::new();
    for l in text.lines().skip(1) {
        let f: Vec<&str> = l.split('\t').collect();
        street.insert(f[0].into(), f[1].into());
        rows.insert((f[0].into(), f[2].parse().unwrap()), f[3..].iter().map(|x| x.parse().unwrap()).collect());
    }
    println!("street  level   n  final:%pot (median)   mbb   avg:%pot (median)   mbb   final/avg  solve s/search (mean, max)");
    for st in ["turn", "river"] {
        let mut levels: Vec<u64> = rows.keys().filter(|k| street[&k.0] == st).map(|k| k.1).collect::<HashSet<_>>().into_iter().collect();
        levels.sort();
        for lv in levels {
            let v: Vec<&Vec<f64>> = rows.iter().filter(|(k, _)| k.1 == lv && street[&k.0] == st).map(|x| x.1).collect();
            let n = v.len() as f64;
            let mean = |f: &dyn Fn(&Vec<f64>) -> f64| v.iter().map(|r| f(r)).sum::<f64>() / n;
            let median = |f: &dyn Fn(&Vec<f64>) -> f64| {
                let mut x: Vec<f64> = v.iter().map(|r| f(r)).collect();
                x.sort_by(|a, b| a.partial_cmp(b).unwrap());
                x[x.len() / 2]
            };
            let pf = |r: &Vec<f64>| 100.0 * r[1] / r[0];
            let pa = |r: &Vec<f64>| 100.0 * r[2] / r[0];
            println!(
                "{st:6} {lv:6} {:4}  {:6.2} ({:6.2})  {:6.0}  {:6.2} ({:6.2})  {:6.0}  {:8.1}  {:8.1} {:8.1}",
                v.len(),
                mean(&pf),
                median(&pf),
                mean(&|r| r[1] * 10.0),
                mean(&pa),
                median(&pa),
                mean(&|r| r[2] * 10.0),
                mean(&|r| r[1]) / mean(&|r| r[2]),
                mean(&|r| r[5]),
                v.iter().map(|r| r[5]).fold(0.0, f64::max)
            );
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let get = |k: &str, d: &str| args.iter().position(|a| a == k).map_or(d.to_string(), |i| args[i + 1].clone());
    let out_path = PathBuf::from(get("--out", "../data/iter_sweep.tsv"));
    if args.iter().any(|a| a == "--summary") {
        return summary(&out_path);
    }
    let load = |p: &str| AbstractionConfig::parse(&std::fs::read_to_string(p).expect("read config")).expect("parse config");
    let root_cfg = load(&get("--search-root", "nlhe/configs/search_root.cfg"));
    let later_cfg = load(&get("--search-later", "nlhe/configs/search_later.cfg"));
    let levels = |k: &str, d: &str| -> Vec<u64> { get(k, d).split(',').map(|x| x.parse().unwrap()).collect() };
    let river_levels = levels("--river-levels", "400,1600,6400,25600");
    let turn_levels = levels("--turn-levels", "150,600,2400,9600");
    let n_river: usize = get("--river", "200").parse().unwrap();
    let n_turn: usize = get("--turn", "100").parse().unwrap();
    let threads: usize = get("--threads", "12").parse().unwrap();
    let max_secs: f64 = get("--max-secs", "1e18").parse().unwrap();
    let dump = args.iter().position(|a| a == "--dump").map(|i| PathBuf::from(&args[i + 1]));
    let enumerate = args.iter().any(|a| a == "--enumerate");
    let k: usize = get("--boards-per-iter", "1").parse().unwrap();
    let rules = Rules { players: 2, ..Rules::default() };
    let t0 = Instant::now();

    let text = std::fs::read_to_string(get("--subgames", "../data/subgames.txt")).expect("read subgames");
    let (mut river, mut turn) = (Vec::new(), Vec::new());
    // identical subgames (same root, beliefs, frozen entries; e.g. both seats of a duplicate deal
    // with the same pre-turn play) are counted once
    let mut seen = HashSet::new();
    for l in text.lines() {
        let (id, rest) = l.split_once(" | ").expect("capture line");
        if !seen.insert(rest) {
            continue;
        }
        let nb: usize = rest.split_whitespace().next().unwrap().parse().unwrap();
        match nb {
            5 if river.len() < n_river => river.push((id.to_string(), rest)),
            4 if turn.len() < n_turn => turn.push((id.to_string(), rest)),
            _ => {}
        }
    }
    let done: HashSet<(String, u64)> = std::fs::read_to_string(&out_path)
        .map(|t| t.lines().skip(1).map(|l| {
            let f: Vec<&str> = l.split('\t').collect();
            (f[0].to_string(), f[2].parse().unwrap())
        }).collect())
        .unwrap_or_default();
    if done.is_empty() {
        std::fs::write(&out_path, format!("{HEADER}\n")).expect("write out");
    }
    // longest jobs first: turns, then rivers
    let jobs: Vec<(&str, &String, &str, &Vec<u64>)> = turn
        .iter()
        .map(|(id, l)| ("turn", id, *l, &turn_levels))
        .chain(river.iter().map(|(id, l)| ("river", id, *l, &river_levels)))
        .filter(|(_, id, _, lv)| !done.contains(&((*id).clone(), *lv.last().unwrap())))
        .collect();
    println!("{} turn + {} river subgames selected, {} to solve; levels river {river_levels:?}, turn {turn_levels:?}; threads {threads}; chance {}", turn.len(), river.len(), jobs.len(), if enumerate { "enumerated".to_string() } else { format!("sampled, {k} board(s) per iteration") });
    let out = Mutex::new(std::fs::OpenOptions::new().append(true).open(&out_path).expect("open out"));
    let dump_f = dump.map(|p| Mutex::new(std::fs::File::create(p).expect("dump file")));
    let next = AtomicUsize::new(0);
    let fin = AtomicUsize::new(0);
    std::thread::scope(|sc| {
        for _ in 0..threads {
            sc.spawn(|| loop {
                if t0.elapsed().as_secs_f64() > max_secs {
                    break;
                }
                let Some(&(street, id, line, lv)) = jobs.get(next.fetch_add(1, Ordering::Relaxed)) else { break };
                let c = parse_capture(line);
                let g = Subgame::build_extra(&rules, &root_cfg, &later_cfg, c.root, c.extra);
                let mut s = VectorSolver::new(g, rules, &c.board, c.ranges);
                if !enumerate {
                    // boards fixed per subgame: seed from the id
                    let seed = id.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| (h ^ b as u64).wrapping_mul(0x100_0000_01b3));
                    s = s.with_sampling(seed).with_boards_per_iter(k);
                }
                s.frozen = c.frozen;
                let pot: f64 = c.root.total.iter().map(|&x| x as f64).sum();
                let mut solve = 0.0;
                for (li, &level) in lv.iter().enumerate() {
                    let t = Instant::now();
                    s.run(level - s.iterations);
                    solve += t.elapsed().as_secs_f64();
                    let (ef, ea) = (s.exploitability(false), s.exploitability(true));
                    let (vf, va) = (s.game_value(false), s.game_value(true));
                    let row = format!("{id}\t{street}\t{level}\t{pot}\t{ef}\t{ea}\t{vf}\t{va}\t{solve:.3}\n");
                    out.lock().unwrap().write_all(row.as_bytes()).unwrap();
                    if li == 0 {
                        if let Some(f) = &dump_f {
                            let mut o = format!("{id}");
                            for avg in [true, false] {
                                for x in s.strategy_all(0, &[], avg) {
                                    o += &format!(" {x}");
                                }
                            }
                            for r in &s.ranges[s.g.nodes[0].state.to_act as usize] {
                                o += &format!(" {r}");
                            }
                            writeln!(f.lock().unwrap(), "{o}").unwrap();
                        }
                    }
                }
                out.lock().unwrap().flush().unwrap();
                let n = fin.fetch_add(1, Ordering::Relaxed) + 1;
                eprintln!("{n}/{} done ({street} {id}, {solve:.0}s solve) at {:.0}s", jobs.len(), t0.elapsed().as_secs_f64());
            });
        }
    });
    println!("stopped after {:.0}s", t0.elapsed().as_secs_f64());
    summary(&out_path);
}
