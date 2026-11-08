//! D-SRCH-3: subgame size (action sequences, players' actions only; Supp p. 13 says 100–2,000)
//! for every subgame root Pluribus searched from, under a search config.
//!
//! `cargo run --release --bin subgame_sizes -- --config configs/search_root.cfg --later configs/search_later.cfg`
//!
//! Roots come from `analysis/subgame_roots.py`. Prints, per (street, players at the root):
//! roots, min / p10 / median / p90 / max sequences, and the share inside [100, 2000]; plus the
//! preflop subgame from the start of the hand (6 players). `--street N` keeps only roots on street
//! N; `--per-root` also prints one "street players sequences" line per root.

use nlhe::betting::{Rules, MAX_PLAYERS};
use nlhe::config::AbstractionConfig;
use nlhe::subgame::{action_sequences, round_start, Subgame};
use std::collections::BTreeMap;

const LO: u64 = 100;
const HI: u64 = 2_000;
/// node cap per subgame (a truncated subgame prints as 18446744073709551615 = "over the cap")
const CAP: usize = 200_000;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let get = |k: &str| args.iter().position(|a| a == k).map(|i| args[i + 1].clone());
    let cfg_path = get("--config").expect("--config");
    let load = |p: &str| AbstractionConfig::parse(&std::fs::read_to_string(p).expect("read config")).expect("parse config");
    let cfg = load(&cfg_path);
    // --later: sizes on the streets after the root's (default: --config for every street)
    let later = get("--later").map(|p| load(&p)).unwrap_or_else(|| cfg.clone());
    let roots = std::fs::read_to_string(get("--roots").unwrap_or("../data/subgame_roots.tsv".into())).expect("read roots");
    let rules = Rules::default();
    // --street N: only roots on that street (1-3); the preflop row needs no filter
    let only: Option<u8> = get("--street").map(|s| s.parse().expect("--street 1..3"));
    // --per-root: also print "street players sequences" for every root (for calibration)
    let per_root = args.iter().any(|a| a == "--per-root");

    let mut by: BTreeMap<(u8, u32), Vec<u64>> = BTreeMap::new();
    for line in roots.lines().skip(1) {
        let f: Vec<&str> = line.split('\t').collect();
        let street: u8 = f[1].parse().unwrap();
        if only.is_some_and(|o| o != street) {
            continue;
        }
        let folded: u8 = f[2].parse().unwrap();
        let mut total = [0u32; MAX_PLAYERS];
        for (p, t) in total.iter_mut().enumerate() {
            *t = f[3 + p].parse().unwrap();
        }
        let root = round_start(&rules, street, total, folded);
        // a subgame over the cap is far outside [100, 2000]; report it as u64::MAX
        let g = Subgame::build_with(&rules, &cfg, &later, root, CAP);
        let n = if g.truncated { u64::MAX } else { action_sequences(&g) };
        if per_root {
            println!("{street}	{}	{n}", rules.n_in_hand(&root));
        }
        by.entry((street, rules.n_in_hand(&root) as u32)).or_default().push(n);
    }
    if only.is_none() {
        by.insert((0, 6), vec![action_sequences(&Subgame::build_with(&rules, &cfg, &later, rules.root(), usize::MAX))]);
    }

    println!("config {cfg_path}");
    println!("street\tplayers\troots\tmin\tp10\tmedian\tp90\tmax\tin_range");
    let names = ["preflop", "flop", "turn", "river"];
    let mut all = Vec::new();
    for ((s, p), mut v) in by {
        v.sort_unstable();
        let q = |x: f64| v[((v.len() - 1) as f64 * x).round() as usize];
        let inr = v.iter().filter(|&&n| (LO..=HI).contains(&n)).count();
        println!(
            "{}\t{p}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.1}%",
            names[s as usize], v.len(), v[0], q(0.1), q(0.5), q(0.9), v[v.len() - 1], 100.0 * inr as f64 / v.len() as f64
        );
        if s > 0 {
            all.extend(v);
        }
    }
    let inr = all.iter().filter(|&&n| (LO..=HI).contains(&n)).count();
    all.sort_unstable();
    println!(
        "post-flop roots: {}, median {}, in [{LO}, {HI}]: {:.1}%",
        all.len(), all[all.len() / 2], 100.0 * inr as f64 / all.len() as f64
    );
}
