//! D-SRCH-14: how often Pluribus' preflop search would trigger in the 10,000 logged hands, and
//! how large those preflop subgames are.
//!
//! `cargo run --release --bin preflop_search -- [--blueprint configs/faithful.cfg]
//!   [--search configs/search_root.cfg] [--lines ../data/preflop_lines.tsv]`
//!
//! Trigger (Supp p. 15): an opponent's preflop raise **more than $100 off every raise size of
//! the blueprint action abstraction** (raise-to amounts of `--blueprint` at the actual state,
//! all-in included) while **no more than four players remain** in the hand (after the raise),
//! Pluribus among them. Lines come from `analysis/subgame_roots.py`.
//!
//! Subgame size per trigger, two readings of "search from the start of the round" (p. 21)
//! once players have folded:
//! * (i) full round: every player's decisions from the start of the hand (as for 6 players);
//! * (ii) folded players fixed: seats that folded before the trigger only fold.
//! Off-tree actions added by Algorithm 2 (AddAction) are not counted.

use nlhe::betting::{Action, Next, Rules};
use nlhe::config::AbstractionConfig;
use nlhe::subgame::{action_sequences, Subgame};
use std::collections::BTreeMap;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let get = |k: &str, d: &str| args.iter().position(|a| a == k).map_or(d.to_string(), |i| args[i + 1].clone());
    let load = |p: &str| AbstractionConfig::parse(&std::fs::read_to_string(p).expect("read config")).expect("parse config");
    let bp_path = get("--blueprint", "nlhe/configs/faithful.cfg");
    let (bp, search) = (load(&bp_path), load(&get("--search", "nlhe/configs/search_root.cfg")));
    let lines = std::fs::read_to_string(get("--lines", "../data/preflop_lines.tsv")).expect("read lines");
    let rules = Rules::default();
    let full = action_sequences(&Subgame::build(&rules, &search, rules.root()));

    let (mut hands, mut hands_trig, mut offtree_all, mut offtree_crowded) = (0u32, 0u32, 0u32, 0u32);
    let mut by_players: BTreeMap<usize, u32> = BTreeMap::new();
    let mut fixed_sizes: Vec<u64> = Vec::new();
    let mut fixed_cache: BTreeMap<u8, u64> = BTreeMap::new();
    let mut off_by: Vec<u32> = Vec::new();
    for line in lines.lines().skip(1) {
        let f: Vec<&str> = line.split('\t').collect();
        let me: u8 = f[1].parse().unwrap();
        hands += 1;
        let mut s = rules.root();
        let mut triggered = false;
        for a in f.get(2).map_or(vec![], |x| x.split(',').filter(|x| !x.is_empty()).collect::<Vec<_>>()) {
            let p: Vec<&str> = a.split_whitespace().collect();
            let seat: u8 = p[0][1..].parse::<u8>().unwrap() - 1;
            assert_eq!(seat, s.to_act, "{}: replay out of order at {a}", f[0]);
            let act = match p[1] {
                "f" => Action::Fold,
                "cc" => Action::Call,
                "cbr" => Action::RaiseTo(p[2].parse().unwrap()),
                v => panic!("unknown verb {v}"),
            };
            let mut off = None;
            if let Action::RaiseTo(x) = act {
                if seat != me {
                    let tos: Vec<u32> = rules.actions(&s, &bp).into_iter().filter_map(|b| if let Action::RaiseTo(t) = b { Some(t) } else { None }).collect();
                    let dist = tos.iter().map(|&t| t.abs_diff(x)).min().unwrap_or(u32::MAX);
                    if dist > 100 {
                        off = Some(dist);
                    }
                }
            }
            let next = rules.apply(&s, act);
            let n = match next {
                Next::Decision(n) | Next::Terminal(n) => n,
            };
            if let Some(d) = off {
                offtree_all += 1;
                let in_hand = rules.n_in_hand(&n);
                if n.folded & (1 << me) == 0 && in_hand <= 4 {
                    triggered = true;
                    off_by.push(d);
                    *by_players.entry(in_hand).or_default() += 1;
                    let folded = n.folded;
                    let size = *fixed_cache.entry(folded).or_insert_with(|| action_sequences(&Subgame::build_preflop_fixed(&rules, &search, folded, usize::MAX)));
                    fixed_sizes.push(size);
                } else {
                    offtree_crowded += 1;
                }
            }
            match next {
                Next::Decision(n) => s = n,
                Next::Terminal(_) => break,
            }
        }
        hands_trig += triggered as u32;
    }
    println!("blueprint sizes: {bp_path}; hands {hands}");
    println!("opponent preflop raises > $100 off every blueprint size: {offtree_all} (with Pluribus out or > 4 players left: {offtree_crowded})");
    println!("search triggers (<= 4 players, Pluribus in): {} in {hands_trig} hands ({:.2}% of hands)", fixed_sizes.len(), 100.0 * hands_trig as f64 / hands as f64);
    println!("  by players remaining: {by_players:?}");
    off_by.sort_unstable();
    if !off_by.is_empty() {
        println!("  distance to the nearest blueprint size: median ${}, p90 ${}", off_by[off_by.len() / 2], off_by[off_by.len() * 9 / 10]);
    }
    println!("subgame size (search sizes {}):", search.name);
    println!("  (i) full round from the start of the hand: {full} action sequences for every trigger");
    fixed_sizes.sort_unstable();
    if !fixed_sizes.is_empty() {
        let inr = fixed_sizes.iter().filter(|&&n| (100..=2000).contains(&n)).count();
        println!(
            "  (ii) folded players fixed: min {} / median {} / max {}; in [100, 2000]: {:.1}% ({} distinct folded sets)",
            fixed_sizes[0],
            fixed_sizes[fixed_sizes.len() / 2],
            fixed_sizes[fixed_sizes.len() - 1],
            100.0 * inr as f64 / fixed_sizes.len() as f64,
            fixed_cache.len()
        );
    }
}
