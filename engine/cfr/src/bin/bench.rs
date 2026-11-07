//! Throughput and convergence benchmark for M1.
//!
//! `cargo run --release --bin bench [-- --quick]`
//!
//! For each (algorithm, game), trains for a fixed iteration budget, records
//! exploitability of the average strategy at log-spaced points (BR time excluded
//! from throughput), and prints iterations/second plus a markdown table.

use cfr::br::{expected_values, exploitability};
use cfr::cfr::Cfr;
use cfr::game::Game;
use cfr::kuhn::Kuhn;
use cfr::leduc::Leduc;
use cfr::mccfr::{Mccfr, MccfrConfig};
use std::time::{Duration, Instant};

struct Row {
    algo: &'static str,
    game: &'static str,
    iters: u64,
    train: Duration,
    expl: f64,
    value: f64,
    infosets: usize,
    curve: Vec<(u64, f64)>,
    note: String,
}

fn checkpoints(total: u64) -> Vec<u64> {
    let mut v = Vec::new();
    let mut x = 10u64;
    while x < total {
        v.push(x);
        v.push((x as f64 * 3.0) as u64);
        x *= 10;
    }
    v.retain(|&c| c < total);
    v.push(total);
    v
}

fn run_cfr<G: Game + Copy>(game: G, total: u64) -> Row {
    let mut s = Cfr::new(game);
    let mut train = Duration::ZERO;
    let mut curve = Vec::new();
    for c in checkpoints(total) {
        let t0 = Instant::now();
        s.run(c - s.iteration);
        train += t0.elapsed();
        curve.push((c, exploitability(&game, &s.average_policy())));
    }
    let pol = s.average_policy();
    Row {
        algo: "CFR (vanilla, alternating)",
        game: game.name(),
        iters: total,
        train,
        expl: curve.last().unwrap().1,
        value: expected_values(&game, &pol)[0],
        infosets: s.table.len(),
        curve,
        note: String::new(),
    }
}

fn run_mccfr<G: Game + Copy>(game: G, total: u64, seed: u64) -> Row {
    let cfg = MccfrConfig::pluribus_scaled(total, game.max_abs_utility(), seed);
    let note = format!(
        "seed {seed}; prune after {}, LCFR until {}, discount every {}, C = {}",
        cfg.prune_threshold, cfg.lcfr_threshold, cfg.discount_interval, cfg.prune_regret_c
    );
    let mut s = Mccfr::new(game, cfg);
    let mut train = Duration::ZERO;
    let mut curve = Vec::new();
    for c in checkpoints(total) {
        let t0 = Instant::now();
        s.run(c - s.t);
        train += t0.elapsed();
        curve.push((c, exploitability(&game, &s.average_policy())));
    }
    let pol = s.average_policy();
    let note = format!(
        "{note}; pruned traversals {}, full {}, pruned actions {}",
        s.stats.pruned_traversals, s.stats.full_traversals, s.stats.pruned_actions
    );
    Row {
        algo: "ES-MCCFR (LCFR + neg-regret pruning)",
        game: game.name(),
        iters: total,
        train,
        expl: curve.last().unwrap().1,
        value: expected_values(&game, &pol)[0],
        infosets: s.table.len(),
        curve,
        note,
    }
}

fn main() {
    let quick = std::env::args().any(|a| a == "--quick");
    let scale = if quick { 10 } else { 1 };
    let rows = vec![
        run_cfr(Kuhn, 100_000 / scale),
        run_cfr(Leduc, 10_000 / scale),
        run_mccfr(Kuhn, 10_000_000 / scale, 1),
        run_mccfr(Leduc, 10_000_000 / scale, 1),
    ];

    println!("| algorithm | game | iterations | train s | iter/s | infosets | exploitability | P0 value |");
    println!("|---|---|---:|---:|---:|---:|---:|---:|");
    for r in &rows {
        println!(
            "| {} | {} | {} | {:.2} | {:.0} | {} | {:.6} | {:.5} |",
            r.algo,
            r.game,
            r.iters,
            r.train.as_secs_f64(),
            r.iters as f64 / r.train.as_secs_f64(),
            r.infosets,
            r.expl,
            r.value
        );
    }
    for r in &rows {
        println!("\n{} on {}: {}", r.algo, r.game, r.note);
        for (i, e) in &r.curve {
            println!("  {i:>10}  {e:.6}");
        }
    }
}
