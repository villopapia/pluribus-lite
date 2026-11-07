//! Pruning-threshold study: how C is scaled, how often it fires, how often a whole
//! node is pruned, and what `always_explore_best` changes.
//!
//! `cargo run --release --bin prune_study`

use cfr::br::exploitability;
use cfr::game::Game;
use cfr::kuhn::Kuhn;
use cfr::leduc::Leduc;
use cfr::mccfr::{Mccfr, MccfrConfig};
use std::time::Instant;

fn study<G: Game + Copy>(game: G, total: u64) {
    let base = MccfrConfig::pluribus_scaled(total, game.max_abs_utility(), 1);
    println!(
        "\n## {} ({} iterations, seed 1; prune after t > {}, LCFR until {}, discount every {})\n",
        game.name(),
        total,
        base.prune_threshold,
        base.lcfr_threshold,
        base.discount_interval
    );
    println!("| C | flag | pruned-traversal nodes | all-pruned nodes | share | pruned actions | min regret at end | exploitability | iter/s |");
    println!("|---:|---|---:|---:|---:|---:|---:|---:|---:|");
    let m = game.max_abs_utility();
    for c in [base.prune_regret_c, -100.0 * m, -2.5 * m] {
        for flag in [false, true] {
            let cfg = MccfrConfig { prune_regret_c: c, always_explore_best: flag, ..base.clone() };
            let mut s = Mccfr::new(game, cfg);
            let t0 = Instant::now();
            s.run(total);
            let secs = t0.elapsed().as_secs_f64();
            let min_r = s
                .table
                .map
                .values()
                .flat_map(|i| i.regret.iter().copied())
                .fold(f64::INFINITY, f64::min);
            let st = &s.stats;
            println!(
                "| {} | {} | {} | {} | {:.4}% | {} | {:.0} | {:.6} | {:.0} |",
                c,
                if flag { "on" } else { "off" },
                st.pruned_traversal_nodes,
                st.all_pruned_nodes,
                100.0 * st.all_pruned_nodes as f64 / st.pruned_traversal_nodes.max(1) as f64,
                st.pruned_actions,
                min_r,
                exploitability(&game, &s.average_policy()),
                total as f64 / secs
            );
        }
    }
}

fn main() {
    study(Kuhn, 2_000_000);
    study(Leduc, 2_000_000);
}
