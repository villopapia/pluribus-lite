//! Count the abstract betting tree for each config and estimate blueprint memory.
//!
//! `cargo run --release --bin m2_estimate -- configs/a.cfg [configs/b.cfg ...]`

use nlhe::betting::{Rules, TreeCounter};
use nlhe::config::{AbstractionConfig, STREETS};
use nlhe::estimate::{estimate, max_equal_buckets, ByteModel, GIB, PLURIBUS_ACTION_SEQUENCES, PLURIBUS_ALLOC_FRACTION};
use std::time::Instant;

fn fmt(n: u64) -> String {
    let s = n.to_string();
    let mut out = String::new();
    for (i, ch) in s.chars().enumerate() {
        if i > 0 && (s.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(ch);
    }
    out
}

fn main() {
    let paths: Vec<String> = std::env::args().skip(1).collect();
    let m = ByteModel::default();
    for p in paths {
        let cfg = AbstractionConfig::parse(&std::fs::read_to_string(&p).expect("read config"))
            .unwrap_or_else(|e| panic!("{p}: {e}"));
        let t0 = Instant::now();
        let mut tc = TreeCounter::new(Rules::default(), &cfg);
        tc.evict_when_full = true; // exact either way; eviction only costs time
        let counts = match tc.count() {
            Ok(c) => c,
            Err(e) => {
                println!("\n## {} ({p}): memo limit hit at {} states, skipped", cfg.name, e.0);
                continue;
            }
        };
        let secs = t0.elapsed().as_secs_f64();
        let e = estimate(&counts, &cfg, &m);
        println!("\n## {} ({p})", cfg.name);
        println!(
            "counted in {secs:.1}s ({} terminal histories), memo {} states at the end, {} evictions\n",
            fmt(counts.terminals),
            fmt(tc.memo_len() as u64),
            tc.evictions
        );
        println!("| street | betting nodes | actions (sum) | avg actions | buckets | infosets | (infoset,action) pairs | regret GiB | avg-strategy GiB | bucket table MiB |");
        println!("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|");
        for (i, s) in e.streets.iter().enumerate() {
            println!(
                "| {} | {} | {} | {:.2} | {} | {} | {} | {:.3} | {:.3} | {:.1} |",
                STREETS[i],
                fmt(s.nodes),
                fmt(s.actions),
                s.actions as f64 / s.nodes.max(1) as f64,
                s.buckets,
                fmt(s.infosets),
                fmt(s.pairs),
                s.regret_bytes as f64 / GIB,
                s.avg_bytes as f64 / GIB,
                s.table_bytes as f64 / 1024.0 / 1024.0
            );
        }
        println!(
            "| **total** | {} | {} | | | {} | {} | | | node index {:.3} GiB |",
            fmt(counts.nodes.iter().sum()),
            fmt(counts.actions.iter().sum()),
            fmt(e.infosets()),
            fmt(e.pairs()),
            e.index_bytes as f64 / GIB
        );
        println!("\n**Dense estimate (regrets + preflop average + tables + node index): {:.2} GiB**", e.total_gib());
        let actions: u64 = counts.actions.iter().sum();
        println!(
            "- action sequences (history, action pairs): {} = {:.2}x Pluribus' {} (supplement p. 13)",
            fmt(actions),
            actions as f64 / PLURIBUS_ACTION_SEQUENCES as f64,
            fmt(PLURIBUS_ACTION_SEQUENCES)
        );
        let lazy = ByteModel { postflop_alloc_fraction: PLURIBUS_ALLOC_FRACTION, tree_node_bytes: 44, ..m };
        let el = estimate(&counts, &cfg, &lazy);
        println!(
            "- **first-use allocation** at Pluribus' encountered share {:.3}, plus the explicit tree ({} nodes x 44 B): **{:.2} GiB**; fits in 0.5 TB (465.66 GiB): {}",
            PLURIBUS_ALLOC_FRACTION,
            fmt(counts.nodes.iter().sum::<u64>() + counts.terminals),
            el.total_gib(),
            if el.total_gib() <= 500e9 / GIB { "yes" } else { "no" }
        );
        for budget in [11.0, 26.0] {
            match max_equal_buckets(&counts, &cfg, &lazy, (budget * GIB) as u64) {
                Some(b) => println!("- max equal post-flop buckets within {budget} GiB (first-use model): {b}"),
                None => println!("- does not fit in {budget} GiB (first-use model) even with 1 bucket"),
            }
        }
    }
}
