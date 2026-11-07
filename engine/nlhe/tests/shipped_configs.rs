//! The shipped M2 configs parse, count deterministically, and fit their memory budgets.

use nlhe::betting::{Rules, TreeCounter};
use nlhe::config::AbstractionConfig;
use nlhe::estimate::{estimate, ByteModel};

fn load(name: &str) -> AbstractionConfig {
    let path = format!("{}/configs/{name}", env!("CARGO_MANIFEST_DIR"));
    AbstractionConfig::parse(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn check(name: &str, budget_gib: f64, expected_nodes: u64, expected_pairs: u64) {
    let cfg = load(name);
    let counts = TreeCounter::new(Rules::default(), &cfg).count().unwrap();
    let e = estimate(&counts, &cfg, &ByteModel::default());
    assert_eq!(counts.nodes.iter().sum::<u64>(), expected_nodes, "{name} nodes");
    assert_eq!(e.pairs(), expected_pairs, "{name} pairs");
    assert!(e.total_gib() <= budget_gib, "{name}: {:.2} GiB > {budget_gib}", e.total_gib());
    assert!(e.total_gib() >= 0.85 * budget_gib, "{name}: {:.2} GiB wastes the budget", e.total_gib());
}

#[test]
fn m2_11g_fits_11_gib() {
    check("m2_11g.cfg", 11.0, 4_426_184, 2_560_302_099);
}

#[test]
fn m2_26g_fits_26_gib() {
    check("m2_26g.cfg", 26.0, 10_816_246, 6_632_088_845);
}
