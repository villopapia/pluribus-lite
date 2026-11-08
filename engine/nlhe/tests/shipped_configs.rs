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

/// faithful.cfg encodes the paper's published structure (docs/fidelity_spec.md section 2).
#[test]
fn faithful_config_matches_the_paper_structure() {
    use nlhe::config::{Limp, RiverMethod};
    let c = load("faithful.cfg");
    assert_eq!(c.buckets, [169, 200, 200, 200]); // p. 13
    assert_eq!(c.limp, Limp::All); // fold and call always legal (p. 13)
    assert_eq!(c.crowded_players, 0);
    assert!(c.multiway.iter().all(|m| m.is_empty()));
    assert!(c.allin);
    assert_eq!(c.river_method, RiverMethod::OnTheFly);
    assert_eq!(c.avg_strategy_streets, [true, false, false, false]); // p. 15
    for street in [2, 3] {
        // p. 12: first raise {0.5x, 1x, all-in}; later raises {1x, all-in}, any player count
        for n in [2, 3, 6] {
            assert_eq!(c.sizes_for(street, 0, n), &[0.5, 1.0]);
            for depth in 1..8 {
                assert_eq!(c.sizes_for(street, depth, n), &[1.0]);
            }
        }
    }
}

/// The pilot keeps the faithful structure and fits 11 GiB with first-use allocation.
#[test]
fn pilot_fit11_fits_with_first_use_allocation() {
    use nlhe::estimate::PLURIBUS_ALLOC_FRACTION;
    let f = load("faithful.cfg");
    let p = load("pilot_dell_fit11.cfg");
    assert_eq!((p.limp, p.crowded_players, p.allin), (f.limp, f.crowded_players, f.allin));
    for street in [2, 3] {
        for depth in 0..6 {
            assert_eq!(p.sizes_for(street, depth, 6), f.sizes_for(street, depth, 6));
        }
    }
    let counts = TreeCounter::new(Rules::default(), &p).count().unwrap();
    let m = ByteModel { postflop_alloc_fraction: PLURIBUS_ALLOC_FRACTION, tree_node_bytes: 44, ..ByteModel::default() };
    let gib = estimate(&counts, &p, &m).total_gib();
    assert!(gib <= 11.0 && gib > 9.0, "{gib:.2} GiB");
}

/// D-SCHED-2: the Dell pilot has an explicit plan; the faithful config leaves it to the run.
#[test]
fn run_plan_is_explicit_for_the_dell_pilot() {
    assert_eq!(load("pilot_dell_fit11.cfg").planned_total_iterations, Some(83_518_000_000));
    assert_eq!(load("faithful.cfg").planned_total_iterations, None);
}
