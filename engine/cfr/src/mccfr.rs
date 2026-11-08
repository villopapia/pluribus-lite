//! External-sampling MCCFR with linear discounting and negative-regret pruning,
//! after Algorithm 1 ("MCCFR with Negative-Regret Pruning") of the Pluribus
//! supplementary materials (Brown & Sandholm, Science 2019).
//!
//! Mapping to the pseudocode:
//! * `iterate`            = one pass of the `for t = 1..T` loop body
//! * `update_strategy`    = UPDATE-STRATEGY (average strategy phi, rounds <= `avg_max_round`)
//! * `traverse`           = TRAVERSE-MCCFR
//! * `traverse_pruned`    = TRAVERSE-MCCFR-P
//! * discount step        = `if t < LCFR_Threshold and t mod Discount_Interval = 0`
//!
//! Differences from the paper (all deliberate, all configurable):
//! * Pluribus measures thresholds in wall-clock minutes; here they are iterations.
//! * Regrets are f64, not integers.
//! * Pluribus only accumulates phi on the preflop (`avg_max_round = 0`); for the small
//!   test games we average every round.
//! * The "player folded / not in hand" branch is not needed for two-player games.

use crate::checkpoint::Checkpoint;
use crate::game::{Game, NodeKind, MAX_ACTIONS};
use crate::rng::Rng;
use crate::table::{Table, TabularPolicy};
use std::io;
use std::path::Path;

#[derive(Clone, Debug, PartialEq)]
pub struct MccfrConfig {
    pub seed: u64,
    /// Strategy_Interval: run UPDATE-STRATEGY every this many iterations.
    pub strategy_interval: u64,
    /// PruneThreshold: pruning is used only when t > prune_threshold.
    pub prune_threshold: u64,
    /// Probability of a non-pruned traversal once pruning is active (0.05 in Pluribus).
    pub explore_prob: f64,
    /// C: actions with regret <= C are skipped in pruned traversals.
    pub prune_regret_c: f64,
    /// LCFR_Threshold: discounting happens only while t < lcfr_threshold.
    pub lcfr_threshold: u64,
    /// Discount_Interval.
    pub discount_interval: u64,
    /// Accumulate phi only at rounds <= this (Pluribus: 0 = preflop).
    pub avg_max_round: u8,
    /// NOT in the paper. In pruned traversals, always explore the action with the
    /// highest regret even if it is <= C, so a node can never have every action pruned.
    pub always_explore_best: bool,
    /// Regret floor (paper: -310,000,000 for every action). `None` = no floor.
    pub regret_floor: Option<f64>,
    /// Paper (supplement pp. 3, 14): pruning never skips actions on the final betting round
    /// or actions that immediately lead to a terminal node. Default true (paper-faithful).
    pub prune_skip_final_round_and_terminal: bool,
}

impl MccfrConfig {
    /// Pluribus' schedule scaled to an iteration budget. Pluribus ran ~8 days
    /// (11,520 min) with PruneThreshold 200 min, LCFR_Threshold 400 min and
    /// Discount_Interval 10 min; we keep the same fractions of the run. C = -300M
    /// chips with 10,000-chip stacks is scaled by the game's max |utility|.
    pub fn pluribus_scaled(total_iters: u64, max_abs_utility: f64, seed: u64) -> Self {
        let frac = |minutes: f64| ((total_iters as f64 * minutes / 11_520.0).round() as u64).max(1);
        MccfrConfig {
            seed,
            strategy_interval: 1,
            prune_threshold: frac(200.0),
            explore_prob: 0.05,
            prune_regret_c: -300_000_000.0 * max_abs_utility / 10_000.0,
            lcfr_threshold: frac(400.0),
            discount_interval: frac(10.0),
            // Paper: average strategy on the first betting round only. For the two-player
            // toy games we average every round: with round-0-only averaging and the current
            // strategy elsewhere, Leduc stays ~1 chip exploitable (see the test
            // `paper_avg_rule_on_leduc_needs_postflop_averaging`). Pluribus averaged post-flop
            // snapshots instead (implemented in M3). Set `avg_max_round = 0` for the literal rule.
            avg_max_round: u8::MAX,
            always_explore_best: false,
            regret_floor: Some(-310_000_000.0 * max_abs_utility / 10_000.0),
            prune_skip_final_round_and_terminal: true,
        }
    }

    fn to_params(&self) -> Vec<f64> {
        vec![
            self.seed as f64,
            self.strategy_interval as f64,
            self.prune_threshold as f64,
            self.explore_prob,
            self.prune_regret_c,
            self.lcfr_threshold as f64,
            self.discount_interval as f64,
            self.avg_max_round as f64,
            self.always_explore_best as u8 as f64,
            self.regret_floor.unwrap_or(f64::NAN),
            self.prune_skip_final_round_and_terminal as u8 as f64,
        ]
    }

    fn from_params(p: &[f64]) -> Self {
        MccfrConfig {
            seed: p[0] as u64,
            strategy_interval: p[1] as u64,
            prune_threshold: p[2] as u64,
            explore_prob: p[3],
            prune_regret_c: p[4],
            lcfr_threshold: p[5] as u64,
            discount_interval: p[6] as u64,
            avg_max_round: p[7] as u8,
            // absent in checkpoints written before the flag existed
            always_explore_best: p.get(8).is_some_and(|&x| x != 0.0),
            // absent in older checkpoints: they ran without floor and without the exception
            regret_floor: p.get(9).copied().filter(|x| !x.is_nan()),
            prune_skip_final_round_and_terminal: p.get(10).is_some_and(|&x| x != 0.0),
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct MccfrStats {
    pub pruned_traversals: u64,
    pub full_traversals: u64,
    /// Actions skipped by negative-regret pruning.
    pub pruned_actions: u64,
    /// Traverser decision nodes visited during pruned traversals.
    pub pruned_traversal_nodes: u64,
    /// Of those, nodes where every action had regret <= C. Without
    /// `always_explore_best` nothing below them is explored and no regret is updated.
    pub all_pruned_nodes: u64,
}

pub struct Mccfr<G: Game> {
    pub game: G,
    pub cfg: MccfrConfig,
    pub table: Table,
    pub t: u64,
    pub rng: Rng,
    pub stats: MccfrStats,
}

impl<G: Game> Mccfr<G> {
    pub fn new(game: G, cfg: MccfrConfig) -> Self {
        let rng = Rng::new(cfg.seed);
        Mccfr { game, cfg, table: Table::new(), t: 0, rng, stats: MccfrStats::default() }
    }

    pub fn iterate(&mut self) {
        self.t += 1;
        let t = self.t;
        let root = self.game.root();
        for p in 0..self.game.num_players() {
            if t % self.cfg.strategy_interval == 0 {
                self.update_strategy(&root, p);
            }
            if t > self.cfg.prune_threshold {
                let q = self.rng.next_f64();
                if q < self.cfg.explore_prob {
                    self.stats.full_traversals += 1;
                    self.traverse(&root, p);
                } else {
                    self.stats.pruned_traversals += 1;
                    self.traverse_pruned(&root, p);
                }
            } else {
                self.stats.full_traversals += 1;
                self.traverse(&root, p);
            }
        }
        if t < self.cfg.lcfr_threshold && t % self.cfg.discount_interval == 0 {
            let k = (t / self.cfg.discount_interval) as f64;
            let d = k / (k + 1.0);
            for inf in self.table.map.values_mut() {
                for r in inf.regret.iter_mut() {
                    *r *= d;
                }
                for s in inf.strat_sum.iter_mut() {
                    *s *= d;
                }
            }
        }
    }

    pub fn run(&mut self, iterations: u64) {
        for _ in 0..iterations {
            self.iterate();
        }
    }

    pub fn average_policy(&self) -> TabularPolicy {
        self.table.average_policy()
    }

    pub fn current_policy(&self) -> TabularPolicy {
        self.table.current_policy()
    }

    fn update_strategy(&mut self, s: &G::State, i: usize) {
        match self.game.kind(s) {
            NodeKind::Terminal => {}
            _ if self.game.round(s) > self.cfg.avg_max_round => {}
            NodeKind::Chance => {
                let a = self.game.sample_chance(s, &mut self.rng);
                self.update_strategy(&self.game.apply(s, a), i);
            }
            NodeKind::Player(p) => {
                let acts = self.game.legal_actions(s);
                let n = acts.len();
                if p == i {
                    let key = self.game.infoset_key(s);
                    let mut sigma = [0.0; MAX_ACTIONS];
                    self.table.strategy(key, n, &mut sigma);
                    let k = self.rng.sample_index(&sigma[..n]);
                    self.table.get_or_insert(key, n).strat_sum[k] += 1.0;
                    self.update_strategy(&self.game.apply(s, acts.as_slice()[k]), i);
                } else {
                    for &a in acts.as_slice() {
                        self.update_strategy(&self.game.apply(s, a), i);
                    }
                }
            }
        }
    }

    fn traverse(&mut self, s: &G::State, i: usize) -> f64 {
        match self.game.kind(s) {
            NodeKind::Terminal => self.game.utility(s, i),
            NodeKind::Chance => {
                let a = self.game.sample_chance(s, &mut self.rng);
                self.traverse(&self.game.apply(s, a), i)
            }
            NodeKind::Player(p) => {
                let key = self.game.infoset_key(s);
                let acts = self.game.legal_actions(s);
                let n = acts.len();
                let mut sigma = [0.0; MAX_ACTIONS];
                self.table.strategy(key, n, &mut sigma);
                if p == i {
                    let mut va = [0.0; MAX_ACTIONS];
                    let mut v = 0.0;
                    for k in 0..n {
                        va[k] = self.traverse(&self.game.apply(s, acts.as_slice()[k]), i);
                        v += sigma[k] * va[k];
                    }
                    let floor = self.cfg.regret_floor;
                    let inf = self.table.get_or_insert(key, n);
                    for k in 0..n {
                        inf.regret[k] += va[k] - v;
                        if let Some(f) = floor {
                            inf.regret[k] = inf.regret[k].max(f);
                        }
                    }
                    v
                } else {
                    let k = self.rng.sample_index(&sigma[..n]);
                    self.traverse(&self.game.apply(s, acts.as_slice()[k]), i)
                }
            }
        }
    }

    fn traverse_pruned(&mut self, s: &G::State, i: usize) -> f64 {
        match self.game.kind(s) {
            NodeKind::Terminal => self.game.utility(s, i),
            NodeKind::Chance => {
                let a = self.game.sample_chance(s, &mut self.rng);
                self.traverse_pruned(&self.game.apply(s, a), i)
            }
            NodeKind::Player(p) => {
                let key = self.game.infoset_key(s);
                let acts = self.game.legal_actions(s);
                let n = acts.len();
                let mut sigma = [0.0; MAX_ACTIONS];
                self.table.strategy(key, n, &mut sigma);
                if p == i {
                    let mut regret = [0.0; MAX_ACTIONS];
                    regret[..n].copy_from_slice(&self.table.map[&key].regret);
                    let mut va = [0.0; MAX_ACTIONS];
                    let mut explored = [false; MAX_ACTIONS];
                    let mut v = 0.0;
                    self.stats.pruned_traversal_nodes += 1;
                    let c = self.cfg.prune_regret_c;
                    if regret[..n].iter().all(|&r| r <= c) {
                        self.stats.all_pruned_nodes += 1;
                    }
                    // first index of the maximum regret (deterministic tie-break)
                    let best = (0..n).fold(0, |b, k| if regret[k] > regret[b] { k } else { b });
                    let final_round = self.game.round(s) + 1 == self.game.num_rounds();
                    for k in 0..n {
                        let child = self.game.apply(s, acts.as_slice()[k]);
                        let exempt = self.cfg.prune_skip_final_round_and_terminal
                            && (final_round || self.game.kind(&child) == NodeKind::Terminal);
                        if exempt || regret[k] > c || (self.cfg.always_explore_best && k == best) {
                            va[k] = self.traverse_pruned(&child, i);
                            explored[k] = true;
                            v += sigma[k] * va[k];
                        } else {
                            self.stats.pruned_actions += 1;
                        }
                    }
                    let floor = self.cfg.regret_floor;
                    let inf = self.table.get_or_insert(key, n);
                    for k in 0..n {
                        if explored[k] {
                            inf.regret[k] += va[k] - v;
                            if let Some(f) = floor {
                                inf.regret[k] = inf.regret[k].max(f);
                            }
                        }
                    }
                    v
                } else {
                    let k = self.rng.sample_index(&sigma[..n]);
                    self.traverse_pruned(&self.game.apply(s, acts.as_slice()[k]), i)
                }
            }
        }
    }

    pub fn save_checkpoint(&self, path: &Path) -> io::Result<()> {
        Checkpoint {
            game: self.game.name().into(),
            algo: "mccfr".into(),
            iteration: self.t,
            rng_state: self.rng.state(),
            params: self.cfg.to_params(),
            table: self.table.clone(),
        }
        .save(path)
    }

    pub fn load_checkpoint(game: G, path: &Path) -> io::Result<Self> {
        let c = Checkpoint::load(path)?;
        if c.game != game.name() || c.algo != "mccfr" {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("checkpoint is {}/{}", c.game, c.algo),
            ));
        }
        Ok(Mccfr {
            game,
            cfg: MccfrConfig::from_params(&c.params),
            table: c.table,
            t: c.iteration,
            rng: Rng::from_state(c.rng_state),
            stats: MccfrStats::default(),
        })
    }

    /// Run until `total` iterations, checkpointing every `every`. If a checkpoint
    /// exists at `path` it is resumed (its stored config wins over `cfg`).
    pub fn run_resumable(
        game: G,
        cfg: MccfrConfig,
        total: u64,
        every: u64,
        path: &Path,
    ) -> io::Result<Self> {
        let mut s = if path.exists() {
            Self::load_checkpoint(game, path)?
        } else {
            Self::new(game, cfg)
        };
        while s.t < total {
            s.iterate();
            if s.t % every == 0 || s.t == total {
                s.save_checkpoint(path)?;
            }
        }
        Ok(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::br::{expected_values, exploitability};
    use crate::kuhn::Kuhn;
    use crate::leduc::Leduc;

    #[test]
    fn kuhn_converges_below_1e3_and_value_minus_1_18() {
        let total = 2_000_000;
        let cfg = MccfrConfig::pluribus_scaled(total, Kuhn.max_abs_utility(), 1);
        let mut s = Mccfr::new(Kuhn, cfg);
        s.run(total);
        let pol = s.average_policy();
        let e = exploitability(&Kuhn, &pol);
        let v = expected_values(&Kuhn, &pol)[0];
        assert!(e < 1e-3, "exploitability {e}");
        assert!((v + 1.0 / 18.0).abs() < 2e-3, "value {v}");
    }

    #[test]
    fn leduc_converges() {
        let total = 1_000_000;
        let cfg = MccfrConfig::pluribus_scaled(total, Leduc.max_abs_utility(), 1);
        let mut s = Mccfr::new(Leduc, cfg);
        s.run(total);
        let pol = s.average_policy();
        let e = exploitability(&Leduc, &pol);
        let v = expected_values(&Leduc, &pol)[0];
        assert!(e < 0.02, "exploitability {e}");
        assert!((v + 0.0856).abs() < 0.02, "value {v}");
    }

    #[test]
    fn aggressive_pruning_still_converges_and_prunes() {
        // C close to zero: pruning fires constantly; the 5% unpruned traversals
        // must still let the solver converge. Aggressive pruning slows convergence
        // (measured 0.007 at 1M vs 0.0013 with the scaled C), so the bar is looser.
        let total = 1_000_000;
        let mut cfg = MccfrConfig::pluribus_scaled(total, Kuhn.max_abs_utility(), 2);
        cfg.prune_regret_c = -5.0;
        cfg.prune_skip_final_round_and_terminal = false; // Kuhn has one round: test the mechanism
        let mut s = Mccfr::new(Kuhn, cfg);
        s.run(total / 10);
        let e_early = exploitability(&Kuhn, &s.average_policy());
        s.run(total - total / 10);
        assert!(s.stats.pruned_actions > 0);
        let e = exploitability(&Kuhn, &s.average_policy());
        assert!(e < 0.01 && e < e_early, "exploitability {e_early} -> {e}");
    }

    #[test]
    fn paper_pruning_exception_never_prunes_kuhn_and_spares_final_round_in_leduc() {
        // Kuhn: one betting round, so everything is "final round": nothing may be pruned.
        let total = 200_000;
        let mut cfg = MccfrConfig::pluribus_scaled(total, Kuhn.max_abs_utility(), 3);
        cfg.prune_regret_c = -5.0;
        assert!(cfg.prune_skip_final_round_and_terminal); // paper-faithful default
        let mut s = Mccfr::new(Kuhn, cfg);
        s.run(total);
        assert!(s.stats.pruned_traversals > 0);
        assert_eq!(s.stats.pruned_actions, 0);
        // Leduc: pruning still happens (round 0, non-terminal actions only)
        let mut cfg = MccfrConfig::pluribus_scaled(total, Leduc.max_abs_utility(), 3);
        cfg.prune_regret_c = -5.0;
        let mut l = Mccfr::new(Leduc, cfg.clone());
        l.run(total);
        let mut cfg_off = cfg;
        cfg_off.prune_skip_final_round_and_terminal = false;
        let mut l_off = Mccfr::new(Leduc, cfg_off);
        l_off.run(total);
        assert!(l.stats.pruned_actions > 0);
        assert!(l.stats.pruned_actions < l_off.stats.pruned_actions);
    }

    /// Documents why M1 averages all rounds: the literal paper rule (phi on round 0 only,
    /// current strategy elsewhere) leaves Leduc far more exploitable.
    #[test]
    fn paper_avg_rule_on_leduc_needs_postflop_averaging() {
        let total = 1_000_000;
        let all = MccfrConfig::pluribus_scaled(total, Leduc.max_abs_utility(), 1);
        let first_only = MccfrConfig { avg_max_round: 0, ..all.clone() };
        let mut a = Mccfr::new(Leduc, all);
        a.run(total);
        let mut b = Mccfr::new(Leduc, first_only);
        b.run(total);
        let ea = exploitability(&Leduc, &a.average_policy());
        let eb = exploitability(&Leduc, &b.average_policy());
        assert!(ea < 0.02, "all-round average {ea}");
        assert!(eb > 10.0 * ea, "round-0-only average {eb} vs all rounds {ea}");
    }

    #[test]
    fn regret_floor_is_respected() {
        let total = 300_000;
        let mut cfg = MccfrConfig::pluribus_scaled(total, Leduc.max_abs_utility(), 4);
        cfg.regret_floor = Some(-50.0);
        let mut s = Mccfr::new(Leduc, cfg);
        s.run(total);
        let min = s.table.map.values().flat_map(|i| i.regret.iter().copied()).fold(f64::INFINITY, f64::min);
        assert!(min >= -50.0, "min regret {min}");
        assert!(min <= -49.0, "floor never reached ({min}); test not exercising it");
    }

    #[test]
    fn same_seed_same_result_different_seed_differs() {
        let cfg = MccfrConfig::pluribus_scaled(5_000, 2.0, 7);
        let mut a = Mccfr::new(Kuhn, cfg.clone());
        let mut b = Mccfr::new(Kuhn, cfg.clone());
        a.run(5_000);
        b.run(5_000);
        assert_eq!(a.table, b.table);
        let mut c = Mccfr::new(Kuhn, MccfrConfig { seed: 8, ..cfg });
        c.run(5_000);
        assert_ne!(a.table, c.table);
    }

    #[test]
    fn discount_factor_schedule() {
        // With discount_interval 1 and only regret updates at t=1, after t=1 the
        // regrets are multiplied by 1/2 (k=1 -> d = 1/2).
        let cfg = MccfrConfig {
            seed: 0,
            strategy_interval: 1,
            prune_threshold: u64::MAX,
            explore_prob: 0.05,
            prune_regret_c: -1e9,
            lcfr_threshold: 2,
            discount_interval: 1,
            avg_max_round: u8::MAX,
            always_explore_best: false,
            regret_floor: None,
            prune_skip_final_round_and_terminal: false,
        };
        let mut s = Mccfr::new(Kuhn, cfg.clone());
        s.iterate();
        let mut undiscounted = Mccfr::new(Kuhn, MccfrConfig { lcfr_threshold: 0, ..cfg });
        undiscounted.iterate();
        for (k, inf) in &s.table.map {
            let u = &undiscounted.table.map[k];
            for j in 0..inf.regret.len() {
                assert!((inf.regret[j] - 0.5 * u.regret[j]).abs() < 1e-12);
                assert!((inf.strat_sum[j] - 0.5 * u.strat_sum[j]).abs() < 1e-12);
            }
        }
    }

    fn all_pruned_cfg(always_explore_best: bool) -> MccfrConfig {
        MccfrConfig {
            seed: 4,
            strategy_interval: 1,
            prune_threshold: 0,      // prune from t = 1
            explore_prob: 0.0,       // never take the unpruned branch
            prune_regret_c: 1e18,    // every regret is <= C: every action pruned
            lcfr_threshold: 0,
            discount_interval: 1,
            avg_max_round: u8::MAX,
            always_explore_best,
            regret_floor: None,
            prune_skip_final_round_and_terminal: false, // test the mechanism itself
        }
    }

    #[test]
    fn all_pruned_nodes_are_counted_and_frozen_without_flag() {
        let mut s = Mccfr::new(Kuhn, all_pruned_cfg(false));
        s.run(1_000);
        assert!(s.stats.pruned_traversal_nodes > 0);
        assert_eq!(s.stats.all_pruned_nodes, s.stats.pruned_traversal_nodes);
        // nothing is ever explored, so no regret ever moves
        assert!(s.table.map.values().all(|inf| inf.regret.iter().all(|&r| r == 0.0)));
    }

    #[test]
    fn always_explore_best_keeps_learning() {
        let mut s = Mccfr::new(Kuhn, all_pruned_cfg(true));
        s.run(1_000);
        assert_eq!(s.stats.all_pruned_nodes, s.stats.pruned_traversal_nodes);
        assert!(s.table.map.values().any(|inf| inf.regret.iter().any(|&r| r != 0.0)));
    }

    #[test]
    fn always_explore_best_converges_with_aggressive_pruning() {
        let total = 1_000_000;
        let mut cfg = MccfrConfig::pluribus_scaled(total, Kuhn.max_abs_utility(), 2);
        cfg.prune_regret_c = -5.0;
        cfg.always_explore_best = true;
        cfg.prune_skip_final_round_and_terminal = false;
        let mut s = Mccfr::new(Kuhn, cfg);
        s.run(total);
        let e = exploitability(&Kuhn, &s.average_policy());
        assert!(e < 0.01, "exploitability {e}");
    }

    #[test]
    fn old_checkpoint_params_default_flag_off() {
        let mut p = MccfrConfig::pluribus_scaled(100, 2.0, 1).to_params();
        p.truncate(8);
        assert!(!MccfrConfig::from_params(&p).always_explore_best);
    }

    #[test]
    fn checkpoint_resume_is_bit_identical() {
        let dir = std::env::temp_dir().join(format!("mccfr_ckpt_{}", std::process::id()));
        let path = dir.join("leduc_mccfr.ckpt");
        let _ = std::fs::remove_dir_all(&dir);
        let total = 20_000;
        let cfg = MccfrConfig::pluribus_scaled(total, Leduc.max_abs_utility(), 3);
        Mccfr::run_resumable(Leduc, cfg.clone(), 8_000, 1_000, &path).unwrap();
        let resumed = Mccfr::run_resumable(Leduc, cfg.clone(), total, 1_000, &path).unwrap();
        let mut straight = Mccfr::new(Leduc, cfg);
        straight.run(total);
        assert_eq!(resumed.t, total);
        assert_eq!(resumed.rng, straight.rng);
        assert_eq!(resumed.table, straight.table);
        std::fs::remove_dir_all(dir).ok();
    }
}
