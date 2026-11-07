//! Vanilla CFR (Zinkevich et al. 2007): full-tree traversal, alternating updates,
//! uniform averaging. One iteration = one traversal per player.

use crate::checkpoint::Checkpoint;
use crate::game::{Game, NodeKind, MAX_ACTIONS};
use crate::table::{Table, TabularPolicy};
use std::io;
use std::path::Path;

pub struct Cfr<G: Game> {
    pub game: G,
    pub table: Table,
    pub iteration: u64,
}

impl<G: Game> Cfr<G> {
    pub fn new(game: G) -> Self {
        Cfr { game, table: Table::new(), iteration: 0 }
    }

    pub fn iterate(&mut self) {
        let root = self.game.root();
        for p in 0..self.game.num_players() {
            self.traverse(&root, p, 1.0, 1.0);
        }
        self.iteration += 1;
    }

    pub fn run(&mut self, iterations: u64) {
        for _ in 0..iterations {
            self.iterate();
        }
    }

    pub fn average_policy(&self) -> TabularPolicy {
        self.table.average_policy()
    }

    /// Returns the counterfactual-weighted value for `i` (unnormalised by pi_o).
    fn traverse(&mut self, s: &G::State, i: usize, pi_i: f64, pi_o: f64) -> f64 {
        match self.game.kind(s) {
            NodeKind::Terminal => self.game.utility(s, i),
            NodeKind::Chance => {
                let c = self.game.chance_outcomes(s);
                let mut v = 0.0;
                for (&a, &p) in c.actions().iter().zip(c.probs()) {
                    v += p * self.traverse(&self.game.apply(s, a), i, pi_i, pi_o * p);
                }
                v
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
                        let child = self.game.apply(s, acts.as_slice()[k]);
                        va[k] = self.traverse(&child, i, pi_i * sigma[k], pi_o);
                        v += sigma[k] * va[k];
                    }
                    let inf = self.table.get_or_insert(key, n);
                    for k in 0..n {
                        inf.regret[k] += pi_o * (va[k] - v);
                        inf.strat_sum[k] += pi_i * sigma[k];
                    }
                    v
                } else {
                    let mut v = 0.0;
                    for k in 0..n {
                        let child = self.game.apply(s, acts.as_slice()[k]);
                        v += sigma[k] * self.traverse(&child, i, pi_i, pi_o * sigma[k]);
                    }
                    v
                }
            }
        }
    }

    pub fn save_checkpoint(&self, path: &Path) -> io::Result<()> {
        Checkpoint {
            game: self.game.name().into(),
            algo: "cfr".into(),
            iteration: self.iteration,
            rng_state: [0; 4],
            params: vec![],
            table: self.table.clone(),
        }
        .save(path)
    }

    pub fn load_checkpoint(game: G, path: &Path) -> io::Result<Self> {
        let c = Checkpoint::load(path)?;
        if c.game != game.name() || c.algo != "cfr" {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("checkpoint is {}/{}", c.game, c.algo),
            ));
        }
        Ok(Cfr { game, table: c.table, iteration: c.iteration })
    }

    /// Run until `total` iterations, checkpointing every `every`; resumes from `path`
    /// if a checkpoint exists there.
    pub fn run_resumable(game: G, total: u64, every: u64, path: &Path) -> io::Result<Self> {
        let mut s = if path.exists() {
            Self::load_checkpoint(game, path)?
        } else {
            Self::new(game)
        };
        while s.iteration < total {
            s.iterate();
            if s.iteration % every == 0 || s.iteration == total {
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
        // Measured: 0.0012 at 30k, 0.00068 at 100k (vanilla CFR ~ 1/sqrt(T)).
        let mut s = Cfr::new(Kuhn);
        s.run(100_000);
        let pol = s.average_policy();
        let e = exploitability(&Kuhn, &pol);
        let v = expected_values(&Kuhn, &pol)[0];
        assert!(e < 1e-3, "exploitability {e}");
        assert!((v + 1.0 / 18.0).abs() < 2e-3, "value {v}");
        assert_eq!(s.table.len(), 12);
    }

    #[test]
    fn leduc_converges() {
        // Measured: 0.0097 at 3k iterations, 0.0067 at 10k.
        let mut s = Cfr::new(Leduc);
        s.run(3_000);
        let pol = s.average_policy();
        let e = exploitability(&Leduc, &pol);
        let v = expected_values(&Leduc, &pol)[0];
        assert!(e < 0.012, "exploitability {e}");
        assert!((v + 0.0856).abs() < 0.01, "value {v}");
        assert_eq!(s.table.len(), 288);
    }

    #[test]
    fn exploitability_decreases() {
        let mut s = Cfr::new(Kuhn);
        s.run(10);
        let e1 = exploitability(&Kuhn, &s.average_policy());
        s.run(990);
        let e2 = exploitability(&Kuhn, &s.average_policy());
        assert!(e2 < e1, "{e1} -> {e2}");
    }

    #[test]
    fn checkpoint_resume_is_identical() {
        let dir = std::env::temp_dir().join(format!("cfr_ckpt_{}", std::process::id()));
        let path = dir.join("kuhn_cfr.ckpt");
        let _ = std::fs::remove_dir_all(&dir);
        // simulate an interrupted run: 300 iterations, then resume to 700
        Cfr::run_resumable(Kuhn, 300, 100, &path).unwrap();
        let resumed = Cfr::run_resumable(Kuhn, 700, 100, &path).unwrap();
        let mut straight = Cfr::new(Kuhn);
        straight.run(700);
        assert_eq!(resumed.iteration, 700);
        assert_eq!(resumed.table, straight.table);
        std::fs::remove_dir_all(dir).ok();
    }
}
