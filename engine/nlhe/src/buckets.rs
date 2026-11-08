//! Card abstraction sources used by the blueprint trainer.
//!
//! * [`preflop_class`]: the 169 lossless preflop classes.
//! * [`TableAbstraction`]: the built M2 abstraction (flop/turn tables, river OCHS k-means), with
//!   optional coarsening to fewer buckets. Buckets are sorted by mean equity, so coarsening
//!   (bucket * target / fine) merges neighbouring buckets.
//! * [`StrengthAbstraction`]: table-free hand-strength classes, used by tests and quick runs.

use crate::abstraction::{centroids_from_bytes, read_abs_file, river_bucket};
use crate::kmeans::{KMeans, Metric};
use crate::ochs::N_CLUSTERS;
use crate::eval_lut::{LHand, Lut, N_CLASSES};
use crate::index::hand_index;
use crate::river_table::{RiverTable, RiverTableMode};
use std::io;
use std::path::Path;

/// 169 preflop classes: 0..13 pairs (deuces..aces), 13..91 suited, 91..169 offsuit
/// (non-pairs indexed by (high, low) with high > low).
#[inline]
pub fn preflop_class(a: u8, b: u8) -> u16 {
    let (ra, rb) = (a / 4, b / 4);
    let (hi, lo) = if ra >= rb { (ra, rb) } else { (rb, ra) };
    if hi == lo {
        return hi as u16;
    }
    let tri = (hi as u16 * (hi as u16 - 1)) / 2 + lo as u16;
    if a % 4 == b % 4 {
        13 + tri
    } else {
        91 + tri
    }
}

/// Which river feature the trainer must compute for `CardAbstraction::river`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RiverFeature {
    /// none (the abstraction looks only at the cards)
    None,
    /// [equity against one random hand] (legacy 1-D river)
    Equity,
    /// OCHS: 8 equities against Johanson et al.'s opponent clusters
    Ochs,
}

pub trait CardAbstraction: Sync + Send {
    /// Buckets per street (preflop is always 169).
    fn buckets(&self) -> [u32; 4];
    fn flop(&self, hole: (u8, u8), board: &[u8; 3]) -> u16;
    fn turn(&self, hole: (u8, u8), board: &[u8; 4]) -> u16;
    /// River bucket from the feature named by `river_feature()`.
    fn river(&self, hole: (u8, u8), board: &[u8; 5], feature: &[f32]) -> u16;
    fn river_feature(&self) -> RiverFeature;
    /// Precomputed river table (fine buckets), if loaded; the trainer then skips the feature.
    fn river_table(&self) -> Option<&RiverTable> {
        None
    }
    /// Map a fine river bucket from the table to the trained bucket count.
    fn river_from_fine(&self, fine: usize) -> u16 {
        fine as u16
    }
}

enum RiverModel {
    Ochs(KMeans),
    Cutoffs(Vec<f32>),
}

pub struct TableAbstraction {
    flop: Vec<u8>,
    turn: Vec<u8>,
    river: RiverModel,
    river_table: Option<RiverTable>,
    fine: [u32; 4],
    target: [u32; 4],
}

impl TableAbstraction {
    /// Load `<dir>/{flop.bkt, turn.bkt}` and the river model: `river.ochs` (OCHS k-means
    /// centroids, preferred) or the legacy `river.cut` (1-D equity cut-offs). Post-flop
    /// streets are coarsened to `target` buckets (<= the built count). u8 tables only.
    pub fn load(dir: &Path, target: u32) -> io::Result<Self> {
        let bad = |m: String| io::Error::new(io::ErrorKind::InvalidData, m);
        let (fh, flop) = read_abs_file(&dir.join("flop.bkt"))?;
        let (th, turn) = read_abs_file(&dir.join("turn.bkt"))?;
        if fh.width != 1 || th.width != 1 {
            return Err(bad("only u8 bucket tables are supported".into()));
        }
        let (river, rk) = if dir.join("river.ochs").exists() {
            let (rh, bytes) = read_abs_file(&dir.join("river.ochs"))?;
            let c = centroids_from_bytes(&bytes);
            let km = KMeans::from_centroids(c, N_CLUSTERS, Metric::L2);
            if km.k != rh.k as usize {
                return Err(bad("river.ochs centroid count mismatch".into()));
            }
            let k = km.k as u32;
            (RiverModel::Ochs(km), k)
        } else {
            let (_, bytes) = read_abs_file(&dir.join("river.cut"))?;
            let cuts = centroids_from_bytes(&bytes);
            let k = cuts.len() as u32 + 1;
            (RiverModel::Cutoffs(cuts), k)
        };
        let fine = [169, fh.k, th.k, rk];
        if target == 0 || fine[1..].iter().any(|&k| target > k) {
            return Err(bad(format!("target {target} must be in 1..=built buckets {fine:?}")));
        }
        Ok(TableAbstraction { flop, turn, river, river_table: None, fine, target: [169, target, target, target] })
    }

    /// Load `<dir>/river.tbl` (built by `build_abstraction --stages river_table` from this
    /// directory's `river.ochs`); afterwards river buckets come from the table.
    pub fn load_river_table(&mut self, dir: &Path, mode: RiverTableMode, verify: bool) -> io::Result<()> {
        let km = match &self.river {
            RiverModel::Ochs(km) => km,
            RiverModel::Cutoffs(_) => {
                return Err(io::Error::new(io::ErrorKind::InvalidInput, "river table needs an OCHS river model"))
            }
        };
        self.river_table = Some(RiverTable::open(&dir.join("river.tbl"), km, mode, verify)?);
        Ok(())
    }

    /// The fine OCHS model (for building the table).
    pub fn river_kmeans(&self) -> Option<&KMeans> {
        match &self.river {
            RiverModel::Ochs(km) => Some(km),
            RiverModel::Cutoffs(_) => None,
        }
    }

    #[inline]
    fn coarse(&self, street: usize, b: usize) -> u16 {
        (b as u64 * self.target[street] as u64 / self.fine[street] as u64) as u16
    }
}

impl CardAbstraction for TableAbstraction {
    fn buckets(&self) -> [u32; 4] {
        self.target
    }
    #[inline]
    fn flop(&self, hole: (u8, u8), board: &[u8; 3]) -> u16 {
        self.coarse(1, self.flop[hand_index(hole, board) as usize] as usize)
    }
    #[inline]
    fn turn(&self, hole: (u8, u8), board: &[u8; 4]) -> u16 {
        self.coarse(2, self.turn[hand_index(hole, board) as usize] as usize)
    }
    #[inline]
    fn river(&self, _hole: (u8, u8), _board: &[u8; 5], feature: &[f32]) -> u16 {
        let fine = match &self.river {
            RiverModel::Ochs(km) => km.nearest(feature).0,
            RiverModel::Cutoffs(c) => river_bucket(c, feature[0]),
        };
        self.coarse(3, fine)
    }
    fn river_table(&self) -> Option<&RiverTable> {
        self.river_table.as_ref()
    }
    #[inline]
    fn river_from_fine(&self, fine: usize) -> u16 {
        self.coarse(3, fine)
    }
    fn river_feature(&self) -> RiverFeature {
        match self.river {
            RiverModel::Ochs(_) => RiverFeature::Ochs,
            RiverModel::Cutoffs(_) => RiverFeature::Equity,
        }
    }
}

/// Hand-strength classes: bucket = dense class of the best hand so far * k / 7462.
/// Ignores draws/potential; meant for tests and smoke runs only.
pub struct StrengthAbstraction {
    pub k: u32,
}

impl StrengthAbstraction {
    #[inline]
    fn of(&self, cards: &[u8]) -> u16 {
        let c = Lut::get().class(LHand::of(cards)) as u64;
        (c * self.k as u64 / N_CLASSES as u64) as u16
    }
}

impl CardAbstraction for StrengthAbstraction {
    fn buckets(&self) -> [u32; 4] {
        [169, self.k, self.k, self.k]
    }
    fn flop(&self, h: (u8, u8), b: &[u8; 3]) -> u16 {
        self.of(&[h.0, h.1, b[0], b[1], b[2]])
    }
    fn turn(&self, h: (u8, u8), b: &[u8; 4]) -> u16 {
        self.of(&[h.0, h.1, b[0], b[1], b[2], b[3]])
    }
    fn river(&self, h: (u8, u8), b: &[u8; 5], _f: &[f32]) -> u16 {
        self.of(&[h.0, h.1, b[0], b[1], b[2], b[3], b[4]])
    }
    fn river_feature(&self) -> RiverFeature {
        RiverFeature::None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::equity::combo_cards;

    #[test]
    fn preflop_classes_are_169_with_right_multiplicities() {
        let mut count = [0u32; 169];
        for (a, b) in combo_cards() {
            count[preflop_class(a, b) as usize] += 1;
            assert_eq!(preflop_class(a, b), preflop_class(b, a));
        }
        assert!(count[..13].iter().all(|&c| c == 6)); // pairs
        assert!(count[13..91].iter().all(|&c| c == 4)); // suited
        assert!(count[91..].iter().all(|&c| c == 12)); // offsuit
        // spot checks: AA = 12, AKs and AKo differ
        assert_eq!(preflop_class(48, 49), 12);
        assert_ne!(preflop_class(48, 44), preflop_class(48, 45));
    }

    #[test]
    fn strength_abstraction_is_monotone_in_class() {
        let s = StrengthAbstraction { k: 8 };
        let royal = s.river((48, 44), &[40, 36, 32, 1, 2], &[]); // As Ks Qs Js Ts (spades = suit 0 here)
        let junk = s.river((0, 5), &[10, 23, 33, 46, 15], &[]);
        assert!(royal > junk);
        assert!(royal < 8);
    }
}
