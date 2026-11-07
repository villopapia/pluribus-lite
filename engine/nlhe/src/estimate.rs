//! Blueprint memory estimator.
//!
//! Infoset = (betting history, current-street card bucket), i.e. imperfect recall of earlier
//! buckets, as in Pluribus. So per street s:
//!   infosets_s = betting_nodes_s * buckets_s
//!   pairs_s    = sum over nodes of legal actions * buckets_s     ((infoset, action) pairs)
//! Bytes:
//!   regrets        pairs_s * regret_bytes                       (all streets)
//!   avg strategy   pairs_s * avg_bytes  only on `avg_strategy_streets` (Pluribus: preflop)
//!   node index     betting_nodes * index_bytes                  (offset into the flat arrays)
//!   bucket tables  direct-indexed (hole, board) entries * (1 or 2 bytes) for flop and turn
//!                  (see index.rs); the river is computed on the fly from equity cut-offs
//!                  (0 bytes) or tabled, per the config's `river_method`.
//! Not included: OS, the rest of the process, per-thread scratch (small), and the
//! one-off memory needed to *build* the abstraction.

use crate::betting::TreeCounts;
use crate::config::{AbstractionConfig, RiverMethod};
use crate::index::ENTRIES;

/// Suit-isomorphic canonical (hole, board) counts per street (Waugh 2013), for reference only;
/// the tables use direct indexing (`index::ENTRIES`).
pub const CANONICAL_HANDS: [u64; 4] = [169, 1_286_792, 55_190_538, 2_428_287_420];

pub const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ByteModel {
    pub regret_bytes: u64,
    pub avg_bytes: u64,
    pub index_bytes: u64,
}

impl Default for ByteModel {
    fn default() -> Self {
        // i32 regrets and u32 average-strategy counts, as Pluribus stored them.
        ByteModel { regret_bytes: 4, avg_bytes: 4, index_bytes: 8 }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct StreetEstimate {
    pub nodes: u64,
    pub actions: u64,
    pub buckets: u64,
    pub infosets: u64,
    pub pairs: u64,
    pub regret_bytes: u64,
    pub avg_bytes: u64,
    pub table_bytes: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Estimate {
    pub streets: [StreetEstimate; 4],
    pub index_bytes: u64,
    pub total_bytes: u64,
}

impl Estimate {
    pub fn total_gib(&self) -> f64 {
        self.total_bytes as f64 / GIB
    }
    pub fn infosets(&self) -> u64 {
        self.streets.iter().map(|s| s.infosets).sum()
    }
    pub fn pairs(&self) -> u64 {
        self.streets.iter().map(|s| s.pairs).sum()
    }
}

fn bucket_id_bytes(buckets: u64) -> u64 {
    if buckets <= 256 {
        1
    } else if buckets <= 65_536 {
        2
    } else {
        4
    }
}

pub fn estimate(counts: &TreeCounts, cfg: &AbstractionConfig, m: &ByteModel) -> Estimate {
    let mk = |s: usize| {
        let b = cfg.buckets[s] as u64;
        let pairs = counts.actions[s] * b;
        let table_bytes = match s {
            0 => 0,
            3 if cfg.river_method == RiverMethod::OnTheFly => 0,
            _ => ENTRIES[s] * bucket_id_bytes(b),
        };
        StreetEstimate {
            nodes: counts.nodes[s],
            actions: counts.actions[s],
            buckets: b,
            infosets: counts.nodes[s] * b,
            pairs,
            regret_bytes: pairs * m.regret_bytes,
            avg_bytes: if cfg.avg_strategy_streets[s] { pairs * m.avg_bytes } else { 0 },
            table_bytes,
        }
    };
    let streets = [mk(0), mk(1), mk(2), mk(3)];
    let index_bytes = counts.nodes.iter().sum::<u64>() * m.index_bytes;
    let total_bytes = index_bytes
        + streets.iter().map(|s| s.regret_bytes + s.avg_bytes + s.table_bytes).sum::<u64>();
    Estimate { streets, index_bytes, total_bytes }
}

/// Largest equal post-flop bucket count B (flop = turn = river = B, preflop fixed) whose
/// estimate fits in `budget_bytes`. Returns None if even B = 1 does not fit.
pub fn max_equal_buckets(
    counts: &TreeCounts,
    cfg: &AbstractionConfig,
    m: &ByteModel,
    budget_bytes: u64,
) -> Option<u32> {
    let fits = |b: u32| {
        let mut c = cfg.clone();
        c.buckets = [cfg.buckets[0], b, b, b];
        estimate(counts, &c, m).total_bytes <= budget_bytes
    };
    if !fits(1) {
        return None;
    }
    let (mut lo, mut hi) = (1u32, 1u32 << 20);
    while lo < hi {
        let mid = lo + (hi - lo + 1) / 2;
        if fits(mid) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    Some(lo)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn counts() -> TreeCounts {
        TreeCounts { nodes: [10, 100, 1_000, 10_000], actions: [30, 250, 2_500, 25_000], terminals: 0 }
    }

    fn cfg(b: [u32; 4]) -> AbstractionConfig {
        let mut c = AbstractionConfig::parse("").unwrap();
        c.buckets = b;
        c
    }

    #[test]
    fn hand_computed_estimate() {
        let e = estimate(&counts(), &cfg([169, 200, 200, 200]), &ByteModel::default());
        assert_eq!(e.streets[0].infosets, 10 * 169);
        assert_eq!(e.streets[0].pairs, 30 * 169);
        assert_eq!(e.streets[0].regret_bytes, 30 * 169 * 4);
        assert_eq!(e.streets[0].avg_bytes, 30 * 169 * 4); // preflop average strategy
        assert_eq!(e.streets[1].avg_bytes, 0);
        assert_eq!(e.streets[3].pairs, 25_000 * 200);
        assert_eq!(e.streets[1].table_bytes, 25_989_600); // 1 byte per (hole, flop)
        assert_eq!(e.streets[2].table_bytes, 305_377_800);
        assert_eq!(e.streets[3].table_bytes, 0); // river on the fly
        assert_eq!(e.index_bytes, 11_110 * 8);
        let expected = 30 * 169 * 8 + (250 + 2_500 + 25_000) * 200 * 4 + 25_989_600 + 305_377_800 + 11_110 * 8;
        assert_eq!(e.total_bytes, expected);
    }

    #[test]
    fn bucket_ids_widen_above_256_and_river_table_option() {
        let mut c = cfg([169, 300, 300, 300]);
        c.river_method = RiverMethod::PrecomputedTable;
        let e = estimate(&counts(), &c, &ByteModel::default());
        assert_eq!(e.streets[1].table_bytes, 2 * 25_989_600);
        assert_eq!(e.streets[3].table_bytes, 2 * 2_809_475_760);
        c.buckets = [169, 256, 256, 256];
        let e = estimate(&counts(), &c, &ByteModel::default());
        assert_eq!(e.streets[3].table_bytes, 2_809_475_760); // 2.62 GiB at u8
    }

    #[test]
    fn max_equal_buckets_is_tight() {
        let c = cfg([169, 1, 1, 1]);
        let m = ByteModel::default();
        let budget = 1_000_000_000; // above the fixed ~331 MB of flop+turn tables
        let b = max_equal_buckets(&counts(), &c, &m, budget).unwrap();
        let at = |b| estimate(&counts(), &cfg([169, b, b, b]), &m).total_bytes;
        assert!(at(b) <= budget);
        assert!(at(b + 1) > budget);
        assert_eq!(max_equal_buckets(&counts(), &c, &m, 10), None);
    }
}
