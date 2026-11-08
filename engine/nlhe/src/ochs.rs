//! Opponent Cluster Hand Strength (OCHS), Johanson, Burch, Valenzano & Bowling, AAMAS 2013
//! (cited by Pluribus as [26] for its information abstraction).
//!
//! The 169 starting hands are partitioned into 8 opponent clusters (Johanson et al. 2013,
//! Table 1, transcribed below). A hand's river OCHS feature is its probability of beating
//! (ties = 1/2) an opponent hand drawn uniformly from each cluster, given card removal.

use crate::equity::{combo_cards, COMBOS};

/// Johanson et al. 2013, Table 1. Row = first rank, column = second rank (2..A).
/// Diagonal = pairs; above the diagonal (row < column) = suited; below = offsuit.
/// Values are the published cluster labels 1..8.
pub const JOHANSON_TABLE1: [[u8; 13]; 13] = [
    [4, 1, 1, 1, 1, 1, 2, 2, 2, 4, 4, 4, 6], // 2
    [1, 6, 1, 1, 1, 1, 2, 2, 3, 4, 4, 6, 6], // 3
    [1, 1, 6, 1, 1, 2, 2, 2, 3, 4, 4, 6, 6], // 4
    [1, 1, 1, 6, 3, 3, 3, 3, 3, 4, 4, 6, 6], // 5
    [1, 1, 1, 1, 7, 3, 3, 3, 3, 4, 5, 6, 6], // 6
    [1, 1, 1, 2, 3, 7, 3, 3, 5, 5, 5, 6, 7], // 7
    [1, 1, 2, 2, 3, 3, 8, 3, 5, 5, 5, 6, 7], // 8
    [2, 2, 2, 2, 3, 3, 3, 8, 5, 5, 5, 7, 7], // 9
    [2, 2, 2, 2, 3, 3, 3, 5, 8, 5, 7, 7, 7], // T
    [2, 2, 4, 4, 4, 4, 5, 5, 5, 8, 7, 7, 7], // J
    [4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 8, 7, 7], // Q
    [4, 4, 4, 6, 6, 6, 6, 6, 7, 7, 7, 8, 7], // K
    [6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 8], // A
];

pub const N_CLUSTERS: usize = 8;

/// Opponent cluster (0..8) of a two-card hand.
#[inline]
pub fn cluster_of(a: u8, b: u8) -> u8 {
    let (ra, rb) = ((a / 4) as usize, (b / 4) as usize);
    let (hi, lo) = if ra >= rb { (ra, rb) } else { (rb, ra) };
    let v = if hi == lo {
        JOHANSON_TABLE1[hi][hi]
    } else if a % 4 == b % 4 {
        JOHANSON_TABLE1[lo][hi] // suited: above the diagonal
    } else {
        JOHANSON_TABLE1[hi][lo] // offsuit: below the diagonal
    };
    v - 1
}

/// Cluster of every combo, in `combo_cards()` order.
pub fn combo_clusters() -> Vec<u8> {
    combo_cards().iter().map(|&(a, b)| cluster_of(a, b)).collect()
}

/// OCHS from per-combo strength classes and card masks on one board (as filled by
/// `HeroRiver`): for the hero combo `hi`, equity against each opponent cluster.
#[inline]
pub fn ochs_from(classes: &[u16; COMBOS], masks: &[u64; COMBOS], clusters: &[u8], hi: usize, out: &mut [f32; N_CLUSTERS]) {
    let hc = classes[hi];
    let hm = masks[hi];
    let mut win = [0u32; N_CLUSTERS];
    let mut tie = [0u32; N_CLUSTERS];
    let mut cnt = [0u32; N_CLUSTERS];
    for j in 0..COMBOS {
        let ok = (masks[j] & hm == 0) as u32;
        let c = clusters[j] as usize;
        cnt[c] += ok;
        win[c] += ok & (classes[j] < hc) as u32;
        tie[c] += ok & (classes[j] == hc) as u32;
    }
    for c in 0..N_CLUSTERS {
        out[c] = if cnt[c] > 0 { (win[c] as f32 + 0.5 * tie[c] as f32) / cnt[c] as f32 } else { 0.5 };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_covers_all_169_classes_with_8_clusters() {
        let mut seen = [false; N_CLUSTERS];
        for r in JOHANSON_TABLE1 {
            for v in r {
                assert!((1..=8).contains(&v));
                seen[(v - 1) as usize] = true;
            }
        }
        assert!(seen.iter().all(|&s| s));
        // spot checks against the published table
        let card = |r: u8, s: u8| r * 4 + s;
        assert_eq!(cluster_of(card(12, 0), card(12, 1)), 7); // AA -> 8
        assert_eq!(cluster_of(card(0, 0), card(0, 1)), 3); // 22 -> 4
        assert_eq!(cluster_of(card(12, 0), card(0, 0)), 5); // A2 suited -> 6
        assert_eq!(cluster_of(card(12, 0), card(0, 1)), 5); // A2 offsuit -> 6
        assert_eq!(cluster_of(card(12, 0), card(11, 0)), 6); // AK suited -> 7
        assert_eq!(cluster_of(card(9, 0), card(8, 1)), 4); // JT offsuit -> 5 (row J, column T)
        assert_eq!(cluster_of(card(9, 0), card(8, 0)), 4); // JT suited -> 5 (row T, column J)
        assert_eq!(cluster_of(card(8, 0), card(8, 1)), 7); // TT -> 8
        assert_eq!(cluster_of(card(0, 0), card(1, 1)), 0); // 32 offsuit -> 1
    }

    #[test]
    fn every_combo_has_a_cluster() {
        let c = combo_clusters();
        assert_eq!(c.len(), COMBOS);
        let mut counts = [0u32; N_CLUSTERS];
        for &x in &c {
            counts[x as usize] += 1;
        }
        assert!(counts.iter().all(|&n| n > 0), "{counts:?}");
        assert_eq!(counts.iter().sum::<u32>(), 1326);
    }
}
