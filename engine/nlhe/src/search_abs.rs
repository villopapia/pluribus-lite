//! D-INFO-6 per-flop search abstraction (`docs/m4_info6_plan.md`): canonical flops and the
//! per-flop table file. The builder is `bin/build_search_abstraction.rs`.
//!
//! Cards: rank = c / 4, suit = c % 4 (as `eval`). A flop's class is its lexicographically
//! smallest sorted image under the 24 suit permutations (D-INFO-6a): 1,755 classes.
//!
//! Table layout for canonical flop F (rest = the 49 other cards, ascending):
//! * river bucket of (river board, combo): board index = pair index (i < j) of rest positions,
//!   `pair_index(i, j)`, 1,176 boards; entry `river[b * COMBOS + combo]`;
//! * turn bucket of (turn, combo): `turn[i * COMBOS + combo]`, i = position of the turn in rest;
//! * blocked entries are `u16::MAX`.
//!
//! File: magic, flop (3 B), pad, k (u16), seed (u64), river restarts, river sample, turn
//! restarts (u32 each), river u16s, turn u16s, then FNV-1a 64 of everything before it.
//! Written to `<name>.tmp` and renamed, so a file that exists and checks is complete.

use crate::abstraction::fnv1a64;
use crate::equity::COMBOS;
use std::io::Write;
use std::path::Path;

pub const REST: usize = 49;
pub const BOARDS: usize = REST * (REST - 1) / 2; // 1,176
const MAGIC: &[u8; 8] = b"INFO6T01";
const HEADER: usize = 8 + 4 + 2 + 2 + 8 + 4 * 3;

fn perms() -> Vec<[u8; 4]> {
    let mut v = Vec::new();
    for a in 0..4u8 {
        for b in 0..4u8 {
            for c in 0..4u8 {
                for d in 0..4u8 {
                    if a != b && a != c && a != d && b != c && b != d && c != d {
                        v.push([a, b, c, d]);
                    }
                }
            }
        }
    }
    v
}

/// The canonical form of a flop (any order) under suit permutation.
pub fn canonical(flop: [u8; 3]) -> [u8; 3] {
    perms()
        .iter()
        .map(|p| {
            let mut f = flop.map(|c| c / 4 * 4 + p[(c % 4) as usize]);
            f.sort_unstable();
            f
        })
        .min()
        .unwrap()
}

/// All canonical flops, ascending (1,755).
pub fn canonical_flops() -> Vec<[u8; 3]> {
    let mut v = Vec::new();
    for a in 0..52u8 {
        for b in a + 1..52 {
            for c in b + 1..52 {
                v.push(canonical([a, b, c]));
            }
        }
    }
    v.sort_unstable();
    v.dedup();
    v
}

/// Index of the pair (i, j), i < j < REST, in lexicographic order.
pub fn pair_index(i: usize, j: usize) -> usize {
    debug_assert!(i < j && j < REST);
    i * (2 * REST - i - 1) / 2 + (j - i - 1)
}

#[derive(Clone, Debug, PartialEq)]
pub struct Table {
    pub flop: [u8; 3],
    pub k: u16,
    pub seed: u64,
    pub river_restarts: u32,
    pub river_sample: u32,
    pub turn_restarts: u32,
    pub river: Vec<u16>, // BOARDS * COMBOS
    pub turn: Vec<u16>,  // REST * COMBOS
}

pub fn file_name(flop: [u8; 3]) -> String {
    format!("flop_{:02}_{:02}_{:02}.bin", flop[0], flop[1], flop[2])
}

pub fn write(dir: &Path, t: &Table) -> std::io::Result<()> {
    assert_eq!(t.river.len(), BOARDS * COMBOS);
    assert_eq!(t.turn.len(), REST * COMBOS);
    let mut b = Vec::with_capacity(HEADER + 2 * (t.river.len() + t.turn.len()) + 8);
    b.extend_from_slice(MAGIC);
    b.extend_from_slice(&[t.flop[0], t.flop[1], t.flop[2], 0]);
    b.extend_from_slice(&t.k.to_le_bytes());
    b.extend_from_slice(&[0, 0]);
    b.extend_from_slice(&t.seed.to_le_bytes());
    for x in [t.river_restarts, t.river_sample, t.turn_restarts] {
        b.extend_from_slice(&x.to_le_bytes());
    }
    for x in t.river.iter().chain(&t.turn) {
        b.extend_from_slice(&x.to_le_bytes());
    }
    b.extend_from_slice(&fnv1a64(&b).to_le_bytes());
    let path = dir.join(file_name(t.flop));
    let tmp = path.with_extension("tmp");
    let mut f = std::fs::File::create(&tmp)?;
    f.write_all(&b)?;
    f.sync_all()?;
    drop(f);
    std::fs::rename(tmp, path)
}

/// Reads and verifies a table; `Err` if missing, truncated, or the checksum fails.
pub fn read(path: &Path) -> std::io::Result<Table> {
    let bad = |m: &str| std::io::Error::new(std::io::ErrorKind::InvalidData, format!("{}: {m}", path.display()));
    let b = std::fs::read(path)?;
    let n = HEADER + 2 * (BOARDS + REST) * COMBOS + 8;
    if b.len() != n || &b[..8] != MAGIC {
        return Err(bad("wrong size or magic"));
    }
    if fnv1a64(&b[..n - 8]).to_le_bytes() != b[n - 8..] {
        return Err(bad("checksum"));
    }
    let u32_at = |o: usize| u32::from_le_bytes(b[o..o + 4].try_into().unwrap());
    let u16s: Vec<u16> = b[HEADER..n - 8].chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
    Ok(Table {
        flop: [b[8], b[9], b[10]],
        k: u16::from_le_bytes([b[12], b[13]]),
        seed: u64::from_le_bytes(b[16..24].try_into().unwrap()),
        river_restarts: u32_at(24),
        river_sample: u32_at(28),
        turn_restarts: u32_at(32),
        river: u16s[..BOARDS * COMBOS].to_vec(),
        turn: u16s[BOARDS * COMBOS..].to_vec(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn there_are_1755_canonical_flops_and_every_flop_maps_to_one() {
        let c = canonical_flops();
        assert_eq!(c.len(), 1755);
        for f in &c {
            assert_eq!(canonical(*f), *f, "class representative is canonical");
        }
        // suit permutations and card order do not change the class
        for p in perms() {
            let f = [44u8, 22, 1];
            let g = [f[2], f[0], f[1]].map(|x| x / 4 * 4 + p[(x % 4) as usize]);
            assert_eq!(canonical(g), canonical(f));
        }
        // rank changes do
        assert_ne!(canonical([0, 4, 8]), canonical([0, 4, 12]));
    }

    #[test]
    fn pair_index_is_a_bijection() {
        let mut seen = vec![false; BOARDS];
        for i in 0..REST {
            for j in i + 1..REST {
                let p = pair_index(i, j);
                assert!(!seen[p]);
                seen[p] = true;
            }
        }
        assert!(seen.iter().all(|&s| s));
    }

    #[test]
    fn table_round_trips_and_corruption_is_detected() {
        let dir = std::env::temp_dir().join(format!("search_abs_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let t = Table {
            flop: [1, 22, 44],
            k: 500,
            seed: 7,
            river_restarts: 5,
            river_sample: 200_000,
            turn_restarts: 5,
            river: (0..BOARDS * COMBOS).map(|i| (i % 500) as u16).collect(),
            turn: (0..REST * COMBOS).map(|i| (i % 499) as u16).collect(),
        };
        write(&dir, &t).unwrap();
        let path = dir.join(file_name(t.flop));
        assert_eq!(read(&path).unwrap(), t);
        assert!(!path.with_extension("tmp").exists());
        let mut b = std::fs::read(&path).unwrap();
        b[100] ^= 1;
        std::fs::write(&path, &b).unwrap();
        assert!(read(&path).is_err());
        std::fs::write(&path, &b[..1000]).unwrap();
        assert!(read(&path).is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
