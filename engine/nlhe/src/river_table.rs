//! Precomputed river bucket table (`river_method = precomputed_table`, docs/HANDOFF.md
//! decision 4): the fine OCHS k-means bucket of every (hole, river board), computed with
//! exactly the code path used on the fly (`HeroRiver::ochs` + `KMeans::nearest`), so the two
//! methods give identical buckets (tested).
//!
//! **Layout (board-major):** row = colex rank of the 5-card board among C(52,5) = 2,598,960
//! boards; inside a row, 1,081 = C(47,2) entries, one per hole pair disjoint from the board,
//! by the colex rank of the hole pair renumbered to skip the board cards. 2,809,475,760 u8
//! entries (2.62 GiB), the same count as the hole-major `index::RIVER_ENTRIES`.
//! Board-major means all players of one deal read one 1,081-byte row: one read per deal.
//!
//! **File:** `river.tbl`: 64-byte header (magic "PLRT", version, k, rows, row length,
//! FNV-1a of the river centroids it was built from, payload FNV-1a), then the payload.
//! Built streaming into `river.tbl.tmp` and renamed at the end; build memory is a few MiB.
//!
//! **Access modes:**
//! * [`RiverTableMode::Ram`]: the whole payload is read into memory (2.62 GiB).
//! * [`RiverTableMode::File`]: one positioned read of the row per deal (std `FileExt`,
//!   Linux `read_at` / Windows `seek_read`, one file handle per worker). Rows are served from
//!   the OS page cache, which is evictable: this is the std-only stand-in for memory
//!   mapping (true mmap needs a crate or OS FFI; see docs/deviations.md D-INFO-7).

use crate::abstraction::{fnv1a64, parallel_chunks, river_features, FeatureScratch};
use crate::index::binom;
use crate::kmeans::KMeans;
use std::fs::{self, File};
use std::io::{self, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

pub const BOARDS: u64 = 2_598_960; // C(52,5)
pub const ROW: usize = 1_081; // C(47,2)
pub const ENTRIES: u64 = BOARDS * ROW as u64;
pub const HEADER: u64 = 64;
const MAGIC: &[u8; 4] = b"PLRT";
const VERSION: u32 = 1;

/// Colex rank of a 5-card board (any order) among all C(52,5) boards.
#[inline]
pub fn board_rank(board: &[u8; 5]) -> u64 {
    let mut b = *board;
    b.sort_unstable();
    b.iter().enumerate().map(|(i, &c)| binom(c as usize, i + 1)).sum()
}

/// Position of the hole pair inside its board's row (0..1081).
#[inline]
pub fn hole_slot(hole: (u8, u8), board: &[u8; 5]) -> usize {
    let skip = |c: u8| c - board.iter().filter(|&&x| x < c).count() as u8;
    let (a, b) = (skip(hole.0), skip(hole.1));
    let (a, b) = if a < b { (a, b) } else { (b, a) };
    (binom(a as usize, 1) + binom(b as usize, 2)) as usize
}

/// Boards in colex order (rank 0, 1, 2, ...): largest card outermost.
pub fn for_each_board_colex(mut f: impl FnMut(u64, [u8; 5])) {
    let mut r = 0u64;
    for c4 in 4..52u8 {
        for c3 in 3..c4 {
            for c2 in 2..c3 {
                for c1 in 1..c2 {
                    for c0 in 0..c1 {
                        f(r, [c0, c1, c2, c3, c4]);
                        r += 1;
                    }
                }
            }
        }
    }
}

/// Hole pairs of one row, in slot order.
pub fn row_holes(board: &[u8; 5]) -> Vec<(u8, u8)> {
    let rest: Vec<u8> = (0..52u8).filter(|c| !board.contains(c)).collect();
    let mut v = vec![(0u8, 0u8); ROW];
    for j in 1..rest.len() {
        for i in 0..j {
            v[(binom(i, 1) + binom(j, 2)) as usize] = (rest[i], rest[j]);
        }
    }
    v
}

pub fn centroid_hash(km: &KMeans) -> u64 {
    fnv1a64(&crate::abstraction::centroids_to_bytes(&km.c))
}

/// Build the table for `km` (fine OCHS centroids) into `path` (streamed, tmp + rename).
/// `rows` < BOARDS builds a prefix only (tests). `progress(rows_done)` is called per block.
pub fn build(path: &Path, km: &KMeans, threads: usize, rows: u64, mut progress: impl FnMut(u64)) -> io::Result<()> {
    assert!(km.k <= 256, "u8 table");
    let boards = {
        let mut v = Vec::with_capacity(rows as usize);
        for_each_board_colex(|r, b| {
            if r < rows {
                v.push(b)
            }
        });
        v
    };
    let tmp = path.with_extension("tbl.tmp");
    let mut w = BufWriter::with_capacity(1 << 22, File::create(&tmp)?);
    let mut h = [0u8; HEADER as usize];
    h[0..4].copy_from_slice(MAGIC);
    h[4..8].copy_from_slice(&VERSION.to_le_bytes());
    h[8..12].copy_from_slice(&(km.k as u32).to_le_bytes());
    h[12..20].copy_from_slice(&rows.to_le_bytes());
    h[20..24].copy_from_slice(&(ROW as u32).to_le_bytes());
    h[24..32].copy_from_slice(&centroid_hash(km).to_le_bytes());
    // payload hash at 32..40 is patched after the payload is written
    w.write_all(&h)?;
    let block = 4096usize;
    let mut buf = vec![0u8; block * ROW];
    let mut fnv: u64 = 0xcbf2_9ce4_8422_2325;
    for (bi, chunk) in boards.chunks(block).enumerate() {
        let base = buf.as_mut_ptr() as usize;
        parallel_chunks(chunk.len(), threads, 16, |a, b| {
            let mut s = FeatureScratch::new();
            let combos = crate::equity::combo_cards();
            for (k, board) in chunk[a..b].iter().enumerate() {
                let row = a + k;
                // SAFETY: each worker writes only rows a..b of `out`, disjoint between workers.
                let dst = unsafe { std::slice::from_raw_parts_mut((base + row * ROW) as *mut u8, ROW) };
                river_features(*board, &mut s, |ci, x| {
                    dst[hole_slot(combos[ci], board)] = km.nearest(x).0 as u8;
                });
            }
        });
        let out = &buf[..chunk.len() * ROW];
        for &byte in out {
            fnv ^= byte as u64;
            fnv = fnv.wrapping_mul(0x0000_0100_0000_01b3);
        }
        w.write_all(out)?;
        progress(((bi + 1) * block).min(boards.len()) as u64);
    }
    w.flush()?;
    let mut f = w.into_inner().map_err(|e| e.into_error())?;
    use std::io::{Seek, SeekFrom};
    f.seek(SeekFrom::Start(32))?;
    f.write_all(&fnv.to_le_bytes())?;
    f.sync_all()?;
    drop(f);
    fs::rename(&tmp, path)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RiverTableMode {
    Ram,
    File,
}

impl std::str::FromStr for RiverTableMode {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "ram" => Ok(RiverTableMode::Ram),
            "file" => Ok(RiverTableMode::File),
            _ => Err(format!("river table mode: expected ram|file, got {s:?}")),
        }
    }
}

pub struct RiverTable {
    pub k: u32,
    pub rows: u64,
    pub path: PathBuf,
    pub payload_hash: u64,
    ram: Option<Vec<u8>>,
}

/// Per-worker reader state (own file handle in File mode).
#[derive(Default)]
pub struct RiverReader {
    file: Option<File>,
}

impl RiverTable {
    /// Open and check the header against the centroids it must have been built from.
    /// `verify` re-hashes the whole payload (reads 2.6 GiB once).
    pub fn open(path: &Path, km: &KMeans, mode: RiverTableMode, verify: bool) -> io::Result<Self> {
        let bad = |m: String| io::Error::new(io::ErrorKind::InvalidData, format!("{}: {m}", path.display()));
        let mut f = File::open(path)?;
        let mut h = [0u8; HEADER as usize];
        f.read_exact(&mut h)?;
        let u32_at = |i: usize| u32::from_le_bytes(h[i..i + 4].try_into().unwrap());
        let u64_at = |i: usize| u64::from_le_bytes(h[i..i + 8].try_into().unwrap());
        if &h[0..4] != MAGIC || u32_at(4) != VERSION {
            return Err(bad("not a river table (magic/version)".into()));
        }
        let (k, rows, row, ch, ph) = (u32_at(8), u64_at(12), u32_at(20) as usize, u64_at(24), u64_at(32));
        if k as usize != km.k || row != ROW || ch != centroid_hash(km) {
            return Err(bad("built for other river centroids (k or centroid hash differs)".into()));
        }
        let len = f.metadata()?.len();
        if len != HEADER + rows * ROW as u64 {
            return Err(bad(format!("size {len} != header + {rows} rows")));
        }
        let ram = if mode == RiverTableMode::Ram {
            let mut v = vec![0u8; (rows * ROW as u64) as usize];
            f.read_exact(&mut v)?;
            if verify && fnv1a64(&v) != ph {
                return Err(bad("payload checksum mismatch".into()));
            }
            Some(v)
        } else {
            if verify {
                // streamed: no table-sized buffer
                let mut r = io::BufReader::with_capacity(1 << 22, &mut f);
                let mut buf = vec![0u8; 1 << 22];
                let mut fnv: u64 = 0xcbf2_9ce4_8422_2325;
                loop {
                    let n = r.read(&mut buf)?;
                    if n == 0 {
                        break;
                    }
                    for &b in &buf[..n] {
                        fnv ^= b as u64;
                        fnv = fnv.wrapping_mul(0x0000_0100_0000_01b3);
                    }
                }
                if fnv != ph {
                    return Err(bad("payload checksum mismatch".into()));
                }
            }
            None
        };
        Ok(RiverTable { k, rows, path: path.to_path_buf(), payload_hash: ph, ram })
    }

    pub fn mode(&self) -> RiverTableMode {
        if self.ram.is_some() {
            RiverTableMode::Ram
        } else {
            RiverTableMode::File
        }
    }

    /// Copy the row of `board` into `out`.
    pub fn read_row(&self, board: &[u8; 5], out: &mut [u8; ROW], rd: &mut RiverReader) -> io::Result<()> {
        let r = board_rank(board);
        assert!(r < self.rows, "board row {r} not in table ({} rows)", self.rows);
        let off = r * ROW as u64;
        if let Some(v) = &self.ram {
            out.copy_from_slice(&v[off as usize..off as usize + ROW]);
            return Ok(());
        }
        if rd.file.is_none() {
            rd.file = Some(File::open(&self.path)?);
        }
        read_exact_at(rd.file.as_ref().unwrap(), out, HEADER + off)
    }
}

#[cfg(unix)]
fn read_exact_at(f: &File, buf: &mut [u8], off: u64) -> io::Result<()> {
    std::os::unix::fs::FileExt::read_exact_at(f, buf, off)
}

#[cfg(windows)]
fn read_exact_at(f: &File, mut buf: &mut [u8], mut off: u64) -> io::Result<()> {
    use std::os::windows::fs::FileExt;
    while !buf.is_empty() {
        let n = f.seek_read(buf, off)?;
        if n == 0 {
            return Err(io::ErrorKind::UnexpectedEof.into());
        }
        buf = &mut buf[n..];
        off += n as u64;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::equity::{HeroRiver, RiverBatch};
    use crate::kmeans::Metric;
    use crate::ochs::{combo_clusters, N_CLUSTERS};

    #[test]
    fn colex_enumeration_matches_board_rank() {
        let mut n = 0u64;
        let mut ok = true;
        for_each_board_colex(|r, b| {
            ok &= board_rank(&b) == r;
            n += 1;
        });
        assert!(ok);
        assert_eq!(n, BOARDS);
        assert_eq!(ENTRIES, crate::index::RIVER_ENTRIES);
    }

    #[test]
    fn hole_slot_is_a_bijection_per_board() {
        for board in [[0u8, 1, 2, 3, 4], [47, 48, 49, 50, 51], [3, 17, 22, 40, 51], [51, 0, 25, 26, 13]] {
            let holes = row_holes(&board);
            let mut seen = vec![false; ROW];
            for a in 0..52u8 {
                for b in a + 1..52 {
                    if board.contains(&a) || board.contains(&b) {
                        continue;
                    }
                    let s = hole_slot((a, b), &board);
                    assert_eq!(s, hole_slot((b, a), &board));
                    assert!(!seen[s]);
                    seen[s] = true;
                    assert_eq!(holes[s], (a, b));
                }
            }
            assert!(seen.iter().all(|&x| x));
        }
    }

    fn toy_km() -> KMeans {
        // 12 deterministic OCHS-like centroids
        let c: Vec<f32> = (0..12 * N_CLUSTERS).map(|i| ((i * 37 % 101) as f32) / 100.0).collect();
        KMeans::from_centroids(c, N_CLUSTERS, Metric::L2)
    }

    /// The table equals the on-the-fly bucket (same features, same nearest search) for every
    /// hole on a prefix of boards, in both access modes.
    #[test]
    fn table_matches_on_the_fly_ochs_both_modes() {
        let km = toy_km();
        let dir = std::env::temp_dir().join(format!("plrt_test_{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("river.tbl");
        let rows = 3_000u64;
        build(&path, &km, 4, rows, |_| {}).unwrap();
        let clusters = combo_clusters();
        let (mut rb, mut hr) = (RiverBatch::new(), HeroRiver::new());
        for mode in [RiverTableMode::Ram, RiverTableMode::File] {
            let t = RiverTable::open(&path, &km, mode, true).unwrap();
            let mut rd = RiverReader::default();
            let mut row = [0u8; ROW];
            let mut checked = 0;
            for_each_board_colex(|r, b| {
                if r >= rows || r % 7 != 0 {
                    return;
                }
                t.read_row(&b, &mut row, &mut rd).unwrap();
                hr.prepare(&mut rb, &b);
                for (s, &(x, y)) in row_holes(&b).iter().enumerate() {
                    let mut f = [0f32; N_CLUSTERS];
                    hr.ochs(y, x, &clusters, &mut f); // order of hole cards must not matter
                    assert_eq!(row[s] as usize, km.nearest(&f).0, "board {b:?} hole {x},{y}");
                    assert_eq!(hole_slot((y, x), &b), s);
                    checked += 1;
                }
            });
            assert!(checked > 400_000);
        }
        // a table built for other centroids is refused
        let mut other = toy_km();
        other.c[0] += 0.5;
        assert!(RiverTable::open(&path, &other, RiverTableMode::File, false).is_err());
        fs::remove_dir_all(&dir).ok();
    }
}
