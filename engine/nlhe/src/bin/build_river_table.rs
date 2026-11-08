//! Build `<dir>/river.tbl`, the precomputed river bucket table for the OCHS model in
//! `<dir>/river.ochs` (see `river_table.rs`; docs/HANDOFF.md decision 4).
//!
//! `build_river_table --abstraction-dir ../data/abstraction/j63 [--threads 12] [--rows N]
//!   [--check-samples 200000]`
//!
//! Streams 2,598,960 board rows (2.62 GiB) into `river.tbl.tmp`, renames on success, then
//! re-opens the table (payload checksum) and compares random (hole, board) samples with the
//! on-the-fly OCHS bucket.

use nlhe::abstraction::{centroids_from_bytes, read_abs_file};
use nlhe::equity::{HeroRiver, RiverBatch};
use nlhe::kmeans::{KMeans, Metric};
use nlhe::ochs::{combo_clusters, N_CLUSTERS};
use nlhe::river_table::*;
use std::path::PathBuf;
use std::time::Instant;

fn main() {
    let v: Vec<String> = std::env::args().skip(1).collect();
    let (mut dir, mut threads, mut rows, mut samples) = (PathBuf::new(), 12usize, BOARDS, 200_000u64);
    let mut i = 0;
    while i < v.len() {
        let x = v.get(i + 1).cloned().unwrap_or_default();
        match v[i].as_str() {
            "--abstraction-dir" => dir = x.into(),
            "--threads" => threads = x.parse().unwrap(),
            "--rows" => rows = x.parse().unwrap(),
            "--check-samples" => samples = x.parse().unwrap(),
            o => panic!("unknown argument {o}"),
        }
        i += 2;
    }
    let (h, bytes) = read_abs_file(&dir.join("river.ochs")).expect("river.ochs");
    let km = KMeans::from_centroids(centroids_from_bytes(&bytes), N_CLUSTERS, Metric::L2);
    assert_eq!(km.k, h.k as usize);
    let path = dir.join("river.tbl");
    println!("building {} for k = {} ({rows} rows, {threads} threads)", path.display(), km.k);
    let t0 = Instant::now();
    let mut last = 0u64;
    build(&path, &km, threads, rows, |done| {
        if done * 20 / rows > last * 20 / rows || done == rows {
            let s = t0.elapsed().as_secs_f64();
            println!("  {done}/{rows} rows, {s:.0}s, eta {:.0}s", s / done as f64 * (rows - done) as f64);
            last = done;
        }
    })
    .expect("build");
    println!("built in {:.1}s", t0.elapsed().as_secs_f64());

    let t1 = Instant::now();
    let t = RiverTable::open(&path, &km, RiverTableMode::File, true).expect("open + checksum");
    println!("checksum ok ({:.1}s), payload fnv {:016x}", t1.elapsed().as_secs_f64(), t.payload_hash);
    let clusters = combo_clusters();
    let (mut rb, mut hr, mut rd) = (RiverBatch::new(), HeroRiver::new(), RiverReader::default());
    let mut row = [0u8; ROW];
    let mut z = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = || {
        z ^= z << 13;
        z ^= z >> 7;
        z ^= z << 17;
        z
    };
    let mut bad = 0u64;
    for _ in 0..samples {
        let mut deck: Vec<u8> = (0..52).collect();
        for j in 0..7 {
            let k = j + (next() % (52 - j as u64)) as usize;
            deck.swap(j, k);
        }
        let board = [deck[0], deck[1], deck[2], deck[3], deck[4]];
        if board_rank(&board) >= t.rows {
            continue;
        }
        t.read_row(&board, &mut row, &mut rd).unwrap();
        hr.prepare(&mut rb, &board);
        let mut f = [0f32; N_CLUSTERS];
        hr.ochs(deck[5], deck[6], &clusters, &mut f);
        bad += (row[hole_slot((deck[5], deck[6]), &board)] as usize != km.nearest(&f).0) as u64;
    }
    println!("random check: {samples} (hole, board) samples, {bad} mismatches vs on-the-fly OCHS");
    assert_eq!(bad, 0);
}
