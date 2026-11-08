//! Head-to-head between two blueprint checkpoints of the same config (duplicate format:
//! A takes each seat in turn against five copies of B, common random numbers).
//!
//! `h2h --config nlhe/configs/m2_11g.cfg --abstraction-dir ../data/abstraction/m2_11g
//!   --buckets 32 --a A.bin --b B.bin [--deals 200000] [--threads 12] [--seed 1]`
//!
//! Reports A's result in chips/hand and mbb/hand (1 bb = 100 chips) with its standard error.

use nlhe::betting::Rules;
use nlhe::blueprint::*;
use nlhe::buckets::{CardAbstraction, TableAbstraction};
use nlhe::config::AbstractionConfig;
use std::path::PathBuf;
use std::time::Instant;

fn main() {
    let v: Vec<String> = std::env::args().skip(1).collect();
    let get = |k: &str| v.iter().position(|x| x == k).map(|i| v[i + 1].clone());
    let cfg = AbstractionConfig::parse(&std::fs::read_to_string(get("--config").expect("--config")).unwrap()).unwrap();
    let abs_dir: PathBuf = get("--abstraction-dir").expect("--abstraction-dir").into();
    let buckets: u32 = get("--buckets").map(|x| x.parse().unwrap()).unwrap_or(32);
    let pa: PathBuf = get("--a").expect("--a").into();
    let pb: PathBuf = get("--b").expect("--b").into();
    let deals: u64 = get("--deals").map(|x| x.parse().unwrap()).unwrap_or(200_000);
    let threads: usize = get("--threads").map(|x| x.parse().unwrap()).unwrap_or(12);
    let seed: u64 = get("--seed").map(|x| x.parse().unwrap()).unwrap_or(1);

    let abs = TableAbstraction::load(&abs_dir, buckets).expect("abstraction");
    let tree = Tree::build(&Rules::default(), &cfg, abs.buckets());
    let a = Storage::new(&tree);
    let b = Storage::new(&tree);
    let ma = load_checkpoint(&pa, &tree, &a).expect("load A");
    let mb = load_checkpoint(&pb, &tree, &b).expect("load B");
    let t0 = Instant::now();
    let (m, se) = head_to_head(&tree, &abs, &a, &b, deals, seed, threads);
    println!(
        "A = iteration {} ({:.1} min) vs B = iteration {} ({:.1} min): {:+.3} chips/hand = {:+.1} mbb/hand (s.e. {:.1}), {} deals x 6 seats in {:.1}s",
        ma.iteration,
        ma.elapsed_secs / 60.0,
        mb.iteration,
        mb.elapsed_secs / 60.0,
        m,
        m * 10.0,
        se * 10.0,
        deals,
        t0.elapsed().as_secs_f64()
    );
}
