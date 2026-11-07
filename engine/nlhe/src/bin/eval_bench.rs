//! Hand-evaluator and river-equity benchmarks for this machine.
//!
//! `cargo run --release --bin eval_bench`
//!
//! Every measurement is repeated 3 times; the median is reported along with min and max.
//! Inputs are pre-generated with a fixed seed so RNG cost is excluded from timings.

use cfr::rng::Rng;
use nlhe::equity::{river_equity_single, RiverBatch};
use nlhe::eval::{card_bit, eval, Hand};
use nlhe::eval_lut::{LHand, Lut};
use std::hint::black_box;
use std::time::Instant;

fn random_cards(rng: &mut Rng, n: usize) -> Vec<u8> {
    let mut deck: Vec<u8> = (0..52).collect();
    for i in 0..n {
        let j = i + (rng.next_u64() % (52 - i as u64)) as usize;
        deck.swap(i, j);
    }
    deck[..n].to_vec()
}

fn random_hands(seed: u64, n: usize) -> Vec<Hand> {
    let mut rng = Rng::new(seed);
    (0..n).map(|_| random_cards(&mut rng, 7).iter().fold(0, |h, &c| h | card_bit(c))).collect()
}

fn median3(mut v: Vec<f64>) -> (f64, f64, f64) {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (v[v.len() / 2], v[0], v[v.len() - 1])
}

/// Random-access evals/s across `threads` threads (each with its own 1M-hand array).
fn evals_per_sec(threads: usize, passes: usize) -> f64 {
    let inputs: Vec<Vec<Hand>> = (0..threads).map(|t| random_hands(100 + t as u64, 1 << 20)).collect();
    let t0 = Instant::now();
    std::thread::scope(|s| {
        for inp in &inputs {
            s.spawn(move || {
                let mut acc = 0u32;
                for _ in 0..passes {
                    for &h in inp {
                        acc = acc.wrapping_add(eval(h));
                    }
                }
                black_box(acc);
            });
        }
    });
    (threads * passes * inputs[0].len()) as f64 / t0.elapsed().as_secs_f64()
}

fn random_lhands(seed: u64, n: usize) -> Vec<LHand> {
    let mut rng = Rng::new(seed);
    (0..n).map(|_| LHand::of(&random_cards(&mut rng, 7))).collect()
}

/// Same as `evals_per_sec` with the lookup-table evaluator.
fn lut_evals_per_sec(threads: usize, passes: usize) -> f64 {
    let lut = Lut::get();
    let inputs: Vec<Vec<LHand>> = (0..threads).map(|t| random_lhands(100 + t as u64, 1 << 20)).collect();
    let t0 = Instant::now();
    std::thread::scope(|s| {
        for inp in &inputs {
            s.spawn(move || {
                let mut acc = 0u32;
                for _ in 0..passes {
                    for &h in inp {
                        acc = acc.wrapping_add(lut.class(h) as u32);
                    }
                }
                black_box(acc);
            });
        }
    });
    (threads * passes * inputs[0].len()) as f64 / t0.elapsed().as_secs_f64()
}

/// Full enumeration of all 133,784,560 seven-card hands, single thread, incremental masks.
fn enumerate_all() -> f64 {
    let bits: Vec<Hand> = (0..52u8).map(card_bit).collect();
    let t0 = Instant::now();
    let mut acc = 0u32;
    let mut n = 0u64;
    for a in 0..52 {
        for b in a + 1..52 {
            let hb = bits[a] | bits[b];
            for c in b + 1..52 {
                let hc = hb | bits[c];
                for d in c + 1..52 {
                    let hd = hc | bits[d];
                    for e in d + 1..52 {
                        let he = hd | bits[e];
                        for f in e + 1..52 {
                            let hf = he | bits[f];
                            for g in f + 1..52 {
                                acc = acc.wrapping_add(eval(hf | bits[g]));
                                n += 1;
                            }
                        }
                    }
                }
            }
        }
    }
    black_box(acc);
    n as f64 / t0.elapsed().as_secs_f64()
}

fn boards(seed: u64, n: usize) -> Vec<([u8; 5], [u8; 2])> {
    let mut rng = Rng::new(seed);
    (0..n)
        .map(|_| {
            let c = random_cards(&mut rng, 7);
            ([c[0], c[1], c[2], c[3], c[4]], [c[5], c[6]])
        })
        .collect()
}

/// microseconds per single-hand equity call
fn single_equity_us(n: usize) -> f64 {
    let b = boards(7, n);
    let t0 = Instant::now();
    let mut acc = 0.0;
    for (board, hole) in &b {
        acc += river_equity_single(board, *hole);
    }
    black_box(acc);
    t0.elapsed().as_secs_f64() * 1e6 / n as f64
}

/// microseconds per whole-board batch (all 1,081 hands), single thread
fn batch_equity_us(n: usize) -> f64 {
    let b = boards(8, n);
    let mut rb = RiverBatch::new();
    let t0 = Instant::now();
    let mut acc = 0.0f32;
    for (board, hole) in &b {
        rb.compute(board);
        acc += rb.equity[nlhe::equity::combo_index(hole[0], hole[1])];
    }
    black_box(acc);
    t0.elapsed().as_secs_f64() * 1e6 / n as f64
}

/// microseconds per whole-board batch with the old oracle + sort path
fn batch_oracle_us(n: usize) -> f64 {
    let b = boards(8, n);
    let mut rb = RiverBatch::new();
    let t0 = Instant::now();
    let mut acc = 0.0f32;
    for (board, hole) in &b {
        rb.compute_oracle(board);
        acc += rb.equity[nlhe::equity::combo_index(hole[0], hole[1])];
    }
    black_box(acc);
    t0.elapsed().as_secs_f64() * 1e6 / n as f64
}

/// microseconds per MCCFR-style call: equities for `k` hero hands on a sampled board
fn heroes_us(n: usize, k: usize) -> f64 {
    let mut rng = Rng::new(9);
    let inputs: Vec<([u8; 5], Vec<(u8, u8)>)> = (0..n)
        .map(|_| {
            let c = random_cards(&mut rng, 17);
            ([c[0], c[1], c[2], c[3], c[4]], (0..k).map(|p| (c[5 + 2 * p], c[6 + 2 * p])).collect())
        })
        .collect();
    let mut rb = RiverBatch::new();
    let mut out = [0f32; 6];
    let t0 = Instant::now();
    for (board, heroes) in &inputs {
        rb.compute_heroes(board, heroes, &mut out);
        black_box(&out);
    }
    t0.elapsed().as_secs_f64() * 1e6 / n as f64
}

/// boards per second for whole-board batches across `threads` threads
fn batch_boards_per_sec(threads: usize, per_thread: usize) -> f64 {
    let inputs: Vec<Vec<([u8; 5], [u8; 2])>> = (0..threads).map(|t| boards(200 + t as u64, per_thread)).collect();
    let t0 = Instant::now();
    std::thread::scope(|s| {
        for inp in &inputs {
            s.spawn(move || {
                let mut rb = RiverBatch::new();
                let mut acc = 0.0f32;
                for (board, _) in inp {
                    rb.compute(board);
                    acc += rb.equity[0].max(0.0);
                }
                black_box(acc);
            });
        }
    });
    (threads * per_thread) as f64 / t0.elapsed().as_secs_f64()
}

fn report(label: &str, unit: &str, f: impl Fn() -> f64) -> f64 {
    let (m, lo, hi) = median3((0..3).map(|_| f()).collect());
    println!("| {label} | {m:.3e} {unit} | {lo:.3e} – {hi:.3e} |");
    m
}

fn main() {
    println!(
        "logical CPUs: {}",
        std::thread::available_parallelism().map(|n| n.get()).unwrap_or(0)
    );
    // warm-up (turbo ramp, page faults); also builds the lookup tables
    let t0 = Instant::now();
    Lut::get();
    println!("lookup tables built in {:.0} ms (lut max probe {})", t0.elapsed().as_secs_f64() * 1e3, Lut::get().max_probe());
    evals_per_sec(12, 2);

    println!("\n| measurement | median | min – max |");
    println!("|---|---:|---:|");
    let e1 = report("random 7-card evals/s, 1 thread", "evals/s", || evals_per_sec(1, 20));
    let e6 = report("random 7-card evals/s, 6 threads", "evals/s", || evals_per_sec(6, 20));
    let e12 = report("random 7-card evals/s, 12 threads", "evals/s", || evals_per_sec(12, 20));
    let l1 = report("LUT random 7-card evals/s, 1 thread", "evals/s", || lut_evals_per_sec(1, 20));
    let l6 = report("LUT random 7-card evals/s, 6 threads", "evals/s", || lut_evals_per_sec(6, 20));
    let l12 = report("LUT random 7-card evals/s, 12 threads", "evals/s", || lut_evals_per_sec(12, 20));
    report("enumerate all 133.8M 7-card hands, 1 thread", "evals/s", enumerate_all);
    let bo = report("river equity, ALL hands, OLD path (oracle eval + sort)", "µs/board", || batch_oracle_us(20_000));
    let s = report("river equity, ONE hand vs random (990 evals)", "µs/call", || single_equity_us(20_000));
    let b = report("river equity, ALL 1,081 hands on a board", "µs/board", || batch_equity_us(20_000));
    let h1 = report("MCCFR path: equities of 1 hero on a board", "µs/call", || heroes_us(20_000, 1));
    let h2 = report("MCCFR path: equities of 2 heroes on a board", "µs/call", || heroes_us(20_000, 2));
    let h6 = report("MCCFR path: equities of 6 heroes on a board", "µs/call", || heroes_us(20_000, 6));
    let b1 = report("whole-board batches, 1 thread", "boards/s", || batch_boards_per_sec(1, 20_000));
    let b6 = report("whole-board batches, 6 threads", "boards/s", || batch_boards_per_sec(6, 20_000));
    let b12 = report("whole-board batches, 12 threads", "boards/s", || batch_boards_per_sec(12, 20_000));

    println!("\nLUT vs oracle evaluator: 1t {:.2}x, 6t {:.2}x, 12t {:.2}x", l1 / e1, l6 / e6, l12 / e12);
    println!("LUT scaling: 12t/6t = {:.2}x", l12 / l6);
    println!("batch: new {:.2} µs vs old {:.2} µs per board = {:.2}x faster", b, bo, bo / b);
    println!("scaling: evals 6t/1t = {:.2}x, 12t/1t = {:.2}x, 12t/6t = {:.2}x", e6 / e1, e12 / e1, e12 / e6);
    println!("scaling: batches 6t/1t = {:.2}x, 12t/1t = {:.2}x, 12t/6t = {:.2}x", b6 / b1, b12 / b1, b12 / b6);
    println!("MCCFR path: 1 hero {h1:.2} µs, 2 heroes {h2:.2} µs, 6 heroes {h6:.2} µs per iteration");
    println!("per player on a 6-way river (batch / 6): {:.2} µs", b / 6.0);
    println!("single-hand call vs batch: {:.1}x", s / b);
    let all_boards = 2_598_960.0;
    println!(
        "all 2,598,960 river boards with the batch at 12 threads: {:.1} s",
        all_boards / b12
    );
    println!(
        "turn features (270,725 turn boards x 48 rivers) at 12 threads: {:.1} s",
        270_725.0 * 48.0 / b12
    );
    println!(
        "flop features, 1,755 canonical flops x 1,176 run-outs at 12 threads: {:.1} s",
        1_755.0 * 1_176.0 / b12
    );
}
