# M3 smoke test: blueprint trainer on m2_11g at reduced size (32 buckets)

Date 2026-11-07/08 (overnight, unattended). Machine: i7-10750H, 6 cores / 12 threads, 16 GB.
All runs at below-normal priority, on AC power, Windows "Balanced" plan.

## Setup

- Config `m2_11g` (betting tree unchanged): 10,126,666 nodes (4,426,184 decision).
- Card abstraction: the built m2_11g tables coarsened 256 → **32** post-flop buckets
  (247 → 32 on the river); preflop 169 lossless.
- Storage: 374,639,571 i32 regrets (1.40 GiB) and 62,402,067 u32 preflop average-strategy
  counts (0.23 GiB).
- Algorithm: external-sampling MCCFR as in `nlhe/src/blueprint.rs`. Linear discounting,
  negative-regret pruning (C = −300M, 5% unpruned), floor −310M, integer half-even rounding,
  preflop-only average strategy (`strategy_interval` = 1,000), lock-free 12 threads.
- Schedule: Pluribus' wall-clock fractions (prune at 200/11,520, LCFR until 400/11,520,
  discount every 10/11,520 of the plan) applied to a planned 45 minutes. A 15 s
  calibration measured 74.6k it/s and planned 201M iterations, giving:
  - prune after 3.50M iterations
  - LCFR until 7.00M
  - discount every 174,905

  Calibration overestimates throughput: on fresh, empty tables traversals are shallow. So the
  schedule ran ~2.5× later in wall time than intended (all events still happened, within
  the first ~5 minutes).
- Run: part A 20 min (fresh), the thread benchmarks, then part B 25 min resumed from part A's
  last checkpoint, for **82,456,200 iterations in 44.1 training minutes**.

Reproduce:

```
cd engine
cargo run --release --bin train_blueprint -- --config nlhe/configs/m2_11g.cfg \
  --abstraction-dir ../data/abstraction/m2_11g --buckets 32 --run-dir ../data/runs/smoke \
  --threads 12 --minutes 20 --plan-minutes 45 --strategy-interval 1000 --checkpoint-minutes 5
# then the same command with --minutes 25 resumes from the newest checkpoint
```

## Measurements

| quantity | value |
|---|---|
| iterations/s, 12 threads | 23–29k in part A after the first minute; 31–36k in part B (cause of the difference not established, see notes) |
| iterations/s, 6 vs 12 threads (bench mode, same checkpoint, 150 s each) | **19,585 vs 25,233 = 1.29×** → use 12 (design rule: ≥ 15%) |
| river share of iteration time | 25% at the start, falling to **~20%** (lazy `HeroRiver` equity + cut-offs) |
| nodes visited per iteration (6 traversals) | 670 → 1,680 (grows as strategies sharpen) |
| peak RAM (working set) | **2.45 GB** training; 4.08 GB in part A (includes the throwaway calibration table); 4.09 GB head-to-head (two checkpoints) |
| checkpoint size / write time | **1.63 GiB**, 3.2–10.3 s (FNV checksum, atomic rename) |
| resume | part B resumed from iteration 31,776,200 with identical schedule and seed; metrics continuous. The bit-identical resume is proven in a single-thread test (`single_thread_is_deterministic_and_resume_is_bit_identical`) |
| regret saturations / floor clamps | 0 saturations; floor clamps begin at ~79M iterations (173 by the end) |
| pruned actions per iteration | 0 until ~50M iterations, then rising to 0.6 (regrets of the worst actions reach C = −300M) |

## Convergence proxies (`docs/m3_smoke_metrics.tsv`, one row per minute)

| training min | iterations | pos. regret / infoset (preflop / river) | infosets touched (preflop / river) | strategy change per minute: preflop avg / post-flop current |
|---:|---:|---:|---:|---:|
| 1.0 | 2.4M | 831 / 749 | 13.7% / 15.0% | 0.352 / 0.153 |
| 5.0 | 9.2M | 1,246 / 1,393 | 18.0% / 23.9% | 0.066 / 0.044 |
| 10 | 17.5M | 1,915 / 2,270 | 20.1% / 28.9% | 0.027 / 0.022 |
| 20 | 31.8M | 2,604 / 3,193 | 22.6% / 34.1% | 0.010 / 0.010 (partial interval) |
| 30.5 | 54.9M | 3,277 / 4,095 | 25.1% / 38.5% | 0.011 / 0.014 |
| 43.5 | 81.2M | 3,772 / 4,758 | 26.9% / 41.3% | 0.0075 / 0.0096 |

- Strategy change is the mean L1 distance (0..2) between consecutive one-minute snapshots, on
  50,000 fixed random infosets per group. It falls ~45× (preflop) and ~16× (post-flop) over the
  run, and is still falling at the end.
- Average positive regret per infoset (chips) keeps growing, but more slowly than t. Divided
  by iterations it falls from ~3.1×10⁻⁴ to ~5.9×10⁻⁵ (river).
- Only 27–41% of the 32-bucket infosets were touched after 82M iterations. Deep raise lines
  are rarely reached, so the full 256-bucket config will need far more iterations for coverage.

## Head-to-head (duplicate, A in each seat vs 5 × B, common random numbers, 200k deals × 6)

| A | B | A's result |
|---|---|---:|
| 44.1 min (82.5M) | 5.0 min (9.2M) | **+360 mbb/hand** (s.e. 16) |
| 44.1 min | 19.5 min (31.8M) | **+110 mbb/hand** (s.e. 10) |
| 19.5 min | 5.0 min | +293 mbb/hand (s.e. 16) |
| 5.0 min | 44.1 min | −345 mbb/hand (s.e. 14) (reverse check, consistent) |

Later checkpoints beat earlier ones monotonically, and the gain per doubling of training is
shrinking. This is a sanity check of learning in the abstract game, not a measure of
strength against good players.

## Notes / caveats

- **Throughput gap between part A and part B.** The thread benchmark run between them gave
  25.2k it/s, which matches part A. Part B ran at ~34k from its start. Candidates: memory
  layout of a fresh process after the calibration table was freed in part A, or background
  load during part A. Not established; re-measure before relying on either number.
- **Multi-threaded runs are not bit-reproducible** (lock-free updates). Single-thread runs are,
  including across checkpoint/resume (tested).
- **Unverified Algorithm 1 details** (pending the PDF) affect pruning only. The
  `prune_skip_river_and_terminal` flag exists, default off.
