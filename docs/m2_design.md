# M2 design: 6-max abstraction (cards + bets) and memory budget

Status: **design only.** Nothing here trains or allocates strategy tables. The betting-tree
counter and memory estimator are implemented and tested (`engine/nlhe`); card abstraction,
equity tables and parallel MCCFR are specified below but not built.

Reproduce every number in the estimator sections:

```
cd engine
cargo run --release --bin m2_estimate -- nlhe/configs/m2_11g.cfg nlhe/configs/m2_26g.cfg
cargo test -p nlhe
```

Paper figures marked **(unverified)** are from memory and will be checked against the
supplement PDF together with the M1 Algorithm 1 diff.

## 1. Game and infoset model

- 6-max NLHE, blinds 50/100, 10,000-chip (100 bb) stacks reset every hand, matching the
  Pluribus logs.
- **Infoset = (betting history, current-street card bucket).** This is imperfect recall:
  earlier streets' buckets are forgotten, as in Pluribus. So per street,
  `infosets = betting nodes × buckets`.
- Seat 0 = SB, 1 = BB, 2 = UTG … 5 = BTN. Preflop the first to act is UTG; post-flop it's the
  first live seat from the SB.

## 2. Card abstraction

| street | method | count (config) | lookup at run time |
|---|---|---|---|
| preflop | lossless: 169 strategically distinct hands | 169 (fixed) | table of 169 |
| flop | k-means, EMD on equity distribution over turn+river run-outs | `buckets[1]` | table: 1,286,792 canonical hands |
| turn | k-means, EMD on equity distribution over river cards | `buckets[2]` | table: 55,190,538 canonical hands |
| river | 1-D k-means on plain equity | `buckets[3]` | **on the fly**: equity → threshold search |

Details:

- **Equity** = probability of beating one uniformly random opponent hand at showdown, ties
  counting half, with card removal. Computed per board by evaluating all 1,081 hole-card
  combos, sorting by strength, and using per-card prefix counts so each hand's win/tie count
  excludes opponent hands that share a card. That's O(1,081) per board after the sort.
- **Canonical hands** use suit isomorphism (Waugh 2013 indexing). Bucket ids are u8 when
  the bucket count is at most 256, otherwise u16.
- **Flop/turn features.**
  - Turn: the 46 river equities of a hand, sorted, quantised to u8. For two equal-size
    samples in 1-D, EMD is exactly the mean |aᵢ − bᵢ| of the sorted values. That makes the
    distance a SIMD-friendly sum of absolute differences, with no histogram binning choice.
  - Flop: the 1,081 turn+river run-out equities, reduced to 50 quantiles (u8). Same distance.
  - Centroid update: the per-coordinate median of sorted features, which minimises L1. The mean
    is an acceptable approximation if the median proves slow.
- **River**: 1-D k-means on equity is equivalent to picking k−1 equity thresholds. So the
  river bucket needs no table, only k−1 numbers plus an equity computation:
  - 1,081 evaluations per river board per MCCFR iteration, shared by all 6 players.
  - Estimated 5–20 µs per iteration (unmeasured).
  - A 2.43e9-entry river table (2.26 GiB at u8) is the alternative; it does not fit the 11 GiB
    budget. `ByteModel::river_table` prices it.
- **Determinism**: k-means++ seeding with a fixed seed. Clustering runs on a fixed-seed
  subsample, then one full assignment pass. Bucket tables are written with a checksum and
  the config hash in the header.
- **Pluribus reference (unverified)**: 169 preflop, 200 buckets per post-flop street in the
  blueprint, k-means on equity distributions.

## 3. Action abstraction

All sizes are pot fractions: `raise_to = max_bet + f × (pot + to_call)`. So f = 1 is a
pot-sized bet or raise, matching `bet_frac_pot` in `analysis/m0_parse.py`. Rules:

- Sizes below the min-raise are dropped, and sizes at or above the stack become all-in.
- All-in is always available (`allin = true`).
- Raise depth d = number of voluntary bets/raises already made on the street. Depths with
  no listed sizes allow all-in only.

Starting point: Pluribus' observed sizes, from `docs/m0_stats.md` split by depth.

| street | depth 0 (open / first bet) | depth 1 (raise) | depth 2 |
|---|---|---|---|
| preflop | 0.4 (2 bb) 572, 0.5 (2.25 bb) 469, 0.6 224, 0.75 106, 1.0 32 | 2.0 67, 1.5 39, 1.0 38, 0.75 36 | ~0.5–1.0 (n = 53) |
| flop | 0.5 380, 1.0 111, 0.25 51, 2.0 27 | 0.5 33, 1.0 14 | rare |
| turn | 0.5 130, 0.25 115, 1.0 60, 1.5 28, 2.0 22 | 0.5 14, 1.0 5 | rare |
| river | 0.5 117, 0.75 43, 1.0 42, 1.5 25, 2.0 14, 3.0 8 | 0.5 10, 1.0 8, 2.0 6, 0.75 5 | rare |

Other evidence used for pruning:

- **Pluribus essentially never open-limps except from the SB**: 123 SB completes, 3 limps
  elsewhere in 10,000 hands.
- **Its flops are 84% heads-up, 14% 3-way, 1% 4-way.**

### Config levers (all in the `.cfg` file)

| key | meaning |
|---|---|
| `buckets = 169, F, T, R` | bucket counts per street |
| `<street>.<depth> = f1, f2, …` | heads-up size set |
| `multiway.<street>.<depth> = …` | post-flop override when ≥ 3 players are in; unset depth falls back to heads-up; empty = all-in only |
| `limp = none \| sb \| all` | who may flat-call the BB in an unraised pot (BB check always allowed) |
| `crowded_players = N` | post-flop with ≥ N players in: no sized bets (check/call/fold/all-in) |
| `allin = true/false` | all-in always available |
| `avg_strategy_streets = preflop` | streets that store an average strategy |

### What drives size

The exploration runs below use 200 buckets per post-flop street.

| action set | limp = all | limp = sb | limp = sb, crowded ≥ 4 | limp = sb, crowded ≥ 3 |
|---|---:|---:|---:|---:|
| compact (1 open, 1 3-bet, 2 post-flop sizes) | 22.7 GiB | 4.1 GiB | 0.56 GiB | 0.20 GiB |
| medium (3 opens, 2 3-bets, 3 sizes) | 130.6 GiB | 26.5 GiB | 6.3 GiB | 3.8 GiB |
| wide (≈ all observed sizes) | > memo cap | 145.1 GiB | 51.6 GiB | 30.7 GiB |

1. **Limping and crowded multiway pots dominate.** They multiply distinct flop situations.
   Restricting them as Pluribus actually plays costs little fidelity.
2. **Preflop breadth is the next driver.** Every distinct preflop line spawns its own
   post-flop tree. Going from 3 opens × 2 three-bets to 4 × 3 roughly doubles memory.
3. **Extra post-flop sizes are cheap by comparison.** Adding a 2× flop bet and a 1.5× river
   bet costs +0.24 GiB.
4. **River nodes are ~60% of all (infoset, action) pairs** (60.6% in config A, 60.1% in B). This fits Pluribus' design:
   a coarse late-street blueprint, refined by real-time search (M4).

## 4. Memory estimator

Code: `engine/nlhe/src/estimate.rs`, `engine/nlhe/src/betting.rs`.

- The betting tree is counted exactly by DFS, memoising only street-start states. That's
  where different histories converge, so the memo holds ≤ ~1M states (~150 MB). There's a
  cap of 4M states.
- It counts decision nodes and the sum of legal actions per street. Then:

```
pairs_s   = actions_s × buckets_s
bytes     = Σ_s pairs_s × 4                (i32 regrets, all streets)
          + Σ_{s ∈ avg streets} pairs_s × 4 (u32 average-strategy counts; Pluribus: preflop only)
          + nodes × 8                      (offset index into flat arrays)
          + flop/turn bucket tables        (canonical hands × 1 or 2 bytes)
```

- Excluded: the OS and other processes, per-thread scratch (small), and the one-off memory
  needed to build the card abstraction (section 5).
- **Flat layout:** `regrets[offset[node] + bucket × n_actions + a]`. There's no hash map at
  training time, so no per-infoset overhead.

Tests: 17 unit tests and 2 integration tests. They cover:
- the config parser and its fallbacks
- the betting rules: blind order, BB option, min-raise drop, all-in collapse, limp
  options, raise reopening, all-in run-out termination
- memoised count equal to a naive DFS on 3- and 4-handed trees
- chip invariants on every path
- a hand-computed estimate, and bucket-id width
- a tight bucket-count solver
- the shipped configs, pinned to exact node and pair counts and checked to sit within 85–100%
  of their budget

### Config A: `m2_11g.cfg` (16 GB machine, ~11 GiB budget)

`limp = sb`, `crowded_players = 4`, multiway (3 players) post-flop sizes {0.5, 1.0} at depth
0 and all-in only above.

| street | sizes depth 0 | depth 1 | depth 2 |
|---|---|---|---|
| preflop | 0.4, 0.5, 0.75 (2, 2.25, 2.875 bb opens) | 1.0, 2.0 | 1.0 |
| flop | 0.25, 0.5, 1.0, 2.0 | 1.0 | all-in |
| turn | 0.25, 0.5, 1.0, 2.0 | 1.0 | all-in |
| river | 0.5, 0.75, 1.0, 1.5, 2.0 | 1.0 | all-in |

| street | betting nodes | avg actions | buckets | infosets | (infoset, action) pairs | regrets | avg strategy |
|---|---:|---:|---:|---:|---:|---:|---:|
| preflop | 171,960 | 2.15 | 169 | 29.1 M | 62.4 M | 0.23 GiB | 0.23 GiB |
| flop | 534,106 | 2.16 | 256 | 136.7 M | 295.2 M | 1.10 GiB | 0 |
| turn | 1,109,464 | 2.29 | 256 | 284.0 M | 650.4 M | 2.42 GiB | 0 |
| river | 2,610,654 | 2.32 | 256 | 668.3 M | 1,552.2 M | 5.78 GiB | 0 |
| **total** | 4,426,184 | | | **1.12 B** | **2.56 B** | | |

**9.86 GiB**, including 53 MiB of bucket tables and 33 MiB of node index.
- Within 11 GiB the same action set allows up to 286 equal post-flop buckets. 256 is kept
  for u8 bucket ids and ~1 GiB headroom.
- On a 16 GB machine (15.8 GiB usable) this leaves ~6 GiB for Windows, the build-time card abstraction (done
  beforehand, not concurrently), and the scheduled 22:45 task. That's tight but workable if
  nothing else large is running.

### Config B: `m2_26g.cfg` (32 GB machine, ~26 GiB budget)

Same structure, with a broader preflop and more raise options.

| street | sizes depth 0 | depth 1 | depth 2 |
|---|---|---|---|
| preflop | 0.4, 0.5, 0.6, 0.75 (2–2.875 bb opens) | 0.75, 1.0, 2.0 | 1.0 |
| flop | 0.25, 0.5, 1.0, 2.0 | 1.0 | all-in |
| turn | 0.25, 0.5, 1.0, 2.0 | 0.5, 1.0 | all-in |
| river | 0.5, 0.75, 1.0, 2.0 | 1.0 | all-in |

| street | betting nodes | avg actions | buckets | infosets | (infoset, action) pairs | regrets | avg strategy |
|---|---:|---:|---:|---:|---:|---:|---:|
| preflop | 378,794 | 2.15 | 169 | 64.0 M | 137.3 M | 0.51 GiB | 0.51 GiB |
| flop | 1,216,494 | 2.17 | 272 | 330.9 M | 719.6 M | 2.68 GiB | 0 |
| turn | 2,809,160 | 2.34 | 272 | 764.1 M | 1,787.0 M | 6.66 GiB | 0 |
| river | 6,411,798 | 2.29 | 272 | 1,744.0 M | 3,988.2 M | 14.86 GiB | 0 |
| **total** | 10,816,246 | | | **2.90 B** | **6.63 B** | | |

**25.40 GiB.** Up to 278 equal buckets fit in 26 GiB. Bucket ids are u16 above 256, costing
+55 MiB.

### Trade-off between the two

At equal memory you can buy buckets or actions. For example, the 11 GiB budget allows
config A's actions at 286 buckets, or config B's actions at only 110 buckets. Pluribus
(unverified) used ~200 post-flop buckets with a richer action set, on a 64-core server with
< 512 GB of RAM. Both configs here are deliberately far smaller. The design bet is
that M4 search compensates for a coarse blueprint, which is the paper's own argument.

## 5. Precomputed equity tables: sizes and build time

### Sizes

| artifact | entries | bytes | kept? |
|---|---:|---:|---|
| preflop class table | 1,326 combos → 169 | < 4 KiB | yes |
| flop bucket table | 1,286,792 | 1.2 MiB (u8) / 2.5 MiB (u16) | yes |
| turn bucket table | 55,190,538 | 53 MiB (u8) / 105 MiB (u16) | yes |
| river bucket thresholds | k − 1 floats | < 2 KiB | yes |
| river equity / bucket table (optional) | 2,428,287,420 | 2.26 GiB (u8), 4.5 GiB (u16) | no (computed on the fly) |
| flop features (build only) | 1.29 M × 50 B | 61 MiB | temporary |
| turn features (build only) | 55.2 M × 46 B | 2.4 GiB if materialised | streamed; k-means fit on a 5 M subsample (~220 MiB) |

### Work (hand-evaluator calls)

| step | boards | evals | with suit isomorphism |
|---|---:|---:|---:|
| river equities (all 5-card boards × 1,081 hands) | 2,598,960 | 2.8 × 10⁹ | ~20× fewer |
| turn features (270,725 turn boards × 48 rivers × 1,081) | 270,725 | 1.4 × 10¹⁰ | ~20× fewer |
| flop features (22,100 flops × 1,176 run-outs × 1,081) | 22,100 | 2.8 × 10¹⁰ | 2.2 × 10⁹ (1,755 canonical flops) |

### Build time on this CPU (i7-10750H, 6 cores / 12 threads): **estimate, unmeasured**

| step | estimate |
|---|---|
| Evaluator throughput assumed | 50 M evals/s per thread (pure-Rust lookup evaluator), ≈ 7.5 effective cores with HT → ~375 M evals/s. Fast table evaluators reach 100–300 M/s/thread, so this is deliberately conservative. |
| flop features | ~10 s (isomorphic) |
| turn features | ~40 s plain, a few seconds isomorphic; equity bookkeeping roughly doubles it |
| river threshold fit | ~10–20 s (one pass over river boards, equity histogram, 1-D k-means) |
| flop k-means (1.29 M × 256 centroids × 50 B × ~50 iterations) | ~1–5 min |
| turn k-means (fit on 5 M subsample × 256 × 46 B × ~30 iterations + one full 55 M assignment pass) | ~5–20 min |
| **total** | **~10–30 min**, dominated by turn clustering. Peak RAM ~3 GiB if turn features are materialised, < 1 GiB streamed. |

The first M2 implementation task is to write the evaluator and benchmark it here, then
replace these estimates with measured numbers. Laptop thermal throttling is the largest
unknown (section 6).

## 6. Throughput test plan: 6 vs 12 threads

**Question.** Hyper-threading helps when threads stall on memory. Blueprint MCCFR is
random-access over multi-GiB arrays, so it should be memory-latency bound; equity building
is compute bound. Measure both, don't assume.

### Workloads

1. **W1, equity build** (compute bound): flop-feature generation for a fixed set of 200
   canonical flops.
2. **W2, MCCFR, small table** (cache-friendly): 6-max with the `compact` action set and 8
   buckets. Tables of a few MiB.
3. **W3, MCCFR, large table** (memory bound): config A action set at 256 buckets, but with
   regrets for only a 2 GiB slice. The table is allocated lazily per touched node, so
   total allocation stays ≤ 2 GiB in the test.
4. **W4, correctness under threads**: Leduc with parallel MCCFR at 1/6/12 threads. Final
   exploitability must be statistically indistinguishable from single-threaded runs.

### Parallel scheme to test

Hogwild-style shared regret table, as Pluribus did with many threads, each thread running
its own traversals with an independent RNG stream (seed = base_seed + thread_id).

- Variant (a): relaxed atomic adds on i32.
- Variant (b): plain racy writes.

Pick (a) unless it costs > 10%. Determinism with > 1 thread is not possible, so W4 is the
correctness gate.

### Protocol

- AC power, Windows "Best performance" power mode. Nothing else heavy running.
- Not within 20 minutes of the 22:45 weekday scheduled task. Training process at
  BELOW_NORMAL priority so that task is never starved.
- Thread counts {1, 2, 4, 6, 8, 10, 12}, pinned (core-first: 0, 2, 4 … then siblings).
- Each point: 60 s warm-up, then 3 × 120 s measurements. Report the median, min and max of
  iterations/s (W2–W4) or boards/s (W1).
- Log CPU clock and package temperature once per second (`typeperf`, the Processor
  Information counters) to detect thermal throttling. Rerun any point whose median clock is
  > 10% below the 6-thread run's clock.

### Decision rule

- Use 12 threads if 12 beats 6 by ≥ 15% on W3 (the realistic case). Otherwise use 6, leaving
  headroom for the OS and the scheduled task.
- Report scaling efficiency (speed-up ÷ threads) for each workload.

### Output

`docs/m2_threads.md` with the table, the clock and temperature traces, and the decision.

## 7. Open items before M2 implementation

- **RAM decision (16 vs 32 GB)** picks config A or B. Both are ready as config files.
- **PDF check:**
  - Pluribus' bucket counts, action abstraction and regret storage
  - average strategy updated only on the first round (affects `avg_strategy_streets`)
  - regret floor
- **Not modelled:**
  - A short all-in raise reopening betting is a simplification (adds a few nodes).
  - The blueprint does not distinguish suits beyond isomorphism.
