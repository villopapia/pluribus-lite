# Overnight report: 2026-11-07 23:30 → 2026-11-08 00:50

All six items were attempted in order; five produced code and results, and item 1 is blocked.
The repo is clean at every commit: each commit was tested on exactly its staged tree
(108 Rust + 10 Python tests). Nothing touched Playground or the 22:45 task (it was Saturday
night anyway). No system settings were changed and nothing was installed. All long jobs ran at
below-normal priority; the longest single job was 25 min.

## Decisions I need from you

1. **Supplement PDF is still missing.** I checked:
   - `C:\Users\User\pluribus-lite\docs\` (any `*.pdf`, any case) at 23:3x and again at 00:45
   - the whole project tree (excluding the venv and the dataset clone)
   - `Downloads`, `Desktop`, `Desktop\poker` and `Documents` (two levels) for recent PDFs or
     names containing pluribus / supplement / brown_sandholm

   No `alg1_diff.md` was written and `mccfr.rs` is untouched. Everything below that depends on
   Algorithm 1 details uses my recollection, marked "unverified":
   - the pruning exceptions (behind `prune_skip_river_and_terminal`, default off)
   - floor −310M, C = −300M
   - average strategy on the preflop only
   - the 200 / 400 / 10-minute schedule
2. **River buckets: 247, not 256.** 1-D k-means merged 9 coincident centroids, because
   equity values bunch up at discrete points. Two ways forward:
   - (a) set the river to 247 in `m2_11g.cfg` (saves 3.5% of river regret memory)
   - (b) switch the river to equal-mass quantile cut-offs, giving exactly k buckets

   Recommendation: **(b)**. It is cheap, exact and still 1-D.
3. **Schedule planning for long runs.** I calibrate for 15 s on an empty table, which
   overestimates throughput ~2.5×, so prune/LCFR/discount events land later than intended.
   For the 16-day run I recommend fixing the schedule in iterations, using the throughput
   measured after a ~10-min warm-up. Alternatively use Pluribus' wall-clock minutes
   directly; both work with the existing code.
4. **`strategy_interval`.** The smoke run used 1,000. Pluribus used 10,000 (unverified). With
   ~25–35k it/s, 10,000 still gives ~3 preflop average updates per second per player, so I
   suggest **10,000** for the long run.
5. **Start the long M3 run on the Dell?** Not started, as instructed. At full m2_11g
   (256 buckets) the expected footprint is ~10.6 GB peak:
   - 10.1 GiB estimate, plus a 0.45 GB tree, minus tables already counted
   - checkpoints ~9.9 GiB each, ~20–60 s to write

   That fits 16 GB only if nothing else large runs. See the projection below.
6. **Disk.** The smoke checkpoints use 15 GB in `data/runs/smoke/` (9 × 1.63 GiB, needed
   only for further head-to-head runs). 157 GB is free. Delete them or keep them?
7. **Unexplained throughput gap.** Part A of the smoke run ran at ~25k it/s and part B at
   ~34k, on the same machine and settings. Worth a controlled re-measurement before
   projecting precisely; the projections below use the full range.

## What was done (commits)

| item | commit | result |
|---|---|---|
| 1 PDF diff | none | blocked: no PDF (paths above) |
| (previous ask) int regrets | `da47d69` | `cfr::intregret`: exact half-even discounting. No drift over 40 discount steps (mean error < 0.02 chips, max within the analytic bound); floor rounding drifts, which the test detects |
| 2 lookup evaluator | `6d66cc6` | `eval_lut.rs` matches the `eval.rs` oracle on **every** 5-, 6- and 7-card hand (133.8M sevens, a superset of the 10M sample). 148 M evals/s on 1 thread (2.35× the oracle). River equity: **17.8 µs/board** (was 33.2); MCCFR per-hero path **6.8 µs (1 hero), 8.5 (2), 12.5 (6)** |
| 3 switch + SLURM | `cb65e12` | `river_method = on_the_fly \| precomputed_table`; direct (hole, board) indexing, bijection-tested; estimator m2_11g = 10.11 GiB; `scripts/slurm_train.sh` (96 h, USR1 → stop file → checkpoint → resubmit); Linux rules in CLAUDE.md |
| 4 card abstraction | `62370e2` | m2_11g built in **410.7 s** at 12 threads (river 9.6 s, flop 135 s, turn 266 s), peak 593 MB. Flop/turn 256 buckets, none empty; sizes 0.19–2.44× mean (flop), 0.12–2.94× (turn). Details: `docs/m2_abstraction_build.md` |
| 5 trainer + smoke | `384bfd7` | 82.5M iterations in 44.1 min at 32 buckets; details below and in `docs/m3_smoke.md` |
| 6 this report | (next commit) | |

## Smoke test (m2_11g tree, 32 post-flop buckets, 12 threads)

| | |
|---|---|
| tree | 10.1M nodes (4.43M decision); 1.40 GiB regrets + 0.23 GiB preflop averages |
| iterations/s | 23–29k (part A), 31–36k (part B); bench at the same checkpoint: **6 threads 19.6k, 12 threads 25.2k (1.29×)** → use 12 |
| river share of iteration time | 25% at the start → **~20%** |
| peak RAM | **2.45 GB** (4.08 GB with the throwaway calibration table) |
| checkpoint | **1.63 GiB**, 3–10 s, checksummed, atomic |
| resume | part B resumed exactly at iteration 31,776,200. A bit-identical resume is proven by a single-thread unit test |
| regrets | 0 saturations; floor clamps start at ~79M iterations; pruning starts at ~50M iterations (0.6 pruned actions/iteration by the end) |
| strategy change per minute (L1, 50k sampled infosets) | preflop 0.352 → 0.0075 (÷47), post-flop 0.153 → 0.0096 (÷16), still falling |
| infosets touched after 82M iterations | preflop 27%, flop 31%, turn 34%, river 41% |
| head-to-head (duplicate, 1.2M hands each) | 44 min vs 5 min: **+360 ± 16 mbb/hand**; 44 vs 20 min: **+110 ± 10**; 20 vs 5 min: +293 ± 16 (monotone; the gain per doubling is shrinking) |

## Projections (speculative: read the assumptions)

### (a) 16 days on this Dell, full m2_11g (256 buckets)

- **Throughput:** 18–30k it/s. The 32-bucket table (1.6 GiB) is already far larger than the
  12 MB cache, so moving to 10 GiB should cost mainly extra TLB misses. That's a modest
  slowdown (my estimate, not measured). It must stay at below-normal priority alongside
  normal use.
- **Iterations:** 1.38M s × 18–30k = **2.5–4.1 × 10¹⁰** (central ~3.3 × 10¹⁰). That is
  ~400× the smoke run, spread over 8× more post-flop infosets, so ~50× more visits per
  infoset than at the end of the smoke run.
- **Coverage:** at 32 buckets, 41% of river infosets were touched after 82M iterations.
  Rarely reached lines (deep raises) will remain thin even after 16 days. Good for the main
  lines, weak in the tails; M4 search is meant to compensate.
- **Convergence (power-law extrapolation, a big if):** post-flop strategy change per minute
  fell as ~t^−0.7. Extrapolating, with the 8× larger infoset count, gives a per-minute change
  of ~5 × 10⁻⁴ after 16 days. That's a stability indicator, not a strength guarantee. Real
  strength needs M4: local best response, or evaluation against the Pluribus hands.
- **Compute comparison:** 16 days × 6 cores ≈ 2,300 core-hours, against Pluribus' reported
  ~12,400 core-hours (unverified), on a far smaller abstraction.

### (b) 8 days on one 64-core / 251 GB HPC node, larger config

- **Throughput:** measured here, 4.2k it/s per physical core (12 threads on 6 cores).
  Assuming 3–4.5k per server core × 64 gives 190–290k it/s. Two caveats:
  - NUMA on a 2-socket node could cost 10–30% unless threads and memory are bound per socket.
  - Using `precomputed_table` for the river removes ~20% of iteration time (×1.25).

  Net **~200–350k it/s**.
- **Iterations:** 691k s × 200–350k = **1.4–2.4 × 10¹¹**, about 4–7× the Dell's 16-day run.
- **Config options** (estimator, all with `limp = sb`, crowded ≥ 4):

  | config | memory |
  |---|---:|
  | m2_26g actions, 272 buckets | 25.9 GiB |
  | m2_26g actions, 400 buckets | 37.3 GiB (42.5 with river table) |
  | m2_26g actions, 600 buckets, river table | 60.3 GiB |
  | "wide" action set (≈ all observed Pluribus sizes), 200 buckets | 51.9 GiB |

- **Recommendation:** m2_26g actions at **~400 buckets with the river table (42.5 GiB)**. It
  roughly matches the Dell run's visits per infoset while adding preflop breadth. Memory
  isn't the constraint there; iterations per infoset are. Going to 600+ buckets or the wide
  action set would leave most infosets under-visited in 8 days.
- **Resubmission:** the SLURM stub chains 96 h jobs; 8 days = 2 jobs.

## Known limitations / follow-ups

- **Card abstraction:** k-medians hit the 30-iteration cap with 0.3–0.5% of points still
  moving. Raising the cap is cheap, since flop/turn fitting took only 11 s and 23 s.
- **Faster turn assignment, if rebuilt often:** exact sum-bound pruning in nearest-centroid
  search (described in `docs/m2_design.md` §8).
- **Head-to-head is self-play between checkpoints.** It shows learning, not strength.
- **Multi-threaded training is not bit-reproducible** (lock-free updates, by design).
