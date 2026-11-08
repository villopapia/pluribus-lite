# Fidelity report (2026-11-08)

What changed to make pluribus-lite follow Pluribus as published, the calibration numbers, and
the open questions. Companion documents:
- `docs/fidelity_spec.md`: every published parameter, with source and page
- `docs/deviations.md`: every difference, with reason, expected impact and measurement
- `docs/eval_plan.md`: how fidelity and strength will be evaluated

No long runs were started.

## 1. Sources (Phase A)

| source | status |
|---|---|
| Supplement, main paper | downloaded to `docs/papers/`; read |
| Brown PhD thesis | **exists**: *Equilibrium Finding for Large Adversarial Imperfect-Information Games*, CMU-CS-20-132, Sept 2020. **§6.6 covers Pluribus** (pp. 189–192); it restates the paper and adds p-values 0.028 / 0.014; no new blueprint or search parameters |
| Johanson 2013 [26], Brown & Sandholm AAAI 2019 [38], Brown-Sandholm-Amos NeurIPS 2018 [41], Ganzfried-Sandholm IJCAI 2013 [39], Brown-Ganzfried-Sandholm AAMAS 2015 [28], Burch et al. AAAI 2018 [44] | all downloaded (SHA-256 in `fidelity_spec.md` §0) |
| **Data File S1** (original logs) | **not retrieved**: science.org serves a Cloudflare challenge (HTTP 403) to automated clients. **The hand-by-hand cross-check of our 10,000 PHH hands is therefore still pending** (D-DATA-1). Needs a manual browser download |

New facts taken into the spec from the main paper:
- 8 days on 64 cores, 12,400 core-hours, < 512 GB (Main p. 887).
- The abstraction was sized for live play in 128 GB with a compressed blueprint (Main p. 888).
- Search takes 1–33 s per subgame, ~20 s per hand (Main p. 889).
- Evaluation: AIVAT and a one-tailed t-test.

## 2. Code changes (Phase C)

| # | change | where | tests |
|---|---|---|---|
| 4 | **Johanson 2013 features.** Flop/turn: 50-bin hand-strength histograms, EMD (as L1 on CDFs), k-means with mean centroids, k-means++, run to convergence. River: **OCHS** with Johanson et al.'s published 8-cluster table, k-means L2. Equal-mass cut-offs removed; the 1-D equity river is kept only as a labelled comparison (`--river-method equity1d`) | `kmeans.rs`, `ochs.rs`, `abstraction.rs`, `buckets.rs`, `build_abstraction` | pruned nearest = brute force; thread-independent k-means; OCHS = brute-force enumeration; histogram features = direct computation; Table 1 spot checks |
| 5 | turn/river raises {0.5×, 1×, all-in} then {1×, all-in} (repeat syntax `turn.1+`) | `config.rs`, `faithful.cfg` | `faithful_config_matches_the_paper_structure` |
| 6 | fold/call always legal: `limp = all`, no multiway restriction in `faithful.cfg`; restrictions remain as labelled pilot-only options | configs | same |
| 7 | **first-use allocation** (except preflop), reached-sequence counters (`seq_reached`) in the metrics | `blueprint.rs`, `train_blueprint` | `first_use_allocation_saves_memory_and_matches_dense` (identical values to dense) |
| 8 | pruning exceptions on, per-iteration skip, C = −300M, floor −310M, int32 | `blueprint.rs`, `mccfr.rs` | pruning-exception and floor tests |
| 9 | preflop φ every 10,000 iterations from paper-minute 800; post-flop snapshots every 200 after 800, averaged on disk | `blueprint.rs` | snapshot accumulator test |
| 10 | paper minute = 1/11,520 of the planned run | `Schedule::pluribus_scaled`, `train_blueprint` | `paper_fraction_schedule_boundaries` |

## 3. Calibration (Phase D)

**Rule** (`analysis/calibrate_sizes.py`):
- Snap Pluribus' logged bets/raises to a pot-fraction grid.
- Per (street, raise depth), keep sizes with share ≥ θ, always keeping the mode.
- Turn/river follow the paper's rule.

Tree size by θ (`m2_estimate`; action sequences = (history, action) pairs, the paper's unit):

| θ | action sequences | × Pluribus (664,845,654) | first-use memory (0.622 + tree) |
|---|---:|---:|---:|
| 0.02 | 5,680,153,231 | 8.54 | 2,959 GiB |
| 0.05 | 2,999,846,709 | 4.51 | 1,567 GiB |
| 0.10 | 1,049,677,353 | 1.58 | — |
| 0.12 | 937,650,987 | 1.41 | 497.4 GiB |
| **0.14** | **718,732,473** | **1.08** | **380.6 GiB** |
| 0.142 | 585,138,185 | 0.88 | 308.0 GiB |
| 0.15 | 418,729,861 | 0.63 | 218.4 GiB |
| 0.20 | 207,769,269 | 0.31 | 106.6 GiB |
| 0.30 | 182,673,081 | 0.27 | 93.8 GiB |

The count is a step function of θ, because only crossing an observed share changes a size set.
**θ = 0.14 is the closest (+8.1%) and is `faithful.cfg`.** As a consistency check, its memory
with first-use allocation (380.6 GiB) lands under the paper's "less than 0.5 TB".

### `faithful.cfg`

| | |
|---|---|
| decision nodes | 340,738,096 (preflop 8.8M, flop 39.0M, turn 96.1M, river 196.9M) |
| action sequences | 718,732,473 |
| infosets | 67.9 billion |
| (infoset, action) pairs | 143.2 billion |
| memory, dense | 548 GiB |
| memory, first-use allocation (Pluribus' 0.622) + explicit tree | **380.6 GiB** |
| **checkpoint size** (allocated regrets + preflop φ + node ids) | **≈ 349 GiB** |
| post-flop snapshot sum file | ≈ 326 GiB; **≈ 650 GiB transiently** while it is rewritten |
| abstraction tables | 0.32 GiB (flop 24.8 MiB, turn 291 MiB; river computed on the fly) |
| **disk for the HPC run** | keeping 2 checkpoints + one being written: 3 × 349 ≈ 1.05 TiB, + snapshots up to 650 GiB → **≈ 1.7 TiB peak** (keeping 3 checkpoints: ≈ 2.0 TiB) |
| checkpoint write time | ≈ 6 min at 1 GB/s, ≈ 20 min at 300 MB/s (streamed; no extra RAM, verified) |

All memory and disk figures scale with the true allocation share, which is only known after a
real run. Pluribus' share was 0.622; our short runs reach 0.08–0.69, depending on tree and
length.

### Pilot configs

| config | change from faithful | sequences | memory (first-use) | fits 11 GiB |
|---|---|---:|---:|---|
| `pilot_dell.cfg` | preflop: 1 open size {0.5}, all-in only above; flop: {0.5} then all-in; 200 buckets | 55,731,171 | 28.6 GiB | **no** |
| `pilot_dell_fit11.cfg` | as above, **63** post-flop buckets | 55,731,171 | 10.9 GiB | yes |

Even the minimal preflop/flop sets leave the paper's structure (limps, unrestricted multiway
pots, the turn/river rule, 200 buckets) at 28.6 GiB. **Reducing only preflop/flop sizes cannot
reach 11 GB.** Each reduction is logged in `deviations.md` D-RES-3/4.

## 4. Card abstraction build (200 buckets, Johanson features)

`data/abstraction/j200`, seed 1, 12 threads, below-normal priority, 188 s in total:

| stage | fit | converged after | bucket sizes |
|---|---|---|---|
| river (OCHS) | 1,000 boards / 1.08M points | 42 iterations | 200 non-empty; 0.12–6.11× mean (independent sample) |
| flop | 1,000 boards / 1.18M points; mean EMD 0.900 bins | 112 iterations | 200 non-empty; 0.10–2.67× mean (all 26M hands) |
| turn | 1,000 boards / 1.13M points; mean EMD 1.060 bins | 147 iterations | 200 non-empty; 0.09–3.60× mean (all 305M hands) |

## 5. Tests and smoke runs (step 6)

- **All tests pass:**
  - 45 `cfr` tests: Kuhn and Leduc convergence, Kuhn exploitability < 0.001, value −1/18; the
    paper's pruning exception and floor
  - 66 `nlhe` unit tests and 4 config tests
  - 11 Python tests
- **Mechanics smoke** (m2_11g tree, 32 buckets, 12 + 3 min): 31.4M iterations.
  - Resumed from a v3 checkpoint at iteration 25,985,612 with continuous metrics.
  - Checkpoints 1.14–1.22 GiB (dense would be 1.63), written in 2.7–3.3 s, with no RAM growth.
  - 69% of dense regret memory allocated by the end.
- **`pilot_dell.cfg` smoke** (new tables, coarsened to 8 buckets to fit free RAM; 6 min):
  21.0M iterations at 52–84k it/s, peak 3.2 GB private.
  - First-use allocation: **8.4% of action sequences reached**, 8.8% of dense regret memory.
  - Pruning active (1.45 pruned actions/iteration by the end); regret floor reached.
  - One post-flop snapshot written (0.9 s).
  - **The OCHS river took ~64% of iteration time** (1-D equity: ~20%).

## 6. Open questions for you

1. **Data File S1:** please download it manually from the Science supplementary page into
   `data/`. The hand-by-hand cross-check (actions, cards, payoffs) runs as soon as it is there.
2. **HPC memory:** `faithful.cfg` needs ≈ 381 GiB with first-use allocation (≈ 548 GiB dense),
   more than the 251 GB node mentioned earlier. Options:
   - (a) a ≥ 512 GB node, as Pluribus used;
   - (b) θ = 0.15 (419M sequences, 0.63× Pluribus, ≈ 218 GiB);
   - (c) faithful sizes with fewer buckets.
3. **Dell pilot:** the paper's structure cannot fit 11 GB at 200 buckets. Options:
   - `pilot_dell_fit11.cfg` (63 buckets);
   - 32 GB of RAM for `pilot_dell.cfg` (28.6 GiB);
   - run pilots on the HPC node instead.
4. **River cost:** OCHS dominates iteration time on small trees. A precomputed OCHS bucket table
   (2.62 GiB at u8, `river_method = precomputed_table`) would remove it on the HPC node. It is
   not built yet (≈ 2.8×10⁹ features).
5. **Ambiguities kept as documented choices:**
   - the traverser's own fold vs "leads to a terminal node" (D-ALG-9);
   - φ from paper-minute 800 (text) vs t = 0 (Algorithm 1) (D-ALG-5);
   - "at most" turn/river sizes (D-ACT-4);
   - the meaning of "fraction of the pot" (D-ACT-5).
6. **Measured-but-open deviations need reduced-config experiments** (sensitivity of θ, OCHS vs
   1-D river, schedule fractions). Each takes hours of compute and was not started (no long runs).
