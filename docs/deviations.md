# Deviations ledger

Every place where pluribus-lite differs from Pluribus as published. The authoritative list
of published parameters is `docs/fidelity_spec.md` (with sources and pages).

**Columns:**
- item | paper | ours | reason | expected impact | how we measure it | status

**Reason codes:**
- **unpublished**: the paper does not specify it, so we had to choose
- **resource**: hardware or time limit
- **impl**: implementation detail with no intended strategic effect
- **pilot**: option used only in pilot configs, off in `faithful.cfg`
- **data**: data availability

**Status:**
- **open**: affects results; not yet measured
- **measured**: number below
- **tested**: covered by an automated test
- **pending**: component not built yet (M4)

Page numbers: Supp = supplement (printed pages), Main = Science 365:885–890.

## A. Action abstraction

| id | item | paper | ours | reason | expected impact | how we measure it | status |
|---|---|---|---|---|---|---|---|
| D-ACT-1 | preflop raise sizes | 1–14 per decision point, "decided by hand based on what raise sizes earlier versions of Pluribus decided to use with significant positive probabilities"; **sets not published** (Supp p. 12) | Same method on the only available evidence, Pluribus' own logged sizes (`analysis/calibrate_sizes.py`): snap to a pot-fraction grid, keep sizes with share ≥ θ per raise depth (always the mode). **θ = 0.14 chosen so the tree has 718,732,473 action sequences, 1.08× the paper's 664,845,654** (Supp p. 13). Result: open {0.4, 0.5, 0.6}; 3-bet {0.88, 1, 1.5, 2}; 4-bet {0.5, 0.75, 0.88, 1}; then {1} | unpublished | Different preflop tree from Pluribus' unknown one; matched in total size, not in shape | Calibration table in `docs/fidelity_report.md` (θ 0.02–0.30 → 0.18B–5.68B sequences); sensitivity: head-to-head θ=0.14 vs θ=0.142 on a reduced config | measured (size) / open (strength) |
| D-ACT-2 | flop raise sizes | "more coarse" than preflop; not published (Supp p. 12) | Same observed-size rule: first bet {0.5, 1}, first raise {0.5, 1}, then {1} | unpublished | Logged flop sizes come from **real-time search** (search is always used post-flop, Supp p. 19), so they are a weaker proxy for blueprint sizes than the preflop ones | Count flop share of sequences; compare with logged search sizes once M4 exists | open |
| D-ACT-3 | sizes at raise depths without evidence | not published | preflop ≥ 4th raise and flop ≥ 3rd raise: {1×} + all-in (mirrors the published turn/river later-raise rule) | unpublished | Small: rare lines | Share of sequences at those depths | open |
| D-ACT-4 | turn/river sizes | "**at most** three" for the first raise {0.5×, 1×, all-in}, "at most two" later {1×, all-in} (Supp p. 12) | Exactly these sets at every turn/river node (config test `faithful_config_matches_the_paper_structure`) | unpublished (where "at most" binds) | If Pluribus used fewer somewhere, our post-flop tree is larger; absorbed by the θ calibration | Part of D-ACT-1 calibration | tested |
| D-ACT-5 | meaning of "fraction of the pot" | not defined (Supp p. 12) | raise-to = max bet + f × (pot + amount to call), i.e. pot **after calling** (standard "pot-sized raise"); bets: f × pot | unpublished | Changes absolute raise sizes (not bets) | Re-run calibration under the "pot before call" convention; compare sequence counts and log-likelihood (eval plan §1.2) | open |
| D-ACT-6 | abstract size below the min-raise | not described | dropped, not clamped | unpublished | Fewer raise options in small pots | Count dropped sizes in the tree | open |
| D-ACT-7 | short all-in re-opening | NLHE rules (Supp p. 4): a raise must be ≥ the previous raise | our betting model lets a short all-in reopen betting | impl | Slightly larger tree; rare with equal 100 bb stacks | Implement exact rule; diff sequence counts | open |
| D-ACT-8 | fold and call | always legal (Supp p. 13) | `faithful.cfg`: `limp = all`, `crowded_players = 0`. The `limp`/`crowded_players` options exist **only** for pilots and are labelled so | pilot | none in faithful | config test | tested |

## B. Information abstraction

| id | item | paper | ours | reason | expected impact | how we measure it | status |
|---|---|---|---|---|---|---|---|
| D-INFO-1 | flop/turn features | 200 buckets per round, "k-means clustering on domain-specific features [26]" (Supp p. 13); features not given. [26] = Johanson et al. 2013: hand-strength histograms + EMD, k-means, k-means++, multiple restarts | Hand-strength histograms (**50 equal bins** of river equity vs a random hand; flop over all 1,081 run-outs, turn over 46 rivers), EMD as L1 between CDFs, **k-means** (mean centroids), k-means++, **one run**, fitted on **1,000 random boards** (1.18M / 1.13M points), run to convergence (<0.1% changes: 112 / 147 iterations), then every (hole, board) assigned (`abstraction.rs`, `kmeans.rs`) | unpublished (bin count, sample, restarts) | Feature family matches [26]; bucket boundaries differ from Pluribus' unknown ones | Mean EMD to centroid (flop 0.900, turn 1.060 bins); bucket-size spread (flop 0.10–2.67×, turn 0.09–3.60× mean); restart-to-restart agreement (adjusted Rand index) | measured / open |
| D-INFO-2 | river feature | not given; [26]'s best river abstraction is **OCHS** (k-means OCHS, L2) with 8 opponent clusters | **OCHS with Johanson et al.'s published 8-cluster partition (their Table 1, transcribed and tested)**; k-means L2, 200 buckets, fitted on 1,000 river boards (42 iterations). River buckets are computed during training from the board (no table) | unpublished | Matches [26]. Cost: OCHS + nearest-centroid takes **~64% of iteration time** in the pilot smoke (vs ~20% for 1-D equity) | Bucket sizes on an independent sample (0.12–6.11× mean); head-to-head OCHS vs legacy 1-D river (`--river-method equity1d`) on a reduced config | measured / open |
| D-INFO-3 | k-means implementation details | [26]: triangle-inequality k-means, k-means++, restarts | exact nearest search pruned by coordinate-sum bounds; deterministic; single run | impl | none expected beyond restarts | `kmeans.rs` tests (pruned = brute force; thread-independent) | tested |
| D-INFO-4 | infoset identity | same action sequence + same bucket (Supp p. 12) | (betting history, current-street bucket): imperfect recall of earlier buckets | — | — | — | match |
| D-INFO-5 | table indexing | not described | direct (hole, board) indexing without suit isomorphism | impl | none (isomorphic hands get the same features up to sampling) | — | — |
| D-INFO-6 | search abstraction (500 buckets per round, per-flop, potential-aware [28]) | Supp p. 13, 22 | not built | — | — | — | pending (M4) |

## C. Blueprint algorithm

| id | item | paper | ours | reason | expected impact | how we measure it | status |
|---|---|---|---|---|---|---|---|
| D-SCHED-1 | time base of the schedule | wall-clock minutes on 64 cores: discount every 10 until 400, prune after 200, φ and snapshots from 800, snapshots every 200 (Supp pp. 14–15, Alg. 1); total 8 days = 11,520 min (Main p. 887) | **a "paper minute" = 1/11,520 of the planned run's iterations** (`Schedule::pluribus_scaled`); the plan = rate in the second half of a warm-up × planned duration | resource | Same schedule *shape* relative to training; absolute iterations per "minute" differ by orders of magnitude | Sensitivity runs with all fractions ×0.5 and ×2 (reduced config); compare proxies (eval plan §5) | open |
| D-SCHED-2 | rate used for the plan | — | one measurement after warm-up. Measured: warm-up rates overestimate sustained throughput, because iterations get **more** expensive as strategies sharpen (smoke runs: nodes/iteration 390 → 540 and 470 → 1,530; river-reach 0.65 → 0.94 and 0.40 → 0.79) | resource | Events fire later in wall time than planned (all still fire) | event iterations vs wall time in logs | measured |
| D-SCHED-3 | discount steps inside the warm-up | — | boundaries passed during the warm-up are skipped and logged (0 for a 16-day plan with a 10-min warm-up; 5–11 in the short smoke tests) | impl | none for long runs | `schedule.txt` | measured |
| D-ALG-1 | strategy interval | 10,000 iterations (Alg. 1) | 10,000 | — | — | — | match |
| D-ALG-2 | pruning | C = −300,000,000; 95% pruned / 5% full iterations; decision per iteration; not on the final round or for actions leading to a terminal node (Supp pp. 3, 14; Alg. 1) | same | — | — | tests `pruning_exception_spares_river_and_terminal_actions`, M1 `paper_pruning_exception…` | tested |
| D-ALG-3 | regret storage, floor | 4-byte ints, floor −310,000,000 (Supp p. 15) | i32, floor −310M, saturating add | — | — | `cfr::intregret` tests | tested |
| D-ALG-4 | rounding of integer updates and discounting | not stated | exact round-half-to-even | unpublished | none measurable (no drift) | `repeated_discounting_does_not_drift` | tested |
| D-ALG-5 | preflop average-strategy start | Alg. 1 initialises φ at t = 0; text: average stored "after the initial 800 minutes" (Supp p. 15) | text version: φ sampling from paper-minute 800 (`avg_start`) | unpublished (ambiguous) | Excludes early noisy strategies | φ from 0 vs from 800 on a reduced run | open |
| D-ALG-6 | post-flop blueprint | current-strategy snapshots every 200 min after 800, averaged, stored on disk (Supp p. 15) | same schedule; snapshots summed in one streamed on-disk file (u16-quantised probabilities; per-node counts; a node first reached later counts as uniform for earlier snapshots) | unpublished (weighting, storage) | small | `snapshot_accumulator_sums_quantised_current_strategies` | tested |
| D-ALG-7 | first-use allocation | allocate an action sequence's regrets when first encountered, except round 1; "> 2×" saving (Supp pp. 3–4, 13) | allocated when the **traverser** first updates a node (opponent visits read zeros = uniform without allocating) | unpublished ("encountered") | memory only; values identical to dense (test) | `first_use_allocation_saves_memory_and_matches_dense`; runs log `alloc_frac_of_dense`, `seq_reached` | tested / measured |
| D-ALG-8 | "P_i not in hand → TRAVERSE(h·0)" | Alg. 1 | return P_i's fixed payoff | impl | none | payoff tests | tested |
| D-ALG-9 | "actions leading to a terminal node" | Supp pp. 3, 14 | literal: the hand ends. The traverser's fold in a multiway pot is **not** exempt | unpublished (ambiguous) | slightly more pruning | switchable; compare pruned-action rate | open |
| D-ALG-10 | parallelism | not described | 12 threads, lock-free relaxed atomics (rare lost updates) | unpublished | negligible; runs with > 1 thread are not bit-reproducible | single-thread bit-reproducibility test | tested |
| D-ALG-11 | chance sampling | inside each traversal | one full deal per (iteration, traverser) | — | equivalent | — | match |

## D. Resources and configs

| id | item | paper | ours | reason | expected impact | how we measure it | status |
|---|---|---|---|---|---|---|---|
| D-RES-1 | hardware, run length | 64 cores, < 512 GB, 8 days, 12,400 core-hours (Main p. 887) | Dell: 6 cores / 12 threads, 16 GB (~5 GB free in normal use) | resource | far fewer iterations | iterations logged | measured |
| D-RES-2 | **faithful config memory** | < 0.5 TB (Supp p. 11) | `faithful.cfg`: dense 548 GiB; with first-use allocation at Pluribus' encountered share 0.622 + explicit tree **380.6 GiB** (fits 0.5 TB). **Does not fit a 251 GB node**; needs ≥ ~400 GB | resource | HPC run needs a large-memory node, or a smaller faithful variant (e.g. θ = 0.15: 419M sequences, 218 GiB) | estimator | measured |
| D-RES-3 | **pilot_dell reductions** (from `faithful.cfg`; structure otherwise identical) | — | preflop open {0.4, 0.5, 0.6} → {0.5}; 3-bet {0.88, 1, 1.5, 2} → all-in only; 4-bet {0.5, 0.75, 0.88, 1} → all-in only; 5-bet+ {1} → all-in only; flop bet {0.5, 1} → {0.5}; flop raise {0.5, 1} → all-in only; flop later {1} → all-in only. **Result: 55.7M sequences (0.08×); 28.6 GiB with first-use allocation, so it does not fit 11 GiB at 200 buckets** | resource | coarser preflop/flop play | estimator | measured |
| D-RES-4 | pilot within 11 GiB | — | `pilot_dell_fit11.cfg`: pilot_dell with **63** post-flop buckets (10.93 GiB) | resource | coarser cards on every post-flop round | estimator; head-to-head vs pilot_dell at equal iterations if RAM allows | measured / open |
| D-RES-5 | tree representation | not described | explicit tree, ~44 B per node incl. terminals (faithful: 718.7M nodes ≈ 29.5 GiB) | impl | memory | estimator | measured |

| D-RUN-1 | **Dell long run (pilot only)** | 8 days, 64 cores, `faithful`-scale abstraction | `pilot_dell_fit11.cfg` (63 post-flop buckets from a dedicated k = 63 Johanson abstraction, `data/abstraction/j63`), 12 threads at below-normal priority, plan 16 days (23,040 min, so paper minute = 2 min), river on the fly. Started 2026-11-08 11:50 as pid 27816; details in `docs/HANDOFF.md` | resource / pilot | Not a fidelity result: reduced action set (D-RES-3) and buckets (D-RES-4). For pipeline validation and pilot-vs-faithful comparisons only | metrics.tsv; eval plan §1–2 | running |

## E. Toy-game reference solver (M1, not the blueprint)

| id | item | paper | ours | reason |
|---|---|---|---|---|
| D-M1-1 | φ rounds | round 1 only | Kuhn/Leduc average all rounds (literal rule leaves Leduc ~1 chip exploitable; test `paper_avg_rule_on_leduc_needs_postflop_averaging`) | impl |
| D-M1-2 | C, floor units | chips at 100 bb | scaled by max\|utility\| / 10,000 | impl |
| D-M1-3 | schedule | minutes | fractions of the run | resource |
| D-M1-4 | regrets | int32 | f64 (reference) | impl |

## F. Data and evaluation

| id | item | paper | ours | reason | how we measure it | status |
|---|---|---|---|---|---|---|
| D-DATA-1 | original logs (Data File S1) | Supp: "Data File S1 as zipped archive" | `uoftcprg/phh-dataset` conversion. **The hand-by-hand cross-check is not done**: science.org answers automated downloads with a Cloudflare challenge (HTTP 403) | data | Manual browser download into `data/`; then a hand-by-hand diff (actions, cards, payoffs) | pending |
| D-DATA-2 | half-chip payoffs | logs | 4 PHH hands have half-chip split results; pokerkit rounds them | data | `m0_parse.py` uses logged values | measured |
| D-EVAL-1 | win-rate measurement | AIVAT, one-tailed t-test (Main p. 889) | not implemented yet | — | `docs/eval_plan.md` §3 | pending |
