# Deviations ledger

Every place where pluribus-lite differs from Pluribus as published (`docs/fidelity_spec.md`).
The page convention is the same as there: printed page numbers.

**Reason codes:**
- **U** = unspecified in the paper; we had to choose
- **R** = resource limit (hardware, time)
- **I** = implementation choice with no intended strategic effect
- **P** = pilot-only option, off in `faithful.cfg`
- **X** = experiment-protocol difference

**Status:**
- **open** = affects results; needs a test or a decision
- **tested** = covered by a test
- **pending** = not built yet

Each entry gives the expected impact and how to test it.

## Game and action abstraction

| id | paper | ours | reason | expected impact | how to test | status |
|---|---|---|---|---|---|---|
| D-ACT-1 | Preflop raise sizes: 1–14 per decision point, chosen by hand from what earlier versions used (p. 12); **list not published** | `faithful.cfg`: the sizes Pluribus itself used in the 10,000 logged hands, snapped to a fixed pot-fraction grid, keeping sizes with ≥ 2% and ≥ 3 of the bets at that raise depth. Open {0.4, 0.5, 0.6, 0.75, 0.88, 1.0}, 3-bet {0.5, 0.75, 0.88, 1.0, 1.25, 1.5, 2.0}, 4-bet {0.5, 0.6, 0.75, 0.88, 1.0, 1.25} | U | Different preflop tree. Together with D-ACT-2..4 our faithful tree has **5.23 billion action sequences = 7.9× Pluribus' 664.8M** (p. 13), so our sets are richer than Pluribus' somewhere | Sequence-count calibration: shrink size sets until the count is within ±20% of 664.8M; sensitivity of head-to-head results to the size set | open |
| D-ACT-2 | Flop sizes: "more coarse" (p. 12); not published | Pluribus' observed flop sizes {0.25, 0.5, 1, 2} first bet, {0.5, 1} first raise | U | Weak proxy: the flop is **always played with real-time search** (p. 19), so logged flop sizes reflect the search abstraction, not the blueprint | As D-ACT-1 | open |
| D-ACT-3 | Raise depths with no evidence (preflop ≥ 4th raise, flop ≥ 3rd) | {1× pot} + all-in, mirroring the paper's later-raise rule for rounds 3–4 | U | Small (rare lines) | Count share of sequences at those depths | open |
| D-ACT-4 | Turn/river: "**at most** three" sizes for the first raise {0.5×, 1×, all-in}, "at most two" later {1×, all-in} (p. 12) | Exactly these sets at every turn/river decision point | U | Upper bound; Pluribus may have used fewer at some nodes → our tree is ≥ theirs | Part of the D-ACT-1 calibration | open |
| D-ACT-5 | "Fractions of the size of the pot" (p. 12); base not defined | Raise-to = max bet + f × (pot + amount to call), i.e. f of the pot **after calling** (standard "pot-sized raise") | U | Changes absolute sizes of raises (not bets) | Recompute the observed-size proxy under the other convention; compare counts | open |
| D-ACT-6 | Abstract size below the minimum legal raise: not described | Dropped (not clamped up to the min-raise) | U | Fewer actions in small pots | Count dropped actions | open |
| D-ACT-7 | A short all-in raise does **not** reopen betting in real NLHE (p. 4: later raises ≥ previous raise) | Our betting rules let a short all-in reopen | I | Slightly larger tree; rare (equal 100 bb stacks make short all-ins rare) | Implement the exact rule; count the difference | open |
| D-ACT-8 | Fold and call always legal (p. 13) | `limp` and `crowded_players` options exist; **`faithful.cfg` uses `limp = all`, `crowded_players = 0`** (no restriction) | P | None in the faithful config | Config test asserts the faithful settings | tested |

## Information abstraction

| id | paper | ours | reason | expected impact | how to test | status |
|---|---|---|---|---|---|---|
| D-INFO-1 | Post-flop: k-means on "domain-specific features" [26] (p. 13); features not published | Flop: 50 quantiles of the 1,081 run-out river equities. Turn: the 46 river equities, sorted. Clustering: **k-medians** with L1 distance (= 1-D EMD between sorted samples), k-means++ seeding, 1,000-board fit sample, 30 iterations | U, I | [26] used EMD on hand-strength histograms for flop/turn (k-means). Ours is the same family; the medoid update minimises L1 instead of the mean | Compare with histogram EMD + k-means on a small game (e.g. Leduc-sized deck) for abstraction quality (exploitability of the abstract equilibrium) | open |
| D-INFO-2 | River features: unpublished; [26]'s best river abstraction uses **OCHS** (equity against clusters of opponent hands) | 1-D equity against a uniformly random hand, k-means on the exact equity distribution | U | 1-D equity ignores **which** hands one beats; OCHS is finer. Our river buckets are coarser in that sense | Implement OCHS (8 opponent clusters) behind a flag; compare head-to-head | open |
| D-INFO-3 | 200 river buckets | 1-D k-means returned **193 distinct buckets** (7 centroids collided on discrete equity values) | I | 7 fewer river buckets than intended | Fixed by empty/duplicate-cluster reseeding (this session); test asserts exactly k | fixing |
| D-INFO-4 | Bucket assignment granularity | Direct (hole, board) tables without suit isomorphism | I | None strategically (same bucket for isomorphic hands up to k-means ties) | Isomorphism consistency test on a sample | open |
| D-INFO-5 | Search abstraction: 500 buckets per round, per-flop potential-aware EMD [28] (p. 13) | Not built (M4) | — | — | — | pending |

## Blueprint algorithm

| id | paper | ours | reason | expected impact | how to test | status |
|---|---|---|---|---|---|---|
| D-SCHED-1 | Schedule in **wall-clock minutes** on 64 cores: discount every 10 until 400, prune after 200, φ and snapshots from 800, snapshots every 200 (pp. 14–15) | A "paper minute" = **1/11,520 of the planned run's iterations** (11,520 min = 8 days, from the main paper, not the supplement). The plan is calibrated from the iteration rate in the second half of a 10-minute warm-up | R | Same schedule *shape* as a fraction of training. Absolute iteration counts differ by orders of magnitude, so "200 minutes" means far fewer iterations for us | Sensitivity: run with schedule fractions ×0.5 and ×2 on a reduced config; compare convergence proxies | open |
| D-SCHED-2 | Rate used to convert minutes | One calibration after warm-up. Measured: warm-up rates overestimate sustained throughput (92k vs ~40k it/s at 32 buckets), because iterations get **more expensive** as strategies sharpen (nodes/iteration 470 → 1,530, river-reach 0.40 → 0.79) | R | Schedule events fire **later in wall time** than planned (all still happen) | Log wall time of each event (now in `schedule.txt` / logs); compare with plan | open |
| D-ALG-1 | Strategy Interval 10,000 (Alg. 1) | 10,000 | — | — | — | match |
| D-ALG-2 | Pruning C = −300M, floor −310M, 95/5 split, per-iteration skip decision, exceptions for final round and terminal-leading actions (pp. 3, 14–15) | Same (`Schedule` defaults; tests `pruning_exception_spares_river_and_terminal_actions`, `regret_floor_is_respected`) | — | — | — | tested |
| D-ALG-3 | φ start: Alg. 1 initialises φ at t = 0; text says the first-round average was stored "after the initial 800 minutes" (p. 15) | Follow the text: preflop φ sampling starts at paper minute 800 (`avg_start`) | U | φ excludes the noisy early strategy (as presumably intended) | Compare φ from 0 vs from 800 on a reduced run | open |
| D-ALG-4 | Post-flop blueprint = average of current-strategy snapshots every 200 min after 800 (p. 15); weighting not stated | Uniform average of regret-matching snapshots, quantised to u16 per action. A node first allocated after some snapshots is counted as uniform for the snapshots it missed | U | Small | Test `snapshot_accumulator_sums_quantised_current_strategies` | tested |
| D-ALG-5 | 4-byte integer regrets (p. 15); rounding not stated | Exact round-half-to-even for updates and discounting | U | None measurable (no drift) | `cfr::intregret` tests | tested |
| D-ALG-6 | Parallelism not described | 12 threads, lock-free relaxed atomics; concurrent updates to the same cell can be lost | U | Negligible at our contention; not bit-reproducible with > 1 thread | Single-thread runs are bit-reproducible (test) | tested |
| D-ALG-7 | First-use allocation "when the action sequence was encountered" (pp. 3, 13) | Allocated when the **traverser first updates** a node; opponent-only visits read zeros (= uniform) without allocating | U | Memory only; no value difference (test shows identical regrets to dense allocation) | `first_use_allocation_saves_memory_and_matches_dense` | tested |
| D-ALG-8 | "P_i not in hand → TRAVERSE(h·0)" (Alg. 1) | Return P_i's fixed payoff immediately | I | None (same value, cheaper) | Payoff tests | tested |
| D-ALG-9 | Pruning exception "actions leading to a terminal node" | Literal: only actions after which **the hand ends**. A traverser fold in a multiway pot is not exempt | U | Slightly more pruning than the alternative reading | Switchable if needed; compare | open |
| D-ALG-10 | Chance sampling inside each traversal (Alg. 1) | One full deal per (iteration, traverser) | — | Equivalent | — | match |

## Toy-game solver (M1 reference, not the blueprint)

| id | paper | ours | reason | note |
|---|---|---|---|---|
| D-M1-1 | φ on round 1 only | Kuhn/Leduc average **all** rounds | I | The literal rule leaves Leduc ~1 chip exploitable (test `paper_avg_rule_on_leduc_needs_postflop_averaging`); M1 is a correctness reference |
| D-M1-2 | C and floor in chips at 100 bb | Scaled by max\|utility\|/10,000 | I | Toy games have different units |
| D-M1-3 | Schedule in minutes | Fractions of the run | R | Toy runs last seconds |
| D-M1-4 | Integer regrets | f64 | I | Reference solver; integer path tested separately |

## Resources

| id | paper | ours | reason | expected impact |
|---|---|---|---|---|
| D-RES-1 | One 64-core node, < 0.5 TB used; ~8 days | Dell laptop: 6 cores / 12 threads, 16 GB RAM (~5 GB actually available with normal desktop use) | R | Far fewer iterations; config must shrink (see D-RES-2) |
| D-RES-2 | 200 buckets, Pluribus' action abstraction | **No config with the paper's structure fits 11 GiB at 200 buckets.** Smallest tested (one open size, then all-in only; one flop size): 28.6 GiB with first-use allocation, or 63 buckets within 11 GiB | R | Pilot runs need fewer buckets, more RAM, or the HPC node. **Decision pending** |
| D-RES-3 | Tree representation unknown | Explicit tree, ~44 B per node (decision + terminal) | I | Memory overhead counted in the estimator |

## Data and evaluation

| id | paper | ours | reason | status |
|---|---|---|---|---|
| D-DATA-1 | Data File S1: the original 10,000-hand logs | Using the `uoftcprg/phh-dataset` conversion; **hand-by-hand cross-check against Data File S1 not done**: science.org returns a Cloudflare challenge to automated downloads | X | **pending: needs a manual browser download** |
| D-DATA-2 | Logs record whole chips | 4 hands in the PHH conversion contain half-chip split results (e.g. 137.5); pokerkit rounds them | X | documented in `analysis/m0_parse.py` |
| D-EVAL-1 | AIVAT adapted to > 2 players (pp. 7–9) | Raw bb/100 only so far | — | pending (M4) |
