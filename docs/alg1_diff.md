# Algorithm 1 vs our code: line-by-line differences

> **Superseded as the authoritative record by `docs/deviations.md`** (all deviations, with
> reasons and tests) and `docs/fidelity_spec.md` (all published parameters). This file keeps
> the original Algorithm 1 comparison. Page numbers here and there are the supplement's
> **printed** page numbers (PDF page = printed + 1).
>
> Since this comparison, the following changed:
> - First-use regret allocation is implemented (S10).
> - Preflop φ sampling starts at paper minute 800, following the text (A2 → D-ALG-3).
> - The schedule is defined as fractions of the planned run (D-SCHED-1).

Source: Brown & Sandholm, *Superhuman AI for multiplayer poker*, Science 2019, supplementary
materials (`docs/19-Science-Superhuman_Supp.pdf`). Sections used:
- "Equilibrium finding" (p. 3)
- "Hardware usage" (p. 11)
- "Further details of the abstraction algorithm" (pp. 12–13)
- "Further details of the blueprint computation algorithm" (pp. 14–15)
- Algorithm 1 (pp. 16–18)

Compared files: `engine/cfr/src/mccfr.rs` (M1, toy games) and `engine/nlhe/src/blueprint.rs`
plus `bin/train_blueprint.rs` (M3, 6-max).

Status legend: **match** = already as in the paper. **fix** = unambiguous difference, changed
behind a flag whose default is paper-faithful. **design** = deliberate difference, kept and
explained. **ambiguous** = needs your review.

## Constants (now verified)

| constant | paper | M1 `mccfr.rs` before | M3 before | status |
|---|---|---|---|---|
| prune threshold C | −300,000,000 (Alg. 1 TRAVERSE-MCCFR-P line 14; text p. 14) | scaled −300M × max\|u\| / 10,000 | −300,000,000 | **match** (M1 scaling kept for toy games) |
| regret floor | −310,000,000 "for every action" (p. 15) | none | −310,000,000 | **fix in M1** (`regret_floor`, scaled like C) |
| explore probability | 0.05 per iteration (Alg. 1 lines 13–14) | 0.05 | 0.05 | **match** |
| strategy interval | 10,000 iterations (Alg. 1 line 10) | 1 (toy) | CLI default 1,000 | **fix in M3** (default 10,000); M1 keeps 1, see D2 |
| prune start | 200 minutes (Alg. 1 line 12; p. 14) | fraction 200/11,520 of run | fraction 200/11,520 of a planned run | **fix in M3**: literal minutes via the post-warm-up rate |
| LCFR / discount stop | first 400 minutes (p. 14) | fraction | fraction | **fix in M3** (literal minutes) |
| discount interval | every 10 minutes (Alg. 1 line 21; p. 14) | fraction | fraction | **fix in M3** (literal minutes) |
| discount factor | d = (T/10) / (T/10 + 1), T = minutes elapsed (p. 14; Alg. 1 line 21) | k/(k+1), k = t / interval | same | **match** (k = number of completed 10-min periods) |
| discount applies to | regrets and average strategy φ (Alg. 1 lines 25–26) | both | both | **match** |
| average strategy (φ) | first betting round only (Alg. 1 lines 6–7, UPDATE-STRATEGY line 2) | all rounds (`avg_max_round = 255`) | preflop only | M3 **match**. M1 **design, kept all rounds**: measured on Leduc (1M iterations), the literal rule with the current strategy on round 2 is ~1.0 chip exploitable vs ~0.016 with all-round averaging (test `paper_avg_rule_on_leduc_needs_postflop_averaging`). Pluribus avoided this with post-flop snapshot averaging, which M3 now does |
| regret storage | 4-byte integers (p. 15) | f64 | i32 | M3 **match**; M1 f64 is **design** (reference solver) |
| post-flop blueprint | average of current-strategy snapshots taken every 200 min after the first 800 min (p. 15) | n/a | current strategy at the last checkpoint | **fix in M3** (snapshot accumulator, default on) |

## Algorithm structure

| # | paper | ours | status |
|---|---|---|---|
| S1 | Iteration t: for each player, UPDATE-STRATEGY if t mod SI = 0, then one traversal (Alg. 1 lines 9–19) | same order, both solvers | **match** |
| S2 | Prune decision drawn **once per iteration per traverser**, not per action (Alg. 1 lines 12–17; p. 3) | same | **match** |
| S3 | **Pruning exceptions:** never skip actions on the **final betting round**, or actions that **immediately lead to a terminal node** (pp. 3 and 14). Pseudocode line 14 omits this, but the text is explicit, twice | M1: no exception. M3: flag `prune_skip_river_and_terminal`, default **off** | **fix**: both solvers get the exception, default **on**. Note: Kuhn has one betting round, so with the exception Kuhn never prunes |
| S4 | Pruned traversal: v(h) sums only explored actions; only explored actions get regret updates (TRAVERSE-MCCFR-P lines 13–22) | same | **match** |
| S5 | CALCULATE-STRATEGY: regret matching, uniform when no positive regret | same | **match** |
| S6 | "P_i not in hand → TRAVERSE(h·0)" (lines 19–20): continue down the tree, P_i's utility unaffected | M1 n/a (2 players). M3 returns P_i's fixed payoff (−contribution) immediately | **match in value** (cheaper, identical result) |
| S7 | UPDATE-STRATEGY: return if terminal, P_i not in hand, or round > 0; P_i samples one action and increments φ by 1; other players: all actions (lines 1–15) | same | **match** |
| S8 | Chance sampled during traversal (external sampling) | one full deal per (iteration, traverser) | **match** (equivalent: each traversal sees one independent chance sample) |
| S9 | Regret floor applied to every action | M3: clamp after every update; M1: none | **fix in M1** (S-constants table) |
| S10 | Regrets allocated **lazily**, only for action sequences encountered (p. 13: 413,507,309 of 664,845,654) | dense allocation | **design**: kept dense (simpler, predictable memory). A lazy table would save up to ~38% on rarely reached lines. Optional later |

## Abstraction (informational: not part of Algorithm 1, not changed)

| item | paper | ours (m2_11g) |
|---|---|---|
| preflop information abstraction | lossless (169) | same |
| post-flop buckets | 200 per round, k-means on domain features (p. 13) | 256 (flop/turn k-medians on equity distributions; river cut-offs) |
| raise sizes | 1–14 per decision point; finest preflop; turn and river: first raise ≤ {0.5×, 1×, all-in}, later raises ≤ {1×, all-in} (p. 12) | turn first bet {0.25, 0.5, 1, 2}, river first bet {0.5, 0.75, 1, 1.5, 2} + all-in; later raises {1} + all-in |
| action sequences | 664,845,654 | 4,426,184 decision nodes (different counting: ours are betting histories with a decision) |

Using the paper's coarser turn/river sizes would shrink our tree considerably, freeing
memory for buckets or preflop breadth. **Decision for you (D5).**

## Ambiguous: please review

- **A1. Does the traverser's own fold count as "leads to a terminal node"?**
  - In a multiway pot, P_i folding does not end the hand, but P_i's payoff is then fixed and
    cheap: the stated reason for the exception ("the cost of examining those payoffs is minor").
  - I implemented the **literal** reading: only children where the hand ends are exempt.
  - The alternative also exempts P_i's fold everywhere.
- **A2. When does preflop φ accumulation start?**
  - Algorithm 1 initialises φ at t = 0 and updates it every 10,000 iterations, discounted
    during LCFR.
  - The text (p. 15) says "after the initial 800 minutes of training, we stored the average
    strategy for the first betting round and used this as the blueprint". That could mean φ
    starts (or is reset) at 800 minutes.
  - I kept the **Algorithm 1** version (from t = 0).
- **A3. Snapshot averaging weights:** "averaging together these snapshots". I use a uniform
  average of the snapshot strategies (each snapshot = regret-matching current strategy).
- **A4. Rounding is not specified** for integer regret updates or discounting. We use exact
  round-half-to-even (unbiased; `cfr::intregret`).
- **A5. "Minutes" are wall-clock on 64 cores.**
  - We map minutes to iterations using the rate measured at the end of a 10-minute warm-up
    (your decision 3).
  - Our iterations per minute are far fewer than Pluribus', so the schedule is faithful in
    time, not in iterations.
- **A6. Total run length is not in the supplement.** The 8 days / ~12,400 core-hours come
  from the main paper (not re-checked here). The supplement does confirm the hardware: one
  64-core node (four 16-core Xeon E5-8860 v3), < 0.5 TB of RAM used.
- **A7. Concurrency (threads, locking) is not described.** We use lock-free relaxed atomics.

## Deliberate differences kept (design)

- **D1. M1 keeps f64 regrets.** It is the correctness reference on Kuhn and Leduc;
  the integer path is tested separately (`cfr::intregret`) and used in M3.
- **D2. M1 keeps `strategy_interval = 1`.** 10,000 is tuned for billions of iterations. With
  2M Kuhn iterations it would leave ~200 φ samples and make the exploitability tests noisy.
  The flag remains settable.
- **D3. M1 scales C and the floor by max |utility| / 10,000** (paper chip units) for the toy
  games, and keeps the schedule as fractions of the run (toy runs last seconds).
- **D4. Dense regret allocation** (S10).

## Changes applied (this commit)

- `mccfr.rs`:
  - new `regret_floor: Option<f64>` (`pluribus_scaled`: −310M × max\|u\|/10,000)
  - new `prune_skip_final_round_and_terminal` (default **true**)
  - `avg_max_round` stays "all rounds" for the toy games (see the table); 0 remains available
  - old checkpoints load with the old behaviour for missing fields
- `blueprint.rs`:
  - `prune_skip_river_and_terminal` default **true**
  - schedule in iterations from paper minutes (warm-up calibration)
  - discount boundaries with an offset (first discount exactly at the end of the 10-min warm-up)
  - post-flop snapshot accumulator (every 200 min after 800 min, flag `snapshot_average`, default on)
- `train_blueprint`: strategy interval default 10,000; `--warmup-minutes 10`.
