# M4 validation on Leduc: depth-limited nested search (stages 1–5)

Run 2026-11-08 on the MSI: `cargo run --release --bin leduc_validate` (paper settings, 70 s) and
`... -- --no-compress` (148 s). Raw output: `data/leduc_validate.out`,
`data/leduc_validate_nocompress.out` (not in git; rerun reproduces it bit for bit, all seeds fixed).
Values are chips per hand (Leduc ante 1). Exploitability = (BR₀ + BR₁) / 2 of the agent's whole
strategy against an exact best response (`cfr::br`); the agent's strategy is tabulated by
replaying Algorithm 2 along every infoset's public history (`cfr::nested::agent_policy`).

## Status: Leduc = boundary-condition study (ruling 2026-11-08)

Criterion (a) below was badly framed for Leduc and is **superseded**. The paper itself says that
unsafe search has no exploitability guarantee (Supp p. 20), that playing the final iteration may
increase exploitability (p. 22), and that compression relies on abstract infosets rarely recurring
(p. 24); Leduc breaks all three conditions (a 4–6-outcome chance node before the root, 288
recurring infosets, an exact best responder). This run therefore measures *where the method's
assumptions stop holding*, not whether Pluribus' search works. The **3 of 8 positive cells** for
(a) (exact continuations, average play; below) are: MCCFR-2k blueprint, agent C (0.314 vs 0.400);
abstracted blueprint, agent A (0.447 vs 0.548); abstracted blueprint, agent C (0.536 vs 0.548).

The replacement validation is **head-to-head, search agent vs blueprint agent, in heads-up NLHE**
(river/turn subgames from a small heads-up blueprint, AIVAT-style variance reduction), with
exploitability as a secondary measure where computable.

## Questions

- **(a)** *(superseded, see above)* Does depth-limited search with the 4 continuation strategies
  give lower exploitability than playing the blueprint alone?
- **(b)** Are blueprint-only leaves measurably worse than the 4 continuation strategies?

## Setup

**Blueprints** (Leduc "rounds" r1 = before the board, r2 = after):

| blueprint | exploitability | r2-only exploitability¹ |
|---|---:|---:|
| MCCFR 2,000 iterations (no abstraction) | 0.3996 | 0.2864 |
| MCCFR 20,000 iterations | 0.1077 | 0.0656 |
| MCCFR 200,000 iterations | 0.0306 | 0.0190 |
| **abstracted**: r2 sees only "paired with the board or not" (own rank forgotten), CFR 5,000 iterations² | 0.5484 | 0.3849 |

¹ Best responder forced to play the blueprint on r1, free on r2 (diagnostic for unsafe search).
² The Pluribus situation: a lossy blueprint abstraction, a lossless search on the current round.

**Agents** (Algorithm 2, `cfr/src/nested.rs`):

| | r1 | r2 | leaves |
|---|---|---|---|
| A | depth-limited search (leaves before the board) | search to the end | 4 continuations |
| B | same | same | blueprint only |
| C | depth-limited search | blueprint | 4 continuations |
| D | same | blueprint | blueprint only |
| E | blueprint (literal Algorithm 2: no r1 search without an off-tree action) | search to the end | — |
| F (diagnostic) | as C | follows the continuation it chose at the leaf | 4 continuations |

Leduc has no off-tree sizes, so Algorithm 2 alone never searches r1 (E). A–D add a search at the
start of the hand, the analogue of Pluribus' preflop search after an off-tree raise (D-SRCH-13).

**Paper settings:** compressed continuations (seed 1), play the final iteration, σ = weighted
average, MCCFR (50k iterations, Pluribus schedule) for r1 subgames and vector Linear CFR (1k) for r2
subgames; solver seeds 1–3, means shown. **Diagnostic settings:** vector Linear CFR 1k for both, play
the weighted average; with and without compression.

## Results

Exploitability (lower is better); in brackets the head-to-head value against the blueprint
(average of both seats, higher is better). **Bold** = below the blueprint's exploitability.

**Paper settings** (compressed, final iteration):

| blueprint (its exploitability) | A | B | C | D | E | F |
|---|---|---|---|---|---|---|
| MCCFR 2k (0.400) | 0.898 (+0.012) | 1.280 (−0.277) | 0.748 (+0.046) | 1.422 (−0.262) | **0.285** (+0.114) | 1.277 (+0.082) |
| MCCFR 20k (0.108) | 1.067 (+0.000) | 0.621 (−0.139) | 0.631 (+0.002) | 0.837 (−0.089) | 0.292 (+0.012) | 1.174 (+0.007) |
| MCCFR 200k (0.031) | 0.958 (−0.022) | 1.182 (−0.257) | 0.573 (−0.003) | 1.507 (−0.030) | 0.361 (+0.008) | 1.031 (+0.005) |
| abstracted (0.548) | 1.254 (+0.019) | 1.956 (−0.037) | 1.216 (+0.004) | 1.836 (−0.058) | **0.217** (+0.109) | 1.519 (+0.172) |

**Diagnostic: vector Linear CFR, plays the weighted average, uncompressed continuations:**

| blueprint (its exploitability) | A | B | C | D | E | F |
|---|---|---|---|---|---|---|
| MCCFR 2k (0.400) | 0.546 (+0.116) | 0.665 (+0.060) | **0.314** (+0.043) | 0.520 (+0.045) | **0.199** (+0.099) | 0.427 (+0.093) |
| MCCFR 20k (0.108) | 0.633 (+0.016) | 0.899 (−0.011) | 0.136 (+0.013) | 0.270 (+0.013) | 0.146 (+0.013) | 0.203 (+0.018) |
| MCCFR 200k (0.031) | 0.295 (+0.004) | 1.055 (−0.017) | 0.064 (+0.003) | 0.140 (+0.004) | 0.195 (+0.005) | 0.107 (+0.002) |
| abstracted (0.548) | **0.447** (+0.120) | **0.500** (+0.128) | **0.536** (+0.038) | 0.594 (+0.021) | **0.096** (+0.124) | **0.473** (+0.099) |

The other two combinations (paper settings uncompressed; diagnostic settings compressed) are in
the raw outputs and follow the same pattern.

## Verdicts

**(b) holds.** Blueprint-only leaves are worse than the 4 continuation strategies in
exploitability in **29 of 32** comparisons (A vs B and C vs D, 4 blueprints, 4 settings; the 3
exceptions all involve agent A), and worse head-to-head against the blueprint in every paper-settings
cell (A vs B and C vs D, 8 of 8): the effect the paper describes (p. 23: blueprint-only leaf values
"can lead to highly exploitable strategies"). The size of the gap is large: e.g. paper settings,
200k blueprint, D 1.507 vs C 0.573.

**(a) does not hold under the paper's settings in Leduc.** No depth-limited agent (A, C) is less
exploitable than its blueprint (0 of 8). With uncompressed continuations and average play it holds
in 3 of 8 cells, including all depth-limited agents against the **abstracted** blueprint (A 0.447,
C 0.536 vs 0.548), and comes close against the strong blueprint (C 0.064 vs 0.031). The literal
Algorithm 2 agent (E) beats the blueprint where the blueprint is weak or abstracted (0.285 / 0.217
paper settings; 0.199 / 0.096 diagnostic), not where it is well trained.

## Why: diagnostics

1. **The search pipeline itself is correct.** With the opponent held to σ on r1, E's r2 play is
   near-optimal: r2-only exploitability **0.0012–0.0019** for all four blueprints (diagnostic
   settings), vs 0.019–0.385 for the blueprints. Beliefs, the subgame root, the solver and freezing
   therefore work; E's remaining exploitability comes entirely from r1 deviations its beliefs
   (built from σ) do not anticipate: the known weakness of **unsafe search** (Supp p. 20: "there are
   cases where it leads to highly exploitable strategies"). The paper's mitigation is a root that
   "follows a chance node with a large branching factor" (p. 21); Leduc's board has 4–6 outcomes,
   NLHE's flop 19,600. Brown & Sandholm 2017 [52] likewise report unsafe solving "did surprisingly
   well in most games" but "substantially worse" in one.
2. **Compression is very costly in a tiny game.** Each compressed continuation is a pure strategy
   (one action per abstract infoset). In Leduc's 288 infosets every abstract infoset recurs
   constantly, so an exact best response exploits it; the paper's justification ("the odds of
   encountering the same abstract infoset multiple times is very low", p. 24) is about NLHE.
   Removing compression lowers C on the 200k blueprint from 0.573 → 0.304 (paper settings) and
   from 0.456 → 0.064 (diagnostic settings), and F from 0.728 → 0.107 (diagnostic).
3. **Playing the final iteration costs exploitability in a tiny game.** E: 0.195 (average) vs
   0.361 (final) on the 200k blueprint; CFR's last iterate does not converge. The paper accepts this
   because "the final iteration's strategy is sufficiently unpredictable that any exploitation is
   infeasible" (p. 22), which an exact best response in Leduc ignores.
4. **Re-searching r2 after a depth-limited r1 search (A) is worse than playing the blueprint after
   it (C)** in 7 of 8 cells of the two tables (exception: abstracted blueprint, diagnostic settings,
   A 0.447 vs C 0.536): r2 is re-solved under beliefs that an exact best responder can steer in r1
   (point 1 again, now after a searched r1).
5. **Following the chosen continuation (F)** is worse than C (plain blueprint after the r1 search)
   in every cell except the abstracted blueprint with exact continuations (0.473 vs 0.536). So the
   mismatch between the continuation an agent picks at a leaf and what it then plays is not what
   drives the exploitability.

## What this means for NLHE

Leduc stresses exactly the three things the paper argues away by scale: a tiny chance branching
factor at the root (unsafe search), recurring abstract infosets (compression), and an exact best
response (final-iteration play). The parts that are not scale arguments hold up: beliefs and the
search are correct (point 1), and the 4 continuation strategies beat blueprint-only leaves (b).
Against the realistic blueprint (abstracted), every agent wins head-to-head with exact
continuations (+0.02 to +0.13) and all of them except D also lower exploitability; under paper
settings the 4-continuation agents and E win head-to-head (+0.004 to +0.17) while the
blueprint-leaf agents lose. Open items: D-SRCH-6 (compression),
D-SRCH-11 (final iteration) now carry Leduc measurements; their NLHE effect needs NLHE evaluation
(head-to-head and AIVAT, `docs/eval_plan.md`).

## NLHE heads-up head-to-head (replaces criterion (a); 2026-11-09)

**Question:** does Algorithm 2 search on the turn and river beat the blueprint it starts from, in
a game with large chance branching (heads-up NLHE)?

**Setup**
- Blueprint `hu_dev`: `nlhe/configs/hu_dev.cfg` (config v2, 2 players, buckets 169/63/63/63 on
  the j63 abstraction, river on the fly), 8,016 decision nodes; trained 90 min on the MSI,
  12 threads, 1.68·10⁹ MCCFR iterations, 54 post-flop snapshots; checkpoint
  `data/runs/hu_dev/ckpt_001681410525.bin`.
- Blueprint agent: blueprint policy on all streets (pre-flop average; post-flop snapshot
  average), off-tree opponent actions by randomised pseudo-harmonic translation (D-SRCH-16).
- Search agent: blueprint pre-flop and flop; on the turn and river, Algorithm 2 with
  `search_root.cfg` / `search_later.cfg` (D-SRCH-3, pot_bet_hu), Bayes-updated beliefs over
  1,326 hands, lossless heads-up subgames with exact enumeration of the river (D-SRCH-15),
  vector Linear CFR (`vsolve`); **turn 150 / river 400 iterations; plays the final iteration**
  (paper, D-SRCH-11); off-tree opponent actions added to the tree and re-searched.
- Match: `hu_match --deals 400 --threads 12 --turn-iters 150 --river-iters 400 --seed 1
  --measure-expl`; each deal (both hands and the full board) played twice with seats swapped
  (duplicate); 1 bb = 100 chips, 100 bb stacks.
- Variance reduction: AIVAT-style chance correction (equity value function at every flop,
  turn and river deal; actions not corrected, D-EVAL-2). Standard errors are over the
  400 deals (duplicate pairs).

**Result** (800 hands, 400 duplicate deals, 235 deals reached the turn; 48.7 min)

| estimator | search − blueprint, mbb/hand | s.e. | 95% CI |
|---|---:|---:|---|
| **AIVAT-style (chance)** | **+353.9** | **351.0** | **[−334, +1,042]** |
| raw | +390.6 | 395.8 | [−385, +1,166] |

- **Not significant:** the CI contains 0. The chance-only correction cut the variance by only
  ×1.27: most of the remaining variance comes from the agents' own sampled actions, which
  this estimator does not correct.
- Separating ±100 mbb/hand at 2 s.e. needs s.e. ≈ 50, i.e. ≈ 39,000 hands (~40 h at this speed);
  ±200 mbb/hand needs ≈ 10,000 hands (~10 h).
- Search statistics: 697 searches (56 re-searches after off-tree opponent actions), 49 s per
  search on average with 12 concurrent searches; 0 zero-probability observations; the
  blueprint agent translated 94 off-tree actions.
- **Secondary, exploitability within the river subgames** (256 searches, chips; best response
  inside the subgame against the solved beliefs): **final iteration 124.6, average 4.4.** The
  strategy actually played (final iteration, D-SRCH-11) is ~28× more exploitable than the
  average at 400 iterations. A `--play-average` match isolates this effect.

**Engineering note:** the first calibration ran ~50× slower than expected. The cause was
memory, not compute: a heads-up turn subgame holds ~9k (node, river) blocks of regrets and
average sums, ~570 MB in f64, so 20 concurrent solvers paged on 16 GB. Fixed in `b052959`
(f32 tables, pooled buffers, 12 threads): 12 concurrent turn solves now take 2.1× the
single-solve time.

### Follow-up 1 (2026-11-09): normalised exploitability, full AIVAT, f32 vs f64

Rerun of the same 400 deals (seed 1, identical hands; raw and chance-only estimates reproduce
exactly), `hu_match ... --out ../data/hu_match_s1.tsv --capture ../data/subgames_s1.txt`.

**Exploitability within the river subgames** (256 searches; % of pot averaged per search, pot =
both contributions at the subgame root; 1 chip = 10 mbb):

| strategy | chips | mbb | % of pot |
|---|---:|---:|---:|
| final iteration (played, paper) | 124.6 | 1,246 | **16.4%** |
| weighted average (σ) | 4.4 | 44 | **0.61%** |

**Full AIVAT** (D-EVAL-2): decision terms at both agents' decisions with the actor's exact
action distribution; ṽ = the search's value of the full history where the search agent acts in a
solved subgame (both hands known, final-iteration strategies, river enumerated), else the equity
value with any outstanding bet counted as returned (the blueprint stores no values).

| estimator | mbb/hand | s.e. | 95% CI | variance vs raw |
|---|---:|---:|---|---:|
| raw | +390.6 | 395.8 | [−385, +1,166] | ×1 |
| AIVAT chance terms | +353.9 | 351.0 | [−334, +1,042] | ×1.27 lower |
| **AIVAT full** | **+480.6** | **296.9** | **[−101, +1,063]** | **×1.78 lower** |

Still not significant. Unbiasedness check (`hu_agent::tests::aivat_terms_are_unbiased_and_reduce_variance`,
3,000 hands of fixed-probability agents on real heads-up hands): mean correction −8 ± 122
chips (0 within s.e.); variance raw 43.0M → chance terms 15.5M → full 10.8M. Valuing an uncalled
bet as already in the pot (the first version) was unbiased too but *raised* the variance
(20.7M); counting it as returned fixed that.

Hands needed with full AIVAT (per-deal s.d. 5,940 mbb, a deal = 2 hands): 95% CI half-width
±200 mbb/hand ≈ 6,800 hands; ±100 ≈ 27,100 hands; s.e. 50 ≈ 28,200 hands.

**f32 vs f64 tables** (D-SRCH-17): 20 captured subgames (10 turn, 10 river) at 150/400
iterations: root average strategies differ by 2.3% reach-weighted (max 7.9%); exploitability of
the average differs by 0.07% of pot on average (max 0.66%), without a consistent direction (mean
turn 2.38% vs 2.37%, river 0.55% vs 0.47%); game values by 0.015% of pot. The final iteration
differs more (2.1 points of pot on average): it is a single noisy iterate either way.

### Follow-up 2 (2026-11-09): iteration sweep, final iteration vs average

`iter_sweep` on distinct subgames captured from the 800-hand run (`data/subgames_s1.txt`: 360
distinct turn, 237 distinct river): 200 river subgames at 400 / 1,600 / 6,400 / 25,600 iterations
and the first 51 turn subgames at 150 / 600 / 2,400 (the remaining 49 turn subgames and the
9,600 level were not run: ~1.4 h and ~3 h more). Exploitability within the subgame against the
captured beliefs, as % of the pot at the subgame root (mean, median in brackets); solve time per
search on one core with 12 searches running concurrently (≈ core-seconds; contention inflates it
≈ 2× over an idle machine). Output: `data/iter_sweep.tsv`.

| street | iterations | final iteration, % pot | average, % pot | final / average | core-s per search (mean, max) |
|---|---:|---:|---:|---:|---:|
| turn | 150 | 21.7 (21.0) | 2.05 (2.22) | 10 | 71, 117 |
| turn | 600 | 11.0 (9.8) | 0.32 (0.29) | 41 | 281, 454 |
| turn | 2,400 | 11.4 (10.1) | 0.05 (0.03) | 294 | 1,088, 1,790 |
| river | 400 | 16.2 (14.3) | 0.63 (0.45) | 28 | 4.3, 8.5 |
| river | 1,600 | 7.8 (6.0) | 0.18 (0.12) | 59 | 17, 34 |
| river | 6,400 | 6.0 (4.8) | 0.04 (0.02) | 202 | 69, 135 |
| river | 25,600 | 5.9 (4.9) | 0.02 (0.01) | 391 | 274, 539 |

**The final iteration does not converge towards the average.** It improves up to ~600 (turn) /
~6,400 (river) iterations and then plateaus at **~11% (turn) and ~6% (river) of the pot**, while
the average keeps converging (0.05% and 0.02%). This is the known behaviour of regret matching:
its average converges, its last iterate need not. The 400-iteration river figure (16.2%)
matches the match's own measurement (16.4%).

**Compute vs Pluribus** (28 cores × 1–33 s ≈ 28–924 core-s per search, Main p. 888): our turn
search at 150 / 600 / 2,400 iterations takes 71 / 281 / 1,088 core-s; river 400 / 6,400 / 25,600
takes 4 / 69 / 274. The final-iteration plateau is reached inside Pluribus's per-search budget on
both streets. (Not like for like: Pluribus samples one board per iteration, D-SRCH-12, and its
subgames have other sizes.)

### Follow-up 3 (2026-11-09): Pluribus-faithful board sampling (D-SRCH-12)

The search solver now samples one board per iteration (`vsolve::with_sampling`, the paper's
"samples one set of public board cards per thread"; one thread per search here) with Linear CFR
weighting (iteration t's regrets and average contributions weighted by t, unchanged).
Unit test: on a turn subgame, sampling (4,400 iterations) and enumeration (100 iterations, about
the same work) bring the average from 1,667 to 12.0 and 10.5 chips (pot 800); the two average
strategies differ by 2.4% (reach-weighted) at the root.

Sweep with the sampling solver on all 100 turn subgames (`data/iter_sweep_sampled.tsv`); river
subgames have no chance nodes, so sampling does not change them (the 9,600 level is new):

| street | iterations | final, % pot | average, % pot | core-s per search (mean, max) |
|---|---:|---:|---:|---:|
| turn (sampled) | 6,400 | 31.4 (28.6) | 2.87 (2.69) | 80, 127 |
| turn (sampled) | 25,600 | 23.1 (22.5) | 0.63 (0.50) | 318, 497 |
| turn (sampled) | 102,400 | 24.3 (24.0) | 0.23 (0.20) | 1,248, 1,962 |
| river | 9,600 | 6.3 (4.8) | 0.03 (0.02) | 100, 201 |

Same 51 turn subgames, sampled vs enumerating at about equal compute (mean % of pot):

| compute | final: enumerating / sampled | average: enumerating / sampled |
|---|---:|---:|
| ~75 core-s (150 / 6,400 it.) | 21.7 / 27.8 | 2.05 / 2.43 |
| ~300 core-s (600 / 25,600 it.) | 11.0 / 21.7 | 0.32 / 0.53 |
| ~1,150 core-s (2,400 / 102,400 it.) | 11.4 / 22.9 | 0.05 / 0.21 |

**Sampling doubles the final iteration's plateau on the turn (~22–23% of the pot vs ~11%)** and
slows the average's convergence at equal compute (0.21% vs 0.05%): the final iterate now also
carries the noise of the one river sampled in the last iteration, and each river's blocks are
updated on ~1/48 of the iterations. The plateau is reached by ~25,600 sampled iterations
(~320 core-s per search, inside Pluribus's 28–924). **Long-run settings: turn 25,600 / river
6,400 iterations, sampling (faithful).**
