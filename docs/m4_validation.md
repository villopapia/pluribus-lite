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
