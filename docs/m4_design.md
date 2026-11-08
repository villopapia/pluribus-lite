# M4 design: real-time search (depth-limited, nested, unsafe)

Written 2026-11-08, before any M4 code. **Spec first:** every stage below quotes the
supplement (Supp, printed pages; text re-read from `docs/papers/19-Science-Superhuman_Supp.pdf`,
pp. 19–25) or Algorithm 2 (p. 25), then says what we build and how it is tested. Anything the
paper leaves open is a new ledger item `D-SRCH-*` in `docs/deviations.md` §G. Published
parameters are in `docs/fidelity_spec.md` §4; this file does not repeat them all.

Sources used beyond the supplement: Johanson et al. 2011 [42] (public chance sampling, vector
form), Brown & Sandholm 2019 [38] (Linear CFR), Ganzfried & Sandholm 2013 [39] (pseudo-harmonic
mapping, formula checked against `docs/papers/028.pdf` §6), Brown & Sandholm 2017 [52].

## 0. Plan and order

Bottom-up; each stage is tested on a game with a known answer before the next starts.

| stage | what | paper | code | status |
|---|---|---|---|---|
| 1 | beliefs: a range over private hands per player, Bayes updates from σ | pp. 19–21 | `cfr/src/search.rs` (Bayes primitive), `nlhe/src/belief.rs` (1,326 combos), `cfr/src/leduc_subgame.rs` (Leduc) | **built** |
| 2 | subgame: root at the start of the current round; leaf rules | p. 21 | `nlhe/src/subgame.rs`; Leduc in `leduc_subgame.rs` | **built** |
| 3 | leaves: 4 continuation strategies, compressed, chosen per infoset | pp. 23–24 | `cfr/src/search.rs` (bias, compression), `nlhe/src/continuation.rs`, `leduc_subgame.rs` | **built** |
| 4 | solvers: Monte Carlo Linear CFR or vector Linear CFR; play the final iterate, σ ← weighted average | pp. 21–22 | — | next |
| 5 | Algorithm 2: nested search, freezing, off-tree opponent actions | pp. 22–23, 25 | — | |
| 6 | preflop: search trigger, randomized pseudo-harmonic translation; deterministic mapping at leaves | pp. 15, 19, 24 | — | |
| V | validation on Leduc (a) search with 4 continuations beats the blueprint; (b) blueprint-only leaves are worse | — | — | after 4–5 |

**Validation game.** Leduc hold'em (`cfr/src/leduc.rs`): 2 players, 2 rounds, exact best response
in milliseconds (`cfr/src/br.rs`). Mapped onto the paper's rules: search on round 1 has leaves at
the chance node before round 2 (the paper's round-1 rule); search on round 2 runs to the end of
the game. Leduc has no >2-player flop, so that leaf rule is tested on the NLHE betting tree
instead (stage 2).

**Blueprint for development.** Stages 1–3 need no trained NLHE blueprint: their tests use
synthetic policies (closures over `(node, bucket)`) and `StrengthAbstraction`, which need no data
files, so every test runs on any machine. Leduc blueprints are trained inside the tests by M1's
MCCFR (seconds). A small NLHE blueprint (trained here, ≤ a few hours, or the 8-bucket smoke
checkpoint) is first needed at stage 4. The Dell's long-run blueprint is for later.

## 1. Beliefs (stage 1)

**Paper (pp. 19–21).** The subgame root is a chance node over the nodes h of the root public
state G with probability π^σ(h) / Σ_{h′∈G} π^σ(h′). Pluribus "maintains a probability
distribution over pairs of private cards a player (including Pluribus itself) could be holding
from an outside observer's perspective"; each of the 1,326 pairs starts at 1/1326. When a round
ends, the new round's start becomes the root, and beliefs are updated "using Bayes' rule based on
strategy profile σ, where σ is the blueprint if real-time search has not yet been conducted;
otherwise σ is the output of the previously-run search" (p. 21), and "σ is updated based on the
weighted average strategy" (p. 22). Unsafe search: σ for an opponent is what Pluribus' own
strategy would be in that situation (p. 20).

**Design.**
- `Beliefs`: one range per player (NLHE: 1,326 combos, `equity::combo_index` order; Leduc: the
  6 cards). Starts uniform.
- Bayes primitive `search::bayes(range, likelihood)`: `r[h] ← r[h]·L[h]`, renormalise; returns
  the observation's probability. Used for both events:
  - an action a by player p at abstract node n: `L[h] = σ(a | n, bucket(h))`;
  - a public card: `L[h] = 0` if h holds that card, else 1.
- The joint root distribution is the product of the ranges, restricted to hands that do not
  share cards: `w(h₁..h_n) = Π r_p[h_p] · [no shared cards]`. This equals π^σ(h) / Σ π^σ(h′):
  π^σ(h) = chance(h) · Π_p Π_{p's actions} σ, chance(h) is constant over card-compatible deals,
  and every player factor depends only on that player's own hand. The test checks this exactly on
  Leduc against brute-force enumeration of the game tree (D-SRCH-1).
- Order of updates: Bayes' rule is a product of likelihoods, so per-action and per-round updates
  give the same result **if the same σ is used**. Stage 5 applies a round's action likelihoods at
  the end of the round with the final σ of that round, as the paper says (a re-search within the
  round replaces σ for the whole round).
- Zero-probability observation (an opponent takes an action σ gives probability 0 for every hand):
  not covered by the paper. We keep the prior and report it (D-SRCH-2).

**Tests (known answers).** Leduc: the subgame root distribution built from Bayes-updated ranges
equals the exact posterior over deals from enumerating every history consistent with the public
sequence (1e-12). NLHE: uniform start; board blocking leaves exactly C(49,2) = 1,176 live combos
on a flop; a likelihood vector built from a policy over buckets equals σ at the bucket computed
independently by the abstraction; joint weight of overlapping hands is 0.

## 2. Subgame construction (stage 2)

**Paper (p. 21).** "If search is being done on the first betting round, then the subgame extends
to the end of the round, with leaf nodes at the chance nodes at the start of the second round. If
search is being done on the second betting round and there were more than two players at the
start of the round, then leaf nodes occur either at the chance nodes at the start of the third
betting round or immediately after the second raise action, whichever is earlier. In all other
cases, the subgame extends to the end of the game." Nested search "always solves starting at the
beginning of the current betting round" (pp. 20–21).

**Design.** `subgame::Subgame::build(rules, search_cfg, root)`: an explicit tree from the
`BState` at the start of the current round (built by replaying the actual actions, so off-tree
sizes are exact). Node kinds: `Decision`, `FoldEnd`, `Showdown` (including all-in run-outs),
`Leaf` (depth limit), `Chance` (a street change inside the subgame; one child, where the solver
deals cards). Every node keeps its `BState` and the edge keeps its `Action`, so leaves can be
mapped to the blueprint (stage 3) and off-tree actions added (stage 5).
- Leaf rule from the root's street and the players in the hand at the root (= start of round):
  - root on the preflop: a street change is a `Leaf`;
  - root on the flop with > 2 players: a street change is a `Leaf`, and so is the state right
    after the 2nd bet/raise of the flop (D-SRCH-4);
  - otherwise: no leaves; street changes are `Chance` nodes.
- A hand that ends before the leaf (fold, everyone all-in) is terminal as usual.
- Raise sizes: the search's own `AbstractionConfig` (unpublished, D-SRCH-3).

**Tests (known answers).** Hand-counted river tree (all-in-only sizes: 4 decisions, 5
terminals). Preflop root, 6 players, blueprint config: decision count equals `Tree::build`'s
preflop decision count, and the leaves equal its preflop→flop edges. Flop root, 3 players: every
leaf is either a turn start or right after the 2nd raise; no flop decision has ≥ 2 raises. Flop
and turn roots heads-up: no leaves, decision count equals an independent enumeration to the end.

### 2.1 Search raise sizes: calibration to 100–2,000 action sequences (D-SRCH-3)

**Paper.** "During real-time search, the number of action sequences (considering only players'
actions, not chance's actions) in the subgame action abstraction is between 100 and 2,000"
(Supp p. 13); "between one and six" raise sizes (pp. 12–13), "typically no more than five"
(p. 22). The sets are not published.

**Unit.** Action sequences = (history, action) pairs over the subgame's decision nodes, the unit
of the blueprint's 664,845,654 (`subgame::action_sequences`). Leaf continuation choices are not
counted (they are not betting actions).

**Roots.** Every round start Pluribus searched from in the 10,000 logged hands where it was still
in the hand and ≥ 2 players could act (`analysis/subgame_roots.py` → `data/subgame_roots.tsv`):
4,041 roots (flop 1,576 / 271 / 21 with 2 / 3 / 4 players; turn 1,137 / 144 / 11; river
813 / 64 / 4). No post-flop root had more than 4 players.

**Method** (`analysis/calibrate_search.py`, the analogue of θ against 664.8M):
1. Evidence: Pluribus' logged post-flop sizes came from search (p. 19). Rank grid sizes per
   (street, raise depth) by usage (non all-in).
2. Root street: top-k per (street, heads-up/multiway, depth 0/1), k = 1..5, + all-in (≤ 6 raise
   sizes); depth ≥ 2: {1×} + all-in. Later streets of the subgame: top-k_l for the first bet,
   k_l = 0..3, raises all-in only.
3. For each cell, maximise the share of its roots in [100, 2,000]; ties → median closest to
   √(100·2,000) on a log scale → fewer later sizes. Order river → turn → flop, heads-up first.

**Why two configs.** With one set per street the heads-up river needs ~5 sizes to reach 100
sequences (median 68 with the blueprint's sizes), and the same river inside every turn subgame
then pushes multiway turns to a median of 8,724 even with one size everywhere. The calibration
chose all-in only on every later street (k_l = 0).

**Result** (`engine/nlhe/configs/search_root.cfg`, `search_later.cfg`; `subgame_sizes`):

| street | players | roots | median | in [100, 2,000] | blueprint sizes (faithful.cfg): median / in range |
|---|---|---:|---:|---:|---|
| flop | 2 | 1,576 | 508 | 100.0% | 4,404 / 17.4% |
| flop | 3 | 271 | 261 | 100.0% | 120 / 98.2% |
| flop | 4 | 21 | 808 | 100.0% | 352 / 100% |
| turn | 2 | 1,137 | 464 | 98.3% | 544 / 96.6% |
| turn | 3 | 144 | 1,560 | 99.3% | 8,214 / 1.4% |
| turn | 4 | 11 | 17,872 | 0.0% | 51,160 / 0% |
| river | 2 | 813 | 404 | 90.3% | 68 / 2.1% |
| river | 3 | 64 | 726 | 100.0% | 942 / 100% |
| river | 4 | 4 | 6,520 | 25.0% | 5,296 / 25% |
| **all post-flop** | | **4,041** | **488** | **97.2%** | 616 / 43.1% |
| preflop (start, 6 players) | 6 | 1 | 245,817 | no | 19,095,525 |

Sizes chosen (pot fractions, + all-in): flop HU {0.5, 1} / {0.5, 1}; flop MW {0.25, 0.5, 1, 2} /
{0.5, 0.75, 1, 2}; turn HU {0.25, 0.5, 1, 1.5, 2} / {0.5}; turn MW {0.5} / {0.5}; river HU
{0.5, 0.75, 1, 1.5, 2} / {0.5, 0.75, 1, 1.5, 2}; river MW {0.5} / {0.5} (depth 0 / depth 1).
Open: 4-way turns stay above 2,000 even at the minimum; the preflop subgame (stage 6) needs a
further restriction that the paper does not describe.

## 3. Leaves: continuation strategies (stage 3)

**Paper (pp. 23–24).** At a leaf, "each player still in the hand simultaneously chooses one of
four different continuation strategies", or any mixture; "the choice a player makes must be
identical for all leaf nodes that are in the same infoset"; it is "just another action in the
subgame", solved by the search algorithm. Both the searcher and the opponents choose (footnote,
p. 24). The four: the unaltered blueprint; fold probability × 5, renormalised; call probability
× 5, renormalised; all raise probabilities × 5, renormalised ("in all decision points").
"The continuation strategies are compressed by sampling an action at each abstract infoset
according to the probability distribution at that abstract infoset … only that action is stored."
A leaf not in the blueprint abstraction is mapped "to the nearest node … using the deterministic
pseudo-harmonic action translation" (stage 6).

**Design.**
- `search::bias(probs, kinds, k, out)`: k = 0 blueprint, 1 fold, 2 call (check counts as call,
  the game's actions being fold/call/raise, p. 4), 3 every raise; × 5 then renormalise. Where the
  biased action is not legal, the strategy is unchanged (D-SRCH-5).
- Compression `search::sample_fixed(seed, infoset, k, probs)`: one action drawn per (abstract
  infoset, k) from a counter-based hash, so the same infoset always gets the same action with no
  table in memory (D-SRCH-6). Abstract infoset = blueprint (node, bucket) in NLHE, the infoset key
  in Leduc.
- Leaf choice: each player still in the hand who can still act (not all-in, D-SRCH-7) gets a
  4-action decision whose infoset is (leaf public state, own private hand), so it cannot see the
  others' choices (simultaneous) and is shared by every leaf in the same infoset.
- Leaf value for a joint choice (k_p): play out the rest of the game, every player following its
  compressed strategy k_p. NLHE: `continuation::rollout` deals the missing board cards and walks
  the blueprint tree (one sample per call, for Monte Carlo solvers, D-SRCH-8). Leduc: exact
  expectation over the remaining board card (the game is tiny).
- Leaf → blueprint node: `continuation::blueprint_node` replays the action history on the
  blueprint tree; off-tree histories return `None` until stage 6 adds the deterministic
  pseudo-harmonic mapping.
- Leduc wrapper `LeducSubgame` implements M1's `Game` trait, so M1's CFR, MCCFR and exact best
  response run on the depth-limited game unchanged (stage 4 and the validation build on this).

**Tests (known answers).** Bias by hand-computed numbers. Compression: deterministic per
(seed, infoset, k), empirical frequencies match the biased distribution, a degenerate
distribution always gives its action. NLHE: action kinds read from the tree equal the betting
rules' labels on a whole tree; an always-call blueprint gives an exact showdown payoff from a leaf;
a replayed history gives the same node as walking the tree by index. Leduc: the wrapper with no
depth limit **is** Leduc (uniform-policy NashConv 4.747222…, the OpenSpiel reference already used
in M1); with blueprint-only leaves and no compression its value equals the full-game value of
"policy in round 1, blueprint after"; there are exactly 30 leaf-choice infosets (5 leaf public
states × 3 ranks × 2 players); CFR on the depth-limited game drives its exploitability toward 0.

## 4. Solvers (stage 4, design only)

**Paper (pp. 21–22).** Large or early subgames: Monte Carlo Linear CFR "just as it was for the
blueprint". Otherwise: "an optimized vector-based form of Linear CFR (38) that samples one set of
public board cards per thread (42)". Play the **final iteration's** strategy; update σ with the
**weighted average**. Thresholds, iteration counts and time limits are not published.

**Plan.** (a) MCCFR: M1's `Mccfr` on the subgame (Leduc wrapper first, then an NLHE subgame
game), linear discounting, no pruning thresholds tuned yet. (b) Vector form: walk the public tree
once per iteration with a vector over hands per player (Johanson 2011's public chance sampling):
reach vectors down, value vectors up, terminal values by sorted hand strength (O(n) showdown);
Linear CFR weights t. Leaf values need the expectation over hands, i.e. the rollout averaged per
hand (design detail for stage 4). (c) The switch rule is unpublished → D-SRCH-9. Search
abstraction for later rounds (500 buckets, per-flop, potential-aware) stays D-INFO-6.

## 5. Nested search (stage 5, design only) — Algorithm 2 (p. 25)

```
I ← ∅;  G_root ← G(I);  σ ← σ_blueprint
OpponentTurn(a): if a ∉ abstraction(I): add a at every node of G(I); σ ← Search(G_root)
                 I ← I·a; CheckNewRound()
OurTurn():       a ~ σ(I); frozen(I) ← true; I ← I·a; CheckNewRound()
CheckNewRound(): if round(G(I)) > round(G_root): G_root ← G(I); σ ← Search(G_root)
```
Freezing is per (infoset, our actual hand) only; opponents are never frozen (p. 23). A round end
freezes everything before the new root and triggers the Bayes update (stage 1).

## 6. Preflop and translation (stage 6, design only)

Round 1: search only if an opponent's raise is more than $100 away from every blueprint size and
at most four players remain (p. 15); otherwise randomized pseudo-harmonic translation and the
blueprint (p. 19). Pseudo-harmonic [39]: for sizes A < x < B in pot units,
f_{A,B}(x) = (B − x)(1 + A) / ((B − A)(1 + x)) = P(map to A); deterministic version maps to A
iff x < x* = (A + B + 2AB) / (A + B + 2). Which pot convention defines "x" for raises (vs. bets)
follows D-ACT-5 (pot after calling); recorded as D-SRCH-10.
