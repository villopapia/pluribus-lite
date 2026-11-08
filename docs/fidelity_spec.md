# Fidelity specification: Pluribus as published

Every parameter of the blueprint and of real-time search stated in the supplementary
materials of Brown & Sandholm, *Superhuman AI for multiplayer poker*, Science 365 (2019),
DOI 10.1126/science.aay2400 (local copy `docs/19-Science-Superhuman_Supp.pdf`, not committed).

**Page convention:** "p. N" is the page number printed in the supplement's footer. The PDF
page is N + 1, because the cover page is unnumbered. Bracketed numbers [26] etc. are the
supplement's reference list.

Where the supplement is silent, the entry says so explicitly. Our choices for those gaps are
recorded in `docs/deviations.md`, never here.

## 1. Game (pp. 4–5)

| item | paper | page |
|---|---|---|
| players | 6 | 4 |
| starting stack | $10,000 every hand (100 big blinds); each hand is an independent sample | 4 |
| blinds | Player 1 posts $50, Player 2 posts $100; no ante | 4–5 |
| actions | fold, call, raise | 4 |
| round end | every player still in the hand has acted and all have the same amount in the pot | 4 |
| minimum raise | first raise of a round ≥ $100; later raises ≥ the previous raise increment; at most the remaining stack | 4 |
| action order | round 1 starts with Player 3; later rounds start with Player 1 (if still in), then in seat order wrapping 6 → 1 | 4 |
| cards | 2 private cards each; flop 3, turn 1, river 1 community cards | 5 |
| showdown | best 5-card hand from 2 private + 5 community; ties split equally | 5 |
| seat rotation between hands | Player 2 becomes Player 1, etc. | 5 |

## 2. Blueprint: abstraction

| item | paper | page |
|---|---|---|
| infoset identity | two infosets are treated identically iff they share the same **action-abstraction sequence** and the same **information-abstraction bucket** | 12 |
| raise sizes | between **1 and 14** different raise sizes depending on the decision point; **all raise sizes are fractions of the pot** | 12 |
| how sizes were chosen | "decided by hand based on what raise sizes earlier versions of Pluribus decided to use with significant positive probabilities" (**sizes not published**) | 12 |
| round 1 (preflop) | "particularly fine-grained", because search is usually not used on this round (**sizes not published**) | 12 |
| round 2 (flop) | "more coarse" (**sizes not published**) | 12 |
| rounds 3–4 (turn, river) | **at most three** sizes for the first raise in the round: **0.5× pot, 1× pot, all-in**; **at most two** for remaining raises: **1× pot, all-in** | 12 |
| fold and call | always included when legal (blueprint and search) | 13 |
| size of the action abstraction | **664,845,654 action sequences**; **413,507,309** were ever encountered in training | 13 |
| round 1 information abstraction | lossless | 13 |
| rounds 2–4 information abstraction | lossy, **200 buckets per round** | 13 |
| bucketing method | **k-means clustering on domain-specific features [26]** (features not published) | 13 |
| [26] | Johanson, Burch, Valenzano, Bowling, AAMAS 2013 (*Evaluating state-space abstractions in extensive-form games*) | ref. list |

## 3. Blueprint: algorithm

| item | paper | page |
|---|---|---|
| algorithm | external-sampling Monte Carlo CFR [29, 35] with linear weighting and negative-regret pruning (Algorithm 1) | 14, 16–18 |
| linear weighting | regrets **and** average strategies discounted for the first **400 minutes**: after every **10 minutes**, multiply by **(T/10) / (T/10 + 1)**, T = minutes elapsed | 14; Alg. 1 lines 20–26 |
| why it stops at 400 min | "the time cost of doing the multiplications with the discount factor is not worth the benefit later on" | 14 |
| pruning start | after the first **200 minutes** | 14; Alg. 1 line 12 |
| pruning rule | in **95%** of iterations, traverser actions with regret **below −300,000,000** are not explored | 14; Alg. 1 TRAVERSE-MCCFR-P line 14 |
| pruning exceptions | never skipped on the **final betting round**, or if the action **immediately leads to a terminal node** | 3, 14 |
| skip decision | sampled **once per iteration** (not per action) | 3; Alg. 1 lines 13–14 |
| unpruned share | in the remaining **5%** of iterations, every traverser action is explored | 14 |
| regret storage | **4-byte integers** | 15 |
| regret floor | **−310,000,000** for every action | 15 |
| memory allocation | regrets for an action sequence allocated **the first time it is encountered**, except on the first betting round (small, allocated up front); reduced memory **by more than 2×** | 3–4, 13 |
| average strategy φ | tracked on the **first betting round only** | 15; Alg. 1 lines 6–7, UPDATE-STRATEGY line 2 |
| φ update | every **Strategy Interval = 10,000** iterations, the traverser samples one action per own infoset from the current strategy and adds 1; all opponent actions are explored | Alg. 1 line 10, UPDATE-STRATEGY |
| φ start | text: "after the initial **800 minutes** of training, we stored the average strategy for the first betting round" (Algorithm 1 itself initialises φ at t = 0; see deviations D-ALG-3) | 15 |
| post-flop blueprint | snapshot of the **current strategy every 200 minutes after the initial 800 minutes**; the post-flop blueprint is the **average of these snapshots** [32]; saved to disk, not in memory | 15 |
| strategy computation | regret matching; uniform if no positive regret | Alg. 1 CALCULATE-STRATEGY |
| P_i not in hand | traversal continues through the remaining tree; P_i's utility is unaffected | Alg. 1 TRAVERSE lines 19–20 |
| final strategy (round 1) | φ normalised | Alg. 1 header |
| hardware | one **64-core** shared-memory node (four 16-core Intel Xeon E5-8860 v3), 3 TB available, **less than 0.5 TB used** | 11 |
| run length | **not in the supplement** (the main paper reports ~8 days / ~12,400 core-hours; not re-verified here) | — |
| number of iterations | not published | — |
| parallelism / locking | not published | — |

## 4. Real-time search (M4 specification)

### 4.1 When search is used (pp. 15, 19)

| rule | paper | page |
|---|---|---|
| round 1 | search only if an opponent chooses a raise size **more than $100 off** every raise size in the blueprint action abstraction **and no more than four players** remain in the hand | 15 |
| round 1, otherwise | map the raise with **randomized pseudo-harmonic action translation** [39] and play the blueprint as if the mapped size had been chosen | 19 |
| rounds 2–4 | search is **always** used | 19 |

### 4.2 Subgame root and beliefs (pp. 19–21)

| rule | paper | page |
|---|---|---|
| root | a chance node over the nodes h of the root public state G, with probability π^σ(h) / Σ_{h'∈G} π^σ(h') | 19–20 |
| σ for the root | Pluribus' own strategy in each situation where an opponent acted, assumed for the opponent (**unsafe search**) [49] | 20 |
| nested form | always solve **from the beginning of the current betting round**, not from the latest decision point | 2, 20–21 |
| own past actions | the searcher's strategy is held fixed for actions it already chose in the betting round | 2, 21 |
| beliefs | for every player (including Pluribus), a distribution over the **1,326** private-card pairs from an outside observer's view; initially **1/1326** each | 21 |
| belief update | when a round ends, the new round's start becomes the subgame root, and beliefs are updated by **Bayes' rule** using σ = the blueprint if no search has been run yet, otherwise the output of the previous search | 21 |
| unsafe vs safe | unsafe search, because in practice it beats safe alternatives head-to-head [50–52] and only needs strategies for hands with non-zero probability (~4× faster) | 20 |

### 4.3 Subgame depth (p. 21)

| case | leaves |
|---|---|
| search on round 1 | the subgame runs to the end of the round; leaves are the chance nodes at the start of round 2 |
| search on round 2 with **more than two players at the start of the round** | leaves at the chance nodes at the start of round 3, **or immediately after the second raise action**, whichever is earlier |
| all other cases | the subgame extends to the end of the game |

### 4.4 Solving the subgame (pp. 21–22)

| rule | paper | page |
|---|---|---|
| large subgame or early in the game | **Monte Carlo Linear CFR**, as for the blueprint | 21 |
| otherwise | optimised **vector-based Linear CFR** [38] that **samples one set of public board cards per thread** [42] | 21–22 |
| strategy played | the **final iteration's** strategy (not the average) | 22 |
| σ for belief updates | the **weighted average** strategy | 22 |
| size thresholds for the two solvers, iteration counts, time limits | not published | — |

### 4.5 Search abstraction and off-tree actions (pp. 13, 22–23)

| rule | paper | page |
|---|---|---|
| current round | **lossless** information abstraction | 13, 22 |
| later rounds | **500 buckets per round** | 13, 22 |
| bucketing | determined **separately for each flop**, potential-aware abstraction [48] with EMD clustering [27], per [28] | 13–14 |
| raise sizes in search | **1 to 6** (p. 12); "typically no more than five" (p. 22) | 12, 22 |
| off-tree opponent action | added as a valid action to the subgame model and **the subgame is searched again from the root** (typically the start of the betting round); the root does not change until a new round | 22 |
| freezing | freeze action probabilities **only for actions Pluribus chose** in the subgame, **only for its actual hand**; opponent probabilities are not frozen; when a round ends the root moves to the new round, freezing everything before it | 23 |

### 4.6 Leaf values: continuation strategies (pp. 23–24)

| rule | paper | page |
|---|---|---|
| at a leaf | every player still in the hand **simultaneously** chooses one of **four** continuation strategies (or any mixture) for the rest of the game; the choice must be identical across leaves in the same infoset; it is an extra action solved by the search algorithm | 23–24 |
| who chooses | **both** the searcher and the opponents (unlike Modicum [41], where only the opponent chose) | 24 (footnote) |
| strategy 1 | the unaltered blueprint | 24 |
| strategy 2 | blueprint biased toward **folding**: fold probability × **5**, renormalised | 24 |
| strategy 3 | biased toward **calling**: call probability × **5**, renormalised | 24 |
| strategy 4 | biased toward **raising**: all raise probabilities × **5**, renormalised | 24 |
| compression | each continuation strategy is stored as **one sampled action per abstract infoset** (minimum bits) | 24 |
| off-tree history at a leaf | map the leaf to the nearest blueprint node by **deterministic pseudo-harmonic action translation** [39] | 24 |

### 4.7 Nested search procedure (Algorithm 2, p. 25)

1. Start with I = the start of the game, G_root = G(I), σ = σ_blueprint.
2. **Opponent action a:** if a is not in the abstraction at I, add a as a legal action at every
   node of the public node G(I) and set σ = Search(G_root). Advance I ← I·a, then
   CheckNewRound.
3. **Our turn:** sample a ~ σ(I) and mark frozen(I) = True, so σ(I) does not change if the
   subgame is re-solved. Advance I ← I·a, then CheckNewRound.
4. **CheckNewRound:** if the betting round of G(I) is later than that of G_root, set
   G_root = G(I) and σ = Search(G_root).

### 4.8 Evaluation (pp. 7–9)

- AIVAT-based variance reduction, adapted to more than two players (p. 8): our M4
  evaluation target.
- The details are in pp. 7–9 and are to be specified in the M4 evaluation plan.
