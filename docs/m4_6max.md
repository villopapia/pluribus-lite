# M4 for 6-max: design note (stage 1, 2026-11-13)

The search works heads-up (`hu_agent`, `vsolve`; `docs/m4_validation.md`). Pluribus searches
6-player pots with the 6-player blueprint (here the Dell's `pilot_fit11_long`, ~40% done,
finishing ~16 Nov). This note lists what changes, in build order. Page references are to the
Science supplement.

## 1. What changes from heads-up

| piece | heads-up now | 6-max | status |
|---|---|---|---|
| beliefs | 2 ranges over 1,326 combos | one range per seat (`Beliefs::uniform(6)` already generic); Bayes update of the actor's range on every action (σ = blueprint, or the previous search's average, D-SRCH-11), card update on every public card; a folded seat's range is no longer used: its cards are unknown to the searcher and their card removal is ignored (D-SRCH-14) | ranges: done; tests in `msolve` |
| subgame root | start of the current round | same: the public state at the start of the round, all seats' contributions and the folded mask (`subgame::round_start(rules, street, total, folded)`) | done (builder is multi-seat) |
| leaf rule | none (turn/river subgames run to the end) | **preflop**: leaves at the flop chance node; **flop with > 2 players at the start of the round**: leaves at the turn chance node or immediately after the 2nd raise, whichever is earlier (p. 21, D-SRCH-4); **flop heads-up, turn, river**: to the end of the game | builder has leaves (`Kind::Leaf`, Leduc-tested); NLHE leaf values missing |
| continuation strategies | — | at a leaf every seat still in the hand chooses among 4 strategies: the blueprint and blueprint biased ×5 towards fold / call / raise (p. 21–22, `cfr::search` + `continuation`; D-SRCH-5, -6, -7, -8); leaf value = expected payoff when each seat plays its chosen continuation to the end (Monte Carlo rollouts over boards and the blueprint's actions, D-SRCH-8) | Leduc done; NLHE rollouts are stage 3 |
| solver | vector Linear CFR, 2 ranges (`vsolve`) | **2 live seats**: `vsolve` unchanged; **3+ live seats**: external-sampling Linear MCCFR (`msolve`): joint deals from the beliefs (card conflicts rejected), infoset = (node, exact hand), regrets and averages weighted by t. Pluribus: Monte Carlo Linear CFR for large or early subgames, vector CFR otherwise (p. 21–22, D-SRCH-9) | `msolve` (river) done |
| information abstraction | lossless on every street of the subgame (D-SRCH-15) | current round lossless (exact hands); later rounds 500 per-flop buckets in the paper (D-INFO-6, not built) → interim: the blueprint's j63 buckets for later rounds inside multiway flop/turn subgames | river needs none |
| off-tree actions | add the action at its state, re-search from the round root (Algorithm 2) | same; multiway subgames are larger, so a re-search costs more (see §3). Preflop: search only when an opponent's raise is > $100 off the blueprint's sizes and ≤ 4 players remain (D-SRCH-14: 1.25% of the 10k Pluribus hands) | HU logic carries over |
| frozen actions | our earlier actions this round keep σ(I) | same (`msolve::MultiSolver::frozen`) | done |
| evaluation | duplicate HU matches + full AIVAT | 6-max: Pluribus used AIVAT adapted to > 2 players (Main p. 889); the decision terms carry over per seat; duplicate deals become seat rotations (6 permutations) | later |

## 2. River first (implemented: `engine/nlhe/src/msolve.rs`)

No chance nodes and no leaves remain on the river, so a multiway river subgame needs only the
beliefs and the solver. `MultiSolver::new(subgame, rules, board, ranges, seed)`:
- live seats = not folded at the root; folded seats get no hand, never act, and keep their
  contributions in the pot (their payoff is −contribution);
- one joint deal per iteration (rejection sampling: probability ∝ Π ranges · compatible);
- payoffs with equal starting stacks have no side pots (asserted);
- `run(iters)`; `strategy(node, combo, avg)`; `table_bytes()`.

Tests: beliefs stay normalised for 6 seats with board-blocked combos at 0; joint deal
frequencies match Π ranges within 0.005 (and conflicting deals never occur); with seats 1 and 3
folded no hand is dealt to them, they never act, every terminal is zero-sum and they lose
exactly their 100 chips; and the "nothing changed" check: on a 3-player river spot a
4,000-iteration search picks the same most likely root action as a 40,000-iteration solve of
the same subgame and beliefs for 83% of the first actor's range. (The real-blueprint check,
search vs blueprint play on on-tree spots, needs the 6-max checkpoint here, §4.)

## 3. Memory and time on the MSI (16 GB)

- **Blueprint in RAM**: regrets 3.44 GiB (Dell metrics, 26% of dense; checkpoint file 3.72 GB)
  + post-flop snapshot average 3.78 GB (loaded whole, u32 per entry) + j63 0.33 GB + tree
  → **~8 GB resident**. Load time ≈ reading 7.5 GB (BufReader, sequential): est. 30–90 s.
- **Per river search**: tables are allocated per visited node, 1,326 × actions × 8 bytes;
  calibrated subgames have 100–2,000 nodes → ≤ 64 MB worst case (measured 2.7 MB on a 259-node
  3-player tree). Many searches in parallel fit next to the blueprint.
- **Time**: 4,000 MCCFR iterations on that tree took 0.01 s (one thread). External sampling
  visits O(depth × actions) nodes per traversal, so 3–6-player river searches are cheap; the
  final iteration-count choice needs the same sweep as heads-up (F-SRCH-1).
- **Turn / flop multiway** (stages 2–3): the turn adds a sampled river per iteration; the
  multiway flop adds leaf rollouts, the expensive part (each leaf value is a blueprint
  rollout; Pluribus parallelised this on 28 cores).
- **Constraint today**: ~0.7 GB available (WSL VM and Chrome). Loading needs ~8 GB free:
  `wsl --shutdown` (WSL is not used by the project) and fewer browser tabs.

## 4. Getting the 6-max blueprint (dev copy)

From the MSI (Git Bash, repo root). The Dell keeps the 2 newest checkpoints (one every 2 h), so
copy the newest within ~4 h of listing it. Syntax verified with a small file over Tailscale.

```bash
mkdir -p data/runs/pilot6
CK=$(ssh user@desktop-g1k8ugg "powershell -NoProfile -Command \"(Get-ChildItem C:\Users\User\pluribus-lite\data\runs\pilot_fit11_long\ckpt_*.bin | Sort-Object Name | Select-Object -Last 1).Name\"" | tr -d '\r')
scp "user@desktop-g1k8ugg:C:/Users/User/pluribus-lite/data/runs/pilot_fit11_long/$CK" \
    "user@desktop-g1k8ugg:C:/Users/User/pluribus-lite/data/runs/pilot_fit11_long/postflop_snapshots.bin" \
    "user@desktop-g1k8ugg:C:/Users/User/pluribus-lite/data/runs/pilot_fit11_long/schedule.txt" data/runs/pilot6/
```

~7.5 GB. The snapshot file is updated in place every few hours; a copy taken during an update
can contain torn records, which the loader detects by checksum and counts as uniform
(`SnapshotAverage::torn`). j63 needs no copy: the MSI's `data/abstraction/j63` is byte-identical
to the Dell's (`docs/j63_hashes.txt`); if wanted anyway:
`scp -r "user@desktop-g1k8ugg:C:/Users/User/pluribus-lite/data/abstraction/j63" data/abstraction/j63_dell`.
The tree is built here from `engine/nlhe/configs/pilot_dell_fit11.cfg` with 63 buckets.

## 5. Ledger items that apply

D-SRCH-1 (joint deal weighting), -2 (zero-probability observations), -3 (subgame sizes; the
multiway root/later configs), -4 (second raise on a multiway flop), -5/-6/-7/-8 (continuation
strategies at leaves), -9 (MCCFR vs vector switch, iterations, time limits), -10 (translation),
-11 (final iteration played, average for beliefs/frozen), -12 (chance sampling), -14 (preflop
trigger, folded players' cards), -15 (abstraction inside subgames; for multiway flop/turn the
interim blueprint buckets), -17 (f32 tables), D-INFO-6 (per-flop 500 buckets), D-EVAL-2
(multiway AIVAT), F-SRCH-1 (final-iteration plateau, to re-measure multiway). New with this
stage: none beyond D-SRCH-9's choice of external sampling with rejection-sampled joint deals
(the paper does not say how Monte Carlo Linear CFR samples private hands in a subgame) — to be
added to the ledger when the 6-max agent plays.

## 6. Next stages

2. 6-max river search agent on the real blueprint: Bayes updates from the blueprint through the
   flop and turn, river search on every river round, play vs the blueprint; measure search vs
   blueprint agreement on on-tree spots; iteration sweep (final vs average).
3. Turn (sampled river per iteration) and flop (leaves + continuation rollouts), then preflop.
