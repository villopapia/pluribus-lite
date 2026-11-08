# D-INFO-6 plan: per-flop 500-bucket potential-aware search abstraction

Plan only (2026-11-08); nothing here is built. Ledger item: D-INFO-6 (`docs/deviations.md` §B).

## 1. What the paper specifies

- Search uses **lossless** information abstraction on the current round and **500 buckets per
  round** on later rounds (Supp p. 13, p. 22).
- "These buckets were determined **separately for each flop** (the first three revealed board
  cards) using an algorithm that considers the future potential of each poker hand [28]. That
  algorithm combines the idea of potential-aware abstraction [48] with clustering based on
  earth-mover distance [27]" (Supp pp. 13–14).
- [28] (Brown, Ganzfried, Sandholm, AAMAS 2015), Algorithm 1: cluster public flops, then for each
  public bucket and each later round cluster the private information states with a per-round
  algorithm A_r; its agent used 500 private buckets per round per public bucket. Pluribus' "for
  each flop" is the limit of one public bucket per flop.
- [27] (Ganzfried & Sandholm, AAAI 2014), Algorithm 1 (the per-round algorithm): build the last
  round's abstraction with any method; then, going backwards, give each hand at round n a
  **histogram over round-(n+1) clusters** (where chance's next card sends it), and cluster these
  histograms with k-means under **EMD whose ground distance is the distance between the
  round-(n+1) cluster means**. Their Algorithm 2 approximates that EMD using the sparsity of the
  histograms (a hand reaches at most 46 river clusters). 25 k-means++ restarts, lowest
  within-cluster SS (as our D-INFO-1).

Not published: the river base features, k-means details, approximation settings, iteration caps,
whether flops were canonicalised, and when the tables were built (offline vs. at search time).

## 2. What has to be built

For a flop-rooted search the flop is lossless (1,176 private hands); the **turn** and **river**
need per-flop buckets. A turn-rooted search uses the same per-flop river buckets. Per flop F:

| round | information situations per flop | features | clustering |
|---|---:|---|---|
| river | 1,176 river boards × 1,081 hands = **1,271,256** | OCHS, 8 equities vs Johanson's opponent clusters (same as our blueprint river, D-INFO-2) | k-means L2 → 500 |
| turn | 47 turns × 1,128 hands = **53,016** | sparse histogram over the 500 river buckets (≤ 46 entries) | k-means, EMD with ground distance = L2 between river cluster means → 500 |

Output per flop: river bucket (u16) per (turn+river, hand) and turn bucket per (turn, hand):
1,271,256 × 2 B + 53,016 × 2 B ≈ **2.65 MB**.

**Flops:** 22,100 raw flops (no suit isomorphism, as everywhere else in the engine, D-INFO-5) or
**1,755 canonical flops** with suit isomorphism (Waugh [47]). Storage: 22,100 × 2.65 MB = 58 GB vs
1,755 × 2.65 MB = **4.6 GB**. The plan adopts canonical flops for this table only (a new deviation
item when built), with a suit-permutation map applied to hands and turn/river cards at lookup.

## 3. Cost: measured (one-flop benchmark, 2026-11-08)

`cargo run --release --bin info6_bench -- --threads 12` on the MSI (i7-14650HX, 12 threads),
flop [44, 22, 1] (raw card indices), k = 500, seed 1; output kept in `data/info6_bench.out`.
EMD = G&S Algorithm 2 (`nlhe::potential`). Turn points carry 10.1 non-zero river buckets on
average, far sparser than the 46 × 500 the old estimate assumed, which is most of the gap.

**Faithful settings** (all points, 25 restarts, exact means): river 52.4 s + turn 424.9 s +
features/histograms ~5 s ≈ **8 min per flop** → 1,755 canonical flops ≈ **235 h on the MSI
(~2,800 thread-hours)**, ≈ 44 h on one 64-core node. The earlier estimate (7.8 h per flop,
~165,000 core-hours) was ~60× too pessimistic and is withdrawn.

**Noise floors** (what "negligible" is measured against):
- river: one independent 25-restart fit (different seed) vs the full fit: within-cluster
  +2.07%, ARI 0.734;
- turn: best vs second-best of the 25 restarts: within-cluster EMD +1.56%, ARI 0.826.

| reduction | speed-up | within-cluster loss vs full | ARI vs full | floor (loss / ARI) | decision |
|---|---:|---:|---:|---|---|
| suit isomorphism (1,755 canonical flops) | ×12.6 flops | 0 (lossless) | — | — | **accepted** (user, 2026-11-08) |
| river: 5 restarts | ×5.4 | +1.08% | 0.733 | +2.07% / 0.734 | **accepted** (within noise) |
| river: fit on a 200k-point sample, assign all | ×3.7 | −0.87% | 0.797 | +2.07% / 0.734 | **accepted** (within noise) |
| turn: Hamerly bounds | ×1.3 | 0 (exact; identical buckets) | 1.000 | — | **accepted** (exact) |
| turn: 5 restarts | ×4.8 | +0.00% measured (best restart was #3 of 25); **expected +1.5%** (best of 5 drawn from the 25 restart values) | 1.000 measured | +1.56% / 0.826 | **at the floor; user decision** |
| turn: fit on a 10k-point sample | ×3.2 | +0.95% | 0.805 | +1.56% / 0.826 | **rejected** (ARI below the floor) |
| turn: means truncated to 64 entries | ×1.0 | 0 | 1.000 | — | **rejected** (no speed-up: means are already sparse) |

**Cost with the accepted set** (river 5 restarts on a 200k sample, turn Hamerly, 25 restarts):
river ≈ 3 s (product of the two river reductions, not measured together) + turn ≈ 339 s ≈
**~6 min per flop → ~170 h on the MSI, ~32 h on a 64-core node**. Adding turn 5 restarts:
~75 s per flop → **~37 h on the MSI** (≤ 2 h chunks, resumable per flop) or ~7 h on HPC.
Memory: < 1 GB per worker. Search-time cost: load one 2.65 MB table per flop (~ms); building
on demand (~1–6 min) is far too slow for search (Pluribus 1–33 s per search, Main p. 888), so the
tables are precomputed.

## 4. Build plan (when approved)

1. **(done, §3)** **Benchmark one flop** end to end at faithful and reduced settings; replace §3's estimates with
   measurements; pick the reductions by quality per second (within-cluster EMD on all points;
   bucket-size spread).
2. **EMD approximation** (G&S Algorithm 2) with known-answer tests: exact EMD from a small
   min-cost-flow solver on random small histograms; the 1-D special case must equal CDF L1; metric
   properties (symmetry, triangle inequality on samples).
3. **Canonical flops**: 1,755 classes, suit-permutation maps; tests: every raw flop maps to one
   class, bucket lookups invariant under suit permutation.
4. **`build_search_abstraction`** (pure Rust, `PathBuf`, `--threads`, `--seed`, `--stop-file`,
   one file per flop with header + checksum, skips finished flops = resumable), written to
   `data/search_abstraction/`. Logged parameters per file.
5. **Integration**: the NLHE search's later-round buckets read the current flop's table; until
   then the search uses the blueprint's buckets for later rounds (interim, recorded under D-INFO-6).
6. **Evaluation**: blueprint-bucket vs per-flop-bucket search in NLHE head-to-head (once the NLHE
   solvers exist), and abstraction quality vs the j63 flop/turn buckets.

## 5. Open questions for the user

- Canonical flops: approved 2026-11-08. Step 1 done (§3).
- Turn 5 restarts sits at the noise floor (expected +1.5%): accept only if the build must run on
  the MSI (~37 h instead of ~170 h); on HPC keep 25 restarts (~32 h on one 64-core node).
