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

## 3. Cost estimate

Anchors measured on the MSI (i7-14650HX, 12 threads), from the j63 build (`data/abstraction/j63/
build_report.md`): river OCHS features 0.4 s per 1,000 boards; k-means L2 (8-D), 1.08M points,
63 clusters, 25 restarts: 48.8 s (≈ 2.4·10¹⁰ distance-flops/s effective); 1-D EMD k-means (L1 on
CDFs), 1.18M points, 63 clusters, 25 restarts: 105.6 s. Precomputed river table (all 2.6M river
boards): 1,783 s (Dell). **Everything below is an estimate scaled from those anchors** (linear in
points × clusters × iterations × restarts); step 1 replaces it with measurements.

**Faithful parameters** (500 buckets, full point sets, 25 restarts, exact-size means):

| per flop | estimate (12 threads) | why |
|---|---:|---|
| river features | ~0.5 s | 1,176 boards |
| river k-means 500 | ~21 s per restart → **~9 min** | 1.27M × 500 × 8 × ~100 iterations |
| turn histograms | < 1 s | 53k hands × 46 rivers, table lookups |
| turn EMD k-means 500 | ~1,100 s per restart → **~7.6 h** | 53k × 500 × ~50 iterations × ~2·10⁴ ops per approximate EMD (46 × ~500 mean support + sort) |
| **total per flop** | **~7.8 h** | |
| all 1,755 canonical flops | **~13,700 h ≈ 1.6 years on the MSI; ~165,000 core-hours** | |

That is infeasible here and large even for HPC (13× the blueprint's 12,400 core-hours), so the
paper's team must have cut costs somewhere unpublished. **Reduced parameters** (each a candidate
ledger item, each measurable against the faithful version on a sample of flops):

| reduction | effect on the turn step | effect |
|---|---|---|
| fit on a sample of 10,000 turn points per flop, then assign all 53,016 once | ×5.3 fewer points in the fit | quality: measure within-cluster EMD of all points |
| Elkan/Hamerly bounds (EMD is a metric when the ground distance is) | ~5–20× fewer EMD evaluations (typical for k-means) | exact, no quality change |
| truncate means to their top 64 entries (renormalised) | ~×8 cheaper EMD | approximation, measure |
| 5 restarts instead of 25 | ×5 | measure the SSE spread (j63: −1% to −6% from 25 restarts) |
| river fit on a 200k-point sample | ×6 for the river step | as above |

With all of them: turn ≈ 10–15 s, river ≈ 15 s, assignment ≈ 5 s → **~30–35 s per flop** →
**~15–17 h for 1,755 flops on the MSI** (12 threads; overnight in ≤ 2 h chunks per CLAUDE.md, resumable
per flop), or ~1 h on a 64-core HPC node. Memory during the build: < 1 GB per worker. Search-time
cost: load one 2.65 MB table per flop (~ms); building one flop on demand (~30 s) is too slow for
search (Pluribus: 1–33 s per search, Main p. 888), so the tables are precomputed.

## 4. Build plan (when approved)

1. **Benchmark one flop** end to end at faithful and reduced settings; replace §3's estimates with
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

- Canonical flops (suit isomorphism) for this table only: needed to keep it at 4.6 GB.
- Which reductions to accept after step 1, and whether the full build runs on the MSI (~16 h) or
  on HPC.
