# M2 card abstraction build (m2_11g)

Command: `cargo run --release --bin build_abstraction -- --config nlhe/configs/m2_11g.cfg --out ../data/abstraction/m2_11g --threads 12 --seed 1 --fit-boards 1000 --max-iters 30`

Run 2026-11-07 23:39, below-normal priority, 12 threads. Peak working set 593 MB. Outputs in `data/abstraction/m2_11g/` (gitignored): flop.bkt 24.8 MiB, turn.bkt 291.2 MiB, river.cut, centroids.

## build m2_11g (threads 12, seed 1, fit boards 1000, max iters 30)

- river: equity histogram over 2,598,960 boards 9.6s, 1-D k-means 0.00s, 246 cut-offs (247 buckets)
  - mass per bucket: buckets 247, entries 2809475760, empty 0, min 2702424 (0.24x mean), p10 6928836, median 10951772, p90 15247784, max 36966264 (3.25x mean)
  - cut-offs: first 0.0042 0.0074, last 0.9966 0.9987
- flop: fit sample 1000 boards / 1176000 points, features 4.7s, k-medians 11.4s (30 iterations, last changed 5444, mean L1 209.9, empty reseeds 0)
  - assignment of all 22100 boards: 118.9s
  - write 24.8 MiB: 0.1s; stage total 135.0s
  - bucket sizes over all (hole, board) pairs: buckets 256, entries 25989600, empty 0, min 19548 (0.19x mean), p10 43752, median 94704, p90 168204, max 247284 (2.44x mean)
  - centroid mean equity: lowest 0.150, median 0.453, highest 0.965
- turn: fit sample 1000 boards / 1128000 points, features 0.6s, k-medians 22.6s (30 iterations, last changed 3164, mean L1 207.8, empty reseeds 0)
  - assignment of all 270725 boards: 241.2s
  - write 291.2 MiB: 1.6s; stage total 266.0s
  - bucket sizes over all (hole, board) pairs: buckets 256, entries 305377800, empty 0, min 140976 (0.12x mean), p10 535044, median 1068480, p90 2044332, max 3506628 (2.94x mean)
  - centroid mean equity: lowest 0.076, median 0.480, highest 0.991
- total wall time 410.7s

## River rebuild 2026-11-08: equal-mass cut-offs (exactly 256 buckets)

The 1-D k-means river (247 effective buckets) was replaced by equal-mass cut-offs
(`river_cutoffs_equal_mass`). Only the river stage was rebuilt (flop/turn skipped via
resume). The old file was kept as `river_kmeans247.cut.bak`.

- river: equity histogram over 2,598,960 boards 9.7 s; 255 cut-offs = **256 buckets, none
  empty**.
- Mass per bucket: min 0.03× mean, median 1.0×, max 3.67× mean. Imbalance comes from point
  masses: a single equity value, e.g. board-playing ties, can't be split by an equity
  threshold.
