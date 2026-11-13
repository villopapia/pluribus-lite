# MSI heads-up evaluation audit — 2026-11-10

**Decision: NO OVERNIGHT EVALUATION STARTED. No evaluation PID or new experiment logs.**

The MSI memory preflight failed. Available physical memory was 865,600 KiB
(0.83 GiB), and 1,435,544 KiB (1.37 GiB) at 23:02 GMT after verification.
`docs/m4_validation.md` documents approximately 570 MB per turn solver with f64
tables; the current f32 tables approximately halve that storage. Twelve concurrent
turn solvers therefore need roughly 3.4 GB for their tables alone, plus abstraction,
blueprint, evaluator, buffers and OS/application headroom. Starting this workload
with the measured available memory cannot be certified safe against paging and
deadline overruns. Free roughly 6 GiB of available physical memory (a conservative
operational target, not a measured peak), then repeat resource/process/provenance
checks and a bounded real-binary smoke test before launching. No applications were
closed to make room. No automatic delayed launch was installed.

Only the MSI (`Micro-Star International`, Katana 15 HX B14WFK) was accessed.
Nothing on the Dell, its trainer, checkpoints, or process was accessed or changed.
No trainer was launched. No poker engine, training source, or strategy artifact was
modified. `CLAUDE.md` was already locally modified at the start and was left alone.

## Opponent and treatment

- Heads-up NLHE, 50/100 blinds, 10,000-chip (100 bb) stacks.
- Opponent: local `data/runs/hu_dev/ckpt_001681410525.bin`, iteration
  1,681,410,525; 54 postflop snapshots from `postflop_snapshots.bin`, according to
  every recorded arm startup. This is the MSI development blueprint, not the Dell
  pilot. Config: `engine/nlhe/configs/hu_dev.cfg`; j63 abstraction, buckets
  169/63/63/63. Opponent plays preflop average (current-strategy fallback at
  unvisited nodes) and postflop snapshot average, with randomized pseudo-harmonic
  translation of off-tree actions. It never searches.
- Search agent uses that same blueprint preflop/flop and searches turn/river.
  Both arms use 25,600 turn and 6,400 river iterations, one sampled board per
  iteration, 12 workers, `search_root.cfg` and `search_later.cfg`, and river
  subgame exploitability diagnostics (`--measure-expl`).
- **Final** means the current regret-matched search policy after the solver's
  iterations. **Average** means the solver's reach-weighted, linearly
  iteration-weighted average policy (`--play-average`). These labels do not select
  different blueprint checkpoints. Belief updates and freezing use the search
  average in both arms. Search branches can diverge after different actions.

## Pairing, seeds and positions

In `engine/nlhe/src/bin/hu_match.rs`, each deal ID `d` gets its own card stream
`Rng::new(mix(seed,d,0))`. A partial shuffle draws four hole cards and five board
cards. The two replayed hands keep cards assigned to physical seats and exchange
the search and blueprint agents: search occupies SB/button once and BB once.
Each TSV row is the search agent's mean over those two hands, in chips/hand.
Multiplying by 10 gives mbb/hand. Workers write rows in completion order, not ID
order; pairing must use ID, never row number.

The search stream is `mix(seed,d,10+search_seat)`; the opponent stream is
`mix(seed,d,20+blueprint_seat)`. Matched arm seeds therefore match cards, seats and
initial agent RNG streams, independently of worker scheduling and chunk boundaries.
Different action paths can consume different amounts of randomness: matching seeds
does not require identical subsequent actions or search samples after divergence.

The prior run used seed 2 throughout. Final has exactly IDs 0–696 (697 deals);
average has exactly IDs 0–634 (635 deals). Neither file has duplicate IDs, missing
IDs in its prefix, or nonfinite raw/chance/full values. There are 635 common deals
and 62 final-only deals. Every completed row includes both positions.

## AIVAT and statistical interpretation

`hu_agent::play_hand` subtracts chance value differences at public-card reveals
and, for both players, `value(taken) - sum(policy(a)*value(a))` at decisions.
The policy is the actor's actual normalized sampling distribution, including
aggregation of abstract actions mapped to the same real action. Search values
evaluate every candidate action using both known hands and enumerate future cards;
they do not use the actual undealt runout. Elsewhere the value is exact showdown
equity with uncalled bets treated as returned. Value evaluation occurs before the
chosen action is observed/researched and does not update CFR regrets or sample
the actor's RNG.

These are valid zero-mean control variates under the implementation's sampling
model. Each arm can use a different value function: that affects variance, not
the zero-mean property. This is an AIVAT-style estimator; the label “full” in the
TSV/report means this implementation's chance plus decision terms. It does not
establish that all features of published AIVAT, such as imaginary observations or
private-card deal corrections, are implemented. The published method permits
heuristic value estimates with known strategies:
https://arxiv.org/abs/1612.06915 .

`analysis/hu_pair.py` correctly intersects IDs, differences the two per-deal
values, and computes sample SD / sqrt(number of deals), with a nominal normal
95% interval (`1.96*SE`). This accounts for the two seat replays as one cluster.
Do not subtract the all-row arm means, which cover different deal sets. Do not
treat this as a direct final-versus-average match, a whole-game exploitability
measurement, or evidence about elite humans / six-player strength.

The reproduced paired full estimate is **−50.1 mbb/hand, SE 120.3, 95% CI
[−286,+186]**, with 635 deals. It is inconclusive. Timed stopping yields a random
sample size; these are nominal descriptive intervals, not a sequential-testing
guarantee. A forced cutoff can leave incomplete work or a partial final row and
requires inspection before reuse. A clean chunk drains all assigned deals.

## Rerun and preservation findings

Originally, both scripts hard-coded `data/hu_long`. `hu_match` reads the first
column into a completed-ID set, skips those IDs and appends missing ones below
`--deals`. Logs append too. A normal same-settings resume therefore extends the
data rather than duplicating finished rows. However:

- Seed and settings were not checked against existing TSVs. Changing them could
  silently mix experiments because deal IDs alone do not identify cards.
- A fresh directory with seed 2 repeats the old samples. It is not independent
  additional evidence and must not be pooled as new deals.
- If no parseable IDs are found, the binary writes a new header, potentially
  replacing an unreadable/header-only file. The analysis loader silently keeps
  the last duplicate ID, so its successful execution alone is not a data audit.
- The original driver only had a soft admission deadline; in-flight work could
  exceed 08:00. It also continued after errors and lacked a driver-level lock.
- Historical logs agree on checkpoint/settings, but old TSVs contain no embedded
  provenance. Their original binaries were not hash-pinned per chunk; historical
  binary identity cannot be proved retroactively from these files alone.

Existing outputs were left in place. SHA-256 before and after:

| File | SHA-256 |
|---|---|
| `data/hu_long/final.tsv` | `BD345A89568173761CB731EC186866666FE1344077E31DE66A100420879616CD` |
| `data/hu_long/average.tsv` | `E8BBCF897E99B5C30C584C4E9FC655230F3413323A534087E5DDDE715F2FCCB9` |

The existing private and release `hu_match.exe` files had identical SHA-256
`7B24DF98AAD25940A25DECBFED74045F0B8FB57C73C2E0A3B975E5C4C4185C66`.

## Script safeguards and verification

Only evaluation launchers were changed, plus a lightweight regression test:

- Explicit `-OutputDir`, seed/thread forwarding and settings manifest. Existing
  legacy results without a manifest are now refused; use a new directory.
- Separate chunk and night mutexes, plus the existing live `hu_match` check.
- Direct owned child with BelowNormal priority, live stdout/stderr logs, PID
  logging, exit-code checking and failure propagation.
- Existing 2,400-second margin retained; no chunk admitted with less than
  600 seconds after subtracting it; admission budget capped at 5,400 seconds.
- Hard child deadline at the earlier of two hours after chunk start and the
  overnight end time. Timeout kills only that owned evaluator and stops the
  driver. Normal completion drains assigned deals. Both completed arms reaching
  the deal cap stop the driver instead of repeatedly launching empty chunks.
- Refuse `-UpdateBinary` when results already exist.

Tests performed:

1. `cargo test --release -p nlhe --lib`: **103 passed**, including AIVAT
   unbiasedness/variance reduction, exact equity, pair values versus game values,
   sampled/enumerated solver convergence and heads-up position rules. Run on MSI.
2. Python suite: **20 passed, 1 failed**. The failure is the existing SLURM syntax
   test passing a Windows path to the WindowsApps/WSL `bash` executable. Passing
   the same script through stdin to `bash -n` succeeds. No SLURM code was changed.
   All three `test_hu_pair.py` tests passed, also in a targeted rerun.
3. `test_hu_long.ps1`: **passed** with a small managed stub executable, covering
   paths with spaces, append/resume, mismatched-settings and legacy rejection,
   binary replacement rejection, locks, BelowNormal child priority, child and
   night-driver error propagation, hard deadline/no surviving child, alternating
   arms, seed/deadline forwarding, and stopping at the completed deal cap.
   Evidence: `data/hu_launcher_test_20261110-230208-781/commands.log` and that
   fixture's per-arm/night/stdout/stderr logs. These are synthetic launcher-test
   outputs, not poker measurements.
4. PowerShell parsing, `git diff --check`, TSV integrity and original-result
   hash checks passed. Final process inspection found no `hu_match` or evaluation
   driver. No production overnight/smoke evaluation was attempted after the
   failed resource preflight. Detached overnight survival was therefore not
   claimed or verified in this session.

## Morning inspection and next attempt

There is no new overnight result, PID or scheduled restart to inspect. From the
repository root, reproduce the existing result and check current resources:

```powershell
analysis\.venv\Scripts\python.exe analysis\hu_pair.py data\hu_long\final.tsv data\hu_long\average.tsv
Get-Content data\hu_long\night.log -Tail 20
Get-Process hu_match -ErrorAction SilentlyContinue
Get-CimInstance Win32_OperatingSystem | Select-Object @{n='AvailableGiB';e={[math]::Round($_.FreePhysicalMemory/1MB,2)}}
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\windows\test_hu_long.ps1
```

After freeing memory, revalidate resources and the real binary, then use a **new**
dated directory and fresh shared seed (20261110 was proposed but never used for
a poker experiment). Preserve the binary/config/abstraction/checkpoint hashes
with that run. A future overnight invocation should explicitly pass `-Until
08:00 -MarginSecs 2400`, the new absolute `-OutputDir`, and matching search
settings/seed to the driver. Launch the driver in an independent hidden Windows
process, verify its live child PID/priority/logs, and record the actual deadline.
For this requested night the deadline would have been **2026-11-11 08:00 GMT**.
Do not rerun the default command against `data/hu_long`; the new legacy-data guard
deliberately prevents modifying those preserved results.
