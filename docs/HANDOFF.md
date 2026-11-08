# Handoff (session ending 2026-11-08 ~12:00)

Read this first. Then read `CLAUDE.md` (rules), `docs/fidelity_spec.md` (published
parameters), `docs/deviations.md` (every difference) and `docs/fidelity_report.md` (latest
numbers). **Fidelity to Pluribus is the top requirement:** spec first, code second; every
deviation documented, justified and, where possible, measured.

## 1. Repo state

- Repository: `C:\Users\User\pluribus-lite`, branch `main`. The working tree is clean after the
  handoff commit, the one that adds this file; `git log -1` shows it. The previous commit
  was `c200c6a`.
- Tags: `m0`, `m1-prelim`, `m1-prelim2`, `m2-design`.
- Tests at handoff: 45 `cfr` tests, 66 `nlhe` unit tests, 4 `nlhe` config tests, 11 Python
  tests; all pass. Run them with `cd engine && cargo test` and
  `cd analysis && .venv/Scripts/python -m pytest -q tests`. Cargo is in `~/.cargo/bin`, which
  is not on PATH.
- Not in git, kept locally:
  - `docs/papers/*.pdf`: sources, see the spec §0
  - `data/`, which contains:
    - `phh-dataset` clone and parquet files
    - abstractions: `data/abstraction/j200` (faithful, Johanson features, 200 buckets),
      `j63` (pilot_fit11, 63 buckets), and legacy `m2_11g`, `k200` (1-D river; safe to delete)
    - runs: `data/runs/*`

## 2. Running background processes

| what | PID | details |
|---|---|---|
| **Dell long run, pilot only** (`train_blueprint`, `pilot_dell_fit11.cfg`) | **27816** (wrapper `cmd.exe` 13412, created via WMI by `WmiPrvSE.exe`, i.e. **detached from the Claude session**) | started 2026-11-08 11:50:20, **BelowNormal** priority, 12 threads, plan and budget 23,040 min (16 days), warm-up 10 min, checkpoint every 120 min, keep 2 |

Paths are under `C:\Users\User\pluribus-lite\data\runs\pilot_fit11_long\`:
- `train.log`: stdout; discount, snapshot and checkpoint events
- `metrics.tsv`: one row every 5 min
- `schedule.txt`: written at the end of the warm-up
- `ckpt_*.bin`: checkpoints

Command line (run from `engine\`):

```
target\release\train_blueprint.exe --config nlhe/configs/pilot_dell_fit11.cfg
  --abstraction-dir ../data/abstraction/j63 --buckets 63 --run-dir ../data/runs/pilot_fit11_long
  --threads 12 --minutes 23040 --plan-minutes 23040 --warmup-minutes 10
  --checkpoint-minutes 120 --keep-checkpoints 2 --log-seconds 300
  --stop-file ../data/runs/pilot_fit11_long/STOP
```

**Check it:**
- `Get-Process -Id 27816 | Select Id, PriorityClass, WorkingSet64`
- `Get-Content C:\Users\User\pluribus-lite\data\runs\pilot_fit11_long\train.log -Tail 5`

**Stop it gracefully** (writes a final checkpoint, then exits):
`New-Item C:\Users\User\pluribus-lite\data\runs\pilot_fit11_long\STOP`.
It checks the file between 20,000-iteration epochs, i.e. within seconds. Delete `STOP`
before restarting.

**Stop it hard:** `Stop-Process -Id 27816`. This loses everything since the last checkpoint,
up to 2 h.

**Resume:** run the same command line again. It always resumes from the newest checkpoint.
**There is no auto-resume after a reboot yet:** a scheduled task needs your approval, see the
task list.

**Memory risk:**
- The estimate is ≈ 10.9 GiB once ~62% of post-flop sequences are allocated (first-use
  model), and RAM use grows over the run.
- About 5 GB was free with normal desktop use at launch.
- If Windows starts paging heavily (machine sluggish, `alloc_gib` in `metrics.tsv` above ~6–8),
  stop the run with the STOP file. It resumes later without loss beyond the last checkpoint.

**Binary caveat:**
- The running binary is the current commit's build. It uses the old snapshot writer (full-file
  rewrite) and the old retention (3 checkpoint files exist briefly while writing).
- Both are harmless at Dell sizes.
- Snapshots and φ start at paper-minute 800 = **~1,600 min ≈ 26.7 h after the start, around
  2026-11-09 14:30**. To move the run onto the new snapshot code (task 1), stop it before then
  with the STOP file, rebuild, and rerun the same command.

No other background processes are running.

## 3. Decisions made this session that are not recorded elsewhere

(Those already recorded in `fidelity_spec.md` / `deviations.md` / `fidelity_report.md` are not
repeated.)

1. **`faithful.cfg` stays at θ = 0.14** (718.7M sequences). Target node: Stanage large-memory
   (~1 TB; 2 TB exist). Recorded in the spec.
2. **Checkpoint retention:** keep **2** files *including* the one being written: delete the
   oldest before writing, so one complete checkpoint always exists. Target disk peak ≈ 1.0 TiB
   for faithful. Not implemented yet (task 1).
3. **Snapshot average:** update in place, in chunks; no full-file rewrite. Not implemented
   yet (task 1).
4. **Precomputed OCHS river table** (2.6 GiB, `river_method = precomputed_table`): to be built,
   tested against on-the-fly OCHS, and benchmarked on the Dell (on-the-fly vs memory-mapped).
   Not started (task 2).
5. **Pilots:**
   - `pilot_dell.cfg` (200 buckets, ≈ 28.6 GiB) is for an **HPC standard node**.
   - `pilot_dell_fit11.cfg` (63 buckets) is the **Dell** long run, now running.
   - The Dell run uses a dedicated k = 63 abstraction (`data/abstraction/j63`), not 200
     buckets merged down to 63.
6. **Old smoke checkpoints:** only the latest per run directory is kept (done).
7. **Data File S1:** you will download it manually into `data\`; the cross-check runs once
   it is there.
8. **No scheduled task** may be created without asking you first.

## 4. Open tasks, in priority order

1. **Disk-safe trainer I/O (decisions 2–3).**
   - In-place chunked snapshot update. Make it crash-safe: each record stores its own
     snapshot count, so a partially applied snapshot is detected and skipped on retry.
   - Delete-before-write checkpoint retention.
   - Tests for both, then report the new faithful disk peak.
   - **Then restart the Dell run on the new binary before ~2026-11-09 14:30** (see §2).
2. **Precomputed OCHS river table (decision 4).**
   - Build the table (2.81×10⁹ (hole, board) entries, u8).
   - Test it against on-the-fly OCHS buckets on random samples.
   - Benchmark it against on-the-fly. Memory-mapping needs OS calls; the pure-Rust `memmap2`
     crate needs no C/C++ toolchain, but adding any crate needs your OK (CLAUDE.md: "prefer
     none"). The alternative is reading the file into RAM.
   - If it is faster, use it for the Dell run at the next restart (log it in deviations).
3. **Data File S1 cross-check** once the zip is in `data\`: hand by hand, compare actions,
   cards and payoffs against our PHH parse; report mismatches.
4. **Auto-resume after reboot:** propose a scheduled task (on logon/boot: run the command in §2
   detached, below-normal) and **ask before creating it**.
5. **Evaluation (`docs/eval_plan.md`):** preflop log-likelihood of Pluribus' actions, behavioural
   statistics, AIVAT; start on Dell-run checkpoints.
6. **Open deviation measurements** (`deviations.md`, status "open"): θ sensitivity, OCHS vs 1-D
   river, schedule-fraction sensitivity, φ start, "pot" convention, D-ALG-9.
7. **M4 search** per `fidelity_spec.md` §4.
8. **Housekeeping:** delete legacy abstractions `data/abstraction/m2_11g` and `k200` (1-D river)
   if no longer needed.

## 5. Half-finished work

- **Nothing is half-coded:** all code is committed and tested. Tasks 1 and 2 above were
  requested but **not started**.
- `docs/fidelity_report.md` describes the new disk plan (§3, "Disk with in-place snapshots")
  as **pending implementation**.
- The Dell run's first checkpoint is due about 2 h after its start (~13:50 on 2026-11-08).
  **Nothing has verified it on this run yet**: check `train.log` for the
  `checkpoint ... written` line.
