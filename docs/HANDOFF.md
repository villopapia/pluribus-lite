# Handoff (updated 2026-11-09 ~06:45)

Read this first. Then read `CLAUDE.md` (rules), `docs/fidelity_spec.md` (published
parameters), `docs/deviations.md` (every difference) and `docs/fidelity_report.md` (latest
numbers). **Fidelity to Pluribus is the top requirement:** spec first, code second; every
deviation documented, justified and, where possible, measured.

## 1. Repo state

- Repository: Dell `C:\Users\User\pluribus-lite` (training), MSI
  `C:\Users\chris\Desktop\pluribus-lite` (development); branch `main`; `git log -1` shows the latest
  commit. Tags: `m0`, `m1-prelim`, `m1-prelim2`, `m2-design`.
- Tests: 65 `cfr`, 99 `nlhe` unit, 5 `nlhe` config tests; 18 Python tests; all pass (2026-11-09, MSI).
- NLHE heads-up head-to-head (search vs blueprint, 800 hands): +354 ± 351 mbb/hand (AIVAT-style), not significant; `docs/m4_validation.md`. History rewritten 2026-11-09 to drop AI trailers (`docs/commit_map.txt`); hook in `scripts/hooks`.
  `cd engine && cargo test --release` (cargo at `~/.cargo/bin`, not on PATH) and
  `cd analysis && .venv/Scripts/python -m pytest -q tests`.
- Not in git, kept locally: `docs/papers/*.pdf` and `docs/sources/*.md` (third-party texts; index with SHA-256 in `docs/sources_review.md`);
  `data/` (phh-dataset, parquet, `aay2400_data_file_s1.zip` + `s1/`, abstractions `j200` (single
  k-means fit; rebuild with 25 restarts before any faithful run), `j63` (25 restarts, 2026-11-08 16:50),
  `j63_r1` (old single-fit j63, for comparison), legacy `m2_11g`, `k200`; runs, with discarded ones
  under `data/runs/discarded/`).

## 0. Where work happens (from 2026-11-08)

- **Development: the MSI** (clone of the private GitHub repo `villopapia/pluribus-lite`).
  `data/`, `docs/papers/` and `docs/sources/` are not in git; copy or rebuild them there as
  needed (abstractions are rebuilt by `build_abstraction` / `build_river_table`).
- **The Dell: only the long run, its watchdog and monitoring.** No builds, benchmarks or tests
  on the Dell while the run is going (they distort its speed and memory). Code changes reach the
  Dell run only through an explicit stop / pull / build / restart.

## 2. Running background processes

| what | PID | details |
|---|---|---|
| **Dell long run, pilot only** (`train_blueprint`, `pilot_dell_fit11.cfg`, `j63` with 25 restarts) | **33564** (cmd 9480), started by the scheduled task, so detached from any Claude session | started **fresh** 2026-11-08 17:04:51 (pid 6984); watchdog stop 19:05 at iteration 404,311,320; **resumed 20:38:06** from `ckpt_000404311320.bin`; BelowNormal, 12 threads, **planned 83,518,000,000 iterations** (explicit, D-SCHED-2; ≈ 16 days at 60,415 it/s), river on the fly, checkpoint every 120 min, keep 2 |

Run dir `C:\Users\User\pluribus-lite\data\runs\pilot_fit11_long\`:
- `launcher.log`: every start/exit with the **exit code** (the place to look first)
- `train.log` / `train.err`: stdout / stderr of the trainer
- `metrics.tsv`: one row every 5 min (`it_per_s`, `alloc_gib`, `pruned_rate`, ...)
- `progress.tsv`: every 6 h, measured vs planned speed and **projected finish (UTC)**
- `watchdog.tsv`: memory every hour, plus every pause/resume event. **Pause/resume (since
  2026-11-08 20:38):** if available RAM < 1 GB, the watchdog creates `STOP` **and the marker
  `STOP.watchdog`** (graceful stop). The launcher stays alive and, once available RAM > 3 GB for
  10 consecutive minutes, deletes both and restarts the trainer from the newest checkpoint.
  **A `STOP` without the marker (yours) is never resumed automatically.** To cancel a pending
  automatic resume, delete `STOP.watchdog`.
- `schedule.txt`, `ckpt_*.bin`, `postflop_snapshots.bin` (from paper-minute 800 ≈ 26.7 h)

**Launcher:** `scripts\windows\run_pilot.ps1` (BelowNormal, private binary copy in
`<run>\bin`, refuses to start if already running or if `STOP` exists).
**Scheduled task:** `pluribus-lite resume pilot` runs the launcher **at logon** of the user
(a boot trigger needs admin rights and was refused). At logon it continues a watchdog pause
(STOP + marker) but never starts over a manual STOP.

**Stop gracefully:** `New-Item C:\Users\User\pluribus-lite\data\runs\pilot_fit11_long\STOP`
(final checkpoint within seconds). **Resume:** delete `STOP`, then
`Start-ScheduledTask -TaskName 'pluribus-lite resume pilot'` (or log on again). To run a new
build: stop, delete `<run>\bin\train_blueprint.exe` (the launcher copies the current build),
start the task.

**Memory:** the regret allocation curve (to 0.48% of the plan) projects **≈ 4.4–6.1 GiB of
regrets at the end** (log-linear vs power-law fit; 2.25 GiB at iteration 404M), i.e. a
trainer private footprint of **≈ 7–9 GiB**, below the old 10.9 GiB estimate. That estimate assumed
Pluribus' encountered share 0.622; the extrapolated share is 0.33–0.47. The Dell has only
~4 GB available with VS Code (4.7 GB), Chrome (1.4 GB) and claude (1.5 GB) open, so expect
watchdog pauses unless these are closed.

## 3. Decisions made on 2026-11-08 (all recorded in deviations.md / fidelity docs)

1. D-ALG-5: φ from t = 0 with discounting; "start at 800" is the ablation flag.
2. D-ALG-9: traverser's fold exempt from pruning; all-in run-outs not; both flags; `pruned_rate`.
3. D-ACT-4 accepted (collapse tested). D-ACT-5: "pot after calling" default, confirmed by the logs.
4. Run plan explicit from an idle-machine benchmark (D-SCHED-2); river on the fly on the Dell.
5. Retention: delete-before-write keeps one complete checkpoint (tested); snapshots in place.
6. Data S1: identical to PHH in all 10,000 hands (`docs/s1_crosscheck.md`); Science's
   supplement PDF is byte-identical to the author's copy.
7. Scheduled tasks: only `pluribus-lite resume pilot` exists (approved). Ask before any other.

## 4. Open tasks, in priority order

1. **Watch the run:** first checkpoint (~19:05 on 2026-11-08), first `progress.tsv` line
   (~23:05), pruning starts at paper-minute 200 (≈ 7.8 h at the planned rate, later at the real
   one), memory (watchdog). The graceful STOP path was verified on the discarded 15:54 run
   (final checkpoint 1.78 GiB in 7.4 s, exit code 0).
2. **Sources:** `docs/sources_review.md`. Applied: D-INFO-1 restarts, D-ACT-7 test, docs. Still
   missing: the Reddit AMA (both the fetch tool and the Chrome extension block reddit.com; the user
   saves it manually).
3. **Before any faithful (HPC) run, `j200` needs:** (a) the **25-restart rebuild**
   (`build_abstraction --config nlhe/configs/faithful.cfg --out ../data/abstraction/j200
   --restarts 25 --seed 1 --fit-boards 1000 --max-iters 300`, ≈ 10 min on 12 threads; move the
   old `j200` aside first, since existing stage files are skipped), and (b) the **precomputed OCHS
   river table** (`build_river_table --abstraction-dir ../data/abstraction/j200`, ≈ 30 min,
   2.62 GiB, self-checks 200,000 random samples). On HPC, use `--river-method precomputed_table
   --river-table-mode ram` (2.56× faster than on the fly on the Dell benchmark;
   `docs/fidelity_report.md`). Then re-benchmark on the node and set `planned_total_iterations`
   for the HPC config (D-SCHED-2).
4. **Evaluation** (`docs/eval_plan.md`): preflop log-likelihood, AIVAT, on Dell checkpoints.
5. **Open deviation measurements** (status "open" in deviations.md), including the D-ALG-5 and
   D-ALG-9 ablations.
6. **M4 search** per `fidelity_spec.md` §4, `docs/m4_design.md`, ledger `deviations.md` §G (D-SRCH-1..13).
   Built on the MSI 2026-11-08: stages 1–3 (NLHE + Leduc), search raise sizes calibrated to 100–2,000
   sequences (D-SRCH-3, `configs/search_root.cfg` + `search_later.cfg`), stages 4–5 for **Leduc**
   (solvers, Algorithm 2). **Leduc validation done** (`docs/m4_validation.md`): (b) holds, (a) fails
   under the paper settings (compression + final-iteration play + unsafe search in a tiny game).
   Next: NLHE solvers and Algorithm 2 on the NLHE subgame (needs a small NLHE blueprint), stage 6
   (preflop translation; preflop subgame size still open), D-INFO-6 per `docs/m4_info6_plan.md`.
7. Housekeeping: delete legacy abstractions `m2_11g`, `k200` if not needed.
8. ~~Pin the Dell's Python env~~ **Closed:** both machines pinned, verified 2026-11-08 (Dell
   `pip freeze` identical to `analysis/requirements.lock`, 19 packages). The interpreters differ
   in patch level only (Dell 3.13.1, MSI 3.13.17).

Closed today: disk-safe I/O (task 1), OCHS river table + benchmark (task 2), Data S1
cross-check (task 3), auto-resume task (task 4).

## 5. Half-finished work

- Nothing half-coded. The first checkpoint of the 17:04 run is due ~19:05.
