# Handoff (updated 2026-11-08 ~16:00)

Read this first. Then read `CLAUDE.md` (rules), `docs/fidelity_spec.md` (published
parameters), `docs/deviations.md` (every difference) and `docs/fidelity_report.md` (latest
numbers). **Fidelity to Pluribus is the top requirement:** spec first, code second; every
deviation documented, justified and, where possible, measured.

## 1. Repo state

- Repository: `C:\Users\User\pluribus-lite`, branch `main`; `git log -1` shows the latest
  commit. Tags: `m0`, `m1-prelim`, `m1-prelim2`, `m2-design`.
- Tests: 45 `cfr`, 72 `nlhe` unit, 5 `nlhe` config tests; 15 Python tests; all pass.
  `cd engine && cargo test --release` (cargo at `~/.cargo/bin`, not on PATH) and
  `cd analysis && .venv/Scripts/python -m pytest -q tests`.
- Not in git, kept locally: `docs/papers/*.pdf`, `docs/sources/` copies are in git (web pages);
  `data/` (phh-dataset, parquet, `aay2400_data_file_s1.zip` + `s1/`, abstractions `j200`,
  `j63` (+ `river.tbl`, 2.62 GiB), legacy `m2_11g`, `k200`, runs).

## 2. Running background processes

| what | PID | details |
|---|---|---|
| **Dell long run, pilot only** (`train_blueprint`, `pilot_dell_fit11.cfg`) | **30348** (cmd 7648), started by the scheduled task, so detached from any Claude session | started 2026-11-08 15:54:06, BelowNormal, 12 threads, **planned 83,518,000,000 iterations** (explicit, D-SCHED-2; ≈ 16 days at 60,415 it/s), river on the fly, checkpoint every 120 min, keep 2 |

Run dir `C:\Users\User\pluribus-lite\data\runs\pilot_fit11_long\`:
- `launcher.log`: every start/exit with the **exit code** (the place to look first)
- `train.log` / `train.err`: stdout / stderr of the trainer
- `metrics.tsv`: one row every 5 min (`it_per_s`, `alloc_gib`, `pruned_rate`, ...)
- `progress.tsv`: every 6 h, measured vs planned speed and **projected finish (UTC)**
- `watchdog.tsv`: memory every hour (sampled every 5 min); **STOP is created automatically
  if available RAM < 1 GB**
- `schedule.txt`, `ckpt_*.bin`, `postflop_snapshots.bin` (from paper-minute 800 ≈ 26.7 h)
- `aborted_20261108/`: logs of the 14:53 attempt (distorted warm-up, see deviations D-RUN-1)

**Launcher:** `scripts\windows\run_pilot.ps1` (BelowNormal, private binary copy in
`<run>\bin`, refuses to start if already running or if `STOP` exists).
**Scheduled task:** `pluribus-lite resume pilot` runs the launcher **at logon** of the user
(a boot trigger needs admin rights and was refused). It never deletes STOP.

**Stop gracefully:** `New-Item C:\Users\User\pluribus-lite\data\runs\pilot_fit11_long\STOP`
(final checkpoint within seconds). **Resume:** delete `STOP`, then
`Start-ScheduledTask -TaskName 'pluribus-lite resume pilot'` (or log on again). To run a new
build: stop, delete `<run>\bin\train_blueprint.exe` (the launcher copies the current build),
start the task.

**Memory:** the pilot heads for ≈ 10.9 GiB; ~5.7 GB was available at launch (top users:
VS Code 4.8 GB, claude 1.5 GB, Chrome 1.4 GB, WebView2 0.8 GB). If the watchdog creates STOP,
close programs, delete STOP and start the task.

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

1. **Watch the run:** first checkpoint (~17:54 on 2026-11-08), first `progress.tsv` line
   (~21:54), memory (watchdog).
2. **Phase A2 sources** (in progress): `docs/sources_review.md` with recommendations; **no code
   changes until the user has reviewed them**. The Reddit AMA could not be fetched (the fetch
   tool blocks reddit.com): needs a manual save into `docs/sources/`.
3. **Evaluation** (`docs/eval_plan.md`): preflop log-likelihood, AIVAT, on Dell checkpoints.
4. **Open deviation measurements** (status "open" in deviations.md), including the D-ALG-5 and
   D-ALG-9 ablations.
5. **M4 search** per `fidelity_spec.md` §4.
6. Housekeeping: delete legacy abstractions `m2_11g`, `k200` if not needed.

Closed today: disk-safe I/O (task 1), OCHS river table + benchmark (task 2), Data S1
cross-check (task 3), auto-resume task (task 4).

## 5. Half-finished work

- Phase A2 (sources review) is in progress; downloaded PDFs are in `docs/papers/`.
- The first checkpoint of the relaunched run has not been verified yet (due ~17:54).
