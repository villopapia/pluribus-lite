# MSI evaluation follow-up — 2026-11-11

**No overnight evaluation was launched.** The full-budget smoke test completed at
23:39:41 GMT on 10 November. No independent overnight driver was subsequently
started before the session paused. On continuation at 09:13 GMT on 11 November,
the requested 08:00 deadline had passed. No launch was attempted and no deadline
was rolled forward. There is no overnight PID or automatic pending launch.

## Memory and decision

- Installed RAM: 16 GiB; Windows usable RAM: 15.71 GiB.
- After Chrome was closed, the initial check on 10 November found 5.88 GiB
  available, 41% commit usage, 11 paging reads/sec and no paging writes in that
  sample. Eight workers were selected conservatively instead of twelve.
- Initial larger consumers included Memory Compression (0.86 GiB working set),
  individual VS Code processes (up to 0.50 GiB), DWM (0.45 GiB), and Defender
  (0.33 GiB). During the smoke test, VS Code's processes together occupied about
  1.92 GiB working set. No user processes or services were stopped.
- During the production-budget smoke test, minimum sampled available RAM was
  **4.522 GiB** and peak sampled evaluator private memory was **1.443 GiB**.
  These are ten-second samples, not a proof of the worst possible deal's peak.
- On continuation the available RAM had fallen to 0.94 GiB; a subsequent sample
  showed 775 MiB available, 61% commit usage and 312 paging reads/sec. Current
  memory pressure independently prevents starting the evaluation.
  A final sample recovered to 1.46 GiB available, still below the prepared
  eight-worker start threshold. The largest individual morning working sets
  included vmmemWSL (1.53 GiB), ntws (1.17 GiB), Chrome (0.78 GiB) and Defender
  (0.55 GiB).

## Completed verification

The opponent, pairing and estimator remain as described in
`HU_EVAL_AUDIT_2026-11-10.md`: local `hu_dev` checkpoint
`ckpt_001681410525.bin`, 54 postflop snapshots, final versus average search play,
25,600 turn / 6,400 river iterations, one board per iteration, and both seat
positions per deal. Worker count is an existing CLI setting; card and actor RNG
streams are derived from seed, deal ID and seat, not worker number.

Smoke outputs:
`data/hu_preflight_20261110_2315/production_smoke/`

- Eight workers, seed 20261109, eight deals per arm, established full iteration
  counts and BelowNormal priority. Each arm completed 16 hands; final performed
  14 searches and average 13.
- The initial eight-minute final-arm limit stopped the owned evaluator after
  seven complete rows. No process survived. The TSV was checked for a complete
  final newline, unique IDs and finite estimator values before resuming.
- A single retry retained the seven rows and recomputed only the unfinished
  deal, with a 15-minute limit per arm. Final exited successfully at 23:31:54
  (507.4 seconds for that resumed invocation); average exited successfully at
  23:39:41 (466.7 seconds). Neither successful invocation hit a memory stop or
  timeout. The original tight deadline failure remains in the logs.
- Morning validation confirmed exactly IDs 0–7 in each arm, no duplicates,
  complete final rows and finite raw/chance/full values. These eight matched
  deals are smoke-test data, not an overnight performance experiment.
- The three paired-analysis Python tests passed. The extended launcher stub
  tests passed, including low-memory preflight rejection, low-memory termination,
  changed-input rejection, priority, locks, resume, deadlines, alternation,
  failure propagation and temporary keep-awake setup.
- For each of seed 2 (old experiment), 20261109 (smoke) and 20261110 (proposed
  experiment), checked 17,000 initial card/actor stream seeds across IDs 0–3399:
  no collisions within or between these sets.
- Original result SHA-256 hashes remain exactly those in the prior audit.
  No poker algorithm, trainer source or strategy artifact was changed. The
  existing local modification to `CLAUDE.md` was left alone. The Dell was not
  accessed.

## Launcher changes made in this follow-up

`hu_long.ps1` now supports an available-memory start threshold and an in-run
reserve check, writing resource samples to `resources.csv`. Crossing the reserve
stops only the owned evaluator and fails the driver, with no automatic retry.
An optional `inputs.json` pins file hashes and the selected latest checkpoint.

`hu_long_night.ps1` forwards these settings and offers temporary idle-sleep
prevention with `SetThreadExecutionState`, released when the driver exits.
No power-plan, page-file, Defender or service settings were modified. This API
does not override intentional user sleep/lid actions; see
[Microsoft's API documentation](https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-setthreadexecutionstate).

Small hidden Windows WMI process probes completed after their launching shells
exited. A direct job query found a shared desktop job with flags 6144, no
kill-on-close flag and no process/job memory limit. An initial check treating any
job membership as a failure was overly strict; the actual job limits were then
inspected. Probe evidence is in `job-details.json`, `isolation-verdict.json` and
the completion markers under the preflight directory. This tested the proposed
detachment mechanism, not an actual overnight evaluator. Microsoft documents
[job behavior and process creation](https://learn.microsoft.com/en-us/windows/win32/procthread/job-objects).

## Prepared but unstarted experiment

`data/hu_long_20261110_seed20261110_t8/` contains only preparation artifacts:
private executable, input hashes, proposed run settings and `launch.ps1`.
There are no final/average experiment TSVs, no night log and no overnight PID.

Proposed settings were seed 20261110, eight workers, 3,400 deals per arm, a
4.5 GiB available-memory start threshold and a 2 GiB reserve. Timing was capped
at 5,400 seconds admitting deals per chunk, two hours hard runtime per chunk,
a 2,400-second end margin, and **2026-11-11 07:59:30 GMT** hard end. The bootstrap
refuses to start after its fixed launch window; do not bypass this guard to roll
the experiment into daytime or another night.

For a future overnight attempt, restore adequate free RAM, choose a new dated
directory and explicit future deadline, and verify the independent driver's
PID/logs before ending the session. The prior 12-worker conservative target
remains approximately 6 GiB available. The validated eight-worker option leaves
more headroom, but neither is suitable at the current available memory.

## Inspection commands

From `C:\Users\chris\Desktop\pluribus-lite`:

```powershell
# Confirm no evaluator, inspect final status
Get-Process hu_match -ErrorAction SilentlyContinue
Get-Content data\hu_long_20261110_seed20261110_t8\not-launched.json

# Original 635-deal paired result (unchanged)
analysis\.venv\Scripts\python.exe analysis\hu_pair.py data\hu_long\final.tsv data\hu_long\average.tsv

# Smoke-test logs and diagnostic-only paired summary (8 deals)
$s = 'data\hu_preflight_20261110_2315\production_smoke'
Get-Content "$s\final.log","$s\average.log" -Tail 15
analysis\.venv\Scripts\python.exe analysis\hu_pair.py "$s\final.tsv" "$s\average.tsv"
```

Do not pool the smoke outputs or the prepared seed-20261110 directory with the
original experiment. No new overnight evidence was collected.
