# Launcher for the Dell pilot long run (docs/HANDOFF.md §2). The run ends when the config's
# planned_total_iterations are done (--minutes 46080 is only a safety cap of 2x the plan).
# Windows-only operational script; the trainer itself stays std-only Rust.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File scripts\windows\run_pilot.ps1 [-UpdateBinary]
#
# * Runs at BelowNormal priority (set on this process; the trainer inherits it).
# * Refuses to start if the trainer is already running for this run dir.
# * Runs a private copy of the binary in <run>\bin, so rebuilding engine\target never touches
#   the running exe. -UpdateBinary refreshes the copy from engine\target\release first.
# * stdout -> train.log, stderr -> train.err (both appended); start/exit/exit code/pause/resume
#   -> launcher.log.
#
# Memory watchdog with pause/resume:
# * While the trainer runs: samples every 5 min, logs to watchdog.tsv every hour. If available
#   RAM < 1 GB and no STOP exists, it creates STOP **and the marker STOP.watchdog** (graceful
#   stop: final checkpoint, exit) and logs a "pause".
# * After a watchdog stop, the launcher stays alive and samples every minute (hourly lines in
#   watchdog.tsv). Once available RAM > 3 GB for 10 consecutive minutes, it deletes STOP and
#   the marker, logs a "resume", and starts the trainer again (it resumes from the newest
#   checkpoint). The launcher relaunches the trainer itself because it *is* the scheduled task
#   'pluribus-lite resume pilot', and that task ignores a second start while running.
# * **A STOP without the marker is a manual stop and is never resumed automatically.** To cancel
#   an automatic resume during a watchdog pause, delete STOP.watchdog (STOP stays).
# * At start: STOP + marker (e.g. a reboot during a pause) -> continue the pause; STOP alone
#   -> manual stop, do not start.
# * A trainer exit with a non-zero code is never restarted automatically.
param([switch]$UpdateBinary)

$ErrorActionPreference = 'Stop'
$repo   = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$engine = Join-Path $repo 'engine'
$run    = Join-Path $repo 'data\runs\pilot_fit11_long'
$stop   = Join-Path $run 'STOP'
$wmark  = Join-Path $run 'STOP.watchdog'
$bindir = Join-Path $run 'bin'
$exe    = Join-Path $bindir 'train_blueprint.exe'
$llog   = Join-Path $run 'launcher.log'
$wlog   = Join-Path $run 'watchdog.tsv'
$minAvailMB    = 1024   # pause below this
$resumeAvailMB = 3072   # resume above this ...
$resumeNeedSec = 600    # ... for this long without interruption
$sampleSec     = 300    # while running
$pauseSampleSec = 60    # while paused
$logEverySec   = 3600   # routine watchdog.tsv line interval

function Log($m) { Add-Content -Path $llog -Value ("{0:yyyy-MM-dd HH:mm:ss}  {1}" -f (Get-Date), $m) }

function WLog($tp, $ev) {
    $m  = Get-CimInstance Win32_PerfFormattedData_PerfOS_Memory
    $pf = (Get-CimInstance Win32_PageFileUsage | Measure-Object CurrentUsage -Sum).Sum
    "{0:yyyy-MM-dd HH:mm:ss}`t{1}`t{2}`t{3}`t{4}`t{5}`t{6}`t{7}`t{8}`t{9}" -f (Get-Date), [int64]$m.AvailableMBytes,
        [int64]($m.CommittedBytes / 1MB), [int64]($m.CommitLimit / 1MB), $pf, $m.PagesPersec,
        $m.PageFaultsPersec, $(if ($tp) { [int64]($tp.WorkingSet64 / 1MB) } else { '' }),
        $(if ($tp) { [int64]($tp.PrivateMemorySize64 / 1MB) } else { '' }), $ev | Add-Content $wlog
}

function AvailMB { [int64](Get-CimInstance Win32_PerfFormattedData_PerfOS_Memory).AvailableMBytes }

# Wait during a watchdog pause. Returns $true to resume, $false if the pause was cancelled.
function WaitForMemory {
    $okSec = 0
    $lastLog = Get-Date
    while ($true) {
        if (-not (Test-Path $wmark)) {
            Log "watchdog pause cancelled (STOP.watchdog removed); not resuming automatically"
            WLog $null "pause cancelled by user (marker removed)"
            return $false
        }
        if (-not (Test-Path $stop)) {
            Log "STOP removed by user during a watchdog pause; resuming now"
            Remove-Item $wmark -ErrorAction SilentlyContinue
            WLog $null "resume (STOP removed by user)"
            return $true
        }
        $avail = AvailMB
        if ($avail -gt $resumeAvailMB) { $okSec += $pauseSampleSec } else { $okSec = 0 }
        if ($okSec -ge $resumeNeedSec) {
            Remove-Item $stop, $wmark -ErrorAction SilentlyContinue
            $ev = "resume: available > $resumeAvailMB MB for $([int]($resumeNeedSec / 60)) min (now $avail MB)"
            Log $ev
            WLog $null $ev
            return $true
        }
        if (((Get-Date) - $lastLog).TotalSeconds -ge $logEverySec) {
            WLog $null "paused (above $resumeAvailMB MB for $([int]($okSec / 60)) min)"
            $lastLog = Get-Date
        }
        Start-Sleep -Seconds $pauseSampleSec
    }
}

New-Item -ItemType Directory -Force $run | Out-Null
(Get-Process -Id $PID).PriorityClass = 'BelowNormal'
if (-not (Test-Path $wlog)) {
    "time`tavail_mb`tcommit_mb`tcommit_limit_mb`tpagefile_used_mb`tpages_per_s`tpage_faults_per_s`ttrainer_ws_mb`ttrainer_private_mb`tevent" | Set-Content $wlog
}

$running = Get-CimInstance Win32_Process -Filter "Name='train_blueprint.exe'" |
    Where-Object { $_.CommandLine -match 'pilot_fit11_long' }
if ($running) { Log "already running (pid $($running.ProcessId -join ',')); not starting"; exit 0 }
if (Test-Path $stop) {
    if (Test-Path $wmark) {
        Log "launcher started during a watchdog pause; waiting for memory"
        if (-not (WaitForMemory)) { exit 0 }
    } else {
        Log "STOP file present (manual stop); not starting (delete $stop to resume)"
        exit 0
    }
}

$src = Join-Path $engine 'target\release\train_blueprint.exe'
if ($UpdateBinary -or -not (Test-Path $exe)) {
    New-Item -ItemType Directory -Force $bindir | Out-Null
    Copy-Item $src $exe -Force
    $head = (& git -C $repo rev-parse --short HEAD 2>$null)
    Log "binary copied from $src (repo HEAD $head, sha256 $((Get-FileHash $exe).Hash.Substring(0,16)))"
}

$args_ = "--config nlhe/configs/pilot_dell_fit11.cfg --abstraction-dir ../data/abstraction/j63 --buckets 63 " +
         "--run-dir ../data/runs/pilot_fit11_long --threads 12 --minutes 46080 --plan-minutes 23040 " +
         "--checkpoint-minutes 120 --keep-checkpoints 2 --log-seconds 300 " +
         "--stop-file ../data/runs/pilot_fit11_long/STOP"
# cmd does the redirection (append); paths contain no spaces.
$cmdline = "/c $exe $args_ >> $run\train.log 2>> $run\train.err"

while ($true) {
    $p = Start-Process -FilePath cmd.exe -ArgumentList $cmdline -WorkingDirectory $engine -WindowStyle Hidden -PassThru
    $null = $p.Handle   # keep the handle so ExitCode is available later
    Start-Sleep -Seconds 5
    $child = Get-CimInstance Win32_Process -Filter "ParentProcessId=$($p.Id)" | Where-Object Name -eq 'train_blueprint.exe'
    Log "started cmd pid $($p.Id), trainer pid $($child.ProcessId)"

    $lastLog = [datetime]::MinValue
    while (-not $p.HasExited) {
        $tp = if ($child) { Get-Process -Id $child.ProcessId -ErrorAction SilentlyContinue } else { $null }
        $avail = AvailMB
        $ev = ''
        if ($avail -lt $minAvailMB -and -not (Test-Path $stop)) {
            New-Item -ItemType File $wmark -Force | Out-Null   # marker first: STOP is then known to be ours
            New-Item -ItemType File $stop | Out-Null
            $ev = "pause: STOP created by watchdog, available $avail MB < $minAvailMB MB"
            Log $ev
        }
        if ($ev -or ((Get-Date) - $lastLog).TotalSeconds -ge $logEverySec) {
            WLog $tp $ev
            $lastLog = Get-Date
        }
        $null = $p.WaitForExit($sampleSec * 1000)
    }
    $code = $p.ExitCode
    Log "trainer exited, exit code $code"
    if ($code -ne 0) { Log "non-zero exit code: not restarting automatically"; exit $code }
    if ((Test-Path $stop) -and (Test-Path $wmark)) {
        if (-not (WaitForMemory)) { exit 0 }
        continue   # resume: start the trainer again (newest checkpoint)
    }
    Log "trainer finished (plan done, manual STOP or time budget); launcher exits"
    exit 0
}
