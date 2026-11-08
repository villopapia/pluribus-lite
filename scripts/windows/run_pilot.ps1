# Launcher for the Dell pilot long run (docs/HANDOFF.md §2). The run ends when the config's
# planned_total_iterations are done (--minutes 46080 is only a safety cap of 2x the plan). Windows-only operational script;
# the trainer itself stays std-only Rust.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File scripts\windows\run_pilot.ps1 [-UpdateBinary]
#
# * Runs at BelowNormal priority (set on this process; the trainer inherits it).
# * Refuses to start if the trainer is already running for this run dir, or if the STOP file
#   exists (a STOP means someone, or the watchdog, wanted the run stopped: delete it to resume).
# * Runs a private copy of the binary in <run>\bin, so rebuilding engine\target never touches
#   the running exe. -UpdateBinary refreshes the copy from engine\target\release first.
# * stdout -> train.log, stderr -> train.err (both appended); start/exit/exit code ->
#   launcher.log.
# * Memory watchdog while the trainer runs: samples every 5 min, logs to watchdog.tsv every
#   hour (and on every alarm). If available RAM < 1 GB, it creates the STOP file (graceful stop with a final checkpoint).
param([switch]$UpdateBinary)

$ErrorActionPreference = 'Stop'
$repo   = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$engine = Join-Path $repo 'engine'
$run    = Join-Path $repo 'data\runs\pilot_fit11_long'
$stop   = Join-Path $run 'STOP'
$bindir = Join-Path $run 'bin'
$exe    = Join-Path $bindir 'train_blueprint.exe'
$llog   = Join-Path $run 'launcher.log'
$wlog   = Join-Path $run 'watchdog.tsv'
$minAvailMB = 1024
$sampleSec  = 300
$logEvery   = 12   # samples per watchdog log line (12 x 5 min = 1 h)

function Log($m) { Add-Content -Path $llog -Value ("{0:yyyy-MM-dd HH:mm:ss}  {1}" -f (Get-Date), $m) }

New-Item -ItemType Directory -Force $run | Out-Null
(Get-Process -Id $PID).PriorityClass = 'BelowNormal'

$running = Get-CimInstance Win32_Process -Filter "Name='train_blueprint.exe'" |
    Where-Object { $_.CommandLine -match 'pilot_fit11_long' }
if ($running) { Log "already running (pid $($running.ProcessId -join ',')); not starting"; exit 0 }
if (Test-Path $stop) { Log "STOP file present; not starting (delete $stop to resume)"; exit 0 }

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
$p = Start-Process -FilePath cmd.exe -ArgumentList $cmdline -WorkingDirectory $engine -WindowStyle Hidden -PassThru
$null = $p.Handle   # keep the handle so ExitCode is available later
Start-Sleep -Seconds 5
$child = Get-CimInstance Win32_Process -Filter "ParentProcessId=$($p.Id)" | Where-Object Name -eq 'train_blueprint.exe'
Log "started cmd pid $($p.Id), trainer pid $($child.ProcessId)"

if (-not (Test-Path $wlog)) {
    "time`tavail_mb`tcommit_mb`tcommit_limit_mb`tpagefile_used_mb`tpages_per_s`tpage_faults_per_s`ttrainer_ws_mb`ttrainer_private_mb`tevent" | Set-Content $wlog
}
$n = 0
while (-not $p.HasExited) {
    $m  = Get-CimInstance Win32_PerfFormattedData_PerfOS_Memory
    $pf = (Get-CimInstance Win32_PageFileUsage | Measure-Object CurrentUsage -Sum).Sum
    $tp = if ($child) { Get-Process -Id $child.ProcessId -ErrorAction SilentlyContinue } else { $null }
    $avail = [int64]$m.AvailableMBytes
    $ev = ''
    if ($avail -lt $minAvailMB -and -not (Test-Path $stop)) {
        New-Item -ItemType File $stop | Out-Null
        $ev = "STOP created: available $avail MB < $minAvailMB MB"
        Log $ev
    }
    if ($n % $logEvery -eq 0 -or $ev) {
        "{0:yyyy-MM-dd HH:mm:ss}`t{1}`t{2}`t{3}`t{4}`t{5}`t{6}`t{7}`t{8}`t{9}" -f (Get-Date), $avail,
            [int64]($m.CommittedBytes / 1MB), [int64]($m.CommitLimit / 1MB), $pf, $m.PagesPersec,
            $m.PageFaultsPersec, $(if ($tp) { [int64]($tp.WorkingSet64 / 1MB) } else { '' }),
            $(if ($tp) { [int64]($tp.PrivateMemorySize64 / 1MB) } else { '' }), $ev | Add-Content $wlog
    }
    $n++
    $null = $p.WaitForExit($sampleSec * 1000)
}
Log "trainer exited, exit code $($p.ExitCode)"
