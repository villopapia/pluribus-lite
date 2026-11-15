# Overnight driver for the D-INFO-6 per-flop search abstraction (docs/m4_info6_plan.md):
# back-to-back <= 2 h chunks of build_search_abstraction until -Until (default 08:00) or until all
# 1,755 flops are done. Resumable: finished flops are skipped, so just run it again next night.
# Windows-only operational script; the builder itself is std-only Rust.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\windows\info6_night.ps1 [-Until 08:00]
#
# * BelowNormal priority. Runs a private copy of the binary (<OutDir>\bin), refreshed by
#   -UpdateBinary (build first: cargo build --release --bin build_search_abstraction).
# * A chunk takes no new flop after its --max-secs (one flop ~1-2 min), and is killed at its hard
#   limit (chunk start + 2 h, or -Until); a killed flop leaves only a .tmp file, rebuilt next time.
# * Stop early: create <OutDir>\STOP (the current flop finishes). Delete it before the next night.
# * Log: <OutDir>\night.log (errors included); per-flop results: <OutDir>\build.tsv.
param(
    [string]$Until = '08:00',
    [string]$OutDir,
    [ValidateRange(1, 24)][int]$Threads = 12,
    [int]$Seed = 1,
    [int]$TurnRestarts = 5,
    [switch]$UpdateBinary
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$dir = Join-Path $repo 'data\search_abstraction'
if ($OutDir) { $dir = [IO.Path]::GetFullPath($OutDir) }
New-Item -ItemType Directory -Force (Join-Path $dir 'bin') | Out-Null
$log = Join-Path $dir 'night.log'
$mutex = New-Object Threading.Mutex($false, 'Local\pluribus-lite-info6-night')
$locked = $false
try {
    try { $locked = $mutex.WaitOne(0) } catch [Threading.AbandonedMutexException] { $locked = $true }
    if (-not $locked) { throw 'Another info6 build owns the lock' }
    (Get-Process -Id $PID).PriorityClass = 'BelowNormal'
    $exe = Join-Path $dir 'bin\build_search_abstraction.exe'
    if ($UpdateBinary -or -not (Test-Path $exe)) {
        Copy-Item (Join-Path $repo 'engine\target\release\build_search_abstraction.exe') $exe -Force
    }
    $stop = Join-Path $dir 'STOP'
    if (Test-Path $stop) { throw "STOP file present: $stop (delete it to build)" }
    $end = [datetime]::Today.Add([timespan]::Parse($Until))
    if ($end -le (Get-Date)) { $end = $end.AddDays(1) }
    Add-Content $log "=== $(Get-Date -Format s) night start, until $($end.ToString('s'))"
    $i = 0
    while ($true) {
        $done = @(Get-ChildItem $dir -Filter 'flop_*.bin').Count
        if ($done -ge 1755) { Add-Content $log "$(Get-Date -Format s) all 1755 flops present"; break }
        if (Test-Path $stop) { break }
        $left = ($end - (Get-Date)).TotalSeconds
        $max = [int][math]::Min(5400, $left - 600) # 10 min margin for the flop in progress
        if ($max -lt 300) { break }
        $i++
        $deadline = (Get-Date).AddHours(2)
        if ($end -lt $deadline) { $deadline = $end }
        $tag = Join-Path $dir ('chunk.{0}' -f (Get-Date -Format 'yyyyMMdd-HHmmss'))
        $a = @('--out', ('"' + $dir + '"'), '--threads', $Threads, '--seed', $Seed, '--turn-restarts', $TurnRestarts,
               '--max-secs', $max, '--stop-file', ('"' + $stop + '"'))
        Add-Content $log "$(Get-Date -Format s) chunk ${i}: $done flops present, --max-secs $max, hard stop $($deadline.ToString('s'))"
        $p = Start-Process -FilePath $exe -ArgumentList $a -WindowStyle Hidden -PassThru `
            -RedirectStandardOutput "$tag.stdout.log" -RedirectStandardError "$tag.stderr.log"
        $null = $p.Handle
        $p.PriorityClass = 'BelowNormal'
        $killed = $false
        while (-not $p.WaitForExit(1000)) {
            if ((Get-Date) -ge $deadline) { $p.Kill(); $killed = $true; $p.WaitForExit(); break }
        }
        $code = $p.ExitCode
        Add-Content $log "$(Get-Date -Format s) chunk $i done (exit $code, killed $killed): $((Get-Content "$tag.stdout.log" -Tail 1))"
        if ($code -ne 0 -and -not $killed) {
            Get-Content "$tag.stderr.log" | Where-Object { $_.Trim() } | ForEach-Object { Add-Content $log "    $_" }
            throw "build_search_abstraction failed with exit code $code"
        }
    }
    Add-Content $log "=== $(Get-Date -Format s) night end after $i chunks, $(@(Get-ChildItem $dir -Filter 'flop_*.bin').Count) flops present"
} catch {
    Add-Content $log "=== $(Get-Date -Format s) night stopped: $_"
    throw
} finally {
    if ($locked) { $mutex.ReleaseMutex() }
    $mutex.Dispose()
}
