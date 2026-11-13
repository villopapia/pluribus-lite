# Overnight driver for the M4 NLHE long run: back-to-back <= 2 h chunks of scripts\windows\hu_long.ps1,
# alternating the arms (final, average) so both cover the same deals, until -Until. Each chunk
# gets --max-secs = min(5400, seconds left - margin), where the margin (default 2400 s) covers
# the deals still in progress when a chunk stops taking new ones; no chunk starts with < 10 min.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\windows\hu_long_night.ps1 [-Until 08:00] [hu_long.ps1 settings]
#
# Log: data\hu_long\night.log. BelowNormal priority (inherited by the chunks).
param(
    [string]$Until = '08:00',
    [ValidateRange(2400, 86400)][int]$MarginSecs = 2400,
    [int]$TurnIters = 25600,
    [int]$RiverIters = 6400,
    [int]$BoardsPerIter = 1,
    [int]$Deals = 3400,
    [int]$Seed = 2,
    [int]$Threads = 12,
    [string]$OutputDir,
    [ValidateRange(0, 128)][double]$MinAvailableGiB = 0,
    [ValidateRange(0, 128)][double]$AbortAvailableGiB = 0,
    [switch]$KeepAwake
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$dir  = Join-Path $repo 'data\hu_long'
if ($OutputDir) { $dir = [IO.Path]::GetFullPath($OutputDir) }
$mutex = New-Object Threading.Mutex($false, 'Local\pluribus-lite-hu-night')
$locked = $false
$awake = $false
try {
try { $locked = $mutex.WaitOne(0) } catch [Threading.AbandonedMutexException] { $locked = $true }
if (-not $locked) { throw 'Another overnight evaluation owns the lock' }
if (Get-Process hu_match -ErrorAction SilentlyContinue) { throw 'hu_match is already running' }
if ($KeepAwake) {
    Add-Type -TypeDefinition 'using System.Runtime.InteropServices; public static class HuNightPower { [DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint flags); }'
    if ([HuNightPower]::SetThreadExecutionState(2147483649) -eq 0) { throw 'Could not prevent idle sleep' }
    $awake = $true
}
New-Item -ItemType Directory -Force $dir | Out-Null
$log  = Join-Path $dir 'night.log'
(Get-Process -Id $PID).PriorityClass = 'BelowNormal'
$end = [datetime]::Today.Add([timespan]::Parse($Until))
if ($end -le (Get-Date)) { $end = $end.AddDays(1) }
Add-Content $log "=== $(Get-Date -Format s) night start, until $($end.ToString('s'))"
Set-Content (Join-Path $dir 'driver.pid') $PID -Encoding ASCII
$arms = @('final', 'average')
$i = 0
while ($true) {
    $counts = @($arms | ForEach-Object {
        $path = Join-Path $dir "$_.tsv"
        if (Test-Path $path) { @(Import-Csv $path -Delimiter "`t").Count } else { 0 }
    })
    if ($counts[0] -ge $Deals -and $counts[1] -ge $Deals) { break }
    $left = ($end - (Get-Date)).TotalSeconds - $MarginSecs
    $max = [int][math]::Min(5400, $left)
    if ($max -lt 600) { break }
    $arm = $arms[$i % 2]
    Add-Content $log "$(Get-Date -Format s) chunk $($i + 1): arm $arm, --max-secs $max"
    & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot 'hu_long.ps1') -Arm $arm -MaxSecs $max `
        -TurnIters $TurnIters -RiverIters $RiverIters -BoardsPerIter $BoardsPerIter -Deals $Deals `
        -Seed $Seed -Threads $Threads -OutputDir $dir -StopAt $end.ToString('o') `
        -MinAvailableGiB $MinAvailableGiB -AbortAvailableGiB $AbortAvailableGiB
    Add-Content $log "$(Get-Date -Format s) chunk $($i + 1) done (exit $LASTEXITCODE)"
    if ($LASTEXITCODE -ne 0) { throw 'Chunk failed; no further arms will start' }
    $i++
}
Add-Content $log "=== $(Get-Date -Format s) night end after $i chunks"
} finally {
    if ($awake) { [HuNightPower]::SetThreadExecutionState(2147483648) | Out-Null }
    if ($locked) { $mutex.ReleaseMutex() }
    $mutex.Dispose()
}
