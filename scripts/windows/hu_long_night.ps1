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
    [int]$MarginSecs = 2400,
    [int]$TurnIters = 25600,
    [int]$RiverIters = 6400,
    [int]$BoardsPerIter = 1,
    [int]$Deals = 3400
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$dir  = Join-Path $repo 'data\hu_long'
New-Item -ItemType Directory -Force $dir | Out-Null
$log  = Join-Path $dir 'night.log'
(Get-Process -Id $PID).PriorityClass = 'BelowNormal'
$end = [datetime]::Today.Add([timespan]::Parse($Until))
if ($end -le (Get-Date)) { $end = $end.AddDays(1) }
Add-Content $log "=== $(Get-Date -Format s) night start, until $($end.ToString('s'))"
$arms = @('final', 'average')
$i = 0
while ($true) {
    $left = ($end - (Get-Date)).TotalSeconds - $MarginSecs
    $max = [int][math]::Min(5400, $left)
    if ($max -lt 600) { break }
    $arm = $arms[$i % 2]
    Add-Content $log "$(Get-Date -Format s) chunk $($i + 1): arm $arm, --max-secs $max"
    & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot 'hu_long.ps1') -Arm $arm -MaxSecs $max `
        -TurnIters $TurnIters -RiverIters $RiverIters -BoardsPerIter $BoardsPerIter -Deals $Deals
    Add-Content $log "$(Get-Date -Format s) chunk $($i + 1) done (exit $LASTEXITCODE)"
    $i++
}
Add-Content $log "=== $(Get-Date -Format s) night end after $i chunks"
