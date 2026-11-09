# One <= 2 h chunk of the M4 NLHE long run (docs/m4_validation.md): search agent vs blueprint,
# heads-up, one arm per call. Both arms use the same seed, so they play the same deals.
# Windows-only operational script; hu_match itself stays std-only Rust.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\windows\hu_long.ps1 -Arm final [-UpdateBinary]
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\windows\hu_long.ps1 -Arm average
#
# * BelowNormal priority (set on this process; hu_match inherits it).
# * Resumable: hu_match skips the deals already in data\hu_long\<arm>.tsv; it stops taking new
#   deals after -MaxSecs (default 6000 s), so a chunk ends within ~2 h. Run it again to continue.
# * Runs a private copy of the binary (data\hu_long\bin), so rebuilding never touches a running
#   exe. -UpdateBinary refreshes it from engine\target\release (build that first).
# * Output appended to data\hu_long\<arm>.log; summary of both arms (paired):
#   analysis\.venv\Scripts\python analysis\hu_pair.py data\hu_long\final.tsv data\hu_long\average.tsv
param(
    [Parameter(Mandatory)][ValidateSet('final', 'average')][string]$Arm,
    [int]$Deals = 3400,
    [int]$TurnIters = 25600,
    [int]$RiverIters = 6400,
    [int]$Seed = 2,
    [int]$Threads = 12,
    [int]$MaxSecs = 6000,
    [switch]$UpdateBinary
)

$ErrorActionPreference = 'Stop'
$repo   = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$engine = Join-Path $repo 'engine'
$dir    = Join-Path $repo 'data\hu_long'
$bindir = Join-Path $dir 'bin'
$exe    = Join-Path $bindir 'hu_match.exe'
New-Item -ItemType Directory -Force $bindir | Out-Null
if ($UpdateBinary -or -not (Test-Path $exe)) {
    Copy-Item (Join-Path $engine 'target\release\hu_match.exe') $exe -Force
}
if (Get-Process hu_match -ErrorAction SilentlyContinue) { throw 'hu_match is already running' }
(Get-Process -Id $PID).PriorityClass = 'BelowNormal'

$hmArgs = @('--deals', $Deals, '--seed', $Seed, '--turn-iters', $TurnIters, '--river-iters', $RiverIters,
            '--threads', $Threads, '--max-secs', $MaxSecs, '--measure-expl', '--out', (Join-Path $dir "$Arm.tsv"))
if ($Arm -eq 'average') { $hmArgs += '--play-average' }
$log = Join-Path $dir "$Arm.log"
Add-Content $log "=== $(Get-Date -Format s) start: $($hmArgs -join ' ')"
Push-Location $engine
# cmd does the redirection: plain-text log, and PS 5.1 does not turn stderr lines into errors
try { cmd /c "`"$exe`" $($hmArgs -join ' ') >> `"$log`" 2>&1" } finally { Pop-Location }
Add-Content $log "=== $(Get-Date -Format s) exit code $LASTEXITCODE"
