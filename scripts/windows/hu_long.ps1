# One <= 2 h chunk of the M4 NLHE long run (docs/m4_validation.md): search agent vs blueprint,
# heads-up, one arm per call. Both arms use the same seed, so they play the same deals.
# Windows-only operational script; hu_match itself stays std-only Rust.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\windows\hu_long.ps1 -Arm final [-UpdateBinary]
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\windows\hu_long.ps1 -Arm average
#
# * BelowNormal priority (set on this process; hu_match inherits it).
# * Resumable with matching settings.json: hu_match skips completed deals; it stops taking new
#   deals after -MaxSecs (default 5400 s; deals in progress then finish, up to ~700 s more), so a chunk ends within ~2 h. Run it again to continue.
# * -OutputDir isolates a new experiment. Legacy results without settings.json are refused.
# * -StopAt enforces the overnight deadline; every chunk also has a 2 h hard limit.
# * Runs a private copy of the binary (<OutputDir>\bin), so rebuilding never touches a running
#   exe. -UpdateBinary refreshes it from engine\target\release (build that first).
# * Output appended to data\hu_long\<arm>.log; summary of both arms (paired):
#   analysis\.venv\Scripts\python analysis\hu_pair.py data\hu_long\final.tsv data\hu_long\average.tsv
param(
    [Parameter(Mandatory)][ValidateSet('final', 'average')][string]$Arm,
    [ValidateRange(1, 1000000)][int]$Deals = 3400,
    [ValidateRange(1, 1000000)][int]$TurnIters = 25600,
    [ValidateRange(1, 1000000)][int]$RiverIters = 6400,
    [ValidateRange(1, 100)][int]$BoardsPerIter = 1,
    [ValidateRange(0, 2147483647)][int]$Seed = 2,
    [ValidateRange(1, 12)][int]$Threads = 12,
    [ValidateRange(1, 5400)][int]$MaxSecs = 5400,
    [string]$OutputDir,
    [datetime]$StopAt = [datetime]::MaxValue,
    [ValidateRange(0, 128)][double]$MinAvailableGiB = 0,
    [ValidateRange(0, 128)][double]$AbortAvailableGiB = 0,
    [switch]$UpdateBinary
)

$ErrorActionPreference = 'Stop'
$repo   = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$engine = Join-Path $repo 'engine'
$dir    = Join-Path $repo 'data\hu_long'
if ($OutputDir) { $dir = [IO.Path]::GetFullPath($OutputDir) }
$mutex = New-Object Threading.Mutex($false, 'Local\pluribus-lite-hu-chunk')
$locked = $false
try {
try { $locked = $mutex.WaitOne(0) } catch [Threading.AbandonedMutexException] { $locked = $true }
if (-not $locked) { throw 'Another evaluation chunk owns the lock' }
if (Get-Process hu_match -ErrorAction SilentlyContinue) { throw 'hu_match is already running' }
if ((Get-Date) -ge $StopAt) { throw 'Evaluation deadline has passed' }
if ($MinAvailableGiB -gt 0) {
    $available = (Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB
    if ($available -lt $MinAvailableGiB) { throw "Insufficient available RAM: $available GiB < $MinAvailableGiB GiB" }
}
# Optional run-specific hashes pin the binary, configs and local opponent artifacts.
$manifestPath = Join-Path $dir 'inputs.json'
if (Test-Path $manifestPath) {
    $manifest = Get-Content $manifestPath -Raw | ConvertFrom-Json
    foreach ($inputFile in $manifest.Files) {
        if ((Get-FileHash -LiteralPath $inputFile.Path -Algorithm SHA256).Hash -ne $inputFile.Hash) {
            throw "Evaluation input changed: $($inputFile.Path)"
        }
    }
    if ($manifest.Checkpoint) {
        $latest = Get-ChildItem -LiteralPath (Split-Path $manifest.Checkpoint) -Filter 'ckpt_*.bin' | Sort-Object Name | Select-Object -Last 1
        if ($latest.FullName -ne $manifest.Checkpoint) { throw 'A different opponent checkpoint would be loaded' }
    }
}
$bindir = Join-Path $dir 'bin'
$exe    = Join-Path $bindir 'hu_match.exe'
New-Item -ItemType Directory -Force $bindir | Out-Null
(Get-Process -Id $PID).PriorityClass = 'BelowNormal'

# A bare deal ID does not encode the seed or search settings. Refuse mixed resumes.
$settings = [ordered]@{ Seed=$Seed; Deals=$Deals; TurnIters=$TurnIters; RiverIters=$RiverIters; BoardsPerIter=$BoardsPerIter; Threads=$Threads }
$settingsJson = $settings | ConvertTo-Json -Compress
$settingsPath = Join-Path $dir 'settings.json'
if (Test-Path $settingsPath) {
    if ((Get-Content $settingsPath -Raw).Trim() -ne $settingsJson) { throw 'Output directory settings differ; use a new directory' }
} elseif ((Test-Path (Join-Path $dir 'final.tsv')) -or (Test-Path (Join-Path $dir 'average.tsv'))) {
    throw 'Existing results have no settings manifest; preserve them and use a new directory'
} else { Set-Content $settingsPath $settingsJson -Encoding ASCII }
if ($UpdateBinary -and ((Test-Path (Join-Path $dir 'final.tsv')) -or (Test-Path (Join-Path $dir 'average.tsv')))) {
    throw 'Do not replace the binary for existing results; use a new output directory'
}
if ($UpdateBinary -or -not (Test-Path $exe)) {
    Copy-Item (Join-Path $engine 'target\release\hu_match.exe') $exe -Force
}

$hmArgs = @('--deals', $Deals, '--seed', $Seed, '--turn-iters', $TurnIters, '--river-iters', $RiverIters, '--boards-per-iter', $BoardsPerIter,
            '--threads', $Threads, '--max-secs', $MaxSecs, '--measure-expl', '--out', ('"' + (Join-Path $dir "$Arm.tsv") + '"'))
if ($Arm -eq 'average') { $hmArgs += '--play-average' }
$log = Join-Path $dir "$Arm.log"
Add-Content $log "=== $(Get-Date -Format s) start: $($hmArgs -join ' ')"
# Direct child, separately redirected files: no shell quoting or stderr-to-error conversion.
$tag = '{0}.{1}.{2}' -f $Arm, (Get-Date -Format 'yyyyMMdd-HHmmss-fff'), $PID
$stdout = Join-Path $dir "$tag.stdout.log"
$stderr = Join-Path $dir "$tag.stderr.log"
$deadline = (Get-Date).AddHours(2)
if ($StopAt -lt $deadline) { $deadline = $StopAt }
$p = Start-Process -FilePath $exe -ArgumentList $hmArgs -WorkingDirectory $engine -WindowStyle Hidden -PassThru -RedirectStandardOutput $stdout -RedirectStandardError $stderr
$null = $p.Handle # retain the exit code after a fast child exits (Windows PowerShell 5.1)
try {
    $p.PriorityClass = 'BelowNormal'
    Add-Content $log "PID $($p.Id); hard stop $($deadline.ToString('s')); stdout $stdout; stderr $stderr"
    $timedOut = $false
    $lowMemory = $false
    $nextMemoryCheck = Get-Date
    $resourceLog = Join-Path $dir 'resources.csv'
    while (-not $p.WaitForExit(500)) {
        if ((Get-Date) -ge $deadline) { $timedOut = $true; $p.Kill(); break }
        if ($AbortAvailableGiB -gt 0 -and (Get-Date) -ge $nextMemoryCheck) {
            $available = (Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB
            $p.Refresh()
            [pscustomobject]@{ Time=(Get-Date -Format o); Arm=$Arm; PID=$p.Id; AvailableGiB=$available; PrivateGiB=($p.PrivateMemorySize64 / 1GB); WorkingSetGiB=($p.WorkingSet64 / 1GB) } |
                Export-Csv $resourceLog -Append -NoTypeInformation
            $nextMemoryCheck = (Get-Date).AddSeconds(10)
            if ($available -lt $AbortAvailableGiB) {
                $lowMemory = $true
                if (-not $p.HasExited) { $p.Kill() }
                break
            }
        }
    }
    $p.WaitForExit()
    $code = $p.ExitCode
    Get-Content $stdout | Add-Content $log
    Get-Content $stderr | Add-Content $log
    Add-Content $log "=== $(Get-Date -Format s) exit code $code; hard timeout $timedOut; low memory $lowMemory"
    if ($lowMemory) { throw 'Available RAM below reserve; evaluator stopped, inspect TSV before resuming' }
    if ($timedOut) { throw 'Evaluation hard deadline reached; inspect TSV before resuming' }
    if ($code -ne 0) { throw "hu_match failed with exit code $code" }
} finally {
    if (-not $p.HasExited) { $p.Kill(); $p.WaitForExit() }
    $p.Dispose()
}
} finally {
    if ($locked) { $mutex.ReleaseMutex() }
    $mutex.Dispose()
}
