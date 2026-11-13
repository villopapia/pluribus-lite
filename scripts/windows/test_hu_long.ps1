# Lightweight launcher regression tests; uses a stub, never poker or training.
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$fixture = Join-Path $repo ('data\hu_launcher_test_' + (Get-Date -Format 'yyyyMMdd-HHmmss-fff'))
$scripts = Join-Path $fixture 'scripts\windows'
$binaryDir = Join-Path $fixture 'engine\target\release'
New-Item -ItemType Directory $scripts,$binaryDir | Out-Null
Copy-Item (Join-Path $PSScriptRoot 'hu_long.ps1'),(Join-Path $PSScriptRoot 'hu_long_night.ps1') $scripts
Add-Type -OutputAssembly (Join-Path $binaryDir 'hu_match.exe') -OutputType ConsoleApplication -TypeDefinition @'
using System;
using System.IO;
using System.Linq;
using System.Threading;
public class FakeMatch {
    public static int Main(string[] args) {
        if (Environment.GetEnvironmentVariable("HU_LAUNCHER_TEST_MODE") == "fail") return 7;
        if (Environment.GetEnvironmentVariable("HU_LAUNCHER_TEST_MODE") == "sleep") Thread.Sleep(30000);
        string path = args[Array.IndexOf(args, "--out") + 1];
        int cap = int.Parse(args[Array.IndexOf(args, "--deals") + 1]);
        if (!File.Exists(path)) File.WriteAllText(path, "deal\traw\tchance\tfull\n");
        int done = File.ReadAllLines(path).Length - 1;
        if (done < cap) File.AppendAllText(path, done + "\t0\t0\t0\n");
        Console.WriteLine(string.Join(" ", args));
        Console.WriteLine("priority " + System.Diagnostics.Process.GetCurrentProcess().PriorityClass);
        return 0;
    }
}
'@
function Assert($condition, $message) { if (-not $condition) { throw $message } }
function Invoke-Chunk([string]$dir, [string[]]$extra = @()) {
    $ErrorActionPreference = 'Continue' # expected child errors must reach the exit-code assertion
    & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $scripts 'hu_long.ps1') -Arm final -OutputDir $dir -Deals 3 -Seed 20261110 @extra 2>&1 | Out-File (Join-Path $fixture 'commands.log') -Append
    return $LASTEXITCODE
}
$dir = Join-Path $fixture 'output with spaces'
Assert ((Invoke-Chunk $dir) -eq 0) 'Fresh chunk failed'
Assert ((Invoke-Chunk $dir) -eq 0) 'Resume failed'
$rows = @(Import-Csv (Join-Path $dir 'final.tsv') -Delimiter "`t")
Assert ($rows.Count -eq 2 -and $rows[0].deal -eq '0' -and $rows[1].deal -eq '1') 'Resume duplicated or overwrote rows'
Assert ((Get-Content (Join-Path $dir 'final.log') -Raw) -match 'priority BelowNormal') 'Child priority was not BelowNormal'
$before = (Get-FileHash (Join-Path $dir 'final.tsv')).Hash
Assert ((Invoke-Chunk $dir @('-TurnIters','10')) -ne 0) 'Mixed settings were accepted'
Assert ((Get-FileHash (Join-Path $dir 'final.tsv')).Hash -eq $before) 'Rejected resume changed data'
Assert ((Invoke-Chunk $dir @('-UpdateBinary')) -ne 0) 'Binary replacement was allowed for existing results'
Assert ((Invoke-Chunk (Join-Path $fixture 'memory-preflight') @('-MinAvailableGiB','128')) -ne 0) 'Low-memory preflight was ignored'
$pinned = Join-Path $fixture 'changed-input'
New-Item -ItemType Directory $pinned | Out-Null
@{ Files=@(@{Path=(Join-Path $binaryDir 'hu_match.exe'); Hash='incorrect'}) } | ConvertTo-Json -Depth 4 | Set-Content (Join-Path $pinned 'inputs.json')
Assert ((Invoke-Chunk $pinned) -ne 0) 'Changed input was accepted'
$legacy = Join-Path $fixture 'legacy'
New-Item -ItemType Directory $legacy | Out-Null
Copy-Item (Join-Path $dir 'final.tsv') $legacy
Assert ((Invoke-Chunk $legacy) -ne 0) 'Unmanifested results were accepted'
$mutex = New-Object Threading.Mutex($true, 'Local\pluribus-lite-hu-chunk')
try { Assert ((Invoke-Chunk (Join-Path $fixture 'duplicate')) -ne 0) 'Duplicate chunk was accepted' }
finally { $mutex.ReleaseMutex(); $mutex.Dispose() }
try {
    $env:HU_LAUNCHER_TEST_MODE = 'fail'
    Assert ((Invoke-Chunk (Join-Path $fixture 'failure')) -ne 0) 'Child failure was hidden'
    $env:HU_LAUNCHER_TEST_MODE = 'sleep'
    $timer = [Diagnostics.Stopwatch]::StartNew()
    Assert ((Invoke-Chunk (Join-Path $fixture 'deadline') @('-StopAt',(Get-Date).AddSeconds(4).ToString('o'))) -ne 0) 'Deadline did not fail'
    Assert ($timer.Elapsed.TotalSeconds -lt 15) 'Deadline did not stop the child promptly'
    Assert (-not (Get-Process hu_match -ErrorAction SilentlyContinue)) 'Child survived deadline'
    Assert ((Invoke-Chunk (Join-Path $fixture 'memory-stop') @('-AbortAvailableGiB','128')) -ne 0) 'Low-memory stop was ignored'
    Assert (-not (Get-Process hu_match -ErrorAction SilentlyContinue)) 'Child survived low-memory stop'
} finally { Remove-Item Env:HU_LAUNCHER_TEST_MODE -ErrorAction SilentlyContinue }
$night = Join-Path $fixture 'night'
$until = (Get-Date).AddHours(3).ToString('HH:mm:ss')
$ErrorActionPreference = 'Continue'
& powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $scripts 'hu_long_night.ps1') -Until $until -Deals 2 -OutputDir $night -Seed 20261110 -KeepAwake -MinAvailableGiB 0.1 -AbortAvailableGiB 0.1 2>&1 | Out-File (Join-Path $fixture 'commands.log') -Append
Assert ($LASTEXITCODE -eq 0) 'Night driver failed'
$ErrorActionPreference = 'Stop'
$log = Get-Content (Join-Path $night 'night.log') -Raw
Assert ($log -match 'chunk 1: arm final' -and $log -match 'chunk 2: arm average' -and $log -match 'night end after 4 chunks') 'Alternation or completed-deal stop failed'
Assert (@(Import-Csv (Join-Path $night 'average.tsv') -Delimiter "`t").Count -eq 2) 'Average arm did not finish'
Assert ((Get-Content (Join-Path $night 'settings.json') -Raw) -match '"Seed":20261110') 'Seed was not forwarded'
Assert ((Get-Content (Join-Path $night 'final.log') -Raw) -match 'hard stop') 'Deadline was not forwarded'
$mutex = New-Object Threading.Mutex($true, 'Local\pluribus-lite-hu-night')
try {
    $ErrorActionPreference = 'Continue'
    & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $scripts 'hu_long_night.ps1') -OutputDir (Join-Path $fixture 'duplicate-night') 2>&1 | Out-File (Join-Path $fixture 'commands.log') -Append
    Assert ($LASTEXITCODE -ne 0) 'Duplicate night driver was accepted'
    $ErrorActionPreference = 'Stop'
} finally { $mutex.ReleaseMutex(); $mutex.Dispose() }
try {
    $env:HU_LAUNCHER_TEST_MODE = 'fail'
    $ErrorActionPreference = 'Continue'
    $failedNight = Join-Path $fixture 'failed-night'
    & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $scripts 'hu_long_night.ps1') -Until $until -OutputDir $failedNight 2>&1 | Out-File (Join-Path $fixture 'commands.log') -Append
    Assert ($LASTEXITCODE -ne 0) 'Night driver hid a failed chunk'
    Assert (-not ((Get-Content (Join-Path $failedNight 'night.log') -Raw) -match 'chunk 2:')) 'Night driver continued after a failure'
} finally {
    Remove-Item Env:HU_LAUNCHER_TEST_MODE -ErrorAction SilentlyContinue
    $ErrorActionPreference = 'Stop'
}
Write-Output "PASS: isolated paths, resume, settings/legacy guards, locks, failure propagation, hard deadline, alternation, seed forwarding, completion stop. Evidence: $fixture"
