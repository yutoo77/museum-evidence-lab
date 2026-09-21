# Pure process-record tests. No real process or Ollama endpoint is touched.
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
. (Join-Path $repoRoot 'lab_process_helpers.ps1')
$script:fakeProcesses = @{}
$script:fakeListeners = @()
$script:checks = 0
function Get-LabProcessInfo([int]$ProcessId) { $script:fakeProcesses[$ProcessId] }
function Get-LabListeners([int]$Port) { @($script:fakeListeners | Where-Object { $_.LocalPort -eq $Port }) }
function Get-LabExpectedExecutable([string]$Service, [string]$Root) {
    if ($Service -eq 'web') { return Join-Path $Root '.venv\Scripts\python.exe' }
    return 'C:\TestOllama\ollama.exe'
}
function Assert-True([bool]$Condition, [string]$Message) {
    if (-not $Condition) { throw $Message }
    $script:checks++
}
function Assert-Rejected([scriptblock]$Action, [string]$Message) {
    $rejected = $false
    try { & $Action | Out-Null } catch { $rejected = $true }
    Assert-True $rejected $Message
}
$testDirectory = Join-Path ([IO.Path]::GetTempPath()) ('museum-lab-launcher-' + [Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $testDirectory | Out-Null
try {
    $created = (Get-Date).AddMinutes(-1)
    $script:fakeProcesses[101] = [PSCustomObject]@{
        ProcessId=101; ParentProcessId=1; CreationDate=$created
        ExecutablePath=(Join-Path $repoRoot '.venv\Scripts\python.exe')
        CommandLine='python -m streamlit run lab_app.py --server.address=127.0.0.1 --server.port=8502'
    }
    $script:fakeProcesses[102] = [PSCustomObject]@{
        ProcessId=102; ParentProcessId=101; CreationDate=$created.AddSeconds(1)
        ExecutablePath='C:\Python311\python.exe'; CommandLine='python -m streamlit run lab_app.py'
    }
    $script:fakeListeners = @([PSCustomObject]@{LocalPort=8502;LocalAddress='127.0.0.1';OwningProcess=102})
    $record = New-LabOwnershipRecord 'web' $repoRoot $testDirectory 101
    Assert-True ($record.RootPid -eq 101 -and $record.ListenerPid -eq 102) 'venv child was not recognized'
    Save-LabOwnershipRecord 'web' $testDirectory $record
    Assert-True ((Confirm-LabOwnership 'web' $repoRoot $testDirectory).RootPid -eq 101) 'saved identity was not accepted'

    $script:fakeProcesses[101].CreationDate = $created.AddSeconds(2)
    Assert-Rejected { Confirm-LabOwnership 'web' $repoRoot $testDirectory } 'reused root PID was accepted'
    $script:fakeProcesses[101].CreationDate = $created
    $script:fakeProcesses[102].CreationDate = $created.AddSeconds(3)
    Assert-Rejected { Confirm-LabOwnership 'web' $repoRoot $testDirectory } 'reused listener PID was accepted'
    $script:fakeProcesses[102].CreationDate = $created.AddSeconds(1)
    $script:fakeListeners[0].LocalAddress = '0.0.0.0'
    Assert-Rejected { Confirm-LabOwnership 'web' $repoRoot $testDirectory } 'non-loopback listener was accepted'
    $script:fakeListeners[0].LocalAddress = '127.0.0.1'
    $script:fakeProcesses[102].ParentProcessId = 999
    Assert-Rejected { Confirm-LabOwnership 'web' $repoRoot $testDirectory } 'unrelated healthy web process was accepted'
    $script:fakeProcesses[102].ParentProcessId = 101
    $script:fakeProcesses[101].ExecutablePath = 'C:\OtherApp\python.exe'
    Assert-Rejected { Confirm-LabOwnership 'web' $repoRoot $testDirectory } 'wrong executable was accepted'
    $script:fakeProcesses[101].ExecutablePath = Join-Path $repoRoot '.venv\Scripts\python.exe'

    $script:fakeProcesses[201] = [PSCustomObject]@{
        ProcessId=201; ParentProcessId=1; CreationDate=$created
        ExecutablePath='C:\TestOllama\ollama.exe'; CommandLine='ollama.exe serve'
    }
    $script:fakeListeners += [PSCustomObject]@{LocalPort=11435;LocalAddress='127.0.0.1';OwningProcess=201}
    Assert-Rejected { New-LabOwnershipRecord 'ollama' $repoRoot $testDirectory 201 } 'missing no-cloud proof was accepted'
    'Ollama cloud disabled: false' | Set-Content -LiteralPath (Join-Path $testDirectory 'ollama.stderr.log')
    Assert-Rejected { New-LabOwnershipRecord 'ollama' $repoRoot $testDirectory 201 } 'cloud-enabled server was accepted'
    'Ollama cloud disabled: true' | Set-Content -LiteralPath (Join-Path $testDirectory 'ollama.stderr.log')
    $daemon = New-LabOwnershipRecord 'ollama' $repoRoot $testDirectory 201
    Assert-True $daemon.CloudDisabled 'no-cloud proof was not recorded'
    (Get-Item -LiteralPath (Join-Path $testDirectory 'ollama.stderr.log')).LastWriteTimeUtc = $created.ToUniversalTime().AddMinutes(-2)
    Assert-Rejected { New-LabOwnershipRecord 'ollama' $repoRoot $testDirectory 201 } 'stale no-cloud log was accepted'
    Write-Host "Launcher ownership checks passed: $script:checks"
} finally {
    # Only remove files created by this test, then its empty private directory.
    foreach ($name in @('web.process.json','ollama.stderr.log')) {
        $path = Join-Path $testDirectory $name
        if (Test-Path -LiteralPath $path) { Remove-Item -LiteralPath $path }
    }
    Remove-Item -LiteralPath $testDirectory
}
