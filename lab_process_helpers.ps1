# Shared ownership checks. Dot-sourcing this file does not start or stop processes.
function Get-LabProcessInfo([int]$ProcessId) {
    Get-CimInstance Win32_Process -Filter "ProcessId = $ProcessId" -ErrorAction SilentlyContinue
}

function Get-LabListeners([int]$Port) {
    @(Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)
}

function Get-LabCreationTicks($Process) {
    if (-not $Process -or -not $Process.CreationDate) { throw 'Cannot verify process start time.' }
    return ([DateTime]$Process.CreationDate).ToUniversalTime().Ticks.ToString()
}

function Test-LabDescendant([int]$ChildId, [int]$RootId) {
    $currentId = $ChildId
    for ($depth = 0; $depth -lt 10; $depth++) {
        if ($currentId -eq $RootId) { return $true }
        $current = Get-LabProcessInfo $currentId
        if (-not $current -or $current.ParentProcessId -le 0 -or $current.ParentProcessId -eq $currentId) { return $false }
        $currentId = [int]$current.ParentProcessId
    }
    return $false
}

function Get-LabExpectedExecutable([string]$Service, [string]$Root) {
    if ($Service -eq 'web') { return [IO.Path]::GetFullPath((Join-Path $Root '.venv\Scripts\python.exe')) }
    return [IO.Path]::GetFullPath((Get-Command ollama -ErrorAction Stop).Source)
}

function Assert-LabRootProcess([string]$Service, [string]$Root, $Process) {
    if (-not $Process) { throw "Recorded $Service process is not running." }
    $expected = Get-LabExpectedExecutable $Service $Root
    if (-not [string]::Equals([string]$Process.ExecutablePath, $expected, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Recorded $Service executable does not belong to this application."
    }
    if ($Service -eq 'web') {
        if ($Process.CommandLine -notmatch '\s-m\s+streamlit\s+run\s+' -or
            $Process.CommandLine -notmatch '(?i)(?:^|[\\\s"])lab_app\.py(?:\s|"|$)' -or
            $Process.CommandLine -notmatch '--server\.address=127\.0\.0\.1(?:\s|$)' -or
            $Process.CommandLine -notmatch '--server\.port=8502(?:\s|$)') {
            throw 'Recorded web process does not match the comparison application.'
        }
    } elseif ($Process.CommandLine -notmatch '\sserve(?:\s|$)') {
        throw 'Recorded Ollama process is not a dedicated server.'
    }
}

function Assert-LabCloudDisabled([string]$LogDirectory, $Process) {
    $path = Join-Path $LogDirectory 'ollama.stderr.log'
    if (-not (Test-Path -LiteralPath $path)) { throw 'Dedicated Ollama startup log is missing.' }
    $log = Get-Item -LiteralPath $path
    $started = ([DateTime]$Process.CreationDate).ToUniversalTime()
    if ($log.LastWriteTimeUtc -lt $started -or
        -not (Select-String -LiteralPath $path -SimpleMatch 'Ollama cloud disabled: true' -Quiet)) {
        throw 'Dedicated Ollama has not confirmed cloud-disabled mode for this process.'
    }
}

function New-LabOwnershipRecord([string]$Service, [string]$Root, [string]$LogDirectory, [int]$RootId) {
    $process = Get-LabProcessInfo $RootId
    Assert-LabRootProcess $Service $Root $process
    $port = if ($Service -eq 'web') { 8502 } else { 11435 }
    $listeners = @(Get-LabListeners $port)
    $owners = @($listeners | Select-Object -ExpandProperty OwningProcess -Unique)
    if ($owners.Count -ne 1 -or @($listeners | Where-Object { $_.LocalAddress -ne '127.0.0.1' }).Count -gt 0 -or
        -not (Test-LabDescendant ([int]$owners[0]) $RootId)) {
        throw "Port $port is not exclusively owned by the recorded comparison process."
    }
    $listener = Get-LabProcessInfo ([int]$owners[0])
    if (-not $listener -or ([DateTime]$listener.CreationDate) -lt ([DateTime]$process.CreationDate)) {
        throw 'Listener does not belong to this process lifetime.'
    }
    if ($Service -eq 'ollama') { Assert-LabCloudDisabled $LogDirectory $process }
    return [PSCustomObject]@{
        SchemaVersion = 1
        Service = $Service
        RootDirectory = [IO.Path]::GetFullPath($Root)
        RootPid = $RootId
        RootCreatedTicks = Get-LabCreationTicks $process
        RootExecutable = [string]$process.ExecutablePath
        RootCommandLine = [string]$process.CommandLine
        ListenerPid = [int]$listener.ProcessId
        ListenerCreatedTicks = Get-LabCreationTicks $listener
        CloudDisabled = ($Service -eq 'ollama')
    }
}

function Save-LabOwnershipRecord([string]$Service, [string]$LogDirectory, $Record) {
    $Record | ConvertTo-Json -Depth 3 | Set-Content -LiteralPath (Join-Path $LogDirectory "$Service.process.json") -Encoding UTF8
}

function Confirm-LabOwnership([string]$Service, [string]$Root, [string]$LogDirectory) {
    $recordPath = Join-Path $LogDirectory "$Service.process.json"
    if (Test-Path -LiteralPath $recordPath) {
        $record = Get-Content -LiteralPath $recordPath -Raw | ConvertFrom-Json
        if ($record.SchemaVersion -ne 1 -or $record.Service -ne $Service -or
            -not [string]::Equals([string]$record.RootDirectory, [IO.Path]::GetFullPath($Root), [StringComparison]::OrdinalIgnoreCase)) {
            throw 'The saved process record is not for this application.'
        }
        $fresh = New-LabOwnershipRecord $Service $Root $LogDirectory ([int]$record.RootPid)
        foreach ($field in @('RootCreatedTicks','RootExecutable','RootCommandLine','ListenerPid','ListenerCreatedTicks')) {
            if ([string]$record.$field -cne [string]$fresh.$field) { throw "The recorded $Service process identity has changed." }
        }
        if ($Service -eq 'ollama' -and $record.CloudDisabled -ne $true) { throw 'No verified cloud-disabled startup record exists.' }
        return $fresh
    }
    # One-time compatibility with the first launcher. Never adopt an arbitrary listener.
    $pidPath = Join-Path $LogDirectory "$Service.pid"
    if (-not (Test-Path -LiteralPath $pidPath)) { throw "An unrecognized process is using the $Service port." }
    $rootId = 0
    if (-not [int]::TryParse((Get-Content -LiteralPath $pidPath -Raw).Trim(), [ref]$rootId) -or $rootId -le 0) { throw 'Invalid process record.' }
    $process = Get-LabProcessInfo $rootId
    if (-not $process -or (Get-Item -LiteralPath $pidPath).LastWriteTimeUtc -lt ([DateTime]$process.CreationDate).ToUniversalTime()) {
        throw 'The legacy process record belongs to a previous process lifetime.'
    }
    $fresh = New-LabOwnershipRecord $Service $Root $LogDirectory $rootId
    Save-LabOwnershipRecord $Service $LogDirectory $fresh
    return $fresh
}
