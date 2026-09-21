[CmdletBinding(SupportsShouldProcess = $true)]
param()

$ErrorActionPreference = 'Stop'
$stopFailed = $false
$labLog = Join-Path $PSScriptRoot 'data\lab\processes'
. (Join-Path $PSScriptRoot 'lab_process_helpers.ps1')
foreach ($service in @('web','ollama')) {
    $port = if ($service -eq 'web') { 8502 } else { 11435 }
    if (@(Get-LabListeners $port).Count -eq 0) { continue }
    try {
        $record = Confirm-LabOwnership $service $PSScriptRoot $labLog
        if (-not $PSCmdlet.ShouldProcess("$service process $($record.RootPid)", 'Stop verified comparison process tree')) { continue }
        # Stop the verified launcher tree, including the venv Python child listener.
        & taskkill.exe /PID $record.RootPid /T /F
        if ($LASTEXITCODE -ne 0) { throw 'Process termination failed.' }
        foreach ($suffix in @('pid','process.json')) {
            $recordPath = Join-Path $labLog "$service.$suffix"
            if (Test-Path -LiteralPath $recordPath) { Remove-Item -LiteralPath $recordPath }
        }
    } catch {
        $stopFailed = $true
        Write-Warning "Skipped unverified process for ${service}: $($_.Exception.Message)"
    }
}
if ($stopFailed) { throw 'One or more services could not be verified; no unrecognized process was stopped.' }
