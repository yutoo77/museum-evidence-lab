[CmdletBinding()]
param([switch]$NoBrowser)

$ErrorActionPreference = 'Stop'
$labRoot = $PSScriptRoot
$labPython = Join-Path $labRoot '.venv\Scripts\python.exe'
$labLog = Join-Path $labRoot 'data\lab\processes'
$labUrl = 'http://127.0.0.1:8502'
. (Join-Path $labRoot 'lab_process_helpers.ps1')
if (-not (Test-Path -LiteralPath $labPython)) {
    throw 'Create .venv with Python 3.11 and install requirements-lab.txt first.'
}
# Validate the same configuration the UI reads before reusing any daemon.
Push-Location $labRoot
try {
    & $labPython -c "import sys; from src.lab.settings import load_lab_settings; s=load_lab_settings(); sys.exit(0 if s.ollama.base_url == 'http://127.0.0.1:11435' else 'Launcher requires the dedicated local Ollama on port 11435')"
    if ($LASTEXITCODE -ne 0) { throw 'Invalid comparison configuration; no service was started.' }
} finally { Pop-Location }
New-Item -ItemType Directory -Path $labLog -Force | Out-Null

function Test-LabEndpoint([string]$Url) {
    $http = $null
    try {
        $handler = New-Object System.Net.Http.HttpClientHandler
        $handler.UseProxy = $false
        $handler.AllowAutoRedirect = $false
        $http = New-Object System.Net.Http.HttpClient($handler)
        $http.Timeout = [TimeSpan]::FromSeconds(2)
        $response = $http.GetAsync($Url).GetAwaiter().GetResult()
        return $response.IsSuccessStatusCode
    } catch { return $false }
    finally { if ($http) { $http.Dispose() } }
}
Add-Type -AssemblyName System.Net.Http

# Dedicated process environment. The existing Ollama server on 11434 is untouched.
if (@(Get-LabListeners 11435).Count -eq 0) {
    $ollamaExe = (Get-Command ollama -ErrorAction Stop).Source
    $names = @('OLLAMA_HOST','OLLAMA_NO_CLOUD','OLLAMA_CONTEXT_LENGTH','OLLAMA_NUM_PARALLEL','OLLAMA_DEBUG_LOG_REQUESTS','OLLAMA_DEBUG')
    $saved = @{}
    foreach ($name in $names) { $saved[$name] = [Environment]::GetEnvironmentVariable($name, 'Process') }
    try {
        $env:OLLAMA_HOST = '127.0.0.1:11435'
        $env:OLLAMA_NO_CLOUD = '1'
        $env:OLLAMA_CONTEXT_LENGTH = '4096'
        $env:OLLAMA_NUM_PARALLEL = '1'
        $env:OLLAMA_DEBUG_LOG_REQUESTS = 'false'
        $env:OLLAMA_DEBUG = 'false'
        $daemon = Start-Process -FilePath $ollamaExe -ArgumentList @('serve') -WorkingDirectory $labRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $labLog 'ollama.stdout.log') -RedirectStandardError (Join-Path $labLog 'ollama.stderr.log')
        $daemon.Id | Out-File -LiteralPath (Join-Path $labLog 'ollama.pid') -Encoding ascii
    } finally {
        foreach ($name in $names) { [Environment]::SetEnvironmentVariable($name, $saved[$name], 'Process') }
    }
    for ($attempt = 0; $attempt -lt 40; $attempt++) {
        if (Test-LabEndpoint 'http://127.0.0.1:11435/api/version') { break }
        Start-Sleep -Milliseconds 500
    }
    if (Test-LabEndpoint 'http://127.0.0.1:11435/api/version') {
        Save-LabOwnershipRecord 'ollama' $labLog (New-LabOwnershipRecord 'ollama' $labRoot $labLog $daemon.Id)
    }
}
if (-not (Test-LabEndpoint 'http://127.0.0.1:11435/api/version')) { throw 'Local Ollama did not start. Check data/lab/processes.' }

# Local binding alone does not disable cloud. Verify identity and startup proof.
Confirm-LabOwnership 'ollama' $labRoot $labLog | Out-Null
if (@(Get-LabListeners 8502).Count -eq 0) {
    $appPath = '"' + (Join-Path $labRoot 'lab_app.py') + '"'
    $web = Start-Process -FilePath $labPython -ArgumentList @('-m','streamlit','run',$appPath,'--server.address=127.0.0.1','--server.port=8502','--browser.gatherUsageStats=false') -WorkingDirectory $labRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $labLog 'web.stdout.log') -RedirectStandardError (Join-Path $labLog 'web.stderr.log')
    $web.Id | Out-File -LiteralPath (Join-Path $labLog 'web.pid') -Encoding ascii
    for ($attempt = 0; $attempt -lt 40; $attempt++) {
        if (Test-LabEndpoint "$labUrl/_stcore/health") { break }
        Start-Sleep -Milliseconds 500
    }
    if (Test-LabEndpoint "$labUrl/_stcore/health") {
        Save-LabOwnershipRecord 'web' $labLog (New-LabOwnershipRecord 'web' $labRoot $labLog $web.Id)
    }
}
Confirm-LabOwnership 'web' $labRoot $labLog | Out-Null
if (-not (Test-LabEndpoint "$labUrl/_stcore/health")) { throw 'The application did not become ready.' }
Write-Host "Museum Evidence Lab: $labUrl"
if (-not $NoBrowser) { Start-Process $labUrl }
