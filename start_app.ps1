[CmdletBinding()]
param(
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonPath = Join-Path $projectRoot ".venv\Scripts\python.exe"
$logDirectory = Join-Path $projectRoot "data\logs"
$appUrl = "http://127.0.0.1:8501"
$healthUrl = "$appUrl/_stcore/health"

function Stop-WithMessage {
    param([string]$Message)

    Write-Host ""
    Write-Host "Startup failed: $Message" -ForegroundColor Red
    Write-Host "See the setup section in README.md."
    exit 1
}

function Test-LocalUrl {
    param([string]$Url)

    try {
        $response = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 2
        return $response.StatusCode -eq 200
    }
    catch {
        return $false
    }
}

Set-Location $projectRoot

if (-not (Test-Path -LiteralPath $pythonPath)) {
    Stop-WithMessage ".venv was not found. Create it with Python 3.11 and install requirements.txt."
}

try {
    $pythonVersion = & $pythonPath -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
}
catch {
    Stop-WithMessage "The Python executable in .venv could not be run."
}
if ($pythonVersion.Trim() -ne "3.11") {
    Stop-WithMessage "Python 3.11 is required. Current version: $pythonVersion"
}

if (-not (Test-LocalUrl "http://127.0.0.1:11434/api/tags")) {
    Stop-WithMessage "Ollama is not running. Start Ollama and run this launcher again."
}

try {
    $ollamaModels = Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/tags" -TimeoutSec 5
    $installedNames = @($ollamaModels.models | ForEach-Object { $_.name })
}
catch {
    Stop-WithMessage "The Ollama model list could not be read."
}

$requiredModels = @("embeddinggemma", "qwen3:1.7b")
$missingModels = @()
foreach ($requiredModel in $requiredModels) {
    $found = $installedNames | Where-Object {
        $_ -eq $requiredModel -or $_ -eq "$requiredModel`:latest"
    }
    if (-not $found) {
        $missingModels += $requiredModel
    }
}
if ($missingModels.Count -gt 0) {
    $commands = ($missingModels | ForEach-Object { "ollama pull $_" }) -join " / "
    Stop-WithMessage "Required models are missing. Run: $commands"
}

New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null

if (-not (Test-LocalUrl $healthUrl)) {
    Write-Host "Starting the science museum RAG app..." -ForegroundColor Cyan
    $stdoutPath = Join-Path $logDirectory "streamlit.stdout.log"
    $stderrPath = Join-Path $logDirectory "streamlit.stderr.log"
    Start-Process `
        -FilePath $pythonPath `
        -ArgumentList @(
            "-m", "streamlit", "run", "app.py",
            "--server.address=127.0.0.1",
            "--server.port=8501",
            "--browser.gatherUsageStats=false"
        ) `
        -WorkingDirectory $projectRoot `
        -WindowStyle Hidden `
        -RedirectStandardOutput $stdoutPath `
        -RedirectStandardError $stderrPath | Out-Null

    $ready = $false
    for ($attempt = 0; $attempt -lt 30; $attempt++) {
        Start-Sleep -Milliseconds 500
        if (Test-LocalUrl $healthUrl) {
            $ready = $true
            break
        }
    }
    if (-not $ready) {
        Stop-WithMessage "Streamlit did not become ready. Check data\logs\streamlit.stderr.log."
    }
}

Write-Host "App is ready: $appUrl" -ForegroundColor Green
if (-not $NoBrowser) {
    Start-Process $appUrl
}
