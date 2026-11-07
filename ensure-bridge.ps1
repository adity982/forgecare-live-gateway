$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
$release = Join-Path $projectRoot 'outputs\ForgeCareAI-Release'
$python = Join-Path $projectRoot 'work\forgecare-venv-3147\Scripts\python.exe'
$dataPathFile = Join-Path $release '.phone-federation-data-path'
$data = if (Test-Path -LiteralPath $dataPathFile) { (Get-Content -LiteralPath $dataPathFile -Raw).Trim() } else { Join-Path $env:LOCALAPPDATA 'ForgeCareAI\data' }
$logs = Join-Path $data 'logs'
$privateKey = Join-Path $data 'bridge-private.pem'
New-Item -ItemType Directory -Path $logs -Force | Out-Null
if (-not (Test-Path -LiteralPath $python)) { throw "Prepared Python interpreter missing: $python" }
if (-not (Test-Path -LiteralPath $privateKey)) { throw "Laptop bridge signing key missing: $privateKey" }

foreach ($url in @('http://127.0.0.1:8000/health', 'http://127.0.0.1:8001/health',
                   'http://127.0.0.1:8501/_stcore/health', 'http://127.0.0.1:8502/health')) {
    try {
        $response = Invoke-WebRequest -Uri $url -TimeoutSec 3 -UseBasicParsing
        if ($response.StatusCode -ne 200) { throw "Local service returned $($response.StatusCode)" }
    } catch {
        $env:FORGECARE_DATA = $data
        $env:FORGECARE_PYTHON = $python
        & (Join-Path $release 'start-laptop.ps1') | Out-Null
        break
    }
}

$pidFile = Join-Path $logs 'bridge-agent.pid'
$existingId = if (Test-Path -LiteralPath $pidFile) { [int](Get-Content -LiteralPath $pidFile -Raw) } else { 0 }
$existing = if ($existingId -gt 0) { Get-CimInstance Win32_Process -Filter "ProcessId = $existingId" -ErrorAction SilentlyContinue } else { $null }
if ($existing -and $existing.CommandLine -match 'bridge_agent\.py') {
    Write-Output "Fixed laptop bridge already running (PID $existingId)."
    exit 0
}

$script = Join-Path $PSScriptRoot 'bridge_agent.py'
$stdout = Join-Path $logs 'bridge-agent.out.log'
$stderr = Join-Path $logs 'bridge-agent.err.log'
$process = Start-Process -FilePath $python -ArgumentList @("`"$script`"", '--key', "`"$privateKey`"") `
    -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
Set-Content -LiteralPath $pidFile -Value $process.Id
Write-Output "Fixed laptop bridge started (PID $($process.Id))."
