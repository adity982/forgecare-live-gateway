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

$services = @(
    @{ Name='operations'; Url='http://127.0.0.1:8000/health'; Command='forgecare.ops_api:app' },
    @{ Name='knowledge'; Url='http://127.0.0.1:8001/health'; Command='forgecare.knowledge_api:app' },
    @{ Name='dashboard'; Url='http://127.0.0.1:8501/_stcore/health'; Command='streamlit run forgecare\ui.py' },
    @{ Name='camera-relay'; Url='http://127.0.0.1:8502/health'; Command='forgecare.relay_api:app' }
)
$needsStart = $false
foreach ($service in $services) {
    try {
        $response = Invoke-WebRequest -Uri $service.Url -TimeoutSec 3 -UseBasicParsing
        if ($response.StatusCode -ne 200) { throw "Local service returned $($response.StatusCode)" }
    } catch {
        $needsStart = $true
        $servicePidFile = Join-Path $logs "$($service.Name).pid"
        if (Test-Path -LiteralPath $servicePidFile) {
            $serviceId = [int](Get-Content -LiteralPath $servicePidFile -Raw)
            $parent = Get-CimInstance Win32_Process -Filter "ProcessId = $serviceId" -ErrorAction SilentlyContinue
            if ($parent -and $parent.CommandLine.Contains($service.Command)) {
                $children = Get-CimInstance Win32_Process | Where-Object {
                    $_.ParentProcessId -eq $serviceId -and $_.CommandLine -and $_.CommandLine.Contains($service.Command)
                }
                foreach ($child in $children) { Stop-Process -Id $child.ProcessId -Force -ErrorAction SilentlyContinue }
                Stop-Process -Id $serviceId -Force -ErrorAction SilentlyContinue
                Write-Output "$($service.Name) was unresponsive; restarting its verified process."
            }
        }
    }
}
if ($needsStart) {
    $env:FORGECARE_DATA = $data
    $env:FORGECARE_PYTHON = $python
    & (Join-Path $release 'start-laptop.ps1') | Out-Null
    foreach ($service in $services) {
        $response = Invoke-WebRequest -Uri $service.Url -TimeoutSec 5 -UseBasicParsing
        if ($response.StatusCode -ne 200) { throw "$($service.Name) did not recover" }
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
