param([switch]$RunSimulation)

$ErrorActionPreference = 'Stop'
$preparedPython = Join-Path $PSScriptRoot '..\..\work\forgecare-venv-3147\Scripts\python.exe'
$localPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
$python = if (Test-Path -LiteralPath $localPython) {
    (Resolve-Path -LiteralPath $localPython).Path
} elseif (Test-Path -LiteralPath $preparedPython) {
    (Resolve-Path -LiteralPath $preparedPython).Path
} else {
    throw 'ForgeCare Python environment is unavailable'
}

if ($RunSimulation) {
    $outputPath = Join-Path $env:LOCALAPPDATA 'ForgeCareAI\synthetic-federation-demo'
    Push-Location $PSScriptRoot
    try {
        & $python -m forgecare.phone_federation_schedule --animate --out $outputPath
        if ($LASTEXITCODE -ne 0) { throw "Synthetic federation simulation failed (exit $LASTEXITCODE)" }
    } finally {
        Pop-Location
    }
    exit 0
}

$action = "powershell.exe -NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`" -RunSimulation"
& schtasks.exe /Create /SC HOURLY /MO 6 /TN 'ForgeCareAI Synthetic FedAvg Demo' /TR $action /F
if ($LASTEXITCODE -ne 0) { throw "Could not create the six-hour synthetic simulation task (exit $LASTEXITCODE)" }
$null = & schtasks.exe /Query /TN 'ForgeCareAI Phone FedAvg' 2>$null
if ($LASTEXITCODE -eq 0) {
    & schtasks.exe /Delete /TN 'ForgeCareAI Phone FedAvg' /F
    if ($LASTEXITCODE -ne 0) { throw "Could not remove the previous phone-data federation task (exit $LASTEXITCODE)" }
}
& schtasks.exe /Query /TN 'ForgeCareAI Synthetic FedAvg Demo' /FO LIST
