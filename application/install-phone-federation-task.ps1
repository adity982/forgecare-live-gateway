$DataPath = $env:FORGECARE_DATA
$ErrorActionPreference = 'Stop'
$runner = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot 'run-phone-federation.ps1')).Path
$preparedPython = Join-Path $PSScriptRoot '..\..\work\forgecare-venv-3147\Scripts\python.exe'
$localPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
$python = if (Test-Path -LiteralPath $localPython) { (Resolve-Path -LiteralPath $localPython).Path } elseif (Test-Path -LiteralPath $preparedPython) { (Resolve-Path -LiteralPath $preparedPython).Path } else { throw 'ForgeCare Python 3.14.7 environment is unavailable' }
Push-Location $PSScriptRoot
try {
    if (-not $DataPath) {
        $privateData = Join-Path $PSScriptRoot 'private_data'
        $dataPath = if (Test-Path -LiteralPath (Join-Path $privateData 'forgecare.sqlite3')) { $privateData } else { (& $python -c 'from forgecare.config import DATA; print(DATA)').Trim() }
    } else {
        $dataPath = $DataPath
    }
    if (-not (Test-Path -LiteralPath (Join-Path $dataPath 'forgecare.sqlite3'))) { throw "ForgeCare database missing at $dataPath" }
    Set-Content -LiteralPath (Join-Path $PSScriptRoot '.phone-federation-data-path') -Value $dataPath -Encoding ASCII
} finally {
    Pop-Location
}
$action = "powershell.exe -NoProfile -ExecutionPolicy Bypass -File `"$runner`""
& schtasks.exe /Create /SC HOURLY /MO 6 /TN 'ForgeCareAI Phone FedAvg' /TR $action /F
if ($LASTEXITCODE -ne 0) { throw "Could not create the six-hour scheduled task (exit $LASTEXITCODE)" }
& schtasks.exe /Query /TN 'ForgeCareAI Phone FedAvg' /FO LIST
