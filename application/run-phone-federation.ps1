$ErrorActionPreference = 'Stop'
$preparedPython = Join-Path $PSScriptRoot '..\..\work\forgecare-venv-3147\Scripts\python.exe'
$localPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
$python = if (Test-Path -LiteralPath $localPython) { (Resolve-Path -LiteralPath $localPython).Path } elseif (Test-Path -LiteralPath $preparedPython) { (Resolve-Path -LiteralPath $preparedPython).Path } else { throw 'ForgeCare Python 3.14.7 environment is unavailable' }
$dataPathFile = Join-Path $PSScriptRoot '.phone-federation-data-path'
if ($env:FORGECARE_DATA) {
    $env:FORGECARE_DATA = (Resolve-Path -LiteralPath $env:FORGECARE_DATA).Path
} elseif (Test-Path -LiteralPath (Join-Path $PSScriptRoot 'private_data\forgecare.sqlite3')) {
    $env:FORGECARE_DATA = Join-Path $PSScriptRoot 'private_data'
} elseif (Test-Path -LiteralPath $dataPathFile) {
    $env:FORGECARE_DATA = (Get-Content -LiteralPath $dataPathFile -Raw).Trim()
} else {
    $env:FORGECARE_DATA = Join-Path $env:LOCALAPPDATA 'ForgeCareAI\data'
}
$logDir = Join-Path $env:FORGECARE_DATA 'logs'
New-Item -ItemType Directory -Path $logDir -Force | Out-Null
$logFile = Join-Path $logDir 'phone-federation-task.log'
"[$(Get-Date -Format o)] Starting scheduled phone federation with $python; data=$env:FORGECARE_DATA; db-bytes=$((Get-Item -LiteralPath (Join-Path $env:FORGECARE_DATA 'forgecare.sqlite3') -ErrorAction SilentlyContinue).Length)" | Add-Content -LiteralPath $logFile
Push-Location $PSScriptRoot
try {
    $stdout = Join-Path $logDir 'phone-federation-stdout.log'
    $stderr = Join-Path $logDir 'phone-federation-stderr.log'
    $process = Start-Process -FilePath $python -ArgumentList '-m forgecare.phone_federation_schedule' -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -Wait -PassThru
    if ($process.ExitCode -ne 0) { throw "Phone federation attempt exited with $($process.ExitCode): $(Get-Content -LiteralPath $stderr -Raw)" }
    Get-Content -LiteralPath $stdout | Add-Content -LiteralPath $logFile
    "[$(Get-Date -Format o)] Finished successfully" | Add-Content -LiteralPath $logFile
} catch {
    "[$(Get-Date -Format o)] Failed: $_" | Add-Content -LiteralPath $logFile
    throw
} finally {
    Pop-Location
}
