$ErrorActionPreference = 'Stop'
$releasePython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
$preparedPython = Join-Path $PSScriptRoot '..\..\work\forgecare-venv-3147\Scripts\python.exe'
if (-not $env:FORGECARE_PYTHON -and -not (Test-Path -LiteralPath $releasePython) -and (Test-Path -LiteralPath $preparedPython)) {
    $env:FORGECARE_PYTHON = (Resolve-Path -LiteralPath $preparedPython).Path
}
& (Join-Path $PSScriptRoot 'run.ps1')
