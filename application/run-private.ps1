$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$settingsPath = Join-Path $PSScriptRoot 'private_config.json'
$dataPath = Join-Path $PSScriptRoot 'private_data'
if (-not (Test-Path -LiteralPath $settingsPath) -or -not (Test-Path -LiteralPath (Join-Path $dataPath 'forgecare.sqlite3'))) {
    throw 'Extract the private local ZIP first; its private_config.json and private_data are required.'
}
$settings = Get-Content -LiteralPath $settingsPath -Raw | ConvertFrom-Json
$env:FORGECARE_DATA = $dataPath
$env:HF_HOME = Join-Path $PSScriptRoot 'model_cache'
$env:TORCH_HOME = Join-Path $PSScriptRoot 'model_cache\torch'
if ($settings.GROQ_API_KEY) { $env:GROQ_API_KEY = [string]$settings.GROQ_API_KEY }
if ($settings.ROBOFLOW_API_KEY) { $env:ROBOFLOW_API_KEY = [string]$settings.ROBOFLOW_API_KEY }
$env:PYTHONUTF8 = '1'
$python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
$readyFile = Join-Path $PSScriptRoot '.private-setup-complete'
$wheelhouse = Join-Path $PSScriptRoot 'setup\wheelhouse'
if (-not $env:FORGECARE_PYTHON -and (-not (Test-Path -LiteralPath $python) -or -not (Test-Path -LiteralPath $readyFile))) {
    if (-not (Test-Path -LiteralPath $python)) {
        $version = & py -3.14 -c 'import sys; print(sys.version_info.major,sys.version_info.minor,sys.version_info.micro,sep=chr(46))'
        if ($version -ne '3.14.7') { throw "Install Python 3.14.7 first; py -3.14 is $version" }
        & py -3.14 -m venv .venv
    }
    $runtimeVersion = & $python -c 'import sys; print(sys.version_info.major,sys.version_info.minor,sys.version_info.micro,sep=chr(46))'
    if ($runtimeVersion -ne '3.14.7') { throw "Private bundle requires Python 3.14.7; .venv is $runtimeVersion" }
    if (Test-Path -LiteralPath $wheelhouse) {
        & $python -m pip install --no-index --find-links $wheelhouse --only-binary=:all: -r requirements.txt
    } else {
        & $python -m pip install --only-binary=:all: -r requirements.txt
    }
    if ($LASTEXITCODE -ne 0) { throw 'Main dependency installation failed' }
    if (Test-Path -LiteralPath $wheelhouse) {
        & $python -m pip install --no-index --find-links $wheelhouse --no-deps -r requirements-vision.txt
    } else {
        & $python -m pip install --no-deps -r requirements-vision.txt
    }
    if ($LASTEXITCODE -ne 0) { throw 'Phone detector dependency installation failed' }
    Set-Content -LiteralPath $readyFile -Value 'Python 3.14.7 private bundle setup completed' -Encoding UTF8
}
& (Join-Path $PSScriptRoot 'run.ps1')
