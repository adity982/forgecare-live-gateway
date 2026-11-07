param([switch]$Install)
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
if (-not $env:GROQ_API_KEY) {
    $userGroqKey = [Environment]::GetEnvironmentVariable('GROQ_API_KEY', 'User')
    if ($userGroqKey) { $env:GROQ_API_KEY = $userGroqKey }
}
if (-not $env:FORGECARE_DATA) { $env:FORGECARE_DATA = Join-Path $env:LOCALAPPDATA 'ForgeCareAI\data' }
$python = if ($env:FORGECARE_PYTHON) { $env:FORGECARE_PYTHON } else { Join-Path $PSScriptRoot '.venv\Scripts\python.exe' }
if (-not (Test-Path -LiteralPath $python)) {
    $basePython = if ($env:FORGECARE_BASE_PYTHON) { $env:FORGECARE_BASE_PYTHON } else { 'py' }
    if ($env:FORGECARE_BASE_PYTHON) {
        $baseVersion = & $basePython -c 'import sys; print(sys.version_info.major,sys.version_info.minor,sys.version_info.micro,sep=chr(46))'
        if ($baseVersion -ne '3.14.7') { throw "ForgeCare requires Python 3.14.7; selected interpreter is $baseVersion" }
        & $basePython -m venv .venv
    } else {
        $installedVersion = & py -3.14 -c 'import sys; print(sys.version_info.major,sys.version_info.minor,sys.version_info.micro,sep=chr(46))'
        if ($installedVersion -ne '3.14.7') { throw "ForgeCare requires Python 3.14.7; py -3.14 is $installedVersion" }
        & py -3.14 -m venv .venv
    }
    $Install = $true
}
if (-not (Test-Path -LiteralPath $python)) { throw "Python environment was not created: $python" }
$runtimeVersion = & $python -c 'import sys; print(sys.version_info.major,sys.version_info.minor,sys.version_info.micro,sep=chr(46))'
if ($runtimeVersion -ne '3.14.7') { throw "ForgeCare requires Python 3.14.7; this environment is $runtimeVersion. Remove or change only this project's .venv, then rerun." }
if ($Install) { & $python -m pip install -r requirements.txt }
$logDir = Join-Path $env:FORGECARE_DATA 'logs'
New-Item -ItemType Directory -Path $logDir -Force | Out-Null
$jobs = @(
    @{ Name='operations'; Arguments=@('-m','uvicorn','forgecare.ops_api:app','--host','127.0.0.1','--port','8000') },
    @{ Name='knowledge'; Arguments=@('-m','uvicorn','forgecare.knowledge_api:app','--host','127.0.0.1','--port','8001') },
    @{ Name='camera-relay'; Arguments=@('-m','uvicorn','forgecare.relay_api:app','--host','127.0.0.1','--port','8502') },
    @{ Name='dashboard'; Arguments=@('-m','streamlit','run','forgecare\ui.py','--server.address','127.0.0.1','--server.port','8501','--server.headless','true') }
)
foreach ($job in $jobs) {
    $pidFile = Join-Path $logDir "$($job.Name).pid"
    if (Test-Path -LiteralPath $pidFile) {
        $existingId = [int](Get-Content -LiteralPath $pidFile -Raw)
        $existing = Get-Process -Id $existingId -ErrorAction SilentlyContinue
        $existingCommand = (Get-CimInstance Win32_Process -Filter "ProcessId = $existingId" -ErrorAction SilentlyContinue).CommandLine
        if ($existing -and $existing.Path -like '*python*' -and $existingCommand -like "*$($job.Arguments[2])*") {
            Write-Host "$($job.Name): already running (PID $existingId)"
            continue
        }
    }
    $stdout = Join-Path $logDir "$($job.Name).out.log"
    $stderr = Join-Path $logDir "$($job.Name).err.log"
    $process = Start-Process -FilePath $python -ArgumentList $job.Arguments -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
    Set-Content -LiteralPath $pidFile -Value $process.Id
    Write-Host "$($job.Name): PID $($process.Id)"
}
foreach ($url in @('http://127.0.0.1:8000/health','http://127.0.0.1:8001/health','http://127.0.0.1:8502/health','http://127.0.0.1:8501/_stcore/health')) {
    $ready = $false
    for ($attempt=0; $attempt -lt 30; $attempt++) {
        try {
            $response = Invoke-WebRequest -UseBasicParsing -Uri $url -TimeoutSec 2
            if ($response.StatusCode -eq 200) { $ready = $true; break }
        } catch { Start-Sleep -Seconds 1 }
    }
    if ($ready) { Write-Host "Ready: $url" }
    else { Write-Warning "Not ready: $url. Check $logDir." }
}
Write-Host 'Open http://127.0.0.1:8501 in your browser.'
