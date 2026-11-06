$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
$release = Join-Path $projectRoot 'outputs\ForgeCareAI-Release'
$cloudflared = Join-Path $projectRoot 'work\cloudflared-2026.10.6.exe'
$data = Join-Path $env:LOCALAPPDATA 'ForgeCareAI\data'
$logs = Join-Path $data 'logs'
New-Item -ItemType Directory -Path $logs -Force | Out-Null

function Test-PublicHealth([string]$url, [string]$suffix) {
    try {
        $response = Invoke-WebRequest -Uri "$url$suffix" -TimeoutSec 10 -UseBasicParsing
        return $response.StatusCode -eq 200
    } catch { return $false }
}

function Get-TunnelUrl([string]$name) {
    $log = Join-Path $logs "$name.err.log"
    if (-not (Test-Path -LiteralPath $log)) { return $null }
    $matchesFound = [regex]::Matches((Get-Content -LiteralPath $log -Raw), 'https://[a-z0-9-]+\.trycloudflare\.com')
    if ($matchesFound.Count -eq 0) { return $null }
    return $matchesFound[$matchesFound.Count - 1].Value
}

function Ensure-Tunnel([string]$name, [int]$port, [string]$healthPath) {
    $pidFile = Join-Path $logs "$name.pid"
    $processId = if (Test-Path -LiteralPath $pidFile) { [int](Get-Content -LiteralPath $pidFile -Raw) } else { 0 }
    $process = if ($processId -gt 0) { Get-Process -Id $processId -ErrorAction SilentlyContinue } else { $null }
    $url = Get-TunnelUrl $name
    $healthy = $false
    if ($process -and $url) {
        for ($attempt = 0; $attempt -lt 3; $attempt++) {
            if (Test-PublicHealth $url $healthPath) { $healthy = $true; break }
            Start-Sleep -Seconds 3
        }
    }
    if ($healthy) { return $url }
    if ($process) { Stop-Process -Id $processId -Force }
    $stderr = Join-Path $logs "$name.err.log"
    $stdout = Join-Path $logs "$name.out.log"
    $newProcess = Start-Process -FilePath $cloudflared -ArgumentList @('tunnel', '--url', "http://127.0.0.1:$port", '--no-autoupdate') -WorkingDirectory $projectRoot -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
    Set-Content -LiteralPath $pidFile -Value $newProcess.Id
    for ($attempt = 0; $attempt -lt 25; $attempt++) {
        Start-Sleep -Seconds 1
        $url = Get-TunnelUrl $name
        if ($url -and (Test-PublicHealth $url $healthPath)) { return $url }
    }
    throw "$name did not become reachable; inspect $stderr"
}

foreach ($url in @('http://127.0.0.1:8000/health', 'http://127.0.0.1:8501/_stcore/health', 'http://127.0.0.1:8502/health')) {
    try { Invoke-WebRequest -Uri $url -TimeoutSec 4 -UseBasicParsing | Out-Null }
    catch {
        & (Join-Path $release 'start-laptop.ps1') | Out-Null
        break
    }
}

$dashboard = Ensure-Tunnel 'cloudflared' 8501 '/_stcore/health'
$relay = Ensure-Tunnel 'relay-tunnel' 8502 '/health'
$targets = @{ dashboard = $dashboard; relay = $relay } | ConvertTo-Json -Depth 2
$targetPath = Join-Path $PSScriptRoot 'targets.json'
$current = Get-Content -LiteralPath $targetPath -Raw | ConvertFrom-Json
if ($current.dashboard -ne $dashboard -or $current.relay -ne $relay) {
    Set-Content -LiteralPath $targetPath -Value $targets -Encoding utf8
    Push-Location $PSScriptRoot
    try {
        & git add -- targets.json
        & git commit -m 'Update live tunnel targets' | Out-Null
        if ($LASTEXITCODE -ne 0) { throw 'Could not save updated tunnel addresses' }
        & git push origin main | Out-Null
        if ($LASTEXITCODE -ne 0) { throw 'Could not publish updated tunnel addresses' }
    } finally { Pop-Location }
}

Set-Content -LiteralPath (Join-Path $data 'dashboard-public-url.txt') -Value 'https://forgecare-ai-live.onrender.com' -Encoding ascii
Set-Content -LiteralPath (Join-Path $data 'relay-public-url.txt') -Value 'https://forgecare-ai-live.onrender.com/camera-relay' -Encoding ascii
Write-Output 'Stable ForgeCare gateway synchronized.'
