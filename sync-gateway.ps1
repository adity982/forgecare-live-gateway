$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
$release = Join-Path $projectRoot 'outputs\ForgeCareAI-Release'
$dataPathFile = Join-Path $release '.phone-federation-data-path'
$data = if (Test-Path -LiteralPath $dataPathFile) { (Get-Content -LiteralPath $dataPathFile -Raw).Trim() } else { Join-Path $env:LOCALAPPDATA 'ForgeCareAI\data' }
$taskLog = Join-Path $data 'logs\sync-gateway-task.log'
New-Item -ItemType Directory -Path (Split-Path -Parent $taskLog) -Force | Out-Null
Start-Transcript -Path $taskLog -Append | Out-Null
trap {
    [Console]::Error.WriteLine($_.ToString())
    Stop-Transcript | Out-Null
    exit 1
}
$cloudflared = Join-Path $projectRoot 'work\cloudflared-2026.10.6.exe'
$logs = Join-Path $data 'logs'
New-Item -ItemType Directory -Path $logs -Force | Out-Null

function Get-TunnelUrl([string]$name) {
    $log = Join-Path $logs "$name.err.log"
    if (-not (Test-Path -LiteralPath $log)) { return $null }
    $matchesFound = [regex]::Matches((Get-Content -LiteralPath $log -Raw), 'https://[a-z0-9-]+\.trycloudflare\.com')
    if ($matchesFound.Count -eq 0) { return $null }
    return $matchesFound[$matchesFound.Count - 1].Value
}

function Test-Tunnel([string]$url, [string]$name) {
    if (-not $url) { return $false }
    $healthPath = if ($name -eq 'cloudflared') { '/_stcore/health' } else { '/health' }
    try {
        # A newly assigned quick-tunnel name can remain negatively cached by
        # Windows DNS even after Cloudflare publishes it. Query public DNS and
        # connect to that address while retaining the correct TLS host name.
        $hostName = ([uri]$url).Host
        $address = Resolve-DnsName -Name $hostName -Type A -Server 1.1.1.1 -ErrorAction Stop |
            Where-Object { $_.IPAddress } | Select-Object -First 1 -ExpandProperty IPAddress
        if (-not $address) { return $false }
        $result = @(& curl.exe --silent --show-error --max-time 6 --resolve "${hostName}:443:${address}" `
            --write-out "`n%{http_code}" ($url + $healthPath) 2>$null)
        if ($LASTEXITCODE -ne 0 -or $result.Count -lt 2 -or $result[-1] -ne '200') { return $false }
        $body = ($result[0..($result.Count - 2)] -join "`n").Trim()
        if ($name -eq 'cloudflared') { return $body -eq 'ok' }
        return ($body | ConvertFrom-Json).ok -eq $true
    } catch { return $false }
}

function Ensure-Tunnel([string]$name, [int]$port) {
    $pidFile = Join-Path $logs "$name.pid"
    $processId = if (Test-Path -LiteralPath $pidFile) { [int](Get-Content -LiteralPath $pidFile -Raw) } else { 0 }
    $process = if ($processId -gt 0) { Get-Process -Id $processId -ErrorAction SilentlyContinue } else { $null }
    $url = Get-TunnelUrl $name
    if ($process -and (Test-Tunnel $url $name)) { return $url }
    if ($process) {
        $command = (Get-CimInstance Win32_Process -Filter "ProcessId = $processId" -ErrorAction SilentlyContinue).CommandLine
        if ($command -match 'cloudflared' -and $command -match "127\.0\.0\.1:$port") {
            Stop-Process -Id $processId -Force
            Start-Sleep -Seconds 2
        }
    }
    $stderr = Join-Path $logs "$name.err.log"
    $stdout = Join-Path $logs "$name.out.log"
    Set-Content -LiteralPath $stderr -Value ''
    $newProcess = Start-Process -FilePath $cloudflared -ArgumentList @('tunnel', '--url', "http://127.0.0.1:$port", '--no-autoupdate') -WorkingDirectory $projectRoot -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
    Set-Content -LiteralPath $pidFile -Value $newProcess.Id
    for ($attempt = 0; $attempt -lt 20; $attempt++) {
        Start-Sleep -Seconds 1
        $url = Get-TunnelUrl $name
        if ($url -and (Test-Tunnel $url $name)) { return $url }
    }
    throw "$name did not become reachable; inspect $stderr"
}

foreach ($url in @('http://127.0.0.1:8000/health', 'http://127.0.0.1:8501/_stcore/health', 'http://127.0.0.1:8502/health')) {
    try { Invoke-WebRequest -Uri $url -TimeoutSec 4 -UseBasicParsing | Out-Null }
    catch {
        $env:FORGECARE_DATA = $data
        & (Join-Path $release 'start-laptop.ps1') | Out-Null
        break
    }
}

$dashboard = Ensure-Tunnel 'cloudflared' 8501
$relay = Ensure-Tunnel 'relay-tunnel' 8502
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
Stop-Transcript | Out-Null
