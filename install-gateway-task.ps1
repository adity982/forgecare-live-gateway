$ErrorActionPreference = 'Stop'
$runner = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot 'sync-gateway.ps1')).Path
$action = "powershell.exe -NoProfile -ExecutionPolicy Bypass -File `"$runner`""
& schtasks.exe /Create /SC MINUTE /MO 5 /TN 'ForgeCareAI Live Gateway' /TR $action /F
if ($LASTEXITCODE -ne 0) { throw "Could not create the five-minute gateway task (exit $LASTEXITCODE)" }
& schtasks.exe /Query /TN 'ForgeCareAI Live Gateway' /FO LIST
