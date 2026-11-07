$ErrorActionPreference = 'Stop'
$runner = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot 'ensure-bridge.ps1')).Path
$action = "powershell.exe -NoProfile -ExecutionPolicy Bypass -File `"$runner`""
& schtasks.exe /Create /SC MINUTE /MO 1 /TN 'ForgeCareAI Live Gateway' /TR $action /F
if ($LASTEXITCODE -ne 0) { throw "Could not create the one-minute bridge task (exit $LASTEXITCODE)" }
& schtasks.exe /Query /TN 'ForgeCareAI Live Gateway' /FO LIST
