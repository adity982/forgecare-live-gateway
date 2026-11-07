$ErrorActionPreference = 'Stop'
$env:FORGECARE_DATA = Join-Path $PSScriptRoot 'private_data'
& (Join-Path $PSScriptRoot 'stop.ps1')
