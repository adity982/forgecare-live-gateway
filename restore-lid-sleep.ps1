$ErrorActionPreference = 'Stop'
$originalScheme = '381b4222-f694-41f0-9685-ff5bb260df2e'
& powercfg /setacvalueindex $originalScheme SUB_BUTTONS LIDACTION 1
if ($LASTEXITCODE -ne 0) { throw 'Could not restore the AC lid-close sleep setting' }
if ((& powercfg /getactivescheme) -match $originalScheme) {
    & powercfg /setactive $originalScheme
    if ($LASTEXITCODE -ne 0) { throw 'Could not apply the restored lid-close setting' }
}
Write-Output 'Original AC lid-close sleep setting restored.'
