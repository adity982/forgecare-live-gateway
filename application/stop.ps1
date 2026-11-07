if (-not $env:FORGECARE_DATA) { $env:FORGECARE_DATA = Join-Path $env:LOCALAPPDATA 'ForgeCareAI\data' }
$logDir = Join-Path $env:FORGECARE_DATA 'logs'
foreach ($name in @('operations','knowledge','camera-relay','dashboard')) {
    $pidFile = Join-Path $logDir "$name.pid"
    if (Test-Path -LiteralPath $pidFile) {
        $processId = [int](Get-Content -LiteralPath $pidFile -Raw)
        $process = Get-Process -Id $processId -ErrorAction SilentlyContinue
        if ($process -and $process.Path -like '*python*') {
            $children = Get-CimInstance Win32_Process -Filter "ParentProcessId = $processId"
            foreach ($child in $children) {
                if ($child.CommandLine -match 'forgecare\.ops_api|forgecare\.knowledge_api|forgecare\.relay_api|streamlit run forgecare\\ui.py') {
                    Stop-Process -Id $child.ProcessId -ErrorAction SilentlyContinue
                }
            }
            Stop-Process -Id $processId -ErrorAction SilentlyContinue
        }
        Remove-Item -LiteralPath $pidFile
    }
}
