# Standard test flow: clean leftovers -> single-instance pytest -> show result
# Usage: powershell -File scripts/run_tests.ps1

$ErrorActionPreference = "Continue"

# 1. clean leftovers (keep Streamlit)
& powershell -NoProfile -File "$PSScriptRoot\manage_procs.ps1" -Clean

# 2. single-instance pytest
$py = "D:\difyzhinengti\xiangmu\envs\edu_agent\python.exe"
$out = "D:\edu_agent\data\pytest_all.txt"
Write-Host "`n=== run full tests (single instance) ==="
Push-Location "D:\edu_agent"
& $py -m pytest backend -q --no-header *> $out
$code = $LASTEXITCODE
Pop-Location

# 3. show result
Write-Host "EXIT=$code"
Get-Content $out | Select-Object -Last 6

# 4. confirm no leftovers
Write-Host "`n=== processes after run ==="
& powershell -NoProfile -File "$PSScriptRoot\manage_procs.ps1"
