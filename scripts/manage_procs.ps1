# Process manager: list/clean edu_agent python processes
# Usage:
#   powershell -File scripts/manage_procs.ps1          # list all python processes
#   powershell -File scripts/manage_procs.ps1 -clean   # clean leftovers (keep Streamlit)
#
# Background: exec session exit does not kill python child processes; leftovers
# compete for CPU and corrupt caches. This script cleans by command-line match.

param([switch]$Clean)

$procs = Get-CimInstance Win32_Process -Filter "Name='python.exe'"
Write-Host "=== python processes ($($procs.Count)) ==="
foreach ($p in $procs) {
    $cmd = $p.CommandLine
    $short = if ($cmd.Length -gt 90) { $cmd.Substring(0, 90) + "..." } else { $cmd }
    $isStreamlit = $cmd -match 'streamlit'
    $isAgent = (-not $isStreamlit) -and ($cmd -match 'edu_agent' -or $cmd -match 'power-ticket')
    $tag = if ($isStreamlit) { '[panel]' } elseif ($isAgent) { '[project]' } else { '[other]' }
    Write-Host ("PID {0,-7} {1} {2}" -f $p.ProcessId, $tag, $short)
}

if ($Clean) {
    Write-Host "`n=== clean leftovers (keep Streamlit) ==="
    $killed = 0
    foreach ($p in $procs) {
        $cmd = $p.CommandLine
        $isStreamlit = $cmd -match 'streamlit'
        if (($cmd -match 'edu_agent') -and (-not $isStreamlit)) {
            & taskkill /PID $p.ProcessId /T /F 2>$null | Out-Null
            Write-Host "  killed PID $($p.ProcessId)"
            $killed++
        }
    }
    Write-Host "clean done, $killed process(es)."
}
