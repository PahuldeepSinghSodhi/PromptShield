# Runs the Locust load test against several worker x ORT-thread configurations.
#   powershell -File scripts\load_matrix.ps1
$ErrorActionPreference = "Continue"  # locust logs to stderr
$root = Split-Path $PSScriptRoot -Parent
$py = Join-Path $root ".venv\Scripts\python.exe"
$locust = Join-Path $root ".venv\Scripts\locust.exe"
$out = Join-Path $root "reports\runtime"
New-Item -ItemType Directory -Force $out | Out-Null

$configs = @(@{w = 1; t = 0}, @{w = 4; t = 4}, @{w = 8; t = 2})
foreach ($c in $configs) {
    $name = "w$($c.w)_t$($c.t)"
    $db = Join-Path $env:TEMP "ps_load_$name.db"
    Remove-Item $db, "$db-wal", "$db-shm" -ErrorAction SilentlyContinue
    $env:PROMPTSHIELD_DATABASE_URL = "sqlite:///$($db -replace '\\','/')"
    $env:PROMPTSHIELD_NER_THREADS = "$($c.t)"
    $env:PROMPTSHIELD_RATE_LIMIT_PER_MINUTE = "1000000"
    $server = Start-Process -FilePath $py -ArgumentList "-m uvicorn promptshield.serving.app:app --port 8010 --workers $($c.w) --log-level warning" -WorkingDirectory $root -PassThru -WindowStyle Hidden
    for ($i = 0; $i -lt 60; $i++) {
        try { Invoke-RestMethod http://localhost:8010/v1/health | Out-Null; break } catch { Start-Sleep -Milliseconds 500 }
    }
    & $locust -f (Join-Path $root "load_tests\locustfile.py") --headless -u 20 -r 5 -t 30s --host http://localhost:8010 --csv (Join-Path $out "locust_$name") --only-summary 2>$null | Out-Null
    Get-CimInstance Win32_Process -Filter "ParentProcessId=$($server.Id)" | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
    Stop-Process -Id $server.Id -Force -ErrorAction SilentlyContinue
    $agg = Import-Csv (Join-Path $out "locust_${name}_stats.csv") | Where-Object Name -eq "Aggregated"
    "{0,-8} req/s={1,7:N1}  p50={2,5} ms  p95={3,5} ms  fails={4}" -f $name, [double]$agg."Requests/s", $agg."50%", $agg."95%", $agg."Failure Count"
}
