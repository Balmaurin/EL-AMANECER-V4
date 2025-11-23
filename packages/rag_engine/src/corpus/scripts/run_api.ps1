
# Carga .env
$envFile = ".env"
if (Test-Path $envFile) {
  Get-Content $envFile | Where-Object {$_ -and $_ -notmatch '^#'} | ForEach-Object {
    $kv = $_.Split('=',2)
    if ($kv.Length -eq 2) {
      $name = $kv[0]; $value = $kv[1]
      $env:$name = $value
    }
  }
}
. .\.venv\Scripts\Activate.ps1
python -m uvicorn server.rag_server:app --host 0.0.0.0 --port 8080
