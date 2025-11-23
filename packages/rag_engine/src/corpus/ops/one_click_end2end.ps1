
Param(
  [string]$Python = "python"
)
Write-Host "==> Creando entorno y deps..."
.\scripts\setup_venv.ps1
Copy-Item .env.example .env -ErrorAction SilentlyContinue
if (-not (Select-String -Path ".env" -Pattern "^RAG_API_KEY=")) {
  Add-Content .env "RAG_API_KEY=dev_local_key"
}
Write-Host "==> Ingesta (si hay datos en .\data_in)..."
python .\rag_cli.py ingest --src data_in
Write-Host "==> Pipeline..."
python .\rag_cli.py pipeline
Write-Host "==> Arrancando stack Docker (API+Prom+Grafana+Redis)..."
.\ops\stack\up.ps1
