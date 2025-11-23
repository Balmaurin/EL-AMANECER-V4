
Param(
  [string]$ComposeFile = ".\ops\stack\docker-compose.yml"
)
docker compose -f $ComposeFile up -d --build
Write-Host "RAG API   -> http://localhost:8080"
Write-Host "Prometheus-> http://localhost:9090"
Write-Host "Grafana   -> http://localhost:3000  (admin / admin)"
