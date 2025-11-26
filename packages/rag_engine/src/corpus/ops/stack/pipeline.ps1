
Param(
  [string]$ComposeFile = ".\ops\stack\docker-compose.yml"
)
docker compose -f $ComposeFile run --rm rag-api python rag_cli.py pipeline
