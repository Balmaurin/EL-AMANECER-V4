
Param(
  [string]$ComposeFile = ".\ops\stack\docker-compose.yml"
)
docker compose -f $ComposeFile down
