
Param(
  [string]$ComposeFile = ".\ops\monitoring\docker-compose.yml"
)
docker compose -f $ComposeFile down
