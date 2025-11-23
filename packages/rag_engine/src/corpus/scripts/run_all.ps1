# One-click: setup venv, pipeline, and run API
Param(
  [int]$Port = 8080
)
.\scripts\setup_venv.ps1
python .\rag_cli.py pipeline
.\scripts\run_api.ps1
