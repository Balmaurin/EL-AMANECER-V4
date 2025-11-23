
Param(
  [string]$Snapshot = ""
)
.\scripts\setup_venv.ps1
python .\rag_cli.py pipeline_ent --snapshot $Snapshot
