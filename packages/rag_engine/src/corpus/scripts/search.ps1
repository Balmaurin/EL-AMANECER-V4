# PowerShell wrapper for local RAG search
# Usage: .\scripts\search.ps1 "query" [top_k]

param(
    [Parameter(Mandatory=$true)][string]$query,
    [Parameter(Mandatory=$false)][int]$top_k = 5
)

& ".\.venv\Scripts\python.exe" scripts/search_entrypoint.py "$query" $top_k
