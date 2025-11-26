
. .\.venv\Scripts\Activate.ps1
ruff . ; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
black --check . ; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
isort --check-only . ; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
bandit -r .
