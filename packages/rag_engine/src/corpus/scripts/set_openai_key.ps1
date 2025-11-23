Param(
  [string]$Key,
  [ValidateSet("dotenv","session","user","all")]
  [string]$Mode = "dotenv"
)

# Si no pasas la clave por parámetro, la pedimos
if (-not $Key -or $Key.Trim().Length -eq 0) {
  $Key = Read-Host -Prompt "Introduce tu OPENAI_API_KEY"
}

if (-not $Key -or $Key.Trim().Length -eq 0) {
  Write-Host "No se proporcionó una clave. Cancelando." -ForegroundColor Yellow
  exit 1
}

switch ($Mode) {
  "dotenv" {
    $dotenvPath = Join-Path -Path (Get-Location) -ChildPath ".env"
    "OPENAI_API_KEY=$Key" | Out-File -FilePath $dotenvPath -Encoding UTF8 -Force
    Write-Host ".env actualizado en $dotenvPath" -ForegroundColor Green
  }
  "session" {
    $env:OPENAI_API_KEY = $Key
    Write-Host "Variable de entorno (sesión) OPENAI_API_KEY establecida." -ForegroundColor Green
  }
  "user" {
    [Environment]::SetEnvironmentVariable("OPENAI_API_KEY", $Key, "User")
    Write-Host "Variable de entorno (usuario) OPENAI_API_KEY guardada." -ForegroundColor Green
  }
  "all" {
    $env:OPENAI_API_KEY = $Key
    Write-Host "Variable de entorno (sesión) OPENAI_API_KEY establecida." -ForegroundColor Green
    [Environment]::SetEnvironmentVariable("OPENAI_API_KEY", $Key, "User")
    Write-Host "Variable de entorno (usuario) OPENAI_API_KEY guardada." -ForegroundColor Green
    $dotenvPath = Join-Path -Path (Get-Location) -ChildPath ".env"
    "OPENAI_API_KEY=$Key" | Out-File -FilePath $dotenvPath -Encoding UTF8 -Force
    Write-Host ".env actualizado en $dotenvPath" -ForegroundColor Green
  }
}

Write-Host "Listo. Vuelve a ejecutar el pipeline: python rag_cli.py pipeline demo" -ForegroundColor Cyan
