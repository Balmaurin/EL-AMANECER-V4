Write-Host "Iniciando EL-AMANECERV3..."

# Detener procesos anteriores
Get-Process -Name "python" -ErrorAction SilentlyContinue | Where-Object {$_.MainWindowTitle -like "*uvicorn*" -or $_.MainWindowTitle -like "*backend*"} | Stop-Process -Force
Start-Sleep -Seconds 2

# Iniciar Backend
Write-Host "Iniciando Backend (API)..."
Start-Process -FilePath "python" -ArgumentList "start_backend.py" -WindowStyle Normal

# Iniciar Frontend
Write-Host "Iniciando Frontend (UI)..."
Start-Process -FilePath "python" -ArgumentList "start_frontend.py" -WindowStyle Normal

# Esperar
Write-Host "Esperando 10 segundos..."
Start-Sleep -Seconds 10

# Abrir Dashboard
Write-Host "Abriendo Dashboard..."
Start-Process "http://localhost:8000"

Write-Host "SISTEMA INICIADO"
Write-Host "Backend: http://localhost:8001"
Write-Host "Dashboard: http://localhost:8000"
