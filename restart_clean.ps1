# Script de Reinicio Limpio - EL-AMANECERV3
# Detiene todas las instancias y reinicia una limpia

Write-Host 'Reinicio Limpio del Sistema EL-AMANECERV3' -ForegroundColor Cyan
Write-Host ''

# 1. Detener todas las instancias de start_system.py
Write-Host 'Deteniendo instancias antiguas...' -ForegroundColor Yellow
Get-Process python -ErrorAction SilentlyContinue | Where-Object {
    $_.CommandLine -like '*start_system.py*'
} | ForEach-Object {
    Write-Host "   Deteniendo PID: $($_.Id)" -ForegroundColor Gray
    Stop-Process -Id $_.Id -Force
}

Start-Sleep -Seconds 2

# 2. Verificar que no queden procesos
$remaining = Get-Process python -ErrorAction SilentlyContinue | Where-Object {
    $_.CommandLine -like '*start_system.py*'
}

if ($remaining) {
    Write-Host 'ADVERTENCIA: Algunos procesos no se pudieron detener' -ForegroundColor Red
    Write-Host '   Por favor cierra manualmente las terminales con start_system.py' -ForegroundColor Red
    exit 1
}

Write-Host 'Todas las instancias detenidas' -ForegroundColor Green
Write-Host ''

# 3. Esperar un momento para liberar recursos
Write-Host 'Esperando 3 segundos para liberar recursos...' -ForegroundColor Yellow
Start-Sleep -Seconds 3

# 4. Lanzar nueva instancia limpia
Write-Host 'Iniciando nueva instancia...' -ForegroundColor Cyan
Start-Process -FilePath 'python' -ArgumentList 'start_system.py' -NoNewWindow

Start-Sleep -Seconds 2

# 5. Verificar que arrancó
$newProcess = Get-Process python -ErrorAction SilentlyContinue | Where-Object {
    $_.CommandLine -like '*start_system.py*'
} | Select-Object -First 1

if ($newProcess) {
    Write-Host 'Sistema reiniciado exitosamente!' -ForegroundColor Green
    Write-Host "   PID: $($newProcess.Id)" -ForegroundColor Gray
    Write-Host ''
    Write-Host 'Dashboard disponible en: http://localhost:8000' -ForegroundColor Cyan
    Write-Host 'Pestana Consciencia activa con datos en tiempo real' -ForegroundColor Cyan
}
else {
    Write-Host 'ERROR: No se pudo iniciar el sistema' -ForegroundColor Red
    Write-Host '   Ejecuta manualmente: python start_system.py' -ForegroundColor Yellow
}
