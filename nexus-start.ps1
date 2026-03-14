<# 
  NEXUS-ATMS — One-Command Startup Script
  ==========================================
  Launches the entire NEXUS-ATMS stack:
    1. FastAPI Backend  (port 8000)
    2. Digital Twin     (Pygame window)
    3. Opens Dashboard  (browser)
  
  Usage:
    .\nexus-start.ps1              # Demo mode (no SUMO needed)
    .\nexus-start.ps1 -WithSumo    # Also launches SUMO simulation
#>

param(
    [switch]$WithSumo
)

$ErrorActionPreference = "Continue"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host ""
Write-Host "  =============================================" -ForegroundColor Cyan
Write-Host "    NEXUS-ATMS  —  Urban Intelligence Platform  " -ForegroundColor Cyan
Write-Host "  =============================================" -ForegroundColor Cyan
Write-Host ""

# -----------------------------------------------------------
# Activate venv
# -----------------------------------------------------------
$VenvActivate = Join-Path $ProjectRoot ".venv\Scripts\Activate.ps1"
if (Test-Path $VenvActivate) {
    Write-Host "[*] Activating virtual environment..." -ForegroundColor Yellow
    & $VenvActivate
} else {
    Write-Host "[!] No .venv found at $VenvActivate — using system Python" -ForegroundColor Red
}

# -----------------------------------------------------------
# Set environment
# -----------------------------------------------------------
$env:DEMO_MODE = "true"
$env:PYTHONPATH = $ProjectRoot

# -----------------------------------------------------------
# SUMO (optional)
# -----------------------------------------------------------
if ($WithSumo) {
    $SumoGui = "C:\Program Files (x86)\Eclipse\Sumo\bin\sumo-gui.exe"
    $SumoCfg = Join-Path $ProjectRoot "networks\grid_4x4\grid_4x4.sumocfg"
    
    if ((Test-Path $SumoGui) -and (Test-Path $SumoCfg)) {
        Write-Host "[1/3] Starting SUMO simulation..." -ForegroundColor Green
        Start-Process -FilePath $SumoGui -ArgumentList "-c", "`"$SumoCfg`""
    } else {
        Write-Host "[!] SUMO not found or config missing — skipping" -ForegroundColor Red
    }
}

# -----------------------------------------------------------
# FastAPI Backend
# -----------------------------------------------------------
Write-Host "[1/3] Starting FastAPI Backend on http://localhost:8000 ..." -ForegroundColor Green
$BackendScript = Join-Path $ProjectRoot "dashboard\backend\main.py"
$BackendProcess = Start-Process -FilePath "python" `
    -ArgumentList "`"$BackendScript`"" `
    -WorkingDirectory $ProjectRoot `
    -PassThru -WindowStyle Normal

Start-Sleep -Seconds 2

# -----------------------------------------------------------
# Digital Twin (Pygame)
# -----------------------------------------------------------
Write-Host "[2/3] Starting Digital Twin renderer..." -ForegroundColor Green
$TwinRunner = Join-Path $ProjectRoot "run_digital_twin.py"
if (Test-Path $TwinRunner) {
    $TwinProcess = Start-Process -FilePath "python" `
        -ArgumentList "`"$TwinRunner`"" `
        -WorkingDirectory $ProjectRoot `
        -PassThru -WindowStyle Normal
}

Start-Sleep -Seconds 1

# -----------------------------------------------------------
# Open Dashboard in Browser
# -----------------------------------------------------------
Write-Host "[3/3] Opening Dashboard in browser..." -ForegroundColor Green
Start-Process "http://localhost:8000"

# -----------------------------------------------------------
# Status
# -----------------------------------------------------------
Write-Host ""
Write-Host "  NEXUS-ATMS is running!" -ForegroundColor Green
Write-Host "  ----------------------------------------" -ForegroundColor DarkGray
Write-Host "    Dashboard : http://localhost:8000" -ForegroundColor White
Write-Host "    API Docs  : http://localhost:8000/docs" -ForegroundColor White
Write-Host "    WebSocket : ws://localhost:8000/ws/live" -ForegroundColor White
Write-Host "  ----------------------------------------" -ForegroundColor DarkGray
Write-Host ""
Write-Host "  Press Ctrl+C to stop all services" -ForegroundColor Yellow
Write-Host ""

# -----------------------------------------------------------
# Wait for exit
# -----------------------------------------------------------
try {
    if ($BackendProcess) {
        $BackendProcess.WaitForExit()
    }
} finally {
    Write-Host "`n[*] Shutting down NEXUS-ATMS..." -ForegroundColor Yellow
    if ($BackendProcess -and !$BackendProcess.HasExited) { $BackendProcess.Kill() }
    if ($TwinProcess -and !$TwinProcess.HasExited) { $TwinProcess.Kill() }
    Write-Host "[*] All services stopped." -ForegroundColor Green
}
