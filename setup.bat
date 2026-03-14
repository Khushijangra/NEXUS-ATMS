@echo off
REM ===================================================
REM  Smart Traffic Management System - Setup Script
REM ===================================================

echo ====================================================
echo  Smart Traffic Management System - Setup
echo ====================================================

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Install Python 3.9+
    exit /b 1
)

REM Check SUMO
if "%SUMO_HOME%"=="" (
    echo [WARN] SUMO_HOME not set. Dashboard demo mode will work but training requires SUMO.
    echo        Install SUMO: https://sumo.dlr.de/docs/Installing/index.html
    echo        Then set: set SUMO_HOME=C:\path\to\sumo
) else (
    echo [OK] SUMO_HOME = %SUMO_HOME%
)

REM Create virtual environment
echo.
echo Creating virtual environment...
python -m venv venv
call venv\Scripts\activate.bat

REM Install dependencies
echo.
echo Installing dependencies...
pip install --upgrade pip
pip install -r requirements.txt

REM Create directories
mkdir logs 2>nul
mkdir models 2>nul
mkdir results 2>nul

echo.
echo ====================================================
echo  Setup Complete!
echo ====================================================
echo.
echo  Quick Start:
echo    1. Activate venv:  venv\Scripts\activate
echo    2. Run demo:       python run_demo.py --dashboard-only
echo    3. Full pipeline:  python run_demo.py
echo.
