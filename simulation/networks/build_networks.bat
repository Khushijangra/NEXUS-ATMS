@echo off
REM ===================================================
REM  Build SUMO network files using netconvert
REM  Run this from the project root directory
REM ===================================================

echo ====================================================
echo  Building SUMO Network Files
echo ====================================================

REM Check SUMO_HOME
if "%SUMO_HOME%"=="" (
    echo [ERROR] SUMO_HOME is not set.
    echo   Set it to your SUMO installation directory, e.g.:
    echo   set SUMO_HOME=C:\Program Files ^(x86^)\Eclipse\Sumo
    exit /b 1
)

echo [OK] SUMO_HOME = %SUMO_HOME%

REM --- Build Single Intersection Network ---
echo.
echo [1/2] Building single_intersection.net.xml ...
netconvert ^
    --node-files=networks/single_intersection.nod.xml ^
    --edge-files=networks/single_intersection.edg.xml ^
    --tllogic-files=networks/single_intersection.tll.xml ^
    --output-file=networks/single_intersection.net.xml ^
    --no-turnarounds true

if errorlevel 1 (
    echo [FAIL] Single intersection network build failed.
    exit /b 1
)
echo [OK] networks/single_intersection.net.xml created successfully.

REM --- Build 2x2 Grid Network ---
echo.
echo [2/2] Building grid_2x2.net.xml ...
netgenerate ^
    --grid ^
    --grid.number=2 ^
    --grid.length=300 ^
    --default.lanenumber=2 ^
    --default.speed=13.89 ^
    --tls.guess=true ^
    --output-file=networks/grid_2x2.net.xml ^
    --no-turnarounds true

if errorlevel 1 (
    echo [FAIL] Grid network build failed.
    exit /b 1
)
echo [OK] networks/grid_2x2.net.xml created successfully.

echo.
echo ====================================================
echo  All networks built successfully!
echo ====================================================
echo.
echo  Verify visually:
echo    sumo-gui -n networks/single_intersection.net.xml
echo    sumo-gui -n networks/grid_2x2.net.xml
echo.
