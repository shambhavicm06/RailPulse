@echo off
setlocal
cd /d "%~dp0"

rem ---- pick the right Python: prefer the project venv from setup.bat ----
set "PY=python"
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"

"%PY%" -c "import sys" >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python was not found.
    echo Install Python 3.10-3.13 from https://www.python.org/downloads/
    echo and TICK "Add Python to PATH" during installation, then re-open cmd.
    pause
    exit /b 1
)

rem ---- train from scratch only if the trained model bundle is missing ----
if not exist "models\bundle.joblib" (
    echo.
    echo No trained model found - running the full pipeline first ...
    echo (1/3) Generating synthetic SWR data
    "%PY%" src\data_generator.py
    if errorlevel 1 goto :err
    echo (2/3) Extracting graph features
    "%PY%" src\feature_engineering.py
    if errorlevel 1 goto :err
    echo (3/3) Training the 10 models
    "%PY%" src\train.py
    if errorlevel 1 goto :err
)

echo.
echo Starting FastAPI backend (port 8000) + Gradio dashboard (port 7860) ...
echo.
echo   RailPulse web app  (login + dark/light + network map):  http://127.0.0.1:8000
echo   Gradio dispatcher console:                              http://127.0.0.1:7860
echo.
echo   Login ->  username: admin    password: swr2026
echo   Press Ctrl+C to stop (or run stop.bat if the port gets stuck).
echo.
"%PY%" src\run.py
goto :eof

:err
echo.
echo [ERROR] A step failed. See the message above.
pause
exit /b 1
