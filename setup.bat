@echo off
setlocal
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python was not found on PATH.
    echo Install Python 3.10-3.13 from https://www.python.org/downloads/
    echo and TICK "Add Python to PATH" during installation, then re-open cmd.
    pause
    exit /b 1
)

echo Creating virtual environment (.venv) ...
python -m venv .venv
call .venv\Scripts\activate.bat

echo Upgrading pip ...
python -m pip install --upgrade pip

echo Installing dependencies (this can take a few minutes) ...
pip install -r requirements.txt

echo.
echo Setup complete! Now double-click start.bat to launch the app.
pause
