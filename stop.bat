@echo off
rem Frees ports 8000 (FastAPI) and 7860 (Gradio) by killing whatever is
rem listening on them. Use this when you get "WinError 10048" (port in use).
setlocal enabledelayedexpansion
echo Stopping any running Railway Cascade processes ...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8000 ^| findstr LISTENING') do (
    echo   killing PID %%a (port 8000)
    taskkill /F /PID %%a >nul 2>nul
)
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :7860 ^| findstr LISTENING') do (
    echo   killing PID %%a (port 7860)
    taskkill /F /PID %%a >nul 2>nul
)
echo Done. You can now run:  python src\run.py
pause
