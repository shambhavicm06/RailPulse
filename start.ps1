# PowerShell launcher for the RailPulse Railway Cascade Predictor.
# If you get an execution-policy error, run once as Administrator:
#   Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# Pick the right Python: prefer the project venv from setup.bat
$py = "python"
if (Test-Path ".venv\Scripts\python.exe") { $py = ".venv\Scripts\python.exe" }

if (-not (Get-Command $py -ErrorAction SilentlyContinue)) {
    Write-Host "[ERROR] Python not found. Install Python 3.10-3.13 and tick 'Add Python to PATH'." -ForegroundColor Red
    exit 1
}

if (-not (Test-Path "models\bundle.joblib")) {
    Write-Host "No trained model found - running the full pipeline first ..."
    & $py src\data_generator.py
    & $py src\feature_engineering.py
    & $py src\train.py
}

Write-Host "Starting FastAPI backend (8000) + Gradio dashboard (7860) ..."
Write-Host "  RailPulse web app  (login + dark/light + network map): http://127.0.0.1:8000"
Write-Host "  Gradio dispatcher console:                             http://127.0.0.1:7860"
Write-Host "  Login -> username: admin   password: swr2026"
& $py src\run.py
