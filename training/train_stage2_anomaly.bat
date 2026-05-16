@echo off
setlocal
cd /d "%~dp0\.."

if not exist ".venv_windows_host\Scripts\python.exe" (
    python -m venv ".venv_windows_host"
)

".venv_windows_host\Scripts\python.exe" -m pip install -r "windows_host\requirements-camera.txt"
".venv_windows_host\Scripts\python.exe" "training\train_stage2_anomaly.py" --dataset "dataset" --output "models\lens_stage2_anomaly.npz"

pause
