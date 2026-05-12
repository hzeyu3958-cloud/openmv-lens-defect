@echo off
setlocal
cd /d "%~dp0\.."

if not exist ".venv_windows_host\Scripts\python.exe" (
    python -m venv ".venv_windows_host"
)

".venv_windows_host\Scripts\python.exe" -m pip install --upgrade pip
".venv_windows_host\Scripts\python.exe" -m pip install -r "windows_host\requirements.txt"
".venv_windows_host\Scripts\python.exe" "windows_host\lens_defect_host.py"
pause
