@echo off
setlocal
cd /d "%~dp0\.."

if not exist ".venv_windows_host\Scripts\python.exe" (
    python -m venv ".venv_windows_host"
)

".venv_windows_host\Scripts\python.exe" -m pip install --upgrade pip
".venv_windows_host\Scripts\python.exe" -m pip install -r "windows_host\requirements.txt"
".venv_windows_host\Scripts\python.exe" -m pip install -r "windows_host\requirements-build.txt"

".venv_windows_host\Scripts\pyinstaller.exe" ^
  --noconfirm ^
  --clean ^
  --onefile ^
  --windowed ^
  --name "LensDefectHost" ^
  "windows_host\lens_defect_host.py"

echo.
echo Built exe: dist\LensDefectHost.exe
pause
