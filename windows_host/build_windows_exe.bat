@echo off
setlocal
cd /d "%~dp0\.."

echo Closing running LensDefectHost windows...
taskkill /F /IM LensDefectHost.exe >nul 2>nul
taskkill /F /IM LensDefectHost_fixed.exe >nul 2>nul

if not exist ".venv_windows_host\Scripts\python.exe" (
    python -m venv ".venv_windows_host"
)

".venv_windows_host\Scripts\python.exe" -m pip install --upgrade pip
".venv_windows_host\Scripts\python.exe" -m pip install -r "windows_host\requirements.txt"
".venv_windows_host\Scripts\python.exe" -m pip install -r "windows_host\requirements-build.txt"

".venv_windows_host\Scripts\pyinstaller.exe" ^
  --noconfirm ^
  --clean ^
  --onedir ^
  --windowed ^
  --paths "windows_host" ^
  --add-data "windows_host\stage2_anomaly.py;." ^
  --collect-all "cv2" ^
  --collect-all "numpy" ^
  --hidden-import "stage2_anomaly" ^
  --name "LensDefectHost" ^
  "windows_host\lens_defect_host.py"

if not exist "release" (
    mkdir "release"
)

if exist "dist\LensDefectHost\LensDefectHost.exe" (
    xcopy /E /I /Y "dist\LensDefectHost\*" "release\" >nul
) else (
    copy /Y "dist\LensDefectHost.exe" "release\LensDefectHost.exe" >nul
)
del /F /Q "dist\LensDefectHost.exe" >nul 2>nul
rmdir /S /Q "dist" >nul 2>nul
rmdir /S /Q "build" >nul 2>nul
del /F /Q "LensDefectHost.exe" >nul 2>nul
del /F /Q "LensDefectHost_fixed.exe" >nul 2>nul
del /F /Q "release\LensDefectHost_fixed.exe" >nul 2>nul

echo.
echo Built exe: release\LensDefectHost.exe
if not defined NO_PAUSE pause
