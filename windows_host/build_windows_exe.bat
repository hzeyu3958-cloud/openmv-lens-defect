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
  --onefile ^
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

for /f "delims=" %%I in ('dir /b "release" 2^>nul') do (
    if /I not "%%I"=="LensDefectHost.exe" (
        if exist "release\%%I\*" (
            rmdir /S /Q "release\%%I"
        ) else (
            del /F /Q "release\%%I" >nul 2>nul
        )
    )
)

for /L %%R in (1,1,8) do (
    del /F /Q "release\LensDefectHost.exe" >nul 2>nul
    copy /Y "dist\LensDefectHost.exe" "release\LensDefectHost.exe" >nul 2>nul
    if exist "release\LensDefectHost.exe" goto copied_release_exe
    timeout /T 2 /NOBREAK >nul
)
echo Failed to overwrite release\LensDefectHost.exe
exit /b 1
:copied_release_exe
rmdir /S /Q "dist" >nul 2>nul
rmdir /S /Q "build" >nul 2>nul
del /F /Q "LensDefectHost.exe" >nul 2>nul
del /F /Q "LensDefectHost_fixed.exe" >nul 2>nul
del /F /Q "release\LensDefectHost_fixed.exe" >nul 2>nul

echo.
echo Built exe: release\LensDefectHost.exe
if not defined NO_PAUSE pause
