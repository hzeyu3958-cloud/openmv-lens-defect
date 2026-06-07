@echo off
setlocal
cd /d "%~dp0\.."

set "PORT=%~1"
if "%PORT%"=="" set "PORT=COM8"

set "COUNT=%~2"
if "%COUNT%"=="" set "COUNT=60"

set "SPLIT=%~3"
if "%SPLIT%"=="" set "SPLIT=train"

set "DELAY=%~4"
if "%DELAY%"=="" set "DELAY=0.60"

if not exist ".venv_windows_host\Scripts\python.exe" (
    python -m venv ".venv_windows_host"
)

".venv_windows_host\Scripts\python.exe" ^
  "windows_host\auto_capture_openmv_dataset.py" ^
  --port "%PORT%" ^
  --baudrate 115200 ^
  --script "openmv\n6_usb_image_capture.py" ^
  --dataset "dataset" ^
  --split "%SPLIT%" ^
  --label "scratch" ^
  --count "%COUNT%" ^
  --delay "%DELAY%"

if errorlevel 1 exit /b %errorlevel%
echo.
echo Saved thin-scratch samples to dataset\%SPLIT%\scratch
