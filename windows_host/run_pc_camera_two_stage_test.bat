@echo off
setlocal
cd /d "%~dp0\.."

if not exist ".venv_windows_host\Scripts\python.exe" (
    python -m venv ".venv_windows_host"
)

".venv_windows_host\Scripts\python.exe" -m pip install -r "windows_host\requirements-camera.txt"

if not exist "models\lens_stage2_anomaly.npz" (
    echo.
    echo 二级异常检测模型不存在：models\lens_stage2_anomaly.npz
    echo 请先采集正常镜片图片，然后运行 training\train_stage2_anomaly.bat
    echo.
    pause
    exit /b 1
)

".venv_windows_host\Scripts\python.exe" "windows_host\pc_camera_rule_test.py" --stage2-model "models\lens_stage2_anomaly.npz"

pause
