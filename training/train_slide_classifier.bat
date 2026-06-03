@echo off
setlocal
cd /d "%~dp0\.."

if not exist ".venv_training\Scripts\python.exe" (
    python -m venv ".venv_training"
)

".venv_training\Scripts\python.exe" -m pip install -r "training\requirements-training.txt"
".venv_training\Scripts\python.exe" "training\train_lens_classifier.py" ^
  --dataset "dataset_slide" ^
  --output "models" ^
  --artifact-prefix "slide_defect" ^
  --summary-name "slide_training_summary.json" ^
  --epochs 20 ^
  --batch-size 16 ^
  --image-size 128

pause
