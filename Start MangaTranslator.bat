@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" (
  echo .venv not found - see README.md
  pause
  exit /b 1
)
if not exist "models\comictextdetector.pt.onnx" (
  .venv\Scripts\python.exe tools\download_models.py
)
start "" ".venv\Scripts\pythonw.exe" -m manga_translator.app %*
