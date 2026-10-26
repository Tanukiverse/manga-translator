@echo off
rem One-time setup on a new Windows PC: Python venv, PyTorch (CUDA if an NVIDIA GPU is present,
rem otherwise CPU), libraries, AI models (~750 MB) and, optionally, the Ollama translation model.
setlocal
cd /d "%~dp0"

rem ---- Python 3.12
set PY=
py -3.12 -c "import sys" >nul 2>&1 && set PY=py -3.12
if not defined PY (
  python -c "import sys; raise SystemExit(0 if sys.version_info[:2]==(3,12) else 1)" >nul 2>&1 && set PY=python
)
if not defined PY (
  echo Python 3.12 was not found. Install it from https://www.python.org/downloads/ and run setup.bat again.
  pause
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo Creating .venv ...
  %PY% -m venv .venv || goto :fail
)
set VPY=.venv\Scripts\python.exe
%VPY% -m pip install --upgrade pip || goto :fail

rem ---- PyTorch: CUDA 12.8 build for NVIDIA GPUs (RTX 50xx needs cu128), CPU build otherwise
where nvidia-smi >nul 2>&1
if %errorlevel%==0 (
  echo NVIDIA GPU found: installing PyTorch with CUDA 12.8 ...
  %VPY% -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128 || goto :fail
) else (
  echo No NVIDIA GPU found: installing the CPU build of PyTorch (works, but removing text is slower^) ...
  %VPY% -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu || goto :fail
)

echo Installing libraries ...
%VPY% -m pip install -r requirements.txt || goto :fail
rem rapidocr pulls in opencv-python (with its own Qt), which clashes with PySide6: keep only the headless build
%VPY% -m pip uninstall -y opencv-python opencv-python-headless >nul 2>&1
%VPY% -m pip install opencv-python-headless==5.0.0.93 || goto :fail

echo Downloading AI models into models\ ...
%VPY% tools\download_models.py || goto :fail

rem ---- Ollama (translator)
where ollama >nul 2>&1
if %errorlevel% neq 0 (
  echo.
  echo Ollama is not installed. Install it from https://ollama.com/download , then run:
  echo   ollama pull hf.co/HauhauCS/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive:Q4_K_M
  goto :done
)
set /p PULL=Download the default translation model now (about 6.3 GB)? [y/N]:
if /i "%PULL%"=="y" ollama pull hf.co/HauhauCS/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive:Q4_K_M

:done
echo.
echo Setup finished. Start the program with "Start MangaTranslator.bat".
pause
exit /b 0

:fail
echo.
echo Setup failed (see the messages above).
pause
exit /b 1
