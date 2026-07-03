@echo off
setlocal
title Centering Analyzer
cd /d "%~dp0"

rem ---- find Python 3 ----
set "PY=py -3"
%PY% -V >nul 2>nul
if errorlevel 1 set "PY=python"
%PY% -V >nul 2>nul
if errorlevel 1 (
  echo Python 3 was not found on this computer.
  echo Install it from https://www.python.org/downloads/ ^(tick "Add python.exe to PATH"^), then run this again.
  pause
  exit /b 1
)

rem ---- first-run dependency install ----
%PY% -c "import cv2, numpy, PIL, pillow_heif" >nul 2>nul
if errorlevel 1 (
  echo First run: installing dependencies ^(1-3 minutes^)...
  %PY% -m pip install --quiet opencv-python-headless numpy pillow pillow-heif
  if errorlevel 1 (
    echo.
    echo Dependency install failed. Check your internet connection and retry.
    pause
    exit /b 1
  )
)

rem ---- launch ----
start "" http://127.0.0.1:8737/
echo Centering Analyzer is starting - your browser will open.
echo Keep this window open while you use it. Close it to stop.
%PY% webapp.py
pause
