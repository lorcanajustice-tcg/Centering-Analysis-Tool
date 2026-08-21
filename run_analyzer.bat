@echo off
setlocal
title Card Centering Checker
cd /d "%~dp0"

rem ---- find Python 3 ----
set "PY=py -3"
%PY% -V >nul 2>nul
if errorlevel 1 set "PY=python"
%PY% -V >nul 2>nul
if errorlevel 1 (
  echo This needs Python 3, and it is not installed on this computer yet.
  echo.
  echo Get it from https://www.python.org/downloads/ - and when the installer
  echo asks, tick the box that says "Add python.exe to PATH". Then run this again.
  pause
  exit /b 1
)

rem ---- first-run dependency install ----
%PY% -c "import cv2, numpy, PIL, pillow_heif" >nul 2>nul
if errorlevel 1 (
  echo First time running this - setting a few things up. Takes 1 to 3 minutes.
  %PY% -m pip install --quiet opencv-python-headless numpy pillow pillow-heif
  if errorlevel 1 (
    echo.
    echo Setup could not finish. Check you are connected to the internet,
    echo then close this window and run it again.
    pause
    exit /b 1
  )
)

rem ---- launch ----
start "" http://127.0.0.1:8737/
echo Card Centering Checker is starting - your browser will open on its own.
echo Leave this window open while you use it. Close it when you are done.
%PY% webapp.py
pause
