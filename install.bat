@echo off
setlocal enabledelayedexpansion
title Whisper Transcriber - Setup

echo ============================================
echo  Whisper Transcriber - First Time Setup
echo ============================================
echo.

REM ── Find a Python interpreter ────────────────────────────────────────────
where py >nul 2>nul
if %ERRORLEVEL%==0 (
    set PYCMD=py -3.13
    py -3.13 --version >nul 2>nul
    if not !ERRORLEVEL!==0 (
        set PYCMD=py -3
    )
) else (
    where python >nul 2>nul
    if %ERRORLEVEL%==0 (
        set PYCMD=python
    ) else (
        echo [ERROR] Python was not found on your system.
        echo Install Python 3.11+ from https://www.python.org/downloads/
        echo IMPORTANT: check "Add python.exe to PATH" during install.
        pause
        exit /b 1
    )
)

echo Using: %PYCMD%
echo.

REM ── Create virtual environment ───────────────────────────────────────────
if not exist ".venv" (
    echo Creating virtual environment in .venv ...
    %PYCMD% -m venv .venv
    if not %ERRORLEVEL%==0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
) else (
    echo Virtual environment already exists, skipping creation.
)

call .venv\Scripts\activate.bat

echo.
echo Upgrading pip...
python -m pip install --upgrade pip >nul

echo.
echo Installing required packages (this can take a few minutes)...
pip install -r requirements.txt
if not %ERRORLEVEL%==0 (
    echo [ERROR] Package install failed. See the message above.
    pause
    exit /b 1
)

echo.
set /p GPU_ANS="Do you have an NVIDIA GPU and want faster CUDA transcription? (y/n): "
if /I "%GPU_ANS%"=="y" (
    echo Installing CUDA libraries...
    pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
)

echo.
set /p DIAR_ANS="Do you want speaker diarization support (who-said-what)? (y/n): "
if /I "%DIAR_ANS%"=="y" (
    echo Installing pyannote.audio + torch, this is a large download...
    pip install pyannote.audio torch
    echo.
    echo NOTE: Diarization also needs a free HuggingFace account.
    echo Accept the model license here: https://hf.co/pyannote/speaker-diarization-3.1
    echo Then paste your HuggingFace token into the app's "Diarize..." dialog.
)

echo.
where ffmpeg >nul 2>nul
if not %ERRORLEVEL%==0 (
    echo [NOTICE] ffmpeg was not found on your PATH.
    echo The app can usually still work using its bundled imageio-ffmpeg fallback.
    echo For best results, install ffmpeg separately with:
    echo     winget install Gyan.FFmpeg
)

echo.
echo ============================================
echo  Setup complete!
echo  Run the app with run.bat, or double-click
echo  transcriber_1.01.pyw after activating .venv
echo ============================================
pause
