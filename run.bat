@echo off
if not exist ".venv\Scripts\pythonw.exe" (
    echo Virtual environment not found. Run install.bat first.
    pause
    exit /b 1
)
start "" ".venv\Scripts\pythonw.exe" "transcriber_1.01.pyw"
