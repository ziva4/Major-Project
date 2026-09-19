@echo off
REM Violence Detection System - Startup Script for Windows

echo 🚨 Violence Detection Surveillance System
echo ===========================================
echo.

REM Check Python version
echo Checking Python installation...
python --version

REM Create or repair virtual environment if needed
if not exist "venv\pyvenv.cfg" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install dependencies
echo Installing dependencies...
pip install -r requirements.txt

REM Start Flask server
echo.
echo Starting Person Detection Dashboard...
echo The pretrained YOLO model downloads once on first startup.
echo 📊 Dashboard: http://localhost:5000
echo 🎥 Camera Feed: http://localhost:5000/video_feed
echo.
echo Press Ctrl+C to stop the server
echo.

python app.py
pause
