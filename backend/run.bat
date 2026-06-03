@echo off
REM Violence Detection System - Startup Script for Windows

echo 🚨 Violence Detection Surveillance System
echo ===========================================
echo.

REM Check Python version
echo Checking Python installation...
python --version

REM Create virtual environment if not exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install dependencies
echo Installing dependencies...
pip install -r requirements.txt

REM Train model if not exists
if not exist "models\violence_detector.h5" (
    echo.
    echo Training violence detection model...
    python train_model.py
    echo.
)

REM Start Flask server
echo.
echo 🚀 Starting Violence Detection Dashboard...
echo 📊 Dashboard: http://localhost:5000
echo 🎥 Camera Feed: http://localhost:5000/video_feed
echo.
echo Press Ctrl+C to stop the server
echo.

python app.py
pause
