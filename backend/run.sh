#!/bin/bash
# Violence Detection System - Startup Script

echo "🚨 Violence Detection Surveillance System"
echo "==========================================="
echo ""

# Check Python version
echo "Checking Python installation..."
python --version

# Create virtual environment if not exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
if [ -f "venv/Scripts/activate" ]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Train model if not exists
if [ ! -f "models/violence_detector.h5" ]; then
    echo ""
    echo "Training violence detection model..."
    python train_model.py
    echo ""
fi

# Start Flask server
echo ""
echo "🚀 Starting Violence Detection Dashboard..."
echo "📊 Dashboard: http://localhost:5000"
echo "🎥 Camera Feed: http://localhost:5000/video_feed"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

python app.py
