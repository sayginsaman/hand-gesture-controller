#!/bin/bash

echo "Hand Gesture Controller Setup Script"
echo "===================================="

# Check if Python 3.12 is available
if command -v python3.12 &> /dev/null; then
    echo "✓ Python 3.12 found"
    PYTHON_CMD="python3.12"
elif command -v python3.11 &> /dev/null; then
    echo "✓ Python 3.11 found"
    PYTHON_CMD="python3.11"
elif command -v python3.10 &> /dev/null; then
    echo "✓ Python 3.10 found"
    PYTHON_CMD="python3.10"
else
    echo "❌ Error: Python 3.10, 3.11, or 3.12 is required"
    echo "MediaPipe doesn't support Python 3.13 yet"
    echo "Please install Python 3.12:"
    echo "  brew install python@3.12"
    exit 1
fi

# Check if virtual environment exists
if [ -d "venv" ]; then
    echo "⚠️  Virtual environment already exists. Removing..."
    rm -rf venv
fi

# Create virtual environment
echo "Creating virtual environment with $PYTHON_CMD..."
$PYTHON_CMD -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "✅ Setup complete!"
echo ""
echo "To run the application:"
echo "1. Activate the virtual environment: source venv/bin/activate"
echo "2. Run the script: python hand_gesture_controller.py"
echo ""
echo "⚠️  Important: Grant camera permissions when prompted!"
echo "   Go to System Preferences > Security & Privacy > Privacy > Camera"
echo "   Add Terminal to the allowed applications" 