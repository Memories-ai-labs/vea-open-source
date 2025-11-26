#!/bin/bash
# VEA Playground Setup Script
# This script sets up the development environment for VEA Playground

set -e  # Exit on error

echo "================================================"
echo "VEA Playground Setup"
echo "================================================"

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
REQUIRED_VERSION="3.11"

echo "[INFO] Detected Python version: $PYTHON_VERSION"

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "[INFO] Creating virtual environment..."
    python3 -m venv .venv
    echo "[SUCCESS] Virtual environment created."
else
    echo "[INFO] Virtual environment already exists."
fi

# Activate virtual environment
echo "[INFO] Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo "[INFO] Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "[INFO] Installing dependencies..."
pip install -r requirements.txt

# Setup ViNet for dynamic cropping
echo "[INFO] Setting up ViNet..."
python -m lib.utils.vinet_setup

# Create local data directories
echo "[INFO] Creating local data directories..."
mkdir -p data/videos
mkdir -p data/indexing
mkdir -p data/outputs
mkdir -p .cache

# Create config.json from example if it doesn't exist
if [ ! -f "config.json" ]; then
    if [ -f "config.example.json" ]; then
        echo "[INFO] Creating config.json from config.example.json..."
        cp config.example.json config.json
        echo "[IMPORTANT] Please edit config.json and add your API keys!"
    else
        echo "[WARNING] config.example.json not found. Please create config.json manually."
    fi
else
    echo "[INFO] config.json already exists."
fi

# Check for ffmpeg
if command -v ffmpeg &> /dev/null; then
    echo "[SUCCESS] ffmpeg is installed."
else
    echo "[WARNING] ffmpeg is not installed. Please install ffmpeg for video processing."
    echo "  Ubuntu/Debian: sudo apt install ffmpeg"
    echo "  macOS: brew install ffmpeg"
fi

echo ""
echo "================================================"
echo "Setup Complete!"
echo "================================================"
echo ""
echo "Next steps:"
echo "1. Edit config.json and add your API keys"
echo "2. Place your video files in data/videos/"
echo "3. Activate the virtual environment: source .venv/bin/activate"
echo "4. Run the server: python -m src.app"
echo ""
echo "For Google Cloud authentication (required for Gemini):"
echo "  gcloud auth application-default login"
echo ""
