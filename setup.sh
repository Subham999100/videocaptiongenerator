#!/bin/bash
# Quick Start Script for Video Caption Generator

echo "=========================================="
echo "Video Caption Generator - Quick Start"
echo "=========================================="

# Check Python installation
echo ""
echo "Checking Python installation..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    echo "✓ Found: $PYTHON_VERSION"
else
    echo "✗ Python 3 not found. Please install Python 3.7+"
    exit 1
fi

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo ""
echo "Creating directory structure..."
mkdir -p data/videos
mkdir -p data/captions
mkdir -p data/features
mkdir -p saved_models

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next Steps:"
echo "1. Add video files to: data/videos/"
echo "2. Add caption files to: data/captions/"
echo "3. Run: python feature_extraction.py"
echo "4. Run: python train.py"
echo "5. Run: python inference.py --video_path <path_to_video>"
echo ""
echo "For detailed instructions, see README.md and TUTORIAL.md"
echo "=========================================="
