#!/bin/bash

# CyberGuard Predictor - Start Script for Cloud Deployment

echo "🚀 Starting CyberGuard Predictor..."

# Set environment variables if not set
export PORT=${PORT:-8000}
export HOST=${HOST:-0.0.0.0}

# Create required directories
mkdir -p logs prediction_output final_models

# Check if models exist
if [ ! -f "final_models/model.pkl" ]; then
    echo "⚠️ Warning: Model files not found. Some features may not work."
    echo "ℹ️ This is normal for first deployment - models will be created during training."
fi

# Start the application
echo "🌐 Starting server on $HOST:$PORT"
python app.py