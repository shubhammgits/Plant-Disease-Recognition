#!/bin/bash

# Startup script for Plant Disease Recognition System

echo "Starting Plant Disease Recognition System..."

# Create necessary directories
mkdir -p uploadimages
mkdir -p static/images

# Set permissions
chmod 755 uploadimages
chmod 755 static

echo "Environment setup complete!"

# Start the application
if [ "$1" = "production" ]; then
    echo "Starting in production mode..."
    gunicorn --bind 0.0.0.0:${PORT:-5000} --workers 1 --timeout 120 app:app
else
    echo "Starting in development mode..."
    python app.py
fi