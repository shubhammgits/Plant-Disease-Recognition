# Dockerfile for Plant Disease Recognition System
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire application
COPY . .

# Create uploads directory
RUN mkdir -p uploadimages

# Expose port
EXPOSE $PORT

# Command to run the application
CMD gunicorn --bind 0.0.0.0:$PORT app:app