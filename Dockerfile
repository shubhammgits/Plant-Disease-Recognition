# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# Use --no-cache-dir to reduce image size
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your project files into the container at /app
COPY . .

# Tell the container to run your app using gunicorn, a production-ready server
# It listens on port 7860, which is what Hugging Face expects
CMD gunicorn --bind 0.0.0.0:$PORT app:app

