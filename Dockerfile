FROM python:3.9-slim

# Install system libraries required by OpenCV on headless Linux systems
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files into the container
COPY . .

# Expose the default Hugging Face Spaces port
EXPOSE 7860

# Run the Flask application
CMD ["python", "app.py"]
