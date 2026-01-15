# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies needed for OpenCV and other packages
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY backend/requirements.txt /app/backend/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r backend/requirements.txt

# Copy backend code
COPY backend/ /app/backend/

# NOTE: For production, datasets are stored in Google Cloud Storage
# Set these environment variables:
#   USE_GCS=true
#   GCS_BUCKET_NAME=your-bucket-name
#
# For local development with bundled datasets, uncomment the lines below:
# COPY sgsl_dataset/ /app/sgsl_dataset/
# COPY sgsl_processed/ /app/sgsl_processed/

# Expose port
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# GCS configuration (override these in Cloud Run or docker-compose)
ENV USE_GCS=true
ENV GCS_BUCKET_NAME=unmute-datasets

# Run the application
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
