# Use a lightweight Python base image
FROM python:3.10-slim

# Environment settings
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set working directory
WORKDIR /app

# Install system dependencies for PDF processing and NLP
RUN apt-get update && apt-get install -y \
    build-essential \
    libpoppler-cpp-dev \
    poppler-utils \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file separately to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data offline if needed
RUN python -m nltk.downloader punkt stopwords

# Copy application code
COPY . .

# Optional input directory
RUN mkdir -p inputs

# Default command to run the application
CMD ["python", "main.py", "--input_json", "input.json", "--output", "output.json"]
