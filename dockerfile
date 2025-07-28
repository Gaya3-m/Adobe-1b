# Use a lightweight Python base image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set working directory
WORKDIR /app

# Install system dependencies required for poppler and basic NLP
RUN apt-get update && apt-get install -y \
    build-essential \
    libpoppler-cpp-dev \
    poppler-utils \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt before copying full app for better cache usage
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data offline if needed
# (Optional: you can include `punkt`, `stopwords` if you use them in extract_keyphrases)
RUN python -m nltk.downloader punkt stopwords

# Copy the application code
COPY . .

# Make input directory (optional)
RUN mkdir -p inputs

# Run the main app (you can override this in docker run if needed)
CMD ["python", "main.py", "--input_json", "input.json", "--output", "output.json"]
