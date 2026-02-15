# Dockerfile for Live Translator Web Server
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    libgomp1 \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install build tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Set working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml ./
COPY src/ ./src/
COPY web/ ./web/

# Install Python dependencies directly (avoid editable install issues)
RUN pip install --no-cache-dir \
    numpy>=1.26 \
    faster-whisper>=1.1.0 \
    ctranslate2>=4.0 \
    sentencepiece>=0.2.0 \
    requests>=2.31 \
    fastapi>=0.104.0 \
    uvicorn[standard]>=0.24.0 \
    python-multipart>=0.0.6 \
    websockets>=12.0 \
    pydub>=0.25.0

# Expose port
EXPOSE 8000

# Run the web server
CMD ["python", "-m", "uvicorn", "src.web_server:app", "--host", "0.0.0.0", "--port", "8000"]
