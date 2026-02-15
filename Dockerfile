# Dockerfile for Live Translator Web Server
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml ./
COPY src/ ./src/
COPY web/ ./web/

# Install Python dependencies
RUN pip install --no-cache-dir -e .

# Download Whisper model (optional - can be downloaded at runtime)
# RUN python -c "from faster_whisper import WhisperModel; WhisperModel('small', device='cpu', compute_type='int8')"

# Expose port
EXPOSE 8000

# Run the web server
CMD ["python", "-m", "uvicorn", "src.web_server:app", "--host", "0.0.0.0", "--port", "8000"]
