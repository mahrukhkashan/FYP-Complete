FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for Tesseract OCR
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    libtesseract-dev \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# FIXED: Download spaCy model directly via pip (bypasses the broken spacy download command)
RUN pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.6.0/en_core_web_sm-3.6.0-py3-none-any.whl

# Copy the rest of your application
COPY . .

# Use Railway's dynamic PORT
ARG PORT=5000
ENV PORT=${PORT}
EXPOSE ${PORT}

# Start the application
CMD ["sh", "-c", "gunicorn run:app --bind 0.0.0.0:${PORT}"]