# syntax=docker/dockerfile:1

FROM python:3.12-slim-bookworm

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    tesseract-ocr \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Copy Python dependency file and install system dependencies
COPY pyproject.toml .
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    tesseract-ocr \
    poppler-utils \
    --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Download and run the official uv installer
ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Ensure uv and installed tools are on PATH
ENV PATH="/root/.local/bin/:$PATH"

# Install Python dependencies
RUN uv sync

# Copy source code and entrypoint script
COPY . .

# Expose ports for FastAPI and Streamlit
EXPOSE 8000 8501

# Set entrypoint
ENTRYPOINT ["/bin/bash", "/entrypoint.sh"]

RUN chmod +x /entrypoint.sh

# Use entrypoint for configurable pipeline execution
ENTRYPOINT ["/entrypoint.sh"]
