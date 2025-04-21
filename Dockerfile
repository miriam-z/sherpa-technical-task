# syntax=docker/dockerfile:1

FROM python:3.12-slim-bookworm

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Install system dependencies (do this ONCE)
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

# Copy Python dependency file
COPY pyproject.toml .

# Download and run the official uv installer
ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Ensure uv and installed tools are on PATH
ENV PATH="/root/.local/bin/:$PATH"

# Install Python dependencies
RUN uv sync

# Copy source code and entrypoint script
COPY . .

# Expose ports for FastAPI, Streamlit, and TruLens dashboard
EXPOSE 8000 8501 8503

RUN chmod +x /app/entrypoint.sh

# Set entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]
