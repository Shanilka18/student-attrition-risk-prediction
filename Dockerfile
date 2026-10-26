# ──────────────────────────────────────────────────────────────────
# Psychosocial Intelligence Framework — Dockerfile
# Python 3.11-slim · FastAPI backend only
# ──────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# Metadata
LABEL maintainer="SLIIT Kandy Research Team"
LABEL description="Psychosocial Risk Index (PRI) — Research API"
LABEL version="1.0.0"

# Research disclaimer embedded in image metadata
ENV ETHICS_DISCLAIMER="RESEARCH NOTICE: The PRI is an academic decision-support indicator for educational research at SLIIT Kandy. It is NOT a clinical psychological evaluation or psychiatric diagnosis."

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Working directory
WORKDIR /app

# Install Python dependencies first (layer-cached)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy project source
COPY src/ src/
COPY config/ config/
COPY artifacts/ artifacts/
COPY data/processed/ data/processed/
COPY data/synthetic/ data/synthetic/

# Create writable directories for uploads and job output
RUN mkdir -p /app/uploads /app/logs

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl --fail http://localhost:8000/health || exit 1

# Non-root user for security
RUN useradd -m -u 1000 priuser && chown -R priuser:priuser /app
USER priuser

# Start FastAPI server
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
