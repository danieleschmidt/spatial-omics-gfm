# Multi-stage Production Dockerfile for Spatial-Omics GFM
# Optimized for production deployment with security and performance

# Build stage
FROM python:3.11-slim as builder

# Set build arguments
ARG BUILD_ENV=production
ARG VERSION=latest

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements-prod.txt requirements.txt
COPY pyproject.toml .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY spatial_omics_gfm/ ./spatial_omics_gfm/
COPY setup.py .

# Install the package
RUN pip install --no-cache-dir .

# Production stage
FROM python:3.11-slim as production

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV ENVIRONMENT=production

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set working directory
WORKDIR /app

# Copy Python environment from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application files
COPY --from=builder /app/spatial_omics_gfm ./spatial_omics_gfm
COPY docker/entrypoint.sh ./entrypoint.sh
COPY docker/healthcheck.py ./healthcheck.py

# Make scripts executable
RUN chmod +x entrypoint.sh healthcheck.py

# Create necessary directories
RUN mkdir -p /app/logs /app/data /app/models && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python healthcheck.py

# Set entrypoint
ENTRYPOINT ["./entrypoint.sh"]

# Default command
CMD ["uvicorn", "spatial_omics_gfm.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Labels for metadata
LABEL maintainer="Daniel Schmidt <daniel@spatial-omics.ai>"
LABEL version="${VERSION}"
LABEL description="Spatial-Omics Graph Foundation Model - Production Container"
LABEL org.opencontainers.image.source="https://github.com/danieleschmidt/spatial-omics-gfm"
LABEL org.opencontainers.image.title="Spatial-Omics GFM"
LABEL org.opencontainers.image.description="Graph Foundation Model for Spatial Transcriptomics"
LABEL org.opencontainers.image.version="${VERSION}"
LABEL org.opencontainers.image.created="$(date -u +%Y-%m-%dT%H:%M:%SZ)"