# Multi-stage Dockerfile for Precise MRD Pipeline API
FROM python:3.11-slim as builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN pip install --no-cache-dir uv && \
    uv sync --frozen

# Production stage
FROM python:3.11-slim

# Install runtime system dependencies
RUN apt-get update && apt-get install -y \
    libomp5 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r mrd && useradd -r -g mrd mrd

# Set work directory
WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy application code
COPY src/precise_mrd /app/precise_mrd

# Create directories for data and results
RUN mkdir -p /app/data /app/api_results && \
    chown -R mrd:mrd /app

# Switch to non-root user
USER mrd

# Set Python path
ENV PYTHONPATH=/app
ENV PATH="/app/.venv/bin:$PATH"

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the API server
CMD ["python", "-m", "precise_mrd.api"]
