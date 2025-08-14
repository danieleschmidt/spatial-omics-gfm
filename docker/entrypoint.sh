#!/bin/bash
# Production entrypoint script for Spatial-Omics GFM
# Handles initialization, configuration, and graceful startup

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}" >&2
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}" >&2
}

warning() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}" >&2
}

info() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] INFO: $1${NC}" >&2
}

# Environment configuration
export ENVIRONMENT=${ENVIRONMENT:-production}
export LOG_LEVEL=${LOG_LEVEL:-INFO}
export PORT=${PORT:-8000}
export WORKERS=${WORKERS:-4}

log "Starting Spatial-Omics GFM container in ${ENVIRONMENT} mode"

# Check if running as root (should not be)
if [ "$EUID" -eq 0 ]; then
    error "Container should not run as root for security reasons"
    exit 1
fi

# Create necessary directories
mkdir -p /app/logs /app/data /app/models /app/tmp

# Set permissions
chmod 755 /app/logs /app/data /app/models /app/tmp

# Environment-specific configuration
case $ENVIRONMENT in
    "development")
        log "Configuring for development environment"
        export DEBUG=true
        export LOG_LEVEL=DEBUG
        export WORKERS=1
        ;;
    "staging")
        log "Configuring for staging environment"
        export DEBUG=false
        export LOG_LEVEL=INFO
        export WORKERS=2
        ;;
    "production")
        log "Configuring for production environment"
        export DEBUG=false
        export LOG_LEVEL=INFO
        export WORKERS=${WORKERS:-4}
        ;;
    *)
        warning "Unknown environment: $ENVIRONMENT, defaulting to production"
        export ENVIRONMENT=production
        export DEBUG=false
        export LOG_LEVEL=INFO
        ;;
esac

# Validate required environment variables
REQUIRED_VARS=()

# Check for required variables based on environment
if [ "$ENVIRONMENT" = "production" ]; then
    # Add production-specific required variables if any
    REQUIRED_VARS+=()
fi

for var in "${REQUIRED_VARS[@]}"; do
    if [ -z "${!var}" ]; then
        error "Required environment variable $var is not set"
        exit 1
    fi
done

# Database connectivity check (if applicable)
check_database() {
    local max_attempts=30
    local attempt=1
    
    if [ -n "$DATABASE_URL" ]; then
        log "Checking database connectivity..."
        
        while [ $attempt -le $max_attempts ]; do
            if python -c "
import os
import sys
import psycopg2
try:
    conn = psycopg2.connect(os.environ['DATABASE_URL'])
    conn.close()
    print('Database connection successful')
    sys.exit(0)
except Exception as e:
    print(f'Database connection failed: {e}')
    sys.exit(1)
" 2>/dev/null; then
                log "Database connection established"
                return 0
            fi
            
            warning "Database connection attempt $attempt/$max_attempts failed"
            sleep 2
            ((attempt++))
        done
        
        error "Failed to connect to database after $max_attempts attempts"
        exit 1
    else
        info "No database configuration found, skipping database check"
    fi
}

# Redis connectivity check (if applicable)
check_redis() {
    if [ -n "$REDIS_URL" ]; then
        log "Checking Redis connectivity..."
        
        if python -c "
import os
import redis
try:
    r = redis.from_url(os.environ['REDIS_URL'])
    r.ping()
    print('Redis connection successful')
except Exception as e:
    print(f'Redis connection failed: {e}')
    exit(1)
" 2>/dev/null; then
            log "Redis connection established"
        else
            error "Failed to connect to Redis"
            exit 1
        fi
    else
        info "No Redis configuration found, skipping Redis check"
    fi
}

# Health check function
health_check() {
    log "Performing initial health check..."
    
    # Check Python imports
    if ! python -c "
import spatial_omics_gfm
print('Package imports successful')
"; then
        error "Failed to import package"
        exit 1
    fi
    
    # Check CUDA availability (if GPU support is needed)
    if [ "$GPU_ENABLED" = "true" ]; then
        python -c "
import torch
if torch.cuda.is_available():
    print(f'CUDA available: {torch.cuda.device_count()} devices')
else:
    print('CUDA not available, running on CPU')
"
    fi
    
    log "Health check passed"
}

# Signal handling for graceful shutdown
handle_signal() {
    log "Received shutdown signal, shutting down gracefully..."
    
    # If there's a main process PID, send it SIGTERM
    if [ -n "$MAIN_PID" ]; then
        kill -TERM "$MAIN_PID" 2>/dev/null || true
        wait "$MAIN_PID"
    fi
    
    log "Shutdown complete"
    exit 0
}

# Set up signal handlers
trap handle_signal SIGTERM SIGINT

# Run pre-startup checks
log "Running pre-startup checks..."
check_database
check_redis
health_check

# Run migrations if needed
if [ "$RUN_MIGRATIONS" = "true" ] && [ -n "$DATABASE_URL" ]; then
    log "Running database migrations..."
    python -m spatial_omics_gfm.migrations || {
        error "Database migrations failed"
        exit 1
    }
fi

# Warm up the application if needed
if [ "$ENABLE_WARMUP" = "true" ]; then
    log "Warming up application..."
    python -c "
import spatial_omics_gfm
# Add any warmup logic here
print('Warmup complete')
" || warning "Warmup failed, continuing anyway"
fi

# Log final configuration
log "Configuration summary:"
info "  Environment: $ENVIRONMENT"
info "  Log Level: $LOG_LEVEL"
info "  Port: $PORT"
info "  Workers: $WORKERS"
info "  Debug: ${DEBUG:-false}"

# Execute the main command
log "Starting main application: $*"

if [ $# -eq 0 ]; then
    # Default command
    exec uvicorn spatial_omics_gfm.api.main:app \
        --host 0.0.0.0 \
        --port "$PORT" \
        --workers "$WORKERS" \
        --log-level "${LOG_LEVEL,,}" \
        --access-log \
        --loop uvloop \
        --http httptools &
else
    # Custom command
    exec "$@" &
fi

# Store the main process PID for signal handling
MAIN_PID=$!
wait $MAIN_PID