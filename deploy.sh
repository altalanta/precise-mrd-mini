#!/bin/bash

# Precise MRD Pipeline Deployment Script
# Supports local Docker deployment and Kubernetes cluster deployment

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
DOCKER_IMAGE="precise-mrd-api"
DOCKER_TAG="latest"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Check dependencies
check_dependencies() {
    log_info "Checking dependencies..."

    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed or not in PATH"
        exit 1
    fi

    # Check kubectl (for Kubernetes deployment)
    if ! command -v kubectl &> /dev/null; then
        log_warn "kubectl not found - Kubernetes deployment will be skipped"
    fi

    # Check Python (for local development)
    if ! command -v python3 &> /dev/null; then
        log_warn "python3 not found - some features may not work"
    fi

    log_success "Dependencies check completed"
}

# Build Docker image
build_docker_image() {
    log_info "Building Docker image: $DOCKER_IMAGE:$DOCKER_TAG"

    cd "$PROJECT_ROOT"

    # Build the image
    docker build -t "$DOCKER_IMAGE:$DOCKER_TAG" -f Dockerfile .

    log_success "Docker image built successfully"
}

# Deploy to Kubernetes
deploy_kubernetes() {
    log_info "Deploying to Kubernetes cluster..."

    cd "$PROJECT_ROOT/k8s"

    # Apply configurations in order
    log_info "Creating PersistentVolumeClaims..."
    kubectl apply -f pvc.yaml

    log_info "Creating ConfigMap..."
    kubectl apply -f configmap.yaml

    log_info "Creating Deployment..."
    kubectl apply -f deployment.yaml

    log_info "Creating HorizontalPodAutoscaler..."
    kubectl apply -f hpa.yaml

    log_info "Creating Ingress..."
    kubectl apply -f ingress.yaml

    log_info "Waiting for deployment to be ready..."
    kubectl rollout status deployment/precise-mrd-api --timeout=300s

    log_success "Kubernetes deployment completed"
    log_info "API will be available at: https://mrd-api.example.com"
    log_info "API documentation at: https://mrd-api.example.com/docs"
}

# Run locally with Docker
run_local_docker() {
    log_info "Running locally with Docker..."

    cd "$PROJECT_ROOT"

    # Create local data directories
    mkdir -p data api_results

    # Run the container
    docker run -d \
        --name precise-mrd-api \
        -p 8000:8000 \
        -v "$(pwd)/data:/app/data" \
        -v "$(pwd)/api_results:/app/api_results" \
        -e PRECISE_MRD_LOG_LEVEL=INFO \
        "$DOCKER_IMAGE:$DOCKER_TAG"

    log_success "API server running locally"
    log_info "API available at: http://localhost:8000"
    log_info "API documentation at: http://localhost:8000/docs"
    log_info "Use 'docker stop precise-mrd-api' to stop the container"
}

# Run locally with Python (development)
run_local_python() {
    log_info "Running locally with Python (development mode)..."

    cd "$PROJECT_ROOT"

    # Install dependencies if needed
    if [ ! -d ".venv" ]; then
        log_info "Installing Python dependencies..."
        pip install uv
        uv sync --extra dev
    fi

    # Run the API server
    export PYTHONPATH="$PROJECT_ROOT/src"
    python -m precise_mrd.api --host 0.0.0.0 --port 8000 --reload
}

# Show usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --local-docker    Run locally using Docker container"
    echo "  --local-python    Run locally using Python (development)"
    echo "  --kubernetes      Deploy to Kubernetes cluster"
    echo "  --build-only      Only build Docker image"
    echo "  --help           Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --local-docker     # Run locally with Docker"
    echo "  $0 --kubernetes       # Deploy to Kubernetes"
    echo "  $0 --build-only       # Build Docker image only"
}

# Main logic
main() {
    local mode=""

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --local-docker)
                mode="local-docker"
                shift
                ;;
            --local-python)
                mode="local-python"
                shift
                ;;
            --kubernetes)
                mode="kubernetes"
                shift
                ;;
            --build-only)
                mode="build-only"
                shift
                ;;
            --help)
                usage
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                usage
                exit 1
                ;;
        esac
    done

    # Default mode
    if [ -z "$mode" ]; then
        if command -v kubectl &> /dev/null; then
            mode="kubernetes"
        elif command -v docker &> /dev/null; then
            mode="local-docker"
        else
            mode="local-python"
        fi
    fi

    log_info "Starting deployment in mode: $mode"

    # Check dependencies first
    check_dependencies

    case $mode in
        "local-docker")
            build_docker_image
            run_local_docker
            ;;
        "local-python")
            run_local_python
            ;;
        "kubernetes")
            build_docker_image
            deploy_kubernetes
            ;;
        "build-only")
            build_docker_image
            ;;
        *)
            log_error "Unknown deployment mode: $mode"
            usage
            exit 1
            ;;
    esac

    log_success "Deployment completed successfully!"
}

# Run main function
main "$@"













