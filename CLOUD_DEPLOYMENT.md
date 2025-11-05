# Cloud-Native Deployment Guide

This guide covers deploying the Precise MRD Pipeline in cloud-native environments with scalability, reliability, and integration capabilities.

## 🚀 Quick Start

### Local Development
```bash
# Start API server locally
./deploy.sh --local-python

# Or with Docker
./deploy.sh --local-docker
```

### Production Deployment
```bash
# Deploy to Kubernetes
./deploy.sh --kubernetes

# Deploy to AWS (requires Terraform)
cd cloud && terraform init && terraform apply
```

## 📋 Deployment Options

### 1. Local Development
- **Docker Compose**: Full local stack with API, Redis, PostgreSQL, and MinIO
- **Python Direct**: Development mode with auto-reload
- **Docker Only**: Containerized API server

### 2. Kubernetes Cluster
- **Horizontal Pod Autoscaling**: Automatic scaling based on CPU/memory
- **Load Balancing**: NGINX Ingress with SSL termination
- **Persistent Storage**: NFS-based storage for data and results
- **Health Checks**: Comprehensive monitoring and health checks

### 3. Serverless (AWS Lambda)
- **On-demand Processing**: Pay-per-use compute for individual samples
- **API Gateway Integration**: RESTful endpoints for sample submission
- **Event-driven**: Trigger processing from various sources

### 4. Multi-Cloud Support
- **AWS**: ECS Fargate, Lambda, S3, RDS
- **GCP**: Cloud Run, Cloud Functions, Cloud Storage, Cloud SQL
- **Azure**: Container Instances, Functions, Blob Storage, SQL Database

## 🔧 API Endpoints

### Core Endpoints
```
POST /submit                    # Submit pipeline job
GET  /status/{job_id}          # Check job status
GET  /results/{job_id}         # Get job results
GET  /download/{job_id}/{type} # Download artifacts
```

### Management Endpoints
```
GET  /health                   # Health check
GET  /jobs                     # List recent jobs
POST /validate-config          # Validate configuration
GET  /config-templates         # Get available templates
POST /config-from-template     # Create config from template
```

### Example Usage
```bash
# Submit a job
curl -X POST "http://localhost:8000/submit" \
  -F "run_id=test_job" \
  -F "seed=42" \
  -F "use_ml_calling=true" \
  -F "ml_model_type=ensemble"

# Check status
curl "http://localhost:8000/status/{job_id}"

# Get results
curl "http://localhost:8000/results/{job_id}"
```

## 🏗️ Architecture

### Local Development Stack
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   API Server    │───▶│     Redis       │    │   PostgreSQL    │
│   (FastAPI)     │    │   (Caching)     │    │   (Metadata)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                        │                     │
         ▼                        ▼                     ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Job Manager   │    │  Pipeline       │    │     MinIO       │
│   (Async Jobs)  │    │  Processing     │    │  (Storage)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Kubernetes Production Stack
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Load Balancer │───▶│   API Pods      │───▶│   NFS Storage   │
│   (NGINX)       │    │   (Auto-scaled) │    │   (Persistent)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                        │                     │
         ▼                        ▼                     ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Ingress       │    │  ConfigMaps     │    │   HPA           │
│   Controller    │    │   (Config)      │    │   (Auto-scale)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Serverless Architecture
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   API Gateway   │───▶│   Lambda        │───▶│   S3 Storage    │
│   (REST API)    │    │   Functions     │    │   (Results)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                        │                     │
         ▼                        ▼                     ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   CloudWatch    │    │   IAM Roles     │    │   CloudTrail    │
│   (Monitoring)  │    │   (Security)    │    │   (Auditing)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🔒 Security Considerations

### API Security
- **Authentication**: JWT tokens or API keys (configurable)
- **Rate Limiting**: Built-in rate limiting via NGINX
- **CORS**: Configurable CORS policies
- **Input Validation**: Comprehensive input sanitization

### Data Security
- **Encryption**: TLS/SSL for all communications
- **Access Control**: Role-based access control (RBAC)
- **Data Isolation**: Separate storage for different tenants
- **Audit Logging**: Complete audit trails for compliance

### Network Security
- **VPC Isolation**: Private subnets for compute resources
- **Security Groups**: Least-privilege firewall rules
- **Network Policies**: Kubernetes network segmentation

## 📊 Monitoring & Observability

### Metrics Collection
- **Application Metrics**: Request latency, error rates, throughput
- **Resource Metrics**: CPU, memory, disk usage
- **Business Metrics**: Processing times, success rates, variant detection rates

### Logging
- **Structured Logging**: JSON-formatted logs with correlation IDs
- **Log Aggregation**: Centralized logging with ELK stack or cloud equivalents
- **Log Retention**: Configurable retention policies

### Alerting
- **Health Checks**: Automatic health monitoring
- **Performance Alerts**: Threshold-based alerting for key metrics
- **Error Alerts**: Immediate notification for critical failures

## 🔧 Configuration Management

### Environment Variables
```bash
# API Configuration
PRECISE_MRD_LOG_LEVEL=INFO
PRECISE_MRD_API_HOST=0.0.0.0
PRECISE_MRD_API_PORT=8000

# Processing Options
ENABLE_PARALLEL_PROCESSING=true
ENABLE_ML_CALLING=true
ENABLE_DEEP_LEARNING=true

# Performance Settings
MAX_CONCURRENT_JOBS=10
JOB_TIMEOUT_SECONDS=3600
CLEANUP_OLD_JOBS_HOURS=24
```

### Configuration Files
- **Kubernetes ConfigMaps**: Centralized configuration management
- **Environment-specific configs**: Dev/staging/production separation
- **Secret Management**: Secure handling of sensitive data

## 🚀 Scaling Strategies

### Horizontal Scaling
- **Kubernetes HPA**: Automatic scaling based on CPU/memory usage
- **Load Balancing**: Distribute traffic across multiple instances
- **Database Scaling**: Read replicas and connection pooling

### Vertical Scaling
- **Resource Allocation**: Configurable CPU/memory per instance
- **Instance Types**: Optimized instance types for different workloads

### Cost Optimization
- **Auto-scaling**: Scale down during low usage periods
- **Spot Instances**: Use preemptible instances for batch processing
- **Storage Tiering**: Use appropriate storage classes

## 🔄 CI/CD Pipeline

### Build Pipeline
```yaml
# GitHub Actions or similar
name: Build and Deploy
on:
  push:
    branches: [main]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Build Docker image
        run: docker build -t precise-mrd-api .
      - name: Push to registry
        run: docker push myregistry/precise-mrd-api:latest
```

### Deployment Pipeline
```yaml
# Kubernetes deployment
- name: Deploy to Kubernetes
  run: |
    kubectl apply -f k8s/
    kubectl rollout status deployment/precise-mrd-api
```

## 🧪 Testing

### Unit Tests
```bash
# Run unit tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src/precise_mrd --cov-report=html
```

### Integration Tests
```bash
# Test API endpoints
pytest tests/test_api.py -v

# Load testing
pytest tests/test_load.py -v
```

### Performance Tests
```bash
# Benchmark tests
pytest tests/test_benchmarks.py -v

# Stress testing
pytest tests/test_stress.py -v
```

## 📚 API Documentation

### OpenAPI/Swagger
- Interactive API documentation available at `/docs`
- ReDoc documentation available at `/redoc`
- JSON schema available at `/openapi.json`

### Client Libraries
```python
# Python client example
import requests

response = requests.post("http://localhost:8000/submit",
    data={
        "run_id": "test_job",
        "seed": 42,
        "use_ml_calling": True
    }
)

job_id = response.json()["job_id"]

# Check status
status = requests.get(f"http://localhost:8000/status/{job_id}").json()
```

## 🔧 Troubleshooting

### Common Issues

#### High Memory Usage
- **Solution**: Enable parallel processing limits and increase memory allocation
- **Check**: Monitor resource usage and adjust HPA thresholds

#### Long Processing Times
- **Solution**: Enable caching and optimize model parameters
- **Check**: Review performance metrics and adjust batch sizes

#### Connection Timeouts
- **Solution**: Increase timeout settings and implement retry logic
- **Check**: Network connectivity and load balancer configuration

### Debug Mode
```bash
# Enable debug logging
export PRECISE_MRD_LOG_LEVEL=DEBUG

# Run with detailed output
python -m precise_mrd.api --reload --log-level debug
```

## 📈 Performance Benchmarks

### Local Development
- **Single Sample**: ~30 seconds (with ML)
- **Batch Processing**: ~5 minutes for 100 samples
- **Memory Usage**: ~1GB for typical workloads

### Kubernetes Production
- **Horizontal Scaling**: 1-10 pods based on load
- **Throughput**: 1000+ samples/hour at peak
- **Availability**: 99.9% uptime with proper configuration

### Serverless Performance
- **Cold Start**: ~2-5 seconds
- **Processing Time**: ~60 seconds per sample
- **Cost Efficiency**: Pay-per-sample model

## 🔄 Migration Guide

### From Local to Kubernetes
1. Build Docker image: `docker build -t precise-mrd-api .`
2. Deploy to cluster: `./deploy.sh --kubernetes`
3. Update DNS records to point to load balancer

### From Monolithic to Serverless
1. Deploy Lambda function using SAM: `sam deploy`
2. Update client applications to use new endpoints
3. Monitor costs and performance

### Version Upgrades
1. Test new version in staging environment
2. Update Docker image tag in Kubernetes manifests
3. Perform rolling update: `kubectl rollout restart deployment/precise-mrd-api`

## 📞 Support & Maintenance

### Regular Tasks
- **Weekly**: Review performance metrics and logs
- **Monthly**: Update dependencies and security patches
- **Quarterly**: Performance optimization and capacity planning

### Emergency Procedures
1. **Service Down**: Check health endpoints and restart containers
2. **High Error Rate**: Review logs and scale up resources
3. **Data Corruption**: Restore from backups and verify integrity

### Contact Information
- **Development Team**: dev@precise-mrd.org
- **Operations Team**: ops@precise-mrd.org
- **Security Team**: security@precise-mrd.org

---

**Note**: This deployment guide assumes familiarity with Docker, Kubernetes, and cloud platforms. For detailed setup instructions for specific cloud providers, refer to the provider-specific documentation in the `cloud/` directory.


