# Deployment Guide for Spatial-Omics GFM

This guide covers deploying the Spatial-Omics Graph Foundation Model in various environments.

## Overview

The Spatial-Omics GFM supports multiple deployment scenarios:
- **Local Development**: Docker Compose for local testing
- **Cloud Deployment**: Kubernetes for production scaling
- **Edge Deployment**: Optimized containers for resource-constrained environments

## Prerequisites

- Docker and Docker Compose
- Kubernetes cluster (for production)
- Python 3.9+ (for local development)
- NVIDIA GPU support (optional, for CUDA acceleration)

## Quick Start with Docker

### 1. Build and Run with Docker Compose

```bash
# Clone the repository
git clone <repository-url>
cd spatial-omics-gfm

# Build and start services
docker-compose -f docker/docker-compose.yml up -d

# Check service health
docker-compose -f docker/docker-compose.yml ps
```

### 2. Access the API

The API will be available at:
- HTTP: `http://localhost:8000`
- Health check: `http://localhost:8000/health`
- API documentation: `http://localhost:8000/docs`

## Production Deployment on Kubernetes

### 1. Prerequisites

```bash
# Ensure you have kubectl configured
kubectl cluster-info

# Create namespace
kubectl create namespace spatial-omics
```

### 2. Deploy Application

```bash
# Apply Kubernetes manifests
kubectl apply -f kubernetes/deployment.yaml -n spatial-omics

# Check deployment status
kubectl get pods -n spatial-omics
kubectl get services -n spatial-omics
```

### 3. Configure Ingress (Optional)

Update the ingress configuration in `kubernetes/deployment.yaml`:

```yaml
spec:
  rules:
  - host: your-domain.com  # Replace with your domain
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: spatial-omics-gfm-service
            port:
              number: 80
```

### 4. SSL/TLS Configuration

For production deployments, configure SSL certificates:

```bash
# Install cert-manager (if not already installed)
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml

# Apply the ingress with TLS configuration
kubectl apply -f kubernetes/deployment.yaml -n spatial-omics
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SPATIAL_OMICS_LOG_LEVEL` | `INFO` | Logging level |
| `SPATIAL_OMICS_DEVICE` | `auto` | Compute device (cpu/cuda/auto) |
| `SPATIAL_OMICS_BATCH_SIZE` | `32` | Default batch size |
| `SPATIAL_OMICS_MAX_MEMORY_GB` | `8.0` | Maximum memory usage |
| `SPATIAL_OMICS_ENABLE_COMPILATION` | `true` | Enable model compilation |
| `SPATIAL_OMICS_ENABLE_QUANTIZATION` | `false` | Enable model quantization |

### Model Configuration

Models can be configured via ConfigMap in Kubernetes:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: spatial-omics-config
data:
  max_memory_gb: "16.0"
  batch_size: "64"
  enable_compilation: "true"
  enable_quantization: "false"
```

## Performance Tuning

### 1. GPU Support

For GPU acceleration, ensure NVIDIA device plugin is installed:

```bash
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.13.0/nvidia-device-plugin.yml
```

Update deployment to request GPU resources:

```yaml
resources:
  limits:
    nvidia.com/gpu: 1
```

### 2. Memory Optimization

For memory-constrained environments:

```yaml
env:
- name: SPATIAL_OMICS_ENABLE_QUANTIZATION
  value: "true"
- name: SPATIAL_OMICS_MAX_MEMORY_GB
  value: "4.0"
```

### 3. Horizontal Scaling

The included HPA configuration automatically scales based on CPU and memory:

```yaml
spec:
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

## Edge Deployment

For edge devices with limited resources:

### 1. Build Edge-Optimized Image

```dockerfile
FROM python:3.10-slim
# ... optimized for edge deployment
```

### 2. Use Quantized Models

```bash
# Enable quantization for smaller model size
export SPATIAL_OMICS_ENABLE_QUANTIZATION=true
export SPATIAL_OMICS_MAX_MEMORY_GB=2.0
```

## Security Configuration

### 1. Network Security

- Use TLS encryption for all external communications
- Configure network policies to restrict pod-to-pod communication
- Enable ingress rate limiting

### 2. Container Security

```yaml
securityContext:
  runAsNonRoot: true
  runAsUser: 1000
  allowPrivilegeEscalation: false
  readOnlyRootFilesystem: true
  capabilities:
    drop:
    - ALL
```

### 3. Secrets Management

Use Kubernetes secrets for sensitive configuration:

```bash
kubectl create secret generic spatial-omics-secrets \
  --from-literal=api-key=your-api-key \
  --from-literal=db-password=your-db-password
```

## Monitoring and Observability

### 1. Health Checks

The application provides health endpoints:
- `/health`: Basic health check
- `/ready`: Readiness check
- `/metrics`: Prometheus metrics

### 2. Logging

Logs are structured JSON format with configurable levels:

```bash
# View logs
kubectl logs -f deployment/spatial-omics-gfm -n spatial-omics
```

### 3. Metrics

Prometheus metrics are exposed at `/metrics`:
- Request duration and count
- Model inference metrics
- Memory and CPU usage
- Error rates

## Backup and Recovery

### 1. Model Backup

```bash
# Backup model data
kubectl exec -it pod-name -n spatial-omics -- tar -czf /tmp/models-backup.tar.gz /app/models
kubectl cp spatial-omics/pod-name:/tmp/models-backup.tar.gz ./models-backup.tar.gz
```

### 2. Configuration Backup

```bash
# Export configuration
kubectl get configmap spatial-omics-config -n spatial-omics -o yaml > config-backup.yaml
kubectl get secret spatial-omics-secrets -n spatial-omics -o yaml > secrets-backup.yaml
```

## Troubleshooting

### Common Issues

1. **Pod stuck in Pending state**
   - Check resource requests vs cluster capacity
   - Verify node affinity/anti-affinity rules

2. **High memory usage**
   - Reduce batch size
   - Enable quantization
   - Check for memory leaks in logs

3. **Slow inference**
   - Verify GPU availability
   - Enable model compilation
   - Check network latency

### Debug Commands

```bash
# Check pod status
kubectl describe pod pod-name -n spatial-omics

# View resource usage
kubectl top pods -n spatial-omics

# Access pod shell
kubectl exec -it pod-name -n spatial-omics -- /bin/bash

# Check service endpoints
kubectl get endpoints -n spatial-omics
```

## Scaling Guidelines

### Vertical Scaling

Increase resources per pod:

```yaml
resources:
  requests:
    cpu: 2000m
    memory: 8Gi
  limits:
    cpu: 8000m
    memory: 16Gi
```

### Horizontal Scaling

Increase replica count:

```bash
kubectl scale deployment spatial-omics-gfm --replicas=5 -n spatial-omics
```

### Auto-scaling

The HPA automatically adjusts replicas based on metrics:
- CPU utilization > 70%
- Memory utilization > 80%
- Custom metrics (queue length, inference time)

## Cost Optimization

1. **Use spot instances** for non-critical workloads
2. **Enable cluster autoscaling** to match demand
3. **Use appropriate instance types** (CPU vs GPU)
4. **Implement request/response caching**
5. **Use model quantization** to reduce memory requirements

## Support

For deployment issues:
1. Check the troubleshooting section above
2. Review application logs
3. Verify configuration settings
4. Contact support with deployment details

## Version Updates

To update to a new version:

```bash
# Update image tag in deployment
kubectl set image deployment/spatial-omics-gfm spatial-omics-gfm=new-image:tag -n spatial-omics

# Monitor rollout
kubectl rollout status deployment/spatial-omics-gfm -n spatial-omics

# Rollback if needed
kubectl rollout undo deployment/spatial-omics-gfm -n spatial-omics
```