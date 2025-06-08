# Darwin Deployment Pipeline Documentation

## Overview

This document provides comprehensive guidance for deploying the Darwin Genetic Algorithm Platform with its integrated monitoring system. The deployment pipeline supports multiple environments and deployment strategies with full observability.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Deployment Environments](#deployment-environments)
3. [Monitoring Integration](#monitoring-integration)
4. [Deployment Methods](#deployment-methods)
5. [Configuration](#configuration)
6. [Security](#security)
7. [Monitoring and Observability](#monitoring-and-observability)
8. [Troubleshooting](#troubleshooting)
9. [Rollback Procedures](#rollback-procedures)

## Prerequisites

### System Requirements

- **Docker**: Version 20.10+
- **Docker Compose**: Version 2.0+
- **Kubernetes**: Version 1.24+ (for K8s deployments)
- **Terraform**: Version 1.0+ (for infrastructure)
- **AWS CLI**: Version 2.0+ (for AWS deployments)
- **Git**: Version 2.30+

### Required Tools

```bash
# Install required tools
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Install kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl

# Install Terraform
wget -O- https://apt.releases.hashicorp.com/gpg | sudo gpg --dearmor -o /usr/share/keyrings/hashicorp-archive-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/hashicorp-archive-keyring.gpg] https://apt.releases.hashicorp.com $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/hashicorp.list
sudo apt update && sudo apt install terraform
```

### Environment Variables

Create a `.env` file with the following required variables:

```bash
# Core Configuration
ENVIRONMENT=production
DEBUG=false
PYTHONPATH=/app/src

# Database Configuration
DB_PASSWORD=your_secure_database_password
REDIS_PASSWORD=your_secure_redis_password

# Security Configuration
JWT_SECRET_KEY=your_jwt_secret_key_minimum_32_characters_long
CORS_ORIGINS=https://darwin.devq.ai,https://dashboard.devq.ai

# Monitoring Configuration (Updated System)
LOGFIRE_TOKEN=your_logfire_token
MONITORING_ENABLED=true
PERFORMANCE_MONITORING=true
HEALTH_CHECK_INTERVAL=30
METRICS_COLLECTION_INTERVAL=15
ALERT_COOLDOWN_PERIOD=300
TRACE_SAMPLE_RATE=0.1

# Notification Configuration
SMTP_SERVER=your_smtp_server
SMTP_PORT=587
SMTP_USERNAME=your_smtp_username
SMTP_PASSWORD=your_smtp_password
SMTP_FROM_EMAIL=noreply@devq.ai
ALERT_EMAIL=alerts@devq.ai
SLACK_WEBHOOK_URL=your_slack_webhook_url

# Infrastructure Configuration
GRAFANA_PASSWORD=your_grafana_admin_password
GRAFANA_SECRET_KEY=your_grafana_secret_key

# AWS Configuration (if using AWS deployment)
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_REGION=us-west-2
```

## Deployment Environments

### Development Environment

Quick setup for local development with hot reloading:

```bash
# Clone repository
git clone https://github.com/devq-ai/darwin.git
cd darwin

# Start development environment
docker-compose up -d

# Access services
# API: http://localhost:8000
# Dashboard: http://localhost:5006
# Health: http://localhost:8000/health/detailed
# Metrics: http://localhost:8000/metrics
# Performance: http://localhost:8000/performance
# Tracing: http://localhost:8000/tracing
```

### Staging Environment

Deploy to staging with monitoring:

```bash
# Deploy to staging
ENVIRONMENT=staging docker-compose -f docker-compose.yml -f docker-compose.override.yml up -d

# Enable monitoring stack
docker-compose --profile monitoring up -d
```

### Production Environment

Full production deployment with comprehensive monitoring:

```bash
# Deploy production with monitoring
./scripts/deployment/deploy.sh --environment production --mode rolling

# Alternative: Blue-green deployment
./scripts/deployment/deploy.sh --environment production --mode blue-green
```

## Monitoring Integration

The updated Darwin monitoring system provides comprehensive observability:

### Core Monitoring Components

1. **Health Monitoring System** (`health.py`)
   - Database, API, MCP server, WebSocket health checks
   - System resource monitoring (CPU, memory, disk)
   - Custom health check registration
   - Graceful degradation when dependencies unavailable

2. **Metrics Collection System** (`metrics.py`)
   - Prometheus-style metrics (Counter, Gauge, Histogram, Timer)
   - Darwin-specific metrics for genetic algorithms
   - Time-series data storage with TTL
   - FastAPI middleware integration

3. **Alert Management System** (`alerts.py`)
   - Configurable alert rules with thresholds
   - Multi-channel notifications (Email, Slack, Webhook)
   - Alert escalation and acknowledgment tracking
   - Rate limiting and cooldown periods

4. **Performance Monitoring System** (`performance.py`)
   - Real-time performance tracking
   - Function profiling and analysis
   - Memory usage tracking
   - Benchmark suite for genetic algorithms

5. **Distributed Tracing System** (`tracing.py`)
   - OpenTelemetry-compatible spans and traces
   - Request correlation across services
   - Custom span creation with attributes
   - Service dependency mapping

6. **Monitoring Utilities** (`utils.py`)
   - Time-series data structures
   - Threshold management
   - Anomaly detection
   - Data export/import utilities

### Monitoring Endpoints

The monitoring system exposes the following endpoints:

```bash
# Health Checks
curl http://localhost:8000/health                    # Basic health
curl http://localhost:8000/health/detailed           # Detailed health
curl http://localhost:8000/health/history           # Health history

# Metrics
curl http://localhost:8000/metrics                  # Prometheus format
curl http://localhost:8000/metrics/json             # JSON format

# Performance
curl http://localhost:8000/performance              # Performance summary
curl http://localhost:8000/performance/profiles     # Profiling data
curl http://localhost:8000/performance/metrics      # Performance metrics

# Tracing
curl http://localhost:8000/tracing                  # Tracing statistics
curl http://localhost:8000/tracing/traces           # Recent traces
curl http://localhost:8000/tracing/errors           # Error traces
```

## Deployment Methods

### Method 1: Docker Compose (Recommended for Development/Staging)

```bash
# Basic deployment
docker-compose -f docker-compose.prod.yml up -d

# With monitoring stack
docker-compose -f docker-compose.prod.yml --profile monitoring up -d

# Scale services
docker-compose -f docker-compose.prod.yml up -d --scale darwin-api=3
```

### Method 2: Kubernetes (Recommended for Production)

```bash
# Apply Kubernetes manifests
kubectl apply -f k8s/

# Check deployment status
kubectl get pods -n darwin-platform
kubectl get services -n darwin-platform

# Monitor rollout
kubectl rollout status deployment/darwin-api -n darwin-platform
```

### Method 3: Automated Deployment Script

```bash
# Full automated deployment
./scripts/deployment/deploy.sh

# Custom configuration
./scripts/deployment/deploy.sh \
  --environment production \
  --mode rolling \
  --no-backup \
  --monitoring-enabled

# Deploy with Terraform infrastructure
cd infrastructure/terraform
terraform init
terraform plan -var-file="production.tfvars"
terraform apply -var-file="production.tfvars"
```

## Configuration

### Monitoring Configuration

The monitoring system can be configured through environment variables:

```bash
# Enable/disable monitoring components
MONITORING_ENABLED=true
PERFORMANCE_MONITORING=true
HEALTH_CHECK_INTERVAL=30
METRICS_COLLECTION_INTERVAL=15

# Alert configuration
ALERT_COOLDOWN_PERIOD=300
ALERT_EMAIL=alerts@devq.ai
SLACK_WEBHOOK_URL=https://hooks.slack.com/...

# Tracing configuration
TRACE_SAMPLE_RATE=0.1
LOGFIRE_TOKEN=your_token

# Threshold configuration
CPU_THRESHOLD_WARNING=70
CPU_THRESHOLD_CRITICAL=90
MEMORY_THRESHOLD_WARNING=80
MEMORY_THRESHOLD_CRITICAL=95
```

### Custom Alert Rules

Create custom alert rules programmatically:

```python
from darwin.monitoring import AlertManager, AlertSeverity, NotificationChannel

alert_manager = AlertManager()

# High API error rate alert
alert_manager.create_rule(
    name="high_api_error_rate",
    description="API error rate is above 5%",
    metric_name="darwin_api_error_rate",
    condition="gt",
    threshold=5.0,
    severity=AlertSeverity.CRITICAL,
    channels=[NotificationChannel.EMAIL, NotificationChannel.SLACK],
    duration=180
)

# Genetic algorithm convergence alert
alert_manager.create_rule(
    name="slow_convergence",
    description="Genetic algorithm convergence is slow",
    metric_name="darwin_convergence_rate",
    condition="lt",
    threshold=0.001,
    severity=AlertSeverity.WARNING,
    channels=[NotificationChannel.EMAIL],
    duration=600
)
```

## Security

### Network Security

```bash
# Enable network policies (Kubernetes)
kubectl apply -f k8s/network-policies.yaml

# Configure firewall rules (Docker)
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw deny 8000/tcp  # Block direct API access
```

### Secrets Management

```bash
# Create Kubernetes secrets
kubectl create secret generic darwin-secrets \
  --from-literal=jwt-secret-key=$JWT_SECRET_KEY \
  --from-literal=db-password=$DB_PASSWORD \
  --from-literal=redis-password=$REDIS_PASSWORD \
  --from-literal=logfire-token=$LOGFIRE_TOKEN \
  -n darwin-platform

# AWS Secrets Manager (Terraform)
terraform apply -target=aws_secretsmanager_secret.darwin_secrets
```

### SSL/TLS Configuration

```bash
# Generate SSL certificates
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout darwin.key -out darwin.crt \
  -subj "/CN=darwin.devq.ai/O=DevQ.AI"

# Configure in nginx
kubectl create secret tls darwin-tls \
  --cert=darwin.crt --key=darwin.key \
  -n darwin-platform
```

## Monitoring and Observability

### Grafana Dashboards

Access monitoring dashboards:

- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090
- **Alert Manager**: http://localhost:9093

### Pre-configured Dashboards

1. **Darwin API Performance**
   - Request rate and latency
   - Error rates and response codes
   - Resource utilization

2. **Genetic Algorithm Metrics**
   - Population fitness distribution
   - Convergence rates
   - Algorithm performance comparisons

3. **System Health**
   - Infrastructure health
   - Service dependencies
   - Alert status

4. **Performance Analysis**
   - Function profiling results
   - Memory usage patterns
   - Bottleneck identification

### Log Aggregation

```bash
# View logs
docker-compose logs -f darwin-api
kubectl logs -f deployment/darwin-api -n darwin-platform

# Structured log queries (if using Loki)
curl -G -s "http://localhost:3100/loki/api/v1/query" \
  --data-urlencode 'query={service="darwin-api"} |= "ERROR"'
```

## Troubleshooting

### Common Issues

#### 1. Monitoring System Not Starting

```bash
# Check monitoring dependencies
docker-compose ps
kubectl get pods -n monitoring

# Verify configuration
curl http://localhost:8000/health/detailed

# Check logs
docker-compose logs darwin-api
```

#### 2. Metrics Not Being Collected

```bash
# Verify metrics endpoint
curl http://localhost:8000/metrics

# Check Prometheus targets
curl http://localhost:9090/api/v1/targets

# Restart metrics collection
docker-compose restart darwin-api
```

#### 3. Alerts Not Firing

```bash
# Check alert rules
curl http://localhost:9090/api/v1/rules

# Verify alert manager configuration
curl http://localhost:9093/api/v1/status

# Test notification channels
docker-compose exec darwin-api python -c "
from darwin.monitoring import AlertManager
alert_manager = AlertManager()
alert_manager.test_notifications()
"
```

#### 4. Performance Issues

```bash
# Check resource usage
docker stats
kubectl top pods -n darwin-platform

# Review performance metrics
curl http://localhost:8000/performance

# Analyze profiling data
curl http://localhost:8000/performance/profiles
```

### Diagnostic Commands

```bash
# Health check all services
./scripts/deployment/health-check.sh

# Generate diagnostic report
./scripts/deployment/diagnose.sh

# Performance benchmark
./scripts/deployment/benchmark.sh

# Backup current state
./scripts/deployment/backup.sh
```

## Rollback Procedures

### Automatic Rollback

The deployment script includes automatic rollback on failure:

```bash
# Deploy with automatic rollback enabled (default)
./scripts/deployment/deploy.sh --environment production

# Disable automatic rollback
./scripts/deployment/deploy.sh --no-rollback
```

### Manual Rollback

#### Docker Compose Rollback

```bash
# Rollback to previous version
docker-compose down
docker-compose -f docker-compose.prod.yml up -d --scale darwin-api=0
docker-compose -f docker-compose.prod.yml up -d
```

#### Kubernetes Rollback

```bash
# Check rollout history
kubectl rollout history deployment/darwin-api -n darwin-platform

# Rollback to previous version
kubectl rollout undo deployment/darwin-api -n darwin-platform

# Rollback to specific revision
kubectl rollout undo deployment/darwin-api --to-revision=2 -n darwin-platform
```

#### Database Rollback

```bash
# Restore from backup
./scripts/deployment/restore.sh --backup-name darwin_backup_20241208_120000

# Verify restoration
curl http://localhost:8000/health/detailed
```

## Performance Optimization

### Monitoring-Based Optimization

Use the monitoring system to identify optimization opportunities:

```bash
# Get performance insights
curl http://localhost:8000/performance/metrics

# Analyze bottlenecks
curl http://localhost:8000/performance/profiles

# Check genetic algorithm performance
curl http://localhost:8000/metrics | grep darwin_optimization
```

### Scaling Recommendations

Based on monitoring data:

```bash
# Scale API instances based on CPU usage
if [ $(curl -s http://localhost:8000/metrics | grep cpu_usage | cut -d' ' -f2) > 80 ]; then
  docker-compose up -d --scale darwin-api=3
fi

# Kubernetes horizontal pod autoscaler
kubectl apply -f k8s/hpa.yaml
```

## Maintenance

### Regular Maintenance Tasks

```bash
# Update monitoring dashboards
./scripts/deployment/update-dashboards.sh

# Rotate secrets
./scripts/deployment/rotate-secrets.sh

# Cleanup old metrics data
./scripts/deployment/cleanup-metrics.sh

# Performance tuning
./scripts/deployment/tune-performance.sh
```

### Backup Strategy

```bash
# Daily automated backup
0 2 * * * /path/to/darwin/scripts/deployment/backup.sh

# Verify backup integrity
./scripts/deployment/verify-backup.sh

# Test restore procedure
./scripts/deployment/test-restore.sh
```

## Support

### Getting Help

- **Documentation**: [GitHub Wiki](https://github.com/devq-ai/darwin/wiki)
- **Issues**: [GitHub Issues](https://github.com/devq-ai/darwin/issues)
- **Monitoring**: Check `/health/detailed` endpoint for system status
- **Logs**: Use monitoring dashboards for centralized logging

### Monitoring Support

The integrated monitoring system provides:

- Real-time health status
- Performance metrics and profiling
- Distributed tracing
- Automated alerting
- Historical data analysis

For monitoring-specific issues, check the monitoring endpoints and Grafana dashboards for detailed insights.
