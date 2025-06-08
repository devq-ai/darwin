# Production Deployment Guide

This comprehensive guide covers deploying Darwin in production environments, including container orchestration, infrastructure as code, monitoring setup, and best practices for scalable, reliable deployments.

## 📋 Table of Contents

1. [Deployment Overview](#deployment-overview)
2. [Environment Setup](#environment-setup)
3. [Docker Deployment](#docker-deployment)
4. [Kubernetes Deployment](#kubernetes-deployment)
5. [Infrastructure as Code](#infrastructure-as-code)
6. [Load Balancing & Scaling](#load-balancing--scaling)
7. [Database Setup](#database-setup)
8. [Monitoring & Observability](#monitoring--observability)
9. [Security Configuration](#security-configuration)
10. [Backup & Recovery](#backup--recovery)
11. [Performance Tuning](#performance-tuning)
12. [Troubleshooting](#troubleshooting)

## 🎯 Deployment Overview

### Architecture Components

```
┌─────────────────────────────────────────────────────────────────┐
│                     Production Architecture                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │ Load        │    │ Darwin      │    │ Darwin      │         │
│  │ Balancer    │───▶│ Instance 1  │    │ Instance 2  │         │
│  │ (nginx)     │    │             │    │             │         │
│  └─────────────┘    └─────────────┘    └─────────────┘         │
│         │                   │                   │               │
│         │            ┌─────────────────────────────────┐       │
│         │            │        Shared Services         │       │
│         │            ├─────────────────────────────────┤       │
│         │            │ • SurrealDB Cluster            │       │
│         │            │ • Redis Cluster                │       │
│         │            │ • Monitoring Stack             │       │
│         │            │ • Backup Services              │       │
│         │            └─────────────────────────────────┘       │
│         │                                                      │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                   External Services                        │ │
│  │ • Logfire (Observability)   • Email/SMS (Alerts)         │ │
│  │ • Object Storage (Backups)   • DNS Provider               │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### Deployment Options

| Option | Use Case | Complexity | Scalability | Cost |
|--------|----------|------------|-------------|------|
| **Single Server** | Development, Small Teams | Low | Limited | Low |
| **Docker Compose** | Small Production | Medium | Medium | Medium |
| **Kubernetes** | Enterprise, High Scale | High | High | High |
| **Managed Services** | Rapid Deployment | Low | High | Medium |

## 🛠️ Environment Setup

### System Requirements

#### Minimum Requirements
- **CPU**: 4 cores
- **RAM**: 8 GB
- **Storage**: 100 GB SSD
- **Network**: 1 Gbps

#### Recommended Requirements
- **CPU**: 8+ cores
- **RAM**: 16+ GB
- **Storage**: 500 GB+ NVMe SSD
- **Network**: 10 Gbps

#### High-Scale Requirements
- **CPU**: 16+ cores
- **RAM**: 32+ GB
- **Storage**: 1 TB+ NVMe SSD
- **Network**: 10+ Gbps

### Operating System Setup

#### Ubuntu 22.04 LTS (Recommended)

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install essential packages
sudo apt install -y \
    curl \
    wget \
    git \
    htop \
    unzip \
    software-properties-common \
    apt-transport-https \
    ca-certificates \
    gnupg \
    lsb-release

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Install kubectl (for Kubernetes deployments)
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl

# Install Terraform
wget -O- https://apt.releases.hashicorp.com/gpg | gpg --dearmor | sudo tee /usr/share/keyrings/hashicorp-archive-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/hashicorp-archive-keyring.gpg] https://apt.releases.hashicorp.com $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/hashicorp.list
sudo apt update && sudo apt install terraform
```

#### CentOS 8 / RHEL 8

```bash
# Update system
sudo dnf update -y

# Install essential packages
sudo dnf install -y \
    curl \
    wget \
    git \
    htop \
    unzip \
    yum-utils \
    device-mapper-persistent-data \
    lvm2

# Install Docker
sudo dnf config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo
sudo dnf install -y docker-ce docker-ce-cli containerd.io
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker $USER

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

### Environment Variables

Create production environment configuration:

```bash
# /opt/darwin/.env
# Production Environment Configuration

# Application Settings
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO
API_HOST=0.0.0.0
API_PORT=8000

# Security Settings
SECRET_KEY=your-256-bit-secret-key-here
JWT_SECRET_KEY=your-jwt-secret-key-here
ENCRYPTION_KEY=your-encryption-key-here
ALLOWED_HOSTS=darwin.yourdomain.com,api.yourdomain.com

# Database Configuration
SURREALDB_URL=ws://surrealdb-cluster:8000/rpc
SURREALDB_USERNAME=darwin_prod
SURREALDB_PASSWORD=your-secure-password
SURREALDB_NAMESPACE=production
SURREALDB_DATABASE=darwin

# Redis Configuration
REDIS_URL=redis://redis-cluster:6379/0
REDIS_PASSWORD=your-redis-password

# Monitoring & Observability
LOGFIRE_TOKEN=your-logfire-token
LOGFIRE_PROJECT_NAME=darwin-production
LOGFIRE_ENVIRONMENT=production
PROMETHEUS_ENABLED=true
PROMETHEUS_PORT=9090

# Performance Settings
WORKERS=4
MAX_REQUESTS=1000
WORKER_TIMEOUT=300
KEEP_ALIVE=2

# External Services
EMAIL_SMTP_HOST=smtp.your-provider.com
EMAIL_SMTP_PORT=587
EMAIL_SMTP_USERNAME=alerts@yourdomain.com
EMAIL_SMTP_PASSWORD=your-email-password

# Backup Configuration
BACKUP_S3_BUCKET=darwin-backups-prod
BACKUP_S3_REGION=us-east-1
BACKUP_SCHEDULE=0 2 * * *  # Daily at 2 AM

# Rate Limiting
RATE_LIMIT_ENABLED=true
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=60

# File Upload Limits
MAX_UPLOAD_SIZE=100MB
ALLOWED_FILE_TYPES=json,csv,yaml
```

## 🐳 Docker Deployment

### Single Server Deployment

#### Quick Start with Docker Compose

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  darwin-app:
    image: devqai/darwin:latest
    restart: unless-stopped
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=production
    env_file:
      - .env
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    depends_on:
      - surrealdb
      - redis
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  surrealdb:
    image: surrealdb/surrealdb:latest
    restart: unless-stopped
    ports:
      - "8001:8000"
    command: start --log trace --user root --pass ${SURREALDB_PASSWORD} file:///data/database.db
    volumes:
      - surrealdb_data:/data
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  redis:
    image: redis:7-alpine
    restart: unless-stopped
    ports:
      - "6379:6379"
    command: redis-server --requirepass ${REDIS_PASSWORD}
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "auth", "${REDIS_PASSWORD}", "ping"]
      interval: 30s
      timeout: 10s
      retries: 3

  nginx:
    image: nginx:alpine
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./nginx/ssl:/etc/nginx/ssl:ro
      - ./nginx/logs:/var/log/nginx
    depends_on:
      - darwin-app
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  prometheus:
    image: prom/prometheus:latest
    restart: unless-stopped
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'

  grafana:
    image: grafana/grafana:latest
    restart: unless-stopped
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana/provisioning:/etc/grafana/provisioning

volumes:
  surrealdb_data:
  redis_data:
  prometheus_data:
  grafana_data:

networks:
  default:
    driver: bridge
```

#### Nginx Configuration

```nginx
# nginx/nginx.conf
events {
    worker_connections 1024;
}

http {
    upstream darwin_backend {
        server darwin-app:8000;
    }

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req_zone $binary_remote_addr zone=dashboard:10m rate=5r/s;

    # SSL Configuration
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;
    ssl_session_cache shared:SSL:10m;

    # Security Headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;

    server {
        listen 80;
        server_name darwin.yourdomain.com;
        return 301 https://$server_name$request_uri;
    }

    server {
        listen 443 ssl http2;
        server_name darwin.yourdomain.com;

        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;

        # API endpoints
        location /api/ {
            limit_req zone=api burst=20 nodelay;

            proxy_pass http://darwin_backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;

            # WebSocket support
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";

            # Timeouts
            proxy_connect_timeout 60s;
            proxy_send_timeout 60s;
            proxy_read_timeout 60s;
        }

        # Dashboard
        location /dashboard/ {
            limit_req zone=dashboard burst=10 nodelay;

            proxy_pass http://darwin_backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }

        # Health check endpoint
        location /health {
            proxy_pass http://darwin_backend;
            access_log off;
        }

        # Static files
        location /static/ {
            alias /app/static/;
            expires 30d;
            add_header Cache-Control "public, immutable";
        }
    }
}
```

#### Deployment Script

```bash
#!/bin/bash
# deploy-docker.sh

set -e

echo "🚀 Starting Darwin Production Deployment"

# Configuration
BACKUP_DIR="/opt/darwin/backups"
DATA_DIR="/opt/darwin/data"
LOG_DIR="/opt/darwin/logs"

# Create directories
sudo mkdir -p $BACKUP_DIR $DATA_DIR $LOG_DIR
sudo chown -R $USER:$USER /opt/darwin

# Backup existing deployment
if docker-compose -f docker-compose.prod.yml ps | grep -q "Up"; then
    echo "📦 Creating backup..."
    docker-compose -f docker-compose.prod.yml exec surrealdb \
        surrealdb export --conn ws://localhost:8000 --user root --pass $SURREALDB_PASSWORD \
        --ns production --db darwin backup-$(date +%Y%m%d-%H%M%S).surql

    echo "⏹️ Stopping existing services..."
    docker-compose -f docker-compose.prod.yml down
fi

# Pull latest images
echo "📥 Pulling latest images..."
docker-compose -f docker-compose.prod.yml pull

# Start services
echo "🔄 Starting services..."
docker-compose -f docker-compose.prod.yml up -d

# Wait for services to be ready
echo "⏳ Waiting for services to be ready..."
sleep 30

# Health check
echo "🏥 Performing health checks..."
for i in {1..10}; do
    if curl -f http://localhost/health > /dev/null 2>&1; then
        echo "✅ Darwin is healthy!"
        break
    fi
    echo "⏳ Waiting for Darwin to be ready... ($i/10)"
    sleep 10
done

# Display status
echo "📊 Service Status:"
docker-compose -f docker-compose.prod.yml ps

echo "🎉 Deployment completed successfully!"
echo "🌐 Access Darwin at: https://darwin.yourdomain.com"
echo "📊 Monitor at: https://darwin.yourdomain.com:3000 (Grafana)"
```

### Multi-Server Docker Swarm

#### Initialize Docker Swarm

```bash
# On manager node
docker swarm init --advertise-addr $(hostname -I | awk '{print $1}')

# Get join token for workers
docker swarm join-token worker

# On worker nodes (run the output from previous command)
docker swarm join --token <token> <manager-ip>:2377
```

#### Docker Swarm Stack

```yaml
# docker-stack.yml
version: '3.8'

services:
  darwin-app:
    image: devqai/darwin:latest
    deploy:
      replicas: 3
      update_config:
        parallelism: 1
        delay: 10s
        failure_action: rollback
      restart_policy:
        condition: on-failure
        delay: 5s
        max_attempts: 3
      placement:
        constraints:
          - node.role == worker
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=production
    env_file:
      - .env
    volumes:
      - darwin_data:/app/data
    networks:
      - darwin_network
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  nginx:
    image: nginx:alpine
    deploy:
      replicas: 2
      placement:
        constraints:
          - node.role == manager
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
    networks:
      - darwin_network
    depends_on:
      - darwin-app

volumes:
  darwin_data:

networks:
  darwin_network:
    driver: overlay
    attachable: true
```

## ☸️ Kubernetes Deployment

### Namespace and RBAC

```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: darwin-production
  labels:
    name: darwin-production
    environment: production

---
apiVersion: v1
kind: ServiceAccount
metadata:
  name: darwin-sa
  namespace: darwin-production

---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  namespace: darwin-production
  name: darwin-role
rules:
- apiGroups: [""]
  resources: ["pods", "services", "configmaps", "secrets"]
  verbs: ["get", "list", "watch"]

---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: darwin-role-binding
  namespace: darwin-production
subjects:
- kind: ServiceAccount
  name: darwin-sa
  namespace: darwin-production
roleRef:
  kind: Role
  name: darwin-role
  apiGroup: rbac.authorization.k8s.io
```

### ConfigMaps and Secrets

```yaml
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: darwin-config
  namespace: darwin-production
data:
  ENVIRONMENT: "production"
  DEBUG: "false"
  LOG_LEVEL: "INFO"
  API_HOST: "0.0.0.0"
  API_PORT: "8000"
  PROMETHEUS_ENABLED: "true"
  PROMETHEUS_PORT: "9090"
  WORKERS: "4"
  MAX_REQUESTS: "1000"
  WORKER_TIMEOUT: "300"

---
apiVersion: v1
kind: Secret
metadata:
  name: darwin-secrets
  namespace: darwin-production
type: Opaque
data:
  SECRET_KEY: <base64-encoded-secret>
  JWT_SECRET_KEY: <base64-encoded-jwt-secret>
  SURREALDB_PASSWORD: <base64-encoded-db-password>
  REDIS_PASSWORD: <base64-encoded-redis-password>
  LOGFIRE_TOKEN: <base64-encoded-logfire-token>
  EMAIL_SMTP_PASSWORD: <base64-encoded-email-password>
```

### Persistent Volumes

```yaml
# k8s/persistent-volumes.yaml
apiVersion: v1
kind: PersistentVolume
metadata:
  name: darwin-data-pv
spec:
  capacity:
    storage: 100Gi
  volumeMode: Filesystem
  accessModes:
  - ReadWriteOnce
  persistentVolumeReclaimPolicy: Retain
  storageClassName: fast-ssd
  hostPath:
    path: /opt/darwin/data

---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: darwin-data-pvc
  namespace: darwin-production
spec:
  accessModes:
  - ReadWriteOnce
  resources:
    requests:
      storage: 100Gi
  storageClassName: fast-ssd

---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: surrealdb-data-pvc
  namespace: darwin-production
spec:
  accessModes:
  - ReadWriteOnce
  resources:
    requests:
      storage: 500Gi
  storageClassName: fast-ssd
```

### Deployments

```yaml
# k8s/darwin-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: darwin-app
  namespace: darwin-production
  labels:
    app: darwin
    component: api
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxUnavailable: 1
      maxSurge: 1
  selector:
    matchLabels:
      app: darwin
      component: api
  template:
    metadata:
      labels:
        app: darwin
        component: api
    spec:
      serviceAccountName: darwin-sa
      containers:
      - name: darwin
        image: devqai/darwin:latest
        ports:
        - containerPort: 8000
          name: http
        - containerPort: 9090
          name: metrics
        envFrom:
        - configMapRef:
            name: darwin-config
        - secretRef:
            name: darwin-secrets
        resources:
          requests:
            cpu: 500m
            memory: 1Gi
          limits:
            cpu: 2000m
            memory: 4Gi
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 3
        volumeMounts:
        - name: darwin-data
          mountPath: /app/data
        - name: logs
          mountPath: /app/logs
      volumes:
      - name: darwin-data
        persistentVolumeClaim:
          claimName: darwin-data-pvc
      - name: logs
        emptyDir: {}

---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: surrealdb
  namespace: darwin-production
  labels:
    app: surrealdb
spec:
  replicas: 1
  selector:
    matchLabels:
      app: surrealdb
  template:
    metadata:
      labels:
        app: surrealdb
    spec:
      containers:
      - name: surrealdb
        image: surrealdb/surrealdb:latest
        ports:
        - containerPort: 8000
        command:
        - start
        - --log
        - trace
        - --user
        - root
        - --pass
        - $(SURREALDB_PASSWORD)
        - file:///data/database.db
        env:
        - name: SURREALDB_PASSWORD
          valueFrom:
            secretKeyRef:
              name: darwin-secrets
              key: SURREALDB_PASSWORD
        resources:
          requests:
            cpu: 1000m
            memory: 2Gi
          limits:
            cpu: 4000m
            memory: 8Gi
        volumeMounts:
        - name: surrealdb-data
          mountPath: /data
      volumes:
      - name: surrealdb-data
        persistentVolumeClaim:
          claimName: surrealdb-data-pvc
```

### Services and Ingress

```yaml
# k8s/services.yaml
apiVersion: v1
kind: Service
metadata:
  name: darwin-service
  namespace: darwin-production
  labels:
    app: darwin
spec:
  selector:
    app: darwin
    component: api
  ports:
  - name: http
    port: 8000
    targetPort: 8000
  - name: metrics
    port: 9090
    targetPort: 9090
  type: ClusterIP

---
apiVersion: v1
kind: Service
metadata:
  name: surrealdb-service
  namespace: darwin-production
  labels:
    app: surrealdb
spec:
  selector:
    app: surrealdb
  ports:
  - name: http
    port: 8000
    targetPort: 8000
  type: ClusterIP

---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: darwin-ingress
  namespace: darwin-production
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/rate-limit: "100"
    nginx.ingress.kubernetes.io/rate-limit-window: "1m"
spec:
  tls:
  - hosts:
    - darwin.yourdomain.com
    secretName: darwin-tls
  rules:
  - host: darwin.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: darwin-service
            port:
              number: 8000
```

### Horizontal Pod Autoscaler

```yaml
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: darwin-hpa
  namespace: darwin-production
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: darwin-app
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
```

## 🏗️ Infrastructure as Code

### Terraform AWS Setup

```hcl
# terraform/main.tf
terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# VPC Configuration
module "vpc" {
  source = "terraform-aws-modules/vpc/aws"

  name = "${var.project_name}-vpc"
  cidr = "10.0.0.0/16"

  azs             = ["${var.aws_region}a", "${var.aws_region}b", "${var.aws_region}c"]
  private_subnets = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
  public_subnets  = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]

  enable_nat_gateway = true
  enable_vpn_gateway = true
  enable_dns_hostnames = true
  enable_dns_support = true

  tags = {
    Environment = var.environment
    Project     = var.project_name
  }
}

# EKS Cluster
module "eks" {
  source = "terraform-aws-modules/eks/aws"

  cluster_name    = "${var.project_name}-eks"
  cluster_version = "1.28"

  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets

  # EKS Managed Node Groups
  eks_managed_node_groups = {
    main = {
      min_size     = 3
      max_size     = 10
      desired_size = 3

      instance_types = ["t3.large"]
      capacity_type  = "ON_DEMAND"

      k8s_labels = {
        Environment = var.environment
        NodeGroup   = "main"
      }
    }
  }

  tags = {
    Environment = var.environment
    Project     = var.project_name
  }
}

# RDS for SurrealDB (if using managed database)
resource "aws_db_instance" "darwin_db" {
  identifier = "${var.project_name}-db"

  engine         = "postgres"
  engine_version = "15.4"
  instance_class = "db.t3.medium"

  allocated_storage     = 100
  max_allocated_storage = 1000
  storage_type         = "gp3"
  storage_encrypted    = true

  db_name  = "darwin"
  username = "darwin"
  password = var.db_password

  vpc_security_group_ids = [aws_security_group.db.id]
  db_subnet_group_name   = aws_db_subnet_group.main.name

  backup_retention_period = 7
  backup_window          = "03:00-04:00"
  maintenance_window     = "sun:04:00-sun:05:00"

  skip_final_snapshot = false
  final_snapshot_identifier = "${var.project_name}-db-final-snapshot"

  tags = {
    Environment = var.environment
    Project     = var.project_name
  }
}

# ElastiCache for Redis
resource "aws_elasticache_subnet_group" "main" {
  name       = "${var.project_name}-cache-subnet"
  subnet_ids = module.vpc.private_subnets
}

resource "aws_elasticache_replication_group" "darwin_redis" {
  replication_group_id         = "${var.project_name}-redis"
  description                  = "Redis cluster for Darwin"

  engine               = "redis"
  engine_version       = "7.0"
  node_type           = "cache.t3.medium"
  port                = 6379
  parameter_group_name = "default.redis7"

  num_cache_clusters = 3

  subnet_group_name  = aws_elasticache_subnet_group.main.name
  security_group_ids = [aws_security_group.redis.id]

  at_rest_encryption_enabled = true
  transit_encryption_enabled = true
  auth_token                 = var.redis_password

  tags = {
    Environment = var.environment
    Project     = var.project_name
  }
}

# Security Groups
resource "aws_security_group" "db" {
  name_prefix = "${var.project_name}-db-"
  vpc_id      = module.vpc.vpc_id

  ingress {
    from_port   = 5432
    to_port     = 5432
    protocol    = "tcp"
    cidr_blocks = [module.vpc.vpc_cidr_block]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "${var.project_name}-db-sg"
  }
}

resource "aws_security_group" "redis" {
  name_prefix = "${var.project_name}-redis-"
  vpc_id      = module.vpc.vpc_id

  ingress {
    from_port   = 6379
    to_port     = 6379
    protocol    = "tcp"
    cidr_blocks = [module.vpc.vpc_cidr_block]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "${var.project_name}-redis-sg"
  }
}

# S3 Bucket for Backups
resource "aws_s3_bucket" "backups" {
  bucket = "${var.project_name}-backups-${random_string.bucket_suffix.result}"
}

resource "random_string" "bucket_suffix" {
  length  = 8
  special = false
  upper   = false
}

resource "aws_s3_bucket_versioning" "backups" {
  bucket = aws_s3_bucket.backups.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_encryption" "backups" {
  bucket = aws_s3_bucket.backups.id

  server_side_encryption_configuration {
    rule {
      apply_server_side_encryption_by_default {
        sse_algorithm = "AES256"
      }
    }
  }
}

# Variables
variable "project_name" {
  description = "Name of the project"
  type        = string
  default     = "darwin"
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "production"
}

variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "db_password" {
  description = "Database password"
  type        = string
  sensitive   = true
}

variable "redis_password" {
  description = "Redis password"
  type        = string
  sensitive   = true
}

# Outputs
output "cluster_endpoint" {
  description = "Endpoint for EKS control plane"
  value       = module.eks.cluster_endpoint
}

output "cluster_security_group_id" {
  description = "Security group ID attached to the EKS cluster"
  value       = module.eks.cluster_security_group_id
}

output "region" {
  description = "AWS region"
  value       = var.aws_region
}

output "cluster_name" {
  description = "Kubernetes Cluster Name"
  value       = module.eks.cluster_name
}
```

### Terraform Deployment Script

```bash
#!/bin/bash
# terraform/deploy.sh

set -e

echo "🚀 Starting Terraform Deployment"

# Initialize Terraform
terraform init

# Plan deployment
terraform plan -var-file="production.tfvars" -out=tfplan

# Apply deployment
terraform apply tfplan

# Configure kubectl
aws eks update-kubeconfig --region $(terraform output -raw region) --name $(terraform output -raw cluster_name)

echo "✅ Infrastructure deployment completed!"
```

## ⚖️ Load Balancing & Scaling

### Application Load Balancer Configuration

```yaml
# k8s/alb-ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: darwin-alb
  namespace: darwin-production
  annotations:
    kubernetes.io/ingress.class: alb
    alb.ingress.kubernetes.io/scheme: internet-facing
    alb.ingress.kubernetes.io/target-type: ip
    alb.ingress.kubernetes.io/certificate-arn: arn:aws:acm:us-east-1:123456789:certificate/your-cert-arn
    alb.ingress.kubernetes.io/ssl-redirect: '443'
    alb.ingress.kubernetes.io/load-balancer-attributes: routing.http2.enabled=true,idle_timeout.timeout_seconds=60
    alb.ingress.kubernetes.io/healthcheck-path: /health
    alb.ingress.kubernetes.io/healthcheck-interval-seconds: '30'
    alb.ingress.kubernetes.io/healthcheck-timeout-seconds: '5'
    alb.ingress.kubernetes.io/healthy-threshold-count: '2'
    alb.ingress.kubernetes.io/unhealthy-threshold-count: '3'
spec:
  rules:
  - host: darwin.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: darwin-service
            port:
              number: 8000
```

### Vertical Pod Autoscaler

```yaml
# k8s/vpa.yaml
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: darwin-vpa
  namespace: darwin-production
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: darwin-app
  updatePolicy:
    updateMode: "Auto"
  resourcePolicy:
    containerPolicies:
    - containerName: darwin
      maxAllowed:
        cpu: 4000m
        memory: 8Gi
      minAllowed:
        cpu: 100m
        memory: 256Mi
```

## 💾 Database Setup

### SurrealDB Cluster Configuration

```yaml
# k8s/surrealdb-cluster.yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: surrealdb-cluster
  namespace: darwin-production
spec:
  serviceName: surrealdb-cluster
  replicas: 3
  selector:
    matchLabels:
      app: surrealdb
  template:
    metadata:
      labels:
        app: surrealdb
    spec:
      containers:
      - name: surrealdb
        image: surrealdb/surrealdb:latest
        ports:
        - containerPort: 8000
          name: http
        command:
        - start
        - --log
        - trace
        - --user
        - root
        - --pass
        - $(SURREALDB_PASSWORD)
        - tikv://tikv-cluster:2379
        env:
        - name: SURREALDB_PASSWORD
          valueFrom:
            secretKeyRef:
              name: darwin-secrets
              key: SURREALDB_PASSWORD
        resources:
          requests:
            cpu: 1000m
            memory: 2Gi
          limits:
            cpu: 2000m
            memory: 4Gi
        volumeMounts:
        - name: surrealdb-data
          mountPath: /data
  volumeClaimTemplates:
  - metadata:
      name: surrealdb-data
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 100Gi
```

### Database Backup Configuration

```bash
#!/bin/bash
# scripts/backup-database.sh

set -e

BACKUP_DIR="/opt/darwin/backups"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
BACKUP_FILE="darwin-backup-${TIMESTAMP}.surql"
S3_BUCKET="darwin-backups-prod"

echo "🔄 Starting database backup..."

# Create backup
kubectl exec -n darwin-production surrealdb-cluster-0 -- \
    surrealdb export \
    --conn ws://localhost:8000 \
    --user root \
    --pass $SURREALDB_PASSWORD \
    --ns production \
    --db darwin \
    /tmp/$BACKUP_FILE

# Copy backup from pod
kubectl cp darwin-production/surrealdb-cluster-0:/tmp/$BACKUP_FILE $BACKUP_DIR/$BACKUP_FILE

# Upload to S3
aws s3 cp $BACKUP_DIR/$BACKUP_FILE s3://$S3_BUCKET/

# Cleanup old backups (keep last 30 days)
find $BACKUP_DIR -name "darwin-backup-*.surql" -mtime +30 -delete

echo "✅ Backup completed: $BACKUP_FILE"
```

## 📊 Monitoring & Observability

### Prometheus Configuration

```yaml
# k8s/prometheus.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-config
  namespace: darwin-production
data:
  prometheus.yml: |
    global:
      scrape_interval: 15s
      evaluation_interval: 15s

    rule_files:
      - "/etc/prometheus/darwin-rules.yml"

    alerting:
      alertmanagers:
        - static_configs:
            - targets:
              - alertmanager:9093

    scrape_configs:
      - job_name: 'darwin-api'
        static_configs:
          - targets: ['darwin-service:9090']
        metrics_path: /metrics
        scrape_interval: 15s

      - job_name: 'kubernetes-pods'
        kubernetes_sd_configs:
          - role: pod
        relabel_configs:
          - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
            action: keep
            regex: true

---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: prometheus
  namespace: darwin-production
spec:
  replicas: 1
  selector:
    matchLabels:
      app: prometheus
  template:
    metadata:
      labels:
        app: prometheus
    spec:
      containers:
      - name: prometheus
        image: prom/prometheus:latest
        ports:
        - containerPort: 9090
        volumeMounts:
        - name: config
          mountPath: /etc/prometheus
        - name: storage
          mountPath: /prometheus
        args:
          - '--config.file=/etc/prometheus/prometheus.yml'
          - '--storage.tsdb.path=/prometheus'
          - '--web.console.libraries=/etc/prometheus/console_libraries'
          - '--web.console.templates=/etc/prometheus/consoles'
          - '--storage.tsdb.retention.time=30d'
          - '--web.enable-lifecycle'
      volumes:
      - name: config
        configMap:
          name: prometheus-config
      - name: storage
        persistentVolumeClaim:
          claimName: prometheus-storage
```

### Grafana Dashboard

```json
{
  "dashboard": {
    "id": null,
    "title": "Darwin Monitoring Dashboard",
    "tags": ["darwin", "monitoring"],
    "timezone": "browser",
    "panels": [
      {
        "title": "API Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(darwin_api_request_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          }
        ]
      },
      {
        "title": "Active Optimizations",
        "type": "stat",
        "targets": [
          {
            "expr": "darwin_active_optimizations_total",
            "legendFormat": "Active"
          }
        ]
      },
      {
        "title": "Error Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(darwin_api_errors_total[5m])",
            "legendFormat": "Error Rate"
          }
        ]
      }
    ]
  }
}
```

## 🔒 Security Configuration

### Network Policies

```yaml
# k8s/network-policies.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: darwin-network-policy
  namespace: darwin-production
spec:
  podSelector:
    matchLabels:
      app: darwin
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    ports:
    - protocol: TCP
      port: 8000
  - from:
    - podSelector:
        matchLabels:
          app: prometheus
    ports:
    - protocol: TCP
      port: 9090
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: surrealdb
    ports:
    - protocol: TCP
      port: 8000
  - to:
    - podSelector:
        matchLabels:
          app: redis
    ports:
    - protocol: TCP
      port: 6379
  - to: []
    ports:
    - protocol: TCP
      port: 53
    - protocol: UDP
      port: 53
    - protocol: TCP
      port: 443
```

### Pod Security Policy

```yaml
# k8s/pod-security.yaml
apiVersion: policy/v1beta1
kind: PodSecurityPolicy
metadata:
  name: darwin-psp
spec:
  privileged: false
  allowPrivilegeEscalation: false
  requiredDropCapabilities:
    - ALL
  volumes:
    - 'configMap'
    - 'emptyDir'
    - 'projected'
    - 'secret'
    - 'downwardAPI'
    - 'persistentVolumeClaim'
  runAsUser:
    rule: 'MustRunAsNonRoot'
  seLinux:
    rule: 'RunAsAny'
  fsGroup:
    rule: 'RunAsAny'
```

## 🗄️ Backup & Recovery

### Automated Backup CronJob

```yaml
# k8s/backup-cronjob.yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: darwin-backup
  namespace: darwin-production
spec:
  schedule: "0 2 * * *"  # Daily at 2 AM
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: backup
            image: devqai/darwin-backup:latest
            command:
            - /bin/bash
            - -c
            - |
              set -e
              TIMESTAMP=$(date +%Y%m%d-%H%M%S)

              # Database backup
              surrealdb export \
                --conn ws://surrealdb-service:8000 \
                --user root \
                --pass $SURREALDB_PASSWORD \
                --ns production \
                --db darwin \
                backup-${TIMESTAMP}.surql

              # Upload to S3
              aws s3 cp backup-${TIMESTAMP}.surql s3://$BACKUP_S3_BUCKET/

              # Cleanup
              rm backup-${TIMESTAMP}.surql
            env:
            - name: SURREALDB_PASSWORD
              valueFrom:
                secretKeyRef:
                  name: darwin-secrets
                  key: SURREALDB_PASSWORD
            - name: BACKUP_S3_BUCKET
              value: "darwin-backups-prod"
            - name: AWS_DEFAULT_REGION
              value: "us-east-1"
          restartPolicy: OnFailure
```

### Disaster Recovery Plan

```bash
#!/bin/bash
# scripts/disaster-recovery.sh

set -e

BACKUP_DATE=${1:-latest}
S3_BUCKET="darwin-backups-prod"
RECOVERY_NAMESPACE="darwin-recovery"

echo "🚨 Starting disaster recovery process..."

# Create recovery namespace
kubectl create namespace $RECOVERY_NAMESPACE || true

# Download latest backup
if [ "$BACKUP_DATE" = "latest" ]; then
    BACKUP_FILE=$(aws s3 ls s3://$S3_BUCKET/ --recursive | sort | tail -n 1 | awk '{print $4}')
else
    BACKUP_FILE="backup-${BACKUP_DATE}.surql"
fi

echo "📥 Downloading backup: $BACKUP_FILE"
aws s3 cp s3://$S3_BUCKET/$BACKUP_FILE ./recovery-backup.surql

# Deploy recovery environment
kubectl apply -f k8s/ -n $RECOVERY_NAMESPACE

# Wait for pods to be ready
kubectl wait --for=condition=ready pod -l app=surrealdb -n $RECOVERY_NAMESPACE --timeout=300s

# Restore database
echo "🔄 Restoring database..."
kubectl exec -n $RECOVERY_NAMESPACE surrealdb-0 -- \
    surrealdb import \
    --conn ws://localhost:8000 \
    --user root \
    --pass $SURREALDB_PASSWORD \
    --ns production \
    --db darwin \
    /tmp/recovery-backup.surql

echo "✅ Disaster recovery completed!"
echo "🌐 Recovery environment available at: recovery.yourdomain.com"
```

## ⚡ Performance Tuning

### JVM Tuning (if applicable)

```bash
# Environment variables for performance
export JAVA_OPTS="-Xmx4g -Xms2g -XX:+UseG1GC -XX:MaxGCPauseMillis=200"
export PYTHON_OPTS="-O"
export WORKERS=4
export WORKER_CONNECTIONS=1000
```

### Database Optimization

```sql
-- SurrealDB optimization queries
DEFINE INDEX optimization_id_idx ON optimizations FIELDS optimization_id;
DEFINE INDEX user_id_idx ON optimizations FIELDS user_id;
DEFINE INDEX created_at_idx ON optimizations FIELDS created_at;
DEFINE INDEX status_idx ON optimizations FIELDS status;

-- Performance monitoring
SELECT COUNT() FROM optimizations WHERE created_at > time::now() - 1h;
SELECT AVG(execution_time) FROM optimizations WHERE status = 'completed';
```

### Resource Limits Optimization

```yaml
# Optimized resource limits
resources:
  requests:
    cpu: 500m      # Start with 0.5 CPU
    memory: 1Gi    # Start with 1GB RAM
  limits:
    cpu: 2000m     # Max 2 CPUs
    memory: 4Gi    # Max 4GB RAM
```

## 🔧 Troubleshooting

### Common Issues and Solutions

#### Pod Crashes

```bash
# Check pod logs
kubectl logs -n darwin-production darwin-app-xxx --previous

# Check pod events
kubectl describe pod -n darwin-production darwin-app-xxx

# Check resource usage
kubectl top pod -n darwin-production
```

#### Database Connection Issues

```bash
# Test database connectivity
kubectl exec -n darwin-production surrealdb-0 -- \
    surrealdb sql --conn ws://localhost:8000 --user root --pass $PASSWORD \
    --ns production --db darwin --pretty \
    "SELECT * FROM $auth LIMIT 1;"

# Check database logs
kubectl logs -n darwin-production surrealdb-0
```

#### Performance Issues

```bash
# Check resource utilization
kubectl top nodes
kubectl top pods -n darwin-production

# Check metrics
curl http://darwin-service:9090/metrics | grep darwin

# Scale up if needed
kubectl scale deployment darwin-app --replicas=5 -n darwin-production
```

### Health Check Commands

```bash
#!/bin/bash
# scripts/health-check.sh

echo "🏥 Darwin Health Check"

# Check API health
curl -f http://darwin.yourdomain.com/health || echo "❌ API unhealthy"

# Check database
kubectl exec -n darwin-production surrealdb-0 -- \
    surrealdb health --conn ws://localhost:8000 || echo "❌ Database unhealthy"

# Check pods
kubectl get pods -n darwin-production

# Check ingress
kubectl get ingress -n darwin-production

echo "✅ Health check completed"
```

### Rollback Procedures

```bash
#!/bin/bash
# scripts/rollback.sh

ROLLBACK_REVISION=${1:-previous}

echo "🔄 Rolling back Darwin deployment..."

# Rollback deployment
kubectl rollout undo deployment/darwin-app -n darwin-production --to-revision=$ROLLBACK_REVISION

# Wait for rollback to complete
kubectl rollout status deployment/darwin-app -n darwin-production

# Verify health
sleep 30
curl -f http://darwin.yourdomain.com/health

echo "✅ Rollback completed successfully"
```

## 📝 Deployment Checklist

### Pre-Deployment

- [ ] Environment variables configured
- [ ] SSL certificates obtained
- [ ] DNS records configured
- [ ] Backup strategy implemented
- [ ] Monitoring configured
- [ ] Security policies reviewed
- [ ] Performance testing completed
- [ ] Disaster recovery plan tested

### Deployment

- [ ] Infrastructure provisioned
- [ ] Database cluster deployed
- [ ] Application deployed
- [ ] Load balancer configured
- [ ] Monitoring enabled
- [ ] Alerts configured
- [ ] Health checks passing
- [ ] SSL certificates valid

### Post-Deployment

- [ ] Application accessible
- [ ] Monitoring data flowing
- [ ] Backups running
- [ ] Performance baseline established
- [ ] Documentation updated
- [ ] Team notified
- [ ] Runbook reviewed
- [ ] Incident response tested

---

**Production Deployment Guide Complete** | For support, contact [ops@devq.ai](mailto:ops@devq.ai)
