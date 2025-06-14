# Darwin Genetic Algorithm Platform - Terraform Infrastructure
# Production-ready AWS infrastructure with comprehensive monitoring integration

terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.20"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.10"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.1"
    }
  }

  backend "s3" {
    bucket         = "devq-terraform-state"
    key            = "darwin/terraform.tfstate"
    region         = "us-west-2"
    encrypt        = true
    dynamodb_table = "terraform-locks"
  }
}

# Provider configurations
provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = "darwin-platform"
      Environment = var.environment
      ManagedBy   = "terraform"
      Owner       = "devq-ai"
      Repository  = "github.com/devq-ai/darwin"
    }
  }
}

provider "kubernetes" {
  host                   = module.eks.cluster_endpoint
  cluster_ca_certificate = base64decode(module.eks.cluster_certificate_authority_data)

  exec {
    api_version = "client.authentication.k8s.io/v1beta1"
    command     = "aws"
    args        = ["eks", "get-token", "--cluster-name", module.eks.cluster_name]
  }
}

provider "helm" {
  kubernetes {
    host                   = module.eks.cluster_endpoint
    cluster_ca_certificate = base64decode(module.eks.cluster_certificate_authority_data)

    exec {
      api_version = "client.authentication.k8s.io/v1beta1"
      command     = "aws"
      args        = ["eks", "get-token", "--cluster-name", module.eks.cluster_name]
    }
  }
}

# Local values
locals {
  name = "darwin-${var.environment}"
  azs  = slice(data.aws_availability_zones.available.names, 0, 3)

  # Monitoring configuration
  monitoring_config = {
    prometheus_retention = "30d"
    grafana_admin_user   = "admin"
    alertmanager_config = {
      slack_api_url    = var.slack_webhook_url
      email_from       = var.alert_email_from
      email_smarthost  = var.smtp_server
    }
  }

  # Darwin application configuration
  darwin_config = {
    image_tag           = var.darwin_image_tag
    replica_count       = var.environment == "production" ? 3 : 2
    cpu_request         = "250m"
    memory_request      = "512Mi"
    cpu_limit           = "1000m"
    memory_limit        = "2Gi"
    health_check_path   = "/health/detailed"
    metrics_path        = "/metrics"
    performance_path    = "/performance"
    tracing_path        = "/tracing"
  }

  tags = {
    Project     = "darwin-platform"
    Environment = var.environment
    Component   = "infrastructure"
  }
}

# Data sources
data "aws_availability_zones" "available" {
  filter {
    name   = "opt-in-status"
    values = ["opt-in-not-required"]
  }
}

data "aws_caller_identity" "current" {}

# Random password generation
resource "random_password" "rds_password" {
  length  = 32
  special = true
}

resource "random_password" "redis_password" {
  length  = 32
  special = false
}

resource "random_password" "grafana_admin_password" {
  length  = 16
  special = false
}

# VPC Module
module "vpc" {
  source = "terraform-aws-modules/vpc/aws"

  name = local.name
  cidr = var.vpc_cidr

  azs             = local.azs
  private_subnets = [for k, v in local.azs : cidrsubnet(var.vpc_cidr, 4, k)]
  public_subnets  = [for k, v in local.azs : cidrsubnet(var.vpc_cidr, 8, k + 48)]
  intra_subnets   = [for k, v in local.azs : cidrsubnet(var.vpc_cidr, 8, k + 52)]

  enable_nat_gateway     = true
  single_nat_gateway     = var.environment != "production"
  enable_vpn_gateway     = false
  enable_dns_hostnames   = true
  enable_dns_support     = true

  # VPC Flow Logs for monitoring
  enable_flow_log                      = true
  create_flow_log_cloudwatch_iam_role  = true
  create_flow_log_cloudwatch_log_group = true

  public_subnet_tags = {
    "kubernetes.io/role/elb" = 1
  }

  private_subnet_tags = {
    "kubernetes.io/role/internal-elb" = 1
  }

  tags = local.tags
}

# EKS Cluster
module "eks" {
  source = "terraform-aws-modules/eks/aws"

  cluster_name    = local.name
  cluster_version = var.kubernetes_version

  vpc_id                         = module.vpc.vpc_id
  subnet_ids                     = module.vpc.private_subnets
  cluster_endpoint_public_access = true

  # Cluster access entries
  access_entries = {
    admin = {
      kubernetes_groups = []
      principal_arn     = "arn:aws:iam::${data.aws_caller_identity.current.account_id}:root"

      policy_associations = {
        admin = {
          policy_arn = "arn:aws:eks::aws:cluster-access-policy/AmazonEKSClusterAdminPolicy"
          access_scope = {
            type = "cluster"
          }
        }
      }
    }
  }

  # EKS Managed Node Groups
  eks_managed_node_groups = {
    # General purpose nodes for Darwin API
    darwin_nodes = {
      name           = "darwin-nodes"
      instance_types = ["m5.large", "m5.xlarge"]
      capacity_type  = "ON_DEMAND"

      min_size     = 2
      max_size     = 10
      desired_size = var.environment == "production" ? 4 : 2

      # Node group configuration
      ami_type       = "AL2_x86_64"
      disk_size      = 50
      disk_type      = "gp3"

      labels = {
        Environment = var.environment
        NodeGroup   = "darwin-nodes"
        Workload    = "general"
      }

      taints = []

      tags = merge(local.tags, {
        Name = "${local.name}-darwin-nodes"
      })
    }

    # Monitoring nodes with higher memory for Prometheus/Grafana
    monitoring_nodes = {
      name           = "monitoring-nodes"
      instance_types = ["m5.xlarge", "m5.2xlarge"]
      capacity_type  = "ON_DEMAND"

      min_size     = 1
      max_size     = 3
      desired_size = var.environment == "production" ? 2 : 1

      ami_type       = "AL2_x86_64"
      disk_size      = 100
      disk_type      = "gp3"

      labels = {
        Environment = var.environment
        NodeGroup   = "monitoring-nodes"
        Workload    = "monitoring"
      }

      taints = [
        {
          key    = "monitoring"
          value  = "true"
          effect = "NO_SCHEDULE"
        }
      ]

      tags = merge(local.tags, {
        Name = "${local.name}-monitoring-nodes"
      })
    }

    # Spot instances for batch processing
    batch_nodes = {
      name           = "batch-nodes"
      instance_types = ["m5.large", "m5.xlarge", "m4.large", "m4.xlarge"]
      capacity_type  = "SPOT"

      min_size     = 0
      max_size     = 20
      desired_size = 2

      ami_type       = "AL2_x86_64"
      disk_size      = 50
      disk_type      = "gp3"

      labels = {
        Environment = var.environment
        NodeGroup   = "batch-nodes"
        Workload    = "batch"
      }

      taints = [
        {
          key    = "batch"
          value  = "true"
          effect = "NO_SCHEDULE"
        }
      ]

      tags = merge(local.tags, {
        Name = "${local.name}-batch-nodes"
      })
    }
  }

  # Cluster addons
  cluster_addons = {
    coredns = {
      most_recent = true
    }
    kube-proxy = {
      most_recent = true
    }
    vpc-cni = {
      most_recent = true
    }
    aws-ebs-csi-driver = {
      most_recent = true
    }
  }

  tags = local.tags
}

# RDS Database for SurrealDB data persistence
resource "aws_db_subnet_group" "darwin" {
  name       = "${local.name}-db-subnet-group"
  subnet_ids = module.vpc.intra_subnets

  tags = merge(local.tags, {
    Name = "${local.name}-db-subnet-group"
  })
}

resource "aws_security_group" "rds" {
  name_prefix = "${local.name}-rds-"
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

  tags = merge(local.tags, {
    Name = "${local.name}-rds-sg"
  })
}

resource "aws_db_instance" "darwin" {
  identifier = "${local.name}-postgres"

  engine         = "postgres"
  engine_version = "15.4"
  instance_class = var.environment == "production" ? "db.t3.medium" : "db.t3.micro"

  allocated_storage     = var.environment == "production" ? 100 : 20
  max_allocated_storage = var.environment == "production" ? 1000 : 100
  storage_type         = "gp3"
  storage_encrypted    = true

  db_name  = "darwin"
  username = "darwin_admin"
  password = random_password.rds_password.result

  vpc_security_group_ids = [aws_security_group.rds.id]
  db_subnet_group_name   = aws_db_subnet_group.darwin.name

  backup_retention_period = var.environment == "production" ? 30 : 7
  backup_window          = "03:00-04:00"
  maintenance_window     = "Mon:04:00-Mon:05:00"

  skip_final_snapshot = var.environment != "production"
  deletion_protection = var.environment == "production"

  performance_insights_enabled = var.environment == "production"
  monitoring_interval         = var.environment == "production" ? 60 : 0

  tags = merge(local.tags, {
    Name = "${local.name}-postgres"
  })
}

# ElastiCache Redis for caching and session storage
resource "aws_elasticache_subnet_group" "darwin" {
  name       = "${local.name}-cache-subnet"
  subnet_ids = module.vpc.intra_subnets

  tags = merge(local.tags, {
    Name = "${local.name}-cache-subnet"
  })
}

resource "aws_security_group" "redis" {
  name_prefix = "${local.name}-redis-"
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

  tags = merge(local.tags, {
    Name = "${local.name}-redis-sg"
  })
}

resource "aws_elasticache_replication_group" "darwin" {
  replication_group_id       = "${local.name}-redis"
  description                = "Redis cluster for Darwin platform"

  node_type            = var.environment == "production" ? "cache.t3.medium" : "cache.t3.micro"
  port                 = 6379
  parameter_group_name = "default.redis7"

  num_cache_clusters = var.environment == "production" ? 3 : 1

  engine_version = "7.0"

  subnet_group_name  = aws_elasticache_subnet_group.darwin.name
  security_group_ids = [aws_security_group.redis.id]

  at_rest_encryption_enabled = true
  transit_encryption_enabled = true
  auth_token                 = random_password.redis_password.result

  snapshot_retention_limit = var.environment == "production" ? 7 : 1
  snapshot_window          = "03:00-05:00"

  tags = merge(local.tags, {
    Name = "${local.name}-redis"
  })
}

# S3 Buckets for data storage and backups
resource "aws_s3_bucket" "darwin_data" {
  bucket = "${local.name}-data-${random_id.bucket_suffix.hex}"

  tags = merge(local.tags, {
    Name = "${local.name}-data"
  })
}

resource "aws_s3_bucket" "darwin_backups" {
  bucket = "${local.name}-backups-${random_id.bucket_suffix.hex}"

  tags = merge(local.tags, {
    Name = "${local.name}-backups"
  })
}

resource "aws_s3_bucket" "darwin_monitoring" {
  bucket = "${local.name}-monitoring-${random_id.bucket_suffix.hex}"

  tags = merge(local.tags, {
    Name = "${local.name}-monitoring"
  })
}

resource "random_id" "bucket_suffix" {
  byte_length = 4
}

# S3 bucket configurations
resource "aws_s3_bucket_versioning" "darwin_data" {
  bucket = aws_s3_bucket.darwin_data.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_versioning" "darwin_backups" {
  bucket = aws_s3_bucket.darwin_backups.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "darwin_data" {
  bucket = aws_s3_bucket.darwin_data.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "darwin_backups" {
  bucket = aws_s3_bucket.darwin_backups.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# CloudWatch Log Groups for monitoring
resource "aws_cloudwatch_log_group" "darwin_api" {
  name              = "/aws/eks/${local.name}/darwin-api"
  retention_in_days = var.environment == "production" ? 30 : 7

  tags = merge(local.tags, {
    Name = "${local.name}-api-logs"
  })
}

resource "aws_cloudwatch_log_group" "darwin_monitoring" {
  name              = "/aws/eks/${local.name}/monitoring"
  retention_in_days = var.environment == "production" ? 90 : 14

  tags = merge(local.tags, {
    Name = "${local.name}-monitoring-logs"
  })
}

# Secrets Manager for sensitive configuration
resource "aws_secretsmanager_secret" "darwin_secrets" {
  name                    = "${local.name}-secrets"
  description             = "Darwin platform secrets"
  recovery_window_in_days = var.environment == "production" ? 30 : 0

  tags = merge(local.tags, {
    Name = "${local.name}-secrets"
  })
}

resource "aws_secretsmanager_secret_version" "darwin_secrets" {
  secret_id = aws_secretsmanager_secret.darwin_secrets.id
  secret_string = jsonencode({
    rds_password            = random_password.rds_password.result
    redis_password          = random_password.redis_password.result
    grafana_admin_password  = random_password.grafana_admin_password.result
    jwt_secret_key         = var.jwt_secret_key
    logfire_token          = var.logfire_token
    slack_webhook_url      = var.slack_webhook_url
    smtp_server            = var.smtp_server
    smtp_username          = var.smtp_username
    smtp_password          = var.smtp_password
  })

  lifecycle {
    ignore_changes = [secret_string]
  }
}

# IAM roles and policies for EKS workloads
resource "aws_iam_role" "darwin_api_role" {
  name = "${local.name}-api-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRoleWithWebIdentity"
        Effect = "Allow"
        Principal = {
          Federated = module.eks.oidc_provider_arn
        }
        Condition = {
          StringEquals = {
            "${replace(module.eks.cluster_oidc_issuer_url, "https://", "")}:sub" = "system:serviceaccount:darwin-platform:darwin-api"
            "${replace(module.eks.cluster_oidc_issuer_url, "https://", "")}:aud" = "sts.amazonaws.com"
          }
        }
      }
    ]
  })

  tags = local.tags
}

resource "aws_iam_policy" "darwin_api_policy" {
  name        = "${local.name}-api-policy"
  description = "IAM policy for Darwin API service"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket"
        ]
        Resource = [
          aws_s3_bucket.darwin_data.arn,
          "${aws_s3_bucket.darwin_data.arn}/*",
          aws_s3_bucket.darwin_backups.arn,
          "${aws_s3_bucket.darwin_backups.arn}/*",
          aws_s3_bucket.darwin_monitoring.arn,
          "${aws_s3_bucket.darwin_monitoring.arn}/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "secretsmanager:GetSecretValue"
        ]
        Resource = [
          aws_secretsmanager_secret.darwin_secrets.arn
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents",
          "logs:DescribeLogStreams"
        ]
        Resource = [
          aws_cloudwatch_log_group.darwin_api.arn,
          "${aws_cloudwatch_log_group.darwin_api.arn}:*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "cloudwatch:PutMetricData",
          "cloudwatch:GetMetricStatistics",
          "cloudwatch:ListMetrics"
        ]
        Resource = "*"
        Condition = {
          StringEquals = {
            "cloudwatch:namespace" = ["Darwin/API", "Darwin/Performance", "Darwin/Health"]
          }
        }
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "darwin_api_policy" {
  role       = aws_iam_role.darwin_api_role.name
  policy_arn = aws_iam_policy.darwin_api_policy.arn
}

# Application Load Balancer
resource "aws_security_group" "alb" {
  name_prefix = "${local.name}-alb-"
  vpc_id      = module.vpc.vpc_id

  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = merge(local.tags, {
    Name = "${local.name}-alb-sg"
  })
}

resource "aws_lb" "darwin" {
  name               = local.name
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  subnets            = module.vpc.public_subnets

  enable_deletion_protection = var.environment == "production"

  access_logs {
    bucket  = aws_s3_bucket.darwin_monitoring.id
    prefix  = "alb-logs"
    enabled = true
  }

  tags = merge(local.tags, {
    Name = "${local.name}-alb"
  })
}

# Route53 hosted zone and records
resource "aws_route53_zone" "darwin" {
  count = var.create_route53_zone ? 1 : 0
  name  = var.domain_name

  tags = merge(local.tags, {
    Name = var.domain_name
  })
}

resource "aws_route53_record" "darwin_api" {
  count   = var.create_route53_zone ? 1 : 0
  zone_id = aws_route53_zone.darwin[0].zone_id
  name    = "api.${var.domain_name}"
  type    = "A"

  alias {
    name                   = aws_lb.darwin.dns_name
    zone_id                = aws_lb.darwin.zone_id
    evaluate_target_health = true
  }
}

resource "aws_route53_record" "darwin_dashboard" {
  count   = var.create_route53_zone ? 1 : 0
  zone_id = aws_route53_zone.darwin[0].zone_id
  name    = "dashboard.${var.domain_name}"
  type    = "A"

  alias {
    name                   = aws_lb.darwin.dns_name
    zone_id                = aws_lb.darwin.zone_id
    evaluate_target_health = true
  }
}

resource "aws_route53_record" "darwin_monitoring" {
  count   = var.create_route53_zone ? 1 : 0
  zone_id = aws_route53_zone.darwin[0].zone_id
  name    = "monitoring.${var.domain_name}"
  type    = "A"

  alias {
    name                   = aws_lb.darwin.dns_name
    zone_id                = aws_lb.darwin.zone_id
    evaluate_target_health = true
  }
}

# Helm releases for monitoring stack
resource "helm_release" "prometheus_stack" {
  name       = "prometheus-stack"
  repository = "https://prometheus-community.github.io/helm-charts"
  chart      = "kube-prometheus-stack"
  version    = "51.2.0"
  namespace  = "monitoring"

  create_namespace = true

  values = [
    templatefile("${path.module}/helm-values/prometheus-stack.yaml", {
      environment                = var.environment
      retention_period          = local.monitoring_config.prometheus_retention
      grafana_admin_password     = random_password.grafana_admin_password.result
      slack_webhook_url          = var.slack_webhook_url
      alert_email                = var.alert_email_from
      smtp_server                = var.smtp_server
      storage_class              = "gp3"
      prometheus_storage_size    = var.environment == "production" ? "100Gi" : "50Gi"
      grafana_storage_size       = var.environment == "production" ? "20Gi" : "10Gi"
      alertmanager_storage_size  = var.environment == "production" ? "10Gi" : "5Gi"
    })
  ]

  depends_on = [module.eks]
}

resource "helm_release" "loki_stack" {
  name       = "loki-stack"
  repository = "https://grafana.github.io/helm-charts"
  chart      = "loki-stack"
  version    = "2.9.11"
  namespace  = "monitoring"

  create_namespace = true

  values = [
    templatefile("${path.module}/helm-values/loki-stack.yaml", {
      environment     = var.environment
      storage_class   = "gp3"
      storage_size    = var.environment == "production" ? "100Gi" : "50Gi"
    })
  ]

  depends_on = [helm_release.prometheus_stack]
}

# Outputs
output "cluster_endpoint" {
  description = "Endpoint for EKS control plane"
  value       = module.eks.cluster_endpoint
}

output "cluster_security_group_id" {
  description = "Security group ids attached to the cluster control plane"
  value       = module.eks.cluster_security_group_id
}

output "cluster_name" {
  description = "Kubernetes Cluster Name"
  value       = module.eks.cluster_name
}

output "rds_endpoint" {
  description = "RDS instance endpoint"
  value       = aws_db_instance.darwin.endpoint
  sensitive   = true
}

output "redis_endpoint" {
  description = "Redis cluster endpoint"
  value       = aws_elasticache_replication_group.darwin.primary_endpoint_address
  sensitive   = true
}

output "s3_bucket_data" {
  description = "S3 bucket for data storage"
  value       = aws_s3_bucket.darwin_data.id
}

output "s3_bucket_backups" {
  description = "S3 bucket for backups"
  value       = aws_s3_bucket.darwin_backups.id
}

output "s3_bucket_monitoring" {
  description = "S3 bucket for monitoring data"
  value       = aws_s3_bucket.darwin_monitoring.id
}

output "load_balancer_dns" {
  description = "DNS name of the load balancer"
  value       = aws_lb.darwin.dns_name
}

output "secrets_manager_arn" {
  description = "ARN of the secrets manager secret"
  value       = aws_secretsmanager_secret.darwin_secrets.arn
}

output "api_service_role_arn" {
  description = "ARN of the API service IAM role"
  value       = aws_iam_role.darwin_api_role.arn
}

output "monitoring_grafana_url" {
  description = "URL for Grafana monitoring dashboard"
  value       = var.create_route53_zone ? "https://monitoring.${var.domain_name}" : "http://${aws_lb.darwin.dns_name}/grafana"
}

output "api_url" {
  description = "URL for Darwin API"
  value       = var.create_route53_zone ? "https://api.${var.domain_name}" : "http://${aws_lb.darwin.dns_name}/api"
}

output "dashboard_url" {
  description = "URL for Darwin Dashboard"
  value       = var.create_route53_zone ? "https://dashboard.${var.domain_name}" : "http://${aws_lb.darwin.dns_name}/dashboard"
}
