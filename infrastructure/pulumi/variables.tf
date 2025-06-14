# Darwin Genetic Algorithm Platform - Terraform Variables
# Configuration variables for AWS infrastructure with comprehensive monitoring

variable "aws_region" {
  description = "AWS region for resources"
  type        = string
  default     = "us-west-2"
}

variable "environment" {
  description = "Environment name (development, staging, production)"
  type        = string
  default     = "production"

  validation {
    condition     = contains(["development", "staging", "production"], var.environment)
    error_message = "Environment must be one of: development, staging, production."
  }
}

variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "kubernetes_version" {
  description = "Kubernetes version for EKS cluster"
  type        = string
  default     = "1.28"
}

variable "domain_name" {
  description = "Domain name for the application"
  type        = string
  default     = "devq.ai"
}

variable "create_route53_zone" {
  description = "Whether to create Route53 hosted zone"
  type        = bool
  default     = false
}

# Darwin Application Configuration
variable "darwin_image_tag" {
  description = "Docker image tag for Darwin application"
  type        = string
  default     = "latest"
}

variable "darwin_replica_count" {
  description = "Number of Darwin API replicas"
  type        = number
  default     = 3

  validation {
    condition     = var.darwin_replica_count >= 1 && var.darwin_replica_count <= 10
    error_message = "Replica count must be between 1 and 10."
  }
}

# Security Configuration
variable "jwt_secret_key" {
  description = "JWT secret key for authentication"
  type        = string
  sensitive   = true

  validation {
    condition     = length(var.jwt_secret_key) >= 32
    error_message = "JWT secret key must be at least 32 characters long."
  }
}

# Monitoring and Observability
variable "logfire_token" {
  description = "Logfire token for observability"
  type        = string
  sensitive   = true
  default     = ""
}

variable "monitoring_enabled" {
  description = "Enable comprehensive monitoring stack"
  type        = bool
  default     = true
}

variable "performance_monitoring_enabled" {
  description = "Enable performance monitoring and profiling"
  type        = bool
  default     = true
}

variable "health_check_interval" {
  description = "Health check interval in seconds"
  type        = number
  default     = 30

  validation {
    condition     = var.health_check_interval >= 10 && var.health_check_interval <= 300
    error_message = "Health check interval must be between 10 and 300 seconds."
  }
}

variable "metrics_collection_interval" {
  description = "Metrics collection interval in seconds"
  type        = number
  default     = 15

  validation {
    condition     = var.metrics_collection_interval >= 5 && var.metrics_collection_interval <= 60
    error_message = "Metrics collection interval must be between 5 and 60 seconds."
  }
}

variable "trace_sample_rate" {
  description = "Distributed tracing sample rate (0.0 to 1.0)"
  type        = number
  default     = 0.1

  validation {
    condition     = var.trace_sample_rate >= 0.0 && var.trace_sample_rate <= 1.0
    error_message = "Trace sample rate must be between 0.0 and 1.0."
  }
}

variable "prometheus_retention_days" {
  description = "Prometheus metrics retention period in days"
  type        = number
  default     = 30

  validation {
    condition     = var.prometheus_retention_days >= 7 && var.prometheus_retention_days <= 365
    error_message = "Prometheus retention must be between 7 and 365 days."
  }
}

# Alert Configuration
variable "alert_cooldown_period" {
  description = "Alert cooldown period in seconds"
  type        = number
  default     = 300

  validation {
    condition     = var.alert_cooldown_period >= 60 && var.alert_cooldown_period <= 3600
    error_message = "Alert cooldown period must be between 60 and 3600 seconds."
  }
}

variable "alert_email_from" {
  description = "Email address for sending alerts"
  type        = string
  default     = "alerts@devq.ai"

  validation {
    condition     = can(regex("^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$", var.alert_email_from))
    error_message = "Alert email must be a valid email address."
  }
}

variable "alert_email_recipients" {
  description = "List of email addresses to receive alerts"
  type        = list(string)
  default     = ["admin@devq.ai"]

  validation {
    condition = alltrue([
      for email in var.alert_email_recipients : can(regex("^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$", email))
    ])
    error_message = "All alert recipient emails must be valid email addresses."
  }
}

# Notification Configuration
variable "slack_webhook_url" {
  description = "Slack webhook URL for notifications"
  type        = string
  sensitive   = true
  default     = ""
}

variable "slack_channel" {
  description = "Slack channel for notifications"
  type        = string
  default     = "#darwin-alerts"
}

# SMTP Configuration
variable "smtp_server" {
  description = "SMTP server for email notifications"
  type        = string
  default     = ""
}

variable "smtp_port" {
  description = "SMTP server port"
  type        = number
  default     = 587

  validation {
    condition     = var.smtp_port > 0 && var.smtp_port <= 65535
    error_message = "SMTP port must be between 1 and 65535."
  }
}

variable "smtp_username" {
  description = "SMTP username for authentication"
  type        = string
  sensitive   = true
  default     = ""
}

variable "smtp_password" {
  description = "SMTP password for authentication"
  type        = string
  sensitive   = true
  default     = ""
}

variable "smtp_use_tls" {
  description = "Use TLS for SMTP connection"
  type        = bool
  default     = true
}

# Database Configuration
variable "database_instance_class" {
  description = "RDS instance class"
  type        = string
  default     = "db.t3.medium"
}

variable "database_allocated_storage" {
  description = "Initial allocated storage for RDS in GB"
  type        = number
  default     = 100

  validation {
    condition     = var.database_allocated_storage >= 20 && var.database_allocated_storage <= 65536
    error_message = "Database allocated storage must be between 20 and 65536 GB."
  }
}

variable "database_max_allocated_storage" {
  description = "Maximum allocated storage for RDS in GB"
  type        = number
  default     = 1000

  validation {
    condition     = var.database_max_allocated_storage >= 20 && var.database_max_allocated_storage <= 65536
    error_message = "Database max allocated storage must be between 20 and 65536 GB."
  }
}

variable "database_backup_retention_period" {
  description = "Database backup retention period in days"
  type        = number
  default     = 30

  validation {
    condition     = var.database_backup_retention_period >= 0 && var.database_backup_retention_period <= 35
    error_message = "Database backup retention period must be between 0 and 35 days."
  }
}

variable "database_deletion_protection" {
  description = "Enable database deletion protection"
  type        = bool
  default     = true
}

# Redis Configuration
variable "redis_node_type" {
  description = "Redis node type"
  type        = string
  default     = "cache.t3.medium"
}

variable "redis_num_cache_clusters" {
  description = "Number of Redis cache clusters"
  type        = number
  default     = 3

  validation {
    condition     = var.redis_num_cache_clusters >= 1 && var.redis_num_cache_clusters <= 20
    error_message = "Redis cluster count must be between 1 and 20."
  }
}

variable "redis_snapshot_retention_limit" {
  description = "Redis snapshot retention limit in days"
  type        = number
  default     = 7

  validation {
    condition     = var.redis_snapshot_retention_limit >= 0 && var.redis_snapshot_retention_limit <= 35
    error_message = "Redis snapshot retention limit must be between 0 and 35 days."
  }
}

# EKS Configuration
variable "eks_node_group_instance_types" {
  description = "Instance types for EKS node groups"
  type        = list(string)
  default     = ["m5.large", "m5.xlarge"]
}

variable "eks_node_group_capacity_type" {
  description = "Capacity type for EKS node groups"
  type        = string
  default     = "ON_DEMAND"

  validation {
    condition     = contains(["ON_DEMAND", "SPOT"], var.eks_node_group_capacity_type)
    error_message = "EKS node group capacity type must be either ON_DEMAND or SPOT."
  }
}

variable "eks_node_group_min_size" {
  description = "Minimum size for EKS node groups"
  type        = number
  default     = 2

  validation {
    condition     = var.eks_node_group_min_size >= 1 && var.eks_node_group_min_size <= 100
    error_message = "EKS node group min size must be between 1 and 100."
  }
}

variable "eks_node_group_max_size" {
  description = "Maximum size for EKS node groups"
  type        = number
  default     = 10

  validation {
    condition     = var.eks_node_group_max_size >= 1 && var.eks_node_group_max_size <= 100
    error_message = "EKS node group max size must be between 1 and 100."
  }
}

variable "eks_node_group_desired_size" {
  description = "Desired size for EKS node groups"
  type        = number
  default     = 3

  validation {
    condition     = var.eks_node_group_desired_size >= 1 && var.eks_node_group_desired_size <= 100
    error_message = "EKS node group desired size must be between 1 and 100."
  }
}

# Storage Configuration
variable "storage_class" {
  description = "Kubernetes storage class for persistent volumes"
  type        = string
  default     = "gp3"

  validation {
    condition     = contains(["gp2", "gp3", "io1", "io2"], var.storage_class)
    error_message = "Storage class must be one of: gp2, gp3, io1, io2."
  }
}

variable "monitoring_storage_size" {
  description = "Storage size for monitoring data in Gi"
  type        = string
  default     = "100Gi"
}

variable "logs_storage_size" {
  description = "Storage size for logs in Gi"
  type        = string
  default     = "50Gi"
}

variable "metrics_storage_size" {
  description = "Storage size for metrics in Gi"
  type        = string
  default     = "200Gi"
}

# Security Configuration
variable "enable_encryption_at_rest" {
  description = "Enable encryption at rest for all storage"
  type        = bool
  default     = true
}

variable "enable_encryption_in_transit" {
  description = "Enable encryption in transit"
  type        = bool
  default     = true
}

variable "enable_network_policies" {
  description = "Enable Kubernetes network policies"
  type        = bool
  default     = true
}

variable "enable_pod_security_policies" {
  description = "Enable Kubernetes pod security policies"
  type        = bool
  default     = true
}

# Backup Configuration
variable "backup_enabled" {
  description = "Enable automated backups"
  type        = bool
  default     = true
}

variable "backup_retention_days" {
  description = "Backup retention period in days"
  type        = number
  default     = 30

  validation {
    condition     = var.backup_retention_days >= 1 && var.backup_retention_days <= 365
    error_message = "Backup retention days must be between 1 and 365."
  }
}

variable "backup_schedule" {
  description = "Backup schedule in cron format"
  type        = string
  default     = "0 2 * * *"  # Daily at 2 AM
}

# Cost Optimization
variable "enable_spot_instances" {
  description = "Enable spot instances for non-critical workloads"
  type        = bool
  default     = false
}

variable "enable_cluster_autoscaler" {
  description = "Enable cluster autoscaler"
  type        = bool
  default     = true
}

variable "enable_vertical_pod_autoscaler" {
  description = "Enable vertical pod autoscaler"
  type        = bool
  default     = true
}

# Compliance and Governance
variable "enable_cloudtrail" {
  description = "Enable AWS CloudTrail for audit logging"
  type        = bool
  default     = true
}

variable "enable_config" {
  description = "Enable AWS Config for compliance monitoring"
  type        = bool
  default     = true
}

variable "enable_guardduty" {
  description = "Enable AWS GuardDuty for threat detection"
  type        = bool
  default     = true
}

# Performance Configuration
variable "enable_performance_insights" {
  description = "Enable RDS Performance Insights"
  type        = bool
  default     = true
}

variable "performance_insights_retention_period" {
  description = "Performance Insights retention period in days"
  type        = number
  default     = 7

  validation {
    condition     = contains([7, 731], var.performance_insights_retention_period)
    error_message = "Performance Insights retention period must be either 7 or 731 days."
  }
}

# Monitoring Thresholds
variable "cpu_threshold_warning" {
  description = "CPU usage threshold for warnings (percentage)"
  type        = number
  default     = 70

  validation {
    condition     = var.cpu_threshold_warning >= 1 && var.cpu_threshold_warning <= 100
    error_message = "CPU threshold warning must be between 1 and 100 percent."
  }
}

variable "cpu_threshold_critical" {
  description = "CPU usage threshold for critical alerts (percentage)"
  type        = number
  default     = 90

  validation {
    condition     = var.cpu_threshold_critical >= 1 && var.cpu_threshold_critical <= 100
    error_message = "CPU threshold critical must be between 1 and 100 percent."
  }
}

variable "memory_threshold_warning" {
  description = "Memory usage threshold for warnings (percentage)"
  type        = number
  default     = 80

  validation {
    condition     = var.memory_threshold_warning >= 1 && var.memory_threshold_warning <= 100
    error_message = "Memory threshold warning must be between 1 and 100 percent."
  }
}

variable "memory_threshold_critical" {
  description = "Memory usage threshold for critical alerts (percentage)"
  type        = number
  default     = 95

  validation {
    condition     = var.memory_threshold_critical >= 1 && var.memory_threshold_critical <= 100
    error_message = "Memory threshold critical must be between 1 and 100 percent."
  }
}

variable "disk_threshold_warning" {
  description = "Disk usage threshold for warnings (percentage)"
  type        = number
  default     = 80

  validation {
    condition     = var.disk_threshold_warning >= 1 && var.disk_threshold_warning <= 100
    error_message = "Disk threshold warning must be between 1 and 100 percent."
  }
}

variable "disk_threshold_critical" {
  description = "Disk usage threshold for critical alerts (percentage)"
  type        = number
  default     = 90

  validation {
    condition     = var.disk_threshold_critical >= 1 && var.disk_threshold_critical <= 100
    error_message = "Disk threshold critical must be between 1 and 100 percent."
  }
}

# API Rate Limiting
variable "api_rate_limit_per_minute" {
  description = "API rate limit per minute per user"
  type        = number
  default     = 100

  validation {
    condition     = var.api_rate_limit_per_minute >= 1 && var.api_rate_limit_per_minute <= 10000
    error_message = "API rate limit must be between 1 and 10000 requests per minute."
  }
}

variable "api_burst_limit" {
  description = "API burst limit for short-term spikes"
  type        = number
  default     = 200

  validation {
    condition     = var.api_burst_limit >= 1 && var.api_burst_limit <= 20000
    error_message = "API burst limit must be between 1 and 20000 requests."
  }
}

# Genetic Algorithm Configuration
variable "max_optimization_concurrent_runs" {
  description = "Maximum number of concurrent genetic algorithm optimization runs"
  type        = number
  default     = 10

  validation {
    condition     = var.max_optimization_concurrent_runs >= 1 && var.max_optimization_concurrent_runs <= 100
    error_message = "Max concurrent optimization runs must be between 1 and 100."
  }
}

variable "optimization_timeout_minutes" {
  description = "Timeout for genetic algorithm optimization runs in minutes"
  type        = number
  default     = 60

  validation {
    condition     = var.optimization_timeout_minutes >= 1 && var.optimization_timeout_minutes <= 1440
    error_message = "Optimization timeout must be between 1 and 1440 minutes (24 hours)."
  }
}

# Tags
variable "additional_tags" {
  description = "Additional tags to apply to all resources"
  type        = map(string)
  default     = {}
}

# Feature Flags
variable "enable_canary_deployments" {
  description = "Enable canary deployment strategy"
  type        = bool
  default     = false
}

variable "enable_blue_green_deployments" {
  description = "Enable blue-green deployment strategy"
  type        = bool
  default     = false
}

variable "enable_chaos_engineering" {
  description = "Enable chaos engineering tools"
  type        = bool
  default     = false
}

variable "enable_load_testing" {
  description = "Enable automated load testing"
  type        = bool
  default     = false
}
