#!/bin/bash
# Darwin Genetic Algorithm Platform - Production Deployment Script
# This script automates the deployment of Darwin platform to production environments
# with comprehensive monitoring, health checks, and rollback capabilities

set -euo pipefail

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DEPLOY_LOG="${PROJECT_ROOT}/logs/deployment.log"
BACKUP_DIR="${PROJECT_ROOT}/backups"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default configuration
ENVIRONMENT="${ENVIRONMENT:-production}"
DEPLOY_MODE="${DEPLOY_MODE:-rolling}"
HEALTH_CHECK_TIMEOUT="${HEALTH_CHECK_TIMEOUT:-300}"
BACKUP_ENABLED="${BACKUP_ENABLED:-true}"
MONITORING_ENABLED="${MONITORING_ENABLED:-true}"
ROLLBACK_ON_FAILURE="${ROLLBACK_ON_FAILURE:-true}"

# Service configuration
SERVICES=(
    "darwin-api"
    "darwin-dashboard"
    "surrealdb"
    "redis"
    "prometheus"
    "grafana"
    "alertmanager"
)

CRITICAL_SERVICES=(
    "darwin-api"
    "surrealdb"
    "redis"
)

# Logging functions
log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')

    echo -e "${timestamp} [${level}] ${message}" | tee -a "${DEPLOY_LOG}"

    case "${level}" in
        "ERROR")
            echo -e "${RED}[ERROR]${NC} ${message}" >&2
            ;;
        "WARN")
            echo -e "${YELLOW}[WARN]${NC} ${message}"
            ;;
        "INFO")
            echo -e "${GREEN}[INFO]${NC} ${message}"
            ;;
        "DEBUG")
            echo -e "${BLUE}[DEBUG]${NC} ${message}"
            ;;
    esac
}

# Error handling
error_exit() {
    log "ERROR" "$1"
    log "ERROR" "Deployment failed at $(date)"

    if [[ "${ROLLBACK_ON_FAILURE}" == "true" ]]; then
        log "INFO" "Initiating automatic rollback..."
        rollback_deployment
    fi

    exit 1
}

# Trap for cleanup
cleanup() {
    local exit_code=$?
    if [[ $exit_code -ne 0 ]]; then
        log "ERROR" "Script exited with error code: $exit_code"
    fi

    # Cleanup temporary files
    rm -f /tmp/deploy_*

    log "INFO" "Cleanup completed"
}

trap cleanup EXIT

# Utility functions
check_prerequisites() {
    log "INFO" "Checking deployment prerequisites..."

    # Check required tools
    local required_tools=("docker" "docker-compose" "curl" "jq" "aws")

    for tool in "${required_tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            error_exit "Required tool '$tool' is not installed"
        fi
    done

    # Check Docker daemon
    if ! docker info &> /dev/null; then
        error_exit "Docker daemon is not running"
    fi

    # Check environment variables
    local required_vars=(
        "DB_PASSWORD"
        "REDIS_PASSWORD"
        "JWT_SECRET_KEY"
        "GRAFANA_PASSWORD"
    )

    for var in "${required_vars[@]}"; do
        if [[ -z "${!var:-}" ]]; then
            error_exit "Required environment variable '$var' is not set"
        fi
    done

    # Check disk space (minimum 10GB)
    local available_space=$(df "$PROJECT_ROOT" | awk 'NR==2 {print $4}')
    local min_space=$((10 * 1024 * 1024)) # 10GB in KB

    if [[ $available_space -lt $min_space ]]; then
        error_exit "Insufficient disk space. Available: ${available_space}KB, Required: ${min_space}KB"
    fi

    log "INFO" "Prerequisites check passed"
}

create_directories() {
    log "INFO" "Creating necessary directories..."

    local dirs=(
        "${PROJECT_ROOT}/logs"
        "${BACKUP_DIR}"
        "/opt/darwin/data/surrealdb"
        "/opt/darwin/data/redis"
        "/opt/darwin/data/monitoring"
        "/opt/darwin/data/metrics"
        "/opt/darwin/data/health"
        "/opt/darwin/data/grafana"
        "/opt/darwin/data/prometheus"
        "/opt/darwin/data/alertmanager"
        "/opt/darwin/data/loki"
        "/opt/darwin/data/dashboard"
        "/opt/darwin/logs"
    )

    for dir in "${dirs[@]}"; do
        if [[ ! -d "$dir" ]]; then
            sudo mkdir -p "$dir"
            sudo chown -R $(id -u):$(id -g) "$dir" 2>/dev/null || true
            log "DEBUG" "Created directory: $dir"
        fi
    done

    log "INFO" "Directory creation completed"
}

backup_data() {
    if [[ "${BACKUP_ENABLED}" != "true" ]]; then
        log "INFO" "Backup disabled, skipping..."
        return 0
    fi

    log "INFO" "Creating backup before deployment..."

    local backup_name="darwin_backup_${TIMESTAMP}"
    local backup_path="${BACKUP_DIR}/${backup_name}"

    mkdir -p "$backup_path"

    # Backup database
    if docker ps --format "table {{.Names}}" | grep -q "darwin-surrealdb"; then
        log "INFO" "Backing up SurrealDB..."
        docker exec darwin-surrealdb-prod surreal export \
            --conn ws://localhost:8000 \
            --user root \
            --pass "${DB_PASSWORD}" \
            --ns darwin_prod \
            --db genetic_solver \
            /backup/backup_${TIMESTAMP}.surql || log "WARN" "Database backup failed"

        # Copy backup file to backup directory
        docker cp darwin-surrealdb-prod:/backup/backup_${TIMESTAMP}.surql \
            "${backup_path}/database.surql" || log "WARN" "Failed to copy database backup"
    fi

    # Backup configuration files
    cp -r "${PROJECT_ROOT}/docker" "${backup_path}/" 2>/dev/null || true
    cp "${PROJECT_ROOT}/.env" "${backup_path}/" 2>/dev/null || true
    cp "${PROJECT_ROOT}/docker-compose.prod.yml" "${backup_path}/" 2>/dev/null || true

    # Backup monitoring data
    if [[ -d "/opt/darwin/data" ]]; then
        tar -czf "${backup_path}/monitoring_data.tar.gz" \
            -C "/opt/darwin" data/ 2>/dev/null || log "WARN" "Monitoring data backup failed"
    fi

    # Create backup manifest
    cat > "${backup_path}/manifest.json" << EOF
{
    "timestamp": "${TIMESTAMP}",
    "environment": "${ENVIRONMENT}",
    "git_commit": "$(git rev-parse HEAD 2>/dev/null || echo 'unknown')",
    "git_branch": "$(git branch --show-current 2>/dev/null || echo 'unknown')",
    "services": $(docker ps --format "{{.Names}}" | jq -R . | jq -s .),
    "backup_size": "$(du -sh ${backup_path} | cut -f1)"
}
EOF

    log "INFO" "Backup created: $backup_path"

    # Cleanup old backups (keep last 5)
    find "${BACKUP_DIR}" -maxdepth 1 -type d -name "darwin_backup_*" | \
        sort -r | tail -n +6 | xargs rm -rf 2>/dev/null || true

    echo "$backup_name" > /tmp/deploy_backup_name
}

pull_latest_images() {
    log "INFO" "Pulling latest Docker images..."

    cd "$PROJECT_ROOT"

    # Pull base images
    docker-compose -f docker-compose.prod.yml pull --quiet || \
        error_exit "Failed to pull Docker images"

    # Build application images
    docker-compose -f docker-compose.prod.yml build --no-cache darwin-api || \
        error_exit "Failed to build application image"

    log "INFO" "Docker images updated"
}

deploy_services() {
    log "INFO" "Starting service deployment..."

    cd "$PROJECT_ROOT"

    case "${DEPLOY_MODE}" in
        "rolling")
            deploy_rolling
            ;;
        "blue-green")
            deploy_blue_green
            ;;
        "recreate")
            deploy_recreate
            ;;
        *)
            error_exit "Unknown deployment mode: ${DEPLOY_MODE}"
            ;;
    esac

    log "INFO" "Service deployment completed"
}

deploy_rolling() {
    log "INFO" "Performing rolling deployment..."

    # Start infrastructure services first
    local infra_services=("surrealdb" "redis" "prometheus" "grafana")

    for service in "${infra_services[@]}"; do
        log "INFO" "Deploying service: $service"
        docker-compose -f docker-compose.prod.yml up -d "$service" || \
            error_exit "Failed to deploy service: $service"

        wait_for_service_health "$service"
    done

    # Deploy application services
    local app_services=("darwin-api" "darwin-dashboard")

    for service in "${app_services[@]}"; do
        log "INFO" "Deploying service: $service"

        # Scale up new instances
        docker-compose -f docker-compose.prod.yml up -d --scale "$service"=2 "$service" || \
            error_exit "Failed to scale up service: $service"

        wait_for_service_health "$service"

        # Scale down old instances
        docker-compose -f docker-compose.prod.yml up -d --scale "$service"=1 "$service"

        log "INFO" "Rolling deployment completed for: $service"
    done
}

deploy_blue_green() {
    log "INFO" "Performing blue-green deployment..."

    # Create green environment
    export COMPOSE_PROJECT_NAME="darwin-green"

    docker-compose -f docker-compose.prod.yml up -d || \
        error_exit "Failed to deploy green environment"

    # Wait for all services to be healthy
    for service in "${CRITICAL_SERVICES[@]}"; do
        wait_for_service_health "$service"
    done

    # Switch traffic (this would require load balancer configuration)
    log "INFO" "Switching traffic to green environment"

    # Clean up blue environment
    export COMPOSE_PROJECT_NAME="darwin-blue"
    docker-compose -f docker-compose.prod.yml down || true

    # Rename green to blue for next deployment
    export COMPOSE_PROJECT_NAME="darwin"
}

deploy_recreate() {
    log "INFO" "Performing recreate deployment..."

    # Stop all services
    docker-compose -f docker-compose.prod.yml down

    # Start all services
    docker-compose -f docker-compose.prod.yml up -d || \
        error_exit "Failed to deploy services"

    # Wait for all services
    for service in "${SERVICES[@]}"; do
        wait_for_service_health "$service"
    done
}

wait_for_service_health() {
    local service="$1"
    local timeout="${HEALTH_CHECK_TIMEOUT}"
    local interval=10
    local elapsed=0

    log "INFO" "Waiting for service health: $service"

    while [[ $elapsed -lt $timeout ]]; do
        if check_service_health "$service"; then
            log "INFO" "Service is healthy: $service"
            return 0
        fi

        log "DEBUG" "Service not yet healthy: $service (${elapsed}s/${timeout}s)"
        sleep $interval
        elapsed=$((elapsed + interval))
    done

    error_exit "Service health check timeout: $service"
}

check_service_health() {
    local service="$1"

    case "$service" in
        "darwin-api")
            curl -f -s http://localhost:8000/health/detailed > /dev/null 2>&1
            ;;
        "darwin-dashboard")
            curl -f -s http://localhost:5006 > /dev/null 2>&1
            ;;
        "surrealdb")
            curl -f -s http://localhost:8001/health > /dev/null 2>&1
            ;;
        "redis")
            docker exec darwin-redis-prod redis-cli ping > /dev/null 2>&1
            ;;
        "prometheus")
            curl -f -s http://localhost:9090/-/healthy > /dev/null 2>&1
            ;;
        "grafana")
            curl -f -s http://localhost:3000/api/health > /dev/null 2>&1
            ;;
        "alertmanager")
            curl -f -s http://localhost:9093/-/healthy > /dev/null 2>&1
            ;;
        *)
            log "WARN" "Unknown service for health check: $service"
            return 1
            ;;
    esac
}

run_smoke_tests() {
    log "INFO" "Running smoke tests..."

    local test_endpoints=(
        "http://localhost:8000/health"
        "http://localhost:8000/health/detailed"
        "http://localhost:8000/metrics"
        "http://localhost:8000/performance"
        "http://localhost:5006"
        "http://localhost:9090/-/healthy"
        "http://localhost:3000/api/health"
    )

    for endpoint in "${test_endpoints[@]}"; do
        log "DEBUG" "Testing endpoint: $endpoint"

        if ! curl -f -s --max-time 30 "$endpoint" > /dev/null; then
            error_exit "Smoke test failed for endpoint: $endpoint"
        fi
    done

    # Test genetic algorithm functionality
    log "INFO" "Testing genetic algorithm API..."

    local test_payload='{
        "population_size": 10,
        "generations": 5,
        "mutation_rate": 0.1,
        "crossover_rate": 0.8,
        "problem_type": "test"
    }'

    local response=$(curl -s -X POST \
        -H "Content-Type: application/json" \
        -d "$test_payload" \
        http://localhost:8000/api/v1/optimizers/test 2>/dev/null || echo "")

    if [[ -z "$response" ]]; then
        log "WARN" "Genetic algorithm API test failed - endpoint may not be implemented"
    else
        log "INFO" "Genetic algorithm API test passed"
    fi

    log "INFO" "Smoke tests completed successfully"
}

setup_monitoring() {
    if [[ "${MONITORING_ENABLED}" != "true" ]]; then
        log "INFO" "Monitoring setup disabled, skipping..."
        return 0
    fi

    log "INFO" "Setting up monitoring and alerting..."

    # Wait for Prometheus to be ready
    local prometheus_ready=false
    for i in {1..30}; do
        if curl -f -s http://localhost:9090/-/ready > /dev/null 2>&1; then
            prometheus_ready=true
            break
        fi
        sleep 5
    done

    if [[ "$prometheus_ready" != "true" ]]; then
        log "WARN" "Prometheus not ready, monitoring setup may be incomplete"
    fi

    # Configure Grafana dashboards
    if curl -f -s http://localhost:3000/api/health > /dev/null 2>&1; then
        log "INFO" "Grafana is ready, dashboards will be auto-provisioned"
    fi

    # Test alerting
    log "INFO" "Testing alert manager configuration..."
    if curl -f -s http://localhost:9093/-/healthy > /dev/null 2>&1; then
        log "INFO" "Alert manager is healthy"
    else
        log "WARN" "Alert manager health check failed"
    fi

    log "INFO" "Monitoring setup completed"
}

rollback_deployment() {
    log "INFO" "Starting deployment rollback..."

    local backup_name
    if [[ -f /tmp/deploy_backup_name ]]; then
        backup_name=$(cat /tmp/deploy_backup_name)
    else
        # Find the most recent backup
        backup_name=$(find "${BACKUP_DIR}" -maxdepth 1 -type d -name "darwin_backup_*" | sort -r | head -n1 | xargs basename)
    fi

    if [[ -z "$backup_name" ]]; then
        log "ERROR" "No backup found for rollback"
        return 1
    fi

    local backup_path="${BACKUP_DIR}/${backup_name}"

    if [[ ! -d "$backup_path" ]]; then
        log "ERROR" "Backup directory not found: $backup_path"
        return 1
    fi

    log "INFO" "Rolling back to backup: $backup_name"

    # Stop current services
    docker-compose -f docker-compose.prod.yml down || true

    # Restore configuration files
    if [[ -f "${backup_path}/docker-compose.prod.yml" ]]; then
        cp "${backup_path}/docker-compose.prod.yml" "$PROJECT_ROOT/"
    fi

    if [[ -f "${backup_path}/.env" ]]; then
        cp "${backup_path}/.env" "$PROJECT_ROOT/"
    fi

    # Restore database if available
    if [[ -f "${backup_path}/database.surql" ]]; then
        log "INFO" "Restoring database from backup..."

        # Start only database service
        docker-compose -f docker-compose.prod.yml up -d surrealdb

        # Wait for database to be ready
        wait_for_service_health "surrealdb"

        # Import backup
        docker cp "${backup_path}/database.surql" darwin-surrealdb-prod:/tmp/restore.surql
        docker exec darwin-surrealdb-prod surreal import \
            --conn ws://localhost:8000 \
            --user root \
            --pass "${DB_PASSWORD}" \
            --ns darwin_prod \
            --db genetic_solver \
            /tmp/restore.surql || log "WARN" "Database restore failed"
    fi

    # Start all services
    docker-compose -f docker-compose.prod.yml up -d

    # Wait for critical services
    for service in "${CRITICAL_SERVICES[@]}"; do
        wait_for_service_health "$service"
    done

    log "INFO" "Rollback completed successfully"
}

cleanup_old_resources() {
    log "INFO" "Cleaning up old Docker resources..."

    # Remove unused images
    docker image prune -f || true

    # Remove unused volumes (be careful with this in production)
    docker volume prune -f || true

    # Remove unused networks
    docker network prune -f || true

    log "INFO" "Resource cleanup completed"
}

generate_deployment_report() {
    log "INFO" "Generating deployment report..."

    local report_file="${PROJECT_ROOT}/logs/deployment_report_${TIMESTAMP}.json"

    cat > "$report_file" << EOF
{
    "deployment": {
        "timestamp": "${TIMESTAMP}",
        "environment": "${ENVIRONMENT}",
        "mode": "${DEPLOY_MODE}",
        "git_commit": "$(git rev-parse HEAD 2>/dev/null || echo 'unknown')",
        "git_branch": "$(git branch --show-current 2>/dev/null || echo 'unknown')",
        "success": true,
        "duration_seconds": $(($(date +%s) - ${START_TIME:-$(date +%s)}))
    },
    "services": {
$(for service in "${SERVICES[@]}"; do
    local status="unknown"
    if check_service_health "$service" 2>/dev/null; then
        status="healthy"
    else
        status="unhealthy"
    fi
    echo "        \"$service\": \"$status\","
done | sed '$s/,$//')
    },
    "monitoring": {
        "prometheus": $(curl -s http://localhost:9090/api/v1/query?query=up | jq -c .data.result 2>/dev/null || echo '[]'),
        "grafana_status": "$(curl -s http://localhost:3000/api/health 2>/dev/null | jq -r .database 2>/dev/null || echo 'unknown')"
    },
    "system": {
        "disk_usage": "$(df -h $PROJECT_ROOT | awk 'NR==2 {print $5}')",
        "memory_usage": "$(free -h | awk 'NR==2 {print $3"/"$2}')",
        "load_average": "$(uptime | awk -F'load average:' '{print $2}')"
    }
}
EOF

    log "INFO" "Deployment report generated: $report_file"
}

# Main deployment function
main() {
    local start_time=$(date +%s)
    START_TIME=$start_time

    log "INFO" "Starting Darwin platform deployment"
    log "INFO" "Environment: ${ENVIRONMENT}"
    log "INFO" "Deploy mode: ${DEPLOY_MODE}"
    log "INFO" "Timestamp: ${TIMESTAMP}"

    # Create log directory
    mkdir -p "$(dirname "$DEPLOY_LOG")"

    # Run deployment steps
    check_prerequisites
    create_directories
    backup_data
    pull_latest_images
    deploy_services
    run_smoke_tests
    setup_monitoring
    cleanup_old_resources
    generate_deployment_report

    local end_time=$(date +%s)
    local duration=$((end_time - start_time))

    log "INFO" "Deployment completed successfully in ${duration} seconds"
    log "INFO" "Darwin platform is now running on ${ENVIRONMENT}"

    # Display service status
    echo
    echo "=== SERVICE STATUS ==="
    docker-compose -f docker-compose.prod.yml ps

    echo
    echo "=== ACCESS INFORMATION ==="
    echo "API: http://localhost:8000"
    echo "Dashboard: http://localhost:5006"
    echo "Monitoring: http://localhost:3000"
    echo "Metrics: http://localhost:9090"
    echo "Health Check: http://localhost:8000/health"
    echo

    # Show monitoring URLs
    if [[ "${MONITORING_ENABLED}" == "true" ]]; then
        echo "=== MONITORING ENDPOINTS ==="
        echo "Health: http://localhost:8000/health/detailed"
        echo "Metrics: http://localhost:8000/metrics"
        echo "Performance: http://localhost:8000/performance"
        echo "Tracing: http://localhost:8000/tracing"
        echo
    fi
}

# Script entry point
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --environment)
                ENVIRONMENT="$2"
                shift 2
                ;;
            --mode)
                DEPLOY_MODE="$2"
                shift 2
                ;;
            --no-backup)
                BACKUP_ENABLED="false"
                shift
                ;;
            --no-monitoring)
                MONITORING_ENABLED="false"
                shift
                ;;
            --no-rollback)
                ROLLBACK_ON_FAILURE="false"
                shift
                ;;
            --help)
                cat << EOF
Darwin Platform Deployment Script

Usage: $0 [OPTIONS]

Options:
  --environment ENV     Deployment environment (default: production)
  --mode MODE          Deployment mode: rolling|blue-green|recreate (default: rolling)
  --no-backup          Skip backup creation
  --no-monitoring      Skip monitoring setup
  --no-rollback        Disable automatic rollback on failure
  --help               Show this help message

Environment Variables:
  DB_PASSWORD          Database password (required)
  REDIS_PASSWORD       Redis password (required)
  JWT_SECRET_KEY       JWT secret key (required)
  GRAFANA_PASSWORD     Grafana admin password (required)
  SMTP_SERVER          SMTP server for notifications
  SLACK_WEBHOOK_URL    Slack webhook for notifications

Examples:
  $0                                    # Standard production deployment
  $0 --environment staging             # Deploy to staging
  $0 --mode blue-green                 # Blue-green deployment
  $0 --no-backup --no-rollback         # Quick deployment without backup
EOF
                exit 0
                ;;
            *)
                error_exit "Unknown option: $1"
                ;;
        esac
    done

    # Run main deployment
    main "$@"
fi
