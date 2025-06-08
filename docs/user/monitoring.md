# Monitoring & Alerts Guide

This guide covers Darwin's comprehensive monitoring system, including health checks, metrics collection, alerting, performance monitoring, and observability features powered by Logfire integration.

## 📊 Overview

Darwin provides enterprise-grade monitoring capabilities that give you complete visibility into your optimization workloads, system health, and performance metrics. The monitoring system is built on Logfire and includes:

- **Health Monitoring**: Real-time system health checks
- **Metrics Collection**: Performance and usage metrics
- **Alert Management**: Intelligent alerting for anomalies
- **Distributed Tracing**: End-to-end request tracing
- **Performance Analytics**: Detailed performance insights
- **Resource Monitoring**: CPU, memory, and disk usage tracking

## 🏥 Health Monitoring

### System Health Checks

Darwin continuously monitors the health of all critical components:

```python
from darwin.monitoring import HealthChecker, HealthStatus

# Initialize health checker
health_checker = HealthChecker()

# Check overall system health
health_status = await health_checker.check_health()

print(f"Overall Status: {health_status.status}")
print(f"Healthy Components: {health_status.healthy_count}")
print(f"Unhealthy Components: {health_status.unhealthy_count}")
```

### Built-in Health Checks

#### Database Connectivity
```python
# SurrealDB health check
surreal_health = await health_checker.check_surrealdb()
print(f"SurrealDB Status: {surreal_health.status}")

# Redis health check
redis_health = await health_checker.check_redis()
print(f"Redis Status: {redis_health.status}")
```

#### API Endpoint Health
```python
# FastAPI health check
api_health = await health_checker.check_api_endpoints()
print(f"API Status: {api_health.status}")

# MCP Server health check
mcp_health = await health_checker.check_mcp_server()
print(f"MCP Server Status: {mcp_health.status}")
```

#### System Resources
```python
# System resource monitoring
system_health = await health_checker.check_system_resources()
print(f"CPU Usage: {system_health.details['cpu_percent']}%")
print(f"Memory Usage: {system_health.details['memory_percent']}%")
print(f"Disk Usage: {system_health.details['disk_percent']}%")
```

### Custom Health Checks

Register custom health checks for your specific requirements:

```python
async def custom_ml_model_health():
    """Custom health check for ML model availability"""
    try:
        # Check if your ML model is loaded and responsive
        model_response_time = await check_model_latency()
        if model_response_time > 1.0:  # 1 second threshold
            return HealthCheckResult(
                name="ml_model",
                status=HealthStatus.DEGRADED,
                message=f"Model response time {model_response_time:.2f}s exceeds threshold"
            )
        return HealthCheckResult(
            name="ml_model",
            status=HealthStatus.HEALTHY,
            message="ML model is responsive"
        )
    except Exception as e:
        return HealthCheckResult(
            name="ml_model",
            status=HealthStatus.UNHEALTHY,
            message=f"ML model check failed: {str(e)}"
        )

# Register custom health check
health_checker.register_check("ml_model", custom_ml_model_health)
```

### Health Check API

Access health information via REST API:

```bash
# Overall health status
curl http://localhost:8000/health

# Detailed health report
curl http://localhost:8000/health/detailed

# Specific component health
curl http://localhost:8000/health/database
curl http://localhost:8000/health/api
curl http://localhost:8000/health/system
```

**Response Example:**
```json
{
  "status": "healthy",
  "timestamp": "2025-01-27T10:30:00Z",
  "components": {
    "database": {
      "status": "healthy",
      "response_time": 0.05,
      "details": {
        "surrealdb": "connected",
        "redis": "connected"
      }
    },
    "api": {
      "status": "healthy",
      "response_time": 0.02,
      "details": {
        "endpoints_checked": 15,
        "all_responsive": true
      }
    },
    "system": {
      "status": "healthy",
      "details": {
        "cpu_percent": 25.4,
        "memory_percent": 68.2,
        "disk_percent": 45.1
      }
    }
  }
}
```

## 📈 Metrics Collection

### Performance Metrics

Darwin automatically collects key performance metrics:

```python
from darwin.monitoring import MetricsCollector

# Initialize metrics collector
metrics = MetricsCollector()

# View current metrics
current_metrics = await metrics.get_current_metrics()
print(f"Active Optimizations: {current_metrics['active_optimizations']}")
print(f"Total Evaluations/sec: {current_metrics['evaluations_per_second']}")
print(f"Average Response Time: {current_metrics['avg_response_time']:.3f}s")
```

### Optimization Metrics

Track genetic algorithm performance:

```python
# Optimization-specific metrics
optimization_metrics = await metrics.get_optimization_metrics()
print(f"Best Fitness Achieved: {optimization_metrics['best_fitness']}")
print(f"Convergence Rate: {optimization_metrics['convergence_rate']}")
print(f"Population Diversity: {optimization_metrics['diversity_index']}")
```

### Custom Metrics

Register and track custom metrics:

```python
# Counter metric
metrics.increment_counter("custom_events", tags={"event_type": "user_action"})

# Gauge metric
metrics.set_gauge("queue_size", 42, tags={"queue_name": "optimization_jobs"})

# Histogram metric
metrics.record_histogram("request_duration", 0.123, tags={"endpoint": "/optimize"})

# Timer context manager
with metrics.timer("database_query_time", tags={"table": "optimizations"}):
    result = await database.query("SELECT * FROM optimizations")
```

### Metrics Dashboard

Access metrics through the built-in dashboard:

```python
from darwin.dashboard import MetricsDashboard

# Launch metrics dashboard
dashboard = MetricsDashboard()
dashboard.serve(port=5007)  # Available at http://localhost:5007
```

## 🚨 Alert Management

### Alert Configuration

Configure alerts for various conditions:

```python
from darwin.monitoring import AlertManager, AlertRule, AlertCondition

# Initialize alert manager
alert_manager = AlertManager()

# High CPU usage alert
cpu_alert = AlertRule(
    name="high_cpu_usage",
    condition=AlertCondition.GREATER_THAN,
    threshold=80.0,
    metric="system.cpu.percent",
    duration_minutes=5,
    severity="warning",
    message="CPU usage above 80% for 5 minutes"
)

# Memory usage alert
memory_alert = AlertRule(
    name="high_memory_usage",
    condition=AlertCondition.GREATER_THAN,
    threshold=90.0,
    metric="system.memory.percent",
    duration_minutes=2,
    severity="critical",
    message="Memory usage above 90% for 2 minutes"
)

# Failed optimization alert
failed_opt_alert = AlertRule(
    name="optimization_failures",
    condition=AlertCondition.GREATER_THAN,
    threshold=5,
    metric="optimization.failures.count",
    duration_minutes=10,
    severity="error",
    message="More than 5 optimization failures in 10 minutes"
)

# Register alerts
alert_manager.add_rule(cpu_alert)
alert_manager.add_rule(memory_alert)
alert_manager.add_rule(failed_opt_alert)
```

### Alert Channels

Configure notification channels:

```python
# Email notifications
email_channel = EmailAlertChannel(
    smtp_server="smtp.gmail.com",
    smtp_port=587,
    username="alerts@yourdomain.com",
    password="your_app_password",
    recipients=["admin@yourdomain.com", "ops-team@yourdomain.com"]
)

# Slack notifications
slack_channel = SlackAlertChannel(
    webhook_url="https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK",
    channel="#darwin-alerts",
    username="Darwin Bot"
)

# Webhook notifications
webhook_channel = WebhookAlertChannel(
    url="https://your-monitoring-system.com/webhook",
    method="POST",
    headers={"Authorization": "Bearer your-token"}
)

# Register channels
alert_manager.add_channel("email", email_channel)
alert_manager.add_channel("slack", slack_channel)
alert_manager.add_channel("webhook", webhook_channel)
```

### Alert Policies

Define when and how alerts are sent:

```python
# Alert policy configuration
alert_policy = AlertPolicy(
    name="production_alerts",
    rules=[
        AlertPolicyRule(
            severity="critical",
            channels=["email", "slack"],
            immediate=True
        ),
        AlertPolicyRule(
            severity="warning",
            channels=["slack"],
            immediate=False,
            batch_interval_minutes=15
        ),
        AlertPolicyRule(
            severity="info",
            channels=["webhook"],
            immediate=False,
            batch_interval_minutes=60
        )
    ]
)

alert_manager.set_policy(alert_policy)
```

## 🔍 Distributed Tracing

### Automatic Tracing

Darwin automatically traces requests through the system:

```python
from darwin.monitoring import Tracer

# Tracer is automatically initialized with Logfire
# All HTTP requests, database queries, and optimization runs are traced
```

### Custom Tracing

Add custom spans for detailed tracing:

```python
import logfire

@logfire.instrument("optimize_portfolio")
async def optimize_portfolio(assets, constraints):
    """Custom traced function"""
    with logfire.span("data_preparation"):
        cleaned_data = prepare_data(assets)

    with logfire.span("algorithm_execution"):
        result = await run_optimization(cleaned_data, constraints)

    with logfire.span("result_processing"):
        processed_result = process_results(result)

    return processed_result
```

### Trace Analysis

Query and analyze traces:

```python
# Get recent traces
traces = await tracer.get_recent_traces(limit=100)

# Get traces for specific operation
optimization_traces = await tracer.get_traces(
    operation="run_optimization",
    start_time=datetime.now() - timedelta(hours=1)
)

# Get slow traces
slow_traces = await tracer.get_slow_traces(
    threshold_seconds=5.0,
    limit=10
)
```

## 📊 Performance Analytics

### Performance Insights

Get detailed performance analytics:

```python
from darwin.monitoring import PerformanceAnalyzer

analyzer = PerformanceAnalyzer()

# Get performance summary
perf_summary = await analyzer.get_performance_summary()
print(f"P95 Response Time: {perf_summary['p95_response_time']:.3f}s")
print(f"Throughput: {perf_summary['requests_per_second']:.1f} req/s")
print(f"Error Rate: {perf_summary['error_rate']:.2%}")

# Get optimization performance
opt_performance = await analyzer.get_optimization_performance()
print(f"Average Convergence Time: {opt_performance['avg_convergence_time']:.1f}s")
print(f"Success Rate: {opt_performance['success_rate']:.2%}")
```

### Performance Recommendations

Get AI-powered performance recommendations:

```python
# Get performance recommendations
recommendations = await analyzer.get_recommendations()

for rec in recommendations:
    print(f"Issue: {rec['issue']}")
    print(f"Impact: {rec['impact']}")
    print(f"Recommendation: {rec['recommendation']}")
    print(f"Priority: {rec['priority']}")
    print("---")
```

## 🖥️ Resource Monitoring

### System Resource Tracking

Monitor system resources in real-time:

```python
from darwin.monitoring import ResourceMonitor

monitor = ResourceMonitor()

# Get current resource usage
resources = await monitor.get_resource_usage()
print(f"CPU: {resources['cpu_percent']:.1f}%")
print(f"Memory: {resources['memory_percent']:.1f}%")
print(f"Disk: {resources['disk_percent']:.1f}%")
print(f"Network I/O: {resources['network_io']['bytes_sent']} bytes/s")

# Get resource history
history = await monitor.get_resource_history(hours=24)
```

### Resource Alerts

Set up resource-based alerts:

```python
# CPU usage alert
monitor.add_threshold_alert(
    metric="cpu_percent",
    threshold=85.0,
    duration_minutes=5,
    callback=lambda: send_alert("High CPU usage detected")
)

# Memory usage alert
monitor.add_threshold_alert(
    metric="memory_percent",
    threshold=90.0,
    duration_minutes=2,
    callback=lambda: send_alert("High memory usage detected")
)
```

## 🔧 Logfire Integration

### Logfire Configuration

Darwin integrates seamlessly with Logfire for observability:

```python
import logfire

# Logfire is automatically configured
# Configure additional settings in .env file:
# LOGFIRE_TOKEN=your_logfire_token
# LOGFIRE_PROJECT_NAME=darwin-production
```

### Custom Logging

Add structured logging with Logfire:

```python
import logfire

# Structured logging
logfire.info(
    "Optimization completed",
    optimization_id="opt_123",
    generation_count=150,
    best_fitness=0.001234,
    execution_time=45.2
)

# Error logging with context
try:
    result = await run_optimization(problem)
except Exception as e:
    logfire.error(
        "Optimization failed",
        optimization_id="opt_123",
        error_type=type(e).__name__,
        error_message=str(e),
        stack_trace=traceback.format_exc()
    )
```

### Logfire Dashboard

Access Logfire features:

1. **Real-time Logs**: View live log streams
2. **Trace Analysis**: Analyze distributed traces
3. **Performance Metrics**: Monitor performance indicators
4. **Error Tracking**: Track and analyze errors
5. **Custom Dashboards**: Create custom monitoring dashboards

## 📱 Monitoring Dashboard

### Built-in Dashboard

Darwin includes a comprehensive monitoring dashboard:

```python
from darwin.dashboard import MonitoringDashboard

# Launch monitoring dashboard
dashboard = MonitoringDashboard()
dashboard.serve(port=5008)  # Available at http://localhost:5008
```

### Dashboard Features

- **System Overview**: Health status, key metrics, alerts
- **Performance Metrics**: Response times, throughput, error rates
- **Resource Usage**: CPU, memory, disk, network usage
- **Optimization Analytics**: Success rates, convergence times
- **Alert History**: Recent alerts and their resolution
- **Trace Viewer**: Distributed trace visualization

### Custom Dashboard Panels

Create custom monitoring panels:

```python
from darwin.dashboard import CustomPanel, MetricWidget

# Create custom panel
custom_panel = CustomPanel("My Optimizations")

# Add metrics widgets
custom_panel.add_widget(MetricWidget(
    title="Active Optimizations",
    metric="optimization.active.count",
    chart_type="gauge"
))

custom_panel.add_widget(MetricWidget(
    title="Success Rate",
    metric="optimization.success.rate",
    chart_type="line",
    time_range="1h"
))

# Add panel to dashboard
dashboard.add_panel(custom_panel)
```

## 🚀 Production Monitoring Setup

### Environment Configuration

Configure monitoring for production:

```bash
# .env file
LOGFIRE_TOKEN=your_production_token
LOGFIRE_PROJECT_NAME=darwin-production

# Health check configuration
HEALTH_CHECK_INTERVAL=30
HEALTH_CHECK_TIMEOUT=10

# Metrics configuration
METRICS_COLLECTION_INTERVAL=15
METRICS_RETENTION_DAYS=30

# Alert configuration
ALERT_CHECK_INTERVAL=60
ALERT_COOLDOWN_MINUTES=15

# Performance monitoring
PERFORMANCE_SAMPLING_RATE=0.1
TRACE_SAMPLING_RATE=0.01
```

### Production Checklist

Before deploying monitoring to production:

- [ ] Configure Logfire token and project
- [ ] Set up alert channels (email, Slack, webhooks)
- [ ] Define alert thresholds for your environment
- [ ] Configure health check endpoints
- [ ] Set up monitoring dashboard access
- [ ] Test alert delivery
- [ ] Configure log retention policies
- [ ] Set up backup monitoring (external health checks)

## 🔍 Troubleshooting

### Common Issues

**Issue**: Health checks failing
```python
# Debug health check issues
health_debug = await health_checker.debug_health_checks()
for check, status in health_debug.items():
    if not status['healthy']:
        print(f"Failed check: {check}")
        print(f"Error: {status['error']}")
```

**Issue**: Metrics not collecting
```bash
# Check metrics collector status
curl http://localhost:8000/health/metrics

# Restart metrics collection
curl -X POST http://localhost:8000/admin/restart-metrics
```

**Issue**: Alerts not firing
```python
# Test alert configuration
test_result = await alert_manager.test_alert_rule("high_cpu_usage")
print(f"Alert test result: {test_result}")

# Check alert history
alert_history = await alert_manager.get_alert_history(hours=24)
```

### Performance Troubleshooting

**Issue**: High monitoring overhead
```python
# Reduce monitoring frequency
monitor.configure(
    health_check_interval=60,  # Increased from 30s
    metrics_collection_interval=30,  # Increased from 15s
    performance_sampling_rate=0.05  # Reduced from 0.1
)
```

## 📚 Advanced Topics

### Custom Metrics Backend

Integrate with external monitoring systems:

```python
from darwin.monitoring import MetricsBackend

class PrometheusBackend(MetricsBackend):
    def __init__(self, prometheus_gateway):
        self.gateway = prometheus_gateway

    async def send_metric(self, name, value, tags=None):
        # Send metric to Prometheus
        await self.gateway.send(name, value, tags)

# Use custom backend
metrics.add_backend(PrometheusBackend("http://prometheus:9091"))
```

### Monitoring Automation

Automate monitoring responses:

```python
from darwin.monitoring import AutomationRule

# Auto-scale rule
auto_scale_rule = AutomationRule(
    name="auto_scale_workers",
    trigger_condition="cpu_percent > 80 AND duration > 300",
    action="scale_workers_up",
    cooldown_minutes=10
)

# Auto-restart rule
auto_restart_rule = AutomationRule(
    name="restart_unhealthy_service",
    trigger_condition="health_status == 'unhealthy' AND duration > 120",
    action="restart_service",
    cooldown_minutes=30
)

monitor.add_automation_rule(auto_scale_rule)
monitor.add_automation_rule(auto_restart_rule)
```

## 📖 Additional Resources

- [Performance Tuning Guide](../operations/performance.md)
- [Troubleshooting Guide](../operations/troubleshooting.md)
- [API Reference](../api/monitoring-api.md)
- [Logfire Documentation](https://logfire.pydantic.dev/docs/)

---

**Need Help?** Contact our support team at [support@devq.ai](mailto:support@devq.ai) or join our [Discord community](https://discord.gg/devqai).
