# Darwin REST API Reference

This document provides comprehensive documentation for Darwin's REST API, including authentication, endpoints, request/response formats, and code examples.

## 📋 Table of Contents

1. [API Overview](#api-overview)
2. [Authentication](#authentication)
3. [Request/Response Format](#requestresponse-format)
4. [Error Handling](#error-handling)
5. [Rate Limiting](#rate-limiting)
6. [Optimization API](#optimization-api)
7. [Problem Definition API](#problem-definition-api)
8. [Results API](#results-api)
9. [Monitoring API](#monitoring-api)
10. [Admin API](#admin-api)
11. [WebSocket API](#websocket-api)
12. [SDK Examples](#sdk-examples)

## 🌐 API Overview

### Base URL
```
Production: https://api.darwin.yourdomain.com/api/v1
Development: http://localhost:8000/api/v1
```

### API Versions
- **v1** (Current): Latest stable API version
- **v2** (Beta): Next generation API with enhanced features

### Content Types
- Request: `application/json`
- Response: `application/json`
- File uploads: `multipart/form-data`

### HTTP Methods
- `GET`: Retrieve resources
- `POST`: Create new resources
- `PUT`: Update entire resources
- `PATCH`: Partial resource updates
- `DELETE`: Remove resources

## 🔐 Authentication

### API Key Authentication

```bash
# Header-based authentication
curl -H "Authorization: Bearer your-api-key" \
     https://api.darwin.yourdomain.com/api/v1/optimizations
```

### JWT Token Authentication

```bash
# Get JWT token
curl -X POST https://api.darwin.yourdomain.com/api/v1/auth/login \
     -H "Content-Type: application/json" \
     -d '{"username": "your-username", "password": "your-password"}'

# Use JWT token
curl -H "Authorization: Bearer your-jwt-token" \
     https://api.darwin.yourdomain.com/api/v1/optimizations
```

### Authentication Endpoints

#### Login
```http
POST /api/v1/auth/login
Content-Type: application/json

{
  "username": "string",
  "password": "string"
}
```

**Response:**
```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "token_type": "bearer",
  "expires_in": 3600,
  "refresh_token": "refresh_token_here",
  "user": {
    "id": "user_123",
    "username": "john_doe",
    "email": "john@example.com"
  }
}
```

#### Refresh Token
```http
POST /api/v1/auth/refresh
Content-Type: application/json

{
  "refresh_token": "string"
}
```

#### Logout
```http
POST /api/v1/auth/logout
Authorization: Bearer your-jwt-token
```

## 📝 Request/Response Format

### Standard Response Structure

```json
{
  "success": true,
  "data": {...},
  "message": "Operation completed successfully",
  "timestamp": "2025-01-27T12:00:00Z",
  "request_id": "req_abc123"
}
```

### Error Response Structure

```json
{
  "success": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid input parameters",
    "details": {
      "field": "population_size",
      "reason": "Must be between 10 and 1000"
    }
  },
  "timestamp": "2025-01-27T12:00:00Z",
  "request_id": "req_abc123"
}
```

### Pagination

```json
{
  "success": true,
  "data": [...],
  "pagination": {
    "page": 1,
    "per_page": 20,
    "total": 150,
    "total_pages": 8,
    "has_next": true,
    "has_prev": false
  }
}
```

## ⚠️ Error Handling

### HTTP Status Codes

| Code | Meaning | Description |
|------|---------|-------------|
| 200 | OK | Request successful |
| 201 | Created | Resource created successfully |
| 204 | No Content | Request successful, no content returned |
| 400 | Bad Request | Invalid request parameters |
| 401 | Unauthorized | Authentication required |
| 403 | Forbidden | Insufficient permissions |
| 404 | Not Found | Resource not found |
| 422 | Unprocessable Entity | Validation error |
| 429 | Too Many Requests | Rate limit exceeded |
| 500 | Internal Server Error | Server error |

### Error Codes

| Code | Description |
|------|-------------|
| `VALIDATION_ERROR` | Request validation failed |
| `AUTHENTICATION_ERROR` | Authentication failed |
| `AUTHORIZATION_ERROR` | Insufficient permissions |
| `RESOURCE_NOT_FOUND` | Requested resource not found |
| `RATE_LIMIT_EXCEEDED` | Rate limit exceeded |
| `OPTIMIZATION_ERROR` | Optimization execution failed |
| `INTERNAL_ERROR` | Internal server error |

## 🚦 Rate Limiting

### Rate Limits

| Endpoint Type | Limit | Window |
|---------------|-------|---------|
| Authentication | 10 requests | 1 minute |
| API Calls | 100 requests | 1 minute |
| File Uploads | 5 requests | 1 minute |
| WebSocket | 1 connection | Per user |

### Rate Limit Headers

```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1643723400
```

## 🧬 Optimization API

### Create Optimization

```http
POST /api/v1/optimizations
Authorization: Bearer your-token
Content-Type: application/json

{
  "name": "Portfolio Optimization",
  "description": "Optimize investment portfolio for maximum return with minimum risk",
  "problem": {
    "variables": [
      {
        "name": "stock_allocation",
        "type": "continuous",
        "bounds": [0.0, 0.6],
        "description": "Allocation to stocks"
      },
      {
        "name": "bond_allocation",
        "type": "continuous",
        "bounds": [0.0, 0.4],
        "description": "Allocation to bonds"
      }
    ],
    "constraints": [
      {
        "type": "equality",
        "expression": "stock_allocation + bond_allocation == 1.0",
        "description": "Total allocation must equal 100%"
      }
    ],
    "objectives": [
      {
        "name": "maximize_return",
        "type": "maximize",
        "function": "portfolio_return"
      },
      {
        "name": "minimize_risk",
        "type": "minimize",
        "function": "portfolio_risk"
      }
    ]
  },
  "config": {
    "algorithm": "nsga2",
    "population_size": 100,
    "max_generations": 200,
    "mutation_rate": 0.1,
    "crossover_rate": 0.8,
    "early_stopping": true,
    "convergence_threshold": 0.001
  },
  "tags": ["finance", "portfolio", "multi-objective"]
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "optimization_id": "opt_123abc",
    "name": "Portfolio Optimization",
    "status": "created",
    "created_at": "2025-01-27T12:00:00Z",
    "estimated_duration": 300,
    "problem_dimensions": 2,
    "population_size": 100,
    "max_generations": 200
  },
  "message": "Optimization created successfully"
}
```

### Start Optimization

```http
POST /api/v1/optimizations/{optimization_id}/start
Authorization: Bearer your-token
```

**Response:**
```json
{
  "success": true,
  "data": {
    "optimization_id": "opt_123abc",
    "status": "running",
    "started_at": "2025-01-27T12:05:00Z",
    "current_generation": 0,
    "best_fitness": null,
    "progress": 0.0
  }
}
```

### Get Optimization Status

```http
GET /api/v1/optimizations/{optimization_id}
Authorization: Bearer your-token
```

**Response:**
```json
{
  "success": true,
  "data": {
    "optimization_id": "opt_123abc",
    "name": "Portfolio Optimization",
    "status": "running",
    "progress": {
      "current_generation": 45,
      "max_generations": 200,
      "percentage": 22.5,
      "estimated_time_remaining": 180
    },
    "metrics": {
      "best_fitness": 0.0234,
      "average_fitness": 0.0567,
      "diversity_index": 0.85,
      "convergence_rate": 0.023
    },
    "created_at": "2025-01-27T12:00:00Z",
    "started_at": "2025-01-27T12:05:00Z",
    "updated_at": "2025-01-27T12:10:00Z"
  }
}
```

### List Optimizations

```http
GET /api/v1/optimizations?page=1&per_page=20&status=completed&tag=finance
Authorization: Bearer your-token
```

**Query Parameters:**
- `page`: Page number (default: 1)
- `per_page`: Items per page (default: 20, max: 100)
- `status`: Filter by status (`created`, `running`, `completed`, `failed`, `cancelled`)
- `tag`: Filter by tag
- `created_after`: Filter by creation date (ISO 8601)
- `created_before`: Filter by creation date (ISO 8601)

**Response:**
```json
{
  "success": true,
  "data": [
    {
      "optimization_id": "opt_123abc",
      "name": "Portfolio Optimization",
      "status": "completed",
      "best_fitness": 0.0123,
      "generations_run": 150,
      "execution_time": 245.6,
      "created_at": "2025-01-27T12:00:00Z",
      "completed_at": "2025-01-27T12:04:05Z"
    }
  ],
  "pagination": {
    "page": 1,
    "per_page": 20,
    "total": 1,
    "total_pages": 1
  }
}
```

### Stop Optimization

```http
POST /api/v1/optimizations/{optimization_id}/stop
Authorization: Bearer your-token
```

### Delete Optimization

```http
DELETE /api/v1/optimizations/{optimization_id}
Authorization: Bearer your-token
```

## 🎯 Problem Definition API

### Create Problem Template

```http
POST /api/v1/problems
Authorization: Bearer your-token
Content-Type: application/json

{
  "name": "Traveling Salesman Problem",
  "description": "Find the shortest route visiting all cities exactly once",
  "category": "combinatorial",
  "variables": [
    {
      "name": "route",
      "type": "permutation",
      "size": 20,
      "description": "Order of cities to visit"
    }
  ],
  "objectives": [
    {
      "name": "minimize_distance",
      "type": "minimize",
      "function": "calculate_total_distance"
    }
  ],
  "default_config": {
    "algorithm": "genetic",
    "population_size": 200,
    "max_generations": 500
  },
  "tags": ["tsp", "routing", "combinatorial"]
}
```

### Get Problem Templates

```http
GET /api/v1/problems?category=finance&tag=portfolio
Authorization: Bearer your-token
```

### Validate Problem Definition

```http
POST /api/v1/problems/validate
Authorization: Bearer your-token
Content-Type: application/json

{
  "variables": [...],
  "constraints": [...],
  "objectives": [...]
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "valid": true,
    "warnings": [
      "No constraints defined - consider adding constraints for better results"
    ],
    "estimated_complexity": "medium",
    "recommended_config": {
      "population_size": 100,
      "max_generations": 200
    }
  }
}
```

## 📊 Results API

### Get Optimization Results

```http
GET /api/v1/optimizations/{optimization_id}/results
Authorization: Bearer your-token
```

**Response:**
```json
{
  "success": true,
  "data": {
    "optimization_id": "opt_123abc",
    "status": "completed",
    "best_solution": {
      "variables": {
        "stock_allocation": 0.65,
        "bond_allocation": 0.35
      },
      "fitness": 0.0123,
      "objectives": {
        "portfolio_return": 0.085,
        "portfolio_risk": 0.023
      }
    },
    "statistics": {
      "generations_run": 150,
      "total_evaluations": 15000,
      "execution_time": 245.6,
      "convergence_generation": 142,
      "final_diversity": 0.23
    },
    "pareto_frontier": [
      {
        "solution": {...},
        "objectives": {...}
      }
    ]
  }
}
```

### Get Evolution History

```http
GET /api/v1/optimizations/{optimization_id}/history?metric=best_fitness
Authorization: Bearer your-token
```

**Response:**
```json
{
  "success": true,
  "data": {
    "generations": [0, 1, 2, ..., 150],
    "best_fitness": [0.5, 0.3, 0.2, ..., 0.0123],
    "average_fitness": [0.8, 0.6, 0.4, ..., 0.045],
    "diversity": [1.0, 0.95, 0.9, ..., 0.23]
  }
}
```

### Export Results

```http
GET /api/v1/optimizations/{optimization_id}/export?format=csv
Authorization: Bearer your-token
```

**Query Parameters:**
- `format`: Export format (`json`, `csv`, `xlsx`)
- `include`: Data to include (`solution`, `history`, `statistics`, `all`)

### Get Result Visualization

```http
GET /api/v1/optimizations/{optimization_id}/visualizations/{chart_type}
Authorization: Bearer your-token
```

**Chart Types:**
- `convergence`: Fitness convergence over generations
- `pareto`: Pareto frontier for multi-objective problems
- `diversity`: Population diversity over time
- `variables`: Variable value distributions

## 📊 Monitoring API

### System Health

```http
GET /api/v1/health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-01-27T12:00:00Z",
  "components": {
    "database": {
      "status": "healthy",
      "response_time": 0.05
    },
    "cache": {
      "status": "healthy",
      "response_time": 0.01
    },
    "api": {
      "status": "healthy",
      "active_optimizations": 5
    }
  },
  "system": {
    "cpu_percent": 45.2,
    "memory_percent": 67.8,
    "disk_percent": 23.1
  }
}
```

### Performance Metrics

```http
GET /api/v1/metrics?time_range=1h
Authorization: Bearer your-token
```

**Response:**
```json
{
  "success": true,
  "data": {
    "api_metrics": {
      "requests_per_second": 12.5,
      "average_response_time": 0.085,
      "error_rate": 0.002
    },
    "optimization_metrics": {
      "active_optimizations": 5,
      "completed_today": 23,
      "average_execution_time": 180.5
    },
    "system_metrics": {
      "cpu_usage": 45.2,
      "memory_usage": 67.8,
      "network_io": 1024000
    }
  }
}
```

### System Status

```http
GET /api/v1/status
Authorization: Bearer your-token
```

**Response:**
```json
{
  "success": true,
  "data": {
    "version": "1.0.0",
    "environment": "production",
    "uptime": 86400,
    "features": {
      "multi_objective": true,
      "constraint_handling": true,
      "parallel_execution": true,
      "mcp_server": true
    },
    "limits": {
      "max_concurrent_optimizations": 10,
      "max_population_size": 1000,
      "max_generations": 5000
    }
  }
}
```

## 👨‍💼 Admin API

### User Management

#### List Users
```http
GET /api/v1/admin/users?page=1&per_page=20
Authorization: Bearer admin-token
```

#### Create User
```http
POST /api/v1/admin/users
Authorization: Bearer admin-token
Content-Type: application/json

{
  "username": "new_user",
  "email": "user@example.com",
  "password": "secure_password",
  "role": "user"
}
```

#### Update User
```http
PUT /api/v1/admin/users/{user_id}
Authorization: Bearer admin-token
Content-Type: application/json

{
  "email": "updated@example.com",
  "role": "admin",
  "active": true
}
```

### System Administration

#### Get System Statistics
```http
GET /api/v1/admin/statistics
Authorization: Bearer admin-token
```

**Response:**
```json
{
  "success": true,
  "data": {
    "users": {
      "total": 150,
      "active": 142,
      "new_this_month": 23
    },
    "optimizations": {
      "total": 5420,
      "completed": 4890,
      "failed": 245,
      "running": 8
    },
    "performance": {
      "average_execution_time": 180.5,
      "success_rate": 0.955,
      "api_uptime": 0.9995
    }
  }
}
```

#### Restart Services
```http
POST /api/v1/admin/services/{service_name}/restart
Authorization: Bearer admin-token
```

## 🔌 WebSocket API

### Connection

```javascript
const ws = new WebSocket('wss://api.darwin.yourdomain.com/ws');

// Authentication
ws.send(JSON.stringify({
  type: 'auth',
  token: 'your-jwt-token'
}));
```

### Subscribe to Optimization Updates

```javascript
// Subscribe to optimization progress
ws.send(JSON.stringify({
  type: 'subscribe',
  channel: 'optimization',
  optimization_id: 'opt_123abc'
}));

// Handle updates
ws.onmessage = function(event) {
  const data = JSON.parse(event.data);
  if (data.type === 'optimization_update') {
    console.log('Progress:', data.progress);
    console.log('Best fitness:', data.best_fitness);
  }
};
```

### Real-time Monitoring

```javascript
// Subscribe to system metrics
ws.send(JSON.stringify({
  type: 'subscribe',
  channel: 'metrics'
}));

// Handle metric updates
ws.onmessage = function(event) {
  const data = JSON.parse(event.data);
  if (data.type === 'metrics_update') {
    updateDashboard(data.metrics);
  }
};
```

## 💾 SDK Examples

### Python SDK

```python
import darwin

# Initialize client
client = darwin.Client(
    api_key="your-api-key",
    base_url="https://api.darwin.yourdomain.com"
)

# Create optimization
problem = darwin.Problem(
    variables=[
        darwin.Variable("x", type="continuous", bounds=[-5, 5]),
        darwin.Variable("y", type="continuous", bounds=[-5, 5])
    ],
    objectives=[
        darwin.Objective("minimize", "rastrigin_2d")
    ]
)

optimization = client.create_optimization(
    name="Rastrigin Optimization",
    problem=problem,
    config=darwin.Config(
        algorithm="genetic",
        population_size=100,
        max_generations=200
    )
)

# Start optimization
optimization.start()

# Monitor progress
for update in optimization.stream_progress():
    print(f"Generation: {update.generation}, Best: {update.best_fitness}")

# Get results
results = optimization.get_results()
print(f"Best solution: {results.best_solution}")
```

### JavaScript SDK

```javascript
import Darwin from 'darwin-js-sdk';

// Initialize client
const darwin = new Darwin({
  apiKey: 'your-api-key',
  baseUrl: 'https://api.darwin.yourdomain.com'
});

// Create optimization
const optimization = await darwin.optimizations.create({
  name: 'Portfolio Optimization',
  problem: {
    variables: [
      { name: 'stocks', type: 'continuous', bounds: [0, 0.8] },
      { name: 'bonds', type: 'continuous', bounds: [0, 0.5] }
    ],
    objectives: [
      { name: 'maximize_return', type: 'maximize' },
      { name: 'minimize_risk', type: 'minimize' }
    ]
  },
  config: {
    algorithm: 'nsga2',
    populationSize: 100,
    maxGenerations: 200
  }
});

// Start optimization
await optimization.start();

// Real-time updates
optimization.onProgress((update) => {
  console.log(`Progress: ${update.progress}%`);
  console.log(`Best fitness: ${update.bestFitness}`);
});

// Wait for completion
const results = await optimization.waitForCompletion();
console.log('Best solution:', results.bestSolution);
```

### cURL Examples

#### Create and Run Optimization

```bash
#!/bin/bash

API_BASE="https://api.darwin.yourdomain.com/api/v1"
TOKEN="your-jwt-token"

# Create optimization
OPTIMIZATION_ID=$(curl -s -X POST "$API_BASE/optimizations" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Test Optimization",
    "problem": {
      "variables": [
        {"name": "x", "type": "continuous", "bounds": [-5, 5]}
      ],
      "objectives": [
        {"name": "minimize", "function": "sphere"}
      ]
    },
    "config": {
      "population_size": 50,
      "max_generations": 100
    }
  }' | jq -r '.data.optimization_id')

echo "Created optimization: $OPTIMIZATION_ID"

# Start optimization
curl -X POST "$API_BASE/optimizations/$OPTIMIZATION_ID/start" \
  -H "Authorization: Bearer $TOKEN"

# Monitor progress
while true; do
  STATUS=$(curl -s "$API_BASE/optimizations/$OPTIMIZATION_ID" \
    -H "Authorization: Bearer $TOKEN" | jq -r '.data.status')

  if [ "$STATUS" = "completed" ] || [ "$STATUS" = "failed" ]; then
    break
  fi

  echo "Status: $STATUS"
  sleep 5
done

# Get results
curl -s "$API_BASE/optimizations/$OPTIMIZATION_ID/results" \
  -H "Authorization: Bearer $TOKEN" | jq '.data.best_solution'
```

## 📚 Additional Resources

### OpenAPI Specification
- **Swagger UI**: https://api.darwin.yourdomain.com/docs
- **OpenAPI JSON**: https://api.darwin.yourdomain.com/openapi.json

### Rate Limiting
- **Current limits**: Check `X-RateLimit-*` headers
- **Upgrade options**: Contact support for higher limits

### Support
- **Documentation**: https://docs.darwin.yourdomain.com
- **GitHub Issues**: https://github.com/devqai/darwin/issues
- **Email Support**: api-support@devq.ai
- **Discord**: https://discord.gg/devqai

### Changelog
- **API Updates**: https://docs.darwin.yourdomain.com/changelog
- **Breaking Changes**: Announced 30 days in advance
- **Deprecation Policy**: 6 months notice for deprecated endpoints

---

**Darwin REST API v1.0** | Last updated: January 2025 | © DevQ.ai
