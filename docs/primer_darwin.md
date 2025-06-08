## Project Inputs
### PROJECT=darwin
### LOCAL=`/Users/dionedge/devqai/`{PROJECT}
### REMOTE=`https://github.com/devqai/`{PROJECT}
### DESCRIPTION:"Interactive analysis suite providing genetic algorithms to use probabilistic solutions through Model Context Protocol and standalone application with Panel Dashboard"
### REFERENCE=pygad:https://github.com/ahmedfgad/GeneticAlgorithmPython

---

## This directory is: {LOCAL}{PROJECT}
### This file is: `primer_darwin.md`
### Environment variables present: `/.env`
### Review rules for this project: {LOCAL}{PROJECT}`/.rules`
### Changelog for this project: {LOCAL}{PROJECT}`/CHANGELOG.md`
### Review settings and development: {LOCAL}{PROJECT}}`/dev_report.md`

## Frameworks
fastapi https://github.com/fastapi/fastapi
logfire https://github.com/pydantic/logfire
surrealdb https://github.com/surrealdb/surrealdb
pygad: https://github.com/ahmedfgad/GeneticAlgorithmPython
pytorch: https://github.com/pytorch/pytorch
bokeh: https://github.com/bokeh/bokeh
panel: https://github.com/holoviz/panel

## Knowledge Base
### Availalbe in ptolemies-mcp:
- https://pygad.readthedocs.io/en/latest/
- https://docs.pytorch.org/docs/stable/index.html
- https://docs.bokeh.org/en/latest/docs/first_steps.html
- https://docs.bokeh.org/en/latest/docs/user_guide.html#userguide
- https://docs.bokeh.org/en/latest/docs/reference.html
- https://panel.holoviz.org/getting_started/installation.html
- https://panel.holoviz.org/getting_started/build_app.html
- https://panel.holoviz.org/getting_started/core_concepts.html
- https://panel.holoviz.org/reference/index.html#
- https://panel.holoviz.org/how_to/index.html
- https://panel.holoviz.org/api/index.html

---

# CORE SPECIFICATIONS

architecture:
  type: "Microservice MCP Server"
  language: "Python 3.8+"
  framework: "FastAPI + PyGAD"
  protocol: "Model Context Protocol (MCP)"
  deployment: "Standalone server, Docker-ready"

core_components:
  genetic-algorithms:
    purpose: "genetic-solver-engine"
    capabilities:

  mcp_server:
    purpose: "FastAPI-based MCP protocol implementation"
    endpoints:
      - "POST /mcp - Main MCP protocol endpoint"
      - "GET /health - Health check"
      - "GET /schema - API documentation"
      - "GET /functions - Available functions"

  interactive_demos:
    purpose: "Educational Jupyter notebooks"
    content:

# FUNCTIONAL REQUIREMENTS

required_functions:
  model_management:

  prediction_analysis:
    - "predict: Generate uncertainty-aware predictions"
    - "create_visualization:"

  real_world_applications:

# TECHNICAL REQUIREMENTS

performance:
  response_time: "<500ms for typical models"
  mcmc_convergence: "R-hat < 1.01"
  memory_usage: "<2GB for standard workloads"
  uptime: "99.9% availability"

dependencies:
  core:
    - "pymc>=5.0.0"
    - "fastapi>=0.100.0"
    - "pydantic>=2.0.0"
    - "mcp>=1.6.0"

  optional:
    - "logfire>=3.0.0 (observability)"
    - "jupyter>=1.0.0 (interactive notebooks)"

# QUALITY REQUIREMENTS

testing:
  coverage:
  test_count:
  pass_rate:
  validation:

reliability:
  error_handling: "Comprehensive throughout codebase"
  validation: "Pydantic schema validation"
  monitoring: "Health endpoints + optional Logfire"
  fallback: "Graceful degradation"

# BUSINESS REQUIREMENTS

target_users:
  - "Data scientists requiring uncertainty quantification"
  - "Business analysts for prediction"
  - "Financial analysts for risk assessment"
  - "ML engineers for model uncertainty"

value_propositions:
  quantified_benefits:

  competitive_advantages:

# DEPLOYMENT REQUIREMENTS

infrastructure:
  minimum:
    cpu: "2 cores"
    memory: "4GB RAM"
    storage: "10GB"
    network: "HTTP/HTTPS access"

  recommended:
    cpu: "4+ cores"
    memory: "8GB RAM"
    storage: "50GB (for large models)"
    network: "Load balancer ready"

startup:
  command: ""
  health_check: "GET /health"
  readiness: "200 OK response"

# INTEGRATION REQUIREMENTS

api_clients:
  supported:
    - "Python HTTP clients"
    - "Node.js Fetch API"
    - "Standard MCP protocol clients"
    - "Jupyter notebook integration"
    - "CLI automation tools"

data_formats:
  input: "JSON (MCP protocol)"
  output: "JSON with statistical results"
  visualization: "Base64 encoded plots"

# SECURITY & COMPLIANCE

security:
  authentication: "Optional (configurable)"
  authorization: "Role-based (if auth enabled)"
  data_privacy: "No persistent data storage"
  network: "HTTPS support"

compliance:
  standards:
    - "Basel III (financial risk)"
    - "FDA guidelines (medical applications)"
    - "GDPR compliance ready"

# EXTENSIBILITY REQUIREMENTS

modularity:
  - "Plugin architecture for new distributions"
  - "Configurable visualization options"
  - "Domain-specific adapters"
  - "Cloud deployment templates"

future_enhancements:
  - "Advanced optimization algorithms"
  - "Multi-model ensembles"
  - "Real-time streaming data"
  - "GPU acceleration support"

# SUCCESS CRITERIA

acceptance_criteria:
  functional:
    - "All core MCP functions operational"
    - "demo notebooks execute successfully"
    - "API responds within performance thresholds"

  quality:
    - "100% test pass rate maintained"
    - "Zero critical security vulnerabilities"
    - "Documentation completeness verified"

  business:
    - "Demonstrable ROI in target applications"
    - "User adoption metrics met"
    - "Customer satisfaction scores >4.5/5"

# MAINTENANCE REQUIREMENTS

monitoring:
  metrics:
    - "Request latency and throughput"
    - "Error rates and types"
    - "Resource utilization"

updates:
  schedule: "Monthly dependency updates"
  testing: "Full regression suite before releases"
  documentation: "Keep in sync with code changes"

support:
  levels:
    - "Community support via GitHub"
    - "Enterprise support available"
    - "Documentation and examples"
    - "Video tutorials and workshops"

---

## Use Taskmaster-AI for Project Management
{
  "mcpServers": {
    "taskmaster-ai": {
      "command": "npx",
      "args": ["-y", "--package=task-master-ai", "task-master-ai"],
      "env": {
        "ANTHROPIC_API_KEY": "YOUR_ANTHROPIC_API_KEY_HERE",
        "MODEL": "claude-sonnet-4-20250514",
        "MAX_TOKENS": 64000,
        "TEMPERATURE": 0.2,
        "DEFAULT_SUBTASKS": 5,
        "DEFAULT_PRIORITY": "medium"
      }
    }
  }
}

---
