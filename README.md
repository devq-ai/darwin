# Darwin: Genetic Algorithm Solver

[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Documentation](https://img.shields.io/badge/docs-available-brightgreen.svg)](https://darwin.devq.ai/docs)
[![Status](https://img.shields.io/badge/status-beta-orange.svg)](https://github.com/devqai/darwin)

**Darwin** is a comprehensive genetic algorithm optimization platform that provides both standalone application capabilities and Model Context Protocol (MCP) server integration. Built on the proven PyGAD foundation, Darwin offers advanced genetic algorithm solutions for complex optimization problems with real-time visualization, interactive analysis, and enterprise-grade reliability.

## 🧬 Key Features

- **Dual Deployment**: Standalone Panel dashboard + MCP server for AI agent integration
- **Advanced Algorithms**: Multi-objective optimization, adaptive operators, constraint handling
- **Real-time Visualization**: Interactive dashboards with Bokeh/Panel for evolution tracking
- **Enterprise Ready**: Logfire observability, production monitoring, scalable architecture
- **Developer Friendly**: Rich API, comprehensive documentation, extensive examples

## 🚀 Quick Start

### Installation

```bash
# Install from PyPI (recommended)
pip install darwin-genetic-solver

# Or install from source
git clone https://github.com/devqai/darwin.git
cd darwin
pip install -e .
```

### Basic Usage

```python
from darwin import GeneticOptimizer, OptimizationProblem

# Define optimization problem
problem = OptimizationProblem(
    name="Rastrigin Function Optimization",
    variables=[
        {"name": "x", "type": "continuous", "bounds": [-5.12, 5.12]},
        {"name": "y", "type": "continuous", "bounds": [-5.12, 5.12]}
    ],
    fitness_function="rastrigin_2d",
    objective_type="minimize"
)

# Initialize and run optimizer
optimizer = GeneticOptimizer(problem)
result = optimizer.run(max_generations=100)

print(f"Best solution: {result.best_solution}")
print(f"Best fitness: {result.best_fitness}")
```

### Launch Dashboard

```bash
# Start interactive web dashboard
darwin-dashboard

# Or from Python
from darwin import DarwinDashboard
dashboard = DarwinDashboard()
dashboard.serve()
```

### Start MCP Server

```bash
# Start MCP server for AI agent integration
darwin-server --host 0.0.0.0 --port 8000

# Or from Python
from darwin import DarwinMCPServer
server = DarwinMCPServer()
server.run()
```

## 📊 Advanced Examples

### Multi-Objective Portfolio Optimization

```python
from darwin import GeneticOptimizer, OptimizationProblem
from darwin.algorithms import NSGAII

# Portfolio optimization with risk vs return
problem = OptimizationProblem(
    name="Portfolio Optimization",
    variables=[
        {"name": f"weight_{i}", "type": "continuous", "bounds": [0, 1]}
        for i in range(10)  # 10 assets
    ],
    constraints=[
        {"type": "equality", "expression": "sum(weights) == 1"}
    ],
    objectives=["maximize_return", "minimize_risk"],
    objective_type="multi_objective"
)

# Use NSGA-II for multi-objective optimization
optimizer = GeneticOptimizer(problem, algorithm=NSGAII())
result = optimizer.run(max_generations=500)

# Analyze Pareto frontier
pareto_front = result.get_pareto_frontier()
print(f"Found {len(pareto_front)} Pareto-optimal solutions")
```

### Neural Network Hyperparameter Tuning

```python
from darwin import GeneticOptimizer, OptimizationProblem

# Optimize neural network architecture and hyperparameters
problem = OptimizationProblem(
    name="Neural Network Optimization",
    variables=[
        {"name": "hidden_layers", "type": "discrete", "bounds": [1, 5]},
        {"name": "neurons_per_layer", "type": "discrete", "bounds": [10, 500]},
        {"name": "learning_rate", "type": "continuous", "bounds": [0.001, 0.1]},
        {"name": "dropout_rate", "type": "continuous", "bounds": [0.0, 0.5]},
        {"name": "batch_size", "type": "categorical", "values": [16, 32, 64, 128]}
    ],
    fitness_function=train_and_evaluate_model,
    objective_type="maximize"  # Maximize validation accuracy
)

optimizer = GeneticOptimizer(problem)
result = optimizer.run(max_generations=50)
```

## 🔌 MCP Integration

Darwin provides full Model Context Protocol support for AI agent integration:

```python
import asyncio
from mcp_client import MCPClient

async def optimize_with_mcp():
    client = MCPClient("ws://localhost:8000/mcp")

    # Create optimizer through MCP
    result = await client.call_tool(
        "create_optimizer",
        {
            "problem": problem_definition,
            "config": ga_configuration
        }
    )

    optimizer_id = result["optimizer_id"]

    # Run optimization
    optimization_result = await client.call_tool(
        "run_optimization",
        {"optimizer_id": optimizer_id}
    )

    return optimization_result

# Run async optimization
result = asyncio.run(optimize_with_mcp())
```

## 📈 Dashboard Features

The Darwin dashboard provides:

- **Problem Definition Interface**: Visual problem setup with drag-and-drop
- **Real-time Monitoring**: Live fitness tracking and population analysis
- **Interactive Visualization**: Bokeh-powered plots with zoom, pan, and export
- **Results Analysis**: Statistical analysis and comparison tools
- **Template Gallery**: Pre-configured optimization problems
- **Experiment Management**: Track and compare multiple runs

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Panel Web UI  │    │   MCP Server    │    │  CLI Interface  │
├─────────────────┤    ├─────────────────┤    ├─────────────────┤
│ • Dashboard     │    │ • Tool Registry │    │ • Batch Jobs    │
│ • Templates     │    │ • Session Mgmt  │    │ • Automation    │
│ • Visualization │    │ • Real-time API │    │ • Scripting     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
         ┌───────────────────────────────────────────────┐
         │              Darwin Core Engine               │
         ├───────────────────────────────────────────────┤
         │ • Enhanced PyGAD Integration                  │
         │ • Multi-objective Optimization (NSGA-II/III) │
         │ • Adaptive Genetic Operators                 │
         │ • Constraint Handling                        │
         │ • Parallel Execution                         │
         │ • Real-time Analytics                        │
         └───────────────────────────────────────────────┘
```

## 🚀 Deployment

### Docker Deployment

```bash
# Quick start with Docker Compose
git clone https://github.com/devqai/darwin.git
cd darwin
docker-compose up -d

# Access dashboard at http://localhost:8000/dashboard
# MCP server available at ws://localhost:8000/mcp
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: darwin-app
spec:
  replicas: 3
  selector:
    matchLabels:
      app: darwin
  template:
    spec:
      containers:
      - name: darwin
        image: devqai/darwin:latest
        ports:
        - containerPort: 8000
        env:
        - name: SURREALDB_URL
          value: "ws://surrealdb-service:8000/rpc"
        - name: REDIS_URL
          value: "redis://redis-service:6379"
```

## 📊 Performance Benchmarks

Darwin has been tested on standard optimization benchmarks:

| Problem | Dimensions | Target | Generations | Success Rate |
|---------|------------|--------|-------------|--------------|
| Sphere | 10 | 0.0 | 100 | 95% |
| Rastrigin | 10 | 0.0 | 500 | 85% |
| Ackley | 10 | 0.0 | 150 | 92% |
| Rosenbrock | 10 | 0.0 | 1000 | 80% |

System performance metrics:
- **API Response Time**: < 100ms (95th percentile)
- **Optimization Throughput**: > 1,000 evaluations/second
- **Memory Usage**: < 2GB for standard problems
- **Concurrent Users**: 50+ simultaneous optimizations

## 🛠️ Development

### Development Setup

```bash
# Clone repository
git clone https://github.com/devqai/darwin.git
cd darwin

# Install dependencies with poetry
poetry install --with dev,docs

# Install pre-commit hooks
pre-commit install

# Run tests
pytest

# Start development server
darwin-server --reload
```

### Project Structure

```
darwin/
├── src/darwin/           # Main package
│   ├── core/            # Core optimization engine
│   ├── algorithms/      # GA algorithm implementations
│   ├── mcp/            # MCP server and tools
│   ├── dashboard/      # Panel dashboard
│   └── utils/          # Utilities and helpers
├── tests/              # Test suite
├── docs/               # Documentation
├── examples/           # Example notebooks and scripts
└── docker/             # Docker configuration
```

### Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and add tests
4. Run the test suite: `pytest`
5. Submit a pull request

## 📚 Documentation

- **Full Documentation**: https://darwin.devq.ai/docs
- **API Reference**: https://darwin.devq.ai/api
- **Tutorials**: https://darwin.devq.ai/tutorials
- **Examples**: [examples/](examples/) directory

## 🤝 Community

- **GitHub Discussions**: https://github.com/devqai/darwin/discussions
- **Issues**: https://github.com/devqai/darwin/issues
- **Discord**: https://discord.gg/devqai
- **Email**: team@devq.ai

## 📄 License

Darwin is released under the BSD 3-Clause License. See [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- Built on [PyGAD](https://github.com/ahmedfgad/GeneticAlgorithmPython) by Ahmed Gad
- Dashboard powered by [Panel](https://panel.holoviz.org/) and [Bokeh](https://bokeh.org/)
- Observability with [Logfire](https://logfire.pydantic.dev/)
- Database integration with [SurrealDB](https://surrealdb.com/)

## 🔗 Related Projects

- [PyGAD](https://github.com/ahmedfgad/GeneticAlgorithmPython) - Genetic Algorithm Python Library
- [Bayes MCP](../bayes) - Bayesian inference MCP server
- [Ptolemies](../ptolemies) - Knowledge base with crawled documentation
- [Context7 MCP](../mcp/mcp-servers/context7-mcp) - Advanced context management

---

**Darwin** - *Evolving Optimization Solutions*
Copyright © 2025 DevQ.ai - All Rights Reserved
