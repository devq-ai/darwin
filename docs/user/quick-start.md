# Quick Start Guide

Get up and running with Darwin in less than 5 minutes! This guide will walk you through installation, basic usage, and your first optimization problem.

## 📦 Installation

### Option 1: Install from PyPI (Recommended)

```bash
pip install darwin-genetic-solver
```

### Option 2: Install from Source

```bash
git clone https://github.com/devqai/darwin.git
cd darwin
pip install -e .
```

### Option 3: Docker Quick Start

```bash
docker run -p 8000:8000 devqai/darwin:latest
```

## 🚀 Your First Optimization

Let's solve the classic Rastrigin function optimization problem:

### 1. Basic Python Usage

```python
from darwin import GeneticOptimizer, OptimizationProblem

# Define the optimization problem
problem = OptimizationProblem(
    name="Rastrigin Function Minimization",
    variables=[
        {"name": "x", "type": "continuous", "bounds": [-5.12, 5.12]},
        {"name": "y", "type": "continuous", "bounds": [-5.12, 5.12]}
    ],
    fitness_function="rastrigin_2d",
    objective_type="minimize"
)

# Create and run optimizer
optimizer = GeneticOptimizer(problem)
result = optimizer.run(max_generations=100)

# Display results
print(f"Best solution: x={result.best_solution[0]:.4f}, y={result.best_solution[1]:.4f}")
print(f"Best fitness: {result.best_fitness:.6f}")
print(f"Generations: {result.generations_run}")
```

**Expected Output:**
```
Best solution: x=0.0021, y=-0.0013
Best fitness: 0.000087
Generations: 87
```

### 2. Launch Interactive Dashboard

```bash
# Start the web dashboard
darwin-dashboard

# Or from Python
from darwin import DarwinDashboard
dashboard = DarwinDashboard()
dashboard.serve()
```

Open your browser to `http://localhost:5006` to access the interactive dashboard.

### 3. Start MCP Server for AI Integration

```bash
# Start MCP server
darwin-server --host 0.0.0.0 --port 8000

# Or from Python
from darwin import DarwinMCPServer
server = DarwinMCPServer()
server.run()
```

## 🎯 Dashboard Quick Tour

### Problem Setup
1. **Navigate to Dashboard**: Open `http://localhost:5006`
2. **Select Template**: Choose "Rastrigin Function" from templates
3. **Configure Parameters**: Adjust population size, generations, mutation rate
4. **Start Optimization**: Click "Run Optimization"

### Real-time Monitoring
- **Fitness Progress**: Watch fitness improve over generations
- **Population Diversity**: Monitor genetic diversity
- **Parameter Evolution**: See how parameters evolve
- **Performance Metrics**: Track optimization speed and efficiency

### Results Analysis
- **Best Solution Visualization**: Interactive plots of best solutions
- **Convergence Analysis**: Detailed convergence statistics
- **Export Results**: Download results as JSON, CSV, or images

## 🔧 Configuration Options

### Basic Configuration

```python
from darwin import GeneticOptimizer, OptimizationConfig

config = OptimizationConfig(
    population_size=100,        # Number of individuals
    max_generations=200,        # Maximum generations
    mutation_rate=0.1,          # Mutation probability
    crossover_rate=0.8,         # Crossover probability
    selection_method="tournament",  # Selection strategy
    early_stopping=True,        # Stop if converged
    verbose=True               # Show progress
)

optimizer = GeneticOptimizer(problem, config=config)
```

### Advanced Configuration

```python
from darwin.algorithms import NSGAII

# Multi-objective optimization
multi_objective_problem = OptimizationProblem(
    name="Portfolio Optimization",
    variables=[
        {"name": f"weight_{i}", "type": "continuous", "bounds": [0, 1]}
        for i in range(5)  # 5 assets
    ],
    constraints=[
        {"type": "equality", "expression": "sum(weights) == 1"}
    ],
    objectives=["maximize_return", "minimize_risk"],
    objective_type="multi_objective"
)

# Use NSGA-II for multi-objective optimization
optimizer = GeneticOptimizer(
    multi_objective_problem,
    algorithm=NSGAII(),
    config=config
)
```

## 📊 Built-in Test Functions

Darwin includes many standard benchmark functions:

```python
# Single-objective functions
functions = [
    "sphere",           # f(x) = sum(x_i^2)
    "rastrigin",        # f(x) = A*n + sum(x_i^2 - A*cos(2*pi*x_i))
    "ackley",           # Complex multimodal function
    "rosenbrock",       # f(x) = sum(100*(x_{i+1} - x_i^2)^2 + (1-x_i)^2)
    "griewank",         # f(x) = 1 + sum(x_i^2)/4000 - prod(cos(x_i/sqrt(i)))
    "schwefel"          # f(x) = 418.9829*n - sum(x_i*sin(sqrt(|x_i|)))
]

# Multi-objective functions
mo_functions = [
    "zdt1", "zdt2", "zdt3",     # ZDT test suite
    "dtlz1", "dtlz2", "dtlz3"   # DTLZ test suite
]
```

## 🔌 MCP Integration Example

Use Darwin from AI agents via MCP:

```python
import asyncio
from mcp_client import MCPClient

async def ai_optimization():
    # Connect to Darwin MCP server
    client = MCPClient("ws://localhost:8000/mcp")

    # Define problem via MCP
    problem_def = {
        "name": "AI Portfolio Optimization",
        "variables": [
            {"name": "tech_allocation", "type": "continuous", "bounds": [0, 0.4]},
            {"name": "healthcare_allocation", "type": "continuous", "bounds": [0, 0.3]},
            {"name": "finance_allocation", "type": "continuous", "bounds": [0, 0.3]}
        ],
        "constraints": [
            {"type": "equality", "expression": "sum(allocations) == 1"}
        ],
        "objectives": ["maximize_return", "minimize_risk"],
        "objective_type": "multi_objective"
    }

    # Create optimizer
    result = await client.call_tool("create_optimizer", {
        "problem": problem_def,
        "config": {"max_generations": 100}
    })

    # Run optimization
    optimization_result = await client.call_tool(
        "run_optimization",
        {"optimizer_id": result["optimizer_id"]}
    )

    return optimization_result

# Run async optimization
result = asyncio.run(ai_optimization())
```

## 🎮 Interactive Examples

### Example 1: Neural Network Hyperparameter Tuning

```python
def train_model(params):
    """Custom fitness function for neural network tuning"""
    learning_rate, batch_size, hidden_layers = params

    # Train your model here
    model = create_model(
        learning_rate=learning_rate,
        batch_size=int(batch_size),
        hidden_layers=int(hidden_layers)
    )

    accuracy = train_and_validate(model)
    return accuracy  # Higher is better

problem = OptimizationProblem(
    name="Neural Network Hyperparameter Optimization",
    variables=[
        {"name": "learning_rate", "type": "continuous", "bounds": [0.001, 0.1]},
        {"name": "batch_size", "type": "discrete", "bounds": [16, 128]},
        {"name": "hidden_layers", "type": "discrete", "bounds": [1, 5]}
    ],
    fitness_function=train_model,
    objective_type="maximize"
)

optimizer = GeneticOptimizer(problem)
result = optimizer.run(max_generations=50)
```

### Example 2: Real-time Optimization Monitoring

```python
from darwin import GeneticOptimizer
from darwin.callbacks import ProgressCallback, PlotCallback

# Create callbacks for real-time monitoring
progress_cb = ProgressCallback(update_frequency=10)
plot_cb = PlotCallback(show_plots=True, save_plots=True)

optimizer = GeneticOptimizer(
    problem,
    callbacks=[progress_cb, plot_cb]
)

# Run with real-time visualization
result = optimizer.run(max_generations=200)
```

## 🎯 Next Steps

### Learn More
- **[Basic Concepts](concepts.md)** - Understand genetic algorithms
- **[Problem Definition](problem-definition.md)** - Define complex optimization problems
- **[Dashboard Guide](dashboard.md)** - Master the interactive interface
- **[API Usage](api-usage.md)** - Integrate Darwin into your applications

### Try Advanced Features
- **[Multi-Objective Optimization](../tutorials/multi-objective.md)** - Pareto frontier optimization
- **[Constraint Handling](../tutorials/constraints.md)** - Complex constraint satisfaction
- **[Custom Algorithms](../tutorials/custom-algorithms.md)** - Implement your own algorithms

### Deploy in Production
- **[Production Deployment](../operations/deployment.md)** - Deploy Darwin at scale
- **[Monitoring Guide](../operations/monitoring.md)** - Set up comprehensive monitoring
- **[Performance Tuning](../operations/performance.md)** - Optimize for your workload

## 🆘 Common Issues

### Installation Problems

**Issue**: `pip install darwin-genetic-solver` fails
```bash
# Solution: Update pip and try again
pip install --upgrade pip
pip install darwin-genetic-solver
```

**Issue**: Import errors with dependencies
```bash
# Solution: Install with all dependencies
pip install "darwin-genetic-solver[all]"
```

### Runtime Problems

**Issue**: Optimization is too slow
```python
# Solution: Reduce population size or use parallel execution
config = OptimizationConfig(
    population_size=50,  # Reduced from default 100
    parallel_execution=True,
    n_jobs=-1  # Use all CPU cores
)
```

**Issue**: Dashboard won't start
```bash
# Solution: Check port availability
darwin-dashboard --port 5007  # Try different port
```

## 💡 Tips for Success

1. **Start Simple**: Begin with built-in test functions before custom problems
2. **Monitor Progress**: Use the dashboard to visualize optimization progress
3. **Tune Parameters**: Adjust population size and mutation rate for your problem
4. **Use Callbacks**: Implement custom callbacks for advanced monitoring
5. **Parallel Execution**: Enable parallel processing for faster optimization
6. **Constraint Handling**: Use constraint penalties for constrained problems
7. **Multi-Objective**: Consider NSGA-II for problems with multiple objectives

## 🎯 Success Metrics

After completing this quick start, you should be able to:

- ✅ Install Darwin successfully
- ✅ Run your first optimization problem
- ✅ Use the interactive dashboard
- ✅ Access the MCP server
- ✅ Configure optimization parameters
- ✅ Analyze optimization results
- ✅ Integrate Darwin into your applications

Ready to dive deeper? Check out our [comprehensive tutorials](../tutorials/README.md) and [user guide](README.md)!

---

**Need Help?** Join our [Discord community](https://discord.gg/devqai) or [open an issue](https://github.com/devqai/darwin/issues) on GitHub.
