# Darwin Dashboard

The Darwin Dashboard is a comprehensive web-based interface for the Darwin genetic algorithm optimization platform. Built with Panel and Bokeh, it provides an intuitive and powerful interface for creating, monitoring, and analyzing genetic algorithm optimization runs.

## 🚀 Features

### 📝 Problem Editor
- Interactive problem definition interface
- Visual variable configuration with constraints
- Code editor for fitness functions and constraints
- Real-time validation and testing
- Template support for common problems

### 📊 Real-time Monitoring
- Live optimization progress tracking
- Real-time fitness evolution plots
- Population diversity monitoring
- Performance metrics and system status
- WebSocket-based updates for instant feedback

### 📈 Advanced Visualizations
- Interactive Bokeh plots with custom tools
- Solution space exploration and clustering
- Pareto frontier visualization for multi-objective problems
- Performance comparison and benchmarking
- Statistical analysis and correlation plots

### 📚 Template Management
- Problem template library with categorization
- Template creation, editing, and sharing
- Version control and usage tracking
- Community templates and examples
- Import/export capabilities

### 🧪 Experiment Tracking
- Comprehensive experiment history
- Run comparison and analysis
- Experiment grouping and tagging
- Performance benchmarking across runs
- Detailed experiment documentation

## 🏗️ Architecture

### Component Structure
```
dashboard/
├── app.py                    # Main dashboard application
├── components/               # UI components
│   ├── problem_editor.py     # Problem definition interface
│   ├── monitoring.py         # Real-time monitoring
│   ├── visualizations.py     # Analytics and plotting
│   ├── templates.py          # Template management
│   └── experiments.py        # Experiment tracking
└── utils/                    # Utilities
    ├── api_client.py         # FastAPI backend client
    └── websocket_manager.py   # Real-time updates
```

### Technology Stack
- **Panel**: Main dashboard framework
- **Bokeh**: Interactive visualizations
- **FastAPI**: Backend API integration
- **WebSockets**: Real-time updates
- **Pandas**: Data manipulation
- **NumPy/SciPy**: Scientific computing
- **Plotly**: Additional plotting capabilities

## 🚀 Quick Start

### Prerequisites
- Python 3.8+ with Darwin project dependencies
- Darwin API server running on localhost:8000
- Modern web browser (Chrome, Firefox, Safari, Edge)

### Installation
The dashboard is included with the Darwin project installation:

```bash
cd devqai/darwin
pip install -e .
```

### Launch Dashboard

#### Option 1: Using the launcher script (recommended)
```bash
python scripts/launch_dashboard.py
```

#### Option 2: Direct Python execution
```bash
python -c "from darwin.dashboard import create_app; create_app().serve()"
```

#### Option 3: Custom configuration
```bash
python scripts/launch_dashboard.py \
  --port 8080 \
  --api-url http://localhost:8000 \
  --autoreload
```

### Accessing the Dashboard
Once started, the dashboard will be available at:
- **URL**: http://localhost:5007 (default)
- **Browser**: Automatically opens if not disabled
- **API**: Connects to http://localhost:8000 (default)

## 📋 Usage Guide

### Creating an Optimization Problem

1. **Navigate to Problem Editor**
   - Click "Problem Editor" in the sidebar
   - Or use the "New Optimization" quick action

2. **Define Problem Metadata**
   - Enter problem name and description
   - Select optimization type (minimize/maximize/multi-objective)

3. **Configure Variables**
   - Add variables with types (continuous/discrete/categorical)
   - Set bounds and descriptions
   - Use the interactive variable form

4. **Write Functions**
   - Define fitness function in Python
   - Add constraints if needed
   - Use the built-in code editor with syntax highlighting

5. **Configure Algorithm**
   - Set population size and generations
   - Choose selection, crossover, and mutation methods
   - Adjust rates and enable/disable elitism

6. **Validate and Run**
   - Click "Validate Problem" to check everything
   - Use "Create & Run" to start optimization

### Monitoring Optimization Runs

1. **Select Active Optimization**
   - Use the optimizer selector in Monitoring tab
   - View current status and progress

2. **Real-time Visualization**
   - Fitness Evolution: Track best/average/worst fitness
   - Population Diversity: Monitor genetic diversity
   - Performance Metrics: CPU, memory, evaluation time

3. **Control Optimization**
   - Start, pause, stop, or restart runs
   - Export results or view detailed analysis

### Analyzing Results

1. **Visualization Engine**
   - Select optimizers to compare
   - Choose visualization type
   - Generate interactive plots

2. **Statistical Analysis**
   - Run convergence analysis
   - Perform parameter sensitivity studies
   - Compare multiple optimization runs

3. **Export and Sharing**
   - Export plots in various formats
   - Generate comprehensive reports
   - Share results with team members

## ⚙️ Configuration

### Environment Variables
```bash
# Dashboard configuration
DARWIN_DASHBOARD_PORT=5007
DARWIN_DASHBOARD_HOST=0.0.0.0

# API configuration
DARWIN_API_URL=http://localhost:8000
DARWIN_API_TIMEOUT=30.0

# WebSocket configuration
DARWIN_WS_URL=ws://localhost:8000/ws/optimization/progress

# Logging
DARWIN_LOG_LEVEL=INFO
```

### Configuration File
Create a `config.yaml` file for advanced configuration:

```yaml
dashboard:
  port: 5007
  host: "0.0.0.0"
  title: "Darwin Genetic Algorithm Optimizer"
  show_browser: true
  autoreload: false

api:
  base_url: "http://localhost:8000"
  timeout: 30.0
  max_retries: 3

websocket:
  url: "ws://localhost:8000/ws/optimization/progress"
  max_reconnect_attempts: 10
  heartbeat_interval: 30.0

logging:
  level: "INFO"
  rich_logging: true
```

Use with: `python scripts/launch_dashboard.py --config config.yaml`

## 🔧 Development

### Development Mode
Enable development features for easier debugging:

```bash
python scripts/launch_dashboard.py --dev
```

This enables:
- Auto-reload on file changes
- Debug logging level
- Rich console output
- Enhanced error messages

### Adding Custom Components

1. **Create Component Class**
```python
import panel as pn
import param
from darwin.dashboard.utils import DarwinAPIClient

class CustomComponent(param.Parameterized):
    def __init__(self, api_client: DarwinAPIClient, **params):
        super().__init__(**params)
        self.api_client = api_client
        self._create_components()

    def create_interface(self):
        return pn.Column(...)
```

2. **Integrate with Main Dashboard**
```python
from darwin.dashboard.app import DarwinDashboard

# Add to dashboard tabs
dashboard.main_tabs.append(("Custom", custom_component.create_interface()))
```

### Extending Visualizations

1. **Add New Plot Types**
```python
def _generate_custom_plot(self, data):
    p = figure(title="Custom Visualization")
    # Add your Bokeh plotting code
    return pn.pane.Bokeh(p)
```

2. **Register Plot Type**
```python
# Add to visualization options
viz_options["Custom Plot"] = "custom_plot"
```

## 🐛 Troubleshooting

### Common Issues

#### Dashboard Won't Start
- Check that all dependencies are installed: `pip install -e .`
- Verify Python version is 3.8+: `python --version`
- Check port availability: `netstat -an | grep 5007`

#### Can't Connect to API
- Verify API server is running: `curl http://localhost:8000/health`
- Check firewall settings and network connectivity
- Confirm API URL configuration

#### WebSocket Connection Issues
- Check WebSocket URL in browser dev tools
- Verify API server supports WebSocket connections
- Try disabling browser extensions that might block WebSockets

#### Performance Issues
- Reduce update intervals in monitoring settings
- Limit the number of concurrent visualizations
- Check browser memory usage and restart if needed

### Debug Mode
Enable detailed debugging:

```bash
python scripts/launch_dashboard.py --log-level DEBUG --dev
```

### Getting Help
- Check logs in the console for error messages
- Visit the GitHub issues page for known problems
- Consult the Darwin documentation for API details

## 📚 API Integration

The dashboard integrates with the Darwin FastAPI backend through:

### REST API Endpoints
- `GET /api/v1/health` - Health check
- `POST /api/v1/optimizers` - Create optimizer
- `GET /api/v1/optimizers/{id}` - Get optimizer details
- `POST /api/v1/optimizers/{id}/run` - Start optimization
- `GET /api/v1/optimizers/{id}/results` - Get results

### WebSocket Endpoints
- `/ws/optimization/progress` - Real-time progress updates
- Message types: optimization_progress, optimization_complete, system_status

### Data Models
The dashboard uses Pydantic models that match the API:
- OptimizationProblem
- GeneticAlgorithm
- OptimizationResults
- ProgressUpdate

## 🚀 Deployment

### Production Deployment

1. **Environment Setup**
```bash
export DARWIN_DASHBOARD_PORT=80
export DARWIN_API_URL=https://api.your-domain.com
export DARWIN_LOG_LEVEL=WARNING
```

2. **Run with Production Settings**
```bash
python scripts/launch_dashboard.py \
  --no-browser \
  --host 0.0.0.0 \
  --port 80 \
  --api-url https://api.your-domain.com
```

3. **Behind Reverse Proxy**
Configure nginx or Apache to proxy requests to the dashboard.

### Docker Deployment
See the main Darwin documentation for Docker deployment instructions.

## 🤝 Contributing

### Code Style
- Follow PEP 8 style guidelines
- Use type hints for all function parameters and returns
- Add docstrings for all public methods
- Format code with Black: `black src/darwin/dashboard/`

### Testing
- Write tests for new components: `tests/dashboard/`
- Run tests: `pytest tests/dashboard/`
- Ensure test coverage: `pytest --cov=darwin.dashboard`

### Pull Requests
1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Update documentation as needed
5. Submit a pull request

## 📄 License

This project is licensed under the BSD 3-Clause License. See the LICENSE file for details.

## 🙏 Acknowledgments

- Panel and Bokeh teams for the excellent visualization frameworks
- FastAPI team for the modern web framework
- The genetic algorithms and optimization research community
- All contributors to the Darwin project
