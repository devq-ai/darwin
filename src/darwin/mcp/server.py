"""
Darwin MCP Server Implementation

This module implements the Model Context Protocol (MCP) server for the Darwin genetic algorithm platform.
It provides tools for creating, running, and monitoring genetic algorithm optimizations through the MCP protocol.

The server exposes the following tools:
- create_optimizer: Initialize a new genetic algorithm optimizer
- run_optimization: Execute an optimization process
- get_results: Retrieve optimization results
- visualize_evolution: Generate evolution plots
- compare_algorithms: Algorithm comparison utilities
- get_status: Get optimizer status and progress
- stop_optimization: Stop a running optimization
- list_optimizers: List all available optimizers
"""

import asyncio
import json
import logging
import uuid
from datetime import UTC, datetime
from typing import Any, Dict, List, Optional, Union

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field, field_validator

from darwin.core.optimizer import GeneticOptimizer
from darwin.core.problem import Constraint, OptimizationProblem, Variable
from darwin.db.manager import DatabaseManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# MCP Protocol Models
class MCPRequest(BaseModel):
    """Base MCP request model following JSON-RPC 2.0 specification"""

    jsonrpc: str = Field("2.0", description="JSON-RPC version")
    id: Union[str, int] = Field(..., description="Request ID")
    method: str = Field(..., description="Method name")
    params: Optional[Dict[str, Any]] = Field(None, description="Method parameters")


class MCPResponse(BaseModel):
    """Base MCP response model following JSON-RPC 2.0 specification"""

    jsonrpc: str = Field("2.0", description="JSON-RPC version")
    id: Union[str, int] = Field(..., description="Request ID")
    result: Optional[Dict[str, Any]] = Field(None, description="Success result")
    error: Optional[Dict[str, Any]] = Field(None, description="Error information")


class MCPError(BaseModel):
    """MCP error model"""

    code: int = Field(..., description="Error code")
    message: str = Field(..., description="Error message")
    data: Optional[Dict[str, Any]] = Field(None, description="Additional error data")


# Tool-specific models
class CreateOptimizerParams(BaseModel):
    """Parameters for create_optimizer tool"""

    problem_name: str = Field(..., description="Name of the optimization problem")
    problem_description: str = Field(..., description="Description of the problem")
    objective_type: str = Field("minimize", description="Optimization objective type")
    variables: List[Dict[str, Any]] = Field(
        ..., description="Problem variables definition"
    )
    constraints: List[Dict[str, Any]] = Field(
        default=[], description="Problem constraints"
    )
    fitness_function: str = Field(..., description="Python code for fitness function")
    population_size: int = Field(50, description="GA population size")
    max_generations: int = Field(100, description="Maximum number of generations")
    selection_type: str = Field("tournament", description="Selection method")
    crossover_type: str = Field("single_point", description="Crossover method")
    mutation_type: str = Field("uniform", description="Mutation method")
    crossover_probability: float = Field(0.8, description="Crossover probability")
    mutation_probability: float = Field(0.1, description="Mutation probability")
    elitism: bool = Field(True, description="Enable elitism")

    @field_validator("objective_type")
    @classmethod
    def validate_objective_type(cls, v):
        if v not in ["minimize", "maximize", "multi_objective"]:
            raise ValueError(
                "objective_type must be 'minimize', 'maximize', or 'multi_objective'"
            )
        return v


class RunOptimizationParams(BaseModel):
    """Parameters for run_optimization tool"""

    optimizer_id: str = Field(..., description="Unique optimizer identifier")
    max_runtime_seconds: Optional[int] = Field(
        None, description="Maximum runtime in seconds"
    )
    save_history: bool = Field(True, description="Save evolution history")
    notify_progress: bool = Field(False, description="Send progress notifications")


class GetResultsParams(BaseModel):
    """Parameters for get_results tool"""

    optimizer_id: str = Field(..., description="Unique optimizer identifier")
    include_history: bool = Field(False, description="Include evolution history")
    include_population: bool = Field(False, description="Include final population")


class VisualizeEvolutionParams(BaseModel):
    """Parameters for visualize_evolution tool"""

    optimizer_id: str = Field(..., description="Unique optimizer identifier")
    plot_type: str = Field("fitness", description="Type of plot to generate")
    output_format: str = Field("json", description="Output format (json, base64_png)")

    @field_validator("plot_type")
    @classmethod
    def validate_plot_type(cls, v):
        valid_types = ["fitness", "diversity", "pareto", "convergence", "population"]
        if v not in valid_types:
            raise ValueError(f"plot_type must be one of {valid_types}")
        return v


class CompareAlgorithmsParams(BaseModel):
    """Parameters for compare_algorithms tool"""

    optimizer_ids: List[str] = Field(
        ..., description="List of optimizer IDs to compare"
    )
    metrics: List[str] = Field(
        ["best_fitness", "convergence_rate"], description="Metrics to compare"
    )
    output_format: str = Field("json", description="Output format")


class MCPServer:
    """MCP Server for Darwin genetic algorithm tools"""

    def __init__(self):
        """Initialize MCP server"""
        self.app = FastAPI(title="Darwin MCP Server", version="1.0.0")
        self.db = DatabaseManager()
        self.optimizers: Dict[str, GeneticOptimizer] = {}
        self.optimization_tasks: Dict[str, asyncio.Task] = {}
        self.websocket_connections: List[WebSocket] = []

        # Register routes
        self._register_routes()

        # Add startup event to connect to database
        @self.app.on_event("startup")
        async def startup_event():
            await self.db.connect()

    def _register_routes(self):
        """Register FastAPI routes for MCP endpoints"""

        @self.app.post("/mcp")
        async def handle_mcp_request(request: MCPRequest) -> MCPResponse:
            """Handle MCP JSON-RPC requests"""
            try:
                result = await self._dispatch_method(
                    request.method, request.params or {}
                )
                return MCPResponse(id=request.id, result=result)
            except Exception as e:
                logger.error(f"Error handling MCP request {request.method}: {str(e)}")
                error = MCPError(
                    code=-32603,  # Internal error
                    message=str(e),
                    data={"method": request.method, "params": request.params},
                )
                return MCPResponse(id=request.id, error=error.model_dump())

        @self.app.websocket("/mcp/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time notifications"""
            await websocket.accept()
            self.websocket_connections.append(websocket)
            try:
                while True:
                    data = await websocket.receive_text()
                    request = MCPRequest.model_validate_json(data)
                    result = await self._dispatch_method(
                        request.method, request.params or {}
                    )
                    response = MCPResponse(id=request.id, result=result)
                    await websocket.send_text(response.model_dump_json())
            except WebSocketDisconnect:
                self.websocket_connections.remove(websocket)
            except Exception as e:
                logger.error(f"WebSocket error: {str(e)}")
                if websocket in self.websocket_connections:
                    self.websocket_connections.remove(websocket)

        @self.app.get("/mcp/tools")
        async def list_tools():
            """List available MCP tools"""
            return {
                "tools": [
                    {
                        "name": "create_optimizer",
                        "description": "Create a new genetic algorithm optimizer",
                        "inputSchema": CreateOptimizerParams.model_json_schema(),
                    },
                    {
                        "name": "run_optimization",
                        "description": "Run an optimization process",
                        "inputSchema": RunOptimizationParams.model_json_schema(),
                    },
                    {
                        "name": "get_results",
                        "description": "Get optimization results",
                        "inputSchema": GetResultsParams.model_json_schema(),
                    },
                    {
                        "name": "visualize_evolution",
                        "description": "Generate evolution visualizations",
                        "inputSchema": VisualizeEvolutionParams.model_json_schema(),
                    },
                    {
                        "name": "compare_algorithms",
                        "description": "Compare multiple algorithms",
                        "inputSchema": CompareAlgorithmsParams.model_json_schema(),
                    },
                    {
                        "name": "get_status",
                        "description": "Get optimizer status",
                        "inputSchema": {
                            "type": "object",
                            "properties": {"optimizer_id": {"type": "string"}},
                        },
                    },
                    {
                        "name": "stop_optimization",
                        "description": "Stop running optimization",
                        "inputSchema": {
                            "type": "object",
                            "properties": {"optimizer_id": {"type": "string"}},
                        },
                    },
                    {
                        "name": "list_optimizers",
                        "description": "List all optimizers",
                        "inputSchema": {"type": "object", "properties": {}},
                    },
                ]
            }

    async def _dispatch_method(
        self, method: str, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Dispatch MCP method to appropriate handler"""
        handlers = {
            "create_optimizer": self._create_optimizer,
            "run_optimization": self._run_optimization,
            "get_results": self._get_results,
            "visualize_evolution": self._visualize_evolution,
            "compare_algorithms": self._compare_algorithms,
            "get_status": self._get_status,
            "stop_optimization": self._stop_optimization,
            "list_optimizers": self._list_optimizers,
        }

        if method not in handlers:
            raise ValueError(f"Unknown method: {method}")

        return await handlers[method](params)

    async def _create_optimizer(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new genetic algorithm optimizer"""
        try:
            # Validate parameters
            create_params = CreateOptimizerParams(**params)

            # Create problem definition
            variables = []
            for var_data in create_params.variables:
                variable = Variable(
                    name=var_data["name"],
                    type=var_data["type"],
                    bounds=tuple(var_data["bounds"]) if "bounds" in var_data else None,
                    gene_space=var_data.get("gene_space"),
                    encoding=var_data.get("encoding", "real"),
                )
                variables.append(variable)

            constraints = []
            for const_data in create_params.constraints:
                constraint = Constraint(
                    name=const_data["name"],
                    type=const_data["type"],
                    expression=const_data["expression"],
                )
                constraints.append(constraint)

            # Create fitness function
            fitness_function = self._create_fitness_function(
                create_params.fitness_function
            )

            problem = OptimizationProblem(
                name=create_params.problem_name,
                description=create_params.problem_description,
                objective_type=create_params.objective_type,
                variables=variables,
                constraints=constraints,
                fitness_function=fitness_function,
            )

            # Create optimizer
            optimizer_id = str(uuid.uuid4())
            optimizer = GeneticOptimizer(
                problem=problem,
                population_size=create_params.population_size,
                max_generations=create_params.max_generations,
                selection_type=create_params.selection_type,
                crossover_type=create_params.crossover_type,
                mutation_type=create_params.mutation_type,
                crossover_probability=create_params.crossover_probability,
                mutation_probability=create_params.mutation_probability,
                elitism=create_params.elitism,
            )

            # Store optimizer
            self.optimizers[optimizer_id] = optimizer
            await self.db.store_optimizer(optimizer_id, optimizer)

            logger.info(f"Created optimizer {optimizer_id}")

            return {
                "optimizer_id": optimizer_id,
                "status": "created",
                "problem_name": create_params.problem_name,
                "config": {
                    "population_size": create_params.population_size,
                    "max_generations": create_params.max_generations,
                    "selection_type": create_params.selection_type,
                    "crossover_type": create_params.crossover_type,
                    "mutation_type": create_params.mutation_type,
                },
                "created_at": datetime.now(UTC).isoformat(),
            }

        except Exception as e:
            logger.error(f"Error creating optimizer: {str(e)}")
            raise HTTPException(
                status_code=400, detail=f"Failed to create optimizer: {str(e)}"
            )

    async def _run_optimization(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Run an optimization process"""
        try:
            run_params = RunOptimizationParams(**params)
            optimizer_id = run_params.optimizer_id

            if optimizer_id not in self.optimizers:
                raise ValueError(f"Optimizer {optimizer_id} not found")

            optimizer = self.optimizers[optimizer_id]

            # Create optimization task
            task = asyncio.create_task(
                self._run_optimization_task(optimizer_id, optimizer, run_params)
            )
            self.optimization_tasks[optimizer_id] = task

            logger.info(f"Started optimization {optimizer_id}")

            return {
                "optimizer_id": optimizer_id,
                "status": "running",
                "started_at": datetime.now(UTC).isoformat(),
            }

        except Exception as e:
            logger.error(f"Error running optimization: {str(e)}")
            raise HTTPException(
                status_code=400, detail=f"Failed to run optimization: {str(e)}"
            )

    async def _run_optimization_task(
        self,
        optimizer_id: str,
        optimizer: GeneticOptimizer,
        params: RunOptimizationParams,
    ):
        """Background task for running optimization"""
        try:
            # Run optimization
            start_time = datetime.now(UTC)
            results = await optimizer.optimize()
            end_time = datetime.now(UTC)

            # Store results
            result_data = {
                "optimizer_id": optimizer_id,
                "status": "completed",
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "best_fitness": float(results.best_fitness),
                "best_solution": results.best_solution.tolist(),
                "generations_completed": results.generations,
                "total_evaluations": results.total_evaluations,
                "convergence_achieved": results.convergence_achieved,
                "execution_time": (end_time - start_time).total_seconds(),
            }

            await self.db.store_optimizer_results(optimizer_id, result_data)

            # Notify WebSocket clients
            await self._notify_clients(
                {
                    "type": "optimization_completed",
                    "optimizer_id": optimizer_id,
                    "results": result_data,
                }
            )

            logger.info(f"Completed optimization {optimizer_id}")

        except Exception as e:
            logger.error(f"Error in optimization task {optimizer_id}: {str(e)}")
            await self._notify_clients(
                {
                    "type": "optimization_failed",
                    "optimizer_id": optimizer_id,
                    "error": str(e),
                }
            )
        finally:
            # Clean up task
            if optimizer_id in self.optimization_tasks:
                del self.optimization_tasks[optimizer_id]

    async def _get_results(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get optimization results"""
        try:
            result_params = GetResultsParams(**params)
            optimizer_id = result_params.optimizer_id

            # Get results from database
            results = await self.db.get_optimizer_results(optimizer_id)

            if not results:
                return {
                    "optimizer_id": optimizer_id,
                    "status": "not_found",
                    "message": "No results found for this optimizer",
                }

            response = {
                "optimizer_id": optimizer_id,
                "status": results.get("status", "unknown"),
                "best_fitness": results.get("best_fitness"),
                "best_solution": results.get("best_solution"),
                "generations_completed": results.get("generations_completed"),
                "total_evaluations": results.get("total_evaluations"),
                "execution_time": results.get("execution_time"),
            }

            if result_params.include_history:
                history = await self.db.get_optimization_history(optimizer_id)
                response["history"] = history

            if result_params.include_population:
                population = await self.db.get_final_population(optimizer_id)
                response["final_population"] = population

            return response

        except Exception as e:
            logger.error(f"Error getting results: {str(e)}")
            raise HTTPException(
                status_code=500, detail=f"Failed to get results: {str(e)}"
            )

    async def _visualize_evolution(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate evolution visualizations"""
        try:
            viz_params = VisualizeEvolutionParams(**params)
            optimizer_id = viz_params.optimizer_id

            # Get optimization history
            history = await self.db.get_optimization_history(optimizer_id)

            if not history:
                return {
                    "optimizer_id": optimizer_id,
                    "status": "no_data",
                    "message": "No history data available for visualization",
                }

            # Generate visualization based on plot type
            if viz_params.plot_type == "fitness":
                plot_data = self._create_fitness_plot(history)
            elif viz_params.plot_type == "diversity":
                plot_data = self._create_diversity_plot(history)
            elif viz_params.plot_type == "convergence":
                plot_data = self._create_convergence_plot(history)
            else:
                plot_data = {"error": f"Unsupported plot type: {viz_params.plot_type}"}

            return {
                "optimizer_id": optimizer_id,
                "plot_type": viz_params.plot_type,
                "data": plot_data,
                "format": viz_params.output_format,
            }

        except Exception as e:
            logger.error(f"Error creating visualization: {str(e)}")
            raise HTTPException(
                status_code=500, detail=f"Failed to create visualization: {str(e)}"
            )

    async def _compare_algorithms(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Compare multiple algorithms"""
        try:
            compare_params = CompareAlgorithmsParams(**params)

            comparison_data = {}
            for optimizer_id in compare_params.optimizer_ids:
                results = await self.db.get_optimizer_results(optimizer_id)
                if results:
                    comparison_data[optimizer_id] = {
                        metric: results.get(metric) for metric in compare_params.metrics
                    }

            return {
                "comparison": comparison_data,
                "metrics": compare_params.metrics,
                "summary": self._generate_comparison_summary(
                    comparison_data, compare_params.metrics
                ),
            }

        except Exception as e:
            logger.error(f"Error comparing algorithms: {str(e)}")
            raise HTTPException(
                status_code=500, detail=f"Failed to compare algorithms: {str(e)}"
            )

    async def _get_status(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get optimizer status"""
        try:
            optimizer_id = params["optimizer_id"]

            if optimizer_id in self.optimization_tasks:
                task = self.optimization_tasks[optimizer_id]
                if task.done():
                    status = "completed" if not task.exception() else "failed"
                else:
                    status = "running"
            elif optimizer_id in self.optimizers:
                status = "ready"
            else:
                status = "not_found"

            return {
                "optimizer_id": optimizer_id,
                "status": status,
                "timestamp": datetime.now(UTC).isoformat(),
            }

        except Exception as e:
            logger.error(f"Error getting status: {str(e)}")
            raise HTTPException(
                status_code=500, detail=f"Failed to get status: {str(e)}"
            )

    async def _stop_optimization(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Stop running optimization"""
        try:
            optimizer_id = params["optimizer_id"]

            if optimizer_id in self.optimization_tasks:
                task = self.optimization_tasks[optimizer_id]
                task.cancel()
                del self.optimization_tasks[optimizer_id]

                return {
                    "optimizer_id": optimizer_id,
                    "status": "stopped",
                    "timestamp": datetime.now(UTC).isoformat(),
                }
            else:
                return {
                    "optimizer_id": optimizer_id,
                    "status": "not_running",
                    "message": "No running optimization found for this optimizer",
                }

        except Exception as e:
            logger.error(f"Error stopping optimization: {str(e)}")
            raise HTTPException(
                status_code=500, detail=f"Failed to stop optimization: {str(e)}"
            )

    async def _list_optimizers(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """List all optimizers"""
        try:
            optimizers_list = []

            for optimizer_id, optimizer in self.optimizers.items():
                status = "ready"
                if optimizer_id in self.optimization_tasks:
                    task = self.optimization_tasks[optimizer_id]
                    if task.done():
                        status = "completed" if not task.exception() else "failed"
                    else:
                        status = "running"

                optimizers_list.append(
                    {
                        "optimizer_id": optimizer_id,
                        "problem_name": optimizer.problem.name,
                        "status": status,
                        "population_size": optimizer.ga_config.population_size,
                        "max_generations": optimizer.ga_config.max_generations,
                    }
                )

            return {"optimizers": optimizers_list, "total_count": len(optimizers_list)}

        except Exception as e:
            logger.error(f"Error listing optimizers: {str(e)}")
            raise HTTPException(
                status_code=500, detail=f"Failed to list optimizers: {str(e)}"
            )

    def _create_fitness_function(self, code: str) -> callable:
        """Create fitness function from code string"""
        # Create a safe execution environment
        safe_globals = {
            "__builtins__": {},
            "abs": abs,
            "min": min,
            "max": max,
            "sum": sum,
            "len": len,
            "range": range,
            "enumerate": enumerate,
            "zip": zip,
        }

        # Allow math operations
        import math

        safe_globals.update(
            {
                "math": math,
                "sqrt": math.sqrt,
                "sin": math.sin,
                "cos": math.cos,
                "exp": math.exp,
                "log": math.log,
                "pow": math.pow,
                "pi": math.pi,
                "e": math.e,
            }
        )

        # Execute code to create function
        exec(code, safe_globals)

        # Return the fitness function (assuming it's named 'fitness')
        if "fitness" not in safe_globals:
            raise ValueError(
                "Fitness function code must define a function named 'fitness'"
            )

        return safe_globals["fitness"]

    def _create_fitness_plot(self, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create fitness evolution plot data"""
        generations = [h["generation"] for h in history]
        best_fitness = [h["best_fitness"] for h in history]
        avg_fitness = [h.get("avg_fitness", 0) for h in history]

        return {
            "x": generations,
            "y_best": best_fitness,
            "y_avg": avg_fitness,
            "title": "Fitness Evolution",
            "x_label": "Generation",
            "y_label": "Fitness",
        }

    def _create_diversity_plot(self, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create diversity plot data"""
        generations = [h["generation"] for h in history]
        diversity = [h.get("diversity", 0) for h in history]

        return {
            "x": generations,
            "y": diversity,
            "title": "Population Diversity",
            "x_label": "Generation",
            "y_label": "Diversity",
        }

    def _create_convergence_plot(self, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create convergence plot data"""
        generations = [h["generation"] for h in history]
        convergence_rate = []

        for i, h in enumerate(history):
            if i == 0:
                convergence_rate.append(0)
            else:
                current_fitness = h["best_fitness"]
                prev_fitness = history[i - 1]["best_fitness"]
                rate = abs(current_fitness - prev_fitness) / (abs(prev_fitness) + 1e-10)
                convergence_rate.append(rate)

        return {
            "x": generations,
            "y": convergence_rate,
            "title": "Convergence Rate",
            "x_label": "Generation",
            "y_label": "Convergence Rate",
        }

    def _generate_comparison_summary(
        self, data: Dict[str, Dict[str, Any]], metrics: List[str]
    ) -> Dict[str, Any]:
        """Generate comparison summary statistics"""
        summary = {}

        for metric in metrics:
            values = [
                opt_data.get(metric)
                for opt_data in data.values()
                if opt_data.get(metric) is not None
            ]
            if values:
                summary[metric] = {
                    "best": min(values) if metric.endswith("fitness") else max(values),
                    "worst": max(values) if metric.endswith("fitness") else min(values),
                    "average": sum(values) / len(values),
                    "count": len(values),
                }

        return summary

    async def _notify_clients(self, message: Dict[str, Any]):
        """Send notification to all connected WebSocket clients"""
        if self.websocket_connections:
            disconnected = []
            for websocket in self.websocket_connections:
                try:
                    await websocket.send_text(json.dumps(message))
                except Exception:
                    disconnected.append(websocket)

            # Remove disconnected clients
            for websocket in disconnected:
                self.websocket_connections.remove(websocket)


# Create MCP server instance
mcp_server = MCPServer()
app = mcp_server.app

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")
