"""
Darwin MCP Client

This module provides a client interface for interacting with the Darwin MCP server.
It simplifies the process of creating optimizers, running optimizations, and retrieving results
through the Model Context Protocol (MCP).
"""

import asyncio
import json
import logging
import uuid
from typing import Any, Dict, List, Optional, Union

import aiohttp
import websockets
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class MCPClientError(Exception):
    """Custom exception for MCP client errors"""

    pass


class MCPRequest(BaseModel):
    """MCP request model"""

    jsonrpc: str = Field("2.0", description="JSON-RPC version")
    id: Union[str, int] = Field(..., description="Request ID")
    method: str = Field(..., description="Method name")
    params: Optional[Dict[str, Any]] = Field(None, description="Method parameters")


class MCPResponse(BaseModel):
    """MCP response model"""

    jsonrpc: str = Field("2.0", description="JSON-RPC version")
    id: Union[str, int] = Field(..., description="Request ID")
    result: Optional[Dict[str, Any]] = Field(None, description="Success result")
    error: Optional[Dict[str, Any]] = Field(None, description="Error information")


class DarwinMCPClient:
    """Client for interacting with Darwin MCP server"""

    def __init__(self, base_url: str = "http://localhost:8001", timeout: int = 30):
        """
        Initialize MCP client

        Args:
            base_url: Base URL of the MCP server
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session: Optional[aiohttp.ClientSession] = None
        self.websocket: Optional[websockets.WebSocketServerProtocol] = None

    async def __aenter__(self):
        """Async context manager entry"""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.disconnect()

    async def connect(self):
        """Establish connection to MCP server"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout)
        )
        logger.info(f"Connected to MCP server at {self.base_url}")

    async def disconnect(self):
        """Close connection to MCP server"""
        if self.websocket:
            await self.websocket.close()
            self.websocket = None

        if self.session:
            await self.session.close()
            self.session = None

        logger.info("Disconnected from MCP server")

    async def _make_request(
        self, method: str, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Make MCP request to server"""
        if not self.session:
            raise MCPClientError(
                "Client not connected. Use async context manager or call connect() first."
            )

        request_id = str(uuid.uuid4())
        request = MCPRequest(id=request_id, method=method, params=params)

        try:
            async with self.session.post(
                f"{self.base_url}/mcp",
                json=request.dict(),
                headers={"Content-Type": "application/json"},
            ) as response:
                response.raise_for_status()
                data = await response.json()

                mcp_response = MCPResponse(**data)

                if mcp_response.error:
                    raise MCPClientError(f"MCP Error: {mcp_response.error}")

                return mcp_response.result or {}

        except aiohttp.ClientError as e:
            raise MCPClientError(f"HTTP request failed: {str(e)}")
        except Exception as e:
            raise MCPClientError(f"Request failed: {str(e)}")

    async def connect_websocket(self, callback: Optional[callable] = None):
        """Connect to WebSocket for real-time notifications"""
        try:
            ws_url = self.base_url.replace("http", "ws") + "/mcp/ws"
            self.websocket = await websockets.connect(ws_url)

            if callback:
                asyncio.create_task(self._handle_websocket_messages(callback))

            logger.info("Connected to WebSocket for real-time notifications")

        except Exception as e:
            raise MCPClientError(f"WebSocket connection failed: {str(e)}")

    async def _handle_websocket_messages(self, callback: callable):
        """Handle incoming WebSocket messages"""
        try:
            async for message in self.websocket:
                data = json.loads(message)
                await callback(data)
        except websockets.exceptions.ConnectionClosed:
            logger.info("WebSocket connection closed")
        except Exception as e:
            logger.error(f"WebSocket error: {str(e)}")

    async def list_tools(self) -> List[Dict[str, Any]]:
        """List available MCP tools"""
        try:
            async with self.session.get(f"{self.base_url}/mcp/tools") as response:
                response.raise_for_status()
                data = await response.json()
                return data.get("tools", [])
        except Exception as e:
            raise MCPClientError(f"Failed to list tools: {str(e)}")

    async def create_optimizer(
        self,
        problem_name: str,
        problem_description: str,
        variables: List[Dict[str, Any]],
        fitness_function: str,
        objective_type: str = "minimize",
        constraints: Optional[List[Dict[str, Any]]] = None,
        population_size: int = 50,
        max_generations: int = 100,
        selection_type: str = "tournament",
        crossover_type: str = "single_point",
        mutation_type: str = "uniform",
        crossover_probability: float = 0.8,
        mutation_probability: float = 0.1,
        elitism: bool = True,
    ) -> Dict[str, Any]:
        """
        Create a new genetic algorithm optimizer

        Args:
            problem_name: Name of the optimization problem
            problem_description: Description of the problem
            variables: List of variable definitions
            fitness_function: Python code for fitness function
            objective_type: Optimization objective ("minimize", "maximize", "multi_objective")
            constraints: List of constraint definitions
            population_size: GA population size
            max_generations: Maximum number of generations
            selection_type: Selection method
            crossover_type: Crossover method
            mutation_type: Mutation method
            crossover_probability: Crossover probability
            mutation_probability: Mutation probability
            elitism: Enable elitism

        Returns:
            Dictionary containing optimizer information
        """
        params = {
            "problem_name": problem_name,
            "problem_description": problem_description,
            "objective_type": objective_type,
            "variables": variables,
            "constraints": constraints or [],
            "fitness_function": fitness_function,
            "population_size": population_size,
            "max_generations": max_generations,
            "selection_type": selection_type,
            "crossover_type": crossover_type,
            "mutation_type": mutation_type,
            "crossover_probability": crossover_probability,
            "mutation_probability": mutation_probability,
            "elitism": elitism,
        }

        return await self._make_request("create_optimizer", params)

    async def run_optimization(
        self,
        optimizer_id: str,
        max_runtime_seconds: Optional[int] = None,
        save_history: bool = True,
        notify_progress: bool = False,
    ) -> Dict[str, Any]:
        """
        Run an optimization process

        Args:
            optimizer_id: Unique optimizer identifier
            max_runtime_seconds: Maximum runtime in seconds
            save_history: Save evolution history
            notify_progress: Send progress notifications

        Returns:
            Dictionary containing run status
        """
        params = {
            "optimizer_id": optimizer_id,
            "max_runtime_seconds": max_runtime_seconds,
            "save_history": save_history,
            "notify_progress": notify_progress,
        }

        return await self._make_request("run_optimization", params)

    async def get_results(
        self,
        optimizer_id: str,
        include_history: bool = False,
        include_population: bool = False,
    ) -> Dict[str, Any]:
        """
        Get optimization results

        Args:
            optimizer_id: Unique optimizer identifier
            include_history: Include evolution history
            include_population: Include final population

        Returns:
            Dictionary containing optimization results
        """
        params = {
            "optimizer_id": optimizer_id,
            "include_history": include_history,
            "include_population": include_population,
        }

        return await self._make_request("get_results", params)

    async def get_status(self, optimizer_id: str) -> Dict[str, Any]:
        """
        Get optimizer status

        Args:
            optimizer_id: Unique optimizer identifier

        Returns:
            Dictionary containing optimizer status
        """
        params = {"optimizer_id": optimizer_id}
        return await self._make_request("get_status", params)

    async def stop_optimization(self, optimizer_id: str) -> Dict[str, Any]:
        """
        Stop running optimization

        Args:
            optimizer_id: Unique optimizer identifier

        Returns:
            Dictionary containing stop status
        """
        params = {"optimizer_id": optimizer_id}
        return await self._make_request("stop_optimization", params)

    async def list_optimizers(self) -> Dict[str, Any]:
        """
        List all optimizers

        Returns:
            Dictionary containing list of optimizers
        """
        return await self._make_request("list_optimizers", {})

    async def visualize_evolution(
        self, optimizer_id: str, plot_type: str = "fitness", output_format: str = "json"
    ) -> Dict[str, Any]:
        """
        Generate evolution visualizations

        Args:
            optimizer_id: Unique optimizer identifier
            plot_type: Type of plot ("fitness", "diversity", "pareto", "convergence")
            output_format: Output format ("json", "base64_png")

        Returns:
            Dictionary containing visualization data
        """
        params = {
            "optimizer_id": optimizer_id,
            "plot_type": plot_type,
            "output_format": output_format,
        }

        return await self._make_request("visualize_evolution", params)

    async def compare_algorithms(
        self,
        optimizer_ids: List[str],
        metrics: Optional[List[str]] = None,
        output_format: str = "json",
    ) -> Dict[str, Any]:
        """
        Compare multiple algorithms

        Args:
            optimizer_ids: List of optimizer IDs to compare
            metrics: Metrics to compare
            output_format: Output format

        Returns:
            Dictionary containing comparison results
        """
        params = {
            "optimizer_ids": optimizer_ids,
            "metrics": metrics or ["best_fitness", "convergence_rate"],
            "output_format": output_format,
        }

        return await self._make_request("compare_algorithms", params)

    async def wait_for_completion(
        self,
        optimizer_id: str,
        poll_interval: float = 1.0,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Wait for optimization to complete

        Args:
            optimizer_id: Unique optimizer identifier
            poll_interval: Polling interval in seconds
            timeout: Maximum wait time in seconds

        Returns:
            Dictionary containing final results
        """
        start_time = asyncio.get_event_loop().time()

        while True:
            status = await self.get_status(optimizer_id)

            if status["status"] in ["completed", "failed", "stopped"]:
                return await self.get_results(optimizer_id)

            if timeout and (asyncio.get_event_loop().time() - start_time) > timeout:
                raise MCPClientError(f"Timeout waiting for optimization {optimizer_id}")

            await asyncio.sleep(poll_interval)


# Convenience functions for common use cases
async def create_simple_optimizer(
    client: DarwinMCPClient,
    problem_name: str,
    variables: List[Dict[str, Any]],
    fitness_function: str,
    **kwargs,
) -> str:
    """
    Create a simple optimizer with minimal configuration

    Args:
        client: MCP client instance
        problem_name: Name of the problem
        variables: Variable definitions
        fitness_function: Fitness function code
        **kwargs: Additional optimizer parameters

    Returns:
        Optimizer ID
    """
    result = await client.create_optimizer(
        problem_name=problem_name,
        problem_description=f"Optimization problem: {problem_name}",
        variables=variables,
        fitness_function=fitness_function,
        **kwargs,
    )

    return result["optimizer_id"]


async def run_simple_optimization(
    client: DarwinMCPClient,
    problem_name: str,
    variables: List[Dict[str, Any]],
    fitness_function: str,
    wait_for_completion: bool = True,
    **kwargs,
) -> Dict[str, Any]:
    """
    Run a complete optimization from start to finish

    Args:
        client: MCP client instance
        problem_name: Name of the problem
        variables: Variable definitions
        fitness_function: Fitness function code
        wait_for_completion: Wait for optimization to complete
        **kwargs: Additional optimizer parameters

    Returns:
        Optimization results
    """
    # Create optimizer
    optimizer_id = await create_simple_optimizer(
        client, problem_name, variables, fitness_function, **kwargs
    )

    # Run optimization
    await client.run_optimization(optimizer_id)

    # Wait for completion if requested
    if wait_for_completion:
        return await client.wait_for_completion(optimizer_id)
    else:
        return {"optimizer_id": optimizer_id, "status": "running"}


# Example usage
if __name__ == "__main__":

    async def example_usage():
        """Example of using the Darwin MCP client"""

        # Define a simple optimization problem
        variables = [
            {
                "name": "x",
                "type": "continuous",
                "bounds": [-10.0, 10.0],
                "encoding": "real",
            },
            {
                "name": "y",
                "type": "continuous",
                "bounds": [-10.0, 10.0],
                "encoding": "real",
            },
        ]

        # Simple sphere function
        fitness_function = """
def fitness(solution):
    x, y = solution
    return x**2 + y**2
"""

        # Use the client
        async with DarwinMCPClient() as client:
            # List available tools
            tools = await client.list_tools()
            print(f"Available tools: {[tool['name'] for tool in tools]}")

            # Run optimization
            results = await run_simple_optimization(
                client=client,
                problem_name="Sphere Function",
                variables=variables,
                fitness_function=fitness_function,
                population_size=30,
                max_generations=50,
            )

            print("Optimization completed!")
            print(f"Best fitness: {results['best_fitness']}")
            print(f"Best solution: {results['best_solution']}")

    # Run example
    asyncio.run(example_usage())
