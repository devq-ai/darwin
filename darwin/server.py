# In darwin/server.py
import os
import sys
import uvicorn
import logfire
from fastapi import FastAPI, Request
from pydantic import BaseModel
import asyncio

# Check if Logfire token is available
logfire_token = os.getenv("LOGFIRE_TOKEN")
if logfire_token:
    try:
        # New Logfire configuration format
        logfire.configure(
            token=logfire_token,
            service_name="darwin-mcp",
            service_version="1.0.0"
        )
        print("Logfire configured successfully")
    except Exception as e:
        print(f"Logfire configuration failed: {str(e)}")
        # Proceed without Logfire
        logfire = None
else:
    print("LOGFIRE_TOKEN not found, proceeding without Logfire")
    logfire = None

app = FastAPI(title="Darwin MCP Server")

if logfire:
    logfire.instrument_fastapi(app)
else:
    print("Skipping FastAPI instrumentation")

class OptimizationRequest(BaseModel):
    name: str
    description: str
    variables: list
    objectives: list
    constraints: list = []
    config: dict = {}

@app.post("/create_optimization")
async def create_optimization(request: OptimizationRequest):
    return {"optimization_id": "opt_123", "status": "created"}

@app.post("/start_optimization/{optimization_id}")
async def start_optimization(optimization_id: str):
    await asyncio.sleep(2)  # Simulate work
    return {"status": "running", "optimization_id": optimization_id}

@app.get("/status/{optimization_id}")
async def get_status(optimization_id: str):
    return {"status": "completed", "generations": 100, "fitness": 0.95}

@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "darwin-mcp"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
