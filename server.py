"""FastAPI server exposing AlzheimerEnv as an HTTP OpenEnv endpoint."""

import sys
import os

# Ensure local modules are importable
sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn

from environment import AlzheimerEnv

app = FastAPI(
    title="AlzheimerEnv",
    description="OpenEnv RL environment for Alzheimer's disease prediction and intervention.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single global environment instance
env = AlzheimerEnv()


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id: int = 1
    patient_id: Optional[str] = None


class StepRequest(BaseModel):
    action: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    return {"message": "AlzheimerEnv is running", "status": "ok"}


@app.get("/health")
def health():
    return {"status": "ok", "env": "AlzheimerEnv", "version": "1.0.0"}


@app.get("/tasks")
def list_tasks():
    return env.list_tasks()


@app.get("/state")
def get_state():
    try:
        return env.state()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reset")
def reset(request: ResetRequest):
    try:
        state = env.reset(task_id=request.task_id, patient_id=request.patient_id)
        return state
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/step")
def step(request: StepRequest):
    try:
        result = env.step(request.action)
        return result
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=7860, reload=False)
