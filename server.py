"""FastAPI server exposing AlzheimerEnv as an HTTP OpenEnv endpoint."""

import sys
import os

# Ensure local modules are importable
sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
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


class AgentRequest(BaseModel):
    env_state: dict
    system_prompt: str
    task_id: int = 1


# ---------------------------------------------------------------------------
# API Routes  (must be defined BEFORE the static catch-all)
# ---------------------------------------------------------------------------

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


@app.post("/agent")
def agent(request: AgentRequest):
    """Proxy LLM call from the UI — avoids browser CORS restrictions."""
    import json
    api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("HF_TOKEN", "")
    api_base = os.environ.get("API_BASE_URL", "https://api.anthropic.com")
    model = os.environ.get("MODEL_NAME", "claude-haiku-4-5-20251001")

    FALLBACKS = {
        1: {"action": "classify:AD", "reasoning": "High gene expression indicates AD", "confidence": 0.85},
        2: {"action": "rank:[APOE,APP,PSEN1]", "reasoning": "Top AD biomarkers by GWAS evidence", "confidence": 0.85},
        3: {"action": "downregulate:APOE", "reasoning": "APOE most overexpressed risk gene", "confidence": 0.85},
    }
    fallback = FALLBACKS.get(request.task_id, FALLBACKS[1])

    if not api_key:
        return fallback

    try:
        from openai import OpenAI
        base_url = api_base.rstrip("/")
        if not base_url.endswith("/v1"):
            base_url += "/v1"
        client = OpenAI(api_key=api_key, base_url=base_url)
        response = client.chat.completions.create(
            model=model,
            max_tokens=128,
            messages=[
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": f"State: {json.dumps(request.env_state)}"},
            ],
            extra_headers={"anthropic-version": "2023-06-01"},
        )
        raw = response.choices[0].message.content.strip()
        try:
            return json.loads(raw.replace("```json", "").replace("```", "").strip())
        except Exception:
            return fallback
    except Exception:
        return fallback


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
# Static files — served from ./static (React build output)
# Only mounted when the build exists (Docker); skipped in bare local dev.
# ---------------------------------------------------------------------------

_static_dir = os.path.join(os.path.dirname(__file__), "static")

if os.path.isdir(os.path.join(_static_dir, "assets")):
    app.mount("/assets", StaticFiles(directory=os.path.join(_static_dir, "assets")), name="assets")


# Catch-all: serve index.html for any unmatched path (React client-side routing)
@app.get("/{full_path:path}")
async def serve_frontend(full_path: str):
    index = os.path.join(_static_dir, "index.html")
    if os.path.isfile(index):
        return FileResponse(index)
    # Fallback when no static build present (local dev without Docker)
    return {"message": "AlzheimerEnv is running", "status": "ok"}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=7860, reload=False)
