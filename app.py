"""
IsoSync FastAPI server — OpenEnv-compatible HTTP interface.

Endpoints:
  POST /reset   — start a new episode, return first observation
  POST /step    — submit a translation, return (obs, reward, done, info)
  GET  /state   — return current environment state
  GET  /info    — environment metadata
  GET  /health  — liveness probe
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from environment import IsoSyncEnvironment

app = FastAPI(
    title="IsoSync",
    description=(
        "RL environment for training LLMs to generate isochronous video "
        "translations under cumulative timing budget constraints."
    ),
    version="0.1.0",
)

_env = IsoSyncEnvironment()


class ResetRequest(BaseModel):
    level: int = 1
    language: str = "Hindi"


class StepRequest(BaseModel):
    translation: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/info")
def info():
    return {
        "name": "IsoSync",
        "version": "0.1.0",
        "languages": ["Hindi", "Portuguese"],
        "curriculum_levels": [1, 2, 3],
        "reward_range": [-1.0, 1.0],
    }


@app.post("/reset")
def reset(req: ResetRequest = ResetRequest()):
    obs = _env.reset(level=req.level, language=req.language)
    return {"observation": obs, "state": _env.state()}


@app.post("/step")
def step(req: StepRequest):
    try:
        obs, reward, done, info = _env.step({"translation": req.translation})
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"observation": obs, "reward": reward, "done": done, "info": info}


@app.get("/state")
def state():
    return {"state": _env.state()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=False)
