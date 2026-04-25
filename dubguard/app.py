"""
DubGuard FastAPI server — OpenEnv-compatible HTTP interface.

Endpoints:
  POST /reset          — start a new episode, return observation
  POST /step           — submit QC action, return reward + done
  GET  /state          — return current observation
  GET  /health         — liveness probe
  GET  /info           — environment metadata

Client code must never import this module directly; use client.py instead.
"""

import sys
import os

# Make dubguard/ importable regardless of working directory
sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from environment.env import DubGuardEnvironment

app = FastAPI(
    title="DubGuard",
    description=(
        "RL environment for LLM-based quality control of AI-dubbed video segments. "
        "An agent reads timing + text context and outputs BLOCK/WARN/PASS decisions "
        "with timing-safe fix suggestions."
    ),
    version="1.0.0",
)

_env = DubGuardEnvironment()


# ── request / response models ──────────────────────────────────────────────────

class ResetRequest(BaseModel):
    difficulty: Optional[str] = Field(
        None,
        description="Filter to 'easy', 'medium', or 'hard' episodes. "
                    "Omit to cycle through all episodes.",
    )


class QCAction(BaseModel):
    segment_id: int = Field(..., description="Numeric ID of the segment being reviewed.")
    error_type: str = Field(
        ...,
        description="One of: none, timing_collision, mistranslation, tone_mismatch, cultural_mismatch",
    )
    severity: str = Field(..., description="One of: PASS, WARN, BLOCK")
    reason: str = Field(..., description="Brief explanation for the decision.")
    suggested_fix: str = Field(
        ...,
        description="Rewritten dubbed text that fits within the timing window.",
    )
    estimated_fix_duration: float = Field(
        ...,
        description="Estimated spoken duration of suggested_fix in seconds.",
    )


# ── endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "episodes_loaded": len(_env)}


@app.get("/info")
def info():
    return {
        "name": "DubGuard",
        "version": "1.0.0",
        "observation_space": _env.observation_space(),
        "action_space": _env.action_space(),
        "reward_range": _env.reward_range(),
        "episodes": len(_env),
    }


@app.post("/reset")
def reset(req: ResetRequest = ResetRequest()):
    obs = _env.reset(difficulty=req.difficulty)
    return {"observation": obs}


@app.post("/step")
def step(action: QCAction):
    try:
        reward, done = _env.step(action.model_dump())
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"reward": reward, "done": done}


@app.get("/state")
def state():
    obs = _env.state()
    if obs is None:
        raise HTTPException(status_code=400, detail="Call /reset before /state.")
    return {"observation": obs}


# ── entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=False)
