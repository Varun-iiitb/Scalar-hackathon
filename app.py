"""
IsoSync FastAPI server — OpenEnv-compatible HTTP interface.

All observation, action, and reward payloads are validated against the
typed Pydantic models in schemas.py per the OpenEnv specification.

Endpoints:
  POST /reset   — start a new episode, return typed Observation
  POST /step    — submit Action, return typed StepResponse
  GET  /state   — return raw environment state (debugging)
  GET  /info    — typed EnvInfo metadata
  GET  /health  — liveness probe
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from environment import IsoSyncEnvironment
from data_gen import CURRICULUM
from schemas import (
    Observation, Action, Reward,
    ResetResponse, StepResponse,
    EnvInfo, CurriculumLevel,
)

app = FastAPI(
    title="IsoSync",
    description=(
        "OpenEnv-compatible RL environment for training LLMs to generate "
        "isochronous video translations under cumulative timing budget "
        "constraints with banking and locale awareness."
    ),
    version="0.2.0",
)

_env = IsoSyncEnvironment()


# ─── helpers ───────────────────────────────────────────────────────────────────

def _build_observation(prompt: str) -> Observation:
    """Wrap a raw prompt string into a typed Observation using env state."""
    state = _env.state()
    seg   = state["current_segment"] or {}
    return Observation(
        prompt                   = prompt,
        episode_id               = state.get("episode_id", "ep_unknown"),
        segment_id               = state["segments_completed"],
        segments_total           = state["total_segments"],
        segments_remaining       = state["total_segments"] - state["segments_completed"],
        original_text            = seg.get("original_text", ""),
        target_language          = state["language"],
        locale                   = seg.get("locale", ""),
        time_window_seconds      = seg.get("original_duration", 0.0),
        max_syllables            = seg.get("max_syllables", 0),
        budget_remaining_seconds = state["budget_remaining"],
        budget_bank_seconds      = state["budget_bank"],
        budget_deficit_seconds   = state["budget_deficit"],
        previous_translation     = state.get("prev_translation"),
        curriculum_level         = state["current_level"],
    )


def _build_reward(info: dict) -> Reward:
    return Reward(
        combined           = info.get("combined_reward",  0.0),
        timing             = info.get("timing_reward",    0.0),
        semantic           = info.get("semantic_reward",  0.0),
        budget             = info.get("budget_reward",    0.0),
        locale             = info.get("locale_reward",    0.0),
        coherence          = info.get("coherence_reward", 0.0),
        estimated_duration = info.get("estimated_duration", 0.0),
    )


# ─── request models ────────────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    level: int = Field(1, ge=1, le=4, description="Curriculum level (1-4)")


# ─── endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/info", response_model=EnvInfo)
def info():
    return EnvInfo(
        description=(
            "RL environment for isochronous video translation. The agent "
            "translates English video segments to Portuguese under a strict "
            "per-segment time window and a shared cumulative budget. Under-runs "
            "save time into a bank that offsets future overruns, creating "
            "genuine strategic depth."
        ),
        languages=["Portuguese"],
        curriculum=[
            CurriculumLevel(
                level=k,
                n_segments=v["n_segments"],
                duration_slack=v["duration_slack"],
                locale_constraints=v["locale_constraints"],
                language=v["language"],
            )
            for k, v in CURRICULUM.items()
        ],
        tags=["dubbing", "translation", "timing", "isochrony", "world-modeling"],
    )


@app.post("/reset", response_model=ResetResponse)
def reset(req: ResetRequest = ResetRequest()):
    if req.level not in CURRICULUM:
        raise HTTPException(
            status_code=400,
            detail=f"level must be one of {list(CURRICULUM)}",
        )
    prompt = _env.reset(level=req.level)
    return ResetResponse(observation=_build_observation(prompt))


@app.post("/step", response_model=StepResponse)
def step(action: Action):
    try:
        obs_str, reward_val, done, info = _env.step({"translation": action.translation})
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    reward = _build_reward(info)
    obs    = None if done else _build_observation(obs_str)
    return StepResponse(observation=obs, reward=reward, done=done, info=info)


@app.get("/state")
def state():
    return {"state": _env.state()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=False)
