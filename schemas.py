"""
IsoSync OpenEnv typed schemas.

Pydantic models defining the typed Observation, Action, and Reward envelopes
required by the OpenEnv specification. These power both the FastAPI endpoint
validation and the auto-generated /openapi.json schema document.
"""

from typing import Optional
from pydantic import BaseModel, Field


# ─── OBSERVATION ───────────────────────────────────────────────────────────────

class Observation(BaseModel):
    """
    Per-segment environment observation.

    `prompt` is the LLM-ready formatted prompt; the remaining fields expose
    the same context as machine-readable data so external agents can build
    their own prompts.
    """
    prompt:                   str   = Field(..., description="LLM-ready prompt with all context")

    episode_id:               str   = Field(..., description="Unique id of the current episode")
    segment_id:               int   = Field(..., description="0-based index of the current segment")
    segments_total:           int   = Field(..., description="Total segments in this episode")
    segments_remaining:       int   = Field(..., description="Segments left to translate")

    original_text:            str   = Field(..., description="English source text for this segment")
    target_language:          str   = Field(..., description="Target language (e.g. Portuguese)")
    locale:                   str   = Field(..., description="Target locale (e.g. Brazil)")

    time_window_seconds:      float = Field(..., description="Hard target duration for the spoken dub")
    max_syllables:            int   = Field(..., description="Syllable cap derived from speaking rate")

    budget_remaining_seconds: float = Field(..., description="Seconds left in the episode budget")
    budget_bank_seconds:      float = Field(..., description="Saved time from earlier under-runs")
    budget_deficit_seconds:   float = Field(..., description="Cumulative overrun so far")

    previous_translation:     Optional[str] = Field(None, description="Previous step's translation, if any")
    curriculum_level:         int   = Field(..., description="Current curriculum level (1-4)")


# ─── ACTION ────────────────────────────────────────────────────────────────────

class Action(BaseModel):
    """One translation submission for the current segment."""
    translation: str = Field(..., min_length=1, description="Translated text in the target language")


# ─── REWARD ────────────────────────────────────────────────────────────────────

class Reward(BaseModel):
    """
    Decomposed reward returned by step().

    `combined` is the scalar signal used by the policy gradient; the
    component fields expose the breakdown for analysis. Components are in
    [0.0, 1.0]; combined is in [-1.0, 1.0].
    """
    combined:           float = Field(..., ge=-1.0, le=1.0, description="Weighted combined reward [-1, 1]")
    timing:             float = Field(..., ge=0.0,  le=1.0, description="Fits the time window")
    semantic:           float = Field(..., ge=0.0,  le=1.0, description="chrF similarity to references")
    budget:             float = Field(..., ge=0.0,  le=1.0, description="Cumulative budget management")
    locale:             float = Field(..., ge=0.0,  le=1.0, description="Locale-appropriate word usage")
    coherence:          float = Field(..., ge=0.0,  le=1.0, description="Numeric/entity consistency across episode")
    estimated_duration: float = Field(..., description="Estimated spoken duration of the translation (s)")


# ─── ENDPOINT RESPONSE ENVELOPES ───────────────────────────────────────────────

class StepResponse(BaseModel):
    observation: Optional[Observation] = Field(None, description="Next observation, or null if done")
    reward:      Reward                = Field(..., description="Reward for the action just taken")
    done:        bool                  = Field(..., description="True when the episode has ended")
    info:        dict                  = Field(default_factory=dict, description="Auxiliary diagnostics")


class ResetResponse(BaseModel):
    observation: Observation = Field(..., description="Initial observation for the new episode")


# ─── INFO / METADATA ───────────────────────────────────────────────────────────

class CurriculumLevel(BaseModel):
    level:              int
    n_segments:         int
    duration_slack:     float
    locale_constraints: bool
    language:           str


class EnvInfo(BaseModel):
    name:              str         = "IsoSync"
    version:           str         = "0.2.0"
    description:       str
    observation_space: str         = "Observation (Pydantic schema; see /openapi.json)"
    action_space:      str         = "Action (Pydantic schema; see /openapi.json)"
    reward_range:      list[float] = [-1.0, 1.0]
    languages:         list[str]
    curriculum:        list[CurriculumLevel]
    tags:              list[str]
