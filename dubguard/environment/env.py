"""
DubGuardEnvironment — RL environment for AI-dubbed video QC.

The agent receives an observation (timing + text context) and returns a QC action.
Ground truth is stored internally and never exposed to the agent.

Episode sampling order:
  1. If real episodes are loaded from data/loader.py, iterate through them.
  2. Falls back to a single hardcoded timing-collision episode if the data
     directory is missing or the loader raises an exception.
"""

import copy
import sys
import os
from rewards.combined import compute_reward

# ── fallback hardcoded episode ─────────────────────────────────────────────────

_FALLBACK_EPISODE = {
    "observation": {
        "episode_id": "ep_fallback_001",
        "segment_id": "seg_0042",
        "difficulty_level": "medium",
        "original": {
            "text": "Welcome back. Let's begin.",
            "start_time": 0.0,
            "end_time": 2.1,
            "duration_seconds": 2.1,
        },
        "dubbed": {
            "text": "वापस स्वागत है। चलिए शुरू करते हैं।",
            "language_code": "hi",
            "locale_code": "hi-IN",
            "estimated_duration_seconds": 5.8,
        },
        "next_segment_start_seconds": 2.3,
        "max_allowed_dubbed_duration_seconds": 2.1,
    },
    "ground_truth": {
        "segment_id": 42,
        "error_type": "timing_collision",
        "severity": "BLOCK",
        "suggested_fix": "Shorten Hindi dub to 'वापस स्वागत। शुरू करें।' (~2.0 s) or extend segment gap.",
        "fix_duration": 2.0,
        "locale_rule": None,
    },
}


def _load_episodes() -> list[dict]:
    try:
        data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from data.loader import load_all
        episodes = load_all(shuffle=True, seed=42)
        if episodes:
            return episodes
    except Exception:
        pass
    return [_FALLBACK_EPISODE]


class DubGuardEnvironment:
    def __init__(self, episodes: list[dict] | None = None):
        """
        Args:
            episodes: Pre-loaded episode list. If None, loaded automatically
                      from data/loader.py (falls back to hardcoded episode).
        """
        self._episodes = episodes if episodes is not None else _load_episodes()
        self._cursor = 0
        self._observation: dict | None = None
        self._ground_truth: dict | None = None

    # ── public API ─────────────────────────────────────────────────────────────

    def reset(self, difficulty: str | None = None) -> dict:
        """
        Advance to the next episode and return the agent-visible observation.

        Args:
            difficulty: "easy", "medium", or "hard". When set, only episodes
                        with a matching difficulty_level are sampled. The pool
                        is reshuffled each time it completes a full cycle so
                        the agent cannot memorise episode order.
        """
        import random as _random
        if difficulty is not None:
            pool_key = f"_pool_{difficulty}"
            if not hasattr(self, pool_key):
                pool = [ep for ep in self._episodes
                        if ep["observation"]["difficulty_level"] == difficulty]
                if not pool:
                    pool = list(self._episodes)
                _random.shuffle(pool)
                setattr(self, pool_key, pool)
                setattr(self, f"_cursor_{difficulty}", 0)

            pool = getattr(self, pool_key)
            cursor = getattr(self, f"_cursor_{difficulty}")
            episode = pool[cursor % len(pool)]
            cursor += 1
            if cursor % len(pool) == 0:
                _random.shuffle(pool)
            setattr(self, f"_cursor_{difficulty}", cursor)
        else:
            episode = self._episodes[self._cursor % len(self._episodes)]
            self._cursor += 1
            if self._cursor % len(self._episodes) == 0:
                import random as _r
                _r.shuffle(self._episodes)

        self._ground_truth = copy.deepcopy(episode["ground_truth"])
        self._observation = copy.deepcopy(episode["observation"])
        return copy.deepcopy(self._observation)

    def step(self, action: dict) -> tuple[dict, bool]:
        """
        Accept the agent's QC action, compute reward, return (reward_dict, done).

        action keys:
          segment_id, error_type, severity (BLOCK|WARN|PASS),
          reason, suggested_fix, estimated_fix_duration
        """
        if self._observation is None:
            raise RuntimeError("Call reset() before step().")
        reward = compute_reward(
            action=action,
            ground_truth=self._ground_truth,
            max_allowed_duration=self._observation["max_allowed_dubbed_duration_seconds"],
            language_code=self._observation["dubbed"]["language_code"],
        )
        return reward, True

    def state(self) -> dict | None:
        """Return the current agent-visible observation (no ground truth)."""
        return copy.deepcopy(self._observation) if self._observation is not None else None

    # ── introspection helpers ──────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._episodes)

    def __repr__(self) -> str:
        return f"DubGuardEnvironment(episodes={len(self._episodes)}, cursor={self._cursor})"
