"""
IsoSync OpenEnv-compatible environment.

An episode = one full video script (N segments by curriculum level).
The agent translates each segment in sequence. Every translation that runs
long deducts from a shared cumulative budget — making this genuinely sequential.

API:
    env = IsoSyncEnvironment()
    obs = env.reset(level=1)
    obs, reward, done, info = env.step({"translation": "..."})
    state = env.state()

Language is determined by curriculum level (see data_gen.CURRICULUM):
    Level 1 & 2 → Portuguese
    Level 3      → Hindi
"""

import signal
import time

from data_gen import generate_episode, SPEAKING_RATES, CURRICULUM
from rewards import (
    compute_reward, timing_reward, semantic_reward,
    budget_reward, locale_reward, count_syllables,
)

# ── OpenEnv base class ────────────────────────────────────────────────────────
try:
    from openenv import Environment
except ImportError:
    class Environment:
        def reset(self, **kwargs): raise NotImplementedError
        def step(self, action: dict): raise NotImplementedError
        def state(self) -> dict: raise NotImplementedError


class _TimeoutError(Exception):
    pass


def _timeout_handler(signum, frame):
    raise _TimeoutError("step() exceeded 30-second limit")


class IsoSyncEnvironment(Environment):
    """
    RL environment for isochronous video translation under a shared
    cumulative timing budget.
    """

    EARLY_TERMINATION_DEFICIT = 3.0

    def __init__(self):
        self._segments: list[dict] = []
        self._current_idx: int = 0
        self._budget_total: float = 0.0
        self._budget_deficit: float = 0.0
        self._episode_translations: list[dict] = []
        self._current_level: int = 1
        self._language: str = "Portuguese"
        self._episode_reward_sum: float = 0.0
        self._done: bool = False
        self._prev_translation: str | None = None   # only previous, not full history

    # ── OpenEnv API ───────────────────────────────────────────────────────────

    def reset(self, level: int = 1) -> str:
        """
        Start a fresh episode.

        Language is fixed by curriculum level (Portuguese for L1/L2, Hindi for L3).
        Returns the formatted observation string for the first segment.
        """
        self._segments             = generate_episode(level=level)
        self._current_idx          = 0
        self._budget_deficit       = 0.0
        self._episode_translations = []
        self._current_level        = level
        self._language             = CURRICULUM[level]["language"]
        self._episode_reward_sum   = 0.0
        self._done                 = False
        self._prev_translation     = None

        slack = CURRICULUM[level]["duration_slack"]
        self._budget_total = (
            sum(s["original_duration"] for s in self._segments)
            + slack * len(self._segments)
        )

        return self._format_observation()

    def step(self, action: dict) -> tuple[str, float, bool, dict]:
        """
        Accept a translation for the current segment and advance the episode.

        Returns (observation, reward, done, info).
        Observation is an empty string when done=True.
        """
        if self._done:
            raise RuntimeError("Episode is done — call reset() first.")

        translation = action.get("translation", "").strip() or "[empty]"

        # ── 30-second timeout (Unix only; skipped on Windows) ─────────────────
        try:
            signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(30)
        except (AttributeError, OSError):
            pass

        try:
            reward, info = self._score_step(translation)
        except _TimeoutError:
            reward = -1.0
            info = {"error": "step_timeout"}
        finally:
            try:
                signal.alarm(0)
            except (AttributeError, OSError):
                pass

        # ── Record and advance ────────────────────────────────────────────────
        self._episode_translations.append({
            "segment_id":  self._segments[self._current_idx]["segment_id"],
            "original":    self._segments[self._current_idx]["original_text"],
            "translation": translation,
            "reward":      reward,
        })
        self._episode_reward_sum += reward
        self._prev_translation = translation
        self._current_idx += 1

        all_done   = self._current_idx >= len(self._segments)
        over_limit = self._budget_deficit > self.EARLY_TERMINATION_DEFICIT

        if all_done or over_limit:
            self._done = True
            info["done_reason"]    = "completed" if all_done else "early_termination"
            info["total_reward"]   = round(self._episode_reward_sum, 4)
            info["final_deficit"]  = round(self._budget_deficit, 4)
            info["n_segments"]     = len(self._segments)
            info["translations"]   = self._episode_translations
            return "", reward, True, info

        return self._format_observation(), reward, False, info

    def state(self) -> dict:
        seg = self._segments[self._current_idx] if self._current_idx < len(self._segments) else None
        return {
            "current_segment":      seg,
            "budget_deficit":       round(self._budget_deficit, 4),
            "budget_remaining":     round(self._budget_total - self._budget_deficit, 4),
            "segments_completed":   self._current_idx,
            "total_segments":       len(self._segments),
            "episode_translations": self._episode_translations,
            "current_level":        self._current_level,
            "language":             self._language,
            "done":                 self._done,
            "prev_translation":     self._prev_translation,
        }

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _score_step(self, translation: str) -> tuple[float, dict]:
        seg             = self._segments[self._current_idx]
        language        = seg["target_language"]
        target_duration = seg["original_duration"]

        rate      = SPEAKING_RATES.get(language, 5.0)
        syllables = count_syllables(translation, language)
        estimated = syllables / rate
        overrun   = max(0.0, estimated - target_duration)
        self._budget_deficit += overrun

        t = timing_reward(translation, target_duration, language)
        s = semantic_reward(translation, seg["reference_translations"])
        b = budget_reward(self._budget_deficit)
        l = locale_reward(translation, seg["locale"])
        combined = compute_reward(translation, seg, self._budget_deficit)

        info = {
            "timing_reward":      t,
            "semantic_reward":    s,
            "budget_reward":      b,
            "locale_reward":      l,
            "combined_reward":    combined,
            "budget_deficit":     round(self._budget_deficit, 4),
            "estimated_duration": round(estimated, 3),
        }
        return combined, info

    def _format_observation(self) -> str:
        """
        Format the current segment into a prompt string.

        Only the IMMEDIATELY PREVIOUS translation is included as context —
        not the full episode history — to keep prompts within the model's
        context window even at level 3 (10 segments).
        """
        if self._current_idx >= len(self._segments):
            return ""

        seg               = self._segments[self._current_idx]
        lang              = seg["target_language"]
        dur               = seg["original_duration"]
        remaining         = len(self._segments) - self._current_idx
        budget_remaining  = self._budget_total - self._budget_deficit

        if self._budget_deficit < 0.2:
            status = "ON TRACK"
        else:
            status = f"OVER by {self._budget_deficit:.1f}s — COMPRESS your translation"

        prev_ctx = (
            f"\nPrevious translation: {self._prev_translation}"
            if self._prev_translation else ""
        )

        return (
            f"You are a professional video dubbing translator.\n\n"
            f"Translate the following English segment into {lang}.\n\n"
            f"Original: {seg['original_text']}\n"
            f"Time window: {dur} seconds\n"
            f"Max syllables: {seg['max_syllables']}\n"
            f"Budget: {budget_remaining:.1f}s left for {remaining} segments\n"
            f"Status: {status}"
            f"{prev_ctx}\n\n"
            f"Rules:\n"
            f"- Your translation MUST fit within {dur} seconds when spoken\n"
            f"- Preserve the core meaning\n"
            f"- Sound natural in {seg['locale']}\n"
            f"- Do not copy the English text\n\n"
            f"Reply with ONLY the translation. No explanation."
        )


if __name__ == "__main__":
    env = IsoSyncEnvironment()
    obs = env.reset(level=1)
    print("=== First observation (Level 1 — Portuguese) ===")
    print(obs)

    # Simulate 2 steps using reference translations
    ep = generate_episode(level=1, seed=0)
    for i in range(min(3, len(ep))):
        ref = ep[i]["reference_translations"][0]
        obs, reward, done, info = env.step({"translation": ref})
        print(f"\n--- Step {i+1} ---")
        print(f"Translation : {ref}")
        print(f"Reward      : {reward}")
        print(f"Done        : {done}")
        if done:
            print(f"Info        : {info}")
            break
        print(f"Next obs snippet: {obs[:120]}...")
