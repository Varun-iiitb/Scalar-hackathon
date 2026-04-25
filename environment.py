"""
IsoSync OpenEnv-compatible environment.

An episode = one full video script (N segments depending on curriculum level).
The agent translates each segment in sequence. Every translation affects the
cumulative timing budget available for remaining segments — making this a
genuinely sequential RL problem.

API:
    env = IsoSyncEnvironment()
    obs = env.reset(level=1, language="Hindi")
    obs, reward, done, info = env.step({"translation": "..."})
    current = env.state()
"""

import signal
import time

from data_gen import generate_episode, SPEAKING_RATES, CURRICULUM
from rewards import compute_reward, timing_reward, semantic_reward, budget_reward, locale_reward, count_syllables

# ── OpenEnv base class ────────────────────────────────────────────────────────
try:
    from openenv import Environment
except ImportError:
    class Environment:
        """Fallback base class when openenv package is not installed."""
        def reset(self, **kwargs):
            raise NotImplementedError
        def step(self, action: dict):
            raise NotImplementedError
        def state(self) -> dict:
            raise NotImplementedError


# ── Timeout helper ────────────────────────────────────────────────────────────

class _TimeoutError(Exception):
    pass


def _timeout_handler(signum, frame):
    raise _TimeoutError("step() timed out after 30 seconds")


# ── Environment ───────────────────────────────────────────────────────────────

class IsoSyncEnvironment(Environment):
    """
    RL environment for isochronous video translation under a shared
    cumulative timing budget.

    The agent must translate each segment such that the dubbed audio
    fits within the segment's time window. Every second over budget
    is deducted from the shared pool — future segments get harder.
    """

    EARLY_TERMINATION_DEFICIT = 3.0  # seconds — episode ends if overrun exceeds this

    def __init__(self):
        self._segments: list[dict] = []
        self._current_idx: int = 0
        self._budget_total: float = 0.0
        self._budget_deficit: float = 0.0
        self._episode_translations: list[dict] = []
        self._current_level: int = 1
        self._language: str = "Hindi"
        self._episode_reward_sum: float = 0.0
        self._done: bool = False

    # ── OpenEnv API ───────────────────────────────────────────────────────────

    def reset(self, level: int = 1, language: str = "Hindi") -> str:
        """
        Start a fresh episode.

        Args:
            level:    Curriculum level 1/2/3.
            language: "Hindi" or "Portuguese".

        Returns:
            Formatted observation string for the first segment.
        """
        self._segments             = generate_episode(level=level, language=language)
        self._current_idx          = 0
        self._budget_deficit       = 0.0
        self._episode_translations = []
        self._current_level        = level
        self._language             = language
        self._episode_reward_sum   = 0.0
        self._done                 = False

        # Total budget = sum of segment durations + slack for this level
        slack          = CURRICULUM[level]["duration_slack"]
        self._budget_total = (
            sum(s["original_duration"] for s in self._segments)
            + slack * len(self._segments)
        )

        return self._format_observation()

    def step(self, action: dict) -> tuple[str, float, bool, dict]:
        """
        Accept a translation for the current segment and advance the episode.

        Args:
            action: {"translation": str}

        Returns:
            (observation, reward, done, info)
            - observation: next segment prompt string (empty string if done)
            - reward:      float in [-1.0, 1.0]
            - done:        True when all segments are done or budget blown
            - info:        dict with per-component scores and episode summary
        """
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        # ── 30-second step timeout ────────────────────────────────────────────
        start = time.time()
        translation = action.get("translation", "").strip()
        if not translation:
            translation = "[empty]"

        try:
            signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(30)
        except (AttributeError, OSError):
            pass  # Windows doesn't support SIGALRM; skip timeout there

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

        # ── Advance episode ───────────────────────────────────────────────────
        self._episode_translations.append({
            "segment_id":  self._segments[self._current_idx]["segment_id"],
            "original":    self._segments[self._current_idx]["original_text"],
            "translation": translation,
            "reward":      reward,
        })
        self._episode_reward_sum += reward
        self._current_idx += 1

        # ── Termination check ─────────────────────────────────────────────────
        all_done = self._current_idx >= len(self._segments)
        over_budget = self._budget_deficit > self.EARLY_TERMINATION_DEFICIT

        if all_done or over_budget:
            self._done = True
            info["done_reason"]     = "completed" if all_done else "early_termination"
            info["total_reward"]    = round(self._episode_reward_sum, 4)
            info["budget_deficit"]  = round(self._budget_deficit, 4)
            info["n_segments"]      = len(self._segments)
            info["translations"]    = self._episode_translations
            return "", reward, True, info

        return self._format_observation(), reward, False, info

    def state(self) -> dict:
        """Return the full current environment state (for debugging/logging)."""
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
        }

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _score_step(self, translation: str) -> tuple[float, dict]:
        """Compute reward and update budget deficit for the current segment."""
        seg = self._segments[self._current_idx]
        language = seg["target_language"]
        target_duration = seg["original_duration"]

        # Update deficit: how far over (or under) this translation runs
        rate       = SPEAKING_RATES.get(language, 5.0)
        syllables  = count_syllables(translation, language)
        estimated  = syllables / rate
        overrun    = max(0.0, estimated - target_duration)
        self._budget_deficit += overrun

        # Per-component scores for logging
        t = timing_reward(translation, target_duration, language)
        s = semantic_reward(translation, seg["reference_translation"])
        b = budget_reward(self._budget_deficit)
        l = locale_reward(translation, seg["locale"])

        combined = compute_reward(translation, seg, self._budget_deficit)

        info = {
            "timing_reward":   t,
            "semantic_reward": s,
            "budget_reward":   b,
            "locale_reward":   l,
            "combined_reward": combined,
            "budget_deficit":  round(self._budget_deficit, 4),
            "estimated_duration": round(estimated, 3),
        }

        return combined, info

    def _format_observation(self) -> str:
        """Format the current segment into a natural-language prompt string."""
        if self._current_idx >= len(self._segments):
            return ""

        seg  = self._segments[self._current_idx]
        lang = seg["target_language"]
        dur  = seg["original_duration"]
        remaining_segments = len(self._segments) - self._current_idx
        budget_remaining   = self._budget_total - self._budget_deficit

        if self._budget_deficit < 0.2:
            budget_status = "ON TRACK"
        else:
            budget_status = f"OVER BUDGET by {self._budget_deficit:.1f}s"

        return (
            f"You are a professional video dubbing translator.\n\n"
            f"Translate the following English segment into {lang}.\n\n"
            f"Original: {seg['original_text']}\n"
            f"Time window: {dur} seconds\n"
            f"Max syllables: {seg['max_syllables']}\n"
            f"Budget remaining: {budget_remaining:.1f}s for {remaining_segments} segments\n"
            f"Budget status: {budget_status}\n"
            f"Locale: {seg['locale']}\n\n"
            f"Rules:\n"
            f"- Your translation MUST fit within {dur} seconds when spoken\n"
            f"- Preserve the core meaning\n"
            f"- Sound natural in {seg['locale']}\n"
            f"- Do not copy the English text\n\n"
            f"Respond with ONLY the translation. No explanation."
        )


if __name__ == "__main__":
    env = IsoSyncEnvironment()
    obs = env.reset(level=1, language="Hindi")
    print("=== First observation ===")
    print(obs)

    # Simulate 2 steps with reference translations
    from data_gen import generate_episode
    ep = generate_episode(level=1, language="Hindi", seed=0)

    for i in range(min(2, len(ep))):
        ref = ep[i]["reference_translation"]
        obs, reward, done, info = env.step({"translation": ref})
        print(f"\n--- Step {i+1} ---")
        print(f"Translation : {ref}")
        print(f"Reward      : {reward}")
        print(f"Done        : {done}")
        print(f"Info        : {info}")
        if done:
            break
