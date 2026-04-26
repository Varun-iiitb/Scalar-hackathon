"""
IsoSync OpenEnv-compatible environment.

Implements MCPEnvironment (base class from openenv or local fallback).
Exposes three MCP tools: translate_segment, check_budget, get_episode_summary.

New in this version:
  - Budget banking: under-runs save time in a bank that offsets future overruns.
    The agent now has a genuine strategic choice — compress early to build slack.
  - Coherence reward: numbers/entities are tracked across all translations;
    inconsistency is penalised every step, not just at episode end.
  - MCPEnvironment: tools() and call_tool() exposed for MCP-aware clients.

API:
    env = IsoSyncEnvironment()
    obs = env.reset(level=1)
    obs, reward, done, info = env.step({"translation": "..."})
    state = env.state()
    tools = env.tools()
    result = env.call_tool("check_budget", {})
"""

import signal
import time

from data_gen import generate_episode, SPEAKING_RATES, CURRICULUM
from rewards import (
    compute_reward, timing_reward, semantic_reward,
    budget_reward, locale_reward, coherence_reward, count_syllables,
)

# ── OpenEnv base classes ──────────────────────────────────────────────────────
try:
    from openenv import Environment, MCPEnvironment
except ImportError:
    class Environment:
        """Fallback when openenv package is not installed."""
        def reset(self, **kwargs):   raise NotImplementedError
        def step(self, action):      raise NotImplementedError
        def state(self) -> dict:     raise NotImplementedError

    class MCPEnvironment(Environment):
        """
        Extension of Environment that exposes actions as named MCP tools.
        Subclasses must implement tools() and may override call_tool().
        """
        def tools(self) -> list[dict]:
            raise NotImplementedError

        def call_tool(self, tool_name: str, parameters: dict):
            if tool_name == "translate_segment":
                return self.step(parameters)
            raise ValueError(f"Unknown tool: {tool_name!r}")


class _TimeoutError(Exception):
    pass


def _timeout_handler(signum, frame):
    raise _TimeoutError("step() exceeded 30-second limit")


# ── Environment ───────────────────────────────────────────────────────────────

class IsoSyncEnvironment(MCPEnvironment):
    """
    RL environment for isochronous video translation under a shared
    cumulative timing budget with banking.

    Budget banking mechanic
    -----------------------
    When the agent's translation is SHORTER than the target duration,
    the saved time goes into a bank. When a future translation OVERRUNS,
    the bank is drawn down first before incrementing the deficit.

    This creates genuine strategic depth: compress early segments to build
    a buffer for semantically dense ones later in the episode.

      bank    += max(0, target − estimated)      # save fast translations
      bank    -= min(bank, overrun)              # draw down before penalising
      deficit += max(0, overrun − bank_drawn)    # only real overruns count
    """

    EARLY_TERMINATION_DEFICIT = 3.0

    def __init__(self):
        self._segments:             list[dict] = []
        self._current_idx:          int   = 0
        self._budget_total:         float = 0.0
        self._budget_deficit:       float = 0.0
        self._budget_bank:          float = 0.0   # ← NEW: saved time
        self._episode_translations: list[dict] = []
        self._current_level:        int   = 1
        self._language:             str   = "Portuguese"
        self._episode_reward_sum:   float = 0.0
        self._done:                 bool  = False
        self._prev_translation:     str | None = None

    # ── OpenEnv API ───────────────────────────────────────────────────────────

    def reset(self, level: int = 1) -> str:
        """
        Start a fresh episode. Language is fixed by curriculum level.
        Returns the formatted observation string for the first segment.
        """
        self._segments             = generate_episode(level=level)
        self._current_idx          = 0
        self._budget_deficit       = 0.0
        self._budget_bank          = 0.0
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
        observation is empty string when done=True.
        """
        if self._done:
            raise RuntimeError("Episode is done — call reset() first.")

        translation = action.get("translation", "").strip() or "[empty]"

        try:
            signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(30)
        except (AttributeError, OSError):
            pass

        try:
            reward, info = self._score_step(translation)
        except _TimeoutError:
            reward = -1.0
            info   = {"error": "step_timeout"}
        finally:
            try:
                signal.alarm(0)
            except (AttributeError, OSError):
                pass

        self._episode_translations.append({
            "segment_id":  self._segments[self._current_idx]["segment_id"],
            "original":    self._segments[self._current_idx]["original_text"],
            "translation": translation,
            "reward":      reward,
        })
        self._episode_reward_sum += reward
        self._prev_translation    = translation
        self._current_idx        += 1

        all_done   = self._current_idx >= len(self._segments)
        over_limit = self._budget_deficit > self.EARLY_TERMINATION_DEFICIT

        if all_done or over_limit:
            self._done = True
            info["done_reason"]   = "completed" if all_done else "early_termination"
            info["total_reward"]  = round(self._episode_reward_sum, 4)
            info["final_deficit"] = round(self._budget_deficit, 4)
            info["budget_bank"]   = round(self._budget_bank, 4)
            info["n_segments"]    = len(self._segments)
            info["translations"]  = self._episode_translations
            return "", reward, True, info

        return self._format_observation(), reward, False, info

    def state(self) -> dict:
        seg = self._segments[self._current_idx] if self._current_idx < len(self._segments) else None
        return {
            "current_segment":      seg,
            "budget_deficit":       round(self._budget_deficit, 4),
            "budget_bank":          round(self._budget_bank, 4),
            "budget_remaining":     round(self._budget_total - self._budget_deficit, 4),
            "segments_completed":   self._current_idx,
            "total_segments":       len(self._segments),
            "episode_translations": self._episode_translations,
            "current_level":        self._current_level,
            "language":             self._language,
            "done":                 self._done,
            "prev_translation":     self._prev_translation,
        }

    # ── MCP tools ─────────────────────────────────────────────────────────────

    def tools(self) -> list[dict]:
        """
        Expose three MCP-style tool definitions for MCP-aware clients.

        Note: reserved names (reset, step, state, close) are intentionally
        avoided per the OpenEnv specification.
        """
        return [
            {
                "name": "translate_segment",
                "description": (
                    "Submit a translation for the current segment. "
                    "Returns (observation, reward, done, info)."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "translation": {
                            "type": "string",
                            "description": "The translated text in the target language.",
                        }
                    },
                    "required": ["translation"],
                },
            },
            {
                "name": "check_budget",
                "description": (
                    "Get the current timing budget status: deficit, bank balance, "
                    "and remaining time for the episode."
                ),
                "parameters": {"type": "object", "properties": {}},
            },
            {
                "name": "get_episode_summary",
                "description": (
                    "Get a full summary of the current episode: all translations "
                    "so far, segment progress, and reward history."
                ),
                "parameters": {"type": "object", "properties": {}},
            },
        ]

    def call_tool(self, tool_name: str, parameters: dict):
        """Route named MCP tool calls to the correct environment method."""
        if tool_name == "translate_segment":
            return self.step(parameters)

        if tool_name == "check_budget":
            s = self.state()
            deficit = s["budget_deficit"]
            bank    = s["budget_bank"]
            if deficit < 0.2:
                status = "ON TRACK"
            else:
                status = f"OVER by {deficit:.1f}s (bank: {bank:.1f}s saved)"
            return {
                "budget_remaining": s["budget_remaining"],
                "budget_bank":      bank,
                "budget_deficit":   deficit,
                "segments_left":    s["total_segments"] - s["segments_completed"],
                "status":           status,
            }

        if tool_name == "get_episode_summary":
            return self.state()

        raise ValueError(f"Unknown tool: {tool_name!r}")

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _score_step(self, translation: str) -> tuple[float, dict]:
        """
        Score one translation step.

        Budget banking logic:
          - Under-run  → saved time goes into self._budget_bank
          - Over-run   → draw from bank first, then increment deficit
        """
        seg             = self._segments[self._current_idx]
        language        = seg["target_language"]
        target_duration = seg["original_duration"]

        rate      = SPEAKING_RATES.get(language, 5.0)
        syllables = count_syllables(translation, language)
        estimated = syllables / rate
        delta     = estimated - target_duration   # positive = over, negative = under

        # ── Budget banking ────────────────────────────────────────────────────
        if delta <= 0:
            # Fast translation — bank the saved time
            self._budget_bank += abs(delta)
        else:
            # Slow translation — draw from bank first
            drawn              = min(self._budget_bank, delta)
            self._budget_bank -= drawn
            effective_overrun  = delta - drawn
            self._budget_deficit += effective_overrun

        # ── Coherence: compute from ALL translations so far (including this one)
        # We append a preview entry so coherence reflects the current translation
        history_preview = self._episode_translations + [{
            "original":    seg["original_text"],
            "translation": translation,
        }]
        c_score = coherence_reward(history_preview)

        # ── Per-component scores ──────────────────────────────────────────────
        t = timing_reward(translation, target_duration, language)
        s = semantic_reward(translation, seg["reference_translations"])
        b = budget_reward(self._budget_deficit)
        l = locale_reward(translation, seg["locale"])

        combined = compute_reward(
            translation, seg, self._budget_deficit, coherence_score=c_score
        )

        info = {
            "timing_reward":      t,
            "semantic_reward":    s,
            "budget_reward":      b,
            "locale_reward":      l,
            "coherence_reward":   c_score,
            "combined_reward":    combined,
            "budget_deficit":     round(self._budget_deficit, 4),
            "budget_bank":        round(self._budget_bank, 4),
            "estimated_duration": round(estimated, 3),
        }
        return combined, info

    def _format_observation(self) -> str:
        """
        Format the current segment into a prompt string.

        Shows budget bank balance so the agent can learn to use it
        strategically. Only the previous translation is included as context —
        not the full history — to stay within the model's context window.
        """
        if self._current_idx >= len(self._segments):
            return ""

        seg      = self._segments[self._current_idx]
        lang     = seg["target_language"]
        dur      = seg["original_duration"]
        max_syl  = seg["max_syllables"]
        # Approximate word budget — Portuguese averages ~2 syllables per word
        max_words = max(2, int(max_syl / 2))
        remaining = len(self._segments) - self._current_idx
        budget_remaining = self._budget_total - self._budget_deficit

        if self._budget_deficit < 0.2:
            if self._budget_bank > 0.2:
                status = f"ON TRACK (bank: +{self._budget_bank:.1f}s saved)"
            else:
                status = "ON TRACK"
        else:
            status = (
                f"OVER by {self._budget_deficit:.1f}s "
                f"(bank: {self._budget_bank:.1f}s) — COMPRESS HARD"
            )

        prev_ctx = (
            f"\nPrevious translation: {self._prev_translation}"
            if self._prev_translation else ""
        )

        return (
            f"DUBBING TRANSLATION — STRICT LENGTH LIMIT\n\n"
            f"*** HARD LIMIT: {max_syl} syllables MAX (about {max_words} words) ***\n"
            f"*** Anything longer breaks the dub — count syllables before writing ***\n\n"
            f"English: \"{seg['original_text']}\"\n"
            f"Translate to {lang} ({seg['locale']}) in <= {max_syl} syllables.\n\n"
            f"Time window: {dur}s   |   Budget: {budget_remaining:.1f}s left for {remaining} segments\n"
            f"Status: {status}"
            f"{prev_ctx}\n\n"
            f"Rules:\n"
            f"1. STAY UNDER {max_syl} SYLLABLES — be concise, drop filler words\n"
            f"2. Keep all numbers exact (e.g. \"thirty\" -> \"30\" or \"trinta\")\n"
            f"3. Sound natural to a {seg['locale']} speaker\n"
            f"4. Do NOT copy the English\n\n"
            f"Translation (<= {max_syl} syllables, no quotes, no explanation):"
        )


if __name__ == "__main__":
    env = IsoSyncEnvironment()

    print("=== MCP Tools ===")
    for tool in env.tools():
        print(f"  {tool['name']}: {tool['description'][:60]}...")

    print("\n=== Level 1 Episode (Portuguese) ===")
    obs = env.reset(level=1)
    print(obs)

    ep = generate_episode(level=1, seed=0)
    for i in range(min(3, len(ep))):
        ref = ep[i]["reference_translations"][0]
        obs, reward, done, info = env.step({"translation": ref})
        print(f"\n--- Step {i+1} ---")
        print(f"Translation : {ref}")
        print(f"Reward      : {reward}  (bank={info['budget_bank']:.2f}s  deficit={info['budget_deficit']:.2f}s)")
        print(f"Coherence   : {info['coherence_reward']}")
        if done:
            print(f"Episode done: {info['done_reason']}")
            break

    print("\n=== check_budget tool ===")
    print(env.call_tool("check_budget", {}))
