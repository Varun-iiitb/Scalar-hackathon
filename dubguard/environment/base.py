"""
OpenEnv base classes — Environment and MCPEnvironment.

These define the standard interface that all OpenEnv-compatible environments
must implement. DubGuardEnvironment inherits from Environment.
"""

from abc import ABC, abstractmethod
from typing import Any


class Environment(ABC):
    """
    Base class for all OpenEnv environments.

    Subclasses must implement reset(), step(), and state().
    The environment is episode-based: reset() starts a new episode,
    step() advances it, and state() returns the current observation.
    """

    @abstractmethod
    def reset(self, **kwargs) -> dict:
        """Start a new episode and return the initial observation."""
        ...

    @abstractmethod
    def step(self, action: dict) -> tuple[dict, bool]:
        """
        Apply action, return (reward_dict, done).

        done=True means the episode has ended and reset() must be called
        before the next step().
        """
        ...

    @abstractmethod
    def state(self) -> dict | None:
        """Return the current agent-visible observation, or None before reset()."""
        ...

    # Optional metadata hooks — subclasses may override

    def observation_space(self) -> dict:
        """Describe the observation space (optional, used by OpenEnv registry)."""
        return {}

    def action_space(self) -> dict:
        """Describe the action space (optional, used by OpenEnv registry)."""
        return {}

    def reward_range(self) -> tuple[float, float]:
        """Return (min_reward, max_reward)."""
        return (-float("inf"), float("inf"))


class MCPEnvironment(Environment):
    """
    Extension of Environment for Model Context Protocol (MCP) tool-call style
    interaction. Adds a tools() method that exposes environment actions as
    named tool definitions consumable by an MCP-aware LLM client.
    """

    @abstractmethod
    def tools(self) -> list[dict]:
        """
        Return a list of MCP-style tool definitions, each with:
          { "name": str, "description": str, "parameters": { JSON Schema } }
        """
        ...

    def call_tool(self, tool_name: str, parameters: dict) -> Any:
        """
        Dispatch a named tool call. Default implementation maps the
        "submit_qc_action" tool to step(). Override for custom routing.
        """
        if tool_name == "submit_qc_action":
            return self.step(parameters)
        raise ValueError(f"Unknown tool: {tool_name!r}")
