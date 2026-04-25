"""
DubGuard client — talks to the FastAPI server over HTTP.

This module has zero imports from the server-side code (environment/, rewards/).
Judges and external users interact with the environment through this interface.

Usage:
    client = DubGuardClient("https://<your-space>.hf.space")
    obs = client.reset()
    reward = client.step({
        "segment_id": obs["segment_id"],
        "error_type": "timing_collision",
        "severity": "BLOCK",
        "reason": "Dub exceeds segment gap",
        "suggested_fix": "Shortened Hindi dub",
        "estimated_fix_duration": 2.0,
    })
    print(reward)
"""

import json
import urllib.request
import urllib.error
from typing import Optional


class DubGuardClient:
    """HTTP client for the DubGuard OpenEnv server."""

    def __init__(self, base_url: str = "http://localhost:7860"):
        self.base_url = base_url.rstrip("/")

    # ── public API mirrors server endpoints ───────────────────────────────────

    def health(self) -> dict:
        return self._get("/health")

    def info(self) -> dict:
        return self._get("/info")

    def reset(self, difficulty: Optional[str] = None) -> dict:
        """Start a new episode. Returns the agent-visible observation."""
        payload = {}
        if difficulty is not None:
            payload["difficulty"] = difficulty
        return self._post("/reset", payload)["observation"]

    def step(self, action: dict) -> dict:
        """
        Submit a QC action. Returns {"reward": {...}, "done": bool}.

        Required action keys:
            segment_id (int), error_type (str), severity (str),
            reason (str), suggested_fix (str), estimated_fix_duration (float)
        """
        return self._post("/step", action)

    def state(self) -> dict:
        """Return the current observation without advancing the episode."""
        return self._get("/state")["observation"]

    # ── internal HTTP helpers ─────────────────────────────────────────────────

    def _get(self, path: str) -> dict:
        url = self.base_url + path
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            body = e.read().decode()
            raise RuntimeError(f"GET {url} → {e.code}: {body}") from e

    def _post(self, path: str, payload: dict) -> dict:
        url = self.base_url + path
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"}, method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            body = e.read().decode()
            raise RuntimeError(f"POST {url} → {e.code}: {body}") from e


# ── quick demo ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:7860"
    client = DubGuardClient(url)

    print("Server info:", json.dumps(client.info(), indent=2))

    obs = client.reset()
    print("\nObservation:")
    print(f"  Episode : {obs['episode_id']}")
    print(f"  Segment : {obs['segment_id']}")
    print(f"  Lang    : {obs['dubbed']['language_code']}")
    print(f"  Original: {obs['original']['text']}")
    print(f"  Dubbed  : {obs['dubbed']['text']}")
    headroom = (obs["max_allowed_dubbed_duration_seconds"]
                - obs["dubbed"]["estimated_duration_seconds"])
    print(f"  Headroom: {headroom:+.2f}s")

    action = {
        "segment_id": int(obs["segment_id"].replace("seg_", "").lstrip("0") or "0"),
        "error_type": "none",
        "severity": "PASS",
        "reason": "Timing is fine.",
        "suggested_fix": "",
        "estimated_fix_duration": 0.0,
    }
    result = client.step(action)
    print("\nReward:", json.dumps(result["reward"], indent=2))
