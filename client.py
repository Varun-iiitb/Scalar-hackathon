"""
IsoSync client — talks to the FastAPI server over HTTP.
Zero imports from server-side modules (environment.py, rewards.py, data_gen.py).

Usage:
    client = IsoSyncClient("https://<your-space>.hf.space")
    obs = client.reset(level=1, language="Hindi")
    result = client.step("उबलते पानी में एक चुटकी नमक डालें।")
    print(result["reward"])
"""

import json
import urllib.request
import urllib.error
from typing import Optional


class IsoSyncClient:
    def __init__(self, base_url: str = "http://localhost:7860"):
        self.base_url = base_url.rstrip("/")

    def health(self) -> dict:
        return self._get("/health")

    def info(self) -> dict:
        return self._get("/info")

    def reset(self, level: int = 1, language: str = "Hindi") -> str:
        """Start a new episode. Returns the first observation string."""
        result = self._post("/reset", {"level": level, "language": language})
        return result["observation"]

    def step(self, translation: str) -> dict:
        """Submit a translation. Returns {"observation", "reward", "done", "info"}."""
        return self._post("/step", {"translation": translation})

    def state(self) -> dict:
        return self._get("/state")["state"]

    def _get(self, path: str) -> dict:
        url = self.base_url + path
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"GET {url} → {e.code}: {e.read().decode()}") from e

    def _post(self, path: str, payload: dict) -> dict:
        url = self.base_url + path
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"}, method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"POST {url} → {e.code}: {e.read().decode()}") from e


if __name__ == "__main__":
    import sys
    url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:7860"
    client = IsoSyncClient(url)

    print("Server:", client.info())
    obs = client.reset(level=1, language="Hindi")
    print("\nObservation:\n", obs)

    result = client.step("उबलते पानी में एक चुटकी नमक डालें।")
    print("\nReward:", result["reward"])
    print("Done:", result["done"])
