"""
End-to-end wiring test for DubGuardEnvironment.
Run from the dubguard/ directory:  python -X utf8 test_env.py
"""

import sys
import os
import json
from collections import Counter

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(__file__))

from environment.env import DubGuardEnvironment
from data.loader import load_all


def main():
    # ── 1. Dataset stats ─────────────────────────────────────────────────────
    all_episodes = load_all(shuffle=False)
    error_counts = Counter(ep["ground_truth"]["error_type"] for ep in all_episodes)
    lang_counts = Counter(ep["observation"]["dubbed"]["language_code"] for ep in all_episodes)

    print("=" * 60)
    print(f"DATASET LOADED: {len(all_episodes)} unique episodes")
    print("By error type:", dict(error_counts))
    print("By language: ", dict(lang_counts))
    print()

    # ── 2. Create env and reset ───────────────────────────────────────────────
    env = DubGuardEnvironment()
    print(repr(env))
    print()

    observation = env.reset()
    print("OBSERVATION (agent-visible, episode 1):")
    print(json.dumps(observation, ensure_ascii=False, indent=2))
    print()

    assert "_ground_truth" not in observation, "FAIL: ground truth leaked into observation!"

    # ── 3. Fake agent action ──────────────────────────────────────────────────
    gt_error = env._ground_truth["error_type"]   # peek for test purposes only
    gt_severity = env._ground_truth["severity"]

    fake_action = {
        "segment_id": env._ground_truth["segment_id"],
        "error_type": gt_error,
        "severity": gt_severity,
        "reason": f"Detected {gt_error} in dubbed segment. Dubbed audio overruns allowed window.",
        "suggested_fix": env._ground_truth["suggested_fix"] or "No fix needed.",
        "estimated_fix_duration": env._ground_truth["fix_duration"],
    }

    print("AGENT ACTION (perfect match for scoring test):")
    print(json.dumps(fake_action, ensure_ascii=False, indent=2))
    print()

    reward, done = env.step(fake_action)
    print("REWARD:")
    print(json.dumps(reward, indent=2))
    print()

    # ── 4. Verify shape and done flag ─────────────────────────────────────────
    expected_keys = {
        "detection_score", "severity_score", "correction_score",
        "cultural_score", "false_positive_penalty", "combined_score",
    }
    assert expected_keys == set(reward.keys()), f"FAIL: unexpected reward keys: {set(reward.keys())}"
    assert done is True, "FAIL: done should be True after single-segment episode."

    print(f"done = {done}  ✓")
    print()

    # ── 5. Cycle through 5 episodes to show variety ───────────────────────────
    print("EPISODE SAMPLE (5 consecutive resets):")
    for n in range(5):
        obs = env.reset()
        gt = env._ground_truth
        print(
            f"  [{n+1}] lang={obs['dubbed']['language_code']}"
            f"  error={gt['error_type']}"
            f"  severity={gt['severity']}"
            f"  difficulty={obs['difficulty_level']}"
        )

    print()
    print("=" * 60)
    print("All checks passed — environment and dataset wired correctly.")


if __name__ == "__main__":
    main()
