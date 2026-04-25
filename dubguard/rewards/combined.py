"""
Combines all five reward components.
Weights: detection=0.40, severity=0.20, correction=0.25, cultural=0.15.
False-positive penalty is additive (not weighted).
"""

from .detection import get_detection_reward
from .severity import get_severity_reward
from .correction import get_correction_reward
from .cultural import get_cultural_reward
from .false_positive import get_false_positive_penalty


def compute_reward(action: dict, ground_truth: dict, max_allowed_duration: float, language_code: str) -> dict:
    detection = get_detection_reward(action, ground_truth)
    severity = get_severity_reward(action, ground_truth)
    correction = get_correction_reward(action, ground_truth, language=language_code)
    cultural = get_cultural_reward(action, ground_truth)
    fp_penalty = get_false_positive_penalty(action, ground_truth)

    combined = (
        (detection * 0.40) +
        (severity * 0.20) +
        (correction * 0.25) +
        (cultural * 0.15) +
        fp_penalty
    )

    return {
        "detection_score": detection,
        "severity_score": severity,
        "correction_score": correction,
        "cultural_score": cultural,
        "false_positive_penalty": fp_penalty,
        "combined_score": combined,
    }
