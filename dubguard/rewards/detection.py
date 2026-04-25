def get_detection_reward(agent_action, ground_truth):
    """
    Rewards correct identification of error type and segment ID.
    Exact match: 1.0
    Correct type, off-by-one segment: 0.5
    Wrong type or correct type but off-by > 1 segment: 0.0
    Handles the clean (no-error) case separately.
    """
    gt_error = ground_truth.get("error_type")
    agent_error = agent_action.get("error_type")

    is_gt_clean = not gt_error or str(gt_error).lower() in ["none", "null", "nan"]
    is_agent_clean = not agent_error or str(agent_error).lower() in ["none", "null", "nan"]

    if is_gt_clean:
        if is_agent_clean:
            return 1.0
        return 0.0  # Agent flagged error on clean segment; penalty in false_positive.py

    if gt_error != agent_error:
        return 0.0

    gt_seg = ground_truth.get("segment_id")
    agent_seg = agent_action.get("segment_id")

    if gt_seg is not None and agent_seg is not None:
        diff = abs(gt_seg - agent_seg)
        if diff == 0:
            return 1.0
        elif diff == 1:
            return 0.5

    return 0.0
