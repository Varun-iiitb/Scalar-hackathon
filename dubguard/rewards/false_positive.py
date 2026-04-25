def get_false_positive_penalty(agent_action, ground_truth):
    """
    Returns -0.3 when ground truth has no error but agent flagged one anyway.
    Returns 0.0 otherwise.
    """
    gt_error = ground_truth.get("error_type")
    agent_error = agent_action.get("error_type")

    is_gt_clean = gt_error is None or str(gt_error).lower() in ["", "none", "nan", "null"]
    is_agent_error = agent_error is not None and str(agent_error).lower() not in ["", "none", "nan", "null"]

    if is_gt_clean and is_agent_error:
        return -0.3

    return 0.0
