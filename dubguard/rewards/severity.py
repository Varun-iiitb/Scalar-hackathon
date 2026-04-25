SEVERITY_MAP = {
    "PASS": 0,
    "WARN": 1,
    "BLOCK": 2
}

def get_severity_reward(agent_action, ground_truth):
    """
    Rewards correct BLOCK/WARN/PASS classification.
    Exact match: 1.0
    Partial (one level off): 0.5
    PASS on a BLOCK-level error: -1.0
    Else: 0.0
    """
    gt_sev_str = ground_truth.get("severity", "PASS")
    agent_sev_str = agent_action.get("severity", "PASS")

    gt_val = SEVERITY_MAP.get(str(gt_sev_str).upper(), 0)
    agent_val = SEVERITY_MAP.get(str(agent_sev_str).upper(), 0)

    diff = gt_val - agent_val

    if diff == 0:
        return 1.0
    elif abs(diff) == 1:
        return 0.5
    elif diff == 2:
        return -1.0  # GT is BLOCK, Agent is PASS
    elif diff == -2:
        return 0.0   # GT is PASS, Agent is BLOCK — handled by false_positive penalty

    return 0.0
