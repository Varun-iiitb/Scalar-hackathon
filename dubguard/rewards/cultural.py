def get_cultural_reward(agent_action, ground_truth):
    """
    Rewards correct identification of locale-specific cultural mismatches.
    Checks error type match and whether the agent cited the correct locale rule.
    Partial credit (0.5) for catching the issue but citing wrong reasoning.
    Returns 1.0 when no cultural error was planted (not applicable).
    """
    gt_error = ground_truth.get("error_type")

    if gt_error != "cultural_mismatch":
        return 1.0

    agent_error = agent_action.get("error_type")
    if agent_error != "cultural_mismatch":
        return 0.0

    gt_rule = ground_truth.get("locale_rule")
    agent_reason = agent_action.get("reason", "")
    agent_rule = agent_action.get("locale_rule", "")

    if gt_rule:
        if gt_rule in agent_reason or gt_rule == agent_rule:
            return 1.0
        return 0.5  # Caught the issue but cited wrong rule

    return 1.0
