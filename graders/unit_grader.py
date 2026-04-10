def grade(agent_value: float, correct_value: float) -> float:
    if correct_value == 0:
        return 0.95 if abs(agent_value) < 1e-9 else 0.05
    error = abs(agent_value - correct_value) / abs(correct_value)
    if error < 0.001:   return 0.95   # exact
    if error < 0.01:    return 0.70   # close
    if error < 0.05:    return 0.30   # ballpark
    return 0.05
