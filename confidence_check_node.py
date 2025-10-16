def confidence_check_node(state, threshold=0.7):
    state["fallback"] = state.get("confidence", 0) < threshold
    return state
