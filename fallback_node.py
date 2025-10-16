def fallback_node(state):
    print(f"[FallbackNode] Low confidence ({state.get('confidence',0)*100:.1f}%). Please clarify: Was this a positive or negative review?")
    user_input = input("User: ").strip().lower()
    state["final_label"] = 1 if "positive" in user_input else 0
    return state
