"""
Quick test for LLM Agent
"""
from llm_agent_1 import LLMAgent

# Initialize agent
print("Initializing LLM agent (this may take a moment on first run)...")
agent = LLMAgent()

# Test goal extraction
print("\n--- Testing Goal Extraction ---")
initial_state = {
    "monkey_location": "posD",
    "banana_location": "posB",
    "box_location": "posA",
    "monkey_on_box": False,
    "has_banana": False
}

goal = agent.extract_goal("How can the monkey get the banana?", initial_state)
print(f"Goal: {goal}")

# Test plan explanation
print("\n--- Testing Plan Explanation ---")
plan = [
    "(move monkey1 posd posc)",
    "(move monkey1 posc posb)",
    "(grab_banana_from_box monkey1 banana1 boxb posb)"
]

explanation = agent.explain_plan(plan, initial_state)
print(f"Explanation:\n{explanation}")