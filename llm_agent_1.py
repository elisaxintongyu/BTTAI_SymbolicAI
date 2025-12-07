"""
LLM Agent Module - GPT4All Version

Handles goal extraction and plan explanation using local GPT4All models.
"""
import json
from typing import Dict, List, Any
from gpt4all import GPT4All


class LLMAgent:
    """LLM agent using GPT4All for natural language processing."""
    
    def __init__(self, model: str = "mistral-7b-instruct-v0.1.Q4_0.gguf"):
        print(f"Attempting to load model: {model}")
        try:
            self.model = GPT4All(model)
            print(f"Successfully loaded model: {model}")
        except Exception as e:
            print(f"Failed to load model '{model}': {e}")
            print("Trying fallback model: orca-mini-3b.gguf2.q4_0.gguf")
            try:
                self.model = GPT4All("orca-mini-3b.gguf2.q4_0.gguf")
                print("Successfully loaded fallback model")
            except Exception as e2:
                print(f"✗ Failed to load fallback model: {e2}")
                raise RuntimeError(
                    f"Could not load any GPT4All model.\n"
                    f"Primary model error: {e}\n"
                    f"Fallback model error: {e2}\n"
                    f"Please install gpt4all: pip install gpt4all"
                ) from e2
    
    def extract_goal(self, question: str, initial_state: Dict) -> Dict[str, Any]:
        """Convert natural language question to goal dictionary."""
        # Include banana_on_box for better context (planner uses it)
        banana_on_box = initial_state.get('banana_on_box', False)
        
        prompt = f"""Extract the goal from this question about the monkey-banana problem.

Current state:
- Monkey location: {initial_state.get('monkey_location', 'Unknown')}
- Banana location: {initial_state.get('banana_location', 'Unknown')}
- Box location: {initial_state.get('box_location', 'Unknown')}
- Banana on box: {banana_on_box}
- Monkey on box: {initial_state.get('monkey_on_box', False)}
- Has banana: {initial_state.get('has_banana', False)}

Question: {question}

Respond with ONLY a JSON object. Examples:
- "get the banana" -> {{"has_banana": true}}
- "move monkey to A" -> {{"monkey_location": "A"}}
- "move box to banana" ->{{"box_at_banana": true}}

Goal JSON:"""
        
        try:
            with self.model.chat_session():
                response = self.model.generate(prompt, max_tokens=200, temp=0.1)
            
            goal_text = self._extract_json(response)
            return json.loads(goal_text)
        except Exception:
            return self._fallback_goal(question)
    
    def explain_plan(self, plan: List[str], initial_state: Dict) -> str:
        """Convert symbolic plan to plain English explanation."""
        if not plan:
            return "No plan is needed. The goal is already satisfied."
        
        monkey_loc = initial_state.get('monkey_location', 'Unknown')
        banana_loc = initial_state.get('banana_location', 'Unknown')
        banana_on_box = initial_state.get('banana_on_box', False)
        plan_str = "\n".join([f"{i+1}. {action}" for i, action in enumerate(plan)])
        
        prompt = f"""Explain this plan for the monkey to get the banana.

Initial state:
- Monkey at: {monkey_loc} (on ground)
- Banana at: {banana_loc} {'(on box)' if banana_on_box else '(on floor)'}

Plan:
{plan_str}

Convert each action to simple English. Format as a numbered list.
Explanation:"""
        
        try:
            with self.model.chat_session():
                response = self.model.generate(prompt, max_tokens=200, temp=0.7)
            
            explanation = response.strip()
            return explanation if len(explanation) > 20 else self._fallback_explanation(plan)
        except Exception:
            return self._fallback_explanation(plan)
    
    def _extract_json(self, text: str) -> str:
        """Extract JSON object from text response."""
        text = text.strip()
        
        # Remove markdown blocks
        if "```" in text:
            parts = text.split("```")
            for part in parts:
                part = part.strip().lstrip("json")
                if part.startswith("{") or part.startswith("["):
                    text = part
                    break
        
        # Find first JSON object
        start = text.find("{")
        if start == -1:
            start = text.find("[")
        if start == -1:
            return '{"has_banana": true}'
        
        # Extract matching brackets
        bracket_count = 0
        for i in range(start, len(text)):
            if text[i] in "{[":
                bracket_count += 1
            elif text[i] in "}]":
                bracket_count -= 1
                if bracket_count == 0:
                    return text[start:i+1]
        
        return '{"has_banana": true}'
    
    def _fallback_goal(self, question: str) -> Dict[str, Any]:
        """Simple pattern matching fallback."""
        q = question.lower()
        if "banana" in q and ("get" in q or "grab" in q):
            return {"has_banana": True}
        elif "move" in q and "box" in q:
            return {"box_at_banana": True}
        return {"has_banana": True}
    
    def _fallback_explanation(self, plan: List[str]) -> str:
        """Simple rule-based explanation fallback."""
        explanations = []
        for i, action in enumerate(plan, 1):
            action = str(action).strip().strip("()")
            parts = action.split()
            
            if not parts:
                explanations.append(f"{i}. {action}")
                continue
            
            action_name = parts[0].lower()
            
            if action_name == "move" and len(parts) >= 4:
                explanations.append(f"{i}. The monkey moves from {parts[2]} to {parts[3]}")
            elif action_name == "push_box" or action_name == "push-box":
                if len(parts) >= 5:
                    explanations.append(f"{i}. The monkey pushes box {parts[2]} from {parts[3]} to {parts[4]}")
                else:
                    explanations.append(f"{i}. The monkey pushes the box")
            elif "climb_on" in action_name or "climb-on" in action_name:
                explanations.append(f"{i}. The monkey climbs onto the box")
            elif "climb_off" in action_name or "climb-off" in action_name:
                explanations.append(f"{i}. The monkey climbs down from the box")
            elif "grab_banana_from_box" in action_name or ("grab" in action_name and "box" in action):
                explanations.append(f"{i}. The monkey grabs the banana from the box")
            elif "grab_banana_from_ground" in action_name or ("grab" in action_name and "ground" in action.lower()):
                explanations.append(f"{i}. The monkey grabs the banana from the ground")
            elif "grab" in action_name:
                explanations.append(f"{i}. The monkey grabs the banana")
            else:
                explanations.append(f"{i}. {action}")
        
        return "\n".join(explanations)