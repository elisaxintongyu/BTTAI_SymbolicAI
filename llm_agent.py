"""
LLM Agent Module for Neural-Symbolic LLM Agent

Handles integration with OpenAI API for:
1. Converting natural language questions to logical goals
2. Explaining symbolic plans in plain English
"""

import os
import logging
from typing import Dict, List, Optional
from openai import OpenAI
from dotenv import load_dotenv  # type: ignore

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMAgent:
    """
    LLM agent for natural language processing and plan explanation.
    """
    
    def __init__(self, model: str = "gpt-4"):
        """
        Initialize the LLM agent.
        
        Args:
            model: OpenAI model to use (default: "gpt-4")
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables. "
                           "Please set it in your .env file.")
        
        self.client = OpenAI(api_key=api_key)
        self.model = model
        logger.info(f"LLM agent initialized with model: {model}")
    
    def extract_goal(self, question: str, initial_state: Dict) -> Dict[str, any]:
        """
        Convert a natural language question into a logical goal.
        
        Args:
            question: Natural language question (e.g., "How can the monkey get the banana?")
            initial_state: Current world state for context
            
        Returns:
            Dictionary representing the goal state
        """
        logger.info(f"Extracting goal from question: {question}")
        
        # Build prompt for goal extraction
        prompt = f"""You are a goal extraction system for a planning problem. 
Given a natural language question about the "Monkeys and Bananas" problem, 
extract the logical goal that should be achieved.

Current world state:
- Monkey location: {initial_state.get('monkey_location', 'Unknown')}
- Banana location: {initial_state.get('banana_location', 'Unknown')}
- Box location: {initial_state.get('box_location', 'Unknown')}
- Monkey on box: {initial_state.get('monkey_on_box', False)}
- Has banana: {initial_state.get('has_banana', False)}

Question: {question}

Based on the question, determine what the goal should be. Common goals include:
- has_banana: True (the monkey should get the banana)
- monkey_location: <location> (the monkey should move to a location)
- box_location: <location> (the box should be moved to a location)

Respond ONLY with a JSON object representing the goal state. 
For example, if the goal is to get the banana, respond with:
{{"has_banana": true}}

If the goal is to move the monkey somewhere, respond with:
{{"monkey_location": "LocationX_Y"}}

Be concise and only include the predicates that need to be true in the goal state.
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a goal extraction system. Return only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=200
            )
            
            goal_text = response.choices[0].message.content.strip()
            logger.info(f"LLM response for goal: {goal_text}")
            
            # Parse JSON response
            import json
            # Remove markdown code blocks if present
            if goal_text.startswith("```"):
                goal_text = goal_text.split("```")[1]
                if goal_text.startswith("json"):
                    goal_text = goal_text[4:]
                goal_text = goal_text.strip()
            
            goal = json.loads(goal_text)
            logger.info(f"Extracted goal: {goal}")
            return goal
            
        except Exception as e:
            logger.error(f"Error extracting goal: {e}")
            # Default goal: get the banana
            logger.warning("Using default goal: has_banana = True")
            return {"has_banana": True}
    
    def explain_plan(self, plan: List[str], initial_state: Dict) -> str:
        """
        Convert a symbolic plan into a step-by-step explanation in plain English.
        
        Args:
            plan: List of action names (e.g., ["Move(monkey, Location0, Location1)", ...])
            initial_state: Initial world state for context
            
        Returns:
            Plain English explanation of the plan
        """
        logger.info(f"Explaining plan with {len(plan)} actions")
        
        if not plan:
            return "No plan is needed. The goal is already satisfied."
        
        # Build prompt for plan explanation
        # Extract key information for better explanation
        monkey_loc = initial_state.get('monkey_location', 'Unknown')
        box_loc = initial_state.get('box_location', 'Unknown')
        banana_loc = initial_state.get('banana_location', 'Unknown')
        monkey_on_box = initial_state.get('monkey_on_box', False)
        box_at_banana = initial_state.get('box_at_banana', False)
        banana_on_box = initial_state.get('banana_on_box', False)
        banana_on_floor = not banana_on_box  # Banana is on floor if NOT on box
        
        prompt = f"""You are explaining a plan for the "Monkeys and Bananas" problem.

CRITICAL INITIAL STATE FACTS:
- Monkey is at: {monkey_loc} (on the GROUND/FLOOR)
- Box is at: {box_loc} (on the GROUND/FLOOR)
- Banana is at: {banana_loc}
- Monkey is on the box: {monkey_on_box} (this is FALSE - monkey is on the GROUND)
- Banana is on the FLOOR: {banana_on_floor} (this is {'TRUE' if banana_on_floor else 'FALSE'} - banana is {'on the FLOOR' if banana_on_floor else 'on top of the box'})

Plan (sequence of actions):
{chr(10).join([f"{i+1}. {action}" for i, action in enumerate(plan)])}

CRITICAL: The banana is {'ON THE FLOOR' if banana_on_floor else 'ON TOP OF THE BOX'}.
{'This means the monkey does NOT need to climb any box. The monkey simply needs to WALK to the banana location and GRAB it.' if banana_on_floor else 'This means the banana is ON TOP OF THE BOX. The monkey MUST: 1) Walk to the box location, 2) Climb on the box, 3) Then grab the banana. NO EXCEPTIONS - climbing is REQUIRED.'}

Convert this symbolic plan into a clear, step-by-step explanation in plain English.
IMPORTANT CORRECTIONS:
- The monkey starts at {monkey_loc} on the GROUND
- The banana is at {banana_loc} {'on the FLOOR' if banana_on_floor else 'on top of the box'}
- {'Since the banana is on the FLOOR, the monkey only needs to WALK to the banana location and GRAB it. NO CLIMBING NEEDED.' if banana_on_floor else 'Since the banana is on the box, the monkey needs to move to the box, climb it, then grab the banana.'}

If the plan is missing a walk-to action but monkey and banana are at different locations, add it: "First, the monkey walks from {monkey_loc} to {banana_loc}."

Make it easy to understand for someone not familiar with symbolic planning.
Use natural language and be conversational but clear.

Format your response as a numbered list of steps.
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that explains plans in plain English."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            explanation = response.choices[0].message.content.strip()
            logger.info("Plan explanation generated")
            return explanation
            
        except Exception as e:
            logger.error(f"Error explaining plan: {e}")
            # Fallback explanation
            return self._fallback_explanation(plan)
    
    def _fallback_explanation(self, plan: List[str]) -> str:
        """
        Generate a simple fallback explanation if LLM fails.
        
        Args:
            plan: List of action names
            
        Returns:
            Simple text explanation
        """
        explanations = []
        for i, action in enumerate(plan, 1):
            if action.startswith("Move"):
                # Parse Move(monkey, from, to)
                parts = action.replace("Move(monkey, ", "").replace(")", "").split(", ")
                if len(parts) == 2:
                    explanations.append(f"{i}. Move the monkey from {parts[0]} to {parts[1]}")
            elif action.startswith("PushBox"):
                # Parse PushBox(from, to)
                parts = action.replace("PushBox(", "").replace(")", "").split(", ")
                if len(parts) == 2:
                    explanations.append(f"{i}. Push the box from {parts[0]} to {parts[1]}")
            elif action == "ClimbUp":
                explanations.append(f"{i}. Climb onto the box")
            elif action == "ClimbDown":
                explanations.append(f"{i}. Climb down from the box")
            elif action == "GrabBanana":
                explanations.append(f"{i}. Grab the banana")
            else:
                explanations.append(f"{i}. {action}")
        
        return "\n".join(explanations)

