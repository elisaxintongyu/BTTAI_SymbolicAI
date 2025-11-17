"""
Symbolic Planner Module for Neural-Symbolic LLM Agent

Uses pyperplan (https://github.com/aibasel/pyperplan) - a lightweight STRIPS planner
written in Python for the Monkeys and Bananas problem.
"""

import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PyperplanPlanner:
    """
    Wrapper around pyperplan for the Monkeys and Bananas planning problem.
    Uses pyperplan via subprocess (command-line interface).
    """
    
    def __init__(self, domain_file: Optional[str] = None):
        """
        Initialize the pyperplan-based planner.
        
        Args:
            domain_file: Path to PDDL domain file (default: uses domain.pddl in project root)
        """
        # Get domain file path
        if domain_file is None:
            # Use domain.pddl in the same directory as this file
            current_dir = Path(__file__).parent
            domain_file = str(current_dir / "domain.pddl")
        
        self.domain_file = domain_file
        
        if not os.path.exists(self.domain_file):
            raise FileNotFoundError(f"Domain file not found: {self.domain_file}")
        
        # Check if pyperplan is available
        try:
            result = subprocess.run(
                ["pyperplan", "--help"],
                capture_output=True,
                timeout=5
            )
            if result.returncode != 0:
                raise FileNotFoundError("pyperplan command not found")
        except FileNotFoundError:
            raise ImportError(
                "pyperplan is not installed or not in PATH. "
                "Install it with: pip install pyperplan"
            )
        
        logger.info(f"Pyperplan planner initialized with domain: {self.domain_file}")
    
    def _state_to_pddl_problem(self, initial_state: Dict[str, Any], 
                               goal: Dict[str, Any]) -> str:
        """
        Convert symbolic state to PDDL problem file content in the new format.
        
        Args:
            initial_state: Initial world state dictionary
            goal: Goal state dictionary
            
        Returns:
            PDDL problem file content as string
        """
        # Extract spatial information
        monkey_pos = initial_state.get("monkey_location")
        box_pos = initial_state.get("box_location")
        banana_pos = initial_state.get("banana_location")
        
        # Determine box labels (A and B) - A is right, B is left
        # For now, use single box, but structure supports multiple
        boxes = ["A", "B"]  # Can be extended for multiple boxes
        
        # Build initial state predicates
        init_predicates = []
        
        # Spatial relationships (LEFT-OF, RIGHT-OF)
        # Based on image: Box B is left, Box A is right, monkey between them
        if len(boxes) >= 2:
            init_predicates.append("    (LEFT-OF B A)")
            init_predicates.append("    (RIGHT-OF A B)")
            # Monkey is between boxes, closer to B
            init_predicates.append("    (LEFT-OF monkey A)")
            init_predicates.append("    (RIGHT-OF monkey B)")
        
        # Clear paths - monkey can walk to banana location
        if monkey_pos and banana_pos:
            # Add path from monkey's location to banana's location
            init_predicates.append(f"    (CLEAR-PATH-BETWEEN monkey {banana_pos})")
            # Also add path from monkey to banana object
            init_predicates.append("    (CLEAR-PATH-BETWEEN monkey banana)")
        if monkey_pos and box_pos:
            init_predicates.append("    (CLEAR-PATH-BETWEEN monkey B)")
        
        # Box states
        for box in boxes:
            init_predicates.append(f"    (CLEAR-ON-TOP {box})")
        
        # Object positions (using locations)
        locations = set()
        if monkey_pos:
            locations.add(monkey_pos)
            init_predicates.append(f"    (AT monkey {monkey_pos})")
        if box_pos:
            locations.add(box_pos)
            init_predicates.append(f"    (AT-OBJ A {box_pos})")
        if banana_pos:
            locations.add(banana_pos)
            init_predicates.append(f"    (AT-OBJ banana {banana_pos})")
        
        # On-floor predicates
        init_predicates.append("    (ON-FLOOR monkey)")
        init_predicates.append("    (ON-FLOOR A)")
        if len(boxes) >= 2:
            init_predicates.append("    (ON-FLOOR B)")
        
        # Check if banana is on floor or on box
        if initial_state.get("banana_on_box", False) or initial_state.get("box_at_banana", False):
            # Banana is on top of box A
            init_predicates.append("    (ON-BOX banana A)")
            # Don't include CLEAR-ON-TOP A since banana is on it
            # Remove it from the list if it was added
            init_predicates = [p for p in init_predicates if "CLEAR-ON-TOP A" not in p]
        else:
            init_predicates.append("    (ON-FLOOR banana)")
        
        # Hand state
        if not initial_state.get("has_banana", False):
            init_predicates.append("    (HAND-EMPTY monkey)")
        
        # Build goal predicates
        goal_predicates = []
        
        if goal.get("has_banana") is True:
            goal_predicates.append("    (HAS monkey banana)")
        elif goal.get("has_banana") is False:
            goal_predicates.append("    (not (HAS monkey banana))")
        
        # If no goal predicates specified, default to has-banana
        if not goal_predicates:
            goal_predicates.append("    (HAS monkey banana)")
        
        # Build PDDL problem file
        init_str = "\n".join(init_predicates) if init_predicates else "    ; No initial predicates"
        goal_str = "\n".join(goal_predicates) if goal_predicates else "    (HAS monkey banana)"
        
        # Add location objects
        locations_list = sorted(list(locations))
        locations_str = "\n    ".join([f"{loc} - location" for loc in locations_list]) if locations_list else ""
        
        problem_content = f"""(define (problem canvas_0_banana1_monkey1_box2)
  (:domain MONKEY)
  
  (:objects
    A B - box
    banana - fruit
    monkey - mammal
    {locations_str}
  )
  
  (:INIT
{init_str}
  )
  
  (:goal
    (and
{goal_str}
    )
  )
)
"""
        return problem_content
    
    def plan(self, initial_state: Dict[str, Any], goal: Dict[str, Any],
             search_algorithm: str = "bfs", heuristic: Optional[str] = None,
             max_depth: int = 50) -> Optional[List[str]]:
        """
        Generate a plan using pyperplan.
        
        Args:
            initial_state: Initial world state dictionary
            goal: Goal state dictionary
            search_algorithm: Search algorithm to use (default: "bfs")
                             Options: "bfs", "gbf", "astar", "dijkstra", etc.
            heuristic: Heuristic to use (default: None)
                      Options: "hff", "hadd", "hmax", etc.
            max_depth: Maximum search depth (default: 50)
            
        Returns:
            List of action names representing the plan, or None if no plan found
        """
        logger.info(f"Planning from initial state: {initial_state}")
        logger.info(f"Goal: {goal}")
        logger.info(f"Using search algorithm: {search_algorithm}, heuristic: {heuristic}")
        
        # Generate PDDL problem file
        problem_content = self._state_to_pddl_problem(initial_state, goal)
        
        # Create temporary problem file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pddl', delete=False) as f:
            problem_file = f.name
            f.write(problem_content)
        
        try:
            # Build pyperplan command
            cmd = ["pyperplan"]
            
            # Add search algorithm and heuristic if specified
            if search_algorithm and search_algorithm != "bfs":
                if search_algorithm == "gbf" and heuristic:
                    cmd.extend(["-H", heuristic, "-s", "gbf"])
                elif search_algorithm == "astar" and heuristic:
                    cmd.extend(["-H", heuristic, "-s", "astar"])
                else:
                    cmd.extend(["-s", search_algorithm])
            
            cmd.extend([self.domain_file, problem_file])
            
            logger.debug(f"Running command: {' '.join(cmd)}")
            
            # Run pyperplan
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode != 0:
                logger.error(f"Pyperplan failed with return code {result.returncode}")
                logger.error(f"Error output: {result.stderr}")
                return None
            
            # Parse plan from output
            plan = self._parse_plan_output(result.stdout, problem_file)
            
            if plan:
                logger.info(f"Plan found with {len(plan)} actions")
            else:
                logger.warning("No plan found!")
            
            return plan
            
        except subprocess.TimeoutExpired:
            logger.error("Planning timed out")
            return None
        except Exception as e:
            logger.error(f"Error during planning: {e}", exc_info=True)
            return None
        finally:
            # Clean up temporary problem file
            try:
                os.unlink(problem_file)
                # Also remove solution file if it exists
                solution_file = problem_file.replace('.pddl', '.soln')
                if os.path.exists(solution_file):
                    os.unlink(solution_file)
            except:
                pass
    
    def _parse_plan_output(self, output: str, problem_file: str) -> Optional[List[str]]:
        """
        Parse plan from pyperplan output or solution file.
        
        Args:
            output: stdout from pyperplan
            problem_file: Path to problem file (solution file will be problem_file.soln)
            
        Returns:
            List of action names, or None if no plan found
        """
        plan = []
        
        # Check for solution file first (pyperplan writes plan to .soln file)
        # Pyperplan creates solution file as {problem_file}.soln (keeps .pddl extension)
        solution_file = problem_file + '.soln'
        if os.path.exists(solution_file):
            try:
                with open(solution_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        # Skip comments and empty lines
                        if line and not line.startswith(';') and not line.startswith('INFO'):
                            # Remove parentheses if present
                            line = line.strip('()')
                            # Only add if it looks like an action (contains action name)
                            if any(action in line.lower() for action in ['move', 'push', 'climb', 'grab']):
                                plan.append(line)
                if plan:
                    logger.debug(f"Read {len(plan)} actions from solution file")
                    return plan
            except Exception as e:
                logger.debug(f"Could not read solution file: {e}")
        
        # Try to parse from stdout - look for action patterns
        lines = output.split('\n')
        for line in lines:
            line = line.strip()
            # Look for lines that contain action names and are not log messages
            if (line and not line.startswith(';') and 
                not line.startswith('INFO') and 
                not line.startswith('Search') and
                not 'parsed' in line.lower() and
                not 'grounding' in line.lower() and
                any(action in line.lower() for action in ['move', 'push', 'climb', 'grab'])):
                # Remove parentheses if present
                line = line.strip('()')
                if line:
                    plan.append(line)
        
        return plan if plan else None


# Alias for backward compatibility
STRIPSPlanner = PyperplanPlanner
