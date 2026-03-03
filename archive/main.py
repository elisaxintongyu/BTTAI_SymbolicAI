"""
Main Orchestration Pipeline for Neural-Symbolic LLM Agent

This is the entry point for the complete system that integrates:
1. Vision module for object detection
2. Symbolic planner for generating action sequences
3. LLM agent for goal extraction and plan explanation
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Optional

from vision import VisionModule
from archive.planner import STRIPSPlanner
from llm_agent import LLMAgent

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NeuralSymbolicAgent:
    """
    Main agent class that orchestrates the complete pipeline.
    """
    
    def __init__(self, image_mode: str = "realistic", verbose: bool = False, model_path: Optional[str] = None):
        """
        Initialize the agent with all components.
        
        Args:
            image_mode: "realistic" or "grid" for image processing
            verbose: Enable verbose logging
            model_path: Optional path to custom ONNX model
        """
        if verbose:
            logging.getLogger().setLevel(logging.DEBUG)
        
        # Initialize vision module (supports both DNN and grid modes)
        try:
            self.vision = VisionModule(mode=image_mode, model_path=model_path)
            if model_path:
                logger.info(f"Using custom trained model: {model_path}")
            elif image_mode == "realistic":
                logger.info("Using DNN-based vision module")
            else:
                logger.info("Using grid-based vision module")
        except Exception as e:
            logger.warning(f"Failed to load vision module: {e}")
            raise
        self.planner = STRIPSPlanner()
        self.llm_agent = LLMAgent()
        
        logger.info("Neural-Symbolic Agent initialized")
    
    def process(self, image_path: str, question: str) -> str:
        """
        Process an image and question to generate a plan explanation.
        
        Args:
            image_path: Path to input image
            question: Natural language question
            
        Returns:
            Plain English explanation of the plan
        """
        logger.info("=" * 60)
        logger.info("Starting Neural-Symbolic Agent Pipeline")
        logger.info("=" * 60)
        
        # Step 1: Detect objects and build initial state
        logger.info("\n[Step 1] Vision Processing: Detecting objects...")
        positions = self.vision.detect_objects(image_path)
        initial_state = self.vision.positions_to_symbolic_state(positions)
        
        logger.info(f"Initial state: {initial_state}")
        
        # Visualize detections
        vis_path = Path(image_path).stem + "_detections.png"
        self.vision.visualize_detections(image_path, positions, output_path=vis_path)
        logger.info(f"Detection visualization saved to: {vis_path}")
        
        # Step 2: Extract goal from natural language question
        logger.info("\n[Step 2] LLM Goal Extraction: Converting question to goal...")
        goal = self.llm_agent.extract_goal(question, initial_state)
        logger.info(f"Extracted goal: {goal}")
        
        # Step 3: Generate plan using symbolic planner
        logger.info("\n[Step 3] Symbolic Planning: Generating action sequence...")
        plan = self.planner.plan(initial_state, goal)
        
        if plan is None:
            logger.warning("No plan found!")
            return "I couldn't find a solution to achieve the goal. The problem might be unsolvable with the current state."
        
        logger.info(f"Plan generated: {plan}")
        
        # Step 4: Explain plan using LLM
        logger.info("\n[Step 4] LLM Plan Explanation: Converting plan to natural language...")
        explanation = self.llm_agent.explain_plan(plan, initial_state)
        
        logger.info("\n[Step 5] Complete!")
        logger.info("=" * 60)
        
        return explanation


def main():
    """
    Main entry point for command-line interface.
    """
    parser = argparse.ArgumentParser(
        description="Neural-Symbolic LLM Agent for Monkeys and Bananas Planning Problem",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py -i scene.jpg -q "How can the monkey get the banana?"
  python main.py -i grid.png -q "Get the banana" --mode grid
  python main.py -i image.jpg -q "Grab banana" --verbose
        """
    )
    
    parser.add_argument(
        "-i", "--image",
        required=True,
        help="Path to input image (realistic or grid-based)"
    )
    
    parser.add_argument(
        "-q", "--question",
        required=True,
        help="Natural language question about the planning problem"
    )
    
    parser.add_argument(
        "-m", "--mode",
        choices=["realistic", "grid"],
        default="realistic",
        help="Image processing mode (default: realistic)"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--model-path",
        default=None,
        help="Path to custom ONNX model (overrides default YOLOX model)"
    )
    
    args = parser.parse_args()
    
    # Validate image path
    image_path = Path(args.image)
    if not image_path.exists():
        logger.error(f"Image file not found: {args.image}")
        sys.exit(1)
    
    try:
        # Initialize agent with optional custom model
        model_path = args.model_path if hasattr(args, 'model_path') and args.model_path else None
        agent = NeuralSymbolicAgent(image_mode=args.mode, verbose=args.verbose, model_path=model_path)
        
        # Process image and question
        result = agent.process(str(image_path), args.question)
        
        # Print result
        print("\n" + "=" * 60)
        print("PLAN EXPLANATION")
        print("=" * 60)
        print(result)
        print("=" * 60 + "\n")
        
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=args.verbose)
        sys.exit(1)


if __name__ == "__main__":
    main()

