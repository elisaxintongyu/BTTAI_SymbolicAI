# Integrating Deep Learning Models With Symbolic Approaches to AI

---

### 👥 **Team Members**

| Name                | GitHub Handle | Contribution                                                                           |
| ------------------- | ------------- | -------------------------------------------------------------------------------------- |
| Era Kalaja          | @csera5       | Computer Vision object detection, Data Exploration, Integration                        |
| Michelle Zuckerberg | @mlzuckerberg | STRIPS planner component development, integration documentation, and team coordination |
| Elisa Yu            | @elisaxintongyu|Built the frontend and integrated backend modules to enable transparent, end-to-end neural-symbolic reasoning from visual input to interpretable action plans.        |
| Bhargavi Patil      | @bhar024      | Deployed LLM using GPT4ALL for conversion of natural language into FOL                 |
|                     |               |                                                                                        |
|                     |               |                                                                                        |

---

## 🎯 **Project Highlights**

- Developed an integrated neural-symbolic AI system combining computer vision, large language models, and a STRIPS planner to address the challenge of translating natural-language questions and visual scenes into executable, interpretable action plans.
- Achieved reliable end-to-end system performance, including high-accuracy object detection, consistent symbolic fact generation, accurate goal translation, and valid action sequences, demonstrating the value of a multi-component pipeline for interpretable reasoning within MIT Lincoln Laboratory's research context.
- Generated actionable insights across the full pipeline, transforming raw images into symbolic propositions, natural language into formal logic, and planner outputs into human-readable explanations, enabling users and stakeholders to understand why and how the system reached its conclusions.
- Implemented a hybrid methodology integrating YOLOv8-based perception, OpenAI-powered goal extraction, and Pyperplan STRIPS planning, satisfying industry expectations for explainable AI by ensuring each module produced traceable, verifiable intermediate outputs.

---

## 👩🏽‍💻 **Setup and Installation**

### **Prerequisites**

- Python 3.7 or higher
- pip (Python package installer)
- OpenAI API key (for LLM component)

### **1. Clone the Repository**

```bash
git clone https://github.com/elisaxintongyu/BTTAI_SymbolicAI.git
cd BTTAI_SymbolicAI
```

### **2. Install Dependencies**

Install all required Python packages and setting up Backend Environment:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload
```

Key dependencies include:

- `ultralytics` (YOLOv8 for computer vision)
- `pyperplan` (STRIPS planner)
- `openai` (LLM integration)
- `torch` and `torchvision` (deep learning frameworks)
- `opencv-python` (image processing)
- `onnxruntime` (model inference)

### **3. Set Up Environment Variables**

Create a `.env` file in the project root directory:

```bash
touch .env
```

Add your OpenAI API key to the `.env` file:

```
OPENAI_API_KEY=your_api_key_here
```

**Note:** The LLM component requires a valid OpenAI API key. You can obtain one from [OpenAI&#39;s website](https://platform.openai.com/api-keys).

### **4. Access the Dataset**

The dataset is included in the repository under `data/monkey_dataset/`:

- Training images: `data/monkey_dataset/train/`
- Validation images: `data/monkey_dataset/valid/`
- Test images: `data/monkey_dataset/test/`

Each split contains both `images/` and `labels/` directories with YOLO-format annotations.

### **5. Run the System**

**Basic Usage:**

Run the frontend for Neural Symbolic Monkeys:

```bash
npm install
npm run dev 
```

---

## 🏗️ **Project Overview**

This project was created as part of the Break Through Tech AI Program, which provides students with industry experience in building real-world AI systems. Through the program's AI Studio, our team partnered with MIT Lincoln Laboratory to explore neural-symbolic AI: the combination of neural network perception with symbolic reasoning.

Our project focused on building an end-to-end neural-symbolic reasoning system that can interpret an image of a scene, understand a natural-language question about it, and generate an interpretable sequence of actions to achieve a goal. To accomplish this, we developed:

- A computer vision module using YOLOv8 to detect objects and convert them into symbolic propositions
- An LLM component that translates user questions into formal logic goals and explains planner output
- A STRIPS planning module (Pyperplan) that computes valid action sequences using symbolic rules
- A custom frontend and integration layer that connects all components into an interactive user experience

This work addresses a broader real-world challenge: creating AI systems that are both capable and explainable. Neural-symbolic systems offer a path toward AI that can perceive complex environments, reason over them, and clearly communicate its decision process. The potential applications span robotics, autonomous agents, decision support, and any domain where transparent, verifiable reasoning is essential.

Our project demonstrates a practical example of how perception, language understanding, and symbolic planning can work together, moving toward more trustworthy, interpretable, and adaptable AI systems.

---

## 📊 **Data Exploration**

### **Dataset Description**

Our project used two custom-curated datasets designed to support both perception and symbolic reasoning components of the system:

1. **Realistic Image Dataset**: Contained images representing monkeys, bananas, and boxes in a room-like environment. These images were manually annotated using Roboflow, producing YOLO-format bounding boxes.
2. **Abstract Grid Dataset**: A semantically equivalent dataset where objects were represented as colored cells on a grid. This allowed controlled experiments on scene understanding without visual noise or variation.

Both datasets supported multi-class object detection, enabling us to identify the monkey, banana, and multiple box types required for downstream symbolic reasoning.

### **Format, Size & Structure**

- **Annotation Format**: YOLOv8 annotations (class, x_center, y_center, width, height)
- **Data Types**: Raster images + bounding box labels
- **Classes**: Monkey, Banana, Box A-E, Background
- **Purpose**: Provide object-level spatial information to be converted into symbolic propositions for planning

### **Preprocessing**

- Image preprocessing: resizing, normalization, and batch preparation
- Train/val/test splits: ensuring generalizable performance across both datasets

### **Challenges & Assumptions**

- Small dataset size: required careful annotation and data augmentation to avoid overfitting.

---

## 🧠 **Model Development**

### **Models Used**

- Our computer vision component is built on YOLOv8n, a lightweight, high-speed object detection model well-suited for real-time detection and small custom datasets. YOLOv8 served as both a feature extractor, learning spatial patterns through convolutional layers and a detection head, predicting bounding boxes, object classes, and confidence scores. This model was selected for its strong performance on small datasets and its ability to reliably detect multiple objects required by the symbolic planner (monkey, banana, boxes A–E).

### **Features & Hyperparameter Strategy**

- Since YOLOv8 extracts hierarchical spatial features automatically, manual feature selection was not required. Instead, the focus was on optimizing training behavior through: Learning rate tuning via YOLO's built-in scheduler, batch size adjustments to prevent overfitting on a small dataset, and data augmentation (horizontal flips, brightness/contrast shifts) to improve generalization.

### **Training Setup**

- The Computer Vision model had a train/validation/test split of 70/20/10. It was evaluated on box_loss, precision, and recall.

---

## 🧩 **STRIPS Planner Component**

### **Overview**

The STRIPS (Stanford Research Institute Problem Solver) planner is a symbolic reasoning module that generates executable action sequences to achieve goals specified in formal logic. This component bridges the gap between the neural perception layer (computer vision) and the natural language interface (LLM), translating symbolic world states into interpretable action plans.

### **Implementation**

**Pyperplan Integration:**

- **Tool**: [Pyperplan](https://github.com/aibasel/pyperplan) - A lightweight, Python-based STRIPS planner
- **Input Format**: PDDL (Planning Domain Definition Language) files
  - `domain.pddl` - Defines the planning domain with predicates and actions
  - `problem.pddl` - Defines initial state, objects, and goal for specific scenarios
- **Output**: Action sequence plan (`.soln` file) containing ordered steps to achieve the goal

### **Planning Domain: Monkey and Banana World**

**Predicates (World State Representation):**

- **Location Predicates**: `(at ?monkey ?location)`, `(box-at ?box ?location)`, `(banana-at ?banana ?location)`
- **Height/Level Predicates**: `(on-ground ?monkey)`, `(on-box ?monkey ?box)`, `(box-on-ground ?box)`, `(banana-on-ground ?banana)`, `(banana-on-box ?banana ?box)`
- **Movement Predicates**: `(adjacent ?loc1 ?loc2)` - Defines which locations are connected
- **Goal Predicate**: `(has-banana ?monkey ?banana)` - Represents successful goal achievement

**Actions (Available Operations):**

1. **`move(?monkey ?from ?to)`** - Monkey moves between adjacent locations (must be on ground)
2. **`climb_on(?monkey ?box ?location)`** - Monkey climbs onto a box at the same location
3. **`climb_off(?monkey ?box ?location)`** - Monkey climbs off a box back to ground
4. **`push_box(?monkey ?box ?from ?to)`** - Monkey pushes a box to an adjacent location (must be on ground)
5. **`grab_banana_from_ground(?monkey ?banana ?location)`** - Monkey grabs a banana from the ground
6. **`grab_banana_from_box(?monkey ?banana ?box ?location)`** - Monkey grabs a banana from on top of a box

### **Search Algorithms**

The planner supports multiple search strategies:

- **Breadth-First Search (BFS)** - Default algorithm, guarantees shortest solution path
- **Greedy Best-First Search (GBF)** - Uses heuristics (e.g., FF heuristic) to explore promising paths first, faster for complex problems
- **A\* Search** - Combines path cost and heuristic estimates for optimal solutions with better efficiency than pure BFS

### **Integration with System Pipeline**

1. **Input**: Receives symbolic initial state from vision module (object positions, relationships) and goal from LLM component (translated from natural language)
2. **Processing**: Converts state/goal dictionaries to PDDL problem format, runs pyperplan search algorithm
3. **Output**: Returns ordered list of actions (e.g., `["move monkey1 posD posC", "move monkey1 posC posB", "grab_banana_from_box monkey1 banana1 boxB posB"]`)
4. **Explanation**: Action sequence is passed to LLM component for natural language explanation

### **Example Scenario**

**Initial State:**

- Monkey at position D
- Box B at position B with banana on top
- Box C at position C
- Adjacency: A↔B↔C↔D

**Goal:** `(has-banana monkey1 banana1)`

**Generated Plan (3 steps):**

1. `(move monkey1 posD posC)` - Move from D to C
2. `(move monkey1 posC posB)` - Move from C to B
3. `(grab_banana_from_box monkey1 banana1 boxB posB)` - Grab the banana from box B

### **Usage**

The planner component is integrated via the `STRIPSPlanner` class (aliased from `PyperplanPlanner`) in `planner.py`:

```python
from planner import STRIPSPlanner

planner = STRIPSPlanner(domain_file="planner/domain.pddl")
plan = planner.plan(
    initial_state={"monkey_location": "posD", "banana_location": "posB", ...},
    goal={"has_banana": True},
    search_algorithm="bfs"
)
```

### **Limitations**

- **Single-level climbing only** - Monkey can climb onto one box, but not onto stacked boxes
- **No box stacking** - Domain lacks `(box-on-box)` predicate, cannot represent vertically stacked boxes
- **Discrete movement** - Monkey moves between adjacent positions without intermediate steps
- **Fixed reach** - Monkey reach height is 3 units (can grab from ground or single box, but not from multiple stacked boxes)

---

## 🤖 **LLM Component**

The LLM component translates natural language queries into formal PDDL goal specifications and converts planner output back into natural language explanations. This addresses the ambiguity of natural language (e.g "get the banana" vs "grab the banana") while maintaining the accuracy required for symbolic planning.

### **Implementation**

Two interchangeable backends are supported:

**OpenAI API**
- Cloud-based inference
- High-quality output
- Requires API key

**GPT4All Local**
- Primary model: Mistral-7B-Instruct (Q4_0, 4GB)
- Fallback model: Orca-Mini-3B (Q4_0, 2GB)
- no API key required
- Slower inference

---

## 🌐 Web Development & System Integration

The web development component of this project provides an interactive interface that connects the full neural-symbolic pipeline—bridging perception, reasoning, and planning into a cohesive user experience. The goal of the web layer was not only to enable usability, but also to surface **intermediate symbolic representations** to support interpretability and debugging.

### Frontend Overview

The frontend was implemented as a lightweight, modular web interface that allows users to:
- Upload an image of a scene (realistic or abstract)
- Enter a natural-language question or goal
- View structured outputs produced at each stage of the pipeline

The interface was designed to emphasize **transparency**, enabling users to inspect how raw visual inputs and natural-language queries are transformed into symbolic logic and executable plans.

**Key responsibilities of the frontend include:**
- Collecting user inputs (image + question)
- Triggering backend inference and planning workflows
- Rendering results in a structured, human-readable format

Displayed outputs include:
- Detected objects and their inferred spatial relationships
- Generated symbolic facts (predicates) derived from vision
- Planner-generated action sequences
- Natural-language explanations of the plan

This design ensures that users can trace the system’s reasoning process end to end, rather than treating it as a black box.

### Backend Integration Layer

The backend integration layer acts as the orchestration point for all system components. It exposes a unified interface that the frontend interacts with, while internally coordinating:

1. **Computer Vision Inference**  
   - Receives the uploaded image
   - Runs YOLO-based object detection
   - Converts detections into symbolic state representations

2. **LLM-Based Reasoning**  
   - Translates natural-language questions into formal goal specifications
   - Generates natural-language explanations from planner outputs

3. **Symbolic Planning**  
   - Constructs PDDL problem files from symbolic state and goals
   - Executes the STRIPS planner
   - Returns ordered action sequences or failure states

The integration layer ensures consistent data formats and clean handoffs between modules, allowing each component to be developed and tested independently while still functioning as part of a unified system.

### Design Philosophy

The web development effort prioritized:
- **Modularity**: Each component (vision, LLM, planner) communicates through clearly defined interfaces.
- **Interpretability**: Intermediate outputs are preserved and displayed rather than hidden.
- **Extensibility**: The architecture supports future additions such as interactive plan editing, alternative planners, or robotic execution backends.

By tightly integrating the frontend with the backend pipeline, the web component transforms the project from a collection of standalone models into a usable, explainable neural-symbolic AI system.

---

## 📈 **Results & Key Findings**

### **Computer Vision Model**

- Computer vision model metrics included: Recall, Precision, box_loss, mAP50. It performed very well across the dataset, with only some challenges distinguishing box E from box B. This challenge likely occurred since box B is seen in the training data over 80 times, while box E is only seen around 6 times so the model was biased towards box B. See below visualizations:

### **STRIPS Planner Component**

- **Success Rate**: The planner successfully generates valid action sequences for all solvable scenarios in the monkey-banana domain. It correctly handles various initial configurations including different monkey positions, box placements, and banana locations.
- **Performance Metrics**:

  - **Plan Generation Time**: Average search time of ~0.003 seconds for typical problems using breadth-first search
  - **Plan Length**: Successfully finds optimal solutions (typically 3-5 steps for standard scenarios)
  - **Search Algorithm Comparison**: BFS guarantees shortest paths, while greedy best-first search with FF heuristic provides faster solutions for more complex problems
- **Integration Success**: The planner seamlessly integrates with the vision module (receiving symbolic state) and LLM component (receiving goals and outputting action sequences). The PDDL problem generation correctly converts dictionary-based state representations into valid PDDL format.
- **Domain Coverage**: Successfully handles all defined actions (move, climb_on, climb_off, push_box, grab_banana_from_ground, grab_banana_from_box) and correctly enforces preconditions and effects for each action.
- **Limitations Observed**: The planner correctly identifies unsolvable scenarios (e.g., when goals are impossible given initial state constraints) and returns appropriate error handling.

### **LLM Component**

_Results and key findings to be filled in by the LLM team._

---

## 🚀 **Next Steps**

### **STRIPS Planner**

- **Stacked boxes** - Would need to add a `(box-on-box)` predicate to represent vertically stacked boxes
- **Box stacking actions** - Would need to add stacking actions for Monkey as well as climbing the stacks
- **Multiple simultaneous goals** - Potentially get one banana first, then re-run to get second, and so on

### **Computer Vision**

- Generalize the CV model for more real world applications

### **LLM**

- Experiment with other models to find an optimal output

---

## 📝 **License**

This project is licensed under the MIT License.

---

## 🙏 **Acknowledgements**

We would like to express our sincere gratitude to everyone who supported and guided us throughout this project.

Our Challenge Advisors at MIT Lincoln Laboratory

- Lee Martie, Technical Staff
- Sandra Hawkins, Assistant Staff

Our TA

- Mimi Lohanimit, EECS Graduate Researcher

Thank you for sharing your expertise, providing thoughtful technical direction, and helping us navigate the complexities of neural-symbolic AI.

Break Through Tech AI Program

Thank you for creating this opportunity and supporting our growth as emerging AI practitioners through hands-on, real-world experience.

AI Studio Teaching Assistants and Program Staff

Your feedback, mentorship, and encouragement were essential in helping us refine our ideas and successfully integrate each component of our system.

Finally, a big thank you to everyone behind the scenes who contributed resources, infrastructure, and continuous support throughout the development of this project.
