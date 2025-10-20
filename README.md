# Neural Symbolic Monkeys - Planner Component

## Project Overview

This is a part of the team project for the AI Studio Challenge. The system combines:

- Computer Vision: Detects objects in images
- LLM Agent: Converts visual data to symbolic propositions
- STRIPS Planner: Generates action sequences to achieve goals
- Response System: Executes the plan

In this overall project, we are interested in creating a neural-symbolic system that combines large language models, vision models, and a planner to answer planning questions. We create a project that is a spin on an old symbolic planning problem of finding the sequence of steps a monkey needs to do to get bananas in a room.

## Setup Instructions

### **Prerequisites**

- Python 3.7 or higher
- pip (Python package installer)

### **Installation**

1. **Install Pyperplan:**

   ```bash
   pip install pyperplan
   ```
2. **Verify Installation:**

   ```bash
   pyperplan --help
   ```
3. **Clone/Download Project Files:**

   - `domain.pddl` - Planning domain definition
   - `problem.pddl` - Example problem file
   - `README.md` - This documentation

### **Quick Test**

```bash
# Run the example problem
pyperplan domain.pddl problem.pddl

# View the solution
cat problem.pddl.soln
```

### **Expected Output**

```
(move monkey1 posd posc)
(move monkey1 posc posb)
(grab_banana monkey1 banana1 posb)
```

## Domain: Monkey and Banana World

### **PDDL Domain and Problem Files**

The STRIPS planner component is now fully implemented using [Pyperplan](https://github.com/aibasel/pyperplan), a lightweight STRIPS planner written in Python.

### **Files Created:**

- `domain.pddl` - Defines the planning domain with predicates and actions
- `problem.pddl` - Defines the initial state and goal for specific scenarios
- `problem.pddl.soln` - Generated solution plan (auto-created by pyperplan)

### **Propositions (World State):**

**Location Predicates:**

- `(at ?monkey ?location)` - Monkey is at a specific location
- `(box-at ?box ?location)` - Box is at a specific location
- `(banana-at ?banana ?location)` - Banana is at a specific location

**Height/Level Predicates:**

- `(on-ground ?monkey)` - Monkey is on the ground
- `(on-box ?monkey ?box)` - Monkey is on top of a box
- `(box-on-ground ?box)` - Box is on the ground
- `(banana-on-ground ?banana)` - Banana is on the ground
- `(banana-on-box ?banana ?box)` - Banana is on top of a box

**Movement Predicates:**

- `(adjacent ?loc1 ?loc2)` - Two locations are adjacent (for movement)

**Goal Predicate:**

- `(has-banana ?monkey ?banana)` - Monkey has successfully grabbed the banana

### **Actions (What the monkey can do):**

1. **`move(?monkey ?from ?to)`** - Monkey moves between adjacent locations

   - *Precondition:* Monkey at source location, locations are adjacent, monkey on ground
   - *Effect:* Monkey moves to destination location
2. **`climb_on(?monkey ?box ?location)`** - Monkey climbs onto a box

   - *Precondition:* Monkey and box at same location, monkey on ground
   - *Effect:* Monkey is now on the box
3. **`climb_off(?monkey ?box ?location)`** - Monkey climbs off a box

   - *Precondition:* Monkey on box, box at location
   - *Effect:* Monkey is now on the ground
4. **`push_box(?monkey ?box ?from ?to)`** - Monkey pushes a box to new location

   - *Precondition:* Monkey and box at source location, locations adjacent, monkey on ground
   - *Effect:* Box moves to destination location
5. **`grab_banana(?monkey ?banana ?location)`** - Monkey grabs a banana

   - *Precondition:* Monkey and banana at same location
   - *Effect:* Monkey has the banana (goal achieved)

### **Example Scenario (Based on Provided Image):**

**Initial State:**

- Monkey at position D (rightmost)
- Box D at position A (leftmost)
- Box B at position B (middle) with banana on top
- Box C at position C (rightmost)
- Adjacency: A↔B↔C↔D

**Goal:** `(has-banana monkey1 banana1)`

**Generated Solution (3 steps):**

1. `(move monkey1 posd posc)` - Move from D to C
2. `(move monkey1 posc posb)` - Move from C to B
3. `(grab_banana monkey1 banana1 posb)` - Grab the banana

### **Testing Results:**

Successfully tested with Pyperplan using both breadth-first search and greedy best-first search with FF heuristic
Planner correctly finds solutions for achievable goals
Plan length: 3 steps for the example scenario
Search time: ~0.003 seconds

### **Domain Limitations:**

**Current Implementation Constraints:**

- **Single-level climbing only** - Monkey can climb onto one box, but not onto boxes that are already stacked
- **Discrete movement** - Monkey "teleports" between adjacent positions (no intermediate steps)
- **Fixed heights** - Monkey (3 units), Boxes/Bananas (1 unit each)
- **Ground-level pushing** - Monkey can only push boxes when on the ground
- **No multi-banana goals** - Current goal predicate only handles one banana at a time

**Scenarios NOT Supported:**

- **Stacked boxes** - Cannot climb onto boxes that are on top of other boxes
- **Multi-level climbing** - Cannot climb from one box to another box
- **Complex stacking** - Cannot handle scenarios like "climb box A, then climb box B on top of A"
- **Multiple simultaneous goals** - Cannot handle "get banana1 AND banana2" in one plan
- **Realistic physics** - No gravity, momentum, or physical constraints

**Height Constraint Examples:**

- **Reachable**: Banana on single box (total height = 2, monkey height = 3)
- **Unreachable**: Banana on 4 stacked boxes (total height = 5, monkey height = 3)