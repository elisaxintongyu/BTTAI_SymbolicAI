# Neural Symbolic Monkeys - Planner Component

## Project Overview

This is a part of the team project for the AI Studio Challenge. The system combines:

- Computer Vision: Detects objects in images
- LLM Agent: Converts visual data to symbolic propositions
- STRIPS Planner: Generates action sequences to achieve goals
- Response System: Executes the plan

In this overall project, we are interested in creating a neural-symbolic system that combines large language models, vision models, and a planner to answer planning questions. We create a project that is a spin on an old symbolic planning problem of finding the sequence of steps a monkey needs to do to get bananas in a room.

## What is the Planner?

### **STRIPS Planning**

STRIPS (Stanford Research Institute Problem Solver) is a classical AI planning approach that:

- Represents world states as sets of logical propositions (predicates)
- Defines actions with preconditions (what must be true to execute) and effects (what changes)
- Searches through possible action sequences to find a path from initial state to goal state

### **Pyperplan**

[Pyperplan](https://github.com/aibasel/pyperplan) is a lightweight STRIPS planner written in Python. It:

- **Input**: Takes PDDL (Planning Domain Definition Language) files:
  - `domain.pddl` - Defines the "rules of the world" (predicates and actions)
  - `problem.pddl` - Defines a specific scenario (initial state, objects, goal)
- **Process**: Uses search algorithms to find valid action sequences (see Search Algorithms section below)
- **Output**: Generates a `.soln` file containing the sequence of actions to achieve the goal

### **Search Algorithms**

The planner explores a tree of possibilities where each node is a world state and each branch is an action.

**1. Breadth-First Search (BFS)**

Strategy: Explore ALL possibilities at distance 1, then ALL at distance 2, then ALL at distance 3, etc.

Example with your monkey problem:
```
Level 0: [Initial state: Monkey at D]
         ↓
Level 1: Try all possible actions from start
         - Move D→C
         - (other actions that don't apply)
         ↓
Level 2: From EACH Level-1 state, try all actions
         - From "Monkey at C": Move C→D, Move C→B, etc.
         ↓
Level 3: From EACH Level-2 state, try all actions
         - Eventually finds: Move D→C, then C→B, then grab_banana ✓
```

- **Pros**: Guaranteed to find the **shortest** solution, systematic
- **Cons**: Explores EVERYTHING (even obviously bad paths), slow for large problems

**2. Greedy Best-First Search (with FF Heuristic)**

Strategy: Use a "smart guess" (heuristic) to explore promising paths first.

The **FF Heuristic** (Fast Forward) estimates: *"How many more actions do I probably need to reach the goal?"*

Example:
```
Initial state: Monkey at D, Banana at B
FF heuristic says: "You're ~3 steps away"

Next, try these actions:
- Move D→C: FF says "now ~2 steps away" ← Looks promising! Explore this first
- Push box: FF says "still ~3 steps away" ← Doesn't help, explore later

After Move D→C:
- Move C→B: FF says "now ~1 step away" ← Great! Explore this
- Move C→D: FF says "now ~4 steps away" ← Going backwards, skip for now

After Move C→B:
- Grab banana: FF says "0 steps away - GOAL!" ✓
```

- **Pros**: Much faster than BFS for complex problems, focuses on promising paths
- **Cons**: NOT guaranteed to find the shortest path, heuristic might be misleading

**3. A\* Search**

Strategy: Combines both approaches - considers:
- **g(n)**: How many steps have I already taken? (like BFS)
- **h(n)**: How many more steps will I probably need? (like Greedy, uses heuristic)
- **f(n) = g(n) + h(n)**: Total estimated cost

Always explores the node with lowest **f(n)** first.

Example:
```
State A: Took 2 steps, estimate 1 more → f = 3
State B: Took 1 step, estimate 3 more → f = 4

A* explores State A first (lower total cost)
```

- **Pros**: **Optimal** - guaranteed to find shortest path (if heuristic is "admissible"), more efficient than pure BFS
- **Cons**: More complex, still slower than pure Greedy for very large problems

**Algorithm Selection:**

By default, Pyperplan uses **BFS**. For this simple monkey-banana problem, all three algorithms find the same 3-step solution quickly. For complex problems with many objects and locations, Greedy or A\* would be significantly faster.

You can specify the search algorithm with:
```bash
pyperplan -s <search_algorithm> domain.pddl problem.pddl
# Options: bfs, astar, gbf (greedy best-first), etc.
```

### **How it Works in This System**

1. **Vision Model** detects objects in an image (boxes, bananas, monkey positions)
2. **LLM Agent** converts visual data into PDDL propositions:
   - `(at monkey1 posD)`
   - `(banana-on-box banana1 boxB)`
   - etc.
3. **Pyperplan** (this component) finds the action sequence:
   - Reads the domain rules
   - Searches for a valid plan
   - Returns steps like: `(move ...)`, `(grab_banana_from_box ...)`
4. **Response System** executes or communicates the plan

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
(grab_banana_from_box monkey1 banana1 boxb posb)
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
5. **`grab_banana_from_ground(?monkey ?banana ?location)`** - Monkey grabs a banana from the ground

   - *Precondition:* Monkey and banana at same location, monkey on ground, banana on ground
   - *Effect:* Monkey has the banana (goal achieved)
6. **`grab_banana_from_box(?monkey ?banana ?box ?location)`** - Monkey grabs a banana from a box

   - *Precondition:* Monkey and banana at same location, monkey on ground, banana on box, box at location
   - *Effect:* Monkey has the banana (goal achieved)

### **Example Scenario (Based on example_image.png):**

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
3. `(grab_banana_from_box monkey1 banana1 boxb posb)` - Grab the banana from box B

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

- **Stacked boxes** - The domain has NO `(box-on-box)` predicate, so it cannot represent boxes stacked on top of other boxes, either in the initial state or as a result of actions. If an input image contains stacked boxes, the system cannot accurately model that scenario.
- **Multi-level climbing** - Monkey cannot climb from one box to another box
- **Box stacking actions** - Monkey cannot stack boxes on top of each other (no `stack_box` action)
- **Multiple simultaneous goals** - Cannot handle "get banana1 AND banana2" in one plan

**Height Constraint Examples:**

The domain enforces height checking through separate grab actions:

- `grab_banana_from_ground` - for bananas on the ground (height 1)
- `grab_banana_from_box` - for bananas on a single box (height 2)

Both scenarios are reachable from ground level (monkey reach = 3 units):

- **Reachable**: Banana on ground (height 1) or on single box (total height = 2)
- **Unreachable (theoretical)**: Banana on 4 stacked boxes (total height = 5, exceeds monkey reach of 3)

**Important:** The domain lacks a `(box-on-box)` predicate, so stacked boxes cannot be represented at all. If an input image shows boxes stacked vertically, the vision/LLM component would need to either:

- Reinterpret stacked boxes as separate boxes at different horizontal positions
- Reject the scenario as unsolvable within the current domain constraints

**To support stacked boxes, the domain would need:**

- New predicate: `(box-on-box ?box1 ?box2)` - to represent vertical stacking
- New action: `stack_box(?monkey ?box1 ?box2 ?location)` - to stack boxes
- Modified `climb_on` action - to allow climbing boxes that may be on other boxes
- Height calculation logic - to track cumulative heights of stacked boxes
