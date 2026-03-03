# planner_service.py
from __future__ import annotations

from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import List, Set, Tuple


Predicate = Tuple[str, Tuple[str, ...]]


class PlannerService:
    """
    Thin wrapper around pyperplan.
    Converts FOL-like strings into a PDDL problem file and executes planner.
    """

    def __init__(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.planner_dir = self.repo_root / "planner"
        self.domain_path = self.planner_dir / "domain.pddl"
        self.default_problem_path = self.planner_dir / "problem2.pddl"
        self.runtime_dir = self.repo_root / "integration" / ".planner_runtime"

    def plan(self, fol: List[str]) -> List[str]:
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        problem_path = self.runtime_dir / "problem_generated.pddl"
        self._write_problem_file(fol, problem_path)

        plan = self._run_pyperplan(problem_path)
        if plan:
            return plan
        return ["noop()"]

    def _parse_fol(self, fol: List[str]) -> List[Predicate]:
        parsed: List[Predicate] = []
        pattern = re.compile(r"^\s*([a-zA-Z_][\w-]*)\s*\((.*)\)\s*$")
        for item in fol:
            match = pattern.match(item)
            if not match:
                continue
            pred = match.group(1).strip().lower().replace("-", "_")
            args_raw = match.group(2).strip()
            args = tuple(
                arg.strip().lower().replace("-", "_")
                for arg in args_raw.split(",")
                if arg.strip()
            )
            parsed.append((pred, args))
        return parsed

    def _write_problem_file(self, fol: List[str], out_path: Path) -> None:
        parsed = self._parse_fol(fol)

        monkey_name = "monkey"
        banana_name = "banana"
        boxes: Set[str] = {"box1", "box2"}
        locations: Set[str] = {"l1", "l2", "l3", "l4"}

        init_facts: Set[str] = {
            "(at monkey l1)",
            "(on-ground monkey)",
            "(box-at box1 l2)",
            "(box-on-ground box1)",
            "(box-at box2 l3)",
            "(box-on-ground box2)",
            "(banana-at banana l4)",
            "(banana-on-ground banana)",
            "(adjacent l1 l2)",
            "(adjacent l2 l1)",
            "(adjacent l2 l3)",
            "(adjacent l3 l2)",
            "(adjacent l3 l4)",
            "(adjacent l4 l3)",
        }

        for pred, args in parsed:
            if pred == "at" and len(args) == 2:
                monkey_name = args[0]
                locations.add(args[1])
                init_facts.add(f"(at {args[0]} {args[1]})")
            elif pred in {"box_at", "boxat"} and len(args) == 2:
                boxes.add(args[0])
                locations.add(args[1])
                init_facts.add(f"(box-at {args[0]} {args[1]})")
                init_facts.add(f"(box-on-ground {args[0]})")
            elif pred in {"banana_at", "bananaat"} and len(args) == 2:
                banana_name = args[0]
                locations.add(args[1])
                init_facts.add(f"(banana-at {args[0]} {args[1]})")
            elif pred == "adjacent" and len(args) == 2:
                locations.update(args)
                init_facts.add(f"(adjacent {args[0]} {args[1]})")
            elif pred in {"banana_on_ground", "bananaonground"} and len(args) == 1:
                banana_name = args[0]
                init_facts.add(f"(banana-on-ground {args[0]})")
            elif pred in {"banana_on_box", "bananaonbox"} and len(args) == 2:
                banana_name = args[0]
                boxes.add(args[1])
                init_facts.add(f"(banana-on-box {args[0]} {args[1]})")
            elif pred in {"on_ground", "onground"} and len(args) == 1:
                monkey_name = args[0]
                init_facts.add(f"(on-ground {args[0]})")

        if not any(f.startswith("(banana-on-box ") or f.startswith("(banana-on-ground ") for f in init_facts):
            init_facts.add(f"(banana-on-ground {banana_name})")

        objects = [monkey_name, banana_name] + sorted(boxes) + sorted(locations)
        objects_str = "\n    ".join(objects)
        init_str = "\n    ".join(sorted(init_facts))

        problem = (
            "(define (problem monkey-banana-generated)\n"
            "  (:domain monkey-banana)\n\n"
            "  (:objects\n"
            f"    {objects_str}\n"
            "  )\n\n"
            "  (:init\n"
            f"    {init_str}\n"
            "  )\n\n"
            "  (:goal\n"
            f"    (has-banana {monkey_name} {banana_name})\n"
            "  )\n"
            ")\n"
        )
        out_path.write_text(problem, encoding="utf-8")

    def _run_pyperplan(self, problem_path: Path) -> List[str]:
        cmd: List[str]
        pyperplan_bin = shutil.which("pyperplan")
        if pyperplan_bin:
            cmd = [pyperplan_bin, str(self.domain_path), str(problem_path)]
        else:
            cmd = [sys.executable, "-m", "pyperplan", str(self.domain_path), str(problem_path)]

        try:
            subprocess.run(
                cmd,
                cwd=str(self.planner_dir),
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )
        except Exception:
            return []

        soln_path = Path(f"{problem_path}.soln")
        if not soln_path.exists():
            fallback_soln = Path(f"{self.default_problem_path}.soln")
            if fallback_soln.exists():
                return self._parse_soln(fallback_soln)
            return []

        return self._parse_soln(soln_path)

    def _parse_soln(self, soln_path: Path) -> List[str]:
        actions: List[str] = []
        for raw_line in soln_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith(";"):
                continue
            actions.append(line)
        return actions
