# llm_client.py
from typing import List
from json import dumps, loads

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

try:
    from .config import settings
    from .models import DetectedObject
except ImportError:  # pragma: no cover - supports running from integration/ directly
    from config import settings
    from models import DetectedObject


class LLMClient:
    """
    Thin OOP wrapper around a LangChain ChatOpenAI model.
    Provides methods tailored to this project (build FOL, explain plan).
    """

    def __init__(self) -> None:
        try:
            self._llm = ChatOpenAI(
                model=settings.openai_model,
                temperature=settings.openai_temperature,
            )
        except Exception:
            self._llm = None

    # --- Core low-level call ---

    def _invoke(self, prompt_template: ChatPromptTemplate, **kwargs) -> str:
        if self._llm is None:
            raise RuntimeError("LLM backend is not configured")
        messages = prompt_template.format_messages(**kwargs)
        resp = self._llm.invoke(messages)
        return resp.content.strip()

    # --- High-level helpers ---

    def build_fol_from_scene(
        self, question: str, objects: List[DetectedObject]
    ) -> List[str]:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "You are a vision-reasoning assistant. "
                        "You receive a list of detected objects with bounding boxes, "
                        "and a user's question about the scene. "
                        "Return a set of logical predicates in first-order logic (FOL) "
                        "that describe the scene and are relevant to answering the question.\n\n"
                        "Use a simple syntax like:\n"
                        "  on(object, surface)\n"
                        "  leftOf(object1, object2)\n"
                        "  above(object1, object2)\n\n"
                        "Return ONLY a JSON array of strings, no extra commentary."
                    ),
                ),
                (
                    "user",
                    (
                        "Question: {question}\n\n"
                        "Detected objects (label and bbox [x,y,w,h]):\n"
                        "{objects_json}"
                    ),
                ),
            ]
        )

        objects_json = dumps([o.model_dump() for o in objects], indent=2)
        text = self._invoke(
            prompt_template=prompt,
            question=question,
            objects_json=objects_json,
        )

        # Try to parse as JSON list of strings
        try:
            fol_list = loads(text)
            if isinstance(fol_list, list) and all(
                isinstance(x, str) for x in fol_list
            ):
                return fol_list
        except Exception:
            pass

        # Fallback: split by lines
        return [line.strip() for line in text.splitlines() if line.strip()]

    def explain_plan(self, plan: List[str]) -> str:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful assistant that explains symbolic plans "
                    "in clear, concise natural language.",
                ),
                (
                    "user",
                    (
                        "Here is a sequence of planning actions in a robot domain:\n"
                        "{plan_str}\n\n"
                        "Explain this as a short, coherent description of what "
                        "the agent should do."
                    ),
                ),
            ]
        )

        plan_str = "\n".join(plan)
        try:
            return self._invoke(prompt_template=prompt, plan_str=plan_str)
        except Exception:
            return self._fallback_plan_explanation(plan)

    def _fallback_plan_explanation(self, plan: List[str]) -> str:
        if not plan:
            return "No valid plan was found."

        steps: List[str] = []
        for action in plan:
            normalized = action.strip().strip("()")
            if not normalized:
                continue

            parts = normalized.split()
            name = parts[0]
            args = parts[1:]

            if name == "move" and len(args) >= 3:
                steps.append(f"move {args[0]} from {args[1]} to {args[2]}")
            elif name == "climb_on" and len(args) >= 3:
                steps.append(f"climb {args[0]} onto {args[1]} at {args[2]}")
            elif name == "climb_off" and len(args) >= 3:
                steps.append(f"climb {args[0]} off {args[1]} at {args[2]}")
            elif name == "push_box" and len(args) >= 4:
                steps.append(f"push {args[1]} from {args[2]} to {args[3]}")
            elif name == "grab_banana_from_ground" and len(args) >= 3:
                steps.append(f"grab {args[1]} from the ground at {args[2]}")
            elif name == "grab_banana_from_box" and len(args) >= 4:
                steps.append(f"grab {args[1]} from {args[2]} at {args[3]}")
            elif name == "noop":
                steps.append("no action can be taken from the current state")
            else:
                human = name.replace("_", " ")
                if args:
                    steps.append(f"{human} ({', '.join(args)})")
                else:
                    steps.append(human)

        if not steps:
            return "No valid plan was found."
        return "The monkey should " + ", then ".join(steps) + "."
