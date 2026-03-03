# main.py
from typing import List, Tuple
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

# ---- LangChain imports ----
from langchain_openai import ChatOpenAI

from gpt4all import GPT4All

# ---- Replace with your real ONNX + planner imports ----
import onnxruntime as ort

session = ort.InferenceSession("yolox_nano.onnx", providers=["CPUExecutionProvider"])
app = FastAPI()


# ========== MODELS ==========

class AskRequest(BaseModel):
    image_url: str
    question: str


class DetectedObject(BaseModel):
    label: str
    bbox: Tuple[float, float, float, float]  # x, y, w, h


class PipelineResponse(BaseModel):
    objects: List[DetectedObject]
    fol: List[str]
    plan: List[str]
    answer: str


# ========== LLM SETUP (LANGCHAIN) ==========

# Make sure OPENAI_API_KEY (or equivalent) is in your env
llm = ChatOpenAI(
    model="gpt-4o-mini",  # or gpt-4o, etc.
    temperature=0.0,
)


def build_fol_from_scene(
    question: str, objects: List[DetectedObject]
) -> List[str]:
    """
    Use LangChain to prompt the LLM to turn object detections + question
    into FOL predicates.
    """

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

    from json import dumps, loads

    formatted = prompt.format_messages(
        question=question,
        objects_json=dumps(
            [obj.model_dump() for obj in objects], indent=2
        ),
    )
    resp = llm.invoke(formatted)
    text = resp.content

    try:
        fol_list = loads(text)
        if isinstance(fol_list, list) and all(
            isinstance(x, str) for x in fol_list
        ):
            return fol_list
    except Exception:
        pass

    # Fallback: naive splitting
    return [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]


def translate_plan_to_natural_language(plan: List[str]) -> str:
    """
    Use LangChain to turn the planner's action sequence into
    a natural-language explanation.
    """

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are a helpful assistant that explains symbolic plans "
                    "in clear, concise natural language."
                ),
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
    msgs = prompt.format_messages(plan_str=plan_str)
    resp = llm.invoke(msgs)
    return resp.content.strip()


# ========== YOLO / CV STUB ==========

def run_yolo_on_image(image_url: str) -> List[DetectedObject]:
    """
    Stub: replace this with your ONNX-based YOLO inference.
    image_url could be a URL or path; adapt as needed.
    """

    # TODO: download or load the image, preprocess, run ONNX
    # session = ort.InferenceSession("model.onnx")
    # ...
    # detections = [...]

    # For now, return a pretend scene: monkey, banana, boxA
    return [
        DetectedObject(label="monkey", bbox=(120, 80, 42, 62)),
        DetectedObject(label="banana", bbox=(300, 50, 40, 30)),
        DetectedObject(label="boxA", bbox=(200, 200, 90, 70)),
    ]


# ========== PLANNER STUB ==========

def run_planner_on_fol(fol: List[str]) -> List[str]:
    """
    Stub: call your actual planner here.
    You might write FOL to a file, run a binary, parse back the plan.
    """

    # Example: naive hand-coded plan for demo
    return [
        "move(monkey, boxA)",
        "climb(monkey, boxA)",
        "grab(monkey, banana)",
    ]


# ========== PIPELINE ENDPOINT ==========

@app.post("/pipeline", response_model=PipelineResponse)
def run_full_pipeline(req: AskRequest) -> PipelineResponse:
    """
    Orchestrates the entire process:

    1. Run YOLO on the image
    2. Use LLM (LangChain) to convert detections + question -> FOL
    3. Run planner on FOL -> symbolic plan
    4. Use LLM to turn plan -> natural language answer
    """

    # 1) CV / YOLO
    objects = run_yolo_on_image(req.image_url)

    # 2) LLM builds FOL from detections + question
    fol = build_fol_from_scene(req.question, objects)

    # 3) Planner consumes FOL → plan
    plan = run_planner_on_fol(fol)

    # 4) LLM explains plan in natural language
    answer = translate_plan_to_natural_language(plan)

    return PipelineResponse(
        objects=objects,
        fol=fol,
        plan=plan,
        answer=answer,
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
