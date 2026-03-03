# fol_service.py
from typing import List

try:
    from .models import DetectedObject
    from .llm_client import LLMClient
except ImportError:  # pragma: no cover - supports running from integration/ directly
    from models import DetectedObject
    from llm_client import LLMClient


class FOLService:
    """
    Uses the LLMClient to convert detections + question
    into a list of FOL predicates.
    """

    def __init__(self, llm_client: LLMClient) -> None:
        self._llm = llm_client

    def build_fol(self, question: str, objects: List[DetectedObject]) -> List[str]:
        return self._llm.build_fol_from_scene(question=question, objects=objects)
