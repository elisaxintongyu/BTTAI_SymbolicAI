# models.py
from typing import List, Tuple
from pydantic import BaseModel


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
    detection_image_url: str | None = None
    grid_image_url: str | None = None
