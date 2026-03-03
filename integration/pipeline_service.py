# pipeline_service.py
from typing import List

try:
    from .models import AskRequest, PipelineResponse
    from .computer_vision.cv_inference.visualize_boundingbox_preds import ImageDetector
    from .fol_service import FOLService
    from .planner_service import PlannerService
    from .llm_client import LLMClient
except ImportError:  # pragma: no cover - supports running from integration/ directly
    from models import AskRequest, PipelineResponse
    from computer_vision.cv_inference.visualize_boundingbox_preds import ImageDetector
    from fol_service import FOLService
    from planner_service import PlannerService
    from llm_client import LLMClient


class PipelineService:
    """
    High-level orchestrator for the full pipeline:

    1. DetectionService: image -> objects
    2. FOLService: question + objects -> FOL
    3. PlannerService: FOL -> plan
    4. LLMClient: plan -> natural language answer
    """

    def __init__(
        self,
        detection_service: ImageDetector | None = None,
        fol_service: FOLService | None = None,
        planner_service: PlannerService | None = None,
        llm_client: LLMClient | None = None,
    ) -> None:
        self.llm_client = llm_client or LLMClient()
        self.detection_service = detection_service or ImageDetector()
        self.fol_service = fol_service or FOLService(self.llm_client)
        self.planner_service = planner_service or PlannerService()

    def run(self, req: AskRequest) -> PipelineResponse:
        # 1) Detect objects
        objects = self.detection_service.detect_objects(req.image_url)
        detection_image_url = getattr(self.detection_service, "last_detection_image_url", None)
        grid_image_url = getattr(self.detection_service, "last_grid_image_url", None)

        # 2) Build FOL from question + objects
        try:
            fol = self.fol_service.build_fol(question=req.question, objects=objects)
        except Exception:
            fol = self._fallback_fol(objects)

        # 3) Plan over FOL
        plan = self.planner_service.plan(fol)

        # 4) Explain plan
        try:
            answer = self.llm_client.explain_plan(plan)
        except Exception:
            answer = " -> ".join(plan)

        # 5) Bundle all into response model
        return PipelineResponse(
            objects=objects,
            fol=fol,
            plan=plan,
            answer=answer,
            detection_image_url=detection_image_url,
            grid_image_url=grid_image_url,
        )

    def _fallback_fol(self, objects) -> List[str]:
        labels = [obj.label.lower() for obj in objects]
        fol = ["at(monkey, l1)", "on_ground(monkey)"]
        if "banana" in labels:
            fol.extend(["banana_at(banana, l4)", "banana_on_ground(banana)"])
        if "boxa" in labels:
            fol.append("box_at(boxa, l2)")
        if "boxb" in labels:
            fol.append("box_at(boxb, l3)")
        return fol
