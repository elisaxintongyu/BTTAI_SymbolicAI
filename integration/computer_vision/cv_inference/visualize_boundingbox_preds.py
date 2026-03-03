from pathlib import Path
from typing import List
import logging

import cv2
from ultralytics import YOLO

try:
    from models import DetectedObject
except ImportError:  # pragma: no cover - supports package/module run modes
    from integration.models import DetectedObject


logger = logging.getLogger(__name__)


class ImageDetector:
    """
    Adapter around Ultralytics YOLO that exposes the service interface expected
    by PipelineService: detect_objects(image_url) -> List[DetectedObject].
    """

    def __init__(self, model_path: str | None = None, output_path: str | None = None):
        repo_root = Path(__file__).resolve().parents[3]
        self.model_path = Path(model_path) if model_path else repo_root / "integration" / "computer_vision" / "vision" / "models" / "best.pt"
        self.output_path = Path(output_path) if output_path else repo_root / "integration" / ".runtime" / "vis_yolo.jpg"
        self.repo_root = repo_root
        self.frontend_public_dir = repo_root / "neuro-symbolic_monkeys" / "public"

        if not self.model_path.exists():
            raise FileNotFoundError(f"YOLO model not found: {self.model_path}")
        self.model = YOLO(str(self.model_path))

    def _resolve_image_path(self, image_url: str) -> Path:
        candidate = Path(image_url)
        if candidate.is_absolute() and candidate.exists():
            return candidate
        if image_url.startswith("/"):
            public_path = self.frontend_public_dir / image_url.lstrip("/")
            if public_path.exists():
                return public_path
        repo_relative = self.repo_root / image_url
        if repo_relative.exists():
            return repo_relative
        raise FileNotFoundError(f"Image not found for url/path: {image_url}")

    def detect_objects(self, image_url: str) -> List[DetectedObject]:
        image_path = self._resolve_image_path(image_url)
        result = self.model(str(image_path), imgsz=1024)[0]

        annotated = result.plot()
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(self.output_path), annotated)

        names = result.names if isinstance(result.names, dict) else {}
        objects: List[DetectedObject] = []
        for box in result.boxes:
            cls_id = int(box.cls.item())
            x_center, y_center, w, h = box.xywh[0].tolist()
            label = str(names.get(cls_id, f"class_{cls_id}"))
            objects.append(
                DetectedObject(
                    label=label,
                    bbox=(float(x_center), float(y_center), float(w), float(h)),
                )
            )

        logger.info("Detected %s object(s) from %s", len(objects), image_path.name)
        return objects
