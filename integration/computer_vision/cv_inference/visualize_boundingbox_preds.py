from pathlib import Path
from typing import List
import logging

try:
    from models import DetectedObject
except ImportError:  # pragma: no cover - supports package/module run modes
    from integration.models import DetectedObject

from archive.grid_detection import run_grid_detection


logger = logging.getLogger(__name__)


class ImageDetector:
    """
    Adapter around Ultralytics YOLO that exposes the service interface expected
    by PipelineService: detect_objects(image_url) -> List[DetectedObject].
    """

    def __init__(self, model_path: str | None = None, output_path: str | None = None):
        repo_root = Path(__file__).resolve().parents[3]
        self.model_path = Path(model_path) if model_path else repo_root / "integration" / "computer_vision" / "vision" / "models" / "best.onnx"
        self.output_path = Path(output_path) if output_path else repo_root / "integration" / ".runtime" / "vis_yolo.jpg"
        self.repo_root = repo_root
        self.frontend_public_dir = repo_root / "neuro-symbolic_monkeys" / "public"
        self.generated_public_dir = self.frontend_public_dir / "generated"
        self.last_detection_image_url: str | None = None
        self.last_grid_image_url: str | None = None

        if not self.model_path.exists():
            raise FileNotFoundError(f"YOLO model not found: {self.model_path}")

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
        self.generated_public_dir.mkdir(parents=True, exist_ok=True)
        stem = image_path.stem
        bbox_file = self.generated_public_dir / f"{stem}_bbox.jpg"
        grid_file = self.generated_public_dir / f"{stem}_grid.jpg"
        detections = run_grid_detection(
            image_path=image_path,
            bbox_out_path=bbox_file,
            grid_out_path=grid_file,
            model_path=self.model_path,
        )
        self.last_detection_image_url = f"/generated/{bbox_file.name}"
        self.last_grid_image_url = f"/generated/{grid_file.name}"

        objects: List[DetectedObject] = []
        for det in detections:
            x = float(det["x"])
            y = float(det["y"])
            w = float(det["w"])
            h = float(det["h"])
            label = str(det["label"])
            x_center = x + (w / 2)
            y_center = y + (h / 2)
            objects.append(
                DetectedObject(
                    label=label,
                    bbox=(float(x_center), float(y_center), float(w), float(h)),
                )
            )

        logger.info("Detected %s object(s) from %s", len(objects), image_path.name)
        return objects
