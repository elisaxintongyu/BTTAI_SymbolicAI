##Vision module to be used to be called by generate_predictions.py when we want to find bounding box coordinates 
from ultralytics import YOLO
import cv2
from dataclasses import dataclass
from pathlib import Path
from typing import List

CLASS_NAMES = ["banana", "boxA", "boxB", "boxC", "boxD", "boxE", "monkey"]

@dataclass
class Detection:
    cls: str
    conf: float
    x1: float
    y1: float
    x2: float
    y2: float

class VisionModule:
    def __init__(self, model_path: str | Path):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        self.model = YOLO(str(model_path))

    def detect(self, image_path: str | Path) -> List[Detection]:
        results = self.model(image_path, imgsz=1024)[0]
        detections = []

        for box, cls_id, conf in zip(
            results.boxes.xyxy.cpu().numpy(),
            results.boxes.cls.cpu().numpy(),
            results.boxes.conf.cpu().numpy()
        ):
            x1, y1, x2, y2 = box
            detections.append(
                Detection(
                    cls=CLASS_NAMES[int(cls_id)],
                    conf=float(conf),
                    x1=float(x1),
                    y1=float(y1),
                    x2=float(x2),
                    y2=float(y2),
                )
            )
        
        return detections
