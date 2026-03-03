from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
import cv2
import numpy as np


CLASS_NAMES = ["banana", "boxA", "boxB", "boxC", "boxD", "boxE", "monkey"]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_model_path() -> Path:
    return _repo_root() / "integration" / "computer_vision" / "vision" / "models" / "best.onnx"


def _hex_to_bgr(hex_color: str) -> Tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    r, g, b = tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    return (b, g, r)


CLASS_COLOR = {
    "banana": _hex_to_bgr("#FFFF00"),
    "boxA": _hex_to_bgr("#8E583F"),
    "boxB": _hex_to_bgr("#8E583F"),
    "boxC": _hex_to_bgr("#8E583F"),
    "boxD": _hex_to_bgr("#8E583F"),
    "boxE": _hex_to_bgr("#8E583F"),
    "monkey": _hex_to_bgr("#808080"),
}


def predict_image(img_path: str | Path, model_path: str | Path | None = None, conf_threshold: float = 0.4) -> List[Dict[str, float | str]]:
    model = Path(model_path) if model_path else _default_model_path()
    if not model.exists():
        raise FileNotFoundError(f"ONNX model not found: {model}")

    image = cv2.imread(str(img_path))
    if image is None:
        raise FileNotFoundError(f"Image not found: {img_path}")
    h, w = image.shape[:2]

    net = cv2.dnn.readNetFromONNX(str(model))
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (640, 640), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward()[0].transpose(1, 0)

    detections: List[Dict[str, float | str]] = []
    for detection in outputs:
        scores = detection[4:]
        class_id = int(np.argmax(scores))
        confidence = float(scores[class_id])
        if confidence <= conf_threshold:
            continue

        cx, cy, bw, bh = detection[:4]
        cx = float(cx) * w / 640
        cy = float(cy) * h / 640
        bw = float(bw) * w / 640
        bh = float(bh) * h / 640

        x1 = int(cx - bw / 2)
        y1 = int(cy - bh / 2)
        x2 = int(cx + bw / 2)
        y2 = int(cy + bh / 2)

        label = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"class_{class_id}"
        detections.append(
            {
                "x": float(x1),
                "y": float(y1),
                "w": float(max(0, x2 - x1)),
                "h": float(max(0, y2 - y1)),
                "label": label,
                "confidence": confidence,
            }
        )
    return detections


def render_bounding_boxes(img_path: str | Path, detections: List[Dict[str, float | str]], out_path: str | Path) -> Path:
    image = cv2.imread(str(img_path))
    if image is None:
        raise FileNotFoundError(f"Image not found: {img_path}")
    annotated = image.copy()
    for det in detections:
        x1 = int(det["x"])
        y1 = int(det["y"])
        w = int(det["w"])
        h = int(det["h"])
        label = str(det["label"])
        x2 = x1 + w
        y2 = y1 + h
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), annotated)
    return out


def render_grid_representation(
    img_path: str | Path,
    detections: List[Dict[str, float | str]],
    out_path: str | Path,
    grid_rows: int = 32,
    grid_cols: int = 32,
) -> Path:
    image = cv2.imread(str(img_path))
    if image is None:
        raise FileNotFoundError(f"Image not found: {img_path}")
    h_img, w_img = image.shape[:2]
    cell_h = max(1, h_img // grid_rows)
    cell_w = max(1, w_img // grid_cols)

    line_thickness = 2
    grid_img = np.zeros(
        (
            grid_rows * cell_h + (grid_rows + 1) * line_thickness,
            grid_cols * cell_w + (grid_cols + 1) * line_thickness,
            3,
        ),
        dtype=np.uint8,
    )
    grid_img[:] = (0, 0, 0)

    cell_colors: List[List[Tuple[int, int, int] | None]] = [[None for _ in range(grid_cols)] for _ in range(grid_rows)]
    sorted_detections = sorted(detections, key=lambda d: 0 if str(d["label"]).startswith("box") else 1)

    for det in sorted_detections:
        x = int(det["x"])
        y = int(det["y"])
        w = int(det["w"])
        h = int(det["h"])
        label = str(det["label"])
        color = CLASS_COLOR.get(label, (255, 255, 255))

        if label.startswith("box"):
            center_x = (x + w // 2) // cell_w
            center_y = (y + h // 2) // cell_h
            if 0 <= center_x < grid_cols and 0 <= center_y < grid_rows and cell_colors[center_y][center_x] is None:
                cell_colors[center_y][center_x] = color
            continue

        grid_x1 = max(0, min(grid_cols - 1, x // cell_w))
        grid_y1 = max(0, min(grid_rows - 1, y // cell_h))
        grid_x2 = max(0, min(grid_cols - 1, (x + w) // cell_w))
        grid_y2 = max(0, min(grid_rows - 1, (y + h) // cell_h))
        for gy in range(grid_y1, grid_y2 + 1):
            for gx in range(grid_x1, grid_x2 + 1):
                cell_colors[gy][gx] = color

    for y in range(grid_rows):
        for x in range(grid_cols):
            color = cell_colors[y][x] if cell_colors[y][x] else (0, 0, 0)
            y_start = y * cell_h + (y + 1) * line_thickness
            y_end = y_start + cell_h
            x_start = x * cell_w + (x + 1) * line_thickness
            x_end = x_start + cell_w
            grid_img[y_start:y_end, x_start:x_end] = color

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), grid_img)
    return out


def run_grid_detection(
    image_path: str | Path,
    bbox_out_path: str | Path,
    grid_out_path: str | Path,
    model_path: str | Path | None = None,
) -> List[Dict[str, float | str]]:
    detections = predict_image(image_path, model_path=model_path)
    render_bounding_boxes(image_path, detections, bbox_out_path)
    render_grid_representation(image_path, detections, grid_out_path)
    return detections

