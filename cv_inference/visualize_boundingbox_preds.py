from ultralytics import YOLO
from pathlib import Path
import cv2

MODEL_PATH = Path("vision/models/best.pt") 
IMAGE_PATH = Path("data/monkey_dataset/test/images/canvas_0_banana3_monkey1_box3_png.rf.b415fe8a11da2ac54f21f1907ed2fb59.jpg")
OUT_PATH = Path("outputs/vis_yolo.jpg")

def main():
    model = YOLO(str(MODEL_PATH))
    results = model(str(IMAGE_PATH), imgsz=1024)[0]
    annotated = results.plot()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT_PATH), annotated)
    print(f"Saved visualization to: {OUT_PATH.resolve()}")

if __name__ == "__main__":
    main()
