##This calls the pretrained model and trains it on our dataset (took about 2 hours to run)
##Images need to be 1024x1024 when given from the UI for best results i think since we trained with 1024x1024
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="monkey.yaml",
    imgsz=1024,
    epochs=100,
    batch=8,
    project="runs",
    name="monkey_detector_v1"
)