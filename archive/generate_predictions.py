import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vision.vision_module import VisionModule

def main():
    model_path = "vision/models/best.pt"
    img_path = "data/monkey_dataset/test/images/canvas_0_banana3_monkey1_box3_png.rf.b415fe8a11da2ac54f21f1907ed2fb59.jpg"

    vm = VisionModule(model_path)
    dets = vm.detect(img_path)

    print("\nDetections:")
    for d in dets:
        print(d)

if __name__ == "__main__":
    main()
