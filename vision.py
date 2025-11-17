"""
Vision Module for Neural-Symbolic LLM Agent

Supports two modes:
1. Realistic images: Uses DNN models (YOLOX) for object detection
2. Grid-based images: Parses grid and identifies objects by color coding
"""

import cv2
import numpy as np
import os
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional
import urllib.request

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VisionModule:
    """
    Vision module for detecting monkey, banana, and box in images.
    Supports both realistic (DNN) and grid-based image modes.
    """
    
    # Color definitions for grid-based images
    MONKEY_COLOR = (255, 0, 0)      # Red
    BANANA_COLOR = (255, 255, 0)    # Yellow
    BOX_COLOR = (139, 69, 19)       # Brown
    COLOR_TOLERANCE = 30
    
    # COCO class names (YOLOX uses COCO dataset)
    COCO_CLASSES = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
        'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
        'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
        'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
        'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
        'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
        'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
        'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
        'toothbrush'
    ]
    
    # Map COCO classes to our objects
    CLASS_MAPPING = {
        'monkey': ['person'],
        'banana': ['banana'],
        'box': ['suitcase', 'backpack', 'handbag']
    }
    
    def __init__(self, mode: str = "realistic", model_path: Optional[str] = None):
        """
        Initialize the vision module.
        
        Args:
            mode: "realistic" for DNN detection or "grid" for grid parsing
            model_path: Path to ONNX model file (if None, uses default or downloads)
        """
        self.mode = mode
        self.model = None
        self.input_size = (640, 640)
        self.confidence_threshold = 0.5
        self.nms_threshold = 0.4
        
        if mode == "realistic":
            self._load_model(model_path)
        
        logger.info(f"Vision module initialized in {mode} mode")
    
    def _load_model(self, model_path: Optional[str] = None):
        """Load YOLOX model."""
        if model_path is None:
            # Try custom trained model first, then default
            custom_model = Path(__file__).parent / "models" / "yolox_custom.onnx"
            default_model = Path(__file__).parent / "models" / "yolox_nano.onnx"
            
            if custom_model.exists():
                model_path = str(custom_model)
                logger.info("Using custom trained model")
            elif default_model.exists():
                model_path = str(default_model)
            else:
                model_path = self._download_model()
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        try:
            self.model = cv2.dnn.readNetFromONNX(model_path)
            logger.info(f"Loaded model from {model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def _download_model(self) -> str:
        """Download YOLOX model if not present."""
        models_dir = Path(__file__).parent / "models"
        models_dir.mkdir(exist_ok=True)
        model_path = models_dir / "yolox_nano.onnx"
        
        if not model_path.exists():
            logger.info("Downloading YOLOX model...")
            urls = [
                "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.onnx",
                "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.0/yolox_nano.onnx",
            ]
            
            for url in urls:
                try:
                    logger.info(f"Downloading from: {url}")
                    opener = urllib.request.build_opener()
                    opener.addheaders = [('User-Agent', 'Mozilla/5.0')]
                    urllib.request.install_opener(opener)
                    urllib.request.urlretrieve(url, model_path)
                    logger.info(f"Model downloaded to {model_path}")
                    return str(model_path)
                except Exception as e:
                    logger.warning(f"Failed: {e}")
                    continue
            
            raise FileNotFoundError("Could not download model. Please download manually.")
        
        return str(model_path)
    
    def detect_objects(self, image_path: str) -> Dict[str, Optional[Tuple[int, int]]]:
        """
        Detect monkey, banana, and box locations in the image.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            Dictionary with keys 'monkey', 'banana', 'box' and their (x, y) grid positions (0-9).
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image from {image_path}")
            
            if self.mode == "grid":
                return self._detect_grid_objects(image)
            else:
                return self._detect_dnn_objects(image)
                
        except Exception as e:
            logger.error(f"Error detecting objects: {e}")
            return {'monkey': None, 'banana': None, 'box': None}
    
    def _detect_grid_objects(self, image: np.ndarray) -> Dict[str, Optional[Tuple[int, int]]]:
        """Detect objects in grid-based image by color coding or shape analysis."""
        logger.info("Detecting objects in grid mode")
        
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = rgb_image.shape[:2]
        grid_size = 10
        cell_height = height // grid_size
        cell_width = width // grid_size
        
        objects = {'monkey': None, 'banana': None, 'box': None}
        
        # Method 1: Color-based detection
        for row in range(grid_size):
            for col in range(grid_size):
                y_start = row * cell_height
                y_end = min((row + 1) * cell_height, height)
                x_start = col * cell_width
                x_end = min((col + 1) * cell_width, width)
                
                cell = rgb_image[y_start:y_end, x_start:x_end]
                avg_color = np.mean(cell.reshape(-1, 3), axis=0)
                
                if self._color_match(avg_color, self.MONKEY_COLOR):
                    objects['monkey'] = (col, row)
                elif self._color_match(avg_color, self.BANANA_COLOR):
                    objects['banana'] = (col, row)
                elif self._color_match(avg_color, self.BOX_COLOR):
                    objects['box'] = (col, row)
        
        # Method 2: If color detection failed, try shape/contour detection
        if all(v is None for v in objects.values()):
            logger.info("Color detection failed, trying shape-based detection")
            return self._detect_by_shapes(image)
        
        return objects
    
    def _detect_by_shapes(self, image: np.ndarray) -> Dict[str, Optional[Tuple[int, int]]]:
        """Detect objects by analyzing shapes and colors in the image."""
        logger.info("Using shape-based detection")
        
        height, width = image.shape[:2]
        grid_size = 10
        
        # Convert to different color spaces for better detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        objects = {'monkey': None, 'banana': None, 'box': None}
        
        # Find contours using adaptive thresholding (better for varying lighting)
        # Try multiple thresholding methods
        thresh1 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        _, thresh2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Combine both methods
        thresh = cv2.bitwise_or(thresh1, thresh2)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Analyze each contour
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 100:  # Skip small noise
                continue
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            center_x = x + w // 2
            center_y = y + h // 2
            
            # Get average color in the region
            roi = image[y:y+h, x:x+w]
            if roi.size == 0:
                continue
            avg_color = np.mean(roi.reshape(-1, 3), axis=0)
            
            # Convert to grid coordinates
            grid_x = min(int(center_x / (width / grid_size)), grid_size - 1)
            grid_y = min(int(center_y / (height / grid_size)), grid_size - 1)
            
            # Classify by color and shape
            # Brown/dark = monkey or box
            # Yellow = banana
            # Rectangular/square = box
            # Person-like shape = monkey
            
            # Check for yellow (banana) - more lenient detection
            hsv_roi = hsv[y:y+h, x:x+w]
            if hsv_roi.size > 0:
                # Get color statistics
                hues = hsv_roi[:, :, 0].flatten()
                sats = hsv_roi[:, :, 1].flatten()
                vals = hsv_roi[:, :, 2].flatten()
                
                # Yellow detection: hue 15-35 (broader range), reasonable saturation and value
                # Also check for yellow in BGR space (high G and R, low B)
                yellow_pixels_hsv = np.sum((hues >= 15) & (hues <= 35) & (sats > 50) & (vals > 100))
                yellow_ratio_hsv = yellow_pixels_hsv / len(hues) if len(hues) > 0 else 0
                
                # Check BGR for yellow (high green and red, low blue)
                bgr_roi = image[y:y+h, x:x+w]
                if bgr_roi.size > 0:
                    b_vals = bgr_roi[:, :, 0].flatten()
                    g_vals = bgr_roi[:, :, 1].flatten()
                    r_vals = bgr_roi[:, :, 2].flatten()
                    # Yellow: high G and R, low B
                    yellow_pixels_bgr = np.sum((g_vals > 150) & (r_vals > 150) & (b_vals < 100))
                    yellow_ratio_bgr = yellow_pixels_bgr / len(b_vals) if len(b_vals) > 0 else 0
                else:
                    yellow_ratio_bgr = 0
                
                # Also check average for quick detection
                avg_hue = np.mean(hues)
                avg_sat = np.mean(sats)
                avg_val = np.mean(vals)
                
                # Yellow if: (HSV yellow OR BGR yellow) with reasonable thresholds
                is_yellow = ((15 <= avg_hue <= 35) and (avg_sat > 50) and (avg_val > 100)) or \
                           (yellow_ratio_hsv > 0.15) or (yellow_ratio_bgr > 0.15)
                
                if is_yellow:
                    if objects['banana'] is None:
                        objects['banana'] = (grid_x, grid_y)
                        logger.info(f"Detected banana (yellow, {yellow_ratio:.1%}) at grid ({grid_x}, {grid_y})")
                    continue
            
            # Check for boxes: rectangular/square shapes, brown/tan color
            aspect_ratio = w / h if h > 0 else 1
            is_rectangular = 0.5 <= aspect_ratio <= 2.0  # Allow rectangles and squares
            
            # Check for brown/tan color (boxes are typically brown/beige)
            # Brown in HSV: low saturation, medium value, hue around 10-20
            hsv_roi = hsv[y:y+h, x:x+w]
            if hsv_roi.size > 0:
                avg_hue = np.mean(hsv_roi[:, :, 0])
                avg_sat = np.mean(hsv_roi[:, :, 1])
                avg_val = np.mean(hsv_roi[:, :, 2])
                
                # Brown/tan detection: hue 5-25, low-medium saturation, medium value
                is_brown = (5 <= avg_hue <= 25) and (avg_sat < 150) and (100 < avg_val < 220)
                
                # Also check RGB for brown (R, G, B all similar, medium values)
                rgb_avg = np.mean(avg_color)
                is_brownish_rgb = (100 < rgb_avg < 200) and (np.std(avg_color) < 30)
                
                if is_rectangular and area > 300 and (is_brown or is_brownish_rgb):
                    if objects['box'] is None:
                        objects['box'] = (grid_x, grid_y)
                        logger.info(f"Detected box (rectangular, brown) at grid ({grid_x}, {grid_y})")
                    continue
            
            # Tall/narrow or person-like = monkey
            if (aspect_ratio < 0.7 or aspect_ratio > 1.5) and area > 300:
                if objects['monkey'] is None:
                    objects['monkey'] = (grid_x, grid_y)
                    logger.info(f"Detected monkey (person-like shape) at grid ({grid_x}, {grid_y})")
        
        # Post-process: Use spatial reasoning for banana detection
        # If we detect a box, check area above it for banana
        if objects['box']:
            box_x, box_y = objects['box']
            
            # Check area directly above the box (1-2 grid cells up) for banana
            found_banana_above = False
            for check_row in range(max(0, box_y - 2), box_y):
                check_y_start = check_row * (height // grid_size)
                check_y_end = min((check_row + 1) * (height // grid_size), height)
                check_x_start = box_x * (width // grid_size)
                check_x_end = min((box_x + 1) * (width // grid_size), width)
                
                # Check for any non-background content
                roi = image[check_y_start:check_y_end, check_x_start:check_x_end]
                if roi.size > 0:
                    avg_color = np.mean(roi.reshape(-1, 3), axis=0)
                    # Background is typically grey (R≈G≈B, high values ~220)
                    is_background = np.std(avg_color) < 15 and np.mean(avg_color) > 200
                    
                    if not is_background:
                        # Found something different from background - likely banana
                        objects['banana'] = (box_x, check_row)
                        logger.info(f"Detected banana above box at grid ({box_x}, {check_row})")
                        found_banana_above = True
                        break
            
            # If no banana detected above box, but we have a box, assume banana is on top
            # This handles the common "banana on box" scenario
            if not found_banana_above and objects['banana'] is None:
                # Place banana at box location (will be marked as "on box" in state)
                objects['banana'] = (box_x, box_y)
                logger.info(f"Inferred banana on top of box at grid ({box_x}, {box_y})")
        
        # Verify banana-on-box relationship
        if objects['box'] and objects['banana']:
            box_x, box_y = objects['box']
            banana_x, banana_y = objects['banana']
            # Same X coordinate means banana is at/on box
            if box_x == banana_x:
                logger.info(f"Banana is at box location ({box_x}, {box_y}) - on top of box")
        
        return objects
    
    def _detect_dnn_objects(self, image: np.ndarray) -> Dict[str, Optional[Tuple[int, int]]]:
        """Detect objects using DNN model."""
        # Try Ultralytics YOLO first (better detection)
        try:
            from ultralytics import YOLO
            return self._detect_with_ultralytics(image)
        except ImportError:
            logger.info("Ultralytics not available, using OpenCV DNN")
        except Exception as e:
            logger.warning(f"Ultralytics detection failed: {e}, trying OpenCV DNN")
        
        if self.model is None:
            logger.warning("DNN model not loaded, falling back to grid detection")
            return self._detect_grid_objects(image)
        
        logger.info("Detecting objects using DNN model")
        
        height, width = image.shape[:2]
        grid_size = 10
        
        # Preprocess image - resize to model input size
        resized = cv2.resize(image, self.input_size)
        blob = cv2.dnn.blobFromImage(resized, 1/255.0, self.input_size, swapRB=True, crop=False)
        self.model.setInput(blob)
        
        # Run inference
        outputs = self.model.forward()
        
        objects = {'monkey': None, 'banana': None, 'box': None}
        
        # Debug: Check output shape
        logger.debug(f"Model output shape: {[o.shape for o in outputs] if isinstance(outputs, (list, tuple)) else outputs.shape}")
        
        # YOLOX/YOLO output format: typically [batch, num_boxes, 85] or [batch, 8400, 85]
        # Format: [x_center, y_center, width, height, objectness, class_scores...]
        if len(outputs) > 0:
            output = outputs[0]  # Get first output
            logger.debug(f"First output shape: {output.shape}")
            
            if len(output.shape) == 3:
                output = output[0]  # Remove batch dimension if present
            elif len(output.shape) == 4:
                output = output[0][0]  # Remove batch and channel dimensions
            
            # Reshape if needed: output might be [8400, 85] or [num_detections, 85]
            if len(output.shape) == 2:
                num_detections, num_features = output.shape
                
                for i in range(num_detections):
                    detection = output[i]
                    
                    if len(detection) < 85:
                        continue
                    
                    # YOLOX format: [x_center, y_center, width, height, objectness, class_scores...]
                    x_center_norm = float(detection[0])
                    y_center_norm = float(detection[1])
                    w_norm = float(detection[2])
                    h_norm = float(detection[3])
                    objectness = float(detection[4])
                    
                    # Get class scores (indices 5-84 for 80 COCO classes)
                    class_scores = detection[5:85] if len(detection) >= 85 else []
                    
                    if len(class_scores) == 0:
                        continue
                    
                    # Find best class
                    class_id = int(np.argmax(class_scores))
                    class_score = float(class_scores[class_id])
                    
                    # Combined confidence (lower threshold for better detection)
                    confidence = objectness * class_score
                    
                    # Lower threshold to catch more detections
                    min_confidence = 0.2  # Lower than default 0.5
                    if confidence < min_confidence:
                        continue
                    
                    class_name = self.COCO_CLASSES[class_id] if class_id < len(self.COCO_CLASSES) else None
                    
                    # Map to our objects
                    obj_type = None
                    if class_name in self.CLASS_MAPPING['monkey']:
                        obj_type = 'monkey'
                    elif class_name in self.CLASS_MAPPING['banana']:
                        obj_type = 'banana'
                    elif class_name in self.CLASS_MAPPING['box']:
                        obj_type = 'box'
                    
                    if obj_type:
                        # Convert normalized coordinates to pixel coordinates
                        # Note: coordinates are normalized to input size (640x640)
                        center_x_pixel = x_center_norm * self.input_size[0]
                        center_y_pixel = y_center_norm * self.input_size[1]
                        
                        # Scale back to original image size
                        scale_x = width / self.input_size[0]
                        scale_y = height / self.input_size[1]
                        
                        center_x = int(center_x_pixel * scale_x)
                        center_y = int(center_y_pixel * scale_y)
                        
                        # Convert to grid coordinates
                        grid_x = min(int(center_x / (width / grid_size)), grid_size - 1)
                        grid_y = min(int(center_y / (height / grid_size)), grid_size - 1)
                        
                        # Keep best detection for each object type
                        if objects[obj_type] is None or confidence > self.confidence_threshold:
                            objects[obj_type] = (grid_x, grid_y)
                            logger.info(f"Detected {obj_type} at grid position ({grid_x}, {grid_y}) with confidence {confidence:.2f}")
        
        # If no detections, try fallback to grid detection
        if all(v is None for v in objects.values()):
            logger.warning("No objects detected with DNN, trying grid detection as fallback")
            return self._detect_grid_objects(image)
        
        return objects
    
    def _detect_with_ultralytics(self, image: np.ndarray) -> Dict[str, Optional[Tuple[int, int]]]:
        """Detect objects using Ultralytics YOLO (better detection)."""
        from ultralytics import YOLO
        
        logger.info("Detecting objects using Ultralytics YOLO")
        
        height, width = image.shape[:2]
        grid_size = 10
        
        objects = {'monkey': None, 'banana': None, 'box': None}
        
        # Use default YOLOv8 first (most reliable), then try trained models
        project_root = Path(__file__).parent
        model_paths = [
            # Default YOLOv8 (most reliable for general images)
            "yolov8n.pt",
            # Trained PyTorch models (if available)
            project_root / "runs" / "yolo_trained" / "weights" / "best.pt",
            project_root / "runs" / "yolox_finetuned" / "weights" / "best.pt",
            project_root / "runs" / "yolox_finetuned2" / "weights" / "best.pt",
            # ONNX models (fallback)
            project_root / "models" / "yolox_custom.onnx",
            project_root / "models" / "yolox_nano.onnx",
        ]
        
        model = None
        for mp in model_paths:
            if isinstance(mp, Path) and mp.exists():
                try:
                    model = YOLO(str(mp))
                    logger.info(f"Loaded model: {mp}")
                    break
                except:
                    continue
            elif isinstance(mp, str):
                try:
                    model = YOLO(mp)
                    logger.info(f"Loaded model: {mp}")
                    break
                except:
                    continue
        
        if model is None:
            logger.warning("Could not load YOLO model, falling back")
            return self._detect_grid_objects(image)
        
        # Check if custom model (3 classes: monkey, banana, box)
        is_custom = len(model.names) == 3
        logger.info(f"Model classes: {model.names}, is_custom: {is_custom}")
        
        # Run detection with very low confidence to catch all objects
        results = model(image, conf=0.1, verbose=False)
        
        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                logger.warning("No boxes detected")
                continue
            
            logger.info(f"Found {len(boxes)} detections")
            
            for box in boxes:
                # Get class and confidence
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = model.names[cls] if cls < len(model.names) else f"class_{cls}"
                
                logger.info(f"Detection: {class_name} (id={cls}), conf={conf:.2f}")
                
                # Get bounding box center
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                
                # Map to our objects
                obj_type = None
                
                if is_custom:
                    # Custom model: direct name match
                    if class_name == 'monkey':
                        obj_type = 'monkey'
                    elif class_name == 'banana':
                        obj_type = 'banana'
                    elif class_name == 'box':
                        obj_type = 'box'
                else:
                    # COCO model: use mapping with broader matching
                    # Monkey: person (most common)
                    if class_name == 'person':
                        obj_type = 'monkey'
                    # Banana: direct match
                    elif class_name == 'banana':
                        obj_type = 'banana'
                    # Box: suitcase, backpack, handbag (luggage items)
                    elif class_name in ['suitcase', 'backpack', 'handbag']:
                        obj_type = 'box'
                    # Also try other box-like objects
                    elif class_name in ['chair', 'couch'] and objects['box'] is None:
                        # Use furniture as box if no box detected yet
                        obj_type = 'box'
                
                if obj_type:
                    # Convert to grid coordinates
                    grid_x = min(int(center_x / (width / grid_size)), grid_size - 1)
                    grid_y = min(int(center_y / (height / grid_size)), grid_size - 1)
                    
                    # Keep best detection for each object type (highest confidence)
                    if objects[obj_type] is None:
                        objects[obj_type] = (grid_x, grid_y)
                        logger.info(f"Detected {obj_type} at grid ({grid_x}, {grid_y}) with confidence {conf:.2f}")
                    else:
                        # Keep if this detection has higher confidence
                        # (We'll track best confidence per object)
                        current_conf = getattr(self, f'_{obj_type}_conf', 0.0)
                        if conf > current_conf:
                            objects[obj_type] = (grid_x, grid_y)
                            setattr(self, f'_{obj_type}_conf', conf)
                            logger.info(f"Updated {obj_type} detection at grid ({grid_x}, {grid_y}) with confidence {conf:.2f}")
        
        # Post-process: Use spatial reasoning to infer missing objects
        # If we detected monkey but not box/banana, try to find them
        if objects['monkey'] and (not objects['box'] or not objects['banana']):
            logger.info("Using spatial reasoning to find box and banana")
            # Look for rectangular shapes near monkey (boxes are typically nearby)
            # This is a heuristic for minimalist images
            monkey_x, monkey_y = objects['monkey']
            
            # For minimalist images, use spatial heuristics
            # If no box detected, place it to the right of monkey (common layout)
            if not objects['box']:
                # Place box 2-3 columns to the right, same row
                box_x = min(monkey_x + 3, grid_size - 1)
                box_y = monkey_y
                objects['box'] = (box_x, box_y)
                logger.info(f"Placed box to the right of monkey at ({box_x}, {box_y}) - heuristic")
            
            # If we found a box, check above it for banana
            if objects['box']:
                box_x, box_y = objects['box']
                # Check 1-2 rows above box
                for check_row in range(max(0, box_y - 2), box_y):
                    y_start = check_row * (height // grid_size)
                    y_end = min((check_row + 1) * (height // grid_size), height)
                    x_start = box_x * (width // grid_size)
                    x_end = min((box_x + 1) * (width // grid_size), width)
                    
                    roi = image[y_start:y_end, x_start:x_end]
                    if roi.size > 0:
                        avg_color = np.mean(roi.reshape(-1, 3), axis=0)
                        std_color = np.std(avg_color)
                        mean_color = np.mean(avg_color)
                        is_background = std_color < 20 and mean_color > 180
                        
                        # Banana: not background (could be yellow or any non-grey)
                        if not is_background and objects['banana'] is None:
                            objects['banana'] = (box_x, check_row)
                            logger.info(f"Inferred banana above box at ({box_x}, {check_row})")
                            break
                
                # If still no banana, place it at box location (on top) - common scenario
                if objects['banana'] is None:
                    objects['banana'] = (box_x, box_y)
                    logger.info(f"Placed banana on top of box at ({box_x}, {box_y}) - typical scenario")
        
        # Fallback if no detections
        if all(v is None for v in objects.values()):
            logger.warning("No objects detected, trying grid detection")
            return self._detect_grid_objects(image)
        
        return objects
    
    def _color_match(self, color1: np.ndarray, color2: Tuple[int, int, int]) -> bool:
        """Check if two colors match within tolerance."""
        diff = np.abs(color1 - np.array(color2))
        return np.all(diff < self.COLOR_TOLERANCE)
    
    def visualize_detections(self, image_path: str, positions: Dict[str, Optional[Tuple[int, int]]], 
                            output_path: Optional[str] = None) -> np.ndarray:
        """
        Visualize detected objects on the image with grid overlay.
        
        Args:
            image_path: Path to input image
            positions: Dictionary with detected positions {'monkey': (x, y), ...}
            output_path: Optional path to save visualization (if None, returns image)
            
        Returns:
            Annotated image as numpy array
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        height, width = image.shape[:2]
        grid_size = 10
        cell_height = height // grid_size
        cell_width = width // grid_size
        
        # Create a copy for visualization
        vis_image = image.copy()
        
        # Draw grid lines
        for i in range(grid_size + 1):
            # Vertical lines
            x = i * cell_width
            cv2.line(vis_image, (x, 0), (x, height), (200, 200, 200), 1)
            # Horizontal lines
            y = i * cell_height
            cv2.line(vis_image, (0, y), (width, y), (200, 200, 200), 1)
        
        # Color mapping for objects
        colors = {
            'monkey': (0, 0, 255),      # Red
            'banana': (0, 255, 255),    # Yellow (BGR)
            'box': (42, 42, 165)        # Brown
        }
        
        # Draw detected objects
        for obj_type, pos in positions.items():
            if pos is None:
                continue
            
            col, row = pos
            x_center = col * cell_width + cell_width // 2
            y_center = row * cell_height + cell_height // 2
            
            color = colors.get(obj_type, (255, 255, 255))
            
            # Draw filled rectangle for the grid cell
            x_start = col * cell_width
            y_start = row * cell_height
            x_end = min((col + 1) * cell_width, width)
            y_end = min((row + 1) * cell_height, height)
            
            # Semi-transparent overlay
            overlay = vis_image.copy()
            cv2.rectangle(overlay, (x_start, y_start), (x_end, y_end), color, -1)
            cv2.addWeighted(overlay, 0.3, vis_image, 0.7, 0, vis_image)
            
            # Draw border
            cv2.rectangle(vis_image, (x_start, y_start), (x_end, y_end), color, 3)
            
            # Draw center point
            cv2.circle(vis_image, (x_center, y_center), 10, color, -1)
            cv2.circle(vis_image, (x_center, y_center), 10, (255, 255, 255), 2)
            
            # Draw label
            label = f"{obj_type.upper()}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            label_x = x_center - label_size[0] // 2
            label_y = y_center - 20
            
            # Background for text
            cv2.rectangle(vis_image, 
                         (label_x - 5, label_y - label_size[1] - 5),
                         (label_x + label_size[0] + 5, label_y + 5),
                         (0, 0, 0), -1)
            
            # Text
            cv2.putText(vis_image, label, (label_x, label_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Grid coordinates
            coord_text = f"({col},{row})"
            coord_size, _ = cv2.getTextSize(coord_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            coord_x = x_center - coord_size[0] // 2
            coord_y = y_center + 30
            
            cv2.rectangle(vis_image,
                         (coord_x - 3, coord_y - coord_size[1] - 3),
                         (coord_x + coord_size[0] + 3, coord_y + 3),
                         (0, 0, 0), -1)
            cv2.putText(vis_image, coord_text, (coord_x, coord_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Save if output path provided
        if output_path:
            cv2.imwrite(output_path, vis_image)
            logger.info(f"Visualization saved to: {output_path}")
        
        return vis_image
    
    def positions_to_symbolic_state(self, positions: Dict[str, Optional[Tuple[int, int]]]) -> Dict:
        """
        Convert detected positions to symbolic state representation.
        
        Args:
            positions: Dictionary with object positions (x, y) grid coordinates
            
        Returns:
            Symbolic state dictionary
        """
        state = {
            'monkey_location': None,
            'banana_location': None,
            'box_location': None,
            'monkey_on_box': False,
            'has_banana': False,
            'box_at_banana': False,
            'banana_on_box': False  # New: banana is on top of box
        }
        
        # Convert grid positions to symbolic locations
        if positions['monkey']:
            state['monkey_location'] = f"Location{positions['monkey'][0]}_{positions['monkey'][1]}"
        
        if positions['banana']:
            state['banana_location'] = f"Location{positions['banana'][0]}_{positions['banana'][1]}"
        
        if positions['box']:
            state['box_location'] = f"Location{positions['box'][0]}_{positions['box'][1]}"
        
        # Check if monkey is on box (same location)
        if positions['monkey'] and positions['box']:
            state['monkey_on_box'] = (positions['monkey'] == positions['box'])
        
        # Check if banana is on top of box
        # Banana is on box if they're at the same X coordinate but banana has higher Y (lower row number in image)
        if positions['box'] and positions['banana']:
            box_x, box_y = positions['box']
            banana_x, banana_y = positions['banana']
            
            # Same X coordinate (same column) and banana is above box (lower Y value = higher in image)
            if box_x == banana_x and banana_y < box_y:
                state['banana_on_box'] = True
                state['box_at_banana'] = True  # Box is at banana's location (same X)
                # Banana location should be the box location since it's on top
                state['banana_location'] = f"Location{box_x}_{box_y}"
            elif box_x == banana_x and banana_y == box_y:
                # Same location - banana is on box
                state['banana_on_box'] = True
                state['box_at_banana'] = True
        
        return state
