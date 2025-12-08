import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout,
    QFileDialog, QTableWidget
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import cv2
import numpy as np

# Path to ONNX model 
MODEL_PATH = "vision/models/best.onnx"
CLASS_NAMES = ["banana", "boxA", "boxB", "boxC", "boxD", "boxE", "monkey"]

def predict_image(img_path):
    # OpenCV DNN
    net = cv2.dnn.readNetFromONNX(MODEL_PATH)
    image = cv2.imread(img_path)
    h, w = image.shape[:2]
    
    # YOLOv8 input size
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (640, 640), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward()
    
    # YOLOv8 output format processing
    outputs = outputs[0].transpose(1, 0) 
    
    results = []
    annotated_img = image.copy()
    conf_threshold = 0.4
    
    for detection in outputs:
        scores = detection[4:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        
        if confidence > conf_threshold:
            # Box coordinates (center_x, center_y, width, height)
            cx, cy, bw, bh = detection[:4]
            
            # Convert to pixel coords
            cx = cx * w / 640
            cy = cy * h / 640
            bw = bw * w / 640
            bh = bh * h / 640
            
            x1 = int(cx - bw / 2)
            y1 = int(cy - bh / 2)
            x2 = int(cx + bw / 2)
            y2 = int(cy + bh / 2)
            
            label = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else str(class_id)
            results.append({
                'x': x1,
                'y': y1,
                'w': x2 - x1,
                'h': y2 - y1,
                'label': label
            })
            
            color = (0, 255, 0)  
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated_img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    return results, annotated_img


# hex to BGR conversion
def hex_to_bgr(hex_color):
    """Convert hex color to BGR tuple"""
    hex_color = hex_color.lstrip('#')
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return (b, g, r)  

CLASS_COLOR = {
    'banana': hex_to_bgr('#FFFF00'),  
    'boxA': hex_to_bgr("#8E583F"),     
    'boxB': hex_to_bgr('#8E583F'),    
    'boxC': hex_to_bgr('#8E583F'),     
    'boxD': hex_to_bgr('#8E583F'),   
    'boxE': hex_to_bgr('#8E583F'),    
    'monkey': hex_to_bgr('#808080')    
}

class ImageGridApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Object Detection & 2D Grid Viewer')
        self.image_label = QLabel('No image loaded')
        self.image_label.setAlignment(Qt.AlignCenter)
        self.grid_table = QTableWidget()
        self.load_btn = QPushButton('Load Image')
        self.load_btn.clicked.connect(self.load_image)
        self.grid_array_btn = QPushButton('Show 2D Array Output')
        self.grid_array_btn.clicked.connect(self.show_grid_array)
        self.save_grid_btn = QPushButton('Save Grid as Image')
        self.save_grid_btn.clicked.connect(self.save_grid_image)
        self.vbox = QVBoxLayout()
        self.vbox.addWidget(self.load_btn)
        self.vbox.addWidget(self.image_label)
        self.vbox.addWidget(QLabel('2D Grid of Detected Objects:'))
        self.vbox.addWidget(self.grid_table)
        self.vbox.addWidget(self.grid_array_btn)
        self.vbox.addWidget(self.save_grid_btn)
        self.setLayout(self.vbox)
        self.image = None
        self.detections = []
        self.grid_rows = 32  # Vertical grid cells 
        self.grid_cols = 32  # Horizontal grid cells 

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, 'Open Image', '', 'Images (*.png *.jpg *.jpeg)')
        if file_path:
            self.image = cv2.imread(file_path)
            # Run detection
            self.detections, self.annotated_img = predict_image(file_path)
            self.display_image_with_boxes()
            self.display_grid()

    def display_image_with_boxes(self):
        # Use annotated image 
        img = self.annotated_img.copy()
        # Overlay grid and color cells
        grid_rows = self.grid_rows
        grid_cols = self.grid_cols
        h_img, w_img = img.shape[:2]
        cell_h = h_img // grid_rows
        cell_w = w_img // grid_cols
        overlay = img.copy()
        alpha = 0.7

        # Build cell color grid with overlap detection
        cell_colors = [[None for _ in range(grid_cols)] for _ in range(grid_rows)]
        
        # Sort detections: boxes first, then banana/monkey
        sorted_detections = sorted(self.detections, key=lambda d: 0 if d['label'].startswith('box') else 1)
        
        for det in sorted_detections:
            x, y, w, h = det['x'], det['y'], det['w'], det['h']
            label = det['label']
            color = CLASS_COLOR.get(label, (255, 255, 255))
            
            if label.startswith('box'):
                # Only mark center cell for boxes
                center_x = (x + w // 2) // cell_w
                center_y = (y + h // 2) // cell_h
                if 0 <= center_x < grid_cols and 0 <= center_y < grid_rows:
                    if cell_colors[center_y][center_x] is None:
                        cell_colors[center_y][center_x] = color
            else:
                # Mark all cells for banana/monkey
                grid_x1 = max(0, min(grid_cols - 1, x // cell_w))
                grid_y1 = max(0, min(grid_rows - 1, y // cell_h))
                grid_x2 = max(0, min(grid_cols - 1, (x + w) // cell_w))
                grid_y2 = max(0, min(grid_rows - 1, (y + h) // cell_h))
                
                for gy in range(grid_y1, grid_y2 + 1):
                    for gx in range(grid_x1, grid_x2 + 1):
                        if 0 <= gx < grid_cols and 0 <= gy < grid_rows:
                            cell_colors[gy][gx] = color
        # Overlay colored cells
        for i in range(grid_rows):
            for j in range(grid_cols):
                color = cell_colors[i][j]
                if color:
                    top_left = (j * cell_w, i * cell_h)
                    bottom_right = ((j + 1) * cell_w, (i + 1) * cell_h)
                    cv2.rectangle(overlay, top_left, bottom_right, color, -1)
        img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
        # Draw grid lines
        for i in range(grid_rows + 1):
            y = i * cell_h
            cv2.line(img, (0, y), (w_img, y), (0, 0, 0), 2)
        for j in range(grid_cols + 1):
            x = j * cell_w
            cv2.line(img, (x, 0), (x, h_img), (0, 0, 0), 2)
        # Show grid overlay with detection results
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_img.shape
        bytes_per_line = ch * w
        qt_img = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img)
        self.image_label.setPixmap(pixmap.scaled(400, 400, Qt.KeepAspectRatio))

    def display_grid(self):
        if self.image is None:
            self.grid_table.clearContents()
            return
        grid_rows = self.grid_rows
        grid_cols = self.grid_cols
        self.grid_table.setRowCount(grid_rows)
        self.grid_table.setColumnCount(grid_cols)
        self.grid_table.setHorizontalHeaderLabels([str(i) for i in range(grid_cols)])
        self.grid_table.setVerticalHeaderLabels([str(i) for i in range(grid_rows)])
        self.grid_table.clearContents()
        img_h, img_w = self.image.shape[:2]
        cell_h = img_h // grid_rows
        cell_w = img_w // grid_cols
        
        # Build cell grid 
        cell_colors = [[None for _ in range(grid_cols)] for _ in range(grid_rows)]
        sorted_detections = sorted(self.detections, key=lambda d: 0 if d['label'].startswith('box') else 1)
        
        for det in sorted_detections:
            x, y, w, h = det['x'], det['y'], det['w'], det['h']
            label = det['label']
            
            if label.startswith('box'):
                # Only mark center cell for boxes
                center_x = (x + w // 2) // cell_w
                center_y = (y + h // 2) // cell_h
                if 0 <= center_x < grid_cols and 0 <= center_y < grid_rows:
                    if cell_colors[center_y][center_x] is None:
                        cell_colors[center_y][center_x] = label
            else:
                # Mark all cells for banana/monkey
                grid_x1 = max(0, min(grid_cols - 1, x // cell_w))
                grid_y1 = max(0, min(grid_rows - 1, y // cell_h))
                grid_x2 = max(0, min(grid_cols - 1, (x + w) // cell_w))
                grid_y2 = max(0, min(grid_rows - 1, (y + h) // cell_h))
                
                for gy in range(grid_y1, grid_y2 + 1):
                    for gx in range(grid_x1, grid_x2 + 1):
                        if 0 <= gx < grid_cols and 0 <= gy < grid_rows:
                            cell_colors[gy][gx] = label
        
        # Fill table with detection labels
        from PyQt5.QtWidgets import QTableWidgetItem
        from PyQt5.QtGui import QColor
        
        for y in range(grid_rows):
            for x in range(grid_cols):
                item = QTableWidgetItem()
                if cell_colors[y][x]:
                    item.setText(cell_colors[y][x])
                    # Set color based on detection
                    bgr_color = CLASS_COLOR.get(cell_colors[y][x], (255, 255, 255))
                    item.setBackground(QColor(bgr_color[2], bgr_color[1], bgr_color[0]))
                self.grid_table.setItem(y, x, item)
        
        # Resize cells to fit
        self.grid_table.resizeColumnsToContents()
        self.grid_table.resizeRowsToContents()
    
    def show_grid_array(self):
        if self.image is None:
            print("No image loaded.")
            return
        grid_rows = self.grid_rows
        grid_cols = self.grid_cols
        img_h, img_w = self.image.shape[:2]
        cell_h = img_h // grid_rows
        cell_w = img_w // grid_cols
        grid = np.zeros((grid_rows, grid_cols, 3), dtype=np.uint8)
        for y in range(grid_rows):
            for x in range(grid_cols):
                cell_img = self.image[y*cell_h:(y+1)*cell_h, x*cell_w:(x+1)*cell_w]
                avg_color = cell_img.mean(axis=(0,1)).astype(np.uint8) if cell_img.size else np.array([0,0,0], dtype=np.uint8)
                grid[y, x] = avg_color
        print("2D Color Grid Output (average RGB per cell):")
        for row in grid:
            print([tuple(cell) for cell in row])
    def save_grid_image(self):
        if self.image is None or not self.detections:
            print("No image or detections loaded.")
            return
        
        grid_rows = self.grid_rows
        grid_cols = self.grid_cols
        img_h, img_w = self.image.shape[:2]
        cell_h = img_h // grid_rows
        cell_w = img_w // grid_cols
        
        line_thickness = 2
        grid_img = np.zeros((grid_rows * cell_h + (grid_rows+1)*line_thickness, 
                            grid_cols * cell_w + (grid_cols+1)*line_thickness, 3), dtype=np.uint8)
        grid_img[:] = (0, 0, 0)  # Black background
        
        # Determine cell colors 
        cell_colors = [[None for _ in range(grid_cols)] for _ in range(grid_rows)]
        sorted_detections = sorted(self.detections, key=lambda d: 0 if d['label'].startswith('box') else 1)
        
        for det in sorted_detections:
            x, y, w, h = det['x'], det['y'], det['w'], det['h']
            label = det['label']
            color = CLASS_COLOR.get(label, (255, 255, 255))
            
            if label.startswith('box'):
                # Only mark center cell for boxes
                center_x = (x + w // 2) // cell_w
                center_y = (y + h // 2) // cell_h
                if 0 <= center_x < grid_cols and 0 <= center_y < grid_rows:
                    if cell_colors[center_y][center_x] is None:
                        cell_colors[center_y][center_x] = color
            else:
                # Mark all cells for banana/monkey
                grid_x1 = max(0, min(grid_cols - 1, x // cell_w))
                grid_y1 = max(0, min(grid_rows - 1, y // cell_h))
                grid_x2 = max(0, min(grid_cols - 1, (x + w) // cell_w))
                grid_y2 = max(0, min(grid_rows - 1, (y + h) // cell_h))
                
                for gy in range(grid_y1, grid_y2 + 1):
                    for gx in range(grid_x1, grid_x2 + 1):
                        if 0 <= gx < grid_cols and 0 <= gy < grid_rows:
                            cell_colors[gy][gx] = color
        
        # Fill grid based on detections
        for y in range(grid_rows):
            for x in range(grid_cols):
                color = cell_colors[y][x] if cell_colors[y][x] else (0, 0, 0)
                y_start = y * cell_h + (y + 1) * line_thickness
                y_end = y_start + cell_h
                x_start = x * cell_w + (x + 1) * line_thickness
                x_end = x_start + cell_w
                grid_img[y_start:y_end, x_start:x_end] = color
        
        file_path, _ = QFileDialog.getSaveFileName(self, 'Save Grid Image', '', 'Images (*.png *.jpg *.jpeg)')
        if file_path:
            cv2.imwrite(file_path, grid_img)
            print(f"Grid image saved to {file_path}")
            print(f"Detected objects: {[det['label'] for det in self.detections]}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ImageGridApp()
    window.resize(600, 800)
    window.show()
    sys.exit(app.exec_())
