import csv
import tkinter as tk
from tkinter import filedialog

import cv2
from ultralytics import YOLO

# Load YOLO model
model = YOLO("C:/Users/diogo/Desktop/python/yolov11/yolov11s/modelo.pt")

# Function to open file dialog and load an image
def load_image():
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    return filedialog.askopenfilename(title="Select an image", filetypes=[("Image files", "*.jpg;*.png;*.jpeg")])

# Load image
image_path = load_image()

if not image_path:
    print("No image selected.")
    exit()

# Read the selected image
frame = cv2.imread(image_path)

if frame is None:
    print("Error loading image.")
    exit()

# Resize image to laptop resolution (1024x640)
frame_resized = cv2.resize(frame, (1024, 640))

# Perform YOLO object detection
results = model(frame_resized, conf=0.5)

# Collect bounding box data
boxes_data = []
for result in results[0].boxes:
    x1, y1, x2, y2 = map(int, result.xyxy[0])  # Get box coordinates
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    boxes_data.append((center_x, center_y, x1, y1, x2, y2))

# Sort detections from left to right, top to bottom
boxes_data.sort(key=lambda box: (box[1], box[0]))  # Sort by Center_Y first, then Center_X

# Save detection results to CSV file
csv_filename = "detections.csv"
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["ID", "Center_X", "Center_Y"])  # CSV Header
    for idx, (center_x, center_y, x1, y1, x2, y2) in enumerate(boxes_data, start=1):
        writer.writerow([idx, center_x, center_y])
        
        # Draw bounding box
        cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw ID number
        cv2.putText(frame_resized, str(idx), (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

# Display image with detections
cv2.imshow("YOLOv11 Detection", frame_resized)

# Wait for user to press a key before closing
cv2.waitKey(0)
cv2.destroyAllWindows()

print(f"Detections saved to {csv_filename}")
