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

# Draw bounding boxes and display probability
boxes_data = []
for result in results[0].boxes:
    x1, y1, x2, y2 = map(int, result.xyxy[0])  # Get box coordinates
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    conf = float(result.conf[0])  # Get confidence score
    label = f"Hole {conf:.2f}"  # Add probability to label

    # Store bounding box data
    boxes_data.append(["Hole", conf, center_x, center_y, x1, y1, x2, y2])

    # Draw bounding box
    cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Put the label (probability) on the box
    cv2.putText(frame_resized, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

# Save detection results to CSV file
csv_filename = "detections.csv"
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Label", "Confidence", "Center_X", "Center_Y", "X1", "Y1", "X2", "Y2"])  # CSV Header
    writer.writerows(boxes_data)

# Display image with detections
cv2.imshow("YOLOv11 Detection", frame_resized)

# Wait for user to press a key before closing
cv2.waitKey(0)
cv2.destroyAllWindows()

print(f"Detections saved to {csv_filename}")
