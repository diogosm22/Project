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

# Group detections by rows and columns
rows = []
current_row = []
last_y = None

for center_x, center_y, x1, y1, x2, y2 in boxes_data:
    if last_y is None or abs(center_y - last_y) <= 40:
        # Add to the same row if Y difference is <= 40
        current_row.append((center_x, center_y, x1, y1, x2, y2))
    else:
        # Move to the next row if Y difference is > 40
        rows.append(current_row)
        current_row = [(center_x, center_y, x1, y1, x2, y2)]
    last_y = center_y

# Add the last row if any
if current_row:
    rows.append(current_row)

# Save detection results to CSV file in the requested format
csv_filename = "detections.csv"
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Detection Coordinates"])  # CSV Header
    
    # Draw bounding boxes and add to CSV in Y,X - center_Y ; center_X format
    for row_idx, row in enumerate(rows, start=1):
        for col_idx, (center_x, center_y, x1, y1, x2, y2) in enumerate(row, start=1):
            # Write the format: Y,X - center_Y ; center_X
            writer.writerow([f"{row_idx},{col_idx} - {center_y} ; {center_x}"])
            
            # Draw bounding box
            cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw ID number (also as Y,X for clarity)
            cv2.putText(frame_resized, f"{row_idx},{col_idx}", (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

# Display image with detections
cv2.imshow("YOLOv11 Detection", frame_resized)

# Wait for user to press a key before closing
cv2.waitKey(0)
cv2.destroyAllWindows()

print(f"Detections saved to {csv_filename}")
