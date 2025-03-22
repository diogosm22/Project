import tkinter as tk
from tkinter import filedialog

import cv2
from ultralytics import YOLO

# Load YOLO model
model = YOLO("C:/Users/diogo/Desktop/python/yolov11/yolov11s/modelo2.pt")

# Function to open file dialog and load an image
def load_image():
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    return filedialog.askopenfilename(title="Select an image", filetypes=[("Image files", "*.jpg;*.png;*.jpeg")])

while True:
    # Load image
    image_path = load_image()

    if not image_path:
        print("No image selected. Exiting.")
        break

    # Read the selected image
    frame = cv2.imread(image_path)

    if frame is None:
        print("Error loading image.")
        break

    # Resize image to laptop resolution (1920x1080)
    frame_resized = cv2.resize(frame, (1024, 640))

    # Perform YOLO object detection
    results = model(frame_resized, conf=0.5)

    # Draw bounding boxes and display probability
    for result in results[0].boxes:
        x1, y1, x2, y2 = map(int, result.xyxy[0])  # Get box coordinates
        conf = float(result.conf[0])  # Get confidence score

        # Draw bounding box
        cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Display image with detections
    cv2.imshow("YOLOv11 Detection", frame_resized)

    # Wait for user interaction
    key = cv2.waitKey(0)

    if key == ord('q') or key == ord('Q'):  # If "Q" is pressed, allow the user to load another image
        cv2.destroyAllWindows()
        continue
    elif key == ord('k') or key == ord('K'):  # If "K" is pressed, exit the program
        print("Exiting program.")
        cv2.destroyAllWindows()
        break
