import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
from ultralytics import YOLO
import random

# Function to draw circle perimeters around detections
def draw_circle_perimeters(frame, detections):
    for x1, y1, x2, y2 in detections:
        center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
        radius = int(max((x2 - x1), (y2 - y1)) / 2)
        cv2.circle(frame, center, radius, (255, 255, 255), 1, cv2.LINE_AA)

# Function to create the zoomed overlay with measurements
def draw_measurements_overlay(frame, bbox, scale_factor):
    """
    Creates an overlay image where the visual part has a fixed radius of 200,
    while the distance values (in cm) are computed from the original segmentation data.
    """
    x1, y1, x2, y2 = bbox
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
    radius = int(max((x2 - x1), (y2 - y1)) / 2)  # Radius of the bounding box in pixels

    # Compute real-world distances in centimeters
    pixel_to_cm = 0.3125 / 10 * scale_factor
    dist_top_cm = radius * pixel_to_cm
    dist_right_cm = radius * pixel_to_cm

    # Fixed radius for visualization
    fixed_radius = 200

    # Create a blank overlay canvas (640x640)
    overlay = np.zeros((640, 640, 3), dtype=np.uint8)
    overlay.fill(50)

    # Center of the canvas
    canvas_center_x, canvas_center_y = 320, 320

    # Draw a fixed-size circle
    cv2.circle(overlay, (canvas_center_x, canvas_center_y), fixed_radius, (255, 255, 255), 2, cv2.LINE_AA)

    # Draw top and right arrows
    cv2.arrowedLine(
        overlay,
        (canvas_center_x, canvas_center_y),
        (canvas_center_x, canvas_center_y - fixed_radius),
        (0, 0, 255), 2, tipLength=0.1
    )
    cv2.arrowedLine(
        overlay,
        (canvas_center_x, canvas_center_y),
        (canvas_center_x + fixed_radius, canvas_center_y),
        (0, 255, 0), 2, tipLength=0.1
    )

    # Display the distance measurements in cm
    font_scale = 0.6
    font_thickness = 2
    cv2.putText(
        overlay,
        f"{dist_top_cm:.2f} cm",
        (canvas_center_x - 80, canvas_center_y - fixed_radius - 20),
        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), font_thickness
    )
    cv2.putText(
        overlay,
        f"{dist_right_cm:.2f} cm",
        (canvas_center_x + fixed_radius + 10, canvas_center_y),
        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), font_thickness
    )

    return overlay

# Load the YOLO model
model = YOLO("C:/Users/diogo/Desktop/python/yolov11/yolov11s/modelo3.pt")

# Function to display the configuration window
def get_parameters():
    global conf_threshold, scale_factor

    def apply_settings():
        global conf_threshold, scale_factor
        conf_threshold = conf_scale.get() / 100
        scale_factor = scale_conv.get()
        param_window.destroy()

    def exit_program():
        print("Leaving.")
        param_window.destroy()
        exit()

    # Tkinter window for parameter settings
    param_window = tk.Tk()
    param_window.title("Configurations")

    # Confidence level slider
    tk.Label(param_window, text="Trust Factor").pack()
    conf_scale = tk.Scale(param_window, from_=1, to=100, orient=tk.HORIZONTAL)
    conf_scale.set(50)
    conf_scale.pack()

    # Scale factor slider
    tk.Label(param_window, text="Scale").pack()
    scale_conv = tk.Scale(param_window, from_=1, to=100, orient=tk.HORIZONTAL)
    scale_conv.set(50)
    scale_conv.pack()

## se for necessario casas decimais ###
    #scale_conv = tk.Scale(param_window, from_=1, to=100, resolution=0, orient=tk.HORIZONTAL)

    # Apply button
    apply_button = tk.Button(param_window, text="Apply", command=apply_settings)
    apply_button.pack()

    # Exit button
    exit_button = tk.Button(param_window, text="Exit", command=exit_program)
    exit_button.pack()

    param_window.mainloop()

# Function to load an image using a file dialog
def load_image():
    file_path = filedialog.askopenfilename(title="Selecione an image", filetypes=[("Images", "*.jpg;*.png;*.jpeg")])
    return file_path

# Main program logic
conf_threshold = 50
scale_factor = 50

while True:  # Outer loop to reopen settings
    get_parameters()  # Open the settings window

    while True:  # Inner loop for image selection and processing
        image_path = load_image()
        if not image_path:
            print("Cancel.")
            break  # Exit the inner loop to reopen settings

        frame = cv2.imread(image_path)
        if frame is None:
            print("Error.")
            continue

        # Resize the image for processing
        frame_resized = cv2.resize(frame, (640, 640))
        original_image = frame_resized.copy()

        # Perform object detection with YOLO
        results = model.predict(frame_resized, conf=conf_threshold, verbose=False, max_det=1000)
        detections = [(int(box.xyxy[0][0]), int(box.xyxy[0][1]), int(box.xyxy[0][2]), int(box.xyxy[0][3])) for box in results[0].boxes]

        # Create a segmented image
        segmented_image = np.zeros_like(frame_resized)
        draw_circle_perimeters(segmented_image, detections)

        # Combine the original and segmented images side-by-side
        combined_image = np.hstack((original_image, segmented_image))

        # Display the combined image
        cv2.namedWindow(" Original | Segmented ", cv2.WINDOW_NORMAL)
        cv2.resizeWindow(" Original | Segmented ", 1280, 640)
        cv2.imshow(" Original | Segmented ", combined_image)

        if detections:
            # Randomly select a bounding box for overlay visualization
            selected_bbox = random.choice(detections)
            overlay = draw_measurements_overlay(original_image, selected_bbox, scale_factor)
            cv2.imshow("Hole Dimensions", overlay)

        # Wait for user input and handle events
        print("'Q' to go back and 'K' to leave")
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            cv2.destroyAllWindows()
            break  # Break inner loop to go back to settings
        elif key == ord('k'):
            print("Leaving")
            cv2.destroyAllWindows()
            exit()

        cv2.destroyAllWindows()
