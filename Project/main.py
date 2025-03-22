# main.py

import tkinter as tk
from tkinter import filedialog, simpledialog
import cv2
import numpy as np
import random
from models.yolo_model import load_model, predict  # Correct import
from utils.image_utils import draw_circle_perimeters, draw_measurements_overlay
from utils.pdf_utils import save_results_as_pdf
from utils.excel_utils import save_results_to_csv
from gui.parameter_window import get_parameters
from config import conf_threshold, scale_factor, margin_value

def main():
    model = load_model()

    while True:
        get_parameters()  # Ask for parameters

        image_path = filedialog.askopenfilename(title="Select an image", filetypes=[("Images", "*.jpg;*.png;*.jpeg")])
        if not image_path:
            print("Cancel.")
            break

        frame = cv2.imread(image_path)
        if frame is None:
            print("Error.")
            continue

        frame_resized = cv2.resize(frame, (640, 640))
        original_image = frame_resized.copy()

        results = predict(model, frame_resized, conf_threshold)
        detections = [(int(box.xyxy[0][0]), int(box.xyxy[0][1]), int(box.xyxy[0][2]), int(box.xyxy[0][3])) for box in results[0].boxes]

        # Randomly select a bounding box for overlay visualization
        if detections:
            selected_bbox = random.choice(detections)
            hole_overlay = draw_measurements_overlay(original_image, selected_bbox, scale_factor)

        # Create segmented image with circle perimeters
        segmented_image = np.zeros_like(frame_resized)
        draw_circle_perimeters(segmented_image, detections)

        # Save results as PDF and CSV
        save_results_as_pdf(original_image, segmented_image, hole_overlay, detections, conf_threshold, scale_factor)
        save_results_to_csv(detections, margin_value)

        break

if __name__ == "__main__":
    main()