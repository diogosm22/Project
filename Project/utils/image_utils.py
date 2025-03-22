# utils/image_utils.py

import cv2
import numpy as np

def draw_circle_perimeters(frame, detections):
    for x1, y1, x2, y2 in detections:
        center = (int((x1 + x2) / 2), int((y1 + y2) / 2))  # Fixed extra parenthesis
        radius = int(max((x2 - x1), (y2 - y1)) / 2)
        cv2.circle(frame, center, radius, (255, 255, 255), 1, cv2.LINE_AA)

def draw_measurements_overlay(frame, bbox, scale_factor):
    x1, y1, x2, y2 = bbox
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
    radius = int(max((x2 - x1), (y2 - y1)) / 2)  # Radius of the bounding box in pixels

    pixel_to_cm = 0.3125 / 10 * scale_factor
    dist_top_cm = radius * pixel_to_cm
    dist_right_cm = radius * pixel_to_cm

    fixed_radius = 200
    overlay = np.zeros((640, 640, 3), dtype=np.uint8)
    overlay.fill(50)

    canvas_center_x, canvas_center_y = 320, 320
    cv2.circle(overlay, (canvas_center_x, canvas_center_y), fixed_radius, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.arrowedLine(overlay, (canvas_center_x, canvas_center_y), (canvas_center_x, canvas_center_y - fixed_radius),
                    (0, 0, 255), 2, tipLength=0.1)
    cv2.arrowedLine(overlay, (canvas_center_x, canvas_center_y), (canvas_center_x + fixed_radius, canvas_center_y),
                    (0, 255, 0), 2, tipLength=0.1)

    font_scale = 0.6
    font_thickness = 2
    cv2.putText(overlay, f"{dist_top_cm:.2f} cm", (canvas_center_x - 80, canvas_center_y - fixed_radius - 20),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), font_thickness)
    cv2.putText(overlay, f"{dist_right_cm:.2f} cm", (canvas_center_x + fixed_radius + 10, canvas_center_y),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), font_thickness)

    return overlay