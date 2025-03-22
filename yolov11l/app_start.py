import tkinter as tk
from tkinter import filedialog
from tkinter import scrolledtext
import cv2
import numpy as np
from ultralytics import YOLO
import random

def extract_bounding_box_area(frame, bbox):
    x1, y1, x2, y2 = bbox
    return frame[y1:y2, x1:x2]

def draw_measurements_on_bbox(frame, bbox, scale_factor):
    x1, y1, x2, y2 = bbox
    radius = int(max((x2 - x1), (y2 - y1)) / 2)
    pixel_to_cm = 0.35 / 10 * scale_factor

    dist_top_px = radius
    dist_right_px = radius

    dist_top_cm = dist_top_px * pixel_to_cm
    dist_right_cm = dist_right_px * pixel_to_cm

    overlay = frame.copy()
    center_x, center_y = (x2 - x1) // 2, (y2 - y1) // 2

    cv2.circle(overlay, (center_x, center_y), radius, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.arrowedLine(overlay, (center_x, center_y), (center_x, center_y - radius), (0, 0, 255), 2, tipLength=0.05)
    cv2.arrowedLine(overlay, (center_x, center_y), (center_x + radius, center_y), (0, 255, 0), 2, tipLength=0.05)

    cv2.putText(overlay, f"{dist_top_cm:.2f} cm", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(overlay, f"{dist_right_cm:.2f} cm", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return overlay

def display_results(num_holes, conf_level, detections):
    result_window = tk.Toplevel()
    result_window.title("Detection Results")
    result_window.geometry("400x300")

    tk.Label(result_window, text=f"Number of Holes Detected: {num_holes}", font=("Arial", 12)).pack()
    tk.Label(result_window, text=f"Confidence Level Used: {conf_level:.2f}", font=("Arial", 12)).pack()

    tk.Label(result_window, text="Detections:", font=("Arial", 12, "bold")).pack()

    text_area = scrolledtext.ScrolledText(result_window, wrap=tk.WORD, width=50, height=10)
    text_area.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

    for det in detections:
        text_area.insert(tk.END, f"{det}\n")

    text_area.config(state=tk.DISABLED)

    close_button = tk.Button(result_window, text="Close", command=result_window.destroy)
    close_button.pack()

model = YOLO("C:/Users/diogo/Desktop/python/yolov11/yolov11s/modelo3.pt")

def get_parameters():
    global conf_threshold, scale_factor

    def apply_settings():
        global conf_threshold, scale_factor
        conf_threshold = conf_scale.get()
        scale_factor = scale_conv.get()
        param_window.destroy()

    param_window = tk.Tk()
    param_window.title("Configurações de Detecção")

    tk.Label(param_window, text="Nível de Confiança").pack()
    conf_scale = tk.Scale(param_window, from_=0, to=1, resolution=0.01, orient=tk.HORIZONTAL)
    conf_scale.set(0.5)
    conf_scale.pack()

    tk.Label(param_window, text="Fator de Conversão de Escala").pack()
    scale_conv = tk.Scale(param_window, from_=0.1, to=100, resolution=0.1, orient=tk.HORIZONTAL)
    scale_conv.set(1.0)
    scale_conv.pack()

    apply_button = tk.Button(param_window, text="Aplicar", command=apply_settings)
    apply_button.pack()

    param_window.mainloop()

def load_image():
    root = tk.Tk()
    root.withdraw()
    return filedialog.askopenfilename(title="Selecione uma imagem", filetypes=[("Imagens", "*.jpg;*.png;*.jpeg")])

conf_threshold = 0.5
scale_factor = 1.0
get_parameters()

while True:
    image_path = load_image()
    if not image_path:
        print("Nenhuma imagem selecionada. Saindo.")
        break

    frame = cv2.imread(image_path)
    if frame is None:
        print("Erro ao carregar a imagem. Tente novamente.")
        continue

    frame_resized = cv2.resize(frame, (640, 640))
    original_image = frame_resized.copy()

    results = model.predict(frame_resized, conf=conf_threshold, verbose=False, max_det=1000)
    detections = [(int(box.xyxy[0][0]), int(box.xyxy[0][1]), int(box.xyxy[0][2]), int(box.xyxy[0][3])) for box in results[0].boxes]

    segmented_image = np.zeros_like(frame_resized)
    draw_circle_perimeters(segmented_image, detections)

    combined_image = np.hstack((original_image, segmented_image))
    cv2.imshow("Original (Esquerda) | Segmentada (Direita)", combined_image)

    num_holes = len(detections)
    detection_list = [f"Detecção {i + 1}: {det}" for i, det in enumerate(detections)]
    display_results(num_holes, conf_threshold, detection_list)

    if detections:
        selected_bbox = random.choice(detections)
        bbox_image = extract_bounding_box_area(original_image, selected_bbox)
        overlay = draw_measurements_on_bbox(bbox_image, selected_bbox, scale_factor)
        cv2.imshow("Bounding Box e Medições", overlay)
        cv2.waitKey(0)
        cv2.destroyWindow("Bounding Box e Medições")

    print("Pressione 'Q' para carregar outra imagem ou 'K' para sair.")
    key = cv2.waitKey(0) & 0xFF
    if key == ord('q'):
        cv2.destroyAllWindows()
        continue
    elif key == ord('k'):
        print("Saindo do programa.")
        cv2.destroyAllWindows()
        break
