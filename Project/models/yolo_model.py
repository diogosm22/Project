# models/yolo_model.py

from ultralytics import YOLO
from config import MODEL_PATH  # Absolute import

def load_model():
    return YOLO(MODEL_PATH)

def predict(model, frame, conf_threshold):
    return model.predict(frame, conf=conf_threshold, verbose=False, max_det=1000)