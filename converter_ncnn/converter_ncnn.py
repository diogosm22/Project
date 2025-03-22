from ultralytics import YOLO

# Load the YOLO11 model
model = YOLO("C:/Users/diogo/Desktop/python/yolov11/modelo.pt")

# Export to NCNN format
model.export(format="ncnn")  # creates '/yolo11n_ncnn_model'