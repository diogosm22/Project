import cv2
from ultralytics import YOLO

# Load YOLO model
model = YOLO("C:/Users/diogo/Desktop/python/yolov11/yolov11s/modelo.pt")

# Initialize camera
cap = cv2.VideoCapture(0)

# Validate camera
if not cap.isOpened():
    print("Erro ao abrir a câmera.")
    exit()

# Real-time detection loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform YOLO object detection
    results = model(frame, conf=0.6)

    # Draw boxes on detections
    for result in results[0].boxes:
        x1, y1, x2, y2 = map(int, result.xyxy[0])  # Get box coordinates
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Show frame
    cv2.imshow("Detection", frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
