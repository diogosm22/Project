import cv2
from ultralytics import YOLO

# Load YOLO model
model = YOLO("C:/Users/diogo/Desktop/python/yolov11/yolov11s/modelo.pt")


# Initialize camera with DirectShow for lower latency
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Set camera resolution & FPS (adjust to match your webcam specs)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_FPS, 4)  # Adjust based on your camera's max FPS

# Validate camera
if not cap.isOpened():
    print("Erro ao abrir a câmera.")
    exit()

# Function to draw circles around detected objects
def draw_circles(frames, detections):
    for x1, y1, x2, y2, label in detections:
        center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
        radius = int(max(abs(x2 - x1), abs(y2 - y1)) / 2)

        cv2.circle(frame, center, radius, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, label, (center[0] - radius, center[1] - radius - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2, cv2.LINE_AA)

# Real-time detection loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("Erro ao capturar o frame.")
        break

    # Perform YOLO object detection
    results = model(frame, conf=0.75, verbose=False)

    detections = []  # Store detections as (x1, y1, x2, y2, label)
    detected_count = 0  # Counter

    for result in results[0].boxes:
        x1, y1, x2, y2 = map(int, result.xyxy[0])  # Get box coordinates
        conf = float(result.conf[0])  # Get confidence score

        if conf > 0.70:  # Confidence threshold
            label = f"Hole {conf*100:.0f}"
            detections.append((x1, y1, x2, y2, label))
            detected_count += 1

    # Filter detections for 'Hole' class
    hole_detections = [detection for detection in detections if "Hole" in detection[4]]

    # Display the count of hole detections
    cv2.putText(frame, f"Total: {len(hole_detections)}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Draw circles on detections
    draw_circles(frame, detections)

    # Show frame in high resolution
    cv2.imshow("Hole_Detection", frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
