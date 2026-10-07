import cv2
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO(r'C:\Users\gvams\Downloads\yolo\yolov8n.pt')

# Initialize webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        break

    # Perform inference
    results = model(frame)

    # Annotate frame
    annotated_frame = results[0].plot()

    # Display
    cv2.imshow('YOLOv8 Customer Detection', annotated_frame)

    # Press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()