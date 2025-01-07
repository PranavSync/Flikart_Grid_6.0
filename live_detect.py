import cv2 as cv
from ultralytics import YOLO
import os

model = YOLO("/home/pranav455/grid001/code/runs/detect/train4/weights/best.pt") 

label_map = {
    0: "dove_deo",
    1: "lays_packet",
    2: "dortitos_packet",
    3: "tide_detergent",
    4: "date_code",
}

# Function to perform detection and draw bounding boxes with labels
def detect_and_annotate(frame, results):
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0]) 
            label_id = int(box.cls[0])
            label_name = label_map.get(label_id, "Unknown")

            cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cv.putText(frame, label_name, (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    return frame

def run_camera_detection():
    cap = cv.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open the camera.")
        return

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame.")
            break

        results = model(frame)
        annotated_frame = detect_and_annotate(frame, results)
        cv.imshow('Live Detection', annotated_frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

run_camera_detection()
