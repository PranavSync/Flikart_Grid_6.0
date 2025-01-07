import cv2 as cv
from ultralytics import YOLO
import easyocr
import os

model = YOLO("/home/pranav455/grid001/code/runs/detect/train4/weights/best.pt") 

# Initialize the OCR reader
reader = easyocr.Reader(['en'], gpu=True)

output_file = 'detected_texts_camera.txt'
cropped_images_folder = 'cropped_images_camera/'
os.makedirs(cropped_images_folder, exist_ok=True)

label_map = {
    0: "dove_deo",        
    1: "lays_packet",
    2: "dortitos_packet",
    3: "tide_detergent",
    4: "date_code",       
}

target_label_name = "date_code"

# Function to detect and extract text from a specific label
def extract_text_from_label(frame, results):
    detected_texts = []
    
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get coordinates of detected box
            label_id = int(box.cls[0])  # Get the label ID
            label_name = label_map.get(label_id, "Unknown")  # Get label name

            if label_name == target_label_name:
                print(f"Detected {label_name} in frame.")

                # Crop the image around the bounding box
                cropped_image = frame[y1:y2, x1:x2]

                cropped_image_path = os.path.join(cropped_images_folder, f"frame_cropped.jpg")
                cv.imwrite(cropped_image_path, cropped_image)

                # Use EasyOCR to read text from the cropped image
                ocr_results = reader.readtext(cropped_image)

                for (_, text, _) in ocr_results:
                    detected_texts.append(text)

    # Write detected texts to the output file in one line
    if detected_texts:
        with open(output_file, 'a') as f:
            detected_texts_line = ' '.join(detected_texts)
            f.write(f"Detected in current frame: {detected_texts_line}\n")
            print(f"Extracted text: {detected_texts_line}")
    else:
        print("No relevant text detected.")

# Function to run the live camera detection
def run_camera_detection():
    cap = cv.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open the camera.")
        return

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame.")
            break

        results = model(frame)
        extract_text_from_label(frame, results)
        cv.imshow('Live Detection', frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

run_camera_detection()
