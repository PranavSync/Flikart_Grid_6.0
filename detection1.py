from ultralytics import YOLO
import os
from collections import Counter

model = YOLO("/home/pranav455/grid001/code/runs/detect/train4/weights/best.pt")

label_count_file = "/home/pranav455/grid001/code/detection_labels_count.txt"
image_dir = "/home/pranav455/grid001/code/data/images/detect/" 

image_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir) if img.endswith(('.png', '.jpg', '.jpeg'))]

label_map = {
    0: "dove_deo",
    1: "lays_packet",
    2: "dortitos_packet",
    3: "tide_detergent",
    4: "date_code",
}

# Function to run detection and update counts
def detect_and_count():
    label_counter = Counter()

    for img_path in image_paths:
        results = model(img_path)

        # Count the labels detected in each result
        for result in results:
            for pred in result.boxes.cls:
                label_id = int(pred)
                print(f"Detected label ID: {label_id}")
                label_name = label_map.get(label_id, f"Unknown label {label_id}")
                label_counter[label_name] += 1

    # Save the label counts to a text file
    with open(label_count_file, "w") as f: 
        for label, count in label_counter.items():
            f.write(f"Label: {label}, Count: {count}\n")

detect_and_count()
