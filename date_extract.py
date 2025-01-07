import os
import cv2 as cv
from ultralytics import YOLO
import easyocr

image_folder = '/home/pranav455/grid001/code/data/images/dates/' 
output_file = 'detected_texts.txt' 
cropped_images_folder = 'cropped_images/' 


os.makedirs(cropped_images_folder, exist_ok=True)

model = YOLO("/home/pranav455/grid001/code/runs/detect/train4/weights/best.pt")
reader = easyocr.Reader(['en'], gpu=True)

# Open the output file
with open(output_file, 'w') as f:

    for filename in os.listdir(image_folder):
        if filename.lower().endswith(('.jpeg', '.jpg', '.png')):

            img_path = os.path.join(image_folder, filename)
            img = cv.imread(img_path)

            results = model(img_path)
            detected_texts = []

            #To crop the detected image
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0]) 
                    cropped_image = img[y1:y2, x1:x2] 

                    cropped_image_path = os.path.join(cropped_images_folder, f"{filename}_cropped.jpg")
                    cv.imwrite(cropped_image_path, cropped_image)

                    #to read text from cropped image
                    ocr_results = reader.readtext(cropped_image)

                    for (_, text, _) in ocr_results:
                        detected_texts.append(text)

            detected_texts_line = ' '.join(detected_texts)
            f.write(f"Results for {filename}: {detected_texts_line}\n")

print(f"Text detection results have been saved to {output_file}.")
print(f"Cropped images have been saved to {cropped_images_folder}.")
