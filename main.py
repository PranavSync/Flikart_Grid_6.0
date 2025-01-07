from ultralytics import YOLO
import os
import sqlite3
from collections import Counter

model = YOLO("yolo11n.pt")
label_count_file = "/home/pranav455/grid001/code/labels_count.txt"

train_results = model.train(
    data="/home/pranav455/grid001/code/testing.yaml",  
    epochs=100,  
    imgsz=640,  
    device=0, 
    lr0=0.001,
)

val_results = model.predict(source="/home/pranav455/grid001/code/data/images/val", save=False)
def update_counts(results):
    label_counter = Counter()

    for result in results:  
        for pred in result.boxes.cls: 
            label_counter[int(pred)] += 1  
    
    with open(label_count_file, "a") as f:
        for label, count in label_counter.items():
            f.write(f"Label: {label}, Count: {count}\n")

update_counts(val_results)
