from ultralytics import YOLO
import os


img_dir = os.path.join(".\data_set","train","Images","IMG_20191208_111228.jpg")
print(img_dir)
model = YOLO("yolov8n.pt")
model.predict(source=img_dir, conf=0.4)