from ultralytics import YOLO
from PIL import Image
import cv2
import cvzone
import math

model = YOLO("yolov8n.pt")

# accepts all formats - image/dir/Path/URL/video/PIL/ndarray. 0 for webcam
# results = model.predict(source="0")
# results = model.predict(source="folder", show=True) # Display preds. Accepts all YOLO predict arguments

# from PIL
# im1 = Image.open("bus.jpg")
# results = model.predict(source=im1, save=True)  # save plotted images

# from ndarray
# im2 = cv2.imread("data_set/test/images/512_1_mp4-67_jpg.rf.202d9fa44c92071050b7d91665bad123.jpg")
# results = model.predict(source=im2, save=True, save_txt=True)  # save predictions as labels

# from list of PIL/ndarray
# results = model.predict(source=[im1, im2])

cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

label = ["person"]

model = YOLO("yolov8n.pt")
while True:
    suc, img = cap.read()
    results = model(img,stream=True)
    for r in results:
        boxs = r.boxes
        for box in boxs:
            cls = box.cls[0]
            conf = math.ceil((box.conf[0]*1000))/1000
            if cls == 0:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                # cv2.rectangle(img, (x1,y1), (x2,y2), (255,0,0), 3)
                w, h = x2-x1, y2-y1
                cvzone.cornerRect(img, (x1, y1, w, h))
                cvzone.putTextRect(img,f'{conf} {label[0]}', (max(0,x1), (max(0,y1))))

    cv2.imshow("image", img)
    cv2.waitKey(1)
