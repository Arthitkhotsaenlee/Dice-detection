import cv2
import numpy as np
from ultralytics import YOLO
import math
import cvzone



cap = cv2.VideoCapture(0)
# cap.set(3,640)
# cap.set(4,640)

model = YOLO("runs/detect/train/weights/best.pt")
classes = ['1','2','3','4','5','6']
cri_conf = 0.3 # the confidence ceriteria

while True:
    suc, img = cap.read()
    if suc:
        results = model(img,stream=True)
        dices_val = []
        total_conf = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                if box and math.ceil((box.conf[0]*1000))/1000 > cri_conf:
                    cls = classes[int(box.cls[0])]
                    dices_val.append(int(cls))
                    conf =  math.ceil((box.conf[0]*1000))/1000
                    total_conf.append(conf)
                    print(cls,conf)
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    # cal with and high
                    w, h = x2-x1, y2-y1 
                    cvzone.cornerRect(img, (x1, y1, w, h))
        if len(total_conf)>0:        
            avg_result = np.array(total_conf).max()
        if len(dices_val) == 3 and avg_result > cri_conf:
            sum_val = np.array(dices_val).sum()
            if sum_val < 11:
                sum_label = "LOW"
            elif sum_val>11:
                sum_label = "HI"
            else:
                sum_label = "11 HI-LOW"
            print(sum_label,)
            cvzone.putTextRect(img, sum_label,(400,50),)
            cvzone.putTextRect(img, str(avg_result),(400,100),)
            cvzone.putTextRect(img, str(dices_val),(400,150),)
        cv2.imshow("Dice-detection", img)
        # Waits for a user input to quit the application
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()