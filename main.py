import cv2
import numpy as np
from ultralytics import YOLO
import math
import cvzone



cap = cv2.VideoCapture(0)
# cap.set(3,640)
# cap.set(4,640)

size = 640

model = YOLO("runs/detect/train/weights/best.pt")
classes = ['1','2','3','4','5','6']
cri_conf = 0.3 # the confidence ceriteria
im_x, im_y = 1280, 720
x_ratio = im_x/size
y_ratio = im_y/size
while True:
    suc, img = cap.read()
    # im_y, im_x = img.shape[0], img.shape[1]
    # print(im_x, im_y)
    resize = cv2.resize(img,(size,size),interpolation = cv2.INTER_LINEAR)
    gray = cv2.cvtColor(resize, cv2.COLOR_BGR2GRAY)
    img_conv = cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB)
    # img[0:100,0:100,0] = np.zeros((100,100))
    im_crop = np.array([])
    if suc:
        results = model(img_conv,stream=True)
        dices_val = []
        total_conf = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                if box and math.ceil((box.conf[0]*1000))/1000 > cri_conf:
                    # get boundary
                    x1, y1, x2, y2 = box.xyxy[0] #get boundary
                    # im_crop = img_conv[int(y1-20):int(y2+20),int(x1-20):int(x2+20),:] # crop images
                    # get information
                    cls = classes[int(box.cls[0])]
                    dices_val.append(int(cls))
                    conf =  math.ceil((box.conf[0]*1000))/1000
                    total_conf.append(conf)
                    # make classification again
                    x1, y1, x2, y2 = int(x1*x_ratio), int(y1*y_ratio), int(x2*x_ratio), int(y2*y_ratio)
                    # cal with and high
                    w, h = x2-x1, y2-y1 
                    cvzone.cornerRect(img, (x1, y1, w, h))
                    cvzone.putTextRect(img, f"{cls} {conf}",(max(0,x1),max(0,y1)))
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
        # if im_crop.any():
            # cv2.imshow("conver",cv2.resize(im_crop,(size,size)))
        
        # Waits for a user input to quit the application
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()