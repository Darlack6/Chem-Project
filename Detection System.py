from ultralytics import YOLO
import cv2
import cvzone
import math
import cvlib as cv
import numpy as np


video = cv2.VideoCapture(0)
video.set(3,1280)
video.set(4,720)

model=YOLO("../Yolo-Weights/yolov8n.pt")

lower = np.array([100,170,90])
upper = np.array([130,255,255])

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

detectedObjects=[]

def windowResize(image, width=None, height=None, inter=cv2.INTER_AREA):
    adjustedsize = None
    (h, w) = image.shape[:2]

    if width is None and height is None: # if no window size change is desired
        return image
    if width is None: #if only height is entered
        r = height / float(h)
        adjustedsize = (int(w * r), height)
    else:
        r = width / float(w) #if only width is entered
        adjustedsize = (width, int(h * r))

    return cv2.resize(image, adjustedsize, interpolation=inter)

def objectNames():
    print("hello")

while True:
    ret, img=video.read()
    objects=img.copy()

    #To create mask:
    image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(image,lower,upper)
    contours,hierachy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    results=model(img,stream=True,verbose=False)

    for r in results:
        boxes=r.boxes
        for box in boxes:
            #creates box around item
            x1,y1,x2,y2=box.xyxy[0]
            x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
            w,h = x2-x1,y2-y1
            cvzone.cornerRect(objects,(x1,y1,w,h),30,5,1,(0,255,0))

            #establishes confidence level
            confidence = math.ceil((box.conf[0]*100))/100

            #establishes item name
            itemName = int(box.cls[0])

            #prints the item name and confidence to screen
            cvzone.putTextRect(objects,f'{classNames[itemName]}{confidence}',(max(0,x1),max(35,y1)),scale=1.5,thickness=1)

    #if len(contours) !=0:
    for i in contours:
        if cv2.contourArea(i) > 700:
            #for rect around color
            x,y,w,h=cv2.boundingRect(i)
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255), 2)

            if(confidence>=0.50):
                if classNames[itemName] in detectedObjects:
                    pass
                else:
                    detectedObjects.append(classNames[itemName])

    maskwindow=windowResize(mask, width=580)
    colorwindow=windowResize(img, width=580)

    cv2.imshow("Mask",maskwindow)
    cv2.imshow("Color Detector", colorwindow)
    cv2.imshow("Object Detector", objects)

    if cv2.waitKey(1) & 0xff == ord('w'):
        print(detectedObjects)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break

video.release()
cv2.destroyAllWindows()

print(detectedObjects)
