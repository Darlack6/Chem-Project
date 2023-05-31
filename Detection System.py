from ultralytics import YOLO
import cv2
import cvzone
import math
import numpy as np
from datetime import datetime
from threading import Timer

video = cv2.VideoCapture(0)
background=cv2.imread("background.jpg")
#video=cv2.VideoCapture("Test-Videos/finalpond.mp4")
#video=cv2.VideoCapture("Test-Videos/lake.mp4")
#video=cv2.VideoCapture("Test-Videos/bottle.mp4")

video.set(3,1280)
video.set(4,720)

model=YOLO("../Yolo-Weights/yolov8l.pt")

#blue
lower = np.array([100,170,100])
upper = np.array([120,255,255])

#white
#lower = np.array([0, 0, 231])
#upper = np.array([180, 18, 255])

classNames = ["person", "bicycle", "car", "motorbike", "aqeroplane", "bus", "train", "truck", "boat",
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

confidence=counter= 0
detectedObjects=[]
time=[]
objectsWithTime=[]

while True:
    #Fetches for camera input
    ret, img=video.read()
    if img is None:
        break
    objects=img.copy()

    #To create mask:
    image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(image,lower,upper)
    contours,hierachy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    #For object detection
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
            cvzone.putTextRect(objects,f'{classNames[itemName]}{confidence}',(max(0,x1),max(35,y1)),scale=2.5,thickness=2)

    #if len(contours) !=0:
    for i in contours:
        if cv2.contourArea(i) > 700:
            #for rect around color
            x,y,w,h=cv2.boundingRect(i)
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255), 2)

            #appends object to list
            if(confidence>=0.60):
                if classNames[itemName] in detectedObjects:
                    pass
                else: #writes object and time to file
                    now=datetime.now()
                    current_time = now.strftime("%H:%M:%S")
                    detectedObjects.append(classNames[itemName])
                    time.append(current_time)
                    objectsWithTime=[detectedObjects[counter],time[counter]]
                    counter+=1
                    with open('Detected-Objects.txt','a') as f:
                        f.write(str(objectsWithTime))
                        f.write("\n")                 

    #resizes windows
    maskwindow=windowResize(mask, width=360)
    colorwindow=windowResize(img, width=360)
    objectwindow=windowResize(objects, width=720)

    #adjust windows positions
    maskwindowname="Mask Window"
    cv2.namedWindow(maskwindowname)
    #cv2.moveWindow(maskwindowname, 150,850)
    colorwindowname="Color Detector"
    cv2.namedWindow(colorwindowname)
    #cv2.moveWindow(colorwindowname, 600,850)
    objectwindowname="Object Detector"
    cv2.namedWindow(objectwindowname)
    cv2.moveWindow(objectwindowname, 300,30)

    #display windows
    #cv2.imshow(" ",background)
    cv2.imshow(maskwindowname,maskwindow)
    cv2.imshow(colorwindowname, colorwindow)
    cv2.imshow(objectwindowname, objectwindow)
    
    #exit statement
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

video.release()
cv2.destroyAllWindows()