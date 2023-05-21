import cv2
import cvlib as cv
from cvlib.object_detection import draw_bbox
import numpy as np

#95-135

lower = np.array([100,170,90])
upper = np.array([130,255,255])

video = cv2.VideoCapture(0)

#rtmp="rtmp://172.20.10.12:1935/live"
#video = cv2.VideoCapture(rtmp)

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

while True:
    
    ret, frame=video.read()
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(image,lower,upper)

    contours,hierachy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if len(contours) !=0:
        for i in contours:
            if cv2.contourArea(i) > 500:
                bbox, label, conf = cv.detect_common_objects(frame)
                output = draw_bbox(frame,bbox,label,conf)
            else:
                output=frame


    livefeed=windowResize(frame, width=1280)
    mask2=windowResize(mask, width=580)
    objectWindow=windowResize(output,width=580)

    cv2.imshow("mask",mask2)
    cv2.imshow("webcam", livefeed)
    cv2.imshow("Object dectection", objectWindow)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break

video.release()
cv2.destroyAllWindows()