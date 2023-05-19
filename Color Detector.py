import cv2
import numpy as np

#95-135

lower = np.array([100,170,90])
upper = np.array([130,255,255])

#video = cv2.VideoCapture(0)

rtmp="rtmp://172.20.10.11:1935/live"
video = cv2.VideoCapture(rtmp)

while True:
    sucesss, img=video.read()
    image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(image,lower,upper)

    contours, hierachy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_none) #Simple saves necessary points rather than chain_approx_none, saves memory

    if len(contours) !=0:
        for contour in contours: 
            if cv2.contourArea(contour) > 500:
                x,y,w,h=cv2.boundingRect(contour)
                cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255), 2)
    
    cv2.imshow("mask",mask)
    cv2.imshow("webcam", img)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break

video.release()
cv2.destroyAllWindows()