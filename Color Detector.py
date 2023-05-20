import cv2
import numpy as np

#95-135

lower = np.array([100,170,90])
upper = np.array([130,255,255])

video = cv2.VideoCapture(0)

#rtmp="rtmp://172.20.10.12:1935/live"
#video = cv2.VideoCapture(rtmp)

def windowResize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)

while True:
    sucesss, img=video.read()
    image2 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(image2,lower,upper)

    contours, hierachy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) #Simple saves necessary points rather than chain_approx_none, saves memory

    if len(contours) !=0:
        for contour in contours: 
            if cv2.contourArea(contour) > 500:
                x,y,w,h=cv2.boundingRect(contour)
                cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255), 2)

    img2=windowResize(img, width=1280)
    mask2=windowResize(mask, width=580)

    #image=cv2.resize(image2,(960,540))
    #mask=cv2.resize(mask2, (960,540))
    cv2.imshow("mask",mask2)
    cv2.imshow("webcam", img2)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break

video.release()
cv2.destroyAllWindows()