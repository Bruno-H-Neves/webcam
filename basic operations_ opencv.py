import cv2
import numpy as np
import mains7

webcam = False
cap = cv2.VideoCapture(0)
cap.set(10,160)
cap.set(3,1920)
cap.set(4,1080)
scale = 3
wP = 210 * scale
hP = 297 * scale



while True:
    success, img = cap.read()
    img = cv2.resize(img, (0,0),None, 0.5, 0.5)
    imgContours, conts = mains7.getContours(img, minArea=500, filter=4, showCanny=True, draw= True)
    key = cv2.waitKey(10)
    if key == 27: break


webcam.release()
cv2.destroyAllWindows()
