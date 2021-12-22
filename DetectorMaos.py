import cv2

from cvzone.HandTrackingModule import HandDetector

cap=cv2.VideoCapture(0)

detector = HandDetector(detectionCon=0.8, maxHands=2)

while True:
    sucess,img=cap.read()
    hands, img = detector.findHands(img)  # with draw
    cv2.imshow('Image', img)
    delay= cv2.waitKey(1)
    if delay == 27:
        break

cap.release()
cv2.destroyAllWindows()