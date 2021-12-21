import cv2
import numpy as np


def getContours(img, cThr=[100,100], showCanny=False, minArea=1000, filter=0, draw= False):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)         #RGB to Gray
    imgBlur = cv2.GaussianBlur(imgGray,(5,5),1)             #Filtro Gaussiano de suavização
    imgCanny = cv2.Canny(imgBlur, cThr[0], cThr[1])         #Filtro Canny Edge
    kernel = np.ones((5,5))                                 #Kernel para operaçoes morfologicas
    imgDial = cv2.dilate(imgCanny,kernel,iterations=2)      #dilatacao de imagem
    imgThre = cv2.erode(imgDial,kernel, iterations=1)       #erosao de imagem
    if showCanny: cv2.imshow('Canny Figure',imgThre)
    contours, hiearchy = cv2.findContours(imgThre,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)    #ident contornos
    finalCountours = []
    for i in contours:
        area = cv2.contourArea(i)
        if area > minArea:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i,0.02*peri, True)
            bbox = cv2.boundingRect(approx)
            if filter>0:
                if len(approx) == filter: finalCountours.append(len(approx), area, approx, bbox,i)
            else: finalCountours.append(len(approx), area, approx, bbox, i)
    finalCountours = sorted(finalCountours, key=lambda x:x[1], reverse=True)
    print(finalCountours)
    if draw:
        for con in finalCountours:
            cv2.drawContours(img,con[4],-1,(0,0,255),3)
    return img, finalCountours
