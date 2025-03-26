import cv2
import numpy as np
import face_recognition

imgShank = face_recognition.load_image_file('ImageBasic/Shank.jpg')
imgShank = cv2.cvtColor(imgShank,cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file('ImageBasic/ShankTest.jpg')
imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgShank)[0]
encodeShank = face_recognition.face_encodings(imgShank)[0]
cv2.rectangle(imgShank,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)

faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2)

results = face_recognition.compare_faces([encodeShank],encodeTest)
faceDis = face_recognition.face_distance([encodeShank],encodeTest)
print(results,faceDis)
cv2.putText(imgTest,f'{results} {round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_TRIPLEX,1,(0,0,255),2)

cv2.imshow('Shank',imgShank)
cv2.imshow('ShankTest',imgTest)
cv2.waitKey(0)