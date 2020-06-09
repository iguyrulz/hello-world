import numpy as np
import cv2
import os
import csv
import facerecog as fr

print(fr)

face_recognizer=cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read(r"C:\Users\viraj\projects\Facial Recognition\trainingData.yml")

cap=cv2.VideoCapture(0)
size=(
    int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
)

name={0:'Viraj',1:'IGUY'}

while True:
    ret,test_img=cap.read()
    faces_detected,gray_img=fr.faceDetection(test_img)
    print("Face Detected", faces_detected)
    for (x,y,w,h) in faces_detected:
        cv2.rectangle(test_img,(x,y),(x+w,y+h),(0,255,0),thickness=5)

    for face in faces_detected:
        (x,y,w,h) = face
        roi_gray = gray_img[y:y+w, x:x+h]
        label, confidence = face_recognizer.predict(roi_gray)
        print("Confidence:", confidence)
        print("Label:", label)
        fr.draw_rect(test_img, face)
        predicted_name = name[label]
        if(confidence>65):
            continue
        fr.put_text(test_img, predicted_name, x, y)
        if label==0:
            print("Viraj")

    resized_img=cv2.resize(test_img,(1000,700))

    cv2.imshow("Face Detection",resized_img)

    if cv2.waitKey(10)==ord('q'):
        break

