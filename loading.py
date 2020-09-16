import numpy as np
import cv2
import os

import facerecog as fr

test_img=cv2.imread(r'C:\Users\viraj\Downloads\Test_Image.jpg')

faces_detected,gray_img=fr.faceDetection(test_img)
print("Face Detected", faces_detected)

face_recognizer=cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read(r"C:\Users\viraj\projects\Facial Recognition\trainingData.yml")

name={0:'Viraj',1:'IGUY'}

for face in faces_detected:
    (x,y,w,h)=face
    roi_gray=gray_img[y:y+w,x:x+h]
    label,confidence=face_recognizer.predict(roi_gray)
    print("Confidence:",confidence)
    print("Label:", label)
    fr.draw_rect(test_img,face)
    predicted_name=name[label]
    fr.put_text(test_img,predicted_name,x,y)

resized_img=cv2.resize(test_img,(1000,700))

cv2.imshow("Face Detection",resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()