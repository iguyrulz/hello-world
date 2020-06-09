import numpy as np
import cv2
import os

import facerecog as fr
#print(fr)

test_img=cv2.imread(r"C:\Users\viraj\Downloads\Test_Image.jpg")

faces_detected,gray_img=fr.faceDetection(test_img)

print("Face Detected", faces_detected)


faces,faceID=fr.labels_for_training_data(r'C:\Users\viraj\projects\Facial Recognition\Images')
face_recognizer= fr.train_Classifier(faces,faceID)
face_recognizer.save(r'C:\Users\viraj\projects\Facial Recognition\trainingData.yml')

name={0:'Viraj',1:"IGUY"}

for face in faces_detected:
    (x,y,w,h)=face
    roi_gray=gray_img[y:y+w,x:x+h]
    label,confidence=face_recognizer.predict(roi_gray)
    print(label)
    print(confidence)
    fr.draw_rect(test_img,face)
    predict_name=name[label]
    fr.put_text(test_img,predict_name,x,y)

resized_img=cv2.resize(test_img,(1000,700))

cv2.imshow("",resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
