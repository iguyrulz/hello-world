import cv2
import sys

ct=0

vidStream= cv2.VideoCapture(0)

while True:
    ret,frame= vidStream.read()
    cv2.imshow("Test Frame", frame)

    cv2.imwrite(r"C:\Users\viraj\projects\Facial Recognition\Images\0\image%04i..jpg" %ct,frame)
    ct +=1

    if cv2.waitKey(10)==ord("q"):
        break