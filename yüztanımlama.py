import cv2
import numpy as np

face_casc= cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

img = cv2.imread("kala2.jpg")

griton = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

faces = face_casc.detectMultiScale(griton,1.1,2)

print(faces)

for(x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)


cv2.imshow("face",img)

cv2.waitKey(0)
cv2.destroyAllWindows()
