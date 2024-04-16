import os

import cv2
import numpy
import face_recognition
import numpy as np
from datetime import datetime

#creating list from folder face_list
path = 'face_list'
images = []
classNames = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

#encoding list of images
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

#def TimeAndList (name):
 #   with open('TimeAndList.csv','r+') as f:
  #      myDataList = f.readline()
   #     nameList = []
    #    print (myDataList)
     #   for line in myDataList:
      #      entry = line.split(',')
       #     nameList.append(entry[0])
        #if name not in nameList:
         #   now = datetime.now()
          #  dtString = now.strftime('%H:%M:%S')
           # f.writelines(f'\n{name},{dtString}')


encodeListKnown = findEncodings(images)
print(len(encodeListKnown), '[Encoding completed]')

#for the capturing webcam image
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0,0), None, 0.25,0.25)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)

#it will grab face location and encoding of current frame
    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
        print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)
            y1,x2,y2,x1 = faceLoc
            y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,0,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_DUPLEX,1,(255,255,255),2)

    cv2.imshow('Webcam',img)
    cv2.waitKey(1)


#test unit
#faceLoc = face_recognition.face_locations(imgMessi)[0]
#encodeMessi = face_recognition.face_encodings(imgMessi)[0]
#cv2.rectangle(imgMessi,(faceLoc[3], faceLoc[0]),(faceLoc[1], faceLoc[2]), (0, 255, 0), 2)
#print(faceLoc)
#faceLocTest = face_recognition.face_locations(imgMessiTest)[0]
#encodeMessiTest = face_recognition.face_encodings(imgMessiTest)[0]
#cv2.rectangle(imgMessiTest,(faceLocTest[3], faceLocTest[0]),(faceLocTest[1], faceLocTest[2]), (0, 255, 0), 2)
#print(faceLocTest)


#importing image and converting it to RGB
#imgMessi = face_recognition.load_image_file('face/Messi.jpg')
#imgMessi = cv2.cvtColor(imgMessi,cv2.COLOR_BGR2RGB)
#imgMessiTest = face_recognition.load_image_file('face/Messi_test.jpg')
#imgMessiTest = cv2.cvtColor(imgMessiTest,cv2.COLOR_BGR2RGB)