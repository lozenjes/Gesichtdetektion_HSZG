import cv2
import numpy
import face_recognition

#importing image and converting it to RGB
imgMessi = face_recognition.load_image_file('face/Messi.jpg')
imgMessi = cv2.cvtColor(imgMessi,cv2.COLOR_BGR2RGB)
imgMessiTest = face_recognition.load_image_file('face/Messi_test.jpg')
imgMessiTest = cv2.cvtColor(imgMessiTest,cv2.COLOR_BGR2RGB)

#detecting face
faceLoc = face_recognition.face_locations(imgMessi)[0]
encodeMessi = face_recognition.face_encodings(imgMessi)[0]
cv2.rectangle(imgMessi,(faceLoc[3], faceLoc[0]),(faceLoc[1], faceLoc[2]), (0, 255, 0), 2)
#print(faceLoc)
faceLocTest = face_recognition.face_locations(imgMessiTest)[0]
encodeMessiTest = face_recognition.face_encodings(imgMessiTest)[0]
cv2.rectangle(imgMessiTest,(faceLocTest[3], faceLocTest[0]),(faceLocTest[1], faceLocTest[2]), (0, 255, 0), 2)
#print(faceLocTest)

#comparing between faces and encoding. the smaller the closer the faces are
results = face_recognition.compare_faces([encodeMessi],encodeMessiTest)
faceDis = face_recognition.face_distance([encodeMessi],encodeMessiTest)
print(results,faceDis)
cv2.putText(imgMessiTest,f'{results} {round(faceDis[0],2)}', (50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2 )

cv2.imshow('Messi',imgMessi)
cv2.imshow('Messi Test',imgMessiTest)
cv2.waitKey(0)

