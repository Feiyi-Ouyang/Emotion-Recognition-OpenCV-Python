import cv2
import numpy as np
import os

# target emotions and emojis
targetEmotion=["HAS","NEU","SUS","DIS"]
emojiPath=os.getcwd()
emojis=["happy","neutral","surprise","disgust"]

# load trained fisher face recognizer
fisherFace = cv2.face.createFisherFaceRecognizer() 
fisherFace.load('fisherFace.xml')

# create face detector
faceDet = cv2.CascadeClassifier('face.xml')

# create the camera object
cap=cv2.VideoCapture(0)

while(True):
  ret,frame=cap.read()  
  # detect face 
  img=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  face=faceDet.detectMultiScale(frame)
  
  if len(face)!=0:
    # choose the first detected face by default 
	faceImg=img[face[0,1]:face[0,1]+face[0,3],face[0,0]:face[0,0]+face[0,2]]  

	# preprocess the face image
	faceImg_rs=cv2.resize(faceImg,(350,350))
	faceImg_eh=cv2.equalizeHist(faceImg_rs) 
	faceImg_norm=faceImg_eh
	norm=np.zeros(np.shape(faceImg_eh))
	norm=cv2.normalize(faceImg_norm, norm, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U) # normalize image
	pred,conf=fisherFace.predict(faceImg_norm)
	print targetEmotion[pred]
	
	# add emoji to the frame
	emojiName=emojiPath+"/"+emojis[pred]+".png"
	emojiImg=cv2.imread(emojiName)
	shape=frame[face[0,1]:face[0,1]+face[0,3],face[0,0]:face[0,0]+face[0,2]].shape
	# resize emoji
	emojiImg=cv2.resize(emojiImg, (shape[0], shape[1]))
	# create mask
	emoji_gray=cv2.cvtColor(emojiImg, cv2.COLOR_BGR2GRAY)
	_, mask=cv2.threshold(emoji_gray, 50, 255, cv2.THRESH_BINARY)
	mask_inv=cv2.bitwise_not(mask)
	# replace the face area with emoji
	roi=frame[face[0,1]:face[0,1]+face[0,3],face[0,0]:face[0,0]+face[0,2]]  
	roi=cv2.bitwise_and(roi, roi, mask=mask_inv)
	frame[face[0,1]:face[0,1]+face[0,3],face[0,0]:face[0,0]+face[0,2]]=cv2.add(roi, emojiImg)
	# show emoji on the frame
	cv2.imshow('frame',frame)

	
  # break if any key	
  if cv2.waitKey(150)>=0 & cv2.waitKey(150)!=255:
    break
    
cap.release()
cv2.destroyAllWindows()