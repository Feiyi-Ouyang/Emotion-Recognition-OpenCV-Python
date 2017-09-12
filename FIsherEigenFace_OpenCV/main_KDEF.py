from __future__ import division
import os
import glob
import cv2
import numpy as np

faceDet = cv2.CascadeClassifier('face.xml')
fisherFace = cv2.face.createFisherFaceRecognizer() 
eigenFace = cv2.face.createEigenFaceRecognizer() 
dataPath="/Users/ouyan103/Google Drive/Course/ComputerVision/CV - Project/dataBase/KDEF"
savePath="/Users/ouyan103/Google Drive/Scripts/OpenCVLearning/Python/emotionDet/fisherDet"
os.chdir(dataPath)
targetEmotion=['HAS', 'SUS', 'DIS', 'NES']

n=2000
trainingData=np.zeros((5,n,350,350))
trainingLabel=np.zeros((5,n,1))
testData=np.zeros((5,n,350,350))
testLabel=np.zeros((5,n,1))
cnt=0
for emotion in targetEmotion:
  for imgName in glob.glob("./*/*%s.JPG" %emotion):
    img=cv2.imread(imgName,0)
    face=faceDet.detectMultiScale(img)
    img=img[face[0,1]:face[0,1]+face[0,3],face[0,0]:face[0,0]+face[0,2]]
    img=cv2.resize(img,(350,350))
    img=cv2.equalizeHist(img) # to improve contrast
    normImg=np.zeros(np.shape(img))
    img=cv2.normalize(img, normImg, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U) # normalize image
    for i in range(5):
	  if cnt % 5 == i:
	    r=np.where(testData[i,:,1,1]==0)[0][0]
	    testData[i,r,:,:]=img
	    testLabel[i,r,:]=targetEmotion.index(emotion)		
	  else:
	    r=np.where(trainingData[i,:,1,1]==0)[0][0]
	    trainingData[i,r,:,:]=img
	    trainingLabel[i,r,:]=targetEmotion.index(emotion)
    cnt+=1

aveCR=0
fault=[]
aveCR2=0
fault2=[]
for i in range(5):
  r=np.where(trainingData[i,:,1,1]==0)[0][0]
  fisherFace.train(np.asarray(trainingData[i,:r,:,:]),np.asarray(trainingLabel[i,:r,0]).astype('int'))
  eigenFace.train(np.asarray(trainingData[i,:r,:,:]),np.asarray(trainingLabel[i,:r,0]).astype('int'))
  cnt=0
  correct=0
  incorrect=0
  cnt2=0
  correct2=0
  incorrect2=0
  r=np.where(testData[i,:,1,1]==0)[0][0]
  for image in testData[i,:r,:]:
	pred,conf=fisherFace.predict(image)
	pred2,conf2=fisherFace.predict(image)
	if pred==testLabel[i,cnt,0]:
	  correct+=1
	  cnt+=1
	else:
	  incorrect+=1
	  cnt+=1  
	  fault.append((testLabel[i,cnt2,0],pred2))
	if pred2==testLabel[i,cnt,0]:
	  correct2+=1
	  cnt2+=1
	else:
	  incorrect2+=1
	  cnt2+=1  
	  fault2.append((testLabel[i,cnt2,0],pred2))
  aveCR+=round(correct/(correct+incorrect),4)
  aveCR2+=round(correct2/(correct2+incorrect2),4)
  print "fisherFace correct rate: ", round(correct/(correct+incorrect),4)*100,"%"
  print "EigenFace correct rate: ", round(correct2/(correct2+incorrect2),4)*100,"%"
print "fisherFace average correct rate: ", aveCR/5*100,"%"
print "eigenFace average correct rate: ", aveCR2/5*100,"%"
os.chdir(savePath)
fisherFace.save("./fisherFace.xml")
eigenFace.save("./eigenFace.xml")
