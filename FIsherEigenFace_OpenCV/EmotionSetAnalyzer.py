from __future__ import division
import os
import glob
import cv2
import numpy as np
import copy


def expandlist(items):
	if len(items) == 1:
		return([items])
	else:
		y = items.pop()
		t = expandlist(items)
		s = copy.deepcopy(t)
		for i in t:
			i.append(y)
		for i in s:
			t.append(i)
		t.append([y])
		return(t)
		
		
faceDet = cv2.CascadeClassifier('face.xml')
fisherFace = cv2.face.createFisherFaceRecognizer() 
eigenFace = cv2.face.createEigenFaceRecognizer() 
dataPath="/Users/ouyan103/Google Drive/Course/ComputerVision/CV - Project/dataBase/KDEF"
savePath="/Users/ouyan103/Google Drive/Scripts/OpenCVLearning/Python/emotionDet/fisherDet"
os.chdir(dataPath)

targetEmotions=["HAS","SAS","SUS","DIS","ANS","AFS"]
lists = expandlist(targetEmotions)  #creates a superset of targetemotions

pops = []	#removes sets of less than 3 or more than 4
for x in lists:   
	if ('HAS' in x and len(x)>2): # want to include happy, test 3 first
		pass		
	else:
		pops.append(x)
for x in pops:
	lists.remove(x)
for x in lists:		#adds neutral state
	x.append("NES")

print lists
maxcorrect = 0	#sets up base to compare
maxcorrect2 = 0
topcorrect = ''
topcorrect2 = ''

for targetEmotion in lists:

	n=2000
	trainingData=np.zeros((n,350,350))
	trainingLabel=np.zeros((n,1))
	testData=np.zeros((n,350,350))
	testLabel=np.zeros((n,1))
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
			if cnt % 5 == 0:
				r=np.where(testData[:,1,1]==0)[0][0]
				testData[r,:,:]=img
				testLabel[r,:]=targetEmotion.index(emotion)		
			else:
				r=np.where(trainingData[:,1,1]==0)[0][0]
				trainingData[r,:,:]=img
				trainingLabel[r,:]=targetEmotion.index(emotion)
			cnt+=1	

	aveCR=0
	fault=[]
	aveCR2=0
	fault2=[]
	
	r=np.where(trainingData[:,1,1]==0)[0][0]
	fisherFace.train(np.asarray(trainingData[:r,:,:]),np.asarray(trainingLabel[:r,0]).astype('int'))
	eigenFace.train(np.asarray(trainingData[:r,:,:]),np.asarray(trainingLabel[:r,0]).astype('int'))
	cnt=0
	correct=0
	incorrect=0
	cnt2=0
	correct2=0
	incorrect2=0
	r=np.where(testData[:,1,1]==0)[0][0]
	for image in testData[:r,:]:
		pred,conf=fisherFace.predict(image)
		pred2,conf2=fisherFace.predict(image)
		if pred==testLabel[cnt,0]:
			correct+=1
			cnt+=1
		else:
			incorrect+=1
			cnt+=1  
			fault.append((testLabel[cnt2,0],pred2))
		if pred2==testLabel[cnt,0]:
			correct2+=1
			cnt2+=1
		else:
			incorrect2+=1
			cnt2+=1  
			fault2.append((testLabel[cnt2,0],pred2))
	aveCR+=round(correct/(correct+incorrect),4)
	aveCR2+=round(correct2/(correct2+incorrect2),4)
	
	CorrectRate = aveCR*100
	CorrectRate2 = aveCR*100
	if CorrectRate > maxcorrect:
		maxcorrect = CorrectRate
		topcorrect = targetEmotion
	if CorrectRate2 > maxcorrect2:
		maxcorrect2 = CorrectRate2  
		topcorrect2 = targetEmotion
print 'maxcorrect (Fisher) = ' , maxcorrect , ', ' , topcorrect
print 'maxcorrect (eigenFace) = ' , maxcorrect2 , ', ' , topcorrect2
		
