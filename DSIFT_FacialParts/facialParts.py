# can't detect the whole nose, only the nostrills
# mouth detector always detects eyes..
import os
import glob
import cv2
import numpy as np

dataPath="/Users/ouyan103/Google Drive/Course/ComputerVision/CV - Project/dataBase/KDEF"
savePath="/Users/ouyan103/Google Drive/Scripts/OpenCVLearning/Python/emotionDet/DSIFT"
os.chdir(dataPath)

sift = cv2.xfeatures2d.SIFT_create()
faceDet = cv2.CascadeClassifier(savePath+'/face.xml')
mouthDet = cv2.CascadeClassifier(savePath+'/mouth.xml')

targetEmotion=["HAS","SUS","ANS","DIS","SAS"]
for emotion in targetEmotion:
  imgNames=glob.glob("./*/*%s.JPG" %emotion)
  for imgName in imgNames:
	img0=cv2.imread(imgName,0) # rawImg
	faceRect=faceDet.detectMultiScale(img0)
	for (x,y,w,h) in faceRect:
	  cv2.rectangle(img0,(x,y),(x+w,y+h),(0,0,0),2)
	  img1=img0[y:y+h,x:x+w]
	  mouthRect=mouthDet.detectMultiScale(img1,minSize=(100,80),maxSize=(500,100))	
	  if np.shape(mouthRect)[0]==3:
		mouthRect=sorted(mouthRect,key=lambda x:x[1])
		if mouthRect[1][1]+mouthRect[1][3]>mouthRect[2][1]: # delete those overlapping 
		  break
	  if np.shape(mouthRect)[0]>3:
		mouthRect=sorted(mouthRect,key=lambda x:x[1])
		del mouthRect[2:-1] # only keeps the up two and the lowest
	  if np.shape(mouthRect)[0]<3: 
		break
	  mouthRect=sorted(mouthRect,key=lambda x:x[0])		
	  count=0
	  partName=["leye","mouth","reye"]
	  for (mx,my,mw,mh) in mouthRect:
		cv2.rectangle(img1,(mx,my),(mx+mw,my+mh),(255,0,0),2)
		cv2.imwrite(savePath+'/'+emotion+'/'+partName[count]+'/'+imgName[-11:],img1[my:my+mh,mx:mx+mw])
		count+=1
	cv2.imshow('img',img0)
	cv2.waitKey(90000)

cv2.destroyAllWindows()  
os.chdir(savePath)
