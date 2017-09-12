from __future__ import division
import glob
import cv2
import numpy as np
import pickle
from matplotlib import pyplot as plt
from dsift import dsift
import pickle
savePath="/Users/ouyan103/Google Drive/Scripts/OpenCVLearning/Python/emotionDet/DSIFT"


svms=[]
parts=["reye","leye","mouth"]
emotions=["HAS","SUS","ANS","DIS","SAS"]
for part in parts:
  svm=cv2.ml.SVM_create()
  trainD=[]  
  trainL=[]
  for emotion in emotions:
	imgNamesAll=glob.glob('%s/%s/%s/*.JPG' %(savePath,emotion,part))
	imgNamesTest=glob.glob('%s/%s/%s/*4*.JPG' %(savePath,emotion,part))
	imgNames=[item for item in imgNamesAll if item not in imgNamesTest]
	for imgName in imgNames:
	  img=cv2.imread(imgName,0)
	  img=cv2.resize(img,(130,80))
	  img=cv2.equalizeHist(img) # to improve contrast
	  normImg=np.zeros(np.shape(img))
	  normImg=cv2.normalize(img, normImg, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U) # normalize image
	  feature=dsift(normImg)
	  label=emotions.index(emotion)
	  trainD.append(feature)
	  trainL.append(label)
  trainD=np.float32(np.asarray(trainD))
  trainL=np.int64(np.asarray(trainL)).reshape((-1,1))
  svm.train(trainD,cv2.ml.ROW_SAMPLE,trainL)
  svms.append(svm)


results=[]
parts=["reye","leye","mouth"]
emotions=["HAS","SUS","DIS","SAS"]
for part in parts:
  svm=svms[parts.index(part)]
  testD=[]  
  testL=[]
  for emotion in emotions:
	imgNamesAll=glob.glob('%s/%s/%s/*.JPG' %(savePath,emotion,part))
	imgNamesTest=glob.glob('%s/%s/%s/*4*.JPG' %(savePath,emotion,part))
	imgNames=imgNamesTest
	for imgName in imgNames:
	  img=cv2.imread(imgName,0)
	  img=cv2.resize(img,(130,80))
	  img=cv2.equalizeHist(img) # to improve contrast
	  normImg=np.zeros(np.shape(img))
	  normImg=cv2.normalize(img, normImg, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U) # normalize image
	  feature=dsift(normImg)
	  label=emotions.index(emotion)
	  testD.append(feature)
	  testL.append(label)
  testD=np.float32(np.asarray(testD))
  testL=np.float32(np.asarray(testL)).reshape((-1,1))
  ret,result=svm.predict(testD)
  results.append(result)
  cr=round(np.sum(result==testL)/testL.shape[0]*100,4)
  print "correct rate: ", cr,"%"


print np.shape(imgNamesAll)
print np.shape(imgNamesTest)

# save 
f = open('data.pkl', 'wb')
pickle.dump(trainD,f)
pickle.dump(trainL,f)
pickle.dump(testD,f)
pickle.dump(testL,f)
f.close()
