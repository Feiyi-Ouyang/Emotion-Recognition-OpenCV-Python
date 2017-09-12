from __future__ import division
from EigenFace import EigenFace
import os
import glob
import cv2
from preProcess import preProcess
import numpy as np
import pickle

f=open('data.pkl','rb')
data=pickle.load(f)
label=pickle.load(f)
f.close()

for j in range(5):
  sel_test=np.array(1);sel_train=np.array(1)
  for i in range(len(label)):
	if i % 5 == j:
	  sel_test=np.hstack((i,sel_test))
	else: 
	  sel_train=np.hstack((i,sel_train))
  sel_test=sel_test[0:-1];sel_train=sel_train[0:-1]

  trainData=data[sel_train]
  trainLabel=label[sel_train]
  testData=data[sel_test]
  testLabel=label[sel_test]

  model=EigenFace(trainData,trainLabel)
  model.train()
  cor=0
  for sel in range(len(testLabel)):
	if model.predict(data[sel,:])==label[sel]:
	  cor+=1
  print "The ", j, "th fold accuracy: ", cor/len(testLabel)*100,"%"




