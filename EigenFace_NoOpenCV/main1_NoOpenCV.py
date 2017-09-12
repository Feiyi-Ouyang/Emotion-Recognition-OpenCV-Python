from __future__ import division
from Fisherfaces2 import Fisherfaces2
import os
import glob
import cv2
from preProcess import preProcess
import numpy as np
import pickle

dataPath="/Users/ouyan103/Google Drive/Course/ComputerVision/CV - Project/dataBase/KDEF/"
targetEmotions=['HAS', 'SUS', 'DIS', 'NES']
targetSubjects='*'

nSuj=141; nSpl=nSuj*len(targetEmotions)
data=np.zeros((nSpl,100*100))
label=np.zeros((nSpl,1))
cnt=0
for emotion in targetEmotions:
  for imgName in glob.glob(dataPath+targetSubjects+"/*%s.JPG" %emotion):
    img=cv2.imread(imgName,0)    
    img_pp=preProcess(img)    
    data[cnt,:]=img_pp
    label[cnt,:]=targetEmotions.index(emotion)
    cnt+=1

idx=np.arange(nSpl); np.random.shuffle(idx)
data=data[idx,:]; label=label[idx,:]    

f=open('data.pkl','wb')
pickle.dump(data,f)
pickle.dump(label,f)
f.close()

