def preProcess(img):
# return a flattened, preprocessed image
  import cv2
  import numpy as np
  faceDet=cv2.CascadeClassifier('face.xml')  
  faceRect=faceDet.detectMultiScale(img)
  img=img[faceRect[0,1]:faceRect[0,1]+faceRect[0,3],faceRect[0,0]:faceRect[0,0]+faceRect[0,2]]          
  img=cv2.resize(img,(100,100))
  img=cv2.equalizeHist(img) 
  normImg=np.zeros(np.shape(img))
  img=cv2.normalize(img, normImg, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U) 
  img=img.flatten()
  return(img)
