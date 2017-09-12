def dsift(image):
  import cv2
  import numpy as np
#   import pickle
  sift = cv2.xfeatures2d.SIFT_create()

#   imageNames = ['AF01HAS.jpg','AF02HAS.jpg','AF01SAS.jpg','AF02SAS.jpg','AF03HAS.jpg','AF03SAS.jpg']
#   DataLabels = np.array([1,1,2,2,1,2]).astype(np.float32).T
#   Data = []
#   for i in range(len(imageNames)):
#   image = cv2.imread(imageName,cv2.IMREAD_GRAYSCALE)
  r,c = image.shape
  keypoints = []
  step = 10
  for x in range(step,r,step):
	for y in range(step,c,step):
	  point = cv2.KeyPoint(x,y,step)
	  keypoints.append(point)
  kps,des = sift.compute(image, keypoints)
  feature = des.flatten()
  return(feature)
# 	if i == 0:
# 	  Data = feature
# 	else:
# 	  Data = np.vstack((feature,Data))
	
#   #save 
#   f = open('emtData.pkl', 'wb')
#   pickle.dump(Data,f)
#   pickle.dump(DataLabels,f)
#   f.close()
