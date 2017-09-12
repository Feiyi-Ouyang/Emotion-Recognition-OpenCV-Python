from scipy.sparse.linalg import eigsh
import numpy as np
import cv2

class EigenFace(object):
	
	def __init__(self,data,label):
	  self.data=data
	  self.label=label
	  
	def train(self):
	  self.mu=np.mean(self.data,axis=0)
	  self.phi=self.data-self.mu
	  scatter=np.matmul(self.phi.T,self.phi)	  	  
	  self.eigenvalues, self.eigenvectors=eigsh(scatter,12)
	  self.weights=self.project(self.data)
	  self.knn=cv2.ml.KNearest_create()
	  self.knn.train(self.weights.astype(np.float32),cv2.ml.ROW_SAMPLE,self.label.astype(np.float32))
	  
	def predict(self,newImg):
	  newImg=np.reshape(newImg,(1,-1))
	  weight=self.project(newImg)
	  ret,results,neighbours,dist=self.knn.findNearest(weight.astype(np.float32),3)
	  return ret
	  	  
	def project(self,X):
	  w=np.matmul(X-self.mu,self.eigenvectors,)
	  return w
	  

	  