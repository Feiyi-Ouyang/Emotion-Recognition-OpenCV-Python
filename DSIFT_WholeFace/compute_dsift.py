import cv2
import os
import numpy as np
import pickle
from dsift import dsift

files = []
num_happy, num_sad, num_angry = 0, 0, 0
for file in os.listdir('happy'):
	if file.split('.')[-1] != 'JPG':
		continue
	num_happy += 1
	files.append('happy/'+file)
for file in os.listdir('sad'):
	if file.split('.')[-1] != 'JPG':
		continue
	num_sad += 1
	files.append('sad/'+file)
for file in os.listdir('angry'):
	if file.split('.')[-1] != 'JPG':
		continue
	num_angry += 1
	files.append('angry/'+file)

print files[0]
image = cv2.imread(files[0], cv2.IMREAD_GRAYSCALE)
feature = dsift(image)
Data = feature
length = feature.shape[0]
for i, file in enumerate(files[1:]):
	image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
	feature = dsift(image)
	if feature.shape[0] != length:
		if file.split('/')[0] == 'happy':
			num_happy -= 1
		if file.split('/')[0] == 'sad':
			num_sad -= 1
		if file.split('/')[0] == 'angry':
			num_angry -= 1
		os.remove(file)
		continue
	print file
	Data = np.vstack((feature, Data))

h = np.full((num_happy,), 1)
s = np.full((num_sad,), 2)
a = np.full((num_angry,), 3)
DataLabels = np.hstack((a, s, h))

print 'Writing pickled file...'
# save
f = open('Data.pkl', 'wb')
pickle.dump(Data, f)
pickle.dump(DataLabels, f)
f.close()
