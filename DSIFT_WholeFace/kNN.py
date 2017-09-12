import cv2
import pickle
import numpy as np
import os
from dsift import dsift

print 'Loading pickled file...'
# load
f = open('Data.pkl', 'rb')
Data = pickle.load(f)
DataLabels = pickle.load(f)
f.close()

knn = cv2.ml.KNearest_create()
knn.train(Data.astype(np.float32), cv2.ml.ROW_SAMPLE, DataLabels.astype(np.float32))

correct, incorrect = 0, 0

for i, test_file in enumerate(os.listdir('test')):
	if test_file.split('.')[1] != 'JPG':
		continue
	image = cv2.imread('test/'+test_file, cv2.IMREAD_GRAYSCALE)
	feature = dsift(image)
	newcomer = np.empty((1, feature.shape[0]))
	newcomer[:] = feature
	ret, results, neighbours, dist = knn.findNearest(newcomer.astype(np.float32), 3)
	print test_file,
	name = test_file.split('.')[0]
	if ret == 1:
		print 'happy, ',
		if name[len(name)-3:] == 'HAS':
			print 'correct'
			correct += 1
		else:
			print 'incorrect'
			incorrect += 1
	elif ret == 2:
		print 'sad',
		if name[len(name)-3:] == 'SAS':
			print 'correct'
			correct += 1
		else:
			print 'incorrect'
			incorrect += 1
	else:
		print 'angry',
		if name[len(name)-3:] == 'ANS':
			print 'correct'
			correct += 1
		else:
			print 'incorrect'
			incorrect += 1

print correct, incorrect

