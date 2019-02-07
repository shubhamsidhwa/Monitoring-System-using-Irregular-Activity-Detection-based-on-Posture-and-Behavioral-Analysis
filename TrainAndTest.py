import numpy
import pandas

from keras.models import Sequential
from keras.layers import Dense

from sklearn.preprocessing import LabelEncoder

import cv2
import numpy as np

import os
import math

# This function takes the input file path and returns the averagr of the 7 Hu moments
def MHI(video_src):
	x8 = []
	h = []

	cam = cv2.VideoCapture(video_src)

	while True:
		ret, frame = cam.read()
		if not ret:
			break
		h, w = frame.shape[:2]
		prev_frame = frame.copy()
		hsv = np.zeros((h, w, 3), np.uint8)
		hsv[:, :, 1] = 255
		motion_history = np.zeros((h, w), np.float32)
		timestamp = 0

		while True:
			ret, frame = cam.read()
			if not ret:
				break
			frame_diff = cv2.absdiff(frame, prev_frame)
			gray_diff = cv2.cvtColor(frame_diff, cv2.COLOR_BGR2GRAY)
			ret, fgmask = cv2.threshold(gray_diff, DEFAULT_THRESHOLD, 1, cv2.THRESH_BINARY)
			fgmask = cv2.GaussianBlur(fgmask, (7, 7), 0)
			fgmask = cv2.dilate(fgmask, None, iterations=2)
			timestamp += 1

			# update motion history
			mx = cv2.motempl.updateMotionHistory(fgmask, motion_history, timestamp, MHI_DURATION)
			mg_mask, mg_orient = cv2.motempl.calcMotionGradient(motion_history, MAX_TIME_DELTA, MIN_TIME_DELTA,
																apertureSize=5)
			seg_mask, seg_bounds = cv2.motempl.segmentMotion(motion_history, timestamp, MAX_TIME_DELTA)

			# normalize motion history
			mh = np.uint8(np.clip((motion_history - (timestamp - MHI_DURATION)) / MHI_DURATION, 0, 1) * 255)

			MEI = (255 * mh > 0).astype(np.uint8)
			hsv[:, :, 0] = mg_orient / 2
			hsv[:, :, 2] = mg_mask * 255
			vis1 = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

			h = cv2.HuMoments(cv2.moments(mh)).flatten()

			x8.append((h[0] + h[1] + h[2] + h[3] + h[4] + h[5] + h[6]) / 7.0)

			mh = cv2.cvtColor(mh, cv2.COLOR_GRAY2BGR)

			prev_frame = frame.copy()
			key = cv2.waitKey(1) & 0xFF
			if key == 27:
				break

	cv2.destroyAllWindows()

	return x8


seed = 7
numpy.random.seed(seed)

#The dataset that is generated before is now read in order to train and test the data
dataframe = pandas.read_csv("dataset_act.csv", header=None)

dataset = dataframe.values

# split into input (X) and output (Y) variables
X = dataset[:,0:1].astype(float)
Y = dataset[:,1]

encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)

#This training model is a sequential model with dense layers to condense the features into either 1 or 0
model = Sequential()
model.add(Dense(32, input_dim=1, kernel_initializer='normal', activation='relu'))
model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#The model is fitted for 500 epochs
model.fit(X,encoded_Y, epochs=500, verbose=True)

print('MODEL IS FITTED')

#The default values for Hu moment computation are set
MHI_DURATION = 0.5
DEFAULT_THRESHOLD = 32
MAX_TIME_DELTA = 0.25
MIN_TIME_DELTA = 0.05

#The test data path
test_path_abnormal = os.path.join(os.getcwd(),"test\\abnormal")
test_list_abnormal=os.listdir(test_path_abnormal)

X_test=np.array([])
Y_test=np.array([])

#Features are extracted for the test data
for file_abnormal in test_list_abnormal:
	video_src = test_path_abnormal + '/' + file_abnormal
	x8 = MHI(video_src)

	X_test = np.append(X_test, np.mean(x8)*10000)
	Y_test = np.append(Y_test, 0)
	print(file_abnormal, ': Feature Extraction for test Data: ',np.mean(x8)*10000)

test_path_normal = os.path.join(os.getcwd(),"test\\normal")
test_list_normal=os.listdir(test_path_normal)

#Features are extracted for the test data
for file_normal in test_list_normal:
	video_src = test_path_normal + '/' + file_normal
	x8 = MHI(video_src)

	X_test = np.append(X_test, np.mean(x8)*10000)
	Y_test = np.append(Y_test, 1)
	
	print(file_normal, 'Feature Extraction for test Data',np.mean(x8)*10000)

print('PREDICTION')

Y_predict = model.predict(X_test)

test_list=[]
test_list = np.append(test_list, test_list_abnormal)
test_list = np.append(test_list, test_list_normal)

i=0;

#Finally, the output is printed as:
for file in test_list:
	print(file)

	if Y_test[i]==0:
		print('Ground Truth: ','Abnormal')
	else:
		print('Ground Truth: ', 'Normal')

	if math.floor(Y_predict[i])==0:
		print('Predicted: ','Abnormal')
	else:
		print('Predicted: ', 'Normal')

	print('')

	i=i+1
