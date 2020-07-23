import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers.pooling import GlobalAveragePooling1D
from keras.layers import Conv1D, MaxPooling1D, LSTM
from keras.layers.wrappers import TimeDistributed
from sklearn.metrics import confusion_matrix
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
import numpy as np
import itertools
import glob
import os
import json
import sys
import time
import statistics

#dictionary mapping classes to asanas
asanas = {0:'bhujangasana', 1:'padamasana', 2:'shavasana', 3:'tadasana', 4:'trikonasana', 5:'vrikshasana'}


# returns the keras model
def get_model():
    model = Sequential([
        TimeDistributed(Conv1D(16,3, activation='relu', padding = "same"),input_shape=[45,18,2]),
        TimeDistributed(BatchNormalization()),
        #TimeDistributed(MaxPooling1D()),
        TimeDistributed(Dropout(0.5)),
        #TimeDistributed(Conv1D(64,3, activation='relu',padding = "same")),
        BatchNormalization(),
        #TimeDistributed(Dropout(0.8)),
        TimeDistributed(Flatten()),
        #TimeDistributed(Dense(30,activation='softmax')),  
        LSTM(20,unit_forget_bias = 0.5, return_sequences = True),
        TimeDistributed(Dense(6,activation='softmax'))        
    ])
    adam = Adam(lr=0.0001)
    model.compile(loss='categorical_crossentropy',
              optimizer= adam,
              metrics=['accuracy'])
    return model

def makePred(Xarr):
	Xtemp = [] 
	Xtemp.append(Xarr)
	Xin = np.asarray(Xtemp) # prepared input of 45 frames for model    
	#getting the predictions for the sequence     
	Yout = model.predict(Xin) #softmax output from the model
	Ycurr = Yout[0].tolist() # current sequence
	# applying thresholding and getting classes
	thr = 0.85 #threshold value
	pred = [] # to be populated with predictions
	for frame in Ycurr:
		Ymax = max(frame)
		if Ymax < thr:
			# confidence too low
			pred.append(6)
		else:
			pred.append(frame.index(Ymax))
	#do polling
	print asanas.get(statistics.mode(pred))
	return
        

def addToPred(Xch, Xarr, count):
    Xarr[count] = Xch
    count = count + 1
    #get prediction if count is 45: all frames acquired
    if count == 45:
    	makePred(Xarr)    	
    	count = 0 # reset counter
    	Xarr = np.empty((45,18,2)) #reset array        
    return Xarr, count


def readFile(filename, Xarr, count):
    with open(filename) as json_data:
            d = json.load(json_data)
            for person in d['people']:
                A = []
                A = [person['pose_keypoints']]
                #A = [person['pose_keypoints_2d']]                
                X = np.asarray(A)
                even = np.arange(0, X.shape[1], 3)
                odd = np.arange(1, X.shape[1], 3)
                X1 =  X[:,even]
                X2 = X[:,odd]
                Xch = np.dstack((X1,X2))
                json_data.close
                # add it to current set of frames                
    return addToPred(Xch,Xarr,count)

#load the model with weights
model = get_model()
model.load_weights("yoga-new-data-seq/weights/val1-73-0.9992.hdf5")
#array to hold curent sequence and other  state variables
Xarr = np.empty((45,18,2))
lastRead = "none"
timeout = time.time() + 3
count = 0
# loop runs until no new file is generated for 3 seconds
while(True):
    #print(count)
    list_of_files = glob.glob("output/*")
    #list_of_files = glob.glob("C:/Users/JLR/Desktop/yoga/openpose-windows/output/*")
    #latest_file = max(list_of_files, key=os.path.getmtime)
    latest_file = max(list_of_files, key=os.path.getctime)
    if lastRead != latest_file:
        try:
            Xarr, count = readFile(latest_file,Xarr,count)
            timeout = time.time() + 3 # will give result if no new file for 3 secs
        except Exception as e:
            print(e) # write ; here to not print errors
    if time.time() > timeout:
        break
    lastRead = latest_file