from __future__ import division
import gym,random
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam  
import numpy as np
import pandas as pd
import os
import librosa as li
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten
class modeler(object):
    def __init__(self):
        self.model=None
    def _build_model(self):
        model = Sequential()
        model.add(Conv1D(64, kernel_size=3, activation="relu", input_shape=(60,4)))
        model.add(Conv1D(32, kernel_size=3, activation="relu"))
        model.add(Flatten())
        model.add(Dense(60, activation="softmax"))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.model=model
    def act(self, state):
        return self.model.predict(state)
    def replay(self, x,y):
        self.model.fit(np.array(x), np.array(y), batch_size=len(x), verbose=0)
path=r"E:\\python\\Actor_"
zcr=[]
rms=[]
mfc=[]
spc=[]
tar=[]
if __name__=="__main__":
    for i in range(1,25):
        if(i<10):
            path+="0"+str(i)
        else:
            path+=str(i)
        for r,d,f in os.walk(path):
            for file in f:
                at,sr=li.load(path+"\\"+file)
                zcr.append(np.mean(li.feature.zero_crossing_rate(at)[0]))
                rms.append(np.mean(li.feature.rms(at)[0]))
                mfc.append(np.mean(li.feature.mfcc(at)[0]))
                spc.append(np.mean(li.feature.spectral_centroid(y=at,sr=sr)[0]))
                tar.append(file.split("-")[2])
    ftur=[zcr,rms,mfc,spc]
    res=[tar]
    f=[[] for x in range(len(zcr))]
    for j in range(len(zcr)):
        f[j].append([zcr[j],rms[j],mfc[j],spc[j]])
    f=np.array(f)
    f=np.reshape(f,(1,60,4))
    a=np.array(res)
    res=np.reshape(a,(4,15))
    res=np.reshape(res,(1,60))
    model=modeler()
    model._build_model()
    model.replay(f,res)
    print(model.act(f[:5]))
