import time
import numpy as np
import cv2
import os
import random
import matplotlib.pyplot as plt
import pickle
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import TensorBoard

NAME=f'cat-vs-dog-prediction-{int(time.time())}'
tensorboard=TensorBoard(log_dir=f'logs\\{NAME}\\')

DIRECTORY=r'D:\project\DL\dogsvscats\data\dataset\training_set'
CATEGORIES = ['cats','dogs']

IMG_SIZE = 100
data=[]

for category in CATEGORIES:
    folder=os.path.join(DIRECTORY,category)
    # print(folder)
    for img in os.listdir(folder):
        img_path=os.path.join(folder,img)
        label= CATEGORIES.index(category)  
        # print(img_path)
        img_arr=cv2.imread(img_path)
        img_arr=cv2.resize(img_arr,(IMG_SIZE,IMG_SIZE))
        # cv2.imshow("Image",img_arr)
        # cv2.waitKey(0)
        # break
        data.append([img_arr,label])

# print(len(data))
random.shuffle(data)

x=[]
y=[]

for features, labels in data:
    x.append(features)
    y.append(labels)

x=np.array(x)
y=np.array(y)

# print(len(x))
# print(len(y))

pickle.dump(x,open('X.pkl','wb'))
pickle.dump(y,open('Y.pkl','wb'))

x=x/255

model=Sequential()

model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D((2,2)))

model.add(Flatten())

model.add(Dense(128, input_shape=x.shape[1:],activation='relu'))

model.add(Dense(2,activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x,y,epochs=5,validation_split=0.1,batch_size=32, callbacks=[tensorboard])


