import numpy as np
from keras.models import Sequential
from keras.layers import ZeroPadding2D,Dense,Activation,Flatten,BatchNormalization,Dropout
from keras.layers import Conv2D,MaxPooling2D
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

data=np.load('data.npy')
target=np.load('target.npy')

model=Sequential()

model.add(Conv2D(50,(3,3),input_shape=data.shape[1:]))
model.add(BatchNormalization(axis=3, name='bn0'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(100,(3,3),activation='relu'))  
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(200,(3,3),activation='relu'))  
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(50,activation='relu'))
model.add(Dense(2,activation='softmax'))

model.summary()

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

train_data,test_data,train_target,test_target=train_test_split(data,target,test_size=0.2)

checkpoint = ModelCheckpoint(
  'model-{epoch:03d}.model',
  monitor='val_loss',
  verbose=0,
  save_best_only=True,
  mode='min')

history=model.fit(
  train_data,
  train_target,
  epochs=20,
  callbacks=[checkpoint],
  validation_split=0.1)

print(model.evaluate(test_data,test_target))