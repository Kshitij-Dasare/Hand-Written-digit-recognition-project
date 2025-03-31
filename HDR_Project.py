import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train.shape, y_train.shape, X_test.shape, y_test.shape

def plot_input_img(i):
  plt.imshow(X_train[0], cmap='binary')
  plt.title(y_train[i])
  plt.show()

for i in range(10):
   plot_input_img(i)

X_train=X_train.astype(np.float32)/255
X_test=X_test.astype(np.float32)/255

X_train.shape

X_train=X_train.reshape((-1,28,28,1))
X_test=X_test.reshape((-1,28,28,1))


X_train.shape

y_train=keras.utils.to_categorical(y_train)
y_test=keras.utils.to_categorical(y_test)

y_train

y_test

from keras.models import Sequential
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,Dropout,MaxPool2D

model=Sequential()
model.add(Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(64,(3,3),activation='relu',))
model.add(MaxPool2D((2,2)))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(10,activation='softmax'))

model.summary()

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

from keras.callbacks import EarlyStopping, ModelCheckpoint
es=EarlyStopping(monitor='val_accuracy',min_delta=0.01,patience=4,verbose=1)
mc=ModelCheckpoint('./bestmodel.h5',monitor='val_accuracy',verbose=1,save_best_only=True)
cb=[es,mc]

his = model.fit(X_train,y_train,epochs=50,validation_split=0.3)
from tensorflow.keras.models import load_model
model.save("bestmodel.h5")  

score = model.evaluate(X_test, y_test)
print("model accuracy ")
print(score)


