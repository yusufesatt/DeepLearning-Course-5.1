# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 15:05:17 2023

@author: yusuf
"""

# %%

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import matplotlib.pyplot as plt
from glob import glob
import pandas as pd
import seaborn as sns

# %%

train = pd.read_csv(r'Datasets/Mnist/train.csv')
test = pd.read_csv(r'Datasets/Mnist/test.csv')

print("Train shape:",train.shape)
print("Test shape:",test.shape)

# %%

Y_train = train['label']
# Drop 'label' column
X_train = train.drop(labels = ['label'], axis=1)
 
# %%

plt.figure(figsize = (15,7))
g = sns.countplot(x=Y_train, palette="icefire")
plt.title("Number of Number Classes")
plt.show()
print("Number of number classes: ",Y_train.value_counts())

# %%
# Plot some samples
plt.subplot(3,2,1)
img1 = X_train.iloc[0].to_numpy().reshape((28,28))
plt.imshow(img1,cmap='gray')
plt.subplot(3,2,2)
img2 = X_train.iloc[10].to_numpy().reshape((28,28))
plt.imshow(img2,cmap='gray')
plt.subplot(3,2,3)
img3 = X_train.iloc[98].to_numpy().reshape((28,28))
plt.imshow(img3,cmap='gray')
plt.subplot(3,2,4)
img4 = X_train.iloc[25].to_numpy().reshape((28,28))
plt.imshow(img4,cmap='gray')
plt.subplot(3,2,5)
img5 = X_train.iloc[120].to_numpy().reshape((28,28))
plt.imshow(img5,cmap='gray')
plt.subplot(3,2,6)
img6 = X_train.iloc[264].to_numpy().reshape((28,28))
plt.imshow(img6,cmap='gray')

plt.show()

# %%
# Normalization
X_train = X_train / 255.0
test = test / 255.0
print("X_train shape: ", X_train.shape)
print("test shape: ", test.shape)

# %%
# Reshape

X_train = X_train.values.reshape(-1,28,28,1) # 784lük resim 28x28x1'e dönüştürüyoruz
test = test.values.reshape(-1,28,28,1)
print("X_train shape: ", X_train.shape)
print("test shape: ", test.shape)

# %%
# Label Encoding - IF there more Labels we could use Glob Function

from keras.utils import to_categorical
Y_train = to_categorical(Y_train, num_classes = 10)

# %%
# Train - Test(Val) Split

from sklearn.model_selection import train_test_split

X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.15, random_state = 42)

print("X_train shape",X_train.shape)
print("X_val shape",X_val.shape)
print("Y_train shape",Y_train.shape)
print("Y_val shape",Y_val.shape)

# %%
from sklearn.metrics import confusion_matrix
import itertools

from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop, Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
 
# %%
# CNN Create

num_of_classes = Y_train.shape[1]

model = Sequential()

model.add(Conv2D(filters = 128, kernel_size = (3,3), padding = "same",
                 activation = "relu", input_shape = (28,28,1)))
model.add(MaxPooling2D(pool_size = (3,3)))
#
model.add(Conv2D(64,3,3))
model.add(Activation("relu"))
model.add(MaxPooling2D(3,3))
#
model.add(Flatten())
model.add(Dense(1024)) #Hidden layer1
model.add(Activation("relu"))
model.add(Dropout(0.25)) # Dropout is a technique where randomly selected neurons are ignored during training - We apply this technique to avoid overfitting
#
model.add(Dense(num_of_classes)) # Output layer size must equal to number of classes (labels)
model.add(Activation("softmax"))

# %%
# Callback - Learning Rate Optimizer

learning_rate_optimizer = ReduceLROnPlateau(monitor = "val_accuracy",
                                           patience = 2, verbose = 1,
                                           factor = 0.5, min_lr = 0.000001)

# %%
# Define optimizer
optimizer = RMSprop()

# optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)

# %%
# Compile model

model.compile(optimizer = optimizer, loss='categorical_crossentropy', metrics = ["accuracy"])

# We need to use categorical_crossentropy as loss function because we used softmax as a last activation func and that's one of the multiclasses act function.

# %%
# Epoch and batch size

epochs = 20
batch_size = 240

# %%
# Model summary

model.summary()

# %%
# Data augmentation

datagen = ImageDataGenerator(
    shear_range = 0.2,
    zoom_range = 0.1,
    width_shift_range = 0.1,
    height_shift_range = 0.1,
    horizontal_flip = True,
    vertical_flip = True)

datagen.fit(X_train)

# %%
# Model fit

history = model.fit(datagen.flow(X_train,Y_train,
                                 batch_size = batch_size),
                                 epochs = epochs,
                                 validation_data = (X_val, Y_val),
                                 steps_per_epoch = X_train.shape[0] // batch_size,
                                 )

# %%
# Model weight save and history 

model.save_weights("CNN/Weights/mnist-20ep-240btch-optimizer-rmsprop.h5")

import json

with open("CNN/Weights/mnist-20ep-240btch-optimizer-rmsprop.json","w") as f:
    json.dump(history.history, f)

# %%
# Test data results

score = model.evaluate(X_val, Y_val, verbose = 0)
print("Test Loss : %f \nTest Accuracy : %f "%(score[0],score[1]))


# %%
# Model Evaluation

print(history.history.keys())
plt.plot(history.history["loss"], label ="Train Loss")
plt.plot(history.history["val_loss"], label ="Test Loss")
plt.legend()
plt.show()

#-----------------------------------------------------------------------

print(history.history.keys())
plt.plot(history.history["accuracy"], label ="Train Accuracy")
plt.plot(history.history["val_accuracy"], label ="Test Accuracy")
plt.legend()


# %% 
# Confusion Matrix

import seaborn as sns
import numpy as np

# Predict the values from the validation dataset
Y_pred = model.predict(X_val)
# Convert predictions classes to one hot vectors 
Y_pred_classes = np.argmax(Y_pred,axis = 1) 
# Convert validation observations to one hot vectors
Y_true = np.argmax(Y_val,axis = 1) 
# compute the confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 
# plot the confusion matrix
f,ax = plt.subplots(figsize=(15,15))
sns.heatmap(confusion_mtx, annot=True, cmap="cubehelix", linewidths=0.01,linecolor="green", fmt= '.1f',ax=ax)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# %%

# Load Model-History

import codecs
with codecs.open("CNN/Weights/mnist-20ep-240btch-optimizer-rmsprop.json", "r",encoding = "utf-8") as f:
    h = json.loads(f.read())
    
plt.plot(h["loss"], label="Train Loss")
plt.plot(h["val_loss"], label="Validation Loss")
plt.legend()
plt.show()
plt.figure()
plt.plot(h["accuracy"], label="Train acc")
plt.plot(h["val_accuracy"], label="Validation acc")
plt.legend()
plt.show()







