# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 11:04:45 2023

@author: yusuf
"""
# %%

from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
from glob import glob

# %%

train_path = "C:/Users/yusuf/Desktop/DeepLearning-Course-5.1/Datasets/Fruits/Training"
test_path = "C:/Users/yusuf/Desktop/DeepLearning-Course-5.1/Datasets/Fruits/Test"


img = load_img(train_path + "/apple_crimson_snow_1/r0_2.jpg")
plt.imshow(img)
plt.axis("off")
plt.show()

# %%

x = img_to_array(img)
print(x.shape)

# %%

numberOfClass = len(glob(train_path + "/*"))
print(numberOfClass)

# %%

vgg = VGG16() # vgg model create

print(vgg.summary()) # vgg layer bilgileri
print(type(vgg)) 

vgg_layer_list = vgg.layers
print(vgg_layer_list)

# %%

# Son dense layerda (predictions) 1000 classla eğitilmiş bir model var fakat benim eğiteceğim model farklılık gösöterebilir
# Bu sebeple son dense layer'ı atacağım

model = Sequential()
for i in range(len(vgg_layer_list)-1): # vgglayer listimin uzunluğunu alıyorum ve en sondaki layerı çıkarıyorum
    model.add(vgg_layer_list[i]) # ve en son layer hariç tüm layerları modelime ekliyorum
    

print(model.summary()) 

# %%

for layers in model.layers:
    layers.trainable = False 

model.add(Dense(numberOfClass, activation="softmax"))

print(model.summary()) 

# Son layerı kendimiz ekledik

# %%

model.compile(loss = "categorical_crossentropy",
              optimizer = "rmsprop",
              metrics = ["accuracy"])

# %%
# Train

train_data = ImageDataGenerator().flow_from_directory(train_path, target_size = (224,224))
test_data = ImageDataGenerator().flow_from_directory(test_path, target_size = (224,224))

batch_size = 32

hist = model.fit_generator(train_data, steps_per_epoch=1600//batch_size,
                           epochs = 25,
                           validation_data=test_data,
                           validation_steps=800//batch_size)

# %%













