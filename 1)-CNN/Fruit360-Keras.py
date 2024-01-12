# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 18:06:42 2023

@author: yusuf
"""

# %%

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import matplotlib.pyplot as plt
from glob import glob
# %%

train_path = r"C:/Users/yusuf/Desktop/DeepLearning-Course-5.1/Datasets/Fruits/Training/"
test_path = r"C:/Users/yusuf/Desktop/DeepLearning-Course-5.1/Datasets/Fruits/Test/"

img = load_img(train_path + "apple_red_3/r0_0.jpg")

plt.imshow(img)
plt.axis('off')
plt.show()

# %%
# Resmin shape'ine bakıyoruz
x = img_to_array(img)
print(x.shape)


# %%
# Kaç classım oldğuna glob ile bakıyoruz
className = glob(train_path + '/*')
numberOfClass = len(className)
print("Number of Class:", numberOfClass)

# %%
# CNN Model
model = Sequential()
model.add(Conv2D(32,(3,3), input_shape = x.shape))
model.add(Activation("relu"))
model.add(MaxPooling2D())
# 
model.add(Conv2D(32,(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D())
#
model.add(Conv2D(64,(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D())
#
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(Dense(numberOfClass))# Output: output sayımdaki nöron sayısı output kadar olmalı
# Outputu class sayısı belirler bu sebeple numofclassı veriyoruz
model.add(Activation('softmax'))

# %%
# Model Compile and Batch size

model.compile(loss = "categorical_crossentropy",
              optimizer = "rmsprop",
              metrics = ["accuracy"])

batch_size = 32

# %%
# Data Generation - Train,Test
# Data sayımız az olduğu için data augmentation kullanarak yeni datalar oluşturacağım

train_datagen = ImageDataGenerator(rescale = 1./255, # Img'lerim rgb olduğu için (0-255) veriyi normalize ediyorum yani 0-1 arasında rescale ediyorum
                   shear_range = 0.3, # Belli bir açıya çevriliyor
                   horizontal_flip=True, # Resmi yatay olarak döndürür
                   zoom_range=0.3) # Resmi yakınlaştırır 

test_datagen = ImageDataGenerator(rescale=1./255) # testte orjinal resimler olması daha iyi fakat bunu da rescale etmek zorundayız

#Bir klasörün altında klasör yani bir class ve içinde resimler var
# Her bir meyveyi farklı sınıflar ayırıyor her bir meyvenin resimlerini de o sınıflara depoluyor
train_generator = train_datagen.flow_from_directory(train_path,
                                                    target_size = x.shape[:2], # Burada x shapedeki 3'ü yani channeli vermiyoruz
                                                    batch_size = batch_size,
                                                    color_mode = "rgb", # Burada rgb olduğunu belirtiyoruz
                                                    class_mode = "categorical")# Birden fazla kategori oldugunu belirtir
                                                    
test_generator = train_datagen.flow_from_directory(test_path,
                                                    target_size = x.shape[:2], # Burada x shapedeki 3'ü yani channeli vermiyoruz
                                                    batch_size = batch_size,
                                                    color_mode = "rgb", # Burada rgb olduğunu belirtiyoruz
                                                    class_mode = "categorical")# Birden fazla kategori oldugunu belirtir              
# %%
# Model fit
hist = model.fit_generator(
    generator=train_generator,
    steps_per_epoch = 1600 // batch_size, # 1 epochda yapılması gereken batch sayısı 
    epochs=2,
    validation_data = test_generator,
    validation_steps = 800 // batch_size
    )

# %%
# Model save
model.save_weights("CNN/Weights/2epoch.h5")

# %%
# Model evaluation
print(hist.history.keys())
plt.plot(hist.history["loss"], label="Train Loss")
plt.plot(hist.history["val_loss"], label="Validation Loss")
plt.legend()
plt.show()
plt.figure()
plt.plot(hist.history["acc"], label="Train acc")
plt.plot(hist.history["val_acc"], label="Validation acc")
plt.legend()
plt.show()

# %%
# Save history
import json

with open("CNN/Weights/2epoch.json","w") as f:
    json.dump(hist.history, f)

# %%
# Load history
"""
Save historyden buraya kadar historydeki modelin sonuçlarını h değişkenine atadık
ve burada bunları grafiğe döktük
Görselleştirmek için jsonunu yazdırıyorum.
"""

import codecs
with codecs.open("CNN/Weights/2epoch.json", "r",encoding = "utf-8") as f:
    h = json.loads(f.read())
    
# Model evaluation
plt.plot(h["loss"], label="Train Loss")
plt.plot(h["val_loss"], label="Validation Loss")
plt.legend()
plt.show()
plt.figure()
plt.plot(h["accuracy"], label="Train acc")
plt.plot(h["val_accuracy"], label="Validation acc")
plt.legend()
plt.show()











