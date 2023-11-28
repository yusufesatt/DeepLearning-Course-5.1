# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 11:23:26 2023

@author: yusuf
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import time

# %%
# Device configuration

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

# %%
# Hedef klasörümde kaç dosya olduğunu yazıyorum
path = r"C:/Users/yusuf/Desktop/DeepLearning-Course-5.1/Datasets/Fruits/Training/apple_6"
dosyalar = []

for dosya in os.listdir(path):
    if os.path.isfile(os.path.join(path, dosya)):
        dosyalar.append(dosya)

dosya_sayisi = len(dosyalar)

print(f"Klasördeki dosya sayısı: {dosya_sayisi}")

# %%

from PIL import Image

def resize_image(image_path, target_size):
    image = Image.open(image_path)
    resized_image = image.resize(target_size)
    return resized_image

resize_image("C:/Users/yusuf/Desktop/DeepLearning-Course-5.1/Datasets/Fruits/Training/apple_6", 224"")
# %%
"""
ilgili klasöre gidip list dir ile tüm image'leri arraya çevirir
"""
def read_images(path, num_img):
    array = np.zeros([num_img, 64*32]) # 0'lardan oluşan 64 satır 32 sütunluk array oluşturuyorum
    i = 0
    for img in os.listdir(path):
        img_path = path + "\\" + img
        img = Image.open(img_path, mode = "r")
        data = np.asarray(img, dtype = "uint8")
        data = data.flatten()
        array[i,:] = data
        i += 1
    return array

# %%

train_negative_path = r"C:/Users/yusuf/Desktop/DeepLearning-Course-5.1/Datasets/Fruits/Training/apple_6"
num_train_negative_img =  315
train_negative_array = read_images(train_negative_path, num_train_negative_img)