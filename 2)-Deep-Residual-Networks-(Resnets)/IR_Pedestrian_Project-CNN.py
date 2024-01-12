# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 18:04:50 2023

@author: yusuf
"""

# %%

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
path = r"C:/Users/yusuf/Desktop/DeepLearning-Course-5.1/Datasets/LSIFar/LSIFIR/Classification/Train/neg"
dosyalar = []

for dosya in os.listdir(path):
    if os.path.isfile(os.path.join(path, dosya)):
        dosyalar.append(dosya)

dosya_sayisi = len(dosyalar)

print(f"Klasördeki dosya sayısı: {dosya_sayisi}")


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
# Read train negative

train_negative_path = r"C:/Users/yusuf/Desktop/DeepLearning-Course-5.1/Datasets/LSIFar/LSIFIR/Classification/Train/neg"
num_train_negative_img =  43390
train_negative_array = read_images(train_negative_path, num_train_negative_img)

# %%
# Torch'da tensorlar kullanılıyor
# Convert tensor
x_train_negative_tensor = torch.from_numpy(train_negative_array)
print("x_train_negative_tensor:", x_train_negative_tensor.size())

# %%

y_train_negative_tensor = torch.zeros(num_train_negative_img, dtype = torch.long)
print("y_train_negative_tensor:", y_train_negative_tensor.size())

# %%
# Read train positive

train_positive_path = r"C:/Users/yusuf/Desktop/DeepLearning-Course-5.1/Datasets/LSIFar/LSIFIR/Classification/Train/pos"
num_train_positive_img =  10208
train_positive_array = read_images(train_positive_path, num_train_positive_img)

# %%
# Convert tensor
x_train_positive_tensor = torch.from_numpy(train_positive_array)
print("x_train_positive_tensor:", x_train_positive_tensor.size())

y_train_positive_tensor = torch.ones(num_train_positive_img, dtype = torch.long)
print("y_train_positive_tensor:", y_train_positive_tensor.size())

# %%
# Concat train

x_train = torch.cat((x_train_negative_tensor,x_train_positive_tensor),0) # yukardan aşağı yapmasını istediğim için 0 seçiyorum 1 yapsaydım sütunları birleştirirdi.
y_train = torch.cat((y_train_negative_tensor,y_train_positive_tensor),0)
print("x_train: ", x_train.size())
print("y_train: ", y_train.size())

# %%
# -----------------------------------------------
# Test data
# read test negative  22050
test_negative_path = r"C:/Users/yusuf/Desktop/DeepLearning-Course-5.1/Datasets/LSIFar/LSIFIR/Classification/Test/neg"
num_test_negative_img = 22050
test_negative_array = read_images(test_negative_path,num_test_negative_img)
x_test_negative_tensor = torch.from_numpy(test_negative_array[:20855,:])
print("x_test_negative_tensor: ",x_test_negative_tensor.size())
y_test_negative_tensor = torch.zeros(20855,dtype = torch.long)
print("y_test_negative_tensor: ",y_test_negative_tensor.size())

# read test positive 5944
test_positive_path = r"C:/Users/yusuf/Desktop/DeepLearning-Course-5.1/Datasets/LSIFar/LSIFIR/Classification/Test/pos"
num_test_positive_img = 5944
test_positive_array = read_images(test_positive_path,num_test_positive_img)
x_test_positive_tensor = torch.from_numpy(test_positive_array)
print("x_test_positive_tensor: ",x_test_positive_tensor.size())
y_test_positive_tensor = torch.zeros(num_test_positive_img,dtype = torch.long)
print("y_test_positive_tensor: ",y_test_positive_tensor.size())

# concat test
x_test = torch.cat((x_test_negative_tensor, x_test_positive_tensor), 0)
y_test = torch.cat((y_test_negative_tensor, y_test_positive_tensor), 0)
print("x_test: ",x_test.size())
print("y_test: ",y_test.size())

# %%
# Visualize
plt.imshow(x_train[43399,:].reshape(64,32), cmap = "gray")

# %%
# CNN Model

# Hyperparameter

"""
Pytorch kütüphanesinde nn modellerini kullanıcam bunu yaparken bir class oluşturuyorum
ve nn.Module veriyorum inherit ediyor
"""
num_epochs = 80
num_classes = 2
batch_size = 8933
learning_rate = 0.00001

class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()  # nn modülünü miras alıyorum
        
        self.conv1 = nn.Conv2d(1, 10, 5)
        self.pool = nn.MaxPool2d(2, 2) 
        self.conv2 = nn.Conv2d(10, 16, 5)
        
        self.fc1 = nn.Linear(16 * 13 * 5, 520)
        self.fc2 = nn.Linear(520, 130)
        self.fc3 = nn.Linear(130, num_classes)
        
    
    def forward(self, x):
        x = self.pool(F.relu((self.conv1(x))))
        x = self.pool(F.relu(self.conv2(x)))
    
        x = x.view(-1, 16 * 13 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x
    

# %%

import torch.utils.data

train = torch.utils.data.TensorDataset(x_train, y_train)
trainloader = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle = True )

test = torch.utils.data.TensorDataset(x_test, y_test)
testloader = torch.utils.data.DataLoader(test, batch_size = batch_size, shuffle = False )

net = Net().to(device)
# %%
# Loss and optimizer

criterion = nn.CrossEntropyLoss()

import torch.optim as optim

optimizer = optim.SGD(net.parameters(), lr = learning_rate, momentum=0.8)

# %%

# Train a network
start = time.time()

train_acc = []
test_acc = []
loss_list = []

use_gpu = True

for epoch in range(num_epochs):
    for i, data in enumerate(trainloader, 0): # enumerate ederek trainloader ve 0 birleştiriyoruz ve  index ve datayı döndürüyoruz
        
        inputs, labels = data
        #input preprocess
        inputs = inputs.view(batch_size, 1, 64, 32) # reshape: 1 burda renksiz olduğunu söylüyor 64,32 de resim size
        inputs = inputs.float()# float
        
        # use gpu
        if use_gpu:
            if torch.cuda.is_available():
                inputs, labels = inputs.to(device), labels.to(device)
    
        # zore gradient
        optimizer.zero_grad() # her adımda gradientleri sıfırlamamız gerekiyor türev sıfırlama
        
        # Forward 
        outputs = net(inputs)
        
        # loss
        loss = criterion(outputs, labels) #outputlar ile y_head değerleri karşılaştırılıyor

        # Backward 
        loss.backward()
        
        # Update weights
        optimizer.step()
        
    # test
    correct = 0 # ne kadar doğru bildiğimiz
    total = 0 # 
    
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            
            images = images.view(batch_size, 1, 64, 32)
            images = images.float()
            
            if use_gpu:
                if torch.cuda.is_available():
                    images, labels = images.to(device), labels.to(device)

            outputs = net(images)
            
            _, predicted = torch.max(outputs.data,1) 
            
            total += labels.size(0) # o an kaç tane verimin olduğunu toplam veri sayımın üzerine ekleyerek gidiyor
            correct += (predicted == labels).sum().item()
            
    
    acc1 = 100*correct/total
    print(f"Epoch {epoch + 1}/{num_epochs} - Accuracy test: {acc1:.2f}%")
    test_acc.append(acc1)

    
    # train
    correct = 0 # ne kadar doğru bildiğimiz
    total = 0 # 
    
    with torch.no_grad(): # Backward prop yapmadığımız için no_grad kullanıyoruz
        for data in trainloader:
            images, labels = data
            
            images = images.view(batch_size, 1, 64, 32)
            images = images.float()
            
            if use_gpu:
                if torch.cuda.is_available():
                    images, labels = images.to(device), labels.to(device)

            outputs = net(images)
            _, predicted = torch.max(outputs.data,1) 
            total += labels.size(0) # o an kaç tane verimin olduğunu toplam veri sayımın üzerine ekleyerek gidiyor
            correct += (predicted == labels).sum().item()
            
    
    acc2 = 100*correct/total
    print(f"Epoch {epoch + 1}/{num_epochs} - Accuracy train: {acc2:.2f}%")
    train_acc.append(acc2)

print("Train is done!")

end = time.time()
process_time = (end - start)/60
print("Process time: ", process_time)

# %%

torch.save(net.state_dict(), 'trained_model_80_epoch.pth')

# %%

net = Net().to(device)

# Eğitilmiş ağırlıkları yükleyin
net.load_state_dict(torch.load("C:/Users/yusuf/Desktop/DeepLearning-Course-5.1/trained_model_80_epoch.pth"))

# Modeli değerlendirmeye alın
net.eval()

# %%
import torch
import torchvision.transforms as transforms
from PIL import Image

image_path = r"C:/Users/yusuf/Desktop/DeepLearning-Course-5.1/Datasets/LSIFar/LSIFIR/Classification/Test/neg/00005.png"
image = Image.open(image_path)

# Resmi modele uygun hale getirin
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Resmi gri tonlamaya dönüştürün (1 kanal)
    transforms.Resize((64, 32)),  # Modelin beklentilerine uygun boyuta yeniden boyutlandırın
    transforms.ToTensor(),  # Torch tensor'ına çevirin
])

input_image = transform(image).unsqueeze(0)  # Batch boyutunu ekleyin

# %%

use_gpu = True

with torch.no_grad():
    if use_gpu and torch.cuda.is_available():
        input_image = input_image.to(device)

    output = net(input_image)
    _, predicted_class = torch.max(output, 1)
    
# Tahmin edilen sınıfı yazdırın

print(f"Predicted Class: {predicted_class.item()}")

# Resmi görselleştirin
plt.imshow(image, cmap='gray')
plt.title(f"Predicted Class: {predicted_class.item()}")
plt.show()




















