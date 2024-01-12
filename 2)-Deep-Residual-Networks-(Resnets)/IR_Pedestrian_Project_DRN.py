# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 15:00:10 2023

@author: yusuf
"""
# %%

import torch 
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os 
import torch.utils.data

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
    # 0'lardan oluşan 64 satır 32 sütunluk array oluşturuyorum
    array = np.zeros([num_img, 64*32])
    i = 0
    for img in os.listdir(path):
        img_path = path + "\\" + img
        img = Image.open(img_path, mode="r")
        data = np.asarray(img, dtype="uint8")
        data = data.flatten()
        array[i, :] = data
        i += 1
    return array

# %%
# Read train negative


train_negative_path = r"C:/Users/yusuf/Desktop/DeepLearning-Course-5.1/Datasets/LSIFar/LSIFIR/Classification/Train/neg"
num_train_negative_img = 43390
train_negative_array = read_images(train_negative_path, num_train_negative_img)

# %%
# Torch'da tensorlar kullanılıyor
# Convert tensor
x_train_negative_tensor = torch.from_numpy(train_negative_array[:42000,:])
print("x_train_negative_tensor:", x_train_negative_tensor.size())

# %%

y_train_negative_tensor = torch.zeros(42000, dtype=torch.long)
print("y_train_negative_tensor:", y_train_negative_tensor.size())

# %%
# Read train positive

train_positive_path = r"C:/Users/yusuf/Desktop/DeepLearning-Course-5.1/Datasets/LSIFar/LSIFIR/Classification/Train/pos"
num_train_positive_img = 10208
train_positive_array = read_images(train_positive_path, num_train_positive_img)

# %%
# Convert tensor
x_train_positive_tensor = torch.from_numpy(train_positive_array[:10000,:])
print("x_train_positive_tensor:", x_train_positive_tensor.size())

y_train_positive_tensor = torch.ones(10000, dtype=torch.long)
print("y_train_positive_tensor:", y_train_positive_tensor.size())

# %%
# Concat train

# yukardan aşağı yapmasını istediğim için 0 seçiyorum 1 yapsaydım sütunları birleştirirdi.
x_train = torch.cat((x_train_negative_tensor, x_train_positive_tensor), 0)
y_train = torch.cat((y_train_negative_tensor, y_train_positive_tensor), 0)
print("x_train: ", x_train.size())
print("y_train: ", y_train.size())

# %%
# -----------------------------------------------
# Test data
# read test negative  22050
test_negative_path = r"C:/Users/yusuf/Desktop/DeepLearning-Course-5.1/Datasets/LSIFar/LSIFIR/Classification/Test/neg"
num_test_negative_img = 22050
test_negative_array = read_images(test_negative_path, num_test_negative_img)
x_test_negative_tensor = torch.from_numpy(test_negative_array[:18056, :])
print("x_test_negative_tensor: ", x_test_negative_tensor.size())
y_test_negative_tensor = torch.zeros(18056, dtype=torch.long)
print("y_test_negative_tensor: ", y_test_negative_tensor.size())

# read test positive 5944
test_positive_path = r"C:/Users/yusuf/Desktop/DeepLearning-Course-5.1/Datasets/LSIFar/LSIFIR/Classification/Test/pos"
num_test_positive_img = 5944
test_positive_array = read_images(test_positive_path, num_test_positive_img)
x_test_positive_tensor = torch.from_numpy(test_positive_array)
print("x_test_positive_tensor: ", x_test_positive_tensor.size())
y_test_positive_tensor = torch.zeros(num_test_positive_img, dtype=torch.long)
print("y_test_positive_tensor: ", y_test_positive_tensor.size())

# concat test
x_test = torch.cat((x_test_negative_tensor, x_test_positive_tensor), 0)
y_test = torch.cat((y_test_negative_tensor, y_test_positive_tensor), 0)
print("x_test: ", x_test.size())
print("y_test: ", y_test.size())

# %%
# Visualize
plt.imshow(x_train[45001, :].reshape(64, 32), cmap="gray")

# %%

# Hyper Parameters

num_epochs = 100
num_classes = 2
batch_size = 2000
learning_rate = 0.0001

train = torch.utils.data.TensorDataset(x_train, y_train)
trainloader = torch.utils.data.DataLoader(
    train, batch_size=batch_size, shuffle=True)

test = torch.utils.data.TensorDataset(x_test, y_test)
testloader = torch.utils.data.DataLoader(
    test, batch_size=batch_size, shuffle=False)

# %%


def conv3x3(in_planes, out_planes, stride = 1):
    return nn.Conv2d(in_planes, out_planes, kernel_size = 3, stride = stride, padding = 1, bias = False)
    # 3x3 olduğu için kernelsize'da 3, padding yapma amacımız inputtaki boyutunu koruması için, batch norm yapacağımız için bias'a gerek yok


def conv1x1(in_planes, out_planes, stride = 1):
    return nn.Conv2d(in_planes, out_planes, kernel_size = 1, stride = stride, bias = False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()  # basic blocku initialize eder

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(0.9)  # %90
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        """
        Forward kısmında init kısmında yazdıklarımızı
        birleştiriyoruz
        """

        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.drop(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.drop(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.relu(out)

        return out

        """
        X inputumuzu ilk olarak conv1 bloğuna girdi daha sonra batch normalization,
        sonra relu, sonra dropout, daha sonra conv2, sonra bn2, sonra dropout, ve sonra tekrar reluya girdi
        daha sonra output olarak return eder.

        """

# %%


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=num_classes):
        super(ResNet, self).__init__()

        self.inplanes = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # layers parametresinin içinde kaç layer olacağı yazacak
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256*block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(
                    m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        """


        Parameters
        ----------
        block : TYPE
            Basic block.
        planes : TYPE
            Input channels.
        blocks : TYPE
            Basic blocklarımdan kaç tane inşaa edeceği
        stride : TYPE, optional
            The default is 1.

        Returns
        -------
        x : TYPE
            DESCRIPTION.

        """

        downsample = None

        if stride != 1 or self.inplanes != planes*block.expansion:  # bunlar input channellara eşit değilse
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes*block.expansion, stride),
                nn.BatchNorm2d(planes*block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))

        # out:1 blockumuzun expansionu her zaman 1
        self.inplanes = planes*block.expansion
        # başlangıçta zaten 1 block eklediğimiz için block eklemeye başlarken range'i 1den başlatacağız

        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

# %%


model = ResNet(BasicBlock, [2, 2, 2]).to(device)

# %%

# Loss and optimizer

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# %%
# Train

loss_list = []
train_acc = []
test_acc = []
use_gpu = True

total_step = len(trainloader)

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(trainloader):
        
        images = images.view(batch_size,1,64,32)
        images = images.float()
        
        # gpu
        if use_gpu:
            if torch.cuda.is_available():
                images, labels = images.to(device), labels.to(device)
            
        outputs = model(images)
        
        loss = criterion(outputs, labels)
        
        # backward and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if i % 2 == 0:
            print("epoch: {} {}/{}".format(epoch,i,total_step))

    # train
    correct = 0
    total = 0
    with torch.no_grad():
        for data in trainloader:
            images, labels = data
            images = images.view(batch_size,1,64,32)
            images = images.float()
            
            # gpu
            if use_gpu:
                if torch.cuda.is_available():
                    images, labels = images.to(device), labels.to(device)
                    
            outputs = model(images)
            _, predicted = torch.max(outputs.data,1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print("Accuracy train %d %%"%(100*correct/total))
    train_acc.append(100*correct/total)

    # test
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.view(batch_size,1,64,32)
            images = images.float()
            
            # gpu
            if use_gpu:
                if torch.cuda.is_available():
                    images, labels = images.to(device), labels.to(device)
                    
            outputs = model(images)
            _, predicted = torch.max(outputs.data,1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print("Accuracy test %d %%"%(100*correct/total))
    train_acc.append(100*correct/total)

    loss_list.append(loss.item())