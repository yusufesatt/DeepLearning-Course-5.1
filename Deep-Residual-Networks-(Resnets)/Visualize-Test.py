# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 23:48:53 2023

@author: yusuf
"""

# Eğitilmiş modeli test etme

# %%

import torch
import torchvision.transforms as transforms
from IR_Pedestrian_Project import Net # Modelinizi içeren modülü ekleyin

# Modelinizi oluşturun
net = Net()

# Eğitilmiş ağırlıkları yükleyin
net.load_state_dict(torch.load("C:/Users/yusuf/Desktop/DeepLearning-Course-5.1/trained_model_80_epoch.pth"))

# Modeli değerlendirmeye alın
net.eval()

# %%

from PIL import Image

# Test etmek istediğiniz resmi yükleyin
image_path = r"C:/Users/yusuf/Desktop/DeepLearning-Course-5.1/Datasets/LSIFar/LSIFIR/Detection/Test/07/00001.png"
image = Image.open(image_path)

# Resmi modele uygun hale getirin
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Resmi gri tonlamaya dönüştürün (1 kanal)
    transforms.Resize((64, 32)),  # Modelin beklentilerine uygun boyuta yeniden boyutlandırın
    transforms.ToTensor()  # Torch tensor'ına çevirin
])

input_image = transform(image).unsqueeze(0)  # Batch boyutunu ekleyin

# %%

with torch.no_grad():
    if use_gpu and torch.cuda.is_available():
        input_image = input_image.to(device)

    output = net(input_image)
    _, predicted_class = torch.max(output, 1)

# Tahmin edilen sınıfı yazdırın
print(f"Predicted Class: {predicted_class.item()}")