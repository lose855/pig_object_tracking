import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import os
import cv2
import numpy as np
import random
import torch.optim as optim
import keyboard
import joblib
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter



class MultiTargetDnn:
    def __init__(self):
        seed = 17 # Random seed
        os.environ['CUDA_LAUNCH_BLOCKING'] = "1" #
        os.environ["CUDA_VISIBLE_DEVICES"] = "0" #
        random.seed(seed) #
        os.environ['PYTHONHASHSEED'] = str(seed) #
        np.random.seed(seed) #
        torch.manual_seed(seed) #
        torch.cuda.manual_seed(seed) #
        torch.backends.cudnn.deterministic = True #

        self.encoder = torch.hub.load('pytorch/vision:v0.10.0', 'resnet152', pretrained=True) # Load Encoder
        self.encoder.to('cuda') # Use cuda
        self.self.drop = nn.Dropout(0.5)
        print("Successful loading: MultiTargetDnn")

    def MakeBatch(self):
        pass

    def Train(self):
        self.encoder.train()


image = Image.open('img.jpg')
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
testTensor = preprocess(image)
encoder = torch.hub.load('pytorch/vision:v0.10.0', 'resnet152', pretrained=True)  # Load Encoder
encoder.to('cuda')  # Use cuda
encoder.eval()
result = encoder(testTensor)
print(result[0])