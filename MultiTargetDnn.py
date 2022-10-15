import torch
import torch.nn as nn
from PIL import Image
import sys
import pygame
from pygame.locals import *
from torchvision import transforms
import os
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

        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.encoder = torch.hub.load('pytorch/vision:v0.10.0', 'resnet152', pretrained=True) # Load Encoder
        self.encoder.to('cuda') # Use cuda
        self.self.drop = nn.Dropout(0.5)
        print("Successful loading: MultiTargetDnn")

    def MakeBatch(self):
        pass

    def Train(self):
        self.encoder.train()

class Annotator: # Make dataset by hand
    def __init__(self, type):
        print("Type : %s is selected"%(type))
        pygame.init()
        self.type = type
        self.fps = 30
        self.black = (0, 0, 0)
        self.white = (255, 255, 255)
        self.width = 1240
        self.height = 720
        self.resize = 640
        self.font = pygame.font.SysFont('arial', 20, True, True)
        self.index = 0 # Image page
        self.framePerSec = pygame.time.Clock()
        self.inputPath = './'+type
        self.images = os.listdir(self.inputPath) # Load images
        self.totalImages = len(self.images)
        self.resultPath = './'+type+"/label"
        if not os.path.isdir(self.resultPath):
            os.mkdir(self.resultPath)
        self.run()

    def run(self):
        display = pygame.display.set_mode((self.width, self.height))
        display.fill(self.black)
        pygame.display.set_caption(self.type)
        isOverlapping = self.font.render("Is overlapping: ", True, self.white)
        numOverlapping = self.font.render("Num overlapping: ", True, self.white)
        overlappingRect = isOverlapping.get_rect()
        overlappingRect.topleft = (20, 20)
        overlappingCursor = Rect(overlappingRect.topright, (3, overlappingRect.height))
        while True:
            display.fill(self.black)
            image = self.images[self.index]
            image = pygame.image.load(self.inputPath+'/'+image)
            imageWidth = image.get_width()
            imageHeight = image.get_height()
            image = pygame.transform.scale(image, (self.resize, self.resize))
            scaleRate = (self.resize/imageHeight, self.resize/imageWidth) # HxW
            display.blit(image, [50, 40])
            imageNumber = self.font.render("Image: %d"%(self.index), True, self.white)
            # coordinate = self.font.render("Coordinate-%d : X: %d Y: %d ", True, self.white)
            display.blit(imageNumber, [700, 40])
            display.blit(isOverlapping, [700, 80])

            display.blit(numOverlapping, [700, 120])
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == K_BACKSPACE:
                        if len(text) > 0:
                            text = text[:-1]
                    else:
                        text += event.unicode
                    if event.key == pygame.K_RIGHT:
                        self.index = (self.index+1) % self.totalImages
                    if event.key == pygame.K_LEFT:
                        if not self.index == 0:
                            self.index -= 1
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 2: # Center button
                        self.index = (self.index+1) % self.totalImages
                        pygame.time.delay(70)
                    if event.button == 5: # Back wheel
                        if not self.index == 0:
                            self.index -= 1

            pygame.display.flip() # Display update
            self.framePerSec.tick(self.fps)




# image = Image.open('img.jpg')
#
# testTensor = preprocess(image).to('cuda')
# testTensor = torch.unsqueeze(testTensor, 0)
# encoder = torch.hub.load('pytorch/vision:v0.10.0', 'resnet152', weights='IMAGENET1K_V2')  # Load Encoder
# encoder.to('cuda')  # Use cuda
# encoder.eval()
# result = encoder(testTensor)
# print(result[0].shape)

annotator = Annotator('overlapping')
