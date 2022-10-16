import torch
import torch.nn as nn
from PIL import Image
from Layouts import Layout
import sys
import pygame
from pygame.locals import *
from torchvision import transforms
import os
import numpy as np
import time
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
        self.layout = Layout()
        self.type = type
        self.focusNow = -1 # default -1
        self.index = 0 # Image page
        self.framePerSec = pygame.time.Clock()
        self.inputPath = './'+type
        self.images = os.listdir(self.inputPath) # Load images
        self.totalImages = len(self.images)
        self.resultPath = './'+type+"/label"
        if not os.path.isdir(self.resultPath):
            os.mkdir(self.resultPath)
        self.run()

    class Coordinate:
        def __init__(self, pos, scaleRate):
            self.x = pos[0]
            self.y = pos[1]
            self.realX = pos[0] // scaleRate[1]
            self.realY = pos[1] // scaleRate[0]
        def transform(self):
            return (self.realX, self.realY)
        def pos(self):
            return (self.x, self.y)


    def run(self):
        # edit1 : isOverlapping True/False
        # edit2 : overlappingNumber Integer
        display = pygame.display.set_mode((self.layout.width, self.layout.height))
        pygame.display.set_caption(self.type)
        isOverlapVal = self.type
        overlapNumval = "0"
        edit1 = self.layout.font.render(isOverlapVal, True, self.layout.white)
        edit2 = self.layout.font.render(overlapNumval, True, self.layout.white)
        coordList = []

        while True:
            display.fill(self.layout.black)
            image = self.images[self.index]
            image = pygame.image.load(self.inputPath+'/'+image)
            imageWidth = image.get_width()
            imageHeight = image.get_height()
            image = pygame.transform.scale(image, (self.layout.resize, self.layout.resize))
            scaleRate = (self.layout.resize/imageHeight, self.layout.resize/imageWidth) # HxW
            display.blit(image, [30, 40])
            imageNumber = self.layout.font.render("Image: %d"%(self.index), True, self.layout.white)
        
            isOverlapping = self.layout.font.render("Is overlap:", True, self.layout.white)
            numOverlapping = self.layout.font.render("Num overlap:", True, self.layout.white)
            positions = self.layout.font.render("Positions:", True, self.layout.white)
            edit1 = self.layout.font.render(isOverlapVal, True, self.layout.white)
            edit2 = self.layout.font.render(overlapNumval, True, self.layout.white)

            display.blit(imageNumber, self.layout.label1)
            display.blit(isOverlapping, self.layout.label2)
            display.blit(numOverlapping, self.layout.label3)
            display.blit(positions, self.layout.label4)
            display.blit(edit1, self.layout.edit1)
            display.blit(edit2, self.layout.edit2)
            
            for event in pygame.event.get(): # Evnet handler
                if event.type == QUIT:
                    pygame.quit()
                    sys.exit()

                if event.type == pygame.KEYDOWN:
                    if event.key == K_BACKSPACE:
                        if self.focusNow == self.layout.edit1Idx:
                            if len(isOverlapVal) > 0:
                                isOverlapVal = isOverlapVal[:-1]
                        elif self.focusNow == self.layout.edit2Idx:
                            if len(overlapNumval) > 0:
                                overlapNumval = overlapNumval[:-1]
                    elif event.key == pygame.K_RIGHT:
                        self.index = (self.index+1) % self.totalImages
                    elif event.key == pygame.K_LEFT:
                        if not self.index == 0:
                            self.index -= 1
                    else:
                        if self.focusNow == self.layout.edit1Idx:
                            isOverlapVal += event.unicode
                        elif self.focusNow == self.layout.edit2Idx:
                            overlapNumval += event.unicode

                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == self.layout.leftButton:  # Left button
                        pos = event.pos
                        if self.layout.edit1Pos[0][0]<=pos[0]<=self.layout.edit1Pos[1][0] and self.layout.edit1Pos[0][1]<=pos[1]<=self.layout.edit1Pos[1][1]:
                            self.focusNow = self.layout.edit1Idx
                            cursor = self.layout.cursorEdit1
                        elif self.layout.edit2Pos[0][0]<=pos[0]<=self.layout.edit2Pos[1][0] and self.layout.edit2Pos[0][1]<=pos[1]<=self.layout.edit2Pos[1][1]:
                            self.focusNow = self.layout.edit2Idx
                            cursor = self.layout.cursorEdit2
                        else:
                            self.focusNow = -1
                        if 30<=pos[0]<=self.layout.resize+30 and 40<=pos[1]<=self.layout.resize+40:
                            coordList.append(self.Coordinate(pos, scaleRate))
                    if event.button == self.layout.centerButton: # Center button
                        self.index = (self.index+1) % self.totalImages
                        coordList = []
                        print(len(coordList))
                        pygame.time.delay(70)
                    if event.button == self.layout.backWheel: # Back wheel
                        if not self.index == 0:
                            self.index -= 1
                            coordList = []

            if time.time() % 1 > 0.5:
                if not self.focusNow == -1:
                    pygame.draw.line(display, self.layout.white, cursor[0], cursor[1])
            if not len(coordList) == 0:
                for coordinate in coordList:
                    pygame.draw.circle(display, self.layout.red, coordinate.pos(), 10, 3)
            pygame.display.flip() # Display update
            self.framePerSec.tick(self.layout.fps)




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
