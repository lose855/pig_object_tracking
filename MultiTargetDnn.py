import torch
import torch.nn as nn
from torchvision.models import *
from PIL import Image
import cv2
import json
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
        self.coordList = []
        self.framePerSec = pygame.time.Clock()
        self.inputPath = './'+type
        self.images = os.listdir(self.inputPath) # Load images
        self.totalImages = len(self.images)
        self.resultPath = './'+type+"/label"
        if not os.path.isdir(self.resultPath):
            os.mkdir(self.resultPath)
        self.run()

    def run(self):
        display = pygame.display.set_mode((self.layout.width, self.layout.height))
        pygame.display.set_caption(self.type)
        if self.type == 'overlapping':
            isOverlapVal = 'True'
        elif self.type == 'notoverlapping':
            isOverlapVal = 'False'
        overlapNumval = "6"

        while True:
            display.fill(self.layout.black)
            image = self.images[self.index]
            image = pygame.image.load(self.inputPath+'/'+image)
            image = pygame.transform.scale(image, (self.layout.resize, self.layout.resize))
            display.blit(image, [30, 40])
            imageNumber = self.layout.font.render("Image: %d"%(self.index), True, self.layout.white)
            isOverlapping = self.layout.font.render("Is overlap:", True, self.layout.white)
            numOverlapping = self.layout.font.render("Num overlap:", True, self.layout.white) # Stop using
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
                        else:
                            if not len(self.coordList) == 0:
                                self.coordList = self.coordList[:-1]
                    elif event.key == pygame.K_RIGHT:
                        self.index = (self.index+1) % self.totalImages
                    elif event.key == pygame.K_LEFT:
                        if not self.index == 0:
                            self.index -= 1
                    elif event.key == pygame.K_DELETE:
                        file = self.inputPath+'/'+self.images[self.index]
                        os.remove(file)
                        del self.images[self.index]
                        self.totalImages = len(self.images)
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
                            if not overlapNumval == '':
                                if len(self.coordList) < int(overlapNumval):
                                    self.coordList.append(pos)
                    if event.button == self.layout.centerButton: # Center button
                        resultName = "/%d.txt"%(self.index)
                        if len(self.coordList) == 0:
                            self.coordList = [(0,0)]
                        result = {
                            'image' : self.images[self.index],
                            'Isoverlap' : bool(isOverlapVal), # True False
                            'Count' : int(overlapNumval),
                            'Pos' : self.coordList,
                        }
                        with open(self.resultPath+resultName, 'w') as file:
                            json.dump(result, file)
                        self.index = (self.index+1) % self.totalImages
                        pygame.time.delay(70)
                    if event.button == self.layout.backWheel: # Back wheel
                        if not self.index == 0:
                            self.index -= 1

            if time.time() % 1 > 0.5:
                if not self.focusNow == -1:
                    pygame.draw.line(display, self.layout.white, cursor[0], cursor[1])
            if not len(self.coordList) == 0:
                standPos = self.layout.standPos
                for idx, coordinate in enumerate(self.coordList):
                    pygame.draw.circle(display, self.layout.red, coordinate, 10, 3)
                    coordinateLabel = "Nubmer: %d Pos: %d %d"%(idx, coordinate[0], coordinate[1])
                    label = self.layout.fontSmall.render(coordinateLabel, True, self.layout.white)
                    display.blit(label, (standPos[0], standPos[1]+(40*idx)))

            pygame.display.flip() # Display update
            self.framePerSec.tick(self.layout.fps)




# image = Image.open('img.jpg').convert('RGB')
# preprocess = transforms.Compose([
#             transforms.Resize(672),
#             transforms.CenterCrop(640),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#         ])
# 
# testTensor = preprocess(image).to('cuda')
# testTensor = torch.unsqueeze(testTensor, 0)
# encoder = resnet152()  # Load Encoder
# encoder.to('cuda')  # Use cuda
# encoder.eval()
# result = encoder(testTensor)
# print(result[0].shape)

annotator = Annotator('overlapping')
