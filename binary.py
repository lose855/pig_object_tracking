import os
import sys
import pygame
import pygame.gfxdraw
from pygame.locals import *
from Layouts import Layout
import cv2
import time
import numpy as np


class Annotator:  # Make dataset by hand
    def __init__(self, type):
        print("Type : %s is selected" % (type))
        pygame.init()
        self.layout = Layout()
        self.type = type
        self.index = 0  # Image page
        self.memory = [] # Save change history
        self.dump = []
        self.isFirst = True
        self.click = False
        self.framePerSec = pygame.time.Clock()
        self.inputPath = './' + type
        self.images = os.listdir(self.inputPath)  # Load images
        self.totalImages = len(self.images)
        self.resultPath = './' + 'binarys'
        if not os.path.isdir(self.resultPath):
            os.mkdir(self.resultPath)
        self.run()

    def clear(self):
        self.memory = []
        self.dump = []
        self.isFirst = True
        self.click = False

    def toBinary(self, img):
        img = cv2.resize(img, (self.layout.resize, self.layout.resize), interpolation=cv2.INTER_LANCZOS4)
        img = cv2.GaussianBlur(img, (0, 0), 3)
        dx = np.abs(cv2.Sobel(img, cv2.CV_64F, 1, 0, 3))
        dy = np.abs(cv2.Sobel(img, cv2.CV_64F, 0, 1, 3))
        sobel = cv2.magnitude(dx, dy)
        sobel = np.clip(sobel, 0, 255).astype(np.uint8)
        blue, _, _ = cv2.split(sobel)  # Yolo image is blue
        ret, result = cv2.threshold(blue, 80, 255, cv2.THRESH_BINARY)
        kernel = np.ones((11, 11), np.uint8)
        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel, iterations=1)
        return result

    def run(self):
        x, y = 0, 1
        display = pygame.display.set_mode((self.layout.width, self.layout.height))
        pygame.display.set_caption(self.type)
        display.fill(self.layout.white)
        while True:
            if self.isFirst == True:
                image = self.images[self.index]
                src = cv2.imread(self.inputPath + '/' + image)
                image = pygame.surfarray.make_surface(src)
                image = pygame.transform.scale(image, (self.layout.resize, self.layout.resize))
                result = self.toBinary(src)
                result = pygame.surfarray.make_surface(result)
                display.blit(image, [self.layout.img1Idx[x], self.layout.img1Idx[y]])
                display.blit(result, [self.layout.img2Idx[x], self.layout.img2Idx[y]])
                self.isFirst = False

            for event in pygame.event.get():  # Event handler
                if event.type == QUIT:
                    pygame.quit()
                    sys.exit()

                elif event.type == pygame.KEYDOWN:
                    if event.key == K_BACKSPACE:
                        if len(self.memory) > 0:
                            result = self.memory[-1]
                            self.memory = self.memory[:-1]
                            display.fill(self.layout.white)
                            display.blit(image, [self.layout.img1Idx[x], self.layout.img1Idx[y]])
                            display.blit(result, [self.layout.img2Idx[x], self.layout.img2Idx[y]])

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == self.layout.leftButton:  # Left button
                        pos = event.pos
                        if self.layout.img2Idx[x] <= pos[x] <= self.layout.resize + self.layout.img2Idx[x] and self.layout.img2Idx[y] <= pos[y] <= self.layout.resize + self.layout.img2Idx[y]:
                            if not self.click:
                                self.click = True
                                xP = pos[x] - self.layout.resize
                                yP = pos[y]
                                self.dump.append((xP, yP))
                            else:
                                xP = pos[x] - self.layout.resize
                                yP = pos[y]
                                startPos = self.dump[-1]
                                endPos = (xP, yP)
                                self.memory.append(result)
                                if 255 in result.get_at((startPos[y], startPos[x])):
                                    color = self.layout.white
                                else:
                                    color = self.layout.black
                                pygame.gfxdraw.line(result, startPos[x], startPos[y], endPos[x], endPos[y], color)
                                self.click = False
                                self.dump = []

                    elif event.button == self.layout.centerButton:  # Center button
                        if not len(self.memory) == 0:
                            resultName = "/%d.png" % (self.index)
                            cv2.imwrite(self.inputPath + '/' + self.images[self.index], image)
                            pygame.image.save(result, self.resultPath + resultName)
                        self.index = (self.index + 1) % self.totalImages
                        self.clear()

                    elif event.button == self.layout.backWheel:  # Back wheel
                        if not self.index == 0:
                            self.index -= 1
                            self.clear()
            pygame.display.update()  # Display update
            self.framePerSec.tick(self.layout.fps)


a = Annotator("overlapping")
a.run()
