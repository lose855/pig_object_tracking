import os
import sys
import pygame
import pygame.gfxdraw
from pygame.locals import *
from PIL import Image
from Layouts import Layout
import cv2
import datetime
from detectors import RCF
import numpy as np

class Annotator:  # Make dataset by hand
    def __init__(self, type, weight):
        print("Type : %s is selected" % (type))
        pygame.init()
        self.inputPath = './' + type
        self.resultPath = './' + type
        # self.resultPath = './' + 'binarys'
        if not os.path.isdir(self.resultPath):
            os.mkdir(self.resultPath)
        self.layout = Layout()
        self.type = type
        self.images = os.listdir(self.inputPath)  # Load images
        self.memory = [] # Save change history
        self.dump = []
        self.isFirst = True
        self.click = False
        self.thick = 5
        self.index = 0  # Image page
        self.weight = weight
        self.totalImages = len(self.images)
        self.framePerSec = pygame.time.Clock()
        self.edgeDetect = RCF(device='cuda')
        # self.Run()

    def Clear(self):
        self.memory = []
        self.dump = []
        self.isFirst = True
        self.click = False

    def Tobinary(self, img):
        start_time = datetime.datetime.now()
        img = cv2.resize(img, (self.layout.resize, self.layout.resize), interpolation=cv2.INTER_LANCZOS4)
        img = cv2.GaussianBlur(img, (0, 0), 1)
        # dx = np.abs(cv2.Sobel(img, cv2.CV_64F, 1, 0, 3))
        # dy = np.abs(cv2.Sobel(img, cv2.CV_64F, 0, 1, 3))
        # sobel = cv2.magnitude(dx, dy)
        # sobel = np.clip(sobel, 0, 255).astype(np.uint8)
        edge = self.edgeDetect.detect_edge(img)
        ret, result = cv2.threshold(edge, 90, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3, 3), np.uint8)
        result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel, iterations=1)
        end_time = datetime.datetime.now()
        time_delta = end_time - start_time
        print('\rExecution : {} seconds'.format(time_delta.microseconds/10**6), end='')
        return result

    def Run(self):
        x, y = 0, 1
        display = pygame.display.set_mode((self.layout.resize, self.layout.resize))
        pygame.display.set_caption(self.type)
        while True:
            if self.isFirst == True:
                image = self.images[self.index]
                pygame.display.set_caption(image)
                src = cv2.imread(self.inputPath + '/' + image)
                src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
                image = pygame.surfarray.make_surface(src)
                image = pygame.transform.scale(image, (self.layout.resize, self.layout.resize))
                image.set_alpha(128)
                result = self.Tobinary(src)
                result = pygame.surfarray.make_surface(result)
                display.fill(self.layout.black)
                display.blit(result, [self.layout.imgIdx[x], self.layout.imgIdx[y]])
                display.blit(image, [self.layout.imgIdx[x], self.layout.imgIdx[y]])
                pygame.display.flip()
                print("\n Image: %d"%(self.index), end='\n')
                self.isFirst = False

            if pygame.mouse.get_pressed() == (0, 0, 1):
                pos = pygame.mouse.get_pos()
                xP = pos[x]
                yP = pos[y]
                self.memory.append(result.copy())  # History record
                pygame.gfxdraw.filled_circle(result, xP, yP, self.thick, self.layout.black)
                display.fill(self.layout.black)
                display.blit(result, [self.layout.imgIdx[x], self.layout.imgIdx[y]])
                display.blit(image, [self.layout.imgIdx[x], self.layout.imgIdx[y]])
                pygame.time.delay(50)

            elif pygame.key.get_pressed()[K_BACKSPACE]:
                if not len(self.memory) == 0:
                    result = self.memory[-1]
                    self.memory = self.memory[:-1]
                    display.fill(self.layout.white)
                    display.blit(result, [self.layout.imgIdx[x], self.layout.imgIdx[y]])
                    display.blit(image, [self.layout.imgIdx[x], self.layout.imgIdx[y]])
                    pygame.time.delay(100)

            for event in pygame.event.get():  # Event handler
                if event.type == QUIT:
                    pygame.quit()
                    sys.exit()

                elif event.type == pygame.KEYDOWN:
                    if event.key == K_BACKSPACE:
                        if not len(self.memory) == 0:
                            result = self.memory[-1]
                            self.memory = self.memory[:-1]
                            display.fill(self.layout.white)
                            display.blit(result, [self.layout.imgIdx[x], self.layout.imgIdx[y]])
                            display.blit(image, [self.layout.imgIdx[x], self.layout.imgIdx[y]])

                    if event.key == K_RETURN: # Go preview
                        if not self.index == 0:
                            self.index -= 1
                            self.clear()

                elif event.type == pygame.MOUSEBUTTONDOWN:

                    if event.button == self.layout.leftButton:  # Left button
                        pos = event.pos
                        if not self.click:
                            xP = pos[x]
                            yP = pos[y]
                            self.dump.append((xP, yP))
                            self.click = True
                        else:
                            xP = pos[x]
                            yP = pos[y]
                            startPos = self.dump[-1]
                            endPos = (xP, yP)
                            self.memory.append(result.copy()) # History record
                            if result.get_at(startPos) == (255, 255, 255, 255):
                                color = self.layout.white
                            else:
                                color = self.layout.black
                            pygame.draw.line(result, color, startPos, endPos, width=self.thick)
                            display.fill(self.layout.white)
                            display.blit(result, [self.layout.imgIdx[x], self.layout.imgIdx[y]])
                            display.blit(image, [self.layout.imgIdx[x], self.layout.imgIdx[y]])
                            self.dump = []
                            self.click = False

                    elif event.button == self.layout.rightButton:  # Right button
                        pos = event.pos
                        xP = pos[x] - self.layout.resize
                        yP = pos[y]
                        self.memory.append(result.copy()) # History record
                        pygame.gfxdraw.filled_circle(result, xP, yP, self.thick, self.layout.black)
                        display.fill(self.layout.black)
                        display.blit(result, [self.layout.imgIdx[x], self.layout.imgIdx[y]])
                        display.blit(image, [self.layout.imgIdx[x], self.layout.imgIdx[y]])

                    elif event.button == self.layout.centerButton:  # Center button
                        if not len(self.memory) == 0:
                            resultName = "/%d.%s"
                            image.set_alpha(0)
                            pygame.image.save(image, self.resultPath + '/' + resultName%(self.index+self.weight, 'jpg'))
                            pygame.image.save(result, self.resultPath + '/'+ resultName%(self.index+self.weight, 'png'))
                            print('\nSaved image')
                        self.index = (self.index + 1) % self.totalImages
                        self.Clear()

                    elif event.button == self.layout.upWheel:  # Back wheel
                        if 5 <= self.thick < 20:
                            self.thick += 1
                            print('\rNow Line Thickness: %d' % (self.thick), end='')

                    elif event.button == self.layout.backWheel:  # Back wheel
                        if 6 <= self.thick <= 20:
                            self.thick -= 1
                            print('\rNow Line Thickness: %d' % (self.thick), end='')

            pygame.display.flip()  # Display update
            self.framePerSec.tick(self.layout.fps)

    def Cleaner(self, paths):
        for path in paths:
            lisImage = os.listdir(path)
            for name in lisImage:
                image = cv2.imread(path+'/'+name)
                kernel = np.ones((5, 5), np.uint8)
                result = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=1)
                result = Image.fromarray(result)
                result = result.convert("RGB")
                result.save(path+'/'+name)

a = Annotator("binarys_test", 100)
a.Run()


