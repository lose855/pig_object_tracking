import cv2
import pygame
class Visualizer:

    def __init__(self):
        print("Successful loading: Visualizer")

    def Show(self, positions):
        for position in positions:
            img = cv2.circle(img, position, 30, (0, 0, 255), 5)
        cv2.imshow("dd", img)
        cv2.waitKey(0)