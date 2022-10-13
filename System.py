from FrameSplitter import FrameSplitter
from MovingBox import MovingBox
from Detector import Detector
from Visualizer import Visualizer
import cv2

class Core:
    def __init__(self):
        print("Successful loading: System")
        self.detector = Detector()
        self.mBox = MovingBox()
        self.gui = Visualizer()

    def Run(self):
        img = cv2.imread("img.jpg") # test img
        positions = self.detector.Detect(img)



if __name__ == '__main__':
    core = Core()
    core.Run()