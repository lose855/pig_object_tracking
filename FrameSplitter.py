import cv2
import os
class FrameSplitter:
    def __init__(self):
        self.outputPath = './images'
        self.outputName = '/image_%d'
        self.outputFormat = '.jpg'
        self.inputPath = './videos/'
        try: # make input, output images directory
            if not os.path.exists(self.outputPath):
                os.makedirs(self.outputPath)
            if not os.path.exists(self.inputPath):
                os.makedirs(self.inputPath)
        except OSError:
            print('Error: Creating directory. ' + self.outputPath)

    def load(self, dir): #
        dir = self.inputPath + dir
        video = cv2.VideoCapture(dir)
        if not video.isOpened():
            raise RuntimeError(" Could not open :", dir)
            exit()
        return video

    def run(self):
        inputList = os.listdir(self.inputPath) # read videos list
        if len(inputList) == 0: raise RuntimeError(" Input videos list is empty please check again")
        count = 0  # for image numbering
        for dir in inputList:
            video = self.load(dir)
            fps = video.get(cv2.CAP_PROP_FPS) # fps
            while (video.isOpened()):
                ret, image = video.read()
                if (int(video.get(1)) % round(fps) == 0):  # split using frame rate
                    cv2.imwrite(self.outputPath + self.outputName%(count) + self.outputFormat, image)
                    print('Saved image number :', count)
                    count += 1
            video.release()

main = FrameSplitter()
main.run()