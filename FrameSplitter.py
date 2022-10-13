import cv2
import os
import threading
class FrameSplitter:
    def __init__(self):
        self.outputPath = './images'
        self.outputName = '/%d_image_%d' # video number + count number
        self.outputFormat = '.jpg'
        self.inputPath = './videos/'
        self.captureSize = 640
        self.operations = []
        try: # make input, output images directory
            if not os.path.exists(self.outputPath):
                os.makedirs(self.outputPath)
            if not os.path.exists(self.inputPath):
                os.makedirs(self.inputPath)
        except OSError:
            print('Error: Creating directory. ' + self.outputPath)

    def Load(self, dir): #
        dir = self.inputPath + dir
        video = cv2.VideoCapture(dir)
        if not video.isOpened():
            raise RuntimeError(" Could not open :", dir)
            exit()
        return video

    def Splitter(self, index, fileName, video): # frame splitter
        count = 13 # for image numbering
        fps = video.get(cv2.CAP_PROP_FPS)  # get fps
        while True:
            ret, image = video.read()
            if not ret:
                self.operations.append(1)
                break
            if (int(video.get(1)) % round(fps) == 0):  # split using frame rate
                image = cv2.resize(image, (640, 640))  # reducing image size (1920, 1080) -> (640, 640)
                cv2.imwrite(self.outputPath + self.outputName % (index, count) + self.outputFormat, image)
                print('Video : %s saved image number : %d\n' % (fileName, count))
                count += 1
        video.release()

    def Run(self):
        inputList = os.listdir(self.inputPath) # read videos list
        totalVideos = len(inputList)
        print("Detected %d videos"%(totalVideos))
        if len(inputList) == 0: raise RuntimeError(" Input videos list is empty please check again")
        for idx, dir in enumerate(inputList):
            video = self.Load(dir)
            worker = threading.Thread(target=self.Splitter, args=(idx, dir, video))
            worker.daemon = True
            worker.start()
        while True: # worker checking
            if len(self.operations) == 13: break # if all worker are finished
        print('Videos: %d was finished'%(totalVideos))