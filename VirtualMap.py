import cv2
import matplotlib.pyplot as plt
class VirtualMap:
    class Map:
        def __init__(self, height, width):
            self.height = height
            self.width = width

        def load(self):
            pass

        def transform(self, images): # images are list contain cropped detection object
            for object in images:
                object = cv2.cvtColor(object, cv2.COLOR_BGR2GRAY)
                ret, object = cv2.threshold(object, 70, 255, cv2.THRESH_BINARY)
                plt.imshow(object, cmap='gray')
                plt.show()

    def __init__(self): # Automatic making virtual map information
        # self.realImage = cv2.VideoCapture(img)
        # self.map = self.makeMap()
        pass

    def makeMap(self):
        height = self.realImage.shape[0]
        width = self.realImage.shape[1]

vv = VirtualMap()
dd = vv.Map(640, 640)
img = [cv2.imread("img.jpg")]
dd.transform(img)