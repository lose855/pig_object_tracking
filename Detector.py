import torch
import cv2
import os
# Model
class Detector:
    def __init__(self):
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path='model.pt')
        self.imagePath = './images'
        self.overlappingPath = './overlapping'
        self.notoverlappingPath = './notoverlapping'
        print("Successful loading: Detector")
    def Detect(self, images):
        results = self.model(images)
        crops = results.crop(save=False)
        # box : x1, y1, x2, y2
        # im : image array
        position = []
        for crop in crops:
            x1, y1, x2, y2 = int(crop['box'][0].item()), int(crop['box'][1].item()), int(
            crop['box'][2].item()), int(crop['box'][3].item())
            xm = int((x2 - x1) // 2) + x1
            ym = int((y2 - y1) // 2) + y1
            position.append((xm, ym))
        return position

    def FindOverlapping(self):
        self.model.eval() # Use test mode
        countOverlap = 0
        count = 0
        inputList = os.listdir(self.imagePath)  # read videos list
        totalImages = len(inputList)
        print("Detected: %d images"%(totalImages))
        for page, image in enumerate(inputList):
            print("Left %d images"%(totalImages-page))
            image = cv2.imread(self.imagePath+'/'+image)
            results = self.model(image)
            crops = results.crop(save=False)
            # box : x1, y1, x2, y2
            # im : image array
            for index, mCrop in enumerate(crops):
                mx1, my1, mx2, my2 = int(mCrop['box'][0].item()), int(mCrop['box'][1].item()),\
                                 int(mCrop['box'][2].item()), int(mCrop['box'][3].item()) # Main box coordinate
                for oCrop in crops[index+1:]:
                    ox1, oy1, ox2, oy2 = int(oCrop['box'][0].item()), int(oCrop['box'][1].item()), \
                                     int(oCrop['box'][2].item()), int(oCrop['box'][3].item()) # Other box coordinate
                    intersectionX = min(mx2, ox2) - max(mx1, ox1)
                    intersectionY = min(my2, oy2) - max(my1, oy1)
                    if intersectionX > 0.3 and intersectionY > 0.3:
                        cv2.imwrite(self.overlappingPath+'/img_%d.jpg' % (countOverlap), mCrop['im'])
                        countOverlap += 1
                        continue
                    elif intersectionX < 0 and intersectionY < 0:
                        cv2.imwrite(self.notoverlappingPath + '/img_%d.jpg' % (count), mCrop['im'])
                        count += 1

detector = Detector()
detector.FindOverlapping()