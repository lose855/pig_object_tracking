import torch
# Model
class Model:
    def __init__(self):
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path='model.pt')
    def detect(self, img):
        results = self.model(img)
        crops = results.crop(save=False)
        results.show()
        exit()
        return crops
        # box : x1, y1, x2, y2
        # im : image array