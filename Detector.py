import torch
# Model
class Detector:
    def __init__(self):
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path='model.pt')
        print("Successful loading: Detector")
    def Detect(self, img):
        results = self.model(img)
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