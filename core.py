from VirtualMap import VirtualMap
from Model import Model
import cv2

def run():
    model = Model()
    map = VirtualMap(1920, 1080)
    img = cv2.imread("img.jpg")
    crops = model.detect(img)
    positions = map.transform(crops)

    for position in positions:
        img = cv2.circle(img, position, 30, (0,0,255), 5)
    cv2.imshow("dd", img)
    cv2.waitKey(0)

run()