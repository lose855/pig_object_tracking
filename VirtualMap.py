import cv2
import sqlite3
import matplotlib.pyplot as plt


class VirtualMap:
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def transform(self, crops): # from bounding box to pig coordinate
        # box : x1, y1, x2, y2
        # im : image array
        position = []
        for crop in crops:
            x1, y1, x2, y2 = int(crop['box'][0].item()), int(crop['box'][1].item()), int(crop['box'][2].item()), int(crop['box'][3].item())
            frame = cv2.cvtColor(crop['im'], cv2.COLOR_BGR2RGB)
            frame, _, _ = cv2.split(frame)# using red channel
            frame = cv2.GaussianBlur(frame, (5, 5), 0)# remove noise
            ret, frame = cv2.threshold(frame, 80, 255, cv2.THRESH_BINARY)
            kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (9, 9))
            frame = cv2.morphologyEx(frame, cv2.MORPH_ERODE, kernel, iterations=2) # remove noise
            contours, _ = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # detect contours
            areaList = [cv2.contourArea(cnt) for cnt in contours]
            objectIndex = areaList.index(max(areaList))
            coordinate = contours[objectIndex]
            epsilon = 0.05 * cv2.arcLength(coordinate, True)
            coordinate = cv2.approxPolyDP(coordinate, epsilon, True)
            cv2.drawContours(frame, coordinate, -1, (0, 255, 0), 4)
            cv2.imshow("dd", frame)
            cv2.waitKey(0)
            moment = cv2.moments(coordinate)
            x = int(moment['m10'] / moment['m00']) + x1
            y = int(moment['m01'] / moment['m00']) + y1
            position.append((x,y))
        return position

# vv = VirtualMap()
# dd = vv.Map(640, 640)
# # img = [cv2.imread("img.jpg")]
# img = [cv2.imread("img.jpg"), cv2.imread("img2.jpg"), cv2.imread("img3.jpg"), cv2.imread("img4.jpg"), cv2.imread("img5.jpg"), cv2.imread("img6.jpg")]
# img.extend([cv2.imread("img7.jpg"), cv2.imread("img8.jpg"), cv2.imread("img9.jpg"), cv2.imread("img10.jpg"), cv2.imread("img11.jpg"), cv2.imread("img12.jpg")])
# dd.transform(img)
