import cv2
import argparse
import re

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--path", default="assets/ocbc.jpg")
args = vars(parser.parse_args())

# test: dbs.jpg | ocbc.jpg
img_color = cv2.imread(args["path"])
img_color = cv2.resize(img_color, None, None, fx=0.5, fy=0.5)
img = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

blurred = cv2.GaussianBlur(img, (7, 7), 0)
blurred = cv2.bilateralFilter(blurred, 5, sigmaColor=50, sigmaSpace=50)
edged = cv2.Canny(blurred, 130, 150, 255)

cv2.imshow("Outline of device", edged)
cv2.waitKey(0)

cnts, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# sort contours by area, and get the largest
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:1]

cv2.drawContours(img_color, cnts, 0, (75, 0, 130), 4)
cv2.imshow("Target Contour", img_color)
cv2.waitKey(0)

x, y, w, h = cv2.boundingRect(cnts[0])
roi = img[y : y + h, x : x + w]
cv2.imshow("ROI", roi)

img_name = re.search("(?<=\/)(.*)(?=\.jpg)", args["path"]).group(1)

cv2.imwrite(f"inter/{img_name}-roi.png", roi)
cv2.waitKey(0)
