import cv2

BCOLOR = (75, 0, 130)
THICKNESS = 4

img_color = cv2.imread("assets/ocbc.jpg")
img_color = cv2.resize(img_color, None, None, fx=0.5, fy=0.5)
img = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

blurred = cv2.GaussianBlur(img, (7, 7), 0)
blurred = cv2.bilateralFilter(blurred, 5, sigmaColor=50, sigmaSpace=50)
edged = cv2.Canny(blurred, 130, 150, 255)

cv2.imshow("Outline of device", edged)
cv2.waitKey(0)

cnts, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# sort contours by area, and get the first 10
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:9]

cv2.drawContours(img_color, cnts, 0, BCOLOR, THICKNESS)
cv2.imshow("Target Contour", img_color)
cv2.waitKey(0)

for i, cnt in enumerate(cnts):
    cv2.drawContours(img_color, cnts, i, BCOLOR, THICKNESS)
    print(f"ContourArea:{cv2.contourArea(cnt)}")
    cv2.imshow("Contour one by one", img_color)
    cv2.waitKey(0)
