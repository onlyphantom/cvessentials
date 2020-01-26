import cv2

PURPLE = (75, 0, 130)
YELLOW = (0, 255, 255)
THICKNESS = 4
FONT = cv2.FONT_HERSHEY_SIMPLEX

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
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]

for i, cnt in enumerate(cnts):
    cv2.drawContours(img_color, cnts, i, PURPLE, THICKNESS)
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(img_color, (x, y), (x + w, y + h), YELLOW, THICKNESS)
    area = round(cv2.contourArea(cnt), 1)
    peri = round(cv2.arcLength(cnt, closed=True), 1)
    print(f"ContourArea:{area}, Peri: {peri}")
    cv2.putText(img_color, "Area:" + str(area), (x, y - 15), FONT, 0.4, PURPLE, 1)
    cv2.putText(img_color, "Perimeter:" + str(peri), (x, y - 5), FONT, 0.4, PURPLE, 1)

cv2.imshow("Contours", img_color)
cv2.waitKey(0)
