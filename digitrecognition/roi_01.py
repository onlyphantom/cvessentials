import cv2
BCOLOR = (75, 0, 130)
THICKNESS = 4

img_color = cv2.imread("assets/ocbc.jpg")
img_color = cv2.resize(img_color, None, None, fx=0.5, fy=0.5)
img = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

x,y,w,h = cv2.selectROI("Region of interest", img)
print(x,y,w,h)

cropped = img[y:y+h, x:x+w]
cv2.imshow("Cropped", cropped)
cv2.waitKey(0)

cv2.rectangle(img_color, (x,y), (x+w,y+h), (255,0,0), 2)
cv2.imshow("Original Image", img_color)
cv2.waitKey(0)
