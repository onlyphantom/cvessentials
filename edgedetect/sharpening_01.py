import numpy as np
import cv2

img = cv2.imread("assets/canal.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

for i in range(3):
    newval = np.round(np.mean(gray[:5, i : i + 5]))
    print(f"Mean of 25x25 pixel #{i+1}: {np.int(newval)}")

cv2.imshow("Gray", gray)
cv2.waitKey(0)

sharpen = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
sharpen_col = cv2.filter2D(img, -1, sharpen)
sharpen_gray = cv2.filter2D(gray, -1, sharpen)

cv2.imshow("Sharpen Colored", sharpen_col)
cv2.waitKey(0)

cv2.imshow("Sharpen Gray", sharpen_gray)
cv2.waitKey(0)

