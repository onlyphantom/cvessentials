import numpy as np
import cv2

KERNEL_SIZE = (5, 5)

img = cv2.imread("assets/canal.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow("Gray", gray)
cv2.waitKey(0)

# Create the following 5x5 
# np.array(
# [[0.04, 0.04, 0.04, 0.04, 0.04],
# [0.04, 0.04, 0.04, 0.04, 0.04],
# [0.04, 0.04, 0.04, 0.04, 0.04],
# [0.04, 0.04, 0.04, 0.04, 0.04],
# [0.04, 0.04, 0.04, 0.04, 0.04]])

mean_blur = np.ones(KERNEL_SIZE, dtype="float32") * (1.0 / (5 ** 2))
smoothed_col = cv2.filter2D(img, -1, mean_blur)
smoothed_gray = cv2.filter2D(gray, -1, mean_blur)

cv2.imshow("Smoothed Colored", smoothed_col)
cv2.waitKey(0)

cv2.imshow("Smoothed Gray", smoothed_gray)
cv2.waitKey(0)
