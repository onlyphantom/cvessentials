import numpy as np
import cv2

KERNEL_SIZE = (5, 5)

img = cv2.imread("assets/canal.png")

meanblurred = cv2.blur(img, KERNEL_SIZE)
gaussianblurred = cv2.GaussianBlur(src=img, ksize=KERNEL_SIZE, sigmaX=0)

cv2.imshow("Mean Blurred", meanblurred)
cv2.waitKey(0)
cv2.imshow("Gaussian Blurred", gaussianblurred)
cv2.waitKey(0)

