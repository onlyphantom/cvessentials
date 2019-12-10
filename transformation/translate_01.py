import numpy as np
import cv2

img = cv2.imread("assets/cvess.png")
cv2.imshow("Original", img)
cv2.waitKey(0)
(h, w) = img.shape[:2]

# Specify our 2x3 matrix
mat = np.float32([[1, 0, -140], [0, 1, 20]])
translated = cv2.warpAffine(img, mat, (w, h))
cv2.imshow("Translated", translated)
cv2.waitKey(0)