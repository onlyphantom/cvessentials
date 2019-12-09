import numpy as np
import cv2

img = cv2.imread("assets/cvess.png")
cv2.imshow("Original", img)
cv2.waitKey(0)
(h, w) = img.shape[:2]

center = (w // 2, h // 2)
# getRotationMatrix2D creates our 2x3 matrix
mat = cv2.getRotationMatrix2D(center, angle=270, scale=1)
print(f'270 degree anti-clockwise: \n {np.round(mat, 2)}')
rotated = cv2.warpAffine(img, mat, (w, h))
cv2.imshow("Rotated", rotated)
cv2.waitKey(0)