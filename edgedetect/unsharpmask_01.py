import numpy as np
import cv2

KERNEL_SIZE = (5, 5)

img = cv2.imread("assets/sarpi.png")
cv2.imshow("Original", img)
cv2.waitKey(0)

amt = 1.5
blurred = cv2.GaussianBlur(img, (5,5), 10)
unsharp = cv2.addWeighted(img, 1+amt, blurred, -amt, 0)

cv2.imshow("Unsharp Masking", unsharp)
cv2.waitKey(0)


