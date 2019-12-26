import numpy as np
import cv2

KERNEL_SIZE = (5, 5)

img = cv2.imread("assets/sarpi.png")
cv2.imshow("Original", img)
cv2.waitKey(0)

amt = 1.5
blurred = cv2.GaussianBlur(img, (5,5), 10)

unsharp_manual = np.clip(img * (1+amt) + blurred * (-amt), 0, 255)
# unsharp_manual = img * (1+amt) + blurred * (-amt)
# unsharp_manual = np.maximum(unsharp_manual, np.zeros(unsharp_manual.shape))
# unsharp_manual = np.minimum(unsharp_manual, 255 * np.ones(unsharp_manual.shape))
unsharp_manual = unsharp_manual.round().astype(np.uint8)

cv2.imshow("Unsharp Masking Manual", unsharp_manual)
cv2.waitKey(0)

