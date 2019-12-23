import numpy as np
import cv2

img = cv2.imread("assets/canal.png")

cv2.imshow("Original", img)
cv2.waitKey(0)

approx_gaussian = (
    np.array(
        [
            [-1, -1, -1, -1, -1],
            [-1, 2, 2, 2, -1],
            [-1, 2, 8, 2, -1],
            [-1, 2, 2, 2, -1],
            [-1, -1, -1, -1, -1],
        ]
    )
    / 8.0
)
sharpen_col = cv2.filter2D(img, -1, approx_gaussian)

cv2.imshow("Sharpen (approx. Gaussian)", sharpen_col)
cv2.waitKey(0)
cv2.waitKey(0)

