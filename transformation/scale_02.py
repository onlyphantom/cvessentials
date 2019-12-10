import numpy as np
import cv2

img = cv2.imread("assets/corgi.png")
cv2.circle(img, (10, 10), 4, (255, 0, 0), -1)
cv2.circle(img, (80, 10), 4, (0, 255, 0), -1)
cv2.circle(img, (10, 80), 4, (0, 0, 255), -1)
cv2.imshow("Original", img)

# custom transformation matrix
mat = np.float32([[1, 0, 0], [0, 1, 0]])
print(mat)
result = cv2.warpAffine(img, M=mat, dsize=(200, 200))
cv2.imshow("Warped", result)

cv2.waitKey(0)
cv2.destroyAllWindows()
