import numpy as np
import cv2

img = cv2.imread("assets/corgi.png")
cv2.circle(img, (10, 10), 4, (0, 255, 255), -1)
cv2.circle(img, (80, 10), 4, (0, 255, 255), -1)
cv2.circle(img, (10, 80), 4, (0, 255, 255), -1)
cv2.imshow("Original", img)

coords_s = np.float32([[10, 10], [80, 10], [10, 80]])
coords_d = np.float32([[10, 10], [80, 10], [10, 80]])

# getAffineTransform creates our 2x3 matrix
mat = cv2.getAffineTransform(src=coords_s, dst=coords_d)
print(mat)
result = cv2.warpAffine(img, M=mat, dsize=(200, 200))
cv2.imshow("Warped", result)

cv2.waitKey(0)
cv2.destroyAllWindows()
