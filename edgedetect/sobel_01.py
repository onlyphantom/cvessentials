import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("assets/sudoku.jpg", 0)
img = cv2.medianBlur(img, 5)
img = cv2.GaussianBlur(img, (7, 7), 0)
cv2.imshow("Image", img)
cv2.waitKey(0)

gradient_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
gradient_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
print(f"Range: {np.min(gradient_x)} | {np.max(gradient_x)}")

gradient_x = np.uint8(np.absolute(gradient_x))
gradient_y = np.uint8(np.absolute(gradient_y))
print(f"Range uint8: {np.min(gradient_x)} | {np.max(gradient_x)}")

cv2.imshow("Gradient X", gradient_x)
cv2.waitKey(0)
cv2.imshow("Gradient Y", gradient_y)
cv2.waitKey(0)

# plt.imshow(gradient_x, cmap="gray")
# plt.show()

