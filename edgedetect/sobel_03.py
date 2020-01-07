import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("assets/castello.png", flags=0)
img = cv2.medianBlur(img, 9)
img = cv2.GaussianBlur(img, (9, 9), 0)

gradient_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
gradient_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

gradient_x = cv2.convertScaleAbs(gradient_x)
gradient_y = cv2.convertScaleAbs(gradient_y)
print(f"Range: {np.min(gradient_x)} | {np.max(gradient_x)}")

gradient_xy = cv2.addWeighted(gradient_x, 0.5, gradient_y, 0.5, 0)

plt.imshow(gradient_xy, cmap="gray")
plt.title("Sobel Edge")
plt.show()