import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread("assets/sudoku.jpg", flags=0)
_, img_threshold = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)

img = cv2.medianBlur(img, 5)

mean_adaptive = cv2.adaptiveThreshold(
    img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2
)
gaussian_adaptive = cv2.adaptiveThreshold(
    img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
)

plt.subplot(2, 2, 1), plt.imshow(img, cmap="gray")
plt.title("Original"), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 2), plt.imshow(img_threshold, cmap="gray")
plt.title("Binary Threshold (global:50)"), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 3), plt.imshow(mean_adaptive, cmap="gray")
plt.title("Mean Adaptive"), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 4), plt.imshow(gaussian_adaptive, cmap="gray")
plt.title("Gaussian Adaptive"), plt.xticks([]), plt.yticks([])
plt.show()

