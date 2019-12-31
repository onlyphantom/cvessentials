import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread("assets/sudoku.jpg", flags=0)

_, img_threshold = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
_, img_trunc = cv2.threshold(img, 90, 255, cv2.THRESH_TRUNC)
_, img_tozero = cv2.threshold(img, 55, 255, cv2.THRESH_TOZERO_INV)

plt.subplot(2, 2, 1), plt.imshow(img, cmap="gray")
plt.title("Original"), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 2), plt.imshow(img_threshold, cmap="gray")
plt.title("Binary Threshold"), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 3), plt.imshow(img_trunc, cmap="gray")
plt.title("To Zero Threshold"), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 4), plt.imshow(img_tozero, cmap="gray")
plt.title("To Zero"), plt.xticks([]), plt.yticks([])
plt.show()

