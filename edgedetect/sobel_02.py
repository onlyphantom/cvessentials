import numpy as np
import cv2
import matplotlib.pyplot as plt

img_original = cv2.imread("assets/castello.png")
img_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)
img = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
img = cv2.medianBlur(img, 9)
img = cv2.GaussianBlur(img, (9, 9), 0)

gradient_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
gradient_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

gradient_x = cv2.convertScaleAbs(gradient_x)
gradient_y = cv2.convertScaleAbs(gradient_y)
print(f"Range: {np.min(gradient_x)} | {np.max(gradient_x)}")

gradient_xy = cv2.addWeighted(gradient_x, 0.5, gradient_y, 0.5, 0)

plt.subplot(2, 2, 1), plt.imshow(img_original)
plt.title("Original"), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 2), plt.imshow(gradient_x, cmap="gray")
plt.title("Gradient X"), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 3), plt.imshow(gradient_y, cmap="gray")
plt.title("Gradient Y"), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 4), plt.imshow(gradient_xy, cmap="gray")
plt.title("Gradient X and Y"), plt.xticks([]), plt.yticks([])
plt.show()
