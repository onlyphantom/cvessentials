import cv2
import matplotlib.pyplot as plt

roi = cv2.imread("inter/ocbc-roi.png", flags=0)
blurred = cv2.bilateralFilter(roi, 5, 30, 60)
edged = cv2.adaptiveThreshold(
    blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 5, 5
)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 5))
dilated = cv2.dilate(edged, kernel, iterations=1)

plt.subplot(2, 2, 1), plt.imshow(roi, cmap="gray")
plt.title("Original"), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 2), plt.imshow(blurred, cmap="gray")
plt.title("Blurred"), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 3), plt.imshow(edged, cmap="gray")
plt.title("Edged"), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 4), plt.imshow(dilated, cmap="gray")
plt.title("Dilated"), plt.xticks([]), plt.yticks([])
plt.show()

