import cv2
import matplotlib.pyplot as plt

roi = cv2.imread("assets/0417s.png", flags=0)
cv2.imshow("Original", roi)
cv2.waitKey(0)

_, thresh = cv2.threshold(roi, 170, 255, cv2.THRESH_BINARY)
# thresh = cv2.adaptiveThreshold(dilated, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 5, 5)
cv2.imshow("Threshold", thresh)
cv2.waitKey(0)

inv = cv2.bitwise_not(thresh)
cv2.imshow("Inverted", inv)
cv2.waitKey(0)


kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
eroded = cv2.erode(inv, kernel, iterations=1)

cv2.imshow("Eroded", eroded)
cv2.waitKey(0)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 6))
dilated = cv2.dilate(eroded, kernel, iterations=1)

cv2.imshow("Dilated", dilated)
cv2.waitKey(0)


plt.subplot(2, 2, 1), plt.imshow(roi, cmap="gray")
plt.title("Original"), plt.xticks([]), plt.yticks([])

plt.subplot(2, 2, 2), plt.imshow(thresh, cmap="gray")
plt.title("Thresholded"), plt.xticks([]), plt.yticks([])

plt.subplot(2, 2, 3), plt.imshow(inv, cmap="gray")
plt.title("Inverted"), plt.xticks([]), plt.yticks([])

plt.subplot(2, 2, 4), plt.imshow(dilated, cmap="gray")
plt.title("Transformed"), plt.xticks([]), plt.yticks([])

plt.show()

