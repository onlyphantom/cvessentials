import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("assets/castello.png", flags=0)
img = cv2.medianBlur(img, 9)
img = cv2.GaussianBlur(img, (9, 9), 0)

def sobel(img, k):
    gradient_x = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    gradient_y = cv2.Sobel(img, cv2.CV_64F, 0, 1)   
    gradient_x = cv2.convertScaleAbs(gradient_x)
    gradient_y = cv2.convertScaleAbs(gradient_y)

    return cv2.addWeighted(gradient_x, 0.5, gradient_y, 0.5, 0)

sobel = sobel(img, 3)
canny = cv2.Canny(img, 50, 180)


plt.subplot(1, 2, 1)
plt.imshow(sobel, cmap="gray")
plt.title("Sobel Edge Detector"), plt.xticks([]), plt.yticks([])

plt.subplot(1, 2, 2)
plt.imshow(canny, cmap="gray")
plt.title("Canny Edge Detector"), plt.xticks([]), plt.yticks([])
plt.show()