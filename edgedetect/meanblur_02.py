import numpy as np
import cv2

KERNEL_SIZE = (5, 5)

img = cv2.imread("assets/canal.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(f'Gray: {gray[:5, :5]}')
print(f'Shape of Original: {gray.shape}')

for i in range(3):
    newval = np.round(np.mean(gray[:5, i:i+5]))
    print(f'Mean of 25x25 pixel #{i+1}: {np.int(newval)}')

cv2.imshow("Gray", gray)
cv2.waitKey(0)

mean_blur = np.ones(KERNEL_SIZE, dtype="float32") * (1.0 / (5 ** 2))
smoothed_col = cv2.filter2D(img, -1, mean_blur)
smoothed_gray = cv2.filter2D(gray, -1, mean_blur)

cv2.imshow("Smoothed Colored", smoothed_col)
cv2.waitKey(0)

cv2.imshow("Smoothed Gray", smoothed_gray)
cv2.waitKey(0)
print(f'Smoothed: {smoothed_gray[:5, :5]}')
print(f'Shape of Smoothed: {smoothed_gray.shape}')
