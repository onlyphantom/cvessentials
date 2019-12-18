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

smoothed_col = cv2.blur(img, KERNEL_SIZE)

# equivalently:
# smoothed_gray = cv2.boxFilter(gray, -1, KERNEL_SIZE)
smoothed_gray = cv2.blur(gray, KERNEL_SIZE)

cv2.imshow("Smoothed Colored", smoothed_col)
cv2.waitKey(0)

cv2.imshow("Smoothed Gray", smoothed_gray)
cv2.waitKey(0)
print(f'Smoothed: {smoothed_gray[:5, :5]}')
print(f'Shape of Smoothed: {smoothed_gray.shape}')
