# if you're following this free study group on computer vision essentials:
# https://www.facebook.com/groups/datasciencetutorials

import cv2

# imread = image read
image = cv2.imread("venice.jpg")
(h, w, d) = image.shape

print(f"Height={h}, Width={w}, Depth={d}")

# roi = region of interest
# create a ROI from x=500, y=200 to x=800, y=500
# format: image[startY:endY, startX:endX]
roi = image[410:570, 650:770]

# imshow = image show
# cv2.imshow("Venice Photograph", image)
cv2.imshow("Region of interest", roi)

# wait for a keypress
cv2.waitKey(0)

# Run this code: Terminal > New Terminal