import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

img = cv2.imread("assets/sarpi.png")
blurred = cv2.GaussianBlur(img, (7, 7), 0)
blurred = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)


size = 0.3
width = int(blurred.shape[1] * size)
height = int(blurred.shape[0] * size)
blurred = cv2.resize(blurred, (width, height), interpolation=cv2.INTER_AREA)

print(f"Shape:{blurred.shape}")
cv2.imshow("Blurred", blurred)
cv2.waitKey(0)

xx, yy = np.mgrid[0 : blurred.shape[0], 0 : blurred.shape[1]]

fig = plt.figure()
ax = fig.gca(projection="3d")
ax.plot_surface(xx, yy, blurred, rstride=1, cstride=1, cmap=plt.cm.gray, linewidth=0)

plt.show()
