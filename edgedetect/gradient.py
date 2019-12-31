import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import cv2

img = cv2.imread("assets/pen.jpg")
flat = (img[:, :, 0] + img[:, :, 1] + img[:, :, 2]) / 3
print(flat.shape)
sa = 16  # sample at every 16


fig, ax = plt.subplots(1, 1)
ret = ax.imshow(
    flat, zorder=0, alpha=1.0, cmap="Greys_r", origin="upper", interpolation="hermite",
)
plt.colorbar(ret)
Y, X = np.mgrid[0 : flat.shape[0] : sa, 0 : flat.shape[1] : sa]
dY, dX = np.gradient(flat[::sa, ::sa])
ax.quiver(X, Y, dX, dY, color="r")
plt.show()

