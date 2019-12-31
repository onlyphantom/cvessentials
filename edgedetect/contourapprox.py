import cv2
import numpy as np


image = cv2.imread("homework/equal.png", flags=0)
cv2.imshow("Original", image)
cv2.waitKey(0)


def edge(image):
    _, edged = cv2.threshold(image, 220, 255, cv2.THRESH_BINARY_INV)
    cv2.imshow("(Edged)", edged)
    cv2.waitKey(0)
    cnts, _ = cv2.findContours(
        # does this need to be changed?
        edged,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    print(f"Cnts Simple Shape (1): {cnts[0].shape}")
    print(f"Cnts Simple Shape (2): {cnts[0].shape}")
    cnts2, _ = cv2.findContours(
        # does this need to be changed?
        edged,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )
    print(f"Cnts NoApprox Shape:{cnts2[0].shape}")
    print(cnts)
    canvas = np.ones(image.shape)
    cv2.drawContours(canvas, cnts, -1, (0, 255, 255), 1)
    cv2.imshow("Contour", canvas)
    cv2.waitKey(0)
    print(f"Found {len(cnts)} shapes!")


if __name__ == "__main__":
    edge(image)
