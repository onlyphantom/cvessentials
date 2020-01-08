import cv2
import numpy as np


image = cv2.imread("assets/pens.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Grayscale", image)
cv2.waitKey(0)

image = cv2.GaussianBlur(image, (3, 3), 0)
cv2.imshow("After Smoothing", image)
cv2.waitKey(0)


def sobel(image):
    # run with col.png for best effect
    # cv2.Sobel last 2 argument -> order of derivatives in x and y direction respectively
    sobelX = cv2.Sobel(image, cv2.CV_64F, 1, 0)  # find vertical edges
    sobelY = cv2.Sobel(image, cv2.CV_64F, 0, 1)  # find horizontal edges along y-axis

    gradient_x = np.uint8(np.absolute(sobelX))
    gradient_y = np.uint8(np.absolute(sobelY))

    sobelCombined = cv2.bitwise_or(gradient_x, gradient_y)
    cv2.imshow("Sobel Combined", sobelCombined)
    cv2.waitKey(0)
    return sobelCombined


def counting_penguins(sobel, image):
    sobeled = sobel(image)
    _, edged = cv2.threshold(sobeled, 20, 255, cv2.THRESH_BINARY)
    cv2.imshow("(Edged)", edged)
    cv2.waitKey(0)
    cnts, _ = cv2.findContours(
        # does this need to be changed?
        edged,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    canvas = np.ones(image.shape)
    cv2.drawContours(canvas, cnts, -1, (0, 255, 255), 1)
    cv2.imshow("Contour", canvas)
    cv2.waitKey(0)

    print(f"Found {len(cnts)} penguins")


if __name__ == "__main__":
    counting_penguins(sobel, image)
