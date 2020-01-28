import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX
CYAN = (255, 255, 0)
DIGITSDICT = {
    (1, 1, 1, 1, 1, 1, 0): 0,
    (0, 1, 1, 0, 0, 0, 0): 1,
    (1, 1, 0, 1, 1, 0, 1): 2,
    (1, 1, 1, 1, 0, 0, 1): 3,
    (0, 1, 1, 0, 0, 1, 1): 4,
    (1, 0, 1, 1, 0, 1, 1): 5,
    (1, 0, 1, 1, 1, 1, 1): 6,
    (1, 1, 1, 0, 0, 1, 0): 7,
    (1, 1, 1, 1, 1, 1, 1): 8,
    (1, 1, 1, 1, 0, 1, 1): 9,
}


# roi_color = cv2.imread("inter/dbs-roi.png")
roi_color = cv2.imread("inter/ocbc-roi.png")
roi = cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY)

RATIO = roi.shape[0] * 0.2

roi = cv2.bilateralFilter(roi, 5, 30, 60)

trimmed = roi[int(RATIO) :, int(RATIO) : roi.shape[1] - int(RATIO)]
roi_color = roi_color[int(RATIO) :, int(RATIO) : roi.shape[1] - int(RATIO)]
cv2.imshow("Blurred and Trimmed", trimmed)
cv2.waitKey(0)

edged = cv2.adaptiveThreshold(
    trimmed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 5, 5
)
cv2.imshow("Edged", edged)
cv2.waitKey(0)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 5))
dilated = cv2.dilate(edged, kernel, iterations=1)

cv2.imshow("Dilated", dilated)
cv2.waitKey(0)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
dilated = cv2.dilate(dilated, kernel, iterations=1)

cv2.imshow("Dilated x2", dilated)
cv2.waitKey(0)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 1),)
eroded = cv2.erode(dilated, kernel, iterations=1)

cv2.imshow("Eroded", eroded)
cv2.waitKey(0)

h = roi.shape[0]
ratio = int(h * 0.07)
eroded[-ratio:,] = 0
eroded[:, :ratio] = 0

cv2.imshow("Eroded + Black", eroded)
cv2.waitKey(0)

cnts, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
digits_cnts = []

canvas = trimmed.copy()
cv2.drawContours(canvas, cnts, -1, (255, 255, 255), 1)
cv2.imshow("All Contours", canvas)
cv2.waitKey(0)

canvas = trimmed.copy()
for cnt in cnts:
    (x, y, w, h) = cv2.boundingRect(cnt)
    if h > 20:
        digits_cnts += [cnt]
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 0, 0), 1)
        cv2.drawContours(canvas, cnt, 0, (255, 255, 255), 1)
        cv2.imshow("Digit Contours", canvas)
        cv2.waitKey(0)

print(f"No. of Digit Contours: {len(digits_cnts)}")


cv2.imshow("Digit Contours", canvas)
cv2.waitKey(0)


sorted_digits = sorted(digits_cnts, key=lambda cnt: cv2.boundingRect(cnt)[0])

canvas = trimmed.copy()


for i, cnt in enumerate(sorted_digits):
    (x, y, w, h) = cv2.boundingRect(cnt)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 0, 0), 1)
    cv2.putText(canvas, str(i), (x, y - 3), FONT, 0.3, (0, 0, 0), 1)

cv2.imshow("All Contours sorted", canvas)
cv2.waitKey(0)

digits = []
canvas = roi_color.copy()
for cnt in sorted_digits:
    (x, y, w, h) = cv2.boundingRect(cnt)
    roi = eroded[y : y + h, x : x + w]
    print(f"W:{w}, H:{h}")
    # convenience units
    qW, qH = int(w * 0.25), int(h * 0.15)
    fractionH, halfH, fractionW = int(h * 0.05), int(h * 0.5), int(w * 0.25)

    # seven segments in the order of wikipedia's illustration
    sevensegs = [
        ((0, 0), (w, qH)),  # a (top bar)
        ((w - qW, 0), (w, halfH)),  # b (upper right)
        ((w - qW, halfH), (w, h)),  # c (lower right)
        ((0, h - qH), (w, h)),  # d (lower bar)
        ((0, halfH), (qW, h)),  # e (lower left)
        ((0, 0), (qW, halfH)),  # f (upper left)
        # ((0, halfH - fractionH), (w, halfH + fractionH)) # center
        (
            (0 + fractionW, halfH - fractionH),
            (w - fractionW, halfH + fractionH),
        ),  # center
    ]

    # initialize to off
    on = [0] * 7

    for (i, ((p1x, p1y), (p2x, p2y))) in enumerate(sevensegs):
        region = roi[p1y:p2y, p1x:p2x]
        print(
            f"{i}: Sum of 1: {np.sum(region == 255)}, Sum of 0: {np.sum(region == 0)}, Shape: {region.shape}, Size: {region.size}"
        )
        if np.sum(region == 255) > region.size * 0.5:
            on[i] = 1
        print(f"State of ON: {on}")

    digit = DIGITSDICT[tuple(on)]
    print(f"Digit is: {digit}")
    digits += [digit]
    cv2.rectangle(canvas, (x, y), (x + w, y + h), CYAN, 1)
    cv2.putText(canvas, str(digit), (x - 5, y + 6), FONT, 0.3, (0, 0, 0), 1)
    cv2.imshow("Digit", canvas)
    cv2.waitKey(0)

print(f"Digits on the token are: {digits}")

