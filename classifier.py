import cv2

image = cv2.imread("grid1.jpg")
original = image.copy()

cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Original", 800, 600)
cv2.imshow("Original", image)
cv2.waitKey(0)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)

thresh = cv2.adaptiveThreshold(
    blur,
    255,
    cv2.ADAPTIVE_THRESH_MEAN_C,
    cv2.THRESH_BINARY,
    11,
    2
)

contours, _ = cv2.findContours(
    thresh,
    cv2.RETR_TREE,
    cv2.CHAIN_APPROX_SIMPLE
)

ROI_number = 0
ROIs = []  # lista in memoria delle celle

for c in contours:
    area = cv2.contourArea(c)
    if area > 2000:   # filtra i contorni piccoli
        x, y, w, h = cv2.boundingRect(c)
        ROI = original[y:y+h, x:x+w]

        # salva ROI su disco
        cv2.imwrite(f"ROI_{ROI_number}.png", ROI)
        ROI_number += 1

        # salva anche in memoria (opzionale)
        # ROIs.append((x, y, ROI))

        # disegna rettangolo sull'immagine originale
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.namedWindow("Detected cells", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Detected cells", 800, 600)
cv2.imshow("Detected cells", image)
cv2.waitKey(0)
cv2.destroyAllWindows()