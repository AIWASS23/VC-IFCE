import cv2

img = cv2.imread('tattoo.jpeg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray1 = cv2.medianBlur(gray, 1)
decalque = cv2.adaptiveThreshold(gray1, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 5)

cv2.imwrite("ink.png", decalque)
