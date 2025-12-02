import cv2
import numpy as np
import processing as p

h = 720
w = 4 * h // 3
cam = 1

cap = cv2.VideoCapture(cam)
if not cap.isOpened():
    raise ValueError("Invalid cam index or camera failure!")

# built in autoWB: 1 for off, 3 for on
# cap.set(cv2.CAP_PROP_AUTO_WB, 1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

ret, frame = cap.read()

while(ret):
    frame = p.PCA_3D(frame, mode="PR", crit=0.15)
    frame = cv2.medianBlur(frame, 3)
    # frame = p.sharpen(frame, sigma=10, strength=0.5)
    # frame = p.AutoExp_Hist(frame)
    cv2.imshow("captured", frame)
    key = cv2.waitKey(500)
    ret, frame = cap.read()
    if(key == 27):
        break
cap.release()
cv2.destroyAllWindows
print("Process terminated.")