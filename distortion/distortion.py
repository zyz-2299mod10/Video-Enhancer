import numpy as np
import cv2

def remove_fish_eye(img):
    DIM=(640, 480)
    K=np.array([[651.1879334257668, 0.0, 320.3126614422408], [0.0, 650.9817909640441, 278.97173817481263], [0.0, 0.0, 1.0]])
    D=np.array([[0.17105350704748695], [-0.5008937086636058], [1.0712933223194445], [-2.032938576668345]])

    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    return undistorted_img