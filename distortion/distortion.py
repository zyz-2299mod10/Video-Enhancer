import numpy as np
import cv2
from scipy.interpolate import RectBivariateSpline

def undistortion(frame, k1=0.5, k2=0.1):
    h, w = frame.shape[:2]
    f = 1.5 * w
    cx, cy = w / 2.0, h / 2.0

    u_out, v_out = np.meshgrid(np.arange(w), np.arange(h))

    x = (u_out-cx) / f
    y = (v_out-cy) / f

    r_square = x**2 + y**2
    k = 1 + k1*r_square + k2*(r_square**2)
    xd, yd = x / k, y / k

    u_in = xd * f + cx
    v_in = yd * f + cy

    output = np.zeros_like(frame, dtype=frame.dtype)
    for c in range(frame.shape[2]):
        func = RectBivariateSpline(np.arange(h), np.arange(w), frame[:, :, c], kx=1, ky=1)
        pixels = func.ev(v_in.flatten(), u_in.flatten()).reshape(h, w)
        output[:, :, c] = np.clip(pixels, 0, 255).astype(frame.dtype)
    
    return output