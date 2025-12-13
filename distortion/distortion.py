import numpy as np
import cv2
import argparse
import os

def remove_fish_eye(img):
    DIM=(640, 480)
    K=np.array([[651.1879334257668, 0.0, 320.3126614422408], [0.0, 650.9817909640441, 278.97173817481263], [0.0, 0.0, 1.0]])
    D=np.array([[0.17105350704748695], [-0.5008937086636058], [1.0712933223194445], [-2.032938576668345]])

    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    return undistorted_img

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Input folder path')
    parser.add_argument('--output', type=str, required=True, help='Output folder path')
    
    args = parser.parse_args()
    input_root = args.input
    output_root = args.output

    # Initialize the Processor

    for root, dirs, files in os.walk(input_root):
        # Sort files to ensure temporal order
        sorted_files = sorted([f for f in files if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp'))])

        for file in sorted_files:
            input_path = os.path.join(root, file)
            rel_path = os.path.relpath(input_path, input_root)
            output_path = os.path.join(output_root, rel_path)
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            img = cv2.imread(input_path)
            if img is None:
                continue

            result_img = remove_fish_eye(img)

            cv2.imwrite(output_path, result_img)

if __name__ == "__main__":
    main()