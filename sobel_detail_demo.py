import cv2
import numpy as np 
import matplotlib.pyplot as plt



def grads_sobel(src, ksize, gamma, demo_plot=False):
    sobelx = cv2.Sobel(src, cv2.CV_64F, 1, 0, ksize=ksize)
    sobely = cv2.Sobel(src, cv2.CV_64F, 0, 1, ksize=ksize)
    grads  = sobelx+sobely
    grads  = grads - np.mean(grads)
    grads  = grads/np.std(grads)
    # gamma correction
    # 1: unchanged
    # >1: increase contrast
    # <1: reduce contrast
    corrected_grads  = np.power(np.absolute(grads), gamma) * np.where(grads>0, 1, -1) 
    if demo_plot:
        plt.figure()
        plt.suptitle("Sobel Gradients")
        plt.subplot(121)
        plt.title("normalized")
        plt.imshow(grads, cmap='grey')
        plt.colorbar()
        plt.axis('off')
        plt.subplot(122)
        plt.title("gamma = " + str(gamma))
        plt.imshow(corrected_grads, cmap='grey')
        plt.colorbar()
        plt.axis('off')
        plt.show()
    return corrected_grads

def enhance_sobel(src, ksize=3, strength=0.25, gamma=0.5, demo_plot=False):
    if src is None:
        print("input: None")
        return 
    if(len(src.shape)) == 3:
        src_HSV = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
        grads  = grads_sobel(src=cv2.medianBlur(src_HSV[:, :, 2], ksize=11), ksize=ksize, gamma=gamma, demo_plot=demo_plot)
        result = (1-strength) * src_HSV[:, :, 2] + strength * grads
        result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        src_HSV[:, :, 2] = result
        return cv2.cvtColor(src_HSV, cv2.COLOR_HSV2BGR)
    else:    
        grads  = grads_sobel(src=src, ksize=ksize, gamma=gamma)
        result = src + strength * grads
        result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return result

# using interpolation on 1+strech * cos(-pi/2, pi/2)  to scale to 1
def linear_fisheye_correction(src, x_strech=0.05, y_strech=0.02):
    if src is None:
        print("input: None")
        return 
    # interpolation with row
    src = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    row_axis = np.arange(src.shape[1]) - src.shape[1]/2
    for rows in range(src.shape[0]):
        offset = src.shape[0]/2
        distortion = (rows-offset)/src.shape[0] * np.pi
        src[rows, :, 0] = np.interp(x=row_axis, xp=row_axis*(1-x_strech*np.cos(distortion)), fp=src[rows, :, 0])
        src[rows, :, 1] = np.interp(x=row_axis, xp=row_axis*(1-x_strech*np.cos(distortion)), fp=src[rows, :, 1])
        src[rows, :, 2] = np.interp(x=row_axis, xp=row_axis*(1-x_strech*np.cos(distortion)), fp=src[rows, :, 2])
    col_axis = np.arange(src.shape[0]) - src.shape[0]/2
    for cols in range(src.shape[1]):
        offset = src.shape[1]/2
        distortion = (cols-offset)/src.shape[0] * np.pi
        src[:, cols, 0] = np.interp(x=col_axis, xp=col_axis*(1-y_strech*np.cos(distortion)), fp=src[:, cols, 0])
        src[:, cols, 1] = np.interp(x=col_axis, xp=col_axis*(1-y_strech*np.cos(distortion)), fp=src[:, cols, 1])
        src[:, cols, 2] = np.interp(x=col_axis, xp=col_axis*(1-y_strech*np.cos(distortion)), fp=src[:, cols, 2])
    return cv2.cvtColor(src, cv2.COLOR_HSV2BGR)

def main():
    # Create a VideoCapture object
    # Use the file path for a video or 0 for a webcam

    cap = cv2.VideoCapture('./data/1.mp4')
    # Set resolution to 1280x720 (HD)
    #cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    #cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    demo_plot = True
    # Check if the video source was opened successfully
    if not cap.isOpened():
        print("Error: Could not open video source.")
        exit()

    ret, frame = cap.read()
    frame = linear_fisheye_correction(frame)
    frame = enhance_sobel(frame, demo_plot=demo_plot)
    demo_plot = False
    # Read and display frames in a loop
    while ret:

        if not ret:
            print("End of video or error reading frame.")
            break

        # Display the frame
        cv2.imshow('Video Feed', frame)

        # Press 'q' to exit. waitKey(25) controls playback speed.
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
        ret, frame = cap.read()
        frame = linear_fisheye_correction(frame)
        frame = enhance_sobel(frame, demo_plot=demo_plot)
    # Release the video source and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()