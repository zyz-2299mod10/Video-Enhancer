import cv2
import time
import argparse
import os
import sys

from light.LightEnhancer import LightEnhancer
from distortion.distortion import remove_fish_eye
from distortion.sobel_detail import enhance_sobel

sys.path.append(os.path.join(os.path.dirname(__file__), 'Real_ESRGAN'))
try:
    from Real_ESRGAN.video_super_resolution import videoSR
    SR_CHECK = True
except:
    SR_CHECK = False

def process_video_file(input_path, output_path, is_demo, test="light"):
    # Open Input Video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {input_path}")
        return

    # Get Video Properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Processing Video: {input_path}")
    print(f"Resolution: {width}x{height} | FPS: {fps} | Total Frames: {total_frames}")

    # Setup Output Writer
    # 'mp4v' is widely supported for .mp4. If it fails, try 'avc1' or 'XVID' (for .avi)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Initialize Enhancer
    enhancer = LightEnhancer()
    
    frame_count = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break # End of video

        # --- Process Frame ---
        if test == "detail":
            frame = remove_fish_eye(frame)
            # frame = enhance_sobel(frame, demo_plot=False)
            result_img = frame

        elif test == "light":
            result_img, mode, val = enhancer.process_frame(frame)

            # Demo Mode: Draw Text
            if is_demo:
                text = f"Mode: {mode} (Avg: {val:.1f})"
                # Draw black outline
                cv2.putText(result_img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
                # Draw green text
                cv2.putText(result_img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        elif test == "all":
            frame, mode, val = enhancer.process_frame(frame)
            frame = remove_fish_eye(frame)
            # result_img = enhance_sobel(frame, demo_plot=False)
            result_img = frame

            # Demo Mode: Draw Text
            if is_demo:
                text = f"Mode: {mode} (Avg: {val:.1f})"
                # Draw black outline
                cv2.putText(result_img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
                # Draw green text
                cv2.putText(result_img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Write to Output Video
        out.write(result_img)

        # Progress Log
        frame_count += 1
        if frame_count % 30 == 0:
            elapsed = time.time() - start_time
            fps_proc = frame_count / elapsed
            print(f"Processing: {frame_count}/{total_frames} frames ({int(frame_count/total_frames*100)}%) | Speed: {fps_proc:.2f} fps", end='\r')

    # Cleanup
    cap.release()
    out.release()

    print(f"\nDone! Saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Path to input video file (e.g., input.mp4)')
    parser.add_argument('--output', type=str, required=True, help='Path to output video file (e.g., output.mp4)')
    parser.add_argument('--demo', action='store_true', help='Draw debug info on frames')
    parser.add_argument('--test', type=str, default="all", help='Which processing method')
    parser.add_argument('--sr', action='store_true', default=False)
    
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    process_video_file(args.input, args.output, args.demo, args.test)

    if args.sr and SR_CHECK:
        print("Starting Super Resolution...")
        # Use a temporary output path to avoid reading/writing same file
        temp_sr_output = args.output.replace(".mp4", "_sr_temp.mp4")
        if temp_sr_output == args.output:
             temp_sr_output = args.output + "_temp.mp4"
             
        videoSR(args.output, temp_sr_output)
        
        # Replace original with SR version
        if os.path.exists(temp_sr_output):
            os.replace(temp_sr_output, args.output)
            print(f"Super Resolution complete. Updated: {args.output}")
        else:
            print("Super Resolution failed, keeping original output.")

if __name__ == "__main__":
    main()