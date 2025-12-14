import cv2
import time
import argparse
import os
import sys
import ffmpeg
import imageio

from light.LightEnhancer import LightEnhancer
from distortion.distortion import remove_fish_eye

sys.path.append(os.path.join(os.path.dirname(__file__), 'Real_ESRGAN'))
try:
    import types
    from torchvision.transforms.functional import rgb_to_grayscale

    # Create a module for `torchvision.transforms.functional_tensor`
    functional_tensor = types.ModuleType("torchvision.transforms.functional_tensor")
    functional_tensor.rgb_to_grayscale = rgb_to_grayscale

    # Add this module to sys.modules so other imports can access it
    sys.modules["torchvision.transforms.functional_tensor"] = functional_tensor

    from Real_ESRGAN.video_super_resolution import videoSR
    SR_CHECK = True
except Exception as e:
    print("Fall back to basic DIP")
    print(f"ERROR: {e}")
    SR_CHECK = False

def setup_ffmpeg_writer(filename, width, height, fps):
    input_stream = ffmpeg.input('pipe:', 
                                format='rawvideo', 
                                pix_fmt='bgr24', 
                                s=f'{width}x{height}', 
                                framerate=fps)

    output_stream = input_stream.output(filename, 
                                        pix_fmt='yuv420p', 
                                        vcodec='libx264', 
                                        preset='medium',
                                        loglevel='error')
    process = output_stream.overwrite_output().run_async(pipe_stdin=True)
    return process


def merge_video_to_gif(
    original_path,
    sr_path,
    output_gif,
    layout="horizontal",
    max_frames=None,
    fps=10
):
    if not os.path.exists(original_path):
        raise FileNotFoundError(original_path)
    if not os.path.exists(sr_path):
        raise FileNotFoundError(sr_path)

    cap_ori = cv2.VideoCapture(original_path)
    cap_sr = cv2.VideoCapture(sr_path)

    if not cap_ori.isOpened() or not cap_sr.isOpened():
        raise RuntimeError("Cannot open video")

    w = int(cap_ori.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap_ori.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames = []
    idx = 0

    while True:
        ret1, f1 = cap_ori.read()
        ret2, f2 = cap_sr.read()

        if not ret1 or not ret2:
            break

        # resize SR if needed
        if f2.shape[:2] != f1.shape[:2]:
            f2 = cv2.resize(f2, (w, h))

        if layout == "horizontal":
            merged = cv2.hconcat([f1, f2])
        else:
            merged = cv2.vconcat([f1, f2])

        # BGR → RGB (GIF needs RGB)
        merged = cv2.cvtColor(merged, cv2.COLOR_BGR2RGB)

        frames.append(merged)
        idx += 1

        if max_frames is not None and idx >= max_frames:
            break

    cap_ori.release()
    cap_sr.release()

    duration = 1.0 / fps  # seconds per frame

    imageio.mimsave(
        output_gif,
        frames,
        duration=duration,
        loop=0   # infinite loop
    )

    print(f"[OK] GIF saved to {output_gif}")
    print(f"[INFO] Frames: {len(frames)}, FPS: {fps}")


    
def demo(output_path, is_demo, test='all'):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise ValueError("Invalid cam index or camera failure!")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    # out_original = cv2.VideoWriter('original.mp4', fourcc, fps, (width, height))
    # out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    original_writer_process = setup_ffmpeg_writer('original.mp4', width, height, fps)
    processed_writer_process = setup_ffmpeg_writer(output_path, width, height, fps)

    enhancer = LightEnhancer()
    ret, frame = cap.read()
    
    while(ret):
        original_frame = frame.copy()
        # --- Process Frame ---
        if test == "detail":
            processed_frame = remove_fish_eye(frame)
            if is_demo:
                text = f"Mode: {mode} (Avg: {val:.1f})"
                cv2.putText(original_frame, f"Original, {text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(processed_frame, f"Distortion Removed, {text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
                cv2.putText(processed_frame, f"Distortion Removed, {text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        elif test == "light":
            processed_frame, mode, val = enhancer.process_frame(frame)

            # Demo Mode: Draw Text
            if is_demo:
                text = f"Mode: {mode} (Avg: {val:.1f})"
                cv2.putText(original_frame, f"Original, {text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                # Draw black outline
                cv2.putText(processed_frame, f"Light Enhancement, {text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
                # Draw green text
                cv2.putText(processed_frame, f"Light Enhancement, {text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        elif test == "all":
            frame, mode, val = enhancer.process_frame(frame)
            processed_frame = remove_fish_eye(frame)

            # Demo Mode: Draw Text
            if is_demo:
                text = f"Mode: {mode} (Avg: {val:.1f})"
                cv2.putText(original_frame, f"Original, {text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                # Draw black outline
                cv2.putText(processed_frame, f"ALL DIP process, {text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
                # Draw green text
                cv2.putText(processed_frame, f"ALL DIP process, {text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # out_original.write(original_frame)
        # out.write(processed_frame)
        original_bytes = original_frame.tobytes()
        processed_bytes = processed_frame.tobytes()

        original_writer_process.stdin.write(original_bytes)
        processed_writer_process.stdin.write(processed_bytes)

        # Concatenate horizontally: Original | Processed
        comparison_image = cv2.hconcat([original_frame, processed_frame])
        
        # Display the combined image 
        cv2.imshow("Original vs Processed", comparison_image)

        key = cv2.waitKey(1)
        ret, frame = cap.read()
        if(key == 27):
            break

    cap.release()

    original_writer_process.stdin.close()
    processed_writer_process.stdin.close()
    
    original_writer_process.wait()
    processed_writer_process.wait()

    cv2.destroyAllWindows()
    print("Process terminated.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, required=True, help='Path to output video file (e.g., output.mp4)')
    parser.add_argument('--demo', action='store_true', help='Draw debug info on frames')
    parser.add_argument('--test', type=str, default="all", help='Which processing method')
    parser.add_argument('--sr', action='store_true', default=False)

    args = parser.parse_args()

    demo(args.output, args.demo, args.test)

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

    merge_video_to_gif(
        original_path="original.mp4",
        sr_path=args.output,
        output_gif="compare.gif",
        layout="horizontal",
        fps=5
    )

if __name__ == "__main__":
    main()