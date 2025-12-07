import cv2
import os
import argparse

VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv')

def extract_frames(video_path, output_root):
    filename = os.path.basename(video_path)
    video_name, _ = os.path.splitext(filename)
    
    output_dir = os.path.join(output_root, video_name)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Processing: {filename} -> {output_dir}")
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Can't open {video_path}")
        return

    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        output_filename = os.path.join(output_dir, f"frame_{frame_count:06d}.png")
        
        cv2.imwrite(output_filename, frame)
        
        frame_count += 1
        
        if frame_count % 100 == 0:
            print(f"  Processed {frame_count} frames...", end='\r')

    cap.release()
    print(f"\nDone! {filename} processed {frame_count} images\n")

def main():
    parser = argparse.ArgumentParser()
    
    # 設定 Arguments
    parser.add_argument('--input', type=str, default="./data")
    parser.add_argument('--output', type=str, default="./data/frame")
    
    args = parser.parse_args()
    
    input_path = args.input
    output_root = args.output

    os.makedirs(output_root, exist_ok=True)

    if os.path.isdir(input_path):
        files = [f for f in os.listdir(input_path) if f.lower().endswith(VIDEO_EXTENSIONS)]

        print(f"Found {len(files)} videos...")
        for f in files:
            full_video_path = os.path.join(input_path, f)
            extract_frames(full_video_path, output_root)
            
    elif os.path.isfile(input_path):
        extract_frames(input_path, output_root)
    else:
        print("Path invalid。")

if __name__ == "__main__":
    main()