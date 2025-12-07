import cv2
import numpy as np
import os
import argparse

# Thresholds
THRESH_OVEREXPOSURE = 200 
THRESH_LOWLIGHT = 80     
THRESH_BLACK_HOLE = 2.0    # Threshold for "pitch black", helps avoid amplifying pure noise

# Algorithm Parameters
GAMMA_VAL = 1.2            
CLAHE_CLIP = 0.5           
TILE_GRID_SIZE = (8, 8)

# Temporal Smoothing (Blending) Weights:
#   Alpha determines how much of the "PREVIOUS" frame we keep.
#   Result = (Previous * Alpha) + (Current * (1 - Alpha))
BLEND_OVER_EX = 0.7        # High: Hide the white screen aggressively
BLEND_LOW_LIGHT = 0.5      # Medium: Smooth out noise flickering, but allow motion
BLEND_NORMAL = 0.1         # Low: Smooth transitions when light returns, acts as mild stabilization

DARKEN_FACTOR = 0.6        # Factor to darken white frames before blending
MAX_BLEND_FRAMES = 8       # Max frames to force blend in overexposure


class VideoEnhancer:
    def __init__(self):
        self.last_good_frame = None
        self.overexposure_counter = 0

    def get_brightness(self, img):
        """
        Calculate the average brightness using Grayscale mean.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return np.mean(gray)

    def apply_gamma_correction(self, img, gamma=1.0):
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
                          for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(img, table)

    def apply_gray_world_wb(self, img):
        b, g, r = cv2.split(img)
        b_avg = np.mean(b)
        g_avg = np.mean(g)
        r_avg = np.mean(r)

        if b_avg == 0 or g_avg == 0 or r_avg == 0:
            return img

        k = (b_avg + g_avg + r_avg) / 3
        k_b = k / b_avg
        k_g = k / g_avg
        k_r = k / r_avg

        b = cv2.addWeighted(b, k_b, 0, 0, 0)
        g = cv2.addWeighted(g, k_g, 0, 0, 0)
        r = cv2.addWeighted(r, k_r, 0, 0, 0)

        return cv2.merge([b, g, r])
    
    def contrast_stretching(self, image, low_percentile=0.1, high_percentile=99.9):
        r_min = np.percentile(image, low_percentile)
        r_max = np.percentile(image, high_percentile)
        
        if r_max - r_min > 0:
            stretched = (image - r_min) / (r_max - r_min) * 255.0
        else:
            stretched = image
        
        stretched = np.clip(stretched, 0, 255)
        return stretched.astype(np.uint8)

    def apply_clahe_enhancement(self, img, clip_limit=2.0):
        # Convert to LAB space
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # Create CLAHE object
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=TILE_GRID_SIZE)
        l_out = clahe.apply(l)

        # Chroma Denoising (Optional: Blur A/B channels to reduce color noise)
        a = cv2.GaussianBlur(a, (3, 3), 0)
        b = cv2.GaussianBlur(b, (3, 3), 0)

        # Merge and convert back to BGR
        merged = cv2.merge((l_out, a, b))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    def temporal_blend_rgb(self, current, alpha):
        """
        Standard RGB Blending (Ghosting is possible).
        Use this when the current frame is garbage (e.g., Overexposure).
        """
        if self.last_good_frame is None: return current
        return cv2.addWeighted(self.last_good_frame, alpha, current, 1 - alpha, 0)

    def temporal_blend_luma(self, current, alpha):
        """
        Luminance-Only Blending.
        Keeps current COLOR (sharp), only smoothes BRIGHTNESS.
        Reduces color artifacts during motion.
        """
        if self.last_good_frame is None: return current

        # 1. Convert both to LAB
        cur_lab = cv2.cvtColor(current, cv2.COLOR_BGR2LAB)
        last_lab = cv2.cvtColor(self.last_good_frame, cv2.COLOR_BGR2LAB)

        cur_l, cur_a, cur_b = cv2.split(cur_lab)
        last_l, _, _ = cv2.split(last_lab) # We don't care about old color

        # 2. Blend L Channel Only
        # new_L = last_L * alpha + cur_L * (1 - alpha)
        new_l = cv2.addWeighted(last_l, alpha, cur_l, 1 - alpha, 0)

        # 3. Merge with CURRENT A/B Channels
        # This ensures the color is exactly where the object is NOW.
        merged = cv2.merge((new_l, cur_a, cur_b))

        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    def process_frame(self, img):
        """
        Main Pipeline
        """
        brightness = self.get_brightness(img)
        mode_str = ""
        processed_img = img.copy()

        # Overexposure
        if brightness > THRESH_OVEREXPOSURE:
            
            # If we have history, blend to hide the flash
            if self.last_good_frame is not None and self.overexposure_counter < MAX_BLEND_FRAMES:
                mode_str = "Overexposure (Blending)"
                
                # Darken current frame to recover info from white clipping
                current_darkened = cv2.convertScaleAbs(img, alpha=DARKEN_FACTOR, beta=0)

                # USE RGB BLEND (Because the RGB in overexposure are all broken)
                processed_img = self.temporal_blend_rgb(current_darkened, BLEND_OVER_EX)

                # Contrast stretching to fix the gray look caused by darkening
                processed_img = self.contrast_stretching(processed_img)
                
                self.overexposure_counter += 1
            else:
                # Timeout or no history -> Gamma Correction
                mode_str = "Overexposure (Gamma)"
                processed_img = self.apply_gamma_correction(processed_img, gamma=GAMMA_VAL)
                
                # blending with slight normal weight to smooth transition
                processed_img = self.temporal_blend_rgb(processed_img, BLEND_NORMAL)

        # Low Light
        elif brightness < THRESH_LOWLIGHT:
            
            # Ultra-low light check: If it's basically pitch black, 
            # don't try to enhance (it just creates noise).
            if brightness < THRESH_BLACK_HOLE:
                mode_str = "Pitch Black"
                # Blend towards black (zeros) instead of snapping to black
                black_frame = np.zeros_like(img)
                processed_img = self.temporal_blend_rgb(black_frame, BLEND_LOW_LIGHT)
            
            else:
                mode_str = "Low Light / Color Fix"
                
                # White Balance
                processed_img = self.apply_gray_world_wb(processed_img)  
                
                # CLAHE
                processed_img = self.apply_clahe_enhancement(processed_img, clip_limit=CLAHE_CLIP) 

                # Median Blur (Denoise)
                processed_img = cv2.medianBlur(processed_img, 3)
                
                # USE LUMA BLEND to keep the noise pattern from "freezing" and avoids ghosting if objects move
                processed_img = self.temporal_blend_luma(processed_img, BLEND_LOW_LIGHT)
            
            self.overexposure_counter = 0

        # Normal Lighting
        else:
            mode_str = "Normal Lighting"
            processed_img = self.temporal_blend_luma(img, BLEND_NORMAL)
            
            self.overexposure_counter = 0

        # Update global state
        self.last_good_frame = processed_img.copy()

        return processed_img, mode_str, brightness

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Input folder path')
    parser.add_argument('--output', type=str, required=True, help='Output folder path')
    parser.add_argument('--demo', action='store_true', help='Draw debug info on images')
    
    args = parser.parse_args()
    input_root = args.input
    output_root = args.output
    is_demo = args.demo

    # Initialize the Processor
    enhancer = VideoEnhancer()

    for root, dirs, files in os.walk(input_root):
        
        # Sort files to ensure temporal order
        sorted_files = sorted([f for f in files if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp'))])
        
        if len(sorted_files) > 0:
            enhancer = VideoEnhancer() 

        for file in sorted_files:
            input_path = os.path.join(root, file)
            rel_path = os.path.relpath(input_path, input_root)
            output_path = os.path.join(output_root, rel_path)
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            img = cv2.imread(input_path)
            if img is None:
                continue

            result_img, mode, val = enhancer.process_frame(img)

            if is_demo:
                text = f"Mode: {mode} (Avg: {val:.1f})"
                cv2.putText(result_img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
                cv2.putText(result_img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imwrite(output_path, result_img)
            print(f"[{mode: <25}] {rel_path} (Bri: {val:.1f})")

if __name__ == "__main__":
    main()