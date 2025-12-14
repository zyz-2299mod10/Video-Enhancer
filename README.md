# DIP Final Project

This project implements a video enhancement pipeline addressing low-light/overexposure conditions, lens distortion, and resolution upscaling.

## Install
```
conda create -n DIP python=3.10
conda activate DIP

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130 (Check your CUDA-torch version from https://pytorch.org/get-started/previous-versions/)
pip install -r requirements.txt

# for Real-ESRGAN
cd Real_ESRGAN
python setup.py develop
```

## Usage

Run the main script `main.py` to process videos:

```bash
python main.py --input <input_video_path> --output <output_video_path> [options]
```

### Options
- `--input`: Path to the input video file.
- `--output`: Path to save the processed video.
- `--demo`: Enable demo mode to draw debug information (Current Mode, Average Brightness) on the video frames.
- `--test`: Select the processing pipeline mode.
  - `light`: Only applying light enhancement.
  - `detail`: Only applying distortion correction.
  - `all` (default): Apply light enhancement followed by distortion correction.
- `--sr`: Enable Real-ESRGAN Super Resolution (Upscale x2 by default).

## Pipeline Overview

The processing pipeline consists of the following stages:

1.  **Light Enhancement**: Adjusts brightness and contrast based on frame analysis.
2.  **Distortion Correction**: Removes fish-eye effects.
3.  **Super Resolution** (Optional): Upscales the final video using Deep Learning.

### 1. Light Enhancement (`light/LightEnhancer.py`)

The `LightEnhancer` class analyzes the average brightness of each frame and applies different strategies:

-   **Low Light (< 80)**:
    -   **White Balance**: Uses "Gray World" assumption to correct color casts.
    -   **CLAHE**: Contrast Limited Adaptive Histogram Equalization to enhance local contrast.
    -   **Denoise**: Applies Median Blur to reduce noise often visible in low light.
    -   **Luma Blending**: Temporally blends the Luminance (L) channel with previous frames to smooth brightness changes and reduce flickering, while keeping current Chrominance (A/B) to preserve motion color.

-   **Overexposure (> 200)**:
    -   **Blending**: If a sudden flash occurs, it blends with the last good frame to "hide" the washout.
    -   **Darkening & Contrast**: Darkens the image and reapplies contrast stretching to recover details.
    -   **Gamma Correction**: If blending isn't possible (e.g., prolonged overexposure), applies Gamma correction (`gamma=0.9`) to compress highlights.

-   **Pitch Black (< 2.0)**:
    -   Prevents noise amplification by blending towards a black frame.

-   **Normal Lighting**:
    -   Applies mild temporal Luma blending to stabilize the video.

### 2. Distortion Correction (`distortion/distortion.py`)

-   **`remove_fish_eye(img)`**:
    -   Uses OpenCV's `fisheye.initUndistortRectifyMap` and `remap`.
    -   Applies a specific Camera Matrix (`K`) and Distortion Coefficients (`D`) calibrated for the capture device to rectify the image.

### 3. Super Resolution (`Real_ESRGAN/video_super_resolution.py`)

-   **Algorithm**: Uses **Real-ESRGAN** (`realesr-general-x4v3`) for high-fidelity video upscaling.
-   **Implementation**:
    -   Splits the video into chunks to run in parallel (multiprocessing).
    -   Can utilize multiple GPUs or maximize single-GPU usage.
    -   Merges processed chunks back together using `ffmpeg`.

## Requirements

-   Python 3.x
-   OpenCV (`opencv-python`)
-   NumPy
-   PyTorch
-   ffmpeg-python
-   BasicSR / Real-ESRGAN dependencies
