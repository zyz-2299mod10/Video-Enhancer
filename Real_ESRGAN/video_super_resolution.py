import cv2
import glob
import mimetypes
import numpy as np
import os
import shutil
import subprocess
import torch
from os import path as osp
from tqdm import tqdm
import sys

try:
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from basicsr.utils.download_util import load_file_from_url
    from realesrgan import RealESRGANer
    from realesrgan.archs.srvgg_arch import SRVGGNetCompact
except ImportError:
    print("Error: Could not import Real-ESRGAN modules. Make sure the Real-ESRGAN submodule is present.")
    raise

try:
    import ffmpeg
except ImportError:
    import pip
    pip.main(['install', '--user', 'ffmpeg-python'])
    import ffmpeg

def get_video_meta_info(video_path):
    ret = {}
    probe = ffmpeg.probe(video_path)
    video_streams = [stream for stream in probe['streams'] if stream['codec_type'] == 'video']
    has_audio = any(stream['codec_type'] == 'audio' for stream in probe['streams'])
    ret['width'] = video_streams[0]['width']
    ret['height'] = video_streams[0]['height']
    ret['fps'] = eval(video_streams[0]['avg_frame_rate'])
    ret['audio'] = ffmpeg.input(video_path).audio if has_audio else None
    ret['nb_frames'] = int(video_streams[0]['nb_frames'])
    return ret

def get_sub_video(input_path, output_folder, num_process, process_idx):
    if num_process == 1:
        return input_path
    meta = get_video_meta_info(input_path)
    duration = int(meta['nb_frames'] / meta['fps'])
    part_time = duration // num_process
    
    video_name = osp.splitext(osp.basename(input_path))[0]
    os.makedirs(osp.join(output_folder, f'{video_name}_inp_tmp_videos'), exist_ok=True)
    out_path = osp.join(output_folder, f'{video_name}_inp_tmp_videos', f'{process_idx:03d}.mp4')
    
    # ffmpeg command to split video
    cmd = [
        'ffmpeg', f'-i {input_path}', '-ss', f'{part_time * process_idx}',
        f'-to {part_time * (process_idx + 1)}' if process_idx != num_process - 1 else '', '-async 1', out_path, '-y'
    ]
    # print(' '.join(cmd))
    subprocess.call(' '.join(cmd), shell=True)
    return out_path

class Reader:
    def __init__(self, args, total_workers=1, worker_idx=0):
        self.args = args
        # Handle sub-video logic inside Reader if needed, or pass pre-split path
        # In this refactor, we pass the split video path in args.input for workers
        
        self.stream_reader = (
            ffmpeg.input(args.input).output('pipe:', format='rawvideo', pix_fmt='bgr24',
                                            loglevel='error').run_async(
                                                pipe_stdin=True, pipe_stdout=True, cmd=args.ffmpeg_bin))
        meta = get_video_meta_info(args.input)
        self.width = meta['width']
        self.height = meta['height']
        self.input_fps = meta['fps']
        self.audio = meta['audio']
        self.nb_frames = meta['nb_frames']
        self.idx = 0

    def get_resolution(self):
        return self.height, self.width

    def get_fps(self):
        if self.args.fps is not None:
            return self.args.fps
        elif self.input_fps is not None:
            return self.input_fps
        return 24

    def get_audio(self):
        return self.audio

    def __len__(self):
        return self.nb_frames

    def get_frame(self):
        img_bytes = self.stream_reader.stdout.read(self.width * self.height * 3)  # 3 bytes for one pixel
        if not img_bytes:
            return None
        img = np.frombuffer(img_bytes, np.uint8).reshape([self.height, self.width, 3])
        return img

    def close(self):
        self.stream_reader.stdin.close()
        self.stream_reader.wait()

class Writer:
    def __init__(self, args, audio, height, width, video_save_path, fps):
        out_width, out_height = int(width * args.outscale), int(height * args.outscale)
        if out_height > 2160:
            print('You are generating video that is larger than 4K, which will be very slow due to IO speed.',
                  'We highly recommend to decrease the outscale(aka, -s).')

        if audio is not None:
            self.stream_writer = (
                ffmpeg.input('pipe:', format='rawvideo', pix_fmt='bgr24', s=f'{out_width}x{out_height}',
                             framerate=fps).output(
                                 audio,
                                 video_save_path,
                                 pix_fmt='yuv420p',
                                 vcodec='libx264',
                                 loglevel='error',
                                 acodec='copy').overwrite_output().run_async(
                                     pipe_stdin=True, pipe_stdout=True, cmd=args.ffmpeg_bin))
        else:
            self.stream_writer = (
                ffmpeg.input('pipe:', format='rawvideo', pix_fmt='bgr24', s=f'{out_width}x{out_height}',
                             framerate=fps).output(
                                 video_save_path, pix_fmt='yuv420p', vcodec='libx264',
                                 loglevel='error').overwrite_output().run_async(
                                     pipe_stdin=True, pipe_stdout=True, cmd=args.ffmpeg_bin))

    def write_frame(self, frame):
        frame = frame.astype(np.uint8).tobytes()
        self.stream_writer.stdin.write(frame)

    def close(self):
        self.stream_writer.stdin.close()
        self.stream_writer.wait()

def inference_video_worker(args, video_save_path, device=None):
    # Load Model (realesr-general-x4v3 only)
    model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
    netscale = 4
    
    weights_dir = os.path.join(os.path.dirname(__file__), 'Real-ESRGAN', 'weights')
    model_path = os.path.join(weights_dir, 'realesr-general-x4v3.pth')
    wdn_model_path = os.path.join(weights_dir, 'realesr-general-wdn-x4v3.pth')
    
    # Use dni
    dni_weight = [args.denoise_strength, 1 - args.denoise_strength]
    model_path = [model_path, wdn_model_path]

    upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        dni_weight=dni_weight,
        model=model,
        tile=args.tile,
        tile_pad=args.tile_pad,
        pre_pad=args.pre_pad,
        half=not args.fp32,
        device=device,
    )

    reader = Reader(args)
    audio = reader.get_audio()
    height, width = reader.get_resolution()
    fps = reader.get_fps()
    
    writer = Writer(args, audio, height, width, video_save_path, fps)

    pbar = tqdm(total=len(reader), unit='frame', desc='inference')
    while True:
        img = reader.get_frame()
        if img is None:
            break

        try:
            output, _ = upsampler.enhance(img, outscale=args.outscale)
        except RuntimeError as error:
            print('Error', error)
            print('If you encounter CUDA out of memory, try to set --tile with a smaller number.')
        else:
            writer.write_frame(output)

        torch.cuda.synchronize(device)
        pbar.update(1)

    reader.close()
    writer.close()

def videoSR(input_path, output_path, tile=0, tile_pad=10, pre_pad=0, fp32=False, outscale=2, num_process_per_gpu=1):
    """
    Perform Real-ESRGAN super-resolution on a video.
    """
    
    # Mocking args object to reuse existing Reader/Writer logic
    class Args:
        pass
    
    args = Args()
    args.input = input_path
    args.output_dir = os.path.dirname(output_path) # Folder for temp files
    args.ffmpeg_bin = 'ffmpeg'
    args.fps = None
    args.outscale = outscale
    args.tile = tile
    args.tile_pad = tile_pad
    args.pre_pad = pre_pad
    args.model_name = 'realesr-general-x4v3'
    args.denoise_strength = 0.5 
    args.face_enhance = False
    args.fp32 = fp32
    
    # Check weights first (download if needed) before spawning processes
    weights_dir = os.path.join(os.path.dirname(__file__), 'Real-ESRGAN', 'weights')
    os.makedirs(weights_dir, exist_ok=True)
    model_path = os.path.join(weights_dir, 'realesr-general-x4v3.pth')
    wdn_model_path = os.path.join(weights_dir, 'realesr-general-wdn-x4v3.pth')
    file_urls = [
         'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth',
         'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth'
    ]
    if not os.path.isfile(model_path) or not os.path.isfile(wdn_model_path):
        print("Downloading weights...")
        for url in file_urls:
            load_file_from_url(url=url, model_dir=weights_dir, progress=True, file_name=None)
            
    num_gpus = torch.cuda.device_count()
    num_process = num_gpus * num_process_per_gpu
    
    # Single Process Case
    if num_process <= 1:
        inference_video_worker(args, output_path, torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        print(f"Video SR finished. Output saved to {output_path}")
        return

    # Multiprocessing Case
    ctx = torch.multiprocessing.get_context('spawn')
    pool = ctx.Pool(num_process)
    
    video_name = osp.splitext(osp.basename(input_path))[0]
    out_tmp_dir = osp.join(args.output_dir, f'{video_name}_out_tmp_videos')
    os.makedirs(out_tmp_dir, exist_ok=True)
    
    print(f"Splitting video into {num_process} parts...")
    
    # We need a new args for each worker with modified input path
    worker_results = []
    
    for i in range(num_process):
        # Create a deep copy or new instance of args for each worker
        sub_args = Args()
        sub_args.__dict__ = args.__dict__.copy()
        
        # Split video -> returns path to sub-video
        sub_input_path = get_sub_video(input_path, args.output_dir, num_process, i)
        sub_args.input = sub_input_path
        
        sub_video_save_path = osp.join(out_tmp_dir, f'{i:03d}.mp4')
        
        pool.apply_async(
            inference_video_worker,
            args=(sub_args, sub_video_save_path, torch.device(i % num_gpus)),
            callback=None 
        )
        
    pool.close()
    pool.join()

    # Combine sub videos
    vidlist_path = f'{args.output_dir}/{video_name}_vidlist.txt'
    with open(vidlist_path, 'w') as f:
        for i in range(num_process):
            f.write(f'file \'{video_name}_out_tmp_videos/{i:03d}.mp4\'\n')

    cmd = [
        args.ffmpeg_bin, '-f', 'concat', '-safe', '0', '-i', vidlist_path, '-c',
        'copy', f'{output_path}', '-y'
    ]
    print('Merging video...')
    # print(' '.join(cmd))
    subprocess.call(cmd)
    
    # Cleanup
    shutil.rmtree(out_tmp_dir)
    inp_tmp_dir = osp.join(args.output_dir, f'{video_name}_inp_tmp_videos')
    if osp.exists(inp_tmp_dir):
        shutil.rmtree(inp_tmp_dir)
    if os.path.exists(vidlist_path):
        os.remove(vidlist_path)

    print(f"Video SR finished. Output saved to {output_path}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True, help='Input video path')
    parser.add_argument('-o', '--output', type=str, required=True, help='Output video path')
    parser.add_argument('-s', '--outscale', type=float, default=2, help='Upsampling scale')
    parser.add_argument('--num_process_per_gpu', type=int, default=1, help='Number of processes per GPU')
    
    args = parser.parse_args()
    
    videoSR(args.input, args.output, outscale=args.outscale, num_process_per_gpu=args.num_process_per_gpu)
