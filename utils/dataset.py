import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

if torch.cuda.is_available():
    device = torch.cuda.current_device()
    torch.cuda.init()
    torch.backends.cuda.matmul.allow_tf32 = True
else:
    raise Exception("unable to initialize CUDA")

import os
import random
import numpy as np
from glob import glob
from PIL import Image

from torchvision.transforms import v2, InterpolationMode
from safetensors.torch import load_file

import decord
decord.bridge.set_bridge('torch')
Image.MAX_IMAGE_PIXELS = None


IMAGE_TYPES = [".jpg", ".jpeg", ".png"]
VIDEO_TYPES = [".mp4", ".mkv", ".mov", ".avi", ".webm"]


BUCKET_RESOLUTIONS_624 = {
    "16x9": (832, 480),
    "4x3":  (704, 544),
    "1x1":  (624, 624),
}


BUCKET_RESOLUTIONS_960 = {
    "16x9": (1280, 720),
    "4x3":  (1088, 832),
    "1x1":  (960,  960),
}


BUCKET_RESOLUTIONS_1024 = {
    "16x9": (1344, 768),
    "4x3":  (1184, 896),
    "1x1":  (1024,  1024),
}


def get_resolution(width, height, buckets):
    ar = width / height
    if ar > 1.528:
        new_width, new_height = buckets["16x9"]
    elif ar > 1.15:
        new_width, new_height = buckets["4x3"]
    elif ar > 0.884:
        new_width, new_height = buckets["1x1"]
    elif ar > 0.669:
        new_height, new_width = buckets["4x3"]
    else:
        new_height, new_width = buckets["16x9"]

    return new_width, new_height


def count_tokens(width, height, frames):
    return (width // 16) * (height // 16) * ((frames - 1) // 4 + 1)


class CombinedDataset(Dataset):
    def __init__(
        self,
        root_folder,
        token_limit = 10_000,
        limit_samples = None,
        max_frame_stride = 4,
        bucket_resolution = 624,
        load_control = False,
        control_suffix = "",
    ):
        self.root_folder = root_folder
        self.token_limit = token_limit
        self.max_frame_stride = max_frame_stride
        self.load_control = load_control
        self.control_suffix = control_suffix
        
        if bucket_resolution == 1024:
            self.bucket_resolution = BUCKET_RESOLUTIONS_1024
        elif bucket_resolution == 960:
            self.bucket_resolution = BUCKET_RESOLUTIONS_960
        else:
            self.bucket_resolution = BUCKET_RESOLUTIONS_624
        
        # search for all files matching image or video extensions
        self.media_files = []
        for ext in IMAGE_TYPES + VIDEO_TYPES:
            self.media_files.extend(
                glob(os.path.join(self.root_folder, "**", "*" + ext), recursive=True)
            )
        
        # pull samples evenly from the whole dataset
        if limit_samples is not None:
            stride = max(1, len(self.media_files) // limit_samples)
            self.media_files = self.media_files[::stride]
            self.media_files = self.media_files[:limit_samples]
    
    def __len__(self):
        return len(self.media_files)
    
    def find_max_frames(self, width, height):
        frames = 1
        tokens = count_tokens(width, height, frames)
        while tokens < self.token_limit:
            new_frames = frames + 4
            new_tokens = count_tokens(width, height, new_frames)
            if new_tokens < self.token_limit:
                frames = new_frames
                tokens = new_tokens
            else:
                return frames
    
    def __getitem__(self, idx):
        ext = os.path.splitext(self.media_files[idx])[1].lower()
        if ext in IMAGE_TYPES:
            image = Image.open(self.media_files[idx]).convert('RGB')
            pixels = torch.as_tensor(np.array(image)).unsqueeze(0) # FHWC
            width, height = get_resolution(pixels.shape[2], pixels.shape[1], self.bucket_resolution)
        else:
            vr = decord.VideoReader(self.media_files[idx])
            orig_height, orig_width = vr[0].shape[:2]
            orig_frames = len(vr)
            
            width, height = get_resolution(orig_width, orig_height, self.bucket_resolution)
            max_frames = self.find_max_frames(width, height)
            stride = max(min(random.randint(1, self.max_frame_stride), orig_frames // max_frames), 1)
            
            # sample a clip from the video based on frame stride and length
            seg_len = min(stride * max_frames, orig_frames)
            start_frame = random.randint(0, orig_frames - seg_len)
            pixels = vr[start_frame : start_frame+seg_len : stride]
            max_frames = ((pixels.shape[0] - 1) // 4) * 4 + 1
            pixels = pixels[:max_frames] # clip frames to match vae
        
        # determine crop dimensions to prevent stretching during resize
        pixels_ar = pixels.shape[2] / pixels.shape[1]
        target_ar = width / height
        if pixels_ar > target_ar:
            crop_width = min(int(pixels.shape[1] * target_ar), pixels.shape[2])
            crop_height = pixels.shape[1]
        elif pixels_ar < target_ar:
            crop_width = pixels.shape[2]
            crop_height = min(int(pixels.shape[2] / target_ar), pixels.shape[1])
        else:
            crop_width = pixels.shape[2]
            crop_height = pixels.shape[1]
        
        # convert to expected dtype, resolution, shape, and value range
        transform = v2.Compose([
            v2.ToDtype(torch.float32, scale=True),
            v2.RandomCrop(size=(crop_height, crop_width)),
            v2.Resize(size=(height, width)),
        ])
        
        pixels = pixels.movedim(3, 1).unsqueeze(0).contiguous() # FHWC -> FCHW -> BFCHW
        pixels = transform(pixels) * 2 - 1
        pixels = torch.clamp(torch.nan_to_num(pixels), min=-1, max=1)
        
        if self.load_control:
            raise NotImplementedError("loading control files from disk is not implemented yet")
        else:
            control = None
        
        # load precomputed text embeddings from file
        embedding_file = os.path.splitext(self.media_files[idx])[0] + "_wan.safetensors"
        if not os.path.exists(embedding_file):
            embedding_file = os.path.join(
                os.path.dirname(self.media_files[idx]),
                random.choice(["caption_original_wan.safetensors", "caption_florence_wan.safetensors"]),
            )
        
        if os.path.exists(embedding_file):
            embedding_dict = load_file(embedding_file)
        else:
            raise Exception(f"No embedding file found for {self.media_files[idx]}, you may need to precompute embeddings with --cache_embeddings")
        
        return {"pixels": pixels, "embedding_dict": embedding_dict, "control": control}
