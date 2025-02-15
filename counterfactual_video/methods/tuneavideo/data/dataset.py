import decord
decord.bridge.set_bridge('torch')

from torch.utils.data import Dataset
from einops import rearrange
import os
from PIL import Image
import numpy as np

class TuneAVideoDataset(Dataset):
    def __init__(
            self,
            video_path: str,
            prompt: str,
            width: int = 512,    # Ensure width is 512
            height: int = 512,   # Ensure height is 512
            n_sample_frames: int = 8,
            sample_start_idx: int = 0,
            sample_frame_rate: int = 1,
    ):
        self.video_path = video_path
        self.prompt = prompt
        self.prompt_ids = None
        self.uncond_prompt_ids = None

        self.width = width
        self.height = height
        self.n_sample_frames = n_sample_frames
        self.sample_start_idx = sample_start_idx
        self.sample_frame_rate = sample_frame_rate

        if 'mp4' not in self.video_path:  # Image Folder
            self.images = []
            for file in sorted(os.listdir(self.video_path), key=lambda x: int(x[:-4])):
                if file.endswith('jpg'):
                    img = Image.open(os.path.join(self.video_path, file)).convert('RGB')
                    img = img.resize((self.width, self.height))  # Resize to 512x512
                 #   print(np.array(img).shape)
                    self.images.append(np.asarray(img))
            self.images = np.stack(self.images)  # Stack into (num_frames, H, W, C)

    def __len__(self):
        return 1

    def __getitem__(self, index):
        # Load and sample video frames
        if 'mp4' in self.video_path:  # Video File
            vr = decord.VideoReader(self.video_path, width=self.width, height=self.height)  # Ensure 512x512
            sample_index = list(range(self.sample_start_idx, len(vr), self.sample_frame_rate))[:self.n_sample_frames]
            video = vr.get_batch(sample_index)  # (F, H, W, C)
        else:
            video = self.images[:self.n_sample_frames]  # (F, H, W, C)

        video = rearrange(video, "f h w c -> f c h w")  # Convert to (F, C, H, W)

        example = {
            "pixel_values": (video / 127.5 - 1.0),  # Normalize to [-1, 1]
            "prompt_ids": self.prompt_ids,
        }

        return example