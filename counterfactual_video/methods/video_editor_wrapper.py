from .tokenflow.preprocess import prep
from .tokenflow.run_tokenflow_pnp import TokenFlow
from .tokenflow.util import seed_everything, save_videos_grid__, save_video
from .tuneavideo.models.modeling_utils import ModelMixin
from .tuneavideo.pipelines.pipeline_tuneavideo import TuneAVideoPipeline
from .tuneavideo.models.unet import UNet3DConditionModel
from .tuneavideo.data.dataset import TuneAVideoDataset
from .flatten.inference import run_flatten_pipeline
from huggingface_hub import snapshot_download
from diffusers import StableDiffusionPipeline
import yaml, json
import argparse
import torch
from tqdm import tqdm
import os
import cv2

import cv2
import os

def video_to_frames(video_path: str, output_dir: str, ext: str = "png"):
    """
    Extract frames from a video and save them as numbered images like 00000.png.

    Args:
        video_path: Path to the input video.
        output_dir: Directory to save frames.
        ext: Image extension ('png', 'jpg', etc.)
    """
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    success, frame = cap.read()

    while success:
        filename = os.path.join(output_dir, f"{frame_count:05d}.{ext}")
        cv2.imwrite(filename, frame)
        frame_count += 1
        success, frame = cap.read()

    cap.release()
    print(f"Saved {frame_count} frames to {output_dir}")
    
class VideoEditorWrapper:
    def __init__(self, config, method="tokenflow"):
        self.method = method
        self.config = config
        self.device = "cuda"
        
        #define Tune-A-Video pipeline
        if self.method == "tuneavideo":
            self.model_key = snapshot_download(self.config["pretrained_model_path"])
            self.train_dataset = TuneAVideoDataset(video_path=self.config["data_path"], n_sample_frames=self.config["video_length"])
            print("Tune-A-Video dataset created!")
            self.unet = UNet3DConditionModel.from_pretrained(self.config["checkpoint_dir"], subfolder='unet', torch_dtype=torch.float16).to(self.device)
            print("UNET Loaded!")
            self.unet.enable_xformers_memory_efficient_attention()
            self.pipe = TuneAVideoPipeline.from_pretrained(self.model_key, unet=self.unet, torch_dtype=torch.float16).to(self.device)
            print("Tune-A-Video pipeline loaded!")
            self.pipe.enable_vae_slicing()
            self.ddim_inv_latent = torch.load(self.config["checkpoint_dir"] + "/inv_latents/ddim_latent-450.pt").to(torch.float16).to(self.device)
            print("Latents loaded!")
            
    
    #run editing pipelines
    def run(self):
        if self.method == "tokenflow":
            #prep(self.opts)
            editor = TokenFlow(self.config)
            edited_frames, _ = editor.edit_video()
            source_frames = editor.frames  # Ensure frames are on GPU
        
        if self.method == "flatten":
            edited_frames, orig_frames = run_flatten_pipeline(prompt = self.config["prompt"], neg_prompt = self.config["neg_prompt"],
                                              guidance_scale = self.config["guidance_scale"], video_path=self.config["data_path"],
                                              output_path = self.config["output_path"])
            orig_frames = orig_frames / 255
            source_frames = orig_frames
        
        
        if self.method == "tuneavideo":
            print("Begin Tune-A-Video Processing!")
            orig_frames = torch.tensor(self.train_dataset.__getitem__()["pixel_values"]).to(self.device)
            orig_frames = (orig_frames + 1) / 2  # Normalize to [0,1]
            source_frames  = orig_frames
            
            with torch.no_grad():
                frames = self.pipe(self.config["prompt"], latents=self.ddim_inv_latent, video_length=self.config["video_length"],
                                  height=512, width=512, num_inference_steps=50, guidance_scale=self.config["guidance_scale"]).videos.to(self.device)

                save_videos_grid__(frames, f'{self.config["output_path"]}/edited_fps20.gif', fps=20)
                print("Frames generated successfully!")
                edited_frames = frames.permute(0, 2, 1, 3, 4).squeeze(0)
            #print(frames.shape)

        return edited_frames, source_frames