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
     #   if method == "tokenflow":
     #     #  self.processor = Preprocess(**kwargs)
     #       prep(**kwargs)
     #       self.editor = TokenFlow(config)
            
    
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
            
    
    #def inversion(self):
    #    pass
    
    #def denoising(self):
    #    pass
    
    
    #def __call__(self):
    #    pass
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=["tuneavideo", "tokenflow", "flatten"], default="tokenflow")
    parser.add_argument('--editing_config', type=str, default='./tokenflow/configs/config_pnp.yaml')
    parser.add_argument('--crf_config_path', type=str, default='../data/celebv_bench/test.json')
    
    opts = parser.parse_args()
    prep_config = dict(data_path="", H=512, W=512, save_dir="latents", sd_version='2.1', steps = 250, 
                       batch_size = 24, save_steps = 50, n_frames = 24, inversion_prompt = "")
    prep_config = argparse.Namespace(**prep_config)


    with open(opts.editing_config, "r") as f:
        config = yaml.safe_load(f)
        base_path = config["output_path"]

        
   # video_editor = VideoEditorWrapper(method=opts.method, config=config, **prep_config)
    
    with open(opts.crf_config_path, "r") as f:
        edited_prompts = json.load(f)
    print(prep_config)
    
    for video_id, prompts in tqdm(edited_prompts.items()):
        config["data_path"] = f"../data/celebv_bench/frames/{video_id}"
        #prep_config["data_path"] = f"../data/celebv_bench/frames/{video_id}"
        config["video"][video_id] = {
            "prompt_variants": {
                "factual": prompts["factual"],
                "counterfactual": prompts["counterfactual"]
            }
        }

        videos = []
        text_descriptions = []
        
        for attr in prompts["counterfactual"].keys():
            print(f"Processing Attribute: {attr}")

            config["output_path"] = os.path.join(base_path + "_cfg_scale_" + str(config["guidance_scale"]),
                                                 config["intervention_type"], "interventions", attr,
                                                 video_id, config["video"][video_id]["prompt_variants"]["counterfactual"][attr])
            os.makedirs(config["output_path"], exist_ok=True)
           # print(config["output_path"])
           # break
            config["prompt"] = config["video"][video_id]["prompt_variants"]["counterfactual"][attr]
            config["latents_path"] = prep_config.save_dir
            prep_config.data_path = f"../data/celebv_bench/frames/{video_id}"
            for i in range(2):
                video_editor = VideoEditorWrapper(config, prep_config)
                frames, path = video_editor.run()
                video_to_frames(path, "frames")
                #new_iter_path = path.split("tokenflow_PnP_fps_20.mp4")[0] 
                config["data_path"] = "frames"
                prep_config.data_path = "frames"
                config["prompt"] = "he is bald"
               # break