import os
import yaml
import torch
import numpy as np
import subprocess
from pathlib import Path
import json
import argparse
import gc
from tqdm import tqdm
import time
from torchvision import transforms
import cv2
import logging
from methods.tuneavideo.models.modeling_utils import ModelMixin
from methods.tuneavideo.pipelines.pipeline_tuneavideo import TuneAVideoPipeline
from methods.tuneavideo.models.unet import UNet3DConditionModel
from methods.tuneavideo.data.dataset import TuneAVideoDataset
from methods.tokenflow.run_tokenflow_pnp import TokenFlow
from methods.flatten.inference import run_flatten_pipeline
from methods.tokenflow.util import seed_everything, save_videos_grid__, save_video
from metrics.clip_consistency import ClipConsistency
from metrics.clip_text_alignment import ClipTextAlignment
from metrics.dover_score import DoverScore
from huggingface_hub import snapshot_download
from diffusers import StableDiffusionPipeline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=["tuneavideo", "tokenflow", "flatten"], default="tokenflow")
    parser.add_argument('--base_config_path', type=str, default='methods/tokenflow/configs/config_pnp.yaml')
    parser.add_argument('--crf_config_path', type=str, default='data/celebv_bench/counterfactual_explicit.json')

    opt = parser.parse_args()
    logger = logging.getLogger(__name__)

    #Load Config
    with open(opt.base_config_path, "r") as f:
        config = yaml.safe_load(f)
        base_path = config["output_path"]
        if "checkpoint_dir" in config.keys():
            base_ckpt_path = config["checkpoint_dir"]

    with open(opt.crf_config_path, "r") as f:
        edited_prompts = json.load(f)

    seed_everything(config["seed"])

    metrics = {}
    video_quality = []  # Measured by DOVER
    text_video_align = {"age": [], "gender": [], "beard": [], "bald": []}  # Measured by CLIP text-frame similarity
    temporal_consistency = []  # Measured by CLIP frame consistency
    #download stable diffusion pipeline
    model_key = snapshot_download(config["pretrained_model_path"])

    for video_id, prompts in tqdm(edited_prompts.items()):
        config["data_path"] = f"data/celebv_bench/frames/{video_id}"
        config["video"][video_id] = {
            "prompt_variants": {
                "factual": prompts["factual"],
                "counterfactual": prompts["counterfactual"]
            }
        }

        videos = []
        text_descriptions = []

        #Load Tune-A-Video Model on GPU
        if opt.method == "flatten":
           # print(config)
           #load the .mp4 for flatten
            config["data_path"] = f"data/celebv_bench/videos/{video_id}.mp4"
            #break
           # pass
        if opt.method == "tuneavideo":
            print("Loading Tune-A-Video checkpoints!")
            config["checkpoint_dir"] = os.path.join(base_ckpt_path, video_id)
            trained_videos = os.listdir("methods/tuneavideo/checkpoints")
            if video_id not in trained_videos:
                print("Video all ready trained!")
                continue

            train_dataset = TuneAVideoDataset(video_path=config["data_path"], n_sample_frames=config["video_length"])
            print("Tune-A-Video dataset created!")

            unet = UNet3DConditionModel.from_pretrained(config["checkpoint_dir"], subfolder='unet', torch_dtype=torch.float16).to(device)
            print("UNET Loaded!")

            unet.enable_xformers_memory_efficient_attention()

            pipe = TuneAVideoPipeline.from_pretrained(model_key, unet=unet, torch_dtype=torch.float16).to(device)
        
            print("Tune-A-Video pipeline loaded!")


            pipe.enable_vae_slicing()
            ddim_inv_latent = torch.load(config["checkpoint_dir"] + "/inv_latents/ddim_latent-450.pt").to(torch.float16).to(device)
            print("Latents loaded!")

        for attr in prompts["counterfactual"].keys():
            print(f"Processing Attribute: {attr}")

            config["output_path"] = os.path.join(base_path + "_cfg_scale_" + str(config["guidance_scale"]),
                                                 config["intervention_type"], "interventions", attr,
                                                 video_id, config["video"][video_id]["prompt_variants"]["counterfactual"][attr])
            os.makedirs(config["output_path"], exist_ok=True)
            print(config["output_path"])
           # break
            config["prompt"] = config["video"][video_id]["prompt_variants"]["counterfactual"][attr]
            assert os.path.exists(config["data_path"]), "Data path does not exist"

            grids_path = os.path.join(base_path + "_cfg_scale_" + str(config["guidance_scale"]), config["intervention_type"])
            
            if opt.method == "flatten":
                frames, orig_frames = run_flatten_pipeline(prompt = config["prompt"], neg_prompt = config["neg_prompt"],
                                              guidance_scale = config["guidance_scale"], video_path=config["data_path"],
                                              output_path = config["output_path"])
                orig_frames = orig_frames / 255
                #frames = frames.squeeze(0)
               # orig_frames = orig_frames.squeeze(0)
                 
              #  print(frames.shape, orig_frames.shape)
            #    break
            #    pass

            #TokenFlow Processing
            if opt.method == "tokenflow":
                os.makedirs(grids_path, exist_ok=True)
                pipeline = TokenFlow(config)
                orig_frames = pipeline.frames.to(device)  # Ensure frames are on GPU
                frames, _ = pipeline.edit_video()
              #  print(frames.shape)

            # Tune-A-Video Processing
            if opt.method == "tuneavideo":
                print("Begin Tune-A-Video Processing!")

                orig_frames = torch.tensor(train_dataset.__getitem__()["pixel_values"]).to(device)
                orig_frames = (orig_frames + 1) / 2  # Normalize to [0,1]

                with torch.no_grad():

                    frames = pipe(config["prompt"], latents=ddim_inv_latent, video_length=config["video_length"],
                                  height=512, width=512, num_inference_steps=50, guidance_scale=config["guidance_scale"]).videos.to(device)

                    save_videos_grid__(frames, f'{config["output_path"]}/edited_fps20.gif', fps=20)
                    print("Frames generated successfully!")
                    frames = frames.permute(0, 2, 1, 3, 4).squeeze(0)
                    print(frames.shape)

            #break
            # Compute Metrics on GPU
            dover_score = DoverScore(device=device).evaluate(frames.to(device))
            clip_score_temp = ClipConsistency(device=device).evaluate(frames.to(device))
            clip_score_align = ClipTextAlignment(device=device).evaluate(frames.to(device), config["prompt"])

            video_quality.append(dover_score)
            temporal_consistency.append(clip_score_temp)
            text_video_align[attr].append(clip_score_align)

            print(f"{video_id}, {config['prompt']}: DOVER:{dover_score}, Text-Video:{clip_score_align}, Temporal:{clip_score_temp}")

            videos.append(frames.permute(1, 0, 2, 3).unsqueeze(0).cpu())  # Convert to CPU before saving
            text_descriptions.append(config["prompt"])
            config["output_path"] = base_path  # Reset base config output


        #Save Video Grid
        videos = [orig_frames.permute(1, 0, 2, 3).unsqueeze(0).cpu()] + videos
        text_descriptions = [config["video"][video_id]["prompt_variants"]["factual"]] + text_descriptions
        videos = torch.concat(videos)
        save_path = grids_path + "/" + f'{video_id}.gif'
        save_videos_grid__(videos, save_path, text_descriptions)

        videos = []  # Reset video list
        text_descriptions = []

#Compute Final Scores
total_text_video_alignment = {key: np.array(value).mean() for key, value in text_video_align.items()}
print("DOVER score:", np.array(video_quality).mean())
print("Text-to-Video alignment CLIP:", total_text_video_alignment)
print("Temporal consistency CLIP:", np.array(temporal_consistency).mean())