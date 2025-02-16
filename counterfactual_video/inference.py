import os
import yaml
import torch
import numpy as np
import subprocess
from pathlib import Path
import yaml
import json
import argparse
import gc
from tqdm import tqdm
from torchvision import transforms
import cv2
import logging
from methods.tuneavideo.models.modeling_utils import ModelMixin
from methods.tuneavideo.pipelines.pipeline_tuneavideo import TuneAVideoPipeline
from methods.tuneavideo.models.unet import UNet3DConditionModel
from methods.tokenflow.run_tokenflow_pnp import TokenFlow
from methods.tokenflow.util import seed_everything, save_videos_grid__
from metrics.clip_consistency import ClipConsistency
from metrics.clip_text_alignment import ClipTextAlignment
from metrics.dover_score import DoverScore





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices =["tuneavideo", "tokenflow"], default="tuneavideo")
    parser.add_argument('--base_config_path', type=str, default='methods/tuneavideo/configs/config_tune_eval.yaml')
    parser.add_argument('--crf_config_path', type=str, default='data/celebv_bench/counterfactual_explicit.json')

    opt = parser.parse_args()
    logger = logging.getLogger(__name__)
   # device = "cuda:1"
    with open(opt.base_config_path, "r") as f:
        config = yaml.safe_load(f)
        base_path = config["output_path"]
        base_ckpt_path = config["checkpoint_dir"]

    with open(opt.crf_config_path, "r") as f:
        edited_prompts = json.load(f)

    seed_everything(config["seed"])
    
    metrics = {}
    video_quality = [] #measured by dover
    text_video_align = {"age":[], "gender":[], "beard":[], "bald":[]} #measured by clip sim between text-frames
    temporal_consistency = [] #measured by clip sim between consecutive frames

    for video_id, prompts in tqdm(edited_prompts.items()):
        config["data_path"] = f"data/celebv_bench/frames/{video_id}"
        config["video"][video_id] = {
                         # Assumes your videos are in "data/video_id"
                       "prompt_variants": {
                                             "factual": prompts["factual"],
                                             "counterfactual": prompts["counterfactual"]
                                         }
                   }
        

        videos = []
        text_descriptions = []
        for attr in prompts["counterfactual"].keys():
            config["output_path"] = os.path.join(config["output_path"] + "_cfg_scale_" + str(config["guidance_scale"]),
                                                 config["intervention_type"], "interventions", attr,
                                             video_id, config["video"][video_id]["prompt_variants"]["counterfactual"][attr])
            os.makedirs(config["output_path"], exist_ok=True)
            config["prompt"] = config["video"][video_id]["prompt_variants"]["counterfactual"][attr]
            assert os.path.exists(config["data_path"]), "Data path does not exist"
           # print(config)

            if opt.method == "tokenflow":
                grids_path =  os.path.join(base_path + "_cfg_scale_" + str(config["guidance_scale"]), config["intervention_type"])
                os.makedirs(grids_path, exist_ok=True)

                pipeline = TokenFlow(config)
                orig_frames = pipeline.frames

                frames = pipeline.edit_video()
            
            if opt.method == "tuneavideo":
                config["checkpoint_dir"] = os.path.join(base_ckpt_path, video_id)
               # print(config["checkpoint_dir"])
                config["checkpoint_dir"] = "methods/tuneavideo/checkpoints/-_zyvfId578_12_1"
                
                torch.cuda.empty_cache()
                gc.collect()
                config["checkpoint_dir"] = "methods/tuneavideo/checkpoints/-_zyvfId578_12_1"
                unet = UNet3DConditionModel.from_pretrained(config["checkpoint_dir"], subfolder='unet', torch_dtype=torch.float16).to('cuda')
                pipe = TuneAVideoPipeline.from_pretrained(config["pretrained_model_path"], unet=unet, torch_dtype=torch.float16).to("cuda")

                #pipe.enable_xformers_memory_efficient_attention
                pipe.enable_vae_slicing()
                ddim_inv_latent = torch.load(config["checkpoint_dir"]+"/inv_latents/ddim_latent-1.pt").to(torch.float16)
                with torch.no_grad():
                    frames = pipe(config["prompt"], latents=ddim_inv_latent, video_length=config["video_length"], height=512, width=512, 
                              num_inference_steps=50, guidance_scale=config["guidance_scale"]).videos
                    #save_videos_grid__(frames, f"{output_dir}/samples/sample-{global_step}/{prompt}.gif")
                    save_videos_grid__(frames, f'{config["output_path"]}/edited_fps20.gif', fps=20)

            #calculate metrics
            dover_score = DoverScore(device=config["device"]).evaluate(frames)
            clip_score_temp = ClipConsistency(device=config["device"]).evaluate(frames)
            clip_score_align = ClipTextAlignment(device=config["device"]).evaluate(frames, config["prompt"])
            video_quality.append(dover_score)
            temporal_consistency.append(clip_score_temp)
            text_video_align[attr].append(clip_score_align)
            prompt = config["prompt"]
            print(f"{video_id}, {prompt}: DOVER:{dover_score}, Text-Video:{clip_score_align}, Temporal:{clip_score_temp}")

            videos.append(frames.permute(1,0,2,3).unsqueeze(0).cpu())
            text_descriptions.append(config["prompt"]) #add the counterfactual prompt
            config["output_path"] = base_path #init base config output



        #plot the videos grid
        videos = [orig_frames.permute(1,0,2,3).unsqueeze(0).cpu()] + videos
        text_descriptions = [config["video"][video_id]["prompt_variants"]["factual"]] + text_descriptions
        videos = torch.concat(videos)
     #   print(videos.shape)
        save_path = grids_path + "/" + f'{video_id}.gif'
     #   print(save_path)
        save_videos_grid__(videos, save_path, text_descriptions)
        videos = [] #empty list for grid plot
        text_descriptions = []

total_text_video_alignment = {key:np.array(value).mean() for key, value in text_video_align.items()}


print("DOVER score:", np.array(video_quality).mean())
print("Text-to-Video alignment CLIP:", total_text_video_alignment)
print("Temporal consistency CLIP:", np.array(temporal_consistency).mean())