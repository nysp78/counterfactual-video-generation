import os
import yaml
import torch
import numpy as np
import subprocess
from pathlib import Path
import yaml
import json
import argparse
from tqdm import tqdm
from torchvision import transforms
import cv2
import logging
from methods.tokenflow.run_tokenflow_pnp import TokenFlow
from methods.tokenflow.util import seed_everything, save_videos_grid, save_videos_grid__
from metrics.clip_consistency import ClipConsistency
from metrics.clip_text_alignment import ClipTextAlignment
from metrics.dover_score import DoverScore





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices =["tuneavideo", "tokenflow"], default="tokenflow")
    parser.add_argument('--base_config_path', type=str, default='methods/tokenflow/configs/config_pnp.yaml')
    parser.add_argument('--crf_config_path', type=str, default='data/celebv_bench/counterfactual_explicit.json')

    opt = parser.parse_args()
    logger = logging.getLogger(__name__)
   # device = "cuda:1"
    with open(opt.base_config_path, "r") as f:
        config = yaml.safe_load(f)
        base_path = config["output_path"]

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