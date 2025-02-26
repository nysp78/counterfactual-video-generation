import os
import argparse
import yaml
import json
import torch
import gc
import sys
from pathlib import Path
from methods.tuneavideo.train_tuneavideo import train

def clean_memory():
    """Free up GPU memory after training each video."""
    torch.cuda.empty_cache()
    gc.collect()
    print("GPU memory cleaned!")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--base_config', type=str, default="methods/tuneavideo/configs/config_tune.yaml")
    parser.add_argument('--source_prompts', type=str, default="data/celebv_bench/source_prompts.json")
    parser.add_argument('--data_dir', type=str, default='data/celebv_bench/frames')

    params = parser.parse_args()
    
    with open(os.path.join(params.base_config), "r") as file:
        config = yaml.safe_load(file)  # Load Tune-A-Video configuration
    
    with open(os.path.join(params.source_prompts), "r") as file:
        source_prompts = json.load(file)  # Load factual prompts

    video_files = [f for f in os.listdir(params.data_dir) if os.path.isdir(os.path.join(params.data_dir, f))]
    base_ckpt_path = config["checkpoint_dir"]

    for idx, video in enumerate(video_files):
        trained_videos = os.listdir("methods/tuneavideo/checkpoints_v2")
        if video in trained_videos:
            print("Video all ready trained!")
            continue
        print(f"Training on video {idx + 1}/{len(video_files)}: {video}")

        config["train_data"]["video_path"] = os.path.join(params.data_dir, video)
        config["train_data"]["prompt"] = source_prompts[video]
        config["checkpoint_dir"] = os.path.join(base_ckpt_path, video)

        try:
            train(**config)
            print(f" Tuning of video: {video} is completed!\n")
        except torch.cuda.OutOfMemoryError:
            print(f" Out of memory on video: {video}. Retrying with smaller batch size...")
            torch.cuda.empty_cache()
            gc.collect()
            config["train_batch_size"] = max(1, config["train_batch_size"] // 2)
            train(**config)

        #Free memory after training each video
        clean_memory()

    print("Tuning of all videos completed successfully!")
