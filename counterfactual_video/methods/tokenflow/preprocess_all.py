import os
import argparse
import json
from pathlib import Path
from preprocess import prep  # Import the preprocessing function from your script


def preprocess_all_videos(data_dir, params):
    """
    extract latents and the self-attention maps from every video
    """

    # List all video files in the `data_dir`
    #print(params)
    with open(os.path.join(params.data_dir, "source_prompts.json"), "r") as file:
        source_prompts = json.load(file) #factual prompts

    #video_files = [f for f in os.listdir(data_dir + "/videos") if f.endswith(('.mp4', '.avi', '.mov'))]  # Adjust formats if needed
    video_files = [f for f in os.listdir(data_dir + "/frames") if os.path.isdir(os.path.join(data_dir + "/frames", f))]
  #  print(video_files)

    print(f"Found {len(video_files)} videos. Processing...")

    for video in video_files:
        video_path = os.path.join(data_dir, "frames/" + video)
      #  print(video_path)

        params.data_path = video_path
        params.inversion_prompt = source_prompts[video_path.split("/")[-1]] #use as key the video identifier 
        print(params.data_path, params.inversion_prompt)

        prep(params)

    print("\n All videos processed successfully!")

    return

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../../data/celebv_bench', help="Directory containing video files")
    parser.add_argument('--H', type=int, default=512)
    parser.add_argument('--W', type=int, default=512)
    parser.add_argument('--save_dir', type=str, default='inverted_latents')
    parser.add_argument('--sd_version', type=str, default='2.1', choices=['1.5', '2.0', '2.1', 'ControlNet', 'depth'])
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--save_steps', type=int, default=50)
    parser.add_argument('--n_frames', type=int, default=24)
    parser.add_argument('--inversion_prompt', type=str, default=' ')  # Can be dynamically set per video

    params = parser.parse_args()

    # Run the function to process multiple videos
    preprocess_all_videos(params.data_dir, params)