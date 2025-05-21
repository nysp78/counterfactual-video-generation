import argparse
import json
import yaml
import os
import torch
from methods.tokenflow.util import seed_everything, save_videos_grid__
from methods.video_editor_wrapper import VideoEditorWrapper
from metrics.clip_text_alignment import ClipTextAlignment
from metrics.clip_consistency import ClipConsistency
from metrics.clip_text_alignment import ClipTextAlignment
from metrics.dover_score import DoverScore
import numpy as np
from tqdm import tqdm
import re, cv2
import textgrad as tg
from PIL import Image
from vlm_wrappers.generate_prompt import gender_from_text, generate_rephrasing_prompt

os.environ["OPENAI_API_KEY"] = "YOUR OPEN_AI KEY"


     
def prompt_optimization_loop(method, config, crf_prompt):
    #tg.set_backward_engine("gpt-4-turbo", override=True)

    #score = -1
    #threshold = 20.0
    final_frames = torch.zeros(24, 3, 512, 512)
    
    llm_prompt = generate_rephrasing_prompt(crf_prompt)
   # print(vlm_prompt)

    question_variable = tg.Variable(llm_prompt, role_description="instruction to the VLM", requires_grad=False)    
    model = tg.BlackboxLLM("gpt-4o")
    response = model(question_variable)

    config["prompt"] = response.value
    print("PROPOSED Prompt:", response.value)
    video_editor = VideoEditorWrapper(config=config, method=method)
    final_frames , source_frames = video_editor.run()
    source_frames = source_frames.to(device)

            
    return final_frames, source_frames
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=["tuneavideo", "tokenflow", "flatten"], default="tokenflow")
    parser.add_argument('--base_config_path', type=str, default='methods/tokenflow/configs/config_pnp.yaml')
    parser.add_argument('--crf_config_path', type=str, default='data/celebv_bench/counterfactual_explicit.json')
    
    
    
    device = "cuda"
    opt = parser.parse_args()

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
    temporal_consistency = [] 
    
    for video_id, prompts in tqdm(edited_prompts.items()):
        config["data_path"] = f"data/celebv_bench/frames/{video_id}"
        
        #factual_frame = config["data_path"]+"/00007.jpg"
        config["video"][video_id] = {
            "prompt_variants": {
                "factual": prompts["factual"],
                "counterfactual": prompts["counterfactual"]
            }
        }
        
        videos = []
        text_descriptions = []

        f_prompt = config["video"][video_id]["prompt_variants"]["factual"]
        

        
        if opt.method == "tuneavideo":
            print("Loading Tune-A-Video checkpoints!")
            config["checkpoint_dir"] = os.path.join(base_ckpt_path, video_id)
            trained_videos = os.listdir("methods/tuneavideo/checkpoints")
            if video_id not in trained_videos:
                print("Video all ready trained!")
                continue
              
        if opt.method == "flatten":

            config["data_path"] = f"data/celebv_bench/videos/{video_id}.mp4"
        
        
     
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
            
            #if opt.method == "tokenflow":
            os.makedirs(grids_path, exist_ok=True)
            frames, orig_frames = prompt_optimization_loop(opt.method, config, config["prompt"])
            
            #set again the original factual prompt    
            config["prompt"] = config["video"][video_id]["prompt_variants"]["counterfactual"][attr]

               
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