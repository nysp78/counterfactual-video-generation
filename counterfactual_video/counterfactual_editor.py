import argparse
import json
import yaml
import os
import torch
from methods.tokenflow.util import seed_everything, save_videos_grid__, save_video
from methods.video_editor_wrapper import VideoEditorWrapper
from metrics.clip_text_alignment import ClipTextAlignment
from metrics.clip_consistency import ClipConsistency
from metrics.clip_text_alignment import ClipTextAlignment
from metrics.dover_score import DoverScore
import numpy as np
from tqdm import tqdm
import re, cv2
import textgrad as tg
from textgrad.autograd import MultimodalLLMCall
from textgrad.loss import ImageQALoss
from PIL import Image
import io
from vlm_wrappers.llava import LlavaNext
from vlm_wrappers.generate_prompt import generate_vlm_prompt__, gender_from_text

os.environ["OPENAI_API_KEY"] = "YOUR OPEN_AI KEY"



def extract_interventions(factual: str, counterfactual: str):
    factual = factual.lower()
    counterfactual = counterfactual.lower()
    interventions = []

    # Gender (updated to use "man" / "woman")
    factual_gender = gender_from_text(factual)
    counter_gender = gender_from_text(counterfactual)

    if factual_gender != counter_gender:
        if counter_gender:
            interventions.append(("gender", counter_gender))
        elif factual_gender:
            # If gender is missing in counterfactual, assume flipped
            interventions.append(("gender", "woman" if factual_gender == "man" else "man"))

    # Age
    if "young" in factual and "old" in counterfactual:
        interventions.append(("age", "old"))
    elif "old" in factual and "young" in counterfactual:
        interventions.append(("age", "young"))

    # Beard
    if "beard" in factual and "beard" not in counterfactual:
        interventions.append(("beard", "no-beard"))
    elif "beard" not in factual and "beard" in counterfactual:
        interventions.append(("beard", "beard"))

    # Bald
    if "bald" in factual and "bald" not in counterfactual:
        interventions.append(("bald", "no-bald"))
    elif "bald" not in factual and "bald" in counterfactual:
        interventions.append(("bald", "bald"))

    return interventions

def tensor_to_bytes(tensor_):
    np_data = (tensor_.cpu().numpy() * 255).astype(np.uint8)

    image_pil = Image.fromarray(np_data)
    png_buffer = io.BytesIO()
    image_pil.save(png_buffer, format='PNG')
    png_bytes = png_buffer.getvalue()
    return png_bytes
       
def prompt_optimization_loop(method, config, attr, f_prompt, crf_prompt, max_epochs = 1):
    tg.set_backward_engine("gpt-4-turbo", override=True)
    
    intervened_attrs = extract_interventions(f_prompt , crf_prompt)
    intervened_attrs_values = [item[1] for item in intervened_attrs]
    target_interventions = ", ".join(intervened_attrs_values)
    

    final_frames = torch.zeros(24, 3, 512, 512)
    intervened_attr = attr
    vlm_prompt = generate_vlm_prompt__(intervened_attr, crf_prompt,  target_interventions, causal_dec=True)

    
    question_variable = tg.Variable(vlm_prompt, role_description="instruction to the VLM", requires_grad=False)    
    crf_prompt_var = tg.Variable(crf_prompt, role_description="prompt to optimize", requires_grad=True)
    config["prompt"] = crf_prompt_var.value #response.value
    print("PROPOSED Prompt:", crf_prompt_var.value)
    video_editor = VideoEditorWrapper(config=config, method=method)
    frames , source_frames = video_editor.run()
    source_frames = source_frames.to(device)
    crf_frame_data = tensor_to_bytes(frames[6].permute(1,2,0))
    crf_img_variable = tg.Variable(crf_frame_data, role_description="image to answer a question about", requires_grad=False)
    

    optimizer = tg.TGD(parameters=[crf_prompt_var])
    print("Start PROMPT OPTIMIZATION")
    for i in range(max_epochs):
        optimizer.zero_grad()
        loss_fn = ImageQALoss(
            evaluation_instruction=f'''You are given a counterfactual image generated by a Text-to-Image (T2I) model using the response variable as the prompt.
            1)Evaluate how well the given image aligns with the specified attributes in the response.

            2)Criticize.
            3)Do not describe or modify any other visual elements such as expression, hairstyle, background, clothing, lighting.
            4)The optimized prompt should not have the format of an instruction (e.g generate an image, focus on etc.)
            5)The optimized prompt should be pushed towards the desired interventions
            6)The prompt should have the similar structure as the original prompt
            7)If the alignment is good return: "no_optimization"
            Do not provide a new answer''',
            engine="gpt-4-turbo")
        
              
        loss = loss_fn(question=question_variable, image=crf_img_variable, response=crf_prompt_var)
        
        if ("no_optimization" in loss.value or "The image aligns well" in loss.value) and (not '''not "no_optimization."''' in loss.value):
            final_frames = frames
            print(loss)
            print(crf_prompt_var.gradients)
            print("NO OPTIMIZATION IS NEEDED")
            break
        print(f"Loss iter:{i}", loss)
        loss.backward()
        optimizer.step()
        config["prompt"] = crf_prompt_var.value
        print("OPTIMIZED PROMPT:", crf_prompt_var.value)
        video_editor = VideoEditorWrapper(config=config, method=method)
        frames , _  = video_editor.run()
        crf_frame_data = tensor_to_bytes(frames[6].permute(1,2,0))
        crf_img_variable = tg.Variable(crf_frame_data, role_description="image to answer a question about", requires_grad=False)
        final_frames = frames

            
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
            frames, orig_frames = prompt_optimization_loop(opt.method, config, attr, 
                                         f_prompt, config["prompt"], max_epochs=2)
            
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