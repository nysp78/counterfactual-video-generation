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
from textgrad.engine_experimental.litellm import LiteLLMEngine
from diffusers import StableDiffusionPipeline
from transformers import AutoModelForCausalLM, AutoTokenizer
from metrics.dover_score import DoverScore
import numpy as np
from tqdm import tqdm
import re, cv2
import textgrad as tg
from PIL import Image
from vlm_wrappers.generate_prompt import gender_from_text, generate_rephrasing_prompt

os.environ["OPENAI_API_KEY"] = "YOUR OPEN_AI KEY"
device="cuda:0"

def load_prompter():
    device = 'cuda:0'
    prompter_model = AutoModelForCausalLM.from_pretrained("microsoft/Promptist").to(device)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return prompter_model, tokenizer

def load_vpo(model_path):
    device = 'cuda:0'
    model = AutoModelForCausalLM.from_pretrained(model_path,  trust_remote_code=True).half().eval().to(device)
# for 8bit
# model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device, load_in_8bit=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    return model, tokenizer

def vpo(model, tokenizer, text):

    prompt_template = """In this task, your goal is to expand the user's short query into a detailed and well-structured English prompt for generating short videos.

Please ensure that the generated video prompt adheres to the following principles:

1. **Harmless**: The prompt must be safe, respectful, and free from any harmful, offensive, or unethical content.
2. **Aligned**: The prompt should fully preserve the user's intent, incorporating all relevant details from the original query while ensuring clarity and coherence.
3. **Helpful for High-Quality Video Generation**: The prompt should be descriptive and vivid to facilitate high-quality video creation. Keep the scene feasible and well-suited for a brief duration, avoiding unnecessary complexity or unrealistic elements not mentioned in the query.

Do not include in the prompt: "Create a video or the duration of the video"
Do not describe other factors of variation.
Describe only the facial characteristics
Generate up to three phrases.
Return just the prompt, nothing else.
User Query:{}

Video Prompt:"""

    messgae = [{'role': 'user', 'content': prompt_template.format(text)}]
    model_inputs = tokenizer.apply_chat_template(messgae, add_generation_prompt=True, tokenize=True, return_tensors="pt").to(device)
    output = model.generate(model_inputs, max_new_tokens=30, do_sample=True, top_p=1.0, temperature=0.7, num_beams=1)
    resp = tokenizer.decode(output[0]).split('<|start_header_id|>assistant<|end_header_id|>')[1].split('<|eot_id|>')[0].strip()

    return resp


def promptist(prompter_model, prompter_tokenizer, plain_text):
    device = "cuda:0"
    input_ids = prompter_tokenizer(plain_text.strip()+" Rephrase:", return_tensors="pt").input_ids
    input_ids = input_ids.to(device)
    eos_id = prompter_tokenizer.eos_token_id
    outputs = prompter_model.generate(input_ids, do_sample=True, max_new_tokens=1024, num_beams=8, num_return_sequences=8, eos_token_id=eos_id, pad_token_id=eos_id, length_penalty=-1.0)
    output_texts = prompter_tokenizer.batch_decode(outputs, skip_special_tokens=True)
    res = output_texts[0].replace(plain_text+" Rephrase:", "").strip()
    return res


def prompt_optimization(method, engine, config, crf_prompt):     

    final_frames = torch.zeros(24, 3, 512, 512)
    if engine["method"] == "gpt-4o":
        llm_prompt = generate_rephrasing_prompt(crf_prompt)
        question_variable = tg.Variable(llm_prompt, role_description="instruction to the LLM", requires_grad=False)    
        model = tg.BlackboxLLM(engine["model"])
        response = model(question_variable)
        response = response.value
    elif engine["method"] == "vpo":
        response = vpo(engine["model"], engine["tokenizer"], crf_prompt)
    else:
        response = promptist(engine["model"], engine["tokenizer"], crf_prompt)
     

    config["prompt"] = response
    print("PROPOSED Prompt:", response)
    video_editor = VideoEditorWrapper(config=config, method=method)
    final_frames , source_frames = video_editor.run()
    source_frames = source_frames.to(device)

            
    return final_frames, source_frames
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=["tuneavideo", "tokenflow", "flatten"], default="flatten")
    parser.add_argument('--prompt_optimization', choices=["gpt-4o", "vpo", "promptist"], default="gpt-4o")
    parser.add_argument('--base_config_path', type=str, default='methods/flatten/configs/config_flatten.yaml')
    parser.add_argument('--crf_config_path', type=str, default='data/celebv_bench/counterfactual_explicit_all.json')
    parser.add_argument('--video_path', type=str, default='data/celebv_bench/videos/')
    parser.add_argument('--data_path', type=str, default='data/celebv_bench/frames/')
    
    
    
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

    if opt.prompt_optimization == "vpo":
        model_path = 'CCCCCC/VPO-5B' #'meta-llama/Meta-Llama-3-8B-Instruct'
        vpo_llm, tokenizer = load_vpo(model_path)
        engine = {"method": "vpo", "model":vpo_llm, "tokenizer": tokenizer}

    elif opt.prompt_optimization == "gpt-4o":
        engine = {"method": "gpt-4o", "model": "gpt-4o", "tokenizer": None}

    else:
        promptist_llm, tokenizer = load_prompter()
        engine = {"method": "promptist", "model": promptist_llm, "tokenizer":tokenizer}

        
    metrics = {}
    video_quality = []  # Measured by DOVER
    text_video_align = {"age": [], "gender": [], "beard": [], "bald": []}  # Measured by CLIP text-frame similarity
    temporal_consistency = [] 
    
    for video_id, prompts in tqdm(edited_prompts.items()):
        config["data_path"] = f"{opt.data_path}/{video_id}" 
        
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

            config["data_path"] = opt.video_path + video_id + ".mp4"
        
        
     
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
            frames, orig_frames = prompt_optimization(opt.method, engine, config, config["prompt"])
            
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
