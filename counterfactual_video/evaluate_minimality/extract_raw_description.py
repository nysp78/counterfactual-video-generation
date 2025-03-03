import json
import yaml
import argparse
import torch
from tqdm import tqdm
import cv2
import json
import numpy as np
import os
from transformers import AutoModelForCausalLM
from torchvision.transforms import Resize, ToPILImage, Compose
from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
from deepseek_vl2.utils.io import load_pil_images


def extract_first_frame(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
   # print(cap)
    
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return
    
    # Read the first frame
    ret, frame = cap.read()
    
    # Release the video capture object
    cap.release()
    return frame



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type = str, default="deepseek-ai/deepseek-vl2-tiny")
    parser.add_argument('--outputs_path', type=str, default="../outputs/tokenflow-results_cfg_scale_4.5")
    parser.add_argument('--method', choices=["tuneavideo", "tokenflow"], default="tokenflow")
    parser.add_argument('--intervention_type', choices=["explicit", "implicit", "breaking_causal"], default="breaking_causal")
    parser.add_argument('--crf_config_path', type=str, default='../data/celebv_bench/counterfactual_breaking_causal.json')
 #   
    opt = parser.parse_args()
    with open(opt.crf_config_path, "r") as f:
        crf_prompts = json.load(f)
        
    
    conversation = [
                {
                    "role": "<|User|>",
                    "content": "<image>Can you describe this image in detail?",
                },
                {"role": "<|Assistant|>", "content": ""},
            ]    
    #define the VLM
    model_path = opt.model
    vl_chat_processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(model_path)
    tokenizer = vl_chat_processor.tokenizer

    vl_gpt: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()
    
    descriptions = {}
    transform = Compose([ToPILImage(), Resize((512,512))])
    for video_id , prompts in tqdm(crf_prompts.items()):
        print("Evaluate video:", video_id)
        descriptions[video_id] = {}

        
        #extract factual frame & derive description
        factual_frame = extract_first_frame(video_path=f"../data/celebv_bench/videos/{video_id}.mp4")
        factual_frame = transform(factual_frame)
        
        prepare_inputs_f = vl_chat_processor(
                                conversations=conversation,
                                images=[factual_frame],
                                force_batchify=True
                            ).to(vl_gpt.device)
        
        inputs_embeds_f = vl_gpt.prepare_inputs_embeds(**prepare_inputs_f)
        
        outputs_f = vl_gpt.language.generate(
                            inputs_embeds=inputs_embeds_f,
                            attention_mask=prepare_inputs_f.attention_mask,
                            pad_token_id=tokenizer.eos_token_id,
                            bos_token_id=tokenizer.bos_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                            max_new_tokens=128,
                            do_sample=False,
                            use_cache=True
                        )
        answer_f = tokenizer.decode(outputs_f[0].cpu().tolist(), skip_special_tokens=True)
        descriptions[video_id]["factual"] = answer_f #store the factual description
        
        
        descriptions[video_id]["counterfactual"] = {}

        for attr in prompts["counterfactual"].keys():
           # print(questions)
           # vl_gpt.cuda().eval()
            crf_prompt = prompts["counterfactual"][attr]
            
            #path of counterfactual
            base_path = f"{opt.outputs_path}/{opt.intervention_type}/interventions/{attr}/{video_id}/{crf_prompt}"
            
            if opt.method == "tokenflow":
                #print("tokenflow")
                counterfactual_frame = extract_first_frame(video_path = base_path + "/tokenflow_PnP_fps_20.mp4")
                #print(counterfactual_frame.shape)
            
            if opt.method == "tuneavideo":
               # print("tuneavideo")
                counterfactual_frame = extract_first_frame(video_path = base_path + "/edited_fps20.gif")

                
            
            counterfactual_frame = transform(counterfactual_frame)
           # frames = [factual_frame, counterfactual_frame]
        
            prepare_inputs_crf = vl_chat_processor(
                                conversations=conversation,
                                images=[counterfactual_frame],
                                force_batchify=True
                            ).to(vl_gpt.device)
        
            inputs_embeds_crf = vl_gpt.prepare_inputs_embeds(**prepare_inputs_crf)
        
            outputs_crf = vl_gpt.language.generate(
                            inputs_embeds=inputs_embeds_crf,
                            attention_mask=prepare_inputs_crf.attention_mask,
                            pad_token_id=tokenizer.eos_token_id,
                            bos_token_id=tokenizer.bos_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                            max_new_tokens=128,
                            do_sample=False,
                            use_cache=True
                        )
            answer_crf = tokenizer.decode(outputs_crf[0].cpu().tolist(), skip_special_tokens=True)
            descriptions[video_id]["counterfactual"][attr] = answer_crf
    
    
    with open(f"raw_descriptions_{opt.method}_{opt.intervention_type}.json", "w") as json_file:
        json.dump(descriptions, json_file, indent=4)
    print("JSON file saved successfully!")