import json
import yaml
import argparse
import torch
from tqdm import tqdm
import cv2
import numpy as np
import os
from torchmetrics.text import BLEUScore
from torchvision.transforms import Resize, ToPILImage, Compose
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline
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
    parser.add_argument('--outputs_path', type=str, default="outputs/tuneavideo-results_cfg_scale_4.5")
    parser.add_argument('--method', choices=["tuneavideo", "tokenflow"], default="tuneavideo")
    parser.add_argument('--intervention_type', choices=["explicit", "implicit", "breaking_causal"], default="explicit")
    parser.add_argument('--crf_config_path', type=str, default='data/celebv_bench/counterfactual_explicit.json')
 #   
    opt = parser.parse_args()
    with open(opt.crf_config_path, "r") as f:
        crf_prompts = json.load(f)
        
    #define the LLM
    #deepseekr1 = pipeline("text-generation", model="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B", 
    #                    torch_dtype=torch.float16)
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    deepseekr1 = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)  # Ensure model is on GPU
        
    #define the VLM
    model_path = opt.model
    vl_chat_processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(model_path)
    tokenizer = vl_chat_processor.tokenizer

    vl_gpt: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    vl_gpt = vl_gpt.to(torch.bfloat16)
        
    #frame = extract_first_frame(video_path=
    #print(frame.shape)
    #pil_img = transforms.ToPILImage()(frame)
    transform = Compose([ToPILImage(), Resize((512,512))])
    for video_id , prompts in tqdm(crf_prompts.items()):
        print("Evaluate video:", video_id)
        
        #extract factual frame
        factual_frame = extract_first_frame(video_path=f"data/celebv_bench/videos/{video_id}.mp4")
        factual_frame = transform(factual_frame)

        for attr in prompts["counterfactual"].keys():
           # print(questions)
            vl_gpt.cuda().eval()
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
            frames = [factual_frame, counterfactual_frame]
            
           
            conversations = [
                                [{
                                    "role": "<|User|>",
                                    "content": "<image>Can you describe this image in detail?",
                                   #  "images": ["/content/factual.png"],
                                },
                                {"role": "<|Assistant|>", "content": ""}],

                                [{
                                    "role": "<|User|>",
                                    "content": "<image>Can you describe this image in detail?",
                                   # "images": ["/content/crf.png"],
                                },
                                 {"role": "<|Assistant|>", "content": ""}]
                            ]
            
            answers = []
            for i, conversation in enumerate(conversations):
                prepare_inputs = vl_chat_processor(
                                conversations=conversation,
                                images=[frames[i]],
                                force_batchify=True
                            ).to(vl_gpt.device)
                
                inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

                # run the model to get the response
                outputs = vl_gpt.language.generate(
                            inputs_embeds=inputs_embeds,
                            attention_mask=prepare_inputs.attention_mask,
                            pad_token_id=tokenizer.eos_token_id,
                            bos_token_id=tokenizer.bos_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                            max_new_tokens=128,
                            do_sample=False,
                            use_cache=True
                        )
                answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
                answers.append(answer)
            vl_gpt.to("cpu")
            deepseekr1.to("cuda")
            factual_description = f'''Describe the text below by excluding the factors: age, gender, beard, baldness.\n text:\n{answer[0]}'''
            counterfactual_description = f'''Describe the text below by excluding the factors: age, gender, beard, baldness.\n text:\n{answer[1]}'''
            factual_description = tokenizer(factual_description, return_tensors="pt").to("cuda")
            counterfactual_description = tokenizer(counterfactual_description, return_tensors="pt").to("cuda")
            
            factual_filtered = deepseekr1.generate(**factual_description, max_new_tokens=100)
            counterfactual_filtered = deepseekr1.generate(**counterfactual_description, max_new_tokens=100)
            #tokenizer.decode(counterfactual_filtered[0], skip_special_tokens=True)
            pred = [tokenizer.decode(counterfactual_filtered[0], skip_special_tokens=True)]
            target = [[tokenizer.decode(factual_filtered[0], skip_special_tokens=True)]]
            bleu_score = BLEUScore()(pred, target)
            print(bleu_score)
            deepseekr1.to("cpu")
                
            break

       # print(answers)
        break