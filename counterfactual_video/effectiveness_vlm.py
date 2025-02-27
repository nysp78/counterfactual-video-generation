import json
import yaml
import argparse
import torch
from tqdm import tqdm
import cv2
import numpy as np
import os
from torchvision.transforms import Resize, ToPILImage, Compose
from transformers import AutoModelForCausalLM

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

# Example usage
#video_file = "input.mp4"  # Change to your input file
#output_image = "first_frame.jpg"  # Change output file name if needed
#extract_first_frame(video_file, output_image)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type = str, default="deepseek-ai/deepseek-vl2-tiny")
    parser.add_argument('--outputs_path', type=str, default="outputs/tuneavideo-results_cfg_scale_6.5")
    parser.add_argument('--method', choices=["tuneavideo", "tokenflow"], default="tuneavideo")
    parser.add_argument('--intervention_type', choices=["explicit", "implicit", "breaking_causal"], default="breaking_causal")
    parser.add_argument('--questions_path', type=str, default='data/celebv_bench/questions_breaking_causal.json')
 #   parser.add_argument('--crf_config_path', type=str, default='data/celebv_bench/counterfactual_explicit.json')
   # frames = extract_first_frame("outputs/tokenflow-results_cfg_scale_4.5/explicit/interventions/beard/aGRVuZHstlU_0_0/She is old, she has beard./tokenflow_PnP_fps_20.mp4")
   # print(frames.shape)
    opt = parser.parse_args()
    with open(opt.questions_path, "r") as f:
        multiple_choice_questions = json.load(f)
        
    #define the VLM
    model_path = opt.model
    vl_chat_processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(model_path)
    tokenizer = vl_chat_processor.tokenizer

    vl_gpt: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()
        
    #frame = extract_first_frame(video_path=
    #print(frame.shape)
    #pil_img = transforms.ToPILImage()(frame)
    transform = Compose([ToPILImage(), Resize((512,512))])
    effectiveness = {"age": [], "gender": [], "beard": [], "bald": []} 
    for video_id , questions in tqdm(multiple_choice_questions.items()):
        print("Evaluate video:", video_id)

        for attr in questions.keys():
           # print(questions)
            crf_prompt = questions[attr]["prompt"]
            base_path = f"{opt.outputs_path}/{opt.intervention_type}/interventions/{attr}/{video_id}/{crf_prompt}"
            if opt.method == "tokenflow":
                #print("tokenflow")
                counterfactual_frame = extract_first_frame(video_path = base_path + "/tokenflow_PnP_fps_20.mp4")
                #print(counterfactual_frame.shape)
            if opt.method == "tuneavideo":
               # print("tuneavideo")
                counterfactual_frame = extract_first_frame(video_path = base_path + "/edited_fps20.gif")

                
            
            counterfactual_frame = transform(counterfactual_frame)
            
            # iterate through possible questions:
            correct_answers = []
            answers = []
            for q in questions[attr]["questions"]:
                que = q["question"]
                ans1 = q["options"]["A"]
                ans2 = q["options"]["B"]
                correct_answers.append(q["correctAnswer"].lower())
                conversation = [  
                        {
                            "role": "<|User|>",
                            "content": f"<image>{que} , select from: ({ans1}, {ans2}).",
                           # "images": ["./outputs/tokenflow-results_cfg_scale_4.5/explicit/interventions/age/-_B4fiuWwmo_0_1/He is old, he has beard, he is bald./img_ode/00000.png"],
                        },
                        {"role": "<|Assistant|>", "content": ""},
                ]
            
                prepare_inputs = vl_chat_processor(
                                conversations=conversation,
                                images=[counterfactual_frame],
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
                            max_new_tokens=512,
                            do_sample=False,
                            use_cache=True
                        )
                answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
              #  print(f"{prepare_inputs['sft_format'][0]}", answer)
               # print(answer)
                answers.append(answer.lower().replace(".", ""))

            
                
            
           # print(correct_answers, answers)
            effectiveness[attr].append((np.array(answers)==np.array(correct_answers)).sum() / len(answers))
                
            
            
        #print(questions.keys())
       # break
    total_effectiveness = {key: np.array(value).mean() for key, value in effectiveness.items()}
    print(total_effectiveness)
    
