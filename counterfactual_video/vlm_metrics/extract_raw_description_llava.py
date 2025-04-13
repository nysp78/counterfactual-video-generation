import json
import yaml
import argparse
import torch
from tqdm import tqdm
import cv2
import numpy as np
import os
import io
import numpy as np
import torch
from decord import cpu, VideoReader, bridge
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import os
from torchvision.transforms import Resize, ToPILImage, Compose
os.environ["DISABLE_FLASH_ATTN"] = "1"
import subprocess
#print("mphka")
import io
import numpy as np
import torch
from decord import cpu, VideoReader, bridge
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import imageio
import cv2
import os

MODEL_PATH = "THUDM/cogvlm2-video-llama3-chat"
transform = Compose([ToPILImage(), Resize((512,512))])
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TORCH_TYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[
    0] >= 8 else torch.float16

parser = argparse.ArgumentParser(description="CogVLM2-Video CLI Demo")
parser.add_argument('--quant', type=int, choices=[4, 8], help='Enable 4-bit or 8-bit precision loading', default=0)
args = parser.parse_args([])

def convert_gif_to_mp4_python(gif_path, output_path=None, fps=20):
    # Default output path
    if output_path is None:
        base = os.path.splitext(os.path.basename(gif_path))[0]
        output_path = f"/tmp/{base}.mp4"

    # Read GIF frames using imageio
    gif = imageio.mimread(gif_path)
    if not gif:
        print(f"[ERROR] GIF is empty or corrupt: {gif_path}")
        return None

    height, width, _ = gif[0].shape

    # OpenCV VideoWriter to write MP4
    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),  # codec
        fps,
        (width, height)
    )

    for frame in gif:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)

    out.release()
    return output_path

    
def load_video(video_data, strategy='chat'):
    bridge.set_bridge('torch')
    
    if video_data.endswith(".gif"):
        print(f"Converting GIF to MP4: {video_data}")
        converted_path = convert_gif_to_mp4_python(video_data)
        if converted_path is None or not os.path.exists(converted_path):
            raise RuntimeError("Could not convert GIF to MP4")
        video_data = converted_path
        
    mp4_stream = video_data
    num_frames = 24
    decord_vr = VideoReader(mp4_stream, ctx=cpu(0))
  #  print(decord_vr.shape)
  #  decord_vr_ = [transform(frame.permute(2, 0, 1)) for frame in decord_vr]
  #  for frame in decord_vr:
  #      print(frame.shape)
  #      frame = transform(frame)


    frame_id_list = None
    total_frames = len(decord_vr)
    if strategy == 'base':
        clip_end_sec = 60
        clip_start_sec = 0
        start_frame = int(clip_start_sec * decord_vr.get_avg_fps())
        end_frame = min(total_frames,
                        int(clip_end_sec * decord_vr.get_avg_fps())) if clip_end_sec is not None else total_frames
        frame_id_list = np.linspace(start_frame, end_frame - 1, num_frames, dtype=int)
    elif strategy == 'chat':
        timestamps = decord_vr.get_frame_timestamp(np.arange(total_frames))
        timestamps = [i[0] for i in timestamps]
        max_second = round(max(timestamps)) + 1
        frame_id_list = []
        for second in range(max_second):
            closest_num = min(timestamps, key=lambda x: abs(x - second))
            index = timestamps.index(closest_num)
            frame_id_list.append(index)
            if len(frame_id_list) >= num_frames:
                break

        # while len(frame_id_list) < num_frames:
        #     frame_id_list.append(frame_id_list[-1])

    video_data = decord_vr.get_batch(frame_id_list)
  #  print(video_data.shape)
    video_data = video_data.permute(3, 0, 1, 2)
    return video_data


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    # padding_side="left"
)



model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=TORCH_TYPE,
    trust_remote_code=True
).eval().to(DEVICE)


def predict(prompt, video_data, temperature):
    strategy = 'chat'

    video = load_video(video_data, strategy=strategy)

    history = []
    query = prompt
    inputs = model.build_conversation_input_ids(
        tokenizer=tokenizer,
        query=query,
        images=[video],
        history=history,
        template_version=strategy
    )
    inputs = {
        'input_ids': inputs['input_ids'].unsqueeze(0).to('cuda'),
        'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to('cuda'),
        'attention_mask': inputs['attention_mask'].unsqueeze(0).to('cuda'),
        'images': [[inputs['images'][0].to('cuda').to(TORCH_TYPE)]],
    }
    gen_kwargs = {
        "max_new_tokens": 512,
        "pad_token_id": 128002,
        "top_k": 1,
        "do_sample": False,
        "top_p": 0.1,
        "temperature": temperature,
    }
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
        outputs = outputs[:, inputs['input_ids'].shape[1]:]
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--outputs_path', type=str, default="../outputs/flatten-results_cfg_scale_4.5")
    parser.add_argument('--method', choices=["tuneavideo", "tokenflow", "flatten"], default="flatten")
    parser.add_argument('--intervention_type', choices=["explicit", "implicit", "breaking_causal"], default="breaking_causal")
    parser.add_argument('--crf_config_path', type=str, default='../data/celebv_bench/counterfactual_breaking_causal.json')
 #   
    opt = parser.parse_args()
    with open(opt.crf_config_path, "r") as f:
        crf_prompts = json.load(f)
        
    
    
    descriptions = {}
    #transform = Compose([ToPILImage(), Resize((512,512))])
    for video_id , prompts in tqdm(crf_prompts.items()):
        print("Evaluate video:", video_id)
        descriptions[video_id] = {}
        

        
        #extract factual frame & derive description
       # factual_frame = extract_first_frame(video_path=f"../data/celebv_bench/videos/{video_id}.mp4")
       # factual_frame = transform(factual_frame)
        
        answer_f = predict(prompt="Describe this video in detail", 
                           video_data=f"../data/celebv_bench/videos/{video_id}.mp4", temperature=0.1)
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
                #counterfactual_frame = extract_first_frame(video_path = base_path + "/tokenflow_PnP_fps_20.mp4")
                #print(counterfactual_frame.shape)
                answer_crf = predict(prompt="Describe this video in detail", 
                           video_data= base_path + "/tokenflow_PnP_fps_20.mp4", temperature=0.1)
            
            if opt.method == "tuneavideo":
                answer_crf = predict(prompt="Describe this video in detail", 
                           video_data= base_path + "/edited_fps20.gif", temperature=0.1)
               # print("tuneavideo")
               # counterfactual_frame = extract_first_frame(video_path = base_path + "/edited_fps20.gif")

                
            
           # counterfactual_frame = transform(counterfactual_frame)
           # frames = [factual_frame, counterfactual_frame]
        
          
           # answer_crf = tokenizer.decode(outputs_crf[0].cpu().tolist(), skip_special_tokens=True)
            descriptions[video_id]["counterfactual"][attr] = answer_crf
    
    
    with open(f"raw_descriptions_cogvlm2_{opt.method}_{opt.intervention_type}.json", "w") as json_file:
        json.dump(descriptions, json_file, indent=4)
    print("JSON file saved successfully!")