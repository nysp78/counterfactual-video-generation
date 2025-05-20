import json
import yaml
import argparse
import torch
from tqdm import tqdm

import cv2
import numpy as np
from torchvision.transforms import Resize, ToPILImage, Compose
from calculate_fvd import calculate_fvd
from calculate_psnr import calculate_psnr
from calculate_ssim import calculate_ssim
from calculate_lpips import calculate_lpips

#from moviepy.editor import VideoFileClip

def load_video(video_path, num_frames=24, img_size=512):
    """
    Load a video from a file (MP4 or GIF), resize it, and normalize pixel values.
    """
    frames = []

    # Check file extension
#    if video_path.endswith(".gif"):
#        # Load GIF using MoviePy
#        clip = VideoFileClip(video_path)
#        for frame in clip.iter_frames(fps=clip.fps, dtype="uint8"):
#            frames.append(frame)
#    else:
        # Load MP4 using OpenCV
    cap = cv2.VideoCapture(video_path)
    while len(frames) < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        frames.append(frame)
    cap.release()

    if len(frames) == 0:
        raise ValueError(f"Could not load any frames from {video_path}")

    # Resize frames to (224, 224)
    frames = [cv2.resize(frame, (img_size, img_size)) for frame in frames]

    # If fewer frames than required, duplicate last frame
    while len(frames) < num_frames:
        frames.append(frames[-1])

    # Convert to tensor array and normalize to [-1, 1]
    video_array = np.array(frames, dtype=np.float32) #torch.tensor(frames, requires_grad=False)#np.array(frames, dtype=np.float32)
   # video_array = (video_array / 127.5) - 1.0  # Normalize
    video_array = video_array / 255.0
   # print(np.max(video_array), np.mean(video_array))

    return video_array

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--outputs_path', type=str, default="../../outputs_rephrasing_LLM_v2/flatten-results_cfg_scale_12.5")
    parser.add_argument('--method', choices=["tuneavideo", "tokenflow", "flatten"], default="flatten")
    parser.add_argument('--intervention_type', choices=["explicit", "implicit", "breaking_causal"], default="explicit")
    parser.add_argument('--crf_config_path', type=str, default='../../data/celebv_bench/counterfactual_explicit.json')
    
    
    opt = parser.parse_args()
    with open(opt.crf_config_path, "r") as f:
        crf_prompts = json.load(f)
        
    
   # transform = Compose([ToPILImage(), Resize((512,512))])
    fvd_scores = []
    lpips_scores = []
    for video_id , prompts in tqdm(crf_prompts.items()):
        print("Evaluate video:", video_id)
       # descriptions[video_id] = {}

        
        #extract factual frame & derive description
        factual_video = load_video(video_path=f"../../data/celebv_bench/videos/{video_id}.mp4")
      #  factual_video = factual_video.permute(0,3,1,2)
        factual_video = torch.from_numpy(factual_video).permute(0,3,1,2).unsqueeze(0)
       # print(factual_video.shape)
       # break
      #  factual_video = transform(load_video)
      
        for attr in prompts["counterfactual"].keys():
            
            crf_prompt = prompts["counterfactual"][attr]
            
            #path of counterfactual
            base_path = f"{opt.outputs_path}/{opt.intervention_type}/interventions/{attr}/{video_id}/{crf_prompt}"
            import glob
            if opt.method == "flatten":
                pattern = f"{base_path}/*_ugly, blurry, low res, unrealistic, unaesthetic_12.5.mp4"
                video_path = glob.glob(pattern)[0]
                #counterfactual_video = load_video(video_path = base_path + f"/{crf_prompt[:10]}_ugly, blurry, low res, unrealistic, unaesthetic_12.5.mp4")
                counterfactual_video = load_video(video_path = video_path)

                
            if opt.method == "tokenflow":
                #print("tokenflow")
                counterfactual_video = load_video(video_path = base_path + "/tokenflow_PnP_fps_20.mp4")
                #print(counterfactual_frame.shape)
            
            if opt.method == "tuneavideo":
               # print("tuneavideo")
                counterfactual_video = load_video(video_path = base_path + "/edited_fps20.gif")
             
            counterfactual_video = torch.from_numpy(counterfactual_video).permute(0,3,1,2).unsqueeze(0)
          #  print("counterfactual video:", counterfactual_video.max())   
            #print(counterfactual_video.shape)
            
            fvd = calculate_fvd(factual_video, counterfactual_video, device="cuda", method='styleganv', only_final=True)
            lpips = calculate_lpips(factual_video, counterfactual_video, device="cuda", only_final=True)
            print(fvd, lpips)
           # print(lpips)
            fvd_scores.append(fvd["value"][0])
            lpips_scores.append(lpips["value"][0])
           # break
      #  break
            
    print("FVD score:", np.array(fvd_scores).mean())
    print("LPIPS score:", np.array(lpips_scores).mean())
    
    
    