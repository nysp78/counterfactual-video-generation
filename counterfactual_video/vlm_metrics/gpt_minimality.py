import json
import yaml
import argparse
import torch
from tqdm import tqdm
import cv2
import numpy as np
import os
import textgrad as tg
import io
from sentence_transformers import SentenceTransformer
from PIL import Image
from textgrad.autograd import MultimodalLLMCall
from torch.nn.functional import cosine_similarity
from torchvision.transforms import Resize, ToPILImage, Compose, ToTensor
from utils import extract_first_frame, extract_nth_frame

os.environ["OPENAI_API_KEY"] = "YOUR OPENAI KEY"

def tensor_to_bytes(tensor_):
    #np_data = tensor_.cpu().numpy().astype(np.uint8)
    np_data = (tensor_.cpu().numpy() * 255).astype()

    image_pil = Image.fromarray(np_data)
    png_buffer = io.BytesIO()
    image_pil.save(png_buffer, format='PNG')
    png_bytes = png_buffer.getvalue()
    return png_bytes

def pil_to_bytes(image_pil):
    png_buffer = io.BytesIO()
    image_pil.save(png_buffer, format='PNG')
    png_bytes = png_buffer.getvalue()
    return png_bytes


    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type = str, default="gpt-4-turbo")
    parser.add_argument('--outputs_path', type=str, default="../path/to/generated/videos")
    parser.add_argument('--method', choices=["tuneavideo", "tokenflow", "flatten"], default="tokenflow")
    parser.add_argument('--crf_config_path', type=str, default='../data/celebv_bench/counterfactual_explicit.json')
 #  
    opt = parser.parse_args()
    with open(opt.crf_config_path, "r") as f:
        crf_prompts = json.load(f)
        
    opt.intervention_type = "explicit"

    
    #define the VLM
    model_path = opt.model
    model = SentenceTransformer("all-MiniLM-L6-v2").to("cuda").eval()
    

    semantic_sim = []
    descriptions = {}
    transform = Compose([ToPILImage(), Resize((512,512))])
    for video_id , prompts in tqdm(crf_prompts.items()):
        print("Evaluate video:", video_id)
        if video_id != "pbkZ9Jp68n8_3_0":
            continue
        descriptions[video_id] = {}

        
        #extract factual frame & derive description
        factual_frame = extract_nth_frame(video_path=f"../data/celebv_bench/videos/{video_id}.mp4", n=6)
        factual_frame = transform(factual_frame)
        
      #  with open(factual_frame, "rb") as f:
      #     factual_frame_data = f.read()
        factual_frame_data = pil_to_bytes(factual_frame)
        
        vlm_prompt ='''Describe this image in detail.
Remove any references to age, gender (man, woman, he, she), beard, hair (including hairstyle, color, style, and facial hair), and baldness from the description.
Return only the filtered version of the text, without commentary or formatting.'''

        image_variable = tg.Variable(factual_frame_data, role_description="image to answer a question about", requires_grad=False)
        question_variable = tg.Variable(vlm_prompt, role_description="instruction to the VLM", requires_grad=False)    
        response = MultimodalLLMCall(model_path)([image_variable, question_variable])
        print("FACTUAL DESCR", response.value)

        descriptions[video_id]["factual"] = response.value 
        embedding1 = model.encode(response.value, convert_to_tensor=True)

        
        
        descriptions[video_id]["counterfactual"] = {}

        for attr in prompts["counterfactual"].keys():
           # print(questions)
           # vl_gpt.cuda().eval()
            crf_prompt = prompts["counterfactual"][attr]
            
            #path of counterfactual
            base_path = f"{opt.outputs_path}/{opt.intervention_type}/interventions/{attr}/{video_id}/{crf_prompt}"
            
            import glob
            if opt.method == "flatten":
                pattern = f"{base_path}/*_ugly, blurry, low res, unrealistic, unaesthetic_7.5.mp4"
                video_path = glob.glob(pattern)[0]
                counterfactual_frame = extract_nth_frame(video_path, n=6)

            if opt.method == "tokenflow":
                counterfactual_frame = extract_nth_frame(video_path = base_path + "/tokenflow_PnP_fps_20.mp4", n=6)
            
            if opt.method == "tuneavideo":
                counterfactual_frame = extract_nth_frame(video_path = base_path + "/edited_fps20.gif", n=6)

                
            
            counterfactual_frame = transform(counterfactual_frame)
            
            counterfactual_frame_data = pil_to_bytes(counterfactual_frame)
            crf_image_variable = tg.Variable(counterfactual_frame_data, role_description="image to answer a question about", requires_grad=False)
            response_crf = MultimodalLLMCall(model_path)([crf_image_variable, question_variable])
            print("COUNTERFACTUAL_DESCR", response_crf.value)

 
                
            descriptions[video_id]["counterfactual"][attr] = response_crf.value
            embedding2 = model.encode(response_crf.value, convert_to_tensor=True)
            
            similarity = cosine_similarity(embedding1.unsqueeze(0), embedding2.unsqueeze(0))
            print(similarity)
            semantic_sim.append(similarity)
        print()



    semantic_sim_scores = [score.cpu() for score in semantic_sim]
    semantic_sim_scores = np.array(semantic_sim_scores)
    print("Semantic sim:", np.mean(semantic_sim_scores))