import json
import yaml
import argparse
import torch
from tqdm import tqdm
import cv2
import numpy as np
import os
#import clip
from torchmetrics.text import BLEUScore
from torchvision.transforms import Resize, ToPILImage, Compose
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.nn.functional import cosine_similarity
from sentence_transformers import SentenceTransformer
from transformers import pipeline

def tf_idf_compute(text1 , text2):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])

    # Convert to PyTorch tensors
    tfidf_tensor = torch.tensor(tfidf_matrix.toarray(), dtype=torch.float32)

    # Compute Cosine Similarity using PyTorch
    similarity = cosine_similarity(tfidf_tensor[0].unsqueeze(0), tfidf_tensor[1].unsqueeze(0))
    return similarity


ex_ids = [
    "b5vjXRWFDYI_9_0",
    "bqy2zrVCt8c_21_0",
    "-c-pO7H1Dlc_0_0",
    "eFaetz1BEYg_0_0",
    "g3grlAFSLIE_51_0",
    "1F5naBzNfi8_5_0"
]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type = str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B")
    parser.add_argument('--description_path', type=str, default="./raw_descriptions/raw_descriptions_tuneavideo_explicit.json")
   
    opt = parser.parse_args()
    with open(opt.description_path, "r") as f:
        raw_descriptions = json.load(f)
        
    #define the LLM
    model_name = opt.model
   # tokenizer = AutoTokenizer.from_pretrained(model_name)
   # deepseekr1 = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
   # deepseekr1.to("cuda")

   # messages = [
   # {"role": "user", "content": None}
   #]
    
 
 
    deepseekr1 = pipeline("text-generation", model=model_name, 
                    torch_dtype=torch.bfloat16, device_map = "auto")     
    
    #models to compute semantic similarity
    model = SentenceTransformer("all-MiniLM-L6-v2").to("cuda").eval()
    #clip_model, preprocess = clip.load("ViT-B/32", device="cuda")
    
        
    bleu_scores = []
    tf_idf_sim = []
    semantic_sim = []
   # clip_sim = []
    for video_id , descr in tqdm(raw_descriptions.items()):
        print("Evaluate video:", video_id)
        if video_id == "dV06cJ5Ijv4_10_0" or video_id == "g_Yrrk4eoXk_13_6" or video_id == "-KBCBTt2ldA_0_1"  or video_id=="-_B4fiuWwmo_27_0" or video_id in ex_ids:
          continue
        factual_description = f"""
Remove any references to age, gender (man, woman, he, she), beard, hair (including hairstyle, color, style, and facial hair), and baldness from the following description.
Return only the filtered version of the text, without commentary or formatting.
        
text:\n{descr['factual']}"""
                                  
    #    print("input prompt:", factual_description)
       # print()
       # print()
       # messages[0]["content"] = factual_description
      #  break
        factual_filtered = deepseekr1(factual_description, max_new_tokens=100000, do_sample = False)
     #   print(factual_filtered[0]["generated_text"])
        f_filtered = factual_filtered[0]["generated_text"].split("</think>")[1] #[1]["content"].split("</think>")[1]
       # print(f_filtered)
       
      #  print(f_filtered)
      #  break
        embedding1 = model.encode(f_filtered, convert_to_tensor=True)
        for attr in descr["counterfactual"].keys():
           # print(questions)
            crf_description = descr["counterfactual"][attr]
          #  print(crf_description)
        
            counterfactual_description = f"""
Remove any references to age, gender (man, woman, he, she), beard, hair (including hairstyle color, style, and facial hair), and baldness from the following description.
Return only the filtered version of the text, without commentary or formatting.

text:\n{crf_description}"""

          #  print("input_crf", counterfactual_description)                               
           # messages[0]["content"] = counterfactual_description
            #factual_filtered = deepseekr1(factual_description, max_new_tokens=100)
            counterfactual_filtered = deepseekr1(counterfactual_description, max_new_tokens=100000, do_sample=False)
          #  print("COUNTERFACTUAL TEXT", counterfactual_filtered[0]["generated_text"])
            crf_filtered = counterfactual_filtered[0]["generated_text"].split("</think>")[1]
           # print()
           # print("counterfactual filtered:", crf_filtered)
            pred = [crf_filtered]
            target = [[f_filtered]]
            bleu_score = BLEUScore()(pred, target)
            tf_idf_score = tf_idf_compute(f_filtered, crf_filtered)
            #embedding1 = model.encode(factual_filtered[0]["generated_text"], convert_to_tensor=True)
            embedding2 = model.encode(crf_filtered, convert_to_tensor=True)
            
            #sentence transformer similarity
            similarity = cosine_similarity(embedding1.unsqueeze(0), embedding2.unsqueeze(0))
            
            semantic_sim.append(similarity)
          #  clip_sim.append(clip_similarity)
            bleu_scores.append(bleu_score)
            tf_idf_sim.append(tf_idf_score)
         #   print("video_id:", video_id, bleu_score, " tf-idf score:", 
         #         tf_idf_score, " semantic sim:", similarity)
         #   break
       # break
                
bleu_scores = np.array(bleu_scores)
tf_idf_scores = np.array(tf_idf_sim)
#semantic_sim_scores = np.array(semantic_sim.cpu())
semantic_sim_scores = [score.cpu() for score in semantic_sim]
#clip_semantic_scores = [score.cpu() for score in clip_sim]
semantic_sim_scores = np.array(semantic_sim_scores)
#clip_semantic_scores = np.array(clip_semantic_scores)
print("BlEU score:", np.mean(bleu_scores))
print("TF-IDF score:", np.mean(tf_idf_scores))
print("Semantic sim:", np.mean(semantic_sim_scores))
#print("CLIP semantic similarity:", np.mean(clip_semantic_scores))