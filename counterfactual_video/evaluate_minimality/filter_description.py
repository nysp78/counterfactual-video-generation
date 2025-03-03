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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type = str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B")
    parser.add_argument('--description_path', type=str, default="raw_descriptions_tokenflow_breaking_causal.json")
   
    opt = parser.parse_args()
    with open(opt.description_path, "r") as f:
        raw_descriptions = json.load(f)
        
    #define the LLM
    model_name = opt.model
    deepseekr1 = pipeline("text-generation", model=model_name, 
                    torch_dtype=torch.float16, device = 0)     
    
    #model to compute semantic similarity
    model = SentenceTransformer("all-MiniLM-L6-v2").to("cuda").eval()
    
        
    bleu_scores = []
    tf_idf_sim = []
    semantic_sim = []
    for video_id , descr in tqdm(raw_descriptions.items()):
        print("Evaluate video:", video_id)
        factual_description = f'''Describe the text below by excluding the factors: 
                                  age, gender, beard, baldness, hair.
                                  \n text:\n{descr["factual"]}'''
        
        factual_filtered = deepseekr1(factual_description, max_new_tokens=100)

        for attr in descr["counterfactual"].keys():
           # print(questions)
            crf_description = descr["counterfactual"][attr]
        
            counterfactual_description = f'''Describe the text below by excluding the factors: 
                                             age, gender, beard, baldness, hair.\n 
                                             text:\n{crf_description}'''
              
            #factual_filtered = deepseekr1(factual_description, max_new_tokens=100)
            counterfactual_filtered = deepseekr1(counterfactual_description, max_new_tokens=100)
            pred = [counterfactual_filtered[0]["generated_text"]]
            target = [[factual_filtered[0]["generated_text"]]]
            bleu_score = BLEUScore()(pred, target)
            tf_idf_score = tf_idf_compute(factual_filtered[0]["generated_text"], counterfactual_filtered[0]["generated_text"])
            embedding1 = model.encode(factual_filtered[0]["generated_text"], convert_to_tensor=True)
            embedding2 = model.encode(counterfactual_filtered[0]["generated_text"], convert_to_tensor=True)
            similarity = cosine_similarity(embedding1.unsqueeze(0), embedding2.unsqueeze(0))
            
            semantic_sim.append(similarity)
            bleu_scores.append(bleu_score)
            tf_idf_sim.append(tf_idf_score)
            print("video_id:", video_id, bleu_score, " tf-idf score:", tf_idf_score, " semantic sim:", similarity)
           # break
       # break
                
bleu_scores = np.array(bleu_scores)
tf_idf_scores = np.array(tf_idf_sim)
#semantic_sim_scores = np.array(semantic_sim.cpu())
semantic_sim_scores = [score.cpu() for score in semantic_sim]
semantic_sim_scores = np.array(semantic_sim_scores)
print("BlEU score:", np.mean(bleu_scores))
print("TF-IDF score:", np.mean(tf_idf_scores))
print("Semantic sim:", np.mean(semantic_sim_scores))