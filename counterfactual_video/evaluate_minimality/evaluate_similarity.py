import json
import yaml
import argparse
import torch
from tqdm import tqdm
import cv2
import numpy as np
import os
import clip
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
    parser.add_argument('--description_path', type=str, default="filtered_descriptions/filtered_descr_tuneavideo_explicit.json")
   
    opt = parser.parse_args()
    with open(opt.description_path, "r") as f:
        filtered_descriptions = json.load(f)
        
    model = SentenceTransformer("all-MiniLM-L6-v2").to("cuda").eval()
    
    
    bleu_scores = []
    tf_idf_sim = []
    semantic_sim = []
    
    for video_id , descr in tqdm(filtered_descriptions.items()):
        print("Evaluate video:", video_id)
        factual_description = descr["factual"]
       
                                
        embedding1 = model.encode(factual_description, convert_to_tensor=True)
        for attr in descr["counterfactual"].keys():
           # print(questions)
            crf_description = descr["counterfactual"][attr]
        
      
            pred = [crf_description]
            target = [[factual_description]]
            bleu_score = BLEUScore()(pred, target)
            tf_idf_score = tf_idf_compute(factual_description, crf_description)
          #  embedding1 = model.encode(factual_filtered[0]["generated_text"], convert_to_tensor=True)
            embedding2 = model.encode(crf_description, convert_to_tensor=True)
         #   tokens = clip.tokenize([factual_filtered[0]["generated_text"], counterfactual_filtered[0]["generated_text"]]).to("cuda")
         #   with torch.no_grad():
         #       text_embeddings = model.encode_text(tokens)  # Get text embeddings

            # Normalize clip embeddings
         #   text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
          #  clip_similarity = torch.cosine_similarity(text_embeddings[0].unsqueeze(0), text_embeddings[1].unsqueeze(0))
            
            #sentence transformer similarity
            similarity = cosine_similarity(embedding1.unsqueeze(0), embedding2.unsqueeze(0))
            
            semantic_sim.append(similarity)
          #  clip_sim.append(clip_similarity)
            bleu_scores.append(bleu_score)
            tf_idf_sim.append(tf_idf_score)
            print("video_id:", video_id, bleu_score, " tf-idf score:", 
                  tf_idf_score, " semantic sim:", similarity)
          #  break
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
    