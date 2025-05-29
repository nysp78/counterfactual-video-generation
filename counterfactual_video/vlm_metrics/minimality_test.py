import json
import argparse
import torch
from tqdm import tqdm
import numpy as np
import os
import textgrad as tg
import io
from sentence_transformers import SentenceTransformer
from PIL import Image
from textgrad.autograd import MultimodalLLMCall
from torch.nn.functional import cosine_similarity
from torchvision.transforms import Resize, ToPILImage, Compose
from utils import extract_nth_frame
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import glob
import pandas as pd
import seaborn as sns


# Set your OpenAI API key securely
os.environ["OPENAI_API_KEY"] = "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"

def pil_to_bytes(image_pil):
    png_buffer = io.BytesIO()
    image_pil.save(png_buffer, format='PNG')
    return png_buffer.getvalue()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt-4-turbo")
    parser.add_argument('--outputs_path', type=str, default="../outputs_all/outputs_v9/tokenflow-results_cfg_scale_4.5")
    parser.add_argument('--method', choices=["tuneavideo", "tokenflow", "flatten"], default="tokenflow")
    parser.add_argument('--crf_config_path', type=str, default='../data/celebv_bench/counterfactual_explicit.json')
    opt = parser.parse_args()

    with open(opt.crf_config_path, "r") as f:
        crf_prompts = json.load(f)

    opt.intervention_type = "explicit"

    model_path = opt.model
    model = SentenceTransformer("all-MiniLM-L6-v2").to("cuda").eval()

    semantic_sim = []
    descriptions = {}
    transform = Compose([ToPILImage(), Resize((512, 512))])

    embeddings = []
    labels = []

    for video_id, prompts in tqdm(crf_prompts.items()):
        print("Evaluate video:", video_id)
       # if video_id != "pbkZ9Jp68n8_3_0":
       #     continue
        descriptions[video_id] = {}

        # Factual frame
        factual_frame = extract_nth_frame(video_path=f"../data/celebv_bench/videos/{video_id}.mp4", n=6)
        factual_frame = transform(factual_frame)
        factual_frame_data = pil_to_bytes(factual_frame)

        vlm_prompt = '''Describe this image in detail.
Remove any references to age, gender (man, woman, he, she), beard, hair (including hairstyle, color, style, and facial hair), and baldness from the description.
Return only the filtered version of the text, without commentary or formatting.'''

        image_variable = tg.Variable(factual_frame_data, role_description="image to answer a question about", requires_grad=False)
        question_variable = tg.Variable(vlm_prompt, role_description="instruction to the VLM", requires_grad=False)
        response = MultimodalLLMCall(model_path)([image_variable, question_variable])
        print("FACTUAL DESCR:", response.value)

        descriptions[video_id]["factual"] = response.value
        embedding1 = model.encode(response.value, convert_to_tensor=True)

        embeddings.append(embedding1.cpu().numpy())
        labels.append(f"{video_id}_factual")

        descriptions[video_id]["counterfactual"] = {}

        for attr in prompts["counterfactual"].keys():
            if attr!="bald":
                continue
            crf_prompt = prompts["counterfactual"][attr]
            base_path = f"{opt.outputs_path}/{opt.intervention_type}/interventions/{attr}/{video_id}/{crf_prompt}"

            if opt.method == "flatten":
                pattern = f"{base_path}/*_ugly, blurry, low res, unrealistic, unaesthetic_7.5.mp4"
                video_path = glob.glob(pattern)[0]
                counterfactual_frame = extract_nth_frame(video_path, n=6)

            elif opt.method == "tokenflow":
                counterfactual_frame = extract_nth_frame(video_path=base_path + "/tokenflow_PnP_fps_20.mp4", n=6)

            elif opt.method == "tuneavideo":
                counterfactual_frame = extract_nth_frame(video_path=base_path + "/edited_fps20.gif", n=6)

            counterfactual_frame = transform(counterfactual_frame)
            counterfactual_frame_data = pil_to_bytes(counterfactual_frame)

            crf_image_variable = tg.Variable(counterfactual_frame_data, role_description="image to answer a question about", requires_grad=False)
            response_crf = MultimodalLLMCall(model_path)([crf_image_variable, question_variable])
            print("COUNTERFACTUAL_DESCR:", response_crf.value)

            descriptions[video_id]["counterfactual"][attr] = response_crf.value
            embedding2 = model.encode(response_crf.value, convert_to_tensor=True)

            similarity = cosine_similarity(embedding1.unsqueeze(0), embedding2.unsqueeze(0))
            print("Semantic similarity:", similarity.item())
            semantic_sim.append(similarity)

            embeddings.append(embedding2.cpu().numpy())
            print(embedding1.shape, embedding2.shape)
            labels.append(f"{video_id}_{attr}")

    # Mean similarity
    semantic_sim_scores = [score.cpu() for score in semantic_sim]
    semantic_sim_scores = np.array(semantic_sim_scores)
    print("Mean semantic similarity:", np.mean(semantic_sim_scores))

    print(len(embeddings))
    print("Generating t-SNE plot...")
    
    # Prepare data
    embeddings_np = np.array(embeddings)
    
    # Determine perplexity
    n_samples = len(embeddings_np)
    perplexity = 1 #min(30, max(5, n_samples // 2))  # More robust perplexity calculation
    
    # Run t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, 
                n_iter=1000, random_state=42, init='pca')
    embeddings_2d = tsne.fit_transform(embeddings_np)
    
    # Create DataFrame for Seaborn
    df = pd.DataFrame({
        'Dimension 1': embeddings_2d[:, 0],
        'Dimension 2': embeddings_2d[:, 1],
        'Type': ['Factual' if 'factual' in label else 'Counterfactual' for label in labels],
        'Video': [label.split('_')[0] for label in labels]  # Extract base video ID
    })
    
    # Set style
    sns.set(style="whitegrid", font_scale=1.2)
    plt.figure(figsize=(12, 10))

# Create the scatter plot
scatter = sns.scatterplot(
    data=df,
    x='Dimension 1',
    y='Dimension 2',
    hue='Type',
    palette= "viridis",
    s=100,  # Size of points
    alpha=0.7,
    edgecolor='none',
)

# Create custom legend with bullet points
#legend_elements = [
#    plt.Line2D([0], [0], marker='o', color='w', label='Factual',
#              markerfacecolor='gray', markersize=10),
#    plt.Line2D([0], [0], marker='o', color='w', label='Counterfactual',
#              markerfacecolor='green', markersize=10)
#]

# Add legend to plot
#plt.legend(handles=legend_elements, title='Video Type',
#           bbox_to_anchor=(1.05, 1), loc='upper left')

# Rest of your plot formatting remains the same
plt.title("VLM minimality Text Embeddings", pad=20)
plt.xlabel("t-SNE Dimension 1", labelpad=10)
plt.ylabel("t-SNE Dimension 2", labelpad=10)
plt.tight_layout()

# Save plot
plt.savefig("tsne_plot_bullets.png", dpi=300, bbox_inches='tight')
plt.close()