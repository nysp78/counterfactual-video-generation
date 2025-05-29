
import json
import argparse
import torch
import numpy as np
from tqdm import tqdm
import cv2
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from lpips_ import calculate_lpips, extract_lpips_embeddings
import os
from calculate_fvd import calculate_fvd

def load_video(video_path, num_frames=24, img_size=512):
    """
    Load a video from a file (MP4 or GIF), resize it, and normalize pixel values.
    """
    frames = []

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

    # Resize frames
    frames = [cv2.resize(frame, (img_size, img_size)) for frame in frames]

    # If fewer frames than required, duplicate last frame
    while len(frames) < num_frames:
        frames.append(frames[-1])

    # Convert to tensor array and normalize
    video_array = np.array(frames, dtype=np.float32)
    video_array = video_array / 255.0

    return video_array

def plot_tsne(embeddings, labels, title="LPIPS Embeddings", save_path="lpips_tsne.png"):
    """
    Plot t-SNE visualization of embeddings with different colors for each label
    """
    # Flatten the temporal dimension if needed
    if len(embeddings.shape) > 2:
        embeddings = embeddings.reshape(-1, embeddings.shape[-1])
    
    # Perform t-SNE dimensionality reduction
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Plot
    plt.figure(figsize=(12, 10))
    sns.scatterplot(
        x=embeddings_2d[:, 0], 
        y=embeddings_2d[:, 1],
        hue=labels,
        palette="viridis",
        alpha=0.7,
        s=100,
        edgecolor='none'
    )
    plt.title(title, fontsize=14)
    plt.xlabel("t-SNE Dimension 1", fontsize=12)
    plt.ylabel("t-SNE Dimension 2", fontsize=12)
    plt.legend(title="Video Type", title_fontsize=12, fontsize=11)
    plt.grid(alpha=0.2)
    plt.tight_layout()
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"t-SNE plot saved to {save_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outputs_path', type=str, default="../../../outputs_all/outputs_v9/tokenflow-results_cfg_scale_4.5")
    parser.add_argument('--method', choices=["tuneavideo", "tokenflow", "flatten"], default="tokenflow")
    parser.add_argument('--crf_config_path', type=str, default='../../../data/celebv_bench/counterfactual_explicit.json')
    parser.add_argument('--save_dir', type=str, default='./results')
    parser.add_argument('--num_frames', type=int, default=24)
    parser.add_argument('--img_size', type=int, default=512)
    opt = parser.parse_args()
    
    with open(opt.crf_config_path, "r") as f:
        crf_prompts = json.load(f)
        
    opt.intervention_type = "explicit"
    
    # Lists to store metrics and embeddings
    fvd_scores = []
    lpips_scores = []
    all_embeddings = []
    all_labels = []  # 0 for factual, 1 for counterfactual
    
    for video_id, prompts in tqdm(crf_prompts.items(), desc="Processing videos"):
        print(f"\nProcessing video: {video_id}")
       # if video_id != "pbkZ9Jp68n8_3_0":
       #     continue
        
        try:
            # Load factual video
            factual_path = f"../../../data/celebv_bench/videos/{video_id}.mp4"
            factual_video = load_video(video_path=factual_path, 
                                     num_frames=opt.num_frames, 
                                     img_size=opt.img_size)
            factual_video = torch.from_numpy(factual_video).permute(0,3,1,2).unsqueeze(0)
            
            # Extract factual embeddings
            factual_embeddings = extract_lpips_embeddings(factual_video, device="cuda")
            all_embeddings.append(factual_embeddings)
            all_labels.extend([0] * factual_embeddings.shape[0])  # 0 for factual
            
            for attr in prompts["counterfactual"].keys():
                crf_prompt = prompts["counterfactual"][attr]
                base_path = f"{opt.outputs_path}/{opt.intervention_type}/interventions/{attr}/{video_id}/{crf_prompt}"
                
                # Load counterfactual video based on method
                try:
                    if opt.method == "flatten":
                        import glob
                        pattern = f"{base_path}/*_ugly, blurry, low res, unrealistic, unaesthetic_7.5.mp4"
                        video_path = glob.glob(pattern)[0]
                        counterfactual_video = load_video(video_path=video_path,
                                                        num_frames=opt.num_frames,
                                                        img_size=opt.img_size)
                    elif opt.method == "tokenflow":
                        video_path = f"{base_path}/tokenflow_PnP_fps_20.mp4"
                        counterfactual_video = load_video(video_path=video_path,
                                                        num_frames=opt.num_frames,
                                                        img_size=opt.img_size)
                    else:  # tuneavideo
                        video_path = f"{base_path}/edited_fps20.gif"
                        counterfactual_video = load_video(video_path=video_path,
                                                        num_frames=opt.num_frames,
                                                        img_size=opt.img_size)
                    
                    counterfactual_video = torch.from_numpy(counterfactual_video).permute(0,3,1,2).unsqueeze(0)
                    
                    # Extract counterfactual embeddings
                    counterfactual_embeddings = extract_lpips_embeddings(counterfactual_video, device="cuda")
                    all_embeddings.append(counterfactual_embeddings)
                    all_labels.extend([1] * counterfactual_embeddings.shape[0])  # 1 for counterfactual
                    
                    # Calculate metrics
                    fvd = calculate_fvd(factual_video, counterfactual_video, device="cuda", method='styleganv', only_final=True)
                    lpips = calculate_lpips(factual_video, counterfactual_video, device="cuda", only_final=True)
                    
                    fvd_scores.append(fvd["value"][0])
                    lpips_scores.append(lpips["value"][0])
                    
                    print(f"Attribute: {attr} | FVD: {fvd['value'][0]:.2f} | LPIPS: {lpips['value'][0]:.3f}")
                    
                except Exception as e:
                    print(f"Error processing counterfactual for {video_id} {attr}: {str(e)}")
                    continue
                    
        except Exception as e:
            print(f"Error processing factual video {video_id}: {str(e)}")
            continue
    
    # Combine all embeddings
    print(len(all_embeddings))
    all_embeddings = np.concatenate(all_embeddings)
    
    # Create labels for the plot
    label_names = ["Factual", "Counterfactual"]
    plot_labels = [label_names[l] for l in all_labels]
    
    # Plot t-SNE
    tsne_save_path = os.path.join(opt.save_dir, "lpips_tsne.png")
    plot_tsne(all_embeddings, plot_labels, 
              title="t-SNE Visualization of LPIPS Embeddings\n(Factual vs Counterfactual Frames)",
              save_path=tsne_save_path)
    
    # Print summary metrics
    print("\n=== Summary Metrics ===")
    print(f"Mean FVD: {np.mean(fvd_scores):.2f} ± {np.std(fvd_scores):.2f}")
    print(f"Mean LPIPS: {np.mean(lpips_scores):.3f} ± {np.std(lpips_scores):.3f}")
    
    # Save metrics to file
    metrics = {
        'fvd': {
            'mean': float(np.mean(fvd_scores)),
            'std': float(np.std(fvd_scores)),
            'all': [float(x) for x in fvd_scores]
        },
        'lpips': {
            'mean': float(np.mean(lpips_scores)),
            'std': float(np.std(lpips_scores)),
            'all': [float(x) for x in lpips_scores]
        }
    }
    
    metrics_path = os.path.join(opt.save_dir, "metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

if __name__ == '__main__':
    main()