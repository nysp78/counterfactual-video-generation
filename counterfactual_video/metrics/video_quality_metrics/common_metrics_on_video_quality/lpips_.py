import numpy as np
import torch
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
import lpips

spatial = False  # Set to False to get global embeddings instead of spatial maps

# Linearly calibrated models (LPIPS)
loss_fn = lpips.LPIPS(net='alex', spatial=spatial)

def trans(x):
    """Transform input to [-1,1] range and handle grayscale"""
    if x.shape[-3] == 1:  # grayscale
        x = x.repeat(1, 1, 3, 1, 1)
    x = x * 2 - 1  # [0,1] -> [-1,1]
    return x

def extract_lpips_embeddings(videos, device, frame_idx=None):
    """Extract embeddings from LPIPS network for a single frame per video.
    
    Args:
        videos: Input tensor of shape (B, T, C, H, W)
        device: Target device (e.g., 'cuda')
        frame_idx: Frame index to use (default: middle frame)
    Returns:
        ndarray: Embeddings of shape (B, D) where D is embedding dimension
    """
    print("Extracting LPIPS embeddings for single frame...")
    videos = trans(videos)
    embeddings = []
    
    loss_fn.to(device)
    
    for video_num in tqdm(range(videos.shape[0]), desc="Extracting embeddings"):
        video = videos[video_num]
        
        # Determine frame index (default to middle frame)
        if frame_idx is None:
            use_idx = len(video) // 2
        else:
            use_idx = frame_idx if frame_idx >= 0 else len(video) + frame_idx
        
        # Validate frame index
        if use_idx < 0 or use_idx >= len(video):
            raise ValueError(f"Frame index {frame_idx} is out of bounds for video length {len(video)}")
        
        img = video[use_idx].unsqueeze(0).to(device)
        
        with torch.no_grad():
            features = loss_fn.net(img)
            
            # Handle different output formats
            if isinstance(features, (tuple, list)):
                features = features[-1]  # Use last feature layer
            
            # Global average pooling
            embedding = torch.mean(features, dim=[2, 3]).cpu().numpy()
            embeddings.append(embedding)
    
    return np.concatenate(embeddings) if embeddings else np.array([])

def calculate_lpips(videos1, videos2, device, frame_idx=None, **kwargs):
    """Calculate LPIPS distance between videos for a specific frame.
    
    Args:
        videos1: Reference videos (B, T, C, H, W)
        videos2: Comparison videos (B, T, C, H, W)
        device: Target device
        frame_idx: Frame index to use (default: middle frame)
        **kwargs: Passed to original calculate_lpips
    """
    if frame_idx is not None:
        # Select specific frame while maintaining dimensions
        videos1 = videos1[:, frame_idx:frame_idx+1]
        videos2 = videos2[:, frame_idx:frame_idx+1]
    
    return _calculate_lpips_full(videos1, videos2, device, **kwargs)

def _calculate_lpips_full(videos1, videos2, device, only_final=False, return_embeddings=False):
    """Original full-frame LPIPS calculation"""
    print("Calculating LPIPS...")
    assert videos1.shape == videos2.shape

    videos1 = trans(videos1)
    videos2 = trans(videos2)

    lpips_results = []
    embeddings1 = []
    embeddings2 = []

    loss_fn.to(device)

    for video_num in tqdm(range(videos1.shape[0]), desc="Processing videos"):
        video1 = videos1[video_num]
        video2 = videos2[video_num]

        video_lpips = []
        video_emb1 = []
        video_emb2 = []
        
        for clip_timestamp in range(len(video1)):
            img1 = video1[clip_timestamp].unsqueeze(0).to(device)
            img2 = video2[clip_timestamp].unsqueeze(0).to(device)
            
            # Calculate distance
            dist = loss_fn.forward(img1, img2).mean()
            video_lpips.append(dist.detach().cpu().item())
            
            if return_embeddings:
                with torch.no_grad():
                    feat1 = loss_fn.net(img1)
                    feat2 = loss_fn.net(img2)
                    
                    if isinstance(feat1, (tuple, list)):
                        feat1 = feat1[-1]
                        feat2 = feat2[-1]
                    
                    emb1 = torch.mean(feat1, dim=[2,3]).cpu().numpy()
                    emb2 = torch.mean(feat2, dim=[2,3]).cpu().numpy()
                    
                video_emb1.append(emb1)
                video_emb2.append(emb2)
                
        lpips_results.append(video_lpips)
        if return_embeddings:
            embeddings1.append(np.concatenate(video_emb1))
            embeddings2.append(np.concatenate(video_emb2))

    lpips_results = np.array(lpips_results)
    
    # Calculate statistics
    if only_final:
        lpips_mean = [np.mean(lpips_results)]
        lpips_std = [np.std(lpips_results)]
    else:
        lpips_mean = [np.mean(lpips_results[:,t]) for t in range(lpips_results.shape[1])]
        lpips_std = [np.std(lpips_results[:,t]) for t in range(lpips_results.shape[1])]

    result = {
        "value": lpips_mean,
        "value_std": lpips_std,
    }
    
    if return_embeddings:
        result.update({
            "embeddings1": np.array(embeddings1) if embeddings1 else np.array([]),
            "embeddings2": np.array(embeddings2) if embeddings2 else np.array([])
        })

    return result

# Test code
def main():
    """Test function"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create test videos
    B, T, C, H, W = 2, 10, 3, 256, 256
    videos1 = torch.rand(B, T, C, H, W)
    videos2 = torch.rand(B, T, C, H, W)
    
    # Test single-frame embedding extraction
    print("\nTesting single-frame embedding extraction:")
    for frame in [0, -1, None]:  # First, last, and middle frames
        embeddings = extract_lpips_embeddings(videos1, device, frame_idx=frame)
        print(f"Frame {frame if frame is not None else 'middle'}: embeddings shape {embeddings.shape}")
    
    # Test single-frame LPIPS calculation
    print("\nTesting single-frame LPIPS:")
    result = calculate_lpips(videos1, videos2, device, frame_idx=0)  # First frame
    print("First frame LPIPS:", result["value"])
    
    # Test full-frame calculation
    print("\nTesting full video LPIPS:")
    result = calculate_lpips(videos1, videos2, device)
    print("LPIPS results (all frames):", result["value"])

if __name__ == "__main__":
    main()