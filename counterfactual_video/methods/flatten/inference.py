def run_flatten_pipeline(
    prompt="She is young",
    neg_prompt="ugly, blurry, low res, unrealistic, unaesthetic",
    guidance_scale=4.5,
    video_path=None,
    sd_path="stabilityai/stable-diffusion-2-1-base",
    output_path="./outputs",
    video_length=24,
    old_qk=0,
    height=512,
    width=512,
    sample_steps=50,
    inject_step=40,
    seed=66,
    frame_rate=1,
    fps=20
):
    import os
    import torch
    import torchvision
    from einops import rearrange
    from diffusers import DDIMScheduler, AutoencoderKL, DDIMInverseScheduler
    from transformers import CLIPTextModel, CLIPTokenizer
    from huggingface_hub import snapshot_download
    import numpy as np 
    
    from .models.pipeline_flatten import FlattenPipeline
    from .models.util import save_videos_grid, read_video, sample_trajectories
    from .models.unet import UNet3DConditionModel

    os.makedirs(output_path, exist_ok=True)
    device = "cuda"

    # Normalize height and width
    height = (height // 32) * 32
    width = (width // 32) * 32
    sd_path = snapshot_download(sd_path)

    # Load models
    tokenizer = CLIPTokenizer.from_pretrained(sd_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(sd_path, subfolder="text_encoder").to(dtype=torch.float16)
    vae = AutoencoderKL.from_pretrained(sd_path, subfolder="vae").to(dtype=torch.float16)
    unet = UNet3DConditionModel.from_pretrained_2d(sd_path, subfolder="unet").to(dtype=torch.float16)
    scheduler = DDIMScheduler.from_pretrained(sd_path, subfolder="scheduler")
    inverse = DDIMInverseScheduler.from_pretrained(sd_path, subfolder="scheduler")

    pipe = FlattenPipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer,
        unet=unet, scheduler=scheduler, inverse_scheduler=inverse
    )
    pipe.enable_vae_slicing()
    pipe.enable_xformers_memory_efficient_attention()
    pipe.to(device)

    # Setup generator
    generator = torch.Generator(device=device).manual_seed(seed)

    # Read video
    video = read_video(video_path=video_path, video_length=video_length,
                       width=width, height=height, frame_rate=frame_rate)
    original_pixels = rearrange(video, "(b f) c h w -> b c f h w", b=1)

    # Save source video
    source_video_path = os.path.join(output_path, "source_video.mp4")
    save_videos_grid(original_pixels, source_video_path, rescale=True)

    # Convert frames to PIL
    t2i_transform = torchvision.transforms.ToPILImage()
    real_frames = [
        t2i_transform(((frame + 1) / 2 * 255).to(torch.uint8)) for frame in video
    ]
    # Compute trajectories
    trajectories = sample_trajectories(source_video_path, device)
    torch.cuda.empty_cache()
    trajectories = {k: v.to(device) for k, v in trajectories.items()}
    

    # Run pipeline
    sample = pipe(
        prompt, video_length=video_length, frames=real_frames,
        num_inference_steps=sample_steps, generator=generator,
        guidance_scale=guidance_scale, negative_prompt=neg_prompt,
        width=width, height=height, trajs=trajectories,
        output_dir="tmp/", inject_step=inject_step, old_qk=old_qk
    ).videos

    temp_video_name = f"{prompt[:10]}_{neg_prompt}_{guidance_scale}"
    output_file = os.path.join(output_path, f"{temp_video_name}.mp4")
    save_videos_grid(sample, output_file, fps=fps)

    print(f"Output saved to: {output_file}")
    orig_frames = [torch.from_numpy(np.array(frame)).permute(2,0,1) for frame in real_frames]
    orig_frames = torch.cat(orig_frames, dim=0).view(-1, 3, 512, 512)
    sample = sample.squeeze(0).permute(1,0,2,3)
    

    return sample , orig_frames