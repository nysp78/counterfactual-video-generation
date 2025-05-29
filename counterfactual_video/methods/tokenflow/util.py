from pathlib import Path
from PIL import Image
import torch
import yaml
import math
import imageio
import torchvision.transforms as T
from torchvision.io import read_video,write_video
import os
import random
import numpy as np
from einops import rearrange
import torchvision
from torchvision.io import write_video
# from kornia.filters import joint_bilateral_blur
from kornia.geometry.transform import remap
from kornia.utils.grid import create_meshgrid
import cv2
from PIL import Image, ImageDraw, ImageFont
import os
import numpy as np
import torch
import torchvision
from einops import rearrange
import imageio
from textwrap import wrap

def save_video_frames(video_path, img_size=(512,512)):
    video, _, _ = read_video(video_path, output_format="TCHW")
    # rotate video -90 degree if video is .mov format. this is a weird bug in torchvision
    if video_path.endswith('.mov'):
        video = T.functional.rotate(video, -90)
    video_name = Path(video_path).stem
    os.makedirs(f'data/{video_name}', exist_ok=True)
    for i in range(len(video)):
        ind = str(i).zfill(5)
        image = T.ToPILImage()(video[i])
        image_resized = image.resize((img_size),  resample=Image.Resampling.LANCZOS)
        image_resized.save(f'data/{video_name}/{ind}.png')

def add_dict_to_yaml_file(file_path, key, value):
    data = {}

    # If the file already exists, load its contents into the data dictionary
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)

    # Add or update the key-value pair
    data[key] = value

    # Save the data back to the YAML file
    with open(file_path, 'w') as file:
        yaml.dump(data, file)
        
def isinstance_str(x: object, cls_name: str):
    """
    Checks whether x has any class *named* cls_name in its ancestry.
    Doesn't require access to the class's implementation.
    
    Useful for patching!
    """

    for _cls in x.__class__.__mro__:
        if _cls.__name__ == cls_name:
            return True
    
    return False


def batch_cosine_sim(x, y):
    if type(x) is list:
        x = torch.cat(x, dim=0)
    if type(y) is list:
        y = torch.cat(y, dim=0)
    x = x / x.norm(dim=-1, keepdim=True)
    y = y / y.norm(dim=-1, keepdim=True)
    similarity = x @ y.T
    return similarity


def load_imgs(data_path, n_frames, device='cuda', pil=False):
    imgs = []
    pils = []
    for i in range(n_frames):
        img_path = os.path.join(data_path, "%05d.jpg" % i)
        if not os.path.exists(img_path):
            img_path = os.path.join(data_path, "%05d.png" % i)
        img_pil = Image.open(img_path)
        pils.append(img_pil)
        img = T.ToTensor()(img_pil).unsqueeze(0)
        imgs.append(img)
    if pil:
        return torch.cat(imgs).to(device), pils
    return torch.cat(imgs).to(device)


def save_videos_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=5, fps=20):
    videos = rearrange(videos, "b c t h w -> t b c h w")
    outputs = []
    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = (x * 255).numpy().astype(np.uint8)
        outputs.append(x)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimsave(path, outputs, fps=fps, loop=0)


def save_videos_grid__(videos: torch.Tensor, path: str, titles=None, rescale=False, n_rows=5, fps=20):
    import os
    import numpy as np
    import torch
    import torchvision
    from PIL import Image, ImageDraw, ImageFont
    from einops import rearrange
    import imageio
    from textwrap import wrap

    videos = rearrange(videos, "b c t h w -> t b c h w")  # Time-major
    outputs = []

    os.makedirs(os.path.dirname(path), exist_ok=True)

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 24)
    except IOError:
        font = ImageFont.load_default()

    # Wrap titles and compute max text height
    max_text_height = 0
    wrapped_titles = []
    if titles:
        for title in titles:
            lines = []
            for line in title.split("\n"):
                lines.extend(wrap(line, width=40))
            wrapped_titles.append(lines)
            line_heights = [font.getbbox(line)[3] - font.getbbox(line)[1] for line in lines]
            total = sum(line_heights) + (len(lines) - 1) * 5
            max_text_height = max(max_text_height, total)

    for frame in videos:
        frame_list = []
        max_width, max_height = 0, 0

        for idx, vid in enumerate(frame):
            img = vid.permute(1, 2, 0).cpu().numpy()
            if rescale:
                img = (img + 1.0) / 2.0
            img = (img * 255).astype(np.uint8)
            img_pil = Image.fromarray(img)

            # Add title above the video
            if titles and idx < len(titles):
                lines = wrapped_titles[idx]
                spacing = 5
                new_height = img_pil.height + max_text_height + 30
                new_img = Image.new("RGB", (img_pil.width, new_height), (255, 255, 255))
                new_img.paste(img_pil, (0, max_text_height + 20))

                draw = ImageDraw.Draw(new_img)
                text_y = 10
                for line_idx, line in enumerate(lines):
                    bbox = font.getbbox(line)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]
                    text_x = (img_pil.width - text_width) // 2
                    color = (0, 0, 0) if line_idx == 0 else (255, 0, 0)
                    draw.text((text_x, text_y), line, font=font, fill=color)
                    text_y += text_height + spacing

                img_pil = new_img

            max_width = max(max_width, img_pil.width)
            max_height = max(max_height, img_pil.height)
            frame_list.append(img_pil)

        # Pad each frame with white, aligned to top
        uniform_frames = []
        for img_pil in frame_list:
            padded = Image.new("RGB", (max_width, max_height), (255, 255, 255))
            offset_x = (max_width - img_pil.width) // 2
            padded.paste(img_pil, (offset_x, 0))  # top-aligned
            uniform_frames.append(np.array(padded))

        # Make grid
        grid = torchvision.utils.make_grid(
            torch.tensor(np.stack(uniform_frames)).permute(0, 3, 1, 2),
            nrow=n_rows
        )
        outputs.append(grid.permute(1, 2, 0).cpu().numpy())

    # Save as GIF
    imageio.mimsave(path, outputs, fps=fps, loop=0)







def save_video(raw_frames, save_path, fps=10):
    video_codec = "libx264"
    video_options = {
        "crf": "18",  # Constant Rate Factor (lower value = higher quality, 18 is a good balance)
        "preset": "slow",  # Encoding preset (e.g., ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow)
    }

    frames = (raw_frames * 255).to(torch.uint8).cpu().permute(0, 2, 3, 1)
    write_video(save_path, frames, fps=fps, video_codec=video_codec, options=video_options)


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


