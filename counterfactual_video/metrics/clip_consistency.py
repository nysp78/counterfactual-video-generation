'''
https://github.com/openai/CLIP
https://github.com/mlfoundations/open_clip
https://huggingface.co/docs/transformers/model_doc/clip#clip

'''
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import transforms
from transformers import CLIPImageProcessor, CLIPVisionModel

logger = logging.getLogger(__name__)

class ClipConsistency(nn.Module):
    pretrained_model_name = 'openai/clip-vit-large-patch14'

    def __init__(
        self,
        device: torch.device = "cuda",
        pretrained_model_name: str = None,
    ):
       # super().__init__(index_file, edit_video_dir, None, edit_prompt, device)
        super().__init__()
        pretrained_model_name = pretrained_model_name or self.pretrained_model_name
        logger.debug(f"Loding model {pretrained_model_name}")
        self.device = device
        self.preprocessor = CLIPImageProcessor.from_pretrained(pretrained_model_name)
        self.model = CLIPVisionModel.from_pretrained(pretrained_model_name)
        self.model.to(self.device)
        self.model.eval()
        logger.debug(f"Model {self.model.__class__.__name__} loaded")

    def range(self):
        return 0, 1

    def preprocess(self, video):
      #  video = sample['edit_video']
        frames = []
        for i, frame in enumerate(video):
            frames.append(self.preprocessor(frame, return_tensors='pt').pixel_values)
        return frames

    @torch.no_grad()
    def evaluate(self, frames) -> float:
        
        transform = transforms.Compose([
                            transforms.Resize(224)]) #resize to clip resolution
        similarity = []
        former_feature = None
        for i, frame in enumerate(frames):
            frame = frame.to(self.device)
           # print(frame.shape)
            frame = frame.unsqueeze(0)
            frame = transform(frame)
            feature: torch.Tensor = self.model(pixel_values=frame).pooler_output
            feature = feature / torch.norm(feature, dim=-1, keepdim=True)

            if i > 0:
                sim = max(0, (feature @ former_feature.T).cpu().squeeze().item())
                similarity.append(sim)
            former_feature = feature
        return sum(similarity) / len(similarity)