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
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer


logger = logging.getLogger(__name__)

class ClipTextAlignment(nn.Module):
    pretrained_model_name = 'openai/clip-vit-large-patch14'

    def __init__(
        self,
        device: torch.device = "cuda",
        pretrained_model_name: str = None,
    ):
        super().__init__()
        self.device = device
        pretrained_model_name = pretrained_model_name or self.pretrained_model_name
        logger.debug(f"Loding model {pretrained_model_name}")
        self.preprocessor = CLIPProcessor.from_pretrained(pretrained_model_name)
        self.model = CLIPModel.from_pretrained(pretrained_model_name)
        self.tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name)
        self.model.to(self.device)
        self.model.eval()
        logger.debug(f"Model {self.model.__class__.__name__} loaded")

    def range(self):
        return 0, 100

    def preprocess(self, video, text):
        #text = sample['edit_prompt']
       # video = sample['edit_video']

        text_inputs = self.preprocessor(
            text=text, padding=True, truncation=True, max_length=77, return_tensors='pt').to(self.device)

        image_inputs = []
        for frame in video:
            image_inputs.append(self.preprocessor(
                images=frame, padding=True, truncation=True, max_length=77, return_tensors='pt').to(self.device))

        return text_inputs, image_inputs

    @torch.no_grad()
    def evaluate(self, video, text) -> float:
        #text_inputs, image_inputs = args
        transform = transforms.Compose([
                            transforms.Resize(224)])

        text_tokens = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=77)

        text_input = text_tokens["input_ids"].to(self.device)
        text_embs = self.model.get_text_features(text_input)
        text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)

        scores = []
        for image_input in video:
            image_input = image_input.unsqueeze(0).to(self.device)
            image_input = transform(image_input)
            image_embs = self.model.get_image_features(image_input)
            image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)
            score = (self.model.logit_scale.exp() * (text_embs @ image_embs.T)).cpu().squeeze().item()
           # Compute cosine similarity (without logit scale)
            #score = torch.nn.functional.cosine_similarity(text_embs, image_embs).cpu().squeeze().item()
            scores.append(score)

        return sum(scores) / len(scores)