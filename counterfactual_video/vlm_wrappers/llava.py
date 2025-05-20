from typing import Union, Optional
import torch
from torchvision import transforms
from PIL import Image
from transformers import (
    LlavaNextProcessor, 
    LlavaNextForConditionalGeneration,
)

class LlavaNext:
    def __init__(
        self,
        model_name: str = "llava-hf/llava-v1.6-7b-hf",
        device: str = "cuda",
    ):
        self.model_name = model_name
        self.device = device
        self.model, self.processor = self._load_model()

    def _load_model(self):
        """Same as before (loads model based on name)."""
        if "llava" in self.model_name.lower():
            processor = LlavaNextProcessor.from_pretrained(self.model_name)
            model = LlavaNextForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
            )
       
        return model, processor

    def generate(
        self,
        image: Union[str, Image.Image, torch.Tensor],  # Accepts path, PIL Image, or tensor
        text_prompt: str,
        max_new_tokens: int = 512,
        do_sample = False,
        **kwargs
    ) -> str:
        """Process image tensor/PIL/path and generate response."""
        # Convert input to PIL Image if it's a tensor
        if isinstance(image, torch.Tensor):
            if image.dim() == 3:  # CHW format (C, H, W)
                image = transforms.ToPILImage()(image.cpu())
            elif image.dim() == 4:  # Batch of images (B, C, H, W)
                image = transforms.ToPILImage()(image[0].cpu())
            else:
                raise ValueError("Image tensor must be CHW or BCHW format!")

        # Load image if path is provided
        elif isinstance(image, str):
            image = Image.open(image)

        # Process inputs (model-specific)
        #if "llava" in self.model_name.lower():
         #   inputs = self.processor(text_prompt, image, return_tensors="pt").to(self.device)
        #elif "blip" in self.model_name.lower():
        #    inputs = self.processor(image, text_prompt, return_tensors="pt").to(self.device)
        conversation = [
        {   
        "role": "user",
        "content": [
          {"type": "text", "text": text_prompt},
          {"type": "image"},
            ],
        },
        ]
        
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self.processor(images=image, text=prompt, return_tensors='pt').to(0, torch.float16)
       # output = self.model.generate(**inputs, max_new_tokens=max_length, do_sample=True, temperature=0.6, top_k=50,           # Control randomness (lower is more deterministic)top_k=50,                 
       # top_p=0.9)


        # Generate and decode
        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample = do_sample, **kwargs)
        return self.processor.decode(outputs[0], skip_special_tokens=True)