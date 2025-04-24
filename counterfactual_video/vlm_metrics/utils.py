import cv2
from typing import Union, Optional
import torch
from torchvision import transforms
from PIL import Image
from transformers import (
    LlavaNextProcessor, 
    LlavaNextForConditionalGeneration,
)

def extract_nth_frame(video_path, n=10):
    """
    Extract the nth frame from a video.
    
    Args:
        video_path (str): Path to the video file.
        n (int): Frame number to extract (1-based index).
        
    Returns:
        frame (numpy.ndarray): The nth frame as a BGR image, or None if extraction fails.
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return None
    
    # Skip frames until reaching the (n-1)th frame
    for _ in range(n - 1):
        ret = cap.grab()  # Fast skip (doesn't decode the frame)
        if not ret:
            print(f"Error: Video has fewer than {n} frames.")
            cap.release()
            return None
    
    # Read the nth frame
    ret, frame = cap.retrieve() if hasattr(cap, 'retrieve') else cap.read()
    
    cap.release()
    
    if not ret:
        print(f"Error: Failed to read frame {n}.")
        return None
    
    return frame  # Returns a BGR numpy array (H, W, 3)



def extract_first_frame(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
   # print(cap)
    
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return
    
    # Read the first frame
    ret, frame = cap.read()
    
    # Release the video capture object
    cap.release()
    return frame


#import textgrad as tg
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
    
    
if __name__ == "__main__":
    vlm = LlavaNext(
    model_name="llava-hf/llava-v1.6-mistral-7b-hf",  # or "Salesforce/blip2-opt-2.7b"
    device="cuda"  # Use "cpu" if no GPU
)