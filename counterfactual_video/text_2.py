from typing import Union, Tuple
import torch
from torchvision import transforms
from PIL import Image
import textgrad as tg
from vlm_wrappers.llava import LlavaNext
from textgrad.autograd import MultimodalLLMCall


class LlavaNextWrapper:
    def __init__(self, llava_model: LlavaNext):
        self.vlm = llava_model
        self.transform = transforms.ToPILImage()
        self.model_string = "llava-hf/llava-v1.6-mistral-7b-hf"
    
    def __call__(self, inputs: Union[Tuple, str], **kwargs) -> str:
        """
        Handle both:
        - (image, text) tuples
        - Text-only inputs
        """
        if isinstance(inputs, tuple):
            image, text = inputs
        else:
            # For text-only, create blank image
            image = Image.new('RGB', (224, 224), (255, 255, 255))
            text = inputs
        
        # Convert if inputs are TextGrad Variables
        image = image.value if hasattr(image, 'value') else image
        text = text.value if hasattr(text, 'value') else text
        
        # Convert tensor to PIL if needed
        if isinstance(image, torch.Tensor):
            if image.dim() == 3:
                image = self.transform(image.cpu())
            elif image.dim() == 4:
                image = self.transform(image[0].cpu())
        
        return self.vlm.generate(image, text, **kwargs)

# Initialize your VLM and wrapper
vlm = LlavaNext(model_name="llava-hf/llava-v1.6-mistral-7b-hf", device="cuda")
wrapped_vlm = LlavaNextWrapper(vlm)


#vlm = LlavaNext(
#    model_name="llava-hf/llava-v1.6-mistral-7b-hf",  # or "Salesforce/blip2-opt-2.7b"
#    device="cuda"  # Use "cpu" if no GPU
#)
tg.set_backward_engine(wrapped_vlm)

model = tg.BlackboxLLM(wrapped_vlm)

image_path = "/home/ubuntu/counterfactual-video-generation/counterfactual_video/outputs_v2/tokenflow-results_cfg_scale_4.5/explicit/interventions/age/-_B4fiuWwmo_0_1/He is old, he has beard, he is bald./vae_recon/00000.png"

with open(image_path, "rb") as f:
    image_data = f.read()
    
image_var = tg.Variable(image_data, 
                          role_description="image to answer a question about", requires_grad=False)

prompt_var = tg.Variable(
    "Describe this image in detail",
    role_description="description prompt", 
    requires_grad=False
)
#answer = model((image_var, prompt_var))

#print(image_var, prompt_var)
response = MultimodalLLMCall(wrapped_vlm)


##question_string = ("If it takes 1 hour to dry 25 shirts under the sun, "
#                   "how long will it take to dry 30 shirts under the sun? "
#                   "Reason step by step")

#question = tg.Variable(question_string,
#                       role_description="question to the LLM",
#                       requires_grad=False)

#answer = model(question)

