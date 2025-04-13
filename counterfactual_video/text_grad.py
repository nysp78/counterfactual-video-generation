import textgrad as tg
from vlm_wrappers.llava import LlavaNext  # Replace with your actual module name
from typing import Tuple

class VLMWrapper:
    def __init__(self, vlm):
        self.vlm = vlm
    
    def __call__(self, image_var: tg.Variable, text_var: tg.Variable) -> tg.Variable:
        """Handle both image and text inputs properly"""
        # Extract values from TextGrad variables
        image = image_var.value
        text = text_var.value
        
        # Call your VLM
        description = self.vlm.generate(image, text)
        
        # Return new TextGrad variable
        return tg.Variable(
            description,
            role_description="VLM generated description",
            requires_grad=True
        )

# Usage:
vlm = LlavaNext(model_name="llava-hf/llava-v1.6-mistral-7b-hf")
wrapped_vlm = VLMWrapper(vlm)
image_path = "/home/ubuntu/counterfactual-video-generation/counterfactual_video/outputs_v2/tokenflow-results_cfg_scale_4.5/explicit/interventions/age/-_B4fiuWwmo_0_1/He is old, he has beard, he is bald./vae_recon/00000.png"

image_var = tg.Variable(
    image_path,
    role_description="input image",
    requires_grad=False
)

prompt_var = tg.Variable(
    "Describe this image in detail",
    role_description="description prompt", 
    requires_grad=False
)

# Get initial description
description = wrapped_vlm(image_var, prompt_var)
print(description)

optimizer = tg.TGD(parameters=[description])

loss_fn = tg.TextLoss("""Critique the description:
1. +1 for each correctly identified object
2. +0.5 for accurate attributes
3. -1 for hallucinations""")

    
# Initialize TextGrad and VLM
#textgrad = tg.TextGrad()
#lava = LlavaNext(model_name="llava-hf/llava-v1.6-mistral-7b-hf", device="cuda")
#vlm = VLMWrapper(lava)

#tg_vlm = tg.BlackboxLLM(vlm)  # Wrap generate method
#image_path = "/home/ubuntu/counterfactual-video-generation/counterfactual_video/outputs_v2/tokenflow-results_cfg_scale_4.5/explicit/interventions/age/-_B4fiuWwmo_0_1/He is old, he has beard, he is bald./vae_recon/00000.png"
#initial_prompt = "Describe this image in extreme detail"
#image_var = tg.Variable(image_path, role_description="input image", requires_grad=False)
#prompt_var = tg.Variable(initial_prompt, role_description="question", requires_grad=False)
# Initial description
#description = tg_vlm((image_var, prompt_var))
#print(f"Initial description:\n{description.value}\n")
#print(tg_vlm)
'''
# Define inputs
image_path = "/home/ubuntu/counterfactual-video-generation/counterfactual_video/outputs_v2/tokenflow-results_cfg_scale_4.5/explicit/interventions/age/-_B4fiuWwmo_0_1/He is old, he has beard, he is bald./vae_recon/00000.png"
initial_question = "Describe this image in detail."

# TextGrad variables
image_var = tg.Variable(image_path, role_description="input image", requires_grad=False)
question_var = tg.Variable(initial_question, role_description="question about the image", requires_grad=False)
answer_var = tg_vlm(image_var, question_var)  # Initial answer

# Define optimization components
optimizer = tg.TGD(parameters=[answer_var])
loss_instruction = """Critique the answer:
1. Award 2 points for mentioning all key objects
2. Deduct 1 point for missing spatial relationships
3. Award 1 point for style consistency"""
loss_fn = tg.TextLoss(loss_instruction)

# Optimization loop
for iteration in range(3):
    print(f"\n=== Iteration {iteration+1} ===")
    
    # Forward pass
    loss = loss_fn(answer_var)
    print(f"Initial Answer:\n{answer_var.value}\n\nFeedback:\n{loss.value}\n")
    
    # Backward pass (compute textual gradients)
    loss.backward()
    
    # Update answer
    optimizer.step()
    
    print(f"Optimized Answer:\n{answer_var.value}\n")
 '''